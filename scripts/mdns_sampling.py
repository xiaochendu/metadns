#!/usr/bin/env python
"""
Standalone script for running MDNS sampling and evaluation.
Replicates functionality of mam_arm_sampling.py for MDNS models.
"""

import argparse
import logging
import os
import pickle as pkl
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to allow imports from MDNS
sys.path.append(str(Path(__file__).resolve().parent.parent))

from model import ExponentialMovingAverage, get_rope_vit_model
from utils import ess
from utils_ising import ising2d_ham
from utils_train import _compute_log_stats, rnd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sampling for MDNS model over temperatures and fields."
    )
    
    # Model arguments
    parser.add_argument("--L", type=int, default=16, help="Lattice size (Linear dimension)")
    parser.add_argument("--embed-dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=4, help="Transformer depth")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--vocab-size", type=int, default=3, help="Vocab size (default 3 for MDNS)")
    
    # Checkpoint
    parser.add_argument(
        "--ckpt", 
        type=str, 
        required=True, 
        help="Path to the model checkpoint."
    )
    
    # Sampling configuration
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[2.269],
        help="Temperatures to evaluate.",
    )
    parser.add_argument(
        "--fields",
        type=float,
        nargs="+",
        default=[0.0],
        help="External fields (chemical potentials) to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for sampling.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4096,
        help="Total number of samples per condition.",
    )
    parser.add_argument(
        "--J",
        type=float,
        default=1.0,
        help="Interaction strength J.",
    )
    
    # Output
    parser.add_argument(
        "--output-folder",
        type=str,
        default="outputs/mdns",
        help="Directory to save results.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="mdns_results.pkl",
        help="Output filename.",
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    
    return parser.parse_args()


def load_model(args, device):
    logger.info(f"Loading model from {args.ckpt}")
    model = get_rope_vit_model(
        L=args.L,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        device=device
    )
    
    # Initialize EMA wrapper often used in MDNS
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
    
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    # Load state dicts
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Check if checkpoint has conditioning weights (for safe loading)
    ckpt_has_weights = "beta_embedder.mlp.0.weight" in state_dict
    
    if not ckpt_has_weights:
        logger.warning("Checkpoint looks unconditional (missing beta_embedder). Loading with strict=False.")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Verify only expected keys are missing
        expected_missing = {"beta_embedder", "h_embedder", "thermo_proj"}
        # Filter expected missing keys to reduce noise
        real_missing = [k for k in missing if not any(e in k for e in expected_missing)]
        if real_missing:
            logger.info(f"Missing keys: {len(missing)}")
            for k in real_missing:
                logger.warning(f"Unexpected missing key: {k}")
    else:
        model.load_state_dict(state_dict)

    if 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        logger.info("Loaded EMA weights into model")
    else:
        logger.warning("No EMA state dict found in checkpoint, using standard weights")
        
    model.eval()
    
    # Determine if we SHOULD condition based on user arguments (Sweep -> Condition)
    # "check if either the number of input --temps and --fields is greater than 1"
    force_conditioning = len(args.temps) > 1 or len(args.fields) > 1
    
    if force_conditioning and not ckpt_has_weights:
        logger.warning("Requesting a sweep (conditioning needed) but checkpoint lacks conditioning weights. Proceeding with random embeddings (IS with unconditioned proposal).")
        
    return model, force_conditioning


def _format_key(temp: float, field: float) -> str:
    return f"{temp:.4f}K_h{field:.4f}"


def run_sampling(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    has_conditioning: bool
) -> Dict[str, Dict[str, Any]]:
    
    configs = defaultdict(list)
    energies = defaultdict(list)
    x_up = defaultdict(list)
    log_rnd_values = defaultdict(list)
    logp_x_values = defaultdict(list)
    logf_t_values = defaultdict(list)
    ness_values: Dict[str, float] = {}
    free_energies: Dict[str, float] = {}
    
    # Define reward function wrapper
    def reward_wrapper(x, beta=None, h=0, J=1, use_bias=False):
        # x is [B, D] in {0, 1}
        # ising2d_ham expects spins in {-1, 1}
        # But wait, utils_train.rnd passes x to reward_model
        # and we saw in ising_model_eval.ipynb:
        # reward = lambda S: -beta * ising2d_ham(2*S-1, J=J, h=h)
        
        # NOTE: utils_train.rnd passes beta_batch if provided.
        # However, ising2d_ham usually takes scalar beta or simple broadcasting.
        # Let's handle both.
        
        spins = 2 * x.float() - 1
        return -beta * ising2d_ham(spins, J=J, h=h)

    # Loop over conditions
    for temp in tqdm(args.temps, desc="Temps"):
        beta = 1.0 / temp
        for field in tqdm(args.fields, desc="Fields", leave=False):
            key = _format_key(temp, field)
            
            samples_collected = 0
            
            batch_configs = []
            batch_energies = []
            batch_x_up = []
            batch_log_rnd = []
            batch_logp_x = []
            batch_logf_t = []
            
            pbar = tqdm(total=args.num_samples, desc=f"Sampling {key}", leave=False)
            
            while samples_collected < args.num_samples:
                current_batch_size = min(args.batch_size, args.num_samples - samples_collected)
                with torch.no_grad():
                    # Handle conditional vs unconditional
                    if has_conditioning:
                        # Pass beta/h to rnd
                        x, log_rnd = rnd(
                            model, 
                            reward_wrapper, 
                            batch_size=current_batch_size, 
                            device=device,
                            beta_batch=torch.full((current_batch_size,), beta, device=device).float(),
                            h_batch=torch.full((current_batch_size,), field, device=device).float(),
                            J=args.J
                        )
                        # Compute stats
                        logf_t, logp_x = _compute_log_stats(
                            x, log_rnd, reward_wrapper, model,
                            beta_batch=torch.full((current_batch_size,), beta, device=device).float(),
                            h_batch=torch.full((current_batch_size,), field, device=device).float(),
                            J=args.J
                        )
                    else:
                        # Unconditional model: do NOT pass beta/h to rnd (so model(x) is used)
                        # But wrap reward to use target beta
                        current_reward_fn = lambda x_in, **kwargs: reward_wrapper(x_in, beta=beta, h=field, J=args.J, **kwargs)
                        
                        x, log_rnd = rnd(
                            model, 
                            current_reward_fn,
                            batch_size=current_batch_size, 
                            device=device,
                            beta_batch=None, 
                            h_batch=None,
                            J=args.J
                        )
                         # Compute stats
                        logf_t, logp_x = _compute_log_stats(
                            x, log_rnd, current_reward_fn, model,
                            beta_batch=None,
                            h_batch=None,
                            J=args.J
                        )

                    # Compute energies (unweighted, raw energy H(s))
                    # H(s) = -J * sum(s_i s_j) - h * sum(s_i)
                    # ising2d_ham yields -beta * H(s). So H(s) = ising2d_ham / (-beta) ??
                    # Wait, ising2d_ham returns ENERGY if beta=1?
                    # ising2d_ham implementation: returns - \sum <neighbors> - h \sum s_i
                    # Wait, ising_model_eval says: reward = ... -beta * ising2d_ham(..., J=J, h=h)
                    # Usually Hamiltonian H is what we want. 
                    # Let's check ising2d_ham in utils_ising.py later if needed.
                    # Assuming ising2d_ham returns the Hamiltonian H(x).
                    # Actually, usually `ising2d_ham` computes the energy H.
                    # And probability is exp(-beta * H).
                    # So log_prob is -beta * H.
                    # In eval notebook: reward = -beta * ising2d_ham(...)
                    # This implies ising2d_ham returns H.
                    
                    spins = 2 * x.float() - 1
                    raw_energy = ising2d_ham(spins, J=args.J, h=field)
                    
                    # Magnetization (x_up)
                    # x is 0,1. x_up is fraction of 1s? Or magnetization per site?
                    # mam_arm_sampling says: x_spin_up = numbers.float().mean(dim=1)
                    # For Ising (0,1), this is density of 1s.
                    mag = x.float().mean(dim=1)
                    
                    # Store batch
                    batch_configs.append(x.cpu().numpy())
                    batch_energies.append(raw_energy.cpu().numpy())
                    batch_x_up.append(mag.cpu().numpy())
                    batch_log_rnd.append(log_rnd.cpu().numpy())
                    batch_logp_x.append(logp_x.cpu().numpy())
                    batch_logf_t.append(logf_t.cpu().numpy())
                    
                    samples_collected += current_batch_size
                    pbar.update(current_batch_size)
            
            pbar.close()
            
            # Concatenate
            configs[key] = np.concatenate(batch_configs, axis=0)
            energies[key] = np.concatenate(batch_energies, axis=0)
            x_up[key] = np.concatenate(batch_x_up, axis=0)
            log_rnd_values[key] = np.concatenate(batch_log_rnd, axis=0)
            logp_x_values[key] = np.concatenate(batch_logp_x, axis=0)
            logf_t_values[key] = np.concatenate(batch_logf_t, axis=0)
            
            # Compute aggregate metrics
            # NESS
            log_rnd_tensor = torch.tensor(log_rnd_values[key], device=device)
            ness_val = ess(log_rnd_tensor, normalize=True)
            ness_values[key] = float(ness_val) if isinstance(ness_val, (float, int)) else ness_val.item()
            
            # Free Energy F = -1/beta * log(Z)
            # We estimate log(Z) via IS: Z = E_q [ p*(x)/q(x) ] (approx)
            # Actually log_rnd = log(p*/q).
            # log Z = log E_q [ exp(log_rnd) ] = logsumexp(log_rnd) - log(N)
            # Free Energy (Dimensionless f = - log Z) or physical F = -kT log Z?
            # Mam_arm_sampling:
            # free_energy = get_ensemble_free_energy(log_rw_tensor, ...)
            # Let's implement simple estimator here: F = - (1/beta) * (logsumexp(log_rnd) - log(N))
            # Wait, log_rnd includes the partition function of p*? No, p* is unnormalized usually.
            # Reward is -beta * H.
            # Target p*(x) = exp(-beta * H).
            # q(x) is model prob.
            # log_rnd = log(p*/q) = -beta * H - log q.
            # E[ exp(log_rnd) ] w.r.t q = \sum q * (p*/q) = \sum p* = Z.
            # So log Z \approx logsumexp(log_rnd) - log(N).
            # Free Energy F = -1/beta * log Z.
            log_Z = torch.logsumexp(log_rnd_tensor, dim=0) - np.log(len(log_rnd_tensor))
            f_val = -(1.0/beta) * log_Z
            free_energies[key] = f_val.item()
            
            logger.info(f"[{key}] NESS: {ness_values[key]:.4f}, Free Energy: {free_energies[key]:.4f}")

    results = {
        "configs": dict(configs),
        "energies": dict(energies),
        "x_up": dict(x_up),
        # Requested "log_rnd" instead of "log_rw"
        "log_rnd": dict(log_rnd_values), 
        "logp_x": dict(logp_x_values),
        "logf_t": dict(logf_t_values),
        "ness": ness_values,
        "free_energies": free_energies,
    }
    
    return results


def save_results(args, results):
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    
    logger.info(f"Saving results to {output_path}")
    with open(output_path, "wb") as f:
        pkl.dump(results, f)


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    model, has_conditioning = load_model(args, device)
    
    # Run sampling
    
    # Check for conditioning logic requested by user
    # Logic now handled inside load_model and returned as has_conditioning (force_conditioning)
    results = run_sampling(model, args, device, has_conditioning)
    
    # Save results
    save_results(args, results)


if __name__ == "__main__":
    main()
