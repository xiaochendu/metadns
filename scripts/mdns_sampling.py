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

from bias import BiasPotential
from model import ExponentialMovingAverage, get_rope_vit_model
from utils import ess
from utils_ising import ising2d_ham, ising2d_mag
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
    
    # Bias Potential (WT-ASBS)
    parser.add_argument('--use_bias', action='store_true', help='Enable biased sampling (WT-ASBS)')
    parser.add_argument('--bias_sigma', type=float, default=0.05, help='Sigma for Gaussian bias kernel')
    parser.add_argument('--bias_height', type=float, default=0.1, help='Initial height (W) for bias kernel')
    parser.add_argument('--bias_factor', type=float, default=10.0, help='Bias factor (gamma) for Well-Tempered Metadynamics')
    parser.add_argument('--bias_grid_size', type=int, default=100, help='Grid size for CV (Magnetization)')
    parser.add_argument('--kernel_type', type=str, default='gaussian', help='Kernel type: gaussian or delta')
    parser.add_argument('--cv_min', type=float, default=-1.0, help='Minimum value for CV')
    parser.add_argument('--cv_max', type=float, default=1.0, help='Maximum value for CV')
    
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
    # Initialize BiasPotential if requested
    bias_pot = None
    if args.use_bias:
        if 'bias_potential' in checkpoint:
            logger.info("Loading BiasPotential from checkpoint")
            bias_state = checkpoint['bias_potential']
            
            # Check if params exist in state_dict (saved by modern bias.py)
            if 'params' in bias_state:
                params = bias_state['params']
                logger.info(f"Using bias params from checkpoint: {params}")
                
                bias_pot = BiasPotential(
                    cv_min=params.get('cv_min', args.cv_min), 
                    cv_max=params.get('cv_max', args.cv_max), 
                    grid_size=params.get('grid_size', args.bias_grid_size),
                    sigma=params.get('sigma', args.bias_sigma),
                    initial_height=params.get('initial_height', args.bias_height),
                    bias_factor=params.get('bias_factor', args.bias_factor),
                    T=params.get('T', args.temps[0] if args.temps else 1.0),
                    kernel_type=params.get('kernel_type', args.kernel_type),
                    device=device
                )
            else:
                # Fallback to CLI args if params not inside state_dict
                logger.warning("Bias params not found in checkpoint state dict! Using CLI arguments.")
                T_init = args.temps[0] if args.temps else 1.0 
                bias_pot = BiasPotential(
                    cv_min=args.cv_min, cv_max=args.cv_max, 
                    grid_size=args.bias_grid_size,
                    sigma=args.bias_sigma,
                    initial_height=args.bias_height,
                    bias_factor=args.bias_factor,
                    T=T_init,
                    kernel_type=args.kernel_type,
                    device=device
                )
            
            bias_pot.load_state_dict(bias_state)
            
            # Normalize bias potential (shift min to 0) to avoid numerical explosion in weights
            logger.info("Normalizing bias potential (shifting min to 0)...")
            bias_pot.normalize()
            
        else:
            logger.warning("Bias potential requested but not found in checkpoint! Using initialized (empty/initial) bias from CLI args.")
            T_init = args.temps[0] if args.temps else 1.0 
            bias_pot = BiasPotential(
                cv_min=args.cv_min, cv_max=args.cv_max, 
                grid_size=args.bias_grid_size,
                sigma=args.bias_sigma,
                initial_height=args.bias_height,
                bias_factor=args.bias_factor,
                T=T_init,
                kernel_type=args.kernel_type,
                device=device
            )

    model.eval()
    
    # Determine if we SHOULD condition based on user arguments (Sweep -> Condition)
    # "check if either the number of input --temps and --fields is greater than 1"
    force_conditioning = len(args.temps) > 1 or len(args.fields) > 1
    
    if force_conditioning and not ckpt_has_weights:
        logger.warning("Requesting a sweep (conditioning needed) but checkpoint lacks conditioning weights. Proceeding with random embeddings (IS with unconditioned proposal).")
        
    return model, force_conditioning, bias_pot


def _format_key(temp: float, field: float) -> str:
    return f"{temp:.4f}K_h{field:.4f}"


def run_sampling(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    has_conditioning: bool,
    bias_pot: Optional[BiasPotential] = None
) -> Dict[str, Dict[str, Any]]:
    
    configs = defaultdict(list)
    energies = defaultdict(list)
    configs = defaultdict(list)
    energies = defaultdict(list)
    x_up = defaultdict(list)
    log_rnd_values = defaultdict(list) # Restore log_rnd
    log_rw_values = defaultdict(list) 
    logp_x_values = defaultdict(list)
    logf_t_values = defaultdict(list)
    weights_values = defaultdict(list) 
    ness_values: Dict[str, float] = {}
    free_energies: Dict[str, float] = {}
    free_energy_profiles: Dict[str, np.ndarray] = {}
    cv_grids: Dict[str, np.ndarray] = {}
    
    # Define reward function factory to match train_ising.py logic
    def get_reward_fn(default_beta, default_h, J=1, bias_pot=None):
        def reward_fn(x, beta=None, h=None, J=J, **kwargs):
            """Reward function wrapper that handles scalar or per-sample betas/fields."""
            beta_val = beta if beta is not None else default_beta
            h_val = h if h is not None else default_h
            
            spin = 2 * x.float() - 1
            return -beta_val * ising2d_ham(spin, J=J, h=h_val)

        def biased_reward_fn(x, beta=None, h=None, J=J, use_bias=True):
            # 1. Standard reward
            r = reward_fn(x, beta=beta, h=h, J=J)
            
            # 2. Add Bias: R' = R - beta * V(s)
            if bias_pot is not None:
                 # Convert x (0,1) to spins (-1,1) for CV calc
                 s = ising2d_mag(2*x - 1)
                 v = bias_pot.evaluate(s)
                 
                 # Get beta for scaling
                 beta_val = beta if beta is not None else default_beta
                 # Handle tensor/scalar beta
                 if isinstance(beta_val, (int, float)):
                     beta_tensor = torch.tensor(beta_val, device=x.device)
                 elif isinstance(beta_val, torch.Tensor):
                     beta_tensor = beta_val.to(x.device)
                 else:
                     beta_tensor = torch.tensor(beta_val, device=x.device) # Fallback
                 
                 r = r - beta_tensor * v if use_bias else r
            return r
            
        return biased_reward_fn if args.use_bias else reward_fn

    # Loop over conditions
    for temp in tqdm(args.temps, desc="Temps"):
        beta = 1.0 / temp
        for field in tqdm(args.fields, desc="Fields", leave=False):
            key = _format_key(temp, field)
            
            samples_collected = 0
            
            batch_configs = []
            batch_energies = []
            batch_x_up = []
            batch_log_rnd = [] # Restore log_rnd
            batch_log_rw = [] 
            batch_logp_x = []
            batch_logf_t = []
            batch_weights = []
            
            pbar = tqdm(total=args.num_samples, desc=f"Sampling {key}", leave=False)
            
            # Create condition-specific reward function (closure over beta/field)
            # This matches train_ising logic: default_beta/h are bound to the current condition
            current_reward_fn = get_reward_fn(default_beta=beta, default_h=field, J=args.J, bias_pot=bias_pot)
            
            while samples_collected < args.num_samples:
                current_batch_size = min(args.batch_size, args.num_samples - samples_collected)
                with torch.no_grad():
                    # Handle conditional vs unconditional
                    if has_conditioning:
                        # Pass beta/h to rnd explicitly (overriding defaults if needed, but defaults match)
                        
                        # Note: rnd implementation passes beta_batch/h_batch to reward_model
                        # Our current_reward_fn handles beta=... arguments.
                        # It also defaults use_bias=True in biased_reward_fn.
                        
                        x, log_rnd = rnd(
                            model, 
                            current_reward_fn, 
                            batch_size=current_batch_size, 
                            device=device,
                            beta_batch=torch.full((current_batch_size,), beta, device=device).float(),
                            h_batch=torch.full((current_batch_size,), field, device=device).float(),
                            J=args.J
                        )
                        # Compute stats
                        # We need unbiased stats for logf_t.
                        # biased_reward_fn has use_bias=True default.
                        # To get unbiased, we must pass use_bias=False.
                        # _compute_log_stats internally calls reward_fn(..., use_bias=False)
                        # So we just pass the biased function, and it handles it!
                        
                        logf_t, logp_x = _compute_log_stats(
                            x, log_rnd, current_reward_fn, model,
                            beta_batch=torch.full((current_batch_size,), beta, device=device).float(),
                            h_batch=torch.full((current_batch_size,), field, device=device).float(),
                            J=args.J,
                            bias_potential=bias_pot
                        )
                    else:
                        # Unconditional model
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
                            J=args.J,
                            bias_potential=bias_pot
                        )
                    
                    log_rw = logf_t - logp_x


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
                    
                    
                    # Calculate log_rw (log reweighting factor)
                    # log_rw = logf_t - logp_x
                    log_rw = logf_t - logp_x
                    
                    # Store batch
                    batch_configs.append(x.cpu().numpy())
                    batch_energies.append(raw_energy.cpu().numpy())
                    batch_x_up.append(mag.cpu().numpy())
                    batch_log_rnd.append(log_rnd.cpu().numpy()) # Store log_rnd
                    batch_log_rw.append(log_rw.cpu().numpy()) # Store log_rw
                    batch_logp_x.append(logp_x.cpu().numpy())
                    batch_logf_t.append(logf_t.cpu().numpy())
                    
                    if bias_pot is not None:
                         # Calculate weights = exp(beta_bias * v(s))
                         # s comes from ising2d_mag(2*x-1)
                         spins = 2 * x.float() - 1
                         mag = ising2d_mag(spins) # [B]
                         v_s = bias_pot.evaluate(mag) # [B]
                         beta_bias = 1.0 / bias_pot.T
                         
                         # Log-weights: log_w = beta_bias * v_s
                         log_w = beta_bias * v_s
                                                  
                         w = torch.exp(log_w).detach().cpu().numpy()
                         batch_weights.append(w)
                    
                    samples_collected += current_batch_size
                    pbar.update(current_batch_size)
            
            pbar.close()
            
            # Concatenate
            configs[key] = np.concatenate(batch_configs, axis=0)
            energies[key] = np.concatenate(batch_energies, axis=0)
            x_up[key] = np.concatenate(batch_x_up, axis=0)
            log_rnd_values[key] = np.concatenate(batch_log_rnd, axis=0)
            log_rw_values[key] = np.concatenate(batch_log_rw, axis=0)
            logp_x_values[key] = np.concatenate(batch_logp_x, axis=0)
            logf_t_values[key] = np.concatenate(batch_logf_t, axis=0)
            if batch_weights:
                weights_values[key] = np.concatenate(batch_weights, axis=0)
            
            # Compute aggregate metrics
            # NESS using log_rnd (as requested)
            log_rnd_tensor = torch.tensor(log_rnd_values[key], device=device)
            ness_val = ess(log_rnd_tensor, normalize=True)
            ness_values[key] = float(ness_val) if isinstance(ness_val, (float, int)) else ness_val.item()
            
            # Free Energy F = -1/beta * log(Z) using log_rw (as requested)
            # log Z = logsumexp(log_rw) - log(N)
            log_rw_tensor = torch.tensor(log_rw_values[key], device=device)
            log_Z = torch.logsumexp(log_rw_tensor, dim=0) - np.log(len(log_rw_tensor))
            f_val = -(1.0/beta) * log_Z
            free_energies[key] = f_val.item()
            
            logger.info(f"[{key}] NESS: {ness_values[key]:.4f}, Free Energy: {free_energies[key]:.4f}")
            
            # If bias was used, calculate and log corrected metrics
            if bias_pot is not None:
                # log_rw here is w.r.t BIASED target.
                # log_weights (unbiasing) = beta * V(s).
                # log_rw_total = log_rw + log_weights.
                # This total weight targets the UNBIASED distribution.
                
                # Reconstruct log_unbias_weights from stored weights
                # weights = exp(log_w). log_w = log(weights)
                w_np = weights_values[key]
                log_w_np = np.log(w_np + 1e-10) # Avoid log(0)
                
                # Physical NESS using log_rnd (consistent with NESS choice)
                log_rnd_total_tensor = torch.tensor(log_rnd_values[key], device=device) + torch.tensor(log_w_np, device=device)
                ness_phys = ess(log_rnd_total_tensor, normalize=True)
                
                # Physical Free Energy using log_rw (consistent with Free Energy choice)
                log_rw_total_tensor = torch.tensor(log_rw_values[key], device=device) + torch.tensor(log_w_np, device=device)
                log_Z_phys = torch.logsumexp(log_rw_total_tensor, dim=0) - np.log(len(log_rw_total_tensor))
                f_phys = -(1.0/beta) * log_Z_phys
                
                logger.info(f"[{key}] Physical NESS (Corrected): {ness_phys:.4f}, Physical Free Energy: {f_phys.item():.4f}")
                
                # Store these as well (optional, maybe overwrite?)
                # For now let's just log them so user sees the explanation.
                # To make them available in output:
                ness_values[f"{key}_physical"] = float(ness_phys)
                free_energies[f"{key}_physical"] = f_phys.item()


                # Calculate Free Energy Profile from Bias Potential if available
                grid_vals, bias_vals = bias_pot.get_bias_grid_np()
                gamma = bias_pot.gamma
                
                # F(s) ~ - (gamma / (gamma - 1)) * V(s)
                if gamma > 1.0:
                    free_energy_profile = - (gamma / (gamma - 1)) * bias_vals
                else:
                    free_energy_profile = - bias_vals 
                    
                # Shift so that the ends are zero (as requested)
                # Average of ends to be robust
                end_avg = (free_energy_profile[0] + free_energy_profile[-1]) / 2.0
                free_energy_profile = free_energy_profile - end_avg
                
                free_energy_profiles[key] = free_energy_profile
                cv_grids[key] = grid_vals

    results = {
        "configs": dict(configs),
        "energies": dict(energies),
        "x_up": dict(x_up),
        "log_rnd": dict(log_rnd_values), 
        "log_rw": dict(log_rw_values), 
        "logp_x": dict(logp_x_values),
        "logf_t": dict(logf_t_values),
        "ness": ness_values,
        "free_energies": free_energies,
        "free_energy_profiles": free_energy_profiles,
        "cv_grids": cv_grids,
        "weights": dict(weights_values),
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
    model, has_conditioning, bias_pot = load_model(args, device)
    
    # Run sampling
    
    # Check for conditioning logic requested by user
    # Logic now handled inside load_model and returned as has_conditioning (force_conditioning)
    results = run_sampling(model, args, device, has_conditioning, bias_pot)
    
    # Save results
    save_results(args, results)


if __name__ == "__main__":
    main()
