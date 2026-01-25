#!/usr/bin/env python
"""
MCMC sampling with pre-trained biased energy landscape for Potts model.
Loads a bias potential from a checkpoint and performs fixed-bias MCMC sampling.
Outputs follow the same format as mdns_sampling.py for compatibility.
"""

import argparse
import json
import logging
import os
import pickle as pkl
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path to allow imports from MDNS
mdns_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(mdns_root))

# Add snowy-flow to path if needed
snowy_flow_path = os.path.abspath(os.path.join(mdns_root, '../snowy-flow-dev'))
if os.path.exists(snowy_flow_path):
    sys.path.insert(0, snowy_flow_path)

from snowyflow.model.energy.ising import LatticePottsModel

from bias import BiasPotentialMultiDim
from mcmc_potts_metad import BiasedLatticePottsModel, compute_cv_potts
from utils_potts import potts2d_ham

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCMC sampling with pre-trained biased energy landscape for Potts model."
    )
    
    # Model arguments
    parser.add_argument("--L", type=int, default=16, help="Lattice size (Linear dimension)")
    parser.add_argument("--q", type=int, default=3, help="Number of Potts states (q)")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling constant")
    parser.add_argument("--beta", type=float, default=0.5, help="Inverse temperature")
    
    # Bias checkpoint (required)
    parser.add_argument(
        "--bias-checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file containing bias potential (ckpt_*.pth or final.pth)"
    )
    
    # Sampling configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total number of samples to generate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for parallel chains"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of MCMC steps between samples (burn-in + thinning)"
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=50,
        help="Number of burn-in steps before collecting samples"
    )
    
    # Output
    parser.add_argument(
        "--output-folder",
        type=str,
        default="outputs/biased_mcmc",
        help="Directory to save results"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="biased_samples.pkl",
        help="Output filename"
    )
    
    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    return parser.parse_args()


def load_bias_from_checkpoint(checkpoint_path: str, device: torch.device, args: argparse.Namespace) -> BiasPotentialMultiDim:
    """
    Load bias potential from checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file (ckpt_*.pth or final.pth)
        device: Torch device
        args: Arguments namespace (for fallback values)
    
    Returns:
        Initialized BiasPotentialMultiDim with loaded state
    """
    logger.info(f"Loading bias potential from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract bias potential state dict
    if "bias_potential" not in checkpoint:
        raise KeyError(f"bias_potential not found in checkpoint. Available keys: {list(checkpoint.keys())}")
    
    bias_state = checkpoint["bias_potential"]
    
    # Extract parameters from state dict
    if "params" in bias_state:
        params = bias_state["params"]
        logger.info(f"Using bias params from checkpoint: {params}")
        
        # Get parameters
        cv_min = params.get("cv_min", [-0.6, -1.0])
        cv_max = params.get("cv_max", [1.1, 1.0])
        grid_size = params.get("grid_size", 100)
        sigma = params.get("sigma", 0.05)
        initial_height = params.get("initial_height", 0.1)
        bias_factor = params.get("bias_factor", 10.0)
        T = params.get("T", 1.0 / args.beta)
        kernel_type = params.get("kernel_type", "gaussian")
        
        # Convert to lists if needed
        if isinstance(cv_min, (int, float)):
            cv_min = [float(cv_min)]
        if isinstance(cv_max, (int, float)):
            cv_max = [float(cv_max)]
        if isinstance(grid_size, (int, str)):
            if isinstance(grid_size, str) and ',' in grid_size:
                grid_size = [int(x.strip()) for x in grid_size.split(',')]
            else:
                grid_size = int(grid_size)
        if isinstance(grid_size, int):
            grid_size = [grid_size] * len(cv_min)
        if isinstance(sigma, (int, float, str)):
            if isinstance(sigma, str) and ',' in sigma:
                sigma = [float(x.strip()) for x in sigma.split(',')]
            else:
                sigma = float(sigma)
        if isinstance(sigma, (int, float)):
            sigma = [float(sigma)] * len(cv_min)
        
        # Get energy scaling from checkpoint args if available
        energy_scaling = 1.0
        if "args" in checkpoint:
            checkpoint_args = checkpoint["args"]
            if "scale_bias_with_size" in checkpoint_args and checkpoint_args["scale_bias_with_size"]:
                # Extract L from checkpoint args or use current args
                L_val = checkpoint_args.get("L", args.L)
                D = L_val ** 2
                energy_scaling = float(D) / 16.0
        
        # Create bias potential
        bias_pot = BiasPotentialMultiDim(
            cv_min=cv_min,
            cv_max=cv_max,
            grid_size=grid_size,
            sigma=sigma,
            initial_height=initial_height,
            bias_factor=bias_factor,
            T=T,
            kernel_type=kernel_type,
            device=device,
            energy_scaling=energy_scaling
        )
        
        # Load state
        bias_pot.load_state_dict(bias_state)
        logger.info("Successfully loaded bias potential from checkpoint")
        
    else:
        raise ValueError("Bias potential state dict does not contain 'params' key. Cannot reconstruct bias potential.")
    
    return bias_pot


def run_biased_mcmc_sampling(
    model: BiasedLatticePottsModel,
    bias_pot: BiasPotentialMultiDim,
    args: argparse.Namespace,
    device: torch.device
) -> Dict[str, any]:
    """
    Run MCMC sampling with fixed bias potential.
    
    Args:
        model: BiasedLatticePottsModel instance
        bias_pot: BiasPotentialMultiDim instance (fixed, not updated)
        args: Arguments namespace
        device: Torch device
    
    Returns:
        Dictionary with results matching mdns_sampling.py format
    """
    logger.info(f"Starting biased MCMC sampling: {args.num_samples} samples, batch_size={args.batch_size}")
    
    # Initialize samples
    samples = model.init_sample(args.batch_size).to(device)
    
    # Temperature and fields tensors
    temps = torch.full((args.batch_size,), 1.0 / args.beta, device=device)
    fields = torch.zeros((args.batch_size,), device=device)
    
    # Storage for results
    all_configs = []
    all_energies = []
    all_x_up = []
    all_weights = []
    all_cv_values = []
    
    # Format key (matching mdns_sampling.py format)
    temp_k = 1.0 / args.beta  # For consistency with mdns_sampling format
    field = 0.0
    key = f"{temp_k:.4f}K_h{field:.4f}"
    
    samples_collected = 0
    pbar = tqdm(total=args.num_samples, desc="Sampling")
    
    # Burn-in phase
    if args.burn_in > 0:
        logger.info(f"Running burn-in: {args.burn_in} steps")
        for _ in range(args.burn_in):
            samples = model.step(samples, temps, fields, criterion="metropolis")
    
    # Sampling phase
    while samples_collected < args.num_samples:
        # Run MCMC steps
        for _ in range(args.num_steps):
            samples = model.step(samples, temps, fields, criterion="metropolis")
        
        # Collect samples from current batch
        current_batch_size = min(args.batch_size, args.num_samples - samples_collected)
        batch_samples = samples[:current_batch_size]
        
        # Compute energies (raw Potts energy, not biased)
        raw_energies = potts2d_ham(batch_samples, J=args.J, q=args.q)  # [B]
        
        # Compute CV values
        cv_batch = compute_cv_potts(batch_samples, args.q, device)  # [B, 2] for q=3
        
        # Compute x_up: fraction of sites in most frequent state
        batch_samples_np = batch_samples.detach().cpu().numpy()
        B, D = batch_samples_np.shape
        L = int(np.sqrt(D))
        batch_samples_2d = batch_samples_np.reshape(B, L, L)
        
        x_up_batch = np.zeros(B)
        for i in range(B):
            sample = batch_samples_2d[i]
            counts = np.bincount(sample.flatten(), minlength=args.q)
            max_count = counts.max()
            x_up_batch[i] = max_count / (L * L)
        
        # Compute unbiasing weights: w = exp(beta * V(s))
        v_s = bias_pot.evaluate(cv_batch)  # [B] bias potential values
        beta_bias = 1.0 / bias_pot.T  # Inverse temperature for bias
        log_w = beta_bias * v_s  # [B]
        
        # Clamp to prevent overflow
        log_w_max = 50.0
        log_w_clamped = torch.clamp(log_w, max=log_w_max)
        w = torch.exp(log_w_clamped).detach().cpu().numpy()
        
        # Store results
        all_configs.append(batch_samples.detach().cpu().numpy())
        all_energies.append(raw_energies.detach().cpu().numpy())
        all_x_up.append(x_up_batch)
        all_weights.append(w)
        all_cv_values.append(cv_batch.detach().cpu().numpy())
        
        samples_collected += current_batch_size
        pbar.update(current_batch_size)
    
    pbar.close()
    
    # Concatenate all results
    configs = np.concatenate(all_configs, axis=0)
    energies = np.concatenate(all_energies, axis=0)
    x_up = np.concatenate(all_x_up, axis=0)
    weights = np.concatenate(all_weights, axis=0)
    cv_values = np.concatenate(all_cv_values, axis=0)
    
    # Format results matching mdns_sampling.py
    results = {
        "configs": {key: configs},
        "energies": {key: energies},
        "x_up": {key: x_up},
        "weights": {key: weights},
        "cv_values": {key: cv_values},
    }
    
    logger.info(f"Sampling complete: {samples_collected} samples collected")
    logger.info(f"Energy range: [{energies.min():.4f}, {energies.max():.4f}]")
    logger.info(f"x_up range: [{x_up.min():.4f}, {x_up.max():.4f}]")
    logger.info(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    
    return results


def save_results(args: argparse.Namespace, results: Dict[str, any]) -> None:
    """Save results to pickle file."""
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    
    logger.info(f"Saving results to {output_path}")
    with open(output_path, "wb") as f:
        pkl.dump(results, f)
    
    # Also save a summary JSON
    summary_path = output_dir / (args.output_name.replace(".pkl", "_summary.json"))
    summary = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "burn_in": args.burn_in,
        "L": args.L,
        "q": args.q,
        "J": args.J,
        "beta": args.beta,
        "bias_checkpoint": args.bias_checkpoint,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


def main():
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load bias potential from checkpoint
    bias_pot = load_bias_from_checkpoint(args.bias_checkpoint, device, args)
    
    # Initialize model with bias
    logger.info(f"Initializing Potts model: L={args.L}, q={args.q}, J={args.J}")
    
    def cv_wrapper(x):
        return compute_cv_potts(x, args.q, device)
    
    model = BiasedLatticePottsModel(
        dim=args.L,
        q=args.q,
        init_sigma=args.J,  # Set sigma to J
        n_samples=args.batch_size,
        rand=True,
        lattice_dim=2,
        bias_potential=bias_pot,
        cv_compute_fn=cv_wrapper
    )
    # Ensure sigma is set correctly
    model.sigma.data.fill_(args.J)
    model.to(device)
    model.sampler.to(device)
    
    # Run sampling
    results = run_biased_mcmc_sampling(model, bias_pot, args, device)
    
    # Save results
    save_results(args, results)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
