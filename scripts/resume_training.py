#!/usr/bin/env python3
"""
Resume training from an existing checkpoint.

This script:
1. Loads the config.json from the training directory
2. Finds the latest checkpoint
3. Extracts the wandb run ID from the wandb directory
4. Calculates remaining epochs to train
5. Resumes training with the correct wandb run ID

Usage:
    python resume_training.py --config_dir <path_to_config_dir> --total_epochs <total_epochs>
    python resume_training.py --config_path <path_to_config.json> --total_epochs <total_epochs>
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def find_latest_checkpoint(config_dir):
    """Find the latest checkpoint in the directory."""
    config_dir = Path(config_dir)
    
    # Check for numbered checkpoints (ckpt_*.pth)
    checkpoint_files = list(config_dir.glob("ckpt_*.pth"))
    if checkpoint_files:
        # Extract epoch numbers and find the latest
        def get_epoch(fpath):
            match = re.search(r'ckpt_(\d+)\.pth', fpath.name)
            return int(match.group(1)) if match else 0
        
        latest_checkpoint = max(checkpoint_files, key=get_epoch)
        epoch = get_epoch(latest_checkpoint)
        return str(latest_checkpoint), epoch
    
    # Fall back to weights.pth if no numbered checkpoints
    weights_path = config_dir / "weights.pth"
    if weights_path.exists():
        # Try to infer epoch from losses length in checkpoint
        # For now, we'll need to load it, but that requires torch
        # So we'll return None for epoch and let the training script handle it
        return str(weights_path), None
    
    raise FileNotFoundError(f"No checkpoint found in {config_dir}")


def get_latest_wandb_run_dir(config_dir):
    """Get the latest wandb run directory."""
    config_dir = Path(config_dir)
    wandb_dir = config_dir / "wandb"
    
    if not wandb_dir.exists():
        return None
    
    # Check for latest-run symlink
    latest_run_link = wandb_dir / "latest-run"
    if latest_run_link.exists() and latest_run_link.is_symlink():
        # Use resolve() to handle both absolute and relative symlinks
        target = latest_run_link.resolve()
        if target.exists():
            return target
    
    # Fallback: look for run directories directly
    run_dirs = list(wandb_dir.glob("run-*"))
    if run_dirs:
        # Get the most recent one
        latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
        return latest_run
    
    return None


def extract_wandb_run_id(config_dir):
    """Extract wandb run ID from the wandb directory."""
    latest_run_dir = get_latest_wandb_run_dir(config_dir)
    if latest_run_dir is None:
        return None
    
    # Extract run ID from directory name: run-YYYYMMDD_HHMMSS-<run_id>
    match = re.search(r'run-\d{8}_\d{6}-([a-z0-9]+)', latest_run_dir.name)
    if match:
        return match.group(1)
    
    return None


def extract_wandb_run_name(config_dir):
    """Extract wandb run name from the wandb metadata file."""
    latest_run_dir = get_latest_wandb_run_dir(config_dir)
    if latest_run_dir is None:
        return None
    
    # Look for wandb-metadata.json in the run directory
    metadata_file = latest_run_dir / "files" / "wandb-metadata.json"
    if not metadata_file.exists():
        # Try alternative location
        metadata_file = latest_run_dir / "wandb-metadata.json"
        if not metadata_file.exists():
            return None
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Extract run name from args array
        args = metadata.get("args", [])
        if isinstance(args, list):
            # Look for --wandb_run_name in args
            for i, arg in enumerate(args):
                if arg == "--wandb_run_name" and i + 1 < len(args):
                    return args[i + 1]
        
        return None
    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Warning: Could not read wandb metadata: {e}")
        return None


def get_current_epoch_from_checkpoint(checkpoint_path):
    """Get current epoch from checkpoint by loading it."""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Current epoch is the length of losses list (0-indexed, so length = next epoch)
        losses = checkpoint.get('losses', [])
        if losses:
            return len(losses)
        # Fallback: try to extract from filename
        match = re.search(r'ckpt_(\d+)\.pth', Path(checkpoint_path).name)
        if match:
            return int(match.group(1))
        return None
    except (ImportError, OSError, RuntimeError) as e:
        print(f"Warning: Could not load checkpoint to get epoch: {e}")
        # Try to extract from filename
        match = re.search(r'ckpt_(\d+)\.pth', Path(checkpoint_path).name)
        if match:
            return int(match.group(1))
        return None


def load_config(config_path):
    """Load config.json."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_training_command(config, checkpoint_path, wandb_run_id, wandb_run_name, total_epochs, current_epoch, device):
    """Build the command to resume training."""
    # Calculate remaining epochs
    remaining_epochs = total_epochs - current_epoch if current_epoch is not None else total_epochs
    
    if remaining_epochs <= 0:
        print(f"Warning: Already trained for {current_epoch} epochs, target is {total_epochs}.")
        print("Setting remaining_epochs to total_epochs to continue training.")
        remaining_epochs = total_epochs
    
    # Base command
    cmd = ["python", "train_ising.py"]
    
    # Add all config parameters
    cmd.extend(["--device", device])
    cmd.extend(["--L", str(config["L"])])
    cmd.extend(["--beta", str(config["beta"])])
    cmd.extend(["--J", str(config["J"])])
    cmd.extend(["--dir_name", config["dir_name"]])
    cmd.extend(["--num_epochs", str(remaining_epochs)])
    cmd.extend(["--resample_every_n_step", str(config["resample_every_n_step"])])
    cmd.extend(["--batch_size", str(config["batch_size"])])
    cmd.extend(["--eval_batch_size", str(config["eval_batch_size"])])
    cmd.extend(["--eval_every", str(config["eval_every"])])
    cmd.extend(["--loss_fn", config["loss_fn"]])
    cmd.extend(["--wdce_num_replicates", str(config["wdce_num_replicates"])])
    cmd.extend(["--resume_from_ckpt", checkpoint_path])
    cmd.extend(["--save_every", str(config["save_every"])])
    
    # Model parameters
    model_cfg = config["model"]
    cmd.extend(["--hidden_size", str(model_cfg["hidden_size"])])
    cmd.extend(["--n_blocks", str(model_cfg["n_blocks"])])
    cmd.extend(["--n_heads", str(model_cfg["n_heads"])])
    cmd.extend(["--dtype", model_cfg["dtype"]])
    if model_cfg.get("use_checkpoint", False):
        cmd.append("--use_checkpoint")
    
    # Temperature/field parameters
    if "temps" in config and config["temps"]:
        cmd.extend(["--temps"] + [str(t) for t in config["temps"]])
    if "fields" in config and config["fields"]:
        cmd.extend(["--fields"] + [str(f) for f in config["fields"]])
    if config.get("sample_delta_temp", False):
        cmd.append("--sample_delta_temp")
    if config.get("sample_delta_field", False):
        cmd.append("--sample_delta_field")
    if "delta_temp" in config:
        cmd.extend(["--delta_temp", str(config["delta_temp"])])
    if "delta_field" in config:
        cmd.extend(["--delta_field", str(config["delta_field"])])
    if "min_temp" in config:
        cmd.extend(["--min_temp", str(config["min_temp"])])
    if "max_temp" in config:
        cmd.extend(["--max_temp", str(config["max_temp"])])
    
    # Bias parameters
    if config.get("use_bias", False):
        cmd.append("--use_bias")
        cmd.extend(["--bias_sigma", str(config["bias_sigma"])])
        cmd.extend(["--bias_height", str(config["bias_height"])])
        cmd.extend(["--bias_factor", str(config["bias_factor"])])
        cmd.extend(["--bias_grid_size", str(config["bias_grid_size"])])
        cmd.extend(["--kernel_type", config["kernel_type"]])
        
        # Set CV defaults based on model type if not in config
        # Ising: CV (magnetization) ranges from -1 to 1
        # CuAu: CV (concentration) ranges from 0 to 1
        if "cv_min" in config and "cv_max" in config:
            cv_min = config["cv_min"]
            cv_max = config["cv_max"]
        else:
            # Determine model type: Ising has "L" and "J", CuAu has "size" or other indicators
            is_ising = "L" in config and "J" in config
            is_cuau = "size" in config or (config.get("dir_name", "").lower().find("cuau") != -1 or 
                                          config.get("dir_name", "").lower().find("au") != -1)
            
            if is_cuau:
                # CuAu: concentration CV ranges from 0 to 1
                cv_min = 0.0
                cv_max = 1.0
                print(f"Using default CV range for CuAu: [{cv_min}, {cv_max}]")
            else:
                # Ising: magnetization CV ranges from -1 to 1 (default)
                cv_min = -1.0
                cv_max = 1.0
                print(f"Using default CV range for Ising: [{cv_min}, {cv_max}]")
        
        cmd.extend(["--cv_min", str(cv_min)])
        cmd.extend(["--cv_max", str(cv_max)])
        
        if config.get("scale_bias_with_size", False):
            cmd.append("--scale_bias_with_size")
    
    # Buffer parameters
    if "buffer_size" in config:
        cmd.extend(["--buffer_size", str(config["buffer_size"])])
    if "buffer_ratio" in config:
        cmd.extend(["--buffer_ratio", str(config["buffer_ratio"])])
    
    # Buffer n_bins: default is 8 if not in config
    buffer_n_bins = config.get("buffer_n_bins", 8)
    cmd.extend(["--buffer_n_bins", str(buffer_n_bins)])
    
    # Buffer strategy: default is "balanced" if not in config
    buffer_strategy = config.get("buffer_strategy", "balanced")
    cmd.extend(["--buffer_strategy", buffer_strategy])
    
    # Wandb parameters
    cmd.append("--wandb")
    if "wandb_project" in config:
        cmd.extend(["--wandb_project", config["wandb_project"]])
    else:
        cmd.extend(["--wandb_project", "mdns-ising"])
    
    # Use wandb run name if provided, otherwise try config, then dir_name
    if wandb_run_name:
        cmd.extend(["--wandb_run_name", wandb_run_name])
    elif "wandb_run_name" in config:
        cmd.extend(["--wandb_run_name", config["wandb_run_name"]])
    elif config.get("dir_name"):
        # Use directory name as run name (fallback)
        run_name = Path(config["dir_name"]).name
        cmd.extend(["--wandb_run_name", run_name])
    
    # Add wandb resume ID if available
    # Note: train_ising.py doesn't currently support wandb resume, so we'll need to modify it
    # For now, we'll pass it as an environment variable or modify the script
    # Actually, wandb.init() with id= and resume="allow" should work
    
    return cmd, remaining_epochs, wandb_run_id


def main():
    parser = argparse.ArgumentParser(
        description="Resume training from an existing checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Resume training to 30000 total epochs
    python resume_training.py --config_dir /path/to/training/dir --total_epochs 30000
    
    # Resume training with specific config file
    python resume_training.py --config_path /path/to/config.json --total_epochs 30000
    
    # Resume training with custom device
    python resume_training.py --config_dir /path/to/training/dir --total_epochs 30000 --device cuda:1
        """
    )
    
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Path to directory containing config.json"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config.json file"
    )
    parser.add_argument(
        "--total_epochs",
        type=int,
        required=True,
        help="Total number of epochs to train (will calculate remaining epochs from current checkpoint)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for training (default: cuda:0)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the command without executing it"
    )
    
    args = parser.parse_args()
    
    # Determine config path
    if args.config_path:
        config_path = Path(args.config_path)
        config_dir = config_path.parent
    elif args.config_dir:
        config_dir = Path(args.config_dir)
        config_path = config_dir / "config.json"
    else:
        parser.error("Either --config_dir or --config_path must be provided")
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Load config
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Find latest checkpoint
    print(f"Looking for checkpoints in: {config_dir}")
    try:
        checkpoint_path, epoch_from_filename = find_latest_checkpoint(config_dir)
        print(f"Found checkpoint: {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Get current epoch
    current_epoch = get_current_epoch_from_checkpoint(checkpoint_path)
    if current_epoch is None and epoch_from_filename is not None:
        current_epoch = epoch_from_filename
    elif current_epoch is None:
        print("Warning: Could not determine current epoch. Will train for total_epochs.")
        current_epoch = 0
    
    print(f"Current epoch: {current_epoch}")
    
    # Extract wandb run ID and run name
    wandb_run_id = extract_wandb_run_id(config_dir)
    if wandb_run_id:
        print(f"Found wandb run ID: {wandb_run_id}")
    else:
        print("Warning: Could not find wandb run ID. Will create new wandb run.")
    
    wandb_run_name = extract_wandb_run_name(config_dir)
    if wandb_run_name:
        print(f"Found wandb run name: {wandb_run_name}")
    
    # Build command
    cmd, remaining_epochs, wandb_run_id = build_training_command(
        config, checkpoint_path, wandb_run_id, wandb_run_name, args.total_epochs, current_epoch, args.device
    )
    
    print(f"\nTarget total epochs: {args.total_epochs}")
    print(f"Current epoch: {current_epoch}")
    print(f"Remaining epochs to train: {remaining_epochs}")
    print(f"\nCommand to execute:")
    print(" ".join(cmd))
    
    if wandb_run_id:
        print("\nNote: Wandb run ID found. The script will automatically resume wandb logging.")
        print(f"      Run ID: {wandb_run_id}")
    
    if args.dry_run:
        print("\nDry run mode - not executing command")
        return
    
    # Change to MDNS directory (parent of scripts directory)
    mdns_dir = Path(__file__).parent.parent
    os.chdir(mdns_dir)
    print(f"Changed to MDNS directory: {mdns_dir}")
    
    # Set wandb environment variables if run ID found
    env = os.environ.copy()
    if wandb_run_id:
        env["WANDB_RESUME"] = "allow"
        env["WANDB_RUN_ID"] = wandb_run_id
        print("\nSetting environment variables:")
        print("  WANDB_RESUME=allow")
        print(f"  WANDB_RUN_ID={wandb_run_id}")
    
    # Execute command
    print("\nExecuting training command...")
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
