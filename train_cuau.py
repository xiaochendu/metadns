
import argparse
import copy
import json
import logging
import os
import time

import ase.build
import ase.io
import ase.units
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from ase import Atoms

import utils_train
from bias import BiasPotential
from energy_cuau import AuCuAlloyModel
from model import ExponentialMovingAverage
from model.transformer import MultiOutputTransformer
from utils import plot_energy_au_conc_distributions

K_B = ase.units.kB  # eV/K

# Try to import clease/icet for setup
try:
    from clease.settings import CEBulk, Concentration
    from icet import ClusterExpansion
except ImportError:
    logging.warning("CLEASE/ICET not found. Energy model initialization may fail.")


class TransformerWrapper(nn.Module):
    """
    Wrapper for MultiOutputTransformer to match the interface expected by utils_train.
    
    Exposes:
        - vocab_size: Number of token types (Cu, Au, Mask)
        - length: Sequence length (number of sites)
        - forward(x, beta=None, h=None): Returns logits [B, L, vocab_size]
    """
    def __init__(self, model, vocab_size, length, device):
        super().__init__()
        self.model = model
        self.vocab_size = vocab_size
        self.length = length
        self.device = device

    def forward(self, x, beta=None, h=None):
        # x is [B, L] indices (0, 1, 2). 2 is Mask.
        # MultiOutputTransformer expects atomic numbers or mapped indices.
        # We configured MultiOutputTransformer with num_atom_types=vocab_size (3).
        # So passing x directly is fine.
        
        # Convert beta (inverse temp) to temp.
        # utils_train might pass None, scalars, or batches.
        # MultiOutputTransformer expects tensor inputs for temp/field.
        
        batch_size = x.shape[0]
        
        # Handle beta -> temp
        if beta is None:
            # Default temp? Or 0?
            # Ideally shouldn't happen during training if reward_fn uses correct beta.
            # Let's assume a default small T or 1.0 if not provided.
            temp = torch.zeros(batch_size, device=self.device)
        elif isinstance(beta, torch.Tensor):
            # Avoid division by zero
            temp = 1.0 / (beta + 1e-6)
        else:
            temp = torch.full((batch_size,), 1.0 / (beta + 1e-6), device=self.device)
            
        # Handle h -> field
        if h is None:
            field = torch.zeros(batch_size, 1, device=self.device)
        elif isinstance(h, torch.Tensor):
            if h.dim() == 1:
                field = h.unsqueeze(-1)
            else:
                field = h
        else:
            field = torch.full((batch_size, 1), h, device=self.device)
            
        outputs = self.model(x, temp=temp, field=field)
        
        # MultiOutputTransformer now returns log probabilities [B, L, vocab_size] (like vit_rope)
        # utils_train expects log probabilities [B, L, vocab_size] and will strip the last dimension
        # (mask token) before sampling: logits = model(x)[:, :, :-1]
        # So we return full log probabilities [B, L, 3] (Cu=0, Au=1, Mask=2), and utils_train strips Mask.
        
        return outputs['scalars']

    @property
    def logits(self):
        # Some utils_train functions might access model.logits(x)
        return self.forward


class CuAuRewardWrapper:
    """
    Wrapper for AuCuAlloyModel to match utils_train reward_fn interface.
    
    Signature: reward_fn(x, beta=None, h=None, J=1, use_bias=True) -> log_reward [B]
    """
    def __init__(self, energy_model, vocab_map={0: 29, 1: 79}, default_temp_k=None):
        """
        Args:
            energy_model: AuCuAlloyModel instance
            vocab_map: Mapping from vocab indices to atomic numbers
            default_temp_k: Default temperature in Kelvin for single-temp case (when beta is None)
        """
        self.energy_model = energy_model
        self.vocab_map = vocab_map # Map 0->Cu(29), 1->Au(79)
        self.default_temp_k = default_temp_k  # Store default temp for single-temp case

    def __call__(self, x, beta=None, h=None, J=1, use_bias=True):
        # x is [B, L] indices (0, 1). MASK (2) should not remain in final samples.
        # AuCuAlloyModel expects {0, 1} inputs and maps internally.
        
        # Convert beta to temperature in Kelvin
        # beta = 1/(kB*T), so T = 1/(kB*beta)
        
        if beta is None:
            # Single temp/field case: utils_train doesn't pass beta
            # Use stored default temperature
            if self.default_temp_k is not None:
                temps = torch.full((x.shape[0],), self.default_temp_k, device=x.device, dtype=torch.float32)
            else:
                # Fallback: assume T=1K (shouldn't happen if default_temp_k is set)
                temps = torch.ones(x.shape[0], device=x.device, dtype=torch.float32)
        else:
            # Convert beta to temperature: T = 1/(kB*beta)
            if isinstance(beta, torch.Tensor):
                beta = beta.to(x.device)
                temps = 1.0 / (K_B * beta)
            else:
                temps = torch.full((x.shape[0],), 1.0 / (K_B * beta), device=x.device, dtype=torch.float32)
        
        # Handle field: h is in energy units (eV), use directly as fields
        if h is None:
            fields = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        else:
            if isinstance(h, torch.Tensor):
                fields = h.to(x.device)
                if fields.dim() == 0:
                    fields = fields.expand(x.shape[0])
            else:
                fields = torch.full((x.shape[0],), float(h), device=x.device, dtype=torch.float32)
        # Get free energies in units of kB*T (dimensionless)
        # get_free_energies returns (E - fields * lattices_factor) / (kB*T)
        # where E is formation energy (relative to pure phases), lower is better
        # For Boltzmann: P(x) ∝ exp(-beta * H) where H is the effective energy
        # In Ising: H = -J*sum(s_i*s_j) - h*sum(s_i), reward = -beta*H
        # For CuAu: H_eff = E - h*M, so log_reward = -beta*(E - h*M) = -(E - h*M)/(kB*T)
        free_energies = self.energy_model(x, temps, fields)  # [B] = (E - h*M)/(kB*T)
        log_reward = -free_energies

        return log_reward


def get_args():
    parser = argparse.ArgumentParser(description="Train CuAu Alloy Model")
    
    # System Config
    parser.add_argument("--size", type=int, nargs=3, default=[4, 4, 4], help="Supercell size (NxNxN)")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input .vasp file")
    parser.add_argument("--eci_file", type=str, default=None, help="Path to ECI JSON file")
    
    # Model Config
    parser.add_argument("--model_type", type=str, choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_embed", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    
    # Training Config (utils_train compatible)
    parser.add_argument("--loss_fn", type=str, default="wdce", choices=["ce", "lv", "wdce", "re_rf"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_steps", type=int, default=100000) # num_epochs in train
    parser.add_argument("--resample_every_n_step", type=int, default=10)
    parser.add_argument("--wdce_num_replicates", type=int, default=8, help="Replicates for WDCE")
    parser.add_argument("--grad_clip", action="store_true")
    parser.add_argument("--gradnorm_clip", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=20, help="Evaluate every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_ckpt", type=str, default=None)
    
    # Temps
    parser.add_argument("--temp_min", type=float, default=300.0)
    parser.add_argument("--temp_max", type=float, default=1000.0)
    # utils_train expects 'temps' arg for multi-temp training
    parser.add_argument("--num_temps", type=int, default=16) # for creating temp grid
    
    # Metadynamics / Bias
    parser.add_argument("--use_bias", action="store_true", help="Use metadynamics bias")
    parser.add_argument("--bias_method", type=str, default="binned", choices=["binned", "gaussian"])
    parser.add_argument("--bias_sigma", type=float, default=0.05)
    parser.add_argument("--bias_height", type=float, default=0.1)
    parser.add_argument("--bias_factor", type=float, default=10.0)
    parser.add_argument("--bias_grid_size", type=int, default=100)
    parser.add_argument("--cv_min", type=float, default=-1.0)
    parser.add_argument("--cv_max", type=float, default=1.0)
    parser.add_argument("--scale_bias_with_size", action="store_true")
    
    # Logging
    parser.add_argument("--out_dir", type=str, default="results/cuau")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="mdns-cuau")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    return parser.parse_args()


def setup_energy_model(args, device):
    """Setup the CuAu Energy Model."""
    if args.eci_file:
         with open(args.eci_file, "r", encoding="utf-8") as f:
            eci = json.load(f)
    else:
         if args.model_type != "dummy": # Keep quiet for tests if needed
            logging.warning("No ECI file provided. Using dummy/default initialization if possible.")
         eci = None

    try:
        conc = Concentration(basis_elements=[["Au", "Cu"]])
        settings = CEBulk(
            crystalstructure="fcc",
            a=3.8, 
            size=args.size,
            concentration=conc,
            db_name="aucu_dft.db",
            max_cluster_dia=[6.0, 4.5, 4.5],
        )
    except Exception as e:
        print(f"Warning: Could not create CEBulk settings: {e}")
        settings = None

    atoms = None
    if args.input_file:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        logging.info(f"Loading structure from {args.input_file}")
        atoms = ase.io.read(args.input_file)
    else:
        if settings is not None:
            atoms = settings.atoms.copy()
        else:
             atoms = ase.build.bulk("Cu", "fcc", a=3.8).repeat(tuple(args.size))

    model = AuCuAlloyModel(structure=atoms, settings=settings, eci=eci)
    return model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    wandb_run = None
    if args.wandb:
        wandb_run = wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name,
            config=args,
            dir=args.out_dir
        )
    
    # 1. Setup Energy Model
    energy_model = setup_energy_model(args, device)
    num_sites = energy_model.num_sites
    
    # 2. Setup Reward Function Wrapper
    # For single temp case, store the temperature so we can compute beta when it's None
    # For single temp: temp_min == temp_max, so use temp_min
    default_temp = args.temp_min if args.num_temps == 1 else None
    reward_fn = CuAuRewardWrapper(energy_model, default_temp_k=default_temp)

    # 3. Setup Neural Model
    # Determine reference positions for RoPE
    if hasattr(energy_model, "atoms"):
        ref_atoms = energy_model.atoms
    else:
        ref_atoms = getattr(energy_model, "atoms", None)
    
    fixed_positions = None
    if ref_atoms is not None:
        # Don't move to device here - let the model move it when .to(device) is called
        fixed_positions = torch.tensor(
            ref_atoms.get_scaled_positions(), dtype=torch.float32
        )
    # Use vocab_size = 3 (Cu=0, Au=1, Mask=2)
    vocab_size = 3
    num_scalars = vocab_size # Output logits for all 3 tokens

    if args.model_type == "transformer":
        base_net = MultiOutputTransformer(
            num_scalars=num_scalars,
            num_marginal=1, # Unused for WDCE training currently
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_embed=args.n_embed,
            max_src_len=num_sites,
            physical_dim=3,
            grid_shape=tuple(args.size),
            fixed_positions=fixed_positions,
            num_atom_types=vocab_size, # Embedding input size
        ).to(device)
        
        net = TransformerWrapper(base_net, vocab_size=vocab_size, length=num_sites, device=device)
    else:
        # MLP not yet adapted for wrapper
        raise NotImplementedError("MLP not yet adapted for new training loop")

    ema = ExponentialMovingAverage(net.parameters(), decay=0.9999) # utils_train uses EMA
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.00)

    # 4. Setup Bias Potential
    bias_potential = None
    if args.use_bias:
        D = num_sites
        energy_scaling_val = float(D) / 16.0 if args.scale_bias_with_size else 1.0
        T_val = args.temp_max # Reference T
        
        print(f"Initializing Bias: sigma={args.bias_sigma}, factor={args.bias_factor}")
        
        kernel_type = "delta" if args.bias_method == "binned" else "gaussian"
        bias_potential = BiasPotential(
            cv_min=args.cv_min,
            cv_max=args.cv_max,
            grid_size=args.bias_grid_size,
            sigma=args.bias_sigma,
            initial_height=args.bias_height,
            bias_factor=args.bias_factor,
            T=T_val,
            kernel_type=kernel_type,
            device=device,
            energy_scaling=energy_scaling_val
        )

    # 5. Load Checkpoint
    checkpoint = (
        torch.load(args.resume_from_ckpt, map_location=device)
        if args.resume_from_ckpt
        else None
    )
    if checkpoint:
        print(f"Loading checkpoint from {args.resume_from_ckpt}")
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("ema_state_dict"):
            ema.load_state_dict(checkpoint["ema_state_dict"])
        if bias_potential and checkpoint.get("bias_potential"):
            bias_potential.load_state_dict(checkpoint["bias_potential"])

    # 6. Prepare Temps/Fields for Training
    # utils_train expects args.temps as list or array if multi-temp
    # Let's generate a linear schedule of temperatures (as betas)
    # CuAu T range: temp_min to temp_max
    # beta = 1/(kB * T). Here we use T directly since reward wrapper maps it?
    # No, utils_train passes beta to model. 
    # Our TransformerWrapper converts beta -> temp = 1/beta.
    # So we should pass beta = 1/T.
    
    # But wait, reward_fn uses beta too: log_reward = -beta * E
    # Ideally, beta should be 1/(kB*T). 
    # For CuAu, units are eV typically. kB = 8.617e-5 eV/K.
    # So beta = 1 / (kB * T).
    # If we pass this beta to TransformerWrapper:
    #   temp = 1/beta = kB * T.  Transformer expects physical T?
    #   MultiOutputTransformer.thermo_embedder embeds 'temp'. 
    #   If it expects K, we need to pass T (in K).
    #   If we pass 1/(kB*T) as beta, then temp = kB*T (energy units).
    #   This might need scaling in TransformerWrapper or just train with energy units.
    #   Usually fine if consistent.
    
    # Set temps array for utils_train
    # For single temp case, utils_train won't use args.temps (use_multi_temp_field=False)
    # But we set it anyway for consistency. Note: get_temp_field_batch expects temps in Kelvin
    # and computes beta = 1/T. So we pass temps in Kelvin, not scaled by kB.
    temps_k = np.linspace(args.temp_min, args.temp_max, args.num_temps)
    args.temps = temps_k  # Pass temperatures in Kelvin (not scaled by kB)
    args.fields = np.zeros(1) # Can extend for chemical potential

    print("Starting training with utils_train.train...")
    
    # Create validation plotting callback
    def validation_plot_callback(x, temps, fields, wandb_run=None, step=None):
        """Callback function for plotting energy and Au concentration distributions during validation."""
        plot_energy_au_conc_distributions(
            x=x,
            energy_model=energy_model,
            temps=temps,
            fields=fields,
            save_path=os.path.join(args.out_dir, f"distributions_step_{step}.png") if step is not None else None,
            wandb_run=wandb_run,
            step=step,
            title_suffix=" (MDNS)",
        )
    
    utils_train.train(
        model=net,
        optimizer=optimizer,
        reward_fn=reward_fn,
        args=args,
        device=device,
        num_epochs=args.n_steps,
        ema=ema,
        wandb_run=wandb_run,
        bias_potential=bias_potential,
        save_dir=args.out_dir,
        cfg_dict=vars(args),
        validation_plot_callback=validation_plot_callback,
    )

if __name__ == "__main__":
    main()
