from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

# For KDE in distribution plots
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# For wandb logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_periodic_table_colormap() -> LinearSegmentedColormap:
    """Create a custom colormap matching the periodic table color scheme.

    Returns a colormap that transitions from dark blue (low values)
    to light yellow/green (high values), matching the periodic table visualization style.
    Higher values appear lighter. This colormap is colorblind-friendly as it avoids
    red-green combinations.

    Returns:
        LinearSegmentedColormap: A custom colormap for heat plots
    """
    # Define colors matching the periodic table palette
    # Reversed: Dark blue -> medium blue -> cyan -> green -> light yellow/green
    # Higher values are now lighter (reversed from original)
    # RGB values chosen to match the periodic table visualization
    colors = [
        (0.03, 0.08, 0.45),  # Dark blue (low values, ~1-10)
        (0.08, 0.25, 0.65),  # Medium blue (~10-100)
        (0.15, 0.50, 0.75),  # Cyan-blue (~100)
        (0.35, 0.70, 0.65),  # Green-cyan (~100-1K)
        (0.60, 0.85, 0.50),  # Medium green (~1K)
        (0.85, 0.95, 0.55),  # Yellow-green (~1K-10K)
        (0.98, 0.98, 0.65),  # Light yellow/green (high values, ~10K)
    ]
    n_bins = 256
    return LinearSegmentedColormap.from_list("periodic_table", colors, N=n_bins)


def sample_categorical(categorical_probs, dtype=torch.float64):
    # do not require probs to be normalized
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs, dtype=dtype) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def sample_categorical_logits(logits, dtype=torch.float64):
    # do not require logits to be log-softmaxed
    gumbel_noise = -(1e-10 - (torch.rand_like(logits, dtype=dtype) + 1e-10).log()).log()
    return (logits + gumbel_noise).argmax(dim=-1)


def ess(log_rnd, normalize=True): 
    """
    log_rnd: [B]
    Compute effective sample size:
        If normalize: divide ESS by batch size, so range is [0, 1]; 
        otherwise, range is [0, B]
    """
    weights = log_rnd.detach().softmax(dim=-1)
    ess = 1 / (weights ** 2).sum().item()
    return ess / log_rnd.shape[0] if normalize else ess


def metric(pmf1, pmf2, method='kl'):
    """[B], [B] -> float"""
    if method == 'tv':
        return .5 * (pmf1 - pmf2).abs().sum().item()
    elif method == 'kl':
        return (pmf1 * (pmf1.log() - pmf2.log()))[pmf1>0].sum().item()
    elif method == 'chi2':
        return ((pmf1 - pmf2) ** 2 / pmf2)[pmf2 > 0].sum().item()
    else:
        raise ValueError(f"Unknown metric: {method}")


def plot_loss_ess(losses, ess, ess_eval=None):
    """
    Plot the loss and ESS over training steps.
    Here the ESS is normalized by the batch size so takes values between 0 and 1.
    """
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    ax[0].plot(losses)
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Steps')
    ax[0].grid()
    ax[1].plot(ess, label='Original', alpha=.75)
    if ess_eval is not None:
        ax[1].plot(ess_eval, label='EMA', alpha=.75)
    ax[1].set_title('ESS / batch_size')
    ax[1].set_xlabel('Steps')
    ax[1].set_ylim(0, 1)
    ax[1].legend()
    ax[1].grid()
    plt.tight_layout()
    plt.show()
    return fig, ax # fig.savefig(...)


class Dict2Obj:
    def __init__(self, dic=None):
        if dic is not None: 
            for key, value in dic.items():
                if isinstance(value, dict):
                    value = Dict2Obj(value)
                setattr(self, key, value)
    def __repr__(self):
        return str(self.__dict__)


def cycleloader(dataloader):
    while True:
        for data in dataloader:
            yield data


def plot_bias_analysis(bias_potential, epoch, s_batch=None, biased_reward=None):
    """
    Plots bias analysis:
    1. Estimated Free Energy F(s) (from Bias Potential)
    2. Raw Histogram P(s) (from batch) - Checks sampling uniformity
    3. Reweighted Histogram P_corr(s) (Likely physical F(s))
    4. Log-Ratio (log_rnd) vs CV (Optional) - Checks convergence
    
    s_batch: Tensor/Array of CV values from current batch
    log_rnd: Tensor/Array of log-RND values (log R_biased - log P_model)
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 1. Estimated Free Energy from Bias
        # F(s) ~ - (gamma / (gamma - 1)) * V(s)
        grid_vals, bias_vals = bias_potential.get_bias_grid_np()
        gamma = bias_potential.gamma
        if gamma > 1.0:
            free_energy_profile = - (gamma / (gamma - 1)) * bias_vals
        else:
            free_energy_profile = - bias_vals # Fallback or standard metadynamics
            
        # Shift to zero minimum for relative interpretation
        # NOTE: Don't shift to better watch convergence
        # free_energy_profile = free_energy_profile - free_energy_profile.min()
            
        axes[0].plot(grid_vals, free_energy_profile, label='F(s) Estimate', color='blue')
        axes[0].set_title(f'Est. Relative Free Energy (Ep {epoch})')
        axes[0].set_xlabel('CV (Magnetization)')
        axes[0].set_ylabel('Energy')
        axes[0].grid(True)
        axes[0].legend()

        # We will need a lot of samples to get a good estimate

        # 2. Raw Distribution (Histogram of s)
        # Should be effectively flat if converged
        if s_batch is not None:
            if isinstance(s_batch, torch.Tensor):
                s_np = s_batch.detach().cpu().numpy()
            else:
                s_np = s_batch
                
            # Histogram
            axes[1].hist(s_np, bins=bias_potential.grid_size, range=(bias_potential.cv_min, bias_potential.cv_max), density=True, alpha=0.6, color='green', label='Sampled')
            axes[1].set_title('Raw Distribution P(s) (Should be Flat)')
            axes[1].set_xlabel('CV')
            axes[1].set_ylabel('Density')
            axes[1].grid(True)
            
            # 3. Corrected Distribution (Reweighted)
            # P_unbiased(s) ~ P_biased(s) * exp(beta * V(s))
            # weight = exp( bias_potential.evaluate(s) / T )
            # We compute weights for the batch
            # Note: bias_potential.evaluate expects tensor
            with torch.no_grad():
                if not isinstance(s_batch, torch.Tensor):
                     s_tens = torch.tensor(s_batch, device=bias_potential.device)
                else:
                     s_tens = s_batch.to(bias_potential.device)
                     
                v_s = bias_potential.evaluate(s_tens) # [B]
                # beta = 1/T
                beta = 1.0 / bias_potential.T
                log_weights = beta * v_s
                # Shift for numerical stability
                log_weights = log_weights - log_weights.max()
                weights = torch.exp(log_weights).cpu().numpy()
            
            axes[2].hist(s_np, bins=bias_potential.grid_size, range=(bias_potential.cv_min, bias_potential.cv_max), density=True, weights=weights, alpha=0.6, color='red', label='Reweighted')
            axes[2].set_title('Corrected Distribution P(s) (Physical)')
            axes[2].set_xlabel('CV')
            axes[2].set_ylabel('Density')
            axes[2].set_ylabel('Density')
            axes[2].grid(True)
            
            # 4. Biased Reward vs CV
            # This is the target landscape the model is trying to learn
            # Ideally should be flat(ter) than the original energy landscape
            if biased_reward is not None:
                if isinstance(biased_reward, torch.Tensor):
                    r_np = biased_reward.detach().cpu().numpy()
                else:
                    r_np = biased_reward
                    
                axes[3].scatter(s_np, r_np, alpha=0.3, s=5, label='Biased Reward')
                axes[3].set_title('Biased Reward vs CV (Target Landscape)')
                axes[3].set_xlabel('CV')
                axes[3].set_ylabel('Energy (-beta*H - beta*V)')
                axes[3].grid(True)
            else:
                axes[3].axis('off')

        else:
             # Hide unused axes if s_batch missing
             axes[1].axis('off')
             axes[2].axis('off')
             axes[3].axis('off')

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error plotting bias analysis: {e}")
        return None


def plot_energy_au_conc_distributions(
    x: torch.Tensor,
    energy_model,
    temps: torch.Tensor,
    fields: torch.Tensor,
    save_path: str = None,
    wandb_run=None,
    step: int = None,
    title_suffix: str = "",
):
    """Plot Energy and Au concentration distributions for validation samples.
    
    Args:
        x: Samples [B, L] with values in {0, 1}
        energy_model: Energy model instance with get_energy() and get_concentrations() methods
        temps: Temperature tensor [B] in Kelvin
        fields: Field tensor [B] in eV
        save_path: Optional path to save the figure
        wandb_run: Optional wandb run object for logging
        step: Optional step number for wandb logging
        title_suffix: Optional suffix for plot title
    """
    device = x.device
    x_np = x.detach().cpu().numpy()
    temps_np = temps.detach().cpu().numpy() if isinstance(temps, torch.Tensor) else np.array([temps])
    fields_np = fields.detach().cpu().numpy() if isinstance(fields, torch.Tensor) else np.array([fields])
    
    # Get unique (T, μ) combinations for single plot per condition
    # For single temp/field case, we'll use the first values
    T_val = float(temps_np[0]) if len(temps_np) > 0 else 0.0
    mu_val = float(fields_np[0]) if len(fields_np) > 0 else 0.0
    
    # Compute energies and Au concentrations
    with torch.no_grad():
        energies = energy_model.get_energy(x)  # [B] in eV
        au_concentrations = energy_model.get_concentrations(x)  # [B] in [0, 1]
    
    energies_np = energies.detach().cpu().numpy()
    au_conc_np = au_concentrations.detach().cpu().numpy()
    
    # Get number of sites for binning
    num_sites = x.shape[1]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Energy Distribution
    ax1.hist(energies_np, bins=num_sites, density=True, alpha=0.6, color='lightblue', edgecolor='black', linewidth=0.5)
    
    # Add KDE
    if len(energies_np) > 1 and HAS_SCIPY:
        try:
            kde_energy = gaussian_kde(energies_np)
            energy_range = np.linspace(energies_np.min(), energies_np.max(), 200)
            kde_vals = kde_energy(energy_range)
            ax1.plot(energy_range, kde_vals, 'b--', linewidth=2, label='KDE')
        except:
            pass  # Skip KDE if it fails
    
    ax1.set_xlabel('Interaction Energy (eV)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'T = {T_val:.1f} K, μ = {mu_val:.3f} eV{title_suffix}', fontsize=11)
    ax1.grid(True, alpha=0.3)
    if HAS_SCIPY:
        ax1.legend()
    
    # Plot 2: Au Concentration Distribution
    ax2.hist(au_conc_np, bins=num_sites, density=True, alpha=0.6, color='lightgreen', edgecolor='black', linewidth=0.5, range=(0.0, 1.0))
    
    # Add KDE
    if len(au_conc_np) > 1 and HAS_SCIPY:
        try:
            kde_conc = gaussian_kde(au_conc_np)
            conc_range = np.linspace(0.0, 1.0, 200)
            kde_vals = kde_conc(conc_range)
            ax2.plot(conc_range, kde_vals, 'g--', linewidth=2, label='KDE')
        except:
            pass  # Skip KDE if it fails
    
    ax2.set_xlabel('Au Concentration', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'T = {T_val:.1f} K, μ = {mu_val:.3f} eV{title_suffix}', fontsize=11)
    ax2.set_xlim(0.0, 1.0)
    ax2.grid(True, alpha=0.3)
    if HAS_SCIPY:
        ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to wandb
    if wandb_run is not None and step is not None and HAS_WANDB:
        wandb_run.log({f"val/distributions{title_suffix}": wandb.Image(fig)}, step=step)
    
    plt.close(fig)