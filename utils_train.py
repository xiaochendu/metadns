import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from utils import ess, plot_bias_analysis, sample_categorical_logits
from utils_ising import ising2d_mag


def save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg, path, bias_potential=None):
    """Helper function to save model checkpoint."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'losses': losses,
        'ess_train': ess_train,
        'ess_eval': ess_eval,
        'bias_potential': bias_potential.state_dict() if bias_potential is not None else None,
        'cfg': cfg
    }, path)


def rnd(model, reward_model, batch_size, device='cuda:0', beta_batch=None, h_batch=None, J=1):
    r"""
    Run random order sampling and compute the RND $\log\frac{dP^*}{dP^u}$ along the trajectory
    
    Args:
        model: The model
        reward_model: Function that takes (x, beta, J, h) and returns rewards [B]
        batch_size: Batch size
        device: Device
        beta_batch: [B] tensor of beta values, or None for single beta (backward compatible)
        h_batch: [B] tensor of field values, or None for single field (backward compatible)
        J: Scalar interaction strength
        
    Returns:
        x: the final samples, [B, D]
        log_rnd: the log RND along this trajectory, [B]
    """
    if hasattr(model, 'module'):
        model = model.module
    
    x = torch.full((batch_size, model.length), model.vocab_size-1).to(device=device, dtype=torch.int64)
    batch_arange = torch.arange(batch_size, device=device)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    log_rnd = torch.zeros(batch_size, device=device) # [B]
    for d in range(model.length-1, -1, -1):
        # Pass beta and h to model if provided
        if beta_batch is not None or h_batch is not None:
            logits = model(x, beta=beta_batch, h=h_batch)[:, :, :-1] # [B, D, N-1]
        else:
            logits = model(x)[:, :, :-1] # [B, D, N-1] - backward compatible
        update = sample_categorical_logits(
            logits[batch_arange, jump_pos[:, d]]) # [B]
        if torch.is_grad_enabled(): # avoid issues with in-place operations
            x = x.clone()
        x[batch_arange, jump_pos[:, d]] = update
        log_rnd += -np.log(model.vocab_size-1) - logits[batch_arange, jump_pos[:, d], update]
    
    # Compute reward with per-sample temperatures/fields if provided
    if beta_batch is not None or h_batch is not None:
        # Ensure tensors are on correct device
        if beta_batch is not None and isinstance(beta_batch, torch.Tensor):
            beta_batch = beta_batch.to(device)
        if h_batch is not None and isinstance(h_batch, torch.Tensor):
            h_batch = h_batch.to(device)
        log_rnd += reward_model(x, beta=beta_batch, h=h_batch if h_batch is not None else 0, J=J)
    else:
        # Backward compatible: use default scalar values
        log_rnd += reward_model(x) # [B]
    
    return x, log_rnd


@torch.no_grad()
def sampling(model, batch_size, rounds=1, device='cuda:0'):
    """Any order autoregressive sampling"""
    if hasattr(model, 'module'):
        model = model.module
    batch_arange = torch.arange(batch_size, device=device)
    all_samples = []
    for _ in tqdm(range(rounds), leave=False):
        x = torch.full((batch_size, model.length), model.vocab_size-1).to(device=device, dtype=torch.int64)
        jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
        for d in tqdm(range(model.length-1, -1, -1), leave=False):
            logits = model.logits(x)[:, :, :-1] # [B, D, N-1], not log-softmaxed but fine
            update = sample_categorical_logits(
                logits[batch_arange, jump_pos[:, d]]) # [B]
            x[batch_arange, jump_pos[:, d]] = update
        all_samples.append(x)
    return torch.cat(all_samples)


def loss_ce(log_rnd):
    """Cross entropy loss KL(P^*||P^u)"""
    weights = log_rnd.detach().softmax(dim=-1)
    return (log_rnd * weights).sum()


def loss_lv(log_rnd):
    r"""Log variance loss Var_{P^\bar{u}}\log\frac{dP^*}{dP^u}"""
    return log_rnd.var()


def loss_re_rf(log_rnd, const=0):
    r"""Relative entropy loss KL(P^u||P^*) with REINFORCE trick"""
    reward_term = (-log_rnd.detach() + const)
    reward_mean = reward_term.mean()
    reward_var = reward_term.var()
    return (-log_rnd * (reward_term - reward_mean) / (reward_var + 1e-8)).mean()


def loss_wdce(model, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l, beta_batch=None, h_batch=None):
    r"""
    Weighted denoising cross entropy loss
    X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
    
    log_rnd: [B]; x: [B, D] (no mask)
    num_replicates: R, number of replicates of each row in x
    weight_func: w(lambda) for each sample, 1/lambda by default
    beta_batch: [B] tensor of beta values (optional, for conditioning)
    h_batch: [B] tensor of field values (optional, for conditioning)
    """
    if hasattr(model, 'module'):
        model = model.module
    
    batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, D]
    batch_weights = log_rnd.detach_().softmax(dim=-1).repeat_interleave(num_replicates, dim=0) # [B*R]
    lamda = torch.rand(batch.shape[0], device=batch.device) # [B*R]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B*R]
    masked_index = torch.rand(*batch.shape, device=batch.device) < lamda[..., None] # [B*R, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size-1, batch)
    
    # Handle beta/h conditioning - replicate them to match perturbed_batch
    if beta_batch is not None or h_batch is not None:
        beta_perturbed = beta_batch.repeat_interleave(num_replicates, dim=0) if beta_batch is not None else None
        h_perturbed = h_batch.repeat_interleave(num_replicates, dim=0) if h_batch is not None else None
        logits = model(perturbed_batch, beta=beta_perturbed, h=h_perturbed)
    else:
        logits = model(perturbed_batch)  # Backward compatible
    
    losses = torch.zeros(*batch.shape, device=batch.device, dtype=logits.dtype) # [B*R, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=batch[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights * batch_weights).mean()


def loss_dce(model, x, weight_func=lambda l: 1/l, beta_batch=None, h_batch=None):
    r"""
    Denoising cross entropy loss, x [B, D] are ground truth samples
    weight_func: w(lambda) for each sample, 1/lambda by default
    beta_batch: [B] tensor of beta values (optional, for conditioning)
    h_batch: [B] tensor of field values (optional, for conditioning)
    """
    lamda = torch.rand(x.shape[0], device=x.device) # [B]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B]
    masked_index = torch.rand(*x.shape, device=x.device) < lamda[..., None] # [B, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size-1, x)
    
    # Handle beta/h conditioning
    if beta_batch is not None or h_batch is not None:
        logits = model(perturbed_batch, beta=beta_batch, h=h_batch)
    else:
        logits = model(perturbed_batch)  # Backward compatible
    
    losses = torch.zeros(*x.shape, device=x.device, dtype=logits.dtype) # [B, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=x[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights).mean()


def log_validation_metrics(
    wandb_run,
    logp_x: torch.Tensor,
    logf_t: torch.Tensor,
    vfe: torch.Tensor,
    beta_batch: Optional[torch.Tensor],
    h_batch: Optional[torch.Tensor],
    step: int,
    log_kwargs: Optional[dict] = None,
) -> None:
    """Log validation metrics per condition and overall, similar to snowyflow.
    
    Args:
        wandb_run: Wandb run object for logging
        logp_x: Log probability of data under model, shape [B]
        logf_t: Log forward probability from MDNS, shape [B]
        vfe: Variational free energy (logp_x - logf_t), shape [B]
        beta_batch: Beta values [B] or None for single condition
        h_batch: Field values [B] or None for single condition
        step: Step/epoch number for logging
        log_kwargs: Additional keyword arguments to pass to wandb.log
    """
    if log_kwargs is None:
        log_kwargs = {}
    
    # Extract temperature and field batches (use defaults if None)
    batch_size = logf_t.shape[0]
    if beta_batch is not None:
        temp_batch = 1.0 / beta_batch.cpu().numpy()  # [B]
    else:
        temp_batch = np.zeros(batch_size)  # Default: single temp (will be 0.0)
    
    if h_batch is not None:
        field_batch = h_batch.cpu().numpy()  # [B]
    else:
        field_batch = np.zeros(batch_size)  # Default: single field (will be 0.0)
    
    # Get unique temperature/field combinations
    conditions = list(zip(temp_batch, field_batch))
    unique_conditions = sorted(set(conditions))
    
    # Separate dictionaries for main val panel and conditions panel
    val_log_data = {}  # Overall metrics for main val panel
    conditions_log_data = {}  # Per-condition metrics for separate panel
    
    # Group metrics by condition
    for temp_val, field_val in unique_conditions:
        condition_name = f"T{temp_val:04.0f}_F{field_val:+.3f}"
        
        # Find indices for this condition
        mask = np.array([(t == temp_val and f == field_val) for t, f in conditions])
        
        # Extract metrics for this condition
        logf_t_cond = logf_t[mask]
        logp_x_cond = logp_x[mask]
        vfe_cond = vfe[mask]
        
        # Log per-condition metrics to separate panel (val_conditions/)
        conditions_log_data[f"logf_t_mean/{condition_name}"] = logf_t_cond.mean().item()
        conditions_log_data[f"logf_t_std/{condition_name}"] = logf_t_cond.std().item()
        conditions_log_data[f"logp_x_mean/{condition_name}"] = logp_x_cond.mean().item()
        conditions_log_data[f"logp_x_std/{condition_name}"] = logp_x_cond.std().item()
        conditions_log_data[f"vfe_mean/{condition_name}"] = vfe_cond.mean().item()
        conditions_log_data[f"vfe_std/{condition_name}"] = vfe_cond.std().item()
    
    # Log temperature and field statistics to conditions panel (only if they vary)
    if beta_batch is not None:
        conditions_log_data["temp/min"] = float(temp_batch.min())
        conditions_log_data["temp/max"] = float(temp_batch.max())
        conditions_log_data["temp/mean"] = float(temp_batch.mean())
        conditions_log_data["temp/std"] = float(temp_batch.std())
    
    if h_batch is not None:
        conditions_log_data["field/min"] = float(field_batch.min())
        conditions_log_data["field/max"] = float(field_batch.max())
        conditions_log_data["field/mean"] = float(field_batch.mean())
        conditions_log_data["field/std"] = float(field_batch.std())
    
    # Add overall statistics to main val panel (always log these)
    val_log_data["logf_t_mean/overall"] = logf_t.mean().item()
    val_log_data["logf_t_std/overall"] = logf_t.std().item()
    val_log_data["logp_x_mean/overall"] = logp_x.mean().item()
    val_log_data["logp_x_std/overall"] = logp_x.std().item()
    val_log_data["vfe_mean/overall"] = vfe.mean().item()
    val_log_data["vfe_std/overall"] = vfe.std().item()
    
    # Prefix keys appropriately for separate panels
    val_log_data_prefixed = {f"val/{k}": v for k, v in val_log_data.items()}
    conditions_log_data_prefixed = {f"val_conditions/{k}": v for k, v in conditions_log_data.items()}
    
    # Log to separate panels
    wandb_run.log(val_log_data_prefixed, step=step, **log_kwargs)
    if conditions_log_data_prefixed:  # Only log if there are per-condition metrics
        wandb_run.log(conditions_log_data_prefixed, step=step, **log_kwargs)


def _compute_log_stats(x, log_rnd, reward_fn, model, beta_batch=None, h_batch=None, J=1,
                       bias_potential=None):
    """Compute logf_t and logp_x given samples and RND values.
    
    Args:
        x: [B, D] samples
        log_rnd: [B] log RND values
        reward_fn: Function that takes (x, beta, J, h) and returns rewards
        model: Model (for getting vocab_size)
        beta_batch: [B] tensor of beta values, or None for scalar beta
        h_batch: [B] tensor of field values, or None for scalar field
        J: Scalar interaction strength
    """
    # Compute logf_t using per-sample betas/fields if provided
    if beta_batch is not None or h_batch is not None:
        logf_t_vals = reward_fn(x, beta=beta_batch, h=h_batch if h_batch is not None else 0, J=J, use_bias=False)
    else:
        logf_t_vals = reward_fn(x, use_bias=False)  # Use default scalar values
    num_states = getattr(model, "vocab_size", 3) - 1  # exclude mask token
    data_dim = x.shape[1]
    uniform_prior_term = -torch.log(torch.tensor(float(num_states), device=x.device))
    uniform_prior_total = uniform_prior_term * data_dim
    logp_x_vals = uniform_prior_total + logf_t_vals - log_rnd
    
    # Correct logp_x vals if bias_potential is provided
    # logp_x_effective = logp_model(x) + beta * V(s)
    # This recovers the effective sampling probability with respect to the unbiased Hamiltonian
    # assuming logf_t_vals is unbiased (which it is, see use_bias=False above)
    if bias_potential is not None:
        with torch.no_grad():
            x_spins = 2 * x - 1
            s = ising2d_mag(x_spins)
            bias_vals = bias_potential.evaluate(s) # [B]
            
            # Apply beta correction
            if beta_batch is not None:
                logp_x_vals = logp_x_vals - beta_batch * bias_vals
            else:
                 # Assume standard T from bias potential or model
                 # Train Ising sets bias_potential.T = 1/beta
                 beta = 1.0 / bias_potential.T
                 logp_x_vals = logp_x_vals - beta * bias_vals
                 
    return logf_t_vals, logp_x_vals


def _visualize_lattices(samples, L, n_rows=2, n_cols=5, max_samples=16, 
                        beta_batch=None, h_batch=None):
    """Visualize Ising lattices in a grid.
    
    Args:
        samples: Tensor of shape [B, L*L] or [B, L, L]
        L: Lattice dimension
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        max_samples: Maximum number of samples to visualize
        beta_batch: [B] tensor of beta values (optional, for sampling from each temp)
        h_batch: [B] tensor of field values (optional, for sampling from each field)
    """
    # Reshape if needed: [B, L*L] -> [B, L, L]
    if samples.ndim == 2:
        B = samples.shape[0]
        samples = samples.view(B, L, L)
    
    # Convert to float and CPU for matplotlib
    samples = samples.float().cpu()
    
    # If beta_batch and h_batch are provided, sample from each unique temp/field combination
    if beta_batch is not None and h_batch is not None:
        # Convert beta to temperature for grouping
        temp_batch = 1.0 / beta_batch.cpu().numpy()
        h_batch_cpu = h_batch.cpu().numpy()
        
        # Find unique temp/field combinations
        # Round to avoid floating point precision issues
        temp_rounded = np.round(temp_batch, decimals=4)
        h_rounded = np.round(h_batch_cpu, decimals=4)
        
        # Create combination keys
        combinations = [(t, h) for t, h in zip(temp_rounded, h_rounded)]
        unique_combos = list(set(combinations))
        
        # Sample at least a few samples from each combination
        samples_to_plot_indices = []
        samples_per_combo = max(1, max_samples // max(len(unique_combos), 1))
        
        for combo in unique_combos:
            # Find indices matching this combination
            matching_indices = np.array([i for i, c in enumerate(combinations) if c == combo])
            # Sample up to samples_per_combo from this combination
            n_samples_from_combo = min(samples_per_combo, len(matching_indices))
            if n_samples_from_combo > 0:
                if len(matching_indices) == 1:
                    # Only one matching index, just use it
                    selected_indices = matching_indices
                else:
                    selected_indices = np.random.choice(
                        matching_indices, size=n_samples_from_combo, replace=False
                    )
                samples_to_plot_indices.extend(selected_indices.tolist())
        
        # Limit total number of samples
        n_plots = min(n_rows * n_cols, len(samples_to_plot_indices), max_samples)
        samples_to_plot_indices = samples_to_plot_indices[:n_plots]
        samples_to_plot = samples[samples_to_plot_indices]
    else:
        # Original behavior: just take first N samples
        n_plots = min(n_rows * n_cols, max_samples, samples.shape[0])
        samples_to_plot = samples[:n_plots]
    
    # Create colormap for binary Ising (blue for 0, pink for 1)
    palette = ["#1f77b4", "#e377c2"]
    cmap = ListedColormap(palette)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    axes = axes.flatten()
    
    for i in range(n_plots):
        ax = axes[i]
        sample_np = samples_to_plot[i].numpy()
        ax.imshow(sample_np, cmap=cmap, origin="lower", vmin=0, vmax=1)
        ax.axis("off")
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    return fig


def train(model, optimizer, reward_fn, args, device, num_epochs = 10000, ema=None,
          losses=None, ess_train=None, ess_eval=None, wandb_run=None, L=None, 
          bias_potential=None, current_fields=None, rng=None, save_dir=None, cfg_dict=None):
    loss_fn = {'ce': loss_ce, 'lv': loss_lv, 're_rf': loss_re_rf,
               'wdce': loss_wdce}.get(args.loss_fn)
    if loss_fn is None:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    # continue recording the metrics from the last training
    losses = [] if losses is None else losses.copy()
    ess_train = [] if ess_train is None else ess_train.copy()
    ess_eval = [] if ess_eval is None else ess_eval.copy()
    pbar = tqdm(range(num_epochs))
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    # Handle multiple temperatures/fields
    from utils_ising import get_temp_field_batch, sample_temp_field
    use_multi_temp_field = hasattr(args, 'temps') and args.temps is not None and (
        len(args.temps) > 1 or (hasattr(args, 'fields') and args.fields is not None and len(args.fields) > 1)
    )
    
    # Initialize temps/fields for dynamic sampling
    # Store starting temps/fields for validation (fixed, not dynamically sampled)
    starting_temps = None
    starting_fields = None
    if use_multi_temp_field:
        if current_temps is None:
             current_temps = np.array(args.temps)
        if current_fields is None:
             current_fields = np.array(args.fields) if hasattr(args, 'fields') and args.fields is not None else np.array([0.0])
        # Store starting values for validation
        starting_temps = current_temps.copy()
        starting_fields = current_fields.copy()
        if rng is None:
            rng = np.random.default_rng(seed=args.seed if args.seed is not None else 42)
    else:
        current_temps = None
        # current_fields = None # Already passed or initialized
        if rng is None:
             rng = None

    x_saved, log_rnd_saved = None, None
    
    for epoch in pbar:
        model.train(); optimizer.zero_grad(); info = {}

        # Generate temperature/field batches if using multiple temps/fields
        beta_batch = None
        h_batch = None
        log_dict_temp_field = {}  # Initialize for logging
        if use_multi_temp_field:
            # Handle dynamic sampling if enabled
            if hasattr(args, 'sample_delta_temp') and args.sample_delta_temp:
                # Use provided min/max temps if available, otherwise compute from current temps
                if hasattr(args, 'min_temp') and args.min_temp is not None:
                    min_temp = args.min_temp
                else:
                    min_temp = current_temps.min() - args.delta_temp
                if hasattr(args, 'max_temp') and args.max_temp is not None:
                    max_temp = args.max_temp
                else:
                    max_temp = current_temps.max() + args.delta_temp
                current_temps = sample_temp_field(
                    current_temps, args.delta_temp, min_temp, max_temp, rng
                )
            if hasattr(args, 'sample_delta_field') and args.sample_delta_field:
                min_field = current_fields.min() - args.delta_field
                max_field = current_fields.max() + args.delta_field
                current_fields = sample_temp_field(
                    current_fields, args.delta_field, min_field, max_field, rng
                )
            
            # Generate batch of temps/fields
            beta_batch, h_batch, _, _, _ = get_temp_field_batch(
                current_temps, current_fields, args.batch_size
            )
            beta_batch = beta_batch.to(device)
            h_batch = h_batch.to(device)
            
            # Log temperature and field statistics
            if beta_batch is not None:
                # Convert beta back to temperature for logging (beta = 1/T, so T = 1/beta)
                temp_batch = 1.0 / beta_batch.cpu().numpy()
                log_dict_temp_field = {
                    "train/temp_min": float(temp_batch.min()),
                    "train/temp_max": float(temp_batch.max()),
                    "train/temp_mean": float(temp_batch.mean()),
                    "train/temp_std": float(temp_batch.std()),
                    "train/field_min": float(h_batch.cpu().min().item()),
                    "train/field_max": float(h_batch.cpu().max().item()),
                    "train/field_mean": float(h_batch.cpu().mean().item()),
                    "train/field_std": float(h_batch.cpu().std().item()),
                }

        if args.loss_fn == 'wdce':
            with torch.no_grad():
                if x_saved is None or epoch % args.resample_every_n_step == 0:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device,
                                     beta_batch=beta_batch, h_batch=h_batch, J=args.J if hasattr(args, 'J') else 1)
                    ema.restore(model.parameters())
                    x_saved, log_rnd_saved = x, log_rnd
                    is_fresh_sample = True
                else:
                    x, log_rnd = x_saved, log_rnd_saved
                    is_fresh_sample = False

            # Note: train_ddp doesn't have multi-temp/field setup, so pass None
            loss = loss_wdce(model, log_rnd, x,
                                num_replicates=args.wdce_num_replicates,
                                beta_batch=None, h_batch=None)
        else:
            x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device,
                             beta_batch=beta_batch, h_batch=h_batch, J=args.J if hasattr(args, 'J') else 1)
            loss = loss_fn(log_rnd)
            is_fresh_sample = True

        # Update bias after sampling (on-policy update)
        # CRITICAL FIX: Only update bias if we actually generated new samples!
        # Otherwise we build a huge wall at the same spot for N steps (sloshing).
        if bias_potential is not None and is_fresh_sample:
            # Calculate CV (Magnetization)
            with torch.no_grad():
                # x is [B, D] in {0, 1} usually? 
                # utils_ising functions usually expect {-1, 1} but handle it?
                # train_ising.py reward_fn converts 0/1 to -1/1.
                # ising2d_mag inside utils_ising expects {-1, 1}
                # rnd returns x in {0..(vocab-1)}. For Ising vocab=2 (0, 1).
                # So we must convert to spins for ising2d_mag: 2*x - 1
                x_spins = 2 * x - 1
                s = ising2d_mag(x_spins)
                bias_potential.update(s)

        # Synchronize loss across processes
        
        logf_t_vals, logp_x_vals = _compute_log_stats(x, log_rnd, reward_fn, model,
                                                       beta_batch=beta_batch, h_batch=h_batch,
                                                       J=args.J if hasattr(args, 'J') else 1,
                                                       bias_potential=bias_potential)
        vfe = logp_x_vals - logf_t_vals  # variational free energy
        ess_train.append(ess(log_rnd))
        info['ess_train'] = ess_train[-1]
        info['loss'] = loss.item()
        losses.append(loss.item())

        if wandb_run is not None:
            log_dict = {
                "train/loss": loss.item(),
                "train/avg_free_energy": vfe.mean().item(),
                "train/std_free_energy": vfe.std().item(),
                "train/avg_logp_x": logp_x_vals.mean().item(),
                "train/std_logp_x": logp_x_vals.std().item(),
                "train/avg_logf_t": logf_t_vals.mean().item(),
                "train/std_logf_t": logf_t_vals.std().item(),
            }
            
            # Log bias stats
            if bias_potential is not None:
                # We can't plot the whole grid every step easily, just stats
                # Using get_bias_grid_np() if available or accessing internal
                if hasattr(bias_potential, 'bias_grid'):
                    bias_grid = bias_potential.bias_grid.detach().cpu().numpy()
                    log_dict["bias/max_height"] = bias_grid.max()
                    log_dict["bias/mean_height"] = bias_grid.mean()
                    # Fraction of visited states (nonzero bias)
                    log_dict["bias/coverage"] = (bias_grid > 1e-6).mean()
                    
                    # Plot bias analysis every 100 epochs (REMOVED: Moved to validation)
                    # if epoch % 100 == 0:
                    #     fig_bias = plot_bias_analysis(bias_potential, epoch, s_batch=s)
                    #     if fig_bias is not None:
                    #         log_dict["bias/analysis_plot"] = wandb.Image(fig_bias)
                    #         plt.close(fig_bias)
            
            # Add temperature and field statistics if available
            if log_dict_temp_field:
                log_dict.update(log_dict_temp_field)
            
            # Log lattice visualization periodically (every 100 epochs)
            if L is not None and epoch % 100 == 0:
                try:
                    fig = _visualize_lattices(x, L, n_rows=2, n_cols=5, max_samples=10,
                                              beta_batch=beta_batch, h_batch=h_batch)
                    log_dict["train/samples"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    # Silently skip visualization if there's an error
                    pass
            
            wandb_run.log(log_dict, step=epoch)
        
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradnorm_clip)
        optimizer.step()
        if ema is not None: ema.update(model.parameters())
        
        pbar.set_postfix(loss=info['loss'], ess=info['ess_train'])

        # Save checkpoint periodically
        if save_dir is not None and getattr(args, 'save_every', 0) > 0 and (epoch + 1) % args.save_every == 0:
            save_path = f"{save_dir}/ckpt_{epoch+1}.pth"
            save_checkpoint(model, optimizer, ema, losses, ess_train, ess_eval, cfg_dict, save_path, bias_potential)
        
        # Evaluate periodically
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                # Generate eval temp/field batches if using multiple temps/fields
                # Use starting temps/fields (fixed) for validation, not dynamically sampled ones
                eval_beta_batch = None
                eval_h_batch = None
                if use_multi_temp_field:
                    eval_beta_batch, eval_h_batch, _, _, _ = get_temp_field_batch(
                        starting_temps, starting_fields, args.eval_batch_size
                    )
                    eval_beta_batch = eval_beta_batch.to(device)
                    eval_h_batch = eval_h_batch.to(device)
                
                x, log_rnd = rnd(model, reward_fn, args.eval_batch_size, device=device,
                                 beta_batch=eval_beta_batch, h_batch=eval_h_batch, 
                                 J=args.J if hasattr(args, 'J') else 1)
                eval_ess = ess(log_rnd)
                ess_eval.append(eval_ess)
                if wandb_run is not None:
                    logf_t_vals, logp_x_vals = _compute_log_stats(x, log_rnd, reward_fn, model,
                                                                   beta_batch=eval_beta_batch, h_batch=eval_h_batch,
                                                                   J=args.J if hasattr(args, 'J') else 1,
                                                                   bias_potential=bias_potential)
                    vfe = logp_x_vals - logf_t_vals  # variational free energy
                    
                    # Use the new per-condition logging function (similar to snowyflow)
                    log_validation_metrics(
                        wandb_run=wandb_run,
                        logp_x=logp_x_vals,
                        logf_t=logf_t_vals,
                        vfe=vfe,
                        beta_batch=eval_beta_batch,
                        h_batch=eval_h_batch,
                        step=epoch,
                    )
                    
                    # Also log ESS (not included in per-condition metrics)
                    wandb_run.log({"val/ess": eval_ess}, step=epoch)
                    
                    # Log lattice visualization during evaluation
                    if L is not None:
                        try:
                            fig = _visualize_lattices(x, L, n_rows=2, n_cols=5, max_samples=10,
                                                      beta_batch=eval_beta_batch, h_batch=eval_h_batch)
                            wandb_run.log({"val/samples": wandb.Image(fig)}, step=epoch)
                            plt.close(fig)
                        except Exception as e:
                            # Silently skip visualization if there's an error
                            pass

                    # Plot bias analysis during validation (uses eval_batch_size)
                    if bias_potential is not None:
                        try:
                            # Convert x(0,1) to spins(-1,1) for CV
                            x_spins_eval = 2 * x - 1
                            s_eval = ising2d_mag(x_spins_eval)
                            fig_bias = plot_bias_analysis(bias_potential, epoch, s_batch=s_eval)
                            if fig_bias is not None:
                                wandb_run.log({"val/bias_analysis_plot": wandb.Image(fig_bias)}, step=epoch)
                                plt.close(fig_bias)
                        except Exception as e:
                            print(f"Error plotting bias analysis during val: {e}")
            model.train()
            
    return model, optimizer, ema, losses, ess_train, ess_eval


def train_ddp(model, optimizer, reward_fn, args, device, num_epochs = 10000, ema=None, scheduler=None,
          losses=None, ess_train=None, ess_eval=None):
    loss_fn = {'ce': loss_ce, 'lv': loss_lv, 're_rf': loss_re_rf,
               'wdce': loss_wdce}.get(args.loss_fn)
    if loss_fn is None:
        raise ValueError(f"Unknown loss function: {args.loss_fn}")

    # continue recording the metrics from the last training
    losses = [] if losses is None else losses.copy()
    ess_train = [] if ess_train is None else ess_train.copy()
    ess_eval = [] if ess_eval is None else ess_eval.copy()
    
    # Only show progress bar on rank 0
    if dist.get_rank() == 0:
        pbar = tqdm(range(num_epochs))
    else:
        pbar = range(num_epochs)
        
    if args.seed is not None:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    x_saved, log_rnd_saved = None, None
    
    for epoch in pbar:
        model.train(); optimizer.zero_grad(); info = {}

        if args.loss_fn == 'wdce':
            with torch.no_grad():
                if x_saved is None or epoch % args.resample_every_n_step == 0:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    # Note: train_ddp doesn't have multi-temp/field setup, so beta_batch/h_batch will be None
                    x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device)
                    ema.restore(model.parameters())
                    x_saved, log_rnd_saved = x, log_rnd
                else:
                    x, log_rnd = x_saved, log_rnd_saved

            # beta_batch and h_batch will be None in train_ddp (no multi-temp/field support yet)
            loss = loss_wdce(model, log_rnd, x,
                                num_replicates=args.wdce_num_replicates,
                                beta_batch=None, h_batch=None)
        else:
            x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device)
            loss = loss_fn(log_rnd)
                
        # Synchronize loss across processes
        dist.all_reduce(loss)
        
        # to average over all processes and accommodate for the reduced batch size
        loss = loss / dist.get_world_size() / dist.get_world_size()
        
        ess_train.append(ess(log_rnd))
        info['ess_train'] = ess_train[-1]
        info['loss'] = loss.item()
        losses.append(loss.item())
        
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradnorm_clip)
        optimizer.step()
        if scheduler is not None: scheduler.step()
        if ema is not None: ema.update(model.parameters())
        
        # Update progress bar only on rank 0
        if dist.get_rank() == 0:
            pbar.set_postfix(loss=info['loss'], ess=info['ess_train'])
        
        # Evaluate periodically
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x, log_rnd = rnd(model, reward_fn, args.eval_batch_size, device=device)
                eval_ess = ess(log_rnd)
                # Synchronize evaluation metrics across processes
                ess_eval.append(eval_ess)
            model.train()
            
    return model, optimizer, ema, losses, ess_train, ess_eval