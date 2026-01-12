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

# Import kB for temperature conversion
try:
    import ase.units
    K_B = ase.units.kB  # eV/K
except ImportError:
    K_B = 8.617333262e-5  # eV/K (fallback)



class ReplayBuffer:
    """Experience Replay Buffer for MDNS.
    
    Stores samples (x, beta, h) and allows sampling mixed batches.
    Designed to be stored on CPU to save VRAM, with on-demand move to GPU.
    """
    def __init__(self, buffer_size, x_shape, device='cpu', dtype=torch.float32):
        self.buffer_size = buffer_size
        self.device = device  # Device where samples are returned (usually GPU)
        self.storage_device = 'cpu' # Store on CPU to save VRAM
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate tensors on CPU
        # x_shape is (D,)
        self.x = torch.zeros((buffer_size, *x_shape), dtype=torch.long, device=self.storage_device)
        self.beta = torch.zeros(buffer_size, dtype=dtype, device=self.storage_device)
        self.h = torch.zeros(buffer_size, dtype=dtype, device=self.storage_device)
        self.has_conditions = False # Track if we are actually storing conditions

    def add(self, x, beta=None, h=None):
        """Add a batch of samples to the buffer."""
        # Clean inputs and move to storage device
        x = x.to(self.storage_device)
        batch_size = x.shape[0]
        
        if batch_size > self.buffer_size:
            # If batch to add is larger than buffer, take last part
            x = x[-self.buffer_size:]
            if beta is not None: beta = beta[-self.buffer_size:]
            if h is not None: h = h[-self.buffer_size:]
            batch_size = x.shape[0]
            
        # Indices for circular buffer
        indices = torch.arange(self.ptr, self.ptr + batch_size) % self.buffer_size
        
        self.x[indices] = x
        
        if beta is not None:
            self.has_conditions = True
            if isinstance(beta, torch.Tensor):
                beta_in = beta.to(self.storage_device)
                if beta_in.ndim == 0:
                    beta_in = beta_in.expand(batch_size)
                self.beta[indices] = beta_in
            else:
                self.beta[indices] = torch.full((batch_size,), beta, device=self.storage_device)
                
        if h is not None:
            self.has_conditions = True
            if isinstance(h, torch.Tensor):
                h_in = h.to(self.storage_device)
                if h_in.ndim == 0:
                    h_in = h_in.expand(batch_size)
                self.h[indices] = h_in
            else:
                self.h[indices] = torch.full((batch_size,), h, device=self.storage_device)
        
        self.ptr = (self.ptr + batch_size) % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size):
        """Sample a batch from the buffer."""
        if self.size == 0:
            return None, None, None
            
        indices = torch.randint(0, self.size, (batch_size,))
        
        x_out = self.x[indices].to(self.device)
        
        if self.has_conditions:
            beta_out = self.beta[indices].to(self.device)
            h_out = self.h[indices].to(self.device)
        else:
            beta_out = None
            h_out = None
            
        return x_out, beta_out, h_out


def compute_model_log_prob(model, x, beta=None, h=None):
    """Compute the model component of log_rnd for samples x.
    
    This function iteratively computes the accumulation term:
       sum_{t} [ -log(V-1) - log P(x_t | x_{<t}) ]
    which matches the calculation in the rnd() sampling function.
    
    Args:
        model: Autoregressive model
        x: [B, D] input samples (fully observed)
        beta: [B] or scalar
        h: [B] or scalar
        
    Returns:
        log_rnd_model_term: [B] The accumulated model term of log_rnd.
           To get full log_rnd, add reward_fn(x): 
           log_rnd = log_rnd_model_term + reward_fn(x)
    """
    if hasattr(model, 'module'):
        model = model.module
        
    batch_size = x.shape[0]
    device = x.device
    
    # Initialize mock 'x' with masks to simulate generation
    # We use the actual token values from input x but revealed iteratively
    x_curr = torch.full((batch_size, model.length), model.vocab_size-1, device=device, dtype=torch.int64)
    
    # Random order for each sample (same as rnd)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    
    batch_arange = torch.arange(batch_size, device=device)
    log_rnd_term = torch.zeros(batch_size, device=device)
    
    for d in range(model.length-1, -1, -1):
        # Forward pass on current partially masked sequence
        if beta is not None or h is not None:
            logits = model(x_curr, beta=beta, h=h)[:, :, :-1]
        else:
            logits = model(x_curr)[:, :, :-1]
            
        # Check if raw logits or log probs
        if (logits > 0).any(): 
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            log_probs = logits

        # Identify which token is being "revealed" at this step
        # In rnd(), 'update' is sampled. Here, 'update' is the true token from x.
        update_pos = jump_pos[:, d] # [B]
        update_val = x[batch_arange, update_pos] # [B]
        
        # Accumulate: -log(V-1) - log P(x_t | ...)
        # Note: model.vocab_size usually includes mask, so vocab_size-1 is number of real tokens.
        log_rnd_term += -np.log(model.vocab_size-1) - log_probs[batch_arange, update_pos, update_val]
        
        # Reveal the token in x_curr for next step
        x_curr[batch_arange, update_pos] = update_val
        
    return log_rnd_term


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
        
        # Both vit_rope (Ising) and MultiOutputTransformer (CuAu) now return log probabilities
        # For backward compatibility with old models that might return raw logits, check and convert if needed
        if (logits > 0).any():
            # Raw logits detected (old model format) - apply log_softmax to get log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # For sampling, we can use either raw logits or log_probs with Gumbel-max
            # Using raw logits is more standard, but log_probs also works (argmax is shift-invariant)
            logits_for_sampling = logits
        else:
            # Already log probabilities (expected for modern models: vit_rope and MultiOutputTransformer)
            log_probs = logits
            logits_for_sampling = logits
        
        update = sample_categorical_logits(
            logits_for_sampling[batch_arange, jump_pos[:, d]]) # [B]
        if torch.is_grad_enabled(): # avoid issues with in-place operations
            x = x.clone()
        x[batch_arange, jump_pos[:, d]] = update
        log_rnd += -np.log(model.vocab_size-1) - log_probs[batch_arange, jump_pos[:, d], update]
    
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
    log_rnd: Optional[torch.Tensor] = None,
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
        log_rnd: Log RND values, shape [B] (optional)
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
        
        # Log log_rnd per condition if provided
        if log_rnd is not None:
            log_rnd_cond = log_rnd[mask]
            conditions_log_data[f"log_rnd_mean/{condition_name}"] = log_rnd_cond.mean().item()
            conditions_log_data[f"log_rnd_std/{condition_name}"] = log_rnd_cond.std().item()
            conditions_log_data[f"log_rnd_min/{condition_name}"] = log_rnd_cond.min().item()
            conditions_log_data[f"log_rnd_max/{condition_name}"] = log_rnd_cond.max().item()
    
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
    
    # Log log_rnd overall statistics if provided
    if log_rnd is not None:
        val_log_data["log_rnd_mean/overall"] = log_rnd.mean().item()
        val_log_data["log_rnd_std/overall"] = log_rnd.std().item()
        val_log_data["log_rnd_min/overall"] = log_rnd.min().item()
        val_log_data["log_rnd_max/overall"] = log_rnd.max().item()
    
    # Prefix keys appropriately for separate panels
    val_log_data_prefixed = {f"val/{k}": v for k, v in val_log_data.items()}
    conditions_log_data_prefixed = {f"val_conditions/{k}": v for k, v in conditions_log_data.items()}
    
    # Log to separate panels
    wandb_run.log(val_log_data_prefixed, step=step, **log_kwargs)
    if conditions_log_data_prefixed:  # Only log if there are per-condition metrics
        wandb_run.log(conditions_log_data_prefixed, step=step, **log_kwargs)


def _compute_log_stats(x, log_rnd, reward_fn, model, beta_batch=None, h_batch=None, J=1,
                       bias_potential=None, cv_compute_fn=None):
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
            if cv_compute_fn is not None:
                s = cv_compute_fn(x)  # Use provided CV computation function
            else:
                # Backward compatible: default to Ising magnetization
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
          bias_potential=None, current_fields=None, rng=None, save_dir=None, cfg_dict=None,
          validation_plot_callback=None, cv_compute_fn=None,
          buffer_size=0, buffer_ratio=0.0):
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

    # Initialize Replay Buffer
    replay_buffer = None
    if buffer_size > 0:
        # Determine dimension D from model
        D = model.length
        replay_buffer = ReplayBuffer(buffer_size, (D,), device=device)
        print(f"Initialized ReplayBuffer with size {buffer_size} and mixing ratio {buffer_ratio}")

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
            # Calculate CV (Magnetization for Ising, Concentration for CuAu, etc.)
            with torch.no_grad():
                if cv_compute_fn is not None:
                    s = cv_compute_fn(x)  # Use provided CV computation function
                else:
                    # Backward compatible: default to Ising magnetization
                    # x is [B, D] in {0, 1} usually? 
                    # utils_ising functions usually expect {-1, 1} but handle it?
                    # train_ising.py reward_fn converts 0/1 to -1/1.
                    # ising2d_mag inside utils_ising expects {-1, 1}
                    # rnd returns x in {0..(vocab-1)}. For Ising vocab=2 (0, 1).
                    # So we must convert to spins for ising2d_mag: 2*x - 1
                    x_spins = 2 * x - 1
                    s = ising2d_mag(x_spins)
                bias_potential.update(s)

        # Add to Replay Buffer - ONLY FRESH SAMPLES
        if replay_buffer is not None and is_fresh_sample:
            replay_buffer.add(x, beta=beta_batch, h=h_batch)
            
        # Prepare Training Batch (Mix Fresh + Replay)
        x_train, log_rnd_train = x, log_rnd
        beta_train, h_train = beta_batch, h_batch
        
        # Only mix if we have fresh samples (user requirement for WDCE)
        if is_fresh_sample and replay_buffer is not None and replay_buffer.size >= args.batch_size:
            n_replay = int(args.batch_size * buffer_ratio)
            if n_replay > 0 and n_replay < args.batch_size:
                # Slice fresh samples
                n_fresh = args.batch_size - n_replay
                x_f = x[:n_fresh]
                log_rnd_f = log_rnd[:n_fresh]
                beta_f = beta_batch[:n_fresh] if beta_batch is not None else None
                h_f = h_batch[:n_fresh] if h_batch is not None else None
                
                # Sample from Buffer
                x_r, beta_r, h_r = replay_buffer.sample(n_replay)
                
                # RECALCULATE log_rnd for Replay Samples
                # log_rnd = log_reward(current_bias) + log_model_term
                # Note: compute_model_log_prob returns sum[ -log(V-1) - log P(x_t) ]
                # So we simply add it to log_reward.
                with torch.no_grad():
                    # Recalculate Reward with CURRENT bias
                    J_val = args.J if hasattr(args, 'J') else 1
                    log_reward_r = reward_fn(x_r, beta=beta_r, h=h_r, J=J_val, use_bias=True)
                    
                    # Recalculate Model Term with CURRENT model
                    log_model_term_r = compute_model_log_prob(model, x_r, beta=beta_r, h=h_r)
                    
                    log_rnd_r = log_reward_r + log_model_term_r
                
                # Combine
                x_train = torch.cat([x_f, x_r], dim=0)
                log_rnd_train = torch.cat([log_rnd_f, log_rnd_r], dim=0)
                
                if beta_batch is not None:
                    # If beta_r came back as None (shouldn't if buffer works right), handle it
                    if beta_r is None: # Fallback
                         if beta_batch is not None:
                             beta_r = beta_batch[:n_replay] 
                    beta_train = torch.cat([beta_f, beta_r], dim=0)
                elif beta_r is not None:
                     # Fresh was None but Replay has Beta? (Unlikely in consistent run)
                     pass

                if h_batch is not None:
                    if h_r is None: 
                        if h_batch is not None:
                            h_r = h_batch[:n_replay]
                    h_train = torch.cat([h_f, h_r], dim=0)

        # Recalculate loss on mixed batch (overwriting previous loss calculation)
        if args.loss_fn == 'wdce':
            loss = loss_wdce(model, log_rnd_train, x_train,
                                num_replicates=args.wdce_num_replicates,
                                beta_batch=beta_train, h_batch=h_train)
        else:
            loss = loss_fn(log_rnd_train)

        # Synchronize loss across processes
        
        logf_t_vals, logp_x_vals = _compute_log_stats(x_train, log_rnd_train, reward_fn, model,
                                                       beta_batch=beta_train, h_batch=h_train,
                                                       J=args.J if hasattr(args, 'J') else 1,
                                                       bias_potential=bias_potential,
                                                       cv_compute_fn=cv_compute_fn)
        vfe = logp_x_vals - logf_t_vals  # variational free energy
        ess_train.append(ess(log_rnd_train))
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
                "train/avg_log_rnd": log_rnd_train.mean().item(),
                "train/std_log_rnd": log_rnd_train.std().item(),
                "train/min_log_rnd": log_rnd_train.min().item(),
                "train/max_log_rnd": log_rnd_train.max().item(),
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
                    fig = _visualize_lattices(x_train, L, n_rows=2, n_cols=5, max_samples=10,
                                              beta_batch=beta_train, h_batch=h_train)
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
                                                                   bias_potential=bias_potential,
                                                                   cv_compute_fn=cv_compute_fn)
                    vfe = logp_x_vals - logf_t_vals  # variational free energy
                    
                    # Use the new per-condition logging function (similar to snowyflow)
                    log_validation_metrics(
                        wandb_run=wandb_run,
                        logp_x=logp_x_vals,
                        logf_t=logf_t_vals,
                        vfe=vfe,
                        beta_batch=eval_beta_batch,
                        h_batch=eval_h_batch,
                        log_rnd=log_rnd,
                        step=epoch,
                    )
                    
                    # Also log ESS (log_rnd stats are included in log_validation_metrics)
                    wandb_run.log({"val/ess": eval_ess}, step=epoch)
                    
                    # Call validation plotting callback if provided
                    if validation_plot_callback is not None:
                        try:
                            # Convert beta/h to temps/fields for plotting
                            if eval_beta_batch is not None:
                                # beta = 1/(kB*T), so T = 1/(kB*beta)
                                if isinstance(eval_beta_batch, torch.Tensor):
                                    plot_temps = 1.0 / (eval_beta_batch * K_B)
                                else:
                                    plot_temps = torch.full((x.shape[0],), 1.0 / (eval_beta_batch * K_B), device=device, dtype=torch.float32)
                            else:
                                # Single temp case - need to get from args or use default
                                if hasattr(args, 'temp_min') and args.num_temps == 1:
                                    plot_temps = torch.full((x.shape[0],), args.temp_min, device=device, dtype=torch.float32)
                                else:
                                    plot_temps = torch.ones(x.shape[0], device=device, dtype=torch.float32)
                            
                            if eval_h_batch is not None:
                                plot_fields = eval_h_batch
                            else:
                                plot_fields = torch.zeros(x.shape[0], device=device, dtype=torch.float32)
                            
                            validation_plot_callback(
                                x=x,
                                temps=plot_temps,
                                fields=plot_fields,
                                wandb_run=wandb_run,
                                step=epoch,
                            )
                        except Exception as e:
                            logging.warning(f"Validation plotting callback failed: {e}")
                    
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
                            # Compute CV using provided function or default to Ising
                            if cv_compute_fn is not None:
                                s_eval = cv_compute_fn(x)  # Use provided CV computation function
                            else:
                                # Backward compatible: default to Ising magnetization
                                x_spins_eval = 2 * x - 1
                                s_eval = ising2d_mag(x_spins_eval)
                            
                            # Compute biased reward: R_biased = R_unbiased - beta * V(s)
                            # logf_t_vals contains R_unbiased values for the batch
                            
                            # Get V(s)
                            with torch.no_grad():
                                v_eval = bias_potential.evaluate(s_eval.to(device))
                                
                                # Get beta (use eval_beta_batch if available, else 1/T from bias_pot)
                                if eval_beta_batch is not None:
                                    beta_val = eval_beta_batch
                                else:
                                    beta_val = 1.0 / bias_potential.T
                                    
                                biased_reward_vals = logf_t_vals - beta_val * v_eval
                                
                            # Sample from Replay Buffer for visualization if available
                            s_buffer = None
                            if replay_buffer is not None and replay_buffer.size > 0:
                                n_sample = min(args.eval_batch_size, replay_buffer.size)
                                x_buf, _, _ = replay_buffer.sample(n_sample)
                                with torch.no_grad():
                                    if cv_compute_fn is not None:
                                        s_buffer = cv_compute_fn(x_buf.to(device))
                                    else:
                                        x_spins_buf = 2 * x_buf.to(device) - 1
                                        s_buffer = ising2d_mag(x_spins_buf)
                            fig_bias = plot_bias_analysis(bias_potential, epoch, s_batch=s_eval, 
                                                          biased_reward=biased_reward_vals, 
                                                          s_buffer=s_buffer)
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