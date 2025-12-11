import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from utils import ess, sample_categorical_logits


def rnd(model, reward_model, batch_size, device='cuda:0'):
    r"""
    Run random order sampling and compute the RND $\log\frac{dP^*}{dP^u}$ along the trajectory
    reward_model: r(X)

    return:
    - x: the final samples, [B, D]
    - log_rnd: the log RND along this trajectory, [B]
    """
    if hasattr(model, 'module'):
        model = model.module
    
    x = torch.full((batch_size, model.length), model.vocab_size-1).to(device=device, dtype=torch.int64)
    batch_arange = torch.arange(batch_size, device=device)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    log_rnd = torch.zeros(batch_size, device=device) # [B]
    for d in range(model.length-1, -1, -1):
        logits = model(x)[:, :, :-1] # [B, D, N-1]
        update = sample_categorical_logits(
            logits[batch_arange, jump_pos[:, d]]) # [B]
        if torch.is_grad_enabled(): # avoid issues with in-place operations
            x = x.clone()
        x[batch_arange, jump_pos[:, d]] = update
        log_rnd += -np.log(model.vocab_size-1) - logits[batch_arange, jump_pos[:, d], update]
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
    return (-log_rnd * (-log_rnd.detach() + const)).mean()


def loss_wdce(model, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l):
    r"""
    Weighted denoising cross entropy loss
    X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
    
    log_rnd: [B]; x: [B, D] (no mask)
    num_replicates: R, number of replicates of each row in x
    weight_func: w(lambda) for each sample, 1/lambda by default
    """
    if hasattr(model, 'module'):
        model = model.module
    
    batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, D]
    batch_weights = log_rnd.detach_().softmax(dim=-1).repeat_interleave(num_replicates, dim=0) # [B*R]
    lamda = torch.rand(batch.shape[0], device=batch.device) # [B*R]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B*R]
    masked_index = torch.rand(*batch.shape, device=batch.device) < lamda[..., None] # [B*R, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size-1, batch)
    logits = model(perturbed_batch)
    losses = torch.zeros(*batch.shape, device=batch.device, dtype=logits.dtype) # [B*R, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=batch[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights * batch_weights).mean()


def loss_dce(model, x, weight_func=lambda l: 1/l):
    r"""
    Denoising cross entropy loss, x [B, D] are ground truth samples
    weight_func: w(lambda) for each sample, 1/lambda by default
    """
    lamda = torch.rand(x.shape[0], device=x.device) # [B]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B]
    masked_index = torch.rand(*x.shape, device=x.device) < lamda[..., None] # [B, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size-1, x)
    logits = model(perturbed_batch)
    losses = torch.zeros(*x.shape, device=x.device, dtype=logits.dtype) # [B, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=x[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights).mean()


def _compute_log_stats(x, log_rnd, reward_fn, model):
    """Compute logf_t and logp_x given samples and RND values."""
    logf_t_vals = reward_fn(x)  # reward_fn already returns logf_t
    num_states = getattr(model, "vocab_size", 3) - 1  # exclude mask token
    data_dim = x.shape[1]
    uniform_prior_term = -torch.log(torch.tensor(float(num_states), device=x.device))
    uniform_prior_total = uniform_prior_term * data_dim
    logp_x_vals = uniform_prior_total + logf_t_vals - log_rnd
    return logf_t_vals, logp_x_vals


def _visualize_lattices(samples, L, n_rows=2, n_cols=5, max_samples=16):
    """Visualize Ising lattices in a grid.
    
    Args:
        samples: Tensor of shape [B, L*L] or [B, L, L]
        L: Lattice dimension
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        max_samples: Maximum number of samples to visualize
    """
    # Reshape if needed: [B, L*L] -> [B, L, L]
    if samples.ndim == 2:
        B = samples.shape[0]
        samples = samples.view(B, L, L)
    
    # Convert to float and CPU for matplotlib
    samples = samples.float().cpu()
    
    # Limit number of samples
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
          losses=None, ess_train=None, ess_eval=None, wandb_run=None, L=None):
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

    x_saved, log_rnd_saved = None, None
    
    for epoch in pbar:
        model.train(); optimizer.zero_grad(); info = {}

        if args.loss_fn == 'wdce':
            with torch.no_grad():
                if x_saved is None or epoch % args.resample_every_n_step == 0:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device)
                    ema.restore(model.parameters())
                    x_saved, log_rnd_saved = x, log_rnd
                else:
                    x, log_rnd = x_saved, log_rnd_saved

            loss = loss_wdce(model, log_rnd, x,
                                num_replicates=args.wdce_num_replicates)
        else:
            x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device)
            loss = loss_fn(log_rnd)
                
        # Synchronize loss across processes
        
        logf_t_vals, logp_x_vals = _compute_log_stats(x, log_rnd, reward_fn, model)
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
            
            # Log lattice visualization periodically (every 100 epochs)
            if L is not None and epoch % 100 == 0:
                try:
                    fig = _visualize_lattices(x, L, n_rows=2, n_cols=5, max_samples=10)
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
        
        # Evaluate periodically
        if epoch % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                x, log_rnd = rnd(model, reward_fn, args.eval_batch_size, device=device)
                eval_ess = ess(log_rnd)
                ess_eval.append(eval_ess)
                if wandb_run is not None:
                    logf_t_vals, logp_x_vals = _compute_log_stats(x, log_rnd, reward_fn, model)
                    vfe = logp_x_vals - logf_t_vals  # variational free energy
                    log_dict = {
                        "val/avg_free_energy": vfe.mean().item(),
                        "val/std_free_energy": vfe.std().item(),
                        "val/avg_logp_x": logp_x_vals.mean().item(),
                        "val/std_logp_x": logp_x_vals.std().item(),
                        "val/avg_logf_t": logf_t_vals.mean().item(),
                        "val/std_logf_t": logf_t_vals.std().item(),
                        "val/ess": eval_ess,
                    }
                    
                    # Log lattice visualization during evaluation
                    if L is not None:
                        try:
                            fig = _visualize_lattices(x, L, n_rows=2, n_cols=5, max_samples=10)
                            log_dict["val/samples"] = wandb.Image(fig)
                            plt.close(fig)
                        except Exception as e:
                            # Silently skip visualization if there's an error
                            pass
                    
                    wandb_run.log(log_dict, step=epoch)
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
                    x, log_rnd = rnd(model, reward_fn, args.batch_size, device=device)
                    ema.restore(model.parameters())
                    x_saved, log_rnd_saved = x, log_rnd
                else:
                    x, log_rnd = x_saved, log_rnd_saved

            loss = loss_wdce(model, log_rnd, x,
                                num_replicates=args.wdce_num_replicates)
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