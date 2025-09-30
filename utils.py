from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np



def compute_ess(log_rnd, normalize=True): 
    """
    log_rnd: [B]
    Compute effective sample size:
        If normalize: divide ESS by batch size, so range is [0, 1]; 
        otherwise, range is [0, B]
    """
    weights = log_rnd.detach().softmax(dim=-1)
    ess = 1 / (weights ** 2).sum().item()
    return ess / log_rnd.shape[0] if normalize else ess


def sample_categorical(categorical_probs, dtype=torch.float64):
    # do not require probs to be normalized
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs, dtype=dtype) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def sample_categorical_logits(logits, dtype=torch.float64):
    # do not require logits to be log-softmaxed
    gumbel_noise = -(1e-10 - (torch.rand_like(logits, dtype=dtype) + 1e-10).log()).log()
    return (logits + gumbel_noise).argmax(dim=-1)


def sample_categorical_gradient(categorical_probs, temp=1.0, dtype=torch.float64):
    """
    Adapted from DRAKES
    do not require probs to be normalized
    """
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs, dtype=dtype) + 1e-10).log()
    # return torch.nn.functional.softmax(
        # (categorical_probs.log() - gumbel_norm.log())/temp, 2)
    return ((categorical_probs.log() - gumbel_norm.log()) / temp).softmax(dim=-1)

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


class LogLinearNoise(nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)
    
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)


class Dict2Obj:
    def __init__(self, dic=None):
        if dic is not None: 
            for key, value in dic.items():
                if isinstance(value, dict):
                    value = Dict2Obj(value)
                setattr(self, key, value)
    def __repr__(self):
        return str(self.__dict__)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    base_lr = optimizer.param_groups[0]["lr"]
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + np.cos(np.pi * progress))
        return max(min_lr / base_lr, cosine)
    return LambdaLR(optimizer, lr_lambda)


def cycleloader(dataloader):
    while True:
        for data in dataloader:
            yield data