"""
Utility functions for the 1D and 2D Ising models

Only MH sampling is implemented by numpy,
other functions are implemented by torch.

Please be aware of the input shape and range before use!
"""


import numpy as np
import torch
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def ising2d_ham(S, J=1.0, h=0.0):
    r"""
    Compute the Hamiltonian for a batch of configurations in a 2D Ising model, with periodic boundary conditions.
    
    Parameters:
    - S: torch.tensor of shape 
        1) (B, L * L):
            each element is -1 or 1, representing spin configurations.
        2) (B, L * L, 2):
            the last dimension contains the probability of that spin being -1 and 1, respectively.
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field strength (default=0.0).

    Returns:
    - hamiltonians: torch.tensor of shape (B,) containing the Hamiltonian for each configuration.
        H = -J \sum_{i \sim j} S_{i} S_{j} - h \sum_{i} S_{i}
    (The p.m.f. is given by p(S) \propto e^{-\beta H(S)})
    """
    if S.ndim == 2:
        assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    elif S.ndim == 3:
        assert S.shape[2] == 2, "Input tensor must have shape (B, L * L, 2)"
        S = S[..., 1] - S[..., 0]  # convert to average spins, now (B, L * L)
    else:
        raise ValueError(f"Invalid input shape {S.shape}.")

    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5))
    Sx = torch.roll(S, shifts=-1, dims=1)  # Sx[i,j] = S[i+1,j]
    Sy = torch.roll(S, shifts=-1, dims=2)  # Sy[i,j] = S[i,j+1]
    interaction_energy = -J * torch.sum(S * (Sx + Sy), dim=(1, 2))
    magnetic_energy = -h * torch.sum(S, dim=(1, 2))
    return interaction_energy + magnetic_energy


def ising2d_get_all_configs(L=4, device='cuda:0'):
    """
    Generate all possible Ising configurations for L x L lattice in increasing order
    e.g., [-1, -1], [-1, 1], [1, -1], [1, 1].
    Return: [2 ** (L ** 2), L ** 2], values are in {1, -1}
    """
    B = 2 ** (L ** 2)
    bits = torch.arange(L ** 2 - 1, -1, -1, device=device)
    return (((torch.arange(B, device=device)[:, None] >> bits) & 1) * 2 - 1).to(torch.int8) # [B, L ** 2]


def ising2d_part_func(L=4, beta=.5, J=1.0, h=0.0, device='cuda:0'):
    """
    Compute the partition function of a 2D Ising model with periodic boundary conditions.
    """
    assert 1 <= L <= 4, "Only support L <= 4 due to memory constraints"
    configs = ising2d_get_all_configs(L=L, device=device) # (2 ** D, D)
    H = ising2d_ham(configs, J=J, h=h) # (2 ** D,)
    return torch.exp(-beta * H).sum().item()


def ising2d_mag(S):
    """
    Compute the magnetization for a batch of configurations.

    Parameters:
    S: torch.tensor of shape (B, L * L) representing K configurations on an L x L lattice.
       Each element in S is +1 or -1.

    Returns:
    - torch.tensor of shape (B,) containing the magnetization for each configuration.
    """
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    return S.float().mean(dim=1)


def ising2d_2pt_corr(S, rx, ry):
    """
    Compute the two-point correlation function for a batch of configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing K configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - rx, ry: int, horizontal and vertical distance between points for correlation calculation.
    
    Return:
    - torch.tensor of shape (B,) containing the two-point correlation for each configuration.
    """
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    S = S.view(S.size(0), int(S.shape[1]**.5), int(S.shape[1]**.5))
    
    
    
    Sx = torch.roll(S, shifts=-rx, dims=1)
    Sy = torch.roll(S, shifts=-ry, dims=2)
    return (S * (Sx + Sy)).float().view(S.size(0), -1).mean(dim=1)


def ising2d_mh(L, J=1, h=0, beta=.5, B=256, num_collect=20000, 
               burn_in=10000, collect_every=1000, init=None):
    """
    Metropolis-Hastings algorithm to sample from the 2D Ising model's distribution.

    Parameters:
    - L: int, size of the lattice (L * L).
    - B: int, number of parallel configurations.
    - num_collect: int, number of times to collect.
    - burn_in: int, number of initial steps to discard (burn-in period).
    - collect_every: int, collect a sample every `collect_every` steps.
    - init: numpy.ndarray of shape (B, L, L) or (B, L * L), initial configuration.
            If None, random configurations are used.
    Returns:
    - samples: numpy.ndarray of shape (num_collect * B, L * L), sampled configurations.
    """
    if init is not None:
        S = init.reshape(B, L, L)
    else:
        S = np.random.choice([-1, 1], size=(B, L, L))
    samples = []
    batch_arange = np.arange(B)
    pbar = tqdm(range(num_collect * collect_every + burn_in))
    for step in pbar:
        i, j = np.random.randint(0, L, size=(B,)), np.random.randint(0, L, size=(B,))
        dH = 2 * J * S[batch_arange, i, j] * (
            S[batch_arange, (i - 1) % L, j] + S[batch_arange, (i + 1) % L, j]
            + S[batch_arange, i, (j - 1) % L] + S[batch_arange, i, (j + 1) % L]
            ) + 2 * h * S[batch_arange, i, j]
        flip = np.random.rand(B) < np.exp(-beta * dH)
        S[batch_arange[flip], i[flip], j[flip]] *= -1
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(np.copy(S))
    return np.array(samples).reshape(-1, L * L)


def ising2d_emp_dist(samples):
    """
    samples: [B, L^2], elements in {-1, 1}
    Output the empirical distribution of the samples as a probability vector of length 2^{L^2}
    The configuations are sorted in increasing order 
    """
    assert torch.all((samples == 1) | (samples == -1)), "All entries of samples must be either 1 or -1"
    B, N = samples.shape  # N = L^2
    bin_samples = ((samples + 1) // 2).to(torch.int32)  # (B, N)
    bits = torch.arange(N - 1, -1, -1, device=samples.device)
    indices = (bin_samples << bits).sum(dim=1)  # (B,)
    counts = torch.bincount(indices, minlength=2 ** N)
    return counts.float() / B


def ising2d_get_pmf(L, J=1.0, h=0.0, beta=1.0, device='cuda:0'):
    """
    Compute the pmf of all configurations (in increasing order), shape [2**(L^2)]
    """
    all_configs = ising2d_get_all_configs(L, device)
    log_pmf = -beta * ising2d_ham(all_configs, J=J, h=h) # [2**D]
    return log_pmf.softmax(dim=0)


def ising2d_row_col_mag(S, axis=0):
    """
    Compute the magnetization for each row or column in a batch of 2D Ising configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing B configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - axis: int, 0 for row-wise magnetization, 1 for column-wise magnetization.
    
    Returns:
    - torch.tensor of shape (B, L) containing the magnetization for each row/column.
    """
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    L = int(S.shape[1]**.5)
    S = S.view(S.size(0), L, L)
    return S.float().mean(dim=axis+1)  # +1 because first dim is batch


########################################
# Codes for 1D have not been checked and we will not use them

def ising1d_mh(N, J, h, beta, steps, burn_in=0, collect_every=100):
    """
    Samples from the 1D Ising model using the Metropolis algorithm.

    Parameters:
    - N: Number of spins
    - J: Coupling constant
    - h: External magnetic field
    - beta: Inverse temperature (1 / (k_B * T))
    - steps: Number of MCMC steps

    The density is exp(-beta * H(sigma)), where
    H(sigma) = -J sum_{i=0}^{N-1} sigma[i] sigma[i+1] - h sum_{i=0}^{N-1} sigma[i]
    """
    spins = np.random.choice([-1, 1], size=N)
    all_spins = []

    def delta_energy(spins, i):
        left = spins[i - 1] if i > 0 else 0  # Handle left boundary
        right = spins[i + 1] if i < N - 1 else 0  # Handle right boundary
        return 2 * J * spins[i] * (left + right) + 2 * h * spins[i]

    for step in range(steps):
        spins = spins.copy()
        i = np.random.randint(0, N)
        dE = delta_energy(spins, i)
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i] *= -1
        if step > burn_in and step % collect_every == 0:
            all_spins.append(spins)

    return np.array(all_spins)


def ising1d_par(N, J, h, beta):
    """
    Calculate the partition function Z of the 1D Ising model
    """
    configurations = np.array(np.meshgrid(*[[-1, 1]] * N)).T.reshape(-1, N)
    interaction_energy = -J * np.sum(configurations[:, :-1] * configurations[:, 1:], axis=1)
    external_field_energy = -h * np.sum(configurations, axis=1)
    energies = interaction_energy + external_field_energy
    return np.sum(np.exp(-beta * energies))


def ising1d_llh(sigmas, J, h, beta, Z=None):
    """
    Compute the log-likelihood of each configuration in the 1D Ising model.
    sigmas is an M x N matrix of configurations (M configurations, each of length N)
    Returns log_likelihoods = -beta * H(sigma) - log(Z)
    """
    interaction_energy = -J * np.sum(sigmas[:, :-1] * sigmas[:, 1:], axis=1)
    external_field_energy = -h * np.sum(sigmas, axis=1)
    energies = interaction_energy + external_field_energy
    if Z is None:
        Z = ising1d_par(N=sigmas.shape[1], J=J, h=h, beta=beta)
    return -beta * energies - np.log(Z)


def visualize_ising(S, k_x, k_y):    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    assert k_x * k_y == S.shape[0], "k_x * k_y must be equal to the number of samples"
    B = S.shape[0]
    # Check if B is a perfect square
    k = int(np.sqrt(B))
    fig, axes = plt.subplots(k_x, k_y, figsize=(1.5*k_x, 1.5*k_y), constrained_layout=True)
    axes = axes.ravel()  # Flatten the axes array for easy iteration
    
    # Create a colormap with q distinct colors
    # palette = ["#440154", "#FDE725", "#ff7f00"]
    palette = ["#313342", "#DEB4B2"]
    # palette = ["white", "black"]
    cmap = ListedColormap(palette)    
    for i in range(B):
        # Plot heatmap for each sample
        im = axes[i].imshow(S[i], cmap=cmap, vmin=-1, vmax=1, origin='lower',
            interpolation='nearest')
        # Hide ticks and their labels but keep the frame
        # axes[i].set_xticks([])
        # axes[i].set_yticks([])
        # axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    # Add colorbar
    # cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05)
    # cbar.set_ticks(np.arange(q))
    # cbar.set_ticklabels(np.arange(q))
    plt.tight_layout()
    return fig

def ising_2pt_corr_direction(S, r_x, r_y, use_x = True, use_y = True):
    """
    Compute the two-point correlation function for a batch of configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing K configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - rx, ry: int, horizontal and vertical distance between points for correlation calculation.
    
    Return:
    - torch.tensor of shape (B,) containing the two-point correlation for each configuration.
    """
    
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(S)
    
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    S = S.reshape(-1, 16, 16)
    
    if use_x:
        S_neighbor = 1/2 * (torch.roll(S, shifts=-r_x, dims=1) + torch.roll(S, shifts=r_x, dims=1))
    if use_y:
        S_neighbor = 1/2 * (torch.roll(S, shifts=-r_y, dims=2) + torch.roll(S, shifts=r_y, dims=2))
        
    return (S * S_neighbor).float().mean()

def ising2d_mag_direction(S, use_row = True, use_col = False):
    """
    Compute the magnetization for each row or column in a batch of 2D Ising configurations.
    
    Parameters:
    - S: torch.tensor of shape (B, L * L) representing B configurations on an L x L lattice.
         Each element in S is +1 or -1.
    - axis: int, 0 for row-wise magnetization, 1 for column-wise magnetization.
    
    Returns:
    - torch.tensor of shape (B, L) containing the magnetization for each row/column.
    """
    
    if not isinstance(S, torch.Tensor):
        S = torch.from_numpy(S)
    assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"
    S = S.view(S.size(0), 16, 16)
    
    if use_row:
        return S.float().mean(dim=1).mean(dim = 0)
    if use_col:
        return S.float().mean(dim=2).mean(dim = 0)