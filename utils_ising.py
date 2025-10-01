"""
Utility functions for the 2D Ising models
Only MH sampling is implemented by numpy, other functions are implemented by torch.
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
    - S: torch.tensor of shape (B, L * L)
        each element is -1 or 1 (not 0 or 1!), representing spin configurations.
    - J: float, interaction strength between neighboring spins (default=1.0).
    - h: float, external magnetic field strength (default=0.0).

    Returns:
    - hamiltonians: torch.tensor of shape (B,) containing the Hamiltonian for each configuration.
        H = -J \sum_{i \sim j} S_{i} S_{j} - h \sum_{i} S_{i}
    (The p.m.f. is given by p(S) \propto e^{-\beta H(S)})
    """
    # assert S.ndim == 2
    # assert torch.all((S == 1) | (S == -1)), "All entries of S must be either 1 or -1"

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


def visualize_ising(S, k_x, k_y):    # Convert to numpy if it's a torch tensor
    S = 2 * S - 1
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    if S.ndim == 2:
        S = S.reshape(-1, int(np.sqrt(S.shape[1])), int(np.sqrt(S.shape[1])))
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


def ising2d_swendsen_wang(L, J=1, beta=.5, B=256, num_collect=20000, 
                         burn_in=10000, collect_every=1000, init=None):
    """
    Swendsen-Wang algorithm to sample from the 2D Ising model's distribution.
    
    Parameters:
    - L: int, size of the lattice (L * L)
    - J: float, coupling constant
    - beta: float, inverse temperature
    - B: int, number of parallel configurations
    - num_collect: int, number of times to collect
    - burn_in: int, number of initial steps to discard
    - collect_every: int, collect a sample every `collect_every` steps
    - init: numpy.ndarray of shape (B, L, L) or (B, L * L), initial configuration
    
    Returns:
    - samples: numpy.ndarray of shape (num_collect * B, L * L), sampled configurations
    """
    # Initialize the lattice
    if init is None:
        S = np.random.choice([-1, 1], size=(B, L, L))
    else:
        S = init.reshape(B, L, L) if init.ndim == 2 else init
    
    samples = []
    total_steps = burn_in + num_collect * collect_every
    
    # Pre-compute bond probability
    p = 1 - np.exp(-2 * beta * J)
    
    for step in tqdm(range(total_steps)):
        # For each configuration in the batch
        for b in range(B):
            # Step 1: Identify bonds between aligned spins
            # Create arrays for horizontal and vertical bonds
            h_bonds = np.zeros((L, L), dtype=bool)  # horizontal bonds
            v_bonds = np.zeros((L, L), dtype=bool)  # vertical bonds
            
            # Check horizontal bonds (same spin alignment)
            h_bonds[:, :-1] = (S[b, :, :-1] == S[b, :, 1:])
            h_bonds[:, -1] = (S[b, :, -1] == S[b, :, 0])  # periodic BC
            
            # Check vertical bonds (same spin alignment)
            v_bonds[:-1, :] = (S[b, :-1, :] == S[b, 1:, :])
            v_bonds[-1, :] = (S[b, -1, :] == S[b, 0, :])  # periodic BC
            
            # Step 2: Activate bonds with probability p
            h_bonds = h_bonds & (np.random.random((L, L)) < p)
            v_bonds = v_bonds & (np.random.random((L, L)) < p)
            
            # Step 3: Identify clusters using Union-Find
            # Initialize parent array for Union-Find
            parent = np.arange(L * L).reshape(L, L)
            rank = np.zeros((L, L), dtype=int)
            
            def find(x, y):
                if parent[x, y] != x * L + y:
                    px, py = parent[x, y] // L, parent[x, y] % L
                    parent[x, y] = find(px, py)
                return parent[x, y]
            
            def union(x1, y1, x2, y2):
                root1 = find(x1, y1)
                root2 = find(x2, y2)
                if root1 != root2:
                    r1, c1 = root1 // L, root1 % L
                    r2, c2 = root2 // L, root2 % L
                    if rank[r1, c1] < rank[r2, c2]:
                        parent[r1, c1] = root2
                    else:
                        parent[r2, c2] = root1
                        if rank[r1, c1] == rank[r2, c2]:
                            rank[r1, c1] += 1
            
            # Process horizontal bonds
            for i in range(L):
                for j in range(L):
                    if h_bonds[i, j]:
                        union(i, j, i, (j + 1) % L)
            
            # Process vertical bonds
            for i in range(L):
                for j in range(L):
                    if v_bonds[i, j]:
                        union(i, j, (i + 1) % L, j)
            
            # Step 4: Identify clusters
            clusters = {}
            for i in range(L):
                for j in range(L):
                    root = find(i, j)
                    if root not in clusters:
                        clusters[root] = []
                    clusters[root].append((i, j))
            
            # Step 5: Flip clusters with probability 0.5
            for cluster in clusters.values():
                # Randomly decide whether to flip the cluster
                if np.random.random() < 0.5:
                    # Flip all spins in the cluster
                    for i, j in cluster:
                        S[b, i, j] *= -1
        
        # Collect samples after burn-in
        if step >= burn_in and (step - burn_in) % collect_every == 0:
            samples.append(S.reshape(B, L*L).copy())
    
    return np.concatenate(samples, axis=0)