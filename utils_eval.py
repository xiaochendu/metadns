import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

def sample_potts(L, q, beta, n_steps=1000, n_samples=1, device='cpu'):
    """
    Sample from Potts model using Metropolis-Hastings algorithm.
    
    Args:
        L (int): Size of the lattice (L x L)
        q (int): Number of states (0 to q-1)
        beta (float): Inverse temperature (coupling strength)
        n_steps (int): Number of MCMC steps
        n_samples (int): Number of independent samples to generate
        device (str): Device to use ('cpu' or 'cuda')
    
    Returns:
        torch.Tensor: Samples of shape (n_samples, L, L)
    """
    # Initialize random configuration
    S = torch.randint(0, q, (n_samples, L, L), device=device)
    
    # Define energy function for a single site
    def site_energy(s, neighbors):
        return -beta * torch.sum(s == neighbors)
    
    progress_bar = tqdm.tqdm(range(n_steps))
    
    for _ in progress_bar:
        # Randomly select sites to update
        i = torch.randint(0, L, (n_samples,), device=device)
        j = torch.randint(0, L, (n_samples,), device=device)
        
        # Get current states
        current_states = S[torch.arange(n_samples), i, j]
        
        # Propose new states
        proposed_states = torch.randint(0, q, (n_samples,), device=device)
        # Ensure proposed states are different from current states
        mask = proposed_states == current_states
        while mask.any():
            proposed_states[mask] = torch.randint(0, q, (mask.sum(),), device=device)
            mask = proposed_states == current_states
        
        # Get neighbors (using periodic boundary conditions)
        neighbors = torch.stack([
            S[torch.arange(n_samples), (i-1)%L, j],  # up
            S[torch.arange(n_samples), (i+1)%L, j],  # down
            S[torch.arange(n_samples), i, (j-1)%L],  # left
            S[torch.arange(n_samples), i, (j+1)%L]   # right
        ], dim=1)
        
        # Calculate energy difference
        current_energy = site_energy(current_states.unsqueeze(1), neighbors)
        proposed_energy = site_energy(proposed_states.unsqueeze(1), neighbors)
        delta_E = proposed_energy - current_energy
        
        # Metropolis acceptance step
        accept = torch.rand(n_samples, device=device) < torch.exp(-delta_E)
        
        # Update states
        S[torch.arange(n_samples), i, j] = torch.where(accept, proposed_states, current_states)
    
    return S
def visualize_potts(S, q):
    """
    Visualize Potts model samples using heatmap.
    
    Args:
        S (torch.Tensor): Potts model samples of shape (B, L, L)
        q (int): Number of states (0 to q-1)
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    B = S.shape[0]
    # Check if B is a perfect square
    k = int(np.sqrt(B))
    if k * k == B:
        # Create a k x k grid of subplots
        fig, axes = plt.subplots(k, k, figsize=(3*k, 3*k))
        axes = axes.ravel()  # Flatten the axes array for easy iteration
    else:
        # Fallback to single row if B is not a perfect square
        fig, axes = plt.subplots(1, B, figsize=(3*B,3))
        if B == 1:
            axes = [axes]
    
    # Create a colormap with q distinct colors
    cmap = 'inferno'
    for i in range(B):
        # Plot heatmap for each sample
        im = axes[i].imshow(S[i], cmap=cmap, vmin=0, vmax=q-1)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    # Add colorbar
    # cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05)
    # cbar.set_ticks(np.arange(q))
    # cbar.set_ticklabels(np.arange(q))
    
    plt.tight_layout()
    plt.show()
    
def visualize_ising2d(S):
    """
    Visualize Ising model samples using heatmap.
    
    Args:
        S (torch.Tensor or numpy.ndarray): Ising model samples of shape (B, L, L) or (B, L*L)
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(S, torch.Tensor):
        S = S.detach().cpu().numpy()
    
    # Reshape if needed
    if S.ndim == 2:
        L = int(np.sqrt(S.shape[1]))
        S = S.reshape(-1, L, L)
    
    B = S.shape[0]
    # Check if B is a perfect square
    k = int(np.sqrt(B))
    if k * k == B:
        # Create a k x k grid of subplots
        fig, axes = plt.subplots(k, k, figsize=(3*k, 3*k))
        axes = axes.ravel()  # Flatten the axes array for easy iteration
    else:
        # Fallback to single row if B is not a perfect square
        fig, axes = plt.subplots(1, B, figsize=(3*B,3))
        if B == 1:
            axes = [axes]
    
    # Create a colormap for binary states
    cmap = 'RdBu'
    for i in range(B):
        # Plot heatmap for each sample
        im = axes[i].imshow(S[i], cmap=cmap, vmin=-1, vmax=1)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    # Add colorbar
    # cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05)
    # cbar.set_ticks([-1, 1])
    # cbar.set_ticklabels(['-1', '1'])
    
    plt.tight_layout()
    plt.show()