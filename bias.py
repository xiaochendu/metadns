import numpy as np
import torch


class BiasPotential:
    """
    Bias Potential for Well-Tempered Metadynamics / Ajoint Schrödinger Bridge Sampler (WT-ASBS).
    
    Supports:
    - 1D Collective Variables (CV), specifically geared towards Magnetization/Composition.
    - Gaussian Kernels (Standard Metadynamics)
    - Delta/Kronecker Kernels (Discrete/Histogram Biasing)
    """
    def __init__(self, 
                 cv_min, 
                 cv_max, 
                 grid_size, 
                 sigma, 
                 initial_height, 
                 bias_factor, 
                 T, 
                 kernel_type='gaussian',
                 device='cpu'):
        """
        Args:
            cv_min (float): Minimum value of the CV (e.g., -1 for magnetization).
            cv_max (float): Maximum value of the CV (e.g., +1 for magnetization).
            grid_size (int): Number of bins in the grid.
            sigma (float): Width of Gaussian kernel (ignored for 'delta').
            initial_height (float): Initial height (W) of the added bias.
            bias_factor (float): Bias factor (gamma). gamma > 1. 
            T (float): Temperature (kB*T units).
            kernel_type (str): 'gaussian' or 'delta'.
            device (str): torch device.
        """
        self.cv_min = cv_min
        self.cv_max = cv_max
        self.grid_size = grid_size
        self.sigma = sigma
        self.initial_height = initial_height
        self.gamma = bias_factor
        self.T = T
        self.kernel_type = kernel_type.lower()
        self.device = device
        
        # Initialize grid
        # using linspace for bucket centers
        self.grid_vals = torch.linspace(cv_min, cv_max, grid_size, device=device)
        self.bias_grid = torch.zeros(grid_size, device=device)
        
        # Precompute kernel constant for delta_T
        # Delta_T = (gamma - 1) * T
        if self.gamma > 1.0:
            self.delta_T = (self.gamma - 1) * self.T
        else:
            # If gamma=1, standard MD (no well-temperedness), effectively infinite delta_T or handled separately
            # But usually WT means gamma > 1. If gamma=1, bias doesn't decay? 
            # Actually standard metadynamics is gamma -> infinity.
            # If gamma = 1, we don't build bias? 
            # Let's assume user provides valid gamma > 1 for WT.
            self.delta_T = 1e9 # Large number implies standard metadynamics (W doesn't decay)

    def _get_gaussian_kernel(self, center):
        """Returns a Gaussian kernel centered at `center` evaluated on the grid."""
        # exp( - (x - center)^2 / (2*sigma^2) )
        return torch.exp( - (self.grid_vals - center)**2 / (2 * self.sigma**2) )

    def _get_delta_kernel(self, center):
        """Returns a Delta kernel (1.0 at nearest bin, 0 elsewhere)."""
        # Find nearest bin index
        idx = torch.argmin(torch.abs(self.grid_vals - center))
        kernel = torch.zeros_like(self.grid_vals)
        kernel[idx] = 1.0
        return kernel

    def update(self, cv_batch):
        """
        Update the bias potential based on a batch of visited CV values.
        
        Args:
           cv_batch (Tensor): [B] tensor of visited CV values.
        """
        # Ensure input is on correct device
        cv_batch = cv_batch.to(self.device)
        
        for cv_val in cv_batch:
            # 1. Evaluate current bias at this CV location
            # Interpolation is ideal, but for discrete grid/dense grid, nearest lookup is often sufficient/faster
            # Let's use linear interpolation for 'gaussian' mode to be smooth, 
            # and nearest bin for 'delta' mode.
            
            if self.kernel_type == 'delta':
                idx = torch.argmin(torch.abs(self.grid_vals - cv_val))
                current_v = self.bias_grid[idx]
            else:
                # Simple finding of nearest for evaluating V(s) for the update rule 
                # (Standard WT uses V(s_t) exactly at the point)
                # For smooth grids, we can interpolate. For now, nearest grid point approximation 
                # is standard in many lightweight implementations.
                idx = torch.argmin(torch.abs(self.grid_vals - cv_val))
                current_v = self.bias_grid[idx]

            # 2. Calculate height scaling (Well-Tempered)
            # W_eff = W * exp( - V(s_t) / (kB * Delta_T) )
            scale_factor = torch.exp( - current_v / self.delta_T )
            height = self.initial_height * scale_factor
            
            # 3. Add kernel
            if self.kernel_type == 'delta':
                # Add directly to the bin
                self.bias_grid[idx] += height
                
            elif self.kernel_type == 'gaussian':
                # Add gaussian to the whole grid
                kernel = self._get_gaussian_kernel(cv_val)
                self.bias_grid += height * kernel

    def evaluate(self, cv_batch):
        """
        Return the bias energy V(s) for a batch of CV values.
        
        Args:
            cv_batch (Tensor): [B] Values of CV.
            
        Returns:
            start_v (Tensor): [B] Bias values.
        """
        cv_batch = cv_batch.to(self.device)
        
        # For each sample, find value from grid
        # We can implement efficient batched interpolation
        
        # Find indices
        # Assuming uniform grid:
        # result = bias_grid[ (val - min) / step ] 
        step = (self.cv_max - self.cv_min) / (self.grid_size - 1)
        
        # Clamping to ensure we stay in bounds
        indices_float = (cv_batch - self.cv_min) / step
        indices_long = torch.round(indices_float).long()
        indices_long = torch.clamp(indices_long, 0, self.grid_size - 1)
        
        return self.bias_grid[indices_long]


    def get_bias_grid_np(self):
        """Return grid for plotting (numpy)"""
        return self.grid_vals.detach().cpu().numpy(), self.bias_grid.detach().cpu().numpy()

    def state_dict(self):
        """Return state dictionary for saving."""
        return {
            'bias_grid': self.bias_grid,
            'grid_vals': self.grid_vals,
            'params': {
                'cv_min': self.cv_min,
                'cv_max': self.cv_max,
                'grid_size': self.grid_size,
                'sigma': self.sigma,
                'initial_height': self.initial_height,
                'bias_factor': self.gamma,
                'T': self.T,
                'kernel_type': self.kernel_type
            }
        }

    def load_state_dict(self, state_dict):
        """Load state from dictionary."""
        self.bias_grid = state_dict['bias_grid'].to(self.device)
        self.grid_vals = state_dict['grid_vals'].to(self.device)
        # We assume params are consistent or handled by init, 
        # but we could optionally overwrite them if needed. 
        # For now, just loading the grid is the most critical part.
