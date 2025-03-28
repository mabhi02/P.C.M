#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Correlation Manifold (PCM) - Ultra-Light Topological Feature Extraction Component

This file implements a highly efficient version of the topological analysis component 
for AI-generated image detection. It analyzes the topological structure of the
phase correlation manifold using persistent homology with extreme optimizations for speed and size.

Key components:
1. Minimal Persistent Homology: Computes essential topological features only
2. Betti Numbers: Focuses only on dimension 0 (connected components)
3. Simplified feature processing: Uses minimal network architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
import gudhi as gd


class TinyPersistentHomologyLayer(nn.Module):
    """Ultra-light persistent homology features with minimal computation"""
    
    def __init__(self, max_edge_length=2.0, num_filtrations=10, max_dimension=0):
        super(TinyPersistentHomologyLayer, self).__init__()
        
        # Only use dimension 0 (connected components)
        self.max_edge_length = max_edge_length
        # Reduce num_filtrations from 20 to 10
        self.num_filtrations = num_filtrations
        self.max_dimension = max_dimension
        
        # Fixed filtration values (non-learnable to save parameters)
        self.register_buffer(
            'filtration_values',
            torch.linspace(0, max_edge_length, num_filtrations)
        )
    
    def forward(self, x):
        """
        Compute persistence diagrams from point cloud data - ultra-light version
        
        Args:
            x (torch.Tensor): Point cloud data [batch_size, num_points, features]
            
        Returns:
            dict: Dictionary containing only essential topological features
        """
        batch_size = x.size(0)
        device = x.device
        
        # Aggressively reduce computation: process only a subset of points
        # Use at most 25 points per batch item
        max_points = min(25, x.size(1))
        if x.size(1) > max_points:
            indices = torch.randperm(x.size(1))[:max_points]
            x = x[:, indices, :]
        
        # Detach and convert to numpy for gudhi
        x_np = x.detach().cpu().numpy()
        
        # Prepare output containers - only store essential data
        all_betti_curves = []
        
        # Process each batch item
        for i in range(batch_size):
            # Compute pairwise distances
            point_cloud = x_np[i]
            distances = squareform(pdist(point_cloud))
            
            # Create Vietoris-Rips complex with minimal dimensionality
            rips_complex = gd.RipsComplex(distance_matrix=distances)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension+1)
            
            # Compute persistence
            simplex_tree.compute_persistence()
            
            # Only compute Betti curves for efficiency
            filtration_values = self.filtration_values.detach().cpu().numpy()
            betti_curves = self._compute_betti_curves(simplex_tree, filtration_values)
            all_betti_curves.append(betti_curves)
        
        # Stack batch results and move back to the original device
        betti_curves_tensor = torch.stack(all_betti_curves, dim=0).to(device)
        
        return {
            'betti_curves': betti_curves_tensor
        }
    
    def _compute_betti_curves(self, simplex_tree, filtration_values):
        """Compute minimal Betti curves"""
        # Only compute for dimension 0
        betti_curves = []
        
        for dim in range(self.max_dimension + 1):
            betti_numbers = []
            
            # Use extremely sparse sampling for efficiency
            sparse_indices = np.linspace(0, len(filtration_values)-1, 5, dtype=int)
            for idx in sparse_indices:
                epsilon = filtration_values[idx]
                # Count number of features alive at this filtration value
                betti = simplex_tree.persistent_betti_numbers(0, epsilon)
                betti_number = betti[dim] if dim < len(betti) else 0
                betti_numbers.append(betti_number)
            
            # Interpolate back to original size
            if len(betti_numbers) < len(filtration_values):
                betti_numbers = np.interp(
                    np.linspace(0, 1, len(filtration_values)),
                    np.linspace(0, 1, len(betti_numbers)),
                    betti_numbers
                ).tolist()
            
            betti_curves.append(betti_numbers)
        
        return torch.tensor(betti_curves, dtype=torch.float32)


class TinyTopologicalFeatureExtraction(nn.Module):
    """Ultra-light topological feature extraction module"""
    
    def __init__(self, 
                 input_dim=32,           # Input feature dimension
                 hidden_dim=32,          # Hidden dimension (reduced)
                 output_dim=16,          # Output feature dimension (reduced)
                 max_edge_length=2.0,    # Maximum edge length for filtration
                 num_filtrations=20,     # Further reduced filtration values
                 max_dimension=1,        # Only dimension 0 for maximum efficiency
                ):
        super(TinyTopologicalFeatureExtraction, self).__init__()
        
        # Persistence homology layer - ultra-light
        self.persistent_homology = TinyPersistentHomologyLayer(
            max_edge_length=max_edge_length,
            num_filtrations=num_filtrations,
            max_dimension=max_dimension
        )
        
        # Process Betti curves - ultra-light (shared weights for all dimensions)
        self.betti_processor = nn.Sequential(
            nn.Linear(num_filtrations, hidden_dim // 2),  # Further reduced hidden dimension
            nn.ReLU(),
            nn.Dropout(0.1),  # Minimal dropout
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Skip persistence images entirely for efficiency
        
        # Quantize weights to reduce memory footprint
        self.quantize_weights()
    
    def quantize_weights(self):
        """Quantize weights to reduce model size"""
        # This is a placeholder - in a real implementation, you would
        # implement actual weight quantization here
        pass
    
    def forward(self, x):
        """
        Extract minimal topological features - ultra-light version
        
        Args:
            x (torch.Tensor): Manifold points [batch_size, num_points, feature_dim]
            
        Returns:
            torch.Tensor: Topological features [batch_size, output_dim]
        """
        device = x.device
        
        # Early stopping if confidence is already high (implementation depends on use case)
        # For now, always proceed with computation
        
        # Compute only essential persistent homology
        homology_data = self.persistent_homology(x)
        betti_curves = homology_data['betti_curves'].to(device)
        
        # Process Betti curves - only use dimension 0 for maximum efficiency
        batch_size = betti_curves.size(0)
        dim = 0  # Only use dimension 0
        dim_betti = betti_curves[:, dim, :]
        topo_features = self.betti_processor(dim_betti)
        
        # We're not computing persistence diagrams or images for efficiency
        # Return minimal data needed for the task
        return topo_features, {
            'betti_curves': betti_curves
        }


def extract_topological_features(manifold_features, device='cpu'):
    """Process manifold features through the tiny topological feature extraction module"""
    # Create the ultra-light topological feature extraction module
    topo_module = TinyTopologicalFeatureExtraction().to(device)
    
    # Process manifold features
    with torch.no_grad():
        topo_features, topo_data = topo_module(manifold_features)
    
    print(f"Manifold feature shape: {manifold_features.shape}")
    print(f"Topological feature shape: {topo_features.shape}")
    
    return topo_features, topo_data


def visualize_betti_curves(betti_curves, filtration_values=None, title="Betti Curves"):
    """Visualize Betti curves (simplified visualization)"""
    import matplotlib.pyplot as plt
    
    if filtration_values is None:
        filtration_values = torch.linspace(0, 1, betti_curves.shape[1])
    
    plt.figure(figsize=(8, 4))
    plt.plot(filtration_values.cpu().numpy(), 
             betti_curves[0].cpu().numpy(), 
             label=f"Dimension 0")
    
    plt.xlabel('Filtration Value')
    plt.ylabel('Betti Number')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt


if __name__ == "__main__":
    # If this file is run directly, perform a quick test
    import os
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy point cloud data for testing
    num_points = 50
    feature_dim = 32
    batch_size = 1
    
    # Create a synthetic point cloud
    point_cloud = torch.randn(batch_size, num_points, feature_dim, device=device)
    
    # Topological feature extraction
    topo_module = TinyTopologicalFeatureExtraction().to(device)
    
    with torch.no_grad():
        topo_features, topo_data = topo_module(point_cloud)
    
    # Visualize Betti curves
    betti_curves = topo_data['betti_curves']
    
    # Plot Betti curves if matplotlib is available
    try:
        plt_betti = visualize_betti_curves(
            betti_curves[0],
            title=f"Betti Curves - Test"
        )
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("Tiny topological feature extraction test completed successfully!")
    print(f"Topological feature shape: {topo_features.shape}")
    
    # Estimate model size
    total_params = sum(p.numel() for p in topo_module.parameters())
    print(f"Total parameters: {total_params}")