#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Correlation Manifold (PCM) - Topological Feature Extraction Component

This file implements the topological analysis component of the PCM architecture
for AI-generated image detection. It analyzes the topological structure of the
phase correlation manifold using persistent homology.

Key components:
1. Differentiable Persistent Homology: Computes topological features that can be used in backpropagation
2. Betti Numbers: Counts the number of connected components, holes, and voids in the manifold
3. Persistence Diagrams: Captures the birth and death of topological features

Topological features capture the "shape" of the phase correlation patterns, which
is fundamentally different between natural and AI-generated images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
import gudhi as gd
from gudhi.representations import PersistenceImage
import pytorch_lightning as pl


class PersistentHomologyLayer(nn.Module):
    """Computes persistent homology features from point cloud data"""
    
    def __init__(self, max_edge_length=2.0, num_filtrations=50, max_dimension=2):
        super(PersistentHomologyLayer, self).__init__()
        
        self.max_edge_length = max_edge_length
        self.num_filtrations = num_filtrations
        self.max_dimension = max_dimension
        
        # Learnable parameters for filtration values
        self.filtration_values = nn.Parameter(
            torch.linspace(0, max_edge_length, num_filtrations)
        )
    
    def forward(self, x):
        """
        Compute persistence diagrams from point cloud data
        
        Args:
            x (torch.Tensor): Point cloud data [batch_size, num_points, features]
            
        Returns:
            dict: Dictionary containing persistence diagrams and Betti numbers
        """
        batch_size = x.size(0)
        
        # Detach and convert to numpy for gudhi
        x_np = x.detach().cpu().numpy()
        
        # Prepare output containers
        all_diagrams = []
        all_betti_curves = []
        
        # Process each batch item
        for i in range(batch_size):
            # Compute pairwise distances
            point_cloud = x_np[i]
            distances = squareform(pdist(point_cloud))
            
            # Create Vietoris-Rips complex
            rips_complex = gd.RipsComplex(distance_matrix=distances)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension+1)
            
            # Compute persistence
            simplex_tree.compute_persistence()
            
            # Get persistence diagram
            diagram = simplex_tree.persistence_diagram()
            
            # Convert to tensor format and store
            diagram_tensor = self._convert_diagram_to_tensor(diagram)
            all_diagrams.append(diagram_tensor)
            
            # Compute Betti curves
            filtration_values = self.filtration_values.detach().cpu().numpy()
            betti_curves = self._compute_betti_curves(simplex_tree, filtration_values)
            all_betti_curves.append(betti_curves)
        
        # Stack batch results
        diagrams_tensor = torch.stack(all_diagrams, dim=0)
        betti_curves_tensor = torch.stack(all_betti_curves, dim=0)
        
        return {
            'persistence_diagrams': diagrams_tensor,
            'betti_curves': betti_curves_tensor
        }
    
    def _convert_diagram_to_tensor(self, diagram):
        """Convert GUDHI persistence diagram to torch tensor"""
        # Create empty tensors for each homology dimension
        max_points = 100  # Maximum number of points to keep
        diagram_tensors = []
        
        for dim in range(self.max_dimension + 1):
            # Extract points for this dimension
            dim_points = np.array([p[1] for p in diagram if p[0] == dim and p[1][1] < float('inf')])
            
            if len(dim_points) > 0:
                # Sort by persistence (death - birth)
                persistence = dim_points[:, 1] - dim_points[:, 0]
                sorted_indices = np.argsort(-persistence)  # Sort in descending order
                dim_points = dim_points[sorted_indices]
                
                # Truncate to max_points
                if len(dim_points) > max_points:
                    dim_points = dim_points[:max_points]
                
                # Pad if necessary
                if len(dim_points) < max_points:
                    pad = np.zeros((max_points - len(dim_points), 2))
                    dim_points = np.vstack([dim_points, pad])
            else:
                # No points in this dimension
                dim_points = np.zeros((max_points, 2))
            
            # Convert to tensor
            diagram_tensors.append(torch.tensor(dim_points, dtype=torch.float32))
        
        # Concatenate all dimensions
        return torch.stack(diagram_tensors, dim=0)
    
    def _compute_betti_curves(self, simplex_tree, filtration_values):
        """Compute Betti curves for each dimension"""
        betti_curves = []
        
        for dim in range(self.max_dimension + 1):
            betti_numbers = []
            
            for epsilon in filtration_values:
                # Count number of features alive at this filtration value
                betti = simplex_tree.persistent_betti_numbers(0, epsilon)
                betti_number = betti[dim] if dim < len(betti) else 0
                betti_numbers.append(betti_number)
            
            betti_curves.append(betti_numbers)
        
        return torch.tensor(betti_curves, dtype=torch.float32)


class PersistenceImageLayer(nn.Module):
    """Converts persistence diagrams to persistence images"""
    
    def __init__(self, resolution=(20, 20), sigma=0.1, weight_function=None):
        super(PersistenceImageLayer, self).__init__()
        
        self.resolution = resolution
        self.sigma = sigma
        
        # Default weight function: linear weighting by persistence
        if weight_function is None:
            self.weight_function = lambda x: x[:, 1] - x[:, 0]  # death - birth
        else:
            self.weight_function = weight_function
    
    def forward(self, persistence_diagrams):
        """
        Convert persistence diagrams to persistence images
        
        Args:
            persistence_diagrams (torch.Tensor): [batch_size, num_dimensions, max_points, 2]
            
        Returns:
            torch.Tensor: Persistence images [batch_size, num_dimensions, resolution[0], resolution[1]]
        """
        batch_size, num_dimensions, max_points, _ = persistence_diagrams.shape
        
        # Prepare output container
        persistence_images = []
        
        # Process each batch and dimension
        for i in range(batch_size):
            dim_images = []
            
            for dim in range(num_dimensions):
                # Get diagram points
                diagram = persistence_diagrams[i, dim]
                
                # Filter out padding (points with birth=death=0)
                non_zero_mask = ~torch.all(diagram == 0, dim=1)
                non_zero_points = diagram[non_zero_mask]
                
                if len(non_zero_points) > 0:
                    # Compute weight values
                    weights = self.weight_function(non_zero_points)
                    
                    # Create persistence image
                    pi = self._diagram_to_image(non_zero_points, weights)
                else:
                    # Empty diagram
                    pi = torch.zeros(self.resolution, dtype=torch.float32)
                
                dim_images.append(pi)
            
            # Stack dimensions
            persistence_images.append(torch.stack(dim_images, dim=0))
        
        # Stack batch
        return torch.stack(persistence_images, dim=0)
    
    def _diagram_to_image(self, diagram, weights):
        """Convert a single persistence diagram to an image"""
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        
        # Create mesh grid for the persistence image
        x_range = torch.linspace(0, 1, self.resolution[0])
        y_range = torch.linspace(0, 1, self.resolution[1])
        xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
        
        # Initialize image
        image = torch.zeros(self.resolution, dtype=torch.float32)
        
        # Add Gaussian for each persistence point
        for i in range(len(births)):
            # Transform persistence point to image coordinates
            birth = births[i]
            death = deaths[i]
            persistence = death - birth
            
            if persistence > 0:
                # Use birth and persistence as coordinates
                x = birth
                y = persistence
                
                # Apply Gaussian kernel
                gaussian = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
                
                # Apply weight
                weighted_gaussian = gaussian * weights[i]
                
                # Add to image
                image += weighted_gaussian
        
        return image


class TopologicalFeatureExtraction(nn.Module):
    """Complete topological feature extraction module"""
    
    def __init__(self, 
                 input_dim=32,           # Input feature dimension
                 hidden_dim=64,          # Hidden dimension
                 output_dim=32,          # Output feature dimension
                 max_edge_length=2.0,    # Maximum edge length for filtration
                 num_filtrations=50,     # Number of filtration values
                 max_dimension=2,        # Maximum homology dimension
                 persistence_image_resolution=(20, 20)  # Resolution of persistence images
                ):
        super(TopologicalFeatureExtraction, self).__init__()
        
        # Persistence homology layer
        self.persistent_homology = PersistentHomologyLayer(
            max_edge_length=max_edge_length,
            num_filtrations=num_filtrations,
            max_dimension=max_dimension
        )
        
        # Persistence image layer
        self.persistence_image = PersistenceImageLayer(
            resolution=persistence_image_resolution
        )
        
        # Process Betti curves
        self.betti_processor = nn.Sequential(
            nn.Linear(num_filtrations, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Process persistence images
        self.image_processor = nn.Sequential(
            nn.Conv2d(max_dimension + 1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            
            
            
            
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Combine features
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        """
        Extract topological features from the manifold
        
        Args:
            x (torch.Tensor): Manifold points [batch_size, num_points, feature_dim]
            
        Returns:
            torch.Tensor: Topological features [batch_size, output_dim]
        """
        # Compute persistent homology
        homology_data = self.persistent_homology(x)
        persistence_diagrams = homology_data['persistence_diagrams']
        betti_curves = homology_data['betti_curves']
        
        # Convert persistence diagrams to images
        persistence_images = self.persistence_image(persistence_diagrams)
        
        # Process Betti curves
        # Flatten for each batch item (concatenate dimensions)
        batch_size = betti_curves.size(0)
        num_dims = betti_curves.size(1)
        num_filtrations = betti_curves.size(2)
        
        betti_features = []
        for dim in range(num_dims):
            dim_betti = betti_curves[:, dim, :]
            dim_features = self.betti_processor(dim_betti)
            betti_features.append(dim_features)
        
        # Combine all dimensions
        combined_betti_features = sum(betti_features) / len(betti_features)
        
        # Process persistence images
        image_features = self.image_processor(persistence_images)
        
        # Combine features
        combined_features = torch.cat([combined_betti_features, image_features], dim=1)
        output_features = self.output_layer(combined_features)
        
        return output_features, {
            'persistence_diagrams': persistence_diagrams,
            'betti_curves': betti_curves,
            'persistence_images': persistence_images
        }


class TopologicalLoss(nn.Module):
    """Loss function based on topological features"""
    
    def __init__(self, lambda_wasserstein=1.0):
        super(TopologicalLoss, self).__init__()
        self.lambda_wasserstein = lambda_wasserstein
    
    def forward(self, features1, features2, diagrams1, diagrams2):
        """
        Compute topological loss between two sets of features
        
        Args:
            features1, features2: Feature vectors [batch_size, feature_dim]
            diagrams1, diagrams2: Persistence diagrams 
                                 [batch_size, num_dimensions, max_points, 2]
            
        Returns:
            torch.Tensor: Loss value
        """
        # Feature distance loss
        feature_loss = F.mse_loss(features1, features2)
        
        # Wasserstein distance between persistence diagrams
        wasserstein_distance = self._wasserstein_distance(diagrams1, diagrams2)
        
        # Total loss
        total_loss = feature_loss + self.lambda_wasserstein * wasserstein_distance
        
        return total_loss
    
    def _wasserstein_distance(self, diagrams1, diagrams2):
        """
        Compute 2-Wasserstein distance between persistence diagrams
        
        This is a differentiable approximation of the Wasserstein distance
        """
        batch_size, num_dims, num_points, _ = diagrams1.shape
        
        total_distance = 0
        
        for b in range(batch_size):
            for d in range(num_dims):
                # Get points for this dimension
                points1 = diagrams1[b, d]
                points2 = diagrams2[b, d]
                
                # Filter out zeros (padding)
                non_zero1 = ~torch.all(points1 == 0, dim=1)
                non_zero2 = ~torch.all(points2 == 0, dim=1)
                
                active_points1 = points1[non_zero1]
                active_points2 = points2[non_zero2]
                
                # If either diagram is empty, skip
                if len(active_points1) == 0 or len(active_points2) == 0:
                    continue
                
                # Compute persistent features
                persistence1 = active_points1[:, 1] - active_points1[:, 0]
                persistence2 = active_points2[:, 1] - active_points2[:, 0]
                
                # Sort by persistence
                _, indices1 = torch.sort(persistence1, descending=True)
                _, indices2 = torch.sort(persistence2, descending=True)
                
                sorted_points1 = active_points1[indices1]
                sorted_points2 = active_points2[indices2]
                
                # Take top points from each
                k = min(len(sorted_points1), len(sorted_points2), 10)
                top_points1 = sorted_points1[:k]
                top_points2 = sorted_points2[:k]
                
                # Compute distance between corresponding points
                if len(top_points1) == len(top_points2):
                    point_distances = torch.sum((top_points1 - top_points2) ** 2, dim=1)
                else:
                    # Pad shorter list with zeros
                    if len(top_points1) < len(top_points2):
                        pad = torch.zeros((len(top_points2) - len(top_points1), 2), 
                                          device=top_points1.device)
                        top_points1 = torch.cat([top_points1, pad], dim=0)
                    else:
                        pad = torch.zeros((len(top_points1) - len(top_points2), 2), 
                                          device=top_points2.device)
                        top_points2 = torch.cat([top_points2, pad], dim=0)
                    
                    point_distances = torch.sum((top_points1 - top_points2) ** 2, dim=1)
                
                # Add to total distance
                dim_distance = torch.mean(point_distances)
                total_distance += dim_distance
        
        # Normalize by batch size and number of dimensions
        avg_distance = total_distance / (batch_size * num_dims)
        
        return avg_distance


def extract_topological_features(manifold_features, device='cpu'):
    """Process manifold features through the topological feature extraction module"""
    # Create topological feature extraction module
    topo_module = TopologicalFeatureExtraction().to(device)
    
    # Process manifold features
    with torch.no_grad():
        topo_features, topo_data = topo_module(manifold_features)
    
    print(f"Manifold feature shape: {manifold_features.shape}")
    print(f"Topological feature shape: {topo_features.shape}")
    
    return topo_features, topo_data


def visualize_persistence_diagram(persistence_diagram, dimension=1, title="Persistence Diagram"):
    """Visualize a persistence diagram for a specific homology dimension"""
    import matplotlib.pyplot as plt
    
    # Extract points for the given dimension
    points = persistence_diagram[dimension]
    
    # Filter out zeros (padding)
    non_zero = ~torch.all(points == 0, dim=1)
    active_points = points[non_zero].cpu().numpy()
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(active_points[:, 0], active_points[:, 1], alpha=0.6)
    
    # Add diagonal line
    max_val = max(active_points.max() if len(active_points) > 0 else 1, 1)
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
    
    plt.xlabel('Birth')
    plt.ylabel('Death')
    plt.title(f"{title} - Dimension {dimension}")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    return plt


def visualize_betti_curves(betti_curves, filtration_values=None, title="Betti Curves"):
    """Visualize Betti curves for all dimensions"""
    import matplotlib.pyplot as plt
    
    num_dims = betti_curves.shape[0]
    
    if filtration_values is None:
        filtration_values = torch.linspace(0, 1, betti_curves.shape[1])
    
    plt.figure(figsize=(10, 6))
    
    for dim in range(num_dims):
        plt.plot(filtration_values.cpu().numpy(), 
                 betti_curves[dim].cpu().numpy(), 
                 label=f"Dimension {dim}")
    
    plt.xlabel('Filtration Value')
    plt.ylabel('Betti Number')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt


def visualize_persistence_image(persistence_image, dimension=1, title="Persistence Image"):
    """Visualize a persistence image for a specific homology dimension"""
    import matplotlib.pyplot as plt
    
    # Extract image for the given dimension
    image = persistence_image[dimension].cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='viridis', origin='lower')
    plt.colorbar(label='Intensity')
    
    plt.xlabel('Birth')
    plt.ylabel('Persistence')
    plt.title(f"{title} - Dimension {dimension}")
    
    return plt


if __name__ == "__main__":
    # If this file is run directly, perform a quick test
    import os
    import torch
    from feature_extraction import FeatureExtractionNetwork, PhaseCorrelationTensorComputation, ImagePreprocessor
    from manifold_learning import ManifoldLearningModule
    from PIL import Image
    import matplotlib.pyplot as plt
    
    # Create directories if they don't exist
    os.makedirs('sample_images', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create or load a sample image
    sample_img_path = 'sample_images/sample_natural.jpg'
    if not os.path.exists(sample_img_path):
        # Create a random image for testing
        img = np.random.rand(256, 256, 3)
        Image.fromarray((img * 255).astype(np.uint8)).save(sample_img_path)
    
    # Load and preprocess the image
    img = Image.open(sample_img_path)
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    processed_img = preprocessor.process(img)
    
    # Add batch dimension
    processed_img = processed_img.unsqueeze(0).to(device)
    
    # Feature extraction
    feature_network = FeatureExtractionNetwork().to(device)
    tensor_computer = PhaseCorrelationTensorComputation().to(device)
    
    with torch.no_grad():
        features, _ = feature_network(processed_img)
        phase_tensor = tensor_computer(features)
    
    # Manifold learning
    manifold_module = ManifoldLearningModule().to(device)
    
    with torch.no_grad():
        manifold_features, _ = manifold_module(phase_tensor)
    
    # Create point cloud data for topological analysis
    # For demonstration, reshape manifold features to create a point cloud
    num_points = 100
    feature_dim = manifold_features.size(1)
    
    # Expand manifold features to create a synthetic point cloud
    point_cloud = manifold_features.unsqueeze(1).expand(-1, num_points, -1)
    # Add some noise to create a more interesting point cloud
    noise = torch.randn(1, num_points, feature_dim, device=device) * 0.1
    point_cloud = point_cloud + noise
    
    # Topological feature extraction
    topo_module = TopologicalFeatureExtraction().to(device)
    
    with torch.no_grad():
        topo_features, topo_data = topo_module(point_cloud)
    
    # Visualize results
    persistence_diagrams = topo_data['persistence_diagrams']
    betti_curves = topo_data['betti_curves']
    persistence_images = topo_data['persistence_images']
    
    # Plot and save
    for dim in range(min(3, persistence_diagrams.size(1))):
        # Plot persistence diagram
        plt_diag = visualize_persistence_diagram(
            persistence_diagrams[0], dimension=dim, 
            title=f"Persistence Diagram - Sample Image"
        )
        plt_diag.savefig(f'output/persistence_diagram_dim{dim}.png')
        
        # Plot persistence image
        plt_img = visualize_persistence_image(
            persistence_images[0], dimension=dim,
            title=f"Persistence Image - Sample Image"
        )
        plt_img.savefig(f'output/persistence_image_dim{dim}.png')
    
    # Plot Betti curves
    plt_betti = visualize_betti_curves(
        betti_curves[0],
        title=f"Betti Curves - Sample Image"
    )
    plt_betti.savefig('output/betti_curves.png')
    
    print("Topological feature extraction test completed successfully!")
    print(f"Topological feature shape: {topo_features.shape}")