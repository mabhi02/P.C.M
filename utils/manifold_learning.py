#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Correlation Manifold (PCM) - Manifold Learning Component

This file implements the manifold learning component of the PCM architecture 
for AI-generated image detection. It maps high-dimensional phase correlation
tensors to a lower-dimensional manifold space.

Key components:
1. Variational Autoencoder (VAE): Compresses tensors into a compact latent space
2. Graph Neural Network: Models relationships between points in the manifold space
3. Latent Space Representation: The final low-dimensional representation

The manifold learning module is crucial for capturing the topological structure
of phase correlations, which differs between natural and AI-generated images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
import numpy as np


class Encoder(nn.Module):
    """Encoder part of the Variational Autoencoder with dynamic input handling"""
    
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=32, spatial_dim=16):
        super(Encoder, self).__init__()
        
        # Initial dimensionality reduction
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # MLP for latent space projection - will be initialized dynamically
        self.fc_mu = None
        self.fc_logvar = None
        self.latent_dim = latent_dim
    
    def forward(self, x):
        # Convolutional encoding
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten for FC layers
        x = x.view(x.size(0), -1)
        flatten_dim = x.size(1)
        
        # Dynamically create the projection layers if they don't exist or if size doesn't match
        if self.fc_mu is None or self.fc_mu.in_features != flatten_dim:
            self.fc_mu = nn.Linear(flatten_dim, self.latent_dim).to(x.device)
            self.fc_logvar = nn.Linear(flatten_dim, self.latent_dim).to(x.device)
        
        # Generate latent parameters
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Decoder(nn.Module):
    """Decoder part of the Variational Autoencoder with dynamic input handling"""
    
    def __init__(self, latent_dim=32, hidden_dim=256, output_dim=128):
        super(Decoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # These will be initialized in the forward pass once we know the output dimensions
        self.fc = None
        self.deconv1 = None
        self.bn1 = None
        self.deconv2 = None
        self.bn2 = None
        self.deconv3 = None
        
        # Will be determined in forward pass
        self.spatial_dim = None
        self.output_height = None
        self.output_width = None
    
    def forward(self, z, output_size=None):
        # Default output shape if not provided
        if output_size is None:
            # Assume square output with default dimensions
            self.output_height = 16
            self.output_width = 16
        else:
            self.output_height, self.output_width = output_size
        
        # Determine spatial dimension after three stride-2 convolutions
        self.spatial_dim = self.output_height // 8
        if self.spatial_dim < 1:
            self.spatial_dim = 1
        
        # Initialize layers if they don't exist
        if self.fc is None:
            self.fc = nn.Linear(self.latent_dim, self.hidden_dim * self.spatial_dim * self.spatial_dim).to(z.device)
            self.deconv1 = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1).to(z.device)
            self.bn1 = nn.BatchNorm2d(self.hidden_dim).to(z.device)
            self.deconv2 = nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1).to(z.device)
            self.bn2 = nn.BatchNorm2d(self.hidden_dim).to(z.device)
            self.deconv3 = nn.ConvTranspose2d(self.hidden_dim, self.output_dim, kernel_size=3, stride=2, padding=1, output_padding=1).to(z.device)
        
        # Project and reshape to spatial feature map
        x = self.fc(z)
        x = x.view(x.size(0), self.hidden_dim, self.spatial_dim, self.spatial_dim)
        
        # Deconvolutional decoding with dynamic resizing
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        
        # Resize to match output dimensions if necessary
        if x.size(2) != self.output_height or x.size(3) != self.output_width:
            x = F.interpolate(x, size=(self.output_height, self.output_width), mode='bilinear', align_corners=False)
        
        return x


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for manifold learning"""
    
    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=32, spatial_dim=16):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, spatial_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, spatial_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from latent Gaussian distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        # Encode input to latent space parameters
        mu, logvar = self.encoder(x)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        # Decode latent vector to reconstruction
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar, z
    
    def encode(self, x):
        """Encode input to latent space (for inference)"""
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z):
        """Decode latent vector to output (for inference)"""
        with torch.no_grad():
            reconstruction = self.decoder(z)
        return reconstruction


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer for the GNN component"""
    
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention mechanism
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x, adj):
        """
        x: Node features [batch_size, num_nodes, in_features]
        adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes = x.size(0), x.size(1)
        
        # Linear transformation
        Wh = self.W(x)  # [batch_size, num_nodes, out_features]
        
        # Compute attention scores
        # Prepare concatenated node pairs
        a_input = torch.cat([Wh.repeat(1, 1, num_nodes).view(batch_size, num_nodes * num_nodes, self.out_features),
                             Wh.repeat(1, num_nodes, 1)], dim=2)
        a_input = a_input.view(batch_size, num_nodes, num_nodes, 2 * self.out_features)
        
        # Compute attention coefficients
        e = self.leakyrelu(self.a(a_input).squeeze(3))
        
        # Mask attention coefficients based on adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to node features
        h_prime = torch.bmm(attention, Wh)
        
        return h_prime


class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for modeling relationships in manifold space"""
    
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32, num_layers=2, dropout=0.6):
        super(GraphNeuralNetwork, self).__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                hidden_dim if i == 0 else hidden_dim,
                hidden_dim if i < num_layers - 1 else output_dim,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else output_dim)
            for i in range(num_layers)
        ])
        
        self.dropout = dropout
    
    def build_adjacency(self, z, k=8):
        """Build adjacency matrix based on k-nearest neighbors in latent space"""
        batch_size, num_nodes = z.size(0), z.size(1)
        
        # Compute pairwise distances
        z_expanded_1 = z.unsqueeze(2)
        z_expanded_2 = z.unsqueeze(1)
        distances = torch.sum((z_expanded_1 - z_expanded_2)**2, dim=3)
        
        # Get k-nearest neighbors (excluding self)
        self_mask = torch.eye(num_nodes, device=z.device).unsqueeze(0).expand(batch_size, -1, -1)
        distances = distances + self_mask * 1e6  # Add large value to self-connections
        
        # Get top-k smallest distances
        _, topk_indices = torch.topk(distances, k=k, dim=2, largest=False)
        
        # Build adjacency matrix
        adj = torch.zeros_like(distances)
        for b in range(batch_size):
            for i in range(num_nodes):
                adj[b, i, topk_indices[b, i]] = 1.0
        
        # Make symmetric
        adj = torch.maximum(adj, adj.transpose(1, 2))
        
        return adj
    
    def forward(self, z):
        """
        z: Latent vectors [batch_size, num_nodes, input_dim]
        """
        batch_size, num_nodes = z.size(0), z.size(1)
        
        # Build adjacency matrix from latent vectors
        adj = self.build_adjacency(z)
        
        # Initial projection
        x = self.input_proj(z)
        
        # Apply GAT layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, adj)
            x_new = norm(x_new)
            
            if i < len(self.gat_layers) - 1:
                x_new = F.relu(x_new)
            
            # Apply dropout for regularization
            x_new = F.dropout(x_new, self.dropout, training=self.training)
            
            # Residual connection
            x = x_new + x if x.size() == x_new.size() else x_new
        
        return x


# Fixed ManifoldLearningModule to be adaptive to input sizes
class ManifoldLearningModule(nn.Module):
    """Fixed manifold learning module with debugging"""
    
    def __init__(self, 
                 input_dim=128,          # Input tensor channels
                 hidden_dim=256,         # Hidden dimensions
                 latent_dim=32,          # Latent space dimension
                 gnn_hidden_dim=64       # GNN hidden dimension
                ):
        super(ManifoldLearningModule, self).__init__()
        
        # Simple encoder network with specific dimensions for your case
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # We'll create these layers in the forward pass after debugging
        self.fc_mu = None
        self.fc_logvar = None
        
        # Simple decoder (we'll initialize in forward after debugging)
        self.fc_decoder = None
        self.decoder = None
        
        # Simplified GNN component
        self.gnn = nn.Linear(latent_dim, latent_dim)
        
        # Final projection and normalization
        self.projection = nn.Linear(latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        
        # Save dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: sample from latent Gaussian distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        """
        x: Phase correlation tensor [batch_size, input_dim, height, width]
        """
        # Debug input shape
        print(f"DEBUG - Input shape: {x.shape}")
        
        try:
            # Forward pass through encoder parts separately for debugging
            # First conv layer
            x1 = self.encoder[0](x)
            print(f"DEBUG - After first conv: {x1.shape}")
            x1 = self.encoder[1](x1)  # BatchNorm
            x1 = self.encoder[2](x1)  # ReLU
            
            # Second conv layer
            x2 = self.encoder[3](x1)
            print(f"DEBUG - After second conv: {x2.shape}")
            x2 = self.encoder[4](x2)  # BatchNorm
            x2 = self.encoder[5](x2)  # ReLU
            
            # Third conv layer
            x3 = self.encoder[6](x2)
            print(f"DEBUG - After third conv: {x3.shape}")
            x3 = self.encoder[7](x3)  # BatchNorm
            x3 = self.encoder[8](x3)  # ReLU
            
            # Flatten
            encoded = torch.flatten(x3, start_dim=1)
            print(f"DEBUG - After flatten: {encoded.shape}")
            
            # Initialize fc layers if needed based on flattened shape
            if self.fc_mu is None:
                flat_dim = encoded.size(1)
                print(f"DEBUG - Creating FC layers with input dim: {flat_dim}")
                
                # Create linear layers with correct dimensions
                self.fc_mu = nn.Linear(flat_dim, self.latent_dim).to(encoded.device)
                self.fc_logvar = nn.Linear(flat_dim, self.latent_dim).to(encoded.device)
                
                # Create decoder components with matching dimensions
                self.fc_decoder = nn.Linear(self.latent_dim, flat_dim).to(encoded.device)
                
                # Calculate sizes for decoder
                print(f"DEBUG - x3 shape before flatten: {x3.shape}")
                num_channels = x3.size(1)
                height = x3.size(2)
                width = x3.size(3)
                
                # Decoder convolutional path
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(num_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.hidden_dim),
                    nn.ReLU(),
                    nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(self.hidden_dim),
                    nn.ReLU(),
                    nn.ConvTranspose2d(self.hidden_dim, self.input_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
                ).to(encoded.device)
                
                # Save dimensions for unflatten operation
                self.decoder_unflatten_dims = (num_channels, height, width)
            
            # Generate latent parameters
            mu = self.fc_mu(encoded)
            logvar = self.fc_logvar(encoded)
            
            # Sample latent vector
            z = self.reparameterize(mu, logvar)
            
            # Simple GNN processing
            z_gnn = self.gnn(z)
            
            # Final projection with residual connection
            z_final = self.projection(z_gnn) + z
            z_final = self.layer_norm(z_final)
            
            # Return manifold representation and VAE outputs
            return z_final, (mu, logvar, z)
            
        except Exception as e:
            # Detailed error reporting
            print(f"DEBUG - Error in forward pass: {e}")
            print(f"DEBUG - Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        if self.fc_decoder is None:
            print("DEBUG - Decoder not initialized yet!")
            # Return dummy tensor
            return torch.zeros((z.size(0), self.input_dim, 16, 16), device=z.device)
        
        try:
            # Decode the latent vector
            print(f"DEBUG - Latent z shape: {z.shape}")
            x = self.fc_decoder(z)
            print(f"DEBUG - After fc_decoder: {x.shape}")
            
            # Unflatten
            x = x.view(x.size(0), *self.decoder_unflatten_dims)
            print(f"DEBUG - After unflatten: {x.shape}")
            
            # Convolutional decoding
            reconstruction = self.decoder(x)
            print(f"DEBUG - Final reconstruction: {reconstruction.shape}")
            
            return reconstruction
        except Exception as e:
            print(f"DEBUG - Error in decode: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy tensor
            return torch.zeros((z.size(0), self.input_dim, 16, 16), device=z.device)
    
    def get_loss(self, x, x_recon, mu, logvar, beta=1.0):
        """Compute VAE loss (reconstruction + KL divergence)"""
        # Reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def process_phase_tensors(phase_tensors, device='cpu'):
    """Process phase correlation tensors through the manifold learning module"""
    # Create manifold learning module
    manifold_module = ManifoldLearningModule().to(device)
    
    # Process tensors
    with torch.no_grad():
        manifold_features, vae_outputs = manifold_module(phase_tensors)
    
    print(f"Phase tensor shape: {phase_tensors.shape}")
    print(f"Manifold feature shape: {manifold_features.shape}")
    
    return manifold_features


def train_manifold_module(dataloader, num_epochs=10, lr=1e-4, device='cpu'):
    """Train the manifold learning module on a dataset of phase correlation tensors"""
    # Create manifold learning module
    manifold_module = ManifoldLearningModule().to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(manifold_module.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        
        for batch in dataloader:
            # Move batch to device
            phase_tensors = batch.to(device)
            
            # Forward pass
            _, (mu, logvar, z) = manifold_module(phase_tensors)
            # Reconstruct input for VAE loss
            x_recon = manifold_module.vae.decode(z)
            
            # Compute loss
            loss, recon_loss, kl_loss = manifold_module.get_loss(phase_tensors, x_recon, mu, logvar)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
        
        # Print epoch statistics
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = recon_loss_sum / len(dataloader)
        avg_kl_loss = kl_loss_sum / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, "
              f"Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")
    
    return manifold_module


if __name__ == "__main__":
    # If this file is run directly, perform a quick test
    import os
    from feature_extraction import FeatureExtractionNetwork, PhaseCorrelationTensorComputation, ImagePreprocessor
    from PIL import Image
    
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
    
    # Extract features using the feature extraction network
    feature_network = FeatureExtractionNetwork().to(device)
    tensor_computer = PhaseCorrelationTensorComputation().to(device)
    
    # Forward pass
    with torch.no_grad():
        features, _ = feature_network(processed_img)
        phase_tensor = tensor_computer(features)
    
    # Process through manifold learning module
    manifold_features = process_phase_tensors(phase_tensor, device)
    
    print("Manifold learning test completed successfully!")