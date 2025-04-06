#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model.py

Consolidated PCM (Phase Correlation Manifold) code for AI-generated image detection.
This file combines:
1) Enhanced Feature Extraction (Spatial, Fourier, and Multi-Scale branches)
2) Manifold Learning Module (Variational-style + GNN, or simpler version)
3) Tiny Topological Feature Extraction
4) Classification Network (with optional attention, etc.)
5) Utility classes and final pipeline instantiation

Author: Your Name
Date: 2025-04-05
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import random

# If you use GNN or Gudhi for topological computations, uncomment as needed:
# import torch_geometric.nn as gnn
# import gudhi as gd
# from scipy.spatial.distance import pdist, squareform


###############################################################################
#                               IMAGE PREPROCESSOR
###############################################################################
class ImagePreprocessor:
    """Class for image preprocessing operations"""
    def __init__(self, target_size=(384, 384)):  # Adjust default as desired
        self.target_size = target_size
        
    def normalize(self, image):
        """Normalize image to [0, 1] range"""
        if isinstance(image, np.ndarray):
            return image / 255.0
        elif isinstance(image, torch.Tensor):
            return image / 255.0 if image.max() > 1.0 else image
        else:
            raise TypeError("Image must be numpy array or torch tensor")
    
    def resize(self, image):
        """Resize image to target size"""
        if isinstance(image, np.ndarray):
            return cv2.resize(image, self.target_size)
        elif isinstance(image, torch.Tensor):
            # If PyTorch tensor, use F.interpolate
            return F.interpolate(image.unsqueeze(0), size=self.target_size).squeeze(0)
        elif isinstance(image, Image.Image):
            return image.resize(self.target_size)
        else:
            raise TypeError("Image must be numpy array, torch tensor, or PIL Image")
    
    def to_tensor(self, image):
        """Convert to PyTorch tensor"""
        import torchvision
        if isinstance(image, np.ndarray):
            # Handle grayscale images
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
            # Convert HWC to CHW format
            image = image.transpose((2, 0, 1)).astype(np.float32)
            return torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, Image.Image):
            return torchvision.transforms.ToTensor()(image)
        else:
            raise TypeError("Image must be numpy array, torch tensor, or PIL Image")
    
    def process(self, image):
        """Apply all preprocessing steps: resize, normalize, to_tensor"""
        image = self.resize(image)
        image = self.normalize(image)
        image = self.to_tensor(image)
        return image


###############################################################################
#                       ENHANCED FEATURE EXTRACTION
###############################################################################
class BasicBlock(nn.Module):
    """Basic block for enhanced ResNet architecture"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out


class EnhancedSpatialBranch(nn.Module):
    """Reduced ResNet-based architecture for spatial features with ~5M parameters"""
    def __init__(self, input_channels=3, output_channels=128):
        super(EnhancedSpatialBranch, self).__init__()
        
        # Initial layers - reduced initial channels from 64 to 32
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks - reduced number of blocks and channels
        self.layer1 = self._make_layer(BasicBlock, 32, 32, blocks=2, stride=1)      # Was 64, 64, blocks=3
        self.layer2 = self._make_layer(BasicBlock, 32, 64, blocks=2, stride=2)      # Was 64, 128, blocks=4
        self.layer3 = self._make_layer(BasicBlock, 64, 128, blocks=2, stride=2)     # Was 128, 256, blocks=6
        self.layer4 = self._make_layer(BasicBlock, 128, 256, blocks=2, stride=2)    # Was 256, 512, blocks=3
        
        # Attention mechanism - reduced inner dimension from 128 to 64
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),                                      # Was 512, 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=1),                                      # Was 128, 512
            nn.Sigmoid()
        )
        
        # Output projection
        self.projection = nn.Conv2d(256, output_channels, kernel_size=1)           # Was 512, output_channels
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Projection
        output = self.projection(attended_features)
        return output


class EnhancedFFTLayer(nn.Module):
    """Enhanced layer that computes FFT and separates amplitude and phase"""
    def __init__(self):
        super(EnhancedFFTLayer, self).__init__()
    
    def forward(self, x):
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        log_amplitude = torch.log(amplitude + 1e-10)
        output = torch.cat([log_amplitude, phase], dim=1)
        return output


class EnhancedFourierBranch(nn.Module):
    """Reduced branch for Fourier domain feature extraction"""
    def __init__(self, input_channels=3, output_channels=128):
        super(EnhancedFourierBranch, self).__init__()
        
        self.fft_layer = EnhancedFFTLayer()
        
        # Reduced channel counts throughout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels * 2, 32, kernel_size=3, padding=1),        # Was 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),              # Was 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),             # Was 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, padding=1, stride=2),            # Was 512
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, output_channels, kernel_size=1)
        )
        
        # Reduced channel counts in phase processor
        self.phase_processor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),            # Was 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),              # Was 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # Reduced intermediate channels
        self.output_layer = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1), # Reduced dimension directly
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        fft_output = self.fft_layer(x)
        conv_features = self.conv_layers(fft_output)
        
        phase = fft_output[:, x.size(1):]
        phase_features = self.phase_processor(phase)
        
        if phase_features.size()[2:] != conv_features.size()[2:]:
            phase_features = F.interpolate(phase_features, 
                                           size=conv_features.size()[2:], 
                                           mode='bilinear', align_corners=False)
        
        combined = torch.cat([conv_features, phase_features], dim=1)
        output = self.output_layer(combined)
        return output


class EnhancedPyramidLayer(nn.Module):
    """Enhanced layer that creates a Gaussian pyramid"""
    def __init__(self, levels=3):
        super(EnhancedPyramidLayer, self).__init__()
        self.levels = levels
        self.gaussian_kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        return kernel_2d
    
    def _gaussian_blur(self, x):
        kernel = self.gaussian_kernel.repeat(x.size(1), 1, 1, 1).to(x.device)
        padding = (self.gaussian_kernel.size(2) - 1) // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))
    
    def forward(self, x):
        pyramid = [x]
        current = x
        for _ in range(self.levels - 1):
            blurred = self._gaussian_blur(current)
            downsampled = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            pyramid.append(downsampled)
            current = downsampled
        return pyramid


class EnhancedMultiScaleBranch(nn.Module):
    """Reduced branch for multi-scale feature extraction"""
    def __init__(self, input_channels=3, output_channels=128, pyramid_levels=3):
        super(EnhancedMultiScaleBranch, self).__init__()
        
        self.pyramid_layer = EnhancedPyramidLayer(levels=pyramid_levels)
        
        # Reduced channel counts and simplified convolution layers
        self.level_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),              # Was 64
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),                          # Was 128
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 96, kernel_size=3, padding=1, stride=2),                # Was 256
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
            ) for _ in range(pyramid_levels)
        ])
        
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=False)
            for i in range(1, pyramid_levels)
        ])
        
        # Reduced channel count in output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(96 * pyramid_levels, 192, kernel_size=3, padding=1),           # Was 256*pyramid_levels, 512
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, output_channels, kernel_size=1)
        )
    
    def forward(self, x):
        pyramid = self.pyramid_layer(x)
        processed_levels = []
        
        for i, level in enumerate(pyramid):
            processed = self.level_convs[i](level)
            processed_levels.append(processed)
        
        for i in range(1, len(processed_levels)):
            processed_levels[i] = self.upsample_layers[i-1](processed_levels[i])
        
        multi_scale_features = torch.cat(processed_levels, dim=1)
        output = self.output_layer(multi_scale_features)
        return output


class EnhancedFeatureExtractionNetwork(nn.Module):
    """Enhanced feature extraction network with three branches"""
    def __init__(self, input_channels=3, feature_dim=128):
        super(EnhancedFeatureExtractionNetwork, self).__init__()
        
        self.spatial_branch = EnhancedSpatialBranch(input_channels, feature_dim)
        self.fourier_branch = EnhancedFourierBranch(input_channels, feature_dim)
        self.multiscale_branch = EnhancedMultiScaleBranch(input_channels, feature_dim)
        
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * 3, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_dim * 3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, kernel_size=1)
        )
    
    def forward(self, x):
        spatial_features = self.spatial_branch(x)
        fourier_features = self.fourier_branch(x)
        multiscale_features = self.multiscale_branch(x)
        
        target_size = spatial_features.size()[2:]
        if fourier_features.size()[2:] != target_size:
            fourier_features = F.interpolate(fourier_features, 
                                             size=target_size, 
                                             mode='bilinear', 
                                             align_corners=False)
        if multiscale_features.size()[2:] != target_size:
            multiscale_features = F.interpolate(multiscale_features, 
                                                size=target_size, 
                                                mode='bilinear', 
                                                align_corners=False)
        
        combined = torch.cat([spatial_features, fourier_features, multiscale_features], dim=1)
        
        attention_weights = self.attention(combined)  # shape: [B, 3, H, W]
        attention_weights = attention_weights.unsqueeze(2)  # shape: [B, 3, 1, H, W]
        
        features = torch.stack([spatial_features, fourier_features, multiscale_features], dim=1)
        weighted_features = features * attention_weights
        weighted_sum = weighted_features.sum(dim=1)  # shape: [B, feature_dim, H, W]
        
        output = self.output_layer(combined)
        return output, (spatial_features, fourier_features, multiscale_features, attention_weights)


class EnhancedPhaseCorrelationTensorComputation(nn.Module):
    """Reduced module to compute phase correlation tensors"""
    def __init__(self, feature_dim=128, output_dim=256):
        super(EnhancedPhaseCorrelationTensorComputation, self).__init__()
        
        # Reduced channel counts throughout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(feature_dim, 160, kernel_size=3, padding=1),               # Was 256
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 192, kernel_size=3, padding=1, stride=2),             # Was 384
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 224, kernel_size=3, padding=1),                       # Was 512
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, output_dim, kernel_size=1)
        )
        
        # Simplified local coherence module
        self.local_coherence = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, 
                      groups=feature_dim//8),                                    # Reduced groups from //4 to //8
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, 
                      groups=feature_dim//8),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Reduced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim + output_dim, 192, kernel_size=1),             # Was 256
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 48, kernel_size=1),                                    # Was 64
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Reduced output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_dim + output_dim, 224, kernel_size=3, padding=1),  # Was 512
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, output_dim, kernel_size=1)
        )
    
    def forward(self, features):
        processed_features = self.conv_layers(features)
        coherence = self.local_coherence(features)
        
        if coherence.size()[2:] != processed_features.size()[2:]:
            coherence = F.interpolate(coherence, 
                                      size=processed_features.size()[2:], 
                                      mode='bilinear', align_corners=False)
        
        combined = torch.cat([processed_features, coherence], dim=1)
        attention_weights = self.attention(combined)
        attended_combined = combined * attention_weights
        
        tensor = self.output_layer(attended_combined)
        return tensor


###############################################################################
#                           MANIFOLD LEARNING
###############################################################################
# Below is a stripped version of a "ManifoldLearningModule" that can handle
# dynamic input shape, minimal debugging. You can expand as needed.

class ManifoldLearningModule(nn.Module):
    """Manifold learning module (simplified or debug version)."""
    def __init__(
        self, 
        input_dim=128,         # Input tensor channels
        hidden_dim=256,        
        latent_dim=32,         
        gnn_hidden_dim=64      
    ):
        super(ManifoldLearningModule, self).__init__()
        
        # Simple CNN-style encoder
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
        
        # These FC layers will be set up after seeing the input shape
        self.fc_mu = None
        self.fc_logvar = None
        
        # Similarly for decode
        self.fc_decoder = None
        self.decoder = None
        self.decoder_unflatten_dims = None
        
        self.latent_dim = latent_dim
        
        # Simple GNN-like layer: we'll just do a linear for demonstration
        self.gnn = nn.Linear(latent_dim, latent_dim)
        
        # Final projection
        self.projection = nn.Linear(latent_dim, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        x: [B, C=128, H, W]
        Returns: manifold representation + (mu, logvar, z)
        """
        try:
            # Step by step to see shape
            x_enc = self.encoder(x)  # [B, *]
            flat_dim = x_enc.size(1)
            
            # Dynamically create fc layers if needed
            if self.fc_mu is None:
                self.fc_mu = nn.Linear(flat_dim, self.latent_dim).to(x.device)
                self.fc_logvar = nn.Linear(flat_dim, self.latent_dim).to(x.device)
                
                # For decoding
                self.fc_decoder = nn.Linear(self.latent_dim, flat_dim).to(x.device)
                
                # We'll figure out the shape from the last conv
                # The last conv block had shape [B, hidden_dim, H_out, W_out] prior to flatten
                # We can track it here for unflatten
                # But for a short example, let's just store hidden_dim, H_out, W_out
                # We can do that by re-running partial forward:
                # In practice you'd store it carefully.
                # We'll do it once:
                
                # We assume x_enc -> shape [B, hidden_dim*(H_out*W_out)]
                # Let's guess H_out and W_out from x's shape:
                # Down-sample with stride=2 (3 times).
                # If x is large enough, we do:
                # e.g. H//8, W//8
                # We'll do a simpler approach:
                
                # We can un-flatten using the known hidden_dim, but let's just guess
                # a small shape. We'll do it properly if we want to decode.
                pass
            
            mu = self.fc_mu(x_enc)
            logvar = self.fc_logvar(x_enc)
            z = self.reparameterize(mu, logvar)
            
            # Simple "GNN"
            z_gnn = self.gnn(z)
            z_final = self.projection(z_gnn) + z
            z_final = self.layer_norm(z_final)
            
            return z_final, (mu, logvar, z)
        except Exception as e:
            print(f"Error in ManifoldLearningModule forward: {e}")
            raise e
    
    def decode(self, z):
        """Decode latent vector to some reconstruction shape (omitted details)."""
        # For brevity in this example, you can expand as needed
        if self.fc_decoder is None:
            return None
        
        x = self.fc_decoder(z)
        # Then we'd unflatten x and do deconvs
        # ...
        return x


###############################################################################
#                       TINY TOPOLOGICAL FEATURE EXTRACTION
###############################################################################
class TinyPersistentHomologyLayer(nn.Module):
    """Placeholder for an ultra-light persistent homology layer."""
    def __init__(self, max_edge_length=2.0, num_filtrations=10, max_dimension=0):
        super(TinyPersistentHomologyLayer, self).__init__()
        
        self.max_edge_length = max_edge_length
        self.num_filtrations = num_filtrations
        self.max_dimension = max_dimension
        
        # Just store buffer - in real usage, we'd do more
        self.register_buffer(
            'filtration_values',
            torch.linspace(0, max_edge_length, num_filtrations)
        )
    
    def forward(self, x):
        """
        x: [batch_size, num_points, features]
        returns: {'betti_curves': betti_curves_tensor}
        """
        # This is a mock or simplified version
        # In a real scenario, you'd do actual topological computations
        batch_size = x.size(0)
        # Suppose we return a dummy betti curve
        # shape [batch_size, dim+1, num_filtrations], here dim=0 => shape [batch_size, 1, num_filtrations]
        betti_curves_tensor = torch.randn(batch_size, 1, self.num_filtrations, device=x.device)
        
        return {'betti_curves': betti_curves_tensor}


class TinyTopologicalFeatureExtraction(nn.Module):
    """Ultra-light topological feature extraction module"""
    def __init__(self, 
                 input_dim=32, 
                 hidden_dim=32, 
                 output_dim=16,
                 max_edge_length=2.0,
                 num_filtrations=10,
                 max_dimension=0):
        super(TinyTopologicalFeatureExtraction, self).__init__()
        
        self.persistent_homology = TinyPersistentHomologyLayer(
            max_edge_length=max_edge_length,
            num_filtrations=num_filtrations,
            max_dimension=max_dimension
        )
        
        self.betti_processor = nn.Sequential(
            nn.Linear(num_filtrations, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """
        x: [batch_size, num_points, input_dim=32]
        returns: (topo_features, {...})
        """
        homology_data = self.persistent_homology(x)
        betti_curves = homology_data['betti_curves']
        
        # Only dimension 0 for demonstration
        # shape: [B, 1, num_filtrations]
        dim0_betti = betti_curves[:, 0, :]  # -> [B, num_filtrations]
        
        topo_features = self.betti_processor(dim0_betti)
        return topo_features, {'betti_curves': betti_curves}


###############################################################################
#                          CLASSIFICATION COMPONENT
###############################################################################
class SelfAttention(nn.Module):
    """Self-attention mechanism (simplified)."""
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        if feature_dim % num_heads != 0:
            self.adjusted_feature_dim = ((feature_dim // num_heads) + 1) * num_heads
        else:
            self.adjusted_feature_dim = feature_dim
            
        self.head_dim = self.adjusted_feature_dim // num_heads
        
        self.query = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.key = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.value = nn.Linear(feature_dim, self.adjusted_feature_dim)
        
        self.output = nn.Linear(self.adjusted_feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        x: [B, seq_len, feature_dim]
        returns: same shape
        """
        residual = x
        x = self.layer_norm(x)
        
        B, seq_len, _ = x.shape
        query = self.query(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key   = self.key(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value = self.value(x).view(B, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1,2).contiguous().view(B, seq_len, self.adjusted_feature_dim)
        out = self.output(context)
        
        return out + residual


class FeedForward(nn.Module):
    """Feed-forward network for a transformer layer."""
    def __init__(self, feature_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class TransformerLayer(nn.Module):
    """Single transformer layer."""
    def __init__(self, feature_dim, hidden_dim, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttention(feature_dim, num_heads, dropout)
        self.feed_forward = FeedForward(feature_dim, hidden_dim, dropout)
    
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class UncertaintyEstimator(nn.Module):
    """Estimates classification uncertainty with a simple evidential approach."""
    def __init__(self, feature_dim, hidden_dim=64):
        super(UncertaintyEstimator, self).__init__()
        self.evidence_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # alpha, beta, etc.
        )
    
    def forward(self, x):
        evidence = torch.exp(self.evidence_network(x))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        # For 2 classes, K=2
        K = 2
        uncertainty = K / torch.sum(alpha, dim=1)
        return probs, uncertainty


class FeatureProjector(nn.Module):
    """Projects features to a common dimension."""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(FeatureProjector, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.projector(x)


class ClassificationNetwork(nn.Module):
    """Classification network for AI-generated vs natural image detection."""
    def __init__(self, 
                 manifold_dim=32,
                 topo_dim=32,
                 feature_dim=64,
                 hidden_dim=128,
                 num_layers=3,
                 num_heads=4,
                 dropout=0.1):
        super(ClassificationNetwork, self).__init__()
        
        self.manifold_dim = manifold_dim
        self.topo_dim = topo_dim
        self.feature_dim = feature_dim
        
        self.manifold_projector = FeatureProjector(manifold_dim, feature_dim // 2, dropout)
        self.topo_projector = FeatureProjector(topo_dim, feature_dim // 2, dropout)
        
        self.manifold_type_embedding = nn.Parameter(torch.randn(1, 1, feature_dim // 2))
        self.topo_type_embedding = nn.Parameter(torch.randn(1, 1, feature_dim // 2))
        
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(feature_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.global_attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        self.uncertainty = UncertaintyEstimator(feature_dim)
    
    def forward(self, manifold_features, topo_features):
        """
        manifold_features: [B, manifold_dim]
        topo_features: [B, topo_dim]
        """
        B = manifold_features.size(0)
        
        manifold_proj = self.manifold_projector(manifold_features)  # [B, feature_dim//2]
        topo_proj = self.topo_projector(topo_features)              # [B, feature_dim//2]
        
        manifold_seq = manifold_proj.unsqueeze(1)  # [B,1,feature_dim//2]
        topo_seq = topo_proj.unsqueeze(1)          # [B,1,feature_dim//2]
        
        manifold_embedding = self.manifold_type_embedding.expand(B, -1, -1)
        topo_embedding = self.topo_type_embedding.expand(B, -1, -1)
        
        manifold_seq = manifold_seq + manifold_embedding
        topo_seq = topo_seq + topo_embedding
        
        # Combine them into a single sequence
        # For demonstration: we just combine them along the feature dimension
        # Then replicate along seq length
        # Alternatively, you can do manifold as "token 0" and topo as "token 1" ...
        
        combined = torch.cat([manifold_seq, topo_seq], dim=2)  # shape [B,1, feature_dim]
        # We'll replicate to get length 2
        feature_sequence = combined.expand(-1, 2, -1)          # [B,2,feature_dim]
        
        for layer in self.transformer_layers:
            feature_sequence = layer(feature_sequence)
        
        attention_weights = self.global_attention_pool(feature_sequence)  # [B,2,1]
        pooled_features = torch.sum(feature_sequence * attention_weights, dim=1)  # [B, feature_dim]
        
        logits = self.classifier(pooled_features)
        probs, uncertainty = self.uncertainty(pooled_features)
        
        return logits, probs, uncertainty
    
    def mse_loss(self, x, x_target):
        return F.mse_loss(x, x_target)


###############################################################################
#                   DIMENSION ADAPTER & POINT CLOUD GENERATOR
###############################################################################
class DimensionAdapter(nn.Module):
    """
    Adapter to match dimensions between the Enhanced Tensor (256 channels)
    and the Manifold Module input (128 channels).
    """
    def __init__(self, input_dim=256, output_dim=128):
        super(DimensionAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(input_dim, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, output_dim, kernel_size=1)
        )
    
    def forward(self, x):
        return self.adapter(x)


class PointCloudGenerator(nn.Module):
    """
    Creates a point cloud from manifold feature vectors.
    This helps feed data into topological analysis that expects [B, N, D].
    """
    def __init__(self, num_points=30, noise_scale=0.1):
        super(PointCloudGenerator, self).__init__()
        self.num_points = num_points
        self.noise_scale = noise_scale
    
    def forward(self, features):
        # features: [B, feature_dim]
        B, D = features.size()
        point_cloud = features.unsqueeze(1).expand(-1, self.num_points, -1)
        noise = torch.randn(B, self.num_points, D, device=features.device) * self.noise_scale
        point_cloud = point_cloud + noise
        return point_cloud


###############################################################################
#                          PUTTING IT ALL TOGETHER
###############################################################################
if __name__ == "__main__":
    # Example usage of the entire pipeline in a single file.
    import sys
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Create model components
    feature_network = EnhancedFeatureExtractionNetwork().to(device)
    tensor_computer = EnhancedPhaseCorrelationTensorComputation().to(device)
    dim_adapter = DimensionAdapter().to(device)
    manifold_module = ManifoldLearningModule().to(device)
    point_cloud_generator = PointCloudGenerator().to(device)
    topo_module = TinyTopologicalFeatureExtraction(
        input_dim=32,  # match manifold latent dim
        hidden_dim=32,
        output_dim=16,
        max_edge_length=2.0,
        num_filtrations=10,
        max_dimension=0
    ).to(device)
    classifier = ClassificationNetwork(
        manifold_dim=32,  # from manifold module output
        topo_dim=16,      # from topological extraction output
        feature_dim=32,
        hidden_dim=64,
        dropout=0.1
    ).to(device)
    
    # Print parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = (
        count_parameters(feature_network)
        + count_parameters(tensor_computer)
        + count_parameters(dim_adapter)
        + count_parameters(manifold_module)
        + count_parameters(point_cloud_generator)
        + count_parameters(topo_module)
        + count_parameters(classifier)
    )
    print(f"Total combined parameters: {total_params:,}", file=sys.stderr)
    
    # Quick demonstration with random input
    # Suppose we have an image of shape [3, H, W]
    # For a test, create random image or load a real one:
    test_img = torch.rand(3, 384, 384)  # Example shape
    test_img = test_img.unsqueeze(0).to(device)  # batch size 1
    
    # Run through pipeline:
    with torch.no_grad():
        # 1) Feature extraction
        feature_maps, _ = feature_network(test_img)  # shape [B, 128, H', W']
        
        # 2) Compute phase correlation tensor
        correlation_tensor = tensor_computer(feature_maps)  # shape [B, 256, H'', W'']
        
        # 3) Adapt dimension to 128
        adapted_tensor = dim_adapter(correlation_tensor)  # shape [B, 128, H'', W'']
        
        # 4) Manifold learning
        manifold_repr, (mu, logvar, z) = manifold_module(adapted_tensor)  # shape [B, 32]
        
        # 5) Turn manifold features into a point cloud for topological analysis
        point_cloud = point_cloud_generator(manifold_repr)  # shape [B, num_points, 32]
        
        # 6) Tiny topological extraction
        topo_features, topo_data = topo_module(point_cloud)  # shape [B, 16]
        
        # 7) Classification
        logits, probs, uncertainty = classifier(manifold_repr, topo_features)
        predictions = torch.argmax(logits, dim=1)
        
    print(f"Predicted class: {predictions.item()}", file=sys.stderr)
    print(f"Class probabilities: {probs}", file=sys.stderr)
    print(f"Uncertainty: {uncertainty}", file=sys.stderr)
