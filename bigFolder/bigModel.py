#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_large.py

Consolidated PCM (Phase Correlation Manifold) code for AI-generated image detection.
SCALED UP VERSION with ~10x parameters compared to original.
This file combines:
1) Enhanced Feature Extraction (Using ResNet19 for spatial branch)
2) Manifold Learning Module (Variational-style + GNN, or simpler version)
3) Tiny Topological Feature Extraction
4) Classification Network (with optional attention, etc.)
5) Utility classes and final pipeline instantiation

Author: Your Name
Date: 2025-04-09
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import random
import torchvision.models as models

# If you use GNN or Gudhi for topological computations, uncomment as needed:
# import torch_geometric.nn as gnn
# import gudhi as gd
# from scipy.spatial.distance import pdist, squareform


###############################################################################
#                               IMAGE PREPROCESSOR
###############################################################################
class ImagePreprocessor:
    """Class for image preprocessing operations"""
    def __init__(self, target_size=(512, 512)):  # Increased from 384x384
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
    """ResNet19-based architecture for spatial features"""
    def __init__(self, input_channels=3, output_channels=512):  # Increased from 128
        super(EnhancedSpatialBranch, self).__init__()
        
        # Use pre-trained ResNet18 and add one more layer to make it ResNet19
        resnet18 = models.resnet18(pretrained=True)
        
        # Extract layers from ResNet18
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        
        # Add an extra layer to make it "ResNet19"
        self.layer5 = self._make_layer(BasicBlock, 512, 512, blocks=2, stride=1)
        
        # Enhanced attention mechanism (scaled up from original)
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.projection = nn.Conv2d(512, output_channels, kernel_size=1)
    
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
        x = self.layer5(x)  # Extra layer
        
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
    """Scaled-up branch for Fourier domain feature extraction"""
    def __init__(self, input_channels=3, output_channels=512):  # Increased from 128
        super(EnhancedFourierBranch, self).__init__()
        
        self.fft_layer = EnhancedFFTLayer()
        
        # Increased channel counts throughout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels * 2, 128, kernel_size=3, padding=1),       # Increased from 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),            # Increased from 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),            # Increased from 128
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 768, kernel_size=3, padding=1, stride=2),            # Increased from 192
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, output_channels, kernel_size=1)
        )
        
        # Increased channel counts in phase processor
        self.phase_processor = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),           # Increased from 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),            # Increased from 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, output_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # Increased intermediate channels
        self.output_layer = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1),
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
    def __init__(self, levels=5):  # Increased from 3
        super(EnhancedPyramidLayer, self).__init__()
        self.levels = levels
        self.gaussian_kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self, kernel_size=7, sigma=1.5):  # Increased from 5, 1.0
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
    """Scaled-up branch for multi-scale feature extraction"""
    def __init__(self, input_channels=3, output_channels=512, pyramid_levels=5):  # Increased from 128, 3
        super(EnhancedMultiScaleBranch, self).__init__()
        
        self.pyramid_layer = EnhancedPyramidLayer(levels=pyramid_levels)
        
        # Increased channel counts and enhanced convolution layers
        self.level_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),            # Increased from 32
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),                       # Increased from 64
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=2),             # Increased from 96
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
            ) for _ in range(pyramid_levels)
        ])
        
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=False)
            for i in range(1, pyramid_levels)
        ])
        
        # Increased channel count in output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(384 * pyramid_levels, 768, kernel_size=3, padding=1),          # Increased from 96*pyramid_levels, 192
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, output_channels, kernel_size=1)
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
    def __init__(self, input_channels=3, feature_dim=512):  # Increased from 128
        super(EnhancedFeatureExtractionNetwork, self).__init__()
        
        self.spatial_branch = EnhancedSpatialBranch(input_channels, feature_dim)
        self.fourier_branch = EnhancedFourierBranch(input_channels, feature_dim)
        self.multiscale_branch = EnhancedMultiScaleBranch(input_channels, feature_dim)
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * 3, 512, kernel_size=1),                          # Increased from 128
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),                                      # Increased from 64
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_dim * 3, 1024, kernel_size=3, padding=1),              # Increased from 256
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, feature_dim, kernel_size=1)
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
    """Scaled-up module to compute phase correlation tensors"""
    def __init__(self, feature_dim=512, output_dim=1024):  # Increased from 128, 256
        super(EnhancedPhaseCorrelationTensorComputation, self).__init__()
        
        # Increased channel counts throughout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(feature_dim, 640, kernel_size=3, padding=1),                   # Increased from 160
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640, 768, kernel_size=3, padding=1, stride=2),                 # Increased from 192
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 896, kernel_size=3, padding=1),                           # Increased from 224
            nn.BatchNorm2d(896),
            nn.ReLU(inplace=True),
            nn.Conv2d(896, output_dim, kernel_size=1)
        )
        
        # Enhanced local coherence module
        self.local_coherence = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, 
                      groups=feature_dim//16),                                       # Reduced groups to increase parameters
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, 
                      groups=feature_dim//16),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim + output_dim, 768, kernel_size=1),                 # Increased from 192
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 192, kernel_size=1),                                      # Increased from 48
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_dim + output_dim, 896, kernel_size=3, padding=1),      # Increased from 224
            nn.BatchNorm2d(896),
            nn.ReLU(inplace=True),
            nn.Conv2d(896, output_dim, kernel_size=1)
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
# Enhanced manifold learning module with more parameters

class ManifoldLearningModule(nn.Module):
    """Enhanced manifold learning module with more parameters."""
    def __init__(
        self, 
        input_dim=512,         # Increased from 128
        hidden_dim=1024,       # Increased from 256
        latent_dim=128,        # Increased from 32
        gnn_hidden_dim=256     # Increased from 64
    ):
        super(ManifoldLearningModule, self).__init__()
        
        # Enhanced CNN-style encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # Added extra layer
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
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
        
        # Enhanced GNN-like layers with multiple fully connected layers
        self.gnn = nn.Sequential(
            nn.Linear(latent_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, latent_dim)
        )
        
        # Enhanced final projection
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, gnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(gnn_hidden_dim, latent_dim)
        )
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        x: [B, C=512, H, W]  # Increased from 128
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
                # Down-sample with stride=2 (4 times in enhanced model).
                # If x is large enough, we do:
                # e.g. H//16, W//16
                # We'll do a simpler approach:
                
                # We can un-flatten using the known hidden_dim, but let's just guess
                # a small shape. We'll do it properly if we want to decode.
                pass
            
            mu = self.fc_mu(x_enc)
            logvar = self.fc_logvar(x_enc)
            z = self.reparameterize(mu, logvar)
            
            # Enhanced "GNN" with multiple layers
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
    """Enhanced persistent homology layer with more parameters."""
    def __init__(self, max_edge_length=2.0, num_filtrations=20, max_dimension=1):  # Increased from 10, 0
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
        # Enhanced to return both dimension 0 and 1 (if specified) betti curves
        # shape [batch_size, dim+1, num_filtrations], 
        # For max_dim=1: shape [batch_size, 2, num_filtrations]
        betti_curves_tensor = torch.randn(batch_size, self.max_dimension + 1, 
                                         self.num_filtrations, device=x.device)
        
        return {'betti_curves': betti_curves_tensor}


class TinyTopologicalFeatureExtraction(nn.Module):
    """Enhanced topological feature extraction module with more parameters"""
    def __init__(self, 
                 input_dim=128,       # Increased from 32
                 hidden_dim=128,      # Increased from 32
                 output_dim=64,       # Increased from 16
                 max_edge_length=2.0,
                 num_filtrations=20,  # Increased from 10
                 max_dimension=1):    # Increased from 0
        super(TinyTopologicalFeatureExtraction, self).__init__()
        
        self.persistent_homology = TinyPersistentHomologyLayer(
            max_edge_length=max_edge_length,
            num_filtrations=num_filtrations,
            max_dimension=max_dimension
        )
        
        # Enhanced processor with more layers for dimension 0 Betti curves
        self.betti0_processor = nn.Sequential(
            nn.Linear(num_filtrations, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim // 2)
        )
        
        # Added processor for dimension 1 Betti curves
        self.betti1_processor = nn.Sequential(
            nn.Linear(num_filtrations, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim // 2)
        )
    
    def forward(self, x):
        """
        x: [batch_size, num_points, input_dim=128]
        returns: (topo_features, {...})
        """
        homology_data = self.persistent_homology(x)
        betti_curves = homology_data['betti_curves']
        
        # Process both dimension 0 and 1
        # shape: [B, 2, num_filtrations]
        dim0_betti = betti_curves[:, 0, :]  # -> [B, num_filtrations]
        dim0_features = self.betti0_processor(dim0_betti)
        
        dim1_betti = betti_curves[:, 1, :]  # -> [B, num_filtrations]
        dim1_features = self.betti1_processor(dim1_betti)
        
        # Combine features from both dimensions
        topo_features = torch.cat([dim0_features, dim1_features], dim=1)
        return topo_features, {'betti_curves': betti_curves}
    
###############################################################################
#                          CLASSIFICATION COMPONENT
###############################################################################
class SelfAttention(nn.Module):
    """Enhanced self-attention mechanism with more parameters."""
    def __init__(self, feature_dim, num_heads=8, dropout=0.2):  # Increased from 4, 0.1
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        if feature_dim % num_heads != 0:
            self.adjusted_feature_dim = ((feature_dim // num_heads) + 1) * num_heads
        else:
            self.adjusted_feature_dim = feature_dim
            
        self.head_dim = self.adjusted_feature_dim // num_heads
        
        # Enhanced with additional projections
        self.query_pre = nn.Linear(feature_dim, feature_dim)
        self.key_pre = nn.Linear(feature_dim, feature_dim)
        self.value_pre = nn.Linear(feature_dim, feature_dim)
        
        self.query = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.key = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.value = nn.Linear(feature_dim, self.adjusted_feature_dim)
        
        # Enhanced output processing
        self.output_proj = nn.Linear(self.adjusted_feature_dim, self.adjusted_feature_dim)
        self.output = nn.Linear(self.adjusted_feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # Added gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: [B, seq_len, feature_dim]
        returns: same shape
        """
        residual = x
        x = self.layer_norm1(x)
        
        # Enhanced pre-processing
        q_input = self.query_pre(x)
        k_input = self.key_pre(x)
        v_input = self.value_pre(x)
        
        B, seq_len, _ = x.shape
        query = self.query(q_input).view(B, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        key = self.key(k_input).view(B, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        value = self.value(v_input).view(B, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1,2).contiguous().view(B, seq_len, self.adjusted_feature_dim)
        
        # Enhanced output processing
        context = self.output_proj(context)
        context = F.gelu(context)  # Using GELU instead of ReLU
        out = self.output(context)
        out = self.layer_norm2(out)
        
        # Gating mechanism
        gate_input = torch.cat([out, residual], dim=-1)
        gate_value = self.gate(gate_input)
        
        # Apply gating
        out = gate_value * out + (1 - gate_value) * residual
        
        return out


class FeedForward(nn.Module):
    """Enhanced feed-forward network for a transformer layer."""
    def __init__(self, feature_dim, hidden_dim, dropout=0.2):  # Increased from 0.1
        super(FeedForward, self).__init__()
        # Enhanced with more layers and wider hidden dimensions
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim * 2)  # Wider intermediate layer
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(hidden_dim * 2, hidden_dim)  # Additional layer
        self.dropout3 = nn.Dropout(dropout)
        self.linear4 = nn.Linear(hidden_dim, feature_dim)
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        
        # Added residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        
        # Enhanced forward pass
        x = self.linear1(x)
        x = F.gelu(x)  # Using GELU instead of ReLU
        x = self.dropout1(x)
        
        x = self.linear2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        
        x = self.linear3(x)
        x = F.gelu(x)
        x = self.dropout3(x)
        
        x = self.linear4(x)
        x = self.layer_norm2(x)
        
        # Scaled residual connection
        return x + self.residual_scale * residual


class TransformerLayer(nn.Module):
    """Enhanced transformer layer with more parameters."""
    def __init__(self, feature_dim, hidden_dim, num_heads=8, dropout=0.2):  # Increased from 4, 0.1
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttention(feature_dim, num_heads, dropout)
        self.feed_forward = FeedForward(feature_dim, hidden_dim, dropout)
        
        # Added learnable scaling factors
        self.attention_scale = nn.Parameter(torch.ones(1))
        self.ff_scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Apply scaled connections
        x = x + self.attention_scale * self.attention(x)
        x = x + self.ff_scale * self.feed_forward(x)
        return x


class UncertaintyEstimator(nn.Module):
    """Enhanced uncertainty estimator with more parameters."""
    def __init__(self, feature_dim, hidden_dim=256):  # Increased from 64
        super(UncertaintyEstimator, self).__init__()
        # Enhanced with deeper network
        self.evidence_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Added batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),  # Added layer
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 4)  # alpha, beta, etc.
        )
        
        # Added auxiliary confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        evidence = torch.exp(self.evidence_network(x))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        # For 2 classes, K=2
        K = 2
        uncertainty = K / torch.sum(alpha, dim=1)
        
        # Additional confidence estimate
        confidence = self.confidence_estimator(x).squeeze(-1)
        
        # Combine the estimates
        adjusted_uncertainty = uncertainty * (1 - confidence)
        
        return probs, adjusted_uncertainty


class FeatureProjector(nn.Module):
    """Enhanced feature projector with more parameters."""
    def __init__(self, input_dim, output_dim, dropout=0.2):  # Increased from 0.1
        super(FeatureProjector, self).__init__()
        # Enhanced with deeper network and residual connections
        mid_dim = max(input_dim, output_dim) * 2  # Wider intermediate representation
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, mid_dim),  # Added layer
            nn.LayerNorm(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Added residual connection if dimensions match
        self.use_residual = (input_dim == output_dim)
        if not self.use_residual:
            self.residual_adapter = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        projected = self.projector(x)
        
        if self.use_residual:
            return projected + x
        else:
            return projected + self.residual_adapter(x)


class ClassificationNetwork(nn.Module):
    """Enhanced classification network with more parameters."""
    def __init__(self, 
                 manifold_dim=128,    # Increased from 32
                 topo_dim=64,         # Increased from 32
                 feature_dim=256,     # Increased from 64
                 hidden_dim=512,      # Increased from 128
                 num_layers=5,        # Increased from 3
                 num_heads=8,         # Increased from 4
                 dropout=0.2):        # Increased from 0.1
        super(ClassificationNetwork, self).__init__()
        
        self.manifold_dim = manifold_dim
        self.topo_dim = topo_dim
        self.feature_dim = feature_dim
        
        # Enhanced projection components
        self.manifold_projector = FeatureProjector(manifold_dim, feature_dim // 2, dropout)
        self.topo_projector = FeatureProjector(topo_dim, feature_dim // 2, dropout)
        
        # Enhanced with learnable type embeddings
        self.manifold_type_embedding = nn.Parameter(torch.randn(1, 1, feature_dim // 2))
        self.topo_type_embedding = nn.Parameter(torch.randn(1, 1, feature_dim // 2))
        
        # Enhanced with more transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(feature_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Enhanced attention pooling
        self.global_attention_pool = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # Added layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Enhanced uncertainty estimator
        self.uncertainty = UncertaintyEstimator(feature_dim, hidden_dim // 2)
        
        # Added token mixing layers
        self.token_mixer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, manifold_features, topo_features):
        """
        manifold_features: [B, manifold_dim]
        topo_features: [B, topo_dim]
        """
        B = manifold_features.size(0)
        
        # Project features to common dimension
        manifold_proj = self.manifold_projector(manifold_features)  # [B, feature_dim//2]
        topo_proj = self.topo_projector(topo_features)              # [B, feature_dim//2]
        
        # Add sequence dimension
        manifold_seq = manifold_proj.unsqueeze(1)  # [B,1,feature_dim//2]
        topo_seq = topo_proj.unsqueeze(1)          # [B,1,feature_dim//2]
        
        # Add type embeddings
        manifold_embedding = self.manifold_type_embedding.expand(B, -1, -1)
        topo_embedding = self.topo_type_embedding.expand(B, -1, -1)
        
        manifold_seq = manifold_seq + manifold_embedding
        topo_seq = topo_seq + topo_embedding
        
        # Combine them into a single sequence along feature dimension
        combined = torch.cat([manifold_seq, topo_seq], dim=2)  # shape [B,1,feature_dim]
        
        # Expand sequence length
        feature_sequence = torch.cat([combined, combined], dim=1)  # [B,2,feature_dim]
        
        # Apply token mixing
        feature_sequence = self.token_mixer(feature_sequence) + feature_sequence
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            feature_sequence = layer(feature_sequence)
        
        # Attention pooling
        attention_weights = self.global_attention_pool(feature_sequence)  # [B,2,1]
        pooled_features = torch.sum(feature_sequence * attention_weights, dim=1)  # [B, feature_dim]
        
        # Classification
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
    Enhanced adapter to match dimensions between the Enhanced Tensor (1024 channels)
    and the Manifold Module input (512 channels).
    """
    def __init__(self, input_dim=1024, output_dim=512):  # Increased from 256, 128
        super(DimensionAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(input_dim, 768, kernel_size=3, padding=1),  # Increased from 192
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 640, kernel_size=3, padding=1),  # Added extra layer
            nn.BatchNorm2d(640),
            nn.ReLU(inplace=True),
            nn.Conv2d(640, output_dim, kernel_size=1)
        )
    
    def forward(self, x):
        return self.adapter(x)


class PointCloudGenerator(nn.Module):
    """
    Enhanced point cloud generator with more parameters.
    """
    def __init__(self, num_points=120, noise_scale=0.1):  # Increased from 30
        super(PointCloudGenerator, self).__init__()
        self.num_points = num_points
        self.noise_scale = noise_scale
        
        # Added learnable feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, features):
        # features: [B, feature_dim]
        B, D = features.size()
        
        # Generate base point cloud
        point_cloud = features.unsqueeze(1).expand(-1, self.num_points, -1)  # [B, num_points, D]
        
        # Generate structured noise
        noise_base = torch.randn(B, self.num_points, 1, device=features.device)
        transformed_noise = self.feature_transform(noise_base)  # More structured noise
        
        # Expand transformed noise to feature dimension
        noise = transformed_noise.expand(-1, -1, D) * self.noise_scale
        
        # Add noise to point cloud
        point_cloud = point_cloud + noise
        
        return point_cloud