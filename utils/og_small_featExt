#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Correlation Manifold (PCM) - Tiny Feature Extraction Component

This file implements a lightweight version of the feature extraction component 
for AI-generated image detection. It preserves the key functionality while
drastically reducing the parameter count.

Key components:
1. Simplified Spatial Branch: Uses a lightweight CNN instead of ResNet
2. Minimal Fourier Branch: Processes frequency domain information with fewer parameters
3. Reduced Multi-Scale Branch: Analyzes fewer scales with shared weights
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import cv2


class ImagePreprocessor:
    """Class for image preprocessing operations"""
    
    def __init__(self, target_size=(256, 256)):  # Reduced from 512x512
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
            return F.interpolate(image.unsqueeze(0), size=self.target_size).squeeze(0)
        elif isinstance(image, Image.Image):
            return image.resize(self.target_size)
        else:
            raise TypeError("Image must be numpy array, torch tensor, or PIL Image")
    
    def to_tensor(self, image):
        """Convert to PyTorch tensor"""
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
        """Apply all preprocessing steps"""
        image = self.resize(image)
        image = self.normalize(image)
        image = self.to_tensor(image)
        return image


class TinySpatialBranch(nn.Module):
    """Lightweight CNN-based branch for spatial domain feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=32):  # Reduced from 64 to 32
        super(TinySpatialBranch, self).__init__()
        
        # Lightweight CNN instead of ResNet
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Simplified attention module
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.projection = nn.Conv2d(64, output_channels, kernel_size=1)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Project to output dimension
        output = self.projection(attended_features)
        
        return output


class TinyFFTLayer(nn.Module):
    """Simplified layer that computes FFT and extracts phase"""
    
    def __init__(self):
        super(TinyFFTLayer, self).__init__()
    
    def forward(self, x):
        # Apply FFT2 to the last 2 dimensions
        x_fft = torch.fft.fft2(x)
        
        # Extract phase only (ignore amplitude for efficiency)
        phase = torch.angle(x_fft)
        
        return phase


class TinyFourierBranch(nn.Module):
    """Lightweight branch for Fourier domain feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=32):  # Reduced from 64 to 32
        super(TinyFourierBranch, self).__init__()
        
        self.fft_layer = TinyFFTLayer()
        
        # Simplified convolutional layers
        self.phase_processor = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1, stride=2)
        )
    
    def forward(self, x):
        # Compute FFT and extract phase
        phase = self.fft_layer(x)
        
        # Process phase directly
        output = self.phase_processor(phase)
        
        return output


class TinyMultiScaleBranch(nn.Module):
    """Lightweight branch for multi-scale feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=32, pyramid_levels=2):  # Reduced from 3 to 2
        super(TinyMultiScaleBranch, self).__init__()
        
        # Simplified downsampling with adaptive pooling
        self.downsample = nn.AdaptiveAvgPool2d(output_size=(32, 32))
        
        # Shared convolutional layers for all levels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1, stride=2)
        )
    
    def forward(self, x):
        # Downsample the input
        downsampled = self.downsample(x)
        
        # Process with shared layers
        output = self.conv_layers(downsampled)
        
        return output


class TinyFeatureExtractionNetwork(nn.Module):
    """Lightweight feature extraction network with three branches"""
    
    def __init__(self, input_channels=3, feature_dim=32):  # Reduced from 64 to 32
        super(TinyFeatureExtractionNetwork, self).__init__()
        
        self.spatial_branch = TinySpatialBranch(input_channels, feature_dim)
        self.fourier_branch = TinyFourierBranch(input_channels, feature_dim)
        self.multiscale_branch = TinyMultiScaleBranch(input_channels, feature_dim)
        
        # Simple averaging instead of learned attention
        # Final output layer with reduced dimensions
        self.output_layer = nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1)
    
    def forward(self, x):
        # Process input through each branch
        spatial_features = self.spatial_branch(x)
        fourier_features = self.fourier_branch(x)
        multiscale_features = self.multiscale_branch(x)
        
        # Resize all features to match spatial features size
        target_size = spatial_features.size()[2:]
        fourier_features = F.interpolate(fourier_features, size=target_size, mode='bilinear', align_corners=False)
        multiscale_features = F.interpolate(multiscale_features, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate features
        combined = torch.cat([spatial_features, fourier_features, multiscale_features], dim=1)
        
        # Process through final layer
        output = self.output_layer(combined)
        
        return output, (spatial_features, fourier_features, multiscale_features)


class TinyPhaseCorrelationTensorComputation(nn.Module):
    """Lightweight module to compute phase correlation tensors"""
    
    def __init__(self, feature_dim=32, output_dim=64):  # Reduced from 64/128 to 32/64
        super(TinyPhaseCorrelationTensorComputation, self).__init__()
        
        # Simplified processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, output_dim, kernel_size=3, padding=1)
        )
        
        # Local phase coherence with fewer parameters
        self.local_coherence = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
        # Skip the attention module to save parameters
        
        # Final layer with fewer parameters
        self.output_layer = nn.Conv2d(feature_dim + output_dim, output_dim, kernel_size=1)
    
    def forward(self, features):
        # Initial processing of features
        processed_features = self.conv_layers(features)
        
        # Compute local phase coherence
        coherence = self.local_coherence(features)
        
        # Combine processed features with coherence
        combined = torch.cat([processed_features, coherence], dim=1)
        
        # Final output without attention
        tensor = self.output_layer(combined)
        
        return tensor


def test_tiny_feature_extraction(image_path, device='cpu'):
    """Test the tiny feature extraction components"""
    # Create preprocessing module
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    # Load and process image
    img = Image.open(image_path)
    processed_img = preprocessor.process(img)
    processed_img = processed_img.unsqueeze(0).to(device)  # Add batch dimension
    
    # Create feature extraction network
    feature_network = TinyFeatureExtractionNetwork().to(device)
    
    # Extract features
    with torch.no_grad():
        features, branch_outputs = feature_network(processed_img)
    
    # Create phase correlation tensor computation module
    tensor_computer = TinyPhaseCorrelationTensorComputation().to(device)
    
    # Compute phase correlation tensor
    with torch.no_grad():
        tensor = tensor_computer(features)
    
    print(f"Input image shape: {processed_img.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Phase correlation tensor shape: {tensor.shape}")
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"TinyFeatureExtractionNetwork parameters: {count_parameters(feature_network):,}")
    print(f"TinyPhaseCorrelationTensorComputation parameters: {count_parameters(tensor_computer):,}")
    print(f"Total parameters: {count_parameters(feature_network) + count_parameters(tensor_computer):,}")
    
    # Return the tensor for further processing
    return tensor


if __name__ == "__main__":
    # If this file is run directly, perform a quick test
    import os
    
    # Create directories if they don't exist
    os.makedirs('sample_images', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # If no test image exists, create a dummy one
    sample_img_path = 'sample_images/sample_natural.jpg'
    if not os.path.exists(sample_img_path):
        # Generate a random image
        random_img = np.random.rand(256, 256, 3)
        from PIL import Image
        Image.fromarray((random_img * 255).astype(np.uint8)).save(sample_img_path)
    
    # Test feature extraction
    tensor = test_tiny_feature_extraction(sample_img_path)
    print("Tiny feature extraction test completed successfully!")