#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Correlation Manifold (PCM) - Feature Extraction Component

This file implements the feature extraction component of the PCM architecture
for AI-generated image detection. It consists of three specialized branches:
1. Spatial Domain Branch: CNN-based processing of direct image features
2. Fourier Domain Branch: Processing in frequency domain to extract phase information
3. Multi-Scale Branch: Analysis across different scales using Gaussian pyramid

These branches work together to extract complementary features for phase correlation tensor computation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2


class ImagePreprocessor:
    """Class for image preprocessing operations"""
    
    def __init__(self, target_size=(512, 512)):
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


class SpatialDomainBranch(nn.Module):
    """CNN-based branch for spatial domain feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=64):
        super(SpatialDomainBranch, self).__init__()
        
        # Use a pretrained ResNet model for feature extraction
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Modify the first layer to accept grayscale images if needed
        if input_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Add attention module
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.projection = nn.Conv2d(512, output_channels, kernel_size=1)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Project to output dimension
        output = self.projection(attended_features)
        
        return output


class FFTLayer(nn.Module):
    """Layer that computes FFT and separates amplitude and phase"""
    
    def __init__(self):
        super(FFTLayer, self).__init__()
    
    def forward(self, x):
        # Apply FFT2 to the last 2 dimensions
        # Input shape: (batch_size, channels, height, width)
        x_fft = torch.fft.fft2(x)
        
        # Shift to center the DC component
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # Extract amplitude (magnitude) and phase
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Log amplitude for better numerical stability
        log_amplitude = torch.log(amplitude + 1e-10)
        
        # Concatenate along the channel dimension
        # Output shape: (batch_size, 2*channels, height, width)
        output = torch.cat([log_amplitude, phase], dim=1)
        
        return output


class FourierDomainBranch(nn.Module):
    """Branch for Fourier domain feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=64):
        super(FourierDomainBranch, self).__init__()
        
        self.fft_layer = FFTLayer()
        
        # Convolutional layers to process frequency domain features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=3, padding=1),  # *2 because we have amplitude and phase
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_channels, kernel_size=1)
        )
        
        # Phase-specific processing
        self.phase_processor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
        )
        
        # Final output layer
        self.output_layer = nn.Conv2d(output_channels * 2, output_channels, kernel_size=1)
    
    def forward(self, x):
        # Compute FFT
        fft_output = self.fft_layer(x)
        
        # Process combined amplitude and phase
        conv_features = self.conv_layers(fft_output)
        
        # Extract and process phase component separately
        phase = fft_output[:, x.size(1):]
        phase_features = self.phase_processor(phase)
        
        # Combine features
        combined = torch.cat([conv_features, phase_features], dim=1)
        output = self.output_layer(combined)
        
        return output


class PyramidLayer(nn.Module):
    """Layer that creates a Gaussian pyramid"""
    
    def __init__(self, levels=3):
        super(PyramidLayer, self).__init__()
        self.levels = levels
        self.gaussian_kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        """Create a Gaussian kernel for blurring"""
        # Create a 1D Gaussian kernel
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create a 2D Gaussian kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        # Repeat for each input channel (for depth-wise convolution)
        return kernel_2d
    
    def _gaussian_blur(self, x):
        """Apply Gaussian blur to input tensor"""
        # Create a kernel for each input channel
        kernel = self.gaussian_kernel.repeat(x.size(1), 1, 1, 1).to(x.device)
        
        # Apply depth-wise convolution
        padding = (self.gaussian_kernel.size(2) - 1) // 2
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))
    
    def forward(self, x):
        """Create a Gaussian pyramid"""
        pyramid = [x]
        current = x
        
        for _ in range(self.levels - 1):
            # Blur and downsample
            blurred = self._gaussian_blur(current)
            downsampled = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            pyramid.append(downsampled)
            current = downsampled
        
        return pyramid


class MultiScaleBranch(nn.Module):
    """Branch for multi-scale feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=64, pyramid_levels=3):
        super(MultiScaleBranch, self).__init__()
        
        self.pyramid_layer = PyramidLayer(levels=pyramid_levels)
        
        # Convolutional layers for each pyramid level
        self.level_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            ) for _ in range(pyramid_levels)
        ])
        
        # Upsampling layers to bring all levels to the same resolution
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=False)
            for i in range(1, pyramid_levels)
        ])
        
        # Final convolutional layer
        self.output_layer = nn.Conv2d(64 * pyramid_levels, output_channels, kernel_size=1)
    
    def forward(self, x):
        # Generate pyramid
        pyramid = self.pyramid_layer(x)
        
        # Process each level
        processed_levels = []
        for i, level in enumerate(pyramid):
            processed = self.level_convs[i](level)
            processed_levels.append(processed)
        
        # Upsample lower resolution levels
        for i in range(1, len(processed_levels)):
            processed_levels[i] = self.upsample_layers[i-1](processed_levels[i])
        
        # Concatenate all levels
        multi_scale_features = torch.cat(processed_levels, dim=1)
        
        # Final processing
        output = self.output_layer(multi_scale_features)
        
        return output


class FeatureExtractionNetwork(nn.Module):
    """Complete feature extraction network with three branches"""
    
    def __init__(self, input_channels=3, feature_dim=64):
        super(FeatureExtractionNetwork, self).__init__()
        
        self.spatial_branch = SpatialDomainBranch(input_channels, feature_dim)
        self.fourier_branch = FourierDomainBranch(input_channels, feature_dim)
        self.multiscale_branch = MultiScaleBranch(input_channels, feature_dim)
        
        # Attention module to weigh the importance of each branch
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim * 3, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Final output layer
        self.output_layer = nn.Conv2d(feature_dim * 3, feature_dim, kernel_size=1)
    
    def forward(self, x):
        # Process input through each branch
        spatial_features = self.spatial_branch(x)
        fourier_features = self.fourier_branch(x)
        multiscale_features = self.multiscale_branch(x)
        
        # Resize all features to match spatial features size
        target_size = spatial_features.size()[2:]
        if fourier_features.size()[2:] != target_size:
            fourier_features = F.interpolate(fourier_features, size=target_size, mode='bilinear', align_corners=False)
        if multiscale_features.size()[2:] != target_size:
            multiscale_features = F.interpolate(multiscale_features, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate features for attention
        combined = torch.cat([spatial_features, fourier_features, multiscale_features], dim=1)
        
        # Apply attention to weight branch importance
        attention_weights = self.attention(combined)
        attention_weights = attention_weights.unsqueeze(2)  # Shape: (batch, 3, 1, height, width)
        
        # Split features for weighted sum
        features = torch.stack([spatial_features, fourier_features, multiscale_features], dim=1)
        weighted_features = features * attention_weights
        weighted_sum = weighted_features.sum(dim=1)
        
        # Also keep the concatenated features for the output layer
        output = self.output_layer(combined)
        
        return output, (spatial_features, fourier_features, multiscale_features, attention_weights)


class PhaseCorrelationTensorComputation(nn.Module):
    """Module to compute phase correlation tensors from extracted features"""
    
    def __init__(self, feature_dim=64, output_dim=128):
        super(PhaseCorrelationTensorComputation, self).__init__()
        
        # Convert features to phase correlation tensors
        self.conv_layers = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, kernel_size=1)
        )
        
        # Local phase coherence computation
        self.local_coherence = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
        # Attention module for phase correlations
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim + output_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Final layer to produce the tensor
        self.output_layer = nn.Conv2d(feature_dim + output_dim, output_dim, kernel_size=1)
    
    def forward(self, features):
        # Initial processing of features
        processed_features = self.conv_layers(features)
        
        # Compute local phase coherence
        coherence = self.local_coherence(features)
        
        # Combine processed features with coherence
        combined = torch.cat([processed_features, coherence], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        attended_combined = combined * attention_weights
        
        # Final output
        tensor = self.output_layer(attended_combined)
        
        return tensor


def test_feature_extraction(image_path, device='cpu'):
    """Test the feature extraction components with a sample image"""
    # Create preprocessing module
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    # Load and process image
    img = Image.open(image_path)
    processed_img = preprocessor.process(img)
    processed_img = processed_img.unsqueeze(0).to(device)  # Add batch dimension
    
    # Create feature extraction network
    feature_network = FeatureExtractionNetwork().to(device)
    
    # Extract features
    with torch.no_grad():
        features, branch_outputs = feature_network(processed_img)
    
    # Create phase correlation tensor computation module
    tensor_computer = PhaseCorrelationTensorComputation().to(device)
    
    # Compute phase correlation tensor
    with torch.no_grad():
        tensor = tensor_computer(features)
    
    print(f"Input image shape: {processed_img.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Phase correlation tensor shape: {tensor.shape}")
    
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
    tensor = test_feature_extraction(sample_img_path)
    print("Feature extraction test completed successfully!")