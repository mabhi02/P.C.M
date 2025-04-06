#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Correlation Manifold (PCM) - Enhanced Medium Feature Extraction Component

This file implements an enhanced medium-sized feature extraction component for AI-generated 
image detection. It uses a proper ResNet architecture with significantly more parameters
to provide better performance while still being more efficient than the full version.

Key components:
1. Enhanced ResNet Spatial Branch: Based on a deeper ResNet architecture
2. Enhanced Fourier Branch: More complex processing of frequency domain information
3. Multi-Scale Branch: Multiple scales with richer feature representations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
import cv2


class ImagePreprocessor:
    """Class for image preprocessing operations"""
    
    def __init__(self, target_size=(384, 384)):  # Medium size between 256 and 512
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


# Enhanced ResNet building blocks
class Bottleneck(nn.Module):
    """Bottleneck block for enhanced ResNet architecture"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # First 1x1 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Third 1x1 convolution
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample if needed
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class BasicBlock(nn.Module):
    """Basic block for enhanced ResNet architecture"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample if needed
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
    """Enhanced ResNet-based architecture for spatial features"""
    
    def __init__(self, input_channels=3, output_channels=128):  # Increased from 48 to 128
        super(EnhancedSpatialBranch, self).__init__()
        
        # Initial processing
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks - more blocks and wider channels than medium version
        self.layer1 = self._make_layer(BasicBlock, 64, 64, blocks=3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 64, 128, blocks=4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 256, blocks=6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 512, blocks=3, stride=2)
        
        # Enhanced attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.projection = nn.Conv2d(512, output_channels, kernel_size=1)
    
    def _make_layer(self, block, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial processing
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply attention
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        # Project to output dimension
        output = self.projection(attended_features)
        
        return output


class EnhancedFFTLayer(nn.Module):
    """Enhanced layer that computes FFT and separates amplitude and phase"""
    
    def __init__(self):
        super(EnhancedFFTLayer, self).__init__()
    
    def forward(self, x):
        # Apply FFT2 to the last 2 dimensions
        x_fft = torch.fft.fft2(x)
        
        # Shift to center the DC component
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # Extract amplitude and phase
        amplitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Log amplitude for better numerical stability
        log_amplitude = torch.log(amplitude + 1e-10)
        
        # Concatenate along the channel dimension
        output = torch.cat([log_amplitude, phase], dim=1)
        
        return output


class EnhancedFourierBranch(nn.Module):
    """Enhanced branch for Fourier domain feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=128):
        super(EnhancedFourierBranch, self).__init__()
        
        self.fft_layer = EnhancedFFTLayer()
        
        # Enhanced convolutional layers to process frequency domain features
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=3, padding=1),  # *2 because we have amplitude and phase
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_channels, kernel_size=1)
        )
        
        # Enhanced phase-specific processing
        self.phase_processor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # Enhanced output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=1)
        )
    
    def forward(self, x):
        # Compute FFT
        fft_output = self.fft_layer(x)
        
        # Process combined amplitude and phase
        conv_features = self.conv_layers(fft_output)
        
        # Extract and process phase component separately
        phase = fft_output[:, x.size(1):]
        phase_features = self.phase_processor(phase)
        
        # Resize phase_features to match conv_features size if needed
        if phase_features.size()[2:] != conv_features.size()[2:]:
            phase_features = F.interpolate(phase_features, size=conv_features.size()[2:], 
                                           mode='bilinear', align_corners=False)
        
        # Combine features
        combined = torch.cat([conv_features, phase_features], dim=1)
        output = self.output_layer(combined)
        
        return output


class EnhancedPyramidLayer(nn.Module):
    """Enhanced layer that creates a Gaussian pyramid"""
    
    def __init__(self, levels=3):  # Increased from 2 to 3 levels
        super(EnhancedPyramidLayer, self).__init__()
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


class EnhancedMultiScaleBranch(nn.Module):
    """Enhanced branch for multi-scale feature extraction"""
    
    def __init__(self, input_channels=3, output_channels=128, pyramid_levels=3):
        super(EnhancedMultiScaleBranch, self).__init__()
        
        self.pyramid_layer = EnhancedPyramidLayer(levels=pyramid_levels)
        
        # Enhanced convolutional layers for each pyramid level
        self.level_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(pyramid_levels)
        ])
        
        # Upsampling layers to bring all levels to the same resolution
        self.upsample_layers = nn.ModuleList([
            nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=False)
            for i in range(1, pyramid_levels)
        ])
        
        # Enhanced final convolutional layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(256 * pyramid_levels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_channels, kernel_size=1)
        )
    
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


class EnhancedFeatureExtractionNetwork(nn.Module):
    """Enhanced feature extraction network with three branches"""
    
    def __init__(self, input_channels=3, feature_dim=128):
        super(EnhancedFeatureExtractionNetwork, self).__init__()
        
        self.spatial_branch = EnhancedSpatialBranch(input_channels, feature_dim)
        self.fourier_branch = EnhancedFourierBranch(input_channels, feature_dim)
        self.multiscale_branch = EnhancedMultiScaleBranch(input_channels, feature_dim)
        
        # Enhanced attention module
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
        
        # Enhanced output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_dim * 3, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feature_dim, kernel_size=1)
        )
    
    def forward(self, x):
        # Process input through each branch
        spatial_features = self.spatial_branch(x)
        fourier_features = self.fourier_branch(x)
        multiscale_features = self.multiscale_branch(x)
        
        # Resize all features to match spatial features size
        target_size = spatial_features.size()[2:]
        if fourier_features.size()[2:] != target_size:
            fourier_features = F.interpolate(fourier_features, size=target_size, 
                                             mode='bilinear', align_corners=False)
        if multiscale_features.size()[2:] != target_size:
            multiscale_features = F.interpolate(multiscale_features, size=target_size, 
                                                mode='bilinear', align_corners=False)
        
        # Concatenate features for attention
        combined = torch.cat([spatial_features, fourier_features, multiscale_features], dim=1)
        
        # Apply attention to weight branch importance
        attention_weights = self.attention(combined)
        attention_weights = attention_weights.unsqueeze(2)  # Shape: (batch, 3, 1, height, width)
        
        # Split features for weighted sum
        features = torch.stack([spatial_features, fourier_features, multiscale_features], dim=1)
        weighted_features = features * attention_weights
        weighted_sum = weighted_features.sum(dim=1)
        
        # Apply final processing to combined features
        output = self.output_layer(combined)
        
        return output, (spatial_features, fourier_features, multiscale_features, attention_weights)


class EnhancedPhaseCorrelationTensorComputation(nn.Module):
    """Enhanced module to compute phase correlation tensors"""
    
    def __init__(self, feature_dim=128, output_dim=256):
        super(EnhancedPhaseCorrelationTensorComputation, self).__init__()
        
        # Enhanced feature processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_dim, kernel_size=1)
        )
        
        # Enhanced local phase coherence
        self.local_coherence = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim//4),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim//4),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, groups=feature_dim//4),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Enhanced attention for phase correlations
        self.attention = nn.Sequential(
            nn.Conv2d(feature_dim + output_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Enhanced final layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(feature_dim + output_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_dim, kernel_size=1)
        )
    
    def forward(self, features):
        # Initial processing of features
        processed_features = self.conv_layers(features)
        
        # Compute local phase coherence
        coherence = self.local_coherence(features)
        
        # Resize coherence to match processed_features if needed
        if coherence.size()[2:] != processed_features.size()[2:]:
            coherence = F.interpolate(coherence, size=processed_features.size()[2:], 
                                      mode='bilinear', align_corners=False)
        
        # Combine processed features with coherence
        combined = torch.cat([processed_features, coherence], dim=1)
        
        # Apply attention
        attention_weights = self.attention(combined)
        attended_combined = combined * attention_weights
        
        # Final output
        tensor = self.output_layer(attended_combined)
        
        return tensor


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_enhanced_medium_feature_extraction(image_path=None, device='cpu'):
    """Test the enhanced medium feature extraction components"""
    # Create preprocessing module
    preprocessor = ImagePreprocessor(target_size=(384, 384))
    
    if image_path:
        # Load and process image
        img = Image.open(image_path)
        processed_img = preprocessor.process(img)
    else:
        # Create a dummy image if no path provided
        processed_img = torch.rand(3, 384, 384)
    
    processed_img = processed_img.unsqueeze(0).to(device)  # Add batch dimension
    
    # Create feature extraction network
    feature_network = EnhancedFeatureExtractionNetwork().to(device)
    
    # Extract features
    with torch.no_grad():
        features, branch_outputs = feature_network(processed_img)
    
    # Create phase correlation tensor computation module
    tensor_computer = EnhancedPhaseCorrelationTensorComputation().to(device)
    
    # Compute phase correlation tensor
    with torch.no_grad():
        tensor = tensor_computer(features)
    
    # Print model information
    print(f"Input image shape: {processed_img.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Phase correlation tensor shape: {tensor.shape}")
    
    # Count parameters
    feature_params = count_parameters(feature_network)
    tensor_params = count_parameters(tensor_computer)
    total_params = feature_params + tensor_params
    
    print(f"EnhancedFeatureExtractionNetwork parameters: {feature_params:,}")
    print(f"EnhancedPhaseCorrelationTensorComputation parameters: {tensor_params:,}")
    print(f"Total parameters: {total_params:,}")
    
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
        random_img = np.random.rand(384, 384, 3)
        Image.fromarray((random_img * 255).astype(np.uint8)).save(sample_img_path)
    
    # Test feature extraction
    tensor = test_enhanced_medium_feature_extraction(sample_img_path)
    print("Enhanced Medium feature extraction test completed successfully!")