#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Correlation Manifold (PCM) - Classification Component

This file implements the classification component of the PCM architecture
for AI-generated image detection. It combines features from both the manifold
learning module and topological analysis to make a final classification decision.

Key components:
1. Transformer Architecture: Processes features with self-attention
2. Feature Fusion Mechanism: Combines manifold and topological features
3. Uncertainty Estimation: Provides confidence scores for classifications

The classification module integrates multiple types of evidence to make a robust
determination of whether an image is natural or AI-generated.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
    """Self-attention mechanism for feature enhancement"""
    
    def __init__(self, feature_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        # Ensure head_dim is valid
        if feature_dim % num_heads != 0:
            # Adjust feature_dim to be divisible by num_heads
            self.adjusted_feature_dim = ((feature_dim // num_heads) + 1) * num_heads
            print(f"Warning: feature_dim {feature_dim} not divisible by num_heads {num_heads}. Adjusting to {self.adjusted_feature_dim}")
        else:
            self.adjusted_feature_dim = feature_dim
            
        self.head_dim = self.adjusted_feature_dim // num_heads
        
        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.key = nn.Linear(feature_dim, self.adjusted_feature_dim)
        self.value = nn.Linear(feature_dim, self.adjusted_feature_dim)
        
        # Output projection
        self.output = nn.Linear(self.adjusted_feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization with correct shape
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        Apply self-attention to input features
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: Attention-weighted features [batch_size, seq_len, feature_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Apply layer normalization
        residual = x
        x = self.layer_norm(x)
        
        # Project to query, key, value
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        query = query.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, value)
        
        # Reshape and project to output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.adjusted_feature_dim)
        output = self.output(context)
        
        # Residual connection
        output = output + residual
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network for transformer architecture"""
    
    def __init__(self, feature_dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, feature_dim)
        self.layer_norm = nn.LayerNorm(feature_dim)
    
    def forward(self, x):
        """
        Apply feed-forward network to input features
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: Processed features [batch_size, seq_len, feature_dim]
        """
        residual = x
        x = self.layer_norm(x)
        
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        output = x + residual
        
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward network"""
    
    def __init__(self, feature_dim, hidden_dim, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        
        # Store feature_dim for creating correct normalization layers
        self.feature_dim = feature_dim
        
        # Create self-attention with the correct feature dimension
        self.attention = SelfAttention(feature_dim, num_heads, dropout)
        
        # Create feed-forward with the correct feature dimension
        self.feed_forward = FeedForward(feature_dim, hidden_dim, dropout)
    
    def forward(self, x):
        """
        Apply transformer layer to input features
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: Processed features [batch_size, seq_len, feature_dim]
        """
        # Check input dimension to catch errors early
        batch_size, seq_len, feature_dim = x.size()
        if feature_dim != self.feature_dim:
            print(f"Warning: Input feature dimension {feature_dim} doesn't match expected feature_dim {self.feature_dim}")
        
        # Apply attention and feed-forward
        x = self.attention(x)
        x = self.feed_forward(x)
        
        return x


class UncertaintyEstimator(nn.Module):
    """Estimates classification uncertainty using evidential deep learning"""
    
    def __init__(self, feature_dim, hidden_dim=64):
        super(UncertaintyEstimator, self).__init__()
        
        self.evidence_network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Alpha, beta, gamma, delta
        )
    
    def forward(self, x):
        """
        Estimate classification uncertainty
        
        Args:
            x (torch.Tensor): Input features [batch_size, feature_dim]
            
        Returns:
            tuple: (class_probs, uncertainty)
                - class_probs: Classification probabilities [batch_size, 2]
                - uncertainty: Uncertainty scores [batch_size]
        """
        # Compute Dirichlet distribution parameters
        evidence = torch.exp(self.evidence_network(x))
        alpha = evidence + 1  # Ensure alpha > 1
        
        # Class probabilities
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S
        
        # Uncertainty (total variance)
        K = 2  # Number of classes
        uncertainty = K / torch.sum(alpha, dim=1)
        
        return probs, uncertainty


class FeatureProjector(nn.Module):
    """Projects features to a common dimension for transformer processing"""
    
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
    """Classification network for AI-generated image detection"""
    
    def __init__(self, 
                 manifold_dim=32,         # Dimension of manifold features
                 topo_dim=32,             # Dimension of topological features
                 feature_dim=64,          # Common feature dimension for transformer
                 hidden_dim=128,          # Hidden dimension
                 num_layers=3,            # Number of transformer layers
                 num_heads=4,             # Number of attention heads
                 dropout=0.1              # Dropout rate
                ):
        super(ClassificationNetwork, self).__init__()
        
        # Store dimensions
        self.manifold_dim = manifold_dim
        self.topo_dim = topo_dim
        self.feature_dim = feature_dim
        
        # Feature projectors to common dimension
        self.manifold_projector = FeatureProjector(manifold_dim, feature_dim // 2, dropout)
        self.topo_projector = FeatureProjector(topo_dim, feature_dim // 2, dropout)
        
        # Feature type embeddings
        self.manifold_type_embedding = nn.Parameter(torch.randn(1, 1, feature_dim // 2))
        self.topo_type_embedding = nn.Parameter(torch.randn(1, 1, feature_dim // 2))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.global_attention_pool = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification: Real vs AI-generated
        )
        
        # Uncertainty estimation
        self.uncertainty = UncertaintyEstimator(feature_dim)
    
    def forward(self, manifold_features, topo_features):
        """
        Classify images as natural or AI-generated
        
        Args:
            manifold_features (torch.Tensor): Manifold features [batch_size, manifold_dim]
            topo_features (torch.Tensor): Topological features [batch_size, topo_dim]
            
        Returns:
            tuple: (logits, class_probs, uncertainty)
                - logits: Classification logits [batch_size, 2]
                - class_probs: Classification probabilities [batch_size, 2]
                - uncertainty: Uncertainty scores [batch_size]
        """
        batch_size = manifold_features.size(0)
        device = manifold_features.device
        
        # Project features to common dimension
        manifold_projected = self.manifold_projector(manifold_features)  # [batch_size, feature_dim//2]
        topo_projected = self.topo_projector(topo_features)  # [batch_size, feature_dim//2]
        
        # Add sequence dimension and type embeddings
        manifold_seq = manifold_projected.unsqueeze(1)  # [batch_size, 1, feature_dim//2]
        topo_seq = topo_projected.unsqueeze(1)  # [batch_size, 1, feature_dim//2]
        
        # Add type embeddings
        manifold_embedding = self.manifold_type_embedding.expand(batch_size, -1, -1)
        topo_embedding = self.topo_type_embedding.expand(batch_size, -1, -1)
        
        manifold_seq = manifold_seq + manifold_embedding
        topo_seq = topo_seq + topo_embedding
        
        # Concatenate along feature dimension
        manifold_topo_combined = torch.cat([manifold_seq, topo_seq], dim=2)  # [batch_size, 1, feature_dim]
        
        # Replicate the combined features to create a sequence
        feature_sequence = manifold_topo_combined.expand(-1, 2, -1)  # [batch_size, 2, feature_dim]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            feature_sequence = layer(feature_sequence)
        
        # Global attention pooling
        attention_weights = self.global_attention_pool(feature_sequence)
        pooled_features = torch.sum(feature_sequence * attention_weights, dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        
        # Uncertainty estimation
        probs, uncertainty = self.uncertainty(pooled_features)
        
        return logits, probs, uncertainty
    
    def mse_loss(self, x, x_target):
        """
        Compute MSE loss between two tensors
        
        Args:
            x (torch.Tensor): Input tensor
            x_target (torch.Tensor): Target tensor
            
        Returns:
            torch.Tensor: MSE loss
        """
        return F.mse_loss(x, x_target)


class FeatureFusionModule(nn.Module):
    """Optional module for feature fusion with attention"""
    
    def __init__(self, manifold_dim, topo_dim, output_dim, dropout=0.1):
        super(FeatureFusionModule, self).__init__()
        
        # Feature projections
        self.manifold_proj = nn.Linear(manifold_dim, output_dim)
        self.topo_proj = nn.Linear(topo_dim, output_dim)
        
        # Cross-attention
        self.cross_attention = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, manifold_features, topo_features):
        # Project features
        manifold_proj = self.manifold_proj(manifold_features)
        topo_proj = self.topo_proj(topo_features)
        
        # Concat for attention
        concat_features = torch.cat([manifold_proj, topo_proj], dim=1)
        
        # Compute attention weights
        attention = self.cross_attention(concat_features)
        
        # Apply attention
        fused = attention[:, 0:1] * manifold_proj + attention[:, 1:2] * topo_proj
        
        # Final projection
        output = self.output_proj(fused)
        
        return output


class MultiScaleFeatureProcessor(nn.Module):
    """Process features at multiple scales for better classification"""
    
    def __init__(self, feature_dim, num_scales=3, dropout=0.1):
        super(MultiScaleFeatureProcessor, self).__init__()
        
        # Multi-scale processing
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_scales)
        ])
        
        # Scale attention
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim * num_scales, num_scales),
            nn.Softmax(dim=1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):
        # Process at different scales
        multi_scale_features = []
        for processor in self.scale_processors:
            scaled_features = processor(x)
            multi_scale_features.append(scaled_features)
        
        # Concatenate for attention
        concat_features = torch.cat(multi_scale_features, dim=1)
        
        # Compute attention weights
        attention = self.scale_attention(concat_features)
        
        # Apply attention
        output = torch.zeros_like(x)
        for i, features in enumerate(multi_scale_features):
            output += attention[:, i:i+1] * features
        
        # Final projection
        output = self.output_proj(output)
        
        return output


def classify_features(manifold_features, topo_features, device='cpu'):
    """
    Classify features using the classification network
    
    Args:
        manifold_features (torch.Tensor): Manifold features [batch_size, manifold_dim]
        topo_features (torch.Tensor): Topological features [batch_size, topo_dim]
        
    Returns:
        tuple: (predictions, probabilities, uncertainty)
    """
    # Create classification network
    classifier = ClassificationNetwork().to(device)
    
    # Process features
    with torch.no_grad():
        logits, probs, uncertainty = classifier(manifold_features, topo_features)
    
    # Get predictions
    predictions = torch.argmax(logits, dim=1)
    
    print(f"Manifold feature shape: {manifold_features.shape}")
    print(f"Topological feature shape: {topo_features.shape}")
    print(f"Prediction: {'AI-Generated' if predictions.item() == 1 else 'Natural'}")
    print(f"Probability: {probs[0][predictions.item()].item():.4f}")
    print(f"Uncertainty: {uncertainty.item():.4f}")
    
    return predictions, probs, uncertainty


# Auxiliary loss functions for training
def contrastive_loss(features1, features2, labels, margin=1.0):
    """Contrastive loss for similar/dissimilar feature pairs"""
    distances = F.pairwise_distance(features1, features2)
    similar_loss = (labels * distances**2)
    dissimilar_loss = ((1-labels) * F.relu(margin - distances)**2)
    return torch.mean(similar_loss + dissimilar_loss)


def triplet_loss(anchor, positive, negative, margin=1.0):
    """Triplet loss for feature embedding learning"""
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    return torch.mean(F.relu(pos_dist - neg_dist + margin))


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(focal_loss)


if __name__ == "__main__":
    # If this file is run directly, perform a quick test
    import os
    import torch
    from feature_extraction import FeatureExtractionNetwork, PhaseCorrelationTensorComputation, ImagePreprocessor
    from manifold_learning import ManifoldLearningModule
    from topological_analysis import TopologicalFeatureExtraction
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
        topo_features, _ = topo_module(point_cloud)
    
    # Classification
    predictions, probs, uncertainty = classify_features(manifold_features, topo_features, device)
    
    print("Classification test completed successfully!")