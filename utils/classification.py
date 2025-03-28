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
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
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
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
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
        
        self.attention = SelfAttention(feature_dim, num_heads, dropout)
        self.feed_forward = FeedForward(feature_dim, hidden_dim, dropout)
    
    def forward(self, x):
        """
        Apply transformer layer to input features
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            
        Returns:
            torch.Tensor: Processed features [batch_size, seq_len, feature_dim]
        """
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


class ClassificationNetwork(nn.Module):
    """Classification network for AI-generated image detection"""
    
    def __init__(self, 
                 manifold_dim=32,         # Dimension of manifold features
                 topo_dim=32,             # Dimension of topological features
                 hidden_dim=128,          # Hidden dimension
                 num_layers=3,            # Number of transformer layers
                 num_heads=4,             # Number of attention heads
                 dropout=0.1              # Dropout rate
                ):
        super(ClassificationNetwork, self).__init__()
        
        # Feature dimension after combination
        self.feature_dim = manifold_dim + topo_dim
        
        # Projection for manifold and topological features
        self.manifold_proj = nn.Linear(manifold_dim, manifold_dim)
        self.topo_proj = nn.Linear(topo_dim, topo_dim)
        
        # Feature type embedding
        self.feature_type_embedding = nn.Embedding(2, self.feature_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                feature_dim=self.feature_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Global pooling
        self.global_attention_pool = nn.Sequential(
            nn.Linear(self.feature_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification: Real vs AI-generated
        )
        
        # Uncertainty estimation
        self.uncertainty = UncertaintyEstimator(self.feature_dim)
    
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
        
        # Project features
        manifold_features = self.manifold_proj(manifold_features)
        topo_features = self.topo_proj(topo_features)
        
        # Create feature sequence with feature type embedding
        feature_types = torch.tensor([0, 1], device=manifold_features.device).unsqueeze(0).expand(batch_size, -1)
        type_embeddings = self.feature_type_embedding(feature_types)
        
        # Combine features with type embeddings
        manifold_with_type = manifold_features.unsqueeze(1) + type_embeddings[:, 0:1, :]
        topo_with_type = topo_features.unsqueeze(1) + type_embeddings[:, 1:2, :]
        
        # Concatenate features to form sequence
        feature_sequence = torch.cat([
            manifold_with_type, 
            topo_with_type
        ], dim=1)
        
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