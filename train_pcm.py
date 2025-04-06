#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pcm.py

Training script for PCM (Phase Correlation Manifold) for AI-generated image detection.
This file sets up data loading, optimizers, and training procedures, saving model weights
throughout the process.

Requires:
  - model.py: Contains network architecture definitions
  - Other dependencies: torch, PIL, pandas, etc.

Usage:
  python train_pcm.py --data_path /path/to/data --batch_size 4 --epochs 10 --output_dir ./models

Author: Your Name
Date: 2025-04-05
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from pathlib import Path
import time

# Import models from model.py
from model import (
    EnhancedFeatureExtractionNetwork,
    EnhancedPhaseCorrelationTensorComputation,
    DimensionAdapter,
    ManifoldLearningModule,
    PointCloudGenerator,
    TinyTopologicalFeatureExtraction,
    ClassificationNetwork,
    ImagePreprocessor
)


class AiVsHumanDataset(Dataset):
    """Dataset class for AI vs Human-generated images"""
    def __init__(self, image_files, labels, target_size=(384, 384)):
        self.image_files = image_files
        self.labels = labels
        self.target_size = target_size
        
        # Define transforms using standard torchvision transformations
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label = self.labels[idx]
        
        try:
            # Open image with PIL and convert to RGB
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img_tensor = self.transform(img)
                return img_tensor, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a placeholder tensor and the label
            return torch.ones(3, *self.target_size), label


def load_dataset(csv_path, val_split=0.2):
    """Load and prepare dataset from CSV file"""
    print(f"Loading dataset from: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"CSV columns: {df.columns.tolist()}")
    print(f"First few rows of the CSV:\n{df.head()}")
    
    # Get the base path for the images
    base_path = os.path.dirname(csv_path)
    
    # Parse file paths and labels from CSV
    all_files = []
    labels = []
    
    # Extract file paths and labels from the DataFrame
    for index, row in df.iterrows():
        file_path = os.path.join(base_path, row['file_name'])
        label = int(row['label'])  # Assuming 0 = Natural, 1 = AI
        
        # Verify that the file exists
        if os.path.exists(file_path):
            all_files.append(file_path)
            labels.append(label)
        else:
            print(f"Warning: File not found: {file_path}")
    
    print(f"Total images found: {len(all_files)}")
    print(f"Natural images: {labels.count(0)}")
    print(f"AI-generated images: {labels.count(1)}")
    
    # Create the dataset
    dataset = AiVsHumanDataset(all_files, labels)
    
    # Split the dataset
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    return train_dataset, val_dataset


def initialize_models(device):
    """Initialize all model components and move to specified device"""
    print(f"Initializing models on {device}")
    
    feature_network = EnhancedFeatureExtractionNetwork().to(device)
    tensor_computer = EnhancedPhaseCorrelationTensorComputation().to(device)
    dim_adapter = DimensionAdapter().to(device)
    manifold_module = ManifoldLearningModule().to(device)
    point_cloud_generator = PointCloudGenerator().to(device)
    topo_module = TinyTopologicalFeatureExtraction(
        input_dim=32,
        hidden_dim=32,
        output_dim=16,
        max_edge_length=2.0,
        num_filtrations=10,
        max_dimension=0
    ).to(device)
    classifier = ClassificationNetwork(
        manifold_dim=32,
        topo_dim=16,
        feature_dim=32,
        hidden_dim=64,
        dropout=0.1
    ).to(device)
    
    # Print model sizes
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Enhanced Feature Extraction Network: {count_parameters(feature_network):,} parameters")
    print(f"Phase Correlation Tensor Computation: {count_parameters(tensor_computer):,} parameters")
    print(f"Dimension Adapter: {count_parameters(dim_adapter):,} parameters")
    print(f"Manifold Learning Module: {count_parameters(manifold_module):,} parameters")
    print(f"Point Cloud Generator: {count_parameters(point_cloud_generator):,} parameters")
    print(f"Tiny Topological Feature Extraction: {count_parameters(topo_module):,} parameters")
    print(f"Classification Network: {count_parameters(classifier):,} parameters")
    
    total_params = (count_parameters(feature_network) +
                   count_parameters(tensor_computer) +
                   count_parameters(dim_adapter) +
                   count_parameters(manifold_module) +
                   count_parameters(point_cloud_generator) +
                   count_parameters(topo_module) +
                   count_parameters(classifier))
    
    print(f"Total: {total_params:,} parameters")
    
    # Return all initialized model components
    return {
        'feature_network': feature_network,
        'tensor_computer': tensor_computer,
        'dim_adapter': dim_adapter,
        'manifold_module': manifold_module,
        'point_cloud_generator': point_cloud_generator,
        'topo_module': topo_module,
        'classifier': classifier
    }


def initialize_optimizers(models):
    """Initialize optimizers for each model component"""
    print("Initializing optimizers")
    
    # Define optimizers
    feature_optimizer = optim.Adam(
        list(models['feature_network'].parameters()) +
        list(models['tensor_computer'].parameters()) +
        list(models['dim_adapter'].parameters()),
        lr=1e-4
    )
    
    manifold_optimizer = optim.Adam(
        models['manifold_module'].parameters(),
        lr=1e-4
    )
    
    topo_optimizer = optim.Adam(
        models['topo_module'].parameters(),
        lr=1e-4
    )
    
    classifier_optimizer = optim.Adam(
        models['classifier'].parameters(),
        lr=1e-4
    )
    
    # Learning rate schedulers
    feature_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        feature_optimizer, mode='min', factor=0.5, patience=5
    )
    
    manifold_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        manifold_optimizer, mode='min', factor=0.5, patience=5
    )
    
    topo_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        topo_optimizer, mode='min', factor=0.5, patience=5
    )
    
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        classifier_optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Return all optimizers and schedulers
    return {
        'optimizers': {
            'feature_optimizer': feature_optimizer,
            'manifold_optimizer': manifold_optimizer,
            'topo_optimizer': topo_optimizer,
            'classifier_optimizer': classifier_optimizer
        },
        'schedulers': {
            'feature_scheduler': feature_scheduler,
            'manifold_scheduler': manifold_scheduler,
            'topo_scheduler': topo_scheduler,
            'classifier_scheduler': classifier_scheduler
        }
    }


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """VAE loss for manifold learning"""
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    return recon_loss + beta * kl_loss


def train_epoch(train_loader, models, optimizers, epoch, device, accumulation_steps=2):
    """
    Train for one epoch with gradient accumulation
    """
    # Set all models to training mode
    for model in models.values():
        model.train()
    
    # Initialize metrics
    total_loss = 0
    total_feature_loss = 0
    total_manifold_loss = 0
    total_topo_loss = 0
    total_classifier_loss = 0
    correct = 0
    total = 0
    
    # Training loop
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    # Zero gradients at the start
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    
    # Get model references for cleaner code
    feature_network = models['feature_network']
    tensor_computer = models['tensor_computer']
    dim_adapter = models['dim_adapter']
    manifold_module = models['manifold_module']
    point_cloud_generator = models['point_cloud_generator']
    topo_module = models['topo_module']
    classifier = models['classifier']
    
    # Define classification criterion
    classification_criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass through feature extraction
        features, _ = feature_network(images)
        phase_tensor = tensor_computer(features)
        
        # Apply dimension adapter to match channel count
        adapted_tensor = dim_adapter(phase_tensor)
        
        # Forward pass through manifold learning
        manifold_features, (mu, logvar, z) = manifold_module(adapted_tensor)
        
        # Reconstruct for VAE loss
        recon_tensor = manifold_module.decode(z)
        
        # Calculate manifold loss
        if recon_tensor is not None:
            manifold_loss = vae_loss(recon_tensor, adapted_tensor, mu, logvar)
        else:
            # If the decoder is not fully initialized, use a simpler loss
            manifold_loss = torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
        
        # Generate point cloud for topological analysis
        point_cloud = point_cloud_generator(manifold_features)
        
        # Forward pass through topological analysis
        topo_features, _ = topo_module(point_cloud)
        
        # Forward pass through classifier
        logits, probs, uncertainty = classifier(manifold_features, topo_features)
        
        # Compute losses
        classifier_loss = classification_criterion(logits, labels)
        feature_loss = F.mse_loss(features, features.detach())  # Dummy loss for feature extraction
        topo_loss = F.mse_loss(topo_features, topo_features.detach())  # Dummy loss for topological analysis
        
        # Combined loss
        loss = classifier_loss + 0.1 * manifold_loss + 0.01 * (feature_loss + topo_loss)
        
        # Scale losses by accumulation steps
        scaled_loss = loss / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        # Update metrics
        total_feature_loss += feature_loss.item()
        total_manifold_loss += manifold_loss.item()
        total_topo_loss += topo_loss.item()
        total_classifier_loss += classifier_loss.item()
        total_loss += loss.item()
        
        # Calculate accuracy
        pred = logits.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        
        # Only step optimizers after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            for optimizer in optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    avg_feature_loss = total_feature_loss / len(train_loader)
    avg_manifold_loss = total_manifold_loss / len(train_loader)
    avg_topo_loss = total_topo_loss / len(train_loader)
    avg_classifier_loss = total_classifier_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'feature_loss': avg_feature_loss,
        'manifold_loss': avg_manifold_loss,
        'topo_loss': avg_topo_loss,
        'classifier_loss': avg_classifier_loss,
        'accuracy': accuracy
    }


def validate(val_loader, models, device):
    """Validate the model on the validation set"""
    # Set all models to evaluation mode
    for model in models.values():
        model.eval()
    
    # Get model references for cleaner code
    feature_network = models['feature_network']
    tensor_computer = models['tensor_computer']
    dim_adapter = models['dim_adapter']
    manifold_module = models['manifold_module']
    point_cloud_generator = models['point_cloud_generator']
    topo_module = models['topo_module']
    classifier = models['classifier']
    
    # Initialize metrics
    classification_criterion = nn.CrossEntropyLoss()
    val_loss = 0
    correct = 0
    total = 0
    
    # Store predictions and true labels for additional metrics
    all_preds = []
    all_labels = []
    all_probs = []
    all_uncertainty = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            features, _ = feature_network(images)
            phase_tensor = tensor_computer(features)
            
            # Apply dimension adapter
            adapted_tensor = dim_adapter(phase_tensor)
            
            # Manifold learning
            manifold_features, _ = manifold_module(adapted_tensor)
            
            # Generate point cloud
            point_cloud = point_cloud_generator(manifold_features)
            
            # Topological analysis
            topo_features, _ = topo_module(point_cloud)
            
            # Classification
            logits, probs, uncertainty = classifier(manifold_features, topo_features)
            
            # Compute loss
            loss = classification_criterion(logits, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            pred = logits.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            # Store for metrics
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
            all_uncertainty.append(uncertainty.cpu())
    
    # Calculate metrics
    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Concatenate predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    all_uncertainty = torch.cat(all_uncertainty)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'uncertainty': all_uncertainty
    }


def save_models(models, output_dir, is_best=False):
    """Save all model components"""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each model component
    prefix = "best_" if is_best else ""
    
    for name, model in models.items():
        torch.save(model.state_dict(), os.path.join(output_dir, f"{prefix}{name}.pth"))
    
    print(f"Models saved to {output_dir}" + (" (best)" if is_best else ""))


def save_training_metrics(metrics, output_dir):
    """Save training metrics to a file"""
    metrics_file = os.path.join(output_dir, "training_metrics.pt")
    torch.save(metrics, metrics_file)
    print(f"Training metrics saved to {metrics_file}")


def plot_training_curves(metrics, output_dir):
    """Plot training and validation loss/accuracy curves"""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accuracies'], label='Train Accuracy')
    plt.plot(metrics['val_accuracies'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    metrics_plot_file = os.path.join(output_dir, "training_curves.png")
    plt.savefig(metrics_plot_file)
    print(f"Training curves saved to {metrics_plot_file}")
    plt.close()


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_dataset, val_dataset = load_dataset(args.csv_path, val_split=args.val_split)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Initialize models
    models = initialize_models(device)
    
    # Initialize optimizers and schedulers
    optimization = initialize_optimizers(models)
    optimizers = optimization['optimizers']
    schedulers = optimization['schedulers']
    
    # Training parameters
    best_val_accuracy = 0
    
    # Metrics tracking
    metrics = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
        'epochs': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}\nEpoch {epoch}/{args.epochs}\n{'='*80}")
        
        # Train for one epoch using gradient accumulation
        train_metrics = train_epoch(
            train_loader, 
            models, 
            optimizers, 
            epoch, 
            device, 
            accumulation_steps=args.accumulation_steps
        )
        
        metrics['train_losses'].append(train_metrics['loss'])
        metrics['train_accuracies'].append(train_metrics['accuracy'])
        
        # Validate
        val_metrics = validate(val_loader, models, device)
        metrics['val_losses'].append(val_metrics['loss'])
        metrics['val_accuracies'].append(val_metrics['accuracy'])
        metrics['epochs'].append(epoch)
        
        # Print metrics
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Update learning rate schedulers
        for scheduler in schedulers.values():
            scheduler.step(val_metrics['loss'])
        
        # Save model at regular intervals
        if epoch % args.save_interval == 0:
            save_models(models, args.output_dir, is_best=False)
            save_training_metrics(metrics, args.output_dir)
            plot_training_curves(metrics, args.output_dir)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            print(f"  New best model with accuracy: {best_val_accuracy:.2f}%")
            save_models(models, args.output_dir, is_best=True)
    
    # Training complete
    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Save final model and metrics
    save_models(models, args.output_dir, is_best=False)
    save_training_metrics(metrics, args.output_dir)
    plot_training_curves(metrics, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PCM model for AI-generated image detection")
    
    # Dataset arguments
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file containing image paths and labels")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train (default: 10)")
    parser.add_argument("--accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps (default: 2)")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="Disable CUDA training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers (default: 4)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Directory to save model checkpoints (default: ./models)")
    parser.add_argument("--save_interval", type=int, default=1,
                        help="Save model every N epochs (default: 1)")
    
    args = parser.parse_args()
    main(args)