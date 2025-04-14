import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib
# Set non-interactive backend for matplotlib (no GUI needed on cluster)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm  # Using regular tqdm instead of notebook version
import random
import pandas as pd
import argparse
from pathlib import Path

# Parse command-line arguments for flexibility in cluster environment
parser = argparse.ArgumentParser(description="PCM Model Training on HPC")
parser.add_argument('--data_dir', type=str, default='/home/a/av/avm/data', 
                    help='Directory containing the dataset')
parser.add_argument('--save_dir', type=str, default='/home/a/av/avm/model_checkpoints', 
                    help='Directory to save model checkpoints')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--gradient_accumulation', type=int, default=8, 
                    help='Number of gradient accumulation steps')
parser.add_argument('--save_freq', type=int, default=50, 
                    help='Save checkpoint frequency (batches)')
parser.add_argument('--resume', type=str, default=None, 
                    help='Path to checkpoint to resume from')
args = parser.parse_args()

# Import our model from mediumModel.py
from mediumModel import (
    EnhancedFeatureExtractionNetwork,
    EnhancedPhaseCorrelationTensorComputation,
    DimensionAdapter,
    ManifoldLearningModule,
    PointCloudGenerator,
    TinyTopologicalFeatureExtraction,
    ClassificationNetwork,
    ImagePreprocessor
)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories
os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'models/medium'), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, 'models/medium/checkpoints'), exist_ok=True)

# Initialize model components from mediumModel.py with medium-sized parameters
print("Initializing model components...")
feature_network = EnhancedFeatureExtractionNetwork(feature_dim=256).to(device)
tensor_computer = EnhancedPhaseCorrelationTensorComputation(feature_dim=256, output_dim=512).to(device)

# Create the dimension adapter
dim_adapter = DimensionAdapter(input_dim=512, output_dim=256).to(device)

# Initialize the manifold module
manifold_module = ManifoldLearningModule(
    input_dim=256,
    hidden_dim=512,
    latent_dim=64,
    gnn_hidden_dim=128
).to(device)

# For topological analysis, we need to create point clouds from manifold features
point_cloud_generator = PointCloudGenerator(num_points=64).to(device)

# Use TinyTopologicalFeatureExtraction from mediumModel.py
topo_module = TinyTopologicalFeatureExtraction(
    input_dim=64,           # Match the output dimension from ManifoldLearningModule
    hidden_dim=64,          # Hidden dimension
    output_dim=32,          # Output dimension
    max_edge_length=2.0,    # Maximum edge length for filtration
    num_filtrations=16,     # Number of filtration values
    max_dimension=1         # Increased to dimension 1 for medium model
).to(device)

# ClassificationNetwork from mediumModel.py
classifier = ClassificationNetwork(
    manifold_dim=64,        # Match the output dimension from ManifoldLearningModule
    topo_dim=32,            # Match the output dimension from TinyTopologicalFeatureExtraction
    feature_dim=128,        # Feature dimension
    hidden_dim=256,         # Hidden dimension
    num_layers=4,           # Number of transformer layers
    num_heads=6,            # Number of attention heads
    dropout=0.15            # Dropout rate
).to(device)

# Print model sizes using count_parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Enhanced Feature Extraction Network: {count_parameters(feature_network):,} parameters")
print(f"Phase Correlation Tensor Computation: {count_parameters(tensor_computer):,} parameters")
print(f"Dimension Adapter: {count_parameters(dim_adapter):,} parameters")
print(f"Manifold Learning Module: {count_parameters(manifold_module):,} parameters")
print(f"Point Cloud Generator: {count_parameters(point_cloud_generator):,} parameters")
print(f"Tiny Topological Feature Extraction: {count_parameters(topo_module):,} parameters")
print(f"Classification Network: {count_parameters(classifier):,} parameters")
total_params = (count_parameters(feature_network) + count_parameters(tensor_computer) + 
               count_parameters(dim_adapter) + count_parameters(manifold_module) + 
               count_parameters(point_cloud_generator) + count_parameters(topo_module) + 
               count_parameters(classifier))
print(f"Total: {total_params:,} parameters")

# Custom dataset class with upsampling to 448x448
class CustomImageDataset(Dataset):
    def __init__(self, image_files, labels, target_size=(448, 448)):
        self.image_files = image_files
        self.labels = labels
        self.target_size = target_size
        
        # Define transforms using standard torchvision transformations
        self.transform = transforms.Compose([
            transforms.Resize(target_size),  # Upsample to 448x448
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
            return torch.zeros(3, *self.target_size), label

def load_dataset_from_csv(csv_path, verbose=True):
    """Load image paths and labels from a CSV file"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    if verbose:
        print(f"\nLoading from: {csv_path}")
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"First few rows of the CSV:\n{df.head()}")
    
    # Get the base path for the images
    base_path = os.path.dirname(csv_path)
    
    # Parse file paths and labels from CSV
    all_files = []
    labels = []
    
    # Extract file paths and labels from the DataFrame
    for index, row in df.iterrows():
        # Handle different column names for file paths
        if 'image_path' in df.columns:
            file_name = row['image_path']
        elif 'file_name' in df.columns:
            file_name = row['file_name']
        else:
            raise ValueError(f"Could not find image path column in CSV: {csv_path}")
        
        # Construct full file path
        file_path = os.path.join(base_path, file_name)
        
        # Handle different label formats
        if 'label' in df.columns:
            # Check if label is already a number or a string
            if isinstance(row['label'], (int, float)) or str(row['label']).isdigit():
                label = int(row['label'])
            else:
                # If label is in text format (AI or Human)
                label_text = str(row['label']).lower()
                label = 1 if label_text == 'ai' else 0
        else:
            raise ValueError(f"Could not find label column in CSV: {csv_path}")
        
        # Try multiple path resolutions
        if os.path.exists(file_path):
            all_files.append(file_path)
            labels.append(label)
        else:
            # Try with just the basename
            alt_path = os.path.join(base_path, os.path.basename(file_name))
            if os.path.exists(alt_path):
                all_files.append(alt_path)
                labels.append(label)
            else:
                # Try looking in a standard subdirectory
                for subdir in ['train_data', 'images', 'data']:
                    alt_path = os.path.join(base_path, subdir, os.path.basename(file_name))
                    if os.path.exists(alt_path):
                        all_files.append(alt_path)
                        labels.append(label)
                        break
                else:  # Executed if the for loop completes normally (no break)
                    if verbose and index < 10:  # Only show first few warnings
                        print(f"Warning: File not found: {file_path}")
    
    if verbose:
        print(f"Successfully loaded {len(all_files)} images from {csv_path}")
        print(f"Human images (label 0): {labels.count(0)}")
        print(f"AI-generated images (label 1): {labels.count(1)}")
    
    return all_files, labels

def combine_datasets(data_dir):
    """Load and combine datasets from both CSV files in the OCF environment"""
    # Paths to the CSV files - updated for OCF environment
    csv_path1 = os.path.join(data_dir, 'images', 'train.csv')
    csv_path2 = os.path.join(data_dir, 'ai_vs_human_gener', 'train.csv')
    
    # Load datasets
    try:
        files1, labels1 = load_dataset_from_csv(csv_path1)
        print(f"Dataset 1: {len(files1)} images loaded")
    except Exception as e:
        print(f"Error loading dataset 1: {e}")
        files1, labels1 = [], []
    
    try:
        files2, labels2 = load_dataset_from_csv(csv_path2)
        print(f"Dataset 2: {len(files2)} images loaded")
    except Exception as e:
        print(f"Error loading dataset 2: {e}")
        files2, labels2 = [], []
    
    # Combine the datasets
    all_files = files1 + files2
    all_labels = labels1 + labels2
    
    print(f"\nCombined dataset statistics:")
    print(f"Total images: {len(all_files)}")
    print(f"Human images (label 0): {all_labels.count(0)}")
    print(f"AI-generated images (label 1): {all_labels.count(1)}")
    
    return all_files, all_labels

def create_dataset(data_dir, batch_size):
    """Create dataset and split it for training on HPC"""
    # Combine datasets
    all_files, all_labels = combine_datasets(data_dir)
    
    if len(all_files) == 0:
        print("No images were loaded. Please check file paths.")
        return None, None
    
    # Create the dataset with upsampling to 448x448
    dataset = CustomImageDataset(all_files, all_labels, target_size=(448, 448))
    
    # Split the dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {train_size} images")
    print(f"Validation set: {val_size} images")
    
    # Create data loaders with adjusted num_workers for HPC
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,  # Pin memory for faster data transfer to GPU
        persistent_workers=True,  # Keep workers alive between batches
        prefetch_factor=2  # Prefetch batches to reduce I/O bottlenecks
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Check a batch of data
    try:
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Sample labels: {labels}")
    except Exception as e:
        import traceback
        print(f"Error loading first batch: {e}")
        traceback.print_exc()
    
    return train_loader, val_loader

# Define loss functions
classification_criterion = nn.CrossEntropyLoss()

# Define optimizers - using separate optimizers for different components
# Adjust learning rates slightly for the medium model (slightly lower than small model)
feature_optimizer = optim.Adam(list(feature_network.parameters()) + 
                              list(tensor_computer.parameters()) + 
                              list(dim_adapter.parameters()), lr=8e-5)
manifold_optimizer = optim.Adam(manifold_module.parameters(), lr=8e-5)
topo_optimizer = optim.Adam(list(point_cloud_generator.parameters()) +
                           list(topo_module.parameters()), lr=8e-5)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=8e-5)

# Learning rate schedulers with adjusted patience for the medium model
feature_scheduler = optim.lr_scheduler.ReduceLROnPlateau(feature_optimizer, mode='min', 
                                                        factor=0.5, patience=3, min_lr=1e-6)
manifold_scheduler = optim.lr_scheduler.ReduceLROnPlateau(manifold_optimizer, mode='min', 
                                                         factor=0.5, patience=3, min_lr=1e-6)
topo_scheduler = optim.lr_scheduler.ReduceLROnPlateau(topo_optimizer, mode='min', 
                                                     factor=0.5, patience=3, min_lr=1e-6)
classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='min', 
                                                           factor=0.5, patience=3, min_lr=1e-6)

def vae_loss(recon_x, x, mu, logvar):
    """
    Computes the VAE loss function with robust handling for significant dimension mismatches.
    """
    # Check if reconstruction was successful
    if recon_x is None:
        # If decode failed, just use KL divergence
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # Handle the specific case where recon_x is [batch_size, 768] and x is [batch_size, 256, 6, 6]
    if len(recon_x.shape) == 2 and len(x.shape) == 4:
        print(f"Handling dimension mismatch: recon_x {recon_x.size()} vs x {x.size()}")
        
        # Since recon_x seems to be a fixed size that doesn't match the flattened input,
        # we need a different approach than trying to reshape or compare directly
        
        # Option 1: Project recon_x to match the flattened size of x
        batch_size = x.size(0)
        flattened_size = x.view(batch_size, -1).size(1)
        
        # Create a projection layer on-the-fly if needed
        if not hasattr(vae_loss, 'projection_layer') or vae_loss.projection_layer.in_features != recon_x.size(1) or vae_loss.projection_layer.out_features != flattened_size:
            vae_loss.projection_layer = nn.Linear(recon_x.size(1), flattened_size).to(recon_x.device)
            # Initialize weights close to identity-like mapping
            nn.init.xavier_uniform_(vae_loss.projection_layer.weight)
            nn.init.zeros_(vae_loss.projection_layer.bias)
            print(f"Created new projection layer: {recon_x.size(1)} -> {flattened_size}")
        
        # Project recon_x to match flattened x dimensions
        projected_recon_x = vae_loss.projection_layer(recon_x)
        
        # Now compare with flattened x
        flattened_x = x.view(batch_size, -1)
        recon_loss = F.mse_loss(projected_recon_x, flattened_x, reduction='sum') / batch_size
        
        # KL divergence remains the same
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Use a smaller weight for the KL term during early training
        kl_weight = min(0.8, 0.1 + 0.01 * getattr(vae_loss, 'call_count', 0))
        
        # Increment call count
        vae_loss.call_count = getattr(vae_loss, 'call_count', 0) + 1
        
        return recon_loss + kl_weight * kld_loss
    
    # For other shape mismatches, try to handle them appropriately
    elif recon_x.size() != x.size():
        print(f"Other dimension mismatch in VAE loss: recon_x {recon_x.size()} vs x {x.size()}")
        try:
            # Try to match dimensions by reshaping if possible
            batch_size = x.size(0)
            
            # If both can be flattened, use MSE on flattened versions
            recon_x_flat = recon_x.view(batch_size, -1)
            x_flat = x.view(batch_size, -1)
            
            # If dimensions still don't match, use a simpler approach
            if recon_x_flat.size(1) != x_flat.size(1):
                print(f"Cannot match dimensions even when flattened: {recon_x_flat.size()} vs {x_flat.size()}")
                # Just use KL divergence as the loss
                return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            
            # If dimensions match after flattening
            recon_loss = F.mse_loss(recon_x_flat, x_flat, reduction='sum') / batch_size
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            return recon_loss + 0.8 * kld_loss
            
        except Exception as e:
            print(f"Failed to handle VAE loss: {e}")
            # Fall back to just KL divergence if all else fails
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # If shapes match exactly, use standard VAE loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + 0.8 * kld_loss

# Function to save models with explicit path
def save_models(save_dir, epoch=None, metrics=None):
    # Create directory if it doesn't exist
    model_dir = os.path.join(save_dir, 'models/medium')
    os.makedirs(model_dir, exist_ok=True)
    
    # Add epoch information to filenames if provided
    suffix = f"_epoch{epoch}" if epoch is not None else ""
    
    # Save model states
    torch.save(feature_network.state_dict(), os.path.join(model_dir, f'feature_network{suffix}.pth'))
    torch.save(tensor_computer.state_dict(), os.path.join(model_dir, f'tensor_computer{suffix}.pth'))
    torch.save(dim_adapter.state_dict(), os.path.join(model_dir, f'dim_adapter{suffix}.pth'))
    torch.save(manifold_module.state_dict(), os.path.join(model_dir, f'manifold_module{suffix}.pth'))
    torch.save(point_cloud_generator.state_dict(), os.path.join(model_dir, f'point_cloud_generator{suffix}.pth'))
    torch.save(topo_module.state_dict(), os.path.join(model_dir, f'topo_module{suffix}.pth'))
    torch.save(classifier.state_dict(), os.path.join(model_dir, f'classifier{suffix}.pth'))
    
    # Save optimizer states
    torch.save(feature_optimizer.state_dict(), os.path.join(model_dir, f'feature_optimizer{suffix}.pth'))
    torch.save(manifold_optimizer.state_dict(), os.path.join(model_dir, f'manifold_optimizer{suffix}.pth'))
    torch.save(topo_optimizer.state_dict(), os.path.join(model_dir, f'topo_optimizer{suffix}.pth'))
    torch.save(classifier_optimizer.state_dict(), os.path.join(model_dir, f'classifier_optimizer{suffix}.pth'))
    
    # Save scheduler states
    torch.save(feature_scheduler.state_dict(), os.path.join(model_dir, f'feature_scheduler{suffix}.pth'))
    torch.save(manifold_scheduler.state_dict(), os.path.join(model_dir, f'manifold_scheduler{suffix}.pth'))
    torch.save(topo_scheduler.state_dict(), os.path.join(model_dir, f'topo_scheduler{suffix}.pth'))
    torch.save(classifier_scheduler.state_dict(), os.path.join(model_dir, f'classifier_scheduler{suffix}.pth'))
    
    # Save metrics if provided
    if metrics is not None:
        # Convert tensors to lists for JSON serialization
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v
        
        import json
        with open(os.path.join(model_dir, f'metrics{suffix}.json'), 'w') as f:
            json.dump(serializable_metrics, f)
    
    print(f"Model saved to {model_dir}{' for epoch ' + str(epoch) if epoch is not None else ''}")

# Training function with gradient accumulation for medium model
def train_epoch_with_accumulation(train_loader, epoch, accumulation_steps=8):
    # Set models to training mode
    feature_network.train()
    tensor_computer.train()
    dim_adapter.train()
    manifold_module.train()
    point_cloud_generator.train()
    topo_module.train()
    classifier.train()
    
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
    feature_optimizer.zero_grad()
    manifold_optimizer.zero_grad()
    topo_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        try:
            # Free memory before forward pass
            torch.cuda.empty_cache()
            
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
                # Use the VAE loss function with appropriate shape checking
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
            
            # More meaningful feature extraction loss for medium model
            feature_loss = torch.mean(torch.abs(features)) * 0.01
            
            # More meaningful topological loss for medium model
            topo_loss = torch.mean(torch.abs(topo_features)) * 0.01
            
            # Combined loss with adjusted weighting for medium model
            loss = classifier_loss + 0.2 * manifold_loss + 0.05 * feature_loss + 0.05 * topo_loss
            
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
                # Apply gradient clipping for stability with medium model
                torch.nn.utils.clip_grad_norm_(feature_network.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(tensor_computer.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(dim_adapter.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(manifold_module.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(point_cloud_generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(topo_module.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                
                feature_optimizer.step()
                manifold_optimizer.step()
                topo_optimizer.step()
                classifier_optimizer.step()
                
                # Zero gradients
                feature_optimizer.zero_grad()
                manifold_optimizer.zero_grad()
                topo_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
            
            # Save checkpoint at specified frequency
            if args.save_freq > 0 and ((batch_idx + 1) % args.save_freq == 0):
                checkpoint_path = os.path.join(args.save_dir, f'models/medium/checkpoints/epoch{epoch}_batch{batch_idx}')
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Calculate current metrics
                current_accuracy = 100. * correct / total if total > 0 else 0
                
                # Save component checkpoints
                torch.save(feature_network.state_dict(), f'{checkpoint_path}/feature_network.pth')
                torch.save(tensor_computer.state_dict(), f'{checkpoint_path}/tensor_computer.pth')
                torch.save(dim_adapter.state_dict(), f'{checkpoint_path}/dim_adapter.pth')
                torch.save(manifold_module.state_dict(), f'{checkpoint_path}/manifold_module.pth')
                torch.save(point_cloud_generator.state_dict(), f'{checkpoint_path}/point_cloud_generator.pth')
                torch.save(topo_module.state_dict(), f'{checkpoint_path}/topo_module.pth')
                torch.save(classifier.state_dict(), f'{checkpoint_path}/classifier.pth')
                
                print(f"\nCheckpoint saved at batch {batch_idx+1} with accuracy: {current_accuracy:.2f}%")
            
            # Update progress bar but not too frequently to reduce overhead
            if batch_idx % 5 == 0:
                progress_bar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total if total > 0 else 0
                })
        
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for better debugging
            
            # Free memory on error
            torch.cuda.empty_cache()
            
            # Skip this batch and continue
            continue
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    avg_feature_loss = total_feature_loss / len(train_loader)
    avg_manifold_loss = total_manifold_loss / len(train_loader)
    avg_topo_loss = total_topo_loss / len(train_loader)
    avg_classifier_loss = total_classifier_loss / len(train_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return {
        'loss': avg_loss,
        'feature_loss': avg_feature_loss,
        'manifold_loss': avg_manifold_loss,
        'topo_loss': avg_topo_loss,
        'classifier_loss': avg_classifier_loss,
        'accuracy': accuracy
    }

# Validation function for medium model with aggressive memory cleaning
def validate(val_loader):
    # Set models to evaluation mode
    feature_network.eval()
    tensor_computer.eval()
    dim_adapter.eval()
    manifold_module.eval()
    point_cloud_generator.eval()
    topo_module.eval()
    classifier.eval()
    
    # Initialize metrics
    val_loss = 0
    val_feature_loss = 0
    val_manifold_loss = 0
    val_topo_loss = 0
    correct = 0
    total = 0
    
    # Store predictions and true labels for additional metrics
    all_preds = []
    all_labels = []
    all_probs = []
    all_uncertainty = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            try:
                # Clear cache before processing
                torch.cuda.empty_cache()
                
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                features, _ = feature_network(images)
                phase_tensor = tensor_computer(features)
                
                # Apply dimension adapter
                adapted_tensor = dim_adapter(phase_tensor)
                
                # Manifold learning
                manifold_features, (mu, logvar, z) = manifold_module(adapted_tensor)
                
                # Try to reconstruct for manifold evaluation
                recon_tensor = manifold_module.decode(z)
                
                # Calculate manifold loss if possible
                if recon_tensor is not None:
                    manifold_loss = vae_loss(recon_tensor, adapted_tensor, mu, logvar)
                else:
                    manifold_loss = torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
                
                # Generate point cloud
                point_cloud = point_cloud_generator(manifold_features)
                
                # Topological analysis
                topo_features, _ = topo_module(point_cloud)
                
                # Classification
                logits, probs, uncertainty = classifier(manifold_features, topo_features)
                
                # Compute losses
                classification_loss = classification_criterion(logits, labels)
                feature_loss = torch.mean(torch.abs(features)) * 0.01
                topo_loss = torch.mean(torch.abs(topo_features)) * 0.01
                
                # Combined loss with adjusted weighting for medium model (same as training)
                loss = classification_loss + 0.2 * manifold_loss + 0.05 * feature_loss + 0.05 * topo_loss
                
                # Update metrics
                val_loss += loss.item()
                val_feature_loss += feature_loss.item()
                val_manifold_loss += manifold_loss.item()
                val_topo_loss += topo_loss.item()
                
                # Calculate accuracy
                pred = logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
                
                # Store for metrics
                all_preds.append(pred.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                all_uncertainty.append(uncertainty.cpu())
                
                # Explicitly delete tensors to free memory
                del images, labels, features, phase_tensor, adapted_tensor
                del manifold_features, mu, logvar, z, recon_tensor
                del point_cloud, topo_features, logits, probs, uncertainty
                del pred, loss, classification_loss, feature_loss, topo_loss, manifold_loss
                torch.cuda.empty_cache()
            
            except Exception as e:
                print(f"Error in validation: {e}")
                import traceback
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
    
    # Calculate metrics
    avg_loss = val_loss / len(val_loader)
    avg_feature_loss = val_feature_loss / len(val_loader)
    avg_manifold_loss = val_manifold_loss / len(val_loader)
    avg_topo_loss = val_topo_loss / len(val_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    # Concatenate predictions and labels
    if all_preds and all_labels and all_probs and all_uncertainty:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        all_uncertainty = torch.cat(all_uncertainty)
    else:
        print("Warning: Some metrics could not be collected during validation")
        all_preds = torch.tensor([])
        all_labels = torch.tensor([])
        all_probs = torch.tensor([])
        all_uncertainty = torch.tensor([])
    
    return {
        'loss': avg_loss,
        'feature_loss': avg_feature_loss,
        'manifold_loss': avg_manifold_loss,
        'topo_loss': avg_topo_loss,
        'accuracy': accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'uncertainty': all_uncertainty
    }

def resume_from_checkpoint(checkpoint_path):
    """Resume training from a checkpoint"""
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Load model weights
    feature_network.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_network.pth')))
    tensor_computer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'tensor_computer.pth')))
    dim_adapter.load_state_dict(torch.load(os.path.join(checkpoint_path, 'dim_adapter.pth')))
    manifold_module.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_module.pth')))
    point_cloud_generator.load_state_dict(torch.load(os.path.join(checkpoint_path, 'point_cloud_generator.pth')))
    topo_module.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_module.pth')))
    classifier.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier.pth')))
    
    # Load optimizer states if available
    if os.path.exists(os.path.join(checkpoint_path, 'feature_optimizer.pth')):
        feature_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_optimizer.pth')))
        manifold_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_optimizer.pth')))
        topo_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_optimizer.pth')))
        classifier_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier_optimizer.pth')))
    
    # Load scheduler states if available
    if os.path.exists(os.path.join(checkpoint_path, 'feature_scheduler.pth')):
        feature_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_scheduler.pth')))
        manifold_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_scheduler.pth')))
        topo_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_scheduler.pth')))
        classifier_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier_scheduler.pth')))
    
    print("Checkpoint loaded successfully")

# Main training function
def train_model():
    # Create dataset and loaders
    print("\nPreparing datasets...")
    train_loader, val_loader = create_dataset(args.data_dir, args.batch_size)
    
    if train_loader is None or val_loader is None:
        print("Error: Could not create data loaders. Exiting.")
        return
    
    # Training parameters
    num_epochs = args.epochs
    gradient_accumulation_steps = args.gradient_accumulation
    best_val_accuracy = 0
    early_stopping_patience = 5
    early_stopping_counter = 0
    
    # Initialize metrics tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        resume_from_checkpoint(args.resume)
        
        # Try to load previous metrics if available
        metrics_path = os.path.join(os.path.dirname(args.resume), 'training_metrics.pt')
        if os.path.exists(metrics_path):
            metrics = torch.load(metrics_path)
            train_losses = metrics.get('train_losses', [])
            train_accuracies = metrics.get('train_accuracies', [])
            val_losses = metrics.get('val_losses', [])
            val_accuracies = metrics.get('val_accuracies', [])
            best_val_accuracy = metrics.get('best_accuracy', 0)
            print(f"Loaded previous metrics. Best validation accuracy: {best_val_accuracy:.2f}%")
    
    print(f"\nStarting training for {num_epochs} epochs with {total_params:,} parameters")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    try:
        for epoch in range(1, num_epochs + 1):
            try:
                print(f"\n{'='*80}\nEpoch {epoch}/{num_epochs}\n{'='*80}")
                
                # Start timing
                start_time.record()
                
                # Train for one epoch
                train_metrics = train_epoch_with_accumulation(
                    train_loader, 
                    epoch, 
                    accumulation_steps=gradient_accumulation_steps
                )
                
                # End timing
                end_time.record()
                torch.cuda.synchronize()
                epoch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                
                # Store training metrics
                train_losses.append(train_metrics['loss'])
                train_accuracies.append(train_metrics['accuracy'])
                
                # Validate
                val_metrics = validate(val_loader)
                val_losses.append(val_metrics['loss'])
                val_accuracies.append(val_metrics['accuracy'])
                
                # Print detailed metrics
                print(f"\nEpoch {epoch}/{num_epochs} completed in {epoch_time:.1f} seconds:")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"    Feature Loss: {train_metrics['feature_loss']:.4f}")
                print(f"    Manifold Loss: {train_metrics['manifold_loss']:.4f}")
                print(f"    Topo Loss: {train_metrics['topo_loss']:.4f}")
                print(f"    Classifier Loss: {train_metrics['classifier_loss']:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"  Learning Rate: {feature_optimizer.param_groups[0]['lr']:.2e}")
                
                # Update learning rate schedulers
                feature_scheduler.step(val_metrics['loss'])
                manifold_scheduler.step(val_metrics['loss'])
                topo_scheduler.step(val_metrics['loss'])
                classifier_scheduler.step(val_metrics['loss'])
                
                # Save training curves
                plt.figure(figsize=(12, 5))
                
                # Plot loss
                plt.subplot(1, 2, 1)
                plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
                plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Val Loss')
                plt.title('Loss vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # Plot accuracy
                plt.subplot(1, 2, 2)
                plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='Train Acc')
                plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='Val Acc')
                plt.title('Accuracy vs. Epoch')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f'results/training_curves_epoch{epoch}.png'))
                plt.close()
                
                # Early stopping check
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    early_stopping_counter = 0
                    
                    # Save best model
                    print(f"  New best model with accuracy: {best_val_accuracy:.2f}%")
                    save_models(args.save_dir, epoch=f"best", metrics={
                        'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics['accuracy']
                    })
                else:
                    early_stopping_counter += 1
                    print(f"  No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                    
                    if early_stopping_counter >= early_stopping_patience:
                        print(f"  Early stopping triggered after {epoch} epochs")
                        break
                
                # Save training metrics
                torch.save({
                    'train_losses': train_losses,
                    'train_accuracies': train_accuracies,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'best_epoch': len(val_accuracies) - early_stopping_counter,
                    'best_accuracy': best_val_accuracy
                }, os.path.join(args.save_dir, 'models/medium/training_metrics.pt'))
                
                # Always save a checkpoint for the current epoch
                save_models(args.save_dir, epoch=epoch)
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to save an emergency checkpoint
                try:
                    save_models(args.save_dir, epoch=f"{epoch}_emergency")
                except Exception as ce:
                    print(f"Could not save emergency checkpoint: {ce}")
                
                # Free memory
                torch.cuda.empty_cache()
                
                # Continue training if possible
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory error detected. Attempting to continue...")
                    continue
                else:
                    raise  # Re-raise for other errors
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save final checkpoint
        save_models(args.save_dir, epoch="interrupted")
    
    except Exception as e:
        print(f"\nTraining stopped due to error: {e}")
        import traceback
        traceback.print_exc()
        # Save emergency checkpoint
        try:
            save_models(args.save_dir, epoch="error_final")
        except:
            print("Could not save final error checkpoint.")
    
    finally:
        # This will run regardless of how the training ends
        print("\nTraining complete or interrupted!")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Final save
        try:
            save_models(args.save_dir, epoch="final")
            print("Final models saved to models/medium/ directory")
            
            # Save a text summary of the training
            with open(os.path.join(args.save_dir, 'results/training_summary.txt'), 'w') as f:
                f.write(f"Medium Model Training Summary (Approx. {total_params:,} parameters)\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Best validation accuracy: {best_val_accuracy:.2f}%\n")
                
                if val_accuracies:
                    best_epoch_idx = val_accuracies.index(max(val_accuracies))
                    f.write(f"Best epoch: {best_epoch_idx + 1}\n\n")
                    
                    f.write("Final metrics:\n")
                    if train_losses:
                        f.write(f"  Train Loss: {train_losses[-1]:.4f}\n")
                    if val_losses:
                        f.write(f"  Validation Loss: {val_losses[-1]:.4f}\n")
                    if train_accuracies:
                        f.write(f"  Train Accuracy: {train_accuracies[-1]:.2f}%\n")
                    if val_accuracies:
                        f.write(f"  Validation Accuracy: {val_accuracies[-1]:.2f}%\n")
                else:
                    f.write("No validation metrics were recorded during training.\n")
                    
            print("Training summary saved to results/training_summary.txt")
            
        except Exception as e:
            print(f"Error in final saving: {e}")

# Main execution
if __name__ == "__main__":
    print("\n" + "="*80)
    print("PCM Medium Model Training on OCF HPC Cluster")
    print("="*80)
    
    print(f"\nEnvironment:")
    print(f"  Python: {torch.__version__}")
    print(f"  Torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"\nArguments:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation}")
    print(f"  Save frequency: {args.save_freq}")
    print(f"  Resume from: {args.resume}")
    
    # Start training
    train_model()