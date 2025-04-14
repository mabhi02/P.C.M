import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib
# Set non-interactive backend for matplotlib (no GUI needed on cluster)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
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
# Add distributed training arguments
parser.add_argument('--gpus', type=int, default=1, 
                    help='Number of GPUs to use for training')
parser.add_argument('--local_rank', type=int, default=-1, 
                    help='Local rank for distributed training')
parser.add_argument('--node_rank', type=int, default=0,
                    help='Node rank for distributed training')
parser.add_argument('--master_addr', default='localhost',
                    help='Master node address')
parser.add_argument('--master_port', default='12355',
                    help='Master node port')
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

def setup(rank, world_size):
    """Initialize distributed training environment"""
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Initialized process {rank} / {world_size}")

def cleanup():
    """Clean up distributed training environment"""
    dist.destroy_process_group()

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# IMPORTANT: Move CustomImageDataset to global scope so it can be pickled
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

# Move these functions to global scope too
def load_dataset_from_csv(csv_path, verbose=True, is_master=False):
    """Load image paths and labels from a CSV file"""
    # Only print verbose info from master process
    verbose = verbose and is_master
    
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
            
        # Convert Windows backslashes to Linux forward slashes
        file_name = file_name.replace('\\', '/')
        
        # Construct full file path - try different ways based on your file structure
        paths_to_try = []
        
        # Primary path: directly from CSV
        paths_to_try.append(os.path.join(base_path, file_name))
        
        # For ai_vs_human_gener, try the train directory
        if 'ai_vs_human_gener' in base_path:
            paths_to_try.append(os.path.join(base_path, 'train', os.path.basename(file_name)))
        
        # For images/ai_vs_human_generated_dataset, try the train directory
        if 'ai_vs_human_generated_dataset' in base_path:
            paths_to_try.append(os.path.join(base_path, 'train', os.path.basename(file_name)))
            
        # Just use basename directly in train directories
        paths_to_try.append(os.path.join(base_path, 'train', os.path.basename(file_name)))
        
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
        
        # Try all possible paths
        found_file = False
        for path in paths_to_try:
            if os.path.exists(path):
                all_files.append(path)
                labels.append(label)
                found_file = True
                break
                
        if not found_file and verbose and index < 10:
            print(f"Warning: File not found. Tried paths: {paths_to_try}")
    
    if verbose:
        print(f"Successfully loaded {len(all_files)} images from {csv_path}")
        print(f"Human images (label 0): {labels.count(0)}")
        print(f"AI-generated images (label 1): {labels.count(1)}")
    
    return all_files, labels

def combine_datasets(data_dir, is_master=False):
    """Load and combine datasets from both CSV files in the OCF environment"""
    # Updated paths based on your file structure
    dataset_paths = []
    
    # Try the ai_vs_human_generated_dataset path
    ai_human_dataset1 = os.path.join(data_dir, 'images', 'ai_vs_human_generated_dataset', 'train.csv')
    if os.path.exists(ai_human_dataset1):
        dataset_paths.append(ai_human_dataset1)
    
    # Try the ai_vs_human_gener path
    ai_human_dataset2 = os.path.join(data_dir, 'ai_vs_human_gener', 'train.csv')
    if os.path.exists(ai_human_dataset2):
        dataset_paths.append(ai_human_dataset2)
    
    # Load all available datasets
    all_files = []
    all_labels = []
    
    for i, dataset_path in enumerate(dataset_paths):
        try:
            files, labels = load_dataset_from_csv(dataset_path, is_master=is_master)
            if is_master:
                print(f"Dataset {i+1}: {len(files)} images loaded")
            all_files.extend(files)
            all_labels.extend(labels)
        except Exception as e:
            if is_master:
                print(f"Error loading dataset {i+1}: {e}")
    
    if is_master:
        print(f"\nCombined dataset statistics:")
        print(f"Total images: {len(all_files)}")
        print(f"Human images (label 0): {all_labels.count(0)}")
        print(f"AI-generated images (label 1): {all_labels.count(1)}")
    
    return all_files, all_labels

def create_dataset(data_dir, batch_size, rank, world_size, is_master=False):
    """Create dataset and split it for distributed training"""
    # Combine datasets
    all_files, all_labels = combine_datasets(data_dir, is_master=is_master)
    
    if len(all_files) == 0:
        if is_master:
            print("No images were loaded. Please check file paths.")
        return None, None, None, None
    
    # Create the dataset with upsampling to 448x448
    dataset = CustomImageDataset(all_files, all_labels, target_size=(448, 448))
    
    # Split the dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use torch.Generator with the same seed on all processes for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    if is_master:
        print(f"\nDataset split:")
        print(f"Training set: {train_size} images")
        print(f"Validation set: {val_size} images")
    
    # Create distributed samplers for training and validation
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Adjust batch size per GPU
    batch_size_per_gpu = batch_size
    
    # Create data loaders with distributed samplers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size_per_gpu, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Check a batch of data (only on master process)
    if is_master:
        try:
            images, labels = next(iter(train_loader))
            print(f"Batch shape: {images.shape}")
            print(f"Sample labels: {labels}")
        except Exception as e:
            import traceback
            print(f"Error loading first batch: {e}")
            traceback.print_exc()
    
    return train_loader, val_loader, train_sampler, val_sampler

# Function to run on each GPU
def train_worker(rank, world_size):
    # Set the device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Set seed for deterministic behavior
    set_seed(42 + rank)  # Different seed per process
    
    # Setup the process group
    setup(rank, world_size)
    
    # Determine if this process is the master
    is_master = rank == 0
    
    # Only print from master process
    if is_master:
        print(f"Training with {world_size} GPUs")
        print(f"Using device: {device}")
    
    # Create directories (only on master process)
    if is_master:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'models/medium'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'models/medium/checkpoints'), exist_ok=True)
    
    # Initialize model components
    if is_master:
        print("Initializing model components...")
    
    feature_network = EnhancedFeatureExtractionNetwork(feature_dim=256).to(device)
    tensor_computer = EnhancedPhaseCorrelationTensorComputation(feature_dim=256, output_dim=512).to(device)
    dim_adapter = DimensionAdapter(input_dim=512, output_dim=256).to(device)
    manifold_module = ManifoldLearningModule(
        input_dim=256,
        hidden_dim=512,
        latent_dim=64,
        gnn_hidden_dim=128
    ).to(device)
    point_cloud_generator = PointCloudGenerator(num_points=64).to(device)
    topo_module = TinyTopologicalFeatureExtraction(
        input_dim=64,
        hidden_dim=64,
        output_dim=32,
        max_edge_length=2.0,
        num_filtrations=16,
        max_dimension=1
    ).to(device)
    classifier = ClassificationNetwork(
        manifold_dim=64,
        topo_dim=32,
        feature_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=6,
        dropout=0.15
    ).to(device)
    
    # Wrap models with DistributedDataParallel
    feature_network = DDP(feature_network, device_ids=[rank], find_unused_parameters=True)
    tensor_computer = DDP(tensor_computer, device_ids=[rank], find_unused_parameters=True)
    dim_adapter = DDP(dim_adapter, device_ids=[rank], find_unused_parameters=True)
    manifold_module = DDP(manifold_module, device_ids=[rank], find_unused_parameters=True)
    point_cloud_generator = DDP(point_cloud_generator, device_ids=[rank], find_unused_parameters=True)
    topo_module = DDP(topo_module, device_ids=[rank], find_unused_parameters=True)
    classifier = DDP(classifier, device_ids=[rank], find_unused_parameters=True)
    
    # Print model sizes (only on master process)
    if is_master:
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
    
    # Create dataset and loaders
    train_loader, val_loader, train_sampler, val_sampler = create_dataset(
        args.data_dir, args.batch_size, rank, world_size, is_master=is_master
    )
    
    if train_loader is None or val_loader is None:
        if is_master:
            print("Error: Could not create data loaders. Exiting.")
        cleanup()
        return
    
    # Define loss functions
    classification_criterion = nn.CrossEntropyLoss()
    
    # Define optimizers
    feature_optimizer = optim.Adam(list(feature_network.parameters()) + 
                                  list(tensor_computer.parameters()) + 
                                  list(dim_adapter.parameters()), lr=8e-5)
    manifold_optimizer = optim.Adam(manifold_module.parameters(), lr=8e-5)
    topo_optimizer = optim.Adam(list(point_cloud_generator.parameters()) +
                               list(topo_module.parameters()), lr=8e-5)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=8e-5)
    
    # Learning rate schedulers
    feature_scheduler = optim.lr_scheduler.ReduceLROnPlateau(feature_optimizer, mode='min', 
                                                            factor=0.5, patience=3, min_lr=1e-6)
    manifold_scheduler = optim.lr_scheduler.ReduceLROnPlateau(manifold_optimizer, mode='min', 
                                                             factor=0.5, patience=3, min_lr=1e-6)
    topo_scheduler = optim.lr_scheduler.ReduceLROnPlateau(topo_optimizer, mode='min', 
                                                         factor=0.5, patience=3, min_lr=1e-6)
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='min', 
                                                               factor=0.5, patience=3, min_lr=1e-6)
    
    def vae_loss(recon_x, x, mu, logvar):
        """VAE loss function with robust handling for dimension mismatches"""
        # Check if reconstruction was successful
        if recon_x is None:
            # If decode failed, just use KL divergence
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        # Handle the specific case where recon_x is [batch_size, 768] and x is [batch_size, 256, 6, 6]
        if len(recon_x.shape) == 2 and len(x.shape) == 4:
            if is_master:
                print(f"Handling dimension mismatch: recon_x {recon_x.size()} vs x {x.size()}")
            
            # Project recon_x to match flattened size of x
            batch_size = x.size(0)
            flattened_size = x.view(batch_size, -1).size(1)
            
            # Create a projection layer on-the-fly if needed
            if not hasattr(vae_loss, 'projection_layer') or vae_loss.projection_layer.in_features != recon_x.size(1) or vae_loss.projection_layer.out_features != flattened_size:
                vae_loss.projection_layer = nn.Linear(recon_x.size(1), flattened_size).to(recon_x.device)
                # Initialize weights
                nn.init.xavier_uniform_(vae_loss.projection_layer.weight)
                nn.init.zeros_(vae_loss.projection_layer.bias)
                if is_master:
                    print(f"Created new projection layer: {recon_x.size(1)} -> {flattened_size}")
            
            # Project recon_x
            projected_recon_x = vae_loss.projection_layer(recon_x)
            
            # Compare with flattened x
            flattened_x = x.view(batch_size, -1)
            recon_loss = F.mse_loss(projected_recon_x, flattened_x, reduction='sum') / batch_size
            
            # KL divergence 
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
            
            # Use a smaller weight for the KL term during early training
            kl_weight = min(0.8, 0.1 + 0.01 * getattr(vae_loss, 'call_count', 0))
            
            # Increment call count
            vae_loss.call_count = getattr(vae_loss, 'call_count', 0) + 1
            
            return recon_loss + kl_weight * kld_loss
        
        # For other shape mismatches
        elif recon_x.size() != x.size():
            if is_master:
                print(f"Other dimension mismatch in VAE loss: recon_x {recon_x.size()} vs x {x.size()}")
            try:
                # Try to match dimensions by reshaping
                batch_size = x.size(0)
                
                # Flatten both tensors
                recon_x_flat = recon_x.view(batch_size, -1)
                x_flat = x.view(batch_size, -1)
                
                # If dimensions still don't match, use a simpler approach
                if recon_x_flat.size(1) != x_flat.size(1):
                    if is_master:
                        print(f"Cannot match dimensions even when flattened: {recon_x_flat.size()} vs {x_flat.size()}")
                    # Just use KL divergence as the loss
                    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                
                # If dimensions match after flattening
                recon_loss = F.mse_loss(recon_x_flat, x_flat, reduction='sum') / batch_size
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                return recon_loss + 0.8 * kld_loss
                
            except Exception as e:
                if is_master:
                    print(f"Failed to handle VAE loss: {e}")
                # Fall back to just KL divergence
                return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        # If shapes match exactly, use standard VAE loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + 0.8 * kld_loss
    
    # Function to save models (only on master process)
    def save_models(save_dir, epoch=None, metrics=None):
        if not is_master:
            return
            
        # Create directory
        model_dir = os.path.join(save_dir, 'models/medium')
        os.makedirs(model_dir, exist_ok=True)
        
        # Add epoch information to filenames
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        
        # Save model states (access .module for DDP models)
        torch.save(feature_network.module.state_dict(), os.path.join(model_dir, f'feature_network{suffix}.pth'))
        torch.save(tensor_computer.module.state_dict(), os.path.join(model_dir, f'tensor_computer{suffix}.pth'))
        torch.save(dim_adapter.module.state_dict(), os.path.join(model_dir, f'dim_adapter{suffix}.pth'))
        torch.save(manifold_module.module.state_dict(), os.path.join(model_dir, f'manifold_module{suffix}.pth'))
        torch.save(point_cloud_generator.module.state_dict(), os.path.join(model_dir, f'point_cloud_generator{suffix}.pth'))
        torch.save(topo_module.module.state_dict(), os.path.join(model_dir, f'topo_module{suffix}.pth'))
        torch.save(classifier.module.state_dict(), os.path.join(model_dir, f'classifier{suffix}.pth'))
        
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
    
    # Training function with gradient accumulation for distributed training
    def train_epoch_with_accumulation(train_loader, epoch, train_sampler=None, accumulation_steps=8):
        # Set train sampler epoch for distributed training
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
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
        
        # Only use tqdm progress bar on master process
        if is_master:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = train_loader
        
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
                recon_tensor = manifold_module.module.decode(z)
                
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
                
                # Save checkpoint at specified frequency (only on master process)
                if is_master and args.save_freq > 0 and ((batch_idx + 1) % args.save_freq == 0):
                    checkpoint_path = os.path.join(args.save_dir, f'models/medium/checkpoints/epoch{epoch}_batch{batch_idx}')
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    # Calculate current metrics
                    current_accuracy = 100. * correct / total if total > 0 else 0
                    
                    # Save checkpoint
                    torch.save(feature_network.module.state_dict(), f'{checkpoint_path}/feature_network.pth')
                    torch.save(tensor_computer.module.state_dict(), f'{checkpoint_path}/tensor_computer.pth')
                    torch.save(dim_adapter.module.state_dict(), f'{checkpoint_path}/dim_adapter.pth')
                    torch.save(manifold_module.module.state_dict(), f'{checkpoint_path}/manifold_module.pth')
                    torch.save(point_cloud_generator.module.state_dict(), f'{checkpoint_path}/point_cloud_generator.pth')
                    torch.save(topo_module.module.state_dict(), f'{checkpoint_path}/topo_module.pth')
                    torch.save(classifier.module.state_dict(), f'{checkpoint_path}/classifier.pth')
                    
                    print(f"\nCheckpoint saved at batch {batch_idx+1} with accuracy: {current_accuracy:.2f}%")
                
                # Update progress bar but not too frequently to reduce overhead
                if is_master and batch_idx % 5 == 0:
                    progress_bar.set_postfix({
                        'loss': total_loss / (batch_idx + 1),
                        'acc': 100. * correct / total if total > 0 else 0
                    })
            
            except Exception as e:
                if is_master:
                    print(f"Error in training batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full traceback for better debugging
                
                # Free memory on error
                torch.cuda.empty_cache()
                
                # Skip this batch and continue
                continue
        
        # Gather metrics from all processes
        # Convert to tensors for all-reduce
        loss_tensor = torch.tensor([total_loss, total_feature_loss, total_manifold_loss, 
                                    total_topo_loss, total_classifier_loss], device=device)
        acc_tensor = torch.tensor([correct, total], device=device)
        
        # All-reduce to get totals from all processes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        
        # Unpack the results
        total_loss, total_feature_loss, total_manifold_loss, total_topo_loss, total_classifier_loss = loss_tensor.tolist()
        correct, total = acc_tensor.tolist()
        
        # Get the total number of batches across all processes
        num_batches = torch.tensor([len(train_loader)], device=device)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        num_batches = num_batches.item() / world_size  # Average by number of processes
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        avg_feature_loss = total_feature_loss / num_batches
        avg_manifold_loss = total_manifold_loss / num_batches
        avg_topo_loss = total_topo_loss / num_batches
        avg_classifier_loss = total_classifier_loss / num_batches
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {
            'loss': avg_loss,
            'feature_loss': avg_feature_loss,
            'manifold_loss': avg_manifold_loss,
            'topo_loss': avg_topo_loss,
            'classifier_loss': avg_classifier_loss,
            'accuracy': accuracy
        }
    
    # Validation function for distributed training
    def validate(val_loader, val_sampler=None):
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
            # Only use tqdm progress bar on master process
            if is_master:
                iter_val_loader = tqdm(val_loader, desc="Validating")
            else:
                iter_val_loader = val_loader
                
            for images, labels in iter_val_loader:
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
                    recon_tensor = manifold_module.module.decode(z)
                    
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
                    
                    # Store for metrics (only on master process to save memory)
                    if is_master:
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
                    if is_master:
                        print(f"Error in validation: {e}")
                        import traceback
                        traceback.print_exc()
                    torch.cuda.empty_cache()
                    continue
        
        # Gather metrics from all processes
        # Convert to tensors for all-reduce
        loss_tensor = torch.tensor([val_loss, val_feature_loss, val_manifold_loss, val_topo_loss], device=device)
        acc_tensor = torch.tensor([correct, total], device=device)
        
        # All-reduce to get totals from all processes
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_tensor, op=dist.ReduceOp.SUM)
        
        # Unpack the results
        val_loss, val_feature_loss, val_manifold_loss, val_topo_loss = loss_tensor.tolist()
        correct, total = acc_tensor.tolist()
        
        # Get the total number of batches across all processes
        num_batches = torch.tensor([len(val_loader)], device=device)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        num_batches = num_batches.item() / world_size  # Average by number of processes
        
        # Calculate metrics
        avg_loss = val_loss / num_batches
        avg_feature_loss = val_feature_loss / num_batches
        avg_manifold_loss = val_manifold_loss / num_batches
        avg_topo_loss = val_topo_loss / num_batches
        accuracy = 100. * correct / total if total > 0 else 0
        
        # Concatenate predictions and labels (only on master process)
        if is_master:
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
            'predictions': all_preds if is_master else None,
            'labels': all_labels if is_master else None,
            'probabilities': all_probs if is_master else None,
            'uncertainty': all_uncertainty if is_master else None
        }
    
    def resume_from_checkpoint(checkpoint_path):
        """Resume training from a checkpoint"""
        if is_master:
            print(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Map location to current device
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        
        # Load state dictionaries for model components
        feature_network.module.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'feature_network.pth'), map_location=map_location))
        tensor_computer.module.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'tensor_computer.pth'), map_location=map_location))
        dim_adapter.module.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'dim_adapter.pth'), map_location=map_location))
        manifold_module.module.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'manifold_module.pth'), map_location=map_location))
        point_cloud_generator.module.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'point_cloud_generator.pth'), map_location=map_location))
        topo_module.module.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'topo_module.pth'), map_location=map_location))
        classifier.module.load_state_dict(
            torch.load(os.path.join(checkpoint_path, 'classifier.pth'), map_location=map_location))
        
        # Only load optimizer states on master process
        if is_master and os.path.exists(os.path.join(checkpoint_path, 'feature_optimizer.pth')):
            feature_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_optimizer.pth')))
            manifold_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_optimizer.pth')))
            topo_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_optimizer.pth')))
            classifier_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier_optimizer.pth')))
        
        # Only load scheduler states on master process
        if is_master and os.path.exists(os.path.join(checkpoint_path, 'feature_scheduler.pth')):
            feature_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_scheduler.pth')))
            manifold_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_scheduler.pth')))
            topo_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_scheduler.pth')))
            classifier_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier_scheduler.pth')))
        
        # Make sure all processes are synced after loading checkpoint
        dist.barrier()
        
        if is_master:
            print("Checkpoint loaded successfully")
    
    # Main training function
    def train_model():
        # Training parameters
        num_epochs = args.epochs
        gradient_accumulation_steps = args.gradient_accumulation
        best_val_accuracy = 0
        early_stopping_patience = 5
        early_stopping_counter = 0
        
        # Initialize metrics tracking (only on master process)
        if is_master:
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
        
        # Resume from checkpoint if specified
        if args.resume is not None:
            resume_from_checkpoint(args.resume)
            
            # Try to load previous metrics if available (only on master process)
            if is_master:
                metrics_path = os.path.join(os.path.dirname(args.resume), 'training_metrics.pt')
                if os.path.exists(metrics_path):
                    metrics = torch.load(metrics_path)
                    train_losses = metrics.get('train_losses', [])
                    train_accuracies = metrics.get('train_accuracies', [])
                    val_losses = metrics.get('val_losses', [])
                    val_accuracies = metrics.get('val_accuracies', [])
                    best_val_accuracy = metrics.get('best_accuracy', 0)
                    print(f"Loaded previous metrics. Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Print total parameters count (only on master process)
        if is_master:
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
                
            total_params = (count_parameters(feature_network) + 
                          count_parameters(tensor_computer) + 
                          count_parameters(dim_adapter) + 
                          count_parameters(manifold_module) + 
                          count_parameters(point_cloud_generator) + 
                          count_parameters(topo_module) + 
                          count_parameters(classifier))
                
            print(f"\nStarting training for {num_epochs} epochs with {total_params:,} parameters")
            print(f"Distributed training with {world_size} processes")
        
        # GPU Events for timing (only on master process)
        if is_master:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
        
        try:
            for epoch in range(1, num_epochs + 1):
                try:
                    if is_master:
                        print(f"\n{'='*80}\nEpoch {epoch}/{num_epochs}\n{'='*80}")
                        
                        # Start timing
                        start_time.record()
                    
                    # Make sure all processes are at the same point before starting epoch
                    dist.barrier()
                    
                    # Train for one epoch
                    train_metrics = train_epoch_with_accumulation(
                        train_loader, 
                        epoch,
                        train_sampler=train_sampler,
                        accumulation_steps=gradient_accumulation_steps
                    )
                    
                    # Make sure all processes are at the same point after training
                    dist.barrier()
                    
                    # End timing (only on master process)
                    if is_master:
                        end_time.record()
                        torch.cuda.synchronize()
                        epoch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                        
                        # Store training metrics
                        train_losses.append(train_metrics['loss'])
                        train_accuracies.append(train_metrics['accuracy'])
                    
                    # Validate
                    val_metrics = validate(val_loader, val_sampler=val_sampler)
                    
                    # Wait for all processes to finish validation
                    dist.barrier()
                    
                    # Only the master process needs to track metrics and save models
                    if is_master:
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
                        
                        # Update learning rate schedulers (only on master process)
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
                                should_stop = True
                            else:
                                should_stop = False
                        
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
                    else:
                        # Default to not stopping for non-master processes
                        should_stop = False
                    
                    # Broadcast early stopping decision to all processes
                    stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
                    dist.broadcast(stop_tensor, 0)  # Broadcast from rank 0
                    
                    if stop_tensor.item() == 1:
                        if is_master:
                            print("Early stopping signal sent to all processes")
                        break
                
                except Exception as e:
                    if is_master:
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
                        if is_master:
                            print("CUDA out of memory error detected. Attempting to continue...")
                        continue
                    else:
                        raise  # Re-raise for other errors
        
        except KeyboardInterrupt:
            if is_master:
                print("\nTraining interrupted by user.")
                # Save final checkpoint
                save_models(args.save_dir, epoch="interrupted")
        
        except Exception as e:
            if is_master:
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
            if is_master:
                print("\nTraining complete or interrupted!")
                print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
                
                # Final save
                try:
                    save_models(args.save_dir, epoch="final")
                    print("Final models saved to models/medium/ directory")
                    
                    # Save a text summary of the training
                    with open(os.path.join(args.save_dir, 'results/training_summary.txt'), 'w') as f:
                        f.write(f"Medium Model Training Summary (Distributed training with {world_size} GPUs)\n")
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
    
    # Run the training
    train_model()
    
    # Clean up distributed training
    cleanup()
    
# Main function to spawn multiple processes for distributed training
def main():
    # Print basic system info
    print("\n" + "="*80)
    print("PCM Medium Model Training on HPC Cluster (Distributed Training)")
    print("="*80)
    
    print(f"\nEnvironment:")
    print(f"  Python: {torch.__version__}")
    print(f"  Torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA version: {torch.version.cuda}")
    
    print(f"\nArguments:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation}")
    print(f"  Save frequency: {args.save_freq}")
    print(f"  Resume from: {args.resume}")
    print(f"  Number of GPUs: {args.gpus}")
    
    # Check if enough GPUs are available
    if torch.cuda.device_count() < args.gpus:
        print(f"Warning: Requested {args.gpus} GPUs but only {torch.cuda.device_count()} are available.")
        args.gpus = torch.cuda.device_count()
        print(f"Using {args.gpus} GPUs instead.")
    
    if args.gpus > 1:
        # Use multiprocessing to spawn processes
        mp.spawn(
            train_worker,
            args=(args.gpus,),
            nprocs=args.gpus,
            join=True
        )
    else:
        # Single GPU training
        train_worker(0, 1)

if __name__ == "__main__":
    # Disable NCCL's use of IB for better compatibility
    os.environ["NCCL_IB_DISABLE"] = "1"
    # For multiple GPUs use shared memory
    os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    
    # Fix for some clusters that may have issues with NCCL
    os.environ["NCCL_P2P_DISABLE"] = "1"
    
    # Start the main function
    main()