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
import sys

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
# Simplified distributed training arguments
parser.add_argument('--gpus', type=int, default=1, 
                    help='Number of GPUs to use for training')
parser.add_argument('--port', type=str, default='29500',
                    help='Port for distributed training')

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
            master_print(f"Error loading image {image_path}: {e}")
            # Return a placeholder tensor and the label
            return torch.zeros(3, *self.target_size), label

# Define the main training function
def main_worker(gpu, ngpus_per_node, args):
    """
    Main worker function that runs on each GPU
    """
    args.gpu = gpu
    
    # Import the model definitions here to avoid issues with multiprocessing
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
    
    if args.gpus > 1:
        # Initialize process group with explicit device ID
        rank = gpu
        torch.cuda.set_device(gpu)  # Set device first
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://127.0.0.1:{args.port}',
            world_size=args.gpus,
            rank=rank
        )
        is_distributed = True
        local_rank = gpu
    else:
        is_distributed = False
        rank = 0
        local_rank = 0
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check if GPU is available
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # Only print once in distributed mode
    def master_print(*args_print, **kwargs):
        if not is_distributed or rank == 0:
            print(*args_print, **kwargs)
    
    master_print(f"Running on: {device}")
    master_print(f"Distributed training: {is_distributed}, Rank: {rank}, GPU: {gpu}")
    
    # Create directories (only on master process)
    if not is_distributed or rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'models/medium'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'models/medium/checkpoints'), exist_ok=True)
    
    # Synchronize all processes
    if is_distributed:
        dist.barrier()
    
    # Initialize model components 
    master_print("Initializing model components...")
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
        input_dim=64,           # Match the output dimension from ManifoldLearningModule
        hidden_dim=64,          # Hidden dimension
        output_dim=32,          # Output dimension
        max_edge_length=2.0,    # Maximum edge length for filtration
        num_filtrations=16,     # Number of filtration values
        max_dimension=1         # Increased to dimension 1 for medium model
    ).to(device)
    classifier = ClassificationNetwork(
        manifold_dim=64,        # Match the output dimension from ManifoldLearningModule
        topo_dim=32,            # Match the output dimension from TinyTopologicalFeatureExtraction
        feature_dim=128,        # Feature dimension
        hidden_dim=256,         # Hidden dimension
        num_layers=4,           # Number of transformer layers
        num_heads=6,            # Number of attention heads
        dropout=0.15            # Dropout rate
    ).to(device)
    
    # Wrap models with DistributedDataParallel if using distributed training
    if is_distributed:
        feature_network = DDP(feature_network, device_ids=[gpu])
        tensor_computer = DDP(tensor_computer, device_ids=[gpu])
        dim_adapter = DDP(dim_adapter, device_ids=[gpu])
        manifold_module = DDP(manifold_module, device_ids=[gpu])
        point_cloud_generator = DDP(point_cloud_generator, device_ids=[gpu])
        topo_module = DDP(topo_module, device_ids=[gpu])
        classifier = DDP(classifier, device_ids=[gpu])
    
    # Print model sizes using count_parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if not is_distributed or rank == 0:
        master_print(f"Enhanced Feature Extraction Network: {count_parameters(feature_network):,} parameters")
        master_print(f"Phase Correlation Tensor Computation: {count_parameters(tensor_computer):,} parameters")
        master_print(f"Dimension Adapter: {count_parameters(dim_adapter):,} parameters")
        master_print(f"Manifold Learning Module: {count_parameters(manifold_module):,} parameters")
        master_print(f"Point Cloud Generator: {count_parameters(point_cloud_generator):,} parameters")
        master_print(f"Tiny Topological Feature Extraction: {count_parameters(topo_module):,} parameters")
        master_print(f"Classification Network: {count_parameters(classifier):,} parameters")
        total_params = (count_parameters(feature_network) + count_parameters(tensor_computer) + 
                       count_parameters(dim_adapter) + count_parameters(manifold_module) + 
                       count_parameters(point_cloud_generator) + count_parameters(topo_module) + 
                       count_parameters(classifier))
        master_print(f"Total: {total_params:,} parameters")
    
    
    def load_dataset_from_csv(csv_path, verbose=True):
        """Load image paths and labels from a CSV file"""
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        if verbose and (not is_distributed or rank == 0):
            master_print(f"\nLoading from: {csv_path}")
            master_print(f"CSV columns: {df.columns.tolist()}")
            master_print(f"First few rows of the CSV:\n{df.head()}")
        
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
                    
            if not found_file and verbose and index < 10 and (not is_distributed or rank == 0):
                master_print(f"Warning: File not found. Tried paths: {paths_to_try}")
        
        if verbose and (not is_distributed or rank == 0):
            master_print(f"Successfully loaded {len(all_files)} images from {csv_path}")
            master_print(f"Human images (label 0): {labels.count(0)}")
            master_print(f"AI-generated images (label 1): {labels.count(1)}")
        
        return all_files, labels
    
    def combine_datasets(data_dir):
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
                files, labels = load_dataset_from_csv(dataset_path)
                if not is_distributed or rank == 0:
                    master_print(f"Dataset {i+1}: {len(files)} images loaded")
                all_files.extend(files)
                all_labels.extend(labels)
            except Exception as e:
                if not is_distributed or rank == 0:
                    master_print(f"Error loading dataset {i+1}: {e}")
        
        if not is_distributed or rank == 0:
            master_print(f"\nCombined dataset statistics:")
            master_print(f"Total images: {len(all_files)}")
            master_print(f"Human images (label 0): {all_labels.count(0)}")
            master_print(f"AI-generated images (label 1): {all_labels.count(1)}")
        
        return all_files, all_labels
    
    def create_dataset(data_dir, batch_size):
        """Create dataset and split it for training with distributed support"""
        # Combine datasets
        all_files, all_labels = combine_datasets(data_dir)
        
        if len(all_files) == 0:
            master_print("No images were loaded. Please check file paths.")
            return None, None, None, None
        
        # Create the dataset with upsampling to 448x448
        dataset = CustomImageDataset(all_files, all_labels, target_size=(448, 448))
        
        # Split the dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Use a fixed generator with a seed for reproducibility
        generator = torch.Generator().manual_seed(42)
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=generator
        )
        
        if not is_distributed or rank == 0:
            master_print(f"\nDataset split:")
            master_print(f"Training set: {train_size} images")
            master_print(f"Validation set: {val_size} images")
        
        # Create samplers for distributed training
        train_sampler = DistributedSampler(train_dataset) if is_distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
        
        # Create data loaders with adjusted num_workers for HPC
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=(train_sampler is None),  # Don't shuffle if using distributed sampler
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,  # Pin memory for faster data transfer to GPU
            persistent_workers=True,  # Keep workers alive between batches
            prefetch_factor=2  # Prefetch batches to reduce I/O bottlenecks
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Check a batch of data on the master process
        if not is_distributed or rank == 0:
            try:
                images, labels = next(iter(train_loader))
                master_print(f"Batch shape: {images.shape}")
                master_print(f"Sample labels: {labels}")
            except Exception as e:
                import traceback
                master_print(f"Error loading first batch: {e}")
                traceback.print_exc()
        
        return train_loader, val_loader, train_sampler, val_sampler
    
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
            if not is_distributed or rank == 0:
                master_print(f"Handling dimension mismatch: recon_x {recon_x.size()} vs x {x.size()}")
            
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
                if not is_distributed or rank == 0:
                    master_print(f"Created new projection layer: {recon_x.size(1)} -> {flattened_size}")
            
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
            if not is_distributed or rank == 0:
                master_print(f"Other dimension mismatch in VAE loss: recon_x {recon_x.size()} vs x {x.size()}")
            try:
                # Try to match dimensions by reshaping if possible
                batch_size = x.size(0)
                
                # If both can be flattened, use MSE on flattened versions
                recon_x_flat = recon_x.view(batch_size, -1)
                x_flat = x.view(batch_size, -1)
                
                # If dimensions still don't match, use a simpler approach
                if recon_x_flat.size(1) != x_flat.size(1):
                    if not is_distributed or rank == 0:
                        master_print(f"Cannot match dimensions even when flattened: {recon_x_flat.size()} vs {x_flat.size()}")
                    # Just use KL divergence as the loss
                    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                
                # If dimensions match after flattening
                recon_loss = F.mse_loss(recon_x_flat, x_flat, reduction='sum') / batch_size
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
                return recon_loss + 0.8 * kld_loss
                
            except Exception as e:
                if not is_distributed or rank == 0:
                    master_print(f"Failed to handle VAE loss: {e}")
                # Fall back to just KL divergence if all else fails
                return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        # If shapes match exactly, use standard VAE loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + 0.8 * kld_loss
    
    # Function to save models with explicit path (only on master process)
    def save_models(save_dir, epoch=None, metrics=None):
        if is_distributed and rank != 0:
            return
        
        # Create directory if it doesn't exist
        model_dir = os.path.join(save_dir, 'models/medium')
        os.makedirs(model_dir, exist_ok=True)
        
        # Add epoch information to filenames if provided
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        
        # Save model states - unwrap DDP models before saving
        def get_model_to_save(model):
            if isinstance(model, DDP):
                return model.module
            return model
        
        # Save model states
        torch.save(get_model_to_save(feature_network).state_dict(), os.path.join(model_dir, f'feature_network{suffix}.pth'))
        torch.save(get_model_to_save(tensor_computer).state_dict(), os.path.join(model_dir, f'tensor_computer{suffix}.pth'))
        torch.save(get_model_to_save(dim_adapter).state_dict(), os.path.join(model_dir, f'dim_adapter{suffix}.pth'))
        torch.save(get_model_to_save(manifold_module).state_dict(), os.path.join(model_dir, f'manifold_module{suffix}.pth'))
        torch.save(get_model_to_save(point_cloud_generator).state_dict(), os.path.join(model_dir, f'point_cloud_generator{suffix}.pth'))
        torch.save(get_model_to_save(topo_module).state_dict(), os.path.join(model_dir, f'topo_module{suffix}.pth'))
        torch.save(get_model_to_save(classifier).state_dict(), os.path.join(model_dir, f'classifier{suffix}.pth'))
        
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
        
        master_print(f"Model saved to {model_dir}{' for epoch ' + str(epoch) if epoch is not None else ''}")
    
    # Helper function to aggregate metrics across distributed processes
    def reduce_tensor(tensor):
        if not is_distributed:
            return tensor
        
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= args.gpus
        return rt
    
    # Training function with gradient accumulation for medium model
    def train_epoch_with_accumulation(train_loader, epoch, train_sampler=None, accumulation_steps=8):
        # Set models to training mode
        feature_network.train()
        tensor_computer.train()
        dim_adapter.train()
        manifold_module.train()
        point_cloud_generator.train()
        topo_module.train()
        classifier.train()
        
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Initialize metrics
        total_loss = 0
        total_feature_loss = 0
        total_manifold_loss = 0
        total_topo_loss = 0
        total_classifier_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        # Only show progress bar on master process
        if not is_distributed or rank == 0:
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
                
                # Save checkpoint at specified frequency (only on master process)
                if (not is_distributed or rank == 0) and args.save_freq > 0 and ((batch_idx + 1) % args.save_freq == 0):
                    checkpoint_path = os.path.join(args.save_dir, f'models/medium/checkpoints/epoch{epoch}_batch{batch_idx}')
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    # Calculate current metrics
                    current_accuracy = 100. * correct / total if total > 0 else 0
                    
                    # Save component checkpoints
                    save_models(checkpoint_path)
                    
                    master_print(f"\nCheckpoint saved at batch {batch_idx+1} with accuracy: {current_accuracy:.2f}%")
                
                # Update progress bar but not too frequently to reduce overhead (only on master process)
                if (not is_distributed or rank == 0) and batch_idx % 5 == 0:
                    if isinstance(progress_bar, tqdm):
                        progress_bar.set_postfix({
                            'loss': total_loss / (batch_idx + 1),
                            'acc': 100. * correct / total if total > 0 else 0
                        })
            
            except Exception as e:
                master_print(f"Error in training batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()  # Print full traceback for better debugging
                
                # Free memory on error
                torch.cuda.empty_cache()
                
                # Skip this batch and continue
                continue
        
        # Synchronize metrics across processes if distributed
        if is_distributed:
            # Convert to tensors for reduction
            loss_tensor = torch.tensor(total_loss).to(device)
            feature_loss_tensor = torch.tensor(total_feature_loss).to(device)
            manifold_loss_tensor = torch.tensor(total_manifold_loss).to(device)
            topo_loss_tensor = torch.tensor(total_topo_loss).to(device)
            classifier_loss_tensor = torch.tensor(total_classifier_loss).to(device)
            correct_tensor = torch.tensor(correct).to(device)
            total_tensor = torch.tensor(total).to(device)
            
            # All-reduce to sum across processes
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(manifold_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(topo_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(classifier_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            
            # Update variables with reduced values
            total_loss = loss_tensor.item()
            total_feature_loss = feature_loss_tensor.item()
            total_manifold_loss = manifold_loss_tensor.item()
            total_topo_loss = topo_loss_tensor.item()
            total_classifier_loss = classifier_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
        
        # Calculate epoch metrics
        num_batches = len(train_loader)
        if is_distributed:
            # Create a tensor to hold the number of batches
            num_batches_tensor = torch.tensor(num_batches).to(device)
            # All-reduce to get the sum across processes
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            # Get the average (each process has a different portion of the data)
            num_batches = num_batches_tensor.item() / args.gpus
        
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
    
    # Function to resume training from checkpoint
    def resume_from_checkpoint(checkpoint_path):
        """Resume training from a checkpoint"""
        master_print(f"Resuming from checkpoint: {checkpoint_path}")
        
        # Ensure all processes are synchronized
        if is_distributed:
            dist.barrier()
        
        # For distributed training, we need to load the model states to the correct device
        map_location = {'cuda:0': f'cuda:{gpu}'} if is_distributed else device
        
        # Helper function to get the module for loading state dict
        def get_module_to_load(model):
            if isinstance(model, DDP):
                return model.module
            return model
        
        # Load model weights
        feature_net_to_load = get_module_to_load(feature_network)
        tensor_comp_to_load = get_module_to_load(tensor_computer)
        dim_adapt_to_load = get_module_to_load(dim_adapter)
        manifold_to_load = get_module_to_load(manifold_module)
        point_cloud_to_load = get_module_to_load(point_cloud_generator)
        topo_to_load = get_module_to_load(topo_module)
        classifier_to_load = get_module_to_load(classifier)
        
        feature_net_to_load.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_network.pth'), map_location=map_location))
        tensor_comp_to_load.load_state_dict(torch.load(os.path.join(checkpoint_path, 'tensor_computer.pth'), map_location=map_location))
        dim_adapt_to_load.load_state_dict(torch.load(os.path.join(checkpoint_path, 'dim_adapter.pth'), map_location=map_location))
        manifold_to_load.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_module.pth'), map_location=map_location))
        point_cloud_to_load.load_state_dict(torch.load(os.path.join(checkpoint_path, 'point_cloud_generator.pth'), map_location=map_location))
        topo_to_load.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_module.pth'), map_location=map_location))
        classifier_to_load.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier.pth'), map_location=map_location))
        
        # Load optimizer states if available - only load on master process for distributed training
        if not is_distributed or rank == 0:
            if os.path.exists(os.path.join(checkpoint_path, 'feature_optimizer.pth')):
                feature_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_optimizer.pth'), map_location=map_location))
                manifold_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_optimizer.pth'), map_location=map_location))
                topo_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_optimizer.pth'), map_location=map_location))
                classifier_optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier_optimizer.pth'), map_location=map_location))
            
            # Load scheduler states if available
            if os.path.exists(os.path.join(checkpoint_path, 'feature_scheduler.pth')):
                feature_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'feature_scheduler.pth'), map_location=map_location))
                manifold_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'manifold_scheduler.pth'), map_location=map_location))
                topo_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'topo_scheduler.pth'), map_location=map_location))
                classifier_scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, 'classifier_scheduler.pth'), map_location=map_location))
        
        # Ensure all processes are synchronized after loading
        if is_distributed:
            dist.barrier()
        
        master_print("Checkpoint loaded successfully")
    
    # Validation function
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
        
        # To collect all predictions and labels for analysis
        all_preds = []
        all_labels = []
        all_probs = []
        all_uncertainty = []
        
        # Validation loop (no gradients needed)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                try:
                    # Forward pass through feature extraction
                    features, _ = feature_network(images)
                    phase_tensor = tensor_computer(features)
                    
                    # Apply dimension adapter
                    adapted_tensor = dim_adapter(phase_tensor)
                    
                    # Forward pass through manifold learning
                    manifold_features, (mu, logvar, z) = manifold_module(adapted_tensor)
                    
                    # Reconstruct for VAE loss
                    recon_tensor = manifold_module.decode(z)
                    
                    # Calculate manifold loss
                    if recon_tensor is not None:
                        manifold_loss = vae_loss(recon_tensor, adapted_tensor, mu, logvar)
                    else:
                        manifold_loss = torch.mean(mu.pow(2) + logvar.exp() - 1 - logvar)
                    
                    # Generate point cloud
                    point_cloud = point_cloud_generator(manifold_features)
                    
                    # Forward pass through topological analysis
                    topo_features, _ = topo_module(point_cloud)
                    
                    # Forward pass through classifier
                    logits, probs, uncertainty = classifier(manifold_features, topo_features)
                    
                    # Compute losses
                    classifier_loss = classification_criterion(logits, labels)
                    feature_loss = torch.mean(torch.abs(features)) * 0.01
                    topo_loss = torch.mean(torch.abs(topo_features)) * 0.01
                    
                    # Combined loss
                    loss = classifier_loss + 0.2 * manifold_loss + 0.05 * feature_loss + 0.05 * topo_loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_feature_loss += feature_loss.item()
                    val_manifold_loss += manifold_loss.item()
                    val_topo_loss += topo_loss.item()
                    
                    # Calculate accuracy
                    pred = logits.argmax(dim=1)
                    correct += pred.eq(labels).sum().item()
                    total += labels.size(0)
                    
                    # Collect predictions and labels for analysis
                    all_preds.append(pred)
                    all_labels.append(labels)
                    all_probs.append(probs)
                    all_uncertainty.append(uncertainty)
                
                except Exception as e:
                    master_print(f"Error in validation batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Free memory on error
                    torch.cuda.empty_cache()
                    
                    # Skip this batch and continue
                    continue
        
        # Synchronize metrics across processes if distributed
        if is_distributed:
            # Convert to tensors for reduction
            loss_tensor = torch.tensor(val_loss).to(device)
            feature_loss_tensor = torch.tensor(val_feature_loss).to(device)
            manifold_loss_tensor = torch.tensor(val_manifold_loss).to(device)
            topo_loss_tensor = torch.tensor(val_topo_loss).to(device)
            correct_tensor = torch.tensor(correct).to(device)
            total_tensor = torch.tensor(total).to(device)
            
            # All-reduce to sum across processes
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(feature_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(manifold_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(topo_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            
            # Update variables with reduced values
            val_loss = loss_tensor.item()
            val_feature_loss = feature_loss_tensor.item()
            val_manifold_loss = manifold_loss_tensor.item()
            val_topo_loss = topo_loss_tensor.item()
            correct = correct_tensor.item()
            total = total_tensor.item()
            
            # We need to gather predictions and labels from all processes
            if all_preds and all_labels and all_probs and all_uncertainty:
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                all_probs = torch.cat(all_probs)
                all_uncertainty = torch.cat(all_uncertainty)
                
                # Get sizes from all processes
                pred_size = torch.tensor([all_preds.size(0)], dtype=torch.long, device=device)
                sizes = [torch.zeros_like(pred_size) for _ in range(args.gpus)]
                dist.all_gather(sizes, pred_size)
                
                # Pad to max size
                max_size = max(size.item() for size in sizes)
                padded_preds = torch.zeros(max_size, dtype=all_preds.dtype, device=device)
                padded_preds[:all_preds.size(0)] = all_preds
                padded_labels = torch.zeros(max_size, dtype=all_labels.dtype, device=device)
                padded_labels[:all_labels.size(0)] = all_labels
                
                # For probs and uncertainty (2D tensors), pad accordingly
                probs_dim = all_probs.size(1) if all_probs.size(0) > 0 else 2  # Default to 2 classes if empty
                padded_probs = torch.zeros(max_size, probs_dim, dtype=all_probs.dtype, device=device)
                if all_probs.size(0) > 0:
                    padded_probs[:all_probs.size(0)] = all_probs
                
                # Uncertainty might be a scalar per example
                uncertainty_shape = all_uncertainty.shape
                if len(uncertainty_shape) == 1:
                    padded_uncertainty = torch.zeros(max_size, dtype=all_uncertainty.dtype, device=device)
                    padded_uncertainty[:all_uncertainty.size(0)] = all_uncertainty
                else:
                    uncertainty_dim = uncertainty_shape[1]
                    padded_uncertainty = torch.zeros(max_size, uncertainty_dim, dtype=all_uncertainty.dtype, device=device)
                    padded_uncertainty[:all_uncertainty.size(0)] = all_uncertainty
                
                # Gather all predictions and labels
                gathered_preds = [torch.zeros_like(padded_preds) for _ in range(args.gpus)]
                gathered_labels = [torch.zeros_like(padded_labels) for _ in range(args.gpus)]
                gathered_probs = [torch.zeros_like(padded_probs) for _ in range(args.gpus)]
                gathered_uncertainty = [torch.zeros_like(padded_uncertainty) for _ in range(args.gpus)]
                
                dist.all_gather(gathered_preds, padded_preds)
                dist.all_gather(gathered_labels, padded_labels)
                dist.all_gather(gathered_probs, padded_probs)
                dist.all_gather(gathered_uncertainty, padded_uncertainty)
                
                # Combine and trim to actual sizes
                all_gathered_preds = []
                all_gathered_labels = []
                all_gathered_probs = []
                all_gathered_uncertainty = []
                
                for i, size in enumerate(sizes):
                    if size.item() > 0:
                        all_gathered_preds.append(gathered_preds[i][:size.item()])
                        all_gathered_labels.append(gathered_labels[i][:size.item()])
                        all_gathered_probs.append(gathered_probs[i][:size.item()])
                        all_gathered_uncertainty.append(gathered_uncertainty[i][:size.item()])
                
                # Combine all gathered predictions and labels
                if all_gathered_preds:
                    all_preds = torch.cat(all_gathered_preds)
                    all_labels = torch.cat(all_gathered_labels)
                    all_probs = torch.cat(all_gathered_probs)
                    all_uncertainty = torch.cat(all_gathered_uncertainty)
                else:
                    all_preds = torch.tensor([], device=device)
                    all_labels = torch.tensor([], device=device)
                    all_probs = torch.tensor([], device=device)
                    all_uncertainty = torch.tensor([], device=device)
            else:
                master_print("Warning: Some metrics could not be collected during validation")
                all_preds = torch.tensor([], device=device)
                all_labels = torch.tensor([], device=device)
                all_probs = torch.tensor([], device=device)
                all_uncertainty = torch.tensor([], device=device)
        else:
            # Non-distributed case, just concatenate as before
            if all_preds and all_labels and all_probs and all_uncertainty:
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                all_probs = torch.cat(all_probs)
                all_uncertainty = torch.cat(all_uncertainty)
            else:
                master_print("Warning: Some metrics could not be collected during validation")
                all_preds = torch.tensor([])
                all_labels = torch.tensor([])
                all_probs = torch.tensor([])
                all_uncertainty = torch.tensor([])
        
        # Calculate metrics
        num_batches = len(val_loader)
        if is_distributed:
            # Create a tensor to hold the number of batches
            num_batches_tensor = torch.tensor(num_batches).to(device)
            # All-reduce to get the sum across processes
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
            # Get the average (each process has a different portion of the data)
            num_batches = num_batches_tensor.item() / args.gpus
        
        avg_loss = val_loss / num_batches
        avg_feature_loss = val_feature_loss / num_batches
        avg_manifold_loss = val_manifold_loss / num_batches
        avg_topo_loss = val_topo_loss / num_batches
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {
            'loss': avg_loss,
            'feature_loss': avg_feature_loss,
            'manifold_loss': avg_manifold_loss,
            'topo_loss': avg_topo_loss,
            'accuracy': accuracy,
            'predictions': all_preds.cpu(),
            'labels': all_labels.cpu(),
            'probabilities': all_probs.cpu(),
            'uncertainty': all_uncertainty.cpu()
        }
    
    # Main training function
def train_model(total_params):
    # Create dataset and loaders
    master_print("\nPreparing datasets...")
    train_loader, val_loader, train_sampler, val_sampler = create_dataset(args.data_dir, args.batch_size)
    
    if train_loader is None or val_loader is None:
        master_print("Error: Could not create data loaders. Exiting.")
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
        
        # Try to load previous metrics if available (only on master process)
        if not is_distributed or rank == 0:
            metrics_path = os.path.join(os.path.dirname(args.resume), 'training_metrics.pt')
            if os.path.exists(metrics_path):
                map_location = {'cuda:0': f'cuda:{gpu}'} if is_distributed else device
                metrics = torch.load(metrics_path, map_location=map_location)
                train_losses = metrics.get('train_losses', [])
                train_accuracies = metrics.get('train_accuracies', [])
                val_losses = metrics.get('val_losses', [])
                val_accuracies = metrics.get('val_accuracies', [])
                best_val_accuracy = metrics.get('best_accuracy', 0)
                master_print(f"Loaded previous metrics. Best validation accuracy: {best_val_accuracy:.2f}%")
    
    # Wait for all processes to be ready before starting training
    if is_distributed:
        dist.barrier()
    
    master_print(f"\nStarting training for {num_epochs} epochs with {total_params:,} parameters")
    master_print(f"Training on {args.gpus} GPUs" if is_distributed else "Training on single GPU")
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    try:
        for epoch in range(1, num_epochs + 1):
            try:
                master_print(f"\n{'='*80}\nEpoch {epoch}/{num_epochs}\n{'='*80}")
                
                # Start timing
                start_time.record()
                
                # Train for one epoch
                train_metrics = train_epoch_with_accumulation(
                    train_loader, 
                    epoch,
                    train_sampler=train_sampler,
                    accumulation_steps=gradient_accumulation_steps
                )
                
                # End timing
                end_time.record()
                torch.cuda.synchronize()
                epoch_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                
                # Store training metrics (only on master process)
                if not is_distributed or rank == 0:
                    train_losses.append(train_metrics['loss'])
                    train_accuracies.append(train_metrics['accuracy'])
                
                # Validate
                val_metrics = validate(val_loader, val_sampler)
                
                # Store validation metrics (only on master process)
                if not is_distributed or rank == 0:
                    val_losses.append(val_metrics['loss'])
                    val_accuracies.append(val_metrics['accuracy'])
                
                # Print detailed metrics (only on master process)
                if not is_distributed or rank == 0:
                    master_print(f"\nEpoch {epoch}/{num_epochs} completed in {epoch_time:.1f} seconds:")
                    master_print(f"  Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
                    master_print(f"    Feature Loss: {train_metrics['feature_loss']:.4f}")
                    master_print(f"    Manifold Loss: {train_metrics['manifold_loss']:.4f}")
                    master_print(f"    Topo Loss: {train_metrics['topo_loss']:.4f}")
                    master_print(f"    Classifier Loss: {train_metrics['classifier_loss']:.4f}")
                    master_print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                    master_print(f"  Learning Rate: {feature_optimizer.param_groups[0]['lr']:.2e}")
                
                # Update learning rate schedulers (only need to do this on one process)
                if not is_distributed or rank == 0:
                    feature_scheduler.step(val_metrics['loss'])
                    manifold_scheduler.step(val_metrics['loss'])
                    topo_scheduler.step(val_metrics['loss'])
                    classifier_scheduler.step(val_metrics['loss'])
                
                # For distributed training, broadcast the new learning rates to all processes
                if is_distributed:
                    for param_group in feature_optimizer.param_groups:
                        lr_tensor = torch.tensor([param_group['lr']], device=device)
                        dist.broadcast(lr_tensor, 0)
                        if rank != 0:
                            param_group['lr'] = lr_tensor.item()
                    
                    for param_group in manifold_optimizer.param_groups:
                        lr_tensor = torch.tensor([param_group['lr']], device=device)
                        dist.broadcast(lr_tensor, 0)
                        if rank != 0:
                            param_group['lr'] = lr_tensor.item()
                    
                    for param_group in topo_optimizer.param_groups:
                        lr_tensor = torch.tensor([param_group['lr']], device=device)
                        dist.broadcast(lr_tensor, 0)
                        if rank != 0:
                            param_group['lr'] = lr_tensor.item()
                    
                    for param_group in classifier_optimizer.param_groups:
                        lr_tensor = torch.tensor([param_group['lr']], device=device)
                        dist.broadcast(lr_tensor, 0)
                        if rank != 0:
                            param_group['lr'] = lr_tensor.item()
                
                # Save training curves (only on master process)
                if not is_distributed or rank == 0:
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
                
                # Early stopping check (only on master process)
                if not is_distributed or rank == 0:
                    if val_metrics['accuracy'] > best_val_accuracy:
                        best_val_accuracy = val_metrics['accuracy']
                        early_stopping_counter = 0
                        
                        # Save best model
                        master_print(f"  New best model with accuracy: {best_val_accuracy:.2f}%")
                        save_models(args.save_dir, epoch=f"best", metrics={
                            'train_loss': train_metrics['loss'],
                            'train_accuracy': train_metrics['accuracy'],
                            'val_loss': val_metrics['loss'],
                            'val_accuracy': val_metrics['accuracy']
                        })
                    else:
                        early_stopping_counter += 1
                        master_print(f"  No improvement in validation accuracy. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                        
                        if early_stopping_counter >= early_stopping_patience:
                            master_print(f"  Early stopping triggered after {epoch} epochs")
                            # Set a flag for early stopping that we can broadcast to all processes
                            if is_distributed:
                                early_stop_flag = torch.tensor([1], device=device)
                            else:
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
                
                # Broadcast early stopping flag to all processes if distributed
                if is_distributed:
                    # Create a tensor for early stopping flag (0 means continue, 1 means stop)
                    if rank == 0:
                        early_stop_flag = torch.tensor([1 if early_stopping_counter >= early_stopping_patience else 0], device=device)
                    else:
                        early_stop_flag = torch.tensor([0], device=device)
                    
                    # Broadcast the early stopping decision from master to all processes
                    dist.broadcast(early_stop_flag, 0)
                    
                    # If early stopping is triggered, break the loop on all processes
                    if early_stop_flag.item() == 1:
                        master_print(f"Process {rank}: Early stopping triggered, ending training")
                        break
                
                # Synchronize all processes at the end of each epoch
                if is_distributed:
                    dist.barrier()
                
            except Exception as e:
                master_print(f"Error in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to save an emergency checkpoint (only on master process)
                if not is_distributed or rank == 0:
                    try:
                        save_models(args.save_dir, epoch=f"{epoch}_emergency")
                    except Exception as ce:
                        master_print(f"Could not save emergency checkpoint: {ce}")
                
                # Free memory
                torch.cuda.empty_cache()
                
                # Continue training if possible
                if "CUDA out of memory" in str(e):
                    master_print("CUDA out of memory error detected. Attempting to continue...")
                    continue
                else:
                    raise  # Re-raise for other errors
    
    except KeyboardInterrupt:
        master_print("\nTraining interrupted by user.")
        # Save final checkpoint (only on master process)
        if not is_distributed or rank == 0:
            save_models(args.save_dir, epoch="interrupted")
    
    except Exception as e:
        master_print(f"\nTraining stopped due to error: {e}")
        import traceback
        traceback.print_exc()
        # Save emergency checkpoint (only on master process)
        if not is_distributed or rank == 0:
            try:
                save_models(args.save_dir, epoch="error_final")
            except:
                master_print("Could not save final error checkpoint.")
    
    finally:
        # Clean up distributed processes
        if is_distributed:
            try:
                dist.destroy_process_group()
                master_print(f"Process {rank}: Process group destroyed successfully")
            except Exception as e:
                master_print(f"Process {rank}: Error destroying process group: {e}")
            
        # This will run regardless of how the training ends (only on master process)
        if not is_distributed or rank == 0:
            master_print("\nTraining complete or interrupted!")
            master_print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
            
            # Final save
            try:
                save_models(args.save_dir, epoch="final")
                master_print("Final models saved to models/medium/ directory")
                
                # Save a text summary of the training
                with open(os.path.join(args.save_dir, 'results/training_summary.txt'), 'w') as f:
                    f.write(f"Medium Model Training Summary (Approx. {total_params:,} parameters)\n")
                    f.write(f"{'='*80}\n\n")
                    f.write(f"Distributed Training: {is_distributed}, Number of GPUs: {args.gpus}\n\n")
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
                        
                master_print("Training summary saved to results/training_summary.txt")
                
            except Exception as e:
                master_print(f"Error in final saving: {e}")

# In the main_worker function, after printing model parameters:
    # Calculate total_params
    total_params = (count_parameters(feature_network) + count_parameters(tensor_computer) + 
                   count_parameters(dim_adapter) + count_parameters(manifold_module) + 
                   count_parameters(point_cloud_generator) + count_parameters(topo_module) + 
                   count_parameters(classifier))
    master_print(f"Total: {total_params:,} parameters")
    
    # Call train_model with the total_params
    train_model(total_params)

# REMOVE: The duplicate code that was outside of any function:
# total_params = (count_parameters(feature_network) + count_parameters(tensor_computer) + 
#               count_parameters(dim_adapter) + count_parameters(manifold_module) + 
#               count_parameters(point_cloud_generator) + count_parameters(topo_module) + 
#               count_parameters(classifier))
# master_print(f"Total: {total_params:,} parameters")
# train_model(total_params)

# Function to spawn multiple processes for multi-GPU training
def spawn_workers(args):
    """Spawn multiple processes, one per GPU"""
    # Print startup information
    print(f"\n{'-'*80}")
    print(f"Starting PCM Medium Model Training with {args.gpus} GPUs")
    print(f"{'-'*80}")
    
    print(f"\nEnvironment:")
    print(f"  Python: {sys.version}")
    print(f"  Torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    
    print(f"\nArguments:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation}")
    print(f"  Save frequency: {args.save_freq}")
    print(f"  Resume from: {args.resume}")
    print(f"  Number of GPUs: {args.gpus}")
    
    # If no CUDA is available, fall back to CPU
    if not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead!")
        args.gpus = 0
        main_worker(0, 1, args)
        return
    
    # Check if we have enough GPUs
    num_gpus_available = torch.cuda.device_count()
    if args.gpus > num_gpus_available:
        print(f"Warning: Requested {args.gpus} GPUs but only {num_gpus_available} are available!")
        args.gpus = num_gpus_available
    
    # Single GPU or CPU case - no need for multiprocessing
    if args.gpus <= 1:
        main_worker(0, 1, args)
        return
    
    # Spawn processes for each GPU
    print(f"Spawning {args.gpus} processes for distributed training...")
    mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))

if __name__ == "__main__":
    # Parse arguments
    args = parser.parse_args()
    
    # Start the training process
    spawn_workers(args)