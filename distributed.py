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
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import pandas as pd
import argparse
from pathlib import Path
import time
from datetime import timedelta

# Import our model components from mediumModel.py
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

# Parse command line arguments
parser = argparse.ArgumentParser(description="PCM Model Distributed Training")
parser.add_argument('--data_dir', type=str, default='./data', 
                    help='Directory containing the dataset')
parser.add_argument('--save_dir', type=str, default='./model_checkpoints', 
                    help='Directory to save model checkpoints')
parser.add_argument('--epochs', type=int, default=15, 
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, 
                    help='Batch size per GPU')
parser.add_argument('--gradient_accumulation', type=int, default=8, 
                    help='Number of gradient accumulation steps')
parser.add_argument('--save_freq', type=int, default=50, 
                    help='Save checkpoint frequency (batches)')
parser.add_argument('--resume', type=str, default=None, 
                    help='Path to checkpoint to resume from')
parser.add_argument('--world_size', type=int, default=None, 
                    help='Number of processes for distributed training')
parser.add_argument('--rank', type=int, default=None, 
                    help='Node rank for distributed training')
parser.add_argument('--local_rank', type=int, default=None, 
                    help='Local rank for distributed training')
parser.add_argument('--dist_url', default='env://', 
                    help='URL used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', 
                    help='Distributed backend: nccl, gloo, etc.')
parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for reproducibility')
parser.add_argument('--use_env', action='store_true', 
                    help='Use environment variables for distributed training')
parser.add_argument('--use_spawn', action='store_true', 
                    help='Use spawn method for multi-processing')
parser.add_argument('--evaluate', action='store_true', 
                    help='Evaluate model on validation set')
parser.add_argument('--cpu_only', action='store_true', 
                    help='Force CPU only mode for testing')
parser.add_argument('--file_store', action='store_true', 
                    help='Use file store instead of env for distributed init (not used anymore)')
parser.add_argument('--simulate_multi_gpu', type=int, default=0,
                    help='Simulate multiple GPUs (2-4) even on a single GPU system')

# Function to initialize distributed training
def init_distributed():
    """Initialize distributed training environment."""
    args = parser.parse_args()
    
    # Support both torch.distributed.launch and manual setting
    if args.use_env:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.rank = int(os.environ.get("RANK", 0))
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        # For manual launch, if args not specified
        if args.local_rank is None:
            args.local_rank = 0
        if args.rank is None:
            args.rank = 0
        if args.world_size is None:
            if args.cpu_only or not torch.cuda.is_available():
                args.world_size = 2  # Use 2 processes for CPU-only mode
            else:
                args.world_size = torch.cuda.device_count()

    # Set up device
    if not args.cpu_only and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    # Initialize distributed process group
    # Always use gloo backend since NCCL is not built into PyTorch
    backend = "gloo"
    
    # Use TCP initialization for all cases now
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend=backend,
        init_method='tcp://localhost:12355',
        world_size=args.world_size,
        rank=args.rank
    )
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create directories for saving checkpoints
    if args.rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'results'), exist_ok=True)
    
    return args

# Custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, image_files, labels, target_size=(448, 448)):
        self.image_files = image_files
        self.labels = labels
        self.target_size = target_size
        
        # Define transforms using standard torchvision transformations
        self.transform = transforms.Compose([
            transforms.Resize(target_size),  # Upsample to target size
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
    """Load and combine datasets from available CSV files"""
    # Updated paths based on possible file structures
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
            files, labels = load_dataset_from_csv(dataset_path, verbose=True, is_master=is_master)
            all_files.extend(files)
            all_labels.extend(labels)
            if is_master:
                print(f"Dataset {i+1}: {len(files)} images")
        except Exception as e:
            if is_master:
                print(f"Error loading dataset {dataset_path}: {e}")
    
    # Check if we have any data
    if len(all_files) == 0:
        raise ValueError(f"No images found in any of the dataset paths: {dataset_paths}")
    
    if is_master:
        print(f"Combined dataset: {len(all_files)} images")
        print(f"Human images (label 0): {all_labels.count(0)}")
        print(f"AI-generated images (label 1): {all_labels.count(1)}")
    
    return all_files, all_labels

def create_dataset(data_dir, batch_size, rank, world_size, is_master=False):
    """Create dataset and data loaders for distributed training"""
    # Load all files and labels
    all_files, all_labels = combine_datasets(data_dir, is_master=is_master)
    
    # Create combined dataset
    dataset = CustomImageDataset(all_files, all_labels)
    
    # Split dataset into training and validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    generator = torch.Generator().manual_seed(42)  # For reproducibility
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    if is_master:
        print(f"Training dataset: {train_size} images")
        print(f"Validation dataset: {val_size} images")
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler, val_sampler

def create_model(device):
    """Initialize model components with DDP-compatible parameters."""
    # Initialize model components
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
    
    # Return all model components
    return {
        "feature_network": feature_network,
        "tensor_computer": tensor_computer,
        "dim_adapter": dim_adapter,
        "manifold_module": manifold_module,
        "point_cloud_generator": point_cloud_generator,
        "topo_module": topo_module,
        "classifier": classifier
    }

def wrap_models_with_ddp(models, device, world_size, rank, cpu_only=False):
    """Wrap model components with DistributedDataParallel."""
    ddp_models = {}
    for name, model in models.items():
        if cpu_only or device.type == 'cpu':
            # CPU mode
            ddp_models[name] = DDP(model)
        elif world_size > torch.cuda.device_count():
            # Simulation mode (multiple workers per GPU)
            ddp_models[name] = DDP(
                model, 
                device_ids=[device.index], 
                output_device=device.index
            )
        else:
            # Normal mode (one worker per GPU)
            ddp_models[name] = DDP(
                model, 
                device_ids=[device.index],
                output_device=device.index
            )
    return ddp_models

def count_parameters(models):
    """Count parameters in all model components."""
    total_params = 0
    params_by_component = {}
    
    for name, model in models.items():
        if isinstance(model, DDP):
            model = model.module
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_by_component[name] = params
        total_params += params
    
    return params_by_component, total_params

def create_optimizers(models, lr=1e-4, weight_decay=1e-5):
    """Create optimizers for all model components."""
    optimizers = {}
    
    for name, model in models.items():
        if isinstance(model, DDP):
            model = model.module
        optimizers[name] = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    return optimizers

def vae_loss(recon_x, x, mu, logvar):
    """Compute VAE loss (reconstruction + KL divergence)."""
    # Compute reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # Compute KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss is weighted sum
    total_loss = recon_loss + 0.1 * kl_loss
    
    return total_loss, recon_loss, kl_loss

def save_models(models, optimizers, epoch, metrics, save_dir, is_master=False):
    """Save model checkpoints (only from master process)."""
    if not is_master:
        return
    
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'models': {},
        'optimizers': {}
    }
    
    # Save model states
    for name, model in models.items():
        if isinstance(model, DDP):
            checkpoint['models'][name] = model.module.state_dict()
        else:
            checkpoint['models'][name] = model.state_dict()
    
    # Save optimizer states
    for name, optimizer in optimizers.items():
        checkpoint['optimizers'][name] = optimizer.state_dict()
    
    # Save the checkpoint
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save the best model if this is the best validation accuracy
    if 'best_val_acc' not in metrics or metrics.get('val_acc', 0) > metrics.get('best_val_acc', 0):
        metrics['best_val_acc'] = metrics.get('val_acc', 0)
        best_path = os.path.join(save_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model with validation accuracy: {metrics['best_val_acc']:.4f}")

def resume_from_checkpoint(models, optimizers, checkpoint_path, device):
    """Resume training from a checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return 0, {}
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    for name, state_dict in checkpoint['models'].items():
        if name in models:
            if isinstance(models[name], DDP):
                models[name].module.load_state_dict(state_dict)
            else:
                models[name].load_state_dict(state_dict)
    
    # Load optimizer states
    for name, state_dict in checkpoint['optimizers'].items():
        if name in optimizers:
            optimizers[name].load_state_dict(state_dict)
    
    return checkpoint.get('epoch', 0) + 1, checkpoint.get('metrics', {})

def train_epoch(models, optimizers, train_loader, train_sampler, epoch, args, device):
    """Run one training epoch with gradient accumulation."""
    # Set all models to training mode
    for model in models.values():
        model.train()
    
    # Set the epoch for the train sampler
    train_sampler.set_epoch(epoch)
    
    # Initialize running metrics
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    
    # Get the gradient accumulation steps
    accumulation_steps = args.gradient_accumulation
    
    # Use tqdm for progress tracking if this is the master process
    is_master = args.rank == 0
    iterator = tqdm(train_loader, desc=f"Epoch {epoch}") if is_master else train_loader
    
    # Zero the gradients at the beginning
    for optimizer in optimizers.values():
        optimizer.zero_grad()
    
    # Process each batch
    for batch_idx, (images, labels) in enumerate(iterator):
        # Move data to the device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass through feature network
        features = models["feature_network"](images)
        
        # Compute tensor correlations
        tensors = models["tensor_computer"](features)
        
        # Adapt dimensions
        adapted_tensors = models["dim_adapter"](tensors)
        
        # Process through manifold module (VAE-like)
        manifold_output = models["manifold_module"](adapted_tensors)
        manifold_features = manifold_output["latent"]
        recon = manifold_output["recon"]
        mu = manifold_output["mu"]
        logvar = manifold_output["logvar"]
        
        # Generate point cloud
        point_cloud = models["point_cloud_generator"](manifold_features)
        
        # Extract topological features
        topo_features = models["topo_module"](point_cloud)
        
        # Classification
        logits = models["classifier"](manifold_features, topo_features)
        
        # Compute BCE loss for classification
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float().unsqueeze(1))
        
        # Compute VAE loss components
        vae_total, recon_loss, kl_loss = vae_loss(recon, adapted_tensors, mu, logvar)
        
        # Total loss - weighted combination
        loss = bce_loss + 0.01 * vae_total
        
        # Scale the loss according to accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).long().squeeze()
            correct = (predictions == labels).sum().item()
            
            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss.item() * accumulation_steps  # Scale back for reporting
            running_loss += loss.item() * accumulation_steps
        
        # Only update optimizer after accumulating gradients
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Update the parameters
            for optimizer in optimizers.values():
                optimizer.step()
            
            # Zero the gradients
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            
            # Log progress
            if is_master and (batch_idx + 1) % (accumulation_steps * 5) == 0:
                avg_loss = running_loss / (accumulation_steps * 5)
                acc = total_correct / total_samples if total_samples > 0 else 0
                tqdm.write(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
                running_loss = 0.0
            
            # Save checkpoint if requested
            if is_master and args.save_freq > 0 and (batch_idx + 1) % (accumulation_steps * args.save_freq) == 0:
                metrics = {
                    'train_loss': total_loss / total_samples if total_samples > 0 else 0,
                    'train_acc': total_correct / total_samples if total_samples > 0 else 0
                }
                save_models(
                    models, optimizers, 
                    f"{epoch}_{batch_idx + 1}",
                    metrics, args.save_dir,
                    is_master=is_master
                )
    
    # Compute epoch metrics
    metrics = {}
    if total_samples > 0:
        metrics['train_loss'] = total_loss / total_samples
        metrics['train_acc'] = total_correct / total_samples
    
    return metrics

def validate(models, val_loader, val_sampler, device):
    """Evaluate model on validation set."""
    # Set all models to evaluation mode
    for model in models.values():
        model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_labels = []
    
    # Process validation data
    with torch.no_grad():
        for images, labels in val_loader:
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass through feature network
            features = models["feature_network"](images)
            
            # Compute tensor correlations
            tensors = models["tensor_computer"](features)
            
            # Adapt dimensions
            adapted_tensors = models["dim_adapter"](tensors)
            
            # Process through manifold module (VAE-like)
            manifold_output = models["manifold_module"](adapted_tensors)
            manifold_features = manifold_output["latent"]
            recon = manifold_output["recon"]
            mu = manifold_output["mu"]
            logvar = manifold_output["logvar"]
            
            # Generate point cloud
            point_cloud = models["point_cloud_generator"](manifold_features)
            
            # Extract topological features
            topo_features = models["topo_module"](point_cloud)
            
            # Classification
            logits = models["classifier"](manifold_features, topo_features)
            
            # Compute BCE loss for classification
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float().unsqueeze(1))
            
            # Compute VAE loss components
            vae_total, recon_loss, kl_loss = vae_loss(recon, adapted_tensors, mu, logvar)
            
            # Total loss - weighted combination
            loss = bce_loss + 0.01 * vae_total
            
            # Update metrics
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            
            predictions = (probs > 0.5).long().squeeze()
            correct = (predictions == labels).sum().item()
            
            total_correct += correct
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
    
    # Compute metrics
    metrics = {}
    if total_samples > 0:
        metrics['val_loss'] = total_loss / total_samples
        metrics['val_acc'] = total_correct / total_samples
    
    return metrics

def setup_worker(rank, world_size, args):
    """Initialize distributed environment for a single worker."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Always use gloo backend since NCCL is not built into PyTorch
    backend = "gloo"
    
    # Initialize process group with TCP initialization
    dist.init_process_group(
        backend=backend,
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )
    
    print(f"Initialized process {rank} / {world_size} with {backend} backend")

def train_and_evaluate(rank, world_size, args):
    """Main training and evaluation function for distributed training."""
    # Initialize process group for this worker
    if args.use_spawn:
        # Use the setup function with unique store path
        setup_worker(rank, world_size, args)
    
    # Check if this is the master process
    is_master = rank == 0
    
    # Set up the device
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        if is_master:
            print(f"Using CPU for training")
    elif world_size > torch.cuda.device_count():
        # Simulation mode: multiple processes share the same GPU
        device_idx = rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{device_idx}')
        if is_master:
            print(f"SIMULATION MODE: {world_size} processes sharing {torch.cuda.device_count()} GPU(s)")
        print(f"Process {rank} using device: {device} (shared)")
        
        # Set device for CUDA operations
        torch.cuda.set_device(device_idx)
    else:
        # Normal mode: one process per GPU
        device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
        torch.cuda.set_device(device.index)
        if is_master:
            print(f"Training with {world_size} GPUs")
        print(f"Process {rank} using device: {device}")
    
    # Create data loaders with distributed samplers
    train_loader, val_loader, train_sampler, val_sampler = create_dataset(
        args.data_dir, args.batch_size, rank, world_size, is_master=is_master
    )
    
    # Create model components
    models = create_model(device)
    
    # Wrap models with DDP
    models = wrap_models_with_ddp(models, device, world_size, rank, cpu_only=args.cpu_only)
    
    # Create optimizers
    optimizers = create_optimizers(models)
    
    # Print model statistics if master
    if is_master:
        params_by_component, total_params = count_parameters(models)
        for name, params in params_by_component.items():
            print(f"{name}: {params:,} parameters")
        print(f"Total: {total_params:,} parameters")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    metrics = {}
    if args.resume:
        start_epoch, metrics = resume_from_checkpoint(
            models, optimizers, args.resume, device
        )
    
    # If just evaluate, don't train
    if args.evaluate:
        val_metrics = validate(models, val_loader, val_sampler, device)
        if is_master:
            print(f"Validation results: Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.4f}")
        return
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        train_metrics = train_epoch(
            models, optimizers, train_loader, train_sampler, epoch, args, device
        )
        
        # Validate
        val_metrics = validate(models, val_loader, val_sampler, device)
        
        # Update and print metrics
        metrics.update(train_metrics)
        metrics.update(val_metrics)
        
        if is_master:
            print(f"Epoch {epoch} results: "
                  f"Train Loss: {metrics['train_loss']:.4f}, "
                  f"Train Acc: {metrics['train_acc']:.4f}, "
                  f"Val Loss: {metrics['val_loss']:.4f}, "
                  f"Val Acc: {metrics['val_acc']:.4f}")
        
        # Save checkpoint
        save_models(
            models, optimizers, epoch, metrics, args.save_dir, is_master=is_master
        )
    
    # Clean up distributed process group
    if args.use_spawn:
        cleanup()

def spawn_workers(args):
    """Spawn multiple processes for multiprocessing-based distributed training."""
    if args.cpu_only or not torch.cuda.is_available():
        world_size = 2  # Use 2 processes for CPU testing
        args.cpu_only = True  # Force CPU mode
        print(f"Using spawn to launch {world_size} CPU processes")
    elif args.simulate_multi_gpu > 0:
        # Simulate multiple GPUs even with a single GPU
        world_size = min(max(args.simulate_multi_gpu, 2), 4)  # Between 2 and 4
        print(f"SIMULATION MODE: Using spawn to simulate {world_size} GPU processes on {torch.cuda.device_count()} actual GPU(s)")
    else:
        world_size = args.world_size or torch.cuda.device_count()
        print(f"Using spawn to launch {world_size} GPU processes")
    
    mp.spawn(
        train_and_evaluate,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

def main():
    """Main entry point for the distributed training script."""
    # Parse command-line arguments
    args = init_distributed()
    
    # Initialize torch distributed environment
    if args.use_spawn:
        # Using spawn for multi-processing
        spawn_workers(args)
    else:
        # Using torch.distributed.launch or other methods
        train_and_evaluate(args.rank, args.world_size, args)

# Main entry point
if __name__ == "__main__":
    main() 