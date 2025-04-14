#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_dist.py

Test script for distributed training functionality.
This script validates if the distributed training setup works correctly.

Usage:
    - For multi-worker simulation on single GPU: python test_dist.py --simulate_multi_gpu 2
    - Or specify exact number of workers: python test_dist.py --world_size 2 --use_spawn
"""

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 112 * 112, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # 224 -> 112
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(-1, 32 * 112 * 112)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Simple dataset for testing
class DummyDataset(Dataset):
    def __init__(self, size=100, img_size=224):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image and label
        img = torch.randn(3, self.img_size, self.img_size)
        label = torch.randint(0, 2, (1,)).float()
        return img, label

def setup(rank, world_size):
    """Initialize distributed environment using TCP."""
    # Set environment variables for distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group with GLOO backend
    dist.init_process_group(
        backend="gloo",
        init_method='tcp://localhost:12355',
        rank=rank,
        world_size=world_size
    )
    
    print(f"Rank {rank}/{world_size}: Initialized distributed process group")

def cleanup():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()

def worker_fn(rank, world_size, args):
    """Run test for each worker."""
    # Initialize process group
    setup(rank, world_size)
    
    # Device setup
    if args.cpu_only or not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"Rank {rank}/{world_size}: Using CPU for computation")
    elif world_size > torch.cuda.device_count():
        # Simulation mode: multiple workers share GPUs
        device_idx = rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{device_idx}')
        print(f"Rank {rank}/{world_size}: Using device {device} (shared)")
        torch.cuda.set_device(device_idx)
    else:
        # Normal mode: one worker per GPU
        device = torch.device(f'cuda:{rank}')
        print(f"Rank {rank}/{world_size}: Using device {device}")
        torch.cuda.set_device(rank)
    
    # Create model and move to device
    model = SimpleModel().to(device)
    
    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)
    
    # Create optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    
    # Create dataset
    dataset = DummyDataset(size=100)
    
    # Create sampler for distributed training
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=0
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Process batches
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Print progress every 10 batches
            if i % 10 == 9:
                print(f"Rank {rank}/{world_size}, Epoch {epoch+1}, "
                      f"Batch {i+1}/{len(dataloader)}, "
                      f"Loss: {running_loss/10:.4f}, "
                      f"Acc: {100*correct/total:.2f}%")
                running_loss = 0.0
    
    # Wait for all processes to finish training
    dist.barrier()
    
    # Verify parameter synchronization by printing first weights
    param = next(ddp_model.parameters())
    param_slice = param.flatten()[:5].tolist()
    print(f"Rank {rank}/{world_size}: First parameters: {param_slice}")
    
    # Cleanup
    cleanup()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Distributed Training")
    parser.add_argument('--world_size', type=int, default=None,
                        help='Number of processes for distributed training')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU mode for testing')
    parser.add_argument('--use_spawn', action='store_true',
                        help='Use spawn method for multiprocessing')
    parser.add_argument('--simulate_multi_gpu', type=int, default=0,
                        help='Simulate given number of workers on available GPUs')
    return parser.parse_args()

def main():
    """Main entry point for test script."""
    args = parse_args()
    
    # Print system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Determine world size (number of processes)
    if args.simulate_multi_gpu > 0:
        world_size = min(max(args.simulate_multi_gpu, 2), 4)  # Between 2 and 4
        print(f"SIMULATION MODE: Using {world_size} workers with {torch.cuda.device_count()} GPUs")
    elif args.world_size is not None:
        world_size = args.world_size
    elif args.cpu_only:
        world_size = 2
    elif torch.cuda.is_available():
        world_size = torch.cuda.device_count()
    else:
        world_size = 2
        args.cpu_only = True
    
    print(f"Starting distributed test with {world_size} workers")
    
    # For Windows, we must use spawn
    if sys.platform.startswith('win'):
        args.use_spawn = True
    
    # Run workers
    if args.use_spawn:
        # Use multiprocessing spawn
        mp.spawn(
            worker_fn,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        # Manual process creation
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=worker_fn, args=(rank, world_size, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
    print("Distributed test completed successfully!")

if __name__ == "__main__":
    # For Windows compatibility
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    main() 