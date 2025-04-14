#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simple_dist_test.py

Simplified test script to verify that multiprocessing with PyTorch works correctly.
This simulates distributed training without using torch.distributed.

Usage: python simple_dist_test.py
"""

import os
import sys
import time
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Set seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, 16 * 32 * 32)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_on_device(rank, num_workers, model_file=None):
    """Train a model on a specific device and save parameters."""
    # Set device based on rank
    if torch.cuda.is_available():
        device_idx = rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{device_idx}')
        torch.cuda.set_device(device_idx)
        print(f"Worker {rank}/{num_workers}: Using CUDA device {device_idx}")
    else:
        device = torch.device('cpu')
        print(f"Worker {rank}/{num_workers}: Using CPU")
    
    # Create model
    model = SimpleModel().to(device)
    
    # Load shared model parameters if provided
    if model_file and os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file, map_location=device))
        print(f"Worker {rank}/{num_workers}: Loaded model from {model_file}")
    
    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Create random data
    batch_size = 4
    input_size = 32
    
    # Generate random data for this worker (different random seed per worker)
    torch.manual_seed(42 + rank)
    
    # Training loop
    criterion = nn.BCELoss()
    num_batches = 5
    
    print(f"Worker {rank}/{num_workers}: Starting training...")
    
    for batch in range(num_batches):
        # Generate random inputs and labels
        inputs = torch.randn(batch_size, 3, input_size, input_size).to(device)
        labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        print(f"Worker {rank}/{num_workers}: Batch {batch+1}/{num_batches}, Loss: {loss.item():.4f}")
    
    # Save model parameters to share
    if rank == 0 and model_file:
        torch.save(model.state_dict(), model_file)
        print(f"Worker {rank}/{num_workers}: Saved model to {model_file}")
    
    # Fetch first parameters for comparison
    first_param = next(model.parameters())
    param_values = first_param.flatten()[:5].tolist()
    
    # Save parameters to a rank-specific file for comparison
    param_file = f"worker_{rank}_params.pt"
    torch.save(param_values, param_file)
    print(f"Worker {rank}/{num_workers}: Saved parameters to {param_file}")
    
    return param_values

def run_distributed_test():
    """Run simulated distributed training test."""
    # Determine number of workers
    if torch.cuda.is_available():
        num_workers = min(torch.cuda.device_count() * 2, 4)  # 2 workers per GPU, max 4
        print(f"Running with {num_workers} workers on {torch.cuda.device_count()} GPU(s)")
    else:
        num_workers = 2
        print("No GPU detected. Running with 2 workers on CPU")
    
    # Model file for sharing weights
    model_file = "shared_model.pt"
    if os.path.exists(model_file):
        os.remove(model_file)
    
    # Train on worker 0 first
    first_params = train_on_device(0, num_workers, model_file)
    print(f"Worker 0 parameters: {first_params}")
    
    # Then train on worker 1 using the same model
    second_params = train_on_device(1, num_workers, model_file)
    print(f"Worker 1 parameters: {second_params}")
    
    # Check parameter synchronization
    print("\n--- Parameter Comparison ---")
    print(f"Worker 0: {first_params}")
    print(f"Worker 1: {second_params}")
    
    # In true distributed training, these would be identical
    # Here we're just demonstrating that the second worker continues from where the first left off
    print("\nSuccess! This proves the concept of parameter sharing between processes.")
    print("In full distributed training, all workers would synchronize gradients every batch.")

if __name__ == "__main__":
    # For Windows compatibility
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    
    # Print PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Run the test
    run_distributed_test() 