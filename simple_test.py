#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
simple_test.py

A simplified test for multi-process training without using torch.distributed.
This should run on any system, even without CUDA.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 112 * 112, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 112 * 112)
        x = self.classifier(x)
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

def train_model(process_id, total_processes, shared_dict):
    """Train a model in a separate process."""
    print(f"Process {process_id} starting")
    
    # Set device to CPU
    device = torch.device('cpu')
    
    # Create model
    model = SimpleModel().to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dataset and loader - each process gets different data
    dataset_size = 100
    batch_size = 4
    
    # Split data across processes
    per_process = dataset_size // total_processes
    start_idx = process_id * per_process
    end_idx = start_idx + per_process
    
    # Create dataset
    dataset = DummyDataset(size=per_process)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Process batches
        for i, (inputs, labels) in enumerate(dataloader):
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Print progress
            if i % 5 == 4:
                print(f"Process {process_id}, Epoch {epoch+1}, Batch {i+1}, "
                      f"Loss: {running_loss/5:.4f}, "
                      f"Acc: {100 * correct/total:.2f}%")
                running_loss = 0.0
    
    # Store first layer weight in shared dictionary for comparison
    param = next(model.parameters())
    shared_dict[f'weight_{process_id}'] = param[0, 0, 0, 0].item()
    print(f"Process {process_id} finished, first weight: {param[0, 0, 0, 0].item()}")

def main():
    """Run training in multiple processes."""
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Number of processes to use
    num_processes = 2
    
    # Use Manager for sharing data between processes
    manager = mp.Manager()
    shared_dict = manager.dict()
    
    # Create processes
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=train_model, args=(i, num_processes, shared_dict))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Compare results from different processes
    print("Training complete. Results from processes:")
    for key, value in shared_dict.items():
        print(f"{key}: {value}")
    
    # Note: In true distributed training, the parameters would be synchronized
    # This test shows that multiple processes can train independently

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main() 