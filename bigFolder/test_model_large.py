#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_model_large_parameters.py

This file imports the classes from `model_large.py` and prints the number of trainable
parameters in each sub-module, along with the total parameter count.
"""

from bigModel import (
    EnhancedFeatureExtractionNetwork,
    EnhancedPhaseCorrelationTensorComputation,
    DimensionAdapter,
    ManifoldLearningModule,
    PointCloudGenerator,
    TinyTopologicalFeatureExtraction,
    ClassificationNetwork
)


def count_parameters(model: object) -> int:
    """Returns the number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Instantiate each model component
    feature_network = EnhancedFeatureExtractionNetwork()
    tensor_computer = EnhancedPhaseCorrelationTensorComputation()
    dim_adapter = DimensionAdapter()
    manifold_module = ManifoldLearningModule()
    point_cloud_generator = PointCloudGenerator()
    topo_module = TinyTopologicalFeatureExtraction(
        input_dim=128,  # Must match the output of the manifold module
        hidden_dim=128,
        output_dim=64,
        max_edge_length=2.0,
        num_filtrations=20,
        max_dimension=1
    )
    classifier = ClassificationNetwork(
        manifold_dim=128,  # must match the manifold module
        topo_dim=64,      # must match the topo module
        feature_dim=256,
        hidden_dim=512,
        dropout=0.2
    )

    # Print individual parameter counts
    print("Parameter Breakdown:")
    print(f"  1) EnhancedFeatureExtractionNetwork    : {count_parameters(feature_network):,}")
    print(f"  2) EnhancedPhaseCorrelationTensorComp. : {count_parameters(tensor_computer):,}")
    print(f"  3) DimensionAdapter                    : {count_parameters(dim_adapter):,}")
    print(f"  4) ManifoldLearningModule             : {count_parameters(manifold_module):,}")
    print(f"  5) PointCloudGenerator                : {count_parameters(point_cloud_generator):,}")
    print(f"  6) TinyTopologicalFeatureExtraction    : {count_parameters(topo_module):,}")
    print(f"  7) ClassificationNetwork               : {count_parameters(classifier):,}")

    # Compute total
    total_params = (
        count_parameters(feature_network)
        + count_parameters(tensor_computer)
        + count_parameters(dim_adapter)
        + count_parameters(manifold_module)
        + count_parameters(point_cloud_generator)
        + count_parameters(topo_module)
        + count_parameters(classifier)
    )
    print(f"\nTotal Parameters: {total_params:,}")
    
    # Compare with original model
    try:
        from midFolder.model import (
            EnhancedFeatureExtractionNetwork as OriginalFeatureNetwork,
            EnhancedPhaseCorrelationTensorComputation as OriginalTensorComputer,
            DimensionAdapter as OriginalDimensionAdapter,
            ManifoldLearningModule as OriginalManifoldModule,
            PointCloudGenerator as OriginalPointCloudGenerator,
            TinyTopologicalFeatureExtraction as OriginalTopoModule,
            ClassificationNetwork as OriginalClassifier
        )
        
        # Instantiate original model components
        original_feature_network = OriginalFeatureNetwork()
        original_tensor_computer = OriginalTensorComputer()
        original_dim_adapter = OriginalDimensionAdapter()
        original_manifold_module = OriginalManifoldModule()
        original_point_cloud_generator = OriginalPointCloudGenerator()
        original_topo_module = OriginalTopoModule(
            input_dim=32,
            hidden_dim=32,
            output_dim=16,
            max_edge_length=2.0,
            num_filtrations=10,
            max_dimension=0
        )
        original_classifier = OriginalClassifier(
            manifold_dim=32,
            topo_dim=16,
            feature_dim=32,
            hidden_dim=64,
            dropout=0.1
        )
        
        # Calculate total original parameters
        original_total_params = (
            count_parameters(original_feature_network)
            + count_parameters(original_tensor_computer)
            + count_parameters(original_dim_adapter)
            + count_parameters(original_manifold_module)
            + count_parameters(original_point_cloud_generator)
            + count_parameters(original_topo_module)
            + count_parameters(original_classifier)
        )
        
        # Print comparison
        print("\nComparison with Original Model:")
        print(f"  Original Model Total Parameters: {original_total_params:,}")
        print(f"  Large Model Total Parameters:   {total_params:,}")
        print(f"  Scale Factor: {total_params / original_total_params:.2f}x")
        
        # Print component-wise comparison
        print("\nComponent-wise Scale Factors:")
        print(f"  1) Feature Extraction Network: {count_parameters(feature_network) / count_parameters(original_feature_network):.2f}x")
        print(f"  2) Phase Correlation Tensor:  {count_parameters(tensor_computer) / count_parameters(original_tensor_computer):.2f}x")
        print(f"  3) Dimension Adapter:         {count_parameters(dim_adapter) / count_parameters(original_dim_adapter):.2f}x")
        print(f"  4) Manifold Learning Module:  {count_parameters(manifold_module) / count_parameters(original_manifold_module):.2f}x")
        print(f"  5) Point Cloud Generator:     {count_parameters(point_cloud_generator) / count_parameters(original_point_cloud_generator):.2f}x")
        print(f"  6) Topological Feature Ext.:  {count_parameters(topo_module) / count_parameters(original_topo_module):.2f}x")
        print(f"  7) Classification Network:    {count_parameters(classifier) / count_parameters(original_classifier):.2f}x")
        
        # Pie chart data for parameter distribution
        print("\nParameter Distribution (% of total):")
        print(f"  1) Feature Extraction Network: {count_parameters(feature_network) / total_params * 100:.1f}%")
        print(f"  2) Phase Correlation Tensor:  {count_parameters(tensor_computer) / total_params * 100:.1f}%")
        print(f"  3) Dimension Adapter:         {count_parameters(dim_adapter) / total_params * 100:.1f}%")
        print(f"  4) Manifold Learning Module:  {count_parameters(manifold_module) / total_params * 100:.1f}%")
        print(f"  5) Point Cloud Generator:     {count_parameters(point_cloud_generator) / total_params * 100:.1f}%")
        print(f"  6) Topological Feature Ext.:  {count_parameters(topo_module) / total_params * 100:.1f}%")
        print(f"  7) Classification Network:    {count_parameters(classifier) / total_params * 100:.1f}%")
        
    except ImportError:
        print("\nCouldn't import original model for comparison. Make sure 'model.py' is available.")
    
    # Optional GPU memory estimation
    try:
        import torch
        import math
        
        # Rough estimate of memory usage per parameter (assuming float32)
        bytes_per_parameter = 4  # float32 = 4 bytes
        
        # Additional memory for gradients, optimizer state, etc. (rough multiplier)
        memory_multiplier = 3  # accounts for parameters, gradients, and optimizer state
        
        # Calculate estimated memory requirements
        estimated_memory_bytes = total_params * bytes_per_parameter * memory_multiplier
        
        # Convert to more readable units
        if estimated_memory_bytes < 1e6:
            memory_str = f"{estimated_memory_bytes / 1e3:.2f} KB"
        elif estimated_memory_bytes < 1e9:
            memory_str = f"{estimated_memory_bytes / 1e6:.2f} MB"
        else:
            memory_str = f"{estimated_memory_bytes / 1e9:.2f} GB"
        
        print(f"\nEstimated GPU Memory Usage: {memory_str} (for float32 parameters, gradients, and optimizer state)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()