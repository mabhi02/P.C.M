#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_medium_model.py

This file imports the classes from `mediumModel.py` and prints the number of trainable
parameters in each sub-module, along with the total parameter count.
"""

from mediumModel import (
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
    # Instantiate each model component with medium architecture settings
    feature_network = EnhancedFeatureExtractionNetwork(feature_dim=256)
    tensor_computer = EnhancedPhaseCorrelationTensorComputation(feature_dim=256, output_dim=512)
    dim_adapter = DimensionAdapter(input_dim=512, output_dim=256)
    manifold_module = ManifoldLearningModule(
        input_dim=256,
        hidden_dim=512,
        latent_dim=64,
        gnn_hidden_dim=128
    )
    point_cloud_generator = PointCloudGenerator(num_points=64)
    topo_module = TinyTopologicalFeatureExtraction(
        input_dim=64,
        hidden_dim=64,
        output_dim=32,
        max_edge_length=2.0,
        num_filtrations=16,
        max_dimension=1
    )
    classifier = ClassificationNetwork(
        manifold_dim=64,
        topo_dim=32,
        feature_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=6,
        dropout=0.15
    )

    # Print individual parameter counts
    print("Parameter Breakdown:")
    print(f"  1) EnhancedFeatureExtractionNetwork    : {count_parameters(feature_network):,}")
    print(f"  2) EnhancedPhaseCorrelationTensorComp. : {count_parameters(tensor_computer):,}")
    print(f"  3) DimensionAdapter                    : {count_parameters(dim_adapter):,}")
    print(f"  4) ManifoldLearningModule              : {count_parameters(manifold_module):,}")
    print(f"  5) PointCloudGenerator                 : {count_parameters(point_cloud_generator):,}")
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
    print(f"Target was ~50M parameters")
    
    # Compare with small model
    try:
        from midFolder.model import (
            EnhancedFeatureExtractionNetwork as SmallFeatureNetwork,
            EnhancedPhaseCorrelationTensorComputation as SmallTensorComputer,
            DimensionAdapter as SmallDimensionAdapter,
            ManifoldLearningModule as SmallManifoldModule,
            PointCloudGenerator as SmallPointCloudGenerator,
            TinyTopologicalFeatureExtraction as SmallTopoModule,
            ClassificationNetwork as SmallClassifier
        )
        
        # Instantiate small model components
        small_feature_network = SmallFeatureNetwork()
        small_tensor_computer = SmallTensorComputer()
        small_dim_adapter = SmallDimensionAdapter()
        small_manifold_module = SmallManifoldModule()
        small_point_cloud_generator = SmallPointCloudGenerator()
        small_topo_module = SmallTopoModule(
            input_dim=32,
            hidden_dim=32,
            output_dim=16,
            max_edge_length=2.0,
            num_filtrations=10,
            max_dimension=0
        )
        small_classifier = SmallClassifier(
            manifold_dim=32,
            topo_dim=16,
            feature_dim=32,
            hidden_dim=64,
            dropout=0.1
        )
        
        # Calculate total small model parameters
        small_total_params = (
            count_parameters(small_feature_network)
            + count_parameters(small_tensor_computer)
            + count_parameters(small_dim_adapter)
            + count_parameters(small_manifold_module)
            + count_parameters(small_point_cloud_generator)
            + count_parameters(small_topo_module)
            + count_parameters(small_classifier)
        )
        
        # Compare with large model
        try:
            from bigFolder.bigModel import (
                EnhancedFeatureExtractionNetwork as LargeFeatureNetwork,
                EnhancedPhaseCorrelationTensorComputation as LargeTensorComputer,
                DimensionAdapter as LargeDimensionAdapter,
                ManifoldLearningModule as LargeManifoldModule,
                PointCloudGenerator as LargePointCloudGenerator,
                TinyTopologicalFeatureExtraction as LargeTopoModule,
                ClassificationNetwork as LargeClassifier
            )
            
            # Instantiate large model components
            large_feature_network = LargeFeatureNetwork()
            large_tensor_computer = LargeTensorComputer()
            large_dim_adapter = LargeDimensionAdapter()
            large_manifold_module = LargeManifoldModule()
            large_point_cloud_generator = LargePointCloudGenerator()
            large_topo_module = LargeTopoModule(
                input_dim=128,
                hidden_dim=128,
                output_dim=64,
                max_edge_length=2.0,
                num_filtrations=20,
                max_dimension=1
            )
            large_classifier = LargeClassifier(
                manifold_dim=128,
                topo_dim=64,
                feature_dim=256,
                hidden_dim=512,
                dropout=0.2
            )
            
            # Calculate total large model parameters
            large_total_params = (
                count_parameters(large_feature_network)
                + count_parameters(large_tensor_computer)
                + count_parameters(large_dim_adapter)
                + count_parameters(large_manifold_module)
                + count_parameters(large_point_cloud_generator)
                + count_parameters(large_topo_module)
                + count_parameters(large_classifier)
            )
            
            # Print comprehensive comparison
            print("\nModel Size Comparison:")
            print(f"  Small Model (9M)  : {small_total_params:,} parameters")
            print(f"  Medium Model (50M): {total_params:,} parameters")
            print(f"  Large Model (221M): {large_total_params:,} parameters")
            
            print("\nScale Factors:")
            print(f"  Medium vs Small: {total_params / small_total_params:.2f}x")
            print(f"  Large vs Medium: {large_total_params / total_params:.2f}x")
            print(f"  Large vs Small: {large_total_params / small_total_params:.2f}x")
            
        except ImportError:
            # If large model is not available, just compare with small
            print("\nComparison with Small Model:")
            print(f"  Small Model Total Parameters: {small_total_params:,}")
            print(f"  Medium Model Total Parameters: {total_params:,}")
            print(f"  Scale Factor: {total_params / small_total_params:.2f}x")
            
        # Print component-wise comparison with small model
        print("\nComponent-wise Comparisons (Medium vs Small):")
        print(f"  1) Feature Extraction Network: {count_parameters(feature_network) / count_parameters(small_feature_network):.2f}x")
        print(f"  2) Phase Correlation Tensor:  {count_parameters(tensor_computer) / count_parameters(small_tensor_computer):.2f}x")
        print(f"  3) Dimension Adapter:         {count_parameters(dim_adapter) / count_parameters(small_dim_adapter):.2f}x")
        print(f"  4) Manifold Learning Module:  {count_parameters(manifold_module) / count_parameters(small_manifold_module):.2f}x")
        print(f"  5) Point Cloud Generator:     {count_parameters(point_cloud_generator) / count_parameters(small_point_cloud_generator):.2f}x")
        print(f"  6) Topological Feature Ext.:  {count_parameters(topo_module) / count_parameters(small_topo_module):.2f}x")
        print(f"  7) Classification Network:    {count_parameters(classifier) / count_parameters(small_classifier):.2f}x")
    
    except ImportError:
        print("\nCouldn't import small model for comparison. Make sure 'model.py' is available.")
        
    # Print parameter distribution
    print("\nParameter Distribution (% of total):")
    print(f"  1) Feature Extraction Network: {count_parameters(feature_network) / total_params * 100:.1f}%")
    print(f"  2) Phase Correlation Tensor:  {count_parameters(tensor_computer) / total_params * 100:.1f}%")
    print(f"  3) Dimension Adapter:         {count_parameters(dim_adapter) / total_params * 100:.1f}%")
    print(f"  4) Manifold Learning Module:  {count_parameters(manifold_module) / total_params * 100:.1f}%")
    print(f"  5) Point Cloud Generator:     {count_parameters(point_cloud_generator) / total_params * 100:.1f}%")
    print(f"  6) Topological Feature Ext.:  {count_parameters(topo_module) / total_params * 100:.1f}%")
    print(f"  7) Classification Network:    {count_parameters(classifier) / total_params * 100:.1f}%")
    
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
        
        # Full batch size estimation (assuming 16GB GPU)
        gpu_memory_gb = 16
        max_batch_size_estimate = math.floor(gpu_memory_gb * 1e9 / estimated_memory_bytes)
        print(f"Rough Maximum Batch Size (16GB GPU): {max_batch_size_estimate}")
        
    except ImportError:
        pass


if __name__ == "__main__":
    main()