#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_model_parameters.py

This file imports the classes from `model.py` and prints the number of trainable
parameters in each sub-module, along with the total parameter count.
"""

from model import (
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
        input_dim=32,  # Must match the output of the manifold module
        hidden_dim=32,
        output_dim=16,
        max_edge_length=2.0,
        num_filtrations=10,
        max_dimension=0
    )
    classifier = ClassificationNetwork(
        manifold_dim=32,  # must match the manifold module
        topo_dim=16,      # must match the topo module
        feature_dim=32,
        hidden_dim=64,
        dropout=0.1
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


if __name__ == "__main__":
    main()
