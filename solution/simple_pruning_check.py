"""
Simple test to verify that pruning works with our fixed code.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.model import IndicSLM, IndicSLMConfig
from src.pruning import PruningConfig, ModelPruner

print("Starting pruning test...")

# Create a simple model
config = IndicSLMConfig(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2
)

model = IndicSLM(config)
print("Model created successfully.")

# Count parameters before pruning
total_params = sum(p.numel() for p in model.parameters())
non_zero_before = sum((p != 0).sum().item() for p in model.parameters())
print(f"Total parameters: {total_params}")
print(f"Non-zero parameters before pruning: {non_zero_before}")

# Configure and apply pruning
print("Setting up pruning configuration...")
pruning_config = PruningConfig(
    enabled=True,
    method="magnitude",
    sparsity=0.3,
    schedule="one-shot"
)

print("Creating pruner...")
pruner = ModelPruner(model, pruning_config)

try:
    print("Applying pruning...")
    sparsity = pruner.apply_pruning(epoch=0)
    print(f"Pruning applied successfully. Target sparsity: 0.3, achieved sparsity: {sparsity:.4f}")

    # Count parameters after pruning
    non_zero_after = sum((p != 0).sum().item() for p in model.parameters())
    achieved_sparsity = 1 - non_zero_after / total_params
    print(f"Non-zero parameters after pruning: {non_zero_after}")
    print(f"Parameters pruned: {non_zero_before - non_zero_after}")
    print(f"Actual achieved sparsity: {achieved_sparsity:.4f}")

    print("Pruning test completed successfully!")
except Exception as e:
    print(f"Error during pruning: {e}")
