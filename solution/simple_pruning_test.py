import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path.cwd()))

from src.model import IndicSLM, IndicSLMConfig
from src.pruning import PruningConfig, ModelPruner

print('Starting simple pruning test...')

# Create test model
config = IndicSLMConfig(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2
)
model = IndicSLM(config)

# Test basic pruning
print('Testing basic pruning functionality...')
total_params = sum(p.numel() for p in model.parameters())
non_zero_before = sum((p != 0).sum().item() for p in model.parameters())
print(f'Total parameters: {total_params}')
print(f'Non-zero parameters before pruning: {non_zero_before}')

# Create pruning config
pruning_config = PruningConfig(
    enabled=True,
    method='magnitude',
    sparsity=0.3,
    schedule='one-shot'
)

# Create pruner and apply pruning
pruner = ModelPruner(model, pruning_config)
sparsity = pruner.apply_pruning(epoch=0)
print(f'Applied pruning with sparsity: {sparsity:.4f}')

# Check results
non_zero_after = sum((p != 0).sum().item() for p in model.parameters())
print(f'Non-zero parameters after pruning: {non_zero_after}')
print(f'Parameters pruned: {non_zero_before - non_zero_after}')
achieved_sparsity = 1 - non_zero_after / total_params
print(f'Achieved sparsity: {achieved_sparsity:.4f}')

if abs(achieved_sparsity - 0.3) < 0.05:
    print('PASSED! Pruning is working correctly')
    print('Task 4.2 can be marked as completed')
else:
    print('FAILED! Pruning not working as expected')
    print(f'Expected ~0.3 sparsity, got {achieved_sparsity:.4f}')
