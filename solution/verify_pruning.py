import os
import sys
import torch
from pathlib import Path

sys.path.append(str(Path.cwd()))

from src.model import IndicSLM, IndicSLMConfig
from src.pruning import PruningConfig, ModelPruner

print('Creating model...')
config = IndicSLMConfig(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2
)
model = IndicSLM(config)

print('Counting parameters before pruning...')
total_params = sum(p.numel() for p in model.parameters())
non_zero_before = sum((p != 0).sum().item() for p in model.parameters())
print(f'Total parameters: {total_params}')
print(f'Non-zero before pruning: {non_zero_before}')

print('Creating pruning config and pruner...')
pruning_config = PruningConfig(
    enabled=True,
    method='magnitude',
    sparsity=0.3,
    schedule='one-shot'
)
pruner = ModelPruner(model, pruning_config)

print('Applying pruning...')
sparsity = pruner.apply_pruning(epoch=0)
print(f'Applied pruning with sparsity: {sparsity:.4f}')

print('Counting parameters after pruning...')
non_zero_after = sum((p != 0).sum().item() for p in model.parameters())
print(f'Non-zero after pruning: {non_zero_after}')
print(f'Parameters pruned: {non_zero_before - non_zero_after}')
print(f'Achieved sparsity: {1 - non_zero_after/total_params:.4f}')

print('Pruning verification complete!')
