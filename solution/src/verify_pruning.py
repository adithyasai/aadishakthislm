"""
Minimal script to test pruning functionality
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.model import IndicSLM, IndicSLMConfig
from src.pruning import apply_magnitude_pruning

# Create a simple model
config = IndicSLMConfig(
    vocab_size=1000,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=128
)

model = IndicSLM(config)

# Print initial parameter stats
total_params = sum(p.numel() for p in model.parameters())
# Output to both console and file
with open("pruning_verification.log", "w") as log_file:
    log_message = f"Original model has {total_params} parameters"
    print(log_message)
    log_file.write(log_message + "\n")

# Apply magnitude pruning
pruned_model = apply_magnitude_pruning(model, sparsity=0.3)

# Count non-zero parameters after pruning
non_zero_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
zero_params = total_params - non_zero_params
sparsity_achieved = zero_params / total_params

with open("pruning_verification.log", "a") as log_file:
    log_message = f"Pruned model has {non_zero_params} non-zero parameters"
    print(log_message)
    log_file.write(log_message + "\n")
    
    log_message = f"Achieved sparsity: {sparsity_achieved:.4f} ({zero_params} zeros of {total_params} total)"
    print(log_message)
    log_file.write(log_message + "\n")

# Test inference
input_ids = torch.randint(0, 1000, (1, 16))
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    output = pruned_model(input_ids=input_ids, attention_mask=attention_mask)

with open("pruning_verification.log", "a") as log_file:
    log_message = f"Inference successful with output shape: {output['logits'].shape}"
    print(log_message)
    log_file.write(log_message + "\n")
    
    log_message = "Pruning test completed successfully!"
    print(log_message)
    log_file.write(log_message + "\n")
