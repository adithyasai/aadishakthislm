import torch
import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import required modules
from src.model import IndicSLM, IndicSLMConfig
from src.pruning import apply_magnitude_pruning

# Create a small model for testing
config = IndicSLMConfig(
    vocab_size=1000,
    hidden_size=32,
    num_hidden_layers=1,
    num_attention_heads=1,
    intermediate_size=64
)

print("Creating test model...")
model = IndicSLM(config)

# Count parameters before pruning
total_params = sum(p.numel() for p in model.parameters())
print(f"Original model has {total_params} parameters")

# Apply magnitude pruning
print("Applying pruning with 0.3 sparsity...")
pruned_model = apply_magnitude_pruning(model, sparsity=0.3)

# Count non-zero parameters after pruning
non_zero_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
zero_params = total_params - non_zero_params
sparsity_achieved = zero_params / total_params

print(f"Pruned model has {non_zero_params} non-zero parameters")
print(f"Achieved sparsity: {sparsity_achieved:.4f} ({zero_params} zeros of {total_params} total)")

# Test inference
print("Testing inference with pruned model...")
input_ids = torch.randint(0, 1000, (1, 8))
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    output = pruned_model(input_ids=input_ids, attention_mask=attention_mask)

print(f"Inference successful with output shape: {output['logits'].shape}")
print("Pruning test completed successfully!")
