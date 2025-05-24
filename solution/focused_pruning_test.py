"""
Focused script to verify pruning effectiveness with printed output
"""
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.model import IndicSLM, IndicSLMConfig
from src.pruning import PruningConfig, ModelPruner

def main():
    print("===== Pruning Verification Script =====")
    
    # Create a small model
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2
    )
    
    model = IndicSLM(config)
    print("Created model for testing")
    
    # Count parameters before pruning
    total_params = sum(p.numel() for p in model.parameters())
    prunable_params = 0
    
    # Count prunable parameters (weights in Linear and Conv2d layers)
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            prunable_params += module.weight.numel()
    
    print(f"Total parameters: {total_params}")
    print(f"Prunable parameters: {prunable_params}")
    print(f"Non-prunable parameters: {total_params - prunable_params}")
    
    # Setup pruning
    pruning_config = PruningConfig(
        enabled=True,
        method="magnitude",
        sparsity=0.3,
        schedule="one-shot"
    )
    
    pruner = ModelPruner(model, pruning_config)
    
    # Apply pruning
    print("\nApplying pruning with target sparsity 0.3...")
    sparsity = pruner.apply_pruning(epoch=0)
    print(f"Reported sparsity from pruner: {sparsity:.4f}")
    
    # Count parameters after pruning
    non_zero_after = sum((p != 0).sum().item() for p in model.parameters())
    zeros_after = total_params - non_zero_after
    prunable_zeros = 0
    
    # Count zeros in pruned layers
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            prunable_zeros += (module.weight == 0).sum().item()
            
    # Calculate sparsities
    prunable_sparsity = prunable_zeros / prunable_params
    overall_sparsity = zeros_after / total_params
    
    print(f"\nAfter pruning:")
    print(f"Non-zero parameters: {non_zero_after}")
    print(f"Zero parameters: {zeros_after}")
    print(f"Zero parameters in prunable layers: {prunable_zeros}")
    print(f"Prunable parameters sparsity: {prunable_sparsity:.4f}")
    print(f"Overall model sparsity: {overall_sparsity:.4f}")
    
    # Test inference with pruned model
    print("\nTesting inference with pruned model...")
    inputs = torch.randint(0, 1000, (1, 8))
    try:
        with torch.no_grad():
            outputs = model(inputs)
        print(f"Inference successful with output shape: {outputs.shape}")
        print("Pruning verification completed successfully!")
    except Exception as e:
        print(f"Inference failed: {e}")
        
    print("\n===== Verification Complete =====")

if __name__ == "__main__":
    main()
