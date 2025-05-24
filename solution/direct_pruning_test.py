"""
Direct printing of pruning results for clear output
"""
import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def print_bold(text):
    """Print text in bold for emphasis"""
    print(f"\033[1m{text}\033[0m")

def main():
    print_bold("===== Pruning Verification =====")
    
    # Import after sys.path is set
    try:
        from src.model import IndicSLM, IndicSLMConfig
        from src.pruning import PruningConfig, ModelPruner
        print("Successfully imported required modules")
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return
    
    # Create a small model
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2
    )
    
    print("\nCreating test model...")
    model = IndicSLM(config)
    
    # Count parameters before pruning
    total_params = sum(p.numel() for p in model.parameters())
    prunable_params = sum(p.weight.numel() for n, p in model.named_modules() 
                        if isinstance(p, (torch.nn.Linear, torch.nn.Conv2d)))
    
    print(f"Total parameters: {total_params}")
    print(f"Prunable parameters: {prunable_params}")
    print(f"Non-prunable parameters: {total_params - prunable_params}")
    
    # Setup pruning
    print("\nSetting up pruning...")
    pruning_config = PruningConfig(
        enabled=True,
        method="magnitude",
        sparsity=0.3,
        schedule="one-shot"
    )
    
    pruner = ModelPruner(model, pruning_config)
    
    # Apply pruning
    print("Applying pruning with target sparsity 0.3...")
    try:
        sparsity = pruner.apply_pruning(epoch=0)
        print(f"Reported sparsity from pruner: {sparsity:.4f}")
    except Exception as e:
        print(f"Error during pruning: {e}")
        return

    # Count parameters after pruning
    prunable_zeros = 0
    prunable_total = 0
    
    # Count zeros in prunable parameters
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            weight = module.weight
            zeros = (weight == 0).sum().item()
            total = weight.numel()
            
            prunable_zeros += zeros
            prunable_total += total
            
            # Print per-layer stats
            print(f"Layer {name}: {zeros}/{total} zeros ({zeros/total:.4f} sparsity)")
    
    # Overall model stats
    non_zero_after = sum((p != 0).sum().item() for p in model.parameters())
    zeros_after = total_params - non_zero_after
    
    print_bold("\nOverall Results:")
    print(f"Prunable parameters: {prunable_total}")
    print(f"Zeros in prunable parameters: {prunable_zeros}")
    print(f"Prunable sparsity: {prunable_zeros/prunable_total:.4f}")
    print(f"Overall model sparsity: {zeros_after/total_params:.4f}")
    
    # Test inference
    print("\nTesting inference with pruned model...")
    inputs = torch.randint(0, 1000, (1, 8))
    try:
        with torch.no_grad():
            outputs = model(inputs)
        print(f"Inference successful with output shape: {outputs.shape}")
        print_bold("Pruning verification completed successfully!")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
    print("")  # Add a blank line at the end
