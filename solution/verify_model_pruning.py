"""
Pruning Verification Tool

This script shows the actual state of model parameters after pruning,
including sparsity by layer and a summary of pruned weights.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.model import IndicSLM, IndicSLMConfig

def verify_pruned_model(model_path=None, sparsity=None):
    """
    Verify the sparsity of a trained model.
    
    Args:
        model_path: Path to the model checkpoint or directory
        sparsity: Expected sparsity level to compare against
    """
    print("\n" + "=" * 60)
    print("PRUNING VERIFICATION TOOL")
    print("=" * 60)
    
    if model_path:
        print(f"\nLoading model from: {model_path}")
        # Load the model here based on path
        # This requires more complex logic based on how your model is saved
        # For this demo we'll create a test model
        model = None
    else:
        print("\nNo model path provided. Creating a test model...")
        config = IndicSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2
        )
        model = IndicSLM(config)
        
        # If sparsity is provided, apply pruning
        if sparsity:
            from src.pruning import PruningConfig, ModelPruner
            print(f"\nApplying {sparsity:.2f} pruning to the test model...")
            pruning_config = PruningConfig(
                enabled=True,
                method="magnitude",
                sparsity=sparsity,
                schedule="one-shot"
            )
            pruner = ModelPruner(model, pruning_config)
            pruner.apply_pruning(epoch=0)
    
    # Count parameters and sparsity
    if model:
        total_params = 0
        total_zeros = 0
        
        print("\nLayer-wise sparsity:")
        print("-" * 40)
        print(f"{'Layer Name':30} | {'Sparsity':10} | {'Shape':15}")
        print("-" * 40)
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Only check weight matrices
                param_total = param.numel()
                param_zeros = (param == 0).sum().item()
                param_sparsity = param_zeros / param_total
                
                total_params += param_total
                total_zeros += param_zeros
                
                print(f"{name:30} | {param_sparsity:8.4f} | {str(list(param.shape)):15}")
        
        # Print summary
        overall_sparsity = total_zeros / total_params if total_params > 0 else 0
        print("\nSummary:")
        print(f"Total parameters: {total_params}")
        print(f"Zero parameters:  {total_zeros}")
        print(f"Overall sparsity: {overall_sparsity:.4f}")
        
        if sparsity:
            print(f"Expected sparsity: {sparsity:.4f}")
            if abs(overall_sparsity - sparsity) < 0.05:
                print("\nVerification: PASSED ✓")
                print(f"The model has been properly pruned to ~{sparsity:.2f} sparsity.")
            else:
                print("\nVerification: WARNING ⚠")
                print(f"Actual sparsity ({overall_sparsity:.4f}) differs from expected ({sparsity:.4f}).")
    else:
        print("\nNo model available for verification.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify model pruning")
    parser.add_argument("--model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("--sparsity", type=float, default=0.3, help="Expected sparsity level")
    
    args = parser.parse_args()
    verify_pruned_model(args.model_path, args.sparsity)
