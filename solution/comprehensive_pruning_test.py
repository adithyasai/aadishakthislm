"""
Comprehensive pruning verification script with detailed metrics
"""

import os
import sys
from pathlib import Path
import torch
import logging

# Configure logging - log only to console for clear output
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent))

try:
    from src.model import IndicSLM, IndicSLMConfig
    from src.pruning import PruningConfig, ModelPruner
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def count_parameters(model, prunable_only=False):
    """Count total and non-zero parameters in the model"""
    total_params = 0
    total_prunable = 0
    non_zero = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        # Check if this is a prunable parameter (weights of Linear or Conv2d layers)
        is_prunable = 'weight' in name and not any(x in name for x in ['.bias', 'embedding', 'layer_norm', 'layernorm'])
        
        if is_prunable:
            total_prunable += param_count
            
        non_zero_count = (param != 0).sum().item()
        non_zero += non_zero_count
        
        if is_prunable:
            sparsity = 1.0 - (non_zero_count / param_count)
            logger.info(f"{name}: {param_count} params, {non_zero_count} non-zero, {sparsity:.4f} sparsity")
    
    return {
        "total_params": total_params,
        "total_prunable": total_prunable,
        "non_zero": non_zero,
        "prunable_sparsity": 1.0 - (non_zero / total_prunable) if total_prunable > 0 else 0,
        "overall_sparsity": 1.0 - (non_zero / total_params) if total_params > 0 else 0
    }

def main():
    logger.info("=== Detailed Pruning Verification ===")
    
    # Create a model
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    model = IndicSLM(config)
    logger.info("Created model for testing")
    
    # Count parameters before pruning
    logger.info("Counting parameters before pruning:")
    before_stats = count_parameters(model)
    logger.info(f"Total parameters: {before_stats['total_params']}")
    logger.info(f"Prunable parameters: {before_stats['total_prunable']}")
    logger.info(f"Non-zero parameters: {before_stats['non_zero']}")
    
    # Setup pruning
    pruning_config = PruningConfig(
        enabled=True,
        method="magnitude",
        sparsity=0.3,
        schedule="one-shot"
    )
    
    pruner = ModelPruner(model, pruning_config)
    
    # Apply pruning
    logger.info("\nApplying pruning with target sparsity 0.3...")
    sparsity = pruner.apply_pruning(epoch=0)
    logger.info(f"Reported sparsity from pruner: {sparsity:.4f}")
    
    # Count parameters after pruning
    logger.info("\nCounting parameters after pruning:")
    after_stats = count_parameters(model)
    logger.info(f"Total parameters: {after_stats['total_params']}")
    logger.info(f"Prunable parameters: {after_stats['total_prunable']}")
    logger.info(f"Non-zero parameters: {after_stats['non_zero']}")
    logger.info(f"Prunable parameters sparsity: {after_stats['prunable_sparsity']:.4f}")
    logger.info(f"Overall model sparsity: {after_stats['overall_sparsity']:.4f}")
    
    # Get pruning statistics from the pruner
    stats = pruner.get_pruning_statistics()
    logger.info("\nPruning statistics from pruner:")
    logger.info(f"Target sparsity: {stats['target_sparsity']:.4f}")
    logger.info(f"Current sparsity: {stats['current_sparsity']:.4f}")
    
    # Test inference with pruned model
    logger.info("\nTesting inference with pruned model...")
    inputs = torch.randint(0, 1000, (1, 8))
    try:
        with torch.no_grad():
            outputs = model(inputs)
        logger.info(f"Inference successful with output shape: {outputs.shape}")
        logger.info("Pruning verification completed successfully!")
    except Exception as e:
        logger.error(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
