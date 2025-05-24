"""
Comprehensive validation test for model pruning functionality.
Tests both one-shot and gradual pruning with different sparsity levels.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pruning_validation.log")
    ]
)
logger = logging.getLogger(__name__)

from src.model import IndicSLM, IndicSLMConfig
from src.pruning import PruningConfig, ModelPruner, apply_magnitude_pruning


def test_one_shot_pruning():
    """Test one-shot magnitude-based pruning"""
    logger.info("\n===== Testing one-shot magnitude-based pruning =====")
    
    # Create a test model
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    model = IndicSLM(config)
    
    # Count parameters before pruning
    total_params = sum(p.numel() for p in model.parameters())
    non_zero_before = sum((p != 0).sum().item() for p in model.parameters())
    
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Non-zero parameters before pruning: {non_zero_before}")
    
    # Create pruning configuration
    pruning_config = PruningConfig(
        enabled=True,
        method="magnitude",
        sparsity=0.3,
        schedule="one-shot"
    )
    
    # Create pruner and apply pruning
    pruner = ModelPruner(model, pruning_config)
    sparsity = pruner.apply_pruning(epoch=0)
    
    # Verify pruning
    non_zero_after = sum((p != 0).sum().item() for p in model.parameters())
    params_pruned = non_zero_before - non_zero_after
    achieved_sparsity = 1 - non_zero_after / total_params
    
    logger.info(f"Applied pruning with target sparsity: 0.3")
    logger.info(f"Achieved sparsity: {achieved_sparsity:.4f}")
    logger.info(f"Parameters pruned: {params_pruned} out of {total_params}")
    logger.info(f"Non-zero parameters after pruning: {non_zero_after}")
    
    # Check if the pruning was applied correctly
    if abs(achieved_sparsity - 0.3) < 0.05:
        logger.info("One-shot pruning test: PASSED ✓")
        return True
    else:
        logger.error("One-shot pruning test: FAILED ✗")
        logger.error(f"Expected sparsity around 0.3, got {achieved_sparsity:.4f}")
        return False


def test_gradual_pruning():
    """Test gradual pruning with increasing sparsity"""
    logger.info("\n===== Testing gradual pruning =====")
    
    # Create a test model
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    model = IndicSLM(config)
    
    # Create pruning configuration for gradual pruning
    pruning_config = PruningConfig(
        enabled=True,
        method="magnitude",
        sparsity=0.5,
        schedule="gradual",
        start_epoch=1,
        end_epoch=5,
        frequency=1
    )
    
    # Create pruner
    pruner = ModelPruner(model, pruning_config)
    
    # Track sparsity over epochs
    initial_params = sum(p.numel() for p in model.parameters())
    initial_non_zero = sum((p != 0).sum().item() for p in model.parameters())
    
    logger.info(f"Total parameters: {initial_params}")
    logger.info(f"Initial non-zero parameters: {initial_non_zero}")
    
    # Simulate gradual pruning over epochs
    sparsities = []
    for epoch in range(6):
        if pruner.should_prune_on_epoch(epoch):
            sparsity = pruner.apply_pruning(epoch)
            non_zero = sum((p != 0).sum().item() for p in model.parameters())
            logger.info(f"Epoch {epoch}: Applied pruning with sparsity {sparsity:.4f}")
            logger.info(f"Epoch {epoch}: Non-zero parameters: {non_zero}")
        else:
            non_zero = sum((p != 0).sum().item() for p in model.parameters())
            sparsity = 1 - non_zero / initial_params
            logger.info(f"Epoch {epoch}: No pruning applied. Current sparsity {sparsity:.4f}")
            logger.info(f"Epoch {epoch}: Non-zero parameters: {non_zero}")
        
        sparsities.append(sparsity)
    
    # Verify that sparsity increases gradually
    is_increasing = all(sparsities[i] <= sparsities[i+1] for i in range(len(sparsities)-1))
    final_sparsity = 1 - sum((p != 0).sum().item() for p in model.parameters()) / initial_params
    
    logger.info(f"Sparsities over epochs: {[f'{s:.4f}' for s in sparsities]}")
    logger.info(f"Final sparsity: {final_sparsity:.4f}")
    
    # Check if the pruning behaved as expected
    if is_increasing and abs(final_sparsity - 0.5) < 0.05:
        logger.info("Gradual pruning test: PASSED ✓")
        return True
    else:
        logger.error("Gradual pruning test: FAILED ✗")
        if not is_increasing:
            logger.error("Sparsity did not increase monotonically over epochs")
        if abs(final_sparsity - 0.5) >= 0.05:
            logger.error(f"Final sparsity {final_sparsity:.4f} differs too much from target 0.5")
        return False


def test_utility_functions():
    """Test utility functions in the pruning module"""
    logger.info("\n===== Testing pruning utility functions =====")
    
    # Test apply_magnitude_pruning function
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64
    )
    
    model = IndicSLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    initial_non_zero = sum((p != 0).sum().item() for p in model.parameters())
    
    # Apply utility function
    logger.info("Testing apply_magnitude_pruning utility function")
    pruned_model = apply_magnitude_pruning(model, sparsity=0.4)
    
    non_zero_after = sum((p != 0).sum().item() for p in pruned_model.parameters())
    achieved_sparsity = 1 - non_zero_after / total_params
    
    logger.info(f"Applied sparsity: {achieved_sparsity:.4f} (target: 0.4)")
    
    if abs(achieved_sparsity - 0.4) < 0.05:
        logger.info("Utility function test: PASSED ✓")
        return True
    else:
        logger.error("Utility function test: FAILED ✗")
        logger.error(f"Expected sparsity around 0.4, got {achieved_sparsity:.4f}")
        return False


if __name__ == "__main__":
    logger.info("Starting pruning validation tests...")
    
    # Run all tests
    one_shot_result = test_one_shot_pruning()
    gradual_result = test_gradual_pruning()
    utility_result = test_utility_functions()
    
    # Summarize results
    logger.info("\n===== Test Results =====")
    logger.info(f"One-shot pruning: {'PASSED ✓' if one_shot_result else 'FAILED ✗'}")
    logger.info(f"Gradual pruning: {'PASSED ✓' if gradual_result else 'FAILED ✗'}")
    logger.info(f"Utility functions: {'PASSED ✓' if utility_result else 'FAILED ✗'}")
    
    if one_shot_result and gradual_result and utility_result:
        logger.info("\n=== All pruning tests PASSED! ===")
        logger.info("Model pruning implementation is working correctly.")
        logger.info("Task 4.2 can be marked as completed.")
    else:
        logger.error("\n=== Some pruning tests FAILED! ===")
        logger.error("Please fix the issues before marking Task 4.2 as completed.")
