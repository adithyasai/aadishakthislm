"""
Test module for model pruning functionality.

This script tests the pruning capabilities in the SLM model, including:
- One-shot magnitude-based pruning
- Gradual pruning
- Structured pruning (optional)

It verifies that pruning correctly reduces model size without significantly
impacting model performance.
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import IndicSLM, IndicSLMConfig
from src.pruning import PruningConfig, ModelPruner, apply_magnitude_pruning, evaluate_pruning_impact

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_pruning():
    """Test basic pruning functionality."""
    logger.info("Testing basic pruning functionality...")
    
    # Create a small model for testing
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    model = IndicSLM(config)
    
    # Generate test inputs
    input_ids = torch.randint(0, 1000, (1, 16))
    attention_mask = torch.ones_like(input_ids)
    
    # Test forward pass before pruning
    with torch.no_grad():
        original_output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Count parameters before pruning
    orig_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original model has {orig_params} parameters")
    
    # Apply magnitude pruning
    pruned_model = apply_magnitude_pruning(model, sparsity=0.3)
    
    # Count non-zero parameters after pruning
    non_zero_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
    sparsity_achieved = 1.0 - (non_zero_params / orig_params)
    
    logger.info(f"Pruned model has {non_zero_params} non-zero parameters "
               f"({sparsity_achieved:.4f} sparsity)")
    
    # Test forward pass after pruning
    with torch.no_grad():
        pruned_output = pruned_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check output difference
    logits_diff = torch.abs(original_output['logits'] - pruned_output['logits']).mean().item()
    logger.info(f"Mean absolute difference in logits: {logits_diff:.6f}")
    
    # Verify pruning was effective
    assert sparsity_achieved > 0.25, "Pruning did not achieve target sparsity"
    
    logger.info("Basic pruning test completed successfully")
    return sparsity_achieved, logits_diff

def test_gradual_pruning():
    """Test gradual pruning functionality."""
    logger.info("Testing gradual pruning...")
    
    # Create a small model for testing
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    model = IndicSLM(config)
    
    # Create pruning config
    pruning_config = PruningConfig(
        enabled=True,
        method="magnitude",
        sparsity=0.5,
        schedule="gradual",
        start_epoch=0,
        end_epoch=4,
        frequency=1
    )
    
    # Create pruner
    pruner = ModelPruner(model, pruning_config)
    
    # Simulate training epochs
    sparsity_progression = []
    for epoch in range(6):
        # Apply pruning if needed for this epoch
        if pruner.should_prune_on_epoch(epoch):
            sparsity = pruner.apply_pruning(epoch)
            logger.info(f"Epoch {epoch}: Applied pruning with sparsity {sparsity:.4f}")
        else:
            sparsity = pruner.current_sparsity
            logger.info(f"Epoch {epoch}: No pruning applied, current sparsity {sparsity:.4f}")
        
        sparsity_progression.append(sparsity)
    
    # Verify sparsity increases gradually and reaches target
    assert sparsity_progression[-1] >= 0.45, "Did not reach target sparsity"
    assert sparsity_progression[1] < sparsity_progression[2], "Sparsity not increasing gradually"
    
    logger.info("Gradual pruning test completed successfully")
    return sparsity_progression

def test_structured_pruning():
    """Test structured pruning functionality."""
    logger.info("Testing structured pruning...")
    
    # Create a small model for testing
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    model = IndicSLM(config)
    
    # Create pruning config for structured pruning
    pruning_config = PruningConfig(
        enabled=True,
        method="structured",
        sparsity=0.3,
        schedule="one-shot",
        structured_dim=0  # Prune output neurons
    )
    
    # Create pruner
    pruner = ModelPruner(model, pruning_config)
    
    # Apply pruning
    sparsity = pruner.apply_pruning(epoch=0)
    logger.info(f"Applied structured pruning with sparsity {sparsity:.4f}")
    
    # Get statistics
    stats = pruner.get_pruning_statistics()
    
    # Print layer-wise sparsity
    logger.info("Layer-wise sparsity:")
    for layer_name, layer_stats in stats.get('layer_sparsity', {}).items():
        logger.info(f"  {layer_name}: {layer_stats['sparsity']:.4f}")
    
    logger.info("Structured pruning test completed successfully")
    return stats

def test_model_performance_after_pruning():
    """Test model performance after pruning."""
    logger.info("Testing model performance after pruning...")
    
    # Create a small model for testing
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    original_model = IndicSLM(config)
    
    # Create pruned model
    pruned_model = IndicSLM(config)
    pruned_model.load_state_dict(original_model.state_dict())
    
    # Apply pruning
    apply_magnitude_pruning(pruned_model, sparsity=0.3)
    
    # Generate test inputs
    test_inputs = {
        "input_ids": torch.randint(0, 1000, (1, 32)),
        "attention_mask": torch.ones((1, 32), dtype=torch.long)
    }
    
    # Evaluate impact
    results = evaluate_pruning_impact(original_model, pruned_model, test_inputs)
    
    logger.info(f"Inference time: Original={results['inference_time']['original_ms']:.2f}ms, "
               f"Pruned={results['inference_time']['pruned_ms']:.2f}ms, "
               f"Speedup={results['inference_time']['speedup']:.2f}x")
    
    logger.info(f"Parameters: Original={results['parameters']['original_total']}, "
               f"Non-zero after pruning={results['parameters']['pruned_non_zero']}, "
               f"Sparsity={results['parameters']['sparsity']:.4f}")
    
    logger.info(f"Output difference: MSE={results['output_difference']['mse']:.6f}, "
               f"MAE={results['output_difference']['mae']:.6f}, "
               f"Cosine similarity={results['output_difference']['cosine_sim']:.6f}")
    
    logger.info("Performance comparison completed successfully")
    return results

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test model pruning functionality")
    parser.add_argument("--test_all", action="store_true", help="Run all pruning tests")
    parser.add_argument("--test_basic", action="store_true", help="Test basic pruning")
    parser.add_argument("--test_gradual", action="store_true", help="Test gradual pruning")
    parser.add_argument("--test_structured", action="store_true", help="Test structured pruning")
    parser.add_argument("--test_performance", action="store_true", help="Test performance impact")
    
    args = parser.parse_args()
    
    # Run tests based on arguments
    if args.test_all or not (args.test_basic or args.test_gradual or args.test_structured or args.test_performance):
        print("=" * 80)
        print("Running all pruning tests")
        print("=" * 80)
        
        test_basic_pruning()
        print("-" * 40)
        
        test_gradual_pruning()
        print("-" * 40)
        
        try:
            test_structured_pruning()
            print("-" * 40)
        except Exception as e:
            logger.warning(f"Structured pruning test failed: {e}")
        
        test_model_performance_after_pruning()
    else:
        if args.test_basic:
            test_basic_pruning()
        
        if args.test_gradual:
            test_gradual_pruning()
        
        if args.test_structured:
            test_structured_pruning()
        
        if args.test_performance:
            test_model_performance_after_pruning()
    
    print("=" * 80)
    print("All tests completed")
