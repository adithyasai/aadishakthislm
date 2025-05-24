"""
Minimal test script for model pruning
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pruning_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_pruning():
    """Test basic pruning functionality"""
    from src.model import IndicSLM, IndicSLMConfig
    from src.pruning import apply_magnitude_pruning
    
    logger.info("Creating test model...")
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64
    )
    
    model = IndicSLM(config)
    
    # Count parameters before pruning
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original model has {total_params} parameters")
    
    # Apply magnitude pruning
    logger.info("Applying pruning with 0.3 sparsity...")
    pruned_model = apply_magnitude_pruning(model, sparsity=0.3)
    
    # Count non-zero parameters after pruning
    non_zero_params = sum((p != 0).sum().item() for p in pruned_model.parameters())
    zero_params = total_params - non_zero_params
    sparsity_achieved = zero_params / total_params
    
    logger.info(f"Pruned model has {non_zero_params} non-zero parameters")
    logger.info(f"Achieved sparsity: {sparsity_achieved:.4f} ({zero_params} zeros of {total_params} total)")
    
    # Generate test inputs
    test_inputs = {
        "input_ids": torch.randint(0, 1000, (1, 8)),
        "attention_mask": torch.ones((1, 8), dtype=torch.long)
    }
    
    # Test inference
    logger.info("Testing inference with pruned model...")
    with torch.no_grad():
        output = pruned_model(**test_inputs)
    
    logger.info(f"Inference successful with output shape: {output['logits'].shape}")
    logger.info("Pruning test completed successfully!")
    
    # Return result
    return {
        "success": True,
        "sparsity": sparsity_achieved,
        "zero_params": zero_params,
        "total_params": total_params,
        "non_zero_params": non_zero_params
    }

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("Running model pruning test")
    logger.info("="*80)
    
    try:
        result = test_pruning()
        logger.info("="*80)
        if result["success"]:
            logger.info("PRUNING TEST PASSED!")
            logger.info(f"Target sparsity: 0.3, Achieved sparsity: {result['sparsity']:.4f}")
        else:
            logger.info("PRUNING TEST FAILED!")
    except Exception as e:
        logger.error(f"Error running pruning test: {str(e)}")
        logger.exception(e)
        logger.info("PRUNING TEST FAILED!")
    
    logger.info("="*80)
