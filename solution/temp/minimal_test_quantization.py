"""
Minimal test script for quantization functionality.
"""

import os
import torch
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model modules
from model import IndicSLM, IndicSLMConfig

def test_simple_quantization():
    """Test basic quantization functionality without external dependencies"""
    logger.info("Testing minimal weight-only quantization...")
    
    # Create a small model for testing
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128
    )
    
    # Initialize model
    model = IndicSLM(config)
    model.eval()
    
    # Verify model can run a forward pass
    batch_size = 2
    seq_length = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logger.info(f"Original model output shape: {output['logits'].shape}")
    
    # Simple weight-only quantization for testing
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get weight tensor
            weight = module.weight.data
            
            # Quantize to 8-bit
            scale = (weight.max() - weight.min()) / 255
            zero_point = (weight.min() / scale).round() if scale != 0 else 0
            
            # Quantize
            quantized = (weight / scale).round() + zero_point if scale != 0 else weight
            quantized = torch.clamp(quantized, 0, 255)
            
            # Dequantize
            dequantized = (quantized - zero_point) * scale
            
            # Replace weights
            module.weight.data = dequantized
    
    # Test quantized model
    quantized_output = model(input_ids=input_ids, attention_mask=attention_mask)
    logger.info(f"Quantized model output shape: {quantized_output['logits'].shape}")
    
    # Calculate difference
    diff = (output['logits'] - quantized_output['logits']).abs().mean().item()
    logger.info(f"Mean absolute difference: {diff:.6f}")
    
    logger.info("Simple quantization test completed successfully!")
    return True

if __name__ == "__main__":
    test_simple_quantization()
