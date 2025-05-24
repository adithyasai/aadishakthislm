"""
Basic test script to verify that our quantization functionality is working correctly.
This script performs weight-only quantization and shows results.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Import from simple_quantization since it's the most compatible
from src.simple_quantization import apply_weight_only_quantization
from src.model import IndicSLM, IndicSLMConfig

def run_simple_test():
    """Run a simplified quantization test that should work on any PyTorch version."""
    print("\n" + "="*60)
    print("TESTING SLM QUANTIZATION")
    print("="*60)
    
    # Print environment info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create a small model for testing
    print("\nCreating test model...")
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,  # Small size for quick testing
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64
    )
    
    # Initialize model
    model = IndicSLM(config)
    model.eval()
    
    # Get model parameter count and size
    param_count = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Original model parameters: {param_count:,}")
    print(f"Original model size: {model_size_mb:.2f} MB")
    
    # Create some test inputs
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Run inference with original model
    print("\nRunning inference with original model...")
    with torch.no_grad():
        start_time = torch.time.HRTimer() if hasattr(torch.time, 'HRTimer') else None
        if start_time:
            start_time.reset()
        else:
            import time
            start_time = time.time()
            
        original_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        if hasattr(torch.time, 'HRTimer'):
            original_time_ms = start_time.elapsed() * 1000
        else:
            original_time_ms = (time.time() - start_time) * 1000
    
    print(f"Original model inference time: {original_time_ms:.2f} ms")
    print(f"Original output shape: {original_outputs['logits'].shape}")
    
    # Test 8-bit weight-only quantization
    print("\nApplying 8-bit weight-only quantization...")
    try:
        quantized_model_8bit = apply_weight_only_quantization(model, bits=8)
        
        # Verify the model was properly quantized
        print(f"Model quantized: {quantized_model_8bit.is_quantized}")
        print(f"Quantization type: {quantized_model_8bit.quantization_type}")
        print(f"Quantization bits: {quantized_model_8bit.quantization_bits}")
        
        # Run inference with 8-bit model
        print("\nRunning inference with 8-bit quantized model...")
        with torch.no_grad():
            if hasattr(torch.time, 'HRTimer'):
                start_time.reset()
            else:
                start_time = time.time()
                
            quantized_outputs_8bit = quantized_model_8bit(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if hasattr(torch.time, 'HRTimer'):
                quantized_time_ms = start_time.elapsed() * 1000
            else:
                quantized_time_ms = (time.time() - start_time) * 1000
        
        print(f"8-bit model inference time: {quantized_time_ms:.2f} ms")
        print(f"8-bit output shape: {quantized_outputs_8bit['logits'].shape}")
        
        # Calculate output difference
        output_diff = (original_outputs['logits'] - quantized_outputs_8bit['logits']).abs().mean().item()
        print(f"Mean absolute difference in outputs: {output_diff:.6f}")
        
        # Test 4-bit weight-only quantization
        print("\nApplying 4-bit weight-only quantization...")
        quantized_model_4bit = apply_weight_only_quantization(model, bits=4)
        
        # Run inference with 4-bit model
        with torch.no_grad():
            quantized_outputs_4bit = quantized_model_4bit(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Calculate output difference for 4-bit
        output_diff_4bit = (original_outputs['logits'] - quantized_outputs_4bit['logits']).abs().mean().item()
        print(f"4-bit mean absolute difference: {output_diff_4bit:.6f}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_simple_test()
    sys.exit(0 if success else 1)
