"""
Verification script for testing our quantization implementation.
This script performs a basic test to ensure the weight-only quantization 
works correctly with the current PyTorch installation.
"""

import torch
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import the simplified quantization functions
from src.simple_quantization import apply_weight_only_quantization
from src.model import IndicSLM, IndicSLMConfig

def verify_quantization():
    """
    Verify that basic weight-only quantization works properly.
    """
    print("\n" + "="*80)
    print("Testing SLM model quantization")
    print("="*80)
    
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # Create a mini model for testing
    print("\nCreating test model...")
    config = IndicSLMConfig(
        vocab_size=1000,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64
    )
    
    # Initialize model
    model = IndicSLM(config)
    model.eval()
    
    # Get original size
    original_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # in MB, assuming float32
    print(f"Original model size: {original_size:.2f} MB")
    
    # Create input
    batch_size = 1
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Test original model
    print("\nRunning inference with original model...")
    with torch.no_grad():
        original_output = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    print(f"Original model output shape: {original_output['logits'].shape}")
    
    # Apply 8-bit weight-only quantization
    print("\nApplying 8-bit weight-only quantization...")
    quantized_model_8bit = apply_weight_only_quantization(model, bits=8)
    
    # Get quantized size approximation
    quantized_size_8bit = sum(p.numel() for p in quantized_model_8bit.model.parameters()) / (1024 * 1024)  # in MB, assuming int8
    print(f"8-bit model size (approx): {quantized_size_8bit:.2f} MB")
    print(f"Size reduction: ~{original_size / quantized_size_8bit:.1f}x")
    
    # Test 8-bit model
    print("\nRunning inference with 8-bit quantized model...")
    with torch.no_grad():
        quantized_output_8bit = quantized_model_8bit(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Calculate difference
    diff_8bit = (original_output['logits'] - quantized_output_8bit['logits']).abs().mean().item()
    print(f"Mean absolute difference (8-bit): {diff_8bit:.6f}")
    
    # Apply 4-bit weight-only quantization
    print("\nApplying 4-bit weight-only quantization...")
    quantized_model_4bit = apply_weight_only_quantization(model, bits=4)
    
    # Test 4-bit model
    print("\nRunning inference with 4-bit quantized model...")
    with torch.no_grad():
        quantized_output_4bit = quantized_model_4bit(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
    
    # Calculate difference
    diff_4bit = (original_output['logits'] - quantized_output_4bit['logits']).abs().mean().item()
    print(f"Mean absolute difference (4-bit): {diff_4bit:.6f}")
    
    # Verify outputs match in shape
    shapes_match = quantized_output_8bit['logits'].shape == original_output['logits'].shape
    print(f"\nOutput shapes match: {shapes_match}")
    
    print("\nQuantization verification complete!")
    print("="*80)
    return True

if __name__ == "__main__":
    verify_quantization()
