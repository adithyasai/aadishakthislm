"""
Debug script for quantization functionality.
"""

import sys
import torch
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def debug_imports():
    print("\nDEBUG: Testing imports...")
    
    try:
        print("Importing model...")
        from src.model import IndicSLM, IndicSLMConfig
        print("Model import successful")
        
        print("Importing simple_quantization...")
        from src.simple_quantization import apply_weight_only_quantization, QuantizedModelWrapper
        print("Simple quantization import successful")
        
        print("Creating test config...")
        config = IndicSLMConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128
        )
        print("Test config created successfully")
        
        print("Creating model...")
        model = IndicSLM(config)
        print("Model created successfully")
        
        print("Testing model forward pass...")
        input_ids = torch.randint(0, 1000, (1, 8))
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Forward pass successful, output shape: {output['logits'].shape}")
        
        print("\nAttempting weight-only quantization...")
        try:
            quantized_model = apply_weight_only_quantization(model, bits=8)
            print("Quantization successful!")
            
            print("Testing quantized model...")
            with torch.no_grad():
                q_output = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"Quantized model forward pass successful, output shape: {q_output['logits'].shape}")
            
        except Exception as e:
            print(f"Quantization failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\nAll import tests completed")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    debug_imports()
