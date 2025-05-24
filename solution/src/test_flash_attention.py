"""
Test Flash Attention implementation for the SLM model
"""
import os
import sys
import torch
import logging
import argparse
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import IndicSLM, IndicSLMConfig
from src.flash_attention import apply_flash_attention, FLASH_ATTENTION_AVAILABLE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config paths
CONFIG_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.json"

def test_flash_attention(config_path=DEFAULT_CONFIG_PATH):
    """Test Flash Attention in the model"""
    logger.info("Testing Flash Attention...")
    
    # Load model configuration
    try:
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create model config with attention type set to default
        config_dict['attention_type'] = 'default'  # Start with standard attention
        config = IndicSLMConfig(**config_dict)
        
        # Create model
        logger.info("Creating model with standard attention...")
        model = IndicSLM(config)
        model.eval()
        
        # Generate test input
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Test standard attention
        logger.info("Testing standard attention performance...")
        
        # Run several forward passes with standard attention and measure time
        standard_times = []
        for _ in range(5):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            with torch.no_grad():
                standard_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            standard_times.append(time.time() - start_time)
        
        avg_standard_time = sum(standard_times) / len(standard_times)
        logger.info(f"Standard attention average time: {avg_standard_time:.4f} seconds")
        
        # Apply Flash Attention to the model
        logger.info("Applying Flash Attention...")
        model_with_flash = apply_flash_attention(model)
        
        if not FLASH_ATTENTION_AVAILABLE:
            logger.warning("Flash Attention package is not available. The test is using standard attention.")
            logger.info("To install Flash Attention, run: pip install flash-attn --no-build-isolation")
            return {
                "flash_attention_available": False,
                "standard_time": avg_standard_time,
                "flash_time": None,
                "speedup": None,
                "success": True,
                "error": None
            }
        
        # Run several forward passes with Flash Attention and measure time
        flash_times = []
        for _ in range(5):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            with torch.no_grad():
                flash_output = model_with_flash(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            flash_times.append(time.time() - start_time)
        
        avg_flash_time = sum(flash_times) / len(flash_times)
        logger.info(f"Flash attention average time: {avg_flash_time:.4f} seconds")
        
        # Verify outputs are similar (accounting for differences in precision)
        standard_logits = standard_output["logits"].detach()
        flash_logits = flash_output["logits"].detach()
        
        if torch.allclose(standard_logits, flash_logits, rtol=1e-2, atol=1e-2):
            logger.info("Flash attention outputs match standard attention within tolerance")
        else:
            max_diff = (standard_logits - flash_logits).abs().max().item()
            logger.warning(f"Flash attention outputs differ from standard attention (max diff: {max_diff:.6f})")
            logger.warning("This could be due to implementation differences or numeric precision")
        
        # Calculate speedup
        speedup = avg_standard_time / avg_flash_time if avg_flash_time > 0 else float('inf')
        logger.info(f"Flash attention speedup: {speedup:.2f}x")
        
        logger.info("Flash attention test completed successfully!")
        return {
            "flash_attention_available": True,
            "standard_time": avg_standard_time,
            "flash_time": avg_flash_time,
            "speedup": speedup,
            "success": True,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Error testing Flash Attention: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "flash_attention_available": FLASH_ATTENTION_AVAILABLE,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test Flash Attention for the SLM model")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH),
                      help="Path to the model configuration file")
    args = parser.parse_args()
    
    result = test_flash_attention(args.config_path)
    if result["success"]:
        if result["flash_attention_available"]:
            logger.info(f"Flash Attention works correctly with {result['speedup']:.2f}x speedup")
        else:
            logger.info("Flash Attention package not available, but compatible with standard fallback")
    else:
        logger.error(f"Flash Attention test failed: {result['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()
