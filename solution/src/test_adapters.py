import os
import sys
import torch
import logging
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import model modules
from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer
from src.language_adapters import detect_language

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config paths
CONFIG_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.json"

def test_adapters():
    """Test language-specific adapters"""
    logger.info("Testing language adapters...")

    try:
        # Create a config with language adapters enabled
        config = IndicSLMConfig(
            vocab_size=10000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=128,
            use_adapters=True,
            adapter_size=32,
            languages=["hi", "te"]
        )
        
        # Initialize model with adapter config
        model = IndicSLM(config)
        
        # Verify model was created with adapters
        assert hasattr(model, "language_adapters"), "Model doesn't have language adapters attribute"
        
        # Test language detection
        test_texts = [
            "यह एक हिंदी वाक्य है।",  # Hindi
            "ఇది తెలుగు వాక్యం.",     # Telugu
            "<hi> हिंदी भाषा एक है",   # Tagged Hindi
            "<te> తెలుగు భాష చాలా అందమైనది", # Tagged Telugu
        ]
        
        logger.info("Testing language detection:")
        for text in test_texts:
            detected = detect_language(text)
            logger.info(f"Text: '{text}' - Detected language: '{detected}'")
        
        # Test forward pass with different language IDs
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        # Test with Hindi language ID
        logger.info("Testing forward pass with Hindi adapter")
        outputs_hindi = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            language_id="hi"
        )
        
        # Test with Telugu language ID
        logger.info("Testing forward pass with Telugu adapter")
        outputs_telugu = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            language_id="te"
        )
        
        # Test that different language IDs produce different outputs
        # The difference should be small but non-zero due to adapter parameters
        diff = (outputs_hindi['logits'] - outputs_telugu['logits']).abs().mean().item()
        logger.info(f"Mean absolute difference between Hindi and Telugu outputs: {diff:.6f}")
        assert diff > 0, "Language adapters are not producing different outputs"
        
        # Verify outputs have the expected shape
        assert outputs_hindi['logits'].shape == (batch_size, seq_length, config.vocab_size), \
            f"Expected output shape {(batch_size, seq_length, config.vocab_size)}, got {outputs_hindi['logits'].shape}"
        
        logger.info("Language adapters test passed")
        return True
    except Exception as e:
        logger.error(f"Language adapters test failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test language adapters")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    test_adapters()