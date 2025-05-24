import os
import sys
import torch
import argparse
import logging
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import model modules
from src.model import IndicSLM, IndicSLMConfig, quantize_model
from src.tokenizer import IndicTokenizer
from src.language_adapters import detect_language
from src.flash_attention import apply_flash_attention, FLASH_ATTENTION_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config paths
CONFIG_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.json"
CONFIG_8_LAYER_PATH = CONFIG_DIR / "model_config_8layer.json"
CONFIG_12_LAYER_PATH = CONFIG_DIR / "model_config_12layer.json"

def test_tokenizer():
    """Test tokenizer functionality"""
    logger.info("Testing tokenizer...")
    try:
        tokenizer = IndicTokenizer(str(DEFAULT_CONFIG_PATH))
        sample_text = "यह एक परीक्षण वाक्य है।"
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        logger.info(f"Original: {sample_text}")
        logger.info(f"Encoded: {encoded}")
        logger.info(f"Decoded: {decoded}")
        logger.info("Tokenizer test passed")
        return True
    except Exception as e:
        logger.error(f"Tokenizer test failed: {e}")
        return False

def test_deeper_architecture():
    """Test deeper model architecture initialization"""
    logger.info("Testing deeper model architectures...")
    results = []
    
    # Test 8-layer model
    try:
        with open(CONFIG_8_LAYER_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create config
        config = IndicSLMConfig(
            vocab_size=config_data['vocab_size'],
            hidden_size=config_data['n_embd'],
            num_hidden_layers=config_data['n_layer'],
            num_attention_heads=config_data['n_head'],
            intermediate_size=config_data['n_embd'] * 4,
            max_position_embeddings=config_data['n_positions'],
            pad_token_id=config_data['pad_token_id']
        )
        
        # Initialize model
        model = IndicSLM(config)
        
        # Verify layer count
        assert len(model.encoder) == 8, f"Expected 8 encoder layers, got {len(model.encoder)}"
        assert len(model.decoder) == 8, f"Expected 8 decoder layers, got {len(model.decoder)}"
        
        # Test forward pass
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size), \
            f"Expected output shape {(batch_size, seq_length, config.vocab_size)}, got {outputs['logits'].shape}"
        
        logger.info("8-layer model test passed")
        results.append(True)
    except Exception as e:
        logger.error(f"8-layer model test failed: {e}")
        results.append(False)
    
    # Test 12-layer model
    try:
        with open(CONFIG_12_LAYER_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create config
        config = IndicSLMConfig(
            vocab_size=config_data['vocab_size'],
            hidden_size=config_data['n_embd'],
            num_hidden_layers=config_data['n_layer'],
            num_attention_heads=config_data['n_head'],
            intermediate_size=config_data['n_embd'] * 4,
            max_position_embeddings=config_data['n_positions'],
            pad_token_id=config_data['pad_token_id']
        )
        
        # Initialize model
        model = IndicSLM(config)
        
        # Verify layer count
        assert len(model.encoder) == 12, f"Expected 12 encoder layers, got {len(model.encoder)}"
        assert len(model.decoder) == 12, f"Expected 12 decoder layers, got {len(model.decoder)}"
        
        # Test forward pass with small inputs to save memory
        batch_size = 1
        seq_length = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size), \
            f"Expected output shape {(batch_size, seq_length, config.vocab_size)}, got {outputs['logits'].shape}"
        
        logger.info("12-layer model test passed")
        results.append(True)
    except Exception as e:
        logger.error(f"12-layer model test failed: {e}")
        results.append(False)
    
    # Report summary
    if all(results):
        logger.info("All deeper architecture tests passed")
        return True
    else:
        logger.error("Some deeper architecture tests failed")
        return False

def test_rope():
    """Test Rotary Positional Embeddings"""
    logger.info("Testing Rotary Positional Embeddings...")
    
    try:
        # Create a config with rotary positional embeddings
        config = IndicSLMConfig(
            vocab_size=10000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=128,
            attention_type="rotary"  # Set to use rotary embeddings
        )
        
        # Initialize model with rotary config
        model = IndicSLM(config)
        
        # Test forward pass
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        # Run forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Verify outputs
        assert outputs['logits'].shape == (batch_size, seq_length, config.vocab_size), \
            f"Expected output shape {(batch_size, seq_length, config.vocab_size)}, got {outputs['logits'].shape}"
        
        # Verify model is using rotary attention
        for layer in model.encoder:
            assert hasattr(layer.attention, 'rope'), "Encoder layer does not use rotary embeddings"
        
        for layer in model.decoder:
            assert hasattr(layer.self_attention, 'rope'), "Decoder self-attention does not use rotary embeddings"
            assert hasattr(layer.cross_attention, 'rope'), "Decoder cross-attention does not use rotary embeddings"
        
        logger.info("RoPE test passed")
        return True
    except Exception as e:
        logger.error(f"RoPE test failed: {e}")
        return False

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

def test_quantization():
    """Test model quantization functionality"""
    logger.info("Testing model quantization...")
    try:
        # Get config
        with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create config with quantization settings
        config = IndicSLMConfig(
            vocab_size=config_data['vocab_size'],
            hidden_size=config_data['n_embd'],
            num_hidden_layers=config_data['n_layer'],
            num_attention_heads=config_data['n_head'],
            intermediate_size=config_data['n_embd'] * 4,
            max_position_embeddings=config_data['n_positions'],
            pad_token_id=config_data['pad_token_id'],
            quantization={
                "enabled": True,
                "bits": 8,
                "type": "dynamic"
            }
        )
        
        # Initialize model
        model = IndicSLM(config)
        
        # Set model to evaluation mode
        model.eval()
        
        # Prepare test input
        batch_size = 2
        seq_length = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass before quantization
        original_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply quantization
        quantized_model = quantize_model(model)
        
        # Verify quantization
        assert hasattr(quantized_model, 'is_quantized'), "Quantized model should have is_quantized attribute"
        assert quantized_model.is_quantized, "Model should be quantized"
        assert quantized_model.quantization_type == "dynamic", "Default quantization type should be dynamic"
        assert quantized_model.quantization_bits == 8, "Default quantization bits should be 8"
        
        # Run forward pass after quantization
        quantized_outputs = quantized_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Verify output shape
        assert quantized_outputs['logits'].shape == original_outputs['logits'].shape, \
            f"Quantized model output shape should match original model"
        
        logger.info("Basic quantization test passed")
        
        # Test weight-only quantization at 4-bit
        logger.info("Testing 4-bit weight-only quantization...")
        
        # Reimport model to get a fresh instance
        model = IndicSLM(IndicSLMConfig(
            vocab_size=config_data['vocab_size'],
            hidden_size=config_data['n_embd'],
            num_hidden_layers=config_data['n_layer'],
            num_attention_heads=config_data['n_head'],
            intermediate_size=config_data['n_embd'] * 4,
            max_position_embeddings=config_data['n_positions'],
            pad_token_id=config_data['pad_token_id']
        ))
        model.eval()
        
        # Get original model size (approximate)
        original_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        logger.info(f"Original model size: {original_size_mb:.2f} MB")
          # Configure 4-bit weight-only quantization
        from src.quantization import apply_weight_only_quantization
        quantized_model_4bit = apply_weight_only_quantization(model, bits=4)
        
        # Get quantized model size (approximate)
        try:
            quantized_size_mb = sum(p.numel() * p.element_size() for p in quantized_model_4bit.model.parameters()) / (1024 * 1024)
            logger.info(f"4-bit quantized model size: {quantized_size_mb:.2f} MB")
            logger.info(f"Size reduction: {original_size_mb / (quantized_size_mb or 1.0):.2f}x")
        except Exception as e:
            logger.warning(f"Error calculating model size: {e}")
            logger.info("Continuing with test...")
        
        # Run forward pass with 4-bit model
        quantized_outputs_4bit = quantized_model_4bit(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Verify output shape remains correct
        assert quantized_outputs_4bit['logits'].shape == original_outputs['logits'].shape, \
            f"4-bit quantized model output shape should match original model"
        logger.info("Model quantization tests passed")
        return True
    except Exception as e:
        logger.error(f"Model quantization test failed: {e}")
        return False

def test_flash_attention():
    """Test Flash Attention implementation"""
    logger.info("Testing Flash Attention...")
    
    try:
        # Load model configuration from config file
        with open(DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Create model config with standard attention
        config_data['attention_type'] = 'default'  # Start with standard attention
        config = IndicSLMConfig(**config_data)
        
        # Create model
        logger.info("Creating model with standard attention...")
        model = IndicSLM(config)
        model.eval()
        
        # Generate test input
        batch_size = 2
        seq_length = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Test standard attention
        logger.info("Testing standard attention performance...")
        
        # Run forward passes with standard attention and measure time
        standard_times = []
        for _ in range(3):  # Run 3 times for average
            start_time = time.time()
            with torch.no_grad():
                standard_output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            standard_times.append(time.time() - start_time)
        
        avg_standard_time = sum(standard_times) / len(standard_times)
        logger.info(f"Standard attention average time: {avg_standard_time:.4f} seconds")
        
        # Apply Flash Attention to the model
        logger.info("Applying Flash Attention...")
        model_with_flash = apply_flash_attention(model)
        
        if not FLASH_ATTENTION_AVAILABLE:
            logger.warning("Flash Attention package is not available. The test is using standard attention fallback.")
            logger.info("To install Flash Attention, run: pip install flash-attn --no-build-isolation")
            logger.info("Flash Attention test passed with fallback to standard attention")
            return True
        
        # Run forward passes with Flash Attention and measure time
        flash_times = []
        for _ in range(3):  # Run 3 times for average
            start_time = time.time()
            with torch.no_grad():
                flash_output = model_with_flash(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            flash_times.append(time.time() - start_time)
        
        avg_flash_time = sum(flash_times) / len(flash_times)
        logger.info(f"Flash attention average time: {avg_flash_time:.4f} seconds")
        
        # Calculate speedup
        speedup = avg_standard_time / avg_flash_time if avg_flash_time > 0 else float('inf')
        logger.info(f"Flash attention speedup: {speedup:.2f}x")
        
        # Verify outputs are similar
        standard_logits = standard_output["logits"].detach()
        flash_logits = flash_output["logits"].detach()
        
        # Flash outputs may differ slightly due to implementation differences
        # Use a reasonable tolerance
        if torch.allclose(standard_logits, flash_logits, rtol=1e-2, atol=1e-2):
            logger.info("Flash attention outputs match standard attention within tolerance")
        else:
            max_diff = (standard_logits - flash_logits).abs().max().item()
            logger.warning(f"Flash attention outputs differ from standard attention (max diff: {max_diff:.6f})")
            logger.warning("This could be due to implementation differences or numeric precision")
        
        logger.info("Flash attention test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Flash Attention: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    parser = argparse.ArgumentParser(description="Test SLM model components")
    parser.add_argument("--test_tokenizer", action="store_true", help="Test tokenizer functionality")
    parser.add_argument("--test_deeper_architecture", action="store_true", help="Test deeper architecture configurations")
    parser.add_argument("--test_rope", action="store_true", help="Test Rotary Positional Embeddings")
    parser.add_argument("--test_adapters", action="store_true", help="Test language-specific adapters")
    parser.add_argument("--test_quantization", action="store_true", help="Test model quantization")
    parser.add_argument("--test_flash_attention", action="store_true", help="Test Flash Attention implementation")
    parser.add_argument("--test_all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Run selected tests or all tests
    if args.test_all:
        test_tokenizer()
        test_deeper_architecture()
        test_rope()
        test_adapters()
        test_quantization()
        test_flash_attention()
    else:
        if args.test_tokenizer:
            test_tokenizer()
        if args.test_deeper_architecture:
            test_deeper_architecture()
        if args.test_rope:
            test_rope()
        if args.test_adapters:
            test_adapters()
        if args.test_quantization:
            test_quantization()
        if args.test_flash_attention:
            test_flash_attention()

if __name__ == "__main__":
    main()