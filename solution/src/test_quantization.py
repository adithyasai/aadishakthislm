"""
Test quantization functionality for SLM models.

This module tests different quantization methods:
1. Dynamic quantization
2. Static quantization
3. Weight-only quantization

It measures the performance impact (speed, memory, accuracy) of each method.
"""

import os
import sys
import time
import torch
import argparse
import logging
import json
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import required modules
from src.model import IndicSLM, IndicSLMConfig, quantize_model
from src.tokenizer import IndicTokenizer
from src.quantization import (
    quantize_dynamic,
    quantize_static,
    apply_weight_only_quantization,
    compare_model_performance,
    create_calibration_data_loader
)
from src.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config paths
CONFIG_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.json"
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def test_dynamic_quantization(model, test_data, bits=8):
    """Test dynamic quantization"""
    logger.info(f"Testing {bits}-bit dynamic quantization...")
    
    # Quantize the model
    quantized_model = quantize_dynamic(model, bits=bits)
    
    # Verify quantization
    assert quantized_model.is_quantized, "Model should be quantized"
    assert quantized_model.quantization_type == "dynamic", "Quantization type should be dynamic"
    assert quantized_model.quantization_bits == bits, f"Quantization bits should be {bits}"
    
    # Compare performance
    results = compare_model_performance(model, quantized_model, test_data)
    
    logger.info(f"Dynamic quantization results:")
    logger.info(f"- Speedup: {results['speedup']:.2f}x")
    logger.info(f"- Size reduction: {results['size_reduction']:.2f}x")
    logger.info(f"- Output difference (MSE): {results['output_difference']}")
    
    return quantized_model, results

def test_static_quantization(model, test_data, calibration_data, bits=8):
    """Test static quantization"""
    logger.info(f"Testing {bits}-bit static quantization...")
    
    # Quantize the model
    quantized_model = quantize_static(model, calibration_data, bits=bits)
    
    # Verify quantization
    assert quantized_model.is_quantized, "Model should be quantized"
    assert quantized_model.quantization_type == "static", "Quantization type should be static"
    assert quantized_model.quantization_bits == bits, f"Quantization bits should be {bits}"
    
    # Compare performance
    results = compare_model_performance(model, quantized_model, test_data)
    
    logger.info(f"Static quantization results:")
    logger.info(f"- Speedup: {results['speedup']:.2f}x")
    logger.info(f"- Size reduction: {results['size_reduction']:.2f}x")
    logger.info(f"- Output difference (MSE): {results['output_difference']}")
    
    return quantized_model, results

def test_weight_only_quantization(model, test_data, bits=8):
    """Test weight-only quantization"""
    logger.info(f"Testing {bits}-bit weight-only quantization...")
    
    # Quantize the model
    quantized_model = apply_weight_only_quantization(model, bits=bits)
    
    # Verify quantization
    assert quantized_model.is_quantized, "Model should be quantized"
    assert quantized_model.quantization_type == "weight_only", "Quantization type should be weight_only"
    assert quantized_model.quantization_bits == bits, f"Quantization bits should be {bits}"
    
    # Compare performance
    results = compare_model_performance(model, quantized_model, test_data)
    
    logger.info(f"Weight-only quantization results:")
    logger.info(f"- Speedup: {results['speedup']:.2f}x")
    logger.info(f"- Size reduction: {results['size_reduction']:.2f}x")
    logger.info(f"- Output difference (MSE): {results['output_difference']}")
    
    return quantized_model, results

def test_model_save_load(quantized_model, model_class, model_config, save_dir="models/quantized"):
    """Test saving and loading quantized model"""
    logger.info("Testing save/load functionality for quantized model...")
    
    # Create a directory for the test
    save_path = Path(__file__).parent.parent / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    quantized_model.save(str(save_path))
    
    # Load the model
    loaded_model = quantized_model.__class__.load(str(save_path), model_class, model_config)
    
    # Verify loaded model has same quantization properties
    assert loaded_model.is_quantized == quantized_model.is_quantized, "Loaded model should have same quantization state"
    assert loaded_model.quantization_type == quantized_model.quantization_type, "Loaded model should have same quantization type"
    assert loaded_model.quantization_bits == quantized_model.quantization_bits, "Loaded model should have same quantization bits"
    
    logger.info("Model save/load test passed")
    return loaded_model

def create_test_data(tokenizer, batch_size=4, seq_length=64, num_batches=10):
    """Create synthetic test data for model evaluation"""
    test_data = []
    
    for _ in range(num_batches):
        # Generate random tokens
        input_ids = torch.randint(
            low=0, 
            high=tokenizer.vocab_size, 
            size=(batch_size, seq_length),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Create attention mask (all 1s for simplicity)
        attention_mask = torch.ones_like(input_ids)
        
        test_data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
    
    return test_data

def main(args):
    logger.info("Initializing test environment...")
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # Create model configuration with quantization settings
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
            "bits": args.bits,
            "type": args.quantization_type
        }
    )
    
    # Initialize model
    model = IndicSLM(config)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Initialize tokenizer
    tokenizer = IndicTokenizer(str(args.config))
    
    # Generate test data
    test_data = create_test_data(tokenizer, batch_size=4, seq_length=64, num_batches=10)
    
    # Test quantization
    if args.test_all or args.quantization_type == "dynamic":
        # Test 8-bit dynamic quantization
        quantized_model_8bit, results_8bit = test_dynamic_quantization(model, test_data, bits=8)
        
        # Test 4-bit dynamic quantization if requested
        if args.test_4bit:
            quantized_model_4bit, results_4bit = test_dynamic_quantization(model, test_data, bits=4)
    
    if args.test_all or args.quantization_type == "static":
        # Create data processor
        data_processor = DataProcessor()
        
        # Create calibration data
        calibration_data = create_calibration_data_loader(
            data_processor,
            str(PROCESSED_DATA_DIR / "train.jsonl"),
            batch_size=4,
            max_samples=100
        )
        
        # Test 8-bit static quantization
        quantized_model_static, results_static = test_static_quantization(
            model, test_data, calibration_data, bits=8
        )
    
    if args.test_all or args.quantization_type == "weight_only":
        # Test 8-bit weight-only quantization
        quantized_model_weight, results_weight = test_weight_only_quantization(model, test_data, bits=8)
        
        # Test 4-bit weight-only quantization if requested
        if args.test_4bit:
            quantized_model_weight_4bit, results_weight_4bit = test_weight_only_quantization(model, test_data, bits=4)
        
        # Test 2-bit weight-only quantization if requested
        if args.test_2bit:
            quantized_model_weight_2bit, results_weight_2bit = test_weight_only_quantization(model, test_data, bits=2)
    
    # Test model saving and loading
    if args.test_save_load:
        quantized_model = (
            quantized_model_8bit if args.quantization_type == "dynamic" else
            quantized_model_static if args.quantization_type == "static" else
            quantized_model_weight
        )
        loaded_model = test_model_save_load(quantized_model, IndicSLM, config)
    
    logger.info("All quantization tests completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model quantization")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to model config")
    parser.add_argument("--quantization_type", type=str, default="dynamic", choices=["dynamic", "static", "weight_only"], help="Quantization type")
    parser.add_argument("--bits", type=int, default=8, choices=[2, 4, 8], help="Quantization precision")
    parser.add_argument("--test_all", action="store_true", help="Test all quantization methods")
    parser.add_argument("--test_4bit", action="store_true", help="Test 4-bit quantization")
    parser.add_argument("--test_2bit", action="store_true", help="Test 2-bit quantization (only for weight-only)")
    parser.add_argument("--test_save_load", action="store_true", help="Test model saving and loading")
    
    args = parser.parse_args()
    main(args)
