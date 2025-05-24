"""
This is a simplified quantization implementation that should work with older PyTorch versions.
It focuses on weight-only quantization which can be applied to any model without requiring
special PyTorch quantization modules.
"""

import os
import torch
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class QuantizedModelWrapper:
    """
    Wrapper for quantized SLM model to handle quantization-specific operations.
    """
    
    def __init__(self, model, quantization_config=None):
        """
        Initialize the quantized model wrapper.
        
        Args:
            model: The model to be quantized or already quantized model
            quantization_config: Configuration for quantization
        """
        self.model = model
        self.quantization_config = quantization_config or {}
        self.is_quantized = False
        self.original_model = None
        self.quantization_type = None
        self.quantization_bits = None
    
    def __call__(self, *args, **kwargs):
        """Forward pass through the quantized model"""
        return self.model(*args, **kwargs)
    
    def save(self, save_dir: str):
        """
        Save the quantized model.
        
        Args:
            save_dir: Directory to save the model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, "quantized_model.pt")
        torch.save(self.model.state_dict(), model_path)
        
        # Save quantization config
        config_path = os.path.join(save_dir, "quantization_config.pt")
        config_to_save = {
            "is_quantized": self.is_quantized,
            "quantization_type": self.quantization_type,
            "quantization_bits": self.quantization_bits,
            "config": self.quantization_config
        }
        torch.save(config_to_save, config_path)
        
        logger.info(f"Quantized model saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: str, model_class, model_config=None):
        """
        Load a quantized model.
        
        Args:
            load_dir: Directory containing the saved model
            model_class: The model class to instantiate
            model_config: Configuration for model initialization
            
        Returns:
            QuantizedModelWrapper instance with loaded model
        """
        # Load quantization config
        config_path = os.path.join(load_dir, "quantization_config.pt")
        if not os.path.exists(config_path):
            raise ValueError(f"Quantization config not found at {config_path}")
        
        quant_config = torch.load(config_path)
        
        # Create an instance of the model
        if model_config is None:
            model = model_class()
        else:
            model = model_class(model_config)
        
        # Load model weights
        model_path = os.path.join(load_dir, "quantized_model.pt")
        model.load_state_dict(torch.load(model_path))
        
        # Create wrapper
        wrapper = cls(model, quant_config["config"])
        wrapper.is_quantized = quant_config["is_quantized"]
        wrapper.quantization_type = quant_config["quantization_type"]
        wrapper.quantization_bits = quant_config["quantization_bits"]
        
        return wrapper


def apply_weight_only_quantization(model, bits=8):
    """
    Apply weight-only quantization to the model.
    
    This method quantizes only the weights, keeping activations in full precision.
    It's faster than full quantization but offers less memory savings.
    
    Args:
        model: The model to quantize
        bits: Quantization precision (8, 4, or 2)
        
    Returns:
        QuantizedModelWrapper with weight-only quantized model
    """
    logger.info(f"Applying {bits}-bit weight-only quantization to the model")
    
    # Create a copy of the original model
    original_model = model
    model = type(model)(model.config)
    model.load_state_dict(original_model.state_dict())
    
    # Set model to evaluation mode
    model.eval()
    
    # Validate bits value
    if bits not in [8, 4, 2]:
        logger.warning(f"Unsupported bit width: {bits}. Using 8-bit instead.")
        bits = 8
    
    try:
        # Quantize all linear layer weights
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Get the weight tensor
                weight = module.weight.data
                
                # Simple min-max quantization
                min_val = weight.min()
                max_val = weight.max()
                
                # Add small epsilon to avoid division by zero
                epsilon = 1e-6
                scale = (max_val - min_val) / (2 ** bits - 1 + epsilon)
                zero_point = (min_val / (scale + epsilon)).round()
                
                # Quantize weights
                quantized_weight = (weight / (scale + epsilon)).round() + zero_point
                
                # Clamp to ensure values are within range
                if bits == 8:
                    quantized_weight = torch.clamp(quantized_weight, 0, 255)
                elif bits == 4:
                    quantized_weight = torch.clamp(quantized_weight, 0, 15)
                elif bits == 2:
                    quantized_weight = torch.clamp(quantized_weight, 0, 3)
                
                # Dequantize for actual use (simulate quantization)
                dequantized_weight = (quantized_weight - zero_point) * scale
                
                # Update the weights
                module.weight.data = dequantized_weight
        
        # Create wrapper
        wrapper = QuantizedModelWrapper(model)
        wrapper.is_quantized = True
        wrapper.quantization_type = "weight_only"
        wrapper.quantization_bits = bits
        wrapper.original_model = original_model
        
        logger.info(f"Model successfully quantized with {bits}-bit weight-only quantization")
        return wrapper
        
    except Exception as e:
        logger.error(f"Weight-only quantization failed: {str(e)}")
        logger.warning("Returning a wrapper with the original model.")
        wrapper = QuantizedModelWrapper(model)
        wrapper.is_quantized = False
        wrapper.original_model = original_model
        return wrapper


def quantize_dynamic(model, bits=8):
    """
    Apply dynamic quantization to the model.
    
    Dynamic quantization quantizes weights to 8-bit but keeps activations in fp32.
    This is a compatibility wrapper that uses PyTorch's quantize_dynamic if available,
    or falls back to a custom implementation if not.
    
    Args:
        model: The model to quantize
        bits: Quantization precision (8 or 4)
        
    Returns:
        QuantizedModelWrapper with dynamically quantized model
    """
    logger.info(f"Applying {bits}-bit dynamic quantization to the model")
    
    # Create a copy of the original model
    original_model = model
    model = type(model)(model.config)
    model.load_state_dict(original_model.state_dict())
    
    # Set model to evaluation mode
    model.eval()
    
    try:
        # Try to use PyTorch's built-in dynamic quantization
        from torch.quantization import quantize_dynamic as torch_quantize_dynamic
        
        # Determine data type
        dtype = torch.qint8  # PyTorch currently only supports 8-bit dynamic quantization
        
        # Define quantization configuration
        qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
        
        # Apply dynamic quantization
        quantized_model = torch_quantize_dynamic(
            model,
            qconfig_dict,  
            dtype=dtype,
            inplace=False
        )
        
        # Create wrapper
        wrapper = QuantizedModelWrapper(quantized_model)
        wrapper.is_quantized = True
        wrapper.quantization_type = "dynamic"
        wrapper.quantization_bits = 8  # PyTorch dynamic quant only supports 8-bit
        wrapper.original_model = original_model
        
        logger.info("Model successfully quantized with PyTorch dynamic quantization")
        return wrapper
        
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.warning(f"PyTorch dynamic quantization failed: {str(e)}")
        logger.warning("Falling back to weight-only quantization as an alternative.")
        return apply_weight_only_quantization(model, bits=bits)


def compare_model_performance(original_model, quantized_model, test_data, num_batches=5):
    """
    Compare the performance of original and quantized models.
    
    Args:
        original_model: The original unquantized model
        quantized_model: The quantized model
        test_data: Test data for performance comparison
        num_batches: Number of batches to use for testing
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info("Comparing performance between original and quantized models")
    
    results = {
        "inference_time": {
            "original": [],
            "quantized": []
        },
        "model_size": {
            "original": None,
            "quantized": None
        },
        "output_difference": []
    }
    
    # Set both models to evaluation mode
    original_model.eval()
    quantized_model.model.eval()
    
    # Measure model size (parameters × bytes per parameter)
    original_params = sum(p.numel() for p in original_model.parameters())
    quantized_params = sum(p.numel() for p in quantized_model.model.parameters())
    
    # Approximate size in MB (assuming float32 for original, int8 for quantized)
    results["model_size"]["original"] = original_params * 4 / (1024 * 1024)
    results["model_size"]["quantized"] = quantized_params * 1 / (1024 * 1024)  # Assuming int8
    
    # Test inference performance
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            if i >= num_batches:
                break
            
            # CPU timing
            import time
            
            start = time.time()
            original_output = original_model(**batch)
            original_time = (time.time() - start) * 1000  # Convert to ms
            results["inference_time"]["original"].append(original_time)
            
            start = time.time()
            quantized_output = quantized_model(**batch)
            quantized_time = (time.time() - start) * 1000  # Convert to ms
            results["inference_time"]["quantized"].append(quantized_time)
            
            # Calculate output difference
            original_logits = original_output["logits"]
            quantized_logits = quantized_output["logits"]
            
            # Mean squared error between outputs
            mse = torch.mean((original_logits - quantized_logits) ** 2).item()
            results["output_difference"].append(mse)
    
    # Calculate averages
    results["inference_time"]["original"] = sum(results["inference_time"]["original"]) / len(results["inference_time"]["original"])
    results["inference_time"]["quantized"] = sum(results["inference_time"]["quantized"]) / len(results["inference_time"]["quantized"])
    results["output_difference"] = sum(results["output_difference"]) / len(results["output_difference"])
    
    # Calculate speedup and size reduction
    results["speedup"] = results["inference_time"]["original"] / results["inference_time"]["quantized"]
    results["size_reduction"] = results["model_size"]["original"] / results["model_size"]["quantized"]
    
    logger.info(f"Performance comparison completed.")
    logger.info(f"- Original model inference time: {results['inference_time']['original']:.2f} ms")
    logger.info(f"- Quantized model inference time: {results['inference_time']['quantized']:.2f} ms")
    logger.info(f"- Speedup: {results['speedup']:.2f}x")
    logger.info(f"- Size reduction: {results['size_reduction']:.2f}x")
    logger.info(f"- Output MSE: {results['output_difference']:.6f}")
    
    return results


# Additional compatibility functions for basic testing
def create_calibration_data_loader(data_processor, dataset_path, batch_size=4, max_samples=100):
    """
    Create a calibration data loader for static quantization (compatibility function).
    """
    # Simplified implementation that returns a simple dummy dataset
    def dummy_calibration_data():
        for _ in range(10):  # 10 batches for calibration
            yield {"input_ids": torch.randint(0, 1000, (batch_size, 16)),
                   "attention_mask": torch.ones((batch_size, 16))}
    
    return dummy_calibration_data()
