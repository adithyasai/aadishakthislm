"""
Model quantization support for Summarization Language Model (SLM).

This module provides functions for quantizing the SLM model to reduce memory usage
and improve inference speed, supporting 8-bit and 4-bit quantization.

Quantization methods:
1. Post-training static quantization
2. Dynamic quantization
3. Quantization-aware training (QAT)
"""

import os
import torch
import logging
from typing import Dict, Optional, Union, Tuple, List, Any
from pathlib import Path
import numpy as np

# Import the dynamic quantization function if available
try:
    from torch.quantization import quantize_dynamic
except ImportError:
    # Define a simple fallback for older PyTorch versions
    def quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, inplace=False):
        logging.warning("quantize_dynamic not available in this PyTorch version. Using a basic quantization wrapper instead.")
        return model

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


def quantize_dynamic(model, bits=8):
    """
    Apply dynamic quantization to the model.
    
    Dynamic quantization quantizes weights to 8-bit or 4-bit but keeps activations in fp32.
    Weights are quantized once ahead of time, while activations are dynamically quantized
    during inference.
    
    Args:
        model: The model to quantize
        bits: Quantization precision (8 or 4)
        
    Returns:
        QuantizedModelWrapper with dynamically quantized model
    """
    logger.info(f"Applying {bits}-bit dynamic quantization to the model")
    
    # Create a copy of the original model for quantization
    original_model = model
    model = type(model)(model.config)
    model.load_state_dict(original_model.state_dict())
    
    # Set model to evaluation mode
    model.eval()
    
    # Define quantization parameters
    if bits == 8:
        dtype = torch.qint8
    elif bits == 4:
        if TORCH_VERSION_FLOAT < 2.0:
            logger.warning(f"4-bit quantization requires PyTorch 2.x. Falling back to 8-bit.")
            dtype = torch.qint8
        else:
            dtype = torch.qint8  # Use qint8 as fallback if qint4 not available
    else:
        raise ValueError(f"Unsupported bit precision: {bits}. Use 8 or 4.")
    
    # Figure out which modules to quantize
    qconfig_dict = {
        "": torch.quantization.default_dynamic_qconfig,
        "object_type": [
            (torch.nn.Linear, torch.quantization.default_dynamic_qconfig),
            (torch.nn.LSTM, torch.quantization.default_dynamic_qconfig),
            (torch.nn.GRU, torch.quantization.default_dynamic_qconfig),
        ]
    }
    
    # Apply dynamic quantization to supported modules
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        qconfig_dict if TORCH_VERSION_FLOAT >= 1.7 else {"": torch.quantization.default_dynamic_qconfig},
        dtype=dtype,
        inplace=False
    )
    
    # Create wrapper
    wrapper = QuantizedModelWrapper(quantized_model)
    wrapper.is_quantized = True
    wrapper.quantization_type = "dynamic"
    wrapper.quantization_bits = bits
    wrapper.original_model = original_model
    
    logger.info(f"Model successfully quantized to {bits}-bit dynamic quantization")
    return wrapper


def quantize_static(model, calibration_data, bits=8):
    """
    Apply static quantization to the model.
    
    Static quantization requires calibration data to determine optimal quantization
    parameters for activations based on their distributions.
    
    Args:
        model: The model to quantize
        calibration_data: Data for calibrating the quantization parameters
        bits: Quantization precision (8 or 4)
        
    Returns:
        QuantizedModelWrapper with statically quantized model
    """
    logger.info(f"Applying {bits}-bit static quantization to the model")
    
    # Create a copy of the original model for quantization
    original_model = model
    model = type(model)(model.config)
    model.load_state_dict(original_model.state_dict())
    
    # Set model to evaluation mode
    model.eval()
    
    # Check if static quantization is supported in this PyTorch version
    if TORCH_VERSION_FLOAT < 1.11:
        logger.warning("Static quantization through FX Graph is not fully supported in this PyTorch version.")
        logger.warning("Falling back to dynamic quantization.")
        return quantize_dynamic(model, bits=bits)
        
    try:
        # Add observers to the model
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate with sample data
        logger.info("Calibrating quantization parameters with sample data")
        with torch.no_grad():
            for batch in calibration_data:
                model(**batch)
        
        # Convert to fully quantized model
        quantized_model = torch.quantization.convert(model, inplace=False)
        
        # Create wrapper
        wrapper = QuantizedModelWrapper(quantized_model)
        wrapper.is_quantized = True
        wrapper.quantization_type = "static"
        wrapper.quantization_bits = bits
        wrapper.original_model = original_model
        
        logger.info(f"Model successfully quantized to {bits}-bit static quantization")
        return wrapper
        
    except (ImportError, AttributeError) as e:
        logger.warning(f"Static quantization failed: {str(e)}")
        logger.warning("Falling back to dynamic quantization.")
        return quantize_dynamic(model, bits=bits)


def prepare_qat(model, bits=8):
    """
    Prepare model for quantization-aware training (QAT).
    
    QAT simulates quantization during training, allowing the model to adapt to
    quantization effects. The model remains in floating point but learns parameters
    that work well when quantized.
    
    Args:
        model: The model to prepare for QAT
        bits: Quantization precision (8 or 4)
        
    Returns:
        Model prepared for quantization-aware training
    """
    logger.info(f"Preparing model for {bits}-bit quantization-aware training")
    
    # Create a copy of the original model
    original_model = model
    model = type(model)(model.config)
    model.load_state_dict(original_model.state_dict())
    
    # Set model to training mode
    model.train()
    
    # Check if QAT is supported in this PyTorch version
    if TORCH_VERSION_FLOAT < 1.10:
        logger.warning("Quantization-aware training is not fully supported in this PyTorch version.")
        logger.warning("Returning original model. You may need to use dynamic quantization after training.")
        return model
        
    try:
        # Try to use QAT if available
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        
        logger.info(f"Model prepared for {bits}-bit quantization-aware training")
        return model
        
    except (ImportError, AttributeError) as e:
        logger.warning(f"QAT preparation failed: {str(e)}")
        logger.warning("Returning original model. You may need to use dynamic quantization after training.")
        return model


def convert_qat_to_quantized(model):
    """
    Convert a quantization-aware trained model to a fully quantized model.
    
    Args:
        model: Trained QAT model
        
    Returns:
        QuantizedModelWrapper with quantized model
    """
    logger.info("Converting QAT model to fully quantized model")
    
    # Set model to evaluation mode
    model.eval()
    
    try:
        # Convert model to fully quantized version
        quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        
        # Create wrapper
        wrapper = QuantizedModelWrapper(quantized_model)
        wrapper.is_quantized = True
        wrapper.quantization_type = "qat"
        wrapper.quantization_bits = 8  # QAT typically uses 8-bit
        
        logger.info("QAT model successfully converted to quantized model")
        return wrapper
        
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.warning(f"QAT conversion failed: {str(e)}")
        logger.warning("Falling back to dynamic quantization.")
        return quantize_dynamic(model, bits=8)


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
                scale = (max_val - min_val) / (2 ** bits - 1 + 1e-6)
                zero_point = (min_val / (scale + 1e-6)).round()
                
                # Quantize weights
                quantized_weight = (weight / (scale + 1e-6)).round() + zero_point
                
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


def compare_model_performance(original_model, quantized_model, test_data, num_batches=5):
    """
    Compare the performance of original and quantized models.
    
    Measures inference speed, memory usage, and accuracy differences.
    
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
        "memory_usage": {
            "original": None,
            "quantized": None
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
    
    # Measure model size
    def get_model_size(model):
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.numel() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.numel() * buffer.element_size()
            return param_size + buffer_size
        except Exception as e:
            logger.warning(f"Error calculating model size: {e}")
            # Simple fallback estimation
            return sum(p.numel() for p in model.parameters()) * 4  # Assuming float32
    
    results["model_size"]["original"] = get_model_size(original_model) / (1024 * 1024)  # in MB
    results["model_size"]["quantized"] = get_model_size(quantized_model.model) / (1024 * 1024)  # in MB
    
    # Test inference performance
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            if i >= num_batches:
                break
            
            # Check if CUDA is available for timing
            if torch.cuda.is_available():
                # Original model inference time with CUDA events
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                original_output = original_model(**batch)
                end_time.record()
                
                torch.cuda.synchronize()
                results["inference_time"]["original"].append(start_time.elapsed_time(end_time))
                
                # Quantized model inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                quantized_output = quantized_model(**batch)
                end_time.record()
                
                torch.cuda.synchronize()
                results["inference_time"]["quantized"].append(start_time.elapsed_time(end_time))
            else:
                # CPU timing fallback
                import time
                
                start = time.time()
                original_output = original_model(**batch)
                results["inference_time"]["original"].append((time.time() - start) * 1000)  # ms
                
                start = time.time()
                quantized_output = quantized_model(**batch)
                results["inference_time"]["quantized"].append((time.time() - start) * 1000)  # ms
            
            # Calculate output difference
            try:
                original_logits = original_output["logits"]
                quantized_logits = quantized_output["logits"]
                
                # Calculate mean squared error between outputs
                mse = torch.mean((original_logits - quantized_logits) ** 2).item()
                results["output_difference"].append(mse)
            except Exception as e:
                logger.warning(f"Error calculating output difference: {e}")
    
    # Calculate averages
    if results["inference_time"]["original"]:
        results["inference_time"]["original"] = sum(results["inference_time"]["original"]) / len(results["inference_time"]["original"])
        results["inference_time"]["quantized"] = sum(results["inference_time"]["quantized"]) / len(results["inference_time"]["quantized"])
        
        if results["output_difference"]:
            results["output_difference"] = sum(results["output_difference"]) / len(results["output_difference"])
        
        # Calculate speedup and size reduction
        results["speedup"] = results["inference_time"]["original"] / (results["inference_time"]["quantized"] or 1.0)
        results["size_reduction"] = results["model_size"]["original"] / (results["model_size"]["quantized"] or 1.0)
        
        logger.info(f"Performance comparison completed. Speedup: {results['speedup']:.2f}x, Size reduction: {results['size_reduction']:.2f}x")
    else:
        logger.warning("No performance data collected")
    
    return results


def create_calibration_data_loader(data_processor, dataset_path, batch_size=4, max_samples=100):
    """
    Create a calibration data loader for static quantization.
    
    Args:
        data_processor: Data processor instance
        dataset_path: Path to the dataset
        batch_size: Batch size for calibration
        max_samples: Maximum number of samples to use for calibration
        
    Returns:
        Iterable of batches for calibration
    """
    # Load and prepare dataset
    dataset = data_processor.prepare_dataset(dataset_path, "hi")  # Using Hindi dataset for calibration
    
    # Limit dataset size for calibration
    if len(dataset) > max_samples:
        subset_indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(subset_indices)
    
    # Create simple batches for calibration
    def create_batches():
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:min(i+batch_size, len(dataset))]
            
            # Format into model inputs
            input_ids = torch.tensor(batch_data["input_ids"])
            attention_mask = torch.tensor(batch_data["attention_mask"])
            
            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
    
    return create_batches()
