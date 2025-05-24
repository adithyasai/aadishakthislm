#!/usr/bin/env python3
"""
Script to fix the ONNX export module syntax issues.

This script recreates the onnx_export.py file with correct syntax.
"""
import os
import shutil

# Define the path to the original file
original_file = 'd:/Personal/IITP/Projects/SLM/solution/src/onnx_export.py'
backup_file = 'd:/Personal/IITP/Projects/SLM/solution/src/onnx_export.py.bak'
fixed_file = 'd:/Personal/IITP/Projects/SLM/solution/src/onnx_export.py.fixed'

# Create a backup of the original file
shutil.copy2(original_file, backup_file)
print(f"Created backup at {backup_file}")

# Create the fixed version of the file
fixed_content = """#!/usr/bin/env python3
\"\"\"
ONNX Export Module for SLM Project

This module provides utilities for exporting the Summarization Language Model (SLM)
to ONNX format for optimized inference across different platforms.

ONNX (Open Neural Network Exchange) is an open standard for representing
machine learning models, allowing models to be transferred between different
frameworks like PyTorch, TensorFlow, and others.
\"\"\"

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, List
import torch
import torch.nn as nn
import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available. Install with: pip install onnx onnxruntime")

# Import the model from the src directory
try:
    from src.model import IndicSLM as SLMModel
    from src.tokenizer import IndicTokenizer as SLMTokenizer
except ImportError:
    # Relative import for when the script is run directly
    from model import IndicSLM as SLMModel
    from tokenizer import IndicTokenizer as SLMTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "final_model"
DEFAULT_OUTPUT_PATH = BASE_DIR / "models" / "onnx_model"
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"

class ONNXExporter:
    \"\"\"
    Handles exporting SLM models to ONNX format with various optimizations.
    \"\"\"
    
    def __init__(self, 
                model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
                config_path: Union[str, Path] = DEFAULT_CONFIG_PATH,
                dynamic_axes: bool = True):
        \"\"\"
        Initialize the ONNX exporter.
        
        Args:
            model_path: Path to the PyTorch model checkpoint
            config_path: Path to the model configuration file
            dynamic_axes: Whether to use dynamic axes for variable-length inputs
        \"\"\"
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX and ONNX Runtime are required. Please install with: "
                            "pip install onnx onnxruntime")
        
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.dynamic_axes = dynamic_axes
        
        # Load model configuration
        self.config = self._load_config()
        
        # Initialize tokenizer
        self.tokenizer = self._load_tokenizer()
        
        # Load the model
        self.model = self._load_model()
        
        # Get model's input and output names
        self.input_names = ['input_ids', 'attention_mask']
        self.output_names = ['logits']
        
        # Set up dynamic axes for variable-length inputs if needed
        if dynamic_axes:
            self.dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        else:
            self.dynamic_axes = None
    
    def _load_config(self) -> Dict:
        \"\"\"Load model configuration from JSON file\"\"\"
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded model configuration from {self.config_path}")
        return config
    
    def _load_tokenizer(self) -> SLMTokenizer:
        \"\"\"Load the tokenizer\"\"\"
        try:
            # Look for tokenizer in various locations
            tokenizer_locations = [
                self.model_path / "tokenizer",
                BASE_DIR / "models",
                BASE_DIR / "models" / "indic_slm_tokenizer.model"
            ]
            
            tokenizer_config_path = None
            for loc in tokenizer_locations:
                if loc.exists():
                    tokenizer_config_path = loc
                    break
            
            if tokenizer_config_path is None:
                # Use default config path
                tokenizer_config_path = self.config_path
            
            tokenizer = SLMTokenizer(config_path=str(tokenizer_config_path))
            logger.info(f"Loaded tokenizer using config: {tokenizer_config_path}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model(self) -> nn.Module:
        \"\"\"Load the PyTorch model\"\"\"
        try:
            # Create model instance from config
            model = SLMModel(self.config)
            
            # Load weights from checkpoint
            # Check for different file formats
            checkpoint_path = self.model_path / "model.safetensors"
            if checkpoint_path.exists():
                import safetensors.torch
                model.load_state_dict(safetensors.torch.load_file(checkpoint_path))
            else:
                checkpoint_path = self.model_path / "pytorch_model.bin"
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    raise FileNotFoundError(f"Could not find model weights at {self.model_path}")
                
            model.eval()  # Set to evaluation mode
            logger.info(f"Loaded model from {checkpoint_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _prepare_dummy_inputs(self, batch_size: int = 1, seq_length: int = 128) -> Dict[str, torch.Tensor]:
        \"\"\"
        Prepare dummy inputs for ONNX export.
        
        Args:
            batch_size: Batch size for dummy inputs
            seq_length: Sequence length for dummy inputs
            
        Returns:
            Dictionary of dummy input tensors
        \"\"\"
        # Create dummy input IDs and attention mask
        input_ids = torch.randint(
            0, self.config.get('vocab_size', 50000), 
            (batch_size, seq_length), 
            dtype=torch.long
        )
        attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def export_to_onnx(self, 
                     output_path: Union[str, Path], 
                     batch_size: int = 1,
                     seq_length: int = 128,
                     opset_version: int = 13) -> str:
        \"\"\"
        Export the model to ONNX format.
        
        Args:
            output_path: Path to save the ONNX model
            batch_size: Batch size for dummy inputs
            seq_length: Sequence length for dummy inputs
            opset_version: ONNX opset version to use
            
        Returns:
            Path to the exported ONNX model
        \"\"\"
        output_path = Path(output_path)
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare inputs for tracing
        dummy_inputs = self._prepare_dummy_inputs(batch_size, seq_length)
        
        try:
            # Export the model to ONNX
            torch.onnx.export(
                self.model,                          # PyTorch model
                (dummy_inputs['input_ids'],
                 dummy_inputs['attention_mask']),    # Model input
                str(output_path),                    # Output file
                export_params=True,                  # Store model weights in the model file
                opset_version=opset_version,         # ONNX opset version
                do_constant_folding=True,            # Optimization: Fold constants
                input_names=self.input_names,        # Input names
                output_names=self.output_names,      # Output names
                dynamic_axes=self.dynamic_axes,      # Dynamic axes for variable-length inputs
                verbose=False
            )
            
            logger.info(f"Model exported to ONNX: {output_path}")
            
            # Verify the model
            self._verify_onnx_model(output_path)
            
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
            raise
    
    def _verify_onnx_model(self, model_path: Union[str, Path]) -> None:
        \"\"\"
        Verify the exported ONNX model.
        
        Args:
            model_path: Path to the ONNX model
        \"\"\"
        try:
            # Load the ONNX model
            onnx_model = onnx.load(str(model_path))
            
            # Check the model
            onnx.checker.check_model(onnx_model)
            
            logger.info("ONNX model verification passed")
            
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            raise
    
    def quantize_onnx_model(self, 
                          input_path: Union[str, Path],
                          output_path: Union[str, Path],
                          quantization_type: str = 'dynamic',
                          per_channel: bool = False) -> str:
        \"\"\"
        Quantize the ONNX model for reduced size and faster inference.
        
        Args:
            input_path: Path to the input ONNX model
            output_path: Path to save the quantized ONNX model
            quantization_type: Type of quantization ('dynamic' or 'static')
            per_channel: Whether to use per-channel quantization
            
        Returns:
            Path to the quantized ONNX model
        \"\"\"
        input_path = str(input_path)
        output_path = str(output_path)
        
        try:
            if quantization_type == 'dynamic':
                # Apply dynamic quantization (weights are quantized to int8)
                quantize_dynamic(
                    model_input=input_path,
                    model_output=output_path,
                    per_channel=per_channel,
                    weight_type=QuantType.QInt8
                )
                logger.info(f"Model quantized (dynamic) and saved to: {output_path}")
            else:
                # Currently only dynamic quantization is supported
                # For static quantization, calibration data would be needed
                logger.error("Only dynamic quantization is currently supported")
                raise ValueError("Only dynamic quantization is currently supported")
            
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to quantize ONNX model: {e}")
            raise
    
    def compare_outputs(self, 
                      onnx_model_path: Union[str, Path],
                      batch_size: int = 1, 
                      seq_length: int = 128) -> Tuple[float, float]:
        \"\"\"
        Compare outputs between PyTorch and ONNX models.
        
        Args:
            onnx_model_path: Path to the ONNX model
            batch_size: Batch size for test inputs
            seq_length: Sequence length for test inputs
            
        Returns:
            Tuple of (max_absolute_diff, mean_absolute_diff)
        \"\"\"
        # Prepare inputs
        dummy_inputs = self._prepare_dummy_inputs(batch_size, seq_length)
        input_ids = dummy_inputs['input_ids']
        attention_mask = dummy_inputs['attention_mask']
        
        # Get PyTorch model prediction
        with torch.no_grad():
            torch_outputs = self.model(input_ids, attention_mask)
            
        # Convert to numpy
        if isinstance(torch_outputs, tuple):
            torch_outputs = torch_outputs[0]  # Assuming the first output is the logits
        torch_logits = torch_outputs.numpy()
        
        # Create ONNX Runtime session
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_session = ort.InferenceSession(
            str(onnx_model_path),
            options,
            providers=['CPUExecutionProvider']
        )
        
        # Get ONNX model prediction
        onnx_inputs = {
            'input_ids': input_ids.numpy(),
            'attention_mask': attention_mask.numpy()
        }
        onnx_outputs = onnx_session.run(None, onnx_inputs)
        onnx_logits = onnx_outputs[0]
        
        # Compare outputs
        max_absolute_diff = np.max(np.abs(torch_logits - onnx_logits))
        mean_absolute_diff = np.mean(np.abs(torch_logits - onnx_logits))
        
        logger.info(f"Max absolute difference: {max_absolute_diff}")
        logger.info(f"Mean absolute difference: {mean_absolute_diff}")
        
        return max_absolute_diff, mean_absolute_diff
    
    def export_with_quantization(self, 
                               output_dir: Union[str, Path],
                               batch_size: int = 1, 
                               seq_length: int = 128, 
                               opset_version: int = 13,
                               quantize: bool = True) -> Dict[str, str]:
        \"\"\"
        Export model to ONNX with optional quantization.
        
        Args:
            output_dir: Directory to save the ONNX models
            batch_size: Batch size for model inputs
            seq_length: Sequence length for model inputs
            opset_version: ONNX opset version to use
            quantize: Whether to also export quantized versions
            
        Returns:
            Dictionary mapping model type to file path
        \"\"\"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model config alongside ONNX models
        config_output = output_dir / "config.json"
        with open(config_output, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)
        
        # Export the base ONNX model
        base_model_path = output_dir / "model.onnx"
        self.export_to_onnx(
            base_model_path,
            batch_size=batch_size,
            seq_length=seq_length,
            opset_version=opset_version
        )
        
        result = {"base": str(base_model_path)}
        
        # Export quantized versions if requested
        if quantize:
            # Dynamic quantized model (int8)
            quant_model_path = output_dir / "model_quantized.onnx"
            self.quantize_onnx_model(
                input_path=base_model_path,
                output_path=quant_model_path,
                quantization_type='dynamic'
            )
            result["quantized"] = str(quant_model_path)
        
        # Compare ONNX model outputs with PyTorch model
        max_diff, mean_diff = self.compare_outputs(
            base_model_path,
            batch_size=batch_size,
            seq_length=seq_length
        )
        
        # Write an export report
        report_path = output_dir / "export_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"SLM Model ONNX Export Report\\n")
            f.write(f"===========================\\n\\n")
            f.write(f"Original model path: {self.model_path}\\n")
            f.write(f"ONNX model path: {base_model_path}\\n")
            if quantize:
                f.write(f"Quantized model path: {quant_model_path}\\n")
            f.write(f"\\nModel Configuration:\\n")
            for key, value in self.config.items():
                f.write(f"  {key}: {value}\\n")
            f.write(f"\\nExport Parameters:\\n")
            f.write(f"  Batch size: {batch_size}\\n")
            f.write(f"  Sequence length: {seq_length}\\n")
            f.write(f"  ONNX opset version: {opset_version}\\n")
            f.write(f"  Dynamic axes: {self.dynamic_axes is not None}\\n")
            f.write(f"\\nOutput Comparison:\\n")
            f.write(f"  Max absolute difference: {max_diff}\\n")
            f.write(f"  Mean absolute difference: {mean_diff}\\n")
        
        result["report"] = str(report_path)
        
        return result


def export_onnx_inference_script(output_dir: Union[str, Path]) -> str:
    \"\"\"
    Generate a Python script for ONNX model inference.
    
    Args:
        output_dir: Directory to save the inference script
        
    Returns:
        Path to the generated inference script
    \"\"\"
    output_dir = Path(output_dir)
    script_path = output_dir / "onnx_inference.py"
    
    script_content = \"\"\"#!/usr/bin/env python3
\"\"\"
ONNX Inference Script for SLM Model

This script demonstrates how to run inference with the ONNX version
of the Summarization Language Model (SLM).
\"\"\"

import os
import json
import argparse
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict, Union, Optional

# Import tokenizer if available
try:
    from src.tokenizer import IndicTokenizer as SLMTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("SLMTokenizer not available. Will use provided tokenizer or dummy inputs.")

def load_config(model_dir: Union[str, Path]) -> Dict:
    \"\"\"Load model configuration\"\"\"
    config_path = Path(model_dir) / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_tokenizer(model_dir: Union[str, Path]) -> Optional:
    \"\"\"Load the tokenizer if available\"\"\"
    if not TOKENIZER_AVAILABLE:
        return None
    
    try:
        tokenizer_path = Path(model_dir) / "tokenizer"
        if not tokenizer_path.exists():
            base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            tokenizer_path = base_dir / "models"
        
        tokenizer = SLMTokenizer(config_path=str(tokenizer_path))
        print(f"Loaded tokenizer from {tokenizer_path}")
        return tokenizer
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return None

def create_onnx_session(model_path: Union[str, Path], use_gpu: bool = False) -> ort.InferenceSession:
    \"\"\"Create an ONNX Runtime session\"\"\"
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Select execution provider
    if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    session = ort.InferenceSession(str(model_path), options, providers=providers)
    print(f"Created ONNX Runtime session with providers: {session.get_providers()}")
    return session

def run_inference(session: ort.InferenceSession, 
                 input_ids: np.ndarray, 
                 attention_mask: np.ndarray) -> np.ndarray:
    \"\"\"Run inference with the ONNX model\"\"\"
    onnx_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    # Run inference
    onnx_outputs = session.run(None, onnx_inputs)
    return onnx_outputs[0]  # Assuming first output is logits

def generate_summary(text: str, 
                   model_dir: Union[str, Path],
                   model_type: str = 'base',
                   use_gpu: bool = False,
                   max_length: int = 128) -> str:
    \"\"\"
    Generate a summary for the input text using the ONNX model.
    
    Args:
        text: Input text to summarize
        model_dir: Directory containing the ONNX model
        model_type: Type of model to use ('base' or 'quantized')
        use_gpu: Whether to use GPU for inference
        max_length: Maximum length of the summary
        
    Returns:
        Generated summary text
    \"\"\"
    model_dir = Path(model_dir)
    
    # Determine model path based on type
    if model_type == 'quantized' and (model_dir / "model_quantized.onnx").exists():
        model_path = model_dir / "model_quantized.onnx"
    else:
        model_path = model_dir / "model.onnx"
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_dir)
    if tokenizer is None:
        raise ValueError("Tokenizer is required for summarization")
    
    # Tokenize input
    inputs = tokenizer.encode(text)
    input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
    attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)
    
    # Create ONNX session and run inference
    session = create_onnx_session(model_path, use_gpu)
    outputs = run_inference(session, input_ids, attention_mask)
    
    # Decode outputs to get summary
    # This is a simplified approach - in a real scenario, you would
    # implement beam search or greedy decoding based on your model architecture
    output_ids = np.argmax(outputs, axis=-1)
    summary = tokenizer.decode(output_ids[0])
    
    return summary

def main():
    \"\"\"Main function\"\"\"
    parser = argparse.ArgumentParser(description='Run inference with ONNX SLM model')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing the ONNX model')
    parser.add_argument('--input_text', type=str, required=True,
                      help='Text to summarize')
    parser.add_argument('--model_type', type=str, default='base', choices=['base', 'quantized'],
                      help='Type of model to use (base or quantized)')
    parser.add_argument('--use_gpu', action='store_true',
                      help='Use GPU for inference if available')
    parser.add_argument('--max_length', type=int, default=128,
                      help='Maximum length of the summary')
    
    args = parser.parse_args()
    
    try:
        # Generate summary
        summary = generate_summary(
            args.input_text,
            args.model_dir,
            args.model_type,
            args.use_gpu,
            args.max_length
        )
        
        print("\\nInput Text:\\n", args.input_text)
        print("\\nGenerated Summary:\\n", summary)
    
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
\"\"\"
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # Make the script executable
    try:
        os.chmod(script_path, 0o755)
    except:
        pass  # May not work on Windows
    
    return str(script_path)


def main():
    \"\"\"Main function for command-line usage\"\"\"
    parser = argparse.ArgumentParser(
        description='Export SLM model to ONNX format')
    
    parser.add_argument(
        '--model_path', type=str, default=str(DEFAULT_MODEL_PATH),
        help='Path to the PyTorch model checkpoint')
    
    parser.add_argument(
        '--output', type=str, default=str(DEFAULT_OUTPUT_PATH),
        help='Output path for the ONNX model')
    
    parser.add_argument(
        '--config_path', type=str, default=str(DEFAULT_CONFIG_PATH),
        help='Path to the model configuration file')
    
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size for the exported model')
    
    parser.add_argument(
        '--seq_length', type=int, default=128,
        help='Sequence length for the exported model')
    
    parser.add_argument(
        '--opset_version', type=int, default=13,
        help='ONNX opset version to use')
    
    parser.add_argument(
        '--no_quantize', action='store_true',
        help='Do not export quantized versions')
    
    parser.add_argument(
        '--no_dynamic_axes', action='store_true',
        help='Do not use dynamic axes for variable-length inputs')
    
    parser.add_argument(
        '--create_inference_script', action='store_true',
        help='Generate a Python script for ONNX model inference')
    
    args = parser.parse_args()
    
    if not ONNX_AVAILABLE:
        logger.error("ONNX and ONNX Runtime are required. Please install with: "
                   "pip install onnx onnxruntime")
        return
    
    try:
        logger.info("Starting ONNX export...")
        
        # Create exporter
        exporter = ONNXExporter(
            model_path=args.model_path,
            config_path=args.config_path,
            dynamic_axes=not args.no_dynamic_axes
        )
        
        # Export the model
        output_dir = Path(args.output)
        model_paths = exporter.export_with_quantization(
            output_dir=output_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            opset_version=args.opset_version,
            quantize=not args.no_quantize
        )
        
        # Create inference script if requested
        if args.create_inference_script:
            inference_script_path = export_onnx_inference_script(output_dir)
            logger.info(f"Created inference script: {inference_script_path}")
        
        logger.info(f"ONNX export completed successfully")
        logger.info(f"Exported models:")
        for model_type, path in model_paths.items():
            logger.info(f"  {model_type}: {path}")
    
    except Exception as e:
        logger.error(f"Error during ONNX export: {e}", exc_info=True)


if __name__ == "__main__":
    main()
"""

# Write the fixed content to the new file
with open(fixed_file, 'w', encoding='utf-8') as f:
    f.write(fixed_content)
print(f"Created fixed file at {fixed_file}")

# Replace the original file with the fixed version
shutil.copy2(fixed_file, original_file)
print(f"Replaced original file with fixed version")

print("Fix completed successfully")
