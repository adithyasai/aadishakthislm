#!/usr/bin/env python3
"""
Example script demonstrating how to use the ONNX export functionality.

This script provides a simple example of how to export an SLM model to ONNX format
and how to use the exported model for inference.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch

# Add the project root to the Python path
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(project_root))

from src.onnx_export import ONNXExporter, export_onnx_inference_script

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_MODEL_PATH = project_root / "models" / "final_model"
DEFAULT_OUTPUT_PATH = project_root / "models" / "onnx_model"
DEFAULT_CONFIG_PATH = project_root / "configs" / "model_config.json"

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Example of ONNX export for SLM models')
    
    parser.add_argument(
        '--model_path', type=str, default=str(DEFAULT_MODEL_PATH),
        help='Path to the PyTorch model')
    
    parser.add_argument(
        '--output_path', type=str, default=str(DEFAULT_OUTPUT_PATH),
        help='Output directory for ONNX models')
    
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
        '--no_quantize', action='store_true',
        help='Skip model quantization')
    
    parser.add_argument(
        '--example_text', type=str, 
        default="This is an example text to summarize using the exported ONNX model.",
        help='Example text for inference demonstration')
    
    args = parser.parse_args()
    
    # Check if requirements are installed
    try:
        import onnx
        import onnxruntime
    except ImportError:
        logger.error("Required packages missing. Install with: pip install onnx onnxruntime")
        return
    
    logger.info("Starting SLM model export to ONNX format...")
    
    try:
        # Create output directory
        output_dir = Path(args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create exporter
        exporter = ONNXExporter(
            model_path=args.model_path,
            config_path=args.config_path
        )
        
        # Export the model with optional quantization
        model_paths = exporter.export_with_quantization(
            output_dir=output_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            quantize=not args.no_quantize
        )
        
        # Generate inference script
        script_path = export_onnx_inference_script(output_dir)
        
        logger.info("\nModel export completed successfully!")
        logger.info(f"Base ONNX model: {model_paths['base']}")
        
        if 'quantized' in model_paths:
            logger.info(f"Quantized ONNX model: {model_paths['quantized']}")
        
        logger.info(f"Inference script: {script_path}")
        
        # Demonstrate using the exported model (if tokenizer is available)
        try:
            from src.tokenizer import SLMTokenizer
            import onnxruntime as ort
            import numpy as np
            
            logger.info("\nDemonstrating inference with the exported ONNX model...")
            
            # Load tokenizer
            tokenizer_path = Path(args.model_path) / "tokenizer"
            if not tokenizer_path.exists():
                tokenizer_path = project_root / "models"
            
            tokenizer = SLMTokenizer.from_pretrained(str(tokenizer_path))
            
            # Tokenize example text
            inputs = tokenizer(args.example_text, return_tensors="np", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Create ONNX session
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Try with quantized model first, fall back to base model
            if 'quantized' in model_paths and not args.no_quantize:
                model_path = model_paths['quantized']
            else:
                model_path = model_paths['base']
            
            session = ort.InferenceSession(
                model_path, 
                session_options,
                providers=['CPUExecutionProvider']
            )
            
            # Run inference
            onnx_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            logger.info(f"Running inference with {Path(model_path).name}...")
            outputs = session.run(None, onnx_inputs)
            logits = outputs[0]
            
            # Get predicted tokens
            predicted_ids = np.argmax(logits, axis=-1)
            
            # Decode tokens
            predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            
            # Print results
            logger.info("\nExample Text:")
            logger.info(f"{args.example_text}")
            logger.info("\nModel Output:")
            logger.info(f"{predicted_text}")
            
        except ImportError as e:
            logger.info(f"\nCould not demonstrate inference: {e}")
            logger.info("To run inference, use the generated script:")
            logger.info(f"python {script_path} --model_dir {output_dir} --input_text \"Your text here\"")
    
    except Exception as e:
        logger.error(f"Error during export or inference: {e}", exc_info=True)

if __name__ == "__main__":
    main()
