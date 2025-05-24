#!/usr/bin/env python3
"""
Test script for ONNX export functionality.

This script tests the ONNX export capabilities for the SLM project,
ensuring that models can be properly exported and quantized.
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
DEFAULT_OUTPUT_PATH = project_root / "models" / "test_onnx"
DEFAULT_CONFIG_PATH = project_root / "configs" / "model_config.json"

def test_basic_export(model_path, output_path, config_path):
    """Test basic ONNX export functionality"""
    logger.info("Testing basic ONNX export...")
    
    try:
        # Create exporter
        exporter = ONNXExporter(
            model_path=model_path,
            config_path=config_path
        )
        
        # Export the model
        output_file = Path(output_path) / "basic_model.onnx"
        exported_path = exporter.export_to_onnx(
            output_path=output_file,
            batch_size=1,
            seq_length=64
        )
        
        logger.info(f"Basic export successful: {exported_path}")
        return True
    
    except Exception as e:
        logger.error(f"Basic export failed: {e}")
        return False

def test_quantization(model_path, output_path, config_path):
    """Test model quantization"""
    logger.info("Testing model quantization...")
    
    try:
        # Create exporter
        exporter = ONNXExporter(
            model_path=model_path,
            config_path=config_path
        )
        
        # Export the model
        output_dir = Path(output_path)
        base_model_path = output_dir / "base_model.onnx"
        
        # Export base model
        exporter.export_to_onnx(
            output_path=base_model_path,
            batch_size=1,
            seq_length=64
        )
        
        # Quantize the model
        quantized_path = output_dir / "quantized_model.onnx"
        exporter.quantize_onnx_model(
            input_path=base_model_path,
            output_path=quantized_path
        )
        
        logger.info(f"Quantization successful: {quantized_path}")
        return True
    
    except Exception as e:
        logger.error(f"Quantization failed: {e}")
        return False

def test_inference_script(output_path):
    """Test creation of the inference script"""
    logger.info("Testing inference script generation...")
    
    try:
        # Generate inference script
        script_path = export_onnx_inference_script(output_path)
        
        logger.info(f"Inference script generation successful: {script_path}")
        return True
    
    except Exception as e:
        logger.error(f"Inference script generation failed: {e}")
        return False

def test_full_export_pipeline(model_path, output_path, config_path):
    """Test the full export pipeline"""
    logger.info("Testing full export pipeline...")
    
    try:
        # Create exporter
        exporter = ONNXExporter(
            model_path=model_path,
            config_path=config_path
        )
        
        # Export with quantization
        output_dir = Path(output_path) / "full_pipeline"
        model_paths = exporter.export_with_quantization(
            output_dir=output_dir,
            batch_size=1,
            seq_length=64,
            quantize=True
        )
        
        # Generate inference script
        script_path = export_onnx_inference_script(output_dir)
        
        logger.info("Full export pipeline successful")
        logger.info(f"Exported models: {model_paths}")
        logger.info(f"Inference script: {script_path}")
        return True
    
    except Exception as e:
        logger.error(f"Full export pipeline failed: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing_deps = []
    
    try:
        import onnx
    except ImportError:
        missing_deps.append('onnx')
    
    try:
        import onnxruntime
    except ImportError:
        missing_deps.append('onnxruntime')
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.warning(f"Install with: pip install {' '.join(missing_deps)}")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test ONNX export functionality')
    
    parser.add_argument('--model_path', type=str, default=str(DEFAULT_MODEL_PATH),
                      help='Path to the PyTorch model')
    
    parser.add_argument('--output_path', type=str, default=str(DEFAULT_OUTPUT_PATH),
                      help='Output directory for ONNX models')
    
    parser.add_argument('--config_path', type=str, default=str(DEFAULT_CONFIG_PATH),
                      help='Path to the model configuration file')
    
    parser.add_argument('--test_basic', action='store_true', 
                      help='Test basic ONNX export')
    
    parser.add_argument('--test_quantization', action='store_true',
                      help='Test model quantization')
    
    parser.add_argument('--test_inference_script', action='store_true',
                      help='Test inference script generation')
    
    parser.add_argument('--test_full_pipeline', action='store_true',
                      help='Test the full export pipeline')
    
    parser.add_argument('--test_all', action='store_true',
                      help='Run all tests')
    
    args = parser.parse_args()
    
    # Check if the necessary dependencies are installed
    if not check_dependencies():
        logger.error("Required dependencies not installed. Please install them and try again.")
        return
    
    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track test results
    results = {}
    
    # Run selected tests
    if args.test_all or args.test_basic:
        results['basic_export'] = test_basic_export(
            args.model_path, args.output_path, args.config_path)
    
    if args.test_all or args.test_quantization:
        results['quantization'] = test_quantization(
            args.model_path, args.output_path, args.config_path)
    
    if args.test_all or args.test_inference_script:
        results['inference_script'] = test_inference_script(args.output_path)
    
    if args.test_all or args.test_full_pipeline:
        results['full_pipeline'] = test_full_export_pipeline(
            args.model_path, args.output_path, args.config_path)
    
    # If no specific tests were selected, run the basic export test
    if not (args.test_all or args.test_basic or args.test_quantization or 
           args.test_inference_script or args.test_full_pipeline):
        results['basic_export'] = test_basic_export(
            args.model_path, args.output_path, args.config_path)
    
    # Print summary
    logger.info("\nTest Results:")
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    # Overall success
    all_passed = all(results.values())
    if all_passed:
        logger.info("\n✅ All tests passed!")
    else:
        logger.info("\n❌ Some tests failed.")

if __name__ == "__main__":
    main()
