#!/usr/bin/env python3
"""
Simple test to verify the ONNX export module imports correctly.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, '.')

try:
    from src.onnx_export import ONNXExporter, export_onnx_inference_script
    print("Import successful!")
    print("ONNX export module is working correctly.")
except Exception as e:
    print(f"Import failed with error: {e}")
    sys.exit(1)

print("All checks passed!")