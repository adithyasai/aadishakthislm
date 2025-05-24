# ONNX Export Documentation

## Overview

The ONNX export functionality enables the deployment of SLM models across different platforms and runtime environments by converting PyTorch models to the ONNX format.

## Features

- **Model Conversion**: Export PyTorch models to ONNX format
- **Dynamic Shapes**: Support for variable-length inputs with dynamic axes
- **Quantization**: Options for 8-bit dynamic quantization to reduce model size
- **Verification**: Tools to verify the correctness of exported models
- **Inference Helpers**: Automatic generation of inference scripts

## Usage

### Basic Export

To export a model to ONNX format:

```bash
python src/onnx_export.py --model_path models/final_model --output models/onnx_model
```

### Export with Quantization

To export a model with quantization:

```bash
python src/onnx_export.py --model_path models/final_model --output models/onnx_model
```

By default, quantization is enabled. To disable it:

```bash
python src/onnx_export.py --model_path models/final_model --output models/onnx_model --no_quantize
```

### Custom Batch Size and Sequence Length

For specific batch size and sequence length:

```bash
python src/onnx_export.py --model_path models/final_model --output models/onnx_model --batch_size 4 --seq_length 256
```

### Generate Inference Script

To generate a Python script for inference:

```bash
python src/onnx_export.py --model_path models/final_model --output models/onnx_model --create_inference_script
```

## Inference Example

After exporting the model, you can use the following code for inference:

```python
import onnxruntime as ort
import numpy as np
from src.tokenizer import IndicTokenizer

# Load the tokenizer
tokenizer = IndicTokenizer(config_path="configs/model_config.json")

# Load the ONNX model
session = ort.InferenceSession("models/onnx_model/model.onnx")

# Prepare input text
text = "This is an example text to summarize using the exported ONNX model."
inputs = tokenizer.encode(text)
input_ids = np.array([inputs["input_ids"]], dtype=np.int64)
attention_mask = np.array([inputs["attention_mask"]], dtype=np.int64)

# Run inference
onnx_inputs = {
    'input_ids': input_ids,
    'attention_mask': attention_mask
}
outputs = session.run(None, onnx_inputs)

# Process output (example for summarization)
output_ids = np.argmax(outputs[0], axis=-1)
summary = tokenizer.decode(output_ids[0])
print(f"Summary: {summary}")
```

## Benefits of ONNX Export

- **Cross-Platform Compatibility**: Deploy to mobile, edge, and server environments
- **Optimization**: Take advantage of platform-specific optimizations for better performance
- **Size Reduction**: Reduce model size through quantization
- **Integration**: Easier integration with production inference systems

## Supported Platforms

The exported ONNX models can be used on various platforms:

- **Mobile Devices**: Android and iOS with ONNX Runtime
- **Web Browsers**: JavaScript applications using ONNX.js
- **Edge Devices**: IoT and embedded systems
- **Cloud Services**: Optimized inference in production environments
- **Desktop Applications**: Cross-platform applications

## Implementation Details

The implementation is contained in the following files:

- `src/onnx_export.py`: Main module for ONNX export
- `src/example_onnx_export.py`: Example script demonstrating export usage
- `src/test_onnx_export.py`: Test suite for ONNX export functionality

The export process consists of:

1. Loading the PyTorch model
2. Creating dummy inputs for tracing
3. Exporting to ONNX format with dynamic axes
4. Verifying the exported model
5. Optional quantization for smaller model size
6. Output verification by comparing PyTorch and ONNX results
