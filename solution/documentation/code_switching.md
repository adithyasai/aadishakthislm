# Code-Switching Implementation for SLM Project

This document provides an overview of the code-switching implementation for the Summarization Language Model (SLM) project focusing on Indic languages.

## Overview

Code-switching refers to the practice of alternating between two or more languages within a single conversation or text. This is particularly common in multilingual environments like India, where speakers often mix English with local languages like Hindi, Telugu, and others.

The implementation provides the following capabilities:

1. **Detection** of code-switched text (particularly English mixed with Hindi or Telugu)
2. **Analysis** of code-switching patterns, including:
   - Switch rates
   - Language distribution
   - Token-level language identification
3. **Special tokenization** for code-switched text that respects language boundaries
4. **Normalization** techniques adapted for mixed-language content
5. **Integration** with the data processing pipeline

## Files and Components

The code-switching implementation consists of the following components:

1. **`code_switching.py`** - Core implementation with the `CodeSwitchingHandler` class
2. **`data_processor.py`** - Integration with the data processing pipeline
3. **Test files**:
   - `test_code_switching.py` - Comprehensive test suite for code-switching functionality
   - `simple_code_switching_test.py` - Simple demonstration of core features
   - `test_code_switching_integration.py` - Tests integration with the data processor

## Using Code-Switching Features

### Basic Usage

```python
from src.code_switching import CodeSwitchingHandler

handler = CodeSwitchingHandler()

text = "मैंने अपना homework complete कर लिया है।"  # Hindi-English example
is_code_switched = handler.detect_code_switching(text)
analysis = handler.analyze_code_switching(text)

print(f"Is code-switched: {is_code_switched}")
print(f"Switch rate: {analysis['switch_rate']}")
print(f"Language distribution: {analysis['language_distribution']}")
```

### Integration with Data Processor

The code-switching functionality is integrated into the `DataProcessor` class and can be enabled during initialization:

```python
from src.data_processor import DataProcessor

processor = DataProcessor(
    enable_code_switching=True,
    code_switching_cache_dir="path/to/cache"  # Optional
)

# Process a dataset with code-switching awareness
dataset = processor.prepare_dataset("path/to/dataset.jsonl", lang="hi")

# Create code-switched training examples
augmented_dataset = processor.create_code_switched_training_examples(
    dataset,
    lang="hi",
    target_lang="en",
    ratio=0.2  # Add 20% more examples with code-switching
)
```

## Key Features

### Language Detection

The system can detect code-switching between:
- English and Hindi
- English and Telugu

Detection is based on character scripts/Unicode ranges and contextual analysis.

### Code-Switching Analysis

For code-switched text, the system provides:
- **Switch rate**: The frequency of language transitions per token
- **Language distribution**: The proportion of tokens in each language
- **Token-level tagging**: Each token is tagged with its identified language

### Special Tokenization

For code-switched text, tokenization preserves language boundaries and handles cases where:
- A single word contains multiple scripts
- Standard tokenization would break meaningful language units

### Normalization

Text normalization for code-switched content applies:
- Language-specific normalization for each language segment
- Special handling for transitions between languages

### Creating Code-Switched Examples

The system can generate synthetic code-switched examples by:
- Replacing selected terms with equivalents in another language
- Maintaining natural switch points between languages
- Preserving grammatical structure

## Limitations

1. The system primarily focuses on English-Hindi and English-Telugu code-switching
2. Sophisticated mixed-language parsing is not fully implemented
3. Generation of authentic code-switched examples is simplified

## Future Improvements

1. Expand to more Indic languages beyond Hindi and Telugu
2. Improve tokenization at morpheme boundaries between languages
3. Add more sophisticated language model-based generation of code-switched examples
4. Implement specialized embedding handling for code-switched tokens

## Testing

Run the tests to verify code-switching functionality:

```bash
# Run detection tests
python src/test_code_switching.py --test_detection

# Run tokenization tests
python src/test_code_switching.py --test_tokenization

# Run integration tests
python src/test_code_switching_integration.py

# Run all tests
python src/test_code_switching.py --test_all
```
