#!/usr/bin/env python3
"""
Test code-switching functionality in the data processing pipeline.

This script validates the code-switching detection, analysis, and processing
capabilities implemented in the SLM project.
"""
import sys
import os
from pathlib import Path
import logging
import argparse
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add project directory to path
project_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(project_dir))

# Import project modules
from src.data_processor import DataProcessor
from src.code_switching import CodeSwitchingHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample test data with code-switching
CODE_SWITCHED_SAMPLES = [
    # Hindi-English
    "मैंने अपना homework complete कर लिया है।",
    "उसने mujhe call किया था लेकिन मैं busy था।",
    "इस problem का solution क्या है?",
    "मैं office से घर आ गया हूँ।",
    "आज का weather बहुत अच्छा है।",
    
    # Telugu-English
    "నేను నా homework పూర్తి చేశాను.",
    "అతను నాకు call చేశాడు కానీ నేను busy గా ఉన్నాను.",
    "ఈ problem కి solution ఏమిటి?",
    "నేను office నుండి ఇంటికి వచ్చాను.",
    "ఈరోజు weather చాలా బాగుంది.",
    
    # Pure Hindi
    "मैंने अपना गृहकार्य पूरा कर लिया है।",
    "उसने मुझे फोन किया था लेकिन मैं व्यस्त था।",
    
    # Pure Telugu
    "నేను నా గృహపని పూర్తి చేశాను.",
    "అతను నాకు ఫోన్ చేశాడు కానీ నేను బిజీగా ఉన్నాను."
]

def verify_code_switching_detection(handler: CodeSwitchingHandler):
    """Test the code-switching detection capability"""
    logger.info("Testing code-switching detection...")
    
    # Test each sample
    results = []
    for sample in CODE_SWITCHED_SAMPLES:
        is_code_switched = handler.detect_code_switching(sample)
        analysis = handler.analyze_code_switching(sample)
        
        # Store results
        results.append({
            'text': sample,
            'is_code_switched': is_code_switched,
            'switch_rate': analysis['switch_rate'],
            'language_distribution': analysis['language_distribution'],
            'token_count': analysis['token_count'],
            'switch_count': analysis['switch_count']
        })
    
    # Print results
    for result in results:
        logger.info(f"Text: {result['text']}")
        logger.info(f"  Is code-switched: {result['is_code_switched']}")
        logger.info(f"  Switch rate: {result['switch_rate']:.3f}")
        logger.info(f"  Language distribution: {result['language_distribution']}")
        logger.info(f"  Tokens: {result['token_count']}, Switches: {result['switch_count']}")
        logger.info("---")
    
    # Calculate accuracy (assuming first 10 are code-switched, last 4 are not)
    true_labels = [True] * 10 + [False] * 4
    predicted_labels = [result['is_code_switched'] for result in results]
    
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    accuracy = correct / len(true_labels)
    
    logger.info(f"Detection accuracy: {accuracy:.2f}")
    return accuracy > 0.85  # At least 85% accuracy

def test_tokenization(handler: CodeSwitchingHandler):
    """Test special tokenization for code-switched text"""
    logger.info("Testing code-switching tokenization...")
    
    # Test on code-switched examples
    for sample in CODE_SWITCHED_SAMPLES[:10]:  # First 10 are code-switched
        # Compare normal tokenization vs. code-switched tokenization
        normal_tokens = sample.split()
        cs_tokens = handler.tokenize_code_switched_text(sample, 'hi' if 'मैंने' in sample else 'te')
        
        logger.info(f"Text: {sample}")
        logger.info(f"  Normal tokens: {normal_tokens}")
        logger.info(f"  CS tokens: {cs_tokens}")
        logger.info("---")

def test_data_processor_integration():
    """Test the integration of code-switching in the data processor"""
    logger.info("Testing data processor integration...")
    
    # Create test data
    test_data_file = project_dir / "test_data_cs.jsonl"
    with open(test_data_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(CODE_SWITCHED_SAMPLES):
            entry = {
                'text': sample,
                'summary': f"Test summary {i+1}",
                'id': f"test_{i+1}"
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Initialize processor with code-switching enabled
    processor = DataProcessor(
        enable_morphology=False,
        enable_augmentation=False,
        enable_code_switching=True,
        code_switching_cache_dir=str(project_dir / "cache")
    )
    
    # Process Hindi samples
    hindi_dataset = processor.prepare_dataset(
        str(test_data_file),
        lang='hi'
    )
    
    logger.info(f"Processed dataset size: {len(hindi_dataset)}")
    
    # Check if code-switching analysis was added
    if 'is_code_switched' in hindi_dataset.column_names:
        code_switched_count = sum(1 for val in hindi_dataset['is_code_switched'] if val)
        logger.info(f"Detected {code_switched_count} code-switched examples")
    else:
        logger.warning("No code-switching analysis found in processed dataset")
    
    # Cleanup
    if test_data_file.exists():
        os.remove(test_data_file)
    
    return 'is_code_switched' in hindi_dataset.column_names

def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test code-switching functionality")
    parser.add_argument("--test_detection", action="store_true", help="Test code-switching detection")
    parser.add_argument("--test_tokenization", action="store_true", help="Test code-switching tokenization")
    parser.add_argument("--test_integration", action="store_true", help="Test integration with data processor")
    parser.add_argument("--test_all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Initialize the code-switching handler
    handler = CodeSwitchingHandler(cache_dir=str(project_dir / "cache"))
    
    if args.test_all or args.test_detection:
        verify_code_switching_detection(handler)
    
    if args.test_all or args.test_tokenization:
        test_tokenization(handler)
    
    if args.test_all or args.test_integration:
        test_data_processor_integration()
    
    if not (args.test_all or args.test_detection or args.test_tokenization or args.test_integration):
        # No specific test requested, run all
        success_detection = verify_code_switching_detection(handler)
        test_tokenization(handler)
        success_integration = test_data_processor_integration()
        
        if success_detection and success_integration:
            logger.info("✅ All code-switching tests passed!")
        else:
            logger.warning("⚠️ Some code-switching tests failed.")
    
if __name__ == "__main__":
    main()
