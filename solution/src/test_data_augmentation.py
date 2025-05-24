#!/usr/bin/env python3
"""
Test script for data augmentation techniques

This script tests the data augmentation functionality for Indic languages
by running various augmentation techniques on sample texts and evaluating
the results.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_augmentation import DataAugmentor, augment_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample texts for testing in different languages
SAMPLE_TEXTS = {
    'hi': [
        "भारत एक विशाल देश है जहां विविधता में एकता देखने को मिलती है। यहां कई भाषाएं, धर्म और संस्कृतियां हैं।",
        "आज का मौसम बहुत सुहावना है। आसमान में बादल छाए हुए हैं और हल्की बारिश हो रही है।",
        "प्रौद्योगिकी के विकास ने हमारे जीवन को बहुत आसान बना दिया है। स्मार्टफोन और इंटरनेट के कारण दुनिया एक छोटे गांव में बदल गई है।"
    ],
    'te': [
        "భారతదేశం చాలా భాషలు మరియు సంస్కృతులు కలిగిన దేశం. వైవిధ్యంలో ఏకత్వం చూపిస్తుంది.",
        "ప్రకృతి అందాలతో నిండిన ప్రదేశాలు భారతదేశంలో చాలా ఉన్నాయి. అటవీ ప్రాంతాలు, పర్వతాలు, నదులు, సముద్రతీరాలు.",
        "స్మార్ట్‌ఫోన్లు మరియు ఇంటర్నెట్ వచ్చిన తర్వాత ప్రపంచం మారిపోయింది. తక్కువ సమయంలో ఎక్కువ పని చేయగలుగుతున్నాము."
    ],
    'en': [
        "India is a diverse country with many languages and cultures. It shows unity in diversity.",
        "The rapid development of technology has significantly changed our lives in the past decade.",
        "Artificial intelligence and machine learning are transforming industries across the globe."
    ]
}

def create_test_dataset(output_path: str):
    """
    Create a test dataset with samples from different languages
    
    Args:
        output_path: Path to save the test dataset
    """
    dataset = []
    
    for lang, texts in SAMPLE_TEXTS.items():
        for i, text in enumerate(texts):
            item = {
                "id": f"{lang}_{i}",
                "text": text,
                "language": lang,
                "summary": f"Sample summary for {lang} text {i}"
            }
            dataset.append(item)
    
    # Save to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Created test dataset with {len(dataset)} samples at {output_path}")
    return dataset

def test_augmentation_techniques():
    """Test individual augmentation techniques on sample texts"""
    augmentor = DataAugmentor()
    
    results = {}
    
    for lang, texts in SAMPLE_TEXTS.items():
        logger.info(f"Testing augmentation techniques for language: {lang}")
        
        lang_results = {}
        
        # Test each technique
        for technique in ['synonym', 'deletion', 'swap', 'paraphrase']:
            technique_results = []
            
            for text in texts:
                augmented = augmentor.augment(
                    text, 
                    lang_code=lang, 
                    techniques=[technique], 
                    augmentation_count=2
                )
                technique_results.append({
                    "original": text,
                    "augmented": augmented
                })
            
            lang_results[technique] = technique_results
        
        # Test back-translation if available
        if hasattr(augmentor.back_translator, 'is_initialized') and augmentor.back_translator.is_initialized:
            bt_results = []
            
            for text in texts:
                augmented = augmentor.augment(
                    text, 
                    lang_code=lang, 
                    techniques=['backtranslate'], 
                    augmentation_count=1
                )
                bt_results.append({
                    "original": text,
                    "augmented": augmented
                })
            
            lang_results['backtranslate'] = bt_results
            
        results[lang] = lang_results
    
    return results

def test_dataset_augmentation(data_dir: Path):
    """Test augmentation on a complete dataset"""
    # Create test dataset
    test_dataset_path = data_dir / "test_augmentation_input.jsonl"
    output_path = data_dir / "test_augmentation_output.jsonl"
    
    # Create test dataset
    create_test_dataset(test_dataset_path)
    
    # Augment for each language
    for lang in SAMPLE_TEXTS.keys():
        lang_output = data_dir / f"test_augmentation_output_{lang}.jsonl"
        
        # Augment with all techniques
        augment_dataset(
            test_dataset_path,
            lang_output,
            lang_code=lang,
            techniques=['synonym', 'deletion', 'swap', 'paraphrase'],
            samples_per_item=2
        )
        
        logger.info(f"Dataset augmentation test completed for {lang}")
        
        # Count samples
        sample_count = 0
        with open(lang_output, 'r', encoding='utf-8') as f:
            for _ in f:
                sample_count += 1
        
        logger.info(f"Augmented dataset contains {sample_count} samples")

def display_augmentation_results(results: Dict):
    """Display augmentation test results in a readable format"""
    for lang, lang_results in results.items():
        print(f"\n=== LANGUAGE: {lang.upper()} ===")
        
        for technique, examples in lang_results.items():
            print(f"\n-- Technique: {technique} --")
            
            for i, example in enumerate(examples):
                print(f"\nExample {i+1}:")
                print(f"Original: {example['original']}")
                print(f"Augmented: {example['augmented'][0]}")
                
                if len(example['augmented']) > 1:
                    print(f"Augmented #2: {example['augmented'][1]}")

def main():
    """Run the data augmentation tests"""
    parser = argparse.ArgumentParser(description="Test data augmentation techniques")
    parser.add_argument("--techniques", action="store_true", help="Test individual augmentation techniques")
    parser.add_argument("--dataset", action="store_true", help="Test dataset augmentation")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no specific tests are requested, run all
    if not (args.techniques or args.dataset):
        args.all = True
    
    # Create data directory if needed
    data_dir = Path(__file__).parent.parent / "data" / "augmentation_test"
    data_dir.mkdir(exist_ok=True, parents=True)
    
    # Run requested tests
    if args.techniques or args.all:
        logger.info("Testing individual augmentation techniques")
        results = test_augmentation_techniques()
        display_augmentation_results(results)
    
    if args.dataset or args.all:
        logger.info("Testing dataset augmentation")
        test_dataset_augmentation(data_dir)

if __name__ == "__main__":
    main()
