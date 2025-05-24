#!/usr/bin/env python3
"""
Test integration of code-switching with the data processor.
"""
import sys
import os
from pathlib import Path
import json

# Add project directory to path
project_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(project_dir))

# Import the data processor
from src.data_processor import DataProcessor

def main():
    print("Testing code-switching integration with data processor...")
    
    # Create test data
    test_data_file = project_dir / "code_switching_test_data.jsonl"
    
    # Sample code-switched texts
    samples = [
        # Hindi-English
        {"text": "मैंने अपना homework complete कर लिया है।", "summary": "Homework completed", "id": "1"},
        {"text": "उसने mujhe call किया था लेकिन मैं busy था।", "summary": "Missed call", "id": "2"},
        # Telugu-English
        {"text": "నేను నా homework పూర్తి చేశాను.", "summary": "Homework done", "id": "3"},
        {"text": "అతను నాకు call చేశాడు కానీ నేను busy గా ఉన్నాను.", "summary": "Call missed", "id": "4"},
        # Pure Hindi
        {"text": "मैंने अपना गृहकार्य पूरा कर लिया है।", "summary": "गृहकार्य पूरा", "id": "5"},
        # Pure Telugu
        {"text": "నేను నా గృహపని పూర్తి చేశాను.", "summary": "గృహపని పూర్తి", "id": "6"}
    ]
    
    # Save test data to file
    with open(test_data_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Created test data file with {len(samples)} examples")
    
    # Initialize the data processor with code-switching enabled
    processor = DataProcessor(
        enable_code_switching=True,
        code_switching_cache_dir=str(project_dir / "cache")
    )
    
    print("Processing Hindi samples...")
    # Process test data for Hindi
    hindi_dataset = processor.prepare_dataset(
        str(test_data_file),
        lang='hi'
    )
    
    print(f"Processed Hindi dataset size: {len(hindi_dataset)}")
    print("Dataset columns:", hindi_dataset.column_names)
    
    # Check code-switching analysis results
    if 'is_code_switched' in hindi_dataset.column_names:
        code_switched = hindi_dataset.filter(lambda example: example['is_code_switched'] == True)
        print(f"Detected {len(code_switched)} code-switched examples out of {len(hindi_dataset)}")
        
        # Print some details
        for i, example in enumerate(hindi_dataset):
            print(f"Example {i+1}: {example['text']}")
            if 'is_code_switched' in example and example['is_code_switched']:
                print(f"  Code-switched: Yes")
                if 'cs_switch_rate' in example:
                    print(f"  Switch rate: {example['cs_switch_rate']:.3f}")
                if 'cs_language_distribution' in example:
                    print(f"  Language distribution: {example['cs_language_distribution']}")
            else:
                print(f"  Code-switched: No")
            print("-" * 50)
    else:
        print("Code-switching analysis not found in processed dataset")
    
    # Clean up test data
    if test_data_file.exists():
        os.remove(test_data_file)
        print("Cleaned up test data file")

if __name__ == "__main__":
    main()
    print("Code-switching integration test completed!")
