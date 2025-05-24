#!/usr/bin/env python3
"""
Simple test to verify code-switching functionality
"""
import sys
import os
from pathlib import Path

# Add project directory to path
project_dir = Path(os.path.abspath(__file__)).parent.parent
sys.path.insert(0, str(project_dir))

# Import the code-switching module
from src.code_switching import CodeSwitchingHandler

def main():
    # Initialize the code-switching handler
    handler = CodeSwitchingHandler()
    
    # Sample code-switched texts
    samples = [
        "मैंने अपना homework complete कर लिया है।",
        "నేను నా homework పూర్తి చేశాను.",
        "मैंने अपना गृहकार्य पूरा कर लिया है।",
        "नमस्ते, my name is Amit. मैं दिल्ली से हूँ।"
    ]
    
    for i, sample in enumerate(samples):
        # Detect if the text contains code-switching
        is_code_switched = handler.detect_code_switching(sample)
        
        # Print results
        print(f"Sample {i+1}: {sample}")
        print(f"Is code-switched: {is_code_switched}")
        if is_code_switched:
            # Get language-tagged tokens
            tokens = sample.split()
            tagged = handler.tag_tokens_with_language(tokens)
            for token, lang in tagged:
                print(f"  {token}: {lang}")
        print("-" * 50)

if __name__ == "__main__":
    main()
    print("Simple test completed successfully!")
