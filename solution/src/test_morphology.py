"""
Test for both Hindi and Telugu morphological analyzers.

This module tests the integration of both morphological analyzers
and demonstrates their capabilities on sample text.
"""

import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.hindi_morphology import HindiMorphologyAnalyzer
from src.telugu_morphology import TeluguMorphologyAnalyzer

def test_hindi_morphology():
    """Test Hindi morphological analyzer with sample text."""
    print("\n" + "="*80)
    print("HINDI MORPHOLOGICAL ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    hindi_analyzer = HindiMorphologyAnalyzer()
    
    # Sample Hindi text
    sample_text = "मैं हिंदी भाषा में बात कर रहा हूँ। यह एक सुंदर भाषा है।"
    print("Sample text:", sample_text)
    
    # Perform analysis
    print("\nWORD ANALYSIS:")
    print("-"*80)
    analyses = hindi_analyzer.analyze_text(sample_text)
    
    for analysis in analyses:
        word = analysis.get('word', '')
        stem = analysis.get('stem', '')
        pos = analysis.get('pos', '')
        
        # Format additional features
        features = []
        for key, value in analysis.items():
            if key not in ['word', 'stem', 'segments', 'pos']:
                features.append(f"{key}={value}")
        
        feature_str = ", ".join(features)
        print(f"Word: {word:<15} Stem: {stem:<15} POS: {pos:<10} Features: {feature_str}")
    
    # Get root forms
    print("\nROOT FORMS:")
    print("-"*80)
    roots = hindi_analyzer.get_root_forms(sample_text)
    print("Root forms:", " ".join(roots))
    
    # Test segmentation on specific words
    print("\nMORPHEME SEGMENTATION:")
    print("-"*80)
    test_words = ["लड़कियों", "खेलते", "जाएँगे", "किताबों"]
    
    for word in test_words:
        segments = hindi_analyzer.segment_word(word)
        print(f"Word: {word:<15} Segments: {' + '.join(segments)}")

def test_telugu_morphology():
    """Test Telugu morphological analyzer with sample text."""
    print("\n" + "="*80)
    print("TELUGU MORPHOLOGICAL ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    telugu_analyzer = TeluguMorphologyAnalyzer()
    
    # Sample Telugu text
    sample_text = "నేను తెలుగులో మాట్లాడుతున్నాను. ఇది చాలా అందమైన భాష."
    print("Sample text:", sample_text)
    
    # Perform analysis
    print("\nWORD ANALYSIS:")
    print("-"*80)
    analyses = telugu_analyzer.analyze_text(sample_text)
    
    for analysis in analyses:
        word = analysis.get('word', '')
        stem = analysis.get('stem', '')
        pos = analysis.get('pos', '')
        
        # Format additional features
        features = []
        for key, value in analysis.items():
            if key not in ['word', 'stem', 'segments', 'pos']:
                features.append(f"{key}={value}")
        
        feature_str = ", ".join(features)
        print(f"Word: {word:<20} Stem: {stem:<20} POS: {pos:<10} Features: {feature_str}")
    
    # Get root forms
    print("\nROOT FORMS:")
    print("-"*80)
    roots = telugu_analyzer.get_root_forms(sample_text)
    print("Root forms:", " ".join(roots))
    
    # Test segmentation on specific words
    print("\nMORPHEME SEGMENTATION:")
    print("-"*80)
    test_words = ["పుస్తకాలు", "చేస్తున్నాడు", "పిల్లవాడికి", "ఇంటినుండి"]
    
    for word in test_words:
        segments = telugu_analyzer.segment_word(word)
        print(f"Word: {word:<20} Segments: {' + '.join(segments)}")
    
    # Test grammatical feature extraction
    print("\nGRAMMATICAL FEATURES:")
    print("-"*80)
    features = telugu_analyzer.get_grammatical_features(sample_text)
    for feature_name, values in features.items():
        if values:
            print(f"{feature_name}: {', '.join(values)}")
    
    # Test lemmatization
    print("\nLEMMATIZATION:")
    print("-"*80)
    lemmatized = telugu_analyzer.lemmatize_text("పిల్లలు పాఠశాలకు వెళ్తున్నారు")
    print("Original: పిల్లలు పాఠశాలకు వెళ్తున్నారు")
    print("Lemmatized:", lemmatized)

def test_comparison():
    """Compare Hindi and Telugu analyzers with similar constructs."""
    print("\n" + "="*80)
    print("COMPARING HINDI AND TELUGU ANALYSIS")
    print("="*80)
    
    # Initialize analyzers
    hindi_analyzer = HindiMorphologyAnalyzer()
    telugu_analyzer = TeluguMorphologyAnalyzer()
    
    # Test cases: word, meaning
    hindi_words = [
        ("मैं", "I"),
        ("मेरा", "my"),
        ("लड़का", "boy"),
        ("लड़की", "girl"),
        ("जाता है", "goes"),
        ("देखेंगे", "will see")
    ]
    
    telugu_words = [
        ("నేను", "I"),
        ("నా", "my"),
        ("అబ్బాయి", "boy"),
        ("అమ్మాయి", "girl"),
        ("వెళ్తున్నాడు", "goes"),
        ("చూస్తారు", "will see")
    ]
    
    print("\nHINDI WORDS:")
    print("-"*80)
    for word, meaning in hindi_words:
        analysis = hindi_analyzer.analyze_word(word)
        stem = analysis.get('stem', '')
        pos = analysis.get('pos', '')
        features = []
        for key, value in analysis.items():
            if key not in ['word', 'stem', 'segments', 'pos']:
                features.append(f"{key}={value}")
        
        feature_str = ", ".join(features)
        print(f"Word: {word:<15} Meaning: {meaning:<10} Stem: {stem:<15} POS: {pos:<10} Features: {feature_str}")
    
    print("\nTELUGU WORDS:")
    print("-"*80)
    for word, meaning in telugu_words:
        analysis = telugu_analyzer.analyze_word(word)
        stem = analysis.get('stem', '')
        pos = analysis.get('pos', '')
        features = []
        for key, value in analysis.items():
            if key not in ['word', 'stem', 'segments', 'pos']:
                features.append(f"{key}={value}")
        
        feature_str = ", ".join(features)
        print(f"Word: {word:<15} Meaning: {meaning:<10} Stem: {stem:<15} POS: {pos:<10} Features: {feature_str}")

if __name__ == "__main__":
    print("Testing Morphological Analyzers")
    print("=" * 80)
    
    # Test Hindi morphology
    test_hindi_morphology()
    
    # Test Telugu morphology
    test_telugu_morphology()
    
    # Compare Hindi and Telugu analysis
    test_comparison()
    
    print("\nAll tests completed successfully!")
