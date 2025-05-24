import os
import sys
import torch
import logging
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import model modules
from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer
from src.language_adapters import detect_language
from src.hindi_morphology import HindiMorphologyAnalyzer
from src.telugu_morphology import TeluguMorphologyAnalyzer
from src.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config paths
CONFIG_DIR = Path(__file__).parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.json"

def test_language_enhanced_model():
    """
    Test language-enhanced model with both adapters and morphological analysis.
    This demonstrates the combined capabilities of language adapters and morphological analyzers.
    """
    logger.info("Testing language-enhanced model...")
    
    # Initialize tokenizer
    tokenizer = IndicTokenizer(str(DEFAULT_CONFIG_PATH))
    
    # Initialize morphological analyzers
    hindi_analyzer = HindiMorphologyAnalyzer()
    telugu_analyzer = TeluguMorphologyAnalyzer()
    
    # Create a config with language adapters enabled
    config = IndicSLMConfig(
        vocab_size=10000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=128,
        use_adapters=True,
        adapter_size=32,
        languages=["hi", "te"]
    )
    
    # Initialize model with adapter config
    model = IndicSLM(config)
    
    # Test texts in Hindi and Telugu
    test_texts = [
        # Hindi examples
        "भारत एक विविधतापूर्ण देश है जहां कई भाषाएँ बोली जाती हैं।",
        "हिंदी भारत की राष्ट्रभाषा है और देश के विभिन्न हिस्सों में बोली जाती है।",
        
        # Telugu examples
        "తెలుగు భాష చాలా అందమైనది మరియు దక్షిణ భారతదేశంలో ప్రధానంగా మాట్లాడబడుతుంది.",
        "ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో తెలుగు అధికారిక భాష."
    ]
    
    logger.info("\n" + "="*50)
    logger.info("DEMONSTRATION: LANGUAGE-SPECIFIC PROCESSING PIPELINE")
    logger.info("="*50)
    
    for text in test_texts:
        logger.info(f"\nInput text: {text}")
        
        # Step 1: Detect language
        lang = detect_language(text)
        logger.info(f"Detected language: {lang}")
        
        # Step 2: Apply morphological analysis
        analyzer = hindi_analyzer if lang == "hi" else telugu_analyzer
        morphology = analyzer.analyze_text(text)
        
        logger.info("Morphological analysis:")
        for idx, word_analysis in enumerate(morphology[:3]):  # Show first 3 words only
            logger.info(f"  Word {idx+1}: {word_analysis['word']}")
            logger.info(f"    Stem: {word_analysis['stem']}")
            logger.info(f"    Segments: {word_analysis['segments']}")
            
            # Show additional features
            for key, value in word_analysis.items():
                if key not in ['word', 'stem', 'segments']:
                    logger.info(f"    {key}: {value}")
        
        if len(morphology) > 3:
            logger.info(f"  ... and {len(morphology) - 3} more words")
        
        # Step 3: Tokenize the text
        encoded = tokenizer.encode(text)
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
        attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long)
        
        logger.info(f"Tokenized ({len(encoded['input_ids'])} tokens)")
        
        # Step 4: Pass through model with language-specific adapters
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                language_id=lang  # Pass the detected language to activate language adapters
            )
        
        # Step 5: Display results
        logger.info(f"Successfully processed with language adapter for {lang}")
        logger.info(f"Output logits shape: {outputs['logits'].shape}")
        logger.info("-"*50)
    
    return True

def test_end_to_end():
    """
    Test an end-to-end pipeline with a sample dataset to demonstrate
    using language adapters and morphological analysis together.
    """
    logger.info("\n" + "="*50)
    logger.info("DEMONSTRATION: END-TO-END PROCESSING PIPELINE")
    logger.info("="*50)
    
    # Initialize data processor with morphology enabled
    processor = DataProcessor(str(DEFAULT_CONFIG_PATH), enable_morphology=True)
    
    # Create a tiny sample dataset with both Hindi and Telugu text
    sample_data = [
        {"text": "भारत एक विविधतापूर्ण देश है। यहां अनेक भाषाएँ बोली जाती हैं।", "lang": "hi", "summary": "भारत में भाषाई विविधता है।"},
        {"text": "హైదరాబాద్ తెలంగాణ రాష్ట్రంలో ఉంది. ఇది ఒక ప్రముఖ నగరం.", "lang": "te", "summary": "హైదరాబాద్ తెలంగాణ ప్రధాన నగరం."}
    ]
    
    # Process each sample with appropriate language analysis
    for item in sample_data:
        text = item["text"]
        lang = item["lang"]
        summary = item["summary"]
        
        logger.info(f"\nProcessing {lang} text: {text}")
        
        # Apply normalization
        normalized_text = processor.normalize_text(text, lang)
        logger.info(f"Normalized: {normalized_text}")
        
        # Get tokenized segments
        segmented_tokens = processor.get_segmented_tokens(text, lang)
        logger.info(f"Segmented tokens (first 3): {segmented_tokens[:3]}")
        
        # Get morphological analysis
        morph_analysis = processor.analyze_morphology(text, lang)
        
        if morph_analysis:
            logger.info(f"Stems: {morph_analysis.get('stems', [])[:5]}")
            logger.info(f"Root forms: {morph_analysis.get('root_forms', [])[:5]}")
            
            # Display morphological features for the first few words
            features = morph_analysis.get('morphology', [])
            for i, feature in enumerate(features[:3]):
                logger.info(f"  Word {i+1} features: {feature}")
        
        logger.info(f"Original Summary: {summary}")
        logger.info("-"*50)
    
    logger.info("\nEnd-to-end pipeline test completed.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Test language-enhanced model")
    parser.add_argument("--model_only", action="store_true", help="Test only the language-enhanced model")
    parser.add_argument("--end_to_end", action="store_true", help="Test the end-to-end pipeline")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--test_all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run selected tests or all tests
    run_all = args.test_all or (not args.model_only and not args.end_to_end)
    
    if args.model_only or run_all:
        test_language_enhanced_model()
    
    if args.end_to_end or run_all:
        test_end_to_end()

if __name__ == "__main__":
    main()