"""
Simple test utility for the data augmentation module.
Run this script to verify the implementation.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

import logging
from src.data_augmentation import DataAugmentor
from src.data_processor import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_augmentation():
    """Test the data augmentation functionality"""
    logger.info("Testing data augmentation module...")
    
    # Test sample texts
    test_texts = {
        'hi': "भारत एक विशाल देश है जहां विविधता में एकता देखने को मिलती है। यहां कई भाषाएं, धर्म और संस्कृतियां हैं।",
        'te': "భారతదేశం చాలా భాషలు మరియు సంస్కృతులు కలిగిన దేశం. వైవిధ్యంలో ఏకత్వం చూపిస్తుంది.",
        'en': "India is a diverse country with many languages and cultures. It shows unity in diversity."
    }
    
    # Test DataAugmentor directly
    augmentor = DataAugmentor()
    
    for lang, text in test_texts.items():
        logger.info(f"Testing {lang} augmentation:")
        logger.info(f"Original: {text}")
        
        # Test each method
        for technique in ['synonym', 'deletion', 'swap']:
            augmented = augmentor.augment(text, lang, [technique], 1)
            logger.info(f"Technique '{technique}': {augmented[0]}")
        logger.info("---")
    
    # Test DataProcessor integration
    logger.info("Testing DataProcessor integration:")
    processor = DataProcessor(enable_augmentation=True)
    
    for lang, text in test_texts.items():
        logger.info(f"DataProcessor augmentation for {lang}:")
        augmented = processor.augment_text(text, lang, ['synonym', 'deletion', 'swap'], 2)
        for i, aug in enumerate(augmented):
            logger.info(f"Augmented #{i+1}: {aug}")
        logger.info("---")
    
    logger.info("Data augmentation testing complete!")

if __name__ == "__main__":
    test_data_augmentation()
