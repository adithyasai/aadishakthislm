#!/usr/bin/env python3
"""
Data Augmentation for Indic Languages

This module provides language-specific data augmentation techniques for Indic languages
to enhance training data for summarization models. Key techniques include:
1. Synonym replacement
2. Back-translation
3. Paraphrasing
4. Random word deletion
5. Word swapping for languages that allow it

These techniques help improve model robustness by creating diverse variations of training examples.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
import re

# Try to import transformers for back-translation
try:
    import torch
    from transformers import MarianMTModel, MarianTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not found. Back-translation will be unavailable.")

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.hindi_morphology import HindiMorphologyAnalyzer
from src.telugu_morphology import TeluguMorphologyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map of language codes to full names
LANGUAGE_MAP = {
    'hi': 'hindi',
    'te': 'telugu',
    'bn': 'bengali',
    'ta': 'tamil',
    'ml': 'malayalam',
    'en': 'english'
}

# Common stopwords for each language
STOPWORDS = {
    'hi': ["और", "का", "एक", "में", "की", "है", "यह", "से", "हैं", "को", "पर", "इस", "होता", "कि", "जो", "कर", "मे", "गया"],
    'te': ["మరియు", "యొక్క", "లో", "ఒక", "ఉంది", "ను", "ఈ", "అది", "వారు", "ఉన్నాయి", "కు", "పై", "ఆ", "చేస్తుంది"],
    'bn': ["এবং", "একটি", "মধ্যে", "এর", "হয়", "থেকে", "হয়েছে", "এই", "করা", "করে", "একটা", "আমি", "তার", "তিনি"],
    'ta': ["மற்றும்", "ஒரு", "இல்", "உள்ள", "இருந்து", "கொண்டு", "இது", "என்று", "அந்த", "அவர்", "நான்", "என்ன"],
    'ml': ["ഒരു", "അത്", "കൂടാതെ", "ഇത്", "ആണ്", "ഉണ്ട്", "ചെയ്യുന്നു", "ചെയ്തു", "കൊണ്ട്", "എന്ന", "അവൻ", "അവൾ"],
    'en': ["the", "and", "a", "to", "of", "in", "is", "that", "for", "it", "with", "as", "was", "on", "be"]
}

class IndiaWordNet:
    """
    Simple wrapper for accessing synonyms for Indian languages.
    In a production environment, this would interface with IndoWordNet or a similar resource.
    Here, we use simple dictionaries for demonstration purposes.
    """
    def __init__(self):
        # These are small sample dictionaries for demonstration purposes
        # In a real implementation, you would load a comprehensive synonym database
        self.synonyms = {
            'hi': {
                'अच्छा': ['बढ़िया', 'उत्तम', 'शानदार', 'उम्दा'],
                'बड़ा': ['विशाल', 'महान', 'विराट', 'विस्तृत'],
                'समय': ['वक़्त', 'अवसर', 'काल', 'वेला'],
                'खुशी': ['आनंद', 'प्रसन्नता', 'हर्ष', 'उल्लास'],
                'काम': ['कार्य', 'व्यापार', 'धंधा', 'रोज़गार']
            },
            'te': {
                'మంచి': ['చక్కని', 'బాగున్న', 'ఉత్తమమైన'],
                'పెద్ద': ['విశాలమైన', 'గొప్ప', 'భారీ'],
                'సమయం': ['కాలం', 'వేళ', 'సందర్భం'],
                'ఆనందం': ['సంతోషం', 'ఉత్సాహం', 'ఉల్లాసం'],
                'పని': ['కార్యం', 'ఉద్యోగం', 'వృత్తి']
            },
            'bn': {
                'ভালো': ['উত্তম', 'সুন্দর', 'চমৎকার'],
                'বড়': ['বিশাল', 'প্রকাণ্ড', 'বিরাট'],
                'সময়': ['কাল', 'বেলা', 'অবসর'],
                'আনন্দ': ['খুশি', 'হর্ষ', 'উল্লাস'],
                'কাজ': ['কর্ম', 'ব্যবসা', 'চাকরি']
            },
            'ta': {
                'நல்ல': ['சிறந்த', 'உயர்ந்த', 'அருமையான'],
                'பெரிய': ['விசாலமான', 'பாரிய', 'மகத்தான'],
                'நேரம்': ['காலம்', 'வேளை', 'சமயம்'],
                'மகிழ்ச்சி': ['சந்தோஷம்', 'களிப்பு', 'ஆனந்தம்'],
                'வேலை': ['பணி', 'தொழில்', 'உழைப்பு']
            },
            'ml': {
                'നല്ല': ['മികച്ച', 'മനോഹരമായ', 'ഉത്തമമായ'],
                'വലുത്': ['വിശാലമായ', 'വിസ്തൃതമായ', 'വിസ്താരമുള്ള'],
                'സമയം': ['കാലം', 'അവസരം', 'വേള'],
                'സന്തോഷം': ['ആനന്ദം', 'പ്രസാദം', 'ഹർഷം'],
                'ജോലി': ['പണി', 'കർമ്മം', 'തൊഴിൽ']
            }
        }
    
    def get_synonyms(self, word: str, lang_code: str) -> List[str]:
        """
        Get synonyms for a word in the specified language
        
        Args:
            word: The word to find synonyms for
            lang_code: Language code ('hi', 'te', 'bn', 'ta', 'ml')
            
        Returns:
            List of synonyms (empty if none found)
        """
        if lang_code not in self.synonyms:
            return []
            
        return self.synonyms[lang_code].get(word, [])

class BackTranslator:
    """
    Handles back-translation for data augmentation using MarianMT models.
    Back-translation translates text to another language and back, 
    creating paraphrased versions with preserved meaning.
    """
    def __init__(self):
        self.models = {}
        self.is_initialized = False
        
        # Initialize only if transformers is available
        if TRANSFORMERS_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize translation models for supported language pairs"""
        try:
            # For Hindi
            self._load_model_pair('hi', 'en')
            
            # For Telugu (if available)
            self._load_model_pair('te', 'en')
            
            # Add more language pairs as they become available in MarianMT
            
            self.is_initialized = True
            logger.info("Back-translation models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing back-translation models: {e}")
            self.is_initialized = False
    
    def _load_model_pair(self, lang_code: str, pivot_lang: str = 'en'):
        """
        Load a pair of translation models for back-translation
        
        Args:
            lang_code: The source language code
            pivot_lang: The pivot language (usually English)
        """
        lang_name = LANGUAGE_MAP.get(lang_code, lang_code)
        pivot_name = LANGUAGE_MAP.get(pivot_lang, pivot_lang)
        
        # Define model names based on language codes
        forward_model_name = f"Helsinki-NLP/opus-mt-{lang_name}-{pivot_name}"
        backward_model_name = f"Helsinki-NLP/opus-mt-{pivot_name}-{lang_name}"
        
        try:
            # Check if models exist in HuggingFace
            from huggingface_hub import model_info
            try:
                model_info(forward_model_name)
                model_info(backward_model_name)
            except Exception:
                logger.warning(f"Translation models for {lang_name}-{pivot_name} not found. Skipping.")
                return
            
            # Load models
            logger.info(f"Loading translation models for {lang_code}-{pivot_lang}")
            
            # Forward direction (lang -> pivot)
            self.models[f"{lang_code}_to_{pivot_lang}_tokenizer"] = MarianTokenizer.from_pretrained(forward_model_name)
            self.models[f"{lang_code}_to_{pivot_lang}_model"] = MarianMTModel.from_pretrained(forward_model_name)
            
            # Backward direction (pivot -> lang)
            self.models[f"{pivot_lang}_to_{lang_code}_tokenizer"] = MarianTokenizer.from_pretrained(backward_model_name)
            self.models[f"{pivot_lang}_to_{lang_code}_model"] = MarianMTModel.from_pretrained(backward_model_name)
            
            logger.info(f"Successfully loaded translation models for {lang_code}-{pivot_lang}")
            
        except Exception as e:
            logger.error(f"Error loading models for {lang_code}-{pivot_lang}: {e}")
    
    def back_translate(self, text: str, lang_code: str, pivot_lang: str = 'en') -> str:
        """
        Perform back-translation on the input text
        
        Args:
            text: The text to back-translate
            lang_code: The source language code
            pivot_lang: The pivot language (usually English)
            
        Returns:
            Back-translated text
        """
        if not TRANSFORMERS_AVAILABLE or not self.is_initialized:
            logger.warning("Back-translation unavailable. Returning original text.")
            return text
            
        # Check if we have the required models
        required_models = [
            f"{lang_code}_to_{pivot_lang}_tokenizer",
            f"{lang_code}_to_{pivot_lang}_model",
            f"{pivot_lang}_to_{lang_code}_tokenizer", 
            f"{pivot_lang}_to_{lang_code}_model"
        ]
        
        if not all(model in self.models for model in required_models):
            logger.warning(f"Translation models for {lang_code}-{pivot_lang} not available. Returning original text.")
            return text
        
        try:
            # Translate to pivot language (e.g., English)
            forward_tokenizer = self.models[f"{lang_code}_to_{pivot_lang}_tokenizer"]
            forward_model = self.models[f"{lang_code}_to_{pivot_lang}_model"]
            
            inputs = forward_tokenizer(text, return_tensors="pt", padding=True)
            translated = forward_model.generate(**inputs)
            pivot_text = forward_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Translate back to original language
            backward_tokenizer = self.models[f"{pivot_lang}_to_{lang_code}_tokenizer"]
            backward_model = self.models[f"{pivot_lang}_to_{lang_code}_model"]
            
            inputs = backward_tokenizer(pivot_text, return_tensors="pt", padding=True)
            translated = backward_model.generate(**inputs)
            back_translated = backward_tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return back_translated
            
        except Exception as e:
            logger.error(f"Error during back-translation: {e}")
            return text

class DataAugmentor:
    """
    Handles data augmentation techniques for Indic languages
    """
    def __init__(self):
        """Initialize the data augmentor with required resources"""
        self.wordnet = IndiaWordNet()
        self.back_translator = BackTranslator()
        
        # Initialize morphology analyzers
        self.hindi_analyzer = HindiMorphologyAnalyzer()
        self.telugu_analyzer = TeluguMorphologyAnalyzer()
        
        logger.info("Data augmentor initialized")
    
    def augment(self, text: str, lang_code: str, techniques: List[str] = None, 
                augmentation_count: int = 1) -> List[str]:
        """
        Augment input text using selected techniques
        
        Args:
            text: Source text to augment
            lang_code: Language code ('hi', 'te', 'bn', 'ta', 'ml')
            techniques: List of techniques to apply ('synonym', 'backtranslate', 
                        'paraphrase', 'deletion', 'swap')
            augmentation_count: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        if techniques is None:
            techniques = ['synonym', 'deletion', 'swap']
            
        results = []
        
        for _ in range(augmentation_count):
            # Choose a random technique from the specified ones
            technique = random.choice(techniques)
            
            if technique == 'synonym':
                augmented = self._synonym_replacement(text, lang_code)
            elif technique == 'backtranslate':
                augmented = self.back_translator.back_translate(text, lang_code)
            elif technique == 'deletion':
                augmented = self._random_deletion(text, lang_code)
            elif technique == 'swap':
                augmented = self._random_swap(text, lang_code)
            elif technique == 'paraphrase':
                # For paraphrasing we use a combination of techniques
                augmented = self._synonym_replacement(text, lang_code)
                augmented = self._random_swap(augmented, lang_code)
            else:
                logger.warning(f"Unknown augmentation technique: {technique}")
                augmented = text
                
            # Ensure we got a different text
            if augmented != text and augmented not in results:
                results.append(augmented)
            
        # If we couldn't generate enough unique augmentations, add the original
        if len(results) < augmentation_count:
            results.append(text)
            
        return results[:augmentation_count]
        
    def _tokenize(self, text: str, lang_code: str) -> List[str]:
        """
        Simple space-based tokenization with special handling for Indic languages
        
        Args:
            text: Text to tokenize
            lang_code: Language code
            
        Returns:
            List of tokens
        """
        # For languages using Devanagari or similar scripts, we need specialized tokenization
        if lang_code == 'hi':
            # Use the morphology analyzer for better tokenization
            if hasattr(self, 'hindi_analyzer'):
                return self.hindi_analyzer.tokenize(text)
        elif lang_code == 'te':
            # Use the Telugu morphology analyzer
            if hasattr(self, 'telugu_analyzer'):
                return self.telugu_analyzer.tokenize(text)
        
        # Default tokenization for other languages
        # This is simplified; in practice, you'd use a specialized tokenizer for each language
        return text.split()
    
    def _detokenize(self, tokens: List[str], lang_code: str) -> str:
        """
        Join tokens back into text with appropriate spacing for the language
        
        Args:
            tokens: List of tokens to join
            lang_code: Language code
            
        Returns:
            Joined text
        """
        # For most languages, simple joining with spaces works
        return " ".join(tokens)
        
    def _synonym_replacement(self, text: str, lang_code: str, replace_ratio: float = 0.3) -> str:
        """
        Replace words with their synonyms
        
        Args:
            text: Source text
            lang_code: Language code
            replace_ratio: Proportion of words to replace (0.0-1.0)
            
        Returns:
            Text with synonyms replaced
        """
        tokens = self._tokenize(text, lang_code)
        
        # Skip stopwords and only consider content words for replacement
        stopwords = set(STOPWORDS.get(lang_code, []))
        
        # Identify candidate words for replacement (not stopwords)
        candidates = [(i, token) for i, token in enumerate(tokens) if token.lower() not in stopwords]
        
        # Determine how many words to replace
        num_to_replace = max(1, int(len(candidates) * replace_ratio))
        num_to_replace = min(num_to_replace, len(candidates))
        
        # Randomly select words to replace
        replace_indices = random.sample(range(len(candidates)), num_to_replace)
        
        # Replace selected words with synonyms
        for idx in replace_indices:
            i, word = candidates[idx]
            synonyms = self.wordnet.get_synonyms(word, lang_code)
            
            if synonyms:
                tokens[i] = random.choice(synonyms)
        
        return self._detokenize(tokens, lang_code)
    
    def _random_deletion(self, text: str, lang_code: str, del_ratio: float = 0.1) -> str:
        """
        Randomly delete words from the text
        
        Args:
            text: Source text
            lang_code: Language code
            del_ratio: Proportion of words to delete (0.0-1.0)
            
        Returns:
            Text with words deleted
        """
        tokens = self._tokenize(text, lang_code)
        
        # Ensure we don't delete too many words
        if len(tokens) <= 3:
            return text
            
        # Only delete non-stopwords to preserve grammaticality
        stopwords = set(STOPWORDS.get(lang_code, []))
        
        # Keep all stopwords and randomly delete some content words
        kept_tokens = []
        for token in tokens:
            # Always keep stopwords, randomly keep content words
            if token.lower() in stopwords or random.random() > del_ratio:
                kept_tokens.append(token)
        
        # If we deleted too many words, ensure we have at least half of the original
        if len(kept_tokens) < len(tokens) / 2:
            # Add random words back
            while len(kept_tokens) < len(tokens) / 2:
                add_idx = random.randint(0, len(tokens) - 1)
                if tokens[add_idx] not in kept_tokens:
                    kept_tokens.append(tokens[add_idx])
            
            # Re-sort tokens to maintain original order
            kept_indices = [i for i, token in enumerate(tokens) if token in kept_tokens]
            kept_tokens = [tokens[i] for i in sorted(kept_indices)]
        
        return self._detokenize(kept_tokens, lang_code)
    
    def _random_swap(self, text: str, lang_code: str, swap_count: int = 2) -> str:
        """
        Randomly swap adjacent words
        
        Args:
            text: Source text
            lang_code: Language code
            swap_count: Number of swaps to perform
            
        Returns:
            Text with words swapped
        """
        tokens = self._tokenize(text, lang_code)
        
        # For languages with strict word order, be more careful with swapping
        strict_order_languages = ['en']
        
        # Adjust swap count based on text length
        max_swaps = min(swap_count, len(tokens) // 2)
        
        # For strict word order languages, reduce swaps further
        if lang_code in strict_order_languages:
            max_swaps = max(1, max_swaps // 2)
        
        # Perform the swaps
        for _ in range(max_swaps):
            if len(tokens) <= 1:
                break
                
            # Select two adjacent positions to swap
            pos = random.randint(0, len(tokens) - 2)
            tokens[pos], tokens[pos + 1] = tokens[pos + 1], tokens[pos]
        
        return self._detokenize(tokens, lang_code)

def augment_dataset(input_file: str, output_file: str, lang_code: str, 
                    techniques: List[str] = None, samples_per_item: int = 2):
    """
    Augment an entire dataset and save results
    
    Args:
        input_file: Path to input dataset (JSONL format)
        output_file: Path to save augmented dataset
        lang_code: Language code
        techniques: List of augmentation techniques
        samples_per_item: Number of augmentations per input item
    """
    logger.info(f"Augmenting dataset: {input_file}")
    
    # Initialize augmentor
    augmentor = DataAugmentor()
    
    # Default techniques
    if techniques is None:
        techniques = ['synonym', 'deletion', 'swap']
    
    # Read input data
    input_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    input_data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return
    
    logger.info(f"Loaded {len(input_data)} samples from {input_file}")
    
    # Augment data
    augmented_data = []
    for item in input_data:
        # Keep original example
        augmented_data.append(item)
        
        # Get fields to augment (typically 'text' or 'document' and 'summary')
        text_field = 'document' if 'document' in item else 'text'
        summary_field = 'summary'
        
        if text_field in item:
            # Augment document text
            augmented_texts = augmentor.augment(
                item[text_field], lang_code, techniques, samples_per_item
            )
            
            # Create new examples with augmented text
            for i, aug_text in enumerate(augmented_texts):
                new_item = item.copy()
                new_item[text_field] = aug_text
                new_item['augmented'] = True
                new_item['augmentation_technique'] = techniques
                augmented_data.append(new_item)
                
            logger.info(f"Augmented sample with {len(augmented_texts)} variations")
    
    # Write output data
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in augmented_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(augmented_data)} samples to {output_file}")
    except Exception as e:
        logger.error(f"Error writing output file: {e}")

def main():
    """Command-line interface for data augmentation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data augmentation for Indic languages")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input dataset file (JSONL format)")
    parser.add_argument("--output", "-o", type=str, required=True, 
                        help="Output augmented dataset file")
    parser.add_argument("--lang", "-l", type=str, default="hi", 
                        choices=["hi", "te", "bn", "ta", "ml", "en"],
                        help="Language code")
    parser.add_argument("--techniques", "-t", type=str, default="synonym,deletion,swap",
                        help="Comma-separated list of augmentation techniques")
    parser.add_argument("--samples", "-s", type=int, default=2,
                        help="Number of augmented samples to generate per input")
    
    args = parser.parse_args()
    
    # Parse techniques
    techniques = args.techniques.split(',')
    
    # Perform augmentation
    augment_dataset(
        args.input,
        args.output,
        args.lang,
        techniques,
        args.samples
    )

if __name__ == "__main__":
    main()
