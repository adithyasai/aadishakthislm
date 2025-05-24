#!/usr/bin/env python3
"""
Code-Switching Handler for Indic Languages

This module provides functionality for detecting and processing code-switched text,
specifically focusing on English mixed with Indic languages like Hindi and Telugu.

Code-switching refers to the practice of alternating between two or more languages
within the same conversation or text. This is common in multilingual contexts like India
where speakers often mix English with their native language.
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Set, Union, Optional
import logging
from pathlib import Path
import json
import os
import pickle
from collections import Counter, defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Character sets for language detection
# Basic Devanagari Unicode range for Hindi
HINDI_UNICODE_RANGE = '\u0900-\u097F'
# Basic Telugu Unicode range
TELUGU_UNICODE_RANGE = '\u0C00-\u0C7F'
# Latin characters for English
ENGLISH_RANGE = 'A-Za-z'

# Regular expressions for language detection
HINDI_REGEX = re.compile(f'[{HINDI_UNICODE_RANGE}]')
TELUGU_REGEX = re.compile(f'[{TELUGU_UNICODE_RANGE}]')
ENGLISH_REGEX = re.compile(f'[{ENGLISH_RANGE}]')

# Dictionary of common English words that are frequently used in code-switched text
# Used to avoid treating them as out-of-vocabulary tokens
COMMON_ENGLISH_WORDS = {
    'the', 'and', 'to', 'of', 'a', 'in', 'is', 'that', 'for', 'it',
    'with', 'as', 'was', 'on', 'be', 'at', 'by', 'this', 'have', 'from',
    'or', 'not', 'but', 'what', 'all', 'were', 'when', 'we', 'there',
    'can', 'an', 'your', 'which', 'their', 'said', 'if', 'will', 'one',
    'about', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
    'him', 'her', 'would', 'make', 'like', 'time', 'no', 'just', 'know',
    'people', 'into', 'year', 'good', 'very', 'now', 'things', 'look',
    'think', 'over', 'also', 'back', 'after', 'use', 'two', 'how', 'our',
    'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because',
    'any', 'day', 'most', 'same', 'us', 'really', 'been', 'much', 'ever',
    'need', 'too', 'here', 'should', 'only', 'both', 'see', 'than', 'who',
    # Tech/internet terms commonly used in code-switching
    'mobile', 'phone', 'internet', 'online', 'computer', 'facebook',
    'twitter', 'instagram', 'whatsapp', 'google', 'email', 'website',
    'download', 'upload', 'share', 'like', 'comment', 'post', 'status',
    'message', 'chat', 'selfie', 'photo', 'video', 'cancel', 'delete',
    'app', 'application', 'id', 'password', 'login', 'logout', 'profile',
    'update', 'user', 'settings', 'bluetooth', 'wifi', 'data', 'battery',
    'charge', 'file', 'folder', 'screen', 'camera', 'call', 'contact'
}

class CodeSwitchingHandler:
    """
    Handles detection and processing of code-switched text in Indic languages
    """
    
    def __init__(self, 
               cache_dir: Optional[str] = None, 
               load_cached_stats: bool = True,
               min_word_freq: int = 2):
        """
        Initialize the code-switching handler
        
        Args:
            cache_dir: Directory for caching statistics
            load_cached_stats: Whether to load cached statistics if available
            min_word_freq: Minimum frequency for a word to be included in statistics
        """
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Statistics about code-switched text
        self.stats = {
            'word_counts': Counter(),  # Count of each word
            'language_tags': {},       # Language tag for each word
            'word_pairs': Counter()    # Count of adjacent word pairs (for transition probabilities)
        }
        
        self.min_word_freq = min_word_freq
        
        # Load cached statistics if available and requested
        if load_cached_stats and cache_dir is not None:
            self._load_cached_stats()
    
    def _load_cached_stats(self):
        """Load cached statistics if available"""
        cache_file = Path(self.cache_dir) / "code_switching_stats.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.stats = pickle.load(f)
                logger.info(f"Loaded code-switching statistics from cache: {len(self.stats['word_counts'])} words")
            except Exception as e:
                logger.warning(f"Failed to load cached statistics: {e}")
    
    def _save_cached_stats(self):
        """Save statistics to cache"""
        if self.cache_dir is None:
            return
            
        cache_file = Path(self.cache_dir) / "code_switching_stats.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.stats, f)
            logger.info(f"Saved code-switching statistics to cache: {len(self.stats['word_counts'])} words")
        except Exception as e:
            logger.warning(f"Failed to save statistics to cache: {e}")
    
    def detect_code_switching(self, text: str) -> bool:
        """
        Detect if the text contains code-switching between English and Indic languages
        
        Args:
            text: The input text to analyze
            
        Returns:
            True if the text contains code-switching, False otherwise
        """
        if not isinstance(text, str) or not text.strip():
            return False
            
        # Check for Hindi characters
        has_hindi = bool(HINDI_REGEX.search(text))
        
        # Check for Telugu characters
        has_telugu = bool(TELUGU_REGEX.search(text))
        
        # Check for English characters
        has_english = bool(ENGLISH_REGEX.search(text))
        
        # Text has code-switching if it contains both English and an Indic language
        return (has_english and (has_hindi or has_telugu))
    
    def identify_language_for_token(self, token: str) -> str:
        """
        Identify the language of a token/word
        
        Args:
            token: The token/word to analyze
            
        Returns:
            Language code ('en', 'hi', 'te', 'other', or 'mixed')
        """
        if not token or token.isspace() or token in string.punctuation:
            return 'other'
        
        # Remove punctuation from start/end of token
        clean_token = token.strip(string.punctuation)
        if not clean_token:
            return 'other'
        
        # Count characters from each language
        hindi_chars = len(HINDI_REGEX.findall(clean_token))
        telugu_chars = len(TELUGU_REGEX.findall(clean_token))
        english_chars = len(ENGLISH_REGEX.findall(clean_token))
        
        total_chars = len(clean_token)
        
        # Check if token is in common English words
        if clean_token.lower() in COMMON_ENGLISH_WORDS:
            return 'en'
        
        # Determine predominant language
        if hindi_chars / total_chars > 0.5:
            return 'hi'
        elif telugu_chars / total_chars > 0.5:
            return 'te'
        elif english_chars / total_chars > 0.5:
            return 'en'
        else:
            return 'mixed'
    
    def tag_tokens_with_language(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Tag each token with its identified language
        
        Args:
            tokens: List of tokens/words
            
        Returns:
            List of (token, language) tuples
        """
        return [(token, self.identify_language_for_token(token)) for token in tokens]
    
    def analyze_code_switching(self, text: str) -> Dict[str, Union[bool, float, Dict]]:
        """
        Analyze code-switching patterns in a text
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not isinstance(text, str) or not text.strip():
            return {'has_code_switching': False, 'stats': {}}
            
        # Split text into tokens (simple whitespace splitting for now)
        tokens = text.split()
        
        # Tag each token with its language
        tagged_tokens = self.tag_tokens_with_language(tokens)
        
        # Calculate statistics
        lang_counts = Counter([lang for _, lang in tagged_tokens])
        
        # Calculate language switch points
        switches = 0
        for i in range(1, len(tagged_tokens)):
            prev_lang = tagged_tokens[i-1][1]
            curr_lang = tagged_tokens[i][1]
            
            # Count a switch only between main languages (en, hi, te)
            if (prev_lang in ('en', 'hi', 'te') and 
                curr_lang in ('en', 'hi', 'te') and
                prev_lang != curr_lang):
                switches += 1
        
        # Calculate switch rate (switches per token)
        switch_rate = switches / len(tokens) if tokens else 0
        
        # Result dictionary
        result = {
            'has_code_switching': self.detect_code_switching(text),
            'switch_rate': switch_rate,
            'token_count': len(tokens),
            'switch_count': switches,
            'language_distribution': {k: v / len(tokens) for k, v in lang_counts.items()},
            'tagged_tokens': tagged_tokens
        }
        
        return result
    
    def update_stats_from_corpus(self, texts: List[str], 
                               primary_lang: str = 'hi') -> Dict[str, Union[int, float]]:
        """
        Update code-switching statistics from a corpus of texts
        
        Args:
            texts: List of texts to analyze
            primary_lang: Primary language of the corpus ('hi' or 'te')
            
        Returns:
            Dictionary with statistics about the corpus
        """
        code_switched_count = 0
        total_switch_rate = 0
        
        for text in texts:
            # Check if text has code-switching
            if self.detect_code_switching(text):
                code_switched_count += 1
                
                # Analyze code-switching
                analysis = self.analyze_code_switching(text)
                total_switch_rate += analysis['switch_rate']
                
                # Get tagged tokens
                tagged_tokens = analysis['tagged_tokens']
                
                # Update word counts and language tags
                for token, lang in tagged_tokens:
                    token_lower = token.lower()
                    self.stats['word_counts'][token_lower] += 1
                    self.stats['language_tags'][token_lower] = lang
                
                # Update word pair counts
                for i in range(1, len(tagged_tokens)):
                    prev_token = tagged_tokens[i-1][0].lower()
                    curr_token = tagged_tokens[i][0].lower()
                    self.stats['word_pairs'][(prev_token, curr_token)] += 1
        
        # Calculate average switch rate for code-switched texts
        avg_switch_rate = total_switch_rate / code_switched_count if code_switched_count > 0 else 0
        
        # Save updated statistics
        self._save_cached_stats()
        
        # Return corpus statistics
        return {
            'total_texts': len(texts),
            'code_switched_texts': code_switched_count,
            'code_switching_ratio': code_switched_count / len(texts) if texts else 0,
            'avg_switch_rate': avg_switch_rate
        }
    
    def get_language_specific_tokens(self, lang: str) -> Set[str]:
        """
        Get tokens specific to a particular language based on collected statistics
        
        Args:
            lang: Language code ('en', 'hi', 'te')
            
        Returns:
            Set of tokens in the specified language
        """
        return {
            word for word, tag in self.stats['language_tags'].items()
            if tag == lang and self.stats['word_counts'][word] >= self.min_word_freq
        }
    
    def tokenize_code_switched_text(self, text: str, primary_lang: str = 'hi') -> List[str]:
        """
        Specially tokenize code-switched text, being aware of language boundaries
        
        Args:
            text: The text to tokenize
            primary_lang: The primary language of the text ('hi' or 'te')
            
        Returns:
            List of tokens
        """
        if not self.detect_code_switching(text):
            # If not code-switched, use simple whitespace tokenization
            return text.split()
        
        # Perform initial whitespace tokenization
        initial_tokens = text.split()
        
        # Tag tokens with language
        tagged_tokens = self.tag_tokens_with_language(initial_tokens)
        
        # Refine tokenization based on language-specific rules
        refined_tokens = []
        
        for token, lang in tagged_tokens:
            if lang == 'en':
                # For English tokens, keep as is
                refined_tokens.append(token)
            elif lang in ('hi', 'te'):
                # For Indic language tokens, apply language-specific tokenization
                # (In a real system, we would use a more sophisticated tokenizer here)
                # For now, we'll just keep the token as is
                refined_tokens.append(token)
            elif lang == 'mixed':
                # For mixed language tokens, try to split based on character script boundaries
                # (This is a simplified approach; a real system would use more sophisticated techniques)
                parts = self._split_mixed_token(token)
                refined_tokens.extend(parts)
            else:
                # For other tokens (e.g., punctuation), keep as is
                refined_tokens.append(token)
        
        return refined_tokens
    
    def _split_mixed_token(self, token: str) -> List[str]:
        """
        Split a mixed-language token based on script boundaries
        
        Args:
            token: The token to split
            
        Returns:
            List of split tokens
        """
        # Initialize variables
        parts = []
        current_part = ""
        current_lang = None
        
        for char in token:
            # Determine character's language
            if HINDI_REGEX.match(char):
                char_lang = 'hi'
            elif TELUGU_REGEX.match(char):
                char_lang = 'te'
            elif ENGLISH_REGEX.match(char):
                char_lang = 'en'
            else:
                char_lang = 'other'
            
            # If language changes, start a new part
            if current_lang is not None and char_lang != current_lang and current_lang != 'other' and char_lang != 'other':
                if current_part:
                    parts.append(current_part)
                current_part = char
            else:
                current_part += char
            
            current_lang = char_lang
        
        # Add the last part
        if current_part:
            parts.append(current_part)
        
        return parts if parts else [token]
    
    def normalize_code_switched_text(self, text: str, primary_lang: str = 'hi') -> str:
        """
        Normalize code-switched text by applying language-specific normalization
        
        Args:
            text: The text to normalize
            primary_lang: The primary language of the text ('hi' or 'te')
            
        Returns:
            Normalized text
        """
        if not self.detect_code_switching(text):
            return text
        
        # Tokenize the text
        tokens = self.tokenize_code_switched_text(text, primary_lang)
        
        # Tag with language
        tagged_tokens = self.tag_tokens_with_language(tokens)
        
        # Normalize each token according to its language
        normalized_tokens = []
        
        for token, lang in tagged_tokens:
            if lang == 'en':
                # Simple normalization for English (lowercase)
                normalized = token.lower()
            elif lang == 'hi':
                # Apply Hindi-specific normalization
                # (In a real system, we would use a proper Hindi normalizer)
                normalized = token
            elif lang == 'te':
                # Apply Telugu-specific normalization
                # (In a real system, we would use a proper Telugu normalizer)
                normalized = token
            else:
                # For other tokens, keep as is
                normalized = token
            
            normalized_tokens.append(normalized)
        
        # Rejoin the tokens
        return ' '.join(normalized_tokens)
    
    def generate_code_switched_examples(self, text: str, target_lang: str, 
                                      switch_rate: float = 0.3) -> str:
        """
        Generate code-switched examples by replacing some words with their equivalents
        in another language.
        
        Args:
            text: Original text in a single language
            target_lang: Target language for code-switching ('en', 'hi', 'te')
            switch_rate: Desired rate of language switching
            
        Returns:
            Code-switched version of the text
        """
        # Only implemented for basic demonstration
        # In a real system, we would use parallel corpora or translation services
        # to generate authentic code-switched examples
        return text


# Utility functions for code-switching datasets
def create_code_switched_dataset(texts: List[str], 
                               summaries: List[str], 
                               primary_lang: str,
                               target_lang: str = 'en',
                               output_file: Optional[str] = None,
                               switch_rate: float = 0.3) -> List[Dict[str, str]]:
    """
    Create a dataset with code-switched examples
    
    Args:
        texts: List of original texts
        summaries: List of summaries
        primary_lang: Primary language of the texts
        target_lang: Target language for code-switching
        output_file: Optional file to save the dataset
        switch_rate: Desired rate of language switching
        
    Returns:
        List of dictionaries with code-switched texts and summaries
    """
    handler = CodeSwitchingHandler()
    result = []
    
    for text, summary in zip(texts, summaries):
        # For demonstration purposes, we're not actually generating authentic
        # code-switched text, but in a real system we would use more sophisticated methods
        code_switched_text = handler.generate_code_switched_examples(
            text, target_lang, switch_rate)
        
        code_switched_summary = handler.generate_code_switched_examples(
            summary, target_lang, switch_rate)
        
        result.append({
            "text": code_switched_text,
            "summary": code_switched_summary,
            "original_text": text,
            "original_summary": summary,
            "is_code_switched": True,
            "primary_lang": primary_lang,
            "target_lang": target_lang
        })
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in result:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return result
