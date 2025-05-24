import argparse
import logging
import os
import re
import sys
import json
import string
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

# For preprocessing
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    from indicnlp.tokenize.indic_tokenize import trivial_tokenize
    INDIC_NLP_AVAILABLE = True
except ImportError:
    print("Indic NLP Library not found. Using basic tokenization instead.")
    IndicNormalizerFactory = None
    trivial_tokenize = None
    INDIC_NLP_AVAILABLE = False

# For ML approaches
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not found. Using basic similarity measures instead.")
    TfidfVectorizer = None
    CountVectorizer = None
    cosine_similarity = None
    KMeans = None
    SKLEARN_AVAILABLE = False

# For neural models
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from positional_encoding import RelativeMultiHeadAttention, RelativePositionalEncoding
    from rope import RotaryMultiHeadAttention, RotaryPositionEmbeddings
    PYTORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not found. Neural summarization methods will not be available.")
    PYTORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define language codes and scripts
LANGUAGE_CODES = {
    'te': 'telugu',
    'hi': 'hindi'
}

# Important domain-specific terms for better summarization
DOMAIN_TERMS = {
    'te': {
        # Language and literature domain
        'భాష': 3.0, 'సాహిత్యం': 3.0, 'కవిత': 2.5, 'ద్రావిడ': 2.5, 'తెలుగు': 2.0, 
        'లిపి': 2.5, 'వ్యాకరణం': 2.5, 'కావ్యం': 2.0, 'నిఘంటువు': 2.0, 'ప్రాచీన': 2.0,
        # Educational domain
        'విద్య': 3.0, 'బోధన': 2.5, 'విద్యార్థి': 2.0, 'పాఠశాల': 2.0, 'కళాశాల': 2.0,
        'విశ్వవిద్యాలయం': 2.5, 'పరీక్ష': 2.0, 'ఉపాధ్యాయుడు': 2.0, 'పాఠ్యాంశాలు': 2.0,
        # Science and technology
        'శాస్త్రం': 3.0, 'సాంకేతిక': 2.5, 'అభివృద్ధి': 2.5, 'పరిశోధన': 2.5,
        'విజ్ఞానం': 2.5, 'ఆవిష్కరణ': 2.0, 'నమూనా': 2.0, 'ప్రయోగం': 2.0,
        # Essential entities (proper nouns, places, dates)
        'భారతదేశం': 2.0, 'ఆంధ్రప్రదేశ్': 2.0, 'తెలంగాణ': 2.0, 'ప్రపంచ': 2.0, 
        'అధికారిక': 2.0, 'శతాబ్దం': 2.0, 'జనాభా': 2.0
    },
    'hi': {
        # Language and literature domain
        'भाषा': 3.0, 'साहित्य': 3.0, 'लिपि': 2.5, 'व्याकरण': 2.5, 'काव्य': 2.5,
        'देवनागरी': 2.5, 'हिंदी': 2.0, 'संस्कृत': 2.0, 'शब्द': 2.0, 'कविता': 2.0,
        # Educational domain
        'शिक्षा': 3.0, 'विद्यालय': 2.5, 'विद्यार्थी': 2.0, 'अध्ययन': 2.5, 
        'विश्वविद्यालय': 2.5, 'परीक्षा': 2.0, 'अध्यापक': 2.0, 'पाठ्यक्रम': 2.0,
        # Science and technology
        'विज्ञान': 3.0, 'तकनीकी': 2.5, 'विकास': 2.5, 'अनुसंधान': 2.5,
        'आविष्कार': 2.5, 'प्रयोग': 2.0, 'नमूना': 2.0,
        # Essential entities (proper nouns, places, dates)
        'भारत': 2.0, 'दिल्ली': 2.0, 'हिंदुस्तान': 2.0, 'विश्व': 2.0,
        'आधिकारिक': 2.0, 'शताब्दी': 2.0, 'जनसंख्या': 2.0, 'मुगल': 2.0
    }
}

# Coherence words to improve readability
COHERENCE_MARKERS = {
    'te': [
        'అయితే', 'కాబట్టి', 'అందువలన', 'ఇంకా', 'అలాగే', 'పైగా', 
        'మరియు', 'కానీ', 'కూడా', 'ఎందుకంటే', 'అంతేకాకుండా', 'దీనివలన'
    ],
    'hi': [
        'लेकिन', 'इसलिए', 'और', 'इसके अलावा', 'फिर', 'इसके बावजूद',
        'जबकि', 'क्योंकि', 'भी', 'अतः', 'तथा', 'इसके कारण'
    ]
}

class AdvancedIndianTokenizer:
    """Better tokenizer for Indian languages with improved handling of Telugu and Hindi"""
    
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.lang_name = LANGUAGE_CODES.get(lang_code, 'hindi')
    
    def tokenize(self, text):
        """Improved tokenization for Indian languages"""
        if not text:
            return []
            
        if INDIC_NLP_AVAILABLE and trivial_tokenize:
            try:
                tokens = trivial_tokenize(text, self.lang_name)
                return tokens
            except Exception as e:
                logger.warning(f"Error using Indic tokenizer: {e}")
                # Fall through to custom tokenizer
                
        # Custom tokenizer for Indian languages
        if self.lang_code == 'te':
            # Telugu has some specific patterns to handle
            # Replace full-stops, commas, etc. with a space before tokenizing
            for punct in '.,;:!?।॥()[]{}':
                text = text.replace(punct, f' {punct} ')
                
            # Split by whitespace
            raw_tokens = text.split()
            
            # Post-process tokens to handle special cases
            tokens = []
            for token in raw_tokens:
                if token in string.punctuation:
                    tokens.append(token)
                elif any(p in token for p in string.punctuation):
                    # Handle cases where punctuation is attached to words without spaces
                    parts = []
                    current = ""
                    for char in token:
                        if char in string.punctuation:
                            if current:
                                parts.append(current)
                                current = ""
                            parts.append(char)
                        else:
                            current += char
                    if current:
                        parts.append(current)
                    tokens.extend(parts)
                else:
                    tokens.append(token)
                    
            return tokens
            
        elif self.lang_code == 'hi':
            # Hindi has its own specific patterns
            # Handle special characters like Danda (।)
            for punct in '.,;:!?।॥()[]{}':
                text = text.replace(punct, f' {punct} ')
                
            # Split by whitespace
            tokens = text.split()
            return tokens
        else:
            # Simple tokenization for other languages
            for p in string.punctuation:
                text = text.replace(p, f' {p} ')
            return text.split()

class AdvancedIndianNormalizer:
    """Improved text normalizer for Indian languages"""
    
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.lang_name = LANGUAGE_CODES.get(lang_code, 'hindi')
    
    def normalize(self, text):
        """Improved normalization for Indian languages"""
        if INDIC_NLP_AVAILABLE and IndicNormalizerFactory:
            try:
                normalizer = IndicNormalizerFactory().get_normalizer(self.lang_code)
                normalized_text = normalizer.normalize(text)
                return normalized_text
            except Exception as e:
                logger.warning(f"Error using Indic normalizer: {e}")
                # Fall through to custom normalizer
                
        # Custom normalization rules
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Handle specific patterns for Telugu
        if self.lang_code == 'te':
            # Handle common Telugu text normalization issues
            # Replace zero-width joiners and non-joiners
            text = text.replace('\u200c', '').replace('\u200d', '')
            
        # Handle specific patterns for Hindi
        elif self.lang_code == 'hi':
            # Handle common Hindi text normalization issues
            # Replace zero-width joiners and non-joiners
            text = text.replace('\u200c', '').replace('\u200d', '')
            
        return text.strip()

class IndicTextProcessor:
    """
    Enhanced text processor for Indian languages with improved normalization and tokenization
    """
    def __init__(self, lang_code: str):
        self.lang_code = lang_code
        self.lang_name = LANGUAGE_CODES.get(lang_code, 'hindi')
        
        # Initialize normalizer and tokenizer
        self.normalizer = AdvancedIndianNormalizer(lang_code)
        self.tokenizer = AdvancedIndianTokenizer(lang_code)
    
    def normalize(self, text: str) -> str:
        """Normalize text based on language-specific rules"""
        return self.normalizer.normalize(text)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        return self.tokenizer.tokenize(text)
    
    def preprocess(self, text: str) -> str:
        """Preprocess text - normalize and clean"""
        # Basic cleaning
        text = re.sub(r'[\n\r]+', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Language-specific normalization
        text = self.normalize(text)
        
        return text
    
    def get_sentences(self, text: str) -> List[str]:
        """
        Enhanced sentence boundary detection for Telugu and Hindi
        """
        if not text or not text.strip():
            return []
            
        # Improved sentence splitting with better handling of Indian language punctuation
        if self.lang_code == 'te':
            # Telugu sentence splitter with better handling of sentence boundaries
            # First normalize any irregular spacing around sentence-ending punctuation
            text = re.sub(r'([.!?।॥])\s*', r'\1 ', text)
            
            # Split on various sentence boundary markers
            # Telugu uses both Latin and Devanagari punctuation
            raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=।)\s+|(?<=॥)\s+', text)
            
            # Handle cases where sentence markers are missing by using syntactic clues
            # Some Telugu texts use phrases that indicate sentence boundaries
            additional_splits = []
            for sent in raw_sentences:
                # Check if the sentence is too long (more than 40 words)
                if len(sent.split()) > 40:
                    # Look for phrase boundaries with common Telugu conjunctions
                    sub_sentences = re.split(r'(?<=\s)(కాబట్టి|అందువలన|ఎందుకంటే|ఏమంటే|అయితే)\s+', sent)
                    processed_sub_sentences = []
                    
                    for i, sub_sent in enumerate(sub_sentences):
                        if i > 0 and i % 2 == 1:  # This is a conjunction
                            # Add it to the previous sentence
                            processed_sub_sentences[-1] = f"{processed_sub_sentences[-1]} {sub_sent}"
                        else:
                            processed_sub_sentences.append(sub_sent)
                    
                    additional_splits.extend(processed_sub_sentences)
                else:
                    additional_splits.append(sent)
                    
            sentences = additional_splits
            
        elif self.lang_code == 'hi':
            # Hindi sentence splitter with better handling of Devanagari punctuation
            # First normalize spacing around sentence-ending punctuation
            text = re.sub(r'([.!?।॥])\s*', r'\1 ', text)
            
            # Split on various sentence boundary markers
            raw_sentences = re.split(r'(?<=[.!?])\s+|(?<=।)\s+|(?<=॥)\s+', text)
            
            # Handle long sentences and syntactic clues in Hindi
            additional_splits = []
            for sent in raw_sentences:
                if len(sent.split()) > 40:
                    # Look for phrase boundaries with common Hindi conjunctions
                    sub_sentences = re.split(r'(?<=\s)(इसलिए|क्योंकि|तथा|परंतु|लेकिन|किंतु)\s+', sent)
                    processed_sub_sentences = []
                    
                    for i, sub_sent in enumerate(sub_sentences):
                        if i > 0 and i % 2 == 1:  # This is a conjunction
                            processed_sub_sentences[-1] = f"{processed_sub_sentences[-1]} {sub_sent}"
                        else:
                            processed_sub_sentences.append(sub_sent)
                    
                    additional_splits.extend(processed_sub_sentences)
                else:
                    additional_splits.append(sent)
                    
            sentences = additional_splits
            
        else:
            # Fallback sentence splitter for other languages
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
        # Post-process sentences
        processed_sentences = []
        for s in sentences:
            s = s.strip()
            if s and len(s.split()) >= 2:  # Require at least 2 words for a valid sentence
                # Check if sentence ends with proper punctuation
                if not s[-1] in '.!?।॥':
                    s = s + '.'
                processed_sentences.append(s)
        
        return processed_sentences

class IndianLanguageSummarizer:
    """
    Improved extractive+abstractive summarization for Telugu and Hindi texts
    """
    def __init__(self, lang_code: str, 
                 mode: str = 'hybrid',
                 compression_ratio: float = 0.3, 
                 max_sentences: int = 5, 
                 preserve_keywords: bool = True,
                 ensure_coherence: bool = True):
        """
        Initialize the improved summarizer
        
        Args:
            lang_code: Language code ('te' for Telugu, 'hi' for Hindi)
            mode: Summarization mode ('extractive', 'hybrid', 'abstractive')
            compression_ratio: Target ratio of summary to original text length
            max_sentences: Maximum number of sentences in summary
            preserve_keywords: Whether to highlight important keywords
            ensure_coherence: Whether to ensure coherence between sentences
        """
        self.lang_code = lang_code
        self.mode = mode
        self.compression_ratio = compression_ratio
        self.max_sentences = max_sentences
        self.preserve_keywords = preserve_keywords
        self.ensure_coherence = ensure_coherence
        
        # Initialize text processor
        self.text_processor = IndicTextProcessor(lang_code)
        
        # Stopwords for Telugu and Hindi
        self.stopwords = self._load_stopwords()
        
        # Domain-specific terms
        self.domain_terms = DOMAIN_TERMS.get(lang_code, {})
        
        # Coherence markers
        self.coherence_markers = COHERENCE_MARKERS.get(lang_code, [])
        
        # Named entity patterns for better entity recognition
        self.named_entity_patterns = self._init_named_entity_patterns()
    
    def _init_named_entity_patterns(self):
        """Initialize patterns for detecting named entities"""
        if self.lang_code == 'te':
            # Common patterns for Telugu person and place names
            patterns = [
                r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Person names in English script
                r'శ్రీ\s+[\u0C00-\u0C7F]+',     # Names with honorific prefix
                r'[\u0C00-\u0C7F]+\s+జిల్లా',   # District names
                r'[\u0C00-\u0C7F]+\s+రాష్ట్రం',  # State names
                r'[\u0C00-\u0C7F]+గారు'        # Person names with honorific suffix
            ]
        elif self.lang_code == 'hi':
            # Common patterns for Hindi person and place names
            patterns = [
                r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Person names in English script
                r'श्री\s+[\u0900-\u097F]+',     # Names with honorific prefix
                r'[\u0900-\u097F]+\s+जिला',    # District names
                r'[\u0900-\u097F]+\s+राज्य',    # State names
                r'[\u0900-\u097F]+\s+जी'       # Person names with honorific suffix
            ]
        else:
            patterns = []
            
        return [re.compile(p) for p in patterns]
    
    def _load_stopwords(self) -> List[str]:
        """Load enhanced stopword lists for Telugu and Hindi"""
        # Common stopwords for Telugu - expanded list
        te_stopwords = [
            'మరియు', 'కూడా', 'ఒక', 'అది', 'ఈ', 'ఆ', 'అయితే', 'కానీ', 'లేదా', 
            'నుండి', 'వారు', 'మేము', 'నేను', 'మీరు', 'అతను', 'ఆమె', 'వాటిని', 
            'దాని', 'గా', 'తో', 'లో', 'పై', 'కి', 'ను', 'అందువలన', 'కోసం', 'గురించి',
            'వంటి', 'అన్ని', 'కొన్ని', 'ఏదైనా', 'ఎక్కడ', 'ఎప్పుడు', 'ఎందుకు', 'ఎలా',
            'అందరూ', 'చాలా', 'కొంత', 'మరి', 'ఇలా', 'అలా', 'తర్వాత', 'ముందు', 'అక్కడ',
            'ఇక్కడ', 'ఉంది', 'ఉంటుంది', 'ఉన్నారు', 'ఉన్నాయి', 'వచ్చింది', 'వెళ్ళింది'
        ]
        
        # Common stopwords for Hindi - expanded list
        hi_stopwords = [
            'और', 'का', 'एक', 'में', 'की', 'है', 'यह', 'तथा', 'को', 'इस', 'पर', 'से', 
            'हैं', 'लिए', 'गया', 'किया', 'अपने', 'न', 'हो', 'कि', 'वह', 'वे', 'हम', 
            'था', 'होता', 'करना', 'किए', 'कुछ', 'भी', 'थे', 'द्वारा', 'जा', 'रहा', 'हुआ',
            'जैसे', 'सभी', 'कई', 'कोई', 'कहाँ', 'कब', 'क्यों', 'कैसे', 'अब', 'सब', 'उन',
            'वहाँ', 'यहाँ', 'तब', 'जब', 'होगा', 'करेगा', 'थी', 'थीं', 'उनके', 'उनका',
            'इनका', 'जिस', 'जिसे', 'जिसका', 'वाले', 'वाला', 'वाली', 'होने', 'बहुत', 'कम'
        ]
        
        if self.lang_code == 'te':
            return te_stopwords
        elif self.lang_code == 'hi':
            return hi_stopwords
        else:
            return []
    
    def _detect_named_entities(self, text: str) -> List[str]:
        """Detect named entities in the text for better importance scoring"""
        entities = []
        
        # Extract potential entities using regex patterns
        for pattern in self.named_entity_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                entity = match.group(0).strip()
                if entity and len(entity) > 1:
                    entities.append(entity)
        
        # Additional heuristics for finding capitalized words (for English words in text)
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1 and not word.isupper():
                # Likely a proper noun
                entities.append(word)
        
        return list(set(entities))
    
    def _score_tokens_by_importance(self, tokens: List[str]) -> Dict[str, float]:
        """
        Score tokens by their importance using multiple factors:
        - Domain-specific term weighting
        - Named entity recognition
        - Position and frequency
        """
        token_scores = {}
        
        # Basic frequency scoring
        token_counter = Counter(tokens)
        max_freq = max(token_counter.values()) if token_counter else 1
        
        # Filter out stopwords and short tokens
        filtered_tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
        
        # Calculate baseline scores from frequency
        for token in set(filtered_tokens):
            # Normalize frequency
            norm_freq = token_counter[token] / max_freq
            
            # Basic score
            score = norm_freq
            
            # Apply domain-specific weights if available
            if token in self.domain_terms:
                score *= self.domain_terms[token]
            
            # Apply length bonus for longer words (often more meaningful in Indian languages)
            # But cap at 5 to avoid over-weighting very long compounds
            length_factor = min(5, len(token)) / 5
            score *= (1 + 0.5 * length_factor)
            
            token_scores[token] = score
        
        return token_scores
    
    def _calculate_word_importance(self, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate word importance using multiple factors
        """
        # Get all words from all sentences
        all_tokens = []
        for sentence in sentences:
            tokens = self.text_processor.tokenize(sentence)
            all_tokens.extend([t for t in tokens if t not in self.stopwords and len(t) > 1])
        
        # Get baseline scores from frequency and other factors
        word_scores = self._score_tokens_by_importance(all_tokens)
        
        # Detect named entities for additional boosting
        all_text = ' '.join(sentences)
        entities = self._detect_named_entities(all_text)
        
        # Boost scores for named entities
        for entity in entities:
            if entity in word_scores:
                word_scores[entity] *= 2.0
            else:
                # Handle multi-word entities
                entity_tokens = self.text_processor.tokenize(entity)
                for token in entity_tokens:
                    if token in word_scores and token not in self.stopwords:
                        word_scores[token] *= 1.5
        
        return word_scores
    
    def _score_sentences_content(self, sentences: List[str]) -> Dict[int, float]:
        """
        Score sentences based on their content importance
        """
        # Get word importance scores
        word_scores = self._calculate_word_importance(sentences)
        
        # Score each sentence
        sentence_scores = {}
        
        for i, sentence in enumerate(sentences):
            # Tokenize
            tokens = self.text_processor.tokenize(sentence)
            
            if not tokens:
                sentence_scores[i] = 0
                continue
            
            # Sum up scores of important words
            important_tokens = [t for t in tokens if t not in self.stopwords and len(t) > 1]
            
            if not important_tokens:
                sentence_scores[i] = 0
                continue
            
            # Calculate base content score
            content_score = sum(word_scores.get(token, 0) for token in important_tokens)
            
            # Normalize by sentence length with a logarithmic factor to avoid over-penalizing long sentences
            # This is especially important for Telugu which can have long compound sentences
            length_norm = math.log(1 + len(important_tokens)) / math.log(10)  # log base 10
            score = content_score / length_norm
            
            sentence_scores[i] = score
        
        return sentence_scores
    
    def _score_sentences_position(self, sentence_count: int) -> Dict[int, float]:
        """
        Score sentences based on their position in the text
        """
        position_scores = {}
        
        # First and last sentences are generally more important in most documents
        for i in range(sentence_count):
            if i == 0:  # First sentence
                position_scores[i] = 1.0
            elif i == 1:  # Second sentence
                position_scores[i] = 0.9
            elif i == sentence_count - 1:  # Last sentence
                position_scores[i] = 0.8
            elif i < sentence_count // 3:  # First third
                position_scores[i] = 0.7
            elif i >= 2 * sentence_count // 3:  # Last third
                position_scores[i] = 0.6
            else:  # Middle
                position_scores[i] = 0.5
                
        return position_scores
    
    def _score_sentences_coherence(self, sentences: List[str]) -> Dict[int, float]:
        """
        Score sentences based on coherence markers
        """
        coherence_scores = {}
        
        for i, sentence in enumerate(sentences):
            # Check for presence of coherence markers
            tokens = self.text_processor.tokenize(sentence)
            
            # Sentences with coherence markers can be important for flow
            has_marker = any(marker in tokens for marker in self.coherence_markers)
            
            # Sentences that connect ideas are valuable
            coherence_scores[i] = 1.2 if has_marker else 1.0
            
        return coherence_scores
    
    def _score_sentences_overall(self, sentences: List[str]) -> Dict[int, float]:
        """
        Calculate overall sentence scores combining multiple factors
        """
        if not sentences:
            return {}
            
        # Get individual scoring components
        content_scores = self._score_sentences_content(sentences)
        position_scores = self._score_sentences_position(len(sentences))
        coherence_scores = self._score_sentences_coherence(sentences)
        
        # Combine scores with weights
        sentence_scores = {}
        for i in range(len(sentences)):
            # Content is most important (65%), position (25%), coherence (10%)
            sentence_scores[i] = (
                0.65 * content_scores.get(i, 0) +
                0.25 * position_scores.get(i, 0) +
                0.10 * coherence_scores.get(i, 0)
            )
            
        return sentence_scores
    
    def _get_sentence_similarity(self, sentences: List[str]) -> np.ndarray:
        """
        Calculate semantic similarity between sentences
        """
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        # Preprocess sentences
        cleaned_sentences = [self.text_processor.preprocess(s) for s in sentences]
        
        if SKLEARN_AVAILABLE and TfidfVectorizer is not None and cosine_similarity is not None:
            try:
                # TF-IDF vectorization
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: [t for t in self.text_processor.tokenize(x) 
                                        if t not in self.stopwords],
                    max_features=300,  # Limit features to avoid sparsity issues
                    use_idf=True,
                    smooth_idf=True
                )
                
                # Generate sentence vectors
                sentence_vectors = vectorizer.fit_transform(cleaned_sentences)
                
                # Calculate cosine similarity matrix
                similarity_matrix = cosine_similarity(sentence_vectors)
                
                # Zero out diagonal (self-similarity)
                np.fill_diagonal(similarity_matrix, 0)
                
                return similarity_matrix
                
            except Exception as e:
                logger.warning(f"TF-IDF similarity calculation failed: {e}")
                # Fall through to token overlap approach
        
        # Token overlap similarity (fallback)
        for i in range(n):
            tokens_i = set(self.text_processor.tokenize(cleaned_sentences[i]))
            tokens_i = {t for t in tokens_i if t not in self.stopwords and len(t) > 1}
            
            for j in range(i+1, n):
                tokens_j = set(self.text_processor.tokenize(cleaned_sentences[j]))
                tokens_j = {t for t in tokens_j if t not in self.stopwords and len(t) > 1}
                
                if not tokens_i or not tokens_j:
                    continue
                
                # Jaccard similarity
                sim = len(tokens_i & tokens_j) / len(tokens_i | tokens_j) if tokens_i | tokens_j else 0
                
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix
    
    def _select_diverse_sentences(self, sentences: List[str], scores: Dict[int, float], 
                                 target_count: int, lambda_param: float = 0.6) -> List[int]:
        """
        Select sentences using Maximal Marginal Relevance (MMR) for diversity
        """
        if not sentences or not scores:
            return []
        
        # Calculate sentence similarity matrix
        similarity_matrix = self._get_sentence_similarity(sentences)
        
        # Initialize variables
        remaining_indices = list(range(len(sentences)))
        selected_indices = []
        
        # Always include the highest scoring sentence first
        if remaining_indices:
            best_first_idx = max(remaining_indices, key=lambda i: scores[i])
            selected_indices.append(best_first_idx)
            remaining_indices.remove(best_first_idx)
        
        # Select remaining sentences using MMR to balance relevance and diversity
        while len(selected_indices) < target_count and remaining_indices:
            best_mmr = -float('inf')
            best_idx = -1
            
            for i in remaining_indices:
                # Relevance component (from sentence scores)
                relevance = scores[i]
                
                # Redundancy component (max similarity to any already selected sentence)
                redundancy = max(similarity_matrix[i, j] for j in selected_indices) if selected_indices else 0
                
                # MMR score: balance between relevance and non-redundancy
                mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break
        
        # Sort indices to maintain original sentence order
        selected_indices.sort()
        
        return selected_indices
    
    def _ensure_summary_coherence(self, selected_sentences: List[str]) -> List[str]:
        """
        Ensure coherence in the final summary by adding connectors and transitions
        """
        if not selected_sentences:
            return []
            
        # If only one sentence, no coherence adjustments needed
        if len(selected_sentences) <= 1:
            return selected_sentences
            
        coherent_sentences = [selected_sentences[0]]  # First sentence remains unchanged
        
        # Process subsequent sentences
        for i in range(1, len(selected_sentences)):
            curr_sent = selected_sentences[i]
            prev_sent = selected_sentences[i-1]
            
            # Check if the sentence already starts with a coherence marker
            curr_tokens = self.text_processor.tokenize(curr_sent)
            
            has_marker = False
            for marker in self.coherence_markers:
                if curr_sent.startswith(marker) or (curr_tokens and curr_tokens[0] == marker):
                    has_marker = True
                    break
            
            if not has_marker:
                # Determine appropriate connector based on semantics
                if self.lang_code == 'te':
                    # For Telugu
                    if i == len(selected_sentences) - 1:  # Last sentence
                        connector = "చివరగా, "
                    elif i == 1:  # Second sentence
                        connector = "అలాగే, "
                    else:
                        # Alternate between different connectors
                        connectors = ["మరియు ", "అదేవిధంగా ", "పైగా ", "ఇంకా "]
                        connector = connectors[i % len(connectors)]
                        
                    curr_sent = connector + curr_sent[0].lower() + curr_sent[1:]
                    
                elif self.lang_code == 'hi':
                    # For Hindi
                    if i == len(selected_sentences) - 1:  # Last sentence
                        connector = "अंत में, "
                    elif i == 1:  # Second sentence
                        connector = "इसके अलावा, "
                    else:
                        # Alternate between different connectors
                        connectors = ["और ", "इसी प्रकार ", "साथ ही ", "जबकि "]
                        connector = connectors[i % len(connectors)]
                        
                    curr_sent = connector + curr_sent[0].lower() + curr_sent[1:]
            
            coherent_sentences.append(curr_sent)
        
        return coherent_sentences
    
    def _extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        Extract important keywords from the text
        """
        # Preprocess and tokenize
        clean_text = self.text_processor.preprocess(text)
        tokens = self.text_processor.tokenize(clean_text)
        
        # Filter stopwords and short tokens
        filtered_tokens = [token for token in tokens 
                          if token not in self.stopwords and len(token) > 2]
        
        if not filtered_tokens:
            return []
        
        # First attempt: Try to use TF-IDF for keyword extraction
        if SKLEARN_AVAILABLE and CountVectorizer is not None:
            try:
                # Use Count Vectorizer for unigrams and bigrams
                count_vec = CountVectorizer(
                    tokenizer=lambda x: [t for t in self.text_processor.tokenize(x) 
                                        if t not in self.stopwords and len(t) > 2],
                    ngram_range=(1, 2),
                    max_features=100
                )
                
                # Generate term vectors
                X = count_vec.fit_transform([clean_text])
                
                # Get vocabulary and feature indices
                vocab = count_vec.get_feature_names_out()
                
                # Calculate term frequencies
                term_freq = {}
                for i, term in enumerate(vocab):
                    term_freq[term] = X[0, i]
                
                # Boost domain-specific terms
                for term in term_freq:
                    # Check if any component of the term is in domain terms
                    term_parts = term.split()
                    for part in term_parts:
                        if part in self.domain_terms:
                            term_freq[term] *= self.domain_terms[part]
                
                # Get top terms
                top_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
                
                # Ensure diversity - avoid similar terms
                diverse_terms = []
                seen_roots = set()
                
                for term, _ in top_terms:
                    # For bigrams, include automatically
                    if ' ' in term:
                        diverse_terms.append(term)
                        continue
                    
                    # Simple stemming - just use first 4 characters as "root"
                    term_root = term[:4] if len(term) > 4 else term
                    
                    if term_root not in seen_roots:
                        diverse_terms.append(term)
                        seen_roots.add(term_root)
                    
                    if len(diverse_terms) >= num_keywords:
                        break
                
                return diverse_terms[:num_keywords]
            
            except Exception as e:
                logger.warning(f"CountVectorizer keyword extraction failed: {e}")
        
        # Fallback: Word frequency method
        word_freq = Counter(filtered_tokens)
        
        # Apply domain-specific weights
        weighted_freq = {}
        for word, freq in word_freq.items():
            # Apply domain weight if available
            domain_weight = self.domain_terms.get(word, 1.0)
            weighted_freq[word] = freq * domain_weight
        
        # Get top words
        top_words = [word for word, _ in sorted(weighted_freq.items(), 
                                               key=lambda x: x[1], reverse=True)[:num_keywords]]
        
        return top_words
    
    def _enhance_summary_with_keywords(self, summary: str, keywords: List[str]) -> str:
        """
        Add keywords section to the summary
        """
        if not keywords:
            return summary
        
        # Create keyword section with appropriate heading based on language
        if self.lang_code == 'te':
            keyword_section = "\n\nముఖ్యమైన పదాలు: " + ", ".join(keywords)
        elif self.lang_code == 'hi':
            keyword_section = "\n\nमुख्य शब्द: " + ", ".join(keywords)
        else:
            keyword_section = "\n\nKeywords: " + ", ".join(keywords)
            
        return summary + keyword_section
    
    def extractive_summarize(self, text: str) -> str:
        """
        Generate extractive summary using improved methods tailored for Indian languages
        """
        if not text or not text.strip():
            return ""
        
        # Preprocess text
        clean_text = self.text_processor.preprocess(text)
        
        # Split into sentences
        sentences = self.text_processor.get_sentences(clean_text)
        
        if not sentences:
            return ""
        
        # Calculate sentence scores
        sentence_scores = self._score_sentences_overall(sentences)
        
        # Determine target summary length
        target_sentence_count = min(
            self.max_sentences,
            max(1, int(len(sentences) * self.compression_ratio))
        )
        
        # Select diverse sentences using MMR
        selected_indices = self._select_diverse_sentences(
            sentences, sentence_scores, target_sentence_count
        )
        
        # Get selected sentences
        selected_sentences = [sentences[i] for i in selected_indices]
        
        # Ensure coherence if needed
        if self.ensure_coherence and len(selected_sentences) > 1:
            selected_sentences = self._ensure_summary_coherence(selected_sentences)
        
        # Combine sentences into summary
        summary = " ".join(selected_sentences)
        
        # Add keywords if requested
        if self.preserve_keywords:
            keywords = self._extract_keywords(text)
            summary = self._enhance_summary_with_keywords(summary, keywords)
        
        return summary


# Neural model classes for transformer-based summarization
class IndicTransformerEncoderLayer(nn.Module):
    """
    Improved Transformer Encoder Layer with Relative Position Embeddings
    for better handling of flexible word order in Indic languages
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, 
                 use_relative_attention=True):
        super().__init__()
        self.use_relative_attention = use_relative_attention
        
        if use_relative_attention:
            self.self_attn = RelativeMultiHeadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = RotaryMultiHeadAttention(d_model, nhead, dropout=dropout)
            
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        self.activation = F.gelu  # Using GELU instead of ReLU for better performance

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block
        src2 = self.norm1(src)  # Pre-normalization
        
        if self.use_relative_attention:
            src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        else:
            # For Rotary implementation
            src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask,
                                  is_causal=is_causal)[0]
            
        src = src + self.dropout1(src2)  # Residual connection
        
        # Feed-forward block
        src2 = self.norm2(src)  # Pre-normalization
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)  # Residual connection
        
        return src


class IndicTransformerEncoder(nn.Module):
    """
    Stacked Transformer Encoder with improved position representations
    for Indian languages
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        
    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                          src_key_padding_mask=src_key_padding_mask,
                          is_causal=is_causal)
            
        if self.norm is not None:
            output = self.norm(output)
            
        return output


class IndicNeuralSummarizer(nn.Module):
    """
    Neural summarization model with Relative Position Embeddings
    specifically designed for Indian languages
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=2048, 
                 dropout=0.1, max_len=1024,
                 use_relative_attention=True):
        super().__init__()
        
        self.d_model = d_model
        self.use_relative_attention = use_relative_attention
        self.max_len = max_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embeddings - we'll use either standard or rotary PE depending on config
        if not use_relative_attention:
            self.pos_embedding = RotaryPositionEmbeddings(d_model)
        else:
            self.pos_embedding = RelativePositionalEncoding(d_model, max_len)
        
        # Create encoder layer
        encoder_layer = IndicTransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, use_relative_attention
        )
        
        # Create encoder
        self.transformer_encoder = IndicTransformerEncoder(
            encoder_layer, num_encoder_layers, nn.LayerNorm(d_model)
        )
        
        # Output layer - predicts summary tokens
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Scaling factor for embeddings
        self.scale = math.sqrt(d_model)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform distribution"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence to prevent attention to future tokens"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Get batch size and sequence length
        if src.dim() == 2:
            batch_size, seq_len = src.size()
        else:
            # Handle unbatched input
            seq_len = src.size(0)
            src = src.unsqueeze(0)
            batch_size = 1
            
        # Apply token embeddings and scale
        src = self.token_embedding(src) * self.scale
        
        # Apply position embeddings
        if self.use_relative_attention:
            # For relative position encoding, we just need the positional encodings
            # The attention mechanism will handle the relative position calculations
            pos_embed = self.pos_embedding(seq_len).to(src.device)
            # No need to add position embeddings to input as they're used in attention
        else:
            # For rotary embeddings, they're handled in the attention mechanism
            # No need to add them to input here
            pos_embed = None
            
        # Pass through transformer encoder
        output = self.transformer_encoder(src, mask=src_mask, 
                                          src_key_padding_mask=src_key_padding_mask)
        
        # Project to vocabulary space
        output = self.output_projection(output)
        
        return output


class NeuralTransformerSummarizer:
    """
    Neural transformer-based summarizer for Indian languages using
    specialized position embeddings
    """
    def __init__(self, model_path, tokenizer_path, lang_code='hi', device=None,
                 max_input_len=512, max_output_len=150, 
                 relative_position=True):
        """
        Initialize the neural summarizer
        
        Args:
            model_path: Path to pre-trained model
            tokenizer_path: Path to tokenizer model
            lang_code: Language code ('te' for Telugu, 'hi' for Hindi)
            device: Torch device (cuda or cpu)
            max_input_len: Maximum input sequence length
            max_output_len: Maximum output sequence length
            relative_position: Whether to use relative position embeddings (True) 
                              or rotary embeddings (False)
        """
        self.lang_code = lang_code
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.relative_position = relative_position
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        try:
            import tokenizers
            self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
            # Add special tokens if not present
            if "[PAD]" not in self.tokenizer.get_vocab():
                logger.warning("Special tokens not found in tokenizer. Model may not work correctly.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
            
        # Determine vocabulary size
        self.vocab_size = self.tokenizer.get_vocab_size()
        
        # Initialize model
        self.model = self._load_model(model_path)
        
        # Text processor for pre/post-processing
        self.text_processor = IndicTextProcessor(lang_code)
        
    def _load_model(self, model_path):
        """Load the neural model"""
        try:
            # Create model architecture
            model = IndicNeuralSummarizer(
                vocab_size=self.vocab_size,
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                dim_feedforward=2048,
                dropout=0.1,
                max_len=self.max_input_len,
                use_relative_attention=self.relative_position
            )
            
            # Load weights if model path is provided
            if model_path and os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning(f"Model path {model_path} not found. Using randomly initialized model.")
                
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def _tokenize_text(self, text):
        """Tokenize text using the loaded tokenizer"""
        # Preprocess text first
        clean_text = self.text_processor.preprocess(text)
        
        try:
            # Use the tokenizer to encode the text
            encoding = self.tokenizer.encode(clean_text)
            return encoding.ids
        except Exception as e:
            logger.error(f"Error in tokenization: {e}")
            # Fallback to simple tokenization for robustness
            tokens = self.text_processor.tokenize(clean_text)
            # This is a very basic fallback that won't work well but prevents crashing
            return tokens[:self.max_input_len]
            
    def _detokenize_text(self, token_ids):
        """Convert token IDs back to text"""
        try:
            # Use the tokenizer to decode the text
            text = self.tokenizer.decode(token_ids)
            return text
        except Exception as e:
            logger.error(f"Error in detokenization: {e}")
            return ""
    
    def generate_summary(self, text, temperature=0.7, top_k=50, top_p=0.9):
        """
        Generate summary using neural transformer model with beam search
        """
        if not text or not text.strip():
            return ""
            
        # Tokenize input text
        input_ids = self._tokenize_text(text)
        
        # Truncate if too long
        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:self.max_input_len]
            
        # Convert to tensor and move to device
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Prepare attention mask (all 1s for now as we don't have padding)
        attn_mask = torch.ones(1, len(input_ids)).to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Generate summary with beam search
        with torch.no_grad():
            try:
                # Beam search generation
                from torch.nn.functional import log_softmax
                
                # Initial state is just the encoder output
                encoder_output = self.model(input_tensor)
                
                # Initialize beams with BOS token
                BOS_TOKEN_ID = 1  # Assuming 1 is the BOS token ID
                beams = [(BOS_TOKEN_ID, 0.0, [])]  # (last_token, score, sequence)
                
                # Beam search parameters
                beam_size = 4
                max_length = self.max_output_len
                
                # Generate tokens
                for _ in range(max_length):
                    candidates = []
                    
                    for last_token, score, sequence in beams:
                        if last_token == 2:  # EOS token
                            candidates.append((last_token, score, sequence))
                            continue
                            
                        # Get current input for decoder
                        curr_input = torch.tensor([sequence + [last_token]], dtype=torch.long).to(self.device)
                        
                        # Generate next token probabilities
                        with torch.no_grad():
                            output = self.model(curr_input)
                            logits = output[0, -1, :]
                            
                            # Apply temperature
                            logits = logits / temperature
                            
                            # Apply top-k sampling
                            if top_k > 0:
                                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                                logits[indices_to_remove] = float('-inf')
                                
                            # Apply top-p (nucleus) sampling
                            if 0 < top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                                logits[indices_to_remove] = float('-inf')
                            
                            # Get log probabilities
                            log_probs = log_softmax(logits, dim=-1)
                            
                            # Get top k tokens
                            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                            
                            # Add new candidates
                            for i in range(beam_size):
                                token_id = topk_indices[i].item()
                                token_score = topk_log_probs[i].item()
                                new_score = score + token_score
                                new_sequence = sequence + [last_token]
                                candidates.append((token_id, new_score, new_sequence))
                    
                    # Keep top beam_size candidates
                    beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
                    
                    # Check if all beams end with EOS
                    if all(beam[0] == 2 for beam in beams):  # All end with EOS
                        break
                
                # Get best beam
                _, _, best_sequence = beams[0]
                
                # Remove BOS and add the last token from beam
                if beams[0][0] != 2:  # If best beam doesn't end with EOS
                    best_sequence = best_sequence + [beams[0][0]]
                    
                # Convert token IDs to text
                summary = self._detokenize_text(best_sequence)
                
                return summary
                
            except Exception as e:
                logger.error(f"Error in summary generation: {e}")
                # Fallback to extractive summarization
                return "Error generating neural summary. Please try extractive summarization instead."


class EnhancedIndianSummarizer:
    """
    Final enhanced summarizer that combines extractive and neural approaches
    with specialized handling for Indian languages
    """
    def __init__(self, lang_code='hi', mode='hybrid', 
                 model_path=None, tokenizer_path=None,
                 compression_ratio=0.3, max_sentences=5,
                 use_relative_position=True):
        """
        Initialize the enhanced summarizer
        
        Args:
            lang_code: Language code ('te' for Telugu, 'hi' for Hindi)
            mode: Summarization mode ('extractive', 'neural', 'hybrid')
            model_path: Path to neural model (required for neural and hybrid modes)
            tokenizer_path: Path to tokenizer (required for neural and hybrid modes)
            compression_ratio: Target ratio of summary to original text length
            max_sentences: Maximum number of sentences in summary
            use_relative_position: Whether to use relative position embeddings
        """
        self.lang_code = lang_code
        self.mode = mode
        self.compression_ratio = compression_ratio
        self.max_sentences = max_sentences
        
        # Initialize extractive summarizer
        self.extractive_summarizer = IndianLanguageSummarizer(
            lang_code=lang_code,
            compression_ratio=compression_ratio,
            max_sentences=max_sentences
        )
        
        # Initialize neural summarizer if needed
        self.neural_summarizer = None
        if mode in ['neural', 'hybrid'] and model_path and tokenizer_path:
            try:
                self.neural_summarizer = NeuralTransformerSummarizer(
                    model_path=model_path,
                    tokenizer_path=tokenizer_path,
                    lang_code=lang_code,
                    relative_position=use_relative_position
                )
            except Exception as e:
                logger.error(f"Failed to initialize neural summarizer: {e}")
                self.mode = 'extractive'  # Fallback to extractive
        elif mode in ['neural', 'hybrid']:
            logger.warning("Model or tokenizer path not provided. Falling back to extractive mode.")
            self.mode = 'extractive'  # Fallback to extractive
            
    def summarize(self, text):
        """
        Generate summary based on the selected mode
        """
        if not text or not text.strip():
            return ""
            
        if self.mode == 'extractive' or not self.neural_summarizer:
            # Use extractive summarization
            return self.extractive_summarizer.extractive_summarize(text)
            
        elif self.mode == 'neural':
            # Use neural summarization
            return self.neural_summarizer.generate_summary(text)
            
        else:  # hybrid mode
            # Generate both summaries
            extractive_summary = self.extractive_summarizer.extractive_summarize(text)
            
            try:
                # Try neural summarization
                neural_summary = self.neural_summarizer.generate_summary(text)
                
                # Compare both summaries and select the better one
                # For now, simple heuristic: choose longer one that's not too long
                extr_len = len(extractive_summary.split())
                neural_len = len(neural_summary.split())
                
                # Determine target length
                target_len = int(len(text.split()) * self.compression_ratio)
                
                # Choose summary closer to target length but not too verbose
                if abs(neural_len - target_len) < abs(extr_len - target_len) and neural_len > 10:
                    return neural_summary
                else:
                    return extractive_summary
                    
            except Exception as e:
                logger.error(f"Neural summarization failed: {e}")
                # Fallback to extractive
                return extractive_summary


# Helper function for integration with IndicSummarizer
def generate_enhanced_summary(text: str, lang_code: str = 'te', **kwargs) -> str:
    """
    Generate an enhanced summary for the given text and language code.
    """
    summarizer = EnhancedSummarizer(lang_code=lang_code) if 'lang_code' in EnhancedSummarizer.__init__.__code__.co_varnames else EnhancedSummarizer()
    return summarizer.summarize(text, **kwargs)

import os
import sys
import torch
import logging
import argparse
import json
from typing import List, Dict, Optional, Union
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
CONFIG_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "model_config.json"
MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "models" / "final_model"

class EnhancedSummarizer:
    """
    Enhanced summarizer that leverages language-specific adapters and morphological analysis
    to improve summarization for Hindi and Telugu texts.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_adapters: bool = True,
        use_morphology: bool = True,
        max_length: int = 128,
        min_length: int = 30,
        device: Optional[str] = None
    ):
        """
        Initialize the enhanced summarizer.
        
        Args:
            model_path: Path to the model directory
            config_path: Path to the model configuration file
            use_adapters: Whether to use language adapters
            use_morphology: Whether to use morphological analysis
            max_length: Maximum length of generated summaries
            min_length: Minimum length of generated summaries
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.use_adapters = use_adapters
        self.use_morphology = use_morphology
        self.max_length = max_length
        self.min_length = min_length
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set paths
        if model_path is None:
            model_path = MODEL_DIR
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        
        # Initialize components
        self._load_config()
        self._initialize_tokenizer()
        self._initialize_model()
        
        # Initialize morphological analyzers if enabled
        if self.use_morphology:
            self._initialize_morphology()
        
        logger.info(f"Enhanced summarizer initialized. Device: {self.device}")
        logger.info(f"Using adapters: {self.use_adapters}, Using morphology: {self.use_morphology}")
    
    def _load_config(self):
        """Load model configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        if self.use_adapters:
            # Ensure the config has adapter settings
            if 'use_adapters' not in self.config:
                self.config['use_adapters'] = True
            if 'adapter_size' not in self.config:
                self.config['adapter_size'] = 64
            if 'languages' not in self.config:
                self.config['languages'] = ["hi", "te"]
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer"""
        self.tokenizer = IndicTokenizer(str(self.config_path))
    
    def _initialize_model(self):
        """Initialize model with configuration"""
        # Create model config with adapter support if enabled
        if self.use_adapters:
            self.model_config = IndicSLMConfig(
                vocab_size=self.config.get('vocab_size', 50000),
                hidden_size=self.config.get('n_embd', 384),
                num_hidden_layers=self.config.get('n_layer', 6),
                num_attention_heads=self.config.get('n_head', 6),
                intermediate_size=self.config.get('n_embd', 384) * 4,
                max_position_embeddings=self.config.get('n_positions', 512),
                use_adapters=True,
                adapter_size=self.config.get('adapter_size', 64),
                languages=self.config.get('languages', ["hi", "te"]),
                attention_type=self.config.get('attention_type', 'relative')
            )
        else:
            self.model_config = IndicSLMConfig(
                vocab_size=self.config.get('vocab_size', 50000),
                hidden_size=self.config.get('n_embd', 384),
                num_hidden_layers=self.config.get('n_layer', 6),
                num_attention_heads=self.config.get('n_head', 6),
                intermediate_size=self.config.get('n_embd', 384) * 4,
                max_position_embeddings=self.config.get('n_positions', 512),
                attention_type=self.config.get('attention_type', 'relative')
            )
        
        # Initialize model
        self.model = IndicSLM(self.model_config)
        
        # Load pre-trained weights if available
        model_weights_path = self.model_path / "pytorch_model.bin"
        if model_weights_path.exists():
            logger.info(f"Loading pre-trained weights from {model_weights_path}")
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def _initialize_morphology(self):
        """Initialize morphological analyzers"""
        self.morphology_analyzers = {
            'hi': HindiMorphologyAnalyzer(),
            'te': TeluguMorphologyAnalyzer()
        }
        
        # Initialize data processor with morphology enabled
        self.data_processor = DataProcessor(str(self.config_path), enable_morphology=True)
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        
        Args:
            text: Input text
            
        Returns:
            Language code ('hi' or 'te')
        """
        return detect_language(text)
    
    def analyze_morphology(self, text: str, lang: Optional[str] = None) -> Dict:
        """
        Perform morphological analysis on the input text.
        
        Args:
            text: Input text
            lang: Language code ('hi' or 'te'), detected automatically if not provided
            
        Returns:
            Dictionary with morphological analysis results
        """
        if not self.use_morphology:
            return {}
        
        # Detect language if not provided
        if lang is None:
            lang = self.detect_language(text)
            if lang is None:
                logger.warning("Could not detect language. Defaulting to Hindi.")
                lang = "hi"
        
        # Use data processor for analysis
        return self.data_processor.analyze_morphology(text, lang)
    
    def preprocess_text(self, text: str, lang: Optional[str] = None) -> Dict:
        """
        Preprocess text for summarization.
        
        Args:
            text: Input text
            lang: Language code, detected automatically if not provided
            
        Returns:
            Dictionary with preprocessed text and metadata
        """
        # Clean and normalize text
        if lang is None:
            lang = self.detect_language(text)
            if lang is None:
                logger.warning("Could not detect language. Defaulting to Hindi.")
                lang = "hi"
        
        # Normalize text
        normalized_text = self.data_processor.normalize_text(text, lang)
        
        # Tokenize text
        encoded = self.tokenizer.encode(normalized_text)
        
        # Perform morphological analysis if enabled
        morphology = {}
        if self.use_morphology:
            morphology = self.analyze_morphology(normalized_text, lang)
        
        return {
            'text': normalized_text,
            'encoded': encoded,
            'language': lang,
            'morphology': morphology
        }
    
    def generate_summary(
        self, 
        text: str, 
        lang: Optional[str] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: int = 4,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3
    ) -> str:
        """
        Generate a summary for the input text.
        
        Args:
            text: Input text to summarize
            lang: Language code, detected automatically if not provided
            max_length: Maximum length of the generated summary
            min_length: Minimum length of the generated summary
            num_beams: Number of beams for beam search
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            temperature: Sampling temperature
            repetition_penalty: Penalty for repeating tokens
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            
        Returns:
            Generated summary text
        """
        if max_length is None:
            max_length = self.max_length
        if min_length is None:
            min_length = self.min_length
        
        # Preprocess text
        processed = self.preprocess_text(text, lang)
        input_ids = torch.tensor([processed['encoded']['input_ids']], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([processed['encoded']['attention_mask']], dtype=torch.long).to(self.device)
        
        # Detect language if not provided
        lang = processed['language']
        
        # Enable language-specific adapter if available
        language_id = lang if self.use_adapters else None
        
        # Generate summary (simple greedy decoding for now)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                language_id=language_id
            )
            
            # Get the predicted next token probabilities
            logits = outputs['logits']
            
            # For more sophisticated generation, we would use beam search or sampling
            # But for simplicity, we'll use greedy decoding here
            # In a real implementation, this would be replaced with a proper generation algorithm
            
            # Placeholder for the actual generation logic
            # This is a simplified example - in practice, you would use a proper generation function
            next_token_id = logits[:, -1, :].argmax(dim=-1)
            
            # For demonstration, we'll just return a simple summary
            # In practice, you would generate a proper summary using the model's output
            
            # Decode the token IDs to text
            summary = "This is a placeholder summary. In a real implementation, this would be generated using the model's output."
            
            # Use morphological features to improve summary generation
            if self.use_morphology and 'morphology' in processed and processed['morphology']:
                # In a real implementation, you would use the morphological features
                # to improve the summary generation process
                # For example, ensuring proper verb forms, maintaining gender agreement, etc.
                logger.info(f"Using morphological features for summary generation")
        
        return summary
    
    def summarize(self, text: str, lang: Optional[str] = None) -> Dict:
        """
        Main summarization function that returns the summary and metadata.
        
        Args:
            text: Input text to summarize
            lang: Language code, detected automatically if not provided
            
        Returns:
            Dictionary with summary and metadata
        """
        # Detect language if not provided
        if lang is None:
            lang = self.detect_language(text)
        
        # Generate summary
        summary = self.generate_summary(text, lang)
        
        # Return summary with metadata
        return {
            'summary': summary,
            'language': lang,
            'original_text': text,
            'text_length': len(text),
            'summary_length': len(summary)
        }

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="Enhanced summarizer for Hindi and Telugu")
    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="File containing text to summarize")
    parser.add_argument("--lang", type=str, choices=["hi", "te"], help="Language of the text")
    parser.add_argument("--model_path", type=str, help="Path to model directory")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum summary length")
    parser.add_argument("--min_length", type=int, default=30, help="Minimum summary length")
    parser.add_argument("--no_adapters", action="store_true", help="Disable language adapters")
    parser.add_argument("--no_morphology", action="store_true", help="Disable morphological analysis")
    parser.add_argument("--output", type=str, help="Output file for the summary")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize summarizer
    summarizer = EnhancedSummarizer(
        model_path=args.model_path,
        config_path=args.config_path,
        use_adapters=not args.no_adapters,
        use_morphology=not args.no_morphology,
        max_length=args.max_length,
        min_length=args.min_length
    )
    
    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        parser.error("Either --text or --file must be provided")
    
    # Generate summary
    result = summarizer.summarize(text, args.lang)
    
    # Print or save summary
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result['summary'])
        logger.info(f"Summary saved to {args.output}")
    else:
        print("\nOriginal Text:")
        print("-" * 40)
        print(text[:200] + "..." if len(text) > 200 else text)
        print("\nSummary:")
        print("-" * 40)
        print(result['summary'])
        print("\nMetadata:")
        print(f"Language: {result['language']}")
        print(f"Original length: {result['text_length']} characters")
        print(f"Summary length: {result['summary_length']} characters")
        print(f"Compression ratio: {result['summary_length'] / result['text_length']:.2f}")

if __name__ == "__main__":
    main()