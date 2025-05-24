#!/usr/bin/env python3
"""
Data processor module for the SLM project.

This module handles data preprocessing, normalization, augmentation,
and other text processing operations for Indic languages.
"""
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd
from datasets import Dataset, DatasetDict
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
import json
import os
from pathlib import Path
import logging
import random

# Import morphological analyzers
from src.hindi_morphology import HindiMorphologyAnalyzer
from src.telugu_morphology import TeluguMorphologyAnalyzer

# Import data augmentation
try:
    from src.data_augmentation import DataAugmentor
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    logging.warning("Data augmentation module not available")

# Import code-switching handler
try:
    from src.code_switching import CodeSwitchingHandler
    CODE_SWITCHING_AVAILABLE = True
except ImportError:
    CODE_SWITCHING_AVAILABLE = False
    logging.warning("Code-switching module not available")

# Setup base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Handles data processing operations for the SLM project,
    including normalization, tokenization, and augmentation.
    """
    
    def __init__(self, config_path: str = str(DEFAULT_CONFIG_PATH), 
                 enable_morphology: bool = False,
                 enable_augmentation: bool = False,
                 enable_code_switching: bool = False,
                 code_switching_cache_dir: Optional[str] = None):
        """
        Initialize the data processor.
        
        Args:
            config_path: Path to configuration file
            enable_morphology: Whether to enable morphological analysis
            enable_augmentation: Whether to enable data augmentation
            enable_code_switching: Whether to enable code-switching handling
            code_switching_cache_dir: Directory for caching code-switching statistics
        """
        self.config = self._load_config(config_path)
        self.normalizers = {
            'te': IndicNormalizerFactory().get_normalizer('te'),
            'hi': IndicNormalizerFactory().get_normalizer('hi')
        }
        
        # Initialize morphological analyzers if enabled
        self.enable_morphology = enable_morphology
        self.morphology_analyzers = {}
        
        if enable_morphology:
            try:
                self.morphology_analyzers['hi'] = HindiMorphologyAnalyzer()
                self.morphology_analyzers['te'] = TeluguMorphologyAnalyzer()
                logger.info("Morphological analyzers initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize morphological analyzers: {e}")
                self.enable_morphology = False
                
        # Initialize data augmentation if enabled
        self.enable_augmentation = enable_augmentation and AUGMENTATION_AVAILABLE
        self.augmentor = None
        
        if self.enable_augmentation:
            try:
                self.augmentor = DataAugmentor()
                logger.info("Data augmentor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize data augmentor: {e}")
                self.enable_augmentation = False
                
        # Initialize code-switching handler if enabled
        self.enable_code_switching = enable_code_switching and CODE_SWITCHING_AVAILABLE
        self.code_switcher = None
        
        if self.enable_code_switching:
            try:
                self.code_switcher = CodeSwitchingHandler(cache_dir=code_switching_cache_dir)
                logger.info("Code-switching handler initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize code-switching handler: {e}")
                self.enable_code_switching = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_text(self, text: str, lang: str) -> str:
        """
        Normalize text for Telugu or Hindi
        
        Args:
            text: Input text
            lang: Language code ('hi' or 'te')
            
        Returns:
            Normalized text
        """
        if not isinstance(text, str):
            return ""
        if lang not in self.normalizers:
            raise ValueError(f"Language {lang} not supported. Use 'te' or 'hi'")
            
        # Check if text contains code-switching and apply special normalization if enabled
        if self.enable_code_switching and self.code_switcher and self.code_switcher.detect_code_switching(text):
            text = self.normalize_code_switched_text(text, lang)
        
        # First normalize using IndicNLP normalizer
        normalized = self.normalizers[lang].normalize(text)
        
        # Additional normalization steps
        normalized = ' '.join(normalized.split())  # Handle multiple spaces
        return normalized.strip()
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove HTML tags (basic)
        text = text.replace('<br>', ' ').replace('<p>', ' ').replace('</p>', ' ')
        # Remove any URLs
        text = ' '.join(word for word in text.split() if not word.startswith('http'))
        return text.strip()
    
    def prepare_dataset(self, data_path: str, lang: str) -> Dataset:
        """
        Prepare dataset for training
        
        Args:
            data_path: Path to data file
            lang: Language code
            
        Returns:
            Processed dataset
        """
        file_extension = os.path.splitext(data_path)[1].lower()
        
        if file_extension == '.jsonl':
            # Read JSONL file
            data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        processed_item = {}
                        
                        # Handle different dataset formats
                        if 'text' in item:
                            processed_item['text'] = self.normalize_text(self.clean_text(item['text']), lang)
                        elif 'article' in item:  # XLSum format
                            processed_item['text'] = self.normalize_text(self.clean_text(item['article']), lang)
                        elif 'content' in item:  # TeSum format
                            processed_item['text'] = self.normalize_text(self.clean_text(item['content']), lang)
                        
                        # Handle summary fields
                        if 'summary' in item:
                            processed_item['summary'] = self.normalize_text(self.clean_text(item['summary']), lang)
                        elif 'headline' in item:  # XLSum format
                            processed_item['summary'] = self.normalize_text(self.clean_text(item['headline']), lang)
                        elif 'title' in item:  # TeSum format
                            processed_item['summary'] = self.normalize_text(self.clean_text(item['title']), lang)
                        
                        # Add morphological analysis if enabled
                        if self.enable_morphology and lang in self.morphology_analyzers and processed_item.get('text'):
                            morph_analysis = self.analyze_morphology(processed_item['text'], lang)
                            processed_item.update(morph_analysis)
                        
                        # Process code-switched text if enabled
                        if self.enable_code_switching and self.code_switcher and processed_item.get('text'):
                            cs_analysis = self.analyze_code_switching(processed_item['text'], lang)
                            if cs_analysis:
                                processed_item.update(cs_analysis)
                        
                        if processed_item.get('text'):  # Only add if we have text content
                            data.append(processed_item)
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON lines
            
            # Convert to DataFrame and then to HuggingFace dataset
            df = pd.DataFrame(data)
            
        else:  # Assume CSV format
            df = pd.read_csv(data_path)
            
            # Clean and normalize text
            if 'text' in df.columns:
                df['text'] = df['text'].apply(self.clean_text)
                df['text'] = df['text'].apply(lambda x: self.normalize_text(x, lang))
            
            if 'summary' in df.columns:
                df['summary'] = df['summary'].apply(self.clean_text)
                df['summary'] = df['summary'].apply(lambda x: self.normalize_text(x, lang))
            
            # Add morphological analysis if enabled
            if self.enable_morphology and lang in self.morphology_analyzers and 'text' in df.columns:
                # Apply morphological analysis to each row
                morph_results = df['text'].apply(lambda x: self.analyze_morphology(x, lang))
                
                # Extract features into separate columns
                if not morph_results.empty and any(morph_results):
                    for key in morph_results.iloc[0].keys():
                        df[f'morph_{key}'] = morph_results.apply(lambda x: x.get(key, None) if x else None)
            
            # Add code-switching analysis if enabled
            if self.enable_code_switching and self.code_switcher and 'text' in df.columns:
                # Apply code-switching analysis to each row
                cs_results = df['text'].apply(lambda x: self.analyze_code_switching(x, lang))
                
                # Extract features into separate columns
                if not cs_results.empty and any(cs_results):
                    sample_result = next((r for r in cs_results if r), {})
                    for key in sample_result.keys():
                        df[f'cs_{key}'] = cs_results.apply(lambda x: x.get(key, None) if x else None)
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def tokenize_text(self, text: str, lang: str) -> List[str]:
        """
        Tokenize text using indic-nlp-library or code-switching-aware tokenization
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            return []
            
        # Check for code-switching and use special tokenization if needed
        if self.enable_code_switching and self.code_switcher and self.code_switcher.detect_code_switching(text):
            return self.code_switcher.tokenize_code_switched_text(text, primary_lang=lang)
        
        # Default to indicnlp tokenization
        return trivial_tokenize(text, lang)
    
    def create_train_val_test_split(self, dataset: Dataset, 
                                  train_size: float = 0.8,
                                  val_size: float = 0.1) -> DatasetDict:
        """
        Split dataset into train, validation, and test sets
        
        Args:
            dataset: Input dataset
            train_size: Proportion of data for training
            val_size: Proportion of data for validation
            
        Returns:
            DatasetDict with train, validation, and test splits
        """
        splits = dataset.train_test_split(test_size=1-train_size)
        test_valid = splits['test'].train_test_split(test_size=0.5)
        
        return DatasetDict({
            'train': splits['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })
        
    def analyze_morphology(self, text: str, lang: str) -> Dict[str, Union[List[str], Dict]]:
        """
        Perform morphological analysis on text.
        
        Args:
            text: Text to analyze
            lang: Language code ('hi' or 'te')
            
        Returns:
            Dictionary with morphological features
        """
        if not self.enable_morphology or lang not in self.morphology_analyzers:
            return {}
        
        analyzer = self.morphology_analyzers[lang]
        
        try:
            # Get all analyses for words in the text
            analyses = analyzer.analyze_text(text)
            
            # Extract stems/roots
            stems = [analysis['stem'] for analysis in analyses]
            
            # For each analysis, extract the most important morphological features
            features = []
            for analysis in analyses:
                feature = {}
                # Copy useful features but skip large fields
                for key, value in analysis.items():
                    if key not in ['word', 'segments']:
                        feature[key] = value
                features.append(feature)
            
            return {
                'stems': stems,
                'root_forms': analyzer.get_root_forms(text),
                'morphology': features
            }
        except Exception as e:
            logger.warning(f"Morphological analysis failed: {e}")
            return {}
            
    def get_segmented_tokens(self, text: str, lang: str) -> List[List[str]]:
        """
        Get tokens segmented into morphemes.
        
        Args:
            text: Text to segment
            lang: Language code ('hi' or 'te')
            
        Returns:
            List of token segments, where each segment is a list of morphemes
        """
        if not self.enable_morphology or lang not in self.morphology_analyzers:
            return []
        
        analyzer = self.morphology_analyzers[lang]
        tokens = self.tokenize_text(text, lang)
        
        segmented_tokens = []
        for token in tokens:
            if lang == 'hi':
                segments = analyzer.segment_word(token)
            else:  # Telugu
                segments = analyzer.segment_word(token)
            
            segmented_tokens.append(segments)
        
        return segmented_tokens
        
    def augment_text(self, text: str, lang: str, 
                   techniques: List[str] = None, 
                   augmentation_count: int = 1) -> List[str]:
        """
        Apply data augmentation to the input text.
        
        Args:
            text: Source text to augment
            lang: Language code ('hi', 'te', 'bn', 'ta', 'ml', 'en')
            techniques: List of techniques to apply ('synonym', 'backtranslate', 
                        'paraphrase', 'deletion', 'swap')
            augmentation_count: Number of augmented versions to generate
            
        Returns:
            List of augmented texts
        """
        if not self.enable_augmentation or not self.augmentor:
            return [text]
        
        return self.augmentor.augment(text, lang, techniques, augmentation_count)
    
    def augment_dataset(self, dataset: Dataset, lang: str, 
                      techniques: List[str] = None,
                      augmentation_ratio: float = 0.3,
                      samples_per_example: int = 1) -> Dataset:
        """
        Augment a dataset with variations of existing examples.
        
        Args:
            dataset: Input dataset to augment
            lang: Language code
            techniques: List of augmentation techniques to apply
            augmentation_ratio: Proportion of dataset to augment (0.0-1.0)
            samples_per_example: Number of augmented samples per selected example
            
        Returns:
            Augmented dataset
        """
        if not self.enable_augmentation or not self.augmentor:
            logger.warning("Data augmentation not available or not enabled")
            return dataset
        
        # Default techniques
        if techniques is None:
            techniques = ['synonym', 'deletion', 'swap']
        
        # Determine how many examples to augment
        num_examples = len(dataset)
        num_to_augment = max(1, int(num_examples * augmentation_ratio))
        
        logger.info(f"Augmenting {num_to_augment} examples from dataset with {num_examples} entries")
        
        # Randomly select examples to augment
        indices_to_augment = random.sample(range(num_examples), num_to_augment)
        
        # Prepare data for augmentation
        original_data = dataset.to_pandas().to_dict('records')
        augmented_data = original_data.copy()
        
        # Augment selected examples
        for idx in indices_to_augment:
            example = original_data[idx]
            
            # Augment text field
            if 'text' in example and example['text']:
                augmented_texts = self.augment_text(
                    example['text'], 
                    lang, 
                    techniques,
                    samples_per_example
                )
                
                # Add augmented examples to dataset
                for i, aug_text in enumerate(augmented_texts[1:]):  # Skip the first one (original)
                    new_example = example.copy()
                    new_example['text'] = aug_text
                    new_example['augmented'] = True
                    new_example['augmentation_techniques'] = ','.join(techniques)
                    augmented_data.append(new_example)
            
            # Optionally augment summary field
            if 'summary' in example and example['summary'] and random.random() < 0.5:
                augmented_summaries = self.augment_text(
                    example['summary'],
                    lang,
                    techniques,
                    1  # Only generate one augmented summary
                )
                
                if len(augmented_summaries) > 1:
                    # Update the last added example with augmented summary
                    if len(augmented_data) > len(original_data):
                        augmented_data[-1]['summary'] = augmented_summaries[1]
        
        # Convert back to dataset
        return Dataset.from_pandas(pd.DataFrame(augmented_data))
    
    # ---------- Code-switching methods ----------
    
    def analyze_code_switching(self, text: str, lang: str) -> Dict:
        """
        Analyze code-switching patterns in the text.
        
        Args:
            text: Input text
            lang: Primary language code
            
        Returns:
            Dictionary with code-switching analysis results
        """
        if not self.enable_code_switching or not self.code_switcher:
            return {}
        
        try:
            # Detect if text contains code-switching
            if self.code_switcher.detect_code_switching(text):
                # Get detailed analysis
                analysis = self.code_switcher.analyze_code_switching(text)
                
                # Extract relevant information for the dataset
                result = {
                    'is_code_switched': True,
                    'cs_switch_rate': analysis['switch_rate'],
                    'cs_token_count': analysis['token_count'],
                    'cs_switch_count': analysis['switch_count']
                }
                
                # Include language distribution if available
                if 'language_distribution' in analysis:
                    # Convert to string representation for easier storage
                    lang_dist = analysis['language_distribution']
                    result['cs_language_distribution'] = ';'.join(f"{k}:{v:.3f}" for k, v in lang_dist.items())
                
                # Tag tokens with their languages (keeping only the first 50 tokens to save space)
                if 'tagged_tokens' in analysis:
                    tagged_tokens = analysis['tagged_tokens'][:50]
                    result['cs_token_languages'] = ','.join(lang for _, lang in tagged_tokens)
                
                return result
            else:
                return {'is_code_switched': False}
                
        except Exception as e:
            logger.warning(f"Code-switching analysis failed: {e}")
            return {'is_code_switched': False, 'cs_error': str(e)}
    
    def normalize_code_switched_text(self, text: str, lang: str) -> str:
        """
        Apply special normalization for code-switched text.
        
        Args:
            text: Input text
            lang: Primary language code
            
        Returns:
            Normalized text
        """
        if not self.enable_code_switching or not self.code_switcher:
            return text
            
        try:
            if self.code_switcher.detect_code_switching(text):
                return self.code_switcher.normalize_code_switched_text(text, primary_lang=lang)
            else:
                return text
        except Exception as e:
            logger.warning(f"Code-switched normalization failed: {e}")
            return text
    
    def create_code_switched_training_examples(self, dataset: Dataset, 
                                             lang: str,
                                             target_lang: str = 'en',
                                             ratio: float = 0.2) -> Dataset:
        """
        Create training examples with code-switched content.
        
        Args:
            dataset: Input dataset
            lang: Primary language code
            target_lang: Target language for code-switching (default: English)
            ratio: Proportion of dataset to convert to code-switched examples
            
        Returns:
            Dataset with added code-switched examples
        """
        if not self.enable_code_switching or not self.code_switcher:
            logger.warning("Code-switching functionality not available")
            return dataset
        
        # Convert dataset to pandas DataFrame for easier manipulation
        df = dataset.to_pandas()
        
        # Determine number of examples to convert
        num_examples = len(df)
        num_to_convert = max(1, int(num_examples * ratio))
        
        logger.info(f"Creating {num_to_convert} code-switched examples from dataset with {num_examples} entries")
        
        # Randomly select examples to convert
        indices_to_convert = random.sample(range(num_examples), num_to_convert)
        
        # Prepare new examples
        code_switched_data = []
        
        for idx in indices_to_convert:
            example = df.iloc[idx].to_dict()
            
            if 'text' in example and example['text']:
                # Generate code-switched versions
                cs_text = self.code_switcher.generate_code_switched_examples(
                    example['text'], 
                    target_lang=target_lang
                )
                
                # Update the example with code-switched text
                cs_example = example.copy()
                cs_example['text'] = cs_text
                cs_example['original_text'] = example['text']
                cs_example['is_code_switched'] = True
                
                # Analyze the code-switched text
                cs_analysis = self.analyze_code_switching(cs_text, lang)
                
                # Add analysis results to the example
                for key, value in cs_analysis.items():
                    cs_example[key] = value
                
                code_switched_data.append(cs_example)
        
        # Create a dataset with the new examples
        cs_dataset = Dataset.from_pandas(pd.DataFrame(code_switched_data))
        
        # Concatenate with the original dataset
        return Dataset.concatenate_datasets([dataset, cs_dataset])
    
    def update_code_switching_stats(self, dataset: Dataset, lang: str) -> Dict:
        """
        Update code-switching statistics from a dataset.
        
        Args:
            dataset: Input dataset
            lang: Primary language code
            
        Returns:
            Dictionary with statistics about code-switching in the dataset
        """
        if not self.enable_code_switching or not self.code_switcher:
            return {'error': 'Code-switching functionality not available'}
            
        try:
            # Convert dataset to list of texts
            texts = dataset['text'] if 'text' in dataset.column_names else []
            
            # Update statistics from corpus
            stats = self.code_switcher.update_stats_from_corpus(texts, primary_lang=lang)
            
            return stats
        except Exception as e:
            logger.error(f"Failed to update code-switching statistics: {e}")
            return {'error': str(e)}
