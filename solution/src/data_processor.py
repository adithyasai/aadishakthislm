from typing import List, Dict, Tuple
import pandas as pd
from datasets import Dataset, DatasetDict
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.indic_tokenize import trivial_tokenize
import json
import os

class DataProcessor:
    def __init__(self, config_path: str = "../configs/model_config.json"):
        self.config = self._load_config(config_path)
        self.normalizers = {
            'te': IndicNormalizerFactory().get_normalizer('te'),
            'hi': IndicNormalizerFactory().get_normalizer('hi')
        }
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_text(self, text: str, lang: str) -> str:
        """Normalize text for Telugu or Hindi"""
        if lang not in self.normalizers:
            raise ValueError(f"Language {lang} not supported. Use 'te' or 'hi'")
        return self.normalizers[lang].normalize(text)
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove HTML tags (basic)
        text = text.replace('<br>', ' ').replace('<p>', ' ').replace('</p>', ' ')
        return text.strip()
    
    def prepare_dataset(self, data_path: str, lang: str) -> Dataset:
        """Prepare dataset for training"""
        # Read the data (assuming CSV format with 'text' and 'summary' columns)
        df = pd.read_csv(data_path)
        
        # Clean and normalize text
        df['text'] = df['text'].apply(self.clean_text)
        df['text'] = df['text'].apply(lambda x: self.normalize_text(x, lang))
        
        if 'summary' in df.columns:
            df['summary'] = df['summary'].apply(self.clean_text)
            df['summary'] = df['summary'].apply(lambda x: self.normalize_text(x, lang))
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def tokenize_text(self, text: str, lang: str) -> List[str]:
        """Tokenize text using indic-nlp-library"""
        return trivial_tokenize(text, lang)
    
    def create_train_val_test_split(self, dataset: Dataset, 
                                  train_size: float = 0.8,
                                  val_size: float = 0.1) -> DatasetDict:
        """Split dataset into train, validation, and test sets"""
        splits = dataset.train_test_split(test_size=1-train_size)
        test_valid = splits['test'].train_test_split(test_size=0.5)
        
        return DatasetDict({
            'train': splits['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })