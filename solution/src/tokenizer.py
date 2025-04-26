from typing import List, Dict, Optional
from tokenizers import Tokenizer, models, pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
import sentencepiece as spm
from pathlib import Path
import json

class IndicTokenizer:
    def __init__(self, config_path: str = "../configs/model_config.json"):
        self.config = self._load_config(config_path)
        self.vocab_size = self.config.get('vocab_size', 32000)
        self.tokenizer = None
        self.sp_model = None
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def train_tokenizer(self, 
                       train_files: List[str],
                       output_dir: str,
                       model_prefix: str = "indic_tokenizer") -> None:
        """Train a SentencePiece tokenizer on Telugu and Hindi data"""
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure SentencePiece training
        spm.SentencePieceTrainer.train(
            input=train_files,
            model_prefix=f"{output_dir}/{model_prefix}",
            vocab_size=self.vocab_size,
            character_coverage=0.9995,  # High coverage for Indic scripts
            model_type="bpe",          # Using BPE algorithm
            user_defined_symbols=["<pad>", "<s>", "</s>", "<mask>", "<te>", "<hi>"],
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            input_sentence_size=1000000,
            shuffle_input_sentence=True,
            train_extremely_large_corpus=True
        )
        
        # Load the trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{output_dir}/{model_prefix}.model")
        
        # Initialize Hugging Face tokenizer with SentencePiece model
        self.tokenizer = Tokenizer(models.BPE.from_file(
            f"{output_dir}/{model_prefix}.model",
            f"{output_dir}/{model_prefix}.vocab"
        ))
        
        # Add special tokens processing
        self.tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> $B </s>",
            special_tokens=[
                ("<s>", self.sp_model.bos_id()),
                ("</s>", self.sp_model.eos_id()),
            ],
        )
        
    def load_tokenizer(self, model_path: str) -> None:
        """Load a pretrained tokenizer"""
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f"{model_path}.model")
        self.tokenizer = Tokenizer.from_file(f"{model_path}.json")
    
    def save_tokenizer(self, output_path: str) -> None:
        """Save the tokenizer files"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Train or load a tokenizer first.")
        self.tokenizer.save(f"{output_path}.json")
    
    def encode(self, 
              text: str, 
              lang: str,
              add_special_tokens: bool = True) -> Dict:
        """Encode text with language-specific prefix"""
        if lang not in ['te', 'hi']:
            raise ValueError("Language must be 'te' or 'hi'")
        
        # Add language token prefix
        text = f"<{lang}> {text}" if add_special_tokens else text
        
        # Encode the text
        encoding = self.tokenizer.encode(text)
        return {
            'input_ids': encoding.ids,
            'attention_mask': [1] * len(encoding.ids),
            'token_type_ids': [0] * len(encoding.ids)
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size"""
        return self.vocab_size