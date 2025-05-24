import sentencepiece as spm
from typing import Dict, List, Union, Optional
import json
import os

class IndicTokenizer:
    def __init__(self, config_path: str = "configs/model_config.json"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.sp_model = spm.SentencePieceProcessor()
        model_path = os.path.join("models", "indic_slm_tokenizer.model")
        if os.path.exists(model_path):
            self.sp_model.load(model_path)
        
        # Define special tokens
        self.pad_token = "[PAD]"
        self.pad_token_id = 0  # Usually 0 in SentencePiece
        self.unk_token = "[UNK]"
        self.unk_token_id = 1  # Usually 1 in SentencePiece
        self.bos_token = "[BOS]"
        self.bos_token_id = 2
        self.eos_token = "[EOS]"
        self.eos_token_id = 3
    
    def encode(self, text: str) -> Dict[str, List[int]]:
        """Encode text to token ids"""
        if not text:
            return {"input_ids": [], "attention_mask": []}
        
        # Add BOS and EOS tokens
        text = f"{self.bos_token} {text} {self.eos_token}"
        
        # Encode the text
        token_ids = self.sp_model.encode(text)
        attention_mask = [1] * len(token_ids)
        
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token ids back to text"""
        text = self.sp_model.decode(token_ids)
        # Remove special tokens if present
        text = text.replace(self.bos_token, "").replace(self.eos_token, "").strip()
        return text
        
    def train_tokenizer(
        self, 
        input_files: List[str], 
        output_dir: str, 
        model_prefix: str,
        vocab_size: Optional[int] = None
    ) -> None:
        """
        Train a SentencePiece tokenizer on input files
        
        Args:
            input_files: List of text files to train on
            output_dir: Directory to save the tokenizer model
            model_prefix: Prefix for the model files
            vocab_size: Size of vocabulary (if None, uses value from config)
        """
        if vocab_size is None:
            vocab_size = self.config.get("vocab_size", 32000)
        
        print(f"Training SentencePiece tokenizer with vocabulary size: {vocab_size}")
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Special tokens to be added
        user_defined_symbols = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token
        ]
        
        # Train the SentencePiece model
        spm.SentencePieceTrainer.train(
            input=",".join(input_files),
            model_prefix=os.path.join(output_dir, model_prefix),
            vocab_size=vocab_size,
            pad_id=self.pad_token_id,
            unk_id=self.unk_token_id,
            bos_id=self.bos_token_id,
            eos_id=self.eos_token_id,
            user_defined_symbols=user_defined_symbols,
            model_type="bpe",  # Using BPE algorithm
            character_coverage=0.9995,  # High coverage for Indic scripts
            input_sentence_size=1000000,  # Limit the number of training sentences to process
            shuffle_input_sentence=True,  # Shuffle the training data
            normalization_rule_name="nmt_nfkc"  # Standard normalization for NMT
        )
        
        # Load the newly trained model
        self.sp_model.load(os.path.join(output_dir, f"{model_prefix}.model"))
        print(f"Tokenizer trained and saved to {output_dir}/{model_prefix}.model")