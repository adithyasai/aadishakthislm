import sys
import os
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer

# Setup paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"

def test_model_initialization():
    # Load configuration
    config_path = "configs/model_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"Configuration loaded from {config_path}")
    print(f"Vocabulary size in config: {config['vocab_size']}")
    
    # Initialize model configuration
    model_config = IndicSLMConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config.get('n_embd', 768),
        num_hidden_layers=config.get('n_layer', 6),
        num_attention_heads=config.get('n_head', 12),
        pad_token_id=config.get('pad_token_id', 0)
    )
    
    print(f"Model config vocabulary size: {model_config.vocab_size}")
    
    # Initialize model
    model = IndicSLM(model_config)
    
    # Get the size of the embedding matrix
    embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"Model embedding matrix size: {embedding_size}")
    
    # Initialize tokenizer
    tokenizer = IndicTokenizer(config_path)
    print(f"Tokenizer vocabulary size: {tokenizer.sp_model.get_piece_size()}")
    
    # Test a simple forward pass
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print(f"Model forward pass successful! Output logits shape: {outputs['logits'].shape}")
        print("Test completed successfully - model is properly initialized with the expanded vocabulary!")
    except Exception as e:
        print(f"Error during forward pass: {e}")

def display_tokenization(text, tokenizer, lang=None):
    """Display how a text is tokenized by showing each token."""
    if lang:
        text = f"<{lang}> {text}"
    
    # Encode the text
    encoding = tokenizer.encode(text)
    input_ids = encoding["input_ids"]
    
    # Get all tokens
    tokens = []
    for token_id in input_ids:
        token = tokenizer.sp_model.id_to_piece(token_id)
        tokens.append(token)
    
    # Print original text
    print(f"Original text: {text}")
    print(f"Token count: {len(tokens)}")
    print("Tokens:", " ".join([f"[{t}]" for t in tokens]))
    print("-" * 80)

def compare_tokenization():
    """Compare tokenization of different languages with the expanded vocabulary."""
    # Initialize tokenizer with expanded vocabulary
    tokenizer = IndicTokenizer(str(CONFIG_PATH))
    
    # Print tokenizer information
    print(f"Tokenizer vocabulary size: {tokenizer.sp_model.get_piece_size()}")
    print(f"First 10 tokens: {[tokenizer.sp_model.id_to_piece(i) for i in range(10)]}")
    print(f"Special tokens: BOS={tokenizer.bos_token}, EOS={tokenizer.eos_token}, PAD={tokenizer.pad_token}, UNK={tokenizer.unk_token}")
    print("-" * 80)
    
    # Test multiple languages
    display_tokenization("भारत एक विविधतापूर्ण देश है जहां कई भाषाएँ बोली जाती हैं।", tokenizer, "hi")
    display_tokenization("తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా మాట్లాడబడే ద్రావిడ భాష.", tokenizer, "te")
    display_tokenization("বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গ রাজ্যের প্রধান ভাষা।", tokenizer, "bn")
    display_tokenization("English is widely used in India for business, education, and administration.", tokenizer, "en")
    display_tokenization("भारत is a diverse देश with many भाषाएँ and संस्कृतियाँ.", tokenizer, "en")

if __name__ == "__main__":
    test_model_initialization()
    compare_tokenization()