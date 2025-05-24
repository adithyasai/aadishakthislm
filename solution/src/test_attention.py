import os
import sys
import torch
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IndicSLMConfig, IndicSLM
from src.tokenizer import IndicTokenizer

# Setup paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"

def test_attention_mechanisms():
    """Test and compare different attention mechanisms for Indic languages."""
    
    # Load configuration
    with open(CONFIG_PATH) as f:
        config_data = json.load(f)
    
    # Initialize tokenizer
    tokenizer = IndicTokenizer(str(CONFIG_PATH))
    print(f"Tokenizer vocabulary size: {tokenizer.sp_model.get_piece_size()}")
    
    # Sample texts in Hindi, Telugu, and mixed language
    sample_texts = {
        "hindi": "<hi> राम और श्याम दोनों बचपन के दोस्त हैं। वे एक ही स्कूल में पढ़ते थे और अब एक ही कंपनी में काम करते हैं।",
        "telugu": "<te> రాము మరియు శ్యాము చిన్నప్పటి నుండి స్నేహితులు. వారు ఒకే స్కూల్లో చదివారు మరియు ఇప్పుడు ఒకే కంపెనీలో పని చేస్తున్నారు.",
        "mixed": "<en> राम and श्याम have been friends since childhood. वे एक ही school में पढ़ते थे and now work in the same कंपनी."
    }
    
    # Encode samples
    encoded_samples = {}
    for name, text in sample_texts.items():
        encoding = tokenizer.encode(text)
        encoded_samples[name] = {
            "input_ids": torch.tensor([encoding["input_ids"]]),
            "attention_mask": torch.tensor([encoding["attention_mask"]])
        }
    
    # Test each attention mechanism
    attention_types = ["default", "relative", "rotary"]
    
    for attention_type in attention_types:
        print(f"\n----- Testing {attention_type.upper()} attention mechanism -----")
        
        # Create configuration with specific attention type
        config = IndicSLMConfig(
            vocab_size=config_data["vocab_size"],
            hidden_size=config_data.get("n_embd", 384),
            num_hidden_layers=config_data.get("n_layer", 6),
            num_attention_heads=config_data.get("n_head", 6),
            attention_type=attention_type
        )
        
        # Initialize model
        model = IndicSLM(config)
        print(f"Model initialized with {attention_type} attention")
        
        # Process each sample
        for name, encoded in encoded_samples.items():
            # Forward pass
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if start_time:
                    start_time.record()
                
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"]
                )
                
                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)
                else:
                    elapsed_time = None
            
            # Print results
            logits = outputs["logits"]
            print(f"\n{name.capitalize()} sample:")
            print(f"  Sequence length: {encoded['input_ids'].size(1)}")
            print(f"  Output logits shape: {logits.shape}")
            if elapsed_time:
                print(f"  Inference time: {elapsed_time:.2f} ms")
            
            # Generate next token prediction
            next_token_logits = logits[0, -1, :]
            top_k = 5
            top_k_indices = torch.topk(next_token_logits, top_k).indices
            
            print(f"  Top {top_k} next token predictions:")
            for i, token_id in enumerate(top_k_indices):
                token = tokenizer.sp_model.id_to_piece(token_id.item())
                score = next_token_logits[token_id].item()
                print(f"    {i+1}. '{token}' (score: {score:.2f})")
    
    print("\nAttention mechanism test completed!")

if __name__ == "__main__":
    test_attention_mechanisms()