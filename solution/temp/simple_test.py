import os
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"
MODEL_DIR = BASE_DIR / "models" / "final_model"

def simple_test():
    # Sample texts
    te_text = """తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష."""
    hi_text = """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है।"""
    
    # Use a pre-trained model to test if the system is working correctly
    try:
        logger.info(f"Testing with pre-trained GPT-2 tiny model...")
        config = AutoConfig.from_pretrained("sshleifer/tiny-gpt2")
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        
        # Encode sample text
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids, 
                max_length=20,
                do_sample=True
            )
        
        # Decode and print output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\nSample GPT-2 generation:")
        print("-" * 80)
        print(generated_text)
        logger.info("Pre-trained model test successful!")
        
    except Exception as e:
        logger.error(f"Error with pre-trained model: {e}")
    
    # Now try to load our trained model
    try:
        logger.info(f"Loading custom model from {MODEL_DIR}...")
        print(f"\nTrying to load model from: {MODEL_DIR}")
        
        # List files in the model directory
        print("\nFiles in model directory:")
        for file in os.listdir(MODEL_DIR):
            print(f" - {file}")
        
        # Try loading the model configuration
        try:
            config = AutoConfig.from_pretrained(MODEL_DIR)
            print("\nLoaded config successfully!")
            print(f"Config details: {config}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            
        # Try loading the model with error details
        try:
            logger.info(f"Loading model...")
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
            print("\nLoaded model successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            
        # Try loading tokenizer from model config
        try:
            from src.tokenizer import IndicTokenizer
            import json
            
            print("\nLoading custom tokenizer...")
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            tokenizer = IndicTokenizer(CONFIG_PATH)
            print("Loaded tokenizer successfully!")
            
            # Test tokenization
            te_encoded = tokenizer.encode(f"<te> {te_text}")
            hi_encoded = tokenizer.encode(f"<hi> {hi_text}")
            
            print(f"\nTokenized Telugu text (first 10 tokens): {te_encoded['input_ids'][:10]}")
            print(f"Tokenized Hindi text (first 10 tokens): {hi_encoded['input_ids'][:10]}")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            
    except Exception as e:
        logger.error(f"Error with custom model testing: {e}")
        
    print("\nTest completed!")

if __name__ == "__main__":
    simple_test()