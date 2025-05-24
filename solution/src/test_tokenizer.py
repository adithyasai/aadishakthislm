import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import IndicTokenizer

def test_tokenizer():
    # Initialize the tokenizer
    tokenizer = IndicTokenizer()
    
    # Print tokenizer info
    print(f"Vocabulary size: {tokenizer.sp_model.get_piece_size()}")
    
    # Test encoding
    test_texts = [
        "नमस्ते भारत",
        "తెలుగు భాష చాలా అందమైనది",
        "ಕನ್ನಡ ನಮ್ಮ ಹೆಮ್ಮೆಯ ಭಾಷೆ",
        "हिन्दी भारत की राष्ट्रभाषा है",
        "This is an English sentence mixed with हिन्दी",
    ]
    
    for text in test_texts:
        print("\nOriginal text:", text)
        encoded = tokenizer.encode(text)
        print("Token IDs:", encoded["input_ids"])
        print("Token pieces:", [tokenizer.sp_model.id_to_piece(id) for id in encoded["input_ids"]])
        decoded = tokenizer.decode(encoded["input_ids"])
        print("Decoded text:", decoded)

if __name__ == "__main__":
    test_tokenizer()