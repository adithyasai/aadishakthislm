import sentencepiece as spm
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup base paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = BASE_DIR / "models"
TOKENIZER_MODEL_PATH = MODELS_DIR / "indic_slm_tokenizer.model"
TOKENIZER_VOCAB_PATH = MODELS_DIR / "indic_slm_tokenizer.vocab"

def initialize_tokenizer():
    """Initialize a simple tokenizer if one doesn't exist"""
    if os.path.exists(TOKENIZER_MODEL_PATH) and os.path.exists(TOKENIZER_VOCAB_PATH):
        logger.info(f"Tokenizer already exists at {TOKENIZER_MODEL_PATH}")
        return
    
    logger.info("Creating simple tokenizer model...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Training data path
    input_path = MODELS_DIR / "tokenizer_train.txt"
    
    # Create a simple training file if it doesn't exist
    if not os.path.exists(input_path):
        logger.info("Creating simple training file for tokenizer...")
        with open(input_path, 'w', encoding='utf-8') as f:
            # Add some basic Telugu and Hindi text
            telugu_text = "తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష."
            hindi_text = "हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है।"
            
            f.write(telugu_text + "\n")
            f.write(hindi_text + "\n")
    
    # Train a small SentencePiece model
    model_prefix = str(MODELS_DIR / "indic_slm_tokenizer")
    
    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=model_prefix,
        vocab_size=1000,  # Small vocab size for quick initialization
        model_type="unigram",
        character_coverage=0.9995,
        user_defined_symbols=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "<te>", "<hi>"],
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        input_sentence_size=10000,
        shuffle_input_sentence=True
    )
    
    logger.info(f"Simple tokenizer created at {model_prefix}.model")

if __name__ == "__main__":
    initialize_tokenizer()