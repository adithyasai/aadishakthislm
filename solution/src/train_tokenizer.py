import os
import argparse
from pathlib import Path
import json
from data_processor import DataProcessor
from tokenizer import IndicTokenizer

def train_combined_tokenizer(
    config_path: str = "configs/model_config.json",
    output_dir: str = "models",
    vocab_size: int = None
) -> None:
    """Train tokenizer on combined Telugu and Hindi datasets"""
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Use provided vocab_size or get from config
    if vocab_size is None:
        vocab_size = config.get("vocab_size", 32000)
    
    # Initialize processors
    data_processor = DataProcessor(config_path)
    tokenizer = IndicTokenizer(config_path)
    
    # Set up paths
    data_dir = Path("data/raw")
    train_data_path = Path(output_dir) / "tokenizer_train.txt"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all text from datasets
    print("Collecting training data for tokenizer...")
    with open(train_data_path, 'w', encoding='utf-8') as f:
        # Process Telugu datasets
        print("Processing Telugu datasets...")
        te_datasets = {
            "xlsum": data_dir / "telugu/telugu_XLSum_v2.0/telugu_train.jsonl",
            "tesum": data_dir / "telugu/TeSum/train.jsonl",
            "indiccorp": data_dir / "telugu/ai4bharat/te_indiccorp.jsonl"
        }
        
        for name, path in te_datasets.items():
            if path.exists():
                print(f"Processing {name}...")
                dataset = data_processor.prepare_dataset(str(path), "te")
                for item in dataset:
                    if isinstance(item, dict) and "text" in item:
                        f.write(item["text"] + "\n")
                    if isinstance(item, dict) and "summary" in item:
                        f.write(item["summary"] + "\n")
        
        # Process Hindi datasets
        print("Processing Hindi datasets...")
        hi_datasets = {
            "xlsum": data_dir / "hindi/hindi_XLSum_v2.0/hindi_train.jsonl",
            "sar": data_dir / "hindi/SAR/SARsample.csv",
            "indiccorp": data_dir / "hindi/ai4bharat/hi_indiccorp.jsonl"
        }
        
        for name, path in hi_datasets.items():
            if path.exists():
                print(f"Processing {name}...")
                dataset = data_processor.prepare_dataset(str(path), "hi")
                for item in dataset:
                    if isinstance(item, dict) and "text" in item:
                        f.write(item["text"] + "\n")
                    if isinstance(item, dict) and "summary" in item:
                        f.write(item["summary"] + "\n")
    
    print(f"Training tokenizer with vocabulary size: {vocab_size}...")
    tokenizer.train_tokenizer(
        input_files=[str(train_data_path)],
        output_dir=output_dir,
        model_prefix="indic_slm_tokenizer",
        vocab_size=vocab_size
    )
    print("Tokenizer training complete!")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for Indic languages")
    parser.add_argument("--config", type=str, default="configs/model_config.json", help="Path to model config file")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save tokenizer model")
    parser.add_argument("--vocab_size", type=int, default=None, help="Size of vocabulary (default: from config)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_combined_tokenizer(
        config_path=args.config,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size
    )