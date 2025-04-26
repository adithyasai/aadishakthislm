import unittest
import sys
import os
import torch
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor
from tokenizer import IndicTokenizer
from model import IndicSLM, IndicSLMConfig

class TestIndicSLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a minimal test config
        cls.config = {
            "model_name": "test_model",
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 512,
            "max_position_embeddings": 128,
            "languages": ["te", "hi"],
            "task": "summarization",
            "training": {
                "batch_size": 2,
                "learning_rate": 1e-4,
                "warmup_steps": 100,
                "max_steps": 1000
            }
        }
        
        # Save test config
        os.makedirs("tests/test_data", exist_ok=True)
        import json
        with open("tests/test_data/test_config.json", "w", encoding='utf-8') as f:
            json.dump(cls.config, f)
            
        # Create test data
        cls.test_data = {
            "te": [
                {"text": "ఇది ఒక పరీక్ష వాక్యం", "summary": "పరీక్ష"},
                {"text": "తెలుగు భాష చాలా అందమైనది", "summary": "తెలుగు అందం"}
            ],
            "hi": [
                {"text": "यह एक परीक्षण वाक्य है", "summary": "परीक्षण"},
                {"text": "हिंदी एक सुंदर भाषा है", "summary": "सुंदर हिंदी"}
            ]
        }
        
        # Save test data
        import pandas as pd
        for lang in ["te", "hi"]:
            df = pd.DataFrame(cls.test_data[lang])
            df.to_csv(f"tests/test_data/{lang}_test.csv", index=False, encoding='utf-8')

    def test_data_processor(self):
        processor = DataProcessor("tests/test_data/test_config.json")
        
        # Test text normalization
        test_text = "తెలుగు  భాష"  # Double space
        normalized = processor.normalize_text(test_text, "te")
        self.assertEqual(normalized.count(" "), 1)
        
        # Test dataset preparation
        dataset = processor.prepare_dataset("tests/test_data/te_test.csv", "te")
        self.assertEqual(len(dataset), 2)
        self.assertTrue("text" in dataset[0])
        self.assertTrue("summary" in dataset[0])

    def test_tokenizer(self):
        tokenizer = IndicTokenizer("tests/test_data/test_config.json")
        
        # Create a small test file for tokenizer training
        with open("tests/test_data/tokenizer_train.txt", "w", encoding="utf-8") as f:
            for lang_data in self.test_data.values():
                for item in lang_data:
                    f.write(item["text"] + "\n")
                    f.write(item["summary"] + "\n")
        
        # Test tokenizer training
        tokenizer.train_tokenizer(
            ["tests/test_data/tokenizer_train.txt"],
            "tests/test_data",
            "test_tokenizer"
        )
        
        # Test encoding
        encoded = tokenizer.encode("తెలుగు", "te")
        self.assertIn("input_ids", encoded)
        self.assertIn("attention_mask", encoded)

    def test_model(self):
        config = IndicSLMConfig(**self.config)
        model = IndicSLM(config)
        
        # Test forward pass
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.zeros_like(input_ids)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        self.assertIn("logits", outputs)
        self.assertEqual(outputs["logits"].shape, (batch_size, seq_length, config.vocab_size))

if __name__ == '__main__':
    unittest.main()