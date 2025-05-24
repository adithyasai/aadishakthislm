import unittest
import sys
import os
import torch
import json
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import DataProcessor
from tokenizer import IndicTokenizer
from model import IndicSLM, IndicSLMConfig

class TestIndicSLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up paths relative to solution directory
        cls.solution_dir = os.path.dirname(os.path.dirname(__file__))
        cls.test_data_dir = os.path.join(cls.solution_dir, "tests", "test_data")
        cls.data_dir = os.path.join(cls.solution_dir, "data")
        
        # Create a minimal test config with smaller vocab size
        cls.config = {
            "model_name": "test_model",
            "vocab_size": 500,  # Reduced from 1000 to handle test data size
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
        os.makedirs(cls.test_data_dir, exist_ok=True)
        with open(os.path.join(cls.test_data_dir, "test_config.json"), "w", encoding='utf-8') as f:
            json.dump(cls.config, f)
            
        # Load sample data from real datasets
        cls.real_data = {
            "te": {
                "xlsum": [],
                "tesum": [],
                "indiccorp": []
            },
            "hi": {
                "xlsum": [],
                "indiccorp": []
            }
        }
        
        # Load samples from XLSum
        try:
            with open(os.path.join(cls.data_dir, "raw/telugu/telugu_XLSum_v2.0/telugu_train.jsonl"), 'r', encoding='utf-8') as f:
                cls.real_data["te"]["xlsum"] = [json.loads(line) for line in f][:5]
            with open(os.path.join(cls.data_dir, "raw/hindi/hindi_XLSum_v2.0/hindi_train.jsonl"), 'r', encoding='utf-8') as f:
                cls.real_data["hi"]["xlsum"] = [json.loads(line) for line in f][:5]
        except Exception as e:
            print(f"Warning: Could not load XLSum data: {e}")
        
        # Load samples from TeSum
        try:
            with open(os.path.join(cls.data_dir, "raw/telugu/TeSum/train.jsonl"), 'r', encoding='utf-8') as f:
                cls.real_data["te"]["tesum"] = [json.loads(line) for line in f][:5]
        except Exception as e:
            print(f"Warning: Could not load TeSum data: {e}")
        
        # Load samples from IndicCorp
        try:
            with open(os.path.join(cls.data_dir, "raw/telugu/ai4bharat/te_indiccorp.jsonl"), 'r', encoding='utf-8') as f:
                cls.real_data["te"]["indiccorp"] = [json.loads(line) for line in f][:5]
            with open(os.path.join(cls.data_dir, "raw/hindi/ai4bharat/hi_indiccorp.jsonl"), 'r', encoding='utf-8') as f:
                cls.real_data["hi"]["indiccorp"] = [json.loads(line) for line in f][:5]
        except Exception as e:
            print(f"Warning: Could not load IndicCorp data: {e}")

    def test_data_processor_basic(self):
        """Test basic data processor functionality"""
        config_path = os.path.join(self.test_data_dir, "test_config.json")
        processor = DataProcessor(config_path)
        
        # Test text normalization
        test_text = "తెలుగు  భాష"  # Double space
        normalized = processor.normalize_text(test_text, "te")
        self.assertEqual(normalized.count(" "), 1)
    
    def test_data_processor_xlsum(self):
        """Test data processor with XLSum dataset"""
        config_path = os.path.join(self.test_data_dir, "test_config.json")
        processor = DataProcessor(config_path)
        
        # Save sample XLSum data for testing
        for lang in ["te", "hi"]:
            if self.real_data[lang]["xlsum"]:
                test_file = os.path.join(self.test_data_dir, f"{lang}_xlsum_test.jsonl")
                with open(test_file, 'w', encoding='utf-8') as f:
                    for item in self.real_data[lang]["xlsum"]:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                
                # Test dataset preparation
                dataset = processor.prepare_dataset(test_file, lang)
                self.assertTrue(len(dataset) > 0)
                self.assertTrue("text" in dataset[0])
                if "summary" in dataset[0]:
                    self.assertIsInstance(dataset[0]["summary"], str)
    
    def test_data_processor_indiccorp(self):
        """Test data processor with IndicCorp dataset"""
        config_path = os.path.join(self.test_data_dir, "test_config.json")
        processor = DataProcessor(config_path)
        
        # Save sample IndicCorp data for testing
        for lang in ["te", "hi"]:
            if self.real_data[lang]["indiccorp"]:
                test_file = os.path.join(self.test_data_dir, f"{lang}_indiccorp_test.jsonl")
                with open(test_file, 'w', encoding='utf-8') as f:
                    for item in self.real_data[lang]["indiccorp"]:
                        json.dump(item, f, ensure_ascii=False)
                        f.write('\n')
                
                # Test dataset preparation
                dataset = processor.prepare_dataset(test_file, lang)
                self.assertTrue(len(dataset) > 0)
                self.assertTrue("text" in dataset[0])

    def test_tokenizer(self):
        """Test tokenizer functionality"""
        config_path = os.path.join(self.test_data_dir, "test_config.json")
        tokenizer = IndicTokenizer(config_path)
        
        # Create training data from real samples
        tokenizer_train_path = os.path.join(self.test_data_dir, "tokenizer_train.txt")
        with open(tokenizer_train_path, "w", encoding="utf-8") as f:
            # Write Telugu samples
            for dataset in self.real_data["te"].values():
                for item in dataset:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            f.write(text + "\n")
            
            # Write Hindi samples
            for dataset in self.real_data["hi"].values():
                for item in dataset:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            f.write(text + "\n")
        
        # Test tokenizer training
        tokenizer.train_tokenizer(
            [tokenizer_train_path],
            self.test_data_dir,
            "test_tokenizer"
        )
        
        # Test encoding for both languages
        for lang, code in [("తెలుగు", "te"), ("हिंदी", "hi")]:
            encoded = tokenizer.encode(lang, code)
            self.assertIn("input_ids", encoded)
            self.assertIn("attention_mask", encoded)
            self.assertIsInstance(encoded["input_ids"], list)
            self.assertEqual(len(encoded["input_ids"]), len(encoded["attention_mask"]))

    def test_model(self):
        """Test model functionality"""
        config = IndicSLMConfig(**self.config)
        model = IndicSLM(config)
        
        # Test forward pass with real tokenized data
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
        
        # Test with labels for loss computation
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        outputs_with_labels = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        self.assertIn("loss", outputs_with_labels)
        self.assertIsInstance(outputs_with_labels["loss"], torch.Tensor)
        self.assertEqual(outputs_with_labels["loss"].shape, torch.Size([]))

if __name__ == '__main__':
    unittest.main()