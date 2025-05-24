import os
import sys
import json
import torch
import logging
import argparse
from pathlib import Path
from collections import defaultdict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer

# Setup paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"
EXPANDED_MODEL_DIR = BASE_DIR / "models" / "expanded_vocab_model" / "final_model"
ORIGINAL_MODEL_DIR = BASE_DIR / "models" / "final_model"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultilingualEvaluator:
    """Evaluates model performance across multiple Indian languages."""
    
    def __init__(self, model_path, config_path=str(CONFIG_PATH)):
        self.config_path = config_path
        self.model_path = model_path
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Initialize tokenizer
        self.tokenizer = IndicTokenizer(config_path)
        
        # Load model configuration
        model_config_path = os.path.join(model_path, "config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config_dict = json.load(f)
            self.model_config = IndicSLMConfig(**model_config_dict)
        else:
            # Fallback to default config
            self.model_config = IndicSLMConfig(
                vocab_size=self.config.get('vocab_size', 50000),
                hidden_size=self.config.get('n_embd', 384),
                num_hidden_layers=self.config.get('n_layer', 6),
                num_attention_heads=self.config.get('n_head', 6),
                pad_token_id=self.config.get('pad_token_id', 0)
            )
        
        # Load model
        self.model = self._load_model()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Print model info
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Vocabulary size: {self.model_config.vocab_size}")
        logger.info(f"Tokenizer vocabulary size: {self.tokenizer.sp_model.get_piece_size()}")
        logger.info(f"Using device: {self.device}")
    
    def _load_model(self):
        """Load the model from the specified path."""
        model = IndicSLM(self.model_config)
        
        # Check if there's a model file to load
        model_safetensors_path = os.path.join(self.model_path, "model.safetensors")
        pytorch_model_path = os.path.join(self.model_path, "pytorch_model.bin")
        model_path = os.path.join(self.model_path, "model.bin")
        
        if os.path.exists(model_safetensors_path):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(model_safetensors_path)
                model.load_state_dict(state_dict)
                logger.info(f"Model loaded from safetensors file: {model_safetensors_path}")
            except Exception as e:
                logger.warning(f"Failed to load safetensors model: {e}")
                logger.warning("Using untrained model.")
        elif os.path.exists(pytorch_model_path):
            try:
                model.load_state_dict(torch.load(pytorch_model_path, map_location="cpu"))
                logger.info(f"Model loaded from: {pytorch_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load PyTorch model: {e}")
                logger.warning("Using untrained model.")
        elif os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                logger.info(f"Model loaded from: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                logger.warning("Using untrained model.")
        else:
            logger.warning(f"No model file found at {self.model_path}. Using untrained model.")
        
        return model
    
    def evaluate_text_coverage(self, texts, lang=None):
        """
        Evaluate how well the model's vocabulary covers the given texts.
        Returns the percentage of unknown tokens.
        """
        total_tokens = 0
        unknown_tokens = 0
        
        for text in texts:
            if lang:
                text = f"<{lang}> {text}"
            
            # Encode the text
            encoding = self.tokenizer.encode(text)
            tokens = encoding["input_ids"]
            
            # Count unknown tokens
            for token in tokens:
                total_tokens += 1
                if token == self.tokenizer.unk_token_id:
                    unknown_tokens += 1
        
        if total_tokens == 0:
            return 0.0
        
        coverage = 1.0 - (unknown_tokens / total_tokens)
        return coverage * 100.0
    
    def evaluate_language_coverage(self):
        """Evaluate vocabulary coverage across different Indian languages."""
        # Test texts in different languages
        language_samples = {
            "hindi": [
                "भारत एक विविधतापूर्ण देश है जहां कई भाषाएँ बोली जाती हैं।",
                "हिंदी भारत की प्रमुख भाषाओं में से एक है और उत्तर भारत में व्यापक रूप से बोली जाती है।",
                "भारतीय संस्कृति विविधता और समृद्धि से भरपूर है।"
            ],
            "telugu": [
                "తెలుగు భాష దక్షిణ భారతదేశంలో ఆంధ్రప్రదేశ్ మరియు తెలంగాణలో ప్రధానంగా మాట్లాడబడుతుంది.",
                "భారతదేశంలో అనేక భాషలు ఉన్నాయి, ప్రతి ప్రాంతానికి దాని సొంత సంస్కృతి, సంప్రదాయాలు ఉన్నాయి.",
                "తెలుగు నాలుగు ద్రావిడ భాషల్లో ఒకటి మరియు దాని చరిత్ర చాలా ప్రాచీనమైనది."
            ],
            "bengali": [
                "বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গ রাজ্যের সরকারি ভাষা।",
                "বাংলা সাহিত্য বিশ্বের অন্যতম সমৃদ্ধ সাহিত্যের ঐতিহ্য।",
                "রবীন্দ্রনাথ ঠাকুর একজন বিখ্যাত বাঙালি কবি, লেখক এবং সংগীতজ্ঞ যিনি প্রথম অ-ইউরোপীয় হিসাবে সাহিত্যে নোবেল পুরস্কার জিতেছিলেন।"
            ],
            "tamil": [
                "தமிழ் மொழி தென்னிந்தியாவில் பேசப்படும் திராவிட மொழிகளில் ஒன்றாகும்.",
                "தமிழ் மொழி இந்திய துணைக்கண்டத்தின் மிகவும் பழமையான மொழிகளில் ஒன்றாகும்.",
                "தமிழக கலாச்சாரம் மற்றும் பாரம்பரியம் மிகவும் பழமையானது மற்றும் செழுமையானது."
            ],
            "malayalam": [
                "മലയാളം ദ്രാവിഡ ഭാഷാ കുടുംബത്തിൽ ഉൾപ്പെടുന്ന ഒരു ഭാഷയാണ്, ഇത് പ്രധാനമായും ഇന്ത്യയിലെ കേരള സംസ്ഥാനത്ത് സംസാരിക്കുന്നു.",
                "മലയാളത്തിന് സമ്പന്നമായ സാഹിത്യ പാരമ്പര്യമുണ്ട്.",
                "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ ഭാഗത്താണ് സ്ഥിതി ചെയ്യുന്നത്, ഇത് അതിന്റെ സുന്ദരമായ കായലുകൾക്കും തെങ്ങിൻ തോപ്പുകൾക്കും പ്രശസ്തമാണ്."
            ],
            "kannada": [
                "ಕನ್ನಡ ಭಾಷೆಯು ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.",
                "ಕನ್ನಡ ಭಾಷೆಯು ಸುಮಾರು ೨೦೦೦ ವರ್ಷಗಳ ಇತಿಹಾಸವನ್ನು ಹೊಂದಿದೆ.",
                "ಸಾಹಿತ್ಯ, ಕಲೆ ಮತ್ತು ವಾಸ್ತುಶಿಲ್ಪದಲ್ಲಿ ಕನ್ನಡ ಸಂಸ್ಕೃತಿ ಸಮೃದ್ಧವಾಗಿದೆ."
            ],
            "marathi": [
                "मराठी ही भारतातील महाराष्ट्र राज्याची अधिकृत भाषा आहे.",
                "मराठी ही इंडो-आर्यन भाषा परिवारातील एक भाषा आहे.",
                "मराठी साहित्य आणि संस्कृती समृद्ध परंपरा आणि इतिहास आहे."
            ],
            "punjabi": [
                "ਪੰਜਾਬੀ ਭਾਰਤ ਦੇ ਪੰਜਾਬ ਰਾਜ ਦੀ ਅਧਿਕਾਰਕ ਭਾਸ਼ਾ ਹੈ।",
                "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਇੰਡੋ-ਆਰੀਅਨ ਪਰਿਵਾਰ ਨਾਲ ਸਬੰਧਤ ਹੈ।",
                "ਪੰਜਾਬੀ ਸੱਭਿਆਚਾਰ ਵਿੱਚ ਭੰਗੜਾ ਅਤੇ ਗਿੱਧਾ ਵਰਗੇ ਪ੍ਰਸਿੱਧ ਨਾਚ ਸ਼ਾਮਲ ਹਨ।"
            ],
            "gujarati": [
                "ગુજરાતી ભાષા ભારતના ગુજરાત રાજ્યની આધિકારિક ભાષા છે.",
                "ગુજરાતી એ ઇન્ડો-આર્યન ભાષા કુટુંબની એક ભાષા છે.",
                "ગુજરાતી સંસ્કૃતિ તેની સમૃદ્ધ વારસો અને ઉત્સવો માટે જાણીતી છે."
            ],
            "english": [
                "India is a diverse country with many languages and cultures.",
                "English is widely used in India for business, education, and administration.",
                "The Indian constitution recognizes Hindi and English as official languages for government business."
            ],
            "mixed": [
                "भारत is a diverse देश with many भाषाएँ and संस्कृतियाँ.",
                "తెలుగు is one of the oldest భాష in India with a rich చరిత్ర.",
                "The ಕರ್ನಾಟಕ government promotes ಕನ್ನಡ language and culture."
            ]
        }
        
        results = {}
        for lang, texts in language_samples.items():
            coverage = self.evaluate_text_coverage(texts, lang if lang != "mixed" else None)
            results[lang] = coverage
            logger.info(f"{lang.capitalize()} language coverage: {coverage:.2f}%")
        
        return results
    
    def evaluate_token_efficiency(self, texts, lang=None):
        """
        Evaluate token efficiency - how many tokens are needed to encode the texts.
        Lower token count means better efficiency.
        """
        total_text_chars = 0
        total_tokens = 0
        
        for text in texts:
            if lang:
                text = f"<{lang}> {text}"
            
            total_text_chars += len(text)
            encoding = self.tokenizer.encode(text)
            total_tokens += len(encoding["input_ids"])
        
        # Characters per token ratio
        chars_per_token = total_text_chars / total_tokens if total_tokens > 0 else 0
        return chars_per_token
    
    def compare_model_outputs(self, prompt, max_length=100):
        """Generate text from a prompt and evaluate the output quality."""
        input_text = prompt
        
        # Encode the input
        encoded_input = self.tokenizer.encode(input_text)
        input_ids = torch.tensor([encoded_input["input_ids"]]).to(self.device)
        attention_mask = torch.tensor([encoded_input["attention_mask"]]).to(self.device)
        
        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decode the output
        output_text = self.tokenizer.decode(output_ids[0])
        
        return output_text
    
    def run_full_evaluation(self):
        """Run a comprehensive evaluation of the model."""
        logger.info("Starting comprehensive evaluation...")
        
        # 1. Evaluate language coverage
        logger.info("\n--- Vocabulary Coverage Evaluation ---")
        coverage_results = self.evaluate_language_coverage()
        
        # 2. Evaluate token efficiency
        logger.info("\n--- Token Efficiency Evaluation ---")
        efficiency_results = {}
        
        # Use the same language samples as in coverage evaluation
        language_samples = {
            "hindi": ["भारत एक विविधतापूर्ण देश है जहां कई भाषाएँ बोली जाती हैं।"],
            "telugu": ["తెలుగు భాష దక్షిణ భారతదేశంలో ఆంధ్రప్రదేశ్ మరియు తెలంగాణలో ప్రధానంగా మాట్లాడబడుతుంది."],
            "bengali": ["বাংলা ভাষা বাংলাদেশ এবং ভারতের পশ্চিমবঙ্গ রাজ্যের সরকারি ভাষা।"],
            "english": ["English is widely used in India for business, education, and administration."]
        }
        
        for lang, texts in language_samples.items():
            efficiency = self.evaluate_token_efficiency(texts, lang)
            efficiency_results[lang] = efficiency
            logger.info(f"{lang.capitalize()} chars per token: {efficiency:.2f}")
        
        # 3. Generate sample outputs
        logger.info("\n--- Text Generation Samples ---")
        
        generation_prompts = {
            "hindi": "<hi> भारत की संस्कृति",
            "telugu": "<te> తెలుగు భాష యొక్క చరిత్ర",
            "english": "The history of Indian languages",
            "mixed": "The relationship between హిందీ and తెలుగు languages"
        }
        
        generation_results = {}
        for lang, prompt in generation_prompts.items():
            logger.info(f"\nPrompt ({lang}): {prompt}")
            output = self.compare_model_outputs(prompt)
            generation_results[lang] = output
            logger.info(f"Generated output: {output}")
        
        return {
            "coverage": coverage_results,
            "efficiency": efficiency_results,
            "generation": generation_results
        }

def compare_models():
    """Compare the performance of the original model vs the expanded vocabulary model."""
    
    results = {}
    
    # Check if original model exists
    if os.path.exists(ORIGINAL_MODEL_DIR):
        logger.info("\n=== Evaluating Original Model (32K vocabulary) ===")
        original_evaluator = MultilingualEvaluator(str(ORIGINAL_MODEL_DIR))
        results["original"] = original_evaluator.run_full_evaluation()
    else:
        logger.warning(f"Original model not found at {ORIGINAL_MODEL_DIR}")
    
    # Check if expanded vocab model exists
    if os.path.exists(EXPANDED_MODEL_DIR):
        logger.info("\n=== Evaluating Expanded Vocabulary Model (50K vocabulary) ===")
        expanded_evaluator = MultilingualEvaluator(str(EXPANDED_MODEL_DIR))
        results["expanded"] = expanded_evaluator.run_full_evaluation()
    else:
        logger.warning(f"Expanded vocabulary model not found at {EXPANDED_MODEL_DIR}")
    
    # Compare results if both models were evaluated
    if "original" in results and "expanded" in results:
        logger.info("\n=== Comparison of Models ===")
        
        # Compare vocabulary coverage
        logger.info("\n--- Vocabulary Coverage Comparison ---")
        for lang in results["original"]["coverage"]:
            orig_cov = results["original"]["coverage"][lang]
            exp_cov = results["expanded"]["coverage"][lang]
            diff = exp_cov - orig_cov
            logger.info(f"{lang.capitalize()}: Original={orig_cov:.2f}%, Expanded={exp_cov:.2f}%, Difference={diff:+.2f}%")
        
        # Compare token efficiency
        logger.info("\n--- Token Efficiency Comparison ---")
        for lang in results["original"]["efficiency"]:
            if lang in results["expanded"]["efficiency"]:
                orig_eff = results["original"]["efficiency"][lang]
                exp_eff = results["expanded"]["efficiency"][lang]
                diff = exp_eff - orig_eff
                logger.info(f"{lang.capitalize()}: Original={orig_eff:.2f}, Expanded={exp_eff:.2f}, Difference={diff:+.2f}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate expanded vocabulary model performance")
    parser.add_argument("--original", type=str, default=str(ORIGINAL_MODEL_DIR), help="Path to original model")
    parser.add_argument("--expanded", type=str, default=str(EXPANDED_MODEL_DIR), help="Path to expanded vocab model")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH), help="Path to config file")
    parser.add_argument("--compare", action="store_true", help="Compare original and expanded models")
    parser.add_argument("--output", type=str, help="Path to save evaluation results as JSON")
    args = parser.parse_args()
    
    # Update paths if provided
    if args.original:
        ORIGINAL_MODEL_DIR = Path(args.original)
    if args.expanded:
        EXPANDED_MODEL_DIR = Path(args.expanded)
    
    if args.compare:
        results = compare_models()
    else:
        # Evaluate only the expanded vocab model
        evaluator = MultilingualEvaluator(str(EXPANDED_MODEL_DIR), args.config)
        results = evaluator.run_full_evaluation()
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {args.output}")