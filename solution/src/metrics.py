import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Tuple
import nltk
import torch
from collections import Counter

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Import specific metrics packages
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from bert_score import score as bert_score
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("Some metrics packages are not available. Install with: pip install rouge-score nltk bert-score transformers")

class IndicMetrics:
    """
    A class for computing various NLP evaluation metrics for Indic languages.
    Includes implementations of BLEU, ROUGE, and BERTScore with Indic language adaptations.
    """
    
    def __init__(self, lang_code: str = 'en'):
        """
        Initialize metrics calculator with language-specific settings
        
        Args:
            lang_code: Language code ('te' for Telugu, 'hi' for Hindi, 'en' for English)
        """
        self.lang_code = lang_code
        self.language_name = self._get_language_name()
        
        # Initialize language-specific tokenizers
        self._initialize_tokenizers()
        
        # Check if required packages are available
        if not METRICS_AVAILABLE:
            logger.warning(f"Required metrics packages not available. Some methods may not work.")
    
    def _get_language_name(self) -> str:
        """Returns the full language name from language code"""
        language_map = {
            'te': 'Telugu',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'ml': 'Malayalam',
            'kn': 'Kannada',
            'pa': 'Punjabi',
            'gu': 'Gujarati',
            'en': 'English'
        }
        return language_map.get(self.lang_code, 'Unknown')
    
    def _initialize_tokenizers(self):
        """Initialize language-specific tokenizers"""
        # Default to NLTK's word tokenizer
        self.word_tokenizer = nltk.word_tokenize
        self.sent_tokenizer = nltk.sent_tokenize
        
        # For Indic languages, we may need specialized tokenizers
        # This could be extended with more sophisticated Indic NLP tokenizers
        if self.lang_code in ['hi', 'te', 'bn', 'ta', 'ml', 'kn', 'pa', 'gu']:
            # Define Indic language tokenization logic
            self.word_tokenizer = self._indic_word_tokenize
            self.sent_tokenizer = self._indic_sent_tokenize
    
    def _indic_word_tokenize(self, text: str) -> List[str]:
        """Tokenize Indic language text into words"""
        # Basic Indic word tokenization - can be improved with specific rules for each language
        # This is a simplified version - in a production system, use a dedicated Indic NLP library
        
        # Remove common punctuation and special characters
        text = text.replace('।', '.').replace('॥', '.').replace('।।', '.')
        
        # For Telugu, handle specific punctuation
        if self.lang_code == 'te':
            text = text.replace('|', ' ').replace('।', '.').replace('॥', '.')
        
        # Split on whitespace after cleaning
        tokens = []
        for word in text.split():
            # Further split if there are punctuation marks attached
            clean_word = ''
            for char in word:
                if char in '.,;?!':
                    if clean_word:
                        tokens.append(clean_word)
                        clean_word = ''
                    tokens.append(char)
                else:
                    clean_word += char
            if clean_word:
                tokens.append(clean_word)
        
        return tokens
    
    def _indic_sent_tokenize(self, text: str) -> List[str]:
        """Tokenize Indic language text into sentences"""
        # Basic Indic sentence tokenization
        # Replace Indic punctuation with their equivalents
        text = text.replace('।', '.').replace('॥', '.').replace('।।', '.')
        
        # Split on common sentence delimiters
        sentences = []
        for sent in nltk.sent_tokenize(text):
            sentences.append(sent.strip())
        
        return sentences

    def calculate_bleu(self, 
                      reference: Union[str, List[str]], 
                      candidate: str,
                      weights: Tuple[float, ...] = None) -> Dict[str, float]:
        """
        Calculate BLEU score for Indic languages with appropriate adaptations.
        
        Args:
            reference: Reference text(s) or list of tokenized reference texts
            candidate: Candidate text to evaluate
            weights: Optional weights for n-grams (default: equal weights up to 4-grams)
            
        Returns:
            Dictionary containing BLEU score and component scores
        """
        if not METRICS_AVAILABLE:
            return {"error": "NLTK BLEU scoring not available"}
        
        # Default to BLEU-4 with equal weights if not specified
        if weights is None:
            weights = (0.25, 0.25, 0.25, 0.25)  # BLEU-4
        
        # Process inputs to expected format
        if isinstance(reference, str):
            reference = [self.word_tokenizer(reference)]
        elif isinstance(reference[0], str) and not isinstance(reference[0], list):
            reference = [self.word_tokenizer(ref) for ref in reference]
        
        if isinstance(candidate, str):
            candidate = self.word_tokenizer(candidate)
        
        # Apply smoothing function for short sentences (common in summaries)
        smoothing = SmoothingFunction().method1
        
        # Calculate BLEU score
        try:
            if len(reference) == 1:
                bleu_score = sentence_bleu(reference, candidate, weights=weights, smoothing_function=smoothing)
            else:
                bleu_score = corpus_bleu([reference], [candidate], weights=weights, smoothing_function=smoothing)
            
            # Calculate individual n-gram scores
            bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            return {
                "bleu": bleu_score,
                "bleu_1": bleu_1,
                "bleu_2": bleu_2, 
                "bleu_3": bleu_3,
                "bleu_4": bleu_4
            }
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return {"error": f"BLEU calculation failed: {e}"}

    def calculate_rouge(self, 
                      reference: str, 
                      candidate: str,
                      rouge_types: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE score for the text with Indic language adaptations.
        
        Args:
            reference: Reference text
            candidate: Candidate text to evaluate
            rouge_types: List of ROUGE metrics to compute (default: ['rouge1', 'rouge2', 'rougeL'])
            
        Returns:
            Dictionary containing ROUGE scores
        """
        if not METRICS_AVAILABLE:
            return {"error": "rouge-score package not available"}
        
        # Default ROUGE metrics if not specified
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        
        try:
            # Initialize scorer with specified metrics
            scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
            
            # For Indic languages, we might want to adapt the tokenization
            # Currently using the default tokenization from rouge-score
            
            # Calculate scores
            rouge_scores = scorer.score(reference, candidate)
            
            # Format results as a nested dictionary
            results = {}
            for rouge_type, score in rouge_scores.items():
                results[rouge_type] = {
                    'precision': score.precision,
                    'recall': score.recall,
                    'fmeasure': score.fmeasure
                }
            
            return results
        except Exception as e:
            logger.error(f"Error calculating ROUGE score: {e}")
            return {"error": f"ROUGE calculation failed: {e}"}

    def calculate_bertscore(self,
                          references: List[str],
                          candidates: List[str],
                          lang: str = None,
                          model_type: str = None) -> Dict[str, List[float]]:
        """
        Calculate BERTScore for Indic languages using appropriate models.
        
        Args:
            references: List of reference texts
            candidates: List of candidate texts to evaluate
            lang: Language code (defaults to self.lang_code)
            model_type: Model to use for BERTScore (default: language-specific model if available)
            
        Returns:
            Dictionary containing BERTScore precision, recall, and F1 scores
        """
        if not METRICS_AVAILABLE:
            return {"error": "bert-score package not available"}
        
        # Use instance language code if not specified
        if lang is None:
            lang = self.lang_code
        
        # Select appropriate model for the language
        if model_type is None:
            # Map language codes to suitable models
            model_map = {
                'hi': 'ai4bharat/indic-bert',  # Hindi
                'te': 'ai4bharat/indic-bert',  # Telugu
                'bn': 'ai4bharat/indic-bert',  # Bengali
                'ta': 'ai4bharat/indic-bert',  # Tamil
                'ml': 'ai4bharat/indic-bert',  # Malayalam
                'kn': 'ai4bharat/indic-bert',  # Kannada
                'pa': 'ai4bharat/indic-bert',  # Punjabi
                'gu': 'ai4bharat/indic-bert',  # Gujarati
                'en': 'roberta-base'           # English
            }
            model_type = model_map.get(lang, 'roberta-base')
        
        try:
            # Calculate BERTScore
            with torch.no_grad():
                P, R, F1 = bert_score(candidates, references, lang=lang, model_type=model_type, verbose=False)
                
            return {
                "precision": P.tolist(),
                "recall": R.tolist(),
                "f1": F1.tolist(),
                "mean_precision": P.mean().item(),
                "mean_recall": R.mean().item(),
                "mean_f1": F1.mean().item()
            }
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            return {"error": f"BERTScore calculation failed: {e}"}

    def evaluate_all_metrics(self,
                           references: List[str],
                           candidates: List[str],
                           lang: str = None) -> Dict[str, Any]:
        """
        Calculate all available metrics for a list of reference and candidate texts.
        
        Args:
            references: List of reference texts
            candidates: List of candidate texts
            lang: Language code (defaults to self.lang_code)
            
        Returns:
            Dictionary containing all metric scores
        """
        if not METRICS_AVAILABLE:
            return {"error": "Metrics packages not available"}
        
        if lang is None:
            lang = self.lang_code
        
        results = {
            "bleu": [],
            "rouge": [],
            "bertscore": {},
        }
        
        # Calculate metrics for each reference-candidate pair
        for ref, cand in zip(references, candidates):
            # BLEU
            bleu_score = self.calculate_bleu(ref, cand)
            results["bleu"].append(bleu_score)
            
            # ROUGE
            rouge_score = self.calculate_rouge(ref, cand)
            results["rouge"].append(rouge_score)
        
        # BERTScore (calculated in batch)
        bertscore = self.calculate_bertscore(references, candidates, lang)
        results["bertscore"] = bertscore
        
        # Calculate average scores
        results["average"] = self._calculate_average_scores(results)
        
        return results
    
    def _calculate_average_scores(self, results: Dict[str, List]) -> Dict[str, float]:
        """Calculate average scores across all metrics"""
        averages = {}
        
        # Average BLEU
        if "bleu" in results and results["bleu"]:
            bleu_scores = [score.get("bleu", 0) for score in results["bleu"] if isinstance(score, dict) and "bleu" in score]
            if bleu_scores:
                averages["bleu"] = sum(bleu_scores) / len(bleu_scores)
            
            # Also average BLEU-1 through BLEU-4
            for i in range(1, 5):
                key = f"bleu_{i}"
                scores = [score.get(key, 0) for score in results["bleu"] if isinstance(score, dict) and key in score]
                if scores:
                    averages[key] = sum(scores) / len(scores)
        
        # Average ROUGE
        if "rouge" in results and results["rouge"]:
            # Get all unique ROUGE types from the results
            rouge_types = set()
            for rouge_result in results["rouge"]:
                if isinstance(rouge_result, dict):
                    rouge_types.update(rouge_result.keys())
            
            # For each ROUGE type, average the F-measures
            for rouge_type in rouge_types:
                if rouge_type == "error":
                    continue
                
                fmeasures = []
                for rouge_result in results["rouge"]:
                    if isinstance(rouge_result, dict) and rouge_type in rouge_result:
                        score = rouge_result[rouge_type].get('fmeasure', 0)
                        fmeasures.append(score)
                
                if fmeasures:
                    averages[f"{rouge_type}_f"] = sum(fmeasures) / len(fmeasures)
        
        # BERTScore averages are already calculated in the function
        if "bertscore" in results and isinstance(results["bertscore"], dict):
            bertscore = results["bertscore"]
            if "mean_f1" in bertscore:
                averages["bertscore_f1"] = bertscore["mean_f1"]
            if "mean_precision" in bertscore:
                averages["bertscore_precision"] = bertscore["mean_precision"]
            if "mean_recall" in bertscore:
                averages["bertscore_recall"] = bertscore["mean_recall"]
        
        return averages


# Utility functions for working with the metrics
def evaluate_summaries(references: List[str], 
                      candidates: List[str], 
                      lang_code: str = 'en') -> Dict[str, Any]:
    """
    Evaluate summaries using multiple metrics
    
    Args:
        references: List of reference summaries
        candidates: List of generated summaries
        lang_code: Language code
        
    Returns:
        Dictionary with evaluation results
    """
    metrics = IndicMetrics(lang_code=lang_code)
    return metrics.evaluate_all_metrics(references, candidates)


def format_metrics_report(metrics_results: Dict[str, Any]) -> str:
    """
    Format metrics results into a readable report
    
    Args:
        metrics_results: Dictionary with metrics results from evaluate_summaries
        
    Returns:
        Formatted string with results
    """
    report = ["=== Summary Evaluation Report ==="]
    
    # Check for errors
    if "error" in metrics_results:
        report.append(f"Error: {metrics_results['error']}")
        return "\n".join(report)
    
    # Add average metrics
    if "average" in metrics_results:
        report.append("\n--- Average Scores ---")
        avg = metrics_results["average"]
        
        # BLEU
        if "bleu" in avg:
            report.append(f"BLEU: {avg['bleu']:.4f}")
        if all(f"bleu_{i}" in avg for i in range(1, 5)):
            report.append(f"BLEU-1/2/3/4: {avg['bleu_1']:.4f}/{avg['bleu_2']:.4f}/{avg['bleu_3']:.4f}/{avg['bleu_4']:.4f}")
        
        # ROUGE
        rouge_scores = []
        for key in avg:
            if key.startswith("rouge") and key.endswith("_f"):
                rouge_type = key.replace("_f", "")
                rouge_scores.append(f"{rouge_type}: {avg[key]:.4f}")
        if rouge_scores:
            report.append("ROUGE F1: " + ", ".join(rouge_scores))
        
        # BERTScore
        if "bertscore_f1" in avg:
            report.append(f"BERTScore F1: {avg['bertscore_f1']:.4f}")
            report.append(f"BERTScore P/R: {avg['bertscore_precision']:.4f}/{avg['bertscore_recall']:.4f}")
    
    # Add detailed scores for individual examples if available
    if len(metrics_results.get("bleu", [])) > 0 and len(metrics_results["bleu"]) < 10:  # Only show details if there aren't too many examples
        report.append("\n--- Individual Example Scores ---")
        for i, (bleu, rouge) in enumerate(zip(metrics_results["bleu"], metrics_results["rouge"])):
            report.append(f"\nExample #{i+1}:")
            report.append(f"  BLEU: {bleu.get('bleu', 0):.4f}")
            
            # Add ROUGE scores for this example
            for rouge_type, scores in rouge.items():
                if rouge_type != "error" and isinstance(scores, dict) and "fmeasure" in scores:
                    report.append(f"  {rouge_type} F1: {scores['fmeasure']:.4f}")
    
    return "\n".join(report)
