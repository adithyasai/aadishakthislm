import os
import sys
import logging
from pathlib import Path
import json
import random
from typing import List, Dict, Any, Tuple
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path if needed
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import summarizers for comparison
from src.neural_summarizer import NeuralSummarizer
from src.improved_summarizer import ImprovedSummarizer, generate_improved_summary

def load_test_data(file_path: str = None, lang: str = None, sample_size: int = 5) -> List[Dict[str, Any]]:
    """
    Load test data for summarization evaluation
    
    Args:
        file_path: Path to test data file (jsonl format expected)
        lang: Language filter ('te' for Telugu, 'hi' for Hindi)
        sample_size: Number of samples to use
        
    Returns:
        List of test samples with text and reference summary
    """
    # Set default path if not provided
    if not file_path:
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = str(base_dir / "data" / "processed" / "xlsum" / "test.jsonl")
    
    try:
        # Load samples from jsonl file
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    # Filter by language if specified
                    if lang and sample.get('lang') != lang:
                        continue
                    samples.append(sample)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line: {line[:50]}...")
                    continue
        
        # Sample random entries if we have more than requested
        if len(samples) > sample_size:
            samples = random.sample(samples, sample_size)
        
        logger.info(f"Loaded {len(samples)} test samples")
        return samples
    
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        # Return empty list if loading fails
        return []

def evaluate_summarizers(test_samples: List[Dict[str, Any]], lang: str = 'te') -> Dict[str, Any]:
    """
    Evaluate and compare summarizers
    
    Args:
        test_samples: List of test samples with text and reference summaries
        lang: Language code ('te' for Telugu, 'hi' for Hindi)
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        "samples": [],
        "timing": {
            "neural": 0,
            "improved": 0
        }
    }
    
    try:
        # Initialize both summarizers
        neural_summarizer = NeuralSummarizer(lang_code=lang)
        improved_summarizer = ImprovedSummarizer(lang_code=lang, attention_type='relative')
        
        logger.info("Starting evaluation of both summarizers")
        
        for i, sample in enumerate(test_samples):
            sample_result = {
                "original": sample.get('text', '')[:500],  # Limit to first 500 chars for display
                "reference": sample.get('summary', ''),
                "summaries": {}
            }
            
            # Generate summary with neural summarizer
            start_time = time.time()
            neural_summary = neural_summarizer.summarize(sample.get('text', ''))
            neural_time = time.time() - start_time
            results["timing"]["neural"] += neural_time
            
            # Generate summary with improved summarizer (relative position embeddings)
            start_time = time.time()
            improved_summary = improved_summarizer.summarize(sample.get('text', ''))
            improved_time = time.time() - start_time
            results["timing"]["improved"] += improved_time
            
            # Store summaries and timing
            sample_result["summaries"] = {
                "neural": {
                    "text": neural_summary,
                    "time": neural_time
                },
                "improved": {
                    "text": improved_summary,
                    "time": improved_time
                }
            }
            
            results["samples"].append(sample_result)
            logger.info(f"Processed sample {i+1}/{len(test_samples)}")
        
        # Calculate average timing
        if test_samples:
            results["timing"]["neural"] /= len(test_samples)
            results["timing"]["improved"] /= len(test_samples)
        
        return results
    
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return results

def display_results(results: Dict[str, Any]) -> None:
    """
    Display evaluation results in a readable format
    
    Args:
        results: Evaluation results from evaluate_summarizers
    """
    print("\n" + "="*80)
    print(" SUMMARIZATION EVALUATION RESULTS ")
    print("="*80)
    
    # Print timing information
    print(f"\nAverage processing time:")
    print(f"  Neural Summarizer:   {results['timing']['neural']:.4f} seconds")
    print(f"  Improved Summarizer: {results['timing']['improved']:.4f} seconds")
    print(f"  Difference:          {results['timing']['improved'] - results['timing']['neural']:.4f} seconds")
    
    # Print sample results
    for i, sample in enumerate(results['samples']):
        print(f"\n\n{'-'*80}")
        print(f"SAMPLE {i+1}:")
        print(f"{'-'*80}")
        
        print(f"\nOriginal text (truncated):")
        print(f"{sample['original']}...")
        
        print(f"\nReference summary:")
        print(f"{sample['reference']}")
        
        print(f"\nNeural Summarizer ({sample['summaries']['neural']['time']:.4f}s):")
        print(f"{sample['summaries']['neural']['text']}")
        
        print(f"\nImproved Summarizer with Relative Position Embeddings ({sample['summaries']['improved']['time']:.4f}s):")
        print(f"{sample['summaries']['improved']['text']}")

def main() -> None:
    """Main function to run the evaluation"""
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate summarizers with relative position embeddings")
        parser.add_argument("--lang", type=str, default="te", choices=["te", "hi"], 
                            help="Language code ('te' for Telugu, 'hi' for Hindi)")
        parser.add_argument("--samples", type=int, default=3, 
                            help="Number of samples to evaluate")
        parser.add_argument("--data", type=str, default=None, 
                            help="Path to test data file (jsonl format)")
        args = parser.parse_args()
        
        # Load test data
        test_samples = load_test_data(file_path=args.data, lang=args.lang, sample_size=args.samples)
        
        if not test_samples:
            print("No test samples found. Please check the data path and language filter.")
            return
        
        # Evaluate summarizers
        results = evaluate_summarizers(test_samples, lang=args.lang)
        
        # Display results
        display_results(results)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()