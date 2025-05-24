#!/usr/bin/env python
"""
SLM Summarization Tool - Command-line interface for the SLM summarization system
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('summarize.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Add the project root to sys.path to enable imports"""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    if project_root not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SLM Summarization Tool - Generate summaries from text in multiple Indian languages",
        epilog="""
Examples:
  # Summarize text from a file using the neural method
  python summarize.py mytext.txt --method neural --language te
  
  # Summarize direct text input with auto language detection
  python summarize.py "నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత..." --method neural
  
  # Compare all summarization methods
  python summarize.py mytext.txt --compare
  
  # Output to a file in JSON format
  python summarize.py mytext.txt --method improved --output summary.json --json
"""
    )
    
    parser.add_argument(
        "input",
        help="Input text or file path to summarize"
    )
    
    parser.add_argument(
        "--method", "-m",
        choices=["simple", "advanced", "neural", "improved", "enhanced", "ensemble"],
        default="advanced",
        help="Summarization method to use (default: advanced)"
    )
    
    parser.add_argument(
        "--language", "-l",
        choices=["hi", "te", "auto"],
        default="auto",
        help="Language of the input text (hi=Hindi, te=Telugu, auto=Auto-detect) (default: auto)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file path (if not specified, prints to console)"
    )
    
    parser.add_argument(
        "--sentences", "-s",
        type=int,
        default=3,
        help="Number of sentences in the summary (for extractive methods) (default: 3)"
    )
    
    parser.add_argument(
        "--max-length", "-ml",
        type=int,
        default=150,
        help="Maximum token length for neural summaries (default: 150)"
    )
    
    parser.add_argument(
        "--compression", "-c",
        type=float,
        default=0.3,
        help="Compression ratio (summary length / original length) (default: 0.3)"
    )
    
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare all methods side by side"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def get_input_text(input_path_or_text):
    """Get input text from file or directly from argument"""
    # Check if input is a file path
    if os.path.isfile(input_path_or_text):
        try:
            with open(input_path_or_text, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)
    else:
        # Assume input is direct text
        return input_path_or_text


def detect_language(text):
    """Detect language of the input text"""
    # Simple heuristic for Hindi vs Telugu detection
    # Hindi uses Devanagari script, Telugu uses Telugu script
    
    # Count Devanagari characters (0900-097F)
    devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    
    # Count Telugu characters (0C00-0C7F)
    telugu_count = sum(1 for char in text if '\u0C00' <= char <= '\u0C7F')
    
    if devanagari_count > telugu_count:
        return 'hi'  # Hindi
    elif telugu_count > devanagari_count:
        return 'te'  # Telugu
    else:
        # Default to Telugu if can't determine
        return 'te'


def summarize_text(text, method, language, sentences, max_length, compression):
    """Generate summary based on specified method"""
    
    if language == 'auto':
        language = detect_language(text)
        logger.info(f"Detected language: {language}")
    
    try:
        # Import the appropriate summarization method
        if method == "simple":
            from src.simple_summary import generate_simple_summary
            summary = generate_simple_summary(text, language, sentences)
        
        elif method == "advanced":
            from src.summarizer import generate_summary
            summary = generate_summary(text, language, sentences)
        
        elif method == "neural":
            from src.neural_summarizer import generate_neural_summary
            summary = generate_neural_summary(text, language, max_length)
            
        elif method == "improved":
            from src.improved_summarizer import generate_improved_summary
            summary = generate_improved_summary(text, language, sentences)
            
        elif method == "enhanced":
            from src.enhanced_summarizer import generate_enhanced_summary
            summary = generate_enhanced_summary(text, language, max_length)
            
        elif method == "ensemble":
            # Use a weighted combination of methods
            from src.summarizer import generate_summary
            from src.neural_summarizer import generate_neural_summary
            from src.improved_summarizer import generate_improved_summary
            
            # Get summaries from different methods
            summary1 = generate_summary(text, language, sentences)
            summary2 = generate_neural_summary(text, language, max_length)
            summary3 = generate_improved_summary(text, language, sentences)
            
            # Combine summaries (simple approach - take the longest summary)
            summaries = [summary1, summary2, summary3]
            summary = max(summaries, key=len)
            
        else:
            logger.error(f"Unknown method: {method}")
            sys.exit(1)
            
        return summary
        
    except ImportError as e:
        logger.error(f"Error importing summarization module: {e}")
        logger.error("Make sure all requirements are installed")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    # Setup import paths
    setup_paths()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get input text
    text = get_input_text(args.input)
    
    if not text:
        logger.error("Input text is empty")
        sys.exit(1)
    
    if args.compare:
        # Compare all methods
        methods = ["simple", "advanced", "neural", "improved", "enhanced", "ensemble"]
        results = {}
        
        for method in methods:
            try:
                logger.info(f"Generating summary using {method} method...")
                summary = summarize_text(
                    text, method, args.language, 
                    args.sentences, args.max_length, args.compression
                )
                results[method] = summary
            except Exception as e:
                logger.error(f"Error with {method} method: {e}")
                results[method] = f"Error: {str(e)}"
        
        # Output results
        if args.json:
            output = json.dumps(results, ensure_ascii=False, indent=2)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output)
            else:
                print(output)
        else:
            print("\n" + "="*80)
            print(f"ORIGINAL TEXT ({len(text)} chars):")
            print("="*80)
            print(text)
            
            for method, summary in results.items():
                print("\n" + "-"*80)
                print(f"{method.upper()} SUMMARY ({len(summary)} chars):")
                print("-"*80)
                print(summary)
    else:
        # Single method
        logger.info(f"Generating summary using {args.method} method...")
        summary = summarize_text(
            text, args.method, args.language, 
            args.sentences, args.max_length, args.compression
        )
        
        # Output summary
        if args.json:
            output = json.dumps({
                "original": text,
                "summary": summary,
                "method": args.method,
                "language": args.language if args.language != "auto" else detect_language(text)
            }, ensure_ascii=False, indent=2)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(output)
            else:
                print(output)
        else:
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(summary)
            else:
                print("\n" + "="*80)
                print(f"ORIGINAL TEXT ({len(text)} chars):")
                print("="*80)
                print(text)
                print("\n" + "-"*80)
                print(f"SUMMARY ({len(summary)} chars):")
                print("-"*80)
                print(summary)
    
    logger.info("Summary generation completed")


if __name__ == "__main__":
    main()