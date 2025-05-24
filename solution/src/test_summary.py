import argparse
import os
from pathlib import Path
import logging
import torch
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.train import generate_summary, CONFIG_PATH, OUTPUT_DIR

def main():
    parser = argparse.ArgumentParser(description='Generate summary for Telugu or Hindi text')
    parser.add_argument('--text', type=str, help='Input text to summarize')
    parser.add_argument('--text_file', type=str, help='Text file containing content to summarize')
    parser.add_argument('--lang', type=str, choices=['te', 'hi'], required=True, help='Language code (te for Telugu, hi for Hindi)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the trained model checkpoint')
    parser.add_argument('--output_file', type=str, help='Path to save the summary')
    
    args = parser.parse_args()
    
    # Get text from file if provided
    if args.text_file and not args.text:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                args.text = f.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return

    # If no text is provided, use a sample text
    if not args.text:
        if args.lang == 'te':
            args.text = """తెలుగు భాష దక్షిణ భారతదేశంలో ప్రధానంగా ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడే ద్రావిడ భాష. ఇది భారతదేశంలో అత్యధికంగా మాట్లాడే భాషలలో నాలుగవది మరియు ద్రావిడ భాషా కుటుంబంలో రెండవ అతిపెద్ద భాష. 2011 జనాభా లెక్కల ప్రకారం, భారతదేశంలో 81.1 మిలియన్లకు పైగా ప్రజలు తెలుగును మాతృభాషగా మాట్లాడతారు, సింగపూర్‌లో అధికారిక భాషగా గుర్తింపు పొందింది. తెలుగు భాష చరిత్ర క్రీ.పూ. 400 నాటిది, ఇది భారతదేశ ప్రాచీన భాషలలో ఒకటి. 10వ శతాబ్దం నాటి నన్నయ్య రచించిన ఆంధ్రమహాభారతాన్ని తెలుగు తొలి కావ్యంగా భావిస్తారు."""
        else:
            args.text = """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत गणराज्य की आधिकारिक भाषाओं में से एक है। हिंदी की लिपि देवनागरी है। वर्तमान हिंदी का विकास खड़ी बोली से हुआ है, जो दिल्ली के आसपास के क्षेत्रों में बोली जाती थी। हिंदी शब्द फारसी मूल का है, जिसका अर्थ है 'सिंधु नदी का'। मुगल काल में सिंधु नदी के पूर्व में रहने वाले लोगों और उनकी भाषा को 'हिंदी' कहा जाता था।"""

    # Find latest checkpoint if model path is not provided
    if args.model_path is None:
        # First check for .ckpt files
        checkpoints = list(Path(OUTPUT_DIR).glob("*.ckpt"))
        
        # If no .ckpt files found, try using the final_model directory
        if not checkpoints:
            final_model_path = Path(OUTPUT_DIR) / "final_model"
            if os.path.exists(final_model_path):
                args.model_path = str(final_model_path)
                logger.info(f"Using final model: {args.model_path}")
            else:
                logger.error("No model checkpoints found and no model path specified")
                return
        else:
            # Sort by modification time to get the latest
            args.model_path = str(sorted(checkpoints, key=os.path.getmtime, reverse=True)[0])
            logger.info(f"Using latest checkpoint: {args.model_path}")

    # Check if model path exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    print(f"\nGenerating summary for {args.lang} text...")
    print("\nOriginal Text:")
    print("-" * 80)
    print(args.text[:500] + "..." if len(args.text) > 500 else args.text)
    
    # Generate summary with error handling
    try:
        # Set memory-efficient device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Use memory-efficient settings
        torch.cuda.empty_cache()
        
        # Generate the summary
        summary = generate_summary(args.text, args.lang, args.model_path)
        
        print("\nGenerated Summary:")
        print("-" * 80)
        print(summary)
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Original Text:\n{args.text}\n\nGenerated Summary:\n{summary}")
            print(f"\nSummary saved to {args.output_file}")
            
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        print("\nError generating summary. Please check logs for details.")

if __name__ == "__main__":
    main()