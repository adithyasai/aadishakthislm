import os
from pathlib import Path
import logging
import sys
from typing import Optional, Dict, Any, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.simple_summary import generate_grammatical_summary
from src.enhanced_summarizer import generate_enhanced_summary
from src.improved_summarizer import generate_improved_summary
from src.neural_summarizer import generate_neural_summary
from src.domain_classifier import classify_domain

class IndicSummarizer:
    """
    Unified interface for Indian language summarization
    Provides access to different summarization methods
    """
    
    METHODS = {
        "basic": "Basic extractive summarization",
        "enhanced": "Enhanced summarization with connectors",
        "improved": "Improved extractive-abstractive hybrid", 
        "neural": "Neural-guided summarization (recommended)"
    }
    
    def __init__(self, method: str = "neural", lang_code: str = "te"):
        """
        Initialize the summarizer
        
        Args:
            method: Summarization method to use (basic, enhanced, improved, neural)
            lang_code: Language code ('te' for Telugu, 'hi' for Hindi)
        """
        self.method = method.lower()
        self.lang_code = lang_code.lower()
        
        if self.method not in self.METHODS:
            logger.warning(f"Unknown method '{method}', falling back to 'neural'")
            self.method = "neural"
            
        if self.lang_code not in ["te", "hi"]:
            logger.warning(f"Unsupported language code '{lang_code}', using 'te' (Telugu)")
            self.lang_code = "te"
            
        logger.info(f"Initialized {self.METHODS.get(self.method)} for {self.get_language_name()}")
    
    def get_language_name(self) -> str:
        """Get the full language name from the language code"""
        languages = {"te": "Telugu", "hi": "Hindi"}
        return languages.get(self.lang_code, "Unknown")
    
    def summarize(self, text: str, **kwargs) -> str:
        """
        Generate a summary of the input text
        
        Args:
            text: Text to summarize
            **kwargs: Additional parameters for the summarization method
                     - compression_ratio: Target ratio of summary to original text (default: 0.5)
                     - max_sentences: Maximum number of sentences in the summary (default: 5)
        
        Returns:
            A coherent summary of the text
        """
        if not text or len(text.strip()) == 0:
            return "Empty text provided."
        
        # Detect domain and adjust parameters if not provided
        domain = classify_domain(text)
        logger.info(f"Detected domain: {domain}")
        # Default parameter adjustments per domain
        domain_params = {
            'legal': {'compression_ratio': 0.3, 'max_sentences': 3},
            'medical': {'compression_ratio': 0.4, 'max_sentences': 4},
            'technical': {'compression_ratio': 0.6, 'max_sentences': 5}
        }
        for param, value in domain_params.get(domain, {}).items():
            kwargs.setdefault(param, value)
        try:
            if self.method == "basic":
                return generate_grammatical_summary(text, self.lang_code, **kwargs)
            elif self.method == "enhanced":
                return generate_enhanced_summary(text, self.lang_code, **kwargs)
            elif self.method == "improved":
                return generate_improved_summary(text, self.lang_code, **kwargs)
            elif self.method == "neural":
                return generate_neural_summary(text, self.lang_code, **kwargs)
            else:
                # Fallback to neural if method is not recognized
                return generate_neural_summary(text, self.lang_code, **kwargs)
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return f"Summarization failed: {str(e)}"
    
    @staticmethod
    def list_available_methods() -> Dict[str, str]:
        """List all available summarization methods with descriptions"""
        return IndicSummarizer.METHODS
    
    @staticmethod
    def create(method: str = "neural", lang_code: str = "te") -> 'IndicSummarizer':
        """Factory method to create a summarizer instance"""
        return IndicSummarizer(method, lang_code)


# Simple usage example
if __name__ == "__main__":
    # Example text in Telugu
    te_text = """నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత భారతదేశం అనేక రంగాల్లో మానవ అభివృద్ధి సూచికలతోపాటు ఆర్థిక, సాంకేతిక, భౌతిక మౌలిక సదుపాయాలు, విదేశాంగ విధానం మరియు జాతీయ భద్రత వంటి కీలక విభాగాల్లో అమూల్యమైన మార్పులకు దారి తీస్తోంది. "ఆత్మనిర్భర్ భారత్" అనే భావనతో దేశీయ పరిశ్రమలను ప్రోత్సహిస్తూ, భారత్‌ను స్వయం సమృద్ధి దిశగా నడిపిస్తున్నారు. మేక్ ఇన్ ఇండియా, డిజిటల్ ఇండియా, స్టార్ట్‌అప్ ఇండియా, స్కిల్ ఇండియా వంటి కార్యక్రమాల ద్వారా యువతలో సృజనాత్మకతను వెలికి తీస్తూ, ప్రపంచంలో భారత యువశక్తి ప్రతిభను చాటుతున్నాడు. రహదారులు, రైళ్లు, ఎయిర్‌పోర్టులు, మెట్రో ప్రాజెక్టులు, బులెట్ ట్రైన్ వంటి ఆధునిక మౌలిక సదుపాయాలపై దృష్టి పెట్టి, గ్రామీణ ప్రాంతాలకు సరైన కనెక్టివిటీని తీసుకువచ్చేలా చర్యలు తీసుకుంటున్నారు."""
    
    # Create an instance of the summarizer (default: neural method, Telugu)
    summarizer = IndicSummarizer.create()
    
    # Generate summary
    summary = summarizer.summarize(te_text)
    
    # Print results
    print(f"Original text ({len(te_text.split())} words):")
    print("-" * 40)
    print(te_text)
    print("\nGenerated summary ({} words):".format(len(summary.split())))
    print("-" * 40)
    print(summary)
    
    # List all available methods
    print("\nAvailable summarization methods:")
    for method, description in IndicSummarizer.list_available_methods().items():
        print(f"- {method}: {description}")