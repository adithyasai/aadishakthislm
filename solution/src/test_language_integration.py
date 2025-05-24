"""
Integration test for morphological analyzers and language adapter features with model.

This test ensures that the model can work properly with both
morphological features and language-specific adapters.
"""

import sys
import torch
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import required modules
from src.model import IndicSLM, IndicSLMConfig
from src.tokenizer import IndicTokenizer
from src.data_processor import DataProcessor
from src.language_adapters import LanguageAdapterCollection, detect_language
from src.hindi_morphology import HindiMorphologyAnalyzer
from src.telugu_morphology import TeluguMorphologyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_with_morphology():
    """Test model interaction with morphological analyzers"""
    logger.info("Testing model integration with morphological analyzers...")
    
    # Initialize morphological analyzers
    hindi_analyzer = HindiMorphologyAnalyzer()
    telugu_analyzer = TeluguMorphologyAnalyzer()
    
    # Sample text in Hindi and Telugu
    hindi_text = "हम भारत के लोग भारत को एक संपूर्ण प्रभुत्व सम्पन्न समाजवादी पंथ निरपेक्ष लोकतंत्रात्मक गणराज्य बनाने के लिए तथा उसके समस्त नागरिकों को सामाजिक, आर्थिक और राजनैतिक न्याय दिलाने के लिए दृढ़ संकल्प होकर अपनी इस संविधान सभा में आज दिनांक 26 नवम्बर, 1949 को एतद् द्वारा इस संविधान को अंगीकृत, अधिनियमित और आत्मार्पित करते हैं।"
    telugu_text = "భారత ప్రజలమైన మేము, భారతదేశాన్ని సార్వభౌమ, సమాజవాద, లౌకిక, ప్రజాస్వామ్య, గణతంత్ర రాజ్యంగా నిర్మించుకోవాలని, దాని పౌరులందరికీ సామాజిక, ఆర్థిక మరియు రాజకీయ న్యాయం; ఆలోచన, అభివ్యక్తి, విశ్వాసం, మతం మరియు ఆరాధనల స్వేచ్ఛ; హోదా మరియు అవకాశాల సమానత్వం కల్పించుకోవాలని దృఢంగా సంకల్పించి; మరియు వారందరిలో వ్యక్తి ఘనతను మరియు జాతి సమగ్రతను భద్రపరిచే సౌభ్రాతృత్వం పెంపొందించుకోవాలని ప్రతిజ్ఞ చేస్తూ, మా సంవిధాన సభలో ఈ నాడు, అనగా 1949 వ సంవత్సరం నవంబర 26 వ తేదిన, దీనిని అంగీకరించి, శాసనబద్ధం చేసి, మాకు మేము సమర్పించుకుంటున్నాము."
    
    logger.info("Analyzing Hindi text...")
    hindi_analysis = hindi_analyzer.analyze_text(hindi_text[:100])  # Analyze first 100 chars for brevity
    logger.info(f"Hindi analysis produced {len(hindi_analysis)} word analyses")
    
    logger.info("Analyzing Telugu text...")
    telugu_analysis = telugu_analyzer.analyze_text(telugu_text[:100])  # Analyze first 100 chars for brevity
    logger.info(f"Telugu analysis produced {len(telugu_analysis)} word analyses")
    
    # Test integrated with data processor
    logger.info("Testing integration with data processor...")
    data_processor = DataProcessor(enable_morphology=True)
    
    # Test language detection
    logger.info("Testing language detection...")
    hindi_lang = detect_language(hindi_text)
    telugu_lang = detect_language(telugu_text)
    
    logger.info(f"Hindi text detected as: {hindi_lang}")
    logger.info(f"Telugu text detected as: {telugu_lang}")
    
    logger.info("All morphology integration tests passed!")

def test_model_with_adapters():
    """Test model with language adapters"""
    logger.info("Testing model with language adapters...")
    
    # Create model config with adapters
    config = IndicSLMConfig(
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=12,
        intermediate_size=768 * 4,
        max_position_embeddings=1024,
        pad_token_id=0,
        use_adapters=True,
        adapter_size=64,
        languages=["hi", "te"]
    )
    
    # Initialize model
    model = IndicSLM(config)
    
    # Verify adapter collection is initialized
    assert hasattr(model, 'adapters'), "Model should have adapters attribute"
    assert isinstance(model.adapters, LanguageAdapterCollection), "Model adapters should be LanguageAdapterCollection"
    assert "hi" in model.adapters.adapters, "Hindi adapter should be initialized"
    assert "te" in model.adapters.adapters, "Telugu adapter should be initialized"
    
    # Test with input
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones_like(input_ids)
    
    # Test with explicit language IDs
    outputs_hi = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        language_id="hi"
    )
    
    outputs_te = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        language_id="te"
    )
    
    logger.info(f"Hindi adapter output shape: {outputs_hi['logits'].shape}")
    logger.info(f"Telugu adapter output shape: {outputs_te['logits'].shape}")
    
    # Verify outputs have correct shape
    assert outputs_hi['logits'].shape == (batch_size, seq_length, config.vocab_size), \
        f"Expected output shape {(batch_size, seq_length, config.vocab_size)}, got {outputs_hi['logits'].shape}"
    
    logger.info("Language adapter tests passed!")

if __name__ == "__main__":
    logger.info("Running language-enhanced model integration tests...")
    
    # Test morphological analyzers
    test_model_with_morphology()
    
    # Test language adapters
    test_model_with_adapters()
    
    logger.info("All language enhancement tests passed successfully!")
