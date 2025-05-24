import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.hindi_morphology import HindiMorphologyAnalyzer
from src.telugu_morphology import TeluguMorphologyAnalyzer
from src.improved_summarizer import generate_improved_summary
from src.neural_summarizer import generate_neural_summary
from src.metrics import IndicMetrics, evaluate_summaries, format_metrics_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hindi_morphology():
    """Test Hindi morphological analyzer"""
    logger.info("Testing Hindi morphological analyzer...")
    
    # Initialize analyzer
    hindi_analyzer = HindiMorphologyAnalyzer()
    
    # Test words to analyze
    test_words = [
        "किताब", 
        "किताबें",
        "जाता", 
        "जाती", 
        "जाएगा", 
        "खिलाड़ियों",
        "करने", 
        "भारतीय", 
        "प्रधानमंत्री",
        "चलते"
    ]
    
    for word in test_words:
        analysis = hindi_analyzer.analyze_word(word)
        logger.info(f"Word: {word}")
        logger.info(f"  Stem: {analysis['stem']}")
        logger.info(f"  Segments: {analysis['segments']}")
        
        # Print additional features if available
        for key, value in analysis.items():
            if key not in ['word', 'stem', 'segments']:
                logger.info(f"  {key}: {value}")
        
        logger.info("-" * 40)
    
    # Test text analysis
    test_text = "भारत की राजधानी नई दिल्ली है। यहां कई ऐतिहासिक स्थान हैं।"
    
    logger.info(f"\nAnalyzing text: {test_text}")
    analyses = hindi_analyzer.analyze_text(test_text)
    
    logger.info("\nExtracted words and their stems:")
    for analysis in analyses:
        logger.info(f"  {analysis['word']} -> {analysis['stem']}")
    
    logger.info("\nRoot forms:")
    root_forms = hindi_analyzer.get_root_forms(test_text)
    logger.info(f"  {root_forms}")
    
    return True

def test_telugu_morphology():
    """Test Telugu morphological analyzer"""
    logger.info("Testing Telugu morphological analyzer...")
    
    # Initialize analyzer
    telugu_analyzer = TeluguMorphologyAnalyzer()
    
    # Test words to analyze
    test_words = [
        "పుస్తకము",  # book
        "పిల్లలు",    # children
        "చేస్తున్నాడు", # he is doing
        "వెళ్తాను",   # I will go
        "అమ్మాయిని",  # girl (accusative)
        "అతడు",      # he
        "వారికి",     # to them
        "తెలుగులో",   # in Telugu
        "చదువుతున్నాను", # I am reading
        "భారతదేశము"  # India
    ]
    
    for word in test_words:
        analysis = telugu_analyzer.analyze_word(word)
        logger.info(f"Word: {word}")
        logger.info(f"  Stem: {analysis['stem']}")
        logger.info(f"  Segments: {analysis['segments']}")
        
        # Print additional features if available
        for key, value in analysis.items():
            if key not in ['word', 'stem', 'segments']:
                logger.info(f"  {key}: {value}")
        
        logger.info("-" * 40)
    
    # Test text analysis
    test_text = "తెలుగు భాష చాలా అందమైనది. భారతదేశంలో చాలా మంది తెలుగు మాట్లాడతారు."
    
    logger.info(f"\nAnalyzing text: {test_text}")
    analyses = telugu_analyzer.analyze_text(test_text)
    
    logger.info("\nExtracted words and their stems:")
    for analysis in analyses:
        logger.info(f"  {analysis['word']} -> {analysis['stem']}")
    
    logger.info("\nRoot forms:")
    root_forms = telugu_analyzer.get_root_forms(test_text)
    logger.info(f"  {root_forms}")
    
    # Test sandhi analysis
    word1 = "రాము"  # Ramu
    word2 = "అడిగాడు"  # asked
    combined = telugu_analyzer.apply_sandhi_rules(word1, word2)
    logger.info(f"\nSandhi test: {word1} + {word2} -> {combined}")
    
    return True

def test_evaluation_metrics():
    """Test evaluation metrics on sample summaries"""
    logger.info("Testing evaluation metrics...")
    
    # Sample texts and summaries for testing
    test_data = [
        # Telugu examples
        {
            'language': 'Telugu',
            'code': 'te', 
            'text': """ప్రస్తుతం భారత్‌-పాకిస్తాన్ మధ్య పరిస్థితి మరింత ఉద్రిక్తంగా మారుతున్నది. కశ్మీర్ సరిహద్దుల్లో తరచూ కాల్పులు జరగడం, ఉగ్రవాద చర్యలు కొనసాగడం ఈ ఉద్రిక్తతకు ప్రధాన కారణాలు. ఇటీవలి కాలంలో పాక్ మద్దతు ఉన్న ఉగ్రవాదుల చొరబాటు ప్రయత్నాలు పెరిగినట్లు భారత సైన్యం పేర్కొంది. అదే సమయంలో, రాజకీయ నేతల మధ్య మాటల యుద్ధం కూడా తీవ్రంగా సాగుతోంది. ఇరు దేశాల ప్రజలు శాంతిని కోరుతున్నా, సరిహద్దుల్లో పరిస్థితి ఇంకా గందరగోళంగా ఉంది. ఈ పరిస్థితిని చర్చల ద్వారా పరిష్కరించాలనే సూచనలు అంతర్జాతీయంగా వెల్లువెత్తుతున్నాయి.""",
            'reference': """భారత్‌-పాకిస్తాన్ మధ్య ఉద్రిక్తతలు పెరిగుతున్నాయి. కశ్మీర్ సరిహద్దుల్లో కాల్పులు, ఉగ్రవాద చర్యలు ప్రధాన కారణాలు కాగా, పాక్ మద్దతు ఉన్న చొరబాట్లు కూడా పెరుగుతున్నాయి. రాజకీయ స్థాయిలో మాటల యుద్ధం జరుగుతోంది. ప్రజలు శాంతిని కోరుతున్నా, పరిస్థితి ఇంకా అస్తవ్యస్థంగా ఉంది. సమస్యను చర్చల ద్వారా పరిష్కరించాలని అంతర్జాతీయ సమాజం సూచిస్తోంది."""
        },
        # Hindi examples
        {
            'language': 'Hindi',
            'code': 'hi',
            'text': """भारत और पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर लगातार गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना के अनुसार, हाल के दिनों में पाकिस्तान समर्थित आतंकवादियों की घुसपैठ की कोशिशों में वृद्धि हुई है। इसी समय, राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। दोनों देशों के लोग शांति चाहते हैं, लेकिन सीमा पर स्थिति अभी भी अस्थिर है। इस स्थिति को वार्ता के माध्यम से सुलझाने के सुझाव अंतरराष्ट्रीय स्तर पर दिए जा रहे हैं।""",
            'reference': """भारत-पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना ने पाकिस्तान समर्थित आतंकवादियों की घुसपैठ में वृद्धि की बात कही है। राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। लोग शांति चाहते हैं, पर स्थिति अस्थिर है। अंतरराष्ट्रीय स्तर पर वार्ता द्वारा समाधान के सुझाव दिए जा रहे हैं।"""
        }
    ]
    
    logger.info("==== SUMMARY EVALUATION TEST ====")
    
    # Test BLEU, ROUGE, and BERTScore for each test case
    for i, test_case in enumerate(test_data):
        lang_code = test_case['code']
        original_text = test_case['text']
        reference = test_case['reference']
        language = test_case['language']
        
        logger.info(f"\n\nTesting {language} evaluation (Test #{i+1}):")
        logger.info("-" * 40)
        
        logger.info(f"Original text ({len(original_text.split())} words):")
        logger.info(original_text[:100] + "...")  # Show beginning of text
        
        logger.info(f"\nReference summary ({len(reference.split())} words):")
        logger.info(reference)
        
        # Generate summaries using both methods
        improved_summary = generate_improved_summary(original_text, lang_code)
        neural_summary = generate_neural_summary(original_text, lang_code)
        
        logger.info(f"\nImproved summary ({len(improved_summary.split())} words):")
        logger.info(improved_summary)
        
        logger.info(f"\nNeural summary ({len(neural_summary.split())} words):")
        logger.info(neural_summary)
        
        # Evaluate using metrics
        metrics = IndicMetrics(lang_code=lang_code)
        
        logger.info("\nEvaluating Improved Summary:")
        improved_results = metrics.evaluate_all_metrics([reference], [improved_summary])
        logger.info(format_metrics_report(improved_results))
        
        logger.info("\nEvaluating Neural Summary:")
        neural_results = metrics.evaluate_all_metrics([reference], [neural_summary])
        logger.info(format_metrics_report(neural_results))
        
        # Compare the two approaches
        logger.info("\nComparison:")
        if improved_results["average"].get("bertscore_f1", 0) > neural_results["average"].get("bertscore_f1", 0):
            logger.info("Improved summarizer produces more semantically similar summaries according to BERTScore")
        else:
            logger.info("Neural summarizer produces more semantically similar summaries according to BERTScore")
        
        if improved_results["average"].get("rouge1_f", 0) > neural_results["average"].get("rouge1_f", 0):
            logger.info("Improved summarizer has better lexical overlap according to ROUGE-1")
        else:
            logger.info("Neural summarizer has better lexical overlap according to ROUGE-1")
    
    return True

def evaluate_all_summaries():
    """Run comprehensive evaluation on all summarization approaches"""
    logger.info("Running comprehensive evaluation of all summarization approaches...")
    
    # Define test cases across languages
    test_cases = [
        # Telugu
        {
            'language': 'Telugu',
            'code': 'te',
            'text': """ప్రస్తుతం భారత్‌-పాకిస్తాన్ మధ్య పరిస్థితి మరింత ఉద్రిక్తంగా మారుతున్నది. కశ్మీర్ సరిహద్దుల్లో తరచూ కాల్పులు జరగడం, ఉగ్రవాద చర్యలు కొనసాగడం ఈ ఉద్రిక్తతకు ప్రధాన కారణాలు. ఇటీవలి కాలంలో పాక్ మద్దతు ఉన్న ఉగ్రవాదుల చొరబాటు ప్రయత్నాలు పెరిగినట్లు భారత సైన్యం పేర్కొంది. అదే సమయంలో, రాజకీయ నేతల మధ్య మాటల యుద్ధం కూడా తీవ్రంగా సాగుతోంది. ఇరు దేశాల ప్రజలు శాంతిని కోరుతున్నా, సరిహద్దుల్లో పరిస్థితి ఇంకా గందరగోళంగా ఉంది. ఈ పరిస్థితిని చర్చల ద్వారా పరిష్కరించాలనే సూచనలు అంతర్జాతీయంగా వెల్లువెత్తుతున్నాయి.""",
            'reference': """భారత్‌-పాకిస్తాన్ మధ్య ఉద్రిక్తతలు పెరిగుతున్నాయి. కశ్మీర్ సరిహద్దుల్లో కాల్పులు, ఉగ్రవాద చర్యలు ప్రధాన కారణాలు కాగా, పాక్ మద్దతు ఉన్న చొరబాట్లు కూడా పెరుగుతున్నాయి. రాజకీయ స్థాయిలో మాటల యుద్ధం జరుగుతోంది. ప్రజలు శాంతిని కోరుతున్నా, పరిస్థితి ఇంకా అస్తవ్యస్థంగా ఉంది. సమస్యను చర్చల ద్వారా పరిష్కరించాలని అంతర్జాతీయ సమాజం సూచిస్తోంది."""
        },
        {
            'language': 'Telugu (Modi Initiatives)',
            'code': 'te',
            'text': """నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత భారతదేశం అనేక రంగాల్లో మానవ అభివృద్ధి సూచికలతోపాటు ఆర్థిక, సాంకేతిక, భౌతిక మౌలిక సదుపాయాలు, విదేశాంగ విధానం మరియు జాతీయ భద్రత వంటి కీలక విభాగాల్లో అమూల్యమైన మార్పులకు దారి తీస్తోంది. "ఆత్మనిర్భర్ భారత్" అనే భావనతో దేశీయ పరిశ్రమలను ప్రోత్సహిస్తూ, భారత్‌ను స్వయం సమృద్ధి దిశగా నడిపిస్తున్నారు. మేక్ ఇన్ ఇండియా, డిజిటల్ ఇండియా, స్టార్ట్‌అప్ ఇండియా, స్కిల్ ఇండియా వంటి కార్యక్రమాల ద్వారా యువతలో సృజనాత్మకతను వెలికి తీస్తూ, ప్రపంచంలో భారత యువశక్తి ప్రతిభను చాటుతున్నాడు. రహదారులు, రైళ్లు, ఎయిర్‌పోర్టులు, మెట్రో ప్రాజెక్టులు, బులెట్ ట్రైన్ వంటి ఆధునిక మౌలిక సదుపాయాలపై దృష్టి పెట్టి, గ్రామీణ ప్రాంతాలకు సరైన కనెక్టివిటీని తీసుకువచ్చేలా చర్యలు తీసుకుంటున్నారు.""",
            'reference': """నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత భారతదేశం అనేక రంగాల్లో అభివృద్ధి సాధిస్తోంది. "ఆత్మనిర్భర్ భారత్" భావనతో దేశీయ పరిశ్రమలను ప్రోత్సహిస్తూ, స్వయం సమృద్ధి దిశగా దేశాన్ని నడిపిస్తున్నారు. మేక్ ఇన్ ఇండియా, డిజిటల్ ఇండియా వంటి కార్యక్రమాల ద్వారా యువతలో సృజనాత్మకతను పెంపొందిస్తున్నారు. అలాగే, రహదారులు, రైళ్లు, మెట్రో ప్రాజెక్టులు వంటి ఆధునిక మౌలిక సదుపాయాలపై దృష్టి పెట్టి, గ్రామీణ ప్రాంతాలకు మెరుగైన కనెక్టివిటీని అందిస్తున్నారు."""
        },
        # Hindi
        {
            'language': 'Hindi (Kashmir Conflict)',
            'code': 'hi',
            'text': """भारत और पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर लगातार गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना के अनुसार, हाल के दिनों में पाकिस्तान समर्थित आतंकवादियों की घुसपैठ की कोशिशों में वृद्धि हुई है। इसी समय, राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। दोनों देशों के लोग शांति चाहते हैं, लेकिन सीमा पर स्थिति अभी भी अस्थिर है। इस स्थिति को वार्ता के माध्यम से सुलझाने के सुझाव अंतरराष्ट्रीय स्तर पर दिए जा रहे हैं।""",
            'reference': """भारत-पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना ने पाकिस्तान समर्थित आतंकवादियों की घुसपैठ में वृद्धि की बात कही है। राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। लोग शांति चाहते हैं, पर स्थिति अस्थिर है। अंतरराष्ट्रीय स्तर पर वार्ता द्वारा समाधान के सुझाव दिए जा रहे हैं।"""
        },
        {
            'language': 'Hindi (Language Info)',
            'code': 'hi',
            'text': """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत गणराज्य की आधिकारिक भाषाओं में से एक है। हिंदी की लिपि देवनागरी है। वर्तमान हिंदी का विकास खड़ी बोली से हुआ है, जो दिल्ली के आसपास के क्षेत्रों में बोली जाती थी। हिंदी शब्द फारसी मूल का है, जिसका अर्थ है 'सिंधु नदी का'। मुगल काल में सिंधु नदी के पूर्व में रहने वाले लोगों और उनकी भाषा को 'हिंदी' कहा जाता था। हिंदी भाषा के विकास को भाषावैज्ञानिक आधार पर चार कालों में बांटा गया है: आदिकाल (12वीं से 14वीं शताब्दी), भक्तिकाल (14वीं से 16वीं शताब्दी), रीतिकाल (17वीं से 19वीं शताब्दी) और आधुनिक काल (19वीं शताब्दी से अब तक)।""",
            'reference': """हिंदी भारत की सबसे अधिक और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत की आधिकारिक भाषाओं में से एक है, जिसकी लिपि देवनागरी है। इसका विकास दिल्ली क्षेत्र की खड़ी बोली से हुआ है। 'हिंदी' शब्द फारसी मूल का है, जिसका अर्थ है 'सिंधु नदी का'। हिंदी भाषा के विकास को चार कालों में बाँटा गया है: आदिकाल, भक्तिकाल, रीतिकाल और आधुनिक काल।"""
        }
    ]
    
    logger.info("===== COMPREHENSIVE SUMMARIZATION EVALUATION =====")
    
    # Evaluate each test case with both summarization approaches
    results = {
        'improved': {
            'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []
        },
        'neural': {
            'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'bertscore': []
        }
    }
    
    for i, test_case in enumerate(test_cases):
        lang_code = test_case['code']
        text = test_case['text']
        reference = test_case['reference']
        language = test_case['language']
        
        logger.info(f"\n\nEvaluating {language} text (Test #{i+1}):")
        logger.info("-" * 60)
        
        # Generate summaries
        improved_summary = generate_improved_summary(text, lang_code)
        neural_summary = generate_neural_summary(text, lang_code)
        
        logger.info(f"Original text length: {len(text.split())} words")
        logger.info(f"Reference summary: {len(reference.split())} words")
        logger.info(f"Improved summary: {len(improved_summary.split())} words")
        logger.info(f"Neural summary: {len(neural_summary.split())} words")
        
        # Calculate metrics for Improved summary
        metrics = IndicMetrics(lang_code=lang_code)
        improved_eval = metrics.evaluate_all_metrics([reference], [improved_summary])
        
        # Calculate metrics for Neural summary
        neural_eval = metrics.evaluate_all_metrics([reference], [neural_summary])
        
        # Record results
        results['improved']['bleu'].append(improved_eval["average"].get("bleu", 0))
        results['improved']['rouge1'].append(improved_eval["average"].get("rouge1_f", 0))
        results['improved']['rouge2'].append(improved_eval["average"].get("rouge2_f", 0))
        results['improved']['rougeL'].append(improved_eval["average"].get("rougeL_f", 0))
        results['improved']['bertscore'].append(improved_eval["average"].get("bertscore_f1", 0))
        
        results['neural']['bleu'].append(neural_eval["average"].get("bleu", 0))
        results['neural']['rouge1'].append(neural_eval["average"].get("rouge1_f", 0))
        results['neural']['rouge2'].append(neural_eval["average"].get("rouge2_f", 0))
        results['neural']['rougeL'].append(neural_eval["average"].get("rougeL_f", 0))
        results['neural']['bertscore'].append(neural_eval["average"].get("bertscore_f1", 0))
        
        # Display individual results
        logger.info(f"\nMetrics for '{language}':")
        logger.info(f"{'Metric':<12} {'Improved':<10} {'Neural':<10} {'Difference':<10}")
        logger.info("-" * 42)
        logger.info(f"{'BLEU':<12} {improved_eval['average'].get('bleu', 0):.4f}      {neural_eval['average'].get('bleu', 0):.4f}      {improved_eval['average'].get('bleu', 0) - neural_eval['average'].get('bleu', 0):.4f}")
        logger.info(f"{'ROUGE-1':<12} {improved_eval['average'].get('rouge1_f', 0):.4f}      {neural_eval['average'].get('rouge1_f', 0):.4f}      {improved_eval['average'].get('rouge1_f', 0) - neural_eval['average'].get('rouge1_f', 0):.4f}")
        logger.info(f"{'ROUGE-2':<12} {improved_eval['average'].get('rouge2_f', 0):.4f}      {neural_eval['average'].get('rouge2_f', 0):.4f}      {improved_eval['average'].get('rouge2_f', 0) - neural_eval['average'].get('rouge2_f', 0):.4f}")
        logger.info(f"{'ROUGE-L':<12} {improved_eval['average'].get('rougeL_f', 0):.4f}      {neural_eval['average'].get('rougeL_f', 0):.4f}      {improved_eval['average'].get('rougeL_f', 0) - neural_eval['average'].get('rougeL_f', 0):.4f}")
        logger.info(f"{'BERTScore':<12} {improved_eval['average'].get('bertscore_f1', 0):.4f}      {neural_eval['average'].get('bertscore_f1', 0):.4f}      {improved_eval['average'].get('bertscore_f1', 0) - neural_eval['average'].get('bertscore_f1', 0):.4f}")
    
    # Calculate and display average metrics
    logger.info("\n\n===== OVERALL EVALUATION RESULTS =====")
    logger.info(f"Number of test cases: {len(test_cases)}")
    
    # Calculate averages
    for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore']:
        improved_avg = sum(results['improved'][metric]) / len(results['improved'][metric]) if results['improved'][metric] else 0
        neural_avg = sum(results['neural'][metric]) / len(results['neural'][metric]) if results['neural'][metric] else 0
        diff = improved_avg - neural_avg
        
        logger.info(f"{metric.upper():<12} Improved: {improved_avg:.4f}  Neural: {neural_avg:.4f}  Diff: {diff:.4f}")
    
    # Overall assessment
    improved_wins = sum(1 for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore'] 
                      if sum(results['improved'][metric]) > sum(results['neural'][metric]))
    neural_wins = sum(1 for metric in ['bleu', 'rouge1', 'rouge2', 'rougeL', 'bertscore']
                    if sum(results['neural'][metric]) > sum(results['improved'][metric]))
    
    logger.info("\nOVERALL ASSESSMENT:")
    if improved_wins > neural_wins:
        logger.info(f"The Improved Summarizer performs better on {improved_wins} out of 5 metrics")
    elif neural_wins > improved_wins:
        logger.info(f"The Neural Summarizer performs better on {neural_wins} out of 5 metrics")
    else:
        logger.info("Both summarizers perform similarly across all metrics")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test summarization and evaluation")
    parser.add_argument("--hindi", action="store_true", help="Test Hindi morphological analyzer")
    parser.add_argument("--telugu", action="store_true", help="Test Telugu morphological analyzer")
    parser.add_argument("--test_all", action="store_true", help="Test both analyzers")
    parser.add_argument("--test_morphology", action="store_true", help="Alias for test_all")
    parser.add_argument("--test_metrics", action="store_true", help="Test evaluation metrics")
    parser.add_argument("--evaluate_all", action="store_true", help="Run comprehensive evaluation of all summarization approaches")
    
    args = parser.parse_args()
    
    # If no specific test selected, run morphology tests
    run_all = args.test_all or args.test_morphology or (not any([args.hindi, args.telugu, args.test_metrics, args.evaluate_all]))
    
    if args.hindi or run_all:
        test_hindi_morphology()
    
    if args.telugu or run_all:
        test_telugu_morphology()
        
    if args.test_metrics:
        test_evaluation_metrics()
        
    if args.evaluate_all:
        evaluate_all_summaries()

if __name__ == "__main__":
    main()