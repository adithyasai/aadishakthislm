import os
import sys
from pathlib import Path

# Add parent directory to Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import our summarization approaches
from src.improved_summarizer import generate_improved_summary
from src.neural_summarizer import generate_neural_summary

def compare_summarization_approaches():
    """Compare different summarization approaches on the same text samples"""
    
    # Example texts from the user's query
    te_text = """ప్రస్తుతం భారత్‌-పాకిస్తాన్ మధ్య పరిస్థితి మరింత ఉద్రిక్తంగా మారుతున్నది. కశ్మీర్ సరిహద్దుల్లో తరచూ కాల్పులు జరగడం, ఉగ్రవాద చర్యలు కొనసాగడం ఈ ఉద్రిక్తతకు ప్రధాన కారణాలు. ఇటీవలి కాలంలో పాక్ మద్దతు ఉన్న ఉగ్రవాదుల చొరబాటు ప్రయత్నాలు పెరిగినట్లు భారత సైన్యం పేర్కొంది. అదే సమయంలో, రాజకీయ నేతల మధ్య మాటల యుద్ధం కూడా తీవ్రంగా సాగుతోంది. ఇరు దేశాల ప్రజలు శాంతిని కోరుతున్నా, సరిహద్దుల్లో పరిస్థితి ఇంకా గందరగోళంగా ఉంది. ఈ పరిస్థితిని చర్చల ద్వారా పరిష్కరించాలనే సూచనలు అంతర్జాతీయంగా వెల్లువెత్తుతున్నాయి."""

    # Adding text about PM Modi's initiatives in India
    te_modi_text = """నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత భారతదేశం అనేక రంగాల్లో మానవ అభివృద్ధి సూచికలతోపాటు ఆర్థిక, సాంకేతిక, భౌతిక మౌలిక సదుపాయాలు, విదేశాంగ విధానం మరియు జాతీయ భద్రత వంటి కీలక విభాగాల్లో అమూల్యమైన మార్పులకు దారి తీస్తోంది. "ఆత్మనిర్భర్ భారత్" అనే భావనతో దేశీయ పరిశ్రమలను ప్రోత్సహిస్తూ, భారత్‌ను స్వయం సమృద్ధి దిశగా నడిపిస్తున్నారు. మేక్ ఇన్ ఇండియా, డిజిటల్ ఇండియా, స్టార్ట్‌అప్ ఇండియా, స్కిల్ ఇండియా వంటి కార్యక్రమాల ద్వారా యువతలో సృజనాత్మకతను వెలికి తీస్తూ, ప్రపంచంలో భారత యువశక్తి ప్రతిభను చాటుతున్నాడు. రహదారులు, రైళ్లు, ఎయిర్‌పోర్టులు, మెట్రో ప్రాజెక్టులు, బులెట్ ట్రైన్ వంటి ఆధునిక మౌలిక సదుపాయాలపై దృష్టి పెట్టి, గ్రామీణ ప్రాంతాలకు సరైన కనెక్టివిటీని తీసుకువచ్చేలా చర్యలు తీసుకుంటున్నారు."""

    hi_text = """हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत गणराज्य की आधिकारिक भाषाओं में से एक है। हिंदी की लिपि देवनागरी है। वर्तमान हिंदी का विकास खड़ी बोली से हुआ है, जो दिल्ली के आसपास के क्षेत्रों में बोली जाती थी। हिंदी शब्द फारसी मूल का है, जिसका अर्थ है 'सिंधु नदी का'। मुगल काल में सिंधु नदी के पूर्व में रहने वाले लोगों और उनकी भाषा को 'हिंदी' कहा जाता था। हिंदी भाषा के विकास को भाषावैज्ञानिक आधार पर चार कालों में बांटा गया है: आदिकाल (12वीं से 14वीं शताब्दी), भक्तिकाल (14वीं से 16वीं शताब्दी), रीतिकाल (17वीं से 19वीं शताब्दी) और आधुनिक काल (19वीं शताब्दी से अब तक)।"""

    hi_text_kashmir = """भारत और पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर लगातार गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना के अनुसार, हाल के दिनों में पाकिस्तान समर्थित आतंकवादियों की घुसपैठ की कोशिशों में वृद्धि हुई है। इसी समय, राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। दोनों देशों के लोग शांति चाहते हैं, लेकिन सीमा पर स्थिति अभी भी अस्थिर है। इस स्थिति को वार्ता के माध्यम से सुलझाने के सुझाव अंतरराष्ट्रीय स्तर पर दिए जा रहे हैं।"""

    # Reference summaries (what we're aiming for)
    te_reference = """భారత్‌-పాకిస్తాన్ మధ్య ఉద్రిక్తతలు పెరిగుతున్నాయి. కశ్మీర్ సరిహద్దుల్లో కాల్పులు, ఉగ్రవాద చర్యలు ప్రధాన కారణాలు కాగా, పాక్ మద్దతు ఉన్న చొరబాట్లు కూడా పెరుగుతున్నాయి. రాజకీయ స్థాయిలో మాటల యుద్ధం జరుగుతోంది. ప్రజలు శాంతిని కోరుతున్నా, పరిస్థితి ఇంకా అస్తవ్యస్థంగా ఉంది. సమస్యను చర్చల ద్వారా పరిష్కరించాలని అంతర్జాతీయ సమాజం సూచిస్తోంది."""

    # Reference summary for Modi's initiatives text
    te_modi_reference = """నరేంద్ర మోదీ ప్రధానమంత్రిగా బాధ్యతలు చేపట్టిన తరువాత భారతదేశం అనేక రంగాల్లో అభివృద్ధి సాధిస్తోంది. "ఆత్మనిర్భర్ భారత్" భావనతో దేశీయ పరిశ్రమలను ప్రోత్సహిస్తూ, స్వయం సమృద్ధి దిశగా దేశాన్ని నడిపిస్తున్నారు. మేక్ ఇన్ ఇండియా, డిజిటల్ ఇండియా వంటి కార్యక్రమాల ద్వారా యువతలో సృజనాత్మకతను పెంపొందిస్తున్నారు. అలాగే, రహదారులు, రైళ్లు, మెట్రో ప్రాజెక్టులు వంటి ఆధునిక మౌలిక సదుపాయాలపై దృష్టి పెట్టి, గ్రామీణ ప్రాంతాలకు మెరుగైన కనెక్టివిటీని అందిస్తున్నారు."""

    hi_reference = """हिंदी भारत की सबसे अधिक और दुनिया की चौथी सबसे अधिक बोली जाने वाली भाषा है। यह भारत की आधिकारिक भाषाओं में से एक है, जिसकी लिपि देवनागरी है। इसका विकास दिल्ली क्षेत्र की खड़ी बोली से हुआ है। 'हिंदी' शब्द फारसी मूल का है, जिसका अर्थ है 'सिंधु नदी का'। हिंदी भाषा के विकास को चार कालों में बाँटा गया है: आदिकाल, भक्तिकाल, रीतिकाल और आधुनिक काल।"""

    hi_kashmir_reference = """भारत-पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना ने पाकिस्तान समर्थित आतंकवादियों की घुसपैठ में वृद्धि की बात कही है। राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। लोग शांति चाहते हैं, पर स्थिति अस्थिर है। अंतरराष्ट्रीय स्तर पर वार्ता द्वारा समाधान के सुझाव दिए जा रहे हैं।"""

    # Test data
    test_data = [
        {
            'language': 'Telugu',
            'code': 'te', 
            'text': te_text,
            'reference': te_reference
        },
        {
            'language': 'Telugu (Modi Initiatives)',
            'code': 'te',
            'text': te_modi_text,
            'reference': te_modi_reference
        },
        {
            'language': 'Hindi (Language Info)',
            'code': 'hi',
            'text': hi_text,
            'reference': hi_reference
        },
        {
            'language': 'Hindi (Kashmir Conflict)',
            'code': 'hi',
            'text': hi_text_kashmir,
            'reference': hi_kashmir_reference
        }
    ]

    print("=" * 80)
    print("COMPARING DIFFERENT SUMMARIZATION APPROACHES")
    print("=" * 80)

    # Run tests for each approach
    for test_case in test_data:
        language = test_case['language']
        lang_code = test_case['code']
        text = test_case['text']
        reference = test_case['reference']
        
        print(f"\n\n{'-' * 40}")
        print(f"LANGUAGE: {language}")
        print(f"{'-' * 40}")
        
        print("\nORIGINAL TEXT:")
        print("-" * 15)
        print(text)
        print(f"(Length: {len(text.split())} words)")
        
        print("\nREFERENCE SUMMARY (Target):")
        print("-" * 15)
        print(reference)
        print(f"(Length: {len(reference.split())} words)")
        
        print("\n1. IMPROVED SUMMARIZER (Previous approach):")
        print("-" * 15)
        improved_summary = generate_improved_summary(text, lang_code)
        print(improved_summary)
        print(f"(Length: {len(improved_summary.split())} words)")
        
        print("\n2. NEURAL SUMMARIZER (New approach):")
        print("-" * 15)
        neural_summary = generate_neural_summary(text, lang_code)
        print(neural_summary)
        print(f"(Length: {len(neural_summary.split())} words)")
        
        # Calculate simple overlap with reference (for quantitative comparison)
        ref_words = set(reference.split())
        improved_words = set(improved_summary.split())
        neural_words = set(neural_summary.split())
        
        improved_overlap = len(ref_words.intersection(improved_words)) / len(ref_words) if ref_words else 0
        neural_overlap = len(ref_words.intersection(neural_words)) / len(ref_words) if ref_words else 0
        
        print("\nQUANTITATIVE COMPARISON:")
        print(f"Improved Summarizer - Content Overlap with Reference: {improved_overlap:.2%}")
        print(f"Neural Summarizer - Content Overlap with Reference: {neural_overlap:.2%}")
        
        # Simple qualitative assessment
        if neural_overlap > improved_overlap:
            print("\nEVALUATION: Neural summarizer produces summaries closer to the reference")
        elif neural_overlap < improved_overlap:
            print("\nEVALUATION: Improved summarizer produces summaries closer to the reference")
        else:
            print("\nEVALUATION: Both approaches perform similarly in terms of content overlap")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    compare_summarization_approaches()