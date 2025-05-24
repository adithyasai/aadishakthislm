import os
import sys
import torch
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import IndicSLMConfig, IndicSLM
from src.tokenizer import IndicTokenizer

# Setup paths
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"

# Test set with long-range dependency examples
# Format: (input text, subject entity, related entity/verb)
LONG_RANGE_TEST_CASES = {
    "hindi": [
        {
            "text": "<hi> राम, जो कि एक प्रसिद्ध वैज्ञानिक है और जिसने कई महत्वपूर्ण अनुसंधान किए हैं जो विज्ञान के क्षेत्र में क्रांतिकारी परिवर्तन लाए हैं, दिल्ली विश्वविद्यालय में प्रोफेसर है।",
            "subject": "राम",
            "related": "प्रोफेसर",
            "distance": 25  # Number of tokens between subject and related entity
        },
        {
            "text": "<hi> मेरी बहन, जिसने पिछले साल एक प्रतिष्ठित विश्वविद्यालय से स्नातक की उपाधि प्राप्त की थी और जो अब एक बड़ी बहुराष्ट्रीय कंपनी में कार्यरत है, आज शाम को हमारे घर आ रही है।",
            "subject": "मेरी बहन",
            "related": "आ रही है",
            "distance": 30
        }
    ],
    "telugu": [
        {
            "text": "<te> రాము, ఎవరైతే ప్రముఖ శాస్త్రవేత్త మరియు అనేక ముఖ్యమైన పరిశోధనలు చేసి, శాస్త్రీయ రంగంలో విప్లవాత్మక మార్పులు తెచ్చారో, ఢిల్లీ విశ్వవిద్యాలయంలో ప్రొఫెసర్ గా ఉన్నారు.",
            "subject": "రాము",
            "related": "ప్రొఫెసర్",
            "distance": 28
        },
        {
            "text": "<te> నా సోదరి, గత సంవత్సరం ప్రతిష్టాత్మక విశ్వవిద్యాలయం నుండి పట్టా పొందింది మరియు ఇప్పుడు పెద్ద బహుళజాతి సంస్థలో పనిచేస్తుంది, ఈ రోజు సాయంత్రం మా ఇంటికి వస్తుంది.",
            "subject": "నా సోదరి",
            "related": "వస్తుంది",
            "distance": 25
        }
    ],
    "mixed": [
        {
            "text": "<en> राम, who is a famous वैज्ञानिक and has conducted several important अनुसंधान which have brought revolutionary changes in the field of science, is a professor at Delhi University.",
            "subject": "राम",
            "related": "professor",
            "distance": 27
        }
    ]
}

# Additional test cases for pronoun reference resolution
REFERENCE_RESOLUTION_TEST_CASES = {
    "hindi": [
        {
            "text": "<hi> राम एक मेहनती छात्र है। वह हमेशा अपनी पढ़ाई पर ध्यान देता है। वह कक्षा में हमेशा प्रथम आता है।",
            "entity": "राम",
            "pronoun": "वह",
            "sentences_apart": 1
        },
        {
            "text": "<hi> मेरी बहन दिल्ली में रहती है। वह एक डॉक्टर है। पिछले साल, वह हमारे घर आई थी।",
            "entity": "मेरी बहन",
            "pronoun": "वह",
            "sentences_apart": 2
        }
    ],
    "telugu": [
        {
            "text": "<te> రాము చాలా కష్టపడి చదివే విద్యార్థి. అతను ఎప్పుడూ తన చదువు మీద శ్రద్ధ చూపిస్తాడు. అతను తరగతిలో ఎప్పుడూ మొదటి స్థానంలో ఉంటాడు.",
            "entity": "రాము",
            "pronoun": "అతను",
            "sentences_apart": 1
        }
    ]
}

def evaluate_long_range_dependencies(model, tokenizer, device="cpu"):
    """
    Evaluate model's ability to handle long-range dependencies
    in Indic languages using specially designed test cases.
    """
    results = []
    
    # Process each test case
    for lang, test_cases in LONG_RANGE_TEST_CASES.items():
        for test_case in test_cases:
            # Encode the text
            encoding = tokenizer.encode(test_case["text"])
            input_ids = torch.tensor([encoding["input_ids"]]).to(device)
            attention_mask = torch.tensor([encoding["attention_mask"]]).to(device)
            
            # Get subject and related entity token indices
            text_tokens = [tokenizer.sp_model.id_to_piece(id) for id in encoding["input_ids"]]
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Get encoder self-attention weights (if possible)
            attention_weights = None
            # Note: This is model-specific and might need adjustment
            if hasattr(model, "encoder") and hasattr(model.encoder[0].attention, "forward"):
                # For custom attention mechanisms that return attention weights
                _, attention_weights = model.encoder[0].attention.forward(
                    query=outputs["logits"], 
                    key=outputs["logits"], 
                    value=outputs["logits"],
                    key_padding_mask=attention_mask.eq(0)
                )
            
            # Calculate perplexity on the sequence
            logits = outputs["logits"]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Shift predictions and labels for perplexity calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), 
                           shift_labels.view(-1))
            
            # Calculate perplexity
            perplexity = torch.exp(torch.mean(loss))
            
            # Store results
            results.append({
                "language": lang,
                "test_case": test_case["text"],
                "subject": test_case["subject"],
                "related": test_case["related"],
                "token_distance": test_case["distance"],
                "perplexity": perplexity.item(),
                "attention_weights": attention_weights
            })
    
    return results

def evaluate_reference_resolution(model, tokenizer, device="cpu"):
    """
    Evaluate model's ability to resolve pronoun references
    across sentences in Indic languages.
    """
    results = []
    
    # Process each test case
    for lang, test_cases in REFERENCE_RESOLUTION_TEST_CASES.items():
        for test_case in test_cases:
            # Encode the text
            encoding = tokenizer.encode(test_case["text"])
            input_ids = torch.tensor([encoding["input_ids"]]).to(device)
            attention_mask = torch.tensor([encoding["attention_mask"]]).to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Calculate perplexity on the sequence
            logits = outputs["logits"]
            
            # Shift predictions and labels for perplexity calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), 
                           shift_labels.view(-1))
            
            # Calculate perplexity
            perplexity = torch.exp(torch.mean(loss))
            
            # Store results
            results.append({
                "language": lang,
                "test_case": test_case["text"],
                "entity": test_case["entity"],
                "pronoun": test_case["pronoun"],
                "sentences_apart": test_case["sentences_apart"],
                "perplexity": perplexity.item()
            })
    
    return results

def evaluate_attention_mechanisms():
    """
    Compare different attention mechanisms on tasks requiring
    long-range dependencies in Indic languages.
    """
    # Load configuration
    with open(CONFIG_PATH) as f:
        config_data = json.load(f)
    
    # Initialize tokenizer
    tokenizer = IndicTokenizer(str(CONFIG_PATH))
    print(f"Tokenizer vocabulary size: {tokenizer.sp_model.get_piece_size()}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test each attention mechanism
    attention_types = ["default", "relative", "rotary"]
    all_long_range_results = []
    all_reference_results = []
    
    for attention_type in attention_types:
        print(f"\n----- Evaluating {attention_type.upper()} attention mechanism -----")
        
        # Create configuration with specific attention type
        config = IndicSLMConfig(
            vocab_size=config_data["vocab_size"],
            hidden_size=config_data.get("n_embd", 384),
            num_hidden_layers=config_data.get("n_layer", 6),
            num_attention_heads=config_data.get("n_head", 6),
            attention_type=attention_type
        )
        
        # Initialize model
        model = IndicSLM(config).to(device)
        print(f"Model initialized with {attention_type} attention")
        
        # Evaluate long-range dependencies
        print("Evaluating long-range dependencies...")
        long_range_results = evaluate_long_range_dependencies(model, tokenizer, device)
        
        # Evaluate reference resolution
        print("Evaluating pronoun reference resolution...")
        reference_results = evaluate_reference_resolution(model, tokenizer, device)
        
        # Add attention type to results
        for result in long_range_results:
            result["attention_type"] = attention_type
        for result in reference_results:
            result["attention_type"] = attention_type
        
        # Aggregate results
        all_long_range_results.extend(long_range_results)
        all_reference_results.extend(reference_results)
        
        # Print summary statistics
        avg_perplexity_long_range = sum(r["perplexity"] for r in long_range_results) / len(long_range_results)
        avg_perplexity_reference = sum(r["perplexity"] for r in reference_results) / len(reference_results)
        
        print(f"Average perplexity on long-range dependency tasks: {avg_perplexity_long_range:.2f}")
        print(f"Average perplexity on reference resolution tasks: {avg_perplexity_reference:.2f}")
    
    # Convert results to DataFrame for analysis
    long_range_df = pd.DataFrame(all_long_range_results)
    reference_df = pd.DataFrame(all_reference_results)
    
    # Compare attention mechanisms
    print("\n----- Attention Mechanism Comparison -----")
    
    # Long-range dependency comparison
    print("\nLong-range dependency performance by attention type:")
    lr_comparison = long_range_df.groupby("attention_type")["perplexity"].mean()
    print(lr_comparison)
    
    # Reference resolution comparison
    print("\nPronoun reference resolution performance by attention type:")
    ref_comparison = reference_df.groupby("attention_type")["perplexity"].mean()
    print(ref_comparison)
    
    # Language-specific comparison
    print("\nPerformance by language and attention type (long-range dependencies):")
    lang_comparison = long_range_df.groupby(["language", "attention_type"])["perplexity"].mean()
    print(lang_comparison)
    
    print("\nEvaluation completed!")
    
    return long_range_df, reference_df

if __name__ == "__main__":
    evaluate_attention_mechanisms()