#!/usr/bin/env python3
"""
Human Evaluation Template Generator for SLM Project

This script generates HTML templates for human evaluation of summarization models.
It can create both comparative evaluation forms (comparing two summarization systems)
and single summary evaluation forms.
"""

import os
import sys
import json
import argparse
import logging
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.improved_summarizer import generate_improved_summary
from src.neural_summarizer import generate_neural_summary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Template paths
TEMPLATE_DIR = Path(__file__).parent.parent / "evaluation" / "templates"
COMPARATIVE_TEMPLATE = TEMPLATE_DIR / "evaluation_form.html"
SINGLE_TEMPLATE = TEMPLATE_DIR / "single_summary_evaluation.html"

# Default output directory for generated forms
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "evaluation" / "generated"

# Language map
LANGUAGE_MAP = {
    'te': 'Telugu',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'ml': 'Malayalam',
    'en': 'English'
}

class HumanEvaluationGenerator:
    """
    Generator for human evaluation forms for summarization systems
    """
    
    def __init__(self, 
               template_dir: Path = TEMPLATE_DIR,
               output_dir: Path = DEFAULT_OUTPUT_DIR,
               randomize_order: bool = True):
        """
        Initialize the human evaluation template generator
        
        Args:
            template_dir: Directory containing HTML templates
            output_dir: Directory to save generated evaluation forms
            randomize_order: Whether to randomize the order of systems in comparative evaluations
        """
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.randomize_order = randomize_order
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load templates
        self._load_templates()
        
        logger.info(f"HumanEvaluationGenerator initialized. Templates dir: {template_dir}")
        logger.info(f"Generated forms will be saved to: {output_dir}")
    
    def _load_templates(self):
        """Load HTML templates from the template directory"""
        try:
            with open(COMPARATIVE_TEMPLATE, 'r', encoding='utf-8') as f:
                self.comparative_template = f.read()
            
            with open(SINGLE_TEMPLATE, 'r', encoding='utf-8') as f:
                self.single_template = f.read()
                
            logger.info("Templates loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Template not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            raise
    
    def _replace_placeholders(self, template: str, data: Dict[str, Any]) -> str:
        """
        Replace placeholders in the template with actual data
        
        Args:
            template: HTML template string
            data: Dictionary of values to replace in the template
            
        Returns:
            Processed HTML with placeholders replaced
        """
        result = template
        
        # Handle conditional blocks first (sections with {{#variable}} content {{/variable}})
        for key in data:
            if data[key] is not None:
                open_tag = f"{{{{#{key}}}}}"
                close_tag = f"{{{{/{key}}}}}"
                
                # Find all conditional blocks for this variable
                start = 0
                while True:
                    start_pos = result.find(open_tag, start)
                    if start_pos == -1:
                        break
                        
                    end_pos = result.find(close_tag, start_pos)
                    if end_pos == -1:
                        break
                    
                    # Extract the block content
                    block_start = start_pos + len(open_tag)
                    block_content = result[block_start:end_pos]
                    
                    # If the value is truthy, keep the content but remove the conditional tags
                    if data[key]:
                        result = result[:start_pos] + block_content + result[end_pos + len(close_tag):]
                        # Don't update start position as the string has changed
                    else:
                        # Remove the entire block including tags
                        result = result[:start_pos] + result[end_pos + len(close_tag):]
                        # Don't update start position as the string has changed
                    
                    # Check if we've reached the end
                    if start_pos >= len(result):
                        break
        
        # Replace simple placeholders {{variable}}
        for key, value in data.items():
            if value is not None:
                placeholder = f"{{{{{key}}}}}"
                result = result.replace(placeholder, str(value))
        
        return result
    
    def generate_comparative_evaluation(self, 
                                      original_text: str,
                                      system_a_summary: str,
                                      system_b_summary: str,
                                      language_code: str = 'en',
                                      reference_summary: Optional[str] = None,
                                      system_a_name: str = "System A",
                                      system_b_name: str = "System B",
                                      evaluation_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate a comparative evaluation form for two summarization systems
        
        Args:
            original_text: The original document text
            system_a_summary: Summary generated by system A
            system_b_summary: Summary generated by system B
            language_code: Language code of the text
            reference_summary: Optional reference summary
            system_a_name: Name of system A
            system_b_name: Name of system B
            evaluation_id: Optional evaluation ID
            
        Returns:
            Tuple of (generated HTML content, filename)
        """
        # Generate unique ID if not provided
        if evaluation_id is None:
            evaluation_id = f"eval_{uuid.uuid4().hex[:8]}"
        
        # Get language name from code
        language_name = LANGUAGE_MAP.get(language_code, "Unknown")
        
        # Randomize system order if requested
        if self.randomize_order and random.random() > 0.5:
            system_a_summary, system_b_summary = system_b_summary, system_a_summary
            system_a_name, system_b_name = system_b_name, system_a_name
        
        # Prepare data for template
        data = {
            "evaluation_id": evaluation_id,
            "document_id": f"doc_{uuid.uuid4().hex[:8]}",
            "evaluator_id": "",  # To be filled by the evaluator
            "language_name": language_name,
            "original_text": original_text,
            "reference_summary": reference_summary,
            "system_a_summary": system_a_summary,
            "system_b_summary": system_b_summary,
            "system_a_name": system_a_name,
            "system_b_name": system_b_name
        }
        
        # Generate the HTML
        html_content = self._replace_placeholders(self.comparative_template, data)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparative_{language_code}_{evaluation_id}_{timestamp}.html"
        
        # Save the HTML file
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comparative evaluation form generated: {output_path}")
        return html_content, filename
    
    def generate_single_evaluation(self,
                                 original_text: str,
                                 system_summary: str,
                                 language_code: str = 'en',
                                 reference_summary: Optional[str] = None,
                                 system_name: str = "System",
                                 evaluation_id: Optional[str] = None) -> Tuple[str, str]:
        """
        Generate an evaluation form for a single summary
        
        Args:
            original_text: The original document text
            system_summary: Summary to be evaluated
            language_code: Language code of the text
            reference_summary: Optional reference summary
            system_name: Name of the summarization system
            evaluation_id: Optional evaluation ID
            
        Returns:
            Tuple of (generated HTML content, filename)
        """
        # Generate unique ID if not provided
        if evaluation_id is None:
            evaluation_id = f"eval_{uuid.uuid4().hex[:8]}"
        
        # Get language name from code
        language_name = LANGUAGE_MAP.get(language_code, "Unknown")
        
        # Prepare data for template
        data = {
            "evaluation_id": evaluation_id,
            "document_id": f"doc_{uuid.uuid4().hex[:8]}",
            "evaluator_id": "",  # To be filled by the evaluator
            "system_id": system_name.replace(" ", "_").lower(),
            "language_name": language_name,
            "original_text": original_text,
            "reference_summary": reference_summary,
            "system_summary": system_summary,
            "system_name": system_name
        }
        
        # Generate the HTML
        html_content = self._replace_placeholders(self.single_template, data)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"single_{language_code}_{system_name.replace(' ', '_')}_{evaluation_id}_{timestamp}.html"
        
        # Save the HTML file
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Single evaluation form generated: {output_path}")
        return html_content, filename
    
    def generate_batch_evaluations(self, 
                                 test_cases: List[Dict[str, str]],
                                 system_names: List[str],
                                 output_csv: Optional[str] = None) -> List[str]:
        """
        Generate a batch of evaluation forms for multiple test cases
        
        Args:
            test_cases: List of dictionaries with keys 'text', 'reference', 'language_code'
            system_names: List of summarization system names
            output_csv: Optional path to save a CSV index of generated forms
            
        Returns:
            List of generated filenames
        """
        generated_files = []
        
        # Track evaluation details for CSV
        evaluations = []
        
        # Generate comparative forms for each pair of systems
        num_systems = len(system_names)
        if num_systems >= 2:
            # For each test case, compare all pairs of systems
            for test_case in test_cases:
                original_text = test_case["text"]
                reference = test_case.get("reference")
                lang_code = test_case.get("language_code", "en")
                
                # Generate summaries for each system
                system_summaries = {}
                for system_name in system_names:
                    # In a real system, you would call the actual summarization systems
                    # Here we're simulating with the existing summarizers
                    if system_name.lower() == "improved":
                        summary = generate_improved_summary(original_text, lang_code)
                    elif system_name.lower() == "neural":
                        summary = generate_neural_summary(original_text, lang_code)
                    else:
                        # For systems without implementation, create a mock summary
                        summary = f"[{system_name} summary for {lang_code} text]"
                    
                    system_summaries[system_name] = summary
                
                # Generate comparative forms for selected pairs of systems
                # Here we're comparing all pairs, but you could select specific pairs
                for i in range(num_systems):
                    for j in range(i+1, num_systems):
                        system_a = system_names[i]
                        system_b = system_names[j]
                        
                        evaluation_id = f"comp_{uuid.uuid4().hex[:8]}"
                        
                        # Generate the form
                        _, filename = self.generate_comparative_evaluation(
                            original_text=original_text,
                            system_a_summary=system_summaries[system_a],
                            system_b_summary=system_summaries[system_b],
                            language_code=lang_code,
                            reference_summary=reference,
                            system_a_name=system_a,
                            system_b_name=system_b,
                            evaluation_id=evaluation_id
                        )
                        
                        generated_files.append(filename)
                        
                        # Record details for CSV
                        evaluations.append({
                            "filename": filename,
                            "evaluation_id": evaluation_id,
                            "type": "comparative",
                            "language_code": lang_code,
                            "system_a": system_a,
                            "system_b": system_b
                        })
        
        # Generate single evaluation forms for each system
        for test_case in test_cases:
            original_text = test_case["text"]
            reference = test_case.get("reference")
            lang_code = test_case.get("language_code", "en")
            
            for system_name in system_names:
                # Generate or retrieve the summary
                if system_name.lower() == "improved":
                    summary = generate_improved_summary(original_text, lang_code)
                elif system_name.lower() == "neural":
                    summary = generate_neural_summary(original_text, lang_code)
                else:
                    # For systems without implementation, create a mock summary
                    summary = f"[{system_name} summary for {lang_code} text]"
                
                evaluation_id = f"single_{uuid.uuid4().hex[:8]}"
                
                # Generate the form
                _, filename = self.generate_single_evaluation(
                    original_text=original_text,
                    system_summary=summary,
                    language_code=lang_code,
                    reference_summary=reference,
                    system_name=system_name,
                    evaluation_id=evaluation_id
                )
                
                generated_files.append(filename)
                
                # Record details for CSV
                evaluations.append({
                    "filename": filename,
                    "evaluation_id": evaluation_id,
                    "type": "single",
                    "language_code": lang_code,
                    "system": system_name
                })
        
        # Save CSV index if requested
        if output_csv:
            import csv
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["filename", "evaluation_id", "type", "language_code", "system", "system_a", "system_b"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for evaluation in evaluations:
                    writer.writerow(evaluation)
            
            logger.info(f"Evaluation index saved to {output_csv}")
        
        return generated_files
    
    def analyze_results(self, results_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze the results of human evaluations
        
        Args:
            results_file: Path to JSON or CSV file with evaluation results
            output_file: Optional path to save the analysis report
            
        Returns:
            Dictionary of analysis results
        """
        # Load results
        if results_file.endswith('.json'):
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        elif results_file.endswith('.csv'):
            import csv
            results = []
            with open(results_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)
        else:
            raise ValueError(f"Unsupported file format: {results_file}")
        
        # Perform analysis (simplified example)
        analysis = {
            "num_evaluations": len(results),
            "comparative": {
                "count": 0,
                "system_wins": {},
                "average_ratings": {}
            },
            "single": {
                "count": 0,
                "system_scores": {},
                "criteria_averages": {}
            }
        }
        
        # Process each evaluation result
        for result in results:
            # Process comparative evaluations
            if result.get("type") == "comparative":
                analysis["comparative"]["count"] += 1
                
                # Record system preferences
                preference = result.get("direct_comparison")
                if preference:
                    if "a" in preference:
                        system = result.get("system_a")
                    elif "b" in preference:
                        system = result.get("system_b")
                    else:
                        system = "equal"
                    
                    if system not in analysis["comparative"]["system_wins"]:
                        analysis["comparative"]["system_wins"][system] = 0
                    analysis["comparative"]["system_wins"][system] += 1
            
            # Process single evaluations
            elif result.get("type") == "single":
                analysis["single"]["count"] += 1
                
                system = result.get("system")
                if system not in analysis["single"]["system_scores"]:
                    analysis["single"]["system_scores"][system] = {
                        "fluency": [],
                        "coherence": [],
                        "relevance": [],
                        "adequacy": [],
                        "overall": []
                    }
                
                # Record scores
                for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]:
                    if criterion in result:
                        analysis["single"]["system_scores"][system][criterion].append(float(result[criterion]))
        
        # Calculate averages for single evaluations
        for system, scores in analysis["single"]["system_scores"].items():
            for criterion, values in scores.items():
                if values:
                    scores[criterion] = sum(values) / len(values)
        
        # Save analysis if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Analysis saved to {output_file}")
        
        return analysis


def generate_test_cases() -> List[Dict[str, str]]:
    """Generate sample test cases for evaluation"""
    return [
        {
            "text": """ప్రస్తుతం భారత్‌-పాకిస్తాన్ మధ్య పరిస్థితి మరింత ఉద్రిక్తంగా మారుతున్నది. కశ్మీర్ సరిహద్దుల్లో తరచూ కాల్పులు జరగడం, ఉగ్రవాద చర్యలు కొనసాగడం ఈ ఉద్రిక్తతకు ప్రధాన కారణాలు. ఇటీవలి కాలంలో పాక్ మద్దతు ఉన్న ఉగ్రవాదుల చొరబాటు ప్రయత్నాలు పెరిగినట్లు భారత సైన్యం పేర్కొంది. అదే సమయంలో, రాజకీయ నేతల మధ్య మాటల యుద్ధం కూడా తీవ్రంగా సాగుతోంది. ఇరు దేశాల ప్రజలు శాంతిని కోరుతున్నా, సరిహద్దుల్లో పరిస్థితి ఇంకా గందరగోళంగా ఉంది. ఈ పరిస్థితిని చర్చల ద్వారా పరిష్కరించాలనే సూచనలు అంతర్జాతీయంగా వెల్లువెత్తుతున్నాయి.""",
            "reference": """భారత్‌-పాకిస్తాన్ మధ్య ఉద్రిక్తతలు పెరిగుతున్నాయి. కశ్మీర్ సరిహద్దుల్లో కాల్పులు, ఉగ్రవాద చర్యలు ప్రధాన కారణాలు కాగా, పాక్ మద్దతు ఉన్న చొరబాట్లు కూడా పెరుగుతున్నాయి. రాజకీయ స్థాయిలో మాటల యుద్ధం జరుగుతోంది. ప్రజలు శాంతిని కోరుతున్నా, పరిస్థితి ఇంకా అస్తవ్యస్థంగా ఉంది. సమస్యను చర్చల ద్వారా పరిష్కరించాలని అంతర్జాతీయ సమాజం సూచిస్తోంది.""",
            "language_code": "te"
        },
        {
            "text": """भारत और पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर लगातार गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना के अनुसार, हाल के दिनों में पाकिस्तान समर्थित आतंकवादियों की घुसपैठ की कोशिशों में वृद्धि हुई है। इसी समय, राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। दोनों देशों के लोग शांति चाहते हैं, लेकिन सीमा पर स्थिति अभी भी अस्थिर है। इस स्थिति को वार्ता के माध्यम से सुलझाने के सुझाव अंतरराष्ट्रीय स्तर पर दिए जा रहे हैं।""",
            "reference": """भारत-पाकिस्तान के बीच तनाव बढ़ रहा है। कश्मीर सीमा पर गोलीबारी और आतंकवादी गतिविधियां जारी हैं। भारतीय सेना ने पाकिस्तान समर्थित आतंकवादियों की घुसपैठ में वृद्धि की बात कही है। राजनीतिक नेताओं के बीच वाक्युद्ध भी तेज है। लोग शांति चाहते हैं, पर स्थिति अस्थिर है। अंतरराष्ट्रीय स्तर पर वार्ता द्वारा समाधान के सुझाव दिए जा रहे हैं।""",
            "language_code": "hi"
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate human evaluation templates for summarization models")
    parser.add_argument("--generate_templates", action="store_true", help="Generate evaluation templates")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to save generated forms")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of evaluation forms to generate in batch mode")
    parser.add_argument("--systems", type=str, default="improved,neural", help="Comma-separated list of system names")
    parser.add_argument("--results_file", type=str, help="Path to evaluation results file for analysis")
    parser.add_argument("--analysis_output", type=str, help="Path to save analysis report")
    
    args = parser.parse_args()
    
    # Create the generator
    output_dir = Path(args.output_dir)
    generator = HumanEvaluationGenerator(output_dir=output_dir)
    
    if args.generate_templates:
        logger.info("Generating human evaluation templates")
        
        # Parse system names
        system_names = [name.strip() for name in args.systems.split(",")]
        logger.info(f"Systems to evaluate: {system_names}")
        
        # Generate test cases
        test_cases = generate_test_cases()
        
        # Generate batch of evaluation forms
        output_csv = output_dir / "evaluation_index.csv"
        generated_files = generator.generate_batch_evaluations(
            test_cases=test_cases,
            system_names=system_names,
            output_csv=output_csv
        )
        
        logger.info(f"Generated {len(generated_files)} evaluation forms")
    
    if args.results_file:
        logger.info(f"Analyzing results from {args.results_file}")
        analysis = generator.analyze_results(
            results_file=args.results_file,
            output_file=args.analysis_output
        )
        
        # Print summary of analysis
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Total evaluations: {analysis['num_evaluations']}")
        
        if analysis["comparative"]["count"] > 0:
            print("\nSystem preferences in comparative evaluations:")
            for system, wins in analysis["comparative"]["system_wins"].items():
                percentage = (wins / analysis["comparative"]["count"]) * 100
                print(f"  {system}: {wins} wins ({percentage:.1f}%)")
        
        if analysis["single"]["count"] > 0:
            print("\nAverage scores in single evaluations:")
            for system, scores in analysis["single"]["system_scores"].items():
                print(f"\n  {system}:")
                for criterion, score in scores.items():
                    if isinstance(score, (int, float)):
                        print(f"    {criterion}: {score:.2f}")
    
    # If no action specified, print help
    if not (args.generate_templates or args.results_file):
        parser.print_help()


if __name__ == "__main__":
    main()
