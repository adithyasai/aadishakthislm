#!/usr/bin/env python3
"""
Result Collection and Analysis Tool for SLM Human Evaluations

This script processes submitted human evaluation forms and generates analysis reports.
It extracts data from evaluation forms and provides statistical analysis of the results.
"""

import os
import sys
import json
import argparse
import logging
import re
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import statistics

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_RESULTS_DIR = Path(__file__).parent / "evaluation" / "results"
DEFAULT_REPORTS_DIR = Path(__file__).parent / "evaluation" / "reports"


class EvaluationResultProcessor:
    """
    Processor for human evaluation result forms
    """
    
    def __init__(self, 
               results_dir: Path = DEFAULT_RESULTS_DIR,
               reports_dir: Path = DEFAULT_REPORTS_DIR):
        """
        Initialize the evaluation result processor
        
        Args:
            results_dir: Directory containing submitted evaluation forms
            reports_dir: Directory to save generated reports
        """
        self.results_dir = results_dir
        self.reports_dir = reports_dir
        
        # Ensure directories exist
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        logger.info(f"EvaluationResultProcessor initialized.")
        logger.info(f"Looking for results in: {results_dir}")
        logger.info(f"Reports will be saved to: {reports_dir}")
    
    def extract_form_data(self, html_file: Path) -> Dict[str, Any]:
        """
        Extract data from a submitted evaluation form
        
        Args:
            html_file: Path to the HTML form file
            
        Returns:
            Dictionary of extracted form data
        """
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract form type
            if "comparative" in html_file.name.lower():
                form_type = "comparative"
            else:
                form_type = "single"
            
            # Extract basic metadata using regex
            data = {
                "filename": html_file.name,
                "type": form_type,
                "evaluation_id": self._extract_value(html_content, "evaluation_id"),
                "document_id": self._extract_value(html_content, "document_id"),
                "evaluator_id": self._extract_value(html_content, "evaluator_id")
            }
            
            # Extract ratings based on form type
            if form_type == "comparative":
                # Extract system A ratings
                for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]:
                    data[f"system_a_{criterion}"] = self._extract_radio_value(html_content, f"system_a_{criterion}")
                
                # Extract system B ratings
                for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]:
                    data[f"system_b_{criterion}"] = self._extract_radio_value(html_content, f"system_b_{criterion}")
                
                # Extract direct comparison
                data["direct_comparison"] = self._extract_radio_value(html_content, "direct_comparison")
                
                # Extract comments
                data["system_a_comments"] = self._extract_textarea_value(html_content, "system_a_comments")
                data["system_b_comments"] = self._extract_textarea_value(html_content, "system_b_comments")
                data["comparison_reason"] = self._extract_textarea_value(html_content, "comparison_reason")
                data["general_comments"] = self._extract_textarea_value(html_content, "general_comments")
                
            else:  # single evaluation form
                # Extract ratings
                for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]:
                    data[criterion] = self._extract_radio_value(html_content, criterion)
                
                # Extract comments
                data["comments"] = self._extract_textarea_value(html_content, "comments")
                data["error_details"] = self._extract_textarea_value(html_content, "error_details")
                data["suggestions"] = self._extract_textarea_value(html_content, "suggestions")
                
                # Extract error types
                data["errors"] = self._extract_checkboxes(html_content, "errors")
            
            logger.info(f"Successfully extracted data from {html_file.name}")
            return data
        
        except Exception as e:
            logger.error(f"Error extracting data from {html_file}: {e}")
            return {"error": str(e), "filename": html_file.name}
    
    def _extract_value(self, html_content: str, field_name: str) -> str:
        """Extract a value from a hidden input field"""
        pattern = f'<input [^>]*name="{field_name}"[^>]*value="([^"]*)"'
        match = re.search(pattern, html_content)
        return match.group(1) if match else ""
    
    def _extract_radio_value(self, html_content: str, field_name: str) -> str:
        """Extract the selected value from a radio button group"""
        pattern = f'<input [^>]*type="radio"[^>]*name="{field_name}"[^>]*value="([^"]*)"[^>]*checked'
        match = re.search(pattern, html_content)
        return match.group(1) if match else ""
    
    def _extract_textarea_value(self, html_content: str, field_name: str) -> str:
        """Extract the value from a textarea field"""
        pattern = f'<textarea [^>]*name="{field_name}"[^>]*>(.*?)</textarea>'
        match = re.search(pattern, html_content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_checkboxes(self, html_content: str, field_name: str) -> List[str]:
        """Extract selected checkbox values"""
        pattern = f'<input [^>]*type="checkbox"[^>]*name="{field_name}\\[\\]"[^>]*value="([^"]*)"[^>]*checked'
        return re.findall(pattern, html_content)
    
    def process_results_directory(self, output_file: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Process all evaluation forms in the results directory
        
        Args:
            output_file: Optional path to save the JSON results
            
        Returns:
            List of dictionaries with extracted data
        """
        results = []
        
        # Process each HTML file in the results directory
        for html_file in self.results_dir.glob("*.html"):
            data = self.extract_form_data(html_file)
            results.append(data)
        
        logger.info(f"Processed {len(results)} evaluation forms")
        
        # Save results if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def generate_csv_report(self, results: List[Dict[str, Any]], output_file: Path) -> None:
        """
        Generate a CSV report from the results
        
        Args:
            results: List of dictionaries with extracted data
            output_file: Path to save the CSV file
        """
        if not results:
            logger.warning("No results to generate report from")
            return
        
        # Get all field names across all results
        field_names = set()
        for result in results:
            field_names.update(result.keys())
        
        # Order common fields first, then alphabetical
        common_fields = ["filename", "type", "evaluation_id", "document_id", "evaluator_id"]
        remaining_fields = sorted(field_names - set(common_fields))
        ordered_fields = common_fields + remaining_fields
        
        # Write CSV file
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_fields)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        logger.info(f"CSV report saved to {output_file}")
    
    def generate_summary_report(self, results: List[Dict[str, Any]], output_file: Path) -> Dict[str, Any]:
        """
        Generate a summary report from the results
        
        Args:
            results: List of dictionaries with extracted data
            output_file: Path to save the report
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            logger.warning("No results to generate report from")
            return {}
        
        # Split results by type
        comparative_results = [r for r in results if r.get("type") == "comparative"]
        single_results = [r for r in results if r.get("type") == "single"]
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(results),
            "comparative": self._analyze_comparative_results(comparative_results),
            "single": self._analyze_single_results(single_results)
        }
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Summary report saved to {output_file}")
        return report
    
    def _analyze_comparative_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparative evaluation results"""
        if not results:
            return {"count": 0}
        
        # Count direct comparisons
        comparison_counts = {}
        for result in results:
            comparison = result.get("direct_comparison", "")
            if comparison:
                if comparison not in comparison_counts:
                    comparison_counts[comparison] = 0
                comparison_counts[comparison] += 1
        
        # Calculate average scores
        system_a_scores = {criterion: [] for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]}
        system_b_scores = {criterion: [] for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]}
        
        for result in results:
            for criterion in system_a_scores:
                value = result.get(f"system_a_{criterion}")
                if value and value.isdigit():
                    system_a_scores[criterion].append(int(value))
                
                value = result.get(f"system_b_{criterion}")
                if value and value.isdigit():
                    system_b_scores[criterion].append(int(value))
        
        # Calculate averages
        system_a_averages = {}
        system_b_averages = {}
        
        for criterion, values in system_a_scores.items():
            if values:
                system_a_averages[criterion] = {
                    "mean": statistics.mean(values),
                    "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        for criterion, values in system_b_scores.items():
            if values:
                system_b_averages[criterion] = {
                    "mean": statistics.mean(values),
                    "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return {
            "count": len(results),
            "direct_comparisons": comparison_counts,
            "system_a": system_a_averages,
            "system_b": system_b_averages
        }
    
    def _analyze_single_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze single evaluation results"""
        if not results:
            return {"count": 0}
        
        # Group results by system name/id
        system_results = {}
        for result in results:
            system = result.get("system_id", "unknown")
            if system not in system_results:
                system_results[system] = []
            system_results[system].append(result)
        
        # Calculate statistics for each system
        system_stats = {}
        for system, sys_results in system_results.items():
            # Collect scores for each criterion
            scores = {criterion: [] for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]}
            error_counts = {}
            
            for result in sys_results:
                # Add scores
                for criterion in scores:
                    value = result.get(criterion)
                    if value and value.isdigit():
                        scores[criterion].append(int(value))
                
                # Count error types
                for error in result.get("errors", []):
                    if error not in error_counts:
                        error_counts[error] = 0
                    error_counts[error] += 1
            
            # Calculate statistics
            averages = {}
            for criterion, values in scores.items():
                if values:
                    averages[criterion] = {
                        "mean": statistics.mean(values),
                        "stddev": statistics.stdev(values) if len(values) > 1 else 0,
                        "min": min(values),
                        "max": max(values),
                        "count": len(values)
                    }
            
            system_stats[system] = {
                "count": len(sys_results),
                "scores": averages,
                "errors": error_counts
            }
        
        return {
            "count": len(results),
            "systems": system_stats
        }
    
    def generate_human_readable_report(self, report: Dict[str, Any], output_file: Path) -> None:
        """
        Generate a human-readable report in markdown format
        
        Args:
            report: Dictionary with report data
            output_file: Path to save the report
        """
        lines = []
        lines.append("# SLM Human Evaluation Results")
        lines.append("")
        lines.append(f"**Report generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total evaluations:** {report.get('total_evaluations', 0)}")
        lines.append("")
        
        # Comparative evaluations
        comparative = report.get("comparative", {})
        lines.append("## Comparative Evaluations")
        lines.append("")
        lines.append(f"**Number of evaluations:** {comparative.get('count', 0)}")
        lines.append("")
        
        if comparative.get("count", 0) > 0:
            # Direct comparisons
            lines.append("### Direct Comparisons")
            lines.append("")
            lines.append("| Preference | Count | Percentage |")
            lines.append("|------------|-------|------------|")
            
            comparisons = comparative.get("direct_comparisons", {})
            total = sum(comparisons.values())
            
            for preference, count in comparisons.items():
                percentage = (count / total) * 100 if total > 0 else 0
                lines.append(f"| {preference} | {count} | {percentage:.1f}% |")
            
            lines.append("")
            
            # System scores
            lines.append("### Average Ratings")
            lines.append("")
            lines.append("| Criterion | System A | System B | Difference |")
            lines.append("|-----------|----------|----------|------------|")
            
            system_a = comparative.get("system_a", {})
            system_b = comparative.get("system_b", {})
            
            for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]:
                a_score = system_a.get(criterion, {}).get("mean", "N/A")
                b_score = system_b.get(criterion, {}).get("mean", "N/A")
                
                if a_score != "N/A" and b_score != "N/A":
                    diff = a_score - b_score
                    lines.append(f"| {criterion.capitalize()} | {a_score:.2f} | {b_score:.2f} | {diff:+.2f} |")
                else:
                    lines.append(f"| {criterion.capitalize()} | {a_score} | {b_score} | N/A |")
            
            lines.append("")
        
        # Single evaluations
        single = report.get("single", {})
        lines.append("## Single Summary Evaluations")
        lines.append("")
        lines.append(f"**Number of evaluations:** {single.get('count', 0)}")
        lines.append("")
        
        if single.get("count", 0) > 0:
            # System scores
            lines.append("### System Scores")
            lines.append("")
            
            systems = single.get("systems", {})
            for system, stats in systems.items():
                lines.append(f"#### {system}")
                lines.append("")
                lines.append(f"**Number of evaluations:** {stats.get('count', 0)}")
                lines.append("")
                
                lines.append("| Criterion | Mean | StdDev | Min | Max |")
                lines.append("|-----------|------|--------|-----|-----|")
                
                scores = stats.get("scores", {})
                for criterion in ["fluency", "coherence", "relevance", "adequacy", "overall"]:
                    if criterion in scores:
                        score = scores[criterion]
                        lines.append(f"| {criterion.capitalize()} | {score.get('mean', 'N/A'):.2f} | {score.get('stddev', 'N/A'):.2f} | {score.get('min', 'N/A')} | {score.get('max', 'N/A')} |")
                
                lines.append("")
                
                # Error types
                errors = stats.get("errors", {})
                if errors:
                    lines.append("**Common Errors:**")
                    lines.append("")
                    lines.append("| Error Type | Count | Percentage |")
                    lines.append("|------------|-------|------------|")
                    
                    for error, count in errors.items():
                        percentage = (count / stats.get('count', 1)) * 100
                        lines.append(f"| {error} | {count} | {percentage:.1f}% |")
                
                lines.append("")
        
        # Write the report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Human-readable report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process and analyze human evaluation results")
    parser.add_argument("--results_dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Directory containing submitted evaluation forms")
    parser.add_argument("--reports_dir", type=str, default=str(DEFAULT_REPORTS_DIR), help="Directory to save generated reports")
    parser.add_argument("--output_json", action="store_true", help="Generate JSON output file")
    parser.add_argument("--output_csv", action="store_true", help="Generate CSV report")
    parser.add_argument("--output_summary", action="store_true", help="Generate summary report")
    parser.add_argument("--output_markdown", action="store_true", help="Generate human-readable markdown report")
    parser.add_argument("--output_all", action="store_true", help="Generate all report types")
    
    args = parser.parse_args()
    
    # Default to generating all reports if none specified
    generate_all = args.output_all or not (args.output_json or args.output_csv or args.output_summary or args.output_markdown)
    
    # Create the processor
    results_dir = Path(args.results_dir)
    reports_dir = Path(args.reports_dir)
    processor = EvaluationResultProcessor(results_dir=results_dir, reports_dir=reports_dir)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process results
    json_output = reports_dir / f"results_{timestamp}.json" if args.output_json or generate_all else None
    results = processor.process_results_directory(output_file=json_output)
    
    if not results:
        logger.warning("No evaluation forms found to process")
        return
    
    # Generate CSV report
    if args.output_csv or generate_all:
        csv_output = reports_dir / f"evaluation_results_{timestamp}.csv"
        processor.generate_csv_report(results, csv_output)
    
    # Generate summary report
    summary_report = None
    if args.output_summary or args.output_markdown or generate_all:
        summary_output = reports_dir / f"summary_report_{timestamp}.json"
        summary_report = processor.generate_summary_report(results, summary_output)
    
    # Generate human-readable report
    if args.output_markdown or generate_all:
        if summary_report:
            markdown_output = reports_dir / f"human_readable_report_{timestamp}.md"
            processor.generate_human_readable_report(summary_report, markdown_output)
    
    logger.info("Processing complete")


if __name__ == "__main__":
    main()
