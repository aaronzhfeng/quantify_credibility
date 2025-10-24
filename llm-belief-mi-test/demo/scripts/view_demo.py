#!/usr/bin/env python3
"""
View and analyze demo JSON files.

Usage:
    python view_demo.py --question 0 --method all
    python view_demo.py --question 0 --method mi --verbose
    python view_demo.py --export-markdown demo_report.md
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional


def load_demo(question_id: int, demo_dir: str = "demo/outputs") -> Dict[str, Any]:
    """Load demo data for a specific question."""
    demo_file = Path(demo_dir) / f"question_{question_id}.json"
    
    if not demo_file.exists():
        raise FileNotFoundError(f"Demo file not found: {demo_file}")
    
    with open(demo_file, 'r') as f:
        return json.load(f)


def print_question_info(data: Dict[str, Any]):
    """Print question information."""
    print("\n" + "="*80)
    print(f"QUESTION {data['question_id']}")
    print("="*80)
    print(f"\nQ: {data['question_text']}")
    print(f"\nChoices:")
    for choice in data['choices']:
        print(f"  {choice}")
    print(f"\nGold Answer: {data['gold_answer']}")
    print()


def print_method_summary(method_name: str, method_data: Dict[str, Any]):
    """Print summary of a method's results."""
    if "error" in method_data:
        print(f"\n{method_name.upper()}: ERROR")
        print(f"  {method_data['error']}")
        return
    
    print(f"\n{method_name.upper()}")
    print("-" * 80)
    print(f"Description: {method_data['description']}")
    
    metrics = method_data['final_metrics']
    print(f"\nResults:")
    print(f"  Predicted: {metrics['predicted']}")
    print(f"  Correct: {'✓' if metrics['correct'] else '✗'}")
    print(f"  Confidence: {metrics['confidence']:.4f}")
    print(f"  MI Score: {metrics['mi_score']:.4f}")
    print(f"  Agreement: {metrics['agreement']:.4f}")


def print_method_details(method_name: str, method_data: Dict[str, Any], verbose: bool = False):
    """Print detailed information about a method."""
    if "error" in method_data:
        print(f"\n{method_name.upper()}: ERROR - {method_data['error']}")
        return
    
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} - DETAILED VIEW")
    print(f"{'='*80}")
    print(f"\nDescription: {method_data['description']}")
    
    # Raw inputs
    print(f"\n{'-'*80}")
    print(f"RAW INPUTS ({len(method_data['raw_inputs'])} total)")
    print(f"{'-'*80}")
    for i, inp in enumerate(method_data['raw_inputs'][:3 if not verbose else None]):
        print(f"\nInput {i}:")
        if isinstance(inp, dict):
            for key, value in inp.items():
                if key == "prompt" and isinstance(value, list):
                    print(f"  {key}: [Message with role '{value[0].get('role', 'N/A')}']")
                elif key == "prompt":
                    print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {key}: {value}")
    
    if not verbose and len(method_data['raw_inputs']) > 3:
        print(f"\n... and {len(method_data['raw_inputs']) - 3} more inputs (use --verbose to see all)")
    
    # Raw outputs
    print(f"\n{'-'*80}")
    print(f"RAW OUTPUTS ({len(method_data['raw_outputs'])} total)")
    print(f"{'-'*80}")
    for i, out in enumerate(method_data['raw_outputs'][:3 if not verbose else None]):
        print(f"\nOutput {i}:")
        if isinstance(out, dict):
            for key, value in out.items():
                print(f"  {key}: {value}")
    
    if not verbose and len(method_data['raw_outputs']) > 3:
        print(f"\n... and {len(method_data['raw_outputs']) - 3} more outputs (use --verbose to see all)")
    
    # Decision process
    print(f"\n{'-'*80}")
    print("DECISION PROCESS")
    print(f"{'-'*80}")
    print_dict_recursive(method_data['decision_process'], indent=0, max_depth=3 if not verbose else 10)
    
    # Final metrics
    print(f"\n{'-'*80}")
    print("FINAL METRICS")
    print(f"{'-'*80}")
    for key, value in method_data['final_metrics'].items():
        print(f"  {key}: {value}")


def print_dict_recursive(d: Dict, indent: int = 0, max_depth: int = 5):
    """Recursively print dictionary with indentation."""
    if indent >= max_depth:
        print("  " * indent + "...")
        return
    
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_dict_recursive(value, indent + 1, max_depth)
        elif isinstance(value, list):
            if len(value) <= 5 or indent < 2:
                print("  " * indent + f"{key}: {value}")
            else:
                print("  " * indent + f"{key}: [{len(value)} items]")
        else:
            print("  " * indent + f"{key}: {value}")


def print_comparison_summary(data: Dict[str, Any]):
    """Print comparison summary across all methods."""
    summary = data['comparison_summary']
    
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print("\nPredictions:")
    for method, pred in summary['all_predictions'].items():
        correct = "✓" if summary['all_correct'][method] else "✗"
        conf = summary['all_confidences'][method]
        print(f"  {method:<20} → {pred}  {correct}  (confidence: {conf:.3f})")
    
    print(f"\nMethods that got it correct: {summary['agreement_across_methods']}/{len(summary['all_predictions'])}")
    print(f"Gold answer: {data['gold_answer']}")


def export_to_markdown(demo_dir: str = "demo/outputs", output_file: str = "demo_report.md"):
    """Export all demo data to a markdown report."""
    demo_path = Path(demo_dir)
    
    if not demo_path.exists():
        print(f"Demo directory not found: {demo_dir}")
        return
    
    # Find all demo files
    demo_files = sorted(demo_path.glob("question_*.json"))
    
    if not demo_files:
        print(f"No demo files found in {demo_dir}")
        return
    
    print(f"Exporting {len(demo_files)} questions to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("# LLM Belief MI - Detailed Demo Report\n\n")
        f.write(f"Generated from {len(demo_files)} OpenBookQA questions\n\n")
        f.write("---\n\n")
        
        for demo_file in demo_files:
            with open(demo_file, 'r') as df:
                data = json.load(df)
            
            # Question header
            f.write(f"## Question {data['question_id']}\n\n")
            f.write(f"**Question:** {data['question_text']}\n\n")
            f.write("**Choices:**\n")
            for choice in data['choices']:
                f.write(f"- {choice}\n")
            f.write(f"\n**Gold Answer:** {data['gold_answer']}\n\n")
            
            # Methods summary table
            f.write("### Results Summary\n\n")
            f.write("| Method | Predicted | Correct | Confidence | MI/Entropy |\n")
            f.write("|--------|-----------|---------|------------|------------|\n")
            
            for method_name, method_data in data['methods'].items():
                if "error" in method_data:
                    continue
                metrics = method_data['final_metrics']
                correct_icon = "✓" if metrics['correct'] else "✗"
                f.write(f"| {method_name} | {metrics['predicted']} | {correct_icon} | "
                       f"{metrics['confidence']:.3f} | {metrics['mi_score']:.3f} |\n")
            
            f.write("\n")
            
            # Decision process for each method
            for method_name, method_data in data['methods'].items():
                if "error" in method_data:
                    continue
                    
                f.write(f"### {method_name}\n\n")
                f.write(f"**Description:** {method_data['description']}\n\n")
                
                # Key decision info
                decision = method_data['decision_process']
                if 'confidence_computation' in decision:
                    f.write(f"**Confidence Calculation:** {decision['confidence_computation']}\n\n")
                
                # Method-specific details
                if method_name == "semantic_entropy" and 'aggregated_distribution' in decision:
                    f.write("**Distribution:**\n")
                    for answer, prob in decision['aggregated_distribution'].items():
                        f.write(f"- {answer}: {prob:.3f}\n")
                    f.write(f"\n**Entropy:** {decision.get('entropy_bits', 0):.3f} bits\n\n")
                
                elif method_name == "mi_method" and 'marginal_distribution' in decision:
                    f.write("**Marginal Distribution:**\n")
                    for answer, prob in decision['marginal_distribution'].items():
                        f.write(f"- {answer}: {prob:.3f}\n")
                    f.write(f"\n**MI Score:** {decision['mi_estimation']['mi_bits']:.3f} bits\n\n")
                
                elif method_name == "self_consistency" and 'vote_counts' in decision:
                    f.write("**Vote Counts:**\n")
                    for answer, count in decision['vote_counts'].items():
                        f.write(f"- {answer}: {count}\n")
                    f.write("\n")
            
            f.write("\n---\n\n")
    
    print(f"✓ Report exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="View demo JSON files")
    parser.add_argument("--question", type=int, help="Question ID to view")
    parser.add_argument("--method", type=str, default="all",
                       help="Method to view (all, greedy, self_consistency, semantic_entropy, self_verification, mi_method)")
    parser.add_argument("--verbose", action="store_true", help="Show full details")
    parser.add_argument("--demo-dir", type=str, default="demo/outputs", help="Demo directory")
    parser.add_argument("--export-markdown", type=str, help="Export all demos to markdown file")
    
    args = parser.parse_args()
    
    # Export to markdown
    if args.export_markdown:
        export_to_markdown(args.demo_dir, args.export_markdown)
        return
    
    # View specific question
    if args.question is None:
        print("Error: Must specify --question or --export-markdown")
        parser.print_help()
        return
    
    try:
        data = load_demo(args.question, args.demo_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Print question info
    print_question_info(data)
    
    # Print method details
    if args.method == "all":
        # Summary of all methods
        for method_name, method_data in data['methods'].items():
            print_method_summary(method_name, method_data)
        
        # Comparison summary
        print_comparison_summary(data)
    else:
        # Detailed view of specific method
        if args.method not in data['methods']:
            print(f"Error: Method '{args.method}' not found. Available: {list(data['methods'].keys())}")
            return
        
        print_method_details(args.method, data['methods'][args.method], args.verbose)
    
    print()


if __name__ == "__main__":
    main()

