#!/usr/bin/env python3
"""
Print a summary table of all results.

Usage:
    python scripts/summarize_results.py
    python scripts/summarize_results.py --dataset openbookqa
"""

import json
import argparse
from pathlib import Path
from glob import glob
from typing import Dict


def extract_info(filepath: str) -> tuple:
    """Extract dataset and method from filepath."""
    path = Path(filepath)
    stem = path.stem
    
    # For new folder structure: outputs/results/dataset/method_200.json
    # Dataset is the parent folder name
    dataset = None
    parent_name = path.parent.name
    
    # Check if parent is a dataset folder
    dataset_list = ['openbookqa', 'arc_challenge', 'arc_easy', 
                   'truthfulqa', 'truthfulqa_mc2', 'squad_v2', 'triviaqa']
    
    if parent_name in dataset_list:
        dataset = parent_name
        method = stem.replace('_200', '').replace('_500', '')
    else:
        # Fall back to old flat structure: try to find dataset in filename
        for ds in dataset_list:
            if ds in stem:
                dataset = ds
                break
        
        # Extract method
        if dataset:
            method = stem.replace(dataset + '_', '').replace('_200', '').replace('_500', '')
        else:
            method = stem
    
    # Pretty names
    method_map = {
        'greedy': 'Greedy',
        'selfcons': 'Self-Consistency',
        'semantic_entropy': 'Semantic Entropy',
        'self_verification': 'Self-Verification',
        'mi': 'MI Method'
    }
    
    dataset_map = {
        'openbookqa': 'OpenBookQA',
        'arc_challenge': 'ARC-Challenge',
        'arc_easy': 'ARC-Easy',
        'truthfulqa': 'TruthfulQA MC1',
        'truthfulqa_mc2': 'TruthfulQA MC2',
        'squad_v2': 'SQuAD v2',
        'triviaqa': 'TriviaQA'
    }
    
    return (
        dataset_map.get(dataset, dataset) if dataset else None,
        method_map.get(method, method.replace('_', ' ').title())
    )


def load_all_results(pattern: str = "outputs/results/*/*_500.json") -> Dict:
    """Load all result files."""
    files = sorted(glob(pattern))
    
    results = {}
    for filepath in files:
        dataset, method = extract_info(filepath)
        
        # Skip if dataset couldn't be extracted
        if dataset is None:
            continue
        
        if dataset not in results:
            results[dataset] = {}
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            results[dataset][method] = data
    
    return results


def print_summary_table(results: Dict, dataset_filter: str = None):
    """Print formatted summary table."""
    
    # Filter by dataset if specified
    if dataset_filter:
        dataset_map = {
            'openbookqa': 'OpenBookQA',
            'arc_challenge': 'ARC-Challenge',
            'arc_easy': 'ARC-Easy',
            'truthfulqa': 'TruthfulQA MC1',
            'truthfulqa_mc2': 'TruthfulQA MC2',
            'squad_v2': 'SQuAD v2',
            'triviaqa': 'TriviaQA'
        }
        pretty_name = dataset_map.get(dataset_filter, dataset_filter)
        if pretty_name in results:
            results = {pretty_name: results[pretty_name]}
        else:
            print(f"No results found for {dataset_filter}")
            return
    
    print("\n" + "="*100)
    print("EVALUATION RESULTS SUMMARY")
    print("="*100)
    
    for dataset, methods in sorted(results.items()):
        print(f"\n{dataset}")
        print("-"*100)
        
        # Determine metric type (MCQ vs open-ended)
        first_method_data = list(methods.values())[0]
        is_open_ended = 'exact_match' in first_method_data
        
        if is_open_ended:
            print(f"{'Method':<25} {'EM':<10} {'F1':<10} {'ECE':<10} {'Conf':<10} {'MI':<10} {'N':<8}")
        else:
            print(f"{'Method':<25} {'Accuracy':<12} {'ECE':<12} {'Avg Conf':<12} {'MI':<12} {'N':<8}")
        print("-"*100)
        
        # Sort by ECE (lower is better)
        sorted_methods = sorted(methods.items(), key=lambda x: x[1]['ece'])
        
        for i, (method, data) in enumerate(sorted_methods):
            ece = data['ece']
            conf = data.get('avg_confidence', 0)
            mi = data.get('avg_mi_bits', 0)
            n = data.get('n_samples', 0)
            
            # Highlight best ECE
            marker = " ⭐" if i == 0 else "   "
            
            if is_open_ended:
                em = data.get('exact_match', 0) * 100
                f1 = data.get('f1', 0) * 100
                print(f"{method:<25} {em:<10.2f} {f1:<10.2f} {ece:<10.4f} {conf:<10.4f} {mi:<10.4f} {n:<8}{marker}")
            else:
                acc = data['accuracy'] * 100
                print(f"{method:<25} {acc:<12.2f} {ece:<12.4f} {conf:<12.4f} {mi:<12.4f} {n:<8}{marker}")
        
        # Calculate improvements
        if len(sorted_methods) > 1:
            best_ece = sorted_methods[0][1]['ece']
            worst_ece = sorted_methods[-1][1]['ece']
            improvement = ((worst_ece - best_ece) / worst_ece) * 100
            print(f"\nBest method ({sorted_methods[0][0]}) improves ECE by {improvement:.1f}% vs worst")
    
    print("\n" + "="*100)
    print("⭐ = Best ECE (Expected Calibration Error) for this dataset")
    print("="*100)
    print()


def print_cross_dataset_comparison(results: Dict):
    """Print comparison across datasets for each method."""
    
    # Collect all methods
    all_methods = set()
    for methods in results.values():
        all_methods.update(methods.keys())
    
    print("\n" + "="*100)
    print("CROSS-DATASET COMPARISON BY METHOD")
    print("="*100)
    
    for method in sorted(all_methods):
        print(f"\n{method}")
        print("-"*100)
        print(f"{'Dataset':<20} {'Acc/EM':<15} {'ECE':<15} {'Avg Confidence':<15}")
        print("-"*100)
        
        for dataset in sorted(results.keys()):
            if method in results[dataset]:
                data = results[dataset][method]
                # Handle both accuracy and exact_match
                if 'accuracy' in data:
                    acc = data['accuracy'] * 100
                elif 'exact_match' in data:
                    acc = data['exact_match'] * 100
                else:
                    acc = 0.0
                
                ece = data['ece']
                conf = data.get('avg_confidence', 0)
                
                print(f"{dataset:<20} {acc:<15.2f} {ece:<15.4f} {conf:<15.4f}")
            else:
                print(f"{dataset:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print("\n" + "="*100)
    print()


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("--dataset", type=str, 
                       choices=['openbookqa', 'arc_challenge', 'arc_easy',
                               'truthfulqa', 'truthfulqa_mc2', 'squad_v2', 'triviaqa'],
                       help="Filter by specific dataset")
    parser.add_argument("--pattern", type=str, default="outputs/results/*/*.json",
                       help="File pattern to match")
    parser.add_argument("--cross-dataset", action="store_true",
                       help="Show cross-dataset comparison by method")
    
    args = parser.parse_args()
    
    results = load_all_results(args.pattern)
    
    if not results:
        print("No results found!")
        print(f"Pattern: {args.pattern}")
        return
    
    print_summary_table(results, args.dataset)
    
    if args.cross_dataset and not args.dataset:
        print_cross_dataset_comparison(results)


if __name__ == "__main__":
    main()

