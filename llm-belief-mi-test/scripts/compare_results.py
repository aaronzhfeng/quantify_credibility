#!/usr/bin/env python3
"""
Compare results from different evaluation methods (greedy, self-consistency, MI).

Usage:
    python compare_results.py outputs/results/*_50.json
    python compare_results.py outputs/results/arc_challenge_*.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def load_results(file_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_method_name(file_path: str) -> str:
    """Extract method name from file path."""
    name = Path(file_path).stem
    
    if 'greedy' in name:
        return 'Greedy'
    elif 'selfcons' in name or 'self_consistency' in name:
        return 'Self-Consistency'
    elif 'mi' in name or 'method' in name:
        return 'MI Method'
    else:
        return name

def compare_results(file_paths: List[str]):
    """Compare results from multiple JSON files."""
    
    if not file_paths:
        print("Usage: python compare_results.py <json_file1> <json_file2> ...")
        print("Example: python compare_results.py outputs/results/*_50.json")
        return
    
    # Load all results
    results = []
    for file_path in file_paths:
        try:
            data = load_results(file_path)
            method = extract_method_name(file_path)
            results.append((method, file_path, data))
        except Exception as e:
            print(f"Error loading {file_path}: {e}", file=sys.stderr)
    
    if not results:
        print("No valid result files found.")
        return
    
    # Print comparison table
    print("\n" + "="*100)
    print("BASELINE COMPARISON")
    print("="*100)
    print(f"{'Method':<25} {'Accuracy':<12} {'ECE':<12} {'Avg Conf':<12} {'Avg MI':<12} {'Samples':<10}")
    print("-"*100)
    
    # Sort by ECE (lower is better)
    results.sort(key=lambda x: x[2]['ece'])
    
    best_ece = results[0][2]['ece']
    
    for method, file_path, data in results:
        accuracy = data.get('accuracy', 0.0)
        ece = data.get('ece', 0.0)
        avg_conf = data.get('avg_confidence', 0.0)
        avg_mi = data.get('avg_mi_bits', 0.0)
        n_samples = data.get('n_samples', 0)
        
        # Mark best ECE with ⭐
        marker = " ⭐ BEST" if abs(ece - best_ece) < 0.001 else ""
        
        print(f"{method:<25} {accuracy:<12.4f} {ece:<12.4f} {avg_conf:<12.4f} {avg_mi:<12.4f} {n_samples:<10}{marker}")
    
    print("="*100)
    
    # Print analysis
    print("\n📊 ANALYSIS:")
    print("-" * 100)
    
    # Accuracy range
    accuracies = [r[2]['accuracy'] for r in results]
    acc_range = max(accuracies) - min(accuracies)
    print(f"Accuracy range: {min(accuracies):.4f} to {max(accuracies):.4f} (spread: {acc_range:.4f})")
    if acc_range < 0.05:
        print("  ✅ Similar accuracy across methods (as expected)")
    else:
        print("  ⚠️  Large accuracy variation - may need more samples")
    
    # ECE comparison
    eces = [r[2]['ece'] for r in results]
    best_method = results[0][0]
    worst_ece = max(eces)
    improvement = ((worst_ece - best_ece) / worst_ece) * 100
    
    print(f"\nECE (Expected Calibration Error) - Lower is better:")
    print(f"  Best: {best_method} with ECE = {best_ece:.4f}")
    print(f"  Improvement over worst: {improvement:.1f}%")
    
    if 'MI' in best_method:
        print("  ✅ MI method has best calibration (key paper result!)")
    else:
        print(f"  ⚠️  {best_method} has best ECE - may need more samples or check implementation")
    
    # Confidence analysis
    print(f"\nConfidence scores:")
    for method, _, data in results:
        avg_conf = data.get('avg_confidence', 0.0)
        print(f"  {method:<25} Avg confidence: {avg_conf:.4f}")
    
    print("\n" + "="*100)
    
    # Print file paths for reference
    print("\nFiles analyzed:")
    for method, file_path, _ in results:
        print(f"  - {file_path}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_results.py <json_file1> <json_file2> ...")
        print("\nExamples:")
        print("  python compare_results.py outputs/results/*_50.json")
        print("  python compare_results.py outputs/results/arc_challenge_*.json")
        print("  python compare_results.py outputs/results/baseline_greedy_50.json outputs/results/mi_method_50.json")
        sys.exit(1)
    
    compare_results(sys.argv[1:])

