#!/usr/bin/env python3
"""
Plot comparison between F1-based and NLI-based evaluation frameworks.

Creates side-by-side plots showing:
- Left: F1-based evaluation (Accuracy & ECE)
- Right: NLI-based evaluation (Accuracy & ECE)

Separate plots for SQuAD v2 and TriviaQA.

Usage:
    python scripts/plot_nli_comparison.py --dataset squad_v2
    python scripts/plot_nli_comparison.py --dataset triviaqa
    python scripts/plot_nli_comparison.py --dataset all
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Consistent color scheme for methods
METHOD_COLORS = {
    'Greedy': '#FDBF6F',           # Orange
    'Self-Consistency': '#B2DF8A',  # Light green
    'MI Method': '#CAB2D6',         # Light purple
}

METHOD_ORDER = ['Greedy', 'Self-Consistency', 'MI Method']


def load_json(filepath: str) -> Optional[Dict]:
    """Load a JSON file, return None if not found."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def collect_f1_metrics(dataset: str, methods: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Collect F1-based evaluation metrics for all methods.
    
    Returns: {method_name: {'accuracy': float, 'ece': float}}
    """
    results = {}
    
    for method in methods:
        method_file = method.lower().replace(' ', '').replace('-', '')
        if method == 'Self-Consistency':
            method_file = 'selfcons'
        elif method == 'MI Method':
            method_file = 'mi'
        else:
            method_file = 'greedy'
        
        # Try original results file
        filepath = f"outputs/results/{dataset}/{method_file}_200.json"
        data = load_json(filepath)
        
        if data:
            results[method] = {
                'accuracy': data.get('exact_match', 0.0),
                'ece': data.get('ece', 0.0)
            }
    
    return results


def collect_nli_metrics(dataset: str, methods: List[str], f1_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Collect NLI-based evaluation metrics for all methods.
    
    For Greedy: Uses nli_analysis files (accuracy only, ECE copied from F1-based)
    For MI/SelfCons: Uses nli_adapted files (accuracy + ECE)
    
    Returns: {method_name: {'accuracy': float, 'ece': float}}
    """
    results = {}
    
    for method in methods:
        method_file = method.lower().replace(' ', '').replace('-', '')
        if method == 'Self-Consistency':
            method_file = 'selfcons'
        elif method == 'MI Method':
            method_file = 'mi'
        else:
            method_file = 'greedy'
        
        if method == 'Greedy':
            # Greedy uses nli_analysis files (no ECE recalculated)
            filepath = f"outputs/nli_analysis/{dataset}_{method_file}_200_analysis.json"
            data = load_json(filepath)
            
            if data and 'summary' in data:
                results[method] = {
                    'accuracy': data['summary'].get('nli_accuracy', 0.0),
                    'ece': f1_metrics.get(method, {}).get('ece', 0.0)  # Copy from F1-based
                }
        else:
            # MI and SelfCons use nli_adapted files
            filepath = f"outputs/nli_adapted/{dataset}_{method_file}_200.json"
            data = load_json(filepath)
            
            if data and 'summary' in data:
                nli_metrics = data['summary'].get('nli_adapted_metrics', {})
                results[method] = {
                    'accuracy': nli_metrics.get('exact_match', 0.0),
                    'ece': nli_metrics.get('ece', 0.0)
                }
    
    return results


def plot_comparison(
    f1_metrics: Dict[str, Dict[str, float]],
    nli_metrics: Dict[str, Dict[str, float]],
    dataset: str,
    output_path: str
):
    """
    Create side-by-side comparison plot: F1 evaluation vs NLI evaluation.
    
    Args:
        f1_metrics: {method: {'accuracy': float, 'ece': float}}
        nli_metrics: {method: {'accuracy': float, 'ece': float or None}}
        dataset: Dataset name
        output_path: Where to save the plot
    """
    # Filter to methods that have both F1 and NLI data
    methods = [m for m in METHOD_ORDER if m in f1_metrics and m in nli_metrics]
    
    if not methods:
        print(f"⚠ No complete data found for {dataset}")
        return
    
    # Prepare data
    f1_accs = [f1_metrics[m]['accuracy'] * 100 for m in methods]
    f1_eces = [f1_metrics[m]['ece'] for m in methods]
    
    nli_accs = [nli_metrics[m]['accuracy'] * 100 for m in methods]
    nli_eces = [nli_metrics[m]['ece'] for m in methods]
    
    # Check which methods have different NLI ECE (not just copied from F1)
    has_different_ece = [nli_metrics[m]['ece'] != f1_metrics[m]['ece'] for m in methods]
    
    colors = [METHOD_COLORS[m] for m in methods]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    y_pos = np.arange(len(methods))
    bar_width = 0.6
    
    # ==================== LEFT COLUMN: F1-based Evaluation ====================
    
    # Top-left: F1 Accuracy
    ax_f1_acc = axes[0, 0]
    bars_f1_acc = ax_f1_acc.barh(y_pos, f1_accs, bar_width, 
                                  color=colors, edgecolor='black', linewidth=1.5)
    ax_f1_acc.set_yticks(y_pos)
    ax_f1_acc.set_yticklabels(methods, fontsize=11)
    ax_f1_acc.set_xlabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax_f1_acc.set_title('F1-Based: Accuracy', fontsize=14, fontweight='bold')
    ax_f1_acc.set_xlim([0, 100])
    ax_f1_acc.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars_f1_acc, f1_accs)):
        width = bar.get_width()
        ax_f1_acc.text(width + 2, y_pos[i], f'{acc:.1f}%', 
                       ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Bottom-left: F1 ECE
    ax_f1_ece = axes[1, 0]
    bars_f1_ece = ax_f1_ece.barh(y_pos, f1_eces, bar_width,
                                  color=colors, edgecolor='black', linewidth=1.5)
    ax_f1_ece.set_yticks(y_pos)
    ax_f1_ece.set_yticklabels(methods, fontsize=11)
    ax_f1_ece.set_xlabel('ECE (Lower is Better)', fontsize=12, fontweight='bold')
    ax_f1_ece.set_title('F1-Based: Calibration Error', fontsize=14, fontweight='bold')
    ax_f1_ece.set_xlim([0, max(f1_eces) * 1.15])
    ax_f1_ece.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Highlight best ECE
    best_f1_idx = np.argmin(f1_eces)
    bars_f1_ece[best_f1_idx].set_edgecolor('green')
    bars_f1_ece[best_f1_idx].set_linewidth(3)
    
    # Add value labels
    for i, (bar, ece) in enumerate(zip(bars_f1_ece, f1_eces)):
        width = bar.get_width()
        ax_f1_ece.text(width + max(f1_eces) * 0.03, y_pos[i], f'{ece:.3f}',
                       ha='left', va='center', fontsize=10, fontweight='bold')
    
    # ==================== RIGHT COLUMN: NLI-based Evaluation ====================
    
    # Top-right: NLI Accuracy
    ax_nli_acc = axes[0, 1]
    bars_nli_acc = ax_nli_acc.barh(y_pos, nli_accs, bar_width,
                                    color=colors, edgecolor='black', linewidth=1.5)
    ax_nli_acc.set_yticks(y_pos)
    ax_nli_acc.set_yticklabels(methods, fontsize=11)
    ax_nli_acc.set_xlabel('NLI-Based Accuracy (%)', fontsize=12, fontweight='bold')
    ax_nli_acc.set_title('NLI-Based: Accuracy', fontsize=14, fontweight='bold')
    ax_nli_acc.set_xlim([0, 100])
    ax_nli_acc.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels and improvement markers
    for i, (bar, acc, f1_acc) in enumerate(zip(bars_nli_acc, nli_accs, f1_accs)):
        width = bar.get_width()
        improvement = acc - f1_acc
        color = 'green' if improvement > 0 else 'red' if improvement < 0 else 'gray'
        sign = '+' if improvement > 0 else ''
        ax_nli_acc.text(width + 2, y_pos[i], 
                        f'{acc:.1f}% ({sign}{improvement:.1f})',
                        ha='left', va='center', fontsize=10, fontweight='bold', color=color)
    
    # Bottom-right: NLI ECE
    ax_nli_ece = axes[1, 1]
    
    # Plot all bars (Greedy will show F1 ECE, others show recalculated)
    bars_nli_ece = ax_nli_ece.barh(y_pos, nli_eces, bar_width,
                                    color=colors, edgecolor='black', linewidth=1.5)
    
    # Highlight best ECE (among recalculated ones)
    recalculated_eces = [nli_eces[i] for i, diff in enumerate(has_different_ece) if diff]
    if recalculated_eces:
        best_nli_idx = np.argmin(nli_eces)
        bars_nli_ece[best_nli_idx].set_edgecolor('green')
        bars_nli_ece[best_nli_idx].set_linewidth(3)
    
    # Add value labels and improvement markers
    for i, (bar, ece, method) in enumerate(zip(bars_nli_ece, nli_eces, methods)):
        width = bar.get_width()
        
        if has_different_ece[i]:
            # Recalculated ECE - show improvement
            improvement = f1_eces[i] - ece  # Positive = better (lower ECE)
            color = 'green' if improvement > 0 else 'red' if improvement < 0 else 'gray'
            sign = '+' if improvement > 0 else ''
            ax_nli_ece.text(width + max(nli_eces) * 0.03, y_pos[i],
                           f'{ece:.3f} ({sign}{improvement:.3f})',
                           ha='left', va='center', fontsize=10, fontweight='bold', color=color)
        else:
            # Copied from F1 (Greedy) - mark as unchanged
            ax_nli_ece.text(width + max(nli_eces) * 0.03, y_pos[i],
                           f'{ece:.3f} (=)',
                           ha='left', va='center', fontsize=10, fontweight='bold', color='gray')
    
    ax_nli_ece.set_yticks(y_pos)
    ax_nli_ece.set_yticklabels(methods, fontsize=11)
    ax_nli_ece.set_xlabel('ECE (Lower is Better)', fontsize=12, fontweight='bold')
    ax_nli_ece.set_title('NLI-Based: Calibration Error', fontsize=14, fontweight='bold')
    ax_nli_ece.set_xlim([0, max(max(f1_eces), max(nli_eces)) * 1.15])  # Use same scale as F1 for comparison
    ax_nli_ece.grid(axis='x', alpha=0.3, linestyle='--')
    
    # ==================== Overall Formatting ====================
    
    # Dataset title
    dataset_names = {
        'squad_v2': 'SQuAD v2',
        'triviaqa': 'TriviaQA'
    }
    fig.suptitle(f'{dataset_names.get(dataset, dataset)} - F1 vs NLI Evaluation Comparison (NLI Threshold: 0.5)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add legends and notes
    fig.text(0.5, 0.045, 
             'Green border = Best ECE | Green/Red numbers = Improvement/Degradation vs F1-based | (=) = Same as F1',
             ha='center', fontsize=10, style='italic')
    fig.text(0.5, 0.02, 
             'Note: Greedy NLI-ECE shows F1-based ECE (single-answer, no distribution to recalculate)',
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_dataset(dataset: str, output_dir: str = "outputs/plots"):
    """Plot comparison for a single dataset."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessing {dataset}...")
    
    methods = ['Greedy', 'Self-Consistency', 'MI Method']
    
    # Collect metrics
    f1_metrics = collect_f1_metrics(dataset, methods)
    nli_metrics = collect_nli_metrics(dataset, methods, f1_metrics)
    
    if not f1_metrics or not nli_metrics:
        print(f"⚠ Incomplete data for {dataset}")
        return
    
    print(f"  F1 metrics: {list(f1_metrics.keys())}")
    print(f"  NLI metrics: {list(nli_metrics.keys())}")
    
    # Create plot
    output_path = f"{output_dir}/{dataset}_nli_comparison.png"
    plot_comparison(f1_metrics, nli_metrics, dataset, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot F1 vs NLI evaluation comparison"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['all', 'squad_v2', 'triviaqa'],
        default='all',
        help="Dataset to plot"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/plots",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("F1 vs NLI EVALUATION COMPARISON PLOTS")
    print("="*80)
    
    datasets = ['squad_v2', 'triviaqa'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        plot_dataset(dataset, args.output_dir)
    
    print()
    print("="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()

