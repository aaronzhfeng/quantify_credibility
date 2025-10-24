#!/usr/bin/env python3
"""
Plot accuracy and ECE results from evaluation runs.

Usage:
    python scripts/plot_results.py --dataset all
    python scripts/plot_results.py --dataset openbookqa
    python scripts/plot_results.py --custom outputs/results/openbookqa_*_500.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_result(filepath: str) -> Dict:
    """Load a single result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_method_name(filepath: str) -> str:
    """Extract method name from filepath."""
    stem = Path(filepath).stem
    
    # Remove dataset prefix and _500 suffix
    for dataset in ['openbookqa', 'arc_challenge', 'arc_easy']:
        if dataset in stem:
            name = stem.replace(dataset + '_', '').replace('_500', '')
            # Pretty names
            name_map = {
                'greedy': 'Greedy',
                'selfcons': 'Self-Consistency',
                'semantic_entropy': 'Semantic Entropy',
                'self_verification': 'Self-Verification',
                'mi': 'MI Method'
            }
            return name_map.get(name, name.replace('_', ' ').title())
    
    return stem


def collect_results(pattern: str) -> Dict[str, Dict[str, float]]:
    """
    Collect results from files matching pattern.
    
    Returns dict: {method_name: {accuracy: float, ece: float, ...}}
    """
    from glob import glob
    
    files = glob(pattern)
    results = {}
    
    for filepath in files:
        method = extract_method_name(filepath)
        data = load_result(filepath)
        results[method] = data
    
    return results


def plot_comparison(
    results: Dict[str, Dict[str, float]],
    title: str,
    output_path: str
):
    """
    Plot accuracy and ECE comparison for multiple methods.
    
    Args:
        results: {method_name: {accuracy: float, ece: float, ...}}
        title: Plot title
        output_path: Where to save the plot
    """
    if not results:
        print(f"No results to plot for {title}")
        return
    
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] * 100 for m in methods]  # Convert to percentage
    eces = [results[m]['ece'] for m in methods]
    
    # Sort by ECE (ascending - lower is better)
    sorted_indices = np.argsort(eces)
    methods = [methods[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    eces = [eces[i] for i in sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # Plot 1: Accuracy
    bars1 = ax1.barh(methods, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 100])
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', ha='left', va='center', fontsize=10)
    
    # Plot 2: ECE (lower is better)
    bars2 = ax2.barh(methods, eces, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('ECE (Expected Calibration Error)', fontsize=12, fontweight='bold')
    ax2.set_title('ECE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, max(eces) * 1.1])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, ece in zip(bars2, eces):
        width = bar.get_width()
        ax2.text(width + max(eces) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{ece:.3f}', ha='left', va='center', fontsize=10)
    
    # Highlight best method (lowest ECE)
    best_idx = 0  # Already sorted by ECE
    bars2[best_idx].set_edgecolor('green')
    bars2[best_idx].set_linewidth(3)
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Add note about sample size
    n_samples = results[methods[0]].get('n_samples', 'N/A')
    fig.text(0.5, 0.02, f'N = {n_samples} examples | Green border = Best ECE', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_all_datasets(
    datasets: List[str],
    output_dir: str = "outputs/plots"
):
    """Plot comparisons for all datasets."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    dataset_names = {
        'openbookqa': 'OpenBookQA',
        'arc_challenge': 'ARC-Challenge',
        'arc_easy': 'ARC-Easy'
    }
    
    all_results = {}
    
    for dataset in datasets:
        pattern = f"outputs/results/{dataset}_*_500.json"
        results = collect_results(pattern)
        
        if results:
            all_results[dataset] = results
            title = f"{dataset_names.get(dataset, dataset)} - Method Comparison (500 examples)"
            output_path = f"{output_dir}/{dataset}_comparison.png"
            plot_comparison(results, title, output_path)
        else:
            print(f"⚠ No results found for {dataset}")
    
    # Create combined comparison across datasets
    if len(all_results) > 1:
        create_combined_plot(all_results, dataset_names, output_dir)


def create_combined_plot(
    all_results: Dict[str, Dict[str, Dict]],
    dataset_names: Dict[str, str],
    output_dir: str
):
    """Create a combined plot showing all datasets and methods."""
    
    # Collect all unique methods
    all_methods = set()
    for results in all_results.values():
        all_methods.update(results.keys())
    all_methods = sorted(all_methods)
    
    # Prepare data
    datasets = list(all_results.keys())
    n_methods = len(all_methods)
    n_datasets = len(datasets)
    
    # Create figure
    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
    method_colors = {method: colors[i] for i, method in enumerate(all_methods)}
    
    for col, dataset in enumerate(datasets):
        results = all_results[dataset]
        
        # Get methods present in this dataset
        methods = [m for m in all_methods if m in results]
        accuracies = [results[m]['accuracy'] * 100 for m in methods]
        eces = [results[m]['ece'] for m in methods]
        bar_colors = [method_colors[m] for m in methods]
        
        # Accuracy plot
        ax_acc = axes[0, col]
        bars = ax_acc.bar(range(len(methods)), accuracies, color=bar_colors, 
                          edgecolor='black', linewidth=1.2)
        ax_acc.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax_acc.set_title(dataset_names.get(dataset, dataset), fontsize=12, fontweight='bold')
        ax_acc.set_xticks(range(len(methods)))
        ax_acc.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax_acc.set_ylim([0, 100])
        ax_acc.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax_acc.text(bar.get_x() + bar.get_width()/2, height + 1,
                       f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # ECE plot
        ax_ece = axes[1, col]
        bars = ax_ece.bar(range(len(methods)), eces, color=bar_colors,
                         edgecolor='black', linewidth=1.2)
        ax_ece.set_ylabel('ECE', fontsize=11, fontweight='bold')
        ax_ece.set_xlabel('Method', fontsize=10)
        ax_ece.set_xticks(range(len(methods)))
        ax_ece.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax_ece.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best ECE
        best_idx = np.argmin(eces)
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
        
        # Add value labels
        for bar, ece in zip(bars, eces):
            height = bar.get_height()
            ax_ece.text(bar.get_x() + bar.get_width()/2, height + max(eces) * 0.02,
                       f'{ece:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle('Method Comparison Across All Datasets (500 examples each)', 
                 fontsize=16, fontweight='bold')
    
    # Add legend
    legend_patches = [mpatches.Patch(color=method_colors[m], label=m) for m in all_methods]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(5, len(all_methods)),
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path = f"{output_dir}/combined_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_custom_files(files: List[str], output_dir: str = "outputs/plots"):
    """Plot comparison for custom list of files."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    for filepath in files:
        if Path(filepath).exists():
            method = extract_method_name(filepath)
            results[method] = load_result(filepath)
    
    if results:
        title = "Method Comparison"
        output_path = f"{output_dir}/custom_comparison.png"
        plot_comparison(results, title, output_path)
    else:
        print("No valid result files found")


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--dataset", type=str, choices=['all', 'openbookqa', 'arc_challenge', 'arc_easy'],
                       default='all', help="Dataset to plot")
    parser.add_argument("--custom", nargs='+', help="Custom list of JSON files to plot")
    parser.add_argument("--output-dir", type=str, default="outputs/plots",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PLOTTING EVALUATION RESULTS")
    print("="*80)
    print()
    
    if args.custom:
        print(f"Plotting custom files: {args.custom}")
        plot_custom_files(args.custom, args.output_dir)
    else:
        if args.dataset == 'all':
            datasets = ['openbookqa', 'arc_challenge', 'arc_easy']
        else:
            datasets = [args.dataset]
        
        print(f"Plotting datasets: {', '.join(datasets)}")
        print(f"Output directory: {args.output_dir}")
        print()
        
        plot_all_datasets(datasets, args.output_dir)
    
    print()
    print("="*80)
    print("PLOTTING COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()

