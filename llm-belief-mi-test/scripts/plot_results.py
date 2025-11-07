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


# Consistent color scheme for methods across all plots
METHOD_COLORS = {
    'Greedy': '#FDBF6F',           # Orange
    'Self-Consistency': '#B2DF8A',  # Light green
    'MI Method': '#CAB2D6',         # Light purple
    'Semantic Entropy': '#FB9A99',  # Light red
    'Self-Verification': '#A6CEE3'  # Light blue
}


def load_result(filepath: str) -> Dict:
    """Load a single result JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_method_name(filepath: str) -> str:
    """Extract a human-friendly method name from the JSON filepath."""
    path = Path(filepath)
    stem = path.stem
    parent = path.parent.name
    
    # List of all dataset folder names
    dataset_names = {
        'openbookqa', 'arc_challenge', 'arc_easy',
        'truthfulqa', 'truthfulqa_mc2', 'squad_v2', 'triviaqa'
    }
    
    # Determine the raw method name
    method = stem
    
    # Check if stem contains dataset prefix (old structure inside dataset folder)
    # e.g., openbookqa/openbookqa_greedy_500.json
    has_dataset_prefix = False
    for dataset in dataset_names:
        if stem.startswith(f"{dataset}_"):
            method = stem.replace(f"{dataset}_", "")
            has_dataset_prefix = True
            break
    
    # If no dataset prefix found and parent is a dataset folder, it's new structure
    # e.g., truthfulqa/greedy_200.json
    if not has_dataset_prefix and parent in dataset_names:
        method = stem
    
    # Remove _200 and _500 suffixes
    method = method.replace('_200', '').replace('_500', '')
    
    # Map to pretty names
    name_map = {
        'greedy': 'Greedy',
        'selfcons': 'Self-Consistency',
        'semantic_entropy': 'Semantic Entropy',
        'self_verification': 'Self-Verification',
        'mi': 'MI Method'
    }
    
    return name_map.get(method, method.replace('_', ' ').title())


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
    Plot performance metrics and ECE comparison for multiple methods.
    
    For MCQ datasets: Shows Accuracy + ECE
    For open-ended datasets: Shows (Exact Match + F1) + ECE
    
    Args:
        results: {method_name: {accuracy/exact_match/f1: float, ece: float, ...}}
        title: Plot title
        output_path: Where to save the plot
    """
    if not results:
        print(f"No results to plot for {title}")
        return
    
    methods = list(results.keys())
    
    # Detect dataset type
    is_open_ended = 'exact_match' in list(results.values())[0]
    
    # Extract ECE for sorting
    eces = [results[m]['ece'] for m in methods]
    
    # Sort by ECE (ascending - lower is better)
    sorted_indices = np.argsort(eces)
    methods = [methods[i] for i in sorted_indices]
    eces = [eces[i] for i in sorted_indices]
    
    # Get method colors
    method_colors = [METHOD_COLORS.get(m, '#CCCCCC') for m in methods]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if is_open_ended:
        # Open-ended: Show EM + F1 as grouped bars
        em_scores = [results[methods[i]]['exact_match'] * 100 for i in range(len(methods))]
        f1_scores = [results[methods[i]]['f1'] * 100 for i in range(len(methods))]
        
        y_pos = np.arange(len(methods))
        bar_height = 0.35
        
        # Plot EM and F1 as grouped horizontal bars
        bars1_em = ax1.barh(y_pos - bar_height/2, em_scores, bar_height, 
                           label='Exact Match', color='#8DD3C7', edgecolor='black', linewidth=1.2)
        bars1_f1 = ax1.barh(y_pos + bar_height/2, f1_scores, bar_height,
                           label='F1 Score', color='#BEBADA', edgecolor='black', linewidth=1.2)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(methods)
        ax1.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xlim([0, 100])
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, (em, f1) in enumerate(zip(em_scores, f1_scores)):
            ax1.text(em + 1, y_pos[i] - bar_height/2, f'{em:.1f}%', 
                    ha='left', va='center', fontsize=9)
            ax1.text(f1 + 1, y_pos[i] + bar_height/2, f'{f1:.1f}%',
                    ha='left', va='center', fontsize=9)
        
    else:
        # MCQ: Show Accuracy only
        accuracies = [results[m]['accuracy'] * 100 for m in methods]
        
        bars1 = ax1.barh(methods, accuracies, color=method_colors, 
                        edgecolor='black', linewidth=1.2)
        ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim([0, 100])
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{acc:.1f}%', ha='left', va='center', fontsize=10)
    
    # Plot 2: ECE (same for both types)
    bars2 = ax2.barh(methods, eces, color=method_colors, edgecolor='black', linewidth=1.2)
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
        'arc_easy': 'ARC-Easy',
        'truthfulqa': 'TruthfulQA MC1',
        'truthfulqa_mc2': 'TruthfulQA MC2',
        'squad_v2': 'SQuAD v2',
        'triviaqa': 'TriviaQA'
    }
    
    all_results = {}
    
    for dataset in datasets:
        # Try new folder structure first, fall back to old flat structure
        pattern = f"outputs/results/{dataset}/*.json"
        results = collect_results(pattern)
        
        if not results:
            # Try old flat structure
            pattern = f"outputs/results/{dataset}_*.json"
            results = collect_results(pattern)
        
        if results:
            all_results[dataset] = results
            # Determine example count from first result
            n_examples = list(results.values())[0].get('n_samples', '?')
            title = f"{dataset_names.get(dataset, dataset)} - Method Comparison ({int(n_examples)} examples)"
            output_path = f"{output_dir}/{dataset}_comparison.png"
            plot_comparison(results, title, output_path)
        else:
            print(f"⚠ No results found for {dataset}")
    
    # Create combined comparison across datasets
    if len(all_results) > 1:
        create_combined_plot(all_results, dataset_names, output_dir)
        # Also create MCQ-only and open-ended-only plots
        create_mcq_comparison_plot(all_results, dataset_names, output_dir)
        create_openended_comparison_plot(all_results, dataset_names, output_dir)


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
    
    # Use consistent method colors
    method_color_map = {method: METHOD_COLORS.get(method, '#CCCCCC') for method in all_methods}
    
    for col, dataset in enumerate(datasets):
        results = all_results[dataset]
        
        # ONLY use methods that actually exist for THIS dataset
        methods = list(results.keys())
        
        # Extract data for methods that exist
        accuracies = []
        for m in methods:
            if 'accuracy' in results[m]:
                accuracies.append(results[m]['accuracy'] * 100)
            elif 'exact_match' in results[m]:
                accuracies.append(results[m]['exact_match'] * 100)
            else:
                accuracies.append(0.0)
        
        eces = [results[m]['ece'] for m in methods]
        
        # Map consistent colors based on method names
        bar_colors = [method_color_map.get(m, '#CCCCCC') for m in methods]
        
        # Accuracy plot (use universal label for all datasets)
        ax_acc = axes[0, col]
        bars = ax_acc.bar(range(len(methods)), accuracies, color=bar_colors, 
                          edgecolor='black', linewidth=1.2)
        
        # Use "Accuracy" as universal label for clean cross-dataset comparison
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
        bars_ece = ax_ece.bar(range(len(methods)), eces, color=bar_colors,
                              edgecolor='black', linewidth=1.2)
        ax_ece.set_ylabel('ECE', fontsize=11, fontweight='bold')
        ax_ece.set_xlabel('Method', fontsize=10)
        ax_ece.set_xticks(range(len(methods)))
        ax_ece.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax_ece.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best ECE
        best_idx = np.argmin(eces)
        bars_ece[best_idx].set_edgecolor('green')
        bars_ece[best_idx].set_linewidth(3)
        
        # Add value labels
        max_ece = max(eces) if eces else 1.0
        for bar, ece in zip(bars_ece, eces):
            height = bar.get_height()
            ax_ece.text(bar.get_x() + bar.get_width()/2, height + max_ece * 0.02,
                       f'{ece:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle('Method Comparison Across All Datasets', 
                 fontsize=16, fontweight='bold')
    
    # Add legend with consistent colors
    legend_patches = [mpatches.Patch(color=method_color_map[m], label=m) for m in all_methods]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(5, len(all_methods)),
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path = f"{output_dir}/combined_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_mcq_comparison_plot(
    all_results: Dict[str, Dict[str, Dict]],
    dataset_names: Dict[str, str],
    output_dir: str
):
    """Create a combined plot for MCQ datasets only."""
    
    # Filter for MCQ datasets
    mcq_datasets = ['openbookqa', 'arc_challenge', 'arc_easy', 'truthfulqa', 'truthfulqa_mc2']
    mcq_results = {k: v for k, v in all_results.items() if k in mcq_datasets}
    
    if not mcq_results:
        print("⚠ No MCQ results found")
        return
    
    # Collect all unique methods
    all_methods = set()
    for results in mcq_results.values():
        all_methods.update(results.keys())
    all_methods = sorted(all_methods)
    
    # Prepare data
    datasets = list(mcq_results.keys())
    n_datasets = len(datasets)
    
    # Create figure
    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    # Use consistent method colors
    method_color_map = {method: METHOD_COLORS.get(method, '#CCCCCC') for method in all_methods}
    
    for col, dataset in enumerate(datasets):
        results = mcq_results[dataset]
        
        # ONLY use methods that actually exist for THIS dataset
        methods = list(results.keys())
        
        # Extract data for methods that exist
        accuracies = [results[m]['accuracy'] * 100 for m in methods]
        eces = [results[m]['ece'] for m in methods]
        
        # Map consistent colors based on method names
        bar_colors = [method_color_map.get(m, '#CCCCCC') for m in methods]
        
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
        bars_ece = ax_ece.bar(range(len(methods)), eces, color=bar_colors,
                              edgecolor='black', linewidth=1.2)
        ax_ece.set_ylabel('ECE', fontsize=11, fontweight='bold')
        ax_ece.set_xlabel('Method', fontsize=10)
        ax_ece.set_xticks(range(len(methods)))
        ax_ece.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax_ece.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best ECE
        best_idx = np.argmin(eces)
        bars_ece[best_idx].set_edgecolor('green')
        bars_ece[best_idx].set_linewidth(3)
        
        # Add value labels
        max_ece = max(eces) if eces else 1.0
        for bar, ece in zip(bars_ece, eces):
            height = bar.get_height()
            ax_ece.text(bar.get_x() + bar.get_width()/2, height + max_ece * 0.02,
                       f'{ece:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle('MCQ Datasets - Method Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Add legend with consistent colors
    legend_patches = [mpatches.Patch(color=method_color_map[m], label=m) for m in all_methods]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(5, len(all_methods)),
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path = f"{output_dir}/mcq_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_openended_comparison_plot(
    all_results: Dict[str, Dict[str, Dict]],
    dataset_names: Dict[str, str],
    output_dir: str
):
    """Create a combined plot for open-ended datasets only."""
    
    # Filter for open-ended datasets
    openended_datasets = ['squad_v2', 'triviaqa']
    openended_results = {k: v for k, v in all_results.items() if k in openended_datasets}
    
    if not openended_results:
        print("⚠ No open-ended results found")
        return
    
    # Collect all unique methods
    all_methods = set()
    for results in openended_results.values():
        all_methods.update(results.keys())
    all_methods = sorted(all_methods)
    
    # Prepare data
    datasets = list(openended_results.keys())
    n_datasets = len(datasets)
    
    # Create figure
    fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 10))
    if n_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    # Use consistent method colors
    method_color_map = {method: METHOD_COLORS.get(method, '#CCCCCC') for method in all_methods}
    
    for col, dataset in enumerate(datasets):
        results = openended_results[dataset]
        
        # ONLY use methods that actually exist for THIS dataset
        methods = list(results.keys())
        
        # Extract data for methods that exist
        accuracies = [results[m]['exact_match'] * 100 for m in methods]
        eces = [results[m]['ece'] for m in methods]
        
        # Map consistent colors based on method names
        bar_colors = [method_color_map.get(m, '#CCCCCC') for m in methods]
        
        # Accuracy plot
        ax_acc = axes[0, col]
        bars = ax_acc.bar(range(len(methods)), accuracies, color=bar_colors, 
                          edgecolor='black', linewidth=1.2)
        
        ax_acc.set_ylabel('Exact Match (%)', fontsize=11, fontweight='bold')
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
        bars_ece = ax_ece.bar(range(len(methods)), eces, color=bar_colors,
                              edgecolor='black', linewidth=1.2)
        ax_ece.set_ylabel('ECE', fontsize=11, fontweight='bold')
        ax_ece.set_xlabel('Method', fontsize=10)
        ax_ece.set_xticks(range(len(methods)))
        ax_ece.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax_ece.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight best ECE
        best_idx = np.argmin(eces)
        bars_ece[best_idx].set_edgecolor('green')
        bars_ece[best_idx].set_linewidth(3)
        
        # Add value labels
        max_ece = max(eces) if eces else 1.0
        for bar, ece in zip(bars_ece, eces):
            height = bar.get_height()
            ax_ece.text(bar.get_x() + bar.get_width()/2, height + max_ece * 0.02,
                       f'{ece:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Overall title
    fig.suptitle('Open-Ended QA Datasets - Method Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Add legend with consistent colors
    legend_patches = [mpatches.Patch(color=method_color_map[m], label=m) for m in all_methods]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(3, len(all_methods)),
              bbox_to_anchor=(0.5, -0.02), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    output_path = f"{output_dir}/openended_comparison.png"
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
    parser.add_argument("--dataset", type=str, 
                       choices=['all', 'openbookqa', 'arc_challenge', 'arc_easy', 
                               'truthfulqa', 'truthfulqa_mc2', 'squad_v2', 'triviaqa'],
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
            # Include all datasets (MCQ + open-ended)
            datasets = ['openbookqa', 'arc_challenge', 'arc_easy', 
                       'truthfulqa', 'truthfulqa_mc2', 'squad_v2', 'triviaqa']
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

