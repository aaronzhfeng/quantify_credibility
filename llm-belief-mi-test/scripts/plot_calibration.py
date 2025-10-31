#!/usr/bin/env python3
"""
Plot calibration curves (reliability diagrams) from CSV results.

Shows how well confidence scores match actual accuracy.

Usage:
    python scripts/plot_calibration.py --dataset openbookqa
    python scripts/plot_calibration.py --dataset all
    python scripts/plot_calibration.py --files outputs/results/openbookqa_*_500.csv
"""

import pandas as pd
import argparse
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import numpy as np


def extract_method_name(filepath: str) -> str:
    """Extract method name from filepath."""
    stem = Path(filepath).stem
    
    for dataset in ['openbookqa', 'arc_challenge', 'arc_easy']:
        if dataset in stem:
            name = stem.replace(dataset + '_', '').replace('_500', '')
            name_map = {
                'greedy': 'Greedy',
                'selfcons': 'Self-Consistency',
                'semantic_entropy': 'Semantic Entropy',
                'self_verification': 'Self-Verification',
                'mi': 'MI Method'
            }
            return name_map.get(name, name.replace('_', ' ').title())
    
    return stem


def compute_calibration_curve(df: pd.DataFrame, n_bins: int = 10) -> tuple:
    """
    Compute calibration curve data.
    
    Returns:
        (bin_confidences, bin_accuracies, bin_counts)
    """
    # Extract data
    confidences = df['confidence'].values
    correct = df['correct'].values
    
    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        else:
            in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_confidences.append(bin_confidence)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(n_in_bin)
        else:
            bin_confidences.append(np.nan)
            bin_accuracies.append(np.nan)
            bin_counts.append(0)
    
    return np.array(bin_confidences), np.array(bin_accuracies), np.array(bin_counts)


def plot_calibration_curves(files: list, title: str, output_path: str, n_bins: int = 10):
    """
    Plot calibration curves for multiple methods.
    
    Args:
        files: List of CSV filepaths
        title: Plot title
        output_path: Where to save plot
        n_bins: Number of calibration bins
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(files)))
    
    for idx, filepath in enumerate(files):
        if not Path(filepath).exists():
            print(f"⚠ File not found: {filepath}")
            continue
        
        method = extract_method_name(filepath)
        df = pd.read_csv(filepath)
        
        # Compute calibration curve
        bin_confs, bin_accs, bin_counts = compute_calibration_curve(df, n_bins)
        
        # Filter out empty bins
        valid = ~np.isnan(bin_confs)
        bin_confs_valid = bin_confs[valid]
        bin_accs_valid = bin_accs[valid]
        bin_counts_valid = bin_counts[valid]
        
        # Plot calibration curve
        ax1.plot(bin_confs_valid, bin_accs_valid, 'o-', 
                label=method, color=colors[idx], linewidth=2, markersize=8)
        
        # Plot histogram of confidences
        df['confidence'].hist(bins=20, alpha=0.5, label=method, 
                             color=colors[idx], ax=ax2, edgecolor='black')
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.7)
    
    # Configure calibration plot
    ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    
    # Configure histogram
    ax2.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, axis='y')
    
    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot calibration curves")
    parser.add_argument("--dataset", type=str,
                       choices=['all', 'openbookqa', 'arc_challenge', 'arc_easy'],
                       default='all', help="Dataset to plot")
    parser.add_argument("--files", nargs='+', help="Custom list of CSV files")
    parser.add_argument("--output-dir", type=str, default="outputs/plots",
                       help="Output directory")
    parser.add_argument("--bins", type=int, default=10, help="Number of calibration bins")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("PLOTTING CALIBRATION CURVES")
    print("="*80)
    print()
    
    if args.files:
        # Custom files
        title = "Calibration Comparison"
        output_path = output_dir / "calibration_custom.png"
        plot_calibration_curves(args.files, title, str(output_path), args.bins)
    else:
        # By dataset
        datasets = ['openbookqa', 'arc_challenge', 'arc_easy'] if args.dataset == 'all' else [args.dataset]
        
        dataset_names = {
            'openbookqa': 'OpenBookQA',
            'arc_challenge': 'ARC-Challenge',
            'arc_easy': 'ARC-Easy'
        }
        
        for dataset in datasets:
            pattern = f"outputs/results/{dataset}/{dataset}_*_500.csv"
            files = sorted(glob(pattern))
            
            if files:
                title = f"{dataset_names.get(dataset, dataset)} - Calibration Analysis"
                output_path = output_dir / f"{dataset}_calibration.png"
                plot_calibration_curves(files, title, str(output_path), args.bins)
            else:
                print(f"⚠ No CSV files found for {dataset}")
    
    print()
    print("="*80)
    print("CALIBRATION PLOTTING COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}/")
    print()


if __name__ == "__main__":
    main()

