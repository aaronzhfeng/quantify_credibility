#!/usr/bin/env python3
"""
Plot ablation study results.

Shows how varying each parameter affects accuracy and ECE.

Usage:
    python scripts/plot_ablation.py --all
    python scripts/plot_ablation.py --parameter temperature
    python scripts/plot_ablation.py --parameter k_chains n_length
"""

import json
import argparse
from pathlib import Path
from glob import glob
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_ablation_results(parameter: str) -> Tuple[List[str], List[float], List[float], List[int]]:
    """
    Load ablation results for a specific parameter.
    
    Returns:
        (labels, accuracies, eces, n_samples)
    """
    # Map parameter names to their directories and labels
    param_configs = {
        'temperature': {
            'dir': 'temperature',
            'files': ['temp0.5', 'temp0.9', 'temp1.3'],
            'labels': ['T=0.5', 'T=0.9', 'T=1.3'],
            'xlabel': 'Temperature',
            'title': 'Temperature Ablation'
        },
        'k_chains': {
            'dir': 'k_chains',
            'files': ['k5', 'k10', 'k20'],
            'labels': ['k=5', 'k=10', 'k=20'],
            'xlabel': 'Number of Chains (k)',
            'title': 'Number of Chains Ablation'
        },
        'n_length': {
            'dir': 'n_length',
            'files': ['n2', 'n3', 'n4'],
            'labels': ['n=2', 'n=3', 'n=4'],
            'xlabel': 'Chain Length (n)',
            'title': 'Chain Length Ablation'
        },
        'mi_method': {
            'dir': 'mi_method',
            'files': ['listing', 'plugin'],
            'labels': ['Listing', 'Plugin'],
            'xlabel': 'MI Estimator',
            'title': 'MI Estimator Ablation'
        },
        'confidence_method': {
            'dir': 'confidence_method',
            'files': ['inverse', 'exp', 'normalized'],
            'labels': ['Inverse', 'Exp', 'Normalized'],
            'xlabel': 'Confidence Method',
            'title': 'Confidence Conversion Ablation'
        },
        'answer_format': {
            'dir': 'answer_format',
            'files': ['strict', 'codeblock'],
            'labels': ['Strict', 'Codeblock'],
            'xlabel': 'Answer Format',
            'title': 'Answer Format Ablation'
        }
    }
    
    if parameter not in param_configs:
        raise ValueError(f"Unknown parameter: {parameter}")
    
    config = param_configs[parameter]
    labels = []
    accuracies = []
    eces = []
    n_samples_list = []
    
    for file_name, label in zip(config['files'], config['labels']):
        json_path = f"outputs/results/ablation/{config['dir']}/{file_name}.json"
        
        if Path(json_path).exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                labels.append(label)
                accuracies.append(data['accuracy'] * 100)  # Convert to percentage
                eces.append(data['ece'])
                n_samples_list.append(data.get('n_samples', 0))
        else:
            print(f"⚠ File not found: {json_path}")
    
    return labels, accuracies, eces, n_samples_list, config


def plot_parameter_ablation(
    parameter: str,
    output_dir: str = "outputs/plots/ablation"
):
    """
    Plot ablation results for a single parameter.
    
    Args:
        parameter: Parameter name (e.g., 'temperature', 'k_chains')
        output_dir: Output directory for plots
    """
    labels, accuracies, eces, n_samples_list, config = load_ablation_results(parameter)
    
    if not labels:
        print(f"No results found for parameter: {parameter}")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Plot 1: Accuracy
    colors_acc = plt.cm.Blues(np.linspace(0.4, 0.8, len(labels)))
    bars1 = ax1.bar(x, accuracies, width, color=colors_acc, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel(config['xlabel'], fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight best accuracy
    best_acc_idx = np.argmax(accuracies)
    bars1[best_acc_idx].set_edgecolor('green')
    bars1[best_acc_idx].set_linewidth(3)
    
    # Plot 2: ECE (lower is better)
    colors_ece = plt.cm.Oranges(np.linspace(0.4, 0.8, len(labels)))
    bars2 = ax2.bar(x, eces, width, color=colors_ece, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel(config['xlabel'], fontsize=12, fontweight='bold')
    ax2.set_ylabel('ECE (Expected Calibration Error)', fontsize=12, fontweight='bold')
    ax2.set_title('ECE (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylim([0, max(eces) * 1.2])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, ece in zip(bars2, eces):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + max(eces) * 0.03,
                f'{ece:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Highlight best ECE (lowest)
    best_ece_idx = np.argmin(eces)
    bars2[best_ece_idx].set_edgecolor('green')
    bars2[best_ece_idx].set_linewidth(3)
    
    # Overall title
    fig.suptitle(f'{config["title"]} (MI Method on OpenBookQA)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add note about sample size and best values
    n = n_samples_list[0] if n_samples_list else 'N/A'
    fig.text(0.5, 0.02, 
             f'N = {n} examples | Green border = Best value | '
             f'Best Acc: {labels[best_acc_idx]} ({accuracies[best_acc_idx]:.1f}%) | '
             f'Best ECE: {labels[best_ece_idx]} ({eces[best_ece_idx]:.4f})', 
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    
    # Save plot
    output_path = Path(output_dir) / f"ablation_{parameter}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_combined_ablation_plot(output_dir: str = "outputs/plots/ablation"):
    """Create a comprehensive plot showing all ablation results."""
    
    parameters = ['temperature', 'k_chains', 'n_length', 'mi_method', 'confidence_method', 'answer_format']
    
    # Create large figure with 3 rows and 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    axes = axes.flatten()
    
    for idx, param in enumerate(parameters):
        try:
            labels, accuracies, eces, n_samples_list, config = load_ablation_results(param)
            
            if not labels:
                continue
            
            ax = axes[idx]
            
            # Create dual y-axis plot
            x = np.arange(len(labels))
            width = 0.35
            
            # Plot accuracy bars
            ax_acc = ax
            bars1 = ax_acc.bar(x - width/2, accuracies, width, 
                              label='Accuracy', color='steelblue', 
                              edgecolor='black', linewidth=1.2)
            ax_acc.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold', color='steelblue')
            ax_acc.tick_params(axis='y', labelcolor='steelblue')
            ax_acc.set_ylim([min(accuracies) * 0.95, max(accuracies) * 1.02])
            
            # Create second y-axis for ECE
            ax_ece = ax_acc.twinx()
            bars2 = ax_ece.bar(x + width/2, eces, width, 
                              label='ECE', color='coral', 
                              edgecolor='black', linewidth=1.2)
            ax_ece.set_ylabel('ECE', fontsize=11, fontweight='bold', color='coral')
            ax_ece.tick_params(axis='y', labelcolor='coral')
            ax_ece.set_ylim([0, max(eces) * 1.3])
            
            # Set x-axis
            ax_acc.set_xticks(x)
            ax_acc.set_xticklabels(labels, fontsize=10)
            ax_acc.set_xlabel(config['xlabel'], fontsize=11, fontweight='bold')
            ax_acc.set_title(config['title'], fontsize=12, fontweight='bold')
            ax_acc.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels
            for bar, acc in zip(bars1, accuracies):
                height = bar.get_height()
                ax_acc.text(bar.get_x() + bar.get_width()/2, height,
                           f'{acc:.1f}', ha='center', va='bottom', 
                           fontsize=8, color='steelblue', fontweight='bold')
            
            for bar, ece in zip(bars2, eces):
                height = bar.get_height()
                ax_ece.text(bar.get_x() + bar.get_width()/2, height,
                           f'{ece:.3f}', ha='center', va='bottom', 
                           fontsize=8, color='coral', fontweight='bold')
            
            # Highlight best values
            best_acc_idx = np.argmax(accuracies)
            bars1[best_acc_idx].set_edgecolor('green')
            bars1[best_acc_idx].set_linewidth(2.5)
            
            best_ece_idx = np.argmin(eces)
            bars2[best_ece_idx].set_edgecolor('darkgreen')
            bars2[best_ece_idx].set_linewidth(2.5)
            
        except Exception as e:
            print(f"⚠ Error plotting {param}: {e}")
            continue
    
    # Overall title
    fig.suptitle('MI Method Parameter Ablation Study (OpenBookQA)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', lw=8, label='Accuracy (%)'),
        Line2D([0], [0], color='coral', lw=8, label='ECE (lower is better)'),
        Line2D([0], [0], color='green', lw=3, label='Best value')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
              bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.99])
    
    # Save plot
    output_path = Path(output_dir) / "ablation_combined.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument("--parameter", type=str, nargs='+',
                       choices=['temperature', 'k_chains', 'n_length', 'mi_method', 
                               'confidence_method', 'answer_format'],
                       help="Specific parameter(s) to plot")
    parser.add_argument("--all", action="store_true",
                       help="Plot all parameters")
    parser.add_argument("--combined", action="store_true",
                       help="Create combined plot with all parameters")
    parser.add_argument("--output-dir", type=str, default="outputs/plots/ablation",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PLOTTING ABLATION STUDY RESULTS")
    print("="*80)
    print()
    
    if args.combined or args.all:
        print("Creating combined ablation plot...")
        create_combined_ablation_plot(args.output_dir)
        print()
    
    if args.all:
        parameters = ['temperature', 'k_chains', 'n_length', 'mi_method', 
                     'confidence_method', 'answer_format']
    elif args.parameter:
        parameters = args.parameter
    else:
        print("Please specify --parameter, --all, or --combined")
        return
    
    for param in parameters:
        print(f"Plotting {param} ablation...")
        plot_parameter_ablation(param, args.output_dir)
    
    print()
    print("="*80)
    print("ABLATION PLOTTING COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()

