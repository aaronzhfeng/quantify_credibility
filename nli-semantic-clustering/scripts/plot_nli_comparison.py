#!/usr/bin/env python3
"""
Plot NLI Evaluation Comparison: TriviaQA vs SQuAD v2

Compares the effect of NLI-based evaluation (argmax mode) across datasets and methods.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_results(result_dir: str):
    """Load argmax results for both datasets."""
    results = {}
    
    files = {
        'triviaqa': {
            'greedy': 'triviaqa_greedy_argmax_full.json',
            'selfcons': 'triviaqa_selfcons_argmax_full.json',
            'mi': 'triviaqa_mi_argmax_full.json',
        },
        'squad_v2': {
            'greedy': 'squad_v2_greedy_argmax_full.json',
            'selfcons': 'squad_v2_selfcons_argmax_full.json',
            'mi': 'squad_v2_mi_argmax_full.json',
        }
    }
    
    for dataset, methods in files.items():
        results[dataset] = {}
        for method, filename in methods.items():
            filepath = Path(result_dir) / filename
            if filepath.exists():
                with open(filepath) as f:
                    data = json.load(f)
                    # Get threshold 0.5 summary
                    summary = data['threshold_summary'].get('0.5', {})
                    results[dataset][method] = summary
            else:
                print(f"Warning: {filepath} not found")
    
    return results


def plot_accuracy_comparison(results, output_dir):
    """Plot accuracy comparison: Original vs NLI-based."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = ['greedy', 'selfcons', 'mi']
    method_labels = ['Greedy', 'Self-Consistency', 'MI']
    x = np.arange(len(methods))
    width = 0.35
    
    datasets = ['triviaqa', 'squad_v2']
    dataset_titles = ['TriviaQA', 'SQuAD v2']
    colors_orig = ['#2ecc71', '#3498db', '#9b59b6']  # Green, Blue, Purple
    colors_nli = ['#27ae60', '#2980b9', '#8e44ad']   # Darker versions
    
    for idx, (dataset, title) in enumerate(zip(datasets, dataset_titles)):
        ax = axes[idx]
        
        acc_orig = []
        acc_nli = []
        
        for method in methods:
            if method in results.get(dataset, {}):
                data = results[dataset][method]
                acc_orig.append(data.get('accuracy_original', 0))
                acc_nli.append(data.get('accuracy_clustered', 0))
            else:
                acc_orig.append(0)
                acc_nli.append(0)
        
        # Plot bars
        bars1 = ax.bar(x - width/2, acc_orig, width, label='F1-based (Original)', 
                       color='#ecf0f1', edgecolor='#34495e', linewidth=1.5)
        bars2 = ax.bar(x + width/2, acc_nli, width, label='NLI-based (Argmax)',
                       color='#3498db', edgecolor='#2980b9', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars1, acc_orig):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        for bar, val in zip(bars2, acc_nli):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add delta annotations
        for i, (orig, nli) in enumerate(zip(acc_orig, acc_nli)):
            delta = nli - orig
            color = '#27ae60' if delta > 0 else '#e74c3c'
            ax.annotate(f'Δ={delta:+.2f}', xy=(i, max(orig, nli) + 0.08),
                       ha='center', fontsize=10, color=color, fontweight='bold')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{title}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels)
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right')
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    plt.suptitle('Accuracy: F1-based vs NLI-based Evaluation (Argmax Mode)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'nli_accuracy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_ece_comparison(results, output_dir):
    """Plot ECE comparison: Original vs NLI-based."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    methods = ['greedy', 'selfcons', 'mi']
    method_labels = ['Greedy', 'Self-Consistency', 'MI']
    x = np.arange(len(methods))
    width = 0.35
    
    datasets = ['triviaqa', 'squad_v2']
    dataset_titles = ['TriviaQA', 'SQuAD v2']
    
    for idx, (dataset, title) in enumerate(zip(datasets, dataset_titles)):
        ax = axes[idx]
        
        ece_orig = []
        ece_nli = []
        
        for method in methods:
            if method in results.get(dataset, {}):
                data = results[dataset][method]
                ece_orig.append(data.get('ece_original', 0))
                ece_nli.append(data.get('ece_clustered', 0))
            else:
                ece_orig.append(0)
                ece_nli.append(0)
        
        # Plot bars
        bars1 = ax.bar(x - width/2, ece_orig, width, label='F1-based (Original)',
                       color='#ecf0f1', edgecolor='#34495e', linewidth=1.5)
        bars2 = ax.bar(x + width/2, ece_nli, width, label='NLI-based (Argmax)',
                       color='#e74c3c', edgecolor='#c0392b', linewidth=1.5)
        
        # Add value labels
        for bar, val in zip(bars1, ece_orig):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, ece_nli):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add delta annotations
        for i, (orig, nli) in enumerate(zip(ece_orig, ece_nli)):
            delta = nli - orig
            color = '#e74c3c' if delta > 0 else '#27ae60'  # Red if ECE increases (bad)
            ax.annotate(f'Δ={delta:+.3f}', xy=(i, max(orig, nli) + 0.04),
                       ha='center', fontsize=9, color=color, fontweight='bold')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('ECE (↓ better)')
        ax.set_title(f'{title}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(method_labels)
        ax.set_ylim(0, 0.8)
        ax.legend(loc='upper right')
    
    plt.suptitle('Expected Calibration Error: F1-based vs NLI-based Evaluation', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'nli_ece_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_delta_summary(results, output_dir):
    """Plot delta summary: How much NLI changes metrics."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    methods = ['greedy', 'selfcons', 'mi']
    method_labels = ['Greedy', 'Self-Cons', 'MI']
    datasets = ['triviaqa', 'squad_v2']
    dataset_labels = ['TriviaQA', 'SQuAD v2']
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Collect deltas
    triviaqa_acc_delta = []
    squad_acc_delta = []
    
    for method in methods:
        if method in results.get('triviaqa', {}):
            triviaqa_acc_delta.append(results['triviaqa'][method].get('accuracy_change', 0))
        else:
            triviaqa_acc_delta.append(0)
        
        if method in results.get('squad_v2', {}):
            squad_acc_delta.append(results['squad_v2'][method].get('accuracy_change', 0))
        else:
            squad_acc_delta.append(0)
    
    # Plot bars
    bars1 = ax.bar(x - width/2, triviaqa_acc_delta, width, label='TriviaQA',
                   color='#2ecc71', edgecolor='#27ae60', linewidth=2)
    bars2 = ax.bar(x + width/2, squad_acc_delta, width, label='SQuAD v2',
                   color='#e74c3c', edgecolor='#c0392b', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars1, triviaqa_acc_delta):
        y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.02
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
               f'{val:+.2f}', ha='center', va=va, fontsize=12, fontweight='bold', color='#27ae60')
    
    for bar, val in zip(bars2, squad_acc_delta):
        y_pos = bar.get_height() + 0.01 if val >= 0 else bar.get_height() - 0.02
        va = 'bottom' if val >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
               f'{val:+.2f}', ha='center', va=va, fontsize=12, fontweight='bold', color='#c0392b')
    
    # Reference line at 0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy Change (Δ)', fontsize=12)
    ax.set_title('Impact of NLI Evaluation on Accuracy\n(Argmax Mode, Threshold=0.5)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.set_ylim(-0.15, 0.25)
    ax.legend(loc='upper right', fontsize=11)
    
    # Add annotation
    ax.annotate('NLI helps TriviaQA (+14-17%)\nbut hurts SQuAD v2 (-7-9%)', 
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'nli_delta_summary.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_summary_table(results):
    """Print summary table."""
    print("\n" + "="*100)
    print("NLI EVALUATION COMPARISON SUMMARY (Argmax Mode, Threshold=0.5)")
    print("="*100)
    
    print(f"\n{'Dataset':<12} {'Method':<12} {'Acc Orig':<10} {'Acc NLI':<10} {'Δ Acc':<10} {'ECE Orig':<10} {'ECE NLI':<10} {'Δ ECE':<10}")
    print("-"*100)
    
    for dataset in ['triviaqa', 'squad_v2']:
        for method in ['greedy', 'selfcons', 'mi']:
            if method in results.get(dataset, {}):
                data = results[dataset][method]
                acc_orig = data.get('accuracy_original', 0)
                acc_nli = data.get('accuracy_clustered', 0)
                acc_delta = data.get('accuracy_change', 0)
                ece_orig = data.get('ece_original', 0)
                ece_nli = data.get('ece_clustered', 0)
                ece_delta = data.get('ece_change', 0)
                
                print(f"{dataset:<12} {method:<12} {acc_orig:<10.3f} {acc_nli:<10.3f} {acc_delta:+10.3f} {ece_orig:<10.3f} {ece_nli:<10.3f} {ece_delta:+10.3f}")
    
    print("="*100)
    print("\nKey Findings:")
    print("  • TriviaQA: NLI evaluation IMPROVES accuracy by +14-17% (semantic matching helps)")
    print("  • SQuAD v2: NLI evaluation HURTS accuracy by -7-9% (extractive QA needs exact match)")
    print("  • ECE generally increases with NLI (except MI on TriviaQA)")
    print("="*100 + "\n")


def main():
    result_dir = Path(__file__).parent.parent / 'results' / 'threshold_sweeps'
    output_dir = Path(__file__).parent.parent / 'results' / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading results...")
    results = load_results(result_dir)
    
    print("\nGenerating plots...")
    plot_accuracy_comparison(results, output_dir)
    plot_ece_comparison(results, output_dir)
    plot_delta_summary(results, output_dir)
    
    print_summary_table(results)
    
    print(f"\n✅ All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

