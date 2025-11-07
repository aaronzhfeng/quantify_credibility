"""
Utility to merge evaluation results from multiple GPU workers.
"""

import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def merge_csv_files(csv_files: List[str], output_path: str):
    """
    Merge multiple CSV files into one.
    
    Args:
        csv_files: List of CSV file paths
        output_path: Output path for merged CSV
    """
    # Read first file to get headers
    valid_files = [f for f in csv_files if Path(f).exists()]
    
    if not valid_files:
        raise ValueError("No valid CSV files to merge")
    
    # Read all rows
    all_rows = []
    fieldnames = None
    
    for csv_file in valid_files:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                all_rows.append(row)
    
    # Write merged CSV
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    logger.info(f"Merged {len(valid_files)} CSV files → {len(all_rows)} total rows")
    
    return len(all_rows)


def merge_json_metrics(json_files: List[str], output_path: str) -> Dict[str, Any]:
    """
    Merge metrics from multiple JSON files.
    
    Computes weighted averages for metrics based on sample counts.
    
    Args:
        json_files: List of JSON file paths
        output_path: Output path for merged JSON
        
    Returns:
        Merged metrics dict
    """
    # Read all JSONs
    metrics_list = []
    for f in json_files:
        if Path(f).exists():
            with open(f, 'r') as file:
                metrics = json.load(file)
                metrics_list.append(metrics)
        else:
            logger.warning(f"JSON file not found: {f}")
    
    if not metrics_list:
        raise ValueError("No valid JSON files to merge")
    
    # Calculate total samples
    total_samples = sum(m['n_samples'] for m in metrics_list)
    
    # Merge metrics with weighted averaging
    merged = {
        "n_samples": total_samples
    }
    
    # Weighted average for scalar metrics
    scalar_metrics = ['accuracy', 'ece', 'avg_confidence', 'avg_mi_bits', 'avg_agreement',
                     'exact_match', 'f1', 'avg_correctness_agreement']
    
    for key in scalar_metrics:
        if key in metrics_list[0]:
            weighted_sum = sum(m[key] * m['n_samples'] for m in metrics_list if key in m)
            merged[key] = weighted_sum / total_samples
    
    # Merge logprob_stats
    if 'logprob_stats' in metrics_list[0]:
        merged_logprob_stats = {
            'total_inferences': sum(m['logprob_stats']['total_inferences'] for m in metrics_list),
            'captured': sum(m['logprob_stats']['captured'] for m in metrics_list),
            'fallback': sum(m['logprob_stats']['fallback'] for m in metrics_list),
        }
        # Recompute capture rate
        merged_logprob_stats['capture_rate'] = (
            merged_logprob_stats['captured'] / merged_logprob_stats['total_inferences']
            if merged_logprob_stats['total_inferences'] > 0 else 0.0
        )
        merged['logprob_stats'] = merged_logprob_stats
    
    # Save merged metrics
    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    logger.info(f"Merged {len(metrics_list)} JSON files → {total_samples} total samples")
    
    return merged


def merge_evaluation_results(worker_outputs: List[str], output_path: str, temp_dir: Path):
    """
    Merge all evaluation results from GPU workers.
    
    Args:
        worker_outputs: List of worker CSV output paths
        output_path: Final output CSV path
        temp_dir: Temporary directory containing worker outputs
    """
    # Merge CSVs
    n_rows = merge_csv_files(worker_outputs, output_path)
    
    # Merge JSONs
    json_outputs = [f.replace('.csv', '.json') for f in worker_outputs]
    json_output_path = output_path.replace('.csv', '.json')
    merged_metrics = merge_json_metrics(json_outputs, json_output_path)
    
    # Print summary
    print(f"  - Merged {len(worker_outputs)} worker outputs")
    print(f"  - Total predictions: {n_rows}")
    print(f"  - Combined metrics (weighted averages):")
    
    for key, value in merged_metrics.items():
        if key != 'logprob_stats' and isinstance(value, (int, float)):
            print(f"      {key:20s}: {value:.4f}")
    
    # Print merged logprob stats
    if 'logprob_stats' in merged_metrics:
        stats = merged_metrics['logprob_stats']
        print()
        print("="*60)
        print("LOGPROB EXTRACTION STATUS (Merged from all GPUs)")
        print("="*60)
        print(f"Total inferences     : {stats['total_inferences']:,}")
        print(f"Successfully captured: {stats['captured']:,} ({stats['capture_rate']*100:.1f}%)")
        print(f"Fallback (0.0)       : {stats['fallback']:,} ({(1-stats['capture_rate'])*100:.1f}%)")
        
        if stats['capture_rate'] < 0.9:
            print(f"\n⚠️  WARNING: Low logprob capture rate!")
            print(f"   See docs/LOGPROB_DIAGNOSTIC.md for details")
        print("="*60)

