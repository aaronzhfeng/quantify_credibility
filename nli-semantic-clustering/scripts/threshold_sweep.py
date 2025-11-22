#!/usr/bin/env python3
"""
Threshold Sweep Script for NLI Clustering Debugging

This script performs a systematic sweep of NLI thresholds to help diagnose
why NLI clustering might be producing worse accuracy and ECE results.

Usage:
    python scripts/threshold_sweep.py \\
        --log-dir /path/to/logs \\
        --output results/threshold_sweep.json \\
        --thresholds 0.3 0.4 0.5 0.6 0.7 \\
        --dataset triviaqa
"""

import argparse
import json
import glob
import sys
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nli_clustering.core import NLIClusteringCache
from nli_clustering.utils import (
    compute_exact_match,
    compute_f1_score,
    compute_ece,
    estimate_mi_listing_nats,
    nats_to_bits,
    mi_to_confidence,
    compute_agreement_fraction,
)


def extract_data_from_log(log_file: str) -> Dict:
    """Extract chains and metadata from log file."""
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    # Try to find method data
    method_data = None
    for key in ["mi_method", "triviaqa_correctness_mi", "self_consistency", "greedy"]:
        if key in log_data.get("methods", {}):
            method_data = log_data["methods"][key]
            break
    
    if method_data is None:
        return None
    
    # Extract chains
    raw_outputs = method_data.get("raw_outputs", [])
    if not raw_outputs:
        return None
    
    # Detect method type
    first_output = raw_outputs[0]
    is_mi_method = "chain_id" in first_output
    
    if is_mi_method:
        # MI method: organize by chain
        chains_dict = {}
        for output in raw_outputs:
            chain_id = output["chain_id"]
            step = output["step"]
            text = output["text"]
            
            if chain_id not in chains_dict:
                chains_dict[chain_id] = {}
            chains_dict[chain_id][step] = text
        
        chains = []
        for chain_id in sorted(chains_dict.keys()):
            chain = [chains_dict[chain_id][step] for step in sorted(chains_dict[chain_id].keys())]
            chains.append(chain)
    else:
        # Self-consistency or Greedy: each sample/output is a single-step chain
        # For greedy, there's only 1 output
        chains = [[output["text"]] for output in sorted(raw_outputs, key=lambda x: x.get("sample_id", 0))]
    
    # Extract metadata
    gold_answers = log_data.get("gold_answer", [])
    if isinstance(gold_answers, str):
        try:
            import ast
            gold_answers = ast.literal_eval(gold_answers)
        except:
            gold_answers = [gold_answers]
    
    original_metrics = method_data.get("final_metrics", {})
    
    return {
        "question_id": log_data.get("question_id"),
        "question_text": log_data.get("question_text"),
        "chains": chains,
        "gold_answers": gold_answers if isinstance(gold_answers, list) else [gold_answers],
        "original_predicted": original_metrics.get("predicted", ""),
        "original_exact_match": original_metrics.get("exact_match", 0.0),
        "original_f1": original_metrics.get("f1", 0.0),
        "original_mi": original_metrics.get("mi_score", 0.0),
        "original_confidence": original_metrics.get("confidence", 0.0),
    }


def evaluate_with_threshold(
    chains: List[List[str]],
    gold_answers: List[str],
    nli_checker: NLIClusteringCache,
    threshold: float,
    original_metrics: Dict,
    is_correctness_based: bool = False,
    use_nli_grading: bool = False,
    use_argmax: bool = False
) -> Dict:
    """Evaluate clustering and metrics with specific threshold."""
    from nli_clustering.core import apply_nli_clustering_to_chains
    from collections import Counter
    
    # Apply NLI clustering
    clustered_chains = apply_nli_clustering_to_chains(chains, nli_checker, threshold, use_argmax=use_argmax)
    
    # Count unique clusters
    n_unique_before = len(set(tuple(chain) for chain in chains))
    n_unique_after = len(set(tuple(chain) for chain in clustered_chains))
    
    # Get final answers
    final_answers_original = [chain[-1] for chain in chains]
    final_answers_clustered = [chain[-1] for chain in clustered_chains]
    
    # Select predicted answer (most common)
    predicted_original = Counter(final_answers_original).most_common(1)[0][0] if final_answers_original else ""
    predicted_clustered = Counter(final_answers_clustered).most_common(1)[0][0] if final_answers_clustered else ""
    
    # Compute MI
    if is_correctness_based:
        # Map to correctness
        correctness_chains = []
        for chain in clustered_chains:
            correctness_chain = []
            for answer in chain:
                is_correct = compute_exact_match(answer, gold_answers) == 1.0
                correctness_chain.append("correct" if is_correct else "incorrect")
            correctness_chains.append(correctness_chain)
        mi_nats = estimate_mi_listing_nats(correctness_chains)
    else:
        mi_nats = estimate_mi_listing_nats(clustered_chains)
    
    mi_bits = nats_to_bits(mi_nats)
    confidence_clustered = mi_to_confidence(mi_nats, method="inverse")
    
    # IMPORTANT: Use ORIGINAL F1-based metrics as baseline
    # These come from pre-computed log files
    em_original = original_metrics.get("exact_match", 0.0)
    f1_original = original_metrics.get("f1", 0.0)
    confidence_original = original_metrics.get("confidence", 0.0)
    mi_original = original_metrics.get("mi_score", 0.0)
    
    # For Greedy (single answer), confidence should come from logprob, not MI
    # MI is always 0 for single answer, which gives confidence=1.0 (incorrect)
    if len(chains) == 1 and len(chains[0]) == 1:
        # Single answer: use original confidence (from model logprob)
        confidence_clustered = confidence_original
    
    # Evaluate NLI-clustered answer
    if use_nli_grading:
        # NEW: Use NLI-based grading (loose, unidirectional)
        # Pass ALL gold answers AND threshold to check against any acceptable answer
        em_clustered = 1.0 if nli_checker.is_correct(predicted_clustered, gold_answers, threshold=threshold, use_argmax=use_argmax) else 0.0
        f1_clustered = compute_f1_score(predicted_clustered, gold_answers)
    else:
        # ORIGINAL: Use F1-based grading (baseline)
        em_clustered = compute_exact_match(predicted_clustered, gold_answers)
        f1_clustered = compute_f1_score(predicted_clustered, gold_answers)
    
    # Agreement
    agreement_original = compute_agreement_fraction(final_answers_original)
    agreement_clustered = compute_agreement_fraction(final_answers_clustered)
    
    return {
        "threshold": threshold,
        "n_unique_before": n_unique_before,
        "n_unique_after": n_unique_after,
        "clustering_reduction": (n_unique_before - n_unique_after) / n_unique_before if n_unique_before > 0 else 0.0,
        "predicted_original": predicted_original,
        "predicted_clustered": predicted_clustered,
        "prediction_changed": predicted_original != predicted_clustered,
        # Original metrics (F1-based, from pre-computed logs)
        "em_original": em_original,
        "f1_original": f1_original,
        "confidence_original": confidence_original,
        "mi_original": mi_original,
        # NLI-clustered metrics
        "em_clustered": em_clustered,
        "f1_clustered": f1_clustered,
        "mi_clustered": mi_bits,
        "confidence_clustered": confidence_clustered,
        # Changes
        "em_change": em_clustered - em_original,
        "f1_change": f1_clustered - f1_original,
        "agreement_original": agreement_original,
        "agreement_clustered": agreement_clustered,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep NLI thresholds for debugging")
    parser.add_argument("--log-dir", type=str, required=True,
                       help="Directory containing question_*.json log files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file for threshold sweep results")
    parser.add_argument("--thresholds", nargs="+", type=float,
                       default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],
                       help="List of thresholds to test")
    parser.add_argument("--nli-model", type=str,
                       default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                       help="NLI model to use")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of questions to process")
    parser.add_argument("--correctness-based", action="store_true",
                       help="Use correctness-based MI (for TriviaQA)")
    parser.add_argument("--use-nli-grading", action="store_true",
                       help="Use NLI for accuracy checking (not just clustering). "
                            "This uses loose unidirectional entailment for grading, "
                            "which should fix the accuracy drop issue.")
    parser.add_argument("--use-argmax", action="store_true",
                       help="Use argmax classification (winner takes all) instead of soft threshold. "
                            "When enabled, entailment is accepted if it's the most likely class, "
                            "regardless of the probability value. This ignores the threshold value.")
    parser.add_argument("--dataset", type=str, default="triviaqa",
                       choices=["triviaqa", "squad_v2"],
                       help="Dataset name (for reporting)")
    
    args = parser.parse_args()
    
    # Check log directory
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory not found: {args.log_dir}")
        return
    
    # Find log files
    log_files = sorted(glob.glob(f"{args.log_dir}/question_*.json"))
    if args.limit:
        log_files = log_files[:args.limit]
    
    if not log_files:
        print(f"Error: No log files found in {args.log_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"NLI Threshold Sweep for Debugging")
    print(f"{'='*80}")
    print(f"Log directory    : {args.log_dir}")
    print(f"Questions        : {len(log_files)}")
    print(f"Thresholds       : {args.thresholds}")
    print(f"NLI model        : {args.nli_model}")
    print(f"Correctness-based: {args.correctness_based}")
    print(f"Dataset          : {args.dataset}")
    print(f"{'='*80}\n")
    
    # Load NLI model
    print("Loading NLI model...")
    nli_checker = NLIClusteringCache(model_name=args.nli_model)
    print("✓ Model loaded\n")
    
    # Process each question across all thresholds
    all_results = []
    
    for log_file in tqdm(log_files, desc="Processing questions"):
        data = extract_data_from_log(log_file)
        if data is None:
            continue
        
        question_results = {
            "question_id": data["question_id"],
            "question_text": data["question_text"],
            "gold_answers": data["gold_answers"],
            "original_metrics": {
                "predicted": data["original_predicted"],
                "exact_match": data["original_exact_match"],
                "f1": data["original_f1"],
                "mi": data["original_mi"],
                "confidence": data["original_confidence"],
            },
            "threshold_results": []
        }
        
        # Test each threshold
        for threshold in args.thresholds:
            result = evaluate_with_threshold(
                data["chains"],
                data["gold_answers"],
                nli_checker,
                threshold,
                original_metrics=question_results["original_metrics"],
                is_correctness_based=args.correctness_based,
                use_nli_grading=args.use_nli_grading,
                use_argmax=args.use_argmax
            )
            question_results["threshold_results"].append(result)
        
        all_results.append(question_results)
    
    # Compute aggregated statistics per threshold
    threshold_summary = {}
    
    for threshold in args.thresholds:
        # Collect metrics for this threshold
        metrics = []
        for qr in all_results:
            for tr in qr["threshold_results"]:
                if tr["threshold"] == threshold:
                    metrics.append(tr)
        
        if not metrics:
            continue
        
        # Compute averages
        threshold_summary[threshold] = {
            "avg_clustering_reduction": np.mean([m["clustering_reduction"] for m in metrics]),
            "avg_n_unique_before": np.mean([m["n_unique_before"] for m in metrics]),
            "avg_n_unique_after": np.mean([m["n_unique_after"] for m in metrics]),
            "predictions_changed": sum(1 for m in metrics if m["prediction_changed"]),
            "predictions_changed_pct": sum(1 for m in metrics if m["prediction_changed"]) / len(metrics) * 100,
            "accuracy_original": np.mean([m["em_original"] for m in metrics]),
            "accuracy_clustered": np.mean([m["em_clustered"] for m in metrics]),
            "accuracy_change": np.mean([m["em_change"] for m in metrics]),
            "f1_original": np.mean([m["f1_original"] for m in metrics]),
            "f1_clustered": np.mean([m["f1_clustered"] for m in metrics]),
            "f1_change": np.mean([m["f1_change"] for m in metrics]),
            "avg_mi_original": np.mean([m["mi_original"] for m in metrics]),
            "avg_mi_clustered": np.mean([m["mi_clustered"] for m in metrics]),
            "avg_confidence_original": np.mean([m["confidence_original"] for m in metrics]),
            "avg_confidence_clustered": np.mean([m["confidence_clustered"] for m in metrics]),
            # Compute ECE (Expected Calibration Error)
            # ECE measures |bin_accuracy - bin_confidence| weighted by bin size
            # bin_accuracy = fraction of correct answers in each confidence bin
            # This gives the mathematically correct calibration metric
            "ece_original": compute_ece(
                predictions=np.array([m["em_original"] for m in metrics]),  # Pass for API compatibility
                confidences=np.array([m["confidence_original"] for m in metrics]),
                labels=np.array([m["em_original"] for m in metrics])  # Actual correctness (0/1)
            ),
            "ece_clustered": compute_ece(
                predictions=np.array([m["em_clustered"] for m in metrics]),  # Pass for API compatibility
                confidences=np.array([m["confidence_clustered"] for m in metrics]),
                labels=np.array([m["em_clustered"] for m in metrics])  # Actual correctness (0/1)
            ),
            # Count improvements and degradations
            "em_improved": sum(1 for m in metrics if m["em_change"] > 0),
            "em_degraded": sum(1 for m in metrics if m["em_change"] < 0),
            "em_unchanged": sum(1 for m in metrics if m["em_change"] == 0),
        }
        
        # Compute ECE change
        threshold_summary[threshold]["ece_change"] = (
            threshold_summary[threshold]["ece_clustered"] - 
            threshold_summary[threshold]["ece_original"]
        )
    
    # Save results
    output_data = {
        "config": {
            "log_directory": args.log_dir,
            "n_questions": len(all_results),
            "thresholds": args.thresholds,
            "nli_model": args.nli_model,
            "correctness_based": args.correctness_based,
            "use_nli_grading": args.use_nli_grading,
            "use_argmax": args.use_argmax,
            "dataset": args.dataset,
        },
        "threshold_summary": threshold_summary,
        "per_question_results": all_results,
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("THRESHOLD SWEEP RESULTS")
    print(f"{'='*120}\n")
    print(f"{'Threshold':<12} {'Clusters':<10} {'Acc Orig':<10} {'Acc NLI':<10} {'Δ Acc':<10} {'ECE Orig':<10} {'ECE NLI':<10} {'Δ ECE':<10} {'Changed':<10}")
    print("-" * 120)
    
    for threshold in sorted(args.thresholds):
        summary = threshold_summary.get(threshold, {})
        if not summary:
            continue
        
        reduction = summary["avg_clustering_reduction"] * 100
        acc_orig = summary["accuracy_original"]
        acc_nli = summary["accuracy_clustered"]
        acc_change = summary["accuracy_change"]
        ece_orig = summary["ece_original"]
        ece_nli = summary["ece_clustered"]
        ece_change = summary["ece_change"]
        changed_pct = summary["predictions_changed_pct"]
        
        print(f"{threshold:<12.2f} {reduction:<9.1f}% {acc_orig:<10.3f} {acc_nli:<10.3f} "
              f"{acc_change:+10.3f} {ece_orig:<10.3f} {ece_nli:<10.3f} {ece_change:+10.3f} {changed_pct:>6.1f}%")
    
    print(f"\n{'='*120}")
    print(f"Detailed results saved to: {args.output}")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()

