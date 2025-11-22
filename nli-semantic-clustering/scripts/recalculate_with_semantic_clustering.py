#!/usr/bin/env python3
"""
Recalculate MI-based metrics from existing log files with NLI clustering.

This script reads existing log files (from previous MI method runs) and recalculates
MI scores, confidence, and ECE by applying NLI semantic clustering to the chains.

Key advantage: No need to re-run expensive Llama inference!
"""

import argparse
import json
import glob
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_belief_mi_test.calibration import (
    NLIClusteringCache,
    apply_nli_clustering_to_chains,
    mi_to_confidence,
    compute_ece
)
from llm_belief_mi_test.mi_estimator import estimate_mi_listing_nats, nats_to_bits
from llm_belief_mi_test.datasets import compute_exact_match, compute_f1_score


def safe_percent_change(new_val, old_val):
    """Safely compute percentage change, handling zero/nan cases."""
    if np.isnan(new_val) or np.isnan(old_val):
        return 0.0
    if old_val == 0:
        return 0.0 if new_val == 0 else float('inf')
    return (new_val - old_val) / old_val * 100


def extract_chains_from_log(log_data: dict, method_key: str = "mi_method") -> Tuple[List[List[str]], Dict]:
    """
    Extract chains of text responses from log file.
    
    Args:
        log_data: Parsed JSON log data
        method_key: Key for the method data (e.g., "mi_method", "triviaqa_correctness_mi")
    
    Returns:
        (chains, metadata) where chains is List[List[str]] and metadata contains original metrics
    """
    # Try different method keys
    method_keys_to_try = [
        method_key,
        "mi_method",
        "triviaqa_correctness_mi",
        "squad_v2_mi",
        "self_consistency"
    ]
    
    method_data = None
    for key in method_keys_to_try:
        if key in log_data.get("methods", {}):
            method_data = log_data["methods"][key]
            break
    
    if method_data is None:
        raise ValueError(f"Could not find method data in log. Available keys: {list(log_data.get('methods', {}).keys())}")
    
    # Extract raw outputs
    raw_outputs = method_data.get("raw_outputs", [])
    if not raw_outputs:
        raise ValueError("No raw_outputs found in log data")
    
    # Detect method type by checking first output
    first_output = raw_outputs[0]
    is_mi_method = "chain_id" in first_output and "step" in first_output
    is_self_consistency = "sample_id" in first_output
    
    if is_mi_method:
        # MI method: Organize by chain_id and step
        chains_dict = {}
        for output in raw_outputs:
            chain_id = output["chain_id"]
            step = output["step"]
            text = output["text"]
            
            if chain_id not in chains_dict:
                chains_dict[chain_id] = {}
            chains_dict[chain_id][step] = text
        
        # Convert to list of chains
        chains = []
        for chain_id in sorted(chains_dict.keys()):
            chain = [chains_dict[chain_id][step] for step in sorted(chains_dict[chain_id].keys())]
            chains.append(chain)
    
    elif is_self_consistency:
        # Self-consistency: Each sample is an independent single-step "chain"
        chains = []
        for output in sorted(raw_outputs, key=lambda x: x["sample_id"]):
            chains.append([output["text"]])  # Single-element chain
    
    else:
        raise ValueError(f"Unknown method format. First output keys: {list(first_output.keys())}")
    
    # Extract metadata
    metadata = {
        "question_id": log_data.get("question_id"),
        "question_text": log_data.get("question_text"),
        "gold_answers": log_data.get("gold_answer", []),
        "original_metrics": method_data.get("final_metrics", {}),
        "decision_process": method_data.get("decision_process", {})
    }
    
    # Parse gold_answers if it's a string (sometimes stored as str representation of list)
    if isinstance(metadata["gold_answers"], str):
        try:
            import ast
            metadata["gold_answers"] = ast.literal_eval(metadata["gold_answers"])
        except:
            # If parsing fails, wrap in list
            metadata["gold_answers"] = [metadata["gold_answers"]]
    
    return chains, metadata


def recalculate_with_nli(
    chains: List[List[str]],
    gold_answers: List[str],
    nli_checker: NLIClusteringCache,
    nli_threshold: float,
    is_correctness_based: bool = False,
    use_nli_grading: bool = False
) -> Dict:
    """
    Recalculate MI and confidence with NLI clustering applied.
    
    Args:
        chains: List of chains (List[List[str]])
        gold_answers: Ground truth answers
        nli_checker: NLI model for clustering
        nli_threshold: Threshold for mutual entailment
        is_correctness_based: If True, map to correctness before MI calculation
    
    Returns:
        Dictionary with recalculated metrics
    """
    # Apply NLI clustering to chains
    clustered_chains = apply_nli_clustering_to_chains(chains, nli_checker, nli_threshold)
    
    if is_correctness_based:
        # Map clustered chains to binary correctness (for TriviaQA)
        correctness_chains = []
        for chain in clustered_chains:
            correctness_chain = []
            for answer_text in chain:
                is_correct = compute_exact_match(answer_text, gold_answers) == 1.0
                correctness_chain.append("correct" if is_correct else "incorrect")
            correctness_chains.append(correctness_chain)
        
        # Compute MI on correctness
        mi_nats = estimate_mi_listing_nats(correctness_chains)
    else:
        # Compute MI directly on clustered text (for SQuAD)
        mi_nats = estimate_mi_listing_nats(clustered_chains)
    
    mi_bits = nats_to_bits(mi_nats)
    confidence = mi_to_confidence(mi_nats, method="inverse")
    
    # Get final answers for answer selection
    final_answers = [chain[-1] for chain in clustered_chains]
    from collections import Counter
    predicted_answer = Counter(final_answers).most_common(1)[0][0] if final_answers else ""
    
    # Evaluate accuracy
    if use_nli_grading:
        # NEW: Use NLI-based grading (loose, unidirectional)
        # This accepts verbose but correct answers
        # Pass ALL gold answers AND threshold to check against any acceptable answer
        exact_match = 1.0 if nli_checker.is_correct(predicted_answer, gold_answers, threshold=nli_threshold) else 0.0
        # Still compute F1 for comparison
        f1 = compute_f1_score(predicted_answer, gold_answers)
    else:
        # ORIGINAL: Use F1-based grading (baseline)
        exact_match = compute_exact_match(predicted_answer, gold_answers)
        f1 = compute_f1_score(predicted_answer, gold_answers)
    
    # Calculate agreement
    from llm_belief_mi_test.evaluation import compute_agreement_fraction
    agreement = compute_agreement_fraction(final_answers)
    
    return {
        "mi_nats": mi_nats,
        "mi_bits": mi_bits,
        "confidence": confidence,
        "predicted_answer": predicted_answer,
        "exact_match": exact_match,
        "f1": f1,
        "agreement": agreement,
        "clustered_chains": clustered_chains,
        "unique_clusters": len(set(tuple(chain) for chain in clustered_chains))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Recalculate MI metrics from existing logs with NLI clustering"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing question_*.json log files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for recalculated results"
    )
    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.5,
        help="Threshold for NLI mutual entailment (default: 0.5)"
    )
    parser.add_argument(
        "--nli-model",
        type=str,
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="NLI model to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to process"
    )
    parser.add_argument(
        "--correctness-based",
        action="store_true",
        help="Use correctness-based MI (for TriviaQA)"
    )
    parser.add_argument(
        "--use-nli-grading",
        action="store_true",
        help="Use NLI for accuracy checking (not just clustering). "
             "This uses loose unidirectional entailment for grading, "
             "which should improve accuracy on verbose answers."
    )
    
    args = parser.parse_args()
    
    # Check log directory exists
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
    print(f"Recalculating MI Metrics with NLI Clustering")
    print(f"{'='*80}")
    print(f"Log directory     : {args.log_dir}")
    print(f"Questions found   : {len(log_files)}")
    print(f"NLI threshold     : {args.nli_threshold}")
    print(f"NLI model         : {args.nli_model}")
    print(f"Correctness-based : {args.correctness_based}")
    print(f"{'='*80}\n")
    
    # Initialize NLI checker
    print("Loading NLI model...")
    nli_checker = NLIClusteringCache(model_name=args.nli_model)
    print("✓ NLI model loaded\n")
    
    # Process each question
    results = []
    original_metrics_list = []
    
    for log_file in tqdm(log_files, desc="Processing questions"):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Extract chains
            chains, metadata = extract_chains_from_log(log_data)
            
            # Store original metrics
            original_metrics = metadata["original_metrics"]
            original_metrics_list.append(original_metrics)
            
            # Recalculate with NLI
            nli_metrics = recalculate_with_nli(
                chains=chains,
                gold_answers=metadata["gold_answers"],
                nli_checker=nli_checker,
                nli_threshold=args.nli_threshold,
                is_correctness_based=args.correctness_based,
                use_nli_grading=args.use_nli_grading
            )
            
            # Combine results
            result = {
                "question_id": metadata["question_id"],
                "question_text": metadata["question_text"],
                "gold_answers": metadata["gold_answers"],
                
                # Original (no NLI)
                "original": {
                    "mi_bits": original_metrics.get("mi_score", 0),
                    "confidence": original_metrics.get("confidence", 0),
                    "predicted": original_metrics.get("predicted", ""),
                    "exact_match": original_metrics.get("exact_match", 0),
                    "f1": original_metrics.get("f1", 0)
                },
                
                # Recalculated with NLI
                "nli_adapted": {
                    "mi_bits": nli_metrics["mi_bits"],
                    "confidence": nli_metrics["confidence"],
                    "predicted": nli_metrics["predicted_answer"],
                    "exact_match": nli_metrics["exact_match"],
                    "f1": nli_metrics["f1"],
                    "unique_clusters": nli_metrics["unique_clusters"]
                },
                
                # Changes
                "changes": {
                    "mi_reduction": nli_metrics["mi_bits"] - original_metrics.get("mi_score", 0),
                    "confidence_increase": nli_metrics["confidence"] - original_metrics.get("confidence", 0),
                    "accuracy_change": nli_metrics["exact_match"] - original_metrics.get("exact_match", 0),
                    "prediction_changed": nli_metrics["predicted_answer"] != original_metrics.get("predicted", "")
                }
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\nWarning: Failed to process {log_file}: {e}")
            continue
    
    # Compute summary statistics
    n = len(results)
    
    # Original metrics
    orig_mi = np.mean([r["original"]["mi_bits"] for r in results])
    orig_conf = np.mean([r["original"]["confidence"] for r in results])
    orig_em = np.mean([r["original"]["exact_match"] for r in results])
    orig_f1 = np.mean([r["original"]["f1"] for r in results])
    
    # NLI-adapted metrics
    nli_mi = np.mean([r["nli_adapted"]["mi_bits"] for r in results])
    nli_conf = np.mean([r["nli_adapted"]["confidence"] for r in results])
    nli_em = np.mean([r["nli_adapted"]["exact_match"] for r in results])
    nli_f1 = np.mean([r["nli_adapted"]["f1"] for r in results])
    
    # Calculate ECE for both
    # Note: We use correctness (exact_match) as labels, and confidence thresholding as predictions
    # This measures calibration: do high-confidence answers have higher accuracy?
    orig_correctness = np.array([r["original"]["exact_match"] for r in results])
    orig_confs = np.array([r["original"]["confidence"] for r in results])
    orig_preds = (orig_confs > 0.5).astype(float)  # Binary prediction: confident or not
    orig_ece = compute_ece(orig_preds, orig_confs, orig_correctness)
    
    nli_correctness = np.array([r["nli_adapted"]["exact_match"] for r in results])
    nli_confs = np.array([r["nli_adapted"]["confidence"] for r in results])
    nli_preds = (nli_confs > 0.5).astype(float)  # Binary prediction: confident or not
    nli_ece = compute_ece(nli_preds, nli_confs, nli_correctness)
    
    # Changes
    predictions_changed = sum(1 for r in results if r["changes"]["prediction_changed"])
    
    summary = {
        "dataset_info": {
            "log_directory": args.log_dir,
            "n_questions": n,
            "nli_threshold": args.nli_threshold,
            "nli_model": args.nli_model,
            "correctness_based": args.correctness_based
        },
        
        "original_metrics": {
            "avg_mi_bits": float(orig_mi),
            "avg_confidence": float(orig_conf),
            "exact_match": float(orig_em),
            "f1": float(orig_f1),
            "ece": float(orig_ece)
        },
        
        "nli_adapted_metrics": {
            "avg_mi_bits": float(nli_mi),
            "avg_confidence": float(nli_conf),
            "exact_match": float(nli_em),
            "f1": float(nli_f1),
            "ece": float(nli_ece)
        },
        
        "improvements": {
            "mi_reduction": float(nli_mi - orig_mi),
            "mi_reduction_pct": float((nli_mi - orig_mi) / orig_mi * 100) if orig_mi > 0 else 0,
            "confidence_increase": float(nli_conf - orig_conf),
            "confidence_increase_pct": float((nli_conf - orig_conf) / orig_conf * 100) if orig_conf > 0 else 0,
            "ece_improvement": float(nli_ece - orig_ece),
            "ece_improvement_pct": float((nli_ece - orig_ece) / orig_ece * 100) if orig_ece > 0 else 0,
            "accuracy_change": float(nli_em - orig_em),
            "predictions_changed": predictions_changed,
            "predictions_changed_pct": float(predictions_changed / n * 100) if n > 0 else 0
        }
    }
    
    # Save results
    output_data = {
        "summary": summary,
        "per_question": results
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"\nOriginal (No NLI Clustering):")
    print(f"  Avg MI          : {orig_mi:.4f} bits")
    print(f"  Avg Confidence  : {orig_conf:.4f}")
    print(f"  Exact Match     : {orig_em:.4f}")
    print(f"  F1 Score        : {orig_f1:.4f}")
    print(f"  ECE             : {orig_ece:.4f}")
    
    print(f"\nNLI-Adapted (With Semantic Clustering):")
    print(f"  Avg MI          : {nli_mi:.4f} bits ({nli_mi - orig_mi:+.4f}, {safe_percent_change(nli_mi, orig_mi):+.1f}%)")
    print(f"  Avg Confidence  : {nli_conf:.4f} ({nli_conf - orig_conf:+.4f}, {safe_percent_change(nli_conf, orig_conf):+.1f}%)")
    print(f"  Exact Match     : {nli_em:.4f} ({nli_em - orig_em:+.4f})")
    print(f"  F1 Score        : {nli_f1:.4f} ({nli_f1 - orig_f1:+.4f})")
    print(f"  ECE             : {nli_ece:.4f} ({nli_ece - orig_ece:+.4f}, {safe_percent_change(nli_ece, orig_ece):+.1f}%)")
    
    print(f"\nPrediction Changes:")
    print(f"  Changed         : {predictions_changed}/{n} ({predictions_changed/n*100:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

