#!/usr/bin/env python3
"""
Analyze mutual entailment clustering using NLI model.

Compares F1-based semantic similarity (current method) with 
NLI-based mutual entailment (proposed method) for answer clustering.

Usage:
    python scripts/analyze_mutual_entailment.py \
        --dataset triviaqa \
        --method mi \
        --limit 200 \
        --output outputs/nli_analysis/triviaqa_mi_200_nli.json
"""

import argparse
import json
import glob
import os
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import numpy as np


@dataclass
class ClusterComparison:
    """Comparison between F1 and NLI clustering for one question."""
    question_id: int
    question_text: str
    unique_answers: List[str]
    n_unique: int
    
    # F1-based clustering (current method)
    f1_n_clusters: int
    f1_clusters: Dict[int, List[str]]
    
    # NLI-based clustering (proposed method)
    nli_n_clusters: int
    nli_clusters: Dict[int, List[str]]
    
    # Comparison metrics
    clustering_agreement: float  # How similar are the two clusterings?
    f1_purity: float  # Avg F1 similarity within NLI clusters
    nli_purity: float  # Avg NLI score within F1 clusters
    
    # Detailed pairwise scores
    pairwise_f1: Dict[str, float]  # "ans1|||ans2" -> f1_score
    pairwise_nli_forward: Dict[str, float]  # "ans1|||ans2" -> P(ans1 entails ans2)
    pairwise_nli_backward: Dict[str, float]  # "ans1|||ans2" -> P(ans2 entails ans1)
    pairwise_mutual: Dict[str, bool]  # "ans1|||ans2" -> mutually_entailed
    
    # NEW: Evaluation comparison (checking predicted vs gold answer)
    predicted_answer: str  # The answer the method predicted
    gold_answers: List[str]  # Ground truth answer(s)
    current_correct: bool  # Was it correct with current evaluation (exact match/F1)?
    nli_correct: bool  # Would it be correct with NLI evaluation?
    nli_eval_changed: bool  # Did NLI evaluation change the result?
    nli_gold_scores: Dict[str, Tuple[float, float]]  # gold_ans -> (forward, backward) entailment probs


class MutualEntailmentChecker:
    """Check mutual entailment using NLI model."""
    
    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli", device: str = None):
        """
        Initialize NLI model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        print(f"Loading NLI model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mapping (DeBERTa-MNLI: 0=contradiction, 1=neutral, 2=entailment)
        self.label2id = self.model.config.label2id
        self.entailment_id = self.label2id.get('entailment', 2)
        
        print(f"Model loaded on {self.device}")
        print(f"Label mapping: {self.label2id}")
        print(f"Entailment label ID: {self.entailment_id}")
    
    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Check if premise entails hypothesis.
        
        Returns:
            P(entailment) between 0 and 1
        """
        inputs = self.tokenizer(
            premise, hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            entailment_prob = probs[0][self.entailment_id].item()
        
        return entailment_prob
    
    def check_mutual_entailment(
        self, 
        text_a: str, 
        text_b: str, 
        threshold: float = 0.5
    ) -> Tuple[bool, float, float]:
        """
        Check bidirectional entailment (mutual entailment).
        
        Returns:
            (is_mutual, forward_prob, backward_prob)
        """
        forward = self.check_entailment(text_a, text_b)
        backward = self.check_entailment(text_b, text_a)
        is_mutual = (forward > threshold) and (backward > threshold)
        
        return is_mutual, forward, backward


def compute_f1_similarity(text1: str, text2: str) -> float:
    """
    Compute F1 similarity based on token overlap (current method).
    
    This is the algorithm currently used in calibration.py.
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    precision = intersection / len(tokens1)
    recall = intersection / len(tokens2)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def cluster_by_f1(answers: List[str], threshold: float = 0.25) -> Dict[int, List[str]]:
    """
    Cluster answers by F1 similarity (current method).
    
    Greedy clustering: each answer joins the first cluster it matches,
    or creates a new cluster.
    """
    clusters = {}
    representatives = {}
    next_id = 0
    
    for answer in answers:
        matched = None
        for cluster_id, rep in representatives.items():
            if compute_f1_similarity(answer, rep) >= threshold:
                matched = cluster_id
                break
        
        if matched is not None:
            clusters[matched].append(answer)
        else:
            clusters[next_id] = [answer]
            representatives[next_id] = answer
            next_id += 1
    
    return clusters


def cluster_by_nli(
    answers: List[str], 
    checker: MutualEntailmentChecker,
    threshold: float = 0.5
) -> Dict[int, List[str]]:
    """
    Cluster answers by mutual entailment (proposed method).
    
    Greedy clustering: each answer joins the first cluster where it
    mutually entails the representative, or creates a new cluster.
    """
    clusters = {}
    representatives = {}
    next_id = 0
    
    for answer in answers:
        matched = None
        for cluster_id, rep in representatives.items():
            is_mutual, _, _ = checker.check_mutual_entailment(answer, rep, threshold)
            if is_mutual:
                matched = cluster_id
                break
        
        if matched is not None:
            clusters[matched].append(answer)
        else:
            clusters[next_id] = [answer]
            representatives[next_id] = answer
            next_id += 1
    
    return clusters


def compute_clustering_agreement(
    clusters_a: Dict[int, List[str]], 
    clusters_b: Dict[int, List[str]]
) -> float:
    """
    Compute agreement between two clusterings using Adjusted Rand Index.
    
    Returns value between 0 (random) and 1 (perfect agreement).
    """
    from sklearn.metrics import adjusted_rand_score
    
    # Build label arrays
    all_items = []
    labels_a = []
    labels_b = []
    
    # Map answers to cluster IDs for clustering A
    item_to_cluster_a = {}
    for cluster_id, items in clusters_a.items():
        for item in items:
            item_to_cluster_a[item] = cluster_id
            all_items.append(item)
    
    # Map answers to cluster IDs for clustering B
    item_to_cluster_b = {}
    for cluster_id, items in clusters_b.items():
        for item in items:
            item_to_cluster_b[item] = cluster_id
    
    # Build parallel label arrays
    for item in all_items:
        labels_a.append(item_to_cluster_a.get(item, -1))
        labels_b.append(item_to_cluster_b.get(item, -1))
    
    if len(labels_a) < 2:
        return 1.0  # Perfect agreement for single item
    
    return adjusted_rand_score(labels_a, labels_b)


def check_nli_correctness(
    predicted: str,
    gold_answers: List[str],
    checker: MutualEntailmentChecker,
    threshold: float = 0.5
) -> Tuple[bool, Dict[str, Tuple[float, float]]]:
    """
    Check if predicted answer is semantically equivalent to any gold answer using NLI.
    
    Args:
        predicted: The model's predicted answer
        gold_answers: List of acceptable gold answers
        checker: NLI mutual entailment checker
        threshold: Threshold for mutual entailment
    
    Returns:
        (is_correct, scores_dict) where scores_dict maps gold_answer -> (forward, backward) probs
    """
    scores = {}
    
    for gold in gold_answers:
        # Normalize strings for comparison
        pred_norm = predicted.strip().strip('"').strip("'").lower()
        gold_norm = gold.strip().strip('"').strip("'").lower()
        
        # Check mutual entailment
        is_mutual, fwd, bwd = checker.check_mutual_entailment(
            pred_norm, gold_norm, threshold
        )
        scores[gold] = (fwd, bwd)
        
        if is_mutual:
            return True, scores  # Semantically correct!
    
    return False, scores


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip quotes)."""
    return answer.strip().strip('"').strip("'").lower()


def check_exact_match(predicted: str, gold_answers: List[str]) -> bool:
    """Check if predicted answer exactly matches any gold answer."""
    pred_norm = normalize_answer(predicted)
    for gold in gold_answers:
        if pred_norm == normalize_answer(gold):
            return True
    return False


def analyze_question(
    question_data: dict,
    checker: MutualEntailmentChecker,
    f1_threshold: float = 0.25,
    nli_threshold: float = 0.5
) -> ClusterComparison:
    """
    Analyze clustering for one question.
    
    Compares F1-based vs NLI-based clustering.
    Also compares current evaluation vs NLI-based evaluation.
    """
    question_id = question_data['question_id']
    question_text = question_data['question_text']
    
    # Extract gold answers
    gold_answer_str = question_data.get('gold_answer', '[]')
    if isinstance(gold_answer_str, str):
        import ast
        try:
            gold_answers = ast.literal_eval(gold_answer_str)
            if not isinstance(gold_answers, list):
                gold_answers = [gold_answers]
        except:
            gold_answers = [gold_answer_str]
    else:
        gold_answers = [gold_answer_str] if not isinstance(gold_answer_str, list) else gold_answer_str
    
    # Extract unique answers
    answers = []
    seen = set()
    predicted_answer = None
    current_correct = False
    
    if 'mi_method' in question_data.get('methods', {}):
        method_data = question_data['methods']['mi_method']
        # Get predicted answer from method
        predicted_answer = method_data.get('predicted_answer', '')
        current_correct = method_data.get('is_correct', False)
    elif 'self_consistency' in question_data.get('methods', {}):
        method_data = question_data['methods']['self_consistency']
        predicted_answer = method_data.get('predicted_answer', '')
        current_correct = method_data.get('is_correct', False)
    else:
        # Greedy method - only 1 answer, skip
        return None
    
    for output in method_data.get('raw_outputs', []):
        text = output['text'].strip().strip('"').strip("'")
        if text not in seen:
            answers.append(text)
            seen.add(text)
    
    if len(answers) < 2:
        # Need at least 2 answers for clustering
        return None
    
    # Cluster by F1
    f1_clusters = cluster_by_f1(answers, threshold=f1_threshold)
    
    # Cluster by NLI
    nli_clusters = cluster_by_nli(answers, checker, threshold=nli_threshold)
    
    # Compute pairwise scores for all pairs
    pairwise_f1 = {}
    pairwise_nli_fwd = {}
    pairwise_nli_bwd = {}
    pairwise_mutual = {}
    
    for i, ans1 in enumerate(answers):
        for j, ans2 in enumerate(answers):
            if i >= j:
                continue
            
            pair_key = f"{ans1}|||{ans2}"
            
            # F1 score
            f1_score = compute_f1_similarity(ans1, ans2)
            pairwise_f1[pair_key] = f1_score
            
            # NLI scores
            is_mutual, fwd, bwd = checker.check_mutual_entailment(ans1, ans2, nli_threshold)
            pairwise_nli_fwd[pair_key] = fwd
            pairwise_nli_bwd[pair_key] = bwd
            pairwise_mutual[pair_key] = is_mutual
    
    # Compute clustering agreement
    agreement = compute_clustering_agreement(f1_clusters, nli_clusters)
    
    # TODO: Compute purity metrics
    f1_purity = 0.0
    nli_purity = 0.0
    
    # NEW: Check NLI-based evaluation (predicted vs gold)
    nli_correct = False
    nli_gold_scores = {}
    
    if predicted_answer and gold_answers:
        nli_correct, nli_gold_scores = check_nli_correctness(
            predicted_answer, gold_answers, checker, nli_threshold
        )
    
    nli_eval_changed = (nli_correct != current_correct)
    
    return ClusterComparison(
        question_id=question_id,
        question_text=question_text,
        unique_answers=answers,
        n_unique=len(answers),
        f1_n_clusters=len(f1_clusters),
        f1_clusters={k: list(v) for k, v in f1_clusters.items()},
        nli_n_clusters=len(nli_clusters),
        nli_clusters={k: list(v) for k, v in nli_clusters.items()},
        clustering_agreement=agreement,
        f1_purity=f1_purity,
        nli_purity=nli_purity,
        pairwise_f1=pairwise_f1,
        pairwise_nli_forward=pairwise_nli_fwd,
        pairwise_nli_backward=pairwise_nli_bwd,
        pairwise_mutual=pairwise_mutual,
        # NEW: Evaluation metrics
        predicted_answer=predicted_answer or "",
        gold_answers=gold_answers,
        current_correct=current_correct,
        nli_correct=nli_correct,
        nli_eval_changed=nli_eval_changed,
        nli_gold_scores=nli_gold_scores
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze mutual entailment clustering")
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['triviaqa', 'squad_v2', 'truthfulqa_mc1', 'truthfulqa_mc2'],
                       help='Dataset name')
    parser.add_argument('--method', type=str, required=True,
                       choices=['mi', 'self-consistency', 'selfcons'],
                       help='Method name (mi or self-consistency)')
    parser.add_argument('--limit', type=int, default=200,
                       help='Number of questions to analyze')
    parser.add_argument('--f1-threshold', type=float, default=0.25,
                       help='F1 similarity threshold (paper uses 0.25)')
    parser.add_argument('--nli-threshold', type=float, default=0.5,
                       help='NLI entailment threshold')
    parser.add_argument('--model', type=str, default='microsoft/deberta-v2-xlarge-mnli',
                       help='NLI model to use')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu, auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Normalize method name
    method_name = 'selfcons' if 'cons' in args.method else 'mi'
    
    # Find log files
    log_dir = f'outputs/logs/{args.dataset}_{method_name}_200'
    if not os.path.exists(log_dir):
        # Try alternative naming
        log_dir = f'outputs/logs/{args.dataset}_{args.method}_200'
    
    if not os.path.exists(log_dir):
        print(f"Error: Log directory not found: {log_dir}")
        return
    
    log_files = sorted(glob.glob(f'{log_dir}/question_*.json'))[:args.limit]
    
    if not log_files:
        print(f"Error: No log files found in {log_dir}")
        return
    
    print(f"\nAnalyzing {len(log_files)} questions from {args.dataset} ({args.method})")
    print(f"Log directory: {log_dir}")
    print(f"F1 threshold: {args.f1_threshold}")
    print(f"NLI threshold: {args.nli_threshold}\n")
    
    # Load NLI model
    checker = MutualEntailmentChecker(model_name=args.model, device=args.device)
    
    # Analyze each question
    results = []
    skipped = 0
    
    start_time = time.time()
    
    for log_file in tqdm(log_files, desc="Analyzing questions"):
        with open(log_file, 'r') as f:
            question_data = json.load(f)
        
        comparison = analyze_question(
            question_data, 
            checker, 
            f1_threshold=args.f1_threshold,
            nli_threshold=args.nli_threshold
        )
        
        if comparison is not None:
            results.append(comparison)
        else:
            skipped += 1
    
    elapsed = time.time() - start_time
    
    # Compute summary statistics
    n_analyzed = len(results)
    avg_unique = np.mean([r.n_unique for r in results]) if results else 0
    avg_f1_clusters = np.mean([r.f1_n_clusters for r in results]) if results else 0
    avg_nli_clusters = np.mean([r.nli_n_clusters for r in results]) if results else 0
    avg_agreement = np.mean([r.clustering_agreement for r in results]) if results else 0
    
    # Count where NLI found more/fewer clusters
    nli_more = sum(1 for r in results if r.nli_n_clusters > r.f1_n_clusters)
    nli_fewer = sum(1 for r in results if r.nli_n_clusters < r.f1_n_clusters)
    nli_same = sum(1 for r in results if r.nli_n_clusters == r.f1_n_clusters)
    
    # NEW: Compute evaluation metrics (accuracy with NLI vs current)
    current_correct = sum(1 for r in results if r.current_correct)
    nli_correct = sum(1 for r in results if r.nli_correct)
    eval_changed = sum(1 for r in results if r.nli_eval_changed)
    
    # Count changes by direction
    wrong_to_right = sum(1 for r in results if not r.current_correct and r.nli_correct)
    right_to_wrong = sum(1 for r in results if r.current_correct and not r.nli_correct)
    
    current_accuracy = current_correct / n_analyzed if n_analyzed > 0 else 0
    nli_accuracy = nli_correct / n_analyzed if n_analyzed > 0 else 0
    accuracy_improvement = nli_accuracy - current_accuracy
    
    summary = {
        'dataset': args.dataset,
        'method': args.method,
        'n_questions_analyzed': n_analyzed,
        'n_questions_skipped': skipped,
        'f1_threshold': args.f1_threshold,
        'nli_threshold': args.nli_threshold,
        'nli_model': args.model,
        'elapsed_seconds': elapsed,
        # Clustering metrics
        'avg_unique_answers': avg_unique,
        'avg_f1_clusters': avg_f1_clusters,
        'avg_nli_clusters': avg_nli_clusters,
        'avg_clustering_agreement': avg_agreement,
        'nli_more_clusters': nli_more,
        'nli_fewer_clusters': nli_fewer,
        'nli_same_clusters': nli_same,
        # NEW: Evaluation metrics
        'current_accuracy': current_accuracy,
        'nli_accuracy': nli_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'current_correct_count': current_correct,
        'nli_correct_count': nli_correct,
        'eval_changed_count': eval_changed,
        'wrong_to_right_count': wrong_to_right,
        'right_to_wrong_count': right_to_wrong,
    }
    
    # Save results
    output_data = {
        'summary': summary,
        'per_question': [asdict(r) for r in results]
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Questions analyzed   : {n_analyzed}")
    print(f"Questions skipped    : {skipped}")
    print(f"Time elapsed         : {elapsed:.1f}s ({elapsed/n_analyzed:.2f}s per question)")
    
    print(f"\n{'='*80}")
    print("CLUSTERING ANALYSIS")
    print(f"{'='*80}")
    print(f"Avg unique answers   : {avg_unique:.1f}")
    print(f"Avg F1 clusters      : {avg_f1_clusters:.1f}")
    print(f"Avg NLI clusters     : {avg_nli_clusters:.1f}")
    print(f"Clustering agreement : {avg_agreement:.3f}")
    print(f"\nNLI vs F1 clustering:")
    print(f"  NLI more clusters  : {nli_more} ({100*nli_more/n_analyzed:.1f}%)")
    print(f"  NLI fewer clusters : {nli_fewer} ({100*nli_fewer/n_analyzed:.1f}%)")
    print(f"  NLI same clusters  : {nli_same} ({100*nli_same/n_analyzed:.1f}%)")
    
    print(f"\n{'='*80}")
    print("EVALUATION ANALYSIS (Predicted vs Gold Answer)")
    print(f"{'='*80}")
    print(f"Current accuracy     : {current_accuracy:.4f} ({current_correct}/{n_analyzed})")
    print(f"NLI accuracy         : {nli_accuracy:.4f} ({nli_correct}/{n_analyzed})")
    print(f"Accuracy improvement : {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%)")
    print(f"\nEvaluation changes:")
    print(f"  Wrong → Right      : {wrong_to_right} (NLI recognized semantic match)")
    print(f"  Right → Wrong      : {right_to_wrong} (NLI rejected false positive)")
    print(f"  Total changed      : {eval_changed} ({100*eval_changed/n_analyzed:.1f}%)")
    
    print(f"\nResults saved to: {args.output}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

