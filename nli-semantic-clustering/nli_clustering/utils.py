"""
Utility functions for NLI clustering evaluation.

This module contains minimal utilities copied from llm-belief-mi-test for:
- Answer normalization and evaluation (exact match, F1 score)
- Mutual Information estimation
- Expected Calibration Error (ECE) computation
- Agreement and entropy metrics
"""

from __future__ import annotations
import re
import string
import math
import numpy as np
from typing import List
from collections import Counter


# =============================================================================
# Answer Normalization and Evaluation Metrics
# =============================================================================

def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison (SQuAD evaluation standard).
    
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    
    Args:
        text: Input text to normalize
    
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute exact match score (0 or 1).
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable answers (empty for unanswerable)
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not ground_truths:  # Unanswerable question
        # Check if model correctly abstained
        normalized_pred = normalize_answer(prediction)
        return 1.0 if normalized_pred in ["unanswerable", "no answer", "cannot answer", ""] else 0.0
    
    normalized_pred = normalize_answer(prediction)
    
    for ground_truth in ground_truths:
        if normalized_pred == normalize_answer(ground_truth):
            return 1.0
    
    return 0.0


def compute_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute F1 score (token-level overlap).
    
    Standard SQuAD evaluation metric.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable answers (empty for unanswerable)
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not ground_truths:  # Unanswerable
        normalized_pred = normalize_answer(prediction)
        return 1.0 if normalized_pred in ["unanswerable", "no answer", "cannot answer", ""] else 0.0
    
    # Compute F1 against all ground truths, take maximum
    f1_scores = []
    normalized_pred = normalize_answer(prediction)
    pred_tokens = normalized_pred.split()
    
    for ground_truth in ground_truths:
        gt_tokens = normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        # Compute overlap
        common_tokens = set(pred_tokens) & set(gt_tokens)
        
        if len(common_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    return max(f1_scores) if f1_scores else 0.0


# =============================================================================
# Mutual Information Estimation
# =============================================================================

def _empirical_entropy(counts: Counter, total: int) -> float:
    """Compute empirical entropy from counts: H = - sum p log p (nats)."""
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return h


def estimate_mi_listing_nats(
    chains: List[List[str]],
    gamma1: float | None = None,
    gamma2: float | None = None,
) -> float:
    """
    Estimate MI using the listing-based algorithm.
    
    This implements Algorithm 1 from the paper, adapted for LLM sampling.
    
    Args:
        chains: List of chains, each chain is a list of answer strings
        gamma1: Stabilization parameter for numerator (default: 1/K)
        gamma2: Stabilization parameter for denominator (default: 1/K)
    
    Returns:
        MI estimate in nats
    """
    if not chains:
        return 0.0
    t = len(chains[0])
    for ch in chains:
        if len(ch) != t:
            raise ValueError("All chains must have the same length")

    # Deduplicate tuples and count occurrences (clusters)
    tuples = [tuple(a.strip() for a in ch) for ch in chains]
    counts: Counter = Counter(tuples)
    unique = list(counts.keys())
    weights = [counts[u] for u in unique]  # size = num_clusters

    k = len(chains)
    if k <= 0:
        return 0.0
    g1 = (1.0 / k) if gamma1 is None else float(gamma1)
    g2 = (1.0 / k) if gamma2 is None else float(gamma2)

    # Empirical distribution on sample (hat_mu)
    total = float(sum(weights))
    hat_mu = [w / total for w in weights]

    # Product-of-marginals on the sampled support
    # For each position i, compute marginal mass for each symbol seen at i
    # using hat_mu weights.
    # Then for each tuple x, hat_mu_prod[x] = Π_i marginal_i[x[i]].
    # Build index maps for efficiency.
    # marginals[i]: dict of value -> mass
    marginals: List[dict[str, float]] = [dict() for _ in range(t)]
    for u, w_hat in zip(unique, hat_mu):
        for i in range(t):
            key = u[i]
            marginals[i][key] = marginals[i].get(key, 0.0) + w_hat

    hat_mu_prod: List[float] = []
    for u in unique:
        prod = 1.0
        for i in range(t):
            prod *= marginals[i].get(u[i], 0.0)
        hat_mu_prod.append(prod)

    # MI estimate with stabilization
    mi_est = 0.0
    for w_hat, w_prod in zip(hat_mu, hat_mu_prod):
        mi_est += w_hat * math.log((w_hat + g1) / (w_prod + g2))
    return max(0.0, mi_est)


def nats_to_bits(nats: float) -> float:
    """Convert nats to bits."""
    return nats / math.log(2.0)


def mi_to_confidence(mi_score: float, method: str = "inverse") -> float:
    """
    Convert MI score to confidence (0 to 1).
    
    Higher MI = higher uncertainty = lower confidence.
    
    Args:
        mi_score: Mutual information score (in nats or bits)
        method: Conversion method
            - "inverse": conf = 1 / (1 + MI)
            - "exponential": conf = exp(-MI)
    
    Returns:
        Confidence score between 0 and 1
    """
    if method == "inverse":
        return 1.0 / (1.0 + mi_score)
    elif method == "exponential":
        return math.exp(-mi_score)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Agreement and Entropy Metrics
# =============================================================================

def compute_agreement_fraction(answers: List[str]) -> float:
    """
    Compute fraction of answers that match the most common answer.
    
    Args:
        answers: List of answer strings
    
    Returns:
        Agreement fraction (0.0 to 1.0)
    """
    if not answers:
        return 0.0
    
    counts = Counter(a.strip() for a in answers)
    most_common_count = counts.most_common(1)[0][1]
    return most_common_count / len(answers)


def compute_entropy_nats(values: List[str]) -> float:
    """
    Compute entropy of discrete distribution.
    
    Args:
        values: List of values
    
    Returns:
        Entropy in nats
    """
    counts = Counter(v.strip() for v in values)
    total = sum(counts.values())
    return _empirical_entropy(counts, total)


# =============================================================================
# Calibration Metrics
# =============================================================================

def compute_ece(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy in each bin.
    
    For calibration measurement, pass the correctness labels (0/1) as labels.
    The bin accuracy is computed as the fraction of correct answers in each
    confidence bin, NOT as (predictions == labels).mean().
    
    Args:
        predictions: Binary array of predictions (0 or 1)
                    For calibration, typically pass the same as labels
        confidences: Confidence scores (0 to 1)
        labels: Ground truth correctness labels (0 or 1)
                This is the actual correctness (EM scores)
        n_bins: Number of calibration bins
        
    Returns:
        ECE value (0 to 1, lower is better)
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        
        if i == n_bins - 1:  # Last bin includes right edge
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            # Compute accuracy in this bin
            # CORRECTED: Use labels.mean() to get fraction of correct answers
            # NOT (predictions == labels).mean() which would be 1.0 when pred==label
            bin_accuracy = labels[in_bin].mean()
            # Compute average confidence in this bin
            bin_confidence = confidences[in_bin].mean()
            # Weight by fraction of samples in bin
            bin_weight = n_in_bin / len(confidences)
            # Add to ECE
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece

