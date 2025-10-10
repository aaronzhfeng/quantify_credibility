from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def compute_agreement_fraction(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts: dict[str, int] = {}
    for v in values:
        key = (v or "").strip()
        counts[key] = counts.get(key, 0) + 1
    max_c = max(counts.values()) if counts else 0
    return max_c / float(len(values)) if values else 0.0


def label_any_correct(final_answers: Sequence[str], gold_answers: Sequence[str], normalizer) -> int:
    gold_n = {normalizer(g) for g in gold_answers if g}
    for a in final_answers:
        if normalizer(a) in gold_n:
            return 1
    return 0


def label_majority_correct(final_answers: Sequence[str], gold_answers: Sequence[str], normalizer) -> int:
    # Majority vote among final answers, then compare to gold
    counts: dict[str, int] = {}
    for a in final_answers:
        counts[normalizer(a)] = counts.get(normalizer(a), 0) + 1
    if not counts:
        return 0
    top_norm, _ = max(counts.items(), key=lambda kv: kv[1])
    gold_n = {normalizer(g) for g in gold_answers if g}
    return 1 if top_norm in gold_n else 0


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    val_n = max(1, int(round(val_fraction * n))) if n > 1 else 0
    val = idxs[:val_n]
    test = idxs[val_n:]
    if not test:
        # Ensure at least one test example when n>=1
        test = val[-1:]
        val = val[:-1]
    return val, test


def compute_ranks(scores: List[float]) -> List[float]:
    # Average ranks for ties (1-based ranks like scipy.stats.rankdata(method='average'))
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(sorted_idx):
        j = i
        s = scores[sorted_idx[i]]
        while j + 1 < len(sorted_idx) and scores[sorted_idx[j + 1]] == s:
            j += 1
        # indices i..j share the same score
        avg_rank = (i + j + 2) / 2.0  # 1-based
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def roc_auc(scores: List[float], labels_pos1: List[int]) -> float:
    # Positive class is label==1. AUC equals probability positive has higher score than negative
    n_pos = sum(1 for y in labels_pos1 if y == 1)
    n_neg = len(labels_pos1) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = compute_ranks(scores)
    sum_pos_ranks = sum(r for r, y in zip(ranks, labels_pos1) if y == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def choose_threshold(scores: List[float], labels_pos1: List[int], maximize: str = "youden") -> float:
    # Evaluate at unique thresholds; prediction: score >= thr -> positive (label 1)
    unique_scores = sorted(set(scores))
    if not unique_scores:
        return 0.0
    best_thr = unique_scores[0]
    best_val = -1e9
    for thr in unique_scores:
        tp = fp = tn = fn = 0
        for s, y in zip(scores, labels_pos1):
            pred = 1 if s >= thr else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        if maximize == "accuracy":
            score = acc
        else:  # youden's J
            score = tpr - fpr
        if score > best_val:
            best_val = score
            best_thr = thr
    return best_thr


def evaluate_at_threshold(scores: List[float], labels_pos1: List[int], thr: float) -> dict:
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels_pos1):
        pred = 1 if s >= thr else 0
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 1 and y == 0:
            fp += 1
        elif pred == 0 and y == 0:
            tn += 1
        else:
            fn += 1
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = roc_auc(scores, labels_pos1)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "threshold": thr,
        "n": len(scores),
    }


def roc_curve_points(scores: List[float], labels_pos1: List[int]) -> Tuple[List[float], List[float]]:
    # Return FPR and TPR lists evaluated at all unique thresholds
    unique_scores = sorted(set(scores))
    fprs: List[float] = []
    tprs: List[float] = []
    for thr in unique_scores:
        tp = fp = tn = fn = 0
        for s, y in zip(scores, labels_pos1):
            pred = 1 if s >= thr else 0
            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fprs.append(fpr)
        tprs.append(tpr)
    return fprs, tprs


def precision_recall_curve_points(scores: List[float], labels_pos1: List[int]) -> Tuple[List[float], List[float]]:
    # Evaluate precision/recall at all unique thresholds (descending scores)
    pairs = sorted(zip(scores, labels_pos1), key=lambda x: x[0], reverse=True)
    tp = fp = 0
    fn = sum(1 for _, y in pairs if y == 1)
    precs: List[float] = []
    recs: List[float] = []
    prev_score = None
    for s, y in pairs:
        # Move threshold to s (include this point)
        if y == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
    return recs, precs



