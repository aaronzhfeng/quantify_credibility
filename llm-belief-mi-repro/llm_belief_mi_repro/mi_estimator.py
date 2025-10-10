from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Tuple


def _tuplify_chain(chain: List[str]) -> Tuple[str, ...]:
    # Normalize whitespace a bit to reduce spurious uniqueness
    return tuple(a.strip() for a in chain)


def _empirical_entropy(counts: Counter, total: int) -> float:
    # H = - sum p log p (nats)
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return h


def estimate_mi_nats(chains: List[List[str]]) -> float:
    """Estimate MI of (Y1,...,Yt) as Sum_i H(Yi) - H(Y1..Yt) from samples.

    This uses a simple plug-in estimator over discrete strings, following the paper's
    finite-sample MI intuition. While biased for small samples, it is effective for a
    basic reproduction.
    """
    if not chains:
        return 0.0
    t = len(chains[0])
    for ch in chains:
        if len(ch) != t:
            raise ValueError("All chains must have the same length")

    n = len(chains)
    joint = Counter(_tuplify_chain(ch) for ch in chains)
    h_joint = _empirical_entropy(joint, n)

    sum_h_marginals = 0.0
    for i in range(t):
        counts_i = Counter(ch[i].strip() for ch in chains)
        sum_h_marginals += _empirical_entropy(counts_i, n)

    mi = max(0.0, sum_h_marginals - h_joint)
    return mi


def nats_to_bits(nats: float) -> float:
    return nats / math.log(2.0)


