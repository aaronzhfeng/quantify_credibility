from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple


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

    This uses a simple plug-in estimator over discrete strings. While biased for small
    samples, it is effective for a basic reproduction.
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


def entropy_nats(values: List[str]) -> float:
    counts = Counter(v.strip() for v in values)
    total = sum(counts.values())
    return _empirical_entropy(counts, total)


# Listing.tex-inspired MI estimator (Algorithm 1) ---------------------------------

def estimate_mi_listing_nats(
    chains: List[List[str]],
    gamma1: float | None = None,
    gamma2: float | None = None,
) -> float:
    """Estimate MI using the paper's Algorithm 1 structure from listing.tex.

    Adapts the code to our LLM sampling setting:
    - We observe K samples of joint tuples (the chains), deduplicate to unique tuples,
      and compute empirical cluster weights (counts).
    - Compute product-of-marginals on the sampled support using empirical weights.
    - Stabilize with gamma1, gamma2 (defaults to 1/K) and sum hat_mu * log((hat_mu+g1)/(hat_mu_prod+g2)).

    This mirrors lines 95-111 of listing.tex, but uses empirical counts instead of
    true probabilities for clusters (which are unavailable in our setting).
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

