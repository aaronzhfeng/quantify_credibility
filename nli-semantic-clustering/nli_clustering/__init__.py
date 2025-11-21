"""
NLI Semantic Clustering Module

This module provides Natural Language Inference (NLI) based semantic clustering
for open-ended question answering systems. It groups semantically equivalent
answers together, enabling better uncertainty quantification and calibration.

Main components:
- NLIClusteringCache: Core NLI model with caching
- cluster_answers_by_nli: Cluster answers by semantic equivalence
- apply_nli_clustering_to_chains: Apply clustering to answer chains
- apply_nli_clustering_to_marginal: Apply clustering to probability distributions
"""

from .core import (
    NLIClusteringCache,
    cluster_answers_by_nli,
    apply_nli_clustering_to_chains,
    apply_nli_clustering_to_marginal,
)

__version__ = "0.1.0"
__all__ = [
    "NLIClusteringCache",
    "cluster_answers_by_nli",
    "apply_nli_clustering_to_chains",
    "apply_nli_clustering_to_marginal",
]

