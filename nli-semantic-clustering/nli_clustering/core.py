"""
Core NLI clustering functionality.

This module implements semantic clustering using Natural Language Inference (NLI)
models. It determines if two text answers are semantically equivalent by checking
bidirectional entailment (mutual entailment).
"""

from typing import List, Dict
import warnings


class NLIClusteringCache:
    """
    Cache for NLI model and clustering results to avoid redundant computations.
    
    This class wraps a pretrained NLI model (default: DeBERTa-MNLI) and provides
    efficient mutual entailment checking with caching for pairwise comparisons.
    
    Attributes:
        model_name: HuggingFace model name
        device: Device to run model on ('cuda' or 'cpu')
        tokenizer: Tokenizer for the NLI model
        model: The NLI model
        _entailment_cache: Cache for pairwise entailment scores
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli", device: str = None):
        """
        Initialize NLI model for semantic clustering.
        
        Args:
            model_name: HuggingFace model identifier for NLI model
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ImportError(
                "NLI clustering requires transformers library. "
                "Install with: pip install transformers torch"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading NLI model for semantic clustering: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        # Cache for pairwise entailment scores
        self._entailment_cache = {}
        
        # Get entailment label ID (DeBERTa-MNLI: 0=contradiction, 1=neutral, 2=entailment)
        self.label2id = self.model.config.label2id
        self.entailment_id = self.label2id.get('entailment', self.label2id.get('ENTAILMENT', 2))
        
        print(f"✓ Model loaded on {device}")
        print(f"  Label mapping: {self.label2id}")
        print(f"  Entailment label ID: {self.entailment_id}")
    
    def check_entailment(
        self,
        premise: str,
        hypothesis: str
    ) -> float:
        """
        Check unidirectional entailment: does premise entail hypothesis?
        
        Args:
            premise: The premise text
            hypothesis: The hypothesis text
        
        Returns:
            Probability of entailment (0.0 to 1.0)
        """
        import torch
        
        # Normalize
        premise = premise.strip().lower()
        hypothesis = hypothesis.strip().lower()
        
        # Check cache
        cache_key = (premise, hypothesis)
        if cache_key in self._entailment_cache:
            return self._entailment_cache[cache_key]
        
        # Compute entailment
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
        
        # Cache result
        self._entailment_cache[cache_key] = entailment_prob
        
        return entailment_prob
    
    def check_mutual_entailment(
        self, 
        text_a: str, 
        text_b: str, 
        threshold: float = 0.5
    ) -> bool:
        """
        Check if two texts are mutually entailed (semantically equivalent).
        
        Two texts are considered semantically equivalent if:
        - text_a entails text_b with probability >= threshold, AND
        - text_b entails text_a with probability >= threshold
        
        Args:
            text_a: First text
            text_b: Second text
            threshold: Minimum P(entailment) for mutual entailment (default: 0.5)
        
        Returns:
            True if texts are mutually entailed, False otherwise
        """
        # Normalize for comparison
        text_a = text_a.strip().lower()
        text_b = text_b.strip().lower()
        
        # Identical strings are always mutually entailed
        if text_a == text_b:
            return True
        
        # Check bidirectional entailment
        fwd_score = self.check_entailment(text_a, text_b)
        bwd_score = self.check_entailment(text_b, text_a)
        
        # Mutual entailment if both directions exceed threshold
        return fwd_score >= threshold and bwd_score >= threshold
    
    def get_entailment_scores(
        self,
        text_a: str,
        text_b: str
    ) -> tuple[float, float]:
        """
        Get bidirectional entailment scores without threshold.
        
        Useful for debugging and threshold adjustment.
        
        Args:
            text_a: First text
            text_b: Second text
        
        Returns:
            (forward_score, backward_score) tuple where:
            - forward_score: P(text_a entails text_b)
            - backward_score: P(text_b entails text_a)
        """
        fwd = self.check_entailment(text_a, text_b)
        bwd = self.check_entailment(text_b, text_a)
        return fwd, bwd
    
    def clear_cache(self):
        """Clear the entailment cache to free memory."""
        self._entailment_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return {
            "cache_size": len(self._entailment_cache),
            "model_name": self.model_name,
            "device": self.device
        }


def cluster_answers_by_nli(
    answers: List[str],
    nli_checker: NLIClusteringCache,
    threshold: float = 0.5
) -> Dict[str, str]:
    """
    Cluster answers by NLI mutual entailment, returning mapping to representatives.
    
    Uses greedy clustering: each answer joins the first cluster it's mutually entailed with,
    or creates a new cluster if no match.
    
    Args:
        answers: List of answer strings to cluster
        nli_checker: NLI model cache for entailment checking
        threshold: Threshold for mutual entailment (default: 0.5)
    
    Returns:
        Dictionary mapping each answer to its cluster representative
    
    Example:
        >>> answers = ["Paris", "paris", "The capital is Paris", "London"]
        >>> nli = NLIClusteringCache()
        >>> mapping = cluster_answers_by_nli(answers, nli, threshold=0.5)
        >>> # "Paris" and "paris" likely map to same representative
        >>> # "The capital is Paris" might also map to same cluster
        >>> # "London" creates separate cluster
    """
    if not answers:
        return {}
    
    cluster_representatives = []
    answer_to_representative = {}
    
    for answer in answers:
        # Find matching cluster
        matched = False
        for rep in cluster_representatives:
            if nli_checker.check_mutual_entailment(answer, rep, threshold):
                answer_to_representative[answer] = rep
                matched = True
                break
        
        if not matched:
            # Create new cluster with this answer as representative
            cluster_representatives.append(answer)
            answer_to_representative[answer] = answer
    
    return answer_to_representative


def apply_nli_clustering_to_chains(
    chains: List[List[str]],
    nli_checker: NLIClusteringCache,
    threshold: float = 0.5
) -> List[List[str]]:
    """
    Apply NLI clustering to all answers in all chains.
    
    This maps each answer to its semantic cluster representative, so MI
    computation measures semantic uncertainty rather than string variation.
    
    Args:
        chains: List of chains, each chain is a list of answer strings
        nli_checker: NLI model cache
        threshold: Threshold for mutual entailment (default: 0.5)
    
    Returns:
        Clustered chains where each answer is replaced by its cluster representative
    
    Example:
        >>> chains = [["Paris", "France"], ["paris", "London"]]
        >>> nli = NLIClusteringCache()
        >>> clustered = apply_nli_clustering_to_chains(chains, nli, 0.5)
        >>> # "Paris" and "paris" map to same representative
        >>> # Result: [["Paris", "France"], ["Paris", "London"]]
    """
    # Collect all unique answers across all chains
    all_answers = set()
    for chain in chains:
        all_answers.update(chain)
    
    # Build clustering mapping
    answer_to_rep = cluster_answers_by_nli(list(all_answers), nli_checker, threshold)
    
    # Apply mapping to all chains
    clustered_chains = []
    for chain in chains:
        clustered_chain = [answer_to_rep.get(ans, ans) for ans in chain]
        clustered_chains.append(clustered_chain)
    
    return clustered_chains


def apply_nli_clustering_to_marginal(
    marginal: Dict[str, float],
    nli_checker: NLIClusteringCache,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Apply NLI clustering to marginal distribution before answer selection.
    
    Groups semantically equivalent answers and sums their probabilities.
    
    Args:
        marginal: Dictionary mapping answer string to probability
        nli_checker: NLI model cache
        threshold: Threshold for mutual entailment (default: 0.5)
    
    Returns:
        Clustered marginal distribution with merged probabilities
    
    Example:
        >>> marginal = {"Paris": 0.3, "paris": 0.2, "London": 0.5}
        >>> nli = NLIClusteringCache()
        >>> clustered = apply_nli_clustering_to_marginal(marginal, nli, 0.5)
        >>> # Result: {"Paris": 0.5, "London": 0.5}  (Paris and paris merged)
    """
    if not marginal:
        return marginal
    
    # Build clustering
    answers = list(marginal.keys())
    answer_to_rep = cluster_answers_by_nli(answers, nli_checker, threshold)
    
    # Merge probabilities by cluster
    clustered_marginal = {}
    for answer, prob in marginal.items():
        rep = answer_to_rep[answer]
        clustered_marginal[rep] = clustered_marginal.get(rep, 0.0) + prob
    
    return clustered_marginal

