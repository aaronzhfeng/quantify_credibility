from __future__ import annotations
import numpy as np
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import Counter
from scipy.stats import entropy as scipy_entropy

from .mi_estimator import estimate_mi_listing_nats, nats_to_bits
from .evaluation import compute_agreement_fraction


@dataclass
class EvaluationResult:
    """Result for a single example."""
    question: str
    predicted: str
    gold: str
    correct: bool
    confidence: float
    mi_score: float
    agreement: float
    chains: List[List[Tuple[str, float]]]  # Chain of (text, logprob) tuples


class LogprobTracker:
    """Track logprob extraction statistics during evaluation."""
    
    def __init__(self):
        self.captured = 0  # Count of successfully extracted logprobs (logprob < 0)
        self.fallback = 0  # Count of fallback cases (logprob == 0.0)
    
    def record(self, logprob: float):
        """Record a single logprob value."""
        if logprob < 0:
            self.captured += 1
        else:
            self.fallback += 1
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics dictionary for metrics."""
        total = self.captured + self.fallback
        return {
            "total_inferences": total,
            "captured": self.captured,
            "fallback": self.fallback,
            "capture_rate": self.captured / total if total > 0 else 0.0
        }
    
    @property
    def total(self) -> int:
        return self.captured + self.fallback


def write_progress(completed_count: int):
    """
    Write progress to file for multi-GPU monitoring.
    
    Checks for MULTI_GPU_PROGRESS_FILE environment variable.
    If present, writes the completed count to that file.
    
    Args:
        completed_count: Number of examples completed so far
    """
    import os
    progress_file = os.environ.get('MULTI_GPU_PROGRESS_FILE')
    if progress_file:
        try:
            with open(progress_file, 'w') as f:
                f.write(str(completed_count))
        except:
            pass  # Silently fail if progress writing fails


def compute_ece(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy.
    
    For calibration measurement, pass the correctness labels (0/1) as labels.
    The bin accuracy is computed as the fraction of correct answers in each
    confidence bin, NOT as (predictions == labels).mean().
    
    Args:
        predictions: Binary array of predictions (0 or 1)
        confidences: Confidence scores (0 to 1)
        labels: Ground truth labels (0 or 1)
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


def mi_to_confidence(mi_score: float, method: str = "inverse") -> float:
    """
    Convert MI score to confidence (0 to 1).
    
    Higher MI = higher uncertainty = lower confidence.
    
    Args:
        mi_score: Mutual information in nats
        method: Conversion method
            - "inverse": 1 / (1 + mi)
            - "exp": exp(-mi)
            - "normalized": 1 - (mi / (mi + 1))
    
    Returns:
        Confidence score in [0, 1]
    """
    if method == "inverse":
        return 1.0 / (1.0 + mi_score)
    elif method == "exp":
        return math.exp(-mi_score)
    elif method == "normalized":
        return 1.0 - (mi_score / (mi_score + 1.0))
    else:
        raise ValueError(f"Unknown method: {method}")


def build_pseudo_joint_with_probs(
    chains_with_logprobs: List[List[Tuple[str, float]]],
    n: int = 2
) -> Dict[Tuple[str, ...], float]:
    """
    Build pseudo joint distribution Q̃(Y1, ..., Yn) from chains with logprobs.
    
    Args:
        chains_with_logprobs: List of K chains, each containing n (text, logprob) tuples
        n: Chain length (pseudo joint dimension)
        
    Returns:
        Dictionary mapping (y1, y2, ..., yn) tuples to joint probabilities
    """
    pseudo_joint = {}
    
    for chain in chains_with_logprobs:
        if len(chain) != n:
            continue
            
        # Extract tuple of responses
        response_tuple = tuple(text for text, _ in chain)
        
        # Compute joint probability: P(Y1, Y2, ..., Yn) = P(Y1) * P(Y2|Y1) * ... * P(Yn|Y1...Yn-1)
        # Logprobs are already conditional, so we just sum and exp
        total_logprob = sum(logprob for _, logprob in chain)
        joint_prob = math.exp(total_logprob)
        
        # Accumulate (same tuple might appear in multiple chains due to sampling)
        pseudo_joint[response_tuple] = pseudo_joint.get(response_tuple, 0.0) + joint_prob
    
    # Normalize (though not strictly necessary for marginalization)
    total = sum(pseudo_joint.values())
    if total > 0:
        pseudo_joint = {k: v / total for k, v in pseudo_joint.items()}
    
    return pseudo_joint


def marginalize_to_final_answer(
    pseudo_joint: Dict[Tuple[str, ...], float],
    n: int = 2
) -> Dict[str, float]:
    """
    Marginalize pseudo joint Q̃(Y1, ..., Yn) to get P(Yn).
    
    This gives the distribution over final answers according to the paper's method.
    
    Args:
        pseudo_joint: Pseudo joint distribution
        n: Chain length
        
    Returns:
        Dictionary mapping final answer (Yn) to marginal probability
    """
    marginal = {}
    
    for response_tuple, joint_prob in pseudo_joint.items():
        if len(response_tuple) >= n:
            # Get final answer (last response in chain)
            final_answer = response_tuple[n - 1]
            marginal[final_answer] = marginal.get(final_answer, 0.0) + joint_prob
    
    return marginal


def select_answer_via_pseudo_joint(
    chains_with_logprobs: List[List[Tuple[str, float]]],
    n: int = 2
) -> str:
    """
    Select answer using marginalized pseudo joint distribution (paper's method).
    
    Args:
        chains_with_logprobs: K chains with logprobs
        n: Chain length
        
    Returns:
        Predicted answer (highest marginal probability)
    """
    # Build pseudo joint
    pseudo_joint = build_pseudo_joint_with_probs(chains_with_logprobs, n)
    
    if not pseudo_joint:
        # Fallback to majority vote if no valid chains
        final_answers = [chain[-1][0] for chain in chains_with_logprobs if chain]
        if final_answers:
            return Counter(final_answers).most_common(1)[0][0]
        return ""
    
    # Marginalize to final answer
    marginal = marginalize_to_final_answer(pseudo_joint, n)
    
    # Select answer with highest marginal probability
    if marginal:
        return max(marginal, key=marginal.get)
    
    # Fallback
    return list(pseudo_joint.keys())[0][-1] if pseudo_joint else ""


def run_chain_with_logprobs(
    client,
    query: str,
    n: int,
    temperature: float,
    max_tokens: int,
    prompt_style: str = "naive",
    choices: List[str] = None,
    choice_texts: List[str] = None,
    answer_format: str = "default"
) -> List[Tuple[str, float]]:
    """
    Run a single chain of length n with logprobs for pseudo joint.
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        query: Question
        n: Chain length
        temperature: Sampling temperature
        max_tokens: Max tokens per response
        prompt_style: Prompt style
        choices: MCQ choice letters (e.g., ["A", "B", "C", "D"])
        choice_texts: MCQ choice texts
        
    Returns:
        List of (response_text, logprob) tuples
    """
    from .iterative_prompting import compose_prompt
    
    chain = []
    previous_answers = []
    
    for _ in range(n):
        messages = compose_prompt(query, previous_answers, prompt_style=prompt_style,
                                 choices=choices, choice_texts=choice_texts,
                                 answer_format=answer_format)
        
        # Get response with logprob
        if hasattr(client, 'chat_completion_with_logprobs'):
            response, logprob = client.chat_completion_with_logprobs(
                messages, 
                temperature=temperature, 
                max_tokens=max_tokens
            )
        else:
            # Fallback if logprobs not available
            response = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
            logprob = 0.0
        
        chain.append((response, logprob))
        previous_answers.append(response)
    
    return chain


def evaluate_mcq_greedy_baseline(
    client,
    examples: List,  # MCQExample instances
    max_tokens: int = 64,
    answer_format: str = "default",
    detailed_logger=None,
    offset: int = 0,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate multiple-choice questions using greedy (temperature=0) baseline.
    
    This is the simplest baseline: single greedy decode per question.
    Confidence is based on the logprob of the greedy response.
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        examples: List of MCQExample instances
        max_tokens: Max tokens per generation
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .datasets import match_answer_to_choices
    from .iterative_prompting import compose_prompt
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc="Evaluating (Greedy Baseline)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Single greedy generation
        messages = compose_prompt(ex.question, [], prompt_style="naive",
                                 choices=ex.choices, choice_texts=ex.choice_texts,
                                 answer_format=answer_format)
        
        # Get response with logprob
        if hasattr(client, 'chat_completion_with_logprobs'):
            response, logprob = client.chat_completion_with_logprobs(
                messages, 
                temperature=0.0,  # Greedy
                max_tokens=max_tokens
            )
        else:
            response = client.chat_completion(messages, temperature=0.0, max_tokens=max_tokens)
            logprob = 0.0
        
        # Track logprob
        logprob_tracker.record(logprob)
        
        # Match to MCQ choices
        predicted_choice = match_answer_to_choices(
            response,
            ex.choice_texts,
            ex.choices,
            answer_format=answer_format
        )
        
        # Confidence from logprob (exp of average token logprob)
        confidence = math.exp(logprob) if logprob < 0 else 0.5
        
        # Check correctness (MC2-aware: matches any correct answer)
        from .datasets import is_answer_correct_mc2
        correct = is_answer_correct_mc2(predicted_choice, ex)
        
        # Create a single chain for consistency with MI results
        chain = [(response, logprob)]
        
        result = EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,
            correct=correct,
            confidence=confidence,
            mi_score=0.0,  # No MI for greedy baseline
            agreement=1.0,  # Single sample, perfect agreement
            chains=[chain]
        )
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            method_data = {
                "description": "Single greedy decode (temperature=0)",
                "raw_inputs": [{
                    "prompt": messages,
                    "temperature": 0.0,
                    "max_tokens": max_tokens
                }],
                "raw_outputs": [{
                    "text": response,
                    "logprob": logprob,
                    "probability": math.exp(logprob) if logprob < 0 else 0.5
                }],
                "decision_process": {
                    "selected_text": response,
                    "matched_choice": predicted_choice,
                    "confidence_computation": f"exp(logprob) = exp({logprob:.4f}) = {confidence:.4f}"
                },
                "final_metrics": {
                    "predicted": predicted_choice,
                    "correct": correct,
                    "confidence": confidence,
                    "mi_score": 0.0,
                    "agreement": 1.0
                }
            }
            
            # Format gold answer for MC2 (show all correct answers)
            if ex.metadata and "correct_choices" in ex.metadata:
                num_correct = ex.metadata.get("num_correct", 1)
                correct_choices_str = ", ".join(ex.metadata.get("correct_choices", [ex.answer_key]))
                gold_answer_formatted = f"{correct_choices_str} (MC2: {num_correct} correct answers)"
            else:
                gold_answer_formatted = ex.answer_key
            
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                gold_answer=gold_answer_formatted,
                method_data=method_data
            )
    
    # Compute aggregate metrics
    correct_arr = np.array([r.correct for r in results], dtype=int)
    confidence_arr = np.array([r.confidence for r in results])
    
    accuracy = correct_arr.mean()
    ece = compute_ece(correct_arr, confidence_arr, correct_arr)
    avg_confidence = confidence_arr.mean()
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(avg_confidence),
        "avg_mi_bits": 0.0,
        "avg_agreement": 1.0,
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_mcq_self_consistency(
    client,
    examples: List,  # MCQExample instances
    k: int = 10,
    temperature: float = 0.5,
    max_tokens: int = 64,
    answer_format: str = "default",
    detailed_logger=None,
    offset: int = 0,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate multiple-choice questions using self-consistency baseline.
    
    This baseline:
    1. Generates k samples at temperature > 0
    2. Uses majority voting to select answer
    3. Confidence is the fraction of samples agreeing with majority
    
    Args:
        client: LLM client
        examples: List of MCQExample instances
        k: Number of samples per question
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .datasets import match_answer_to_choices
    from .iterative_prompting import compose_prompt
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc="Evaluating (Self-Consistency)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate k samples
        responses = []
        chains = []
        sample_data = []  # For logging
        
        for sample_idx in range(k):
            messages = compose_prompt(ex.question, [], prompt_style="naive",
                                     choices=ex.choices, choice_texts=ex.choice_texts,
                                     answer_format=answer_format)
            
            if hasattr(client, 'chat_completion_with_logprobs'):
                response, logprob = client.chat_completion_with_logprobs(
                    messages, 
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
                logprob = 0.0
            
            # Track logprob
            logprob_tracker.record(logprob)
            
            responses.append(response)
            chains.append([(response, logprob)])
            
            # Capture for logging
            if detailed_logger:
                sample_data.append({
                    "sample_id": sample_idx,
                    "prompt": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response": response,
                    "logprob": logprob,
                    "probability": math.exp(logprob) if logprob < 0 else 0.5
                })
        
        # Majority voting
        vote_counts = Counter(responses)
        predicted_answer_text, max_count = vote_counts.most_common(1)[0]
        
        # Match to MCQ choices
        predicted_choice = match_answer_to_choices(
            predicted_answer_text,
            ex.choice_texts,
            ex.choices,
            answer_format=answer_format
        )
        
        # Confidence = agreement fraction (fraction voting for majority)
        confidence = max_count / k
        agreement = confidence  # Same as confidence for this method
        
        # Check correctness (MC2-aware: matches any correct answer)
        from .datasets import is_answer_correct_mc2
        correct = is_answer_correct_mc2(predicted_choice, ex)
        
        result = EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,
            correct=correct,
            confidence=confidence,
            mi_score=0.0,  # No MI for self-consistency
            agreement=agreement,
            chains=chains
        )
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            method_data = {
                "description": f"k={k} samples, majority voting",
                "raw_inputs": [{"sample_id": s["sample_id"], "prompt": s["prompt"], "temperature": s["temperature"], "max_tokens": s["max_tokens"]} for s in sample_data],
                "raw_outputs": [{"sample_id": s["sample_id"], "text": s["response"], "logprob": s["logprob"], "probability": s["probability"]} for s in sample_data],
                "decision_process": {
                    "all_responses": responses,
                    "vote_counts": dict(vote_counts),
                    "majority": predicted_answer_text,
                    "matched_choice": predicted_choice,
                    "confidence_computation": f"{max_count}/{k} = {confidence:.4f}"
                },
                "final_metrics": {
                    "predicted": predicted_choice,
                    "correct": correct,
                    "confidence": confidence,
                    "mi_score": 0.0,
                    "agreement": agreement
                }
            }
            
            # Format gold answer for MC2 (show all correct answers)
            if ex.metadata and "correct_choices" in ex.metadata:
                num_correct = ex.metadata.get("num_correct", 1)
                correct_choices_str = ", ".join(ex.metadata.get("correct_choices", [ex.answer_key]))
                gold_answer_formatted = f"{correct_choices_str} (MC2: {num_correct} correct answers)"
            else:
                gold_answer_formatted = ex.answer_key
            
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                gold_answer=gold_answer_formatted,
                method_data=method_data
            )
    
    # Compute aggregate metrics
    correct_arr = np.array([r.correct for r in results], dtype=int)
    confidence_arr = np.array([r.confidence for r in results])
    
    accuracy = correct_arr.mean()
    ece = compute_ece(correct_arr, confidence_arr, correct_arr)
    avg_confidence = confidence_arr.mean()
    avg_agreement = np.mean([r.agreement for r in results])
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(avg_confidence),
        "avg_mi_bits": 0.0,
        "avg_agreement": float(avg_agreement),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def compute_f1_similarity(text1: str, text2: str) -> float:
    """
    Compute F1 similarity score between two texts (from paper).
    
    Uses token-based F1 score: intersection over tokens.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        F1 score between 0 and 1
    """
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()
    
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0
    
    # Count token occurrences (handle repetitions)
    from collections import Counter
    count1 = Counter(tokens1)
    count2 = Counter(tokens2)
    
    # Intersection: minimum count of common tokens
    intersection = sum(min(count1[tok], count2[tok]) for tok in count1 if tok in count2)
    
    if intersection == 0:
        return 0.0
    
    precision = intersection / len(tokens1)
    recall = intersection / len(tokens2)
    
    if precision + recall == 0:
        return 0.0
        
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def group_by_semantic_equivalence(
    samples: List[Tuple[str, float]],
    threshold: float = 0.25
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Group samples by F1 similarity using greedy clustering.
    
    Args:
        samples: List of (text, probability) tuples
        threshold: F1 similarity threshold for grouping
        
    Returns:
        Dict mapping cluster_id -> list of (text, prob) tuples
    """
    clusters = {}
    cluster_representatives = {}
    next_cluster_id = 0
    
    for text, prob in samples:
        # Find matching cluster
        matched_cluster = None
        for cluster_id, rep_text in cluster_representatives.items():
            if compute_f1_similarity(text, rep_text) >= threshold:
                matched_cluster = cluster_id
                break
        
        if matched_cluster is not None:
            # Add to existing cluster
            clusters[matched_cluster].append((text, prob))
        else:
            # Create new cluster
            clusters[next_cluster_id] = [(text, prob)]
            cluster_representatives[next_cluster_id] = text
            next_cluster_id += 1
    
    return clusters


def evaluate_mcq_semantic_entropy(
    client,
    examples: List,  # MCQExample instances
    k: int = 10,
    temperature: float = 0.9,
    max_tokens: int = 64,
    similarity_threshold: float = 0.25,
    answer_format: str = "default",
    detailed_logger=None,
    offset: int = 0,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate MCQ using Semantic Entropy method (Kuhn et al. 2023).
    
    This method:
    1. Generates k samples at temperature > 0 with logprobs
    2. Groups semantically equivalent responses using F1 similarity
    3. Aggregates probabilities within each group
    4. Calculates entropy of the grouped distribution
    5. Confidence = exp(-entropy)
    
    Args:
        client: LLM client
        examples: List of MCQExample instances
        k: Number of samples per question
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        similarity_threshold: F1 threshold for semantic equivalence (paper uses 0.25)
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .datasets import match_answer_to_choices
    from .iterative_prompting import compose_prompt
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc="Evaluating (Semantic Entropy)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate k samples with probabilities
        samples = []
        chains = []
        
        for _ in range(k):
            messages = compose_prompt(ex.question, [], prompt_style="naive",
                                     choices=ex.choices, choice_texts=ex.choice_texts,
                                     answer_format=answer_format)
            
            if hasattr(client, 'chat_completion_with_logprobs'):
                response, logprob = client.chat_completion_with_logprobs(
                    messages, 
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                prob = math.exp(logprob)  # Convert logprob to probability
            else:
                response = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
                prob = 1.0 / k  # Uniform if no logprobs
                logprob = math.log(prob)
            
            # Track logprob
            logprob_tracker.record(logprob)
            
            samples.append((response, prob))
            chains.append([(response, logprob)])
        
        # Group by semantic equivalence using F1 similarity
        clusters = group_by_semantic_equivalence(samples, threshold=similarity_threshold)
        
        # Aggregate probabilities for each cluster
        cluster_distributions = {}
        for cluster_id, cluster_samples in clusters.items():
            # Use first text as representative
            representative_text = cluster_samples[0][0]
            # Sum probabilities
            total_prob = sum(prob for _, prob in cluster_samples)
            cluster_distributions[representative_text] = total_prob
        
        # Normalize to get probability distribution
        total_prob = sum(cluster_distributions.values())
        if total_prob > 0:
            distribution = {k: v/total_prob for k, v in cluster_distributions.items()}
        else:
            distribution = {list(cluster_distributions.keys())[0]: 1.0} if cluster_distributions else {}
        
        # Select answer with highest probability
        if distribution:
            predicted_answer_text = max(distribution, key=distribution.get)
        else:
            # Fallback
            predicted_answer_text = samples[0][0] if samples else ""
        
        # Match to MCQ choices
        predicted_choice = match_answer_to_choices(
            predicted_answer_text,
            ex.choice_texts,
            ex.choices,
            answer_format=answer_format
        )
        
        # Calculate entropy (in nats, then convert to bits for consistency)
        probs = np.array(list(distribution.values()))
        if len(probs) > 0 and np.sum(probs) > 0:
            entropy_nats = scipy_entropy(probs, base=np.e)
        else:
            entropy_nats = 0.0
        
        entropy_bits = entropy_nats / math.log(2)  # Convert to bits
        
        # Confidence from entropy (lower entropy = higher confidence)
        confidence = math.exp(-entropy_nats)
        
        # Agreement = probability of predicted answer
        agreement = distribution.get(predicted_answer_text, 0.0)
        
        # Check correctness
        correct = (predicted_choice == ex.answer_key)
        
        results.append(EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,
            correct=correct,
            confidence=confidence,
            mi_score=entropy_bits,  # Store entropy in mi_score field
            agreement=agreement,
            chains=chains
        ))
        write_progress(len(results))
    
    # Compute aggregate metrics
    correct_arr = np.array([r.correct for r in results], dtype=int)
    confidence_arr = np.array([r.confidence for r in results])
    
    accuracy = correct_arr.mean()
    ece = compute_ece(correct_arr, confidence_arr, correct_arr)
    avg_confidence = confidence_arr.mean()
    avg_entropy = np.mean([r.mi_score for r in results])
    avg_agreement = np.mean([r.agreement for r in results])
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(avg_confidence),
        "avg_mi_bits": float(avg_entropy),  # Actually entropy, but using same field
        "avg_agreement": float(avg_agreement),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_mcq_self_verification(
    client,
    examples: List,  # MCQExample instances
    k: int = 10,
    temperature: float = 0.9,
    max_tokens: int = 64,
    answer_format: str = "default",
    detailed_logger=None,
    offset: int = 0,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate MCQ using Self-Verification method (from paper).
    
    This method:
    1. Generates k samples, selects best answer (highest aggregated prob)
    2. Asks model to verify: "Is this answer correct? True or False"
    3. Confidence = P(True) / (P(True) + P(False))
    
    Args:
        client: LLM client
        examples: List of MCQExample instances
        k: Number of samples for initial selection
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .datasets import match_answer_to_choices
    from .iterative_prompting import compose_prompt
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc="Evaluating (Self-Verification)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Step 1: Generate k samples to find best answer
        samples = []
        chains = []
        
        for _ in range(k):
            messages = compose_prompt(ex.question, [], prompt_style="naive",
                                     choices=ex.choices, choice_texts=ex.choice_texts,
                                     answer_format=answer_format)
            
            if hasattr(client, 'chat_completion_with_logprobs'):
                response, logprob = client.chat_completion_with_logprobs(
                    messages, 
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                prob = math.exp(logprob)
            else:
                response = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
                prob = 1.0
                logprob = 0.0
            
            # Track logprob
            logprob_tracker.record(logprob)
            
            samples.append((response, prob))
            chains.append([(response, logprob)])
        
        # Aggregate probabilities by matched choice
        choice_probs = {}
        choice_to_text = {}
        for text, prob in samples:
            choice = match_answer_to_choices(text, ex.choice_texts, ex.choices, answer_format=answer_format)
            choice_probs[choice] = choice_probs.get(choice, 0.0) + prob
            if choice not in choice_to_text:
                choice_to_text[choice] = text
        
        # Select answer with highest aggregated probability
        if choice_probs:
            predicted_choice = max(choice_probs, key=choice_probs.get)
            predicted_text = choice_to_text[predicted_choice]
        else:
            predicted_choice = ex.choices[0]
            predicted_text = samples[0][0] if samples else ""
        
        # Step 2: Self-verification
        verification_prompt = [{
            "role": "user",
            "content": f"""Consider the following question:
Q: {ex.question}

One answer to question Q is: {predicted_text}

Is the above answer to question Q correct? Answer True or False.
A:"""
        }]
        
        # Get model's verification
        if hasattr(client, 'chat_completion_with_logprobs'):
            verification_response, verification_logprob = client.chat_completion_with_logprobs(
                verification_prompt,
                temperature=0.0,  # Greedy for verification
                max_tokens=5
            )
            # Track verification logprob
            logprob_tracker.record(verification_logprob)
            
            # Parse response for True/False
            response_lower = verification_response.strip().lower()
            
            # Simple heuristic for confidence based on response
            if 'true' in response_lower:
                confidence = 0.9  # High confidence if says True
            elif 'false' in response_lower:
                confidence = 0.1  # Low confidence if says False  
            else:
                confidence = 0.5  # Uncertain if unclear
        else:
            confidence = 0.5  # Fallback
        
        # Agreement = aggregated probability of predicted choice
        total_prob = sum(choice_probs.values())
        agreement = choice_probs.get(predicted_choice, 0.0) / total_prob if total_prob > 0 else 0.0
        
        # Check correctness
        correct = (predicted_choice == ex.answer_key)
        
        results.append(EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,
            correct=correct,
            confidence=confidence,
            mi_score=0.0,  # No MI for self-verification
            agreement=agreement,
            chains=chains
        ))
        write_progress(len(results))
    
    # Compute aggregate metrics
    correct_arr = np.array([r.correct for r in results], dtype=int)
    confidence_arr = np.array([r.confidence for r in results])
    
    accuracy = correct_arr.mean()
    ece = compute_ece(correct_arr, confidence_arr, correct_arr)
    avg_confidence = confidence_arr.mean()
    avg_agreement = np.mean([r.agreement for r in results])
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(avg_confidence),
        "avg_mi_bits": 0.0,
        "avg_agreement": float(avg_agreement),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_mcq_with_mi(
    client,
    examples: List,  # MCQExample instances
    k: int = 10,
    n: int = 2,
    temperature: float = 0.5,
    max_tokens: int = 64,
    mi_method: str = "listing",
    confidence_method: str = "inverse",
    offset: int = 0,
    answer_format: str = "default",
    detailed_logger=None,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate multiple-choice questions using MI-based method with pseudo joint.
    
    This implements the paper's approach:
    1. Generate K chains of length n with logprobs
    2. Build pseudo joint distribution Q̃(Y1, ..., Yn)
    3. Marginalize to get P(Yn) for final answer selection
    4. Compute MI from pseudo joint for confidence
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        examples: List of MCQExample instances
        k: Number of independent chains per question (default 10, from paper)
        n: Chain length / pseudo joint dimension (default 2, from paper)
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        mi_method: MI estimator ("plugin" or "listing")
        confidence_method: How to convert MI to confidence
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .mi_estimator import estimate_mi_nats
    from .datasets import match_answer_to_choices
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc="Evaluating") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate K chains of length n with logprobs
        chains_with_logprobs = []
        for _ in range(k):
            chain = run_chain_with_logprobs(
                client=client,
                query=ex.question,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_style="naive",
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                answer_format=answer_format
            )
            chains_with_logprobs.append(chain)
            
            # Track logprobs from this chain
            for _, logprob in chain:
                logprob_tracker.record(logprob)
        
        # Select answer using marginalized pseudo joint (paper's method)
        predicted_answer_text = select_answer_via_pseudo_joint(chains_with_logprobs, n)
        
        # Match to MCQ choices
        predicted_choice = match_answer_to_choices(
            predicted_answer_text,
            ex.choice_texts,
            ex.choices,
            answer_format=answer_format
        )
        
        # Compute MI for uncertainty estimation
        # Extract just the text responses for MI computation
        chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
        
        if mi_method == "listing":
            mi_nats = estimate_mi_listing_nats(chains_text)
        else:
            mi_nats = estimate_mi_nats(chains_text)
        
        mi_bits = nats_to_bits(mi_nats)
        
        # Get final answers for agreement computation
        final_answers = [chain[-1][0] for chain in chains_with_logprobs]
        agreement = compute_agreement_fraction(final_answers)
        
        # Convert MI to confidence
        confidence = mi_to_confidence(mi_nats, method=confidence_method)
        
        # Check correctness
        correct = (predicted_choice == ex.answer_key)
        
        result = EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,
            correct=correct,
            confidence=confidence,
            mi_score=mi_bits,
            agreement=agreement,
            chains=chains_with_logprobs
        )
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            from .iterative_prompting import compose_prompt
            
            # Capture chain data for logging - need to reconstruct prompts
            raw_inputs = []
            raw_outputs = []
            for chain_idx, chain in enumerate(chains_with_logprobs):
                # Reconstruct the prompts for this chain
                previous_answers = []
                for step_idx, (response, logprob) in enumerate(chain):
                    # Generate the prompt that was used for this step
                    prompt = compose_prompt(
                        ex.question, 
                        previous_answers,
                        prompt_style="naive",
                        choices=ex.choices,
                        choice_texts=ex.choice_texts,
                        answer_format=answer_format
                    )
                    
                    raw_inputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                    raw_outputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "text": response,
                        "logprob": logprob,
                        "probability": math.exp(logprob) if logprob < 0 else 0.5
                    })
                    
                    # Add this response to previous_answers for next step
                    previous_answers.append(response)
            
            method_data = {
                "description": f"k={k} chains of length n={n}, MI estimation",
                "raw_inputs": raw_inputs,
                "raw_outputs": raw_outputs,
                "decision_process": {
                    "num_chains": k,
                    "chain_length": n,
                    "total_inferences": k * n,
                    "mi_method": mi_method,
                    "mi_nats": mi_nats,
                    "mi_bits": mi_bits,
                    "selected_answer": predicted_answer_text,
                    "matched_choice": predicted_choice,
                    "confidence_computation": f"1/(1 + mi_nats) = 1/(1 + {mi_nats:.4f}) = {confidence:.4f}",
                    "agreement": f"{agreement:.2%} ({agreement*k:.0f}/{k} chains agree on final answer)"
                },
                "final_metrics": {
                    "predicted": predicted_choice,
                    "correct": correct,
                    "confidence": confidence,
                    "mi_score": mi_bits,
                    "agreement": agreement
                }
            }
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                gold_answer=ex.answer_key,
                method_data=method_data
            )
    
    # Compute aggregate metrics
    correct_arr = np.array([r.correct for r in results], dtype=int)
    confidence_arr = np.array([r.confidence for r in results])
    
    accuracy = correct_arr.mean()
    ece = compute_ece(correct_arr, confidence_arr, correct_arr)
    avg_confidence = confidence_arr.mean()
    avg_mi = np.mean([r.mi_score for r in results])
    avg_agreement = np.mean([r.agreement for r in results])
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(avg_confidence),
        "avg_mi_bits": float(avg_mi),
        "avg_agreement": float(avg_agreement),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_extractive_qa_greedy(
    client,
    examples: List,  # ExtractiveQAExample instances
    max_tokens: int = 50,
    prompt_composer=None,  # Optional: compose_prompt_extractive or compose_prompt_trivia
    dataset_name: str = "Extractive QA",
    detailed_logger=None,
    offset: int = 0,
    verbose: bool = True
) -> Tuple[Dict[str, float], List]:
    """
    Evaluate extractive QA using greedy (temperature=0) baseline.
    
    This is the simplest baseline: single greedy decode per question.
    Confidence is based on the logprob of the greedy response.
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        examples: List of ExtractiveQAExample instances
        max_tokens: Max tokens per generation
        prompt_composer: Function to compose prompts (compose_prompt_extractive or compose_prompt_trivia)
        dataset_name: Name for logging/display
        detailed_logger: Optional logger for saving traces
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .datasets import compute_exact_match, compute_f1_score
    from .iterative_prompting import compose_prompt_extractive
    
    # Default to extractive prompt if not provided
    if prompt_composer is None:
        prompt_composer = compose_prompt_extractive
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc=f"Evaluating {dataset_name} (Greedy)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Single greedy generation
        if prompt_composer == compose_prompt_extractive:
            messages = prompt_composer(
                question=ex.question,
                context=ex.context,
                previous_answers=[],
                answer_format="strict"
            )
        else:  # compose_prompt_trivia
            messages = prompt_composer(
                question=ex.question,
                previous_answers=[],
                answer_format="strict"
            )
        
        # Get response with logprob
        if hasattr(client, 'chat_completion_with_logprobs'):
            response, logprob = client.chat_completion_with_logprobs(
                messages, 
                temperature=0.0,  # Greedy
                max_tokens=max_tokens
            )
        else:
            response = client.chat_completion(messages, temperature=0.0, max_tokens=max_tokens)
            logprob = 0.0
        
        # Track logprob
        logprob_tracker.record(logprob)
        
        # Confidence from logprob
        confidence = np.exp(logprob) if logprob < 0 else 0.5
        
        # Evaluate with extractive QA metrics
        exact_match = compute_exact_match(response, ex.answers)
        f1 = compute_f1_score(response, ex.answers)
        
        result = {
            "id": ex.id,
            "question": ex.question,
            "predicted": response,
            "gold_answers": ex.answers,
            "exact_match": exact_match,
            "f1": f1,
            "confidence": confidence,
            "mi_score": 0.0,  # No MI for greedy baseline
            "agreement": 1.0,  # Single sample, perfect agreement
        }
        if hasattr(ex, 'is_impossible'):
            result["is_impossible"] = ex.is_impossible
        
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            method_data = {
                "method_name": f"{dataset_name.lower()}_greedy",
                "method_description": "Single greedy decode (temperature=0)",
                "raw_inputs": [{
                    "prompt": messages,
                    "temperature": 0.0,
                    "max_tokens": max_tokens
                }],
                "raw_outputs": [{
                    "text": response,
                    "logprob": logprob,
                    "probability": confidence
                }],
                "decision_process": {
                    "method": "greedy",
                    "temperature": 0.0,
                    "selected_answer": response,
                    "confidence_from_logprob": f"{confidence:.4f}"
                },
                "final_metrics": {
                    "predicted": response,
                    "gold_answers": ex.answers,
                    "exact_match": exact_match,
                    "f1": f1,
                    "confidence": float(confidence),
                    "mi_score": 0.0,
                    "agreement": 1.0
                }
            }
            
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=[],
                choice_texts=[],
                gold_answer=str(ex.answers),
                method_data=method_data
            )
    
    # Compute aggregate metrics
    exact_match_arr = np.array([r["exact_match"] for r in results])
    f1_arr = np.array([r["f1"] for r in results])
    confidence_arr = np.array([r["confidence"] for r in results])
    
    # ECE using exact match as correctness
    ece = compute_ece(exact_match_arr, confidence_arr, exact_match_arr)
    
    metrics = {
        "exact_match": float(exact_match_arr.mean()),
        "f1": float(f1_arr.mean()),
        "ece": float(ece),
        "avg_confidence": float(confidence_arr.mean()),
        "avg_mi_bits": 0.0,
        "avg_agreement": 1.0,
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_extractive_qa_self_consistency(
    client,
    examples: List,  # ExtractiveQAExample instances
    k: int = 10,
    temperature: float = 0.9,
    max_tokens: int = 50,
    prompt_composer=None,  # Optional: compose_prompt_extractive or compose_prompt_trivia
    dataset_name: str = "Extractive QA",
    detailed_logger=None,
    offset: int = 0,
    verbose: bool = True
) -> Tuple[Dict[str, float], List]:
    """
    Evaluate extractive QA using self-consistency baseline.
    
    This baseline:
    1. Generates k samples at temperature > 0
    2. Normalizes answers and uses majority voting
    3. Confidence is the fraction of samples agreeing with majority
    
    Args:
        client: LLM client
        examples: List of ExtractiveQAExample instances
        k: Number of samples per question
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        prompt_composer: Function to compose prompts
        dataset_name: Name for logging/display
        detailed_logger: Optional logger for saving traces
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from collections import Counter
    from .datasets import compute_exact_match, compute_f1_score, normalize_answer
    from .iterative_prompting import compose_prompt_extractive
    
    # Default to extractive prompt if not provided
    if prompt_composer is None:
        prompt_composer = compose_prompt_extractive
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc=f"Evaluating {dataset_name} (Self-Consistency)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate k samples
        responses = []
        sample_data = []  # For logging
        
        for sample_idx in range(k):
            if prompt_composer == compose_prompt_extractive:
                messages = prompt_composer(
                    question=ex.question,
                    context=ex.context,
                    previous_answers=[],
                    answer_format="strict"
                )
            else:  # compose_prompt_trivia
                messages = prompt_composer(
                    question=ex.question,
                    previous_answers=[],
                    answer_format="strict"
                )
            
            if hasattr(client, 'chat_completion_with_logprobs'):
                response, logprob = client.chat_completion_with_logprobs(
                    messages, 
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
                logprob = 0.0
            
            # Track logprob
            logprob_tracker.record(logprob)
            
            responses.append(response)
            
            # Capture for logging
            if detailed_logger:
                sample_data.append({
                    "sample_id": sample_idx,
                    "prompt": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response": response,
                    "logprob": logprob,
                    "probability": np.exp(logprob) if logprob < 0 else 0.5
                })
        
        # Majority voting on NORMALIZED answers
        normalized_responses = [normalize_answer(r) for r in responses]
        vote_counts = Counter(normalized_responses)
        predicted_normalized, max_count = vote_counts.most_common(1)[0]
        
        # Get original (non-normalized) answer corresponding to the majority
        # Find first response that normalizes to the majority answer
        predicted_answer = responses[normalized_responses.index(predicted_normalized)]
        
        # Confidence = agreement fraction (fraction voting for majority)
        confidence = max_count / k
        agreement = confidence  # Same as confidence for this method
        
        # Evaluate with extractive QA metrics
        exact_match = compute_exact_match(predicted_answer, ex.answers)
        f1 = compute_f1_score(predicted_answer, ex.answers)
        
        result = {
            "id": ex.id,
            "question": ex.question,
            "predicted": predicted_answer,
            "gold_answers": ex.answers,
            "exact_match": exact_match,
            "f1": f1,
            "confidence": confidence,
            "mi_score": 0.0,  # No MI for self-consistency
            "agreement": agreement,
        }
        if hasattr(ex, 'is_impossible'):
            result["is_impossible"] = ex.is_impossible
        
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            method_data = {
                "method_name": f"{dataset_name.lower()}_self_consistency",
                "method_description": f"k={k} samples with majority voting (temperature={temperature})",
                "raw_inputs": [s["prompt"] for s in sample_data],
                "raw_outputs": [{
                    "sample_id": s["sample_id"],
                    "text": s["response"],
                    "logprob": s["logprob"],
                    "probability": s["probability"]
                } for s in sample_data],
                "decision_process": {
                    "method": "self-consistency",
                    "k": k,
                    "temperature": temperature,
                    "vote_distribution": dict(vote_counts),
                    "selected_answer": predicted_answer,
                    "majority_count": max_count,
                    "confidence": f"{confidence:.4f} ({max_count}/{k} votes)"
                },
                "final_metrics": {
                    "predicted": predicted_answer,
                    "gold_answers": ex.answers,
                    "exact_match": exact_match,
                    "f1": f1,
                    "confidence": float(confidence),
                    "mi_score": 0.0,
                    "agreement": float(agreement)
                }
            }
            
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=[],
                choice_texts=[],
                gold_answer=str(ex.answers),
                method_data=method_data
            )
    
    # Compute aggregate metrics
    exact_match_arr = np.array([r["exact_match"] for r in results])
    f1_arr = np.array([r["f1"] for r in results])
    confidence_arr = np.array([r["confidence"] for r in results])
    
    # ECE using exact match as correctness
    ece = compute_ece(exact_match_arr, confidence_arr, exact_match_arr)
    
    metrics = {
        "exact_match": float(exact_match_arr.mean()),
        "f1": float(f1_arr.mean()),
        "ece": float(ece),
        "avg_confidence": float(confidence_arr.mean()),
        "avg_mi_bits": 0.0,
        "avg_agreement": float(sum(r["agreement"] for r in results) / len(results)),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_extractive_qa_with_mi(
    client,
    examples: List,  # ExtractiveQAExample instances
    k: int = 10,
    n: int = 2,
    temperature: float = 0.9,
    max_tokens: int = 50,
    mi_method: str = "listing",
    confidence_method: str = "inverse",
    offset: int = 0,
    detailed_logger=None,
    verbose: bool = True,
    use_nli_clustering: bool = False,
    nli_threshold: float = 0.5,
    nli_model: str = "microsoft/deberta-v2-xlarge-mnli"
) -> Tuple[Dict[str, float], List]:
    """
    Evaluate extractive QA (SQuAD-style) using MI method.
    
    Similar to evaluate_mcq_with_mi but adapted for extractive answers.
    Uses exact match and F1 score instead of MCQ accuracy.
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        examples: List of ExtractiveQAExample instances
        k: Number of independent chains per question (default 10, from paper)
        n: Chain length / pseudo joint dimension (default 2, from paper)
        temperature: Sampling temperature
        max_tokens: Max tokens per generation (50 for extractive answers)
        mi_method: MI estimator ("plugin" or "listing")
        confidence_method: How to convert MI to confidence
        detailed_logger: Optional logger for saving traces
        verbose: Show progress bar
        use_nli_clustering: If True, cluster answers semantically before MI computation
        nli_threshold: Threshold for NLI mutual entailment (default 0.5)
        nli_model: NLI model to use for clustering
        
    Returns:
        (metrics_dict, results_list)
        metrics include: exact_match, f1, ece, avg_confidence, avg_mi_bits
    """
    from tqdm import tqdm
    from .mi_estimator import estimate_mi_listing_nats, estimate_mi_nats, nats_to_bits
    from .datasets import compute_exact_match, compute_f1_score
    from .iterative_prompting import compose_prompt_extractive
    from .evaluation import compute_agreement_fraction
    
    # Initialize NLI clustering if enabled
    nli_checker = None
    if use_nli_clustering:
        nli_checker = NLIClusteringCache(model_name=nli_model)
        print(f"✓ NLI clustering enabled (threshold={nli_threshold})")
    
    results = []
    logprob_tracker = LogprobTracker()
    desc = "Evaluating (Extractive QA + MI + NLI)" if use_nli_clustering else "Evaluating (Extractive QA + MI)"
    iterator = tqdm(examples, desc=desc) if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate K chains of length n with logprobs
        chains_with_logprobs = []
        for _ in range(k):
            chain = []
            previous_answers = []
            
            for _ in range(n):
                messages = compose_prompt_extractive(
                    question=ex.question,
                    context=ex.context,
                    previous_answers=previous_answers,
                    answer_format="strict"
                )
                
                if hasattr(client, 'chat_completion_with_logprobs'):
                    response, logprob = client.chat_completion_with_logprobs(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    response = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
                    logprob = 0.0
                
                # Track logprob
                logprob_tracker.record(logprob)
                chain.append((response, logprob))
                previous_answers.append(response)
            
            chains_with_logprobs.append(chain)
        
        # Select answer via pseudo-joint marginalization
        predicted_answer = select_answer_via_pseudo_joint(chains_with_logprobs, n)
        
        # Compute MI for uncertainty estimation
        chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
        
        # Apply NLI clustering to chains before MI computation (if enabled)
        if use_nli_clustering and nli_checker:
            chains_for_mi = apply_nli_clustering_to_chains(chains_text, nli_checker, nli_threshold)
        else:
            chains_for_mi = chains_text
        
        if mi_method == "listing":
            mi_nats = estimate_mi_listing_nats(chains_for_mi)
        else:
            mi_nats = estimate_mi_nats(chains_for_mi)
        
        mi_bits = nats_to_bits(mi_nats)
        
        # Get final answers for agreement computation
        final_answers = [chain[-1][0] for chain in chains_with_logprobs]
        agreement = compute_agreement_fraction(final_answers)
        
        # Convert MI to confidence
        confidence = mi_to_confidence(mi_nats, method=confidence_method)
        
        # Evaluate with SQuAD metrics
        exact_match = compute_exact_match(predicted_answer, ex.answers)
        f1 = compute_f1_score(predicted_answer, ex.answers)
        
        result = {
            "id": ex.id,
            "question": ex.question,
            "predicted": predicted_answer,
            "gold_answers": ex.answers,
            "exact_match": exact_match,
            "f1": f1,
            "confidence": confidence,
            "mi_score": mi_bits,
            "agreement": agreement,
            "is_impossible": ex.is_impossible
        }
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            # Capture chain data for logging
            raw_inputs = []
            raw_outputs = []
            for chain_idx, chain in enumerate(chains_with_logprobs):
                previous_answers = []
                for step_idx, (response, logprob) in enumerate(chain):
                    # Generate the prompt that was used for this step
                    prompt = compose_prompt_extractive(
                        question=ex.question,
                        context=ex.context,
                        previous_answers=previous_answers,
                        answer_format="strict"
                    )
                    
                    raw_inputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                    raw_outputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "text": response,
                        "logprob": logprob,
                        "probability": math.exp(logprob) if logprob < 0 else 0.5
                    })
                    
                    previous_answers.append(response)
            
            method_data = {
                "description": f"k={k} chains of length n={n}, MI estimation (Extractive QA)",
                "raw_inputs": raw_inputs,
                "raw_outputs": raw_outputs,
                "decision_process": {
                    "num_chains": k,
                    "chain_length": n,
                    "total_inferences": k * n,
                    "mi_method": mi_method,
                    "mi_nats": mi_nats,
                    "mi_bits": mi_bits,
                    "selected_answer": predicted_answer,
                    "confidence_computation": f"1/(1 + mi_nats) = 1/(1 + {mi_nats:.4f}) = {confidence:.4f}",
                    "agreement": f"{agreement:.2%} ({agreement*k:.0f}/{k} chains agree on final answer)"
                },
                "final_metrics": {
                    "predicted": predicted_answer,
                    "gold_answers": ex.answers,
                    "exact_match": exact_match,
                    "f1": f1,
                    "confidence": confidence,
                    "mi_score": mi_bits,
                    "agreement": agreement,
                    "is_impossible": ex.is_impossible
                }
            }
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=[],  # No choices for extractive QA
                choice_texts=[],
                gold_answer=str(ex.answers),  # Convert list to string
                method_data=method_data
            )
    
    # Compute aggregate metrics
    exact_match_arr = np.array([r["exact_match"] for r in results])
    f1_arr = np.array([r["f1"] for r in results])
    confidence_arr = np.array([r["confidence"] for r in results])
    
    # ECE using exact match as correctness
    ece = compute_ece(exact_match_arr, confidence_arr, exact_match_arr)
    
    metrics = {
        "exact_match": float(exact_match_arr.mean()),
        "f1": float(f1_arr.mean()),
        "ece": float(ece),
        "avg_confidence": float(confidence_arr.mean()),
        "avg_mi_bits": float(sum(r["mi_score"] for r in results) / len(results)),
        "avg_agreement": float(sum(r["agreement"] for r in results) / len(results)),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_triviaqa_with_mi(
    client,
    examples: List,  # ExtractiveQAExample instances (from TriviaQA)
    k: int = 10,
    n: int = 2,
    temperature: float = 0.9,
    max_tokens: int = 50,
    mi_method: str = "listing",
    confidence_method: str = "inverse",
    offset: int = 0,
    detailed_logger=None,
    verbose: bool = True,
    use_nli_clustering: bool = False,
    nli_threshold: float = 0.5,
    nli_model: str = "microsoft/deberta-v2-xlarge-mnli"
) -> Tuple[Dict[str, float], List]:
    """
    Evaluate TriviaQA using MI method with correctness-based MI.
    
    Similar to evaluate_extractive_qa_with_mi but uses trivia-specific prompts
    (no context) and maps chains to binary correctness for MI computation.
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        examples: List of ExtractiveQAExample instances (from load_triviaqa)
        k: Number of independent chains per question
        n: Chain length / pseudo joint dimension
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        mi_method: MI estimator ("plugin" or "listing")
        confidence_method: How to convert MI to confidence
        detailed_logger: Optional logger for saving traces
        verbose: Show progress bar
        use_nli_clustering: If True, cluster answers semantically before correctness mapping
        nli_threshold: Threshold for NLI mutual entailment (default 0.5)
        nli_model: NLI model to use for clustering
        
    Returns:
        (metrics_dict, results_list)
        metrics include: exact_match, f1, ece, avg_confidence, avg_mi_bits
    """
    from tqdm import tqdm
    from .mi_estimator import estimate_mi_listing_nats, estimate_mi_nats, nats_to_bits
    from .datasets import compute_exact_match, compute_f1_score
    from .iterative_prompting import compose_prompt_trivia
    from .evaluation import compute_agreement_fraction
    
    # Initialize NLI clustering if enabled
    nli_checker = None
    if use_nli_clustering:
        nli_checker = NLIClusteringCache(model_name=nli_model)
        print(f"✓ NLI clustering enabled for correctness-based MI (threshold={nli_threshold})")
    
    results = []
    logprob_tracker = LogprobTracker()
    desc = "Evaluating TriviaQA (MI + NLI)" if use_nli_clustering else "Evaluating TriviaQA (MI)"
    iterator = tqdm(examples, desc=desc) if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate K chains of length n
        chains_with_logprobs = []
        for _ in range(k):
            chain = []
            previous_answers = []
            
            for _ in range(n):
                messages = compose_prompt_trivia(
                    question=ex.question,
                    previous_answers=previous_answers,
                    answer_format="strict"
                )
                
                if hasattr(client, 'chat_completion_with_logprobs'):
                    response, logprob = client.chat_completion_with_logprobs(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                else:
                    response = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
                    logprob = 0.0
                
                # Track logprob
                logprob_tracker.record(logprob)
                chain.append((response, logprob))
                previous_answers.append(response)
            
            chains_with_logprobs.append(chain)
        
        # Select answer via pseudo-joint marginalization
        predicted_answer = select_answer_via_pseudo_joint(chains_with_logprobs, n)
        
        # ===== CORRECTNESS-BASED MI =====
        # Transform chains to binary correctness for MI computation
        chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
        
        # Apply NLI clustering BEFORE mapping to correctness (if enabled)
        # This ensures semantically equivalent answers map to same cluster → same correctness
        if use_nli_clustering and nli_checker:
            chains_for_correctness = apply_nli_clustering_to_chains(chains_text, nli_checker, nli_threshold)
        else:
            chains_for_correctness = chains_text
        
        correctness_chains = []
        for chain in chains_for_correctness:
            correctness_chain = []
            for answer_text in chain:
                # Check if answer is correct (matches any alias)
                is_correct = compute_exact_match(answer_text, ex.answers) == 1.0
                correctness_chain.append("correct" if is_correct else "incorrect")
            correctness_chains.append(correctness_chain)
        
        # Compute MI on correctness (binary space)
        if mi_method == "listing":
            mi_nats = estimate_mi_listing_nats(correctness_chains)
        else:
            mi_nats = estimate_mi_nats(correctness_chains)
        
        mi_bits = nats_to_bits(mi_nats)
        
        # Compute correctness agreement
        final_correctness = [chain[-1] for chain in correctness_chains]
        agreement = compute_agreement_fraction(final_correctness)
        
        # Convert MI to confidence
        confidence = mi_to_confidence(mi_nats, method=confidence_method)
        
        # Evaluate with TriviaQA metrics (EM + F1)
        exact_match = compute_exact_match(predicted_answer, ex.answers)
        f1 = compute_f1_score(predicted_answer, ex.answers)
        
        result = {
            "id": ex.id,
            "question": ex.question,
            "predicted": predicted_answer,
            "gold_answers": ex.answers,
            "exact_match": exact_match,
            "f1": f1,
            "confidence": confidence,
            "mi_score": mi_bits,
            "agreement": agreement,  # Agreement on correctness
        }
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            raw_inputs = []
            raw_outputs = []
            for chain_idx, chain in enumerate(chains_with_logprobs):
                previous_answers = []
                for step_idx, (response, logprob) in enumerate(chain):
                    prompt = compose_prompt_trivia(
                        question=ex.question,
                        previous_answers=previous_answers,
                        answer_format="strict"
                    )
                    
                    raw_inputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                    raw_outputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "text": response,
                        "logprob": logprob,
                        "probability": np.exp(logprob) if logprob != 0.0 else 0.5
                    })
                    
                    previous_answers.append(response)
            
            method_data = {
                "method_name": "triviaqa_correctness_mi",
                "method_description": f"k={k} chains of length n={n}, correctness-based MI (TriviaQA)",
                "raw_inputs": raw_inputs,
                "raw_outputs": raw_outputs,
                "decision_process": {
                    "num_chains": k,
                    "chain_length": n,
                    "total_inferences": k * n,
                    "mi_method": mi_method,
                    "mi_nats": float(mi_nats),
                    "mi_bits": float(mi_bits),
                    "selected_answer": predicted_answer,
                    "confidence_computation": f"MI-based ({confidence_method}): {confidence:.4f}",
                    "correctness_agreement": f"{agreement*100:.2f}% ({int(agreement*k)}/{k} chains agree on correctness)"
                },
                "final_metrics": {
                    "predicted": predicted_answer,
                    "gold_answers": ex.answers,
                    "exact_match": exact_match,
                    "f1": f1,
                    "confidence": float(confidence),
                    "mi_score": float(mi_bits),
                    "agreement": float(agreement)
                }
            }
            
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=[],
                choice_texts=[],
                gold_answer=str(ex.answers),
                method_data=method_data
            )
    
    # Compute aggregate metrics
    exact_match_arr = np.array([r["exact_match"] for r in results])
    f1_arr = np.array([r["f1"] for r in results])
    confidence_arr = np.array([r["confidence"] for r in results])
    
    # ECE using exact match as correctness
    ece = compute_ece(exact_match_arr, confidence_arr, exact_match_arr)
    
    metrics = {
        "exact_match": float(exact_match_arr.mean()),
        "f1": float(f1_arr.mean()),
        "ece": float(ece),
        "avg_confidence": float(confidence_arr.mean()),
        "avg_mi_bits": float(sum(r["mi_score"] for r in results) / len(results)),
        "avg_correctness_agreement": float(sum(r["agreement"] for r in results) / len(results)),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_truthfulqa_with_correctness_mi(
    client,
    examples: List,  # MCQExample instances
    k: int = 10,
    n: int = 2,
    temperature: float = 0.5,
    max_tokens: int = 64,
    mi_method: str = "listing",
    confidence_method: str = "inverse",
    offset: int = 0,
    answer_format: str = "default",
    detailed_logger=None,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate TruthfulQA using correctness-based MI instead of choice-based MI.
    
    Key difference from standard MCQ evaluation:
    - Instead of computing MI over answer choices (A, B, C, ...),
    - We compute MI over correctness (correct vs incorrect)
    - This measures uncertainty about WHETHER the answer is truthful,
    - Rather than uncertainty about WHICH answer to choose.
    
    This approach is more appropriate for TruthfulQA because:
    1. Questions can have many answer choices (>26)
    2. The task is to distinguish truthful from false statements
    3. We care about correctness agreement, not choice agreement
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        examples: List of MCQExample instances (TruthfulQA MC1)
        k: Number of independent chains per question
        n: Chain length / pseudo joint dimension
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        mi_method: MI estimator ("plugin" or "listing")
        confidence_method: How to convert MI to confidence
        answer_format: Answer format for prompting
        detailed_logger: Optional logger for detailed traces
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .mi_estimator import estimate_mi_nats, estimate_mi_listing_nats
    from .datasets import match_answer_to_choices
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc="Evaluating TruthfulQA (correctness MI)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate K chains of length n
        chains_with_logprobs = []
        for _ in range(k):
            chain = run_chain_with_logprobs(
                client=client,
                query=ex.question,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_style="naive",
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                answer_format=answer_format
            )
            chains_with_logprobs.append(chain)
            
            # Track logprobs from this chain
            for _, logprob in chain:
                logprob_tracker.record(logprob)
        
        # Select answer using marginalized pseudo joint
        predicted_answer_text = select_answer_via_pseudo_joint(chains_with_logprobs, n)
        
        # Match to MCQ choices
        predicted_choice = match_answer_to_choices(
            predicted_answer_text,
            ex.choice_texts,
            ex.choices,
            answer_format=answer_format
        )
        
        # ===== CORRECTNESS-BASED MI =====
        # Transform chains from choices to binary correctness
        chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
        
        # Map each answer in each chain to "correct" or "incorrect"
        correctness_chains = []
        for chain in chains_text:
            correctness_chain = []
            for answer_text in chain:
                # Match this answer to a choice
                matched_choice = match_answer_to_choices(
                    answer_text,
                    ex.choice_texts,
                    ex.choices,
                    answer_format=answer_format
                )
                # Check if it's correct
                is_correct = (matched_choice == ex.answer_key)
                correctness_chain.append("correct" if is_correct else "incorrect")
            correctness_chains.append(correctness_chain)
        
        # Compute MI on correctness (binary space) instead of choices
        if mi_method == "listing":
            mi_nats = estimate_mi_listing_nats(correctness_chains)
        else:
            mi_nats = estimate_mi_nats(correctness_chains)
        
        mi_bits = nats_to_bits(mi_nats)
        
        # Compute correctness agreement (what fraction agree on correctness)
        final_correctness = [chain[-1] for chain in correctness_chains]
        agreement = compute_agreement_fraction(final_correctness)
        
        # Convert MI to confidence
        confidence = mi_to_confidence(mi_nats, method=confidence_method)
        
        # Check if prediction is correct
        correct = (predicted_choice == ex.answer_key)
        
        result = EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,
            correct=correct,
            confidence=confidence,
            mi_score=mi_bits,
            agreement=agreement,  # Now: agreement on correctness, not on choice
            chains=chains_with_logprobs
        )
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            from .iterative_prompting import compose_prompt
            
            raw_inputs = []
            raw_outputs = []
            for chain_idx, chain in enumerate(chains_with_logprobs):
                previous_answers = []
                for step_idx, (response, logprob) in enumerate(chain):
                    prompt = compose_prompt(
                        ex.question, 
                        previous_answers,
                        prompt_style="naive",
                        choices=ex.choices,
                        choice_texts=ex.choice_texts,
                        answer_format=answer_format
                    )
                    
                    raw_inputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                    
                    raw_outputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "text": response,
                        "logprob": logprob,
                        "probability": np.exp(logprob) if logprob != 0.0 else 0.5
                    })
                    
                    previous_answers.append(response)
            
            # Log the question with correctness-based MI metadata
            method_data = {
                "method_name": "truthfulqa_correctness_mi",
                "method_description": f"k={k} chains of length n={n}, correctness-based MI estimation",
                "raw_inputs": raw_inputs,
                "raw_outputs": raw_outputs,
                "decision_process": {
                    "num_chains": k,
                    "chain_length": n,
                    "total_inferences": k * n,
                    "mi_method": mi_method,
                    "mi_nats": float(mi_nats),
                    "mi_bits": float(mi_bits),
                    "selected_answer": predicted_choice,
                    "matched_choice": predicted_choice,
                    "confidence_computation": f"MI-based ({confidence_method}): {confidence:.4f}",
                    "correctness_agreement": f"{agreement*100:.2f}% ({int(agreement*k)}/{k} chains agree on correctness)"
                },
                "final_metrics": {
                    "predicted": predicted_choice,
                    "correct": correct,
                    "confidence": float(confidence),
                    "mi_score": float(mi_bits),
                    "agreement": float(agreement)
                }
            }
            
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                gold_answer=ex.answer_key,
                method_data=method_data
            )
    
    # Compute overall metrics
    correct_arr = np.array([r.correct for r in results], dtype=float)
    confidence_arr = np.array([r.confidence for r in results], dtype=float)
    
    accuracy = correct_arr.mean()
    ece = compute_ece(correct_arr, confidence_arr, correct_arr)
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(confidence_arr.mean()),
        "avg_mi_bits": float(sum(r.mi_score for r in results) / len(results)),
        "avg_correctness_agreement": float(sum(r.agreement for r in results) / len(results)),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


def evaluate_truthfulqa_mc2_with_correctness_mi(
    client,
    examples: List,  # MCQExample instances with MC2 metadata
    k: int = 10,
    n: int = 2,
    temperature: float = 0.5,
    max_tokens: int = 64,
    mi_method: str = "listing",
    confidence_method: str = "inverse",
    offset: int = 0,
    answer_format: str = "default",
    detailed_logger=None,
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate TruthfulQA MC2 (multi-true) using correctness-based MI.
    
    Key difference from MC1:
    - MC1: Exactly 1 correct answer per question
    - MC2: Multiple correct answers per question (≥1)
    - Correctness check: Answer is correct if it matches ANY of the true answers
    
    This approach measures uncertainty about WHETHER the answer is truthful,
    using the same correctness-based MI as MC1, but with multi-label correctness.
    
    Args:
        client: LLM client with chat_completion_with_logprobs method
        examples: List of MCQExample instances (TruthfulQA MC2 with metadata)
        k: Number of independent chains per question
        n: Chain length / pseudo joint dimension
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        mi_method: MI estimator ("plugin" or "listing")
        confidence_method: How to convert MI to confidence
        answer_format: Answer format for prompting
        detailed_logger: Optional logger for detailed traces
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .mi_estimator import estimate_mi_nats, estimate_mi_listing_nats
    from .datasets import match_answer_to_choices, is_answer_correct_mc2
    
    results = []
    logprob_tracker = LogprobTracker()
    iterator = tqdm(examples, desc="Evaluating TruthfulQA MC2 (multi-true correctness MI)") if verbose else examples
    
    for ex_idx, ex in enumerate(iterator):
        # Generate K chains of length n
        chains_with_logprobs = []
        for _ in range(k):
            chain = run_chain_with_logprobs(
                client=client,
                query=ex.question,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_style="naive",
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                answer_format=answer_format
            )
            chains_with_logprobs.append(chain)
            
            # Track logprobs from this chain
            for _, logprob in chain:
                logprob_tracker.record(logprob)
        
        # Select answer using marginalized pseudo joint
        predicted_answer_text = select_answer_via_pseudo_joint(chains_with_logprobs, n)
        
        # Match to MCQ choices
        predicted_choice = match_answer_to_choices(
            predicted_answer_text,
            ex.choice_texts,
            ex.choices,
            answer_format=answer_format
        )
        
        # ===== CORRECTNESS-BASED MI (MC2 variant) =====
        # Transform chains from choices to binary correctness
        chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
        
        # Map each answer in each chain to "correct" or "incorrect"
        # MC2: Answer is correct if it matches ANY of the true answers
        correctness_chains = []
        for chain in chains_text:
            correctness_chain = []
            for answer_text in chain:
                # Match this answer to a choice
                matched_choice = match_answer_to_choices(
                    answer_text,
                    ex.choice_texts,
                    ex.choices,
                    answer_format=answer_format
                )
                # Check if it's correct using MC2 logic (matches any true answer)
                is_correct = is_answer_correct_mc2(matched_choice, ex)
                correctness_chain.append("correct" if is_correct else "incorrect")
            correctness_chains.append(correctness_chain)
        
        # Compute MI on correctness (binary space) instead of choices
        if mi_method == "listing":
            mi_nats = estimate_mi_listing_nats(correctness_chains)
        else:
            mi_nats = estimate_mi_nats(correctness_chains)
        
        mi_bits = nats_to_bits(mi_nats)
        
        # Compute correctness agreement (what fraction agree on correctness)
        final_correctness = [chain[-1] for chain in correctness_chains]
        agreement = compute_agreement_fraction(final_correctness)
        
        # Convert MI to confidence
        confidence = mi_to_confidence(mi_nats, method=confidence_method)
        
        # Check if prediction is correct (MC2: matches any true answer)
        correct = is_answer_correct_mc2(predicted_choice, ex)
        
        result = EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,  # Primary answer (first correct one)
            correct=correct,
            confidence=confidence,
            mi_score=mi_bits,
            agreement=agreement,  # Agreement on correctness, not on specific choice
            chains=chains_with_logprobs
        )
        results.append(result)
        write_progress(len(results))
        
        # Log detailed trace if logger provided
        if detailed_logger:
            from .iterative_prompting import compose_prompt
            
            raw_inputs = []
            raw_outputs = []
            for chain_idx, chain in enumerate(chains_with_logprobs):
                previous_answers = []
                for step_idx, (response, logprob) in enumerate(chain):
                    prompt = compose_prompt(
                        ex.question, 
                        previous_answers,
                        prompt_style="naive",
                        choices=ex.choices,
                        choice_texts=ex.choice_texts,
                        answer_format=answer_format
                    )
                    
                    raw_inputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "prompt": prompt,
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    })
                    
                    raw_outputs.append({
                        "chain_id": chain_idx,
                        "step": step_idx,
                        "text": response,
                        "logprob": logprob,
                        "probability": np.exp(logprob) if logprob != 0.0 else 0.5
                    })
                    
                    previous_answers.append(response)
            
            # Get MC2 metadata for logging
            num_correct = ex.metadata.get("num_correct", 1) if ex.metadata else 1
            correct_choices_str = ", ".join(ex.metadata.get("correct_choices", [ex.answer_key])) if ex.metadata else ex.answer_key
            
            # Log the question with MC2 correctness-based MI metadata
            method_data = {
                "method_name": "truthfulqa_mc2_correctness_mi",
                "method_description": f"k={k} chains of length n={n}, MC2 correctness-based MI (multi-true)",
                "raw_inputs": raw_inputs,
                "raw_outputs": raw_outputs,
                "decision_process": {
                    "num_chains": k,
                    "chain_length": n,
                    "total_inferences": k * n,
                    "mi_method": mi_method,
                    "mi_nats": float(mi_nats),
                    "mi_bits": float(mi_bits),
                    "selected_answer": predicted_choice,
                    "matched_choice": predicted_choice,
                    "confidence_computation": f"MI-based ({confidence_method}): {confidence:.4f}",
                    "correctness_agreement": f"{agreement*100:.2f}% ({int(agreement*k)}/{k} chains agree on correctness)",
                    "mc2_info": f"{num_correct} correct answers: {correct_choices_str}"
                },
                "final_metrics": {
                    "predicted": predicted_choice,
                    "correct": correct,
                    "confidence": float(confidence),
                    "mi_score": float(mi_bits),
                    "agreement": float(agreement)
                }
            }
            
            detailed_logger.log_question(
                question_id=offset + ex_idx,
                question_text=ex.question,
                choices=ex.choices,
                choice_texts=ex.choice_texts,
                gold_answer=f"{correct_choices_str} (MC2: {num_correct} correct answers)",
                method_data=method_data
            )
    
    # Compute overall metrics
    correct_arr = np.array([r.correct for r in results], dtype=float)
    confidence_arr = np.array([r.confidence for r in results], dtype=float)
    
    accuracy = correct_arr.mean()
    ece = compute_ece(correct_arr, confidence_arr, correct_arr)
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(confidence_arr.mean()),
        "avg_mi_bits": float(sum(r.mi_score for r in results) / len(results)),
        "avg_correctness_agreement": float(sum(r.agreement for r in results) / len(results)),
        "n_samples": len(results),
        "logprob_stats": logprob_tracker.get_stats()
    }
    
    return metrics, results


# =============================================================================
# NLI-Based Semantic Clustering for MI Method
# =============================================================================

class NLIClusteringCache:
    """
    Cache for NLI model and clustering results to avoid redundant computations.
    Used when --use-nli-clustering flag is enabled.
    """
    def __init__(self, model_name: str = "microsoft/deberta-v2-xlarge-mnli", device: str = None):
        """Initialize NLI model for semantic clustering."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ImportError(
                "NLI clustering requires transformers library. "
                "Install with: pip install transformers"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading NLI model for semantic clustering: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        
        # Cache for pairwise entailment scores
        self._entailment_cache = {}
    
    def check_mutual_entailment(
        self, 
        text_a: str, 
        text_b: str, 
        threshold: float = 0.5
    ) -> bool:
        """
        Check if two texts are mutually entailed (semantically equivalent).
        
        Args:
            text_a: First text
            text_b: Second text
            threshold: Minimum P(entailment) for mutual entailment
        
        Returns:
            True if texts are mutually entailed, False otherwise
        """
        import torch
        
        # Normalize for comparison
        text_a = text_a.strip().lower()
        text_b = text_b.strip().lower()
        
        if text_a == text_b:
            return True
        
        # Check cache
        cache_key_fwd = (text_a, text_b)
        cache_key_bwd = (text_b, text_a)
        
        if cache_key_fwd in self._entailment_cache:
            fwd_score = self._entailment_cache[cache_key_fwd]
        else:
            # Forward: does text_a entail text_b?
            inputs = self.tokenizer(text_a, text_b, return_tensors="pt", 
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                fwd_score = probs[0][0].item()  # P(entailment) for DeBERTa-MNLI
            
            self._entailment_cache[cache_key_fwd] = fwd_score
        
        if cache_key_bwd in self._entailment_cache:
            bwd_score = self._entailment_cache[cache_key_bwd]
        else:
            # Backward: does text_b entail text_a?
            inputs = self.tokenizer(text_b, text_a, return_tensors="pt",
                                   truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                bwd_score = probs[0][0].item()
            
            self._entailment_cache[cache_key_bwd] = bwd_score
        
        # Mutual entailment if both directions exceed threshold
        return fwd_score >= threshold and bwd_score >= threshold


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
        answers: List of answer strings
        nli_checker: NLI model cache for entailment checking
        threshold: Threshold for mutual entailment
    
    Returns:
        Dictionary mapping each answer to its cluster representative
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
        threshold: Threshold for mutual entailment
    
    Returns:
        Clustered chains where each answer is replaced by its cluster representative
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
        threshold: Threshold for mutual entailment
    
    Returns:
        Clustered marginal distribution with merged probabilities
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
