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


def compute_ece(
    predictions: np.ndarray,
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between confidence and accuracy.
    
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
            bin_accuracy = (predictions[in_bin] == labels[in_bin]).mean()
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
        
        # Match to MCQ choices
        predicted_choice = match_answer_to_choices(
            response,
            ex.choice_texts,
            ex.choices,
            answer_format=answer_format
        )
        
        # Confidence from logprob (exp of average token logprob)
        confidence = math.exp(logprob) if logprob < 0 else 0.5
        
        # Check correctness
        correct = (predicted_choice == ex.answer_key)
        
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
            detailed_logger.log_question(
                question_id=ex_idx,
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
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(avg_confidence),
        "avg_mi_bits": 0.0,
        "avg_agreement": 1.0,
        "n_samples": len(results),
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
        
        # Check correctness
        correct = (predicted_choice == ex.answer_key)
        
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
            detailed_logger.log_question(
                question_id=ex_idx,
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
    avg_agreement = np.mean([r.agreement for r in results])
    
    metrics = {
        "accuracy": float(accuracy),
        "ece": float(ece),
        "avg_confidence": float(avg_confidence),
        "avg_mi_bits": 0.0,
        "avg_agreement": float(avg_agreement),
        "n_samples": len(results),
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
                question_id=ex_idx,
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
    }
    
    return metrics, results

