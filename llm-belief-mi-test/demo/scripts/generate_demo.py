#!/usr/bin/env python3
"""
Generate comprehensive demo showing all 5 evaluation methods on first 5 OpenBookQA questions.

Captures everything: raw inputs, raw outputs, intermediate computations, decision logic.
"""

import sys
import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.stats import entropy as scipy_entropy

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_belief_mi_test.llm_client_local import LocalLlamaClient
from llm_belief_mi_test.datasets import load_openbookqa, match_answer_to_choices
from llm_belief_mi_test.iterative_prompting import compose_prompt
from llm_belief_mi_test.calibration import (
    compute_f1_similarity,
    group_by_semantic_equivalence,
    build_pseudo_joint_with_probs,
    marginalize_to_final_answer
)
from llm_belief_mi_test.mi_estimator import estimate_mi_listing_nats, nats_to_bits


def run_greedy_method(client, example, max_tokens=100) -> Dict[str, Any]:
    """Run greedy method and capture all details."""
    messages = compose_prompt(example.question, [], prompt_style="naive",
                             choices=example.choices, choice_texts=example.choice_texts)
    
    # Capture raw input
    raw_input = {
        "prompt": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    
    # Get response
    response, logprob = client.chat_completion_with_logprobs(
        messages,
        temperature=0.0,
        max_tokens=max_tokens
    )
    
    # Capture raw output
    raw_output = {
        "text": response,
        "logprob": logprob,
        "probability": math.exp(logprob)
    }
    
    # Decision process
    matched_choice = match_answer_to_choices(response, example.choice_texts, example.choices)
    confidence = math.exp(logprob) if logprob < 0 else 0.5
    
    decision_process = {
        "selected_text": response,
        "matched_choice": matched_choice,
        "confidence_computation": f"exp(logprob) = exp({logprob:.4f}) = {confidence:.4f}"
    }
    
    # Final metrics
    final_metrics = {
        "predicted": matched_choice,
        "correct": matched_choice == example.answer_key,
        "confidence": confidence,
        "mi_score": 0.0,
        "agreement": 1.0
    }
    
    return {
        "description": "Single greedy decode (temperature=0)",
        "raw_inputs": [raw_input],
        "raw_outputs": [raw_output],
        "decision_process": decision_process,
        "final_metrics": final_metrics
    }


def run_self_consistency_method(client, example, k=10, temperature=0.9, max_tokens=100) -> Dict[str, Any]:
    """Run self-consistency method and capture all details."""
    raw_inputs = []
    raw_outputs = []
    samples = []
    
    # Generate k samples
    for i in range(k):
        messages = compose_prompt(example.question, [], prompt_style="naive",
                                 choices=example.choices, choice_texts=example.choice_texts)
        
        raw_inputs.append({
            "sample_id": i,
            "prompt": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        response, logprob = client.chat_completion_with_logprobs(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        raw_outputs.append({
            "sample_id": i,
            "text": response,
            "logprob": logprob,
            "probability": math.exp(logprob)
        })
        
        samples.append(response)
    
    # Count votes
    from collections import Counter
    vote_counts = Counter(samples)
    majority_text, max_count = vote_counts.most_common(1)[0]
    
    # Match to choice
    matched_choice = match_answer_to_choices(majority_text, example.choice_texts, example.choices)
    confidence = max_count / k
    
    decision_process = {
        "all_responses": samples,
        "vote_counts": dict(vote_counts),
        "majority": majority_text,
        "matched_choice": matched_choice,
        "confidence_computation": f"{max_count}/{k} = {confidence:.4f}"
    }
    
    final_metrics = {
        "predicted": matched_choice,
        "correct": matched_choice == example.answer_key,
        "confidence": confidence,
        "mi_score": 0.0,
        "agreement": confidence
    }
    
    return {
        "description": f"k={k} samples, majority voting",
        "raw_inputs": raw_inputs,
        "raw_outputs": raw_outputs,
        "decision_process": decision_process,
        "final_metrics": final_metrics
    }


def run_semantic_entropy_method(client, example, k=10, temperature=0.9, max_tokens=100, threshold=0.25) -> Dict[str, Any]:
    """Run semantic entropy method and capture all details."""
    raw_inputs = []
    raw_outputs = []
    samples = []
    
    # Generate k samples
    for i in range(k):
        messages = compose_prompt(example.question, [], prompt_style="naive",
                                 choices=example.choices, choice_texts=example.choice_texts)
        
        raw_inputs.append({
            "sample_id": i,
            "prompt": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        response, logprob = client.chat_completion_with_logprobs(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        prob = math.exp(logprob)
        
        raw_outputs.append({
            "sample_id": i,
            "text": response,
            "logprob": logprob,
            "probability": prob
        })
        
        samples.append((response, prob))
    
    # Compute similarity matrix
    texts = [s[0] for s in samples]
    n = len(texts)
    similarity_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = compute_f1_similarity(texts[i], texts[j])
    
    # Group by semantic equivalence
    clusters = group_by_semantic_equivalence(samples, threshold=threshold)
    
    # Format clusters for output
    semantic_clusters = {}
    cluster_distributions = {}
    for cluster_id, cluster_samples in clusters.items():
        representative_text = cluster_samples[0][0]
        total_prob = sum(prob for _, prob in cluster_samples)
        
        semantic_clusters[f"cluster_{cluster_id}"] = {
            "representative": representative_text,
            "texts": [text for text, _ in cluster_samples],
            "probs": [prob for _, prob in cluster_samples],
            "total_prob": total_prob
        }
        cluster_distributions[representative_text] = total_prob
    
    # Normalize distribution
    total_prob = sum(cluster_distributions.values())
    if total_prob > 0:
        distribution = {k: v/total_prob for k, v in cluster_distributions.items()}
    else:
        distribution = {list(cluster_distributions.keys())[0]: 1.0} if cluster_distributions else {}
    
    # Calculate entropy
    probs = np.array(list(distribution.values()))
    if len(probs) > 0 and np.sum(probs) > 0:
        entropy_nats = scipy_entropy(probs, base=np.e)
    else:
        entropy_nats = 0.0
    
    entropy_bits = entropy_nats / math.log(2)
    confidence = math.exp(-entropy_nats)
    
    # Select answer
    predicted_text = max(distribution, key=distribution.get) if distribution else samples[0][0]
    matched_choice = match_answer_to_choices(predicted_text, example.choice_texts, example.choices)
    
    decision_process = {
        "similarity_matrix": similarity_matrix,
        "similarity_threshold": threshold,
        "semantic_clusters": semantic_clusters,
        "aggregated_distribution": distribution,
        "entropy_nats": entropy_nats,
        "entropy_bits": entropy_bits,
        "confidence_computation": f"exp(-entropy_nats) = exp(-{entropy_nats:.4f}) = {confidence:.4f}"
    }
    
    final_metrics = {
        "predicted": matched_choice,
        "correct": matched_choice == example.answer_key,
        "confidence": confidence,
        "mi_score": entropy_bits,
        "agreement": distribution.get(predicted_text, 0.0)
    }
    
    return {
        "description": f"k={k} samples, entropy-based confidence with F1 similarity",
        "raw_inputs": raw_inputs,
        "raw_outputs": raw_outputs,
        "decision_process": decision_process,
        "final_metrics": final_metrics
    }


def run_self_verification_method(client, example, k=10, temperature=0.9, max_tokens=100) -> Dict[str, Any]:
    """Run self-verification method and capture all details."""
    raw_inputs = []
    raw_outputs = []
    samples = []
    
    # Step 1: Generate k samples
    for i in range(k):
        messages = compose_prompt(example.question, [], prompt_style="naive",
                                 choices=example.choices, choice_texts=example.choice_texts)
        
        raw_inputs.append({
            "sample_id": i,
            "step": "initial_generation",
            "prompt": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        response, logprob = client.chat_completion_with_logprobs(
            messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        prob = math.exp(logprob)
        
        raw_outputs.append({
            "sample_id": i,
            "step": "initial_generation",
            "text": response,
            "logprob": logprob,
            "probability": prob
        })
        
        samples.append((response, prob))
    
    # Aggregate by choice
    choice_probs = {}
    choice_to_text = {}
    for text, prob in samples:
        choice = match_answer_to_choices(text, example.choice_texts, example.choices)
        choice_probs[choice] = choice_probs.get(choice, 0.0) + prob
        if choice not in choice_to_text:
            choice_to_text[choice] = text
    
    # Select best answer
    if choice_probs:
        predicted_choice = max(choice_probs, key=choice_probs.get)
        predicted_text = choice_to_text[predicted_choice]
    else:
        predicted_choice = example.choices[0]
        predicted_text = samples[0][0] if samples else ""
    
    # Step 2: Verification
    verification_prompt = [{
        "role": "user",
        "content": f"""Consider the following question:
Q: {example.question}

One answer to question Q is: {predicted_text}

Is the above answer to question Q correct? Answer True or False.
A:"""
    }]
    
    raw_inputs.append({
        "sample_id": k,
        "step": "verification",
        "prompt": verification_prompt,
        "temperature": 0.0,
        "max_tokens": 5
    })
    
    verification_response, verification_logprob = client.chat_completion_with_logprobs(
        verification_prompt,
        temperature=0.0,
        max_tokens=5
    )
    
    # Parse verification
    response_lower = verification_response.strip().lower()
    if 'true' in response_lower:
        confidence = 0.9
        verdict = "True"
    elif 'false' in response_lower:
        confidence = 0.1
        verdict = "False"
    else:
        confidence = 0.5
        verdict = "Unclear"
    
    raw_outputs.append({
        "sample_id": k,
        "step": "verification",
        "text": verification_response,
        "logprob": verification_logprob,
        "parsed_verdict": verdict
    })
    
    # Agreement
    total_prob = sum(choice_probs.values())
    agreement = choice_probs.get(predicted_choice, 0.0) / total_prob if total_prob > 0 else 0.0
    
    decision_process = {
        "initial_selection": {
            "choice_probabilities": choice_probs,
            "selected_choice": predicted_choice,
            "selected_text": predicted_text
        },
        "verification_prompt": verification_prompt[0]["content"],
        "verification_response": {
            "raw_text": verification_response,
            "parsed_verdict": verdict,
            "logprob": verification_logprob
        },
        "confidence_computation": f"Based on verdict '{verdict}': {confidence:.4f}"
    }
    
    final_metrics = {
        "predicted": predicted_choice,
        "correct": predicted_choice == example.answer_key,
        "confidence": confidence,
        "mi_score": 0.0,
        "agreement": agreement
    }
    
    return {
        "description": f"k={k} samples + verification query",
        "raw_inputs": raw_inputs,
        "raw_outputs": raw_outputs,
        "decision_process": decision_process,
        "final_metrics": final_metrics
    }


def run_mi_method(client, example, k=10, n=2, temperature=0.9, max_tokens=100) -> Dict[str, Any]:
    """Run MI method and capture all details."""
    raw_inputs = []
    raw_outputs = []
    chains_with_logprobs = []
    
    # Generate k chains of length n
    for chain_id in range(k):
        chain = []
        previous_answers = []
        
        for step in range(n):
            messages = compose_prompt(example.question, previous_answers, prompt_style="naive",
                                     choices=example.choices, choice_texts=example.choice_texts)
            
            raw_inputs.append({
                "chain_id": chain_id,
                "step": step,
                "prompt": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            response, logprob = client.chat_completion_with_logprobs(
                messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raw_outputs.append({
                "chain_id": chain_id,
                "step": step,
                "text": response,
                "logprob": logprob,
                "probability": math.exp(logprob)
            })
            
            chain.append((response, logprob))
            previous_answers.append(response)
        
        chains_with_logprobs.append(chain)
    
    # Build pseudo joint
    pseudo_joint = build_pseudo_joint_with_probs(chains_with_logprobs, n)
    
    # Convert tuple keys to strings for JSON
    pseudo_joint_str = {str(k): v for k, v in pseudo_joint.items()}
    
    # Marginalize to final answer
    marginal = marginalize_to_final_answer(pseudo_joint, n)
    
    # Select answer
    if marginal:
        predicted_text = max(marginal, key=marginal.get)
    else:
        predicted_text = chains_with_logprobs[0][-1][0] if chains_with_logprobs else ""
    
    matched_choice = match_answer_to_choices(predicted_text, example.choice_texts, example.choices)
    
    # Compute MI
    chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
    mi_nats = estimate_mi_listing_nats(chains_text)
    mi_bits = nats_to_bits(mi_nats)
    
    # Confidence
    confidence = 1.0 / (1.0 + mi_nats)
    
    # Agreement
    final_answers = [chain[-1][0] for chain in chains_with_logprobs]
    from collections import Counter
    agreement = Counter(final_answers).most_common(1)[0][1] / k if final_answers else 0.0
    
    decision_process = {
        "chains": [
            [{"text": text, "logprob": logprob} for text, logprob in chain]
            for chain in chains_with_logprobs
        ],
        "pseudo_joint": pseudo_joint_str,
        "marginal_distribution": marginal,
        "mi_estimation": {
            "method": "listing",
            "mi_nats": mi_nats,
            "mi_bits": mi_bits
        },
        "confidence_computation": f"1/(1 + mi_nats) = 1/(1 + {mi_nats:.4f}) = {confidence:.4f}"
    }
    
    final_metrics = {
        "predicted": matched_choice,
        "correct": matched_choice == example.answer_key,
        "confidence": confidence,
        "mi_score": mi_bits,
        "agreement": agreement
    }
    
    return {
        "description": f"k={k} chains of length n={n}, MI estimation",
        "raw_inputs": raw_inputs,
        "raw_outputs": raw_outputs,
        "decision_process": decision_process,
        "final_metrics": final_metrics
    }


def generate_demo(num_questions=5, output_dir="demo/outputs"):
    """Generate demo data for first N questions from OpenBookQA."""
    
    print("="*80)
    print("GENERATING COMPREHENSIVE DEMO")
    print("="*80)
    print()
    
    # Load dataset
    print(f"Loading first {num_questions} questions from OpenBookQA...")
    examples = load_openbookqa("test", limit=num_questions)
    print(f"Loaded {len(examples)} examples\n")
    
    # Initialize client
    print("Initializing model...")
    client = LocalLlamaClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=True,
        cache=None  # Disable cache for demo
    )
    print("Model loaded\n")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each question
    for idx, ex in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"Processing Question {idx + 1}/{len(examples)}")
        print(f"{'='*80}")
        print(f"Q: {ex.question[:100]}...")
        print()
        
        # Build comprehensive result
        result = {
            "question_id": idx,
            "question_text": ex.question,
            "choices": [f"{choice}: {text}" for choice, text in zip(ex.choices, ex.choice_texts)],
            "gold_answer": ex.answer_key,
            "methods": {}
        }
        
        # Run each method
        methods = [
            ("greedy", lambda: run_greedy_method(client, ex)),
            ("self_consistency", lambda: run_self_consistency_method(client, ex)),
            ("semantic_entropy", lambda: run_semantic_entropy_method(client, ex)),
            ("self_verification", lambda: run_self_verification_method(client, ex)),
            ("mi_method", lambda: run_mi_method(client, ex))
        ]
        
        for method_name, method_func in methods:
            print(f"  Running {method_name}...")
            try:
                result["methods"][method_name] = method_func()
                pred = result["methods"][method_name]["final_metrics"]["predicted"]
                correct = result["methods"][method_name]["final_metrics"]["correct"]
                conf = result["methods"][method_name]["final_metrics"]["confidence"]
                print(f"    → Predicted: {pred}, Correct: {correct}, Confidence: {conf:.3f}")
            except Exception as e:
                print(f"    ✗ Error: {e}")
                result["methods"][method_name] = {"error": str(e)}
        
        # Add comparison summary
        result["comparison_summary"] = {
            "all_predictions": {
                method: data["final_metrics"]["predicted"]
                for method, data in result["methods"].items()
                if "final_metrics" in data
            },
            "all_correct": {
                method: data["final_metrics"]["correct"]
                for method, data in result["methods"].items()
                if "final_metrics" in data
            },
            "all_confidences": {
                method: data["final_metrics"]["confidence"]
                for method, data in result["methods"].items()
                if "final_metrics" in data
            },
            "agreement_across_methods": sum(
                1 for method, data in result["methods"].items()
                if "final_metrics" in data and data["final_metrics"]["predicted"] == ex.answer_key
            )
        }
        
        # Save to file
        output_file = Path(output_dir) / f"question_{idx}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"  ✓ Saved to {output_file}")
    
    print(f"\n{'='*80}")
    print("DEMO GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nGenerated {len(examples)} demo files in {output_dir}/")
    print(f"Each file contains detailed traces for all 5 methods.\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate comprehensive demo data")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to process")
    parser.add_argument("--output-dir", type=str, default="demo/outputs", help="Output directory")
    
    args = parser.parse_args()
    
    generate_demo(args.num_questions, args.output_dir)

