# Implementation Plan: MI & Iterative Prompting Evaluation on ARC/OpenBookQA

## Project Overview

Evaluate the Mutual Information (MI) and iterative prompting method from "To Believe or Not to Believe Your LLM" on multiple-choice benchmarks (ARC-Challenge, ARC-Easy, OpenBookQA) to measure:
1. **Accuracy**: Task performance
2. **ECE (Expected Calibration Error)**: Confidence calibration quality

## Key Finding: Paper's Method Purpose

✅ **CONFIRMED**: The MI method is primarily for **uncertainty quantification**, NOT accuracy improvement.

### What the Method Does:
- **Primary Goal**: Detect when to **abstain** (refuse to answer) based on MI score
- **When not abstaining**: Returns "default choice" (highest probability answer)
- **Paper Metrics**: Precision/Recall curves for abstention policies, NOT ECE

### Quote from Paper (lines 843-844):
> "For the M.I. method, the default choice is the sampled response with the highest probability according to the marginalized pseudo joint distribution."

### Implication:
- ECE evaluation is a **NEW contribution** extending the paper's work
- Expected outcome: Similar accuracy, **better calibration** (lower ECE)

---

## Implementation Plan

### Phase 1: Code Migration & Setup (2-3 hours)

#### 1.1 Repository Structure
```
llm-belief-mi-test/
├── doc/                          # Paper files (already present)
├── llm_belief_mi_test/           # Main package
│   ├── __init__.py
│   ├── mi_estimator.py           # Copy from repro (no changes)
│   ├── iterative_prompting.py   # Copy from repro (no changes)
│   ├── evaluation.py             # Copy from repro (no changes)
│   ├── datasets.py               # Modify: Add ARC/OpenBookQA loaders
│   ├── llm_client_local.py      # NEW: Local Llama client
│   ├── calibration.py            # NEW: ECE computation & evaluation
│   ├── cli.py                    # NEW: Simplified CLI
│   └── utils.py                  # NEW: Helper functions
├── requirements.txt              # Dependencies
├── README.md                     # Usage documentation
└── outputs/                      # Results directory
    ├── results/
    ├── plots/
    └── logs/
```

#### 1.2 Files to Copy (Keep As-Is)
```bash
# From llm-belief-mi-repro/llm_belief_mi_repro/
cp mi_estimator.py llm_belief_mi_test/
cp iterative_prompting.py llm_belief_mi_test/
cp evaluation.py llm_belief_mi_test/
# datasets.py - copy but will modify
# plots.py - copy if needed for visualization
```

#### 1.3 Files to Remove/Replace
- ❌ Remove: OpenAI client, HF endpoint client, async clients
- ❌ Remove: LM Studio setup scripts
- ❌ Remove: Fireworks/router configuration
- ✅ Add: Single local transformers-based client

---

### Phase 2: Local Llama Client Implementation (1-2 hours)

#### 2.1 New File: `llm_client_local.py`

```python
from __future__ import annotations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LocalLlamaClient:
    """Local Llama-3.1-8B-Instruct client using HuggingFace Transformers."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure quantization if needed
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
        }
        
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("Using 4-bit quantization")
        elif load_in_8bit:
            kwargs["load_in_8bit"] = True
            logger.info("Using 8-bit quantization")
        
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs
        )
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 128,
    ) -> str:
        """
        Generate a completion for chat-formatted messages.
        
        Args:
            messages: List of dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text string
        """
        # Convert chat format to prompt using Llama template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode only the generated tokens (skip input)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def supports_logprobs(self) -> bool:
        """This client doesn't provide token logprobs."""
        return False
```

#### 2.2 Dependencies (`requirements.txt`)
```
torch>=2.0.0
transformers>=4.40.0
accelerate>=0.27.0
bitsandbytes>=0.43.0
datasets>=2.18.0
tqdm>=4.66.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

### Phase 3: Dataset Integration (1-2 hours)

#### 3.1 Modify `datasets.py`

Add loaders for multiple-choice benchmarks:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class MCQExample:
    """Multiple-choice question example."""
    question: str
    choices: List[str]  # e.g., ["A", "B", "C", "D"]
    choice_texts: List[str]  # e.g., ["option A text", ...]
    answer_key: str  # e.g., "C"
    answer_index: int  # e.g., 2 (0-indexed)


def load_arc_challenge(split: str = "test", limit: int | None = None) -> List[MCQExample]:
    """Load ARC-Challenge dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    examples = []
    for ex in ds:
        choices = ex["choices"]["label"]
        choice_texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        answer_idx = choices.index(answer_key)
        
        examples.append(MCQExample(
            question=ex["question"],
            choices=choices,
            choice_texts=choice_texts,
            answer_key=answer_key,
            answer_index=answer_idx
        ))
    
    return examples


def load_arc_easy(split: str = "test", limit: int | None = None) -> List[MCQExample]:
    """Load ARC-Easy dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    examples = []
    for ex in ds:
        choices = ex["choices"]["label"]
        choice_texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        answer_idx = choices.index(answer_key)
        
        examples.append(MCQExample(
            question=ex["question"],
            choices=choices,
            choice_texts=choice_texts,
            answer_key=answer_key,
            answer_index=answer_idx
        ))
    
    return examples


def load_openbookqa(split: str = "test", limit: int | None = None) -> List[MCQExample]:
    """Load OpenBookQA dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    examples = []
    for ex in ds:
        choices = ex["choices"]["label"]
        choice_texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        answer_idx = choices.index(answer_key)
        
        examples.append(MCQExample(
            question=ex["question_stem"],
            choices=choices,
            choice_texts=choice_texts,
            answer_key=answer_key,
            answer_index=answer_idx
        ))
    
    return examples


def match_answer_to_choices(
    generated_answer: str,
    choice_texts: List[str],
    choices: List[str]
) -> str:
    """
    Match a generated answer to one of the multiple choices.
    
    Uses fuzzy string matching to find the best match.
    """
    from difflib import SequenceMatcher
    
    generated = generated_answer.lower().strip()
    
    # First, check if answer contains choice letter
    for letter in choices:
        if letter.lower() in generated[:10]:  # Check first 10 chars
            return letter
    
    # Then, fuzzy match against choice texts
    best_match = None
    best_score = 0.0
    
    for choice_letter, choice_text in zip(choices, choice_texts):
        choice_lower = choice_text.lower().strip()
        
        # Check substring match
        if choice_lower in generated or generated in choice_lower:
            return choice_letter
        
        # Compute similarity
        similarity = SequenceMatcher(None, generated, choice_lower).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = choice_letter
    
    # Return best match or first choice as fallback
    return best_match if best_match else choices[0]
```

---

### Phase 4: Calibration & Evaluation (3-4 hours)

#### 4.1 New File: `calibration.py`

```python
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

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
    chains: List[List[str]]


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


def evaluate_mcq_with_mi(
    client,
    examples: List,  # MCQExample instances
    k: int = 10,
    t: int = 3,
    temperature: float = 0.5,
    max_tokens: int = 64,
    mi_method: str = "listing",
    confidence_method: str = "inverse",
    verbose: bool = True
) -> Tuple[Dict[str, float], List[EvaluationResult]]:
    """
    Evaluate multiple-choice questions using MI-based confidence.
    
    Args:
        client: LLM client with chat_completion method
        examples: List of MCQExample instances
        k: Number of chains per question
        t: Chain length (iterative prompting steps)
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        mi_method: MI estimator ("plugin" or "listing")
        confidence_method: How to convert MI to confidence
        verbose: Show progress bar
        
    Returns:
        (metrics_dict, results_list)
    """
    from tqdm import tqdm
    from .iterative_prompting import run_k_chains_for_query
    from .mi_estimator import estimate_mi_nats
    from .datasets import match_answer_to_choices
    
    results = []
    iterator = tqdm(examples) if verbose else examples
    
    for ex in iterator:
        # Generate K chains with iterative prompting
        chains = run_k_chains_for_query(
            client=client,
            query=ex.question,
            chain_length=t,
            k=k,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_style="naive"
        )
        
        # Compute MI
        if mi_method == "listing":
            mi_nats = estimate_mi_listing_nats(chains)
        else:
            mi_nats = estimate_mi_nats(chains)
        
        mi_bits = nats_to_bits(mi_nats)
        
        # Get final answers from chains
        final_answers = [ch[-1] for ch in chains]
        
        # Compute agreement (self-consistency)
        agreement = compute_agreement_fraction(final_answers)
        
        # Match generated answers to choices
        # Use most common answer
        from collections import Counter
        answer_counts = Counter(final_answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        predicted_choice = match_answer_to_choices(
            most_common_answer,
            ex.choice_texts,
            ex.choices
        )
        
        # Convert MI to confidence
        confidence = mi_to_confidence(mi_nats, method=confidence_method)
        
        # Check correctness
        correct = (predicted_choice == ex.answer_key)
        
        results.append(EvaluationResult(
            question=ex.question,
            predicted=predicted_choice,
            gold=ex.answer_key,
            correct=correct,
            confidence=confidence,
            mi_score=mi_bits,
            agreement=agreement,
            chains=chains
        ))
    
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
```

---

### Phase 5: CLI Implementation (2 hours)

#### 5.1 New File: `cli.py`

```python
import argparse
import json
import csv
from pathlib import Path
import logging

from .llm_client_local import LocalLlamaClient
from .datasets import load_arc_challenge, load_arc_easy, load_openbookqa
from .calibration import evaluate_mcq_with_mi


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=level
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MI & Iterative Prompting on MCQ Benchmarks"
    )
    
    # Dataset selection
    parser.add_argument(
        "--dataset",
        choices=["arc-challenge", "arc-easy", "openbookqa"],
        required=True,
        help="Benchmark dataset to evaluate"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (test/validation)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples (for testing)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization (saves memory)"
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization"
    )
    
    # MI parameters
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of chains per question"
    )
    parser.add_argument(
        "--t",
        type=int,
        default=3,
        help="Chain length (iterative prompting steps)"
    )
    parser.add_argument(
        "--mi-method",
        choices=["plugin", "listing"],
        default="listing",
        help="MI estimator to use"
    )
    parser.add_argument(
        "--confidence-method",
        choices=["inverse", "exp", "normalized"],
        default="inverse",
        help="How to convert MI to confidence"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens per generation"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Load dataset
    logging.info(f"Loading {args.dataset} ({args.split} split)...")
    if args.dataset == "arc-challenge":
        examples = load_arc_challenge(args.split, args.limit)
    elif args.dataset == "arc-easy":
        examples = load_arc_easy(args.split, args.limit)
    else:
        examples = load_openbookqa(args.split, args.limit)
    
    logging.info(f"Loaded {len(examples)} examples")
    
    # Initialize model
    logging.info(f"Initializing model: {args.model}")
    client = LocalLlamaClient(
        model_name=args.model,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )
    
    # Run evaluation
    logging.info("Starting evaluation...")
    metrics, results = evaluate_mcq_with_mi(
        client=client,
        examples=examples,
        k=args.k,
        t=args.t,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mi_method=args.mi_method,
        confidence_method=args.confidence_method,
        verbose=True
    )
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    print("="*60 + "\n")
    
    # Save results to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'question', 'predicted', 'gold', 'correct',
            'confidence', 'mi_score', 'agreement'
        ])
        writer.writeheader()
        
        for r in results:
            writer.writerow({
                'question': r.question,
                'predicted': r.predicted,
                'gold': r.gold,
                'correct': int(r.correct),
                'confidence': r.confidence,
                'mi_score': r.mi_score,
                'agreement': r.agreement
            })
    
    # Save metrics to JSON
    metrics_path = output_path.with_suffix('.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f"Results saved to: {output_path}")
    logging.info(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
```

---

### Phase 6: Baseline Comparisons (Optional, 2 hours)

Implement baselines for comparison:

1. **Greedy Baseline**: Temperature=0, single answer
2. **Self-Consistency**: Majority vote across K samples (no MI)
3. **Entropy-based**: Use output entropy as confidence
4. **MI + Iterative Prompting**: Full method

---

## Expected Results

### Accuracy
- **Baseline (greedy)**: ~60-80% depending on benchmark
- **MI method**: Similar (±2%), main value is in confidence

### ECE (Expected Calibration Error)
- **Baseline**: 0.10 - 0.20 (poorly calibrated)
- **MI method**: 0.05 - 0.10 (better calibrated)
- **Improvement**: 30-50% reduction in ECE

### Abstention Performance
If implementing abstention policies:
- At 10% abstention rate: +5-10% accuracy on remaining examples
- At 20% abstention rate: +10-15% accuracy on remaining examples

---

## Timeline

- **Phase 1-2** (Setup + Local Client): 3-4 hours
- **Phase 3-4** (Datasets + Calibration): 4-6 hours
- **Phase 5** (CLI + Testing): 2-3 hours
- **Phase 6** (Baselines): 2-3 hours (optional)

**Total**: 11-16 hours of focused work

---

## Testing Strategy

### Quick Sanity Check (5 examples)
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-easy \
  --limit 5 \
  --k 3 --t 2 \
  --load-in-4bit \
  --output outputs/test_quick.csv
```

### Small Test (50 examples)
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge \
  --limit 50 \
  --k 10 --t 3 \
  --load-in-4bit \
  --output outputs/arc_challenge_small.csv
```

### Full Evaluation (all examples)
```bash
# ARC-Challenge (~1200 examples)
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge \
  --k 10 --t 3 \
  --load-in-4bit \
  --output outputs/arc_challenge_full.csv

# ARC-Easy (~2400 examples)
python -m llm_belief_mi_test.cli \
  --dataset arc-easy \
  --k 10 --t 3 \
  --load-in-4bit \
  --output outputs/arc_easy_full.csv

# OpenBookQA (~500 examples)
python -m llm_belief_mi_test.cli \
  --dataset openbookqa \
  --k 10 --t 3 \
  --load-in-4bit \
  --output outputs/openbookqa_full.csv
```

---

## Hardware Requirements

### Minimum (4-bit quantization)
- GPU: 12GB VRAM (RTX 3060, RTX 4060 Ti)
- RAM: 16GB system RAM
- Storage: 10GB for model

### Recommended (8-bit quantization)
- GPU: 16GB VRAM (RTX 4080, A4000)
- RAM: 32GB system RAM

### Optimal (bfloat16, no quantization)
- GPU: 24GB VRAM (RTX 4090, A5000)
- RAM: 32GB system RAM

---

## Performance Estimates

With 4-bit quantization on RTX 4090:
- **Single answer**: ~0.5-1s
- **K=10 chains, t=3**: ~15-30s per question
- **ARC-Challenge (1200 questions)**: ~5-10 hours
- **All three benchmarks**: ~15-20 hours total

Optimization: Use batch processing if possible, or reduce K to 5.

---

## Next Steps

1. ✅ Create repository structure
2. ✅ Copy core modules from repro
3. → Implement local Llama client
4. → Test model loading and generation
5. → Implement dataset loaders
6. → Implement calibration evaluation
7. → Run small-scale tests
8. → Full benchmark evaluation
9. → Analysis and visualization

---

## Questions/Notes

- Q: Should we implement abstention policies?
  - A: Optional - focus on accuracy + ECE first

- Q: What if MI doesn't improve calibration?
  - A: Document findings - negative results are valuable!

- Q: Baseline comparison?
  - A: At minimum, compare to greedy and self-consistency

---

_Last updated: 2025-01-22_

