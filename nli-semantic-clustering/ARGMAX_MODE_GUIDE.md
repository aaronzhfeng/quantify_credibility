# Argmax Mode Guide

## Overview

The NLI module now supports **two decision modes** for determining entailment:

1. **Soft Threshold Mode** (default): Uses probability scores with tunable threshold
2. **Argmax Mode** (new): Uses winner-takes-all classification

## How It Works

### DeBERTa NLI Model Output

The model outputs 3 probabilities for each sentence pair:

```python
[P(entailment), P(neutral), P(contradiction)]
# Example: [0.45, 0.35, 0.20]  (sum = 1.0)
```

### Soft Threshold Mode (Default)

```python
# Extract entailment probability
entailment_score = probs[entailment_id]  # 0.45

# Compare against threshold
is_entailed = entailment_score >= threshold  # 0.45 >= 0.5 → False
```

**Characteristics:**
- ✓ **Tunable**: Adjust strictness with threshold (0.3 = lenient, 0.7 = strict)
- ✓ **Conservative**: Requires strong entailment evidence
- ✓ **Good for debugging**: Threshold sweeps reveal sensitivity
- ✗ Ignores neutral/contradiction information

### Argmax Mode (New)

```python
# Find which class has highest probability
predicted_class = argmax(probs)  # Returns: entailment

# Binary decision
is_entailed = (predicted_class == entailment)  # True
```

**Characteristics:**
- ✓ **Uses all 3 classes**: Picks most likely relationship
- ✓ **More aggressive**: Entailment wins even with low scores
- ✗ **Not tunable**: Binary decision, threshold is ignored
- ✗ May over-cluster weak relationships

## Comparison Examples

### Example 1: Clear Entailment
```
Probs: [0.95 (entailment), 0.03 (neutral), 0.02 (contradiction)]

Soft (threshold=0.5): 0.95 >= 0.5 → ✓ Accept
Argmax:               argmax = entailment → ✓ Accept

Result: SAME
```

### Example 2: Weak Entailment
```
Probs: [0.45 (entailment), 0.35 (neutral), 0.20 (contradiction)]

Soft (threshold=0.5): 0.45 >= 0.5 → ✗ Reject
Soft (threshold=0.3): 0.45 >= 0.3 → ✓ Accept
Argmax:               argmax = entailment → ✓ Accept

Result: DIFFERENT - Argmax is more lenient
```

### Example 3: Ambiguous (Neutral Wins)
```
Probs: [0.40 (entailment), 0.50 (neutral), 0.10 (contradiction)]

Soft (threshold=0.5): 0.40 >= 0.5 → ✗ Reject
Argmax:               argmax = neutral → ✗ Reject

Result: SAME, but for different reasons
```

### Example 4: Contradiction
```
Probs: [0.20 (entailment), 0.30 (neutral), 0.50 (contradiction)]

Soft (threshold=0.5): 0.20 >= 0.5 → ✗ Reject
Argmax:               argmax = contradiction → ✗ Reject

Result: SAME
```

## Usage

### Command Line

Add `--use-argmax` flag to any threshold sweep command:

```bash
# Soft threshold mode (default)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_greedy \
  --output results/soft_threshold.json \
  --thresholds 0.3 0.5 0.7 \
  --use-nli-grading \
  --limit 200

# Argmax mode (threshold ignored)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_greedy \
  --output results/argmax.json \
  --thresholds 0.5 \
  --use-nli-grading \
  --use-argmax \
  --limit 200
```

**Note:** When using `--use-argmax`, the threshold value is ignored (but still required for API compatibility).

### Python API

```python
from nli_clustering.core import NLIClusteringCache

nli = NLIClusteringCache()

# Soft threshold mode
is_correct_soft = nli.is_correct(
    prediction="The capital is Paris",
    gold_labels=["Paris"],
    threshold=0.5,
    use_argmax=False  # Default
)

# Argmax mode
is_correct_argmax = nli.is_correct(
    prediction="The capital is Paris",
    gold_labels=["Paris"],
    threshold=0.5,  # Ignored
    use_argmax=True
)
```

## When to Use Each Mode

### Use Soft Threshold Mode If:
- You want to **tune strictness** (e.g., test 0.3, 0.5, 0.7)
- You need **conservative clustering** (high confidence required)
- You're **debugging** and want to see sensitivity to threshold
- You want to avoid over-clustering weak relationships

### Use Argmax Mode If:
- Someone suggested using the **standard classifier interpretation**
- You want **maximum recall** (accept more matches)
- You want to use **all 3 NLI classes** (not just entailment)
- You prefer the model's **learned decision boundary** over manual threshold

## Empirical Results (TriviaQA Self-Consistency, 10 questions)

```
Method              Clusters   Acc Orig   Acc NLI   Δ Acc   ECE Orig   ECE NLI   Δ ECE
--------------------------------------------------------------------------------------
Soft (thresh=0.3)   15.8%      0.500      0.600    +0.100   0.090      0.400    +0.310
Soft (thresh=0.5)   13.3%      0.500      0.600    +0.100   0.090      0.400    +0.310
Soft (thresh=0.7)   10.8%      0.500      0.600    +0.100   0.090      0.400    +0.310
Argmax              13.3%      0.500      0.600    +0.100   0.090      0.400    +0.310
```

**Observation:** Argmax mode behaves similarly to soft threshold mode at ~0.5, which makes sense since the model was trained to optimize argmax classification.

## Recommendation

**Start with soft threshold mode** (default) because:
1. It's more flexible (tunable)
2. It's better for debugging (see threshold sensitivity)
3. You can sweep thresholds to find optimal trade-off
4. It's more conservative (avoids false clustering)

**Try argmax mode** if soft threshold results are too strict or if you want to compare against the "standard" way of interpreting classifiers.

## Technical Details

### Implementation

The argmax mode is implemented at the lowest level (`check_entailment`) and propagates through:
- `check_entailment(use_argmax=True)` → Returns 0.0 or 1.0 instead of probability
- `check_mutual_entailment(use_argmax=True)` → Uses argmax for both directions
- `cluster_answers_by_nli(use_argmax=True)` → Clusters using argmax
- `is_correct(use_argmax=True)` → Grades using argmax
- `apply_nli_clustering_to_chains(use_argmax=True)` → Full pipeline with argmax

### Caching

Argmax and soft threshold results are cached separately (different cache keys), so you can switch between modes without performance penalty.

