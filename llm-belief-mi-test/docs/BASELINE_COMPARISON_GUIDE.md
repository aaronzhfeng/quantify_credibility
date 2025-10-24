# Baseline Comparison Guide

## Overview

To properly evaluate the MI-based method, you need to compare it against baselines. This guide shows you how to run all three methods and compare their performance.

## Three Methods Available

### 1. **Greedy Baseline** (`--method greedy`)
- **How it works**: Single greedy decode (temperature=0) per question
- **Confidence**: Based on token logprobs
- **Cost**: **Cheapest** - 1 generation per question
- **Time**: ~5 minutes for 50 examples
- **Use case**: Fastest baseline, deterministic results

### 2. **Self-Consistency Baseline** (`--method self-consistency`)
- **How it works**: Generate k samples, use majority voting
- **Confidence**: Fraction of samples agreeing with majority
- **Cost**: k generations per question (e.g., k=10)
- **Time**: ~30 minutes for 50 examples with k=10
- **Use case**: Standard baseline from literature

### 3. **MI Method** (`--method mi`, default)
- **How it works**: k chains of length n, pseudo joint + MI estimation
- **Confidence**: Converted from MI score (lower MI = higher confidence)
- **Cost**: **Most expensive** - k×n generations per question
- **Time**: ~30 minutes for 50 examples with k=10, n=2
- **Use case**: Paper's method - should have better calibration (lower ECE)

## Expected Results

Based on the paper, you should see:

| Method | Accuracy | ECE | Notes |
|--------|----------|-----|-------|
| Greedy | 50-65% | 0.10-0.20 | Baseline performance |
| Self-Consistency | 50-65% | 0.10-0.20 | Similar to greedy |
| **MI Method** | 50-65% | **0.05-0.12** | ✅ **Better calibration (lower ECE)** |

**Key insight**: Accuracy should be similar across methods. The **ECE (Expected Calibration Error)** is the critical metric - MI method should have **lower ECE** (better calibrated confidence scores).

## Quick Test (5 examples)

Test all three methods on the same 5 examples:

```bash
# 1. Greedy baseline (~30 seconds)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 5 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_greedy_5.csv

# 2. Self-consistency baseline (~2 minutes)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-easy --limit 5 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_selfcons_5.csv

# 3. MI method (~3 minutes)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/mi_method_5.csv
```

**Compare the JSON files:**
```bash
cat outputs/results/baseline_greedy_5.json
cat outputs/results/baseline_selfcons_5.json
cat outputs/results/mi_method_5.json
```

Look for:
- ✅ Similar `accuracy` across methods
- ✅ Lower `ece` for MI method (key result!)
- ✅ Different `avg_confidence` patterns

## Small Test (50 examples)

Run on 50 examples for more robust comparison:

```bash
# 1. Greedy baseline (~3 minutes)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge --limit 50 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_greedy_50.csv

# 2. Self-consistency (~25 minutes)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge --limit 50 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_selfcons_50.csv

# 3. MI method (~30 minutes)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 50 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/mi_method_50.csv
```

## Full Evaluation

For publication-quality results, run on complete datasets:

### ARC-Challenge (1,172 examples)

```bash
# Greedy (~35 minutes)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/arc_challenge_greedy.csv

# Self-consistency (~6 hours with k=10)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/arc_challenge_selfcons.csv

# MI method (~7 hours with k=10, n=2)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/arc_challenge_mi.csv
```

### OpenBookQA (500 examples)

```bash
# Greedy (~15 minutes)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/openbookqa_greedy.csv

# Self-consistency (~3 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/openbookqa_selfcons.csv

# MI method (~3 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/openbookqa_mi.csv
```

## Analyzing Results

### 1. Quick Comparison Script

Create a simple comparison script:

```python
import json

methods = ['greedy', 'selfcons', 'mi']
dataset = 'arc_challenge_50'

print(f"{'Method':<20} {'Accuracy':<12} {'ECE':<12} {'Avg Conf':<12}")
print("="*60)

for method in methods:
    with open(f'outputs/results/baseline_{method}_{dataset}.json' if method != 'mi' 
              else f'outputs/results/mi_method_{dataset}.json') as f:
        data = json.load(f)
        print(f"{method:<20} {data['accuracy']:<12.4f} {data['ece']:<12.4f} {data['avg_confidence']:<12.4f}")
```

### 2. Key Metrics to Compare

**Accuracy** (all methods should be similar):
```bash
# Should be within ±2% across methods
grep "accuracy" outputs/results/*_50.json
```

**ECE - Expected Calibration Error** (MI should be lowest):
```bash
# Lower is better - MI should win here!
grep "ece" outputs/results/*_50.json
```

**Confidence Distribution**:
- Greedy: Often overconfident (high confidence, low ECE)
- Self-consistency: Variable based on agreement
- MI: Better calibrated (confidence matches actual correctness)

### 3. Statistical Significance

For robust results, run on at least 200 examples per method. With 50 examples, trends should be visible but may have higher variance.

## Cost & Time Estimates

### For 50 Examples

| Method | API Calls | Time (L4 GPU) | Relative Cost |
|--------|-----------|---------------|---------------|
| Greedy | 50 | ~3 min | 1× |
| Self-Consistency (k=10) | 500 | ~25 min | 10× |
| MI (k=10, n=2) | 1,000 | ~30 min | 20× |

### For Full ARC-Challenge (1,172 examples)

| Method | API Calls | Time (L4 GPU) | Relative Cost |
|--------|-----------|---------------|---------------|
| Greedy | 1,172 | ~35 min | 1× |
| Self-Consistency (k=10) | 11,720 | ~6 hours | 10× |
| MI (k=10, n=2) | 23,440 | ~7 hours | 20× |

## Recommended Workflow

**Phase 1: Quick Validation** (1 hour total)
1. Run all 3 methods on 5 examples
2. Verify MI has lower ECE
3. Check that accuracies are similar

**Phase 2: Small-Scale Test** (2 hours total)
1. Run all 3 methods on 50 examples
2. Analyze ECE differences
3. Verify trends match paper's findings

**Phase 3: Full Evaluation** (20+ hours total)
1. Run greedy baseline on all datasets (~2 hours)
2. Run self-consistency on key dataset (~6 hours)
3. Run MI method on all datasets (~14 hours)

## Important Notes

### Cache Behavior
- **Greedy** (temp=0): Cache enabled ✅ - reuses results
- **Self-consistency** (temp=0.9): Cache disabled ❌ - preserves diversity
- **MI** (temp=0.9): Cache disabled ❌ - preserves diversity

### Temperature Settings
- **Greedy**: Always temperature=0 (deterministic)
- **Self-consistency**: Use temperature=0.9 (from paper)
- **MI**: Use temperature=0.9 (from paper)

### Fair Comparison
To fairly compare methods:
1. ✅ Use same dataset and split
2. ✅ Use same model and quantization
3. ✅ Use same max_tokens
4. ✅ Use same k for self-consistency and MI
5. ✅ Use temperature=0.9 for sampling methods

## Example Output Comparison

After running all three methods on 50 examples, you should see:

```
GREEDY BASELINE:
accuracy      : 0.5800
ece           : 0.1523
avg_confidence: 0.7234

SELF-CONSISTENCY:
accuracy      : 0.6000
ece           : 0.1345
avg_confidence: 0.7100

MI METHOD:
accuracy      : 0.6000
ece           : 0.0892  ← BEST (lowest ECE)
avg_confidence: 0.6543
```

**Key finding**: MI method achieves similar accuracy but **better calibration** (lower ECE)!

## Troubleshooting

### "All methods give same ECE"
- Check that you're using different methods (check the logs)
- Ensure sample size is large enough (at least 50 examples)
- Verify MI scores are non-zero (check CSV files)

### "MI method takes too long"
- Reduce k (try k=5 instead of k=10)
- Reduce n (try n=1, though n=2 is from paper)
- Start with smaller datasets (5 or 50 examples)

### "Greedy baseline has best ECE"
- This can happen on small samples (luck)
- Run on larger sample (200+ examples)
- Check that confidence scores are being computed correctly

## Next Steps

1. ✅ Run quick 5-example test to verify setup
2. ✅ Run 50-example test for all methods
3. ✅ Analyze ECE differences
4. ✅ If MI wins on ECE, proceed to full evaluation
5. ✅ Document findings and create comparison plots

---

**Remember**: The goal is to show that MI method achieves **better calibration (lower ECE)** while maintaining similar accuracy. This is the paper's main contribution!

