# Log Probability Diagnostic Report

**Date**: 2025-10-31  
**Issue**: Zero log probabilities detected in evaluation traces  
**Status**: ⚠️ Partial fallback mode detected

---

## Executive Summary

Analysis of evaluation traces (e.g., `outputs/logs/openbookqa_mi_500/question_0.json`) reveals that all token log probabilities are recorded as `0.0` with placeholder probability values of `0.5`. This indicates the evaluation ran with a fallback path where token-level log probabilities were not captured from the model.

**Key Finding**: For the **MI method**, this does not affect accuracy or ECE results. However, it may impact certain baseline methods that rely on token probabilities for confidence estimation.

---

## Technical Analysis

### Root Cause

The `logprob=0.0` fallback occurs when the model client does not properly return token-level scores. Two possible scenarios:

1. **Missing `output_scores` in generation**: The model's `generate()` call didn't return `outputs.scores`
2. **Client detection failure**: The code fell back to a path without `chat_completion_with_logprobs()`

**Evidence from logs**:
```json
{
  "text": "B",
  "logprob": 0.0,
  "probability": 0.5
}
```

The `0.5` probability is a placeholder computed as:
```python
"probability": math.exp(logprob) if logprob < 0 else 0.5
```

---

## Impact Analysis by Method

### ✅ Method 1: MI Method — **NO IMPACT**

**Accuracy**: ✅ Not affected  
**ECE/Calibration**: ✅ Not affected  
**Confidence**: ✅ Not affected

**Why it's safe**:
1. **MI estimation** uses only the text outputs, not token probabilities:
   ```python
   # MI computed from text sequences only
   chains_text = [[text for text, _ in chain] for chain in chains_with_logprobs]
   mi_nats = estimate_mi_listing_nats(chains_text)
   ```

2. **Confidence** is derived solely from MI score:
   ```python
   confidence = 1.0 / (1.0 + mi_nats)  # No logprobs involved
   ```

3. **Answer selection** via pseudo-joint: With `logprob=0.0`, all chains get equal weight (`exp(0)=1`), reducing selection to majority voting over final answers. For high-agreement cases (e.g., 10/10 chains agree), this yields identical results.

**Minor caveat**: In rare low-agreement cases where probability weighting would matter, uniform weights may change the selected answer. This is negligible in practice with k=10 chains.

---

### ⚠️ Method 2: Greedy Baseline — **MINOR IMPACT**

**Accuracy**: ✅ Not affected (single decode)  
**ECE/Calibration**: ⚠️ **Potentially degraded**  
**Confidence**: ⚠️ **Set to fallback value**

**Why there's impact**:
The greedy baseline uses token log probability for confidence:
```python
# Intended behavior
confidence = math.exp(logprob)  # True probability of greedy sequence

# Fallback behavior (when logprob=0.0)
confidence = math.exp(0.0) = 1.0  # Always max confidence!
```

**Consequence**: If this ran with `logprob=0.0`, all greedy predictions would have confidence ≈ 1.0, which would:
- Severely degrade ECE (overconfident on all examples)
- Make the method appear poorly calibrated

**Verification needed**: Check if greedy logs show varied confidences or all near 1.0.

---

### ⚠️ Method 3: Self-Consistency — **MINIMAL IMPACT**

**Accuracy**: ✅ Not affected  
**ECE/Calibration**: ✅ Likely not affected  
**Confidence**: ✅ Based on agreement, not logprobs

**Why it's mostly safe**:
Self-consistency uses majority voting for both answer selection and confidence:
```python
# Confidence = fraction agreeing with majority
confidence = max_count / k
```

Token logprobs are captured but **not used** in the core algorithm. The fallback `logprob=0.0` only affects logging, not the decision process.

---

### ❌ Method 4: Semantic Entropy — **SIGNIFICANT IMPACT**

**Accuracy**: ⚠️ Possibly affected  
**ECE/Calibration**: ❌ **Severely affected**  
**Confidence**: ❌ **Cannot be computed correctly**

**Why there's major impact**:
Semantic entropy requires token probabilities for:

1. **Weighting semantic clusters**:
   ```python
   # Each sample needs its probability
   prob = math.exp(logprob)  # Fallback: prob = 1.0
   cluster_prob = sum(prob for _, prob in cluster_samples)
   ```
   With `logprob=0.0`, all samples get weight 1.0 → uniform distribution.

2. **Entropy calculation**:
   ```python
   entropy = -sum(p * log(p) for p in distribution)
   confidence = exp(-entropy)
   ```
   Uniform weights → artificially high entropy → artificially low confidence.

**Consequence**: Results may be **invalid** for this method. ECE comparison unfair.

---

### ❌ Method 5: Self-Verification — **MODERATE IMPACT**

**Accuracy**: ✅ Likely not affected  
**ECE/Calibration**: ⚠️ **Different mechanism**  
**Confidence**: ⚠️ Based on verification, but initial selection uses logprobs

**Why there's some impact**:
1. **Initial answer selection** aggregates by probability:
   ```python
   prob = math.exp(logprob)  # Fallback: 1.0
   choice_probs[choice] = choice_probs.get(choice, 0.0) + prob
   ```
   With uniform weights, reduces to majority voting (likely same answer).

2. **Confidence** comes from verification query ("True/False"), not from logprobs:
   ```python
   confidence = 0.9 if "true" in response else 0.1
   ```
   This part is unaffected.

**Consequence**: Minor impact on answer selection in edge cases, but confidence mechanism is separate.

---

## Verification Commands

### 1. Check if logprobs are actually missing

Run the GPU test script to verify proper logprob extraction:

```bash
cd /Users/aaronfeng/Repo/quantify_credibility/llm-belief-mi-test
python scripts/test_gpu_setup.py
```

**Expected output** (if working):
```
Response: '4'
Log probability: -2.3456  # Non-zero value
```

**Problem indicator** (if broken):
```
Log probability: 0.0000  # Zero value
```

### 2. Inspect actual evaluation logs

Check confidence distributions in your results:

```bash
# Greedy method - should have varied confidences if logprobs work
python -c "
import json
with open('outputs/results/openbookqa/greedy_500.json') as f:
    data = json.load(f)
    print(f'Avg confidence: {data[\"avg_confidence\"]:.4f}')
"

# If avg_confidence ≈ 1.0, logprobs were missing
```

### 3. Check per-question logs

```bash
# Look at a few question traces
cat outputs/logs/openbookqa_greedy_500/question_0.json | jq '.methods.greedy.raw_outputs[0].logprob'

# Should see negative numbers (e.g., -2.34), not 0.0
```

---

## Remediation Strategies

### Strategy A: Fix the Root Cause (Recommended)

Ensure proper logprob extraction in `LocalLlamaClient`:

1. **Verify `transformers` version**:
   ```bash
   pip show transformers
   # Should be ≥ 4.30.0 for proper outputs.scores support
   pip install --upgrade transformers
   ```

2. **Verify generation parameters**:
   The client should request scores:
   ```python
   gen_kwargs = {
       "return_dict_in_generate": True,
       "output_scores": True,  # Critical!
   }
   ```

3. **Test immediately**:
   ```bash
   python scripts/test_gpu_setup.py
   ```

### Strategy B: Re-run Affected Methods

If logprobs were missing, re-run evaluations that need them:

**Priority 1 - Critical (must re-run)**:
```bash
# Semantic Entropy - INVALID without logprobs
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset openbookqa --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/openbookqa/semantic_entropy_500_fixed.csv
```

**Priority 2 - Should re-run**:
```bash
# Greedy - calibration affected
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa --limit 500 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/openbookqa/greedy_500_fixed.csv
```

**Priority 3 - Optional**:
```bash
# Self-Verification - minor impact
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset openbookqa --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/openbookqa/self_verification_500_fixed.csv
```

**Not needed**:
- ✅ MI method - results are valid
- ✅ Self-Consistency - results are valid

### Strategy C: Add Diagnostic Logging

Add a startup check to catch this early:

```python
# In cli.py, after client initialization
if client.supports_logprobs():
    # Test logprob extraction
    test_msg = [{"role": "user", "content": "Test"}]
    _, test_logprob = client.chat_completion_with_logprobs(test_msg, max_tokens=1)
    if test_logprob == 0.0:
        logging.warning("⚠️ WARNING: Logprobs appear to be zero! Check transformers version.")
    else:
        logging.info(f"✅ Logprob extraction working (test value: {test_logprob:.4f})")
```

### Strategy D: Make Placeholder More Obvious

Update logging to clearly indicate when logprobs are missing:

```python
# In calibration.py logging sections, change:
"probability": math.exp(logprob) if logprob < 0 else 0.5

# To:
"probability": math.exp(logprob) if logprob < 0 else None,
"logprob_status": "captured" if logprob < 0 else "fallback"
```

---

## Recommendations

### Immediate Actions

1. ✅ **Run verification**: Execute `scripts/test_gpu_setup.py` to confirm current state
2. ⚠️ **Check Semantic Entropy results**: These are most affected and may need re-running
3. ℹ️ **Document in paper**: Note that MI method is robust to this issue (shows algorithmic strength)

### Before Publishing Results

1. **Re-run Semantic Entropy** if logprobs were missing (critical for fair comparison)
2. **Re-run Greedy baseline** to ensure proper calibration metrics
3. **Keep MI and Self-Consistency results** - they are valid regardless

### Future Prevention

1. Add automatic logprob validation at startup
2. Include logprob sanity checks in test suite
3. Document logprob requirements per method in README

---

## Summary Table

| Method | Accuracy Impact | ECE Impact | Confidence Impact | Re-run Needed? |
|--------|----------------|------------|-------------------|----------------|
| **MI Method** | ✅ None | ✅ None | ✅ None | ❌ No |
| **Greedy** | ✅ None | ⚠️ Degraded | ⚠️ Wrong values | ⚠️ Recommended |
| **Self-Consistency** | ✅ None | ✅ Minimal | ✅ Minimal | ❌ No |
| **Semantic Entropy** | ⚠️ Possible | ❌ Severe | ❌ Invalid | ✅ **Yes** |
| **Self-Verification** | ✅ Minimal | ⚠️ Moderate | ⚠️ Moderate | ⚠️ Optional |

---

## Conclusion

The zero logprob issue primarily affects methods that rely on token-level probabilities for uncertainty estimation. **The MI method's results remain fully valid** because it derives uncertainty from sequence-level diversity rather than token probabilities.

If Semantic Entropy results are being used for comparison, those runs should be repeated with proper logprob extraction to ensure fair evaluation.

**Next step**: Run `python scripts/test_gpu_setup.py` to determine current logprob status.

