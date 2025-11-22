# ECE Bug Fix Summary

## What Was Fixed

The `compute_ece()` function had a **mathematical bug** that made it measure the wrong metric.

### The Bug

```python
# BUGGY (original):
bin_accuracy = (predictions[in_bin] == labels[in_bin]).mean()
```

When `predictions == labels` (both are correctness arrays), this always equals `1.0`, making ECE measure `|1 - confidence|` instead of `|actual_accuracy - confidence|`.

### The Fix

```python
# CORRECTED:
bin_accuracy = labels[in_bin].mean()
```

This correctly computes the fraction of correct answers in each confidence bin.

## Files Modified

### New Module (nli-semantic-clustering)
1. `nli_clustering/utils.py` - Line 306: Changed `(predictions[in_bin] == labels[in_bin]).mean()` → `labels[in_bin].mean()`
2. `scripts/threshold_sweep.py` - Line 333-347: Updated comments to reflect corrected ECE computation

### Old Module (llm-belief-mi-test)
3. `llm_belief_mi_test/calibration.py` - Line 109: Changed `(predictions[in_bin] == labels[in_bin]).mean()` → `labels[in_bin].mean()`

## Impact on Results

### ECE Baseline Values (TriviaQA Greedy, 200 questions)
- **Before (buggy):** ECE = 0.601
- **After (corrected):** ECE = 0.126 ✓

The corrected ECE is ~5x lower, showing that the baseline model is actually **better calibrated** than the buggy metric suggested.

### New Insight: NLI Clustering Degrades Calibration

With the corrected ECE, we now see the real problem:

**TriviaQA Self-Consistency (20 questions, threshold=0.5):**
- Accuracy: 0.450 → **0.550** (+10% improvement ✓)
- ECE: 0.185 → **0.450** (+0.265 degradation ✗)

**Why?**
- After NLI clustering: MI → 0, so `confidence = 1.0` for all questions
- But actual accuracy is only 55%
- Calibration error: |1.0 - 0.55| = 0.45 (very poor!)

**The Trade-off:**
- ✓ NLI grading increases accuracy
- ✗ MI-based confidence becomes overconfident

## Verification

Both implementations tested and verified:

```bash
# New module
ECE (nli-semantic-clustering): 0.126 ✓

# Old module  
ECE (llm-belief-mi-test): 0.126 ✓
```

## Next Steps

Consider these options for addressing the calibration degradation:

1. **Option A:** Use original confidence for both ECE Orig and ECE NLI (hold confidence constant)
2. **Option B:** Develop a better confidence estimator that works with NLI clustering
3. **Option C:** Report MI and confidence changes as separate metrics

The goal should be to improve **both accuracy AND calibration**, not sacrifice one for the other.

