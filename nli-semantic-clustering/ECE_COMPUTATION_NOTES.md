# ECE Computation Notes

## FIXED ✓ - ECE Bug Corrected in Both Folders

### The Bug

The original `compute_ece()` implementation was **mathematically incorrect**:

```python
# BUGGY (original):
bin_accuracy = (predictions[in_bin] == labels[in_bin]).mean()  # Always 1.0 when pred==label!
ECE = sum(bin_weight * |1.0 - bin_confidence|)  # Wrong!
```

This computed `|1 - confidence|` instead of the correct calibration metric.

### The Fix

Updated in **both** folders:
- `nli-semantic-clustering/nli_clustering/utils.py`
- `llm-belief-mi-test/llm_belief_mi_test/calibration.py`

```python
# CORRECTED:
bin_accuracy = labels[in_bin].mean()  # Fraction of correct answers in bin
ECE = sum(bin_weight * |actual_accuracy - bin_confidence|)  # Correct!
```

### ECE Values Comparison

**TriviaQA Greedy (200 questions):**
- Buggy ECE: 0.601
- **Corrected ECE: 0.126** ✓

### New Insight: Calibration Degradation After NLI Clustering

With the **corrected ECE formula**, we now see the real problem:

**TriviaQA Self-Consistency (20 questions):**
- Acc Orig: 0.450 | Acc NLI: 0.550 | Δ Acc: **+0.100** ✓ (accuracy improved!)
- ECE Orig: 0.185 | ECE NLI: 0.450 | Δ ECE: **+0.265** ✗ (calibration degraded!)
- Avg Confidence Clustered: **1.000** (all samples overconfident!)

**Root Cause:**
1. NLI clustering reduces MI to ~0 (eliminates uncertainty)
2. MI = 0 → `confidence_clustered = 1.0` for all questions
3. But actual accuracy after NLI grading is only 55%
4. Calibration error: |1.0 - 0.55| = 0.45 (very poor!)

**The Trade-off:**
- ✓ NLI grading increases accuracy (better semantic matching)
- ✗ MI-based confidence becomes overconfident (worse calibration)

### Possible Solutions

**Option A: Use Original Confidence for ECE NLI**

Measure: "Does NLI improve accuracy while holding confidence constant?"

```python
ece_clustered = compute_ece(
    predictions=em_clustered,
    confidences=confidence_original,  # Keep original confidence
    labels=em_clustered
)
```

**Option B: Report Separate Metrics**

Add columns for MI and confidence changes to show the uncertainty reduction explicitly.

**Option C: Recalibrate Confidence After Clustering**

Develop a better confidence estimator that accounts for both MI and NLI grading quality.

---

## Test Results Summary (CORRECTED ECE)

### TriviaQA Greedy (200 questions, threshold=0.5)
- Acc Orig: 0.500 | Acc NLI: 0.650 | Δ Acc: **+0.150** ✓
- ECE Orig: 0.126 | ECE NLI: 0.251 | Δ ECE: **+0.126** ✗
- (No clustering because greedy = 1 sample, but NLI grading changes EM)

### TriviaQA Self-Consistency (20 questions, threshold=0.5)
- Acc Orig: 0.450 | Acc NLI: 0.550 | Δ Acc: **+0.100** ✓
- ECE Orig: 0.185 | ECE NLI: 0.450 | Δ ECE: **+0.265** ✗
- Clustering: 19.7% reduction
- **Problem: confidence_clustered = 1.0 for all questions (overconfident!)**

### Key Findings

1. **Accuracy improves** with NLI grading (+10-15%)
2. **Calibration degrades** significantly because MI-based confidence becomes 1.0
3. The corrected ECE formula reveals the true calibration problem that was hidden by the bug

---

## Next Steps

1. Consider using **Option A** (original confidence for ECE comparison)
2. Or develop a better confidence estimator that works with NLI clustering
3. The goal should be to **improve both accuracy AND calibration**, not just one

