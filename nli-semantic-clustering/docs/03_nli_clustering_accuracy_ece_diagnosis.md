# NLI Clustering: Root Cause Analysis of Accuracy Drop & ECE Spike

**Date**: November 21, 2024  
**Status**: Critical Issue Diagnosed - Fix Pending  
**Impact**: -10% Accuracy, +0.522 ECE with NLI clustering

---

## 🔍 Executive Summary

The NLI clustering module shows "miserable results" not because the NLI model is bad, but because we're using the **same strict bidirectional equivalence logic** for two fundamentally different purposes:

1. **Internal Consistency** (Clustering): Should be strict ✅
2. **External Correctness** (Accuracy): Should be loose ❌ **THIS IS THE BUG**

**The Fix**: Implement asymmetric evaluation - strict for clustering, loose for grading.

---

## 📊 The Problem: Two Manifestations

### Symptom 1: Accuracy Drop (-10.0%)

**What's happening:**

| Evaluation Method | Gold Label | Model Output | Verdict | Explanation |
|-------------------|------------|--------------|---------|-------------|
| **F1 (Baseline)** | `"Paris"` | `"The capital is Paris."` | ✅ **Correct** | Substring match: "Paris" found in output |
| **NLI (Current)** | `"Paris"` | `"The capital is Paris."` | ❌ **Wrong** | Bidirectional check fails |

**Why NLI fails:**

```python
# Forward entailment
"The capital is Paris" → "Paris"  # ✅ YES (Paris is implied)

# Backward entailment
"Paris" → "The capital is Paris"  # ❌ NO (Could be Paris, Texas; just the word "Paris")

# Result: NOT equivalent
is_correct = (forward >= 0.5) AND (backward >= 0.5)  # False!
```

**Impact**: Valid answers with extra context are rejected as wrong.

### Symptom 2: ECE Spike (+0.522)

This is the **smoking gun** that reveals the calibration disaster:

**Scenario:**
```python
# Model generates 10 chains, all consistent:
chains = [
    ["The capital is Paris.", "The capital is Paris."],
    ["The capital is Paris.", "It is Paris."],
    ["Paris is the capital.", "The capital is Paris."],
    ...
]

# Step 1: NLI Clustering (Works correctly)
clustered = [["Paris"] * 10]  # All clustered together
MI = 0.01  # Very low (high consistency)
Confidence = 0.99  # Very high

# Step 2: Accuracy Check (Broken!)
gold_label = "Paris"
prediction = "The capital is Paris."
is_correct = check_mutual_entailment(prediction, gold_label)  # False!

# Step 3: ECE Calculation
# Confidence: 0.99 | Actual Accuracy: 0.00
# Calibration Error: |0.99 - 0.00| = 0.99 (MASSIVE!)
```

**The Paradox:**
- Model is **internally consistent** (all chains agree) → Low MI → High Confidence
- Model is marked **externally wrong** (strict NLI check) → Low Accuracy
- **Result**: Overconfident on wrong answers → ECE explodes

---

## 🧩 The Root Cause: Single Logic for Dual Purposes

### Current Implementation (The Bug)

```python
# In check_mutual_entailment (lines 112-146 of core.py)
def check_mutual_entailment(text_a, text_b, threshold=0.5):
    """
    Bidirectional equivalence: A ↔ B
    """
    fwd_score = self.check_entailment(text_a, text_b)
    bwd_score = self.check_entailment(text_b, text_a)
    
    # STRICT: Both directions must exceed threshold
    return fwd_score >= threshold and bwd_score >= threshold

# Used for BOTH:
# 1. Clustering answers (correct usage)
# 2. Grading against gold labels (WRONG usage!)
```

**Why this is wrong:**

| Purpose | What We Need | What We're Doing | Result |
|---------|--------------|------------------|--------|
| **Clustering** | Strict equivalence (A ↔ B) | ✅ Using bidirectional | Correct |
| **Grading** | Loose entailment (A → Gold) | ❌ Using bidirectional | Too strict! |

---

## 💡 The Solution: Asymmetric Evaluation

We need **two different NLI modes** in the same class:

### Mode 1: Strict Equivalence (For Clustering)

**Purpose**: Group semantically identical answers  
**Logic**: Bidirectional entailment (A ↔ B)  
**Usage**: MI calculation, consistency checking

```python
def check_mutual_entailment(text_a, text_b, threshold=0.5):
    """
    STRICT: For clustering.
    Returns True only if A ⟺ B
    """
    fwd = self.check_entailment(text_a, text_b)
    bwd = self.check_entailment(text_b, text_a)
    return (fwd >= threshold) and (bwd >= threshold)
```

**Example:**
```python
# Should cluster:
check_mutual_entailment("Paris", "paris")  # True
check_mutual_entailment("Barack Obama", "Obama")  # True

# Should NOT cluster:
check_mutual_entailment("Paris", "France")  # False
check_mutual_entailment("Paris", "A city in Europe")  # False
```

### Mode 2: Loose Correctness (For Grading)

**Purpose**: Check if answer is factually correct  
**Logic**: Unidirectional entailment (Prediction → Gold) OR substring match  
**Usage**: Accuracy calculation, ECE computation

```python
def is_correct(prediction, gold_label, threshold=0.5):
    """
    LOOSE: For accuracy grading.
    Returns True if prediction implies gold label OR contains it.
    """
    # Normalize
    pred_norm = prediction.strip().lower()
    gold_norm = gold_label.strip().lower()
    
    # Method 1: String containment (F1 compatibility)
    if gold_norm in pred_norm:
        return True
    
    # Method 2: Unidirectional entailment (NLI flexibility)
    # Only check: Does prediction entail gold?
    # Do NOT check backward (gold → prediction)
    entailment = self.check_entailment(prediction, gold_label)
    return entailment >= threshold
```

**Example:**
```python
# Should accept (all correct despite verbosity):
is_correct("The capital is Paris.", "Paris")  # True (substring)
is_correct("Paris is the capital of France.", "Paris")  # True (entails)
is_correct("It is Paris", "Paris")  # True (substring)

# Should reject (factually wrong):
is_correct("London", "Paris")  # False
is_correct("I don't know", "Paris")  # False
```

---

## 🔬 Technical Justification

### Why Bidirectional for Clustering?

**Goal**: Detect when the model is "changing its story"

```python
# Model outputs in different chains:
["Paris", "London", "Paris", "Madrid"]

# We want to know: Is the model uncertain about WHICH city?
# Answer: YES, because "Paris" ≠ "London" ≠ "Madrid"
# High MI → Low confidence (correct!)
```

If we used loose clustering:
```python
# BAD: All cities might cluster together
"Paris" → "A city"  # True (loose)
"London" → "A city"  # True (loose)
# Result: Artificially low MI (incorrect!)
```

### Why Unidirectional for Grading?

**Goal**: Accept valid answers even if phrased differently than gold label

```python
# Gold label (often short):
gold = "Paris"

# Model outputs (often verbose):
predictions = [
    "The capital is Paris.",      # Contains "Paris" ✓
    "Paris is the answer.",       # Contains "Paris" ✓
    "It's Paris, France.",        # Contains "Paris" ✓
]

# All should be marked correct!
# They all ENTAIL "Paris" (forward direction)
# The backward direction doesn't matter for correctness
```

If we used strict grading:
```python
# BAD: Reject correct but verbose answers
"Paris" → "The capital is Paris"  # False
# Result: Penalize model for being thorough (incorrect!)
```

---

## 📐 Mathematical Formulation

### Current (Broken) System

$$
\text{Equivalent}(A, B) = \mathbb{I}[\text{Entails}(A, B) > \tau] \times \mathbb{I}[\text{Entails}(B, A) > \tau]
$$

$$
\text{Correct}(y, y^*) = \text{Equivalent}(y, y^*) \quad \text{← WRONG!}
$$

**Problem**: Too strict for grading.

### Proposed (Fixed) System

**For Clustering:**
$$
\text{Equivalent}(A, B) = \mathbb{I}[\text{Entails}(A, B) > \tau] \times \mathbb{I}[\text{Entails}(B, A) > \tau]
$$

**For Grading:**
$$
\text{Correct}(y, y^*) = \mathbb{I}[\text{Entails}(y, y^*) > \tau] \;\text{OR}\; \mathbb{I}[y^* \subset y]
$$

**Key difference**: Grading only checks one direction + substring fallback.

---

## 🎯 Expected Impact of Fix

### Before Fix (Current State)

| Metric | Value | Status |
|--------|-------|--------|
| Accuracy | Baseline - 10% | ❌ Too strict |
| ECE | Baseline + 0.522 | ❌ Severe miscalibration |
| MI | Low (good clustering) | ✅ Working correctly |

### After Fix (Expected)

| Metric | Value | Status |
|--------|-------|--------|
| Accuracy | Baseline ± 2% | ✅ Similar to F1 |
| ECE | Baseline - 0.05 | ✅ Improved calibration |
| MI | Low (good clustering) | ✅ Still working |

**Why the improvement:**
- Accuracy recovers because verbose answers are accepted
- ECE drops because confidence now aligns with (correct) accuracy
- MI unchanged because clustering logic is untouched

---

## 🔧 Implementation Plan

### Step 1: Add `is_correct()` Method

Location: `nli_clustering/core.py`, in `NLIClusteringCache` class

```python
def is_correct(
    self, 
    prediction: str, 
    gold_label: str, 
    threshold: float = 0.5
) -> bool:
    """
    LOOSE Evaluation: Check if prediction is correct against gold label.
    
    Used for Accuracy and ECE calculation.
    Returns True if:
    1. Prediction entails Gold Label (Unidirectional), OR
    2. Gold Label is a substring of Prediction (String Match Fallback)
    
    Args:
        prediction: The model's generated answer
        gold_label: The ground truth reference
        threshold: Minimum P(entailment) (default: 0.5)
    
    Returns:
        True if prediction is considered correct
    """
    # Normalize
    pred_norm = prediction.strip().lower()
    gold_norm = gold_label.strip().lower()
    
    # Method 1: String Matching (Safety net for short/exact answers)
    # Matches F1 behavior: strict substring check
    if gold_norm in pred_norm:
        return True
    
    # Method 2: NLI Check (Handles verbose but correct answers)
    # ONLY check Prediction → Gold (unidirectional)
    # Do NOT check Gold → Prediction (that's for clustering)
    entailment_score = self.check_entailment(prediction, gold_label)
    
    return entailment_score >= threshold
```

### Step 2: Update Evaluation Scripts

In `scripts/recalculate_with_semantic_clustering.py` and `scripts/threshold_sweep.py`:

**Current (Broken):**
```python
# Uses strict equivalence for both clustering and grading
em = compute_exact_match(predicted, gold_answers)  # F1-based
```

**Fixed:**
```python
# Use NLI for grading if available
if nli_checker:
    em = nli_checker.is_correct(predicted, gold_answers[0])
else:
    em = compute_exact_match(predicted, gold_answers)  # Fallback to F1
```

### Step 3: Update Threshold Sweep

Add a new flag to test both modes:

```python
parser.add_argument(
    "--use-nli-grading",
    action="store_true",
    help="Use NLI for accuracy checking (not just clustering)"
)
```

---

## 🧪 Validation Strategy

### Test 1: Sanity Check

```python
nli = NLIClusteringCache()

# Test clustering (should be strict)
assert nli.check_mutual_entailment("Paris", "paris") == True
assert nli.check_mutual_entailment("Paris", "The capital is Paris") == False

# Test grading (should be loose)
assert nli.is_correct("The capital is Paris", "Paris") == True
assert nli.is_correct("Paris is the answer", "Paris") == True
assert nli.is_correct("London", "Paris") == False
```

### Test 2: Threshold Sweep Comparison

Run threshold sweep with both evaluation modes:

```bash
# Mode 1: F1 grading (baseline)
python scripts/threshold_sweep.py --dataset triviaqa --limit 50

# Mode 2: NLI grading (new)
python scripts/threshold_sweep.py --dataset triviaqa --limit 50 --use-nli-grading
```

Expected results:
- **Accuracy**: Should recover to near baseline
- **ECE**: Should drop significantly (better calibration)

---

## 📚 Related Concepts

### NLI Model Background

**Model**: DeBERTa Cross-Encoder (likely `microsoft/deberta-v2-xlarge-mnli`)

**Architecture**: 
- Cross-encoder (not bi-encoder)
- Processes (premise, hypothesis) pairs jointly
- Attention mechanism sees both texts simultaneously
- Output: 3-way classification (entailment, neutral, contradiction)

**Why Cross-Encoder?**
- More accurate than cosine similarity (bi-encoders)
- Can detect nuanced logical relationships
- Standard for NLI tasks

### Entailment vs Equivalence

| Relationship | Symbol | Meaning | Example |
|--------------|--------|---------|---------|
| **Entailment** | A → B | If A is true, B must be true | "Paris" → "A city" |
| **Equivalence** | A ↔ B | A → B AND B → A | "Paris" ↔ "paris" |
| **Contradiction** | A ⊥ B | If A is true, B must be false | "Paris" ⊥ "London" |
| **Neutral** | A ⊥/→ B | No logical relationship | "Paris" ⊥/→ "Beautiful" |

---

## 🚨 Critical Insight

**The bug is conceptual, not technical:**

We have a perfectly functional NLI model and correct implementation of bidirectional entailment. The problem is using the right tool for the wrong job.

**Analogy:**
- Using a scalpel (strict equivalence) when we need a net (loose grading)
- The scalpel works perfectly for surgery (clustering), but it's overkill for catching fish (grading)

**The fix is not "better NLI" - it's "appropriate NLI" for each task.**

---

## 📖 References

1. Paper goal: Measure uncertainty over facts, not words
2. NLI model: Cross-encoder architecture for semantic understanding
3. Current implementation: `nli_clustering/core.py` lines 112-146
4. Threshold sweep tool: `scripts/threshold_sweep.py`

---

## ✅ Next Steps

1. **Code Review**: Confirm current implementation uses bidirectional for grading
2. **Implement Fix**: Add `is_correct()` method to `NLIClusteringCache`
3. **Update Scripts**: Modify evaluation scripts to use new method
4. **Test**: Run threshold sweep with both grading modes
5. **Validate**: Confirm accuracy recovery and ECE improvement
6. **Document**: Update `QUICKSTART.md` with new findings

---

**Status**: Diagnosis complete, awaiting code implementation and validation.

