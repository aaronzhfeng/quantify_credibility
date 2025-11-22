# Using NLI Grading Mode: Fixing the Accuracy Drop

**Date**: November 21, 2024  
**Status**: Implemented and Ready for Testing  
**Related**: `01_nli_clustering_accuracy_ece_diagnosis.md`

---

## 🎯 Quick Summary

We've implemented a **dual-mode NLI system** that fixes the accuracy drop issue:

1. **Clustering Mode** (strict): Uses bidirectional equivalence → for MI calculation
2. **Grading Mode** (loose): Uses unidirectional entailment → for accuracy checking

**New flag**: `--use-nli-grading` enables loose grading while keeping strict clustering.

---

## 🚀 How to Use

### Basic Usage

```bash
# OLD: Uses F1 grading (baseline)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/triviaqa_baseline.json \
  --thresholds 0.5 \
  --correctness-based

# NEW: Uses NLI grading (should fix accuracy drop)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/triviaqa_nli_grading.json \
  --thresholds 0.5 \
  --correctness-based \
  --use-nli-grading  # ← NEW FLAG
```

### Recalculation Script

```bash
# Recalculate with NLI grading
python scripts/recalculate_with_semantic_clustering.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/triviaqa_nli_grading_recalc.json \
  --correctness-based \
  --nli-threshold 0.5 \
  --use-nli-grading  # ← NEW FLAG
```

---

## 📊 Expected Results

### Before Fix (Without --use-nli-grading)

```
Threshold    Clusters   Acc Orig   Acc NLI    Δ Acc      Changed
------------------------------------------------------------------------------
0.50         28.3%      0.450      0.340      -0.110     12.4%  ← BAD
```

**Problem**: Strict bidirectional check rejects "The capital is Paris" for gold="Paris"

### After Fix (With --use-nli-grading)

```
Threshold    Clusters   Acc Orig   Acc NLI    Δ Acc      Changed
------------------------------------------------------------------------------
0.50         28.3%      0.450      0.445      -0.005     12.4%  ← FIXED
```

**Fixed**: Loose unidirectional check accepts verbose but correct answers

---

## 🔧 What Changed

### New Method in `nli_clustering/core.py`

```python
class NLIClusteringCache:
    
    def check_mutual_entailment(self, text_a, text_b, threshold=0.5):
        """
        STRICT: For clustering.
        Checks A ↔ B (bidirectional)
        """
        # Implementation unchanged
    
    def is_correct(self, prediction, gold_label, threshold=0.5):
        """
        LOOSE: For grading.
        Checks prediction → gold OR substring match
        """
        # NEW METHOD
        # 1. Check substring (fast path)
        if gold_label.lower() in prediction.lower():
            return True
        
        # 2. Check unidirectional entailment
        # ONLY check: Does prediction entail gold?
        # Do NOT check: Does gold entail prediction?
        return self.check_entailment(prediction, gold_label) >= threshold
```

### Key Differences

| Aspect | check_mutual_entailment() | is_correct() |
|--------|---------------------------|--------------|
| **Purpose** | Clustering | Grading |
| **Logic** | A ↔ B (bidirectional) | A → B OR substring |
| **Strictness** | High | Low |
| **Use Case** | MI calculation | Accuracy/ECE |
| **Example** | "Paris" ≠ "The capital is Paris" | "The capital is Paris" = "Paris" ✓ |

---

## 🧪 Testing

### Run Validation Tests

```bash
# Test the new is_correct() method
python scripts/test_is_correct.py
```

**Expected output:**
```
✅ PASS | Exact match
✅ PASS | Case insensitive
✅ PASS | Substring match
✅ PASS | Contains gold label
✅ PASS | Verbose but correct
❌ PASS | Wrong answer (correctly rejected)
```

### Manual Testing

```python
from nli_clustering.core import NLIClusteringCache

nli = NLIClusteringCache()

# Test clustering (should be strict)
print(nli.check_mutual_entailment("Paris", "The capital is Paris"))
# → False (not equivalent)

# Test grading (should be loose)
print(nli.is_correct("The capital is Paris", "Paris"))
# → True (correct despite verbosity)
```

---

## 📈 Comparison Study

### Experiment 1: Baseline vs NLI Grading

```bash
# Run both modes on same data
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/f1_grading.json \
  --thresholds 0.5 \
  --correctness-based \
  --limit 50

python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/nli_grading.json \
  --thresholds 0.5 \
  --correctness-based \
  --use-nli-grading \
  --limit 50

# Compare results
python -c "
import json
f1 = json.load(open('results/f1_grading.json'))
nli = json.load(open('results/nli_grading.json'))

print(f\"F1 Grading Accuracy: {f1['threshold_summary'][0.5]['accuracy_clustered']:.3f}\")
print(f\"NLI Grading Accuracy: {nli['threshold_summary'][0.5]['accuracy_clustered']:.3f}\")
print(f\"Improvement: {(nli['threshold_summary'][0.5]['accuracy_clustered'] - f1['threshold_summary'][0.5]['accuracy_clustered']):.3f}\")
"
```

### Experiment 2: Threshold Sensitivity

Test if NLI grading is less sensitive to threshold:

```bash
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/nli_threshold_sensitivity.json \
  --thresholds 0.3 0.4 0.5 0.6 0.7 \
  --correctness-based \
  --use-nli-grading \
  --limit 50
```

**Hypothesis**: Accuracy should be more stable across thresholds with NLI grading.

---

## 🔍 Debugging Tips

### Check Entailment Scores

If results are unexpected, inspect raw entailment scores:

```python
from nli_clustering.core import NLIClusteringCache

nli = NLIClusteringCache()

# Get scores for a specific case
pred = "The capital of France is Paris."
gold = "Paris"

fwd, bwd = nli.get_entailment_scores(pred, gold)
print(f"Forward (pred → gold): {fwd:.3f}")
print(f"Backward (gold → pred): {bwd:.3f}")

# Check both grading modes
print(f"Mutual entailment: {nli.check_mutual_entailment(pred, gold)}")
print(f"Is correct: {nli.is_correct(pred, gold)}")
```

### Identify Problematic Cases

Find questions where grading mode matters:

```bash
python -c "
import json
data = json.load(open('results/triviaqa_nli_grading.json'))

print('Cases where NLI grading helps:')
for q in data['per_question_results']:
    for t in q['threshold_results']:
        if t['threshold'] == 0.5:
            if t['em_change'] > 0:  # NLI improved accuracy
                print(f\"Q: {q['question_text'][:60]}\")
                print(f\"  Pred: {t['predicted_clustered']}\")
                print(f\"  Gold: {q['gold_answers']}\")
                print()
"
```

---

## ⚠️ Important Notes

### When to Use NLI Grading

**Use `--use-nli-grading` when:**
- ✅ Model generates verbose answers (e.g., "The capital is Paris")
- ✅ Gold labels are short (e.g., "Paris")
- ✅ You care about semantic correctness over exact wording
- ✅ Dataset: TriviaQA (knowledge-based QA)

**Don't use `--use-nli-grading` when:**
- ❌ Dataset: SQuAD v2 (extractive span matching)
- ❌ Answers are already short and concise
- ❌ You need exact string matching for downstream tasks

### Clustering vs Grading

**Remember**: The flag only affects **grading**, not **clustering**!

```python
# With --use-nli-grading:

# Clustering (unchanged):
"Paris" and "The capital is Paris" → separate clusters (correct)

# Grading (changed):
"The capital is Paris" vs gold="Paris" → correct (fixed!)
```

---

## 🎓 Technical Details

### Why Unidirectional Works

**Formal logic:**

```
Prediction: "The capital of France is Paris"
Gold: "Paris"

Forward entailment:
P("Paris" | "The capital of France is Paris") = 0.95 ✓

Backward entailment:
P("The capital of France is Paris" | "Paris") = 0.15 ✗

Result:
- Mutual entailment: False (0.95 ∧ 0.15 = False)
- Is correct: True (0.95 ≥ 0.5 = True)
```

**Key insight**: We don't care if "Paris" alone implies the full statement. We only care if the full statement implies "Paris" (which it does).

### Substring Fallback

The method uses substring matching as first check:

```python
if gold_norm in pred_norm:
    return True  # Fast path, no NLI needed
```

**Why?**
1. Speed: String matching is instant
2. Compatibility: Maintains F1-like behavior for simple cases
3. Safety: Ensures short exact answers always work

---

## 📚 Examples

### Example 1: Verbose Answer

```
Gold: "Paris"
Prediction: "The capital of France is Paris."

F1 Grading: Correct (substring match)
NLI (mutual): Wrong (not bidirectionally equivalent)
NLI (is_correct): Correct (unidirectional + substring) ✓
```

### Example 2: Paraphrase

```
Gold: "Barack Obama"
Prediction: "President Obama"

F1 Grading: Wrong (no substring match)
NLI (mutual): Wrong (not bidirectionally equivalent)
NLI (is_correct): Correct (unidirectional entailment) ✓
```

### Example 3: Wrong Answer

```
Gold: "Paris"
Prediction: "London"

F1 Grading: Wrong ✓
NLI (mutual): Wrong ✓
NLI (is_correct): Wrong ✓
```

---

## 🔄 Migration Path

### Step 1: Validate on Sample

```bash
# Test on 20 questions first
python scripts/test_is_correct.py
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/test_nli_grading.json \
  --thresholds 0.5 \
  --correctness-based \
  --use-nli-grading \
  --limit 20
```

### Step 2: Full Dataset

```bash
# Run on full 200 questions
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/triviaqa_nli_grading_full.json \
  --thresholds 0.4 0.5 0.6 \
  --correctness-based \
  --use-nli-grading
```

### Step 3: Compare and Decide

```python
import json
baseline = json.load(open('results/baseline.json'))
nli_grading = json.load(open('results/nli_grading_full.json'))

# Check if fix worked
acc_improvement = nli_grading['...']['accuracy_clustered'] - baseline['...']['accuracy_clustered']
print(f"Accuracy improvement: {acc_improvement:+.3f}")

if acc_improvement > -0.02:
    print("✅ Fix successful! Accuracy recovered.")
else:
    print("⚠️  Accuracy still dropping. Try alternative models.")
```

### Step 4: Port to Main Repo

If successful, port the changes back to `llm-belief-mi-test`:

1. Copy `is_correct()` method to `calibration.py`
2. Update evaluation functions to use it
3. Add `--use-nli-grading` flag to CLI
4. Update `COMMANDS_NLI.md` with new usage

---

## 📖 References

- Diagnosis document: `docs/01_nli_clustering_accuracy_ece_diagnosis.md`
- Test script: `scripts/test_is_correct.py`
- Implementation: `nli_clustering/core.py` lines 148-215
- Related issue: ECE spike (+0.522) explained in diagnosis doc

---

## ✅ Success Criteria

The fix is successful if:

1. **Accuracy**: Recovers to within 2% of baseline
2. **ECE**: Improves by at least 0.05
3. **MI**: Remains low (clustering still works)
4. **Tests**: All validation tests pass

**Next Step**: Run threshold sweep with `--use-nli-grading` and verify results!

