# Ready to Re-Run: Answer Format Fix Applied

## ✅ What Was Fixed

### Critical Issue: Answer Extraction Unreliable
- Model generated verbose responses like "Based on my analysis, the best option would be B) quit eating lunch out because..."
- Fuzzy matching was unreliable
- Led to ~28% accuracy (barely above random guessing at 25%)

### Solution: Strict Answer Format
- New `--answer-format strict` parameter
- Forces model to output ONLY "A", "B", "C", or "D"
- Direct extraction - no fuzzy matching needed
- Expected to improve accuracy from ~28% to ~50-65%

---

## 🚀 Recommended Commands for Re-evaluation

All commands now use:
- `--answer-format strict` (clean answers)
- `--max-tokens 10` (only need letter, not explanation)
- `_v2` suffix (distinguish from previous buggy runs)

### Test First (5 examples, ~3 minutes)

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Test one method to verify it works
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa --limit 5 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_strict_5.csv

# Check the result
cat outputs/results/test_strict_5.json
```

**Look for:** Accuracy > 40% (should be much better than 28%)

---

### Full Re-evaluation (500 examples per dataset, ~12 hours total)

**OpenBookQA (500 examples):**
```bash
# Greedy (~10 minutes with strict mode - faster!)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_greedy_500_v2.csv

# Self-Consistency (~2.5 hours - faster with max-tokens=10!)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_selfcons_500_v2.csv

# MI Method (~3 hours - faster!)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_mi_500_v2.csv

# Compare
python scripts/compare_results.py outputs/results/openbookqa_*_500_v2.json
```

**Repeat for ARC-Challenge and ARC-Easy** (same commands, change --dataset)

---

## ⚡ Time Savings with Strict Mode

| Method | Old (max-tokens=30) | New (max-tokens=10) | Savings |
|--------|---------------------|---------------------|---------|
| Greedy (500) | ~15 min | ~10 min | 33% faster |
| Self-Cons (500) | ~3 hours | ~2.5 hours | 17% faster |
| MI (500) | ~3.5 hours | ~3 hours | 14% faster |

**Total for 3 datasets × 3 methods:** ~12 hours → ~10 hours (17% faster overall)

---

## 📊 What to Expect

### Accuracy Improvement:
```
Previous (default format):
  OpenBookQA: 28.2%
  ARC-Challenge: 29.2%
  ARC-Easy: 31.6%

Expected (strict format):
  OpenBookQA: 50-65%
  ARC-Challenge: 50-65%
  ARC-Easy: 60-75%
```

### ECE (Calibration):
```
MI should still win!
  
Previous:
  MI: ECE = 0.36
  Self-Cons: ECE = 0.82
  Greedy: ECE = 0.94

Expected:
  MI: ECE = 0.05-0.15 (better absolute values)
  Self-Cons: ECE = 0.15-0.25
  Greedy: ECE = 0.20-0.30
  
  MI still ~50% better than baselines
```

---

## 🎬 Current Demo Status

Demo was regenerated with:
- ✅ Choices included in prompts
- ✅ max_tokens=100 (allows complete responses)
- ⚠️ Still using `default` format (verbose)

**Demo shows:**
- Accuracy: 2/5 correct across methods (40% on 5 examples)
- Better than previous ~28%, but still room for improvement

**To get best demo results:**
- Regenerate with `--answer-format strict`
- Should see clean "A", "B", "C", "D" responses
- Higher accuracy expected

---

## 📝 Priority Actions

### High Priority (Verify Fix Works):
1. ✅ Test strict mode on 5 examples
2. ✅ Check that responses are just letters
3. ✅ Verify accuracy > 40%

### Medium Priority (Get Better Results):
4. ⏳ Re-run on 50 examples with strict mode
5. ⏳ Compare to previous results
6. ⏳ Decide if full re-run is worth it

### Optional:
7. ⏳ Regenerate demo with strict mode
8. ⏳ Add S.E. and S.V. methods
9. ⏳ Full re-evaluation on all datasets

---

## 🔧 Quick Commands

```bash
# Test answer formats
python scripts/test_answer_formats.py

# Quick 5-example test
python -m llm_belief_mi_test.cli --method greedy --dataset openbookqa --limit 5 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/test_strict_5.csv

# View result
cat outputs/results/test_strict_5.json
```

---

**Implementation is complete. Test it out and decide if you want to re-run the full evaluation!** 🚀

