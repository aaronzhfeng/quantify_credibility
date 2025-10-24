# Quick Start: Baseline Comparisons

## TL;DR - Run This Now! 🚀

To properly evaluate the MI method, you need to compare it against baselines. Here's the fastest way to get started:

### 1. Test Everything Works (2 minutes)
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test
python test_baselines.py
```

This will test all three methods on 3 examples and verify they work correctly.

### 2. Quick Comparison on 5 Examples (5 minutes)
```bash
# Greedy baseline (~30 sec)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 5 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_greedy_5.csv

# Self-consistency baseline (~2 min)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-easy --limit 5 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_selfcons_5.csv

# MI method (~3 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/mi_method_5.csv

# Compare results
python compare_results.py outputs/results/*_5.json
```

### 3. What You Should See
```
BASELINE COMPARISON
============================================================================
Method                    Accuracy     ECE          Avg Conf     
----------------------------------------------------------------------------
MI Method                 0.6000       0.0892       0.6543       ⭐ BEST
Self-Consistency          0.6000       0.1345       0.7100       
Greedy                    0.5800       0.1523       0.7234       
============================================================================

✅ MI method has best calibration (key paper result!)
```

**Key**: MI method should have **lower ECE** (better calibration) than baselines!

---

## Three Methods Explained

### 🎯 Greedy Baseline
- **What**: Single greedy decode (temp=0)
- **Cost**: Cheapest (1 gen/question)
- **Time**: Fastest (~3 min for 50 examples)
- **Use**: Quick baseline

### 🎲 Self-Consistency
- **What**: k samples + majority vote
- **Cost**: Medium (10 gens/question)
- **Time**: ~25 min for 50 examples
- **Use**: Fair sampling comparison

### 🧠 MI Method (Paper's Approach)
- **What**: k chains + MI estimation
- **Cost**: Highest (20 gens/question)
- **Time**: ~30 min for 50 examples
- **Use**: Should have **best ECE**

---

## Next: Run on 50 Examples (1 hour total)

For more reliable results:

```bash
# ARC-Challenge, 50 examples each method

# Greedy (~3 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge --limit 50 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_greedy_50.csv

# Self-consistency (~25 min)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge --limit 50 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/baseline_selfcons_50.csv

# MI method (~30 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 50 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 \
  --output outputs/results/mi_method_50.csv

# Compare
python compare_results.py outputs/results/*_50.json
```

---

## Pro Tip: Run Incrementally

**Day 1**: 5 examples (verify it works)
```bash
# See commands above
```

**Day 2**: 50 examples (robust comparison)
```bash
# See commands above
```

**Day 3+**: Full dataset (publication quality)
```bash
# See BASELINE_COMPARISON_GUIDE.md
```

---

## What to Look For

### ✅ Expected (Paper's Claims)
- **Similar accuracy** across all methods (~50-65%)
- **Lower ECE** for MI method (e.g., 0.08 vs 0.15)
- **Better calibration** = confidence scores match actual correctness

### ❌ Potential Issues
- If greedy has best ECE on 5 examples → Normal variance, run on 50
- If all methods have same ECE → Increase sample size
- If MI is much slower → Try `--k 5` instead of `--k 10`

---

## Files Created

- ✅ `BASELINE_COMPARISON_GUIDE.md` - Comprehensive guide
- ✅ `compare_results.py` - Compare JSON results
- ✅ `test_baselines.py` - Verify methods work
- ✅ Modified CLI to support `--method` flag

---

## Remember

**The point of baselines**: Show that MI method achieves **better calibration (lower ECE)** while maintaining similar accuracy. This validates the paper's main contribution! 🎯

For detailed information, see:
- **BASELINE_COMPARISON_GUIDE.md** - Full guide
- **BASELINE_METHODS_ADDED.md** - Technical details
- **README.md** - Updated with baseline examples

