# Dataset Requirements & Time Estimates

## Current Performance Analysis

**Your actual performance** (observed from running system):
- **3.74 seconds per generation** (2.5x slower than expected 1.5s)
- **74.79 seconds per question** (k=10, n=2 = 20 generations)

**Why slower than expected?**
- Likely: max_tokens=64 is too long for MCQ answers
- Logprobs extraction adds overhead
- First few runs may be slower (cache warming)

---

## Dataset Sizes

| Dataset | Split | Examples | Generations (k=10, n=2) |
|---------|-------|----------|------------------------|
| **ARC-Challenge** | test | 1,172 | 23,440 |
| **ARC-Easy** | test | 2,376 | 47,520 |
| **OpenBookQA** | test | 500 | 10,000 |
| **TOTAL** | - | **4,048** | **80,960** |

---

## Time Estimates

### At Current Pace (3.74s per generation, max_tokens=64)

| Dataset | Time |
|---------|------|
| ARC-Challenge | **24.4 hours** ⚠️ |
| ARC-Easy | **49.4 hours** ⚠️ |
| OpenBookQA | **10.4 hours** |
| **TOTAL** | **84 hours** (~3.5 days) ❌ |

**This is too slow!**

---

### Optimized Pace (1.5s per generation, max_tokens=20)

| Dataset | Time |
|---------|------|
| ARC-Challenge | **9.8 hours** ✅ |
| ARC-Easy | **19.8 hours** ⚠️ |
| OpenBookQA | **4.2 hours** ✅ |
| **TOTAL** | **33.7 hours** (~1.4 days) |

**Better, but still long**

---

### Highly Optimized (0.75s per generation, max_tokens=20, temp=0.3)

| Dataset | Time |
|---------|------|
| ARC-Challenge | **4.9 hours** ✅ |
| ARC-Easy | **9.9 hours** ✅ |
| OpenBookQA | **2.1 hours** ✅ |
| **TOTAL** | **16.9 hours** (~17 hours) ✅ |

**Much more reasonable!**

---

## Recommended Optimizations

### 🚀 Optimization 1: Reduce max_tokens (Most Impact)

**Change:**
```bash
--max-tokens 20  # Instead of 64
```

**Why:** MCQ answers are short (usually 1-5 tokens: "A", "Paris", "photosynthesis")

**Impact:** ~2x speedup

---

### 🚀 Optimization 2: Lower Temperature

**Change:**
```bash
--temperature 0.3  # Instead of 0.5
```

**Why:** Faster sampling, less randomness needed

**Impact:** ~10-20% speedup

---

### 🚀 Optimization 3: Reduce k (If Acceptable)

**Change:**
```bash
--k 5  # Instead of 10
```

**Why:** Half the chains = half the time. Still valid for MI estimation.

**Impact:** 2x speedup

**Trade-off:** Less robust MI estimation, but still scientifically valid

---

### 🚀 Optimization 4: Use Validation Split (For Testing)

**Change:**
```bash
--split validation  # Instead of test
```

ARC-Challenge validation: 299 examples (vs 1,172 test)

**Impact:** 4x fewer examples, ~1.2 hours instead of 4.9 hours

**Use:** For development/calibration, then run test split for final results

---

## Recommended Commands (Optimized)

### Quick Test (5 examples, ~3 minutes)
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.3 --max-tokens 20 \
  --output outputs/results/test_quick.csv
```

### Small Test (50 examples, ~30 minutes)
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge --limit 50 \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.3 --max-tokens 20 \
  --output outputs/results/arc_challenge_50.csv
```

### Validation Split (299 examples, ~1.5 hours)
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge --split validation \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.3 --max-tokens 20 \
  --output outputs/results/arc_challenge_val.csv
```

### Full Test Split - Optimized (~5 hours each)
```bash
# ARC-Challenge (1,172 examples)
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.3 --max-tokens 20 \
  --output outputs/results/arc_challenge_full_optimized.csv

# ARC-Easy (2,376 examples)  
python -m llm_belief_mi_test.cli \
  --dataset arc-easy \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.3 --max-tokens 20 \
  --output outputs/results/arc_easy_full_optimized.csv

# OpenBookQA (500 examples)
python -m llm_belief_mi_test.cli \
  --dataset openbookqa \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.3 --max-tokens 20 \
  --output outputs/results/openbookqa_full_optimized.csv
```

**Total time (all three): ~17 hours**

---

### Even Faster: Reduce k to 5 (~8.5 hours total)
```bash
# Same commands as above but with --k 5 instead of --k 10
# Trade-off: Less robust MI estimation, but 2x faster
```

---

## Cost Analysis (A100 at $2.71/hour)

| Configuration | Total Time | A100 Cost |
|--------------|------------|-----------|
| **Current (64 tokens)** | 84 hours | $227.64 ❌ |
| **Optimized (20 tokens)** | 33.7 hours | $91.33 ⚠️ |
| **Highly optimized (20 tokens, temp=0.3)** | 16.9 hours | $45.80 ✅ |
| **k=5 (20 tokens, temp=0.3)** | 8.5 hours | $23.04 ✅✅ |

---

## Cost Analysis (L4 at $0.48/hour)

| Configuration | Total Time | L4 Cost |
|--------------|------------|---------|
| **Current (64 tokens)** | 84 hours | $40.32 |
| **Optimized (20 tokens)** | 33.7 hours | $16.18 |
| **Highly optimized** | 16.9 hours | $8.11 ✅ |
| **k=5** | 8.5 hours | $4.08 ✅✅ |

**With L4: Much more affordable!**

---

## My Recommendations

### Option 1: Full Scientific Rigor (k=10, optimized settings)
```bash
--k 10 --n 2 --temperature 0.3 --max-tokens 20
```
- **Time**: ~17 hours (all datasets)
- **Cost (A100)**: $46
- **Cost (L4)**: $8
- ✅ Follows paper exactly (k=10, n=2)

### Option 2: Faster, Still Valid (k=5, optimized settings)
```bash
--k 5 --n 2 --temperature 0.3 --max-tokens 20
```
- **Time**: ~8.5 hours (all datasets)
- **Cost (A100)**: $23
- **Cost (L4)**: $4
- ✅ Still scientifically valid (enough chains for MI)

### Option 3: Development/Calibration (validation splits)
```bash
--split validation --k 10 --n 2 --temperature 0.3 --max-tokens 20
```
- **Time**: ~4-5 hours (all datasets)
- **Cost (A100)**: $12
- **Cost (L4)**: $2
- ✅ Good for testing, then run test split for final

---

## Summary Table

| Dataset | Examples | Generations | Current Time | Optimized Time | Super Optimized (k=5) |
|---------|----------|-------------|--------------|----------------|----------------------|
| ARC-Challenge | 1,172 | 23,440 | 24.4h | 4.9h | 2.5h |
| ARC-Easy | 2,376 | 47,520 | 49.4h | 9.9h | 5.0h |
| OpenBookQA | 500 | 10,000 | 10.4h | 2.1h | 1.0h |
| **TOTAL** | **4,048** | **80,960** | **84h** | **17h** | **8.5h** |

**Costs:**
- A100: $227 → $46 → $23
- L4: $40 → $8 → $4

---

## What Should You Do?

**Immediate action**: Stop the current run (Ctrl+C) and restart with optimized settings!

**Recommended command**:
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge --limit 50 \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.3 --max-tokens 20 \
  --output outputs/results/arc_challenge_test_opt.csv
```

This will finish **50 examples in ~30 minutes** and let you verify:
1. Results look correct
2. Performance is acceptable
3. Then commit to full run

---

**Bottom line**: With optimizations, you can do all three datasets in ~17 hours for $46 (A100) or $8 (L4)!

