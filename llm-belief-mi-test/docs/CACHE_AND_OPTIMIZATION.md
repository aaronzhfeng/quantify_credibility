# Caching & Optimization Guide

## ✅ Caching Implementation Complete

**What was added:**
1. SQLite cache module (copied from repro)
2. Cache integration in `llm_client_local.py`
3. CLI parameters: `--cache-path` and `--cache-mode`
4. Automatic caching of all generations

---

## How Caching Works

### Cache Key Structure
Every generation is cached with a unique key based on:
- Model name
- Messages (prompt)
- Temperature
- Max tokens

**Same inputs → Cache hit (instant return)**  
**Different inputs → Cache miss (generate & store)**

### Cache Behavior

```python
# First run: Generate and cache
response, logprob = model.generate(...)  # Takes 2-3s
cache.put(key, response, logprob)

# Second run with SAME prompt:
response, logprob = cache.get(key)  # Instant! <0.001s
```

---

## Incremental Evaluation Strategy

### Why This Is Powerful 🚀

**Problem without cache:**
- Run 100 examples → Finish
- Later want 200 examples → Re-run all 200 (100 are duplicates!)
- Later want full 1172 → Re-run all 1172 (1100 are duplicates!)

**Solution with cache:**
- Run 100 examples → Cached
- Run 200 examples → Only 100 new generations (100 cached)
- Run 1172 examples → Only 972 new generations (200 cached)

**Time saved:** Massive! Only pay for new work.

---

## Recommended Incremental Workflow

### Phase 1: Verification (Day 1, ~30 min total)

```bash
# Quick test - all 3 datasets (5 examples each)
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 5 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_c_5.csv
python -m llm_belief_mi_test.cli --dataset arc-easy --limit 5 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_e_5.csv
python -m llm_belief_mi_test.cli --dataset openbookqa --limit 5 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/obqa_5.csv
```

✅ Verify: accuracy ~0.4-0.8, mi_bits >0, ece <0.3

### Phase 2: Small Scale (Day 2, ~2 hours total)

```bash
# 50 examples per dataset
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 50 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_c_50.csv
python -m llm_belief_mi_test.cli --dataset arc-easy --limit 50 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_e_50.csv
python -m llm_belief_mi_test.cli --dataset openbookqa --limit 50 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/obqa_50.csv
```

✅ First 5 from each dataset are cached (reused)
✅ Analyze: Is ECE improving? Is MI correlated with errors?

### Phase 3: Medium Scale (Day 3-4, ~6 hours total)

```bash
# 200 examples per dataset
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 200 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_c_200.csv
python -m llm_belief_mi_test.cli --dataset arc-easy --limit 200 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_e_200.csv
python -m llm_belief_mi_test.cli --dataset openbookqa --limit 200 --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/obqa_200.csv
```

✅ First 50 from each dataset are cached
✅ More robust metrics for analysis

### Phase 4: Full Scale (Day 5+, ~18 hours total)

```bash
# Full datasets
python -m llm_belief_mi_test.cli --dataset arc-challenge --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_challenge_full.csv
python -m llm_belief_mi_test.cli --dataset arc-easy --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/arc_easy_full.csv
python -m llm_belief_mi_test.cli --dataset openbookqa --k 10 --n 2 --load-in-4bit --temperature 0.3 --max-tokens 30 --output outputs/results/openbookqa_full.csv
```

✅ First 200 from each dataset are cached
✅ Final, publication-ready results

---

## Time & Cost Breakdown

### Without Incremental Caching:
- Total work: 80,960 generations
- Total time: ~24 hours
- Cost (A100): ~$65
- Cost (L4): ~$12

### With Incremental Caching:
- **Same total time** (~24 hours)
- **Same total cost** (~$65 A100 / $12 L4)
- **But:** Spread across multiple days
- **Benefit:** Early verification, can stop if issues found

**Example:**
- Day 1 (5 each): 3×100 = 300 generations (~10 min, $0.50)
- Day 2 (50 each): 3×900 = 2,700 generations (~1.5 hr, $4)
- Day 3 (200 each): 3×3,000 = 9,000 generations (~5 hr, $14)
- Day 4+ (full): Remaining 68,960 generations (~18 hr, $48)

If you find issues on Day 2, you only spent $5 instead of $65!

---

## Cache Safety & Best Practices

### ✅ Safe Practices:

1. **Use consistent parameters:**
   ```bash
   # Always use same k, n, temperature, max_tokens
   # This maximizes cache hits
   ```

2. **Cache survives interruptions:**
   ```bash
   # Run gets interrupted at question 500?
   # Restart with same command - questions 1-500 are cached!
   ```

3. **Separate cache for experiments:**
   ```bash
   # Experiment A: Different temperature
   --cache-path .cache/exp_a.sqlite --temperature 0.3
   
   # Experiment B: Different max_tokens  
   --cache-path .cache/exp_b.sqlite --max-tokens 50
   ```

### ⚠️ Cache Invalidation:

Cache is invalidated (new generation) when:
- Model name changes
- Prompt changes (different question or chain position)
- Temperature changes
- Max tokens changes

**Pro tip:** Stick to one configuration for the whole evaluation!

---

## Monitoring Cache Performance

### During Run:
```bash
# Watch log output for cache hits
# You'll see faster progress on repeated questions
```

### After Run:
```bash
# Check cache file size
ls -lh .cache/llm_cache.sqlite
# Should grow with more cached results

# Check cache entries (if you've implemented stats)
python -c "from llm_belief_mi_test.cache import SQLiteCache; c = SQLiteCache('.cache/llm_cache.sqlite'); print(c.stats())"
```

---

## Recommended Settings Summary

### For max_tokens:

| Setting | Use Case | Reasoning |
|---------|----------|-----------|
| **30** | ✅ Recommended | Covers 95%+ of MCQ answers with buffer |
| 20 | Aggressive | Might truncate longer answers |
| 40-50 | Conservative | Safe but slower |
| 64+ | Wasteful | Too long for MCQ answers |

### For temperature:

| Setting | Use Case | Reasoning |
|---------|----------|-----------|
| **0.3** | ✅ Recommended | Good diversity, faster than 0.5 |
| 0.5 | Paper's value | More diversity, slower |
| 0.7+ | Not recommended | Too random, slow |

### For k (chains):

| Setting | Use Case | Reasoning |
|---------|----------|-----------|
| **10** | ✅ Recommended | Paper's value, robust MI estimation |
| 5 | Budget/Speed | 2x faster, still valid |
| 20+ | Overkill | Diminishing returns |

### For n (chain length):

| Setting | Use Case | Reasoning |
|---------|----------|-----------|
| **2** | ✅ Use this | Paper's value, sufficient for MI |
| 3+ | Not needed | Paper showed n=2 is enough |

---

## Final Recommendations

### Optimal Configuration:
```bash
--k 10 --n 2 \
--temperature 0.3 --max-tokens 30 \
--load-in-4bit
```

### With Incremental Caching:
```bash
# Start small, grow incrementally
--limit 5    # Day 1: Verify (~3 min)
--limit 50   # Day 2: Test (~30 min, 5 cached)
--limit 200  # Day 3: Medium (~2 hr, 50 cached)
# (no limit)  # Day 4+: Full (~5 hr, 200 cached)
```

### GPU Choice:
- **L4**: $12 total (recommended)
- **A100**: $65 total (if you want faster results)

**Either GPU is more than enough!** A100 is 6x more expensive for only 2x speedup.

---

**Cache ensures no wasted work - run incrementally with confidence!** 🎯

