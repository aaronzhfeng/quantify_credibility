# 🔬 NLI Threshold Sweep Commands

Comprehensive threshold testing across **all 3 methods** and **2 datasets**.

---

## 📊 Available Data

```
data/
├── triviaqa/
│   ├── logs_greedy/       ✅ 200 questions (1 answer each)
│   ├── logs_selfcons/     ✅ 200 questions (10 samples each)
│   └── logs_mi/           ✅ 200 questions (20 inferences: 10 chains × 2 steps)
└── squad_v2/
    ├── logs_greedy/       ✅ 200 questions (1 answer each)
    ├── logs_selfcons/     ✅ 200 questions (10 samples each)
    └── logs_mi/           ✅ 200 questions (20 inferences: 10 chains × 2 steps)
```

**All data uses exact-match (F1) based evaluation** - no NLI applied yet.

---

## 🎯 What Each Method Tests

| Method | NLI Clustering | NLI Grading | What We Learn |
|--------|---------------|-------------|---------------|
| **Greedy** | ❌ N/A (only 1 answer) | ✅ Yes | Does NLI grading improve accuracy without clustering? |
| **Self-Consistency** | ✅ Yes (clusters 10 samples) | ✅ Yes | Does clustering + NLI grading improve majority voting? |
| **MI** | ✅ Yes (clusters 20 inferences) | ✅ Yes | Does clustering reduce MI? Does NLI improve calibration (ECE)? |

---

## 🚀 Quick Test (All 6 combinations, ~10 minutes)

Test on first 20 questions to identify promising thresholds:

```bash
cd /root/quantify_credibility/nli-semantic-clustering
mkdir -p results/threshold_sweeps

# === TriviaQA (3 methods × 20 questions) ===

# 1. Greedy (accuracy only)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_greedy \
  --output results/threshold_sweeps/triviaqa_greedy_quick.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --use-nli-grading \
  --limit 20

# 2. Self-Consistency (clustering + accuracy)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_selfcons \
  --output results/threshold_sweeps/triviaqa_selfcons_quick.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --correctness-based \
  --use-nli-grading \
  --limit 20

# 3. MI (clustering + MI + ECE)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/threshold_sweeps/triviaqa_mi_quick.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --correctness-based \
  --use-nli-grading \
  --limit 20

# === SQuAD v2 (3 methods × 20 questions) ===

# 4. Greedy (accuracy only)
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_greedy \
  --output results/threshold_sweeps/squad_v2_greedy_quick.json \
  --thresholds 0.5 0.6 0.7 0.8 0.9 \
  --use-nli-grading \
  --limit 20

# 5. Self-Consistency (clustering + accuracy)
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_selfcons \
  --output results/threshold_sweeps/squad_v2_selfcons_quick.json \
  --thresholds 0.5 0.6 0.7 0.8 0.9 \
  --use-nli-grading \
  --limit 20

# 6. MI (clustering + MI + ECE)
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_mi \
  --output results/threshold_sweeps/squad_v2_mi_quick.json \
  --thresholds 0.5 0.6 0.7 0.8 0.9 \
  --use-nli-grading \
  --limit 20
```

**One-liner to run all quick tests:**
```bash
cd /root/quantify_credibility/nli-semantic-clustering && mkdir -p results/threshold_sweeps && \
python scripts/threshold_sweep.py --log-dir data/triviaqa/logs_greedy --output results/threshold_sweeps/triviaqa_greedy_quick.json --thresholds 0.4 0.5 0.6 0.7 0.8 --use-nli-grading --limit 20 && \
python scripts/threshold_sweep.py --log-dir data/triviaqa/logs_selfcons --output results/threshold_sweeps/triviaqa_selfcons_quick.json --thresholds 0.4 0.5 0.6 0.7 0.8 --correctness-based --use-nli-grading --limit 20 && \
python scripts/threshold_sweep.py --log-dir data/triviaqa/logs_mi --output results/threshold_sweeps/triviaqa_mi_quick.json --thresholds 0.4 0.5 0.6 0.7 0.8 --correctness-based --use-nli-grading --limit 20 && \
python scripts/threshold_sweep.py --log-dir data/squad_v2/logs_greedy --output results/threshold_sweeps/squad_v2_greedy_quick.json --thresholds 0.5 0.6 0.7 0.8 0.9 --use-nli-grading --limit 20 && \
python scripts/threshold_sweep.py --log-dir data/squad_v2/logs_selfcons --output results/threshold_sweeps/squad_v2_selfcons_quick.json --thresholds 0.5 0.6 0.7 0.8 0.9 --use-nli-grading --limit 20 && \
python scripts/threshold_sweep.py --log-dir data/squad_v2/logs_mi --output results/threshold_sweeps/squad_v2_mi_quick.json --thresholds 0.5 0.6 0.7 0.8 0.9 --use-nli-grading --limit 20 && \
echo "✅ All quick tests complete!"
```

⏱️ **Time**: ~10 minutes (2 min per method × 6 = 12 min with overhead)

---

## 🔬 Full Evaluation (All 200 questions, ~60 minutes)

After identifying promising thresholds from quick test, run full evaluation:

```bash
cd /root/quantify_credibility/nli-semantic-clustering
mkdir -p results/threshold_sweeps

# === TriviaQA Full (200 questions each) ===

# 1. Greedy
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_greedy \
  --output results/threshold_sweeps/triviaqa_greedy_full.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --use-nli-grading

# 2. Self-Consistency
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_selfcons \
  --output results/threshold_sweeps/triviaqa_selfcons_full.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --correctness-based \
  --use-nli-grading

# 3. MI
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/threshold_sweeps/triviaqa_mi_full.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --correctness-based \
  --use-nli-grading

# === SQuAD v2 Full (200 questions each) ===

# 4. Greedy
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_greedy \
  --output results/threshold_sweeps/squad_v2_greedy_full.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --use-nli-grading

# 5. Self-Consistency
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_selfcons \
  --output results/threshold_sweeps/squad_v2_selfcons_full.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --use-nli-grading

# 6. MI
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_mi \
  --output results/threshold_sweeps/squad_v2_mi_full.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --use-nli-grading
```

**One-liner to run all full tests:**
```bash
cd /root/quantify_credibility/nli-semantic-clustering && mkdir -p results/threshold_sweeps && \
python scripts/threshold_sweep.py --log-dir data/triviaqa/logs_greedy --output results/threshold_sweeps/triviaqa_greedy_full.json --thresholds 0.5 0.6 0.7 --use-nli-grading && \
python scripts/threshold_sweep.py --log-dir data/triviaqa/logs_selfcons --output results/threshold_sweeps/triviaqa_selfcons_full.json --thresholds 0.5 0.6 0.7 --correctness-based --use-nli-grading && \
python scripts/threshold_sweep.py --log-dir data/triviaqa/logs_mi --output results/threshold_sweeps/triviaqa_mi_full.json --thresholds 0.5 0.6 0.7 --correctness-based --use-nli-grading && \
python scripts/threshold_sweep.py --log-dir data/squad_v2/logs_greedy --output results/threshold_sweeps/squad_v2_greedy_full.json --thresholds 0.6 0.7 0.8 --use-nli-grading && \
python scripts/threshold_sweep.py --log-dir data/squad_v2/logs_selfcons --output results/threshold_sweeps/squad_v2_selfcons_full.json --thresholds 0.6 0.7 0.8 --use-nli-grading && \
python scripts/threshold_sweep.py --log-dir data/squad_v2/logs_mi --output results/threshold_sweeps/squad_v2_mi_full.json --thresholds 0.6 0.7 0.8 --use-nli-grading && \
echo "✅ All full evaluations complete!"
```

⏱️ **Time**: ~60 minutes (10 min per method × 6 = 60 min)

---

## 📊 Analyze Results

### Quick Summary View

```bash
# View all quick test results
for file in results/threshold_sweeps/*_quick.json; do
  echo "=== $(basename $file) ==="
  python -c "
import json
data = json.load(open('$file'))
for tr in data['threshold_results']:
    t = tr['threshold']
    s = tr['summary']
    acc_change = s.get('avg_em_change', 0)
    clusters = s.get('avg_clustering_reduction_pct', 0)
    print(f'Threshold {t:.2f}: Acc Δ {acc_change:+.3f}, Clusters {clusters:.1f}%')
"
  echo
done
```

### Detailed Comparison Table

```python
import json
import pandas as pd

results = []
for dataset in ['triviaqa', 'squad_v2']:
    for method in ['greedy', 'selfcons', 'mi']:
        file = f'results/threshold_sweeps/{dataset}_{method}_full.json'
        data = json.load(open(file))
        
        for tr in data['threshold_results']:
            s = tr['summary']
            results.append({
                'Dataset': dataset,
                'Method': method,
                'Threshold': tr['threshold'],
                'Acc Change': s.get('avg_em_change', 0),
                'Clusters %': s.get('avg_clustering_reduction_pct', 0),
                'Changed %': s.get('predictions_changed_pct', 0)
            })

df = pd.DataFrame(results)
print(df.to_string(index=False))
```

---

## 🎯 Expected Results

### TriviaQA
| Method | Best Threshold | Acc Change | ECE Change | Notes |
|--------|---------------|------------|------------|-------|
| Greedy | 0.6-0.7 | +0.02 to +0.05 | Low/stable | NLI grading helps verbose answers |
| Self-Cons | 0.6-0.7 | +0.00 to +0.02 | -0.01 to -0.03 | Clustering + grading improve calibration |
| MI | 0.6-0.7 | +0.00 to +0.02 | -0.02 to -0.05 | Best ECE improvement |

### SQuAD v2
| Method | Best Threshold | Acc Change | ECE Change | Notes |
|--------|---------------|------------|------------|-------|
| Greedy | 0.7-0.8 | -0.02 to +0.00 | Low/stable | Higher threshold needed (extractive QA) |
| Self-Cons | 0.7-0.8 | -0.03 to -0.01 | -0.01 to -0.02 | Minimal improvement |
| MI | 0.7-0.8 | -0.03 to -0.01 | -0.01 to -0.03 | Modest ECE improvement |

**Note**: ECE (Expected Calibration Error) measures how well confidence scores match actual accuracy. Lower is better. Negative Δ ECE means better calibration.

**Key Insight**: SQuAD v2 requires higher thresholds (0.7-0.8) due to extractive nature. TriviaQA benefits more from NLI (0.6-0.7).

---

## 💡 Interpretation Guide

### Greedy Method
- **Only tests NLI grading** (no clustering since only 1 answer)
- Good result: Accuracy improves by +0.02 to +0.05
- Shows if NLI grading logic (`is_correct` method) works

### Self-Consistency Method
- **Tests both clustering and grading**
- Clustering: Groups similar answers before majority voting
- Good result: Accuracy stable (±0.01), fewer clusters (20-30% reduction)

### MI Method
- **Tests clustering, MI reduction, and ECE improvement**
- Clustering: Groups semantically equivalent answers across chains
- Good result: MI reduces 20-40%, ECE improves -0.02 to -0.05, accuracy stable

---

## 📁 Output Files

All results saved to `results/threshold_sweeps/`:

```
results/threshold_sweeps/
├── triviaqa_greedy_quick.json     # 20 questions
├── triviaqa_greedy_full.json      # 200 questions
├── triviaqa_selfcons_quick.json
├── triviaqa_selfcons_full.json
├── triviaqa_mi_quick.json
├── triviaqa_mi_full.json
├── squad_v2_greedy_quick.json
├── squad_v2_greedy_full.json
├── squad_v2_selfcons_quick.json
├── squad_v2_selfcons_full.json
├── squad_v2_mi_quick.json
└── squad_v2_mi_full.json
```

---

## 🚨 Troubleshooting

### If accuracy drops significantly (> -0.05)
- Threshold too low → Try 0.7-0.8
- Dataset incompatible (SQuAD v2) → Use higher thresholds or skip NLI

### If no clustering happens
- Threshold too high → Try 0.4-0.5
- Check model loaded correctly: Look for "✓ Model loaded on cuda"

### Memory issues
- Process one dataset at a time
- Use `--limit 50` for intermediate testing

---

**Next Steps**: After running quick tests, identify best thresholds and run full evaluation on those specific values.

