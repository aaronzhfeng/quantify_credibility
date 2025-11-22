# Quick Start Guide: NLI Semantic Clustering Evaluation

Get started evaluating NLI clustering quality in **5 minutes**.

---

## 💡 How This Works

**No LLM inference needed!** This module analyzes **pre-computed model outputs** stored locally:

```
data/
├── triviaqa/
│   ├── logs_greedy/       # 200 questions (1 answer each)
│   ├── logs_selfcons/     # 200 questions (10 samples each)
│   └── logs_mi/           # 200 questions (20 inferences: 10 chains × 2 steps)
└── squad_v2/
    ├── logs_greedy/       # 200 questions (1 answer each)
    ├── logs_selfcons/     # 200 questions (10 samples each)
    └── logs_mi/           # 200 questions (20 inferences: 10 chains × 2 steps)
```

The scripts apply **different NLI thresholds** to these existing outputs to find optimal clustering settings. The `--limit 20` flag processes only the first 20 questions for quick testing.

**For comprehensive testing across all 3 methods**, see `COMMANDS_THRESHOLD_SWEEP.md`.

---

## 📖 Problem & Solution

**Problem**: NLI clustering with strict bidirectional equivalence causes:
- ❌ Accuracy drops by 8-12% (SQuAD v2) or variable (TriviaQA)
- ❌ ECE spikes instead of improving
- ❌ Valid but verbose answers marked as incorrect

**Root Cause**: Using same strict logic for clustering (internal consistency) AND grading (correctness check).

**Solution**: Asymmetric evaluation
- **Clustering**: Keep strict bidirectional entailment (A⇔B)
- **Grading**: Use loose unidirectional entailment (A⇒B) with `--use-nli-grading`

See `docs/03_nli_clustering_accuracy_ece_diagnosis.md` for full diagnosis.

---

## 🔧 Setup (30 seconds)

```bash
cd /root/quantify_credibility/nli-semantic-clustering
pip install -r requirements.txt
```

**Model**: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (auto-downloaded on first run, ~1.7 GB)
- DeBERTa v3 large model fine-tuned on multiple NLI datasets (MNLI, FEVER, ANLI, LingNLI, WANLI)
- State-of-the-art zero-shot classification and semantic entailment

---

## 🎯 TriviaQA Evaluation

### Quick Test (20 questions, 2 minutes)

```bash
# Create output directory
mkdir -p results

# Test with NLI grading (recommended)
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/triviaqa_quick_nli.json \
  --thresholds 0.4 0.5 0.6 0.7 \
  --correctness-based \
  --use-nli-grading \
  --limit 20
```

### Full Evaluation (200 questions, 10 minutes)

```bash
# Full threshold sweep with NLI grading
python scripts/threshold_sweep.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/triviaqa_full_sweep.json \
  --thresholds 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9\
  --correctness-based \
  --use-nli-grading

# Apply best threshold and recalculate metrics
python scripts/recalculate_with_semantic_clustering.py \
  --log-dir data/triviaqa/logs_mi \
  --output results/triviaqa_nli_final.json \
  --nli-threshold 0.6 \
  --correctness-based \
  --use-nli-grading
```

### 📊 Expected Results

| Threshold | Clusters | Acc Change | ECE Change | Status |
|-----------|----------|------------|------------|--------|
| 0.40 | 35-40% | -0.015 | +0.02 | Too loose |
| 0.50 | 28-32% | -0.010 | +0.01 | Better |
| 0.60 | 22-26% | -0.002 | -0.01 | Good ✓ |
| 0.70 | 15-20% | 0.000 | -0.02 | Best ✓ |

---

## 🧪 SQuAD v2 Evaluation

### Quick Test (20 questions, 2 minutes)

```bash
# Test with NLI grading
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_mi \
  --output results/squad_v2_quick_nli.json \
  --thresholds 0.5 0.6 0.7 0.8 \
  --use-nli-grading \
  --limit 20
```

### Full Evaluation (200 questions, 10 minutes)

```bash
# Full threshold sweep with NLI grading
python scripts/threshold_sweep.py \
  --log-dir data/squad_v2/logs_mi \
  --output results/squad_v2_full_sweep.json \
  --thresholds 0.5 0.6 0.7 0.8 \
  --use-nli-grading

# Apply best threshold and recalculate metrics
python scripts/recalculate_with_semantic_clustering.py \
  --log-dir data/squad_v2/logs_mi \
  --output results/squad_v2_nli_final.json \
  --nli-threshold 0.7 \
  --use-nli-grading
```

### 📊 Expected Results

| Threshold | Clusters | Acc Change | ECE Change | Status |
|-----------|----------|------------|------------|--------|
| 0.50 | 30-35% | -0.08 | +0.03 | Too loose |
| 0.60 | 24-28% | -0.05 | +0.01 | Better |
| 0.70 | 18-22% | -0.02 | -0.01 | Good ✓ |
| 0.80 | 12-16% | 0.000 | -0.02 | Best ✓ |

**Note**: SQuAD v2 requires higher thresholds due to extractive QA nature.

---

## 🔍 Inspect Results

```bash
# View summary
python -c "
import json
data = json.load(open('results/triviaqa_full_sweep.json'))
for result in data['threshold_results']:
    t = result['threshold']
    acc_change = result['summary']['avg_em_change']
    clusters_pct = result['summary']['avg_clustering_reduction_pct']
    changed_pct = result['summary']['predictions_changed_pct']
    print(f\"Threshold {t:.2f}: Clusters {clusters_pct:.1f}%, Acc Δ {acc_change:+.3f}, Changed {changed_pct:.1f}%\")
"

# Check questions with accuracy drop
python -c "
import json
data = json.load(open('results/triviaqa_full_sweep.json'))
threshold_data = [t for t in data['threshold_results'] if t['threshold'] == 0.5][0]
print('\\n🔍 Questions where accuracy dropped at threshold 0.5:\\n')
for q in threshold_data['per_question_results']:
    if q['em_change'] < 0:
        print(f\"Q: {q['question_text'][:60]}...\")
        print(f\"  Prediction: '{q['predicted_clustered'][:40]}...'\")
        print(f\"  Gold: {q['gold_answers'][0]}\")
        print()
"
```

---

## 🎯 Success Criteria

A good threshold should achieve:
- ✅ Accuracy change: ≥ -0.01 (minimal drop)
- ✅ ECE improvement: -0.02 or better
- ✅ Clustering reduction: 20-30%
- ✅ Predictions changed: < 15%

---

## 📁 Key Files

- **Core NLI logic**: `nli_clustering/core.py`
- **Threshold sweep**: `scripts/threshold_sweep.py`
- **Recalculation**: `scripts/recalculate_with_semantic_clustering.py`
- **Full guide**: `README.md`
- **Diagnosis**: `docs/03_nli_clustering_accuracy_ece_diagnosis.md`

---

## 💡 Pro Tips

1. **Start small**: Test on 20 questions first to identify promising thresholds
2. **Use NLI grading**: Always include `--use-nli-grading` flag for accuracy evaluation
3. **Higher for SQuAD v2**: Extractive QA needs stricter thresholds (0.7-0.8)
4. **Lower for TriviaQA**: Open-domain QA works well with 0.6-0.7

---

**Next**: See `README.md` for detailed debugging and `docs/04_using_nli_grading_mode.md` for implementation details.

