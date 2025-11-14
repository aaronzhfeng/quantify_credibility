# 🔄 Recalculate MI Metrics with NLI (No Re-Inference!)

## Quick Start

### What This Does

Recalculates MI, confidence, and ECE from **existing log files** by applying NLI semantic clustering to the chains - **without re-running expensive Llama inference!**

### Why Use This?

✅ **8× faster** - ~5 min instead of ~40 min per dataset  
✅ **No re-inference** - Reuses all existing Llama outputs  
✅ **Perfect comparison** - Same chains, only clustering differs  
✅ **Fast experimentation** - Try different thresholds instantly  

---

## Basic Usage

### TriviaQA (Correctness-Based MI)

```bash
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_mi_200 \
  --nli-threshold 0.5 \
  --correctness-based \
  --output outputs/nli_adaptation/triviaqa_mi_200_nli.json
```

### SQuAD v2 (Direct MI)

```bash
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/squad_v2_mi_200 \
  --nli-threshold 0.5 \
  --output outputs/nli_adaptation/squad_v2_mi_200_nli.json
```

---

## Quick Test (2 questions)

```bash
# First, make sure you have log files
ls outputs/logs/triviaqa_mi_200/question_*.json | head -2

# Recalculate just 2 questions
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_mi_200 \
  --limit 2 \
  --correctness-based \
  --output outputs/test/nli_recalc_test.json

# View results
cat outputs/test/nli_recalc_test.json | jq '.summary.improvements'
```

---

## Command Reference

### Required Arguments

- `--log-dir` - Directory with `question_*.json` files
- `--output` - Output JSON file path

### Optional Arguments

- `--nli-threshold 0.5` - Mutual entailment threshold
- `--nli-model microsoft/deberta-v2-xlarge-mnli` - NLI model
- `--correctness-based` - Use for TriviaQA (maps to correct/incorrect)
- `--limit N` - Process only first N questions

---

## Expected Results

### Console Output

```
================================================================================
Recalculating MI Metrics with NLI Clustering
================================================================================
Log directory     : outputs/logs/triviaqa_mi_200
Questions found   : 200
NLI threshold     : 0.5
Correctness-based : True
================================================================================

Loading NLI model...
✓ NLI model loaded

Processing questions: 100%|████████████| 200/200 [04:23<00:00,  1.3s/it]

================================================================================
RESULTS SUMMARY
================================================================================

Original (No NLI Clustering):
  Avg MI          : 0.6500 bits
  Avg Confidence  : 0.5800
  Exact Match     : 0.4500
  F1 Score        : 0.5200
  ECE             : 0.1200

NLI-Adapted (With Semantic Clustering):
  Avg MI          : 0.4200 bits (-0.2300, -35.4%)
  Avg Confidence  : 0.7200 (+0.1400, +24.1%)
  Exact Match     : 0.4600 (+0.0100)
  F1 Score        : 0.5300 (+0.0100)
  ECE             : 0.0700 (-0.0500, -41.7%)

Prediction Changes:
  Changed         : 12/200 (6.0%)

================================================================================
Results saved to: outputs/nli_adaptation/triviaqa_mi_200_nli.json
================================================================================
```

### JSON Output Structure

```json
{
  "summary": {
    "original_metrics": { "avg_mi_bits": 0.65, "ece": 0.12, ... },
    "nli_adapted_metrics": { "avg_mi_bits": 0.42, "ece": 0.07, ... },
    "improvements": {
      "mi_reduction": -0.23,
      "mi_reduction_pct": -35.4,
      "ece_improvement": -0.05,
      "ece_improvement_pct": -41.7
    }
  },
  "per_question": [ ... ]
}
```

---

## Ablation: Try Different Thresholds

```bash
# Test multiple thresholds
for thresh in 0.3 0.5 0.7; do
  python scripts/recalculate_with_nli.py \
    --log-dir outputs/logs/triviaqa_mi_200 \
    --nli-threshold $thresh \
    --correctness-based \
    --output outputs/nli_adaptation/triviaqa_thresh_${thresh}.json
done

# Compare ECE improvements
for thresh in 0.3 0.5 0.7; do
  echo "Threshold $thresh:"
  cat outputs/nli_adaptation/triviaqa_thresh_${thresh}.json | \
    jq '.summary.improvements | {ece_improvement, ece_improvement_pct}'
done
```

---

## Verify Consistency with Live NLI

Run both approaches and compare:

```bash
# 1. Run with live NLI clustering (during inference)
python -m llm_belief_mi_test.cli \
  --method mi --dataset triviaqa --limit 50 \
  --use-nli-clustering --correctness-based \
  --output outputs/results/triviaqa/mi_live_50.csv

# 2. Run without NLI, then recalculate
python -m llm_belief_mi_test.cli \
  --method mi --dataset triviaqa --limit 50 \
  --output outputs/results/triviaqa/mi_baseline_50.csv

python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_mi_50 \
  --correctness-based \
  --output outputs/nli_adaptation/triviaqa_adapted_50.json

# 3. Compare metrics - should be identical!
echo "Live NLI:"
cat outputs/results/triviaqa/mi_live_50.json | jq '{avg_mi_bits, avg_confidence, ece}'

echo "Post-hoc NLI:"
cat outputs/nli_adaptation/triviaqa_adapted_50.json | jq '.summary.nli_adapted_metrics | {avg_mi_bits, avg_confidence, ece}'
```

---

## Time Estimates

| Dataset | Questions | Original Inference | Recalculation | Speedup |
|---------|-----------|-------------------|---------------|---------|
| TriviaQA | 200 | ~40 min (4 GPUs) | ~5 min | 8× |
| SQuAD v2 | 200 | ~30 min (4 GPUs) | ~3 min | 10× |
| TriviaQA | 50 | ~10 min | ~1 min | 10× |

*First run adds 2-5 min for NLI model download*

---

## When to Use Each Approach

### Use `--use-nli-clustering` (Live)

- Running new evaluations from scratch
- Need final results for publication
- Want multi-GPU parallelization
- Don't have existing log files

### Use `recalculate_with_nli.py` (Post-hoc)

- Already have log files from previous runs
- Want to experiment with different thresholds
- Need quick ablation studies
- Comparing NLI vs non-NLI retroactively
- Limited compute time (8× faster!)

---

## Troubleshooting

### "No log files found"
- Check `--log-dir` path is correct
- Ensure you've run MI method first to generate logs
- Look for `outputs/logs/{dataset}_mi_{limit}/question_*.json`

### "Could not find method data"
- Script auto-detects method type
- Works with: `mi_method`, `triviaqa_correctness_mi`, `squad_v2_mi`
- Check your log files have `methods` field

### Model download timeout
- Pre-download: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/deberta-v2-xlarge-mnli')"`
- Or use smaller model: `--nli-model microsoft/deberta-base-mnli`

---

## Complete Example Workflow

```bash
# 1. Run baseline MI evaluation (no NLI)
python -m llm_belief_mi_test.cli \
  --method mi --dataset triviaqa --limit 200 \
  --output outputs/results/triviaqa/mi_baseline_200.csv

# 2. Recalculate with NLI (fast!)
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_mi_200 \
  --correctness-based \
  --output outputs/nli_adaptation/triviaqa_nli_200.json

# 3. View improvements
cat outputs/nli_adaptation/triviaqa_nli_200.json | \
  jq '.summary.improvements'

# 4. Experiment with thresholds
for t in 0.3 0.4 0.5 0.6 0.7; do
  python scripts/recalculate_with_nli.py \
    --log-dir outputs/logs/triviaqa_mi_200 \
    --nli-threshold $t --correctness-based \
    --output outputs/nli_adaptation/triviaqa_thresh_${t}.json
done

# 5. Plot ECE vs threshold
python scripts/plot_nli_ablation.py \
  --input "outputs/nli_adaptation/triviaqa_thresh_*.json" \
  --output outputs/plots/nli_threshold_ablation.png
```

---

## See Also

- **[COMMANDS_NLI.md](COMMANDS_NLI.md)** - Complete NLI documentation
- **[NLI_CLUSTERING_IMPLEMENTATION.md](NLI_CLUSTERING_IMPLEMENTATION.md)** - Implementation details
- **[TEST_NLI_CLUSTERING.md](TEST_NLI_CLUSTERING.md)** - Testing guide
