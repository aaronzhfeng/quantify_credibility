# 🔬 NLI-Enhanced Evaluation Commands

Two types of NLI enhancement for **sampling-based methods** (Self-Consistency & MI only).

---

## 📖 Quick Overview

| Enhancement Type | What It Does | Works For | Output Location |
|-----------------|--------------|-----------|-----------------|
| **Part 1: Accuracy Evaluation** | Compares F1 vs NLI clustering quality + NLI-based accuracy checking | ✅ Greedy (accuracy only)<br>✅ Self-Consistency<br>✅ MI | `outputs/nli_analysis/` |
| **Part 2: Clustering Recalculation** | Groups semantically similar answers, recalculates MI/confidence/ECE | ❌ Greedy (not supported)<br>✅ Self-Consistency<br>✅ MI | `outputs/nli_adapted/` |

**Note:** 
- **Part 1**: All methods supported. Greedy gets accuracy evaluation only (no clustering analysis).
- **Part 2**: Only sampling-based methods (Self-Consistency, MI) can use NLI clustering recalculation.
- **SQuAD v2 Issue**: Both NLI and semantic models reduce accuracy (-8% to -12%). Stick with F1 evaluation. See "Alternative Models" section for test results.

---

## 🎯 Part 1: NLI Clustering Analysis & Accuracy Evaluation

**What this does:** 
1. Compares F1-based vs NLI-based clustering quality (sampling methods only)
2. Uses NLI to check if predicted answers semantically match gold answers (all methods)

**Works for:** 
- **Clustering + Accuracy**: Self-Consistency, MI (generate multiple answers)
- **Accuracy only**: Greedy (single answer, no clustering)

### 📋 All 6 Commands (3 methods × 2 datasets)

```bash
# Create output directory
mkdir -p outputs/nli_analysis

# ===== TriviaQA (3 methods) =====

# 1. TriviaQA - Self-Consistency (NLI clustering + accuracy)
python scripts/analyze_mutual_entailment.py \
  --dataset triviaqa --method self-consistency --limit 200 \
  --output outputs/nli_analysis/triviaqa_selfcons_200_analysis.json

# 2. TriviaQA - MI (NLI clustering + accuracy)
python scripts/analyze_mutual_entailment.py \
  --dataset triviaqa --method mi --limit 200 \
  --output outputs/nli_analysis/triviaqa_mi_200_analysis.json

# 3. TriviaQA - Greedy (NLI accuracy only, no clustering)
python scripts/analyze_mutual_entailment.py \
  --dataset triviaqa --method greedy --limit 200 \
  --output outputs/nli_analysis/triviaqa_greedy_200_analysis.json

# ===== SQuAD v2 (3 methods) =====

# 4. SQuAD v2 - Self-Consistency (NLI clustering + accuracy)
python scripts/analyze_mutual_entailment.py \
  --dataset squad_v2 --method self-consistency --limit 200 \
  --output outputs/nli_analysis/squad_v2_selfcons_200_analysis.json

# 5. SQuAD v2 - MI (NLI clustering + accuracy)
python scripts/analyze_mutual_entailment.py \
  --dataset squad_v2 --method mi --limit 200 \
  --output outputs/nli_analysis/squad_v2_mi_200_analysis.json

# 6. SQuAD v2 - Greedy (NLI accuracy only, no clustering)
python scripts/analyze_mutual_entailment.py \
  --dataset squad_v2 --method greedy --limit 200 \
  --output outputs/nli_analysis/squad_v2_greedy_200_analysis.json
```

**One command to run all:**
```bash
cd quantify_credibility/llm-belief-mi-test && \
mkdir -p outputs/nli_analysis && \
echo "=== TriviaQA ===" && \
python scripts/analyze_mutual_entailment.py --dataset triviaqa --method self-consistency --limit 200 --output outputs/nli_analysis/triviaqa_selfcons_200_analysis.json && \
python scripts/analyze_mutual_entailment.py --dataset triviaqa --method mi --limit 200 --output outputs/nli_analysis/triviaqa_mi_200_analysis.json && \
python scripts/analyze_mutual_entailment.py --dataset triviaqa --method greedy --limit 200 --output outputs/nli_analysis/triviaqa_greedy_200_analysis.json && \
echo "=== SQuAD v2 ===" && \
python scripts/analyze_mutual_entailment.py --dataset squad_v2 --method self-consistency --limit 200 --output outputs/nli_analysis/squad_v2_selfcons_200_analysis.json && \
python scripts/analyze_mutual_entailment.py --dataset squad_v2 --method mi --limit 200 --output outputs/nli_analysis/squad_v2_mi_200_analysis.json && \
python scripts/analyze_mutual_entailment.py --dataset squad_v2 --method greedy --limit 200 --output outputs/nli_analysis/squad_v2_greedy_200_analysis.json && \
echo "=== All NLI analyses complete! ==="
```

### ⏱️ Time: ~10-12 minutes total
- Sampling methods (Self-Consistency, MI): ~2 minutes each (4 runs = ~8 min)
- Greedy method: ~1 minute each (2 runs = ~2 min)
- First run: +2-5 min for model download (one-time)

### 📊 Expected Results

**Clustering Quality (Self-Consistency & MI only):**
- F1 vs NLI agreement: 0.70-0.85
- NLI typically creates fewer, more semantically coherent clusters

**Accuracy Improvement (All methods):**
| Method | Original Accuracy | NLI Accuracy | Improvement |
|--------|------------------|--------------|-------------|
| Greedy | 0.38 | 0.42-0.44 | +4-6% |
| Self-Consistency | 0.48 | 0.53-0.55 | +5-7% |
| MI | 0.45 | 0.50-0.52 | +5-7% |

**Key insight:** NLI captures semantic equivalence missed by F1 token overlap.

**Note:** Greedy output will show "No questions with multiple answers - clustering analysis skipped."

---

## 🧪 Alternative Models for SQuAD v2 (Tested - Not Recommended)

**Problem:** DeBERTa-MNLI decreases SQuAD v2 accuracy by ~12% (entailment model incompatible with extractive QA).

**Hypothesis:** Semantic similarity models might work better for extractive QA.

**Result:** ❌ **Still decreases accuracy** (best: -8% at threshold 0.65)

### Test Commands (SQuAD v2 only)

```bash
cd /root/quantify_credibility/llm-belief-mi-test
mkdir -p outputs/nli_analysis/semantic_models

# Multi-QA MPNet with optimized threshold
python scripts/analyze_mutual_entailment.py \
  --dataset squad_v2 --method greedy --limit 200 \
  --model sentence-transformers/multi-qa-mpnet-base-cos-v1 \
  --nli-threshold 0.65 \
  --output outputs/nli_analysis/semantic_models/squad_v2_greedy_multiqa_0.65.json
```

### Results Summary

| Model | Threshold | Wrong→Right | Right→Wrong | Net | Accuracy Loss |
|-------|-----------|-------------|-------------|-----|---------------|
| DeBERTa-MNLI | 0.5 | 3 | 27 | -24 | **-12.0%** |
| Multi-QA MPNet | 0.85 | 4 | 31 | -27 | **-13.5%** |
| Multi-QA MPNet | 0.75 | 6 | 27 | -21 | **-10.5%** |
| Multi-QA MPNet | **0.65** | 11 | 27 | -16 | **-8.0%** ✓ Best |

### Conclusion

**All semantic/NLI models hurt SQuAD v2 accuracy.** The issue is fundamental:
- **F1 evaluation**: Rewards token overlap (appropriate for extractive spans)
- **Semantic models**: Require full semantic equivalence (too strict for partial spans)
- **27 questions** consistently fail across all models/thresholds

**Recommendation: Use standard F1 evaluation for SQuAD v2.** NLI/semantic enhancements are only beneficial for TriviaQA.

---

## 🔄 Part 2: NLI Clustering Recalculation (Sampling Methods Only)

**What this does:** Applies NLI clustering to group semantically similar answers, then recalculates MI, confidence, and ECE.

**Works for:** Self-Consistency, MI (requires multiple answers)  
**Doesn't work for:** Greedy (only 1 answer generated)

### 📋 All 4 Commands (2 methods × 2 datasets)

```bash
# Create output directory
mkdir -p outputs/nli_adapted

# ===== TriviaQA (2 methods) =====

# 1. TriviaQA - Self-Consistency (with NLI clustering)
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_selfcons_200 \
  --output outputs/nli_adapted/triviaqa_selfcons_200.json \
  --correctness-based \
  --nli-threshold 0.5

# 2. TriviaQA - MI (with NLI clustering - ADVANCED)
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_mi_200 \
  --output outputs/nli_adapted/triviaqa_mi_200.json \
  --correctness-based \
  --nli-threshold 0.5

# ===== SQuAD v2 (2 methods) =====

# 3. SQuAD v2 - Self-Consistency (with NLI clustering)
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/squad_v2_selfcons_200 \
  --output outputs/nli_adapted/squad_v2_selfcons_200.json \
  --nli-threshold 0.5

# 4. SQuAD v2 - MI (with NLI clustering - ADVANCED)
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/squad_v2_mi_200 \
  --output outputs/nli_adapted/squad_v2_mi_200.json \
  --nli-threshold 0.5
```

**One command to run all:**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
mkdir -p outputs/nli_adapted && \
echo "=== TriviaQA ===" && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/triviaqa_selfcons_200 --output outputs/nli_adapted/triviaqa_selfcons_200.json --correctness-based --nli-threshold 0.5 && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/triviaqa_mi_200 --output outputs/nli_adapted/triviaqa_mi_200.json --correctness-based --nli-threshold 0.5 && \
echo "=== SQuAD v2 ===" && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/squad_v2_selfcons_200 --output outputs/nli_adapted/squad_v2_selfcons_200.json --nli-threshold 0.5 && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/squad_v2_mi_200 --output outputs/nli_adapted/squad_v2_mi_200.json --nli-threshold 0.5 && \
echo "=== All NLI clustering complete! ==="
```

### ⏱️ Time: ~10 minutes total
- Each dataset: ~2-3 minutes
- First run: +2-5 min for model download (one-time)

### 📊 Expected Results

| Method | MI Reduction | Confidence Increase | ECE Improvement |
|--------|--------------|---------------------|-----------------|
| Self-Consistency | -20% to -30% | +15% to +25% | -30% to -40% |
| MI (Advanced) | -30% to -40% | +20% to +30% | -35% to -45% |

**Key insight:** Clustering reduces MI by recognizing semantic equivalence, improving calibration.

---

## 📤 Output Files

### Part 1: Accuracy Evaluation Output (in `outputs/nli_analysis/`)

```json
{
  "summary": {
    "current_accuracy": 0.45,
    "nli_accuracy": 0.52,
    "accuracy_improvement": 0.07,
    "wrong_to_right_count": 18,
    "right_to_wrong_count": 4
  }
}
```

### Part 2: Clustering Recalculation Output (in `outputs/nli_adapted/`)

```json
{
  "summary": {
    "original_metrics": {
      "avg_mi_bits": 0.65,
      "avg_confidence": 0.58,
      "exact_match": 0.45,
      "ece": 0.12
    },
    "nli_adapted_metrics": {
      "avg_mi_bits": 0.42,
      "avg_confidence": 0.72,
      "exact_match": 0.46,
      "ece": 0.07
    },
    "improvements": {
      "mi_reduction_pct": -35.4,
      "confidence_increase_pct": +24.1,
      "ece_improvement_pct": -41.7
    }
  }
}
```

---

## 🚀 Combo: Run Both Parts Together

For complete NLI analysis (clustering analysis + recalculation), run both parts:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
mkdir -p outputs/nli_analysis outputs/nli_adapted && \
\
echo "========================================" && \
echo "PART 1: NLI CLUSTERING ANALYSIS (4 runs)" && \
echo "========================================" && \
echo "=== TriviaQA ===" && \
python scripts/analyze_mutual_entailment.py --dataset triviaqa --method self-consistency --limit 200 --output outputs/nli_analysis/triviaqa_selfcons_200_analysis.json && \
python scripts/analyze_mutual_entailment.py --dataset triviaqa --method mi --limit 200 --output outputs/nli_analysis/triviaqa_mi_200_analysis.json && \
echo "=== SQuAD v2 ===" && \
python scripts/analyze_mutual_entailment.py --dataset squad_v2 --method self-consistency --limit 200 --output outputs/nli_analysis/squad_v2_selfcons_200_analysis.json && \
python scripts/analyze_mutual_entailment.py --dataset squad_v2 --method mi --limit 200 --output outputs/nli_analysis/squad_v2_mi_200_analysis.json && \
\
echo "" && \
echo "========================================" && \
echo "PART 2: NLI CLUSTERING RECALCULATION (4 runs)" && \
echo "========================================" && \
echo "=== TriviaQA ===" && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/triviaqa_selfcons_200 --output outputs/nli_adapted/triviaqa_selfcons_200.json --correctness-based --nli-threshold 0.5 && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/triviaqa_mi_200 --output outputs/nli_adapted/triviaqa_mi_200.json --correctness-based --nli-threshold 0.5 && \
echo "=== SQuAD v2 ===" && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/squad_v2_selfcons_200 --output outputs/nli_adapted/squad_v2_selfcons_200.json --nli-threshold 0.5 && \
python scripts/recalculate_with_nli.py --log-dir outputs/logs/squad_v2_mi_200 --output outputs/nli_adapted/squad_v2_mi_200.json --nli-threshold 0.5 && \
\
echo "" && \
echo "========================================" && \
echo "✅ ALL NLI ANALYSES COMPLETE!" && \
echo "========================================" && \
echo "Analysis results: outputs/nli_analysis/" && \
echo "Recalculation results: outputs/nli_adapted/"
```

### ⏱️ Total Time: ~18 minutes
- Part 1 (4 clustering analyses): ~8 minutes
- Part 2 (4 clustering recalculations): ~10 minutes
- First run: +2-5 min for model download (one-time)

---

## 🔧 Requirements

```bash
pip install transformers scikit-learn
```

**Model used:** `microsoft/deberta-v2-xlarge-mnli` (state-of-the-art NLI model, auto-downloaded)

---

## 🎛️ Advanced Options

### Experiment with Different NLI Thresholds

```bash
# Try different NLI thresholds (0.3 = loose, 0.7 = strict)
for threshold in 0.3 0.5 0.7; do
  python scripts/recalculate_with_nli.py \
    --log-dir outputs/logs/triviaqa_mi_200 \
    --nli-threshold $threshold \
    --correctness-based \
    --output outputs/nli_adapted/triviaqa_mi_200_nli_${threshold}.json
done

# Compare results
for threshold in 0.3 0.5 0.7; do
  echo "Threshold $threshold:"
  cat outputs/nli_adapted/triviaqa_mi_200_nli_${threshold}.json | \
    jq '.summary.improvements | {mi_reduction_pct, ece_improvement_pct}'
done
```

### Run on Custom Subsets

```bash
# Process only first 50 questions (for testing)
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_mi_200 \
  --output outputs/nli_adapted/triviaqa_mi_50_test.json \
  --correctness-based \
  --limit 50
```

### Use Different NLI Models

```bash
# Use smaller/faster model (300 MB instead of 1.6 GB)
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/squad_v2_mi_200 \
  --output outputs/nli_adapted/squad_v2_mi_200_base.json \
  --nli-model microsoft/deberta-base-mnli
```

---

## 📖 Understanding the Output

### Key Metrics

**MI Reduction %**: How much semantic clustering reduces mutual information
- Negative = lower MI = better semantic consistency
- Expected: -20% to -40%

**Confidence Increase %**: How much confidence improves with semantic clustering
- Positive = higher confidence = more certain predictions
- Expected: +15% to +30%

**ECE Improvement %**: How much calibration improves
- Negative = lower ECE = better calibration
- Expected: -30% to -45%

**Predictions Changed**: Number of final answers that changed
- Usually 5-10% of questions
- Changed because clustered marginal distribution shifted mode

---

## 🔗 Related Documentation

- **[docs/24_recalculate_nli_guide.md](docs/24_recalculate_nli_guide.md)** - Comprehensive guide
- **[docs/23_nli_clustering_implementation.md](docs/23_nli_clustering_implementation.md)** - Technical implementation
- **[docs/19_nli_mutual_entailment_summary.md](docs/19_nli_mutual_entailment_summary.md)** - Overview
- **[scripts/recalculate_with_nli.py](scripts/recalculate_with_nli.py)** - Recalculation script

---

## ❓ Troubleshooting

### Missing Log Files

**Issue**: `FileNotFoundError: outputs/logs/{dataset}_{method}_{limit}/`

**Solution**: Run the evaluation first using commands in [COMMANDS_OPENENDED.md](COMMANDS_OPENENDED.md)

```bash
# Example: Generate logs for TriviaQA MI
python -m llm_belief_mi_test.cli \
  --method mi --dataset triviaqa --limit 200 \
  --output outputs/results/triviaqa/mi_200.csv
```

### Model Download Timeout

**Issue**: First run downloads 1.6 GB model, may timeout on slow connections

**Solution**: Pre-download the model:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "microsoft/deberta-v2-xlarge-mnli"
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
```

### Low GPU Memory

**Issue**: OOM errors with DeBERTa-xlarge

**Solution**: Use smaller model:
```bash
python scripts/recalculate_with_nli.py \
  --log-dir outputs/logs/triviaqa_mi_200 \
  --output outputs/nli_adapted/triviaqa_mi_200.json \
  --nli-model microsoft/deberta-base-mnli \
  --correctness-based
```

---

## 📚 Appendix: Alternative Approaches

### A. Live NLI Clustering (During Inference)

If you want to run NLI clustering **during** inference (not post-hoc), use the `--use-nli-clustering` flag:

```bash
# New evaluation with live NLI clustering
python -m llm_belief_mi_test.cli \
  --method mi --dataset triviaqa --limit 200 \
  --use-nli-clustering --nli-threshold 0.5 \
  --output outputs/results/triviaqa/mi_nli_200.csv
```

**Pros**: Direct integration, no separate step  
**Cons**: 30-50% slower, can't experiment with thresholds afterward  
**When to use**: Final production runs, new evaluations

### B. Post-hoc Analysis (Clustering Comparison)

For detailed comparison of F1 vs NLI clustering quality:

```bash
# Detailed clustering analysis
python scripts/analyze_mutual_entailment.py \
  --dataset triviaqa --method mi --limit 200 \
  --output outputs/nli_analysis/triviaqa_mi_clustering_analysis.json
```

**Pros**: Detailed per-question breakdown, pairwise similarity scores  
**Cons**: Different output format, focused on analysis not recalculation  
**When to use**: Understanding clustering differences, research analysis

