# NLI Semantic Clustering - Debugging & Threshold Adjustment

This module provides tools for debugging and adjusting NLI-based semantic clustering for open-ended question answering systems. It was extracted from `llm-belief-mi-test` to enable focused development and threshold tuning.

## 🎯 Purpose

The NLI clustering approach groups semantically equivalent answers together to reduce spurious uncertainty in MI-based confidence estimation. However, initial results showed:

- **Worse accuracy**: NLI clustering reduced accuracy by 8-12% on some datasets
- **Worse ECE**: Calibration metrics degraded instead of improving

**Root Cause Identified**: Using strict bidirectional equivalence for BOTH clustering AND grading.

**Solution Implemented**: Dual-mode NLI system:
- **Clustering**: Strict bidirectional (A ↔ B) - preserves uncertainty measurement
- **Grading**: Loose unidirectional (A → B) + substring - accepts verbose answers

This module provides tools to **test and validate** this fix through threshold adjustment and detailed analysis.

## 📁 Structure

```
nli-semantic-clustering/
├── nli_clustering/              # Core Python package
│   ├── __init__.py
│   ├── core.py                  # NLI model + clustering algorithms
│   └── utils.py                 # Evaluation metrics, MI estimation
├── scripts/                     # Analysis and debugging scripts
│   ├── analyze_clustering_quality.py    # Compare F1 vs NLI clustering
│   ├── recalculate_with_semantic_clustering.py  # Recalculate MI with NLI
│   └── threshold_sweep.py       # KEY: Systematic threshold debugging
├── data/                        # Sample data from main repo
│   ├── triviaqa/
│   │   └── logs_mi/            # Sample question logs for debugging
│   └── squad_v2/
│       └── logs_mi/            # Sample question logs
├── results/
│   ├── baseline/               # Non-NLI baseline results for comparison
│   └── nli_experiments/        # Previous NLI experiment results
├── docs/                       # Additional documentation
├── examples/                   # Usage examples
└── requirements.txt
```

## 🚀 Quick Start: Debugging TriviaQA

### 1. Install Dependencies

```bash
cd nli-semantic-clustering
pip install -r requirements.txt
```

### 2. Run Threshold Sweep (PRIMARY DEBUGGING TOOL)

This systematically tests different thresholds to find optimal values:

```bash
python scripts/threshold_sweep.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --output results/triviaqa_threshold_sweep.json \
  --thresholds 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 \
  --correctness-based \
  --dataset triviaqa \
  --limit 50  # Start with 50 questions for quick iteration
```

**Output:**
```
Threshold    Clusters   Acc Orig   Acc NLI    Δ Acc      Changed
------------------------------------------------------------------------------
0.30         45.2%      0.450      0.420      -0.030     23.5%
0.40         35.8%      0.450      0.435      -0.015     18.2%
0.50         28.3%      0.450      0.440      -0.010     12.4%
0.60         22.1%      0.450      0.448      -0.002      8.7%
0.70         15.5%      0.450      0.450       0.000      5.1%
```

### 3. Analyze Results

```bash
# Look at per-question details
python -c "import json; data = json.load(open('results/triviaqa_threshold_sweep.json')); 
print('Questions where NLI hurt accuracy:')
for q in data['per_question_results'][:10]:
    for t in q['threshold_results']:
        if t['threshold'] == 0.5 and t['em_change'] < 0:
            print(f\"Q: {q['question_text'][:60]}...\")
            print(f\"  Original: {t['predicted_original']}\")
            print(f\"  Clustered: {t['predicted_clustered']}\")
            print(f\"  Gold: {q['gold_answers']}\")
            print()
"
```

### 4. Adjust Threshold in Core Code

Based on sweep results, modify the default threshold in `nli_clustering/core.py`:

```python
# Before
def check_mutual_entailment(self, text_a, text_b, threshold: float = 0.5):

# After (if 0.6 works better)
def check_mutual_entailment(self, text_a, text_b, threshold: float = 0.6):
```

## 🔍 Understanding the Problem

### Why NLI Might Hurt Accuracy

1. **Over-clustering**: Threshold too low → groups dissimilar answers → wrong answer wins
2. **Under-clustering**: Threshold too high → no benefit, still loses accuracy due to edge cases
3. **Model mismatch**: DeBERTa-MNLI trained on sentence pairs, not short answers
4. **Extractive vs. generative**: SQuAD needs span matching, not semantic equivalence

### Key Diagnostic Questions

Run these to understand what's happening:

```bash
# 1. How much clustering is happening?
python scripts/analyze_clustering_quality.py \
  --dataset triviaqa --method mi --limit 50 \
  --output results/debug_clustering.json

# Check: avg_f1_clusters vs avg_nli_clusters
# Too similar? NLI not helping
# Too different? NLI over-clustering

# 2. Which questions degrade?
python -c "
import json
data = json.load(open('results/triviaqa_threshold_sweep.json'))
degraded = []
for q in data['per_question_results']:
    tr = [t for t in q['threshold_results'] if t['threshold'] == 0.5][0]
    if tr['em_change'] < 0:
        degraded.append({
            'question': q['question_text'][:60],
            'original': tr['predicted_original'],
            'clustered': tr['predicted_clustered'],
            'gold': q['gold_answers']
        })
print(f'Questions degraded: {len(degraded)}')
for d in degraded[:5]:
    print(f\"Q: {d['question']}\")
    print(f\"  Orig pred: {d['original']}\")
    print(f\"  NLI pred: {d['clustered']}\")
    print(f\"  Gold: {d['gold']}\")
    print()
"
```

## 🛠️ Debugging Workflow

### Phase 1: Understand Current Performance

```bash
# 1. Compare baseline vs NLI results
ls -lh results/baseline/
ls -lh results/nli_experiments/

# 2. Check accuracy drop
python -c "
import json
baseline = json.load(open('results/baseline/mi_200.json'))
nli = json.load(open('results/nli_experiments/triviaqa_mi_200.json'))
print(f\"Baseline EM: {baseline['exact_match']:.3f}\")
print(f\"NLI EM: {nli['summary']['nli_adapted_metrics']['exact_match']:.3f}\")
print(f\"Drop: {(nli['summary']['nli_adapted_metrics']['exact_match'] - baseline['exact_match']):.3f}\")
"
```

### Phase 2: Threshold Sweep

```bash
# Test thresholds 0.3 to 0.7 in steps of 0.05
python scripts/threshold_sweep.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --output results/sweep_triviaqa_full.json \
  --thresholds 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 \
  --correctness-based \
  --limit 200  # Full dataset
```

### Phase 3: Inspect Failure Cases

```bash
# Get detailed entailment scores for failing cases
python scripts/inspect_entailment_scores.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --threshold-sweep-result results/sweep_triviaqa_full.json \
  --output results/entailment_scores_analysis.json
```

### Phase 4: Try Alternative Models

If DeBERTa-MNLI doesn't work well, try:

```bash
# Smaller model (faster, may be less accurate)
python scripts/threshold_sweep.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --nli-model microsoft/deberta-base-mnli \
  --output results/sweep_base_model.json

# Semantic similarity model (different approach)
# Note: Requires modification to core.py to support sentence-transformers
```

## 📊 Expected Results

### Good Threshold (Example: 0.6)
- **Accuracy change**: -0.002 to +0.005 (minimal degradation or slight improvement)
- **Clustering reduction**: 20-30% (moderate clustering)
- **ECE improvement**: -0.02 to -0.05 (better calibration)
- **MI reduction**: -20% to -35% (reduced uncertainty)

### Bad Threshold (Example: 0.3)
- **Accuracy change**: -0.03 to -0.08 (significant degradation)
- **Clustering reduction**: 40-60% (over-clustering)
- **ECE degradation**: +0.02 to +0.05 (worse calibration)
- **MI reduction**: -40% to -60% (over-reduced uncertainty)

## 🔧 Advanced: Modifying the Clustering Algorithm

If threshold adjustment isn't enough, you can modify the clustering logic in `nli_clustering/core.py`:

### Option 1: Asymmetric Thresholds

```python
def check_mutual_entailment(self, text_a, text_b, 
                            threshold_fwd: float = 0.5,
                            threshold_bwd: float = 0.5):
    fwd = self.check_entailment(text_a, text_b)
    bwd = self.check_entailment(text_b, text_a)
    return fwd >= threshold_fwd and bwd >= threshold_bwd
```

### Option 2: Minimum Score Requirement

```python
def check_mutual_entailment(self, text_a, text_b, threshold: float = 0.5, min_score: float = 0.3):
    fwd = self.check_entailment(text_a, text_b)
    bwd = self.check_entailment(text_b, text_a)
    return (fwd >= threshold and bwd >= threshold) and min(fwd, bwd) >= min_score
```

### Option 3: Weighted Average

```python
def check_mutual_entailment(self, text_a, text_b, threshold: float = 0.5):
    fwd = self.check_entailment(text_a, text_b)
    bwd = self.check_entailment(text_b, text_a)
    avg = (fwd + bwd) / 2.0
    return avg >= threshold and min(fwd, bwd) >= threshold - 0.1
```

## 📝 Key Files for Modification

1. **`nli_clustering/core.py`**: Core NLI model and clustering logic
   - Modify `check_mutual_entailment()` for threshold changes
   - Modify `cluster_answers_by_nli()` for algorithm changes

2. **`scripts/threshold_sweep.py`**: Primary debugging tool
   - Add new metrics or diagnostics
   - Change threshold ranges

3. **`scripts/analyze_clustering_quality.py`**: Detailed clustering analysis
   - Compare F1 vs NLI clustering
   - Per-question diagnostics

## 🎓 Understanding the Metrics

- **Clustering Reduction**: % of unique answers reduced by clustering
  - Too high (>40%): Over-clustering, grouping dissimilar answers
  - Too low (<15%): Under-clustering, no benefit
  - Good range: 20-30%

- **Accuracy Change**: Difference in exact match after clustering
  - Negative: NLI grouping wrong answers together
  - Zero: Neutral (ideal if ECE improves)
  - Positive: Rare, but possible if NLI fixes string variations

- **ECE Change**: Difference in Expected Calibration Error
  - Negative: Better calibration (goal)
  - Positive: Worse calibration (problem)

- **MI Reduction**: How much MI decreases with clustering
  - Semantic clustering should reduce MI (less string variation)
  - But if accuracy drops, lower MI is meaningless

## 🚨 Troubleshooting

### Issue: Threshold sweep shows all thresholds hurt accuracy

**Possible causes:**
1. Dataset is extractive (SQuAD) → Use F1 clustering, not NLI
2. NLI model mismatch → Try semantic similarity models
3. Answer format incompatible → Answers too short or too long

**Solution:** Consider using F1-based clustering instead

### Issue: No improvement in ECE despite good accuracy

**Possible causes:**
1. MI recalculation not applied correctly
2. Confidence mapping function needs adjustment
3. Baseline already well-calibrated

**Solution:** Check `mi_to_confidence()` mapping in `utils.py`

### Issue: Out of memory during threshold sweep

**Solution:**
```bash
# Use smaller model
--nli-model microsoft/deberta-base-mnli

# Or process in batches
--limit 50  # Process 50 at a time
```

## 📚 Next Steps After Debugging

1. **Optimal threshold found**: Update `llm-belief-mi-test` with findings
2. **Algorithm modification needed**: Document changes and benchmark
3. **Model change needed**: Test alternative NLI/semantic models
4. **Abandon NLI for dataset**: Use F1 clustering, document why

## 🔗 Related Files in Main Repo

- `llm-belief-mi-test/llm_belief_mi_test/calibration.py` - Lines 2488-2694 (original NLI code)
- `llm-belief-mi-test/COMMANDS_NLI.md` - User-facing NLI documentation
- `llm-belief-mi-test/docs/23_nli_clustering_implementation.md` - Technical details

## 💡 Pro Tips

1. **Start small**: Use `--limit 20` for rapid iteration during debugging
2. **Visualize**: Plot accuracy vs threshold to find sweet spot
3. **Inspect failures**: Manual inspection of degraded cases is invaluable
4. **Compare datasets**: What works for TriviaQA may not work for SQuAD
5. **Cache results**: Threshold sweep results can be reused for analysis

## 📧 Support

This is a standalone debugging module. Modifications here don't affect the main `llm-belief-mi-test` repo until you port changes back.

