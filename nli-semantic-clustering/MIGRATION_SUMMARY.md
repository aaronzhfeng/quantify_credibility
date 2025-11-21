# NLI Module Migration Summary

**Date**: November 21, 2024  
**Purpose**: Extract NLI clustering code to standalone module for debugging and threshold adjustment

## ✅ Migration Complete

The NLI semantic clustering module has been successfully extracted to:
```
/root/quantify_credibility/nli-semantic-clustering/
```

## 📦 What Was Extracted

### Core Code
- ✅ `nli_clustering/core.py` - NLIClusteringCache + clustering functions (from `calibration.py` lines 2488-2694)
- ✅ `nli_clustering/utils.py` - Evaluation metrics, MI estimator, ECE computation
- ✅ `nli_clustering/__init__.py` - Package initialization

### Scripts
- ✅ `scripts/analyze_clustering_quality.py` - Compare F1 vs NLI clustering (from `analyze_mutual_entailment.py`)
- ✅ `scripts/recalculate_with_semantic_clustering.py` - Recalculate MI with NLI (from `recalculate_with_nli.py`)
- ✅ `scripts/threshold_sweep.py` - **NEW**: Systematic threshold debugging tool

### Data & Results
- ✅ `data/triviaqa/logs_mi/` - Sample of 20 question logs for debugging
- ✅ `data/squad_v2/logs_mi/` - Sample of 20 question logs for debugging
- ✅ `results/baseline/` - Non-NLI baseline results (greedy, selfcons, mi)
- ✅ `results/nli_experiments/` - Previous NLI experiment results for comparison

### Documentation
- ✅ `README.md` - Comprehensive debugging guide with troubleshooting
- ✅ `QUICKSTART.md` - 5-minute quick start guide
- ✅ `requirements.txt` - Dependencies
- ✅ `setup.py` - Package setup for pip installation

## 🎯 Key Features

### 1. Standalone & Modifiable
- **No dependencies** on main `llm-belief-mi-test` code (all utilities copied)
- Can modify threshold, clustering algorithm, or NLI model **directly**
- Changes won't affect main repo until you port them back

### 2. Debugging-Focused
- **Threshold sweep**: Test multiple thresholds systematically
- **Per-question analysis**: See exactly which questions degrade
- **Clustering metrics**: Understand how much clustering is happening
- **Comparison baseline**: Built-in non-NLI results for comparison

### 3. Ready for TriviaQA & SQuAD
- Sample data included for both datasets
- Baseline results copied for comparison
- Correctness-based MI support for TriviaQA

## 🚀 Quick Start

```bash
cd /root/quantify_credibility/nli-semantic-clustering

# Install dependencies
pip install -r requirements.txt

# Run threshold sweep on TriviaQA (2 min for 20 questions)
python scripts/threshold_sweep.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --output results/quick_test.json \
  --thresholds 0.4 0.5 0.6 0.7 \
  --correctness-based \
  --limit 20

# See QUICKSTART.md for full workflow
```

## 🔧 How to Debug

### Problem: NLI Clustering Hurts Accuracy

**Hypothesis 1: Threshold too low (over-clustering)**
```bash
# Test higher thresholds
python scripts/threshold_sweep.py ... --thresholds 0.6 0.65 0.7
```

**Hypothesis 2: NLI model mismatch**
```bash
# Try smaller model
python scripts/threshold_sweep.py ... --nli-model microsoft/deberta-base-mnli
```

**Hypothesis 3: Dataset incompatible with NLI**
```bash
# Analyze which questions fail
python -c "import json; ..." # See QUICKSTART.md
```

### Problem: ECE Still Degraded

**Solution**: Check MI-to-confidence mapping in `nli_clustering/utils.py`:
```python
def mi_to_confidence(mi_score: float, method: str = "inverse"):
    if method == "inverse":
        return 1.0 / (1.0 + mi_score)
    # Try: return math.exp(-mi_score) instead
```

## 📊 Current Status

### Known Issues (from main repo)
1. **TriviaQA**: Accuracy drops with NLI clustering at threshold 0.5
2. **SQuAD v2**: Accuracy drops 8-12%, incompatible with extractive QA
3. **ECE**: Gets worse instead of better

### Your Tasks
1. ✅ Module extracted and ready
2. ⏳ Run threshold sweep on TriviaQA
3. ⏳ Identify optimal threshold or root cause
4. ⏳ Test on SQuAD v2 (if needed)
5. ⏳ Port findings back to main repo

## 🔄 Porting Changes Back

Once you've found optimal settings:

1. **Update main repo threshold**:
   ```python
   # In llm-belief-mi-test/llm_belief_mi_test/calibration.py
   # Line ~2524
   def check_mutual_entailment(..., threshold: float = 0.6):  # Your new value
   ```

2. **Document findings**:
   ```bash
   # Create docs/XX_nli_threshold_fix.md in main repo
   ```

3. **Update COMMANDS_NLI.md**:
   ```markdown
   # Update recommended threshold in usage examples
   --nli-threshold 0.6  # Previously 0.5
   ```

## 📁 File Locations

### In This Module
- Core code: `nli_clustering/`
- Scripts: `scripts/`
- Data: `data/triviaqa/`, `data/squad_v2/`
- Results: `results/baseline/`, `results/nli_experiments/`

### In Main Repo (Reference Only)
- Original code: `llm-belief-mi-test/llm_belief_mi_test/calibration.py` (lines 2488-2694)
- Full logs: `llm-belief-mi-test/outputs/logs/triviaqa_mi_200/`
- Full logs: `llm-belief-mi-test/outputs/logs/squad_v2_mi_200/`
- Commands: `llm-belief-mi-test/COMMANDS_NLI.md`

## 💡 Tips

1. **Start small**: Use `--limit 20` for rapid iteration
2. **Visualize**: Plot accuracy vs threshold to find patterns
3. **Compare datasets**: TriviaQA and SQuAD may need different thresholds
4. **Manual inspection**: Look at specific failing cases to understand why
5. **Cache-friendly**: Threshold sweep caches NLI computations for speed

## 🎓 Understanding the Metrics

From `threshold_sweep.py` output:

- **Clusters**: % reduction in unique answers (20-30% is good)
- **Acc Orig**: Baseline accuracy without NLI
- **Acc NLI**: Accuracy with NLI clustering
- **Δ Acc**: Change in accuracy (≥ -0.01 is acceptable)
- **Changed**: % of predictions that changed

## ❓ Questions?

See `README.md` for:
- Detailed debugging workflow
- Troubleshooting common issues
- Advanced clustering modifications
- Alternative NLI models

See `QUICKSTART.md` for:
- 5-minute quick start
- Step-by-step debugging example
- Success criteria

## 🔗 Related Documentation

- Main repo NLI docs: `llm-belief-mi-test/COMMANDS_NLI.md`
- Main repo tech docs: `llm-belief-mi-test/docs/23_nli_clustering_implementation.md`
- Paper: arXiv:2406.02543v2 (mutual information for uncertainty quantification)

---

**Status**: ✅ Ready for debugging  
**Next Step**: Run `python scripts/threshold_sweep.py` on TriviaQA (see QUICKSTART.md)

