# Quick Start Guide: NLI Threshold Debugging

This guide gets you started debugging NLI clustering in **5 minutes**.

## Problem Statement

Current NLI clustering results:
- ❌ **TriviaQA**: Accuracy drops by ~X% with threshold 0.5
- ❌ **SQuAD v2**: Accuracy drops by 8-12% 
- ❌ **ECE**: Gets worse instead of better

**Goal**: Find optimal thresholds or identify if NLI is fundamentally incompatible.

## Step 1: Install (30 seconds)

```bash
cd /root/quantify_credibility/nli-semantic-clustering
pip install -r requirements.txt
```

## Step 2: Quick Test on 20 Questions (2 minutes)

```bash
# Test TriviaQA with different thresholds
python scripts/threshold_sweep.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --output results/quick_test.json \
  --thresholds 0.4 0.5 0.6 0.7 \
  --correctness-based \
  --limit 20
```

**Look at the output table:**
```
Threshold    Clusters   Acc Orig   Acc NLI    Δ Acc      Changed
------------------------------------------------------------------------------
0.40         35.8%      0.450      0.435      -0.015     18.2%  ← BAD
0.50         28.3%      0.450      0.440      -0.010     12.4%  ← Still bad
0.60         22.1%      0.450      0.448      -0.002      8.7%  ← Better!
0.70         15.5%      0.450      0.450       0.000      5.1%  ← Best!
```

## Step 3: Inspect What Went Wrong

```bash
# Which questions degraded?
python -c "
import json
data = json.load(open('results/quick_test.json'))
print('\\n🔍 Questions where NLI (threshold=0.5) hurt accuracy:\\n')
for q in data['per_question_results']:
    tr = [t for t in q['threshold_results'] if t['threshold'] == 0.5][0]
    if tr['em_change'] < 0:
        print(f\"❌ Q: {q['question_text'][:50]}...\")
        print(f\"   Original prediction: '{tr['predicted_original']}'\")
        print(f\"   NLI prediction:      '{tr['predicted_clustered']}'\")
        print(f\"   Gold answer:         {q['gold_answers']}\")
        print()
"
```

## Step 4: Test Full Dataset (10 minutes)

```bash
# Once you've identified a promising threshold, test on full dataset
python scripts/threshold_sweep.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --output results/triviaqa_full_sweep.json \
  --thresholds 0.5 0.55 0.6 0.65 0.7 \
  --correctness-based \
  --limit 200  # Full dataset
```

## Step 5: Apply Fix

If you found optimal threshold (e.g., 0.6 instead of 0.5):

```python
# Edit: nli_clustering/core.py line ~2524
def check_mutual_entailment(
    self, 
    text_a: str, 
    text_b: str, 
    threshold: float = 0.6  # Changed from 0.5
) -> bool:
```

## Step 6: Verify Improvement

```bash
# Re-run with new threshold
python scripts/recalculate_with_semantic_clustering.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --output results/triviaqa_with_new_threshold.json \
  --nli-threshold 0.6 \
  --correctness-based

# Check results
python -c "
import json
data = json.load(open('results/triviaqa_with_new_threshold.json'))
s = data['summary']
orig = s['original_metrics']
nli = s['nli_adapted_metrics']
print(f\"\\n📊 Results with threshold 0.6:\\n\")
print(f\"Accuracy:   {orig['exact_match']:.3f} → {nli['exact_match']:.3f} ({nli['exact_match']-orig['exact_match']:+.3f})\")
print(f\"ECE:        {orig['ece']:.3f} → {nli['ece']:.3f} ({nli['ece']-orig['ece']:+.3f})\")
print(f\"MI:         {orig['avg_mi_bits']:.3f} → {nli['avg_mi_bits']:.3f} ({100*(nli['avg_mi_bits']-orig['avg_mi_bits'])/orig['avg_mi_bits']:.1f}%)\")
print(f\"Confidence: {orig['avg_confidence']:.3f} → {nli['avg_confidence']:.3f} ({100*(nli['avg_confidence']-orig['avg_confidence'])/orig['avg_confidence']:.1f}%)\")
"
```

## 🎯 Success Criteria

You've found a good threshold if:
- ✅ Accuracy change: ≥ -0.01 (minimal degradation)
- ✅ ECE improvement: -0.02 or better
- ✅ Clustering reduction: 20-30%

## 🚨 If No Threshold Works

If all thresholds hurt accuracy significantly:

1. **SQuAD v2**: Use F1 clustering instead (extractive QA is incompatible with NLI)
2. **TriviaQA**: Try alternative NLI models or semantic similarity
3. **Both**: Consider abandoning NLI for this dataset

## 📁 Key Files

- **Core code**: `nli_clustering/core.py`
- **Debugging script**: `scripts/threshold_sweep.py`
- **Analysis**: `scripts/analyze_clustering_quality.py`
- **Full docs**: `README.md`

## 💡 Pro Tip

Run threshold sweep on a small sample first (20 questions, 2 min) to quickly identify promising ranges, then do full evaluation (200 questions, 10 min).

---

**Next**: See `README.md` for advanced debugging techniques.

