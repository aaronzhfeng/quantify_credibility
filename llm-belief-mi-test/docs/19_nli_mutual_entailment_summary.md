# NLI Mutual Entailment Analysis - Summary

## Overview

This module performs **TWO types of analysis** using NLI-based mutual entailment:

1. **Clustering Analysis**: Compare F1 vs NLI for grouping multiple model answers
2. **Evaluation Analysis**: Compare current evaluation (exact match) vs NLI for checking if predicted answer matches gold answer

### Which Methods Get Analyzed?

```
Dataset: TriviaQA & SQuAD v2
├── Greedy baseline
│   └── Generates: 1 answer per question
│   └── Analysis: ❌ SKIP (nothing to cluster)
│
├── Self-Consistency baseline  
│   └── Generates: k=10 samples per question
│   └── Uses: F1 clustering for majority voting
│   └── Analysis: ✅ ANALYZE (compare F1 vs NLI clustering)
│
└── MI method
    └── Generates: k=10 chains × n=2 steps
    └── Uses: F1 clustering for semantic equivalence
    └── Analysis: ✅ ANALYZE (compare F1 vs NLI clustering)
```

**Total:** 4 files analyzed (2 datasets × 2 methods with clustering)

## Two Use Cases for Mutual Entailment

### Use Case 1: Clustering (Grouping Model Answers)

**Purpose:** Group the model's own multiple answers that mean the same thing

```
Model generates 10 samples:
["Octopussy", "Octopussy film", "All Time High", ...]

F1 clustering: 3 clusters (keeps "Octopussy" and "Octopussy film" separate)
NLI clustering: 2 clusters (merges them as semantically equivalent)

Impact: Better confidence estimation (votes properly aggregated)
```

### Use Case 2: Evaluation (Checking Against Gold Answer)

**Purpose:** Check if model's prediction semantically matches the gold answer

```
Model predicts: "Richard I of Normandy"
Gold answer: "Richard I"

Current evaluation: Exact match = 0.0 ❌ (different strings!)
NLI evaluation: Mutual entailment = 0.95 → Correct ✓

Impact: Fairer accuracy measurement (model gets credit for semantic correctness)
```

**This script analyzes BOTH use cases!**

## Answers to Your Questions

### 1. Time Estimate for All Analyses

**Total time: ~8 minutes** for all 6 result files (4 analyzed, 2 greedy skipped)

**What gets analyzed:**
- ✅ **Self-Consistency** (baseline): Generates k=10 samples → needs clustering
- ✅ **MI method**: Generates k=10 chains → needs clustering
- ❌ **Greedy** (baseline): Generates 1 answer → no clustering needed, skip

Breakdown:
- **TriviaQA MI**: ~3 minutes (200 questions × ~3.2 unique answers = ~1,000 NLI comparisons)
- **TriviaQA Self-Consistency**: ~3 minutes (also has multiple samples to cluster!)
- **SQuAD v2 MI**: ~1 minute (200 questions × ~1.8 unique answers = ~300 NLI comparisons)
- **SQuAD v2 Self-Consistency**: ~1 minute (also has multiple samples to cluster!)
- **Greedy methods**: Skipped (only 1 answer per question, no clustering needed)

**Why so fast?**
- Fewer unique answers than expected (~2-3 per question, not 10-20)
- DeBERTa-xlarge on GPU: ~100-150ms per bidirectional comparison
- Most questions converge to similar answers

### 2. NLI Result Format and Storage

**Output location**: `outputs/nli_analysis/`

Each run produces a JSON file saved in the same directory structure:
```
outputs/
├── nli_analysis/          # NEW: NLI analysis results
│   ├── triviaqa_mi_200_nli.json
│   ├── triviaqa_selfcons_200_nli.json
│   ├── squad_v2_mi_200_nli.json
│   └── squad_v2_selfcons_200_nli.json
├── results/               # Original evaluation results
│   ├── triviaqa/
│   │   ├── mi_200.csv
│   │   └── mi_200.json
│   └── squad_v2/
│       ├── mi_200.csv
│       └── mi_200.json
└── logs/                  # Detailed per-question logs
    ├── triviaqa_mi_200/
    └── squad_v2_mi_200/
```

**JSON format includes:**

1. **Summary statistics** (top-level metrics):
   - Number of questions analyzed
   - Average clustering agreement (F1 vs NLI)
   - How often NLI found more/fewer/same clusters
   - Time elapsed

2. **Per-question details** (for deep analysis):
   - All unique answers
   - F1-based clustering result
   - NLI-based clustering result
   - Pairwise F1 scores
   - Pairwise NLI entailment probabilities (forward & backward)
   - Which pairs are mutually entailing

**Example summary with NEW evaluation metrics:**
```json
{
  "summary": {
    "dataset": "triviaqa",
    "method": "mi",
    "n_questions_analyzed": 200,
    
    // Clustering metrics
    "avg_clustering_agreement": 0.78,
    "nli_fewer_clusters": 120,
    
    // NEW: Evaluation metrics
    "current_accuracy": 0.505,      // With exact match evaluation
    "nli_accuracy": 0.565,          // With NLI evaluation
    "accuracy_improvement": 0.060,  // +6.0% improvement!
    "wrong_to_right_count": 15,     // NLI recognized semantic matches
    "right_to_wrong_count": 3       // NLI corrected false positives
  }
}
```

**Example per-question entry:**
```json
{
  "question_id": 5,
  "question_text": "Rita Coolidge sang the title song for which Bond film?",
  "unique_answers": ["Octopussy", "All Time High", "All Night Long"],
  "f1_n_clusters": 3,
  "nli_n_clusters": 2,
  "pairwise_mutual": {
    "All Time High|||All Night Long": true
  },
  // NEW: Evaluation fields
  "predicted_answer": "Octopussy",
  "gold_answers": ["Octopussy", "Octopussy film"],
  "current_correct": true,    // Exact match found it correct
  "nli_correct": true,        // NLI also finds it correct
  "nli_eval_changed": false   // No change in this case
}
```

## Why Analyze Self-Consistency Baseline?

**Great question!** Yes, the baseline self-consistency method ALSO uses clustering, so it needs mutual entailment analysis too.

### How Self-Consistency Works

```python
# Generate k=10 samples
samples = ["Octopussy", "All Time High", "Octopussy", "All Night Long", ...]

# Cluster semantically equivalent answers (currently uses F1)
clusters = group_by_semantic_equivalence(samples)
# → {"Octopussy": 5 votes, "All Time High": 3 votes, "All Night Long": 2 votes}

# Majority voting
predicted_answer = max(clusters, key=lambda x: len(clusters[x]))
confidence = len(clusters[predicted_answer]) / k
```

**The problem:** If F1 fails to merge "Octopussy" and "Octopussy film", they get separate clusters!
- F1 clustering: 3 clusters → lower confidence
- NLI clustering: 2 clusters (merges paraphrases) → higher confidence

### Impact on Results

If we switch from F1 to NLI clustering:

**Self-Consistency might:**
- Get HIGHER confidence (fewer clusters due to better merging)
- Get DIFFERENT predictions (if vote distribution changes)
- Have BETTER calibration (confidence better reflects uncertainty)

**MI method might:**
- Calculate DIFFERENT MI values (fewer clusters → different entropy)
- Have BETTER calibration too

### Analysis vs Implementation

**This script does:** Post-hoc analysis to see IF switching matters

**Next step if it matters:** Implement NLI in production code

```python
# In calibration.py - FUTURE CHANGE if analysis shows improvement
def evaluate_mcq_self_consistency(...):
    # OLD:
    clusters = group_by_semantic_equivalence(samples, threshold=0.25)
    
    # NEW (if NLI proves better):
    clusters = group_by_mutual_entailment(samples, nli_checker, threshold=0.5)
```

## What NLI Model Does

**Technical answer:** Uses a **specialized NLI classifier** (DeBERTa-MNLI), NOT the general LLM.

- **Type**: Discriminative neural network trained on entailment tasks
- **Speed**: 100-150ms per comparison on GPU
- **Size**: ~1.5 GB (much smaller than Llama)
- **Output**: Probability scores for {entailment, neutral, contradiction}
- **Training**: Pre-trained on MNLI/SNLI datasets (hundreds of thousands of entailment pairs)

**Comparison to alternatives:**

| Method | Speed | Accuracy | Cost |
|--------|-------|----------|------|
| F1 algorithm | <1ms | Poor (lexical only) | Free |
| **NLI model (DeBERTa)** | ~100ms | Excellent | **Free*** |
| General LLM (Llama) | ~2-5s | Variable | High compute |

*One-time download

## How to Run

**Quick start (all datasets, 8 minutes):**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Ensure transformers is installed
pip install transformers scikit-learn

# Run all analyses
bash -c '
mkdir -p outputs/nli_analysis && \
python scripts/analyze_mutual_entailment.py --dataset triviaqa --method mi --limit 200 --output outputs/nli_analysis/triviaqa_mi_200_nli.json && \
python scripts/analyze_mutual_entailment.py --dataset triviaqa --method self-consistency --limit 200 --output outputs/nli_analysis/triviaqa_selfcons_200_nli.json && \
python scripts/analyze_mutual_entailment.py --dataset squad_v2 --method mi --limit 200 --output outputs/nli_analysis/squad_v2_mi_200_nli.json && \
python scripts/analyze_mutual_entailment.py --dataset squad_v2 --method self-consistency --limit 200 --output outputs/nli_analysis/squad_v2_selfcons_200_nli.json
'
```

See `../COMMANDS_NLI.md` for detailed commands.

## What to Expect

**Key findings to look for:**

1. **Clustering agreement**: Typically 0.6-0.8
   - High = F1 and NLI mostly agree
   - Low = NLI captures semantics F1 misses

2. **NLI fewer clusters**: Usually majority case
   - NLI merges paraphrases/synonyms F1 keeps separate
   - Example: "Richard I" + "Richard the First" → same cluster

3. **NLI more clusters**: Less common
   - F1 incorrectly merged semantically different answers
   - Example: Movie name + song title separated by NLI

4. **Edge cases**: 
   - Song titles vs movie names
   - Partial names vs full names
   - Synonyms and paraphrases

## Output Usage

After running analysis, use the JSON files to:

1. **Quantify improvement**: Compare clustering agreement scores
2. **Find examples**: Identify questions where NLI differs most from F1
3. **Validate approach**: Check if NLI clusters match human intuition
4. **Make decision**: Determine if NLI-based clustering is worth implementing in production

## Next Steps

If NLI analysis shows significant improvement:

1. Replace `group_by_semantic_equivalence()` in `calibration.py` with NLI version
2. Re-run evaluations with NLI clustering
3. Compare ECE/calibration metrics
4. Update paper/documentation with new results

If F1 and NLI largely agree:
- Current F1 method may be sufficient
- Document analysis as validation
- Consider NLI for edge cases only

