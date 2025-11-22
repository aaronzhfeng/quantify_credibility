# Data Structure Documentation

This document describes the pre-computed evaluation data available for NLI threshold testing.

---

## 📊 Complete Data Inventory

### Directory Structure

```
data/
├── triviaqa/
│   ├── logs_greedy/       # 200 files (question_0.json to question_199.json)
│   ├── logs_selfcons/     # 200 files (question_0.json to question_199.json)
│   ├── logs_mi/           # 200 files (question_0.json to question_199.json)
│   └── validation.jsonl   # Full TriviaQA validation set (17,945 questions)
└── squad_v2/
    ├── logs_greedy/       # 200 files (question_0.json to question_199.json)
    ├── logs_selfcons/     # 200 files (question_0.json to question_199.json)
    ├── logs_mi/           # 200 files (question_0.json to question_199.json)
    └── validation.jsonl   # Full SQuAD v2 validation set (11,874 questions)
```

**Total**: 1,200 pre-computed question files (2 datasets × 3 methods × 200 questions)

---

## 🔍 File Format Examples

### Greedy Method
- **File size**: ~2-3 KB per question
- **Content**: Single greedy decode (temperature=0)
- **Structure**:

```json
{
  "question_id": 0,
  "question_text": "Who was the man behind The Chipmunks?",
  "gold_answer": "['David Seville']",
  "methods": {
    "greedy": {
      "method_name": "triviaqa_greedy",
      "raw_outputs": [
        {
          "text": "David Seville, a pseudonym for Ross Bagdasarian Sr.",
          "logprob": -4.998017212252307,
          "probability": 0.006751320171539775
        }
      ],
      "final_metrics": {
        "predicted": "David Seville, a pseudonym for Ross Bagdasarian Sr.",
        "exact_match": 0.0,
        "f1": 0.4444444444444445,
        "confidence": 0.006751320171539775
      }
    }
  }
}
```

### Self-Consistency Method
- **File size**: ~5-8 KB per question
- **Content**: 10 independent samples (temperature=0.9)
- **Structure**:

```json
{
  "question_id": 0,
  "question_text": "Who was the man behind The Chipmunks?",
  "gold_answer": "['David Seville']",
  "methods": {
    "self_consistency": {
      "method_name": "triviaqa_self_consistency",
      "raw_outputs": [
        {"text": "David Seville.", "logprob": -1.90, "probability": 0.149},
        {"text": "Ross Bagdasarian Sr.", "logprob": -1.43, "probability": 0.240},
        // ... 8 more samples
      ],
      "decision_process": {
        "num_samples": 10,
        "selected_answer": "David Seville.",
        "vote_counts": {"David Seville.": 5, "Ross Bagdasarian Sr.": 3, ...}
      },
      "final_metrics": {
        "predicted": "David Seville.",
        "exact_match": 1.0,
        "f1": 1.0,
        "confidence": 0.5,
        "agreement": 0.5
      }
    }
  }
}
```

### MI Method
- **File size**: ~15-20 KB per question
- **Content**: 10 chains × 2 steps = 20 total inferences (temperature=0.9)
- **Structure**:

```json
{
  "question_id": 0,
  "question_text": "Who was the man behind The Chipmunks?",
  "gold_answer": "['David Seville']",
  "methods": {
    "mi_method": {
      "method_name": "triviaqa_correctness_mi",
      "raw_inputs": [
        {"chain_id": 0, "step": 0, "prompt": [...], "temperature": 0.9},
        {"chain_id": 0, "step": 1, "prompt": [...], "temperature": 0.9},
        // ... 18 more inferences
      ],
      "raw_outputs": [
        {"chain_id": 0, "step": 0, "text": "Ross Bagdasarian Sr.", "logprob": -1.43, "probability": 0.240},
        {"chain_id": 0, "step": 1, "text": "David Seville...", "logprob": -11.32, "probability": 0.000012},
        // ... 18 more outputs
      ],
      "decision_process": {
        "num_chains": 10,
        "chain_length": 2,
        "total_inferences": 20,
        "mi_method": "listing",
        "mi_nats": 0.0,
        "mi_bits": 0.0,
        "selected_answer": "Ross Bagdasarian Sr.",
        "correctness_agreement": "100.00% (10/10 chains agree on correctness)"
      },
      "final_metrics": {
        "predicted": "Ross Bagdasarian Sr.",
        "exact_match": 0.0,
        "f1": 0.0,
        "confidence": 1.0,
        "mi_score": 0.0,
        "agreement": 1.0
      }
    }
  }
}
```

---

## 📈 Data Characteristics

### TriviaQA
- **Domain**: Open-domain factual questions
- **Answer style**: Short entity names or phrases
- **Example**: "Who was the man behind The Chipmunks?" → "David Seville"
- **Challenge**: Multiple valid phrasings (e.g., "David Seville" vs "Ross Bagdasarian Sr.")

### SQuAD v2
- **Domain**: Extractive reading comprehension
- **Answer style**: Text spans from passages
- **Example**: Question references a passage, answer is exact substring
- **Challenge**: Answers are often longer phrases or sentences

---

## 🎯 What's Pre-computed

All files contain **exact-match (F1) based evaluation** results:
- ✅ Model outputs (text, logprobs, probabilities)
- ✅ F1 scores and exact match metrics
- ✅ Confidence scores
- ✅ Agreement fractions (for multi-sample methods)
- ✅ MI scores (for MI method)
- ❌ **NO NLI clustering applied yet**
- ❌ **NO NLI grading applied yet**

This is the **baseline data** that the threshold sweep scripts process to apply NLI clustering/grading.

---

## 🔬 How Scripts Use This Data

### `threshold_sweep.py`
1. Reads pre-computed answers from JSON files
2. Applies NLI clustering with specified threshold
3. Recalculates final answer (mode of clustered distribution)
4. Applies NLI grading (if `--use-nli-grading` flag set)
5. Compares original vs NLI-enhanced metrics

### `recalculate_with_semantic_clustering.py`
1. Reads pre-computed answers from JSON files
2. Applies NLI clustering with single threshold
3. Recalculates MI, confidence, ECE
4. Saves detailed per-question results

---

## 📏 Data Statistics

### Greedy
- **Answers per question**: 1
- **Total inferences**: 200 per dataset
- **Avg file size**: 2.5 KB

### Self-Consistency
- **Answers per question**: 10
- **Total inferences**: 2,000 per dataset (10 × 200)
- **Avg file size**: 6.5 KB

### MI
- **Answers per question**: 20 (10 chains × 2 steps)
- **Total inferences**: 4,000 per dataset (20 × 200)
- **Avg file size**: 18 KB

### Total Storage
- **TriviaQA**: ~3.5 MB (200 files × 3 methods)
- **SQuAD v2**: ~3.5 MB (200 files × 3 methods)
- **Combined**: ~7 MB

---

## 🔄 Data Source

All files were copied from:
```bash
/root/quantify_credibility/llm-belief-mi-test/outputs/logs/
├── triviaqa_greedy_200/
├── triviaqa_selfcons_200/
├── triviaqa_mi_200/
├── squad_v2_greedy_200/
├── squad_v2_selfcons_200/
└── squad_v2_mi_200/
```

Generated using the main evaluation pipeline in `llm-belief-mi-test` with:
- Model: Llama 3 8B Instruct
- Temperature: 0.0 (greedy), 0.9 (self-consistency, MI)
- Max tokens: 50
- Evaluation metric: F1 score (SQuAD-style)

---

## 💡 Usage Notes

1. **No need to re-run inference**: All model outputs are pre-computed
2. **Fast iteration**: Testing new thresholds takes seconds, not hours
3. **Reproducible**: Same model outputs, only clustering/grading logic changes
4. **Expandable**: Can add more questions by copying from original repo

---

**See also**:
- `COMMANDS_THRESHOLD_SWEEP.md` - Commands to test different thresholds
- `QUICKSTART.md` - Quick 5-minute test guide
- `README.md` - Full module documentation

