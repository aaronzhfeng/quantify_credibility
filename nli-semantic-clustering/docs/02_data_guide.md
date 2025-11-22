# Data Guide

This guide explains what data you need and how to get it.

## 🎯 What Data Do You Need?

For NLI threshold debugging, you need **log files from previous evaluations**, NOT the raw datasets.

### ✅ Already Included (Sample Data)
- `data/triviaqa/logs_mi/` - 20 sample questions for quick testing
- `data/squad_v2/logs_mi/` - 20 sample questions for quick testing
- `results/baseline/` - Non-NLI baseline results
- `results/nli_experiments/` - Previous NLI experiment results

### 📥 To Get Full Log Files (200 questions)

The log files are in the main `llm-belief-mi-test` repo:

```bash
# Copy all TriviaQA logs (200 questions)
cp ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200/question_*.json \
   data/triviaqa/logs_mi/

# Copy all SQuAD v2 logs (200 questions)
cp ../llm-belief-mi-test/outputs/logs/squad_v2_mi_200/question_*.json \
   data/squad_v2/logs_mi/
```

**These log files contain:**
- Question text
- Generated chains of answers
- Gold answers
- Original metrics (MI, confidence, accuracy)

This is everything the threshold sweep needs!

## 📚 Optional: Download Raw Datasets

If you want to inspect the original datasets (for reference only), use:

```bash
# Install datasets library (if not already installed)
pip install datasets

# Download datasets (they'll be cached by HuggingFace)
python scripts/download_datasets.py

# Options:
python scripts/download_datasets.py --datasets triviaqa squad_v2  # Both
python scripts/download_datasets.py --datasets triviaqa            # Just TriviaQA
python scripts/download_datasets.py --limit 200                    # First 200 only
```

**Note:** Raw datasets are NOT needed for threshold debugging! The log files already contain all the information you need.

## 🗂️ Data Structure

After copying log files, you'll have:

```
data/
├── triviaqa/
│   ├── logs_mi/               # Log files from MI method evaluation
│   │   ├── question_0.json
│   │   ├── question_1.json
│   │   └── ... (up to question_199.json)
│   └── validation.jsonl       # Optional: Raw dataset (if downloaded)
└── squad_v2/
    ├── logs_mi/               # Log files from MI method evaluation
    │   ├── question_0.json
    │   ├── question_1.json
    │   └── ... (up to question_199.json)
    └── validation.jsonl       # Optional: Raw dataset (if downloaded)
```

## 📋 Log File Format

Each `question_*.json` contains:

```json
{
  "question_id": 0,
  "question_text": "What is the capital of France?",
  "gold_answer": ["Paris", "paris"],
  "methods": {
    "mi_method": {
      "raw_outputs": [
        {"chain_id": 0, "step": 0, "text": "Paris"},
        {"chain_id": 0, "step": 1, "text": "Paris"},
        ...
      ],
      "final_metrics": {
        "predicted": "Paris",
        "exact_match": 1.0,
        "f1": 1.0,
        "mi_score": 0.123,
        "confidence": 0.891
      }
    }
  }
}
```

This is what the threshold sweep script reads!

## 🔍 Checking What You Have

```bash
# Count log files
echo "TriviaQA logs: $(ls data/triviaqa/logs_mi/*.json 2>/dev/null | wc -l)"
echo "SQuAD v2 logs: $(ls data/squad_v2/logs_mi/*.json 2>/dev/null | wc -l)"

# Inspect a sample log file
python -c "
import json
with open('data/triviaqa/logs_mi/question_0.json') as f:
    data = json.load(f)
    print(f\"Question: {data['question_text']}\")
    print(f\"Gold answers: {data.get('gold_answer', 'N/A')}\")
    methods = data.get('methods', {})
    print(f\"Methods available: {list(methods.keys())}\")
"
```

## ⚡ Quick Start

If you only have 20 sample questions and want to test the full dataset:

```bash
# Option 1: Copy from main repo
cp ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200/question_*.json \
   data/triviaqa/logs_mi/

# Option 2: Run evaluation in main repo first
cd ../llm-belief-mi-test
python -m llm_belief_mi_test.cli \
  --method mi --dataset triviaqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --output outputs/results/triviaqa/mi_200.csv

# Then copy the logs
cp outputs/logs/triviaqa_mi_200/question_*.json \
   ../nli-semantic-clustering/data/triviaqa/logs_mi/
```

## 💡 Summary

- **For threshold debugging**: Only need log files from `llm-belief-mi-test/outputs/logs/`
- **Currently have**: 20 sample questions (enough for quick testing)
- **To get full data**: Copy 200 log files from main repo
- **Raw datasets**: Optional, not needed for debugging

**Ready to start?** See `QUICKSTART.md` for threshold sweep instructions!

