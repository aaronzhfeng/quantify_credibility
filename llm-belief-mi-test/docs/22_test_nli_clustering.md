# Testing NLI Clustering Implementation

## Quick Syntax Check

Verify the code has no syntax errors:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test
python -m py_compile llm_belief_mi_test/calibration.py
python -m py_compile llm_belief_mi_test/cli.py
echo "✓ Syntax check passed"
```

## Test Command (Small Dataset)

Run MI method with NLI clustering on a tiny dataset to verify it works:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Test with 2 examples (will take ~5-10 minutes including model download)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa \
  --split validation \
  --limit 2 \
  --k 3 \
  --n 2 \
  --temperature 0.9 \
  --max-tokens 20 \
  --use-nli-clustering \
  --nli-threshold 0.5 \
  --load-in-4bit \
  --output outputs/test/nli_clustering_test.csv
```

## What to Expect

**First run:**
1. Downloads DeBERTa-MNLI model (~1.6 GB, 2-5 minutes)
2. Shows: "Loading NLI model for semantic clustering..."
3. Shows: "✓ NLI clustering enabled (threshold=0.5)"
4. Progress bar: "Evaluating TriviaQA (MI + NLI)"
5. Completes in ~5-10 minutes (small test)

**Subsequent runs:**
- Model already cached, runs faster
- ~3-5 minutes for 2 examples

## Expected Output

```
Loading NLI model for semantic clustering: microsoft/deberta-v2-xlarge-mnli
✓ NLI clustering enabled for correctness-based MI (threshold=0.5)
Evaluating TriviaQA (MI + NLI): 100%|████████| 2/2 [02:34<00:00, 77.3s/it]

Results saved to: outputs/test/nli_clustering_test.csv
```

## Verify Results

Check that:
- ✅ CSV file created in `outputs/test/`
- ✅ JSON file created in `outputs/test/`
- ✅ ECE and confidence values are reasonable
- ✅ No errors during execution

## Compare with Baseline

Run without NLI clustering to compare:

```bash
# Baseline (no NLI clustering)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa \
  --split validation \
  --limit 2 \
  --k 3 \
  --n 2 \
  --temperature 0.9 \
  --max-tokens 20 \
  --load-in-4bit \
  --output outputs/test/baseline_test.csv

# Compare MI scores and confidence
echo "Baseline:"
cat outputs/test/baseline_test.json | jq '.avg_mi_bits, .avg_confidence'

echo "With NLI clustering:"
cat outputs/test/nli_clustering_test.json | jq '.avg_mi_bits, .avg_confidence'
```

**Expected difference:**
- **MI score**: Lower with NLI (semantic agreement detected)
- **Confidence**: Higher with NLI (more consistent semantically)

## Troubleshooting

### Model Download Fails
- Check internet connection
- Try smaller model: `--nli-model microsoft/deberta-base-mnli`

### Out of Memory
- Reduce batch size or use smaller model
- NLI model needs ~4GB VRAM

### Import Error
```bash
pip install transformers scikit-learn torch
```

