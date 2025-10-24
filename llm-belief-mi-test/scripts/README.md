# Scripts

This folder contains test scripts and utilities for the LLM Belief MI project.

## 🧪 Test Scripts

### `test_gpu_setup.py`
**Purpose**: Verify GPU, model loading, and basic functionality

**Usage**:
```bash
python scripts/test_gpu_setup.py
```

**What it tests**:
- ✅ GPU availability
- ✅ Model loading (Llama-3.1-8B with 4-bit quantization)
- ✅ Logprobs extraction
- ✅ Pseudo joint selection demo
- ✅ Performance estimates

**When to use**: First step after installation to verify everything works.

---

### `test_baselines.py`
**Purpose**: Test all three evaluation methods on a small sample

**Usage**:
```bash
python scripts/test_baselines.py
```

**What it tests**:
- ✅ Greedy baseline
- ✅ Self-consistency baseline
- ✅ MI method
- ✅ All methods on 3 examples

**When to use**: Before running full evaluations to verify all methods work.

---

### `test_temperature_diversity.py`
**Purpose**: Verify that temperature sampling produces diverse outputs

**Usage**:
```bash
python scripts/test_temperature_diversity.py
```

**What it tests**:
- ✅ Chain diversity with temperature=0.9
- ✅ Cache bug is fixed (chains are not identical)

**When to use**: If you suspect cache is breaking diversity.

---

### `test_model_load.py`
**Purpose**: Test basic model loading

**Usage**:
```bash
python scripts/test_model_load.py
```

**When to use**: If you have model loading issues.

---

### `test_model_1b.py`
**Purpose**: Test with smaller 1B model

**Usage**:
```bash
python scripts/test_model_1b.py
```

**When to use**: If you have limited GPU memory.

---

### `test_phi3.py`
**Purpose**: Test with Phi-3 model

**Usage**:
```bash
python scripts/test_phi3.py
```

**When to use**: Alternative to Llama if you don't have access.

---

## 🔧 Utility Scripts

### `compare_results.py`
**Purpose**: Compare results from multiple JSON files

**Usage**:
```bash
# Compare all results from 500-example runs
python scripts/compare_results.py outputs/results/*_500.json

# Compare specific datasets
python scripts/compare_results.py outputs/results/arc_challenge_*_500.json
```

**Output**:
- Comparison table sorted by ECE
- Analysis of accuracy range
- ECE improvement percentage
- Identifies best method

**When to use**: After running baselines to see which method has best calibration.

---

### `plot_results.py`
**Purpose**: Generate accuracy and ECE comparison plots

**Usage**:
```bash
# Plot all datasets
python scripts/plot_results.py --dataset all

# Plot specific dataset
python scripts/plot_results.py --dataset openbookqa

# Custom files
python scripts/plot_results.py --custom outputs/results/openbookqa_*_500.json
```

**Output**:
- Individual comparison plots per dataset
- Combined plot with all datasets
- Side-by-side accuracy and ECE bars

**When to use**: Visualize method comparisons for presentations/papers.

---

### `plot_calibration.py`
**Purpose**: Generate calibration curves (reliability diagrams)

**Usage**:
```bash
# Plot calibration for all datasets
python scripts/plot_calibration.py --dataset all

# Custom files
python scripts/plot_calibration.py --files outputs/results/openbookqa_*_500.csv
```

**Output**:
- Reliability diagrams (confidence vs actual accuracy)
- Confidence distribution histograms
- Visual assessment of calibration quality

**When to use**: Understand how well-calibrated each method is.

---

### `summarize_results.py`
**Purpose**: Print summary tables of all results

**Usage**:
```bash
# All datasets
python scripts/summarize_results.py

# Specific dataset
python scripts/summarize_results.py --dataset openbookqa

# Cross-dataset comparison
python scripts/summarize_results.py --cross-dataset
```

**Output**:
- Formatted tables with accuracy, ECE, confidence
- Best method highlighted
- Improvement percentages

**When to use**: Quick overview of all results.

---

### `visualize_all.sh`
**Purpose**: Run all visualization scripts at once

**Usage**:
```bash
bash scripts/visualize_all.sh
```

**Output**:
- All plots and summaries generated
- Saved to outputs/plots/

**When to use**: Generate complete visualization suite for all results.

---

### `RUN_BASELINE_COMPARISON_500.sh`
**Purpose**: Run all 9 baseline evaluations automatically

**Usage**:
```bash
bash scripts/RUN_BASELINE_COMPARISON_500.sh
```

**What it runs**:
- 3 datasets (ARC-Challenge, ARC-Easy, OpenBookQA)
- 3 methods each (Greedy, Self-Consistency, MI)
- 500 examples per evaluation
- Automatic comparison after each dataset

**Time**: ~12 hours total

**When to use**: For complete baseline comparison runs.

---

## 📋 Recommended Workflow

1. **Initial verification** (~5 min):
   ```bash
   python scripts/test_gpu_setup.py
   ```

2. **Test all methods** (~10 min):
   ```bash
   python scripts/test_baselines.py
   ```

3. **Run evaluations** (~12 hours):
   ```bash
   bash scripts/RUN_BASELINE_COMPARISON_500.sh
   ```

4. **Compare results** (~1 min):
   ```bash
   python scripts/compare_results.py outputs/results/*_500.json
   ```

---

## 🐛 Troubleshooting

**"ModuleNotFoundError"**
- Make sure you're in the project root: `cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test`
- Run from project root: `python scripts/test_gpu_setup.py`

**"GPU not available"**
- Check GPU allocation in your environment
- See [docs/ENVIRONMENT_ISSUES.md](../docs/ENVIRONMENT_ISSUES.md)

**"Model loading failed"**
- Check HuggingFace authentication
- See [docs/AUTHENTICATION_GUIDE.md](../docs/AUTHENTICATION_GUIDE.md)

---

💡 **Tip**: All scripts assume you're running from the project root directory.

