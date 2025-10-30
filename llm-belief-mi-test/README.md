# LLM Belief MI - Benchmark Evaluation

Evaluation of the Mutual Information (MI) and iterative prompting method from "To Believe or Not to Believe Your LLM" on multiple-choice benchmarks (ARC-Challenge, ARC-Easy, OpenBookQA).

## 📋 Current Status

- ✅ **Implementation complete and ready to use!**
- ✅ Proper pseudo joint selection implemented (paper's method)
- ✅ Logprobs extraction for probability-weighted selection
- ✅ MI estimation for uncertainty quantification
- ✅ ECE computation for calibration evaluation
- ✅ Full CLI with all benchmarks (ARC/OpenBookQA)
- ✅ **Baseline methods** for comparison (greedy, self-consistency)

**See [docs/IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md) for usage instructions!** ⭐
**See [docs/BASELINE_COMPARISON_GUIDE.md](docs/BASELINE_COMPARISON_GUIDE.md) for baseline comparisons!** 📊

## Goal

Measure how the MI-based uncertainty quantification method affects:
1. **Accuracy**: Task performance on MCQ benchmarks
2. **ECE (Expected Calibration Error)**: Quality of confidence calibration

## Key Insight

The paper's method is designed for **uncertainty quantification**, not accuracy improvement. We expect:
- ✅ Similar accuracy to baselines
- ✅ **Better calibration** (lower ECE) - main contribution
- ✅ Better abstention policies (when allowed to skip)

## 📚 Documentation

All detailed guides are in the [`docs/`](docs/) folder:

- **[docs/IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md)**: ⭐ **START HERE** - Implementation complete, ready to use!
- **[docs/BASELINE_COMPARISON_GUIDE.md](docs/BASELINE_COMPARISON_GUIDE.md)**: 📊 **Baseline comparison guide**
- **[docs/QUICK_START_BASELINES.md](docs/QUICK_START_BASELINES.md)**: Quick reference for running baselines
- **[docs/COMMANDS_500_EXAMPLES.txt](docs/COMMANDS_500_EXAMPLES.txt)**: Copy-paste commands for 500-example runs
- **[docs/AUTHENTICATION_GUIDE.md](docs/AUTHENTICATION_GUIDE.md)**: HuggingFace authentication for Llama models

See [`docs/`](docs/) folder for complete documentation index.

## Quick Start

**This README contains everything you need to run the evaluation.** Just follow these steps:

### 1. Installation

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test
pip install -r requirements.txt
```

### 2. Authentication (Required for Llama Models)

```bash
# Export your HuggingFace token
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Or login via CLI (persistent)
huggingface-cli login
```

**Get your token**: https://huggingface.co/settings/tokens  
**Request Llama access**: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

### 3. Verify GPU Setup

```bash
# Test GPU, model loading, and pseudo joint selection
python scripts/test_gpu_setup.py
```

This will:
- ✅ Check GPU availability
- ✅ Load Llama-3.1-8B with 4-bit quantization
- ✅ Test logprobs extraction
- ✅ Demo pseudo joint selection
- ✅ Show performance estimates

### 4. Understanding CLI Arguments

Before running evaluations, let's understand all the arguments for `python -m llm_belief_mi_test.cli`:

#### **Method Selection**
- `--method {mi,greedy,self-consistency,semantic-entropy,self-verification}` 
  - Choose evaluation method (default: `mi`)
  - `mi`: MI-based uncertainty quantification (paper's main method)
  - `greedy`: Single greedy decode baseline (temperature=0)
  - `self-consistency`: k samples with majority voting
  - `semantic-entropy`: k samples with F1 clustering + entropy confidence
  - `self-verification`: k samples + verification query

#### **Dataset Selection**
- `--dataset {arc-challenge,arc-easy,openbookqa}` **[REQUIRED]**
  - Benchmark dataset to evaluate
- `--split {test,validation}` (default: `test`)
  - Dataset split to use
- `--limit N`
  - Limit to first N examples (useful for testing)

#### **Model Configuration**
- `--model MODEL_NAME` (default: `meta-llama/Llama-3.1-8B-Instruct`)
  - HuggingFace model name or local path
- `--load-in-4bit` **[RECOMMENDED]**
  - Use 4-bit quantization (saves memory, faster)
- `--load-in-8bit`
  - Use 8-bit quantization (alternative to 4-bit)

#### **Generation Parameters**
- `--temperature TEMP` (default: 0.5)
  - Sampling temperature
  - **Use 0.9 for MI/sampling methods** (paper's value)
  - Use 0.0 for greedy baseline
- `--max-tokens N` (default: 64)
  - Maximum tokens per generation
  - **Use 10 with `--answer-format strict`** for efficiency
  - Use 30-100 for default format
- `--answer-format {default,strict,codeblock}` (default: `default`)
  - **`strict`** ✅ **[RECOMMENDED]**: Model outputs only "A", "B", "C", or "D"
    - Clean, fast (1-5 tokens), no parsing errors
    - Use with `--max-tokens 10`
  - `default`: Verbose natural language responses
    - Use with `--max-tokens 30-100`
  - `codeblock`: Answer in triple backticks like `\`\`\`A\`\`\``

#### **MI-Specific Parameters** (only for `--method mi`)
- `--k N` (default: 10)
  - Number of independent chains per question (paper's value)
- `--n N` (default: 2)
  - Chain length / pseudo joint dimension (paper's value)
- `--mi-method {plugin,listing}` (default: `listing`)
  - MI estimator to use
- `--confidence-method {inverse,exp,normalized}` (default: `inverse`)
  - How to convert MI to confidence score

#### **Sampling Parameters** (for self-consistency, semantic-entropy, self-verification)
- `--k N` (default: 10)
  - Number of samples to generate per question

#### **Output**
- `--output PATH` **[REQUIRED]**
  - Output CSV file path
  - Also creates: `{output}.json` (metrics) and `logs/{run_name}/question_*.json` (detailed traces)
  - Example: `--output outputs/results/test.csv`
    - Creates: `outputs/results/test.csv`, `outputs/results/test.json`
    - Creates: `outputs/logs/test/question_0.json`, `question_1.json`, ...

#### **Caching**
- `--cache-path PATH` (default: `.cache/llm_cache.sqlite`)
  - SQLite cache file location
- `--cache-mode {readwrite,read,write,off}` (default: `readwrite`)
  - Cache mode (auto-disabled for temperature > 0)

#### **Other**
- `--verbose`
  - Enable verbose logging

#### **Example Commands**

**Quick test with strict mode (RECOMMENDED):**
```bash
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_mi.csv
```

**Greedy baseline:**
```bash
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 100 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_greedy.csv
```

**Self-consistency with 20 samples:**
```bash
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge \
  --k 20 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_selfcons.csv
```

### 5. Run Quick Test (5 examples)

```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_quick.csv
```

**Note**: Using **`--answer-format strict`** for clean single-letter responses and **temperature=0.9** (from paper) for proper diversity in chains. Detailed logs automatically saved to `outputs/logs/test_quick/`.

### 6. Run Baseline Comparisons (RECOMMENDED!)

To properly evaluate the MI method, compare it against baselines:

```bash
# Quick 5-example test with all methods (~5 min total)

# Greedy baseline (fastest, ~30 sec)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 5 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/baseline_greedy_5.csv

# Self-consistency baseline (~2 min)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-easy --limit 5 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/baseline_selfcons_5.csv

# MI method (~3 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/mi_method_5.csv

# Compare results
python scripts/compare_results.py outputs/results/*_5.json
```

**Expected**: Similar accuracy, **lower ECE** for MI method (key result!)

See **[docs/BASELINE_COMPARISON_GUIDE.md](docs/BASELINE_COMPARISON_GUIDE.md)** for detailed comparison guide.

### 7. Run Full Baseline Comparison (RECOMMENDED - 500 examples per dataset)

For fair comparison, run all 3 methods on 500 examples from each dataset:

**ARC-Challenge (500 examples)**
```bash
# Greedy baseline (~10 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge --limit 500 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge_greedy_500.csv

# Self-consistency baseline (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge_selfcons_500.csv

# MI method (~2.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge_mi_500.csv
```

**ARC-Easy (500 examples)**
```bash
# Greedy baseline (~10 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 500 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy_greedy_500.csv

# Self-consistency baseline (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy_selfcons_500.csv

# MI method (~2.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy_mi_500.csv
```

**OpenBookQA (500 examples - full dataset)**
```bash
# Greedy baseline (~10 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_greedy_500.csv

# Self-consistency baseline (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_selfcons_500.csv

# MI method (~2.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_mi_500.csv
```

**Compare results across all datasets:**
```bash
# Compare each dataset
python scripts/compare_results.py outputs/results/arc_challenge_*_500.json
python scripts/compare_results.py outputs/results/arc_easy_*_500.json
python scripts/compare_results.py outputs/results/openbookqa_*_500.json
```

**Or run all at once:**
```bash
bash scripts/RUN_BASELINE_COMPARISON_500.sh
```

**Total time: ~8 hours (3 datasets × ~2.7 hours each) | Fair comparison on same sample size! ✅**

**Note**: With `--answer-format strict` and `--max-tokens 10`, evaluation is ~30% faster than default format!

**Note**: Using **temperature=0.9** from paper (line 799) for proper diversity.

### 8. Additional Methods: Semantic Entropy & Self-Verification (NEW!)

Two additional baseline methods from the paper are now available:

**Semantic Entropy (S.E.)** - Kuhn et al. 2023:
```bash
# Run on OpenBookQA (500 examples, ~2 hours)
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_semantic_entropy_500.csv
```

**Self-Verification (S.V.)** - KCAHD 2022:
```bash
# Run on OpenBookQA (500 examples, ~2.5 hours - includes verification step)
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa_self_verification_500.csv
```

**Compare all 5 methods:**
```bash
python scripts/compare_results.py outputs/results/openbookqa_*_500.json
```

**Expected ranking (from paper):** MI ≥ S.E. > Self-Consistency > S.V. > Greedy (on ECE)

### 9. Detailed Demo: Understand How Methods Work

Generate comprehensive trace data showing exactly how each method works:

```bash
# Generate demo for first 5 OpenBookQA questions (all 5 methods)
python demo/scripts/generate_demo.py

# View summary of all methods for question 0
python demo/scripts/view_demo.py --question 0 --method all

# View detailed trace of MI method
python demo/scripts/view_demo.py --question 0 --method mi_method --verbose

# Export markdown report
python demo/scripts/view_demo.py --export-markdown demo/demo_report.md
```

**Demo captures:**
- Raw prompts sent to model (all k×n queries)
- Raw model responses (text + logprobs)
- Intermediate computations (distributions, similarities, pseudo joints)
- Decision logic for each method
- All metrics (confidence, MI, entropy, agreement)

See [`demo/README.md`](demo/README.md) for full documentation.

### 10. Visualize Results

Generate plots and summaries from your evaluation results:

```bash
# Quick summary table
python scripts/summarize_results.py

# Generate all plots (accuracy, ECE, calibration curves)
bash scripts/visualize_all.sh

# Or generate specific plots:

# Comparison plots (accuracy + ECE bars)
python scripts/plot_results.py --dataset all

# Calibration curves (reliability diagrams)
python scripts/plot_calibration.py --dataset all

# Summary for specific dataset
python scripts/summarize_results.py --dataset openbookqa
```

**Generated plots** (saved to `outputs/plots/`):
- **Comparison plots**: Side-by-side accuracy and ECE bars for each dataset
- **Combined plot**: All datasets and methods in one view
- **Calibration curves**: Reliability diagrams showing confidence vs actual accuracy
- **Confidence distributions**: Histograms of confidence scores

💡 **Incremental Runs:**
```bash
# First: Run 100 examples
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 100 --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge_100.csv

# Later: Run full dataset (same questions, new chains)
python -m llm_belief_mi_test.cli --dataset arc-challenge --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge_full.csv
```

💡 **Detailed Logging:**
- All evaluations automatically save detailed per-question logs to `outputs/logs/{run_name}/question_*.json`
- Includes prompts, raw outputs, decision process, and metrics for debugging
- Example: `--output results/test.csv` creates logs at `logs/test/question_0.json`, etc.
- See `LOGGING_STATUS_FINAL.md` for details

⚠️ **Note**: Cache doesn't help with sampling (temp>0) to preserve diversity. But incremental runs help verify correctness before committing to full evaluation!

## 💾 Caching Strategy (Incremental Runs)

**Cache enables smart incremental evaluation:**

### Example Workflow:
```bash
# Day 1: Test with 50 examples (~20 min)
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 50 --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_50.csv

# Day 2: Expand to 200 examples (~1.5 hours)
# Chains regenerated (temp=0.9), but same questions
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 200 --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_200.csv

# Day 3: Full dataset (~3.5 hours)
# Processes all questions (cache doesn't apply with temp=0.9)
python -m llm_belief_mi_test.cli --dataset arc-challenge --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_full.csv
```

**Note**: With temperature=0.9 (sampling mode), cache is disabled to preserve chain diversity. Incremental runs still help verify correctness before committing to full evaluation!

### Cache Management:
```bash
# View cache stats
python -c "from llm_belief_mi_test.cache import SQLiteCache; c = SQLiteCache('.cache/llm_cache.sqlite'); print(f'Cache hits: {c.stats()}')"

# Clear cache if needed
rm -rf .cache/llm_cache.sqlite

# Disable cache for testing
--cache-mode off
```

**💡 Cache behavior update:**
- Cache is **automatically disabled** when temperature > 0 (sampling mode)
- This is **critical** for MI estimation - chains must be diverse!
- Cache **helps when re-running** same questions/datasets, not within K chains
- For baseline comparisons with temperature=0, cache works normally

## ⚡ Performance Optimization (IMPORTANT!)

**Your current run is too slow!** Here's why and how to fix it:

### Problem
- Current: 3.74s per generation → **84 hours total** ❌
- Expected: 1.5s per generation → **34 hours total** ⚠️

### Solution: Use Paper's Parameters

**From paper (line 799): temperature=0.9, k=10**

```bash
--temperature 0.9 --max-tokens 30 --k 10 --n 2
```

**Why these settings?**
- **temperature=0.9**: Paper's value - creates diversity in chains for MI estimation
- **max-tokens=30**: Optimized for MCQ (covers 95%+ of answers)
  - ⚠️ `max_tokens` is ONLY for generated ANSWER, not the question (question is INPUT)
- **k=10, n=2**: Paper's values for proper MI estimation

**Impact:**
- Time: ~24 hours for all datasets (vs 84 hours with max_tokens=64)
- A100 cost: ~$65 (vs $227)
- L4 cost: ~$12 (vs $40) ✅ **Recommended**

### Dataset Size & Time Summary

| Dataset | Examples | Generations | Time (current) | Time (optimized) |
|---------|----------|-------------|----------------|------------------|
| ARC-Challenge | 1,172 | 23,440 | 24.4h | **~7h** ✅ |
| ARC-Easy | 2,376 | 47,520 | 49.4h | **~14h** ✅ |
| OpenBookQA | 500 | 10,000 | 10.4h | **~3h** ✅ |
| **TOTAL** | **4,048** | **80,960** | 84h ❌ | **~24h** ✅ |

**See [DATASET_REQUIREMENTS.md](DATASET_REQUIREMENTS.md) for detailed analysis.**

---

## Parameters

### Model Configuration
- `--model`: Model name (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `--load-in-4bit`: Use 4-bit quantization (recommended, saves memory)
- `--load-in-8bit`: Use 8-bit quantization

### MI Parameters
- `--k`: Number of independent chains per question (default: 10, from paper)
- `--n`: Chain length / pseudo joint dimension (default: 2, from paper)
- `--mi-method`: MI estimator - `plugin` or `listing` (default: `listing`)
- `--confidence-method`: MI→confidence conversion - `inverse`, `exp`, `normalized` (default: `inverse`)

### Generation
- `--temperature`: Sampling temperature (default: 0.5, **use 0.9 from paper for diversity!**)
- `--max-tokens`: Max tokens per generation (default: 64, **use 30 for MCQ efficiency**)

### Caching (Prevents Duplicate Work!)
- `--cache-path`: SQLite cache file (default: `.cache/llm_cache.sqlite`)
- `--cache-mode`: Cache mode - `readwrite` (default), `read`, `write`, or `off`

**How caching works:**
- ✅ Cache is **disabled during sampling** (temperature > 0) to preserve diversity
- ✅ Cache is **enabled for deterministic** generation (temperature = 0)
- ✅ Ensures K chains are truly independent and diverse
- ✅ Run 100 examples, then 1000 - questions are reused (but chains regenerated)
- ⚠️ **Important**: Cache only helps when re-running with same questions, not within K chains

**Why this matters for MI:**
- MI requires diverse chains to measure epistemic uncertainty
- Caching within K chains would make all chains identical (MI=0)
- Cache is smart: disabled during sampling, enabled for greedy baseline comparisons

### Dataset
- `--dataset`: `arc-challenge`, `arc-easy`, `openbookqa`
- `--split`: Dataset split (default: `test`)
- `--limit`: Limit examples for testing (optional)

## Output

### Files Generated

Each run produces **two files** in your specified output location:

1. **CSV file** (`your_output.csv`): Per-question details
   - Question text
   - Predicted answer (A/B/C/D)
   - Ground truth answer
   - Correctness (0/1)
   - Confidence score (0-1)
   - MI score (bits)
   - Agreement across chains

2. **JSON file** (`your_output.json`): Aggregate metrics
   - Accuracy
   - ECE (Expected Calibration Error) ← **KEY METRIC**
   - Average confidence
   - Average MI
   - Sample count

### Example Output Locations

```bash
# Your outputs will be saved to:
outputs/results/test_quick.csv           # Quick test results
outputs/results/test_quick.json          # Quick test metrics
outputs/results/arc_challenge_full.csv   # Full ARC-Challenge results
outputs/results/arc_challenge_full.json  # Full ARC-Challenge metrics
# ... and so on
```

### Console Output

During evaluation, you'll see:
```
EVALUATION RESULTS
============================================================
accuracy            : 0.7250
ece                 : 0.0823
avg_confidence      : 0.6892
avg_mi_bits         : 0.5431
avg_agreement       : 0.7100
n_samples           : 200
============================================================
```

## Project Structure

```
llm-belief-mi-test/
├── README.md                    # ⭐ Start here!
├── requirements.txt             # Dependencies
│
├── llm_belief_mi_test/          # Main package
│   ├── cli.py                  # Command-line interface
│   ├── calibration.py          # ECE & evaluation (with baselines)
│   ├── llm_client_local.py     # Local Llama client
│   ├── mi_estimator.py         # MI computation
│   ├── iterative_prompting.py  # Iterative prompting chains
│   ├── datasets.py             # Dataset loaders
│   ├── evaluation.py           # Metrics utilities
│   └── cache.py                # SQLite caching
│
├── scripts/                     # 🔧 Test & utility scripts
│   ├── test_gpu_setup.py       # Verify GPU and model
│   ├── test_baselines.py       # Test all 3 methods
│   ├── compare_results.py      # Compare baseline results
│   └── RUN_BASELINE_COMPARISON_500.sh  # Run all baselines
│
├── docs/                        # 📚 Detailed documentation
│   ├── BASELINE_COMPARISON_GUIDE.md
│   ├── QUICK_START_BASELINES.md
│   ├── COMMANDS_500_EXAMPLES.txt
│   ├── IMPLEMENTATION_COMPLETE.md
│   └── ... (see docs/ for full list)
│
├── outputs/                     # 💾 Results and logs
│   ├── results/                # CSV & JSON outputs
│   ├── plots/                  # Visualizations
│   └── logs/                   # Execution logs
│
└── doc/                         # 📄 Original paper files
    └── arXiv-2406.02543v2/
```

## Hardware Requirements

### Minimum (4-bit quantization)
- GPU: 12GB VRAM (RTX 3060, RTX 4060 Ti)
- RAM: 16GB
- Estimated time: 10-20 hours for all benchmarks

### Recommended
- GPU: 16GB+ VRAM (RTX 4080, A4000+)
- RAM: 32GB
- Estimated time: 5-10 hours for all benchmarks

## Performance Estimates

With K=10, t=3, 4-bit quantization:
- Single question: ~15-30 seconds
- ARC-Challenge (1200 q's): ~5-10 hours
- ARC-Easy (2400 q's): ~10-15 hours
- OpenBookQA (500 q's): ~2-4 hours

## Troubleshooting

### Out of Memory
```bash
# Use 4-bit quantization
--load-in-4bit

# Or reduce chains/length
--k 5 --t 2
```

### Model Download Issues
```bash
# Set HuggingFace token if needed
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Or use cache directory
export HF_HOME="/path/to/cache"
```

### Slow Generation
- Reduce K (number of chains)
- Reduce t (chain length)
- Use larger batch size (if implementing batching)

## ✅ Complete Workflow (Recommended Incremental Approach)

**Everything you need is in this README. Follow these steps:**

### One-Time Setup (~10 minutes)
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test
pip install -r requirements.txt
export HF_TOKEN="hf_YOUR_TOKEN_HERE"  # Replace with your token
python scripts/test_gpu_setup.py  # Verify GPU and model loading
```

### Incremental Evaluation (Recommended!)

**Day 1: Quick Test** (5 examples, ~2 minutes)
```bash
python -m llm_belief_mi_test.cli --dataset arc-easy --limit 5 --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_easy_5.csv
# ✅ Verify results: MI>0, agreement<1.0, accuracy>0.2
# ✅ Check logs: ls outputs/logs/arc_easy_5/
```

**Day 2: Small Test** (50 examples, ~20 minutes)
```bash
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 50 --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge_50.csv
# ✅ Analyze accuracy (~50-65%), ECE, MI behavior
```

**Day 3: Medium Test** (200 examples, ~1.5 hours)
```bash
python -m llm_belief_mi_test.cli --dataset arc-challenge --limit 200 --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge_200.csv
# ✅ More robust metrics for ECE analysis
```

**Day 4+: Full Evaluation** (~3.5 hours)
```bash
python -m llm_belief_mi_test.cli --dataset arc-challenge --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge_full.csv
# ✅ Complete results for publication
```

**Total time**: ~3.5 hours per dataset with strict mode, spread across days with verification! 🎯

### Where Your Files Are Saved

**All outputs go to** `outputs/results/`:
- ✅ CSV files: Per-question details
- ✅ JSON files: Aggregate metrics (accuracy, ECE, etc.)
- ✅ Automatically created if doesn't exist

**You specify the output path** with `--output` flag:
```bash
--output outputs/results/my_experiment.csv
```
This creates:
- `outputs/results/my_experiment.csv` (details)
- `outputs/results/my_experiment.json` (metrics)

## Next Steps After Running

1. ✅ **Check CSV file**: Open in Excel/Pandas to see per-question results
2. ✅ **Check JSON file**: View aggregate metrics (accuracy, ECE)
3. ✅ **Compare ECE**: Lower ECE = better calibration (your key metric!)
4. ✅ **Analyze MI scores**: High MI = uncertain predictions
5. ✅ **Compare benchmarks**: See how method performs across datasets

## References

- Paper: "To Believe or Not to Believe Your LLM" (DeepMind, 2024)
- Llama 3.1: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- ARC: https://allenai.org/data/arc
- OpenBookQA: https://allenai.org/data/open-book-qa

---

For detailed implementation notes, see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

