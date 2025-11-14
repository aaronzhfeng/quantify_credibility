# LLM Belief MI - Benchmark Evaluation

Evaluation of the Mutual Information (MI) and iterative prompting method from "To Believe or Not to Believe Your LLM" on multiple-choice benchmarks (ARC-Challenge, ARC-Easy, OpenBookQA, TruthfulQA MC1/MC2) and extractive QA datasets (SQuAD v2, TriviaQA).

## 📋 Current Status

- ✅ **Implementation complete and ready to use!**
- ✅ Proper pseudo joint selection implemented (paper's method)
- ✅ Logprobs extraction for probability-weighted selection
- ✅ MI estimation for uncertainty quantification
- ✅ ECE computation for calibration evaluation
- ✅ Full CLI with all benchmarks (ARC/OpenBookQA/TruthfulQA MC1&MC2/SQuAD v2/TriviaQA)
- ✅ **Baseline methods** for comparison (greedy, self-consistency, semantic-entropy, self-verification)
- ✅ **Correctness-based MI** for extractive QA and truthfulness evaluation

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

### Command Reference Guides (Quick Access):

- **[COMMANDS_MCQ.md](COMMANDS_MCQ.md)**: 📋 **MCQ datasets** - ARC-Challenge, ARC-Easy, OpenBookQA (all 5 methods)
- **[COMMANDS_OPENENDED.md](COMMANDS_OPENENDED.md)**: 📋 **Open-ended datasets** - TruthfulQA, SQuAD v2, TriviaQA (3 methods)
- **[COMMANDS_NLI.md](COMMANDS_NLI.md)**: 🔬 **NLI analysis** - Semantic equivalence & mutual entailment (post-hoc analysis)

### Detailed Guides:

- **[docs/IMPLEMENTATION_COMPLETE.md](docs/IMPLEMENTATION_COMPLETE.md)**: ⭐ **START HERE** - Implementation complete, ready to use!
- **[docs/BASELINE_COMPARISON_GUIDE.md](docs/BASELINE_COMPARISON_GUIDE.md)**: 📊 **Baseline comparison guide**
- **[docs/QUICK_START_BASELINES.md](docs/QUICK_START_BASELINES.md)**: Quick reference for running baselines
- **[docs/COMMANDS_500_EXAMPLES.txt](docs/COMMANDS_500_EXAMPLES.txt)**: Copy-paste commands for 500-example runs
- **[docs/AUTHENTICATION_GUIDE.md](docs/AUTHENTICATION_GUIDE.md)**: HuggingFace authentication for Llama models

### Diagnostics & Troubleshooting:

- **[docs/LOGPROB_DIAGNOSTIC.md](docs/LOGPROB_DIAGNOSTIC.md)**: ⚠️ **Log probability validation** - Impact analysis by method, verification commands, remediation strategies

### Theory & Algorithms:

- **[theory/MI_ALGORITHMS.md](../theory/MI_ALGORITHMS.md)**: 📐 **MI estimators explained** - Listing, Plugin, and Original paper algorithm
- **[theory/MI_ESTIMATOR_EXAMPLE.md](../theory/MI_ESTIMATOR_EXAMPLE.md)**: 📊 **Worked example** - Step-by-step MI calculation

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
- `--dataset {arc-challenge,arc-easy,openbookqa,squad-v2,truthfulqa-mc1,truthfulqa-mc2,triviaqa}` **[REQUIRED]**
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

### 7. Run Full Method Comparison (RECOMMENDED - 500 examples per dataset)

For comprehensive comparison, run all 5 methods on 500 examples from each dataset. Each dataset includes commands for individual methods plus a combined command to run all methods sequentially.

---

#### **ARC-Challenge (500 examples)**

**Run methods individually:**

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge --limit 500 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/greedy_500.csv

# 2. Self-consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/selfcons_500.csv

# 3. MI method (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/mi_500.csv

# 4. Semantic Entropy (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/semantic_entropy_500.csv

# 5. Self-Verification (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/self_verification_500.csv
```

**Or run all 5 methods sequentially (~6.5 hours):**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method greedy --dataset arc-challenge --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/greedy_500.csv 2>&1 | tee outputs/logs/arc_challenge_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset arc-challenge --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/selfcons_500.csv 2>&1 | tee outputs/logs/arc_challenge_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset arc-challenge --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/mi_500.csv 2>&1 | tee outputs/logs/arc_challenge_mi_500.log && \
python -m llm_belief_mi_test.cli --method semantic-entropy --dataset arc-challenge --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/semantic_entropy_500.csv 2>&1 | tee outputs/logs/arc_challenge_semantic_entropy_500.log && \
python -m llm_belief_mi_test.cli --method self-verification --dataset arc-challenge --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/self_verification_500.csv 2>&1 | tee outputs/logs/arc_challenge_self_verification_500.log
```

---

#### **ARC-Easy (500 examples)**

**Run methods individually:**

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 500 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/greedy_500.csv

# 2. Self-consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/selfcons_500.csv

# 3. MI method (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/mi_500.csv

# 4. Semantic Entropy (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/semantic_entropy_500.csv

# 5. Self-Verification (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/self_verification_500.csv
```

**Or run all 5 methods sequentially (~6.5 hours):**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method greedy --dataset arc-easy --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/greedy_500.csv 2>&1 | tee outputs/logs/arc_easy_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset arc-easy --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/selfcons_500.csv 2>&1 | tee outputs/logs/arc_easy_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset arc-easy --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/mi_500.csv 2>&1 | tee outputs/logs/arc_easy_mi_500.log && \
python -m llm_belief_mi_test.cli --method semantic-entropy --dataset arc-easy --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/semantic_entropy_500.csv 2>&1 | tee outputs/logs/arc_easy_semantic_entropy_500.log && \
python -m llm_belief_mi_test.cli --method self-verification --dataset arc-easy --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/self_verification_500.csv 2>&1 | tee outputs/logs/arc_easy_self_verification_500.log
```

---

#### **OpenBookQA (500 examples - full dataset)**

**Run methods individually:**

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/greedy_500.csv

# 2. Self-consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/selfcons_500.csv

# 3. MI method (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/mi_500.csv

# 4. Semantic Entropy (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/semantic_entropy_500.csv

# 5. Self-Verification (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/self_verification_500.csv
```

**Or run all 5 methods sequentially (~6.5 hours):**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method greedy --dataset openbookqa --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/greedy_500.csv 2>&1 | tee outputs/logs/openbookqa_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset openbookqa --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/selfcons_500.csv 2>&1 | tee outputs/logs/openbookqa_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/mi_500.csv 2>&1 | tee outputs/logs/openbookqa_mi_500.log && \
python -m llm_belief_mi_test.cli --method semantic-entropy --dataset openbookqa --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/semantic_entropy_500.csv 2>&1 | tee outputs/logs/openbookqa_semantic_entropy_500.log && \
python -m llm_belief_mi_test.cli --method self-verification --dataset openbookqa --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/self_verification_500.csv 2>&1 | tee outputs/logs/openbookqa_self_verification_500.log
```

---

#### **SQuAD v2 (500 examples - Extractive QA)**

**Quick test (5 examples):**

```bash
# Test with 5 examples (~2 minutes)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset squad-v2 --split validation --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 50 \
  --output outputs/results/squad_v2/mi_test_5.csv
```

**Full evaluation (500 examples - all methods):**

```bash
# 1. Greedy baseline (~5 min for 500 examples)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset squad-v2 --split validation --limit 500 \
  --load-in-4bit \
  --max-tokens 50 \
  --output outputs/results/squad_v2/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset squad-v2 --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 50 \
  --output outputs/results/squad_v2/selfcons_500.csv

# 3. MI method (~2 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset squad-v2 --split validation --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 50 \
  --output outputs/results/squad_v2/mi_500.csv
```

**Run all 3 methods sequentially (~3.5 hours):**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method greedy --dataset squad-v2 --split validation --limit 500 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/greedy_500.csv 2>&1 | tee outputs/logs/squad_v2_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset squad-v2 --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/selfcons_500.csv 2>&1 | tee outputs/logs/squad_v2_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset squad-v2 --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/mi_500.csv 2>&1 | tee outputs/logs/squad_v2_mi_500.log
```

**Key differences from MCQ datasets:**
- **No answer-format flag**: Extractive QA uses strict format by default
- **max-tokens=50**: Longer to accommodate extractive answers (vs 10 for MCQ)
- **Metrics**: Uses Exact Match (EM) and F1 score instead of accuracy
- **Unanswerable questions**: SQuAD v2 includes ~50k adversarial unanswerable questions
- **Context**: Each question includes a Wikipedia paragraph as context

**Expected output metrics:**
```
exact_match        : 0.6500  # Exact string match with ground truth
f1                 : 0.7200  # Token-level F1 score (standard SQuAD metric)
ece                : 0.0800  # Expected Calibration Error (lower is better)
avg_confidence     : 0.6800
avg_mi_bits        : 0.5200
avg_agreement      : 0.7100
```

**Dataset info:**
- Total: ~12k validation examples
- ~6k answerable + ~6k unanswerable
- Answers are text spans extracted from context
- Model must learn to output "UNANSWERABLE" for impossible questions

---

#### **TruthfulQA MC1 (817 examples - Truthfulness Evaluation)**

**Quick test (5 examples):**

```bash
# Test with 5 examples (~2 minutes)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc1 --split validation --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/mi_test_5.csv
```

**Full evaluation (500 examples - all available methods):**

```bash
# 1. Greedy baseline (~5 min for 500 examples)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset truthfulqa-mc1 --split validation --limit 500 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset truthfulqa-mc1 --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/selfcons_500.csv

# 3. MI method (~2.5 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc1 --split validation --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/mi_500.csv
```

**Run all 3 methods sequentially (~4 hours):**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method greedy --dataset truthfulqa-mc1 --split validation --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/greedy_500.csv 2>&1 | tee outputs/logs/truthfulqa_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset truthfulqa-mc1 --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/selfcons_500.csv 2>&1 | tee outputs/logs/truthfulqa_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset truthfulqa-mc1 --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/mi_500.csv 2>&1 | tee outputs/logs/truthfulqa_mi_500.log
```

**Key characteristics:**
- **Tests truthfulness**: Questions designed to elicit common human misconceptions
- **Single correct answer**: MC1 format has exactly one true answer among 4-5 choices
- **38 categories**: Health, law, finance, politics, science, etc.
- **Answer format**: Same as other MCQ datasets (A/B/C/D/E)
- **Evaluation**: Standard accuracy + ECE (uses existing MCQ infrastructure)

**Expected output metrics:**
```
accuracy           : 0.5500  # Lower than ARC (questions are adversarial)
ece                : 0.0700  # MI should still provide good calibration
avg_confidence     : 0.6200
avg_mi_bits        : 0.6800  # Higher MI expected (more uncertainty)
avg_agreement      : 0.6500
```

**Dataset info:**
- Total: 817 questions (complete dataset in validation split)
- Categories: Misconceptions, conspiracies, myths, falsehoods
- Designed to be challenging - tests if model repeats human errors
- No train/test split (only validation available)

---

#### **TruthfulQA MC2 (817 examples - Multi-True Truthfulness Evaluation)**

**Quick test (5 examples):**

```bash
# Test with 5 examples (~2 minutes)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc2 --split validation --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/mi_test_5.csv
```

**Full evaluation (500 examples - all available methods):**

```bash
# 1. Greedy baseline (~5 min for 500 examples)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset truthfulqa-mc2 --split validation --limit 500 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset truthfulqa-mc2 --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/selfcons_500.csv

# 3. MI method (~2.5 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc2 --split validation --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/mi_500.csv
```

**Run all 3 methods sequentially (~4 hours):**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method greedy --dataset truthfulqa-mc2 --split validation --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/greedy_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc2_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset truthfulqa-mc2 --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/selfcons_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc2_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset truthfulqa-mc2 --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/mi_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc2_mi_500.log
```

**Key characteristics:**
- **Multi-true format**: Questions have MULTIPLE correct answers (≥1)
- **Harder than MC1**: Model must identify ALL truthful statements, not just one
- **Correctness-based MI**: MI computed on binary correctness (matches ANY true answer)
- **Partial credit**: Answering any correct option counts as correct
- **38 categories**: Same as MC1 (Health, law, finance, politics, science, etc.)

**Expected output metrics:**
```
accuracy           : 0.6500  # Higher than MC1 (partial credit from multiple correct answers)
ece                : 0.0650  # MI provides good calibration
avg_confidence     : 0.6400
avg_mi_bits        : 0.6500
avg_correctness_agreement : 0.6800
```

**Dataset info:**
- Total: 817 questions (complete dataset in validation split)
- Same questions as MC1 but different answer set (multiple correct per question)
- More forgiving scoring (any correct answer counts)
- Tests ability to recognize multiple truths

---

#### **TriviaQA (87,622 examples - Open-Domain Trivia)**

**Quick test (5 examples):**

```bash
# Test with 5 examples (~2 minutes)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa --split validation --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 50 \
  --output outputs/results/triviaqa/mi_test_5.csv
```

**Full evaluation (500 examples - all available methods):**

```bash
# 1. Greedy baseline (~5 min for 500 examples)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset triviaqa --split validation --limit 500 \
  --load-in-4bit \
  --max-tokens 50 \
  --output outputs/results/triviaqa/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset triviaqa --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 50 \
  --output outputs/results/triviaqa/selfcons_500.csv

# 3. MI method (~2.5 hours for 500 examples)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa --split validation --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 50 \
  --output outputs/results/triviaqa/mi_500.csv
```

**Run all 3 methods sequentially (~4 hours):**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method greedy --dataset triviaqa --split validation --limit 500 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/greedy_500.csv 2>&1 | tee outputs/logs/triviaqa_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset triviaqa --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/selfcons_500.csv 2>&1 | tee outputs/logs/triviaqa_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset triviaqa --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/mi_500.csv 2>&1 | tee outputs/logs/triviaqa_mi_500.log
```

**Key characteristics:**
- **Open-domain QA**: Pure knowledge testing (no context provided)
- **Uses rc.nocontext subset**: Excludes search results/evidence documents
- **Multiple answer aliases**: Each question has several acceptable answer forms
  - Example: "Sinclair Lewis", "Harry Sinclair Lewis", "Lewis, (Harry) Sinclair"
- **Correctness-based MI**: MI computed on binary correctness (matches any alias)
- **Evaluation**: Exact Match (EM) + F1 score (standard TriviaQA metrics)

**Expected output metrics:**
```
exact_match        : 0.4500  # Challenging (pure knowledge, no context)
f1                 : 0.5200  # Higher than EM (partial credit for token overlap)
ece                : 0.0900  # MI still provides calibration
avg_confidence     : 0.5800
avg_mi_bits        : 0.7500  # Higher MI (more uncertainty without context)
avg_correctness_agreement : 0.6200
```

**Dataset info:**
- Total: 87,622 train / 11,313 validation / 10,832 test
- Subset used: `rc.nocontext` (no evidence documents)
- Questions from trivia websites and quiz bowls
- Answer evaluation: Matches any alias after normalization

**Differences from SQuAD v2:**
- **No context**: Pure knowledge test (vs reading comprehension)
- **Always answerable**: No unanswerable questions (vs ~50% in SQuAD v2)
- **More aliases**: Typically 3-10 answer variations (vs 1-3 in SQuAD)
- **max-tokens=50**: Same as SQuAD (vs 10 for MCQ)

---

#### **Compare Results**

**Compare each dataset:**
```bash
python scripts/compare_results.py outputs/results/arc_challenge/*.json
python scripts/compare_results.py outputs/results/arc_easy/*.json
python scripts/compare_results.py outputs/results/openbookqa/*.json
python scripts/compare_results.py outputs/results/squad_v2/*.json
python scripts/compare_results.py outputs/results/truthfulqa/*.json
python scripts/compare_results.py outputs/results/truthfulqa_mc2/*.json
python scripts/compare_results.py outputs/results/triviaqa/*.json
```

**Visualize all results:**
```bash
bash scripts/visualize_all.sh
```

---

**Total time per dataset:**
- **MCQ datasets** (ARC, OpenBookQA): ~6.5 hours | All 5 methods with comprehensive comparison! ✅
- **Open-ended datasets** (TruthfulQA, SQuAD v2, TriviaQA): ~4 hours | 3 methods (Greedy, Self-Consistency, MI) ✅

**Expected ranking (from paper):** MI ≥ Semantic Entropy > Self-Consistency > Self-Verification > Greedy (on ECE)

**Methods available by dataset type:**
- **MCQ** (ARC, OpenBookQA): All 5 methods (Greedy, Self-Consistency, Semantic Entropy, Self-Verification, MI)
- **TruthfulQA MC1/MC2**: 3 methods (Greedy, Self-Consistency, MI with correctness-based evaluation)
- **SQuAD v2, TriviaQA**: 3 methods (Greedy, Self-Consistency, MI with correctness-based evaluation)

**Notes:**
- Combined commands use `&&` so if one method fails, the rest won't run
- Logs are saved with `tee` to both screen and log files
- With `--answer-format strict` and `--max-tokens 10`, evaluation is ~70% faster than default (MCQ)
- For extractive QA, use `--max-tokens 50` to accommodate longer answers
- All detailed per-question logs automatically saved to `outputs/logs/{run_name}/`

---

### 8. Parameter Ablation Study (MI Method on OpenBookQA)

Systematic exploration of MI method parameters to understand their impact on accuracy and calibration (ECE). This ablation study varies one parameter at a time from the baseline configuration.

**Baseline Configuration (from paper):**
- Temperature: 0.9
- k (number of chains): 10
- n (chain length): 2
- MI estimator: listing
- Confidence method: inverse
- Answer format: strict
- Max tokens: 10

**Total experiments: 16 runs (~10 hours for 200 examples each) | One parameter varied at a time**

---

#### **8.1. Temperature Ablation** (3 runs, ~1.8 hours for 200 examples)

**What it controls:** Sampling diversity in response generation.

**Formula:** Softmax with temperature: `P(token) ∝ exp(logit / T)`
- **Low T (0.5)**: Sharper distribution → more deterministic, less diverse chains
- **Medium T (0.9)**: Paper's baseline → balanced exploration
- **High T (1.3)**: Flatter distribution → more random, highly diverse chains

**Why this matters for MI:**
- MI measures epistemic uncertainty through chain diversity
- Too low: Chains too similar → underestimates uncertainty (low MI)
- Too high: Chains too random → overestimates uncertainty (high MI)
- Optimal: Captures genuine model uncertainty

**Expected results:**
- T=0.5: Lower MI, higher accuracy, possibly worse ECE (overconfident)
- T=0.9: Baseline performance (paper-validated)
- T=1.3: Higher MI, lower accuracy, potentially better ECE if model is overconfident

**Commands:**

```bash
# Temperature = 0.5 (lower diversity)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.5 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/temperature/temp0.5.csv

# Temperature = 0.9 (baseline, paper's value)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/temperature/temp0.9.csv

# Temperature = 1.3 (higher diversity)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 1.3 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/temperature/temp1.3.csv
```

**Combined command:**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.5 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/temperature/temp0.5.csv 2>&1 | tee outputs/logs/ablation_temp0.5.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/temperature/temp0.9.csv 2>&1 | tee outputs/logs/ablation_temp0.9.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 1.3 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/temperature/temp1.3.csv 2>&1 | tee outputs/logs/ablation_temp1.3.log
```

---

#### **8.2. Number of Chains (k) Ablation** (3 runs, ~1.5 hours for 200 examples)

**What it controls:** Number of independent sampling chains per question.

**Formula:** Each chain samples response sequence independently with temperature T
- **k=5**: Fewer chains → faster, but less robust statistics
- **k=10**: Paper's baseline → good balance
- **k=20**: More chains → better statistics, more expensive

**Why this matters for MI:**
- MI estimation requires multiple samples: `MI = Σ H(Yi) - H(Y1,...,Yn)`
- More chains → better empirical estimates of marginal/joint entropies
- Trade-off: Computation cost vs statistical robustness

**Expected results:**
- k=5: Noisier MI estimates, faster runtime
- k=10: Baseline (paper-validated)
- k=20: More stable MI/ECE, 2× runtime

**Commands:**

```bash
# k = 5 (fewer chains, faster)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 5 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/k_chains/k5.csv

# k = 10 (baseline, paper's value)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/k_chains/k10.csv

# k = 20 (more chains, better statistics)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 20 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/k_chains/k20.csv
```

**Combined command:**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 5 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/k_chains/k5.csv 2>&1 | tee outputs/logs/ablation_k5.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/k_chains/k10.csv 2>&1 | tee outputs/logs/ablation_k10.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 20 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/k_chains/k20.csv 2>&1 | tee outputs/logs/ablation_k20.log
```

---

#### **8.3. Chain Length (n) Ablation** (3 runs, ~2.4 hours for 200 examples)

**What it controls:** Length of iterative prompting sequence (pseudo-joint dimension).

**Formula:** Pseudo-joint distribution `Q̃(Y1, Y2, ..., Yn)` with MI:
```
MI(Y1; Y2; ...; Yn) = Σᵢ H(Yᵢ) - H(Y1,...,Yn)
```
- **n=2**: Shortest chain → (initial, refined) → fast
- **n=3**: Medium chain → (initial, refined₁, refined₂) → more context
- **n=4**: Longest chain → maximum iterative refinement → richest MI signal

**Why this matters for MI:**
- Longer chains capture more iterative reasoning
- Each step conditions on previous answers in chain
- Higher n → richer dependency structure → potentially more informative MI
- Trade-off: More queries per chain (n × k total queries)

**Expected results:**
- n=2: Baseline (paper-validated)
- n=3: More context, potentially better uncertainty capture, +50% runtime
- n=4: Maximum refinement, highest MI potential, +100% runtime

**Commands:**

```bash
# n = 2 (baseline, paper's value)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/n_length/n2.csv

# n = 3 (longer chains, more context)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 3 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/n_length/n3.csv

# n = 4 (longest chains, maximum iterative context)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 4 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/n_length/n4.csv
```

**Combined command:**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/n_length/n2.csv 2>&1 | tee outputs/logs/ablation_n2.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 3 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/n_length/n3.csv 2>&1 | tee outputs/logs/ablation_n3.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 4 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/n_length/n4.csv 2>&1 | tee outputs/logs/ablation_n4.log
```

---

#### **8.4. MI Estimator Method Ablation** (2 runs, ~1.2 hours for 200 examples)

**What it controls:** Algorithm for estimating mutual information from samples.

**Formulas:**

**Plugin estimator (simple):**
```
MI = Σᵢ H(Yᵢ) - H(Y1,...,Yn)
H(X) = -Σ p(x) log p(x)  [empirical probabilities]
```
- Direct plug-in of empirical distributions
- Simple and intuitive
- Biased for small samples (underestimates MI)

**Listing estimator (paper's Algorithm 1):**
```
MI = Σ μ̂ · log((μ̂ + γ₁) / (μ̂_prod + γ₂))
μ̂ = empirical joint probabilities
μ̂_prod = product of marginals
γ₁, γ₂ = 1/k (regularization)
```
- From paper's listing.tex (Algorithm 1)
- Regularized with smoothing parameters
- Better for small sample sizes
- More sophisticated estimation

**Why this matters:**
- Different estimators may give different MI scales
- Should show similar trends across experiments
- Tests robustness of conclusions to estimation method

**Expected results:**
- **Plugin**: Lower MI values (finite sample bias), simpler
- **Listing**: Higher MI values (regularization), paper's default
- Both should rank questions similarly (correlation ~0.9+)

**Commands:**

```bash
# MI method = listing (baseline, paper's default)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --mi-method listing \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/mi_method/listing.csv

# MI method = plugin (alternative estimator)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --mi-method plugin \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/mi_method/plugin.csv
```

**Combined command:**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --mi-method listing --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/mi_method/listing.csv 2>&1 | tee outputs/logs/ablation_listing.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --mi-method plugin --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/mi_method/plugin.csv 2>&1 | tee outputs/logs/ablation_plugin.log
```

---

#### **8.5. Confidence Conversion Method Ablation** (3 runs, ~1.8 hours for 200 examples)

**What it controls:** How MI (uncertainty) is converted to confidence score.

**Formulas (all map high MI → low confidence):**

**Inverse (baseline):**
```
confidence = 1 / (1 + MI)
```
- Linear-like decay
- MI=0 → conf=1.0, MI=1 → conf=0.5, MI=5 → conf=0.17

**Exponential:**
```
confidence = exp(-MI)
```
- Aggressive penalty for high uncertainty
- MI=0 → conf=1.0, MI=1 → conf=0.37, MI=5 → conf=0.007
- Much steeper drop than inverse

**Normalized:**
```
confidence = 1 - (MI / (MI + 1))
```
- Mathematically equivalent to inverse!
- MI=0 → conf=1.0, MI=1 → conf=0.5, MI=5 → conf=0.17
- Alternative formulation of same function

**Why this matters for calibration:**
- ECE directly depends on confidence values
- Different mappings change calibration curves
- Exp gives lower confidences → might improve ECE if model overconfident
- Tests sensitivity of ECE to confidence scale

**Expected results:**
- **Inverse**: Baseline ECE (paper's default)
- **Exp**: Lower confidences, different ECE (might improve if overconfident)
- **Normalized**: Nearly identical to inverse (mathematical equivalence)

**Commands:**

```bash
# Confidence = inverse (baseline, 1/(1+MI))
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --confidence-method inverse \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/confidence_method/inverse.csv

# Confidence = exp (exponential: exp(-MI))
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --confidence-method exp \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/confidence_method/exp.csv

# Confidence = normalized (1 - MI/(MI+1))
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --confidence-method normalized \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/confidence_method/normalized.csv
```

**Combined command:**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --confidence-method inverse --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/confidence_method/inverse.csv 2>&1 | tee outputs/logs/ablation_conf_inverse.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --confidence-method exp --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/confidence_method/exp.csv 2>&1 | tee outputs/logs/ablation_conf_exp.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --confidence-method normalized --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/confidence_method/normalized.csv 2>&1 | tee outputs/logs/ablation_conf_normalized.log
```

---

#### **8.6. Answer Format Ablation** (2 runs, ~1.6 hours for 200 examples)

**What it controls:** Output format constraint and parsing strategy.

**Formats:**

**Strict (baseline):**
- System prompt: "Output ONLY the letter (A, B, C, or D)"
- Model outputs: "A" or "B" (1-2 tokens)
- Extraction: Direct (first character)
- **Pros**: Fast, 100% parseable, minimal output variance
- **Cons**: Restricts model expression, might not capture natural uncertainty

**Codeblock:**
- System prompt: "Output answer in triple backticks: \`\`\`A\`\`\`"
- Model outputs: \`\`\`B\`\`\` (with markers)
- Extraction: Parse codeblock content
- **Pros**: Clear delimiter, still parseable, allows slight formatting flexibility
- **Cons**: More tokens (~5-10), slightly more verbose

**Why we skip "default" format:**
- Default uses fuzzy matching (substring search, similarity)
- Introduces **parsing uncertainty** on top of model uncertainty
- MI should measure model uncertainty, not extraction noise!
- Results would confound two sources of variance

**Why this matters:**
- Strict may suppress genuine model uncertainty (forced brevity)
- Codeblock allows slightly more expression while remaining parseable
- Tests if format restriction affects MI/ECE

**Expected results:**
- **Strict**: Baseline (fastest, most constrained)
- **Codeblock**: Slightly longer runtime, possibly different MI if format affects generation

**Commands:**

```bash
# Answer format = strict (baseline, single letter)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/answer_format/strict.csv

# Answer format = codeblock (answer in triple backticks)
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 30 --answer-format codeblock \
  --output outputs/results/ablation/answer_format/codeblock.csv
```

**Combined command:**
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/answer_format/strict.csv 2>&1 | tee outputs/logs/ablation_format_strict.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 30 --answer-format codeblock --output outputs/results/ablation/answer_format/codeblock.csv 2>&1 | tee outputs/logs/ablation_format_codeblock.log
```

---

#### **8.7. Run All Ablations Sequentially** (~10 hours total for 200 examples each)

Run all 16 ablation experiments in one command:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
# Temperature ablation (3 runs)
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.5 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/temperature/temp0.5.csv 2>&1 | tee outputs/logs/ablation_temp0.5.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/temperature/temp0.9.csv 2>&1 | tee outputs/logs/ablation_temp0.9.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 1.3 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/temperature/temp1.3.csv 2>&1 | tee outputs/logs/ablation_temp1.3.log && \
# k ablation (3 runs)
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 5 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/k_chains/k5.csv 2>&1 | tee outputs/logs/ablation_k5.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/k_chains/k10.csv 2>&1 | tee outputs/logs/ablation_k10.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 20 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/k_chains/k20.csv 2>&1 | tee outputs/logs/ablation_k20.log && \
# n ablation (3 runs)
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/n_length/n2.csv 2>&1 | tee outputs/logs/ablation_n2.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 3 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/n_length/n3.csv 2>&1 | tee outputs/logs/ablation_n3.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 4 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/n_length/n4.csv 2>&1 | tee outputs/logs/ablation_n4.log && \
# MI method ablation (2 runs)
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --mi-method listing --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/mi_method/listing.csv 2>&1 | tee outputs/logs/ablation_listing.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --mi-method plugin --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/mi_method/plugin.csv 2>&1 | tee outputs/logs/ablation_plugin.log && \
# Confidence method ablation (3 runs)
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --confidence-method inverse --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/confidence_method/inverse.csv 2>&1 | tee outputs/logs/ablation_conf_inverse.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --confidence-method exp --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/confidence_method/exp.csv 2>&1 | tee outputs/logs/ablation_conf_exp.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --confidence-method normalized --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/confidence_method/normalized.csv 2>&1 | tee outputs/logs/ablation_conf_normalized.log && \
# Answer format ablation (2 runs)
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/ablation/answer_format/strict.csv 2>&1 | tee outputs/logs/ablation_format_strict.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 30 --answer-format codeblock --output outputs/results/ablation/answer_format/codeblock.csv 2>&1 | tee outputs/logs/ablation_format_codeblock.log
```

---

#### **8.8. Analyze Ablation Results**

After running ablations, compare and visualize the results:

```bash
# Compare each ablation category
python scripts/compare_results.py outputs/results/ablation/temperature/*.json
python scripts/compare_results.py outputs/results/ablation/k_chains/*.json
python scripts/compare_results.py outputs/results/ablation/n_length/*.json
python scripts/compare_results.py outputs/results/ablation/mi_method/*.json
python scripts/compare_results.py outputs/results/ablation/confidence_method/*.json
python scripts/compare_results.py outputs/results/ablation/answer_format/*.json

# Compare all ablation results together
python scripts/compare_results.py outputs/results/ablation/*/*.json

# Generate summary table
python scripts/summarize_results.py --pattern "ablation/*" --export-csv outputs/results/ablation_summary.csv

# Visualize all results
bash scripts/visualize_all.sh
```

**Key questions to answer:**
1. **Temperature**: Does higher T increase MI? At what cost to accuracy?
2. **k (chains)**: Do more chains stabilize ECE? Diminishing returns?
3. **n (length)**: Do longer chains improve uncertainty quantification?
4. **MI estimator**: Are results robust to estimation method?
5. **Confidence method**: Which mapping gives best ECE?
6. **Answer format**: Does format constraint affect MI measurement?

**Expected insights:**
- **Sensitivity**: Which parameters matter most for ECE?
- **Robustness**: Are conclusions stable across parameter choices?
- **Optimal settings**: Best configuration for your use case
- **Trade-offs**: Performance vs computational cost

**Total: 16 experiments (~24 hours) | Comprehensive parameter sensitivity analysis! 📊**

---

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

#### **10.1. Standard Method Comparison**

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

---

#### **10.2. Ablation Study Visualization**

After running ablation experiments (Section 8), visualize the results:

```bash
# Generate all ablation plots
bash scripts/visualize_ablations.sh

# Or plot specific parameters:

# Temperature ablation (T=0.5, 0.9, 1.3)
python scripts/plot_ablation.py --parameter temperature

# Number of chains (k=5, 10, 20)
python scripts/plot_ablation.py --parameter k_chains

# Chain length (n=2, 3, 4)
python scripts/plot_ablation.py --parameter n_length

# MI estimator (listing vs plugin)
python scripts/plot_ablation.py --parameter mi_method

# Confidence conversion (inverse, exp, normalized)
python scripts/plot_ablation.py --parameter confidence_method

# Answer format (strict vs codeblock)
python scripts/plot_ablation.py --parameter answer_format

# Combined plot showing all parameters
python scripts/plot_ablation.py --combined

# Plot all parameters individually + combined
python scripts/plot_ablation.py --all
```

**Generated ablation plots** (saved to `outputs/plots/ablation/`):
- **Individual parameter plots**: Separate plots for each ablated parameter
- **Combined plot**: All 6 parameter ablations in a single comprehensive view
- Each plot shows both accuracy and ECE side-by-side
- Green borders highlight the best-performing values
- Dual y-axis in combined plot for clear comparison

**Usage examples:**
```bash
# After running temperature ablation (Section 8.1):
python scripts/plot_ablation.py --parameter temperature
# Output: outputs/plots/ablation/ablation_temperature.png

# After running all ablations (Section 8.7):
bash scripts/visualize_ablations.sh
# Output: All 7 plots in outputs/plots/ablation/
```

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
- `--dataset`: `arc-challenge`, `arc-easy`, `openbookqa`, `squad-v2`, `truthfulqa-mc1`, `truthfulqa-mc2`, `triviaqa`
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
- SQuAD v2: https://huggingface.co/datasets/rajpurkar/squad_v2
- TruthfulQA: https://huggingface.co/datasets/truthful_qa
- TriviaQA: https://huggingface.co/datasets/mandarjoshi/trivia_qa

---

For detailed implementation notes, see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

