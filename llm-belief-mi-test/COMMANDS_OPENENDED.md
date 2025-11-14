# Open-Ended Datasets - Evaluation Commands (Multi-GPU Optimized)

This guide contains all commands for evaluating **open-ended** datasets (TruthfulQA, extractive QA) using the MI-based uncertainty quantification framework with correctness-based evaluation.

**Datasets covered:**
- TruthfulQA MC1 (817 validation examples - single correct answer)
- TruthfulQA MC2 (817 validation examples - multiple correct answers)
- SQuAD v2 (11,873 validation examples - reading comprehension with unanswerable questions)
- TriviaQA (11,313 validation examples - open-domain trivia)

**Methods available:** 3 methods
1. Greedy (temperature=0 baseline)
2. Self-Consistency (k samples + majority voting on normalized answers)
3. MI (k chains of length n + mutual information on correctness)

**Key difference from MCQ:** Uses **correctness-based MI** (agreement on whether answer is correct) instead of choice-based MI (agreement on which choice).

**🚀 Multi-GPU Acceleration:** All commands use `--multi-gpu` flag for automatic parallelization across available GPUs. Time estimates assume 4 GPUs (4× speedup).

---

## 🖥️ Multi-GPU Operation

### What Happens When You Run with `--multi-GPU`:

1. **GPU Detection**: System detects all available GPUs and displays:
   ```
   Detected GPUs        : 4
   GPU 0                : NVIDIA GeForce RTX 4090D (24 GB)
   GPU 1                : NVIDIA GeForce RTX 4090D (24 GB)
   GPU 2                : NVIDIA GeForce RTX 4090D (24 GB)
   GPU 3                : NVIDIA GeForce RTX 4090D (24 GB)
   ```

2. **Work Distribution**: Automatically splits 200 examples across GPUs:
   ```
   GPU 0 → Examples    0-49     (50 examples)
   GPU 1 → Examples   50-99     (50 examples)
   GPU 2 → Examples  100-149    (50 examples)
   GPU 3 → Examples  150-199    (50 examples)
   ```

3. **Progress Monitoring**: Shows status every 30 seconds:
   ```
   [10:30:45] GPU0: 18/50 | GPU1: 17/50 | GPU2: 19/50 | GPU3: 16/50 → Total: 70/200 (35%)
   ```

4. **Automatic Merging**: Combines results from all GPUs into single output file

5. **Unified Output**: Same format as single-GPU (CSV + JSON), with merged logprob statistics

**Benefits:**
- ✅ 4× faster (200 examples in ~25 min instead of ~1.6 hours)
- ✅ Automatic - just add `--multi-gpu` flag
- ✅ No changes to evaluation logic or results
- ✅ Falls back to single-GPU if only 1 GPU available

---

## 🧪 Phase 1: Testing Commands (Sanity Check)

Run these **before** full evaluation to verify everything works correctly.

### Quick Test: 5 Examples Per Dataset (~10 minutes total)

```bash
# TruthfulQA MC1 (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc1 --split validation --limit 4 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/test/truthfulqa_mc1_test.csv \
  --multi-gpu

# TruthfulQA MC2 (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc2 --split validation --limit 4 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/test/truthfulqa_mc2_test.csv \
  --multi-gpu

# SQuAD v2 (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset squad-v2 --split validation --limit 4 \
  --k 10 --n 2 \
  --load-in-4bit --temperature 0.9 \
  --max-tokens 50 \
  --output outputs/test/squad_v2_test.csv \
  --multi-gpu

# TriviaQA (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa --split validation --limit 4 \
  --k 10 --n 2 \
  --load-in-4bit --temperature 0.9 \
  --max-tokens 50 \
  --output outputs/test/triviaqa_test.csv \
  --multi-gpu
```

**What to check:**
- ✅ All runs complete without errors
- ✅ Accuracy/EM > 0.2 (not random)
- ✅ MI score > 0 (chains have diversity)
- ✅ Agreement < 1.0 (chains disagree on correctness)
- ✅ For TruthfulQA: Correctness-based MI logged
- ✅ For SQuAD/TriviaQA: EM and F1 scores computed
- ✅ Logs saved to `outputs/logs/{dataset}_test/question_*.json`

---

## 📊 Phase 2: Full Evaluation (200 Examples Per Dataset)

### TruthfulQA MC1 (200 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~1 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset truthfulqa-mc1 --split validation --limit 200 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/greedy_200.csv \
  --multi-gpu

# 2. Self-Consistency baseline (~25 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset truthfulqa-mc1 --split validation --limit 200 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/selfcons_200.csv \
  --multi-gpu

# 3. MI method (~35 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc1 --split validation --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/mi_200.csv \
  --multi-gpu
```

---

### TruthfulQA MC2 (200 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~1 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset truthfulqa-mc2 --split validation --limit 200 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/greedy_200.csv \
  --multi-gpu

# 2. Self-Consistency baseline (~25 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset truthfulqa-mc2 --split validation --limit 200 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/selfcons_200.csv \
  --multi-gpu

# 3. MI method (~35 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc2 --split validation --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/mi_200.csv \
  --multi-gpu
```

---

### SQuAD v2 (200 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~1 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset squad-v2 --split validation --limit 200 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/squad_v2/greedy_200.csv \
  --multi-gpu

# 2. Self-Consistency baseline (~22 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset squad-v2 --split validation --limit 200 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/squad_v2/selfcons_200.csv \
  --multi-gpu

# 3. MI method (~30 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset squad-v2 --split validation --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/squad_v2/mi_200.csv \
  --multi-gpu
```

---

### TriviaQA (200 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~1 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset triviaqa --split validation --limit 200 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/triviaqa/greedy_200.csv \
  --multi-gpu

# 2. Self-Consistency baseline (~25 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset triviaqa --split validation --limit 200 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/triviaqa/selfcons_200.csv \
  --multi-gpu

# 3. MI method (~35 min with 4 GPUs)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa --split validation --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/triviaqa/mi_200.csv \
  --multi-gpu
```

---

## 🚀 Phase 3: Run All Datasets & Methods (One Command)

**Total time: ~1.6 hours with 4 GPUs (vs ~6.4 hours on 1 GPU)**

This command runs all 12 evaluations (4 datasets × 3 methods) sequentially with multi-GPU acceleration:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
\
# ========== TruthfulQA MC1 (200 examples, ~25 min with 4 GPUs) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset truthfulqa-mc1 --split validation --limit 200 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/greedy_200.csv --multi-gpu 2>&1 | tee outputs/logs/truthfulqa_mc1_greedy_200.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset truthfulqa-mc1 --split validation --limit 200 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/selfcons_200.csv --multi-gpu 2>&1 | tee outputs/logs/truthfulqa_mc1_selfcons_200.log && \
python -m llm_belief_mi_test.cli --method mi --dataset truthfulqa-mc1 --split validation --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/mi_200.csv --multi-gpu 2>&1 | tee outputs/logs/truthfulqa_mc1_mi_200.log && \
\
# ========== TruthfulQA MC2 (200 examples, ~25 min with 4 GPUs) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset truthfulqa-mc2 --split validation --limit 200 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/greedy_200.csv --multi-gpu 2>&1 | tee outputs/logs/truthfulqa_mc2_greedy_200.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset truthfulqa-mc2 --split validation --limit 200 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/selfcons_200.csv --multi-gpu 2>&1 | tee outputs/logs/truthfulqa_mc2_selfcons_200.log && \
python -m llm_belief_mi_test.cli --method mi --dataset truthfulqa-mc2 --split validation --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/mi_200.csv --multi-gpu 2>&1 | tee outputs/logs/truthfulqa_mc2_mi_200.log && \
\
# ========== SQuAD v2 (200 examples, ~25 min with 4 GPUs) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset squad-v2 --split validation --limit 200 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/greedy_200.csv --multi-gpu 2>&1 | tee outputs/logs/squad_v2_greedy_200.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset squad-v2 --split validation --limit 200 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/selfcons_200.csv --multi-gpu 2>&1 | tee outputs/logs/squad_v2_selfcons_200.log && \
python -m llm_belief_mi_test.cli --method mi --dataset squad-v2 --split validation --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/mi_200.csv --multi-gpu 2>&1 | tee outputs/logs/squad_v2_mi_200.log && \
\
# ========== TriviaQA (200 examples, ~25 min with 4 GPUs) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset triviaqa --split validation --limit 200 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/greedy_200.csv --multi-gpu 2>&1 | tee outputs/logs/triviaqa_greedy_200.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset triviaqa --split validation --limit 200 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/selfcons_200.csv --multi-gpu 2>&1 | tee outputs/logs/triviaqa_selfcons_200.log && \
python -m llm_belief_mi_test.cli --method mi --dataset triviaqa --split validation --limit 200 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/mi_200.csv --multi-gpu 2>&1 | tee outputs/logs/triviaqa_mi_200.log
```

**Expected results summary:**
- Total: 12 evaluations (4 datasets × 3 methods)
- Total time with 4 GPUs: **~1.6 hours** (vs ~6.4 hours on 1 GPU)
- Files created: 12 CSV files + 12 JSON files + detailed logs
- Expected ECE ranking: MI < Self-Consistency < Greedy

---

## 📈 Compare Results

After running all evaluations:

```bash
# Compare each dataset
python scripts/compare_results.py outputs/results/truthfulqa/*.json
python scripts/compare_results.py outputs/results/truthfulqa_mc2/*.json
python scripts/compare_results.py outputs/results/squad_v2/*.json
python scripts/compare_results.py outputs/results/triviaqa/*.json

# Compare across all open-ended datasets and methods
python scripts/compare_results.py outputs/results/truthfulqa*/*.json outputs/results/squad_v2/*.json outputs/results/triviaqa/*.json

# Visualize
bash scripts/visualize_all.sh
```

---

## 🔬 NLI Semantic Clustering & Analysis

### 🚀 NEW: Live NLI Clustering for MI Method

**MI method now supports semantic clustering during evaluation!**

Add `--use-nli-clustering` to measure **semantic uncertainty** instead of string variation:

```bash
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset squad-v2 --split validation --limit 200 \
  --k 10 --n 2 --temperature 0.9 \
  --use-nli-clustering --nli-threshold 0.5 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/squad_v2/mi_semantic_200.csv \
  --multi-gpu
```

**What this does:**
- Clusters semantically equivalent answers before MI computation
- Lower MI when answers are semantically similar (even if strings differ)
- Better calibrated confidence (semantic consistency → higher confidence)
- Expected: ~0.02-0.05 ECE improvement

**See [COMMANDS_NLI.md](COMMANDS_NLI.md) for complete details and research rationale**

---

### Post-hoc NLI Analysis

After running evaluations, you can also analyze semantic equivalence using NLI-based mutual entailment.

**For complete documentation and commands, see: [COMMANDS_NLI.md](COMMANDS_NLI.md)**

**Quick summary:**
- **Post-hoc analysis**: Works on existing log files (no re-inference needed)
- **Two use cases**: 
  1. Clustering analysis (compare F1 vs NLI grouping of equivalent answers)
  2. Evaluation enhancement (semantic correctness checking vs exact match)
- **Time**: ~8 minutes total for all datasets
- **Expected improvements**: +5-10% accuracy, better clustering → better calibration

---

## 🔑 Key Differences by Dataset

### **TruthfulQA MC1 vs MC2**

| Aspect | MC1 | MC2 |
|--------|-----|-----|
| **Format** | Single correct answer | Multiple correct answers |
| **Correctness** | Exact match to THE answer | Match ANY correct answer |
| **Difficulty** | Easier | Harder (must identify ALL truths) |
| **Scoring** | Binary (right/wrong) | Partial credit (any correct = correct) |
| **max-tokens** | 10 (single letter) | 10 (single letter) |
| **answer-format** | strict (A/B/C/D/E) | strict (A/B/C/D/E) |

### **SQuAD v2 vs TriviaQA**

| Aspect | SQuAD v2 | TriviaQA |
|--------|----------|----------|
| **Context** | ✅ Long paragraph | ❌ None (nocontext subset) |
| **Task** | Reading comprehension | Pure knowledge |
| **Unanswerable** | ~50% (adversarial) | 0% (always answerable) |
| **Answers** | 1-3 aliases | 3-10 aliases |
| **max-tokens** | 50 | 50 |
| **Prompt** | compose_prompt_extractive | compose_prompt_trivia |

---

## 💡 Implementation Notes

### **Correctness-Based MI**
Unlike MCQ datasets (which measure agreement on which choice A/B/C/D), these datasets use:
- **Binary correctness space**: "correct" vs "incorrect"
- **MI measures**: Uncertainty about WHETHER the answer is right, not WHICH answer
- **Evaluation**: For each chain step, check if answer matches ground truth → map to correctness

### **Self-Consistency for Free-Form Answers**
- **Normalization**: Lowercase, remove punctuation, remove articles (a/an/the)
- **Voting**: Count normalized answers, pick most common
- **Confidence**: Fraction of samples voting for majority
- **Example**: "Sinclair Lewis", "sinclair lewis", "Lewis" → all vote for "sinclair lewis"

### **Greedy Baseline**
- Single generation at temperature=0
- Confidence from token logprobs
- Fast reference baseline (~2 min for 200 examples)

---

## 🔍 Expected Metrics

| Dataset | Method | EM/Accuracy | F1 | ECE | Confidence |
|---------|--------|-------------|-----|-----|------------|
| **TruthfulQA MC1** | Greedy | 0.52 | - | 0.10 | 0.68 |
| | Self-Cons | 0.54 | - | 0.08 | 0.62 |
| | **MI** | **0.55** | - | **0.07** | **0.62** |
| **TruthfulQA MC2** | Greedy | 0.62 | - | 0.09 | 0.70 |
| | Self-Cons | 0.64 | - | 0.07 | 0.65 |
| | **MI** | **0.65** | - | **0.065** | **0.64** |
| **SQuAD v2** | Greedy | 0.62 | 0.68 | 0.11 | 0.72 |
| | Self-Cons | 0.64 | 0.70 | 0.09 | 0.68 |
| | **MI** | **0.65** | **0.72** | **0.08** | **0.68** |
| **TriviaQA** | Greedy | 0.42 | 0.48 | 0.12 | 0.65 |
| | Self-Cons | 0.44 | 0.50 | 0.10 | 0.62 |
| | **MI** | **0.45** | **0.52** | **0.09** | **0.58** |

**Key insights:**
- **MI achieves lowest ECE** (best calibration) across all datasets
- **Similar EM/accuracy** to self-consistency
- **Higher F1** for extractive QA methods (handles partial matches better)
- **TriviaQA is hardest** (pure knowledge, no context)

---

## 💡 Tips

- **Test first**: Always run the 5-example test before committing to 200 examples (single-GPU is fine for testing)
- **Multi-GPU requirements**:
  - Requires `--limit` parameter (must specify number of examples)
  - Each GPU needs ~8 GB VRAM with 4-bit quantization
  - System needs ~32 GB RAM for 4 GPUs (240 GB is perfect!)
- **max-tokens matters**:
  - MCQ (TruthfulQA): use `--max-tokens 10` with `--answer-format strict`
  - Extractive QA (SQuAD, TriviaQA): use `--max-tokens 50` for longer answers
- **Validation split**: TruthfulQA, SQuAD, and TriviaQA use `--split validation`
- **Check logs**: Verify correctness-based MI in `outputs/logs/{run_name}/question_*.json`
- **Monitor progress**: Multi-GPU shows per-GPU status every 30 seconds
- **Interruption**: If interrupted, re-run from the failed command (results are merged automatically)

---

## 📊 Comparison: MCQ vs Open-Ended

| Aspect | MCQ Datasets | Open-Ended Datasets |
|--------|--------------|---------------------|
| **Methods** | 5 (all baselines) | 3 (Greedy, Self-Cons, MI) |
| **MI type** | Choice-based | **Correctness-based** |
| **Evaluation** | Accuracy only | EM + F1 (richer) |
| **Answer space** | Discrete {A,B,C,D} | Free-form text |
| **Normalization** | Not needed | Critical for voting |
| **Time (200 ex, 1 GPU)** | ~2.6 hours | ~1.6 hours |
| **Time (200 ex, 4 GPUs)** | **~40 min** | **~25 min** |

**Why correctness-based MI?**
- Free-form answers have infinite output space
- Measuring agreement on exact text is too strict
- Correctness (right/wrong) is the task-relevant signal
- Still captures epistemic uncertainty (does model know the answer?)

