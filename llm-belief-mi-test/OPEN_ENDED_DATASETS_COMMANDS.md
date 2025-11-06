# Open-Ended Datasets - Evaluation Commands

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

---

## 🧪 Phase 1: Testing Commands (Sanity Check)

Run these **before** full evaluation to verify everything works correctly.

### Quick Test: 5 Examples Per Dataset (~10 minutes total)

```bash
# TruthfulQA MC1 (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc1 --split validation --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/test/truthfulqa_mc1_test.csv

# TruthfulQA MC2 (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc2 --split validation --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/test/truthfulqa_mc2_test.csv

# SQuAD v2 (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset squad-v2 --split validation --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit --temperature 0.9 \
  --max-tokens 50 \
  --output outputs/test/squad_v2_test.csv

# TriviaQA (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa --split validation --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit --temperature 0.9 \
  --max-tokens 50 \
  --output outputs/test/triviaqa_test.csv
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

## 📊 Phase 2: Full Evaluation (500 Examples Per Dataset)

### TruthfulQA MC1 (500 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset truthfulqa-mc1 --split validation --limit 500 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset truthfulqa-mc1 --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/selfcons_500.csv

# 3. MI method (~2.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc1 --split validation --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa/mi_500.csv
```

---

### TruthfulQA MC2 (500 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset truthfulqa-mc2 --split validation --limit 500 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset truthfulqa-mc2 --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/selfcons_500.csv

# 3. MI method (~2.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset truthfulqa-mc2 --split validation --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/truthfulqa_mc2/mi_500.csv
```

---

### SQuAD v2 (500 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset squad-v2 --split validation --limit 500 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/squad_v2/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset squad-v2 --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/squad_v2/selfcons_500.csv

# 3. MI method (~2 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset squad-v2 --split validation --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/squad_v2/mi_500.csv
```

---

### TriviaQA (500 examples)

**Individual Methods:**

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset triviaqa --split validation --limit 500 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/triviaqa/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset triviaqa --split validation --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/triviaqa/selfcons_500.csv

# 3. MI method (~2.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset triviaqa --split validation --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --output outputs/results/triviaqa/mi_500.csv
```

---

## 🚀 Phase 3: Run All Datasets & Methods (One Command)

**Total time: ~16 hours for all 4 datasets × 3 methods**

This command runs all 12 evaluations (4 datasets × 3 methods) sequentially:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
\
# ========== TruthfulQA MC1 (500 examples, ~4 hours) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset truthfulqa-mc1 --split validation --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/greedy_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc1_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset truthfulqa-mc1 --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/selfcons_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc1_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset truthfulqa-mc1 --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa/mi_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc1_mi_500.log && \
\
# ========== TruthfulQA MC2 (500 examples, ~4 hours) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset truthfulqa-mc2 --split validation --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/greedy_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc2_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset truthfulqa-mc2 --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/selfcons_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc2_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset truthfulqa-mc2 --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/truthfulqa_mc2/mi_500.csv 2>&1 | tee outputs/logs/truthfulqa_mc2_mi_500.log && \
\
# ========== SQuAD v2 (500 examples, ~3.5 hours) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset squad-v2 --split validation --limit 500 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/greedy_500.csv 2>&1 | tee outputs/logs/squad_v2_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset squad-v2 --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/selfcons_500.csv 2>&1 | tee outputs/logs/squad_v2_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset squad-v2 --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/squad_v2/mi_500.csv 2>&1 | tee outputs/logs/squad_v2_mi_500.log && \
\
# ========== TriviaQA (500 examples, ~4 hours) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset triviaqa --split validation --limit 500 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/greedy_500.csv 2>&1 | tee outputs/logs/triviaqa_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset triviaqa --split validation --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/selfcons_500.csv 2>&1 | tee outputs/logs/triviaqa_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset triviaqa --split validation --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 50 --output outputs/results/triviaqa/mi_500.csv 2>&1 | tee outputs/logs/triviaqa_mi_500.log
```

**Expected results summary:**
- Total: 12 evaluations (4 datasets × 3 methods)
- Total time: ~15.5 hours
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
- Fast reference baseline (~5 min for 500 examples)

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

- **Test first**: Always run the 5-example test before committing to 500 examples
- **max-tokens matters**:
  - MCQ (TruthfulQA): use `--max-tokens 10` with `--answer-format strict`
  - Extractive QA (SQuAD, TriviaQA): use `--max-tokens 50` for longer answers
- **Validation split**: TruthfulQA, SQuAD, and TriviaQA use `--split validation`
- **Check logs**: Verify correctness-based MI in `outputs/logs/{run_name}/question_*.json`
- **Monitor progress**: Use `tee` to save logs while watching progress
- **Interruption**: If interrupted, re-run from the failed command

---

## 📊 Comparison: MCQ vs Open-Ended

| Aspect | MCQ Datasets | Open-Ended Datasets |
|--------|--------------|---------------------|
| **Methods** | 5 (all baselines) | 3 (Greedy, Self-Cons, MI) |
| **MI type** | Choice-based | **Correctness-based** |
| **Evaluation** | Accuracy only | EM + F1 (richer) |
| **Answer space** | Discrete {A,B,C,D} | Free-form text |
| **Normalization** | Not needed | Critical for voting |
| **Time (500 ex)** | ~6.5 hours | ~4 hours |

**Why correctness-based MI?**
- Free-form answers have infinite output space
- Measuring agreement on exact text is too strict
- Correctness (right/wrong) is the task-relevant signal
- Still captures epistemic uncertainty (does model know the answer?)

