git# Multiple-Choice Datasets - Evaluation Commands

This guide contains all commands for evaluating **multiple-choice question (MCQ)** datasets using the MI-based uncertainty quantification framework.

**Datasets covered:**
- ARC-Challenge (1,172 test examples)
- ARC-Easy (2,376 test examples)
- OpenBookQA (500 test examples)

**Methods available:** All 5 methods
1. Greedy (temperature=0 baseline)
2. Self-Consistency (k samples + majority voting)
3. Semantic Entropy (k samples + semantic clustering + entropy)
4. Self-Verification (k samples + verification)
5. MI (k chains of length n + mutual information)

---

## 🧪 Phase 1: Testing Commands (Sanity Check)

Run these **before** full evaluation to verify everything works correctly.

### Quick Test: 5 Examples Per Dataset (~10 minutes total)

```bash
# ARC-Challenge (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/test/arc_challenge_test.csv

# ARC-Easy (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/test/arc_easy_test.csv

# OpenBookQA (5 examples, ~2 min)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/test/openbookqa_test.csv
```

**What to check:**
- ✅ All runs complete without errors
- ✅ Accuracy > 0.2 (not random guessing)
- ✅ MI score > 0 (chains have some diversity)
- ✅ Agreement < 1.0 (chains don't all agree)
- ✅ Logs saved to `outputs/logs/{dataset}_test/question_*.json`

---

## 📊 Phase 2: Full Evaluation (500 Examples Per Dataset)

### ARC-Challenge (500 examples)

#### Individual Methods:

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge --limit 500 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/selfcons_500.csv

# 3. MI method (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/mi_500.csv

# 4. Semantic Entropy (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/semantic_entropy_500.csv

# 5. Self-Verification (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge/self_verification_500.csv
```

---

### ARC-Easy (500 examples)

#### Individual Methods:

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 500 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/selfcons_500.csv

# 3. MI method (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/mi_500.csv

# 4. Semantic Entropy (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/semantic_entropy_500.csv

# 5. Self-Verification (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_easy/self_verification_500.csv
```

---

### OpenBookQA (500 examples - full dataset)

#### Individual Methods:

```bash
# 1. Greedy baseline (~5 min)
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/greedy_500.csv

# 2. Self-Consistency baseline (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/selfcons_500.csv

# 3. MI method (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/mi_500.csv

# 4. Semantic Entropy (~1.5 hours)
python -m llm_belief_mi_test.cli \
  --method semantic-entropy \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/semantic_entropy_500.csv

# 5. Self-Verification (~2 hours)
python -m llm_belief_mi_test.cli \
  --method self-verification \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/openbookqa/self_verification_500.csv
```

---

## 🚀 Phase 3: Run All Datasets & Methods (One Command)

**Total time: ~19.5 hours for all 3 datasets × 5 methods**

This command runs all 15 evaluations (3 datasets × 5 methods) sequentially:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test && \
\
# ========== ARC-Challenge (500 examples, ~6.5 hours) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset arc-challenge --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/greedy_500.csv 2>&1 | tee outputs/logs/arc_challenge_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset arc-challenge --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/selfcons_500.csv 2>&1 | tee outputs/logs/arc_challenge_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset arc-challenge --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/mi_500.csv 2>&1 | tee outputs/logs/arc_challenge_mi_500.log && \
python -m llm_belief_mi_test.cli --method semantic-entropy --dataset arc-challenge --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/semantic_entropy_500.csv 2>&1 | tee outputs/logs/arc_challenge_semantic_entropy_500.log && \
python -m llm_belief_mi_test.cli --method self-verification --dataset arc-challenge --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_challenge/self_verification_500.csv 2>&1 | tee outputs/logs/arc_challenge_self_verification_500.log && \
\
# ========== ARC-Easy (500 examples, ~6.5 hours) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset arc-easy --limit 500 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/greedy_500.csv 2>&1 | tee outputs/logs/arc_easy_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset arc-easy --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/selfcons_500.csv 2>&1 | tee outputs/logs/arc_easy_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset arc-easy --limit 500 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/mi_500.csv 2>&1 | tee outputs/logs/arc_easy_mi_500.log && \
python -m llm_belief_mi_test.cli --method semantic-entropy --dataset arc-easy --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/semantic_entropy_500.csv 2>&1 | tee outputs/logs/arc_easy_semantic_entropy_500.log && \
python -m llm_belief_mi_test.cli --method self-verification --dataset arc-easy --limit 500 --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/arc_easy/self_verification_500.csv 2>&1 | tee outputs/logs/arc_easy_self_verification_500.log && \
\
# ========== OpenBookQA (500 examples, ~6.5 hours) ==========
python -m llm_belief_mi_test.cli --method greedy --dataset openbookqa --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/greedy_500.csv 2>&1 | tee outputs/logs/openbookqa_greedy_500.log && \
python -m llm_belief_mi_test.cli --method self-consistency --dataset openbookqa --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/selfcons_500.csv 2>&1 | tee outputs/logs/openbookqa_selfcons_500.log && \
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/mi_500.csv 2>&1 | tee outputs/logs/openbookqa_mi_500.log && \
python -m llm_belief_mi_test.cli --method semantic-entropy --dataset openbookqa --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/semantic_entropy_500.csv 2>&1 | tee outputs/logs/openbookqa_semantic_entropy_500.log && \
python -m llm_belief_mi_test.cli --method self-verification --dataset openbookqa --k 10 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/openbookqa/self_verification_500.csv 2>&1 | tee outputs/logs/openbookqa_self_verification_500.log
```

**Expected results summary:**
- Total: 15 evaluations (3 datasets × 5 methods)
- Total time: ~19.5 hours
- Files created: 15 CSV files + 15 JSON files + detailed logs
- Expected ECE ranking: MI ≤ Semantic Entropy < Self-Consistency < Self-Verification < Greedy

---

## 📈 Compare Results

After running all evaluations:

```bash
# Compare all MCQ datasets
python scripts/compare_results.py outputs/results/arc_challenge/*.json
python scripts/compare_results.py outputs/results/arc_easy/*.json
python scripts/compare_results.py outputs/results/openbookqa/*.json

# Compare across all MCQ datasets and methods
python scripts/compare_results.py outputs/results/arc_*/*.json outputs/results/openbookqa/*.json

# Visualize
bash scripts/visualize_all.sh
```

---

## 💡 Tips

- **Test first**: Always run the 5-example test before committing to 500 examples
- **Check logs**: Verify detailed logs in `outputs/logs/{run_name}/question_*.json`
- **Monitor progress**: Use `tee` to save logs while watching progress
- **Interruption**: If interrupted, re-run from the failed command (cache won't help with temp=0.9, but previous outputs are saved)
- **Incremental**: Can run methods separately or use the combined command for overnight runs

---

## 🔍 Expected Metrics

| Dataset | Method | Accuracy | ECE | Confidence |
|---------|--------|----------|-----|------------|
| **ARC-Challenge** | Greedy | 0.55 | 0.12 | 0.70 |
| | Self-Cons | 0.57 | 0.10 | 0.65 |
| | Semantic Entropy | 0.57 | 0.09 | 0.62 |
| | Self-Verif | 0.56 | 0.11 | 0.64 |
| | **MI** | **0.57** | **0.08** | **0.63** |
| **ARC-Easy** | Greedy | 0.72 | 0.10 | 0.78 |
| | Self-Cons | 0.74 | 0.08 | 0.72 |
| | Semantic Entropy | 0.74 | 0.07 | 0.70 |
| | Self-Verif | 0.73 | 0.09 | 0.71 |
| | **MI** | **0.74** | **0.06** | **0.69** |
| **OpenBookQA** | Greedy | 0.65 | 0.11 | 0.72 |
| | Self-Cons | 0.67 | 0.09 | 0.68 |
| | Semantic Entropy | 0.67 | 0.08 | 0.66 |
| | Self-Verif | 0.66 | 0.10 | 0.67 |
| | **MI** | **0.67** | **0.07** | **0.65** |

**Key insight:** MI should achieve **lowest ECE** (best calibration) while maintaining similar accuracy to other methods.

