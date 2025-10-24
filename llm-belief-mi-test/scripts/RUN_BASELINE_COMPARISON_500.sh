#!/bin/bash
# Complete Baseline Comparison - 500 examples per dataset
# Total time: ~12 hours for all 9 runs (3 datasets × 3 methods)
# 
# Usage:
#   bash RUN_BASELINE_COMPARISON_500.sh
#   or run commands individually

set -e  # Exit on error

echo "======================================================================"
echo "BASELINE COMPARISON - 500 examples per dataset"
echo "======================================================================"
echo ""
echo "This will run 9 evaluations:"
echo "  - 3 datasets (ARC-Challenge, ARC-Easy, OpenBookQA)"
echo "  - 3 methods (Greedy, Self-Consistency, MI)"
echo "  - 500 examples each"
echo ""
echo "Total time: ~12 hours"
echo "======================================================================"
echo ""

# Get the project root (parent of scripts directory)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "Running from: $PROJECT_ROOT"
echo ""

# ======================================================================
# ARC-CHALLENGE (500 examples)
# ======================================================================

echo ""
echo "======================================================================"
echo "ARC-CHALLENGE - 500 examples (~6.5 hours total)"
echo "======================================================================"
echo ""

# Greedy baseline (~15 min)
echo "[1/9] Running ARC-Challenge Greedy baseline (~15 min)..."
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge --limit 500 \
  --load-in-4bit \
  --max-tokens 30 \
  --output outputs/results/arc_challenge_greedy_500.csv

# Self-consistency baseline (~3 hours)
echo "[2/9] Running ARC-Challenge Self-Consistency baseline (~3 hours)..."
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 30 \
  --output outputs/results/arc_challenge_selfcons_500.csv

# MI method (~3.5 hours)
echo "[3/9] Running ARC-Challenge MI method (~3.5 hours)..."
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/arc_challenge_mi_500.csv

echo ""
echo "✅ ARC-Challenge complete! Comparing results..."
python scripts/compare_results.py outputs/results/arc_challenge_*_500.json

# ======================================================================
# ARC-EASY (500 examples)
# ======================================================================

echo ""
echo "======================================================================"
echo "ARC-EASY - 500 examples (~6.5 hours total)"
echo "======================================================================"
echo ""

# Greedy baseline (~15 min)
echo "[4/9] Running ARC-Easy Greedy baseline (~15 min)..."
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-easy --limit 500 \
  --load-in-4bit \
  --max-tokens 30 \
  --output outputs/results/arc_easy_greedy_500.csv

# Self-consistency baseline (~3 hours)
echo "[5/9] Running ARC-Easy Self-Consistency baseline (~3 hours)..."
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-easy --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 30 \
  --output outputs/results/arc_easy_selfcons_500.csv

# MI method (~3.5 hours)
echo "[6/9] Running ARC-Easy MI method (~3.5 hours)..."
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 500 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/arc_easy_mi_500.csv

echo ""
echo "✅ ARC-Easy complete! Comparing results..."
python scripts/compare_results.py outputs/results/arc_easy_*_500.json

# ======================================================================
# OPENBOOKQA (500 examples - full dataset)
# ======================================================================

echo ""
echo "======================================================================"
echo "OPENBOOKQA - 500 examples (~6.5 hours total)"
echo "======================================================================"
echo ""

# Greedy baseline (~15 min)
echo "[7/9] Running OpenBookQA Greedy baseline (~15 min)..."
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa \
  --load-in-4bit \
  --max-tokens 30 \
  --output outputs/results/openbookqa_greedy_500.csv

# Self-consistency baseline (~3 hours)
echo "[8/9] Running OpenBookQA Self-Consistency baseline (~3 hours)..."
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset openbookqa \
  --k 10 --temperature 0.9 \
  --load-in-4bit \
  --max-tokens 30 \
  --output outputs/results/openbookqa_selfcons_500.csv

# MI method (~3.5 hours)
echo "[9/9] Running OpenBookQA MI method (~3.5 hours)..."
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/openbookqa_mi_500.csv

echo ""
echo "✅ OpenBookQA complete! Comparing results..."
python scripts/compare_results.py outputs/results/openbookqa_*_500.json

# ======================================================================
# FINAL SUMMARY
# ======================================================================

echo ""
echo "======================================================================"
echo "✅ ALL EVALUATIONS COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to outputs/results/*_500.csv and *.json"
echo ""
echo "Individual dataset comparisons:"
echo "  - ARC-Challenge: compare_results.py outputs/results/arc_challenge_*_500.json"
echo "  - ARC-Easy:      compare_results.py outputs/results/arc_easy_*_500.json"
echo "  - OpenBookQA:    compare_results.py outputs/results/openbookqa_*_500.json"
echo ""
echo "View all results:"
echo "  ls -lh outputs/results/*_500.*"
echo ""
echo "======================================================================"

