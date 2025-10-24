#!/bin/bash
# Generate all visualizations for 500-example results
#
# Usage: bash scripts/visualize_all.sh

set -e

echo "========================================================================"
echo "GENERATING ALL VISUALIZATIONS"
echo "========================================================================"
echo ""

cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Create output directory
mkdir -p outputs/plots

echo "[1/3] Generating summary table..."
python scripts/summarize_results.py

echo ""
echo "[2/3] Generating comparison plots..."
python scripts/plot_results.py --dataset all

echo ""
echo "[3/3] Generating calibration curves..."
python scripts/plot_calibration.py --dataset all

echo ""
echo "========================================================================"
echo "✓ ALL VISUALIZATIONS COMPLETE"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - outputs/plots/openbookqa_comparison.png"
echo "  - outputs/plots/arc_challenge_comparison.png"
echo "  - outputs/plots/arc_easy_comparison.png"
echo "  - outputs/plots/combined_comparison.png"
echo "  - outputs/plots/openbookqa_calibration.png"
echo "  - outputs/plots/arc_challenge_calibration.png"
echo "  - outputs/plots/arc_easy_calibration.png"
echo ""
echo "View plots:"
echo "  ls -lh outputs/plots/"
echo ""

