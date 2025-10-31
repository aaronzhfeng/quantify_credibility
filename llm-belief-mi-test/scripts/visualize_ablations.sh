#!/bin/bash
# Generate all visualizations for ablation study results
#
# Usage: bash scripts/visualize_ablations.sh

set -e

echo "========================================================================"
echo "GENERATING ABLATION STUDY VISUALIZATIONS"
echo "========================================================================"
echo ""

cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Create output directory
mkdir -p outputs/plots/ablation

echo "[1/7] Plotting temperature ablation..."
python scripts/plot_ablation.py --parameter temperature

echo ""
echo "[2/7] Plotting k-chains ablation..."
python scripts/plot_ablation.py --parameter k_chains

echo ""
echo "[3/7] Plotting chain-length ablation..."
python scripts/plot_ablation.py --parameter n_length

echo ""
echo "[4/7] Plotting MI estimator ablation..."
python scripts/plot_ablation.py --parameter mi_method

echo ""
echo "[5/7] Plotting confidence method ablation..."
python scripts/plot_ablation.py --parameter confidence_method

echo ""
echo "[6/7] Plotting answer format ablation..."
python scripts/plot_ablation.py --parameter answer_format

echo ""
echo "[7/7] Creating combined ablation plot..."
python scripts/plot_ablation.py --combined

echo ""
echo "========================================================================"
echo "✓ ALL ABLATION VISUALIZATIONS COMPLETE"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - outputs/plots/ablation/ablation_temperature.png"
echo "  - outputs/plots/ablation/ablation_k_chains.png"
echo "  - outputs/plots/ablation/ablation_n_length.png"
echo "  - outputs/plots/ablation/ablation_mi_method.png"
echo "  - outputs/plots/ablation/ablation_confidence_method.png"
echo "  - outputs/plots/ablation/ablation_answer_format.png"
echo "  - outputs/plots/ablation/ablation_combined.png"
echo ""
echo "View plots:"
echo "  ls -lh outputs/plots/ablation/"
echo ""
echo "Quick view combined plot:"
echo "  python scripts/plot_ablation.py --combined"
echo ""

