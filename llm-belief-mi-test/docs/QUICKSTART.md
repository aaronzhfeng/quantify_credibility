# Quick Start Guide

## Prerequisites
- GPU with 12+ GB VRAM (or use 4-bit quantization)
- HuggingFace account with Llama access
- Python 3.10+

## Setup (5 minutes)

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your HuggingFace token (REPLACE WITH YOUR TOKEN)
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# 3. Test GPU setup
python test_gpu_setup.py
```

## Run Evaluation

### Quick Test (5 examples, ~3 minutes)
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/test.csv
```

### Full Benchmark with Optimized Settings (~7-14 hours each)
```bash
# ARC-Challenge (1,172 examples, ~7 hours)
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/arc_challenge.csv

# ARC-Easy (2,376 examples, ~14 hours)
python -m llm_belief_mi_test.cli \
  --dataset arc-easy \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/arc_easy.csv

# OpenBookQA (500 examples, ~3 hours)
python -m llm_belief_mi_test.cli \
  --dataset openbookqa \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/openbookqa.csv
```

**Total: ~24 hours | A100: ~$65 | L4: ~$12**

**Parameters from paper**: k=10, n=2, temperature=0.9

## View Results

```bash
# CSV output (per question)
cat outputs/results/test.csv

# JSON metrics (summary)
cat outputs/results/test.json
```

Key metrics:
- **accuracy**: Task performance
- **ece**: Calibration error (lower = better) ← YOUR KEY METRIC
- **avg_mi_bits**: Average mutual information
- **avg_confidence**: Average confidence score

## Troubleshooting

```bash
# If "no GPU found":
nvidia-smi  # Check GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If "authentication error":
export HF_TOKEN="your_token_here"
# Or: huggingface-cli login

# If "out of memory":
# Add --load-in-4bit flag (already in examples above)
```

## Documentation

- **WHAT_WAS_IMPLEMENTED.md**: What I built for you
- **IMPLEMENTATION_COMPLETE.md**: Detailed usage guide
- **AUTHENTICATION_GUIDE.md**: HuggingFace setup

---

**That's it! You're ready to run evaluations.** 🚀
