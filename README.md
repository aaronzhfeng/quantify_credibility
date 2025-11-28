# Quantify Credibility: LLM Uncertainty Quantification

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

A comprehensive research toolkit for **quantifying uncertainty and credibility in Large Language Model outputs** using Mutual Information (MI) estimation, iterative prompting, and NLI-based semantic clustering. Implements methods from *"To Believe or Not to Believe Your LLM"* (DeepMind, 2024).

## 🎯 Key Features

- **MI-Based Uncertainty Quantification**: Measure epistemic uncertainty through mutual information between iterative prompts
- **Expected Calibration Error (ECE)**: Evaluate how well model confidence aligns with actual accuracy
- **Multiple Baseline Methods**: Greedy, self-consistency, semantic-entropy, self-verification
- **NLI Semantic Clustering**: Group semantically equivalent answers using DeBERTa-MNLI
- **Multi-Benchmark Support**: ARC, OpenBookQA, TruthfulQA, SQuAD v2, TriviaQA

## 📁 Repository Structure

```
quantify_credibility/
├── llm-belief-mi-test/            # 🧪 Main MI evaluation framework
│   ├── llm_belief_mi_test/        # Core Python package
│   │   ├── cli.py                 # Command-line interface
│   │   ├── calibration.py         # ECE & evaluation (with baselines)
│   │   ├── mi_estimator.py        # MI computation (listing/plugin)
│   │   ├── iterative_prompting.py # Iterative prompting chains
│   │   ├── llm_client_local.py    # Local Llama client
│   │   ├── datasets.py            # Dataset loaders
│   │   └── cache.py               # SQLite caching
│   ├── scripts/                   # Utility scripts
│   │   ├── test_gpu_setup.py      # GPU verification
│   │   ├── compare_results.py     # Results comparison
│   │   └── plot_results.py        # Visualization
│   ├── outputs/                   # Results & logs
│   └── docs/                      # Detailed documentation
│
├── llm-belief-mi-repro/           # 📊 Paper reproduction experiments
│   ├── llm_belief_mi_repro/       # Reproduction code
│   └── outputs/                   # Reproduction results
│
├── nli-semantic-clustering/       # 🔍 NLI clustering module
│   ├── nli_clustering/            # Core NLI package
│   │   ├── core.py               # DeBERTa-MNLI model & clustering
│   │   └── utils.py              # Evaluation metrics
│   ├── scripts/
│   │   └── threshold_sweep.py    # Threshold optimization
│   └── data/                     # Sample data
│
└── theory/                        # 📚 Theoretical foundations
    ├── MI_ALGORITHMS.md          # MI estimator algorithms
    ├── MI_ECE_FORMULAS.md        # Mathematical formulas
    └── MI_ESTIMATOR_EXAMPLE.md   # Worked examples
```

## 🚀 Quick Start

### Installation

```bash
cd quantify_credibility/llm-belief-mi-test
pip install -r requirements.txt

# Set HuggingFace token for Llama models
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

### Verify Setup

```bash
python scripts/test_gpu_setup.py
```

### Run Quick Test

```bash
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_quick.csv
```

## 📊 Evaluation Methods

### 1. MI Method (Paper's Main Contribution)
Uses iterative prompting to estimate mutual information between response chains.

```bash
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/mi_500.csv
```

### 2. Baseline Methods

```bash
# Greedy (single decode)
python -m llm_belief_mi_test.cli --method greedy --dataset arc-easy ...

# Self-Consistency (k samples + majority vote)
python -m llm_belief_mi_test.cli --method self-consistency --k 10 ...

# Semantic Entropy (F1 clustering + entropy)
python -m llm_belief_mi_test.cli --method semantic-entropy --k 10 ...

# Self-Verification (samples + verification query)
python -m llm_belief_mi_test.cli --method self-verification --k 10 ...
```

## 🔬 Supported Datasets

| Dataset | Type | Size | Key Metric |
|---------|------|------|------------|
| **ARC-Challenge** | MCQ | 1,172 | Accuracy |
| **ARC-Easy** | MCQ | 2,376 | Accuracy |
| **OpenBookQA** | MCQ | 500 | Accuracy |
| **TruthfulQA MC1** | MCQ | 817 | Accuracy |
| **TruthfulQA MC2** | MCQ (multi-true) | 817 | Accuracy |
| **SQuAD v2** | Extractive QA | 11,873 | EM/F1 |
| **TriviaQA** | Open-domain QA | 87,622 | EM/F1 |

## 📈 Key Metrics

### Expected Calibration Error (ECE)
Measures how well model confidence correlates with actual accuracy:
```
ECE = Σ (n_b / N) |accuracy_b - confidence_b|
```
**Lower ECE = Better Calibration** (main contribution of MI method)

### Mutual Information (MI)
Captures epistemic uncertainty through iterative prompting:
```
MI(Y₁; Y₂; ...; Yₙ) = Σᵢ H(Yᵢ) - H(Y₁,...,Yₙ)
```
**Higher MI = More Uncertainty**

## 🎛️ Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--k` | Number of independent chains | 10 |
| `--n` | Chain length (pseudo-joint dimension) | 2 |
| `--temperature` | Sampling temperature | 0.9 |
| `--mi-method` | MI estimator (`listing`/`plugin`) | listing |
| `--confidence-method` | MI→confidence mapping | inverse |
| `--answer-format` | Output format (`strict`/`codeblock`) | strict |

## 📊 Expected Results

**Key Insight**: MI method is designed for **better calibration** (lower ECE), not necessarily higher accuracy.

| Method | ARC-C Acc | ARC-C ECE | ARC-E Acc | ARC-E ECE |
|--------|-----------|-----------|-----------|-----------|
| Greedy | ~65% | ~0.12 | ~80% | ~0.08 |
| Self-Consistency | ~67% | ~0.10 | ~82% | ~0.06 |
| **MI Method** | ~66% | **~0.05** | ~81% | **~0.04** |

## 🔍 NLI Semantic Clustering

The NLI module groups semantically equivalent answers to reduce spurious uncertainty:

```bash
cd nli-semantic-clustering
python scripts/threshold_sweep.py \
  --log-dir ../llm-belief-mi-test/outputs/logs/triviaqa_mi_200 \
  --thresholds 0.3 0.4 0.5 0.6 0.7 \
  --correctness-based \
  --output results/threshold_sweep.json
```

### Dual-Mode System
- **Clustering**: Strict bidirectional entailment (A ↔ B)
- **Grading**: Loose unidirectional (A → B) + substring matching

## 📚 Theoretical Background

### MI Estimation Algorithms

**Plugin Estimator** (simple):
```
MI = Σᵢ H(Yᵢ) - H(Y₁,...,Yₙ)
H(X) = -Σ p(x) log p(x)
```

**Listing Estimator** (paper's Algorithm 1):
```
MI = Σ μ̂ · log((μ̂ + γ₁) / (μ̂_prod + γ₂))
γ₁, γ₂ = 1/k  (regularization)
```

See `theory/MI_ALGORITHMS.md` for detailed explanations.

## ⚙️ Hardware Requirements

### Minimum (4-bit quantization)
- GPU: 12GB VRAM (RTX 3060, RTX 4060 Ti)
- RAM: 16GB

### Recommended
- GPU: 16GB+ VRAM (RTX 4080, A4000+)
- RAM: 32GB

### Time Estimates (k=10, n=2)
| Dataset | Examples | Time (A100) |
|---------|----------|-------------|
| ARC-Challenge | 1,172 | ~3-4 hours |
| ARC-Easy | 2,376 | ~6-7 hours |
| OpenBookQA | 500 | ~1.5 hours |

## 📖 Documentation

- **[llm-belief-mi-test/README.md](llm-belief-mi-test/README.md)** - Full evaluation guide
- **[theory/MI_ALGORITHMS.md](theory/MI_ALGORITHMS.md)** - MI estimator details
- **[theory/MI_ECE_FORMULAS.md](theory/MI_ECE_FORMULAS.md)** - Mathematical formulas
- **[nli-semantic-clustering/README.md](nli-semantic-clustering/README.md)** - NLI clustering guide

## 📚 References

- **Paper**: "To Believe or Not to Believe Your LLM" (DeepMind, 2024) - [arXiv:2406.02543](https://arxiv.org/abs/2406.02543)
- **Llama 3.1**: [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **DeBERTa-MNLI**: [microsoft/deberta-v2-xlarge-mnli](https://huggingface.co/microsoft/deberta-v2-xlarge-mnli)

### Datasets
- [ARC](https://allenai.org/data/arc) - AI2 Reasoning Challenge
- [OpenBookQA](https://allenai.org/data/open-book-qa) - Open-domain QA
- [TruthfulQA](https://huggingface.co/datasets/truthful_qa) - Truthfulness evaluation
- [SQuAD v2](https://huggingface.co/datasets/rajpurkar/squad_v2) - Reading comprehension
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) - Trivia QA

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

*Research toolkit for quantifying LLM uncertainty and calibration.*
