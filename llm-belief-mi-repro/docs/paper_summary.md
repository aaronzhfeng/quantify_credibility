# Paper Summaries

## Implementing Llama 3.1 8B Locally — Condensed

- Hardware: ~16 GB VRAM for bf16; use 4–8 bit quant if VRAM is tight.
- LM Studio route (recommended for simplicity):
  - Install app, run once, bootstrap the `lms` CLI.
  - Download model: `lms get llama-3.1-8b[@quant]` (choose a quantization if needed).
  - Start server: `lms server start` (OpenAI-compatible API at `http://localhost:1234/v1`).
  - Load model: `lms load "Llama-3.1-8B-Instruct" --gpu=1.0 --identifier llama31_8b`.
  - Test with curl/OpenAI SDKs by setting `api_base=http://localhost:1234/v1`.
- Alternative: Hugging Face Transformers path to load 8B directly in Python (PyTorch CUDA must match drivers).
- Mini-reproduction: Use the API to run small slices of benchmarks (MMLU, CommonSenseQA, ARC) and compute accuracy.

## To Believe or Not to Believe Your LLM (DeepMind, 2024) — Condensed

- Goal: Detect unreliability driven by high epistemic uncertainty (vs. aleatoric uncertainty when multiple answers exist).
- Method: Iterative prompting forms an LLM-derived joint distribution over multiple responses (conditioning each response on prior ones). Define an information-theoretic metric and a computable lower bound via mutual information (MI) over the response chain.
- Key idea: If new responses strongly depend on prior ones, MI rises → high epistemic uncertainty; if independent, MI stays low (aligns with ground-truth product distribution when many answers are valid).
- Experiments: On closed-book QA (TriviaQA, AmbigQA, WordNet-synth), MI-thresholding outperforms likelihood and often entropy, especially on datasets mixing single- and multi-label samples.
- Mechanistic sketch: If the query weakly overlaps principal components of the key-query product, repeated context items can dominate generation (copying effect), explaining probability amplification.

## How this repo maps to the paper

- Iterative prompting: The CLI creates a chain of responses by appending prior answers to the next prompt.
- MI estimator: A plug-in estimate of I(Y1..Yt) ≈ ∑ H(Yi) − H(Y1..Yt) over observed strings acts as the computable lower-bound proxy.
- Local API: Works directly with LM Studio’s OpenAI-compatible server, keeping setup predictable and lightweight.
