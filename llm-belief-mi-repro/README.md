# llm-belief-mi-repro

Reproduction scaffold for the paper "To Believe or Not to Believe Your LLM" (DeepMind, 2024). It implements the iterative prompting procedure and a simple finite-sample mutual information (MI) estimator to flag likely hallucinations based on epistemic uncertainty.

## What this repo currently supports

- Iterative prompting chains with length `t`, repeated `K` times per question
- Two MI estimators
  - `plugin`: Σ H(Y_i) − H(Y_1..Y_t) via empirical counts
  - `listing`: paper-inspired Algorithm 1 from `listing.tex` with γ-smoothing
- Baselines (per question)
  - Agreement (self-consistency fraction across K finals)
  - Entropy of final answers (in bits)
  - Optional: Greedy logprob (requires backend token logprobs)
  - Optional: Self-verification confidence (one verify call per chain)
- Calibration/evaluation on a validation split; prints Accuracy/ROC AUC on test
- CSV outputs with per-question scores; ROC plotting utility with optional save
- TQDM progress bars for questions and total API calls

## Prerequisites

- Python 3.10+
- LM Studio app installed (for a local OpenAI-compatible API): `http://localhost:1234/v1`
- A local model available in LM Studio, e.g., `Llama-3.1-8B-Instruct`
- Optional: `datasets` (to download TriviaQA), `matplotlib` (to save/show plots)

## Quickstart (LM Studio)

```bash
cd llm-belief-mi-repro
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ensure LM Studio server is running locally
# Example (after installing LM Studio CLI):
# lms server start
# lms load "Llama-3.1-8B-Instruct" --gpu=1.0 --identifier llama31_8b

# Run a toy experiment (20 queries, chain length t=3) against the local API
python -m llm_belief_mi_repro.cli run \
  --toy \
  --n 20 \
  --t 3 \
  --model llama31_8b \
  --base-url http://localhost:1234/v1 \
  --output results_toy.csv
```

The script prints the estimated MI of the response chain (nats and bits). The output CSV includes each query with the chain of responses (`y1..yt`).

## Notes

-- Base URL and API key can be overridden with `--base-url` and `--api-key`, or via env vars `LLM_API_BASE` and `LLM_API_KEY`.
-- Request count for dataset runs is roughly `#examples × K × t` plus any enabled baselines (see below).

## LM Studio setup details

1. Install and launch LM Studio once to initialize.
2. Bootstrap the CLI (`lms`), then start the local server:
   ```bash
   lms server start
   ```
3. Download an 8B model (choose quantization if your GPU is constrained):
   ```bash
   lms get llama-3.1-8b
   ```
4. Load the instruct model fully on GPU and assign an identifier:
   ```bash
   lms load "Llama-3.1-8B-Instruct" --gpu=1.0 --identifier llama31_8b
   ```
5. Sanity check via curl:
   ```bash
   curl http://localhost:1234/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"llama31_8b","messages":[{"role":"user","content":"Hello!"}]}'
   ```

## Troubleshooting

- If the server is not reachable, confirm `lms server status` and that the port matches `--base-url`.
- If responses are slow or OOM occurs, try a smaller quant (e.g., 4–6 bit) when running `lms get`.
- If client complains about API key, provide any dummy key via `--api-key` (LM Studio typically doesn’t require one locally).

## Docs

- Condensed summaries of the two PDFs: see `docs/paper_summary.md`.


## Dataset: TriviaQA validation subset (optional helper)

Download a small validation subset to CSV using Hugging Face Datasets:

```bash
pip install datasets
python -m llm_belief_mi_repro.scripts.download_triviaqa_subset \
  --output triviaqa_val_subset.csv \
  --n 200 \
  --config rc
```

The CSV has columns: `question`, `answers` (pipe-separated aliases). You can also supply your own CSV or JSONL; see `llm_belief_mi_repro/datasets.py` for the expected fields.

## Run per-question MI on a subset

```bash
python -m llm_belief_mi_repro.cli run_dataset \
  --input triviaqa_val_subset.csv \
  --limit 10 --k 20 --t 3 --mi listing \
  --model meta-llama-3.1-8b-instruct --base-url http://127.0.0.1:1234/v1 \
  --temperature 0.5 --max-tokens 64 \
  --output results_triviaqa_10_k20_t3_listing.csv
```

During the run you will see two progress bars: one for questions, one for total API calls.

### Default baselines (no extra API calls)

- Agreement: fraction of identical final answers across the K chains (self-consistency).
- Semantic entropy: entropy of the final answers across chains (reported in bits as `entropy_bits`).

### Optional baselines

- `--baseline-greedy`: greedy decode with token logprobs (one extra call per question). Requires backend support for logprobs; values are omitted if unavailable.
- `--baseline-verify`: self-verification confidence (one extra call per chain). Prompts the model to output a numeric 0–1 confidence for each final answer and averages the scores.

Example with baselines:

```bash
python -m llm_belief_mi_repro.cli run_dataset \
  --input triviaqa_val_subset.csv \
  --limit 10 --k 20 --t 3 --mi listing \
  --baseline-greedy --baseline-verify \
  --model meta-llama-3.1-8b-instruct --base-url http://127.0.0.1:1234/v1 \
  --temperature 0.5 --max-tokens 64 \
  --output results_triviaqa_10_k20_t3_listing_baselines.csv
```

### Outputs and metrics

- Per-question CSV includes: `mi_bits`, `agreement`, `entropy_bits`, and (if enabled) `greedy_logprob`, `verify_score`, plus run settings.
- After the run, the script prints test metrics for MI and baselines (accuracy, precision/recall/F1, ROC AUC) with thresholds chosen on a validation split.

## Plot ROC (and save files)

```bash
pip install matplotlib  # if not already installed
python -m llm_belief_mi_repro.cli plot_roc \
  --input results_triviaqa_10_k20_t3_listing.csv \
  --score-col mi_bits \
  --save plots/roc_mi_bits_listing.png
```

Use `--score-col agreement` or `--score-col entropy_bits` to plot baselines. The plot will be saved if `--save` is provided and also displayed on screen.


