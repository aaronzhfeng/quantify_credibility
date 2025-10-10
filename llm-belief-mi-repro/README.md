# llm-belief-mi-repro

Reproduction scaffold for the paper "To Believe or Not to Believe Your LLM" (DeepMind, 2024). It implements the iterative prompting procedure and a simple finite-sample mutual information (MI) estimator to flag likely hallucinations based on epistemic uncertainty.

## Prerequisites

- Python 3.10+
- LM Studio app installed (for a local OpenAI-compatible API): `http://localhost:1234/v1`
- A local model available in LM Studio, e.g., `Llama-3.1-8B-Instruct`

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
-- This scaffold focuses on the paper's core idea: iterative prompting and MI-based detection. You can add calibration and evaluation pipelines (e.g., TriviaQA/AmbigQA) as needed.

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


