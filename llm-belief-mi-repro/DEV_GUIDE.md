# llm-belief-mi-repro — Developer Guide

This guide documents the implemented functionality, how it fits together, and how to test each capability with runnable commands.

The project reproduces the core idea from "To Believe or Not to Believe Your LLM" by forming iterative prompting chains and estimating mutual information (MI) as a proxy for epistemic uncertainty. It now includes async cloud inference, caching, logging, multiple datasets, prompt variants, baselines, plotting, and a YAML config runner.

---

## 1) Environment Setup

- Python 3.10+
- Install dependencies:

```bash
cd /Users/aaronfeng/Repo/quantify_credibility/llm-belief-mi-repro
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

- Providers (choose one and source the script):
  - HF Inference Endpoint (default): `source ./set_hf_env.sh endpoint` (uses `HF_ENDPOINT_URL`, defaulted to your endpoint; sets `PROVIDER=hf`)
  - HF Router + Fireworks: `source ./set_hf_env.sh router` (sets `PROVIDER=openai`)
  - Local LM Studio: `source ./set_hf_env.sh local` (sets `PROVIDER=openai`)

---

## 2) Output structure for tests

All test outputs go to `outputs/` to keep artifacts tidy:
```bash
mkdir -p outputs/plots outputs/logs outputs/results outputs/subsets
```

## 3) Prompting (naive) and MI estimators

Variants in `llm_belief_mi_repro/iterative_prompting.py`:
- `naive` (default), `wrong_prev`, `critique`

Sanity for MI estimators (no provider needed):
```bash
python - <<'PY'
from llm_belief_mi_repro.mi_estimator import estimate_mi_nats, estimate_mi_listing_nats, nats_to_bits
chains = [["London","London","London"],["London","Paris","London"],["Paris","Paris","Paris"],["Paris","London","Paris"]]
print("MI plugin bits:", nats_to_bits(estimate_mi_nats(chains)))
print("MI listing bits:", nats_to_bits(estimate_mi_listing_nats(chains)))
PY
```

---

---

---

## 4) Datasets and download helpers

Download small subsets:
```bash
# TriviaQA subset to CSV
python -m llm_belief_mi_repro.scripts.download_triviaqa_subset \
  --output outputs/subsets/triviaqa_val_subset.csv --n 200 --config rc

# AmbigQA subset to JSONL (optional; HF loader works without a file)
python -m llm_belief_mi_repro.scripts.download_ambigqa_subset \
  --output outputs/subsets/ambigqa_val_subset.csv --n 200 --split validation
```

---

## 5) Metrics (all four: MI, S.E., T0, S.V.) with async + logprobs

Runs below use the Hugging Face Inference Endpoint client. Ensure `source ./set_hf_env.sh endpoint` and use `--provider hf` (default also reads `PROVIDER` from env).

- TriviaQA (naive prompt, MI listing, all metrics) — HF Endpoint:
```bash
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input outputs/subsets/triviaqa_val_subset.csv \
  --limit 200 --k 10 --t 3 --prompt-style naive \
  --mi listing \
  --async --concurrency 509 --provider hf \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/triviaqa_200_k10_t3_naive_listing_metrics.csv \
  --log-dir outputs/logs
```

- AmbigQA (local CSV), naive prompt, all metrics — HF Endpoint:
```bash
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset ambigqa --input outputs/subsets/ambigqa_val_subset.csv \
  --limit 200 --k 10 --t 3 --prompt-style naive \
  --mi listing \
  --async --concurrency 50 --provider hf \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/ambigqa_200_k10_t3_naive_listing_metrics.csv \
  --log-dir outputs/logs
```

- Fireworks via HF Router (OpenAI-compatible; supports T0):
```bash
# Set router envs then run
source ./set_hf_env.sh router

python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input outputs/subsets/triviaqa_val_subset.csv \
  --limit 200 --k 10 --t 3 --prompt-style naive \
  --mi listing \
  --async --concurrency 50 --provider openai \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/triviaqa_200_k10_t3_naive_listing_metrics_fireworks.csv \
  --log-dir outputs/logs

python -m llm_belief_mi_repro.cli run_dataset \
  --dataset ambigqa --input outputs/subsets/ambigqa_val_subset.csv \
  --limit 200 --k 10 --t 3 --prompt-style naive \
  --mi listing \
  --async --concurrency 50 --provider openai \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/ambigqa_200_k10_t3_naive_listing_metrics_fireworks.csv \
  --log-dir outputs/logs
```

- Local LM Studio (OpenAI-compatible; T0 may be unavailable):
```bash
# Start LM Studio server locally, then:
source ./set_hf_env.sh local

python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input outputs/subsets/triviaqa_val_subset.csv \
  --limit 50 --k 5 --t 3 --prompt-style naive \
  --mi listing \
  --provider openai \
  --temperature 0.5 --max-tokens 64 \
  --baseline-verify \
  --output outputs/results/triviaqa_50_k5_t3_naive_listing_metrics_local.csv \
  --log-dir outputs/logs

python -m llm_belief_mi_repro.cli run_dataset \
  --dataset ambigqa --input outputs/subsets/ambigqa_val_subset.csv \
  --limit 50 --k 5 --t 3 --prompt-style naive \
  --mi listing \
  --provider openai \
  --temperature 0.5 --max-tokens 64 \
  --baseline-verify \
  --output outputs/results/ambigqa_50_k5_t3_naive_listing_metrics_local.csv \
  --log-dir outputs/logs
```

Optional: switch label policy (default `any` vs `majority`): add `--label-policy majority`.

---

## 6) Prompt variants quick checks

Change only `--prompt-style`:
```bash
# Wrong-prev on TriviaQA — HF Endpoint
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input outputs/subsets/triviaqa_val_subset.csv \
  --limit 100 --k 10 --t 3 --prompt-style wrong_prev \
  --mi listing --async --concurrency 50 --provider hf \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/triviaqa_100_k10_t3_wrongprev.csv --log-dir outputs/logs

# Critique on AmbigQA — HF Endpoint
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset ambigqa --split validation --input "" \
  --limit 100 --k 10 --t 3 --prompt-style critique \
  --mi listing --async --concurrency 50 --provider hf \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/ambigqa_100_k10_t3_critique.csv --log-dir outputs/logs
```

---

## 7) Async Cloud Inference & Concurrency

Implemented in `llm_belief_mi_repro/llm_client.py` and wired in the CLI with `--async --concurrency`.
- Chains preserve per-step order; multiple chains and questions run concurrently.
- For HF router, set envs via `set_hf_env.sh`.

Test async (HF router):
```bash
export HF_TOKEN="hf_XXXXXXXXXXXXXXXX"
export HF_ROUTER_BASE="https://router.huggingface.co"
export LLM_MODEL="meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"
source ./set_hf_env.sh
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input triviaqa_val_subset.csv \
  --limit 300 --k 10 --t 3 --prompt-style naive \
  --mi listing \
  --async --concurrency 50 \
  --temperature 0.5 --max-tokens 64 \
  --output results_triviaqa_300_k10_t3_async.csv
```

---

## 8) Caching (SQLite)

Implemented in `llm_belief_mi_repro/cache.py` and integrated in both clients.
- Cache key = SHA-256 of canonicalized request payload (`url`, `model`, `messages`, `temperature`, `max_tokens`, options)
- Cache DB: `.cache/llm_cache.sqlite` (WAL mode)
- Modes: `readwrite|read|write|off`

Test cache hit/miss:
```bash
# First run populates cache
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input triviaqa_val_subset.csv \
  --limit 20 --k 5 --t 3 --prompt-style naive \
  --mi listing \
  --cache-path .cache/llm_cache.sqlite --cache-mode readwrite \
  --output results_cache_demo_1.csv

# Second run should be faster due to cache hits
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input triviaqa_val_subset.csv \
  --limit 20 --k 5 --t 3 --prompt-style naive \
  --mi listing \
  --cache-path .cache/llm_cache.sqlite --cache-mode readwrite \
  --output results_cache_demo_2.csv
```

---

## 9) Logging (Prompts & Run Metadata)

- A run ID is generated per invocation; logs go under `logs/<run_id>/`:
  - `prompts.jsonl`: one JSON record per chain step (messages, response text, token logprobs if requested, latency)
  - `run_metadata.json`: inputs, parameters, provider, outputs, metrics

Enable logging:
```bash
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input triviaqa_val_subset.csv \
  --limit 10 --k 3 --t 3 --prompt-style naive \
  --mi listing \
  --log-dir logs --log-verbosity minimal \
  --output results_logging_demo.csv
```

Inspect logs:
```bash
RUN_DIR=$(ls -dt logs/run_* | head -n1)
head -n 3 "$RUN_DIR/prompts.jsonl"
cat "$RUN_DIR/run_metadata.json"
```

---

## 10) Plotting (ROC/PR)

Implemented in `llm_belief_mi_repro/plots.py` and CLI `plot_roc`, `plot_pr`.
- ROC: one score at a time (`mi_bits|agreement|entropy_bits|verify_score|greedy_logprob`)
- PR: aggregates available series in one figure (MI, S.E., T0, S.V.); agreement is excluded by default per request.

Examples (per-dataset):
```bash
# ROC for MI (TriviaQA)
python -m llm_belief_mi_repro.cli plot_roc \
  --input outputs/results/triviaqa_200_k10_t3_naive_listing_metrics.csv \
  --score-col mi_bits \
  --save outputs/plots/roc_mi_triviaqa.png

# PR for all four metrics (TriviaQA)
python -m llm_belief_mi_repro.cli plot_pr \
  --input outputs/results/triviaqa_200_k10_t3_naive_listing_metrics.csv \
  --save outputs/plots/pr_all_triviaqa.png

# ROC/PR for AmbigQA
python -m llm_belief_mi_repro.cli plot_roc \
  --input outputs/results/ambigqa_200_k10_t3_naive_listing_metrics.csv \
  --score-col mi_bits \
  --save outputs/plots/roc_mi_ambigqa.png

python -m llm_belief_mi_repro.cli plot_pr \
  --input outputs/results/ambigqa_200_k10_t3_naive_listing_metrics.csv \
  --save outputs/plots/pr_all_ambigqa.png
```

---

## 11) YAML Config Runner

Subcommand `from_config` accepts a YAML file and maps to `run_dataset` arguments.

Example config (`run.yaml`):
```yaml
provider:
  base_url: ${LLM_API_BASE}   # e.g., https://router.huggingface.co/v1
  api_key: ${LLM_API_KEY}     # hf token
  model: ${LLM_MODEL}         # meta-llama/Llama-3.1-8B-Instruct:fireworks-ai

dataset:
  name: ambigqa
  split: validation
  input: ""
  limit: 1000

prompting:
  style: naive
  temperature: 0.5
  max_tokens: 64
  K: 10
  t: 3

estimators:
  mi: listing
  baselines: [greedy, self_verify]

execution:
  async: true
  concurrency: 50
  cache_path: .cache/llm_cache.sqlite
  cache_mode: readwrite
  log_dir: logs
  log_verbosity: minimal
  output_csv: results/ambigqa_1000_k10_t3_async.csv
```

Run:
```bash
python -m llm_belief_mi_repro.cli from_config --config run.yaml
```

---

## 12) Experiment Grid Script

Script: `llm_belief_mi_repro/scripts/experiment_grid.py` to sweep datasets × prompt styles.

Example:
```bash
python -m llm_belief_mi_repro.scripts.experiment_grid \
  --model "$LLM_MODEL" \
  --base-url "$LLM_API_BASE" --api-key "$LLM_API_KEY" \
  --datasets triviaqa,ambigqa --prompt-styles naive,wrong_prev,critique \
  --limit 1000 --k 10 --t 3 --mi listing \
  --async --concurrency 50 --baseline-greedy --baseline-verify \
  --input-triviaqa triviaqa_val_subset.csv --outdir results
```

---

## 13) Prompt Transcript (Demo)

Subcommand `dump_prompts` writes an example transcript (`SYSTEM`, `USER`, `ASSISTANT`) for a small subset.

Example:
```bash
python -m llm_belief_mi_repro.cli dump_prompts \
  --input triviaqa_val_subset.csv --limit 3 --t 3 \
  --base-url "$LLM_API_BASE" \
  --temperature 0.5 --max-tokens 128 \
  --output prompts/prompt_example.txt
```

---

## 14) Polarity & Score Orientation (sanity)

- Positive class in PR/ROC is hallucination (incorrect) = 1.
- Scores are oriented so higher = more likely hallucination:
  - MI, entropy: already oriented
  - Agreement: inverted (1 − agreement)
  - T0: negative logprob (−logprob)
  - S.V.: inverted (1 − score)

---

## 15) Known Limitations & Tips

- Token logprobs are provider-dependent; when absent, T0 columns will be omitted.
- For smooth PR, increase dataset size (e.g., ≥1k examples). Small `limit` yields jagged curves.
- When running locally, use smaller `--limit`, `--k`, and `--t` to keep latency manageable.

---

## 16) File Map (Key Implementations)

- `llm_belief_mi_repro/iterative_prompting.py`: prompt composition + chain runners (with prompt styles)
- `llm_belief_mi_repro/llm_client.py`: sync + async OpenAI-compatible clients (+logprobs, cache hooks)
- `llm_belief_mi_repro/mi_estimator.py`: plug-in MI + listing-style MI, entropy utilities
- `llm_belief_mi_repro/datasets.py`: TriviaQA/AmbigQA loaders + normalization
- `llm_belief_mi_repro/plots.py`: ROC/PR plotting
- `llm_belief_mi_repro/cache.py`: SQLite cache
- `llm_belief_mi_repro/evaluation.py`: metrics & curve points
- `llm_belief_mi_repro/scripts/*`: downloader and grid script
- `llm_belief_mi_repro/cli.py`: CLI subcommands & orchestration

---

Happy experimenting!

---

## 17) Quick sanity mini-batch (endpoint vs. router vs. local)

Use these to quickly verify outputs are PR-plot-ready and that T0 auto-skip behaves correctly.

Endpoint (auto-skip T0 if unsupported)
```bash
source ./set_hf_env.sh endpoint
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input outputs/subsets/triviaqa_val_subset.csv \
  --limit 6 --k 3 --t 2 --prompt-style naive \
  --mi listing --async --concurrency 10 --provider hf \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/triviaqa_quick_endpoint.csv \
  --log-dir outputs/logs

# PR plot (picks up whichever series are present; T0 omitted if auto-skipped)
python -m llm_belief_mi_repro.cli plot_pr \
  --input outputs/results/triviaqa_quick_endpoint.csv \
  --save outputs/plots/pr_all_quick_endpoint.png
```

Fireworks via router (T0 should be present)
```bash
source ./set_hf_env.sh router
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input outputs/subsets/triviaqa_val_subset.csv \
  --limit 6 --k 3 --t 2 --prompt-style naive \
  --mi listing --async --concurrency 10 --provider openai \
  --temperature 0.5 --max-tokens 64 \
  --baseline-greedy --baseline-verify \
  --output outputs/results/triviaqa_quick_fireworks.csv \
  --log-dir outputs/logs

python -m llm_belief_mi_repro.cli plot_pr \
  --input outputs/results/triviaqa_quick_fireworks.csv \
  --save outputs/plots/pr_all_quick_fireworks.png
```

Local LM Studio (T0 often unavailable; S.V. and MI/S.E. should still work)
```bash
source ./set_hf_env.sh local
python -m llm_belief_mi_repro.cli run_dataset \
  --dataset triviaqa --input outputs/subsets/triviaqa_val_subset.csv \
  --limit 6 --k 3 --t 2 --prompt-style naive \
  --mi listing --provider openai \
  --temperature 0.5 --max-tokens 64 \
  --baseline-verify \
  --output outputs/results/triviaqa_quick_local.csv \
  --log-dir outputs/logs

python -m llm_belief_mi_repro.cli plot_pr \
  --input outputs/results/triviaqa_quick_local.csv \
  --save outputs/plots/pr_all_quick_local.png
```
