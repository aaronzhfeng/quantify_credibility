from __future__ import annotations

import argparse
import itertools
import os
import shlex
import subprocess
import sys
from typing import List


def run_cmd(cmd: List[str]) -> int:
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep datasets x prompt styles x estimators with async concurrency")
    p.add_argument("--base-url", type=str, default=os.environ.get("LLM_API_BASE", ""), help="OpenAI-compatible base URL")
    p.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY", ""), help="API key")
    p.add_argument("--model", type=str, required=True, help="Model identifier")
    p.add_argument("--datasets", type=str, default="triviaqa,ambigqa", help="Comma list: triviaqa,ambigqa")
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--t", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt-styles", type=str, default="naive,wrong_prev,critique")
    p.add_argument("--mi", type=str, default="listing", choices=["listing", "plugin"])
    p.add_argument("--concurrency", type=int, default=50)
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--baseline-greedy", action="store_true")
    p.add_argument("--baseline-verify", action="store_true")
    p.add_argument("--input-triviaqa", type=str, default="", help="Optional path to TriviaQA CSV (else required)")
    p.add_argument("--input-ambigqa", type=str, default="", help="Optional AmbigQA JSONL; if empty uses HF")
    p.add_argument("--outdir", type=str, default="results", help="Directory for output CSVs")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    prompt_styles = [s.strip() for s in args.prompt_styles.split(",") if s.strip()]

    for ds, ps in itertools.product(datasets, prompt_styles):
        output = os.path.join(
            args.outdir,
            f"results_{ds}_{args.limit}_k{args.k}_t{args.t}_{ps}.csv",
        )
        input_arg = ""
        if ds == "triviaqa":
            if not args.input_triviaqa:
                print("[WARN] --input-triviaqa is empty; please provide a CSV path.")
            input_arg = args.input_triviaqa
        elif ds == "ambigqa":
            input_arg = args.input_ambigqa  # can be empty to use HF loader

        cmd = [
            sys.executable,
            "-m",
            "llm_belief_mi_repro.cli",
            "run_dataset",
            "--dataset",
            ds,
            "--split",
            args.split,
            "--input",
            input_arg,
            "--limit",
            str(args.limit),
            "--k",
            str(args.k),
            "--t",
            str(args.t),
            "--prompt-style",
            ps,
            "--mi",
            args.mi,
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--api-key",
            args.api_key,
            "--async",
            "--concurrency",
            str(args.concurrency),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--output",
            output,
        ]
        if args.baseline_greedy:
            cmd.append("--baseline-greedy")
        if args.baseline_verify:
            cmd.append("--baseline-verify")
        rc = run_cmd(cmd)
        if rc != 0:
            print(f"[ERROR] command failed with exit {rc}")
            break


if __name__ == "__main__":
    main()


