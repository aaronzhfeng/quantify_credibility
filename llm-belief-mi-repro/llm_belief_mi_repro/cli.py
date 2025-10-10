import argparse
import csv
import os
import random
from typing import List, Optional

from .llm_client import OpenAICompatibleLLMClient
from .iterative_prompting import compose_prompt, run_chain_for_query
from .mi_estimator import estimate_mi_nats, nats_to_bits
from .datasets import load_toy_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce iterative prompting + MI from 'To Believe or Not to Believe Your LLM'"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run iterative prompting on a dataset and estimate MI.")
    run_p.add_argument("--toy", action="store_true", help="Use a built-in toy set of questions.")
    run_p.add_argument("--n", type=int, default=20, help="Number of questions to run.")
    run_p.add_argument("--t", type=int, default=3, help="Chain length (number of responses).")
    run_p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("LLM_API_BASE", "http://localhost:1234/v1"),
        help="OpenAI-compatible API base URL",
    )
    run_p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("LLM_API_KEY", "lm-studio"),
        help="API key (often unused for local servers)",
    )
    run_p.add_argument("--model", type=str, required=True, help="Model name/identifier.")
    run_p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    run_p.add_argument("--max-tokens", type=int, default=128, help="Max tokens per response.")
    run_p.add_argument("--seed", type=int, default=0, help="Random seed.")
    run_p.add_argument("--output", type=str, default="", help="Optional CSV output path.")

    return parser.parse_args()


def write_csv(output_path: str, queries: List[str], chains: List[List[str]]) -> None:
    if not output_path:
        return
    fieldnames = ["query"] + [f"y{i+1}" for i in range(len(chains[0]) if chains else 0)]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for q, ch in zip(queries, chains):
            row = {"query": q}
            for i, yi in enumerate(ch):
                row[f"y{i+1}"] = yi
            w.writerow(row)


def cmd_run(args: argparse.Namespace) -> None:
    random.seed(args.seed)

    if args.toy:
        questions = load_toy_questions()
    else:
        raise SystemExit("Please pass --toy for now or extend dataset loaders.")

    if args.n > 0:
        questions = questions[: args.n]

    client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        request_timeout_s=120,
    )

    chains: List[List[str]] = []
    for q in questions:
        chain = run_chain_for_query(
            client=client,
            query=q,
            chain_length=args.t,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        chains.append(chain)

    mi_nats = estimate_mi_nats(chains)
    mi_bits = nats_to_bits(mi_nats)
    print(f"Estimated MI: {mi_nats:.4f} nats ({mi_bits:.4f} bits)")

    write_csv(args.output, questions, chains)
    if args.output:
        print(f"Wrote CSV: {args.output}")


def main() -> None:
    args = parse_args()
    if args.command == "run":
        cmd_run(args)
        return
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()


