import argparse
import csv
import os
import random
from typing import List, Optional
from tqdm import tqdm

from .llm_client import OpenAICompatibleLLMClient
from .iterative_prompting import compose_prompt, run_chain_for_query, run_k_chains_for_query
from .mi_estimator import estimate_mi_nats, nats_to_bits, entropy_nats, estimate_mi_listing_nats
from .datasets import load_toy_questions, load_triviaqa_subset, answers_match, QAExample
from .evaluation import (
    compute_agreement_fraction,
    split_indices,
    choose_threshold,
    evaluate_at_threshold,
    roc_curve_points,
    precision_recall_curve_points,
)


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

    run_ds = sub.add_parser("run_dataset", help="Run per-question K-chain MI on a TriviaQA subset.")
    run_ds.add_argument("--input", type=str, required=True, help="Path to TriviaQA subset (.jsonl/.csv)")
    run_ds.add_argument("--limit", type=int, default=100, help="Max examples to load")
    run_ds.add_argument("--k", type=int, default=20, help="Number of chains per question")
    run_ds.add_argument("--t", type=int, default=3, help="Chain length")
    run_ds.add_argument("--base-url", type=str, default=os.environ.get("LLM_API_BASE", "http://localhost:1234/v1"))
    run_ds.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY", "lm-studio"))
    run_ds.add_argument("--model", type=str, required=True)
    run_ds.add_argument("--temperature", type=float, default=0.0)
    run_ds.add_argument("--max-tokens", type=int, default=64)
    run_ds.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction for threshold selection")
    run_ds.add_argument("--seed", type=int, default=0)
    run_ds.add_argument("--output", type=str, required=True, help="Output CSV path")
    run_ds.add_argument("--mi", type=str, default="plugin", choices=["plugin", "listing"], help="MI estimator to use")
    run_ds.add_argument("--baseline-greedy", action="store_true", help="Compute greedy logprob baseline (requires logprobs)")
    run_ds.add_argument("--baseline-verify", action="store_true", help="Compute self-verification baseline (extra calls)")

    plot_p = sub.add_parser("plot_roc", help="Plot ROC from a results CSV (requires matplotlib)")
    plot_p.add_argument("--input", type=str, required=True, help="Per-question results CSV")
    plot_p.add_argument(
        "--score-col",
        type=str,
        default="mi_bits",
        help="Which score column to plot (mi_bits|agreement|entropy_bits)",
    )
    plot_p.add_argument("--save", type=str, default="", help="Path to save the figure (optional)")

    pr_p = sub.add_parser("plot_pr", help="Plot PR curves for all available scores in a CSV")
    pr_p.add_argument("--input", type=str, required=True, help="Per-question results CSV")
    pr_p.add_argument("--save", type=str, default="", help="Path to save the figure (optional)")

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


def cmd_run_dataset(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    examples: List[QAExample] = load_triviaqa_subset(args.input, limit=args.limit)

    client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        request_timeout_s=120,
    )

    # Per-question K chains and per-question MI
    rows = []
    mi_scores: List[float] = []
    agree_scores: List[float] = []
    entropy_scores_bits: List[float] = []
    labels: List[int] = []

    q_bar = tqdm(examples, total=len(examples), desc="Questions", unit="q")
    extra_per_q = 1 if args.baseline_greedy else 0
    extra_per_chain = 1 if args.baseline_verify else 0
    total_calls = len(examples) * (args.k * max(1, args.t) + extra_per_q + args.k * extra_per_chain)
    call_bar = tqdm(total=total_calls, desc="API calls", unit="call")
    def on_call():
        call_bar.update(1)

    for ex in q_bar:
        chains = run_k_chains_for_query(
            client=client,
            query=ex.question,
            chain_length=args.t,
            k=args.k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            on_request=on_call,
        )
        # Final answers are the last step y_t of each chain
        finals = [ch[-1] if ch else "" for ch in chains]
        if args.mi == "listing":
            mi_n = estimate_mi_listing_nats(chains)
        else:
            mi_n = estimate_mi_nats(chains)
        mi_b = nats_to_bits(mi_n)
        agree = compute_agreement_fraction(finals)
        ent_n = entropy_nats(finals)
        ent_b = nats_to_bits(ent_n)
        # Simple label: any final answers match any gold
        from .datasets import normalize_answer  # local import to avoid cycle

        lab = 1 if any(normalize_answer(a) in {normalize_answer(g) for g in ex.answers} for a in finals) else 0

        greedy_lp = None
        if args.baseline_greedy:
            # Greedy decode with logprobs for the last step only
            messages = compose_prompt(ex.question, [])
            text, token_lps = client.chat_completion_with_logprobs(
                messages,
                temperature=0.0,
                max_tokens=args.max_tokens,
            )
            if token_lps is not None:
                greedy_lp = sum(token_lps)
            on_call()

        verify_score = None
        if args.baseline_verify:
            # Self-verification: for each final answer, ask model for a 0-1 confidence
            prompt = (
                "You are a strict judge. Given the question and an answer, output only a number between 0 and 1 "
                "representing confidence that the answer is correct. No text, just the number."
            )
            scores = []
            for ans in finals:
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Question: {ex.question}\nAnswer: {ans}\nConfidence (0-1):"},
                ]
                resp = client.chat_completion(messages, temperature=0.0, max_tokens=8)
                try:
                    scores.append(float(resp.strip().split()[0]))
                except Exception:
                    scores.append(0.0)
                on_call()
            if scores:
                verify_score = sum(scores) / len(scores)

        mi_scores.append(mi_b)
        agree_scores.append(agree)
        entropy_scores_bits.append(ent_b)
        labels.append(lab)
        rows.append({
            "question": ex.question,
            "mi_bits": f"{mi_b:.6f}",
            "agreement": f"{agree:.6f}",
            "entropy_bits": f"{ent_b:.6f}",
            "k": args.k,
            "t": args.t,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "label_any_correct": lab,
            "gold_answers": " | ".join(ex.answers),
            **({"greedy_logprob": f"{greedy_lp:.6f}"} if greedy_lp is not None else {}),
            **({"verify_score": f"{verify_score:.6f}"} if verify_score is not None else {}),
        })

    # Calibration: split and choose threshold on val
    val_idx, test_idx = split_indices(len(mi_scores), val_fraction=args.val_frac, seed=args.seed)
    val_scores = [mi_scores[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    thr = choose_threshold(val_scores, val_labels, maximize="youden")
    thr_agree = choose_threshold([agree_scores[i] for i in val_idx], [labels[i] for i in val_idx], maximize="youden")
    thr_ent = choose_threshold([entropy_scores_bits[i] for i in val_idx], [labels[i] for i in val_idx], maximize="youden")

    test_scores = [mi_scores[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    metrics = evaluate_at_threshold(test_scores, test_labels, thr)
    metrics_agree = evaluate_at_threshold([agree_scores[i] for i in test_idx], [labels[i] for i in test_idx], thr_agree)
    metrics_ent = evaluate_at_threshold([entropy_scores_bits[i] for i in test_idx], [labels[i] for i in test_idx], thr_ent)
    if any("greedy_logprob" in r for r in rows):
        val_greedy = [float(rows[i]["greedy_logprob"]) for i in val_idx if "greedy_logprob" in rows[i]]
        test_greedy = [float(rows[i]["greedy_logprob"]) for i in test_idx if "greedy_logprob" in rows[i]]
        labels_val = [labels[i] for i in val_idx][: len(val_greedy)]
        labels_test = [labels[i] for i in test_idx][: len(test_greedy)]
        thr_greedy = choose_threshold(val_greedy, labels_val, maximize="youden") if val_greedy else 0.0
        metrics_greedy = evaluate_at_threshold(test_greedy, labels_test, thr_greedy) if test_greedy else None
    else:
        metrics_greedy = None
    if any("verify_score" in r for r in rows):
        val_ver = [float(rows[i]["verify_score"]) for i in val_idx if "verify_score" in rows[i]]
        test_ver = [float(rows[i]["verify_score"]) for i in test_idx if "verify_score" in rows[i]]
        labels_val2 = [labels[i] for i in val_idx][: len(val_ver)]
        labels_test2 = [labels[i] for i in test_idx][: len(test_ver)]
        thr_ver = choose_threshold(val_ver, labels_val2, maximize="youden") if val_ver else 0.0
        metrics_ver = evaluate_at_threshold(test_ver, labels_test2, thr_ver) if test_ver else None
    else:
        metrics_ver = None

    # Write per-question rows
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["question"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    call_bar.close(); q_bar.close()
    print(f"Wrote per-question results: {args.output}")
    print("Test metrics (MI bits):", metrics)
    print("Test metrics (Agreement):", metrics_agree)
    print("Test metrics (Entropy bits):", metrics_ent)
    if metrics_greedy is not None:
        print("Test metrics (Greedy logprob):", metrics_greedy)
    if metrics_ver is not None:
        print("Test metrics (Self-verify):", metrics_ver)

    # Optional: ROC curve points printed for plotting scripts
    fpr, tpr = roc_curve_points(test_scores, test_labels)
    print("ROC curve points for MI (FPR,TPR) few samples:", list(zip(fpr[:5], tpr[:5])), "... (total:", len(fpr), ")")


def cmd_plot_roc(args: argparse.Namespace) -> None:
    import csv as _csv
    from .plots import try_plot_roc_curve

    scores: List[float] = []
    labels: List[int] = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                s = float(row.get(args.score_col, "nan"))
                y = int(row.get("label_any_correct", 0))
            except Exception:
                continue
            if not (s == s):
                continue
            scores.append(s)
            labels.append(y)
    fpr, tpr = roc_curve_points(scores, labels)
    save_path = args.save if getattr(args, "save", "") else None
    try_plot_roc_curve(fpr, tpr, title=f"ROC for {args.score_col}", save_path=save_path)


def cmd_plot_pr(args: argparse.Namespace) -> None:
    import csv as _csv
    from .plots import try_plot_pr_curves

    with open(args.input, "r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = list(reader)

    candidate_cols = [
        ("M.I. score", "mi_bits"),
        ("S.E. score", "entropy_bits"),
        ("T0 score", "greedy_logprob"),
        ("S.V. score", "verify_score"),
    ]
    series = []
    labels_list = [int(r.get("label_any_correct", 0)) for r in rows]
    for label, col in candidate_cols:
        if rows and col in rows[0]:
            try:
                scores = [float(r.get(col, "nan")) for r in rows]
            except Exception:
                continue
            pairs = [(s, y) for s, y in zip(scores, labels_list) if s == s]
            if not pairs:
                continue
            s_aligned = [p[0] for p in pairs]
            y_aligned = [p[1] for p in pairs]
            rec, prec = precision_recall_curve_points(s_aligned, y_aligned)
            series.append((label, rec, prec))

    if not series:
        print("No score columns found to plot.")
        return

    save_path = args.save if getattr(args, "save", "") else None
    try_plot_pr_curves(series, title="Precision-Recall (all scores)", save_path=save_path)


def main() -> None:
    args = parse_args()
    if args.command == "run":
        cmd_run(args)
        return
    if args.command == "run_dataset":
        cmd_run_dataset(args)
        return
    if args.command == "plot_roc":
        cmd_plot_roc(args)
        return
    if args.command == "plot_pr":
        cmd_plot_pr(args)
        return
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()


