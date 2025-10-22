import argparse
import csv
import os
import random
from typing import List, Optional
from tqdm import tqdm

from .llm_client import (
    OpenAICompatibleLLMClient,
    AsyncOpenAICompatibleLLMClient,
    HuggingFaceInferenceClient,
    AsyncHuggingFaceInferenceClient,
)
from .iterative_prompting import compose_prompt, run_chain_for_query, run_k_chains_for_query
from .mi_estimator import estimate_mi_nats, nats_to_bits, entropy_nats, estimate_mi_listing_nats
from .datasets import load_toy_questions, load_triviaqa_subset, load_ambigqa_subset, answers_match, QAExample
from .evaluation import (
    compute_agreement_fraction,
    label_any_correct,
    label_majority_correct,
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
    run_p.add_argument(
        "--model",
        type=str,
        required=False,
        default=os.environ.get("LLM_MODEL", ""),
        help="Model name/identifier (or set env LLM_MODEL)",
    )
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
    run_ds.add_argument(
        "--model",
        type=str,
        required=False,
        default=os.environ.get("LLM_MODEL", ""),
        help="Model name/identifier (or set env LLM_MODEL)",
    )
    run_ds.add_argument("--temperature", type=float, default=0.0)
    run_ds.add_argument("--max-tokens", type=int, default=64)
    run_ds.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction for threshold selection")
    run_ds.add_argument("--seed", type=int, default=0)
    run_ds.add_argument("--output", type=str, required=True, help="Output CSV path")
    run_ds.add_argument("--mi", type=str, default="plugin", choices=["plugin", "listing"], help="MI estimator to use")
    run_ds.add_argument("--baseline-greedy", action="store_true", help="Compute greedy logprob baseline (requires logprobs)")
    run_ds.add_argument("--baseline-verify", action="store_true", help="Compute self-verification baseline (extra calls)")
    run_ds.add_argument("--dataset", type=str, default="triviaqa", choices=["triviaqa", "ambigqa"], help="Dataset source")
    run_ds.add_argument("--split", type=str, default="validation", help="Dataset split (if using HF)")
    run_ds.add_argument("--prompt-style", type=str, default="naive", choices=["naive", "wrong_prev", "critique"], help="Prompt variant for iterative prompting")
    run_ds.add_argument("--async", dest="do_async", action="store_true", help="Use async client for concurrency")
    run_ds.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests when async is enabled")
    import os as _os_env
    run_ds.add_argument("--provider", type=str, default=_os_env.environ.get("PROVIDER", "openai"), choices=["openai", "hf"], help="Backend provider: openai-compatible or Hugging Face Inference endpoint")
    run_ds.add_argument("--cache-path", type=str, default=".cache/llm_cache.sqlite", help="SQLite cache path")
    run_ds.add_argument("--cache-mode", type=str, default="readwrite", choices=["readwrite", "read", "write", "off"], help="Cache mode")
    run_ds.add_argument("--log-dir", type=str, default="logs", help="Directory to write logs for this run")
    run_ds.add_argument("--log-verbosity", type=str, default="minimal", choices=["minimal", "full"], help="Logging verbosity")
    run_ds.add_argument("--label-policy", type=str, default="any", choices=["any", "majority"], help="Ground-truth labeling: any correct vs majority correct")

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

    cfg_p = sub.add_parser("from_config", help="Run from a YAML config")
    cfg_p.add_argument("--config", type=str, required=True, help="Path to YAML config")

    dump_p = sub.add_parser("dump_prompts", help="Create an example prompt transcript for a subset using iterative prompting")
    dump_p.add_argument("--input", type=str, required=True, help="Path to TriviaQA subset (.jsonl/.csv)")
    dump_p.add_argument("--limit", type=int, default=3, help="How many questions to include")
    dump_p.add_argument("--t", type=int, default=3, help="Chain length")
    dump_p.add_argument("--base-url", type=str, default=os.environ.get("LLM_API_BASE", "http://localhost:1234/v1"))
    dump_p.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY", "lm-studio"))
    dump_p.add_argument("--model", type=str, required=True)
    dump_p.add_argument("--temperature", type=float, default=0.5)
    dump_p.add_argument("--max-tokens", type=int, default=128)
    dump_p.add_argument("--output", type=str, required=True, help="Output transcript path (e.g., prompts/prompt_example.txt)")

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
    if args.dataset == "ambigqa":
        src_path = args.input if args.input and args.input.strip() else None
        examples = load_ambigqa_subset(src_path, split=args.split, limit=args.limit)
    else:
        examples = load_triviaqa_subset(args.input, limit=args.limit)

    # init cache
    from .cache import SQLiteCache
    cache = SQLiteCache(args.cache_path, mode=args.cache_mode)

    if args.do_async:
        import asyncio
        if args.provider == "hf":
            async_client = AsyncHuggingFaceInferenceClient(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                request_timeout_s=120,
                semaphore=asyncio.Semaphore(max(1, int(args.concurrency))),
                cache=cache,
            )
        else:
            async_client = AsyncOpenAICompatibleLLMClient(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                request_timeout_s=120,
                semaphore=asyncio.Semaphore(max(1, int(args.concurrency))),
                cache=cache,
            )
    else:
        if args.provider == "hf":
            client = HuggingFaceInferenceClient(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                request_timeout_s=120,
                cache=cache,
            )
        else:
            client = OpenAICompatibleLLMClient(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                request_timeout_s=120,
                cache=cache,
            )

    # Per-question K chains and per-question MI
    rows = []
    mi_scores: List[float] = []
    agree_scores: List[float] = []
    entropy_scores_bits: List[float] = []
    labels: List[int] = []

    # logging setup
    import uuid, json, time, os as _os
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    _os.makedirs(args.log_dir, exist_ok=True)
    run_dir = _os.path.join(args.log_dir, run_id)
    _os.makedirs(run_dir, exist_ok=True)
    prompts_path = _os.path.join(run_dir, "prompts.jsonl")
    prompts_f = open(prompts_path, "w", encoding="utf-8")
    # raw responses folder
    raw_dir = _os.path.join(run_dir, "raw")
    _os.makedirs(raw_dir, exist_ok=True)

    q_bar = tqdm(examples, total=len(examples), desc="Questions", unit="q")
    extra_per_q = 1 if args.baseline_greedy else 0
    extra_per_chain = 1 if args.baseline_verify else 0
    total_calls = len(examples) * (args.k * max(1, args.t) + extra_per_q + args.k * extra_per_chain)
    call_bar = tqdm(total=total_calls, desc="API calls", unit="call")
    def on_call():
        call_bar.update(1)

    greedy_capable: Optional[bool] = None  # None=unknown, True=supported, False=unsupported
    verify_capable: Optional[bool] = None  # None=unknown, True=supported, False=unsupported
    for ex in q_bar:
        if args.do_async:
            # Async: run K chains concurrently but steps sequentially per chain
            import asyncio

            async def run_one_chain(prev: list[str]) -> list[str]:
                answers: list[str] = []
                for _ in range(max(1, args.t)):
                    messages = compose_prompt(ex.question, answers, prompt_style=args.prompt_style)
                    t0 = time.time()
                    # save raw response as a JSON file per step
                    def _save_raw(data):
                        try:
                            fname = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.json"
                            with open(_os.path.join(raw_dir, fname), "w", encoding="utf-8") as rf:
                                json.dump(data, rf, ensure_ascii=False)
                        except Exception:
                            pass
                    text, token_lps = await async_client.chat_completion(
                        messages=messages,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        logprobs=False,
                        top_logprobs=1,
                        on_raw=_save_raw,
                    )
                    on_call()
                    answers.append(text)
                    rec = {
                        "run_id": run_id,
                        "dataset": args.dataset,
                        "question": ex.question,
                        "chain_step": len(answers),
                        "messages": messages if args.log_verbosity == "full" else {"user": messages[-1]["content"]},
                        "response_text": text,
                        "token_logprobs": token_lps if args.log_verbosity == "full" else None,
                        "latency_ms": int((time.time() - t0) * 1000),
                    }
                    prompts_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return answers

            tasks = [run_one_chain([]) for _ in range(max(1, args.k))]
            chains = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
        else:
            chains = run_k_chains_for_query(
                client=client,
                query=ex.question,
                chain_length=args.t,
                k=args.k,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                on_request=on_call,
                prompt_style=args.prompt_style,
            )
            # log sync path
            # Only log the last step per chain to limit volume in minimal mode
            if args.log_verbosity == "full":
                pass
            else:
                for ch in chains:
                    rec = {
                        "run_id": run_id,
                        "dataset": args.dataset,
                        "question": ex.question,
                        "chain_step": len(ch),
                        "messages": {"user": compose_prompt(ex.question, ch[:-1], prompt_style=args.prompt_style)[-1]["content"]},
                        "response_text": ch[-1] if ch else "",
                        "token_logprobs": None,
                        "latency_ms": None,
                    }
                    prompts_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
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
        # Label policy: any correct (default) or majority correct
        if args.label_policy == "majority":
            lab = label_majority_correct(finals, ex.answers, normalizer=lambda s: s.strip().lower())
        else:
            from .datasets import normalize_answer  # local import to avoid cycle
            lab = 1 if any(normalize_answer(a) in {normalize_answer(g) for g in ex.answers} for a in finals) else 0

        greedy_lp = None
        greedy_lp_avg = None
        if args.baseline_greedy:
            # Greedy decode with logprobs for the last step only
            if greedy_capable is not False:
                msg = compose_prompt(ex.question, [], prompt_style=args.prompt_style)
                token_lps = None
                try:
                    if args.do_async:
                        import asyncio
                        def _save_raw_g(d):
                            try:
                                fname = f"greedy_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.json"
                                with open(_os.path.join(raw_dir, fname), "w", encoding="utf-8") as rf:
                                    json.dump(d, rf, ensure_ascii=False)
                            except Exception:
                                pass
                        text, token_lps = asyncio.get_event_loop().run_until_complete(
                            async_client.chat_completion(
                                messages=msg,
                                temperature=0.0,
                                max_tokens=args.max_tokens,
                                logprobs=True,
                                top_logprobs=1,
                                on_raw=_save_raw_g,
                            )
                        )
                    else:
                        # Prefer provider-specific logprob API when available
                        if args.provider == "hf":
                            def _save_raw_g2(d):
                                try:
                                    fname = f"greedy_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.json"
                                    with open(_os.path.join(raw_dir, fname), "w", encoding="utf-8") as rf:
                                        json.dump(d, rf, ensure_ascii=False)
                                except Exception:
                                    pass
                            text, token_lps = client.chat_completion_with_logprobs(
                                msg,
                                temperature=0.0,
                                max_tokens=args.max_tokens,
                                top_logprobs=1,
                                on_raw=_save_raw_g2,
                            )
                        else:
                            def _save_raw_g3(d):
                                try:
                                    fname = f"greedy_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.json"
                                    with open(_os.path.join(raw_dir, fname), "w", encoding="utf-8") as rf:
                                        json.dump(d, rf, ensure_ascii=False)
                                except Exception:
                                    pass
                            text, token_lps = client.chat_completion_with_logprobs(
                                msg,
                                temperature=0.0,
                                max_tokens=args.max_tokens,
                                top_logprobs=1,
                                on_raw=_save_raw_g3,
                            )
                except Exception:
                    token_lps = None
                if token_lps is not None:
                    greedy_lp = sum(token_lps)
                    greedy_lp_avg = greedy_lp / max(1, len(token_lps))
                    greedy_capable = True
                else:
                    # Auto-skip T0 for subsequent questions if unsupported
                    greedy_capable = False
                on_call()

        verify_score = None
        if args.baseline_verify:
            # Self-verification: for each final answer, ask model for a 0-1 confidence
            prompt = (
                "You are a strict judge. Given the question and an answer, output only a number between 0 and 1 "
                "representing confidence that the answer is correct. No text, just the number."
            )
            scores = []
            if verify_capable is not False:
                if args.do_async:
                    import asyncio

                    async def get_score_async(ans: str) -> float:
                        messages = [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": f"Question: {ex.question}\nAnswer: {ans}\nConfidence (0-1):"},
                        ]
                        text, _ = await async_client.chat_completion(messages=messages, temperature=0.0, max_tokens=8)
                        try:
                            return float(text.strip().split()[0])
                        except Exception:
                            return 0.0
                    try:
                        scores = asyncio.get_event_loop().run_until_complete(asyncio.gather(*(get_score_async(a) for a in finals)))
                        for _ in finals:
                            on_call()
                        verify_capable = True
                    except Exception:
                        verify_capable = False
                        scores = []
                else:
                    try:
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
                        verify_capable = True
                    except Exception:
                        verify_capable = False
                        scores = []
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
            **({"greedy_logprob_avg": f"{greedy_lp_avg:.6f}"} if greedy_lp_avg is not None else {}),
            **({"verify_score": f"{verify_score:.6f}"} if verify_score is not None else {}),
        })

    # Calibration: positive class = hallucination (incorrect). Orient scores accordingly.
    labels_err = [1 - y for y in labels]
    val_idx, test_idx = split_indices(len(mi_scores), val_fraction=args.val_frac, seed=args.seed)

    # MI (already uncertainty)
    val_scores = [mi_scores[i] for i in val_idx]
    val_labels = [labels_err[i] for i in val_idx]
    thr = choose_threshold(val_scores, val_labels, maximize="youden")

    test_scores = [mi_scores[i] for i in test_idx]
    test_labels = [labels_err[i] for i in test_idx]
    metrics = evaluate_at_threshold(test_scores, test_labels, thr)

    # Agreement (invert to uncertainty)
    agree_unc_val = [1.0 - agree_scores[i] for i in val_idx]
    agree_unc_test = [1.0 - agree_scores[i] for i in test_idx]
    thr_agree = choose_threshold(agree_unc_val, [labels_err[i] for i in val_idx], maximize="youden")
    metrics_agree = evaluate_at_threshold(agree_unc_test, [labels_err[i] for i in test_idx], thr_agree)

    # Entropy bits (already uncertainty)
    thr_ent = choose_threshold([entropy_scores_bits[i] for i in val_idx], [labels_err[i] for i in val_idx], maximize="youden")
    metrics_ent = evaluate_at_threshold([entropy_scores_bits[i] for i in test_idx], [labels_err[i] for i in test_idx], thr_ent)

    # Greedy logprob baseline (if available): invert sign so higher=worse
    if any("greedy_logprob" in r for r in rows):
        val_greedy_raw = [float(rows[i]["greedy_logprob"]) for i in val_idx if "greedy_logprob" in rows[i]]
        test_greedy_raw = [float(rows[i]["greedy_logprob"]) for i in test_idx if "greedy_logprob" in rows[i]]
        val_greedy = [-x for x in val_greedy_raw]
        test_greedy = [-x for x in test_greedy_raw]
        labels_val = [labels_err[i] for i in val_idx if "greedy_logprob" in rows[i]]
        labels_test = [labels_err[i] for i in test_idx if "greedy_logprob" in rows[i]]
        thr_greedy = choose_threshold(val_greedy, labels_val, maximize="youden") if val_greedy else 0.0
        metrics_greedy = evaluate_at_threshold(test_greedy, labels_test, thr_greedy) if test_greedy else None
    else:
        metrics_greedy = None

    # Self-verify baseline (invert to uncertainty)
    if any("verify_score" in r for r in rows):
        val_ver_raw = [float(rows[i]["verify_score"]) for i in val_idx if "verify_score" in rows[i]]
        test_ver_raw = [float(rows[i]["verify_score"]) for i in test_idx if "verify_score" in rows[i]]
        val_ver = [1.0 - x for x in val_ver_raw]
        test_ver = [1.0 - x for x in test_ver_raw]
        labels_val2 = [labels_err[i] for i in val_idx if "verify_score" in rows[i]]
        labels_test2 = [labels_err[i] for i in test_idx if "verify_score" in rows[i]]
        thr_ver = choose_threshold(val_ver, labels_val2, maximize="youden") if val_ver else 0.0
        metrics_ver = evaluate_at_threshold(test_ver, labels_test2, thr_ver) if test_ver else None
    else:
        metrics_ver = None

    # Write per-question rows (ensure header includes union of all keys)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        if rows:
            all_keys = set()
            for r in rows:
                all_keys.update(r.keys())
            fieldnames = list(all_keys)
        else:
            fieldnames = ["question"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    call_bar.close(); q_bar.close()
    print(f"Wrote per-question results: {args.output}")
    # write run metadata and close logs
    meta = {
        "run_id": run_id,
        "base_url": args.base_url,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "k": args.k,
        "t": args.t,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_style": args.prompt_style,
        "mi": args.mi,
        "baselines": {"greedy": bool(args.baseline_greedy), "self_verify": bool(args.baseline_verify)},
        "async": bool(args.do_async),
        "concurrency": args.concurrency,
        "cache_path": args.cache_path,
        "cache_mode": args.cache_mode,
        "log_dir": run_dir,
        "results_csv": args.output,
        "metrics": {"mi": metrics, "agreement": metrics_agree, "entropy": metrics_ent, "greedy": metrics_greedy, "self_verify": metrics_ver},
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    prompts_f.close()
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
            # Orient score so higher = more likely hallucination
            if args.score_col == "agreement":
                s = 1.0 - s
            elif args.score_col == "verify_score":
                s = 1.0 - s
            elif args.score_col == "greedy_logprob":
                s = -s
            scores.append(s)
            labels.append(1 - y)
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
    labels_list = [1 - int(r.get("label_any_correct", 0)) for r in rows]  # 1 = hallucination
    for label, col in candidate_cols:
        if rows and col in rows[0]:
            try:
                raw_scores = [float(r.get(col, "nan")) for r in rows]
            except Exception:
                continue
            # Orient scores so higher = more likely hallucination
            if col == "agreement":
                scores = [1.0 - s for s in raw_scores]
            elif col == "verify_score":
                scores = [1.0 - s for s in raw_scores]
            elif col == "greedy_logprob":
                scores = [-s for s in raw_scores]
            else:
                scores = raw_scores
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


def cmd_dump_prompts(args: argparse.Namespace) -> None:
    examples: List[QAExample] = load_triviaqa_subset(args.input, limit=args.limit)
    client = OpenAICompatibleLLMClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        request_timeout_s=120,
    )

    lines: List[str] = []
    lines.append("SYSTEM:\nYou are a helpful, concise assistant. Answer accurately. If unsure, say so briefly.\n")
    for idx, ex in enumerate(examples, start=1):
        answers: List[str] = []
        for step in range(max(1, args.t)):
            history = "".join(f"\nAnother answer to question Q is: {a}" for a in answers)
            user = (
                "Consider the following question (Q) and previous answers if any." +
                history +
                f"\nProvide an answer to the following question:\nQ: {ex.question}\nA:"
            )
            lines.append("---\n")
            lines.append(f"Q{idx} (step {step+1}):\nUSER:\n{user}\n")
            # Optionally query the model to fill the chain for demonstration
            resp = client.chat_completion(
                messages=[{"role": "system", "content": "You are a helpful, concise assistant. Answer accurately. If unsure, say so briefly."}, {"role": "user", "content": user}],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            answers.append(resp)
            lines.append(f"ASSISTANT:\n{resp}\n")

    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"Wrote prompt transcript: {args.output}")


def cmd_from_config(args: argparse.Namespace) -> None:
    import yaml, sys, os
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    provider = cfg.get("provider", {})
    dataset = cfg.get("dataset", {})
    prompting = cfg.get("prompting", {})
    estimators = cfg.get("estimators", {})
    execution = cfg.get("execution", {})

    # Expand environment variables in provider fields if present
    base_url = os.path.expandvars(str(provider.get("base_url", "")))
    api_key = os.path.expandvars(str(provider.get("api_key", "")))

    argv = [
        "run_dataset",
        "--dataset", str(dataset.get("name", "triviaqa")),
        "--split", str(dataset.get("split", "validation")),
        "--input", str(dataset.get("input", "")),
        "--limit", str(dataset.get("limit", 1000)),
        "--k", str(prompting.get("K", 10)),
        "--t", str(prompting.get("t", 3)),
        "--prompt-style", str(prompting.get("style", "naive")),
        "--mi", str(estimators.get("mi", "listing")),
        "--model", str(provider.get("model", "")),
        "--base-url", base_url,
        "--api-key", api_key,
        "--temperature", str(prompting.get("temperature", 0.5)),
        "--max-tokens", str(prompting.get("max_tokens", 64)),
        "--output", str(execution.get("output_csv", "results.csv")),
        "--cache-path", str(execution.get("cache_path", ".cache/llm_cache.sqlite")),
        "--cache-mode", str(execution.get("cache_mode", "readwrite")),
        "--log-dir", str(execution.get("log_dir", "logs")),
        "--log-verbosity", str(execution.get("log_verbosity", "minimal")),
    ]
    if bool(execution.get("async", True)):
        argv.extend(["--async", "--concurrency", str(execution.get("concurrency", 50))])
    if "greedy" in (estimators.get("baselines", []) or []):
        argv.append("--baseline-greedy")
    if "self_verify" in (estimators.get("baselines", []) or []):
        argv.append("--baseline-verify")
    sys.argv = [sys.argv[0]] + argv
    main()


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
    if args.command == "from_config":
        cmd_from_config(args)
        return
    if args.command == "dump_prompts":
        cmd_dump_prompts(args)
        return
    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()


