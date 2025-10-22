from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List


def main() -> None:
    p = argparse.ArgumentParser(description="Download AmbigQA subset to JSONL/CSV via datasets")
    p.add_argument("--output", type=str, required=True, help="Output path (.jsonl or .csv)")
    p.add_argument("--n", type=int, default=200, help="Number of validation examples")
    p.add_argument("--split", type=str, default="validation", help="Dataset split")
    p.add_argument("--format", type=str, default="auto", choices=["auto", "jsonl", "csv"], help="Output format; default infers from extension")
    args = p.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Please `pip install datasets` to use this script.") from exc

    ds = load_dataset("sewon/ambig_qa", "light", split=args.split)
    take_n = min(int(args.n), len(ds))
    ds = ds.shuffle(seed=0).select(range(take_n))

    fmt = args.format
    if fmt == "auto":
        ext = os.path.splitext(args.output)[1].lower()
        fmt = "csv" if ext == ".csv" else "jsonl"

    rows: List[dict] = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        gold: List[str] = []
        ann = ex.get("annotations") or {}
        types = ann.get("type") or []
        answers_list = ann.get("answer") or []
        qa_pairs = ann.get("qaPairs") or []
        for i, t in enumerate(types):
            if t == "singleAnswer":
                try:
                    vals = answers_list[i]
                    if isinstance(vals, list):
                        for s in vals:
                            if isinstance(s, str) and s.strip():
                                gold.append(s.strip())
                except Exception:
                    pass
            else:
                try:
                    pair = qa_pairs[i]
                    ans = pair.get("answer") if isinstance(pair, dict) else None
                    if isinstance(ans, list):
                        for grp in ans:
                            if isinstance(grp, list):
                                for s in grp:
                                    if isinstance(s, str) and s.strip():
                                        gold.append(s.strip())
                except Exception:
                    pass
        if q and gold:
            seen = set(); uniq: List[str] = []
            for s in gold:
                if s not in seen:
                    seen.add(s); uniq.append(s)
            rows.append({"question": q, "answers": uniq})

    if fmt == "csv":
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["question", "answers"])
            w.writeheader()
            for r in rows:
                w.writerow({"question": r["question"], "answers": " | ".join(r["answers"])})
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()


