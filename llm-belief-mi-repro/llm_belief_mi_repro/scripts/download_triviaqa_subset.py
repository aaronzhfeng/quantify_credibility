from __future__ import annotations

import argparse
import csv
from typing import List


def coerce_answers(ans_obj) -> List[str]:
    if ans_obj is None:
        return []
    out: List[str] = []
    if isinstance(ans_obj, dict):
        v = ans_obj.get("value")
        if v:
            out.append(str(v))
        out += [str(a) for a in (ans_obj.get("aliases") or [])]
        out += [str(a) for a in (ans_obj.get("normalized_aliases") or [])]
    elif isinstance(ans_obj, list):
        out += [str(a) for a in ans_obj]
    elif isinstance(ans_obj, str):
        out.append(ans_obj)
    return [s.strip() for s in out if s and s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TriviaQA validation subset to CSV/JSONL via datasets")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--n", type=int, default=200, help="Number of validation examples")
    parser.add_argument("--config", type=str, default="rc", help="trivia_qa configuration (rc|unfiltered)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Please `pip install datasets` to use this script.") from exc

    ds = load_dataset("trivia_qa", args.config, split="validation")
    take_n = min(int(args.n), len(ds))
    ds = ds.shuffle(seed=0).select(range(take_n))

    rows = []
    for ex in ds:
        q = (ex.get("question") or "").strip()
        ans = ex.get("answer") or ex.get("answers")
        answers = coerce_answers(ans)
        if q and answers:
            # dedup while preserving order
            seen = set()
            uniq: List[str] = []
            for a in answers:
                if a not in seen:
                    seen.add(a)
                    uniq.append(a)
            rows.append({"question": q, "answers": " | ".join(uniq)})

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answers"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()


