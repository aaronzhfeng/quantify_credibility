from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def load_toy_questions() -> List[str]:
    # Simple, factual and multi-label-ish prompts to exercise the pipeline
    return [
        "What is the capital of the UK?",
        "Name a city in the UK",
        "Name a yellow fruit",
        "Name an alcoholic drink",
        "Who was the first US president?",
        "Which actor became M in the Bond film Skyfall?",
        "Which can last longer without water: a camel or a rat?",
        "If Monday’s child is fair of face what is Saturday’s child?",
        "What is the largest country in the world?",
        "Who is the author of The Grapes of Wrath?",
    ]


# --- TriviaQA subset loader -------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Normalize answers for rough exact-match comparison.

    Lowercases, strips punctuation, articles, and extra whitespace.
    """
    s = text.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s)
    return s.strip()


@dataclass
class QAExample:
    question: str
    answers: List[str]  # one or more canonical answers


def _coerce_answers(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str):
                out.append(x)
        return out
    return []


def load_triviaqa_subset(input_path: str, limit: int | None = None) -> List[QAExample]:
    """Load a TriviaQA subset from JSONL or CSV.

    Expected fields:
    - question: str
    - answers or answer: list[str] or str
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"TriviaQA subset not found: {input_path}")

    examples: List[QAExample] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = str(obj.get("question", "")).strip()
                ans = _coerce_answers(obj.get("answers") or obj.get("answer"))
                if q and ans:
                    examples.append(QAExample(question=q, answers=ans))
                if limit is not None and len(examples) >= limit:
                    break
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = str(row.get("question", "")).strip()
                # answers can be pipe- or semicolon-separated; fall back to single 'answer'
                answers_field = row.get("answers") or row.get("answer") or ""
                if isinstance(answers_field, str):
                    # split on common separators
                    parts = [p.strip() for p in re.split(r"[|;]", answers_field) if p.strip()]
                else:
                    parts = []
                if q and parts:
                    examples.append(QAExample(question=q, answers=parts))
                if limit is not None and len(examples) >= limit:
                    break
    else:
        raise ValueError("Unsupported TriviaQA subset format (use .jsonl or .csv)")

    return examples


# --- AmbigQA loader (HF or JSONL) -------------------------------------------

def load_ambigqa_subset(input_path: str | None = None, split: str = "validation", limit: int | None = None) -> List[QAExample]:
    """Load AmbigQA from HF datasets (preferred), CSV, or JSONL.

    HF schema fields (ambig_qa):
    - question: str
    - annotations: list with possible answers (string lists)
    We flatten all gold answers from annotations.

    CSV/JSONL fallback (local files):
    - expect fields: question, answers (pipe- or semicolon-separated for CSV; list[str] for JSONL)
    """
    examples: List[QAExample] = []
    if input_path is None:
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise SystemExit("Please provide input_path for AmbigQA or install datasets.") from exc
        # Use the namespaced repo id and the "light" config by default
        ds = load_dataset("sewon/ambig_qa", "light", split=split)
        n = len(ds) if limit is None else min(limit, len(ds))
        for ex in ds.select(range(n)):
            q = str(ex.get("question", "")).strip()
            gold: List[str] = []
            ann = ex.get("annotations") or {}
            types = ann.get("type") or []
            answers_list = ann.get("answer") or []  # sequence of sequence[str]
            qa_pairs = ann.get("qaPairs") or []     # sequence of {question, answer}
            # Iterate over annotation indices and collect all strings
            for i, t in enumerate(types):
                if t == "singleAnswer":
                    # answers_list[i] is a list of strings; sometimes nested lists — flatten defensively
                    try:
                        vals = answers_list[i]
                        if isinstance(vals, list):
                            for s in vals:
                                if isinstance(s, str) and s.strip():
                                    gold.append(s)
                    except Exception:
                        pass
                else:
                    # multipleQAs: qa_pairs[i]['answer'] is list[list[str]]; flatten
                    try:
                        pair = qa_pairs[i]
                        ans = pair.get("answer") if isinstance(pair, dict) else None
                        if isinstance(ans, list):
                            for grp in ans:
                                if isinstance(grp, list):
                                    for s in grp:
                                        if isinstance(s, str) and s.strip():
                                            gold.append(s)
                    except Exception:
                        pass
            # de-duplicate while preserving order
            if q and gold:
                seen = set()
                uniq: List[str] = []
                for s in gold:
                    if s not in seen:
                        seen.add(s)
                        uniq.append(s)
                examples.append(QAExample(question=q, answers=uniq))
        return examples
    # Local file fallback: JSONL or CSV
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"AmbigQA file not found: {input_path}")
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = str(obj.get("question", "")).strip()
                ans = _coerce_answers(obj.get("answers"))
                if q and ans:
                    examples.append(QAExample(question=q, answers=ans))
                if limit is not None and len(examples) >= limit:
                    break
        return examples
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = str(row.get("question", "")).strip()
                answers_field = row.get("answers") or row.get("answer") or ""
                if isinstance(answers_field, str):
                    parts = [p.strip() for p in re.split(r"[|;]", answers_field) if p.strip()]
                else:
                    parts = []
                if q and parts:
                    examples.append(QAExample(question=q, answers=parts))
                if limit is not None and len(examples) >= limit:
                    break
        return examples
    raise ValueError("Unsupported AmbigQA format; use HF datasets, JSONL, or CSV")


def answers_match(prediction: str, gold_answers: Sequence[str]) -> bool:
    if not gold_answers:
        return False
    pred_n = normalize_answer(prediction)
    gold_n = {normalize_answer(a) for a in gold_answers}
    return pred_n in gold_n


