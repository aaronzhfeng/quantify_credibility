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


def answers_match(prediction: str, gold_answers: Sequence[str]) -> bool:
    if not gold_answers:
        return False
    pred_n = normalize_answer(prediction)
    gold_n = {normalize_answer(a) for a in gold_answers}
    return pred_n in gold_n


