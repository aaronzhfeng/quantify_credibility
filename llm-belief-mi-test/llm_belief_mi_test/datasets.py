from __future__ import annotations

from dataclasses import dataclass
from typing import List
from difflib import SequenceMatcher


@dataclass
class MCQExample:
    """Multiple-choice question example."""
    question: str
    choices: List[str]  # e.g., ["A", "B", "C", "D"]
    choice_texts: List[str]  # e.g., ["option A text", ...]
    answer_key: str  # e.g., "C"
    answer_index: int  # e.g., 2 (0-indexed)


def load_arc_challenge(split: str = "test", limit: int | None = None) -> List[MCQExample]:
    """Load ARC-Challenge dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    examples = []
    for ex in ds:
        choices = ex["choices"]["label"]
        choice_texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        answer_idx = choices.index(answer_key)
        
        examples.append(MCQExample(
            question=ex["question"],
            choices=choices,
            choice_texts=choice_texts,
            answer_key=answer_key,
            answer_index=answer_idx
        ))
    
    return examples


def load_arc_easy(split: str = "test", limit: int | None = None) -> List[MCQExample]:
    """Load ARC-Easy dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    examples = []
    for ex in ds:
        choices = ex["choices"]["label"]
        choice_texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        answer_idx = choices.index(answer_key)
        
        examples.append(MCQExample(
            question=ex["question"],
            choices=choices,
            choice_texts=choice_texts,
            answer_key=answer_key,
            answer_index=answer_idx
        ))
    
    return examples


def load_openbookqa(split: str = "test", limit: int | None = None) -> List[MCQExample]:
    """Load OpenBookQA dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    
    examples = []
    for ex in ds:
        choices = ex["choices"]["label"]
        choice_texts = ex["choices"]["text"]
        answer_key = ex["answerKey"]
        answer_idx = choices.index(answer_key)
        
        examples.append(MCQExample(
            question=ex["question_stem"],
            choices=choices,
            choice_texts=choice_texts,
            answer_key=answer_key,
            answer_index=answer_idx
        ))
    
    return examples


def match_answer_to_choices(
    generated_answer: str,
    choice_texts: List[str],
    choices: List[str]
) -> str:
    """
    Match a generated answer to one of the multiple choices.
    
    Uses fuzzy string matching to find the best match.
    """
    generated = generated_answer.lower().strip()
    
    # First, check if answer contains choice letter
    for letter in choices:
        if letter.lower() in generated[:10]:  # Check first 10 chars
            return letter
    
    # Then, fuzzy match against choice texts
    best_match = None
    best_score = 0.0
    
    for choice_letter, choice_text in zip(choices, choice_texts):
        choice_lower = choice_text.lower().strip()
        
        # Check substring match
        if choice_lower in generated or generated in choice_lower:
            return choice_letter
        
        # Compute similarity
        similarity = SequenceMatcher(None, generated, choice_lower).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = choice_letter
    
    # Return best match or first choice as fallback
    return best_match if best_match else choices[0]

