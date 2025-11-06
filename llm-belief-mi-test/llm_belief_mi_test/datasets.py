from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
from difflib import SequenceMatcher
import re
import string


@dataclass
class MCQExample:
    """Multiple-choice question example."""
    question: str
    choices: List[str]  # e.g., ["A", "B", "C", "D"]
    choice_texts: List[str]  # e.g., ["option A text", ...]
    answer_key: str  # e.g., "C"
    answer_index: int  # e.g., 2 (0-indexed)
    metadata: Dict[str, Any] | None = None  # Optional metadata for MC2, etc.


@dataclass
class ExtractiveQAExample:
    """Extractive question answering example (SQuAD, TriviaQA, etc.)."""
    id: str
    question: str
    context: str  # Background paragraph
    answers: List[str]  # Multiple acceptable answers (or empty for unanswerable)
    is_impossible: bool = False  # True for SQuAD v2 unanswerable questions
    
    @property
    def has_answer(self) -> bool:
        return len(self.answers) > 0 and not self.is_impossible


def load_arc_challenge(split: str = "test", limit: int | None = None, offset: int = 0) -> List[MCQExample]:
    """Load ARC-Challenge dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if limit or offset:
        start_idx = offset
        end_idx = offset + limit if limit else len(ds)
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
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


def load_arc_easy(split: str = "test", limit: int | None = None, offset: int = 0) -> List[MCQExample]:
    """Load ARC-Easy dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split)
    if limit or offset:
        start_idx = offset
        end_idx = offset + limit if limit else len(ds)
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
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


def load_openbookqa(split: str = "test", limit: int | None = None, offset: int = 0) -> List[MCQExample]:
    """Load OpenBookQA dataset."""
    from datasets import load_dataset
    
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    if limit or offset:
        start_idx = offset
        end_idx = offset + limit if limit else len(ds)
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
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


def extract_answer_from_codeblock(text: str, choices: List[str]) -> str:
    """
    Extract answer from codeblock format like ```A```
    
    Args:
        text: Generated text that may contain ```letter```
        choices: Valid choice letters
        
    Returns:
        Extracted letter or None
    """
    import re
    
    # Look for ```X``` pattern
    pattern = r'```\s*([A-Za-z])\s*```'
    matches = re.findall(pattern, text)
    
    if matches:
        for match in matches:
            letter = match.upper()
            if letter in choices:
                return letter
    
    return None


def extract_answer_strict(text: str, choices: List[str]) -> str:
    """
    Extract answer from strict format (should be just "A", "B", "C", or "D")
    
    Args:
        text: Generated text (should be single letter)
        choices: Valid choice letters
        
    Returns:
        Extracted letter or None
    """
    # Clean the text
    cleaned = text.strip().upper()
    
    # Check if it's exactly one of the choices
    if cleaned in choices:
        return cleaned
    
    # Check if first character is a choice
    if cleaned and cleaned[0] in choices:
        return cleaned[0]
    
    return None


def match_answer_to_choices(
    generated_answer: str,
    choice_texts: List[str],
    choices: List[str],
    answer_format: str = "default"
) -> str:
    """
    Match a generated answer to one of the multiple choices.
    
    Args:
        generated_answer: Text generated by model
        choice_texts: List of choice text descriptions
        choices: List of choice letters (e.g., ["A", "B", "C", "D"])
        answer_format: Format used ("default", "strict", "codeblock")
        
    Returns:
        Matched choice letter
    """
    generated = generated_answer.strip()
    
    # Try format-specific extraction first
    if answer_format == "codeblock":
        extracted = extract_answer_from_codeblock(generated, choices)
        if extracted:
            return extracted
    elif answer_format == "strict":
        extracted = extract_answer_strict(generated, choices)
        if extracted:
            return extracted
    
    # Fallback to fuzzy matching
    generated_lower = generated.lower()
    
    # First, check if answer contains choice letter at start
    for letter in choices:
        if generated_lower.startswith(letter.lower()):
            return letter
        # Check for pattern like "A)" or "A:"
        if generated_lower.startswith(f"{letter.lower()})") or generated_lower.startswith(f"{letter.lower()}:"):
            return letter
    
    # Check first 20 chars for letter
    for letter in choices:
        if letter.lower() in generated_lower[:20]:
            return letter
    
    # Then, fuzzy match against choice texts
    best_match = None
    best_score = 0.0
    
    for choice_letter, choice_text in zip(choices, choice_texts):
        choice_lower = choice_text.lower().strip()
        
        # Check substring match
        if choice_lower in generated_lower or generated_lower in choice_lower:
            return choice_letter
        
        # Compute similarity
        similarity = SequenceMatcher(None, generated_lower, choice_lower).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = choice_letter
    
    # Return best match or first choice as fallback
    return best_match if best_match else choices[0]


def generate_choice_letters(num_choices: int) -> List[str]:
    """
    Generate choice letters dynamically: A, B, ..., Z, AA, AB, ..., AZ, BA, ...
    
    Args:
        num_choices: Number of choice letters to generate
        
    Returns:
        List of choice letters
    """
    letters = []
    for i in range(num_choices):
        if i < 26:
            # A-Z (0-25)
            letters.append(chr(65 + i))
        else:
            # AA, AB, ..., AZ, BA, BB, ... (26+)
            first = chr(65 + (i - 26) // 26)
            second = chr(65 + (i - 26) % 26)
            letters.append(first + second)
    return letters


def is_answer_correct_mc2(predicted_choice: str, example: MCQExample) -> bool:
    """
    Check if prediction is correct for MC2 (multi-label) format.
    
    For MC2: Answer is correct if it matches ANY of the true answers.
    For MC1 (no metadata): Falls back to standard single-answer check.
    
    Args:
        predicted_choice: The predicted choice letter (e.g., "B")
        example: MCQExample instance (possibly with MC2 metadata)
        
    Returns:
        True if prediction is correct, False otherwise
    """
    if example.metadata and "correct_choices" in example.metadata:
        # MC2: Check if predicted choice is in the set of all correct choices
        return predicted_choice in example.metadata["correct_choices"]
    else:
        # MC1 or no metadata: Standard single-answer check
        return predicted_choice == example.answer_key


def load_truthfulqa_mc1(split: str = "validation", limit: int | None = None, offset: int = 0) -> List[MCQExample]:
    """
    Load TruthfulQA MC1 (single-true) dataset.
    
    MC1 format: Each question has multiple answer choices with exactly one correct answer.
    Tests whether model can identify truthful statements vs common misconceptions.
    
    Note: Number of choices varies per question (can exceed 5).
    
    Args:
        split: Dataset split (TruthfulQA only has "validation", ignores this parameter)
        limit: Optional limit on number of examples
        offset: Skip first N examples (for multi-GPU parallelism)
        
    Returns:
        List of MCQExample instances
    """
    from datasets import load_dataset
    
    # Load TruthfulQA multiple-choice dataset
    # Note: TruthfulQA only has a "validation" split, ignore the split parameter
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if limit or offset:
        start_idx = offset
        end_idx = offset + limit if limit else len(ds)
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
    examples = []
    for ex in ds:
        # MC1 structure:
        # - mc1_targets: {"choices": [...], "labels": [1, 0, 0, 0]}
        # - labels: 1 = true/correct answer, 0 = false/incorrect answer
        mc1_choices = ex["mc1_targets"]["choices"]
        mc1_labels = ex["mc1_targets"]["labels"]
        
        # Find the single correct answer (label=1)
        try:
            correct_idx = mc1_labels.index(1)
        except ValueError:
            # Skip examples without a correct answer
            continue
        
        # Generate choice letters dynamically (A-Z, then AA, AB, ...)
        num_choices = len(mc1_choices)
        choices = generate_choice_letters(num_choices)
        
        examples.append(MCQExample(
            question=ex["question"],
            choices=choices,
            choice_texts=mc1_choices,
            answer_key=choices[correct_idx],
            answer_index=correct_idx
        ))
    
    return examples


def load_truthfulqa_mc2(split: str = "validation", limit: int | None = None, offset: int = 0) -> List[MCQExample]:
    """
    Load TruthfulQA MC2 (multi-true) dataset.
    
    MC2 format: Each question has MULTIPLE correct answers (≥1).
    Tests whether model can identify ALL truthful statements.
    This is harder than MC1 because the model must recognize multiple truths.
    
    Note: Number of choices varies per question (can exceed 5).
    
    Args:
        split: Dataset split (TruthfulQA only has "validation", ignores this parameter)
        limit: Optional limit on number of examples
        offset: Skip first N examples (for multi-GPU parallelism)
        
    Returns:
        List of MCQExample instances with metadata containing all correct answers
    """
    from datasets import load_dataset
    
    # Load TruthfulQA multiple-choice dataset
    # Note: TruthfulQA only has a "validation" split, ignore the split parameter
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    if limit or offset:
        start_idx = offset
        end_idx = offset + limit if limit else len(ds)
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
    examples = []
    for ex in ds:
        # MC2 structure:
        # - mc2_targets: {"choices": [...], "labels": [1, 1, 0, 1, 0, ...]}
        # - labels: 1 = true answer, 0 = false answer (MULTIPLE 1s possible)
        mc2_choices = ex["mc2_targets"]["choices"]
        mc2_labels = ex["mc2_targets"]["labels"]
        
        # Find ALL correct answer indices
        correct_indices = [i for i, label in enumerate(mc2_labels) if label == 1]
        
        if not correct_indices:
            # Skip examples without any correct answers
            continue
        
        # Generate choice letters dynamically (A-Z, then AA, AB, ...)
        num_choices = len(mc2_choices)
        choices = generate_choice_letters(num_choices)
        
        # Get all correct choice letters
        correct_choices = [choices[i] for i in correct_indices]
        
        # Store first correct answer as primary (for compatibility with single-answer APIs)
        # But also store all correct indices/choices in metadata for MC2 evaluation
        examples.append(MCQExample(
            question=ex["question"],
            choices=choices,
            choice_texts=mc2_choices,
            answer_key=correct_choices[0],  # Primary answer (first correct one)
            answer_index=correct_indices[0],
            metadata={
                "correct_indices": correct_indices,
                "correct_choices": correct_choices,
                "mc2_labels": mc2_labels,
                "num_correct": len(correct_indices),
                "num_incorrect": mc2_labels.count(0)
            }
        ))
    
    return examples


def load_squad_v2(split: str = "validation", limit: int | None = None, offset: int = 0) -> List[ExtractiveQAExample]:
    """
    Load SQuAD v2 dataset.
    
    Args:
        split: "train" or "validation"
        limit: Optional limit on number of examples
        offset: Skip first N examples (for multi-GPU parallelism)
        
    Returns:
        List of ExtractiveQAExample instances
    """
    from datasets import load_dataset
    
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    if limit or offset:
        start_idx = offset
        end_idx = offset + limit if limit else len(ds)
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
    examples = []
    for ex in ds:
        # SQuAD v2 structure:
        # - answers: {"text": [...], "answer_start": [...]}
        # - Empty answers list means unanswerable
        answer_texts = ex["answers"]["text"]
        is_impossible = len(answer_texts) == 0
        
        examples.append(ExtractiveQAExample(
            id=ex["id"],
            question=ex["question"],
            context=ex["context"],
            answers=answer_texts if answer_texts else [],
            is_impossible=is_impossible
        ))
    
    return examples


def load_triviaqa(split: str = "validation", limit: int | None = None, offset: int = 0) -> List[ExtractiveQAExample]:
    """
    Load TriviaQA dataset (rc.nocontext subset - no evidence documents).
    
    TriviaQA is an open-domain question answering dataset with trivia questions.
    Each question has multiple acceptable answer aliases (e.g., "Sinclair Lewis", 
    "Harry Sinclair Lewis", "Lewis, (Harry) Sinclair").
    
    We use the "rc.nocontext" subset which excludes the search results/evidence
    documents to keep prompts short and focus on pure knowledge testing.
    
    Args:
        split: "train", "validation", or "test"
        limit: Optional limit on number of examples
        offset: Skip first N examples (for multi-GPU parallelism)
        
    Returns:
        List of ExtractiveQAExample instances
    """
    from datasets import load_dataset
    
    # Load the rc.nocontext subset (no search results/evidence documents)
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    if limit or offset:
        start_idx = offset
        end_idx = offset + limit if limit else len(ds)
        ds = ds.select(range(start_idx, min(end_idx, len(ds))))
    
    examples = []
    for ex in ds:
        # TriviaQA structure:
        # - question: The trivia question
        # - question_id: Unique identifier
        # - answer: dict with "value", "aliases", "normalized_aliases", etc.
        #   - "aliases" contains multiple acceptable answer forms
        #   - e.g., ["Sinclair Lewis", "Harry Sinclair Lewis", "Lewis, (Harry) Sinclair"]
        
        # Use all aliases as acceptable answers
        answer_aliases = ex["answer"]["aliases"]
        
        examples.append(ExtractiveQAExample(
            id=ex["question_id"],
            question=ex["question"],
            context="",  # No context in nocontext subset
            answers=answer_aliases,  # Multiple acceptable answer forms
            is_impossible=False  # TriviaQA questions are always answerable
        ))
    
    return examples


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison (SQuAD evaluation standard).
    
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = ''.join(ch if ch not in string.punctuation else ' ' for ch in text)
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute exact match score (0 or 1).
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable answers (empty for unanswerable)
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not ground_truths:  # Unanswerable question
        # Check if model correctly abstained
        normalized_pred = normalize_answer(prediction)
        return 1.0 if normalized_pred in ["unanswerable", "no answer", "cannot answer", ""] else 0.0
    
    normalized_pred = normalize_answer(prediction)
    
    for ground_truth in ground_truths:
        if normalized_pred == normalize_answer(ground_truth):
            return 1.0
    
    return 0.0


def compute_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """
    Compute F1 score (token-level overlap).
    
    Standard SQuAD evaluation metric.
    
    Args:
        prediction: Model's predicted answer
        ground_truths: List of acceptable answers (empty for unanswerable)
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not ground_truths:  # Unanswerable
        normalized_pred = normalize_answer(prediction)
        return 1.0 if normalized_pred in ["unanswerable", "no answer", "cannot answer", ""] else 0.0
    
    # Compute F1 against all ground truths, take maximum
    f1_scores = []
    normalized_pred = normalize_answer(prediction)
    pred_tokens = normalized_pred.split()
    
    for ground_truth in ground_truths:
        gt_tokens = normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        # Compute overlap
        common_tokens = set(pred_tokens) & set(gt_tokens)
        
        if len(common_tokens) == 0:
            f1_scores.append(0.0)
            continue
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(gt_tokens)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    return max(f1_scores) if f1_scores else 0.0

