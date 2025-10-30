from __future__ import annotations

from typing import List, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_client_local import LocalLlamaClient


SYSTEM_PROMPT_DEFAULT = (
    "You are a helpful, concise assistant. Answer accurately. If unsure, say so briefly."
)

SYSTEM_PROMPT_STRICT = (
    "You are answering a multiple-choice question. "
    "Your response MUST be ONLY the letter of the correct answer (A, B, C, or D). "
    "Do NOT include any explanation, reasoning, or additional text. "
    "Output ONLY: A, B, C, or D."
)

SYSTEM_PROMPT_CODEBLOCK = (
    "You are answering a multiple-choice question. "
    "Put your answer (A, B, C, or D) inside triple backticks like this: ```A``` "
    "You may include brief explanation before or after the code block, but the answer MUST be in the code block."
)


def compose_prompt(
    query: str, 
    previous_answers: List[str], 
    *, 
    prompt_style: str = "naive",
    choices: List[str] = None,
    choice_texts: List[str] = None,
    answer_format: str = "default"
) -> List[dict]:
    """Compose messages for OpenAI-style chat with iterative previous answers in context.

    The paper's construction adds previous responses into the prompt. Here we place them
    in the user message as a short list of previously given answers to encourage (or test)
    sensitivity to prior outputs.
    
    Args:
        query: The question text
        previous_answers: List of previous answers in the chain
        prompt_style: Style of prompt ("naive", "wrong_prev", "critique")
        choices: List of choice letters (e.g., ["A", "B", "C", "D"])
        choice_texts: List of choice texts corresponding to letters
        answer_format: Format for answer ("default", "strict", "codeblock")
    """
    previous = [ans.strip() for ans in previous_answers]
    
    # Select system prompt based on answer format
    if answer_format == "strict":
        system_prompt = SYSTEM_PROMPT_STRICT
    elif answer_format == "codeblock":
        system_prompt = SYSTEM_PROMPT_CODEBLOCK
    else:
        system_prompt = SYSTEM_PROMPT_DEFAULT
    
    # Format MCQ choices if provided
    choices_text = ""
    if choices and choice_texts:
        choices_text = "\n\nChoices:\n" + "\n".join(
            f"{letter}) {text}" for letter, text in zip(choices, choice_texts)
        )

    if prompt_style == "wrong_prev":
        history = "".join(
            f"\nA wrong answer sometimes given is: {ans}" for ans in previous
        )
        user = (
            "Consider the following question (Q) and previous answers if any." +
            history +
            "\nPlease answer correctly." +
            f"\nQ: {query}{choices_text}\nA:"
        )
    elif prompt_style == "critique":
        history = "".join(
            f"\nEarlier answers may contain mistakes: {ans}" for ans in previous
        )
        user = (
            "Consider the following question (Q) and previous answers if any." +
            history +
            "\nProvide a corrected answer and a brief rationale." +
            f"\nQ: {query}{choices_text}\nA:"
        )
    else:
        # "naive" style: Use paper's exact MI chaining format
        # Paper format (line 264-272):
        # "Consider the following question:"
        # "Q: [query]"
        # "One answer to question Q is Y1. Another answer to question Q is Y2."
        # [Choices]
        # "Provide an answer to the following question:"
        # "Q: [query]"
        # "A:"
        
        if previous:
            # Match previous answers (letters like "B") to full choice text
            # So we show "B) quit eating lunch out" instead of just "B"
            full_previous = []
            for ans in previous:
                ans_upper = ans.strip().upper()
                # Find matching choice
                if choices and choice_texts:
                    matched = False
                    for letter, text in zip(choices, choice_texts):
                        if ans_upper == letter.upper() or ans_upper.startswith(letter.upper()):
                            # Show as "B) quit eating lunch out"
                            full_previous.append(f"{letter}) {text}")
                            matched = True
                            break
                    if not matched:
                        # Fallback: just use the answer if no match
                        full_previous.append(ans)
                else:
                    # No choices provided, use answer as-is
                    full_previous.append(ans)
            
            # Build history in paper's format: "One answer... Another answer..."
            if len(full_previous) == 1:
                history = f"\n\nOne answer to question Q is {full_previous[0]}.\n"
            else:
                history = f"\n\nOne answer to question Q is {full_previous[0]}."
                for ans in full_previous[1:]:
                    history += f" Another answer to question Q is {ans}."
                history += "\n"
        else:
            history = ""
        
        # Paper's exact format
        user = (
            f"Consider the following question:\n"
            f"Q: {query}\n"
            f"{history}"
            f"{choices_text}\n\n"
            f"Provide an answer to the following question:\n\n"
            f"Q: {query}\n\nA:"
        )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def run_chain_for_query(
    client,  # LocalLlamaClient or compatible
    query: str,
    chain_length: int,
    temperature: float,
    max_tokens: int,
    on_request: Optional[Callable[[], None]] = None,
    prompt_style: str = "naive",
) -> List[str]:
    answers: List[str] = []
    for _ in range(max(1, chain_length)):
        messages = compose_prompt(query, answers, prompt_style=prompt_style)
        ans = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
        if on_request is not None:
            on_request()
        answers.append(ans)
    return answers



def run_k_chains_for_query(
    client,  # LocalLlamaClient or compatible
    query: str,
    chain_length: int,
    k: int,
    temperature: float,
    max_tokens: int,
    on_request: Optional[Callable[[], None]] = None,
    prompt_style: str = "naive",
) -> List[List[str]]:
    """Run K independent chains for a single question.

    Each chain conditions subsequent answers on previous ones, but chains are
    independent across K repetitions.
    """
    chains: List[List[str]] = []
    for _ in range(max(1, k)):
        chains.append(
            run_chain_for_query(
                client=client,
                query=query,
                chain_length=chain_length,
                temperature=temperature,
                max_tokens=max_tokens,
                on_request=on_request,
                prompt_style=prompt_style,
            )
        )
    return chains


