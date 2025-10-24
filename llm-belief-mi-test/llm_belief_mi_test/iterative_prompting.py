from __future__ import annotations

from typing import List, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_client_local import LocalLlamaClient


SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer accurately. If unsure, say so briefly."
)


def compose_prompt(query: str, previous_answers: List[str], *, prompt_style: str = "naive") -> List[dict]:
    """Compose messages for OpenAI-style chat with iterative previous answers in context.

    The paper's construction adds previous responses into the prompt. Here we place them
    in the user message as a short list of previously given answers to encourage (or test)
    sensitivity to prior outputs.
    """
    previous = [ans.strip() for ans in previous_answers]

    if prompt_style == "wrong_prev":
        history = "".join(
            f"\nA wrong answer sometimes given is: {ans}" for ans in previous
        )
        user = (
            "Consider the following question (Q) and previous answers if any." +
            history +
            "\nPlease answer correctly." +
            f"\nQ: {query}\nA:"
        )
    elif prompt_style == "critique":
        history = "".join(
            f"\nEarlier answers may contain mistakes: {ans}" for ans in previous
        )
        user = (
            "Consider the following question (Q) and previous answers if any." +
            history +
            "\nProvide a corrected answer and a brief rationale." +
            f"\nQ: {query}\nA:"
        )
    else:
        history = "".join(
            f"\nAnother answer to question Q is: {ans}" for ans in previous
        )
        user = f"Consider the following question (Q) and previous answers if any.{history}\nProvide an answer to the following question:\nQ: {query}\nA:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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


