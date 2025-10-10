from __future__ import annotations

from typing import List, Callable, Optional

from .llm_client import OpenAICompatibleLLMClient


SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. Answer accurately. If unsure, say so briefly."
)


def compose_prompt(query: str, previous_answers: List[str]) -> List[dict]:
    """Compose messages for OpenAI-style chat with iterative previous answers in context.

    The paper's construction adds previous responses into the prompt. Here we place them
    in the user message as a short list of previously given answers to encourage (or test)
    sensitivity to prior outputs.
    """
    history = "".join(
        f"\nAnother answer to question Q is: {ans.strip()}" for ans in previous_answers
    )
    user = f"Consider the following question (Q) and previous answers if any.{history}\nProvide an answer to the following question:\nQ: {query}\nA:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def run_chain_for_query(
    client: OpenAICompatibleLLMClient,
    query: str,
    chain_length: int,
    temperature: float,
    max_tokens: int,
    on_request: Optional[Callable[[], None]] = None,
) -> List[str]:
    answers: List[str] = []
    for _ in range(max(1, chain_length)):
        messages = compose_prompt(query, answers)
        ans = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
        if on_request is not None:
            on_request()
        answers.append(ans)
    return answers



def run_k_chains_for_query(
    client: OpenAICompatibleLLMClient,
    query: str,
    chain_length: int,
    k: int,
    temperature: float,
    max_tokens: int,
    on_request: Optional[Callable[[], None]] = None,
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
            )
        )
    return chains


