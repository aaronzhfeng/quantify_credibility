from __future__ import annotations

from typing import List

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
) -> List[str]:
    answers: List[str] = []
    for _ in range(max(1, chain_length)):
        messages = compose_prompt(query, answers)
        ans = client.chat_completion(messages, temperature=temperature, max_tokens=max_tokens)
        answers.append(ans)
    return answers


