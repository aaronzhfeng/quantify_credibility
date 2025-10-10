from __future__ import annotations

from typing import List


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


