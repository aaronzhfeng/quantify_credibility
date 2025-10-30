#!/usr/bin/env python3
"""
Test different answer formats to verify prompt formatting and answer extraction.

Usage:
    python scripts/test_answer_formats.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_belief_mi_test.llm_client_local import LocalLlamaClient
from llm_belief_mi_test.datasets import load_openbookqa
from llm_belief_mi_test.iterative_prompting import compose_prompt
from llm_belief_mi_test.datasets import match_answer_to_choices


def test_answer_formats():
    """Test all three answer formats."""
    
    print("="*80)
    print("TESTING ANSWER FORMATS")
    print("="*80)
    
    # Load one example
    examples = load_openbookqa("test", limit=1)
    ex = examples[0]
    
    print(f"\nQuestion: {ex.question}")
    print(f"Choices: {ex.choices}")
    print(f"Gold: {ex.answer_key}\n")
    
    # Initialize model
    print("Loading model...")
    client = LocalLlamaClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=True,
        cache=None
    )
    print("Model loaded\n")
    
    # Test each format
    formats = ["default", "strict", "codeblock"]
    
    for fmt in formats:
        print(f"\n{'='*80}")
        print(f"FORMAT: {fmt.upper()}")
        print(f"{'='*80}")
        
        # Compose prompt
        messages = compose_prompt(
            ex.question, [],
            prompt_style="naive",
            choices=ex.choices,
            choice_texts=ex.choice_texts,
            answer_format=fmt
        )
        
        # Show system prompt
        print(f"\nSystem Prompt:")
        print(f"  {messages[0]['content'][:100]}...")
        
        # Show user prompt (first 300 chars)
        print(f"\nUser Prompt (first 300 chars):")
        print(f"  {messages[1]['content'][:300]}...")
        
        # Get response
        print(f"\nGenerating response...")
        response, logprob = client.chat_completion_with_logprobs(
            messages,
            temperature=0.0,
            max_tokens=50
        )
        
        print(f"\nModel Response:")
        print(f"  '{response}'")
        
        # Extract answer
        matched = match_answer_to_choices(
            response,
            ex.choice_texts,
            ex.choices,
            answer_format=fmt
        )
        
        print(f"\nExtracted Answer: {matched}")
        print(f"Gold Answer: {ex.answer_key}")
        print(f"Correct: {matched == ex.answer_key}")
        print(f"Confidence: {math.exp(logprob):.6f}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import math
    test_answer_formats()

