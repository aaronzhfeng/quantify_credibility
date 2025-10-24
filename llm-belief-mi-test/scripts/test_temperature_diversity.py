#!/usr/bin/env python
"""Test that temperature creates diversity (important for MI estimation!)"""

import sys
sys.path.insert(0, '/teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test')

from llm_belief_mi_test.llm_client_local import LocalLlamaClient

print("="*70)
print("Testing Temperature Diversity (Critical for MI Method)")
print("="*70)

print("\nLoading model...")
client = LocalLlamaClient(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    cache=None  # Explicitly disable cache for this test
)

# Test question
messages = [{"role": "user", "content": "Name a color. Answer with one word."}]

print("\n" + "="*70)
print("Test 1: temperature=0.0 (should be identical)")
print("="*70)

responses_temp0 = []
for i in range(5):
    response, _ = client.chat_completion_with_logprobs(messages, temperature=0.0, max_tokens=10)
    responses_temp0.append(response)
    print(f"  {i+1}. '{response}'")

unique_temp0 = len(set(responses_temp0))
print(f"\nUnique answers: {unique_temp0}/5")
if unique_temp0 == 1:
    print("✅ PASS: All identical (expected for temperature=0.0)")
else:
    print("⚠️  UNEXPECTED: Should all be identical with temperature=0.0")

print("\n" + "="*70)
print("Test 2: temperature=0.3 (should be diverse)")
print("="*70)

responses_temp03 = []
for i in range(10):
    response, _ = client.chat_completion_with_logprobs(messages, temperature=0.3, max_tokens=10)
    responses_temp03.append(response)
    print(f"  {i+1}. '{response}'")

unique_temp03 = len(set(responses_temp03))
print(f"\nUnique answers: {unique_temp03}/10")

if unique_temp03 >= 3:
    print(f"✅ PASS: Good diversity ({unique_temp03} unique answers)")
elif unique_temp03 >= 2:
    print(f"⚠️  WARNING: Low diversity ({unique_temp03} unique answers)")
    print("   Expected at least 3-4 unique answers with temp=0.3")
else:
    print(f"❌ FAIL: No diversity! All chains will be identical!")
    print("   This breaks the MI method - chains must be different")

print("\n" + "="*70)
print("Test 3: Simulate k=10 chains for MI")
print("="*70)

from llm_belief_mi_test.calibration import run_chain_with_logprobs
from llm_belief_mi_test.mi_estimator import estimate_mi_listing_nats, nats_to_bits

question = "What is the capital of France?"
chains = []

print(f"\nGenerating 10 chains of length 2...")
for i in range(10):
    chain = run_chain_with_logprobs(
        client=client,
        query=question,
        n=2,
        temperature=0.3,
        max_tokens=10,
        prompt_style="naive"
    )
    chains.append(chain)
    final_answer = chain[-1][0]
    print(f"  Chain {i+1}: Y1='{chain[0][0]}', Y2='{final_answer}'")

# Extract just text for MI computation
chains_text = [[text for text, _ in chain] for chain in chains]

# Compute MI
mi_nats = estimate_mi_listing_nats(chains_text)
mi_bits = nats_to_bits(mi_nats)

# Compute agreement
final_answers = [chain[-1][0] for chain in chains]
from collections import Counter
counts = Counter(final_answers)
most_common_count = counts.most_common(1)[0][1] if counts else 0
agreement = most_common_count / len(final_answers)

print(f"\nResults:")
print(f"  MI: {mi_bits:.4f} bits")
print(f"  Agreement: {agreement:.2f}")
print(f"  Unique final answers: {len(set(final_answers))}")

print("\n" + "="*70)
if mi_bits > 0.1 and agreement < 0.9:
    print("✅ SUCCESS: MI method working correctly!")
    print("   - MI > 0 (chains show dependence)")
    print("   - Agreement < 1.0 (chains are diverse)")
    print("\n   The bug is FIXED! You can proceed with evaluation.")
elif mi_bits == 0.0 and agreement == 1.0:
    print("❌ FAIL: All chains identical - cache bug still present!")
    print("   Do NOT proceed - fix needed")
else:
    print(f"⚠️  BORDERLINE: MI={mi_bits:.4f}, Agreement={agreement:.2f}")
    print("   May work, but diversity could be better")

print("="*70)

