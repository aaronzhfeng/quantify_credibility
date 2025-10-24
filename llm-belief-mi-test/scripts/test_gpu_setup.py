#!/usr/bin/env python
"""Test GPU setup and model loading with pseudo joint selection."""

import sys
import time
sys.path.insert(0, '/teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test')

print("="*60)
print("Testing GPU Setup with Llama-3.1-8B-Instruct")
print("="*60)

# Check GPU
import torch
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No GPU detected - will run on CPU (slow)")

# Load model
print("\n🔄 Loading model...")
from llm_belief_mi_test.llm_client_local import LocalLlamaClient

start = time.time()
client = LocalLlamaClient(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=torch.cuda.is_available()  # Only use 4-bit if GPU available
)
load_time = time.time() - start

print(f"✅ Model loaded in {load_time:.1f} seconds")

# Test with logprobs
print("\n🔄 Testing generation with logprobs...")
messages = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]

start = time.time()
response, logprob = client.chat_completion_with_logprobs(messages, temperature=0.0, max_tokens=10)
gen_time = time.time() - start

print(f"\nResponse: '{response}'")
print(f"Log probability: {logprob:.4f}")
print(f"Time: {gen_time:.2f} seconds")

# Test pseudo joint selection
print("\n🔄 Testing pseudo joint selection (k=5, n=2)...")
from llm_belief_mi_test.calibration import run_chain_with_logprobs, select_answer_via_pseudo_joint

question = "What is the capital of France?"
k, n = 5, 2

chains = []
start = time.time()
for i in range(k):
    chain = run_chain_with_logprobs(
        client=client,
        query=question,
        n=n,
        temperature=0.5,
        max_tokens=20,
        prompt_style="naive"
    )
    chains.append(chain)
    print(f"  Chain {i+1}: {[text for text, _ in chain]}")

total_time = time.time() - start

# Select answer via pseudo joint
predicted = select_answer_via_pseudo_joint(chains, n=n)
print(f"\nSelected answer (via marginalized pseudo joint): '{predicted}'")
print(f"Total time for k={k}, n={n}: {total_time:.2f} seconds")

# Performance estimate
print("\n📊 Performance Estimates:")
time_per_question = (total_time / 5) * 10  # Scale to k=10
print(f"  Per question (k=10, n=2): ~{time_per_question:.0f} seconds")
print(f"  50 questions: ~{time_per_question * 50 / 60:.1f} minutes")
print(f"  ARC-Challenge (1200): ~{time_per_question * 1200 / 3600:.1f} hours")

if gen_time < 2:
    print("\n✅ Performance: Excellent (GPU)")
elif gen_time < 10:
    print("\n⚠️  Performance: Acceptable")
else:
    print("\n❌ Performance: Slow (consider using GPU)")

print("\n" + "="*60)
print("✅ ALL TESTS PASSED")
print("="*60)
print("\nReady to run full evaluation!")
print("\nNext step:")
print("  python -m llm_belief_mi_test.cli \\")
print("    --dataset arc-easy --limit 10 --k 10 --n 2 \\")
print("    --load-in-4bit --output outputs/results/test.csv")

