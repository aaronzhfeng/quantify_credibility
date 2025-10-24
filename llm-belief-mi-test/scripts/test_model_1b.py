#!/usr/bin/env python
"""Test script with Llama-3.2-1B-Instruct (smaller model for CPU)."""

import sys
import time
sys.path.insert(0, '/teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test')

from llm_belief_mi_test.llm_client_local import LocalLlamaClient

print("="*60)
print("Testing Local Llama Client with 1B Model")
print("="*60)

print("\n🔄 Initializing client with Llama-3.2-1B-Instruct...")
print("   Model size: ~2-4 GB (much smaller than 8B)")
print("   First run will download the model")

start_time = time.time()

try:
    client = LocalLlamaClient(
        model_name="meta-llama/Llama-3.2-1B-Instruct",  # Smaller model
        use_cpu=True
    )
    
    load_time = time.time() - start_time
    print(f"\n✅ Model loaded in {load_time:.1f} seconds")
    
except Exception as e:
    print(f"\n❌ Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection")
    print("2. Check disk space")
    print("3. Set HF_TOKEN if needed: export HF_TOKEN='your_token'")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test generation
print("\n🔄 Testing chat completion...")

test_cases = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        "expected": "4"
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."}
        ],
        "expected": "Paris"
    }
]

total_time = 0
success_count = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n  Test {i}: {test['messages'][1]['content']}")
    
    try:
        gen_start = time.time()
        response = client.chat_completion(
            test["messages"],
            temperature=0.0,
            max_tokens=20
        )
        gen_time = time.time() - gen_start
        total_time += gen_time
        
        print(f"    Response: '{response}'")
        print(f"    Time: {gen_time:.1f}s")
        
        # Check if response contains expected answer (flexible matching)
        if test["expected"].lower() in response.lower():
            print(f"    ✅ Correct")
            success_count += 1
        else:
            print(f"    ⚠️  Expected '{test['expected']}' but got '{response}'")
        
    except Exception as e:
        print(f"    ❌ Generation failed: {e}")

avg_time = total_time / len(test_cases)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Tests passed: {success_count}/{len(test_cases)}")
print(f"Average generation time: {avg_time:.1f} seconds")

# Performance assessment
if avg_time < 5:
    perf_level = "✅ Excellent"
    recommendation = "Full evaluation is feasible!"
elif avg_time < 10:
    perf_level = "✅ Good"
    recommendation = "Can run medium-scale evaluation (100-500 examples)"
elif avg_time < 20:
    perf_level = "⚠️  Acceptable"
    recommendation = "Start with small samples (50-100 examples)"
else:
    perf_level = "❌ Too Slow"
    recommendation = "Consider GPU environment or API-based approach"

print(f"Performance: {perf_level}")
print(f"Recommendation: {recommendation}")

# Estimate full run time
print(f"\n📊 Estimated time for evaluation (K=10, t=3):")
time_per_question = avg_time * 10 * 3  # K chains × t length
print(f"   Per question: ~{time_per_question:.0f} seconds ({time_per_question/60:.1f} min)")
print(f"   50 questions: ~{time_per_question * 50 / 3600:.1f} hours")
print(f"   ARC-Challenge (1200): ~{time_per_question * 1200 / 3600:.1f} hours")
print(f"   ARC-Easy (2400): ~{time_per_question * 2400 / 3600:.1f} hours")
print(f"   OpenBookQA (500): ~{time_per_question * 500 / 3600:.1f} hours")

if success_count == len(test_cases):
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Ready to proceed!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run quick test: python -m llm_belief_mi_test.cli --dataset arc-easy --limit 5 --k 3 --t 2")
    print("2. If successful, try: --limit 50 --k 10 --t 3")
else:
    print("\n⚠️  Some tests failed, but the model is loaded and working")
    print("   This is acceptable for testing the MI method")

