#!/usr/bin/env python
"""Test with Phi-3-mini (no authentication required, works on CPU)."""

import sys
import time
sys.path.insert(0, '/teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test')

from llm_belief_mi_test.llm_client_local import LocalLlamaClient

print("="*60)
print("Testing Phi-3-mini-4k-instruct (No Auth Required)")
print("="*60)

print("\n🔄 Initializing Phi-3-mini...")
print("   ✅ No HuggingFace authentication needed")
print("   Model size: ~8 GB (should fit in 17GB RAM)")

start_time = time.time()

try:
    client = LocalLlamaClient(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        use_cpu=True
    )
    
    load_time = time.time() - start_time
    print(f"\n✅ Model loaded in {load_time:.1f} seconds")
    
except Exception as e:
    print(f"\n❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test generation
print("\n🔄 Testing generation...")

test_messages = [
    {"role": "user", "content": "What is 2+2? Answer with just the number."}
]

try:
    gen_start = time.time()
    response = client.chat_completion(test_messages, temperature=0.0, max_tokens=10)
    gen_time = time.time() - gen_start
    
    print(f"\n✅ Generation completed in {gen_time:.1f} seconds")
    print(f"   Response: '{response}'")
    
    # Estimate performance
    time_per_question = gen_time * 10 * 3  # K=10, t=3
    print(f"\n📊 Performance Estimates (K=10, t=3):")
    print(f"   Per question: ~{time_per_question:.0f} seconds")
    print(f"   50 questions: ~{time_per_question * 50 / 3600:.1f} hours")
    print(f"   200 questions: ~{time_per_question * 200 / 3600:.1f} hours")
    
    print("\n✅ Phi-3-mini is ready to use!")
    print("\nNext step: Copy core modules and implement full evaluation")
    
except Exception as e:
    print(f"\n❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

