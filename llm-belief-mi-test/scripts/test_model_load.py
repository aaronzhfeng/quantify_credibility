#!/usr/bin/env python
"""Test script to verify model loading and generation."""

import sys
import time
sys.path.insert(0, '/teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test')

from llm_belief_mi_test.llm_client_local import LocalLlamaClient

print("="*60)
print("Testing Local Llama Client")
print("="*60)

print("\n🔄 Initializing client (this will download the model if not cached)...")
print("   Model size: ~8-16 GB - first run will take a while to download")

start_time = time.time()

try:
    client = LocalLlamaClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        use_cpu=True  # Explicitly use CPU
    )
    
    load_time = time.time() - start_time
    print(f"\n✅ Model loaded in {load_time:.1f} seconds")
    
except Exception as e:
    print(f"\n❌ Failed to load model: {e}")
    print("\nTroubleshooting:")
    print("1. Check internet connection (model needs to download ~8-16 GB)")
    print("2. Check disk space (need ~20 GB free)")
    print("3. Set HF_TOKEN if model requires authentication:")
    print("   export HF_TOKEN='your_token'")
    sys.exit(1)

# Test generation
print("\n🔄 Testing chat completion (this may take 10-30 seconds on CPU)...")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2? Answer in one word."}
]

try:
    gen_start = time.time()
    response = client.chat_completion(messages, temperature=0.0, max_tokens=10)
    gen_time = time.time() - gen_start
    
    print(f"\n✅ Generation completed in {gen_time:.1f} seconds")
    print(f"   Response: '{response}'")
    
    # Performance assessment
    if gen_time < 5:
        print("\n✅ Performance: Good (< 5 seconds per response)")
    elif gen_time < 15:
        print("\n⚠️  Performance: Acceptable (5-15 seconds per response)")
        print("   Small-scale testing is feasible but large runs will be slow")
    else:
        print("\n⚠️  Performance: Slow (> 15 seconds per response)")
        print("   Consider:")
        print("   - Using a GPU-enabled environment")
        print("   - Testing with very small samples (--limit 5-10)")
        print("   - Using a smaller model (e.g., Llama-3.2-1B)")
    
    # Estimate full run time
    print(f"\n📊 Estimated time for full evaluation:")
    time_per_question = gen_time * 10 * 3  # K=10 chains, t=3 length
    print(f"   Per question (K=10, t=3): ~{time_per_question:.0f} seconds")
    print(f"   ARC-Challenge (1200 q's): ~{time_per_question * 1200 / 3600:.1f} hours")
    print(f"   ARC-Easy (2400 q's): ~{time_per_question * 2400 / 3600:.1f} hours")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)
    print("\nRecommendation:")
    if gen_time < 15:
        print("- Start with small test: --limit 5 --k 3 --t 2")
        print("- Then try medium test: --limit 50 --k 5 --t 2")
    else:
        print("- CPU inference is very slow for this model")
        print("- Options:")
        print("  1. Use GPU-enabled environment")
        print("  2. Use a smaller model (Llama-3.2-1B or Llama-3.2-3B)")
        print("  3. Run very small samples (--limit 10 --k 3 --t 2)")
    
except Exception as e:
    print(f"\n❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

