#!/usr/bin/env python3
"""
Quick test to verify all three evaluation methods work correctly.

Usage:
    python test_baselines.py
"""

from llm_belief_mi_test.llm_client_local import LocalLlamaClient
from llm_belief_mi_test.datasets import load_arc_easy
from llm_belief_mi_test.calibration import (
    evaluate_mcq_greedy_baseline,
    evaluate_mcq_self_consistency,
    evaluate_mcq_with_mi
)

def test_all_methods():
    """Test all three evaluation methods on 3 examples."""
    
    print("="*80)
    print("TESTING BASELINE METHODS")
    print("="*80)
    
    # Load a few examples
    print("\n1. Loading dataset (3 examples from ARC-Easy)...")
    examples = load_arc_easy("test", limit=3)
    print(f"   ✅ Loaded {len(examples)} examples")
    
    # Initialize client
    print("\n2. Initializing model...")
    client = LocalLlamaClient(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=True,
        cache=None  # Disable cache for testing
    )
    print("   ✅ Model loaded")
    
    # Test greedy baseline
    print("\n3. Testing GREEDY BASELINE...")
    try:
        metrics_greedy, results_greedy = evaluate_mcq_greedy_baseline(
            client=client,
            examples=examples,
            max_tokens=30,
            verbose=False
        )
        print(f"   ✅ Greedy baseline works!")
        print(f"      Accuracy: {metrics_greedy['accuracy']:.4f}")
        print(f"      ECE: {metrics_greedy['ece']:.4f}")
    except Exception as e:
        print(f"   ❌ Greedy baseline failed: {e}")
        raise
    
    # Test self-consistency
    print("\n4. Testing SELF-CONSISTENCY BASELINE (k=5)...")
    try:
        metrics_sc, results_sc = evaluate_mcq_self_consistency(
            client=client,
            examples=examples,
            k=5,
            temperature=0.9,
            max_tokens=30,
            verbose=False
        )
        print(f"   ✅ Self-consistency baseline works!")
        print(f"      Accuracy: {metrics_sc['accuracy']:.4f}")
        print(f"      ECE: {metrics_sc['ece']:.4f}")
        print(f"      Avg Agreement: {metrics_sc['avg_agreement']:.4f}")
    except Exception as e:
        print(f"   ❌ Self-consistency failed: {e}")
        raise
    
    # Test MI method
    print("\n5. Testing MI METHOD (k=5, n=2)...")
    try:
        metrics_mi, results_mi = evaluate_mcq_with_mi(
            client=client,
            examples=examples,
            k=5,
            n=2,
            temperature=0.9,
            max_tokens=30,
            verbose=False
        )
        print(f"   ✅ MI method works!")
        print(f"      Accuracy: {metrics_mi['accuracy']:.4f}")
        print(f"      ECE: {metrics_mi['ece']:.4f}")
        print(f"      Avg MI: {metrics_mi['avg_mi_bits']:.4f} bits")
    except Exception as e:
        print(f"   ❌ MI method failed: {e}")
        raise
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Method':<25} {'Accuracy':<12} {'ECE':<12}")
    print("-"*80)
    print(f"{'Greedy Baseline':<25} {metrics_greedy['accuracy']:<12.4f} {metrics_greedy['ece']:<12.4f}")
    print(f"{'Self-Consistency':<25} {metrics_sc['accuracy']:<12.4f} {metrics_sc['ece']:<12.4f}")
    print(f"{'MI Method':<25} {metrics_mi['accuracy']:<12.4f} {metrics_mi['ece']:<12.4f}")
    print("="*80)
    
    print("\n✅ ALL TESTS PASSED!")
    print("\nNext steps:")
    print("  1. Run full comparison on 50 examples (see BASELINE_COMPARISON_GUIDE.md)")
    print("  2. Compare ECE across methods - MI should have lower ECE")
    print("  3. Run on full datasets for publication-quality results")
    print()

if __name__ == "__main__":
    test_all_methods()

