#!/usr/bin/env python3
"""
Validation tests for the is_correct() method.

This script tests the new NLI-based grading logic to ensure it:
1. Accepts verbose but correct answers
2. Rejects incorrect answers
3. Works better than strict bidirectional equivalence

Usage:
    python scripts/test_is_correct.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nli_clustering.core import NLIClusteringCache


def test_basic_functionality():
    """Test basic is_correct() functionality."""
    print("="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    nli = NLIClusteringCache()
    
    test_cases = [
        # (prediction, gold_label, expected_result, description)
        ("Paris", "Paris", True, "Exact match"),
        ("paris", "Paris", True, "Case insensitive"),
        ("The capital is Paris.", "Paris", True, "Substring match"),
        ("Paris is the answer.", "Paris", True, "Contains gold label"),
        ("It's Paris, France.", "Paris", True, "Contains gold label with context"),
        ("The capital of France is Paris.", "Paris", True, "Verbose but correct"),
        ("London", "Paris", False, "Wrong answer"),
        ("I don't know", "Paris", False, "No answer"),
        ("Paris, Texas", "Paris", True, "Contains gold (may be debatable)"),
    ]
    
    passed = 0
    failed = 0
    
    for pred, gold, expected, desc in test_cases:
        result = nli.is_correct(pred, gold)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} | {desc}")
        print(f"        Pred: '{pred}' | Gold: '{gold}' | Expected: {expected} | Got: {result}")
        print()
    
    print(f"\nResults: {passed}/{len(test_cases)} passed, {failed}/{len(test_cases)} failed")
    return failed == 0


def test_vs_mutual_entailment():
    """Compare is_correct() vs check_mutual_entailment()."""
    print("\n" + "="*80)
    print("TEST 2: is_correct() vs check_mutual_entailment()")
    print("="*80)
    
    nli = NLIClusteringCache()
    
    # Cases where is_correct should be TRUE but mutual_entailment should be FALSE
    test_cases = [
        ("The capital is Paris.", "Paris"),
        ("Paris is the capital of France.", "Paris"),
        ("The answer is Paris.", "Paris"),
    ]
    
    print("\nThese should be CORRECT (is_correct=True) but NOT EQUIVALENT (mutual=False):\n")
    
    all_correct = True
    for pred, gold in test_cases:
        is_correct_result = nli.is_correct(pred, gold)
        mutual_result = nli.check_mutual_entailment(pred, gold)
        
        # Get entailment scores for debugging
        fwd, bwd = nli.get_entailment_scores(pred, gold)
        
        correct_behavior = is_correct_result == True and mutual_result == False
        status = "✅ PASS" if correct_behavior else "❌ FAIL"
        
        if not correct_behavior:
            all_correct = False
        
        print(f"{status}")
        print(f"  Prediction: '{pred}'")
        print(f"  Gold: '{gold}'")
        print(f"  is_correct(): {is_correct_result} (should be True)")
        print(f"  mutual_entailment(): {mutual_result} (should be False)")
        print(f"  Entailment scores: Forward={fwd:.3f}, Backward={bwd:.3f}")
        print()
    
    return all_correct


def test_correctness_based_mi():
    """Test that is_correct works for correctness-based MI."""
    print("\n" + "="*80)
    print("TEST 3: Correctness-Based MI Compatibility")
    print("="*80)
    
    nli = NLIClusteringCache()
    
    # Simulate answers from multiple chains
    answers = [
        "Barack Obama",
        "Obama",
        "President Obama",
        "Barack Hussein Obama",
    ]
    
    gold = "Barack Obama"
    
    print(f"\nGold label: '{gold}'")
    print(f"Testing {len(answers)} answer variations:\n")
    
    all_pass = True
    for ans in answers:
        is_correct = nli.is_correct(ans, gold)
        status = "✅" if is_correct else "❌"
        
        if not is_correct:
            all_pass = False
        
        print(f"{status} '{ans}' → {'Correct' if is_correct else 'Wrong'}")
    
    print(f"\nAll answers should be marked as correct: {all_pass}")
    return all_pass


def test_edge_cases():
    """Test edge cases and corner scenarios."""
    print("\n" + "="*80)
    print("TEST 4: Edge Cases")
    print("="*80)
    
    nli = NLIClusteringCache()
    
    test_cases = [
        # (prediction, gold_label, expected_result, description)
        ("", "Paris", False, "Empty prediction"),
        ("Paris", "", True, "Empty gold (debatable)"),
        ("  Paris  ", "Paris", True, "Whitespace handling"),
        ("PARIS", "paris", True, "Case variations"),
        ("Paris.", "Paris", True, "Punctuation"),
        ("Paris?", "Paris", True, "Question mark"),
        ("Paris!", "Paris", True, "Exclamation"),
    ]
    
    passed = 0
    for pred, gold, expected, desc in test_cases:
        result = nli.is_correct(pred, gold)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        if result == expected:
            passed += 1
        
        print(f"{status} | {desc}")
        print(f"        Pred: '{pred}' | Gold: '{gold}' | Expected: {expected} | Got: {result}")
    
    print(f"\n{passed}/{len(test_cases)} edge cases handled correctly")
    return passed == len(test_cases)


def main():
    print("\n" + "🧪" * 40)
    print("NLI is_correct() Validation Tests")
    print("🧪" * 40 + "\n")
    
    results = []
    
    # Run all test suites
    results.append(("Basic Functionality", test_basic_functionality()))
    results.append(("is_correct vs mutual_entailment", test_vs_mutual_entailment()))
    results.append(("Correctness-Based MI", test_correctness_based_mi()))
    results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The is_correct() method is working as expected.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

