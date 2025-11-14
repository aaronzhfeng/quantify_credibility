# Detailed Logging Implementation Status

## ✅ COMPLETED

### 1. DetailedLogger Class
- Created `/llm_belief_mi_test/detailed_logger.py`
- Provides `log_question()` method to save per-question JSON files
- Automatically creates log directories based on output CSV name
- Example: `--output results/test.csv` → logs to `logs/test/question_*.json`

### 2. CLI Integration
- Added DetailedLogger creation in `cli.py`
- Logger passed to all evaluation functions
- Logs directory structure: `outputs/logs/{run_name}/question_{id}.json`

### 3. Function Signatures Updated
- ✅ `evaluate_mcq_greedy_baseline()` - parameter added + logging implemented
- ✅ `evaluate_mcq_self_consistency()` - parameter added
- ✅ `evaluate_mcq_semantic_entropy()` - parameter added
- ✅ `evaluate_mcq_self_verification()` - parameter added
- ✅ `evaluate_mcq_with_mi()` - parameter added

## ⏳ IN PROGRESS

### 4. Logging Logic Implementation
Need to add detailed logging calls in each evaluation function (similar to greedy baseline):

**Pattern for each method:**
```python
for ex_idx, ex in enumerate(iterator):
    # ... method-specific logic ...
    
    # Log detailed trace if logger provided
    if detailed_logger:
        method_data = {
            "description": "Method description",
            "raw_inputs": [list of prompts],
            "raw_outputs": [list of responses with logprobs],
            "decision_process": {method-specific decision logic},
            "final_metrics": {
                "predicted": predicted_choice,
                "correct": correct,
                "confidence": confidence,
                ...
            }
        }
        detailed_logger.log_question(
            question_id=ex_idx,
            question_text=ex.question,
            choices=ex.choices,
            choice_texts=ex.choice_texts,
            gold_answer=ex.answer_key,
            method_data=method_data
        )
```

**Status by method:**
- ✅ **Greedy** - COMPLETE
- ⏳ **Self-Consistency** - Need to add logging (captures k samples, majority voting)
- ⏳ **Semantic Entropy** - Need to add logging (captures k samples, F1 clustering, entropy calc)
- ⏳ **Self-Verification** - Need to add logging (captures k samples + verification query)
- ⏳ **MI Method** - Need to add logging (captures k×n chains, pseudo joint, MI estimation)

## 📝 REMAINING WORK

For each of the 4 remaining methods, need to:
1. Change `for ex in iterator:` to `for ex_idx, ex in enumerate(iterator):`
2. Capture all intermediate data (prompts, responses, decision process)
3. Add logging call with complete method_data dictionary

## 🧪 TESTING

Once complete, test with:
```bash
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa --limit 5 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_logging.csv

# Check logs created:
ls outputs/logs/test_logging/
# Should see: question_0.json, question_1.json, ..., question_4.json
```

## 📊 OUTPUT FORMAT

Each `question_*.json` file contains:
```json
{
  "question_id": 0,
  "question_text": "...",
  "choices": ["A: ...", "B: ...", "C: ...", "D: ..."],
  "gold_answer": "B",
  "methods": {
    "greedy": {  // or "self_consistency", "semantic_entropy", etc.
      "description": "...",
      "raw_inputs": [...],
      "raw_outputs": [...],
      "decision_process": {...},
      "final_metrics": {...}
    }
  }
}
```

Key difference from demo: Only ONE method in `methods` object (the method being run).

