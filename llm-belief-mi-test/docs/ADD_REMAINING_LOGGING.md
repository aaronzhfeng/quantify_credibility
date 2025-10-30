# Remaining Logging Implementation

## STATUS: 2/5 Methods Complete

✅ **Greedy** - Complete  
✅ **Self-Consistency** - Complete  
⏳ **Semantic Entropy** - Need to implement  
⏳ **Self-Verification** - Need to implement  
⏳ **MI Method** - Need to implement  

## Implementation Guide for Remaining 3 Methods

Each method needs the same pattern:

### 1. Change loop to enumerate
```python
# Before:
for ex in iterator:

# After:
for ex_idx, ex in enumerate(iterator):
```

### 2. Capture sample data during generation
```python
sample_data = []  # Add at start of loop

# Inside generation loop:
if detailed_logger:
    sample_data.append({
        "sample_id": i,
        "prompt": messages,
        "response": response,
        "logprob": logprob,
        ...
    })
```

### 3. Add logging after result creation
```python
result = EvaluationResult(...)
results.append(result)

if detailed_logger:
    method_data = {
        "description": "...",
        "raw_inputs": [...],
        "raw_outputs": [...],
        "decision_process": {...},
        "final_metrics": {...}
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

## Method-Specific Details

### Semantic Entropy (line ~645)
- Captures k samples with probabilities
- Groups by semantic similarity (F1)
- Computes entropy of clustered distribution
- **Decision process**: Include similarity_matrix, semantic_clusters, aggregated_distribution, entropy calculation

### Self-Verification (line ~791)
- Generates k initial samples
- Selects best answer
- Runs verification query
- **Decision process**: Include initial_selection, verification_prompt, verification_response, confidence mapping

### MI Method (line ~948)
- Generates k chains of length n
- Builds pseudo joint distribution
- Computes MI
- **Decision process**: Include chains, pseudo_joint, marginal_distribution, mi_estimation

## Quick Test After Implementation

```bash
# Test with greedy (already works):
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset openbookqa --limit 2 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_log_greedy.csv

# Check logs:
ls outputs/logs/test_log_greedy/
cat outputs/logs/test_log_greedy/question_0.json | head -50

# Test with each other method once implemented
```

## Alternative: Simplified Logging

If full detail is too complex, could implement simplified version:
- Just save prompts, responses, and final metrics
- Skip detailed decision_process
- Faster to implement, still useful for debugging

User can decide which approach to use based on needs.

