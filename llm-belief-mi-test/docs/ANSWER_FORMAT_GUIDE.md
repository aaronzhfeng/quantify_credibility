# Answer Format Guide

## Problem: Verbose Responses Make Answer Extraction Difficult

With default prompting, models often generate verbose responses like:

```
"I'm happy to help! Based on the options provided, the best answer would be B) quit eating lunch out. This choice directly addresses the goal of saving money, as eating out can be a significant expense..."
```

This makes answer extraction unreliable and requires complex fuzzy matching.

---

## Solution: Three Answer Format Modes

### Mode 1: `default` (Original Behavior)

**System Prompt:**
```
"You are a helpful, concise assistant. Answer accurately. If unsure, say so briefly."
```

**Typical Response:**
```
"Based on general knowledge, the most likely place to have fog is:

A) a marsh

Marshes are often near water and surrounded by vegetation, which can lead to high humidity..."
```

**Extraction:** Fuzzy matching (checks for letter in first 20 chars, then text similarity)

**Pros:** Natural language responses  
**Cons:** Hard to extract, verbose, may fail to parse

---

### Mode 2: `strict` (Recommended!) ✅

**System Prompt:**
```
"You are answering a multiple-choice question. 
Your response MUST be ONLY the letter of the correct answer (A, B, C, or D). 
Do NOT include any explanation, reasoning, or additional text. 
Output ONLY: A, B, C, or D."
```

**Typical Response:**
```
"B"
```

**Extraction:** Direct - checks if response is exactly "A", "B", "C", or "D"

**Pros:** 
- Clean, unambiguous
- Easy to extract
- No parsing errors
- Faster (fewer tokens)

**Cons:**
- No explanation (but not needed for evaluation)

---

### Mode 3: `codeblock`

**System Prompt:**
```
"You are answering a multiple-choice question. 
Put your answer (A, B, C, or D) inside triple backticks like this: ```A``` 
You may include brief explanation before or after the code block, but the answer MUST be in the code block."
```

**Typical Response:**
```
"The correct answer is ```B``` because marshes have high humidity which leads to fog formation."
```

**Extraction:** Regex pattern `\`\`\`\s*([A-D])\s*\`\`\`` 

**Pros:**
- Unambiguous extraction
- Allows explanation for debugging
- Model-friendly format

**Cons:**
- More verbose than strict
- May fail if model doesn't follow format

---

## Usage

### Command Line

```bash
# Default mode (verbose responses)
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa --limit 50 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --answer-format default \
  --output outputs/results/test_default.csv

# Strict mode (only letter) - RECOMMENDED
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa --limit 50 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/test_strict.csv

# Codeblock mode
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset openbookqa --limit 50 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 50 \
  --answer-format codeblock \
  --output outputs/results/test_codeblock.csv
```

---

## Recommended Settings for Re-running Experiments

Based on the issues found, use **strict mode** for all evaluations:

### Full Baseline Comparison with Strict Mode

```bash
# ARC-Challenge (500 examples)

# Greedy
python -m llm_belief_mi_test.cli \
  --method greedy \
  --dataset arc-challenge --limit 500 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge_greedy_500_v2.csv

# Self-Consistency
python -m llm_belief_mi_test.cli \
  --method self-consistency \
  --dataset arc-challenge --limit 500 \
  --k 10 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge_selfcons_500_v2.csv

# MI Method
python -m llm_belief_mi_test.cli \
  --method mi \
  --dataset arc-challenge --limit 500 \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 \
  --answer-format strict \
  --output outputs/results/arc_challenge_mi_500_v2.csv
```

**Note the changes:**
- `--answer-format strict` (NEW!)
- `--max-tokens 10` (reduced from 30, since only outputting letter)
- `_v2` in output name (distinguish from previous buggy runs)

---

## Expected Improvements

### With Strict Mode:

**Accuracy:**
- Previous (default, buggy): ~28-32%
- Expected (strict): ~50-65% (significant improvement!)

**Answer Extraction:**
- Previous: Complex fuzzy matching, prone to errors
- Expected: Direct letter extraction, no ambiguity

**Speed:**
- Previous: 30-100 tokens per response
- Expected: 1-5 tokens per response (~10× faster!)

**ECE:**
- MI should still outperform baselines
- Absolute values may change, but relative advantage preserved

---

## Testing

### Quick Test (5 examples, all 3 formats)

```bash
for format in default strict codeblock; do
  python -m llm_belief_mi_test.cli \
    --method greedy \
    --dataset openbookqa --limit 5 \
    --load-in-4bit \
    --max-tokens $([ "$format" = "strict" ] && echo "10" || echo "50") \
    --answer-format $format \
    --output outputs/results/test_${format}_5.csv
  
  echo "Format: $format"
  cat outputs/results/test_${format}_5.json
  echo ""
done
```

Or use the test script:
```bash
python scripts/test_answer_formats.py
```

---

## Migration Path

### Step 1: Test on 5 Examples
```bash
# Test strict mode
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 5 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/test_strict_5.csv

# Check results
cat outputs/results/test_strict_5.json
```

### Step 2: Run on 50 Examples
```bash
# If test looks good, try 50
python -m llm_belief_mi_test.cli --method mi --dataset openbookqa --limit 50 --k 10 --n 2 --temperature 0.9 --load-in-4bit --max-tokens 10 --answer-format strict --output outputs/results/test_strict_50.csv
```

### Step 3: Full Re-evaluation (if needed)
```bash
# Run all methods on all datasets with strict mode
# See commands above
```

---

## Comparison: Old vs New

| Aspect | Old (Default) | New (Strict) |
|--------|---------------|--------------|
| **Response** | "Based on... the answer is B..." | "B" |
| **Tokens** | 30-100 | 1-5 |
| **Extraction** | Fuzzy matching | Direct |
| **Accuracy** | ~28% (guessing?) | ~50-65% (expected) |
| **Speed** | 1× | ~10× faster |
| **Reliability** | Medium | High |

---

## Implementation Details

### System Prompts

All prompts still include the MCQ choices in the user message:
```
Q: A person wants to start saving money...

Choices:
A) make more phone calls
B) quit eating lunch out
C) buy less with monopoly money
D) have lunch with friends
A:
```

But the **system prompt** changes the response style:
- Default: Allows explanation
- Strict: Forces single letter
- Codeblock: Letter in code block

### Answer Extraction

**Strict mode** (`extract_answer_strict`):
1. Strip and uppercase response
2. Check if exactly matches A/B/C/D
3. If not, check if first char is A/B/C/D
4. Fallback to fuzzy matching

**Codeblock mode** (`extract_answer_from_codeblock`):
1. Regex search for ```X``` pattern
2. Extract letter inside backticks
3. Validate it's a valid choice
4. Fallback to fuzzy matching

---

**Recommended: Use `--answer-format strict` for all future evaluations!** ✅

