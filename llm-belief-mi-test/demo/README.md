# Demo: Detailed Method Comparison

This folder contains a comprehensive demonstration of all 5 evaluation methods on the first 5 questions from OpenBookQA, with complete raw data, intermediate computations, and decision traces.

## Purpose

The demo system provides:
- **Raw transparency**: All prompts sent to the model and responses received
- **Decision traceability**: How each method converts model outputs to final predictions
- **Method comparison**: Side-by-side analysis of all 5 approaches
- **Educational value**: Understand exactly how each method works

## Quick Start

### Generate Demo Data

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Generate demo for first 5 OpenBookQA questions
python demo/scripts/generate_demo.py

# Or specify custom number of questions
python demo/scripts/generate_demo.py --num-questions 10
```

**Time**: ~30-45 minutes for 5 questions (all 5 methods × 5 questions)

### View Demo Data

```bash
# View summary of all methods for question 0
python demo/scripts/view_demo.py --question 0 --method all

# View detailed trace of MI method for question 0
python demo/scripts/view_demo.py --question 0 --method mi_method --verbose

# View specific method
python demo/scripts/view_demo.py --question 2 --method semantic_entropy

# Export all demos to markdown report
python demo/scripts/view_demo.py --export-markdown demo/demo_report.md
```

## Directory Structure

```
demo/
├── README.md              # This file
├── scripts/
│   ├── generate_demo.py   # Generate comprehensive demo data
│   └── view_demo.py       # View and analyze demo data
└── outputs/
    ├── question_0.json    # Demo data for question 0
    ├── question_1.json    # Demo data for question 1
    ├── question_2.json
    ├── question_3.json
    └── question_4.json
```

## JSON Schema

Each `question_*.json` file contains:

### Top Level
```json
{
  "question_id": 0,
  "question_text": "...",
  "choices": ["A: ...", "B: ...", "C: ...", "D: ..."],
  "gold_answer": "B",
  "methods": { ... },
  "comparison_summary": { ... }
}
```

### Per Method (e.g., "greedy", "mi_method")
```json
"greedy": {
  "description": "Single greedy decode (temperature=0)",
  "raw_inputs": [
    {
      "prompt": [...],  // Full message list sent to model
      "temperature": 0.0,
      "max_tokens": 30
    }
  ],
  "raw_outputs": [
    {
      "text": "B",  // Raw text response
      "logprob": -0.12,  // Log probability
      "probability": 0.887  // Converted probability
    }
  ],
  "decision_process": {
    "selected_text": "B",
    "matched_choice": "B",
    "confidence_computation": "exp(logprob) = exp(-0.12) = 0.887"
  },
  "final_metrics": {
    "predicted": "B",
    "correct": true,
    "confidence": 0.887,
    "mi_score": 0.0,
    "agreement": 1.0
  }
}
```

## Method-Specific Details

### Greedy
- **Raw inputs**: 1 prompt (temperature=0)
- **Raw outputs**: 1 response with logprob
- **Decision process**: Direct choice matching, confidence from logprob

### Self-Consistency
- **Raw inputs**: k=10 prompts (temperature=0.9)
- **Raw outputs**: 10 responses with logprobs
- **Decision process**: Vote counts, majority selection, confidence = agreement fraction

### Semantic Entropy
- **Raw inputs**: k=10 prompts (temperature=0.9)
- **Raw outputs**: 10 responses with logprobs
- **Decision process**:
  - Similarity matrix (F1 scores between all pairs)
  - Semantic clusters (grouped by F1 ≥ 0.25)
  - Aggregated distribution (summed probabilities per cluster)
  - Entropy calculation
  - Confidence = exp(-entropy)

### Self-Verification
- **Raw inputs**: k=10 initial prompts + 1 verification prompt
- **Raw outputs**: 10 initial responses + 1 verification response
- **Decision process**:
  - Initial selection (highest aggregated probability)
  - Verification prompt with True/False question
  - Confidence from verification response

### MI Method
- **Raw inputs**: k×n=20 prompts (10 chains × 2 steps)
- **Raw outputs**: 20 responses with logprobs
- **Decision process**:
  - Chains (paired responses per chain)
  - Pseudo joint distribution
  - Marginal distribution
  - MI estimation
  - Confidence = 1/(1 + MI)

## Example Use Cases

### 1. Understand How MI Method Works
```bash
python demo/scripts/view_demo.py --question 0 --method mi_method --verbose
```
See all 20 generations (10 chains × 2 steps), how they're combined into pseudo joint, and how MI is computed.

### 2. Compare All Methods
```bash
python demo/scripts/view_demo.py --question 0 --method all
```
See predictions, correctness, and confidence for all 5 methods side-by-side.

### 3. Analyze Semantic Clustering
```bash
python demo/scripts/view_demo.py --question 1 --method semantic_entropy --verbose
```
See which responses get grouped together based on F1 similarity.

### 4. Generate Report
```bash
python demo/scripts/view_demo.py --export-markdown demo_report.md
```
Create markdown report with all questions and all methods for sharing/documentation.

## What Each Field Means

### Raw Inputs
- **prompt**: Actual message list sent to the model (includes system/user messages)
- **temperature**: Sampling temperature (0=greedy, 0.9=diverse)
- **max_tokens**: Maximum tokens in response

### Raw Outputs  
- **text**: Actual text generated by model
- **logprob**: Log probability of the generated sequence
- **probability**: Converted probability (exp(logprob))

### Decision Process
Method-specific intermediate computations showing how raw outputs are transformed into final prediction and confidence.

### Final Metrics
- **predicted**: Final predicted choice (A/B/C/D)
- **correct**: Whether prediction matches gold answer
- **confidence**: Confidence score (0-1)
- **mi_score**: MI in bits (for MI method) or entropy (for S.E.)
- **agreement**: Agreement measure (method-specific)

## Comparison Summary

Shows all methods' predictions, correctness, and confidences for easy comparison:
```json
"comparison_summary": {
  "all_predictions": {
    "greedy": "B",
    "self_consistency": "B",
    "semantic_entropy": "B",
    "self_verification": "A",
    "mi_method": "B"
  },
  "all_correct": {
    "greedy": true,
    "self_consistency": true,
    "semantic_entropy": true,
    "self_verification": false,
    "mi_method": true
  },
  "all_confidences": {
    "greedy": 0.887,
    "self_consistency": 0.7,
    "semantic_entropy": 0.528,
    "self_verification": 0.1,
    "mi_method": 0.746
  },
  "agreement_across_methods": 4
}
```

## Tips

### For Debugging
- Use `--verbose` flag to see ALL raw inputs/outputs
- Check `raw_inputs` to verify prompts are correct
- Check `raw_outputs` to see what model actually generated
- Check `decision_process` to trace how prediction was made

### For Analysis
- Compare `confidence` across methods for same question
- Look at `similarity_matrix` in semantic_entropy to see clustering
- Examine `pseudo_joint` in mi_method to understand distributions
- Check `agreement` to see consensus within k samples

### For Presentation
- Use `--export-markdown` to create shareable reports
- Include specific questions in papers/presentations
- Show decision traces to explain method differences

## Notes

- **File size**: Each demo file is ~50-200KB depending on k and n parameters
- **Generation time**: ~30-45 minutes for 5 questions with default settings
- **Cache**: Demo generation disables cache to show true sampling behavior
- **Reproducibility**: Results vary due to sampling (temperature=0.9), but decision logic is deterministic

## Troubleshooting

**"Demo file not found"**: Run `generate_demo.py` first to create demo data

**"Out of memory"**: Reduce number of questions or use `--load-in-4bit`

**"Generation takes too long"**: Expected - each question runs all 5 methods, each with k=10 samples

---

**Generated demos provide complete transparency into how each evaluation method works!**

