"""
Test script to show what DeBERTa NLI model actually outputs.
"""
import sys
sys.path.insert(0, '/root/quantify_credibility/nli-semantic-clustering')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the NLI model
model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Get label mapping
label2id = model.config.label2id
id2label = model.config.id2label
print(f"\nLabel mapping: {label2id}")
print(f"Reverse mapping: {id2label}")

# Test examples
test_cases = [
    {
        "name": "Clear Entailment",
        "premise": "David Seville created The Chipmunks",
        "hypothesis": "The Chipmunks were created by David Seville"
    },
    {
        "name": "Partial Entailment (verbose answer)",
        "premise": "David Seville, a pseudonym for Ross Bagdasarian Sr.",
        "hypothesis": "David Seville"
    },
    {
        "name": "Contradiction",
        "premise": "Paris is the capital of France",
        "hypothesis": "London is the capital of France"
    },
    {
        "name": "Neutral",
        "premise": "Paris is the capital of France",
        "hypothesis": "Paris is a beautiful city"
    }
]

print("\n" + "="*80)
print("TESTING NLI MODEL OUTPUT")
print("="*80)

for test in test_cases:
    print(f"\n### {test['name']}")
    print(f"Premise:    \"{test['premise']}\"")
    print(f"Hypothesis: \"{test['hypothesis']}\"")
    
    # Tokenize and run model
    inputs = tokenizer(
        test['premise'], 
        test['hypothesis'],
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
    
    # Display all three scores
    print(f"\nRaw logits: {logits.tolist()}")
    print(f"\nProbabilities (sum to 1.0):")
    for label_id in sorted(id2label.keys()):
        label_name = id2label[label_id]
        prob = probs[label_id].item()
        marker = " ← WE USE THIS" if label_name == "entailment" else ""
        print(f"  {label_id} ({label_name:14s}): {prob:.4f}{marker}")
    
    predicted_label = id2label[probs.argmax().item()]
    print(f"\nModel prediction: {predicted_label}")

print("\n" + "="*80)
print("CONCLUSION:")
print("- The model outputs 3 probabilities for each sentence pair")
print("- These represent: contradiction, neutral, entailment")
print("- We ONLY use the entailment probability for our clustering")
print("- We ignore contradiction and neutral scores")
print("="*80)

