# Next Steps: Setting Up Local Llama Model

## Current Status ✅

The following have been completed:
- ✅ Project structure created
- ✅ Documentation written (README.md, IMPLEMENTATION_PLAN.md)
- ✅ Dependencies specified (requirements.txt)
- ✅ Output directories created

## Step 1: Install Dependencies

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Install core dependencies
pip install torch transformers accelerate bitsandbytes datasets tqdm numpy pandas matplotlib scikit-learn

# Or use requirements.txt
pip install -r requirements.txt
```

## Step 2: Verify GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
GPU count: 1 (or more)
GPU name: NVIDIA ... (your GPU name)
```

## Step 3: Test Model Download & Loading

### Option A: Test with Minimal Script (Recommended First)

Create a test script:

```bash
cat > test_model_load.py << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,
)

print("Model loaded successfully!")
print(f"Model device: {model.device}")
print(f"Model dtype: {model.dtype}")

# Test generation
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nGenerating response...")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

print(f"Response: {response}")
print("\n✅ Everything works!")
EOF

python test_model_load.py
```

### Option B: Direct Import Test

```bash
python << 'EOF'
import sys
sys.path.insert(0, '/teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test')

from llm_belief_mi_test.llm_client_local import LocalLlamaClient

print("Initializing client with 4-bit quantization...")
client = LocalLlamaClient(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True
)

print("Testing chat completion...")
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = client.chat_completion(messages, temperature=0.0, max_tokens=50)
print(f"Response: {response}")
print("\n✅ Client works!")
EOF
```

## Step 4: Troubleshooting Common Issues

### Issue 1: Out of Memory

**Solution 1**: Use 4-bit quantization (already included above)

**Solution 2**: Clear GPU memory
```bash
python -c "import torch; torch.cuda.empty_cache()"
```

**Solution 3**: Check GPU memory usage
```bash
nvidia-smi
```

### Issue 2: Model Download Fails

**Solution 1**: Check internet connection

**Solution 2**: Set HuggingFace token (if needed)
```bash
export HF_TOKEN="your_huggingface_token"
# Or in Python:
# huggingface_hub.login(token="your_token")
```

**Solution 3**: Specify cache directory
```bash
export HF_HOME="/path/to/large/disk/.cache/huggingface"
```

### Issue 3: Import Errors

**Solution**: Ensure you're in the right directory
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test
export PYTHONPATH="${PYTHONPATH}:/teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test"
```

## Step 5: Copy Core Modules from Repro

Once the model works, copy the MI estimator and related files:

```bash
# Navigate to test directory
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Copy core modules (no modifications needed)
cp ../llm-belief-mi-repro/llm_belief_mi_repro/mi_estimator.py llm_belief_mi_test/
cp ../llm-belief-mi-repro/llm_belief_mi_repro/iterative_prompting.py llm_belief_mi_test/
cp ../llm-belief-mi-repro/llm_belief_mi_repro/evaluation.py llm_belief_mi_test/

# Copy datasets.py (will need modifications)
cp ../llm-belief-mi-repro/llm_belief_mi_repro/datasets.py llm_belief_mi_test/

echo "Core modules copied!"
```

## Step 6: Implement Local Client

Create the local Llama client (see IMPLEMENTATION_PLAN.md Phase 2):

```bash
# This file needs to be created as per the plan
# File: llm_belief_mi_test/llm_client_local.py
```

The implementation is detailed in IMPLEMENTATION_PLAN.md.

## Step 7: Quick Sanity Test

Once all modules are in place:

```bash
# Test with 5 examples from ARC-Easy
python -m llm_belief_mi_test.cli \
  --dataset arc-easy \
  --limit 5 \
  --k 3 --t 2 \
  --load-in-4bit \
  --output outputs/results/test_quick.csv \
  --verbose
```

Expected runtime: ~2-3 minutes for 5 examples

## Step 8: Small-Scale Test

```bash
# Test with 50 examples
python -m llm_belief_mi_test.cli \
  --dataset arc-easy \
  --limit 50 \
  --k 10 --t 3 \
  --load-in-4bit \
  --output outputs/results/arc_easy_50.csv
```

Expected runtime: ~20-40 minutes for 50 examples

## Step 9: Full Evaluation

Once everything works on small scale:

```bash
# Run all benchmarks
./run_all_benchmarks.sh  # (create this script or run commands from README.md)
```

## Hardware Checks Before Starting

### Minimum Requirements Check
```bash
python << 'EOF'
import torch
import psutil

print("=== Hardware Check ===")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"GPU Memory Free: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

print(f"\nRAM Total: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"RAM Available: {psutil.virtual_memory().available / 1e9:.1f} GB")

# Check if we can run
gpu_ok = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 10e9
ram_ok = psutil.virtual_memory().total > 12e9

if gpu_ok and ram_ok:
    print("\n✅ Hardware meets minimum requirements!")
    print("   You can proceed with 4-bit quantization.")
else:
    print("\n⚠️ Warning: Hardware may be insufficient")
    if not gpu_ok:
        print("   - Need at least 12GB GPU memory (use 4-bit quantization)")
    if not ram_ok:
        print("   - Need at least 16GB system RAM")
EOF
```

## Quick Reference Commands

```bash
# Check GPU status
nvidia-smi

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check Python packages
pip list | grep -E "torch|transformers|accelerate|bitsandbytes"

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Check disk space (models are large!)
df -h

# Kill all Python processes (if something hangs)
pkill -9 python
```

## Status Tracking

- [ ] Dependencies installed
- [ ] GPU verified
- [ ] Model downloads successfully
- [ ] Test generation works
- [ ] Core modules copied
- [ ] Local client implemented
- [ ] CLI works
- [ ] Quick test (5 examples) passes
- [ ] Small test (50 examples) passes
- [ ] Ready for full evaluation

## Getting Help

If you encounter issues:

1. Check the IMPLEMENTATION_PLAN.md for detailed specifications
2. Review error messages carefully (often indicate missing dependencies)
3. Verify GPU memory with `nvidia-smi`
4. Try with fewer chains/shorter lengths: `--k 3 --t 2`
5. Use 4-bit quantization: `--load-in-4bit`

---

**Current Priority**: Proceed to Step 2-3 (verify GPU and test model loading)

