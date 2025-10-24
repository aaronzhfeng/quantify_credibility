# HuggingFace Authentication Guide

## Issue

Meta's Llama models (including 3.1 and 3.2) are **gated** - they require:
1. HuggingFace account
2. Accepting Meta's license agreement
3. Authentication token

Error: `401 Client Error: Unauthorized` - Access to model is restricted

---

## Solution A: Authenticate with HuggingFace (Required for Llama Models)

### Step 1: Create HuggingFace Account
1. Go to https://huggingface.co/join
2. Create a free account

### Step 2: Request Access to Llama Models
1. Go to model page:
   - **Llama-3.2-1B**: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - **Llama-3.1-8B**: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Accept Meta's license agreement
4. Access is usually granted immediately

### Step 3: Create Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name: `llm-belief-mi-test` (or any name)
4. Type: **Read** (sufficient for downloading models)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)

### Step 4: Login via CLI

**Option A: Using huggingface-cli (Recommended)**
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

**Option B: Using Python**
```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")
```

**Option C: Environment Variable**
```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
# Then run your scripts
```

**Option D: In Code (for testing)**
```python
import os
os.environ["HF_TOKEN"] = "hf_YOUR_TOKEN_HERE"
# Then import and use models
```

### Step 5: Verify Access
```bash
python -c "from huggingface_hub import whoami; print(whoami())"
```

Should show your username and token info.

### Step 6: Test Model Loading
```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test
python test_model_1b.py
```

---

## Solution B: Use Ungated Models (No Authentication Required) ✅ **QUICK START**

Several high-quality models don't require authentication:

### Option 1: Mistral-7B-Instruct-v0.3 ✨ **RECOMMENDED**

**Advantages:**
- ✅ No authentication required
- ✅ High quality (competitive with Llama)
- ✅ ~7B parameters (similar to target)
- ✅ Good instruction-following

**Implementation:**
```python
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
```

**Memory requirements:**
- CPU: ~14-16 GB RAM (may still be tight)
- GPU: ~14 GB VRAM (with 4-bit: ~8 GB)

### Option 2: Phi-3-mini-4k-instruct ⭐ **BEST FOR CPU**

**Advantages:**
- ✅ No authentication required
- ✅ Only 3.8B parameters
- ✅ Excellent quality for size
- ✅ Fast on CPU (~3-8 seconds/generation)
- ✅ Will definitely fit in 17GB RAM

**Implementation:**
```python
model_name = "microsoft/Phi-3-mini-4k-instruct"
```

**Memory requirements:**
- CPU: ~8-10 GB RAM ✅ **WILL WORK**
- GPU: ~8 GB VRAM

### Option 3: Qwen2.5-7B-Instruct

**Advantages:**
- ✅ No authentication required
- ✅ State-of-the-art for 7B size
- ✅ Excellent on benchmarks

**Implementation:**
```python
model_name = "Qwen/Qwen2.5-7B-Instruct"
```

**Memory requirements:**
- CPU: ~14-16 GB RAM (borderline)
- GPU: ~14 GB VRAM

### Option 4: Gemma-2-2b-it (Smallest, Fastest)

**Advantages:**
- ✅ No authentication required
- ✅ Only 2B parameters
- ✅ Very fast on CPU (~2-5 seconds/generation)
- ✅ From Google

**Implementation:**
```python
model_name = "google/gemma-2-2b-it"
```

**Memory requirements:**
- CPU: ~4-6 GB RAM ✅ **WILL WORK**
- GPU: ~4 GB VRAM

---

## Model Comparison for CPU Inference

| Model | Params | Auth Required? | CPU RAM | Est. Speed* | Quality |
|-------|--------|---------------|---------|-------------|---------|
| **Phi-3-mini** | 3.8B | ❌ No | ~10 GB | ~5s | ⭐⭐⭐⭐ |
| **Gemma-2-2b** | 2B | ❌ No | ~6 GB | ~3s | ⭐⭐⭐ |
| Mistral-7B | 7B | ❌ No | ~16 GB | ~15s | ⭐⭐⭐⭐⭐ |
| Qwen2.5-7B | 7B | ❌ No | ~16 GB | ~15s | ⭐⭐⭐⭐⭐ |
| Llama-3.2-1B | 1B | ✅ Yes | ~4 GB | ~2s | ⭐⭐⭐ |
| Llama-3.2-3B | 3B | ✅ Yes | ~8 GB | ~5s | ⭐⭐⭐⭐ |
| Llama-3.1-8B | 8B | ✅ Yes | ❌ OOM | N/A | ⭐⭐⭐⭐⭐ |

*Estimated generation time per response on 4-core CPU

---

## Recommended Path Forward

### Path 1: Quick Start with Phi-3-mini (Best for Immediate Results) ✅

1. **No authentication needed**
2. **Works on current hardware**
3. **Good quality**

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Create test with Phi-3
python - <<'EOF'
import sys
sys.path.insert(0, '.')
from llm_belief_mi_test.llm_client_local import LocalLlamaClient

print("Testing Phi-3-mini (no auth required)...")
client = LocalLlamaClient(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    use_cpu=True
)

messages = [{"role": "user", "content": "What is 2+2?"}]
response = client.chat_completion(messages, max_tokens=10)
print(f"Response: {response}")
print("✅ Works!")
EOF
```

### Path 2: Authenticate for Llama Models (If Preferred)

1. Follow Steps 1-4 above
2. Use `huggingface-cli login`
3. Test with `python test_model_1b.py`

---

## Quick Decision Matrix

**Use Phi-3-mini if:**
- ✅ You want to start immediately
- ✅ CPU-only environment
- ✅ Don't want to deal with authentication

**Use Llama-3.2-1B if:**
- ✅ You want exact replication of paper's architecture
- ✅ Willing to authenticate with HuggingFace
- ✅ Want smallest Llama model

**Use Gemma-2-2b if:**
- ✅ Want Google's model
- ✅ Need very fast inference
- ✅ No authentication hassle

**Get GPU access if:**
- ✅ You need Llama-3.1-8B specifically
- ✅ Want fast, large-scale evaluation
- ✅ Have access to Colab/Kaggle/cloud

---

## Next Steps

### Option A: Use Phi-3-mini (Recommended for immediate progress)
```bash
# Update test script
sed -i 's/Llama-3.2-1B-Instruct/Phi-3-mini-4k-instruct/g' test_model_1b.py
sed -i 's/meta-llama/microsoft/g' test_model_1b.py

# Run test
python test_model_1b.py
```

### Option B: Authenticate for Llama
```bash
pip install huggingface_hub
huggingface-cli login
# Then paste your token
python test_model_1b.py
```

---

_Which approach would you like to take?_

**My recommendation**: Start with **Phi-3-mini** to get immediate results, then optionally authenticate for Llama models later if needed.

