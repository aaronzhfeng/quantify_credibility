# 🚀 RUN THIS FIRST - Quick Verification

## ✅ Bugs Fixed & Ready to Test

### What Was Fixed:
1. ✅ **Cache bug**: Now disabled during sampling (preserves diversity)
2. ✅ **Temperature**: Corrected to 0.9 (from paper)
3. ✅ **max_tokens**: Set to 30 (optimized for MCQ)

---

## 🧪 Quick Verification Test (3 minutes)

Run this command to verify everything works:

```bash
cd /teamspace/studios/this_studio/quantify_credibility/llm-belief-mi-test

# Clear old bad cache
rm -rf .cache/llm_cache.sqlite

# Run quick test with CORRECTED settings
python -m llm_belief_mi_test.cli \
  --dataset arc-easy --limit 5 \
  --k 10 --n 2 \
  --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/test_corrected.csv
```

---

## ✅ Expected Results (Healthy)

```json
{
  "accuracy": 0.40-0.80,     ← Reasonable for 5 examples
  "ece": 0.00-0.30,          ← Some calibration error
  "avg_mi_bits": 0.2-1.0,    ← MI detected! (not 0.0)
  "avg_agreement": 0.3-0.7   ← Chains diverse (not 1.0)
}
```

### Red Flags (If You See These):
- ❌ `avg_mi_bits: 0.0000` - Chains not diverse, bug still present
- ❌ `avg_agreement: 1.0000` - All chains identical, bug still present
- ❌ `accuracy: 0.20` AND `agreement: 1.0` - Model stuck, wrong answer

### Good Signs:
- ✅ `avg_mi_bits > 0.1` - MI is working!
- ✅ `avg_agreement < 0.9` - Chains are diverse!
- ✅ `accuracy: 0.4-0.8` - Reasonable performance

---

## 📊 View Your Results

```bash
# Quick look at metrics
cat outputs/results/test_corrected.json

# See per-question details
head -20 outputs/results/test_corrected.csv
```

Look for:
- Different predicted answers across questions
- Varying MI scores
- Varying confidence levels

---

## Next Steps

### ✅ If Results Look Good (MI>0, agreement<0.9):

**Proceed with 50 examples:**
```bash
python -m llm_belief_mi_test.cli \
  --dataset arc-challenge --limit 50 \
  --k 10 --n 2 --load-in-4bit \
  --temperature 0.9 --max-tokens 30 \
  --output outputs/results/arc_challenge_50.csv
```

Expected time: ~30 minutes  
Expected results:
- Accuracy: ~50-65%
- MI: ~0.3-0.8 bits average
- Agreement: ~0.4-0.7

### ❌ If Results Still Bad (MI=0, agreement=1.0):

Something is still wrong. Check:

1. **Is temperature being applied?**
   ```bash
   python test_temperature_diversity.py
   ```

2. **Check actual CSV output:**
   ```bash
   cat outputs/results/test_corrected.csv
   ```
   - Are answers all the same?

3. **Run diagnostic:**
   ```python
   # Generate same question 5 times
   # Should get different answers with temp=0.9
   ```

---

## Summary

**Status**: ✅ Bugs fixed, parameters corrected

**Paper's exact settings**:
- k=10 (chains)
- n=2 (chain length)
- temperature=0.9
- max_tokens=30 (our optimization)

**What to run now:**
```bash
# This command - should take ~3 minutes
python -m llm_belief_mi_test.cli --dataset arc-easy --limit 5 --k 10 --n 2 --load-in-4bit --temperature 0.9 --max-tokens 30 --output outputs/results/test_corrected.csv

# Then check
cat outputs/results/test_corrected.json
```

**Expected**: MI>0.1, agreement<0.9, accuracy>0.3

**If good**: Proceed to 50, then 200, then full!

**If bad**: Debug with `test_temperature_diversity.py`

---

**You're ready to go!** 🚀

