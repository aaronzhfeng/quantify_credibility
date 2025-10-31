# Results Folder Structure

The `outputs/results/` folder is now organized for better management of evaluation results and ablation experiments.

## **📁 Current Organization**

```
outputs/results/
├── arc_challenge/          # ARC-Challenge dataset results
│   ├── greedy_500.csv
│   ├── greedy_500.json
│   ├── selfcons_500.csv
│   ├── selfcons_500.json
│   ├── mi_500.csv
│   ├── mi_500.json
│   ├── semantic_entropy_500.csv
│   ├── semantic_entropy_500.json
│   ├── self_verification_500.csv
│   └── self_verification_500.json
│
├── arc_easy/               # ARC-Easy dataset results
│   ├── greedy_500.csv
│   ├── greedy_500.json
│   ├── selfcons_500.csv
│   ├── selfcons_500.json
│   ├── mi_500.csv
│   ├── mi_500.json
│   ├── semantic_entropy_500.csv
│   ├── semantic_entropy_500.json
│   ├── self_verification_500.csv
│   └── self_verification_500.json
│
├── openbookqa/             # OpenBookQA dataset results
│   ├── greedy_500.csv
│   ├── greedy_500.json
│   ├── selfcons_500.csv
│   ├── selfcons_500.json
│   ├── mi_500.csv
│   ├── mi_500.json
│   ├── semantic_entropy_500.csv
│   ├── semantic_entropy_500.json
│   ├── self_verification_500.csv
│   └── self_verification_500.json
│
├── test/                   # Test runs (quick verification)
│   ├── test_greedy.csv
│   ├── test_greedy.json
│   ├── test_mi.csv
│   ├── test_mi.json
│   ├── test_mi_prompts.csv
│   ├── test_mi_prompts.json
│   ├── test_fixed_prompts.csv
│   └── test_fixed_prompts.json
│
└── ablation/               # Ablation study results
    ├── temperature/        # Temperature ablation (T=0.5, 0.9, 1.3)
    │   ├── temp0.5.csv
    │   ├── temp0.5.json
    │   ├── temp0.9.csv
    │   ├── temp0.9.json
    │   ├── temp1.3.csv
    │   └── temp1.3.json
    │
    ├── k_chains/           # Number of chains ablation (k=5, 10, 20)
    │   ├── k5.csv
    │   ├── k5.json
    │   ├── k10.csv
    │   ├── k10.json
    │   ├── k20.csv
    │   └── k20.json
    │
    ├── n_length/           # Chain length ablation (n=2, 3, 4)
    │   ├── n2.csv
    │   ├── n2.json
    │   ├── n3.csv
    │   ├── n3.json
    │   ├── n4.csv
    │   └── n4.json
    │
    ├── mi_method/          # MI estimator ablation (listing vs plugin)
    │   ├── listing.csv
    │   ├── listing.json
    │   ├── plugin.csv
    │   └── plugin.json
    │
    ├── confidence_method/  # Confidence conversion ablation
    │   ├── inverse.csv
    │   ├── inverse.json
    │   ├── exp.csv
    │   ├── exp.json
    │   ├── normalized.csv
    │   └── normalized.json
    │
    └── answer_format/      # Answer format ablation (strict vs codeblock)
        ├── strict.csv
        ├── strict.json
        ├── codeblock.csv
        └── codeblock.json
```

---

## **🎯 Benefits of This Organization**

### **1. Clear Separation**
- **By Dataset**: Each dataset has its own folder (arc_challenge, arc_easy, openbookqa)
- **By Purpose**: Test runs separate from production results
- **By Experiment**: Ablation studies organized by parameter type

### **2. Easy Navigation**
```bash
# Compare all methods for a specific dataset
python scripts/compare_results.py outputs/results/arc_challenge/*.json

# Analyze temperature ablation
python scripts/compare_results.py outputs/results/ablation/temperature/*.json

# View test results
ls outputs/results/test/
```

### **3. Scalability**
- Adding new datasets? Just create a new folder
- Running more ablations? Each parameter has its own subfolder
- Easy to archive or delete specific experiments

### **4. Prevents Clutter**
- No more flat directory with 100+ files
- Files grouped by logical categories
- Easy to find specific results

---

## **📊 Comparison Commands**

### **Compare Methods Within a Dataset**
```bash
# ARC-Challenge: All 5 methods
python scripts/compare_results.py outputs/results/arc_challenge/*.json

# ARC-Easy: All 5 methods
python scripts/compare_results.py outputs/results/arc_easy/*.json

# OpenBookQA: All 5 methods
python scripts/compare_results.py outputs/results/openbookqa/*.json
```

### **Compare Across Datasets (Same Method)**
```bash
# MI method across all datasets
python scripts/compare_results.py \
  outputs/results/arc_challenge/mi_500.json \
  outputs/results/arc_easy/mi_500.json \
  outputs/results/openbookqa/mi_500.json

# Greedy baseline across all datasets
python scripts/compare_results.py \
  outputs/results/*/greedy_500.json
```

### **Ablation Analysis**
```bash
# Temperature sensitivity
python scripts/compare_results.py outputs/results/ablation/temperature/*.json

# Number of chains impact
python scripts/compare_results.py outputs/results/ablation/k_chains/*.json

# Chain length effect
python scripts/compare_results.py outputs/results/ablation/n_length/*.json

# All ablations together
python scripts/compare_results.py outputs/results/ablation/*/*.json
```

---

## **🔄 Migration Summary**

### **Files Moved:**
- ✅ **30 dataset files** → Organized by dataset (arc_challenge/, arc_easy/, openbookqa/)
- ✅ **8 test files** → Moved to test/
- ✅ **Ablation structure** → Created organized subfolders

### **Before Reorganization:**
```
outputs/results/
├── arc_challenge_greedy_500.csv
├── arc_challenge_selfcons_500.csv
├── arc_challenge_mi_500.csv
├── ...
├── openbookqa_greedy_500.csv
├── test_greedy.csv
└── (38 files in flat structure)
```

### **After Reorganization:**
```
outputs/results/
├── arc_challenge/  (10 files)
├── arc_easy/       (10 files)
├── openbookqa/     (10 files)
├── test/           (8 files)
└── ablation/       (organized subfolders)
```

---

## **🚀 Usage Examples**

### **Running New Experiments**

**Standard evaluation (uses dataset folders):**
```bash
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa \
  --k 10 --n 2 --temperature 0.9 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/openbookqa/mi_500.csv
```

**Ablation study (uses ablation subfolders):**
```bash
python -m llm_belief_mi_test.cli \
  --method mi --dataset openbookqa \
  --k 10 --n 2 --temperature 1.3 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/ablation/temperature/temp1.3.csv
```

**Test run (uses test folder):**
```bash
python -m llm_belief_mi_test.cli \
  --method greedy --dataset openbookqa --limit 5 \
  --load-in-4bit --max-tokens 10 --answer-format strict \
  --output outputs/results/test/quick_test.csv
```

---

## **📝 Notes**

- All commands in the README have been updated to reflect this new structure
- Detailed logs are still saved to `outputs/logs/{run_name}/`
- Visualization scripts automatically work with the new structure
- The folder organization is created automatically when running commands

---

**Total structure:** 4 dataset folders + 1 test folder + 6 ablation subfolders = **Clean, organized, scalable!** ✨

