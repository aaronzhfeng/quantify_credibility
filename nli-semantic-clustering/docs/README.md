# NLI Semantic Clustering - Documentation Index

This folder contains chronologically numbered documentation following the project's development timeline.

---

## 📚 Document Index

### Phase 1: Setup & Migration (01-02)

| # | Document | Description |
|---|----------|-------------|
| **01** | [nli_module_migration_summary.md](01_nli_module_migration_summary.md) | Complete migration summary: what was extracted, folder structure, quick start |
| **02** | [data_guide.md](02_data_guide.md) | Data requirements: log files vs raw datasets, how to get data |

### Phase 2: Diagnosis & Solution (03-04)

| # | Document | Description |
|---|----------|-------------|
| **03** | [nli_clustering_accuracy_ece_diagnosis.md](03_nli_clustering_accuracy_ece_diagnosis.md) | Root cause analysis: why accuracy drops and ECE spikes, the bidirectional vs unidirectional issue |
| **04** | [using_nli_grading_mode.md](04_using_nli_grading_mode.md) | How to use the fix: `--use-nli-grading` flag, testing, expected results |

### Phase 3: Experiments & Analysis (05+)

| # | Document | Description |
|---|----------|-------------|
| **05** | [argmax_mode_experiment_results.md](05_argmax_mode_experiment_results.md) | Argmax mode experiments on TriviaQA vs SQuAD v2, root cause analysis of dataset differences |

---

## 🎯 Quick Navigation

### For First-Time Users
1. Start with [01_nli_module_migration_summary.md](01_nli_module_migration_summary.md) - understand what this module is
2. Read [02_data_guide.md](02_data_guide.md) - get your data ready
3. Follow [../QUICKSTART.md](../QUICKSTART.md) - 5-minute quick start

### For Debugging
1. Read [03_nli_clustering_accuracy_ece_diagnosis.md](03_nli_clustering_accuracy_ece_diagnosis.md) - understand the problem
2. Apply [04_using_nli_grading_mode.md](04_using_nli_grading_mode.md) - test the solution

### For Implementation
1. Review [03_nli_clustering_accuracy_ece_diagnosis.md](03_nli_clustering_accuracy_ece_diagnosis.md) - technical details
2. Check [04_using_nli_grading_mode.md](04_using_nli_grading_mode.md) - code examples and API

---

## 📝 Numbering Convention

Documents are numbered chronologically (01-05, etc.) in the order they were created. This provides a historical timeline of the project's development:

- **01-02**: Initial setup and data preparation
- **03-04**: Problem diagnosis and solution implementation
- **05+**: Experiments and analysis results

When adding new documentation:
1. Assign the next sequential number (05, 06, etc.)
2. Use lowercase with underscores (snake_case)
3. Use descriptive names that clearly indicate content
4. Update this README with the new document

---

## 🔗 Related Documentation

- [../README.md](../README.md) - Main project README with debugging guide
- [../QUICKSTART.md](../QUICKSTART.md) - 5-minute quick start guide
- [../llm-belief-mi-test/COMMANDS_NLI.md](../../llm-belief-mi-test/COMMANDS_NLI.md) - Original NLI commands in main repo

---

## 📖 Document Summaries

### 01 - NLI Module Migration Summary
**Purpose**: Overview of what was extracted from main repo and why  
**Key Topics**: Folder structure, extracted code, quick start commands  
**When to Read**: First time using this module

### 02 - Data Guide  
**Purpose**: Explain data requirements and how to obtain data  
**Key Topics**: Log files vs raw datasets, data structure, copying commands  
**When to Read**: Before running any experiments

### 03 - NLI Clustering Accuracy & ECE Diagnosis
**Purpose**: Root cause analysis of accuracy drop and ECE spike  
**Key Topics**: Bidirectional vs unidirectional entailment, mathematical formulation, why it fails  
**When to Read**: Want to understand the problem deeply

### 04 - Using NLI Grading Mode
**Purpose**: How to use the fix (`--use-nli-grading` flag)  
**Key Topics**: Usage examples, expected results, testing, migration path  
**When to Read**: Ready to test the solution

### 05 - Argmax Mode Experiment Results
**Purpose**: Document argmax mode experiments comparing TriviaQA vs SQuAD v2  
**Key Topics**: Argmax vs soft threshold, unanswerable questions issue, dataset characteristics, recommendations  
**When to Read**: Understanding why NLI helps some datasets but not others

---

**Last Updated**: December 5, 2024

