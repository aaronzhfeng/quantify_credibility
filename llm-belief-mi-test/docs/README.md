# Documentation Index

This directory contains all project documentation organized chronologically by development phase.

## Quick Navigation

- **[Command References](#command-references)** - Step-by-step evaluation commands (kept in root for easy access)
- **[Phase 1: Core Implementation](#phase-1-core-implementation-01-10)** - Initial setup and core features
- **[Phase 2: Diagnostics & Fixes](#phase-2-diagnostics--fixes-11-17)** - Bug fixes and improvements
- **[Phase 3: Advanced Features](#phase-3-advanced-features-18-21)** - Multi-GPU and NLI analysis

---

## Command References

These files are kept in the root directory for easy access:

- **[`../README.md`](../README.md)** - Main project README with complete setup guide
- **[`../COMMANDS_MCQ.md`](../COMMANDS_MCQ.md)** - Commands for MCQ datasets (ARC, OpenBookQA)
- **[`../COMMANDS_OPENENDED.md`](../COMMANDS_OPENENDED.md)** - Commands for open-ended datasets (TriviaQA, SQuAD)
- **[`../COMMANDS_NLI.md`](../COMMANDS_NLI.md)** - Commands for NLI mutual entailment analysis

---

## Phase 1: Core Implementation (01-10)

### 01. Implementation Status
**[01_implementation_status.md](01_implementation_status.md)**
- Original implementation tracking
- Feature development timeline

### 02. Implementation Summary
**[02_implementation_summary.md](02_implementation_summary.md)**
- Summary of core features
- What was built and why

### 03. Implementation Complete Summary
**[03_implementation_complete_summary.md](03_implementation_complete_summary.md)**
- Completion status
- All features verified

### 04. Folder Structure
**[04_folder_structure.md](04_folder_structure.md)**
- Project structure and organization
- Where to find everything

### 05. Answer Format Guide
**[05_answer_format_guide.md](05_answer_format_guide.md)**
- How answer formats work (strict/default/codeblock)
- Format selection guide

### 06. Quick Reference
**[06_quick_reference.md](06_quick_reference.md)**
- Quick command reference
- Common operations

### 07. Complete Implementation Status
**[07_complete_implementation_status.md](07_complete_implementation_status.md)**
- Full feature checklist
- Verification guide

### 08. Ready to Rerun
**[08_ready_to_rerun.md](08_ready_to_rerun.md)**
- Ready to run guide
- Setup verification

### 09. What's Next
**[09_whats_next.md](09_whats_next.md)**
- Next steps and future work
- Planned features

### 10. Visualization Guide
**[10_visualization_guide.md](10_visualization_guide.md)**
- How to generate plots and visualizations
- Chart types and customization

---

## Phase 2: Diagnostics & Fixes (11-17)

### 11. Critical Fix Applied
**[11_critical_fix_applied.md](11_critical_fix_applied.md)**
- Important bug fixes
- What broke and how it was fixed

### 12. MI Prompt Fix Summary
**[12_mi_prompt_fix_summary.md](12_mi_prompt_fix_summary.md)**
- MI method prompt corrections
- Prompt engineering improvements

### 13. Prompt Fix Summary
**[13_prompt_fix_summary.md](13_prompt_fix_summary.md)**
- General prompt fixes
- Standardization across methods

### 14. Logging Status Final
**[14_logging_status_final.md](14_logging_status_final.md)**
- Logging system status
- What gets logged where

### 15. Add Remaining Logging
**[15_add_remaining_logging.md](15_add_remaining_logging.md)**
- Additional logging features
- Enhanced debugging capabilities

### 16. Logprob Diagnostic
**[16_logprob_diagnostic.md](16_logprob_diagnostic.md)**
- Log probability analysis and validation
- Understanding model confidence

### 17. Reorganization Summary
**[17_reorganization_summary.md](17_reorganization_summary.md)**
- Code reorganization notes
- New structure and naming conventions

---

## Phase 3: Advanced Features (18-25)

### 18. Multi-GPU Support
**[18_multi_gpu_logging_complete.md](18_multi_gpu_logging_complete.md)**
- Parallel evaluation across multiple GPUs
- 4× speedup for large datasets
- Automatic work distribution and merging

### 19. NLI Mutual Entailment - Overview
**[19_nli_mutual_entailment_summary.md](19_nli_mutual_entailment_summary.md)**
- What mutual entailment is and why it matters
- Two use cases: clustering and evaluation
- Time estimates and requirements
- Complete implementation guide

### 20. NLI Evaluation Enhancement
**[20_nli_evaluation_enhancement.md](20_nli_evaluation_enhancement.md)**
- Using NLI for semantic evaluation (not just clustering)
- Comparing exact match vs NLI-based correctness checking
- Expected accuracy improvements (+5-10%)
- Impact on research findings

### 21. NLI Implementation Summary (Post-hoc Analysis)
**[21_nli_implementation_summary.md](21_nli_implementation_summary.md)**
- Complete implementation details for post-hoc analysis
- What was implemented and why
- How to run analysis on existing logs
- Expected findings and next steps

### 22. Test Guide for NLI Clustering
**[22_test_nli_clustering.md](22_test_nli_clustering.md)**
- Step-by-step testing guide for live NLI clustering
- Syntax check and quick test commands
- Expected output and verification
- Troubleshooting common issues

### 23. NLI Clustering Implementation (Live)
**[23_nli_clustering_implementation.md](23_nli_clustering_implementation.md)**
- Complete implementation of live NLI clustering (Option 3)
- Semantic MI: measures semantic uncertainty vs string variation
- Modified functions in calibration.py and cli.py
- Research implications and ablation studies

### 24. Recalculate with NLI Guide
**[24_recalculate_nli_guide.md](24_recalculate_nli_guide.md)**
- Complete guide for post-hoc recalculation from logs
- 8× faster than re-running inference
- Threshold ablation and experimentation
- Example workflows and troubleshooting

### 25. NLI Recalculation Complete
**[25_nli_recalculation_complete.md](25_nli_recalculation_complete.md)**
- Final implementation summary
- Two approaches: live vs post-hoc
- Performance comparisons and research value
- Complete feature list

---

## Document Organization Guidelines

Following `.cursorrules`:

1. **Numbering**: Files use `##_descriptive_topic_name.md` format
2. **Naming**: Snake_case with clear, descriptive names
3. **Chronology**: Numbers reflect development timeline
4. **Index**: This README groups docs by phase with navigation links

### Adding New Documentation

When adding new documentation:

1. Assign next sequential number (22, 23, etc.)
2. Use snake_case: `##_clear_descriptive_name.md`
3. Place in `docs/` folder
4. Update this README in appropriate phase
5. Add one-line description

Example:
```bash
# Create new doc
touch docs/22_new_feature_guide.md

# Update docs/README.md
# Add link and description in appropriate phase
```

---

## Related Directories

- **[`../demo/`](../demo/)** - Interactive demo for understanding methods
- **[`../scripts/`](../scripts/)** - Analysis and visualization scripts
- **[`../outputs/`](../outputs/)** - Evaluation results, logs, and plots
- **[`../outputs/nli_analysis/`](../outputs/nli_analysis/)** - NLI analysis results

---

## Theory & Algorithms

For theoretical background, see:

- **[`../theory/`](../../theory/)** - Mathematical foundations (in parent directory)
  - `MI_ALGORITHMS.md` - Mutual information estimators
  - `MI_ECE_FORMULAS.md` - Calibration metrics
  - `MI_ESTIMATOR_EXAMPLE.md` - Worked examples

---

## Quick Starts by Task

### Run Evaluation
1. See **[`../README.md`](../README.md)** for setup
2. See **[`../COMMANDS_MCQ.md`](../COMMANDS_MCQ.md)** or **[`../COMMANDS_OPENENDED.md`](../COMMANDS_OPENENDED.md)** for commands

### Analyze Results
1. **[10_visualization_guide.md](10_visualization_guide.md)** - Generate plots
2. **[`../scripts/README.md`](../scripts/README.md)** - Analysis scripts

### Understand NLI Analysis
1. **[19_nli_mutual_entailment_summary.md](19_nli_mutual_entailment_summary.md)** - Overview
2. **[20_nli_evaluation_enhancement.md](20_nli_evaluation_enhancement.md)** - Evaluation use case
3. **[21_nli_implementation_summary.md](21_nli_implementation_summary.md)** - Implementation

### Diagnose Issues
1. **[16_logprob_diagnostic.md](16_logprob_diagnostic.md)** - Log probability issues
2. **[11_critical_fix_applied.md](11_critical_fix_applied.md)** - Known fixes

---

## Document History

This documentation evolved through three phases:

**Phase 1** (01-10): Core implementation of MI-based uncertainty quantification  
**Phase 2** (11-17): Bug fixes, diagnostics, and logging improvements  
**Phase 3** (18-25): Advanced features (Multi-GPU, NLI semantic clustering, post-hoc recalculation)  

### Phase 3 Breakdown:
- **18**: Multi-GPU parallelization (4× speedup)
- **19-21**: NLI post-hoc analysis from logs
- **22-23**: Live NLI clustering during inference (semantic MI)
- **24-25**: Post-hoc recalculation from logs (8× faster experimentation)

Each numbered document represents a milestone in the project's development.

