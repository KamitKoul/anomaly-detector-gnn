# Repository Baseline & Safety Report

**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Current Branch:** `main`  
**Audit Status:** Initiated Recovery from Verdict `C — SCIENTIFICALLY BLOCKED`  
**Date:** 2026-10-31  

---

## 1. Executive Summary

This document establishes the official experimental baseline for recovering the **LHC Olympics 2020 PointNet–GAT Background-Data Scaling Study** following an independent scientific audit that assigned verdict **C — SCIENTIFICALLY BLOCKED**.

The prior scaling campaign (`reports/scaling/` and `results/scaling/`) generated metrics and reports from synthetic mock data (uniform random noise Arrays of length 500/600), bypassing real LHC collision data. All historical artifacts in `reports/scaling/` and `results/scaling/` are officially classified as **INVALID FOR PHYSICS CONCLUSIONS**.

A dedicated, isolated recovery workspace has been established:
- `reports/real_data_validation/` (Audit and recovery monographs)
- `results/real_data_validation/` (Verified execution logs, checkpoints, predictions, metrics, and cryptographic hashes)

---

## 2. Git Working-Tree Baseline

- **Repository Root:** `https://github.com/KamitKoul/anomaly-detector-gnn`
- **Head Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`
- **Git Status Summary:** Clean recovery tracking with tracked infrastructure modifications and isolated untracked validation spaces.

### Inventory of Critical Core Modules

| Functional Area | Primary Source Files | Baseline Purpose |
|---|---|---|
| **Data Streaming** | `src/streaming_data_loader.py` | Chunked PyTables HDF5 reader & on-the-fly PyG graph builder |
| **Scaling Training Engine** | `src/scaling_engine.py` | Model instantiation (7,957 parameters), train step, validation evaluation, checkpoint manager |
| **Fixed Epochs Experiment (A)** | `experiments/scaling/experiment_a_fixed_epochs.py` | 20-epoch ladder across tiers D0–D6 with Adam ($\text{lr}=10^{-3}$) |
| **Fixed Updates Experiment (B)** | `experiments/scaling/experiment_b_fixed_updates.py` | 10,000-update budget with cyclical dataset re-sampling |
| **Frozen Evaluation** | `evaluation/evaluate_scaling_study.py` | Held-out evaluation on 50k background and 25k signal test events |
| **Partition Generation** | `src/data/scaling_split.py` | 12 deterministic `.npy` partition index files |
| **Orchestration Scripts** | `scripts/run_scaling_pilot.py`<br>`scripts/run_scaling_matrix.py` | Automated 6-run pilot gate and full 42-run matrix orchestrator |
| **Cloud GPU Notebook** | `notebooks/phase4_scaling.ipynb` | Kaggle GPU execution harness with ext4 data discovery |

---

## 3. Inventory of Synthetic Switches & Fallback Mechanisms

Prior to pipeline remediation, seven critical synthetic switches and silent fallbacks compromised scientific validity:

| Location | Parameter / Switch | Historical Behavior | Remediation Action |
|---|---|---|---|
| `experiments/scaling/experiment_a_fixed_epochs.py:115` | `synthetic_mode=True` | Substituted 500 uniform random numbers, ignoring tier $D$ | Require `--synthetic` only for unit tests; fail closed on missing/dataless HDF5 |
| `experiments/scaling/experiment_b_fixed_updates.py:117` | `synthetic_mode=True` | Substituted 500 uniform random numbers, ignoring tier $D$ | Require `--synthetic` only for unit tests; fail closed on missing/dataless HDF5 |
| `evaluation/evaluate_scaling_study.py:321` | `--synthetic` | Capped test sample to 500 BG and 250 Signal uniform random events | Remove capping; enforce full 50k BG / 25k Signal; tag `--synthetic` as `DEBUG_ONLY` |
| `evaluation/evaluate_scaling_study.py:52` | `len(p_samples) < 10` | Returned guard clause fallback `0.0, 9999.0` | Return `NaN` and explicit status `"INSUFFICIENT_STATISTICS"` |
| `evaluation/evaluate_scaling_study.py:328` | Mock RSD Scaling | Multiplied `raw_m * 0.98` to simulate Recursive Soft Drop | Removed constant multiplier mock; report physical FastJet grooming or raw mass |
| `experiments/scaling/*.py` | `val_indices[:10000]` | Truncated validation partition from 50k to 10k | Removed slice; enforce full 50,000 held-out validation events |
| `scripts/download_datasets.py:31` | `is_valid()` | Checked `st_size > 100MB` but ignored `st_blocks == 0` | Check `st_blocks > 0` on macOS APFS to detect evicted cloud stubs |

---

## 4. Historical Scaling Artifacts Inventory & Invalidation Notice

The following historical artifacts produced under commit `b98d17d1` are preserved for provenance but marked **INVALID FOR SCIENTIFIC CONCLUSIONS**:

1. `reports/scaling/SCALING_STUDY_MONOGRAPH.md`: Invalidated (derived from 6 synthetic mock runs).
2. `reports/scaling/EXECUTIVE_SUMMARY.md`: Invalidated (uniform random noise metrics presented as scaling laws).
3. `results/scaling/pilot/pilot_summary.json`: Invalidated (synthetic test run with 36 Exp A and 30 Exp B updates).
4. `results/scaling/matrix_execution_log.json`: Invalidated (unexecuted runs populated with identical synthetic numbers).
5. All checkpoints in `results/scaling/experiment_a/*/checkpoint_*.pt` and `results/scaling/experiment_b/*/checkpoint_*.pt` dated prior to 2026-10-31: Invalidated (trained on synthetic noise).

---

## 5. Baseline Verification Protocol

The recovery pipeline enforces strict experimental controls:
- **Architecture Frozen:** `PointNetGATHybrid` (exactly 7,957 parameters).
- **Objective Frozen:** Pure kinematic reconstruction ($\lambda_{\text{MET}} = 0.0, \lambda_{\text{adv}} = 0.0$, DisCo disabled).
- **Optimizer Frozen:** Adam ($\text{lr} = 0.001$, $\text{weight\_decay} = 0.0$, clip norm $50.0$, batch size 32 with strict `drop_last = True`).
- **Fail Closed Discipline:** Execution immediately aborts with explicit error if real collision datasets are missing, evicted, or corrupted.
