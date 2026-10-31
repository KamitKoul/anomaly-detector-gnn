# Forensic Re-Verification of Audit Findings

**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Auditor Status:** Senior ML Systems & Particle Physics Reproducibility Engineer  
**Date:** 2026-10-31  

---

## 1. Forensic Summary of Findings BLK-01 Through BLK-07

An exhaustive forensic examination of the source code, historical run logs, configuration manifests, and binary artifacts confirmed **all seven reported audit findings** and identified the exact underlying code mechanisms responsible.

| ID | Finding | Historical Code Path | Forensic Confirmation | Remediated Status |
|---|---|---|---|---|
| **BLK-01** | Synthetic mode generated 500/600 uniform random numbers, ignoring requested tier $D$ | `experiments/scaling/experiment_a_fixed_epochs.py:118`<br>`experiments/scaling/experiment_b_fixed_updates.py:120` | **CONFIRMED**: `np.random.uniform(-3, 3, (synthetic_events, 2101))` was hardcoded. Tiers D0 (5k), D3 (100k), and D6 (900k) all ran on the same 500/600 mock events. | **REMEDIATED**: Synthetic mode isolated strictly for unit tests. Tier sizes strictly enforced on disk. |
| **BLK-02** | Only 6 of 42 matrix runs were executed, all in synthetic mode | `results/scaling/matrix_execution_log.json`<br>`results/scaling/pilot/pilot_summary.json` | **CONFIRMED**: Checkpoints existed only for D0, D3, D6 ($s=42$) across Exp A & B (6 runs total). The other 36 runs were missing or synthetic copies. | **REMEDIATED**: Full 42-run matrix gated in `results/real_data_validation/run_registry.csv` with status `NOT_STARTED` pending GPU execution. |
| **BLK-03** | Evaluation used 500 background and 250 signal synthetic events | `evaluation/evaluate_scaling_study.py:323-326` | **CONFIRMED**: `n_bg = min(args.limit_bg, 500)` and `n_sig = min(args.limit_sig, 250)` forced evaluation on 500 BG and 250 Signal uniform random vectors. Empirical AUC $0.4856$ was noise vs noise. | **REMEDIATED**: Capping removed. Enforced full 50,000 held-out background and 25,000 signal events. Synthetic mode tagged `DEBUG_ONLY`. |
| **BLK-04** | Top-1% JSD returned `(0.0, 9999.0)` on sample size $< 10$ | `evaluation/evaluate_scaling_study.py:52-53` | **CONFIRMED**: `if len(p_samples) < 10 or len(q_samples) < 10: return 0.0, 9999.0`. For $N=500$, Top 1% had $N=5 < 10$, triggering this guard clause fallback. | **REMEDIATED**: Misleading guard clause removed. Returns `NaN` and explicit structured status `"INSUFFICIENT_STATISTICS"`. |
| **BLK-05** | Local HDF5 files unavailable due to macOS APFS dataless stub behavior | `data/*.h5`<br>`scripts/download_datasets.py:31` | **CONFIRMED**: `download_datasets.py:is_valid()` checked only `st_size > 100MB`, which returned `True` for dataless APFS files (`st_blocks == 0`). `tables.open_file` hung or failed. | **REMEDIATED**: Fixed `is_valid()` to verify `st_blocks > 0` on macOS. Materialized physical bytes on disk (`LHC Olympics 2020 Background MC Pythia.h5` and `W_prime_signal.h5`). |
| **BLK-06** | Single-seed bootstrap uncertainty reported as multi-seed variance | `evaluation/evaluate_scaling_study.py:72-113` | **CONFIRMED**: `compute_bootstrap_ci()` computed stratified test-set bootstrap error ($N=500$ resamples) on a single seed ($s=42$) and this was reported in monograph tables as model seed variation. | **REMEDIATED**: Separated within-run test bootstrap variance from cross-seed variance across independent seeds (42, 123, 999). |
| **BLK-07** | Actual pilot optimizer updates were 36 (Exp A) and 30 (Exp B) instead of intended budgets | `scripts/run_scaling_pilot.py:64, 79` | **CONFIRMED**: In synthetic pilot mode, `epochs=2` was hardcoded for Exp A (yielding $2 \times \lfloor 600/32 \rfloor = 36$ updates) and `total_updates=30` was hardcoded for Exp B. | **REMEDIATED**: Full training budgets enforced: Exp A executes $U(D) = 20 \times \lfloor D / 32 \rfloor$; Exp B executes exactly 10,000 updates. |

---

## 2. Comprehensive Repository-Wide Pattern Audit

Every occurrence of synthetic, mock, and fallback patterns across the codebase was inventoried and classified:

### 1. `synthetic_mode` & `synthetic_events`
- `src/streaming_data_loader.py`: Safe in unit tests when `synthetic_source` is explicitly supplied. Now checks `_verify_file_readiness()` when `synthetic_source is None`.
- `experiments/scaling/experiment_a_fixed_epochs.py`: Remediated. Defaults to `synthetic_mode=False`. Added `--bg-file`, `--data-dir` and strict fail-closed checks.
- `experiments/scaling/experiment_b_fixed_updates.py`: Remediated. Defaults to `synthetic_mode=False`. Added `--bg-file`, `--data-dir` and strict fail-closed checks.
- `scripts/run_scaling_pilot.py`: Remediated. Defaulted `synthetic_mode=False`. Added CLI flags.

### 2. `--synthetic` CLI Flag
- `evaluation/evaluate_scaling_study.py`: Remediated. Tagged with prominent console warning; adds `"is_debug_only": True` and `"is_synthetic": True` to JSON metadata; output cannot be ingested into scientific reports.
- `scripts/run_scaling_matrix.py`: Remediated. Defaults to `synthetic=False`. Passes real file paths to sub-processes.

### 3. `np.random.uniform` & `np.random.normal`
- `tests/test_*.py`: Permitted exclusively in unit tests for generating reproducible fixture tensors.
- `src/remediation_diagnostics.py`: Permitted exclusively in synthetic benchmark utilities for GPU profiling.
- Removed from all production training and evaluation execution paths.

### 4. `limit_bg` & `limit_sig`
- `evaluation/evaluate_scaling_study.py`: Defaults restored to 50,000 and 25,000. Any run with smaller subsets is automatically tagged `"is_debug_only": True`.

### 5. `fallback` & `mock`
- Removed the `0.0, 9999.0` mass-sculpting guard clause.
- Removed the constant `* 0.98` mock Recursive Soft Drop scaling.

---

## 3. End-to-End Execution Trace (Production Mode)

```text
1. CLI Invocation:
   python experiments/scaling/experiment_a_fixed_epochs.py --tier D0 --seed 42 --epochs 20
        ↓
2. File Verification Gate (Fail-Closed):
   Verifies: Path.exists() == True, is_file() == True, st_blocks > 0
   Dataset SHA-256: 97edb2ac7a894102a5c8377e489c73ae871d487d114430e917b9f684dea36ed4
        ↓
3. Partition Loading:
   Loads data/splits/bg_train_d0_5k.npy (5,000 indices in [0, 999,999])
   Loads data/splits/bg_val_50k.npy (50,000 indices in [0, 999,999])
        ↓
4. Chunked Streaming Ingestion:
   Reads 2,000-event chunks using PyTables node[sorted_idx, :]
   Multi-worker sharded without in-memory accumulation
        ↓
5. Dynamic Graph Construction:
   Vectorized particle extraction (pT > 0, eta, phi)
   Delta-R edge construction (threshold = 0.4)
   PyG Batch assembly (batch_size = 32, drop_last = True)
        ↓
6. Forward & Loss Calculation:
   PointNetGATHybrid (7,957 parameters)
   Pure kinematic reconstruction loss (recon=1.0, MET=0.0, adv=0.0)
        ↓
7. Backward & Optimizer Step:
   Gradient clipping at 50.0; Adam(lr=0.001) update
        ↓
8. Unsupervised Validation Evaluation:
   torch.no_grad() evaluation over all 50,000 validation events
   Checkpoint saved strictly on minimum validation loss
        ↓
9. Frozen Test Evaluation:
   50,000 test background events + 25,000 test signal events
   Computes Inclusive AUC, Matched-mass AUC, Rejection, JSD, 1/JSD
```
