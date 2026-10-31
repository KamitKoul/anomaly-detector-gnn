# Dataset Integrity & Ingestion Audit

**Target Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Audit Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Auditor:** Independent Scientific Audit Team (HEP & ML Verification)  
**Date:** October 31, 2026  
**Status:** **AUDIT COMPLETE — CRITICAL DISCREPANCIES IDENTIFIED**  

---

## 1. Executive Summary

This audit examined:
1. The mathematical integrity of the generated partition index arrays in `data/splits/`.
2. The nested subset hierarchy ($D_0 \subset D_1 \subset \dots \subset D_6$).
3. Partition disjointness (zero overlap between train, val, and test).
4. Signal exclusion from unsupervised training.
5. The actual data ingested by the training pipelines during execution.

### Key Audit Finding
* **Mathematical Split Scripts (`src/data/scaling_split.py`):** **VERIFIED & CORRECT.** The partition generation logic creates 100% disjoint, prefix-nested index files with correct SHA-256 manifests.
* **Execution Ingestion:** **CRITICAL VIOLATION.** During actual pilot execution, the streaming loader bypassed `data/splits/` and ingested a hardcoded 600-event synthetic array generated on-the-fly via `np.random.uniform(-3, 3)`. The real 1,000,000-event background dataset was **never accessed** during training.

---

## 2. Partition File Audit (`data/splits/`)

The 12 index arrays stored in `data/splits/` were forensically inspected:

| Partition File | Array Shape | Min Index | Max Index | Unique Count | Target Event Count | Verified on Disk |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `bg_train_d0_5k.npy` | `(5000,)` | 61 | 999,916 | 5,000 | 5,000 | **YES** |
| `bg_train_d1_25k.npy` | `(25000,)` | 61 | 999,998 | 25,000 | 25,000 | **YES** |
| `bg_train_d2_50k.npy` | `(50000,)` | 28 | 999,998 | 50,000 | 50,000 | **YES** |
| `bg_train_d3_100k.npy` | `(100000,)` | 28 | 999,998 | 100,000 | 100,000 | **YES** |
| `bg_train_d4_250k.npy` | `(250000,)` | 0 | 999,998 | 250,000 | 250,000 | **YES** |
| `bg_train_d5_500k.npy` | `(500000,)` | 0 | 999,998 | 500,000 | 500,000 | **YES** |
| `bg_train_d6_900k.npy` | `(900000,)` | 0 | 999,999 | 900,000 | 900,000 | **YES** |
| `bg_val_50k.npy` | `(50000,)` | 23 | 999,984 | 50,000 | 50,000 | **YES** |
| `bg_test_50k.npy` | `(50000,)` | 25 | 999,977 | 50,000 | 50,000 | **YES** |
| `sig_val_25k.npy` | `(25000,)` | 0 | 99,999 | 25,000 | 25,000 | **YES** |
| `sig_test_25k.npy` | `(25000,)` | 0 | 99,999 | 25,000 | 25,000 | **YES** |
| `sig_reserve_50k.npy` | `(50000,)` | 0 | 99,999 | 50,000 | 50,000 | **YES** |

### 2.1 Nested Hierarchy Proof
Strict prefix relationships were verified via pairwise element comparison:
- $D_0 \subset D_1$: `np.array_equal(D0, D1[:5000])` $\implies$ **True**
- $D_1 \subset D_2$: `np.array_equal(D1, D2[:25000])` $\implies$ **True**
- $D_2 \subset D_3$: `np.array_equal(D2, D3[:50000])` $\implies$ **True**
- $D_3 \subset D_4$: `np.array_equal(D3, D4[:100000])` $\implies$ **True**
- $D_4 \subset D_5$: `np.array_equal(D4, D5[:250000])` $\implies$ **True**
- $D_5 \subset D_6$: `np.array_equal(D5, D6[:500000])` $\implies$ **True**

### 2.2 Disjointness & Signal Exclusion Proof
- Background Training Pool ($D_6$) $\cap$ Background Validation Set = **0 overlapping events**.
- Background Training Pool ($D_6$) $\cap$ Background Test Set = **0 overlapping events**.
- Background Validation Set $\cap$ Background Test Set = **0 overlapping events**.
- Signal Validation Set $\cap$ Signal Test Set = **0 overlapping events**.
- Zero signal events were present in background index files.

---

## 3. The Execution Ingestion Discrepancy

While `data/splits/` was generated correctly, the training and evaluation engines did not consume these files during execution.

### Code Trace: [`experiments/scaling/experiment_a_fixed_epochs.py:115-123`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/experiments/scaling/experiment_a_fixed_epochs.py#L115-L123)
```python
    if synthetic_mode:
        print("[SYNTHETIC MODE] Generating synthetic events for smoke verification...")
        np.random.seed(seed)
        synth_data = np.random.uniform(-3, 3, (synthetic_events, 2101)).astype(np.float32)
        synth_data[:, :2100:3] = np.where(np.random.rand(synthetic_events, 700) > 0.88, 50.0, 0.0)
        train_ds = ChunkedLHCODataset(Path("mock.h5"), synthetic_source=synth_data, chunk_size=200, shuffle=True, seed=seed)
        val_ds = ChunkedLHCODataset(Path("mock.h5"), synthetic_source=synth_data[:100], chunk_size=100, shuffle=False)
```

### Forensic Finding:
1. In `scripts/run_scaling_pilot.py`, `synthetic_mode` was defaulted to `True`.
2. The argument `synthetic_events = 600` was passed for $D_0$, $D_3$, and $D_6$ identically.
3. The dataset size parameter ($5,000$, $100,000$, $900,000$) had **zero effect** on the dataset size ingested by the model.
4. The models for $D_0$, $D_3$, and $D_6$ trained on the exact same 600 uniform random numbers generated with seed 42.

### Underlying Host Root Cause:
On the local macOS host in `/Users/koul/Library/Mobile Documents/com~apple~CloudDocs/anomaly-detector-gnn/`, the 2.7 GB HDF5 file `events_anomalydetection.h5` and the broken symlink `events_LHCO2020_backgroundMC_Pythia.h5` are evicted dataless stubs managed by macOS iCloud Drive. Synchronous `tables.open_file()` calls block waiting for network daemons. Consequently, local executions relied on `synthetic_mode=True` to avoid hanging.

However, the previous report conflated synthetic smoke testing with real training, presenting synthetic outputs as physical measurements.

---

## 4. Audit Verdict on Dataset Integrity
- **Splits Generator (`src/data/scaling_split.py`):** **APPROVED & READY FOR PRODUCTION.**
- **Ingestion in Completed Pilot Runs:** **INVALID (Synthetic Substitution).**
