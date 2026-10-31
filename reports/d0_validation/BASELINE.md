# Baseline & Repository Safety Audit

**Date:** 2026-10-31  
**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Git Branch:** `main`  
**Working Tree Status:** Clean with local modifications staged in recovery workspaces.

---

## 1. Commit and Working Tree State

The repository is on branch `main` at commit:
```text
b98d17d174d3ebaef4f5ea080357824600400672
```

Working tree status confirms that all historical files and reports remain preserved. No historical synthetic results or past logs have been deleted or overwritten.

---

## 2. Inventory of Existing Modifications

The following production modules were remediated during the previous recovery session to establish physical real-data loading and fail-closed integrity:
1. `src/streaming_data_loader.py`:
   - Fixed PyTables 2D CArray coordinate point-indexing defect (`node[sorted_idx, :]`).
   - Added APFS dataless stub detection (`os.stat().st_blocks > 0`) in `_verify_file_readiness()`.
   - Preserved `drop_last=False` for general DataLoader and enforced `drop_last=True` for training loops.
2. `experiments/scaling/experiment_a_fixed_epochs.py`:
   - Removed 10,000-event validation truncation slice; enforced evaluation on all 50,000 held-out background validation events (`bg_val_50k.npy`).
   - Added fail-closed checks asserting physical HDF5 dataset existence and hydration.
   - Added `--bg-file`, `--data-dir`, and `--epochs` CLI arguments.
3. `experiments/scaling/experiment_b_fixed_updates.py`:
   - Enforced evaluation on full 50,000 background validation events.
   - Verified 10,000 optimizer update budget and epoch recycling formula.
4. `evaluation/evaluate_scaling_study.py`:
   - Fixed PyTables 2D CArray indexing on test slices (`bg_node[bg_test_idx, :]` and `sig_node[sig_test_idx, :]`).
   - Eliminated fabricated guard clause `(0.0, 9999.0)`; returns `NaN` with structured status `"INSUFFICIENT_STATISTICS"`.
   - Eliminated mock RSD scalar scaling `0.98`.
   - Removed silent 500/250 event truncation, defaulting to 50,000 BG and 25,000 Signal test events.
5. `scripts/download_datasets.py`:
   - Added `st_blocks > 0` validation.

---

## 3. Verification of Previous Smoke-Test Artifacts

The single-run smoke test (Exp A, D0, Seed 42, 1 Epoch) artifacts saved in `results/real_data_validation/` were audited and their cryptographic SHA-256 digests verified:

| File | Expected SHA-256 Digest | Verified? |
|---|---|---|
| `results/real_data_validation/checkpoints/expA_d0_5k_s42_checkpoint_best.pt` | `3b704d956850ab7548b54bff5fe6a66dbd675b3f3377ec3ad8ad587ddb9a8104` | **YES** |
| `results/real_data_validation/checkpoints/expA_d0_5k_s42_checkpoint_final.pt` | `ce8534efe69ee33f9219e21f4f50a678992c8963f35480c958150be7006fbad5` | **YES** |
| `results/real_data_validation/execution_logs/expA_d0_5k_s42_training_log.json` | `5060affbe409d81fd8655cde75b79c683471903331a284704ea518bd84b1e344` | **YES** |
| `results/real_data_validation/metrics/expA_d0_5k_s42_eval_metrics.json` | `071677a3ed7d305db307a7a324ea685fe0ed430def768d4dd9176e0d9a04e42a` | **YES** |

---

## 4. Physical Dataset Manifests

The physical dataset assets remain hydrated and intact:
- **Background (`data/events_LHCO2020_backgroundMC_Pythia.h5`):** 2,697,677,287 bytes, 5,268,904 blocks, SHA-256: `97edb2ac7a894102a5c8377e489c73ae871d487d114430e917b9f684dea36ed4`.
- **Signal (`data/W_prime_signal.h5`):** 1,681,638,808 bytes, 3,284,456 blocks, SHA-256: `624afc85740010fd4e0668ca772784785918891b5bd0e22ca07a4a5ffc7346dc`.

All historical synthetic and recovery artifacts are preserved.
