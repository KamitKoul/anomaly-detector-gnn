# Scientific Blocker Register & Resolution Log

**Date:** 2026-10-31  
**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Scope:** Verification and Resolution of Audit Blockers BLK-01 through BLK-07 and Newly Identified Systemic Defects.

---

## 1. Status Register of Historical Blockers

| ID | Description | Root Cause | Remediation & Resolution | Status |
|---|---|---|---|---|
| **BLK-01** | Synthetic mode generated 600 uniform random events and ignored the requested dataset tier | In `experiment_a_fixed_epochs.py` and `experiment_b_fixed_updates.py`, `--synthetic` created 600 uniform mock events and overrode tier logic. | Synthetic generation isolated behind `--synthetic` and tagged `[SYNTHETIC_DEBUG_ONLY]`. All production runs fail closed if real HDF5 data is missing. | **RESOLVED** |
| **BLK-02** | Only 6 of 42 runs executed in synthetic mode; full matrix claimed complete in monograph | Experimental suite had only been executed in local synthetic mode. Results were extrapolated across all tiers. | Created authoritative [`results/real_data_validation/run_registry.csv`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/results/real_data_validation/run_registry.csv) explicitly distinguishing `COMPLETED` from `NOT_STARTED`. Large-scale runs properly gated to cloud GPUs. | **RESOLVED** |
| **BLK-03** | Evaluation capped at 500 background and 250 signal synthetic events | `evaluation/evaluate_scaling_study.py` truncated input events to 500 / 250 when debug flags were present. | Removed slicing. Evaluation executes strictly against full 50,000 background (`bg_test_50k.npy`) and 25,000 signal (`sig_test_25k.npy`) collision events. | **RESOLVED** |
| **BLK-04** | Top 1% JSD returned `(0.0, 9999.0)` when $N < 10$; mock RSD scalar `0.98` used | Misleading guard clause fabricated perfect mass invariance. Mock RSD scalar scaled raw mass by `0.98`. | Removed guard clause; returns `float('nan')` with status `"INSUFFICIENT_STATISTICS"`. Mock scalar removed. Real Top-1% JSD evaluated ($0.013763$, $1/\text{JSD} = 72.7$) over 500 surviving events. | **RESOLVED** |
| **BLK-05** | Local HDF5 files unavailable due to macOS iCloud APFS dataless stubs | Workspace is located inside iCloud Drive (`~/Library/Mobile Documents/com~apple~CloudDocs/`). macOS APFS evicts files when storage is tight. | Added `st_blocks > 0` validation to fail closed immediately. Hydrated both files locally (2.51 GB background, 1.57 GB signal) with physical block allocation verified. | **RESOLVED** |
| **BLK-06** | Single-seed bootstrap uncertainty reported as multi-seed model variance | Finite-sample test set bootstrap percentiles were mislabeled as training seed standard deviations. | Bootstrapping strictly documented as event-level sample efficiency confidence intervals; multi-seed variance strictly requires independent runs across Seeds 42, 43, and 44. | **RESOLVED** |
| **BLK-07** | Optimizer updates truncated (36 updates for Exp A, 30 for Exp B) in synthetic runs | Synthetic loops ran for arbitrary truncated steps rather than mathematical formulas. | Update budgets strictly tied to mathematics ($U(D) = 20 \times \lfloor D/32 \rfloor$ for Exp A; 10,000 updates for Exp B). Full-budget D0 execution executing exactly 3,120 updates. | **RESOLVED** |

---

## 2. Status Register of Newly Discovered Systemic Defects

| ID | Description | Root Cause | Remediation & Resolution | Status |
|---|---|---|---|---|
| **NEW-01** | PyTables 2D CArray Point Indexing Defect | PyTables 3.x treats `carray[index_array]` on 2D arrays as coordinate point selection across both dimensions, failing with `ValueError: operands could not be broadcast together with shapes (2000,) (2,)`. | Applied explicit column slicing `carray[index_array, :]` across `src/streaming_data_loader.py` and `evaluation/evaluate_scaling_study.py`. | **RESOLVED** |
| **NEW-02** | APFS Low Headroom Autonomous Eviction | macOS APFS volume has 9.5 GiB available (96% full). When available storage is $<10$ GiB, `bird` daemon aggressively purges $>1$ GB files in `com~apple~CloudDocs/`. | Fail-closed checks prevent hanging. Files hydrated via `brctl download` and monitored during execution. Prepared cloud runner `notebooks/phase4_scaling.ipynb` for non-evictable execution. | **RESOLVED / MITIGATED** |
| **NEW-03** | Scale Imbalance Dominating Composite Scores | Angular error residuals ($S_{\text{ang}} \sim 0.16-0.35$) are $6.2-8.9\times$ larger than $p_T$ residuals ($S_{pT} \sim 0.026-0.039$). | Audited and mathematically documented in `SCORE_DEFINITION_AUDIT.md`. Proved that angular error contributes $>60\%$ of $S_{\text{weighted}}$ variance, pulling AUC below 0.5. | **DOCUMENTED & AUDITED** |
| **NEW-04** | Angular Anti-Discrimination Mechanism | Boosted 2-prong signal subjets are tightly collimated ($\Delta R < 0.4$), enabling the GAT to reconstruct spatial coordinates with lower error than wide-angle diffuse QCD radiation. | Investigated across 7 hypotheses in `ANGULAR_SCORE_INVESTIGATION.md`. Confirmed physical origin (multiplicity $182.7$ vs $139.7$); scores kept independent. | **RESOLVED** |
