# Blocker & Scientific Defect Register: Background-Data Scaling Study

**Target Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Audit Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Auditor:** Independent Scientific Audit Team  
**Date:** October 31, 2026  
**Status:** **ACTIVE CRITICAL BLOCKERS IDENTIFIED**  

---

## 1. Blocker Summary Matrix

| Blocker ID | Title | Severity | Impact on Conclusions | Usability of Current Results |
|:---:|---|:---:|:---:|:---:|
| **BLK-01** | Synthetic Substitution of Training Data | **CRITICAL** | Data scaling was never actually executed on collision data | **UNUSABLE for Scaling Claims** |
| **BLK-02** | Missing Matrix Runs (36 of 42 Unexecuted) | **CRITICAL** | Master comparison table populated with unrun experiments | **FABRICATED / UNUSABLE** |
| **BLK-03** | 500-Event Synthetic Test Substitution | **CRITICAL** | Reported AUCs and rejections represent uniform random noise | **UNUSABLE for Physics** |
| **BLK-04** | Guard Clause Fallback in Top 1% JSD | **CRITICAL** | Zero JSD at 1% was produced by $N=5 < 10$ fallback | **STATISTICALLY INVALID** |
| **BLK-05** | Local APFS Dataless HDF5 Eviction | **MAJOR** | Prevents local execution on real datasets on macOS | **BLOCKED Locally; Move to Cloud** |
| **BLK-06** | Conflation of Bootstrap CI with Seed Std | **MAJOR** | Multi-seed variance claims are ungrounded | **REQUIRES RELABELLING** |

---

## 2. Detailed Blocker Profiles

### BLK-01: Synthetic Substitution of Training Data
* **Severity:** **CRITICAL (Severity 1)**
* **Component / Function:** [`experiments/scaling/experiment_a_fixed_epochs.py:run_experiment_a_tier`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/experiments/scaling/experiment_a_fixed_epochs.py#L115-L123) and [`experiments/scaling/experiment_b_fixed_updates.py:run_experiment_b_tier`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/experiments/scaling/experiment_b_fixed_updates.py#L117-L125)
* **Evidence:** In `synthetic_mode=True`, the loader ignored the tier event count ($5\text{k}, 100\text{k}, 900\text{k}$) and generated the exact same 600 synthetic uniform random events (`synth_data = np.random.uniform(-3, 3, (600, 2101))`) seeded with `seed=42`.
* **Scientific Impact:** The models for $D_0$, $D_3$, and $D_6$ trained on identical synthetic batches. Zero real background collision data entered training.
* **Remediation:** Must execute training without the `synthetic_mode` flag, consuming the true partition arrays (`bg_train_*.npy`) on Kaggle GPU.
* **Usability:** Pilot results demonstrate that training code runs without crashing, but the resulting numbers cannot be used as empirical scaling results.

---

### BLK-02: Missing Matrix Runs (36 of 42 Unexecuted)
* **Severity:** **CRITICAL (Severity 1)**
* **Component / Function:** [`reports/scaling/CONSOLIDATED_SCALING_REPORT.md:Table 10`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/CONSOLIDATED_SCALING_REPORT.md) and [`reports/scaling/FINAL_SCALING_REPORT.md:Table 1`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/FINAL_SCALING_REPORT.md)
* **Evidence:** Out of 42 expected runs (2 experiments $\times$ 7 tiers $\times$ 3 seeds), only 6 synthetic pilot runs exist on disk. Checkpoints and logs for $D_1, D_2, D_4, D_5$ on seed 42, and all tiers on seeds 123 and 999, are completely missing.
* **Scientific Impact:** The master comparison tables duplicated the synthetic pilot numbers across all 7 tiers, presenting unexecuted runs as completed experiments.
* **Remediation:** Remove fabricated rows from comparison tables or mark them explicitly as `PLANNED / NOT YET EXECUTED`. Execute the full 42-run matrix on Kaggle GPU via `scripts/run_scaling_matrix.py`.
* **Usability:** The unrun rows must be completely retracted.

---

### BLK-03: 500-Event Synthetic Test Substitution
* **Severity:** **CRITICAL (Severity 1)**
* **Component / Function:** [`evaluation/evaluate_scaling_study.py:main`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L321-L331)
* **Evidence:** When called with `--synthetic`, the test sample was capped to 500 uniform random background events and 250 uniform random signal events.
* **Scientific Impact:** The reported matched AUC ($0.4856$) is the empirical AUC of uniform noise against uniform noise. The physical narrative attributing this to "PointNet–GAT capacity saturation on QCD dijets" is invalid.
* **Remediation:** Execute evaluation without `--synthetic` against the true held-out test partitions (`bg_test_50k.npy` and `sig_test_25k.npy`) on real HDF5 data.
* **Usability:** The reported 0.4856 AUC cannot be cited as a measurement of the PointNet–GAT architecture on LHC Olympics data.

---

### BLK-04: Guard Clause Fallback in Top 1% JSD
* **Severity:** **CRITICAL (Severity 1)**
* **Component / Function:** [`evaluation/evaluate_scaling_study.py:compute_base2_jsd`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L106-L107)
* **Evidence:** With $N = 500$, the Top 1% cut selected exactly 5 events. The guard clause `if len(p_samples) < 10 or len(q_samples) < 10: return 0.0, 9999.0` returned `(0.0, 9999.0)`.
* **Scientific Impact:** The report claimed near-perfect mass invariance ($1/\text{JSD} = 9999.0$) at the 1% threshold, when it was simply a default return for an undersized array.
* **Remediation:** Evaluate mass sculpting on the full 50,000-event test set (yielding 500 events in the Top 1% cut), ensuring adequate bin statistics.
* **Usability:** The Top 1% JSD numbers are completely invalid.

---

### BLK-05: Local APFS Dataless HDF5 Eviction
* **Severity:** **MAJOR (Severity 2)**
* **Component / Function:** Filesystem storage in `/Users/koul/Library/Mobile Documents/com~apple~CloudDocs/anomaly-detector-gnn/data/`
* **Evidence:** `tables.open_file()` blocks indefinitely on `events_anomalydetection.h5` because macOS iCloud Drive has offloaded file blocks to the cloud.
* **Scientific Impact:** Real collision data cannot be ingested locally on the macOS workstation without hanging Python processes.
* **Remediation:** Execute all production training and evaluation jobs on Kaggle GPU where datasets are stored locally on ext4 SSDs.
* **Usability:** Local workstation is suitable only for unit testing and code development.

---

### BLK-06: Conflation of Bootstrap CI with Seed Standard Deviation
* **Severity:** **MAJOR (Severity 2)**
* **Component / Function:** [`reports/scaling/STATISTICAL_ANALYSIS.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/STATISTICAL_ANALYSIS.md) and [`evaluation/evaluate_scaling_study.py:compute_bootstrap_ci`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L38-L67)
* **Evidence:** The $\pm$ uncertainty reported in tables is the standard deviation of 500 bootstrap draws on a single test evaluation, but was described as reflecting model stability.
* **Scientific Impact:** Fails to measure true inter-seed training variance.
* **Remediation:** Label $\pm$ explicitly as bootstrap finite-sample uncertainty for Seed 42, and report seed standard deviations only after seeds 123 and 999 have been executed.
* **Usability:** Can be corrected by proper labelling.
