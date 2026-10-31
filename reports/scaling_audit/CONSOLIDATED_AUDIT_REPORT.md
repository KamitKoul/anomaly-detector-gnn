# Consolidated Independent Scientific Audit: PointNet–GAT Background-Data Scaling Study

**Target Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Primary Audited Report:** `reports/scaling/CONSOLIDATED_SCALING_REPORT.md`  
**Audit Scope:** Full verification of experimental execution, data integrity, evaluation methodology, and published claims across the 42-run scaling matrix.  
**Lead Auditor:** Independent Senior Scientific & ML Verification Auditor  
**Date:** October 31, 2026  
**Final Audit Verdict:** **C. SCIENTIFICALLY BLOCKED**  

---

## Table of Contents

1. [Executive Summary & Final Audit Verdict](#1-executive-summary--final-audit-verdict)
2. [AUDIT 1 — Forensic Analysis of Identical Metrics Across Tiers](#2-audit-1--forensic-analysis-of-identical-metrics-across-tiers)
3. [AUDIT 2 — 42-Run Matrix Completeness & Execution Inventory](#3-audit-2--42-run-matrix-completeness--execution-inventory)
4. [AUDIT 3 — Dataset Partition Integrity vs Execution Ingestion](#4-audit-3--dataset-partition-integrity-vs-execution-ingestion)
5. [AUDIT 4 — Test Sample Size Discrepancy & 500-Event Mock Substitution](#5-audit-4--test-sample-size-discrepancy--500-event-mock-substitution)
6. [AUDIT 5 — ROC-AUC Orientation & Synthetic Noise Analysis](#6-audit-5--roc-auc-orientation--synthetic-noise-analysis)
7. [AUDIT 6 — Mass Sculpting Statistical Validity & Guard-Clause Fallback](#7-audit-6--mass-sculpting-statistical-validity--guard-clause-fallback)
8. [AUDIT 7 & 8 — Statistical Confounding & Checkpoint Audits](#8-audit-7--8--statistical-confounding--checkpoint-audits)
9. [Comprehensive Blocker & Defect Register](#9-comprehensive-blocker--defect-register)
10. [Verified Codebase Assets vs Required Cloud GPU Execution](#10-verified-codebase-assets-vs-required-cloud-gpu-execution)
11. [Actionable Remediation Roadmap](#11-actionable-remediation-roadmap)

---

## 1. Executive Summary & Final Audit Verdict

### 1.1 The Audit Mandate
An independent scientific audit was executed to determine whether the results presented in the *Consolidated Scientific Monograph: Background-Data Scaling Study for the LHC Olympics PointNet–GAT Anomaly Detector* were genuinely produced by independent training runs across the 1,000,000-event LHC Olympics dataset, or whether implementation errors, data substitutions, or reporting defects caused different experiment tiers to appear identical.

### 1.2 Final Verdict: **CLASSIFICATION C — SCIENTIFICALLY BLOCKED**

| Audit Dimension | Target Requirement | Forensic Reality | Classification |
|---|---|---|:---:|
| **1. Metric Diversity** | Independent training runs across D0–D6 | Identical numbers produced by hardcoded 600-event synthetic noise | **BLOCKED** |
| **2. Matrix Completeness** | 42 independent runs (2 x 7 x 3) | 0 completed on real data; 36 completely missing; 6 synthetic | **BLOCKED** |
| **3. Dataset Ingestion** | Streaming loader consuming D0–D6 | Partition files bypassed; synthetic mock data fed to model | **BLOCKED** |
| **4. Test Sample Size** | 50,000 background / 25,000 signal test set | Evaluated on 500 background / 250 signal uniform random floats | **BLOCKED** |
| **5. Discrimination AUC** | Valid physical anomaly detection on W' -> XY | 0.4856 AUC is the empirical AUC of random noise vs random noise | **BLOCKED** |
| **6. Mass Sculpting JSD** | Continuous JSD on 40-bin dijet mass histograms | Top 1% JSD = 0.0 is an unhandled N=5 < 10 guard-clause fallback | **BLOCKED** |
| **7. Multi-Seed Uncertainty** | Sample std (N-1) across seeds 42, 123, 999 | Seeds 123 and 999 unexecuted; bootstrap std presented as seed variance | **BLOCKED** |
| **8. Code Infrastructure** | Modular architecture, deterministic splits, unit tests | All 72 unit tests pass; architecture and splitting tools sound | **VERIFIED** |

**Verdict Statement:** The software architecture, mathematical split generators, and automated test suites in this repository are mathematically sound, modular, and ready for production. However, the reported experimental conclusions are **scientifically blocked from acceptance** because the published results were produced by a synthetic smoke-test harness running on uniform random numbers on macOS, which was subsequently multiplied across all 7 dataset tiers in the comparison tables.

---

## 2. AUDIT 1 — Forensic Analysis of Identical Metrics Across Tiers

### 2.1 The Empirical Anomaly
The published reports cited:
- Experiment A (D0, D3, D6): Validation loss was reported as **2.5881** across all tiers; matched ROC-AUC was **0.4856**; gradient norms were identical (18.75).
- Experiment B (D0, D3, D6): Validation loss was reported as **2.5144** across all tiers; matched ROC-AUC was **0.4593**; gradient norms were identical (18.12).
- Master Comparison Table (Table 10): Reported identical values across all seven tiers (D0 through D6) for both Experiment A and Experiment B.

### 2.2 Forensic Code Trace
Tracing the complete execution pipeline from configuration to final report:

```
[scripts/run_scaling_pilot.py]
          │
          ├──> Invokes run_experiment_a_tier(tier, seed=42, synthetic_mode=True, synthetic_events=600)
          │
[experiments/scaling/experiment_a_fixed_epochs.py:115-123]
          │
          ├──> Checks if synthetic_mode:
          │    np.random.seed(42)
          │    synth_data = np.random.uniform(-3, 3, (600, 2101))
          │    IGNORES tier_events (5,000 vs 100,000 vs 900,000)!
          │
          ├──> Trains PointNetGATHybrid initialized with seed=42 on identical 600 synthetic events
          │    Yields identical weights, identical grad norms (18.75), identical val loss (2.58813)
          │
[scripts/run_scaling_pilot.py:108-112]
          │
          ├──> Invokes evaluate_scaling_study.py --checkpoint ... --synthetic
          │
[evaluation/evaluate_scaling_study.py:321-331]
          │
          ├──> Checks if args.synthetic:
          │    np.random.seed(42)
          │    Generates identical 500 background and 250 signal uniform random floats
          │    Evaluates identical checkpoints against identical mock test arrays
          │    Yields identical ROC-AUC (0.48555 -> 0.4856) and identical JSD
          │
[reports/scaling/CONSOLIDATED_SCALING_REPORT.md]
          │
          └──> Master comparison table copies identical synthetic pilot numbers across D0..D6!
```

### 2.3 Checkpoint Weight Identity Proof
Pairwise parameter subtraction between `checkpoint_best.pt` for `expA_d0_5k_s42` and `expA_d3_100k_s42` yielded:
$$\max_{\theta} |\theta_{D0} - \theta_{D3}| = 3.28 \times 10^{-7}$$
This difference is bounded by 32-bit floating-point accumulation roundoff. The models did not converge to identical attractors; they executed identical matrix multiplications on identical synthetic arrays.

### 2.4 Cryptographic Run-Identity Table
| Run ID | Experiment | Dataset Tier | Dataset Size | Seed | Config Hash | Manifest Hash | Checkpoint Hash | Prediction Hash | Evaluation Hash | Actual Updates | Execution Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| expA_d0_5k_s42 | A | D0 | 5000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | fdb1c70e9ebfeae9 | NOT_SAVED_TO_DISK | 04d10b4e9476212e | 36 | COMPLETED_SYNTHETIC |
| expA_d0_5k_s123 | A | D0 | 5000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d0_5k_s999 | A | D0 | 5000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d1_25k_s42 | A | D1 | 25000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d1_25k_s123 | A | D1 | 25000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d1_25k_s999 | A | D1 | 25000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d2_50k_s42 | A | D2 | 50000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d2_50k_s123 | A | D2 | 50000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d2_50k_s999 | A | D2 | 50000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d3_100k_s42 | A | D3 | 100000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | d85278aba13fc53c | NOT_SAVED_TO_DISK | 621a155cf43ca3c0 | 36 | COMPLETED_SYNTHETIC |
| expA_d3_100k_s123 | A | D3 | 100000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d3_100k_s999 | A | D3 | 100000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d4_250k_s42 | A | D4 | 250000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d4_250k_s123 | A | D4 | 250000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d4_250k_s999 | A | D4 | 250000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d5_500k_s42 | A | D5 | 500000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d5_500k_s123 | A | D5 | 500000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d5_500k_s999 | A | D5 | 500000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d6_900k_s42 | A | D6 | 900000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | c5d5a693e68a18e0 | NOT_SAVED_TO_DISK | cedc08fb45a6a042 | 36 | COMPLETED_SYNTHETIC |
| expA_d6_900k_s123 | A | D6 | 900000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expA_d6_900k_s999 | A | D6 | 900000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d0_5k_s42 | B | D0 | 5000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | bd6610a024138a23 | NOT_SAVED_TO_DISK | ebf1307f7c6da3e5 | 30 | COMPLETED_SYNTHETIC |
| expB_d0_5k_s123 | B | D0 | 5000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d0_5k_s999 | B | D0 | 5000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d1_25k_s42 | B | D1 | 25000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d1_25k_s123 | B | D1 | 25000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d1_25k_s999 | B | D1 | 25000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d2_50k_s42 | B | D2 | 50000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d2_50k_s123 | B | D2 | 50000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d2_50k_s999 | B | D2 | 50000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d3_100k_s42 | B | D3 | 100000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | 58ec0e0395c63b80 | NOT_SAVED_TO_DISK | d11f600cded18350 | 30 | COMPLETED_SYNTHETIC |
| expB_d3_100k_s123 | B | D3 | 100000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d3_100k_s999 | B | D3 | 100000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d4_250k_s42 | B | D4 | 250000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d4_250k_s123 | B | D4 | 250000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d4_250k_s999 | B | D4 | 250000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d5_500k_s42 | B | D5 | 500000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d5_500k_s123 | B | D5 | 500000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d5_500k_s999 | B | D5 | 500000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d6_900k_s42 | B | D6 | 900000 | 42 | INLINE_BASE_YAML | 29292ba3d0f64eab | 3dd930dd8a4da7b8 | NOT_SAVED_TO_DISK | c530e697c4c6cb04 | 30 | COMPLETED_SYNTHETIC |
| expB_d6_900k_s123 | B | D6 | 900000 | 123 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |
| expB_d6_900k_s999 | B | D6 | 900000 | 999 | INLINE_BASE_YAML | 29292ba3d0f64eab | MISSING | MISSING | MISSING | NOT_RUN | MISSING |

---

## 3. AUDIT 2 — 42-Run Matrix Completeness & Execution Inventory

The Master Protocol defined a matrix of **42 runs**:
2 Experiments x 7 Tiers (D0–D6) x 3 Seeds (42, 123, 999) = 42 Runs

### 3.1 Execution Status Breakdown
* **Completed on Real LHC Dataset:** **0 / 42 (0.0%)**
* **Executed in Synthetic Smoke-Test Mode:** **6 / 42 (14.3%)**
* **Completely Missing / Unexecuted:** **36 / 42 (85.7%)**

### 3.2 Matrix Inventory Table
| Run ID | Experiment | Tier | Target Events | Seed | Config Exists | Training Executed | Training Completed | Target Updates | Actual Updates | Checkpoint Exists | Evaluation Executed | Metrics File Exists | Included in Report | Audit Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| expA_d0_5k_s42 | A | D0 | 5000 | 42 | YES | YES (SYNTHETIC) | YES (SYNTHETIC) | 3120 | 36 | YES | YES (500 EV MOCK) | YES | YES (DUPLICATED / SYNTHETIC) | COMPLETED_SYNTHETIC |
| expA_d0_5k_s123 | A | D0 | 5000 | 123 | YES | NO | NO | 3120 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d0_5k_s999 | A | D0 | 5000 | 999 | YES | NO | NO | 3120 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d1_25k_s42 | A | D1 | 25000 | 42 | YES | NO | NO | 15620 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d1_25k_s123 | A | D1 | 25000 | 123 | YES | NO | NO | 15620 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d1_25k_s999 | A | D1 | 25000 | 999 | YES | NO | NO | 15620 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d2_50k_s42 | A | D2 | 50000 | 42 | YES | NO | NO | 31240 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d2_50k_s123 | A | D2 | 50000 | 123 | YES | NO | NO | 31240 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d2_50k_s999 | A | D2 | 50000 | 999 | YES | NO | NO | 31240 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d3_100k_s42 | A | D3 | 100000 | 42 | YES | YES (SYNTHETIC) | YES (SYNTHETIC) | 62500 | 36 | YES | YES (500 EV MOCK) | YES | YES (DUPLICATED / SYNTHETIC) | COMPLETED_SYNTHETIC |
| expA_d3_100k_s123 | A | D3 | 100000 | 123 | YES | NO | NO | 62500 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d3_100k_s999 | A | D3 | 100000 | 999 | YES | NO | NO | 62500 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d4_250k_s42 | A | D4 | 250000 | 42 | YES | NO | NO | 156240 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d4_250k_s123 | A | D4 | 250000 | 123 | YES | NO | NO | 156240 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d4_250k_s999 | A | D4 | 250000 | 999 | YES | NO | NO | 156240 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d5_500k_s42 | A | D5 | 500000 | 42 | YES | NO | NO | 312500 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d5_500k_s123 | A | D5 | 500000 | 123 | YES | NO | NO | 312500 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d5_500k_s999 | A | D5 | 500000 | 999 | YES | NO | NO | 312500 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d6_900k_s42 | A | D6 | 900000 | 42 | YES | YES (SYNTHETIC) | YES (SYNTHETIC) | 562500 | 36 | YES | YES (500 EV MOCK) | YES | YES (DUPLICATED / SYNTHETIC) | COMPLETED_SYNTHETIC |
| expA_d6_900k_s123 | A | D6 | 900000 | 123 | YES | NO | NO | 562500 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expA_d6_900k_s999 | A | D6 | 900000 | 999 | YES | NO | NO | 562500 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d0_5k_s42 | B | D0 | 5000 | 42 | YES | YES (SYNTHETIC) | YES (SYNTHETIC) | 10000 | 30 | YES | YES (500 EV MOCK) | YES | YES (DUPLICATED / SYNTHETIC) | COMPLETED_SYNTHETIC |
| expB_d0_5k_s123 | B | D0 | 5000 | 123 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d0_5k_s999 | B | D0 | 5000 | 999 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d1_25k_s42 | B | D1 | 25000 | 42 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d1_25k_s123 | B | D1 | 25000 | 123 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d1_25k_s999 | B | D1 | 25000 | 999 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d2_50k_s42 | B | D2 | 50000 | 42 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d2_50k_s123 | B | D2 | 50000 | 123 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d2_50k_s999 | B | D2 | 50000 | 999 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d3_100k_s42 | B | D3 | 100000 | 42 | YES | YES (SYNTHETIC) | YES (SYNTHETIC) | 10000 | 30 | YES | YES (500 EV MOCK) | YES | YES (DUPLICATED / SYNTHETIC) | COMPLETED_SYNTHETIC |
| expB_d3_100k_s123 | B | D3 | 100000 | 123 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d3_100k_s999 | B | D3 | 100000 | 999 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d4_250k_s42 | B | D4 | 250000 | 42 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d4_250k_s123 | B | D4 | 250000 | 123 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d4_250k_s999 | B | D4 | 250000 | 999 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d5_500k_s42 | B | D5 | 500000 | 42 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d5_500k_s123 | B | D5 | 500000 | 123 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d5_500k_s999 | B | D5 | 500000 | 999 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d6_900k_s42 | B | D6 | 900000 | 42 | YES | YES (SYNTHETIC) | YES (SYNTHETIC) | 10000 | 30 | YES | YES (500 EV MOCK) | YES | YES (DUPLICATED / SYNTHETIC) | COMPLETED_SYNTHETIC |
| expB_d6_900k_s123 | B | D6 | 900000 | 123 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |
| expB_d6_900k_s999 | B | D6 | 900000 | 999 | YES | NO | NO | 10000 | NOT_RUN | NO | NO | NO | YES (FABRICATED / COPIED) | MISSING |

**Audit Conclusion on Matrix Completeness:** The master comparison tables published in [`FINAL_SCALING_REPORT.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/FINAL_SCALING_REPORT.md) and [`CONSOLIDATED_SCALING_REPORT.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/CONSOLIDATED_SCALING_REPORT.md) populated entries for D1, D2, D4, D5 across both experiments by duplicating synthetic pilot numbers. These rows must be completely retracted.

---

## 4. AUDIT 3 — Dataset Partition Integrity vs Execution Ingestion

### 4.1 Verification of `data/splits/` Index Arrays
The partition files generated by `src/data/scaling_split.py` were audited:
- **Strict Prefix Hierarchy:**
  $$D_0 \subset D_1 \subset D_2 \subset D_3 \subset D_4 \subset D_5 \subset D_6$$
  verified via exact element-wise array matching (`np.array_equal`).
- **Partition Disjointness:** Zero overlap between Training Pool ($D_6$), Validation (50k), and Test (50k) arrays.
- **Zero Signal Leakage:** Background index arrays contain zero indices from signal files.
- **Cryptographic Hashes:** All binary `.npy` files match SHA-256 digests in `data/splits/dataset_manifest.json`.

### 4.2 Ingestion Failure
While `data/splits/` is mathematically sound, it was **never consumed** during pilot execution because `experiments/scaling/experiment_a_fixed_epochs.py:115` routed execution to mock synthetic arrays.

### 4.3 Workstation Filesystem Root Cause
On the local macOS host in `/Users/koul/Library/Mobile Documents/com~apple~CloudDocs/anomaly-detector-gnn/data/`, the 2.7 GB HDF5 file `events_anomalydetection.h5` and symlink `events_LHCO2020_backgroundMC_Pythia.h5` are evicted dataless APFS files managed by iCloud Drive. Python `tables.open_file()` calls block waiting for network daemons. Local execution relied on `--synthetic` to avoid hanging, but synthetic smoke outputs were subsequently misreported as real physics.

---

## 5. AUDIT 4 — Test Sample Size Discrepancy & 500-Event Mock Substitution

### 5.1 The Contradiction
* **Narrative Claim:** Models were evaluated on an immutable held-out test set of **50,000 background events** and **25,000 signal events**.
* **Empirical Data:** The mass-sculpting tables published:
  - Top 10% Cut: **50 / 500 surviving events**
  - Top 5% Cut: **25 / 500 surviving events**
  - Top 1% Cut: **5 / 500 surviving events**

### 5.2 Code Trace: `evaluation/evaluate_scaling_study.py:323-324`
```python
    if args.synthetic:
        n_bg = min(args.limit_bg, 500)
        n_sig = min(args.limit_sig, 250)
```
In `--synthetic` mode, `n_bg` was capped at 500, and `n_sig` was capped at 250.
Evaluating 500 background events at 10%, 5%, and 1% selection thresholds produced exactly 50, 25, and 5 surviving events. These mock counts were published directly into reports alongside narrative claims of a 50,000-event evaluation.

---

## 6. AUDIT 5 — ROC-AUC Orientation & Synthetic Noise Analysis

### 6.1 Score & Label Directionality
- Background Label = `0`, Signal Label = `1`.
- Anomaly Score = Constituent reconstruction MSE. Higher error $\implies$ anomalous / signal.
- The score orientation in `evaluation/evaluate_scaling_study.py` is correct.

### 6.2 Cause of Matched AUC = 0.4856
In `evaluation/evaluate_scaling_study.py:325-326`:
```python
    bg_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_bg)]
    sig_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_sig)]
```
Both background and signal constituent arrays were drawn from the **identical uniform random generator** `np.random.uniform(-3, 3)`. The autoencoder reconstructed both classes with identical error distributions. Computing ROC-AUC on identical distributions yields the expected null value of $0.50 \pm \sigma_{\text{stat}}$. On Seed 42, the realization was **0.48555** (reported as **0.4856**).

The narrative in [`FINAL_SCALING_REPORT.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/FINAL_SCALING_REPORT.md) that this proved "PointNet–GAT capacity saturation on QCD dijets" is an invalid rationalization of random noise.

---

## 7. AUDIT 6 — Mass Sculpting Statistical Validity & Guard-Clause Fallback

### 7.1 Guard-Clause Fallback in Top 1% JSD
In `evaluation/evaluate_scaling_study.py:106-107`:
```python
    if len(p_samples) < 10 or len(q_samples) < 10:
        return 0.0, 9999.0
```
For the Top 1% cut on 500 events, $N_{\text{cut}} = 5 < 10$. The function returned `(0.0, 9999.0)`.
The reports cited $\text{JSD} = 0.0000$ and $1/\text{JSD} = 9999.0$ as evidence of near-perfect mass invariance, when it was simply a default return for an undersized array.

### 7.2 Bin Sparsity in 40-Bin Histograms
For Top 10% ($N=50$) and Top 5% ($N=25$) cuts across 40 bins:
- Over 65% of bins have zero counts ($N_i = 0$).
- Divergence values are dominated by epsilon clipping (`np.clip(p, 1e-12, 1.0)`).

### 7.3 Invalidation of the RSD Grooming Claim
The claim that Recursive Soft Drop grooming reduced mass sculpting was derived from `bg_rsd_m = bg_raw_m * 0.98`. Multiplying uniform noise by 0.98 shifted bin alignments; it had no connection to physical jet grooming.

---

## 8. AUDIT 7 & 8 — Statistical Confounding & Checkpoint Audits

1. **Uncertainty Conflation:** The $\pm$ values (e.g., $0.4856 \pm 0.0349$) represent the **bootstrap sample standard deviation on a single mock evaluation**, not multi-seed model variance.
2. **Multi-Seed Variance Undefined:** Because seeds 123 and 999 were never executed, sample standard deviations across training seeds cannot be computed ($N=1, N-1=0$).
3. **Power-Law Modeling Degeneracy:** Fitting $M(N) = M_\infty - a N^{-\alpha}$ across identical copied points resulted in a degenerate fit with zero slope ($\alpha = 0$).

---

## 9. Comprehensive Blocker & Defect Register

| Blocker ID | Severity | File & Function | Defect Summary | Scientific Impact | Remediation Pathway |
|:---:|:---:|---|---|---|---|
| **BLK-01** | **CRITICAL** | `experiment_a_fixed_epochs.py:115` | Synthetic substitution ignored dataset size; trained on 600 uniform floats | Data scaling was never tested on collision data | Run training without `synthetic_mode` on Kaggle GPU |
| **BLK-02** | **CRITICAL** | `FINAL_SCALING_REPORT.md:Table 1` | 36 of 42 matrix runs missing; table entries populated by duplication | Fabricated completion of unrun matrix tiers | Retract unrun rows; execute all 42 runs on Kaggle GPU |
| **BLK-03** | **CRITICAL** | `evaluate_scaling_study.py:323` | Capped test sample to 500 background and 250 signal uniform random floats | AUC = 0.4856 is noise vs noise, not physics | Evaluate on real held-out 50k bg / 25k sig test set |
| **BLK-04** | **CRITICAL** | `evaluate_scaling_study.py:106` | Guard clause returned `(0.0, 9999.0)` when N=5 < 10 | Top 1% JSD = 0.0 is an unhandled code fallback | Evaluate on 50,000-event test set (N_cut = 500) |
| **BLK-05** | **MAJOR** | Local `data/` filesystem | APFS dataless iCloud file eviction blocks local HDF5 reading | Cannot train or evaluate on real datasets locally | Execute all production jobs on Kaggle GPU ext4 disk |
| **BLK-06** | **MAJOR** | `STATISTICAL_ANALYSIS.md` | Reported test bootstrap standard deviation as multi-seed training variance | Fails to measure true inter-seed model stability | Label $\pm$ as bootstrap CI; compute seed std after 3 seeds |

---

## 10. Verified Codebase Assets vs Required Cloud GPU Execution

### What is Verified & Production-Ready:
1. **Model Architecture:** [`src/run_lhco_gnn.py:PointNetGATHybrid`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/run_lhco_gnn.py#L580) is verified at exactly **7,957 parameters**.
2. **Dataset Partitioning:** [`src/data/scaling_split.py`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/data/scaling_split.py) correctly generates deterministic, disjoint, prefix-nested index files with valid SHA-256 manifests.
3. **Automated Unit Tests:** All **72 unit tests** in `tests/` pass cleanly in 2.3 seconds.
4. **Execution Harnesses:** [`scripts/run_scaling_matrix.py`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/scripts/run_scaling_matrix.py) and [`notebooks/phase4_scaling.ipynb`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/notebooks/phase4_scaling.ipynb) are verified and ready for cloud GPU execution.

### What is Blocked & Requires Rerunning:
- All empirical metrics and conclusions regarding background data scaling.
- Training and evaluation must be conducted on Kaggle GPU with local ext4 disk mounts.

---

## 11. Actionable Remediation Roadmap

```
                                  REMEDIATION ROADMAP
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     PHASE 1: RETRACT MOCK REPORTS                   PHASE 2: CLOUD GPU EXECUTION
     -----------------------------                   ----------------------------
     - Retract unrun rows in tables                  - Open notebooks/phase4_scaling.ipynb
     - Label pilot as synthetic smoke tests          - Attach LHC Olympics HDF5 dataset
     - Mark Gates 0-4 as PENDING EXECUTION           - Run scripts/run_scaling_matrix.py
                                                     - Train all 42 runs on Tesla T4/P100
                                                                  │
                                                                  ▼
                                                     PHASE 3: FROZEN TEST EVALUATION
                                                     -------------------------------
                                                     - Run evaluate_scaling_study without --synthetic
                                                     - Evaluate full 50,000 bg & 25,000 sig events
                                                     - Compute true JSD (500 events in Top 1%)
                                                     - Compute true multi-seed standard deviations
                                                     - Publish empirical scaling laws
```

### Actionable Commands for Kaggle GPU Execution:
1. Open [`notebooks/phase4_scaling.ipynb`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/notebooks/phase4_scaling.ipynb) on Kaggle with an NVIDIA Tesla T4 or P100 GPU.
2. Link the Kaggle dataset files:
   ```bash
   ln -sf /kaggle/input/.../events_LHCO2020_backgroundMC_Pythia.h5 data/events_LHCO2020_backgroundMC_Pythia.h5
   ln -sf /kaggle/input/.../W_prime_signal.h5 data/W_prime_signal.h5
   ```
3. Run the full matrix runner across all tiers and seeds:
   ```bash
   python scripts/run_scaling_matrix.py --experiment both --tier all --seed all
   ```
4. Package results for download:
   ```bash
   tar -czvf /kaggle/working/scaling_results_bundle.tar.gz results/scaling/
   ```

---

## 12. Final Certification Statement

* **Audit Verdict:** **C. SCIENTIFICALLY BLOCKED**
* **Certification:** This audit certifies that the PointNet–GAT scaling pipeline software is fully prepared, mathematically verified, and test-covered. However, all published scaling conclusions must remain blocked until the 42-run matrix is executed on the real LHC Olympics dataset via Kaggle GPU.
