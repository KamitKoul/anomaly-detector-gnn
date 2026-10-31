# Evaluation Sample Size & Test Partition Audit

**Target Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Audit Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Auditor:** Independent Scientific Audit Team  
**Date:** October 31, 2026  
**Status:** **CRITICAL METHODOLOGICAL CONTRADICTION IDENTIFIED**  

---

## 1. Executive Summary

A critical discrepancy exists between the high-level narrative and the empirical numbers in the evaluation reports:
* **The Narrative Claim:** All reports ([`EXPERIMENT_PROTOCOL.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/EXPERIMENT_PROTOCOL.md), [`EVALUATION_RESULTS.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/EVALUATION_RESULTS.md), [`CONSOLIDATED_SCALING_REPORT.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/CONSOLIDATED_SCALING_REPORT.md)) claim that models were evaluated against an immutable held-out test set of **50,000 background events** (`bg_test_50k.npy`) and **25,000 signal events** (`sig_test_25k.npy`).
* **The Empirical Data:** The mass-sculpting tables across all reports explicitly document:
  - Top 10% Cut: **50 / 500 surviving events**
  - Top 5% Cut: **25 / 500 surviving events**
  - Top 1% Cut: **5 / 500 surviving events**

This audit traces the code implementation to explain how a 500-event sample was substituted for the intended 50,000-event test set and documents the scientific consequences.

---

## 2. Code Trace & Root Cause Identification

### Location: [`evaluation/evaluate_scaling_study.py:321-331`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L321-L331)

```python
    if args.synthetic:
        np.random.seed(42)
        n_bg = min(args.limit_bg, 500)
        n_sig = min(args.limit_sig, 250)
        bg_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_bg)]
        sig_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_sig)]
        bg_raw_m = np.random.uniform(2500, 4500, n_bg)
        bg_rsd_m = bg_raw_m * 0.98
        sig_raw_m = np.random.normal(3500, 80, n_sig)
        sig_rsd_m = sig_raw_m * 0.99
    else:
        # Load real test partition indices
        splits_dir = REPO_ROOT / "data" / "splits"
        bg_test_idx = np.load(splits_dir / "bg_test_50k.npy")[:args.limit_bg]
        sig_test_idx = np.load(splits_dir / "sig_test_25k.npy")[:args.limit_sig]
        ...
```

### Forensic Sequence of Events:
1. When `scripts/run_scaling_pilot.py` invoked the evaluation harness:
   ```python
   os.system(f".venv/bin/python evaluation/evaluate_scaling_study.py --checkpoint '{ckpt_a}' --output-json '{eval_json_a}' {'--synthetic' if synthetic_mode else ''}")
   ```
   Because `synthetic_mode` was `True`, the `--synthetic` flag was passed to the evaluation CLI.
2. In synthetic mode, `n_bg` was capped at `min(50000, 500) = 500`, and `n_sig` was capped at `min(25000, 250) = 250`.
3. The evaluation scored exactly **500 synthetic background events** and **250 synthetic signal events**.
4. In mass sculpting:
   - Top 10% threshold: $500 \times 0.10 = \mathbf{50\text{ events}}$.
   - Top 5% threshold: $500 \times 0.05 = \mathbf{25\text{ events}}$.
   - Top 1% threshold: $500 \times 0.01 = \mathbf{5\text{ events}}$.
5. The outputs were saved to `results/scaling/pilot/experiment_a/expA_d0_5k_s42/eval_metrics.json`.
6. The reporting scripts extracted these exact counts and placed them into the research reports without identifying that they represented a 500-event synthetic mock test rather than the real 50,000-event held-out test partition.

---

## 3. Discrepancy Breakdown by Metric

| Reported Metric | Narrative Claim | Actual Evaluation Reality | Discrepancy Factor |
|---|---|---|:---:|
| **Background Sample Size ($N_{\text{bg}}$)** | 50,000 Pythia QCD Events | 500 Synthetic Uniform Floats | **$100\times$ Under-sampled** |
| **Signal Sample Size ($N_{\text{sig}}$)** | 25,000 $W' \to XY$ Events | 250 Synthetic Uniform Floats | **$100\times$ Under-sampled** |
| **Top 10% Surviving Background** | Claimed $\sim 5,000$ Events | Exactly 50 Events | $100\times$ Deficit |
| **Top 5% Surviving Background** | Claimed $\sim 2,500$ Events | Exactly 25 Events | $100\times$ Deficit |
| **Top 1% Surviving Background** | Claimed $\sim 500$ Events | Exactly 5 Events | $100\times$ Deficit |
| **Matched Mass Window ($[3.3, 3.7]$)** | Clustered FastJet Dijets | Synthetic Uniform $[2500, 4500]$ | Unphysical distribution |

---

## 4. Can Metrics Be Recalculated Locally on Held-Out Test Data?

* **Audit Test:** Attempted to execute `evaluation/evaluate_scaling_study.py` without the `--synthetic` flag to evaluate the checkpoints on real held-out data.
* **Finding:** Execution hangs indefinitely on `tables.open_file("data/events_LHCO2020_backgroundMC_Pythia.h5")` and `tables.open_file("data/events_anomalydetection.h5")`.
* **Reason:** On the local macOS workstation in `~/Library/Mobile Documents/com~apple~CloudDocs/`, the large HDF5 files are offloaded, dataless APFS files managed by iCloud Drive. Python file descriptor opens block waiting for iCloud system daemons.
* **Remediation Requirement:** Full-scale evaluation across the true 50,000 background and 25,000 signal test partitions cannot be completed locally under current APFS eviction conditions. It must be executed in a dedicated Linux environment with local ext4 disk storage (specifically, on Kaggle GPU via [`notebooks/phase4_scaling.ipynb`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/notebooks/phase4_scaling.ipynb)).

---

## 5. Audit Verdict on Sample Size
* **Severity:** **CRITICAL (Severity 1)**
* **Impact:** All reported evaluation metrics (ROC-AUCs, rejection factors, and JSD values) reflect a 500-event synthetic random noise test, not the LHC Olympics collision dataset.
* **Status:** Must be prominently retracted in the audit register and superseded by real cloud GPU evaluations.
