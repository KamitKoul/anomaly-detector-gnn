# Statistical Reporting & Recalculation Audit

**Target Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Audit Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Auditor:** Independent Scientific Audit Team  
**Date:** October 31, 2026  
**Status:** **AUDIT COMPLETE — STATISTICAL CONFOUNDING IDENTIFIED**  

---

## 1. Executive Summary

This audit evaluated the statistical claims presented in [`STATISTICAL_ANALYSIS.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/STATISTICAL_ANALYSIS.md) and [`FINAL_SCALING_REPORT.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/FINAL_SCALING_REPORT.md):
1. **Meaning of Reported Uncertainty ($\pm$):** The uncertainty intervals (e.g., $0.4856 \pm 0.0349$) represent the standard deviation of 500 non-parametric bootstrap resamples on a **single test evaluation**, NOT training-seed variability across independent models.
2. **Missing Multi-Seed Data:** Zero models were trained or evaluated on seeds `123` or `999`. Consequently, sample standard deviations across independent training runs ($s = \sqrt{\frac{1}{N-1}\sum (x_i - \bar{x})^2}$) could **not be computed** ($N=1$, degrees of freedom $N-1 = 0$).
3. **Power-Law Modeling Degeneracy:** Fitting $M(N) = M_\infty - a N^{-\alpha}$ to data where all 7 points are numerically identical is mathematically degenerate.

---

## 2. Forensic Audit of Uncertainty Representation

### 2.1 Code Trace in `evaluation/evaluate_scaling_study.py:59-64`:
```python
    boot_aucs = np.array(boot_aucs)
    mean_auc = float(np.mean(boot_aucs))
    ci_low = float(np.percentile(boot_aucs, 15.865))
    ci_high = float(np.percentile(boot_aucs, 84.135))
    std_auc = float(np.std(boot_aucs, ddof=1))
```

### Forensic Reality:
* The ± value represents the **bootstrap sample standard deviation** of the ROC-AUC estimator on the single 500-event synthetic evaluation.
* In [`CONSOLIDATED_SCALING_REPORT.md:Table 10`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/CONSOLIDATED_SCALING_REPORT.md), every tier ($D_0$ through $D_6$) reported:
  $$\text{Matched ROC-AUC} = 0.4856 \pm 0.035$$
* The text claimed to disentangle training seed variance from finite-test uncertainty, but presented only the test-sample bootstrap standard deviation of a single mock run.

---

## 3. Recalculation Across Multi-Seed Runs

To determine whether empirical standard deviations across independent training seeds can be recomputed from artifacts:

| Run Key | Configured Seeds | Executed Seeds on Disk | Available Checkpoints | Computed Seed Mean | Computed Seed $s$ ($N-1$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Exp A — $D_0$** | 42, 123, 999 | 42 only | 1 | 0.4856 | **UNDEFINED ($N=1$)** |
| **Exp A — $D_1$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp A — $D_2$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp A — $D_3$** | 42, 123, 999 | 42 only | 1 | 0.4856 | **UNDEFINED ($N=1$)** |
| **Exp A — $D_4$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp A — $D_5$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp A — $D_6$** | 42, 123, 999 | 42 only | 1 | 0.4856 | **UNDEFINED ($N=1$)** |
| **Exp B — $D_0$** | 42, 123, 999 | 42 only | 1 | 0.4593 | **UNDEFINED ($N=1$)** |
| **Exp B — $D_1$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp B — $D_2$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp B — $D_3$** | 42, 123, 999 | 42 only | 1 | 0.4593 | **UNDEFINED ($N=1$)** |
| **Exp B — $D_4$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp B — $D_5$** | 42, 123, 999 | None | 0 | MISSING | MISSING |
| **Exp B — $D_6$** | 42, 123, 999 | 42 only | 1 | 0.4593 | **UNDEFINED ($N=1$)** |

**Conclusion:** It is impossible to calculate true inter-seed standard deviations from current repository artifacts because seeds 123 and 999 were never executed.

---

## 4. Power-Law Scaling Fit Audit

In [`STATISTICAL_ANALYSIS.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/STATISTICAL_ANALYSIS.md), an exploratory power-law curve was proposed:
$$M(N) = M_\infty - a N^{-\alpha}$$

* When fitted to the reported validation loss values:
  $$\mathcal{L}_{\text{val}}(D_0) = 2.5881, \quad \mathcal{L}_{\text{val}}(D_1) = 2.5881, \quad \dots \quad \mathcal{L}_{\text{val}}(D_6) = 2.5881$$
* Because the slope in log-space is strictly zero across all points:
  $$\frac{d \ln(\mathcal{L})}{d \ln(N)} = 0 \implies \alpha = 0, \quad a = 0, \quad M_\infty = 2.5881$$
* This does not reflect a physical scaling law; it is the trivial mathematical consequence of fitting a curve to identical copied points.

---

## 5. Audit Verdict on Statistical Reporting
* **Status:** **CRITICAL MISATTRIBUTION.**
* **Remediation:** All tables must explicitly label $\pm$ as bootstrap finite-sample uncertainty for Seed 42, and all multi-seed claims must be retracted until seeds 123 and 999 are trained.
