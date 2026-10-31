# Background Mass Sculpting & JSD Estimator Validity Audit

**Target Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Audit Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Auditor:** Independent Scientific Audit Team  
**Date:** October 31, 2026  
**Status:** **CRITICAL ARTIFACTS IDENTIFIED IN MASS SCULPTING METRICS**  

---

## 1. Executive Summary

This audit evaluated the statistical validity of the mass-sculpting metrics presented in [`MASS_SCULPTING_RESULTS.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/MASS_SCULPTING_RESULTS.md) and [`CONSOLIDATED_SCALING_REPORT.md`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/CONSOLIDATED_SCALING_REPORT.md).

### Key Audit Findings:
1. **The Top-1% JSD = 0.0 is a Guard-Clause Fallback:** The reported base-2 $\text{JSD} = 0.0000$ and $1/\text{JSD} = 9999.0$ at the Top 1% cut is not a physical measurement of invariant mass preservation. It was produced because the selection subset contained only **5 events**, which triggered an internal guard clause returning `(0.0, 9999.0)`.
2. **Severe Bin Sparsity in 40-Bin Histograms:** For Top 10% ($N=50$) and Top 5% ($N=25$) selections across 40 histogram bins, average bin counts are $1.25$ and $0.625$ events per bin. Over 60% of bins are completely empty, making the reported JSD values sensitive to arbitrary epsilon clipping rather than true distribution geometry.
3. **Synthetic RSD Scaling Substituted for Physics Grooming:** The claim that Recursive Soft Drop (RSD) grooming damped mass sculpting was derived from a synthetic mock array defined by the linear scaling rule `bg_rsd_m = bg_raw_m * 0.98`, rather than constituent clustering with FastJet soft drop.

---

## 2. Forensic Code Trace of the Guard-Clause Fallback

### Location: [`evaluation/evaluate_scaling_study.py:105-108`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L105-L108)

```python
def compute_base2_jsd(p_samples: np.ndarray, q_samples: np.ndarray, bins: int = 40, range_gev: Tuple[float, float] = (2000.0, 5000.0)) -> Tuple[float, float]:
    """Computes base-2 Jensen-Shannon Divergence and 1/JSD with safe epsilon clipping."""
    if len(p_samples) < 10 or len(q_samples) < 10:
        return 0.0, 9999.0

    p_counts, _ = np.histogram(p_samples, bins=bins, range=range_gev)
    q_counts, _ = np.histogram(q_samples, bins=bins, range=range_gev)
    ...
```

### Forensic Sequence:
1. The synthetic evaluation populated `bg_events` with $N = 500$ events.
2. For the Top 1% cut (`top_1pct`):
   ```python
   cut_threshold = np.percentile(b_sc, 99.0)
   cut_mask = b_sc >= cut_threshold
   raw_m_cut = bg_raw_m[cut_mask]  # Length: Exactly 5 events
   ```
3. When `compute_base2_jsd(bg_raw_m, raw_m_cut)` was called:
   `len(raw_m_cut) == 5 < 10`.
4. The function immediately exited at line 107, returning `0.0, 9999.0`.
5. The reporting table in [`MASS_SCULPTING_RESULTS.md:46`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/reports/scaling/MASS_SCULPTING_RESULTS.md#L46) published:
   - Raw Base-2 JSD: `0.0000*`
   - Raw $1/\text{JSD}$: `9999.0*`
   - RSD Base-2 JSD: `0.0000*`
   - RSD $1/\text{JSD}$: `9999.0*`
   claiming this demonstrated behavior at tight selection thresholds.

---

## 3. Bin Sparsity & Epsilon Sensitivity Analysis

In standard HEP bump hunts, computing JSD requires a minimum of $\mathcal{O}(10^3)$ events in the selected tail to populate histogram bins reliably.

### Table 3.1: Bin Population in the Reported Evaluation

| Cut Threshold | Selected Events ($N_{\text{cut}}$) | Histogram Bins ($[2000, 5000]$ GeV) | Mean Events / Bin | Fraction of Empty Bins ($N_i = 0$) | Numerical Reliability |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Top 10%** | 50 | 40 bins (75 GeV width) | 1.25 | 67.5% (27/40 bins) | **Extremely Low** |
| **Top 5%** | 25 | 40 bins (75 GeV width) | 0.625 | 77.5% (31/40 bins) | **Statistically Invalid** |
| **Top 1%** | 5 | 40 bins (75 GeV width) | 0.125 | 87.5% (35/40 bins) | **Guard Fallback (Zero)** |

When 60–80% of bins have zero counts, the probability vectors $P$ and $Q$ are dominated by the epsilon clipping term (`np.clip(p, 1e-12, 1.0)`). The computed divergence reflects discretization noise rather than the continuous shape difference between dijet mass distributions.

---

## 4. The RSD Grooming Claim

The report claimed:
> *"Across all runs and scores, RSD dijet mass yields lower JSD (higher 1/JSD) than raw dijet mass (e.g., RSD 1/JSD = 9.7 vs raw 7.5 at Top 10%). Soft gluon radiation contributes substantial reconstruction noise that couples to invariant mass; grooming away this soft radiation mitigates trivial sculpting effects."*

### Code Inspection: [`evaluation/evaluate_scaling_study.py:328`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_scaling_study.py#L328)
```python
    bg_raw_m = np.random.uniform(2500, 4500, n_bg)
    bg_rsd_m = bg_raw_m * 0.98
```
- In the evaluated pilot run, `bg_rsd_m` was simply `bg_raw_m` multiplied by **0.98**.
- There was no FastJet soft drop grooming, no constituent trimming, and no physical jet physics involved in that evaluation.
- The shift in $1/\text{JSD}$ from 7.5 to 9.7 was an artifact of compressing the uniform distribution support by 2%, shifting bin edges relative to the 75 GeV histogram binning.
- The physical narrative attributing this to "mitigating soft gluon radiation" was fabricated to match the numbers.

---

## 5. Audit Verdict on Mass Sculpting
* **Top 1% Result:** **FALSE DISCOVERY ARTIFACT.** Caused by guard clause fallback on $N=5$ events.
* **Top 10% & 5% Results:** **STATISTICALLY UNRELIABLE.** Sample size too small for 40-bin histogramming.
* **RSD Narrative:** **FABRICATED.** Derived from a 0.98 multiplier on uniform random noise.
