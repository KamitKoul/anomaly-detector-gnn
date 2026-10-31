# Mass Sculpting & Mass Decorrelation Scientific Validation

**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Evaluation Scope:** 50,000 Real Held-Out QCD Background Test Events (`bg_test_50k.npy`)  
**Mass Observable:** Reconstructed Dijet Invariant Mass $m_{jj} \in [2000, 5000]$ GeV  
**Binning Protocol:** Uniform 40 bins ($75$ GeV/bin width)  
**Date:** 2026-10-31  

---

## 1. Remediation of Mass-Sculpting Fallback Defects

In historical scaling monographs, the PointNet–GAT Hybrid was claimed to exhibit "near-perfect mass invariance" at tight anomaly thresholds:
- Reported Top 1% Base-2 JSD: $0.000000$
- Reported Top 1% Inverse JSD ($1/\text{JSD}$): $9999.0$

### Forensic Root Cause
The independent audit uncovered line 52 of `evaluation/evaluate_scaling_study.py`:
```python
if len(p_samples) < 10 or len(q_samples) < 10:
    return 0.0, 9999.0
```
In synthetic evaluation mode, the test set was capped to $N=500$ events. At the Top 1% selection threshold:
$$N_{\text{selected}} = 0.01 \times 500 = 5 \text{ events}$$
Since $5 < 10$, the conditional branch triggered unconditionally, returning `(0.0, 9999.0)`. A numerical safeguard designed to prevent zero-division was converted into a published scientific claim of ideal mass invariance.

### Remediated Mathematical Implementation
The guard clause fallback has been completely removed and replaced with a rigorous statistical status protocol:
```python
def compute_base2_jsd(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int = 40,
    range_gev: Tuple[float, float] = (2000.0, 5000.0),
) -> Tuple[float, float, str, int, int]:
    if len(p_samples) < 10 or len(q_samples) < 10:
        return float("nan"), float("nan"), "INSUFFICIENT_STATISTICS", 0, 0
    # ...
```
- If sample size $N < 10$: Returns `NaN` and explicit status `"INSUFFICIENT_STATISTICS"`.
- Under the full real test partition ($N_{\text{bg}} = 50,000$):
  - Top 10%: $N = 5,000$ events ($\gg 10$) $\to$ Statistically valid.
  - Top 5%: $N = 2,500$ events ($\gg 10$) $\to$ Statistically valid.
  - Top 1%: $N = 500$ events ($\gg 10$) $\to$ Statistically valid.
- Epsilon clipping ($10^{-12}$) and base-2 Jensen-Shannon divergence:
  $$\text{JSD}_2(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$

---

## 2. Invalidation of Mock Recursive Soft Drop (RSD) Scaling

Historical synthetic scripts simulated groomed jet mass through a constant multiplier:
```python
bg_rsd_m = bg_raw_m * 0.98  # Constant mock scaling
```
Multiplying a continuous distribution by a scalar shifts the mean without altering the relative shape or testing actual collinear/soft gluon grooming. 

### Correct Physical Grooming Protocol
1. In `src/physics_grooming.py`, Recursive Soft Drop is implemented using FastJet clustering ($R=1.0$, anti-$k_t$) followed by Cambridge-Aachen re-clustering with grooming parameters:
   $$z_{\text{cut}} = 0.05, \quad \beta = 1.0$$
2. On the full 50,000 background test events, genuine dijet clustering is executed.
3. If FastJet grooming is unavailable in a minimalist environment, RSD metrics are explicitly reported as `None` with status `"NOT_CONFIGURED"`, rather than using mathematical approximations.

---

## 3. Metrics Definition & Frozen Protocol

For every evaluated model checkpoint, mass sculpting is characterized by:

1. **Information-Theoretic Divergence:**
   - Base-2 Jensen-Shannon Divergence ($\text{JSD}_2 \in [0, 1]$)
   - Background Rejection Factor ($1 / \text{JSD}_2$)
   - Measured at Top 10%, Top 5%, and Top 1% anomaly cuts.
2. **Linear & Monotonic Correlations:**
   - Pearson correlation coefficient ($r$): Linear dependence between anomaly score and raw dijet mass.
   - Spearman rank correlation ($\rho$): Monotonic ranking dependence.
3. **Non-Linear Distance Correlation ($\text{dCor}^2$):**
   - Implemented via unbiased distance covariance over a standardized 1,000-event batch.
   - $\text{dCor}^2 = 0 \iff X$ and $Y$ are strictly independent.
4. **Histogram Diagnostics:**
   - Bin width: $75$ GeV ($[2000, 5000]$ GeV across 40 bins).
   - Empty bin monitoring: Counts empty bins in baseline and selected distributions.
