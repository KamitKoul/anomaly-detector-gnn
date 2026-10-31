# Statistical Analysis & Scaling Modeling: Background-Data Study

**Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Subject:** Statistical Modeling, Uncertainty Separation & Scaling Laws  
**Date:** October 31, 2026  
**Status:** Protocol Frozen & Pilot Audited  

---

## 1. Disentangling Seed Variability and Finite-Sample Uncertainty

A central methodological requirement of Phase 8 is the strict separation of two distinct sources of experimental uncertainty:

```
                            SOURCES OF EXPERIMENTAL UNCERTAINTY
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
   TRAINING-SEED VARIABILITY                                       FINITE-TEST UNCERTAINTY
   -------------------------                                       -----------------------
   Origin: Initial weights & batch order                           Origin: Finite test sample size
   Evaluated over: Seeds (42, 123, 999)                            Evaluated over: B=1,000 bootstrap replicas
   Captures: Model training landscape variance                     Captures: Test estimator variance
   Reported as: Sample standard deviation (N-1)                    Reported as: 68.3% Central CI [16th, 84th]
```

### Critical Scientific Rules:
1. **Never conflate bootstrap resamples with training runs:** A single model checkpoint evaluated over 1,000 bootstrap draws does not measure model training stability; it only measures the precision of the test set estimator.
2. **Standard Deviation Conventions:** All multi-seed standard deviations use Bessel's correction for sample standard deviation:
   $$s = \sqrt{\frac{1}{N - 1} \sum_{i=1}^{N} (x_i - \bar{x})^2}$$
3. **Bootstrap Procedure:** Class-stratified bootstrap resampling is applied to the test predictions with $B = 1,000$ resamples, preserving the exact background and signal proportions. The 68.3% confidence interval is extracted non-parametrically from the 15.85th and 84.15th empirical percentiles.

---

## 2. Power-Law Scaling Model Formulation

Following modern empirical scaling paradigms in deep learning, performance metrics $M(N)$ (e.g., validation reconstruction loss $\mathcal{L}$, matched ROC-AUC, or background rejection $R_{30}$) as a function of background dataset size $N$ are modeled using a three-parameter power law:

$$M(N) = M_\infty - a N^{-\alpha}$$

where:
- $M_\infty$: The theoretical asymptotic performance ceiling as $N \to \infty$.
- $a$: The scale prefactor governing the penalty incurred by finite dataset size.
- $\alpha$: The empirical power-law scaling exponent ($\alpha > 0$).
- $N$: Background training event count ($N \in \{5\text{k}, 25\text{k}, 50\text{k}, 100\text{k}, 250\text{k}, 500\text{k}, 900\text{k}\}$).

### Loss Formulation for Generalization Error:
For validation reconstruction loss, which decreases with scale:
$$\mathcal{L}(N) = \mathcal{L}_\infty + a N^{-\alpha}$$
In log-log coordinates after subtracting the asymptote $\mathcal{L}_\infty$:
$$\ln\left(\mathcal{L}(N) - \mathcal{L}_\infty\right) = \ln(a) - \alpha \ln(N)$$

### Scientific Constraints on Scaling Claims:
1. **Degrees of Freedom Caution:** With exactly 7 discrete dataset tiers ($D_0$ through $D_6$), fitting a 3-parameter non-linear curve leaves only 4 degrees of freedom. Therefore, power-law fits are reported as exploratory empirical models with full fit residuals rather than fundamental physical laws.
2. **The Capacity Saturation Hypothesis:** For a fixed, compact architecture (7,957 parameters), if the scaling exponent $\alpha \approx 0$ or if residuals deviate systematically, the data indicates that **model capacity is the binding constraint**, not training dataset volume.

---

## 3. Pilot Benchmark Analysis & Baseline Variance

### 3.1 Empirical Test Uncertainty from Pilot Gate Checkpoints (Seed 42)
Across 1,000 class-stratified bootstrap replicas on the held-out test dataset:

| Model Run | Metric | Estimator Mean | 68.3% CI Low | 68.3% CI High | Bootstrap $\sigma_{\text{stat}}$ |
|---|---|:---:|:---:|:---:|:---:|
| **`expA_d0_5k_s42`** | Matched ROC-AUC | 0.4856 | 0.4491 | 0.5185 | 0.0349 |
| | Inclusive ROC-AUC | 0.4642 | 0.4431 | 0.4866 | 0.0226 |
| | Narrow ROC-AUC | 0.4327 | 0.3598 | 0.5031 | 0.0738 |
| | Rejection at 30% Eff | 3.19 | 2.82 | 3.65 | 0.41 |
| **`expB_d0_5k_s42`** | Matched ROC-AUC | 0.4593 | 0.4224 | 0.4962 | 0.0369 |
| | Inclusive ROC-AUC | 0.4526 | 0.4312 | 0.4745 | 0.0223 |
| | Narrow ROC-AUC | 0.4286 | 0.3541 | 0.4988 | 0.0722 |
| | Rejection at 30% Eff | 2.50 | 2.15 | 2.92 | 0.38 |

### 3.2 Key Statistical Observations:
1. **Window-Dependent Test Uncertainty:**
   In the narrow resonance window ($m_{jj} \in [3.45, 3.55]$ TeV), the statistical standard deviation is $\sigma_{\text{stat}} \approx 0.074$, more than **3.2 times larger** than the inclusive uncertainty ($\sigma_{\text{stat}} \approx 0.023$). Any comparison across dataset sizes within a narrow window must account for this wider confidence interval before asserting a scaling trend.
2. **Statistical Distinguishability:**
   The difference in matched AUC between Experiment A ($0.4856 \pm 0.0349$) and Experiment B ($0.4593 \pm 0.0369$) is $\Delta \text{AUC} = 0.0263$, which lies within $1\sigma$ combined statistical uncertainty ($\sqrt{0.0349^2 + 0.0369^2} = 0.0508$). Thus, under short pilot optimization, the two configurations are statistically indistinguishable.
