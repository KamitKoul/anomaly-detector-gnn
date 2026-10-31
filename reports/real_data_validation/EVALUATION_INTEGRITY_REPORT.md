# Scientific Evaluation Integrity Report

**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Evaluation Scope:** 50,000 Held-Out Real Background Events (`bg_test_50k.npy`) and 25,000 Held-Out Real Signal Events (`sig_test_25k.npy`)  
**Evaluation Framework:** `evaluation/evaluate_scaling_study.py`  
**Date:** 2026-10-31  

---

## 1. Frozen Evaluation Protocol

To ensure reproducible, un-compromised scientific benchmarks:
1. **Sample Size Non-Negotiable Contract:**
   - Background test events: exactly 50,000 events (`data/splits/bg_test_50k.npy`).
   - Signal test events: exactly 25,000 events (`data/splits/sig_test_25k.npy`).
   - Historical silent capping (`min(args.limit_bg, 500)`) has been eliminated.
2. **Zero Signal Contamination:**
   - Signal events are strictly isolated in `data/splits/sig_*.npy`.
   - Signal data is never loaded during training or unsupervised validation checkpoint selection.
3. **Score Orientation Verification:**
   - Higher reconstruction error indicates greater anomaly divergence:
     $$S(x) = \frac{1}{|V|} \sum_{i \in V} \|x_i - \hat{x}_i\|^2$$
   - Signal events are labeled $y=1$; background events are labeled $y=0$.
   - ROC curves and AUCs are calculated with standard positive score orientation.

---

## 2. Standardized Anomaly Score Definitions

Four standardized anomaly scores are evaluated simultaneously for every checkpoint:

1. **Full Kinematic Reconstruction Error ($S_{\text{full}}$):**
   $$S_{\text{full}} = \frac{1}{3} \left[ (p_T - \hat{p}_T)^2 + (\eta - \hat{\eta})^2 + \Delta\phi^2 \right]$$
2. **$p_T$-Only Reconstruction Error ($S_{pT}$):**
   $$S_{pT} = (p_T - \hat{p}_T)^2$$
3. **Angular-Only Reconstruction Error ($S_{\text{ang}}$):**
   $$S_{\text{ang}} = \frac{1}{2} \left[ (\eta - \hat{\eta})^2 + \Delta\phi^2 \right]$$
   where periodic boundary distance is strictly enforced:
   $$\Delta\phi = \min(|\phi - \hat{\phi}|, 2\pi - |\phi - \hat{\phi}|)$$
4. **Weighted Combined Reconstruction Error ($S_{\text{weighted}}$):**
   $$S_{\text{weighted}} = 0.8 \, S_{pT} + 0.2 \, S_{\text{ang}}$$

---

## 3. Real Collision Physics Benchmark (D0 Baseline Model)

Evaluated on the newly trained real-data Experiment A Tier D0 model state (`expA_d0_5k_s42_checkpoint_best.pt`):

| Anomaly Score | Inclusive AUC | Matched-Mass AUC [3.3, 3.7] TeV | Narrow-Window AUC [3.45, 3.55] TeV | Rejection @ 30% Eff | Rejection @ 50% Eff | Max SIC |
|---|---|---|---|---|---|---|
| **$S_{\text{full}}$** | $0.4338 \pm 0.0022$ | $0.4333 \pm 0.0040$ | $0.4348 \pm 0.0074$ | 2.73 | 1.67 | 1.000 |
| **$S_{pT}$** | **$0.5691 \pm 0.0022$** | **$0.5606 \pm 0.0044$** | **$0.5623 \pm 0.0084$** | **4.43** | **2.35** | **1.000** |
| **$S_{\text{ang}}$** | $0.4239 \pm 0.0022$ | $0.4245 \pm 0.0040$ | $0.4258 \pm 0.0073$ | 2.67 | 1.62 | 1.000 |
| **$S_{\text{weighted}}$** | $0.4857 \pm 0.0022$ | $0.4811 \pm 0.0043$ | $0.4825 \pm 0.0080$ | 3.21 | 1.91 | 1.000 |

### High-Energy Physics Interpretation
1. **$p_T$ Reconstruction is Discriminative:** The $p_T$ autoencoder score correctly discriminates heavy 2-prong $W' \to XY$ signal decay products from QCD dijet continuum ($0.561$ matched AUC), rejecting background by $4.43\times$ at 30% signal efficiency.
2. **Angular Coordinates Inversion:** In pure kinematic reconstruction, diffuse QCD gluon/quark showers exhibit higher spatial angular dispersion errors than highly boosted, collimated decay cores. Consequently, $S_{\text{ang}}$ produces an inverted discrimination profile ($0.424 < 0.5$).
3. **Weighting Mechanism Impact:** In the composite score $0.8 S_{pT} + 0.2 S_{\text{ang}}$, the inverted angular term penalizes the discriminative power of $S_{pT}$, dragging matched AUC down to $0.481$.
4. **Physical Reality vs Synthetic Illusion:** Under historical synthetic noise, all four scores reported identical artificial values ($\sim 0.4856$). Real data restores true physical differentiation between kinematic observables.
