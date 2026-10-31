# Reconstruction Score Definition and Numerical Scaling Audit

**Date:** 2026-10-31  
**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Status:** **AUDITED & DOCUMENTED**

---

## 1. Mathematical Definitions of the Four Anomaly Scores

All anomaly scores are computed using the reconstructed node coordinates from the PointNet–GAT Hybrid autoencoder. For each event $g$ containing $N_g$ visible particles plus 1 MET node ($N_g^{\text{total}} = N_g + 1$ nodes):

### Score 1 — $p_T$ Reconstruction Error ($S_{pT}$)
Measures transverse momentum reconstruction error in log-compressed feature space:
\[
S_{pT}(g) = \frac{1}{N_g^{\text{total}}} \sum_{i=1}^{N_g^{\text{total}}} \left(\log(1 + p_{T,i}^{\text{recon}}) - \log(1 + p_{T,i})\right)^2
\]

### Score 2 — Angular Reconstruction Error ($S_{\text{ang}}$)
Measures spatial orientation error across pseudorapidity ($\eta$) and azimuthal angle ($\phi$):
\[
S_{\text{ang}}(g) = \frac{1}{N_g^{\text{total}}} \sum_{i=1}^{N_g^{\text{total}}} \frac{1}{2} \left[ (\eta_i^{\text{recon}} - \eta_i)^2 + (\Delta\phi_i)^2 \right]
\]
where $\Delta\phi_i$ is computed with strict periodic Riemannian wrap-around on the circle $S^1$:
\[
\Delta\phi_i = \min\left( |\phi_i^{\text{recon}} - \phi_i|, \, 2\pi - |\phi_i^{\text{recon}} - \phi_i| \right)
\]
**Boundary Verification:** Direct subtraction is avoided. Boundary wrap-around cases (e.g., $\phi = 3.14$ vs. $\phi = -3.14$) evaluate to $\Delta\phi = 0.00318$ rather than $6.28$, preventing false high anomaly scores near $\pm \pi$.

### Score 3 — Weighted Reconstruction Error ($S_{\text{weighted}}$)
A linear combination of transverse momentum error and angular error:
\[
S_{\text{weighted}}(g) = 0.8 \, S_{pT}(g) + 0.2 \, S_{\text{ang}}(g)
\]

### Score 4 — Full Reconstruction Error ($S_{\text{full}}$)
The unweighted arithmetic mean across all three coordinate reconstruction errors:
\[
S_{\text{full}}(g) = \frac{1}{N_g^{\text{total}}} \sum_{i=1}^{N_g^{\text{total}}} \frac{1}{3} \left[ \left(\log(1 + p_{T,i}^{\text{recon}}) - \log(1 + p_{T,i})\right)^2 + (\eta_i^{\text{recon}} - \eta_i)^2 + (\Delta\phi_i)^2 \right]
\]

---

## 2. Numerical Scale Disparity & Component Contribution

Evaluation across 50,000 background and 25,000 signal collision events demonstrates an order-of-magnitude scale disparity between coordinate residuals:

| Score Metric | Cut Threshold (Top 10%) | Cut Threshold (Top 5%) | Cut Threshold (Top 1%) | Matched AUC |
|---|---:|---:|---:|---:|
| **$S_{pT}$ (Transverse Momentum)** | 0.02644 | 0.03045 | 0.03896 | **0.5606** |
| **$S_{\text{ang}}$ (Angular Coordinates)** | 0.16416 | 0.21399 | 0.34853 | **0.4245** |
| **$S_{\text{weighted}}$ ($0.8 S_{pT} + 0.2 S_{\text{ang}}$)** | 0.05103 | 0.06234 | 0.09089 | **0.4811** |
| **$S_{\text{full}}$ (Unweighted Mean)** | 0.11667 | 0.15062 | 0.24022 | **0.4333** |

### Numerical Dominance Analysis
1. **Residual Scale Ratio:** Angular error residuals ($S_{\text{ang}} \sim 0.16 - 0.35$) are **$6.2 \times$ to $8.9 \times$ larger** than momentum residuals ($S_{pT} \sim 0.026 - 0.039$).
2. **Effective Weighting in $S_{\text{weighted}}$:** Even though configured as $80\%$ $p_T$ and $20\%$ angular, the $6.2\times$ scale dominance means that the effective contribution to the variance of $S_{\text{weighted}}$ is:
   $$\text{Weight}_{pT} \times \text{Scale}_{pT} = 0.8 \times 0.0264 = 0.0211$$
   $$\text{Weight}_{\text{ang}} \times \text{Scale}_{\text{ang}} = 0.2 \times 0.1642 = 0.0328$$
   The angular component contributes **$60.8\%$ of the score magnitude**, completely overpowering the $p_T$ component!
3. **Impact on AUC:** Because $S_{\text{ang}}$ exhibits an AUC of $0.4245$ (acting as an anti-discriminator), allowing it to numerically dominate $S_{\text{weighted}}$ pulls the overall matched AUC down from $0.5606$ to $0.4811$.
4. **Conclusion:** All four scores are mathematically consistent, but heterogeneous coordinate units without standardization cause unweighted or lightly weighted sums to be dominated by angular error. Per scientific constraints, score definitions remain unmanipulated.
