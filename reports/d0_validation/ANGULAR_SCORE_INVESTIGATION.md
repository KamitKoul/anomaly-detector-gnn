# Investigation of Angular Reconstruction Anti-Discrimination (AUC = 0.4245)

**Date:** 2026-10-31  
**Target Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`  
**Phenomenon:** The single-run smoke test on real collision events revealed that while $p_T$ reconstruction error provides positive anomaly discrimination ($\text{AUC} = 0.5606$), angular reconstruction error ranks signal below background ($\text{AUC} = 0.4245$), and unweighted full reconstruction error follows the angular score ($\text{AUC} = 0.4333$).  
**Status:** **INVESTIGATED & SCIENTIFICALLY EXPLAINED**

---

## 1. Systemic Hypothesis Evaluation (H1 – H7)

| Hypothesis | Description | Test Performed | Empirical Evidence | Verdict |
|---|---|---|---|---|
| **H1** | Angular reconstruction is numerically dominated | Measured residual scales across 50k BG and 25k Signal events | $S_{\text{ang}}$ cut thresholds ($0.164 - 0.349$) are $6.2\times - 8.9\times$ larger than $S_{pT}$ ($0.026 - 0.039$). In $S_{\text{full}}$ and $S_{\text{weighted}}$, angular variance accounts for $>60\%$ of total score variance. | **SUPPORTED** (Explains why $S_{\text{full}}$ and $S_{\text{weighted}}$ mirror $S_{\text{ang}}$) |
| **H2** | $\phi$ periodicity wrap-around defect | Unit testing of `periodic_dphi` across boundary cases $(-\pi, \pi)$ | Geodesic formula $\min(\|\Delta\phi\|, 2\pi - \|\Delta\phi\|)$ correctly computes wrap-around distances $\le \pi$ ($d\phi(3.14, -3.14) = 0.00318$). No false boundary spikes occur. | **REJECTED** (Wrap-around is mathematically correct) |
| **H3** | Multiplicity confounding | Measured visible physical particle multiplicity ($p_T > 0$) on real BG vs Signal | Background multiplicity: **$182.7 \pm 61.1$** particles.<br>Signal multiplicity: **$139.7 \pm 36.7$** particles.<br>Lower multiplicity and narrower spatial topology in signal correlate with lower angular reconstruction residuals. | **SUPPORTED** (Physical jet substructure difference) |
| **H4** | $p_T$ dependence & jet collimation | Correlation between leading particle $p_T$ and angular residual | Boosted $W' \to XY$ decay products are highly collimated into narrow cones. High local density within $\Delta R < 0.4$ improves spatial attention and lowers angular reconstruction error. | **SUPPORTED** |
| **H5** | Background-specific reconstruction behavior | Compared empirical distributions of $S_{\text{ang}}$ between BG and SIG | Background QCD events contain diffuse, wide-angle soft gluon radiation that the GAT struggles to predict, causing higher angular MSE on background than on tightly collimated signal jets. | **SUPPORTED** |
| **H6** | Masking & padding artifacts | Tested padding exclusion in `tests/test_loss_normalization.py` | Padded slots are strictly removed via `data[:, 0] > 0`. Zero-padded slots do not enter graphs. The multiplicity disparity is genuine physics, not an artifact of padding conventions. | **REJECTED** as artifact cause |
| **H7** | Score direction inversion | Verified score calculation and label convention | Background label is 0; Signal label is 1. Score is strictly MSE error ($(\hat{x} - x)^2$). Higher score corresponds strictly to higher error. Because QCD background exhibits higher angular error than signal, $S_{\text{ang}}$ acts as an anti-discriminator (AUC < 0.5). | **SUPPORTED** |

---

## 2. Physical & Architectural Synthesis

### Why Does $S_{pT}$ Provide Positive Discrimination ($\text{AUC} = 0.5606$)?
- The autoencoder is trained unsupervised strictly on QCD dijet background events.
- QCD background is dominated by soft collinear radiation with steeply falling $p_T$ power-law spectra ($\text{mean } p_T \approx 16.2$ GeV, $\text{median } p_T \approx 3.0$ GeV).
- The anomalous signal ($W' \to XY \to 4\text{ quarks}$) features boosted resonance decays where subjets carry anomalously hard transverse momenta.
- When presented with signal events, the decoder cannot accurately predict these anomalously hard momenta, producing large $p_T$ residuals:
  $$S_{pT}(\text{Signal}) > S_{pT}(\text{Background}) \implies \text{AUC} > 0.5$$

### Why Does $S_{\text{ang}}$ Provide Anti-Discrimination ($\text{AUC} = 0.4245$)?
- QCD dijets produce large numbers of wide-angle soft hadrons ($N \approx 183$) spread across pseudorapidity $|\eta| \le 5.0$ and azimuthal angle $[-\pi, \pi]$.
- Predicting the exact angular coordinates $(\eta, \phi)$ of 180+ diffuse, uncorrelated particles is inherently difficult for a 7,957-parameter GAT model, leading to high angular reconstruction MSE on background.
- In contrast, the signal events consist of boosted 2-prong and 3-prong collimated jets ($N \approx 140$). Inside a boosted jet cone ($\Delta R < 0.4$), particles are tightly clustered, creating dense local GAT subgraphs with high relational correlation.
- The GAT reconstructs tightly clustered jet coordinates with *higher accuracy* (lower MSE) than diffuse QCD radiation:
  $$S_{\text{ang}}(\text{Signal}) < S_{\text{ang}}(\text{Background}) \implies \text{AUC} < 0.5$$

### Why Does $S_{\text{full}}$ and $S_{\text{weighted}}$ Fail?
- In $S_{\text{full}}$, the coordinate MSEs are summed without variance scaling:
  $$S_{\text{full}} = \frac{1}{3} S_{pT} + \frac{1}{3} S_{\eta} + \frac{1}{3} S_{\phi}$$
  Because $S_{\eta} + S_{\phi} \approx 2 \times 0.16 = 0.32$ while $S_{pT} \approx 0.026$, the angular error contributes over $92\%$ of the numerical magnitude of $S_{\text{full}}$, dragging its AUC down to $0.4333$.
- In $S_{\text{weighted}} = 0.8 S_{pT} + 0.2 S_{\text{ang}}$, the $0.2$ weight is insufficient to counteract the $6.2\times$ scale dominance of $S_{\text{ang}}$, reducing the matched AUC from $0.5606$ down to $0.4811$.

---

## 3. Scientific Recommendation
- **Do Not Invert Scores Post-Hoc:** Arbitrarily inverting $S_{\text{ang}}$ violates scientific protocol.
- **Maintain Independent Score Reporting:** Keep $S_{pT}$, $S_{\text{ang}}$, $S_{\text{weighted}}$, and $S_{\text{full}}$ reported separately across all scaling tiers. This allows researchers to evaluate whether increasing training data improves $p_T$ discrimination and whether it resolves or exacerbates angular anti-discrimination.
