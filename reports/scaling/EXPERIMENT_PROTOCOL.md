# Research Protocol: Background-Data Scaling Study in LHC Olympics Anomaly Detection

**Document ID:** `reports/scaling/EXPERIMENT_PROTOCOL.md`  
**Status:** Frozen Scientific Protocol (Version 1.0.0)  
**Research Lead:** Antigravity Senior ML Research Engineer (HEP Specialist)  
**Date:** October 31, 2026  

---

## 1. Primary Research Question & Scientific Rationale

### Research Question
> **"How does background training-data scale affect anomaly discrimination, training stability, phase-space generalization, and background-mass sculpting in a fixed PointNet–GAT anomaly detector?"**

### Scientific Context & Hypothesis
Unsupervised anomaly detection in collider physics operates under the hypothesis that a deep generative model or autoencoder trained strictly on Standard Model background events (QCD multijets) will reconstruct anomalous new physics events (such as $W' \to XY$) with significantly higher reconstruction error.

However, two critical scientific questions remain unanswered in high-energy physics graph architectures:
1. **Sample Complexity vs Substructure Representation:** Does increasing background event exposure from 5,000 to 900,000 events allow the Graph Attention Network to learn subtler high-order gluon/quark correlations without overfitting to training phase-space?
2. **Data-Scale Mass Sculpting Trade-off:** Does increased data scale naturally mitigate artificial mass sculpting, or does a more tightly converged background density estimator sculpt the falling dijet invariant mass spectrum even more aggressively?

---

## 2. Experimental Control & Invariance Guarantees

To ensure that changes in performance can be causally attributed to background data scale and not to confounding optimization or architectural variables, the following components are **strictly frozen across all runs**:

| System Component | Frozen Specification | Invariance Enforcement |
| :--- | :--- | :--- |
| **Model Architecture** | `PointNetGATHybrid` | Exactly 7,957 parameters, 2 GAT layers, 4 heads, PointNet $\phi$ MLP |
| **Input Coordinates** | $(p_T, \eta, \phi)$ | Max 700 particles, $p_T \to \ln(1+p_T)$, periodic $\phi \in [0, \pi]$ |
| **Graph Neighborhood** | $\Delta R < 0.4$ | Directed edges, excluding self-loops, periodic wrap-around on azimuthal difference |
| **Loss Function** | Reconstruction-Only | $\mathcal{L} = \mathcal{L}_{\text{recon}}$; $\lambda_{\text{MET}} = 0.0, \lambda_{\text{adv}} = 0.0$ |
| **Batch Size** | $B = 32$ | Exactly 32 collision graphs per mini-batch across all tiers $D_0$ through $D_6$ |
| **Optimizer** | Adam | $\text{lr} = 0.001$, $\beta = (0.9, 0.999)$, $\epsilon = 10^{-8}$, $\text{weight\_decay} = 0.0$ |
| **Gradient Clipping** | Max Norm = 50.0 | Applied to exact FP32 gradients after AMP unscaling |
| **Drop-Last Policy** | `drop_last = True` | Strict rejection of partial mini-batches; dropped event counts logged |
| **Validation Set** | 50,000 BG | Strictly held-out; used exclusively for minimum reconstruction loss early stopping |
| **Test Set** | 50,000 BG / 25,000 SIG | Strictly held-out; identical across every evaluation |

---

## 3. Disentangled Experimental Designs: Experiment A vs Experiment B

A fundamental flaw in naive scaling studies is conflating **data scale** with **total optimizer updates**. If a model trains for 20 epochs on 5,000 events ($D_0$), it performs 3,120 updates. If it trains for 20 epochs on 900,000 events ($D_6$), it performs 562,500 updates (a 180x difference in gradient descent steps!).

To mathematically disentangle data diversity from compute budget, we execute two complementary experiments:

### Experiment A: Fixed Epochs (Passes per Event Held Constant)
* **Hypothesis:** Exposing the network to more unique physical collision topologies while holding the number of passes over each event constant ($E = 20$) improves generalizable feature extraction.
* **Update Budget:** $U(D) = 20 \times \lfloor D / 32 \rfloor$.
* **Expected Updates:**
  - $D_0$ (5k): 3,120 updates
  - $D_1$ (25k): 15,620 updates
  - $D_2$ (50k): 31,240 updates
  - $D_3$ (100k): 62,500 updates
  - $D_4$ (250k): 156,240 updates
  - $D_5$ (500k): 312,500 updates
  - $D_6$ (900k): 562,500 updates

### Experiment B: Fixed Optimizer Steps (Gradient Descent Compute Held Constant)
* **Hypothesis:** Access to a larger and more diverse background pool improves anomaly detection even when the optimizer update budget is held strictly constant.
* **Update Budget:** Exactly **10,000 optimizer steps** ($U = 10,000$).
* **Event Presentations:** Exactly **320,000 event presentations** ($10,000 \times 32$).
* **Mechanism:**
  - For $D_0$ (5,000 events), each event is seen on average 64 times ($320,000 / 5,000$).
  - For $D_6$ (900,000 events), each event is seen at most once, and the model observes only a 35.5% fraction of the available pool.

---

## 4. Evaluation and Anomaly Scoring Protocol

The frozen evaluation harness evaluates four anomaly score formulations across both the inclusive spectrum and targeted signal mass windows:

1. **Score Formulations:**
   - Full: $S_{\text{full}} = \frac{1}{3}(S_{p_T} + S_\eta + S_\phi)$
   - $p_T$-Only: $S_{p_T} = \text{MSE}(\hat{p}_T, p_T)$
   - Angular-Only: $S_{\text{ang}} = \frac{1}{2}(\text{MSE}_\eta + \text{MSE}_\phi)$
   - Weighted: $S_{\text{weighted}} = 0.8 S_{p_T} + 0.2 S_{\text{ang}}$

2. **Discrimination Metrics:**
   - Inclusive ROC-AUC ($m_{jj} \in [2.5, 5.0]$ TeV).
   - Mass-Matched ROC-AUC ($m_{jj} \in [3.3, 3.7]$ TeV).
   - Narrow-Window ROC-AUC ($m_{jj} \in [3.45, 3.55]$ TeV).
   - Background Rejection ($1/\text{FPR}$) at $\epsilon_S = 0.30$ and $\epsilon_S = 0.50$.
   - Continuous Significance Improvement Characteristic: $\text{SIC}(\epsilon_S) = \frac{\epsilon_S}{\sqrt{\text{FPR}(\epsilon_S)}}$.

3. **Mass-Sculpting Metrics:**
   - Base-2 Jensen-Shannon Divergence ($1/\text{JSD}$ and raw $\text{JSD}$) at Top 10%, Top 5%, and Top 1% anomaly cuts.
   - Evaluated on both raw dijet mass ($m_{\text{raw}}$) and Recursive Soft Drop ($m_{\text{RSD}}$).
   - Pearson $r$ and Spearman $\rho$ correlation.
