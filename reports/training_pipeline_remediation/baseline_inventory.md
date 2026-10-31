a# Baseline Inventory: PointNet-GAT Anomaly Detection Pipeline

**Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Date:** October 31, 2026  
**Scope:** Phase 0 Infrastructure & Scientific Baseline Inventory  
**Auditor:** Scientific Remediation Agent  

---

## 1. System and Environment Provenance

- **Git Commit Hash:** `b98d17d174d3ebaef4f5ea080357824600400672` (Branch `main`, fully synchronized with `origin/main`)
- **Python Runtime:** Python 3.12.14 (`.venv/bin/python`)
- **PyTorch Version:** `2.14.0` (with CUDA/MPS acceleration support)
- **PyTorch Geometric (PyG) Version:** `2.8.0.post1`
- **PyTables Version:** `3.11.1` (HDF5 C-library backend 1.14.x)
- **SciPy Version:** `1.18.1`
- **Scikit-learn Version:** `1.9.1`
- **NumPy Version:** `2.4.2`

---

## 2. Dataset Files and Dimension Inventory

The experimental suite interfaces with datasets from the **LHC Olympics 2020 Anomaly Detection Challenge**:

| File Identifier | Path in Repo | Zenodo Record | Shape / Dimensions | Description |
|---|---|---|---|---|
| **Background MC** | `data/events_LHCO2020_backgroundMC_Pythia.h5` | [4536624](https://zenodo.org/records/4536624) | `(1,000,000, 2100)` float64 | Pure QCD dijet events. 700 particles $\times$ 3 ($p_T, \eta, \phi$). |
| **R&D Mixed Dataset** | Download cache (`events_anomalydetection.h5`) | [4536377](https://zenodo.org/records/4536377) | `(1,100,000, 2101)` float64 | 1M QCD background (`truth=0`) + 100k $W' \to XY$ signal (`truth=1`). |
| **Extracted Signal** | `data/W_prime_signal.h5` | Extracted via [`src/download_signal.py#L101`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/download_signal.py#L101) | `(100,000, 2101)` float64 | Pure $W' \to X(\to q\bar{q})Y(\to q\bar{q})$ resonance events ($m_{W'}=3.5$ TeV). |

- **Event Identifiers:** None. Events are ordered strictly by row index in HDF5 node `/df/block0_values`.
- **Event Weights:** None (unweighted simulation, implicit weight = 1.0).
- **Dijet Invariant Mass Range:**
  - Background: Continuous falling spectrum $m_{jj} \in [2500, 6000]$ GeV.
  - Signal: Resonant peak at $m_{jj} \approx 3500$ GeV with daughter subjets $m_X \approx 500$ GeV, $m_Y \approx 100$ GeV.

---

## 3. Current Sample Slicing and Split Accounting

| Stage | Sourced File | Row Slice | Sample Count | Note |
|---|---|---|---:|---|
| **Training (All Models)** | `events_LHCO2020_backgroundMC_Pythia.h5` | `[0, 5000)` | **5,000** | Sequential first 5,000 rows ([`src/train_experiment.py#L135-L156`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/train_experiment.py#L135-L156)) |
| **Validation (In-Training)** | None | None | **0** | No validation split or early stopping exists |
| **Evaluation Background** | `events_LHCO2020_backgroundMC_Pythia.h5` | `[5000, 15000)` | **10,000** | Sliced using `skip=5000, max_events=10000` ([`evaluation/evaluate_controlled_ablation.py#L196`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_controlled_ablation.py#L196)) |
| **Evaluation Signal** | `data/W_prime_signal.h5` | `[0, 5000)` | **5,000** | Sequential first 5,000 signal rows ([`evaluation/evaluate_controlled_ablation.py#L197`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_controlled_ablation.py#L197)) |
| **Unused Background** | `events_LHCO2020_backgroundMC_Pythia.h5` | `[15000, 1000000)` | **985,000** | 98.5% of background data unused |
| **Unused Signal** | `data/W_prime_signal.h5` | `[5000, 100000)` | **95,000** | 95.0% of signal data unused |

---

## 4. Current Optimizer & Training Hyperparameters

Reference: [`src/train_experiment.py#L406-L445`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/train_experiment.py#L406-L445) and [`configs/hybrid_pointnet_gat/*.json`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/configs/hybrid_pointnet_gat/hybrid_h0_lambda000_s42.json#L10-L16).

- **Optimizer:** `torch.optim.Adam`
- **Learning Rate:** $\eta = 0.001$ (fixed, no scheduler)
- **Weight Decay:** $0.0$
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)`
- **Batch Size:** $B = 32$
- **Epochs:** $E = 20$
- **Batches per Epoch:** $\lceil 5000 / 32 \rceil = 157$ batches
- **Total Optimizer Updates:** $157 \times 20 = \mathbf{3,140}$ parameter updates
- **Dataset Loading Strategy:** All 5,000 PyG `Data` graph objects constructed in-memory before training loop.

---

## 5. Loss Aggregation Conventions

The total loss minimized during training is:
$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{recon}} + \lambda_{\mathrm{MET}} \mathcal{L}_{\mathrm{MET}} + \lambda_{\mathrm{adv}} \mathcal{L}_{\mathrm{adv}}$$

### Component Definitions:
1. **Kinematic Reconstruction Loss ($\mathcal{L}_{\mathrm{recon}}$):**  
   Defined in [`src/run_lhco_gnn.py#L149-L175`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/run_lhco_gnn.py#L149-L175):
   $$\mathcal{L}_{\mathrm{pt}} = \frac{1}{N_{\mathrm{nodes}}} \sum_{i=1}^{N_{\mathrm{nodes}}} (\hat{x}_{i, 0} - x_{i, 0})^2$$
   $$\mathcal{L}_{\mathrm{ang}} = \frac{1}{2 N_{\mathrm{nodes}}} \sum_{i=1}^{N_{\mathrm{nodes}}} \left[ (\hat{x}_{i, 1} - x_{i, 1})^2 + \Delta\phi(\hat{x}_{i, 2}, x_{i, 2})^2 \right]$$
   $$\mathcal{L}_{\mathrm{recon}} = 1.0 \cdot \mathcal{L}_{\mathrm{pt}} + 1.0 \cdot \mathcal{L}_{\mathrm{ang}}$$
   Where $N_{\mathrm{nodes}}$ is the total number of particles across all events in the mini-batch ($\sim 32 \times 700 = 22,400$).

2. **MET Prediction Loss ($\mathcal{L}_{\mathrm{MET}}$):**  
   Defined in [`src/train_experiment.py#L279-L281`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/train_experiment.py#L279-L281):
   $$\mathcal{L}_{\mathrm{MET}} = \frac{1}{2 B} \sum_{b=1}^{B} \left[ (\widehat{\mathrm{MET}}_{x, b} - \mathrm{MET}_{x, b})^2 + (\widehat{\mathrm{MET}}_{y, b} - \mathrm{MET}_{y, b})^2 \right]$$
   - Target units: Unnormalized raw $\text{GeV}$.
   - Typical scale: $\mathcal{L}_{\mathrm{MET}} \sim 4,400\text{ GeV}^2$.
   - Configured weight: $\lambda_{\mathrm{MET}} = 0.10$.

3. **DisCo Regularization Loss ($\mathcal{L}_{\mathrm{adv}}$):**  
   Defined in [`src/adversarial_unlearning_ae.py#L92-L132`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/src/adversarial_unlearning_ae.py#L92-L132):
   $$\mathcal{L}_{\mathrm{adv}} = \mathrm{dCor}^2(S_{\mathrm{batch}}, M_{\mathrm{batch}})$$
   - Evaluated on $B=32$ samples per batch.
   - Target $M$: `compute_invariant_mass(particles) / 1000.0` (all 700 particles in event, in TeV).
   - Configured weight: $\lambda_{\mathrm{adv}} \in [0.0, 1.0]$.

---

## 6. Evaluation Score Definitions

Defined in [`evaluation/evaluate_controlled_ablation.py#L280-L299`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/evaluation/evaluate_controlled_ablation.py#L280-L299):

For event $b$ containing $n_b$ particles:
- **Full Score:** $S_{\mathrm{full}}(b) = \frac{1}{n_b} \sum_{i=1}^{n_b} \frac{1}{3} \left[ (\hat{p}_{T, i} - p_{T, i})^2 + (\hat{\eta}_i - \eta_i)^2 + \Delta\phi(\hat{\phi}_i, \phi_i)^2 \right]$
- **$p_T$-Only Score:** $S_{p_T}(b) = \frac{1}{n_b} \sum_{i=1}^{n_b} (\hat{p}_{T, i} - p_{T, i})^2$
- **Angular-Only Score:** $S_{\mathrm{ang}}(b) = \frac{1}{n_b} \sum_{i=1}^{n_b} \frac{1}{2} \left[ (\hat{\eta}_i - \eta_i)^2 + \Delta\phi(\hat{\phi}_i, \phi_i)^2 \right]$
- **Weighted Score:** $S_{\mathrm{weighted}}(b) = 0.8 \cdot S_{p_T}(b) + 0.2 \cdot S_{\mathrm{ang}}(b)$

Evaluation Metrics:
- **Inclusive ROC-AUC:** Full evaluated sample (10k BG, 5k Signal).
- **Matched ROC-AUC:** Restricting both BG and Signal to $m_{jj} \in [3300, 3700]$ GeV.
- **Micro-Fine ROC-AUC:** Restricting both to $m_{jj} \in [3450, 3550]$ GeV.
- **Rejection at 30% / 50% TPR ($1/\mathrm{FPR}$):** Background rejection factor.
- **Mass Sculpting Metrics:** Base-2 Jensen-Shannon Divergence smoothness ($1/\mathrm{JSD}$) on raw dijet mass and Recursive Soft Drop mass at Top 10%, Top 5%, and Top 1% score cuts.
- **Linear & Distance Correlations:** Pearson $r$, Spearman $\rho$, and population Distance Correlation $\mathrm{dCor}(S, m_{jj})$.
