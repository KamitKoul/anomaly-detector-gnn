# Real-Data Smoke Test Execution Report

**Experiment:** Experiment A (Fixed Epochs)  
**Tier:** D0 (5,000 Real Background Events)  
**Seed:** 42  
**Device:** Apple Silicon Metal Performance Shaders (`mps`)  
**Timestamp:** 2026-10-31T12:50:00Z  
**Verdict:** **PASS — ALL 15 INTEGRITY GATES VERIFIED**  

---

## 1. Fifteen Verification Integrity Gates

| Gate # | Integrity Requirement | Measured Value / Evidence | Verification Status |
|---|---|---|:---:|
| **1** | Real HDF5 input loaded | `/Users/koul/.../data/events_LHCO2020_backgroundMC_Pythia.h5` | **PASS** |
| **2** | Correct 5,000-event partition selected | `data/splits/bg_train_d0_5k.npy` (indices $\in [0, 999,999]$) | **PASS** |
| **3** | Actual physics tensors reach model | Batch shape: $[N_{\text{nodes}}, 3]$, zero NaNs, zero Infs | **PASS** |
| **4** | Forward pass succeeds | Reconstruction tensor $\hat{X}$ output shape matches $X$ | **PASS** |
| **5** | Reconstruction loss is finite | Initial train loss: $1.23246$ | **PASS** |
| **6** | Backward pass succeeds | PyTorch autograd backward completed without error | **PASS** |
| **7** | Gradients are finite | Mean gradient norm: $1.7471$ (clipped at $50.0$) | **PASS** |
| **8** | Optimizer updates occur | Adam optimizer steps executed for every mini-batch | **PASS** |
| **9** | Update count matches configured budget | 156 updates ($1 \times \lfloor 5000 / 32 \rfloor = 156$ updates) | **PASS** |
| **10** | Checkpoints saved | `checkpoint_best.pt` and `checkpoint_final.pt` generated | **PASS** |
| **11** | Checkpoint reload succeeds | Weights loaded cleanly into `PointNetGATHybrid` (7,957 params) | **PASS** |
| **12** | Real validation data evaluated | Full 50,000 held-out background events ($\mathcal{L}_{\text{val}} = 0.11032$) | **PASS** |
| **13** | Real prediction arrays saved | Evaluated on frozen test partitions; results logged | **PASS** |
| **14** | Evaluation sample counts correct | Exactly 50,000 Background and 25,000 Signal test events | **PASS** |
| **15** | No synthetic code path used | `metadata.is_synthetic == False`, `metadata.is_debug_only == False` | **PASS** |

---

## 2. Cryptographic Provenance Manifest

All generated artifacts for the real-data smoke test have been cryptographically hashed:

```text
3b704d956850ab7548b54bff5fe6a66dbd675b3f3377ec3ad8ad587ddb9a8104  expA_d0_5k_s42_checkpoint_best.pt
ce8534efe69ee33f9219e21f4f50a678992c8963f35480c958150be7006fbad5  expA_d0_5k_s42_checkpoint_final.pt
5060affbe409d81fd8655cde75b79c683471903331a284704ea518bd84b1e344  expA_d0_5k_s42_training_log.json
071677a3ed7d305db307a7a324ea685fe0ed430def768d4dd9176e0d9a04e42a  expA_d0_5k_s42_eval_metrics.json
```

- **Git Commit:** `b98d17d174d3ebaef4f5ea080357824600400672`
- **Background Dataset SHA-256:** `97edb2ac7a894102a5c8377e489c73ae871d487d114430e917b9f684dea36ed4`
- **Signal Dataset SHA-256:** `624afc85740010fd4e0668ca772784785918891b5bd0e22ca07a4a5ffc7346dc`
- **Training Duration:** 113.98 seconds (Train step + 50k validation step)
- **Evaluation Duration:** 610.49 seconds (50k BG + 25k Signal FastJet grooming + GNN scoring + 6,000 bootstrap iterations)

---

## 3. Real Physics Discrimination & Mass-Sculpting Baseline

| Anomaly Score Metric | Inclusive Test AUC | Matched-Mass AUC [3.3, 3.7] TeV | Narrow-Window AUC [3.45, 3.55] TeV | Rejection at 30% Eff | Top 10% 1/JSD |
|---|---|---|---|---|---|
| **$S_{\text{full}}$ (All Kinematics)** | $0.4338 \pm 0.0022$ | $0.4333 \pm 0.0040$ | $0.4348 \pm 0.0074$ | 2.73 | 620.8 |
| **$S_{pT}$ ($p_T$ Component Only)** | **$0.5691 \pm 0.0022$** | **$0.5606 \pm 0.0044$** | **$0.5623 \pm 0.0084$** | **4.43** | 125.5 |
| **$S_{\text{ang}}$ ($\eta, \phi$ Angles Only)** | $0.4239 \pm 0.0022$ | $0.4245 \pm 0.0040$ | $0.4258 \pm 0.0073$ | 2.67 | 635.9 |
| **$S_{\text{weighted}}$ ($0.8 S_{pT} + 0.2 S_{\text{ang}}$)** | $0.4857 \pm 0.0022$ | $0.4811 \pm 0.0043$ | $0.4825 \pm 0.0080$ | 3.21 | 506.1 |

### Scientific Finding on Real Collision Data
- Unsupervised reconstruction of particle $p_T$ provides positive anomaly discrimination ($0.561$ matched AUC), whereas angular coordinate reconstruction error is anti-correlated with anomaly status ($0.424$).
- The historical weighting $0.8 S_{pT} + 0.2 S_{\text{ang}}$ drags the combined discrimination down to $0.481$.
- At Top 1% anomaly cut on real data, Base-2 JSD is $0.013763$ ($1/\text{JSD} = 72.7$), proving that the previously reported ideal invariance ($0.0, 9999.0$) was purely an artifact of the guard-clause fallback on $N=5 < 10$.
