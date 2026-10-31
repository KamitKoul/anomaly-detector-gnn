# Scientific Experiment Plan: Large-Scale Dataset Scaling (D0 to D6)

**Repository:** `https://github.com/KamitKoul/anomaly-detector-gnn`  
**Date:** October 31, 2026  
**Auditor:** Scientific Remediation Agent  
**Scope:** Phase 7 Experimental Design for Data-Scaling across Seeds 42, 123, 999  
**Execution Status:** Ready for GPU Deployment (Configurations generated; Execution paused per completion gate)  

---

## 1. Executive Summary

To definitively answer whether the PointNet-GAT Hybrid's weak anomaly discrimination (AUC $\approx 0.56 - 0.59$) and seed instability are caused by data starvation, we have prepared a comprehensive large-scale scaling study spanning seven dataset magnitudes ($5,000$ to $1,000,000$ events) across three independent random seeds (`42, 123, 999`).

The study implements two distinct, scientifically isolated experimental regimes:
- **Experiment A (Equal Epochs):** Fixes training at $20$ epochs across all scales, testing the impact of dataset size when compute scales proportionally ($3,140 \to 78,140$ parameter updates).
- **Experiment B (Equal Optimizer Updates):** Fixes compute at $\approx 10,000$ optimizer updates across all scales, isolating the effect of dataset diversity from compute budget (varying data exposures from $64\times$ at D0 down to $3\times$ at D6).

All 42 JSON configurations have been generated and validated in [`configs/large_scale_scaling/`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/configs/large_scale_scaling).

---

## 2. Experimental Factor Isolation Protocol

> [!IMPORTANT]
> **Strict Scientific Rule:** To prevent conflation of confounding factors, we do NOT vary architecture, loss scaling, DisCo regularization, and dataset size simultaneously.

1. **Architecture Fixed:** `pointnet_gat_hybrid` (7,957 parameters, 4 attention heads, 32 latent dimensions) across all runs.
2. **Optimizer Fixed:** Adam, learning rate $\eta = 0.001$, zero weight decay, gradient clip norm $50.0$.
3. **MET Loss Rescaled:** Remediated $\lambda_{\mathrm{MET}} = 0.0010$ (eliminates historical gradient swamping).
4. **DisCo Regularization Isolated:** Initially $\lambda_{\mathrm{adv}} = 0.0$ across all D0–D6 baseline runs to isolate the pure effect of training data volume. DisCo regularized variants will be evaluated in a secondary phase once the unregularized scaling curve is established.
5. **Fixed Evaluation and Validation Sets:** All configurations use the identical held-out validation set (`bg_val_50k.npy`, $50,000$ events) and publication test set (`bg_test_50k.npy`, $50,000$ events; `sig_test_25k.npy`, $25,000$ events).

---

## 3. Scaling Grid and Training Budget Specifications

### Scaling Points Summary (Seeds: `42, 123, 999`)

| Scale Point | Background Train Events | Partition Array (`data/splits/`) | Batch Size ($B$) | Batches / Epoch | Exp A: Epochs ($E_A$) | Exp A: Total Updates | Exp B: Epochs ($E_B$) | Exp B: Total Updates | Exp B: Data Exposures |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **D0** | $5,000$ | `bg_train_d0_5k.npy` | 32 | 157 | 20 | **$3,140$** | 64 | **$10,048$** | $64\times$ |
| **D1** | $25,000$ | `bg_train_d1_25k.npy` | 32 | 782 | 20 | **$15,640$** | 13 | **$10,166$** | $13\times$ |
| **D2** | $50,000$ | `bg_train_d2_50k.npy` | 64 | 782 | 20 | **$15,640$** | 13 | **$10,166$** | $13\times$ |
| **D3** | $100,000$ | `bg_train_d3_100k.npy` | 128 | 782 | 20 | **$15,640$** | 13 | **$10,166$** | $13\times$ |
| **D4** | $250,000$ | `bg_train_d4_250k.npy` | 128 | 1,954 | 20 | **$39,080$** | 5 | **$9,770$** | $5\times$ |
| **D5** | $500,000$ | `bg_train_d5_500k.npy` | 128 | 3,907 | 20 | **$78,140$** | 3 | **$11,721$** | $3\times$ |
| **D6 (Full)**| $900,000$ | `bg_train_d6_900k.npy` | 256 | 3,516 | 10 | **$35,160$** | 3 | **$10,548$** | $3\times$ |

---

## 4. Scientific Questions Answered by Regimes A and B

```
                     +-------------------------------------------------------------+
                     |            TWO SCIENTIFIC SCALING COMPARISONS               |
                     +------------------------------+------------------------------+
                     | EXPERIMENT A: Equal Epochs   | EXPERIMENT B: Equal Updates  |
                     | Fixed: 20 Epochs             | Fixed: ~10,000 Updates       |
                     | Compute scales with data     | Compute held strictly fixed  |
                     +------------------------------+------------------------------+
                     | Answers:                     | Answers:                     |
                     | "Does giving the GNN more    | "Is 1M unique events seen    |
                     | data and proportional compute| 3 times superior to 5k events|
                     | break through the AUC 0.60   | recycled 64 times under the  |
                     | performance plateau?"        | exact same compute budget?"  |
                     +------------------------------+------------------------------+
```

---

## 5. Configuration Inventory

All 42 JSON configurations are generated in [`configs/large_scale_scaling/`](file:///Users/koul/Library/Mobile%20Documents/com~apple~CloudDocs/anomaly-detector-gnn/configs/large_scale_scaling):

### Experiment A (Equal Epochs: 21 Configs)
- `scale_expA_d0_5k_s42.json`, `scale_expA_d0_5k_s123.json`, `scale_expA_d0_5k_s999.json`
- `scale_expA_d1_25k_s42.json`, `scale_expA_d1_25k_s123.json`, `scale_expA_d1_25k_s999.json`
- `scale_expA_d2_50k_s42.json`, `scale_expA_d2_50k_s123.json`, `scale_expA_d2_50k_s999.json`
- `scale_expA_d3_100k_s42.json`, `scale_expA_d3_100k_s123.json`, `scale_expA_d3_100k_s999.json`
- `scale_expA_d4_250k_s42.json`, `scale_expA_d4_250k_s123.json`, `scale_expA_d4_250k_s999.json`
- `scale_expA_d5_500k_s42.json`, `scale_expA_d5_500k_s123.json`, `scale_expA_d5_500k_s999.json`
- `scale_expA_d6_1m_s42.json`, `scale_expA_d6_1m_s123.json`, `scale_expA_d6_1m_s999.json`

### Experiment B (Equal Updates: 21 Configs)
- `scale_expB_d0_5k_s42.json`, `scale_expB_d0_5k_s123.json`, `scale_expB_d0_5k_s999.json`
- `scale_expB_d1_25k_s42.json`, `scale_expB_d1_25k_s123.json`, `scale_expB_d1_25k_s999.json`
- `scale_expB_d2_50k_s42.json`, `scale_expB_d2_50k_s123.json`, `scale_expB_d2_50k_s999.json`
- `scale_expB_d3_100k_s42.json`, `scale_expB_d3_100k_s123.json`, `scale_expB_d3_100k_s999.json`
- `scale_expB_d4_250k_s42.json`, `scale_expB_d4_250k_s123.json`, `scale_expB_d4_250k_s999.json`
- `scale_expB_d5_500k_s42.json`, `scale_expB_d5_500k_s123.json`, `scale_expB_d5_500k_s999.json`
- `scale_expB_d6_1m_s42.json`, `scale_expB_d6_1m_s123.json`, `scale_expB_d6_1m_s999.json`

---

## 6. Execution Guidelines for Cloud GPU (Kaggle / Colab)

1. **Storage Setup:**  
   Download datasets to `/kaggle/working/data` or `/tmp/data` using `scripts/download_datasets.py`.
2. **Streaming Execution:**  
   The execution script will invoke `ChunkedLHCODataset` with `num_workers=2` and `pin_memory=True`. Peak RAM will remain below $500$ MB throughout the entire run.
3. **Execution Prohibition:**  
   As specified in the completion gate instructions, training of these large-scale configurations is **strictly deferred** until the user reviews and authorizes execution.
