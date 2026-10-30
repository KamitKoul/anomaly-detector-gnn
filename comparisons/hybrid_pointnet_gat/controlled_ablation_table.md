# Controlled Architecture & Anomaly Score Ablation Summary

All AUCs reported with 68% stratified bootstrap confidence intervals (1000 resamples).
JSD metrics strictly computed in base-2 (bits).

| Experiment | Model Class | Seed | Full Matched AUC | pT-Only Matched AUC | Angular Matched AUC | 1/JSD Raw (Top 10%) | Pearson r |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `hybrid_baseline_s123` | pointnet_gat_hybrid | 123 | 0.5587 ± 0.0086 | 0.5815 ± 0.0086 | 0.5443 ± 0.0086 | 71.3 ± 8.5 | +0.1089 |
| `hybrid_baseline_s42` | pointnet_gat_hybrid | 42 | 0.5728 ± 0.0087 | 0.4381 ± 0.0088 | 0.5883 ± 0.0088 | 121.2 ± 14.5 | +0.1052 |
| `hybrid_baseline_s999` | pointnet_gat_hybrid | 999 | 0.6442 ± 0.0088 | 0.6824 ± 0.0088 | 0.6008 ± 0.0089 | 90.8 ± 10.8 | +0.0839 |
| `hybrid_disco_weighted_s123` | score_decorrelated_hybrid | 123 | 0.6159 ± 0.0088 | 0.6214 ± 0.0086 | 0.5653 ± 0.0091 | 160.2 ± 14.2 | +0.0114 |
| `hybrid_disco_weighted_s42` | score_decorrelated_hybrid | 42 | 0.4413 ± 0.0090 | 0.4795 ± 0.0092 | 0.3874 ± 0.0090 | 148.9 ± 15.9 | +0.0166 |
| `hybrid_disco_weighted_s999` | score_decorrelated_hybrid | 999 | 0.6149 ± 0.0095 | 0.5789 ± 0.0094 | 0.6138 ± 0.0093 | 107.8 ± 11.4 | +0.1000 |
