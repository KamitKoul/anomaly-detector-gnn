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
| `hybrid_h0_lambda000_s123` | pointnet_gat_hybrid | 123 | 0.5587 ± 0.0086 | 0.5815 ± 0.0086 | 0.5443 ± 0.0086 | 71.3 ± 8.5 | +0.1089 |
| `hybrid_h0_lambda000_s42` | pointnet_gat_hybrid | 42 | 0.5728 ± 0.0087 | 0.4381 ± 0.0088 | 0.5883 ± 0.0088 | 121.2 ± 14.5 | +0.1052 |
| `hybrid_h0_lambda000_s999` | pointnet_gat_hybrid | 999 | 0.6442 ± 0.0088 | 0.6824 ± 0.0088 | 0.6008 ± 0.0089 | 90.8 ± 10.8 | +0.0839 |
| `hybrid_h1_lambda010_s123` | score_decorrelated_hybrid | 123 | 0.5905 ± 0.0092 | 0.5179 ± 0.0093 | 0.6165 ± 0.0089 | 126.3 ± 13.1 | +0.0361 |
| `hybrid_h1_lambda010_s42` | score_decorrelated_hybrid | 42 | 0.4753 ± 0.0091 | 0.4672 ± 0.0090 | 0.4951 ± 0.0090 | 135.6 ± 13.4 | +0.0576 |
| `hybrid_h1_lambda010_s999` | score_decorrelated_hybrid | 999 | 0.7018 ± 0.0079 | 0.7020 ± 0.0082 | 0.5472 ± 0.0092 | 42.0 ± 4.6 | +0.1995 |
| `hybrid_h2_lambda025_s123` | score_decorrelated_hybrid | 123 | 0.5250 ± 0.0092 | 0.5722 ± 0.0089 | 0.4626 ± 0.0094 | 103.1 ± 11.3 | +0.0351 |
| `hybrid_h2_lambda025_s42` | score_decorrelated_hybrid | 42 | 0.4592 ± 0.0088 | 0.4911 ± 0.0090 | 0.4208 ± 0.0089 | 107.5 ± 11.0 | +0.0229 |
| `hybrid_h2_lambda025_s999` | score_decorrelated_hybrid | 999 | 0.6051 ± 0.0089 | 0.5681 ± 0.0090 | 0.6172 ± 0.0085 | 64.7 ± 7.4 | +0.1225 |
| `hybrid_h3_lambda050_s123` | score_decorrelated_hybrid | 123 | 0.4995 ± 0.0094 | 0.4775 ± 0.0095 | 0.5487 ± 0.0091 | 171.0 ± 16.1 | +0.0317 |
| `hybrid_h3_lambda050_s42` | score_decorrelated_hybrid | 42 | 0.4331 ± 0.0088 | 0.5333 ± 0.0093 | 0.3727 ± 0.0084 | 138.6 ± 14.1 | +0.0107 |
| `hybrid_h3_lambda050_s999` | score_decorrelated_hybrid | 999 | 0.5239 ± 0.0094 | 0.3393 ± 0.0084 | 0.6660 ± 0.0082 | 92.5 ± 10.5 | +0.0421 |
| `hybrid_h4_lambda100_s123` | score_decorrelated_hybrid | 123 | 0.6159 ± 0.0088 | 0.6214 ± 0.0086 | 0.5653 ± 0.0091 | 160.2 ± 14.2 | +0.0114 |
| `hybrid_h4_lambda100_s42` | score_decorrelated_hybrid | 42 | 0.4413 ± 0.0090 | 0.4795 ± 0.0092 | 0.3874 ± 0.0090 | 148.9 ± 15.9 | +0.0166 |
| `hybrid_h4_lambda100_s999` | score_decorrelated_hybrid | 999 | 0.6149 ± 0.0095 | 0.5789 ± 0.0094 | 0.6138 ± 0.0093 | 107.8 ± 11.4 | +0.1000 |
