# Controlled Architecture & Anomaly Score Ablation Summary

All AUCs reported with 68% stratified bootstrap confidence intervals (1000 resamples).
JSD metrics strictly computed in base-2 (bits).

| Experiment | Model Class | Seed | Full Matched AUC | pT-Only Matched AUC | Angular Matched AUC | 1/JSD Raw (Top 10%) | Pearson r |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `gat_baseline_ref_s123` | score_decorrelated_gat | 123 | 0.7398 ± 0.0087 | 0.7565 ± 0.0083 | 0.4250 ± 0.0089 | 53.2 ± 6.7 | +0.1529 |
| `gat_baseline_ref_s42` | score_decorrelated_gat | 42 | 0.7456 ± 0.0086 | 0.7474 ± 0.0085 | 0.3746 ± 0.0086 | 32.0 ± 3.6 | +0.1859 |
| `gat_baseline_ref_s999` | score_decorrelated_gat | 999 | 0.7382 ± 0.0087 | 0.7496 ± 0.0085 | 0.4014 ± 0.0088 | 41.6 ± 5.1 | +0.1794 |
| `gat_score_disco_heavy_s42` | score_decorrelated_gat | 42 | 0.7453 ± 0.0087 | 0.7488 ± 0.0086 | 0.4544 ± 0.0090 | 32.9 ± 3.7 | +0.1960 |
| `gat_score_disco_s123` | score_decorrelated_gat | 123 | 0.7424 ± 0.0086 | 0.7448 ± 0.0085 | 0.3523 ± 0.0085 | 30.9 ± 3.4 | +0.1833 |
| `gat_score_disco_s42` | score_decorrelated_gat | 42 | 0.7361 ± 0.0086 | 0.7491 ± 0.0084 | 0.3946 ± 0.0089 | 43.2 ± 4.8 | +0.1817 |
| `gat_score_disco_s999` | score_decorrelated_gat | 999 | 0.7283 ± 0.0089 | 0.7352 ± 0.0088 | 0.3358 ± 0.0083 | 31.5 ± 3.5 | +0.1991 |
| `gat_score_disco_weighted_s123` | score_decorrelated_gat | 123 | 0.7366 ± 0.0087 | 0.7566 ± 0.0084 | 0.4511 ± 0.0094 | 61.3 ± 7.3 | +0.1425 |
| `gat_score_disco_weighted_s42` | score_decorrelated_gat | 42 | 0.7338 ± 0.0088 | 0.7647 ± 0.0082 | 0.4408 ± 0.0095 | 75.0 ± 9.2 | +0.1271 |
| `gat_score_disco_weighted_s999` | score_decorrelated_gat | 999 | 0.7105 ± 0.0091 | 0.7465 ± 0.0085 | 0.3992 ± 0.0085 | 68.2 ± 8.0 | +0.1391 |
| `gat_score_grl_s123` | score_decorrelated_gat | 123 | 0.7439 ± 0.0086 | 0.7457 ± 0.0085 | 0.3639 ± 0.0085 | 32.3 ± 3.4 | +0.1863 |
| `gat_score_grl_s42` | score_decorrelated_gat | 42 | 0.7427 ± 0.0085 | 0.7573 ± 0.0083 | 0.4213 ± 0.0091 | 44.6 ± 5.1 | +0.1747 |
| `gat_score_grl_s999` | score_decorrelated_gat | 999 | 0.7257 ± 0.0089 | 0.7322 ± 0.0088 | 0.3508 ± 0.0083 | 32.2 ± 3.5 | +0.1943 |
