# Controlled Architecture & Anomaly Score Ablation Summary

All AUCs reported with 68% stratified bootstrap confidence intervals (1000 resamples).
JSD metrics strictly computed in base-2 (bits).

| Experiment | Model Class | Seed | Full Matched AUC | pT-Only Matched AUC | Angular Matched AUC | 1/JSD Raw (Top 10%) | Pearson r |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_mlp_s123` | baseline_mlp | 123 | 0.5180 ± 0.0094 | 0.6667 ± 0.0088 | 0.4191 ± 0.0093 | 138.8 ± 12.8 | -0.0467 |
| `baseline_mlp_s42` | baseline_mlp | 42 | 0.5185 ± 0.0097 | 0.6584 ± 0.0090 | 0.4465 ± 0.0096 | 133.2 ± 13.4 | -0.0557 |
| `baseline_mlp_s999` | baseline_mlp | 999 | 0.5028 ± 0.0094 | 0.6613 ± 0.0089 | 0.4223 ± 0.0092 | 96.5 ± 9.4 | -0.0482 |
| `deepsets_pointnet_s123` | deepsets_pointnet | 123 | 0.5893 ± 0.0088 | 0.4855 ± 0.0095 | 0.6063 ± 0.0088 | 41.2 ± 4.7 | +0.1190 |
| `deepsets_pointnet_s42` | deepsets_pointnet | 42 | 0.7717 ± 0.0073 | 0.5440 ± 0.0093 | 0.7778 ± 0.0072 | 51.4 ± 5.8 | +0.1313 |
| `deepsets_pointnet_s999` | deepsets_pointnet | 999 | 0.7291 ± 0.0079 | 0.6104 ± 0.0088 | 0.7542 ± 0.0078 | 66.2 ± 7.9 | +0.0811 |
| `gat_relational_s123` | gat_autoencoder | 123 | 0.7675 ± 0.0082 | 0.7677 ± 0.0082 | 0.3850 ± 0.0088 | 37.5 ± 4.2 | +0.1845 |
| `gat_relational_s42` | gat_autoencoder | 42 | 0.4841 ± 0.0094 | 0.5111 ± 0.0093 | 0.3335 ± 0.0084 | 145.1 ± 14.4 | +0.0331 |
| `gat_relational_s999` | gat_autoencoder | 999 | 0.7714 ± 0.0082 | 0.7713 ± 0.0082 | 0.3855 ± 0.0087 | 36.5 ± 3.9 | +0.1797 |
| `nodewise_ae_s123` | node_wise_ae | 123 | 0.4588 ± 0.0092 | 0.4849 ± 0.0092 | 0.4585 ± 0.0091 | 57.5 ± 6.9 | +0.1353 |
| `nodewise_ae_s42` | node_wise_ae | 42 | 0.5092 ± 0.0095 | 0.4419 ± 0.0093 | 0.5220 ± 0.0094 | 55.1 ± 6.6 | +0.1532 |
| `nodewise_ae_s999` | node_wise_ae | 999 | 0.4379 ± 0.0094 | 0.4143 ± 0.0089 | 0.4493 ± 0.0095 | 53.2 ± 6.5 | +0.1527 |
