# Controlled Architecture & Anomaly Score Ablation Summary

| Experiment | Model Class | Seed | Full Matched AUC | pT-Only Matched AUC | Angular Matched AUC | 1/JSD Raw (Top 10%) | Pearson r |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_mlp_s123` | baseline_mlp | 123 | 0.5164 | 0.6586 | 0.4315 | 179.7 ± 17.2 | -0.0372 |
| `baseline_mlp_s42` | baseline_mlp | 42 | 0.5123 | 0.6591 | 0.4268 | 200.9 ± 18.6 | -0.0361 |
| `baseline_mlp_s999` | baseline_mlp | 999 | 0.5117 | 0.6600 | 0.4241 | 211.3 ± 18.2 | -0.0364 |
| `deepsets_pointnet_s123` | deepsets_pointnet | 123 | 0.7320 | 0.7524 | 0.7016 | 64.6 ± 7.5 | +0.1166 |
| `deepsets_pointnet_s42` | deepsets_pointnet | 42 | 0.6943 | 0.7323 | 0.6098 | 34.9 ± 4.0 | +0.1723 |
| `deepsets_pointnet_s999` | deepsets_pointnet | 999 | 0.7565 | 0.6404 | 0.7563 | 63.0 ± 7.1 | +0.1226 |
| `gat_relational_s123` | gat_autoencoder | 123 | 0.6838 | 0.7692 | 0.4274 | 140.9 ± 13.8 | +0.0852 |
| `gat_relational_s42` | gat_autoencoder | 42 | 0.6867 | 0.7701 | 0.4337 | 148.2 ± 14.9 | +0.0818 |
| `gat_relational_s999` | gat_autoencoder | 999 | 0.6831 | 0.7707 | 0.4224 | 120.9 ± 12.3 | +0.0855 |
| `nodewise_ae_s123` | node_wise_ae | 123 | 0.4137 | 0.4254 | 0.4236 | 94.4 ± 11.3 | +0.1394 |
| `nodewise_ae_s42` | node_wise_ae | 42 | 0.4333 | 0.4420 | 0.4410 | 64.8 ± 7.5 | +0.1361 |
| `nodewise_ae_s999` | node_wise_ae | 999 | 0.4689 | 0.4159 | 0.4952 | 72.2 ± 8.2 | +0.1216 |
