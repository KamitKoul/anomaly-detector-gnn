# Comparison Summary: Architecture

| Experiment | ROC AUC | Rejection @ 50% Eff | Separation Ratio | Significance |
| :--- | :---: | :---: | :---: | :---: |
| `baseline_mlp` | 0.4848 | 1.91 | 0.9755x | -0.04σ |
| `gat_standalone` | 0.7815 | 6.27 | 1.2414x | 0.96σ |
| `gat_met_production` | 0.6584 | 3.25 | 1.1561x | 0.36σ |
| `gat_met_decorrelation` | 0.6608 | 3.25 | 1.1596x | 0.35σ |
