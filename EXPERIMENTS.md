# Controlled Experiment Framework Guide

This document describes the reproducible experiment framework for the `anomaly-detector-gnn` research repository, established in Phase 1 following the frozen baseline (`pre-experiments-684c99e` at commit `684c99e`).

---

## 1. Experiment Families

The framework supports four controlled scientific experiment families. Each experiment varies exactly **one** scientific factor at a time while holding all other conditions identical to the reference baseline.

| Family | Subdirectory | Scientific Factor Varied | Pre-Configured Variants |
| :--- | :--- | :--- | :--- |
| **A. Architecture Ablation** | `configs/architecture/` | Model inductive bias & multi-task heads | `baseline_mlp`, `gat_standalone`, `gat_met_production` (Baseline), `gat_met_decorrelation` |
| **B. Graph Construction** | `configs/graph_construction/` | Spatial proximity cutoff ($\Delta R$) | `dr_02` ($\Delta R=0.2$), `dr_04_baseline` ($\Delta R=0.4$), `dr_06` ($\Delta R=0.6$) |
| **C. MET Loss Weight** | `configs/met_weight/` | Physical vector conservation weight ($\lambda_{\text{MET}}$) | `lambda_0` ($0.0$), `lambda_005` ($0.05$), `lambda_010_baseline` ($0.10$), `lambda_020` ($0.20$) |
| **D. Mass Decorrelation** | `configs/mass_decorrelation/` | Adversarial invariant mass unlearning ($\lambda_{\text{adv}}$) | `without_decorrelation` ($\lambda_{\text{adv}}=0.0$), `with_decorrelation` ($\lambda_{\text{adv}}=0.05$) |

---

## 2. Configuration System

Configurations are self-contained, human-readable JSON files validated by [`src/experiment_framework.py`](file:///Users/jhum/Documents/Padhai%20stuff/college/mca/coding/anomaly-detector-gnn/src/experiment_framework.py).

### Core Fields
Every configuration explicitly defines:
- **Identity & Provenance:** `experiment_name`, `experiment_family`, `description`, `parent_reference`, `seed`.
- **Dataset Specification:** `background_path`, `signal_path`, `train_limit` (5000), `eval_limit` (500).
- **Training Hyperparameters:** `epochs`, `batch_size` (32), `learning_rate` (0.001), `optimizer` ("adam").
- **Model Topology:** `architecture_type`, `input_dim`, `hidden_dim`, `num_heads`.
- **Graph Construction:** `delta_r_threshold`, `log_pt` (True), `include_met_node` (True), `phi_met_mode` ("atan2").
- **Loss Objectives:** `loss_recon_type` ("mse"), `met_loss_weight`, `adv_loss_weight`, `grl_alpha` (1.0), `mass_norm_gev` (1000.0).
- **Anomaly Scoring & Thresholds:** `metric` ("reconstruction_mse"), `percentile_threshold` (90.0).
- **Target Output Directory:** `experiments/<family>/<experiment_name>`.

---

## 3. Experiment Output Contract

Every experiment run must be self-contained in its designated directory:

```
experiments/<family>/<experiment_name>/
├── config.json          # Full configuration + Git metadata (commit, branch, timestamp)
├── checkpoint.pt        # Trained PyTorch state dictionary (generated during training)
├── predictions.npz      # Saved event-level scores and predictions (generated during evaluation)
├── metrics.json         # Summary metrics: ROC AUC, rejections, separation ratio, significance
└── plots/               # Experiment-specific figures (e.g., ROC curve, bump hunt spectrum)
```

No placeholder or fabricated prediction/metric files are created; files are produced only upon execution.

---

## 4. Reproducibility Requirements

When an experiment is launched, the framework automatically records:
1. Exact Git commit SHA (`git rev-parse HEAD`).
2. Working tree cleanliness status (`is_dirty`).
3. Active Git branch.
4. UTC timestamp of execution.
5. Random seed applied to PyTorch, NumPy, and Python runtime.
6. Dataset paths referenced (without duplicating heavy data files).

---

## 5. Execution Workflow

### A. Validate All Configurations
Verify that all configuration files adhere to schema constraints:
```bash
python -m src.experiment_framework --verify-all
```

### B. Dry-Run an Experiment Setup
Verify output contract directory creation and data references without launching training:
```bash
python -m src.experiment_framework --config configs/architecture/gat_met_production.json --dry-run
```

### C. Single Experiment Evaluation
Evaluate a trained experiment directory once checkpoint and datasets are available:
```bash
python evaluation/evaluate_single.py --experiment experiments/architecture/gat_met_production
```

### D. Generate Family Comparison Table
Aggregate completed runs into a Markdown table under `comparisons/<family>/`:
```bash
python evaluation/compare_experiments.py --family architecture
```
