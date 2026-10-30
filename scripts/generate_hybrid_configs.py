"""
generate_hybrid_configs.py
==========================
Generates JSON configuration files for the Hybrid PointNet-GAT experiments:
- Baseline Hybrid PointNet-GAT (lambda_adv = 0): Seeds 42, 123, 999
- Score-Level DisCo Hybrid PointNet-GAT (lambda_adv = 1.0): Seeds 42, 123, 999
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "hybrid_pointnet_gat"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = {
    "experiment_family": "hybrid_pointnet_gat",
    "parent_reference": "controlled_structure_be84f02",
    "dataset": {
        "background_path": "data/events_LHCO2020_backgroundMC_Pythia.h5",
        "signal_path": "data/W_prime_signal.h5",
        "train_limit": 5000,
        "eval_limit": 500
    },
    "training": {
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam"
    },
    "graph_construction": {
        "delta_r_threshold": 0.4,
        "log_pt": True,
        "include_met_node": True,
        "phi_met_mode": "atan2"
    },
    "anomaly_scoring": {
        "metric": "reconstruction_mse",
        "percentile_threshold": 90.0
    }
}

EXPERIMENT_MATRIX = [
    # 1. Baseline Standalone Hybrid (Seeds 42, 123, 999)
    {
        "name": "hybrid_baseline_s42",
        "seed": 42,
        "desc": "Hybrid PointNet-GAT Autoencoder Baseline (lambda_adv = 0.0, seed 42)",
        "arch_type": "pointnet_gat_hybrid",
        "decorr_type": "none",
        "score_type": "weighted",
        "lambda_adv": 0.0,
        "lambda_met": 0.1,
    },
    {
        "name": "hybrid_baseline_s123",
        "seed": 123,
        "desc": "Hybrid PointNet-GAT Autoencoder Baseline (lambda_adv = 0.0, seed 123)",
        "arch_type": "pointnet_gat_hybrid",
        "decorr_type": "none",
        "score_type": "weighted",
        "lambda_adv": 0.0,
        "lambda_met": 0.1,
    },
    {
        "name": "hybrid_baseline_s999",
        "seed": 999,
        "desc": "Hybrid PointNet-GAT Autoencoder Baseline (lambda_adv = 0.0, seed 999)",
        "arch_type": "pointnet_gat_hybrid",
        "decorr_type": "none",
        "score_type": "weighted",
        "lambda_adv": 0.0,
        "lambda_met": 0.1,
    },
    # 2. Score-Level DisCo Weighted Hybrid (Seeds 42, 123, 999)
    {
        "name": "hybrid_disco_weighted_s42",
        "seed": 42,
        "desc": "Hybrid PointNet-GAT with Score-Level DisCo on Weighted Score (lambda_adv = 1.0, seed 42)",
        "arch_type": "score_decorrelated_hybrid",
        "decorr_type": "disco",
        "score_type": "weighted",
        "lambda_adv": 1.0,
        "lambda_met": 0.1,
    },
    {
        "name": "hybrid_disco_weighted_s123",
        "seed": 123,
        "desc": "Hybrid PointNet-GAT with Score-Level DisCo on Weighted Score (lambda_adv = 1.0, seed 123)",
        "arch_type": "score_decorrelated_hybrid",
        "decorr_type": "disco",
        "score_type": "weighted",
        "lambda_adv": 1.0,
        "lambda_met": 0.1,
    },
    {
        "name": "hybrid_disco_weighted_s999",
        "seed": 999,
        "desc": "Hybrid PointNet-GAT with Score-Level DisCo on Weighted Score (lambda_adv = 1.0, seed 999)",
        "arch_type": "score_decorrelated_hybrid",
        "decorr_type": "disco",
        "score_type": "weighted",
        "lambda_adv": 1.0,
        "lambda_met": 0.1,
    },
]

def main():
    print(f"[INFO] Generating {len(EXPERIMENT_MATRIX)} configs in {CONFIG_DIR}...")
    for exp in EXPERIMENT_MATRIX:
        cfg = dict(BASE_CONFIG)
        cfg["experiment_name"] = exp["name"]
        cfg["description"] = exp["desc"]
        cfg["seed"] = exp["seed"]
        cfg["training"] = dict(BASE_CONFIG["training"])
        cfg["training"]["seed"] = exp["seed"]
        cfg["model"] = {
            "architecture_type": exp["arch_type"],
            "input_dim": 3,
            "hidden_dim": 32,
            "num_heads": 4,
            "decorrelation_type": exp["decorr_type"],
            "score_type": exp["score_type"],
        }
        cfg["objectives"] = {
            "loss_recon_type": "mse",
            "met_loss_weight": exp["lambda_met"],
            "adv_loss_weight": exp["lambda_adv"],
            "grl_alpha": 1.0,
            "mass_norm_gev": 1000.0,
        }
        cfg["output_dir"] = f"experiments/hybrid_pointnet_gat/{exp['name']}"
        cfg["metadata"] = {
            "model_class": "pointnet_gat_hybrid",
            "decorrelation_mechanism": "score_level" if exp["lambda_adv"] > 0 else "none",
        }

        out_path = CONFIG_DIR / f"{exp['name']}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Created: {out_path.name}")

    print("[SUCCESS] All Hybrid PointNet-GAT configurations successfully generated!")

if __name__ == "__main__":
    main()
