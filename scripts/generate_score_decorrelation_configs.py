"""
generate_score_decorrelation_configs.py
=======================================
Generates JSON configuration files for the Score-Level Mass Decorrelation experiments:
- Score-Level Distance Correlation (DisCo): Seeds 42, 123, 999
- Score-Level Gradient Reversal Layer (Score-GRL): Seeds 42, 123, 999
- Parameter variations (Heavy DisCo, Weighted Score DisCo, Baseline Control)
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "score_decorrelation"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = {
    "experiment_family": "score_decorrelation",
    "parent_reference": "controlled_structure_f5baacd",
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
    # 1. Score-Level DisCo (Seeds 42, 123, 999)
    {
        "name": "gat_score_disco_s42",
        "seed": 42,
        "desc": "Score-Level DisCo Decorrelation on pT anomaly score (lambda_adv = 1.0, seed 42)",
        "decorr_type": "disco",
        "score_type": "pt_only",
        "lambda_adv": 1.0,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    {
        "name": "gat_score_disco_s123",
        "seed": 123,
        "desc": "Score-Level DisCo Decorrelation on pT anomaly score (lambda_adv = 1.0, seed 123)",
        "decorr_type": "disco",
        "score_type": "pt_only",
        "lambda_adv": 1.0,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    {
        "name": "gat_score_disco_s999",
        "seed": 999,
        "desc": "Score-Level DisCo Decorrelation on pT anomaly score (lambda_adv = 1.0, seed 999)",
        "decorr_type": "disco",
        "score_type": "pt_only",
        "lambda_adv": 1.0,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    # 2. Score-Level GRL Adversary (Seeds 42, 123, 999)
    {
        "name": "gat_score_grl_s42",
        "seed": 42,
        "desc": "Score-Level GRL Adversary predicting dijet mass from pT score (lambda_adv = 0.05, seed 42)",
        "decorr_type": "grl",
        "score_type": "pt_only",
        "lambda_adv": 0.05,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    {
        "name": "gat_score_grl_s123",
        "seed": 123,
        "desc": "Score-Level GRL Adversary predicting dijet mass from pT score (lambda_adv = 0.05, seed 123)",
        "decorr_type": "grl",
        "score_type": "pt_only",
        "lambda_adv": 0.05,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    {
        "name": "gat_score_grl_s999",
        "seed": 999,
        "desc": "Score-Level GRL Adversary predicting dijet mass from pT score (lambda_adv = 0.05, seed 999)",
        "decorr_type": "grl",
        "score_type": "pt_only",
        "lambda_adv": 0.05,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    # 3. DisCo Heavy Regularization
    {
        "name": "gat_score_disco_heavy_s42",
        "seed": 42,
        "desc": "Score-Level DisCo with Heavy Regularization (lambda_adv = 2.5, seed 42)",
        "decorr_type": "disco",
        "score_type": "pt_only",
        "lambda_adv": 2.5,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    # 4. DisCo on Weighted Score (0.8 pT + 0.2 Ang)
    {
        "name": "gat_score_disco_weighted_s42",
        "seed": 42,
        "desc": "Score-Level DisCo on Weighted Anomaly Score (0.8 pT + 0.2 Ang, lambda_adv = 1.0, seed 42)",
        "decorr_type": "disco",
        "score_type": "weighted",
        "lambda_adv": 1.0,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
    },
    # 5. Baseline Reference Control (Zero Adversary)
    {
        "name": "gat_baseline_ref_s42",
        "seed": 42,
        "desc": "Un-Decorrelated Control Reference (lambda_adv = 0.0, seed 42)",
        "decorr_type": "disco",
        "score_type": "pt_only",
        "lambda_adv": 0.0,
        "lambda_met": 0.1,
        "grl_alpha": 1.0,
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
            "architecture_type": "score_decorrelated_gat",
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
            "grl_alpha": exp["grl_alpha"],
            "mass_norm_gev": 1000.0,
        }
        cfg["output_dir"] = f"experiments/score_decorrelation/{exp['name']}"
        cfg["metadata"] = {
            "model_class": f"score_{exp['decorr_type']}_{exp['score_type']}",
            "decorrelation_mechanism": "score_level",
        }

        out_path = CONFIG_DIR / f"{exp['name']}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  Created: {out_path.name}")

    print("[SUCCESS] All score decorrelation configurations successfully generated!")

if __name__ == "__main__":
    main()
