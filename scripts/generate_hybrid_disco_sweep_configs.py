"""
generate_hybrid_disco_sweep_configs.py
======================================
Generates the 15 controlled experiment configurations for the PointNet-GAT
Hybrid DisCo Loss Optimization sweep:
  - H0: lambda_disco = 0.00 (Seeds 42, 123, 999)
  - H1: lambda_disco = 0.10 (Seeds 42, 123, 999)
  - H2: lambda_disco = 0.25 (Seeds 42, 123, 999)
  - H3: lambda_disco = 0.50 (Seeds 42, 123, 999)
  - H4: lambda_disco = 1.00 (Seeds 42, 123, 999)
"""

import json
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs" / "hybrid_pointnet_gat"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
EXP_DIR = REPO_ROOT / "experiments" / "hybrid_pointnet_gat"

SWEEPS = [
    ("h0", 0.00, "hybrid_baseline"),
    ("h1", 0.10, None),
    ("h2", 0.25, None),
    ("h3", 0.50, None),
    ("h4", 1.00, "hybrid_disco_weighted"),
]

SEEDS = [42, 123, 999]


def main():
    generated_configs = []
    
    for tag, lam, source_prefix in SWEEPS:
        lam_str = f"{int(lam*100):03d}"
        for seed in SEEDS:
            exp_name = f"hybrid_{tag}_lambda{lam_str}_s{seed}"
            is_disco = (lam > 0.0)
            
            cfg = {
                "experiment_family": "hybrid_pointnet_gat",
                "parent_reference": "controlled_structure_be84f02",
                "dataset": {
                    "background_path": "data/events_LHCO2020_backgroundMC_Pythia.h5",
                    "signal_path": "data/W_prime_signal.h5",
                    "train_limit": 5000,
                    "eval_limit": 500,
                },
                "training": {
                    "epochs": 20,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "seed": seed,
                },
                "graph_construction": {
                    "delta_r_threshold": 0.4,
                    "log_pt": True,
                    "include_met_node": True,
                    "phi_met_mode": "atan2",
                },
                "anomaly_scoring": {
                    "metric": "reconstruction_mse",
                    "percentile_threshold": 90.0,
                },
                "experiment_name": exp_name,
                "description": f"Hybrid PointNet-GAT DisCo Sweep {tag.upper()} (lambda_adv = {lam:.2f}, seed {seed})",
                "seed": seed,
                "model": {
                    "architecture_type": "score_decorrelated_hybrid" if is_disco else "pointnet_gat_hybrid",
                    "input_dim": 3,
                    "hidden_dim": 32,
                    "num_heads": 4,
                    "decorrelation_type": "disco" if is_disco else "none",
                    "score_type": "weighted",
                },
                "objectives": {
                    "loss_recon_type": "mse",
                    "met_loss_weight": 0.1,
                    "adv_loss_weight": lam,
                    "grl_alpha": 1.0,
                    "mass_norm_gev": 1000.0,
                },
                "output_dir": f"experiments/hybrid_pointnet_gat/{exp_name}",
                "metadata": {
                    "model_class": "pointnet_gat_hybrid",
                    "decorrelation_mechanism": "score_level" if is_disco else "none",
                    "lambda_disco": lam,
                    "sweep_tag": tag,
                },
            }
            
            cfg_file = CONFIG_DIR / f"{exp_name}.json"
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            generated_configs.append(cfg_file)
            print(f"[CREATED CONFIG] {cfg_file.name}")
            
            # If checkpoint already exists from previous runs (H0 or H4), populate directory
            target_exp_dir = EXP_DIR / exp_name
            if source_prefix:
                src_dir = EXP_DIR / f"{source_prefix}_s{seed}"
                src_ckpt = src_dir / "checkpoint.pt"
                src_log = src_dir / "training_log.json"
                if src_ckpt.exists():
                    target_exp_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_ckpt, target_exp_dir / "checkpoint.pt")
                    if src_log.exists():
                        shutil.copy2(src_log, target_exp_dir / "training_log.json")
                    with open(target_exp_dir / "config.json", "w", encoding="utf-8") as f:
                        json.dump(cfg, f, indent=2)
                    print(f"  -> Linked existing checkpoint from {src_dir.name} ({src_ckpt.stat().st_size / 1024:.1f} KB)")

    print(f"\n[DONE] Generated {len(generated_configs)} experiment configurations in {CONFIG_DIR}")


if __name__ == "__main__":
    main()
