"""
experiment_framework.py — Controlled Experiment Framework for Anomaly Detection GNN
===================================================================================
Provides configuration management, metadata recording, output contract enforcement,
and dry-run initialization for reproducible high-energy physics anomaly detection experiments.

Experiment Families Supported:
  1. Architecture ablation (architecture/)
  2. Graph-construction sensitivity (graph_construction/)
  3. MET-objective weighting (met_weight/)
  4. Adversarial mass-decorrelation (mass_decorrelation/)

Design Mandates:
  - Zero heavy external dependencies (uses standard library json, subprocess, dataclasses).
  - Explicit provenance tracking (records Git commit SHA, branch, seed, and timestamps).
  - Preserves existing model architectures, checkpoint formats, and pipeline code.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

VALID_FAMILIES = (
    "architecture",
    "graph_construction",
    "met_weight",
    "mass_decorrelation",
)


@dataclass
class DatasetConfig:
    background_path: str = "data/events_LHCO2020_backgroundMC_Pythia.h5"
    signal_path: str = "data/W_prime_signal.h5"
    train_limit: int = 5000
    eval_limit: int = 500


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    seed: int = 42


@dataclass
class ModelConfig:
    architecture_type: str = "gat_autoencoder"  # gat_autoencoder, baseline_mlp, adversarial_gat
    input_dim: int = 3
    hidden_dim: int = 32
    num_heads: int = 4


@dataclass
class GraphConstructionConfig:
    delta_r_threshold: float = 0.4
    log_pt: bool = True
    include_met_node: bool = True
    phi_met_mode: str = "atan2"  # atan2(MET_y, MET_x) or zero


@dataclass
class ObjectivesConfig:
    loss_recon_type: str = "mse"
    met_loss_weight: float = 0.1
    adv_loss_weight: float = 0.0
    grl_alpha: float = 1.0
    mass_norm_gev: float = 1000.0


@dataclass
class AnomalyScoringConfig:
    metric: str = "reconstruction_mse"
    percentile_threshold: float = 90.0


@dataclass
class ExperimentConfig:
    experiment_name: str
    experiment_family: str
    description: str = ""
    parent_reference: str = "pre-experiments-684c99e"
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    graph_construction: GraphConstructionConfig = field(default_factory=GraphConstructionConfig)
    objectives: ObjectivesConfig = field(default_factory=ObjectivesConfig)
    anomaly_scoring: AnomalyScoringConfig = field(default_factory=AnomalyScoringConfig)
    output_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        data = dict(data)
        if "dataset" in data and isinstance(data["dataset"], dict):
            data["dataset"] = DatasetConfig(**data["dataset"])
        if "training" in data and isinstance(data["training"], dict):
            data["training"] = TrainingConfig(**data["training"])
        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
        if "graph_construction" in data and isinstance(data["graph_construction"], dict):
            data["graph_construction"] = GraphConstructionConfig(**data["graph_construction"])
        if "objectives" in data and isinstance(data["objectives"], dict):
            data["objectives"] = ObjectivesConfig(**data["objectives"])
        if "anomaly_scoring" in data and isinstance(data["anomaly_scoring"], dict):
            data["anomaly_scoring"] = AnomalyScoringConfig(**data["anomaly_scoring"])
        return cls(**data)


def get_git_metadata() -> Dict[str, Any]:
    """Retrieves current Git commit, branch, and status locally without network calls."""
    metadata: Dict[str, Any] = {
        "commit": "unknown",
        "branch": "unknown",
        "is_dirty": False,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        metadata["commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        metadata["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        metadata["is_dirty"] = len(status) > 0
    except Exception:
        pass

    return metadata


def validate_config(config: ExperimentConfig) -> List[str]:
    """Validates an ExperimentConfig against schema and domain constraints."""
    errors: List[str] = []

    if not config.experiment_name:
        errors.append("experiment_name must not be empty.")

    if config.experiment_family not in VALID_FAMILIES:
        errors.append(
            f"Invalid experiment_family: '{config.experiment_family}'. "
            f"Must be one of: {VALID_FAMILIES}"
        )

    if config.graph_construction.delta_r_threshold <= 0:
        errors.append("delta_r_threshold must be strictly positive.")

    if config.training.epochs <= 0:
        errors.append("epochs must be > 0.")

    if config.training.batch_size <= 0:
        errors.append("batch_size must be > 0.")

    if config.training.learning_rate <= 0:
        errors.append("learning_rate must be > 0.")

    if config.objectives.met_loss_weight < 0:
        errors.append("met_loss_weight cannot be negative.")

    if config.objectives.adv_loss_weight < 0:
        errors.append("adv_loss_weight cannot be negative.")

    return errors


def load_config(path: str | Path) -> ExperimentConfig:
    """Loads and parses an ExperimentConfig from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Saves an ExperimentConfig to a formatted JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def init_experiment_dir(config: ExperimentConfig, repo_root: Optional[Path] = None) -> Path:
    """
    Initializes a reproducible experiment directory adhering to the output contract:
      experiments/<family>/<experiment_name>/
      ├── config.json
      └── plots/
    """
    if repo_root is None:
        repo_root = Path.cwd()

    if not config.output_dir:
        config.output_dir = str(
            Path("experiments") / config.experiment_family / config.experiment_name
        )

    target_dir = repo_root / config.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Attach current Git provenance metadata
    config.metadata.update(get_git_metadata())

    # Write frozen configuration file
    config_dest = target_dir / "config.json"
    save_config(config, config_dest)

    return target_dir


def check_dataset_availability(config: ExperimentConfig, repo_root: Optional[Path] = None) -> Dict[str, bool]:
    """Checks whether the datasets referenced in the configuration exist on disk."""
    if repo_root is None:
        repo_root = Path.cwd()

    bg_path = repo_root / config.dataset.background_path
    sig_path = repo_root / config.dataset.signal_path

    return {
        "background_available": bg_path.exists(),
        "signal_available": sig_path.exists(),
        "background_path": str(bg_path),
        "signal_path": str(sig_path),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Controlled Experiment Framework for Anomaly Detection GNN"
    )
    parser.add_argument(
        "--config", type=str, help="Path to experiment configuration JSON"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, check datasets, and verify output contract without training",
    )
    parser.add_argument(
        "--verify-all",
        action="store_true",
        help="Validate all JSON configs under configs/ directory",
    )

    args = parser.parse_args()

    if args.verify_all:
        configs_dir = Path("configs")
        if not configs_dir.exists():
            print("No configs/ directory found.")
            sys.exit(1)

        all_configs = list(configs_dir.glob("*/*.json"))
        print(f"Found {len(all_configs)} configuration files under {configs_dir}/:")
        has_errors = False
        for cfg_path in all_configs:
            try:
                cfg = load_config(cfg_path)
                errors = validate_config(cfg)
                if errors:
                    print(f"  [FAIL] {cfg_path}: {', '.join(errors)}")
                    has_errors = True
                else:
                    ds = check_dataset_availability(cfg)
                    ds_str = (
                        "data present"
                        if ds["background_available"] and ds["signal_available"]
                        else "data referenced (unfetched)"
                    )
                    print(f"  [OK]   {cfg_path.relative_to(configs_dir)} ({cfg.experiment_family}) — {ds_str}")
            except Exception as e:
                print(f"  [ERR]  {cfg_path}: {e}")
                has_errors = True

        sys.exit(1 if has_errors else 0)

    if args.config:
        cfg = load_config(args.config)
        errors = validate_config(cfg)
        if errors:
            print("Configuration errors detected:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)

        print(f"Loaded config: {cfg.experiment_name} (Family: {cfg.experiment_family})")
        print(f"Description:   {cfg.description}")
        print(f"Reference:     {cfg.parent_reference}")

        ds = check_dataset_availability(cfg)
        print("Dataset availability:")
        print(f"  Background: {ds['background_available']} ({ds['background_path']})")
        print(f"  Signal:     {ds['signal_available']} ({ds['signal_path']})")

        if args.dry_run:
            exp_dir = init_experiment_dir(cfg)
            print(f"Dry-run initialized experiment directory: {exp_dir}")
            print("Output contract verified:")
            print(f"  - {exp_dir / 'config.json'} [Created]")
            print(f"  - {exp_dir / 'plots'} [Created]")
            print("  - checkpoint.pt [Pending execution]")
            print("  - predictions.npz [Pending execution]")
            print("  - metrics.json [Pending execution]")
            print("Dry-run complete. No training was performed.")
            sys.exit(0)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
