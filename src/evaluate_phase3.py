"""
Phase 3 controlled evaluation.

Evaluates the four architecture variants using the same held-out population
and reconstruction-MSE anomaly score.

Models:
  - baseline_mlp
  - gat_standalone
  - gat_met_production
  - gat_met_decorrelation

Evaluation population:
  - 500 held-out background events
  - 500 signal events

Background evaluation begins after the 5,000 background events used for
training. The exact same evaluation population is reused for every model.

No training is performed.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

from src.experiment_framework import ExperimentConfig
from src.run_lhco_gnn import (
    create_model,
    event_to_flat_features,
    event_to_graph,
    stream_filtered_events,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = [
    "baseline_mlp",
    "gat_standalone",
    "gat_met_production",
    "gat_met_decorrelation",
]


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration."""
    with path.open("r") as f:
        raw = json.load(f)

    return ExperimentConfig.from_dict(raw)


def load_events(
    data_path: Path,
    is_signal: bool,
    limit: int,
    skip: int = 0,
) -> List[np.ndarray]:
    """
    Load exactly `limit` filtered events after skipping `skip` events.

    `skip` is applied after truth filtering, so for background this allows
    the evaluation population to begin after the training population.
    """
    events = []
    skipped = 0

    for chunk in stream_filtered_events(
        str(data_path),
        is_signal=is_signal,
        chunk_size=1000,
    ):
        for event in chunk:
            if skipped < skip:
                skipped += 1
                continue

            events.append(np.asarray(event))

            if len(events) >= limit:
                return events

    raise RuntimeError(
        f"Requested {limit} events after skipping {skip}, but only found "
        f"{len(events)} available events in {data_path}"
    )


def load_evaluation_population(
    config: ExperimentConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """
    Load the common held-out evaluation population once.

    Background:
        skip the training population, then collect eval_limit events.

    Signal:
        collect eval_limit signal events. Signal is not used for training.
    """
    bg_path = REPO_ROOT / config.dataset.background_path
    signal_path = REPO_ROOT / config.dataset.signal_path

    eval_limit = config.dataset.eval_limit
    train_limit = config.dataset.train_limit

    print("\n" + "=" * 70)
    print("LOADING COMMON HELD-OUT EVALUATION POPULATION")
    print(f"Background file: {bg_path}")
    print(f"Signal file:     {signal_path}")
    print(f"Training BG population skipped: {train_limit}")
    print(f"Evaluation BG events:            {eval_limit}")
    print(f"Evaluation signal events:        {eval_limit}")

    background = load_events(
        bg_path,
        is_signal=False,
        limit=eval_limit,
        skip=train_limit,
    )

    signal = load_events(
        signal_path,
        is_signal=True,
        limit=eval_limit,
        skip=0,
    )

    population_metadata = {
        "background_file": str(bg_path.relative_to(REPO_ROOT)),
        "signal_file": str(signal_path.relative_to(REPO_ROOT)),
        "background_training_events_skipped": train_limit,
        "background_evaluation_events": len(background),
        "signal_evaluation_events": len(signal),
        "anomaly_score": "reconstruction_mse",
    }

    print(f"Loaded background: {len(background)}")
    print(f"Loaded signal:     {len(signal)}")
    print("=" * 70)

    return background, signal, population_metadata


def reconstruction_score_baseline(
    model: torch.nn.Module,
    event: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    """Reconstruction MSE for the baseline MLP."""
    x = event_to_flat_features(
        event,
        input_dim=config.model.input_dim,
        log_pt=config.graph_construction.log_pt,
    ).unsqueeze(0).to(device)

    recon_x = model(x)

    return float(F.mse_loss(recon_x, x, reduction="mean").item())


def reconstruction_score_gnn(
    model: torch.nn.Module,
    event: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    """Reconstruction MSE for either GAT architecture."""
    graph = event_to_graph(
        event,
        log_pt=config.graph_construction.log_pt,
        delta_r_threshold=config.graph_construction.delta_r_threshold,
        include_met_node=config.graph_construction.include_met_node,
        phi_met_mode=config.graph_construction.phi_met_mode,
    )

    batch = Batch.from_data_list([graph]).to(device)

    if config.model.architecture_type == "adversarial_gat":
        recon_x, _, _ = model(
            batch.x,
            batch.edge_index,
            batch.batch,
            alpha=1.0,
        )
    else:
        recon_x, _ = model(
            batch.x,
            batch.edge_index,
            batch.batch,
        )

    return float(F.mse_loss(recon_x, batch.x, reduction="mean").item())


@torch.no_grad()
def score_events(
    model: torch.nn.Module,
    config: ExperimentConfig,
    events: List[np.ndarray],
    device: torch.device,
) -> np.ndarray:
    """Score events using the common reconstruction-MSE anomaly score."""
    model.eval()

    scores = []

    for event in events:
        if config.model.architecture_type == "baseline_mlp":
            score = reconstruction_score_baseline(
                model,
                event,
                config,
                device,
            )
        else:
            score = reconstruction_score_gnn(
                model,
                event,
                config,
                device,
            )

        scores.append(score)

    return np.asarray(scores, dtype=np.float64)


def evaluate_model(
    experiment_name: str,
    device: torch.device,
    background: List[np.ndarray],
    signal: List[np.ndarray],
    population_metadata: Dict,
) -> Dict:
    """Evaluate one Phase 3 experiment on the common population."""
    exp_dir = REPO_ROOT / "experiments" / "architecture" / experiment_name
    config_path = exp_dir / "config.json"
    checkpoint_path = exp_dir / "checkpoint.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    config = load_config(config_path)

    print(f"\n{'=' * 70}")
    print(f"Evaluating: {experiment_name}")
    print(f"Architecture: {config.model.architecture_type}")
    print(f"Checkpoint: {checkpoint_path}")

    model = create_model(config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    bg_scores = score_events(
        model,
        config,
        background,
        device,
    )

    signal_scores = score_events(
        model,
        config,
        signal,
        device,
    )

    y_true = np.concatenate([
        np.zeros(len(bg_scores), dtype=np.int64),
        np.ones(len(signal_scores), dtype=np.int64),
    ])

    y_score = np.concatenate([
        bg_scores,
        signal_scores,
    ])

    auc = float(roc_auc_score(y_true, y_score))

    bg_mean = float(np.mean(bg_scores))
    bg_std = float(np.std(bg_scores))
    sig_mean = float(np.mean(signal_scores))
    sig_std = float(np.std(signal_scores))

    separation = (
        sig_mean / bg_mean
        if bg_mean != 0
        else float("inf")
    )

    significance = (
        (sig_mean - bg_mean) / bg_std
        if bg_std != 0
        else float("inf")
    )

    result = {
        "experiment": experiment_name,
        "architecture_type": config.model.architecture_type,
        "checkpoint": str(checkpoint_path.relative_to(REPO_ROOT)),
        "evaluation": population_metadata,
        "metrics": {
            "roc_auc": auc,
            "background_mean": bg_mean,
            "background_std": bg_std,
            "signal_mean": sig_mean,
            "signal_std": sig_std,
            "separation_ratio": separation,
            "gaussian_significance": significance,
        },
    }

    output_dir = (
        REPO_ROOT
        / "experiments"
        / "evaluation"
        / "phase3"
        / experiment_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_dir / "scores.npz",
        background_scores=bg_scores,
        signal_scores=signal_scores,
    )

    with (output_dir / "results.json").open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Background:   {len(bg_scores)}")
    print(f"Signal:       {len(signal_scores)}")
    print(f"ROC-AUC:      {auc:.6f}")
    print(f"BG mean:      {bg_mean:.6f}")
    print(f"BG std:       {bg_std:.6f}")
    print(f"SIG mean:     {sig_mean:.6f}")
    print(f"SIG std:      {sig_std:.6f}")
    print(f"Separation:   {separation:.6f}")
    print(f"Significance: {significance:.6f}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--experiment",
        default=None,
        choices=EXPERIMENTS,
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.experiment:
        experiments = [args.experiment]
    else:
        experiments = EXPERIMENTS

    # Use the first experiment's configuration to define the common
    # evaluation population. All four Phase 3 configurations use the
    # same dataset and evaluation limits by design.
    reference_config_path = (
        REPO_ROOT
        / "experiments"
        / "architecture"
        / EXPERIMENTS[0]
        / "config.json"
    )

    if not reference_config_path.exists():
        raise FileNotFoundError(
            f"Missing reference config: {reference_config_path}"
        )

    reference_config = load_config(reference_config_path)

    background, signal, population_metadata = load_evaluation_population(
        reference_config
    )

    all_results = []

    for experiment_name in experiments:
        result = evaluate_model(
            experiment_name,
            device,
            background,
            signal,
            population_metadata,
        )
        all_results.append(result)

    output_root = (
        REPO_ROOT
        / "experiments"
        / "evaluation"
        / "phase3"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "evaluation": population_metadata,
        "device": str(device),
        "experiments": experiments,
        "results": all_results,
    }

    with (output_root / "phase3_results.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("PHASE 3 EVALUATION COMPLETE")
    print(f"Results: {output_root / 'phase3_results.json'}")


if __name__ == "__main__":
    main()
