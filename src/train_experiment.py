"""
train_experiment.py — Unified Config-Driven Training Engine
============================================================
Executes controlled training runs strictly driven by experiment configurations.

Supports all 4 controlled architecture variants:
  1. baseline_mlp: Tabular MLP autoencoder (64 -> 32 -> 8 -> 32 -> 64)
  2. gat_standalone: Graph Attention Network without MET prediction (lambda_MET=0)
  3. gat_met_production: Multi-task GAT autoencoder with MET prediction (lambda_MET=0.1)
  4. gat_met_decorrelation: GAT autoencoder with MET prediction + GRL mass adversary

Constraints:
  - Strictly writes outputs to experiments/<family>/<experiment_name>/
  - Never overwrites baseline models in models/*.pt
  - Config-driven loss weights (lambda_MET, lambda_adv, grl_alpha)
  - Full provenance tracking (git commit SHA, seed, device, environment)
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_framework import (
    load_config,
    save_config,
    check_dataset_availability,
    get_git_metadata,
    ExperimentConfig,
)
from src.run_lhco_gnn import (
    create_model,
    event_to_graph,
    event_to_flat_features,
    extract_particles,
    stream_filtered_events,
    get_device,
    BaselineMLP,
    NodeGNN,
)
from src.adversarial_unlearning_ae import (
    AdversarialNodeGNN,
    compute_invariant_mass,
)


def set_seed(seed: int) -> None:
    """Sets random seeds for python, numpy, and torch across all available backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass


def load_training_dataset(
    config: ExperimentConfig,
    limit: Optional[int] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[Any, str]:
    """
    Loads training data based on configuration specifications.
    
    Returns:
      (data_list_or_dataset, dataset_type) where dataset_type is 'graph' or 'flat'.
    """
    root = repo_root or REPO_ROOT
    bg_path = root / config.dataset.background_path
    train_limit = limit if limit is not None else config.dataset.train_limit
    arch_type = config.model.architecture_type

    if not bg_path.exists():
        raise FileNotFoundError(f"Background dataset not found at: {bg_path}")

    # Flat tabular representation for baseline_mlp
    if arch_type == "baseline_mlp":
        flat_tensors = []
        count = 0
        for chunk in stream_filtered_events(str(bg_path), is_signal=False, chunk_size=1000):
            for i in range(len(chunk)):
                if count >= train_limit:
                    break
                event = chunk[i]
                t = event_to_flat_features(
                    event,
                    input_dim=config.model.input_dim,
                    log_pt=config.graph_construction.log_pt,
                )
                flat_tensors.append(t)
                count += 1
            if count >= train_limit:
                break

        x_stacked = torch.stack(flat_tensors)
        return TensorDataset(x_stacked), "flat"

    # Graph representation for GAT models
    graph_list = []
    count = 0
    delta_r = config.graph_construction.delta_r_threshold
    log_pt = config.graph_construction.log_pt
    include_met = config.graph_construction.include_met_node
    phi_mode = config.graph_construction.phi_met_mode
    is_adversarial = (arch_type == "adversarial_gat") or (config.objectives.adv_loss_weight > 0)
    mass_norm = config.objectives.mass_norm_gev

    for chunk in stream_filtered_events(str(bg_path), is_signal=False, chunk_size=1000):
        for i in range(len(chunk)):
            if count >= train_limit:
                break
            event = chunk[i]
            graph = event_to_graph(
                event,
                log_pt=log_pt,
                delta_r_threshold=delta_r,
                include_met_node=include_met,
                phi_met_mode=phi_mode,
            )

            if is_adversarial:
                particles = extract_particles(event)
                mass = compute_invariant_mass(particles)
                graph.mass = torch.tensor([mass / mass_norm], dtype=torch.float)

            graph_list.append(graph)
            count += 1
        if count >= train_limit:
            break

    return graph_list, "graph"


def train_single_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ExperimentConfig,
    dataset_type: str,
) -> Dict[str, float]:
    """Runs a single epoch of training respecting config loss weights."""
    model.train()
    total_loss = 0.0
    recon_loss_sum = 0.0
    met_loss_sum = 0.0
    adv_loss_sum = 0.0
    num_batches = 0

    lambda_met = config.objectives.met_loss_weight
    lambda_adv = config.objectives.adv_loss_weight
    grl_alpha = config.objectives.grl_alpha

    for batch in loader:
        optimizer.zero_grad()

        if dataset_type == "flat":
            # Flat tabular autoencoder
            batch_x = batch[0].to(device)
            recon_x = model(batch_x)
            loss_recon = F.mse_loss(recon_x, batch_x)
            loss = loss_recon
            loss_met = torch.tensor(0.0)
            loss_adv = torch.tensor(0.0)
        else:
            # Graph models
            batch = batch.to(device)
            if isinstance(model, AdversarialNodeGNN):
                recon_x, pred_met, pred_mass = model(
                    batch.x, batch.edge_index, batch.batch, alpha=grl_alpha
                )
                loss_recon = F.mse_loss(recon_x, batch.x)
                loss = loss_recon

                if lambda_met > 0.0:
                    loss_met = F.mse_loss(pred_met, batch.y.view(-1, 2))
                    loss = loss + lambda_met * loss_met
                else:
                    loss_met = torch.tensor(0.0)

                if lambda_adv > 0.0 and hasattr(batch, "mass"):
                    loss_adv = F.mse_loss(pred_mass.view(-1, 1), batch.mass.view(-1, 1))
                    loss = loss + lambda_adv * loss_adv
                else:
                    loss_adv = torch.tensor(0.0)

            elif isinstance(model, NodeGNN):
                recon_x, pred_met = model(batch.x, batch.edge_index, batch.batch)
                loss_recon = F.mse_loss(recon_x, batch.x)
                loss = loss_recon

                if lambda_met > 0.0:
                    loss_met = F.mse_loss(pred_met, batch.y.view(-1, 2))
                    loss = loss + lambda_met * loss_met
                else:
                    loss_met = torch.tensor(0.0)

                loss_adv = torch.tensor(0.0)
            else:
                raise ValueError(f"Unknown model class: {type(model)}")

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_loss_sum += loss_recon.item()
        met_loss_sum += loss_met.item()
        adv_loss_sum += loss_adv.item()
        num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_recon": recon_loss_sum / max(num_batches, 1),
        "loss_met": met_loss_sum / max(num_batches, 1),
        "loss_adv": adv_loss_sum / max(num_batches, 1),
    }


def run_training_experiment(
    config_path: Path,
    dry_run: bool = False,
    override_epochs: Optional[int] = None,
    override_limit: Optional[int] = None,
    override_device: Optional[str] = None,
    override_output_dir: Optional[str] = None,
    is_smoke_test: bool = False,
) -> int:
    """Executes or dry-runs an experiment training pipeline."""
    t_start = time.time()
    config = load_config(config_path)

    # Resolve output directory
    if override_output_dir:
        output_dir = Path(override_output_dir)
    elif config.output_dir:
        output_dir = REPO_ROOT / config.output_dir
    else:
        output_dir = REPO_ROOT / "experiments" / config.experiment_family / config.experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine execution device
    if override_device:
        device = torch.device(override_device)
    else:
        device = get_device()

    # Determine epochs and event limit
    epochs = override_epochs if override_epochs is not None else config.training.epochs
    train_limit = override_limit if override_limit is not None else config.dataset.train_limit

    print("=" * 70)
    print(f"EXPERIMENT RUNNER: {config.experiment_name} ({config.experiment_family})")
    print(f"Description:       {config.description}")
    print(f"Architecture:      {config.model.architecture_type}")
    print(f"Device:            {device}")
    print(f"Output Directory:  {output_dir}")
    print(f"Seed:              {config.seed}")
    print(f"Epochs:            {epochs}")
    print(f"Train limit:       {train_limit} events")
    print(f"Loss weights:      recon=1.0, MET={config.objectives.met_loss_weight}, adv={config.objectives.adv_loss_weight}")
    print("=" * 70)

    # Check dataset availability
    ds_status = check_dataset_availability(config, repo_root=REPO_ROOT)
    print(f"Dataset status:    Background={ds_status['background_available']}, Signal={ds_status['signal_available']}")
    if not ds_status["background_available"]:
        print(f"[ERROR] Required background dataset not found: {ds_status['background_path']}")
        return 1

    # Instantiate model
    model = create_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:  total={total_params:,}, trainable={trainable_params:,}")

    # Dry-run validation check
    if dry_run:
        print("[STATUS] Running in DRY-RUN mode. Validating configuration and paths...")
        # Save validated configuration with dry-run provenance into target directory
        dry_run_config_path = output_dir / "config.json"
        config.metadata.update(get_git_metadata())
        config.metadata["execution_device"] = str(device)
        config.metadata["dry_run"] = True
        save_config(config, dry_run_config_path)
        print(f"[DRY RUN SUCCESSFUL] Model initialized, config verified, written to {dry_run_config_path}")
        return 0

    # Set seeds for reproducible execution
    set_seed(config.seed)

    # Load dataset
    print(f"[INFO] Loading {train_limit} background events...")
    dataset, dataset_type = load_training_dataset(config, limit=train_limit, repo_root=REPO_ROOT)
    print(f"[INFO] Loaded {len(dataset)} items (type: {dataset_type}).")

    # Create data loader
    batch_size = config.training.batch_size
    if dataset_type == "flat":
        loader = TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize optimizer
    lr = config.training.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    print(f"[INFO] Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        metrics = train_single_epoch(model, loader, optimizer, device, config, dataset_type)
        ep_t1 = time.time()
        metrics["epoch"] = epoch
        metrics["epoch_time_sec"] = round(ep_t1 - ep_t0, 3)
        history.append(metrics)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Loss: {metrics['loss']:.5f} | "
            f"Recon: {metrics['loss_recon']:.5f} | "
            f"MET: {metrics['loss_met']:.5f} | "
            f"Adv: {metrics['loss_adv']:.5f} | "
            f"Time: {metrics['epoch_time_sec']}s"
        )

    # Output artifact persistence strictly to output_dir
    checkpoint_path = output_dir / "checkpoint.pt"
    saved_config_path = output_dir / "config.json"
    log_path = output_dir / "training_log.json"

    # Save state dict
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Checkpoint saved: {checkpoint_path}")

    # Save finalized config with provenance metadata
    config.metadata.update(get_git_metadata())
    config.metadata["execution_device"] = str(device)
    config.metadata["dry_run"] = False
    save_config(config, saved_config_path)
    print(f"[INFO] Configuration saved: {saved_config_path}")

    # Save training log
    log_data = {
        "experiment_name": config.experiment_name,
        "experiment_family": config.experiment_family,
        "is_smoke_test": is_smoke_test,
        "epochs_trained": epochs,
        "events_used": train_limit,
        "total_time_sec": round(time.time() - t_start, 2),
        "history": history,
        "final_loss": history[-1] if history else {},
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"[INFO] Training log saved: {log_path}")

    print(f"[SUCCESS] Training completed in {time.time() - t_start:.2f}s.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified config-driven training engine for controlled experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment JSON config file (e.g., configs/architecture/gat_met_production.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, environment, and datasets without executing training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs (e.g. for smoke testing)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional override for number of training events (e.g. for smoke testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for compute device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for output directory",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Flag to mark execution as a pipeline smoke test",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    sys.exit(
        run_training_experiment(
            config_path=config_path,
            dry_run=args.dry_run,
            override_epochs=args.epochs,
            override_limit=args.limit,
            override_device=args.device,
            override_output_dir=args.output_dir,
            is_smoke_test=args.smoke_test,
        )
    )


if __name__ == "__main__":
    main()
