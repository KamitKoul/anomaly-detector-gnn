"""
experiments/scaling/experiment_a_fixed_epochs.py — Experiment A: Fixed Epochs Runner
=====================================================================================
Research Question:
  "What happens when the model receives more unique background events while
   the number of passes over each event remains fixed?"

Configuration:
  - Dataset tiers: D0 (5k), D1 (25k), D2 (50k), D3 (100k), D4 (250k), D5 (500k), D6 (900k)
  - Epochs: 20
  - Batch size: 32 (drop_last=True)
  - Optimizer: Adam (lr=0.001, weight_decay=0.0)
  - Gradient clipping: 50.0
  - Expected updates: U(D) = 20 * floor(D / 32)
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader as PyGDataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.scaling_engine import (
    get_git_commit,
    get_manifest_sha,
    set_deterministic_seed,
    create_scaling_model,
    ScalingCheckpointManager,
    execute_train_step,
    evaluate_held_out_validation,
)
from src.run_lhco_gnn import get_device
from src.streaming_data_loader import ChunkedLHCODataset, create_streaming_dataloader
from src.data.scaling_split import TIER_SIZES


def run_experiment_a_tier(
    tier: str,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    device_override: Optional[str] = None,
    output_root: Optional[Path] = None,
    synthetic_mode: bool = False,
    synthetic_events: int = 500,
    bg_file: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes Experiment A for a single dataset tier and random seed."""
    assert tier in TIER_SIZES, f"Unknown tier: {tier}. Must be one of {list(TIER_SIZES.keys())}"

    tier_events = TIER_SIZES[tier]
    set_deterministic_seed(seed)

    device = torch.device(device_override) if device_override else get_device(verbose=False)
    use_amp = (device.type == "cuda")

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    exp_id = f"expA_{tier.lower()}_{tier_events//1000}k_s{seed}"
    out_dir = (output_root or (REPO_ROOT / "results" / "scaling")) / "experiment_a" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO_ROOT / "results" / "scaling" / "manifests" / "dataset_manifest.json"
    manifest_sha = get_manifest_sha(manifest_path)
    git_commit = get_git_commit()

    meta = {
        "experiment_type": "experiment_a_fixed_epochs",
        "experiment_id": exp_id,
        "tier": tier,
        "tier_events": tier_events,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "drop_last": True,
        "git_commit": git_commit,
        "manifest_sha256": manifest_sha,
        "device": str(device),
        "synthetic_mode": synthetic_mode,
    }

    # Model and optimizer
    model = create_scaling_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    ckpt_manager = ScalingCheckpointManager(out_dir, meta)

    # Data setup
    resolved_data_dir = Path(data_dir) if data_dir else (REPO_ROOT / "data")
    splits_dir = resolved_data_dir / "splits"
    train_indices_file = splits_dir / f"bg_train_{tier.lower()}_{tier_events//1000}k.npy"
    val_indices_file = splits_dir / "bg_val_50k.npy"

    if bg_file:
        bg_h5_path = Path(bg_file)
    else:
        bg_h5_path = resolved_data_dir / "events_LHCO2020_backgroundMC_Pythia.h5"
        if not bg_h5_path.exists():
            alt_candidates = [
                Path("/kaggle/input/lhc-olympics-2020-pythia8-qcd/events_LHCO2020_backgroundMC_Pythia.h5"),
                REPO_ROOT.parent / "LHC Olympics 2020 Background MC Pythia.h5",
            ]
            for cand in alt_candidates:
                if cand.exists():
                    bg_h5_path = cand
                    break

    batches_per_epoch = tier_events // batch_size
    dropped_per_epoch = tier_events % batch_size
    expected_total_updates = epochs * batches_per_epoch

    print("\n" + "=" * 80)
    print(f"  EXPERIMENT A (FIXED EPOCHS): {exp_id}")
    print(f"  Tier: {tier} ({tier_events:,} events) | Seed: {seed} | Epochs: {epochs}")
    print(f"  Batches/Epoch: {batches_per_epoch:,} | Dropped/Epoch: {dropped_per_epoch} events")
    print(f"  Expected Total Updates: {expected_total_updates:,} | Device: {device}")
    print(f"  Background File: {bg_h5_path}")
    print("=" * 80)

    if synthetic_mode:
        print("[SYNTHETIC_DEBUG_ONLY] Generating synthetic events for smoke verification...")
        np.random.seed(seed)
        synth_data = np.random.uniform(-3, 3, (synthetic_events, 2101)).astype(np.float32)
        synth_data[:, :2100:3] = np.where(np.random.rand(synthetic_events, 700) > 0.88, 50.0, 0.0)
        train_ds = ChunkedLHCODataset(Path("mock.h5"), synthetic_source=synth_data, chunk_size=200, shuffle=True, seed=seed)
        val_ds = ChunkedLHCODataset(Path("mock.h5"), synthetic_source=synth_data[:100], chunk_size=100, shuffle=False)
        train_loader = create_streaming_dataloader(train_ds, batch_size=batch_size, num_workers=0)
        val_loader = create_streaming_dataloader(val_ds, batch_size=batch_size, num_workers=0)
    else:
        # Strict fail-closed verification
        if not bg_h5_path.exists():
            raise FileNotFoundError(
                f"[FAIL-CLOSED] Background dataset missing at {bg_h5_path}. "
                "Real collision data is required for production training."
            )
        import sys
        st = bg_h5_path.stat()
        if sys.platform == "darwin" and hasattr(st, "st_blocks") and st.st_blocks == 0:
            raise RuntimeError(
                f"[FAIL-CLOSED] Background file {bg_h5_path} is an APFS dataless file (st_blocks=0). "
                "Physical data must be downloaded locally or run on cloud GPU."
            )

        assert train_indices_file.exists(), f"Partition file missing: {train_indices_file}"
        assert val_indices_file.exists(), f"Partition file missing: {val_indices_file}"
        train_indices = np.load(train_indices_file)
        val_indices = np.load(val_indices_file)

        workers = 2 if device.type == "cuda" else 0
        pin_mem = (device.type == "cuda")

        train_ds = ChunkedLHCODataset(
            bg_h5_path,
            event_indices=train_indices,
            chunk_size=2000,
            shuffle=True,
            seed=seed,
        )
        val_ds = ChunkedLHCODataset(
            bg_h5_path,
            event_indices=val_indices,  # Full 50k validation partition per scientific protocol
            chunk_size=2000,
            shuffle=False,
        )

        train_loader = create_streaming_dataloader(
            train_ds,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin_mem,
            persistent_workers=(workers > 0),
            prefetch_factor=2 if workers > 0 else None,
        )
        val_loader = create_streaming_dataloader(
            val_ds,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=pin_mem,
        )

    # Training execution loop
    actual_updates = 0
    total_events_presented = 0
    history = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t_ep_0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()

        epoch_loss_sum = 0.0
        epoch_grad_norm_sum = 0.0
        epoch_batches = 0

        for batch in train_loader:
            if batch.num_graphs < batch_size:
                # drop_last strict enforcement
                continue

            loss, grad_norm = execute_train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )

            epoch_loss_sum += loss
            epoch_grad_norm_sum += grad_norm
            epoch_batches += 1
            actual_updates += 1
            total_events_presented += batch.num_graphs

        t_ep_1 = time.time()
        mean_train_loss = epoch_loss_sum / max(epoch_batches, 1)
        mean_grad_norm = epoch_grad_norm_sum / max(epoch_batches, 1)

        # Unsupervised validation evaluation
        val_loss = evaluate_held_out_validation(model, val_loader, device)
        is_best = ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            step=actual_updates,
            val_loss=val_loss,
            is_final=(epoch == epochs),
        )

        ep_duration = round(t_ep_1 - t_ep_0, 2)
        history.append({
            "epoch": epoch,
            "train_loss": round(mean_train_loss, 5),
            "val_loss": round(val_loss, 5),
            "grad_norm": round(mean_grad_norm, 4),
            "epoch_time_sec": ep_duration,
            "updates_in_epoch": epoch_batches,
            "cumulative_updates": actual_updates,
            "is_best": is_best,
        })

        star = " *" if is_best else ""
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train: {mean_train_loss:.5f} | Val: {val_loss:.5f}{star} | GradNorm: {mean_grad_norm:.3f} | Updates: {actual_updates:,} ({ep_duration}s)")

    total_duration = round(time.time() - t_start, 2)

    summary = {
        "meta": meta,
        "total_duration_sec": total_duration,
        "actual_updates": actual_updates,
        "expected_updates": expected_total_updates,
        "total_event_presentations": total_events_presented,
        "best_val_loss": ckpt_manager.best_val_loss,
        "best_epoch": ckpt_manager.best_epoch,
        "history": history,
    }

    log_path = out_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[COMPLETED] Experiment A ({exp_id}) finished in {total_duration}s. Best Val Loss: {ckpt_manager.best_val_loss:.5f} (Epoch {ckpt_manager.best_epoch})")
    print(f"[LOG SAVED] {log_path}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment A: Fixed Epochs Scaling Runner")
    parser.add_argument("--tier", type=str, default="D0", choices=list(TIER_SIZES.keys()), help="Dataset tier to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory root")
    parser.add_argument("--bg-file", type=str, default=None, help="Explicit path to background HDF5 dataset")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory containing splits")
    parser.add_argument("--synthetic", action="store_true", help="Run in fast synthetic debug mode (DEBUG_ONLY)")
    args = parser.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else None
    run_experiment_a_tier(
        tier=args.tier,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device_override=args.device,
        output_root=out_root,
        synthetic_mode=args.synthetic,
        bg_file=Path(args.bg_file) if args.bg_file else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )


if __name__ == "__main__":
    main()
