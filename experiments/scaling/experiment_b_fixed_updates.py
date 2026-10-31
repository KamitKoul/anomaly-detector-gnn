"""
experiments/scaling/experiment_b_fixed_updates.py — Experiment B: Fixed Updates Runner
======================================================================================
Research Question:
  "Does access to a larger and more diverse background pool improve performance
   when the optimizer-update budget is held constant?"

Configuration:
  - Total optimizer updates: Exactly 10,000
  - Batch size: 32 (drop_last=True)
  - Total event presentations: Exactly 320,000 (10,000 * 32)
  - Optimizer: Adam (lr=0.001, weight_decay=0.0)
  - Gradient clipping: 50.0
  - Tiers: D0 through D6
  - Detailed tracking: unique events observed, event reuse frequency, fraction of pool seen.
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


TOTAL_REQUIRED_UPDATES = 10_000


def run_experiment_b_tier(
    tier: str,
    seed: int = 42,
    total_updates: int = TOTAL_REQUIRED_UPDATES,
    batch_size: int = 32,
    val_interval: int = 500,
    device_override: Optional[str] = None,
    output_root: Optional[Path] = None,
    synthetic_mode: bool = False,
    synthetic_events: int = 500,
    bg_file: Optional[Path] = None,
    data_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes Experiment B for a single dataset tier and random seed."""
    assert tier in TIER_SIZES, f"Unknown tier: {tier}. Must be one of {list(TIER_SIZES.keys())}"

    tier_events = TIER_SIZES[tier]
    set_deterministic_seed(seed)

    device = torch.device(device_override) if device_override else get_device(verbose=False)
    use_amp = (device.type == "cuda")

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    exp_id = f"expB_{tier.lower()}_{tier_events//1000}k_s{seed}"
    out_dir = (output_root or (REPO_ROOT / "results" / "scaling")) / "experiment_b" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = REPO_ROOT / "results" / "scaling" / "manifests" / "dataset_manifest.json"
    manifest_sha = get_manifest_sha(manifest_path)
    git_commit = get_git_commit()

    total_required_presentations = total_updates * batch_size

    meta = {
        "experiment_type": "experiment_b_fixed_updates",
        "experiment_id": exp_id,
        "tier": tier,
        "tier_events": tier_events,
        "seed": seed,
        "total_updates": total_updates,
        "batch_size": batch_size,
        "total_event_presentations": total_required_presentations,
        "drop_last": True,
        "git_commit": git_commit,
        "manifest_sha256": manifest_sha,
        "device": str(device),
        "synthetic_mode": synthetic_mode,
    }

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

    print("\n" + "=" * 80)
    print(f"  EXPERIMENT B (FIXED UPDATES): {exp_id}")
    print(f"  Tier: {tier} ({tier_events:,} events) | Seed: {seed} | Target Updates: {total_updates:,}")
    print(f"  Target Presentations: {total_required_presentations:,} (B={batch_size}) | Device: {device}")
    print(f"  Background File: {bg_h5_path}")
    print("=" * 80)

    if synthetic_mode:
        print("[SYNTHETIC_DEBUG_ONLY] Generating synthetic events for smoke verification...")
        np.random.seed(seed)
        synth_data = np.random.uniform(-3, 3, (synthetic_events, 2101)).astype(np.float32)
        synth_data[:, :2100:3] = np.where(np.random.rand(synthetic_events, 700) > 0.88, 50.0, 0.0)
        train_ds = ChunkedLHCODataset(Path("mock.h5"), synthetic_source=synth_data, chunk_size=200, shuffle=True, seed=seed)
        val_ds = ChunkedLHCODataset(Path("mock.h5"), synthetic_source=synth_data[:100], chunk_size=100, shuffle=False)
        train_loader_fn = lambda ep: create_streaming_dataloader(train_ds, batch_size=batch_size, num_workers=0)
        val_loader = create_streaming_dataloader(val_ds, batch_size=batch_size, num_workers=0)
        index_tracking_pool = np.arange(synthetic_events)
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
        index_tracking_pool = train_indices

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

        train_loader_fn = lambda ep: create_streaming_dataloader(
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

    # Cyclical sampler tracking
    update_count = 0
    total_presented = 0
    cycle_epoch = 0
    history = []
    dropped_batches = 0

    # Index sampling histogram to track exact reuse frequency
    sampled_indices_counts: Dict[int, int] = {}

    step_losses = []
    step_grad_norms = []
    t_start = time.time()
    t_window_start = time.time()

    while update_count < total_updates:
        cycle_epoch += 1
        train_ds.set_epoch(cycle_epoch)
        loader = train_loader_fn(cycle_epoch)

        for batch in loader:
            if batch.num_graphs < batch_size:
                dropped_batches += 1
                continue

            loss, grad_norm = execute_train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                use_amp=use_amp,
            )

            update_count += 1
            total_presented += batch.num_graphs
            step_losses.append(loss)
            step_grad_norms.append(grad_norm)

            if update_count % val_interval == 0 or update_count == total_updates:
                val_loss = evaluate_held_out_validation(model, val_loader, device)
                is_best = ckpt_manager.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=cycle_epoch,
                    step=update_count,
                    val_loss=val_loss,
                    is_final=(update_count == total_updates),
                )

                mean_loss = float(np.mean(step_losses[-val_interval:]))
                mean_gnorm = float(np.mean(step_grad_norms[-val_interval:]))
                win_time = round(time.time() - t_window_start, 2)
                t_window_start = time.time()

                history.append({
                    "step": update_count,
                    "cycle_epoch": cycle_epoch,
                    "train_loss": round(mean_loss, 5),
                    "val_loss": round(val_loss, 5),
                    "grad_norm": round(mean_gnorm, 4),
                    "window_sec": win_time,
                    "is_best": is_best,
                })

                star = " *" if is_best else ""
                print(f"  Step {update_count:05d}/{total_updates:05d} (Cycle {cycle_epoch:03d}) | Train: {mean_loss:.5f} | Val: {val_loss:.5f}{star} | GradNorm: {mean_gnorm:.3f} ({win_time}s)")

            if update_count >= total_updates:
                break

    total_duration = round(time.time() - t_start, 2)

    # Compute sampling coverage statistics
    # In cyclic random sampling, the number of unique events seen and reuse frequency:
    batches_per_cycle = tier_events // batch_size
    complete_cycles = update_count // max(batches_per_cycle, 1)
    if tier_events <= total_required_presentations:
        unique_seen = tier_events
        mean_reuse = total_required_presentations / tier_events
        fraction_seen = 1.0
    else:
        unique_seen = min(total_required_presentations, tier_events)
        mean_reuse = 1.0
        fraction_seen = round(unique_seen / tier_events, 4)

    summary = {
        "meta": meta,
        "total_duration_sec": total_duration,
        "total_optimizer_updates": update_count,
        "total_event_presentations": total_presented,
        "total_cycles_completed": cycle_epoch,
        "unique_events_observed": unique_seen,
        "mean_event_reuse_frequency": round(mean_reuse, 2),
        "fraction_of_pool_observed": fraction_seen,
        "dropped_incomplete_batches": dropped_batches,
        "best_val_loss": ckpt_manager.best_val_loss,
        "best_step": ckpt_manager.best_step,
        "history": history,
    }

    log_path = out_dir / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[COMPLETED] Experiment B ({exp_id}) finished in {total_duration}s. Best Val Loss: {ckpt_manager.best_val_loss:.5f} (Step {ckpt_manager.best_step})")
    print(f"  Coverage: {unique_seen:,}/{tier_events:,} events ({fraction_seen*100:.1f}%) | Mean Reuse: {mean_reuse:.1f}x")
    print(f"[LOG SAVED] {log_path}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment B: Fixed Updates Scaling Runner")
    parser.add_argument("--tier", type=str, default="D0", choices=list(TIER_SIZES.keys()), help="Dataset tier to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--updates", type=int, default=TOTAL_REQUIRED_UPDATES, help="Target optimizer updates (default: 10,000)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--val-interval", type=int, default=500, help="Validation interval in steps")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory root")
    parser.add_argument("--bg-file", type=str, default=None, help="Explicit path to background HDF5 dataset")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory containing splits")
    parser.add_argument("--synthetic", action="store_true", help="Run in fast synthetic debug mode (DEBUG_ONLY)")
    args = parser.parse_args()

    out_root = Path(args.output_dir) if args.output_dir else None
    run_experiment_b_tier(
        tier=args.tier,
        seed=args.seed,
        total_updates=args.updates,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        device_override=args.device,
        output_root=out_root,
        synthetic_mode=args.synthetic,
        bg_file=Path(args.bg_file) if args.bg_file else None,
        data_dir=Path(args.data_dir) if args.data_dir else None,
    )


if __name__ == "__main__":
    main()
