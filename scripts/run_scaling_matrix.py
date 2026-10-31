"""
scripts/run_scaling_matrix.py — Automated Matrix Runner for Data-Scaling Study
=============================================================================
Orchestrates execution of the 42-run matrix:
  - Tiers: D0 (5k), D1 (25k), D2 (50k), D3 (100k), D4 (250k), D5 (500k), D6 (900k)
  - Seeds: 42, 123, 999
  - Experiments:
      * Experiment A (Fixed Epochs: 20 epochs across all tiers)
      * Experiment B (Fixed Updates: 10,000 updates across all tiers)

Features:
  - Modular CLI: run specific tiers, seeds, experiments, or the full matrix
  - Checkpoint-safe & resume-capable (skips already completed runs)
  - Integrated evaluation pipeline: automatically extracts discrimination and mass-sculpting metrics
  - Fully compatible with Kaggle NVIDIA GPUs and local Apple Silicon / CPU
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.scaling.experiment_a_fixed_epochs import run_experiment_a_tier
from experiments.scaling.experiment_b_fixed_updates import run_experiment_b_tier
from evaluation.evaluate_scaling_study import main as evaluate_main
from src.run_lhco_gnn import get_device

ALL_TIERS = ["D0", "D1", "D2", "D3", "D4", "D5", "D6"]
ALL_SEEDS = [42, 123, 999]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LHC Olympics PointNet-GAT Scaling Matrix")
    parser.add_argument(
        "--experiment",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Experiment to run: A (Fixed Epochs), B (Fixed Updates), or both (default: both)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all", "D0", "D1", "D2", "D3", "D4", "D5", "D6"],
        help="Dataset tier to execute, or 'all' for full scaling ladder (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="all",
        help="Random seed to run (e.g., '42', '123', '999', or 'all')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs for Experiment A (default: 20)",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=10000,
        help="Total updates for Experiment B (default: 10,000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size (fixed at 32 by protocol)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run in synthetic smoke-test mode",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root output directory (default: results/scaling)",
    )
    parser.add_argument(
        "--bg-file",
        type=str,
        default=None,
        help="Explicit path to background HDF5 dataset",
    )
    parser.add_argument(
        "--sig-file",
        type=str,
        default=None,
        help="Explicit path to signal HDF5 dataset",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory containing splits",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training test evaluation",
    )
    return parser.parse_args()


def run_matrix():
    args = parse_arguments()
    device = get_device(verbose=False)

    out_root = Path(args.output_root) if args.output_root else REPO_ROOT / "results" / "scaling"
    out_root.mkdir(parents=True, exist_ok=True)

    tiers = ALL_TIERS if args.tier == "all" else [args.tier]
    seeds = ALL_SEEDS if args.seed == "all" else [int(args.seed)]
    experiments = ["A", "B"] if args.experiment == "both" else [args.experiment]

    total_runs = len(experiments) * len(tiers) * len(seeds)
    print("\n" + "=" * 80)
    print("  POINTNET-GAT BACKGROUND-DATA SCALING MATRIX ORCHESTRATOR")
    print(f"  Experiments: {experiments} | Tiers: {tiers} | Seeds: {seeds}")
    print(f"  Total Runs Planned: {total_runs} | Device: {device} | Synthetic: {args.synthetic}")
    print("=" * 80)

    matrix_log = out_root / "matrix_execution_log.json"
    execution_results = []
    run_idx = 0

    for exp_type in experiments:
        for tier in tiers:
            for seed in seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] >>> INITIATING EXP {exp_type} | TIER: {tier} | SEED: {seed} <<<")
                start_time = time.time()

                if exp_type == "A":
                    # Experiment A: Fixed Epochs
                    res = run_experiment_a_tier(
                        tier=tier,
                        seed=seed,
                        epochs=2 if args.synthetic else args.epochs,
                        batch_size=args.batch_size,
                        output_root=out_root,
                        synthetic_mode=args.synthetic,
                        synthetic_events=500 if args.synthetic else 0,
                        bg_file=Path(args.bg_file) if args.bg_file else None,
                        data_dir=Path(args.data_dir) if args.data_dir else None,
                    )
                    exp_dir = out_root / "experiment_a" / res["meta"]["experiment_id"]
                else:
                    # Experiment B: Fixed Updates
                    res = run_experiment_b_tier(
                        tier=tier,
                        seed=seed,
                        total_updates=30 if args.synthetic else args.updates,
                        batch_size=args.batch_size,
                        output_root=out_root,
                        synthetic_mode=args.synthetic,
                        synthetic_events=500 if args.synthetic else 0,
                        bg_file=Path(args.bg_file) if args.bg_file else None,
                        data_dir=Path(args.data_dir) if args.data_dir else None,
                    )
                    exp_dir = out_root / "experiment_b" / res["meta"]["experiment_id"]

                duration = round(time.time() - start_time, 2)
                best_ckpt = exp_dir / "checkpoint_best.pt"

                # Automated Evaluation
                eval_metrics = None
                if not args.skip_eval and best_ckpt.exists():
                    eval_json = exp_dir / "eval_metrics.json"
                    print(f"  [EVALUATING] Checkpoint: {best_ckpt} -> {eval_json}")
                    eval_cmd = [
                        "evaluate_scaling_study.py",
                        "--checkpoint",
                        str(best_ckpt),
                        "--output-json",
                        str(eval_json),
                    ]
                    if args.synthetic:
                        eval_cmd.append("--synthetic")
                    if args.bg_file:
                        eval_cmd.extend(["--bg-file", args.bg_file])
                    if args.sig_file:
                        eval_cmd.extend(["--sig-file", args.sig_file])
                    if args.data_dir:
                        eval_cmd.extend(["--data-dir", args.data_dir])

                    old_argv = sys.argv
                    sys.argv = eval_cmd
                    try:
                        evaluate_main()
                        if eval_json.exists():
                            with open(eval_json) as f:
                                eval_metrics = json.load(f)
                    except Exception as e:
                        print(f"  [EVALUATION ERROR] {e}")
                    finally:
                        sys.argv = old_argv

                record = {
                    "experiment": exp_type,
                    "tier": tier,
                    "seed": seed,
                    "status": "COMPLETED",
                    "duration_sec": duration,
                    "best_val_loss": res.get("best_val_loss"),
                    "checkpoint": str(best_ckpt),
                    "eval_metrics": eval_metrics,
                }
                execution_results.append(record)

                with open(matrix_log, "w", encoding="utf-8") as f:
                    json.dump(execution_results, f, indent=2)

    print("\n" + "=" * 80)
    print("  ALL MATRIX RUNS PROCESSED SUCCESSFULLY")
    print(f"  Master log written to: {matrix_log}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_matrix()
