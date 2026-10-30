"""
run_training_suite.py — Master Automated Training & Evaluation Pipeline
=======================================================================
Executes multi-experiment suites end-to-end directly in Python:
1. In-process execution guarantees real-time, unbuffered stdout streaming in Jupyter/Kaggle.
2. Automatically manages GPU memory by clearing cache between runs (torch.cuda.empty_cache()).
3. Automatic checkpoint verification (skips already completed runs unless --force).
4. Automated post-training multi-score evaluation with 1000-resample bootstrap.
5. Automatically compresses results into a 1-click downloadable ZIP archive.

Usage:
  python scripts/run_training_suite.py --family score_decorrelation
  python scripts/run_training_suite.py --family controlled_structure
"""

import os
import sys
import glob
import time
import shutil
import zipfile
import argparse
from pathlib import Path

# Add repo root to Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from src.train_experiment import run_training_experiment
from evaluation.evaluate_controlled_ablation import evaluate_controlled_experiments


def train_and_evaluate_suite(
    family: str,
    force: bool = False,
    dry_run: bool = False,
    eval_limit: int = 10000,
    signal_limit: int = 5000,
    make_zip: bool = True,
) -> None:
    t_suite_start = time.time()
    config_dir = REPO_ROOT / "configs" / family
    configs = sorted(glob.glob(str(config_dir / "*.json")))

    print("=" * 80)
    print(f"MASTER TRAINING SUITE: {family.upper()}")
    print(f"Found {len(configs)} configurations in {config_dir}")
    print(f"Device: {'NVIDIA CUDA GPU' if torch.cuda.is_available() else 'Apple Silicon / CPU'}")
    print("=" * 80)

    if not configs:
        print(f"[ERROR] No configuration files found in: {config_dir}")
        return

    # Pre-flight: verify datasets exist or auto-discover from Kaggle
    bg_target = REPO_ROOT / "data" / "events_LHCO2020_backgroundMC_Pythia.h5"
    sig_target = REPO_ROOT / "data" / "W_prime_signal.h5"
    (REPO_ROOT / "data").mkdir(parents=True, exist_ok=True)

    def is_valid_file(p: Path) -> bool:
        return p.exists() and p.stat().st_size > 100_000_000

    if not is_valid_file(bg_target):
        # Auto-discover from /kaggle/input or parent directories
        search_paths = [Path("/kaggle/input"), REPO_ROOT.parent]
        candidates = []
        for sp in search_paths:
            if sp.exists():
                candidates.extend(sp.glob("**/*background*.h5"))
        candidates = [c for c in candidates if is_valid_file(c)]
        if candidates:
            print(f"[PRE-FLIGHT] Discovered Background dataset at: {candidates[0]}")
            if bg_target.is_symlink() or bg_target.exists():
                bg_target.unlink()
            try:
                bg_target.symlink_to(candidates[0])
                print(f"[PRE-FLIGHT] Symlinked to {bg_target}")
            except Exception as e:
                print(f"[PRE-FLIGHT] Symlink failed ({e}), will access directly if needed.")

    if not is_valid_file(sig_target):
        search_paths = [Path("/kaggle/input"), REPO_ROOT.parent]
        candidates = []
        for sp in search_paths:
            if sp.exists():
                candidates.extend(sp.glob("**/*W_prime*.h5"))
                candidates.extend(sp.glob("**/*anomalydetection*.h5"))
        candidates = [c for c in candidates if is_valid_file(c)]
        if candidates:
            print(f"[PRE-FLIGHT] Discovered Signal dataset at: {candidates[0]}")
            if sig_target.is_symlink() or sig_target.exists():
                sig_target.unlink()
            try:
                sig_target.symlink_to(candidates[0])
                print(f"[PRE-FLIGHT] Symlinked to {sig_target}")
            except Exception as e:
                print(f"[PRE-FLIGHT] Symlink failed ({e}), will access directly if needed.")

    # Validate dataset readiness before starting multi-hour run
    if not is_valid_file(bg_target) and not dry_run:
        print("\n" + "!" * 80)
        print(f"[FATAL] Required background dataset is missing or incomplete: {bg_target}")
        print("Please ensure the dataset is downloaded before running the training suite.")
        print("In Kaggle, run:")
        print('  !wget -q --show-progress -c -O data/events_LHCO2020_backgroundMC_Pythia.h5 \\')
        print('    "https://zenodo.org/api/records/4536624/files/events_LHCO2020_backgroundMC_Pythia.h5/content"')
        print("!" * 80 + "\n")
        return

    # Phase 1: Train all experiments in suite
    completed_runs = []
    failed_runs = []

    for idx, cfg_path_str in enumerate(configs, 1):
        cfg_path = Path(cfg_path_str)
        exp_name = cfg_path.stem
        ckpt_path = REPO_ROOT / "experiments" / family / exp_name / "checkpoint.pt"

        print(f"\n{'#' * 80}")
        print(f"[{idx}/{len(configs)}] EXPERIMENT: {exp_name}")
        print(f"{'#' * 80}")
        sys.stdout.flush()

        if ckpt_path.exists() and ckpt_path.stat().st_size > 1000 and not force and not dry_run:
            print(f"[INFO] Checkpoint exists ({ckpt_path.stat().st_size / 1024:.1f} KB). Skipping training.")
            completed_runs.append(exp_name)
            continue

        t0 = time.time()
        try:
            exit_code = run_training_experiment(cfg_path, dry_run=dry_run)
            if exit_code == 0:
                completed_runs.append(exp_name)
                print(f"[SUCCESS] {exp_name} completed in {time.time() - t0:.1f}s.")
            else:
                failed_runs.append((exp_name, f"Exit code {exit_code}"))
                print(f"[FAILURE] {exp_name} failed with exit code {exit_code}.")
        except Exception as e:
            failed_runs.append((exp_name, str(e)))
            print(f"[ERROR] Exception occurred during training of {exp_name}: {e}")
            import traceback
            traceback.print_exc()

        sys.stdout.flush()

        # Clean VRAM after each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"TRAINING PHASE COMPLETE: {len(completed_runs)} succeeded, {len(failed_runs)} failed.")
    print("=" * 80)

    if failed_runs:
        print("[WARNING] The following runs encountered issues:")
        for name, err in failed_runs:
            print(f"  - {name}: {err}")

    if dry_run:
        print("[INFO] Dry run finished. Skipping evaluation.")
        return

    if not completed_runs:
        print("\n" + "!" * 80)
        print(f"[FATAL] Zero experiments succeeded! Aborting evaluation and bundle creation.")
        print("Please check the error logs above for failure causes.")
        print("!" * 80 + "\n")
        return

    # Phase 2: Automated Evaluation with 1000-resample bootstrap
    print("\n" + "=" * 80)
    print(f"PHASE 2: EXECUTING EVALUATION SUITE ({family})")
    print(f"Background Events: {eval_limit} | Signal Events: {signal_limit}")
    print("=" * 80)
    sys.stdout.flush()

    try:
        evaluate_controlled_experiments(
            family=family,
            eval_limit=eval_limit,
            signal_limit=signal_limit,
        )
    except Exception as e:
        print(f"[ERROR] Evaluation encountered an exception: {e}")
        import traceback
        traceback.print_exc()

    # Phase 3: Package Results into ZIP Bundle
    if make_zip:
        zip_filename = f"{family}_results_bundle.zip"
        zip_path = REPO_ROOT / zip_filename
        print(f"\n[INFO] Compressing outputs into {zip_path}...")
        
        file_count = 0
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add comparisons
            comp_dir = REPO_ROOT / "comparisons" / family
            if comp_dir.exists():
                for root, _, files in os.walk(comp_dir):
                    for f in files:
                        p = Path(root) / f
                        zf.write(p, arcname=p.relative_to(REPO_ROOT))
                        file_count += 1
            # Add experiments (logs and configs, omitting giant scratch files)
            exp_dir = REPO_ROOT / "experiments" / family
            if exp_dir.exists():
                for root, _, files in os.walk(exp_dir):
                    for f in files:
                        if f.endswith((".json", ".pt", ".png", ".md")):
                            p = Path(root) / f
                            zf.write(p, arcname=p.relative_to(REPO_ROOT))
                            file_count += 1

        if file_count == 0:
            print(f"[WARNING] No files found to package for {family}! (0 files added)")
        else:
            print(f"[SUCCESS] Archive created: {zip_path} ({file_count} files, {zip_path.stat().st_size / (1024**2):.2f} MB)")
            # Also copy to /kaggle/working if on Kaggle
            if Path("/kaggle/working").exists():
                kaggle_out = Path("/kaggle/working") / zip_filename
                try:
                    shutil.copy2(zip_path, kaggle_out)
                    print(f"[KAGGLE] Copied bundle to {kaggle_out} for sidebar download.")
                except Exception:
                    pass

            print("\n" + "=" * 80)
            print("To download the results bundle, run the download cell provided in the guide.")
            print("=" * 80)

    print(f"\n[ALL DONE] Total pipeline elapsed time: {time.time() - t_suite_start:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Master Automated Training & Evaluation Suite")
    parser.add_argument(
        "--family",
        type=str,
        default="score_decorrelation",
        choices=["score_decorrelation", "controlled_structure", "hybrid_pointnet_gat", "architecture", "graph_construction", "met_weight"],
        help="Experiment family to execute",
    )
    parser.add_argument("--force", action="store_true", help="Force re-training even if checkpoint exists")
    parser.add_argument("--dry-run", action="store_true", help="Validate configs and model init without training")
    parser.add_argument("--eval-limit", type=int, default=10000, help="Number of background events for evaluation")
    parser.add_argument("--signal-limit", type=int, default=5000, help="Number of signal events for evaluation")
    args = parser.parse_args()

    train_and_evaluate_suite(
        family=args.family,
        force=args.force,
        dry_run=args.dry_run,
        eval_limit=args.eval_limit,
        signal_limit=args.signal_limit,
    )


if __name__ == "__main__":
    main()
