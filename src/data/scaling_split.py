"""
src/data/scaling_split.py — Deterministic Dataset Splitting & Nested Subset Generator
====================================================================================
Generates cryptographically verified, deterministic partitions and nested subsets
for the LHC Olympics 2020 background-data scaling experiment:
  1. Partitions 1,000,000 background events into:
     - 900,000 Training Pool
     - 50,000 Held-Out Validation Pool
     - 50,000 Held-Out Final Test Pool
  2. Constructs strict prefix-based nested training subsets:
     D0 (5k) ⊂ D1 (25k) ⊂ D2 (50k) ⊂ D3 (100k) ⊂ D4 (250k) ⊂ D5 (500k) ⊂ D6 (900k)
  3. Partitions 100,000 signal events into:
     - 25,000 Final Test Pool
     - 25,000 Diagnostic Validation Pool
     - 50,000 Signal Reserve Pool
  4. Generates immutable SHA-256 manifests verifying zero mutual leakage.

Usage:
  python src/data/scaling_split.py [--output-dir data/splits] [--seed 20261031]
"""

import sys
import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SPLIT_SEED = 20261031
TOTAL_BG_EVENTS = 1_000_000
TOTAL_SIG_EVENTS = 100_000

BG_VAL_SIZE = 50_000
BG_TEST_SIZE = 50_000
BG_TRAIN_POOL_SIZE = 900_000

SIG_VAL_SIZE = 25_000
SIG_TEST_SIZE = 25_000
SIG_RESERVE_SIZE = 50_000

TIER_SIZES = {
    "D0": 5_000,
    "D1": 25_000,
    "D2": 50_000,
    "D3": 100_000,
    "D4": 250_000,
    "D5": 500_000,
    "D6": 900_000,
}


def compute_sha256(filepath: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_scaling_splits(
    output_dir: Path,
    manifest_dir: Path,
    seed: int = SPLIT_SEED,
) -> Dict[str, Any]:
    """Generates deterministic nested splits, saving .npy arrays and JSON manifest."""
    output_dir = Path(output_dir)
    manifest_dir = Path(manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  SCALING DATASET SPLIT & NESTED SUBSET GENERATION")
    print(f"  Seed: {seed} | Output Dir: {output_dir}")
    print("=" * 80)

    # 1. Background Deterministic Permutation
    rng_bg = np.random.RandomState(seed)
    bg_perm = rng_bg.permutation(TOTAL_BG_EVENTS).astype(np.int64)

    # Allocate held-out test and validation first
    bg_test_idx = np.sort(bg_perm[0 : BG_TEST_SIZE])
    bg_val_idx = np.sort(bg_perm[BG_TEST_SIZE : BG_TEST_SIZE + BG_VAL_SIZE])

    # The remaining 900,000 events constitute the Training Pool
    bg_train_pool = bg_perm[BG_TEST_SIZE + BG_VAL_SIZE :]
    assert len(bg_train_pool) == BG_TRAIN_POOL_SIZE

    # Prefix-based nested subset generation on the training pool
    tier_arrays: Dict[str, np.ndarray] = {}
    for tier_name, count in TIER_SIZES.items():
        # Prefix slice guarantees D_i ⊂ D_{i+1}
        subset = bg_train_pool[:count]
        tier_arrays[tier_name] = subset

    # 2. Signal Deterministic Permutation
    rng_sig = np.random.RandomState(seed + 1)
    sig_perm = rng_sig.permutation(TOTAL_SIG_EVENTS).astype(np.int64)

    sig_test_idx = np.sort(sig_perm[0 : SIG_TEST_SIZE])
    sig_val_idx = np.sort(sig_perm[SIG_TEST_SIZE : SIG_TEST_SIZE + SIG_VAL_SIZE])
    sig_reserve_idx = np.sort(sig_perm[SIG_TEST_SIZE + SIG_VAL_SIZE :])

    # 3. Mandatory Integrity Assertions
    print("[INTEGRITY CHECK 1] Checking Background Split Exclusivity...")
    assert len(np.intersect1d(bg_test_idx, bg_val_idx)) == 0, "BG test and val overlap!"
    assert len(np.intersect1d(bg_test_idx, bg_train_pool)) == 0, "BG test and train overlap!"
    assert len(np.intersect1d(bg_val_idx, bg_train_pool)) == 0, "BG val and train overlap!"

    print("[INTEGRITY CHECK 2] Checking Nested Subset Relationships (D0 ⊂ D1 ⊂ ... ⊂ D6)...")
    tiers = list(TIER_SIZES.keys())
    for i in range(len(tiers) - 1):
        t_curr = tiers[i]
        t_next = tiers[i + 1]
        set_curr = set(tier_arrays[t_curr])
        set_next = set(tier_arrays[t_next])
        assert set_curr.issubset(set_next), f"Violation: {t_curr} is not a subset of {t_next}!"
        assert len(set_curr) < len(set_next), f"Violation: {t_curr} not strictly smaller than {t_next}!"

    print("[INTEGRITY CHECK 3] Checking Signal Split Exclusivity...")
    assert len(np.intersect1d(sig_test_idx, sig_val_idx)) == 0, "Signal test and val overlap!"
    assert len(np.intersect1d(sig_test_idx, sig_reserve_idx)) == 0, "Signal test and reserve overlap!"
    assert len(np.intersect1d(sig_val_idx, sig_reserve_idx)) == 0, "Signal val and reserve overlap!"

    # 4. Save Index Arrays to Disk
    files_saved: Dict[str, Path] = {
        "bg_test_50k.npy": bg_test_idx,
        "bg_val_50k.npy": bg_val_idx,
        "bg_train_d0_5k.npy": tier_arrays["D0"],
        "bg_train_d1_25k.npy": tier_arrays["D1"],
        "bg_train_d2_50k.npy": tier_arrays["D2"],
        "bg_train_d3_100k.npy": tier_arrays["D3"],
        "bg_train_d4_250k.npy": tier_arrays["D4"],
        "bg_train_d5_500k.npy": tier_arrays["D5"],
        "bg_train_d6_900k.npy": tier_arrays["D6"],
        "sig_test_25k.npy": sig_test_idx,
        "sig_val_25k.npy": sig_val_idx,
        "sig_reserve_50k.npy": sig_reserve_idx,
    }

    manifest_entries: Dict[str, Any] = {}
    print("\n[SAVING ARRAYS & COMPUTING SHA-256 HASHES]")
    for fname, arr in files_saved.items():
        fpath = output_dir / fname
        np.save(fpath, arr)
        sha = compute_sha256(fpath)
        manifest_entries[fname] = {
            "path": str(fpath.relative_to(REPO_ROOT) if fpath.is_relative_to(REPO_ROOT) else fpath),
            "events": len(arr),
            "dtype": str(arr.dtype),
            "sha256": sha,
            "min_index": int(np.min(arr)),
            "max_index": int(np.max(arr)),
        }
        print(f"  {fname:<24} | Events: {len(arr):>7} | SHA256: {sha[:16]}...")

    manifest = {
        "protocol": "lhc_olympics_data_scaling_split",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_background_events": TOTAL_BG_EVENTS,
        "total_signal_events": TOTAL_SIG_EVENTS,
        "partitions": manifest_entries,
        "nested_hierarchy": {
            "expression": "D0 (5k) ⊂ D1 (25k) ⊂ D2 (50k) ⊂ D3 (100k) ⊂ D4 (250k) ⊂ D5 (500k) ⊂ D6 (900k)",
            "verified": True,
        },
        "leakage_audit": {
            "bg_test_train_val_overlap": 0,
            "signal_leakage_into_train": 0,
            "verified": True,
        },
    }

    # Save manifest in data/splits/ and results/scaling/manifests/
    for dest_dir in [output_dir, manifest_dir]:
        manifest_path = dest_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[MANIFEST WRITTEN] {manifest_path}")

    print("\n[SUCCESS] All partitions verified and written successfully.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Deterministic Scaling Dataset Split Generator")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for index arrays")
    parser.add_argument("--manifest-dir", type=str, default="results/scaling/manifests", help="Output directory for JSON manifest")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Random seed for deterministic permutation")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.output_dir
    man_dir = REPO_ROOT / args.manifest_dir
    generate_scaling_splits(output_dir=out_dir, manifest_dir=man_dir, seed=args.seed)


if __name__ == "__main__":
    main()
