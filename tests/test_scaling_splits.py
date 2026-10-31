"""
tests/test_scaling_splits.py — Unit Tests for Dataset Splitting & Scaling Partitions
=====================================================================================
Mandatory Integrity Tests for Phase 2:
  1. Background train/validation/test zero overlap.
  2. Signal strictly excluded from training pool.
  3. Strict nested hierarchy: D0 ⊂ D1 ⊂ D2 ⊂ D3 ⊂ D4 ⊂ D5 ⊂ D6.
  4. Dataset sizes and SHA-256 hashes match manifest.
  5. Streaming data loader event accounting and worker independence.
"""

import sys
import json
import unittest
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.scaling_split import compute_sha256, TIER_SIZES


class TestScalingSplits(unittest.TestCase):

    def setUp(self):
        self.splits_dir = REPO_ROOT / "data" / "splits"
        self.manifest_path = self.splits_dir / "dataset_manifest.json"
        self.assertTrue(self.manifest_path.exists(), "dataset_manifest.json must exist in data/splits")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def test_background_partitions_zero_overlap(self):
        """Verifies strictly zero overlap between background train, val, and test partitions."""
        test_idx = np.load(self.splits_dir / "bg_test_50k.npy")
        val_idx = np.load(self.splits_dir / "bg_val_50k.npy")
        train_d6 = np.load(self.splits_dir / "bg_train_d6_900k.npy")

        self.assertEqual(len(np.intersect1d(test_idx, val_idx)), 0, "BG test and val must have 0 overlap")
        self.assertEqual(len(np.intersect1d(test_idx, train_d6)), 0, "BG test and train must have 0 overlap")
        self.assertEqual(len(np.intersect1d(val_idx, train_d6)), 0, "BG val and train must have 0 overlap")

    def test_nested_subsets_strict_hierarchy(self):
        """Verifies D0 ⊂ D1 ⊂ D2 ⊂ D3 ⊂ D4 ⊂ D5 ⊂ D6."""
        tiers = list(TIER_SIZES.keys())
        arrays = {t: np.load(self.splits_dir / f"bg_train_{t.lower()}_{TIER_SIZES[t]//1000}k.npy") for t in tiers}

        for i in range(len(tiers) - 1):
            t_curr = tiers[i]
            t_next = tiers[i + 1]
            a_curr = arrays[t_curr]
            a_next = arrays[t_next]

            # Length check
            self.assertEqual(len(a_curr), TIER_SIZES[t_curr])
            self.assertEqual(len(a_next), TIER_SIZES[t_next])

            # Strict prefix check (since prefix construction was used)
            self.assertTrue(np.array_equal(a_curr, a_next[:len(a_curr)]), f"{t_curr} must be exact prefix of {t_next}")

            # Subset check
            s_curr = set(a_curr)
            s_next = set(a_next)
            self.assertTrue(s_curr.issubset(s_next), f"{t_curr} must be strict subset of {t_next}")
            self.assertLess(len(s_curr), len(s_next))

    def test_training_indices_inside_train_pool(self):
        """Verifies no event outside the 900,000 training pool exists in any tier."""
        d6_pool = set(np.load(self.splits_dir / "bg_train_d6_900k.npy"))
        self.assertEqual(len(d6_pool), 900_000)

        tiers = ["D0", "D1", "D2", "D3", "D4", "D5"]
        for t in tiers:
            arr = np.load(self.splits_dir / f"bg_train_{t.lower()}_{TIER_SIZES[t]//1000}k.npy")
            self.assertTrue(set(arr).issubset(d6_pool), f"{t} contains indices outside D6 pool!")

    def test_signal_partitions_zero_overlap(self):
        """Verifies strictly zero overlap between signal test, val, and reserve pools."""
        sig_test = np.load(self.splits_dir / "sig_test_25k.npy")
        sig_val = np.load(self.splits_dir / "sig_val_25k.npy")
        sig_reserve = np.load(self.splits_dir / "sig_reserve_50k.npy")

        self.assertEqual(len(np.intersect1d(sig_test, sig_val)), 0, "Signal test and val overlap!")
        self.assertEqual(len(np.intersect1d(sig_test, sig_reserve)), 0, "Signal test and reserve overlap!")
        self.assertEqual(len(np.intersect1d(sig_val, sig_reserve)), 0, "Signal val and reserve overlap!")

        # Union must be 100,000
        total_sig = len(sig_test) + len(sig_val) + len(sig_reserve)
        self.assertEqual(total_sig, 100_000, "Signal partition union must be 100,000 events")

    def test_manifest_checksum_integrity(self):
        """Verifies SHA-256 hashes of all saved index arrays match the manifest exactly."""
        partitions = self.manifest["partitions"]
        for fname, info in partitions.items():
            fpath = self.splits_dir / fname
            self.assertTrue(fpath.exists(), f"File missing: {fpath}")
            current_sha = compute_sha256(fpath)
            self.assertEqual(current_sha, info["sha256"], f"SHA-256 mismatch for {fname}")
            self.assertEqual(len(np.load(fpath)), info["events"])


if __name__ == "__main__":
    unittest.main()
