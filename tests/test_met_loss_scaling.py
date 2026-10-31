"""
test_met_loss_scaling.py — Unit Tests for MET Target Scaling and Gradient Balance
==================================================================================
Verifies:
  1. MET calculation units and mathematical correctness.
  2. MET loss reduction consistency.
  3. Shared parameter gradient computation.
  4. Gradient ratio under remediated loss scaling (preventing gradient swamping).
"""

import sys
import unittest
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_lhco_gnn import PointNetGATHybrid, compute_MET
from src.remediation_diagnostics import measure_met_gradients, create_synthetic_collision_batch


class TestMETLossScaling(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4)

    def test_compute_met_units_and_physics(self):
        """Verifies compute_MET correctly sums px, py and returns vector in GeV."""
        particles = np.array([
            [100.0, 0.0, 0.0],
            [50.0, 1.0, np.pi / 2.0],
        ])
        met, met_x, met_y = compute_MET(particles)

        self.assertAlmostEqual(met_x, -100.0, places=4)
        self.assertAlmostEqual(met_y, -50.0, places=4)
        self.assertAlmostEqual(met, np.sqrt(100.0**2 + 50.0**2), places=4)

    def test_met_head_shared_parameters_exist(self):
        """Identifies parameters shared between reconstruction and MET regression."""
        shared_params = [name for name, p in self.model.named_parameters() if "point_phi" in name]
        self.assertEqual(len(shared_params), 4, f"Expected 4 point_phi tensors, got {shared_params}")
        self.assertIn("point_phi.0.weight", shared_params)
        self.assertIn("point_phi.2.weight", shared_params)

    def test_historical_gradient_swamping(self):
        """Demonstrates that historical lambda=0.1 on raw GeV produces g_MET >> g_recon."""
        batch = create_synthetic_collision_batch(batch_size=32, seed=42)
        res = measure_met_gradients(self.model, batch, lambda_met=0.1, met_scale=1.0)
        # Ratio g_MET / g_recon must be > 1.0 (historical imbalance)
        self.assertGreater(res["ratio_shared"], 2.0, "Historical MET gradient should dominate reconstruction")

    def test_remediated_gradient_balance(self):
        """Verifies that remediated lambda_met=0.001 achieves healthy auxiliary balance (ratio in [0.01, 0.30])."""
        batch = create_synthetic_collision_batch(batch_size=32, seed=42)
        res = measure_met_gradients(self.model, batch, lambda_met=0.001, met_scale=1.0)
        ratio = res["ratio_shared"]
        self.assertGreaterEqual(ratio, 0.01, "Auxiliary MET gradient must not be zero")
        self.assertLessEqual(ratio, 0.30, "Auxiliary MET gradient must not dominate reconstruction (>30%)")


if __name__ == "__main__":
    unittest.main()
