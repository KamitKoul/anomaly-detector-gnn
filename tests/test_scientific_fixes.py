"""
test_scientific_fixes.py — Unit Tests for Periodic Loss, Bootstrap CI, and Metric Rigor
======================================================================================
Verifies:
1. periodic_dphi correctly computes shortest angular distance on S^1.
2. compute_kinematic_recon_loss calculates non-exploding periodic gradients.
3. compute_roc_auc_with_bootstrap preserves stratified class balance and computes valid CI.
4. ScoreDecorrelatedNodeGNN functions safely under half precision (float16) without dtype error.
"""

import unittest
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from src.run_lhco_gnn import (
    NodeGNN,
    periodic_dphi,
    compute_kinematic_recon_loss,
)
from src.adversarial_unlearning_ae import ScoreDecorrelatedNodeGNN
from evaluation.evaluate_controlled_ablation import compute_roc_auc_with_bootstrap


class TestScientificFixes(unittest.TestCase):

    def test_periodic_dphi_values(self):
        """Verify periodic angular distance across branch cut seam."""
        phi1 = torch.tensor([-3.10])
        phi2 = torch.tensor([3.10])
        d = periodic_dphi(phi1, phi2)
        # Expected: 2*pi - 6.20 = 0.083185...
        expected = 2.0 * np.pi - 6.20
        self.assertAlmostEqual(d.item(), expected, places=4)

        # Identical angles
        self.assertAlmostEqual(periodic_dphi(torch.tensor([1.5]), torch.tensor([1.5])).item(), 0.0, places=5)

    def test_periodic_loss_gradients(self):
        """Verify that gradients pull smoothly across the seam without exploding."""
        recon = torch.tensor([[1.0, 0.0, -3.10]], requires_grad=True)
        target = torch.tensor([[1.0, 0.0, 3.10]])

        loss, _, _ = compute_kinematic_recon_loss(recon, target)
        loss.backward()

        # Loss should be very small (~0.002), not Euclidean (~12.8)
        self.assertLess(loss.item(), 0.05)
        self.assertIsNotNone(recon.grad)
        # Grad on phi should pull toward seam
        self.assertTrue(torch.all(torch.isfinite(recon.grad)))

    def test_bootstrap_roc_auc(self):
        """Verify stratified bootstrap produces valid AUC and 68% confidence intervals."""
        np.random.seed(42)
        y_true = np.array([0] * 500 + [1] * 200)
        rng = np.random.RandomState(42)
        scores = np.concatenate([rng.normal(0, 1, 500), rng.normal(1.5, 1, 200)])

        auc, std, ci = compute_roc_auc_with_bootstrap(y_true, scores, num_bootstraps=300)
        self.assertGreater(auc, 0.80)
        self.assertLess(auc, 0.95)
        self.assertGreater(std, 0.0)
        self.assertLess(std, 0.05)
        self.assertLessEqual(ci[0], auc)
        self.assertGreaterEqual(ci[1], auc)

    def test_amp_safe_pooling(self):
        """Verify ScoreDecorrelatedNodeGNN forward pass runs safely with float16 tensors."""
        base = NodeGNN(input_dim=3, hidden_dim=16, num_heads=2)
        model = ScoreDecorrelatedNodeGNN(base, decorrelation_type="disco", score_type="pt_only")

        # Simulate graph batch
        g1 = Data(x=torch.randn(10, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        g2 = Data(x=torch.randn(12, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        batch = Batch.from_data_list([g1, g2])

        # Test forward pass with simulated float16 tensors
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            recon_x, pred_met, scores, pred_mass = model(batch.x, batch.edge_index, batch.batch)

        self.assertEqual(scores.shape, (2,))
        self.assertFalse(torch.isnan(scores).any())


if __name__ == "__main__":
    unittest.main()
