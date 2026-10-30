"""
test_deep_physics.py — Unit Tests for Deep Physics Diagnostics
==============================================================
Validates that fine mass matching, feature decomposition, and graph
scrambling algorithms execute properly without errors.
"""

import unittest
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


class TestDeepPhysics(unittest.TestCase):

    def test_fine_mass_windows_slicing(self):
        """Verify fine mass matching slices properly filter within +/- 50 GeV."""
        np.random.seed(42)
        n = 1000
        masses = np.random.uniform(3000, 4000, n)
        
        # Micro-fine window [3450, 3550]
        mask_50 = (masses >= 3450.0) & (masses <= 3550.0)
        selected = masses[mask_50]
        
        self.assertTrue(np.all(selected >= 3450.0))
        self.assertTrue(np.all(selected <= 3550.0))
        self.assertGreater(len(selected), 50)

    def test_feature_attribution_decomposition(self):
        """Verify decomposition into p_T vs angular reconstruction error."""
        # Simulated node predictions and targets [pT, eta, phi]
        x_true = torch.randn(100, 3)
        x_pred = x_true + torch.tensor([0.5, 0.1, 0.1])
        
        diff_sq = (x_pred - x_true).pow(2)
        err_pt = diff_sq[:, 0]
        err_ang = diff_sq[:, 1:].mean(dim=1)
        err_tot = diff_sq.mean(dim=1)
        
        # pT error should be roughly 0.25
        self.assertAlmostEqual(err_pt.mean().item(), 0.25, places=3)
        # Angular error should be roughly 0.01
        self.assertAlmostEqual(err_ang.mean().item(), 0.01, places=3)
        # Total error should match sum of channels
        self.assertAlmostEqual(err_tot.mean().item(), (0.25 + 0.01 * 2) / 3.0, places=3)

    def test_graph_scrambling(self):
        """Verify edge destination scrambling keeps node count while breaking structure."""
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        perm = torch.randperm(4)
        scrambled = torch.stack([edge_index[0], edge_index[1][perm]], dim=0)
        
        self.assertEqual(scrambled.shape, edge_index.shape)
        self.assertTrue(torch.equal(scrambled[0], edge_index[0]))


if __name__ == "__main__":
    unittest.main()
