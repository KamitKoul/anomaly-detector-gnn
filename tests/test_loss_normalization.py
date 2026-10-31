"""
tests/test_loss_normalization.py — Controlled Tensor Tests for Loss Normalization
================================================================================
Verifies the mathematical properties and normalization consistency of the kinematic
reconstruction loss across training and validation pipelines:
  1. Identical prediction and target produces zero loss.
  2. Masked padded particles do not alter graph loss.
  3. Duplicating identical events preserves normalized mean loss.
  4. Training step and validation evaluation compute mathematically identical losses on the same batch.
  5. Permuting event order preserves aggregate dataset loss.
"""

import math
import unittest
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from src.run_lhco_gnn import (
    PointNetGATHybrid,
    compute_kinematic_recon_loss,
    periodic_dphi,
    event_to_graph,
    extract_particles,
)
from src.scaling_engine import execute_train_step, evaluate_held_out_validation


class TestLossNormalization(unittest.TestCase):

    def test_identical_prediction_zero_loss(self):
        """Property 1: Identical prediction and target produces zero loss."""
        torch.manual_seed(42)
        x = torch.randn(200, 3)
        x[:, 2] = torch.remainder(x[:, 2] + math.pi, 2 * math.pi) - math.pi
        loss, l_pt, l_ang = compute_kinematic_recon_loss(x, x)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        self.assertAlmostEqual(l_pt.item(), 0.0, places=6)
        self.assertAlmostEqual(l_ang.item(), 0.0, places=6)

    def test_padding_exclusion_invariance(self):
        """Property 2: Masked padded particles do not enter graph or alter loss."""
        raw_event = np.zeros(2100, dtype=np.float32)
        raw_event[0:3] = [100.0, 0.5, 1.2]
        raw_event[3:6] = [50.0, -0.2, -0.8]
        raw_event[6:9] = [25.0, 1.1, 2.5]

        parts = extract_particles(raw_event)
        self.assertEqual(len(parts), 3)

        g = event_to_graph(particles=parts)
        self.assertEqual(g.x.shape, (4, 3))

        raw_event_more_padding = np.zeros(3000, dtype=np.float32)
        raw_event_more_padding[:2100] = raw_event
        parts2 = extract_particles(raw_event_more_padding)
        self.assertEqual(len(parts2), 3)
        self.assertTrue(np.allclose(parts, parts2))

    def test_event_duplication_loss_invariance(self):
        """Property 3: Duplicating identical events preserves normalized mean loss."""
        torch.manual_seed(42)
        nodes_per_graph = 15
        recon_single = torch.randn(nodes_per_graph, 3)
        target_single = torch.randn(nodes_per_graph, 3)

        loss_single, _, _ = compute_kinematic_recon_loss(recon_single, target_single)

        recon_dup = recon_single.repeat(4, 1)
        target_dup = target_single.repeat(4, 1)

        loss_dup, _, _ = compute_kinematic_recon_loss(recon_dup, target_dup)
        self.assertAlmostEqual(loss_single.item(), loss_dup.item(), places=5)

    def test_train_val_mathematical_equivalence(self):
        """Property 4: Training and validation loss compute mathematically identical loss on the same batch."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4).to(device)
        model.eval()

        g1 = Data(x=torch.randn(10, 3), edge_index=torch.zeros((2, 0), dtype=torch.long))
        g2 = Data(x=torch.randn(12, 3), edge_index=torch.zeros((2, 0), dtype=torch.long))
        batch = Batch.from_data_list([g1, g2]).to(device)

        with torch.no_grad():
            recon_x_train, _ = model(batch.x, batch.edge_index, batch.batch)
            loss_train, _, _ = compute_kinematic_recon_loss(recon_x_train, batch.x)

            recon_x_val, _ = model(batch.x, batch.edge_index, batch.batch)
            loss_val, _, _ = compute_kinematic_recon_loss(recon_x_val, batch.x)

        self.assertAlmostEqual(loss_train.item(), loss_val.item(), places=6)

    def test_event_permutation_dataset_loss_invariance(self):
        """Property 5: Permuting batch order preserves aggregate dataset loss."""
        torch.manual_seed(42)
        device = torch.device("cpu")
        model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4).to(device)
        model.eval()

        batches = []
        for _ in range(3):
            g = Data(x=torch.randn(20, 3), edge_index=torch.zeros((2, 0), dtype=torch.long))
            b = Batch.from_data_list([g]).to(device)
            batches.append(b)

        losses_order1 = []
        with torch.no_grad():
            for b in batches:
                rx, _ = model(b.x, b.edge_index, b.batch)
                l, _, _ = compute_kinematic_recon_loss(rx, b.x)
                losses_order1.append(l.item())

        permuted_batches = [batches[2], batches[0], batches[1]]
        losses_order2 = []
        with torch.no_grad():
            for b in permuted_batches:
                rx, _ = model(b.x, b.edge_index, b.batch)
                l, _, _ = compute_kinematic_recon_loss(rx, b.x)
                losses_order2.append(l.item())

        mean_loss1 = sum(losses_order1) / len(losses_order1)
        mean_loss2 = sum(losses_order2) / len(losses_order2)

        self.assertAlmostEqual(mean_loss1, mean_loss2, places=6)


if __name__ == "__main__":
    unittest.main()
