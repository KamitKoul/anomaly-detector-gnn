"""
test_score_decorrelation.py — Unit Tests for Score-Level Decorrelation Mechanics
================================================================================
Verifies that:
1. compute_distance_correlation behaves correctly on independent vs correlated variables.
2. ScoreGRLAdversary backpropagates reversed gradients into GAT decoder weights.
3. ScoreDecorrelatedNodeGNN computes scalar scores across score_type formulations (pt_only, weighted, full).
4. DisCo loss backpropagates directly through the decoder weights.
5. create_model correctly builds score_decorrelated_gat from config dict.
"""

import unittest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from src.run_lhco_gnn import NodeGNN, create_model
from src.adversarial_unlearning_ae import (
    compute_distance_correlation,
    ScoreGRLAdversary,
    ScoreDecorrelatedNodeGNN,
)


class TestScoreDecorrelation(unittest.TestCase):

    def test_distance_correlation_properties(self):
        """Verify dCor^2 is near 1 for linear relation and near 0 for independent variables."""
        torch.manual_seed(42)
        n = 200
        x = torch.randn(n)

        # Perfect linear relation: dCor^2 should be 1.0
        y_linear = 3.0 * x + 2.0
        dcor_lin = compute_distance_correlation(x, y_linear)
        self.assertAlmostEqual(dcor_lin.item(), 1.0, places=3)

        # Independent uniform variables: dCor^2 should be very small
        y_indep = torch.rand(n)
        dcor_indep = compute_distance_correlation(x, y_indep)
        self.assertLess(dcor_indep.item(), 0.1)

    def test_score_grl_gradient_flow(self):
        """Verify that gradients from score adversary backpropagate into base model weights."""
        base_model = NodeGNN(input_dim=3, hidden_dim=32, num_heads=2)
        model = ScoreDecorrelatedNodeGNN(base_model, decorrelation_type="grl", score_type="pt_only")

        g1 = Data(x=torch.randn(10, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        g2 = Data(x=torch.randn(12, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        batch = Batch.from_data_list([g1, g2])

        recon_x, pred_met, scores, pred_mass = model(batch.x, batch.edge_index, batch.batch, alpha=1.0)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(pred_mass.shape, (2, 1))

        # Simulate target mass [3.5, 3.2] TeV
        target_mass = torch.tensor([[3.5], [3.2]])
        loss = (recon_x - batch.x).pow(2).mean() + 0.1 * (pred_mass - target_mass).pow(2).mean()
        loss.backward()

        # Verify base_model.decoder received gradients
        for p in base_model.decoder.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.any(p.grad != 0.0))

    def test_disco_gradient_flow(self):
        """Verify that distance correlation penalty backpropagates into base model weights."""
        base_model = NodeGNN(input_dim=3, hidden_dim=32, num_heads=2)
        model = ScoreDecorrelatedNodeGNN(base_model, decorrelation_type="disco", score_type="pt_only")

        g1 = Data(x=torch.randn(10, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        g2 = Data(x=torch.randn(12, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        g3 = Data(x=torch.randn(14, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        g4 = Data(x=torch.randn(8, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        batch = Batch.from_data_list([g1, g2, g3, g4])

        recon_x, pred_met, scores, pred_mass = model(batch.x, batch.edge_index, batch.batch)
        self.assertEqual(scores.shape, (4,))
        self.assertIsNone(pred_mass)

        # Compute DisCo penalty with simulated masses
        masses = torch.tensor([3.5, 3.2, 3.8, 3.1])
        loss_disco = compute_distance_correlation(scores, masses)
        loss = (recon_x - batch.x).pow(2).mean() + 1.0 * loss_disco
        loss.backward()

        # Verify decoder received gradients
        for p in base_model.decoder.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.any(p.grad != 0.0))

    def test_score_formulations(self):
        """Verify pt_only, weighted, and full score computations produce valid shapes."""
        base_model = NodeGNN(input_dim=3, hidden_dim=16, num_heads=2)
        g1 = Data(x=torch.randn(10, 3), edge_index=torch.zeros((2, 0), dtype=torch.long), y=torch.tensor([0.0, 0.0]))
        batch = Batch.from_data_list([g1])

        for stype in ["pt_only", "weighted", "full", "ang_only"]:
            m = ScoreDecorrelatedNodeGNN(base_model, decorrelation_type="disco", score_type=stype)
            _, _, sc, _ = m(batch.x, batch.edge_index, batch.batch)
            self.assertEqual(sc.shape, (1,))
            self.assertFalse(torch.isnan(sc).any())

    def test_create_model_factory(self):
        """Verify create_model builds ScoreDecorrelatedNodeGNN correctly."""
        cfg = {
            "model": {
                "architecture_type": "score_decorrelated_gat",
                "input_dim": 3,
                "hidden_dim": 32,
                "num_heads": 4,
                "decorrelation_type": "disco",
                "score_type": "pt_only",
            }
        }
        model = create_model(cfg)
        self.assertIsInstance(model, ScoreDecorrelatedNodeGNN)
        self.assertEqual(model.decorrelation_type, "disco")
        self.assertEqual(model.score_type, "pt_only")


if __name__ == "__main__":
    unittest.main()
