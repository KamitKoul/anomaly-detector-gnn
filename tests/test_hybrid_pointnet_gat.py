"""
test_hybrid_pointnet_gat.py — Unit Tests for Hybrid PointNet-GAT Architecture
=============================================================================
Tests PointNetGATHybrid and its score-decorrelated wrapper:
  1. Forward pass shapes (recon_x, pred_met)
  2. Backward gradient flow through both GAT and PointNet branches
  3. Parameter budget verification (~7-8k parameters)
  4. Model factory instantiation and score decorrelation wrapping
"""

import unittest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from src.run_lhco_gnn import (
    PointNetGATHybrid,
    create_model,
)
from src.adversarial_unlearning_ae import ScoreDecorrelatedNodeGNN


class TestHybridPointNetGAT(unittest.TestCase):

    def setUp(self):
        # Create a dummy batch of 2 collision graphs with edges
        edge_index_1 = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=torch.long)
        edge_index_2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        g1 = Data(
            x=torch.randn(10, 3),
            edge_index=edge_index_1,
            y=torch.tensor([45.0, -15.0]),
            mass=torch.tensor(3500.0),
        )
        g2 = Data(
            x=torch.randn(8, 3),
            edge_index=edge_index_2,
            y=torch.tensor([12.0, 28.0]),
            mass=torch.tensor(3600.0),
        )
        self.batch = Batch.from_data_list([g1, g2])

    def test_forward_pass_and_shapes(self):
        """Verify output tensor shapes from both reconstruction and MET heads."""
        model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4)
        recon_x, pred_met = model(self.batch.x, self.batch.edge_index, self.batch.batch)

        self.assertEqual(recon_x.shape, (18, 3))
        self.assertEqual(pred_met.shape, (2, 2))

    def test_gradient_flow_both_branches(self):
        """Verify backpropagation updates weights in both GAT and PointNet branches."""
        model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4)
        recon_x, pred_met = model(self.batch.x, self.batch.edge_index, self.batch.batch)

        loss = F.mse_loss(recon_x, self.batch.x) + 0.1 * F.mse_loss(pred_met, self.batch.y.view(-1, 2))
        loss.backward()

        # Check GAT conv1 gradients
        self.assertIsNotNone(model.conv1.lin.weight.grad)
        self.assertGreater(model.conv1.lin.weight.grad.abs().sum().item(), 0.0)

        # Check PointNet point_phi gradients
        self.assertIsNotNone(model.point_phi[0].weight.grad)
        self.assertGreater(model.point_phi[0].weight.grad.abs().sum().item(), 0.0)

        # Check Decoder gradients
        self.assertIsNotNone(model.decoder[0].weight.grad)
        self.assertGreater(model.decoder[0].weight.grad.abs().sum().item(), 0.0)

    def test_parameter_budget(self):
        """Verify total parameters are within the ~7-8.5k budget."""
        model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[TEST] PointNetGATHybrid parameter count: {param_count}")
        self.assertGreater(param_count, 6500)
        self.assertLess(param_count, 9000)

    def test_model_factory_standalone(self):
        """Verify create_model instantiates PointNetGATHybrid correctly."""
        cfg = {"model": {"architecture_type": "pointnet_gat_hybrid", "input_dim": 3, "hidden_dim": 32, "num_heads": 4}}
        model = create_model(cfg)
        self.assertIsInstance(model, PointNetGATHybrid)

    def test_model_factory_score_decorrelation(self):
        """Verify create_model wraps PointNetGATHybrid with ScoreDecorrelatedNodeGNN."""
        cfg = {
            "model": {
                "architecture_type": "score_decorrelated_hybrid",
                "input_dim": 3,
                "hidden_dim": 32,
                "num_heads": 4,
                "decorrelation_type": "disco",
                "score_type": "weighted",
            }
        }
        model = create_model(cfg)
        self.assertIsInstance(model, ScoreDecorrelatedNodeGNN)
        self.assertIsInstance(model.base_model, PointNetGATHybrid)

        # Test forward pass with score computation
        recon_x, pred_met, scores, pred_mass = model(self.batch.x, self.batch.edge_index, self.batch.batch)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(recon_x.shape, (18, 3))


if __name__ == "__main__":
    unittest.main()
