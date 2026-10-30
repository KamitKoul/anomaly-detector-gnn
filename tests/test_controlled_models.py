"""
test_controlled_models.py — Unit Tests for Controlled Graph Structure Architectures
===================================================================================
Tests NodeWiseAE (Model C) and DeepSetsPointNet (Model D) for forward, backward,
parameter counts, and model factory instantiation.
"""

import unittest
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from src.run_lhco_gnn import (
    NodeGNN,
    BaselineMLP,
    NodeWiseAE,
    DeepSetsPointNet,
    create_model,
)


class TestControlledModels(unittest.TestCase):

    def setUp(self):
        # Create a dummy batch of 2 graphs
        g1 = Data(
            x=torch.randn(10, 3),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.tensor([50.0, -20.0]),
        )
        g2 = Data(
            x=torch.randn(15, 3),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            y=torch.tensor([10.0, 30.0]),
        )
        self.batch = Batch.from_data_list([g1, g2])

    def test_nodewise_ae_forward_and_params(self):
        """Verify NodeWiseAE forward pass, output shapes, and parameter budget."""
        model = NodeWiseAE(input_dim=3, hidden_dim=32, wide_dim=128)
        recon_x, pred_met = model(self.batch.x, self.batch.edge_index, self.batch.batch)
        
        self.assertEqual(recon_x.shape, (25, 3))
        self.assertEqual(pred_met.shape, (2, 2))
        
        # Backward pass check
        loss = F.mse_loss(recon_x, self.batch.x) + 0.1 * F.mse_loss(pred_met, self.batch.y.view(-1, 2))
        loss.backward()
        
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 5000)
        self.assertLess(param_count, 6500)
        print(f"[TEST] NodeWiseAE parameter count: {param_count}")

    def test_deepsets_pointnet_forward_and_params(self):
        """Verify DeepSetsPointNet forward pass, output shapes, and parameter budget."""
        model = DeepSetsPointNet(input_dim=3, hidden_dim=32)
        recon_x, pred_met = model(self.batch.x, self.batch.edge_index, self.batch.batch)
        
        self.assertEqual(recon_x.shape, (25, 3))
        self.assertEqual(pred_met.shape, (2, 2))
        
        # Backward pass check
        loss = F.mse_loss(recon_x, self.batch.x) + 0.1 * F.mse_loss(pred_met, self.batch.y.view(-1, 2))
        loss.backward()
        
        param_count = sum(p.numel() for p in model.parameters())
        self.assertGreater(param_count, 5000)
        self.assertLess(param_count, 6500)
        print(f"[TEST] DeepSetsPointNet parameter count: {param_count}")

    def test_model_factory(self):
        """Verify create_model creates correct instances from config dicts."""
        cfg_nodewise = {"model": {"architecture_type": "node_wise_ae", "input_dim": 3, "hidden_dim": 32}}
        m_nw = create_model(cfg_nodewise)
        self.assertIsInstance(m_nw, NodeWiseAE)

        cfg_deepsets = {"model": {"architecture_type": "deepsets_pointnet", "input_dim": 3, "hidden_dim": 32}}
        m_ds = create_model(cfg_deepsets)
        self.assertIsInstance(m_ds, DeepSetsPointNet)


if __name__ == "__main__":
    unittest.main()
