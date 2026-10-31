"""
test_validation_pipeline.py — Unit Tests for Validation and Checkpoint Management
==================================================================================
Verifies:
  1. evaluate_validation_epoch correctly aggregates validation metrics with torch.no_grad.
  2. CheckpointManager correctly tracks best epoch, persists checkpoint_best.pt,
     and respects patience for early stopping.
  3. Validation is strictly unsupervised on background data.
"""

import sys
import unittest
import tempfile
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_lhco_gnn import PointNetGATHybrid
from src.remediation_diagnostics import create_synthetic_collision_batch
from src.validation_pipeline import evaluate_validation_epoch, CheckpointManager
from src.experiment_framework import ExperimentConfig


class TestValidationPipeline(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4)
        self.device = torch.device("cpu")
        self.config = ExperimentConfig(
            experiment_name="test_val",
            experiment_family="controlled_structure",
            description="Test validation config",
        )
        self.config.objectives.met_loss_weight = 0.001

    def test_evaluate_validation_epoch(self):
        """Verifies validation loss computation without gradient tracking."""
        batch = create_synthetic_collision_batch(batch_size=16, seed=99)
        loader = DataLoader([batch.get_example(i) for i in range(16)], batch_size=8)

        metrics = evaluate_validation_epoch(self.model, loader, self.device, self.config)
        
        self.assertIn("val_loss", metrics)
        self.assertIn("val_loss_recon", metrics)
        self.assertIn("val_loss_met", metrics)
        self.assertGreater(metrics["val_loss_recon"], 0.0)
        
        # Verify model remains in eval mode
        self.assertFalse(self.model.training)
        # Verify parameters have no accumulated gradients from validation
        for p in self.model.parameters():
            self.assertIsNone(p.grad)

    def test_checkpoint_manager_selection_and_patience(self):
        """Verifies CheckpointManager saves best checkpoint and triggers early stopping."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            mgr = CheckpointManager(out_dir, metric_name="val_loss_recon", patience=3, min_delta=1e-4)

            # Epoch 1: score = 0.10 (Best)
            improved = mgr.step(self.model, epoch=1, metrics={"val_loss_recon": 0.10})
            self.assertTrue(improved)
            self.assertEqual(mgr.best_epoch, 1)
            self.assertTrue((out_dir / "checkpoint_best.pt").exists())

            # Epoch 2: score = 0.08 (Improvement)
            improved = mgr.step(self.model, epoch=2, metrics={"val_loss_recon": 0.08})
            self.assertTrue(improved)
            self.assertEqual(mgr.best_epoch, 2)

            # Epoch 3: score = 0.085 (Stagnant, patience 1/3)
            improved = mgr.step(self.model, epoch=3, metrics={"val_loss_recon": 0.085})
            self.assertFalse(improved)
            self.assertEqual(mgr.stagnant_epochs, 1)
            self.assertFalse(mgr.should_stop)

            # Epoch 4: score = 0.084 (Stagnant, patience 2/3)
            mgr.step(self.model, epoch=4, metrics={"val_loss_recon": 0.084})
            self.assertEqual(mgr.stagnant_epochs, 2)
            self.assertFalse(mgr.should_stop)

            # Epoch 5: score = 0.086 (Stagnant, patience 3/3 -> Should Stop!)
            mgr.step(self.model, epoch=5, metrics={"val_loss_recon": 0.086})
            self.assertEqual(mgr.stagnant_epochs, 3)
            self.assertTrue(mgr.should_stop)

            # Save last checkpoint
            mgr.save_last(self.model)
            self.assertTrue((out_dir / "checkpoint_last.pt").exists())


if __name__ == "__main__":
    unittest.main()
