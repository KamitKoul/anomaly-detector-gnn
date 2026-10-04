"""
test_controlled_pipeline.py — Controlled Pipeline & Model Unit Tests
===================================================================
Tests model factory instantiation, checkpoint compatibility, graph construction
configuration fidelity, flat feature extraction, and dry-run execution.
"""

import os
import sys
import unittest
from pathlib import Path

import torch
import numpy as np

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_framework import load_config
from src.run_lhco_gnn import (
    create_model,
    event_to_graph,
    event_to_flat_features,
    stream_filtered_events,
    BaselineMLP,
    NodeGNN,
)
from src.adversarial_unlearning_ae import AdversarialNodeGNN
from src.train_experiment import run_training_experiment


class TestControlledPipeline(unittest.TestCase):

    def test_model_factory_types(self):
        """Verify create_model returns exact expected types for all 4 configs."""
        configs = {
            "baseline_mlp": BaselineMLP,
            "gat_standalone": NodeGNN,
            "gat_met_production": NodeGNN,
            "gat_met_decorrelation": AdversarialNodeGNN,
        }
        for name, expected_cls in configs.items():
            cfg_path = REPO_ROOT / "configs" / "architecture" / f"{name}.json"
            cfg = load_config(cfg_path)
            model = create_model(cfg)
            self.assertIsInstance(
                model,
                expected_cls,
                f"Config {name} did not instantiate {expected_cls.__name__}",
            )

    def test_baseline_mlp_strict_load(self):
        """Verify BaselineMLP matches models/autoencoder_bg.pt state dict exactly."""
        ckpt_path = REPO_ROOT / "models" / "autoencoder_bg.pt"
        if ckpt_path.exists():
            model = BaselineMLP()
            sd = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(sd, strict=True)
            dummy_in = torch.randn(4, 64)
            out = model(dummy_in)
            self.assertEqual(out.shape, (4, 64))

    def test_nodegnn_strict_load(self):
        """Verify NodeGNN matches models/discovery_model.pt state dict exactly."""
        ckpt_path = REPO_ROOT / "models" / "discovery_model.pt"
        if ckpt_path.exists():
            model = NodeGNN()
            sd = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(sd, strict=True)

    def test_adversarial_nodegnn_strict_load(self):
        """Verify AdversarialNodeGNN matches models/unlearning_model.pt state dict exactly."""
        ckpt_path = REPO_ROOT / "models" / "unlearning_model.pt"
        if ckpt_path.exists():
            model = AdversarialNodeGNN()
            sd = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(sd, strict=True)

    def test_event_to_flat_features(self):
        """Verify flat feature extraction truncates and log-scales pT properly."""
        dummy_event = np.zeros(2100, dtype=np.float32)
        dummy_event[0] = 100.0  # particle 0 pt
        dummy_event[1] = 0.5    # particle 0 eta
        dummy_event[2] = -1.2   # particle 0 phi
        dummy_event[3] = 50.0   # particle 1 pt

        feat = event_to_flat_features(dummy_event, input_dim=64, log_pt=True)
        self.assertEqual(feat.shape, (64,))
        self.assertAlmostEqual(feat[0].item(), np.log1p(100.0), places=4)
        self.assertAlmostEqual(feat[1].item(), 0.5, places=4)
        self.assertAlmostEqual(feat[2].item(), -1.2, places=4)
        self.assertAlmostEqual(feat[3].item(), np.log1p(50.0), places=4)

    def test_event_to_graph_met_node_toggle(self):
        """Verify event_to_graph includes or excludes MET node based on flag."""
        dummy_event = np.zeros(2100, dtype=np.float32)
        # 2 visible particles
        dummy_event[0] = 100.0; dummy_event[1] = 0.5; dummy_event[2] = 0.1
        dummy_event[3] = 80.0; dummy_event[4] = 0.6; dummy_event[5] = 0.2

        # With MET node
        g_with = event_to_graph(dummy_event, include_met_node=True)
        # Without MET node
        g_without = event_to_graph(dummy_event, include_met_node=False)

        # 2 particles + 1 MET node = 3 nodes vs 2 nodes
        self.assertEqual(g_with.x.shape[0], 3)
        self.assertEqual(g_without.x.shape[0], 2)

    def test_dry_run_architecture_configs(self):
        """Verify dry-run succeeds for all four architecture configs."""
        configs = [
            "baseline_mlp",
            "gat_standalone",
            "gat_met_production",
            "gat_met_decorrelation",
        ]
        for name in configs:
            cfg_path = REPO_ROOT / "configs" / "architecture" / f"{name}.json"
            ret = run_training_experiment(
                config_path=cfg_path,
                dry_run=True,
                override_output_dir=str(REPO_ROOT / "experiments" / "test_dry_runs" / name),
            )
            self.assertEqual(ret, 0, f"Dry-run failed for {name}")

    def test_device_selection_priority(self):
        """Verify device selection priority: CUDA -> MPS -> CPU."""
        from unittest import mock
        from src.run_lhco_gnn import get_device

        # Case 1: CUDA available -> selects CUDA
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.get_device_name", return_value="Tesla T4"):
            d = get_device(verbose=False)
            self.assertEqual(d.type, "cuda")

        # Case 2: CUDA unavailable, MPS available -> selects MPS
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.backends.mps.is_available", return_value=True):
            d = get_device(verbose=False)
            self.assertEqual(d.type, "mps")

        # Case 3: CUDA and MPS unavailable -> selects CPU
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.backends.mps.is_available", return_value=False):
            d = get_device(verbose=False)
            self.assertEqual(d.type, "cpu")

    def test_device_memory_diagnostics(self):
        """Verify get_memory_diagnostics returns correct schema for CPU and CUDA."""
        from unittest import mock
        from src.train_experiment import get_memory_diagnostics

        cpu_diag = get_memory_diagnostics(torch.device("cpu"))
        self.assertEqual(cpu_diag["device_type"], "cpu")

        with mock.patch("torch.cuda.current_device", return_value=0), \
             mock.patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-40GB"), \
             mock.patch("torch.cuda.memory_allocated", return_value=1024 * 1024 * 50), \
             mock.patch("torch.cuda.memory_reserved", return_value=1024 * 1024 * 100), \
             mock.patch("torch.cuda.max_memory_allocated", return_value=1024 * 1024 * 75):
            cuda_diag = get_memory_diagnostics(torch.device("cuda:0"))
            self.assertEqual(cuda_diag["device_type"], "cuda")
            self.assertEqual(cuda_diag["device_name"], "NVIDIA A100-SXM4-40GB")
            self.assertEqual(cuda_diag["allocated_mb"], 50.0)
            self.assertEqual(cuda_diag["reserved_mb"], 100.0)
            self.assertEqual(cuda_diag["max_allocated_mb"], 75.0)


if __name__ == "__main__":
    unittest.main()
