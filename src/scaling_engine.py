"""
src/scaling_engine.py — Shared Scientific Training Engine for Scaling Studies
=============================================================================
Provides core execution primitives for Experiment A (Fixed Epochs) and
Experiment B (Fixed Updates):
  1. Strict experimental control: Fixed PointNetGATHybrid (7,957 parameters),
     reconstruction-only objective, fixed Adam optimizer, gradient clipping (50.0).
  2. Provenance recording: Git commit, dataset manifest SHA-256, RNG states.
  3. Unsupervised validation: Checkpoints selected strictly on minimum background
     validation reconstruction loss. Zero signal leakage.
  4. Precise accounting: Event presentations, unique coverage, dropped batches.
"""

import sys
import os
import time
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent

from src.run_lhco_gnn import (
    PointNetGATHybrid,
    compute_kinematic_recon_loss,
    get_device,
)
from src.streaming_data_loader import ChunkedLHCODataset, create_streaming_dataloader


def get_git_commit() -> str:
    """Returns current git commit SHA if available."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def get_manifest_sha(manifest_path: Path) -> str:
    """Computes SHA-256 of the dataset manifest."""
    if not manifest_path.exists():
        return "missing"
    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def set_deterministic_seed(seed: int):
    """Sets deterministic random states across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_scaling_model(device: torch.device) -> PointNetGATHybrid:
    """Instantiates and verifies the fixed PointNetGATHybrid architecture."""
    model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 7957, f"Parameter budget mismatch: expected 7,957, got {param_count}"
    return model


class ScalingCheckpointManager:
    """Manages best and final checkpoint persistence with zero signal leakage."""

    def __init__(self, output_dir: Path, experiment_meta: Dict[str, Any]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_meta = experiment_meta
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.best_step = 0

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Any,
        epoch: int,
        step: int,
        val_loss: float,
        is_final: bool = False,
    ) -> bool:
        """Saves checkpoint if validation loss improves or if final."""
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_step = step

        # Capture complete RNG states for absolute bitwise restartability
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        payload = {
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "best_step": self.best_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "rng_states": rng_states,
            "meta": self.experiment_meta,
        }

        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(payload, best_path)

        if is_final:
            final_path = self.output_dir / "checkpoint_final.pt"
            torch.save(payload, final_path)

        return is_best


def execute_train_step(
    model: nn.Module,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any = None,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Executes a single forward, reconstruction loss, backward, and optimizer step."""
    optimizer.zero_grad()
    batch = batch.to(device, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        recon_x, _ = model(batch.x, batch.edge_index, batch.batch)
        loss, loss_pt, loss_ang = compute_kinematic_recon_loss(recon_x, batch.x)

    if use_amp and scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()

    grad_norm = float(total_norm.item() if hasattr(total_norm, "item") else total_norm)
    return float(loss.item()), grad_norm


def evaluate_held_out_validation(
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
) -> float:
    """Evaluates background reconstruction loss with torch.no_grad on validation partition."""
    model.eval()
    total_val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            recon_x, _ = model(batch.x, batch.edge_index, batch.batch)
            loss, _, _ = compute_kinematic_recon_loss(recon_x, batch.x)
            total_val_loss += float(loss.item())
            num_batches += 1

    model.train()
    return total_val_loss / max(num_batches, 1)
