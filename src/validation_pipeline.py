"""
validation_pipeline.py — Independent Validation and Unsupervised Checkpoint Selection
======================================================================================
Implements rigorous, statistically sound in-training validation for unsupervised
anomaly detectors:
  1. Evaluates model on an independent held-out background validation set.
  2. Tracks val_recon, val_met, val_disco, and val_total.
  3. Selects best checkpoint based strictly on background reconstruction loss
     (NEVER using signal evaluation AUC or test set metrics).
  4. Manages early stopping and persistent checkpointing (best vs last).
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.run_lhco_gnn import (
    compute_kinematic_recon_loss,
    NodeGNN,
    PointNetGATHybrid,
)
from src.adversarial_unlearning_ae import (
    ScoreDecorrelatedNodeGNN,
    AdversarialNodeGNN,
    compute_distance_correlation,
)


def evaluate_validation_epoch(
    model: nn.Module,
    val_loader: Any,
    device: torch.device,
    config: Any,
) -> Dict[str, float]:
    """
    Computes validation metrics on held-out background events with zero gradient tracking.
    Guarantees no signal data is accessed during unsupervised model evaluation.
    """
    model.eval()
    total_recon = 0.0
    total_met = 0.0
    total_adv = 0.0
    num_batches = 0

    lambda_met = getattr(config.objectives, "met_loss_weight", 0.0) if hasattr(config, "objectives") else 0.0
    lambda_adv = getattr(config.objectives, "adv_loss_weight", 0.0) if hasattr(config, "objectives") else 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            if isinstance(model, ScoreDecorrelatedNodeGNN):
                recon_x, pred_met, scores, _ = model(batch.x, batch.edge_index, batch.batch)
                loss_recon, _, _ = compute_kinematic_recon_loss(recon_x, batch.x)
                loss_met = F.mse_loss(pred_met, batch.y.view(-1, 2)) if lambda_met > 0 else torch.tensor(0.0)
                
                if lambda_adv > 0 and hasattr(batch, "mass"):
                    loss_adv = compute_distance_correlation(scores, batch.mass.view(-1))
                else:
                    loss_adv = torch.tensor(0.0)
            elif isinstance(model, (NodeGNN, PointNetGATHybrid)):
                recon_x, pred_met = model(batch.x, batch.edge_index, batch.batch)
                loss_recon, _, _ = compute_kinematic_recon_loss(recon_x, batch.x)
                loss_met = F.mse_loss(pred_met, batch.y.view(-1, 2)) if lambda_met > 0 else torch.tensor(0.0)
                loss_adv = torch.tensor(0.0)
            else:
                recon_x = model(batch.x)
                loss_recon, _, _ = compute_kinematic_recon_loss(recon_x, batch.x)
                loss_met = torch.tensor(0.0)
                loss_adv = torch.tensor(0.0)

            total_recon += float(loss_recon.item())
            total_met += float(loss_met.item())
            total_adv += float(loss_adv.item())
            num_batches += 1

    n_b = max(num_batches, 1)
    val_recon = total_recon / n_b
    val_met = total_met / n_b
    val_adv = total_adv / n_b
    val_total = val_recon + lambda_met * val_met + lambda_adv * val_adv

    return {
        "val_loss": val_total,
        "val_loss_recon": val_recon,
        "val_loss_met": val_met,
        "val_loss_adv": val_adv,
    }


class CheckpointManager:
    """
    Manages model checkpoint persistence during training:
      - Tracks minimum validation metric (default: val_loss_recon).
      - Saves 'checkpoint_best.pt' when improved.
      - Saves 'checkpoint_last.pt' at final epoch.
      - Handles patience-based early stopping.
    """

    def __init__(
        self,
        output_dir: Path,
        metric_name: str = "val_loss_recon",
        patience: int = 10,
        min_delta: float = 1e-5,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.patience = patience
        self.min_delta = min_delta

        self.best_score = float("inf")
        self.best_epoch = 0
        self.stagnant_epochs = 0
        self.should_stop = False

    def step(self, model: nn.Module, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Processes an epoch's metrics. Returns True if a new best checkpoint was saved.
        """
        current_score = metrics.get(self.metric_name, float("inf"))
        improved = (self.best_score - current_score) > self.min_delta

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.stagnant_epochs = 0
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(model.state_dict(), best_path)
            return True
        else:
            self.stagnant_epochs += 1
            if self.stagnant_epochs >= self.patience:
                self.should_stop = True
            return False

    def save_last(self, model: nn.Module):
        """Saves final checkpoint regardless of validation score."""
        last_path = self.output_dir / "checkpoint_last.pt"
        torch.save(model.state_dict(), last_path)
