"""
remediation_diagnostics.py — Scientific Diagnostic Utilities for Gradient Auditing & Testing
=============================================================================================
Contains robust, self-contained helpers for measuring:
  - Reconstruction vs Auxiliary (MET, DisCo) gradients on shared and global parameters.
  - Generating deterministic synthetic collision batches for testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
from torch_geometric.data import Data, Batch

from src.run_lhco_gnn import compute_kinematic_recon_loss, build_deltaR_edges


def create_synthetic_collision_batch(
    batch_size: int = 32,
    num_particles_range: Tuple[int, int] = (150, 250),
    seed: int = 42,
) -> Batch:
    """Generates physically realistic PyG collision graphs matching LHCO dijet kinematics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    graphs = []
    for b in range(batch_size):
        n_p = np.random.randint(num_particles_range[0], num_particles_range[1])
        # Particle pT: power-law / exponential distribution (GeV)
        pt = np.random.exponential(scale=25.0, size=n_p) + 1.0  # min 1 GeV
        # Two hard leading particles mimicking dijet cores
        pt[0] = np.random.uniform(400.0, 1200.0)
        pt[1] = np.random.uniform(300.0, 1000.0)

        eta = np.random.uniform(-2.5, 2.5, size=n_p)
        phi = np.random.uniform(-np.pi, np.pi, size=n_p)

        # Invert MET
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        met_x = float(-np.sum(px))
        met_y = float(-np.sum(py))
        met = float(np.sqrt(met_x**2 + met_y**2))

        # Add MET node
        phi_met = float(np.arctan2(met_y, met_x)) if met > 0 else 0.0
        all_particles = np.vstack([np.column_stack([pt, eta, phi]), [[met, 0.0, phi_met]]])

        # Edges
        edge_index = build_deltaR_edges(all_particles, threshold=0.4)

        # Log1p pT scaling
        features = all_particles.copy()
        features[:, 0] = np.log1p(features[:, 0])

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor([met_x, met_y], dtype=torch.float32)
        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return Batch.from_data_list(graphs)


def measure_met_gradients(
    model: nn.Module,
    batch: Batch,
    lambda_met: float = 0.1,
    met_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Measures gradients of L_recon and (lambda_met * L_MET) independently,
    computing their norms and cosine alignment on shared parameters and full model.
    """
    model.zero_grad()

    recon_x, pred_met = model(batch.x, batch.edge_index, batch.batch)

    # 1. Reconstruction loss
    loss_recon, loss_pt, loss_ang = compute_kinematic_recon_loss(recon_x, batch.x)
    loss_recon.backward(retain_graph=True)

    # Extract recon grads
    grad_recon_shared = []
    grad_recon_all = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            g = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            grad_recon_all.append(g.view(-1))
            if "point_phi" in name:
                grad_recon_shared.append(g.view(-1))

    grad_recon_shared = torch.cat(grad_recon_shared)
    grad_recon_all = torch.cat(grad_recon_all)

    # 2. MET loss
    model.zero_grad()
    target_met = batch.y.view(-1, 2) * met_scale
    loss_met_raw = F.mse_loss(pred_met * met_scale, target_met)
    loss_met_weighted = lambda_met * loss_met_raw
    loss_met_weighted.backward()

    grad_met_shared = []
    grad_met_all = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            g = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            grad_met_all.append(g.view(-1))
            if "point_phi" in name:
                grad_met_shared.append(g.view(-1))

    grad_met_shared = torch.cat(grad_met_shared)
    grad_met_all = torch.cat(grad_met_all)

    # Norms
    norm_recon_shared = float(torch.norm(grad_recon_shared).item())
    norm_met_shared = float(torch.norm(grad_met_shared).item())

    norm_recon_all = float(torch.norm(grad_recon_all).item())
    norm_met_all = float(torch.norm(grad_met_all).item())

    # Cosine alignments
    eps = 1e-9
    cos_shared = float((torch.dot(grad_recon_shared, grad_met_shared) / (norm_recon_shared * norm_met_shared + eps)).item())
    cos_all = float((torch.dot(grad_recon_all, grad_met_all) / (norm_recon_all * norm_met_all + eps)).item())

    ratio_shared = norm_met_shared / (norm_recon_shared + eps)
    ratio_all = norm_met_all / (norm_recon_all + eps)

    return {
        "loss_recon": float(loss_recon.item()),
        "loss_met_raw": float(loss_met_raw.item()),
        "norm_recon_shared": norm_recon_shared,
        "norm_met_shared": norm_met_shared,
        "ratio_shared": ratio_shared,
        "cos_shared": cos_shared,
        "norm_recon_all": norm_recon_all,
        "norm_met_all": norm_met_all,
        "ratio_all": ratio_all,
        "cos_all": cos_all,
    }
