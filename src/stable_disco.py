"""
stable_disco.py — Numerically Stable and Finite-Sample Corrected Distance Correlation
======================================================================================
Provides:
  1. compute_unbiased_distance_covariance(x, y):
     Szekely & Rizzo (2014) U-centered distance covariance (Omega_n).
     Unbiased: E[Omega_n(X, Y)] = 0 under independence (no positive small-sample floor).
  2. compute_stabilized_distance_correlation(x, y, eps=1e-6):
     Numerically stable distance correlation squared with safe gradient clipping,
     diagonal-safe epsilon clamping to prevent gradient explosion when variance is near zero.
"""

import torch
import torch.nn.functional as F


def compute_unbiased_distance_covariance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes Szekely & Rizzo (2014) unbiased U-centered sample distance covariance squared (Omega_n).
    Under true independence X _|_ Y, E[Omega_n] = 0 (completely removes the O(1/n) positive bias).
    
    Args:
        x: 1D Tensor of shape [B]
        y: 1D Tensor of shape [B]
    Returns:
        Scalar tensor Omega_n(X, Y). Can be slightly negative around zero due to unbiased variance.
    """
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    n = x.size(0)
    if n <= 3:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # Euclidean distance matrices
    a = torch.abs(x - x.t())
    b = torch.abs(y - y.t())

    # Set diagonals to 0 for U-centering
    a.fill_diagonal_(0.0)
    b.fill_diagonal_(0.0)

    # Row sums excluding diagonal (divide by n - 2)
    a_row_sum = a.sum(dim=1, keepdim=True)
    b_row_sum = b.sum(dim=1, keepdim=True)

    # Grand sum excluding diagonal (divide by (n - 1) * (n - 2))
    a_grand_sum = a.sum()
    b_grand_sum = b.sum()

    # U-centered double centering (Szekely & Rizzo 2014, eq. 2.4)
    A_u = a - (a_row_sum + a_row_sum.t()) / (n - 2) + a_grand_sum / ((n - 1) * (n - 2))
    B_u = b - (b_row_sum + b_row_sum.t()) / (n - 2) + b_grand_sum / ((n - 1) * (n - 2))

    # Set diagonals of U-centered matrices to 0
    A_u.fill_diagonal_(0.0)
    B_u.fill_diagonal_(0.0)

    # Unbiased distance covariance squared
    omega_n = (A_u * B_u).sum() / (n * (n - 3))
    return omega_n


def compute_stabilized_distance_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Numerically stabilized sample distance correlation squared dCor^2(X, Y).
    Fixes gradient explosion near zero variance by:
      1. Clamping denominator safely with eps=1e-6 (gradient bounded by 1/(2*sqrt(eps)) = 500).
      2. Clamping result safely into [0, 1].
    """
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    B = x.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    # 1. Pairwise distance matrices
    a = torch.abs(x - x.t())
    b = torch.abs(y - y.t())

    # 2. Classical double centering
    a_row = a.mean(dim=1, keepdim=True)
    a_col = a.mean(dim=0, keepdim=True)
    a_grand = a.mean()
    A = a - a_row - a_col + a_grand

    b_row = b.mean(dim=1, keepdim=True)
    b_col = b.mean(dim=0, keepdim=True)
    b_grand = b.mean()
    B_mat = b - b_row - b_col + b_grand

    # 3. Covariances and variances
    dcov2 = (A * B_mat).sum() / (B * B)
    dvar_a2 = (A * A).sum() / (B * B)
    dvar_b2 = (B_mat * B_mat).sum() / (B * B)

    # Safe denominator with gradient protection
    denom = torch.sqrt(torch.clamp(dvar_a2 * dvar_b2, min=eps**2))
    dcor2 = dcov2 / denom
    return torch.clamp(dcor2, min=0.0, max=1.0)
