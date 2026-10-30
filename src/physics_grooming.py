"""
physics_grooming.py — Cambridge-Aachen Clustering & Recursive Soft Drop (m_RSD)
=================================================================================
Implements jet clustering and substructure grooming for multijet collider events:
  1. Cambridge-Aachen (C/A) sequential recombining algorithm (d_ij = Delta R_ij^2).
  2. Recursive Soft Drop (RSD) grooming (Dreyer, Neckebroek, Soyez, JHEP 06 (2018) 093):
     Recursive de-clustering condition:
       z = min(pT_1, pT_2) / (pT_1 + pT_2) > z_cut * (Delta R_12 / R_0)^beta
     Default parameters: z_cut = 0.15, beta = 0.7, R_0 = 1.0 (Kasieczka et al. 2026).
  3. Dijet invariant mass calculation (m_RSD) for multijet resonance restoration.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Any
import numpy as np


def p4_from_pt_eta_phi(pt: float, eta: float, phi: float) -> np.ndarray:
    """Computes relativistic 4-momentum [E, px, py, pz] for a massless particle."""
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    energy = pt * math.cosh(eta)
    return np.array([energy, px, py, pz], dtype=np.float64)


def invariant_mass_from_p4(p4: np.ndarray) -> float:
    """Computes invariant mass in GeV from a 4-momentum vector: M = sqrt(E^2 - |p|^2)."""
    m2 = p4[0] ** 2 - np.sum(p4[1:] ** 2)
    return float(math.sqrt(max(0.0, m2)))


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    """Computes Delta R distance on the (eta, phi) cylinder with periodic boundary wrap-around."""
    d_eta = eta1 - eta2
    d_phi = abs(phi1 - phi2)
    if d_phi > math.pi:
        d_phi = 2.0 * math.pi - d_phi
    return math.sqrt(d_eta ** 2 + d_phi ** 2)


class ClusterNode:
    """Represents a pseudojet or particle in a Cambridge-Aachen binary clustering tree."""

    __slots__ = ("p4", "child1", "child2", "pt", "eta", "phi")

    def __init__(
        self,
        p4: np.ndarray,
        child1: Optional["ClusterNode"] = None,
        child2: Optional["ClusterNode"] = None,
    ):
        self.p4 = p4
        self.child1 = child1
        self.child2 = child2
        self.pt = math.sqrt(p4[1] ** 2 + p4[2] ** 2)
        p_tot = math.sqrt(p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2)
        denom = max(p_tot - p4[3], 1e-12)
        self.eta = 0.5 * math.log((p_tot + p4[3]) / denom)
        self.phi = math.atan2(p4[2], p4[1])

    def is_leaf(self) -> bool:
        return self.child1 is None and self.child2 is None


def cluster_cambridge_aachen(
    particles: np.ndarray,
    r0: float = 1.0,
) -> List[ClusterNode]:
    """
    Sequentially clusters particles into jets using the Cambridge-Aachen metric:
      d_ij = Delta R_ij^2
      d_iB = R_0^2
    Iteratively merges the closest pair until all remaining clusters have Delta R > R_0.
    Returns clustered jets sorted by descending transverse momentum (pT).
    Optimized with vectorized pairwise distance matrices.
    """
    nodes: List[ClusterNode] = []
    for p in particles:
        if p[0] > 0.0:
            nodes.append(ClusterNode(p4_from_pt_eta_phi(float(p[0]), float(p[1]), float(p[2]))))

    if len(nodes) == 0:
        return []

    r0_sq = r0 ** 2
    eta = np.array([n.eta for n in nodes], dtype=np.float64)
    phi = np.array([n.phi for n in nodes], dtype=np.float64)

    # Initial pairwise Delta R^2 matrix
    d_eta = eta[:, None] - eta[None, :]
    d_phi = np.abs(phi[:, None] - phi[None, :])
    d_phi = np.where(d_phi > np.pi, 2 * np.pi - d_phi, d_phi)
    dist_mat = d_eta ** 2 + d_phi ** 2
    np.fill_diagonal(dist_mat, np.inf)

    while len(nodes) > 1:
        min_idx = int(np.argmin(dist_mat))
        min_dr2 = float(dist_mat.flat[min_idx])
        if min_dr2 > r0_sq:
            break

        i, j = np.unravel_index(min_idx, dist_mat.shape)
        if i > j:
            i, j = j, i

        merged_p4 = nodes[i].p4 + nodes[j].p4
        merged_node = ClusterNode(merged_p4, child1=nodes[i], child2=nodes[j])

        # Remove higher index j first to preserve index i
        nodes.pop(j)
        nodes.pop(i)

        mask = np.ones(dist_mat.shape[0], dtype=bool)
        mask[i] = False
        mask[j] = False
        dist_mat = dist_mat[mask][:, mask]
        eta = eta[mask]
        phi = phi[mask]

        if len(nodes) > 0:
            d_eta_new = np.abs(eta - merged_node.eta)
            d_phi_new = np.abs(phi - merged_node.phi)
            d_phi_new = np.where(d_phi_new > np.pi, 2 * np.pi - d_phi_new, d_phi_new)
            new_col = (d_eta_new ** 2 + d_phi_new ** 2).reshape(-1, 1)

            dist_mat = np.hstack([dist_mat, new_col])
            new_row = np.append(new_col.flatten(), np.inf).reshape(1, -1)
            dist_mat = np.vstack([dist_mat, new_row])

            eta = np.append(eta, merged_node.eta)
            phi = np.append(phi, merged_node.phi)

        nodes.append(merged_node)

    # Sort final jets by descending pT
    nodes.sort(key=lambda node: node.pt, reverse=True)
    return nodes


def recursive_soft_drop(
    node: ClusterNode,
    z_cut: float = 0.15,
    beta: float = 0.7,
    r0: float = 1.0,
) -> List[ClusterNode]:
    """
    Recursively de-clusters a Cambridge-Aachen jet tree to strip soft wide-angle radiation.
    At each splitting J -> J_1 + J_2 (with pT_1 >= pT_2):
      Check condition: z = pT_2 / (pT_1 + pT_2) > z_cut * (Delta R_12 / R_0)^beta
      - If satisfied: keep both branches and recurse down both to strip substructure radiation.
      - If violated: drop softer branch J_2 and continue recursing down harder branch J_1.
    """
    if node.is_leaf():
        return [node]

    left = node.child1
    right = node.child2
    assert left is not None and right is not None

    # Order branches so left is always the harder branch
    if right.pt > left.pt:
        left, right = right, left

    dr = delta_r(left.eta, left.phi, right.eta, right.phi)
    z = right.pt / (left.pt + right.pt) if (left.pt + right.pt) > 0 else 0.0
    threshold = z_cut * ((dr / r0) ** beta)

    if z > threshold:
        # Passes condition: keep both and recurse down both branches
        return recursive_soft_drop(left, z_cut, beta, r0) + recursive_soft_drop(right, z_cut, beta, r0)
    else:
        # Soft emission: prune right branch and recurse down harder left branch
        return recursive_soft_drop(left, z_cut, beta, r0)


def compute_groomed_dijet_mass(
    event_particles: np.ndarray,
    z_cut: float = 0.15,
    beta: float = 0.7,
    r0: float = 1.0,
    num_jets: int = 2,
) -> Tuple[float, float]:
    """
    Given an event's particles [pT, eta, phi]:
      1. Performs C/A clustering to identify leading fat jets (default: top 2 dijets).
      2. Computes the raw dijet invariant mass.
      3. Applies Recursive Soft Drop grooming to the leading jets.
      4. Computes and returns (m_raw, m_RSD) in GeV.
    """
    jets = cluster_cambridge_aachen(event_particles, r0=r0)
    if len(jets) == 0:
        return 0.0, 0.0

    selected_jets = jets[:num_jets]

    # Raw invariant mass
    raw_p4 = np.zeros(4, dtype=np.float64)
    for j in selected_jets:
        raw_p4 += j.p4
    raw_mass = invariant_mass_from_p4(raw_p4)

    # Groomed invariant mass (m_RSD)
    groomed_p4 = np.zeros(4, dtype=np.float64)
    for j in selected_jets:
        leaves = recursive_soft_drop(j, z_cut=z_cut, beta=beta, r0=r0)
        for leaf in leaves:
            groomed_p4 += leaf.p4
    groomed_mass = invariant_mass_from_p4(groomed_p4)

    return raw_mass, groomed_mass
