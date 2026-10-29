"""
test_physics_grooming.py — Unit Tests for Cambridge-Aachen & Recursive Soft Drop
=================================================================================
Validates jet clustering, kinematic four-momenta, and Recursive Soft Drop (m_RSD)
pruning against theoretical expectations.
"""

import math
import unittest
import numpy as np

from src.physics_grooming import (
    p4_from_pt_eta_phi,
    invariant_mass_from_p4,
    delta_r,
    ClusterNode,
    cluster_cambridge_aachen,
    recursive_soft_drop,
    compute_groomed_dijet_mass,
)


class TestPhysicsGrooming(unittest.TestCase):

    def test_p4_and_invariant_mass(self):
        """Verify relativistic 4-vector kinematics and mass calculation."""
        # Single massless particle has invariant mass = 0
        p4 = p4_from_pt_eta_phi(100.0, 0.5, 1.2)
        m = invariant_mass_from_p4(p4)
        self.assertAlmostEqual(m, 0.0, places=5)

        # Two back-to-back particles with pT=500 at eta=0
        p1 = p4_from_pt_eta_phi(500.0, 0.0, 0.0)
        p2 = p4_from_pt_eta_phi(500.0, 0.0, math.pi)
        p_tot = p1 + p2
        m_tot = invariant_mass_from_p4(p_tot)
        # Invariant mass should be 1000 GeV
        self.assertAlmostEqual(m_tot, 1000.0, places=4)

    def test_delta_r_periodic_boundary(self):
        """Verify delta_R handles the [-pi, pi] azimuthal wrapping."""
        # Across the branch cut: phi1 = 3.10, phi2 = -3.10 -> Delta phi = 0.083
        dr = delta_r(0.0, 3.10, 0.0, -3.10)
        expected_dphi = 2.0 * math.pi - 6.20
        self.assertAlmostEqual(dr, expected_dphi, places=5)

    def test_soft_wide_angle_pruning(self):
        """
        Verify that Recursive Soft Drop reliably prunes soft wide-angle emissions
        while preserving hard collinear substructure.
        """
        # Hard 2-prong subjet (collinear quark pair, Delta R = 0.1)
        p1 = p4_from_pt_eta_phi(500.0, 0.0, 0.0)
        p2 = p4_from_pt_eta_phi(500.0, 0.1, 0.0)
        p_true = p1 + p2
        m_true = invariant_mass_from_p4(p_true)

        # Soft wide-angle gluon emission (Delta R ~ 0.9, pT = 5 GeV)
        p_soft = p4_from_pt_eta_phi(5.0, 0.8, 0.5)

        # Construct tree
        node1 = ClusterNode(p1)
        node2 = ClusterNode(p2)
        node_hard = ClusterNode(p1 + p2, child1=node1, child2=node2)
        node_soft = ClusterNode(p_soft)
        root = ClusterNode(p1 + p2 + p_soft, child1=node_hard, child2=node_soft)

        raw_mass = invariant_mass_from_p4(root.p4)
        # Raw mass is inflated by the soft emission
        self.assertGreater(raw_mass, m_true + 20.0)

        # Groomed leaves
        groomed_leaves = recursive_soft_drop(root, z_cut=0.15, beta=0.7, r0=1.0)
        groomed_p4 = sum(leaf.p4 for leaf in groomed_leaves)
        groomed_mass = invariant_mass_from_p4(groomed_p4)

        # Grooming must prune the soft gluon and recover the exact 2-prong invariant mass
        self.assertAlmostEqual(groomed_mass, m_true, places=4)

    def test_cambridge_aachen_clustering_order(self):
        """Verify C/A merges smallest Delta R pairs first and orders final jets by pT."""
        # 3 particles: p1 and p2 are very close (Delta R = 0.05); p3 is further away (Delta R = 0.8)
        particles = np.array([
            [400.0, 0.0, 0.0],
            [350.0, 0.03, 0.04],  # Delta R to p1 = 0.05
            [200.0, 0.6, 0.5],    # Delta R to p1 ~ 0.78
        ])

        jets = cluster_cambridge_aachen(particles, r0=1.0)
        self.assertEqual(len(jets), 1)  # All 3 merge into a single fat jet since Delta R < 1.0

        fat_jet = jets[0]
        self.assertFalse(fat_jet.is_leaf())
        # The last merge was between the p1+p2 cluster and p3
        # The inner cluster must have children p1 and p2
        inner = fat_jet.child1 if not fat_jet.child1.is_leaf() else fat_jet.child2
        self.assertTrue(inner.child1.is_leaf() and inner.child2.is_leaf())

    def test_compute_groomed_dijet_mass_interface(self):
        """Verify the full compute_groomed_dijet_mass pipeline on synthetic collision event."""
        event_particles = np.array([
            [600.0, 0.1, 0.0],
            [500.0, 0.2, 0.0],   # Jet 1 prongs
            [600.0, -0.1, 3.14],
            [500.0, -0.2, 3.14], # Jet 2 prongs
            [4.0, 0.8, 1.0],     # soft radiation
        ])

        raw_m, groomed_m = compute_groomed_dijet_mass(event_particles, z_cut=0.15, beta=0.7, r0=1.0)
        self.assertGreater(raw_m, 2000.0)
        self.assertGreater(groomed_m, 2000.0)
        # Soft radiation was pruned, so groomed mass is strictly cleaner/lower than raw mass
        self.assertLessEqual(groomed_m, raw_m)


if __name__ == "__main__":
    unittest.main()
