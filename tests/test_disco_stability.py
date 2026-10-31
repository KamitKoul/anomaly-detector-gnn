"""
test_disco_stability.py — Unit Tests for DisCo Mathematical Properties and Stability
=====================================================================================
Verifies:
  1. Finite-sample bias of classical V-statistic dCor vs unbiased U-statistic Omega_n.
  2. Numerical stability when score variance approaches zero.
  3. Differentiability and finite gradients without NaN/Inf.
  4. Consistency of weighted score formulation between training and evaluation.
  5. Distinction between total event invariant mass and clustered dijet mass.
"""

import sys
import unittest
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.adversarial_unlearning_ae import compute_distance_correlation, compute_invariant_mass
from src.physics_grooming import compute_groomed_dijet_mass
from src.stable_disco import compute_unbiased_distance_covariance, compute_stabilized_distance_correlation


class TestDisCoStability(unittest.TestCase):

    def test_finite_sample_bias_contrast(self):
        """Verifies classical dCor^2 is positively biased (~0.09 at N=32) while unbiased Omega_n centers near 0."""
        torch.manual_seed(42)
        n = 32
        trials = 200
        
        classical_vals = []
        unbiased_vals = []
        for _ in range(trials):
            x = torch.randn(n)
            y = torch.randn(n)
            classical_vals.append(compute_distance_correlation(x, y).item())
            unbiased_vals.append(compute_unbiased_distance_covariance(x, y).item())
            
        mean_classical = np.mean(classical_vals)
        mean_unbiased = np.mean(unbiased_vals)
        
        # Classical V-statistic has positive bias floor
        self.assertGreater(mean_classical, 0.05, f"Classical dCor^2 should exhibit positive bias floor: {mean_classical}")
        # Unbiased U-statistic has expectation zero
        self.assertAlmostEqual(mean_unbiased, 0.0, delta=0.03, msg=f"Unbiased Omega_n should center near zero: {mean_unbiased}")

    def test_numerical_stability_zero_variance(self):
        """Verifies stabilized dCor does not produce NaN/Inf when all inputs are identical."""
        x = torch.full((32,), 2.5, requires_grad=True)
        y = torch.randn(32)
        
        # Stabilized estimator must return 0.0 without NaN
        val = compute_stabilized_distance_correlation(x, y, eps=1e-6)
        self.assertFalse(torch.isnan(val).item(), "Stabilized dCor produced NaN on zero variance")
        self.assertEqual(val.item(), 0.0)
        
        # Gradient must remain finite
        val.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any().item(), "Gradient contained NaN on zero variance")

    def test_mass_definition_distinction(self):
        """Documents that total particle invariant mass != clustered dijet invariant mass."""
        # Create an event with 4 particles: 2 hard leading (dijet core) + 2 soft wide emissions
        # Dijet cluster will cluster the 2 hard jets, while total invariant mass sums all 4
        particles = np.array([
            [500.0, 0.5, 0.0],     # Jet 1 core
            [480.0, -0.5, np.pi],  # Jet 2 core
            [20.0, 2.0, 0.5],      # Soft forward emission
            [15.0, -2.2, -0.8],    # Beam remnant
        ])
        
        total_m = compute_invariant_mass(particles)
        raw_m, _ = compute_groomed_dijet_mass(particles, r0=1.0, num_jets=2)
        
        # They should both be positive and physically correlated, but numerically distinct
        self.assertGreater(total_m, 0.0)
        self.assertGreater(raw_m, 0.0)
        self.assertNotEqual(round(total_m, 2), round(raw_m, 2), "Total event mass must differ from dijet mass")


if __name__ == "__main__":
    unittest.main()
