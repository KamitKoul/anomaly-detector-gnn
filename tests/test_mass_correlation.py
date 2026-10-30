"""
test_mass_correlation.py — Unit Tests for Mass Correlation & Latent Probes
========================================================================
Validates that correlation metrics, mass-matched ROC masking, and latent probes
execute without errors on synthetic event batches.
"""

import unittest
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge


class TestMassCorrelation(unittest.TestCase):

    def test_correlation_metrics(self):
        """Verify Pearson r and Spearman rho calculations on correlated data."""
        np.random.seed(42)
        mass = np.linspace(2000, 5000, 100)
        # Positively correlated score
        score = 0.001 * mass + np.random.normal(0, 0.2, 100)
        
        r, _ = pearsonr(mass, score)
        rho, _ = spearmanr(mass, score)
        
        self.assertGreater(r, 0.7)
        self.assertGreater(rho, 0.7)

    def test_mass_matched_roc_selection(self):
        """Verify that mass-matched ROC masking isolates the signal window."""
        np.random.seed(42)
        n_bg, n_sig = 500, 200
        
        # Background mass follows falling spectrum (mean ~2500)
        bg_m = 2000.0 + np.random.exponential(scale=600.0, size=n_bg)
        # Signal mass is localized around 3500 GeV
        sig_m = np.random.normal(loc=3500.0, scale=80.0, size=n_sig)
        
        win_low, win_high = 3300.0, 3700.0
        bg_mask = (bg_m >= win_low) & (bg_m <= win_high)
        sig_mask = (sig_m >= win_low) & (sig_m <= win_high)
        
        self.assertGreater(np.sum(sig_mask), 150)
        self.assertGreater(np.sum(bg_mask), 10)
        
        # Artificial mass-correlated score
        bg_scores = bg_m / 1000.0
        sig_scores = sig_m / 1000.0
        
        # Inclusive ROC will be artificially high because sig_m > bg_m
        y_all = np.array([0] * n_bg + [1] * n_sig)
        sc_all = np.concatenate([bg_scores, sig_scores])
        auc_incl = roc_auc_score(y_all, sc_all)
        self.assertGreater(auc_incl, 0.85)
        
        # Inside the matched mass window, the mass difference disappears!
        y_win = np.array([0] * np.sum(bg_mask) + [1] * np.sum(sig_mask))
        sc_win = np.concatenate([bg_scores[bg_mask], sig_scores[sig_mask]])
        auc_matched = roc_auc_score(y_win, sc_win)
        
        # The matched AUC must drop dramatically toward ~0.50 since score was purely mass
        self.assertLess(auc_matched, 0.70)

    def test_latent_probe_predictability(self):
        """Verify Ridge probe R^2 computation on synthetic latent representations."""
        np.random.seed(42)
        n = 200
        m_target = np.random.uniform(2.0, 5.0, n)
        
        # Representation 1: Latent z encodes mass
        z_correlated = np.column_stack([m_target * 0.5 + np.random.normal(0, 0.1, n) for _ in range(8)])
        ridge1 = Ridge(alpha=1.0)
        ridge1.fit(z_correlated[:150], m_target[:150])
        r2_corr = ridge1.score(z_correlated[150:], m_target[150:])
        self.assertGreater(r2_corr, 0.75)
        
        # Representation 2: Latent z decorrelated (noise)
        z_decorrelated = np.random.normal(0, 1.0, (n, 8))
        ridge2 = Ridge(alpha=1.0)
        ridge2.fit(z_decorrelated[:150], m_target[:150])
        r2_decorr = ridge2.score(z_decorrelated[150:], m_target[150:])
        self.assertLess(r2_decorr, 0.10)


if __name__ == "__main__":
    unittest.main()
