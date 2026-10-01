"""
test_experiment_framework.py — Phase 1 Lightweight Structural Test Suite
========================================================================
Runs non-training, lightweight structural validation tests for the controlled
experiment framework:
  1. Config file parsing and schema validation
  2. Representation of all 4 experiment families
  3. Git metadata retrieval
  4. Output contract directory initialization (dry-run)
  5. Evaluation and comparison scaffold handling of missing prerequisites
  6. Python syntax checks across all existing and new source files
"""

import os
import sys
import unittest
import tempfile
import py_compile
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_framework import (
    VALID_FAMILIES,
    load_config,
    save_config,
    validate_config,
    get_git_metadata,
    init_experiment_dir,
    check_dataset_availability,
    ExperimentConfig,
)
from evaluation.evaluate_single import evaluate_experiment
from evaluation.compare_experiments import compare_family


class TestExperimentFramework(unittest.TestCase):

    def test_all_configs_valid_json_and_schema(self):
        """All configuration files in configs/ must be valid JSON and pass schema validation."""
        configs_dir = REPO_ROOT / "configs"
        self.assertTrue(configs_dir.exists(), "configs/ directory must exist")

        config_files = list(configs_dir.glob("*/*.json"))
        self.assertGreaterEqual(len(config_files), 10, "Expected at least 10 config files")

        for cfg_path in config_files:
            cfg = load_config(cfg_path)
            errors = validate_config(cfg)
            self.assertEqual(errors, [], f"Validation failed for {cfg_path}: {errors}")
            self.assertIn(cfg.experiment_family, VALID_FAMILIES)

    def test_four_families_represented(self):
        """All 4 experiment families must have pre-configured variants."""
        configs_dir = REPO_ROOT / "configs"
        for family in VALID_FAMILIES:
            family_dir = configs_dir / family
            self.assertTrue(family_dir.exists(), f"Directory missing for family: {family}")
            configs = list(family_dir.glob("*.json"))
            self.assertGreaterEqual(
                len(configs), 2, f"Family '{family}' should have at least 2 variants"
            )

    def test_git_metadata_retrieval(self):
        """get_git_metadata must return a 40-character hex commit SHA."""
        meta = get_git_metadata()
        self.assertIn("commit", meta)
        self.assertIn("branch", meta)
        self.assertIn("recorded_at", meta)
        self.assertEqual(len(meta["commit"]), 40, f"Invalid SHA: {meta['commit']}")

    def test_output_contract_dry_run(self):
        """init_experiment_dir must create output dir with config.json and plots/ subfolder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            cfg = load_config(REPO_ROOT / "configs/architecture/gat_met_production.json")
            exp_dir = init_experiment_dir(cfg, repo_root=tmp_root)

            self.assertTrue((exp_dir / "config.json").exists())
            self.assertTrue((exp_dir / "plots").is_dir())

            # Verify saved config contains metadata
            saved_cfg = load_config(exp_dir / "config.json")
            self.assertIn("commit", saved_cfg.metadata)

    def test_dataset_availability_check(self):
        """check_dataset_availability correctly reports status without raising exceptions."""
        cfg = load_config(REPO_ROOT / "configs/architecture/gat_met_production.json")
        ds = check_dataset_availability(cfg, repo_root=REPO_ROOT)
        self.assertIn("background_available", ds)
        self.assertIn("signal_available", ds)
        # Expected False currently since dataset has not been downloaded to local data/
        self.assertIsInstance(ds["background_available"], bool)
        self.assertIsInstance(ds["signal_available"], bool)

    def test_evaluation_scaffold_missing_prerequisites(self):
        """evaluate_experiment must gracefully exit without crashing when checkpoint is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            cfg = load_config(REPO_ROOT / "configs/architecture/gat_met_production.json")
            exp_dir = init_experiment_dir(cfg, repo_root=tmp_root)

            code = evaluate_experiment(exp_dir)
            self.assertEqual(code, 0, "Evaluation scaffold should exit gracefully when pending training")

    def test_comparison_scaffold_empty_family(self):
        """compare_family must gracefully handle empty directory without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            for fam in VALID_FAMILIES:
                code = compare_family(fam, repo_root=tmp_root)
                self.assertEqual(code, 0, f"Comparison scaffold failed on empty family: {fam}")

    def test_source_code_syntax(self):
        """All python scripts in src/, evaluation/, and tests/ must compile without syntax errors."""
        py_files = list(REPO_ROOT.glob("src/*.py")) + list(REPO_ROOT.glob("evaluation/*.py"))
        for py_path in py_files:
            try:
                py_compile.compile(str(py_path), doraise=True)
            except Exception as e:
                self.fail(f"Syntax error in {py_path}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
