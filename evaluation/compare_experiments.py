"""
compare_experiments.py — Multi-Experiment Comparison Scaffold
=============================================================
Aggregates completed metrics from an experiment family into comparison tables
and summaries under:
  comparisons/<family>/

Usage:
  python evaluation/compare_experiments.py --family architecture

Design:
  - Scans experiments/<family>/*/metrics.json.
  - Generates Markdown comparison tables without fabricating missing results.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_framework import VALID_FAMILIES


def compare_family(family: str, repo_root: Path) -> int:
    """Collects metrics from experiments/<family> and writes summary comparison."""
    exp_family_dir = repo_root / "experiments" / family
    comp_family_dir = repo_root / "comparisons" / family
    comp_family_dir.mkdir(parents=True, exist_ok=True)

    if not exp_family_dir.exists():
        print(f"[STATUS] No experiment directory found at: {exp_family_dir}")
        return 0

    metric_files = list(exp_family_dir.glob("*/metrics.json"))
    if not metric_files:
        print(f"[STATUS] No metrics.json files found under: {exp_family_dir}/*/")
        print("         Experiments have not been executed yet. Skipping comparison generation.")
        return 0

    print(f"[INFO] Found {len(metric_files)} completed experiment runs in family '{family}':")
    results: List[Dict[str, Any]] = []
    for mf in metric_files:
        try:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["experiment_name"] = mf.parent.name
            results.append(data)
        except Exception as e:
            print(f"  [WARN] Failed to read {mf}: {e}")

    # Render Markdown summary table
    table_lines = [
        f"# Comparison Summary: {family.replace('_', ' ').title()}",
        "",
        "| Experiment | ROC AUC | Rejection @ 50% Eff | Separation Ratio | Significance |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        table_lines.append(
            f"| `{r.get('experiment_name', '-')}` | "
            f"{r.get('roc_auc', '-')} | "
            f"{r.get('background_rejection_at_50_eff', '-')} | "
            f"{r.get('separation_ratio', '-')}x | "
            f"{r.get('significance_sigma', '-')}σ |"
        )

    out_file = comp_family_dir / "comparison_table.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines) + "\n")

    print(f"[SUCCESS] Comparison table written to: {out_file}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Compare experiments within an experiment family")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        choices=VALID_FAMILIES,
        help="Experiment family to aggregate and compare",
    )
    args = parser.parse_args()

    sys.exit(compare_family(args.family, REPO_ROOT))


if __name__ == "__main__":
    main()
