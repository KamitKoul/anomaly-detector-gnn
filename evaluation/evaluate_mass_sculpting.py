"""
evaluate_mass_sculpting.py — Standalone Background Mass Sculpting & 1/JSD Evaluator
====================================================================================
Measures background mass sculpting across trained models to test whether anomaly
score thresholds artificially shape the background invariant mass distribution.

Metric:
  Background Smoothness = 1 / JSD(P(m_bkg) || P(m_bkg | score > tau))
  where tau is the anomaly threshold at 10% (90th percentile), 5% (95th percentile),
  and 1% (99th percentile) selection efficiency.

Usage:
  python evaluation/evaluate_mass_sculpting.py --family architecture --limit 10000
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_framework import load_config, check_dataset_availability, VALID_FAMILIES
from src.run_lhco_gnn import (
    create_model,
    event_to_graph,
    chunk_to_flat_features,
    stream_filtered_events,
    extract_particles,
    get_device,
)
from src.adversarial_unlearning_ae import AdversarialNodeGNN, compute_invariant_mass
from src.physics_grooming import compute_groomed_dijet_mass


def compute_jsd(p_samples: np.ndarray, q_samples: np.ndarray, bins: np.ndarray) -> float:
    """Computes Jensen-Shannon Divergence between two mass distributions."""
    from scipy.spatial.distance import jensenshannon

    p_hist, _ = np.histogram(p_samples, bins=bins, density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, density=True)

    eps = 1e-12
    p_norm = (p_hist + eps) / np.sum(p_hist + eps)
    q_norm = (q_hist + eps) / np.sum(q_hist + eps)

    js_dist = jensenshannon(p_norm, q_norm, base=np.e)
    return float(js_dist ** 2)


def evaluate_model_mass_sculpting(
    exp_dir: Path,
    eval_limit: int = 10000,
    bins: Optional[np.ndarray] = None,
) -> Optional[Dict[str, Any]]:
    """Evaluates background mass sculpting (1/JSD) for a single experiment."""
    config_path = exp_dir / "config.json"
    ckpt_path = exp_dir / "checkpoint.pt"

    if not config_path.exists() or not ckpt_path.exists():
        print(f"[STATUS] Skipping {exp_dir.name}: missing config.json or checkpoint.pt")
        return None

    config = load_config(config_path)
    ds = check_dataset_availability(config, repo_root=REPO_ROOT)
    if not ds["background_available"]:
        print(f"[STATUS] Skipping {exp_dir.name}: background dataset not available locally")
        return None

    device = get_device()
    model = create_model(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    bg_path = REPO_ROOT / config.dataset.background_path
    arch_type = config.model.architecture_type
    train_skip = getattr(config.dataset, "train_limit", 0)

    print(f"\n[INFO] Evaluating mass sculpting for {config.experiment_name} ({eval_limit} events)...")

    scores: List[float] = []
    masses_raw: List[float] = []
    masses_rsd: List[float] = []
    skipped = 0
    count = 0

    if arch_type == "baseline_mlp":
        for chunk in stream_filtered_events(str(bg_path), is_signal=False, chunk_size=1000):
            chunk_len = len(chunk)
            if chunk_len == 0:
                continue

            start_idx = 0
            if skipped < train_skip:
                needed = train_skip - skipped
                if chunk_len <= needed:
                    skipped += chunk_len
                    continue
                else:
                    start_idx = needed
                    skipped = train_skip

            available = chunk_len - start_idx
            remaining = eval_limit - count
            take_len = min(available, remaining)
            chunk_slice = chunk[start_idx : start_idx + take_len]

            batch_x = chunk_to_flat_features(
                chunk_slice,
                input_dim=config.model.input_dim,
                log_pt=config.graph_construction.log_pt,
            ).to(device, non_blocking=True)

            with torch.no_grad():
                recon_x = model(batch_x)
                batch_scores = F.mse_loss(recon_x, batch_x, reduction="none").mean(dim=1)

            scores.extend(batch_scores.cpu().tolist())

            # Extract mass for each event
            for i in range(len(chunk_slice)):
                event = chunk_slice[i]
                parts = extract_particles(event)
                raw_m, rsd_m = compute_groomed_dijet_mass(parts)
                masses_raw.append(raw_m)
                masses_rsd.append(rsd_m)

            count += take_len
            if count >= eval_limit:
                break
    else:
        delta_r = config.graph_construction.delta_r_threshold
        log_pt = config.graph_construction.log_pt
        include_met = config.graph_construction.include_met_node
        phi_mode = config.graph_construction.phi_met_mode
        graph_buffer: List[Any] = []
        batch_size = 128

        def flush_graphs():
            nonlocal graph_buffer, scores
            if not graph_buffer:
                return
            from torch_geometric.data import Batch
            batched = Batch.from_data_list(graph_buffer).to(device, non_blocking=True)
            with torch.no_grad():
                if isinstance(model, AdversarialNodeGNN):
                    recon_x, _, _ = model(batched.x, batched.edge_index, batched.batch)
                else:
                    recon_x, _ = model(batched.x, batched.edge_index, batched.batch)

                node_err = (recon_x - batched.x).pow(2).mean(dim=1)
                num_g = batched.num_graphs
                sums = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, node_err)
                counts = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, torch.ones_like(node_err))
                batch_scores = (sums / counts.clamp(min=1.0)).cpu().tolist()
            scores.extend(batch_scores)
            graph_buffer.clear()

        for chunk in stream_filtered_events(str(bg_path), is_signal=False, chunk_size=1000):
            for i in range(len(chunk)):
                if skipped < train_skip:
                    skipped += 1
                    continue
                if count >= eval_limit:
                    break

                event = chunk[i]
                graph = event_to_graph(
                    event,
                    log_pt=log_pt,
                    delta_r_threshold=delta_r,
                    include_met_node=include_met,
                    phi_met_mode=phi_mode,
                )
                graph_buffer.append(graph)

                parts = extract_particles(event)
                raw_m, rsd_m = compute_groomed_dijet_mass(parts)
                masses_raw.append(raw_m)
                masses_rsd.append(rsd_m)

                count += 1
                if len(graph_buffer) >= batch_size:
                    flush_graphs()
            if count >= eval_limit:
                break
        flush_graphs()

    scores_arr = np.array(scores)
    m_raw_arr = np.array(masses_raw)
    m_rsd_arr = np.array(masses_rsd)

    if bins is None:
        bins = np.linspace(2000.0, 5000.0, 40)

    # Compute JSD and 1/JSD at 10% (90th percentile), 5% (95th), and 1% (99th) cuts
    percentiles = [90.0, 95.0, 99.0]
    results_dict: Dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "experiment_family": config.experiment_family,
        "eval_count": len(scores_arr),
        "cuts": {},
    }

    for p in percentiles:
        eff_pct = 100.0 - p
        tau = float(np.percentile(scores_arr, p))
        mask_cut = scores_arr > tau

        # Raw mass sculpting
        jsd_raw = compute_jsd(m_raw_arr, m_raw_arr[mask_cut], bins)
        inv_jsd_raw = round(1.0 / jsd_raw, 2) if jsd_raw > 1e-7 else 9999.0

        # Groomed m_RSD mass sculpting
        jsd_rsd = compute_jsd(m_rsd_arr, m_rsd_arr[mask_cut], bins)
        inv_jsd_rsd = round(1.0 / jsd_rsd, 2) if jsd_rsd > 1e-7 else 9999.0

        results_dict["cuts"][f"top_{eff_pct:.0f}pct"] = {
            "threshold": round(tau, 6),
            "jsd_raw": round(jsd_raw, 6),
            "inv_jsd_raw": inv_jsd_raw,
            "jsd_rsd": round(jsd_rsd, 6),
            "inv_jsd_rsd": inv_jsd_rsd,
        }

    # Save to experiment directory
    out_json = exp_dir / "mass_sculpting.json"
    with open(out_json, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"[SUCCESS] Mass sculpting metrics saved: {out_json}")
    top10 = results_dict["cuts"]["top_10pct"]
    print(f"  [RESULT] Top 10% Cut -> 1/JSD (Raw): {top10['inv_jsd_raw']} | 1/JSD (RSD): {top10['inv_jsd_rsd']}")

    return results_dict


def evaluate_family_mass_sculpting(family: str, eval_limit: int = 10000) -> int:
    """Evaluates all experiments in a family and writes comparison table."""
    family_dir = REPO_ROOT / "experiments" / family
    if not family_dir.exists():
        print(f"[STATUS] Family directory not found: {family_dir}")
        return 0

    exp_dirs = [d for d in family_dir.iterdir() if d.is_dir()]
    if not exp_dirs:
        print(f"[STATUS] No experiment runs in {family_dir}")
        return 0

    all_results = []
    for ed in exp_dirs:
        res = evaluate_model_mass_sculpting(ed, eval_limit=eval_limit)
        if res:
            all_results.append(res)

    if not all_results:
        print("[STATUS] No valid checkpoints or datasets found for mass sculpting evaluation.")
        return 0

    # Build comparison summary table
    table_lines = [
        f"# Mass Sculpting Summary: {family.replace('_', ' ').title()}",
        "",
        "| Experiment | 1/JSD (Raw, Top 10%) | 1/JSD (RSD, Top 10%) | 1/JSD (Raw, Top 5%) | 1/JSD (RSD, Top 5%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in all_results:
        c10 = r["cuts"].get("top_10pct", {})
        c5 = r["cuts"].get("top_5pct", {})
        table_lines.append(
            f"| `{r['experiment_name']}` | "
            f"{c10.get('inv_jsd_raw', '-')} | "
            f"{c10.get('inv_jsd_rsd', '-')} | "
            f"{c5.get('inv_jsd_raw', '-')} | "
            f"{c5.get('inv_jsd_rsd', '-') } |"
        )

    out_file = REPO_ROOT / "comparisons" / family / "mass_sculpting_table.md"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write("\n".join(table_lines) + "\n")

    print(f"\n[SUCCESS] Mass sculpting comparison table written to: {out_file}")
    print("\n" + "\n".join(table_lines))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate background mass sculpting (1/JSD)")
    parser.add_argument(
        "--family",
        type=str,
        default="architecture",
        choices=VALID_FAMILIES,
        help="Experiment family to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of background events to evaluate for mass sculpting",
    )
    args = parser.parse_args()

    sys.exit(evaluate_family_mass_sculpting(args.family, eval_limit=args.limit))


if __name__ == "__main__":
    main()
