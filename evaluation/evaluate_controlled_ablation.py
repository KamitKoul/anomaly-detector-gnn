"""
evaluate_controlled_ablation.py — Controlled Structure & Anomaly-Score Ablation
==============================================================================
Systematic evaluation suite for the 4 controlled model architectures:
  - Model A: Baseline Flat MLP (baseline_mlp)
  - Model B: Relational GAT (gat_relational)
  - Model C: Node-Wise Autoencoder (nodewise_ae)
  - Model D: DeepSets PointNet (deepsets_pointnet)

Evaluates 4 anomaly-score formulations across multiple random seeds (42, 123, 999):
  1. Full 3D Reconstruction Error
  2. p_T-Only Reconstruction Error
  3. Angular-Only (eta, phi) Reconstruction Error
  4. Optimal Weighted Reconstruction Error

Computes:
  - Inclusive ROC-AUC, Matched ROC-AUC ([3.3, 3.7] TeV), Micro-Fine ROC-AUC ([3.45, 3.55] TeV)
  - Background Rejection (1/FPR) at 30% and 50% signal efficiency
  - Pearson r and Spearman rho correlation with invariant mass
  - Raw and RSD mass sculpting (1/JSD) at Top 10%, Top 5%, Top 1% cuts with bootstrap uncertainties

Usage:
  python evaluation/evaluate_controlled_ablation.py --family controlled_structure --limit 10000 --signal-limit 5000
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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_framework import load_config
from src.run_lhco_gnn import (
    create_model,
    event_to_graph,
    chunk_to_flat_features,
    stream_filtered_events,
    extract_particles,
    get_device,
)
from src.physics_grooming import compute_groomed_dijet_mass


def compute_jsd_smoothness(
    masses_full: np.ndarray,
    masses_cut: np.ndarray,
    bins: int = 40,
    range_gev: Tuple[float, float] = (2000.0, 5000.0),
    num_bootstraps: int = 500,
) -> Tuple[float, float, float]:
    """Computes JSD divergence and 1/JSD smoothness with bootstrap uncertainty."""
    if len(masses_cut) < 10:
        return 0.0, 0.0, 0.0

    p_counts, edges = np.histogram(masses_full, bins=bins, range=range_gev)
    q_counts, _ = np.histogram(masses_cut, bins=bins, range=range_gev)

    p = p_counts.astype(np.float64) / (np.sum(p_counts) + 1e-12)
    q = q_counts.astype(np.float64) / (np.sum(q_counts) + 1e-12)

    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()

    js_div = float(jensenshannon(p, q, base=2.0) ** 2)
    inv_jsd = 1.0 / js_div if js_div > 1e-9 else 9999.0

    # Bootstrap uncertainty
    n_cut = len(masses_cut)
    rng = np.random.RandomState(42)
    boot_invs = []
    for _ in range(num_bootstraps):
        resample = rng.choice(masses_cut, size=n_cut, replace=True)
        q_b, _ = np.histogram(resample, bins=bins, range=range_gev)
        qb = q_b.astype(np.float64) / (np.sum(q_b) + 1e-12)
        qb = np.clip(qb, 1e-12, 1.0)
        qb /= qb.sum()
        d = float(jensenshannon(p, qb, base=2.0) ** 2)
        if d > 1e-9:
            boot_invs.append(1.0 / d)

    unc = float(np.std(boot_invs)) if boot_invs else 0.0
    return js_div, inv_jsd, unc


def compute_roc_auc_with_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_bootstraps: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Computes ROC-AUC with stratified non-parametric bootstrap uncertainty and 68% CI.
    Preserves background-to-signal class balance across resamples.

    Returns:
        auc_nominal, auc_std, (auc_ci_lower, auc_ci_upper)
    """
    auc_nominal = float(roc_auc_score(y_true, y_score))
    idx_bg = np.where(y_true == 0)[0]
    idx_sig = np.where(y_true == 1)[0]
    n_bg = len(idx_bg)
    n_sig = len(idx_sig)

    if n_bg < 5 or n_sig < 5:
        return auc_nominal, 0.0, (auc_nominal, auc_nominal)

    rng = np.random.RandomState(seed)
    boot_aucs = []
    for _ in range(num_bootstraps):
        b_bg = rng.choice(idx_bg, size=n_bg, replace=True)
        b_sig = rng.choice(idx_sig, size=n_sig, replace=True)
        b_idx = np.concatenate([b_bg, b_sig])
        boot_aucs.append(roc_auc_score(y_true[b_idx], y_score[b_idx]))

    boot_aucs = np.array(boot_aucs)
    auc_std = float(np.std(boot_aucs))
    ci_lower = float(np.percentile(boot_aucs, 15.865))
    ci_upper = float(np.percentile(boot_aucs, 84.135))
    return auc_nominal, auc_std, (ci_lower, ci_upper)


def evaluate_controlled_experiments(
    family: str = "controlled_structure",
    eval_limit: int = 10000,
    signal_limit: int = 5000,
) -> Dict[str, Any]:
    """Runs complete ablation analysis across all trained checkpoints in family."""
    device = get_device()
    print(f"[DEVICE] Compute device: {device}")
    family_dir = REPO_ROOT / "experiments" / family
    configs_dir = REPO_ROOT / "configs" / family

    if not family_dir.exists():
        print(f"[INFO] Family directory does not exist: {family_dir}. Checkpoints must be trained first.")
        return {}

    # Find all available checkpoints
    available_runs = []
    for exp_path in sorted(family_dir.iterdir()):
        if exp_path.is_dir() and (exp_path / "checkpoint.pt").exists() and (exp_path / "config.json").exists():
            available_runs.append(exp_path)

    if not available_runs:
        print(f"[INFO] No trained checkpoints found in {family_dir}.")
        print("       Please run training in Kaggle GPU environment first.")
        return {}

    print(f"[INFO] Found {len(available_runs)} trained experiment checkpoints to evaluate.")

    # Load background and signal events
    sample_cfg = load_config(available_runs[0] / "config.json")
    bg_path = REPO_ROOT / sample_cfg.dataset.background_path
    sig_path = REPO_ROOT / sample_cfg.dataset.signal_path

    print(f"\n[PHASE 1] Pre-processing {eval_limit} Background and {signal_limit} Signal events...")

    def stream_and_cache_events(file_path: Path, is_signal: bool, max_events: int, skip: int = 0):
        evs, raw_ms, rsd_ms = [], [], []
        count = 0
        skipped = 0
        for chunk in stream_filtered_events(str(file_path), is_signal=is_signal, chunk_size=1000):
            for i in range(len(chunk)):
                if skipped < skip:
                    skipped += 1
                    continue
                if count >= max_events:
                    break
                ev = chunk[i]
                parts = extract_particles(ev)
                raw_m, rsd_m = compute_groomed_dijet_mass(parts)
                evs.append(ev)
                raw_ms.append(raw_m)
                rsd_ms.append(rsd_m)
                count += 1
            if count >= max_events:
                break
        return evs, np.array(raw_ms), np.array(rsd_ms)

    bg_events, bg_raw_m, bg_rsd_m = stream_and_cache_events(bg_path, is_signal=False, max_events=eval_limit, skip=5000)
    sig_events, sig_raw_m, sig_rsd_m = stream_and_cache_events(sig_path, is_signal=True, max_events=signal_limit, skip=0)
    print(f"Loaded {len(bg_events)} background and {len(sig_events)} signal events.")

    from torch_geometric.data import Batch

    def score_model_on_events(model, cfg, events_list):
        arch = cfg.model.architecture_type
        sc_full, sc_pt, sc_ang, sc_w = [], [], [], []

        if arch == "baseline_mlp":
            for i in range(0, len(events_list), 128):
                batch_evs = events_list[i : i + 128]
                x_flat = chunk_to_flat_features(batch_evs, input_dim=cfg.model.input_dim, log_pt=cfg.graph_construction.log_pt).to(device)
                with torch.no_grad():
                    recon = model(x_flat)
                    input_dim = cfg.model.input_dim
                    pt_cols = list(range(0, input_dim, 3))
                    eta_cols = list(range(1, input_dim, 3))
                    phi_cols = list(range(2, input_dim, 3))

                    pt_err = (recon[:, pt_cols] - x_flat[:, pt_cols]).pow(2).mean(dim=1)
                    eta_err = (recon[:, eta_cols] - x_flat[:, eta_cols]).pow(2).mean(dim=1)
                    dphi = torch.abs(recon[:, phi_cols] - x_flat[:, phi_cols])
                    phi_err = torch.min(dphi, 2.0 * torch.pi - dphi).pow(2).mean(dim=1)
                    ang_err = 0.5 * (eta_err + phi_err)
                    full_err = (pt_err + eta_err + phi_err) / 3.0
                    w_err = 0.8 * pt_err + 0.2 * ang_err

                sc_full.extend(full_err.cpu().tolist())
                sc_pt.extend(pt_err.cpu().tolist())
                sc_ang.extend(ang_err.cpu().tolist())
                sc_w.extend(w_err.cpu().tolist())
        else:
            buffer = []
            for ev in events_list:
                g = event_to_graph(
                    ev,
                    log_pt=cfg.graph_construction.log_pt,
                    delta_r_threshold=cfg.graph_construction.delta_r_threshold,
                    include_met_node=cfg.graph_construction.include_met_node,
                    phi_met_mode=cfg.graph_construction.phi_met_mode,
                )
                buffer.append(g)
                if len(buffer) >= 128:
                    batched = Batch.from_data_list(buffer).to(device)
                    with torch.no_grad():
                        out = model(batched.x, batched.edge_index, batched.batch)
                        recon_x = out[0] if isinstance(out, (tuple, list)) else out
                        
                        # Kinematic errors with periodic S^1 phi wrapping
                        err_pt = (recon_x[:, 0] - batched.x[:, 0]).pow(2)
                        err_eta = (recon_x[:, 1] - batched.x[:, 1]).pow(2)
                        dphi = torch.abs(recon_x[:, 2] - batched.x[:, 2])
                        err_phi = torch.min(dphi, 2.0 * torch.pi - dphi).pow(2)
                        err_ang = 0.5 * (err_eta + err_phi)
                        err_full = (err_pt + err_eta + err_phi) / 3.0
                        err_w = 0.8 * err_pt + 0.2 * err_ang

                        num_g = batched.num_graphs
                        c_pt = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_pt)
                        c_ang = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_ang)
                        c_full = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_full)
                        c_w = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_w)
                        counts = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, torch.ones_like(err_full))

                        sc_pt.extend((c_pt / counts.clamp(min=1.0)).cpu().tolist())
                        sc_ang.extend((c_ang / counts.clamp(min=1.0)).cpu().tolist())
                        sc_full.extend((c_full / counts.clamp(min=1.0)).cpu().tolist())
                        sc_w.extend((c_w / counts.clamp(min=1.0)).cpu().tolist())
                    buffer.clear()
            if buffer:
                batched = Batch.from_data_list(buffer).to(device)
                with torch.no_grad():
                    out = model(batched.x, batched.edge_index, batched.batch)
                    recon_x = out[0] if isinstance(out, (tuple, list)) else out
                    
                    err_pt = (recon_x[:, 0] - batched.x[:, 0]).pow(2)
                    err_eta = (recon_x[:, 1] - batched.x[:, 1]).pow(2)
                    dphi = torch.abs(recon_x[:, 2] - batched.x[:, 2])
                    err_phi = torch.min(dphi, 2.0 * torch.pi - dphi).pow(2)
                    err_ang = 0.5 * (err_eta + err_phi)
                    err_full = (err_pt + err_eta + err_phi) / 3.0
                    err_w = 0.8 * err_pt + 0.2 * err_ang

                    num_g = batched.num_graphs
                    c_pt = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_pt)
                    c_ang = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_ang)
                    c_full = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_full)
                    c_w = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_w)
                    counts = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, torch.ones_like(err_full))

                    sc_pt.extend((c_pt / counts.clamp(min=1.0)).cpu().tolist())
                    sc_ang.extend((c_ang / counts.clamp(min=1.0)).cpu().tolist())
                    sc_full.extend((c_full / counts.clamp(min=1.0)).cpu().tolist())
                    sc_w.extend((c_w / counts.clamp(min=1.0)).cpu().tolist())
                buffer.clear()

        return {
            "full": np.array(sc_full),
            "pt_only": np.array(sc_pt),
            "ang_only": np.array(sc_ang),
            "weighted": np.array(sc_w),
        }

    # Evaluate each available experiment
    results = {}
    win_std_bg = (bg_raw_m >= 3300.0) & (bg_raw_m <= 3700.0)
    win_std_sig = (sig_raw_m >= 3300.0) & (sig_raw_m <= 3700.0)

    win_micro_bg = (bg_raw_m >= 3450.0) & (bg_raw_m <= 3550.0)
    win_micro_sig = (sig_raw_m >= 3450.0) & (sig_raw_m <= 3550.0)

    for exp_dir in available_runs:
        exp_name = exp_dir.name
        cfg = load_config(exp_dir / "config.json")
        model = create_model(cfg).to(device)
        model.load_state_dict(torch.load(exp_dir / "checkpoint.pt", map_location=device))
        model.eval()

        print(f"\n[EVALUATING] {exp_name} ({cfg.model.architecture_type}, seed={cfg.seed})...")
        bg_scores_dict = score_model_on_events(model, cfg, bg_events)
        sig_scores_dict = score_model_on_events(model, cfg, sig_events)

        exp_metrics = {}
        for score_key in ["full", "pt_only", "ang_only", "weighted"]:
            bg_s = bg_scores_dict[score_key]
            sig_s = sig_scores_dict[score_key]

            # Inclusive ROC with 1000-sample stratified bootstrap
            y_incl = np.array([0] * len(bg_s) + [1] * len(sig_s))
            s_incl = np.concatenate([bg_s, sig_s])
            auc_incl, unc_incl, ci_incl = compute_roc_auc_with_bootstrap(y_incl, s_incl, num_bootstraps=1000)

            # Matched ROC [3300, 3700] GeV
            y_match = np.array([0] * int(np.sum(win_std_bg)) + [1] * int(np.sum(win_std_sig)))
            s_match = np.concatenate([bg_s[win_std_bg], sig_s[win_std_sig]])
            auc_match, unc_match, ci_match = compute_roc_auc_with_bootstrap(y_match, s_match, num_bootstraps=1000)

            # Micro-fine ROC [3450, 3550] GeV
            y_micro = np.array([0] * int(np.sum(win_micro_bg)) + [1] * int(np.sum(win_micro_sig)))
            s_micro = np.concatenate([bg_s[win_micro_bg], sig_s[win_micro_sig]])
            auc_micro, unc_micro, ci_micro = compute_roc_auc_with_bootstrap(y_micro, s_micro, num_bootstraps=1000)

            # Background Rejection (1/FPR) at 30% and 50% signal efficiency
            fpr, tpr, _ = roc_curve(y_incl, s_incl)
            rej_30 = float("inf")
            rej_50 = float("inf")
            idx_30 = np.where(tpr >= 0.30)[0]
            if len(idx_30) > 0 and fpr[idx_30[0]] > 0:
                rej_30 = float(1.0 / fpr[idx_30[0]])
            idx_50 = np.where(tpr >= 0.50)[0]
            if len(idx_50) > 0 and fpr[idx_50[0]] > 0:
                rej_50 = float(1.0 / fpr[idx_50[0]])

            # Correlation with mass
            r_val, _ = pearsonr(bg_raw_m, bg_s)
            rho_val, _ = spearmanr(bg_raw_m, bg_s)

            # Mass sculpting cuts (Top 10%, Top 5%) - Strictly standardized base-2 JSD (bits)
            tau_10 = float(np.percentile(bg_s, 90.0))
            cut_10 = bg_s > tau_10
            _, inv_raw_10, unc_raw_10 = compute_jsd_smoothness(bg_raw_m, bg_raw_m[cut_10])
            _, inv_rsd_10, unc_rsd_10 = compute_jsd_smoothness(bg_rsd_m, bg_rsd_m[cut_10])

            tau_5 = float(np.percentile(bg_s, 95.0))
            cut_5 = bg_s > tau_5
            _, inv_raw_5, unc_raw_5 = compute_jsd_smoothness(bg_raw_m, bg_raw_m[cut_5])
            _, inv_rsd_5, unc_rsd_5 = compute_jsd_smoothness(bg_rsd_m, bg_rsd_m[cut_5])

            exp_metrics[score_key] = {
                "auc_inclusive": round(auc_incl, 4),
                "auc_inclusive_unc": round(unc_incl, 4),
                "auc_inclusive_ci": [round(ci_incl[0], 4), round(ci_incl[1], 4)],
                "auc_matched": round(auc_match, 4),
                "auc_matched_unc": round(unc_match, 4),
                "auc_matched_ci": [round(ci_match[0], 4), round(ci_match[1], 4)],
                "auc_micro_fine": round(auc_micro, 4),
                "auc_micro_fine_unc": round(unc_micro, 4),
                "auc_micro_fine_ci": [round(ci_micro[0], 4), round(ci_micro[1], 4)],
                "rejection_30": round(rej_30, 2) if rej_30 != float("inf") else -1.0,
                "rejection_50": round(rej_50, 2) if rej_50 != float("inf") else -1.0,
                "pearson_r": round(float(r_val), 4),
                "spearman_rho": round(float(rho_val), 4),
                "inv_jsd_raw_top10": round(inv_raw_10, 2),
                "inv_jsd_raw_top10_unc": round(unc_raw_10, 2),
                "inv_jsd_rsd_top10": round(inv_rsd_10, 2),
                "inv_jsd_rsd_top10_unc": round(unc_rsd_10, 2),
                "inv_jsd_raw_top5": round(inv_raw_5, 2),
                "inv_jsd_raw_top5_unc": round(unc_raw_5, 2),
            }

        results[exp_name] = {
            "model_class": getattr(cfg.metadata, "model_class", cfg.model.architecture_type),
            "seed": cfg.seed,
            "metrics": exp_metrics,
        }

    # Save summary JSON
    out_dir = REPO_ROOT / "comparisons" / family
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "controlled_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Ablation summary written to: {summary_path}")

    # Write Markdown table with bootstrap confidence intervals
    md_path = out_dir / "controlled_ablation_table.md"
    with open(md_path, "w") as f:
        f.write("# Controlled Architecture & Anomaly Score Ablation Summary\n\n")
        f.write("All AUCs reported with 68% stratified bootstrap confidence intervals (1000 resamples).\n")
        f.write("JSD metrics strictly computed in base-2 (bits).\n\n")
        f.write("| Experiment | Model Class | Seed | Full Matched AUC | pT-Only Matched AUC | Angular Matched AUC | 1/JSD Raw (Top 10%) | Pearson r |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for name, data in sorted(results.items()):
            m = data["metrics"]
            f.write(
                f"| `{name}` | {data['model_class']} | {data['seed']} | "
                f"{m['full']['auc_matched']:.4f} ± {m['full']['auc_matched_unc']:.4f} | "
                f"{m['pt_only']['auc_matched']:.4f} ± {m['pt_only']['auc_matched_unc']:.4f} | "
                f"{m['ang_only']['auc_matched']:.4f} ± {m['ang_only']['auc_matched_unc']:.4f} | "
                f"{m['full']['inv_jsd_raw_top10']:.1f} ± {m['full']['inv_jsd_raw_top10_unc']:.1f} | "
                f"{m['full']['pearson_r']:+.4f} |\n"
            )
    print(f"[SUCCESS] Markdown table written to: {md_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="controlled_structure")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--signal-limit", type=int, default=5000)
    args = parser.parse_args()

    evaluate_controlled_experiments(
        family=args.family,
        eval_limit=args.limit,
        signal_limit=args.signal_limit,
    )
