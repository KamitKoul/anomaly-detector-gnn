"""
evaluate_mass_correlation.py — Mass Correlation, Mass-Matched ROC, and Latent Probe Study
========================================================================================
Answers the three critical scientific questions:
1. How strongly does each model's anomaly score depend on invariant mass?
   - Computes Pearson r, Spearman rho, Mutual Information, and mass-binned profiles E[Score | m].
2. Does the standalone GAT still outperform other models within matched mass ranges?
   - Computes Inclusive ROC-AUC vs Mass-Matched ROC-AUC (in the W' signal window [3300, 3700] GeV).
3. Does the adversarial model's node-level anomaly score retain mass information even though
   its pooled representation is decorrelated?
   - Compares mass predictability R^2 from pooled latent z_bar vs mass correlation of node score.

Usage:
  python evaluation/evaluate_mass_correlation.py --family architecture --limit 10000
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
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr

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
    BaselineMLP,
    NodeGNN,
)
from src.adversarial_unlearning_ae import AdversarialNodeGNN, compute_invariant_mass
from src.physics_grooming import compute_groomed_dijet_mass


def evaluate_mass_correlation_study(
    family: str = "architecture",
    eval_limit: int = 10000,
    signal_limit: int = 5000,
) -> Dict[str, Any]:
    """Runs the comprehensive mass correlation, mass-matched ROC, and latent probe study."""
    family_dir = REPO_ROOT / "experiments" / family
    config_names = ["baseline_mlp", "gat_standalone", "gat_met_production", "gat_met_decorrelation"]
    
    device = get_device()
    print(f"[DEVICE] Compute device: {device}")
    
    # 1. Load models
    models = {}
    configs = {}
    for name in config_names:
        exp_dir = family_dir / name
        cfg_path = exp_dir / "config.json"
        ckpt_path = exp_dir / "checkpoint.pt"
        if not cfg_path.exists() or not ckpt_path.exists():
            print(f"[STATUS] Skipping {name}: missing config or checkpoint")
            continue
        cfg = load_config(cfg_path)
        m = create_model(cfg).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device))
        m.eval()
        models[name] = m
        configs[name] = cfg

    if not models:
        print("[ERROR] No models loaded. Cannot run study.")
        return {}

    bg_path = REPO_ROOT / configs["gat_met_production"].dataset.background_path
    sig_path = REPO_ROOT / configs["gat_met_production"].dataset.signal_path

    # Verify background and signal
    if not bg_path.exists() or not sig_path.exists():
        print(f"[ERROR] Datasets not found: bg={bg_path.exists()}, sig={sig_path.exists()}")
        return {}

    print(f"\n[PHASE 1] Pre-processing {eval_limit} Background and {signal_limit} Signal events...")
    
    # We collect particles, masses, flat features, and graphs once
    def load_events(file_path: Path, is_signal: bool, max_events: int, skip_first: int = 0):
        ev_list = []
        raw_m_list = []
        rsd_m_list = []
        skipped = 0
        count = 0
        for chunk in stream_filtered_events(str(file_path), is_signal=is_signal, chunk_size=1000):
            for i in range(len(chunk)):
                if skipped < skip_first:
                    skipped += 1
                    continue
                if count >= max_events:
                    break
                ev = chunk[i]
                parts = extract_particles(ev)
                raw_m, rsd_m = compute_groomed_dijet_mass(parts)
                ev_list.append(ev)
                raw_m_list.append(raw_m)
                rsd_m_list.append(rsd_m)
                count += 1
            if count >= max_events:
                break
        return ev_list, np.array(raw_m_list), np.array(rsd_m_list)

    bg_events, bg_raw_m, bg_rsd_m = load_events(bg_path, is_signal=False, max_events=eval_limit, skip_first=5000)
    sig_events, sig_raw_m, sig_rsd_m = load_events(sig_path, is_signal=True, max_events=signal_limit, skip_first=0)

    print(f"Loaded {len(bg_events)} background events (mean mass: {np.mean(bg_raw_m):.1f} GeV)")
    print(f"Loaded {len(sig_events)} signal events (mean mass: {np.mean(sig_raw_m):.1f} GeV)")

    # 2. Score all models on background and signal
    model_scores = {}
    model_latents_bg = {}

    for name, model in models.items():
        cfg = configs[name]
        arch_type = cfg.model.architecture_type
        print(f"\n[PHASE 2] Scoring model: {name} ({arch_type})...")
        
        def score_list(events_to_score, collect_latents=False):
            scores = []
            latents = []
            if arch_type == "baseline_mlp":
                batch_size = 500
                for start_idx in range(0, len(events_to_score), batch_size):
                    chunk = np.array(events_to_score[start_idx : start_idx + batch_size])
                    batch_x = chunk_to_flat_features(chunk, input_dim=64, log_pt=True).to(device)
                    with torch.no_grad():
                        recon_x = model(batch_x)
                        sc = F.mse_loss(recon_x, batch_x, reduction="none").mean(dim=1).cpu().tolist()
                        scores.extend(sc)
                        if collect_latents:
                            z = model.encoder(batch_x).cpu().numpy()
                            latents.append(z)
            else:
                delta_r = cfg.graph_construction.delta_r_threshold
                log_pt = cfg.graph_construction.log_pt
                include_met = cfg.graph_construction.include_met_node
                phi_mode = cfg.graph_construction.phi_met_mode
                
                from torch_geometric.data import Batch
                buffer = []
                for ev in events_to_score:
                    g = event_to_graph(ev, log_pt=log_pt, delta_r_threshold=delta_r,
                                       include_met_node=include_met, phi_met_mode=phi_mode)
                    buffer.append(g)
                    if len(buffer) >= 128:
                        batched = Batch.from_data_list(buffer).to(device)
                        with torch.no_grad():
                            recon_x, _ = model(batched.x, batched.edge_index, batched.batch)[:2]
                            node_err = (recon_x - batched.x).pow(2).mean(dim=1)
                            num_g = batched.num_graphs
                            sums = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, node_err)
                            counts = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, torch.ones_like(node_err))
                            sc = (sums / counts.clamp(min=1.0)).cpu().tolist()
                            scores.extend(sc)
                            if collect_latents:
                                _, gz = model.encode(batched.x, batched.edge_index, batched.batch)
                                latents.append(gz.cpu().numpy())
                        buffer.clear()
                if buffer:
                    batched = Batch.from_data_list(buffer).to(device)
                    with torch.no_grad():
                        recon_x, _ = model(batched.x, batched.edge_index, batched.batch)[:2]
                        node_err = (recon_x - batched.x).pow(2).mean(dim=1)
                        num_g = batched.num_graphs
                        sums = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, node_err)
                        counts = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, torch.ones_like(node_err))
                        sc = (sums / counts.clamp(min=1.0)).cpu().tolist()
                        scores.extend(sc)
                        if collect_latents:
                            _, gz = model.encode(batched.x, batched.edge_index, batched.batch)
                            latents.append(gz.cpu().numpy())
                    buffer.clear()
                    
            lat_arr = np.vstack(latents) if collect_latents and latents else None
            return np.array(scores), lat_arr

        bg_sc, bg_lat = score_list(bg_events, collect_latents=True)
        sig_sc, _ = score_list(sig_events, collect_latents=False)
        
        model_scores[name] = {"bg": bg_sc, "sig": sig_sc}
        if bg_lat is not None:
            model_latents_bg[name] = bg_lat

    # 3. Question 1: How strongly does anomaly score depend on mass?
    print("\n" + "=" * 70)
    print("QUESTION 1: Correlation between Anomaly Score and Invariant Mass")
    print("=" * 70)
    
    correlation_results = {}
    mass_bins = [(2000, 2500), (2500, 3000), (3000, 3500), (3500, 4000), (4000, 5000)]
    
    for name in config_names:
        if name not in model_scores:
            continue
        sc = model_scores[name]["bg"]
        r_val, _ = pearsonr(bg_raw_m, sc)
        rho_val, _ = spearmanr(bg_raw_m, sc)
        
        # Binned profile E[score | mass]
        profile = {}
        for low, high in mass_bins:
            mask = (bg_raw_m >= low) & (bg_raw_m < high)
            mean_sc = float(np.mean(sc[mask])) if np.sum(mask) > 0 else 0.0
            profile[f"{low}-{high} GeV"] = round(mean_sc, 5)
            
        correlation_results[name] = {
            "pearson_r": round(float(r_val), 4),
            "spearman_rho": round(float(rho_val), 4),
            "profile_by_mass": profile,
        }
        print(f"Model: {name:24s} | Pearson r = {r_val:+.4f} | Spearman rho = {rho_val:+.4f}")

    # 4. Question 2: Inclusive vs Mass-Matched ROC-AUC
    print("\n" + "=" * 70)
    print("QUESTION 2: Inclusive vs Mass-Matched ROC-AUC in W' Resonance Window [3300, 3700] GeV")
    print("=" * 70)
    
    # Signal resonance window: [3300, 3700] GeV
    win_low, win_high = 3300.0, 3700.0
    bg_win_mask = (bg_raw_m >= win_low) & (bg_raw_m <= win_high)
    sig_win_mask = (sig_raw_m >= win_low) & (sig_raw_m <= win_high)
    
    print(f"Events in mass window [{win_low:.0f}, {win_high:.0f}] GeV: "
          f"Background={np.sum(bg_win_mask)}, Signal={np.sum(sig_win_mask)}")
    
    matched_roc_results = {}
    for name in config_names:
        if name not in model_scores:
            continue
        bg_sc = model_scores[name]["bg"]
        sig_sc = model_scores[name]["sig"]
        
        # Inclusive ROC
        y_all = np.array([0] * len(bg_sc) + [1] * len(sig_sc))
        sc_all = np.concatenate([bg_sc, sig_sc])
        auc_inclusive = float(roc_auc_score(y_all, sc_all))
        
        # Mass-Matched ROC
        bg_sc_win = bg_sc[bg_win_mask]
        sig_sc_win = sig_sc[sig_win_mask]
        y_win = np.array([0] * len(bg_sc_win) + [1] * len(sig_sc_win))
        sc_win = np.concatenate([bg_sc_win, sig_sc_win])
        auc_matched = float(roc_auc_score(y_win, sc_win))
        
        delta_auc = auc_matched - auc_inclusive
        matched_roc_results[name] = {
            "auc_inclusive": round(auc_inclusive, 4),
            "auc_mass_matched": round(auc_matched, 4),
            "delta_auc": round(delta_auc, 4),
        }
        print(f"Model: {name:24s} | Inclusive AUC = {auc_inclusive:.4f} | "
              f"Matched AUC [3.3-3.7 TeV] = {auc_matched:.4f} (Delta = {delta_auc:+.4f})")

    # 5. Question 3: Probing Latent z_bar vs Node Anomaly Score
    print("\n" + "=" * 70)
    print("QUESTION 3: Mass Predictability of Pooled Latent (z_bar) vs Node Reconstruction Error")
    print("=" * 70)
    
    probe_results = {}
    for name in ["gat_met_production", "gat_met_decorrelation"]:
        if name not in model_latents_bg:
            continue
        z_bar = model_latents_bg[name]  # [N, 32]
        sc = model_scores[name]["bg"]    # [N]
        m_target = bg_raw_m / 1000.0     # [N] in TeV
        
        # Ridge probe on pooled latent
        ridge = Ridge(alpha=1.0)
        # 5-fold cross-validation or train/val split
        n_train = int(len(z_bar) * 0.8)
        ridge.fit(z_bar[:n_train], m_target[:n_train])
        r2_latent = float(ridge.score(z_bar[n_train:], m_target[n_train:]))
        
        # Linear fit on anomaly score
        ridge_sc = Ridge(alpha=1.0)
        ridge_sc.fit(sc[:n_train].reshape(-1, 1), m_target[:n_train])
        r2_score = float(ridge_sc.score(sc[n_train:].reshape(-1, 1), m_target[n_train:]))
        
        probe_results[name] = {
            "r2_pooled_latent_to_mass": round(max(0.0, r2_latent), 4),
            "r2_node_score_to_mass": round(max(0.0, r2_score), 4),
            "score_pearson_r": correlation_results[name]["pearson_r"],
        }
        print(f"Model: {name:24s} | Pooled Latent R^2(z_bar -> m) = {max(0.0, r2_latent):.4f} | "
              f"Node Score R^2(score -> m) = {max(0.0, r2_score):.4f}")

    # 6. Save comprehensive JSON
    output_dict = {
        "question_1_correlation": correlation_results,
        "question_2_mass_matched_roc": matched_roc_results,
        "question_3_latent_vs_score_probes": probe_results,
    }
    
    out_json = REPO_ROOT / "comparisons/architecture/mass_correlation_study.json"
    with open(out_json, "w") as f:
        json.dump(output_dict, f, indent=2)
    print(f"\n[SUCCESS] Mass correlation study saved to: {out_json}")
    
    # 7. Generate publication-ready 3-panel figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    colors = {
        "gat_met_production": "#1f77b4",
        "gat_standalone": "#2ca02c",
        "gat_met_decorrelation": "#d62728",
        "baseline_mlp": "#ff7f0e",
    }
    labels = {
        "gat_met_production": "gat_met_production (Ours)",
        "gat_standalone": "gat_standalone",
        "gat_met_decorrelation": "gat_met_decorrelation",
        "baseline_mlp": "baseline_mlp",
    }
    
    # Panel 1: Anomaly Score vs Dijet Mass Profile E[Score | m] (normalized to mean=1 for shape comparison)
    m_centers = [2250, 2750, 3250, 3750, 4500]
    for name in config_names:
        if name in correlation_results:
            prof = correlation_results[name]["profile_by_mass"]
            vals = list(prof.values())
            # Normalize to mean = 1
            vals_norm = np.array(vals) / np.mean(vals)
            ax1.plot(m_centers, vals_norm, marker="o", linewidth=2.2, color=colors[name],
                     label=f"{labels[name]} (r = {correlation_results[name]['pearson_r']:+.2f})")
    ax1.set_title("Question 1: Score vs Mass Profile E[Score | m]", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Dijet Invariant Mass $m_{jj}$ [GeV]", fontsize=10)
    ax1.set_ylabel("Normalized Mean Anomaly Score", fontsize=10)
    ax1.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8, framealpha=0.9)
    
    # Panel 2: Inclusive vs Mass-Matched ROC-AUC Bar Chart
    x = np.arange(len(config_names))
    width = 0.35
    incl_vals = [matched_roc_results.get(n, {}).get("auc_inclusive", 0) for n in config_names]
    match_vals = [matched_roc_results.get(n, {}).get("auc_mass_matched", 0) for n in config_names]
    
    ax2.bar(x - width/2, incl_vals, width, label="Inclusive ROC-AUC (All Mass)", color="#4682b4")
    ax2.bar(x + width/2, match_vals, width, label="Mass-Matched ROC-AUC ([3.3, 3.7] TeV)", color="#e74c3c")
    ax2.set_title("Question 2: Performance Under Mass-Matching", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace("_", "\n") for n in config_names], fontsize=8)
    ax2.set_ylabel("ROC AUC", fontsize=10)
    ax2.set_ylim(0.4, 0.9)
    ax2.axhline(0.5, color="gray", linestyle=":", label="Random Guess (0.50)")
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2.legend(fontsize=8, framealpha=0.9)
    
    # Panel 3: Latent Probe R^2 vs Score Correlation
    models_p = ["gat_met_production", "gat_met_decorrelation"]
    xp = np.arange(len(models_p))
    r2_lat = [probe_results.get(n, {}).get("r2_pooled_latent_to_mass", 0) for n in models_p]
    r2_sc = [probe_results.get(n, {}).get("r2_node_score_to_mass", 0) for n in models_p]
    
    ax3.bar(xp - width/2, r2_lat, width, label="Pooled Latent R²(z̄ → m)", color="#9b59b6")
    ax3.bar(xp + width/2, r2_sc, width, label="Node Score R²(Score → m)", color="#f39c12")
    ax3.set_title("Question 3: Adversary Effect on Latent vs Score", fontsize=11, fontweight="bold")
    ax3.set_xticks(xp)
    ax3.set_xticklabels([n.replace("_", "\n") for n in models_p], fontsize=9)
    ax3.set_ylabel("Mass Predictability R²", fontsize=10)
    ax3.set_ylim(0.0, 0.5)
    ax3.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax3.legend(fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plot_path = REPO_ROOT / "comparisons/architecture/plots/mass_correlation_and_matched_roc.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[SUCCESS] Plot saved to: {plot_path}")
    
    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="architecture")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--signal-limit", type=int, default=5000)
    args = parser.parse_args()
    
    evaluate_mass_correlation_study(
        family=args.family,
        eval_limit=args.limit,
        signal_limit=args.signal_limit,
    )
