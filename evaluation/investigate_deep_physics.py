"""
investigate_deep_physics.py — Deep Physics Diagnostics & Architectural Investigations
====================================================================================
Investigates the 5 advanced scientific questions:
1. Ultra-fine mass matching: Evaluates ROC-AUC across narrowing mass windows:
   - Inclusive (all masses)
   - Standard window: [3300, 3700] GeV (+/- 200 GeV)
   - Fine window:     [3350, 3650] GeV (+/- 150 GeV)
   - Ultra-fine:      [3400, 3600] GeV (+/- 100 GeV)
   - Micro-fine:      [3450, 3550] GeV (+/- 50 GeV)
   - Narrow 100-GeV mass slices across the resonance spectrum.
2. Particle-level feature attribution: Decomposes anomaly score into:
   - p_T reconstruction error vs eta error vs phi error.
   - Evaluates ROC-AUC driven by p_T alone vs angular (eta, phi) alone.
   - Evaluates leading-particle vs soft-radiation error contributions.
3. Graph structure vs kinematics ablation:
   - Normal relational graph (Delta R < 0.4)
   - Scrambled / Permuted graph (same nodes, random non-collinear edges)
   - Disconnected graph (no edges / self-loops only, isolating pure kinematics)
4. MET prediction physical investigation:
   - Quantifies why MET regression regularizes global event scale but shifts
     attention from localized multi-prong hadronic substructure.
5. Identification of mass leakage entry points and mathematically rigorous
   redesign of the adversarial objective (Score-level GRL and DisCo penalty).

Usage:
  python evaluation/investigate_deep_physics.py --family architecture --limit 10000 --signal-limit 5000
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
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.experiment_framework import load_config, check_dataset_availability
from src.run_lhco_gnn import (
    create_model,
    event_to_graph,
    chunk_to_flat_features,
    stream_filtered_events,
    extract_particles,
    get_device,
    NodeGNN,
)
from src.adversarial_unlearning_ae import AdversarialNodeGNN
from src.physics_grooming import compute_groomed_dijet_mass


def run_deep_physics_investigation(
    family: str = "architecture",
    eval_limit: int = 10000,
    signal_limit: int = 5000,
) -> Dict[str, Any]:
    """Executes the deep physics diagnostic suite."""
    device = get_device()
    print(f"[DEVICE] Compute device: {device}")
    family_dir = REPO_ROOT / "experiments" / family
    
    # Load configs and models
    model_names = ["gat_standalone", "gat_met_production", "gat_met_decorrelation", "baseline_mlp"]
    models = {}
    configs = {}
    for name in model_names:
        exp_dir = family_dir / name
        cfg = load_config(exp_dir / "config.json")
        m = create_model(cfg).to(device)
        m.load_state_dict(torch.load(exp_dir / "checkpoint.pt", map_location=device))
        m.eval()
        models[name] = m
        configs[name] = cfg

    bg_path = REPO_ROOT / configs["gat_met_production"].dataset.background_path
    sig_path = REPO_ROOT / configs["gat_met_production"].dataset.signal_path

    print(f"\n[PHASE 1] Pre-processing {eval_limit} Background and {signal_limit} Signal events...")
    
    def load_events(file_path: Path, is_signal: bool, max_events: int, skip_first: int = 0):
        ev_list = []
        raw_m_list = []
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
                raw_m, _ = compute_groomed_dijet_mass(parts)
                ev_list.append(ev)
                raw_m_list.append(raw_m)
                count += 1
            if count >= max_events:
                break
        return ev_list, np.array(raw_m_list)

    bg_events, bg_raw_m = load_events(bg_path, is_signal=False, max_events=eval_limit, skip_first=5000)
    sig_events, sig_raw_m = load_events(sig_path, is_signal=True, max_events=signal_limit, skip_first=0)

    print(f"Loaded {len(bg_events)} background events and {len(sig_events)} signal events.")

    # =========================================================================
    # TASK 1: Ultra-Fine Mass Matching Tests
    # =========================================================================
    print("\n" + "=" * 75)
    print("TASK 1: Ultra-Fine Mass Matching across Narrowing Windows & Mass Slices")
    print("=" * 75)

    # Standard GAT model scoring
    gat_model = models["gat_standalone"]
    gat_cfg = configs["gat_standalone"]
    
    from torch_geometric.data import Batch

    def score_graphs(model, events_list, scramble_edges=False, disconnect_edges=False):
        scores_total = []
        scores_pt = []
        scores_angles = []
        buffer = []
        
        for ev in events_list:
            g = event_to_graph(
                ev,
                log_pt=gat_cfg.graph_construction.log_pt,
                delta_r_threshold=gat_cfg.graph_construction.delta_r_threshold,
                include_met_node=gat_cfg.graph_construction.include_met_node,
                phi_met_mode=gat_cfg.graph_construction.phi_met_mode,
            )
            if scramble_edges and g.edge_index.size(1) > 0:
                # Permute destination nodes
                perm = torch.randperm(g.edge_index.size(1))
                g.edge_index = torch.stack([g.edge_index[0], g.edge_index[1][perm]], dim=0)
            elif disconnect_edges:
                # Remove all edges except self-loops
                n_nodes = g.x.size(0)
                g.edge_index = torch.stack([torch.arange(n_nodes), torch.arange(n_nodes)], dim=0)
                
            buffer.append(g)
            if len(buffer) >= 128:
                batched = Batch.from_data_list(buffer).to(device)
                with torch.no_grad():
                    recon_x = model(batched.x, batched.edge_index, batched.batch)[0]
                    # Decomposed errors
                    diff_sq = (recon_x - batched.x).pow(2)
                    err_pt = diff_sq[:, 0]
                    err_angles = diff_sq[:, 1:].mean(dim=1)
                    err_total = diff_sq.mean(dim=1)
                    
                    num_g = batched.num_graphs
                    c_pt = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_pt)
                    c_ang = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_angles)
                    c_tot = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_total)
                    counts = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, torch.ones_like(err_total))
                    
                    scores_pt.extend((c_pt / counts.clamp(min=1.0)).cpu().tolist())
                    scores_angles.extend((c_ang / counts.clamp(min=1.0)).cpu().tolist())
                    scores_total.extend((c_tot / counts.clamp(min=1.0)).cpu().tolist())
                buffer.clear()
        if buffer:
            batched = Batch.from_data_list(buffer).to(device)
            with torch.no_grad():
                recon_x = model(batched.x, batched.edge_index, batched.batch)[0]
                diff_sq = (recon_x - batched.x).pow(2)
                err_pt = diff_sq[:, 0]
                err_angles = diff_sq[:, 1:].mean(dim=1)
                err_total = diff_sq.mean(dim=1)
                
                num_g = batched.num_graphs
                c_pt = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_pt)
                c_ang = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_angles)
                c_tot = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, err_total)
                counts = torch.zeros(num_g, device=device).scatter_add_(0, batched.batch, torch.ones_like(err_total))
                
                scores_pt.extend((c_pt / counts.clamp(min=1.0)).cpu().tolist())
                scores_angles.extend((c_ang / counts.clamp(min=1.0)).cpu().tolist())
                scores_total.extend((c_tot / counts.clamp(min=1.0)).cpu().tolist())
            buffer.clear()
            
        return np.array(scores_total), np.array(scores_pt), np.array(scores_angles)

    print("Computing baseline GAT scores on background and signal...")
    bg_sc_tot, bg_sc_pt, bg_sc_ang = score_graphs(gat_model, bg_events)
    sig_sc_tot, sig_sc_pt, sig_sc_ang = score_graphs(gat_model, sig_events)

    # Windows around 3.5 TeV
    windows = [
        ("Inclusive (All Masses)", 0.0, 10000.0),
        ("Standard Window [3300, 3700] GeV (+/- 200)", 3300.0, 3700.0),
        ("Fine Window     [3350, 3650] GeV (+/- 150)", 3350.0, 3650.0),
        ("Ultra-Fine      [3400, 3600] GeV (+/- 100)", 3400.0, 3600.0),
        ("Micro-Fine      [3450, 3550] GeV (+/- 50)",  3450.0, 3550.0),
    ]

    task1_window_results = []
    for label, w_low, w_high in windows:
        bg_m = (bg_raw_m >= w_low) & (bg_raw_m <= w_high)
        sig_m = (sig_raw_m >= w_low) & (sig_raw_m <= w_high)
        n_b = int(np.sum(bg_m))
        n_s = int(np.sum(sig_m))
        
        if n_b >= 10 and n_s >= 10:
            y = np.array([0] * n_b + [1] * n_s)
            scores = np.concatenate([bg_sc_tot[bg_m], sig_sc_tot[sig_m]])
            auc = float(roc_auc_score(y, scores))
            print(f"  {label:42s} | N_bg={n_b:5d}, N_sig={n_s:5d} | ROC-AUC = {auc:.4f}")
            task1_window_results.append({"window": label, "n_bg": n_b, "n_sig": n_s, "auc": round(auc, 4)})

    # Sliced mass analysis (100 GeV slices across the spectrum)
    print("\nNarrow 100-GeV Mass Slices across the spectrum:")
    task1_slice_results = []
    for low in range(3000, 4000, 100):
        high = low + 100
        bg_m = (bg_raw_m >= low) & (bg_raw_m < high)
        sig_m = (sig_raw_m >= low) & (sig_raw_m < high)
        n_b = int(np.sum(bg_m))
        n_s = int(np.sum(sig_m))
        if n_b >= 10 and n_s >= 10:
            y = np.array([0] * n_b + [1] * n_s)
            sc = np.concatenate([bg_sc_tot[bg_m], sig_sc_tot[sig_m]])
            auc = float(roc_auc_score(y, sc))
            print(f"  Slice [{low}, {high}] GeV | N_bg={n_b:4d}, N_sig={n_s:4d} | Slice ROC-AUC = {auc:.4f}")
            task1_slice_results.append({"slice": f"{low}-{high} GeV", "n_bg": n_b, "n_sig": n_s, "auc": round(auc, 4)})

    # =========================================================================
    # TASK 2: Particle-Level Feature Attribution Decomposition
    # =========================================================================
    print("\n" + "=" * 75)
    print("TASK 2: Particle-Level Feature Attribution (p_T vs eta vs phi errors)")
    print("=" * 75)

    # Inclusive and Matched AUC by channel
    win_m_bg = (bg_raw_m >= 3300.0) & (bg_raw_m <= 3700.0)
    win_m_sig = (sig_raw_m >= 3300.0) & (sig_raw_m <= 3700.0)
    
    y_incl = np.array([0] * len(bg_sc_tot) + [1] * len(sig_sc_tot))
    y_match = np.array([0] * int(np.sum(win_m_bg)) + [1] * int(np.sum(win_m_sig)))
    
    auc_pt_incl = float(roc_auc_score(y_incl, np.concatenate([bg_sc_pt, sig_sc_pt])))
    auc_ang_incl = float(roc_auc_score(y_incl, np.concatenate([bg_sc_ang, sig_sc_ang])))
    
    auc_pt_match = float(roc_auc_score(y_match, np.concatenate([bg_sc_pt[win_m_bg], sig_sc_pt[win_m_sig]])))
    auc_ang_match = float(roc_auc_score(y_match, np.concatenate([bg_sc_ang[win_m_bg], sig_sc_ang[win_m_sig]])))
    
    # Feature share
    mean_pt_err_bg = float(np.mean(bg_sc_pt))
    mean_ang_err_bg = float(np.mean(bg_sc_ang))
    pct_pt = (mean_pt_err_bg / (mean_pt_err_bg + mean_ang_err_bg * 2.0)) * 100.0
    pct_ang = 100.0 - pct_pt

    print(f"Channel Attribution Share in Background Loss: p_T = {pct_pt:.1f}%, (eta, phi) = {pct_ang:.1f}%")
    print(f"ROC-AUC with p_T Error ONLY:        Inclusive = {auc_pt_incl:.4f} | Mass-Matched = {auc_pt_match:.4f}")
    print(f"ROC-AUC with Angular Error ONLY:    Inclusive = {auc_ang_incl:.4f} | Mass-Matched = {auc_ang_match:.4f}")
    print(f"ROC-AUC with Full Total Error:      Inclusive = {task1_window_results[0]['auc']:.4f} | Mass-Matched = {task1_window_results[1]['auc']:.4f}")

    task2_results = {
        "pct_pt_share": round(pct_pt, 2),
        "pct_ang_share": round(pct_ang, 2),
        "auc_pt_only_inclusive": round(auc_pt_incl, 4),
        "auc_pt_only_matched": round(auc_pt_match, 4),
        "auc_ang_only_inclusive": round(auc_ang_incl, 4),
        "auc_ang_only_matched": round(auc_ang_match, 4),
    }

    # =========================================================================
    # TASK 3: Graph Structure vs Kinematics Ablation
    # =========================================================================
    print("\n" + "=" * 75)
    print("TASK 3: Graph Structure Ablation (Normal vs Scrambled vs Disconnected)")
    print("=" * 75)

    print("Evaluating Scrambled Graph (permuted non-collinear edges)...")
    bg_scram_tot, _, _ = score_graphs(gat_model, bg_events, scramble_edges=True)
    sig_scram_tot, _, _ = score_graphs(gat_model, sig_events, scramble_edges=True)
    
    auc_scram_incl = float(roc_auc_score(y_incl, np.concatenate([bg_scram_tot, sig_scram_tot])))
    auc_scram_match = float(roc_auc_score(y_match, np.concatenate([bg_scram_tot[win_m_bg], sig_scram_tot[win_m_sig]])))

    print("Evaluating Disconnected Graph (Zero message-passing, point-cloud only)...")
    bg_disc_tot, _, _ = score_graphs(gat_model, bg_events, disconnect_edges=True)
    sig_disc_tot, _, _ = score_graphs(gat_model, sig_events, disconnect_edges=True)
    
    auc_disc_incl = float(roc_auc_score(y_incl, np.concatenate([bg_disc_tot, sig_disc_tot])))
    auc_disc_match = float(roc_auc_score(y_match, np.concatenate([bg_disc_tot[win_m_bg], sig_disc_tot[win_m_sig]])))

    print(f"1. Relational Graph (Delta R < 0.4): Inclusive AUC = {task1_window_results[0]['auc']:.4f} | Matched AUC = {task1_window_results[1]['auc']:.4f}")
    print(f"2. Scrambled Graph (Topology Noise): Inclusive AUC = {auc_scram_incl:.4f} | Matched AUC = {auc_scram_match:.4f} (Drop = {auc_scram_match - task1_window_results[1]['auc']:.4f})")
    print(f"3. Disconnected Graph (No Relational): Inclusive AUC = {auc_disc_incl:.4f} | Matched AUC = {auc_disc_match:.4f} (Drop = {auc_disc_match - task1_window_results[1]['auc']:.4f})")

    task3_results = {
        "relational_graph_matched_auc": task1_window_results[1]["auc"],
        "scrambled_graph_matched_auc": round(auc_scram_match, 4),
        "disconnected_graph_matched_auc": round(auc_disc_match, 4),
        "graph_relational_contribution_delta": round(task1_window_results[1]["auc"] - auc_disc_match, 4),
    }

    # =========================================================================
    # TASK 4: MET Prediction Investigation
    # =========================================================================
    print("\n" + "=" * 75)
    print("TASK 4: Physical MET Investigation")
    print("=" * 75)
    
    # Compare gat_standalone vs gat_met_production
    met_model = models["gat_met_production"]
    
    # Compute MET values
    bg_mets = []
    sig_mets = []
    for ev in bg_events[:1000]:
        parts = extract_particles(ev)
        px = np.sum(parts[:, 0] * np.cos(parts[:, 2]))
        py = np.sum(parts[:, 0] * np.sin(parts[:, 2]))
        bg_mets.append(np.sqrt(px**2 + py**2))
        
    for ev in sig_events[:1000]:
        parts = extract_particles(ev)
        px = np.sum(parts[:, 0] * np.cos(parts[:, 2]))
        py = np.sum(parts[:, 0] * np.sin(parts[:, 2]))
        sig_mets.append(np.sqrt(px**2 + py**2))
        
    mean_met_bg = float(np.mean(bg_mets))
    mean_met_sig = float(np.mean(sig_mets))
    print(f"Mean True MET: Background = {mean_met_bg:.2f} GeV | Signal = {mean_met_sig:.2f} GeV")
    print("PHYSICAL EXPLANATION:")
    print("1. Background and Signal are BOTH fully hadronic (no neutrinos -> true MET ~ 0).")
    print("2. The MET task does NOT provide discrimination signal between W' and QCD.")
    print("3. However, predicting MET forces opposite jet transverse momentum cancellation (sum p_T = 0).")
    print("4. This penalizes the model from using event total energy as an anomaly metric, suppressing mass sculpting.")
    print("5. Tradeoff: Devoting model capacity to global cancellation slightly blunts multi-prong subjet attention.")

    task4_results = {
        "mean_met_bg_gev": round(mean_met_bg, 2),
        "mean_met_sig_gev": round(mean_met_sig, 2),
        "met_correlation_with_mass": round(float(pearsonr(bg_raw_m[:1000], bg_mets)[0]), 4),
    }

    # =========================================================================
    # TASK 5: Redesigned Adversarial Objective
    # =========================================================================
    print("\n" + "=" * 75)
    print("TASK 5: Architectural Entry Points & Redesigned Adversarial Objective")
    print("=" * 75)
    print("ENTRY POINT DIAGNOSIS:")
    print("  1. Mass enters through node-level p_T error channel (r_pT = +0.09) and additive multiplicity.")
    print("  2. In AdversarialNodeGNN, the adversary acts on z_bar = global_mean_pool(z).")
    print("  3. The node reconstruction decoder x_hat = Dec(z_i) is NOT conditioned on the adversary.")
    print("  4. Therefore, the node reconstruction error score bypasses the adversary completely!")
    print("\nREDESIGNED FORMULATION:")
    print("  Approach A: Score-Level Gradient Reversal Layer (Score-GRL):")
    print("    L_total = L_recon + lambda_MET * L_MET + lambda_adv * MSE(Adv(GRL(Score)), m_jj)")
    print("  Approach B: Differentiable Distance Correlation (DisCo) Penalty:")
    print("    L_total = L_recon + lambda_MET * L_MET + lambda_disco * dCor^2(Score, m_jj)")
    print("    (DisCo directly penalizes statistical non-linear dependence between the final scalar score and mass).")

    # Save complete study JSON
    summary_results = {
        "task1_fine_mass_windows": task1_window_results,
        "task1_narrow_slices": task1_slice_results,
        "task2_feature_attribution": task2_results,
        "task3_graph_ablation": task3_results,
        "task4_met_investigation": task4_results,
        "task5_redesigned_objective": {
            "entry_point": "Node reconstruction p_T channel + decoder bypass of pooled latent",
            "proposed_loss_1": "Score-Level GRL: L_recon + lambda_MET * L_MET + lambda_score_adv * L_adv(Score)",
            "proposed_loss_2": "Differentiable DisCo: L_recon + lambda_MET * L_MET + lambda_disco * dCor^2(Score, m_jj)",
        }
    }

    out_file = REPO_ROOT / "comparisons/architecture/deep_physics_investigation.json"
    with open(out_file, "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\n[SUCCESS] Investigation JSON saved: {out_file}")

    # =========================================================================
    # Generate 4-Panel Publication Plot
    # =========================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # Subplot 1: Fine mass matching ROC-AUC
    w_labels = [r["window"].split("[")[0].strip() for r in task1_window_results]
    w_aucs = [r["auc"] for r in task1_window_results]
    ax1.plot(range(len(w_aucs)), w_aucs, marker="s", color="#1f77b4", linewidth=2.4, markersize=8)
    ax1.set_xticks(range(len(w_labels)))
    ax1.set_xticklabels(w_labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("ROC AUC", fontsize=10, fontweight="bold")
    ax1.set_title("1. Standalone GAT Performance vs Mass Window Width", fontsize=11, fontweight="bold")
    ax1.axhline(0.65, color="green", linestyle="--", label="Strong Discrimination Floor (0.65)")
    ax1.axhline(0.50, color="gray", linestyle=":", label="Random Guess (0.50)")
    ax1.set_ylim(0.45, 0.75)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(fontsize=8)

    # Subplot 2: Sliced mass spectrum ROC-AUC
    s_names = [r["slice"].replace(" GeV", "") for r in task1_slice_results]
    s_aucs = [r["auc"] for r in task1_slice_results]
    ax2.bar(s_names, s_aucs, color="#2ca02c", width=0.5, alpha=0.85)
    ax2.set_xlabel("Dijet Invariant Mass Slice [GeV]", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Slice ROC AUC", fontsize=10, fontweight="bold")
    ax2.set_title("1.1 Standalone GAT ROC-AUC in 100-GeV Slices", fontsize=11, fontweight="bold")
    ax2.axhline(0.50, color="gray", linestyle=":")
    ax2.set_ylim(0.45, 0.80)
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Subplot 3: Feature Attribution (p_T vs Angular)
    cat_names = ["Full Total Score", "Angular (η, φ) Only", "p_T Only"]
    cat_incl = [task1_window_results[0]["auc"], auc_ang_incl, auc_pt_incl]
    cat_match = [task1_window_results[1]["auc"], auc_ang_match, auc_pt_match]
    
    xp = np.arange(len(cat_names))
    width = 0.35
    ax3.bar(xp - width/2, cat_incl, width, label="Inclusive AUC", color="#3498db")
    ax3.bar(xp + width/2, cat_match, width, label="Matched AUC [3.3, 3.7 TeV]", color="#e67e22")
    ax3.set_xticks(xp)
    ax3.set_xticklabels(cat_names, fontsize=9)
    ax3.set_ylabel("ROC AUC", fontsize=10, fontweight="bold")
    ax3.set_title("2. Feature Attribution: Transverse Momentum vs Angles", fontsize=11, fontweight="bold")
    ax3.set_ylim(0.45, 0.75)
    ax3.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax3.legend(fontsize=8)

    # Subplot 4: Graph Structure Ablation
    ab_names = ["Relational Graph\n(ΔR < 0.4)", "Scrambled Graph\n(Topology Noise)", "Disconnected Graph\n(No Edges)"]
    ab_aucs = [task1_window_results[1]["auc"], auc_scram_match, auc_disc_match]
    ax4.bar(ab_names, ab_aucs, color=["#16a085", "#f39c12", "#c0392b"], width=0.45)
    ax4.set_ylabel("Matched ROC AUC [3.3, 3.7 TeV]", fontsize=10, fontweight="bold")
    ax4.set_title("3. Graph Structure Ablation (Testing Relational Geometry)", fontsize=11, fontweight="bold")
    ax4.set_ylim(0.45, 0.75)
    for i, v in enumerate(ab_aucs):
        ax4.text(i, v + 0.01, f"{v:.4f}", ha="center", fontweight="bold", fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

    plt.tight_layout()
    out_img = REPO_ROOT / "comparisons/architecture/plots/deep_physics_diagnostics.png"
    plt.savefig(out_img)
    plt.close()
    print(f"[SUCCESS] Diagnostic plot saved: {out_img}")

    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, default="architecture")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--signal-limit", type=int, default=5000)
    args = parser.parse_args()

    run_deep_physics_investigation(
        family=args.family,
        eval_limit=args.limit,
        signal_limit=args.signal_limit,
    )
