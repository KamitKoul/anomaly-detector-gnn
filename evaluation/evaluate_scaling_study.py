"""
evaluation/evaluate_scaling_study.py — Frozen Evaluation Engine for Scaling Experiments
========================================================================================
Implements the frozen Phase 7 evaluation protocol:
  1. Loads held-out test background (bg_test_50k.npy) and test signal (sig_test_25k.npy).
  2. Evaluates the 4 standardized anomaly scores:
       - Full reconstruction error
       - pT-only reconstruction error
       - Angular-only (eta, phi) reconstruction error
       - Weighted reconstruction error: 0.8 * S_pT + 0.2 * S_ang
  3. Computes:
       - Inclusive ROC-AUC
       - Mass-matched ROC-AUC ([3.3, 3.7] TeV)
       - Narrow-window ROC-AUC ([3.45, 3.55] TeV)
       - Background rejection (1/FPR) at 30% and 50% signal efficiency
       - Continuous Significance Improvement Characteristic (SIC) and Max SIC
       - Raw and RSD mass sculpting (Base-2 JSD and 1/JSD) at Top 10%, 5%, 1% cuts
       - Linear & rank mass correlations (Pearson r, Spearman rho, DisCo)
       - 68.3% stratified bootstrap confidence intervals
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
from torch_geometric.data import Batch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.run_lhco_gnn import (
    PointNetGATHybrid,
    extract_particles,
    event_to_graph,
    get_device,
)
from src.physics_grooming import compute_groomed_dijet_mass
from src.adversarial_unlearning_ae import compute_distance_correlation


def compute_base2_jsd(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    bins: int = 40,
    range_gev: Tuple[float, float] = (2000.0, 5000.0),
) -> Tuple[float, float, str, int, int]:
    """Computes base-2 Jensen-Shannon Divergence and 1/JSD.
    Returns (js_div, inv_jsd, status, empty_bins_p, empty_bins_q).
    """
    if len(p_samples) < 10 or len(q_samples) < 10:
        return float("nan"), float("nan"), "INSUFFICIENT_STATISTICS", 0, 0

    p_counts, _ = np.histogram(p_samples, bins=bins, range=range_gev)
    q_counts, _ = np.histogram(q_samples, bins=bins, range=range_gev)

    empty_p = int(np.sum(p_counts == 0))
    empty_q = int(np.sum(q_counts == 0))

    p = p_counts.astype(np.float64) / (np.sum(p_counts) + 1e-12)
    q = q_counts.astype(np.float64) / (np.sum(q_counts) + 1e-12)

    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()

    js_div = float(jensenshannon(p, q, base=2.0) ** 2)
    inv_jsd = 1.0 / js_div if js_div > 1e-9 else float("inf")
    return js_div, inv_jsd, "VALID", empty_p, empty_q


def compute_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes stratified bootstrap estimate and 68.3% confidence interval for ROC-AUC."""
    rng = np.random.RandomState(seed)
    idx_bg = np.where(y_true == 0)[0]
    idx_sig = np.where(y_true == 1)[0]

    n_bg = len(idx_bg)
    n_sig = len(idx_sig)

    boot_aucs = []
    for _ in range(n_bootstraps):
        sample_bg = rng.choice(idx_bg, size=n_bg, replace=True)
        sample_sig = rng.choice(idx_sig, size=n_sig, replace=True)
        boot_idx = np.concatenate([sample_bg, sample_sig])

        try:
            auc = roc_auc_score(y_true[boot_idx], scores[boot_idx])
            boot_aucs.append(auc)
        except Exception:
            continue

    if not boot_aucs:
        return {"mean": float(roc_auc_score(y_true, scores)), "ci_low": 0.0, "ci_high": 1.0, "std": 0.0}

    boot_aucs = np.array(boot_aucs)
    mean_auc = float(np.mean(boot_aucs))
    ci_low = float(np.percentile(boot_aucs, 15.865))
    ci_high = float(np.percentile(boot_aucs, 84.135))
    std_auc = float(np.std(boot_aucs, ddof=1))

    return {
        "mean": round(mean_auc, 5),
        "ci_low": round(ci_low, 5),
        "ci_high": round(ci_high, 5),
        "std": round(std_auc, 5),
    }


def compute_rejection_at_efficiencies(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Computes background rejection (1/FPR) and SIC at 30% and 50% signal efficiency."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    eps = 1e-9

    res = {}
    for eff in [0.30, 0.50]:
        idx = np.where(tpr >= eff)[0]
        if len(idx) > 0 and fpr[idx[0]] > 0:
            rej = float(1.0 / fpr[idx[0]])
            sic = float(tpr[idx[0]] / np.sqrt(max(fpr[idx[0]], eps)))
        else:
            rej = -1.0
            sic = 0.0
        res[f"rejection_at_{int(eff*100)}_eff"] = round(rej, 2)
        res[f"sic_at_{int(eff*100)}_eff"] = round(sic, 3)

    # Max SIC curve
    sic_curve = tpr / np.sqrt(np.clip(fpr, 1e-6, 1.0))
    res["max_sic"] = round(float(np.max(sic_curve)), 3)
    return res


def evaluate_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    bg_events: List[np.ndarray],
    bg_raw_m: np.ndarray,
    bg_rsd_m: np.ndarray,
    sig_events: List[np.ndarray],
    sig_raw_m: np.ndarray,
    sig_rsd_m: np.ndarray,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """Evaluates a single model checkpoint across the held-out test dataset."""
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model = PointNetGATHybrid(input_dim=3, hidden_dim=32, num_heads=4).to(device)

    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.eval()

    def score_population(events_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        sc_full, sc_pt, sc_ang, sc_w = [], [], [], []
        buffer = []

        for ev in events_list:
            parts = extract_particles(ev)
            g = event_to_graph(particles=parts)
            buffer.append(g)

            if len(buffer) >= batch_size:
                batched = Batch.from_data_list(buffer).to(device)
                with torch.no_grad():
                    recon_x, _ = model(batched.x, batched.edge_index, batched.batch)
                    err_pt = (recon_x[:, 0] - batched.x[:, 0]).pow(2)
                    err_eta = (recon_x[:, 1] - batched.x[:, 1]).pow(2)
                    dphi = torch.abs(recon_x[:, 2] - batched.x[:, 2])
                    err_phi = torch.min(dphi, 2.0 * np.pi - dphi).pow(2)
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
                recon_x, _ = model(batched.x, batched.edge_index, batched.batch)
                err_pt = (recon_x[:, 0] - batched.x[:, 0]).pow(2)
                err_eta = (recon_x[:, 1] - batched.x[:, 1]).pow(2)
                dphi = torch.abs(recon_x[:, 2] - batched.x[:, 2])
                err_phi = torch.min(dphi, 2.0 * np.pi - dphi).pow(2)
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

    bg_scores = score_population(bg_events)
    sig_scores = score_population(sig_events)

    n_bg = len(bg_events)
    n_sig = len(sig_events)
    y_all = np.array([0] * n_bg + [1] * n_sig)

    # Mass windows
    win_matched_bg = (bg_raw_m >= 3300.0) & (bg_raw_m <= 3700.0)
    win_matched_sig = (sig_raw_m >= 3300.0) & (sig_raw_m <= 3700.0)
    y_matched = np.array([0] * np.sum(win_matched_bg) + [1] * np.sum(win_matched_sig))

    win_narrow_bg = (bg_raw_m >= 3450.0) & (bg_raw_m <= 3550.0)
    win_narrow_sig = (sig_raw_m >= 3450.0) & (sig_raw_m <= 3550.0)
    y_narrow = np.array([0] * np.sum(win_narrow_bg) + [1] * np.sum(win_narrow_sig))

    results_by_score = {}

    for score_name in ["full", "pt_only", "ang_only", "weighted"]:
        b_sc = bg_scores[score_name]
        s_sc = sig_scores[score_name]
        all_sc = np.concatenate([b_sc, s_sc])

        # 1. Discrimination Metrics
        auc_inclusive = compute_bootstrap_ci(y_all, all_sc)

        matched_sc = np.concatenate([b_sc[win_matched_bg], s_sc[win_matched_sig]])
        auc_matched = compute_bootstrap_ci(y_matched, matched_sc)

        narrow_sc = np.concatenate([b_sc[win_narrow_bg], s_sc[win_narrow_sig]])
        auc_narrow = compute_bootstrap_ci(y_narrow, narrow_sc)

        rejection_metrics = compute_rejection_at_efficiencies(y_matched, matched_sc)

        # 2. Linear and Rank Correlations on Background
        r_val, _ = pearsonr(bg_raw_m, b_sc)
        rho_val, _ = spearmanr(bg_raw_m, b_sc)
        disco_val = compute_distance_correlation(torch.tensor(b_sc[:1000]), torch.tensor(bg_raw_m[:1000])).item()

        # 3. Mass Sculpting (JSD and 1/JSD)
        sculpting_cuts = {}
        for pct in [10, 5, 1]:
            cut_threshold = np.percentile(b_sc, 100 - pct)
            cut_mask = b_sc >= cut_threshold
            raw_m_cut = bg_raw_m[cut_mask]
            rsd_m_cut = bg_rsd_m[cut_mask]

            jsd_raw, inv_jsd_raw, status_raw, empty_p, empty_q_raw = compute_base2_jsd(bg_raw_m, raw_m_cut)
            jsd_rsd, inv_jsd_rsd, status_rsd, _, empty_q_rsd = compute_base2_jsd(bg_rsd_m, rsd_m_cut)

            sculpting_cuts[f"top_{pct}pct"] = {
                "cut_threshold": round(float(cut_threshold), 5),
                "surviving_events": int(len(raw_m_cut)),
                "status": status_raw if status_raw == status_rsd else f"raw:{status_raw},rsd:{status_rsd}",
                "raw_jsd_base2": round(jsd_raw, 6) if not np.isnan(jsd_raw) else None,
                "raw_inv_jsd": round(inv_jsd_raw, 1) if (not np.isnan(inv_jsd_raw) and not np.isinf(inv_jsd_raw)) else None,
                "rsd_jsd_base2": round(jsd_rsd, 6) if not np.isnan(jsd_rsd) else None,
                "rsd_inv_jsd": round(inv_jsd_rsd, 1) if (not np.isnan(inv_jsd_rsd) and not np.isinf(inv_jsd_rsd)) else None,
                "empty_bins": {"baseline": empty_p, "raw_cut": empty_q_raw, "rsd_cut": empty_q_rsd},
            }

        results_by_score[score_name] = {
            "auc_inclusive": auc_inclusive,
            "auc_matched": auc_matched,
            "auc_narrow": auc_narrow,
            "rejection": rejection_metrics,
            "pearson_r": round(float(r_val), 4),
            "spearman_rho": round(float(rho_val), 4),
            "disco_dcor2": round(float(disco_val), 4),
            "mass_sculpting": sculpting_cuts,
        }

    return results_by_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate Scaling Study Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    parser.add_argument("--output-json", type=str, default=None, help="Destination JSON path")
    parser.add_argument("--limit-bg", type=int, default=50000, help="Background test limit (default: 50,000)")
    parser.add_argument("--limit-sig", type=int, default=25000, help="Signal test limit (default: 25,000)")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--bg-file", type=str, default=None, help="Explicit path to background HDF5")
    parser.add_argument("--sig-file", type=str, default=None, help="Explicit path to signal HDF5")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory containing splits")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic data for smoke verification (DEBUG ONLY)")
    parser.add_argument("--debug-only", action="store_true", help="Explicitly mark run as non-production debug run")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device(args.device) if args.device else get_device(verbose=False)

    print("\n" + "=" * 80)
    print("  SCALING STUDY FROZEN EVALUATION HARNESS")
    print(f"  Checkpoint: {ckpt_path} | Device: {device}")
    print("=" * 80)

    is_debug = args.synthetic or args.debug_only or (args.limit_bg < 50000) or (args.limit_sig < 25000)

    if args.synthetic:
        print("[WARNING] Running with SYNTHETIC data. Results are marked DEBUG_ONLY and cannot be used for physics conclusions.")
        np.random.seed(42)
        n_bg = min(args.limit_bg, 500)
        n_sig = min(args.limit_sig, 250)
        bg_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_bg)]
        sig_events = [np.random.uniform(-3, 3, 2101).astype(np.float32) for _ in range(n_sig)]
        bg_raw_m = np.random.uniform(2500, 4500, n_bg)
        bg_rsd_m = bg_raw_m.copy()  # Constant scaling fake removed
        sig_raw_m = np.random.normal(3500, 80, n_sig)
        sig_rsd_m = sig_raw_m.copy()
    else:
        # Load real test partition indices
        data_dir = Path(args.data_dir) if args.data_dir else (REPO_ROOT / "data")
        splits_dir = data_dir / "splits"
        assert splits_dir.exists(), f"Splits directory not found: {splits_dir}"

        bg_test_file = splits_dir / "bg_test_50k.npy"
        sig_test_file = splits_dir / "sig_test_25k.npy"
        assert bg_test_file.exists(), f"Background test split missing: {bg_test_file}"
        assert sig_test_file.exists(), f"Signal test split missing: {sig_test_file}"

        bg_test_idx = np.load(bg_test_file)[:args.limit_bg]
        sig_test_idx = np.load(sig_test_file)[:args.limit_sig]

        # Resolve paths
        if args.bg_file:
            bg_h5_path = Path(args.bg_file)
        else:
            bg_h5_path = data_dir / "events_LHCO2020_backgroundMC_Pythia.h5"
            if not bg_h5_path.exists():
                alt_bg = Path("/kaggle/input/lhc-olympics-2020-pythia8-qcd/events_LHCO2020_backgroundMC_Pythia.h5")
                if alt_bg.exists():
                    bg_h5_path = alt_bg

        if args.sig_file:
            sig_h5_path = Path(args.sig_file)
        else:
            sig_h5_path = data_dir / "W_prime_signal.h5"
            if not sig_h5_path.exists():
                alt_sig = Path("/kaggle/input/lhc-olympics-2020-pythia8-qcd/W_prime_signal.h5")
                if alt_sig.exists():
                    sig_h5_path = alt_sig

        # Strict fail-closed verification
        def _verify_file(p: Path, name: str):
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(
                    f"[FAIL-CLOSED] {name} dataset missing at {p}. "
                    "Evaluation requires real physical test data."
                )
            import sys
            st = p.stat()
            if sys.platform == "darwin" and hasattr(st, "st_blocks") and st.st_blocks == 0:
                raise RuntimeError(
                    f"[FAIL-CLOSED] {name} file {p} is an APFS dataless file (st_blocks=0). "
                    "Evaluation cannot proceed without materialized file contents."
                )

        _verify_file(bg_h5_path, "Background")
        _verify_file(sig_h5_path, "Signal")

        print(f"[INFO] Streaming {len(bg_test_idx):,} held-out Background and {len(sig_test_idx):,} Signal test events...")
        import tables

        with tables.open_file(str(bg_h5_path), mode="r") as h5_bg:
            bg_node = h5_bg.get_node("/df/block0_values")
            bg_events_raw = bg_node[bg_test_idx, :]
            bg_events = [row for row in bg_events_raw]

        with tables.open_file(str(sig_h5_path), mode="r") as h5_sig:
            sig_node = h5_sig.get_node("/df/block0_values")
            sig_events_raw = sig_node[sig_test_idx, :]
            sig_events = [row for row in sig_events_raw]

        # Compute dijet masses
        bg_raw_m, bg_rsd_m = [], []
        for ev in bg_events:
            p = extract_particles(ev)
            rm, rsdm = compute_groomed_dijet_mass(p)
            bg_raw_m.append(rm)
            bg_rsd_m.append(rsdm)
        bg_raw_m = np.array(bg_raw_m)
        bg_rsd_m = np.array(bg_rsd_m)

        sig_raw_m, sig_rsd_m = [], []
        for ev in sig_events:
            p = extract_particles(ev)
            rm, rsdm = compute_groomed_dijet_mass(p)
            sig_raw_m.append(rm)
            sig_rsd_m.append(rsdm)
        sig_raw_m = np.array(sig_raw_m)
        sig_rsd_m = np.array(sig_rsd_m)

    metrics = evaluate_checkpoint(
        checkpoint_path=ckpt_path,
        device=device,
        bg_events=bg_events,
        bg_raw_m=bg_raw_m,
        bg_rsd_m=bg_rsd_m,
        sig_events=sig_events,
        sig_raw_m=sig_raw_m,
        sig_rsd_m=sig_rsd_m,
    )

    metrics["metadata"] = {
        "is_debug_only": is_debug,
        "is_synthetic": args.synthetic,
        "bg_events_evaluated": len(bg_events),
        "sig_events_evaluated": len(sig_events),
        "checkpoint_path": str(ckpt_path),
    }

    out_json = Path(args.output_json) if args.output_json else ckpt_path.parent / "eval_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[EVALUATION COMPLETE] Written metrics to {out_json}")
    for sc in ["weighted", "full"]:
        top10 = metrics[sc]['mass_sculpting']['top_10pct']
        inv_jsd_str = f"{top10['raw_inv_jsd']:.1f}" if top10['raw_inv_jsd'] is not None else "N/A"
        print(f"  Score: {sc:<10} | Matched AUC: {metrics[sc]['auc_matched']['mean']:.4f} ± {metrics[sc]['auc_matched']['std']:.4f} | 1/JSD (Top 10%): {inv_jsd_str} | Pearson r: {metrics[sc]['pearson_r']:.4f}")


if __name__ == "__main__":
    main()
