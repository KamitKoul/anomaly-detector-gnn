"""
plot_hybrid_comparison.py
=========================
Generates publication-quality 4-panel diagnostic figure comparing the novel
PointNet-GAT Hybrid architecture against Pure GAT, Pure DeepSets, and Flat MLP:
- Panel A: Anomaly Discrimination vs Raw Mass Invariance Pareto Frontier (Matched AUC vs 1/JSD_raw)
- Panel B: Background Rejection R30 vs SoftDrop RSD Mass Invariance (R30 vs 1/JSD_rsd)
- Panel C: Dijet Invariant Mass Sculpting / Linear Correlation (Pearson r) across architectures
- Panel D: Score Decomposition & Energy-Flow Ablation across the Hybrid and Control models
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'lines.linewidth': 2.0
})

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    hybrid_json = REPO_ROOT / "comparisons/hybrid_pointnet_gat/controlled_ablation_summary.json"
    ctrl_json = REPO_ROOT / "comparisons/controlled_structure/controlled_ablation_summary.json"
    disco_json = REPO_ROOT / "comparisons/score_decorrelation/controlled_ablation_summary.json"

    if not hybrid_json.exists():
        print(f"[ERROR] Missing {hybrid_json}")
        return

    hybrid_data = load_json(hybrid_json)
    ctrl_data = load_json(ctrl_json) if ctrl_json.exists() else {}
    disco_data = load_json(disco_json) if disco_json.exists() else {}

    out_dir = REPO_ROOT / "comparisons/hybrid_pointnet_gat/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hybrid_pointnet_gat_diagnostics.png"

    # Define model cohorts (each with 3 seeds)
    cohorts = [
        {
            "name": "Flat MLP",
            "keys": ["baseline_mlp_s42", "baseline_mlp_s123", "baseline_mlp_s999"],
            "dict": ctrl_data,
            "color": "#7f7f7f",
            "marker": "s",
        },
        {
            "name": "Pure DeepSets",
            "keys": ["deepsets_pointnet_s42", "deepsets_pointnet_s123", "deepsets_pointnet_s999"],
            "dict": ctrl_data,
            "color": "#ff7f0e",
            "marker": "^",
        },
        {
            "name": "Pure Relational GAT",
            "keys": ["gat_relational_s42", "gat_relational_s123", "gat_relational_s999"],
            "dict": ctrl_data,
            "color": "#1f77b4",
            "marker": "o",
        },
        {
            "name": "GAT + DisCo (Weighted)",
            "keys": ["gat_score_disco_weighted_s42", "gat_score_disco_weighted_s123", "gat_score_disco_weighted_s999"],
            "dict": disco_data,
            "color": "#2ca02c",
            "marker": "D",
        },
        {
            "name": "Hybrid Baseline (Ours)",
            "keys": ["hybrid_baseline_s42", "hybrid_baseline_s123", "hybrid_baseline_s999"],
            "dict": hybrid_data,
            "color": "#9467bd",
            "marker": "p",
        },
        {
            "name": "Hybrid + DisCo (Ours)",
            "keys": ["hybrid_disco_weighted_s42", "hybrid_disco_weighted_s123", "hybrid_disco_weighted_s999"],
            "dict": hybrid_data,
            "color": "#d62728",
            "marker": "*",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ----------------------------------------------------
    # Panel A: Pareto Frontier (Matched AUC vs 1/JSD Raw Top 10%)
    # ----------------------------------------------------
    ax = axes[0, 0]
    for c in cohorts:
        runs = [c["dict"][k] for k in c["keys"] if k in c["dict"]]
        if not runs:
            continue
        # Use optimal score type if available, else pt_only
        sc_name = "weighted" if "weighted" in runs[0]["metrics"] else "pt_only"
        aucs = [r["metrics"][sc_name]["auc_matched"] for r in runs]
        jsds = [r["metrics"][sc_name]["inv_jsd_raw_top10"] for r in runs]

        m_auc, s_auc = np.mean(aucs), np.std(aucs, ddof=1)
        m_jsd, s_jsd = np.mean(jsds), np.std(jsds, ddof=1)

        ax.errorbar(
            m_auc,
            m_jsd,
            xerr=s_auc,
            yerr=s_jsd,
            fmt=c["marker"],
            color=c["color"],
            label=f"{c['name']} ({m_auc:.3f}, {m_jsd:.1f})",
            markersize=9 if c["marker"] != "*" else 14,
            capsize=4,
            elinewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("Signal Discrimination: Matched ROC-AUC ([3.3, 3.7] TeV)")
    ax.set_ylabel("Raw Mass Invariance: $1/\\mathrm{JSD}_\\mathrm{raw}$ (Top 10% cut)")
    ax.set_title("(a) Discrimination vs. Mass Sculpting Pareto Frontier")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    # ----------------------------------------------------
    # Panel B: Background Rejection R30 vs SoftDrop RSD 1/JSD
    # ----------------------------------------------------
    ax = axes[0, 1]
    for c in cohorts:
        runs = [c["dict"][k] for k in c["keys"] if k in c["dict"]]
        if not runs:
            continue
        sc_name = "weighted" if "weighted" in runs[0]["metrics"] else "pt_only"
        r30s = [r["metrics"][sc_name]["rejection_30"] for r in runs]
        jsd_rsds = [r["metrics"][sc_name]["inv_jsd_rsd_top10"] for r in runs]

        m_r30, s_r30 = np.mean(r30s), np.std(r30s, ddof=1)
        m_rsd, s_rsd = np.mean(jsd_rsds), np.std(jsd_rsds, ddof=1)

        ax.errorbar(
            m_r30,
            m_rsd,
            xerr=s_r30,
            yerr=s_rsd,
            fmt=c["marker"],
            color=c["color"],
            label=c["name"],
            markersize=9 if c["marker"] != "*" else 14,
            capsize=4,
            elinewidth=1.5,
            alpha=0.9,
        )

    ax.set_xlabel("Background Rejection at 30% Signal Efficiency ($R_{30} = 1 / \\mathrm{FPR}$)")
    ax.set_ylabel("SoftDrop Mass Invariance: $1/\\mathrm{JSD}_\\mathrm{rsd}$ (Top 10% cut)")
    ax.set_title("(b) Operating Point Rejection vs. Substructure Mass Preservation")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    # ----------------------------------------------------
    # Panel C: Linear Mass Correlation (Dijet Pearson r)
    # ----------------------------------------------------
    ax = axes[1, 0]
    names = []
    pearsons_mean = []
    pearsons_std = []
    bar_colors = []

    for c in cohorts:
        runs = [c["dict"][k] for k in c["keys"] if k in c["dict"]]
        if not runs:
            continue
        sc_name = "weighted" if "weighted" in runs[0]["metrics"] else "pt_only"
        r_vals = [r["metrics"][sc_name]["pearson_r"] for r in runs]
        names.append(c["name"].replace(" (Ours)", "").replace(" (Control)", ""))
        pearsons_mean.append(np.mean(r_vals))
        pearsons_std.append(np.std(r_vals, ddof=1))
        bar_colors.append(c["color"])

    y_pos = np.arange(len(names))
    ax.barh(y_pos, pearsons_mean, xerr=pearsons_std, color=bar_colors, alpha=0.8, capsize=4, edgecolor="black", height=0.55)
    ax.axvline(0, color="black", linestyle="-", lw=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Dijet Invariant Mass Linear Correlation: Pearson $r$")
    ax.set_title("(c) Dijet Mass Correlation Suppression ($r \\to 0$ = zero sculpting)")
    ax.grid(True, linestyle="--", alpha=0.6)

    # ----------------------------------------------------
    # Panel D: Score Decomposition in Hybrid Architecture
    # ----------------------------------------------------
    ax = axes[1, 1]
    score_types = ["full", "pt_only", "ang_only", "weighted"]
    score_labels = ["Full 3D", "$p_T$-Only", "Angular $(\\eta, \\phi)$", "Weighted $(0.8 p_T + 0.2 \\Omega)$"]
    x = np.arange(len(score_labels))
    width = 0.35

    # Hybrid Baseline vs Hybrid DisCo
    h_base_runs = [hybrid_data[k] for k in ["hybrid_baseline_s42", "hybrid_baseline_s123", "hybrid_baseline_s999"]]
    h_disco_runs = [hybrid_data[k] for k in ["hybrid_disco_weighted_s42", "hybrid_disco_weighted_s123", "hybrid_disco_weighted_s999"]]

    base_auc_m = [np.mean([r["metrics"][s]["auc_matched"] for r in h_base_runs]) for s in score_types]
    base_auc_s = [np.std([r["metrics"][s]["auc_matched"] for r in h_base_runs], ddof=1) for s in score_types]

    disco_auc_m = [np.mean([r["metrics"][s]["auc_matched"] for r in h_disco_runs]) for s in score_types]
    disco_auc_s = [np.std([r["metrics"][s]["auc_matched"] for r in h_disco_runs], ddof=1) for s in score_types]

    rects1 = ax.bar(x - width/2, base_auc_m, width, yerr=base_auc_s, label="Hybrid Baseline (λ=0)", color="#9467bd", capsize=4, alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, disco_auc_m, width, yerr=disco_auc_s, label="Hybrid + DisCo (λ=1.0)", color="#d62728", capsize=4, alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(score_labels)
    ax.set_ylabel("Matched ROC-AUC ([3.3, 3.7] TeV)")
    ax.set_title("(d) Hybrid Score Decomposition: Local vs Global Energy Flow")
    ax.set_ylim(0.3, 0.8)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.suptitle("PointNet-GAT Hybrid Architecture Performance & Invariance Benchmark (LHC Olympics 2020)", y=0.99)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Diagnostic plot generated: {out_file}")


if __name__ == "__main__":
    main()
