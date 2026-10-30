"""
plot_disco_sweep_diagnostics.py
===============================
Generates publication-quality 4-panel diagnostic figure for the PointNet-GAT
Hybrid DisCo Loss Optimization Sweep (H0 to H4, lambda in [0.0, 1.0]):
- Panel A: DisCo Response Curves: Discrimination (Matched AUC) & Invariance (1/JSD) vs Lambda
- Panel B: Global Pareto Frontier: Discrimination vs Mass Invariance across All Repository Models
- Panel C: Statistical Independence Metrics: Pearson r and Held-Out Distance Correlation vs Lambda
- Panel D: Multi-Score Formulation Profile across the DisCo Sweep
"""

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


def main():
    summary_path = REPO_ROOT / "comparisons/hybrid_pointnet_gat/controlled_ablation_summary.json"
    ctrl_path = REPO_ROOT / "comparisons/controlled_structure/controlled_ablation_summary.json"
    disco_path = REPO_ROOT / "comparisons/score_decorrelation/controlled_ablation_summary.json"

    with open(summary_path) as f:
        hybrid_data = json.load(f)
    ctrl_data = json.load(open(ctrl_path)) if ctrl_path.exists() else {}
    disco_data = json.load(open(disco_path)) if disco_path.exists() else {}

    out_dir = REPO_ROOT / "comparisons/hybrid_pointnet_gat/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "disco_sweep_publication_diagnostics.png"

    sweeps = [
        (0.00, "hybrid_h0_lambda000", "H0 (λ=0.00)", "#9467bd"),
        (0.10, "hybrid_h1_lambda010", "H1 (λ=0.10)", "#1f77b4"),
        (0.25, "hybrid_h2_lambda025", "H2 (λ=0.25)", "#2ca02c"),
        (0.50, "hybrid_h3_lambda050", "H3 (λ=0.50)", "#ff7f0e"),
        (1.00, "hybrid_h4_lambda100", "H4 (λ=1.00)", "#d62728"),
    ]
    seeds = [42, 123, 999]

    lambdas = [s[0] for s in sweeps]
    auc_means, auc_stds = [], []
    jsd_means, jsd_stds = [], []
    r_means, r_stds = [], []
    dcor_means, dcor_stds = [], []

    for lam, prefix, label, color in sweeps:
        runs = [hybrid_data[f"{prefix}_s{s}"]["metrics"]["weighted"] for s in seeds]
        aucs = [r["auc_matched"] for r in runs]
        jsds = [r["inv_jsd_raw_top10"] for r in runs]
        rs = [r["pearson_r"] for r in runs]
        dcors = [r.get("distance_correlation", 0.0) for r in runs]

        auc_means.append(np.mean(aucs))
        auc_stds.append(np.std(aucs, ddof=1))
        jsd_means.append(np.mean(jsds))
        jsd_stds.append(np.std(jsds, ddof=1))
        r_means.append(np.mean(rs))
        r_stds.append(np.std(rs, ddof=1))
        dcor_means.append(np.mean(dcors))
        dcor_stds.append(np.std(dcors, ddof=1))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ----------------------------------------------------
    # Panel A: DisCo Response Curves (Dual Axis)
    # ----------------------------------------------------
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()

    l1 = ax1.errorbar(lambdas, auc_means, yerr=auc_stds, fmt="-o", color="#2b5c8f", lw=2.2, capsize=4, label="Matched ROC-AUC")
    l2 = ax1_twin.errorbar(lambdas, jsd_means, yerr=jsd_stds, fmt="--s", color="#d95f02", lw=2.2, capsize=4, label="Raw Mass 1/JSD (Top 10%)")

    ax1.set_xlabel(r"DisCo Penalty Weight: $\lambda_\mathrm{disco}$")
    ax1.set_ylabel("Signal Discrimination: Matched ROC-AUC", color="#2b5c8f")
    ax1_twin.set_ylabel(r"Mass Invariance: $1/\mathrm{JSD}_\mathrm{raw}$ (Top 10%)", color="#d95f02")
    ax1.set_title(r"(a) DisCo Regularization Response Curves")
    ax1.set_xticks(lambdas)
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    # ----------------------------------------------------
    # Panel B: Global Pareto Frontier
    # ----------------------------------------------------
    ax2 = axes[0, 1]

    # External controls
    benchmarks = [
        ("Flat MLP", ["baseline_mlp_s42", "baseline_mlp_s123", "baseline_mlp_s999"], ctrl_data, "#7f7f7f", "s"),
        ("Pure DeepSets", ["deepsets_pointnet_s42", "deepsets_pointnet_s123", "deepsets_pointnet_s999"], ctrl_data, "#8c564b", "^"),
        ("Pure GAT", ["gat_relational_s42", "gat_relational_s123", "gat_relational_s999"], ctrl_data, "#17becf", "o"),
        ("GAT + DisCo (Weighted)", ["gat_score_disco_weighted_s42", "gat_score_disco_weighted_s123", "gat_score_disco_weighted_s999"], disco_data, "#2ca02c", "D"),
    ]

    for bname, bkeys, bdict, bcol, bmarker in benchmarks:
        if all(k in bdict for k in bkeys):
            runs = [bdict[k] for k in bkeys]
            sc_key = "weighted" if "weighted" in runs[0]["metrics"] else "pt_only"
            b_aucs = [r["metrics"][sc_key]["auc_matched"] for r in runs]
            b_jsds = [r["metrics"][sc_key]["inv_jsd_raw_top10"] for r in runs]
            m_a, s_a = np.mean(b_aucs), np.std(b_aucs, ddof=1)
            m_j, s_j = np.mean(b_jsds), np.std(b_jsds, ddof=1)
            ax2.errorbar(m_a, m_j, xerr=s_a, yerr=s_j, fmt=bmarker, color=bcol, label=f"{bname} ({m_a:.3f}, {m_j:.1f})", markersize=9, capsize=4, elinewidth=1.5, alpha=0.85)

    # Sweep models
    for i, (lam, prefix, label, color) in enumerate(sweeps):
        ax2.errorbar(auc_means[i], jsd_means[i], xerr=auc_stds[i], yerr=jsd_stds[i], fmt="*", color=color, label=f"{label} ({auc_means[i]:.3f}, {jsd_means[i]:.1f})", markersize=14, capsize=4, elinewidth=1.5)

    ax2.set_xlabel("Signal Discrimination: Matched ROC-AUC ([3.3, 3.7] TeV)")
    ax2.set_ylabel(r"Raw Mass Invariance: $1/\mathrm{JSD}_\mathrm{raw}$ (Top 10%)")
    ax2.set_title("(b) Global Discrimination vs. Mass Sculpting Pareto Frontier")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    # ----------------------------------------------------
    # Panel C: Statistical Independence Metrics vs Lambda
    # ----------------------------------------------------
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()

    l3 = ax3.errorbar(lambdas, r_means, yerr=r_stds, fmt="-o", color="#7570b3", lw=2.2, capsize=4, label="Dijet Mass Pearson $r$")
    l4 = ax3_twin.errorbar(lambdas, dcor_means, yerr=dcor_stds, fmt="--^", color="#1b9e77", lw=2.2, capsize=4, label=r"Held-Out Distance Correlation $d\mathrm{Cor}$")

    ax3.axhline(0.0, color="gray", linestyle=":", lw=1.2)
    ax3.set_xlabel(r"DisCo Penalty Weight: $\lambda_\mathrm{disco}$")
    ax3.set_ylabel("Linear Correlation: Pearson $r$", color="#7570b3")
    ax3_twin.set_ylabel(r"Non-Linear Dependence: Held-Out $d\mathrm{Cor}$ (10k events)", color="#1b9e77")
    ax3.set_title(r"(c) Linear ($r$) vs. Non-Linear ($d\mathrm{Cor}$) Decorrelation Dynamics")
    ax3.set_xticks(lambdas)
    ax3.grid(True, linestyle="--", alpha=0.5)

    lines2 = [l3, l4]
    labels2 = [l.get_label() for l in lines2]
    ax3.legend(lines2, labels2, loc="upper right", frameon=True)

    # ----------------------------------------------------
    # Panel D: Multi-Score Formulation Profile across Sweep
    # ----------------------------------------------------
    ax4 = axes[1, 1]
    score_types = ["full", "pt_only", "ang_only", "weighted"]
    score_names = ["Full 3D", "$p_T$-Only", "Angular $(\\eta, \\phi)$", "Weighted $(0.8 p_T + 0.2 \\Omega)$"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    x = np.arange(len(sweeps))
    width = 0.20

    for idx, sc in enumerate(score_types):
        sc_means = []
        sc_stds = []
        for lam, prefix, _, _ in sweeps:
            runs = [hybrid_data[f"{prefix}_s{s}"]["metrics"][sc] for s in seeds]
            sc_means.append(np.mean([r["auc_matched"] for r in runs]))
            sc_stds.append(np.std([r["auc_matched"] for r in runs], ddof=1))

        ax4.bar(x + (idx - 1.5) * width, sc_means, width, yerr=sc_stds, label=score_names[idx], color=colors[idx], capsize=3, alpha=0.85, edgecolor="black")

    ax4.set_xticks(x)
    ax4.set_xticklabels([s[2] for s in sweeps], rotation=15)
    ax4.set_ylabel("Matched ROC-AUC ([3.3, 3.7] TeV)")
    ax4.set_title("(d) Anomaly Score Decomposition across DisCo Sweep")
    ax4.set_ylim(0.3, 0.8)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("PointNet-GAT Hybrid: DisCo Regularization Sweep & Operating Point Optimization", y=0.99)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Diagnostic plot generated: {out_file}")


if __name__ == "__main__":
    main()
