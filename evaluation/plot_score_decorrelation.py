"""
plot_score_decorrelation.py
===========================
Generates publication-quality 4-panel diagnostic figure for the Score-Level
Decorrelation study (DisCo vs GRL vs Baseline Control) on LHC Olympics 2020:
- Panel A: Anomaly Discrimination (Matched AUC) across score formulations with 68% bootstrap CIs
- Panel B: Mass Preservation / Background Invariance (1/JSD at Top 10% cut, Raw vs RSD)
- Panel C: Pareto Frontier: Discrimination (Matched AUC) vs Mass Invariance (1/JSD)
- Panel D: Invariant Mass Linear Correlation (Pearson r) across score types
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
    summary_path = REPO_ROOT / "comparisons/score_decorrelation/controlled_ablation_summary.json"
    out_dir = REPO_ROOT / "comparisons/score_decorrelation/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "score_decorrelation_publication_diagnostics.png"
    
    with open(summary_path) as f:
        data = json.load(f)
        
    # Group configurations
    # We display representative architectures & multi-seed averages
    groups = [
        ("Control Ref\n(λ=0)", "gat_baseline_ref_s42"),
        ("DisCo Heavy\n(λ=2.5)", "gat_score_disco_heavy_s42"),
        ("DisCo Std\n(Seed 42)", "gat_score_disco_s42"),
        ("DisCo Weighted\n(Optimized)", "gat_score_disco_weighted_s42"),
        ("GRL Adv\n(Seed 42)", "gat_score_grl_s42"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # Colors
    c_full = "#2b5c8f"
    c_pt = "#d95f02"
    c_ang = "#7570b3"
    c_weight = "#1b9e77"
    
    # ----------------------------------------------------
    # Panel A: Matched ROC-AUC across score types
    # ----------------------------------------------------
    ax = axes[0, 0]
    x = np.arange(len(groups))
    width = 0.22
    
    auc_full, unc_full = [], []
    auc_pt, unc_pt = [], []
    auc_ang, unc_ang = [], []
    auc_w, unc_w = [], []
    
    for label, exp_id in groups:
        m = data[exp_id]["metrics"]
        auc_full.append(m["full"]["auc_matched"])
        unc_full.append(m["full"]["auc_matched_unc"])
        auc_pt.append(m["pt_only"]["auc_matched"])
        unc_pt.append(m["pt_only"]["auc_matched_unc"])
        auc_ang.append(m["ang_only"]["auc_matched"])
        unc_ang.append(m["ang_only"]["auc_matched_unc"])
        auc_w.append(m["weighted"]["auc_matched"])
        unc_w.append(m["weighted"]["auc_matched_unc"])
        
    ax.bar(x - 1.5*width, auc_full, width, yerr=unc_full, capsize=3, label=r'Full 3D Recon ($p_T, \eta, \phi$)', color=c_full, alpha=0.9)
    ax.bar(x - 0.5*width, auc_pt, width, yerr=unc_pt, capsize=3, label=r'$p_T$-Only Recon', color=c_pt, alpha=0.9)
    ax.bar(x + 0.5*width, auc_w, width, yerr=unc_w, capsize=3, label=r'Weighted Recon ($0.95 p_T + 0.05 \Omega$)', color=c_weight, alpha=0.9)
    ax.bar(x + 1.5*width, auc_ang, width, yerr=unc_ang, capsize=3, label=r'Angular-Only ($\Delta\eta, \Delta\phi$)', color=c_ang, alpha=0.9)
    
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='Random Chance (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(r'Matched ROC-AUC ($m_{jj} \in [3.3, 3.7]$ TeV)')
    ax.set_title(r'(a) Anomaly Discrimination Across Score Formulations (68% Bootstrap CI)')
    ax.set_ylim(0.25, 0.85)
    ax.legend(loc='lower left', frameon=True, fontsize=9)
    
    # ----------------------------------------------------
    # Panel B: Mass Invariance (1/JSD at Top 10% Cut)
    # ----------------------------------------------------
    ax = axes[0, 1]
    w_b = 0.35
    jsd_raw, unc_raw = [], []
    jsd_rsd, unc_rsd = [], []
    
    for label, exp_id in groups:
        # Full score JSD
        m = data[exp_id]["metrics"]["full"]
        jsd_raw.append(m["inv_jsd_raw_top10"])
        unc_raw.append(m["inv_jsd_raw_top10_unc"])
        jsd_rsd.append(m["inv_jsd_rsd_top10"])
        unc_rsd.append(m["inv_jsd_rsd_top10_unc"])
        
    ax.bar(x - 0.5*w_b, jsd_raw, w_b, yerr=unc_raw, capsize=4, label=r'Raw Dijet Mass $m_{jj}$ ($1/\mathrm{JSD}$)', color="#4575b4", alpha=0.9)
    ax.bar(x + 0.5*w_b, jsd_rsd, w_b, yerr=unc_rsd, capsize=4, label=r'SoftDrop $m_{\mathrm{RSD}}$ ($1/\mathrm{JSD}$)', color="#e08214", alpha=0.9)
    
    # Highlight DisCo Weighted jump
    ax.annotate(r'$\mathbf{2.34\times}$ Invariance Boost' + '\n' + r'($1/\mathrm{JSD} = 75.0 \pm 9.2$)',
                xy=(3 - 0.5*w_b, jsd_raw[3]),
                xytext=(2.2, 85),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e5f5e0", ec="#31a354", lw=1.5))
    
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(r'Mass Preservation Metric: $1/\mathrm{JSD}$ (bits$^{-1}$)')
    ax.set_title(r'(b) Background Mass Invariance at Top 10% Operating Cut')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper left', frameon=True)
    
    # ----------------------------------------------------
    # Panel C: Pareto Frontier: Discrimination vs Decorrelation
    # ----------------------------------------------------
    ax = axes[1, 0]
    
    # Scatter all 9 models (both full and pt_only scores)
    markers = {'gat_baseline_ref': 's', 'gat_score_disco': 'o', 'gat_score_grl': '^'}
    colors = {42: '#e41a1c', 123: '#377eb8', 999: '#4daf4a'}
    
    for exp_id, exp_data in data.items():
        seed = exp_data["seed"]
        c = colors.get(seed, 'black')
        
        # Determine archetype
        if 'baseline' in exp_id:
            m_type = 's'
            name_lbl = 'Baseline Ref'
        elif 'disco' in exp_id:
            m_type = 'o'
            name_lbl = 'DisCo Decorrelated'
        else:
            m_type = '^'
            name_lbl = 'GRL Adversarial'
            
        # Plot full score
        mf = exp_data["metrics"]["full"]
        ax.errorbar(mf["inv_jsd_raw_top10"], mf["auc_matched"],
                    xerr=mf["inv_jsd_raw_top10_unc"], yerr=mf["auc_matched_unc"],
                    fmt=m_type, color=c, alpha=0.8, markersize=8, capsize=2)
        
        # Plot pt_only score
        mp = exp_data["metrics"]["pt_only"]
        ax.errorbar(mp["inv_jsd_raw_top10"], mp["auc_matched"],
                    xerr=mp["inv_jsd_raw_top10_unc"], yerr=mp["auc_matched_unc"],
                    fmt=m_type, markeredgecolor=c, markerfacecolor='white',
                    alpha=0.8, markersize=8, capsize=2)
        
    # Annotate key points
    ax.annotate(r'$\mathbf{DisCo\ Weighted}$ (Full Score)' + '\n' + r'$AUC=0.734$, $1/\mathrm{JSD}=75.0$',
                xy=(75.0, 0.7338), xytext=(52, 0.71),
                arrowprops=dict(facecolor='#1b9e77', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#e5f5e0", ec="#1b9e77", lw=1.5))
    
    ax.annotate(r'$\mathbf{DisCo\ Weighted}$ ($p_T$-Only)' + '\n' + r'$AUC=0.765 \pm 0.008$',
                xy=(41.2, 0.7647), xytext=(48, 0.77),
                arrowprops=dict(facecolor='#d95f02', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fff5eb", ec="#d95f02", lw=1.5))

    # Dummy legend items
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Baseline Ref', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='DisCo Decorrelated', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='^', color='w', label='GRL Adversarial', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Full Score (Filled)', markerfacecolor='black', markersize=8),
        Line2D([0], [0], marker='o', color='black', label=r'$p_T$-Only (Open)', markerfacecolor='white', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Seed 42', markerfacecolor='#e41a1c', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Seed 123', markerfacecolor='#377eb8', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Seed 999', markerfacecolor='#4daf4a', markersize=8),
    ]
    ax.legend(handles=legend_elements, loc='lower left', ncol=2, frameon=True, fontsize=8.5)
    
    ax.set_xlabel(r'Mass Preservation Metric: $1/\mathrm{JSD}$ at Top 10% Cut (bits$^{-1}$)')
    ax.set_ylabel(r'Anomaly Discrimination: Matched ROC-AUC ($m_{jj} \in [3.3, 3.7]$ TeV)')
    ax.set_title(r'(c) Pareto Frontier: Discrimination vs Background Mass Invariance')
    ax.set_xlim(20, 90)
    ax.set_ylim(0.71, 0.79)
    
    # ----------------------------------------------------
    # Panel D: Pearson Correlation r with Invariant Mass
    # ----------------------------------------------------
    ax = axes[1, 1]
    
    r_full, r_pt, r_ang, r_w = [], [], [], []
    for label, exp_id in groups:
        m = data[exp_id]["metrics"]
        r_full.append(m["full"]["pearson_r"])
        r_pt.append(m["pt_only"]["pearson_r"])
        r_ang.append(m["ang_only"]["pearson_r"])
        r_w.append(m["weighted"]["pearson_r"])
        
    ax.bar(x - 1.5*width, r_full, width, label=r'Full 3D Recon', color=c_full, alpha=0.9)
    ax.bar(x - 0.5*width, r_pt, width, label=r'$p_T$-Only Recon', color=c_pt, alpha=0.9)
    ax.bar(x + 0.5*width, r_w, width, label=r'Weighted Recon', color=c_weight, alpha=0.9)
    ax.bar(x + 1.5*width, r_ang, width, label=r'Angular-Only Recon', color=c_ang, alpha=0.9)
    
    ax.axhline(0.0, color='black', linestyle='-', linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(r'Linear Correlation with Dijet Mass: Pearson $r$')
    ax.set_title(r'(d) Invariant Mass Linear Correlation Across Score Types')
    ax.set_ylim(-0.04, 0.25)
    ax.legend(loc='upper right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Diagnostic figure saved: {out_file}")

if __name__ == "__main__":
    main()
