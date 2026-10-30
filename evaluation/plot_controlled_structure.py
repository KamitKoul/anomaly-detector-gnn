"""
plot_controlled_structure.py
============================
Generates publication-quality 4-panel diagnostic figure for the Controlled
Graph-Structure and Anomaly-Score Ablation study on LHC Olympics 2020:
- Panel A: Matched ROC-AUC across architectures and score formulations (Full, pT-only, Angular, Weighted)
- Panel B: Background Rejection (R30) at 30% Signal Efficiency Collider Operating Point
- Panel C: Pareto Frontier: Anomaly Discrimination (Matched AUC) vs Mass Preservation (1/JSD)
- Panel D: Stability across Mass Windows (Inclusive vs Matched [3.3, 3.7] TeV vs Micro-Fine [3.45, 3.55] TeV)
"""

import os
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
    summary_path = REPO_ROOT / "comparisons/controlled_structure/controlled_ablation_summary.json"
    out_dir = REPO_ROOT / "comparisons/controlled_structure/plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "controlled_structure_publication_diagnostics.png"
    
    with open(summary_path) as f:
        data = json.load(f)
        
    models = ['baseline_mlp', 'nodewise_ae', 'gat_relational', 'deepsets_pointnet']
    model_labels = {
        'baseline_mlp': 'Model A:\nFlat MLP',
        'nodewise_ae': 'Model C:\nNode-Wise AE\n(No MP)',
        'gat_relational': 'Model B:\nRelational GAT\n(ΔR < 0.4)',
        'deepsets_pointnet': 'Model D:\nDeepSets\nPointNet'
    }
    
    score_labels = {
        'full': 'Full (pT, η, φ)',
        'pt_only': 'pT-Only',
        'ang_only': 'Angular-Only (η, φ)',
        'weighted': 'Weighted (0.8 pT + 0.2 Ang)'
    }
    
    score_colors = {
        'full': '#1f77b4',       # Blue
        'pt_only': '#2ca02c',    # Green
        'ang_only': '#d62728',   # Red
        'weighted': '#ff7f0e'    # Orange
    }
    
    model_colors = {
        'baseline_mlp': '#7f7f7f',      # Gray
        'nodewise_ae': '#e377c2',       # Pink
        'gat_relational': '#1f77b4',    # Blue
        'deepsets_pointnet': '#9467bd'  # Purple
    }

    # Aggregate stats per model and score
    stats = {m: {s: {'auc_matched': [], 'auc_inclusive': [], 'auc_micro_fine': [],
                     'rejection_30': [], 'rejection_50': [], 'inv_jsd_raw_top10': [], 'pearson_r': []}
                 for s in ['full', 'pt_only', 'ang_only', 'weighted']}
             for m in models}

    for exp_name, val in data.items():
        mkey = None
        for k in models:
            if exp_name.startswith(k):
                mkey = k
                break
        if not mkey:
            continue
        for s in ['full', 'pt_only', 'ang_only', 'weighted']:
            sm = val['metrics'][s]
            for metric in ['auc_matched', 'auc_inclusive', 'auc_micro_fine',
                           'rejection_30', 'rejection_50', 'inv_jsd_raw_top10', 'pearson_r']:
                stats[mkey][s][metric].append(sm[metric])

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ----------------------------------------------------
    # Panel A: Matched ROC-AUC across Architectures & Scores
    # ----------------------------------------------------
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.2
    
    for i, s in enumerate(['full', 'pt_only', 'ang_only', 'weighted']):
        means = [np.mean(stats[m][s]['auc_matched']) for m in models]
        stds = [np.std(stats[m][s]['auc_matched']) for m in models]
        rects = ax.bar(x + (i - 1.5) * width, means, width, yerr=stds,
                       label=score_labels[s], color=score_colors[s], alpha=0.9, capsize=4)
        
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Random Guess (0.50)')
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in models])
    ax.set_ylabel("Matched ROC-AUC [3.3 - 3.7 TeV]")
    ax.set_ylim(0.35, 0.85)
    ax.set_title("(a) Anomaly Discrimination by Graph Structure & Score Formulation", fontweight='bold')
    ax.legend(frameon=True, loc='upper left', ncol=2)
    ax.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # Panel B: Background Rejection R30 at 30% Signal Efficiency
    # ----------------------------------------------------
    ax = axes[0, 1]
    for i, s in enumerate(['full', 'pt_only', 'weighted']):
        means = [np.mean(stats[m][s]['rejection_30']) for m in models]
        stds = [np.std(stats[m][s]['rejection_30']) for m in models]
        ax.bar(x + (i - 1.0) * (width * 1.3), means, width * 1.3, yerr=stds,
               label=score_labels[s], color=score_colors[s], alpha=0.9, capsize=4)
        
    ax.set_xticks(x)
    ax.set_xticklabels([model_labels[m] for m in models])
    ax.set_ylabel("Background Rejection $R_{30} = 1 / \\epsilon_{B}$ (@ $\\epsilon_S = 30\\%$)")
    ax.set_ylim(0, 26)
    ax.set_title("(b) Collider Operating Point: Background Rejection at $\\epsilon_S = 30\\%$", fontweight='bold')
    ax.legend(frameon=True, loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------------------------
    # Panel C: Pareto Frontier: AUC vs Mass Preservation (1/JSD)
    # ----------------------------------------------------
    ax = axes[1, 0]
    markers = {'full': 'o', 'pt_only': 's', 'ang_only': '^', 'weighted': 'D'}
    
    for m in models:
        for s in ['full', 'pt_only', 'weighted']:
            auc_m = np.mean(stats[m][s]['auc_matched'])
            auc_s = np.std(stats[m][s]['auc_matched'])
            jsd_m = np.mean(stats[m][s]['inv_jsd_raw_top10'])
            jsd_s = np.std(stats[m][s]['inv_jsd_raw_top10'])
            
            label_text = f"{m.split('_')[0].upper()} ({score_labels[s].split()[0]})"
            ax.errorbar(jsd_m, auc_m, xerr=jsd_s, yerr=auc_s, fmt=markers[s],
                        color=model_colors[m], markersize=9, capsize=4, elinewidth=1.5,
                        label=f"{model_labels[m].split()[0]} {model_labels[m].split()[1]} [{s}]")
            
            # Annotate points
            offset = (5, 5) if s != 'pt_only' else (5, -12)
            ax.annotate(f"{m.split('_')[0]} ({s[:4]})", (jsd_m, auc_m),
                        textcoords="offset points", xytext=offset, fontsize=8)

    ax.set_xlabel("Background Mass Preservation $1/\\mathrm{JSD}_{\\mathrm{raw}}$ (Top 10% Cut) [Higher = Less Sculpting]")
    ax.set_ylabel("Matched ROC-AUC [3.3 - 3.7 TeV] [Higher = Better Discovery]")
    ax.set_title("(c) Pareto Frontier: Discrimination vs Mass Preservation", fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Custom legend for Pareto
    model_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=model_colors[m], markersize=9, label=model_labels[m].replace('\n', ' '))
                     for m in models]
    score_marker_handles = [plt.Line2D([0], [0], marker=markers[s], color='gray', linestyle='none', markersize=8, label=score_labels[s])
                            for s in ['full', 'pt_only', 'weighted']]
    ax.legend(handles=model_handles + score_marker_handles, frameon=True, loc='lower right', fontsize=8.5)

    # ----------------------------------------------------
    # Panel D: Stability Across Mass Windows
    # ----------------------------------------------------
    ax = axes[1, 1]
    window_names = ['Inclusive\n[All Masses]', 'Matched Window\n[3.3 - 3.7 TeV]', 'Micro-Fine\n[3.45 - 3.55 TeV]']
    
    for m in models:
        # Use full score for window comparison
        inc_m = np.mean(stats[m]['full']['auc_inclusive'])
        mat_m = np.mean(stats[m]['full']['auc_matched'])
        ufine_m = np.mean(stats[m]['full']['auc_micro_fine'])
        
        inc_s = np.std(stats[m]['full']['auc_inclusive'])
        mat_s = np.std(stats[m]['full']['auc_matched'])
        ufine_s = np.std(stats[m]['full']['auc_micro_fine'])
        
        ax.errorbar([0, 1, 2], [inc_m, mat_m, ufine_m],
                    yerr=[inc_s, mat_s, ufine_s],
                    marker='o', markersize=8, capsize=4, linewidth=2.2,
                    label=model_labels[m].replace('\n', ' '),
                    color=model_colors[m])

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Random Guess')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(window_names)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.35, 0.85)
    ax.set_title("(d) Anomaly Score Robustness across Narrowing Mass Slices (Full Score)", fontweight='bold')
    ax.legend(frameon=True, loc='lower left')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Controlled Graph-Structure & Anomaly-Score Investigation (LHC Olympics 2020 W' → XY)",
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved publication diagnostic plot to {out_file}")

if __name__ == "__main__":
    main()
