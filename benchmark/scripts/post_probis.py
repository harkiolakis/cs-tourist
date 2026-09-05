#!/usr/bin/env python3
"""Post-ProBiS processing: generate updated figures, re-run functional prediction,
and prepare ProBiS report section.

Run after probis_eval.csv is available.
"""

import csv
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths
CS_TOURIST_FILE = '/mnt/results/benchmark/data/cs_tourist_eval.csv'
PROBIS_FILE = '/mnt/results/benchmark/data/probis_eval.csv'
FUNC_PRED_FILE = '/mnt/results/benchmark/data/functional_prediction_results.csv'
FIGURE_DIR = '/mnt/results/benchmark/figures'
REPORT_SECTIONS = '/mnt/results/benchmark/report_v10_sections.md'

# Matplotlib settings
matplotlib.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
matplotlib.rcParams['svg.fonttype'] = 'none'

# Color palette
COLORS = {
    'cs_modeC': '#E9ED4C',
    'cs_modeA_r6': '#75A025',
    'cs_modeA_r8': '#FF9400',
    'cs_modeA_r10': '#FD9BED',
    'cs_modeB': '#0279EE',
    'tourist_full': '#888888',
    'tmalign': '#444444',
    'probis': '#FF4444',
}

METHOD_LABELS = {
    'cs_modeC': 'CS Mode C',
    'cs_modeA_r6': 'CS Mode A r6',
    'cs_modeA_r8': 'CS Mode A r8',
    'cs_modeA_r10': 'CS Mode A r10',
    'cs_modeB': 'CS Mode B',
    'tourist_full': 'TOURIST full',
    'tmalign': 'TM-align',
    'probis': 'ProBiS',
}

METHOD_ORDER = ['cs_modeC', 'cs_modeA_r6', 'cs_modeA_r8', 'cs_modeA_r10',
                'cs_modeB', 'probis', 'tmalign', 'tourist_full']


def load_results():
    """Load CS-TOURIST and ProBiS results into a single DataFrame."""
    # Load CS-TOURIST results
    cs_df = pd.read_csv(CS_TOURIST_FILE)
    print(f"Loaded CS-TOURIST: {len(cs_df)} rows, methods: {cs_df['method'].unique()}")

    # Load ProBiS results
    probis_rows = []
    if os.path.exists(PROBIS_FILE):
        probis_df = pd.read_csv(PROBIS_FILE)
        print(f"Loaded ProBiS: {len(probis_df)} rows")
        # Ensure same columns
        for col in cs_df.columns:
            if col not in probis_df.columns:
                probis_df[col] = 0
        probis_rows = probis_df[cs_df.columns].values.tolist()
        probis_combined = pd.concat([cs_df, probis_df[cs_df.columns]], ignore_index=True)
    else:
        print(f"WARNING: ProBiS file not found at {PROBIS_FILE}")
        probis_combined = cs_df

    return probis_combined


def compute_summary(df):
    """Compute concordance summary by method and CATH relationship."""
    valid = df[df['error'].isna() | (df['error'].fillna('') == '')]

    summary = []
    for method in METHOD_ORDER:
        if method not in valid['method'].unique():
            continue
        mdf = valid[valid['method'] == method]
        for rel in ['overall', 'same_h', 'same_t_diff_h', 'diff_t']:
            if rel == 'overall':
                rdf = mdf
            else:
                rdf = mdf[mdf['cath_relation'] == rel]
            if len(rdf) == 0:
                continue
            summary.append({
                'method': method,
                'cath_relation': rel,
                'n': len(rdf),
                'mean_concordance': rdf['concordance'].mean(),
                'mean_recall': rdf['recall'].mean(),
                'mean_precision': rdf['precision'].mean(),
                'mean_n_aligned': rdf['n_aligned'].mean(),
            })

    return pd.DataFrame(summary)


def plot_concordance_with_probis(df, summary):
    """Generate updated concordance bar chart including ProBiS."""
    valid = df[df['error'].isna() | (df['error'].fillna('') == '')]
    methods_present = [m for m in METHOD_ORDER if m in valid['method'].unique()]
    rels = ['same_h', 'same_t_diff_h', 'diff_t']
    rel_labels = ['Same HSF', 'Same Topo, Diff HSF', 'Diff Topo']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(rels))
    width = 0.8 / len(methods_present)

    for i, method in enumerate(methods_present):
        vals = []
        for rel in rels:
            mdf = valid[(valid['method'] == method) & (valid['cath_relation'] == rel)]
            vals.append(mdf['concordance'].mean() if len(mdf) > 0 else 0)
        offset = (i - len(methods_present) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS[method],
                      color=COLORS.get(method, '#CCCCCC'), edgecolor='black', linewidth=0.5)
        # Add value labels on top
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(rel_labels, fontsize=11)
    ax.set_ylabel('Mean Concordance', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    png_path = os.path.join(FIGURE_DIR, 'cs_tourist_concordance_v10.png')
    svg_path = os.path.join(FIGURE_DIR, 'cs_tourist_concordance_v10.svg')
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved concordance figure: {png_path}")


def plot_precision_recall_with_probis(df):
    """Generate updated precision-recall scatter including ProBiS."""
    valid = df[df['error'].isna() | (df['error'].fillna('') == '')]
    methods_present = [m for m in METHOD_ORDER if m in valid['method'].unique()]

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in methods_present:
        mdf = valid[valid['method'] == method]
        if len(mdf) == 0:
            continue
        mean_recall = mdf['recall'].mean()
        mean_prec = mdf['precision'].mean()
        ax.scatter(mean_recall, mean_prec, s=120, color=COLORS.get(method, '#CCCCCC'),
                   edgecolor='black', linewidth=1, zorder=5, label=METHOD_LABELS[method])
        ax.annotate(METHOD_LABELS[method], (mean_recall, mean_prec),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlabel('Mean Recall', fontsize=12)
    ax.set_ylabel('Mean Precision', fontsize=12)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(-0.05, 1.0)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    png_path = os.path.join(FIGURE_DIR, 'cs_tourist_precision_recall_v10.png')
    svg_path = os.path.join(FIGURE_DIR, 'cs_tourist_precision_recall_v10.svg')
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved precision-recall figure: {png_path}")


def plot_func_pred_with_probis():
    """Generate updated functional prediction AUC figure including ProBiS."""
    if not os.path.exists(FUNC_PRED_FILE):
        print("Functional prediction results not found, skipping figure")
        return

    df = pd.read_csv(FUNC_PRED_FILE)
    # Filter to AUC rows (not top-K/MAP)
    auc_df = df[~df['label_type'].str.contains('top|MAP', na=False)]

    label_types = ['EC_class', 'EC_subclass', 'CATH_H', 'CATH_T']
    label_labels = ['EC Class', 'EC Subclass', 'CATH H', 'CATH T']
    methods_present = [m for m in METHOD_ORDER if m in auc_df['method'].unique()]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(label_types))
    width = 0.8 / len(methods_present)

    for i, method in enumerate(methods_present):
        vals = []
        for lt in label_types:
            mdf = auc_df[(auc_df['method'] == method) & (auc_df['label_type'] == lt)]
            vals.append(mdf['auc'].values[0] if len(mdf) > 0 else 0)
        offset = (i - len(methods_present) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=METHOD_LABELS[method],
                      color=COLORS.get(method, '#CCCCCC'), edgecolor='black', linewidth=0.5)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(label_labels, fontsize=11)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    png_path = os.path.join(FIGURE_DIR, 'functional_prediction_auc_v10.png')
    svg_path = os.path.join(FIGURE_DIR, 'functional_prediction_auc_v10.svg')
    fig.savefig(png_path, dpi=150)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved functional prediction figure: {png_path}")


def generate_probis_report_section(df, summary):
    """Generate the ProBiS comparison report section."""
    valid = df[df['error'].isna() | (df['error'].fillna('') == '')]
    probis_valid = valid[valid['method'] == 'probis']

    if len(probis_valid) == 0:
        return "### V10.6 ProBiS Comparison\n\nProBiS evaluation did not produce valid results.\n"

    # Overall stats
    n = len(probis_valid)
    mean_conc = probis_valid['concordance'].mean()
    mean_recall = probis_valid['recall'].mean()
    mean_prec = probis_valid['precision'].mean()
    mean_aln = probis_valid['n_aligned'].mean()

    # By CATH relationship
    by_rel = {}
    for rel in ['same_h', 'same_t_diff_h', 'diff_t']:
        rdf = probis_valid[probis_valid['cath_relation'] == rel]
        if len(rdf) > 0:
            by_rel[rel] = {
                'n': len(rdf),
                'conc': rdf['concordance'].mean(),
                'recall': rdf['recall'].mean(),
                'prec': rdf['precision'].mean(),
            }

    # Error stats
    all_probis = df[df['method'] == 'probis']
    errors = all_probis[all_probis['error'].notna() & (all_probis['error'].fillna('') != '')]
    err_types = errors['error'].value_counts().to_dict() if len(errors) > 0 else {}

    section = f"""### V10.6 ProBiS Comparison

ProBiS (Protein Binding Sites) is a dedicated local structural alignment tool designed specifically for binding site comparison. It finds local structural similarities between protein surfaces without requiring global fold similarity, making it the most directly comparable tool to CS-TOURIST Mode C.

#### Method

ProBiS v2.4.7 (static Linux binary) was run on all CSA pairs using the `-compare` mode with default parameters. The `.nosql` output was parsed to extract residue correspondences, and the best local alignment (by number of correspondences) was selected for each pair. Catalytic site concordance was computed using the same formula as for all other methods.

#### Results

| Metric | ProBiS | CS Mode C | TM-align | TOURIST full |
|--------|--------|-----------|----------|-------------|
| N valid pairs | {n} | 924 | 924 | 924 |
| Mean concordance | {mean_conc:.4f} | 0.8571 | 0.0355 | 0.0340 |
| Mean recall | {mean_recall:.4f} | 0.6185 | 0.0442 | 0.0450 |
| Mean precision | {mean_prec:.4f} | 0.8571 | 0.0597 | 0.0592 |
| Mean n_aligned | {mean_aln:.1f} | 2.9 | 165.4 | 264.3 |

#### Results by CATH Relationship

| Method | same_h | same_t_diff_h | diff_t |
|--------|--------|---------------|--------|
| ProBiS | {by_rel.get('same_h', {}).get('conc', 'N/A'):.4f} | {by_rel.get('same_t_diff_h', {}).get('conc', 'N/A'):.4f} | {by_rel.get('diff_t', {}).get('conc', 'N/A'):.4f} |
| CS Mode C | 0.8653 | 0.8674 | 0.8509 |
| TM-align | 0.1034 | 0.0327 | 0.0126 |
| TOURIST full | 0.1035 | 0.0204 | 0.0140 |

#### Error Analysis
"""

    if err_types:
        section += f"\nProBiS failed on {len(errors)} pairs ({len(errors)/len(all_probis)*100:.1f}%):\n"
        for err, count in sorted(err_types.items(), key=lambda x: -x[1]):
            section += f"- {err}: {count}\n"
    else:
        section += "\nNo errors encountered.\n"

    section += f"""
#### Key Findings

1. **ProBiS concordance: {mean_conc:.4f}** — [comparison to CS Mode C and other methods]

2. **ProBiS aligns {mean_aln:.1f} residues on average** — significantly more than CS Mode C (2.9) but fewer than TM-align (165.4). ProBiS finds local structural similarities across larger surface patches.

3. **Cross-fold performance**: ProBiS achieves concordance of {by_rel.get('diff_t', {}).get('conc', 0):.4f} on diff_t pairs, [comparison to CS Mode C's 0.8509]

4. **ProBiS is a dedicated binding site alignment tool**, making it the most informative comparison for assessing CS-TOURIST's competitive position. [interpretation of results]
"""

    return section


def main():
    print("=" * 80)
    print("POST-PROBIS PROCESSING")
    print("=" * 80)

    # Load combined results
    df = load_results()

    # Compute summary
    summary = compute_summary(df)
    print("\nSummary:")
    print(summary.to_string(index=False))

    # Generate figures
    print("\nGenerating figures...")
    plot_concordance_with_probis(df, summary)
    plot_precision_recall_with_probis(df)

    # Re-run functional prediction (call the script)
    print("\nRe-running functional prediction with ProBiS...")
    os.system(f'python3 /mnt/results/benchmark/scripts/functional_prediction.py')
    plot_func_pred_with_probis()

    # Generate ProBiS report section
    print("\nGenerating ProBiS report section...")
    section = generate_probis_report_section(df, summary)

    # Write section to file
    section_file = '/workspace/probis_report_section.md'
    with open(section_file, 'w') as f:
        f.write(section)
    print(f"Saved ProBiS section to {section_file}")
    print("\n--- ProBiS Section ---")
    print(section)


if __name__ == '__main__':
    main()
