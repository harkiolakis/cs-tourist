#!/usr/bin/env python3
"""Generate Figure 5: Concordance distribution v10 (exact-offset) vs. v11 (connectivity-aware).

This script reproduces the v11_concordance_comparison.png figure from the paper,
showing how the connectivity-aware descriptor resolves the bimodal failure mode
of the exact-offset descriptor.

Usage:
    python benchmark/scripts/generate_concordance_figure.py \
        --v10-data data/cs_turist_eval.csv \
        --v11-data data/cs_tourist_eval_v11conn.csv \
        --output figures/v11_concordance_comparison.png
"""

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.rcParams as rc_params
import numpy as np
import pandas as pd

rc_params['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rc_params['svg.fonttype'] = 'none'


def plot_concordance_comparison(v10_path, v11_path, output_path):
    """Create side-by-side concordance histograms for v10 and v11."""
    # Load data
    v10 = pd.read_csv(v10_path)
    v11 = pd.read_csv(v11_path)

    # Filter to CS Mode C, no errors
    v10_mc = v10[(v10['method'] == 'cs_modeC') &
                 (v10['error'].isna() | (v10['error'] == '') | (v10['error'] == 'nan'))]
    v11_mc = v11[(v11['method'] == 'cs_modeC') &
                 (v11['error'].isna() | (v11['error'] == '') | (v11['error'] == 'nan'))]

    v10_conc = v10_mc['concordance'].dropna().values
    v11_conc = v11_mc['concordance'].dropna().values

    # Statistics
    v10_perfect = (v10_conc == 1.0).mean() * 100
    v10_fail = (v10_conc == 0.0).mean() * 100
    v11_perfect = (v11_conc == 1.0).mean() * 100
    v11_fail = (v11_conc == 0.0).mean() * 100

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.linspace(-0.05, 1.05, 50)

    # Left: v10 (exact-offset)
    ax1.hist(v10_conc, bins=bins, color='#FF9400', edgecolor='black',
             linewidth=0.5, alpha=0.85)
    ax1.set_title('Exact-offset (v10)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Concordance', fontsize=12)
    ax1.set_ylabel('Number of pairs', fontsize=12)
    ax1.text(0.05, 0.95, f'Perfect: {v10_perfect:.1f}%\nFail: {v10_fail:.1f}%',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right: v11 (connectivity-aware)
    ax2.hist(v11_conc, bins=bins, color='#0279EE', edgecolor='black',
             linewidth=0.5, alpha=0.85)
    ax2.set_title('Connectivity-aware (v11)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Concordance', fontsize=12)
    ax2.set_ylabel('Number of pairs', fontsize=12)
    ax2.text(0.05, 0.95, f'Perfect: {v11_perfect:.1f}%\nFail: {v11_fail:.1f}%',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  v10: {v10_perfect:.1f}% perfect, {v10_fail:.1f}% fail (n={len(v10_conc)})")
    print(f"  v11: {v11_perfect:.1f}% perfect, {v11_fail:.1f}% fail (n={len(v11_conc)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate concordance comparison figure')
    parser.add_argument('--v10-data', required=True, help='Path to v10 (exact-offset) results CSV')
    parser.add_argument('--v11-data', required=True, help='Path to v11 (connectivity-aware) results CSV')
    parser.add_argument('--output', default='v11_concordance_comparison.png',
                        help='Output figure path')
    args = parser.parse_args()

    plot_concordance_comparison(args.v10_data, args.v11_data, args.output)
