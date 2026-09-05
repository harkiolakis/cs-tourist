#!/usr/bin/env python3
"""
Evaluate and compare all methods on the catalytic site detection task.

Methods compared:
  - TOURIST (full alignment)
  - TOURIST (CS Mode A, r=6)
  - TOURIST (CS Mode C correspondence - separate analysis)
  - TM-align (backbone alignment baseline)
  - pyScoMotif (strict)
  - pyScoMotif (relaxed)
  - GASS
  - ProBiS

Metrics (standard ML definitions):
  - Recall = |predicted ∩ known| / |known|
  - Precision = |predicted ∩ known| / |predicted|
  - F1 = 2 * P * R / (P + R)
  - Runtime (seconds per pair)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

# Style
rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rcParams['svg.fonttype'] = 'none'

# Paths
DATA_DIR = "/mnt/results/benchmark/data"
FIGURES_DIR = "/mnt/results/benchmark/figures"
TOURIST_CSV = os.path.join(DATA_DIR, "cs_tourist_eval_v11conn.csv")
PROBIS_CSV = os.path.join(DATA_DIR, "probis_eval.csv")
PYSCOM_CSV = os.path.join(DATA_DIR, "competitor_pyscomotif.csv")
PYSCOM_RELAXED_CSV = os.path.join(DATA_DIR, "competitor_pyscomotif_relaxed.csv")
GASS_CSV = os.path.join(DATA_DIR, "competitor_gass.csv")
OUTPUT_DIR = DATA_DIR

def compute_f1(precision, recall):
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def load_tourist_results():
    """Load TOURIST results and recompute metrics with standard ML definitions.
    
    From raw columns:
      q_cat_aligned = number of A's catalytic residues aligned (-> predicted count)
      t_cat_total = total catalytic residues in B (-> known count)
      cat_to_cat = aligned pairs where both are catalytic (-> true positives)
    
    Standard ML:
      Recall = cat_to_cat / t_cat_total  (fraction of known catalytic residues found)
      Precision = cat_to_cat / q_cat_aligned  (fraction of predicted that are correct)
    """
    df = pd.read_csv(TOURIST_CSV)
    
    # Select methods to compare
    methods_map = {
        'tourist_full': 'TOURIST (full)',
        'cs_modeA_r6': 'TOURIST (Mode A)',
        'cs_modeC_conn': 'TOURIST (Mode C)',
        'tmalign': 'TM-align',
    }
    
    results = []
    for method, label in methods_map.items():
        md = df[df['method'] == method].copy()
        if len(md) == 0:
            continue
        
        # Recompute metrics with standard definitions
        md['recall_std'] = md.apply(
            lambda r: r['cat_to_cat'] / r['t_cat_total'] if r['t_cat_total'] > 0 else 0, axis=1)
        md['precision_std'] = md.apply(
            lambda r: r['cat_to_cat'] / r['q_cat_aligned'] if r['q_cat_aligned'] > 0 else 0, axis=1)
        md['f1_std'] = md.apply(
            lambda r: compute_f1(r['precision_std'], r['recall_std']), axis=1)
        
        md['method_label'] = label
        md['runtime_sec'] = 0  # Not tracked in existing results
        
        results.append(md[['pair_idx', 'method_label', 'query', 'target', 
                          'q_chain', 't_chain', 'cath_relation',
                          'recall_std', 'precision_std', 'f1_std', 'runtime_sec']].rename(columns={
            'recall_std': 'recall', 'precision_std': 'precision', 'f1_std': 'f1',
            'method_label': 'method'}))
    
    return pd.concat(results, ignore_index=True)

def load_pyscomotif_results():
    """Load pyScoMotif results (strict and relaxed)."""
    frames = []
    
    # Strict
    if os.path.exists(PYSCOM_CSV):
        df = pd.read_csv(PYSCOM_CSV)
        df['method'] = df['method'].map({
            'pyscomotif_strict': 'pyScoMotif (strict)',
            'pyscomotif_relaxed': 'pyScoMotif (relaxed)',
        })
        frames.append(df[['pair_idx', 'method', 'query', 'target', 'q_chain', 't_chain',
                          'cath_relation', 'recall', 'precision', 'f1', 'runtime_sec']])
    
    # Relaxed
    if os.path.exists(PYSCOM_RELAXED_CSV):
        df = pd.read_csv(PYSCOM_RELAXED_CSV)
        df['method'] = df['method'].map({
            'pyscomotif_strict': 'pyScoMotif (strict)',
            'pyscomotif_relaxed': 'pyScoMotif (relaxed)',
        })
        frames.append(df[['pair_idx', 'method', 'query', 'target', 'q_chain', 't_chain',
                          'cath_relation', 'recall', 'precision', 'f1', 'runtime_sec']])
    
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def load_gass_results():
    """Load GASS results."""
    if not os.path.exists(GASS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(GASS_CSV)
    df['method'] = 'GASS'
    return df[['pair_idx', 'method', 'query', 'target', 'q_chain', 't_chain',
              'cath_relation', 'recall', 'precision', 'f1', 'runtime_sec']]

def load_probis_results():
    """Load ProBiS results and recompute metrics with standard definitions."""
    df = pd.read_csv(PROBIS_CSV)
    
    # ProBiS has same columns as TOURIST
    df['recall_std'] = df.apply(
        lambda r: r['cat_to_cat'] / r['t_cat_total'] if r['t_cat_total'] > 0 else 0, axis=1)
    df['precision_std'] = df.apply(
        lambda r: r['cat_to_cat'] / r['q_cat_aligned'] if r['q_cat_aligned'] > 0 else 0, axis=1)
    df['f1_std'] = df.apply(
        lambda r: compute_f1(r['precision_std'], r['recall_std']), axis=1)
    
    df['method'] = 'ProBiS'
    df['runtime_sec'] = 0
    
    return df[['pair_idx', 'method', 'query', 'target', 'q_chain', 't_chain',
              'cath_relation', 'recall_std', 'precision_std', 'f1_std', 'runtime_sec']].rename(columns={
        'recall_std': 'recall', 'precision_std': 'precision', 'f1_std': 'f1'})

def generate_summary(all_results):
    """Generate summary statistics by method and CATH relation."""
    summaries = []
    
    for method in all_results['method'].unique():
        md = all_results[all_results['method'] == method]
        
        for cath in ['overall', 'same_h', 'same_t_diff_h', 'diff_t']:
            if cath == 'overall':
                cd = md
            else:
                cd = md[md['cath_relation'] == cath]
            
            if len(cd) == 0:
                continue
            
            summaries.append({
                'method': method,
                'cath_relation': cath,
                'n_pairs': len(cd),
                'mean_recall': cd['recall'].mean(),
                'median_recall': cd['recall'].median(),
                'mean_precision': cd['precision'].mean(),
                'median_precision': cd['precision'].median(),
                'mean_f1': cd['f1'].mean(),
                'median_f1': cd['f1'].median(),
                'mean_runtime': cd['runtime_sec'].mean() if cd['runtime_sec'].sum() > 0 else None,
                'perfect_recall_pct': (cd['recall'] == 1.0).mean() * 100,
                'zero_recall_pct': (cd['recall'] == 0.0).mean() * 100,
            })
    
    return pd.DataFrame(summaries)

def plot_overall_comparison(summary, output_path):
    """Bar chart: recall, precision, F1 by method (overall)."""
    overall = summary[summary['cath_relation'] == 'overall'].copy()
    
    # Order methods
    method_order = [
        'TOURIST (Mode C)', 'TOURIST (Mode A)', 'TOURIST (full)',
        'TM-align', 'pyScoMotif (strict)', 'pyScoMotif (relaxed)',
        'GASS', 'ProBiS'
    ]
    overall['method'] = pd.Categorical(overall['method'], categories=method_order, ordered=True)
    overall = overall.sort_values('method').dropna(subset=['method'])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [('mean_recall', 'Recall'), ('mean_precision', 'Precision'), ('mean_f1', 'F1')]
    colors = ['#0279EE', '#75A025', '#FD9BED', '#FF9400', '#E9ED4C', '#75A025', '#FF9400', '#999999']
    
    for ax, (col, title) in zip(axes, metrics):
        vals = overall[col].values
        labels = overall['method'].values
        x = range(len(vals))
        bars = ax.bar(x, vals, color=colors[:len(vals)], edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(f'{title} (overall)', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_cath_breakdown(summary, output_path):
    """Grouped bar chart: recall by CATH relation x method."""
    cath_order = ['same_h', 'same_t_diff_h', 'diff_t']
    cath_labels = ['Same H', 'Same T diff H', 'Diff T']
    
    method_order = [
        'TOURIST (Mode C)', 'TOURIST (Mode A)', 'TOURIST (full)',
        'TM-align', 'pyScoMotif (strict)', 'pyScoMotif (relaxed)',
        'GASS', 'ProBiS'
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [('mean_recall', 'Recall'), ('mean_precision', 'Precision'), ('mean_f1', 'F1')]
    
    for ax, (col, title) in zip(axes, metrics):
        x = np.arange(len(cath_order))
        width = 0.1
        n_methods = 0
        
        for mi, method in enumerate(method_order):
            vals = []
            for cath in cath_order:
                row = summary[(summary['method'] == method) & (summary['cath_relation'] == cath)]
                if len(row) > 0:
                    vals.append(row[col].values[0])
                else:
                    vals.append(0)
            
            if all(v == 0 for v in vals):
                continue
            
            offset = (mi - len(method_order)/2) * width + width/2
            color = plt.cm.tab10(mi / 10)
            ax.bar(x + offset, vals, width, label=method, color=color, edgecolor='black', linewidth=0.3)
            n_methods += 1
        
        ax.set_xticks(x)
        ax.set_xticklabels(cath_labels, fontsize=9)
        ax.set_ylabel(title)
        ax.set_title(f'{title} by CATH relation', fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def plot_runtime_comparison(summary, output_path):
    """Bar chart: average runtime per pair by method."""
    runtime_data = summary[
        (summary['cath_relation'] == 'overall') & 
        (summary['mean_runtime'].notna()) & 
        (summary['mean_runtime'] > 0)
    ].copy()
    
    if len(runtime_data) == 0:
        print("  No runtime data available, skipping runtime plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(runtime_data))
    bars = ax.bar(x, runtime_data['mean_runtime'].values, 
                  color='#0279EE', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(runtime_data['method'].values, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Runtime (seconds per pair)')
    ax.set_title('Runtime comparison', fontsize=11)
    
    for bar, val in zip(bars, runtime_data['mean_runtime'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    print("Loading results...")
    
    # Load all results
    tourist = load_tourist_results()
    print(f"  TOURIST: {len(tourist)} rows, methods: {tourist['method'].unique().tolist()}")
    
    pyscom = load_pyscomotif_results()
    print(f"  pyScoMotif: {len(pyscom)} rows")
    
    gass = load_gass_results()
    print(f"  GASS: {len(gass)} rows")
    
    probis = load_probis_results()
    print(f"  ProBiS: {len(probis)} rows")
    
    # Merge all
    all_results = pd.concat([tourist, pyscom, gass, probis], ignore_index=True)
    print(f"\nTotal: {len(all_results)} rows across {all_results['method'].nunique()} methods")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = generate_summary(all_results)
    summary_path = os.path.join(OUTPUT_DIR, "competitor_comparison_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    
    # Print summary table
    print("\n=== Overall Results ===")
    overall = summary[summary['cath_relation'] == 'overall']
    print(overall[['method', 'n_pairs', 'mean_recall', 'mean_precision', 'mean_f1', 
                   'perfect_recall_pct']].to_string(index=False))
    
    print("\n=== Results by CATH Relation ===")
    for cath in ['same_h', 'same_t_diff_h', 'diff_t']:
        cd = summary[summary['cath_relation'] == cath]
        if len(cd) > 0:
            print(f"\n--- {cath} ---")
            print(cd[['method', 'n_pairs', 'mean_recall', 'mean_precision', 'mean_f1']].to_string(index=False))
    
    # Generate figures
    print("\nGenerating figures...")
    plot_overall_comparison(summary, os.path.join(FIGURES_DIR, "competitor_comparison.png"))
    plot_cath_breakdown(summary, os.path.join(FIGURES_DIR, "competitor_cath_breakdown.png"))
    plot_runtime_comparison(summary, os.path.join(FIGURES_DIR, "competitor_runtime.png"))
    
    # Save merged results
    all_path = os.path.join(OUTPUT_DIR, "competitor_all_results.csv")
    all_results.to_csv(all_path, index=False)
    print(f"\nSaved all results: {all_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
