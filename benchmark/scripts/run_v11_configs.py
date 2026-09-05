#!/usr/bin/env python3
"""Run v11 benchmark configs on 50K pairs (or pilot subset).

Reads v11_configs.json, runs each config via benchmark_fast.py,
computes AUC at 4 levels, and saves results.

Usage:
    python run_v11_configs.py --configs v11_configs.json --n-pairs 5000 --output-dir /mnt/results/benchmark/data/pilot_v11
    python run_v11_configs.py --configs v11_configs.json --n-pairs 50000 --output-dir /mnt/results/benchmark/data/v11_full
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

PAIR_LIST = '/mnt/shared-workspace/pair_list.csv'
SCRIPTS_DIR = '/mnt/results/benchmark/scripts'


def get_labels(df, level):
    """Get binary labels for a given classification level."""
    if level == 'scop_sf':
        mask = df['scop_relation'].isin(['same_sf', 'diff_fold'])
        labels = (df.loc[mask, 'scop_relation'] == 'same_sf').astype(int)
        return labels, mask
    elif level == 'scop_fold':
        mask = df['scop_relation'].isin(['same_sf', 'same_fold_diff_sf', 'diff_fold'])
        labels = df.loc[mask, 'scop_relation'].isin(['same_sf', 'same_fold_diff_sf']).astype(int)
        return labels, mask
    elif level == 'cath_h':
        mask = df['cath_relation'].isin(['same_h', 'diff_t'])
        labels = (df.loc[mask, 'cath_relation'] == 'same_h').astype(int)
        return labels, mask
    elif level == 'cath_t':
        mask = df['cath_relation'].isin(['same_h', 'same_t_diff_h', 'diff_t'])
        labels = df.loc[mask, 'cath_relation'].isin(['same_h', 'same_t_diff_h']).astype(int)
        return labels, mask
    return None, None


def compute_auc(df, score_col, level):
    """Compute AUC for a given score column at a given level."""
    labels, mask = get_labels(df, level)
    if labels is None:
        return None
    scores = df.loc[mask, score_col].astype(float)
    if labels.nunique() < 2:
        return None
    return roc_auc_score(labels, scores)


def run_config(config, n_pairs, output_dir):
    """Run a single config on n_pairs and return the output file path."""
    import pandas as pd

    name = config['name']
    output_file = os.path.join(output_dir, f'{name}.csv')

    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        if len(df) >= n_pairs * 0.95:
            print(f'  {name}: already complete ({len(df)} rows)', flush=True)
            return output_file

    # Build command
    cmd = [
        sys.executable, '-u',
        os.path.join(SCRIPTS_DIR, 'benchmark_fast.py'),
        '--n-pairs', str(n_pairs),
        '--output', output_file,
    ]

    # Add descriptor args
    desc = config.get('descriptor', 'radial3')
    cmd.extend(['--descriptor', desc])
    if 'radius' in config:
        cmd.extend(['--radius', str(config['radius'])])
    if 'r1' in config:
        cmd.extend(['--r1', str(config['r1'])])
    if 'r2' in config:
        cmd.extend(['--r2', str(config['r2'])])

    # Cache file
    cache_file = config.get('cache_file', '')
    if cache_file:
        cmd.extend(['--cache-file', cache_file])

    # Refinement
    if config.get('refined'):
        cmd.append('--refined')

    # BLOSUM
    if 'beta' in config and config['beta'] > 0:
        cmd.extend(['--beta', str(config['beta'])])

    # Affine gap
    if 'gap_open' in config and config['gap_open'] is not None:
        cmd.extend(['--gap-open', str(config['gap_open'])])
        if 'gap_extend' in config:
            cmd.extend(['--gap-extend', str(config['gap_extend'])])

    # SSE
    if 'sse_weight' in config and config['sse_weight'] > 0:
        cmd.extend(['--sse-weight', str(config['sse_weight'])])
        cmd.extend(['--sse-file', '/mnt/shared-workspace/shared/sse_annotations.pkl'])

    # v11: Connectivity-aware scoring
    if 'w_conn' in config and config['w_conn'] > 0:
        cmd.extend(['--w-conn', str(config['w_conn'])])
        if 'conn_cache_file' in config:
            cmd.extend(['--conn-cache-file', config['conn_cache_file']])

    # v11: Multi-scale scoring
    if config.get('multi_scale'):
        cmd.append('--multi-scale')
        if 'ms_cache_files' in config:
            cmd.extend(['--ms-cache-files', config['ms_cache_files']])
        if 'ms_weights' in config:
            cmd.extend(['--ms-weights', config['ms_weights']])

    # Run
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f'  {name}: FAILED (exit {result.returncode})', flush=True)
        print(f'  STDERR: {result.stderr[:500]}', flush=True)
        return None

    print(f'  {name}: done in {elapsed:.1f}s', flush=True)
    return output_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-pairs', type=int, default=50000)
    parser.add_argument('--output-dir', default='/mnt/results/benchmark/data/v11_full')
    parser.add_argument('--configs', default='/mnt/results/benchmark/scripts/v11_configs.json')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    import pandas as pd

    # Load pair list for AUC computation
    pairs_df = pd.read_csv(PAIR_LIST)

    # Load configs
    with open(args.configs) as f:
        configs = json.load(f)

    print(f'Running {len(configs)} configs on {args.n_pairs} pairs', flush=True)
    print(f'Output dir: {args.output_dir}', flush=True)

    # Run each config
    results_files = []
    for i, config in enumerate(configs):
        print(f'\n[{i+1}/{len(configs)}] {config["name"]}...', flush=True)
        output_file = run_config(config, args.n_pairs, args.output_dir)
        if output_file:
            results_files.append((config['name'], output_file))

    # Compute AUC for each config
    print(f'\n{"="*80}', flush=True)
    print(f'AUC RESULTS ({args.n_pairs} pairs)', flush=True)
    print(f'{"="*80}', flush=True)
    print(f'{"Config":<30} {"SCOPe SF":>10} {"SCOPe Fold":>11} {"CATH H":>10} {"CATH T":>10}', flush=True)
    print(f'{"-"*30} {"-"*10} {"-"*11} {"-"*10} {"-"*10}', flush=True)

    all_auc = []
    for name, output_file in results_files:
        df = pd.read_csv(output_file)
        # Merge with pair list for relations
        merged = df.merge(pairs_df,
            left_on=['query', 'target', 'q_chain', 't_chain'],
            right_on=['query_id', 'target_id', 'query_chain', 'target_chain'],
            how='inner')

        # Find score column (tourist_tm_like or turist_tm_like)
        score_col = 'tourist_tm_like' if 'tourist_tm_like' in merged.columns else None
        if score_col is None:
            for col in merged.columns:
                if 'tm_like' in col.lower():
                    score_col = col
                    break

        aucs = {}
        for level in ['scop_sf', 'scop_fold', 'cath_h', 'cath_t']:
            auc = compute_auc(merged, score_col, level)
            aucs[level] = auc

        print(f'{name:<30} {aucs["scop_sf"] or 0:>10.4f} {aucs["scop_fold"] or 0:>11.4f} {aucs["cath_h"] or 0:>10.4f} {aucs["cath_t"] or 0:>10.4f}', flush=True)

        for level, auc in aucs.items():
            if auc is not None:
                all_auc.append({
                    'method': name,
                    'level': level,
                    'auc': auc,
                    'n_pairs': len(merged),
                })

    # Save AUC results
    auc_file = os.path.join(args.output_dir, 'v11_auc_results.csv')
    pd.DataFrame(all_auc).to_csv(auc_file, index=False)
    print(f'\nSaved AUC to {auc_file}', flush=True)


if __name__ == '__main__':
    main()
