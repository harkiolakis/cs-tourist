#!/usr/bin/env python3
"""Run ProBiS on CSA pairs and compute catalytic concordance.

ProBiS finds local structural alignments between protein surfaces.
We parse the .nosql output to extract residue correspondences and
compute catalytic concordance for each pair.

Usage:
    python run_probis.py [--start 0] [--end 1000]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import subprocess
import numpy as np
from collections import defaultdict

sys.path.insert(0, '/mnt/shared-workspace/tourist')
sys.path.insert(0, '/mnt/results/benchmark/scripts')

from tourist.io import parse_chain_trace

PDB_BENCHMARK = '/mnt/shared-workspace/pdb_benchmark'
PDB_CSA = '/mnt/shared-workspace/pdb_csa'
CSA_SITES = '/mnt/shared-workspace/shared/csa_sites.json'
PAIR_FILE = '/mnt/shared-workspace/shared/csa_pairs_1000.csv'
OUTPUT_DIR = '/mnt/results/benchmark/data'
PROBIS = '/workspace/probis'
WORK_DIR = '/workspace/probis_work'

def get_pdb_path(pdb_id):
    p1 = os.path.join(PDB_BENCHMARK, f'{pdb_id}.pdb')
    p2 = os.path.join(PDB_CSA, f'{pdb_id}.pdb')
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    return None

def build_resseq_map(trace):
    return {trace.res_ids[i][1]: i for i in range(len(trace.res_ids))}

def convert_sites_to_trace_idx(site_list, resseq_map):
    result = set()
    for resseq in site_list:
        idx = resseq_map.get(resseq)
        if idx is not None:
            result.add(idx)
    return result

def compute_concordance(correspondence, q_cat_set, t_cat_set):
    q_cat_aligned = set()
    t_cat_aligned = set()
    cat_to_cat = 0
    for (qi, tj) in correspondence:
        q_is_cat = qi in q_cat_set
        t_is_cat = tj in t_cat_set
        if q_is_cat:
            q_cat_aligned.add(qi)
        if t_is_cat:
            t_cat_aligned.add(tj)
        if q_is_cat and t_is_cat:
            cat_to_cat += 1
    denom = len(q_cat_aligned) + len(t_cat_aligned) - cat_to_cat
    concordance = cat_to_cat / denom if denom > 0 else 0.0
    recall = cat_to_cat / len(q_cat_set) if len(q_cat_set) > 0 else 0.0
    precision = cat_to_cat / len(t_cat_aligned) if len(t_cat_aligned) > 0 else 0.0
    return {
        'q_cat_total': len(q_cat_set), 't_cat_total': len(t_cat_set),
        'q_cat_aligned': len(q_cat_aligned), 't_cat_aligned': len(t_cat_aligned),
        'cat_to_cat': cat_to_cat, 'concordance': concordance,
        'recall': recall, 'precision': precision,
    }

def parse_probis_nosql(nosql_file, q_chain, t_chain):
    """Parse ProBiS .nosql output file.
    
    Returns list of alignments, each with:
    - alignment_no: int
    - scores: dict of score values
    - correspondences: list of (q_resseq, t_resseq) tuples
    """
    if not os.path.exists(nosql_file):
        return []
    
    with open(nosql_file) as f:
        content = f.read().strip()
    
    if not content:
        return []
    
    # Split into lines (each line is one target comparison)
    lines = content.split('\n')
    
    alignments = []
    
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 4:
            continue
        
        # Parse score tuples (third field)
        score_str = parts[2]
        score_tuples = re.findall(r'\(([^)]+)\)', score_str)
        
        # Parse residue correspondences (fourth field)
        corr_str = parts[3]
        corr_tuples = re.findall(r'\(([^)]+)\)', corr_str)
        
        # Group correspondences by alignment number
        by_aln = defaultdict(list)
        for ct in corr_tuples:
            fields = ct.split(',')
            if len(fields) >= 8:
                # (type, chain1, resname1, resseq1, chain2, resname2, resseq2, aln_no)
                try:
                    q_resseq = int(fields[3])
                    t_resseq = int(fields[6])
                    aln_no = int(fields[7])
                    by_aln[aln_no].append((q_resseq, t_resseq))
                except ValueError:
                    continue
        
        # Parse scores by alignment
        score_by_aln = {}
        for st in score_tuples:
            fields = st.split(',')
            if len(fields) >= 4:
                try:
                    aln_no = int(fields[0])
                    n_aligned = int(fields[1])
                    # Extract scores (varies by version)
                    score_by_aln[aln_no] = {
                        'n_aligned': n_aligned,
                        'raw_scores': fields[2:],
                    }
                except ValueError:
                    continue
        
        # Build alignment objects
        for aln_no, corrs in by_aln.items():
            scores = score_by_aln.get(aln_no, {})
            alignments.append({
                'alignment_no': aln_no,
                'n_correspondences': len(corrs),
                'correspondences': corrs,
                'scores': scores,
            })
    
    return alignments

def run_probis_pair(q_id, t_id, q_chain, t_chain, work_dir):
    """Run ProBiS on a single pair and return best alignment correspondences."""
    q_pdb = get_pdb_path(q_id)
    t_pdb = get_pdb_path(t_id)
    
    if not q_pdb or not t_pdb:
        return [], 'pdb_missing'
    
    nosql_file = os.path.join(work_dir, f'{q_id}_{q_chain}_{t_id}_{t_chain}.nosql')
    
    # Remove old nosql file
    if os.path.exists(nosql_file):
        os.remove(nosql_file)
    
    try:
        result = subprocess.run(
            [PROBIS, '-compare', '-f1', q_pdb, '-c1', q_chain,
             '-f2', t_pdb, '-c2', t_chain, '-nosql', nosql_file],
            capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return [], 'timeout'
    except Exception as e:
        return [], str(e)[:100]
    
    # Parse output
    alignments = parse_probis_nosql(nosql_file, q_chain, t_chain)
    
    if not alignments:
        return [], 'no_alignments'
    
    # Select best alignment (most correspondences)
    best = max(alignments, key=lambda a: a['n_correspondences'])
    
    return best['correspondences'], ''

def main():
    parser = argparse.ArgumentParser(description='Run ProBiS on CSA pairs')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    args = parser.parse_args()
    
    os.makedirs(WORK_DIR, exist_ok=True)
    
    # Load data
    with open(CSA_SITES) as f:
        csa = json.load(f)
    with open(PAIR_FILE) as f:
        pairs = list(csv.DictReader(f))
    
    start = max(0, args.start)
    end = min(len(pairs), args.end)
    subset = pairs[start:end]
    
    print(f"Running ProBiS on {len(subset)} pairs ({start}-{end})", flush=True)
    
    t0 = time.time()
    results = []
    
    for i, p in enumerate(subset):
        idx = start + i
        q_id = p['query_id']
        t_id = p['target_id']
        q_chain = p['query_chain']
        t_chain = p['target_chain']
        cath_rel = p['cath_relation']
        
        q_cat_list = csa.get(q_id, {}).get(q_chain, [])
        t_cat_list = csa.get(t_id, {}).get(t_chain, [])
        
        if len(q_cat_list) < 2 or len(t_cat_list) < 2:
            continue
        
        # Get PDB paths
        q_pdb = get_pdb_path(q_id)
        t_pdb = get_pdb_path(t_id)
        if not q_pdb or not t_pdb:
            results.append({
                'pair_idx': idx, 'method': 'probis',
                'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
                'cath_relation': cath_rel,
                'n_aligned': 0, 'rmsd': 0, 'tm_like': 0,
                'q_cat_total': len(q_cat_list), 't_cat_total': len(t_cat_list),
                'q_cat_aligned': 0, 't_cat_aligned': 0, 'cat_to_cat': 0,
                'concordance': 0, 'recall': 0, 'precision': 0,
                'error': 'pdb_missing',
            })
            continue
        
        # Parse chains to get trace indices
        try:
            trace_a = parse_chain_trace(q_pdb, q_chain)
            trace_b = parse_chain_trace(t_pdb, t_chain)
        except Exception as e:
            results.append({
                'pair_idx': idx, 'method': 'probis',
                'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
                'cath_relation': cath_rel,
                'n_aligned': 0, 'rmsd': 0, 'tm_like': 0,
                'q_cat_total': len(q_cat_list), 't_cat_total': len(t_cat_list),
                'q_cat_aligned': 0, 't_cat_aligned': 0, 'cat_to_cat': 0,
                'concordance': 0, 'recall': 0, 'precision': 0,
                'error': f'parse:{str(e)[:80]}',
            })
            continue
        
        q_resseq_map = build_resseq_map(trace_a)
        t_resseq_map = build_resseq_map(trace_b)
        q_cat_set = convert_sites_to_trace_idx(q_cat_list, q_resseq_map)
        t_cat_set = convert_sites_to_trace_idx(t_cat_list, t_resseq_map)
        
        # Run ProBiS
        corr_resseq, err = run_probis_pair(q_id, t_id, q_chain, t_chain, WORK_DIR)
        
        # Convert resseq correspondences to trace indices
        corr_trace = []
        for q_resseq, t_resseq in corr_resseq:
            q_idx = q_resseq_map.get(q_resseq)
            t_idx = t_resseq_map.get(t_resseq)
            if q_idx is not None and t_idx is not None:
                corr_trace.append((q_idx, t_idx))
        
        conc = compute_concordance(corr_trace, q_cat_set, t_cat_set)
        
        results.append({
            'pair_idx': idx, 'method': 'probis',
            'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
            'cath_relation': cath_rel,
            'n_aligned': len(corr_trace), 'rmsd': 0, 'tm_like': 0,
            'q_cat_total': conc['q_cat_total'], 't_cat_total': conc['t_cat_total'],
            'q_cat_aligned': conc['q_cat_aligned'], 't_cat_aligned': conc['t_cat_aligned'],
            'cat_to_cat': conc['cat_to_cat'],
            'concordance': conc['concordance'],
            'recall': conc['recall'], 'precision': conc['precision'],
            'error': err,
        })
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(subset) - i - 1) / rate
            print(f"  {i+1}/{len(subset)} ({elapsed:.0f}s, {rate:.1f} pairs/s, ETA {eta:.0f}s)", flush=True)
    
    elapsed = time.time() - t0
    print(f"\nDone: {len(results)} results in {elapsed:.1f}s", flush=True)
    
    # Save results
    suffix = f'_{start}_{end}' if start > 0 or end < 1000 else ''
    detail_file = os.path.join(OUTPUT_DIR, f'probis_eval{suffix}.csv')
    fieldnames = ['pair_idx', 'method', 'query', 'target', 'q_chain', 't_chain',
                  'cath_relation', 'n_aligned', 'rmsd', 'tm_like',
                  'q_cat_total', 't_cat_total', 'q_cat_aligned', 't_cat_aligned',
                  'cat_to_cat', 'concordance', 'recall', 'precision', 'error']
    
    tmp_path = f'/workspace/probis_eval{suffix}.csv'
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    os.system(f'cp {tmp_path} {detail_file}')
    print(f"Saved to {detail_file}", flush=True)
    
    # Summary
    valid = [r for r in results if not r['error']]
    errors = [r for r in results if r['error']]
    print(f"\nValid: {len(valid)}, Errors: {len(errors)}", flush=True)
    if errors:
        err_counts = defaultdict(int)
        for r in errors:
            err_counts[r['error']] += 1
        print("Error types:", flush=True)
        for err, count in sorted(err_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {count}", flush=True)
    
    if valid:
        mean_conc = np.mean([r['concordance'] for r in valid])
        mean_recall = np.mean([r['recall'] for r in valid])
        mean_prec = np.mean([r['precision'] for r in valid])
        mean_aln = np.mean([r['n_aligned'] for r in valid])
        print(f"\nProBiS Summary:", flush=True)
        print(f"  Mean concordance: {mean_conc:.4f}", flush=True)
        print(f"  Mean recall: {mean_recall:.4f}", flush=True)
        print(f"  Mean precision: {mean_prec:.4f}", flush=True)
        print(f"  Mean n_aligned: {mean_aln:.1f}", flush=True)
        
        # By CATH relationship
        for rel in ['same_h', 'same_t_diff_h', 'diff_t']:
            rel_rows = [r for r in valid if r['cath_relation'] == rel]
            if rel_rows:
                mc = np.mean([r['concordance'] for r in rel_rows])
                mr = np.mean([r['recall'] for r in rel_rows])
                mp = np.mean([r['precision'] for r in rel_rows])
                print(f"  {rel} ({len(rel_rows)}): conc={mc:.4f}, recall={mr:.4f}, prec={mp:.4f}", flush=True)

if __name__ == '__main__':
    main()
