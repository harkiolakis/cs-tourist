#!/usr/bin/env python3
"""Full CS-TOURIST evaluation on 1000 CSA pairs.

Runs all three CS-TOURIST modes (A: neighborhood, B: catalytic frame, C: catalytic-only)
plus standard TOURIST (full-chain) and TM-align for comparison.
Computes catalytic concordance for each method.

Usage:
    python cs_tourist_eval.py [--workers 4] [--start 0] [--end 1000]
"""

import argparse
import csv
import json
import os
import sys
import time
import pickle
import subprocess
import numpy as np
from multiprocessing import Pool
from collections import defaultdict

sys.path.insert(0, '/mnt/shared-workspace/tourist')
sys.path.insert(0, '/mnt/results/benchmark/scripts')

from tourist.io import parse_chain_trace
from tourist.descriptor import (
    relative_offsets_multi,
    build_all_signatures_radial,
    build_all_signatures_radial_3sphere,
    RADIAL_MATCH_SLOTS,
    RADIAL_MATCH_SLOTS_3SPHERE,
)
from tourist.superpose import kabsch, apply_transform
from tourist_fast import dp_local_float, dp_local_affine, warmup
from cs_tourist import (
    run_cs_tourist_mode_a,
    run_cs_tourist_mode_b,
    run_cs_tourist_mode_c,
    build_compact_signatures,
    compute_radial_score_matrix,
    compute_blosum_score_matrix,
    compute_d0,
)

# Paths
PDB_BENCHMARK = '/mnt/shared-workspace/pdb_benchmark'
PDB_CSA = '/mnt/shared-workspace/pdb_csa'
CSA_SITES = '/workspace/csa_sites.json'
PAIR_FILE = '/mnt/shared-workspace/shared/csa_pairs_1000.csv'
RADIAL_CACHE = '/mnt/shared-workspace/shared/cache_radial_r8.pkl'
OUTPUT_DIR = '/mnt/results/benchmark/data'
TMALIGN = '/workspace/TMalign'

# Constants
GAP_PENALTY = -2.0
RADIUS = 8.0
MAX_ITER = 5
DIST_WEIGHT = 1.0

# Global state for workers
_g = {}

def init_worker(use_connectivity=False, w_conn=1.0):
    """Initialize worker process."""
    warmup()
    with open(CSA_SITES) as f:
        _g['csa'] = json.load(f)
    with open(PAIR_FILE) as f:
        _g['pairs'] = list(csv.DictReader(f))
    _g['use_connectivity'] = use_connectivity
    _g['w_conn'] = w_conn

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

def run_standard_tourist(trace_a, compact_a, trace_b, compact_b, beta=0.5, refined=True):
    """Run standard full-chain TOURIST alignment."""
    coords_a = trace_a.coords
    coords_b = trace_b.coords
    nA, nB = len(coords_a), len(coords_b)
    L_query = nA

    S_octant = compute_radial_score_matrix(compact_a, compact_b, nA, nB, RADIAL_MATCH_SLOTS)
    S_base = S_octant
    if beta > 0:
        S_blosum = compute_blosum_score_matrix(trace_a.sequence, trace_b.sequence, nA, nB)
        S_base = S_base + beta * S_blosum

    S_float = S_base.astype(float)

    def do_dp(S):
        return dp_local_float(S, gap_penalty=GAP_PENALTY)

    if not refined:
        pairs, _ = do_dp(S_float)
        if len(pairs) < 3:
            return [], 0, 0, 0.0
        a_idx = np.array([p[0] for p in pairs])
        b_idx = np.array([p[1] for p in pairs])
        R, t, rmsd = kabsch(coords_b[b_idx], coords_a[a_idx])
        n_aligned = len(pairs)
    else:
        d0 = compute_d0(min(nA, nB))
        d0_q = compute_d0(L_query)
        pairs, _ = do_dp(S_float)
        if len(pairs) < 3:
            return [], 0, 0, 0.0
        a_idx = np.array([p[0] for p in pairs])
        b_idx = np.array([p[1] for p in pairs])
        R, t, rmsd = kabsch(coords_b[b_idx], coords_a[a_idx])
        best_tm = (len(pairs) / L_query) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))
        best_pairs = pairs
        best_rmsd = rmsd

        for iteration in range(1, MAX_ITER + 1):
            transformed_b = apply_transform(coords_b, R, t)
            diffs = coords_a[:, None, :] - transformed_b[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            dscore = 1.0 / (1.0 + (distances / d0) ** 2)
            S_hybrid = S_float + DIST_WEIGHT * dscore * 7.0
            pairs_new, _ = do_dp(S_hybrid)
            if len(pairs_new) < 3:
                break
            a_idx = np.array([p[0] for p in pairs_new])
            b_idx = np.array([p[1] for p in pairs_new])
            R_new, t_new, rmsd_new = kabsch(coords_b[b_idx], coords_a[a_idx])
            tm_new = (len(pairs_new) / L_query) * (1.0 / (1.0 + (rmsd_new / d0_q) ** 2))
            if tm_new > best_tm:
                improvement = tm_new - best_tm
                best_tm = tm_new
                best_pairs = pairs_new
                best_rmsd = rmsd_new
                R, t = R_new, t_new
            else:
                R, t = R_new, t_new
                improvement = 0.0
            if improvement < 0.001:
                break

        pairs = best_pairs
        rmsd = best_rmsd
        n_aligned = len(pairs)

    d0_q = compute_d0(L_query)
    tm_like = (n_aligned / L_query) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))
    return list(pairs), rmsd, n_aligned, tm_like

def parse_tmalign_alignment(stdout):
    lines = stdout.split('\n')
    denotes_idx = None
    for i, line in enumerate(lines):
        if 'denotes' in line:
            denotes_idx = i
            break
    if denotes_idx is None or denotes_idx + 3 >= len(lines):
        return [], 0.0, 0.0
    query_line = lines[denotes_idx + 1]
    match_line = lines[denotes_idx + 2]
    target_line = lines[denotes_idx + 3]
    
    # Extract TM-score and RMSD
    tm_score = 0.0
    rmsd = 0.0
    for line in lines:
        if line.startswith('TM-score='):
            try:
                tm_score = float(line.split('=')[1].split()[0])
            except:
                pass
        if 'RMSD=' in line and 'in the superimposed' in line:
            try:
                rmsd = float(line.split('RMSD=')[1].split()[0])
            except:
                pass
    
    correspondences = []
    q_idx = 0
    t_idx = 0
    for qc, mc, tc in zip(query_line, match_line, target_line):
        if qc != '-' and qc != ' ':
            if tc != '-' and tc != ' ':
                if mc == ':' or mc == '.' or mc == ' ':
                    correspondences.append((q_idx, t_idx))
            if tc == '-' or tc == ' ':
                q_idx += 1
            else:
                q_idx += 1
                t_idx += 1
        else:
            if tc != '-' and tc != ' ':
                t_idx += 1
    
    return correspondences, tm_score, rmsd

def run_tmalign(q_id, t_id, q_chain, t_chain):
    q_pdb = get_pdb_path(q_id)
    t_pdb = get_pdb_path(t_id)
    if not q_pdb or not t_pdb:
        return [], 0.0, 0.0, 'pdb_missing'
    try:
        result = subprocess.run(
            [TMALIGN, q_pdb, t_pdb],
            capture_output=True, text=True, timeout=60
        )
        corr, tm_score, rmsd = parse_tmalign_alignment(result.stdout)
        return corr, tm_score, rmsd, ''
    except Exception as e:
        return [], 0.0, 0.0, str(e)[:100]

def process_pair(task):
    """Process a single CSA pair with all methods."""
    idx, p = task
    q_id = p['query_id']
    t_id = p['target_id']
    q_chain = p['query_chain']
    t_chain = p['target_chain']
    cath_rel = p['cath_relation']
    
    csa = _g['csa']
    results = []
    
    # Get catalytic sites
    q_cat_list = csa.get(q_id, {}).get(q_chain, [])
    t_cat_list = csa.get(t_id, {}).get(t_chain, [])
    
    if len(q_cat_list) < 2 or len(t_cat_list) < 2:
        return results
    
    # Get PDB paths
    q_pdb = get_pdb_path(q_id)
    t_pdb = get_pdb_path(t_id)
    if not q_pdb or not t_pdb:
        return results
    
    # Parse chains
    try:
        trace_a = parse_chain_trace(q_pdb, q_chain)
        trace_b = parse_chain_trace(t_pdb, t_chain)
    except Exception as e:
        return results
    
    # Convert catalytic sites to trace indices
    q_resseq_map = build_resseq_map(trace_a)
    t_resseq_map = build_resseq_map(trace_b)
    q_cat_set = convert_sites_to_trace_idx(q_cat_list, q_resseq_map)
    t_cat_set = convert_sites_to_trace_idx(t_cat_list, t_resseq_map)
    
    if len(q_cat_set) < 2 or len(t_cat_set) < 2:
        return results
    
    # --- Standard TOURIST (full-chain) ---
    try:
        compact_a_full = build_compact_signatures(trace_a.coords, RADIUS,
                                                   n_slots=16, match_slots=set(RADIAL_MATCH_SLOTS))
        compact_b_full = build_compact_signatures(trace_b.coords, RADIUS,
                                                   n_slots=16, match_slots=set(RADIAL_MATCH_SLOTS))
        pairs, rmsd, n_aln, tm_like = run_standard_tourist(
            trace_a, compact_a_full, trace_b, compact_b_full, beta=0.5, refined=True)
        conc = compute_concordance(pairs, q_cat_set, t_cat_set)
        results.append({
            'pair_idx': idx, 'method': 'tourist_full',
            'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
            'cath_relation': cath_rel,
            'n_aligned': n_aln, 'rmsd': rmsd, 'tm_like': tm_like,
            'q_cat_total': conc['q_cat_total'], 't_cat_total': conc['t_cat_total'],
            'q_cat_aligned': conc['q_cat_aligned'], 't_cat_aligned': conc['t_cat_aligned'],
            'cat_to_cat': conc['cat_to_cat'],
            'concordance': conc['concordance'],
            'recall': conc['recall'], 'precision': conc['precision'],
            'error': '',
        })
    except Exception as e:
        results.append({
            'pair_idx': idx, 'method': 'tourist_full',
            'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
            'cath_relation': cath_rel,
            'n_aligned': 0, 'rmsd': 0, 'tm_like': 0,
            'q_cat_total': len(q_cat_set), 't_cat_total': len(t_cat_set),
            'q_cat_aligned': 0, 't_cat_aligned': 0, 'cat_to_cat': 0,
            'concordance': 0, 'recall': 0, 'precision': 0,
            'error': str(e)[:100],
        })
    
    # --- CS-TOURIST Mode A (neighborhood extraction) ---
    for nb_radius in [6.0, 8.0, 10.0]:
        try:
            pairs, rmsd, n_aln, tm_like, info = run_cs_tourist_mode_a(
                trace_a, trace_b, q_cat_set, t_cat_set,
                neighborhood_radius=nb_radius, descriptor_radius=8.0,
                refined=True, beta=0.5)
            conc = compute_concordance(pairs, q_cat_set, t_cat_set)
            results.append({
                'pair_idx': idx, 'method': f'cs_modeA_r{int(nb_radius)}',
                'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
                'cath_relation': cath_rel,
                'n_aligned': n_aln, 'rmsd': rmsd, 'tm_like': tm_like,
                'q_cat_total': conc['q_cat_total'], 't_cat_total': conc['t_cat_total'],
                'q_cat_aligned': conc['q_cat_aligned'], 't_cat_aligned': conc['t_cat_aligned'],
                'cat_to_cat': conc['cat_to_cat'],
                'concordance': conc['concordance'],
                'recall': conc['recall'], 'precision': conc['precision'],
                'error': '',
            })
        except Exception as e:
            results.append({
                'pair_idx': idx, 'method': f'cs_modeA_r{int(nb_radius)}',
                'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
                'cath_relation': cath_rel,
                'n_aligned': 0, 'rmsd': 0, 'tm_like': 0,
                'q_cat_total': len(q_cat_set), 't_cat_total': len(t_cat_set),
                'q_cat_aligned': 0, 't_cat_aligned': 0, 'cat_to_cat': 0,
                'concordance': 0, 'recall': 0, 'precision': 0,
                'error': str(e)[:100],
            })
    
    # --- CS-TOURIST Mode B (catalytic-site-centered frame) ---
    try:
        pairs, rmsd, n_aln, tm_like, info = run_cs_tourist_mode_b(
            trace_a, trace_b, q_cat_set, t_cat_set,
            radius=12.0, r1=4.0, r2=8.0,
            refined=True, beta=0.5)
        conc = compute_concordance(pairs, q_cat_set, t_cat_set)
        results.append({
            'pair_idx': idx, 'method': 'cs_modeB',
            'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
            'cath_relation': cath_rel,
            'n_aligned': n_aln, 'rmsd': rmsd, 'tm_like': tm_like,
            'q_cat_total': conc['q_cat_total'], 't_cat_total': conc['t_cat_total'],
            'q_cat_aligned': conc['q_cat_aligned'], 't_cat_aligned': conc['t_cat_aligned'],
            'cat_to_cat': conc['cat_to_cat'],
            'concordance': conc['concordance'],
            'recall': conc['recall'], 'precision': conc['precision'],
            'error': '',
        })
    except Exception as e:
        results.append({
            'pair_idx': idx, 'method': 'cs_modeB',
            'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
            'cath_relation': cath_rel,
            'n_aligned': 0, 'rmsd': 0, 'tm_like': 0,
            'q_cat_total': len(q_cat_set), 't_cat_total': len(t_cat_set),
            'q_cat_aligned': 0, 't_cat_aligned': 0, 'cat_to_cat': 0,
            'concordance': 0, 'recall': 0, 'precision': 0,
            'error': str(e)[:100],
        })
    
    # --- CS-TOURIST Mode C (catalytic-only) ---
    try:
        pairs, rmsd, n_aln, tm_like, info = run_cs_tourist_mode_c(
            trace_a, trace_b, q_cat_set, t_cat_set,
            descriptor_radius=8.0, refined=True, beta=0.5,
            dist_constraint_weight=2.0)
        conc = compute_concordance(pairs, q_cat_set, t_cat_set)
        results.append({
            'pair_idx': idx, 'method': 'cs_modeC',
            'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
            'cath_relation': cath_rel,
            'n_aligned': n_aln, 'rmsd': rmsd, 'tm_like': tm_like,
            'q_cat_total': conc['q_cat_total'], 't_cat_total': conc['t_cat_total'],
            'q_cat_aligned': conc['q_cat_aligned'], 't_cat_aligned': conc['t_cat_aligned'],
            'cat_to_cat': conc['cat_to_cat'],
            'concordance': conc['concordance'],
            'recall': conc['recall'], 'precision': conc['precision'],
            'error': '',
        })
    except Exception as e:
        results.append({
            'pair_idx': idx, 'method': 'cs_modeC',
            'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
            'cath_relation': cath_rel,
            'n_aligned': 0, 'rmsd': 0, 'tm_like': 0,
            'q_cat_total': len(q_cat_set), 't_cat_total': len(t_cat_set),
            'q_cat_aligned': 0, 't_cat_aligned': 0, 'cat_to_cat': 0,
            'concordance': 0, 'recall': 0, 'precision': 0,
            'error': str(e)[:100],
        })

    # --- CS-TOURIST Mode C with connectivity-aware descriptor ---
    if _g.get('use_connectivity', False):
        try:
            pairs, rmsd, n_aln, tm_like, info = run_cs_tourist_mode_c(
                trace_a, trace_b, q_cat_set, t_cat_set,
                descriptor_radius=8.0, refined=True, beta=0.5,
                dist_constraint_weight=2.0,
                use_connectivity=True,
                w_conn=_g.get('w_conn', 1.0))
            conc = compute_concordance(pairs, q_cat_set, t_cat_set)
            results.append({
                'pair_idx': idx, 'method': 'cs_modeC_conn',
                'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
                'cath_relation': cath_rel,
                'n_aligned': n_aln, 'rmsd': rmsd, 'tm_like': tm_like,
                'q_cat_total': conc['q_cat_total'], 't_cat_total': conc['t_cat_total'],
                'q_cat_aligned': conc['q_cat_aligned'], 't_cat_aligned': conc['t_cat_aligned'],
                'cat_to_cat': conc['cat_to_cat'],
                'concordance': conc['concordance'],
                'recall': conc['recall'], 'precision': conc['precision'],
                'error': '',
            })
        except Exception as e:
            results.append({
                'pair_idx': idx, 'method': 'cs_modeC_conn',
                'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
                'cath_relation': cath_rel,
                'n_aligned': 0, 'rmsd': 0, 'tm_like': 0,
                'q_cat_total': len(q_cat_set), 't_cat_total': len(t_cat_set),
                'q_cat_aligned': 0, 't_cat_aligned': 0, 'cat_to_cat': 0,
                'concordance': 0, 'recall': 0, 'precision': 0,
                'error': str(e)[:100],
            })
    
    # --- TM-align ---
    tm_corr, tm_score, tm_rmsd, tm_err = run_tmalign(q_id, t_id, q_chain, t_chain)
    tm_conc = compute_concordance(tm_corr, q_cat_set, t_cat_set)
    results.append({
        'pair_idx': idx, 'method': 'tmalign',
        'query': q_id, 'target': t_id, 'q_chain': q_chain, 't_chain': t_chain,
        'cath_relation': cath_rel,
        'n_aligned': len(tm_corr), 'rmsd': tm_rmsd, 'tm_like': tm_score,
        'q_cat_total': tm_conc['q_cat_total'], 't_cat_total': tm_conc['t_cat_total'],
        'q_cat_aligned': tm_conc['q_cat_aligned'], 't_cat_aligned': tm_conc['t_cat_aligned'],
        'cat_to_cat': tm_conc['cat_to_cat'],
        'concordance': tm_conc['concordance'],
        'recall': tm_conc['recall'], 'precision': tm_conc['precision'],
        'error': tm_err,
    })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='CS-TOURIST evaluation on 1000 CSA pairs')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=1000)
    parser.add_argument('--use-connectivity', action='store_true',
                        help='Enable connectivity-aware Mode C (cs_modeC_conn)')
    parser.add_argument('--w-conn', type=float, default=1.0,
                        help='Connectivity channel weight for dual-channel Mode C')
    args = parser.parse_args()

    init_worker(use_connectivity=args.use_connectivity, w_conn=args.w_conn)
    pairs = _g['pairs']
    
    start = max(0, args.start)
    end = min(len(pairs), args.end)
    subset = [(i, pairs[i]) for i in range(start, end)]
    
    print(f"Running CS-TOURIST evaluation on {len(subset)} pairs ({start}-{end})", flush=True)
    print(f"Methods: tourist_full, cs_modeA_r6, cs_modeA_r8, cs_modeA_r10, cs_modeB, cs_modeC, tmalign", flush=True)
    
    t0 = time.time()
    all_results = []
    
    if args.workers <= 1:
        for i, task in enumerate(subset):
            results = process_pair(task)
            all_results.extend(results)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(subset) - i - 1) / rate
                print(f"  {i+1}/{len(subset)} ({elapsed:.0f}s, {rate:.1f} pairs/s, ETA {eta:.0f}s)", flush=True)
    else:
        with Pool(args.workers, initializer=init_worker,
                  initargs=(args.use_connectivity, args.w_conn)) as pool:
            for i, results in enumerate(pool.imap(process_pair, subset, chunksize=10)):
                all_results.extend(results)
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(subset) - i - 1) / rate
                    print(f"  {i+1}/{len(subset)} ({elapsed:.0f}s, {rate:.1f} pairs/s, ETA {eta:.0f}s)", flush=True)
    
    elapsed = time.time() - t0
    print(f"\nDone: {len(all_results)} results in {elapsed:.1f}s", flush=True)
    
    # Save detailed results
    suffix = f'_{start}_{end}' if start > 0 or end < 1000 else ''
    if args.use_connectivity:
        suffix += '_v11conn'
    detail_file = os.path.join(OUTPUT_DIR, f'cs_tourist_eval{suffix}.csv')
    fieldnames = ['pair_idx', 'method', 'query', 'target', 'q_chain', 't_chain',
                  'cath_relation', 'n_aligned', 'rmsd', 'tm_like',
                  'q_cat_total', 't_cat_total', 'q_cat_aligned', 't_cat_aligned',
                  'cat_to_cat', 'concordance', 'recall', 'precision', 'error']
    
    # Write to /workspace first, then copy
    tmp_path = f'/workspace/cs_tourist_eval{suffix}.csv'
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    os.system(f'cp {tmp_path} {detail_file}')
    print(f"Saved detailed results to {detail_file}", flush=True)
    
    # Summary by method
    by_method = defaultdict(list)
    for r in all_results:
        if not r['error']:
            by_method[r['method']].append(r)
    
    print(f"\n{'='*120}", flush=True)
    print(f"CS-TOURIST EVALUATION SUMMARY", flush=True)
    print(f"{'='*120}", flush=True)
    print(f"{'Method':<20} {'N':>6} {'Mean conc':>10} {'Mean recall':>12} {'Mean prec':>10} "
          f"{'Mean n_aln':>11} {'Mean tm':>10}", flush=True)
    print(f"{'-'*20} {'-'*6} {'-'*10} {'-'*12} {'-'*10} {'-'*11} {'-'*10}", flush=True)
    
    summary_rows = []
    for method in sorted(by_method.keys()):
        rows = by_method[method]
        n = len(rows)
        mean_conc = np.mean([r['concordance'] for r in rows])
        mean_recall = np.mean([r['recall'] for r in rows])
        mean_prec = np.mean([r['precision'] for r in rows])
        mean_aln = np.mean([r['n_aligned'] for r in rows])
        mean_tm = np.mean([r['tm_like'] for r in rows])
        print(f"{method:<20} {n:>6} {mean_conc:>10.4f} {mean_recall:>12.4f} "
              f"{mean_prec:>10.4f} {mean_aln:>11.1f} {mean_tm:>10.4f}", flush=True)
        summary_rows.append({
            'method': method, 'n_pairs': n,
            'mean_concordance': mean_conc,
            'mean_recall': mean_recall,
            'mean_precision': mean_prec,
            'mean_n_aligned': mean_aln,
            'mean_tm_like': mean_tm,
        })
    
    # By CATH relationship
    for rel in ['same_h', 'same_t_diff_h', 'diff_t']:
        rel_rows = [r for r in all_results if r['cath_relation'] == rel and not r['error']]
        if not rel_rows:
            continue
        by_m = defaultdict(list)
        for r in rel_rows:
            by_m[r['method']].append(r)
        print(f"\n--- {rel} ({len(set(r['pair_idx'] for r in rel_rows))} pairs) ---", flush=True)
        print(f"{'Method':<20} {'N':>6} {'Mean conc':>10} {'Mean recall':>12} {'Mean prec':>10}", flush=True)
        for method in sorted(by_m.keys()):
            rows = by_m[method]
            n = len(rows)
            mean_conc = np.mean([r['concordance'] for r in rows])
            mean_recall = np.mean([r['recall'] for r in rows])
            mean_prec = np.mean([r['precision'] for r in rows])
            print(f"{method:<20} {n:>6} {mean_conc:>10.4f} {mean_recall:>12.4f} {mean_prec:>10.4f}", flush=True)
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, f'cs_tourist_summary{suffix}.csv')
    tmp_sum = f'/workspace/cs_tourist_summary{suffix}.csv'
    with open(tmp_sum, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'n_pairs', 'mean_concordance',
                                               'mean_recall', 'mean_precision',
                                               'mean_n_aligned', 'mean_tm_like'])
        writer.writeheader()
        writer.writerows(summary_rows)
    os.system(f'cp {tmp_sum} {summary_file}')
    print(f"\nSaved summary to {summary_file}", flush=True)

if __name__ == '__main__':
    main()
