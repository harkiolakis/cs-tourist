#!/usr/bin/env python3
"""Negative control: decoy residue experiment for CS-TOURIST Mode C.

For each enzyme pair, replaces the target's catalytic residues with:
  (a) Random non-catalytic residues (any position)
  (b) Surface-exposed non-catalytic residues (proxy: low Cα neighbor count)

Runs CS-TOURIST Mode C with real query catalytic residues + decoy target residues.
Compares RMSD to the real catalytic-vs-catalytic RMSD.

If the method is just "aligning any two labelled sets," decoy RMSD ≈ real RMSD.
If the method detects genuine geometric similarity, decoy RMSD >> real RMSD.

Usage:
    python decoy_experiment.py [--n_decoys 5] [--start 0] [--end 924] [--workers 4]
"""

import argparse
import csv
import json
import os
import sys
import time
import random
import numpy as np
from multiprocessing import Pool

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
from cs_tourist import run_cs_tourist_mode_c

# Paths
PDB_BENCHMARK = '/mnt/shared-workspace/pdb_benchmark'
PDB_CSA = '/mnt/shared-workspace/pdb_csa'
CSA_SITES = '/mnt/shared-workspace/shared/csa_sites.json'
EVAL_CSV = '/mnt/results/benchmark/data/cs_tourist_eval_v11conn.csv'
OUTPUT_DIR = '/mnt/results/benchmark/data'

# Global state for workers
_g = {}


def init_worker():
    """Initialize worker process."""
    warmup()
    with open(CSA_SITES) as f:
        _g['csa'] = json.load(f)


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


def compute_ca_neighbor_counts(coords, cutoff=10.0):
    """Compute Cα neighbor count for each residue (proxy for solvent accessibility).
    Lower neighbor count = more surface-exposed."""
    n = len(coords)
    if n == 0:
        return np.zeros(0)
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, 1e10)
    counts = np.sum(dists < cutoff, axis=1)
    return counts


def select_decoy_residues(n_decoy, n_total, exclude_set, mode='random', 
                          neighbor_counts=None, rng=None):
    """Select decoy residues.
    
    mode='random': any non-excluded residue
    mode='surface': from the top 50% most surface-exposed non-excluded residues
    """
    if rng is None:
        rng = random.Random()
    
    available = [i for i in range(n_total) if i not in exclude_set]
    if len(available) < n_decoy:
        # Not enough non-catalytic residues; use what we have
        return set(available[:n_decoy])
    
    if mode == 'surface' and neighbor_counts is not None:
        # Sort available by neighbor count (ascending = more surface)
        available_sorted = sorted(available, key=lambda i: neighbor_counts[i])
        # Take top 50% most surface-exposed
        n_surface = max(n_decoy, len(available_sorted) // 2)
        surface_pool = available_sorted[:n_surface]
        selected = rng.sample(surface_pool, min(n_decoy, len(surface_pool)))
    else:
        selected = rng.sample(available, n_decoy)
    
    return set(selected)


def process_decoy_pair(task):
    """Process a single pair with decoy residues."""
    idx, p, n_decoys = task
    
    q_id = p['query']
    t_id = p['target']
    q_chain = p['q_chain']
    t_chain = p['t_chain']
    cath_rel = p['cath_relation']
    real_rmsd = float(p['rmsd']) if p['rmsd'] else 0.0
    real_n_aligned = int(p['n_aligned'])
    
    csa = _g['csa']
    results = []
    
    # Get catalytic sites
    # csa_sites.json format: {pdb_id: {chain: [residue_numbers]}}
    q_cat_list = csa.get(q_id, {}).get(q_chain, [])
    t_cat_list = csa.get(t_id, {}).get(t_chain, [])
    
    # Handle case where annotations might be in nested format
    if isinstance(q_cat_list, dict):
        q_cat_list = q_cat_list.get('catalytic', [])
    if isinstance(t_cat_list, dict):
        t_cat_list = t_cat_list.get('catalytic', [])
    
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
    except Exception:
        return results
    
    # Convert catalytic sites to trace indices
    q_resseq_map = build_resseq_map(trace_a)
    t_resseq_map = build_resseq_map(trace_b)
    q_cat_set = convert_sites_to_trace_idx(q_cat_list, q_resseq_map)
    t_cat_set = convert_sites_to_trace_idx(t_cat_list, t_resseq_map)
    
    if len(q_cat_set) < 2 or len(t_cat_set) < 2:
        return results
    
    n_target = len(trace_b.coords)
    n_decoy_needed = len(t_cat_set)
    
    # Compute neighbor counts for surface selection
    t_neighbor_counts = compute_ca_neighbor_counts(trace_b.coords)
    
    # Run multiple decoy trials
    for trial in range(n_decoys):
        rng = random.Random(idx * 1000 + trial)
        
        for decoy_mode in ['random', 'surface']:
            decoy_set = select_decoy_residues(
                n_decoy_needed, n_target, t_cat_set, 
                mode=decoy_mode, neighbor_counts=t_neighbor_counts, rng=rng)
            
            if len(decoy_set) < 2:
                continue
            
            try:
                pairs, rmsd, n_aln, tm_like, info = run_cs_tourist_mode_c(
                    trace_a, trace_b, q_cat_set, decoy_set,
                    descriptor_radius=8.0, refined=True, beta=0.5,
                    dist_constraint_weight=2.0,
                    use_connectivity=True, w_conn=1.0)
                
                results.append({
                    'pair_idx': idx,
                    'query': q_id, 'target': t_id,
                    'q_chain': q_chain, 't_chain': t_chain,
                    'cath_relation': cath_rel,
                    'real_rmsd': real_rmsd,
                    'real_n_aligned': real_n_aligned,
                    'q_cat_total': len(q_cat_set),
                    't_cat_total': len(t_cat_set),
                    'decoy_mode': decoy_mode,
                    'trial': trial,
                    'decoy_rmsd': rmsd,
                    'decoy_n_aligned': n_aln,
                    'decoy_tm_like': tm_like,
                    'error': info.get('error', '') if isinstance(info, dict) else '',
                })
            except Exception as e:
                results.append({
                    'pair_idx': idx,
                    'query': q_id, 'target': t_id,
                    'q_chain': q_chain, 't_chain': t_chain,
                    'cath_relation': cath_rel,
                    'real_rmsd': real_rmsd,
                    'real_n_aligned': real_n_aligned,
                    'q_cat_total': len(q_cat_set),
                    't_cat_total': len(t_cat_set),
                    'decoy_mode': decoy_mode,
                    'trial': trial,
                    'decoy_rmsd': 0,
                    'decoy_n_aligned': 0,
                    'decoy_tm_like': 0,
                    'error': str(e)[:100],
                })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_decoys', type=int, default=5, 
                        help='Number of decoy trials per pair')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=924)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    # Load the real evaluation results (cs_modeC_conn only)
    pairs = []
    with open(EVAL_CSV) as f:
        for row in csv.DictReader(f):
            if row['method'] == 'cs_modeC_conn':
                pairs.append(row)
    
    print(f"Loaded {len(pairs)} cs_modeC_conn pairs")
    
    # Filter to requested range
    pairs = pairs[args.start:args.end]
    print(f"Processing {len(pairs)} pairs (range {args.start}-{args.end})")
    
    # Create tasks
    tasks = [(i, p, args.n_decoys) for i, p in enumerate(pairs)]
    
    # Process with multiprocessing
    all_results = []
    t0 = time.time()
    
    with Pool(args.workers, initializer=init_worker) as pool:
        for i, results in enumerate(pool.imap_unordered(process_decoy_pair, tasks)):
            all_results.extend(results)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(pairs) - i - 1) / rate
                print(f"  {i+1}/{len(pairs)} pairs done, "
                      f"{len(all_results)} results, "
                      f"{rate:.1f} pairs/s, ETA {eta:.0f}s")
    
    elapsed = time.time() - t0
    print(f"\nDone: {len(all_results)} results in {elapsed:.1f}s")
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'decoy_experiment_results.csv')
    if all_results:
        fields = list(all_results[0].keys())
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Saved to {output_path}")
    
    # Quick summary
    if all_results:
        real_rmsds = [r['real_rmsd'] for r in all_results if r['real_rmsd'] > 0]
        decoy_random = [r['decoy_rmsd'] for r in all_results 
                       if r['decoy_mode'] == 'random' and r['decoy_rmsd'] > 0]
        decoy_surface = [r['decoy_rmsd'] for r in all_results 
                        if r['decoy_mode'] == 'surface' and r['decoy_rmsd'] > 0]
        
        print(f"\n=== Summary ===")
        print(f"Real catalytic RMSD: median={np.median(real_rmsds):.2f} Å, "
              f"mean={np.mean(real_rmsds):.2f} Å")
        print(f"Decoy (random) RMSD: median={np.median(decoy_random):.2f} Å, "
              f"mean={np.mean(decoy_random):.2f} Å")
        print(f"Decoy (surface) RMSD: median={np.median(decoy_surface):.2f} Å, "
              f"mean={np.mean(decoy_surface):.2f} Å")
        print(f"Ratio (decoy_random/real): {np.median(decoy_random)/np.median(real_rmsds):.2f}x")
        print(f"Ratio (decoy_surface/real): {np.median(decoy_surface)/np.median(real_rmsds):.2f}x")


if __name__ == '__main__':
    main()
