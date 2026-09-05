#!/usr/bin/env python3
"""
Benchmark GASS-Metal against TOURIST on catalytic site detection task.

For each CSA pair (A, B):
  - Use A's catalytic residues as a template
  - Search for the template in B's structure using GASS genetic algorithm
  - Parse ActiveSitesFound.txt for predicted catalytic residues in B
  - Compare to B's known catalytic residues from CSA

Runs with CA reference atom and no mutations (strict mode).
"""

import json
import os
import re
import subprocess
import sys
import time
import shutil
import pandas as pd
from pathlib import Path

# Paths
CSA_PAIRS = "/mnt/shared-workspace/shared/csa_pairs_1000.csv"
CSA_SITES = "/mnt/shared-workspace/shared/csa_sites.json"
PDB_DIR = "/mnt/shared-workspace/pdb_csa/"
GASS_DIR = "/workspace/gassmetal-local"
OUTPUT_DIR = "/mnt/results/benchmark/data"

def get_residue_names(pdb_path, chain_id, resseqs):
    """Parse PDB file to get 3-letter residue names for given chain and resseq numbers."""
    found = {}
    resseq_set = set(resseqs)
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21].strip()
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    continue
                resname = line[17:20].strip()
                if chain == chain_id and resseq in resseq_set and resseq not in found:
                    found[resseq] = resname
    return found

def build_gass_template(pdb_path, chain_id, resseqs):
    """Build GASS template string: 'RESNAME,POS,CHAIN;RESNAME,POS,CHAIN;...'"""
    names = get_residue_names(pdb_path, chain_id, resseqs)
    parts = []
    for r in resseqs:
        if r in names:
            parts.append(f"{names[r]},{r},{chain_id}")
        else:
            return None  # Can't build template if residue not found
    return ";".join(parts) + ";"

def parse_gass_output(filepath, target_chain=None):
    """Parse ActiveSitesFound.txt to extract best match predicted residues.
    Format (tab-separated):
      col0: run_id, col1: rank, col2: RMSD, col3: found_site,
      col4: pdb_id, col5: template, col6: ec, col7: uniprot, col8: score, col9: n_residues
    found_site format: 'ASP 85 A;SER 82 A;CYS 77 A;...'
    
    If target_chain is given, prefers matches with residues on that chain:
      - First tries to find the best match (lowest RMSD) with >=1 residue on target_chain
      - Falls back to overall best match if none found on target_chain
    
    Returns (predicted_residues_list, rmsd) or ([], None) if no results.
    predicted_residues_list = [(chain, resseq, resname), ...]
    """
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return [], None
    
    all_matches = []
    
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split('\t')
            if len(cols) < 4:
                continue
            try:
                rmsd = float(cols[2])
            except (ValueError, IndexError):
                continue
            
            # Parse found site (col 3): "ASP 85 A;SER 82 A;CYS 77 A;..."
            found_site = cols[3]
            predicted = []
            for residue_entry in found_site.split(';'):
                residue_entry = residue_entry.strip()
                if not residue_entry:
                    continue
                m = re.match(r'^(\w+)\s+(\d+)\s+(\w+)$', residue_entry)
                if m:
                    resname = m.group(1)
                    resseq = int(m.group(2))
                    chain = m.group(3)
                    predicted.append((chain, resseq, resname))
            
            all_matches.append((rmsd, predicted))
    
    if not all_matches:
        return [], None
    
    # Sort by RMSD
    all_matches.sort(key=lambda x: x[0])
    
    # If target_chain specified, prefer matches with residues on that chain
    if target_chain:
        chain_matches = [(r, p) for r, p in all_matches 
                         if any(res[0] == target_chain for res in p)]
        if chain_matches:
            best_rmsd, best_predicted = chain_matches[0]
        else:
            best_rmsd, best_predicted = all_matches[0]
    else:
        best_rmsd, best_predicted = all_matches[0]
    
    return best_predicted, best_rmsd

def compute_metrics(predicted_resseqs, known_resseqs):
    """Compute recall, precision, F1."""
    predicted_set = set(predicted_resseqs)
    known_set = set(known_resseqs)
    
    if len(known_set) == 0:
        return 0.0, 0.0, 0.0, 0
    
    tp = len(predicted_set & known_set)
    recall = tp / len(known_set) if len(known_set) > 0 else 0.0
    precision = tp / len(predicted_set) if len(predicted_set) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return recall, precision, f1, tp

def run_gass(query_pdb, template_str, target_pdb, target_chain=None, ref_atom="CA"):
    """Run GASS-Metal rungass.py and return (predicted_residues, rmsd, runtime_sec, error)."""
    # rungass.py signature: reference_pdb template_site mutations target_pdb reference_atom
    cmd = [
        "python3", os.path.join(GASS_DIR, "rungass.py"),
        query_pdb,
        template_str,
        "",  # no mutations (strict mode)
        target_pdb,
        ref_atom
    ]
    
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=GASS_DIR
        )
        runtime = time.time() - t0
        
        # Parse output, preferring matches on target chain
        active_sites_path = os.path.join(GASS_DIR, "ActiveSitesFound.txt")
        predicted, rmsd = parse_gass_output(active_sites_path, target_chain)
        
        error = ""
        if len(predicted) == 0:
            error = "no_results"
        
        return predicted, rmsd, runtime, error
        
    except subprocess.TimeoutExpired:
        return [], None, time.time() - t0, "timeout"
    except Exception as e:
        return [], None, time.time() - t0, str(e)[:100]

def main():
    # Load data
    pairs = pd.read_csv(CSA_PAIRS)
    with open(CSA_SITES) as f:
        csa_sites = json.load(f)
    
    print(f"Loaded {len(pairs)} CSA pairs")
    
    # Results storage
    all_results = []
    
    # Pre-build templates for all unique query proteins
    print("Pre-building GASS templates...")
    template_cache = {}
    for _, row in pairs.iterrows():
        query_id = row['query_id']
        query_chain = row['query_chain']
        key = f"{query_id}_{query_chain}"
        
        if key in template_cache:
            continue
        
        if query_id not in csa_sites or query_chain not in csa_sites[query_id]:
            template_cache[key] = None
            continue
        
        cat_resseqs = csa_sites[query_id][query_chain]
        pdb_path = os.path.join(PDB_DIR, f"{query_id}.pdb")
        
        if not os.path.exists(pdb_path):
            template_cache[key] = None
            continue
        
        template = build_gass_template(pdb_path, query_chain, cat_resseqs)
        template_cache[key] = template
        
        if template is None:
            print(f"  WARNING: Could not build template for {key}")
    
    print(f"  Built {sum(1 for v in template_cache.values() if v is not None)} templates")
    
    # Run GASS for each pair
    print("\nRunning GASS benchmark...")
    for idx, row in pairs.iterrows():
        pair_idx = row['pair_idx']
        query_id = row['query_id']
        query_chain = row['query_chain']
        target_id = row['target_id']
        target_chain = row['target_chain']
        cath_rel = row['cath_relation']
        q_cat_total = row['query_n_cat']
        t_cat_total = row['target_n_cat']
        
        q_key = f"{query_id}_{query_chain}"
        template = template_cache.get(q_key)
        
        if template is None:
            all_results.append({
                'pair_idx': pair_idx, 'method': 'gass',
                'query': query_id, 'target': target_id,
                'q_chain': query_chain, 't_chain': target_chain,
                'cath_relation': cath_rel,
                'q_cat_total': q_cat_total, 't_cat_total': t_cat_total,
                'predicted_residues': '', 'known_residues': '',
                'tp': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0,
                'rmsd': None, 'runtime_sec': 0.0, 'error': 'no_template'
            })
            continue
        
        target_pdb = os.path.join(PDB_DIR, f"{target_id}.pdb")
        query_pdb = os.path.join(PDB_DIR, f"{query_id}.pdb")
        
        if not os.path.exists(target_pdb) or not os.path.exists(query_pdb):
            all_results.append({
                'pair_idx': pair_idx, 'method': 'gass',
                'query': query_id, 'target': target_id,
                'q_chain': query_chain, 't_chain': target_chain,
                'cath_relation': cath_rel,
                'q_cat_total': q_cat_total, 't_cat_total': t_cat_total,
                'predicted_residues': '', 'known_residues': '',
                'tp': 0, 'recall': 0.0, 'precision': 0.0, 'f1': 0.0,
                'rmsd': None, 'runtime_sec': 0.0, 'error': 'missing_pdb'
            })
            continue
        
        # Get known catalytic residues in target
        known_resseqs = csa_sites.get(target_id, {}).get(target_chain, [])
        
        # Run GASS
        predicted, rmsd, runtime, error = run_gass(query_pdb, template, target_pdb, target_chain)
        
        # Filter predicted to target chain
        predicted_resseqs = [r[1] for r in predicted if r[0] == target_chain]
        
        # Compute metrics
        recall, precision, f1, tp = compute_metrics(predicted_resseqs, known_resseqs)
        
        all_results.append({
            'pair_idx': pair_idx, 'method': 'gass',
            'query': query_id, 'target': target_id,
            'q_chain': query_chain, 't_chain': target_chain,
            'cath_relation': cath_rel,
            'q_cat_total': q_cat_total, 't_cat_total': t_cat_total,
            'predicted_residues': str(predicted_resseqs),
            'known_residues': str(known_resseqs),
            'tp': tp, 'recall': recall, 'precision': precision, 'f1': f1,
            'rmsd': rmsd, 'runtime_sec': runtime, 'error': error
        })
        
        if (idx + 1) % 100 == 0:
            done = len(all_results)
            avg_recall = np.mean([r['recall'] for r in all_results[-100:]])
            avg_time = np.mean([r['runtime_sec'] for r in all_results[-100:]])
            print(f"  {done}/{len(pairs)} pairs done | recent avg recall={avg_recall:.3f} | avg time={avg_time:.2f}s")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(OUTPUT_DIR, "competitor_gass.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} results to {output_path}")
    
    # Print summary
    md = results_df
    print(f"\nGASS summary:")
    print(f"  N pairs: {len(md)}")
    print(f"  Mean recall: {md['recall'].mean():.4f}")
    print(f"  Mean precision: {md['precision'].mean():.4f}")
    print(f"  Mean F1: {md['f1'].mean():.4f}")
    print(f"  Mean runtime: {md['runtime_sec'].mean():.2f}s")
    for cath in ['same_h', 'same_t_diff_h', 'diff_t']:
        cd = md[md['cath_relation'] == cath]
        if len(cd) > 0:
            print(f"  {cath}: recall={cd['recall'].mean():.4f}, precision={cd['precision'].mean():.4f}, F1={cd['f1'].mean():.4f}")

if __name__ == "__main__":
    import numpy as np
    main()
