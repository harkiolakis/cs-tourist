#!/usr/bin/env python3
"""
Benchmark pyScoMotif against TOURIST on catalytic site detection task.
Uses Python API directly to avoid subprocess overhead.

For each unique query (PDB + chain) with >=3 catalytic residues present in the structure:
  1. Get catalytic residues from CSA
  2. Filter to residues actually present in the PDB file
  3. Search pyScoMotif index for similar motifs (strict policy)
  4. Calculate RMSD and write to temp CSV
  5. For each pair involving this query, filter results to target PDB
  6. Parse matched residues as predicted catalytic residues
  7. Compare to target's known catalytic residues

Metrics (standard ML definitions):
  Recall = |predicted ∩ known| / |known|
  Precision = |predicted ∩ known| / |predicted|
  F1 = 2 * P * R / (P + R)
"""

import json
import os
import re
import sys
import time
import tempfile
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Import pyScoMotif API directly
from pyscomotif.motif_search import search_index_for_PDBs_with_similar_motifs
from pyscomotif.RMSD_calculation import calculate_RMSD_between_motif_and_similar_motifs

# Paths
CSA_PAIRS = "/mnt/shared-workspace/shared/csa_pairs_1000.csv"
CSA_SITES = "/mnt/shared-workspace/shared/csa_sites.json"
PDB_DIR = "/mnt/shared-workspace/pdb_csa/"
INDEX_DIR = "/workspace/pyscom_index"
OUTPUT_DIR = "/mnt/results/benchmark/data"

# Parameters
RMSD_THRESHOLD = 3.0
RMSD_ATOMS = "CA+sidechain"  # pyScoMotif default
N_CORES = 8
DISTANCE_DELTA = 2.0  # default
ANGLE_DELTA = 30.0  # default
POLICY = os.environ.get("PYSCOM_POLICY", "strict")
MAX_MUTATIONS = int(os.environ.get("PYSCOM_MAX_MUT", "1"))
OUTPUT_SUFFIX = os.environ.get("PYSCOM_SUFFIX", "")


def parse_similar_motif(motif_str):
    """Parse pyScoMotif similar_motif_found string.
    Format: 'A7D A70C A178C A180H A8S A147E'
    Each token: chain(1) + resseq(digits) + aa(1 letter)
    Returns list of (chain, resseq, aa) tuples.
    """
    residues = []
    for token in motif_str.split():
        m = re.match(r'^([A-Za-z])(\d+)([A-Z])$', token)
        if m:
            chain = m.group(1)
            resseq = int(m.group(2))
            aa = m.group(3)
            residues.append((chain, resseq, aa))
    return residues


def compute_f1(precision, recall):
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def get_available_residues(pdb_path, chain):
    """Get set of residue numbers present in the PDB file for a given chain."""
    available = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and len(line) > 26 and line[21] == chain:
                try:
                    resseq = int(line[22:26])
                    available.add(resseq)
                except ValueError:
                    pass
    return available


def main():
    # Load data
    pairs_df = pd.read_csv(CSA_PAIRS)
    with open(CSA_SITES) as f:
        csa_sites = json.load(f)

    # Group pairs by (query_id, query_chain)
    query_groups = defaultdict(list)
    for _, row in pairs_df.iterrows():
        key = (row['query_id'], row['query_chain'])
        query_groups[key].append(row)

    print(f"Total pairs: {len(pairs_df)}")
    print(f"Unique (query_id, query_chain): {len(query_groups)}")

    # Filter queries: need >=3 catalytic residues present in PDB
    valid_queries = {}
    skipped_lt3 = 0
    skipped_missing = 0
    for (qid, qchain), pair_list in query_groups.items():
        qcat = csa_sites.get(qid, {}).get(qchain, [])
        if len(qcat) < 3:
            skipped_lt3 += len(pair_list)
            continue

        pdb_path = os.path.join(PDB_DIR, f"{qid}.pdb")
        if not os.path.exists(pdb_path):
            skipped_missing += len(pair_list)
            continue

        # Check which residues are present
        available = get_available_residues(pdb_path, qchain)
        present_cat = [r for r in qcat if r in available]

        if len(present_cat) < 3:
            skipped_missing += len(pair_list)
            continue

        # Store with the filtered motif
        valid_queries[(qid, qchain)] = (pair_list, present_cat)

    total_valid_pairs = sum(len(v[0]) for v in valid_queries.values())
    print(f"Valid queries (>=3 residues present): {len(valid_queries)}")
    print(f"Pairs skipped (<3 cat total): {skipped_lt3}")
    print(f"Pairs skipped (missing residues): {skipped_missing}")
    print(f"Pairs to process: {total_valid_pairs}")
    print(f"Policy: {POLICY}, RMSD threshold: {RMSD_THRESHOLD}, atoms: {RMSD_ATOMS}")
    print(f"N_cores: {N_CORES}")
    print()

    results = []
    n_processed = 0
    n_errors = 0
    n_no_hits = 0
    start_time = time.time()

    for (qid, qchain), (pair_list, present_cat) in valid_queries.items():
        # Format motif: chain + resseq (e.g., "A7", "A70", "A178")
        motif = tuple(f"{qchain}{r}" for r in present_cat)

        pdb_path = Path(os.path.join(PDB_DIR, f"{qid}.pdb"))

        query_start = time.time()

        # Search index
        try:
            motif_MST, PDBs_with_similar_motifs = search_index_for_PDBs_with_similar_motifs(
                index_folder_path=Path(INDEX_DIR),
                PDB_file=pdb_path,
                motif=motif,
                residue_type_policy=POLICY,
                max_n_mutated_residues=MAX_MUTATIONS,
                distance_delta_thr=DISTANCE_DELTA,
                angle_delta_thr=ANGLE_DELTA,
                n_cores=N_CORES
            )
        except Exception as e:
            print(f"  ERROR searching {qid} chain {qchain}: {e}")
            n_errors += 1
            n_processed += 1
            continue

        # Check if any hits
        total_hits = sum(len(pdbs) for solutions in PDBs_with_similar_motifs.values() for pdbs in solutions.values())
        if total_hits == 0:
            n_no_hits += 1

        # Calculate RMSD and write to temp CSV
        temp_csv = tempfile.mktemp(suffix=".csv")
        try:
            calculate_RMSD_between_motif_and_similar_motifs(
                motif_MST=motif_MST,
                PDB_file=pdb_path,
                PDBs_with_similar_motifs=PDBs_with_similar_motifs,
                index_folder_path=Path(INDEX_DIR),
                RMSD_atoms=RMSD_ATOMS,
                RMSD_threshold=RMSD_THRESHOLD,
                n_cores=N_CORES,
                results_output_path=Path(temp_csv),
                sort_results=True
            )
        except Exception as e:
            print(f"  ERROR calculating RMSD for {qid}: {e}")
            n_errors += 1
            n_processed += 1
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            continue

        query_elapsed = time.time() - query_start

        # Read results (handle the Unnamed: 0 index column)
        try:
            search_df = pd.read_csv(temp_csv, index_col=0)
        except Exception:
            search_df = pd.DataFrame()
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

        # For each pair involving this query, find matches for the target
        for pair_row in pair_list:
            tid = pair_row['target_id']
            tchain = pair_row['target_chain']
            pair_idx = pair_row['pair_idx']
            cath_rel = pair_row['cath_relation']

            # Filter results to target PDB (PDB_ID is lowercase in results)
            if len(search_df) > 0:
                target_matches = search_df[search_df['PDB_ID'] == tid.lower()]
            else:
                target_matches = pd.DataFrame()

            if len(target_matches) == 0:
                # No match found
                results.append({
                    'pair_idx': pair_idx,
                    'method': f'pyscomotif_{POLICY}',
                    'query': qid,
                    'target': tid,
                    'q_chain': qchain,
                    't_chain': tchain,
                    'cath_relation': cath_rel,
                    'cat_to_cat': 0,
                    'q_cat_aligned': 0,
                    't_cat_total': pair_row['target_n_cat'],
                    'recall': 0.0,
                    'precision': 0.0,
                    'f1': 0.0,
                    'rmsd': None,
                    'n_mutations': None,
                    'runtime_sec': query_elapsed / len(pair_list),
                })
                continue

            # Prefer matches with residues on the target chain
            best_match = None
            best_rmsd = float('inf')

            for _, match_row in target_matches.iterrows():
                motif_found = str(match_row['similar_motif_found'])
                residues = parse_similar_motif(motif_found)

                # Check if any residues are on the target chain
                on_target_chain = [r for r in residues if r[0] == tchain]

                if on_target_chain:
                    rmsd = float(match_row['RMSD'])
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_match = match_row

            # If no match on target chain, use overall best (already sorted by RMSD)
            if best_match is None:
                best_match = target_matches.iloc[0]
                best_rmsd = float(best_match['RMSD'])

            # Parse best match residues
            motif_found = str(best_match['similar_motif_found'])
            residues = parse_similar_motif(motif_found)

            # Get predicted residues on target chain (or all if none on target chain)
            predicted_on_target = [(r[1], r[2]) for r in residues if r[0] == tchain]
            if not predicted_on_target:
                predicted_on_target = [(r[1], r[2]) for r in residues]

            predicted_resseqs = set(r[0] for r in predicted_on_target)

            # Get target's known catalytic residues
            tcat = set(csa_sites.get(tid, {}).get(tchain, []))

            # Compute metrics (standard ML definitions)
            cat_to_cat = len(predicted_resseqs & tcat)
            q_cat_aligned = len(predicted_resseqs)  # all predicted residues
            t_cat_total = len(tcat) if len(tcat) > 0 else pair_row['target_n_cat']

            recall = cat_to_cat / t_cat_total if t_cat_total > 0 else 0.0
            precision = cat_to_cat / q_cat_aligned if q_cat_aligned > 0 else 0.0
            f1 = compute_f1(precision, recall)

            results.append({
                'pair_idx': pair_idx,
                'method': f'pyscomotif_{POLICY}',
                'query': qid,
                'target': tid,
                'q_chain': qchain,
                't_chain': tchain,
                'cath_relation': cath_rel,
                'cat_to_cat': cat_to_cat,
                'q_cat_aligned': q_cat_aligned,
                't_cat_total': t_cat_total,
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'rmsd': best_rmsd,
                'n_mutations': int(best_match['n_mutations']),
                'runtime_sec': query_elapsed / len(pair_list),
            })

        n_processed += 1
        if n_processed % 10 == 0:
            elapsed = time.time() - start_time
            rate = n_processed / elapsed
            remaining = (len(valid_queries) - n_processed) / rate
            print(f"  Processed {n_processed}/{len(valid_queries)} queries "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                  f"{n_errors} errors, {n_no_hits} no-hit queries)")

    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, f'competitor_pyscomotif{OUTPUT_SUFFIX}.csv')
    results_df.to_csv(output_path, index=False)

    total_time = time.time() - start_time
    print(f"\nSaved {len(results_df)} results to {output_path}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Errors: {n_errors}, No-hit queries: {n_no_hits}")

    # Quick summary
    if len(results_df) > 0:
        print(f"\nQuick summary ({POLICY} policy):")
        print(f"  Mean recall: {results_df['recall'].mean():.4f}")
        print(f"  Mean precision: {results_df['precision'].mean():.4f}")
        print(f"  Mean F1: {results_df['f1'].mean():.4f}")
        print(f"  Zero recall: {(results_df['recall'] == 0).sum()}/{len(results_df)} "
              f"({(results_df['recall'] == 0).mean()*100:.1f}%)")
        print(f"  Perfect recall: {(results_df['recall'] == 1.0).sum()}/{len(results_df)} "
              f"({(results_df['recall'] == 1.0).mean()*100:.1f}%)")

        # By CATH relation
        for cath in ['same_h', 'same_t_diff_h', 'diff_t']:
            cd = results_df[results_df['cath_relation'] == cath]
            if len(cd) > 0:
                print(f"  {cath}: recall={cd['recall'].mean():.4f}, "
                      f"precision={cd['precision'].mean():.4f}, "
                      f"f1={cd['f1'].mean():.4f}, n={len(cd)}")


if __name__ == '__main__':
    main()
