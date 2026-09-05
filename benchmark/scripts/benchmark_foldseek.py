#!/usr/bin/env python3
"""Run Foldseek on all benchmark pairs using a pre-created target database.

Creates a Foldseek database from all unique target PDB files once,
then searches each query against it with --exact-tmscore 1.
"""

import csv
import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

FOLDSEEK = '/workspace/foldseek/bin/foldseek'
PDB_DIR = '/workspace/pdb_cache'  # Local cache on worker-0
PAIR_LIST = '/mnt/shared-workspace/pair_list.csv'
OUTPUT = '/mnt/shared-workspace/results_foldseek.csv'
WORK_DIR = '/workspace/foldseek_bench'

FORMAT_STRING = "query,target,qtmscore,ttmscore,alntmscore,rmsd,lddt,alnlen,evalue,qcov,tcov"


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    # Load pairs
    with open(PAIR_LIST) as f:
        reader = csv.DictReader(f)
        pairs = list(reader)

    queries = sorted(set((p['query_id'], p['query_chain']) for p in pairs))
    targets = sorted(set(p['target_id'].lower() for p in pairs))
    print(f"Queries: {len(queries)}, Target PDBs: {len(targets)}")

    # ── Step 1: Create target database (once) ──────────────────────────
    target_dir = os.path.join(WORK_DIR, 'target_pdbs')
    os.makedirs(target_dir, exist_ok=True)

    print("Symlinking target PDB files...")
    for t_id in targets:
        src = os.path.join(PDB_DIR, f'{t_id}.pdb')
        dst = os.path.join(target_dir, f'{t_id}.pdb')
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    db_path = os.path.join(WORK_DIR, 'target_db')
    print("Creating Foldseek target database...")
    cmd = [FOLDSEEK, 'createdb', target_dir, db_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"  createdb stderr: {proc.stderr[-500:]}")
        sys.exit(1)
    # Parse how many were ignored
    for line in proc.stderr.split('\n'):
        if 'Ignore' in line or 'created' in line.lower():
            print(f"  {line.strip()}")
    print("  Target database created")

    # ── Step 2: Search each query ───────────────────────────────────────
    all_results = []
    t0 = time.time()

    for qi, (q_id, q_chain) in enumerate(queries):
        q_pdb = os.path.join(PDB_DIR, f'{q_id.lower()}.pdb')
        if not os.path.exists(q_pdb):
            print(f"  Skipping {q_id}: PDB not found")
            continue

        result_file = os.path.join(WORK_DIR, f'result_{q_id}.m8')
        tmp_dir = os.path.join(WORK_DIR, f'tmp_{q_id}')

        cmd = [
            FOLDSEEK, 'easy-search',
            q_pdb, db_path,
            result_file, tmp_dir,
            '--format-output', FORMAT_STRING,
            '-e', '10000',
            '--exact-tmscore', '1',
            '--max-seqs', '3000',
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print(f"  Search failed for {q_id}: {proc.stderr[-200:]}")
            continue

        # Parse results
        if os.path.exists(result_file):
            with open(result_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 11:
                        all_results.append({
                            'query': parts[0][:4].upper(),
                            'target': parts[1][:4].upper(),
                            'foldseek_qtmscore': float(parts[2]) if parts[2] else -1,
                            'foldseek_ttmscore': float(parts[3]) if parts[3] else -1,
                            'foldseek_alntmscore': float(parts[4]) if parts[4] else -1,
                            'foldseek_rmsd': float(parts[5]) if parts[5] else -1,
                            'foldseek_lddt': float(parts[6]) if parts[6] else -1,
                            'foldseek_alnlen': int(parts[7]) if parts[7] else 0,
                            'foldseek_evalue': float(parts[8]) if parts[8] else -1,
                            'foldseek_qcov': float(parts[9]) if parts[9] else -1,
                            'foldseek_tcov': float(parts[10]) if parts[10] else -1,
                        })

        # Clean up per-query temp files
        if os.path.exists(result_file):
            os.remove(result_file)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if (qi + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (qi + 1) / elapsed
            eta = (len(queries) - qi - 1) / rate
            print(f"  Query {qi+1}/{len(queries)} ({q_id}) — {rate:.1f}/s, ETA {eta:.0f}s, results: {len(all_results)}")

    elapsed = time.time() - t0
    print(f"\nFoldseek done: {len(all_results)} results in {elapsed:.1f}s")

    # ── Step 3: Filter to benchmark pairs and save ──────────────────────
    valid_pairs = set()
    for p in pairs:
        valid_pairs.add((p['query_id'].lower(), p['target_id'].lower()))

    filtered = []
    for r in all_results:
        key = (r['query'].lower(), r['target'].lower())
        if key in valid_pairs:
            filtered.append(r)

    print(f"Filtered to benchmark pairs: {len(filtered)}/{len(all_results)}")

    if filtered:
        with open(OUTPUT, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=filtered[0].keys())
            writer.writeheader()
            writer.writerows(filtered)
        print(f"Saved to {OUTPUT}")

    # Report missing pairs
    found_pairs = set((r['query'].lower(), r['target'].lower()) for r in filtered)
    missing = valid_pairs - found_pairs
    print(f"Missing pairs (no Foldseek result): {len(missing)}/{len(valid_pairs)}")

    # Clean up database files
    for f in os.listdir(WORK_DIR):
        if f.startswith('target_db') or f.startswith('target_pdbs'):
            path = os.path.join(WORK_DIR, f)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    main()
