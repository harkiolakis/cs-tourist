#!/usr/bin/env python3
"""Run TM-align on benchmark pairs. Supports splitting across machines.

Usage:
  python benchmark_tmalign.py --start 0 --end 10000    # first chunk
  python benchmark_tmalign.py --start 10000 --end 20000 # second chunk
  python benchmark_tmalign.py                             # all pairs
"""

import csv
import os
import re
import sys
import time
import argparse
import subprocess
from multiprocessing import Pool
from pathlib import Path

PAIR_LIST = '/mnt/shared-workspace/pair_list.csv'
PDB_DIR = '/mnt/shared-workspace/pdb_benchmark'
OUTPUT = '/mnt/shared-workspace/results_tmalign'
CHECKPOINT_DIR = '/mnt/shared-workspace'

# Find TMalign binary: check local, then shared workspace
if os.path.exists('/workspace/USalign/TMalign'):
    TMALIGN = '/workspace/USalign/TMalign'
elif os.path.exists('/workspace/TMalign'):
    TMALIGN = '/workspace/TMalign'
else:
    # Copy from shared workspace to local
    import shutil
    src = '/mnt/shared-workspace/TMalign'
    dst = '/workspace/TMalign'
    if os.path.exists(src):
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        TMALIGN = dst
    else:
        raise FileNotFoundError("TMalign binary not found")


def parse_tmalign_output(stdout):
    """Parse TM-align stdout for TM-score, RMSD, aligned length."""
    # TM-score normalized by Structure_1 (query)
    tmscore_q = None
    tmscore_t = None
    rmsd = None
    aln_len = None

    # Find all TM-score lines
    tmscore_lines = re.findall(r'TM-score\s*=\s*([\d.]+)', stdout)
    if len(tmscore_lines) >= 2:
        tmscore_q = float(tmscore_lines[0])
        tmscore_t = float(tmscore_lines[1])

    # Aligned length
    m = re.search(r'Aligned length=\s*(\d+)', stdout)
    if m:
        aln_len = int(m.group(1))

    # RMSD
    m = re.search(r'RMSD=\s*([\d.]+)', stdout)
    if m:
        rmsd = float(m.group(1))

    return tmscore_q, tmscore_t, rmsd, aln_len


def run_tmalign_pair(args):
    """Run TM-align on a single pair."""
    query, target, q_chain, t_chain, idx = args

    q_pdb = os.path.join(PDB_DIR, f'{query.lower()}.pdb')
    t_pdb = os.path.join(PDB_DIR, f'{target.lower()}.pdb')

    if not os.path.exists(q_pdb) or not os.path.exists(t_pdb):
        return {
            'query': query, 'target': target,
            'q_chain': q_chain, 't_chain': t_chain,
            'tmalign_tmscore_q': -1, 'tmalign_tmscore_t': -1,
            'tmalign_rmsd': -1, 'tmalign_alnlen': 0,
            'error': 'PDB file missing'
        }

    cmd = [TMALIGN, q_pdb, t_pdb, '-chain1', q_chain, '-chain2', t_chain]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        stdout = proc.stdout

        tmscore_q, tmscore_t, rmsd, aln_len = parse_tmalign_output(stdout)

        return {
            'query': query, 'target': target,
            'q_chain': q_chain, 't_chain': t_chain,
            'tmalign_tmscore_q': tmscore_q if tmscore_q is not None else -1,
            'tmalign_tmscore_t': tmscore_t if tmscore_t is not None else -1,
            'tmalign_rmsd': rmsd if rmsd is not None else -1,
            'tmalign_alnlen': aln_len if aln_len is not None else 0,
            'error': '' if tmscore_q is not None else 'parse_failed',
        }
    except subprocess.TimeoutExpired:
        return {
            'query': query, 'target': target,
            'q_chain': q_chain, 't_chain': t_chain,
            'tmalign_tmscore_q': -1, 'tmalign_tmscore_t': -1,
            'tmalign_rmsd': -1, 'tmalign_alnlen': 0,
            'error': 'timeout'
        }
    except Exception as e:
        return {
            'query': query, 'target': target,
            'q_chain': q_chain, 't_chain': t_chain,
            'tmalign_tmscore_q': -1, 'tmalign_tmscore_t': -1,
            'tmalign_rmsd': -1, 'tmalign_alnlen': 0,
            'error': str(e)[:200]
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=None, help='End index (exclusive)')
    parser.add_argument('--worker-id', type=str, default='w0', help='Worker ID for output filename')
    parser.add_argument('--n-procs', type=int, default=16, help='Number of parallel processes')
    args = parser.parse_args()

    with open(PAIR_LIST) as f:
        reader = csv.DictReader(f)
        pairs = list(reader)

    end = args.end if args.end is not None else len(pairs)
    chunk = pairs[args.start:end]
    print(f"Worker {args.worker_id}: pairs {args.start}-{end} ({len(chunk)} pairs)")

    args_list = [
        (p['query_id'], p['target_id'], p['query_chain'], p['target_chain'], args.start + i)
        for i, p in enumerate(chunk)
    ]

    output_file = f"{OUTPUT}_{args.worker_id}.csv"
    results = []
    t0 = time.time()

    with Pool(args.n_procs) as pool:
        for i, result in enumerate(pool.imap(run_tmalign_pair, args_list, chunksize=50)):
            results.append(result)
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(chunk) - i - 1) / rate
                errors = sum(1 for r in results if r['error'])
                print(f"  Progress: {i+1}/{len(chunk)} ({rate:.1f}/s, ETA {eta:.0f}s, errors={errors})")

                # Checkpoint
                ckpt_path = os.path.join(CHECKPOINT_DIR, f'tmalign_{args.worker_id}_ckpt_{i+1}.csv')
                with open(ckpt_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)

    elapsed = time.time() - t0
    errors = sum(1 for r in results if r['error'])
    print(f"\nDone: {len(results)} pairs in {elapsed:.1f}s ({len(results)/elapsed:.1f}/s, {errors} errors)")

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved to {output_file}")

    # Clean up checkpoints
    for fn in os.listdir(CHECKPOINT_DIR):
        if fn.startswith(f'tmalign_{args.worker_id}_ckpt_'):
            os.remove(os.path.join(CHECKPOINT_DIR, fn))
    print("Cleaned up checkpoints")


if __name__ == '__main__':
    main()
