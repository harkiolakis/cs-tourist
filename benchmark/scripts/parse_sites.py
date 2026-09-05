#!/usr/bin/env python3
"""Parse PDB SITE records for catalytic (AC) and binding (BC) site annotations.

Scans all PDB files in the benchmark directory, extracts SITE records,
and saves a structured JSON mapping pdb_id -> chain -> {catalytic, binding} residue sets.

Usage:
  python parse_sites.py --pdb-dir /mnt/shared-workspace/pdb_benchmark \
      --output /mnt/shared-workspace/pdb_site_annotations.json
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict


def parse_site_records(pdb_path):
    """Parse SITE records from a PDB file.

    SITE record format (columns):
      1-6:   "SITE  "
      8-10:  sequence number
      12-14: site ID (e.g., AC1, BC1)
      16-17: number of residues
      19-21: residue name (3-letter code)
      23:    chain ID
      24-27: residue sequence number
      28:    insertion code
      (repeats for each residue in the record)

    Returns dict: {chain_id: {'catalytic': set(), 'binding': set()}}
    """
    sites = defaultdict(lambda: {'catalytic': set(), 'binding': set()})

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('SITE'):
                continue

            # Extract site ID (columns 12-14, 0-indexed 11-14)
            site_id = line[11:15].strip()

            # Classify: AC = catalytic, BC = binding
            if site_id.startswith('AC'):
                site_type = 'catalytic'
            elif site_id.startswith('BC'):
                site_type = 'binding'
            else:
                continue  # Skip other site types

            # Parse residue entries (starting at column 19, 0-indexed 18)
            # Each residue entry is 11 characters: 3-letter code + chain + resseq + icode
            # Format: "RES CHAIN RESSEQ ICODE"
            # Actually, PDB SITE format has residues in groups:
            # cols 19-21: resName, 23: chainID, 24-27: resSeq, 28: iCode
            # Then repeats at 30-32, 34, 35-38, 39, etc.
            # Each group is 11 columns wide (19-29, 30-40, 41-51, 52-62)

            # Parse using column positions
            # Group 1: cols 19-28 (0-indexed 18-27)
            # Group 2: cols 30-39 (0-indexed 29-38)
            # Group 3: cols 41-50 (0-indexed 40-49)
            # Group 4: cols 52-61 (0-indexed 51-60)

            for start in [18, 29, 40, 51]:
                if start + 10 > len(line):
                    break
                res_name = line[start:start + 3].strip()
                if not res_name:
                    continue
                chain = line[start + 4:start + 5].strip()
                resseq_str = line[start + 5:start + 9].strip()
                if not resseq_str:
                    continue
                try:
                    resseq = int(resseq_str)
                except ValueError:
                    continue
                icode = line[start + 9:start + 10].strip()

                if chain:
                    sites[chain][site_type].add(resseq)

    # Convert sets to sorted lists for JSON serialization
    result = {}
    for chain, site_dict in sites.items():
        result[chain] = {
            'catalytic': sorted(site_dict['catalytic']),
            'binding': sorted(site_dict['binding']),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description='Parse PDB SITE records')
    parser.add_argument('--pdb-dir', default='/mnt/shared-workspace/pdb_benchmark')
    parser.add_argument('--output', default='/mnt/shared-workspace/pdb_site_annotations.json')
    args = parser.parse_args()

    pdb_files = [f for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
    print(f"Found {len(pdb_files)} PDB files")

    annotations = {}
    n_with_sites = 0
    n_catalytic = 0
    n_binding = 0

    for i, fname in enumerate(pdb_files):
        pdb_id = fname[:-4].lower()  # Remove .pdb, lowercase
        path = os.path.join(args.pdb_dir, fname)
        sites = parse_site_records(path)
        if sites:
            annotations[pdb_id] = sites
            n_with_sites += 1
            for chain, site_dict in sites.items():
                n_catalytic += len(site_dict['catalytic'])
                n_binding += len(site_dict['binding'])

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(pdb_files)} files, {n_with_sites} with SITE records")

    print(f"\nDone: {n_with_sites}/{len(pdb_files)} files have SITE records")
    print(f"  Catalytic site residues (total): {n_catalytic}")
    print(f"  Binding site residues (total): {n_binding}")

    with open(args.output, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved to {args.output}")

    # Print some examples
    print("\nExamples:")
    for pdb_id in list(annotations.keys())[:5]:
        print(f"  {pdb_id}: {annotations[pdb_id]}")


if __name__ == '__main__':
    main()
