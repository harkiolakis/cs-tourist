#!/usr/bin/env python3
"""Case study demo: reproduce the three diff-T case studies from the paper.

Usage:
    python examples/case_study_demo.py

Requires PDB files for 1R44, 1O98, 1HRK, 1JXA, 1AQ2, 1QUM
(available from RCSB or the Zenodo archive).
"""

import sys
import os

# Add turist to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from turist.io import parse_chain_trace
from turist.descriptor import compute_descriptor
from turist.matching import align_catalytic_sites

# Case studies from the paper (Section 3.6)
# All from diff-T (different CATH topologies), all with precision = 1.0
CASE_STUDIES = [
    {
        "name": "Case 1: VanX dipeptidase vs. phosphoglycerate mutase",
        "pdb_a": "1r44A.pdb",
        "pdb_b": "1o98A.pdb",
        "cat_a": [47, 87, 145, 181, 233],   # 5 catalytic residues
        "cat_b": [11, 12, 62, 88, 137, 138, 168, 188, 211, 235],  # 10
        "expected_recall": 0.50,
        "expected_rmsd": 4.65,
    },
    {
        "name": "Case 2: Ferrochelatase vs. glucosamine-6-phosphate synthase",
        "pdb_a": "1hrkA.pdb",
        "pdb_b": "1jxaA.pdb",
        "cat_a": [63, 196, 233, 263, 264, 383],  # 6 catalytic residues
        "cat_b": [1, 2, 34, 37, 60, 73, 98, 99, 162, 300],  # 10
        "expected_recall": 0.60,
        "expected_rmsd": 9.22,
    },
    {
        "name": "Case 3: PEP carboxykinase vs. endonuclease IV",
        "pdb_a": "1aq2A.pdb",
        "pdb_b": "1qumA.pdb",
        "cat_a": [1, 2, 87, 88, 89, 207, 230, 254],  # 8 catalytic residues
        "cat_b": [1, 2, 3, 69, 70, 109, 147, 148, 149, 175, 219],  # 11
        "expected_recall": 0.73,
        "expected_rmsd": 6.74,
    },
]


def run_case_study(case, pdb_dir="."):
    """Run a single case study."""
    print(f"\n{'='*60}")
    print(f"  {case['name']}")
    print(f"{'='*60}")

    path_a = os.path.join(pdb_dir, case["pdb_a"])
    path_b = os.path.join(pdb_dir, case["pdb_b"])

    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"  [SKIP] PDB files not found in {pdb_dir}/")
        print(f"    Need: {case['pdb_a']}, {case['pdb_b']}")
        return

    # Parse Cα traces
    trace_a = parse_chain_trace(path_a)
    trace_b = parse_chain_trace(path_b)

    # Run CS-TOURIST Mode C with connectivity-aware descriptor
    result = align_catalytic_sites(
        trace_a, trace_b,
        case["cat_a"], case["cat_b"],
        mode="C",
        connectivity_aware=True,
        radius=8.0,
    )

    print(f"  Recall:    {result.recall:.3f}  (expected: {case['expected_recall']:.2f})")
    print(f"  Precision: {result.precision:.3f}  (expected: 1.00)")
    print(f"  RMSD:      {result.rmsd:.2f} Å  (expected: {case['expected_rmsd']:.2f})")
    print(f"  Aligned:   {result.n_aligned} residue pairs")


if __name__ == "__main__":
    pdb_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    print("CS-TOURIST Case Study Demo")
    print(f"PDB directory: {pdb_dir}")

    for case in CASE_STUDIES:
        run_case_study(case, pdb_dir)

    print(f"\n{'='*60}")
    print("Done. All case studies should show precision = 1.0")
    print("with recall matching the expected values from the paper.")
