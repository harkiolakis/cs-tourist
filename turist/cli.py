"""Command-line interface for TURIST.

Usage:
    turist align <A> <B> [--mode sequence|exact|dp] [--radius 5.7]
                 [--chain ID] [--chain-a ID] [--chain-b ID]
                 [--min-agreement 0.5] [--out results.json] [--write-pdb out.pdb]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .align import align_structures
from .superpose import write_superposed_pdb


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="turist",
        description="TURIST: pairwise 3D protein structure alignment "
                    "(Harkiolakis trace-defined Cα environment signatures).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("align", help="Align two structures and report RMSD + transform.")
    a.add_argument("a", help="Reference structure: PDB file path or 4-char PDB ID.")
    a.add_argument("b", help="Mobile structure: PDB file path or 4-char PDB ID.")
    a.add_argument("--mode", default="sequence", choices=["sequence", "exact", "dp"],
                   help="Matching strategy (default: sequence).")
    a.add_argument("--radius", type=float, default=5.7,
                   help="TURIST environment sphere radius in Angstrom (default: 5.7).")
    a.add_argument("--chain", default=None, help="Chain id to use for both structures.")
    a.add_argument("--chain-a", default=None, help="Chain id for structure A.")
    a.add_argument("--chain-b", default=None, help="Chain id for structure B.")
    a.add_argument("--min-agreement", type=float, default=0.5,
                   help="sequence-mode TURIST trim threshold (default: 0.5).")
    a.add_argument("--out", default=None, help="Write full results to this JSON file.")
    a.add_argument("--write-pdb", default=None,
                   help="Write the mobile structure superposed onto the reference to this PDB file.")
    a.add_argument("--cache-dir", default=None, help="Directory to cache fetched PDB files.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "align":
        res = align_structures(
            args.a, args.b,
            mode=args.mode,
            radius=args.radius,
            chain=args.chain,
            chain_a=args.chain_a,
            chain_b=args.chain_b,
            min_agreement=args.min_agreement,
            cache_dir=args.cache_dir,
        )

        print(f"TURIST alignment  | mode={res.mode}  radius={res.radius} A")
        print(f"  A: {os.path.basename(res.source_a)} chain {res.chain_a} ({res.n_residues_a} res)")
        print(f"  B: {os.path.basename(res.source_b)} chain {res.chain_b} ({res.n_residues_b} res)")
        print(f"  aligned Cα pairs: {res.n_aligned}")
        print(f"  RMSD: {res.rmsd:.4f} A")
        print("  4x4 transform (B -> A):")
        for row in res.transform:
            print("    " + "  ".join(f"{v:9.5f}" for v in row))
        if res.info:
            print("  match info: " + json.dumps(res.info))

        if args.out:
            with open(args.out, "w") as f:
                json.dump(res.to_dict(), f, indent=2)
            print(f"  results written to {args.out}")

        if args.write_pdb:
            write_superposed_pdb(res.source_b, args.write_pdb,
                                 res.transform[:3, :3], res.transform[:3, 3],
                                 chain_id=res.chain_b)
            print(f"  superposed PDB written to {args.write_pdb}")

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
