"""PDB input: parsing Cα traces + sequences, and fetching structures by ID."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from Bio.PDB import PDBList, PDBParser

# Standard 20 amino acids (3-letter codes). Used to filter real residues.
STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

# 3-letter -> 1-letter for sequence extraction.
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


@dataclass
class ChainTrace:
    """A single chain's Cα trace in residue order.

    Attributes:
        coords: (N, 3) array of Cα coordinates.
        sequence: 1-letter sequence string of length N.
        res_ids: list of residue identifiers (hetero flag, resseq, icode).
        chain_id: chain identifier.
    """

    coords: np.ndarray
    sequence: str
    res_ids: list
    chain_id: str

    def __len__(self) -> int:
        return len(self.sequence)


def _looks_like_pdb_id(s: str) -> bool:
    """Heuristic: a 4-character alphanumeric token with no path separator."""
    s = s.strip()
    return len(s) == 4 and s.isalnum() and "/" not in s and "\\" not in s


def fetch_pdb(pdb_id: str, dest_dir: Optional[str] = None) -> str:
    """Download a PDB file by 4-character ID and return the local path."""
    pdb_id = pdb_id.lower()
    dest_dir = dest_dir or tempfile.gettempdir()
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        # Try RCSB direct download first (works for current entries).
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            # Fall back to Biopython's PDBList (handles obsolete entries).
            pl = PDBList()
            pl.retrieve_pdb_file(pdb_id, pdir=dest_dir, file_format="pdb")
            # PDBList nests under a two-letter dir; locate the file.
            cand = os.path.join(dest_dir, pdb_id[1:3], f"pdb{pdb_id}.ent")
            if os.path.exists(cand):
                os.replace(cand, path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"Could not fetch PDB entry {pdb_id}")
    return path


def resolve_structure(source: str, cache_dir: Optional[str] = None) -> str:
    """Return a local PDB file path for a file path or a PDB ID."""
    if os.path.exists(source):
        return source
    if _looks_like_pdb_id(source):
        return fetch_pdb(source, dest_dir=cache_dir)
    raise FileNotFoundError(f"Not an existing file or a valid PDB ID: {source!r}")


def parse_chain_trace(
    pdb_path: str,
    chain_id: Optional[str] = None,
    model_id: int = 0,
) -> ChainTrace:
    """Parse a PDB file and return the Cα trace of one chain.

    Args:
        pdb_path: path to a PDB file.
        chain_id: chain to use; if None, the first chain with standard residues.
        model_id: NMR/coordinate model index (0 = first).
    """
    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("struct", pdb_path)
    model = structure[model_id]

    chosen = None
    if chain_id is not None:
        if chain_id in model:
            chosen = model[chain_id]
        else:
            raise KeyError(f"Chain {chain_id!r} not found in {pdb_path}")
    else:
        # Pick the chain with the most standard-amino-acid residues.
        best, best_n = None, 0
        for chain in model:
            n = sum(1 for res in chain if res.resname in STANDARD_AA)
            if n > best_n:
                best, best_n = chain, n
        chosen = best
        if chosen is None:
            raise ValueError(f"No standard amino-acid chain found in {pdb_path}")

    coords: List[np.ndarray] = []
    seq_chars: List[str] = []
    res_ids: list = []
    for res in chosen:
        if res.resname not in STANDARD_AA:
            continue
        if "CA" not in res:
            continue
        ca = res["CA"]
        # Skip alternate conformations beyond the first.
        if ca.get_altloc() not in (" ", "", None, "A"):
            continue
        coords.append(ca.get_coord().astype(float))
        seq_chars.append(AA3TO1.get(res.resname, "X"))
        res_ids.append(res.id)

    if len(coords) < 3:
        raise ValueError(
            f"Chain {chosen.id!r} in {pdb_path} has only {len(coords)} Cα atoms; "
            "need at least 3 to build TURIST frames."
        )

    return ChainTrace(
        coords=np.asarray(coords, dtype=float),
        sequence="".join(seq_chars),
        res_ids=res_ids,
        chain_id=chosen.id,
    )
