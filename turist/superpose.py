"""Rigid-body superposition (Kabsch), RMSD, 4x4 transform, and PDB export."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np


def kabsch(b_coords: np.ndarray, a_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimal rigid-body fit of ``b_coords`` onto ``a_coords`` (paired rows).

    Returns ``(R, t, rmsd)`` such that ``R @ b + t`` best matches ``a``.
    ``a_coords`` is the reference (fixed); ``b_coords`` is moved.
    """
    a = np.asarray(a_coords, dtype=float)
    b = np.asarray(b_coords, dtype=float)
    if a.shape != b.shape or a.shape[1] != 3:
        raise ValueError("Coordinate arrays must share shape (N, 3).")

    centroid_a = a.mean(axis=0)
    centroid_b = b.mean(axis=0)
    a_c = a - centroid_a
    b_c = b - centroid_b

    # Covariance matrix H = sum b_i a_i^T  (rotate b onto a).
    H = b_c.T @ a_c
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_a - R @ centroid_b

    moved = (R @ b.T).T + t
    diff = moved - a
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return R, t, rmsd


def transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Assemble a 4x4 homogeneous transform mapping ``b -> a``."""
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid transform ``R @ x + t`` to (N, 3) coordinates."""
    coords = np.asarray(coords, dtype=float)
    return (R @ coords.T).T + t


def write_superposed_pdb(
    src_pdb_path: str,
    dest_pdb_path: str,
    R: np.ndarray,
    t: np.ndarray,
    chain_id: Optional[str] = None,
    model_id: int = 0,
) -> None:
    """Write a copy of ``src_pdb_path`` with all atoms transformed by R, t.

    Only the chosen chain (or all standard chains if None) is written, so the
    output is a clean superposed structure ready to overlay on the reference.
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    parser = PDBParser(QUIET=True, PERMISSIVE=True)
    structure = parser.get_structure("b", src_pdb_path)
    model = structure[model_id]

    class _ChainSelect(Select):
        def accept_chain(self, chain):
            if chain_id is None:
                return True
            return chain.id == chain_id

    # Apply the transform to every atom of the selected chain(s).
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            for atom in res:
                atom.set_coord(apply_transform(atom.get_coord(), R, t))

    io = PDBIO()
    io.set_structure(structure)
    io.save(dest_pdb_path, _ChainSelect())
