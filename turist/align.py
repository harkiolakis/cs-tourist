"""Orchestrator: parse -> descriptors -> match -> superpose -> result."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

from . import io as _io
from .descriptor import DEFAULT_RADIUS, build_all_signatures
from .matching import match as _match
from .matching import _precompute_score_matrix, dp_local_float, dp_semiglobal_float
from .descriptor import relative_offsets, EMPTY
from .superpose import kabsch, transform_matrix, apply_transform


@dataclass
class AlignmentResult:
    """Outcome of a TURIST pairwise alignment."""

    mode: str
    radius: float
    chain_a: str
    chain_b: str
    n_residues_a: int
    n_residues_b: int
    n_aligned: int
    rmsd: float
    transform: np.ndarray  # 4x4
    correspondence: List[Tuple[int, int]] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    source_a: str = ""
    source_b: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["transform"] = self.transform.tolist()
        return d


def align_structures(
    a: str,
    b: str,
    mode: str = "sequence",
    radius: float = DEFAULT_RADIUS,
    chain: Optional[str] = None,
    chain_a: Optional[str] = None,
    chain_b: Optional[str] = None,
    min_agreement: float = 0.5,
    gap_penalty: int = -2,
    cache_dir: Optional[str] = None,
) -> AlignmentResult:
    """Align two protein structures with TURIST.

    Args:
        a, b: PDB file paths or 4-character PDB IDs.
        mode: 'sequence' (default), 'exact', 'dp', 'local', or 'semiglobal'.
        radius: TURIST environment sphere radius in Angstrom (default 5.7).
        chain: chain id to use for both structures (overrides chain_a/chain_b).
        chain_a, chain_b: per-structure chain ids.
        min_agreement: TURIST agreement threshold for trimming in 'sequence' mode.
        gap_penalty: gap penalty for 'dp', 'local', and 'semiglobal' modes (default -2).
        cache_dir: where to cache fetched PDB files.

    Returns:
        AlignmentResult with RMSD, 4x4 transform, and correspondence.
    """
    if chain is not None:
        chain_a = chain_a or chain
        chain_b = chain_b or chain

    path_a = _io.resolve_structure(a, cache_dir=cache_dir)
    path_b = _io.resolve_structure(b, cache_dir=cache_dir)
    trace_a = _io.parse_chain_trace(path_a, chain_id=chain_a)
    trace_b = _io.parse_chain_trace(path_b, chain_id=chain_b)

    sigs_a = build_all_signatures(trace_a.coords, radius)
    sigs_b = build_all_signatures(trace_b.coords, radius)

    pairs, info = _match(
        mode, trace_a.sequence, trace_b.sequence, sigs_a, sigs_b,
        min_agreement=min_agreement,
        gap_penalty=gap_penalty,
    )

    if len(pairs) < 3:
        raise RuntimeError(
            f"Only {len(pairs)} corresponding Cα pairs found (mode={mode!r}); "
            "need at least 3 for a rigid-body fit. Try another mode or radius."
        )

    a_idx = np.array([p[0] for p in pairs])
    b_idx = np.array([p[1] for p in pairs])
    R, t, rmsd = kabsch(trace_b.coords[b_idx], trace_a.coords[a_idx])
    M = transform_matrix(R, t)

    return AlignmentResult(
        mode=mode,
        radius=radius,
        chain_a=trace_a.chain_id,
        chain_b=trace_b.chain_id,
        n_residues_a=len(trace_a),
        n_residues_b=len(trace_b),
        n_aligned=len(pairs),
        rmsd=rmsd,
        transform=M,
        correspondence=[(int(i), int(j)) for i, j in pairs],
        info=info,
        source_a=path_a,
        source_b=path_b,
    )


def _compute_d0(L: int) -> float:
    """TM-score normalisation distance."""
    return max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def _tm_like_score(n_aligned: int, rmsd: float, L_query: int) -> float:
    """TM-score-like normalised score."""
    d0 = _compute_d0(L_query)
    return (n_aligned / L_query) * (1.0 / (1.0 + (rmsd / d0) ** 2))


def align_structures_refined(
    a: str,
    b: str,
    mode: str = "local",
    radius: float = 8.0,
    chain: Optional[str] = None,
    chain_a: Optional[str] = None,
    chain_b: Optional[str] = None,
    gap_penalty: int = -2,
    max_iterations: int = 5,
    convergence_threshold: float = 0.001,
    distance_weight: float = 1.0,
    cache_dir: Optional[str] = None,
) -> AlignmentResult:
    """Align two structures with iterative superposition refinement.

    After an initial octant-agreement DP alignment, iteratively re-aligns
    using a hybrid scoring function that combines TURIST's octant-agreement
    signal with a TM-align-style distance-based score derived from the
    current superposition. The best alignment (highest TM-like score)
    across all iterations is returned.

    Args:
        a, b: PDB file paths or 4-character PDB IDs.
        mode: 'local' or 'semiglobal' for the DP alignment.
        radius: TURIST environment sphere radius in Angstrom.
        chain, chain_a, chain_b: chain selection.
        gap_penalty: gap penalty for DP (default -2).
        max_iterations: maximum refinement iterations (default 5).
        convergence_threshold: stop if TM-like improvement < this (default 0.001).
        distance_weight: weight for distance score in hybrid matrix (default 1.0).
        cache_dir: where to cache fetched PDB files.

    Returns:
        AlignmentResult with refined correspondence, RMSD, and per-iteration
        info in the ``info`` dict.
    """
    if chain is not None:
        chain_a = chain_a or chain
        chain_b = chain_b or chain

    path_a = _io.resolve_structure(a, cache_dir=cache_dir)
    path_b = _io.resolve_structure(b, cache_dir=cache_dir)
    trace_a = _io.parse_chain_trace(path_a, chain_id=chain_a)
    trace_b = _io.parse_chain_trace(path_b, chain_id=chain_b)

    coords_a = trace_a.coords
    coords_b = trace_b.coords
    nA, nB = len(coords_a), len(coords_b)
    L_query = nA  # query-normalised

    sigs_a = build_all_signatures(coords_a, radius)
    sigs_b = build_all_signatures(coords_b, radius)

    # Precompute octant-agreement score matrix (reused across iterations).
    rels_a = [relative_offsets(s, i) for i, s in enumerate(sigs_a)]
    rels_b = [relative_offsets(s, j) for j, s in enumerate(sigs_b)]
    S_octant = _precompute_score_matrix(rels_a, rels_b).astype(float)  # (nA, nB)

    d0 = _compute_d0(min(nA, nB))
    gap_f = float(gap_penalty)

    # --- Initial alignment (octant-only) ---
    pairs, info_init = _match(
        mode, trace_a.sequence, trace_b.sequence, sigs_a, sigs_b,
        gap_penalty=gap_penalty,
    )
    if len(pairs) < 3:
        raise RuntimeError(
            f"Only {len(pairs)} corresponding Cα pairs found (mode={mode!r}); "
            "need at least 3 for a rigid-body fit."
        )

    a_idx = np.array([p[0] for p in pairs])
    b_idx = np.array([p[1] for p in pairs])
    R, t, rmsd = kabsch(coords_b[b_idx], coords_a[a_idx])
    best_tm = _tm_like_score(len(pairs), rmsd, L_query)

    best_pairs = pairs
    best_R, best_t, best_rmsd = R, t, rmsd

    iter_scores = [best_tm]
    iter_rmsds = [rmsd]

    # --- Refinement loop ---
    for iteration in range(1, max_iterations + 1):
        # Transform target coords into query frame.
        transformed_b = apply_transform(coords_b, R, t)  # (nB, 3)

        # All-pairs distances in superposed frame (vectorised).
        diffs = coords_a[:, None, :] - transformed_b[None, :, :]  # (nA, nB, 3)
        distances = np.sqrt(np.sum(diffs * diffs, axis=2))  # (nA, nB)

        # Distance-based score (TM-score kernel).
        dscore = 1.0 / (1.0 + (distances / d0) ** 2)  # (nA, nB), range 0–1

        # Hybrid score matrix: octant + weighted distance (scaled to 0–7).
        S_hybrid = S_octant + distance_weight * dscore * 7.0

        # Re-run DP with hybrid scores.
        if mode == "local":
            pairs_new, info_new = dp_local_float(S_hybrid, gap_penalty=gap_f)
        elif mode == "semiglobal":
            pairs_new, info_new = dp_semiglobal_float(S_hybrid, gap_penalty=gap_f)
        else:
            raise ValueError(f"Refinement supports 'local' or 'semiglobal', got {mode!r}")

        if len(pairs_new) < 3:
            break  # Cannot superpose with < 3 pairs; keep previous best.

        a_idx = np.array([p[0] for p in pairs_new])
        b_idx = np.array([p[1] for p in pairs_new])
        R_new, t_new, rmsd_new = kabsch(coords_b[b_idx], coords_a[a_idx])
        tm_new = _tm_like_score(len(pairs_new), rmsd_new, L_query)

        iter_scores.append(tm_new)
        iter_rmsds.append(rmsd_new)

        # Track best alignment across all iterations.
        if tm_new > best_tm:
            improvement = tm_new - best_tm
            best_tm = tm_new
            best_pairs = pairs_new
            best_R, best_t, best_rmsd = R_new, t_new, rmsd_new
            R, t = R_new, t_new  # Update for next iteration's superposition
        else:
            # No improvement; still update superposition for next iteration
            # (the new alignment may guide toward a better local optimum).
            R, t = R_new, t_new
            improvement = 0.0

        if improvement < convergence_threshold:
            break

    M = transform_matrix(best_R, best_t)

    info = {
        "mode": mode,
        "refined": True,
        "max_iterations": max_iterations,
        "iterations_run": len(iter_scores) - 1,
        "distance_weight": distance_weight,
        "per_iteration_tm_like": iter_scores,
        "per_iteration_rmsd": iter_rmsds,
        "best_tm_like": best_tm,
        "alignment_score": info_init.get("alignment_score", 0),
        "n_used": len(best_pairs),
    }

    return AlignmentResult(
        mode=f"{mode}_refined",
        radius=radius,
        chain_a=trace_a.chain_id,
        chain_b=trace_b.chain_id,
        n_residues_a=nA,
        n_residues_b=nB,
        n_aligned=len(best_pairs),
        rmsd=best_rmsd,
        transform=M,
        correspondence=[(int(i), int(j)) for i, j in best_pairs],
        info=info,
        source_a=path_a,
        source_b=path_b,
    )
