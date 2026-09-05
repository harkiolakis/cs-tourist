"""Correspondence establishment between two Cα traces.

Three selectable strategies:

* ``sequence`` (default) -- global Needleman-Wunsch (BLOSUM62) sequence
  alignment anchors the residue correspondence; TURIST relative-offset
  agreement scores each aligned pair and trims low-confidence regions.
* ``exact`` -- structural only. Finds maximal diagonal runs of residue pairs
  whose *relative* neighbour offsets agree on every shared octant.
* ``dp`` -- structural only. Octant-agreement scores feed a Needleman-Wunsch
  dynamic program over the two signature sequences (gaps allowed).

Every matcher returns ``(pairs, info)`` where ``pairs`` is a list of
0-based ``(i, j)`` chain-index pairs and ``info`` carries mode-specific stats.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .descriptor import EMPTY, relative_offsets

Pairs = List[Tuple[int, int]]

# 0-based octant slots ignored during matching. Slot 3 (BUR) always holds the
# immediately preceding Cα (prev1 lies exactly on the -x axis -> back, with
# up/right decided by the positive tie-break), at a fixed relative offset of
# -1 for every residue. It is structurally invariant and carries no
# discriminative information, so it is excluded from agreement/exact matching.
INVARIANT_SLOTS = {3}


def signature_agreement(rel_a: np.ndarray, rel_b: np.ndarray) -> Tuple[int, int]:
    """Return ``(matches, shared)``: number of octants where both signatures
    are occupied and their relative offsets are equal, and the number of
    octants occupied in both. Invariant (non-discriminative) slots are
    excluded."""
    if rel_a is None or rel_b is None:
        return 0, 0
    mask = np.ones(8, dtype=bool)
    mask[list(INVARIANT_SLOTS)] = False
    both = (rel_a != EMPTY) & (rel_b != EMPTY) & mask
    shared = int(both.sum())
    if shared == 0:
        return 0, 0
    matches = int(((rel_a == rel_b) & both).sum())
    return matches, shared


def _agreement_fraction(rel_a, rel_b) -> float:
    m, shared = signature_agreement(rel_a, rel_b)
    return m / shared if shared > 0 else 0.0


def _precompute_score_matrix(rels_a, rels_b) -> np.ndarray:
    """Vectorised nA×nB matrix of octant-agreement scores (0–7).

    Replaces the per-cell ``signature_agreement`` call inside DP loops.
    """
    ra = np.asarray(rels_a)  # (nA, 8)
    rb = np.asarray(rels_b)  # (nB, 8)
    mask = np.ones(8, dtype=bool)
    mask[list(INVARIANT_SLOTS)] = False
    both = (ra[:, None, :] != EMPTY) & (rb[None, :, :] != EMPTY) & mask
    return ((ra[:, None, :] == rb[None, :, :]) & both).sum(axis=2)


def _precompute_score_matrix_multi(rels_a_multi, rels_b_multi) -> np.ndarray:
    """Build nA×nB score matrix using any-pair offset matching (v4).

    For each cell (i, j) and each octant k (excluding invariant BUR slot 3):
    score +1 if ANY offset in rels_a_multi[i][k] equals ANY offset in
    rels_b_multi[j][k].  Score range: 0–7.

    Uses one-hot boolean matrices per octant and matrix multiplication
    for vectorised any-pair intersection detection.
    """
    nA = len(rels_a_multi)
    nB = len(rels_b_multi)
    if nA == 0 or nB == 0:
        return np.zeros((nA, nB), dtype=int)

    # Common offset range covering both proteins
    max_range = max(nA, nB) - 1
    if max_range < 1:
        max_range = 1
    D = 2 * max_range + 1

    score = np.zeros((nA, nB), dtype=int)

    for slot in range(8):
        if slot in INVARIANT_SLOTS:
            continue

        # Build one-hot boolean matrices: A[i, o+max_range]=1 if offset o
        # is present in octant `slot` at position i of protein A.
        A = np.zeros((nA, D), dtype=np.int8)
        B = np.zeros((nB, D), dtype=np.int8)

        for i in range(nA):
            r = rels_a_multi[i]
            if r is not None:
                for offset in r[slot]:
                    col = offset + max_range
                    if 0 <= col < D:
                        A[i, col] = 1

        for j in range(nB):
            r = rels_b_multi[j]
            if r is not None:
                for offset in r[slot]:
                    col = offset + max_range
                    if 0 <= col < D:
                        B[j, col] = 1

        # (nA, D) @ (D, nB) → (nA, nB): entry = count of common offsets
        matches = A.astype(np.float32) @ B.T.astype(np.float32)
        score += (matches > 0).astype(int)

    return score


# --------------------------------------------------------------------------- #
# sequence mode
# --------------------------------------------------------------------------- #

def match_sequence(seq_a: str, seq_b: str,
                   sigs_a: List[Optional[np.ndarray]],
                   sigs_b: List[Optional[np.ndarray]],
                   min_agreement: float = 0.5,
                   min_pairs: int = 3) -> Tuple[Pairs, dict]:
    """Sequence-anchored matching with TURIST refinement."""
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(seq_a, seq_b)[0]

    raw_pairs: Pairs = []
    for (a_start, a_end), (b_start, b_end) in zip(alignment.aligned[0],
                                                  alignment.aligned[1]):
        for k in range(a_end - a_start):
            raw_pairs.append((a_start + k, b_start + k))

    # TURIST refinement: score each anchored pair, trim low-confidence ones.
    refined: Pairs = []
    agreements: List[float] = []
    for (i, j) in raw_pairs:
        rel_a = relative_offsets(sigs_a[i], i)
        rel_b = relative_offsets(sigs_b[j], j)
        agree = _agreement_fraction(rel_a, rel_b)
        agreements.append(agree)
        if agree >= min_agreement:
            refined.append((i, j))

    used = refined if len(refined) >= min_pairs else raw_pairs
    info = {
        "mode": "sequence",
        "n_sequence_anchors": len(raw_pairs),
        "n_after_trim": len(refined),
        "n_used": len(used),
        "mean_agreement": float(np.mean(agreements)) if agreements else 0.0,
        "fell_back_to_anchors": len(refined) < min_pairs,
    }
    return used, info


# --------------------------------------------------------------------------- #
# exact mode
# --------------------------------------------------------------------------- #

def _exact_match(rel_a, rel_b, min_shared: int = 2) -> bool:
    """True if every (informative) octant occupied in *both* signatures has
    equal relative offset, with at least ``min_shared`` such shared octants."""
    matches, shared = signature_agreement(rel_a, rel_b)
    return shared >= min_shared and matches == shared


def match_exact(sigs_a, sigs_b, min_run: int = 3) -> Tuple[Pairs, dict]:
    """Structural exact matching via maximal diagonal runs of agreeing pairs,
    chained into a single monotonic (non-overlapping) correspondence."""
    nA, nB = len(sigs_a), len(sigs_b)
    rels_a = [relative_offsets(s, i) for i, s in enumerate(sigs_a)]
    rels_b = [relative_offsets(s, j) for j, s in enumerate(sigs_b)]

    match = [[_exact_match(rels_a[i], rels_b[j]) for j in range(nB)]
             for i in range(nA)]

    # Find maximal diagonal runs (only start where the previous diagonal cell
    # is not a match, so each run is recorded once at its true start).
    runs: List[Tuple[int, int, int]] = []  # (i_start, j_start, length)
    for i0 in range(nA):
        for j0 in range(nB):
            if not match[i0][j0]:
                continue
            if i0 > 0 and j0 > 0 and match[i0 - 1][j0 - 1]:
                continue
            k = 0
            while i0 + k < nA and j0 + k < nB and match[i0 + k][j0 + k]:
                k += 1
            if k >= min_run:
                runs.append((i0, j0, k))

    # Chain non-overlapping, monotonic runs to maximise total covered length
    # (weighted interval scheduling over diagonal segments).
    runs.sort(key=lambda r: (r[0], r[1]))
    n = len(runs)
    if n == 0:
        return [], {"mode": "exact", "n_runs": 0, "longest_run": 0, "n_used": 0}

    best = [0] * n
    prev = [-1] * n
    for k in range(n):
        i_s, j_s, L = runs[k]
        best[k] = L
        for m in range(k):
            mi_s, mj_s, mL = runs[m]
            if (mi_s + mL - 1 < i_s) and (mj_s + mL - 1 < j_s):
                if best[m] + L > best[k]:
                    best[k] = best[m] + L
                    prev[k] = m
    end = max(range(n), key=lambda k: best[k])
    chain = []
    while end != -1:
        chain.append(end)
        end = prev[end]
    chain.reverse()

    pairs: Pairs = []
    for k in chain:
        i_s, j_s, L = runs[k]
        for t in range(L):
            pairs.append((i_s + t, j_s + t))

    info = {
        "mode": "exact",
        "n_runs": len(runs),
        "longest_run": max(r[2] for r in runs),
        "n_chained_runs": len(chain),
        "n_used": len(pairs),
    }
    return pairs, info


# --------------------------------------------------------------------------- #
# dp mode
# --------------------------------------------------------------------------- #

def match_dp(sigs_a, sigs_b, gap_penalty: int = -2) -> Tuple[Pairs, dict]:
    """Structural dynamic-programming alignment using octant-agreement scores."""
    nA, nB = len(sigs_a), len(sigs_b)
    rels_a = [relative_offsets(s, i) for i, s in enumerate(sigs_a)]
    rels_b = [relative_offsets(s, j) for j, s in enumerate(sigs_b)]
    S = _precompute_score_matrix(rels_a, rels_b)  # (nA, nB)

    # Needleman-Wunsch DP (integer scores).
    F = np.zeros((nA + 1, nB + 1), dtype=int)
    F[1:, 0] = np.arange(1, nA + 1) * gap_penalty
    F[0, 1:] = np.arange(1, nB + 1) * gap_penalty
    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            diag = F[i - 1, j - 1] + S[i - 1, j - 1]
            up = F[i - 1, j] + gap_penalty
            left = F[i, j - 1] + gap_penalty
            F[i, j] = max(diag, up, left)

    # Traceback (prefer diagonal, then up, then left).
    i, j = nA, nB
    pairs: Pairs = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and F[i, j] == F[i - 1, j - 1] + S[i - 1, j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and F[i, j] == F[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    info = {
        "mode": "dp",
        "alignment_score": int(F[nA, nB]),
        "n_used": len(pairs),
    }
    return pairs, info


# --------------------------------------------------------------------------- #
# local mode (Smith-Waterman)
# --------------------------------------------------------------------------- #

def match_local(sigs_a, sigs_b, gap_penalty: int = -2) -> Tuple[Pairs, dict]:
    """Smith-Waterman local alignment using octant-agreement scores.

    Finds the highest-scoring local region of structural similarity.
    The 0-floor allows the alignment to restart at any position, naturally
    identifying the most structurally similar segment without forcing
    global correspondence.
    """
    nA, nB = len(sigs_a), len(sigs_b)
    rels_a = [relative_offsets(s, i) for i, s in enumerate(sigs_a)]
    rels_b = [relative_offsets(s, j) for j, s in enumerate(sigs_b)]
    S = _precompute_score_matrix(rels_a, rels_b)  # (nA, nB)

    # Smith-Waterman DP: zero-initialized boundaries, 0-floor in recurrence.
    F = np.zeros((nA + 1, nB + 1), dtype=int)
    max_val = 0
    max_i, max_j = 0, 0
    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            diag = F[i - 1, j - 1] + S[i - 1, j - 1]
            up = F[i - 1, j] + gap_penalty
            left = F[i, j - 1] + gap_penalty
            best = max(0, diag, up, left)
            F[i, j] = best
            if best > max_val:
                max_val = best
                max_i, max_j = i, j

    # Traceback from global max, stop at 0.
    i, j = max_i, max_j
    pairs: Pairs = []
    while i > 0 and j > 0 and F[i, j] > 0:
        if F[i, j] == F[i - 1, j - 1] + S[i - 1, j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif F[i, j] == F[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    info = {
        "mode": "local",
        "alignment_score": int(max_val),
        "n_used": len(pairs),
    }
    return pairs, info


# --------------------------------------------------------------------------- #
# semiglobal mode (free end gaps in both sequences)
# --------------------------------------------------------------------------- #

def match_semiglobal(sigs_a, sigs_b, gap_penalty: int = -2) -> Tuple[Pairs, dict]:
    """Semi-global alignment with free end gaps in both sequences.

    No penalty for gaps at the beginning or end of either sequence, but
    internal gaps are penalised. This allows the alignment to extend to
    the termini of one or both proteins while still matching locally
    in the other — useful for proteins of similar size where end-to-end
    alignment is expected but local structural differences exist.
    """
    nA, nB = len(sigs_a), len(sigs_b)
    rels_a = [relative_offsets(s, i) for i, s in enumerate(sigs_a)]
    rels_b = [relative_offsets(s, j) for j, s in enumerate(sigs_b)]
    S = _precompute_score_matrix(rels_a, rels_b)  # (nA, nB)

    # Semi-global DP: zero-initialized boundaries (free end gaps), no 0-floor.
    F = np.zeros((nA + 1, nB + 1), dtype=int)
    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            diag = F[i - 1, j - 1] + S[i - 1, j - 1]
            up = F[i - 1, j] + gap_penalty
            left = F[i, j - 1] + gap_penalty
            F[i, j] = max(diag, up, left)

    # Find best endpoint in last row or last column.
    last_row_best = int(np.argmax(F[nA, 1:])) + 1  # j index
    last_col_best = int(np.argmax(F[1:, nB])) + 1  # i index
    if F[nA, last_row_best] >= F[last_col_best, nB]:
        i, j = nA, last_row_best
        best_score = int(F[nA, last_row_best])
    else:
        i, j = last_col_best, nB
        best_score = int(F[last_col_best, nB])

    # Traceback until reaching row 0 or column 0.
    pairs: Pairs = []
    while i > 0 and j > 0:
        if F[i, j] == F[i - 1, j - 1] + S[i - 1, j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif F[i, j] == F[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    info = {
        "mode": "semiglobal",
        "alignment_score": best_score,
        "n_used": len(pairs),
    }
    return pairs, info


# --------------------------------------------------------------------------- #
# float-DP for iterative refinement (accept external score matrix)
# --------------------------------------------------------------------------- #

def dp_local_float(S: np.ndarray, gap_penalty: float = -2.0) -> Tuple[Pairs, dict]:
    """Smith-Waterman local DP with a pre-built float score matrix.

    Same recurrence as ``match_local`` but accepts an externally computed
    score matrix (e.g. hybrid octant + distance scores from refinement).
    """
    nA, nB = S.shape
    F = np.zeros((nA + 1, nB + 1), dtype=float)
    max_val = 0.0
    max_i, max_j = 0, 0
    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            diag = F[i - 1, j - 1] + S[i - 1, j - 1]
            up = F[i - 1, j] + gap_penalty
            left = F[i, j - 1] + gap_penalty
            best = max(0.0, diag, up, left)
            F[i, j] = best
            if best > max_val:
                max_val = best
                max_i, max_j = i, j

    i, j = max_i, max_j
    pairs: Pairs = []
    while i > 0 and j > 0 and F[i, j] > 0:
        if F[i, j] == F[i - 1, j - 1] + S[i - 1, j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif F[i, j] == F[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    info = {"mode": "local_float", "alignment_score": float(max_val), "n_used": len(pairs)}
    return pairs, info


def dp_semiglobal_float(S: np.ndarray, gap_penalty: float = -2.0) -> Tuple[Pairs, dict]:
    """Semi-global DP with a pre-built float score matrix.

    Same recurrence as ``match_semiglobal`` but accepts an externally
    computed score matrix.
    """
    nA, nB = S.shape
    F = np.zeros((nA + 1, nB + 1), dtype=float)
    for i in range(1, nA + 1):
        for j in range(1, nB + 1):
            diag = F[i - 1, j - 1] + S[i - 1, j - 1]
            up = F[i - 1, j] + gap_penalty
            left = F[i, j - 1] + gap_penalty
            F[i, j] = max(diag, up, left)

    last_row_best = int(np.argmax(F[nA, 1:])) + 1
    last_col_best = int(np.argmax(F[1:, nB])) + 1
    if F[nA, last_row_best] >= F[last_col_best, nB]:
        i, j = nA, last_row_best
        best_score = float(F[nA, last_row_best])
    else:
        i, j = last_col_best, nB
        best_score = float(F[last_col_best, nB])

    pairs: Pairs = []
    while i > 0 and j > 0:
        if F[i, j] == F[i - 1, j - 1] + S[i - 1, j - 1]:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif F[i, j] == F[i - 1, j] + gap_penalty:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    info = {"mode": "semiglobal_float", "alignment_score": best_score, "n_used": len(pairs)}
    return pairs, info


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #

def match(mode: str, seq_a: str, seq_b: str,
          sigs_a, sigs_b, **kwargs) -> Tuple[Pairs, dict]:
    if mode == "sequence":
        return match_sequence(seq_a, seq_b, sigs_a, sigs_b,
                              min_agreement=kwargs.get("min_agreement", 0.5),
                              min_pairs=kwargs.get("min_pairs", 3))
    if mode == "exact":
        return match_exact(sigs_a, sigs_b, min_run=kwargs.get("min_run", 3))
    if mode == "dp":
        return match_dp(sigs_a, sigs_b, gap_penalty=kwargs.get("gap_penalty", -2))
    if mode == "local":
        return match_local(sigs_a, sigs_b, gap_penalty=kwargs.get("gap_penalty", -2))
    if mode == "semiglobal":
        return match_semiglobal(sigs_a, sigs_b, gap_penalty=kwargs.get("gap_penalty", -2))
    raise ValueError(f"Unknown mode {mode!r}; expected 'sequence', 'exact', 'dp', 'local', or 'semiglobal'.")
