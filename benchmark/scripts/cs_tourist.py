#!/usr/bin/env python3
"""Catalytic-Site-Limited TOURIST (CS-TOURIST).

Three modes for aligning proteins focused on catalytic sites:
  Mode A: Neighborhood extraction — align only residues near catalytic residues
  Mode B: Catalytic-site-centered frame — descriptor centered on catalytic site geometry
  Mode C: Catalytic-only alignment — align only catalytic residues to each other

Author: TOURIST project
"""

import numpy as np
from typing import List, Optional, Tuple, Set
import sys

sys.path.insert(0, '/mnt/shared-workspace/tourist')
sys.path.insert(0, '/mnt/results/benchmark/scripts')

from tourist.descriptor import (
    build_local_frame,
    compute_signature_radial,
    compute_signature_radial_3sphere,
    compute_signature_connectivity_aware,
    build_all_signatures_radial,
    build_all_signatures_radial_3sphere,
    build_all_signatures_connectivity_aware,
    relative_offsets_multi,
    RADIAL_MATCH_SLOTS,
    RADIAL_MATCH_SLOTS_3SPHERE,
    CONNECTIVITY_MATCH_SLOTS,
    EMPTY,
)
from tourist.superpose import kabsch, apply_transform
from tourist_fast import dp_local_float, dp_local_affine, warmup


# BLOSUM62 matrix
AA_ORDER = 'ARNDCQEGHILKMFPSTWYV'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
BLOSUM62 = np.array([
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-3],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-3,-2, 1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-3],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2,-3, 3,-3],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0, 1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,-3,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-3,-2,-2, 0,-3,-1, 4],
], dtype=np.float32)


def compute_d0(L):
    if L < 15:
        return 0.5
    return max(0.5, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def seq_to_indices(seq):
    arr = np.full(len(seq), -1, dtype=np.int32)
    for i, aa in enumerate(seq):
        idx = AA_TO_IDX.get(aa, -1)
        if idx >= 0:
            arr[i] = idx
    return arr


def compute_blosum_score_matrix(seq_a, seq_b, nA, nB):
    idx_a = seq_to_indices(seq_a)
    idx_b = seq_to_indices(seq_b)
    S = np.zeros((nA, nB), dtype=np.float32)
    mask_a = idx_a >= 0
    mask_b = idx_b >= 0
    if mask_a.any() and mask_b.any():
        sub = BLOSUM62[np.ix_(idx_a[mask_a], idx_b[mask_b])]
        S[np.ix_(np.where(mask_a)[0], np.where(mask_b)[0])] = sub
    return S


def build_compact_signatures(coords, radius, n_slots=16, match_slots=None):
    """Build compact per-slot signatures for a chain.

    Returns list of (positions, offsets) tuples indexed by slot.
    positions: residue indices that have a match in this slot
    offsets: corresponding relative offsets
    """
    sigs = build_all_signatures_radial(coords, radius)
    n = len(coords)
    compact = []
    for slot in range(n_slots):
        if match_slots is not None and slot not in match_slots:
            compact.append((np.array([], dtype=np.int32), np.array([], dtype=np.int32)))
            continue
        positions = []
        offsets = []
        for ii in range(n):
            rel = relative_offsets_multi(sigs[ii], ii)
            if rel is not None and slot < len(rel):
                for off in rel[slot]:
                    positions.append(ii)
                    offsets.append(off)
        compact.append((np.array(positions, dtype=np.int32), np.array(offsets, dtype=np.int32)))
    return compact


def compute_radial_score_matrix(compact_a, compact_b, nA, nB, match_slots):
    max_range = max(nA, nB) - 1
    if max_range < 1:
        max_range = 1
    D = 2 * max_range + 1
    score = np.zeros((nA, nB), dtype=np.float32)
    for slot in match_slots:
        pos_a, off_a = compact_a[slot]
        pos_b, off_b = compact_b[slot]
        if len(pos_a) == 0 or len(pos_b) == 0:
            continue
        A = np.zeros((nA, D), dtype=np.float32)
        A[pos_a, off_a + max_range] = 1.0
        B = np.zeros((nB, D), dtype=np.float32)
        B[pos_b, off_b + max_range] = 1.0
        matches = A @ B.T
        score += (matches > 0).astype(np.float32)
    return score


# ---------------------------------------------------------------------------
# Mode A: Neighborhood Extraction
# ---------------------------------------------------------------------------

def extract_catalytic_neighborhood(trace, cat_indices, radius=10.0):
    """Extract residues within `radius` Angstroms of any catalytic residue.

    Returns (sub_coords, sub_seq, sub_indices, sub_cat_indices)
    where sub_indices maps sub-trace positions back to original trace indices.
    """
    coords = trace.coords
    n = len(coords)

    if len(cat_indices) == 0:
        return np.array([]), '', np.array([], dtype=int), np.array([], dtype=int)

    cat_coords = coords[list(cat_indices)]
    diffs = coords[:, None, :] - cat_coords[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    min_dists = dists.min(axis=1)

    neighborhood_mask = min_dists <= radius
    sub_indices = np.where(neighborhood_mask)[0]

    if len(sub_indices) < 3:
        return np.array([]), '', np.array([], dtype=int), np.array([], dtype=int)

    sub_coords = coords[sub_indices]
    sub_seq = ''.join(trace.sequence[i] for i in sub_indices)

    idx_map = {orig: sub for sub, orig in enumerate(sub_indices)}
    sub_cat_indices = np.array([idx_map[c] for c in cat_indices if c in idx_map], dtype=int)

    return sub_coords, sub_seq, sub_indices, sub_cat_indices


class SubTrace:
    """Minimal trace-like object for sub-traces."""
    def __init__(self, coords, sequence, res_ids):
        self.coords = coords
        self.sequence = sequence
        self.res_ids = res_ids


def run_cs_tourist_mode_a(trace_a, trace_b, cat_a, cat_b,
                         neighborhood_radius=10.0,
                         descriptor_radius=8.0,
                         refined=True, beta=0.5,
                         gap_penalty=-2.0,
                         gap_open=None, gap_extend=-1.0):
    """Mode A: Neighborhood extraction alignment.

    Returns (correspondence_pairs, rmsd, n_aligned, tm_like, sub_info)
    where correspondence_pairs are in original trace indices.
    """
    sub_coords_a, sub_seq_a, sub_idx_a, sub_cat_a = extract_catalytic_neighborhood(
        trace_a, cat_a, neighborhood_radius)
    sub_coords_b, sub_seq_b, sub_idx_b, sub_cat_b = extract_catalytic_neighborhood(
        trace_b, cat_b, neighborhood_radius)

    if len(sub_coords_a) < 3 or len(sub_coords_b) < 3:
        return [], 0, 0, 0.0, {'n_sub_a': len(sub_coords_a), 'n_sub_b': len(sub_coords_b)}

    nA, nB = len(sub_coords_a), len(sub_coords_b)
    L_query = nA

    sub_res_ids_a = [trace_a.res_ids[i] for i in sub_idx_a]
    sub_res_ids_b = [trace_b.res_ids[i] for i in sub_idx_b]

    # Build compact signatures in correct per-slot format
    compact_a = build_compact_signatures(sub_coords_a, descriptor_radius,
                                         n_slots=16, match_slots=set(RADIAL_MATCH_SLOTS))
    compact_b = build_compact_signatures(sub_coords_b, descriptor_radius,
                                         n_slots=16, match_slots=set(RADIAL_MATCH_SLOTS))

    S_octant = compute_radial_score_matrix(compact_a, compact_b, nA, nB, RADIAL_MATCH_SLOTS)
    S_base = S_octant
    if beta > 0:
        S_blosum = compute_blosum_score_matrix(sub_seq_a, sub_seq_b, nA, nB)
        S_base = S_base + beta * S_blosum

    S_float = S_base.astype(float)

    def do_dp(S):
        if gap_open is not None:
            return dp_local_affine(S, gap_open=gap_open, gap_extend=gap_extend)
        else:
            return dp_local_float(S, gap_penalty=gap_penalty)

    if not refined:
        pairs, _ = do_dp(S_float)
        if len(pairs) < 3:
            return [], 0, 0, 0.0, {'n_sub_a': nA, 'n_sub_b': nB}
        a_idx = np.array([p[0] for p in pairs])
        b_idx = np.array([p[1] for p in pairs])
        R, t, rmsd = kabsch(sub_coords_b[b_idx], sub_coords_a[a_idx])
        n_aligned = len(pairs)
    else:
        d0 = compute_d0(min(nA, nB))
        d0_q = compute_d0(L_query)
        pairs, _ = do_dp(S_float)
        if len(pairs) < 3:
            return [], 0, 0, 0.0, {'n_sub_a': nA, 'n_sub_b': nB}
        a_idx = np.array([p[0] for p in pairs])
        b_idx = np.array([p[1] for p in pairs])
        R, t, rmsd = kabsch(sub_coords_b[b_idx], sub_coords_a[a_idx])
        best_tm = (len(pairs) / L_query) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))
        best_pairs = pairs
        best_rmsd = rmsd

        for iteration in range(1, 6):
            transformed_b = apply_transform(sub_coords_b, R, t)
            diffs = sub_coords_a[:, None, :] - transformed_b[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            dscore = 1.0 / (1.0 + (distances / d0) ** 2)
            S_hybrid = S_float + 1.0 * dscore * 7.0
            pairs_new, _ = do_dp(S_hybrid)
            if len(pairs_new) < 3:
                break
            a_idx = np.array([p[0] for p in pairs_new])
            b_idx = np.array([p[1] for p in pairs_new])
            R_new, t_new, rmsd_new = kabsch(sub_coords_b[b_idx], sub_coords_a[a_idx])
            tm_new = (len(pairs_new) / L_query) * (1.0 / (1.0 + (rmsd_new / d0_q) ** 2))
            if tm_new > best_tm:
                improvement = tm_new - best_tm
                best_tm = tm_new
                best_pairs = pairs_new
                best_rmsd = rmsd_new
                R, t = R_new, t_new
            else:
                R, t = R_new, t_new
                improvement = 0.0
            if improvement < 0.001:
                break

        pairs = best_pairs
        rmsd = best_rmsd
        n_aligned = len(pairs)

    d0_q = compute_d0(L_query)
    tm_like = (n_aligned / L_query) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))

    orig_pairs = [(int(sub_idx_a[a]), int(sub_idx_b[b])) for a, b in pairs]

    sub_info = {
        'n_sub_a': nA, 'n_sub_b': nB,
        'sub_cat_a': sub_cat_a.tolist(), 'sub_cat_b': sub_cat_b.tolist(),
        'sub_idx_a': sub_idx_a.tolist(), 'sub_idx_b': sub_idx_b.tolist(),
    }

    return orig_pairs, rmsd, n_aligned, tm_like, sub_info


# ---------------------------------------------------------------------------
# Mode B: Catalytic-Site-Centered Frame
# ---------------------------------------------------------------------------

def build_catalytic_frame(coords, cat_indices):
    """Build a local coordinate frame centered on the catalytic site.

    Origin = centroid of catalytic CA atoms.
    Axes = principal axes from SVD of centered catalytic CA coordinates.

    Returns (3x3 array of unit row vectors, centroid) or None if degenerate.
    """
    if len(cat_indices) < 2:
        return None

    cat_coords = coords[list(cat_indices)]
    centroid = cat_coords.mean(axis=0)
    centered = cat_coords - centroid

    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    x = Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12)
    if len(Vt) > 1:
        y = Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
    else:
        y = np.cross(x, np.array([0, 0, 1.0]))
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(x, np.array([0, 1.0, 0]))
        y = y / (np.linalg.norm(y) + 1e-12)

    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)

    if np.dot(np.cross(x, y), z) < 0:
        y = -y

    return np.vstack([x, y, z]), centroid


def compute_catalytic_site_signature(coords, cat_indices, radius=12.0, r1=4.0, r2=8.0):
    """Compute catalytic-site-centered 3-sphere signature.

    Unlike per-residue descriptors, this produces a SINGLE signature for the
    entire protein, centered on the catalytic site.

    Returns (24-element list of lists) where each slot contains residue indices
    in that octant x zone, or None if frame is degenerate.
    """
    frame_result = build_catalytic_frame(coords, cat_indices)
    if frame_result is None:
        return None
    frame, centroid = frame_result

    n = len(coords)
    diffs = coords - centroid
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    mask = (dists <= radius) & (dists > 1e-6)

    local = diffs @ frame.T

    # Octant assignment (1-based)
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    # Radial zones
    zone = np.zeros(n, dtype=int)
    zone[dists >= r1] = 1
    zone[dists >= r2] = 2

    # Build 24 slots: slot = (octant-1) * 3 + zone
    sig = [[] for _ in range(24)]
    for i in range(n):
        if not mask[i]:
            continue
        slot = (int(octants[i]) - 1) * 3 + int(zone[i])
        sig[slot].append(i)

    for slot in range(24):
        if sig[slot]:
            indices = np.array(sig[slot])
            order = np.argsort(dists[indices])
            sig[slot] = indices[order].tolist()

    return sig


def catalytic_site_signature_match(sig_a, sig_b, match_slots, nA, nB):
    """Compute match score between two catalytic site signatures.

    Returns a score matrix (nA, nB) where entry [i,j] is incremented
    if residues i and j appear in matching slots.
    """
    score = np.zeros((nA, nB), dtype=np.float32)
    for slot in match_slots:
        residues_a = sig_a[slot]
        residues_b = sig_b[slot]
        if not residues_a or not residues_b:
            continue
        for ra in residues_a:
            for rb in residues_b:
                score[ra, rb] += 1.0
    return score


def run_cs_tourist_mode_b(trace_a, trace_b, cat_a, cat_b,
                         radius=12.0, r1=4.0, r2=8.0,
                         refined=True, beta=0.5,
                         gap_penalty=-2.0,
                         gap_open=None, gap_extend=-1.0):
    """Mode B: Catalytic-site-centered frame alignment.

    Returns (correspondence_pairs, rmsd, n_aligned, tm_like, info)
    """
    coords_a = trace_a.coords
    coords_b = trace_b.coords
    nA, nB = len(coords_a), len(coords_b)
    L_query = nA

    sig_a = compute_catalytic_site_signature(coords_a, cat_a, radius, r1, r2)
    sig_b = compute_catalytic_site_signature(coords_b, cat_b, radius, r1, r2)

    if sig_a is None or sig_b is None:
        return [], 0, 0, 0.0, {'error': 'degenerate_frame'}

    S_sig = catalytic_site_signature_match(sig_a, sig_b, RADIAL_MATCH_SLOTS_3SPHERE, nA, nB)
    S_base = S_sig
    if beta > 0:
        S_blosum = compute_blosum_score_matrix(trace_a.sequence, trace_b.sequence, nA, nB)
        S_base = S_base + beta * S_blosum

    S_float = S_base.astype(float)

    def do_dp(S):
        if gap_open is not None:
            return dp_local_affine(S, gap_open=gap_open, gap_extend=gap_extend)
        else:
            return dp_local_float(S, gap_penalty=gap_penalty)

    if not refined:
        pairs, _ = do_dp(S_float)
        if len(pairs) < 3:
            return [], 0, 0, 0.0, {'error': 'too_few_pairs'}
        a_idx = np.array([p[0] for p in pairs])
        b_idx = np.array([p[1] for p in pairs])
        R, t, rmsd = kabsch(coords_b[b_idx], coords_a[a_idx])
        n_aligned = len(pairs)
    else:
        d0 = compute_d0(min(nA, nB))
        d0_q = compute_d0(L_query)
        pairs, _ = do_dp(S_float)
        if len(pairs) < 3:
            return [], 0, 0, 0.0, {'error': 'too_few_pairs'}
        a_idx = np.array([p[0] for p in pairs])
        b_idx = np.array([p[1] for p in pairs])
        R, t, rmsd = kabsch(coords_b[b_idx], coords_a[a_idx])
        best_tm = (len(pairs) / L_query) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))
        best_pairs = pairs
        best_rmsd = rmsd

        for iteration in range(1, 6):
            transformed_b = apply_transform(coords_b, R, t)
            diffs = coords_a[:, None, :] - transformed_b[None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            dscore = 1.0 / (1.0 + (distances / d0) ** 2)
            S_hybrid = S_float + 1.0 * dscore * 7.0
            pairs_new, _ = do_dp(S_hybrid)
            if len(pairs_new) < 3:
                break
            a_idx = np.array([p[0] for p in pairs_new])
            b_idx = np.array([p[1] for p in pairs_new])
            R_new, t_new, rmsd_new = kabsch(coords_b[b_idx], coords_a[a_idx])
            tm_new = (len(pairs_new) / L_query) * (1.0 / (1.0 + (rmsd_new / d0_q) ** 2))
            if tm_new > best_tm:
                improvement = tm_new - best_tm
                best_tm = tm_new
                best_pairs = pairs_new
                best_rmsd = rmsd_new
                R, t = R_new, t_new
            else:
                R, t = R_new, t_new
                improvement = 0.0
            if improvement < 0.001:
                break

        pairs = best_pairs
        rmsd = best_rmsd
        n_aligned = len(pairs)

    d0_q = compute_d0(L_query)
    tm_like = (n_aligned / L_query) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))

    return list(pairs), rmsd, n_aligned, tm_like, {'n_cat_a': len(cat_a), 'n_cat_b': len(cat_b)}


# ---------------------------------------------------------------------------
# Mode C: Catalytic-Only Alignment
# ---------------------------------------------------------------------------

def run_cs_tourist_mode_c(trace_a, trace_b, cat_a, cat_b,
                         descriptor_radius=8.0,
                         refined=True, beta=0.5,
                         gap_penalty=-2.0,
                         gap_open=None, gap_extend=-1.0,
                         dist_constraint_weight=2.0,
                         use_connectivity=False,
                         conn_radius=12.0, conn_r1=4.0, conn_r2=8.0,
                         w_conn=1.0):
    """Mode C: Catalytic-only alignment.

    Compute descriptors only at catalytic residues, align catalytic-to-catalytic.
    Adds distance constraint: penalize matches where inter-catalytic-residue
    distances differ between query and target.

    When use_connectivity=True, uses the connectivity-aware descriptor (72
    sub-slots: 24 spatial x 3 connectivity classes) with class-presence
    matching. If w_conn > 0, combines with 3-sphere exact-offset matching
    (dual-channel): S = S_exact + w_conn * S_conn. This captures the
    connectivity pattern (local/medium/long-range) of neighbours around
    each catalytic residue, which is more tolerant of insertions/deletions
    than exact-offset matching alone.

    Returns (correspondence_pairs, rmsd, n_aligned, tm_like, info)
    """
    coords_a = trace_a.coords
    coords_b = trace_b.coords

    cat_a_list = sorted(cat_a)
    cat_b_list = sorted(cat_b)
    nA, nB = len(cat_a_list), len(cat_b_list)

    if nA < 2 or nB < 2:
        return [], 0, 0, 0.0, {'error': 'too_few_cat', 'n_cat_a': nA, 'n_cat_b': nB}

    L_query = nA

    if use_connectivity:
        # --- Connectivity-aware descriptor path ---
        # Compute connectivity-aware signatures (72 sub-slots) at catalytic residues
        sigs_a_conn = []
        for ci in cat_a_list:
            sig = compute_signature_connectivity_aware(
                coords_a, ci, conn_radius, conn_r1, conn_r2) if ci >= 2 else None
            sigs_a_conn.append(sig)

        sigs_b_conn = []
        for ci in cat_b_list:
            sig = compute_signature_connectivity_aware(
                coords_b, ci, conn_radius, conn_r1, conn_r2) if ci >= 2 else None
            sigs_b_conn.append(sig)

        # Class-presence matching: for each pair (i,j), count sub-slots
        # where both residues have >=1 neighbour (binary presence)
        S_conn = np.zeros((nA, nB), dtype=np.float32)
        for i in range(nA):
            if sigs_a_conn[i] is None:
                continue
            rel_a = relative_offsets_multi(sigs_a_conn[i], cat_a_list[i])
            if rel_a is None:
                continue
            presence_a = set()
            for sub_slot in CONNECTIVITY_MATCH_SLOTS:
                if sub_slot < len(rel_a) and len(rel_a[sub_slot]) > 0:
                    presence_a.add(sub_slot)
            if not presence_a:
                continue
            for j in range(nB):
                if sigs_b_conn[j] is None:
                    continue
                rel_b = relative_offsets_multi(sigs_b_conn[j], cat_b_list[j])
                if rel_b is None:
                    continue
                presence_b = set()
                for sub_slot in CONNECTIVITY_MATCH_SLOTS:
                    if sub_slot < len(rel_b) and len(rel_b[sub_slot]) > 0:
                        presence_b.add(sub_slot)
                if not presence_b:
                    continue
                S_conn[i, j] = len(presence_a & presence_b)

        if w_conn > 0:
            # Dual-channel: add 3-sphere exact-offset matching
            S_exact = np.zeros((nA, nB), dtype=np.float32)
            for i in range(nA):
                ci_a = cat_a_list[i]
                if ci_a < 2:
                    continue
                sig_a = compute_signature_radial_3sphere(
                    coords_a, ci_a, conn_radius, conn_r1, conn_r2)
                if sig_a is None:
                    continue
                rel_a = relative_offsets_multi(sig_a, ci_a)
                if rel_a is None:
                    continue
                sig_set_a = set()
                for slot in RADIAL_MATCH_SLOTS_3SPHERE:
                    if slot < len(rel_a):
                        for off in rel_a[slot]:
                            sig_set_a.add((slot, off))
                if not sig_set_a:
                    continue
                for j in range(nB):
                    ci_b = cat_b_list[j]
                    if ci_b < 2:
                        continue
                    sig_b = compute_signature_radial_3sphere(
                        coords_b, ci_b, conn_radius, conn_r1, conn_r2)
                    if sig_b is None:
                        continue
                    rel_b = relative_offsets_multi(sig_b, ci_b)
                    if rel_b is None:
                        continue
                    sig_set_b = set()
                    for slot in RADIAL_MATCH_SLOTS_3SPHERE:
                        if slot < len(rel_b):
                            for off in rel_b[slot]:
                                sig_set_b.add((slot, off))
                    if not sig_set_b:
                        continue
                    S_exact[i, j] = len(sig_set_a & sig_set_b)
            S_base = S_exact + w_conn * S_conn
        else:
            S_base = S_conn
    else:
        # --- Original 2-sphere descriptor path ---
        # Compute descriptors at catalytic residues only
        sigs_a = []
        for ci in cat_a_list:
            if ci >= 2:
                sig = compute_signature_radial(coords_a, ci, descriptor_radius)
            else:
                sig = None
            sigs_a.append(sig)

        sigs_b = []
        for ci in cat_b_list:
            if ci >= 2:
                sig = compute_signature_radial(coords_b, ci, descriptor_radius)
            else:
                sig = None
            sigs_b.append(sig)

        # Direct descriptor comparison for catalytic residues
        # For each pair (i, j), count matching (slot, offset) pairs
        S_octant = np.zeros((nA, nB), dtype=np.float32)
        for i in range(nA):
            if sigs_a[i] is None:
                continue
            rel_a = relative_offsets_multi(sigs_a[i], cat_a_list[i])
            if rel_a is None:
                continue
            # Build set of (slot, offset) for residue i
            sig_set_a = set()
            for slot in RADIAL_MATCH_SLOTS:
                if slot < len(rel_a):
                    for off in rel_a[slot]:
                        sig_set_a.add((slot, off))
            if not sig_set_a:
                continue
            for j in range(nB):
                if sigs_b[j] is None:
                    continue
                rel_b = relative_offsets_multi(sigs_b[j], cat_b_list[j])
                if rel_b is None:
                    continue
                sig_set_b = set()
                for slot in RADIAL_MATCH_SLOTS:
                    if slot < len(rel_b):
                        for off in rel_b[slot]:
                            sig_set_b.add((slot, off))
                if not sig_set_b:
                    continue
                # Count matching (slot, offset) pairs
                S_octant[i, j] = len(sig_set_a & sig_set_b)

        S_base = S_octant

    if beta > 0:
        cat_seq_a = ''.join(trace_a.sequence[i] for i in cat_a_list)
        cat_seq_b = ''.join(trace_b.sequence[i] for i in cat_b_list)
        S_blosum = compute_blosum_score_matrix(cat_seq_a, cat_seq_b, nA, nB)
        S_base = S_base + beta * S_blosum

    # Distance constraint: penalize matches where inter-catalytic distances differ
    if dist_constraint_weight > 0 and nA >= 2 and nB >= 2:
        cat_coords_a = coords_a[cat_a_list]
        cat_coords_b = coords_b[cat_b_list]
        dist_a = np.sqrt(np.sum((cat_coords_a[:, None] - cat_coords_a[None, :]) ** 2, axis=2))
        dist_b = np.sqrt(np.sum((cat_coords_b[:, None] - cat_coords_b[None, :]) ** 2, axis=2))

        S_dist = np.zeros((nA, nB), dtype=np.float32)
        for i in range(nA):
            for j in range(nB):
                score = 0.0
                for k in range(nA):
                    if k == i:
                        continue
                    da_ik = dist_a[i, k]
                    best_compat = 0.0
                    for jp in range(nB):
                        if jp == j:
                            continue
                        db_jjp = dist_b[j, jp]
                        diff = abs(da_ik - db_jjp)
                        compat = max(0, 1.0 - diff / 3.0)
                        if compat > best_compat:
                            best_compat = compat
                    score += best_compat
                S_dist[i, j] = score / max(1, nA - 1)
        S_base = S_base + dist_constraint_weight * S_dist

    S_float = S_base.astype(float)

    def do_dp(S):
        if gap_open is not None:
            return dp_local_affine(S, gap_open=gap_open, gap_extend=gap_extend)
        else:
            return dp_local_float(S, gap_penalty=gap_penalty)

    pairs, _ = do_dp(S_float)

    if len(pairs) < 2:
        return [], 0, 0, 0.0, {'error': 'too_few_pairs', 'n_cat_a': nA, 'n_cat_b': nB}

    orig_pairs = [(cat_a_list[a], cat_b_list[b]) for a, b in pairs]

    a_idx = np.array([p[0] for p in orig_pairs])
    b_idx = np.array([p[1] for p in orig_pairs])
    R, t, rmsd = kabsch(coords_b[b_idx], coords_a[a_idx])
    n_aligned = len(orig_pairs)

    if refined and n_aligned >= 3:
        d0 = compute_d0(min(nA, nB))
        d0_q = compute_d0(L_query)
        best_tm = (n_aligned / L_query) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))
        best_pairs = orig_pairs
        best_rmsd = rmsd

        for iteration in range(1, 4):
            transformed_b = apply_transform(coords_b, R, t)
            diffs = coords_a[cat_a_list][:, None, :] - transformed_b[cat_b_list][None, :, :]
            distances = np.sqrt(np.sum(diffs * diffs, axis=2))
            dscore = 1.0 / (1.0 + (distances / d0) ** 2)
            S_hybrid = S_float + 1.0 * dscore * 7.0
            pairs_new, _ = do_dp(S_hybrid)
            if len(pairs_new) < 2:
                break
            orig_pairs_new = [(cat_a_list[a], cat_b_list[b]) for a, b in pairs_new]
            a_idx = np.array([p[0] for p in orig_pairs_new])
            b_idx = np.array([p[1] for p in orig_pairs_new])
            R_new, t_new, rmsd_new = kabsch(coords_b[b_idx], coords_a[a_idx])
            tm_new = (len(pairs_new) / L_query) * (1.0 / (1.0 + (rmsd_new / d0_q) ** 2))
            if tm_new > best_tm:
                improvement = tm_new - best_tm
                best_tm = tm_new
                best_pairs = orig_pairs_new
                best_rmsd = rmsd_new
                R, t = R_new, t_new
            else:
                R, t = R_new, t_new
                improvement = 0.0
            if improvement < 0.001:
                break

        orig_pairs = best_pairs
        rmsd = best_rmsd
        n_aligned = len(orig_pairs)

    d0_q = compute_d0(max(nA, nB))
    tm_like = (n_aligned / max(nA, nB)) * (1.0 / (1.0 + (rmsd / d0_q) ** 2))

    return orig_pairs, rmsd, n_aligned, tm_like, {'n_cat_a': nA, 'n_cat_b': nB}
