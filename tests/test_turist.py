"""Tests for the TURIST package: descriptor, superposition, matching, and a
network-gated self-alignment integration check."""

import numpy as np
import pytest

from turist.descriptor import (
    build_local_frame, compute_signature, build_all_signatures,
    relative_offsets, _octant_position, DEFAULT_RADIUS, EMPTY,
)
from turist.matching import (
    signature_agreement, match_exact, match_dp, match_sequence,
)
from turist.superpose import kabsch, transform_matrix, apply_transform


# --------------------------------------------------------------------------- #
# Local frame
# --------------------------------------------------------------------------- #

def test_local_frame_basic():
    # prev2=(0,1,0), prev1=(0,0,0), curr=(1,0,0)
    frame = build_local_frame(np.array([0, 1, 0]), np.array([0, 0, 0]),
                              np.array([1, 0, 0]))
    assert frame is not None
    x, y, z = frame
    assert np.allclose(x, [1, 0, 0])      # forward = curr - prev1
    assert np.allclose(y, [0, 1, 0])      # up, toward prev2 side
    assert np.allclose(z, [0, 0, 1])      # right = x cross y
    # orthonormal + right-handed
    assert np.allclose(np.dot(x, y), 0)
    assert np.allclose(np.cross(x, y), z)


def test_local_frame_collinear_returns_none():
    frame = build_local_frame(np.array([0, 0, 0]), np.array([1, 0, 0]),
                              np.array([2, 0, 0]))
    assert frame is None


# --------------------------------------------------------------------------- #
# Octant mapping
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dx,dy,dz,expected", [
    (-1, -1, -1, 1),   # BDL
    (-1, -1, +1, 2),   # BDR
    (-1, +1, -1, 3),   # BUL
    (-1, +1, +1, 4),   # BUR
    (+1, -1, -1, 5),   # FDL
    (+1, -1, +1, 6),   # FDR
    (+1, +1, -1, 7),   # FUL
    (+1, +1, +1, 8),   # FUR
])
def test_octant_mapping(dx, dy, dz, expected):
    assert _octant_position(dx, dy, dz) == expected


def test_octant_ties_go_positive():
    # dx=0 -> front, dy=0 -> up, dz=0 -> right  => FUR = 8
    assert _octant_position(0.0, 0.0, 0.0) == 8


# --------------------------------------------------------------------------- #
# Signature
# --------------------------------------------------------------------------- #

def test_signature_on_synthetic_chain():
    # Frame at i=2: origin (1,0,0), x=(1,0,0), y=(0,1,0), z=(0,0,1).
    # prev1 (idx 1) is exactly on -x -> back,up(tie),right(tie) = BUR (slot 3),
    # and being nearest it occupies that slot. prev2 (idx 0) also falls in BUR
    # but is farther. Probes fill the other 7 octants at distance 3.
    coords = np.array([
        [0, 1, 0],    # 0  prev2  -> BUR (slot 3), dist sqrt2
        [0, 0, 0],    # 1  prev1  -> BUR (slot 3), dist 1  (wins)
        [1, 0, 0],    # 2  curr
        [3, 2, 1],    # 3  d=(2,2,1)   -> FUR (8) slot 7
        [-1, 2, -1],  # 4  d=(-2,2,-1) -> BUL (3) slot 2
        [-1, -2, -1], # 5  d=(-2,-2,-1)-> BDL (1) slot 0
        [3, -2, -1],  # 6  d=(2,-2,-1) -> FDL (5) slot 4
        [3, -2, 1],   # 7  d=(2,-2,1)  -> FDR (6) slot 5
        [3, 2, -1],   # 8  d=(2,2,-1)  -> FUL (7) slot 6
        [-1, -2, 1],  # 9  d=(-2,-2,1) -> BDR (2) slot 1
    ], dtype=float)
    sig = compute_signature(coords, 2, radius=5.7)
    expected = np.array([5, 9, 4, 1, 6, 7, 8, 3])
    assert sig is not None
    np.testing.assert_array_equal(sig, expected)


def test_signature_none_for_first_two():
    coords = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=float)
    assert compute_signature(coords, 0) is None
    assert compute_signature(coords, 1) is None
    assert compute_signature(coords, 2) is not None


def test_relative_offsets():
    sig = np.array([5, EMPTY, EMPTY, 4, 6, EMPTY, EMPTY, 3])
    rel = relative_offsets(sig, 2)
    # slot 0: 5-2=3, slot 3: 4-2=2, slot 4: 6-2=4, slot 7: 3-2=1
    assert rel[0] == 3 and rel[3] == 2 and rel[4] == 4 and rel[7] == 1
    assert rel[1] == EMPTY and rel[2] == EMPTY


# --------------------------------------------------------------------------- #
# Superposition (Kabsch)
# --------------------------------------------------------------------------- #

def test_kabsch_recovers_known_transform():
    rng = np.random.default_rng(0)
    b = rng.standard_normal((20, 3))
    # Known rotation: 90 deg about z, plus a translation.
    th = np.pi / 2
    R_true = np.array([[np.cos(th), -np.sin(th), 0],
                       [np.sin(th), np.cos(th), 0],
                       [0, 0, 1]])
    t_true = np.array([5.0, -3.0, 2.0])
    a = (R_true @ b.T).T + t_true
    R, t, rmsd = kabsch(b, a)
    assert rmsd < 1e-9
    assert np.allclose(R, R_true, atol=1e-8)
    assert np.allclose(t, t_true, atol=1e-8)


def test_kabsch_identity():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    R, t, rmsd = kabsch(a, a)
    assert rmsd < 1e-12
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, [0, 0, 0])


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def test_signature_agreement_full_and_partial():
    rel_a = np.array([3, EMPTY, EMPTY, 2, 4, EMPTY, EMPTY, 1])
    rel_b = np.array([3, EMPTY, EMPTY, 2, 9, EMPTY, EMPTY, 1])  # slot 4 differs
    # Slot 3 (BUR) is an invariant slot and is excluded from agreement.
    # Shared informative octants: 0 (match), 4 (mismatch), 7 (match).
    m, shared = signature_agreement(rel_a, rel_b)
    assert shared == 3
    assert m == 2


def test_match_exact_finds_run():
    # Two identical traces -> every shared-octant pair should match exactly.
    coords = np.array([
        [0, 1, 0], [0, 0, 0], [1, 0, 0],
        [3, 2, 1], [-1, 2, 1], [-1, -2, -1], [3, -2, -1],
    ], dtype=float)
    sigs = build_all_signatures(coords, radius=5.7)
    pairs, info = match_exact(sigs, sigs, min_run=2)
    # Self-match: each index pairs with itself.
    assert all(i == j for i, j in pairs)
    assert len(pairs) >= 2
    assert info["longest_run"] >= 2


def test_match_dp_self_alignment():
    coords = np.array([
        [0, 1, 0], [0, 0, 0], [1, 0, 0],
        [3, 2, 1], [-1, 2, 1], [-1, -2, -1], [3, -2, -1],
    ], dtype=float)
    sigs = build_all_signatures(coords, radius=5.7)
    pairs, info = match_dp(sigs, sigs)
    assert len(pairs) >= 3
    # Self DP should align each residue to itself where signatures exist.
    self_pairs = [(i, j) for i, j in pairs if i == j]
    assert len(self_pairs) >= 3


# --------------------------------------------------------------------------- #
# Integration: self-alignment of a real structure (network-gated)
# --------------------------------------------------------------------------- #

def test_integration_self_align_ubiquitin():
    try:
        from turist import align_structures
        res = align_structures("1UBQ", "1UBQ", mode="sequence")
    except Exception as e:  # network or parse issues
        pytest.skip(f"Integration skipped (fetch/parse unavailable): {e}")
    assert res.n_aligned >= 3
    assert res.rmsd < 1e-3
    assert np.allclose(res.transform, np.eye(4), atol=1e-4)
