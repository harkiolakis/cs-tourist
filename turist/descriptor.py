"""TURIST descriptor: trace-defined local coordinate frame and 8-octant
Cα-environment signatures.

For each Cα at chain index ``i`` (i >= 2) a local right-handed frame is built
from the preceding trace, and every neighbouring Cα within a sphere of radius
``R`` is binned into one of 8 octants (back/front x down/up x left/right).
The slot stores the *absolute chain index* of that neighbour (or -1 if empty).

Octant -> array position (1-based, matching the algorithm document):

    BDL=1  BDR=2  BUL=3  BUR=4
    FDL=5  FDR=6  FUL=7  FUR=8

where B/F = back/front (x sign), D/U = down/up (y sign), L/R = left/right
(z sign). Equivalently:  pos = 1 + (front?4:0) + (up?2:0) + (right?1:0).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# Default sphere radius: 1.5 x the typical 3.8 A consecutive-Cα distance.
DEFAULT_RADIUS = 5.7

# Sentinel for an empty octant slot.
EMPTY = -1


def _unit(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along ``v`` (zero vector stays zero)."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(3)
    return v / n


def build_local_frame(ca_prev2: np.ndarray, ca_prev1: np.ndarray,
                      ca_curr: np.ndarray) -> Optional[np.ndarray]:
    """Build the local right-handed frame at ``ca_curr``.

    Args:
        ca_prev2: Cα at index i-2.
        ca_prev1: Cα at index i-1.
        ca_curr:  Cα at index i (frame origin).

    Returns:
        A (3, 3) array whose rows are (x, y, z) unit vectors, or None if the
        three points are collinear and the frame is undefined.
    """
    x = _unit(ca_curr - ca_prev1)                      # forward
    toward_prev2 = ca_prev2 - ca_curr                  # points to the i-2 side
    y_raw = toward_prev2 - np.dot(toward_prev2, x) * x  # perp. component, in plane
    y = _unit(y_raw)
    if np.linalg.norm(y_raw) < 1e-9:
        # Collinear trace: no unique "up"; frame undefined.
        return None
    z = np.cross(x, y)
    z = _unit(z)
    return np.vstack([x, y, z])


def _octant_position(dx: float, dy: float, dz: float) -> int:
    """Map local-frame coordinate signs to a 1-based octant position.

    Axis-plane ties (a component ~ 0) are assigned to the positive side.
    """
    front = dx >= 0.0
    up = dy >= 0.0
    right = dz >= 0.0
    return 1 + (4 if front else 0) + (2 if up else 0) + (1 if right else 0)


def compute_signature(coords: np.ndarray, i: int,
                      radius: float = DEFAULT_RADIUS) -> Optional[np.ndarray]:
    """Compute the 8-element TURIST signature for Cα at index ``i``.

    Returns an int array of length 8 (1-based octant slots stored at indices
    0..7), with EMPTY (-1) for unoccupied octants, or None if the frame is
    undefined (i < 2 or collinear preceding trace).
    """
    n = len(coords)
    if i < 2:
        return None
    frame = build_local_frame(coords[i - 2], coords[i - 1], coords[i])
    if frame is None:
        return None
    x, y, z = frame[0], frame[1], frame[2]

    sig = np.full(8, EMPTY, dtype=int)
    for j in range(n):
        if j == i:
            continue
        d = coords[j] - coords[i]
        dist = np.linalg.norm(d)
        if dist > radius or dist < 1e-6:
            continue
        dx, dy, dz = float(d @ x), float(d @ y), float(d @ z)
        pos = _octant_position(dx, dy, dz)  # 1..8
        slot = pos - 1
        if sig[slot] == EMPTY:
            sig[slot] = j
        else:
            # Two neighbours in one octant (rare at R=5.7 A): keep the nearer.
            if dist < np.linalg.norm(coords[sig[slot]] - coords[i]):
                sig[slot] = j
    return sig


def build_all_signatures(coords: np.ndarray,
                         radius: float = DEFAULT_RADIUS) -> List[Optional[np.ndarray]]:
    """Compute signatures for every Cα. Entry is None where the frame is
    undefined (first two residues, or a collinear stretch)."""
    return [compute_signature(coords, i, radius) for i in range(len(coords))]


def relative_offsets(sig: np.ndarray, i: int) -> np.ndarray:
    """Return per-octant relative offsets (neighbour_index - i), EMPTY where
    the octant is unoccupied. Used for cross-protein matching."""
    if sig is None:
        return np.full(8, EMPTY, dtype=int)
    rel = sig - i
    rel[sig == EMPTY] = EMPTY
    return rel


# --------------------------------------------------------------------------- #
# Multi-residue descriptors (v4: keep ALL residues per octant)
# --------------------------------------------------------------------------- #

def compute_signature_multi(coords: np.ndarray, i: int,
                            radius: float = DEFAULT_RADIUS) -> Optional[List[List[int]]]:
    """Compute multi-residue TURIST signature for Cα at index ``i``.

    Unlike ``compute_signature`` which keeps only the nearest residue per
    octant, this stores ALL residues falling into each octant within radius,
    sorted by distance.

    Returns a list of 8 lists (one per octant slot 0..7), each containing
    the chain indices of residues in that octant sorted by distance, or
    None if the frame is undefined (i < 2 or collinear preceding trace).
    """
    n = len(coords)
    if i < 2:
        return None
    frame = build_local_frame(coords[i - 2], coords[i - 1], coords[i])
    if frame is None:
        return None

    # Vectorised: transform all residues into local frame at once
    diffs = coords - coords[i]                       # (n, 3)
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))   # (n,)
    mask = (dists <= radius) & (dists > 1e-6)
    mask[i] = False
    local = diffs @ frame.T                           # (n, 3)

    # Octant assignment (1-based): 1 + front*4 + up*2 + right*1
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    sig_multi: List[List[int]] = []
    for slot in range(8):
        oct_mask = mask & (octants == slot + 1)
        if not np.any(oct_mask):
            sig_multi.append([])
            continue
        indices = np.where(oct_mask)[0]
        order = np.argsort(dists[indices])
        sig_multi.append(indices[order].tolist())
    return sig_multi


def build_all_signatures_multi(coords: np.ndarray,
                               radius: float = DEFAULT_RADIUS
                               ) -> List[Optional[List[List[int]]]]:
    """Compute multi-residue signatures for every Cα."""
    return [compute_signature_multi(coords, i, radius) for i in range(len(coords))]


def relative_offsets_multi(sig_multi: Optional[List[List[int]]],
                           i: int) -> Optional[List[List[int]]]:
    """Convert multi-residue signature to per-octant relative offsets.

    Returns a list of 8 lists of int offsets (neighbour_index - i), or
    None if ``sig_multi`` is None. Empty list where octant is unoccupied.
    """
    if sig_multi is None:
        return None
    return [[j - i for j in slot_list] for slot_list in sig_multi]


# --------------------------------------------------------------------------- #
# Subdivided octant descriptors (v4 Task 2: radius-independent angular bins)
# --------------------------------------------------------------------------- #

def compute_signature_subdivided(
    coords: np.ndarray, i: int,
    radius: float = 100.0,
    n_sub: int = 9,
) -> Optional[List[List[int]]]:
    """Compute subdivided-octant TURIST signature for Cα at index ``i``.

    Uses a large radius (effectively whole protein) and subdivides each
    octant into ``n_sub`` equal angular subsections (n_sub must be a
    perfect square, e.g. 9 = 3×3 or 16 = 4×4). Within each subsection,
    the nearest residue is kept. This allows up to ``n_sub`` residues
    per octant.

    Angular parameterisation within each octant uses absolute component
    values so that every octant maps to the same [0, π/2] × [0, π/2]
    angular range:
      α = atan2(|dy|, |dx|)          — azimuthal angle in local xy-plane
      β = atan2(|dz|, √(dx²+dy²))    — elevation from local xy-plane

    Returns a list of 8 lists (one per octant slot 0..7), each containing
    up to ``n_sub`` residue indices sorted by distance, or None if the
    frame is undefined.
    """
    n = len(coords)
    if i < 2:
        return None
    frame = build_local_frame(coords[i - 2], coords[i - 1], coords[i])
    if frame is None:
        return None

    n_side = int(round(n_sub ** 0.5))
    if n_side * n_side != n_sub:
        raise ValueError(f"n_sub must be a perfect square, got {n_sub}")

    # Vectorised: transform all residues into local frame
    diffs = coords - coords[i]                       # (n, 3)
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))   # (n,)
    mask = (dists <= radius) & (dists > 1e-6)
    mask[i] = False
    local = diffs @ frame.T                           # (n, 3)

    # Octant assignment (1-based)
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    # Angular coordinates within octant (octant-independent ranges)
    adx = np.abs(local[:, 0])
    ady = np.abs(local[:, 1])
    adz = np.abs(local[:, 2])
    alpha = np.arctan2(ady, adx)                          # [0, π/2]
    beta = np.arctan2(adz, np.sqrt(adx * adx + ady * ady))  # [0, π/2]

    # Bin into n_side × n_side subsections
    half_pi = np.pi / 2.0
    alpha_bin = np.minimum((alpha / half_pi * n_side).astype(int), n_side - 1)
    beta_bin = np.minimum((beta / half_pi * n_side).astype(int), n_side - 1)
    sub_idx = alpha_bin * n_side + beta_bin

    sig_multi: List[List[int]] = []
    for slot in range(8):
        oct_mask = mask & (octants == slot + 1)
        if not np.any(oct_mask):
            sig_multi.append([])
            continue
        indices = np.where(oct_mask)[0]
        subs = sub_idx[indices]
        dists_sub = dists[indices]

        # Keep nearest per subsection: sort by distance, then unique by subsection
        order = np.argsort(dists_sub)
        indices_sorted = indices[order]
        subs_sorted = subs[order]
        _, first_idx = np.unique(subs_sorted, return_index=True)
        nearest = indices_sorted[first_idx]

        # Sort nearest residues by distance for consistent ordering
        nearest_dists = dists[nearest]
        final_order = np.argsort(nearest_dists)
        sig_multi.append(nearest[final_order].tolist())

    return sig_multi


def build_all_signatures_subdivided(
    coords: np.ndarray,
    radius: float = 100.0,
    n_sub: int = 9,
) -> List[Optional[List[List[int]]]]:
    """Compute subdivided-octant signatures for every Cα."""
    return [compute_signature_subdivided(coords, i, radius, n_sub)
            for i in range(len(coords))]


# --------------------------------------------------------------------------- #
# 6-subsegment octant descriptors (v4 Task 2b: spherical-triangle subdivision)
# --------------------------------------------------------------------------- #

def compute_signature_subdivided6(
    coords: np.ndarray, i: int,
    radius: float = 100.0,
) -> Optional[List[List[int]]]:
    """Compute 6-subsegment octant TURIST signature for Cα at index ``i``.

    Each octant's spherical surface is a spherical triangle with 3 arcs
    (edges).  Each arc is split in half, and the 6 half-arcs are connected
    to the centroid of the spherical triangle and to the sphere center
    (reference Cα), forming 6 wedge-shaped sub-segments.

    The 6 sub-segments correspond to the 6 possible rankings of the three
    absolute local-frame components (|dx|, |dy|, |dz|):

      sub 0:  largest |dx|, |dy| >= |dz|   (near x-axis, toward y)
      sub 1:  largest |dy|, |dx| >= |dz|   (near y-axis, toward x)
      sub 2:  largest |dy|, |dz| >  |dx|   (near y-axis, toward z)
      sub 3:  largest |dz|, |dy| >= |dx|   (near z-axis, toward y)
      sub 4:  largest |dz|, |dx| >  |dy|   (near z-axis, toward x)
      sub 5:  largest |dx|, |dz| >  |dy|   (near x-axis, toward z)

    Within each sub-segment, the nearest residue to the reference point
    (sphere center) is kept.  This allows up to 6 residues per octant.

    Returns a list of 8 lists (one per octant slot 0..7), each containing
    up to 6 residue indices sorted by distance, or None if the frame is
    undefined.
    """
    n = len(coords)
    if i < 2:
        return None
    frame = build_local_frame(coords[i - 2], coords[i - 1], coords[i])
    if frame is None:
        return None

    # Vectorised: transform all residues into local frame
    diffs = coords - coords[i]                       # (n, 3)
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))   # (n,)
    mask = (dists <= radius) & (dists > 1e-6)
    mask[i] = False
    local = diffs @ frame.T                           # (n, 3)

    # Octant assignment (1-based)
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    # Absolute values for sub-segment assignment
    adx = np.abs(local[:, 0])
    ady = np.abs(local[:, 1])
    adz = np.abs(local[:, 2])

    # Sub-segment (0-5) from ranking of (adx, ady, adz)
    abs_vals = np.stack([adx, ady, adz], axis=1)     # (n, 3)
    largest = np.argmax(abs_vals, axis=1)             # 0=x, 1=y, 2=z

    sub_idx = np.empty(n, dtype=int)
    m_x = largest == 0
    sub_idx[m_x & (ady >= adz)] = 0
    sub_idx[m_x & (ady <  adz)] = 5
    m_y = largest == 1
    sub_idx[m_y & (adx >= adz)] = 1
    sub_idx[m_y & (adx <  adz)] = 2
    m_z = largest == 2
    sub_idx[m_z & (ady >= adx)] = 3
    sub_idx[m_z & (ady <  adx)] = 4

    sig_multi: List[List[int]] = []
    for slot in range(8):
        oct_mask = mask & (octants == slot + 1)
        if not np.any(oct_mask):
            sig_multi.append([])
            continue
        indices = np.where(oct_mask)[0]
        subs = sub_idx[indices]
        dists_sub = dists[indices]

        # Keep nearest per sub-segment
        order = np.argsort(dists_sub)
        indices_sorted = indices[order]
        subs_sorted = subs[order]
        _, first_idx = np.unique(subs_sorted, return_index=True)
        nearest = indices_sorted[first_idx]

        nearest_dists = dists[nearest]
        final_order = np.argsort(nearest_dists)
        sig_multi.append(nearest[final_order].tolist())

    return sig_multi


def build_all_signatures_subdivided6(
    coords: np.ndarray,
    radius: float = 100.0,
) -> List[Optional[List[List[int]]]]:
    """Compute 6-subsegment octant signatures for every Cα."""
    return [compute_signature_subdivided6(coords, i, radius)
            for i in range(len(coords))]


# --------------------------------------------------------------------------- #
# v5: Side chain volume weights + side chain centroid descriptors
# --------------------------------------------------------------------------- #

# Side chain volumes (Å³) from Zamyatnin (1972), commonly used in ProtScale.
# Used for weighted octant filling: bulkier residues contribute more to
# the match score.
SIDE_CHAIN_VOLUMES = {
    'G': 0.0,   'A': 27.0,  'S': 29.0,  'C': 33.0,  'V': 43.0,
    'T': 44.0,  'I': 47.0,  'P': 48.0,  'L': 48.0,  'N': 52.0,
    'D': 54.0,  'Q': 58.0,  'K': 62.0,  'E': 67.0,  'M': 72.0,
    'H': 76.0,  'F': 82.0,  'R': 94.0,  'Y': 93.0,  'W': 113.0,
}

MAX_SIDE_CHAIN_VOLUME = 113.0  # Trp

# Backbone atom names (excluded when computing side chain centroids)
_BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}


def get_side_chain_volume(residue_code: str) -> float:
    """Return normalized side chain volume for a 1-letter amino acid code.

    Returns 0.0 for unknown residues.  Normalized to [0, 1] by dividing
    by MAX_SIDE_CHAIN_VOLUME.
    """
    return SIDE_CHAIN_VOLUMES.get(residue_code, 0.0) / MAX_SIDE_CHAIN_VOLUME


def parse_pdb_sidechain_atoms(pdb_path: str, chain_id: str = 'A'):
    """Parse side chain heavy atoms from a PDB file, keyed by residue seq number.

    Returns:
        dict: res_seq (int) → list of (x, y, z) side chain atom positions
              (excluding backbone N, CA, C, O, OXT)
    """
    from collections import defaultdict

    three_to_one = {
        'GLY': 'G', 'ALA': 'A', 'SER': 'S', 'CYS': 'C', 'VAL': 'V',
        'THR': 'T', 'ILE': 'I', 'PRO': 'P', 'LEU': 'L', 'ASN': 'N',
        'ASP': 'D', 'GLN': 'Q', 'LYS': 'K', 'GLU': 'E', 'MET': 'M',
        'HIS': 'H', 'PHE': 'F', 'ARG': 'R', 'TYR': 'Y', 'TRP': 'W',
        'MSE': 'M',
    }
    _STANDARD_RES = set(three_to_one.keys())

    sc_atoms_by_res = defaultdict(list)  # res_seq -> [(x, y, z), ...]
    in_first_model = True

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                in_first_model = False
                continue
            if line.startswith('MODEL'):
                model_num = int(line[10:14].strip()) if line[10:14].strip() else 1
                in_first_model = (model_num <= 1)
                continue
            if not in_first_model:
                continue
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            altloc = line[16].strip()
            if altloc not in ('', 'A'):
                continue
            chain = line[21].strip()
            if chain != chain_id:
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            if res_name not in _STANDARD_RES:
                continue
            if atom_name in _BACKBONE_ATOMS:
                continue
            res_seq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else ''
            if element == 'H' or (not element and atom_name[0] == 'H'):
                continue
            sc_atoms_by_res[res_seq].append((x, y, z))

    return dict(sc_atoms_by_res)


def compute_centroids_for_trace(trace, pdb_path: str, chain_id: str = 'A'):
    """Compute side chain centroids aligned to a ChainTrace.

    For each residue in the trace, looks up its side chain atoms from the
    PDB file and computes the centroid.  Glycine or residues with missing
    side chain atoms fall back to Cα.

    Args:
        trace: ChainTrace from parse_chain_trace
        pdb_path: path to the PDB file
        chain_id: chain identifier

    Returns:
        centroids: (n_res, 3) array aligned to trace.coords
    """
    sc_atoms_by_res = parse_pdb_sidechain_atoms(pdb_path, chain_id)
    n = len(trace.coords)
    centroids = np.zeros((n, 3))

    for i in range(n):
        # res_ids[i] = (hetero_flag, resseq, icode)
        res_seq = trace.res_ids[i][1]
        sc_atoms = sc_atoms_by_res.get(res_seq, [])
        if sc_atoms:
            centroids[i] = np.mean(sc_atoms, axis=0)
        else:
            centroids[i] = trace.coords[i]  # fallback to Cα

    return centroids


def parse_pdb_all_atoms(pdb_path: str, chain_id: str = 'A'):
    """Parse all heavy atoms from a PDB file for one chain.

    Returns:
        residues: list of (res_seq, res_name_1letter, [(atom_name, x, y, z), ...])
        ca_coords: (n_res, 3) array of Cα coordinates
        sequences: string of 1-letter residue codes
    """
    from collections import defaultdict
    import os

    three_to_one = {
        'GLY': 'G', 'ALA': 'A', 'SER': 'S', 'CYS': 'C', 'VAL': 'V',
        'THR': 'T', 'ILE': 'I', 'PRO': 'P', 'LEU': 'L', 'ASN': 'N',
        'ASP': 'D', 'GLN': 'Q', 'LYS': 'K', 'GLU': 'E', 'MET': 'M',
        'HIS': 'H', 'PHE': 'F', 'ARG': 'R', 'TYR': 'Y', 'TRP': 'W',
        'MSE': 'M',  # selenomethionine → Met
    }

    # Only include standard amino acids (and MSE) — exclude waters, ligands, ions
    _STANDARD_RES = set(three_to_one.keys())

    residue_atoms = defaultdict(list)  # res_seq -> [(atom_name, x, y, z)]
    residue_names = {}  # res_seq -> 3-letter name
    ca_positions = {}   # res_seq -> (x, y, z)
    in_first_model = True

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ENDMDL'):
                in_first_model = False
                continue
            if line.startswith('MODEL'):
                # Only process model 1 (or first model encountered)
                model_num = int(line[10:14].strip()) if line[10:14].strip() else 1
                in_first_model = (model_num <= 1)
                continue
            if not in_first_model:
                continue
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            altloc = line[16].strip()
            if altloc not in ('', 'A'):
                continue
            chain = line[21].strip()
            if chain != chain_id:
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            # Skip non-standard residues (waters, ligands, ions)
            if res_name not in _STANDARD_RES:
                continue
            res_seq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() if len(line) > 76 else ''
            # Skip hydrogens
            if element == 'H' or (not element and atom_name[0] == 'H'):
                continue

            residue_atoms[res_seq].append((atom_name, x, y, z))
            residue_names[res_seq] = res_name
            if atom_name == 'CA':
                ca_positions[res_seq] = np.array([x, y, z])

    # Build ordered lists — only include residues that have a Cα
    sorted_seqs = sorted(residue_atoms.keys())
    sorted_seqs = [rs for rs in sorted_seqs if rs in ca_positions]
    residues = []
    ca_list = []
    seq_chars = []

    for rs in sorted_seqs:
        rname = residue_names.get(rs, 'UNK')
        one = three_to_one.get(rname, 'X')
        atoms = residue_atoms[rs]
        residues.append((rs, one, atoms))
        seq_chars.append(one)
        ca_list.append(ca_positions.get(rs, np.zeros(3)))

    ca_coords = np.array(ca_list)
    sequence = ''.join(seq_chars)
    return residues, ca_coords, sequence


def compute_side_chain_centroids(residues, ca_coords):
    """Compute side chain centroid for each residue.

    For each residue, the centroid is the mean position of all side chain
    heavy atoms (excluding backbone N, CA, C, O, OXT).  Glycine (no side
    chain) falls back to Cα.  Residues with missing side chain atoms use
    whatever atoms are available; if no side chain atoms at all, use Cα.

    Args:
        residues: list from parse_pdb_all_atoms
        ca_coords: (n_res, 3) Cα coordinates

    Returns:
        centroids: (n_res, 3) side chain centroid coordinates
    """
    n = len(residues)
    centroids = np.zeros((n, 3))

    for idx, (res_seq, one_letter, atoms) in enumerate(residues):
        sc_atoms = [(x, y, z) for (name, x, y, z) in atoms
                    if name not in _BACKBONE_ATOMS]
        if sc_atoms:
            centroids[idx] = np.mean(sc_atoms, axis=0)
        else:
            # No side chain atoms (Gly or missing) → use Cα
            centroids[idx] = ca_coords[idx]

    return centroids


def compute_signature_subdivided6_centroid(
    ca_coords: np.ndarray, centroid_coords: np.ndarray, i: int,
    radius: float = 100.0,
) -> Optional[List[List[int]]]:
    """Compute 6-subsegment octant signature using side chain centroids.

    The local frame is built from Cα positions (backbone trace), but
    octants are filled using side chain centroid positions instead of Cα.
    This captures side chain bulk and direction in the descriptor.

    Args:
        ca_coords: (n, 3) Cα coordinates — used for local frame construction
        centroid_coords: (n, 3) side chain centroid coordinates — used for
            octant filling
        i: reference residue index
        radius: sphere radius

    Returns:
        List of 8 lists of residue indices (up to 6 per octant), or None.
    """
    n = len(ca_coords)
    if i < 2:
        return None
    frame = build_local_frame(ca_coords[i - 2], ca_coords[i - 1], ca_coords[i])
    if frame is None:
        return None

    # Use centroid positions for filling, but origin is Cα of reference residue
    diffs = centroid_coords - ca_coords[i]              # (n, 3)
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))      # (n,)
    mask = (dists <= radius) & (dists > 1e-6)
    mask[i] = False
    local = diffs @ frame.T                              # (n, 3)

    # Octant assignment (1-based)
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    # Absolute values for sub-segment assignment
    adx = np.abs(local[:, 0])
    ady = np.abs(local[:, 1])
    adz = np.abs(local[:, 2])

    # Sub-segment (0-5) from ranking of (adx, ady, adz)
    abs_vals = np.stack([adx, ady, adz], axis=1)
    largest = np.argmax(abs_vals, axis=1)

    sub_idx = np.empty(n, dtype=int)
    m_x = largest == 0
    sub_idx[m_x & (ady >= adz)] = 0
    sub_idx[m_x & (ady < adz)] = 5
    m_y = largest == 1
    sub_idx[m_y & (adx >= adz)] = 1
    sub_idx[m_y & (adx < adz)] = 2
    m_z = largest == 2
    sub_idx[m_z & (ady >= adx)] = 3
    sub_idx[m_z & (ady < adx)] = 4

    sig_multi: List[List[int]] = []
    for slot in range(8):
        oct_mask = mask & (octants == slot + 1)
        if not np.any(oct_mask):
            sig_multi.append([])
            continue
        indices = np.where(oct_mask)[0]
        subs = sub_idx[indices]
        dists_sub = dists[indices]

        order = np.argsort(dists_sub)
        indices_sorted = indices[order]
        subs_sorted = subs[order]
        _, first_idx = np.unique(subs_sorted, return_index=True)
        nearest = indices_sorted[first_idx]

        nearest_dists = dists[nearest]
        final_order = np.argsort(nearest_dists)
        sig_multi.append(nearest[final_order].tolist())

    return sig_multi


def build_all_signatures_subdivided6_centroid(
    ca_coords: np.ndarray, centroid_coords: np.ndarray,
    radius: float = 100.0,
) -> List[Optional[List[List[int]]]]:
    """Compute 6-subsegment centroid octant signatures for every residue."""
    return [compute_signature_subdivided6_centroid(ca_coords, centroid_coords, i, radius)
            for i in range(len(ca_coords))]


# --------------------------------------------------------------------------- #
# Radial inner/outer descriptors (v7: binary radial zone per octant)
# --------------------------------------------------------------------------- #
# Each octant is split into two concentric radial zones:
#   inner:  distance < radius/2   (direct contacts, tight packing)
#   outer:  radius/2 <= distance <= radius  (second-shell contacts)
#
# Slot mapping:  slot = octant * 2 + zone  (zone: 0=inner, 1=outer)
#   BDL_inner=0  BDL_outer=1
#   BDR_inner=2  BDR_outer=3
#   BUL_inner=4  BUL_outer=5
#   BUR_inner=6  BUR_outer=7   (both excluded — BUR is invariant)
#   FDL_inner=8  FDL_outer=9
#   FDR_inner=10 FDR_outer=11
#   FUL_inner=12 FUL_outer=13
#   FUR_inner=14 FUR_outer=15
#
# Matchable slots (excluding BUR): 14 out of 16.

RADIAL_MATCH_SLOTS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]


def compute_signature_radial(
    coords: np.ndarray, i: int,
    radius: float = DEFAULT_RADIUS,
) -> Optional[List[List[int]]]:
    """Compute radial TURIST signature: 16 slots (8 octants x 2 radial zones).

    Each octant is split at radius/2 into an inner and outer zone.
    All residues within each sub-octant are stored, sorted by distance.

    Returns a list of 16 lists (slot 0..15), each containing chain indices
    of residues in that sub-octant sorted by distance, or None if the
    frame is undefined (i < 2 or collinear preceding trace).
    """
    n = len(coords)
    if i < 2:
        return None
    frame = build_local_frame(coords[i - 2], coords[i - 1], coords[i])
    if frame is None:
        return None

    # Vectorised: transform all residues into local frame
    diffs = coords - coords[i]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    mask = (dists <= radius) & (dists > 1e-6)
    mask[i] = False
    local = diffs @ frame.T

    # Octant assignment (1-based)
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    # Radial zone: inner (0) if dist < radius/2, outer (1) otherwise
    half_r = radius / 2.0
    inner_mask = dists < half_r

    sig_radial: List[List[int]] = []
    for slot in range(8):
        oct_mask = mask & (octants == slot + 1)

        # Inner zone (slot * 2)
        inner = oct_mask & inner_mask
        if np.any(inner):
            idx = np.where(inner)[0]
            order = np.argsort(dists[idx])
            sig_radial.append(idx[order].tolist())
        else:
            sig_radial.append([])

        # Outer zone (slot * 2 + 1)
        outer = oct_mask & ~inner_mask
        if np.any(outer):
            idx = np.where(outer)[0]
            order = np.argsort(dists[idx])
            sig_radial.append(idx[order].tolist())
        else:
            sig_radial.append([])

    return sig_radial


def build_all_signatures_radial(
    coords: np.ndarray,
    radius: float = DEFAULT_RADIUS,
) -> List[Optional[List[List[int]]]]:
    """Compute radial signatures for every Cα."""
    return [compute_signature_radial(coords, i, radius) for i in range(len(coords))]


# --------------------------------------------------------------------------- #
# Radial 3-sphere descriptors (v9: three concentric zones per octant)
# --------------------------------------------------------------------------- #
# Each octant is split into three concentric radial zones:
#   inner:   distance < r1            (direct contacts, tight packing)
#   middle:  r1 <= distance < r2      (second-shell contacts)
#   outer:   r2 <= distance <= radius (third-shell, global fold signal)
#
# Slot mapping:  slot = octant * 3 + zone  (zone: 0=inner, 1=middle, 2=outer)
#   BDL_inner=0  BDL_middle=1  BDL_outer=2
#   BDR_inner=3  BDR_middle=4  BDR_outer=5
#   BUL_inner=6  BUL_middle=7  BUL_outer=8
#   BUR_inner=9  BUR_middle=10 BUR_outer=11  (all excluded -- BUR is invariant)
#   FDL_inner=12 FDL_middle=13 FDL_outer=14
#   FDR_inner=15 FDR_middle=16 FDR_outer=17
#   FUL_inner=18 FUL_middle=19 FUL_outer=20
#   FUR_inner=21 FUR_middle=22 FUR_outer=23
#
# Matchable slots (excluding BUR): 21 out of 24.

RADIAL_MATCH_SLOTS_3SPHERE = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
]


def compute_signature_radial_3sphere(
    coords: np.ndarray, i: int,
    radius: float = 12.0,
    r1: float = 4.0,
    r2: float = 8.0,
) -> Optional[List[List[int]]]:
    """Compute radial 3-sphere TURIST signature: 24 slots (8 octants x 3 zones).

    Each octant is split at r1 and r2 into inner, middle, and outer zones.
    All residues within each sub-octant are stored, sorted by distance.

    Returns a list of 24 lists (slot 0..23), each containing chain indices
    of residues in that sub-octant sorted by distance, or None if the
    frame is undefined (i < 2 or collinear preceding trace).
    """
    n = len(coords)
    if i < 2:
        return None
    frame = build_local_frame(coords[i - 2], coords[i - 1], coords[i])
    if frame is None:
        return None

    # Vectorised: transform all residues into local frame
    diffs = coords - coords[i]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    mask = (dists <= radius) & (dists > 1e-6)
    mask[i] = False
    local = diffs @ frame.T

    # Octant assignment (1-based)
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    # Radial zone assignment: 0=inner, 1=middle, 2=outer
    zone = np.zeros(len(coords), dtype=int)
    zone[dists >= r1] = 1
    zone[dists >= r2] = 2

    sig: List[List[int]] = []
    for slot in range(8):
        oct_mask = mask & (octants == slot + 1)

        for z in range(3):
            z_mask = oct_mask & (zone == z)
            if np.any(z_mask):
                idx = np.where(z_mask)[0]
                order = np.argsort(dists[idx])
                sig.append(idx[order].tolist())
            else:
                sig.append([])

    return sig


def build_all_signatures_radial_3sphere(
    coords: np.ndarray,
    radius: float = 12.0,
    r1: float = 4.0,
    r2: float = 8.0,
) -> List[Optional[List[List[int]]]]:
    """Compute 3-sphere radial signatures for every Ca."""
    return [compute_signature_radial_3sphere(coords, i, radius, r1, r2)
            for i in range(len(coords))]


def compute_signature_subdivided6_radial(
    coords: np.ndarray, i: int,
    radius: float = 100.0,
) -> Optional[List[List[int]]]:
    """Compute 6-subsegment + radial octant signature: 96 slots.

    Combines angular subdivision (6 subsections per octant) with radial
    split (inner/outer). Each sub-region keeps the nearest residue.

    Slot mapping: slot = octant * 12 + sub_idx * 2 + zone
    where sub_idx in 0..5, zone in {0=inner, 1=outer}.

    Returns a list of 96 lists (8 octants x 6 subs x 2 zones), each with
    up to 1 residue (nearest), or None if frame undefined.
    """
    n = len(coords)
    if i < 2:
        return None
    frame = build_local_frame(coords[i - 2], coords[i - 1], coords[i])
    if frame is None:
        return None

    diffs = coords - coords[i]
    dists = np.sqrt(np.sum(diffs * diffs, axis=1))
    mask = (dists <= radius) & (dists > 1e-6)
    mask[i] = False
    local = diffs @ frame.T

    # Octant assignment (1-based)
    octants = (1 + (local[:, 0] >= 0).astype(int) * 4
                   + (local[:, 1] >= 0).astype(int) * 2
                   + (local[:, 2] >= 0).astype(int))

    # Angular sub-segment (0-5) from ranking of (|dx|, |dy|, |dz|)
    adx = np.abs(local[:, 0])
    ady = np.abs(local[:, 1])
    adz = np.abs(local[:, 2])
    abs_vals = np.stack([adx, ady, adz], axis=1)
    largest = np.argmax(abs_vals, axis=1)

    sub_idx = np.empty(n, dtype=int)
    m_x = largest == 0
    sub_idx[m_x & (ady >= adz)] = 0
    sub_idx[m_x & (ady < adz)] = 5
    m_y = largest == 1
    sub_idx[m_y & (adx >= adz)] = 1
    sub_idx[m_y & (adx < adz)] = 2
    m_z = largest == 2
    sub_idx[m_z & (ady >= adx)] = 3
    sub_idx[m_z & (ady < adx)] = 4

    # Radial zone
    half_r = radius / 2.0
    inner_mask = dists < half_r

    sig: List[List[int]] = []
    for slot in range(8):
        oct_mask = mask & (octants == slot + 1)
        for sub in range(6):
            sub_mask = oct_mask & (sub_idx == sub)

            # Inner
            inner = sub_mask & inner_mask
            if np.any(inner):
                idx = np.where(inner)[0]
                nearest = idx[np.argmin(dists[idx])]
                sig.append([int(nearest)])
            else:
                sig.append([])

            # Outer
            outer = sub_mask & ~inner_mask
            if np.any(outer):
                idx = np.where(outer)[0]
                nearest = idx[np.argmin(dists[idx])]
                sig.append([int(nearest)])
            else:
                sig.append([])

    return sig


def build_all_signatures_subdivided6_radial(
    coords: np.ndarray,
    radius: float = 100.0,
) -> List[Optional[List[List[int]]]]:
    """Compute 6-subsegment + radial signatures for every Cα."""
    return [compute_signature_subdivided6_radial(coords, i, radius)
            for i in range(len(coords))]
