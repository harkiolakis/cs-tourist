#!/usr/bin/env python3
"""Build v11 descriptor caches: connectivity-aware + multi-scale 3-sphere.

Loads the existing 2-sphere cache (which has trace.coords for all chains)
and rebuilds compact descriptors using:
  1. Connectivity-aware descriptor (72 sub-slots, r=12, r1=4, r2=8)
  2. 3-sphere at r=8 (r1=3, r2=6) for multi-scale local
  3. 3-sphere at r=16 (r1=5, r2=11) for multi-scale global

Usage:
    python build_v11_caches.py [--caches all|connectivity|r8|r16]
"""

import argparse
import os
import pickle
import time
import numpy as np
import sys

sys.path.insert(0, '/mnt/shared-workspace/tourist')
sys.path.insert(0, '/mnt/results/benchmark/scripts')

from tourist.descriptor import (
    relative_offsets_multi,
    RADIAL_MATCH_SLOTS_3SPHERE,
    CONNECTIVITY_MATCH_SLOTS,
    build_all_signatures_radial_3sphere,
    build_all_signatures_connectivity_aware,
)

SLOTS_3SPHERE = RADIAL_MATCH_SLOTS_3SPHERE
SLOTS_CONN = CONNECTIVITY_MATCH_SLOTS

CONFIGS = {
    'connectivity': {
        'type': 'connectivity',
        'radius': 12.0, 'r1': 4.0, 'r2': 8.0,
        'n_slots': 72, 'match_slots': SLOTS_CONN,
        'build_fn': build_all_signatures_connectivity_aware,
        'out_tag': 'conn_r12_r14_r28',
    },
    'r8': {
        'type': '3sphere',
        'radius': 8.0, 'r1': 3.0, 'r2': 6.0,
        'n_slots': 24, 'match_slots': SLOTS_3SPHERE,
        'build_fn': build_all_signatures_radial_3sphere,
        'out_tag': 'radial3_r8_r13_r26',
    },
    'r16': {
        'type': '3sphere',
        'radius': 16.0, 'r1': 5.0, 'r2': 11.0,
        'n_slots': 24, 'match_slots': SLOTS_3SPHERE,
        'build_fn': build_all_signatures_radial_3sphere,
        'out_tag': 'radial3_r16_r15_r211',
    },
}


def build_cache_from_2sphere(cache_2sphere, cfg):
    """Build compact descriptors from existing 2-sphere cache."""
    n_slots = cfg['n_slots']
    match_slots = cfg['match_slots']
    build_fn = cfg['build_fn']
    radius, r1, r2 = cfg['radius'], cfg['r1'], cfg['r2']

    new_cache = {}
    t0 = time.time()
    keys = sorted(cache_2sphere.keys())
    for idx, key in enumerate(keys):
        trace = cache_2sphere[key][0]
        coords = trace.coords
        n = len(coords)

        try:
            sigs = build_fn(coords, radius, r1, r2)
        except Exception as e:
            print(f"  WARNING: {key}: build failed: {e}", flush=True)
            continue

        compact = []
        for slot in range(n_slots):
            if slot not in match_slots:
                compact.append((np.array([], dtype=np.int32),
                                np.array([], dtype=np.int32)))
                continue
            positions = []
            offsets = []
            for ii in range(n):
                rel = relative_offsets_multi(sigs[ii], ii)
                if rel is not None and slot < len(rel):
                    for off in rel[slot]:
                        positions.append(ii)
                        offsets.append(off)
            compact.append((np.array(positions, dtype=np.int32),
                            np.array(offsets, dtype=np.int32)))

        new_cache[key] = (trace, compact)

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {idx+1}/{len(keys)} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  Built {len(new_cache)} chains in {elapsed:.1f}s", flush=True)
    return new_cache


def main():
    parser = argparse.ArgumentParser(description='Build v11 descriptor caches')
    parser.add_argument('--caches', default='all', help='Comma-separated names or "all"')
    parser.add_argument('--cache-2sphere', default='/mnt/shared-workspace/shared/cache_radial_r8.pkl')
    args = parser.parse_args()

    if args.caches == 'all':
        config_names = list(CONFIGS.keys())
    else:
        config_names = [c.strip() for c in args.caches.split(',')]

    # Load 2-sphere cache
    print(f"Loading 2-sphere cache from {args.cache_2sphere}...", flush=True)
    t0 = time.time()
    with open(args.cache_2sphere, 'rb') as f:
        cache_2sphere = pickle.load(f)
    print(f"  Loaded {len(cache_2sphere)} chains in {time.time()-t0:.1f}s", flush=True)

    for name in config_names:
        cfg = CONFIGS[name]
        print(f"\nBuilding {name} cache (type={cfg['type']}, "
              f"r={cfg['radius']}, r1={cfg['r1']}, r2={cfg['r2']}, "
              f"slots={cfg['n_slots']})...", flush=True)

        cache = build_cache_from_2sphere(cache_2sphere, cfg)

        out_path = f"/mnt/shared-workspace/shared/cache_{cfg['out_tag']}.pkl"
        tmp_path = f"/workspace/cache_{cfg['out_tag']}.pkl"

        print(f"  Saving to {out_path}...", flush=True)
        with open(tmp_path, 'wb') as f:
            pickle.dump(cache, f)
        os.system(f'cp {tmp_path} {out_path}')
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved {len(cache)} chains ({size_mb:.1f} MB)", flush=True)

    print("\nAll caches built successfully.", flush=True)


if __name__ == '__main__':
    main()
