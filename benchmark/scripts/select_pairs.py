#!/usr/bin/env python3
"""Select 100 queries from diverse CATH topologies, 500 stratified targets per query."""

import json
import csv
import random
import sys
from collections import defaultdict

random.seed(42)

# ── Load classifications ───────────────────────────────────────────────
with open('/mnt/shared-workspace/entry_classifications.json') as f:
    entries = json.load(f)

print(f"Total entries: {len(entries)}")

# Index entries
has_cath = [e for e in entries if e['cath_homologous_sf']]
has_scop = [e for e in entries if e['scop_superfamily']]
print(f"Entries with CATH: {len(has_cath)}")
print(f"Entries with SCOPe: {len(has_scop)}")

# Build lookup: (pdb_id, chain) -> entry
entry_lookup = {}
for e in entries:
    key = (e['pdb_id'].lower(), e['chain'].upper())
    entry_lookup[key] = e

# ── Group by CATH topology for query selection ─────────────────────────
topo_groups = defaultdict(list)
for e in has_cath:
    topo_groups[e['cath_topology']].append(e)

print(f"\nCATH topologies with entries: {len(topo_groups)}")

# Sort topologies by entry count (descending)
sorted_topos = sorted(topo_groups.items(), key=lambda x: -len(x[1]))

# ── Select 100 queries ─────────────────────────────────────────────────
# Strategy: pick one query from each of the top 100 topologies
# Within each topology, prefer entries that also have SCOPe classification
# Ensure no two queries from the same CATH homologous superfamily

queries = []
used_hsfs = set()

for topo, topo_entries in sorted_topos:
    if len(queries) >= 100:
        break

    # Prefer entries with SCOPe, then random within the topology
    with_scop = [e for e in topo_entries if e['scop_superfamily']]
    without_scop = [e for e in topo_entries if not e['scop_superfamily']]

    random.shuffle(with_scop)
    random.shuffle(without_scop)

    candidates = with_scop + without_scop

    for e in candidates:
        if e['cath_homologous_sf'] not in used_hsfs:
            queries.append(e)
            used_hsfs.add(e['cath_homologous_sf'])
            break

print(f"Selected {len(queries)} queries from {len(set(q['cath_topology'] for q in queries))} CATH topologies")
print(f"Queries with SCOPe: {sum(1 for q in queries if q['scop_superfamily'])}/{len(queries)}")

# ── Select 500 targets per query ───────────────────────────────────────
# Stratified sampling:
#   Same CATH H-level: up to 50
#   Same CATH T-level, different H: up to 100
#   Different CATH T-level: up to 350
# Exclude self-hits

pairs = []

for qi, q in enumerate(queries):
    q_key = (q['pdb_id'], q['chain'])
    q_topo = q['cath_topology']
    q_hsf = q['cath_homologous_sf']

    # Partition all entries by relationship to query
    same_h = []       # same homologous superfamily
    same_t_diff_h = [] # same topology, different H
    diff_t = []        # different topology
    no_cath = []       # no CATH classification

    for e in entries:
        e_key = (e['pdb_id'], e['chain'])
        if e_key == q_key:
            continue  # skip self

        if not e['cath_homologous_sf']:
            no_cath.append(e)
        elif e['cath_homologous_sf'] == q_hsf:
            same_h.append(e)
        elif e['cath_topology'] == q_topo:
            same_t_diff_h.append(e)
        else:
            diff_t.append(e)

    # Sample
    random.shuffle(same_h)
    random.shuffle(same_t_diff_h)
    random.shuffle(diff_t)
    random.shuffle(no_cath)

    targets = []
    targets.extend(same_h[:50])
    targets.extend(same_t_diff_h[:100])

    # Fill remaining with diff_t, then no_cath if needed
    remaining = 500 - len(targets)
    targets.extend(diff_t[:remaining])

    if len(targets) < 500:
        # Top up with no_cath entries
        remaining2 = 500 - len(targets)
        targets.extend(no_cath[:remaining2])

    # Build pairs with relationship labels
    for t in targets:
        # CATH relationship
        if not t['cath_homologous_sf']:
            cath_rel = 'unclassified'
        elif t['cath_homologous_sf'] == q_hsf:
            cath_rel = 'same_h'
        elif t['cath_topology'] == q_topo:
            cath_rel = 'same_t_diff_h'
        else:
            cath_rel = 'diff_t'

        # SCOPe relationship
        if not q['scop_superfamily'] or not t['scop_superfamily']:
            scop_rel = 'unclassified'
        elif t['scop_superfamily'] == q['scop_superfamily']:
            scop_rel = 'same_sf'
        elif t['scop_fold'] == q['scop_fold']:
            scop_rel = 'same_fold_diff_sf'
        else:
            scop_rel = 'diff_fold'

        pairs.append({
            'query_id': q['pdb_id'],
            'query_chain': q['chain'],
            'target_id': t['pdb_id'],
            'target_chain': t['chain'],
            'query_n_res': q['n_res'],
            'target_n_res': t['n_res'],
            'scop_relation': scop_rel,
            'cath_relation': cath_rel,
            'query_scop_sf': q['scop_superfamily'],
            'target_scop_sf': t['scop_superfamily'],
            'query_cath_h': q['cath_homologous_sf'],
            'target_cath_h': t['cath_homologous_sf'],
        })

    if (qi + 1) % 20 == 0:
        print(f"  Query {qi+1}/{len(queries)}: {q['pdb_id']}_{q['chain']} — "
              f"same_h={len(same_h)}, same_t_diff_h={len(same_t_diff_h)}, "
              f"diff_t={len(diff_t)}, targets={len(targets)}")

print(f"\nTotal pairs: {len(pairs)}")

# ── Summary statistics ─────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Pair relationship distribution:")

print("\nCATH relationships:")
cath_rels = defaultdict(int)
for p in pairs:
    cath_rels[p['cath_relation']] += 1
for rel, count in sorted(cath_rels.items(), key=lambda x: -x[1]):
    print(f"  {rel}: {count} ({100*count/len(pairs):.1f}%)")

print("\nSCOPe relationships:")
scop_rels = defaultdict(int)
for p in pairs:
    scop_rels[p['scop_relation']] += 1
for rel, count in sorted(scop_rels.items(), key=lambda x: -x[1]):
    print(f"  {rel}: {count} ({100*count/len(pairs):.1f}%)")

# Unique PDB files needed
unique_pdbs = set()
for p in pairs:
    unique_pdbs.add(p['query_id'].lower())
    unique_pdbs.add(p['target_id'].lower())
print(f"\nUnique PDB files needed: {len(unique_pdbs)}")

# ── Save pair list ─────────────────────────────────────────────────────
csv_path = '/mnt/shared-workspace/pair_list.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=pairs[0].keys())
    writer.writeheader()
    writer.writerows(pairs)
print(f"\nSaved pair list to {csv_path}")

# Save query list
query_csv = '/mnt/shared-workspace/query_list.csv'
with open(query_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['query_id', 'query_chain', 'query_n_res',
                                            'query_scop_sf', 'query_cath_h', 'query_cath_topo'])
    writer.writeheader()
    for q in queries:
        writer.writerow({
            'query_id': q['pdb_id'],
            'query_chain': q['chain'],
            'query_n_res': q['n_res'],
            'query_scop_sf': q['scop_superfamily'],
            'query_cath_h': q['cath_homologous_sf'],
            'query_cath_topo': q['cath_topology'],
        })
print(f"Saved query list to {query_csv}")
