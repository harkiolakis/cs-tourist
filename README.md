# CS-TOURIST

**Connectivity-aware geometric descriptors for catalytic site mapping across non-homologous proteins.**

CS-TOURIST encodes the local structural environment of each catalytic Cα atom using octant-binned geometric descriptors augmented with sequence-connectivity classes. Each catalytic Cα neighborhood is encoded as 72 sub-slots combining spatial direction (8 octants × 3 radial zones) with sequence-connectivity class (local, medium, long-range), scored via class-presence matching that is robust to insertions and deletions.

## Key results

Across 924 enzyme pairs from the Mechanism and Catalytic Site Atlas:

| Metric | CS-TOURIST Mode C | Best competitor |
|--------|-------------------|-----------------|
| Recall | 81.0% | 4.8% (TM-align) |
| Precision | 99.4% | 5.7% (TM-align) |
| F1 | 0.872 | 0.047 (TM-align) |

CS-TOURIST achieves 17× to 118× higher recall than all existing methods, with 79.3% recall even across different CATH topologies.

## Installation

```bash
cd cs-tourist
pip install -e .
```

Dependencies: `numpy`, `biopython`, `numba`, `pandas`, `matplotlib`, `biotite`

## Usage

### Command-line interface

```bash
turist align <pdb_a> <pdb_b> [options]
```

### Catalytic site mapping (Mode C)

```python
from turist.io import parse_chain_trace
from turist.descriptor import compute_descriptor
from turist.matching import align_catalytic_sites

# Parse Cα traces
trace_a = parse_chain_trace("1r44A.pdb")
trace_b = parse_chain_trace("1o98A.pdb")

# Catalytic residue indices (1-based)
cat_a = [47, 87, 145, 181, 233]
cat_b = [11, 12, 62, 88, 137, 138, 168, 188, 211, 235]

# Compute connectivity-aware descriptors and align
result = align_catalytic_sites(trace_a, trace_b, cat_a, cat_b,
                                mode="C", connectivity_aware=True)
print(f"Recall: {result.recall:.3f}")
print(f"Precision: {result.precision:.3f}")
print(f"RMSD: {result.rmsd:.2f} Å")
```

## Repository structure

```
cs-tourist/
├── turist/                    # Core Python package
│   ├── descriptor.py          # Octant descriptor computation
│   ├── matching.py            # Connectivity-aware scoring & DP alignment
│   ├── align.py               # Full alignment pipeline
│   ├── superpose.py           # Kabsch superposition
│   ├── io.py                  # PDB parsing
│   └── cli.py                 # Command-line interface
├── tests/
│   └── test_turist.py         # Unit tests
├── benchmark/
│   ├── scripts/               # Benchmark & evaluation scripts
│   │   ├── cs_tourist.py          # CS-TOURIST Mode C runner
│   │   ├── cs_tourist_eval.py     # Evaluation metrics
│   │   ├── competitor_eval.py     # Cross-method comparison + figures
│   │   ├── competitor_gass.py     # GASS-Metal wrapper
│   │   ├── competitor_pyscomotif.py  # pyScoMotif wrapper
│   │   ├── run_probis.py          # ProBiS wrapper
│   │   ├── benchmark_tmalign.py   # TM-align/US-align wrapper
│   │   ├── benchmark_foldseek.py  # Foldseek wrapper
│   │   ├── select_pairs.py        # Benchmark pair selection
│   │   ├── parse_sites.py         # M-CSA annotation parsing
│   │   ├── build_v11_caches.py    # Descriptor cache builder
│   │   └── run_v11_configs.py     # Config runner
│   └── configs/
│       └── v11_configs.json       # Connectivity-aware configuration
└── examples/
    └── case_study_demo.py     # Case study reproduction
```

## Reproducing the benchmark

1. **Download benchmark data** from the Zenodo archive (DOI: [to be added])

2. **Run CS-TOURIST Mode C:**
   ```bash
   python benchmark/scripts/cs_tourist.py --pairs data/pair_list.csv \
       --annotations data/pdb_site_annotations.json \
       --output data/cs_tourist_eval_v11conn.csv \
       --mode C --connectivity-aware
   ```

3. **Run competitor methods:**
   ```bash
   python benchmark/scripts/benchmark_tmalign.py --pairs data/pair_list.csv
   python benchmark/scripts/benchmark_foldseek.py --pairs data/pair_list.csv
   python benchmark/scripts/run_probis.py --pairs data/pair_list.csv
   python benchmark/scripts/competitor_gass.py --pairs data/pair_list.csv
   python benchmark/scripts/competitor_pyscomotif.py --pairs data/pair_list.csv
   ```

4. **Evaluate and generate figures:**
   ```bash
   python benchmark/scripts/competitor_eval.py
   ```

## Citation

Harkiolakis, N. (2026) Connectivity-aware geometric descriptors for catalytic site mapping across non-homologous proteins. *Bioinformatics* (in press).

## License

MIT License — see [LICENSE](LICENSE).

## Contact

[Email to be completed]
