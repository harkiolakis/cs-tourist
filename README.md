# CS-TOURIST

**Connectivity-aware geometric descriptors for catalytic site alignment across non-homologous proteins.**

CS-TOURIST encodes the local structural environment of each catalytic Cα atom using octant-binned geometric descriptors augmented with sequence-connectivity classes. Each catalytic Cα neighborhood is encoded as 72 sub-slots combining spatial direction (8 octants × 3 radial zones) with sequence-connectivity class (local, medium, long-range), scored via class-presence matching that is robust to insertions and deletions.

## Key results

Across 924 enzyme pairs from the Mechanism and Catalytic Site Atlas, evaluated by catalytic-site RMSD after optimal superposition:

| Metric | CS-TOURIST Mode C | Random baseline |
|--------|-------------------|-----------------|
| Median RMSD | 4.9 Å | ~21 Å (estimated) |
| RMSD < 2 Å | 19.4% of pairs | — |
| RMSD < 3 Å | 29.1% of pairs | — |
| RMSD < 5 Å | 51.2% of pairs | — |

### Negative controls (decoy experiment)

To verify that CS-TOURIST's geometric scoring detects genuine catalytic site similarity rather than merely aligning arbitrary labelled residue sets, we ran a decoy negative control: for each of the 924 pairs, the target's catalytic residues were replaced with non-catalytic decoy residues (5 random + 5 surface-exposed trials per pair = 9,240 decoy alignments).

| Metric | Real catalytic | Decoy (random) | Decoy (surface) |
|--------|---------------|----------------|-----------------|
| Median RMSD | 4.9 Å | 11.0 Å (2.2×) | 13.3 Å (2.7×) |
| AUC | — | 0.78 | 0.83 |
| Per-pair detection rate | — | 0.81 | 0.86 |
| Wilcoxon p | — | 3.1 × 10⁻¹¹¹ | 3.2 × 10⁻¹³¹ |

Real catalytic residues produce 2.2–2.7× lower RMSD than decoys with astronomical statistical significance, confirming that the geometric scoring genuinely detects catalytic site similarity.

### Important caveats

- **Precision is tautological**: Mode C restricts dynamic programming to catalytic residues only, so precision = 1.0 by construction. It carries no discriminative information and is not reported as a performance metric.
- **Recall is dominated by residue-count ratios**: A random-alignment baseline achieves recall 0.838, slightly exceeding CS-TOURIST's actual recall of 0.810. Recall is therefore insufficient as a standalone metric; RMSD is the primary evaluation metric.
- **The competitor comparison is not apples-to-apples**: CS-TOURIST receives catalytic residue labels on both query and target, while competitors (TM-align, Foldseek, Dali, etc.) perform unsupervised whole-structure alignment. The methods solve different tasks and should be considered complementary.

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

### Catalytic site alignment (Mode C)

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
print(f"RMSD: {result.rmsd:.2f} Å")
print(f"Recall: {result.recall:.3f}")
```

### Negative control (decoy experiment)

```bash
python benchmark/scripts/decoy_experiment.py \
    --pairs data/pair_list.csv \
    --annotations data/csa_sites.json \
    --pdb-dir data/pdbs/ \
    --eval-csv data/cs_tourist_eval_v11conn.csv \
    --output data/decoy_experiment_results.csv \
    --trials 5
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
│   │   ├── cs_tourist.py              # CS-TOURIST Mode C runner
│   │   ├── cs_tourist_eval.py         # Evaluation metrics
│   │   ├── decoy_experiment.py        # Negative control (decoy residues)
│   │   ├── competitor_eval.py         # Cross-method comparison + figures
│   │   ├── competitor_gass.py         # GASS-Metal wrapper
│   │   ├── competitor_pyscomotif.py   # pyScoMotif wrapper
│   │   ├── run_probis.py              # ProBiS wrapper
│   │   ├── benchmark_tmalign.py       # TM-align/US-align wrapper
│   │   ├── benchmark_foldseek.py      # Foldseek wrapper
│   │   ├── select_pairs.py            # Benchmark pair selection
│   │   ├── parse_sites.py             # M-CSA annotation parsing
│   │   ├── build_v11_caches.py        # Descriptor cache builder
│   │   └── run_v11_configs.py         # Config runner
│   └── configs/
│       └── v11_configs.json           # Connectivity-aware configuration
├── examples/
│   └── case_study_demo.py     # Case study reproduction
└── paper_v2.md                # Revised manuscript (Markdown)
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

3. **Run negative control (decoy experiment):**
   ```bash
   python benchmark/scripts/decoy_experiment.py \
       --pairs data/pair_list.csv \
       --annotations data/csa_sites.json \
       --pdb-dir data/pdbs/ \
       --eval-csv data/cs_tourist_eval_v11conn.csv \
       --output data/decoy_experiment_results.csv \
       --trials 5
   ```

4. **Run competitor methods:**
   ```bash
   python benchmark/scripts/benchmark_tmalign.py --pairs data/pair_list.csv
   python benchmark/scripts/benchmark_foldseek.py --pairs data/pair_list.csv
   python benchmark/scripts/run_probis.py --pairs data/pair_list.csv
   python benchmark/scripts/competitor_gass.py --pairs data/pair_list.csv
   python benchmark/scripts/competitor_pyscomotif.py --pairs data/pair_list.csv
   ```

5. **Evaluate and generate figures:**
   ```bash
   python benchmark/scripts/competitor_eval.py
   ```

## Citation

Harkiolakis, N. (2026) Connectivity-aware geometric descriptors for catalytic
site alignment across non-homologous proteins. *Bioinformatics* (submitted).

## License

MIT License — see [LICENSE](LICENSE).

## Contact

[Email to be completed]
