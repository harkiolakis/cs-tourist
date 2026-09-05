"""TURIST: pairwise 3D protein structure alignment via trace-defined local
Cα environment signatures (Harkiolakis).

The public API is :func:`align_structures`.
"""

from .align import align_structures, AlignmentResult

__all__ = ["align_structures", "AlignmentResult"]
__version__ = "0.1.0"
