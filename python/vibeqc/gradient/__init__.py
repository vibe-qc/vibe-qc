"""Analytic nuclear gradients for non-mean-field wavefunction methods.

Provides the CASSCF analytic gradient (state-specific and state-averaged,
z-vector-free + gated W^z correction + numerical FD fallback), the CASPT2
gradient (CASSCF + FD of IC-CASPT2 E^(2)), and the SC-NEVPT2 gradient
(CASSCF + FD of SC-NEVPT2 E^(2)).
"""

from __future__ import annotations

from ._caspt2 import compute_caspt2_gradient
from ._casscf import compute_casscf_gradient
from ._ms_caspt2_nac import compute_ms_caspt2_nac
from ._nevpt2 import compute_nevpt2_gradient

__all__ = [
    "compute_casscf_gradient",
    "compute_caspt2_gradient",
    "compute_nevpt2_gradient",
    "compute_ms_caspt2_nac",
]
