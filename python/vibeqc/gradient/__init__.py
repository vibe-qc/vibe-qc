"""Analytic nuclear gradients for non-mean-field wavefunction methods.

Provides the CASSCF analytic gradient (state-specific and state-averaged,
with a numerical full-energy FD cross-check), the CASPT2 gradient
(CASSCF + FD of IC-CASPT2 E^(2)), and the SC-NEVPT2 gradient
(CASSCF + FD of SC-NEVPT2 E^(2)).

The CASSCF analytic gradient is the complete derivative of a variational
energy: at a stationary CASSCF the first-order wavefunction response
vanishes, so there is no z-vector term to add, and the finite-basis
geometry dependence is carried by the overlap (Pulay) contribution. The
former ``compute_wz=True`` W^z correction was retired for producing
spurious components and is now a warned no-op alias (GitLab #516, #119);
``compute_wz="numerical"`` still selects the full-energy FD path.
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
