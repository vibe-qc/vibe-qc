"""GPAW plane-wave-limit atomization-energy reference generator.

vibe-qc's out-of-process (§10) VASP replacement for producing PW-limit
atomization references for the revised pob basis-set paper. See
:mod:`pw_reference` for the boundary discipline and method notes.
"""
from .pw_reference import (
    PW_XC,
    CutoffPoint,
    PwAtomizationResult,
    compute_pw_atomization,
)
from .systems import BY_ID, HUND_MAGMOM, VALIDATION_SET, PwSystem

__all__ = [
    "PW_XC",
    "CutoffPoint",
    "PwAtomizationResult",
    "compute_pw_atomization",
    "PwSystem",
    "VALIDATION_SET",
    "BY_ID",
    "HUND_MAGMOM",
]
