"""Semiempirical structure-optimization stack (DFTB0, SCC-DFTB, ...).

Public API
----------
- :class:`SemiempiricalModel` -- abstract base class
- :class:`DFTB0Model` -- non-self-consistent tight-binding
- :class:`SCCDFTBModel` -- self-consistent-charge tight-binding
- :class:`GFN2Model` -- experimental molecular GFN2-xTB wrapper
- :class:`PM6Model` / :class:`UPM6Model` -- molecular PM6 wrappers
- :class:`OMxModel` -- molecular OM1/OM2/OM3 wrapper
- :class:`PeriodicPM6Model` -- periodic PM6 with FD grad/stress
- :class:`PeriodicOMxModel` -- fail-closed periodic OMx compatibility wrapper
- :class:`DispersionCorrectedModel` -- D3(BJ) dispersion wrapper
- :func:`run_dftb0` -- one-shot DFTB0 energy (C++ backend)
- :func:`run_scc_dftb` -- SCC-DFTB energy
- :func:`d3bj_energy` -- D3(BJ) dispersion energy
- :func:`compute_stress_fd` -- finite-difference stress tensor
- :func:`optimize_cell` -- variable-cell optimization
- :func:`optimize_pm6_cell` -- PM6 cell optimization
- :func:`optimize_omx_cell` -- fail-closed periodic OMx compatibility helper
- :func:`preoptimize_molecule` -- molecular semiempirical preoptimization
- :func:`preoptimize_periodic` -- semiempirical -> ab initio preoptimization
- :func:`run_semiempirical` -- unified molecular semiempirical dispatcher
- :func:`run_native_energy` -- unified molecular native energy facade
- :class:`SCFNonConvergenceError` -- loud k-route SCF non-convergence
- :func:`semiempirical_route_status` -- implementation status for public routes
- :func:`semiempirical_route_runtime_available` -- local runtime-asset preflight
"""

from __future__ import annotations

# Loud k-route SCF non-convergence (issue #342): raised by the native
# k-point routes (run_scc_dftb_kpoints, run_gfn2_xtb_kpoints) when the
# SCC loop exhausts its budget and the caller did not pass
# ``allow_unconverged=True``.  RuntimeError subclass.
from vibeqc._vibeqc_core.semiempirical import SCFNonConvergenceError

from .dftb0 import (
    DFTB0Model,
    SCCDFTBModel,
    UDFTB0Model,
    USCCDFTBModel,
    run_dftb0,
    run_scc_dftb,
)
from .dispersion import DFTB_D3_DEFAULTS, DispersionCorrectedModel, d3bj_energy
from .model import SemiempiricalModel
from .native import native_route_descriptor, run_native_energy
from .parameters import SemiempiricalParameters
from .periodic import (
    compute_stress_fd,
    optimize_cell,
    optimize_omx_cell,
    optimize_pm6_cell,
)
from .preoptimize import preoptimize_molecule, preoptimize_periodic
from .runner import (
    MOLECULAR_SEMIEMPIRICAL_METHODS,
    SEMIEMPIRICAL_METHODS,
    SemiempiricalEnergyError,
    SemiempiricalResult,
    run_ccm,
    run_seccm,
    run_semiempirical,
)
from .routes import (
    GFN2SECCMHamiltonianIdentity,
    GFN2SECCMRestartIdentity,
    GFN2SECCMRunControls,
    SemiempiricalRoutePlan,
)
from .seccm import (
    DFTB0_SECCM_PARAMETER_SET,
    DFTB0SECCMResult,
    GFN2_SECCM_PARAMETER_SET,
    GFN2SECCMAttempt,
    GFN2SECCMConvergenceError,
    GFN2SECCMResult,
    GFN2SECCMStateChange,
    compare_gfn2_seccm_states,
    OMxSECCMResult,
    PM6_SECCM_PARAMETER_SET,
    PM6SECCMResult,
    SCCDFTB_SECCM_PARAMETER_SET,
    SCCDFTBSECCMResult,
    run_dftb0_seccm,
    run_gfn2_seccm,
    run_omx_seccm,
    run_pm6_seccm,
    run_scc_dftb_seccm,
)
from .status import (
    BACKEND_GATED_EXPERIMENTAL,
    BACKEND_MIXED_NATIVE,
    BACKEND_NATIVE,
    BACKEND_NATIVE_FD,
    BACKEND_PYTHON_REFERENCE,
    REFERENCE_ONLY_SEMIEMPIRICAL_ROUTES,
    SEMIEMPIRICAL_ROUTE_ALIASES,
    SEMIEMPIRICAL_ROUTE_STATUS,
    SemiempiricalRouteStatus,
    semiempirical_route_runtime_available,
    semiempirical_route_status,
)


class NDDOExperimentalWarning(UserWarning):
    """Emitted when an OM1 calculation runs through a public molecular API.

    The published OM1 evaluates its core-valence effective core potential
    analytically (Kolb & Thiel, J. Comput. Chem. 14, 775 (1993)) and
    publishes no semiempirical ECP parameters (Dral 2016 Table 1).
    vibe-qc does not implement that analytic ECP yet, so OM1 is missing
    part of its core-valence Pauli repulsion: bonds to heavy atoms come
    out ~0.3 A short and the uncapped F2 orthogonalization term can
    variationally collapse at close non-bonded contacts (e.g. eclipsed
    ethane).  PM6, OM2 and OM3 carry their full published Hamiltonians
    and do not warn.  See ``docs/user_guide/semiempirical.md``.
    """


class DFTB0RepulsivePlaceholderWarning(UserWarning):
    """Emitted when a DFTB0/SCC-DFTB run consumes a repulsive pair with no
    parameterized entry (issue #306).

    The built-in DFTB parameter sets carry tuned R^-12 entries only for the
    {H, C, N, O, F, P, S, Cl} pairs. Every other element pair falls back to
    a combining-rule ``A/R^12`` estimate that is numerically irrelevant at
    equilibrium separations, so absolute total energies, equilibrium
    geometries, EOS fits, and a0/B0/V0/cohesive energies are NOT chemistry
    for such systems. At fixed geometry the repulsive term is a single
    additive constant, so *differences* taken at identical geometry
    (k-mesh convergence, supercell convergence, Madelung on/off, Gamma vs
    multi-k) remain exactly valid. See ``docs/user_guide/semiempirical.md``.
    """


_LAZY_EXPORTS = {
    "GFN2D4UnsupportedWarning": (
        "vibeqc.semiempirical.methods.gfn2",
        "GFN2D4UnsupportedWarning",
    ),
    "GFN2ExperimentalWarning": (
        "vibeqc.semiempirical.methods.gfn2",
        "GFN2ExperimentalWarning",
    ),
    "GFN2Model": ("vibeqc.semiempirical.methods.gfn2", "GFN2Model"),
    "OMxModel": ("vibeqc.semiempirical.methods.omx", "OMxModel"),
    "PM6Model": ("vibeqc.semiempirical.methods.pm6", "PM6Model"),
    "UPM6Model": ("vibeqc.semiempirical.methods.pm6", "UPM6Model"),
    "PeriodicOMxModel": (
        "vibeqc.semiempirical.methods.periodic_omx",
        "PeriodicOMxModel",
    ),
    "PeriodicPM6Model": (
        "vibeqc.semiempirical.methods.periodic_pm6",
        "PeriodicPM6Model",
    ),
}


def __getattr__(name: str):
    """Lazily expose optional/heavier semiempirical model wrappers."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "SemiempiricalModel",
    "DFTB0Model",
    "DFTB0_SECCM_PARAMETER_SET",
    "DFTB0SECCMResult",
    "SCCDFTBModel",
    "UDFTB0Model",
    "USCCDFTBModel",
    "GFN2Model",
    "GFN2D4UnsupportedWarning",
    "GFN2ExperimentalWarning",
    "NDDOExperimentalWarning",
    "DFTB0RepulsivePlaceholderWarning",
    "PM6Model",
    "UPM6Model",
    "OMxModel",
    "PeriodicPM6Model",
    "PeriodicOMxModel",
    "DispersionCorrectedModel",
    "run_dftb0",
    "run_dftb0_seccm",
    "run_scc_dftb",
    "SCCDFTB_SECCM_PARAMETER_SET",
    "SCCDFTBSECCMResult",
    "run_scc_dftb_seccm",
    "PM6_SECCM_PARAMETER_SET",
    "PM6SECCMResult",
    "run_pm6_seccm",
    "OMxSECCMResult",
    "run_omx_seccm",
    "GFN2_SECCM_PARAMETER_SET",
    "GFN2SECCMAttempt",
    "GFN2SECCMConvergenceError",
    "GFN2SECCMResult",
    "GFN2SECCMStateChange",
    "compare_gfn2_seccm_states",
    "SCFNonConvergenceError",
    "GFN2SECCMHamiltonianIdentity",
    "GFN2SECCMRestartIdentity",
    "GFN2SECCMRunControls",
    "run_gfn2_seccm",
    "d3bj_energy",
    "SemiempiricalParameters",
    "DFTB_D3_DEFAULTS",
    "compute_stress_fd",
    "optimize_cell",
    "optimize_pm6_cell",
    "optimize_omx_cell",
    "preoptimize_molecule",
    "preoptimize_periodic",
    "MOLECULAR_SEMIEMPIRICAL_METHODS",
    "SEMIEMPIRICAL_METHODS",
    "SemiempiricalEnergyError",
    "SemiempiricalResult",
    "SemiempiricalRoutePlan",
    "run_ccm",
    "run_seccm",
    "run_semiempirical",
    "native_route_descriptor",
    "run_native_energy",
    "BACKEND_GATED_EXPERIMENTAL",
    "BACKEND_MIXED_NATIVE",
    "BACKEND_NATIVE",
    "BACKEND_NATIVE_FD",
    "BACKEND_PYTHON_REFERENCE",
    "REFERENCE_ONLY_SEMIEMPIRICAL_ROUTES",
    "SEMIEMPIRICAL_ROUTE_ALIASES",
    "SEMIEMPIRICAL_ROUTE_STATUS",
    "SemiempiricalRouteStatus",
    "semiempirical_route_runtime_available",
    "semiempirical_route_status",
]
