"""Hamiltonian-independent topology support for semiempirical CCM routes."""

from __future__ import annotations

from .dftb0 import (
    DFTB0_SECCM_PARAMETER_SET,
    DFTB0SECCMResult,
    run_dftb0_seccm,
)
from .gfn2 import (
    GFN2_SECCM_PARAMETER_SET,
    GFN2SECCMAttempt,
    GFN2SECCMConvergenceError,
    GFN2SECCMResult,
    GFN2SECCMStateChange,
    compare_gfn2_seccm_states,
    run_gfn2_seccm,
)
from .omx import OMxSECCMResult, run_omx_seccm
from .pm6 import PM6_SECCM_PARAMETER_SET, PM6SECCMResult, run_pm6_seccm
from .scc_dftb import (
    SCCDFTB_SECCM_PARAMETER_SET,
    SCCDFTBSECCMResult,
    run_scc_dftb_seccm,
)
from .topology import (
    AtomEquivalenceKey,
    FiniteTranslationLabel,
    ImageShellLabel,
    SECCMCandidateClassification,
    SECCMFiniteGroup,
    SECCMImage,
    SECCMTopology,
    SECCMTopologyChangedError,
    SECCMTopologyError,
    SECCMTopologyProvenanceError,
    SECCMTranslationAction,
    TopologyDiagnostic,
    TopologyDiagnosticCode,
    WSNeighbor,
    WignerSeitzCells,
    bind_complete_translation_action,
    bind_finite_group,
    build_seccm_topology,
    build_wigner_seitz,
)

__all__ = [
    "AtomEquivalenceKey",
    "DFTB0_SECCM_PARAMETER_SET",
    "DFTB0SECCMResult",
    "FiniteTranslationLabel",
    "ImageShellLabel",
    "SECCMCandidateClassification",
    "SECCMFiniteGroup",
    "SECCMImage",
    "SECCMTopology",
    "SECCMTopologyChangedError",
    "SECCMTopologyError",
    "SECCMTopologyProvenanceError",
    "SECCMTranslationAction",
    "TopologyDiagnostic",
    "TopologyDiagnosticCode",
    "WSNeighbor",
    "WignerSeitzCells",
    "bind_complete_translation_action",
    "bind_finite_group",
    "build_seccm_topology",
    "build_wigner_seitz",
    "run_dftb0_seccm",
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
    "run_gfn2_seccm",
]
