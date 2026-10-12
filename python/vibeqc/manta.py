"""Experimental MANTA integral providers.

The reference builder uses libint; DirectJKBuilder uses independent MD kernels.
Both use an unreduced ordered shell traversal, without changing SCF defaults.
Cartesian coefficients retain libint's shell-wide axial normalization.
Returned tiles own their storage; native consumers use workspace-borrowed views.
Resource plans include an explicit backend allowance and are not RSS proofs.
"""

from ._vibeqc_core._manta import (
    HFGradientPlan,
    HFGradientReceipt,
    HFGradientResult,
    plan_hf_lagrangian_gradient,
    compute_hf_lagrangian_gradient,
    BatchPlan,
    BatchResult,
    BuildPlan,
    BuildReceipt,
    Capability,
    DirectJKBuilder,
    DerivativePlan,
    DerivativeTiles,
    plan_derivatives,
    FourierPlan,
    FourierReceipt,
    FourierResult,
    plan_fourier,
    plan_auxiliary_fourier,
    compute_fourier_panel,
    compute_auxiliary_fourier_panel,
    FittingPlan,
    FittingTiles,
    plan_fitting,
    JKResult,
    KernelRequest,
    OneElectronRequest,
    OneElectronDerivative,
    OneElectronPlan,
    OneElectronReceipt,
    OneElectronResult,
    plan_one_electron,
    compute_one_electron,
    plan_one_electron_derivative,
    compute_one_electron_derivative,
    ReferenceJKBuilder,
    ResourceBudget,
    RIOptions,
    RIPlan,
    RIMetricInfo,
    RIReceipt,
    RIResult,
    RIExchangePair,
    RIJKBuilder,
    plan_ri,
    TileSession,
    capability,
    md_capability,
    md_plan,
    boys,
    range_boys,
    cartesian_to_spherical,
    plan,
)

__all__ = [
    "BatchPlan", "BatchResult",
    "HFGradientPlan", "HFGradientReceipt", "HFGradientResult",
    "plan_hf_lagrangian_gradient", "compute_hf_lagrangian_gradient",
    "DerivativePlan", "DerivativeTiles", "plan_derivatives",
    "RIOptions", "RIPlan", "RIMetricInfo", "RIReceipt", "RIResult", "RIExchangePair", "RIJKBuilder", "plan_ri",
    "BuildPlan", "BuildReceipt", "Capability", "JKResult", "KernelRequest",
    "ReferenceJKBuilder", "ResourceBudget", "TileSession", "capability", "plan",
    "OneElectronRequest", "OneElectronPlan", "OneElectronReceipt", "OneElectronResult",
    "plan_one_electron", "compute_one_electron",
    "OneElectronDerivative", "plan_one_electron_derivative", "compute_one_electron_derivative",
    "FourierPlan", "FourierReceipt", "FourierResult", "plan_fourier", "plan_auxiliary_fourier",
    "compute_fourier_panel", "compute_auxiliary_fourier_panel",
    "FittingPlan", "FittingTiles", "plan_fitting",
    "DirectJKBuilder", "md_capability", "md_plan", "boys", "range_boys", "cartesian_to_spherical",
]

from .manta_scf import (
    IntegralEngineOptions, CoulombBuildOptions, ExchangeBuildOptions,
    molecular_engine_provenance,
)
__all__ += ["IntegralEngineOptions", "CoulombBuildOptions", "ExchangeBuildOptions", "molecular_engine_provenance"]

from .manta_geometry import MolecularHFGeometrySource
__all__ += ["MolecularHFGeometrySource"]

from ._vibeqc_core._manta import (
    BlochFourierPlan, BlochFourierReceipt, BlochFourierResult,
    plan_bloch_fourier, compute_bloch_fourier_panel,
)
__all__ += ["BlochFourierPlan", "BlochFourierReceipt", "BlochFourierResult",
            "plan_bloch_fourier", "compute_bloch_fourier_panel"]

from ._vibeqc_core._manta import (
    PeriodicHartreePlan, PeriodicHartreeReceipt, PeriodicHartreeResult,
    plan_periodic_hartree, compute_periodic_hartree,
)
__all__ += ["PeriodicHartreePlan", "PeriodicHartreeReceipt", "PeriodicHartreeResult",
            "plan_periodic_hartree", "compute_periodic_hartree"]

# Conservative molecular BUPO shell-pair geometry, independent of J/K selection.
from ._vibeqc_core._manta import (
    ExtentOptions, ExtentPlan, PairExtent, ExtentReceipt, PairExtents,
    plan_pair_extents, compute_pair_extents,
)
__all__ += ["ExtentOptions", "ExtentPlan", "PairExtent", "ExtentReceipt", "PairExtents",
            "plan_pair_extents", "compute_pair_extents"]

from ._vibeqc_core._manta import (
    MultipolePlan, MultipoleMatrix, plan_multipoles, multipole_translation,
    multipole_coulomb, compact_multipole_remainder,
)
__all__ += ["MultipolePlan", "MultipoleMatrix", "plan_multipoles", "multipole_translation",
            "multipole_coulomb", "compact_multipole_remainder"]

# Bounded 3D Ewald complement; not a periodic BUPO traversal or error certificate.
from ._vibeqc_core._manta import (
    LatticeTailPlan, LatticeTailMatrix, plan_lattice_tail, multipole_lattice_tail,
)
__all__ += ["LatticeTailPlan", "LatticeTailMatrix", "plan_lattice_tail", "multipole_lattice_tail"]

from ._vibeqc_core._manta import (
    BubbleOptions, BubblePlan, BubbleNode, BubbleBlock, BubbleTree,
    BubbleTraversal, BubbleTraversalReceipt, plan_bubbles,
)
__all__ += ["BubbleOptions", "BubblePlan", "BubbleNode", "BubbleBlock", "BubbleTree",
            "BubbleTraversal", "BubbleTraversalReceipt", "plan_bubbles"]

from ._vibeqc_core._manta import (
    PairMomentPlan, PairMomentReceipt, PairMomentResult,
    plan_pair_moments, compute_pair_moments,
    AuxiliaryMomentPlan, AuxiliaryMomentReceipt, AuxiliaryMomentResult,
    plan_auxiliary_moments, compute_auxiliary_moments,
)
__all__ += ["PairMomentPlan", "PairMomentReceipt", "PairMomentResult",
            "plan_pair_moments", "compute_pair_moments",
            "AuxiliaryMomentPlan", "AuxiliaryMomentReceipt", "AuxiliaryMomentResult",
            "plan_auxiliary_moments", "compute_auxiliary_moments"]

from ._vibeqc_core._manta import (
    BubbleErrorPlan, BubbleEnvelope, BubbleErrorBound, BubbleInteraction,
    BubbleErrorModel, plan_bubble_errors,
)
__all__ += ["BubbleErrorPlan", "BubbleEnvelope", "BubbleErrorBound", "BubbleInteraction",
            "BubbleErrorModel", "plan_bubble_errors"]

from ._vibeqc_core._manta import ShellTileSession, plan_shell_tiles
__all__ += ["ShellTileSession", "plan_shell_tiles"]

from ._vibeqc_core._manta import BupoOptions, BupoPlan, BupoReceipt, BupoResult, BupoJBuilder
__all__ += ["BupoOptions", "BupoPlan", "BupoReceipt", "BupoResult", "BupoJBuilder"]

from ._vibeqc_core._manta import (
    AuxiliaryExtentPlan, AuxiliaryExtent, AuxiliaryExtentReceipt, AuxiliaryExtents,
    plan_auxiliary_extents, compute_auxiliary_extents,
)
__all__ += ["AuxiliaryExtentPlan", "AuxiliaryExtent", "AuxiliaryExtentReceipt", "AuxiliaryExtents",
            "plan_auxiliary_extents", "compute_auxiliary_extents"]

from ._vibeqc_core._manta import (
    BupoFittingOptions, BupoFittingPlan, BupoFittingReceipt,
    BupoFittingForward, BupoFittingAdjoint, BupoFittingCoupling,
)
__all__ += ["BupoFittingOptions", "BupoFittingPlan", "BupoFittingReceipt",
            "BupoFittingForward", "BupoFittingAdjoint", "BupoFittingCoupling"]

from ._vibeqc_core._manta import BupoRIPlan, BupoRIReceipt, BupoRIResult, BupoRIJBuilder
__all__ += ["BupoRIPlan", "BupoRIReceipt", "BupoRIResult", "BupoRIJBuilder"]

from ._vibeqc_core._manta import (
    RIMetricResponsePlan, RIMetricResponseInfo, RIMetricDerivative,
    RIMetricResponse, plan_ri_metric_response,
)
__all__ += ["RIMetricResponsePlan", "RIMetricResponseInfo", "RIMetricDerivative",
            "RIMetricResponse", "plan_ri_metric_response"]

from ._vibeqc_core._manta import (
    RIJGradientPlan, RIJGradientReceipt, RIJGradientResult,
    plan_ri_coulomb_gradient, compute_ri_coulomb_gradient,
)
__all__ += ["RIJGradientPlan", "RIJGradientReceipt", "RIJGradientResult",
            "plan_ri_coulomb_gradient", "compute_ri_coulomb_gradient"]

from ._vibeqc_core._manta import (
    RIKGradientPlan, RIKGradientReceipt, RIKGradientResult,
    plan_ri_exchange_gradient, compute_ri_exchange_gradient,
)
__all__ += ["RIKGradientPlan", "RIKGradientReceipt", "RIKGradientResult",
            "plan_ri_exchange_gradient", "compute_ri_exchange_gradient"]

from ._vibeqc_core._manta import (
    RIHFGradientResult, plan_hf_ri_lagrangian_gradient, compute_hf_ri_lagrangian_gradient,
)
__all__ += ["RIHFGradientResult", "plan_hf_ri_lagrangian_gradient", "compute_hf_ri_lagrangian_gradient"]

from ._vibeqc_core._manta import (
    BlochOneElectronPlan, BlochOneElectronReceipt, BlochOneElectronResult,
    plan_bloch_one_electron, compute_bloch_one_electron,
)
__all__ += ["BlochOneElectronPlan", "BlochOneElectronReceipt", "BlochOneElectronResult",
            "plan_bloch_one_electron", "compute_bloch_one_electron"]

from .manta_periodic import compute_one_electron_lattice
__all__ += ["compute_one_electron_lattice"]

from ._vibeqc_core._manta import (
    PeriodicNuclearPlan, PeriodicNuclearReceipt, PeriodicNuclearResult,
    plan_periodic_nuclear, compute_periodic_nuclear,
)
__all__ += ["PeriodicNuclearPlan", "PeriodicNuclearReceipt", "PeriodicNuclearResult",
            "plan_periodic_nuclear", "compute_periodic_nuclear"]

from .manta_periodic import compute_nuclear_lattice
__all__ += ["compute_nuclear_lattice"]

from ._vibeqc_core._manta import (
    PeriodicExchangePlan, PeriodicExchangeReceipt, PeriodicExchangeResult,
    plan_periodic_exchange, compute_periodic_exchange,
)
__all__ += ["PeriodicExchangePlan", "PeriodicExchangeReceipt", "PeriodicExchangeResult",
            "plan_periodic_exchange", "compute_periodic_exchange"]

from .manta_periodic import build_periodic_exchange
__all__ += ["build_periodic_exchange"]

from ._vibeqc_core._manta import (
    PeriodicFockPlan, PeriodicFockReceipt, PeriodicFockResult,
    plan_periodic_fock, compute_periodic_fock,
)
__all__ += ["PeriodicFockPlan", "PeriodicFockReceipt", "PeriodicFockResult",
            "plan_periodic_fock", "compute_periodic_fock"]

from .manta_periodic import build_periodic_fock
__all__ += ["build_periodic_fock"]

from ._vibeqc_core._manta import (
    PeriodicRHFOptions, PeriodicRHFPlan, PeriodicRHFReceipt, PeriodicRHFResult, PeriodicMadelungPlan,
    plan_periodic_rhf,
)
__all__ += ["PeriodicRHFOptions", "PeriodicRHFPlan", "PeriodicRHFReceipt", "PeriodicRHFResult",
            "plan_periodic_rhf", "run_periodic_rhf", "PeriodicMadelungPlan"]

from .manta_periodic import run_periodic_rhf

# Reuse the canonical exact-address mesh used by existing periodic consumers.
from ._vibeqc_core import _RegularKMesh as RegularKMesh
__all__ += ["RegularKMesh"]

from ._vibeqc_core._manta import (
    PeriodicRHFDomainPlan, PeriodicRHFDomainReceipt, PeriodicRHFDomains,
    plan_periodic_rhf_from_cutoffs, build_periodic_rhf_domains,
)
from .manta_periodic import run_periodic_rhf_from_cutoffs
__all__ += ["PeriodicRHFDomainPlan", "PeriodicRHFDomainReceipt", "PeriodicRHFDomains",
            "plan_periodic_rhf_from_cutoffs", "build_periodic_rhf_domains", "run_periodic_rhf_from_cutoffs"]

from .manta_periodic_job import (PeriodicIntegralOptions, PeriodicXCOptions, PeriodicFittingOptions, PeriodicMANTAResult,
    PeriodicMANTAUHFResult, PeriodicMANTARKSResult, PeriodicMANTAUKSResult)
__all__ += ["PeriodicIntegralOptions", "PeriodicXCOptions", "PeriodicFittingOptions", "PeriodicMANTAResult", "PeriodicMANTAUHFResult",
            "PeriodicMANTARKSResult", "PeriodicMANTAUKSResult"]

from ._vibeqc_core._manta import (
    PeriodicUHFOptions, PeriodicUHFPlan, PeriodicUHFReceipt, PeriodicUHFResult,
    plan_periodic_uhf,
)
from .manta_periodic import run_periodic_uhf
__all__ += ["PeriodicUHFOptions", "PeriodicUHFPlan", "PeriodicUHFReceipt", "PeriodicUHFResult",
            "plan_periodic_uhf", "run_periodic_uhf"]

from ._vibeqc_core._manta import (
    PeriodicUHFDomainPlan, PeriodicUHFDomains,
    plan_periodic_uhf_from_cutoffs, build_periodic_uhf_domains,
)
from .manta_periodic import run_periodic_uhf_from_cutoffs
__all__ += ["PeriodicUHFDomainPlan", "PeriodicUHFDomains",
            "plan_periodic_uhf_from_cutoffs", "build_periodic_uhf_domains", "run_periodic_uhf_from_cutoffs"]

from ._vibeqc_core._manta import (
    PeriodicLocalExchangePlan, PeriodicLocalExchangeReceipt, PeriodicLocalExchangeResult,
    plan_periodic_local_exchange, compute_periodic_local_exchange,
)
__all__ += ["PeriodicLocalExchangePlan", "PeriodicLocalExchangeReceipt", "PeriodicLocalExchangeResult",
            "plan_periodic_local_exchange", "compute_periodic_local_exchange"]

from ._vibeqc_core._manta import (
    PeriodicSemilocalXCPlan, PeriodicSemilocalXCReceipt, PeriodicSemilocalXCResult,
    plan_periodic_semilocal_xc, compute_periodic_semilocal_xc,
)
__all__ += ["PeriodicSemilocalXCPlan", "PeriodicSemilocalXCReceipt", "PeriodicSemilocalXCResult",
            "plan_periodic_semilocal_xc", "compute_periodic_semilocal_xc"]

from ._vibeqc_core._manta import (
    PeriodicKSOptions, PeriodicKSPlan, PeriodicKSReceipt, PeriodicKSResult,
    PeriodicRIKSOptions, PeriodicRIGlobalHybridKSOptions,
    plan_periodic_ri_ks, plan_periodic_ri_global_hybrid_ks,
    PeriodicGlobalHybridKSOptions, plan_periodic_global_hybrid_ks, run_periodic_global_hybrid_ks,
    PeriodicRangeSeparatedKSOptions, PeriodicKSExchangeComponentPlan, PeriodicKSExchangeComponentReceipt,
    PeriodicRIRangeSeparatedKSOptions, plan_periodic_ri_range_separated_ks,
    plan_periodic_range_separated_ks, run_periodic_range_separated_ks,
    plan_periodic_ks, run_periodic_ks,
)
from .manta_periodic import run_periodic_ri_ks, run_periodic_ri_global_hybrid_ks, run_periodic_ri_range_separated_ks
__all__ += ["PeriodicKSOptions", "PeriodicKSPlan", "PeriodicKSReceipt", "PeriodicKSResult",
            "PeriodicRIKSOptions", "PeriodicRIGlobalHybridKSOptions", "plan_periodic_ri_ks", "run_periodic_ri_ks",
            "plan_periodic_ri_global_hybrid_ks", "run_periodic_ri_global_hybrid_ks",
            "plan_periodic_ks", "run_periodic_ks", "PeriodicGlobalHybridKSOptions",
            "plan_periodic_global_hybrid_ks", "run_periodic_global_hybrid_ks",
            "PeriodicRangeSeparatedKSOptions", "PeriodicKSExchangeComponentPlan", "PeriodicKSExchangeComponentReceipt",
            "plan_periodic_range_separated_ks", "run_periodic_range_separated_ks"]
__all__ += ["PeriodicRIRangeSeparatedKSOptions", "plan_periodic_ri_range_separated_ks", "run_periodic_ri_range_separated_ks"]

from ._vibeqc_core._manta import (
    PeriodicKSDomainPlan, PeriodicKSDomainReceipt, PeriodicKSDomains,
    plan_periodic_ks_from_cutoffs, build_periodic_ks_domains, run_periodic_ks_from_cutoffs,
)
__all__ += ["PeriodicKSDomainPlan", "PeriodicKSDomainReceipt", "PeriodicKSDomains",
            "plan_periodic_ks_from_cutoffs", "build_periodic_ks_domains", "run_periodic_ks_from_cutoffs"]

from ._vibeqc_core._manta import (
    PeriodicQuadratureOptions, PeriodicQuadraturePlan, PeriodicQuadratureReceipt, PeriodicQuadratureResult,
    plan_periodic_quadrature, build_periodic_quadrature,
)
__all__ += ["PeriodicQuadratureOptions", "PeriodicQuadraturePlan", "PeriodicQuadratureReceipt", "PeriodicQuadratureResult",
            "plan_periodic_quadrature", "build_periodic_quadrature"]

from ._vibeqc_core._manta import (
    PeriodicKSGridPlan, plan_periodic_ks_with_quadrature, run_periodic_ks_with_quadrature,
)
__all__ += ["PeriodicKSGridPlan", "plan_periodic_ks_with_quadrature", "run_periodic_ks_with_quadrature"]

from ._vibeqc_core._manta import (
    PeriodicGlobalHybridKSDomainPlan, PeriodicGlobalHybridKSDomainReceipt, PeriodicGlobalHybridKSDomains,
    PeriodicGlobalHybridKSGridPlan, plan_periodic_global_hybrid_ks_from_cutoffs,
    build_periodic_global_hybrid_ks_domains, run_periodic_global_hybrid_ks_from_cutoffs,
    plan_periodic_global_hybrid_ks_with_quadrature, run_periodic_global_hybrid_ks_with_quadrature,
)
__all__ += ["PeriodicGlobalHybridKSDomainPlan", "PeriodicGlobalHybridKSDomainReceipt", "PeriodicGlobalHybridKSDomains",
            "PeriodicGlobalHybridKSGridPlan", "plan_periodic_global_hybrid_ks_from_cutoffs",
            "build_periodic_global_hybrid_ks_domains", "run_periodic_global_hybrid_ks_from_cutoffs",
            "plan_periodic_global_hybrid_ks_with_quadrature", "run_periodic_global_hybrid_ks_with_quadrature"]

from ._vibeqc_core._manta import (
    PeriodicRangeSeparatedKSDomainPlan, PeriodicRangeSeparatedKSDomainReceipt, PeriodicRangeSeparatedKSDomains,
    PeriodicRangeSeparatedKSGridPlan, plan_periodic_range_separated_ks_from_cutoffs,
    build_periodic_range_separated_ks_domains, run_periodic_range_separated_ks_from_cutoffs,
    plan_periodic_range_separated_ks_with_quadrature, run_periodic_range_separated_ks_with_quadrature,
)
__all__ += ["PeriodicRangeSeparatedKSDomainPlan", "PeriodicRangeSeparatedKSDomainReceipt", "PeriodicRangeSeparatedKSDomains",
            "PeriodicRangeSeparatedKSGridPlan", "plan_periodic_range_separated_ks_from_cutoffs",
            "build_periodic_range_separated_ks_domains", "run_periodic_range_separated_ks_from_cutoffs",
            "plan_periodic_range_separated_ks_with_quadrature", "run_periodic_range_separated_ks_with_quadrature"]

from ._vibeqc_core._manta import (
    PeriodicXCRefinementPlan, PeriodicXCRefinementStep, PeriodicXCRefinementResult,
    plan_periodic_xc_refinement, assess_periodic_xc_refinement,
    plan_periodic_hybrid_xc_refinement, assess_periodic_hybrid_xc_refinement,
)
__all__ += ["PeriodicXCRefinementPlan", "PeriodicXCRefinementStep", "PeriodicXCRefinementResult",
            "plan_periodic_xc_refinement", "assess_periodic_xc_refinement",
            "plan_periodic_hybrid_xc_refinement", "assess_periodic_hybrid_xc_refinement"]

from ._vibeqc_core._manta import (
    PeriodicGDFShortRangeOptions, PeriodicGDFShortRangePlan, PeriodicGDFShortRangeReceipt,
    PeriodicGDFShortRangeResult, plan_periodic_gdf_short_range, compute_periodic_gdf_short_range,
)
__all__ += ["PeriodicGDFShortRangeOptions", "PeriodicGDFShortRangePlan", "PeriodicGDFShortRangeReceipt",
            "PeriodicGDFShortRangeResult", "plan_periodic_gdf_short_range", "compute_periodic_gdf_short_range"]

from ._vibeqc_core._manta import (
    PeriodicGDFKernelOptions, PeriodicGDFPlan, PeriodicGDFReceipt, PeriodicGDFResult, plan_periodic_gdf, compute_periodic_gdf,
)
__all__ += ["PeriodicGDFKernelOptions", "PeriodicGDFPlan", "PeriodicGDFReceipt", "PeriodicGDFResult", "plan_periodic_gdf", "compute_periodic_gdf"]

from ._vibeqc_core._manta import PeriodicGDFTilesPlan, PeriodicGDFTile, PeriodicGDFTiles, plan_periodic_gdf_tiles
__all__ += ["PeriodicGDFTilesPlan", "PeriodicGDFTile", "PeriodicGDFTiles", "plan_periodic_gdf_tiles"]

from ._vibeqc_core._manta import (
    PeriodicRIQPlan, PeriodicRIQMetricInfo, PeriodicRIQReceipt, PeriodicRIQResult,
    PeriodicRIQBuilder, plan_periodic_ri_q,
)
__all__ += [
    "PeriodicRIQPlan", "PeriodicRIQMetricInfo", "PeriodicRIQReceipt", "PeriodicRIQResult",
    "PeriodicRIQBuilder", "plan_periodic_ri_q",
]

from ._vibeqc_core._manta import (
    PeriodicRIFockPlan, PeriodicRIFockInfo, PeriodicRIFockReceipt, PeriodicRIFockResult,
    PeriodicRIFockBuilder, plan_periodic_ri_fock,
)
__all__ += [
    "PeriodicRIFockPlan", "PeriodicRIFockInfo", "PeriodicRIFockReceipt", "PeriodicRIFockResult",
    "PeriodicRIFockBuilder", "plan_periodic_ri_fock",
]

from ._vibeqc_core._manta import (
    PeriodicRIUHFOptions, PeriodicRIUHFPlan, PeriodicRIUHFReceipt, PeriodicRIUHFResult,
    plan_periodic_ri_uhf,
)
from .manta_periodic import run_periodic_ri_uhf
__all__ += [
    "PeriodicRIUHFOptions", "PeriodicRIUHFPlan", "PeriodicRIUHFReceipt", "PeriodicRIUHFResult",
    "plan_periodic_ri_uhf", "run_periodic_ri_uhf",
]
from ._vibeqc_core._manta import (
    PeriodicRIUHFDomainPlan, PeriodicRIUHFDomains,
    plan_periodic_ri_uhf_from_cutoffs, build_periodic_ri_uhf_domains,
)
from .manta_periodic import run_periodic_ri_uhf_from_cutoffs
__all__ += ["PeriodicRIUHFDomainPlan", "PeriodicRIUHFDomains",
            "plan_periodic_ri_uhf_from_cutoffs", "build_periodic_ri_uhf_domains",
            "run_periodic_ri_uhf_from_cutoffs"]

from ._vibeqc_core._manta import (
    PeriodicRIRHFOptions, PeriodicRIRHFPlan, PeriodicRIRHFReceipt, PeriodicRIRHFResult,
    plan_periodic_ri_rhf,
)
from .manta_periodic import run_periodic_ri_rhf
__all__ += [
    "PeriodicRIRHFOptions", "PeriodicRIRHFPlan", "PeriodicRIRHFReceipt", "PeriodicRIRHFResult",
    "plan_periodic_ri_rhf", "run_periodic_ri_rhf",
]
from ._vibeqc_core._manta import (
    PeriodicRIRHFDomainPlan, PeriodicRIRHFDomains,
    plan_periodic_ri_rhf_from_cutoffs, build_periodic_ri_rhf_domains,
)
from .manta_periodic import run_periodic_ri_rhf_from_cutoffs
__all__ += ["PeriodicRIRHFDomainPlan", "PeriodicRIRHFDomains",
            "plan_periodic_ri_rhf_from_cutoffs", "build_periodic_ri_rhf_domains",
            "run_periodic_ri_rhf_from_cutoffs"]

# Pure fitted factories reuse the general pure KS domain/grid schema; the
# nested KS plan retains the complete fixed-projector fitting admission.
from ._vibeqc_core._manta import (
    plan_periodic_ri_ks_from_cutoffs, build_periodic_ri_ks_domains,
    plan_periodic_ri_ks_with_quadrature,
    PeriodicRIGlobalHybridKSDomainPlan, PeriodicRIGlobalHybridKSDomainReceipt,
    PeriodicRIGlobalHybridKSDomains, PeriodicRIGlobalHybridKSGridPlan,
    plan_periodic_ri_global_hybrid_ks_from_cutoffs, build_periodic_ri_global_hybrid_ks_domains,
    plan_periodic_ri_global_hybrid_ks_with_quadrature,
)
from .manta_periodic import (
    run_periodic_ri_ks_from_cutoffs, run_periodic_ri_ks_with_quadrature,
    run_periodic_ri_global_hybrid_ks_from_cutoffs, run_periodic_ri_global_hybrid_ks_with_quadrature,
)
__all__ += [
    "plan_periodic_ri_ks_from_cutoffs", "build_periodic_ri_ks_domains",
    "plan_periodic_ri_ks_with_quadrature", "run_periodic_ri_ks_from_cutoffs", "run_periodic_ri_ks_with_quadrature",
    "PeriodicRIGlobalHybridKSDomainPlan", "PeriodicRIGlobalHybridKSDomainReceipt",
    "PeriodicRIGlobalHybridKSDomains", "PeriodicRIGlobalHybridKSGridPlan",
    "plan_periodic_ri_global_hybrid_ks_from_cutoffs", "build_periodic_ri_global_hybrid_ks_domains",
    "plan_periodic_ri_global_hybrid_ks_with_quadrature", "run_periodic_ri_global_hybrid_ks_from_cutoffs",
    "run_periodic_ri_global_hybrid_ks_with_quadrature",
]

from ._vibeqc_core._manta import (
    PeriodicRIRangeSeparatedKSDomainPlan, PeriodicRIRangeSeparatedKSDomainReceipt,
    PeriodicRIRangeSeparatedKSDomains, PeriodicRIRangeSeparatedKSGridPlan,
    plan_periodic_ri_range_separated_ks_from_cutoffs, build_periodic_ri_range_separated_ks_domains,
    plan_periodic_ri_range_separated_ks_with_quadrature,
)
from .manta_periodic import (
    run_periodic_ri_range_separated_ks_from_cutoffs, run_periodic_ri_range_separated_ks_with_quadrature,
)
__all__ += [
    "PeriodicRIRangeSeparatedKSDomainPlan", "PeriodicRIRangeSeparatedKSDomainReceipt",
    "PeriodicRIRangeSeparatedKSDomains", "PeriodicRIRangeSeparatedKSGridPlan",
    "plan_periodic_ri_range_separated_ks_from_cutoffs", "build_periodic_ri_range_separated_ks_domains",
    "plan_periodic_ri_range_separated_ks_with_quadrature", "run_periodic_ri_range_separated_ks_from_cutoffs",
    "run_periodic_ri_range_separated_ks_with_quadrature",
]
