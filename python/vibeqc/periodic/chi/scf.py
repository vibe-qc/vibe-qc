"""Independent finite-torus HF/KS formulation for ``aiccm2026dev-b``.

The module deliberately does not import :mod:`vibeqc.periodic.ccm`.  Its
finite cyclic cluster is the Born--von Karman translation group
``Z/N1 x Z/N2 x Z/N3``.  The SCF is evaluated in the group's irreducible
representations (a Gamma-centred k mesh), which is an exact unitary change
of basis from a translation-invariant Gamma-point supercell calculation for
the same already specified χ-CCM finite Hamiltonian.  This internal
representation theorem does not identify χ-CCM with the separately
constructed union-and-weight Γ-CCM approach.

Wigner--Seitz weights below resolve tied minimum-image representatives.
They form a partition of unity for a translation class; they are not
centre-dependent multipliers on ordinary free-space electron-repulsion
integrals.  That distinction preserves the permutation symmetries required
by a single RHF energy functional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from itertools import product
from time import perf_counter
from typing import Sequence
import warnings

import numpy as np

from ..._vibeqc_core import (
    BasisSet,
    Functional,
    GridOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    PeriodicXCDensityDomain,
)
from ...kpoints import KPoints
from ...level_shift_schedule import LevelShiftSchedule
from ...mpi import LatticeOutputPartition, mpi_world
from ...pbc_bipole import PBCBipoleRHFResult, run_pbc_bipole_rhf
from ...pbc_bipole_common import resolve_fock_mixing
from ...pbc_bipole_fock import (
    BipoleOutputCellFarmingExecution,
    BipoleScreenedExchangeExecution,
)
from ...pbc_bipole_rks import PBCBipoleRKSResult, run_pbc_bipole_rks
from ...pbc_bipole_uhf import PBCBipoleUHFResult, run_pbc_bipole_uhf
from ...pbc_bipole_uks import PBCBipoleUKSResult, run_pbc_bipole_uks
from ...pbc_gdf import _auto_rsgdf_tail_ke_cutoff
from ...periodic_k_gdf import (
    PeriodicKRHFGDFResult,
    PeriodicKRKSGDFResult,
    PeriodicKUHFGDFResult,
    run_krhf_periodic_gdf,
    run_krks_periodic_gdf,
    run_kuhf_periodic_gdf,
    run_kuks_periodic_gdf,
)
from .symmetry import (
    AICCM2026DevBSymmetryDiagnostics,
    AICCM2026DevBSymmetryMode,
    AICCM2026DevBSymmetryPlan,
    build_aiccm2026dev_b_symmetry_plan,
    gamma_matrix_symmetry_residual,
    shell_pair_orbits,
    shell_quartet_orbits,
)
from ...progress import ProgressLogger
from ...periodic_screened_exchange import (
    PeriodicExchangeAssembly,
    reject_unscreened_range_separated,
    resolve_periodic_exchange,
)

__all__ = [
    "AICCM2026DevBChargeBookkeeping",
    "AICCM2026DevBDiagnostics",
    "AICCM2026DevBBackend",
    "AICCM2026DevBExactExchangeAssembly",
    "AICCM2026DevBExperimentalWarning",
    "AICCM2026DevBFiniteTorusConvention",
    "AICCM2026DevBLatticeExtension",
    "AICCM2026DevBResidueTransformExecution",
    "WignerSeitzRepresentative",
    "aiccm2026dev_b_charge_bookkeeping",
    "cyclic_lattice_extension",
    "cyclic_gamma_mesh",
    "inverse_bloch_transform",
    "pair_wigner_seitz_representatives",
    "rhf_idempotency_error",
    "run_aiccm2026dev_b_rhf",
    "run_aiccm2026dev_b_rks",
    "run_aiccm2026dev_b_uhf",
    "run_aiccm2026dev_b_uks",
    "uhf_idempotency_error",
    "wigner_seitz_representatives",
]


class AICCM2026DevBExperimentalWarning(UserWarning):
    """The independently derived B stream is not production-certified."""


def _warn_experimental() -> None:
    warnings.warn(
        "aiccm2026dev-b is experimental: match the dimensional Coulomb gauge, "
        "converge the cyclic lattice extension, and inspect the attached "
        "invariants before "
        "using an energy quantitatively",
        category=AICCM2026DevBExperimentalWarning,
        stacklevel=3,
    )


class AICCM2026DevBBackend(str, Enum):
    """Electron-repulsion backend for the same finite-torus functional."""

    FOUR_CENTER = "four_center"
    RI = "ri"
    RIJCOSX = "rijcosx"


_CCM_APPROACH = "chi-ccm"
_CCM_CONSTRUCTION = "finite-translation-group-character"
_EVALUATION_REPRESENTATION = "gamma-centred-character-mesh"
_DIRECT_RUNTIME_BACKEND = "pbc-bipole"
_M5_SR_IMAGE_DOMAIN_POLICY = "m5-qqr-padded-erfc/v1"
_M5_SR_PHYSICAL_DOMAIN_POLICY = "m5-physical-pair-midpoint-erfc/v1"
_M5_SR_IMAGE_PRECISION = 1.0e-6
_EXACT_EXCHANGE_ASSEMBLY_SCHEMA = (
    "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
)
_EXACT_EXCHANGE_RESOLVER = (
    "vibeqc.periodic_screened_exchange.resolve_periodic_exchange"
)
_SCREENED_EXCHANGE_ASSEMBLIES = {
    "short-range-direct",
    "full-range-minus-long-range",
}
_EXTERNAL_XC_CAPABILITY_VERSION = 1
_EXTERNAL_XC_GRID_PROFILES = frozenset({"", "pyscf-level3"})


def _require_supported_external_xc(
    functional: Functional,
    backend: AICCM2026DevBBackend,
    *,
    system: PeriodicSystem,
    where: str,
) -> bool:
    """Validate the deliberately narrow external-XC χ capability.

    Returns ``False`` for libxc functionals and ``True`` for an accepted
    provider. Unsupported capabilities and backends fail closed here,
    including for callers that bypass :func:`run_periodic_job` and invoke the
    χ runners directly.
    """

    if not bool(getattr(functional, "is_external", False)):
        return False

    name = str(getattr(functional, "name", "")).strip().lower()
    if int(system.dim) != 3:
        raise NotImplementedError(
            f"{where}: external XC provider {name!r} through experimental "
            "Chi-CCM requires the validated 3D periodic Coulomb gauge"
        )
    if backend is not AICCM2026DevBBackend.FOUR_CENTER:
        raise NotImplementedError(
            f"{where}: external XC provider {name!r} through experimental "
            "Chi-CCM is currently accepted only with "
            "backend='four_center'. The RI and RIJCOSX routes remain gated "
            "pending an independent fitted-XC acceptance calculation."
        )

    capabilities = getattr(functional, "external_capabilities", None)
    if not isinstance(capabilities, dict):
        raise RuntimeError(
            f"{where}: external-XC capability metadata is unavailable for "
            f"provider {name!r}"
        )
    version = int(capabilities.get("version", -1))
    profile = str(capabilities.get("required_grid_profile", "")).strip().lower()
    if (
        version != _EXTERNAL_XC_CAPABILITY_VERSION
        or profile not in _EXTERNAL_XC_GRID_PROFILES
    ):
        raise RuntimeError(
            f"{where}: unsupported external-XC capability for {name!r} "
            f"(version={version}, required_grid_profile={profile!r})"
        )
    if float(getattr(functional, "hf_exchange_fraction", 0.0)) != 0.0:
        raise NotImplementedError(
            f"{where}: external XC provider {name!r} has a nonzero exact-"
            "exchange fraction. The accepted χ provider shape is pure "
            "full-grid XC; external-provider hybrids remain gated."
        )
    return True


def _configure_external_xc_grid(
    functional: Functional,
    options: PeriodicKSOptions,
    *,
    caller_supplied_options: bool,
    where: str,
) -> None:
    """Resolve a v1 provider's grid profile before χ setup or SCF work."""

    if not bool(getattr(functional, "is_external", False)):
        return
    if not bool(options.use_periodic_becke):
        raise ValueError(
            f"{where}: external XC requires use_periodic_becke=True so the "
            "periodic-lattice density and integration partition use the same "
            "image domain"
        )
    image_radius = float(options.becke_image_radius_bohr)
    if image_radius <= 0.0:
        raise ValueError(
            f"{where}: external XC requires becke_image_radius_bohr > 0"
        )
    if float(options.lattice_opts.cutoff_bohr) < image_radius:
        raise ValueError(
            f"{where}: external XC lattice cutoff_bohr must be at least "
            "becke_image_radius_bohr"
        )
    options.lattice_opts.becke_image_radius_bohr = image_radius
    capabilities = getattr(functional, "external_capabilities", None) or {}
    required = str(capabilities.get("required_grid_profile", "") or "")
    if not required:
        return
    required_options = GridOptions()
    required_options.atomic_grid_profile = required
    if not caller_supplied_options:
        options.grid.atomic_grid_profile = required_options.atomic_grid_profile
        return
    if options.grid.atomic_grid_profile != required_options.atomic_grid_profile:
        actual = str(options.grid.atomic_grid_profile).rsplit(".", 1)[-1]
        raise ValueError(
            f"{where}: external XC functional {functional.name!r} requires "
            f"grid profile {required!r}, but the supplied PeriodicKSOptions "
            f"selects {actual!r}."
        )


@dataclass(frozen=True)
class AICCM2026DevBExactExchangeAssembly:
    """Versioned route-resolved exact-exchange assembly for one B SCF run.

    The descriptor mirrors the shared periodic functional resolver used by
    the selected SCF route.  A B runner attaches an active screened label only
    after the selected backend exposes matching branch-level execution
    evidence.
    ``omega_screen_bohr_inv`` is the physical erfc screening parameter and is
    deliberately named so it cannot be confused with a numerical Ewald split
    parameter.
    """

    c_full: float
    c_sr: float
    omega_screen_bohr_inv: float
    screened_exchange_applicability: str = "inactive"
    screened_exchange_assembly: str = "not-applicable"
    schema: str = field(init=False, default=_EXACT_EXCHANGE_ASSEMBLY_SCHEMA)
    resolver: str = field(init=False, default=_EXACT_EXCHANGE_RESOLVER)

    def __post_init__(self) -> None:
        normalized: dict[str, float] = {}
        for name in ("c_full", "c_sr", "omega_screen_bohr_inv"):
            value = getattr(self, name)
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a finite float; got {value!r}")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{name} must be a finite float; got {value!r}"
                ) from exc
            if not np.isfinite(number):
                raise ValueError(f"{name} must be finite; got {value!r}")
            normalized[name] = number
            object.__setattr__(self, name, number)

        c_sr = normalized["c_sr"]
        c_full = normalized["c_full"]
        omega_screen = normalized["omega_screen_bohr_inv"]
        if c_sr == 0.0 and omega_screen != 0.0:
            raise ValueError(
                "omega_screen_bohr_inv must be zero when c_sr is zero; "
                f"got {omega_screen!r}"
            )
        if c_sr < 0.0:
            raise ValueError(
                "c_sr must be nonnegative on supported χ-CCM-B routes; "
                f"got {c_sr!r}"
            )
        if c_sr != 0.0 and omega_screen <= 0.0:
            raise ValueError(
                "omega_screen_bohr_inv must be positive when c_sr is nonzero; "
                f"got {omega_screen!r}"
            )
        if c_full != 0.0 and c_sr != 0.0:
            raise ValueError(
                "χ-CCM-B does not support simultaneous full-range and "
                "screened exact-exchange arms"
            )

        applicability = self.screened_exchange_applicability
        if not isinstance(applicability, str) or applicability not in {
            "active",
            "inactive",
        }:
            raise ValueError(
                "screened_exchange_applicability must be 'active' or "
                f"'inactive'; got {applicability!r}"
            )
        expected_applicability = "active" if c_sr != 0.0 else "inactive"
        if applicability != expected_applicability:
            raise ValueError(
                "screened_exchange_applicability must be "
                f"{expected_applicability!r} when c_sr={c_sr!r}; "
                f"got {applicability!r}"
            )

        assembly = self.screened_exchange_assembly
        if not isinstance(assembly, str):
            raise ValueError(
                "screened_exchange_assembly must be text; "
                f"got {assembly!r}"
            )
        if applicability == "inactive":
            if assembly != "not-applicable":
                raise ValueError(
                    "inactive screened exchange must use "
                    "screened_exchange_assembly='not-applicable'"
                )
        elif assembly not in _SCREENED_EXCHANGE_ASSEMBLIES:
            raise ValueError(
                "active screened exchange must use "
                "'short-range-direct' or 'full-range-minus-long-range'; "
                f"got {assembly!r}"
            )


@dataclass(frozen=True)
class AICCM2026DevBFiniteTorusConvention:
    """Finite-N Hamiltonian convention attached to every B-stream result.

    The descriptor is intentionally explicit about the exchange ``q=0`` seam:
    finite-N HF, MP2, CCSD(T), and local-correlation numbers are comparable
    only when this convention matches.  A strict-zero-mode exchange reference
    is a different finite torus, even though it has the same thermodynamic
    target.
    """

    coulomb_kernel: str
    exchange_q0: str
    exchange_q0_applicability: str
    boundary_model: str
    periodic_dimension: int
    character_mesh_shape: tuple[int, int, int]
    bvk_madelung_supercell_repetitions: tuple[int, int, int]
    bvk_madelung_supercell_lattice_bohr: tuple[tuple[float, float, float], ...]
    lattice_vector_convention: str = "columns"
    ccm_approach: str = _CCM_APPROACH
    ccm_construction: str = _CCM_CONSTRUCTION
    evaluation_representation: str = _EVALUATION_REPRESENTATION
    orbital_energy_convention: str = field(init=False)

    def __post_init__(self) -> None:
        identity = {
            "ccm_approach": (self.ccm_approach, _CCM_APPROACH),
            "ccm_construction": (self.ccm_construction, _CCM_CONSTRUCTION),
            "evaluation_representation": (
                self.evaluation_representation,
                _EVALUATION_REPRESENTATION,
            ),
        }
        for field_name, (actual, expected) in identity.items():
            if actual != expected:
                raise ValueError(
                    f"{field_name} must be {expected!r}; got {actual!r}"
                )
        if self.lattice_vector_convention != "columns":
            raise ValueError(
                "lattice_vector_convention must be 'columns'; "
                f"got {self.lattice_vector_convention!r}"
            )
        if self.exchange_q0_applicability == "active":
            orbital_energy_convention = (
                "HF/Kohn-Sham eigenvalues include the declared exchange_q0 seam"
            )
        elif self.exchange_q0_applicability == "inactive":
            orbital_energy_convention = (
                "Kohn-Sham eigenvalues do not include the declared exchange_q0 "
                "seam because the full-range exact-exchange coefficient is zero"
            )
        else:
            raise ValueError(
                "exchange_q0_applicability must be 'active' or 'inactive'; "
                f"got {self.exchange_q0_applicability!r}"
            )
        object.__setattr__(
            self,
            "orbital_energy_convention",
            orbital_energy_convention,
        )


@dataclass(frozen=True)
class AICCM2026DevBChargeBookkeeping:
    """Effective charge/electron convention used by a χ-CCM SCF run.

    ``physical_electrons`` is the full all-electron primitive-cell count from
    :class:`PeriodicSystem`.  When a periodic ECP is attached to the SCF
    options, ``ecp_total_ncore`` electrons are removed from the variational
    space and the Coulomb background must use the corresponding
    ``effective_nuclear_charges``.  The effective net charge is therefore
    ``sum(effective_nuclear_charges) - effective_electrons``; production
    χ-CCM routes require it to vanish.
    """

    physical_electrons: int
    effective_electrons: int
    ecp_total_ncore: int
    effective_nuclear_charges: tuple[float, ...]
    effective_nuclear_charge: float
    effective_net_charge: float
    has_ecp: bool


@dataclass(frozen=True)
class WignerSeitzRepresentative:
    """One tied minimum-image representative of a cyclic translation.

    ``residue`` labels the element of the finite translation group.
    ``translation`` is an integer primitive-cell translation representing
    that residue, and all representatives for one residue have equal
    ``weight = 1 / multiplicity``.
    """

    residue: tuple[int, int, int]
    translation: tuple[int, int, int]
    displacement_bohr: tuple[float, float, float]
    weight: float


@dataclass(frozen=True)
class AICCM2026DevBLatticeExtension:
    """Real-space definition of the finite Born--von Karman torus.

    ``repetitions`` gives the number of primitive translations in each
    lattice direction.  The cyclic supercell vectors are
    ``A_i = repetitions[i] * a_i`` and its Wigner--Seitz half-width is
    ``repetitions[i] / 2`` in primitive-vector coordinates.  The reciprocal
    character group is therefore exactly the Gamma-centred uniform net with
    the same integer tuple; it is a representation of this real-space
    object, not an independent approximation parameter.
    """

    repetitions: tuple[int, int, int]
    wigner_seitz_half_extent: tuple[float, float, float]
    supercell_lattice_bohr: np.ndarray
    n_cells: int


@dataclass(frozen=True)
class AICCM2026DevBResidueTransformExecution:
    """Executed χ finite-group residue inverse-transform schedule.

    One task is one element of the finite translation group.  The supplied
    Wigner--Seitz representative set is grouped before tasks are created;
    its weights certify a partition of unity but do not multiply the
    finite-character transform a second time.  Production density diagnostics
    identify that set explicitly as the zero-offset cell representatives, not
    pair- or quartet-offset representatives.
    """

    strategy: str
    world_size: int
    rank: int
    n_residues: int
    n_representatives: int
    character_mesh: tuple[int, int, int]
    character_labels: tuple[tuple[int, int, int], ...]
    residue_keys: tuple[tuple[int, int, int], ...]
    local_residue_indices: tuple[int, ...]
    local_residue_keys: tuple[tuple[int, int, int], ...]
    task_counts: tuple[int, ...]
    ordered_residue_fingerprint: str
    construction_fingerprint: str
    representative_scope: str
    representative_multiplicities: tuple[int, ...]
    representative_weight_sums: tuple[float, ...]
    schema: str = field(
        init=False,
        default="vibeqc.aiccm2026dev-b.residue-inverse-transform-execution/v1",
    )
    task_kind: str = field(init=False, default="chi-residue-inverse-transform")
    result_distribution: str = field(
        init=False,
        default="all-ranks-canonical-residue-order",
    )
    complete_character_sum: bool = field(init=False, default=True)
    supplied_representatives_grouped: bool = field(init=False, default=True)
    extra_wigner_weight_applied: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        """Reject internally contradictory public execution records."""

        if self.strategy not in ("block", "cyclic"):
            raise ValueError("residue transform strategy must be block or cyclic")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("residue transform rank/world size is inconsistent")
        if len(self.character_mesh) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
            for value in self.character_mesh
        ):
            raise ValueError("residue transform character mesh must be positive 3-D")
        mesh = tuple(int(value) for value in self.character_mesh)
        expected_keys = tuple(product(*(range(value) for value in mesh)))
        if self.n_residues != len(expected_keys) or self.residue_keys != expected_keys:
            raise ValueError("residue transform keys do not cover the finite group")
        if (
            len(self.character_labels) != self.n_residues
            or len(set(self.character_labels)) != self.n_residues
            or set(self.character_labels) != set(expected_keys)
        ):
            raise ValueError("residue transform labels do not cover the dual group")
        if len(self.representative_multiplicities) != self.n_residues or any(
            value <= 0 for value in self.representative_multiplicities
        ):
            raise ValueError("residue transform representative counts are invalid")
        if self.n_representatives != sum(self.representative_multiplicities):
            raise ValueError("residue transform representative census is inconsistent")
        if len(self.representative_weight_sums) != self.n_residues or any(
            not np.isclose(value, 1.0, atol=1.0e-12, rtol=0.0)
            for value in self.representative_weight_sums
        ):
            raise ValueError("residue transform representative weights are invalid")
        if not isinstance(self.representative_scope, str) or not self.representative_scope:
            raise ValueError("residue transform representative scope must be nonempty")

        if self.strategy == "cyclic":
            expected_local = tuple(range(self.rank, self.n_residues, self.world_size))
            expected_counts = tuple(
                len(range(rank, self.n_residues, self.world_size))
                for rank in range(self.world_size)
            )
        else:
            base, remainder = divmod(self.n_residues, self.world_size)

            def _block_indices(rank: int) -> tuple[int, ...]:
                start = rank * base + min(rank, remainder)
                stop = start + base + (1 if rank < remainder else 0)
                return tuple(range(start, stop))

            expected_local = _block_indices(self.rank)
            expected_counts = tuple(
                len(_block_indices(rank)) for rank in range(self.world_size)
            )
        if self.local_residue_indices != expected_local:
            raise ValueError("residue transform local ownership is inconsistent")
        if self.local_residue_keys != tuple(
            self.residue_keys[index] for index in expected_local
        ):
            raise ValueError("residue transform local keys are inconsistent")
        if self.task_counts != expected_counts:
            raise ValueError("residue transform per-rank census is inconsistent")
        for label, fingerprint in (
            ("ordered-residue", self.ordered_residue_fingerprint),
            ("construction", self.construction_fingerprint),
        ):
            if len(fingerprint) != 64 or any(
                character not in "0123456789abcdef" for character in fingerprint
            ):
                raise ValueError(f"residue transform {label} fingerprint is invalid")

    @property
    def active(self) -> bool:
        """Whether more than one MPI rank executed the declared schedule."""

        return self.world_size > 1


@dataclass(frozen=True)
class AICCM2026DevBDiagnostics:
    """Internal-consistency data attached to an AICCM2026DEV_B result."""

    mesh: tuple[int, int, int]
    n_cyclic_cells: int
    n_kpoints: int
    wigner_seitz_partition_error: float
    density_idempotency_error: float
    electron_count_error: float
    inverse_bloch_imaginary_residual: float
    backend: str
    electronic_method: str
    wall_time_seconds: float
    exact_exchange_assembly: AICCM2026DevBExactExchangeAssembly
    lattice_extension: tuple[int, int, int] | None = None
    wigner_seitz_half_extent: tuple[float, float, float] | None = None
    n_alpha_error: float | None = None
    n_beta_error: float | None = None
    s_squared: float | None = None
    s_squared_ideal: float | None = None
    scf_trace_length: int = 0
    final_scf_delta_e_ha: float | None = None
    final_scf_grad_norm: float | None = None
    final_scf_diis_subspace: int | None = None
    physical_electron_count: int | None = None
    effective_electron_count: int | None = None
    ecp_total_ncore: int = 0
    effective_nuclear_charge: float | None = None
    effective_nuclear_charges: tuple[float, ...] | None = None
    effective_net_charge: float | None = None
    use_diis: bool | None = None
    diis_start_iter: int | None = None
    diis_subspace_size: int | None = None
    scf_accelerator: str | None = None
    damping: float | None = None
    dynamic_damping: bool | None = None
    # Executed effective previous-Fock weight, after backend defaults.
    fock_mixing: float | None = None
    level_shift: float | None = None
    # Executed effective warm-up length; zero is persistent, inactive,
    # or superseded by an explicit schedule.
    level_shift_warmup_cycles: int | None = None
    smearing_temperature: float | None = None
    variational_space: str = "translation-invariant closed-shell determinants"
    coulomb_gauge: str = "neutral BvK-periodic Coulomb; G=0 removed"
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention | None = None
    direct_output_cell_farming: BipoleOutputCellFarmingExecution | None = None
    residue_inverse_transform: AICCM2026DevBResidueTransformExecution | None = None

    @property
    def coulomb_kernel(self) -> str:
        if self.finite_torus_convention is None:
            return "not-recorded"
        return self.finite_torus_convention.coulomb_kernel

    @property
    def exchange_q0(self) -> str:
        if self.finite_torus_convention is None:
            return "not-recorded"
        return self.finite_torus_convention.exchange_q0

    @property
    def exchange_q0_applicability(self) -> str:
        if self.finite_torus_convention is None:
            return "not-recorded"
        return self.finite_torus_convention.exchange_q0_applicability

    @property
    def boundary_model(self) -> str:
        if self.finite_torus_convention is None:
            return "not-recorded"
        return self.finite_torus_convention.boundary_model

    @property
    def ccm_approach(self) -> str:
        if self.finite_torus_convention is None:
            return "not-recorded"
        return self.finite_torus_convention.ccm_approach

    @property
    def ccm_construction(self) -> str:
        if self.finite_torus_convention is None:
            return "not-recorded"
        return self.finite_torus_convention.ccm_construction

    @property
    def evaluation_representation(self) -> str:
        if self.finite_torus_convention is None:
            return "not-recorded"
        return self.finite_torus_convention.evaluation_representation


def _normalise_mesh(
    system: PeriodicSystem,
    mesh: int | Sequence[int],
) -> tuple[int, int, int]:
    dim = int(system.dim)
    if dim not in (1, 2, 3):
        raise ValueError(f"aiccm2026dev-b requires dim=1, 2, or 3; got {dim}")
    if isinstance(mesh, (int, np.integer)):
        values = [int(mesh)] * dim
    else:
        values = [int(value) for value in mesh]
    if len(values) == dim:
        values += [1] * (3 - dim)
    elif len(values) != 3:
        raise ValueError(
            f"aiccm2026dev-b mesh must have length {dim} or 3 for dim={dim}; "
            f"got {values!r}"
        )
    if any(value < 1 for value in values):
        raise ValueError(f"aiccm2026dev-b mesh entries must be >= 1; got {values!r}")
    if any(values[axis] != 1 for axis in range(dim, 3)):
        raise ValueError(
            "aiccm2026dev-b inactive lattice directions must have mesh size 1; "
            f"got {values!r} for dim={dim}"
        )
    return tuple(values)  # type: ignore[return-value]


def _resolve_lattice_extension(
    system: PeriodicSystem,
    lattice_extension: int | Sequence[int] | None,
    *,
    mesh: int | Sequence[int] | None = None,
    wigner_seitz_shells: int | Sequence[int] | None = None,
) -> tuple[int, int, int]:
    """Resolve mutually exclusive real- and reciprocal-space controls.

    ``wigner_seitz_shells=s`` requests ``s`` complete primitive-cell layers
    on either side of the central cell, hence the odd cyclic order
    ``N=2s+1``.  ``lattice_extension=N`` also admits even orders; then the
    boundary at ``+/-N/2`` is shared and receives the exact tied-image
    weights returned by :func:`wigner_seitz_representatives`.
    """

    supplied = sum(
        value is not None for value in (lattice_extension, mesh, wigner_seitz_shells)
    )
    if supplied > 1:
        raise ValueError(
            "choose exactly one of lattice_extension, mesh, or "
            "wigner_seitz_shells"
        )
    if wigner_seitz_shells is not None:
        dim = int(system.dim)
        if isinstance(wigner_seitz_shells, (int, np.integer)):
            shells = [int(wigner_seitz_shells)] * dim
        else:
            shells = [int(value) for value in wigner_seitz_shells]
        if len(shells) == dim:
            shells += [0] * (3 - dim)
        if len(shells) != 3 or any(value < 0 for value in shells):
            raise ValueError("wigner_seitz_shells must contain nonnegative integers")
        if any(shells[axis] != 0 for axis in range(dim, 3)):
            raise ValueError("inactive directions must have zero Wigner--Seitz shells")
        return _normalise_mesh(
            system,
            [2 * shells[axis] + 1 if axis < dim else 1 for axis in range(3)],
        )
    selected = mesh if mesh is not None else lattice_extension
    if selected is None:
        selected = [1] * int(system.dim)
    return _normalise_mesh(system, selected)


def cyclic_lattice_extension(
    system: PeriodicSystem,
    lattice_extension: int | Sequence[int] | None = None,
    *,
    mesh: int | Sequence[int] | None = None,
    wigner_seitz_shells: int | Sequence[int] | None = None,
) -> AICCM2026DevBLatticeExtension:
    """Construct the user-facing real-space cyclic extension descriptor."""

    repetitions = _resolve_lattice_extension(
        system,
        lattice_extension,
        mesh=mesh,
        wigner_seitz_shells=wigner_seitz_shells,
    )
    lattice = np.asarray(system.lattice, dtype=float)
    supercell = lattice @ np.diag(np.asarray(repetitions, dtype=float))
    return AICCM2026DevBLatticeExtension(
        repetitions=repetitions,
        wigner_seitz_half_extent=tuple(0.5 * value for value in repetitions),
        supercell_lattice_bohr=supercell,
        n_cells=int(np.prod(repetitions)),
    )


def _finite_torus_convention(
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
    extension: AICCM2026DevBLatticeExtension | None = None,
    *,
    full_range_exchange_coefficient: float,
) -> AICCM2026DevBFiniteTorusConvention:
    """Return the declared finite-N Hamiltonian convention for B production."""

    dim = int(system.dim)
    if extension is None:
        extension = cyclic_lattice_extension(system, mesh)
    c_full = float(full_range_exchange_coefficient)
    if not np.isfinite(c_full):
        raise ValueError(
            "full_range_exchange_coefficient must be finite; "
            f"got {full_range_exchange_coefficient!r}"
        )
    exchange_q0_applicability = "active" if c_full != 0.0 else "inactive"
    boundary_model = "3d-periodic" if dim == 3 else "3d-periodic-vacuum"
    lattice_matrix = tuple(
        tuple(float(value) for value in row)
        for row in np.asarray(extension.supercell_lattice_bohr, dtype=float)
    )
    return AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel="3d-periodic-g0",
        exchange_q0="bvk-ewald",
        exchange_q0_applicability=exchange_q0_applicability,
        boundary_model=boundary_model,
        periodic_dimension=dim,
        character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=extension.repetitions,
        bvk_madelung_supercell_lattice_bohr=lattice_matrix,
    )


def _resolve_exact_exchange_coefficients(
    functional: str | None,
    *,
    spin: int,
    where: str,
) -> PeriodicExchangeAssembly:
    """Resolve selected-route coefficients through the shared EXX contract."""

    func = None if functional is None else Functional(str(functional), int(spin))
    return resolve_periodic_exchange(func, where=where)


def _finalize_exact_exchange_assembly(
    result: object,
    backend: AICCM2026DevBBackend,
    coefficients: PeriodicExchangeAssembly,
    *,
    where: str,
) -> AICCM2026DevBExactExchangeAssembly:
    """Bind resolver coefficients to the screened branch that actually ran."""

    execution = getattr(result, "screened_exchange_execution", None)
    c_full = float(coefficients.c_full)
    c_sr = float(coefficients.c_sr)
    omega_screen = float(coefficients.omega_screen)
    if c_sr == 0.0:
        if execution is not None:
            raise RuntimeError(
                f"{where}: the backend reported screened-exchange execution "
                "for a route with c_sr=0"
            )
        return AICCM2026DevBExactExchangeAssembly(
            c_full=c_full,
            c_sr=c_sr,
            omega_screen_bohr_inv=omega_screen,
            screened_exchange_applicability="inactive",
            screened_exchange_assembly="not-applicable",
        )

    if backend is not AICCM2026DevBBackend.FOUR_CENTER:
        raise RuntimeError(
            f"{where}: active screened exchange is not implemented for "
            f"backend={backend.value!r}"
        )
    if not isinstance(execution, BipoleScreenedExchangeExecution):
        raise RuntimeError(
            f"{where}: active screened exchange lacks BIPOLE branch-level "
            "execution evidence"
        )
    expected_execution = BipoleScreenedExchangeExecution(
        c_sr=c_sr,
        omega_screen_bohr_inv=omega_screen,
    )
    if execution != expected_execution:
        raise RuntimeError(
            f"{where}: BIPOLE screened-exchange execution evidence "
            "contradicts the live resolver"
        )
    return AICCM2026DevBExactExchangeAssembly(
        c_full=c_full,
        c_sr=c_sr,
        omega_screen_bohr_inv=omega_screen,
        screened_exchange_applicability="active",
        screened_exchange_assembly=execution.assembly,
    )


def cyclic_gamma_mesh(
    system: PeriodicSystem,
    mesh: int | Sequence[int],
) -> KPoints:
    """Return the reciprocal irreps of a finite cyclic cluster.

    A cluster with ``N1*N2*N3`` primitive cells has exactly the uniform,
    unreduced Gamma-centred ``(N1,N2,N3)`` reciprocal mesh.  No classical
    even-mesh half shift and no symmetry reduction are allowed here because
    either would change the finite translation group.
    """

    mesh_tuple = _normalise_mesh(system, mesh)
    return KPoints.gamma_centred(system, mesh_tuple, symmetry=False)


def _nearest_representatives(
    active_lattice: np.ndarray,
    mesh: tuple[int, int, int],
    residue: tuple[int, int, int],
    dim: int,
    tolerance: float,
    offset_bohr: np.ndarray,
) -> list[tuple[int, int, int]]:
    """Solve the small closest-vector problem with a proved stopping bound."""

    sigma_min = float(np.linalg.svd(active_lattice, compute_uv=False)[-1])
    if sigma_min <= 0.0:
        raise ValueError("aiccm2026dev-b requires linearly independent active vectors")

    best_sq = np.inf
    winners: list[tuple[int, int, int]] = []
    radius = 0
    while True:
        shifts = product(range(-radius, radius + 1), repeat=dim)
        for shift_active in shifts:
            translation = [0, 0, 0]
            for axis in range(dim):
                translation[axis] = residue[axis] + mesh[axis] * shift_active[axis]
            translation_tuple = tuple(translation)
            displacement = (
                active_lattice @ np.asarray(translation[:dim], dtype=float)
                + offset_bohr
            )
            norm_sq = float(displacement @ displacement)
            scale = max(1.0, best_sq if np.isfinite(best_sq) else 1.0)
            if norm_sq < best_sq - tolerance * scale:
                best_sq = norm_sq
                winners = [translation_tuple]
            elif abs(norm_sq - best_sq) <= tolerance * scale:
                if translation_tuple not in winners:
                    winners.append(translation_tuple)

        # If an unseen shift has a component outside [-radius, radius], its
        # integer coefficient has this lower bound.  Multiplication by the
        # smallest singular value gives a Cartesian-distance lower bound.
        unseen_component = min(
            mesh[axis] * (radius + 1) - abs(residue[axis])
            for axis in range(dim)
        )
        unseen_lower = max(
            0.0,
            sigma_min * max(0, unseen_component) - float(np.linalg.norm(offset_bohr)),
        )
        unseen_lower_sq = unseen_lower**2
        if unseen_lower_sq > best_sq + tolerance * max(1.0, best_sq):
            return sorted(winners)
        radius += 1
        if radius > 64:
            raise RuntimeError(
                "aiccm2026dev-b Wigner-Seitz closest-vector search did not "
                "reach its stopping bound"
            )


def wigner_seitz_representatives(
    system: PeriodicSystem,
    mesh: int | Sequence[int],
    *,
    tolerance: float = 1e-12,
    offset_bohr: Sequence[float] = (0.0, 0.0, 0.0),
) -> tuple[WignerSeitzRepresentative, ...]:
    """Construct the exact minimum-image partition of the cyclic group.

    ``offset_bohr`` is the intra-cell displacement from the centre defining
    the Wigner--Seitz region to the translated centre. Boundary points have
    more than one equally short representative. Each receives the inverse
    multiplicity, so the weights for every group element sum to one. This is
    the only Wigner--Seitz weighting used by ``aiccm2026dev-b``.
    """

    if tolerance <= 0.0:
        raise ValueError("aiccm2026dev-b Wigner-Seitz tolerance must be positive")
    mesh_tuple = _normalise_mesh(system, mesh)
    dim = int(system.dim)
    lattice = np.asarray(system.lattice, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(f"aiccm2026dev-b lattice must have shape (3,3); got {lattice.shape}")
    # PeriodicSystem stores lattice vectors as columns.  Restrict the
    # closest-vector generator to the active primitive directions.
    active_lattice = lattice[:, :dim]
    offset = np.asarray(offset_bohr, dtype=float)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("aiccm2026dev-b Wigner-Seitz offset must be a finite 3-vector")

    output: list[WignerSeitzRepresentative] = []
    residue_ranges = [range(mesh_tuple[axis]) for axis in range(dim)]
    for active_residue in product(*residue_ranges):
        residue = tuple(active_residue) + (0,) * (3 - dim)
        representatives = _nearest_representatives(
            active_lattice,
            mesh_tuple,
            residue,
            dim,
            tolerance,
            offset,
        )
        weight = 1.0 / len(representatives)
        for translation in representatives:
            displacement = lattice @ np.asarray(translation, dtype=float) + offset
            output.append(
                WignerSeitzRepresentative(
                    residue=residue,
                    translation=translation,
                    displacement_bohr=tuple(float(value) for value in displacement),
                    weight=weight,
                )
            )
    return tuple(output)


def pair_wigner_seitz_representatives(
    system: PeriodicSystem,
    mesh: int | Sequence[int],
    centre_a_bohr: Sequence[float],
    centre_b_bohr: Sequence[float],
    *,
    tolerance: float = 1e-12,
) -> tuple[WignerSeitzRepresentative, ...]:
    """Wigner--Seitz representatives for translations of centre B about A."""

    centre_a = np.asarray(centre_a_bohr, dtype=float)
    centre_b = np.asarray(centre_b_bohr, dtype=float)
    if centre_a.shape != (3,) or centre_b.shape != (3,):
        raise ValueError("aiccm2026dev-b pair centres must be Cartesian 3-vectors")
    return wigner_seitz_representatives(
        system,
        mesh,
        tolerance=tolerance,
        offset_bohr=centre_b - centre_a,
    )


def _validated_inverse_bloch_inputs(
    matrices_k: Sequence[np.ndarray],
    kpoints_frac: np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = np.asarray(matrices_k, dtype=np.complex128)
    kfrac = np.asarray(kpoints_frac, dtype=float).reshape(-1, 3)
    if matrices.ndim != 3 or matrices.shape[0] != kfrac.shape[0]:
        raise ValueError(
            "inverse_bloch_transform expects one square matrix per k-point; "
            f"got matrices {matrices.shape}, k-points {kfrac.shape}"
        )
    if matrices.shape[1] != matrices.shape[2]:
        raise ValueError("inverse_bloch_transform matrices must be square")
    if weights is None:
        weights_array = np.full(kfrac.shape[0], 1.0 / kfrac.shape[0])
    else:
        weights_array = np.asarray(weights, dtype=float).reshape(-1)
    if weights_array.shape != (kfrac.shape[0],):
        raise ValueError("inverse_bloch_transform weight count does not match k-points")
    if not np.isclose(weights_array.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("inverse_bloch_transform weights must sum to one")
    return matrices, kfrac, weights_array


def inverse_bloch_transform(
    matrices_k: Sequence[np.ndarray],
    kpoints_frac: np.ndarray,
    translations: Sequence[Sequence[int]],
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Transform k-block matrices to primitive-translation blocks.

    The convention is ``M(R)=sum_k w_k exp(-2*pi*i*k.R) M(k)``.
    This is the exact finite-group Fourier transform, not a numerical
    quadrature approximation.
    """

    matrices, kfrac, weights_array = _validated_inverse_bloch_inputs(
        matrices_k,
        kpoints_frac,
        weights,
    )
    translations_array = np.asarray(translations, dtype=int).reshape(-1, 3)

    native = _inverse_bloch_transform_native(
        matrices,
        kfrac,
        translations_array,
        weights_array,
    )
    if native is not None:
        return native

    phases = np.exp(-2j * np.pi * (translations_array @ kfrac.T))
    return np.einsum("rk,k,kij->rij", phases, weights_array, matrices, optimize=True)


def _group_wigner_seitz_residues(
    representatives: Sequence[WignerSeitzRepresentative],
    mesh: tuple[int, int, int],
) -> tuple[
    tuple[tuple[int, int, int], ...],
    tuple[int, ...],
    tuple[float, ...],
    tuple[
        tuple[tuple[int, int, int], tuple[int, int, int], float],
        ...,
    ],
]:
    """Return canonical finite-group residues and their alias evidence."""

    if not representatives:
        raise ValueError(
            "aiccm2026dev-b residue transform requires at least one "
            "Wigner-Seitz representative"
        )
    grouped: dict[
        tuple[int, int, int],
        list[tuple[tuple[int, int, int], float]],
    ] = {}
    for position, representative in enumerate(representatives):
        try:
            raw_residue = tuple(representative.residue)
            raw_translation = tuple(representative.translation)
        except TypeError as exc:
            raise ValueError(
                "aiccm2026dev-b residue transform representatives require "
                "iterable residue and translation keys"
            ) from exc
        if len(raw_residue) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw_residue
        ):
            raise ValueError(
                "aiccm2026dev-b residue transform requires three-integer "
                f"residue keys; representative {position} has {raw_residue!r}"
            )
        residue = tuple(int(value) for value in raw_residue)
        if len(raw_translation) != 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in raw_translation
        ):
            raise ValueError(
                "aiccm2026dev-b residue transform requires three-integer "
                "representative translations; representative "
                f"{position} has {raw_translation!r}"
            )
        translation = tuple(int(value) for value in raw_translation)
        if tuple(
            translation[axis] % mesh[axis] for axis in range(3)
        ) != residue:
            raise ValueError(
                "aiccm2026dev-b residue transform representative translation "
                f"{translation!r} is not congruent to residue {residue!r}"
            )
        weight = float(representative.weight)
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "aiccm2026dev-b residue transform requires finite positive "
                f"representative weights; residue {residue!r} has {weight!r}"
            )
        grouped.setdefault(residue, []).append((translation, weight))

    residue_keys = tuple(product(*(range(value) for value in mesh)))
    if set(grouped) != set(residue_keys):
        raise ValueError(
            "aiccm2026dev-b residue transform representatives do not cover "
            "the complete finite translation group"
        )
    multiplicities: list[int] = []
    weight_sums: list[float] = []
    normalized: list[
        tuple[tuple[int, int, int], tuple[int, int, int], float]
    ] = []
    for residue in residue_keys:
        aliases = grouped[residue]
        translations = [translation for translation, _ in aliases]
        if len(set(translations)) != len(translations):
            raise ValueError(
                "aiccm2026dev-b residue transform found a duplicate "
                f"Wigner-Seitz representative for residue {residue!r}"
            )
        weight_sum = float(sum(weight for _, weight in aliases))
        if not np.isclose(weight_sum, 1.0, atol=1.0e-12, rtol=0.0):
            raise ValueError(
                "aiccm2026dev-b residue transform requires tied "
                "representative weights to sum to one; residue "
                f"{residue!r} sums to {weight_sum:.17g}"
            )
        expected_weight = 1.0 / len(aliases)
        if any(
            not np.isclose(weight, expected_weight, atol=1.0e-12, rtol=0.0)
            for _, weight in aliases
        ):
            raise ValueError(
                "aiccm2026dev-b residue transform requires equal weights "
                f"for tied representatives of residue {residue!r}"
            )
        multiplicities.append(len(aliases))
        weight_sums.append(weight_sum)
        normalized.extend(
            (residue, translation, weight)
            for translation, weight in sorted(aliases)
        )
    return (
        residue_keys,
        tuple(multiplicities),
        tuple(weight_sums),
        tuple(normalized),
    )


def _validated_residue_character_plan(
    kfrac: np.ndarray,
    weights: np.ndarray,
    mesh: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]]:
    """Bind a complete uniform dual character net to its finite group."""

    try:
        raw_mesh = tuple(mesh)
    except TypeError as exc:
        raise ValueError(
            "aiccm2026dev-b residue transform requires a 3-D mesh"
        ) from exc
    if len(raw_mesh) != 3 or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in raw_mesh
    ):
        raise ValueError(
            "aiccm2026dev-b residue transform mesh must contain three "
            "positive integers"
        )
    mesh_tuple = tuple(int(value) for value in raw_mesh)
    n_characters = int(np.prod(np.asarray(mesh_tuple, dtype=object)))
    if kfrac.shape[0] != n_characters:
        raise ValueError(
            "aiccm2026dev-b residue transform character count does not "
            "match the finite-group order"
        )
    if not np.all(np.isfinite(kfrac)):
        raise ValueError(
            "aiccm2026dev-b residue transform characters must be finite"
        )
    scaled = kfrac * np.asarray(mesh_tuple, dtype=float)[None, :]
    nearest = np.rint(scaled)
    if not np.allclose(scaled, nearest, atol=1.0e-10, rtol=0.0):
        raise ValueError(
            "aiccm2026dev-b residue transform characters are not on the "
            "declared dual finite-group net"
        )
    labels = tuple(
        tuple(
            int(nearest[index, axis]) % mesh_tuple[axis]
            for axis in range(3)
        )
        for index in range(n_characters)
    )
    expected = tuple(product(*(range(value) for value in mesh_tuple)))
    if len(set(labels)) != n_characters or set(labels) != set(expected):
        raise ValueError(
            "aiccm2026dev-b residue transform characters do not cover the "
            "complete dual finite group"
        )
    if not np.allclose(
        weights,
        np.full(n_characters, 1.0 / n_characters),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "aiccm2026dev-b residue transform requires uniform finite-group "
            "character weights"
        )
    return mesh_tuple, labels


def _residue_construction_fingerprint(
    mesh: tuple[int, int, int],
    kfrac: np.ndarray,
    weights: np.ndarray,
    labels: tuple[tuple[int, int, int], ...],
    representatives: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int], float],
        ...,
    ],
    representative_scope: str,
) -> str:
    """Hash the finite-group plan that surrounds the ordered task keys."""

    digest = sha256()
    digest.update(b"vibeqc-chi-residue-construction-v1\0")
    digest.update(representative_scope.encode("utf-8"))
    digest.update(b"\0")
    for value in mesh:
        digest.update(int(value).to_bytes(8, "big", signed=False))
    for label in labels:
        for value in label:
            digest.update(int(value).to_bytes(8, "big", signed=False))
    digest.update(np.asarray(kfrac, dtype=">f8").tobytes())
    digest.update(np.asarray(weights, dtype=">f8").tobytes())
    for residue, translation, weight in representatives:
        for value in (*residue, *translation):
            digest.update(int(value).to_bytes(8, "big", signed=True))
        digest.update(np.asarray([weight], dtype=">f8").tobytes())
    return digest.hexdigest()


def _require_matching_residue_construction(fingerprint: str) -> None:
    """Fail if active ranks supplied different finite-group plans."""

    world = mpi_world()
    if not world.active:
        return
    fingerprints = list(world.comm.allgather(fingerprint))
    if len(fingerprints) != world.size or any(
        value != fingerprint for value in fingerprints
    ):
        raise RuntimeError(
            "aiccm2026dev-b residue transform ranks disagree on the "
            "finite-group construction fingerprint"
        )


def _distributed_residue_inverse_bloch_transform(
    matrices_k: Sequence[np.ndarray],
    kpoints_frac: np.ndarray,
    representatives: Sequence[WignerSeitzRepresentative],
    weights: Sequence[float] | np.ndarray | None = None,
    *,
    mesh: Sequence[int],
    representative_scope: str,
    strategy: str = "cyclic",
) -> tuple[np.ndarray, AICCM2026DevBResidueTransformExecution]:
    """Transform complete characters on rank-owned χ residues exactly."""

    matrices, kfrac, weights_array = _validated_inverse_bloch_inputs(
        matrices_k,
        kpoints_frac,
        weights,
    )
    mesh_tuple, character_labels = _validated_residue_character_plan(
        kfrac,
        weights_array,
        mesh,
    )
    if not isinstance(representative_scope, str) or not representative_scope:
        raise ValueError(
            "aiccm2026dev-b residue transform representative scope must be "
            "nonempty"
        )
    residue_keys, multiplicities, weight_sums, normalized_representatives = (
        _group_wigner_seitz_residues(representatives, mesh_tuple)
    )
    construction_fingerprint = _residue_construction_fingerprint(
        mesh_tuple,
        kfrac,
        weights_array,
        character_labels,
        normalized_representatives,
        representative_scope,
    )
    _require_matching_residue_construction(construction_fingerprint)
    partition = LatticeOutputPartition.create(residue_keys, strategy=strategy)
    local_values: list[np.ndarray]
    if partition.n_local == 0:
        # The native subset convention uses an empty list to mean all outputs.
        # An oversubscribed rank must therefore skip it and contribute an
        # explicitly empty parcel to the exact ordered gather.
        local_values = []
    else:
        local_blocks = inverse_bloch_transform(
            matrices,
            kfrac,
            partition.local_task_keys,
            weights_array,
        )
        local_values = [np.asarray(block) for block in local_blocks]
    gathered = partition.allgather_ordered(local_values)
    blocks = np.stack([np.asarray(block) for block in gathered], axis=0)
    execution = AICCM2026DevBResidueTransformExecution(
        strategy=partition.strategy,
        world_size=partition.size,
        rank=partition.rank,
        n_residues=partition.n_tasks,
        n_representatives=len(representatives),
        character_mesh=mesh_tuple,
        character_labels=character_labels,
        residue_keys=partition.task_keys,
        local_residue_indices=partition.local_indices,
        local_residue_keys=partition.local_task_keys,
        task_counts=partition.task_counts,
        ordered_residue_fingerprint=partition.task_fingerprint,
        construction_fingerprint=construction_fingerprint,
        representative_scope=representative_scope,
        representative_multiplicities=multiplicities,
        representative_weight_sums=weight_sums,
    )
    return blocks, execution


def _inverse_bloch_transform_native(
    matrices: np.ndarray,
    kfrac: np.ndarray,
    translations: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray | None:
    """Return native inverse Bloch blocks, or ``None`` for older extensions."""

    try:
        from ... import _vibeqc_core
    except ImportError:
        return None
    kernel = getattr(_vibeqc_core, "aiccm2026dev_b_inverse_bloch_transform", None)
    if kernel is None:
        return None
    return np.asarray(
        kernel(
            np.ascontiguousarray(matrices, dtype=np.complex128),
            np.ascontiguousarray(kfrac, dtype=float),
            np.ascontiguousarray(translations, dtype=np.int64),
            np.ascontiguousarray(weights, dtype=float),
        ),
        dtype=np.complex128,
    )


def rhf_idempotency_error(
    density_k: Sequence[np.ndarray],
    overlap_k: Sequence[np.ndarray],
) -> float:
    """Maximum Frobenius residual of ``D(k) S(k) D(k) = 2 D(k)``."""

    if len(density_k) != len(overlap_k) or not density_k:
        raise ValueError("rhf_idempotency_error requires matching non-empty k blocks")
    return max(
        float(np.linalg.norm(density @ overlap @ density - 2.0 * density))
        for density, overlap in zip(density_k, overlap_k)
    )


def uhf_idempotency_error(
    density_alpha_k: Sequence[np.ndarray],
    density_beta_k: Sequence[np.ndarray],
    overlap_k: Sequence[np.ndarray],
) -> float:
    """Maximum spin-projector residual ``P_sigma S P_sigma - P_sigma``."""

    if not density_alpha_k or not (
        len(density_alpha_k) == len(density_beta_k) == len(overlap_k)
    ):
        raise ValueError("uhf_idempotency_error requires matching non-empty blocks")
    return max(
        float(np.linalg.norm(density @ overlap @ density - density))
        for blocks in (density_alpha_k, density_beta_k)
        for density, overlap in zip(blocks, overlap_k)
    )


def _electron_count_error(
    density_k: Sequence[np.ndarray],
    overlap_k: Sequence[np.ndarray],
    weights: np.ndarray,
    expected: int,
) -> float:
    count = sum(
        float(weight * np.trace(density @ overlap).real)
        for weight, density, overlap in zip(weights, density_k, overlap_k)
    )
    return abs(count - expected)


def _resolve_backend(
    backend: str | AICCM2026DevBBackend,
) -> AICCM2026DevBBackend:
    if isinstance(backend, AICCM2026DevBBackend):
        return backend
    try:
        return AICCM2026DevBBackend(str(backend).strip().lower().replace("-", "_"))
    except ValueError as exc:
        choices = ", ".join(member.value for member in AICCM2026DevBBackend)
        raise ValueError(
            f"unknown aiccm2026dev-b backend {backend!r}; choose {choices}"
        ) from exc


def _reject_four_center_aux_basis(
    backend: AICCM2026DevBBackend,
    aux_basis: str | None,
    *,
    where: str,
) -> None:
    """Reject an auxiliary-basis request that the direct operator cannot use."""

    if backend is not AICCM2026DevBBackend.FOUR_CENTER or aux_basis is None:
        return
    raise ValueError(
        f"{where}: aux_basis cannot be used with backend='four_center'. "
        "The complete four-center BIPOLE Hamiltonian has no density-fitting "
        "auxiliary basis; omit aux_basis, or select backend='ri' or "
        "backend='rijcosx' so the requested basis is executed."
    )


def _require_supported_scf_dimension(system: PeriodicSystem) -> None:
    """Reject dimensional Coulomb gauges that every current SCF path lacks.

    The 3D BIPOLE path has a neutral Ewald split. The current 1D/2D
    fallback is a direct-truncated interaction, so its absolute energy is
    not the declared finite-torus Hamiltonian. The lower-dimensional fitted
    mesh is also unavailable because collapsing every transverse reciprocal
    component does not define a Coulomb kernel. Returning a converged number
    from either family is worse than failing.
    """

    if int(system.dim) == 3:
        return
    raise NotImplementedError(
        "aiccm2026dev-b SCF is currently enabled only for 3D periodicity. "
        "The 1D/2D four-centre path lacks a neutral wire/slab "
        "Coulomb gauge and was found to over-bind chain benchmarks by a "
        "Madelung-scale constant shift. RI and RIJCOSX are not alternatives: "
        "their shared lower-dimensional mesh collapses transverse reciprocal "
        "structure and is not a Coulomb kernel. All backends remain blocked "
        "until one neutral wire/slab Hamiltonian is implemented."
    )


def _require_zero_unrestricted_level_shift(
    options: PeriodicRHFOptions | PeriodicKSOptions,
    *,
    where: str,
) -> None:
    """Fail closed until every unrestricted backend shares one shift operator."""

    try:
        static_shift = float(getattr(options, "level_shift", 0.0))
        raw_schedule = getattr(options, "level_shift_schedule", ())
        schedule = (
            ()
            if raw_schedule is None
            else tuple(float(value) for value in raw_schedule)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{where}: unrestricted level shifts must be finite scalars"
        ) from exc
    shifts = (static_shift, *schedule)
    if any(not np.isfinite(value) for value in shifts):
        raise ValueError(
            f"{where}: unrestricted level shifts must be finite scalars"
        )
    if any(value < 0.0 for value in shifts):
        raise ValueError(
            f"{where}: unrestricted level shifts must be non-negative"
        )
    if any(value != 0.0 for value in shifts):
        raise NotImplementedError(
            f"{where}: χ-CCM-B unrestricted level shifting is not validated. "
            "The four_center UHF/UKS implementation currently applies the "
            "restricted -(b/2) S D_sigma S coefficient to spin densities, "
            "while RI and RIJCOSX do not execute the requested shift. Use "
            "level_shift=0 with an empty or all-zero level_shift_schedule "
            "until one spin-projector convention is implemented and "
            "parity-tested across all three backends."
        )


def _require_zero_gamma_ri_rhf_level_shift(
    options: PeriodicRHFOptions,
    *,
    where: str,
) -> None:
    """Keep Gamma-only fitted RHF on the declared periodic operator."""

    try:
        static_shift = float(getattr(options, "level_shift", 0.0))
        raw_schedule = getattr(options, "level_shift_schedule", ())
        schedule = (
            ()
            if raw_schedule is None
            else tuple(float(value) for value in raw_schedule)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{where}: Gamma-only RI RHF level shifts must be finite scalars"
        ) from exc
    shifts = (static_shift, *schedule)
    if any(not np.isfinite(value) for value in shifts):
        raise ValueError(
            f"{where}: Gamma-only RI RHF level shifts must be finite scalars"
        )
    if any(value < 0.0 for value in shifts):
        raise ValueError(
            f"{where}: Gamma-only RI RHF level shifts must be non-negative"
        )
    if any(value != 0.0 for value in shifts):
        raise NotImplementedError(
            f"{where}: χ-CCM-B Gamma-only RI RHF cannot execute level "
            "shifting without leaving the declared fitted periodic "
            "operator or ignoring the request. A nonzero static shift "
            "selects the legacy cutoff-selected Gamma GDF fallback, while "
            "the pure Gamma PBC-GDF route does not execute a nonzero "
            "level_shift_schedule. Use level_shift=0 with an empty or "
            "all-zero schedule. For a static shift only, backend="
            "'four_center' is an alternative; scheduled shifting requires "
            "at least two cyclic cells."
        )


def _require_zero_restricted_four_center_level_shift_schedule(
    options: PeriodicRHFOptions | PeriodicKSOptions,
    *,
    where: str,
) -> None:
    """Reject a schedule that the restricted BIPOLE calls do not transport."""

    try:
        raw_schedule = getattr(options, "level_shift_schedule", ())
        schedule = (
            ()
            if raw_schedule is None
            else tuple(float(value) for value in raw_schedule)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{where}: restricted four_center level-shift schedule entries "
            "must be finite scalars"
        ) from exc
    if any(not np.isfinite(value) for value in schedule):
        raise ValueError(
            f"{where}: restricted four_center level-shift schedule entries "
            "must be finite scalars"
        )
    if any(value < 0.0 for value in schedule):
        raise ValueError(
            f"{where}: restricted four_center level-shift schedule entries "
            "must be non-negative"
        )
    if schedule:
        raise NotImplementedError(
            f"{where}: χ-CCM-B restricted four_center does not transport "
            "options.level_shift_schedule into the BIPOLE SCF driver. Use "
            "an empty schedule, or use a fitted backend with at least two "
            "cyclic cells. Static level_shift remains available on "
            "four_center only when the schedule is empty."
        )


def _resolve_restricted_four_center_level_shift_warmup(
    options: PeriodicRHFOptions | PeriodicKSOptions,
    *,
    where: str,
) -> tuple[LevelShiftSchedule | None, int]:
    """Lower the restricted four-center warm-up onto BIPOLE's schedule."""

    try:
        level_shift = float(getattr(options, "level_shift", 0.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{where}: restricted four_center level_shift must be a finite "
            "scalar"
        ) from exc
    if not np.isfinite(level_shift):
        raise ValueError(
            f"{where}: restricted four_center level_shift must be a finite "
            "scalar"
        )
    if level_shift < 0.0:
        raise ValueError(
            f"{where}: restricted four_center level_shift must be non-negative"
        )
    if level_shift == 0.0:
        # The warm-up modifier is dormant when there is no static shift.
        return None, 0

    raw_warmup = getattr(options, "level_shift_warmup_cycles", -1)
    if raw_warmup is None:
        raw_warmup = -1
    if isinstance(raw_warmup, (bool, np.bool_)):
        raise ValueError(
            f"{where}: level_shift_warmup_cycles must be an integer >= 0 "
            "or -1 for auto"
        )
    try:
        numeric_warmup = float(raw_warmup)
        warmup = int(raw_warmup)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{where}: level_shift_warmup_cycles must be an integer >= 0 "
            "or -1 for auto"
        ) from exc
    if (
        not np.isfinite(numeric_warmup)
        or numeric_warmup != float(warmup)
        or warmup < -1
    ):
        raise ValueError(
            f"{where}: level_shift_warmup_cycles must be an integer >= 0 "
            f"or -1 for auto; got {raw_warmup!r}"
        )

    if warmup == 0:
        # An explicit zero retains BIPOLE's persistent static-shift mode.
        return None, 0

    requested_cycles = 5 if warmup == -1 else warmup
    max_iter = int(getattr(options, "max_iter", 0))
    if max_iter <= 1:
        raise ValueError(
            f"{where}: an active level-shift warm-up requires max_iter >= 2 "
            "to leave an unshifted tail cycle"
        )
    resolved_cycles = min(requested_cycles, max_iter - 1)
    schedule = LevelShiftSchedule([level_shift] * resolved_cycles + [0.0])
    return schedule, resolved_cycles


def _require_inactive_quadratic_fallback(
    options: PeriodicRHFOptions | PeriodicKSOptions,
    *,
    where: str,
) -> None:
    """Reject a quadratic-SCF request that no current χ engine executes."""

    raw_iteration = getattr(options, "quadratic_fallback_iter", 0)
    if isinstance(raw_iteration, (bool, np.bool_)):
        raise ValueError(
            f"{where}: quadratic_fallback_iter must be a non-negative integer"
        )
    try:
        numeric_iteration = float(raw_iteration)
        iteration = int(raw_iteration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{where}: quadratic_fallback_iter must be a non-negative integer"
        ) from exc
    if (
        not np.isfinite(numeric_iteration)
        or numeric_iteration != float(iteration)
        or iteration < 0
    ):
        raise ValueError(
            f"{where}: quadratic_fallback_iter must be a non-negative integer; "
            f"got {raw_iteration!r}"
        )
    if iteration > 0:
        raise NotImplementedError(
            f"{where}: χ-CCM-B quadratic SCF fallback is not implemented. "
            "The four_center and fitted B engines do not execute "
            "quadratic_fallback_iter, quadratic_fallback_shift, or "
            "quadratic_fallback_max_step. Use quadratic_fallback_iter=0 "
            "until one verified update contract is implemented across the "
            "selected backend."
        )


def _resolve_symmetry_mode(
    mode: str | AICCM2026DevBSymmetryMode,
) -> AICCM2026DevBSymmetryMode:
    if isinstance(mode, AICCM2026DevBSymmetryMode):
        return mode
    try:
        return AICCM2026DevBSymmetryMode(str(mode).strip().lower())
    except ValueError as exc:
        choices = ", ".join(member.value for member in AICCM2026DevBSymmetryMode)
        raise ValueError(
            f"unknown aiccm2026dev-b symmetry mode {mode!r}; choose {choices}"
        ) from exc


def _build_symmetry_plan(
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
    mode: str | AICCM2026DevBSymmetryMode,
    *,
    symmetry_precision: float,
    symmetry_require_full_group: bool,
) -> AICCM2026DevBSymmetryPlan | None:
    selected = _resolve_symmetry_mode(mode)
    if selected is AICCM2026DevBSymmetryMode.OFF:
        return None
    if selected is AICCM2026DevBSymmetryMode.INTEGRALS:
        raise NotImplementedError(
            "aiccm2026dev-b symmetry integral acceleration is disabled: "
            "general-k AO sewing phases and libint shell-quartet petite-list "
            "scattering have not passed the symmetry-off parity gate"
        )
    return build_aiccm2026dev_b_symmetry_plan(
        system,
        mesh,
        symprec=symmetry_precision,
        require_full_space_group=symmetry_require_full_group,
    )


def _attach_symmetry_diagnostics(
    result: object,
    system: PeriodicSystem,
    basis: BasisSet,
    symmetry_plan: AICCM2026DevBSymmetryPlan | None,
    density_k: Sequence[np.ndarray],
) -> None:
    if symmetry_plan is None:
        return
    fock_residual: float | None = None
    density_residual: float | None = None
    if symmetry_plan.n_kpoints_full == 1:
        fock = getattr(result, "fock", None)
        if isinstance(fock, (list, tuple)):
            fock = fock[0]
        if fock is not None:
            fock_residual = gamma_matrix_symmetry_residual(
                np.asarray(fock), system, basis, symmetry_plan
            )
        density_residual = gamma_matrix_symmetry_residual(
            np.asarray(density_k[0]), system, basis, symmetry_plan
        )
    pair_orbits = shell_pair_orbits(basis, symmetry_plan)
    n_shells = len(list(basis.shells()))
    quartet_orbits = (
        shell_quartet_orbits(basis, symmetry_plan)
        if n_shells <= 24
        else None
    )
    setattr(
        result,
        "aiccm2026dev_b_symmetry",
        AICCM2026DevBSymmetryDiagnostics(
            plan=symmetry_plan,
            gamma_fock_residual=fock_residual,
            gamma_density_residual=density_residual,
            n_shell_pairs=sum(len(orbit) for orbit in pair_orbits),
            n_unique_shell_pairs=len(pair_orbits),
            n_shell_quartets=(
                None
                if quartet_orbits is None
                else sum(len(orbit) for orbit in quartet_orbits)
            ),
            n_unique_shell_quartets=(
                None if quartet_orbits is None else len(quartet_orbits)
            ),
        ),
    )


def _option_sequence(options: object | None, name: str) -> tuple[object, ...]:
    if options is None or not hasattr(options, name):
        return ()
    value = getattr(options, name)
    if value is None:
        return ()
    try:
        return tuple(value)
    except TypeError:
        return ()


def aiccm2026dev_b_charge_bookkeeping(
    system: PeriodicSystem,
    options: object | None = None,
) -> AICCM2026DevBChargeBookkeeping:
    """Return the effective charge/electron convention for a χ-CCM run."""

    atoms = list(system.unit_cell)
    physical_electrons = int(system.n_electrons())
    bare_charges = tuple(float(atom.Z) for atom in atoms)
    ecp_total_ncore = int(getattr(options, "ecp_total_ncore", 0) or 0)
    effective_charges_raw = _option_sequence(options, "ecp_effective_charges")
    ecp_blocks = _option_sequence(options, "ecp_primitive_blocks")
    ecp_centers = _option_sequence(options, "ecp_home_centers")
    has_ecp = bool(effective_charges_raw or ecp_blocks or ecp_centers or ecp_total_ncore)
    if ecp_total_ncore < 0:
        raise ValueError("aiccm2026dev-b ECP core electron count must be non-negative")
    if ecp_total_ncore > physical_electrons:
        raise ValueError(
            "aiccm2026dev-b ECP core electron count exceeds the physical "
            f"electron count ({ecp_total_ncore} > {physical_electrons})"
        )
    if has_ecp:
        if len(effective_charges_raw) != len(atoms):
            raise ValueError(
                "aiccm2026dev-b ECP runs require one effective nuclear charge "
                "per primitive-cell atom"
            )
        effective_charges = tuple(float(charge) for charge in effective_charges_raw)
        expected_charge = float(sum(bare_charges) - ecp_total_ncore)
        observed_charge = float(sum(effective_charges))
        if abs(observed_charge - expected_charge) > 1.0e-8:
            raise ValueError(
                "aiccm2026dev-b ECP effective nuclear charges are inconsistent "
                "with ecp_total_ncore: "
                f"sum(Z_eff)={observed_charge:.12g}, expected {expected_charge:.12g}"
            )
    else:
        effective_charges = bare_charges
        observed_charge = float(sum(effective_charges))
    effective_electrons = physical_electrons - ecp_total_ncore
    net_charge = observed_charge - float(effective_electrons)
    return AICCM2026DevBChargeBookkeeping(
        physical_electrons=physical_electrons,
        effective_electrons=effective_electrons,
        ecp_total_ncore=ecp_total_ncore,
        effective_nuclear_charges=effective_charges,
        effective_nuclear_charge=observed_charge,
        effective_net_charge=net_charge,
        has_ecp=has_ecp,
    )


def _validate_closed_shell_problem(system: PeriodicSystem, options: object) -> None:
    if int(system.dim) not in (1, 2, 3):
        raise ValueError(
            "aiccm2026dev-b requires a 1D, 2D, or 3D periodic system; "
            f"got dim={system.dim}"
        )
    bookkeeping = aiccm2026dev_b_charge_bookkeeping(system, options)
    if abs(bookkeeping.effective_net_charge) > 1.0e-8:
        raise ValueError(
            "aiccm2026dev-b requires a neutral effective primitive cell; "
            f"effective net charge is {bookkeeping.effective_net_charge:.12g}. "
            "Charged cells need an explicitly selected background convention"
        )
    if int(system.multiplicity) != 1:
        raise ValueError("aiccm2026dev-b implements closed-shell RHF/RKS only")
    if bookkeeping.effective_electrons % 2:
        raise ValueError(
            "aiccm2026dev-b requires an even effective electron count; "
            f"got {bookkeeping.effective_electrons}"
        )
    if float(getattr(options, "smearing_temperature", 0.0) or 0.0) != 0.0:
        raise ValueError(
            "aiccm2026dev-b is a zero-temperature idempotent variational "
            "problem; smearing is not implemented"
        )


def _validate_open_shell_problem(system: PeriodicSystem, options: object) -> None:
    if int(system.dim) not in (1, 2, 3):
        raise ValueError(
            "aiccm2026dev-b requires a 1D, 2D, or 3D periodic system; "
            f"got dim={system.dim}"
        )
    bookkeeping = aiccm2026dev_b_charge_bookkeeping(system, options)
    if abs(bookkeeping.effective_net_charge) > 1.0e-8:
        raise ValueError(
            "aiccm2026dev-b requires a neutral effective primitive cell; "
            f"effective net charge is {bookkeeping.effective_net_charge:.12g}. "
            "Charged cells need an explicitly selected background convention"
        )
    multiplicity = int(system.multiplicity)
    n_electrons = bookkeeping.effective_electrons
    if (
        multiplicity < 1
        or multiplicity > n_electrons + 1
        or (n_electrons + multiplicity - 1) % 2
    ):
        raise ValueError(
            "aiccm2026dev-b electron count and multiplicity do not define "
            "integer alpha/beta occupations"
        )
    if float(getattr(options, "smearing_temperature", 0.0) or 0.0) != 0.0:
        raise ValueError(
            "aiccm2026dev-b UHF/UKS currently minimizes idempotent zero-"
            "temperature spin projectors; ensemble smearing is not enabled"
        )


def _density_blocks_per_k(
    result: object,
    n_electrons: int,
) -> list[np.ndarray]:
    density = getattr(result, "density")
    if isinstance(density, (list, tuple)):
        return [np.asarray(block) for block in density]
    coefficients = list(getattr(result, "mo_coeffs"))
    occupations = getattr(result, "occupations", None)
    if isinstance(occupations, (list, tuple)) and len(occupations) == len(coefficients):
        return [
            (np.asarray(coeff) * np.asarray(occ)[None, :]) @ np.asarray(coeff).conj().T
            for coeff, occ in zip(coefficients, occupations)
        ]
    n_occ = n_electrons // 2
    return [
        2.0 * np.asarray(coeff)[:, :n_occ] @ np.asarray(coeff)[:, :n_occ].conj().T
        for coeff in coefficients
    ]


def _spin_density_blocks_per_k(
    result: object,
    spin: str,
    n_occupied: int,
) -> list[np.ndarray]:
    density = getattr(result, f"density_{spin}")
    if isinstance(density, (list, tuple)):
        return [np.asarray(block) for block in density]
    coefficients = list(getattr(result, f"mo_coeffs_{spin}"))
    occupations = getattr(result, f"occupations_{spin}", None)
    if isinstance(occupations, (list, tuple)) and len(occupations) == len(
        coefficients
    ):
        return [
            (np.asarray(coeff) * np.asarray(occ)[None, :])
            @ np.asarray(coeff).conj().T
            for coeff, occ in zip(coefficients, occupations)
        ]
    return [
        np.asarray(coeff)[:, :n_occupied]
        @ np.asarray(coeff)[:, :n_occupied].conj().T
        for coeff in coefficients
    ]


def _trace_field(item: object, name: str, index: int) -> object | None:
    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, (tuple, list)) and len(item) > index:
        return item[index]
    return None


def _final_scf_trace_values(
    result: object,
) -> tuple[int, float | None, float | None, int | None]:
    trace = list(getattr(result, "scf_trace", []) or [])
    if not trace:
        return 0, None, None, None
    last = trace[-1]
    delta_e = _trace_field(last, "delta_e", 2)
    grad_norm = _trace_field(last, "grad_norm", 3)
    diis_subspace = _trace_field(last, "diis_subspace", 4)
    return (
        len(trace),
        None if delta_e is None else float(delta_e),
        None if grad_norm is None else float(grad_norm),
        None if diis_subspace is None else int(diis_subspace),
    )


def _scf_accelerator_name(options: object | None) -> str | None:
    if options is None or not hasattr(options, "scf_accelerator"):
        return None
    accelerator = getattr(options, "scf_accelerator")
    name = getattr(accelerator, "name", None)
    if isinstance(name, str):
        return name
    return str(accelerator).split(".")[-1]


def _executed_fock_mixing(
    result: object,
    options: object | None,
) -> float | None:
    """Return the backend-resolved previous-Fock weight when available."""

    executed = getattr(result, "fock_mixing", None)
    if executed is not None:
        return float(executed)
    if options is None:
        return None
    return float(getattr(options, "fock_mixing", 0.0))


def _executed_level_shift_warmup_cycles(
    result: object,
    options: object | None,
    direct_override: int | None,
) -> int | None:
    """Return the effective warm-up length after selector resolution."""

    if direct_override is not None:
        return int(direct_override)
    if options is None:
        return None
    raw_schedule = getattr(options, "level_shift_schedule", ())
    if raw_schedule is not None and len(raw_schedule) > 0:
        # An explicit schedule supersedes the warm-up modifier.
        return 0
    if float(getattr(options, "level_shift", 0.0)) == 0.0:
        return 0
    backend_value = getattr(result, "level_shift_warmup_cycles", None)
    if backend_value is not None:
        return int(backend_value)
    return int(getattr(options, "level_shift_warmup_cycles", 0))


def _attach_diagnostics(
    result: object,
    system: PeriodicSystem,
    basis: BasisSet,
    kpoints: KPoints,
    mesh: tuple[int, int, int],
    backend: AICCM2026DevBBackend,
    electronic_method: str,
    exact_exchange_coefficients: PeriodicExchangeAssembly,
    elapsed: float,
    symmetry_plan: AICCM2026DevBSymmetryPlan | None = None,
    options: object | None = None,
    executed_level_shift_warmup_cycles: int | None = None,
) -> object:
    runtime_backend = getattr(result, "runtime_backend", None)
    if runtime_backend is None:
        runtime_backend = getattr(result, "backend", None)
    if not isinstance(runtime_backend, str) or not runtime_backend.strip():
        raise RuntimeError(
            "aiccm2026dev-b backend did not expose its executed low-level "
            "backend label"
        )
    if backend is AICCM2026DevBBackend.FOUR_CENTER:
        if runtime_backend != _DIRECT_RUNTIME_BACKEND:
            raise RuntimeError(
                "aiccm2026dev-b four_center selected an inconsistent runtime "
                f"backend {runtime_backend!r}"
            )
    else:
        runtime_backend_base = runtime_backend.removesuffix("+PARITY_HELD")
        method_name = electronic_method.partition("/")[0].lower()
        if backend is AICCM2026DevBBackend.RI:
            allowed_runtime_backends = {
                "native-gamma-gdf-via-k-gdf",
                f"native-multi-k-gdf-gdf-{method_name}",
            }
        else:
            allowed_runtime_backends = {
                f"native-multi-k-gdf-cosx-{method_name}",
            }
        if runtime_backend_base not in allowed_runtime_backends:
            raise RuntimeError(
                f"aiccm2026dev-b {backend.value} selected an inconsistent "
                f"runtime backend {runtime_backend!r} for {electronic_method}"
            )
    exact_exchange_assembly = _finalize_exact_exchange_assembly(
        result,
        backend,
        exact_exchange_coefficients,
        where=f"aiccm2026dev-b {electronic_method}",
    )
    setattr(result, "runtime_backend", runtime_backend)
    # IID 344: structured form of the ``+PARITY_HELD`` marker, derived from
    # the runtime backend string (the hold mechanism's canonical carrier) so
    # the flag can never drift from it. Consumers separate held rows on this
    # field instead of substring-matching the backend or grepping .err.
    setattr(result, "parity_held", "+PARITY_HELD" in runtime_backend)
    if backend is AICCM2026DevBBackend.FOUR_CENTER:
        sr_image_extent = getattr(result, "sr_image_extent_bohr", None)
        if sr_image_extent is not None:
            # The physical route measures interaction reach between pair
            # midpoints, not the legacy absolute ket-image radius. Consumers
            # qualified for that legacy finite operator must not inherit an
            # attestation for the different physical quartet support (#243).
            lattice_options = getattr(options, "lattice_opts", None)
            physical_pairs = bool(getattr(lattice_options, "pair_complete_1e", False))
            domain_policy = (
                _M5_SR_PHYSICAL_DOMAIN_POLICY
                if physical_pairs else _M5_SR_IMAGE_DOMAIN_POLICY
            )
            setattr(result, "sr_image_domain_policy", domain_policy)
            setattr(result, "sr_image_precision", _M5_SR_IMAGE_PRECISION)

    representatives = wigner_seitz_representatives(system, mesh)
    residue_sums: dict[tuple[int, int, int], float] = {}
    for representative in representatives:
        residue_sums[representative.residue] = (
            residue_sums.get(representative.residue, 0.0) + representative.weight
        )
    partition_error = max(abs(total - 1.0) for total in residue_sums.values())
    overlap_k = [np.asarray(block) for block in getattr(result, "overlap")]
    charge_bookkeeping = aiccm2026dev_b_charge_bookkeeping(system, options)
    effective_electrons = charge_bookkeeping.effective_electrons
    is_unrestricted = hasattr(result, "density_alpha")
    n_alpha_error = None
    n_beta_error = None
    if is_unrestricted:
        two_s = int(system.multiplicity) - 1
        n_alpha = (effective_electrons + two_s) // 2
        n_beta = (effective_electrons - two_s) // 2
        density_alpha_k = _spin_density_blocks_per_k(result, "alpha", n_alpha)
        density_beta_k = _spin_density_blocks_per_k(result, "beta", n_beta)
        density_k = [
            alpha + beta
            for alpha, beta in zip(density_alpha_k, density_beta_k)
        ]
        idempotency_error = uhf_idempotency_error(
            density_alpha_k, density_beta_k, overlap_k
        )
        n_alpha_error = _electron_count_error(
            density_alpha_k,
            overlap_k,
            np.asarray(kpoints.weights, dtype=float),
            n_alpha,
        )
        n_beta_error = _electron_count_error(
            density_beta_k,
            overlap_k,
            np.asarray(kpoints.weights, dtype=float),
            n_beta,
        )
        alpha_blocks, alpha_residue_execution = (
            _distributed_residue_inverse_bloch_transform(
                density_alpha_k,
                kpoints.kpoints_frac,
                representatives,
                kpoints.weights,
                mesh=mesh,
                representative_scope="zero-offset-cell",
            )
        )
        beta_blocks, beta_residue_execution = (
            _distributed_residue_inverse_bloch_transform(
                density_beta_k,
                kpoints.kpoints_frac,
                representatives,
                kpoints.weights,
                mesh=mesh,
                representative_scope="zero-offset-cell",
            )
        )
        if alpha_residue_execution != beta_residue_execution:
            raise RuntimeError(
                "aiccm2026dev-b alpha/beta residue-transform schedules differ"
            )
        transformed = [alpha_blocks, beta_blocks]
        residue_transform_execution = alpha_residue_execution
        imaginary_residual = max(
            float(np.max(np.abs(blocks.imag))) for blocks in transformed
        )
        variational_space = "translation-invariant unrestricted determinants"
    else:
        density_k = _density_blocks_per_k(result, effective_electrons)
        idempotency_error = rhf_idempotency_error(density_k, overlap_k)
        density_blocks, residue_transform_execution = (
            _distributed_residue_inverse_bloch_transform(
                density_k,
                kpoints.kpoints_frac,
                representatives,
                kpoints.weights,
                mesh=mesh,
                representative_scope="zero-offset-cell",
            )
        )
        imaginary_residual = float(np.max(np.abs(density_blocks.imag)))
        variational_space = "translation-invariant closed-shell determinants"
    extension = cyclic_lattice_extension(system, mesh)
    convention = _finite_torus_convention(
        system,
        mesh,
        extension,
        full_range_exchange_coefficient=exact_exchange_assembly.c_full,
    )
    (
        scf_trace_length,
        final_scf_delta_e,
        final_scf_grad_norm,
        final_scf_diis_subspace,
    ) = _final_scf_trace_values(result)
    diagnostics = AICCM2026DevBDiagnostics(
        mesh=mesh,
        n_cyclic_cells=int(np.prod(mesh)),
        n_kpoints=len(kpoints.weights),
        wigner_seitz_partition_error=partition_error,
        density_idempotency_error=idempotency_error,
        electron_count_error=_electron_count_error(
            density_k,
            overlap_k,
            np.asarray(kpoints.weights, dtype=float),
            effective_electrons,
        ),
        inverse_bloch_imaginary_residual=imaginary_residual,
        backend=backend.value,
        electronic_method=electronic_method,
        wall_time_seconds=elapsed,
        exact_exchange_assembly=exact_exchange_assembly,
        lattice_extension=extension.repetitions,
        wigner_seitz_half_extent=extension.wigner_seitz_half_extent,
        n_alpha_error=n_alpha_error,
        n_beta_error=n_beta_error,
        s_squared=(
            float(getattr(result, "s_squared"))
            if hasattr(result, "s_squared")
            else None
        ),
        s_squared_ideal=(
            float(getattr(result, "s_squared_ideal"))
            if hasattr(result, "s_squared_ideal")
            else None
        ),
        scf_trace_length=scf_trace_length,
        final_scf_delta_e_ha=final_scf_delta_e,
        final_scf_grad_norm=final_scf_grad_norm,
        final_scf_diis_subspace=final_scf_diis_subspace,
        physical_electron_count=charge_bookkeeping.physical_electrons,
        effective_electron_count=effective_electrons,
        ecp_total_ncore=charge_bookkeeping.ecp_total_ncore,
        effective_nuclear_charge=charge_bookkeeping.effective_nuclear_charge,
        effective_nuclear_charges=charge_bookkeeping.effective_nuclear_charges,
        effective_net_charge=charge_bookkeeping.effective_net_charge,
        use_diis=(
            None if options is None else bool(getattr(options, "use_diis", False))
        ),
        diis_start_iter=(
            None if options is None else int(getattr(options, "diis_start_iter", 0))
        ),
        diis_subspace_size=(
            None if options is None else int(getattr(options, "diis_subspace_size", 0))
        ),
        scf_accelerator=_scf_accelerator_name(options),
        damping=(
            None if options is None else float(getattr(options, "damping", 0.0))
        ),
        dynamic_damping=(
            None
            if options is None
            else bool(getattr(options, "dynamic_damping", False))
        ),
        fock_mixing=_executed_fock_mixing(result, options),
        level_shift=(
            None if options is None else float(getattr(options, "level_shift", 0.0))
        ),
        level_shift_warmup_cycles=(
            _executed_level_shift_warmup_cycles(
                result,
                options,
                executed_level_shift_warmup_cycles,
            )
        ),
        smearing_temperature=(
            None
            if options is None
            else float(getattr(options, "smearing_temperature", 0.0))
        ),
        variational_space=variational_space,
        finite_torus_convention=convention,
        direct_output_cell_farming=getattr(
            result, "output_cell_farming_execution", None
        ),
        residue_inverse_transform=residue_transform_execution,
    )
    setattr(result, "backend", f"aiccm2026dev-b-{backend.value}")
    setattr(result, "aiccm2026dev_b", diagnostics)
    setattr(
        result,
        "residue_inverse_transform_execution",
        residue_transform_execution,
    )
    setattr(result, "exact_exchange_assembly", exact_exchange_assembly)
    setattr(result, "finite_torus_convention", convention)
    setattr(result, "ccm_approach", convention.ccm_approach)
    setattr(result, "ccm_construction", convention.ccm_construction)
    setattr(
        result,
        "evaluation_representation",
        convention.evaluation_representation,
    )
    setattr(result, "coulomb_kernel", convention.coulomb_kernel)
    setattr(result, "exchange_q0", convention.exchange_q0)
    setattr(
        result,
        "exchange_q0_applicability",
        convention.exchange_q0_applicability,
    )
    setattr(result, "boundary_model", convention.boundary_model)
    setattr(result, "effective_n_electrons", effective_electrons)
    setattr(result, "effective_nuclear_charge", charge_bookkeeping.effective_nuclear_charge)
    setattr(result, "effective_nuclear_charges", charge_bookkeeping.effective_nuclear_charges)
    setattr(result, "ecp_total_ncore", charge_bookkeeping.ecp_total_ncore)
    setattr(result, "kpoints_frac", np.asarray(kpoints.kpoints_frac, dtype=float))
    setattr(result, "kpoints_cart", np.asarray(kpoints.kpoints_cart, dtype=float))
    setattr(result, "kpoint_weights", np.asarray(kpoints.weights, dtype=float))
    _attach_symmetry_diagnostics(
        result,
        system,
        basis,
        symmetry_plan,
        density_k,
    )
    return result


def _attach_fitted_selector_settings(
    result: object,
    selected: AICCM2026DevBBackend,
    *,
    gdf_method: str,
    rsgdf_ke_cutoff: float,
    rsgdf_tail_ke_cutoff: float | None,
    mdf_ke_cutoff: float,
) -> None:
    """Record fitted settings forwarded by the B selector, not telemetry."""

    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        return
    setattr(result, "aiccm_resolved_gdf_method", str(gdf_method))
    setattr(result, "aiccm_resolved_rsgdf_ke_cutoff", float(rsgdf_ke_cutoff))
    setattr(
        result,
        "aiccm_resolved_rsgdf_tail_ke_cutoff",
        (
            None
            if rsgdf_tail_ke_cutoff is None
            else float(rsgdf_tail_ke_cutoff)
        ),
    )
    setattr(result, "aiccm_resolved_mdf_ke_cutoff", float(mdf_ke_cutoff))


def _resolve_fitted_rsgdf_tail(
    selected: AICCM2026DevBBackend,
    *,
    gdf_method: str,
    rsgdf_ke_cutoff: float,
    rsgdf_tail_ke_cutoff: float | None,
    where: str,
) -> float | None:
    """Validate the D106 diagnostic-only high-|G| tail transport.

    This proves only that the selector forwards an effective complementary
    reciprocal shell to the shared RSGDF driver.  It deliberately does not
    qualify the route under D93's numerical-support contract.
    """

    if rsgdf_tail_ke_cutoff is None:
        return None
    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        raise NotImplementedError(
            f"{where}: rsgdf_tail_ke_cutoff applies only to the fitted "
            "RI and RIJCOSX backends; backend='four_center' has no RSGDF fit."
        )
    if str(gdf_method) != "rsgdf":
        raise NotImplementedError(
            f"{where}: rsgdf_tail_ke_cutoff requires gdf_method='rsgdf'; "
            f"got {gdf_method!r}. MDF has no high-|G| RSGDF tail."
        )
    if isinstance(rsgdf_tail_ke_cutoff, (bool, np.bool_)) or isinstance(
        rsgdf_ke_cutoff,
        (bool, np.bool_),
    ):
        raise ValueError(
            f"{where}: rsgdf_ke_cutoff must be positive and finite, and "
            "rsgdf_tail_ke_cutoff must be finite and strictly greater; "
            "booleans are invalid."
        )
    try:
        tail = float(rsgdf_tail_ke_cutoff)
        base = float(rsgdf_ke_cutoff)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{where}: rsgdf_ke_cutoff must be positive and finite, and "
            "rsgdf_tail_ke_cutoff must be finite and strictly greater."
        ) from exc
    if (
        not np.isfinite(tail)
        or not np.isfinite(base)
        or base <= 0.0
        or tail <= base
    ):
        raise ValueError(
            f"{where}: rsgdf_ke_cutoff must be positive and finite, and "
            "rsgdf_tail_ke_cutoff must be finite and strictly greater; "
            f"got base={base!r} Ha and tail={tail!r} Ha."
        )
    return tail


def _reject_implicit_gamma_rsgdf_tail(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: Sequence[int],
    selected: AICCM2026DevBBackend,
    *,
    gdf_method: str,
    rsgdf_ke_cutoff: float,
    rsgdf_tail_ke_cutoff: float | None,
    fock_mixing: float,
    where: str,
) -> None:
    """Keep a null B tail from activating the generic Gamma auto default.

    The shared Gamma-only RHF driver can resolve ``None`` to an extended
    dense-core production tail. D93 has no executed-tail provenance for that
    implicit extension, so B requires it to be explicit before entering SCF.
    A resolver value at the base cutoff builds no complementary shell and is
    inactive. True multi-k paths and Gamma systems that need no tail are
    unchanged.
    """

    if (
        selected is not AICCM2026DevBBackend.RI
        or int(np.prod(tuple(int(value) for value in lattice_extension))) != 1
        or str(gdf_method) != "rsgdf"
        or rsgdf_tail_ke_cutoff is not None
        or float(fock_mixing) != 0.0
    ):
        return
    auto_tail = _auto_rsgdf_tail_ke_cutoff(
        system,
        str(gdf_method),
        basis,
        None,
        rsgdf_ke_cutoff=float(rsgdf_ke_cutoff),
    )
    if auto_tail is None or float(auto_tail) <= float(rsgdf_ke_cutoff):
        return
    raise NotImplementedError(
        f"{where}: the shared Gamma RSGDF production default would "
        "auto-resolve rsgdf_tail_ke_cutoff=None to "
        f"{float(auto_tail):.12g} Ha for this basis. χ-CCM-B does not "
        "inherit an implicit tail without executed B support provenance; "
        f"pass rsgdf_tail_ke_cutoff={float(auto_tail):.12g} explicitly for "
        "D106 diagnostic transport, or use at least two cyclic cells. The "
        "fitted route remains D93 not-qualified."
    )


def run_aiccm2026dev_b_rhf(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    mesh: int | Sequence[int] | None = None,
    wigner_seitz_shells: int | Sequence[int] | None = None,
    backend: str | AICCM2026DevBBackend = AICCM2026DevBBackend.FOUR_CENTER,
    aux_basis: str | None = None,
    gdf_method: str = "rsgdf",
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: float | None = None,
    mdf_ke_cutoff: float = 40.0,
    fock_mixing: float | None = None,
    symmetry_mode: str | AICCM2026DevBSymmetryMode = (
        AICCM2026DevBSymmetryMode.OFF
    ),
    symmetry_precision: float = 1.0e-5,
    symmetry_require_full_group: bool = False,
    progress: bool | ProgressLogger | None = None,
    verbose: int | None = None,
) -> PeriodicKRHFGDFResult | PBCBipoleRHFResult:
    """Minimise the finite-torus closed-shell RHF energy per cell."""

    _warn_experimental()
    mesh_tuple = _resolve_lattice_extension(
        system,
        lattice_extension,
        mesh=mesh,
        wigner_seitz_shells=wigner_seitz_shells,
    )
    selected = _resolve_backend(backend)
    opts = options if options is not None else PeriodicRHFOptions()
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_aiccm2026dev_b_rhf",
    )
    _validate_closed_shell_problem(system, opts)
    _require_supported_scf_dimension(system)
    direct_level_shift_schedule = None
    executed_level_shift_warmup_cycles = None
    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        _require_zero_restricted_four_center_level_shift_schedule(
            opts,
            where="run_aiccm2026dev_b_rhf",
        )
        (
            direct_level_shift_schedule,
            executed_level_shift_warmup_cycles,
        ) = _resolve_restricted_four_center_level_shift_warmup(
            opts,
            where="run_aiccm2026dev_b_rhf",
        )
    if (
        selected is not AICCM2026DevBBackend.FOUR_CENTER
        and gdf_method not in ("rsgdf", "mdf")
    ):
        raise ValueError(
            "aiccm2026dev-b requires gdf_method='rsgdf' or 'mdf'; the q-only "
            "compcell fit is not a consistent finite-torus Hamiltonian"
        )
    resolved_rsgdf_tail = _resolve_fitted_rsgdf_tail(
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        where="run_aiccm2026dev_b_rhf",
    )
    kpoints = cyclic_gamma_mesh(system, mesh_tuple)
    exact_exchange_coefficients = _resolve_exact_exchange_coefficients(
        None,
        spin=1,
        where="run_aiccm2026dev_b_rhf",
    )
    symmetry_plan = _build_symmetry_plan(
        system,
        mesh_tuple,
        symmetry_mode,
        symmetry_precision=symmetry_precision,
        symmetry_require_full_group=symmetry_require_full_group,
    )
    if selected is AICCM2026DevBBackend.RIJCOSX and int(np.prod(mesh_tuple)) == 1:
        raise NotImplementedError(
            "aiccm2026dev-b RIJCOSX requires at least two cyclic cells; the "
            "native multi-k COSX bridge has no Gamma-only implementation"
        )
    if (
        selected is AICCM2026DevBBackend.RI
        and int(system.dim) == 3
        and int(np.prod(mesh_tuple)) == 1
        and requested_fock_mixing != 0.0
    ):
        raise NotImplementedError(
            "aiccm2026dev-b Gamma-only RI RHF cannot execute previous-Fock "
            "mixing without switching to the legacy cutoff-selected Gamma "
            "GDF fallback; use fock_mixing=0 or at least two cyclic cells"
        )
    if (
        selected is AICCM2026DevBBackend.RI
        and int(system.dim) == 3
        and int(np.prod(mesh_tuple)) == 1
    ):
        _require_zero_gamma_ri_rhf_level_shift(
            opts,
            where="run_aiccm2026dev_b_rhf",
        )
    _reject_implicit_gamma_rsgdf_tail(
        system,
        basis,
        mesh_tuple,
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
        fock_mixing=requested_fock_mixing,
        where="run_aiccm2026dev_b_rhf",
    )
    _require_inactive_quadratic_fallback(
        opts,
        where="run_aiccm2026dev_b_rhf",
    )
    _reject_four_center_aux_basis(
        selected,
        aux_basis,
        where="run_aiccm2026dev_b_rhf",
    )

    started = perf_counter()
    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        use_3d_ewald = int(system.dim) == 3
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kpoints.to_bloch_kmesh(),
            opts,
            level_shift_schedule=direct_level_shift_schedule,
            fock_mixing=requested_fock_mixing,
            use_ewald_j_split=use_3d_ewald,
            use_exchange_ewald_split=use_3d_ewald,
            exchange_exxdiv=("ewald" if use_3d_ewald else "none"),
            use_multipole_far_field=False,
            sr_image_precision=_M5_SR_IMAGE_PRECISION,
            farm_output_cells=True,
            output_cell_farming_strategy="cyclic",
            output_cell_farming_task_kind="chi-direct-output-cell",
            progress=progress,
            verbose=verbose,
        )
    else:
        result = run_krhf_periodic_gdf(
            system,
            basis,
            kpoints,
            opts,
            functional=None,
            aux_basis=aux_basis,
            fock_mixing=requested_fock_mixing,
            use_compcell=selected is AICCM2026DevBBackend.RIJCOSX,
            k_exchange=(
                "cosx" if selected is AICCM2026DevBBackend.RIJCOSX else "gdf"
            ),
            gdf_method=gdf_method,
            rsgdf_ke_cutoff=rsgdf_ke_cutoff,
            rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
            mdf_ke_cutoff=mdf_ke_cutoff,
            ibz_native=False,
            compute_gradient=False,
            check_energy_sanity=True,
            progress=progress,
            verbose=verbose,
        )
    result = _attach_diagnostics(
        result,
        system,
        basis,
        kpoints,
        mesh_tuple,
        selected,
        "RHF",
        exact_exchange_coefficients,
        perf_counter() - started,
        symmetry_plan,
        opts,
        executed_level_shift_warmup_cycles,
    )
    _attach_fitted_selector_settings(
        result,
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
        mdf_ke_cutoff=mdf_ke_cutoff,
    )
    return result


def run_aiccm2026dev_b_rks(
    system: PeriodicSystem,
    basis: BasisSet,
    functional: str,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicKSOptions | None = None,
    *,
    mesh: int | Sequence[int] | None = None,
    wigner_seitz_shells: int | Sequence[int] | None = None,
    backend: str | AICCM2026DevBBackend = AICCM2026DevBBackend.FOUR_CENTER,
    aux_basis: str | None = None,
    gdf_method: str = "rsgdf",
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: float | None = None,
    mdf_ke_cutoff: float = 40.0,
    fock_mixing: float | None = None,
    symmetry_mode: str | AICCM2026DevBSymmetryMode = (
        AICCM2026DevBSymmetryMode.OFF
    ),
    symmetry_precision: float = 1.0e-5,
    symmetry_require_full_group: bool = False,
    progress: bool | ProgressLogger | None = None,
    verbose: int | None = None,
) -> PeriodicKRKSGDFResult | PBCBipoleRKSResult:
    """Minimise the finite-torus closed-shell Kohn--Sham energy per cell."""

    _warn_experimental()
    if not str(functional).strip():
        raise ValueError("aiccm2026dev-b RKS requires a functional name")
    mesh_tuple = _resolve_lattice_extension(
        system,
        lattice_extension,
        mesh=mesh,
        wigner_seitz_shells=wigner_seitz_shells,
    )
    selected = _resolve_backend(backend)
    opts = options if options is not None else PeriodicKSOptions()
    opts.functional = str(functional)
    func = Functional(str(functional), 1)
    uses_external_xc = _require_supported_external_xc(
        func,
        selected,
        system=system,
        where="run_aiccm2026dev_b_rks",
    )
    if uses_external_xc:
        from ...pbc_bipole_common import reject_bipole_ecp_options

        reject_bipole_ecp_options(
            opts,
            driver="run_aiccm2026dev_b_rks",
            basis=basis,
            system=system,
        )
    _configure_external_xc_grid(
        func,
        opts,
        caller_supplied_options=options is not None,
        where="run_aiccm2026dev_b_rks",
    )
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_aiccm2026dev_b_rks",
    )
    _validate_closed_shell_problem(system, opts)
    _require_supported_scf_dimension(system)
    direct_level_shift_schedule = None
    executed_level_shift_warmup_cycles = None
    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        _require_zero_restricted_four_center_level_shift_schedule(
            opts,
            where="run_aiccm2026dev_b_rks",
        )
        (
            direct_level_shift_schedule,
            executed_level_shift_warmup_cycles,
        ) = _resolve_restricted_four_center_level_shift_warmup(
            opts,
            where="run_aiccm2026dev_b_rks",
        )
    if selected is not AICCM2026DevBBackend.FOUR_CENTER:
        try:
            reject_unscreened_range_separated(
                Functional(str(functional), 1),
                where=(
                    "run_aiccm2026dev_b_rks "
                    f"backend={selected.value!r}"
                ),
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "aiccm2026dev-b RI and RIJCOSX backends remain "
                "fail-closed for range-separated functionals. Their "
                "χ-specific operator, validation, and provenance contract "
                "covers full-range exchange only; shared generic COSX "
                "screened-exchange support does not widen that contract. "
                "The separately gated experimental 3D HSE path uses "
                "backend='four_center'."
            ) from exc
    if (
        selected is not AICCM2026DevBBackend.FOUR_CENTER
        and gdf_method not in ("rsgdf", "mdf")
    ):
        raise ValueError(
            "aiccm2026dev-b requires gdf_method='rsgdf' or 'mdf'; the q-only "
            "compcell fit is not a consistent finite-torus Hamiltonian"
        )
    resolved_rsgdf_tail = _resolve_fitted_rsgdf_tail(
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        where="run_aiccm2026dev_b_rks",
    )
    kpoints = cyclic_gamma_mesh(system, mesh_tuple)
    exact_exchange_coefficients = _resolve_exact_exchange_coefficients(
        str(functional),
        spin=1,
        where="run_aiccm2026dev_b_rks",
    )
    symmetry_plan = _build_symmetry_plan(
        system,
        mesh_tuple,
        symmetry_mode,
        symmetry_precision=symmetry_precision,
        symmetry_require_full_group=symmetry_require_full_group,
    )
    if (
        selected is not AICCM2026DevBBackend.FOUR_CENTER
        and int(np.prod(mesh_tuple)) == 1
    ):
        raise NotImplementedError(
            "aiccm2026dev-b RI and RIJCOSX RKS require at least two cyclic "
            "cells because the native Gamma RKS GDF path is not pair-resolved"
        )
    _require_inactive_quadratic_fallback(
        opts,
        where="run_aiccm2026dev_b_rks",
    )
    _reject_four_center_aux_basis(
        selected,
        aux_basis,
        where="run_aiccm2026dev_b_rks",
    )

    started = perf_counter()
    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        use_3d_ewald = int(system.dim) == 3
        farm_direct_output_cells = (
            float(exact_exchange_coefficients.c_sr) == 0.0
        )
        result = run_pbc_bipole_rks(
            system,
            basis,
            kpoints.to_bloch_kmesh(),
            opts,
            functional=str(functional),
            level_shift_schedule=direct_level_shift_schedule,
            fock_mixing=requested_fock_mixing,
            use_ewald_j_split=use_3d_ewald,
            use_exchange_ewald_split=use_3d_ewald,
            exchange_exxdiv=("ewald" if use_3d_ewald else "none"),
            use_multipole_far_field=False,
            sr_image_precision=_M5_SR_IMAGE_PRECISION,
            farm_output_cells=farm_direct_output_cells,
            output_cell_farming_strategy="cyclic",
            output_cell_farming_task_kind="chi-direct-output-cell",
            xc_density_domain=(
                PeriodicXCDensityDomain.PERIODIC_LATTICE
                if uses_external_xc
                else PeriodicXCDensityDomain.AUTO
            ),
            progress=progress,
            verbose=verbose,
        )
    else:
        result = run_krks_periodic_gdf(
            system,
            basis,
            kpoints,
            opts,
            functional=str(functional),
            aux_basis=aux_basis,
            fock_mixing=requested_fock_mixing,
            use_compcell=True,
            k_exchange=(
                "cosx" if selected is AICCM2026DevBBackend.RIJCOSX else "gdf"
            ),
            gdf_method=gdf_method,
            rsgdf_ke_cutoff=rsgdf_ke_cutoff,
            rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
            mdf_ke_cutoff=mdf_ke_cutoff,
            ibz_native=False,
            compute_gradient=False,
            check_energy_sanity=True,
            xc_density_domain=(
                PeriodicXCDensityDomain.PERIODIC_LATTICE
                if uses_external_xc
                else PeriodicXCDensityDomain.AUTO
            ),
            progress=progress,
            verbose=verbose,
        )
    result = _attach_diagnostics(
        result,
        system,
        basis,
        kpoints,
        mesh_tuple,
        selected,
        f"RKS/{functional}",
        exact_exchange_coefficients,
        perf_counter() - started,
        symmetry_plan,
        opts,
        executed_level_shift_warmup_cycles,
    )
    _attach_fitted_selector_settings(
        result,
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
        mdf_ke_cutoff=mdf_ke_cutoff,
    )
    return result


def run_aiccm2026dev_b_uhf(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    mesh: int | Sequence[int] | None = None,
    wigner_seitz_shells: int | Sequence[int] | None = None,
    backend: str | AICCM2026DevBBackend = AICCM2026DevBBackend.FOUR_CENTER,
    aux_basis: str | None = None,
    gdf_method: str = "rsgdf",
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: float | None = None,
    mdf_ke_cutoff: float = 40.0,
    fock_mixing: float | None = None,
    symmetry_mode: str | AICCM2026DevBSymmetryMode = (
        AICCM2026DevBSymmetryMode.OFF
    ),
    symmetry_precision: float = 1.0e-5,
    symmetry_require_full_group: bool = False,
    progress: bool | ProgressLogger | None = None,
    verbose: int | None = None,
) -> PeriodicKUHFGDFResult | PBCBipoleUHFResult:
    """Minimize the finite-torus unrestricted Hartree--Fock functional."""

    _warn_experimental()
    extension = _resolve_lattice_extension(
        system,
        lattice_extension,
        mesh=mesh,
        wigner_seitz_shells=wigner_seitz_shells,
    )
    selected = _resolve_backend(backend)
    opts = options if options is not None else PeriodicRHFOptions()
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_aiccm2026dev_b_uhf",
    )
    _validate_open_shell_problem(system, opts)
    _require_supported_scf_dimension(system)
    _require_zero_unrestricted_level_shift(
        opts,
        where="run_aiccm2026dev_b_uhf",
    )
    if (
        selected is not AICCM2026DevBBackend.FOUR_CENTER
        and requested_fock_mixing != 0.0
    ):
        raise NotImplementedError(
            f"aiccm2026dev-b {selected.value} UHF does not implement "
            "previous-Fock mixing; use fock_mixing=0 or select "
            "backend='four_center'"
        )
    if selected is not AICCM2026DevBBackend.FOUR_CENTER and gdf_method not in (
        "rsgdf",
        "mdf",
    ):
        raise ValueError("aiccm2026dev-b UHF requires gdf_method='rsgdf' or 'mdf'")
    resolved_rsgdf_tail = _resolve_fitted_rsgdf_tail(
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        where="run_aiccm2026dev_b_uhf",
    )
    if selected is AICCM2026DevBBackend.RIJCOSX and int(np.prod(extension)) == 1:
        raise NotImplementedError(
            "aiccm2026dev-b RIJCOSX requires at least two cyclic cells"
        )
    kpoints = cyclic_gamma_mesh(system, extension)
    exact_exchange_coefficients = _resolve_exact_exchange_coefficients(
        None,
        spin=2,
        where="run_aiccm2026dev_b_uhf",
    )
    symmetry_plan = _build_symmetry_plan(
        system,
        extension,
        symmetry_mode,
        symmetry_precision=symmetry_precision,
        symmetry_require_full_group=symmetry_require_full_group,
    )
    _require_inactive_quadratic_fallback(
        opts,
        where="run_aiccm2026dev_b_uhf",
    )
    _reject_four_center_aux_basis(
        selected,
        aux_basis,
        where="run_aiccm2026dev_b_uhf",
    )
    started = perf_counter()
    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        use_3d_ewald = int(system.dim) == 3
        result = run_pbc_bipole_uhf(
            system,
            basis,
            kpoints.to_bloch_kmesh(),
            opts,
            fock_mixing=requested_fock_mixing,
            use_ewald_j_split=use_3d_ewald,
            use_exchange_ewald_split=use_3d_ewald,
            exchange_exxdiv=("ewald" if use_3d_ewald else "none"),
            use_multipole_far_field=False,
            sr_image_precision=_M5_SR_IMAGE_PRECISION,
            progress=progress,
            verbose=verbose,
        )
    else:
        result = run_kuhf_periodic_gdf(
            system,
            basis,
            kpoints,
            opts,
            functional=None,
            aux_basis=aux_basis,
            gdf_method=gdf_method,
            rsgdf_ke_cutoff=rsgdf_ke_cutoff,
            rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
            mdf_ke_cutoff=mdf_ke_cutoff,
            ibz_native=False,
            compute_gradient=False,
            k_exchange=(
                "cosx" if selected is AICCM2026DevBBackend.RIJCOSX else "gdf"
            ),
            check_energy_sanity=True,
            progress=progress,
            verbose=verbose,
        )
    result = _attach_diagnostics(
        result,
        system,
        basis,
        kpoints,
        extension,
        selected,
        "UHF",
        exact_exchange_coefficients,
        perf_counter() - started,
        symmetry_plan,
        opts,
    )
    _attach_fitted_selector_settings(
        result,
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
        mdf_ke_cutoff=mdf_ke_cutoff,
    )
    return result


def run_aiccm2026dev_b_uks(
    system: PeriodicSystem,
    basis: BasisSet,
    functional: str,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicKSOptions | None = None,
    *,
    mesh: int | Sequence[int] | None = None,
    wigner_seitz_shells: int | Sequence[int] | None = None,
    backend: str | AICCM2026DevBBackend = AICCM2026DevBBackend.FOUR_CENTER,
    aux_basis: str | None = None,
    gdf_method: str = "rsgdf",
    rsgdf_ke_cutoff: float = 200.0,
    rsgdf_tail_ke_cutoff: float | None = None,
    mdf_ke_cutoff: float = 40.0,
    fock_mixing: float | None = None,
    symmetry_mode: str | AICCM2026DevBSymmetryMode = (
        AICCM2026DevBSymmetryMode.OFF
    ),
    symmetry_precision: float = 1.0e-5,
    symmetry_require_full_group: bool = False,
    progress: bool | ProgressLogger | None = None,
    verbose: int | None = None,
) -> PeriodicKUHFGDFResult | PBCBipoleUKSResult:
    """Minimize the finite-torus spin-polarized Kohn--Sham functional."""

    _warn_experimental()
    if not str(functional).strip():
        raise ValueError("aiccm2026dev-b UKS requires a functional name")
    extension = _resolve_lattice_extension(
        system,
        lattice_extension,
        mesh=mesh,
        wigner_seitz_shells=wigner_seitz_shells,
    )
    selected = _resolve_backend(backend)
    opts = options if options is not None else PeriodicKSOptions()
    opts.functional = str(functional)
    func = Functional(str(functional), 2)
    uses_external_xc = _require_supported_external_xc(
        func,
        selected,
        system=system,
        where="run_aiccm2026dev_b_uks",
    )
    if uses_external_xc:
        from ...pbc_bipole_common import reject_bipole_ecp_options

        reject_bipole_ecp_options(
            opts,
            driver="run_aiccm2026dev_b_uks",
            basis=basis,
            system=system,
        )
    _configure_external_xc_grid(
        func,
        opts,
        caller_supplied_options=options is not None,
        where="run_aiccm2026dev_b_uks",
    )
    requested_fock_mixing = resolve_fock_mixing(
        opts,
        fock_mixing,
        where="run_aiccm2026dev_b_uks",
    )
    _validate_open_shell_problem(system, opts)
    _require_supported_scf_dimension(system)
    _require_zero_unrestricted_level_shift(
        opts,
        where="run_aiccm2026dev_b_uks",
    )
    if selected is not AICCM2026DevBBackend.FOUR_CENTER:
        try:
            reject_unscreened_range_separated(
                Functional(str(functional), 2),
                where=(
                    "run_aiccm2026dev_b_uks "
                    f"backend={selected.value!r}"
                ),
            )
        except NotImplementedError as exc:
            raise NotImplementedError(
                "aiccm2026dev-b RI and RIJCOSX backends remain "
                "fail-closed for range-separated functionals. Their "
                "χ-specific operator, validation, and provenance contract "
                "covers full-range exchange only; shared generic COSX "
                "screened-exchange support does not widen that contract. "
                "The separately gated experimental 3D HSE path uses "
                "backend='four_center'."
            ) from exc
    if (
        selected is not AICCM2026DevBBackend.FOUR_CENTER
        and requested_fock_mixing != 0.0
    ):
        raise NotImplementedError(
            f"aiccm2026dev-b {selected.value} UKS does not implement "
            "previous-Fock mixing; use fock_mixing=0 or select "
            "backend='four_center'"
        )
    if selected is not AICCM2026DevBBackend.FOUR_CENTER and gdf_method not in (
        "rsgdf",
        "mdf",
    ):
        raise ValueError("aiccm2026dev-b UKS requires gdf_method='rsgdf' or 'mdf'")
    resolved_rsgdf_tail = _resolve_fitted_rsgdf_tail(
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=rsgdf_tail_ke_cutoff,
        where="run_aiccm2026dev_b_uks",
    )
    if selected is AICCM2026DevBBackend.RIJCOSX and int(np.prod(extension)) == 1:
        raise NotImplementedError(
            "aiccm2026dev-b RIJCOSX requires at least two cyclic cells"
        )
    kpoints = cyclic_gamma_mesh(system, extension)
    exact_exchange_coefficients = _resolve_exact_exchange_coefficients(
        str(functional),
        spin=2,
        where="run_aiccm2026dev_b_uks",
    )
    symmetry_plan = _build_symmetry_plan(
        system,
        extension,
        symmetry_mode,
        symmetry_precision=symmetry_precision,
        symmetry_require_full_group=symmetry_require_full_group,
    )
    _require_inactive_quadratic_fallback(
        opts,
        where="run_aiccm2026dev_b_uks",
    )
    _reject_four_center_aux_basis(
        selected,
        aux_basis,
        where="run_aiccm2026dev_b_uks",
    )
    started = perf_counter()
    if selected is AICCM2026DevBBackend.FOUR_CENTER:
        use_3d_ewald = int(system.dim) == 3
        result = run_pbc_bipole_uks(
            system,
            basis,
            kpoints.to_bloch_kmesh(),
            opts,
            functional=str(functional),
            fock_mixing=requested_fock_mixing,
            use_ewald_j_split=use_3d_ewald,
            use_exchange_ewald_split=use_3d_ewald,
            exchange_exxdiv=("ewald" if use_3d_ewald else "none"),
            use_multipole_far_field=False,
            sr_image_precision=_M5_SR_IMAGE_PRECISION,
            xc_density_domain=(
                PeriodicXCDensityDomain.PERIODIC_LATTICE
                if uses_external_xc
                else PeriodicXCDensityDomain.AUTO
            ),
            progress=progress,
            verbose=verbose,
        )
    else:
        result = run_kuks_periodic_gdf(
            system,
            basis,
            kpoints,
            opts,
            functional=str(functional),
            aux_basis=aux_basis,
            gdf_method=gdf_method,
            rsgdf_ke_cutoff=rsgdf_ke_cutoff,
            rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
            mdf_ke_cutoff=mdf_ke_cutoff,
            ibz_native=False,
            compute_gradient=False,
            k_exchange=(
                "cosx" if selected is AICCM2026DevBBackend.RIJCOSX else "gdf"
            ),
            check_energy_sanity=True,
            xc_density_domain=(
                PeriodicXCDensityDomain.PERIODIC_LATTICE
                if uses_external_xc
                else PeriodicXCDensityDomain.AUTO
            ),
            progress=progress,
            verbose=verbose,
        )
    result = _attach_diagnostics(
        result,
        system,
        basis,
        kpoints,
        extension,
        selected,
        f"UKS/{functional}",
        exact_exchange_coefficients,
        perf_counter() - started,
        symmetry_plan,
        opts,
    )
    _attach_fitted_selector_settings(
        result,
        selected,
        gdf_method=gdf_method,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        rsgdf_tail_ke_cutoff=resolved_rsgdf_tail,
        mdf_ke_cutoff=mdf_ke_cutoff,
    )
    return result
