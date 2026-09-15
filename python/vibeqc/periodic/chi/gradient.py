"""Analytic-gradient surface for χ-CCM / ``aiccm2026dev-b``.

This module intentionally fails closed. The union-and-weight Γ-CCM gradient
derives the direct-torus WSSC molecular-kernel energy, while the neutral
fitted-torus real-Gamma/GDF control has a separate representation-control
gradient that does not yet bind the B-owned χ state, operator, and support.
The χ production route differentiates a declared finite-character Hamiltonian
with ``coulomb_kernel="3d-periodic-g0"`` and
``exchange_q0="bvk-ewald"``. Reusing either unqualified derivative here would
change or leave unattested the finite-N Hamiltonian behind a force calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .scf import (
    AICCM2026DevBFiniteTorusConvention,
    _density_blocks_per_k,
    _spin_density_blocks_per_k,
    inverse_bloch_transform,
)

if TYPE_CHECKING:
    from ..._vibeqc_core import LatticeMatrixSet

__all__ = [
    "AICCM2026DevBEwaldElectrostaticEnergyComponents",
    "AICCM2026DevBEwaldElectrostaticGradientComponents",
    "AICCM2026DevBGradientStatus",
    "aiccm2026dev_b_gradient_status",
    "compute_aiccm2026dev_b_ewald_electrostatic_energy_components",
    "compute_aiccm2026dev_b_ewald_electrostatic_gradient_components",
    "compute_aiccm2026dev_b_ewald_electron_nuclear_gradient",
    "compute_aiccm2026dev_b_ewald_nuclear_gradient",
    "compute_aiccm2026dev_b_fixed_density_kinetic_energy",
    "compute_aiccm2026dev_b_fixed_density_kinetic_gradient",
    "compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy",
    "compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient",
    "compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy",
    "compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient",
    "compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice",
    "compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice",
    "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient",
    "compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian",
    "compute_aiccm2026dev_b_gradient",
    "compute_aiccm2026dev_b_scf_density_lattice",
    "compute_aiccm2026dev_b_scf_ewald_electrostatic_energy_components",
    "compute_aiccm2026dev_b_scf_ewald_electrostatic_gradient_components",
    "run_aiccm2026dev_b_gradient",
]


@dataclass(frozen=True)
class AICCM2026DevBGradientStatus:
    """Current analytic-gradient status for one χ-CCM finite torus."""

    electronic_method: str
    backend: str
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    lattice_extension: tuple[int, int, int] | None
    analytic_gradient_implemented: bool
    blocked_terms: tuple[str, ...]
    reason: str
    implemented_terms: tuple[str, ...] = ()

    @property
    def coulomb_kernel(self) -> str:
        return self.finite_torus_convention.coulomb_kernel

    @property
    def exchange_q0(self) -> str:
        return self.finite_torus_convention.exchange_q0

    @property
    def exchange_q0_applicability(self) -> str:
        return self.finite_torus_convention.exchange_q0_applicability

    @property
    def boundary_model(self) -> str:
        return self.finite_torus_convention.boundary_model


@dataclass(frozen=True)
class AICCM2026DevBEwaldElectrostaticEnergyComponents:
    """Fixed-density 3D Ewald electrostatic energy pieces.

    ``total_fixed_density`` is only the sum of ``nuclear`` and
    ``electron_nuclear`` for the supplied real-torus density.  It is a value
    companion for component-level finite-difference audits, not a total SCF
    or post-HF energy derivative assembly.
    """

    nuclear: float
    electron_nuclear: float
    total_fixed_density: float
    ewald_alpha_bohr_inv: float
    nuclear_cutoff_bohr: float
    reciprocal_cutoff_bohr_inv: float
    precision: float
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention


@dataclass(frozen=True)
class AICCM2026DevBEwaldElectrostaticGradientComponents:
    """Fixed-density 3D Ewald electrostatic derivative pieces.

    ``total_fixed_density`` is only the sum of ``nuclear`` and
    ``electron_nuclear`` for the supplied real-torus density.  It is not a
    total SCF or post-HF force because the Pulay/adjoint, exchange seam,
    RI/RIJCOSX response, and correlated response terms are intentionally not
    included.
    """

    nuclear: np.ndarray
    electron_nuclear: np.ndarray
    total_fixed_density: np.ndarray
    ewald_alpha_bohr_inv: float
    nuclear_cutoff_bohr: float
    reciprocal_cutoff_bohr_inv: float
    precision: float
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention


def _positive_integer_triplet(
    value: object,
    label: str,
) -> tuple[int, int, int]:
    """Return one exact positive integer triplet or fail provenance closed."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"χ-CCM-B {label} must be a positive integer triplet") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"χ-CCM-B {label} must be a finite length-3 triplet")
    rounded = np.rint(array)
    if not np.array_equal(array, rounded):
        raise ValueError(f"χ-CCM-B {label} must contain exact integers")
    values = tuple(int(item) for item in rounded)
    if any(item < 1 for item in values):
        raise ValueError(f"χ-CCM-B {label} entries must be positive")
    return values


def _diagnostics_from_result(result: object) -> object | None:
    return getattr(result, "aiccm2026dev_b", None)


def _finite_torus_convention_from_result(
    result: object,
) -> AICCM2026DevBFiniteTorusConvention:
    diagnostics = _diagnostics_from_result(result)
    candidates = [
        ("diagnostics", getattr(diagnostics, "finite_torus_convention", None)),
        ("result", getattr(result, "finite_torus_convention", None)),
    ]
    present = [(label, value) for label, value in candidates if value is not None]
    if not present:
        raise TypeError(
            "χ-CCM gradient status requires an aiccm2026dev-b result carrying "
            "a finite-torus convention descriptor"
        )
    for label, convention in present:
        if not isinstance(convention, AICCM2026DevBFiniteTorusConvention):
            raise TypeError(
                "χ-CCM gradient status found a finite-torus convention on "
                f"{label} with an unexpected type {type(convention).__name__!r}"
            )
    convention = present[0][1]
    for label, candidate in present[1:]:
        if candidate != convention:
            raise ValueError(
                "χ-CCM gradient status requires the diagnostics and top-level "
                f"finite-torus conventions to agree; {label} differs"
            )
    return convention


def aiccm2026dev_b_gradient_status(result: object) -> AICCM2026DevBGradientStatus:
    """Return the fail-closed χ-CCM analytic-gradient status for ``result``.

    The status object is useful for output manifests and workflow guards: it
    records the same finite-torus Hamiltonian convention as the parent SCF or
    post-HF object, while explicitly stating that no analytic force is
    currently returned for this B-line Hamiltonian.
    """

    convention = _finite_torus_convention_from_result(result)
    diagnostics = _diagnostics_from_result(result)
    electronic_method = str(
        getattr(
            diagnostics,
            "electronic_method",
            getattr(result, "backend", type(result).__name__),
        )
    )
    backend = str(getattr(result, "backend", getattr(diagnostics, "backend", "unknown")))
    convention_mesh = _positive_integer_triplet(
        convention.character_mesh_shape,
        "convention character mesh",
    )
    convention_repetitions = _positive_integer_triplet(
        convention.bvk_madelung_supercell_repetitions,
        "convention BvK repetitions",
    )
    if convention_repetitions != convention_mesh:
        raise ValueError(
            "χ-CCM gradient status requires convention character mesh and "
            "BvK repetitions to agree"
        )
    mesh_aliases: list[tuple[str, object]] = []
    if diagnostics is not None:
        mesh_aliases.extend(
            [
                ("diagnostics mesh", getattr(diagnostics, "mesh", None)),
                (
                    "diagnostics lattice extension",
                    getattr(diagnostics, "lattice_extension", None),
                ),
            ]
        )
    mesh_aliases.extend(
        [
            ("result mesh", getattr(result, "mesh", None)),
            ("result lattice extension", getattr(result, "lattice_extension", None)),
        ]
    )
    present_mesh_aliases = [
        (label, value) for label, value in mesh_aliases if value is not None
    ]
    if not present_mesh_aliases:
        raise TypeError(
            "χ-CCM gradient status requires an explicit result mesh or "
            "lattice-extension alias independent of the convention descriptor"
        )
    validated_aliases: list[tuple[int, int, int]] = []
    for label, value in present_mesh_aliases:
        alias = _positive_integer_triplet(value, label)
        if alias != convention_mesh:
            raise ValueError(
                "χ-CCM gradient status requires every result mesh alias to "
                f"match the convention character mesh; {label} is {alias}, "
                f"expected {convention_mesh}"
            )
        validated_aliases.append(alias)
    lattice_extension = validated_aliases[0]
    return AICCM2026DevBGradientStatus(
        electronic_method=electronic_method,
        backend=backend,
        finite_torus_convention=convention,
        lattice_extension=lattice_extension,
        analytic_gradient_implemented=False,
        blocked_terms=(
            "variational SCF assembly beyond fixed-matrix component derivatives",
            "production BvK exchange-q=0 seam assembly or screened-exchange "
            "kernel derivative",
            "stationary SCF energy-weighted density binding and Pulay/adjoint "
            "assembly",
            "RI/RIJCOSX three-center derivative and metric response",
            "DFT exchange-correlation quadrature and grid derivative where applicable",
            "post-HF relaxed-density and amplitude-response contributions",
        ),
        reason=(
            "χ-CCM analytic gradients must differentiate the declared "
            "finite-character Hamiltonian. The union-and-weight Γ-CCM "
            "WSSC gradient is a different derivative, while the neutral "
            "fitted-torus real-Gamma/GDF control gradient lacks the "
            "B-owned state, operator, and support binding; neither is "
            "substituted."
        ),
        implemented_terms=(
            "3D Ewald nuclear-repulsion derivative component",
            "fixed-density restricted active BvK exchange-q=0 seam energy component",
            "fixed-density restricted active BvK exchange-q=0 seam AO-centre "
            "derivative component",
            "fixed-density unrestricted active BvK exchange-q=0 seam energy "
            "component",
            "fixed-density unrestricted active BvK exchange-q=0 seam AO-centre "
            "derivative component",
            "fixed energy-weighted 3D overlap Lagrangian component",
            "fixed energy-weighted 3D overlap AO-centre derivative component",
            "fixed-input restricted energy-weighted-density algebra and "
            "inverse-character fold",
            "fixed-input unrestricted energy-weighted-density spin algebra "
            "and inverse-character fold",
            "fixed-density 3D kinetic energy component",
            "fixed-density 3D kinetic AO-centre derivative component",
            "fixed-density 3D Ewald electrostatic energy component bundle",
            "fixed-density 3D Ewald electron-nuclear derivative component",
            "fixed-density 3D Ewald electrostatic component bundle",
            "SCF density inverse-Bloch fold for fixed-density components",
            "result/system finite-torus mesh and BvK lattice binding",
            "SCF-density 3D Ewald electrostatic energy component bundle",
            "SCF-density 3D Ewald electrostatic component bundle",
        ),
    )


def _require_3d_ewald_component(
    system: object,
    status: AICCM2026DevBGradientStatus,
    component_name: str,
) -> None:
    convention = status.finite_torus_convention
    if convention.lattice_vector_convention != "columns":
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires "
            "lattice_vector_convention='columns'"
        )
    if convention.coulomb_kernel != "3d-periodic-g0":
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires "
            "coulomb_kernel='3d-periodic-g0'"
        )
    if convention.exchange_q0 != "bvk-ewald":
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires "
            "exchange_q0='bvk-ewald'"
        )
    convention_dim = int(convention.periodic_dimension)
    system_dim = int(getattr(system, "dim", -1))
    if convention_dim != 3 or system_dim != 3:
        raise NotImplementedError(
            f"χ-CCM-B {component_name} is implemented only "
            "for 3D periodic finite-torus results"
        )
    if convention.boundary_model != "3d-periodic":
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires boundary_model='3d-periodic'"
        )
    mesh = _positive_integer_triplet(
        convention.character_mesh_shape,
        "convention character mesh",
    )
    repetitions = _positive_integer_triplet(
        convention.bvk_madelung_supercell_repetitions,
        "convention BvK repetitions",
    )
    if repetitions != mesh or status.lattice_extension != mesh:
        raise ValueError(
            f"χ-CCM-B {component_name} requires result mesh, character mesh, "
            "and BvK repetitions to agree"
        )
    system_lattice = np.asarray(getattr(system, "lattice", None), dtype=float)
    if system_lattice.shape != (3, 3) or not np.all(np.isfinite(system_lattice)):
        raise ValueError(
            f"χ-CCM-B {component_name} requires a finite 3x3 system lattice"
        )
    recorded_lattice = np.asarray(
        convention.bvk_madelung_supercell_lattice_bohr,
        dtype=float,
    )
    if recorded_lattice.shape != (3, 3) or not np.all(np.isfinite(recorded_lattice)):
        raise ValueError(
            f"χ-CCM-B {component_name} requires a finite recorded 3x3 BvK lattice"
        )
    expected_lattice = system_lattice @ np.diag(np.asarray(mesh, dtype=float))
    if not np.allclose(
        recorded_lattice,
        expected_lattice,
        atol=1.0e-12,
        rtol=0.0,
    ):
        maximum_error = float(np.max(np.abs(recorded_lattice - expected_lattice)))
        raise ValueError(
            f"χ-CCM-B {component_name} recorded BvK supercell lattice does "
            "not match system.lattice @ diag(character_mesh); maximum "
            f"difference is {maximum_error:.3e} bohr"
        )


def _require_complete_gamma_character_mesh(
    kpoints_frac: np.ndarray,
    weights: np.ndarray,
    mesh: tuple[int, int, int],
    component_name: str = "density extraction",
) -> np.ndarray:
    """Require the complete unreduced Gamma-centred net and return labels."""

    expected_count = int(np.prod(mesh))
    if kpoints_frac.shape != (expected_count, 3):
        raise ValueError(
            f"χ-CCM-B {component_name} requires the complete unreduced "
            f"Gamma-centred character mesh {mesh} with {expected_count} points"
        )
    if weights.shape != (expected_count,) or not np.all(np.isfinite(weights)):
        raise ValueError(
            f"χ-CCM-B {component_name} requires one finite weight per "
            "character point"
        )
    if not np.allclose(
        weights,
        np.full(expected_count, 1.0 / expected_count),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise ValueError(
            f"χ-CCM-B {component_name} requires uniform weights on the "
            "complete character mesh"
        )
    if not np.all(np.isfinite(kpoints_frac)):
        raise ValueError(f"χ-CCM-B {component_name} requires finite k-points")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = kpoints_frac * np.asarray(mesh, dtype=float)[None, :]
    exact_integer_limit = float((1 << 53) - 1)
    if not np.all(np.isfinite(scaled)) or np.any(
        np.abs(scaled) > exact_integer_limit
    ):
        raise ValueError(
            f"χ-CCM-B {component_name} requires character coordinates "
            "within the supported exact-integer range"
        )
    integer_labels = np.rint(scaled)
    if not np.allclose(scaled, integer_labels, atol=1.0e-12, rtol=0.0):
        raise ValueError(
            f"χ-CCM-B {component_name} requires an unshifted Gamma-centred "
            "character mesh"
        )
    mesh_array = np.asarray(mesh, dtype=int)
    integer_labels = integer_labels.astype(np.int64)
    residues = {
        tuple(int(value) for value in np.mod(label, mesh_array))
        for label in integer_labels
    }
    if len(residues) != expected_count:
        raise ValueError(
            f"χ-CCM-B {component_name} requires every finite-character "
            "residue exactly once"
        )
    return integer_labels


def _real_character_metadata(
    value: object,
    label: str,
    component_name: str,
) -> np.ndarray:
    """Return character metadata without silently discarding an imaginary part."""

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"χ-CCM-B {component_name} requires real numeric {label}"
        ) from exc
    if np.iscomplexobj(raw):
        if not np.all(np.isfinite(raw.imag)) or np.any(raw.imag != 0.0):
            raise ValueError(
                f"χ-CCM-B {component_name} requires real numeric {label}"
            )
        raw = raw.real
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"χ-CCM-B {component_name} requires real numeric {label}"
        ) from exc


def _ewald_alpha_from_result(
    result: object,
    ewald_alpha: float | None,
    component_name: str,
) -> float:
    alpha = (
        float(ewald_alpha)
        if ewald_alpha is not None
        else getattr(result, "ewald_alpha_bohr_inv", None)
    )
    if alpha is None or float(alpha) <= 0.0:
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires the Ewald alpha used by the "
            "SCF energy"
        )
    return float(alpha)


def _require_ewald_lattice_options(
    lattice_options: object,
    component_name: str,
) -> None:
    from ..._vibeqc_core import CoulombMethod

    if getattr(lattice_options, "coulomb_method", None) != CoulombMethod.EWALD_3D:
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires "
            "lattice_options.coulomb_method == CoulombMethod.EWALD_3D"
        )


def _cell_keys(lattice_set: object) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        tuple(int(value) for value in np.asarray(cell.index, dtype=int).reshape(3))
        for cell in getattr(lattice_set, "cells")
    )


def _require_matching_density_cell_list(
    density: object,
    template: object,
    component_name: str,
) -> None:
    density_cells = tuple(getattr(density, "cells"))
    template_cells = tuple(getattr(template, "cells"))
    density_keys = _cell_keys(density)
    template_keys = _cell_keys(template)
    density_blocks = tuple(np.asarray(block) for block in getattr(density, "blocks"))
    density_nbf = int(getattr(density, "nbf"))
    template_nbf = int(getattr(template, "nbf"))
    if density_nbf != template_nbf:
        raise ValueError(
            f"χ-CCM-B {component_name} requires density AO dimension "
            f"{template_nbf}, found {density_nbf}"
        )
    if len(density_blocks) != len(density_keys):
        raise ValueError(
            f"χ-CCM-B {component_name} requires density blocks to align with "
            f"density cells; found {len(density_blocks)} blocks for "
            f"{len(density_keys)} cells"
        )
    expected_shape = (template_nbf, template_nbf)
    for key, block in zip(density_keys, density_blocks):
        if block.shape != expected_shape:
            raise ValueError(
                f"χ-CCM-B {component_name} requires density block shape "
                f"{expected_shape} at cell {key}, found {block.shape}"
            )
    if density_keys == template_keys:
        for key, density_cell, template_cell in zip(
            density_keys,
            density_cells,
            template_cells,
        ):
            density_cart = np.asarray(density_cell.r_cart, dtype=float).reshape(3)
            template_cart = np.asarray(template_cell.r_cart, dtype=float).reshape(3)
            if not np.allclose(
                density_cart,
                template_cart,
                atol=1.0e-12,
                rtol=1.0e-12,
            ):
                maximum_error = float(np.max(np.abs(density_cart - template_cart)))
                raise ValueError(
                    f"χ-CCM-B {component_name} requires density Cartesian "
                    f"translations to match lattice_options exactly; cell "
                    f"{key} differs by {maximum_error:.3e} bohr"
                )
        return
    density_key_set = set(density_keys)
    template_key_set = set(template_keys)
    missing = [key for key in template_keys if key not in density_key_set]
    extra = [key for key in density_keys if key not in template_key_set]
    details: list[str] = []
    if missing:
        details.append(f"missing cells {missing[:4]}")
    if extra:
        details.append(f"extra cells {extra[:4]}")
    if not details:
        details.append("same cells in a different order")
    raise ValueError(
        f"χ-CCM-B {component_name} requires the fixed-density cell list to "
        "match lattice_options exactly; " + ", ".join(details)
    )


def _effective_electrons_from_result(result: object) -> int:
    diagnostics = _diagnostics_from_result(result)
    value = getattr(result, "effective_n_electrons", None)
    if value is None and diagnostics is not None:
        value = getattr(diagnostics, "effective_electron_count", None)
    if value is None:
        raise TypeError(
            "χ-CCM-B density extraction requires a result with the effective "
            "electron count recorded"
        )
    return int(value)


def _spin_occupations_from_effective_count(
    system: object,
    effective_electrons: int,
) -> tuple[int, int]:
    two_s = int(getattr(system, "multiplicity", 1)) - 1
    alpha_twice = int(effective_electrons) + two_s
    beta_twice = int(effective_electrons) - two_s
    if alpha_twice < 0 or beta_twice < 0 or alpha_twice % 2 or beta_twice % 2:
        raise ValueError(
            "χ-CCM-B density extraction requires the effective electron count "
            "and multiplicity to define non-negative integer alpha/beta "
            "occupations"
        )
    return alpha_twice // 2, beta_twice // 2


def _require_closed_shell_effective_count(
    system: object,
    effective_electrons: int,
) -> None:
    multiplicity = int(getattr(system, "multiplicity", 1))
    if (
        multiplicity != 1
        or int(effective_electrons) < 0
        or int(effective_electrons) % 2
    ):
        raise ValueError(
            "χ-CCM-B density extraction requires closed-shell records to "
            "carry singlet multiplicity and a non-negative even effective "
            "electron count"
        )


def _total_density_blocks_per_k(
    system: object,
    result: object,
    effective_electrons: int,
) -> list[np.ndarray]:
    if hasattr(result, "density_alpha") and hasattr(result, "density_beta"):
        n_alpha, n_beta = _spin_occupations_from_effective_count(
            system,
            effective_electrons,
        )
        alpha = _spin_density_blocks_per_k(result, "alpha", n_alpha)
        beta = _spin_density_blocks_per_k(result, "beta", n_beta)
        if len(alpha) != len(beta):
            raise ValueError("χ-CCM-B alpha/beta density block counts differ")
        return [a + b for a, b in zip(alpha, beta)]
    _require_closed_shell_effective_count(system, effective_electrons)
    return _density_blocks_per_k(result, effective_electrons)


def _require_density_k_ao_shape(
    density_k: Sequence[np.ndarray],
    basis: object,
    component_name: str = "density extraction",
    density_label: str = "",
) -> None:
    nbf = int(getattr(basis, "nbasis"))
    expected_shape = (nbf, nbf)
    labelled = f"{density_label} " if density_label else ""
    for index, block in enumerate(density_k):
        shape = np.asarray(block).shape
        if shape != expected_shape:
            raise ValueError(
                f"χ-CCM-B {component_name} requires {labelled}k-density block "
                f"{index} to have AO shape {expected_shape}, found {shape}"
            )


def compute_aiccm2026dev_b_scf_density_lattice(
    system: object,
    basis: object,
    result: object,
    *,
    lattice_options: object,
    imaginary_tolerance: float = 1.0e-7,
) -> "LatticeMatrixSet":
    """Return the χ-CCM-B SCF density on a lattice-sum cell list.

    The returned :class:`LatticeMatrixSet` uses the same cell ordering as
    ``compute_overlap_lattice(basis, system, lattice_options)``.  Its blocks
    are the finite-character inverse Bloch transform of the converged SCF
    k-density, so it can be passed directly to fixed-density component helpers
    such as
    :func:`compute_aiccm2026dev_b_ewald_electron_nuclear_gradient`.

    This is still only density plumbing for component-level gradient
    derivations.  It does not add Pulay/adjoint, exchange-seam, RI/RIJCOSX, or
    response terms, and the total-gradient entry point remains fail-closed.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "SCF density inverse-Bloch fold",
    )
    effective_electrons = _effective_electrons_from_result(result)
    density_k = _total_density_blocks_per_k(system, result, effective_electrons)
    kpoints_frac = _real_character_metadata(
        getattr(result, "kpoints_frac"),
        "k-points",
        "density extraction",
    ).reshape(-1, 3)
    weights = _real_character_metadata(
        getattr(result, "kpoint_weights"),
        "character weights",
        "density extraction",
    ).reshape(-1)
    if len(density_k) != kpoints_frac.shape[0] or weights.shape != (
        kpoints_frac.shape[0],
    ):
        raise ValueError(
            "χ-CCM-B density extraction requires matching density, k-point, "
            "and weight counts"
        )
    _require_complete_gamma_character_mesh(
        kpoints_frac,
        weights,
        status.lattice_extension,
    )
    _require_density_k_ao_shape(density_k, basis)

    from ..._vibeqc_core import compute_overlap_lattice

    density_lattice = compute_overlap_lattice(basis, system, lattice_options)
    translations = [
        tuple(int(value) for value in np.asarray(cell.index, dtype=int).reshape(3))
        for cell in density_lattice.cells
    ]
    blocks = inverse_bloch_transform(density_k, kpoints_frac, translations, weights)
    imaginary_residual = float(np.max(np.abs(blocks.imag))) if blocks.size else 0.0
    if imaginary_residual > float(imaginary_tolerance):
        raise NotImplementedError(
            "χ-CCM-B SCF density extraction would discard a non-real "
            f"finite-torus density residue ({imaginary_residual:.3e})"
        )
    for index, block in enumerate(blocks.real):
        density_lattice.set_block(index, np.ascontiguousarray(block, dtype=float))
    return density_lattice


def _validated_fixed_density_kinetic_operator(
    system: object,
    basis: object,
    density: object,
    lattice_options: object,
    component_name: str,
) -> object:
    """Build the kinetic lattice operator after exact support validation."""

    from ..._vibeqc_core import compute_kinetic_lattice

    kinetic = compute_kinetic_lattice(basis, system, lattice_options)
    _require_matching_density_cell_list(density, kinetic, component_name)
    return kinetic


def compute_aiccm2026dev_b_fixed_density_kinetic_energy(
    system: object,
    basis: object,
    result: object,
    density: object,
    *,
    lattice_options: object,
) -> float:
    """Return the fixed-density 3D kinetic-energy component.

    The scalar is
    ``sum_g Tr[D(g) T(g)]`` for the caller-supplied real-torus density and
    lattice-cell support.  The AO coefficients in ``D(g)`` and the lattice
    vectors are held fixed.  This is the value companion for checking the
    kinetic AO-centre derivative below; it is not a total SCF energy or a
    relaxed response quantity.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "fixed-density kinetic energy component",
    )
    kinetic = _validated_fixed_density_kinetic_operator(
        system,
        basis,
        density,
        lattice_options,
        "fixed-density kinetic energy component",
    )
    from ...pbc_bipole_common import _lattice_contract

    return float(
        _lattice_contract(
            density,
            kinetic,
            operator_name="fixed-density kinetic energy component",
        )
    )


def compute_aiccm2026dev_b_fixed_density_kinetic_gradient(
    system: object,
    basis: object,
    result: object,
    density: object,
    *,
    lattice_options: object,
) -> np.ndarray:
    """Return the fixed-density 3D kinetic AO-centre derivative.

    This differentiates only
    ``sum_g Tr[D(g) T(g)]`` with respect to primitive-cell nuclear positions,
    holding the numerical density blocks and lattice fixed.  Both the bra and
    image-ket Gaussian-centre derivatives are accumulated onto their owning
    primitive-cell atoms.  Density response, the energy-weighted overlap
    Pulay term, electron-nuclear and electron-electron derivatives, the BvK
    exchange seam, and correlated response are excluded deliberately.

    The caller must supply the exact real-space cell support being audited.
    In particular, this helper does not claim that arbitrary
    ``lattice_options`` reproduce the internally resolved one-electron cutoff
    of an RI or RIJCOSX SCF calculation.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "fixed-density kinetic gradient component",
    )
    _validated_fixed_density_kinetic_operator(
        system,
        basis,
        density,
        lattice_options,
        "fixed-density kinetic gradient component",
    )
    from ..._vibeqc_core import kinetic_lattice_gradient_contribution

    return np.asarray(
        kinetic_lattice_gradient_contribution(
            basis,
            system,
            density,
            lattice_options,
        ),
        dtype=float,
    )


def _validated_fixed_energy_weighted_overlap_operator(
    system: object,
    basis: object,
    energy_weighted_density: object,
    lattice_options: object,
    component_name: str,
) -> object:
    """Build the overlap lattice operator after exact support validation."""

    from ..._vibeqc_core import compute_overlap_lattice

    overlap = compute_overlap_lattice(basis, system, lattice_options)
    _require_matching_density_cell_list(
        energy_weighted_density,
        overlap,
        component_name,
    )
    return overlap


def compute_aiccm2026dev_b_fixed_energy_weighted_overlap_lagrangian(
    system: object,
    basis: object,
    result: object,
    energy_weighted_density: object,
    *,
    lattice_options: object,
) -> float:
    """Return the fixed energy-weighted overlap Lagrangian component.

    The scalar is
    ``-sum_g,mu,nu W_mu,nu(g) S_mu,nu(g)`` for a caller-supplied
    real-torus energy-weighted density ``W`` and lattice-cell support.  It is
    the value companion for checking the overlap AO-centre derivative below,
    not an electronic-energy contribution.  In particular, this helper does
    not construct the stationary SCF ``W`` or complete a Pulay/adjoint
    assembly.  ``result`` supplies only the declared χ-CCM-B convention; this
    helper does not claim that the caller's ``W`` was extracted from it.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "fixed energy-weighted overlap Lagrangian component",
    )
    overlap = _validated_fixed_energy_weighted_overlap_operator(
        system,
        basis,
        energy_weighted_density,
        lattice_options,
        "fixed energy-weighted overlap Lagrangian component",
    )
    from ...pbc_bipole_common import _lattice_contract

    return -float(
        _lattice_contract(
            energy_weighted_density,
            overlap,
            operator_name="fixed energy-weighted overlap Lagrangian component",
        )
    )


def compute_aiccm2026dev_b_fixed_energy_weighted_overlap_gradient(
    system: object,
    basis: object,
    result: object,
    energy_weighted_density: object,
    *,
    lattice_options: object,
) -> np.ndarray:
    """Return the fixed energy-weighted overlap AO-centre derivative.

    This differentiates only
    ``-sum_g,mu,nu W_mu,nu(g) S_mu,nu(g)`` with respect to primitive-cell
    nuclear positions, holding the numerical ``W`` blocks and lattice fixed.
    Both the bra and image-ket Gaussian-centre derivatives are accumulated
    onto their owning primitive-cell atoms.  Constructing the stationary SCF
    energy-weighted density, orbital response, the remaining Hamiltonian
    derivatives, and total Pulay/adjoint assembly are excluded deliberately.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "fixed energy-weighted overlap gradient component",
    )
    _validated_fixed_energy_weighted_overlap_operator(
        system,
        basis,
        energy_weighted_density,
        lattice_options,
        "fixed energy-weighted overlap gradient component",
    )
    from ..._vibeqc_core import overlap_lattice_gradient_contribution

    return np.asarray(
        overlap_lattice_gradient_contribution(
            basis,
            system,
            energy_weighted_density,
            lattice_options,
        ),
        dtype=float,
    )


def _positive_fixed_w_tolerance(value: object, label: str) -> float:
    """Return one positive finite fixed-input matrix tolerance."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"χ-CCM-B {label} must be a positive finite scalar")
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"χ-CCM-B {label} must be a positive finite scalar"
        ) from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"χ-CCM-B {label} must be a positive finite scalar")
    return tolerance


def _fixed_w_square_character_stack(
    blocks: Sequence[np.ndarray],
    *,
    count: int,
    nbf: int,
    label: str,
    component_name: str,
) -> np.ndarray:
    """Return a finite square AO matrix at every character."""

    try:
        matrices = tuple(np.asarray(block, dtype=np.complex128) for block in blocks)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"χ-CCM-B {component_name} requires finite {label} blocks"
        ) from exc
    if len(matrices) != count:
        raise ValueError(
            f"χ-CCM-B {component_name} requires one {label} block per "
            f"character; got {len(matrices)} for {count} characters"
        )
    expected = (nbf, nbf)
    for index, matrix in enumerate(matrices):
        if matrix.shape != expected:
            raise ValueError(
                f"χ-CCM-B {component_name} requires {label} block {index} "
                f"to have AO shape {expected}, found {matrix.shape}"
            )
    stack = np.stack(matrices)
    if not np.all(np.isfinite(stack.real)) or not np.all(np.isfinite(stack.imag)):
        raise ValueError(
            f"χ-CCM-B {component_name} requires finite {label} blocks"
        )
    return stack


def _fixed_w_occupied_character_stack(
    blocks: Sequence[np.ndarray],
    *,
    count: int,
    nbf: int,
    component_name: str,
    spin_label: str = "",
    allow_zero_rank: bool = False,
) -> np.ndarray:
    """Return one finite common-rank occupied coefficient block per character."""

    label_prefix = f"{spin_label} " if spin_label else ""
    try:
        coefficients = tuple(
            np.asarray(block, dtype=np.complex128) for block in blocks
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"χ-CCM-B {component_name} requires finite {label_prefix}occupied "
            "coefficient blocks"
        ) from exc
    if len(coefficients) != count:
        raise ValueError(
            f"χ-CCM-B {component_name} requires one {label_prefix}occupied "
            "coefficient "
            f"block per character; got {len(coefficients)} for {count} characters"
        )
    n_occ: int | None = None
    for index, coefficient in enumerate(coefficients):
        if coefficient.ndim != 2 or coefficient.shape[0] != nbf:
            raise ValueError(
                f"χ-CCM-B {component_name} requires {label_prefix}occupied "
                "coefficient "
                f"block {index} to have {nbf} AO rows, found {coefficient.shape}"
            )
        if n_occ is None:
            n_occ = int(coefficient.shape[1])
            minimum_rank = 0 if allow_zero_rank else 1
            if n_occ < minimum_rank or n_occ > nbf:
                lower_bound = 0 if allow_zero_rank else 1
                raise ValueError(
                    f"χ-CCM-B {component_name} requires a "
                    f"{label_prefix}occupied rank between {lower_bound} and "
                    f"{nbf}, found {n_occ}"
                )
        elif coefficient.shape[1] != n_occ:
            raise ValueError(
                f"χ-CCM-B {component_name} requires one common "
                f"{label_prefix}occupied "
                f"rank; block {index} has {coefficient.shape[1]}, expected {n_occ}"
            )
    stack = np.stack(coefficients)
    if not np.all(np.isfinite(stack.real)) or not np.all(np.isfinite(stack.imag)):
        raise ValueError(
            f"χ-CCM-B {component_name} requires finite {label_prefix}occupied "
            "coefficient blocks"
        )
    return stack


def _fixed_w_time_reversal_residual(
    matrices: np.ndarray,
    character_labels: np.ndarray,
    mesh: tuple[int, int, int],
) -> float:
    """Return ``max_q ||M(-q)-M(q)*||_max`` on a complete mesh."""

    mesh_array = np.asarray(mesh, dtype=int)
    residues = [
        tuple(int(value) for value in np.mod(label, mesh_array))
        for label in character_labels
    ]
    residue_to_index = {residue: index for index, residue in enumerate(residues)}
    residual = 0.0
    for index, residue in enumerate(residues):
        opposite = tuple(
            int(value) for value in np.mod(-np.asarray(residue), mesh_array)
        )
        with np.errstate(over="ignore", invalid="ignore"):
            difference = (
                matrices[residue_to_index[opposite]]
                - matrices[index].conj()
            )
        if not np.all(np.isfinite(difference)):
            return float("inf")
        residual = max(
            residual,
            float(np.max(np.abs(difference))),
        )
    return residual


def _require_finite_fixed_w_array(
    array: np.ndarray,
    label: str,
    component_name: str,
) -> None:
    """Fail closed when fixed-input W algebra produces non-finite values."""

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"χ-CCM-B {component_name} produced non-finite {label}"
        )


def _fixed_w_valid_ks_method_token(method: str, prefix: str) -> bool:
    """Return whether one normalized method names a single KS functional."""

    if not method.startswith(prefix):
        return False
    functional = method.removeprefix(prefix)
    components = functional.split("/")
    return (
        bool(functional.strip())
        and functional == functional.strip()
        and all(
            bool(component)
            and component == component.strip()
            and all(
                character.isprintable()
                and (not character.isspace() or character == " ")
                for character in component
            )
            for component in components
        )
    )


def _require_fixed_w_spin_method(
    status: AICCM2026DevBGradientStatus,
    spin_mode: str,
    component_name: str,
) -> None:
    """Require one well-formed restricted or unrestricted SCF method token."""

    method = status.electronic_method.upper()
    if spin_mode == "restricted":
        valid = method == "RHF" or _fixed_w_valid_ks_method_token(method, "RKS/")
        if not valid:
            raise NotImplementedError(
                f"χ-CCM-B {component_name} currently accepts only restricted "
                f"RHF/RKS records, found {status.electronic_method!r}"
            )
        return
    if spin_mode == "unrestricted":
        valid = method == "UHF" or _fixed_w_valid_ks_method_token(method, "UKS/")
        if not valid:
            raise NotImplementedError(
                f"χ-CCM-B {component_name} currently accepts only unrestricted "
                f"UHF or UKS/<functional> records, found "
                f"{status.electronic_method!r}"
            )
        return
    raise ValueError(f"unknown χ-CCM-B fixed-input W spin mode {spin_mode!r}")


def _validate_fixed_w_matrix_symmetries(
    matrices: np.ndarray,
    *,
    label: str,
    character_labels: np.ndarray,
    mesh: tuple[int, int, int],
    tolerance: float,
    component_name: str,
) -> None:
    """Require finite Hermitian, time-reversal-consistent character blocks."""

    with np.errstate(over="ignore", invalid="ignore"):
        hermiticity_difference = matrices - matrices.conj().transpose(0, 2, 1)
    _require_finite_fixed_w_array(
        hermiticity_difference,
        f"{label} Hermiticity residual",
        component_name,
    )
    hermiticity_residual = float(np.max(np.abs(hermiticity_difference)))
    if hermiticity_residual > tolerance:
        raise ValueError(
            f"χ-CCM-B {component_name} requires Hermitian {label} blocks; "
            f"maximum residual is {hermiticity_residual:.3e}"
        )
    time_reversal_residual = _fixed_w_time_reversal_residual(
        matrices,
        character_labels,
        mesh,
    )
    if time_reversal_residual > tolerance:
        if label.startswith(("alpha ", "beta ")):
            raise NotImplementedError(
                f"χ-CCM-B {component_name} requires {label} blocks to be "
                "time-reversal-consistent; maximum residual is "
                f"{time_reversal_residual:.3e}"
            )
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires time-reversal-consistent "
            f"{label} blocks; maximum residual is {time_reversal_residual:.3e}"
        )


def _fixed_w_character_channel(
    overlap_input: np.ndarray,
    occupied: np.ndarray,
    fock: np.ndarray,
    *,
    occupation_factor: float,
    spin_label: str,
    character_labels: np.ndarray,
    mesh: tuple[int, int, int],
    consistency: float,
    component_name: str,
) -> np.ndarray:
    """Construct and validate one spin channel of fixed-input ``W(q)``."""

    count, nbf, _ = overlap_input.shape
    n_occ = int(occupied.shape[2])
    identity = np.eye(n_occ)
    label_prefix = f"{spin_label} " if spin_label else ""
    projectors = np.empty((count, nbf, nbf), dtype=np.complex128)
    energy_weighted = np.empty_like(projectors)
    for index, (overlap_q, occupied_q, fock_q) in enumerate(
        zip(overlap_input, occupied, fock)
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            gram = occupied_q.conj().T @ overlap_q @ occupied_q
        _require_finite_fixed_w_array(
            gram,
            f"{label_prefix}occupied Gram block at character {index}",
            component_name,
        )
        gram_delta = gram - identity
        orthonormality_residual = (
            float(np.max(np.abs(gram_delta))) if gram_delta.size else 0.0
        )
        if orthonormality_residual > consistency:
            raise ValueError(
                f"χ-CCM-B {component_name} requires {label_prefix}occupied "
                f"S-orthonormality at character {index}; maximum residual is "
                f"{orthonormality_residual:.3e}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            projector = occupied_q @ occupied_q.conj().T
            lagrange = occupied_q.conj().T @ fock_q @ occupied_q
        _require_finite_fixed_w_array(
            projector,
            f"{label_prefix}occupied projector at character {index}",
            component_name,
        )
        _require_finite_fixed_w_array(
            lagrange,
            f"{label_prefix}occupied Lagrange block at character {index}",
            component_name,
        )
        projectors[index] = projector
        with np.errstate(over="ignore", invalid="ignore"):
            lagrange_delta = lagrange - lagrange.conj().T
        _require_finite_fixed_w_array(
            lagrange_delta,
            f"{label_prefix}occupied Lagrange Hermiticity residual at "
            f"character {index}",
            component_name,
        )
        lagrange_hermiticity = (
            float(np.max(np.abs(lagrange_delta))) if lagrange_delta.size else 0.0
        )
        if lagrange_hermiticity > consistency:
            raise ValueError(
                f"χ-CCM-B {component_name} produced a non-Hermitian "
                f"{label_prefix}occupied Lagrange block at character {index}; "
                f"maximum residual is {lagrange_hermiticity:.3e}"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            weighted_q = (
                occupation_factor
                * occupied_q
                @ lagrange
                @ occupied_q.conj().T
            )
        _require_finite_fixed_w_array(
            weighted_q,
            f"{label_prefix}energy-weighted density at character {index}",
            component_name,
        )
        energy_weighted[index] = weighted_q
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_hermiticity_delta = (
            energy_weighted - energy_weighted.conj().transpose(0, 2, 1)
        )
    _require_finite_fixed_w_array(
        weighted_hermiticity_delta,
        f"{label_prefix}energy-weighted-density Hermiticity residual",
        component_name,
    )
    weighted_hermiticity = float(np.max(np.abs(weighted_hermiticity_delta)))
    if weighted_hermiticity > consistency:
        raise ValueError(
            f"χ-CCM-B {component_name} produced non-Hermitian "
            f"{label_prefix}W(q) blocks; maximum residual is "
            f"{weighted_hermiticity:.3e}"
        )
    for label, matrices in (
        (f"{label_prefix}occupied projector", projectors),
        (f"{label_prefix}energy-weighted density", energy_weighted),
    ):
        time_reversal_residual = _fixed_w_time_reversal_residual(
            matrices,
            character_labels,
            mesh,
        )
        if time_reversal_residual > consistency:
            raise NotImplementedError(
                f"χ-CCM-B {component_name} requires a time-reversal-consistent "
                f"{label}; maximum residual is {time_reversal_residual:.3e}"
            )
    return energy_weighted


def _compute_aiccm2026dev_b_fixed_input_energy_weighted_density_lattice(
    system: object,
    basis: object,
    result: object,
    overlap_k: Sequence[np.ndarray],
    channels: Sequence[
        tuple[
            str,
            Sequence[np.ndarray],
            Sequence[np.ndarray],
            float,
            bool,
        ]
    ],
    *,
    lattice_options: object,
    consistency_tolerance: float,
    imaginary_tolerance: float,
    spin_mode: str,
    component_name: str,
) -> "LatticeMatrixSet":
    """Shared fixed-input restricted/unrestricted ``W(g)`` audit core."""

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(system, status, component_name)
    _require_fixed_w_spin_method(status, spin_mode, component_name)
    for attribute in ("kpoints_frac", "kpoint_weights"):
        if not hasattr(result, attribute):
            raise TypeError(
                f"χ-CCM-B {component_name} requires result.{attribute} from "
                "the executed complete character net"
            )
    mesh = status.lattice_extension
    if mesh is None:  # Defensive: status validation currently forbids this.
        raise TypeError(f"χ-CCM-B {component_name} requires a result mesh")
    kpoints_frac = _real_character_metadata(
        getattr(result, "kpoints_frac"),
        "k-points",
        component_name,
    ).reshape(-1, 3)
    weights = _real_character_metadata(
        getattr(result, "kpoint_weights"),
        "character weights",
        component_name,
    ).reshape(-1)
    character_labels = _require_complete_gamma_character_mesh(
        kpoints_frac,
        weights,
        mesh,
        component_name,
    )
    mesh_array = np.asarray(mesh, dtype=np.int64)
    canonical_kpoints_frac = (
        np.mod(character_labels, mesh_array[None, :]) / mesh_array[None, :]
    )
    consistency = _positive_fixed_w_tolerance(
        consistency_tolerance,
        "fixed-input W consistency_tolerance",
    )
    imaginary = _positive_fixed_w_tolerance(
        imaginary_tolerance,
        "fixed-input W imaginary_tolerance",
    )
    nbf = int(getattr(basis, "nbasis"))
    count = int(kpoints_frac.shape[0])
    overlap_input = _fixed_w_square_character_stack(
        overlap_k,
        count=count,
        nbf=nbf,
        label="overlap",
        component_name=component_name,
    )
    channel_inputs: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for spin_label, occupied_blocks, fock_blocks, factor, allow_zero in channels:
        label_prefix = f"{spin_label} " if spin_label else ""
        fock = _fixed_w_square_character_stack(
            fock_blocks,
            count=count,
            nbf=nbf,
            label=f"{label_prefix}candidate variational Fock",
            component_name=component_name,
        )
        occupied = _fixed_w_occupied_character_stack(
            occupied_blocks,
            count=count,
            nbf=nbf,
            component_name=component_name,
            spin_label=spin_label,
            allow_zero_rank=allow_zero,
        )
        channel_inputs.append((spin_label, occupied, fock, factor))
    if spin_mode == "unrestricted" and not any(
        occupied.shape[2] for _, occupied, _, _ in channel_inputs
    ):
        raise ValueError(
            f"χ-CCM-B {component_name} requires at least one occupied spin orbital"
        )

    from ..._vibeqc_core import compute_overlap_lattice

    overlap_support = compute_overlap_lattice(basis, system, lattice_options)
    translations = [
        tuple(int(value) for value in np.asarray(cell.index, dtype=int).reshape(3))
        for cell in overlap_support.cells
    ]
    with np.errstate(over="ignore", invalid="ignore"):
        phases = np.exp(
            2j
            * np.pi
            * (np.asarray(translations, dtype=int) @ canonical_kpoints_frac.T)
        )
        support_overlap_k = np.einsum(
            "gk,gij->kij",
            phases,
            np.asarray(overlap_support.blocks, dtype=float),
            optimize=True,
        )
    _require_finite_fixed_w_array(phases, "support phases", component_name)
    _require_finite_fixed_w_array(
        support_overlap_k,
        "support overlap blocks",
        component_name,
    )
    _validate_fixed_w_matrix_symmetries(
        overlap_input,
        label="overlap",
        character_labels=character_labels,
        mesh=mesh,
        tolerance=consistency,
        component_name=component_name,
    )
    for spin_label, _, fock, _ in channel_inputs:
        label_prefix = f"{spin_label} " if spin_label else ""
        _validate_fixed_w_matrix_symmetries(
            fock,
            label=f"{label_prefix}candidate variational Fock",
            character_labels=character_labels,
            mesh=mesh,
            tolerance=consistency,
            component_name=component_name,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        support_delta = overlap_input - support_overlap_k
    _require_finite_fixed_w_array(
        support_delta,
        "support-overlap residual",
        component_name,
    )
    support_difference = float(np.max(np.abs(support_delta)))
    if support_difference > consistency:
        raise ValueError(
            f"χ-CCM-B {component_name} explicit overlap blocks do not match "
            "the requested lattice support; maximum residual is "
            f"{support_difference:.3e}"
        )

    energy_weighted = np.zeros((count, nbf, nbf), dtype=np.complex128)
    for spin_label, occupied, fock, factor in channel_inputs:
        channel_weighted = _fixed_w_character_channel(
            overlap_input,
            occupied,
            fock,
            occupation_factor=factor,
            spin_label=spin_label,
            character_labels=character_labels,
            mesh=mesh,
            consistency=consistency,
            component_name=component_name,
        )
        if spin_mode == "unrestricted":
            with np.errstate(over="ignore", invalid="ignore"):
                channel_blocks = inverse_bloch_transform(
                    channel_weighted,
                    canonical_kpoints_frac,
                    translations,
                    weights,
                )
            label_prefix = f"{spin_label} " if spin_label else ""
            _require_finite_fixed_w_array(
                channel_blocks,
                f"{label_prefix}inverse-character density blocks",
                component_name,
            )
            channel_imaginary_residual = (
                float(np.max(np.abs(channel_blocks.imag)))
                if channel_blocks.size
                else 0.0
            )
            if channel_imaginary_residual > imaginary:
                raise NotImplementedError(
                    f"χ-CCM-B {component_name} {label_prefix}channel would "
                    "discard a non-real inverse-character residue "
                    f"({channel_imaginary_residual:.3e})"
                )
        with np.errstate(over="ignore", invalid="ignore"):
            energy_weighted = energy_weighted + channel_weighted
        _require_finite_fixed_w_array(
            energy_weighted,
            "spin-summed energy-weighted density",
            component_name,
        )
    if spin_mode == "unrestricted":
        _validate_fixed_w_matrix_symmetries(
            energy_weighted,
            label="spin-summed energy-weighted density",
            character_labels=character_labels,
            mesh=mesh,
            tolerance=consistency,
            component_name=component_name,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        blocks = inverse_bloch_transform(
            energy_weighted,
            canonical_kpoints_frac,
            translations,
            weights,
        )
    _require_finite_fixed_w_array(
        blocks,
        "inverse-character density blocks",
        component_name,
    )
    imaginary_residual = float(np.max(np.abs(blocks.imag))) if blocks.size else 0.0
    if imaginary_residual > imaginary:
        raise NotImplementedError(
            f"χ-CCM-B {component_name} would discard a non-real inverse-"
            f"character residue ({imaginary_residual:.3e})"
        )
    for index, block in enumerate(blocks.real):
        overlap_support.set_block(
            index,
            np.ascontiguousarray(block, dtype=float),
        )
    return overlap_support


def compute_aiccm2026dev_b_fixed_input_restricted_energy_weighted_density_lattice(
    system: object,
    basis: object,
    result: object,
    overlap_k: Sequence[np.ndarray],
    occupied_coefficients_k: Sequence[np.ndarray],
    candidate_variational_fock_k: Sequence[np.ndarray],
    *,
    lattice_options: object,
    consistency_tolerance: float = 1.0e-9,
    imaginary_tolerance: float = 1.0e-7,
) -> "LatticeMatrixSet":
    """Construct a fixed-input restricted candidate ``W(g)`` for D80.

    For each declared finite character, this audit helper evaluates

    ``Lambda(q) = C_occ(q).conj().T @ F(q) @ C_occ(q)`` and
    ``W(q) = 2 C_occ(q) @ Lambda(q) @ C_occ(q).conj().T``.

    The explicit overlap blocks must equal the Bloch sum of the exact
    ``lattice_options`` support used by the returned
    :class:`LatticeMatrixSet`.  The helper validates occupied
    ``S(q)``-orthonormality, Hermiticity, and time reversal before applying
    the declared inverse-character transform.  It does not rebuild or attest
    a final SCF Fock, infer occupations, establish variational closure, or
    construct the stationary D85 object.  The output is only a fixed-input
    candidate suitable for the existing D80 scalar and derivative audits.
    """

    return _compute_aiccm2026dev_b_fixed_input_energy_weighted_density_lattice(
        system,
        basis,
        result,
        overlap_k,
        (("", occupied_coefficients_k, candidate_variational_fock_k, 2.0, False),),
        lattice_options=lattice_options,
        consistency_tolerance=consistency_tolerance,
        imaginary_tolerance=imaginary_tolerance,
        spin_mode="restricted",
        component_name="fixed-input restricted energy-weighted-density audit",
    )


def compute_aiccm2026dev_b_fixed_input_unrestricted_energy_weighted_density_lattice(
    system: object,
    basis: object,
    result: object,
    overlap_k: Sequence[np.ndarray],
    occupied_coefficients_alpha_k: Sequence[np.ndarray],
    occupied_coefficients_beta_k: Sequence[np.ndarray],
    candidate_variational_fock_alpha_k: Sequence[np.ndarray],
    candidate_variational_fock_beta_k: Sequence[np.ndarray],
    *,
    lattice_options: object,
    consistency_tolerance: float = 1.0e-9,
    imaginary_tolerance: float = 1.0e-7,
) -> "LatticeMatrixSet":
    """Construct a fixed-input unrestricted candidate ``W(g)`` for D80.

    For each spin ``s`` and declared finite character, this audit helper
    evaluates ``Lambda_s(q) = C_s(q).conj().T @ F_s(q) @ C_s(q)`` and returns
    the inverse-character transform of
    ``W(q) = sum_s C_s(q) @ Lambda_s(q) @ C_s(q).conj().T``.  Alpha and beta
    inputs, including each inverse-transform reality residue, are validated
    independently before their spin sum is formed.  Coefficient columns are
    unit-occupied; a zero-rank spin channel is valid, but both channels cannot
    be empty.

    As in the restricted companion, every input is explicit.  This helper
    does not infer occupations, read stored orbital energies, rebuild or
    attest a final SCF Fock, establish backend variational closure, or create
    the stationary D85 object.  Only the spin-summed real-space candidate
    needed by the spin-independent D80 overlap audit is returned.
    """

    return _compute_aiccm2026dev_b_fixed_input_energy_weighted_density_lattice(
        system,
        basis,
        result,
        overlap_k,
        (
            (
                "alpha",
                occupied_coefficients_alpha_k,
                candidate_variational_fock_alpha_k,
                1.0,
                True,
            ),
            (
                "beta",
                occupied_coefficients_beta_k,
                candidate_variational_fock_beta_k,
                1.0,
                True,
            ),
        ),
        lattice_options=lattice_options,
        consistency_tolerance=consistency_tolerance,
        imaginary_tolerance=imaginary_tolerance,
        spin_mode="unrestricted",
        component_name="fixed-input unrestricted energy-weighted-density audit",
    )


def _positive_seam_operator_coefficient(value: object) -> float:
    """Return one explicit positive finite active-seam coefficient."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            "χ-CCM-B BvK exchange seam operator coefficient eta "
            "must be a positive finite scalar"
        )
    try:
        eta = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "χ-CCM-B BvK exchange seam operator coefficient eta "
            "must be a positive finite scalar"
        ) from exc
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError(
            "χ-CCM-B BvK exchange seam operator coefficient eta "
            "must be positive and finite"
        )
    return eta


def _positive_matrix_tolerance(value: object) -> float:
    """Return one positive finite matrix-consistency tolerance."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError("χ-CCM-B seam imaginary_tolerance must be positive")
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "χ-CCM-B seam imaginary_tolerance must be a positive finite scalar"
        ) from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "χ-CCM-B seam imaginary_tolerance must be positive and finite"
        )
    return tolerance


def _validated_bvk_exchange_seam_density(
    density_k: Sequence[np.ndarray],
    *,
    basis: object,
    kpoints_frac: np.ndarray,
    mesh: tuple[int, int, int],
    tolerance: float,
    component_name: str,
    density_label: str,
) -> np.ndarray:
    """Return one independently validated character-density stack."""

    try:
        density_blocks = tuple(
            np.asarray(block, dtype=np.complex128) for block in density_k
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"χ-CCM-B {component_name} requires a finite {density_label} "
            "density matrix at every character"
        ) from exc
    if len(density_blocks) != kpoints_frac.shape[0]:
        count_label = "density" if density_label == "restricted" else (
            f"{density_label} density"
        )
        raise ValueError(
            f"χ-CCM-B {component_name} requires matching {count_label}, "
            "k-point, and weight counts"
        )
    _require_density_k_ao_shape(
        density_blocks,
        basis,
        component_name,
        density_label,
    )
    density = np.stack(density_blocks)
    if not np.all(np.isfinite(density.real)) or not np.all(
        np.isfinite(density.imag)
    ):
        raise ValueError(
            f"χ-CCM-B {component_name} requires finite {density_label} "
            "density blocks"
        )
    hermiticity_residual = float(
        np.max(np.abs(density - density.conj().transpose(0, 2, 1)))
    )
    if hermiticity_residual > tolerance:
        raise ValueError(
            f"χ-CCM-B {component_name} requires Hermitian {density_label} "
            f"density blocks; maximum residual is {hermiticity_residual:.3e}"
        )

    mesh_array = np.asarray(mesh, dtype=int)
    labels = np.rint(kpoints_frac * mesh_array[None, :]).astype(int)
    residues = [
        tuple(int(value) for value in np.mod(label, mesh_array))
        for label in labels
    ]
    residue_to_index = {residue: index for index, residue in enumerate(residues)}
    time_reversal_residual = 0.0
    for index, residue in enumerate(residues):
        opposite = tuple(
            int(value) for value in np.mod(-np.asarray(residue), mesh_array)
        )
        opposite_index = residue_to_index[opposite]
        time_reversal_residual = max(
            time_reversal_residual,
            float(np.max(np.abs(density[opposite_index] - density[index].conj()))),
        )
    if time_reversal_residual > tolerance:
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires a time-reversal-consistent "
            f"{density_label} character density; maximum residual is "
            f"{time_reversal_residual:.3e}"
        )
    return density


def _bvk_exchange_seam_support_inputs(
    system: object,
    basis: object,
    result: object,
    *,
    operator_coefficient_eta: float,
    lattice_options: object,
    imaginary_tolerance: float,
    component_name: str,
    spin_mode: str,
) -> tuple[
    AICCM2026DevBGradientStatus,
    float,
    float,
    np.ndarray,
    np.ndarray,
    object,
    list[tuple[int, int, int]],
    np.ndarray,
]:
    """Validate the shared active-seam convention and overlap support."""

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(system, status, component_name)
    if status.exchange_q0_applicability != "active":
        raise NotImplementedError(
            f"χ-CCM-B {component_name} requires "
            "exchange_q0_applicability='active'; the declared full-range "
            f"seam is {status.exchange_q0_applicability!r}"
        )
    method = status.electronic_method.upper()
    if spin_mode == "restricted":
        if method != "RHF" and not method.startswith("RKS/"):
            raise NotImplementedError(
                f"χ-CCM-B {component_name} currently accepts only restricted "
                f"RHF/RKS records, found {status.electronic_method!r}"
            )
    elif spin_mode == "unrestricted":
        uks_functional = (
            method.removeprefix("UKS/") if method.startswith("UKS/") else ""
        )
        is_uks = (
            bool(uks_functional.strip())
            and uks_functional == uks_functional.strip()
            and "/" not in uks_functional
        )
        if method != "UHF" and not is_uks:
            raise NotImplementedError(
                f"χ-CCM-B {component_name} currently accepts only unrestricted "
                f"UHF or UKS/<functional> records, found "
                f"{status.electronic_method!r}"
            )
        if any(
            getattr(result, attribute, None) is None
            for attribute in ("density_alpha", "density_beta")
        ):
            raise TypeError(
                f"χ-CCM-B {component_name} requires an unrestricted result "
                "carrying density_alpha and density_beta"
            )
    else:  # pragma: no cover - private programming invariant
        raise ValueError(f"unknown χ-CCM-B seam spin mode {spin_mode!r}")

    eta = _positive_seam_operator_coefficient(operator_coefficient_eta)
    tolerance = _positive_matrix_tolerance(imaginary_tolerance)
    for attribute in ("kpoints_frac", "kpoint_weights"):
        if not hasattr(result, attribute):
            raise TypeError(
                f"χ-CCM-B {component_name} requires result.{attribute} from "
                "the executed complete character net"
            )
    kpoints_frac = _real_character_metadata(
        getattr(result, "kpoints_frac"),
        "k-points",
        component_name,
    ).reshape(-1, 3)
    weights = _real_character_metadata(
        getattr(result, "kpoint_weights"),
        "character weights",
        component_name,
    ).reshape(-1)
    mesh = status.lattice_extension
    if mesh is None:  # Defensive: status validation currently forbids this.
        raise TypeError(f"χ-CCM-B {component_name} requires a result mesh")
    _require_complete_gamma_character_mesh(
        kpoints_frac,
        weights,
        mesh,
        component_name,
    )
    from ..._vibeqc_core import compute_overlap_lattice

    overlap = compute_overlap_lattice(basis, system, lattice_options)
    translations = [
        tuple(int(value) for value in np.asarray(cell.index, dtype=int).reshape(3))
        for cell in overlap.cells
    ]
    phases = np.exp(
        2j
        * np.pi
        * (np.asarray(translations, dtype=int) @ kpoints_frac.T)
    )
    overlap_k = np.einsum(
        "gk,gij->kij",
        phases,
        np.asarray(overlap.blocks, dtype=float),
        optimize=True,
    )
    overlap_hermiticity_residual = float(
        np.max(np.abs(overlap_k - overlap_k.conj().transpose(0, 2, 1)))
    )
    if overlap_hermiticity_residual > tolerance:
        raise NotImplementedError(
            f"χ-CCM-B {component_name} overlap support does not produce "
            "Hermitian character blocks; maximum residual is "
            f"{overlap_hermiticity_residual:.3e}"
        )
    return (
        status,
        eta,
        tolerance,
        kpoints_frac,
        weights,
        overlap,
        translations,
        overlap_k,
    )


def _restricted_bvk_exchange_seam_inputs(
    system: object,
    basis: object,
    result: object,
    density_k: Sequence[np.ndarray],
    *,
    operator_coefficient_eta: float,
    lattice_options: object,
    imaginary_tolerance: float,
    component_name: str,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    object,
    list[tuple[int, int, int]],
    np.ndarray,
]:
    """Validate and construct fixed-density restricted seam inputs."""

    (
        status,
        eta,
        tolerance,
        kpoints_frac,
        weights,
        overlap,
        translations,
        overlap_k,
    ) = _bvk_exchange_seam_support_inputs(
        system,
        basis,
        result,
        operator_coefficient_eta=operator_coefficient_eta,
        lattice_options=lattice_options,
        imaginary_tolerance=imaginary_tolerance,
        component_name=component_name,
        spin_mode="restricted",
    )
    mesh = status.lattice_extension
    if mesh is None:  # Defensive: shared support validation forbids this.
        raise TypeError(f"χ-CCM-B {component_name} requires a result mesh")
    density = _validated_bvk_exchange_seam_density(
        density_k,
        basis=basis,
        kpoints_frac=kpoints_frac,
        mesh=mesh,
        tolerance=tolerance,
        component_name=component_name,
        density_label="restricted",
    )
    return (
        eta,
        density,
        kpoints_frac,
        weights,
        overlap,
        translations,
        overlap_k,
    )


def _unrestricted_bvk_exchange_seam_inputs(
    system: object,
    basis: object,
    result: object,
    density_alpha_k: Sequence[np.ndarray],
    density_beta_k: Sequence[np.ndarray],
    *,
    operator_coefficient_eta: float,
    lattice_options: object,
    imaginary_tolerance: float,
    component_name: str,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    object,
    list[tuple[int, int, int]],
    np.ndarray,
]:
    """Validate both spins and construct fixed-density unrestricted inputs."""

    (
        status,
        eta,
        tolerance,
        kpoints_frac,
        weights,
        overlap,
        translations,
        overlap_k,
    ) = _bvk_exchange_seam_support_inputs(
        system,
        basis,
        result,
        operator_coefficient_eta=operator_coefficient_eta,
        lattice_options=lattice_options,
        imaginary_tolerance=imaginary_tolerance,
        component_name=component_name,
        spin_mode="unrestricted",
    )
    mesh = status.lattice_extension
    if mesh is None:  # Defensive: shared support validation forbids this.
        raise TypeError(f"χ-CCM-B {component_name} requires a result mesh")
    density_alpha = _validated_bvk_exchange_seam_density(
        density_alpha_k,
        basis=basis,
        kpoints_frac=kpoints_frac,
        mesh=mesh,
        tolerance=tolerance,
        component_name=component_name,
        density_label="alpha",
    )
    density_beta = _validated_bvk_exchange_seam_density(
        density_beta_k,
        basis=basis,
        kpoints_frac=kpoints_frac,
        mesh=mesh,
        tolerance=tolerance,
        component_name=component_name,
        density_label="beta",
    )
    return (
        eta,
        density_alpha,
        density_beta,
        kpoints_frac,
        weights,
        overlap,
        translations,
        overlap_k,
    )


def compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_energy(
    system: object,
    basis: object,
    result: object,
    density_k: Sequence[np.ndarray],
    *,
    operator_coefficient_eta: float,
    lattice_options: object,
    imaginary_tolerance: float = 1.0e-7,
) -> float:
    """Return the fixed-density restricted active BvK seam energy.

    For the caller-supplied spin-summed restricted character density, this
    evaluates
    ``-eta/4 sum_q w_q Tr[D(q) S(q) D(q) S(q)]``.  ``eta`` is the complete
    positive finite effective seam coefficient after the full-range exchange
    fraction: the BvK Madelung coefficient for RHF, or that coefficient
    multiplied by the executed full-range hybrid fraction.  It is explicit
    because current
    χ results record applicability but do not yet bind the numerical
    coefficient.  The density and overlap support are fixed audit inputs, not
    an attested reconstruction of the final SCF state.
    """

    component_name = "fixed-density restricted BvK exchange seam energy component"
    eta, density, _, weights, _, _, overlap_k = (
        _restricted_bvk_exchange_seam_inputs(
            system,
            basis,
            result,
            density_k,
            operator_coefficient_eta=operator_coefficient_eta,
            lattice_options=lattice_options,
            imaginary_tolerance=imaginary_tolerance,
            component_name=component_name,
        )
    )
    traces = np.asarray(
        [
            np.trace(density_q @ overlap_q @ density_q @ overlap_q)
            for density_q, overlap_q in zip(density, overlap_k)
        ],
        dtype=np.complex128,
    )
    imaginary_residual = float(np.max(np.abs(traces.imag)))
    if imaginary_residual > float(imaginary_tolerance):
        raise NotImplementedError(
            f"χ-CCM-B {component_name} has a non-real character contraction "
            f"residue ({imaginary_residual:.3e})"
        )
    return -0.25 * eta * float(np.dot(weights, traces.real))


def compute_aiccm2026dev_b_fixed_density_restricted_bvk_exchange_seam_gradient(
    system: object,
    basis: object,
    result: object,
    density_k: Sequence[np.ndarray],
    *,
    operator_coefficient_eta: float,
    lattice_options: object,
    imaginary_tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Return the restricted active BvK seam AO-centre derivative.

    This differentiates only the matching fixed-density seam scalar, with
    ``density_k``, ``eta``, and the primitive lattice held fixed.  The
    derivative is
    ``-eta/2 sum_q w_q Tr[D(q) S(q) D(q) dS(q)]``.  It is a component audit,
    not a total force or stationary-SCF wrapper.  The unrestricted component
    has a separate spin-resolved helper below; HSE screened exchange,
    density/orbital response, and support provenance remain outside this API.
    """

    component_name = (
        "fixed-density restricted BvK exchange seam gradient component"
    )
    (
        eta,
        density,
        kpoints_frac,
        weights,
        overlap_weight,
        translations,
        overlap_k,
    ) = _restricted_bvk_exchange_seam_inputs(
        system,
        basis,
        result,
        density_k,
        operator_coefficient_eta=operator_coefficient_eta,
        lattice_options=lattice_options,
        imaginary_tolerance=imaginary_tolerance,
        component_name=component_name,
    )
    weighted_overlap_k = np.asarray(
        [
            0.5 * eta * (density_q @ overlap_q @ density_q)
            for density_q, overlap_q in zip(density, overlap_k)
        ],
        dtype=np.complex128,
    )
    weighted_overlap_blocks = inverse_bloch_transform(
        weighted_overlap_k,
        kpoints_frac,
        translations,
        weights,
    )
    imaginary_residual = (
        float(np.max(np.abs(weighted_overlap_blocks.imag)))
        if weighted_overlap_blocks.size
        else 0.0
    )
    if imaginary_residual > float(imaginary_tolerance):
        raise NotImplementedError(
            f"χ-CCM-B {component_name} would discard a non-real finite-torus "
            f"operator residue ({imaginary_residual:.3e})"
        )
    for index, block in enumerate(weighted_overlap_blocks.real):
        overlap_weight.set_block(index, np.ascontiguousarray(block, dtype=float))

    from ..._vibeqc_core import overlap_lattice_gradient_contribution

    return np.asarray(
        overlap_lattice_gradient_contribution(
            basis,
            system,
            overlap_weight,
            lattice_options,
        ),
        dtype=float,
    )


def compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_energy(
    system: object,
    basis: object,
    result: object,
    density_alpha_k: Sequence[np.ndarray],
    density_beta_k: Sequence[np.ndarray],
    *,
    operator_coefficient_eta: float,
    lattice_options: object,
    imaginary_tolerance: float = 1.0e-7,
) -> float:
    """Return the fixed-density unrestricted active BvK seam energy.

    For caller-supplied per-spin character densities this evaluates
    ``-eta/2 sum_q w_q sum_s Tr[D_s(q) S(q) D_s(q) S(q)]``.  There are no
    alpha-beta exchange contractions.  ``eta`` is an explicit positive audit
    coefficient because applicability records only that a full-range arm is
    nonzero; they do not bind its numerical coefficient or sign.  The supplied
    spin densities, basis, and overlap support are likewise not attested to the
    final SCF state.
    """

    component_name = (
        "fixed-density unrestricted BvK exchange seam energy component"
    )
    (
        eta,
        density_alpha,
        density_beta,
        _,
        weights,
        _,
        _,
        overlap_k,
    ) = _unrestricted_bvk_exchange_seam_inputs(
        system,
        basis,
        result,
        density_alpha_k,
        density_beta_k,
        operator_coefficient_eta=operator_coefficient_eta,
        lattice_options=lattice_options,
        imaginary_tolerance=imaginary_tolerance,
        component_name=component_name,
    )
    traces_by_spin = np.asarray(
        [
            [
                np.trace(density_q @ overlap_q @ density_q @ overlap_q)
                for density_q, overlap_q in zip(density, overlap_k)
            ]
            for density in (density_alpha, density_beta)
        ],
        dtype=np.complex128,
    )
    imaginary_residual = float(np.max(np.abs(traces_by_spin.imag)))
    if imaginary_residual > float(imaginary_tolerance):
        raise NotImplementedError(
            f"χ-CCM-B {component_name} has a non-real per-spin contraction "
            f"residue ({imaginary_residual:.3e})"
        )
    traces = traces_by_spin.real.sum(axis=0)
    return -0.5 * eta * float(np.dot(weights, traces))


def compute_aiccm2026dev_b_fixed_density_unrestricted_bvk_exchange_seam_gradient(
    system: object,
    basis: object,
    result: object,
    density_alpha_k: Sequence[np.ndarray],
    density_beta_k: Sequence[np.ndarray],
    *,
    operator_coefficient_eta: float,
    lattice_options: object,
    imaginary_tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Return the unrestricted active BvK seam AO-centre derivative.

    Both spin densities, ``eta``, and the primitive lattice are fixed.  The
    derivative is
    ``-eta sum_q w_q sum_s Tr[D_s(q) S(q) D_s(q) dS(q)]``.  This independently
    checkable component neither reconstructs the executed unrestricted state
    nor supplies screened exchange, response, or a total χ-CCM-B force.
    """

    component_name = (
        "fixed-density unrestricted BvK exchange seam gradient component"
    )
    (
        eta,
        density_alpha,
        density_beta,
        kpoints_frac,
        weights,
        overlap_weight,
        translations,
        overlap_k,
    ) = _unrestricted_bvk_exchange_seam_inputs(
        system,
        basis,
        result,
        density_alpha_k,
        density_beta_k,
        operator_coefficient_eta=operator_coefficient_eta,
        lattice_options=lattice_options,
        imaginary_tolerance=imaginary_tolerance,
        component_name=component_name,
    )
    weighted_overlap_blocks_by_spin = [
        inverse_bloch_transform(
            np.asarray(
                [
                    eta * (density_q @ overlap_q @ density_q)
                    for density_q, overlap_q in zip(density, overlap_k)
                ],
                dtype=np.complex128,
            ),
            kpoints_frac,
            translations,
            weights,
        )
        for density in (density_alpha, density_beta)
    ]
    imaginary_residual = max(
        float(np.max(np.abs(blocks.imag))) if blocks.size else 0.0
        for blocks in weighted_overlap_blocks_by_spin
    )
    if imaginary_residual > float(imaginary_tolerance):
        raise NotImplementedError(
            f"χ-CCM-B {component_name} would discard a non-real per-spin "
            f"finite-torus operator residue ({imaginary_residual:.3e})"
        )
    weighted_overlap_blocks = (
        weighted_overlap_blocks_by_spin[0] + weighted_overlap_blocks_by_spin[1]
    )
    for index, block in enumerate(weighted_overlap_blocks.real):
        overlap_weight.set_block(index, np.ascontiguousarray(block, dtype=float))

    from ..._vibeqc_core import overlap_lattice_gradient_contribution

    return np.asarray(
        overlap_lattice_gradient_contribution(
            basis,
            system,
            overlap_weight,
            lattice_options,
        ),
        dtype=float,
    )


def compute_aiccm2026dev_b_ewald_nuclear_gradient(
    system: object,
    result: object,
    *,
    ewald_options: object | None = None,
) -> np.ndarray:
    """Return the 3D Ewald nuclear-repulsion derivative component.

    This is a deliberately narrow, independently checkable term in the
    χ-CCM-B gradient derivation:
    ``∂E_nn^Ewald/∂R_A`` for the declared ``3d-periodic-g0`` finite-torus
    Hamiltonian.  It is **not** the total SCF or post-HF gradient; the public
    total-gradient entry points below still fail closed until the
    electron-nuclear, Pulay, exchange-seam, RI/RIJCOSX, and response terms are
    implemented and assembled.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "Ewald nuclear-gradient component",
    )
    from ..._vibeqc_core import EwaldOptions, ewald_nuclear_repulsion_gradient

    options = ewald_options if ewald_options is not None else EwaldOptions()
    return np.asarray(
        ewald_nuclear_repulsion_gradient(system, options),
        dtype=float,
    )


def compute_aiccm2026dev_b_ewald_electron_nuclear_gradient(
    system: object,
    basis: object,
    result: object,
    density: object,
    *,
    lattice_options: object,
    ewald_alpha: float | None = None,
    precision: float = 1.0e-8,
) -> np.ndarray:
    """Return the fixed-density 3D Ewald electron-nuclear component.

    The returned matrix is the derivative of
    ``sum_g Tr[D(g) V_ne^Ewald(g)]`` at fixed real-torus density blocks
    ``D(g)``.  It validates the χ-CCM-B finite-torus convention carried by
    ``result`` and differentiates the Ewald electron-nuclear operator for the
    caller-supplied lattice-sum options.  It is still only a component term:
    the total analytic force remains unavailable until the energy-weighted
    Pulay adjoint, exchange seam, RI/RIJCOSX response, and post-HF response
    pieces are assembled and tested.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "Ewald electron-nuclear gradient component",
    )
    alpha = _ewald_alpha_from_result(
        result,
        ewald_alpha,
        "Ewald electron-nuclear gradient component",
    )
    from ..._vibeqc_core import compute_overlap_lattice
    from ...bipole_gradient import _v_ne_ewald_gradient

    _require_ewald_lattice_options(
        lattice_options,
        "Ewald electron-nuclear gradient component",
    )
    overlap = compute_overlap_lattice(basis, system, lattice_options)
    _require_matching_density_cell_list(
        density,
        overlap,
        "Ewald electron-nuclear gradient component",
    )
    return np.asarray(
        _v_ne_ewald_gradient(
            system,
            basis,
            density,
            lattice_options,
            alpha,
            precision=float(precision),
        ),
        dtype=float,
    )


def compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
    system: object,
    basis: object,
    result: object,
    density: object,
    *,
    lattice_options: object,
    ewald_alpha: float | None = None,
    precision: float = 1.0e-8,
) -> AICCM2026DevBEwaldElectrostaticEnergyComponents:
    """Return fixed-density Ewald nuclear and electron-nuclear energies.

    This is the scalar companion to
    :func:`compute_aiccm2026dev_b_ewald_electrostatic_gradient_components`.
    It evaluates only ``E_nn^Ewald + sum_g Tr[D(g) V_ne^Ewald(g)]`` for the
    caller-supplied real-torus density at the declared χ-CCM-B finite-torus
    convention.  It is useful for central-difference audits of the component
    gradients, but it is not a total SCF energy response.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "Ewald electrostatic energy component bundle",
    )
    alpha = _ewald_alpha_from_result(
        result,
        ewald_alpha,
        "Ewald electrostatic energy component bundle",
    )
    _require_ewald_lattice_options(
        lattice_options,
        "Ewald electrostatic energy component bundle",
    )

    from ..._vibeqc_core import compute_overlap_lattice, ewald_nuclear_repulsion
    from ...bipole_gradient import _matching_ewald_options
    from ...pbc_bipole_common import (
        _compute_nuclear_lattice_ewald_reciprocal_ft,
        _lattice_contract,
    )

    ewald_options = _matching_ewald_options(system, lattice_options, alpha)
    if ewald_options is None:
        raise NotImplementedError(
            "χ-CCM-B Ewald electrostatic energy component bundle requires "
            "3D Ewald nuclear options"
        )
    overlap = compute_overlap_lattice(basis, system, lattice_options)
    _require_matching_density_cell_list(
        density,
        overlap,
        "Ewald electrostatic energy component bundle",
    )
    v_ne, _ = _compute_nuclear_lattice_ewald_reciprocal_ft(
        basis,
        system,
        lattice_options,
        ewald_options,
        overlap,
        precision=float(precision),
        K_max=float(ewald_options.recip_cutoff_bohr_inv),
    )
    nuclear = float(ewald_nuclear_repulsion(system, ewald_options))
    electron_nuclear = float(
        _lattice_contract(
            density,
            v_ne,
            operator_name="Ewald electrostatic energy component bundle",
        )
    )
    total = nuclear + electron_nuclear
    return AICCM2026DevBEwaldElectrostaticEnergyComponents(
        nuclear=nuclear,
        electron_nuclear=electron_nuclear,
        total_fixed_density=total,
        ewald_alpha_bohr_inv=alpha,
        nuclear_cutoff_bohr=float(getattr(lattice_options, "nuclear_cutoff_bohr")),
        reciprocal_cutoff_bohr_inv=float(ewald_options.recip_cutoff_bohr_inv),
        precision=float(precision),
        finite_torus_convention=status.finite_torus_convention,
    )


def compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(
    system: object,
    basis: object,
    result: object,
    density: object,
    *,
    lattice_options: object,
    ewald_alpha: float | None = None,
    precision: float = 1.0e-8,
) -> AICCM2026DevBEwaldElectrostaticGradientComponents:
    """Return fixed-density Ewald nuclear and electron-nuclear components.

    This helper assembles the two independently checked 3D Ewald
    electrostatic derivative pieces already available to χ-CCM-B:
    ``∂E_nn^Ewald/∂R_A`` and
    ``∂ sum_g Tr[D(g) V_ne^Ewald(g)] / ∂R_A`` for the caller-supplied
    real-torus density.  The matching nuclear Ewald options are rebuilt from
    ``lattice_options`` and the SCF-recorded Ewald alpha, so the returned
    metadata records the exact alpha and cutoffs used for the component
    bundle.  It is not a total analytic force.
    """

    status = aiccm2026dev_b_gradient_status(result)
    _require_3d_ewald_component(
        system,
        status,
        "Ewald electrostatic gradient component bundle",
    )
    alpha = _ewald_alpha_from_result(
        result,
        ewald_alpha,
        "Ewald electrostatic gradient component bundle",
    )
    _require_ewald_lattice_options(
        lattice_options,
        "Ewald electrostatic gradient component bundle",
    )

    from ...bipole_gradient import _matching_ewald_options
    from ..._vibeqc_core import compute_overlap_lattice

    ewald_options = _matching_ewald_options(system, lattice_options, alpha)
    if ewald_options is None:
        raise NotImplementedError(
            "χ-CCM-B Ewald electrostatic gradient component bundle requires "
            "3D Ewald nuclear options"
        )
    overlap = compute_overlap_lattice(basis, system, lattice_options)
    _require_matching_density_cell_list(
        density,
        overlap,
        "Ewald electrostatic gradient component bundle",
    )
    nuclear = compute_aiccm2026dev_b_ewald_nuclear_gradient(
        system,
        result,
        ewald_options=ewald_options,
    )
    electron_nuclear = compute_aiccm2026dev_b_ewald_electron_nuclear_gradient(
        system,
        basis,
        result,
        density,
        lattice_options=lattice_options,
        ewald_alpha=alpha,
        precision=precision,
    )
    total = np.asarray(nuclear, dtype=float) + np.asarray(
        electron_nuclear,
        dtype=float,
    )
    return AICCM2026DevBEwaldElectrostaticGradientComponents(
        nuclear=np.asarray(nuclear, dtype=float),
        electron_nuclear=np.asarray(electron_nuclear, dtype=float),
        total_fixed_density=total,
        ewald_alpha_bohr_inv=alpha,
        nuclear_cutoff_bohr=float(getattr(lattice_options, "nuclear_cutoff_bohr")),
        reciprocal_cutoff_bohr_inv=float(ewald_options.recip_cutoff_bohr_inv),
        precision=float(precision),
        finite_torus_convention=status.finite_torus_convention,
    )


def compute_aiccm2026dev_b_scf_ewald_electrostatic_energy_components(
    system: object,
    basis: object,
    result: object,
    *,
    lattice_options: object,
    ewald_alpha: float | None = None,
    precision: float = 1.0e-8,
    imaginary_tolerance: float = 1.0e-7,
) -> AICCM2026DevBEwaldElectrostaticEnergyComponents:
    """Return Ewald electrostatic energies at the stored SCF density.

    This helper folds the stored χ-CCM-B SCF density onto the requested
    lattice-cell list and evaluates only the fixed-density electrostatic value
    ``E_nn^Ewald + sum_g Tr[D_scf(g) V_ne^Ewald(g)]``.  It intentionally does
    not add Pulay/adjoint, exchange-seam, RI/RIJCOSX, or response terms.
    """

    density = compute_aiccm2026dev_b_scf_density_lattice(
        system,
        basis,
        result,
        lattice_options=lattice_options,
        imaginary_tolerance=imaginary_tolerance,
    )
    return compute_aiccm2026dev_b_ewald_electrostatic_energy_components(
        system,
        basis,
        result,
        density,
        lattice_options=lattice_options,
        ewald_alpha=ewald_alpha,
        precision=precision,
    )


def compute_aiccm2026dev_b_scf_ewald_electrostatic_gradient_components(
    system: object,
    basis: object,
    result: object,
    *,
    lattice_options: object,
    ewald_alpha: float | None = None,
    precision: float = 1.0e-8,
    imaginary_tolerance: float = 1.0e-7,
) -> AICCM2026DevBEwaldElectrostaticGradientComponents:
    """Return Ewald electrostatic components at the stored SCF density.

    This helper combines the B-owned SCF density inverse-Bloch fold with the
    fixed-density electrostatic component bundle.  It therefore differentiates
    only ``E_nn^Ewald + sum_g Tr[D_scf(g) V_ne^Ewald(g)]`` at the stored
    density.  It is still not a total SCF force: Pulay/adjoint, exchange-seam,
    RI/RIJCOSX response, and post-HF response terms remain absent.
    """

    density = compute_aiccm2026dev_b_scf_density_lattice(
        system,
        basis,
        result,
        lattice_options=lattice_options,
        imaginary_tolerance=imaginary_tolerance,
    )
    return compute_aiccm2026dev_b_ewald_electrostatic_gradient_components(
        system,
        basis,
        result,
        density,
        lattice_options=lattice_options,
        ewald_alpha=ewald_alpha,
        precision=precision,
    )


def _unavailable_message(status: AICCM2026DevBGradientStatus) -> str:
    blocked = "; ".join(status.blocked_terms)
    return (
        "χ-CCM / aiccm2026dev-b analytic nuclear gradients are not "
        "implemented for the declared finite-torus Hamiltonian "
        f"(coulomb_kernel={status.coulomb_kernel!r}, "
        f"exchange_q0={status.exchange_q0!r}, "
        f"boundary_model={status.boundary_model!r}). "
        "Neither the union-and-weight Γ-CCM gradient nor the neutral "
        "fitted-torus real-Gamma/GDF control gradient is reused because "
        "neither supplies the declared B-owned χ gradient contract. "
        f"Open χ-CCM gradient terms: {blocked}. See "
        "docs/user_guide/aiccm2026dev_b.md."
    )


def compute_aiccm2026dev_b_gradient(*args: object, **_kwargs: object):
    """Fail closed for χ-CCM analytic nuclear gradients.

    The first positional argument must be the χ-CCM SCF or post-HF result.  The
    loose signature mirrors the molecular and periodic gradient entry points,
    so callers can probe support without learning an intermediate API while
    still receiving a precise Hamiltonian-convention diagnostic.
    """

    if not args:
        raise TypeError("compute_aiccm2026dev_b_gradient requires a result object")
    status = aiccm2026dev_b_gradient_status(args[0])
    raise NotImplementedError(_unavailable_message(status))


def run_aiccm2026dev_b_gradient(*args: object, **kwargs: object):
    """Alias for :func:`compute_aiccm2026dev_b_gradient`."""

    return compute_aiccm2026dev_b_gradient(*args, **kwargs)
