"""Momentum-conserving post-HF methods for AICCM2026DEV-B.

The routines in this module consume the same finite-group RI Hamiltonian as
``vibeqc.periodic.chi.scf``. They do not use the historical CCM tensors or the
legacy Gamma-supercell HF reference.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from numbers import Real
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from ..._vibeqc_core import (
    BasisSet,
    LatticeSumOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    aiccm2026dev_b_3index_mo_transform_complex,
    aiccm2026dev_b_home_auxiliary_3index_mo_transform,
    aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex,
    aiccm2026dev_b_lpq_to_real_home_auxiliary,
    aiccm2026dev_b_lpq_to_real_supercell,
    aiccm2026dev_b_matrix_to_real_supercell,
    aiccm2026dev_b_mp2_energy_from_lov,
    aiccm2026dev_b_real_mp2_energy_from_lov,
)
from ...aux_basis import (
    build_lpq_bloch_native_fft,
    default_aux_for,
    make_aux_basis_set,
    make_modrho_aux_basis,
)
from .scf import (
    AICCM2026DevBDiagnostics,
    AICCM2026DevBFiniteTorusConvention,
    _normalise_mesh,
    _resolve_lattice_extension,
    cyclic_gamma_mesh,
    run_aiccm2026dev_b_rhf,
    run_aiccm2026dev_b_uhf,
)

__all__ = [
    "AICCM2026DevBLocalCorrelationSpace",
    "AICCM2026DevBDLPNOCCSDResult",
    "AICCM2026DevBDLPNOMP2Result",
    "AICCM2026DevBMP2Result",
    "AICCM2026DevBUCCSDResult",
    "AICCM2026DevBUMP2Result",
    "run_aiccm2026dev_b_dlpno_ccsd",
    "run_aiccm2026dev_b_dlpno_ccsd_t",
    "run_aiccm2026dev_b_dlpno_mp2",
    "run_aiccm2026dev_b_dlpno_uccsd",
    "run_aiccm2026dev_b_dlpno_uccsd_t",
    "run_aiccm2026dev_b_dlpno_ump2",
    "run_aiccm2026dev_b_ccsd",
    "run_aiccm2026dev_b_ccsd_t",
    "run_aiccm2026dev_b_mp2",
    "run_aiccm2026dev_b_uccsd",
    "run_aiccm2026dev_b_uccsd_t",
    "run_aiccm2026dev_b_ump2",
]


@dataclass(frozen=True)
class AICCM2026DevBLocalCorrelationSpace:
    """Validated finite-torus PAO and occupied-pair bookkeeping.

    ``representative_pair_acceleration`` stays false until the pair-amplitude
    scatter has an energy-parity gate.  The orbit count is therefore a
    measured reduction opportunity, not a timing claim.
    """

    pao_rank: int
    canonical_virtual_rank: int
    pao_discarded_rank: int
    pao_orthonormality_error: float
    pao_occupied_leakage_error: float
    n_occupied: int
    n_pairs: int
    n_translation_unique_pairs: int
    translation_reduction_factor: float
    translation_covariance_error: float
    translation_orbits_validated: bool
    representative_pair_acceleration: bool = False


class _FiniteTorusConventionMixin:
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention

    @property
    def ccm_approach(self) -> str:
        return self.finite_torus_convention.ccm_approach

    @property
    def ccm_construction(self) -> str:
        return self.finite_torus_convention.ccm_construction

    @property
    def evaluation_representation(self) -> str:
        return self.finite_torus_convention.evaluation_representation

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


def _validate_posthf_numerical_support_cutoffs(
    *,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
) -> None:
    """Fail closed on invalid numerical support before reference SCF work."""

    for name, value in (
        ("lattice_cutoff_bohr", lattice_cutoff_bohr),
        ("rsgdf_ke_cutoff", rsgdf_ke_cutoff),
        ("gdf_linear_dep_threshold", gdf_linear_dep_threshold),
    ):
        message = (
            f"aiccm2026dev-b post-HF {name} must be a positive finite "
            "real number"
        )
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(message)
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError(message) from None
        if not np.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(message)


@dataclass(frozen=True)
class AICCM2026DevBMP2Result(_FiniteTorusConventionMixin):
    """Canonical finite-torus RI-MP2 energy per primitive cell."""

    mesh: tuple[int, int, int]
    n_cyclic_cells: int
    n_kpoints: int
    n_occ: int
    n_vir: int
    n_aux: int
    e_hf_per_cell: float
    e_corr_per_cell: float
    e_corr_ss_per_cell: float
    e_corr_os_per_cell: float
    e_total_per_cell: float
    max_energy_imaginary_residual: float
    momentum_conservation_error: float
    aux_basis_name: str
    hf_diagnostics: AICCM2026DevBDiagnostics
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    hf_result: object
    backend: str = "aiccm2026dev-b-ri-mp2"

    @property
    def energy(self) -> float:
        """Total RI-MP2 energy per primitive cell in hartree."""

        return self.e_total_per_cell


@dataclass(frozen=True)
class AICCM2026DevBUMP2Result(_FiniteTorusConventionMixin):
    """Unrestricted RI-MP2 or local-PNO UMP2 result per primitive cell."""

    mesh: tuple[int, int, int]
    n_cyclic_cells: int
    e_hf_per_cell: float
    e_corr_per_cell: float
    e_aa_per_cell: float
    e_bb_per_cell: float
    e_ab_per_cell: float
    e_total_per_cell: float
    n_pairs_aa: int
    n_pairs_bb: int
    n_pairs_ab: int
    converged: bool
    localization: str
    cderi_imaginary_residual: float
    cderi_symmetry_residual: float
    matrix_imaginary_residual: float
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    solver_result: object
    hf_result: object
    localization_result: object | None = None
    backend: str = "aiccm2026dev-b-ri-ump2"

    @property
    def energy(self) -> float:
        return self.e_total_per_cell


@dataclass(frozen=True)
class AICCM2026DevBUCCSDResult(_FiniteTorusConventionMixin):
    """Full-domain or PNO-truncated UCCSD(T) pilot result per cell.

    The current solver is the explicitly gated O(N^6) projection oracle.  It
    establishes the unrestricted finite-torus equations and exact PNO limit;
    it is not presented as the later reduced-scaling production kernel.
    """

    mesh: tuple[int, int, int]
    n_cyclic_cells: int
    e_hf_per_cell: float
    e_corr_per_cell: float
    e_t_per_cell: float
    e_total_per_cell: float
    n_pairs: int
    n_iter: int
    converged: bool
    t1_norm: float
    avg_pno: float
    localization: str
    with_triples: bool
    cderi_imaginary_residual: float
    cderi_symmetry_residual: float
    matrix_imaginary_residual: float
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    solver_result: object
    hf_result: object
    localization_result: object | None = None
    backend: str = "aiccm2026dev-b-uccsd"

    @property
    def energy(self) -> float:
        return self.e_total_per_cell


@dataclass(frozen=True)
class AICCM2026DevBDLPNOMP2Result(_FiniteTorusConventionMixin):
    """Local-PNO MP2 result on the B finite torus, per primitive cell."""

    mesh: tuple[int, int, int]
    n_cyclic_cells: int
    e_hf_per_cell: float
    e_corr_per_cell: float
    raw_local_e_corr_per_cell: float
    complete_space_correction_per_cell: float
    e_total_per_cell: float
    n_pairs: int
    n_pairs_screened: int
    n_iter: int
    converged: bool
    localization: str
    cderi_imaginary_residual: float
    cderi_symmetry_residual: float
    matrix_imaginary_residual: float
    localization_result: object | None
    local_correlation_space: AICCM2026DevBLocalCorrelationSpace | None
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    solver_result: object
    hf_result: object
    backend: str = "aiccm2026dev-b-dlpno-mp2"

    @property
    def energy(self) -> float:
        """Total local-PNO MP2 energy per primitive cell in hartree."""

        return self.e_total_per_cell


@dataclass(frozen=True)
class AICCM2026DevBDLPNOCCSDResult(_FiniteTorusConventionMixin):
    """Local-PNO CCSD or CCSD(T) result on the B finite torus."""

    mesh: tuple[int, int, int]
    n_cyclic_cells: int
    e_hf_per_cell: float
    e_corr_per_cell: float
    e_t_per_cell: float
    e_total_per_cell: float
    n_pairs: int
    n_pairs_screened: int
    n_iter: int
    converged: bool
    t1_norm: float
    localization: str
    with_triples: bool
    triples_mode: str
    cderi_imaginary_residual: float
    cderi_symmetry_residual: float
    matrix_imaginary_residual: float
    localization_result: object | None
    local_correlation_space: AICCM2026DevBLocalCorrelationSpace | None
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    solver_result: object
    hf_result: object
    backend: str

    @property
    def energy(self) -> float:
        """Total local-PNO CC energy per primitive cell in hartree."""

        return self.e_total_per_cell


def _reference_finite_torus_convention(
    hf_result: object,
) -> AICCM2026DevBFiniteTorusConvention:
    diagnostics = getattr(hf_result, "aiccm2026dev_b", None)
    if diagnostics is None:
        raise RuntimeError(
            "aiccm2026dev-b post-HF requires a B-stream HF reference carrying "
            "finite-torus convention diagnostics"
        )
    convention = getattr(diagnostics, "finite_torus_convention", None)
    if convention is None:
        raise RuntimeError(
            "aiccm2026dev-b post-HF reference predates explicit finite-torus "
            "convention metadata"
        )
    if convention.coulomb_kernel != "3d-periodic-g0":
        raise ValueError(
            "aiccm2026dev-b post-HF cannot compare references with different "
            f"Coulomb kernels: expected '3d-periodic-g0', got "
            f"{convention.coulomb_kernel!r}"
        )
    if convention.exchange_q0 != "bvk-ewald":
        raise ValueError(
            "aiccm2026dev-b post-HF finite-N correlation inherits the "
            "exchange q=0 convention through the HF orbital energies and "
            f"denominators; expected exchange_q0='bvk-ewald', got "
            f"{convention.exchange_q0!r}"
        )
    if convention.exchange_q0_applicability != "active":
        raise ValueError(
            "aiccm2026dev-b post-HF requires a live full-range exchange-q=0 "
            "seam in the HF reference denominators; expected "
            "exchange_q0_applicability='active', got "
            f"{convention.exchange_q0_applicability!r}"
        )
    ecp_total_ncore = int(getattr(diagnostics, "ecp_total_ncore", 0) or 0)
    if ecp_total_ncore:
        raise NotImplementedError(
            "aiccm2026dev-b post-HF with periodic ECP references is not yet "
            "validated: the effective electron count, frozen-core convention, "
            "and auxiliary/ECP provenance must be carried through the "
            f"correlation denominators (ecp_total_ncore={ecp_total_ncore})"
        )
    return convention


def _build_canonical_lpq_cache(
    system: PeriodicSystem,
    basis: BasisSet,
    kpoints_cart: np.ndarray,
    aux_name: str,
    *,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
) -> tuple[dict[tuple[int, int], np.ndarray], object]:
    molecule = system.unit_cell_molecule()
    auxiliary = make_aux_basis_set(molecule, aux_name=aux_name)
    auxiliary_modrho = make_modrho_aux_basis(auxiliary, molecule)
    lattice_options = LatticeSumOptions()
    lattice_options.cutoff_bohr = float(lattice_cutoff_bohr)
    kcart = np.asarray(kpoints_cart, dtype=float).reshape(-1, 3)
    cache: dict[tuple[int, int], np.ndarray] = {}
    for ki, bra_k in enumerate(kcart):
        for ka, ket_k in enumerate(kcart):
            cache[(ki, ka)] = build_lpq_bloch_native_fft(
                system,
                basis,
                auxiliary_modrho,
                bra_k,
                ket_k,
                ke_cutoff=float(rsgdf_ke_cutoff),
                lat_opts=lattice_options,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                canonical_auxiliary_basis=True,
            )
    return cache, auxiliary


def _build_canonical_lov_cache(
    system: PeriodicSystem,
    basis: BasisSet,
    kpoints_cart: np.ndarray,
    coefficients: Sequence[np.ndarray],
    n_occ: int,
    aux_name: str,
    *,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
) -> tuple[dict[tuple[int, int], np.ndarray], object]:
    """Build canonical MP2 ``L(P,i,a;k_i,k_a)`` factors without storing Lpq.

    Canonical character-space RI-MP2 only needs occupied-virtual transformed
    three-center factors. The real-torus local-correlation path still needs the
    full AO-space pair cache, but MP2 can stream each pair-resolved ``Lpq``
    block directly through the AO-to-MO transform and then release it.
    """

    molecule = system.unit_cell_molecule()
    auxiliary = make_aux_basis_set(molecule, aux_name=aux_name)
    auxiliary_modrho = make_modrho_aux_basis(auxiliary, molecule)
    lattice_options = LatticeSumOptions()
    lattice_options.cutoff_bohr = float(lattice_cutoff_bohr)
    kcart = np.asarray(kpoints_cart, dtype=float).reshape(-1, 3)
    if len(coefficients) != len(kcart):
        raise ValueError(
            "AICCM2026DEV-B canonical MP2 needs one coefficient block per "
            f"character point; got {len(coefficients)} blocks for "
            f"{len(kcart)} k-points"
        )
    n_aux = int(auxiliary.nbasis)
    lov: dict[tuple[int, int], np.ndarray] = {}
    for ki, bra_k in enumerate(kcart):
        occupied = np.asarray(coefficients[ki])[:, :n_occ]
        for ka, ket_k in enumerate(kcart):
            virtual = np.asarray(coefficients[ka])[:, n_occ:]
            lpq = build_lpq_bloch_native_fft(
                system,
                basis,
                auxiliary_modrho,
                bra_k,
                ket_k,
                ke_cutoff=float(rsgdf_ke_cutoff),
                lat_opts=lattice_options,
                linear_dep_thr=float(gdf_linear_dep_threshold),
                canonical_auxiliary_basis=True,
            )
            if lpq.shape[0] != n_aux:
                raise RuntimeError("canonical AICCM2026DEV-B MP2 cderi rank mismatch")
            lov[(ki, ka)] = aiccm2026dev_b_3index_mo_transform_complex(
                np.ascontiguousarray(np.asarray(lpq, dtype=np.complex128)),
                np.ascontiguousarray(np.asarray(occupied, dtype=np.complex128)),
                np.ascontiguousarray(np.asarray(virtual, dtype=np.complex128)),
            )
            del lpq
    return lov, auxiliary


def _cyclic_translations(mesh: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(
        [
            (i, j, k)
            for i in range(mesh[0])
            for j in range(mesh[1])
            for k in range(mesh[2])
        ],
        dtype=int,
    )


def _periodic_matrix_to_real_supercell(
    matrices_k: Sequence[np.ndarray],
    kpoints_frac: np.ndarray,
    translations: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Transform character blocks to the AO matrix of the finite torus."""

    matrix, imaginary = aiccm2026dev_b_matrix_to_real_supercell(
        np.ascontiguousarray(np.asarray(matrices_k, dtype=np.complex128)),
        np.ascontiguousarray(np.asarray(kpoints_frac, dtype=float).reshape(-1, 3)),
        np.ascontiguousarray(np.asarray(translations, dtype=np.int64).reshape(-1, 3)),
    )
    return np.asarray(matrix, dtype=float), float(imaginary)


def _lpq_cache_to_real_supercell(
    lpq_cache: dict[tuple[int, int], np.ndarray],
    kpoints_frac: np.ndarray,
    mesh: tuple[int, int, int],
) -> tuple[np.ndarray, float, float]:
    """Inverse-transform pair-resolved RI factors to the real finite torus."""

    translations = _cyclic_translations(mesh)
    kfrac = np.asarray(kpoints_frac, dtype=float).reshape(-1, 3)
    n_cells = len(translations)
    if len(kfrac) != n_cells:
        raise ValueError("AICCM2026DEV-B cderi transform needs one character per cell")
    sample = lpq_cache[(0, 0)]
    n_aux, nbf, nbf_right = sample.shape
    if nbf != nbf_right:
        raise ValueError("AICCM2026DEV-B pair cderi matrices must be square")
    complex_cache = {
        key: np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
        for key, value in lpq_cache.items()
    }
    factors, imaginary, symmetry = aiccm2026dev_b_lpq_to_real_supercell(
        complex_cache,
        np.ascontiguousarray(kfrac),
        np.ascontiguousarray(translations, dtype=np.int64),
    )
    complex_cache.clear()
    expected = (n_cells * n_aux, n_cells * nbf, n_cells * nbf)
    if tuple(factors.shape) != expected:
        raise RuntimeError(
            "AICCM2026DEV-B native cderi transform returned shape "
            f"{tuple(factors.shape)}, expected {expected}"
        )
    return np.asarray(factors, dtype=float), float(imaginary), float(symmetry)


def _lpq_cache_to_real_home_auxiliary(
    lpq_cache: dict[tuple[int, int], np.ndarray],
    kpoints_frac: np.ndarray,
    mesh: tuple[int, int, int],
) -> tuple[np.ndarray, float, float]:
    """Retain the home-auxiliary row of a canonical real-torus RI factor.

    Every momentum pair must use the same full primitive-auxiliary frame, as
    enforced by ``_build_canonical_lpq_cache``. Reduced q-dependent auxiliary
    eigenvector frames do not satisfy this storage contract.
    """

    translations = _cyclic_translations(mesh)
    kfrac = np.asarray(kpoints_frac, dtype=float).reshape(-1, 3)
    n_cells = len(translations)
    if len(kfrac) != n_cells:
        raise ValueError("AICCM2026DEV-B cderi transform needs one character per cell")
    sample = lpq_cache[(0, 0)]
    n_aux, nbf, nbf_right = sample.shape
    if nbf != nbf_right:
        raise ValueError("AICCM2026DEV-B pair cderi matrices must be square")
    complex_cache = {
        key: np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
        for key, value in lpq_cache.items()
    }
    factors, imaginary, symmetry = aiccm2026dev_b_lpq_to_real_home_auxiliary(
        complex_cache,
        np.ascontiguousarray(kfrac),
        np.ascontiguousarray(translations, dtype=np.int64),
        np.ascontiguousarray(np.asarray(mesh, dtype=np.int64)),
    )
    complex_cache.clear()
    expected = (n_aux, n_cells * nbf, n_cells * nbf)
    if tuple(factors.shape) != expected:
        raise RuntimeError(
            "AICCM2026DEV-B native home-auxiliary cderi transform returned "
            f"shape {tuple(factors.shape)}, expected {expected}"
        )
    return np.asarray(factors, dtype=float), float(imaginary), float(symmetry)


def _build_supercell_system(
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
    *,
    multiplicity: int = 1,
) -> PeriodicSystem:
    from ..._vibeqc_core import Atom

    lattice = np.asarray(system.lattice, dtype=float)
    super_lattice = lattice @ np.diag(np.asarray(mesh, dtype=float))
    atoms = []
    for translation in _cyclic_translations(mesh):
        shift = lattice @ np.asarray(translation, dtype=float)
        for atom in system.unit_cell:
            position = np.asarray(atom.xyz, dtype=float) + shift
            atoms.append(Atom(int(atom.Z), position.tolist()))
    return PeriodicSystem(
        int(system.dim),
        super_lattice,
        atoms,
        charge=int(system.charge) * int(np.prod(mesh)),
        multiplicity=int(multiplicity),
    )


class _BRealDFAdapter:
    def __init__(
        self,
        home_factors: np.ndarray,
        auxiliary_basis: object,
        mesh: tuple[int, int, int],
    ):
        factors = np.ascontiguousarray(np.asarray(home_factors, dtype=float))
        mesh_tuple = tuple(int(value) for value in mesh)
        if factors.ndim != 3 or factors.shape[1] != factors.shape[2]:
            raise ValueError(
                "AICCM2026DEV-B home-auxiliary factors must have shape "
                "(n_aux_unit, n_ao, n_ao)"
            )
        if len(mesh_tuple) != 3 or any(value < 1 for value in mesh_tuple):
            raise ValueError(
                "AICCM2026DEV-B adapter mesh must be three positive integers"
            )
        n_cells = int(np.prod(mesh_tuple))
        if factors.shape[1] % n_cells != 0:
            raise ValueError(
                "AICCM2026DEV-B home-auxiliary AO rank does not match the "
                "cyclic mesh"
            )
        self.home_factors = factors
        self.mesh = mesh_tuple
        self.aux_basis = auxiliary_basis
        self.n_aux = n_cells * factors.shape[0]
        self.n_orb = factors.shape[1]

    def mo_transform(
        self,
        coefficients_left: np.ndarray,
        coefficients_right: np.ndarray | None = None,
    ) -> np.ndarray:
        right = coefficients_left if coefficients_right is None else coefficients_right
        left_array = np.asarray(coefficients_left)
        right_array = np.asarray(right)
        mesh_array = np.ascontiguousarray(np.asarray(self.mesh, dtype=np.int64))
        if np.iscomplexobj(left_array) or np.iscomplexobj(right_array):
            return aiccm2026dev_b_home_auxiliary_3index_mo_transform_complex(
                np.ascontiguousarray(self.home_factors, dtype=np.complex128),
                np.ascontiguousarray(left_array, dtype=np.complex128),
                np.ascontiguousarray(right_array, dtype=np.complex128),
                mesh_array,
            )
        return aiccm2026dev_b_home_auxiliary_3index_mo_transform(
            self.home_factors,
            np.ascontiguousarray(left_array, dtype=float),
            np.ascontiguousarray(right_array, dtype=float),
            mesh_array,
        )


@contextmanager
def _patched_overlap(overlap: np.ndarray):
    import vibeqc._vibeqc_core as core

    original = core.compute_overlap
    core.compute_overlap = lambda _basis: overlap
    try:
        yield
    finally:
        core.compute_overlap = original


@dataclass(frozen=True)
class _BRealReference:
    super_system: PeriodicSystem
    molecule: object
    basis: BasisSet
    df: _BRealDFAdapter
    hf: object
    n_cells: int
    cderi_imaginary_residual: float
    cderi_symmetry_residual: float
    matrix_imaginary_residual: float


def _build_b_real_reference(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: tuple[int, int, int],
    hf_result: object,
    lpq_cache: dict[tuple[int, int], np.ndarray],
    auxiliary_name: str,
) -> _BRealReference:
    convention = _reference_finite_torus_convention(hf_result)
    if tuple(convention.character_mesh_shape) != tuple(mesh):
        raise ValueError(
            "aiccm2026dev-b post-HF reference convention mesh "
            f"{convention.character_mesh_shape!r} does not match requested "
            f"mesh {mesh!r}"
        )
    kpoints = cyclic_gamma_mesh(system, mesh)
    translations = _cyclic_translations(mesh)
    overlap, overlap_imaginary = _periodic_matrix_to_real_supercell(
        hf_result.overlap,
        np.asarray(kpoints.kpoints_frac),
        translations,
    )
    fock, fock_imaginary = _periodic_matrix_to_real_supercell(
        hf_result.fock,
        np.asarray(kpoints.kpoints_frac),
        translations,
    )
    home_factors, cderi_imaginary, cderi_symmetry = (
        _lpq_cache_to_real_home_auxiliary(
            lpq_cache,
            np.asarray(kpoints.kpoints_frac),
            mesh,
        )
    )
    lpq_cache.clear()
    if max(cderi_imaginary, cderi_symmetry) > 1e-7:
        raise RuntimeError(
            "AICCM2026DEV-B real RI transform violated Hermitian symmetry: "
            f"imaginary={cderi_imaginary:.3e}, "
            f"symmetry={cderi_symmetry:.3e}"
        )

    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(overlap)
    keep = overlap_eigenvalues > 1e-10 * float(overlap_eigenvalues[-1])
    if int(np.count_nonzero(keep)) != overlap.shape[0]:
        raise RuntimeError(
            "AICCM2026DEV-B real-torus post-HF reference is linearly dependent"
        )
    orthogonalizer = overlap_eigenvectors[:, keep] / np.sqrt(
        overlap_eigenvalues[keep]
    )[None, :]
    reduced_fock = orthogonalizer.T @ fock @ orthogonalizer
    orbital_energies, reduced_coefficients = np.linalg.eigh(
        0.5 * (reduced_fock + reduced_fock.T)
    )
    coefficients = orthogonalizer @ reduced_coefficients

    super_system = _build_supercell_system(system, mesh)
    molecule = super_system.unit_cell_molecule()
    super_basis = BasisSet(molecule, basis.name)
    if int(super_basis.nbasis) != home_factors.shape[1]:
        raise RuntimeError(
            "AICCM2026DEV-B supercell basis ordering does not match the "
            "finite-group AO transform"
        )
    super_auxiliary = make_aux_basis_set(molecule, aux_name=auxiliary_name)
    n_cells = int(np.prod(mesh))
    if int(super_auxiliary.nbasis) != n_cells * home_factors.shape[0]:
        raise RuntimeError(
            "AICCM2026DEV-B supercell auxiliary ordering does not match the "
            "finite-group cderi transform"
        )
    df = _BRealDFAdapter(home_factors, super_auxiliary, mesh)
    hf = SimpleNamespace(
        energy=float(hf_result.energy) * n_cells,
        converged=True,
        mo_coeffs=coefficients,
        mo_energies=orbital_energies,
        fock=fock,
        overlap=overlap,
    )
    return _BRealReference(
        super_system=super_system,
        molecule=molecule,
        basis=super_basis,
        df=df,
        hf=hf,
        n_cells=n_cells,
        cderi_imaginary_residual=cderi_imaginary,
        cderi_symmetry_residual=cderi_symmetry,
        matrix_imaginary_residual=max(overlap_imaginary, fock_imaginary),
    )


@dataclass(frozen=True)
class _BRealUReference:
    super_system: PeriodicSystem
    molecule: object
    basis: BasisSet
    df: _BRealDFAdapter
    hf: object
    n_cells: int
    cderi_imaginary_residual: float
    cderi_symmetry_residual: float
    matrix_imaginary_residual: float


def _metric_orbitals(overlap: np.ndarray, fock: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(overlap)
    keep = overlap_eigenvalues > 1e-10 * float(overlap_eigenvalues[-1])
    if int(np.count_nonzero(keep)) != overlap.shape[0]:
        raise RuntimeError(
            "AICCM2026DEV-B real-torus post-HF reference is linearly dependent"
        )
    orthogonalizer = overlap_eigenvectors[:, keep] / np.sqrt(
        overlap_eigenvalues[keep]
    )[None, :]
    reduced_fock = orthogonalizer.T @ fock @ orthogonalizer
    energies, reduced_coefficients = np.linalg.eigh(
        0.5 * (reduced_fock + reduced_fock.T)
    )
    return energies, orthogonalizer @ reduced_coefficients


def _build_b_real_ureference(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: tuple[int, int, int],
    hf_result: object,
    lpq_cache: dict[tuple[int, int], np.ndarray],
    auxiliary_name: str,
) -> _BRealUReference:
    """Transform the finite-character UHF Hamiltonian to one real torus."""

    convention = _reference_finite_torus_convention(hf_result)
    if tuple(convention.character_mesh_shape) != tuple(mesh):
        raise ValueError(
            "aiccm2026dev-b unrestricted post-HF reference convention mesh "
            f"{convention.character_mesh_shape!r} does not match requested "
            f"mesh {mesh!r}"
        )
    kpoints = cyclic_gamma_mesh(system, mesh)
    translations = _cyclic_translations(mesh)
    overlap, overlap_imaginary = _periodic_matrix_to_real_supercell(
        hf_result.overlap,
        np.asarray(kpoints.kpoints_frac),
        translations,
    )
    fock_alpha, fock_alpha_imaginary = _periodic_matrix_to_real_supercell(
        hf_result.fock_alpha,
        np.asarray(kpoints.kpoints_frac),
        translations,
    )
    fock_beta, fock_beta_imaginary = _periodic_matrix_to_real_supercell(
        hf_result.fock_beta,
        np.asarray(kpoints.kpoints_frac),
        translations,
    )
    home_factors, cderi_imaginary, cderi_symmetry = (
        _lpq_cache_to_real_home_auxiliary(
            lpq_cache,
            np.asarray(kpoints.kpoints_frac),
            mesh,
        )
    )
    lpq_cache.clear()
    if max(cderi_imaginary, cderi_symmetry) > 1e-7:
        raise RuntimeError(
            "AICCM2026DEV-B real RI transform violated Hermitian symmetry: "
            f"imaginary={cderi_imaginary:.3e}, "
            f"symmetry={cderi_symmetry:.3e}"
        )
    energies_alpha, coefficients_alpha = _metric_orbitals(overlap, fock_alpha)
    energies_beta, coefficients_beta = _metric_orbitals(overlap, fock_beta)

    n_cells = int(np.prod(mesh))
    super_multiplicity = n_cells * (int(system.multiplicity) - 1) + 1
    super_system = _build_supercell_system(
        system,
        mesh,
        multiplicity=super_multiplicity,
    )
    molecule = super_system.unit_cell_molecule()
    super_basis = BasisSet(molecule, basis.name)
    if int(super_basis.nbasis) != home_factors.shape[1]:
        raise RuntimeError(
            "AICCM2026DEV-B supercell basis ordering does not match the "
            "finite-group AO transform"
        )
    super_auxiliary = make_aux_basis_set(molecule, aux_name=auxiliary_name)
    if int(super_auxiliary.nbasis) != n_cells * home_factors.shape[0]:
        raise RuntimeError(
            "AICCM2026DEV-B supercell auxiliary ordering does not match the "
            "finite-group cderi transform"
        )
    df = _BRealDFAdapter(home_factors, super_auxiliary, mesh)
    hf = SimpleNamespace(
        energy=float(hf_result.energy) * n_cells,
        converged=True,
        mo_coeffs_alpha=coefficients_alpha,
        mo_coeffs_beta=coefficients_beta,
        mo_energies_alpha=energies_alpha,
        mo_energies_beta=energies_beta,
        fock_alpha=fock_alpha,
        fock_beta=fock_beta,
        overlap=overlap,
    )
    return _BRealUReference(
        super_system=super_system,
        molecule=molecule,
        basis=super_basis,
        df=df,
        hf=hf,
        n_cells=n_cells,
        cderi_imaginary_residual=cderi_imaginary,
        cderi_symmetry_residual=cderi_symmetry,
        matrix_imaginary_residual=max(
            overlap_imaginary,
            fock_alpha_imaginary,
            fock_beta_imaginary,
        ),
    )


def _prepare_b_real_reference(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: tuple[int, int, int],
    options: PeriodicRHFOptions | None,
    *,
    auxiliary_name: str,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
    progress: bool | object | None,
    verbose: int | None,
) -> tuple[_BRealReference, object]:
    if int(system.dim) != 3:
        raise NotImplementedError(
            "aiccm2026dev-b local correlation is currently restricted to 3D "
            "because the 1D/2D fitted and direct long-range gauges are not "
            "yet matched"
        )
    hf_result = run_aiccm2026dev_b_rhf(
        system,
        basis,
        mesh,
        options,
        backend="ri",
        aux_basis=auxiliary_name,
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        progress=progress,
        verbose=verbose,
    )
    if not bool(hf_result.converged):
        raise RuntimeError(
            "aiccm2026dev-b local correlation requires a converged RI-RHF reference"
        )
    kpoints = cyclic_gamma_mesh(system, mesh)
    lpq_cache, _ = _build_canonical_lpq_cache(
        system,
        basis,
        np.asarray(kpoints.kpoints_cart, dtype=float),
        auxiliary_name,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )
    return (
        _build_b_real_reference(
            system,
            basis,
            mesh,
            hf_result,
            lpq_cache,
            auxiliary_name,
        ),
        hf_result,
    )


def _prepare_b_real_ureference(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: tuple[int, int, int],
    options: PeriodicRHFOptions | None,
    *,
    auxiliary_name: str,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
    progress: bool | object | None,
    verbose: int | None,
) -> tuple[_BRealUReference, object]:
    if int(system.dim) != 3:
        raise NotImplementedError(
            "aiccm2026dev-b unrestricted local correlation is currently "
            "restricted to 3D because the 1D/2D fitted and direct long-range "
            "gauges are not yet matched"
        )
    hf_result = run_aiccm2026dev_b_uhf(
        system,
        basis,
        mesh,
        options,
        backend="ri",
        aux_basis=auxiliary_name,
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        progress=progress,
        verbose=verbose,
    )
    if not bool(hf_result.converged):
        raise RuntimeError(
            "aiccm2026dev-b unrestricted correlation requires a converged RI-UHF reference"
        )
    kpoints = cyclic_gamma_mesh(system, mesh)
    lpq_cache, _ = _build_canonical_lpq_cache(
        system,
        basis,
        np.asarray(kpoints.kpoints_cart, dtype=float),
        auxiliary_name,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )
    return (
        _build_b_real_ureference(
            system,
            basis,
            mesh,
            hf_result,
            lpq_cache,
            auxiliary_name,
        ),
        hf_result,
    )


def _localize_b_real_reference(
    reference: _BRealReference,
    method: str,
    *,
    hf_result: object,
    system: PeriodicSystem,
    primitive_basis: BasisSet,
    mesh: tuple[int, int, int],
) -> tuple[object, str, object | None]:
    if method == "none":
        return reference.hf, method, None
    if method in {"wannier", "iao"}:
        from .localization import (
            localize_aiccm2026dev_b_occupied,
        )

        localization = localize_aiccm2026dev_b_occupied(
            hf_result,
            system,
            primitive_basis,
            mesh,
            method=method,
        )
        n_occ = int(reference.molecule.n_electrons()) // 2
        occupied = np.asarray(localization.coefficients)
        if occupied.shape != (int(reference.basis.nbasis), n_occ):
            raise RuntimeError(
                "AICCM2026DEV-B localized occupied coefficients do not match "
                "the real-torus reference"
            )
        if np.iscomplexobj(occupied):
            real_trial = occupied.real
            gram = real_trial.T @ reference.hf.overlap @ real_trial
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (gram + gram.T)
            )
            if eigenvalues[0] <= 1e-10 * eigenvalues[-1]:
                raise RuntimeError(
                    "AICCM2026DEV-B localization has no full-rank real "
                    "time-reversal gauge for the real local solver"
                )
            occupied_real = real_trial @ (
                eigenvectors / np.sqrt(eigenvalues)[None, :]
            ) @ eigenvectors.T
            projector_error = float(
                np.linalg.norm(
                    occupied @ occupied.conj().T
                    - occupied_real @ occupied_real.T
                )
            )
            if projector_error > 1e-7:
                raise RuntimeError(
                    "AICCM2026DEV-B real-gauge projection changed the "
                    f"occupied projector; error={projector_error:.3e}"
                )
            occupied = occupied_real
        coefficients = np.asarray(reference.hf.mo_coeffs)
        localized_coefficients = np.concatenate(
            [occupied, coefficients[:, n_occ:]],
            axis=1,
        )
        metric_error = float(
            np.max(
                np.abs(
                    localized_coefficients.conj().T
                    @ reference.hf.overlap
                    @ localized_coefficients
                    - np.eye(localized_coefficients.shape[1])
                )
            )
        )
        if metric_error > 1e-7:
            raise RuntimeError(
                "AICCM2026DEV-B localized occupied/virtual union is not "
                f"metric orthonormal; error={metric_error:.3e}"
            )
        proxy = SimpleNamespace(
            energy=reference.hf.energy,
            converged=True,
            mo_coeffs=np.asarray(localized_coefficients, dtype=float),
            mo_energies=reference.hf.mo_energies,
            fock=reference.hf.fock,
            overlap=reference.hf.overlap,
        )
        return proxy, method, localization
    if method != "pipek-mezey":
        raise ValueError(
            "aiccm2026dev-b local correlation supports localise='wannier', "
            "'iao', 'pipek-mezey', or 'none'; molecular Boys localization "
            "is not periodic-boundary safe"
        )
    from ...periodic_localise import localise_periodic_gamma

    n_occ = int(reference.molecule.n_electrons()) // 2
    localized = localise_periodic_gamma(
        reference.hf,
        reference.basis,
        reference.super_system,
        method="pipek-mezey",
        n_occ=n_occ,
    )
    coefficients = np.asarray(reference.hf.mo_coeffs)
    localized_coefficients = np.concatenate(
        [localized.C_loc, coefficients[:, n_occ:]],
        axis=1,
    )
    proxy = SimpleNamespace(
        energy=reference.hf.energy,
        converged=True,
        mo_coeffs=localized_coefficients,
        mo_energies=reference.hf.mo_energies,
        fock=reference.hf.fock,
        overlap=reference.hf.overlap,
    )
    return proxy, method, None


def _real_occupied_gauge(
    occupied: np.ndarray,
    overlap: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    if not np.iscomplexobj(occupied):
        return np.asarray(occupied, dtype=float)
    real_trial = occupied.real
    gram = real_trial.T @ overlap @ real_trial
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (gram + gram.T))
    if eigenvalues[0] <= 1e-10 * eigenvalues[-1]:
        raise RuntimeError(
            f"AICCM2026DEV-B {label} localization has no full-rank real "
            "time-reversal gauge"
        )
    real_occupied = real_trial @ (
        eigenvectors / np.sqrt(eigenvalues)[None, :]
    ) @ eigenvectors.T
    projector_error = float(
        np.linalg.norm(
            occupied @ occupied.conj().T
            - real_occupied @ real_occupied.T
        )
    )
    if projector_error > 1e-7:
        raise RuntimeError(
            f"AICCM2026DEV-B {label} real-gauge projection changed the "
            f"occupied projector; error={projector_error:.3e}"
        )
    return real_occupied


def _localize_b_real_ureference(
    reference: _BRealUReference,
    method: str,
    *,
    hf_result: object,
    system: PeriodicSystem,
    primitive_basis: BasisSet,
    mesh: tuple[int, int, int],
) -> tuple[object, str, object | None]:
    if method == "none":
        return reference.hf, method, None
    if method not in {"wannier", "iao"}:
        raise ValueError(
            "aiccm2026dev-b unrestricted local correlation supports the "
            "PBC-safe localise='wannier' or 'iao' gauges, or exact-limit "
            "localise='none'"
        )
    from .localization import (
        localize_aiccm2026dev_b_unrestricted_occupied,
    )

    localization = localize_aiccm2026dev_b_unrestricted_occupied(
        hf_result,
        system,
        primitive_basis,
        mesh,
        method=method,
    )
    ne = int(reference.molecule.n_electrons())
    two_s = int(reference.molecule.multiplicity) - 1
    n_alpha = (ne + two_s) // 2
    n_beta = (ne - two_s) // 2
    occupied_alpha = _real_occupied_gauge(
        np.asarray(localization.alpha.coefficients),
        np.asarray(reference.hf.overlap),
        label="alpha",
    )
    if occupied_alpha.shape[1] != n_alpha:
        raise RuntimeError("AICCM2026DEV-B alpha localized rank is inconsistent")
    if n_beta:
        if localization.beta is None:
            raise RuntimeError("AICCM2026DEV-B beta localization is missing")
        occupied_beta = _real_occupied_gauge(
            np.asarray(localization.beta.coefficients),
            np.asarray(reference.hf.overlap),
            label="beta",
        )
    else:
        occupied_beta = np.empty((int(reference.basis.nbasis), 0))
    if occupied_beta.shape[1] != n_beta:
        raise RuntimeError("AICCM2026DEV-B beta localized rank is inconsistent")

    coefficients_alpha = np.concatenate(
        [occupied_alpha, np.asarray(reference.hf.mo_coeffs_alpha)[:, n_alpha:]],
        axis=1,
    )
    coefficients_beta = np.concatenate(
        [occupied_beta, np.asarray(reference.hf.mo_coeffs_beta)[:, n_beta:]],
        axis=1,
    )
    for label, coefficients in (
        ("alpha", coefficients_alpha),
        ("beta", coefficients_beta),
    ):
        error = float(
            np.max(
                np.abs(
                    coefficients.T @ reference.hf.overlap @ coefficients
                    - np.eye(coefficients.shape[1])
                )
            )
        )
        if error > 1e-7:
            raise RuntimeError(
                f"AICCM2026DEV-B {label} localized occupied/virtual union "
                f"is not metric orthonormal; error={error:.3e}"
            )
    proxy = SimpleNamespace(
        energy=reference.hf.energy,
        converged=True,
        mo_coeffs_alpha=coefficients_alpha,
        mo_coeffs_beta=coefficients_beta,
        mo_energies_alpha=reference.hf.mo_energies_alpha,
        mo_energies_beta=reference.hf.mo_energies_beta,
        fock_alpha=reference.hf.fock_alpha,
        fock_beta=reference.hf.fock_beta,
        overlap=reference.hf.overlap,
    )
    return proxy, method, localization


def _local_correlation_space(
    reference: _BRealReference,
    localized_hf: object,
    localization_result: object | None,
) -> AICCM2026DevBLocalCorrelationSpace | None:
    """Build exact-limit PAOs and finite-translation pair orbits.

    A translation permutation is only available from the B-specific
    Wannier/IAO path.  Pipek-Mezey and canonical real-torus orbitals may mix
    degenerate translation sectors, so manufacturing a cell label for them
    would be unsafe.
    """

    if localization_result is None:
        return None
    from .pno import (
        enumerate_pair_orbits,
        projected_atomic_orbitals,
    )

    n_occ = int(reference.molecule.n_electrons()) // 2
    occupied = np.asarray(localized_hf.mo_coeffs)[:, :n_occ]
    paos = projected_atomic_orbitals(
        occupied,
        np.asarray(reference.hf.overlap),
        np.asarray(reference.hf.fock),
    )
    permutations = np.asarray(localization_result.translation_permutations)
    orbits = enumerate_pair_orbits(permutations, n_cells=reference.n_cells)
    n_pairs = n_occ * (n_occ + 1) // 2
    if sum(orbit.multiplicity for orbit in orbits) != n_pairs:
        raise RuntimeError("AICCM2026DEV-B translation pair orbits are incomplete")
    return AICCM2026DevBLocalCorrelationSpace(
        pao_rank=paos.rank,
        canonical_virtual_rank=int(reference.basis.nbasis) - n_occ,
        pao_discarded_rank=paos.discarded_rank,
        pao_orthonormality_error=paos.orthonormality_error,
        pao_occupied_leakage_error=paos.occupied_leakage_error,
        n_occupied=n_occ,
        n_pairs=n_pairs,
        n_translation_unique_pairs=len(orbits),
        translation_reduction_factor=n_pairs / len(orbits),
        translation_covariance_error=float(
            localization_result.translation_covariance_error
        ),
        translation_orbits_validated=bool(
            localization_result.translation_covariance_error < 1e-8
        ),
    )


def _complete_space_mp2_audit(
    reference: _BRealReference,
) -> float:
    """Evaluate the real canonical gauge of the complete-PAO MP2 limit."""

    from .pno import (
        projected_atomic_orbitals,
    )

    n_occ = int(reference.molecule.n_electrons()) // 2
    occupied = np.asarray(reference.hf.mo_coeffs)[:, :n_occ]
    paos = projected_atomic_orbitals(
        occupied,
        np.asarray(reference.hf.overlap),
        np.asarray(reference.hf.fock),
    )
    fock = np.asarray(reference.hf.fock)
    virtual = np.asarray(paos.coefficients)
    f_occ = occupied.T @ fock @ occupied
    f_vir = virtual.T @ fock @ virtual
    eps_occ, rotation_occ = np.linalg.eigh(0.5 * (f_occ + f_occ.T))
    eps_vir, rotation_vir = np.linalg.eigh(0.5 * (f_vir + f_vir.T))
    factors_ov = reference.df.mo_transform(
        occupied @ rotation_occ,
        virtual @ rotation_vir,
    )
    return float(
        aiccm2026dev_b_real_mp2_energy_from_lov(
            np.ascontiguousarray(np.asarray(factors_ov, dtype=float)),
            np.ascontiguousarray(np.asarray(eps_occ, dtype=float)),
            np.ascontiguousarray(np.asarray(eps_vir, dtype=float)),
            1.0e-12,
        )
    )


def run_aiccm2026dev_b_mp2(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: int | Sequence[int] = (1, 1, 1),
    options: PeriodicRHFOptions | None = None,
    *,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBMP2Result:
    """Run canonical closed-shell RI-MP2 on the B finite translation group.

    The RHF reference is the B-stream pair-resolved RI calculation on the full
    unreduced character mesh. Three crystal momenta are summed independently;
    the fourth is fixed by ``ki - ka + kj - kb = G``. The returned energy is
    per primitive cell.
    """

    mesh_tuple = _normalise_mesh(system, mesh)
    if int(system.dim) != 3:
        raise NotImplementedError(
            "aiccm2026dev-b MP2 is currently restricted to 3D because the "
            "1D/2D fitted and direct long-range gauges are not yet matched"
        )
    _validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )

    aux_name = aux_basis or default_aux_for(basis.name)
    hf_result = run_aiccm2026dev_b_rhf(
        system,
        basis,
        mesh_tuple,
        options,
        backend="ri",
        aux_basis=aux_name,
        gdf_method="rsgdf",
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        progress=progress,
        verbose=verbose,
    )
    if not bool(hf_result.converged):
        raise RuntimeError("aiccm2026dev-b MP2 requires a converged RI-RHF reference")
    convention = _reference_finite_torus_convention(hf_result)

    kpoints = cyclic_gamma_mesh(system, mesh_tuple)
    kcart = np.asarray(kpoints.kpoints_cart, dtype=float)
    n_k = len(kcart)
    n_occ = int(system.n_electrons()) // 2
    n_orb = int(basis.nbasis)
    n_vir = n_orb - n_occ
    if n_vir < 1:
        raise RuntimeError("aiccm2026dev-b MP2 requires at least one virtual orbital")

    coefficients = [np.asarray(value) for value in hf_result.mo_coeffs]
    energies = [np.asarray(value, dtype=float) for value in hf_result.mo_energies]
    lov, auxiliary = _build_canonical_lov_cache(
        system,
        basis,
        kcart,
        coefficients,
        n_occ,
        aux_name,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )
    n_aux = int(auxiliary.nbasis)

    energy_array = np.vstack([np.asarray(block, dtype=float) for block in energies])
    e_ss, e_os, maximum_imaginary = aiccm2026dev_b_mp2_energy_from_lov(
        {
            key: np.ascontiguousarray(np.asarray(value, dtype=np.complex128))
            for key, value in lov.items()
        },
        np.ascontiguousarray(energy_array, dtype=float),
        mesh_tuple,
        (0, 0, 0),
        1.0e-12,
    )
    lov.clear()
    e_ss = float(e_ss)
    e_os = float(e_os)
    maximum_imaginary = float(maximum_imaginary)
    e_corr = e_ss + e_os
    e_hf = float(hf_result.energy)
    return AICCM2026DevBMP2Result(
        mesh=mesh_tuple,
        n_cyclic_cells=int(np.prod(mesh_tuple)),
        n_kpoints=n_k,
        n_occ=n_occ,
        n_vir=n_vir,
        n_aux=n_aux,
        e_hf_per_cell=e_hf,
        e_corr_per_cell=e_corr,
        e_corr_ss_per_cell=e_ss,
        e_corr_os_per_cell=e_os,
        e_total_per_cell=e_hf + e_corr,
        max_energy_imaginary_residual=maximum_imaginary,
        momentum_conservation_error=0.0,
        aux_basis_name=aux_name,
        hf_diagnostics=hf_result.aiccm2026dev_b,
        finite_torus_convention=convention,
        hf_result=hf_result,
    )


def run_aiccm2026dev_b_dlpno_mp2(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: int | Sequence[int] = (1, 1, 1),
    options: PeriodicRHFOptions | None = None,
    *,
    dlpno_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBDLPNOMP2Result:
    """Run local-PNO MP2 on the exact real representation of the B torus.

    The default uses PBC-safe Pipek-Mezey occupied localization. Pair-distance
    screening and local fitting remain disabled because their current
    molecular distance definitions are not invariant across a torus boundary.
    """

    from ...dlpno.mp2 import DLPNOMP2Options, run_dlpno_mp2

    mesh_tuple = _normalise_mesh(system, mesh)
    aux_name = aux_basis or default_aux_for(basis.name)
    # The B-stream finite-torus evidence predates the molecular #140/#448
    # convention sweep. Keep that reference sequence all-electron and pin
    # every pre-sweep threshold rather than inheriting molecular defaults.
    local_options = (
        DLPNOMP2Options(
            localise="pipek-mezey",
            n_frozen=0,
            tcut_pno=1e-8,
            tcut_pno_weak=1e-7,
            tcut_mkn=1e-3,
            tcut_pairs=0.0,
            tcut_pairs_weak=0.0,
        )
        if dlpno_options is None
        else dlpno_options
    )
    if not isinstance(local_options, DLPNOMP2Options):
        raise TypeError("dlpno_options must be a DLPNOMP2Options instance")
    if local_options.localise not in {"wannier", "iao", "pipek-mezey", "none"}:
        raise ValueError(
            "aiccm2026dev-b DLPNO-MP2 requires PBC-safe "
            "localise='wannier', 'iao', 'pipek-mezey', or exact-limit "
            "localise='none'"
        )
    if local_options.tcut_pairs != 0.0 or local_options.tcut_pairs_weak != 0.0:
        raise NotImplementedError(
            "aiccm2026dev-b DLPNO-MP2 pair-distance screening awaits a "
            "minimum-image periodic distance and transition-moment estimator"
        )
    if local_options.local_df:
        raise NotImplementedError(
            "aiccm2026dev-b DLPNO-MP2 local fitting awaits minimum-image "
            "auxiliary domains"
        )
    _validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )

    reference, hf_result = _prepare_b_real_reference(
        system,
        basis,
        mesh_tuple,
        options,
        auxiliary_name=aux_name,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )
    convention = _reference_finite_torus_convention(hf_result)
    localized_hf, localization, localization_result = _localize_b_real_reference(
        reference,
        local_options.localise,
        hf_result=hf_result,
        system=system,
        primitive_basis=basis,
        mesh=mesh_tuple,
    )
    local_space = _local_correlation_space(
        reference,
        localized_hf,
        localization_result,
    )
    solver_options = replace(local_options, localise="none")
    with _patched_overlap(reference.hf.overlap):
        result = run_dlpno_mp2(
            reference.molecule,
            reference.basis,
            localized_hf,
            reference.df,
            solver_options,
        )
    raw_local_correlation = float(result.e_corr)
    complete_space_correction = 0.0
    complete_space_limit = (
        local_options.tcut_pno == 0.0
        and local_options.tcut_pno_weak == 0.0
        and local_options.tcut_mkn == 0.0
    )
    if complete_space_limit:
        exact_correlation = _complete_space_mp2_audit(reference)
        complete_space_correction = exact_correlation - raw_local_correlation
        result.e_pno_correction += complete_space_correction
        result.e_corr = exact_correlation
        result.e_total = float(result.e_hf) + exact_correlation
    n_cells = reference.n_cells
    return AICCM2026DevBDLPNOMP2Result(
        mesh=mesh_tuple,
        n_cyclic_cells=n_cells,
        e_hf_per_cell=float(result.e_hf) / n_cells,
        e_corr_per_cell=float(result.e_corr) / n_cells,
        raw_local_e_corr_per_cell=raw_local_correlation / n_cells,
        complete_space_correction_per_cell=complete_space_correction / n_cells,
        e_total_per_cell=float(result.e_total) / n_cells,
        n_pairs=int(result.n_pairs),
        n_pairs_screened=int(result.n_pairs_screened),
        n_iter=int(result.n_iter),
        converged=bool(result.converged),
        localization=localization,
        cderi_imaginary_residual=reference.cderi_imaginary_residual,
        cderi_symmetry_residual=reference.cderi_symmetry_residual,
        matrix_imaginary_residual=reference.matrix_imaginary_residual,
        localization_result=localization_result,
        local_correlation_space=local_space,
        finite_torus_convention=convention,
        solver_result=result,
        hf_result=hf_result,
    )


def _run_aiccm2026dev_b_dlpno_cc(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: int | Sequence[int],
    options: PeriodicRHFOptions | None,
    *,
    cc_options: object | None,
    with_triples: bool,
    aux_basis: str | None,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
    progress: bool | object | None,
    verbose: int | None,
) -> AICCM2026DevBDLPNOCCSDResult:
    from ...dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd

    mesh_tuple = _normalise_mesh(system, mesh)
    aux_name = aux_basis or default_aux_for(basis.name)
    # As for periodic DLPNO-MP2 above, this default is a historical B-stream
    # reference convention, not an implicit request for molecular defaults.
    local_options = (
        LocalCCSDOptions(
            localise="pipek-mezey",
            n_frozen=0,
            tcut_pno=1e-7,
            tcut_mkn=0.0,
            tcut_pairs=0.0,
            coupling_radius=0.0,
            residual_domain="pair",
            compute_triples=with_triples,
        )
        if cc_options is None
        else cc_options
    )
    if not isinstance(local_options, LocalCCSDOptions):
        raise TypeError("cc_options must be a LocalCCSDOptions instance")
    if local_options.localise not in {"wannier", "iao", "pipek-mezey", "none"}:
        raise ValueError(
            "aiccm2026dev-b DLPNO-CC requires PBC-safe "
            "localise='wannier', 'iao', 'pipek-mezey', or exact-limit "
            "localise='none'"
        )
    if local_options.coupling_radius != 0.0:
        raise NotImplementedError(
            "aiccm2026dev-b DLPNO-CC occupied-distance screening awaits "
            "minimum-image periodic Wannier distances"
        )
    _validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )

    reference, hf_result = _prepare_b_real_reference(
        system,
        basis,
        mesh_tuple,
        options,
        auxiliary_name=aux_name,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )
    convention = _reference_finite_torus_convention(hf_result)
    localized_hf, localization, localization_result = _localize_b_real_reference(
        reference,
        local_options.localise,
        hf_result=hf_result,
        system=system,
        primitive_basis=basis,
        mesh=mesh_tuple,
    )
    local_space = _local_correlation_space(
        reference,
        localized_hf,
        localization_result,
    )
    solver_options = replace(
        local_options,
        localise="none",
        compute_triples=with_triples,
    )
    with _patched_overlap(reference.hf.overlap):
        result = run_local_dlpno_ccsd(
            reference.molecule,
            reference.basis,
            localized_hf,
            reference.df,
            solver_options,
        )
    n_cells = reference.n_cells
    triples = float(result.e_t) if with_triples else 0.0
    total = float(result.e_hf) + float(result.e_corr) + triples
    method = "dlpno-ccsd(t)" if with_triples else "dlpno-ccsd"
    return AICCM2026DevBDLPNOCCSDResult(
        mesh=mesh_tuple,
        n_cyclic_cells=n_cells,
        e_hf_per_cell=float(result.e_hf) / n_cells,
        e_corr_per_cell=float(result.e_corr) / n_cells,
        e_t_per_cell=triples / n_cells,
        e_total_per_cell=total / n_cells,
        n_pairs=int(result.n_pairs),
        n_pairs_screened=int(result.n_screened),
        n_iter=int(result.n_iter),
        converged=bool(result.converged),
        t1_norm=float(result.t1_norm),
        localization=localization,
        with_triples=with_triples,
        triples_mode=str(solver_options.triples_mode),
        cderi_imaginary_residual=reference.cderi_imaginary_residual,
        cderi_symmetry_residual=reference.cderi_symmetry_residual,
        matrix_imaginary_residual=reference.matrix_imaginary_residual,
        localization_result=localization_result,
        local_correlation_space=local_space,
        finite_torus_convention=convention,
        solver_result=result,
        hf_result=hf_result,
        backend=f"aiccm2026dev-b-{method}",
    )


def run_aiccm2026dev_b_dlpno_ccsd(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: int | Sequence[int] = (1, 1, 1),
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBDLPNOCCSDResult:
    """Run local-PNO CCSD on the real finite-torus B Hamiltonian."""

    return _run_aiccm2026dev_b_dlpno_cc(
        system,
        basis,
        mesh,
        options,
        cc_options=cc_options,
        with_triples=False,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def run_aiccm2026dev_b_dlpno_ccsd_t(
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: int | Sequence[int] = (1, 1, 1),
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBDLPNOCCSDResult:
    """Run local-PNO CCSD(T) on the real finite-torus B Hamiltonian."""

    return _run_aiccm2026dev_b_dlpno_cc(
        system,
        basis,
        mesh,
        options,
        cc_options=cc_options,
        with_triples=True,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def _run_aiccm2026dev_b_canonical_cc(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None,
    options: PeriodicRHFOptions | None,
    *,
    with_triples: bool,
    cc_options: object | None,
    aux_basis: str | None,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
    progress: bool | object | None,
    verbose: int | None,
) -> AICCM2026DevBDLPNOCCSDResult:
    """Select the complete-domain limit of the restricted local solver."""

    from ...dlpno.ccsd_local_solver import LocalCCSDOptions

    selected = (
        LocalCCSDOptions(
            localise="none",
            n_frozen=0,
            tcut_pno=0.0,
            tcut_mkn=0.0,
            tcut_pairs=0.0,
            coupling_radius=0.0,
            residual_domain="full",
            compute_triples=with_triples,
            triples_mode="exact",
        )
        if cc_options is None
        else cc_options
    )
    if not isinstance(selected, LocalCCSDOptions):
        raise TypeError("cc_options must be a LocalCCSDOptions instance")
    if (
        selected.localise != "none"
        or selected.tcut_pno != 0.0
        or selected.tcut_mkn != 0.0
        or selected.tcut_pairs != 0.0
        or selected.coupling_radius != 0.0
    ):
        raise ValueError(
            "canonical aiccm2026dev-b CC requires canonical occupieds, "
            "complete PNO domains, all pairs, and complete occupied coupling"
        )
    result = _run_aiccm2026dev_b_dlpno_cc(
        system,
        basis,
        _resolve_lattice_extension(system, lattice_extension),
        options,
        cc_options=replace(selected, compute_triples=with_triples),
        with_triples=with_triples,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )
    method = "ccsd(t)" if with_triples else "ccsd"
    return replace(result, backend=f"aiccm2026dev-b-{method}")


def run_aiccm2026dev_b_ccsd(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBDLPNOCCSDResult:
    """Run complete-domain finite-torus restricted CCSD."""

    return _run_aiccm2026dev_b_canonical_cc(
        system,
        basis,
        lattice_extension,
        options,
        with_triples=False,
        cc_options=cc_options,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def run_aiccm2026dev_b_ccsd_t(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBDLPNOCCSDResult:
    """Run complete-domain finite-torus restricted CCSD(T)."""

    return _run_aiccm2026dev_b_canonical_cc(
        system,
        basis,
        lattice_extension,
        options,
        with_triples=True,
        cc_options=cc_options,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def _run_aiccm2026dev_b_ump2(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None,
    options: PeriodicRHFOptions | None,
    *,
    ump2_options: object,
    aux_basis: str | None,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
    progress: bool | object | None,
    verbose: int | None,
) -> AICCM2026DevBUMP2Result:
    from ...dlpno.ump2 import DLPNOUMP2Options, run_dlpno_ump2

    if not isinstance(ump2_options, DLPNOUMP2Options):
        raise TypeError("ump2_options must be a DLPNOUMP2Options instance")
    if ump2_options.localise not in {"none", "wannier", "iao"}:
        raise ValueError(
            "aiccm2026dev-b UMP2 requires PBC-safe localise='wannier' or "
            "'iao', or exact-limit localise='none'"
        )
    _validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )
    extension = _resolve_lattice_extension(system, lattice_extension)
    aux_name = aux_basis or default_aux_for(basis.name)
    reference, hf_result = _prepare_b_real_ureference(
        system,
        basis,
        extension,
        options,
        auxiliary_name=aux_name,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )
    convention = _reference_finite_torus_convention(hf_result)
    localized_hf, localization, localization_result = _localize_b_real_ureference(
        reference,
        ump2_options.localise,
        hf_result=hf_result,
        system=system,
        primitive_basis=basis,
        mesh=extension,
    )
    solver_options = replace(
        ump2_options,
        localise=("none" if localization == "none" else "external"),
    )
    with _patched_overlap(reference.hf.overlap):
        result = run_dlpno_ump2(
            reference.molecule,
            reference.basis,
            localized_hf,
            reference.df,
            solver_options,
        )
    n_cells = reference.n_cells
    local = localization != "none"
    return AICCM2026DevBUMP2Result(
        mesh=extension,
        n_cyclic_cells=n_cells,
        e_hf_per_cell=float(result.e_hf) / n_cells,
        e_corr_per_cell=float(result.e_corr) / n_cells,
        e_aa_per_cell=float(result.e_aa) / n_cells,
        e_bb_per_cell=float(result.e_bb) / n_cells,
        e_ab_per_cell=float(result.e_ab) / n_cells,
        e_total_per_cell=float(result.e_total) / n_cells,
        n_pairs_aa=int(result.n_pairs_aa),
        n_pairs_bb=int(result.n_pairs_bb),
        n_pairs_ab=int(result.n_pairs_ab),
        converged=bool(result.converged),
        localization=localization,
        cderi_imaginary_residual=reference.cderi_imaginary_residual,
        cderi_symmetry_residual=reference.cderi_symmetry_residual,
        matrix_imaginary_residual=reference.matrix_imaginary_residual,
        finite_torus_convention=convention,
        solver_result=result,
        hf_result=hf_result,
        localization_result=localization_result,
        backend=(
            "aiccm2026dev-b-dlpno-ump2" if local else "aiccm2026dev-b-ri-ump2"
        ),
    )


def run_aiccm2026dev_b_ump2(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    ump2_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBUMP2Result:
    """Run canonical finite-torus RI-UMP2 in the exact PNO limit."""

    from ...dlpno.ump2 import DLPNOUMP2Options

    selected = (
        DLPNOUMP2Options(
            localise="none", n_frozen=0, tcut_pno=0.0, tcut_pairs=0.0
        )
        if ump2_options is None
        else ump2_options
    )
    if selected.localise != "none" or selected.tcut_pno != 0.0:
        raise ValueError(
            "canonical aiccm2026dev-b UMP2 requires localise='none' and tcut_pno=0"
        )
    return _run_aiccm2026dev_b_ump2(
        system,
        basis,
        lattice_extension,
        options,
        ump2_options=selected,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def run_aiccm2026dev_b_dlpno_ump2(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    ump2_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBUMP2Result:
    """Run PBC-localized, pair-natural-orbital UMP2 on the B torus."""

    from ...dlpno.ump2 import DLPNOUMP2Options

    selected = (
        DLPNOUMP2Options(
            localise="wannier", n_frozen=0, tcut_pno=1e-8, tcut_pairs=0.0
        )
        if ump2_options is None
        else ump2_options
    )
    return _run_aiccm2026dev_b_ump2(
        system,
        basis,
        lattice_extension,
        options,
        ump2_options=selected,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def _run_aiccm2026dev_b_uccsd(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None,
    options: PeriodicRHFOptions | None,
    *,
    cc_options: object,
    with_triples: bool,
    aux_basis: str | None,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
    progress: bool | object | None,
    verbose: int | None,
) -> AICCM2026DevBUCCSDResult:
    from ...dlpno.uccsd import DLPNOUCCSDPilotOptions, run_dlpno_uccsd_pilot

    if not isinstance(cc_options, DLPNOUCCSDPilotOptions):
        raise TypeError("cc_options must be a DLPNOUCCSDPilotOptions instance")
    if cc_options.localise not in {"none", "wannier", "iao"}:
        raise ValueError(
            "aiccm2026dev-b UCCSD requires PBC-safe localise='wannier' or "
            "'iao', or exact-limit localise='none'"
        )
    _validate_posthf_numerical_support_cutoffs(
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
    )
    extension = _resolve_lattice_extension(system, lattice_extension)
    aux_name = aux_basis or default_aux_for(basis.name)
    reference, hf_result = _prepare_b_real_ureference(
        system,
        basis,
        extension,
        options,
        auxiliary_name=aux_name,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )
    convention = _reference_finite_torus_convention(hf_result)
    localized_hf, localization, localization_result = _localize_b_real_ureference(
        reference,
        cc_options.localise,
        hf_result=hf_result,
        system=system,
        primitive_basis=basis,
        mesh=extension,
    )
    solver_options = replace(
        cc_options,
        localise=("none" if localization == "none" else "external"),
        compute_triples=with_triples,
    )
    with _patched_overlap(reference.hf.overlap):
        result = run_dlpno_uccsd_pilot(
            reference.molecule,
            reference.basis,
            localized_hf,
            reference.df,
            solver_options,
        )
    n_cells = reference.n_cells
    triples = float(result.e_t) if with_triples else 0.0
    local = localization != "none" or cc_options.tcut_pno > 0.0
    method = "uccsd(t)" if with_triples else "uccsd"
    prefix = "dlpno-" if local else ""
    return AICCM2026DevBUCCSDResult(
        mesh=extension,
        n_cyclic_cells=n_cells,
        e_hf_per_cell=float(result.e_hf) / n_cells,
        e_corr_per_cell=float(result.e_corr) / n_cells,
        e_t_per_cell=triples / n_cells,
        e_total_per_cell=(float(result.e_hf) + float(result.e_corr) + triples)
        / n_cells,
        n_pairs=int(result.n_pairs),
        n_iter=int(result.n_iter),
        converged=bool(result.converged),
        t1_norm=float(result.t1_norm),
        avg_pno=float(result.avg_pno),
        localization=localization,
        with_triples=with_triples,
        cderi_imaginary_residual=reference.cderi_imaginary_residual,
        cderi_symmetry_residual=reference.cderi_symmetry_residual,
        matrix_imaginary_residual=reference.matrix_imaginary_residual,
        finite_torus_convention=convention,
        solver_result=result,
        hf_result=hf_result,
        localization_result=localization_result,
        backend=f"aiccm2026dev-b-{prefix}{method}",
    )


def _canonical_ucc_options(options: object | None, *, triples: bool) -> object:
    from ...dlpno.uccsd import DLPNOUCCSDPilotOptions

    # The finite-torus canonical oracle remains explicitly all-electron even
    # after the molecular frozen-core default changes under issue #140.
    selected = (
        DLPNOUCCSDPilotOptions(
            localise="none",
            n_frozen=0,
            tcut_pno=0.0,
            compute_triples=triples,
        )
        if options is None
        else options
    )
    if selected.localise != "none" or selected.tcut_pno != 0.0:
        raise ValueError(
            "canonical aiccm2026dev-b UCCSD requires localise='none' and tcut_pno=0"
        )
    return selected


def run_aiccm2026dev_b_uccsd(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBUCCSDResult:
    """Run full-domain finite-torus UCCSD with the O(N^6) oracle."""

    return _run_aiccm2026dev_b_uccsd(
        system,
        basis,
        lattice_extension,
        options,
        cc_options=_canonical_ucc_options(cc_options, triples=False),
        with_triples=False,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def run_aiccm2026dev_b_uccsd_t(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBUCCSDResult:
    """Run full-domain finite-torus UCCSD(T) with the O(N^6) oracle."""

    return _run_aiccm2026dev_b_uccsd(
        system,
        basis,
        lattice_extension,
        options,
        cc_options=_canonical_ucc_options(cc_options, triples=True),
        with_triples=True,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def _run_dlpno_ucc(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None,
    options: PeriodicRHFOptions | None,
    *,
    cc_options: object | None,
    triples: bool,
    aux_basis: str | None,
    lattice_cutoff_bohr: float,
    rsgdf_ke_cutoff: float,
    gdf_linear_dep_threshold: float,
    progress: bool | object | None,
    verbose: int | None,
) -> AICCM2026DevBUCCSDResult:
    from ...dlpno.uccsd import DLPNOUCCSDPilotOptions

    # Preserve the pre-#140/#448 periodic pilot convention independently of
    # the molecular DLPNO-UCCSD defaults.
    selected = (
        DLPNOUCCSDPilotOptions(
            localise="wannier",
            n_frozen=0,
            tcut_pno=1e-7,
            compute_triples=triples,
        )
        if cc_options is None
        else cc_options
    )
    return _run_aiccm2026dev_b_uccsd(
        system,
        basis,
        lattice_extension,
        options,
        cc_options=selected,
        with_triples=triples,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def run_aiccm2026dev_b_dlpno_uccsd(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBUCCSDResult:
    """Run the unrestricted local-PNO CCSD correctness pilot."""

    return _run_dlpno_ucc(
        system,
        basis,
        lattice_extension,
        options,
        cc_options=cc_options,
        triples=False,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )


def run_aiccm2026dev_b_dlpno_uccsd_t(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_extension: int | Sequence[int] | None = None,
    options: PeriodicRHFOptions | None = None,
    *,
    cc_options: object | None = None,
    aux_basis: str | None = None,
    lattice_cutoff_bohr: float = 15.0,
    rsgdf_ke_cutoff: float = 200.0,
    gdf_linear_dep_threshold: float = 1e-9,
    progress: bool | object | None = None,
    verbose: int | None = None,
) -> AICCM2026DevBUCCSDResult:
    """Run the unrestricted local-PNO CCSD(T) correctness pilot."""

    return _run_dlpno_ucc(
        system,
        basis,
        lattice_extension,
        options,
        cc_options=cc_options,
        triples=True,
        aux_basis=aux_basis,
        lattice_cutoff_bohr=lattice_cutoff_bohr,
        rsgdf_ke_cutoff=rsgdf_ke_cutoff,
        gdf_linear_dep_threshold=gdf_linear_dep_threshold,
        progress=progress,
        verbose=verbose,
    )
