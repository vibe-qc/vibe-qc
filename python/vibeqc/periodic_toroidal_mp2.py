"""Toroidal (Born-von Kármán) megacell periodic Γ RI-MP2 -- Stages 5a, 5b, 5c.

The toroidal megacell route (maintainer decision 2026-06-17;
``handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md`` Sec. 6) uses a **periodic** supercell
with exact translational symmetry. By the BvK theorem, a Γ-point calculation
on an N-cell supercell is equivalent to an N-point k-mesh calculation on the
unit cell when both calculations use the same finite Hamiltonian, Coulomb /
exchange gauge, and fitted operator.  A cross-code KMP2 calculation with a
different fitted Hamiltonian is therefore not an exact finite-N oracle for this
legacy producer.

Stage 5a establishes the **toroidal canonical MP2**: periodic Γ HF on the
supercell via the GDF-hosted driver with periodic Ewald-J / real-space-K
routing -> C, e; a separately built periodic Γ compensated-cell DF tensor
(Stage 4) -> MO transform (ia|jb); RI-MP2 energy.

Stage 5b adds a **translational pair-family decomposition**: localise -> Wannier
centroids -> cell tiles -> group occupied pairs (i,j) by relative cell L ->
per-L MP2, numerically equal to 5a all-pairs at the pinned tolerance.  The
current implementation still evaluates every pair before partitioning the sum;
representative-only family evaluation remains an efficiency follow-up.

Stage 5c adds **DLPNO-MP2, DLPNO-CCSD, and DLPNO-CCSD(T)** on the toroidal
megacell: adapts the periodic Γ DF tensor + HF reference to the molecular
DLPNO drivers (:func:`~vibeqc.dlpno.mp2.run_dlpno_mp2`,
:func:`~vibeqc.dlpno.ccsd_local_solver.run_local_dlpno_ccsd`) -- the toroidal
counterpart of the open-megacell ``megacell_run_job``.

Scope / boundary
----------------
- Γ-point only (toroidal = Γ on the supercell). Multi-k generalization is Task 3.
- Native vibe-qc Γ driver (``run_rhf_periodic_gamma_gdf``) with its periodic
  Ewald-J / real-space-K branch forced for every toroidal supercell.
- Periodic-HF ``exxdiv='ewald'`` orbital energies used as-is; bare finite-N
  energy is size-dependent and is extrapolated in N. Exact finite-N character-
  mesh equivalence requires a matched Hamiltonian; the matched AICCM-B route
  carries that external KMP2 regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ToroidalMP2Result",
    "ToroidalPairResult",
    "ToroidalDLPNOResult",
    "ToroidalDLPNOCCSDResult",
    "ToroidalPairClassification",
    "build_toroidal_supercell",
    "toroidal_mp2_gamma",
    "toroidal_mp2_gamma_pairs",
    "toroidal_dlpno_mp2",
    "toroidal_dlpno_ccsd",
    "toroidal_dlpno_ccsd_t",
    "classify_toroidal_pairs",
    "ToroidalTDLResult",
    "toroidal_mp2_tdl",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_int_tuple(nrep: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(x) for x in nrep)


@dataclass(frozen=True)
class _ToroidalImageSearch:
    """Prepared lattice data for repeated three-dimensional CVP solves."""

    lattice_scale: float
    reduced_lattice: np.ndarray
    inverse_transform: np.ndarray
    qr_upper: np.ndarray


def _gram_schmidt_columns(lattice: np.ndarray):
    """Return Gram-Schmidt columns, coefficients, and squared norms."""
    ndim = lattice.shape[1]
    orthogonal = np.zeros_like(lattice)
    coefficients = np.zeros((ndim, ndim), dtype=float)
    norms_sq = np.zeros(ndim, dtype=float)
    for column in range(ndim):
        vector = lattice[:, column].copy()
        for previous in range(column):
            norm_sq = norms_sq[previous]
            if not np.isfinite(norm_sq) or norm_sq <= np.finfo(float).tiny:
                raise ValueError(
                    "toroidal minimum-image search requires a full-rank lattice"
                )
            coefficient = (
                float(lattice[:, column] @ orthogonal[:, previous]) / norm_sq
            )
            coefficients[column, previous] = coefficient
            vector -= coefficient * orthogonal[:, previous]
        orthogonal[:, column] = vector
        norms_sq[column] = float(vector @ vector)
    return orthogonal, coefficients, norms_sq


def _unimodular_inverse(transform: np.ndarray) -> np.ndarray:
    """Exact inverse of a three-dimensional integer unimodular matrix."""
    matrix = [[int(transform[row, col]) for col in range(3)] for row in range(3)]
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (
        d * h - e * g
    )
    if abs(determinant) != 1:
        raise ValueError("lattice-basis reduction lost unimodularity")
    adjugate = np.array(
        [
            [e * i - f * h, c * h - b * i, b * f - c * e],
            [f * g - d * i, a * i - c * g, c * d - a * f],
            [d * h - e * g, b * g - a * h, a * e - b * d],
        ],
        dtype=object,
    )
    return determinant * adjugate


def _reduce_lattice_basis(lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LLL-reduce a three-dimensional basis and retain its integer transform."""
    original = np.array(lattice, dtype=float, copy=True)
    reduced = original.copy()
    ndim = reduced.shape[1]
    transform = np.eye(ndim, dtype=object)
    delta = 0.75
    column = 1
    iterations = 0
    while column < ndim:
        iterations += 1
        if iterations > 10_000:
            raise ValueError("toroidal lattice-basis reduction did not converge")

        # Size-reduce the active vector against every earlier vector.  Each
        # update subtracts an integer column multiple, preserving the lattice.
        for previous in range(column - 1, -1, -1):
            working_scale = float(np.max(np.abs(reduced)))
            if not np.isfinite(working_scale) or working_scale <= 0.0:
                raise ValueError(
                    "toroidal minimum-image search requires a full-rank lattice"
                )
            _orthogonal, coefficients, _norms_sq = _gram_schmidt_columns(
                reduced / working_scale
            )
            multiple = int(np.rint(coefficients[column, previous]))
            if multiple:
                reduced[:, column] -= multiple * reduced[:, previous]
                transform[:, column] -= multiple * transform[:, previous]

        working_scale = float(np.max(np.abs(reduced)))
        _orthogonal, coefficients, norms_sq = _gram_schmidt_columns(
            reduced / working_scale
        )
        lovasz_bound = (
            delta - coefficients[column, column - 1] ** 2
        ) * norms_sq[column - 1]
        if norms_sq[column] >= lovasz_bound:
            column += 1
            continue

        reduced[:, [column - 1, column]] = reduced[:, [column, column - 1]]
        transform[:, [column - 1, column]] = transform[
            :, [column, column - 1]
        ]
        column = max(1, column - 1)

    # The in-place float updates above choose the integer transform, but a long
    # shear can accumulate cancellation error in the working basis.  Rebuild
    # B @ U from the exact rational values of the input floats and round only
    # once per element, preserving the lattice represented by the caller.
    reconstructed = np.empty_like(original)
    for row in range(ndim):
        for column in range(ndim):
            exact_value = sum(
                Fraction.from_float(float(original[row, source]))
                * int(transform[source, column])
                for source in range(ndim)
            )
            reconstructed[row, column] = float(exact_value)
    return reconstructed, transform


def _prepare_toroidal_image_search(
    super_lattice: np.ndarray,
) -> _ToroidalImageSearch:
    """Validate and LLL-reduce a torus lattice for exact image searches."""
    input_scale = float(np.max(np.abs(super_lattice)))
    if not np.isfinite(input_scale) or input_scale <= 0.0:
        raise ValueError(
            "classify_toroidal_pairs requires a finite full-rank lattice"
        )
    reduced_physical, transform = _reduce_lattice_basis(super_lattice)
    lattice_scale = float(np.max(np.abs(reduced_physical)))
    reduced_lattice = reduced_physical / lattice_scale
    singular_values = np.linalg.svd(reduced_lattice, compute_uv=False)
    sigma_min = float(singular_values[-1])
    rank_tolerance = (
        np.sqrt(np.finfo(float).eps) * float(singular_values[0])
        if singular_values.size
        else 0.0
    )
    if not np.all(np.isfinite(singular_values)) or sigma_min <= rank_tolerance:
        raise ValueError(
            "classify_toroidal_pairs requires a numerically full-rank lattice"
        )

    _q_matrix, qr_upper = np.linalg.qr(reduced_lattice)
    return _ToroidalImageSearch(
        lattice_scale=lattice_scale,
        reduced_lattice=reduced_lattice,
        inverse_transform=_unimodular_inverse(transform),
        qr_upper=qr_upper,
    )


def _toroidal_minimum_image(
    relative_cell: Sequence[int],
    nrep: Sequence[int],
    search: _ToroidalImageSearch,
) -> np.ndarray:
    """Exact closest Cartesian image of a relative cell on a skew torus.

    With supercell lattice ``B = A @ diag(nrep)`` and fractional supercell
    displacement ``f = L / nrep``, images are ``B @ (f - m)`` for integer
    ``m``.  A unimodular LLL reduction first replaces ``B`` by a shorter basis
    of the same lattice.  QR sphere decoding then enumerates only integer
    points inside the best distance found so far.  Back-substitution bounds
    every level of the three-dimensional search, so the result is finite and
    exact without the potentially enormous Cartesian box used by a
    singular-value bound.
    """
    reps = tuple(int(value) for value in nrep)
    relative = tuple(int(value) for value in relative_cell)
    # If B_reduced = B @ U, reduced coordinates are g = U^-1 f.  Form
    # g modulo integers with exact integer remainders so a huge unimodular
    # shear cannot erase the small fractional displacement in float arithmetic.
    common_denominator = lcm(*reps)
    reduced_fractional = np.array(
        [
            (
                sum(
                    int(search.inverse_transform[row, column])
                    * relative[column]
                    * (common_denominator // reps[column])
                    for column in range(3)
                )
                % common_denominator
            )
            / common_denominator
            for row in range(3)
        ],
        dtype=float,
    )
    return _sphere_decode_min_image(reduced_fractional, search)


def _sphere_decode_min_image(
    reduced_fractional: np.ndarray,
    search: _ToroidalImageSearch,
) -> np.ndarray:
    """Exact closest Cartesian image for reduced fractional coordinates.

    Shared QR sphere-decoding core of :func:`_toroidal_minimum_image`
    (integer relative cells, exact rational preparation) and the
    continuous centroid-difference path
    (:func:`toroidal_pair_distance_fn`).  ``reduced_fractional`` are
    coordinates in the LLL-reduced supercell basis; the search returns
    the Cartesian displacement to the nearest lattice image.
    """
    target = search.qr_upper @ reduced_fractional
    qr_upper = search.qr_upper
    ndim = 3

    # Babai nearest-plane gives a tight finite sphere before the exact
    # branch-and-bound search.  The diagonal is nonzero because the caller has
    # already rejected numerically rank-deficient lattices.
    seed = np.zeros(ndim, dtype=int)
    for axis in range(ndim - 1, -1, -1):
        upper_tail = float(qr_upper[axis, axis + 1 :] @ seed[axis + 1 :])
        centre = (target[axis] - upper_tail) / qr_upper[axis, axis]
        seed[axis] = int(np.rint(centre))

    best_image = seed.copy()
    residual = target - qr_upper @ seed
    best_sq = float(residual @ residual)
    candidate = seed.copy()
    epsilon = np.finfo(float).eps
    best_norm = np.sqrt(max(0.0, best_sq))
    arithmetic_scale = max(1.0, float(np.linalg.norm(target)) + best_norm)
    numerical_slack = 256.0 * (
        epsilon * best_norm * arithmetic_scale
        + epsilon * epsilon * arithmetic_scale * arithmetic_scale
    )

    def _integers_nearest_first(centre: float, lower: int, upper: int):
        nearest = min(max(int(np.rint(centre)), lower), upper)
        yield nearest
        for offset in range(1, max(nearest - lower, upper - nearest) + 1):
            left = nearest - offset
            right = nearest + offset
            choices = []
            if left >= lower:
                choices.append(left)
            if right <= upper:
                choices.append(right)
            yield from sorted(choices, key=lambda value: abs(value - centre))

    def _search(axis: int, partial_sq: float) -> None:
        nonlocal best_image, best_sq
        if axis < 0:
            if partial_sq < best_sq:
                best_sq = partial_sq
                best_image = candidate.copy()
            return

        remaining = best_sq + numerical_slack - partial_sq
        if remaining < 0.0:
            return
        upper_tail = float(
            qr_upper[axis, axis + 1 :] @ candidate[axis + 1 :]
        )
        diagonal = float(qr_upper[axis, axis])
        centre = (target[axis] - upper_tail) / diagonal
        radius = np.sqrt(max(0.0, remaining)) / abs(diagonal)
        rounding_pad = 64.0 * np.finfo(float).eps * max(1.0, abs(centre), radius)
        lower = int(np.ceil(centre - radius - rounding_pad))
        upper = int(np.floor(centre + radius + rounding_pad))
        if lower > upper:
            return

        for value in _integers_nearest_first(centre, lower, upper):
            row_residual = target[axis] - upper_tail - diagonal * value
            next_sq = partial_sq + row_residual * row_residual
            if next_sq <= best_sq + numerical_slack:
                candidate[axis] = value
                _search(axis - 1, next_sq)

    _search(ndim - 1, 0.0)
    reduced_displacement = reduced_fractional - best_image
    return search.lattice_scale * (
        search.reduced_lattice @ reduced_displacement
    )


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ToroidalMP2Result:
    """Toroidal megacell periodic Γ RI-MP2 energy.

    Attributes
    ----------
    nrep : tuple of int
        Supercell replication ``(n1, n2, n3)``.
    n_cells : int
        Number of unit cells in the supercell (``n1.n2.n3``).
    n_atoms : int
        Atoms in the supercell.
    n_bf : int
        Orbital basis functions in the supercell.
    n_occ : int
        Occupied MOs in the supercell (closed-shell: n_elec // 2).
    n_vir : int
        Virtual MOs in the supercell.
    n_fit : int
        Number of DF fit vectors.
    e_hf : float
        Supercell HF energy (Ha, whole supercell).
    e_hf_per_cell : float
        ``e_hf / n_cells`` -- HF energy per unit cell.
    e_corr : float
        Supercell RI-MP2 correlation energy (Ha, whole supercell).
    e_corr_per_cell : float
        ``e_corr / n_cells`` -- per-unit-cell correlation energy. It maps to a
        character-mesh result only for the same finite Hamiltonian and gauge.
    e_total : float
        ``e_hf + e_corr`` (whole supercell).
    aux_basis_name : str
        Auxiliary (fitting) basis used.
    """

    nrep: Tuple[int, int, int]
    n_cells: int
    n_atoms: int
    n_bf: int
    n_occ: int = 0
    n_vir: int = 0
    n_fit: int = 0
    e_hf: float = 0.0
    e_hf_per_cell: float = 0.0
    e_corr: float = 0.0
    e_corr_per_cell: float = 0.0
    e_total: float = 0.0
    aux_basis_name: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_toroidal_supercell(system, nrep: Sequence[int]):
    """Replicate a periodic unit cell into a toroidal (periodic) supercell.

    Unlike :func:`vibeqc.periodic_megacell_mp2.build_supercell_molecule` which
    creates an *open-boundary* finite cluster, this creates a *periodic* supercell
    (the toroidal/BvK boundary conditions): atoms are replicated and the lattice
    vectors are scaled to enclose the full supercell. A Γ-point calculation on
    this supercell is equivalent to an N-point character-mesh calculation on the
    original unit cell when the finite Hamiltonian and gauge are identical (BvK
    theorem).

    Parameters
    ----------
    system : PeriodicSystem
        The periodic unit cell.
    nrep : (int, int, int)
        Replication along each lattice vector.  All components must be >= 1.

    Returns
    -------
    PeriodicSystem
        A toroidal supercell -- a new ``PeriodicSystem`` with scaled lattice and
        replicated atoms, carrying the scaled charge and multiplicity.
    """
    from ._vibeqc_core import Atom, PeriodicSystem

    n1, n2, n3 = _to_int_tuple(nrep[:3])
    if min(n1, n2, n3) < 1:
        raise ValueError(f"nrep components must be >= 1, got {tuple(nrep)}")

    lattice = np.asarray(system.lattice, dtype=float)
    # PeriodicSystem stores Cartesian lattice vectors as columns.  Scaling
    # replication axis i therefore scales column i: A_super = A @ diag(nrep).
    reps = np.array([n1, n2, n3], dtype=float)
    lattice_super = lattice @ np.diag(reps)

    cell_atoms = list(system.unit_cell)
    out_atoms: list = []
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                shift = lattice @ np.array([i, j, k], dtype=float)
                for a in cell_atoms:
                    pos = (np.asarray(a.xyz, dtype=float) + shift).tolist()
                    out_atoms.append(Atom(int(a.Z), pos))

    n_cells = n1 * n2 * n3
    charge = int(system.charge) * n_cells
    # Closed-shell toroidal megacell: an even-electron neutral cell stays
    # closed-shell; carry the unit-cell multiplicity only when n_cells == 1,
    # else default to closed shell (multiplicity 1).
    mult = int(system.multiplicity) if n_cells == 1 else 1
    return PeriodicSystem(
        3,
        lattice_super,
        out_atoms,
        charge=charge,
        multiplicity=mult,
    )


def _canonical_mp2_corr_from_M(
    M: np.ndarray,
    eps_occ: np.ndarray,
    eps_vir: np.ndarray,
    n_occ: int,
    n_vir: int,
) -> Tuple[float, float]:
    """Canonical RI-MP2 correlation ``(e_os, e_ss)`` from the MO-ERI matrix.

    Reference Python implementation of the closed-shell RMP2 energy
    contraction, retained as the parity oracle for the C++ kernel
    (:func:`_canonical_mp2_corr_cpp`).  ``M[(i,a),(j,b)] = (ia|jb)``.

        e_os = S_{ijab}  (ia|jb)^2 / D
        e_ss = S_{ijab} [(ia|jb)^2 - (ia|jb)(ib|ja)] / D
        D    = e_i + e_j - e_a - e_b   (< 0 for bound states)
    """
    e_os = 0.0
    e_ss = 0.0
    for i in range(n_occ):
        eps_i = float(eps_occ[i])
        for j in range(n_occ):
            eps_j = float(eps_occ[j])
            for a in range(n_vir):
                eps_a = float(eps_vir[a])
                ia_base = i * n_vir + a
                for b in range(n_vir):
                    eps_b = float(eps_vir[b])
                    jb = j * n_vir + b
                    iajb = float(M[ia_base, jb])
                    ibja = float(M[i * n_vir + b, j * n_vir + a])
                    denom = eps_i + eps_j - eps_a - eps_b
                    t_os = iajb * iajb / denom
                    e_os += t_os
                    e_ss += t_os - iajb * ibja / denom
    return e_os, e_ss


def _canonical_mp2_corr_cpp(
    B_mo: np.ndarray,
    mo_energies: np.ndarray,
    n_occ: int,
    n_vir: int,
) -> Tuple[float, float]:
    """Canonical RI-MP2 correlation ``(e_os, e_ss)`` via the C++/OpenMP kernel.

    Routes the closed-shell Γ RMP2 contraction through the shared
    ``aiccm2026dev_b_mp2_energy_from_lov`` kernel (the same one the multi-k
    canonical post-HF path uses).  The single-k-point (n_k = 1) reduction of
    that kernel is exactly the toroidal Γ contraction: with
    ``lov[(0,0)] = B[P,i,a]``, ``energies = (eps_occ ‖ eps_vir)`` and
    ``mesh = (1, 1, 1)`` it returns the same ``(e_os, e_ss)`` as
    :func:`_canonical_mp2_corr_from_M`, but in C++ without the
    O(n_occ² · n_vir²) Python loop (and without materialising the
    ``(n_occ·n_vir)²`` MO-ERI matrix).
    """
    from ._vibeqc_core import aiccm2026dev_b_mp2_energy_from_lov

    n_aux = int(B_mo.shape[0])
    lov = {
        (0, 0): np.ascontiguousarray(
            np.asarray(B_mo, dtype=np.complex128).reshape(n_aux, n_occ, n_vir)
        )
    }
    energies = np.ascontiguousarray(
        np.asarray(mo_energies, dtype=float).reshape(1, n_occ + n_vir)
    )
    # Kernel returns (e_ss, e_os, max_imag); denominator tolerance 1e-12
    # matches the multi-k canonical path. For real Γ data max_imag ~ 0.
    e_ss, e_os, _max_imag = aiccm2026dev_b_mp2_energy_from_lov(
        lov, energies, (1, 1, 1), (0, 0, 0), 1.0e-12
    )
    return float(e_os), float(e_ss)


def toroidal_mp2_gamma(
    system,
    basis_name: str,
    nrep: Sequence[int] = (1, 1, 1),
    *,
    aux_basis: Optional[str] = None,
    rhf_options=None,
    _energy_backend: str = "cpp",
) -> ToroidalMP2Result:
    """Toroidal megacell periodic Γ RI-MP2 -- native vibe-qc periodic DF-MP2.

    Builds an ``n1xn2xn3`` **periodic** (toroidal) supercell, runs periodic
    Γ GDF HF (vibe-qc's ``run_rhf_periodic_gamma_gdf``), builds the periodic
    Γ DF 3-index tensor via :func:`vibeqc.periodic_df.build_periodic_gamma_df`,
    transforms it to the MO basis, and computes the canonical RI-MP2 correlation
    energy per unit cell.

    The per-cell correlation energy at supercell size N is the toroidal (BvK)
    finite-N MP2. It is exactly equivalent to an N-point character-mesh MP2 only
    when both routes share the same finite Hamiltonian, gauge, and fitted
    operator. Extrapolate in N (5b+) to approach the TDL.

    Parameters
    ----------
    system : PeriodicSystem
        Periodic unit cell.
    basis_name : str
        Orbital basis (e.g. ``"sto-3g"``, ``"pob-tzvp-rev2"``).
    nrep : (int, int, int)
        Supercell replication.  ``(1,1,1)`` is the unit cell.
    aux_basis : str, optional
        RI auxiliary basis for density fitting.  Auto-selected if omitted
        (the DF-RI auxiliary for *correlation*, distinct from the JK aux).
    rhf_options
        Optional ``PeriodicRHFOptions`` override for the supercell HF.

    Returns
    -------
    ToroidalMP2Result
    """
    from . import BasisSet
    from ._vibeqc_core import PeriodicRHFOptions as _PeriodicRHFOptions
    from .aux_basis import default_aux_for, make_aux_basis_set
    from .periodic_df import build_periodic_gamma_df
    from .periodic_rhf_gdf import run_rhf_periodic_gamma_gdf

    nrep_tup = _to_int_tuple(nrep[:3])
    n_cells = nrep_tup[0] * nrep_tup[1] * nrep_tup[2]

    # 1. Build toroidal supercell.
    super_system = build_toroidal_supercell(system, nrep_tup)

    # 2. Build orbital + auxiliary basis sets.
    mol = super_system.unit_cell_molecule()
    basis = BasisSet(mol, basis_name)

    # Correlation RI auxiliary: default_aux_for(basis_name) returns the *JK-fit*
    # auxiliary (same as the periodic GDF SCF uses).  For correlation we prefer
    # the *RI* auxiliary (e.g. def2-svp-ri).  If the caller supplies aux_basis,
    # use it; otherwise fall back to the JK-fit default (consistent with the
    # periodic_df module, which uses the same default as the SCF driver).
    aux_name = aux_basis or default_aux_for(basis_name)
    aux = make_aux_basis_set(mol, aux_name=aux_name)

    # 3. Periodic Γ GDF HF on the supercell.
    opts = rhf_options if rhf_options is not None else _PeriodicRHFOptions()
    hf_res = run_rhf_periodic_gamma_gdf(
        super_system,
        basis,
        opts,
        aux_basis=aux_name,
        _force_ewald_jk=True,
    )
    if not getattr(hf_res, "converged", False):
        raise RuntimeError(
            f"toroidal_mp2_gamma: periodic Γ HF on the {nrep_tup} "
            "supercell did not converge"
        )

    n_elec = super_system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "toroidal_mp2_gamma: closed-shell RHF requires even electron count"
        )
    n_occ = n_elec // 2
    nbf = hf_res.mo_coeffs.shape[0]
    n_vir = nbf - n_occ
    if n_vir < 1:
        raise RuntimeError(
            "toroidal_mp2_gamma: no virtual orbitals -- basis too small for correlation"
        )

    C_occ = hf_res.mo_coeffs[:, :n_occ]
    C_vir = hf_res.mo_coeffs[:, n_occ:]
    eps_occ = hf_res.mo_energies[:n_occ]
    eps_vir = hf_res.mo_energies[n_occ:]

    # 4. Periodic Γ DF 3-index tensor.
    df = build_periodic_gamma_df(
        super_system,
        basis,
        aux_basis=aux_name,
    )
    n_fit = df.n_fit

    # 5. MO transform -> B[P, i, a] shape = (n_fit, n_occ, n_vir).
    B_mo = df.mo_transform(C_occ, C_vir)  # (n_fit, n_occ, n_vir)

    # 6. RI-MP2 correlation energy -- same formula as the C++ ``run_mp2``
    #    (cpp/src/mp2.cpp, lines 189-225):
    #
    #      (ia|jb) = S_P B[P,i,a] . B[P,j,b]
    #
    #      e_os = S_{i,j,a,b}  (ia|jb)^2 / Δ
    #      e_ss = S_{i,j,a,b} [(ia|jb)^2 - (ia|jb)(ib|ja)] / Δ
    #      e_corr = c_os.e_os + c_ss.e_ss   (c_os=c_ss=1 -> canonical RMP2)
    #
    #    Δ = e_i + e_j - e_a - e_b  (always < 0 for bound states).
    #
    #    Build the (n_occ*n_vir, n_occ*n_vir) MO-ERI block via a single GEMM
    #    (cf. mp2.cpp line 180: ``mo_view = B_mo.transpose() * B_mo``).
    # 6b. Correlation-energy contraction.  Default to the C++/OpenMP kernel
    #     (``aiccm2026dev_b_mp2_energy_from_lov`` reduced to n_k = 1), which
    #     replaces the O(n_occ² · n_vir²) Python loop and never materialises
    #     the (n_occ·n_vir)² MO-ERI matrix.  The Python reference
    #     (``_canonical_mp2_corr_from_M``) is retained for parity testing and
    #     selectable via the private ``_energy_backend`` argument.
    if _energy_backend == "cpp":
        e_os, e_ss = _canonical_mp2_corr_cpp(
            B_mo, hf_res.mo_energies, n_occ, n_vir
        )
    elif _energy_backend == "python":
        nov = n_occ * n_vir
        B_2d = np.asarray(B_mo, dtype=float).reshape(n_fit, nov)
        M = B_2d.T @ B_2d  # (nov, nov); M[(i,a),(j,b)] = (ia|jb)
        M = 0.5 * (M + M.T)  # symmetrise away numerical asymmetry
        e_os, e_ss = _canonical_mp2_corr_from_M(
            M, eps_occ, eps_vir, n_occ, n_vir
        )
    else:
        raise ValueError(
            f"toroidal_mp2_gamma: unknown _energy_backend {_energy_backend!r} "
            "(expected 'cpp' or 'python')"
        )

    e_corr = e_os + e_ss  # c_os = c_ss = 1 (canonical RMP2)

    return ToroidalMP2Result(
        nrep=nrep_tup,
        n_cells=n_cells,
        n_atoms=len(list(mol.atoms)),
        n_bf=nbf,
        n_occ=n_occ,
        n_vir=n_vir,
        n_fit=n_fit,
        e_hf=float(hf_res.energy),
        e_hf_per_cell=float(hf_res.energy) / n_cells,
        e_corr=e_corr,
        e_corr_per_cell=e_corr / n_cells,
        e_total=float(hf_res.energy) + e_corr,
        aux_basis_name=aux_name,
    )


# ---------------------------------------------------------------------------
# Stage 5b -- Translational pair-family decomposition
# ---------------------------------------------------------------------------


def _assign_wannier_cells(
    centroids: np.ndarray,
    lattice_unit: np.ndarray,
    nrep: Tuple[int, int, int],
) -> np.ndarray:
    """Assign each Wannier orbital to a supercell tile via its centroid.

    Fractional coordinates solve ``lattice_unit @ f = centroid`` because
    :class:`PeriodicSystem` stores its Cartesian lattice vectors as columns.
    The *unit-cell* lattice is used (not the supercell lattice), so the floor
    of ``f`` gives the integer cell index (i, j, k).  Wrapped modulo ``nrep``
    so centroids near a boundary don't escape the torus.
    """
    n1, n2, n3 = nrep
    frac = np.linalg.solve(lattice_unit, centroids.T).T  # (n_occ, 3)
    cells = np.floor(frac).astype(int)  # (n_occ, 3)
    cells[:, 0] %= n1
    cells[:, 1] %= n2
    cells[:, 2] %= n3
    return cells


@dataclass
class ToroidalPairResult:
    """Toroidal megacell MP2 with translational pair-family decomposition.

    Groups all occupied-occupied pairs by their relative cell vector ``L``
    (Wannier centroid difference, modulo the supercell replication), computes
    per-``L`` MP2 contributions, and verifies the total agrees with the
    all-pairs :func:`toroidal_mp2_gamma` result to the pinned numerical
    tolerance.  This is a decomposition, not yet representative-only
    evaluation of one pair per family.

    Attributes
    ----------
    nrep, n_cells, n_atoms, n_bf, n_occ, n_vir, n_fit
        Supercell bookkeeping (as in :class:`ToroidalMP2Result`).
    e_hf, e_hf_per_cell, e_total
        HF energies (whole supercell / per cell / total).
    unique_L : int
        Number of unique relative-cell vectors ``L`` among the occupied pairs.
    e_corr_total : float
        Total RI-MP2 correlation energy -- MUST equal the all-pairs value.
    e_corr_per_cell : float
        ``e_corr_total / n_cells``.
    e_corr_by_L : dict of ``(int,int,int) -> float``
        Per-``L`` MP2 contribution (Ha, whole supercell).  ``sum(e_corr_by_L)``
        agrees with ``e_corr_total`` to the pinned numerical tolerance.
    num_pairs_by_L : dict of ``(int,int,int) -> int``
        Number of occupied-occupied pairs ``(i,j)`` at each ``L``.
    wannier_cells : (n_occ, 3) int ndarray
        Cell tile assignment for each canonical occupied orbital (mapped via
        the dominant Wannier partner through the unitary mixing matrix).
    """

    nrep: Tuple[int, int, int]
    n_cells: int
    n_atoms: int
    n_bf: int
    n_occ: int
    n_vir: int
    n_fit: int
    e_hf: float
    e_hf_per_cell: float
    e_total: float
    unique_L: int
    e_corr_total: float
    e_corr_per_cell: float
    e_corr_by_L: Dict[Tuple[int, int, int], float]
    num_pairs_by_L: Dict[Tuple[int, int, int], int]
    wannier_cells: np.ndarray


def toroidal_mp2_gamma_pairs(
    system,
    basis_name: str,
    nrep: Sequence[int] = (1, 1, 1),
    *,
    aux_basis: Optional[str] = None,
    localise_method: str = "boys",
    rhf_options=None,
) -> ToroidalPairResult:
    """Toroidal megacell MP2 with translational pair decomposition (Stage 5b).

    Same calculation as :func:`toroidal_mp2_gamma`, but additionally:

    1. Localises the supercell occupied orbitals (Wannier functions).
    2. Assigns each Wannier orbital to a cell tile via its position centroid.
    3. Maps canonical occupied orbitals to cells through the unitary mixing
       matrix (Wannier rotation), so each canonical i is assigned the cell of
       its dominant localised partner.
    4. Groups all ``(i,j)`` canonical occupied pairs by their relative cell
       vector ``L = cell(j) - cell(i) (mod nrep)``.
    5. Computes per-``L`` MP2 correlation contributions.

    The total ``e_corr_total = sum(e_corr_by_L)`` agrees with the all-pairs
    :func:`toroidal_mp2_gamma` result to 1e-12 Ha; it is the same terms, summed
    in a different partition order.  The per-``L`` breakdown reveals the
    distance decay of the correlation energy and provides infrastructure for a
    future screening-aware DLPNO driver.

    Parameters
    ----------
    system, basis_name, nrep, aux_basis, rhf_options
        As in :func:`toroidal_mp2_gamma`.
    localise_method : str
        ``"boys"`` (default, faithful for non-wrapping orbitals) or
        ``"pipek-mezey"`` (PBC-safe in all regimes).

    Returns
    -------
    ToroidalPairResult
    """
    from . import BasisSet
    from ._vibeqc_core import PeriodicRHFOptions as _PeriodicRHFOptions
    from .aux_basis import default_aux_for, make_aux_basis_set
    from .periodic_df import build_periodic_gamma_df
    from .periodic_localise import localise_periodic_gamma
    from .periodic_rhf_gdf import run_rhf_periodic_gamma_gdf

    nrep_tup = _to_int_tuple(nrep[:3])
    n_cells = nrep_tup[0] * nrep_tup[1] * nrep_tup[2]

    # 1. Build toroidal supercell.
    super_system = build_toroidal_supercell(system, nrep_tup)

    # 2. Basis.
    mol = super_system.unit_cell_molecule()
    basis = BasisSet(mol, basis_name)
    aux_name = aux_basis or default_aux_for(basis_name)
    _aux = make_aux_basis_set(mol, aux_name=aux_name)

    # 3. Periodic Γ GDF HF.
    opts = rhf_options if rhf_options is not None else _PeriodicRHFOptions()
    hf_res = run_rhf_periodic_gamma_gdf(
        super_system,
        basis,
        opts,
        aux_basis=aux_name,
        _force_ewald_jk=True,
    )
    if not getattr(hf_res, "converged", False):
        raise RuntimeError(
            f"toroidal_mp2_gamma_pairs: periodic Γ HF on {nrep_tup} "
            "supercell did not converge"
        )

    n_elec = super_system.n_electrons()
    if n_elec % 2 != 0:
        raise ValueError(
            "toroidal_mp2_gamma_pairs: closed-shell RHF requires even electron count"
        )
    n_occ = n_elec // 2
    nbf = hf_res.mo_coeffs.shape[0]
    n_vir = nbf - n_occ
    if n_vir < 1:
        raise RuntimeError("toroidal_mp2_gamma_pairs: no virtual orbitals")

    C_occ = hf_res.mo_coeffs[:, :n_occ]
    C_vir = hf_res.mo_coeffs[:, n_occ:]
    eps_occ = hf_res.mo_energies[:n_occ]
    eps_vir = hf_res.mo_energies[n_occ:]

    # 4. Periodic Γ DF 3-index tensor + MO transform.
    df = build_periodic_gamma_df(super_system, basis, aux_basis=aux_name)
    n_fit = df.n_fit
    B_mo = df.mo_transform(C_occ, C_vir)

    # 5. MO-ERI matrix M[(i,a),(j,b)] = (ia|jb).
    nov = n_occ * n_vir
    B_2d = np.asarray(B_mo, dtype=float).reshape(n_fit, nov)
    M = B_2d.T @ B_2d
    M = 0.5 * (M + M.T)

    # 6. Localise occupied orbitals -> Wannier functions + centroids.
    loc = localise_periodic_gamma(
        hf_res,
        basis,
        super_system,
        method=localise_method,
        n_occ=n_occ,
    )
    U = np.asarray(loc.U, dtype=float)  # (n_occ, n_occ): C_loc = C_can @ U
    centroids = np.asarray(loc.centroids, dtype=float)  # (n_occ, 3)

    # 7. Assign each Wannier to a cell tile.
    lattice_unit = np.asarray(system.lattice, dtype=float)
    wannier_cell = _assign_wannier_cells(centroids, lattice_unit, nrep_tup)

    # 8. Map canonical orbitals to cells via the unitary mixing matrix.
    #    Canonical i -> cell of its dominant Wannier partner l = argmax |U[i,l]|.
    U_abs = np.abs(U)
    dominant_loc = np.argmax(U_abs, axis=1)  # (n_occ,)
    canonical_cell = wannier_cell[dominant_loc]  # (n_occ, 3)

    # 9. Group canonical occupied pairs by relative cell L, accumulating MP2.
    e_corr_by_L: Dict[Tuple[int, int, int], float] = {}
    num_pairs_by_L: Dict[Tuple[int, int, int], int] = {}

    for i in range(n_occ):
        eps_i = float(eps_occ[i])
        ci = canonical_cell[i]
        for j in range(n_occ):
            eps_j = float(eps_occ[j])
            cj = canonical_cell[j]
            # Relative cell vector L (mod nrep).
            d = cj - ci
            L = (
                int(d[0] % nrep_tup[0]),
                int(d[1] % nrep_tup[1]),
                int(d[2] % nrep_tup[2]),
            )
            e_pair = 0.0
            for a in range(n_vir):
                eps_a = float(eps_vir[a])
                ia_base = i * n_vir + a
                for b in range(n_vir):
                    eps_b = float(eps_vir[b])
                    jb = j * n_vir + b
                    iajb = float(M[ia_base, jb])
                    ibja = float(M[i * n_vir + b, j * n_vir + a])
                    denom = eps_i + eps_j - eps_a - eps_b
                    # (ia|jb).[2(ia|jb) - (ib|ja)] / Δ
                    e_pair += iajb * (2.0 * iajb - ibja) / denom
            e_corr_by_L[L] = e_corr_by_L.get(L, 0.0) + e_pair
            num_pairs_by_L[L] = num_pairs_by_L.get(L, 0) + 1

    e_corr_total = sum(e_corr_by_L.values())

    # Bit-identical gate: compute all-pairs total independently and assert ==.
    e_corr_all = 0.0
    for i in range(n_occ):
        eps_i = float(eps_occ[i])
        for j in range(n_occ):
            eps_j = float(eps_occ[j])
            for a in range(n_vir):
                eps_a = float(eps_vir[a])
                ia_base = i * n_vir + a
                for b in range(n_vir):
                    eps_b = float(eps_vir[b])
                    jb = j * n_vir + b
                    iajb = float(M[ia_base, jb])
                    ibja = float(M[i * n_vir + b, j * n_vir + a])
                    denom = eps_i + eps_j - eps_a - eps_b
                    t_os = iajb * iajb / denom
                    e_corr_all += 2.0 * t_os - iajb * ibja / denom
    if not np.isclose(e_corr_total, e_corr_all, rtol=0, atol=1e-12):
        raise AssertionError(
            f"toroidal_mp2_gamma_pairs: partitioned e_corr_total "
            f"({e_corr_total:.14f}) != all-pairs ({e_corr_all:.14f}), "
            f"Δ = {abs(e_corr_total - e_corr_all):.2e}"
        )

    return ToroidalPairResult(
        nrep=nrep_tup,
        n_cells=n_cells,
        n_atoms=len(list(mol.atoms)),
        n_bf=nbf,
        n_occ=n_occ,
        n_vir=n_vir,
        n_fit=n_fit,
        e_hf=float(hf_res.energy),
        e_hf_per_cell=float(hf_res.energy) / n_cells,
        e_total=float(hf_res.energy) + e_corr_total,
        unique_L=len(e_corr_by_L),
        e_corr_total=e_corr_total,
        e_corr_per_cell=e_corr_total / n_cells,
        e_corr_by_L=e_corr_by_L,
        num_pairs_by_L=num_pairs_by_L,
        wannier_cells=canonical_cell,
    )


# ---------------------------------------------------------------------------
# Stage 5c -- DLPNO pair screening (dipole-approximation classifier)
# ---------------------------------------------------------------------------


@dataclass
class ToroidalPairClassification:
    """Pair classification for a toroidal megacell by relative cell vector L.

    Each unique relative-cell vector ``L`` in the toroidal supercell is assigned
    a classification (strong / weak / distant) based on a dipole-approximation
    pair-energy estimate and a distance cutoff.  It identifies candidate
    families for future DLPNO pair screening; the classifier does not itself
    skip correlation work, and the current toroidal DLPNO wrappers do not
    consume its result.

    Attributes
    ----------
    nrep : tuple of int
        Supercell replication ``(n1, n2, n3)``.
    n_cells : int
        Number of unique L values (= n1.n2.n3).
    tcut_pairs : float
        Strong-pair energy threshold (Ha), passed by the caller.
    r_cutoff : float
        Distance cutoff (Bohr) beyond which pairs are always distant.
    strong_L : list of (int,int,int)
        L families classified as candidates for full PNO treatment.
    weak_L : list of (int,int,int)
        L families classified as candidates for coarse PNO treatment.
    distant_L : list of (int,int,int)
        L families classified as candidates for skipped / dipole-only treatment.
    e_est_by_L : dict of (int,int,int) -> float
        Dipole-approximation pair-energy estimate per L (Ha).
    dist_by_L : dict of (int,int,int) -> float
        Centre-of-cell distance per L (Bohr).
    """

    nrep: Tuple[int, int, int]
    n_cells: int
    tcut_pairs: float
    r_cutoff: float
    strong_L: list
    weak_L: list
    distant_L: list
    e_est_by_L: dict
    dist_by_L: dict

    @property
    def n_strong(self) -> int:
        return len(self.strong_L)

    @property
    def n_weak(self) -> int:
        return len(self.weak_L)

    @property
    def n_distant(self) -> int:
        return len(self.distant_L)

    @property
    def total_pairs_skipped(self) -> int:
        """Backward-compatible alias for :attr:`total_distant_cell_pairs`.

        This counts translational cell-pair placements, not occupied-orbital
        pairs or work already skipped.  The classifier has no orbital-count
        input, and current toroidal DLPNO wrappers do not consume it.
        """
        return self.total_distant_cell_pairs

    @property
    def total_distant_cell_pairs(self) -> int:
        """Distant translational cell-pair placements across the torus."""
        # Each L family has one placement per reference-cell translation.
        return self.n_distant * self.n_cells


def classify_toroidal_pairs(
    system,
    nrep: Sequence[int],
    *,
    tcut_pairs: float = 1e-4,
    r_close: float = 8.0,
    r_cutoff: float = 15.0,
) -> ToroidalPairClassification:
    """Classify toroidal-megacell pair families by relative cell vector L.

    For a toroidal supercell of size ``n1xn2xn3``, there are ``n_cells`` unique
    relative-cell vectors ``L``.  Each ``L`` corresponds to a translational
    family of ``n_cells`` cell-pair placements, with potentially multiple
    occupied-orbital pairs per placement.  Classification is based on the exact
    toroidal minimum-image distance
    ``R = min_m |A L - A diag(nrep) m|`` for integer ``m`` and a
    dipole-approximation pair-energy estimate:

    * **strong** -- ``R <= r_close`` (nearby, always full treatment) or
      ``|E_est| >= tcut_pairs`` (dipole estimate above threshold)
    * **weak** -- ``|E_est| < tcut_pairs`` with
      ``r_close < R < r_cutoff``
    * **distant** -- ``R >= r_cutoff``

    The ``r_close`` guard is essential: the bare dipole approximation
    ``1/R⁶`` underestimates the correlation between adjacent cells by
    orders of magnitude (a calibrated prefactor is system-dependent).
    Pairs within ``r_close`` Bohr are always strong -- the actual MP2
    correlation there is chemically significant regardless of the
    dipole estimate.

    This is the screening infrastructure for Stage 5c DLPNO: distant
    families identify the work that a screening-aware correlation driver could
    skip.  This classifier itself performs no correlation work, and the current
    toroidal DLPNO wrappers do not yet consume it.

    Parameters
    ----------
    system : PeriodicSystem
        The unit cell (provides lattice vectors).
    nrep : (int, int, int)
        Toroidal supercell replication.
    tcut_pairs : float
        Strong-pair energy threshold (Ha).  Default ``1e-4`` ≈ NormalPNO.
    r_close : float
        Distance (Bohr) within which pairs are always strong, regardless
        of the dipole estimate.  Must satisfy ``0 <= r_close < r_cutoff``.
        Default ``8.0`` Bohr ≈ 4.2 Å.
    r_cutoff : float
        Distance cutoff (Bohr).  Pairs at >= ``r_cutoff`` are always distant.
        Default ``15.0`` Bohr ≈ 7.9 Å.

    Returns
    -------
    ToroidalPairClassification

    Raises
    ------
    ValueError
        If replication or distance thresholds are invalid, or if the lattice
        remains numerically rank-deficient after unimodular basis reduction.
    """
    n1, n2, n3 = (int(x) for x in nrep[:3])
    if min(n1, n2, n3) < 1:
        raise ValueError(f"nrep components must be >= 1, got {tuple(nrep)}")
    if not 0.0 <= r_close < r_cutoff:
        raise ValueError(
            "pair-distance thresholds must satisfy 0 <= r_close < r_cutoff, "
            f"got r_close={r_close} and r_cutoff={r_cutoff}"
        )
    n_cells = n1 * n2 * n3
    lattice = np.asarray(system.lattice, dtype=float)  # columns are a1, a2, a3
    reps = np.array([n1, n2, n3], dtype=float)
    super_lattice = lattice @ np.diag(reps)
    image_search = _prepare_toroidal_image_search(super_lattice)

    # Enumerate all unique L values on the torus.
    e_est_by_L: dict = {}
    dist_by_L: dict = {}
    for dx in range(n1):
        for dy in range(n2):
            for dz in range(n3):
                shift = _toroidal_minimum_image(
                    (dx, dy, dz), (n1, n2, n3), image_search
                )
                R = float(np.linalg.norm(shift))
                L = (dx, dy, dz)
                dist_by_L[L] = R
                if R < 1e-10:
                    e_est_by_L[L] = -1.0  # same-cell: always strong
                else:
                    e_est_by_L[L] = -1.0 / (R**6)  # dipole approximation

    # Classify L families.
    tcp = abs(tcut_pairs)
    strong_L: list = []
    weak_L: list = []
    distant_L: list = []

    for L in sorted(e_est_by_L.keys()):
        e_est = e_est_by_L[L]
        R = dist_by_L[L]
        if R <= r_close:
            strong_L.append(L)  # nearby: always strong
        elif R >= r_cutoff:
            distant_L.append(L)
        elif abs(e_est) >= tcp:
            strong_L.append(L)
        else:
            weak_L.append(L)  # dipole estimate below threshold but not cutoff

    return ToroidalPairClassification(
        nrep=(n1, n2, n3),
        n_cells=n_cells,
        tcut_pairs=tcp,
        r_cutoff=r_cutoff,
        strong_L=strong_L,
        weak_L=weak_L,
        distant_L=distant_L,
        e_est_by_L=e_est_by_L,
        dist_by_L=dist_by_L,
    )


def toroidal_pair_distance_fn(system, nrep: Sequence[int]):
    """Exact minimum-image centroid-pair distance on the toroidal supercell.

    Returns a callable ``distance(r_i, r_j) -> float`` (bohr) computing the
    EXACT toroidal minimum-image distance
    ``min_m |(r_j - r_i) - A_super m|`` over integer images ``m`` of the
    supercell lattice ``A_super = A @ diag(nrep)``, via the same LLL
    reduction + QR sphere decoding as :func:`classify_toroidal_pairs`.

    This is the distance the DLPNO drivers must use on a toroidal
    reference: the open-cluster Euclidean centroid distance OVERESTIMATES
    the separation of wrap-around pairs (up to the half-super-period), so
    the molecular drivers' distant-pair dipole screen and coupling-radius
    sets would falsely screen / under-couple pairs that are adjacent on
    the torus.  Pass the returned callable as
    ``DLPNOMP2Options.pair_distance_fn`` /
    ``LocalCCSDOptions.pair_distance_fn`` -- the toroidal DLPNO wrappers
    in this module do so by default.

    Parameters
    ----------
    system : PeriodicSystem
        The unit cell (provides lattice vectors, columns = a1, a2, a3).
    nrep : (int, int, int)
        Toroidal supercell replication.

    Returns
    -------
    callable
        ``distance(r_i, r_j) -> float`` (bohr, minimum-image).
    """
    n1, n2, n3 = (int(x) for x in nrep[:3])
    if min(n1, n2, n3) < 1:
        raise ValueError(f"nrep components must be >= 1, got {tuple(nrep)}")
    lattice = np.asarray(system.lattice, dtype=float)
    super_lattice = lattice @ np.diag([float(n1), float(n2), float(n3)])
    search = _prepare_toroidal_image_search(super_lattice)
    super_inv = np.linalg.inv(super_lattice)
    # Reduced coordinates: B_reduced = B @ U  =>  g = U^-1 (B^-1 d).
    to_reduced = np.asarray(search.inverse_transform, dtype=float) @ super_inv

    def _distance(r_i, r_j) -> float:
        d = np.asarray(r_j, dtype=float).reshape(3) - np.asarray(
            r_i, dtype=float
        ).reshape(3)
        g = to_reduced @ d
        g_frac = g - np.floor(g)
        shift = _sphere_decode_min_image(g_frac, search)
        return float(np.linalg.norm(shift))

    return _distance


# ---------------------------------------------------------------------------
# Stage 5c -- DLPNO local approximation on the toroidal megacell
# ---------------------------------------------------------------------------


class _PeriodicDFAdapter:
    """Adapt :class:`PeriodicGammaDF` to the molecular :class:`DensityFitting`
    interface expected by the DLPNO driver.

    The periodic ``Lpq`` tensor IS the Cholesky-rotated B-tensor (the "fitted
    cderi"), so ``self.B = self.three_center = Lpq``.  The metric is identity
    (B is already Cholesky-orthogonal).  ``mo_transform`` delegates directly.
    """

    def __init__(self, periodic_df, aux_basis_set):
        self._pdf = periodic_df
        self.aux_basis = aux_basis_set
        self.n_orb = periodic_df.nbf
        self.n_aux = periodic_df.n_fit
        # Lpq is already Cholesky-rotated (the fitted cderi).
        L = periodic_df.three_center
        self.B = L
        self.three_center = L
        self.metric = np.eye(self.n_aux)

    def mo_transform(self, C_left, C_right=None):
        return self._pdf.mo_transform(C_left, C_right)


def _pre_sweep_toroidal_dlpno_mp2_options():
    """Return the periodic DLPNO-MP2 convention predating issues #140/#448."""
    from .dlpno.mp2 import DLPNOMP2Options

    return DLPNOMP2Options(
        n_frozen=0,
        tcut_pno=1e-8,
        tcut_pno_weak=1e-7,
        tcut_mkn=1e-3,
        tcut_pairs=1e-6,
        tcut_pairs_weak=1e-4,
    )


def _pre_sweep_toroidal_dlpno_cc_options():
    """Return the periodic local-CC convention predating issues #140/#448."""
    from .dlpno.ccsd_local_solver import LocalCCSDOptions

    return LocalCCSDOptions(
        n_frozen=0,
        tcut_pno=1e-7,
        tcut_mkn=0.0,
        tcut_pairs=1e-4,
        residual_domain="pair",
    )


@dataclass
class ToroidalDLPNOResult:
    """Toroidal megacell DLPNO-MP2 correlation energy -- Stage 5c.

    Runs the molecular DLPNO-MP2 driver on the toroidal supercell, using the
    periodic Γ GDF HF reference and the periodic Γ DF tensor (adapted to the
    molecular DensityFitting interface).  The result is the **total** supercell
    DLPNO-MP2 correlation energy; divide by ``n_cells`` for the per-cell value.

    Attributes
    ----------
    nrep, n_cells, n_atoms, n_bf, n_occ, n_vir, n_fit
        Supercell bookkeeping.
    e_hf : float
        Periodic Γ GDF HF energy (Ha, whole supercell).
    e_hf_per_cell : float
        ``e_hf / n_cells``.
    e_corr : float
        DLPNO-MP2 correlation energy (Ha, whole supercell).
    e_corr_per_cell : float
        ``e_corr / n_cells`` -- per-unit-cell DLPNO-MP2 correlation.
    e_total : float
        ``e_hf + e_corr``.
    dlpno_result : DLPNOMP2Result
        Raw molecular DLPNO-MP2 result (pair energies, PNO counts, etc.).
    pair_classification : ToroidalPairClassification
        The :func:`classify_toroidal_pairs` family view consumed by this
        run.  The driver's distant-pair screen uses the matching toroidal
        minimum-image distance (:func:`toroidal_pair_distance_fn`), so
        wrap-around pairs are never screened by their open-cluster
        Euclidean separation.
    """

    nrep: Tuple[int, int, int]
    n_cells: int
    n_atoms: int
    n_bf: int
    n_occ: int
    n_vir: int
    n_fit: int
    e_hf: float
    e_hf_per_cell: float
    e_corr: float
    e_corr_per_cell: float
    e_total: float
    dlpno_result: object = None  # DLPNOMP2Result
    pair_classification: object = None  # ToroidalPairClassification


def toroidal_dlpno_mp2(
    system,
    basis_name: str,
    nrep: Sequence[int] = (1, 1, 1),
    *,
    aux_basis: Optional[str] = None,
    rhf_options=None,
    dlpno_options=None,
) -> ToroidalDLPNOResult:
    """Toroidal megacell DLPNO-MP2 via molecular DLPNO on the periodic reference.

    Builds the toroidal supercell, runs periodic Γ GDF HF, builds the periodic
    Γ DF 3-index tensor, adapts both to the molecular DLPNO-MP2 interface, and
    runs the full molecular DLPNO-MP2 driver (:func:`vibeqc.dlpno.mp2.run_dlpno_mp2`).

    This is the toroidal counterpart of the open-megacell
    ``megacell_run_job(method="dlpno-mp2")``: the supercell is a **periodic**
    (not open-boundary) cluster, so the HF reference includes Ewald/Madelung
    stabilisation and the DF integrals are the periodic Γ tensor.  The per-cell
    DLPNO-MP2 energy converges to the thermodynamic limit as the supercell grows.

    Parameters
    ----------
    system, basis_name, nrep, aux_basis, rhf_options
        As in :func:`toroidal_mp2_gamma`.
    dlpno_options : DLPNOMP2Options, optional
        Options for the DLPNO-MP2 driver. The periodic default retains the
        pre-#448 settings (Boys localisation, tcut_pno=1e-8,
        tcut_pairs=1e-6); pass an options object to select another convention.

    Returns
    -------
    ToroidalDLPNOResult
    """
    from . import BasisSet
    from ._vibeqc_core import PeriodicRHFOptions as _PeriodicRHFOptions
    from .aux_basis import default_aux_for, make_aux_basis_set
    from .dlpno.mp2 import run_dlpno_mp2
    from .periodic_df import build_periodic_gamma_df
    from .periodic_rhf_gdf import run_rhf_periodic_gamma_gdf

    nrep_tup = _to_int_tuple(nrep[:3])
    n_cells = nrep_tup[0] * nrep_tup[1] * nrep_tup[2]

    # 1. Build toroidal supercell.
    super_system = build_toroidal_supercell(system, nrep_tup)

    # 2. Molecular data for the DLPNO driver.
    mol = super_system.unit_cell_molecule()
    basis = BasisSet(mol, basis_name)
    aux_name = aux_basis or default_aux_for(basis_name)
    aux = make_aux_basis_set(mol, aux_name=aux_name)

    # 3. Periodic Γ GDF HF -- the toroidal reference.
    opts = rhf_options if rhf_options is not None else _PeriodicRHFOptions()
    hf_res = run_rhf_periodic_gamma_gdf(
        super_system,
        basis,
        opts,
        aux_basis=aux_name,
        _force_ewald_jk=True,
    )
    if not getattr(hf_res, "converged", False):
        raise RuntimeError(
            f"toroidal_dlpno_mp2: periodic Γ HF on {nrep_tup} "
            "supercell did not converge"
        )

    # 4. Periodic Γ DF 3-index tensor -> adapted to DensityFitting interface.
    pdf = build_periodic_gamma_df(super_system, basis, aux_basis=aux_name)
    df = _PeriodicDFAdapter(pdf, aux)

    # 5. Run molecular DLPNO-MP2 with periodic reference.
    #    The DLPNO code internally calls compute_overlap(basis), which returns
    #    the molecular (home-cell-only) overlap.  The periodic MOs are
    #    S_periodic-orthonormal, not S_molecular-orthonormal (they differ by
    #    image-cell AO overlaps).  We patch the overlap to use the periodic
    #    S(Γ) from the HF result so that all projections and orthogonalizations
    #    in the DLPNO code are consistent with the periodic MO basis.
    from contextlib import contextmanager

    import vibeqc._vibeqc_core as _core

    _S_periodic = np.asarray(hf_res.overlap, dtype=float)
    _orig_compute_overlap = _core.compute_overlap

    @contextmanager
    def _patched_overlap():
        def _patched(basis_arg):
            return _S_periodic

        _core.compute_overlap = _patched
        try:
            yield
        finally:
            _core.compute_overlap = _orig_compute_overlap

    # 6. Consume the toroidal pair classification: the driver's distant-pair
    #    dipole screen must use the toroidal minimum-image distance, not the
    #    open-cluster Euclidean centroid distance (which overestimates
    #    wrap-around separations and would falsely screen adjacent-on-torus
    #    pairs). A caller-supplied pair_distance_fn wins.
    import dataclasses as _dataclasses

    # Preserve the finite-torus validation convention: its existing causal
    # pair-screening and per-cell energy anchors use the pre-#448 thresholds
    # and are all-electron evidence, separate from the molecular #140/#448
    # convention sweep.
    dlpno_opts = (
        dlpno_options
        if dlpno_options is not None
        else _pre_sweep_toroidal_dlpno_mp2_options()
    )
    if getattr(dlpno_opts, "pair_distance_fn", None) is None:
        dlpno_opts = _dataclasses.replace(
            dlpno_opts,
            pair_distance_fn=toroidal_pair_distance_fn(system, nrep_tup),
        )
    classification = classify_toroidal_pairs(system, nrep_tup)

    with _patched_overlap():
        dlpno_res = run_dlpno_mp2(mol, basis, hf_res, df, options=dlpno_opts)

    n_elec = super_system.n_electrons()
    n_occ = n_elec // 2
    nbf = hf_res.mo_coeffs.shape[0]
    n_vir = nbf - n_occ

    return ToroidalDLPNOResult(
        nrep=nrep_tup,
        n_cells=n_cells,
        n_atoms=len(list(mol.atoms)),
        n_bf=nbf,
        n_occ=n_occ,
        n_vir=n_vir,
        n_fit=pdf.n_fit,
        e_hf=float(hf_res.energy),
        e_hf_per_cell=float(hf_res.energy) / n_cells,
        e_corr=float(dlpno_res.e_corr),
        e_corr_per_cell=float(dlpno_res.e_corr) / n_cells,
        e_total=float(hf_res.energy) + float(dlpno_res.e_corr),
        dlpno_result=dlpno_res,
        pair_classification=classification,
    )


# ---------------------------------------------------------------------------
# Stage 5c extended -- DLPNO-CCSD and DLPNO-CCSD(T) on the toroidal megacell
# ---------------------------------------------------------------------------


@dataclass
class ToroidalDLPNOCCSDResult:
    """Toroidal megacell DLPNO-CCSD/(T) correlation energy."""

    nrep: Tuple[int, int, int]
    n_cells: int
    n_atoms: int
    n_bf: int
    n_occ: int
    n_vir: int
    n_fit: int
    e_hf: float
    e_hf_per_cell: float
    e_corr: float
    e_corr_per_cell: float
    e_t: Optional[float]
    e_t_per_cell: Optional[float]
    e_total: float
    cc_result: object = None
    pair_classification: object = None  # ToroidalPairClassification


def _run_toroidal_dlpno_cc(
    system,
    basis_name: str,
    nrep: Sequence[int],
    *,
    aux_basis: Optional[str] = None,
    rhf_options=None,
    cc_options=None,
    with_triples: bool = False,
):
    """Shared setup for DLPNO-CCSD/(T) on the toroidal megacell."""
    from contextlib import contextmanager

    import vibeqc._vibeqc_core as _core

    from . import BasisSet
    from ._vibeqc_core import PeriodicRHFOptions as _PeriodicRHFOptions
    from .aux_basis import default_aux_for, make_aux_basis_set
    from .dlpno.ccsd_local_solver import run_local_dlpno_ccsd
    from .periodic_df import build_periodic_gamma_df
    from .periodic_rhf_gdf import run_rhf_periodic_gamma_gdf

    nrep_tup = _to_int_tuple(nrep[:3])
    n_cells = nrep_tup[0] * nrep_tup[1] * nrep_tup[2]

    super_system = build_toroidal_supercell(system, nrep_tup)
    mol = super_system.unit_cell_molecule()
    basis = BasisSet(mol, basis_name)
    aux_name = aux_basis or default_aux_for(basis_name)
    aux = make_aux_basis_set(mol, aux_name=aux_name)

    opts = rhf_options if rhf_options is not None else _PeriodicRHFOptions()
    hf_res = run_rhf_periodic_gamma_gdf(
        super_system,
        basis,
        opts,
        aux_basis=aux_name,
        _force_ewald_jk=True,
    )
    if not getattr(hf_res, "converged", False):
        raise RuntimeError(
            f"toroidal DLPNO-CC: periodic Γ HF on {nrep_tup} supercell did not converge"
        )

    pdf = build_periodic_gamma_df(super_system, basis, aux_basis=aux_name)
    df = _PeriodicDFAdapter(pdf, aux)

    _S_periodic = np.asarray(hf_res.overlap, dtype=float)
    _orig = _core.compute_overlap

    @contextmanager
    def _patch():
        _core.compute_overlap = lambda _: _S_periodic
        try:
            yield
        finally:
            _core.compute_overlap = _orig

    # Consume the toroidal pair classification: the CC solver's
    # coupling-radius occupied sets must use the toroidal minimum-image
    # distance (open-cluster Euclidean distances under-couple wrap-around
    # pairs). A caller-supplied pair_distance_fn wins.
    import dataclasses as _dataclasses

    # As above, the toroidal correctness ratchet retains its pre-#448
    # thresholds and remains explicitly all-electron until its own convention
    # sweep is reviewed.
    cc_opts = (
        cc_options
        if cc_options is not None
        else _pre_sweep_toroidal_dlpno_cc_options()
    )
    if getattr(cc_opts, "pair_distance_fn", None) is None:
        cc_opts = _dataclasses.replace(
            cc_opts,
            pair_distance_fn=toroidal_pair_distance_fn(system, nrep_tup),
        )
    if with_triples:
        cc_opts.compute_triples = True
    classification = classify_toroidal_pairs(system, nrep_tup)

    with _patch():
        cc_res = run_local_dlpno_ccsd(mol, basis, hf_res, df, cc_opts)

    n_elec = super_system.n_electrons()
    n_occ = n_elec // 2
    nbf = hf_res.mo_coeffs.shape[0]
    n_vir = nbf - n_occ
    e_t = float(getattr(cc_res, "e_t", 0.0) or 0.0) if with_triples else None

    return ToroidalDLPNOCCSDResult(
        nrep=nrep_tup,
        n_cells=n_cells,
        n_atoms=len(list(mol.atoms)),
        n_bf=nbf,
        n_occ=n_occ,
        n_vir=n_vir,
        n_fit=pdf.n_fit,
        e_hf=float(hf_res.energy),
        e_hf_per_cell=float(hf_res.energy) / n_cells,
        e_corr=float(cc_res.e_corr),
        e_corr_per_cell=float(cc_res.e_corr) / n_cells,
        e_t=e_t,
        e_t_per_cell=e_t / n_cells if e_t is not None else None,
        e_total=float(hf_res.energy) + float(cc_res.e_corr) + (e_t or 0.0),
        cc_result=cc_res,
        pair_classification=classification,
    )


def toroidal_dlpno_ccsd(
    system,
    basis_name: str,
    nrep: Sequence[int] = (1, 1, 1),
    *,
    aux_basis=None,
    rhf_options=None,
    cc_options=None,
) -> ToroidalDLPNOCCSDResult:
    """Toroidal megacell DLPNO-CCSD."""
    return _run_toroidal_dlpno_cc(
        system,
        basis_name,
        nrep,
        aux_basis=aux_basis,
        rhf_options=rhf_options,
        cc_options=cc_options,
        with_triples=False,
    )


def toroidal_dlpno_ccsd_t(
    system,
    basis_name: str,
    nrep: Sequence[int] = (1, 1, 1),
    *,
    aux_basis=None,
    rhf_options=None,
    cc_options=None,
) -> ToroidalDLPNOCCSDResult:
    """Toroidal megacell DLPNO-CCSD(T)."""
    return _run_toroidal_dlpno_cc(
        system,
        basis_name,
        nrep,
        aux_basis=aux_basis,
        rhf_options=rhf_options,
        cc_options=cc_options,
        with_triples=True,
    )


# ---------------------------------------------------------------------------
# Toroidal TDL extrapolation
# ---------------------------------------------------------------------------


@dataclass
class ToroidalTDLResult:
    """Thermodynamic-limit per-cell MP2 correlation from a toroidal size series.

    For a toroidal (BvK) megacell the total correlation energy is
    E_corr(N) = b*N + a to leading order: b is the bulk per-cell
    correlation energy and a the size-independent boundary term.
    """
    sizes: Tuple[int, ...]
    e_corr_total: Tuple[float, ...]
    e_corr_per_cell: Tuple[float, ...]
    increments: Tuple[float, ...]
    e_corr_per_cell_bulk: float
    edge_correction: float
    fit_max_residual: float


def toroidal_mp2_tdl(
    system,
    basis_name: str,
    sizes: Sequence[int] = (2, 3, 4, 5, 6),
    *,
    axis: int = 2,
    **toroidal_kwargs,
) -> ToroidalTDLResult:
    """Extrapolate toroidal megacell MP2 to the thermodynamic limit.

    Runs toroidal_mp2_gamma for a 1-D size series along axis and
    linear-fits E_corr_total(N) = b*N + a.  The slope b is the bulk
    per-cell correlation energy.
    """
    sizes = [int(n) for n in sizes]
    if len(set(sizes)) < 2:
        raise ValueError("toroidal_mp2_tdl needs >= 2 distinct sizes for a fit")
    ax = int(axis)

    totals, per_cell = [], []
    for n in sizes:
        nrep = [1, 1, 1]
        nrep[ax] = n
        r = toroidal_mp2_gamma(system, basis_name, tuple(nrep), **toroidal_kwargs)
        totals.append(r.e_corr)
        per_cell.append(r.e_corr_per_cell)

    ns = np.asarray(sizes, dtype=float)
    tot = np.asarray(totals, dtype=float)
    coef, *_ = np.linalg.lstsq(np.vstack([ns, np.ones_like(ns)]).T, tot, rcond=None)
    b, a = float(coef[0]), float(coef[1])
    resid = float(np.max(np.abs(tot - (b * ns + a))))
    increments = tuple(
        float(totals[i] - totals[i - 1]) for i in range(1, len(totals))
    )

    return ToroidalTDLResult(
        sizes=tuple(sizes),
        e_corr_total=tuple(totals),
        e_corr_per_cell=tuple(per_cell),
        increments=increments,
        e_corr_per_cell_bulk=b,
        edge_correction=a,
        fit_max_residual=resid,
    )
