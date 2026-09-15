"""Occupied localization on the AICCM2026DEV-B finite translation group.

This module is deliberately B-stream specific.  It transforms the occupied
Bloch blocks on the complete cyclic character mesh to the real finite torus
and applies a unitary rotation of that *entire* occupied space.  Consequently
the occupied projector, density, and any SCF energy functional of that density
are unchanged.

Two experimental gauges are provided:

``wannier``
    Jointly diagonalizes projected circular AO-centre operators.  This is the
    finite-AO-centre approximation to the Marzari--Vanderbilt spread, not the
    exact continuum functional: the latter needs cross-k matrix elements of
    ``exp(i G.r)``, which the integral layer does not yet expose.

``iao``
    Builds Knizia intrinsic atomic orbitals from an ANO-RCC-MB reference and
    maximizes their atomic populations.  The default cross-basis overlap is a
    finite-supercell approximation; its boundary error is covered by the same
    wrap-around diagnostic as the Wannier path.

Metallic/entangled occupied spaces are rejected.  No disentanglement is
silently attempted.

References
----------
* Marzari and Vanderbilt, Phys. Rev. B 56, 12847 (1997),
  doi:10.1103/PhysRevB.56.12847.
* Knizia, J. Chem. Theory Comput. 9, 4834 (2013),
  doi:10.1021/ct400687b.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Sequence
import warnings

import numpy as np

from ..._vibeqc_core import (
    BasisSet,
    PeriodicSystem,
    aiccm2026dev_b_canonical_wannier_coefficients,
    aiccm2026dev_b_localization_projector_audit,
    aiccm2026dev_b_matrix_to_real_supercell,
)
from ...properties import _shell_nao, _shell_to_atom
from .scf import _normalise_mesh, cyclic_gamma_mesh

__all__ = [
    "AICCM2026DevBLocalizationMethod",
    "AICCM2026DevBLocalizationResult",
    "AICCM2026DevBUnrestrictedLocalizationResult",
    "AICCM2026DevBLocalizationWarning",
    "localize_aiccm2026dev_b_occupied",
    "localize_aiccm2026dev_b_occupied_blocks",
    "localize_aiccm2026dev_b_unrestricted_occupied",
]


class AICCM2026DevBLocalizationMethod(str, Enum):
    """Available occupied-space gauges."""

    WANNIER = "wannier"
    IAO = "iao"


class AICCM2026DevBLocalizationWarning(UserWarning):
    """The finite torus is too short to resolve a localized-orbital tail."""


@dataclass(frozen=True)
class AICCM2026DevBLocalizationResult:
    """Localized occupied orbitals on the real BvK torus.

    ``coefficients`` has shape ``(N*nbf, N*nocc)`` and is the representation
    consumed by the B-stream local-correlation route.  Its columns are
    orthonormal in ``overlap``.  Coordinates and spreads are in bohr and
    bohr squared, respectively.
    """

    method: str
    coefficients: np.ndarray
    canonical_wannier_coefficients: np.ndarray
    unitary: np.ndarray
    overlap: np.ndarray
    fock: np.ndarray | None
    translations: np.ndarray
    translation_permutations: np.ndarray
    centers_fractional: np.ndarray
    centers_bohr: np.ndarray
    spreads_bohr2: np.ndarray
    atomic_populations: np.ndarray
    aliasing_fraction: np.ndarray
    aliasing_detected: np.ndarray
    objective_initial: float
    objective_final: float
    density_invariance_error: float
    one_particle_energy_error: float
    orthonormality_error: float
    unitary_error: float
    translation_projector_error: float
    translation_covariance_error: float
    real_gauge_residual: float
    band_gap_hartree: float
    n_cells: int
    n_occ_per_cell: int
    iao_reference_basis: str | None = None
    iao_orthonormality_error: float | None = None
    spread_model: str = "projected circular AO-centre approximation"


@dataclass(frozen=True)
class AICCM2026DevBUnrestrictedLocalizationResult:
    """Independent occupied-space gauges for alpha and beta spin.

    UHF permits different occupied projectors for the two spins.  A common
    rotation would therefore be an artificial constraint; each projector is
    localized independently and both invariance audits must pass.  ``beta``
    is ``None`` for a fully spin-polarized reference.
    """

    alpha: AICCM2026DevBLocalizationResult
    beta: AICCM2026DevBLocalizationResult | None
    density_invariance_error: float
    one_particle_energy_error: float


def _translations(mesh: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(list(product(*(range(value) for value in mesh))), dtype=int)


def _hermitian_power(matrix: np.ndarray, power: float, threshold: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    if values[-1] <= 0.0 or values[0] <= threshold * values[-1]:
        raise np.linalg.LinAlgError(
            "AICCM2026DEV-B localization encountered a linearly dependent metric"
        )
    return (vectors * values[None, :] ** power) @ vectors.conj().T


def _real_space_matrix(
    matrices_k: Sequence[np.ndarray], phases: np.ndarray
) -> np.ndarray:
    matrices = np.asarray(matrices_k, dtype=complex)
    n_cells = phases.shape[1]
    blocks = np.einsum(
        "kt,kmn,ku->tmun", phases, matrices, phases.conj(), optimize=True
    ) / n_cells
    matrix = blocks.reshape(
        n_cells * matrices.shape[1], n_cells * matrices.shape[2]
    )
    return 0.5 * (matrix + matrix.conj().T)


def _real_space_matrix_from_k(
    matrices_k: Sequence[np.ndarray],
    kpoints_fractional: np.ndarray,
    translations: np.ndarray,
    *,
    label: str,
    imaginary_tolerance: float = 1e-8,
) -> np.ndarray:
    matrix, imaginary = aiccm2026dev_b_matrix_to_real_supercell(
        np.ascontiguousarray(np.asarray(matrices_k, dtype=np.complex128)),
        np.ascontiguousarray(
            np.asarray(kpoints_fractional, dtype=float).reshape(-1, 3)
        ),
        np.ascontiguousarray(np.asarray(translations, dtype=np.int64).reshape(-1, 3)),
    )
    if float(imaginary) > imaginary_tolerance:
        raise RuntimeError(
            "AICCM2026DEV-B localization expected a real finite-torus "
            f"{label} matrix; maximum imaginary residual is "
            f"{float(imaginary):.3e}"
        )
    return np.asarray(matrix, dtype=float)


def _canonical_wannier_coefficients(
    coefficients_k: Sequence[np.ndarray],
    phases: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    coefficients = np.asarray(coefficients_k, dtype=complex)[:, :, :n_occ]
    # w_(R,n) = N^-1 sum_k exp[-ik.R] psi_(n,k), while the AO coefficient
    # of psi_(n,k) in cell T carries exp[+ik.T].
    blocks = np.einsum(
        "kt,kmn,kr->tmrn", phases, coefficients, phases.conj(), optimize=True
    ) / phases.shape[0]
    return blocks.reshape(
        phases.shape[1] * coefficients.shape[1],
        phases.shape[1] * n_occ,
    )


def _canonical_wannier_coefficients_from_k(
    coefficients_k: Sequence[np.ndarray],
    kpoints_fractional: np.ndarray,
    translations: np.ndarray,
    n_occ: int,
) -> np.ndarray:
    return np.asarray(
        aiccm2026dev_b_canonical_wannier_coefficients(
            np.ascontiguousarray(np.asarray(coefficients_k, dtype=np.complex128)),
            np.ascontiguousarray(
                np.asarray(kpoints_fractional, dtype=float).reshape(-1, 3)
            ),
            np.ascontiguousarray(
                np.asarray(translations, dtype=np.int64).reshape(-1, 3)
            ),
            int(n_occ),
        ),
        dtype=complex,
    )


def _objective(matrices: Sequence[np.ndarray]) -> float:
    return float(
        sum(np.sum(np.real(np.diag(matrix)) ** 2) for matrix in matrices)
    )


def _pair_rotation(
    matrices: Sequence[np.ndarray], i: int, j: int
) -> tuple[complex, complex, float]:
    """Return the best complex Jacobi rotation for one orbital pair.

    For a phase ``phi``, the half-difference of every transformed diagonal is
    ``delta*cos(2 theta) + Re(exp(i phi) b)*sin(2 theta)``.  Its squared norm
    is the largest eigenvalue of a real 2x2 matrix.  A deterministic phase
    grid makes the remaining one-dimensional maximization robust and avoids
    a new optimizer dependency.
    """

    delta = np.asarray(
        [0.5 * float(np.real(m[i, i] - m[j, j])) for m in matrices]
    )
    offdiag = np.asarray([m[i, j] for m in matrices], dtype=complex)
    old = float(delta @ delta)
    best = old
    best_theta = 0.0
    best_phi = 0.0
    for phi in np.linspace(-np.pi, np.pi, 129, endpoint=False):
        q = np.real(np.exp(1j * phi) * offdiag)
        gram = np.asarray(
            [[delta @ delta, delta @ q], [delta @ q, q @ q]], dtype=float
        )
        values, vectors = np.linalg.eigh(gram)
        if values[-1] <= best + 1e-16:
            continue
        vector = vectors[:, -1]
        t = float(np.arctan2(vector[1], vector[0]))
        while t > 0.5 * np.pi:
            t -= np.pi
        while t < -0.5 * np.pi:
            t += np.pi
        best = float(values[-1])
        best_theta = 0.5 * t
        best_phi = float(phi)
    c = complex(np.cos(best_theta))
    s = complex(np.exp(1j * best_phi) * np.sin(best_theta))
    return c, s, max(0.0, 2.0 * (best - old))


def _joint_diagonalize(
    property_matrices: Sequence[np.ndarray],
    *,
    max_sweeps: int,
    tolerance: float,
) -> tuple[np.ndarray, list[np.ndarray], float, float]:
    matrices = [
        np.array(value, dtype=complex, copy=True) for value in property_matrices
    ]
    n_orbitals = matrices[0].shape[0]
    unitary = np.eye(n_orbitals, dtype=complex)
    initial = _objective(matrices)
    for _ in range(max_sweeps):
        sweep_gain = 0.0
        for i in range(n_orbitals):
            for j in range(i + 1, n_orbitals):
                c, s, gain = _pair_rotation(matrices, i, j)
                if gain <= tolerance:
                    continue
                rotation = np.eye(n_orbitals, dtype=complex)
                rotation[i, i] = c
                rotation[j, j] = c
                rotation[i, j] = -s.conjugate()
                rotation[j, i] = s
                matrices = [
                    rotation.conj().T @ matrix @ rotation for matrix in matrices
                ]
                unitary = unitary @ rotation
                sweep_gain += gain
        if sweep_gain <= tolerance:
            break
    return unitary, matrices, initial, _objective(matrices)


def _lowdin_atomic_properties(
    coefficients: np.ndarray,
    overlap_sqrt: np.ndarray,
    atom_indices: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    orthogonal_coefficients = overlap_sqrt @ coefficients
    matrices = []
    for atom in range(int(atom_indices.max()) + 1):
        rows = atom_indices == atom
        block = orthogonal_coefficients[rows]
        matrices.append(block.conj().T @ block)
    return matrices, orthogonal_coefficients


def _iao_properties(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    target_reference_overlap: np.ndarray,
    reference_overlap: np.ndarray,
    reference_atom_indices: np.ndarray,
    *,
    threshold: float,
) -> tuple[list[np.ndarray], float]:
    """Construct Knizia IAOs and their atomic population operators."""

    s_inv = _hermitian_power(overlap, -1.0, threshold)
    sr_inv = _hermitian_power(reference_overlap, -1.0, threshold)
    p12 = s_inv @ target_reference_overlap
    depolarized = p12 @ sr_inv @ target_reference_overlap.conj().T @ coefficients
    metric = depolarized.conj().T @ overlap @ depolarized
    try:
        depolarized = depolarized @ _hermitian_power(metric, -0.5, threshold)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "AICCM2026DEV-B IAO reference does not span the occupied space"
        ) from exc

    projector_occ = coefficients @ coefficients.conj().T @ overlap
    projector_dep = depolarized @ depolarized.conj().T @ overlap
    identity = np.eye(overlap.shape[0])
    iao_raw = (
        projector_occ @ projector_dep
        + (identity - projector_occ) @ (identity - projector_dep)
    ) @ p12
    iao_metric = iao_raw.conj().T @ overlap @ iao_raw
    iaos = iao_raw @ _hermitian_power(iao_metric, -0.5, threshold)
    error = float(
        np.linalg.norm(iaos.conj().T @ overlap @ iaos - np.eye(iaos.shape[1]))
    )

    amplitudes = iaos.conj().T @ overlap @ coefficients
    matrices = []
    for atom in range(int(reference_atom_indices.max()) + 1):
        rows = reference_atom_indices == atom
        block = amplitudes[rows]
        matrices.append(block.conj().T @ block)
    completeness = sum(matrices)
    if np.linalg.norm(completeness - np.eye(coefficients.shape[1])) > 1e-6:
        raise RuntimeError(
            "AICCM2026DEV-B IAO reference does not span the occupied space"
        )
    return matrices, error


def _centers_spreads_aliasing(
    coefficients: np.ndarray,
    overlap_sqrt: np.ndarray,
    ao_fractional: np.ndarray,
    lattice: np.ndarray,
    mesh: tuple[int, int, int],
    dimension: int,
    alias_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orthogonal = overlap_sqrt @ coefficients
    probabilities = np.abs(orthogonal) ** 2
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    n_orbitals = coefficients.shape[1]
    centers = np.zeros((n_orbitals, 3))
    spread_components = np.zeros((n_orbitals, 3))
    alias = np.zeros(n_orbitals)
    active = range(dimension)

    for axis in range(3):
        coordinate = ao_fractional[:, axis]
        if axis in active:
            period = float(mesh[axis])
            phase = np.exp(2j * np.pi * coordinate / period)
            moment = phase @ probabilities
            centers[:, axis] = np.mod(np.angle(moment), 2.0 * np.pi) * period / (
                2.0 * np.pi
            )
            modulus = np.clip(np.abs(moment), 1e-15, 1.0)
            length = float(np.linalg.norm(lattice[:, axis]) * mesh[axis])
            spread_components[:, axis] = (
                -(length / (2.0 * np.pi)) ** 2 * np.log(modulus**2)
            )
            distance = np.mod(
                coordinate[:, None]
                - centers[None, :, axis]
                + 0.5 * period,
                period,
            ) - 0.5 * period
            antipodal = np.abs(distance) >= max(0.0, 0.5 * period - 0.5)
            alias = np.maximum(alias, np.sum(probabilities * antipodal, axis=0))
        else:
            centers[:, axis] = coordinate @ probabilities
            delta = coordinate[:, None] - centers[None, :, axis]
            spread_components[:, axis] = (
                np.linalg.norm(lattice[:, axis]) ** 2
                * np.sum(probabilities * delta**2, axis=0)
            )
    spreads = spread_components.sum(axis=1)
    shortest = min(
        float(np.linalg.norm(lattice[:, axis]) * mesh[axis]) for axis in active
    )
    detected = (alias > alias_threshold) | (np.sqrt(spreads) > 0.25 * shortest)
    return centers, centers @ lattice.T, spreads, alias, detected


def _translation_projector_error(
    projector: np.ndarray,
    translations: np.ndarray,
    mesh: tuple[int, int, int],
    nbf: int,
) -> float:
    if len(translations) == 1:
        return 0.0
    lookup = {tuple(value): index for index, value in enumerate(translations)}
    blocks = projector.reshape(len(translations), nbf, len(translations), nbf)
    maximum = 0.0
    for axis in range(3):
        source = []
        for translation in translations:
            shifted = np.array(translation, copy=True)
            shifted[axis] = (shifted[axis] - 1) % mesh[axis]
            source.append(lookup[tuple(shifted)])
        shifted = blocks[np.ix_(source, np.arange(nbf), source, np.arange(nbf))]
        # np.ix_ above groups the two cell and two AO axes in their original
        # order, unlike chained advanced indexing.
        maximum = max(maximum, float(np.linalg.norm(shifted - blocks)))
    return maximum


def _localization_projector_audit_python(
    canonical: np.ndarray,
    localized: np.ndarray,
    fock: np.ndarray | None,
    translations: np.ndarray,
    mesh: tuple[int, int, int],
    nbf: int,
) -> tuple[float, float, float]:
    """Dense NumPy oracle for the native low-rank localization audit."""

    projector0 = canonical @ canonical.conj().T
    projector1 = localized @ localized.conj().T
    density_error = float(np.linalg.norm(projector1 - projector0))
    energy_error = 0.0
    if fock is not None:
        energy_error = abs(
            float(
                np.trace(projector1 @ fock).real
                - np.trace(projector0 @ fock).real
            )
        )
    translation_error = _translation_projector_error(
        projector1,
        translations,
        mesh,
        nbf,
    )
    return density_error, energy_error, translation_error


def _localization_projector_audit(
    canonical: np.ndarray,
    localized: np.ndarray,
    fock: np.ndarray | None,
    translations: np.ndarray,
    mesh: tuple[int, int, int],
    nbf: int,
) -> tuple[float, float, float]:
    """Evaluate projector invariants without constructing AO projectors."""

    native_fock = None
    if fock is not None:
        fock_array = np.asarray(fock)
        native_fock = np.ascontiguousarray(
            fock_array,
            dtype=(np.complex128 if np.iscomplexobj(fock_array) else np.float64),
        )
    values = aiccm2026dev_b_localization_projector_audit(
        np.ascontiguousarray(canonical, dtype=np.complex128),
        np.ascontiguousarray(localized, dtype=np.complex128),
        native_fock,
        np.ascontiguousarray(translations, dtype=np.int64),
        np.ascontiguousarray(mesh, dtype=np.int64),
        int(nbf),
    )
    return tuple(float(value) for value in values)


def _translation_permutations(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    translations: np.ndarray,
    mesh: tuple[int, int, int],
    nbf: int,
) -> tuple[np.ndarray, float]:
    """Map each localized orbital under the three cyclic generators.

    A genuine Wannier gauge turns a primitive translation into a permutation
    (and phases) of the localized columns.  The returned table is safer for
    local-correlation consumers than assuming a post-localization column
    order.  The residual is ``max(1-|overlap|)`` for the selected images.
    """

    lookup = {tuple(value): index for index, value in enumerate(translations)}
    permutations = np.zeros((3, coefficients.shape[1]), dtype=int)
    maximum_error = 0.0
    blocks = coefficients.reshape(len(translations), nbf, -1)
    for axis in range(3):
        source = []
        for translation in translations:
            shifted = np.array(translation, copy=True)
            shifted[axis] = (shifted[axis] - 1) % mesh[axis]
            source.append(lookup[tuple(shifted)])
        translated = blocks[np.asarray(source)].reshape(coefficients.shape)
        metric_overlap = coefficients.conj().T @ overlap @ translated
        chosen = np.argmax(np.abs(metric_overlap), axis=0)
        permutations[axis] = chosen
        maximum_error = max(
            maximum_error,
            float(np.max(1.0 - np.abs(metric_overlap[chosen, np.arange(len(chosen))]))),
        )
    return permutations, maximum_error


def localize_aiccm2026dev_b_occupied_blocks(
    coefficients_k: Sequence[np.ndarray],
    overlap_k: Sequence[np.ndarray],
    kpoints_fractional: np.ndarray,
    mesh: Sequence[int],
    n_occ_per_cell: int,
    ao_fractional: np.ndarray,
    ao_atom_indices: np.ndarray,
    lattice_bohr: np.ndarray,
    *,
    dimension: int = 3,
    method: str | AICCM2026DevBLocalizationMethod = "wannier",
    fock_k: Sequence[np.ndarray] | None = None,
    orbital_energies_k: Sequence[np.ndarray] | None = None,
    iao_target_reference_overlap: np.ndarray | None = None,
    iao_reference_overlap: np.ndarray | None = None,
    iao_reference_atom_indices: np.ndarray | None = None,
    iao_reference_basis: str | None = None,
    linear_dependency_threshold: float = 1e-10,
    gap_tolerance: float = 1e-6,
    max_sweeps: int = 100,
    localization_tolerance: float = 1e-12,
    alias_threshold: float = 0.05,
) -> AICCM2026DevBLocalizationResult:
    """Localize occupied blocks with no dependency on an SCF result class."""

    selected = AICCM2026DevBLocalizationMethod(method)
    mesh_tuple = tuple(int(value) for value in mesh)
    if len(mesh_tuple) != 3 or any(value < 1 for value in mesh_tuple):
        raise ValueError(
            "AICCM2026DEV-B localization mesh must contain three positives"
        )
    translations = _translations(mesh_tuple)
    n_cells = len(translations)
    kfrac = np.asarray(kpoints_fractional, dtype=float).reshape(-1, 3)
    if len(kfrac) != n_cells:
        raise ValueError("localization needs the complete unreduced cyclic k mesh")
    coefficients = [np.asarray(value, dtype=complex) for value in coefficients_k]
    overlaps = [np.asarray(value, dtype=complex) for value in overlap_k]
    if len(coefficients) != n_cells or len(overlaps) != n_cells:
        raise ValueError("localization needs one coefficient and overlap block per k")
    nbf = coefficients[0].shape[0]
    if n_occ_per_cell < 1 or n_occ_per_cell > coefficients[0].shape[1]:
        raise ValueError("invalid occupied-orbital count for B localization")
    if any(value.shape[0] != nbf for value in coefficients):
        raise ValueError("inconsistent AO dimensions across the cyclic mesh")
    ao_fractional = np.asarray(ao_fractional, dtype=float)
    ao_atom_indices = np.asarray(ao_atom_indices, dtype=int)
    if ao_fractional.shape != (n_cells * nbf, 3):
        raise ValueError("AO fractional coordinates do not match the real torus")
    if ao_atom_indices.shape != (n_cells * nbf,):
        raise ValueError("AO atom indices do not match the real torus")

    gap = np.inf
    if orbital_energies_k is not None and n_occ_per_cell < coefficients[0].shape[1]:
        energies = [np.asarray(value, dtype=float) for value in orbital_energies_k]
        valence = max(float(value[n_occ_per_cell - 1]) for value in energies)
        conduction = min(float(value[n_occ_per_cell]) for value in energies)
        gap = conduction - valence
        if gap <= gap_tolerance:
            raise ValueError(
                "AICCM2026DEV-B occupied localization requires an isolated, "
                f"gapped occupied manifold; gap={gap:.3e} hartree"
            )

    phases = np.exp(2j * np.pi * (kfrac @ translations.T))
    overlap = _real_space_matrix_from_k(
        overlaps,
        kfrac,
        translations,
        label="overlap",
    )
    overlap_sqrt = _hermitian_power(overlap, 0.5, linear_dependency_threshold)
    canonical = _canonical_wannier_coefficients_from_k(
        coefficients,
        kfrac,
        translations,
        n_occ_per_cell,
    )
    identity_occ = np.eye(n_cells * n_occ_per_cell)
    orth0 = canonical.conj().T @ overlap @ canonical
    if np.linalg.norm(orth0 - identity_occ) > 1e-7:
        raise RuntimeError(
            "AICCM202DEV-B Bloch-to-Wannier transform is not metric-unitary"
        )

    if selected is AICCM2026DevBLocalizationMethod.WANNIER:
        orthogonal = overlap_sqrt @ canonical
        properties: list[np.ndarray] = []
        for axis in range(dimension):
            angle = 2.0 * np.pi * ao_fractional[:, axis] / mesh_tuple[axis]
            for value in (np.cos(angle), np.sin(angle)):
                properties.append(orthogonal.conj().T @ (value[:, None] * orthogonal))
        iao_error = None
    else:
        if any(
            value is None
            for value in (
                iao_target_reference_overlap,
                iao_reference_overlap,
                iao_reference_atom_indices,
            )
        ):
            raise ValueError("IAO localization requires reference overlap data")
        properties, iao_error = _iao_properties(
            canonical,
            overlap,
            np.asarray(iao_target_reference_overlap),
            np.asarray(iao_reference_overlap),
            np.asarray(iao_reference_atom_indices, dtype=int),
            threshold=linear_dependency_threshold,
        )

    unitary, rotated_properties, objective_initial, objective_final = (
        _joint_diagonalize(
            properties,
            max_sweeps=max_sweeps,
            tolerance=localization_tolerance,
        )
    )
    localized = canonical @ unitary

    # Fix only the physically irrelevant per-orbital phase.  This makes the
    # reported real residual meaningful for centrosymmetric systems.
    for orbital in range(localized.shape[1]):
        pivot = int(np.argmax(np.abs(localized[:, orbital])))
        phase = np.angle(localized[pivot, orbital])
        localized[:, orbital] *= np.exp(-1j * phase)
    unitary = canonical.conj().T @ overlap @ localized

    orth_error = float(
        np.linalg.norm(localized.conj().T @ overlap @ localized - identity_occ)
    )
    unitary_error = float(np.linalg.norm(unitary.conj().T @ unitary - identity_occ))
    fock = None
    if fock_k is not None:
        fock = _real_space_matrix_from_k(
            fock_k,
            kfrac,
            translations,
            label="Fock",
        )
    density_error, energy_error, translation_projector_error = (
        _localization_projector_audit(
            canonical,
            localized,
            fock,
            translations,
            mesh_tuple,
            nbf,
        )
    )

    atomic_properties, _ = _lowdin_atomic_properties(
        localized, overlap_sqrt, ao_atom_indices
    )
    atomic_populations = np.stack(
        [np.real(np.diag(value)) for value in atomic_properties], axis=1
    )
    centers_fractional, centers_bohr, spreads, alias, detected = (
        _centers_spreads_aliasing(
            localized,
            overlap_sqrt,
            ao_fractional,
            np.asarray(lattice_bohr, dtype=float),
            mesh_tuple,
            dimension,
            alias_threshold,
        )
    )
    if np.any(detected):
        warnings.warn(
            "AICCM2026DEV-B localized-orbital weight reaches the cyclic "
            "antipode; increase the cluster before using local domains",
            AICCM2026DevBLocalizationWarning,
            stacklevel=2,
        )

    translation_permutations, translation_covariance_error = (
        _translation_permutations(
            localized, overlap, translations, mesh_tuple, nbf
        )
    )
    return AICCM2026DevBLocalizationResult(
        method=selected.value,
        coefficients=localized,
        canonical_wannier_coefficients=canonical,
        unitary=unitary,
        overlap=overlap,
        fock=fock,
        translations=translations,
        translation_permutations=translation_permutations,
        centers_fractional=centers_fractional,
        centers_bohr=centers_bohr,
        spreads_bohr2=spreads,
        atomic_populations=atomic_populations,
        aliasing_fraction=alias,
        aliasing_detected=detected,
        objective_initial=objective_initial,
        objective_final=objective_final,
        density_invariance_error=density_error,
        one_particle_energy_error=energy_error,
        orthonormality_error=orth_error,
        unitary_error=unitary_error,
        translation_projector_error=translation_projector_error,
        translation_covariance_error=translation_covariance_error,
        real_gauge_residual=float(
            np.linalg.norm(localized.imag) / max(np.linalg.norm(localized), 1e-30)
        ),
        band_gap_hartree=float(gap),
        n_cells=n_cells,
        n_occ_per_cell=n_occ_per_cell,
        iao_reference_basis=iao_reference_basis,
        iao_orthonormality_error=iao_error,
    )


def _ao_metadata(basis: BasisSet, system: PeriodicSystem, translations: np.ndarray):
    # The atom map is the canonical derivation in vibeqc.properties; the
    # shell origins are laid out here with the same per-shell count, so
    # the two stay the same length on a Cartesian basis. The local copy
    # this replaces counted 2l+1 per shell while preallocating nbasis
    # rows, which left a silently unwritten tail past the first d shell.
    primitive_atoms = _shell_to_atom(basis)
    nbf = int(basis.nbasis)
    if primitive_atoms.size != nbf:
        raise RuntimeError("basis shell metadata does not match the AO dimension")
    primitive_centers = np.zeros((nbf, 3))
    cursor = 0
    for shell in basis.shells():
        count = _shell_nao(shell)
        primitive_centers[cursor : cursor + count] = np.asarray(shell.origin)
        cursor += count
    lattice = np.asarray(system.lattice, dtype=float)
    primitive_fractional = primitive_centers @ np.linalg.inv(lattice).T
    fractional = np.concatenate(
        [primitive_fractional + translation for translation in translations], axis=0
    )
    n_atoms = len(system.unit_cell)
    atoms = np.concatenate(
        [primitive_atoms + cell * n_atoms for cell in range(len(translations))]
    )
    return fractional, atoms


def _reference_atom_indices(reference_basis: BasisSet) -> np.ndarray:
    """AO -> atom map over the IAO reference basis, as a platform-int array.

    Delegates to :func:`vibeqc.properties._shell_to_atom`; the local copy
    this replaces wrote a 2l+1 cursor into an array preallocated at
    ``nbasis``, so a Cartesian reference basis left a tail of zeros that
    read as atom 0.
    """
    indices = _shell_to_atom(reference_basis)
    if indices.size != int(reference_basis.nbasis):
        raise RuntimeError("basis shell metadata does not match the AO dimension")
    return indices.astype(int, copy=False)


def _as_matrix_blocks(value: object) -> list[np.ndarray]:
    array = np.asarray(value)
    if array.ndim == 2:
        return [array]
    return [np.asarray(block) for block in value]  # type: ignore[union-attr]


def _as_vector_blocks(value: object) -> list[np.ndarray]:
    array = np.asarray(value)
    if array.ndim == 1:
        return [array]
    return [np.asarray(block) for block in value]  # type: ignore[union-attr]


def localize_aiccm2026dev_b_occupied(
    result: object,
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: int | Sequence[int] | None = None,
    *,
    method: str | AICCM2026DevBLocalizationMethod = "wannier",
    iao_reference_basis: str = "ano-rcc-mb",
    **kwargs,
) -> AICCM2026DevBLocalizationResult:
    """Localize the converged occupied space of an AICCM2026DEV-B SCF.

    The returned real-torus coefficients are suitable for PAO/domain
    construction.  The function rejects unconverged references and metallic
    manifolds; disentanglement remains a later extension.
    """

    if not bool(getattr(result, "converged", False)):
        raise ValueError("AICCM2026DEV-B localization requires a converged SCF")
    if not hasattr(result, "aiccm2026dev_b"):
        raise TypeError("result is not an AICCM2026DEV-B SCF result")
    selected_mesh = (
        getattr(result.aiccm2026dev_b, "mesh") if mesh is None else mesh
    )
    mesh_tuple = _normalise_mesh(system, selected_mesh)
    kpoints = cyclic_gamma_mesh(system, mesh_tuple)
    translations = _translations(mesh_tuple)
    ao_fractional, ao_atoms = _ao_metadata(basis, system, translations)

    iao_cross = None
    iao_overlap = None
    iao_atoms = None
    selected = AICCM2026DevBLocalizationMethod(method)
    if selected is AICCM2026DevBLocalizationMethod.IAO:
        from ..._vibeqc_core import compute_overlap, compute_overlap_two_basis
        from .posthf import _build_supercell_system

        super_system = _build_supercell_system(system, mesh_tuple)
        molecule = super_system.unit_cell_molecule()
        target_basis = BasisSet(molecule, basis.name)
        reference_basis = BasisSet(molecule, iao_reference_basis)
        iao_cross = np.asarray(
            compute_overlap_two_basis(target_basis, reference_basis), dtype=float
        )
        iao_overlap = np.asarray(compute_overlap(reference_basis), dtype=float)
        iao_atoms = _reference_atom_indices(reference_basis)

    return localize_aiccm2026dev_b_occupied_blocks(
        _as_matrix_blocks(result.mo_coeffs),
        _as_matrix_blocks(result.overlap),
        np.asarray(kpoints.kpoints_frac),
        mesh_tuple,
        int(system.n_electrons()) // 2,
        ao_fractional,
        ao_atoms,
        np.asarray(system.lattice, dtype=float),
        dimension=int(system.dim),
        method=selected,
        fock_k=(
            _as_matrix_blocks(result.fock) if hasattr(result, "fock") else None
        ),
        orbital_energies_k=(
            _as_vector_blocks(result.mo_energies)
            if hasattr(result, "mo_energies")
            else None
        ),
        iao_target_reference_overlap=iao_cross,
        iao_reference_overlap=iao_overlap,
        iao_reference_atom_indices=iao_atoms,
        iao_reference_basis=(iao_reference_basis if selected.value == "iao" else None),
        **kwargs,
    )


def localize_aiccm2026dev_b_unrestricted_occupied(
    result: object,
    system: PeriodicSystem,
    basis: BasisSet,
    mesh: int | Sequence[int] | None = None,
    *,
    method: str | AICCM2026DevBLocalizationMethod = "wannier",
    iao_reference_basis: str = "ano-rcc-mb",
    **kwargs,
) -> AICCM2026DevBUnrestrictedLocalizationResult:
    """Localize the alpha and beta occupied projectors of a B-stream UHF.

    The spin spaces are rotated separately.  This preserves both spin
    densities and hence preserves the total density, spin density, and UHF
    energy.  Metallic spin manifolds are rejected by the same finite-gap
    check used for the restricted path.
    """

    if not bool(getattr(result, "converged", False)):
        raise ValueError("AICCM2026DEV-B localization requires a converged SCF")
    if not hasattr(result, "aiccm2026dev_b"):
        raise TypeError("result is not an AICCM2026DEV-B SCF result")
    if not hasattr(result, "mo_coeffs_alpha"):
        raise TypeError("result is not an unrestricted AICCM2026DEV-B result")

    selected_mesh = (
        getattr(result.aiccm2026dev_b, "mesh") if mesh is None else mesh
    )
    mesh_tuple = _normalise_mesh(system, selected_mesh)
    kpoints = cyclic_gamma_mesh(system, mesh_tuple)
    translations = _translations(mesh_tuple)
    ao_fractional, ao_atoms = _ao_metadata(basis, system, translations)
    selected = AICCM2026DevBLocalizationMethod(method)

    iao_cross = None
    iao_overlap = None
    iao_atoms = None
    if selected is AICCM2026DevBLocalizationMethod.IAO:
        from ..._vibeqc_core import compute_overlap, compute_overlap_two_basis
        from .posthf import _build_supercell_system

        super_system = _build_supercell_system(
            system,
            mesh_tuple,
            multiplicity=int(np.prod(mesh_tuple)) * (int(system.multiplicity) - 1)
            + 1,
        )
        molecule = super_system.unit_cell_molecule()
        target_basis = BasisSet(molecule, basis.name)
        reference_basis = BasisSet(molecule, iao_reference_basis)
        iao_cross = np.asarray(
            compute_overlap_two_basis(target_basis, reference_basis), dtype=float
        )
        iao_overlap = np.asarray(compute_overlap(reference_basis), dtype=float)
        iao_atoms = _reference_atom_indices(reference_basis)

    ne = int(system.n_electrons())
    two_s = int(system.multiplicity) - 1
    n_alpha = (ne + two_s) // 2
    n_beta = (ne - two_s) // 2
    common = dict(
        overlap_k=_as_matrix_blocks(result.overlap),
        kpoints_fractional=np.asarray(kpoints.kpoints_frac),
        mesh=mesh_tuple,
        ao_fractional=ao_fractional,
        ao_atom_indices=ao_atoms,
        lattice_bohr=np.asarray(system.lattice, dtype=float),
        dimension=int(system.dim),
        method=selected,
        iao_target_reference_overlap=iao_cross,
        iao_reference_overlap=iao_overlap,
        iao_reference_atom_indices=iao_atoms,
        iao_reference_basis=(iao_reference_basis if selected.value == "iao" else None),
        **kwargs,
    )
    alpha = localize_aiccm2026dev_b_occupied_blocks(
        _as_matrix_blocks(result.mo_coeffs_alpha),
        n_occ_per_cell=n_alpha,
        fock_k=_as_matrix_blocks(result.fock_alpha),
        orbital_energies_k=_as_vector_blocks(result.mo_energies_alpha),
        **common,
    )
    beta = None
    if n_beta:
        beta = localize_aiccm2026dev_b_occupied_blocks(
            _as_matrix_blocks(result.mo_coeffs_beta),
            n_occ_per_cell=n_beta,
            fock_k=_as_matrix_blocks(result.fock_beta),
            orbital_energies_k=_as_vector_blocks(result.mo_energies_beta),
            **common,
        )
    return AICCM2026DevBUnrestrictedLocalizationResult(
        alpha=alpha,
        beta=beta,
        density_invariance_error=max(
            alpha.density_invariance_error,
            0.0 if beta is None else beta.density_invariance_error,
        ),
        one_particle_energy_error=max(
            alpha.one_particle_energy_error,
            0.0 if beta is None else beta.one_particle_energy_error,
        ),
    )
