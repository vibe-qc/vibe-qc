"""Numerical MS/XMS-CASPT2 nonadiabatic derivative couplings.

The derivative coupling between physical states Q and P is evaluated from
the symmetrized CASPT2 wavefunction definition

    d_QP = 1/2 [<Q|d Psi_P/dX> + <Psi_Q|d P/dX>].

Here ``|P>`` is the multistate-mixed zeroth-order CASSCF reference and
``|Psi_P>`` additionally contains the internally contracted first-order
CASPT2 wavefunction.  This is Eq. 30 of Park and Shiozaki, JCTC 13, 2561
(2017), DOI 10.1021/acs.jctc.7b00018.  Central finite differences reoptimize
the SA-CASSCF orbitals and CI vectors at every displaced geometry.  Cross-
geometry determinant overlaps use the native two-basis AO overlap integral,
so AO-centre motion (the determinant/CSF term) is retained rather than being
silently dropped.

This first production-shaped route deliberately targets the explicit,
unshifted, small-space IC-CASPT2 engine.  It is expensive (2 * 3N displaced
SA-CASSCF plus MS-CASPT2 solves) and fails closed when state tracking or the
nondegenerate-state envelope is lost.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

__all__ = ["compute_ms_caspt2_nac"]


@dataclass
class _MSWavefunctionPoint:
    basis: object
    mo_coeff: np.ndarray
    energies: np.ndarray
    zero_states: list[dict[int, float]]
    full_states: list[dict[int, float]]


def _linear_combination(
    states: list[dict[int, float]], coefficients: np.ndarray
) -> dict[int, float]:
    result: dict[int, float] = {}
    for coefficient, state in zip(coefficients, states):
        if abs(coefficient) < 1e-15:
            continue
        for determinant, value in state.items():
            result[determinant] = result.get(determinant, 0.0) + float(
                coefficient * value
            )
    return {key: value for key, value in result.items() if abs(value) > 1e-14}


def _backtransform_state(
    state: dict[int, float],
    rotation: np.ndarray,
    n_core: int,
    n_active_orb: int,
) -> dict[int, float]:
    """Express a semicanonical determinant state in the input MO basis."""
    norb = rotation.shape[0]
    blocks = (
        tuple(range(0, n_core)),
        tuple(range(n_core, n_core + n_active_orb)),
        tuple(range(n_core + n_active_orb, norb)),
    )

    def spin_expansions(occupied: list[int]):
        terms: list[tuple[tuple[int, ...], float]] = [((), 1.0)]
        occupied_set = set(occupied)
        for block in blocks:
            source = tuple(orbital for orbital in block if orbital in occupied_set)
            if not source:
                choices = [((), 1.0)]
            elif np.allclose(
                rotation[np.ix_(block, block)], np.eye(len(block)), atol=1e-14
            ):
                choices = [(source, 1.0)]
            else:
                choices = []
                for target in combinations(block, len(source)):
                    coefficient = float(
                        np.linalg.det(rotation[np.ix_(target, source)])
                    )
                    if abs(coefficient) > 1e-14:
                        choices.append((target, coefficient))
            terms = [
                (left + right, left_coefficient * right_coefficient)
                for left, left_coefficient in terms
                for right, right_coefficient in choices
            ]
        return terms

    transformed: dict[int, float] = {}
    for determinant, coefficient in state.items():
        alpha = [orbital for orbital in range(norb) if determinant >> orbital & 1]
        beta = [
            orbital
            for orbital in range(norb)
            if determinant >> (norb + orbital) & 1
        ]
        for alpha_target, alpha_coefficient in spin_expansions(alpha):
            alpha_mask = sum(1 << orbital for orbital in alpha_target)
            for beta_target, beta_coefficient in spin_expansions(beta):
                beta_mask = sum(1 << (norb + orbital) for orbital in beta_target)
                target = alpha_mask | beta_mask
                transformed[target] = transformed.get(target, 0.0) + float(
                    coefficient * alpha_coefficient * beta_coefficient
                )
    return {
        determinant: coefficient
        for determinant, coefficient in transformed.items()
        if abs(coefficient) > 1e-14
    }


def _physical_states(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_core: int,
    n_active_orb: int,
    n_active_elec: int,
    nroots: int,
    mode: str,
    ms2: int,
    n_frozen: int,
    imaginary: float,
) -> tuple[np.ndarray, list[dict[int, float]], list[dict[int, float]]]:
    """Build physical zeroth- and first-order-corrected MS states."""
    from vibeqc.solvers._mrpt import _add
    from vibeqc.solvers._ms_caspt2 import _heff_pt2_blocks, _spin_pure_roots

    norb = h1e_mo.shape[0]
    ci_cols, e_ref, _s2, cas = _spin_pure_roots(
        h1e_mo,
        h2e_mo,
        n_core,
        n_active_orb,
        n_active_elec,
        ms2,
        nroots,
    )
    blocks = _heff_pt2_blocks(
        ci_cols,
        cas.determinants,
        h1e_mo,
        h2e_mo.transpose(0, 2, 1, 3).copy(),
        n_core,
        n_active_orb,
        norb,
        mode=mode,
        n_frozen=n_frozen,
        imaginary=imaginary,
        return_wavefunctions=True,
    )
    rotation = np.asarray(blocks["u_xms"])
    reference = rotation.T @ np.diag(np.asarray(e_ref)) @ rotation
    heff = reference.copy()
    heff[np.diag_indices(nroots)] += np.asarray(blocks["e2"])
    heff += np.asarray(blocks["coup"])
    energies, mixing = eigh(0.5 * (heff + heff.T))
    for root in range(nroots):
        pivot = int(np.argmax(np.abs(mixing[:, root])))
        if mixing[pivot, root] < 0.0:
            mixing[:, root] *= -1.0

    model_zero = list(blocks["refs"])
    model_first_order = [
        _backtransform_state(state, rotation, n_core, n_active_orb)
        for state, rotation in zip(
            blocks["psi1s"], blocks["orbital_rotations"]
        )
    ]
    model_full = [
        _add(reference_state, first_order)
        for reference_state, first_order in zip(model_zero, model_first_order)
    ]
    zero = [
        _linear_combination(model_zero, mixing[:, root]) for root in range(nroots)
    ]
    full = [
        _linear_combination(model_full, mixing[:, root]) for root in range(nroots)
    ]
    return np.asarray(energies), zero, full


def _determinant_overlap(
    bra: int, ket: int, orbital_overlap: np.ndarray, norb: int
) -> float:
    bra_occ = [index for index in range(2 * norb) if (bra >> index) & 1]
    ket_occ = [index for index in range(2 * norb) if (ket >> index) & 1]
    if len(bra_occ) != len(ket_occ):
        return 0.0
    matrix = np.zeros((len(bra_occ), len(ket_occ)))
    for row, left in enumerate(bra_occ):
        left_spin, left_orbital = divmod(left, norb)
        for column, right in enumerate(ket_occ):
            right_spin, right_orbital = divmod(right, norb)
            if left_spin == right_spin:
                matrix[row, column] = orbital_overlap[left_orbital, right_orbital]
    return float(np.linalg.det(matrix))


def _wavefunction_overlap(
    bra: dict[int, float],
    ket: dict[int, float],
    orbital_overlap: np.ndarray,
    cache: dict[tuple[int, int], float],
) -> float:
    norb = orbital_overlap.shape[0]
    value = 0.0
    for bra_det, bra_coefficient in bra.items():
        for ket_det, ket_coefficient in ket.items():
            key = (bra_det, ket_det)
            determinant_value = cache.get(key)
            if determinant_value is None:
                determinant_value = _determinant_overlap(
                    bra_det, ket_det, orbital_overlap, norb
                )
                cache[key] = determinant_value
            value += bra_coefficient * ket_coefficient * determinant_value
    return float(value)


def _align_to_reference(
    reference: _MSWavefunctionPoint,
    displaced: _MSWavefunctionPoint,
    orbital_overlap: np.ndarray,
    tracking_tolerance: float,
) -> _MSWavefunctionPoint:
    cache: dict[tuple[int, int], float] = {}
    overlap = np.asarray(
        [
            [
                _wavefunction_overlap(left, right, orbital_overlap, cache)
                for right in displaced.zero_states
            ]
            for left in reference.zero_states
        ]
    )
    rows, columns = linear_sum_assignment(-np.abs(overlap))
    assignment = np.empty(len(rows), dtype=int)
    assignment[rows] = columns
    tracked = overlap[np.arange(len(rows)), assignment]
    if np.min(np.abs(tracked)) < tracking_tolerance:
        raise RuntimeError(
            "MS-CASPT2 NAC state tracking failed: smallest reference/displaced "
            f"state overlap is {np.min(np.abs(tracked)):.3f}, below "
            f"tracking_tolerance={tracking_tolerance:.3f}. Reduce fd_step or "
            "choose a geometry away from a state-character discontinuity."
        )
    phases = np.where(tracked < 0.0, -1.0, 1.0)

    def reorder(states: list[dict[int, float]]) -> list[dict[int, float]]:
        return [
            {key: float(phases[root] * value) for key, value in states[index].items()}
            for root, index in enumerate(assignment)
        ]

    return _MSWavefunctionPoint(
        basis=displaced.basis,
        mo_coeff=displaced.mo_coeff,
        energies=displaced.energies[assignment],
        zero_states=reorder(displaced.zero_states),
        full_states=reorder(displaced.full_states),
    )


def compute_ms_caspt2_nac(
    molecule,
    basis,
    C_mo: np.ndarray,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    n_core: int,
    n_active_orb: int,
    *,
    n_active_elec: int,
    sa_weights: list[float],
    nroots: int,
    state_pair: tuple[int, int],
    mode: str = "xms",
    ms2: int = 0,
    n_frozen: int = 0,
    imaginary: float = 0.0,
    fd_step: float = 1e-3,
    gap_tolerance: float = 1e-6,
    tracking_tolerance: float = 0.5,
    max_wavefunction_determinants: int = 4096,
    casscf_orbital_step: str = "auto",
    casscf_spin_pure: bool | None = True,
    casscf_max_macro: int = 100,
    casscf_conv_tol_grad: float = 1e-6,
    casscf_trust: float = 0.3,
) -> np.ndarray:
    """Return the state-pair MS/XMS-CASPT2 derivative coupling.

    The returned ``(n_atoms, 3)`` array is in bohr^-1 and follows the ordered
    pair convention ``d[state_pair[0], state_pair[1]]``.  Reversing the pair
    reverses the vector sign after using the same continuous state gauge.

    The current verified envelope is an exact-CI, closed-shell, unshifted,
    SA-CASSCF reference on the explicit IC-CASPT2 engine.  The caller is
    responsible for enforcing the public option-level gate; this low-level
    helper independently validates the state and numerical-response inputs.

    Manually supplied ECP-derived orbitals, integrals, or reference data
    paired with an all-electron-named basis are unsupported.  This API can
    reject an ECP attached to ``basis``, but cannot infer the Hamiltonian
    provenance of the supplied arrays.
    """
    from vibeqc.ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        molecule,
        basis,
        route="compute_ms_caspt2_nac",
    )
    from vibeqc._vibeqc_core import (
        Atom,
        Molecule,
        compute_eri,
        compute_kinetic,
        compute_nuclear,
        compute_overlap,
        compute_overlap_two_basis,
    )
    from vibeqc.solvers._casscf import casscf

    if mode not in ("ms", "xms"):
        raise ValueError(f"mode must be 'ms' or 'xms', got {mode!r}")
    if nroots < 2:
        raise ValueError("MS-CASPT2 NAC requires nroots >= 2")
    if len(sa_weights) != nroots:
        raise ValueError(
            f"exactly the {nroots} model states must be SA-averaged; got "
            f"{len(sa_weights)} weights"
        )
    if fd_step <= 0.0 or not np.isfinite(fd_step):
        raise ValueError(f"fd_step must be finite and > 0, got {fd_step!r}")
    if len(state_pair) != 2:
        raise ValueError(
            f"state_pair must contain two distinct roots, got {state_pair!r}"
        )
    q_root, p_root = state_pair
    if q_root == p_root or not (0 <= q_root < nroots and 0 <= p_root < nroots):
        raise ValueError(
            f"state_pair must contain two distinct roots in [0, {nroots}), "
            f"got {state_pair!r}"
        )
    if gap_tolerance <= 0.0 or not np.isfinite(gap_tolerance):
        raise ValueError(
            "gap_tolerance must be finite and > 0, got "
            f"{gap_tolerance!r}"
        )
    if not 0.0 < tracking_tolerance <= 1.0:
        raise ValueError(
            "tracking_tolerance must lie in (0, 1], got "
            f"{tracking_tolerance!r}"
        )
    if max_wavefunction_determinants < 1:
        raise ValueError("max_wavefunction_determinants must be >= 1")

    energies, zero, full = _physical_states(
        h1e_cas,
        h2e_cas,
        n_core,
        n_active_orb,
        n_active_elec,
        nroots,
        mode,
        ms2,
        n_frozen,
        imaginary,
    )
    reference = _MSWavefunctionPoint(
        basis=basis,
        mo_coeff=np.asarray(C_mo),
        energies=energies,
        zero_states=zero,
        full_states=full,
    )
    gap = abs(float(energies[p_root] - energies[q_root]))
    if gap < gap_tolerance:
        raise ValueError(
            f"MS-CASPT2 NAC is gauge-singular for state gap {gap:.3e} Ha; "
            f"the verified nondegenerate envelope requires >= {gap_tolerance:.3e} Ha"
        )
    largest_state = max(len(state) for state in zero + full)
    if largest_state > max_wavefunction_determinants:
        raise NotImplementedError(
            f"MS-CASPT2 NAC wavefunction has {largest_state} determinants, "
            f"exceeding max_wavefunction_determinants={max_wavefunction_determinants}"
        )

    atoms = [(int(atom.Z), [float(value) for value in atom.xyz]) for atom in molecule.atoms]

    def displaced_point(atom_index: int, component: int, delta: float):
        moved = [(z, list(xyz)) for z, xyz in atoms]
        moved[atom_index][1][component] += delta
        displaced_molecule = Molecule([Atom(z, xyz) for z, xyz in moved])
        displaced_basis = type(basis)(displaced_molecule, basis.name)
        overlap_ao = np.asarray(compute_overlap(displaced_basis))
        metric = C_mo.T @ overlap_ao @ C_mo
        metric_values, metric_vectors = eigh(metric)
        if np.min(metric_values) <= 1e-10:
            raise RuntimeError(
                "MS-CASPT2 NAC orbital connection became linearly dependent"
            )
        connected_mo = C_mo @ (
            metric_vectors
            @ np.diag(1.0 / np.sqrt(metric_values))
            @ metric_vectors.T
        )
        kinetic = np.asarray(compute_kinetic(displaced_basis))
        nuclear = np.asarray(compute_nuclear(displaced_basis, displaced_molecule))
        eri_ao = np.asarray(compute_eri(displaced_basis))
        h1_start = connected_mo.T @ (kinetic + nuclear) @ connected_mo
        h2_start = np.einsum(
            "ap,bq,cr,ds,abcd->pqrs",
            connected_mo,
            connected_mo,
            connected_mo,
            connected_mo,
            eri_ao.transpose(0, 2, 1, 3),
            optimize=True,
        )
        sc = casscf(
            h1_start,
            h2_start,
            n_active_elec=n_active_elec,
            n_active_orb=n_active_orb,
            n_core=n_core,
            nuclear_repulsion=displaced_molecule.nuclear_repulsion(),
            ms2=ms2,
            nroots=len(sa_weights),
            weights=list(sa_weights),
            orbital_step=casscf_orbital_step,
            spin_pure=casscf_spin_pure,
            max_macro=casscf_max_macro,
            conv_tol_grad=casscf_conv_tol_grad,
            trust=casscf_trust,
        )
        if not sc.converged:
            raise RuntimeError(
                "displaced SA-CASSCF did not converge for MS-CASPT2 NAC: "
                f"|g|={sc.grad_norm:.3e} after {sc.n_iter} iterations"
            )
        displaced_energies, displaced_zero, displaced_full = _physical_states(
            sc.h1e_cas,
            sc.h2e_cas,
            n_core,
            n_active_orb,
            n_active_elec,
            nroots,
            mode,
            ms2,
            n_frozen,
            imaginary,
        )
        point = _MSWavefunctionPoint(
            basis=displaced_basis,
            mo_coeff=connected_mo @ sc.mo_rotation,
            energies=displaced_energies,
            zero_states=displaced_zero,
            full_states=displaced_full,
        )
        displaced_largest = max(
            len(state) for state in displaced_zero + displaced_full
        )
        if displaced_largest > max_wavefunction_determinants:
            raise NotImplementedError(
                "displaced MS-CASPT2 NAC wavefunction exceeds "
                f"max_wavefunction_determinants={max_wavefunction_determinants}"
            )
        cross_ao = np.asarray(compute_overlap_two_basis(basis, displaced_basis))
        cross_mo = C_mo.T @ cross_ao @ point.mo_coeff
        return _align_to_reference(
            reference, point, cross_mo, tracking_tolerance
        ), cross_mo

    result = np.zeros((len(atoms), 3))
    for atom_index in range(len(atoms)):
        for component in range(3):
            plus, overlap_plus = displaced_point(atom_index, component, fd_step)
            minus, overlap_minus = displaced_point(atom_index, component, -fd_step)
            plus_cache: dict[tuple[int, int], float] = {}
            minus_cache: dict[tuple[int, int], float] = {}
            def pair_value(
                point: _MSWavefunctionPoint,
                orbital_overlap: np.ndarray,
                cache: dict[tuple[int, int], float],
                left_root: int,
                right_root: int,
            ) -> float:
                return _wavefunction_overlap(
                    reference.zero_states[left_root],
                    point.full_states[right_root],
                    orbital_overlap,
                    cache,
                ) + _wavefunction_overlap(
                    reference.full_states[left_root],
                    point.zero_states[right_root],
                    orbital_overlap,
                    cache,
                )

            forward = (
                pair_value(plus, overlap_plus, plus_cache, q_root, p_root)
                - pair_value(minus, overlap_minus, minus_cache, q_root, p_root)
            ) / (4.0 * fd_step)
            reverse = (
                pair_value(plus, overlap_plus, plus_cache, p_root, q_root)
                - pair_value(minus, overlap_minus, minus_cache, p_root, q_root)
            ) / (4.0 * fd_step)
            # Exact real-state derivative couplings are anti-Hermitian.
            # Antisymmetrizing the two independently differenced directions
            # removes the O(h^2) gauge/tracking residue and makes the ordered
            # state-pair convention exact at the API boundary.
            result[atom_index, component] = 0.5 * (forward - reverse)
    return result
