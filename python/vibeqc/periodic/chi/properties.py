"""One-particle properties derivable from an AICCM2026DEV-B SCF state.

Only gauge-invariant quantities obtainable from the converged one-particle
density and finite-torus orbital spectrum live here.  A cell dipole is not a
periodic observable, polarization requires a Berry phase, and correlated
response properties require relaxed correlated density matrices; those are
rejected rather than approximated with a cluster position operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from ..._vibeqc_core import (
    BasisSet,
    PeriodicSystem,
    aiccm2026dev_b_mayer_pair_tensor,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from ...bands import BandStructure, KPath, band_structure
from ...basis_crystal import _ELEMENT_SYMBOLS
from .scf import (
    AICCM2026DevBFiniteTorusConvention,
    _density_blocks_per_k,
    _spin_density_blocks_per_k,
    aiccm2026dev_b_charge_bookkeeping,
    inverse_bloch_transform,
)

__all__ = [
    "AICCM2026DevBBondAnalysis",
    "AICCM2026DevBBondOrder",
    "AICCM2026DevBSCFProperties",
    "aiccm2026dev_b_band_structure",
    "aiccm2026dev_b_mayer_bond_orders",
    "aiccm2026dev_b_one_electron_expectation",
    "derive_aiccm2026dev_b_scf_properties",
]


@dataclass(frozen=True)
class AICCM2026DevBSCFProperties:
    """Gauge-invariant finite-torus SCF properties per primitive cell."""

    electronic_method: str
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    lattice_extension: tuple[int, int, int]
    n_electrons: float
    n_alpha: float | None
    n_beta: float | None
    mulliken_charges: np.ndarray
    mulliken_spin_populations: np.ndarray | None
    band_gap_hartree: float
    alpha_gap_hartree: float | None
    beta_gap_hartree: float | None
    is_metallic_at_finite_net: bool
    density_idempotency_error: float
    s_squared: float | None
    s_squared_ideal: float | None
    unavailable_periodic_observables: tuple[str, ...] = (
        "cell dipole (origin/gauge dependent)",
        "Berry-phase polarization (not implemented)",
        "SCF response properties and analytic derivatives (not implemented)",
        "correlated properties without a relaxed correlated 1-RDM",
    )


@dataclass(frozen=True)
class AICCM2026DevBBondOrder:
    """One primitive-cell Mayer bond order on the finite cyclic torus."""

    atom_i: int
    atom_j: int
    symbol_i: str
    symbol_j: str
    translation: tuple[int, int, int]
    distance_bohr: float
    order: float


@dataclass(frozen=True)
class AICCM2026DevBBondAnalysis:
    """Mayer bond-order analysis folded from the cyclic torus to one cell."""

    bonds: tuple[AICCM2026DevBBondOrder, ...]
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention
    lattice_extension: tuple[int, int, int]
    threshold: float
    translational_spread: float
    convention: str = (
        "Mayer bond order from the finite BvK density and overlap; "
        "primitive-cell bonds are averaged over all equivalent origins"
    )


def _blocks(value: object) -> list[np.ndarray]:
    # LatticeMatrixSet is no longer iterable at the binding level; read its
    # per-cell 2D blocks through the ``.blocks`` accessor (same list pybind11
    # exposes, mirroring how periodic_cosx_k.py / the fold builder consume it).
    lattice_blocks = getattr(value, "blocks", None)
    if lattice_blocks is not None:
        return [np.asarray(block) for block in lattice_blocks]
    array = np.asarray(value)
    if array.ndim == 2:
        return [array]
    return [np.asarray(block) for block in value]  # type: ignore[union-attr]


def _vector_blocks(value: object) -> list[np.ndarray]:
    lattice_blocks = getattr(value, "blocks", None)
    if lattice_blocks is not None:
        return [np.asarray(block) for block in lattice_blocks]
    array = np.asarray(value)
    if array.ndim == 1:
        return [array]
    return [np.asarray(block) for block in value]  # type: ignore[union-attr]


def _ao_atom_indices(basis: BasisSet) -> np.ndarray:
    indices: list[int] = []
    for shell in basis.shells():
        angular = int(shell.l)
        n_functions = (
            2 * angular + 1
            if bool(shell.pure)
            else (angular + 1) * (angular + 2) // 2
        )
        indices.extend([int(shell.atom_index)] * n_functions)
    if len(indices) != int(basis.nbasis):
        raise RuntimeError("basis shell metadata does not match the AO dimension")
    return np.asarray(indices, dtype=int)


def _mesh_from_result(result: object) -> tuple[int, int, int]:
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    if diagnostics is None:
        raise TypeError("result is not an AICCM2026DEV-B SCF result")
    return tuple(int(value) for value in diagnostics.mesh)


def _finite_torus_convention_from_result(
    result: object,
) -> AICCM2026DevBFiniteTorusConvention:
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    if diagnostics is None:
        raise TypeError("result is not an AICCM2026DEV-B SCF result")
    convention = getattr(diagnostics, "finite_torus_convention", None)
    if convention is None:
        convention = getattr(result, "finite_torus_convention", None)
    if convention is None:
        raise TypeError(
            "AICCM2026DEV-B properties require a finite-torus convention "
            "descriptor on the SCF result"
        )
    return convention


def _charge_bookkeeping_from_result(
    result: object,
    system: PeriodicSystem,
):
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    if diagnostics is None:
        raise TypeError("result is not an AICCM2026DEV-B SCF result")
    effective_electrons = getattr(diagnostics, "effective_electron_count", None)
    effective_charges = getattr(diagnostics, "effective_nuclear_charges", None)
    if effective_electrons is not None and effective_charges is not None:
        return int(effective_electrons), np.asarray(effective_charges, dtype=float)
    bookkeeping = aiccm2026dev_b_charge_bookkeeping(system, None)
    return (
        int(bookkeeping.effective_electrons),
        np.asarray(bookkeeping.effective_nuclear_charges, dtype=float),
    )


def _cell_residues(mesh: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return [
        tuple(int(value) for value in residue)
        for residue in product(*(range(int(n)) for n in mesh))
    ]


def _centered_translation(
    residue: Sequence[int],
    mesh: tuple[int, int, int],
) -> tuple[int, int, int]:
    out: list[int] = []
    for value, period in zip(residue, mesh):
        integer = int(value)
        if 2 * integer > int(period):
            integer -= int(period)
        out.append(integer)
    return tuple(out)  # type: ignore[return-value]


def _positive_residue(
    translation: Sequence[int],
    mesh: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(int(value) % int(period) for value, period in zip(translation, mesh))


def _translation_vector(
    system: PeriodicSystem,
    translation: Sequence[int],
) -> np.ndarray:
    lattice = np.asarray(system.lattice, dtype=float)
    return lattice @ np.asarray(translation, dtype=float)


def _element_symbol(atomic_number: int) -> str:
    if 0 <= int(atomic_number) < len(_ELEMENT_SYMBOLS):
        symbol = _ELEMENT_SYMBOLS[int(atomic_number)]
        if symbol:
            return symbol
    return f"Z{int(atomic_number)}"


def _as_real_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    real = np.real_if_close(np.asarray(matrix), tol=1000)
    if np.iscomplexobj(real):
        max_imag = float(np.max(np.abs(real.imag)))
        if max_imag > 1.0e-8:
            raise RuntimeError(
                f"{label} has non-negligible imaginary residue {max_imag:.3e}"
            )
        real = real.real
    return np.asarray(real, dtype=float)


def _finite_torus_full_matrix(
    blocks_by_residue: dict[tuple[int, int, int], np.ndarray],
    mesh: tuple[int, int, int],
) -> np.ndarray:
    residues = _cell_residues(mesh)
    n_cells = len(residues)
    nbf = next(iter(blocks_by_residue.values())).shape[0]
    full = np.zeros((n_cells * nbf, n_cells * nbf), dtype=float)
    for origin_index, origin in enumerate(residues):
        row = slice(origin_index * nbf, (origin_index + 1) * nbf)
        for target_index, target in enumerate(residues):
            delta = tuple(
                (int(target[axis]) - int(origin[axis])) % int(mesh[axis])
                for axis in range(3)
            )
            col = slice(target_index * nbf, (target_index + 1) * nbf)
            full[row, col] = blocks_by_residue[delta]
    return full


def _stack_residue_blocks(
    blocks_by_residue: dict[tuple[int, int, int], np.ndarray],
    mesh: tuple[int, int, int],
) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack([blocks_by_residue[residue] for residue in _cell_residues(mesh)]),
        dtype=float,
    )


def _finite_torus_blocks_from_k(
    matrix_k: object,
    result: object,
    mesh: tuple[int, int, int],
) -> dict[tuple[int, int, int], np.ndarray]:
    residues = _cell_residues(mesh)
    blocks = inverse_bloch_transform(
        _blocks(matrix_k),
        np.asarray(result.kpoints_frac, dtype=float),
        residues,
        np.asarray(result.kpoint_weights, dtype=float),
    )
    return {
        residue: _as_real_matrix(block, "finite-torus inverse Bloch block")
        for residue, block in zip(residues, blocks)
    }


def _finite_torus_lattice_set(
    matrix_k: object,
    result: object,
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
):
    translations = [
        _centered_translation(residue, mesh) for residue in _cell_residues(mesh)
    ]
    blocks_by_residue = _finite_torus_blocks_from_k(matrix_k, result, mesh)
    block_list = [
        blocks_by_residue[_positive_residue(translation, mesh)]
        for translation in translations
    ]
    max_radius = max(
        float(np.linalg.norm(_translation_vector(system, t)))
        for t in translations
    )
    cell_pool = direct_lattice_cells(system, max_radius + 1.0e-8)
    cell_by_index = {
        tuple(int(x) for x in np.asarray(cell.index, dtype=int)): cell
        for cell in cell_pool
    }
    try:
        cells = [cell_by_index[translation] for translation in translations]
    except KeyError as exc:
        raise RuntimeError(
            "could not build a LatticeMatrixSet for the finite-torus band path; "
            "the requested cyclic translation was not enumerated by direct_lattice_cells"
        ) from exc
    nbf = int(block_list[0].shape[0])
    return make_lattice_matrix_set(nbf, cells, block_list)


def _mayer_atom_pair_matrix(
    density: np.ndarray,
    overlap: np.ndarray,
    ao_atoms: np.ndarray,
    *,
    density_beta: np.ndarray | None = None,
) -> np.ndarray:
    n_atoms = int(np.max(ao_atoms)) + 1
    if density_beta is None:
        ps = density @ overlap
        contributions = np.real(ps * ps.T)
    else:
        ps_alpha = density @ overlap
        ps_beta = density_beta @ overlap
        contributions = 2.0 * np.real(ps_alpha * ps_alpha.T + ps_beta * ps_beta.T)
    out = np.zeros((n_atoms, n_atoms), dtype=float)
    for mu, atom_mu in enumerate(ao_atoms):
        np.add.at(out, (int(atom_mu), ao_atoms), contributions[mu, :])
    return out


def _canonical_bond_key(
    atom_i: int,
    atom_j: int,
    translation: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    forward = (int(atom_i), int(atom_j), *translation)
    reverse = (int(atom_j), int(atom_i), *(-int(x) for x in translation))
    return min(forward, reverse)


def _fold_mayer_bonds(
    pair_matrix: np.ndarray,
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
    *,
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention,
    threshold: float,
    max_bonds: int | None,
) -> AICCM2026DevBBondAnalysis:
    residues = _cell_residues(mesh)
    n_cells = len(residues)
    n_unit_atoms = len(system.unit_cell)
    atom_positions = np.asarray([atom.xyz for atom in system.unit_cell], dtype=float)
    totals: dict[tuple[int, int, int, int, int], list[float]] = {}
    labels: dict[
        tuple[int, int, int, int, int],
        tuple[int, int, tuple[int, int, int]],
    ] = {}
    for origin_cell_index, origin in enumerate(residues):
        for target_cell_index, target in enumerate(residues):
            residue = tuple(
                (int(target[axis]) - int(origin[axis])) % int(mesh[axis])
                for axis in range(3)
            )
            translation = _centered_translation(residue, mesh)
            for atom_i in range(n_unit_atoms):
                full_i = origin_cell_index * n_unit_atoms + atom_i
                for atom_j in range(n_unit_atoms):
                    if all(value == 0 for value in translation) and atom_i == atom_j:
                        continue
                    full_j = target_cell_index * n_unit_atoms + atom_j
                    key = _canonical_bond_key(atom_i, atom_j, translation)
                    totals.setdefault(key, []).append(
                        float(pair_matrix[full_i, full_j])
                    )
                    labels[key] = (atom_i, atom_j, translation)
    bonds: list[AICCM2026DevBBondOrder] = []
    max_spread = 0.0
    seen: set[tuple[int, int, int, int, int]] = set()
    for key, values in totals.items():
        if key in seen:
            continue
        seen.add(key)
        avg = float(np.mean(values))
        max_spread = max(max_spread, float(np.max(values) - np.min(values)))
        if avg < threshold:
            continue
        atom_i, atom_j, translation = labels[key]
        displacement = (
            atom_positions[atom_j]
            + _translation_vector(system, translation)
            - atom_positions[atom_i]
        )
        bonds.append(
            AICCM2026DevBBondOrder(
                atom_i=atom_i,
                atom_j=atom_j,
                symbol_i=_element_symbol(int(system.unit_cell[atom_i].Z)),
                symbol_j=_element_symbol(int(system.unit_cell[atom_j].Z)),
                translation=translation,
                distance_bohr=float(np.linalg.norm(displacement)),
                order=avg,
            )
        )
    bonds.sort(
        key=lambda bond: (
            bond.distance_bohr,
            -bond.order,
            bond.atom_i,
            bond.atom_j,
            bond.translation,
        )
    )
    if max_bonds is not None:
        bonds = bonds[: int(max_bonds)]
    return AICCM2026DevBBondAnalysis(
        bonds=tuple(bonds),
        finite_torus_convention=finite_torus_convention,
        lattice_extension=mesh,
        threshold=float(threshold),
        translational_spread=max_spread,
    )


def _fold_mayer_bond_tensor(
    pair_tensor: np.ndarray,
    system: PeriodicSystem,
    mesh: tuple[int, int, int],
    *,
    finite_torus_convention: AICCM2026DevBFiniteTorusConvention,
    threshold: float,
    max_bonds: int | None,
) -> AICCM2026DevBBondAnalysis:
    residues = _cell_residues(mesh)
    tensor = np.asarray(pair_tensor, dtype=float)
    n_unit_atoms = len(system.unit_cell)
    if tensor.shape != (len(residues), n_unit_atoms, n_unit_atoms):
        raise ValueError(
            "pair_tensor must have shape "
            f"({len(residues)}, {n_unit_atoms}, {n_unit_atoms}), got "
            f"{tensor.shape}"
        )
    atom_positions = np.asarray([atom.xyz for atom in system.unit_cell], dtype=float)
    totals: dict[tuple[int, int, int, int, int], list[float]] = {}
    labels: dict[
        tuple[int, int, int, int, int],
        tuple[int, int, tuple[int, int, int]],
    ] = {}
    for residue_index, residue in enumerate(residues):
        translation = _centered_translation(residue, mesh)
        for atom_i in range(n_unit_atoms):
            for atom_j in range(n_unit_atoms):
                if all(value == 0 for value in translation) and atom_i == atom_j:
                    continue
                key = _canonical_bond_key(atom_i, atom_j, translation)
                totals.setdefault(key, []).append(
                    float(tensor[residue_index, atom_i, atom_j])
                )
                labels[key] = (atom_i, atom_j, translation)

    bonds: list[AICCM2026DevBBondOrder] = []
    max_spread = 0.0
    seen: set[tuple[int, int, int, int, int]] = set()
    for key, values in totals.items():
        if key in seen:
            continue
        seen.add(key)
        avg = float(np.mean(values))
        max_spread = max(max_spread, float(np.max(values) - np.min(values)))
        if avg < threshold:
            continue
        atom_i, atom_j, translation = labels[key]
        displacement = (
            atom_positions[atom_j]
            + _translation_vector(system, translation)
            - atom_positions[atom_i]
        )
        bonds.append(
            AICCM2026DevBBondOrder(
                atom_i=atom_i,
                atom_j=atom_j,
                symbol_i=_element_symbol(int(system.unit_cell[atom_i].Z)),
                symbol_j=_element_symbol(int(system.unit_cell[atom_j].Z)),
                translation=translation,
                distance_bohr=float(np.linalg.norm(displacement)),
                order=avg,
            )
        )
    bonds.sort(
        key=lambda bond: (
            bond.distance_bohr,
            -bond.order,
            bond.atom_i,
            bond.atom_j,
            bond.translation,
        )
    )
    if max_bonds is not None:
        bonds = bonds[: int(max_bonds)]
    return AICCM2026DevBBondAnalysis(
        bonds=tuple(bonds),
        finite_torus_convention=finite_torus_convention,
        lattice_extension=mesh,
        threshold=float(threshold),
        translational_spread=max_spread,
    )


def aiccm2026dev_b_one_electron_expectation(
    result: object,
    operator_k: Sequence[np.ndarray] | np.ndarray,
    *,
    spin: str = "total",
) -> float:
    """Return ``sum_k w_k Tr[P(k) O(k)]`` per primitive cell.

    ``spin`` may be ``total``, ``alpha``, ``beta``, or ``difference`` for an
    unrestricted reference.  Restricted references only accept ``total``.
    The physical units are the units of the supplied operator matrix.
    """

    if not hasattr(result, "aiccm2026dev_b"):
        raise TypeError("result is not an AICCM2026DEV-B SCF result")
    operators = _blocks(operator_k)
    weights = np.asarray(result.kpoint_weights, dtype=float)
    if len(operators) != len(weights):
        raise ValueError("operator block count does not match the finite character net")
    if hasattr(result, "density_alpha"):
        alpha = _blocks(result.density_alpha)
        beta = _blocks(result.density_beta)
        if spin == "alpha":
            densities = alpha
        elif spin == "beta":
            densities = beta
        elif spin == "total":
            densities = [a + b for a, b in zip(alpha, beta)]
        elif spin == "difference":
            densities = [a - b for a, b in zip(alpha, beta)]
        else:
            raise ValueError("spin must be total, alpha, beta, or difference")
    else:
        if spin != "total":
            raise ValueError("restricted references only define spin='total'")
        densities = _blocks(result.density)
    value = sum(
        weight * np.einsum("mn,nm->", density, operator, optimize=True)
        for weight, density, operator in zip(weights, densities, operators)
    )
    if abs(complex(value).imag) > 1e-9:
        raise RuntimeError("one-electron expectation has a non-negligible imaginary part")
    return float(complex(value).real)


def aiccm2026dev_b_band_structure(
    result: object,
    system: PeriodicSystem,
    kpath: KPath,
    *,
    spin: str = "total",
    n_electrons_per_cell: int | None = None,
) -> BandStructure:
    """Sample the converged finite-torus Fock matrix along a k-path.

    The returned bands are exact at the cyclic cluster's folded character
    points.  Between them the path is the finite-torus interpolation defined by
    the reconstructed real-space Fock blocks, so it should be converged with
    respect to ``lattice_extension`` before quantitative interpretation.
    """

    if spin not in {"total", "alpha", "beta"}:
        raise ValueError("spin must be total, alpha, or beta")
    if spin != "total":
        attr = f"fock_{spin}"
        if not hasattr(result, attr):
            raise ValueError("spin-resolved bands require an unrestricted result")
        fock_k = getattr(result, attr)
    else:
        fock_k = getattr(result, "fock")
    mesh = _mesh_from_result(result)
    fock_real = _finite_torus_lattice_set(fock_k, result, system, mesh)
    overlap_real = _finite_torus_lattice_set(
        getattr(result, "overlap"),
        result,
        system,
        mesh,
    )
    effective_electrons, _ = _charge_bookkeeping_from_result(result, system)
    n_elec = effective_electrons if n_electrons_per_cell is None else int(
        n_electrons_per_cell
    )
    bands = band_structure(
        fock_real,
        overlap_real,
        kpath,
        n_electrons_per_cell=n_elec,
    )
    convention = _finite_torus_convention_from_result(result)
    exact_exchange_assembly = getattr(
        result,
        "exact_exchange_assembly",
        None,
    )
    if exact_exchange_assembly is None:
        raise RuntimeError(
            "χ-CCM-B band metadata requires the versioned exact-exchange "
            "assembly"
        )
    bands.metadata.update(
        {
            "method": "aiccm2026dev-b",
            "method_name": "χ-CCM",
            "finite_torus_convention": convention,
            "coulomb_kernel": convention.coulomb_kernel,
            "exchange_q0": convention.exchange_q0,
            "exchange_q0_applicability": convention.exchange_q0_applicability,
            "exact_exchange_assembly": exact_exchange_assembly,
            "boundary_model": convention.boundary_model,
        }
    )
    return bands


def aiccm2026dev_b_mayer_bond_orders(
    result: object,
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    threshold: float = 0.05,
    max_bonds: int | None = 40,
) -> AICCM2026DevBBondAnalysis:
    """Return primitive-cell Mayer bond orders from a χ-CCM SCF result."""

    mesh = _mesh_from_result(result)
    ao_atoms_unit = _ao_atom_indices(basis)
    overlap_blocks = _finite_torus_blocks_from_k(
        getattr(result, "overlap"),
        result,
        mesh,
    )
    overlap_stack = _stack_residue_blocks(overlap_blocks, mesh)
    del overlap_blocks
    effective_electrons, _ = _charge_bookkeeping_from_result(result, system)
    if hasattr(result, "density_alpha"):
        two_s = int(system.multiplicity) - 1
        n_alpha = (effective_electrons + two_s) // 2
        n_beta = (effective_electrons - two_s) // 2
        alpha_blocks = _finite_torus_blocks_from_k(
            _spin_density_blocks_per_k(result, "alpha", n_alpha),
            result,
            mesh,
        )
        beta_blocks = _finite_torus_blocks_from_k(
            _spin_density_blocks_per_k(result, "beta", n_beta),
            result,
            mesh,
        )
        pair_tensor = aiccm2026dev_b_mayer_pair_tensor(
            _stack_residue_blocks(alpha_blocks, mesh),
            overlap_stack,
            np.asarray(ao_atoms_unit, dtype=np.int64),
            np.asarray(mesh, dtype=np.int64),
            density_beta_blocks=_stack_residue_blocks(beta_blocks, mesh),
        )
        del alpha_blocks, beta_blocks
    else:
        density_blocks = _finite_torus_blocks_from_k(
            _density_blocks_per_k(result, effective_electrons),
            result,
            mesh,
        )
        pair_tensor = aiccm2026dev_b_mayer_pair_tensor(
            _stack_residue_blocks(density_blocks, mesh),
            overlap_stack,
            np.asarray(ao_atoms_unit, dtype=np.int64),
            np.asarray(mesh, dtype=np.int64),
        )
        del density_blocks
    del overlap_stack
    return _fold_mayer_bond_tensor(
        pair_tensor,
        system,
        mesh,
        finite_torus_convention=_finite_torus_convention_from_result(result),
        threshold=threshold,
        max_bonds=max_bonds,
    )


def _gap(energies: object, n_occ: int) -> float:
    blocks = _vector_blocks(energies)
    if n_occ <= 0 or n_occ >= blocks[0].size:
        return float("inf")
    homo = max(float(np.asarray(block)[n_occ - 1]) for block in blocks)
    lumo = min(float(np.asarray(block)[n_occ]) for block in blocks)
    return lumo - homo


def derive_aiccm2026dev_b_scf_properties(
    result: object,
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    metallic_gap_tolerance_hartree: float = 1e-6,
) -> AICCM2026DevBSCFProperties:
    """Derive all currently supported gauge-invariant SCF properties."""

    if not bool(getattr(result, "converged", False)):
        raise ValueError("properties require a converged AICCM2026DEV-B SCF")
    diagnostics = getattr(result, "aiccm2026dev_b", None)
    if diagnostics is None:
        raise TypeError("result is not an AICCM2026DEV-B SCF result")
    overlap = _blocks(result.overlap)
    weights = np.asarray(result.kpoint_weights, dtype=float)
    ao_atoms = _ao_atom_indices(basis)
    n_atoms = len(system.unit_cell)

    def populations(densities: list[np.ndarray]) -> np.ndarray:
        population = np.zeros(n_atoms)
        for weight, density, metric in zip(weights, densities, overlap):
            gross = np.real(np.diag(density @ metric))
            population += weight * np.bincount(
                ao_atoms,
                weights=gross,
                minlength=n_atoms,
            )
        return population

    ne, effective_charges = _charge_bookkeeping_from_result(result, system)
    if hasattr(result, "density_alpha"):
        alpha_density = _blocks(result.density_alpha)
        beta_density = _blocks(result.density_beta)
        alpha_pop = populations(alpha_density)
        beta_pop = populations(beta_density)
        total_pop = alpha_pop + beta_pop
        spin_pop = alpha_pop - beta_pop
        two_s = int(system.multiplicity) - 1
        n_alpha_occ = (ne + two_s) // 2
        n_beta_occ = (ne - two_s) // 2
        alpha_gap = _gap(result.mo_energies_alpha, n_alpha_occ)
        beta_gap = _gap(result.mo_energies_beta, n_beta_occ)
        gap = min(alpha_gap, beta_gap)
        n_alpha = float(alpha_pop.sum())
        n_beta = float(beta_pop.sum())
    else:
        total_pop = populations(_blocks(result.density))
        spin_pop = None
        gap = _gap(result.mo_energies, ne // 2)
        alpha_gap = None
        beta_gap = None
        n_alpha = None
        n_beta = None
    charges = effective_charges - total_pop
    return AICCM2026DevBSCFProperties(
        electronic_method=str(diagnostics.electronic_method),
        finite_torus_convention=_finite_torus_convention_from_result(result),
        lattice_extension=tuple(int(value) for value in diagnostics.mesh),
        n_electrons=float(total_pop.sum()),
        n_alpha=n_alpha,
        n_beta=n_beta,
        mulliken_charges=charges,
        mulliken_spin_populations=spin_pop,
        band_gap_hartree=float(gap),
        alpha_gap_hartree=(None if alpha_gap is None else float(alpha_gap)),
        beta_gap_hartree=(None if beta_gap is None else float(beta_gap)),
        is_metallic_at_finite_net=bool(gap <= metallic_gap_tolerance_hartree),
        density_idempotency_error=float(diagnostics.density_idempotency_error),
        s_squared=getattr(diagnostics, "s_squared", None),
        s_squared_ideal=getattr(diagnostics, "s_squared_ideal", None),
    )
