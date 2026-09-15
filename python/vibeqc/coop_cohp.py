"""COOP/COHP bonding analysis for periodic systems.

COOP (Crystal Orbital Overlap Population) and COHP (Crystal Orbital
Hamilton Population) are energy-resolved projections of the overlap or
Hamiltonian matrix onto atom-pair subspaces. They are the standard
tools for bonding/antibonding analysis of periodic systems.

At each k-point, for each band n and atom pair (A, B)::

    COOP_{AB}(E) = Σ_k w_k Σ_n  [ Σ_{μ∈A,ν∈B}  S_{μν}(k) · C*_{μn}(k) · C_{νn}(k) ] · δ(E - ε_n(k))
    COHP_{AB}(E) = Σ_k w_k Σ_n  [ Σ_{μ∈A,ν∈B}  H_{μν}(k) · C*_{μn}(k) · C_{νn}(k) ] · δ(E - ε_n(k))

where the delta function is broadened by a Gaussian of width *sigma*.

Negative COHP = bonding, positive = antibonding. The energy integral
up to E_F gives ICOOP/ICOHP — the bond strength.

Following the Lobster convention, -COHP is plotted / returned so that
bonding states appear positive (same sign convention as COOP).

References
----------
* Hughbanks & Hoffmann, JACS 105, 3528 (1983) -- original COOP
* Dronskowski & Blöchl, J. Phys. Chem. 97, 8617 (1993) -- COHP
* Deringer, Tchougréeff & Dronskowski, J. Phys. Chem. A 115, 5461 (2011)
* Maintz et al., J. Comput. Chem. 37, 1030 (2016) -- Lobster 3.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    BlochKMesh,
    LatticeMatrixSet,
    PeriodicSystem,
    bloch_sum,
    diagonalize_bloch,
)
from ._vibeqc_core import (
    cohp_weights_k as _cohp_weights_k,
)
from ._vibeqc_core import (
    coop_weights_k as _coop_weights_k,
)
from ._vibeqc_core import (
    gaussian_broaden_projected as _gaussian_broaden_projected,
)
from .bands import _atom_label, _shell_to_atom

__all__ = [
    "COOPCOHPResult",
    "compute_coop_cohp",
    "ao_pairs_per_atom_pair",
    "periodic_mayer_bond_orders",
]

# Bohr -> Ångström (CODATA 2018).
_BOHR_TO_ANGSTROM = 0.529177210903


# ---------------------------------------------------------------------------
# AO-pair grouping
# ---------------------------------------------------------------------------


def ao_pairs_per_atom_pair(
    system: PeriodicSystem,
    basis: BasisSet,
    pair_distance_cutoff: float = 8.0,
) -> Dict[Tuple[int, int], Tuple[List[int], List[int]]]:
    """Partition AO indices into (atom-A, atom-B) pairs.

    Returns a dict keyed by ``(atom_idx_A, atom_idx_B)`` with A < B
    (0-based indices), values ``(ao_indices_A, ao_indices_B)``. Only
    pairs whose interatomic distance is ≤ ``pair_distance_cutoff``
    (bohr) are included.

    Parameters
    ----------
    system : PeriodicSystem
        Periodic system whose unit-cell atoms provide the pair list.
    basis : BasisSet
        AO basis set.
    pair_distance_cutoff : float
        Maximum interatomic distance (bohr) to consider a pair.
    """
    ao_to_atom = _shell_to_atom(basis)
    n_atoms = len(system.unit_cell)
    coords = np.asarray([np.asarray(atom.xyz) for atom in system.unit_cell])

    out: Dict[Tuple[int, int], Tuple[List[int], List[int]]] = {}
    for a in range(n_atoms):
        ao_a = [int(i) for i in np.where(ao_to_atom == a)[0]]
        if not ao_a:
            continue
        for b in range(a + 1, n_atoms):
            ao_b = [int(i) for i in np.where(ao_to_atom == b)[0]]
            if not ao_b:
                continue
            dist = float(np.linalg.norm(coords[a] - coords[b]))
            if dist <= pair_distance_cutoff:
                out[(a, b)] = (ao_a, ao_b)
    return out


# ---------------------------------------------------------------------------
# Pair metadata helper
# ---------------------------------------------------------------------------


def _symbol_from_label(label: str) -> str:
    """Extract element symbol from an ``'_atom_label'`` string like ``'H1'``, ``'Ne2'``."""
    return label.rstrip("0123456789")


def _pair_metadata(
    system: PeriodicSystem,
    pairs: Sequence[Tuple[int, int]],
) -> List[dict]:
    """Build metadata for each pair: atom indices, symbols, distance."""
    coords = np.asarray([np.asarray(atom.xyz) for atom in system.unit_cell])
    atoms = list(system.unit_cell)
    result: List[dict] = []
    for a, b in pairs:
        dist = float(np.linalg.norm(coords[a] - coords[b]))
        result.append(
            {
                "i": int(a),
                "j": int(b),
                "symbol_i": _symbol_from_label(
                    _atom_label(int(atoms[a].Z), int(a) + 1)
                ),
                "symbol_j": _symbol_from_label(
                    _atom_label(int(atoms[b].Z), int(b) + 1)
                ),
                "distance_ang": float(dist * _BOHR_TO_ANGSTROM),
            }
        )
    return result


# Bohr -> Ångström (CODATA 2018).
_BOHR_TO_ANGSTROM = 0.529177210903


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass
class COOPCOHPResult:
    """COOP/COHP for a set of atom pairs over an energy grid.

    Attributes
    ----------
    energies : np.ndarray, shape ``(n_e,)``
        Energy grid in Hartree.
    coop : np.ndarray
        COOP(E) where bonding = positive. Shape ``(n_pairs, n_e)``
        (restricted) or ``(n_spin, n_pairs, n_e)`` (spin-polarized).
    cohp : np.ndarray or None
        -COHP(E) where bonding = positive (Lobster convention). Same
        shape as ``coop``.  ``None`` if no Hamiltonian was supplied.
    integrated_coop : np.ndarray, shape ``(n_pairs,)``
        ICOOP, the integral of COOP(E) up to ``fermi_energy``.
    integrated_cohp : np.ndarray or None
        -ICOHP, same convention.
    pairs : list[dict]
        Per-pair metadata: ``i``, ``j``, ``symbol_i``, ``symbol_j``,
        ``distance_ang``.
    fermi_energy : float
        Fermi energy in Hartree.
    sigma : float
        Gaussian broadening width in Hartree.
    """

    energies: np.ndarray
    coop: np.ndarray
    cohp: Optional[np.ndarray]
    integrated_coop: np.ndarray
    integrated_cohp: Optional[np.ndarray]
    pairs: list
    fermi_energy: float
    sigma: float


# ---------------------------------------------------------------------------
# Core compute function
# ---------------------------------------------------------------------------


def compute_coop_cohp(
    F_terms: Union[LatticeMatrixSet, Sequence[LatticeMatrixSet]],
    S_real: LatticeMatrixSet,
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    *,
    H_terms: Union[LatticeMatrixSet, Sequence[LatticeMatrixSet], None] = None,
    include_cohp: bool = False,
    F_terms_beta: Union[LatticeMatrixSet, Sequence[LatticeMatrixSet], None] = None,
    pair_distance_cutoff: float = 8.0,
    sigma: float = 0.01,
    energy_grid: Optional[np.ndarray] = None,
    n_grid: int = 401,
    pad: float = 5.0,
    n_electrons_per_cell: Optional[int] = None,
) -> COOPCOHPResult:
    """Compute COOP and optionally COHP for a periodic system.

    For every k-point in ``kmesh``, Bloch-sums ``F(k)``, ``S(k)``, and
    diagonalizes, and projects the overlap and analyzed Hamiltonian
    density onto atom-pair subspaces with Gaussian broadening.

    Parameters
    ----------
    F_terms : LatticeMatrixSet or sequence of LatticeMatrixSet
        Fock matrix (or Fock terms) for the α spin channel. When multiple
        terms are given they are Bloch-summed independently and added in
        k-space.
    S_real : LatticeMatrixSet
        Real-space overlap matrix.
    system : PeriodicSystem
    basis : BasisSet
    kmesh : BlochKMesh
        Monkhorst-Pack k-mesh.
    H_terms : LatticeMatrixSet, sequence of LatticeMatrixSet, or None
        Compatibility input enabling COHP. Its Bloch sum must equal the
        analyzed Fock in every spin channel. A different operator is refused
        rather than labeled COHP; explicit operator weights can be computed
        with the low-level ``cohp_weights_k`` kernel.
    include_cohp : bool
        Compute COHP with the same spin Hamiltonian used for the eigenpairs.
    F_terms_beta : LatticeMatrixSet, sequence, or None
        Fock terms for the β spin channel. If given, spin-polarized
        COOP/COHP is computed: ``result.coop`` and ``result.cohp``
        have shape ``(2, n_pairs, n_e)``.
    pair_distance_cutoff : float
        Maximum interatomic distance (bohr) to include a pair.
    sigma : float
        Gaussian broadening width (Hartree).
    energy_grid : np.ndarray or None
        If ``None``, constructed from the eigenvalue range.
    n_grid : int
        Number of energy grid points when ``energy_grid`` is ``None``.
    pad : float
        Number of sigma to extend the grid beyond the eigenvalue range.
    n_electrons_per_cell : int or None
        Used to infer the Fermi energy (highest occupied eigenvalue).

    Returns
    -------
    COOPCOHPResult
        ``.coop`` and ``.cohp`` shape: ``(n_pairs, n_e)`` (restricted)
        or ``(2, n_pairs, n_e)`` (spin-polarized).
        ``.integrated_coop`` / ``.integrated_cohp``: ``(n_pairs,)`` or
        ``(2, n_pairs,)``.
    """
    # Normalize F_terms to a list
    if isinstance(F_terms, LatticeMatrixSet):
        fock_list: Sequence[LatticeMatrixSet] = [F_terms]
    else:
        fock_list = F_terms
    if len(fock_list) == 0:
        raise ValueError("compute_coop_cohp: F_terms must not be empty")

    # Normalize beta Fock terms if given
    if F_terms_beta is not None:
        if isinstance(F_terms_beta, LatticeMatrixSet):
            fock_list_beta: Optional[Sequence[LatticeMatrixSet]] = [F_terms_beta]
        else:
            fock_list_beta = list(F_terms_beta)
        if len(fock_list_beta) == 0:
            raise ValueError("compute_coop_cohp: F_terms_beta must not be empty")
        spin_channels: list[Sequence[LatticeMatrixSet]] = [fock_list, fock_list_beta]
        n_spin = 2
    else:
        spin_channels = [fock_list]
        n_spin = 1

    # Normalize H_terms to a list
    if H_terms is None:
        h_list: Optional[Sequence[LatticeMatrixSet]] = None
    elif isinstance(H_terms, LatticeMatrixSet):
        h_list = [H_terms]
    else:
        h_list = list(H_terms)
    compute_cohp = bool(include_cohp) or bool(h_list)

    nbf = S_real.nbf

    # Build atom-pair groups
    pairs_raw = ao_pairs_per_atom_pair(system, basis, pair_distance_cutoff)
    if not pairs_raw:
        raise ValueError(
            "compute_coop_cohp: no atom pairs found within "
            f"pair_distance_cutoff={pair_distance_cutoff} bohr"
        )
    pair_keys = sorted(pairs_raw.keys())  # stable ordering
    n_pairs = len(pair_keys)

    # Pre-extract AO index arrays
    ao_indices: List[Tuple[np.ndarray, np.ndarray]] = []
    for a, b in pair_keys:
        ao_a, ao_b = pairs_raw[(a, b)]
        ao_indices.append(
            (np.asarray(ao_a, dtype=np.intp), np.asarray(ao_b, dtype=np.intp))
        )

    # k-point arrays
    kpoints_cart = np.asarray([np.asarray(k) for k in kmesh.kpoints])
    k_weights = np.asarray(kmesh.weights, dtype=float)
    if abs(k_weights.sum() - 1.0) > 1e-8:
        k_weights = k_weights / k_weights.sum()

    n_k = kpoints_cart.shape[0]

    # Build the C++-friendly pair list once.
    _cpp_pairs = [(a.tolist(), b.tolist()) for a, b in ao_indices]

    # Accumulate per-spin results
    coop_projections: list[np.ndarray] = []
    cohp_projections: list[np.ndarray] = []
    coop_integrated: list[np.ndarray] = []
    cohp_integrated: list[np.ndarray] = []
    all_energies: list[np.ndarray] = []  # for Fermi energy estimation
    all_coop_weights = []
    all_cohp_weights = []

    for spin_idx, fock_spin in enumerate(spin_channels):
        # Per-k storage for this spin
        energies_per_k = np.empty((n_k, nbf), dtype=float)
        coop_weights = np.zeros((n_pairs, n_k, nbf), dtype=float)
        cohp_weights = (
            np.zeros((n_pairs, n_k, nbf), dtype=float) if compute_cohp else None
        )

        for ki in range(n_k):
            # Bloch-sum F(k) from all terms
            Fk = bloch_sum(fock_spin[0], kpoints_cart[ki])
            for term in fock_spin[1:]:
                Fk = Fk + bloch_sum(term, kpoints_cart[ki])
            Sk = bloch_sum(S_real, kpoints_cart[ki])
            # Hermitize
            Fk = 0.5 * (Fk + Fk.conj().T)
            Sk = 0.5 * (Sk + Sk.conj().T)

            # Diagonalize
            bd = diagonalize_bloch(Fk, Sk)
            energies_per_k[ki, :] = np.asarray(bd.energies)
            Ck = np.asarray(bd.coefficients)  # (nbf, nbf)

            # COOP weights via C++ kernel
            coop_weights[:, ki, :] = np.asarray(_coop_weights_k(Ck, Sk, _cpp_pairs))

            # COHP weights via C++ kernel
            if compute_cohp:
                if h_list:
                    Hk = sum(bloch_sum(term, kpoints_cart[ki]) for term in h_list)
                    Hk = 0.5 * (Hk + Hk.conj().T)
                    if not np.allclose(Hk, Fk, atol=1e-12, rtol=1e-12):
                        raise ValueError(
                            "COHP requires the analyzed spin Hamiltonian; H_terms "
                            "describes a different operator. Use include_cohp=True "
                            "for standard COHP or cohp_weights_k for explicit "
                            "operator projections."
                        )
                cohp_weights[:, ki, :] = np.asarray(_cohp_weights_k(Ck, Fk, _cpp_pairs))

        all_energies.append(energies_per_k)
        all_coop_weights.append(coop_weights)
        all_cohp_weights.append(cohp_weights)

    # Resolve one grid from every spin before broadening. A grid fixed from
    # alpha alone can silently discard beta bands of a magnetic state.
    if energy_grid is None:
        e_min = min(float(e.min()) for e in all_energies) - pad * sigma
        e_max = max(float(e.max()) for e in all_energies) + pad * sigma
        energy_grid = np.linspace(e_min, e_max, n_grid)
    for energies_per_k, coop_weights, cohp_weights in zip(
        all_energies, all_coop_weights, all_cohp_weights,
    ):
        coop_projections.append(np.asarray(_gaussian_broaden_projected(
            energies_per_k, coop_weights.reshape(n_pairs, n_k * nbf),
            k_weights, energy_grid, sigma,
        )))
        if compute_cohp:
            cohp_projections.append(-np.asarray(_gaussian_broaden_projected(
                energies_per_k, cohp_weights.reshape(n_pairs, n_k * nbf),
                k_weights, energy_grid, sigma,
            )))

    # Stack spin results
    if n_spin == 1:
        coop_proj: np.ndarray = coop_projections[0]
        cohp_proj: Optional[np.ndarray] = cohp_projections[0] if compute_cohp else None
    else:
        coop_proj = np.stack(coop_projections, axis=0)  # (2, n_pairs, n_e)
        cohp_proj = np.stack(cohp_projections, axis=0) if compute_cohp else None

    # Fermi energy
    if n_spin == 1:
        stacked_e = all_energies[0]  # (n_k, nbf)
        e_fermi: float = 0.0
        if n_electrons_per_cell is not None and n_electrons_per_cell % 2 == 0:
            n_occ = n_electrons_per_cell // 2
            if n_occ > 0:
                e_fermi = float(stacked_e[:, :n_occ].max())
        else:
            e_fermi = float(stacked_e[:, 0].max()) if nbf > 0 else 0.0
        # Integrate to E_F
        integrated_coop = _integrate_to_fermi(energy_grid, coop_proj, e_fermi)
        integrated_cohp = (
            _integrate_to_fermi(energy_grid, cohp_proj, e_fermi)
            if cohp_proj is not None
            else None
        )
    else:
        # Spin-polarized: per-spin Fermi levels from separate eigenvalue stacks.
        # For unrestricted, α and β have their own occupation counts.
        n_alpha = (
            (n_electrons_per_cell // 2 + 1)
            if (n_electrons_per_cell is not None and n_electrons_per_cell % 2 != 0)
            else (
                n_electrons_per_cell // 2 if n_electrons_per_cell is not None else nbf
            )
        )
        n_beta = (
            (n_electrons_per_cell // 2) if n_electrons_per_cell is not None else nbf
        )
        e_fermi_spin = []
        for s, spin_e in enumerate(all_energies):
            n_occ_spin = n_alpha if s == 0 else n_beta
            if n_occ_spin > 0 and n_occ_spin <= spin_e.shape[1]:
                e_fermi_spin.append(float(spin_e[:, :n_occ_spin].max()))
            else:
                e_fermi_spin.append(float(spin_e[:, 0].max()) if nbf > 0 else 0.0)
        # Use the higher Fermi level as the global reference for result.fermi_energy.
        e_fermi = max(e_fermi_spin)
        # Integrate each spin to its own Fermi level.
        integrated_coop = np.stack(
            [
                _integrate_to_fermi(energy_grid, coop_proj[s], e_fermi_spin[s])
                for s in range(2)
            ],
            axis=0,
        )
        integrated_cohp = (
            np.stack(
                [
                    _integrate_to_fermi(energy_grid, cohp_proj[s], e_fermi_spin[s])
                    for s in range(2)
                ],
                axis=0,
            )
            if cohp_proj is not None
            else None
        )

    # Pair metadata
    pairs = _pair_metadata(system, pair_keys)

    return COOPCOHPResult(
        energies=energy_grid,
        coop=coop_proj,
        cohp=cohp_proj,
        integrated_coop=integrated_coop,
        integrated_cohp=integrated_cohp,
        pairs=pairs,
        fermi_energy=e_fermi,
        sigma=sigma,
    )


def _integrate_to_fermi(
    energy_grid: np.ndarray,
    projection: np.ndarray,
    e_fermi: float,
) -> np.ndarray:
    """Trapezoidal integration of projection(E) up to e_fermi.

    ``projection`` shape: ``(n_pairs, n_e)``.  Returns shape ``(n_pairs,)``.
    """
    mask = energy_grid <= e_fermi
    n_pairs = projection.shape[0]
    result = np.zeros(n_pairs, dtype=float)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return result
    for p in range(n_pairs):
        result[p] = float(np.trapezoid(projection[p, idx], energy_grid[idx]))
    return result


# ---------------------------------------------------------------------------
# Periodic Mayer bond orders
# ---------------------------------------------------------------------------


def periodic_mayer_bond_orders(
    F_terms: Union[LatticeMatrixSet, Sequence[LatticeMatrixSet]],
    S_real: LatticeMatrixSet,
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh: BlochKMesh,
    *,
    n_electrons_per_cell: Optional[int] = None,
    F_terms_beta: Union[LatticeMatrixSet, Sequence[LatticeMatrixSet], None] = None,
) -> np.ndarray:
    """Compute periodic Mayer bond orders over a k-mesh.

    The Mayer bond order between atoms A and B is::

        B_AB = Σ_k w_k Σ_{μ∈A, ν∈B} [(P(k)·S(k))_{μν} · (P(k)·S(k))_{νμ}]

    where P(k) = C(k)·n·C(k)† is the density matrix at k with
    occupation numbers n. For unrestricted, the α and β contributions
    are summed with a factor of 2 per the standard molecular convention.

    Parameters
    ----------
    F_terms : LatticeMatrixSet or sequence
        Fock terms for the α spin (or restricted) channel.
    S_real : LatticeMatrixSet
        Real-space overlap.
    system, basis, kmesh : standard periodic inputs.
    n_electrons_per_cell : int or None
        Used to determine occupation. If None, all bands are occupied
        (non-interacting Hcore limit).
    F_terms_beta : optional
        Fock terms for the β spin channel. Triggers unrestricted
        computation.

    Returns
    -------
    np.ndarray, shape ``(n_atoms, n_atoms)``
        Symmetric bond-order matrix. Off-diagonal entries B_AB are the
        Interatomic Mayer bond orders; diagonal entries are zero.
    """
    nbf = S_real.nbf
    ao_to_atom = _shell_to_atom(basis)
    n_atoms = len(system.unit_cell)

    # Normalize F_terms
    if isinstance(F_terms, LatticeMatrixSet):
        fock_list: Sequence[LatticeMatrixSet] = [F_terms]
    else:
        fock_list = F_terms

    # Normalize beta Fock terms
    if F_terms_beta is not None:
        if isinstance(F_terms_beta, LatticeMatrixSet):
            fock_list_beta: Sequence[LatticeMatrixSet] = [F_terms_beta]
        else:
            fock_list_beta = list(F_terms_beta)
        spin_channels = [fock_list, fock_list_beta]
    else:
        spin_channels = [fock_list]

    # k-point arrays
    kpoints_cart = np.asarray([np.asarray(k) for k in kmesh.kpoints])
    k_weights = np.asarray(kmesh.weights, dtype=float)
    if abs(k_weights.sum() - 1.0) > 1e-8:
        k_weights = k_weights / k_weights.sum()
    n_k = kpoints_cart.shape[0]

    if n_electrons_per_cell is None:
        counts = [nbf] * len(spin_channels)
    elif len(spin_channels) == 1:
        if int(n_electrons_per_cell) % 2:
            raise ValueError("Restricted Mayer analysis needs an even electron count")
        counts = [int(n_electrons_per_cell) // 2]
    else:
        unpaired = int(system.multiplicity) - 1
        electrons = int(n_electrons_per_cell)
        if (electrons + unpaired) % 2:
            raise ValueError("Mayer electron count and multiplicity disagree")
        counts = [(electrons + unpaired) // 2, (electrons - unpaired) // 2]
    if any(count < 0 or count > nbf for count in counts):
        raise ValueError("Mayer occupation count exceeds the AO space")
    overlaps = [np.asarray(bloch_sum(S_real, k)) for k in kpoints_cart]
    density_channels = []
    capacity = 2.0 if len(spin_channels) == 1 else 1.0
    for fock_spin, count in zip(spin_channels, counts):
        densities = []
        for k, overlap in zip(kpoints_cart, overlaps):
            fock = sum(np.asarray(bloch_sum(term, k)) for term in fock_spin)
            bd = diagonalize_bloch(fock, overlap)
            occupied = np.asarray(bd.coefficients)[:, :count]
            densities.append(capacity * occupied @ occupied.conj().T)
        density_channels.append(densities)
    return _periodic_mayer_from_density(
        density_channels, overlaps, k_weights, ao_to_atom, n_atoms,
    )


def _periodic_mayer_from_density(
    density_channels, overlaps, weights, ao_to_atom, n_atoms,
) -> np.ndarray:
    """Mayer indices from accepted densities, including fractional occupations.

    Mayer, DOI 10.1002/jcc.20494, Eqs. (43)-(46): contract reverse indices
    of P*S; this product is generally not Hermitian in a nonorthogonal AO
    basis. Two singly occupied spin channels reduce to the restricted
    spin-summed expression. The k average sums the corresponding lattice
    products by the inverse-Bloch identity.
    """
    if len(density_channels) not in (1, 2):
        raise ValueError("Mayer analysis needs one or two density channels")
    weights = np.asarray(weights, dtype=float)
    atoms = np.asarray(ao_to_atom, dtype=int)
    nbf = len(atoms)
    if (weights.shape != (len(overlaps),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("Mayer analysis requires normalized nonnegative k weights")
    result = np.zeros((n_atoms, n_atoms))
    spin_factor = 1.0 if len(density_channels) == 1 else 2.0
    for densities in density_channels:
        if len(densities) != len(overlaps):
            raise ValueError("Mayer density and overlap meshes differ")
        for density, overlap, weight in zip(densities, overlaps, weights):
            density, overlap = np.asarray(density), np.asarray(overlap)
            for matrix in (density, overlap):
                if (matrix.shape != (nbf, nbf) or not np.isfinite(matrix).all()
                        or not np.allclose(matrix, matrix.conj().T, atol=1e-10, rtol=0)):
                    raise ValueError("Mayer analysis requires finite Hermitian AO matrices")
            ps = density @ overlap
            contribution = spin_factor * float(weight) * (ps * ps.T).real
            np.add.at(result, (atoms[:, None], atoms[None, :]), contribution)
    np.fill_diagonal(result, 0.)
    return 0.5 * (result + result.T)
