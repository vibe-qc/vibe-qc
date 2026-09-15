"""Derivable properties on the cyclic-cluster (Born-von-Kármán torus) -- Task C.

Which one-particle properties are *well defined* on a neutral finite BvK torus,
and which need care:

* **HOMO-LUMO / finite-cluster gap** -- well defined for the selected finite
  Hamiltonian. If that Hamiltonian is block-circulant, its real-supercell and
  character spectra are Fourier-related. Calling the result a bulk band gap or
  comparing it across Γ-CCM, χ-CCM, and neutral controls still requires an
  operator-specific convergence and construction audit.
* **Mulliken / Löwdin populations** -- well defined. They are basis-local
  partitions of the (cluster) density with the cyclic overlap ``S^CCM``; by
  translational symmetry every image of a unit-cell atom carries the same charge
  (the spread across images is a symmetry diagnostic), so the per-cell charge is
  unambiguous.
* **Electric dipole** -- *care needed*. A finite cluster has a well-defined
  electric dipole, but it is a **cluster (surface-terminated) quantity**, not the
  bulk polarization: the position operator ``r`` is not a legitimate operator on
  a torus (Resta 1998), and the plain home-cell ``<mu|r|ν>`` wraps for orbitals
  that cross the cell boundary (the same aliasing as the Wannier-centroid wrap in
  Task 1 Sec. 1.3). :func:`ccm_dipole` returns the finite-cluster dipole and is
  faithful only for non-wrapping (dilute / molecule-in-a-box) clusters; the bulk
  polarization needs the Berry-phase / Resta operator (an open item).
* **Forces / gradients** -- the WSSC-weighted-integral derivatives are not yet
  available analytically; numerical (finite-difference) gradients are tractable
  on the energy and are the recommended route until the analytic path lands
  (open item, see the handover).

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014); Resta,
Phys. Rev. Lett. 80, 1800 (1998) (the periodic position operator).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np

from vibeqc._vibeqc_core import direct_lattice_cells, make_lattice_matrix_set
from vibeqc.bands import BandStructure, KPath, band_structure
from vibeqc.basis_crystal import _ELEMENT_SYMBOLS
from vibeqc.spin_channels import spin_densities

__all__ = [
    "CCMBondAnalysis",
    "CCMBondOrder",
    "CCMGap",
    "CCMPopulation",
    "ccm_band_structure",
    "ccm_homo_lumo_gap",
    "ccm_mayer_bond_orders",
    "ccm_mulliken_charges",
    "ccm_lowdin_charges",
    "ccm_dipole",
    "ccm_numerical_gradient",
]


@dataclass
class CCMGap:
    """HOMO-LUMO / fundamental gap of the cyclic cluster (hartree)."""

    gap: float
    homo: float
    lumo: float
    homo_index: int          # 0-based index of the HOMO in the sorted spectrum
    spin: str = "restricted"


@dataclass
class CCMPopulation:
    """Atomic populations / charges (Mulliken or Löwdin) on the cyclic cluster."""

    charges: np.ndarray            # per supercell atom (n_atoms,)
    charges_per_cell: np.ndarray   # per unit-cell atom (n_basis_atoms,), image-averaged
    populations: np.ndarray        # per supercell atom electron population
    translational_spread: float    # max charge spread across images of a cell atom
    scheme: str                    # "mulliken" | "lowdin"


@dataclass(frozen=True)
class CCMBondOrder:
    """One primitive-cell Mayer bond order on the A-stream cyclic cluster."""

    atom_i: int
    atom_j: int
    symbol_i: str
    symbol_j: str
    translation: tuple[int, int, int]
    distance_bohr: float
    order: float


@dataclass(frozen=True)
class CCMBondAnalysis:
    """Mayer bond-order analysis folded from the cyclic cluster to one cell."""

    bonds: tuple[CCMBondOrder, ...]
    nrep: tuple[int, int, int]
    threshold: float
    translational_spread: float
    convention: str = (
        "Mayer bond order from the CCM density and cyclic overlap; "
        "primitive-cell bonds are averaged over all equivalent origins"
    )


def _total_density(scf_result) -> np.ndarray:
    """Total density (closed- or open-shell), real part for periodic results.

    Open-shell detection tests the VALUE, not ``hasattr``, via the shared
    :func:`vibeqc.spin_channels.spin_densities`. The supercell-Γ runner
    results (``CCMRealGammaResult``, ``CCMFourCentreResult``) declare
    ``density_alpha`` / ``density_beta`` as fields that are ``None`` on a
    closed-shell run, so a ``hasattr`` test takes the open-shell branch and
    evaluates ``None + None``. That trap cost #679 once and was still live
    here: it turned the whole population sidecar into a file of ``TypeError``
    rows on a real-Γ job.
    """
    alpha, beta = spin_densities(scf_result)
    if alpha is not None:
        P = np.asarray(alpha) + np.asarray(beta)
    else:
        P = np.asarray(scf_result.density)
    if np.iscomplexobj(P):
        P = np.real_if_close(P, tol=1000).real
    return np.asarray(P, dtype=float)


def _overlap(scf_result, ccm) -> np.ndarray:
    S = getattr(scf_result, "overlap", None)
    if S is None:
        from .integrals import ccm_overlap
        S = ccm_overlap(ccm)
    S = np.asarray(S)
    if np.iscomplexobj(S):
        S = np.real_if_close(S, tol=1000).real
    return np.asarray(S, dtype=float)


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


def _cell_residues(nrep: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return [
        tuple(int(value) for value in residue)
        for residue in product(*(range(int(n)) for n in nrep))
    ]


def _centered_translation(
    residue: tuple[int, int, int],
    nrep: tuple[int, int, int],
) -> tuple[int, int, int]:
    out: list[int] = []
    for value, period in zip(residue, nrep):
        integer = int(value)
        if 2 * integer > int(period):
            integer -= int(period)
        out.append(integer)
    return tuple(out)  # type: ignore[return-value]


def _positive_residue(
    translation: tuple[int, int, int],
    nrep: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(int(value) % int(period) for value, period in zip(translation, nrep))


def _translation_vector(ccm, translation: tuple[int, int, int]) -> np.ndarray:
    return np.asarray(translation, dtype=float) @ np.asarray(
        ccm.unit_vectors,
        dtype=float,
    )


def _element_symbol(atomic_number: int) -> str:
    if 0 <= int(atomic_number) < len(_ELEMENT_SYMBOLS):
        symbol = _ELEMENT_SYMBOLS[int(atomic_number)]
        if symbol:
            return symbol
    return f"Z{int(atomic_number)}"


def _canonical_bond_key(
    atom_i: int,
    atom_j: int,
    translation: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    forward = (int(atom_i), int(atom_j), *translation)
    reverse = (int(atom_j), int(atom_i), *(-int(value) for value in translation))
    return min(forward, reverse)


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


def _fold_mayer_bonds(
    pair_matrix: np.ndarray,
    ccm,
    *,
    threshold: float,
    max_bonds: int | None,
) -> CCMBondAnalysis:
    nrep = tuple(int(value) for value in ccm.nrep)
    residues = _cell_residues(nrep)
    n_unit_atoms = int(ccm.n_basis_atoms)
    atom_positions = np.asarray(ccm.unit_atom_pos, dtype=float)
    totals: dict[tuple[int, int, int, int, int], list[float]] = {}
    labels: dict[
        tuple[int, int, int, int, int],
        tuple[int, int, tuple[int, int, int]],
    ] = {}
    for origin_cell_index, origin in enumerate(residues):
        for target_cell_index, target in enumerate(residues):
            residue = tuple(
                (int(target[axis]) - int(origin[axis])) % int(nrep[axis])
                for axis in range(3)
            )
            translation = _centered_translation(residue, nrep)
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
    bonds: list[CCMBondOrder] = []
    max_spread = 0.0
    for key, values in totals.items():
        avg = float(np.mean(values))
        max_spread = max(max_spread, float(np.max(values) - np.min(values)))
        if avg < threshold:
            continue
        atom_i, atom_j, translation = labels[key]
        displacement = (
            atom_positions[atom_j]
            + _translation_vector(ccm, translation)
            - atom_positions[atom_i]
        )
        bonds.append(
            CCMBondOrder(
                atom_i=atom_i,
                atom_j=atom_j,
                symbol_i=_element_symbol(int(ccm.unit_system.unit_cell[atom_i].Z)),
                symbol_j=_element_symbol(int(ccm.unit_system.unit_cell[atom_j].Z)),
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
    return CCMBondAnalysis(
        bonds=tuple(bonds),
        nrep=nrep,
        threshold=float(threshold),
        translational_spread=max_spread,
    )


def _blocks_from_supercell_matrix(
    matrix: np.ndarray,
    ccm,
) -> dict[tuple[int, int, int], np.ndarray]:
    mat = _as_real_matrix(matrix, "CCM supercell matrix")
    nrep = tuple(int(value) for value in ccm.nrep)
    residues = _cell_residues(nrep)
    n_cells = len(residues)
    if mat.shape[0] != mat.shape[1] or mat.shape[0] % n_cells:
        raise ValueError("CCM matrix shape is not compatible with the cyclic cluster")
    unit_nbf = mat.shape[0] // n_cells
    accum = {
        residue: np.zeros((unit_nbf, unit_nbf), dtype=float)
        for residue in residues
    }
    counts = {residue: 0 for residue in residues}
    for origin_index, origin in enumerate(residues):
        row = slice(origin_index * unit_nbf, (origin_index + 1) * unit_nbf)
        for target_index, target in enumerate(residues):
            residue = tuple(
                (int(target[axis]) - int(origin[axis])) % int(nrep[axis])
                for axis in range(3)
            )
            col = slice(target_index * unit_nbf, (target_index + 1) * unit_nbf)
            accum[residue] += mat[row, col]
            counts[residue] += 1
    return {residue: accum[residue] / counts[residue] for residue in residues}


def _lattice_matrix_set_from_ccm_blocks(
    blocks_by_residue: dict[tuple[int, int, int], np.ndarray],
    ccm,
):
    nrep = tuple(int(value) for value in ccm.nrep)
    translations = [
        _centered_translation(residue, nrep)
        for residue in _cell_residues(nrep)
    ]
    blocks = [
        blocks_by_residue[_positive_residue(translation, nrep)]
        for translation in translations
    ]
    max_radius = max(
        float(np.linalg.norm(_translation_vector(ccm, t)))
        for t in translations
    )
    cell_pool = direct_lattice_cells(ccm.unit_system, max_radius + 1.0e-8)
    cell_by_index = {
        tuple(int(x) for x in np.asarray(cell.index, dtype=int)): cell
        for cell in cell_pool
    }
    try:
        cells = [cell_by_index[translation] for translation in translations]
    except KeyError as exc:
        raise RuntimeError(
            "could not build a LatticeMatrixSet for the CCM band path; "
            "the requested cyclic translation was not enumerated by direct_lattice_cells"
        ) from exc
    nbf = int(blocks[0].shape[0])
    return make_lattice_matrix_set(nbf, cells, blocks)


def _fold_to_cell(values: np.ndarray, ccm):
    """Average a per-supercell-atom quantity over translational images.

    Returns ``(per_cell, spread)`` where ``per_cell[b]`` is the mean over all
    supercell atoms with unit-cell-atom index ``b`` and ``spread`` is the largest
    max-min across any image set (≈ 0 by cyclic symmetry -- a diagnostic).
    """
    beta = np.asarray(ccm.atom_beta, dtype=int)
    nb = int(ccm.n_basis_atoms)
    per_cell = np.zeros(nb, dtype=float)
    spread = 0.0
    for b in range(nb):
        vals = values[beta == b]
        per_cell[b] = float(vals.mean()) if vals.size else 0.0
        if vals.size:
            spread = max(spread, float(vals.max() - vals.min()))
    return per_cell, spread


def ccm_homo_lumo_gap(scf_result, ccm) -> CCMGap:
    """HOMO-LUMO / fundamental gap of the cyclic cluster (hartree).

    Closed-shell (RHF/RKS): ``gap = e[n_occ] - e[n_occ-1]`` over the sorted
    supercell-Γ eigenvalues. Open-shell (UHF/UKS): the spin-resolved gap
    ``LUMO - HOMO`` with ``HOMO = max(e^a[n_a-1], e^b[n_b-1])`` and
    ``LUMO = min(e^a[n_a], e^b[n_b])`` over the two spin spectra. This is the
    finite-cluster gap of the selected Hamiltonian. For a block-circulant
    Hamiltonian its character representation has the same spectrum, but bulk
    and cross-construction interpretations need separate convergence evidence.
    """
    if hasattr(scf_result, "mo_energies_alpha"):
        ea = np.sort(np.asarray(scf_result.mo_energies_alpha, dtype=float).real)
        eb = np.sort(np.asarray(scf_result.mo_energies_beta, dtype=float).real)
        na, nb = int(scf_result.n_alpha), int(scf_result.n_beta)
        homo = max(float(ea[na - 1]) if na > 0 else -np.inf,
                   float(eb[nb - 1]) if nb > 0 else -np.inf)
        lumo = min(float(ea[na]) if na < ea.size else np.inf,
                   float(eb[nb]) if nb < eb.size else np.inf)
        return CCMGap(gap=lumo - homo, homo=homo, lumo=lumo,
                      homo_index=max(na, nb) - 1, spin="unrestricted")
    eps = np.asarray(scf_result.mo_energies, dtype=float)
    if np.iscomplexobj(eps):
        eps = eps.real
    eps = np.sort(eps)
    n_occ = int(ccm.supercell.n_electrons()) // 2
    if not (0 < n_occ < eps.size):
        raise ValueError(f"cannot locate HOMO/LUMO: n_occ={n_occ}, n_mo={eps.size}")
    homo, lumo = float(eps[n_occ - 1]), float(eps[n_occ])
    return CCMGap(gap=lumo - homo, homo=homo, lumo=lumo, homo_index=n_occ - 1)


def _population_charges(scf_result, ccm, diag: np.ndarray, scheme: str) -> CCMPopulation:
    ao_atom = np.asarray(ccm.ao_atom, dtype=int)
    n_atoms = int(ccm.n_atoms)
    pop = np.zeros(n_atoms, dtype=float)
    np.add.at(pop, ao_atom, diag)
    Z = np.array([int(at.Z) for at in ccm.supercell.atoms], dtype=float)
    charges = Z - pop
    per_cell, spread = _fold_to_cell(charges, ccm)
    return CCMPopulation(charges=charges, charges_per_cell=per_cell,
                         populations=pop, translational_spread=spread, scheme=scheme)


def ccm_mulliken_charges(scf_result, ccm) -> CCMPopulation:
    """Mulliken atomic charges on the cyclic cluster: ``q_A = Z_A - S_{muinA}(D S^CCM)_mumu``.

    Uses the cyclic overlap ``S^CCM`` and the CCM density. Charges sum to the
    cluster charge; the per-cell charges are the translational averages (the
    ``translational_spread`` reports the residual image-to-image variation, ≈ 0
    by cyclic symmetry).
    """
    D = _total_density(scf_result)
    S = _overlap(scf_result, ccm)
    diag = np.einsum("ij,ji->i", D, S)            # diag(D . S^CCM)
    return _population_charges(scf_result, ccm, diag, "mulliken")


def ccm_lowdin_charges(scf_result, ccm) -> CCMPopulation:
    """Löwdin atomic charges: ``q_A = Z_A - S_{muinA}(S^{1/2} D S^{1/2})_mumu`` (S = S^CCM).

    Less basis-sensitive than Mulliken (symmetric orthogonalization partitions
    the overlap equally between the two AOs).
    """
    D = _total_density(scf_result)
    S = _overlap(scf_result, ccm)
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    if float(np.min(w)) < 1e-10:
        raise ValueError(
            f"ccm_lowdin_charges: S^CCM near-singular (min eig {float(np.min(w)):.2e}); "
            "enlarge the cluster or screen diffuse functions.")
    S_half = (V * np.sqrt(w)) @ V.T
    diag = np.einsum("ij,jk,ki->i", S_half, D, S_half)
    return _population_charges(scf_result, ccm, diag, "lowdin")


def ccm_mayer_bond_orders(
    scf_result,
    ccm,
    *,
    threshold: float = 0.05,
    max_bonds: int | None = 40,
) -> CCMBondAnalysis:
    """Return primitive-cell Mayer bond orders from a CCM SCF result.

    The finite cyclic-cluster density and overlap are first analysed in the
    ordinary AO Mayer metric.  Atom-pair values are then averaged over all
    equivalent cell origins and folded to the primitive cell.  The reported
    ``translational_spread`` is a diagnostic for residual symmetry breaking in
    the converged cluster density.
    """

    S = _overlap(scf_result, ccm)
    alpha, beta = spin_densities(scf_result)
    if alpha is not None:
        pair_matrix = _mayer_atom_pair_matrix(
            _as_real_matrix(np.asarray(alpha), "CCM alpha density"),
            S,
            np.asarray(ccm.ao_atom, dtype=int),
            density_beta=_as_real_matrix(
                np.asarray(beta),
                "CCM beta density",
            ),
        )
    else:
        pair_matrix = _mayer_atom_pair_matrix(
            _total_density(scf_result),
            S,
            np.asarray(ccm.ao_atom, dtype=int),
        )
    return _fold_mayer_bonds(
        pair_matrix,
        ccm,
        threshold=threshold,
        max_bonds=max_bonds,
    )


def ccm_band_structure(
    scf_result,
    ccm,
    kpath: KPath,
    *,
    spin: str = "total",
    n_electrons_per_cell: int | None = None,
) -> BandStructure:
    """Sample the converged CCM Fock matrix along a primitive-cell k-path.

    The A-stream SCF is stored as a supercell-Gamma cyclic cluster.  This helper
    block-averages the converged supercell Fock and overlap into finite-torus
    real-space blocks and passes them to :func:`vibeqc.bands.band_structure`.
    The interpolation is exact on the cluster's folded character net and should
    be converged against ``ccm.nrep`` for quantitative band plots.
    """

    if spin not in {"total", "alpha", "beta"}:
        raise ValueError("spin must be total, alpha, or beta")
    if spin == "total":
        fock = getattr(scf_result, "fock")
    else:
        attr = f"fock_{spin}"
        if not hasattr(scf_result, attr):
            raise ValueError("spin-resolved bands require an unrestricted CCM result")
        fock = getattr(scf_result, attr)
    fock_blocks = _blocks_from_supercell_matrix(np.asarray(fock), ccm)
    overlap_blocks = _blocks_from_supercell_matrix(_overlap(scf_result, ccm), ccm)
    n_elec = (
        int(ccm.unit_system.n_electrons())
        if n_electrons_per_cell is None
        else int(n_electrons_per_cell)
    )
    return band_structure(
        _lattice_matrix_set_from_ccm_blocks(fock_blocks, ccm),
        _lattice_matrix_set_from_ccm_blocks(overlap_blocks, ccm),
        kpath,
        n_electrons_per_cell=n_elec,
    )


def ccm_dipole(scf_result, ccm, *, origin=None) -> np.ndarray:
    """Finite-cluster electric dipole (a.u.), ``mu = S_A Z_A R_A - Tr(D <r>)``.

    .. warning::

       This is the **finite (surface-terminated) cluster dipole**, *not* the
       bulk polarization. The position operator is not well defined on a torus
       (Resta 1998) and the plain home-cell ``<mu|r|ν>`` wraps for orbitals that
       cross the cell boundary, so this value is faithful only for non-wrapping
       (dilute / molecule-in-a-box) clusters. For a dense crystal use it only as
       a symmetry indicator (it is ≈ 0 for a centrosymmetric/neutral cluster);
       the bulk polarization needs the Berry-phase / Resta operator (open item).

    Origin defaults to the cluster centroid; for a neutral cluster the result is
    origin-independent.
    """
    from vibeqc import compute_dipole

    pos = np.asarray(ccm.atom_positions, dtype=float)
    Z = np.array([int(at.Z) for at in ccm.supercell.atoms], dtype=float)
    if origin is None:
        origin_vec = pos.mean(axis=0)
    else:
        origin_vec = np.asarray(origin, dtype=float)
        if origin_vec.shape != (3,):
            raise ValueError("ccm_dipole: origin must be a 3-vector (bohr)")

    D = _total_density(scf_result)
    dip = compute_dipole(ccm.basis, [float(x) for x in origin_vec])
    mu_e = -np.array([
        np.einsum("ij,ji->", D, np.asarray(dip.x)),
        np.einsum("ij,ji->", D, np.asarray(dip.y)),
        np.einsum("ij,ji->", D, np.asarray(dip.z)),
    ], dtype=float)
    mu_n = (Z[:, None] * (pos - origin_vec[None, :])).sum(axis=0)
    return mu_n + mu_e


def _displaced_unit_system(unit_system, atom_index: int, delta: np.ndarray):
    """Copy of ``unit_system`` with one unit-cell atom shifted by ``delta`` (bohr)."""
    from vibeqc import PeriodicSystem
    from vibeqc._vibeqc_core import Atom

    atoms = []
    for k, at in enumerate(unit_system.unit_cell):
        xyz = np.asarray(at.xyz, dtype=float).copy()
        if k == atom_index:
            xyz = xyz + np.asarray(delta, dtype=float)
        atoms.append(Atom(int(at.Z), xyz.tolist()))
    return PeriodicSystem(
        int(unit_system.dim), np.asarray(unit_system.lattice, dtype=float),
        atoms, charge=int(unit_system.charge),
        multiplicity=int(unit_system.multiplicity))


def ccm_numerical_gradient(ccm, *, runner=None, method: str = "aiccm2026dev-a",
                           h: float = 1.0e-3) -> np.ndarray:
    """Finite-difference nuclear gradient ``dE_total/dR`` per unit-cell atom (a.u.).

    Central differences: each unit-cell atom is displaced by ``±h`` along each
    Cartesian axis; because the supercell is built by replication, the
    displacement propagates to **every periodic image**, so this is the gradient
    under the cyclic (periodic) constraint. The force is ``-gradient``.

    Well defined and tractable (no analytic WSSC-integral derivatives needed); it
    costs ``6 . n_cell_atoms`` SCF evaluations. ``runner(ccm) -> result`` must
    expose ``.energy`` (the total cluster energy); defaults to the scalable
    closed-shell HF-CCM with ``method``.

    Returns ``(n_cell_atoms, 3)``. By translational invariance the per-cell forces
    sum to ≈ 0 (a built-in sanity check).
    """
    from .system import CCMSystem

    if runner is None:
        from .scf import run_ccm_rhf_scalable

        def runner(c):
            return run_ccm_rhf_scalable(c, method=method)

    nrep = ccm.nrep
    basis = ccm.basis_name
    n_cell = int(ccm.n_basis_atoms)
    grad = np.zeros((n_cell, 3), dtype=float)
    for a in range(n_cell):
        for d in range(3):
            e = {}
            for sign in (+1.0, -1.0):
                delta = np.zeros(3)
                delta[d] = sign * h
                unit_disp = _displaced_unit_system(ccm.unit_system, a, delta)
                ccm_disp = CCMSystem(unit_disp, nrep, basis,
                                     weight_tol_bohr=ccm.weight_tol_bohr)
                e[sign] = float(runner(ccm_disp).energy)
            grad[a, d] = (e[+1.0] - e[-1.0]) / (2.0 * h)
    return grad
