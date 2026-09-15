"""Provisional Natural Bond Orbital (NBO) analysis.

Provides exploratory natural-orbital classification and second-order
donor-acceptor analysis. Natural Population Analysis (NPA) entry points fail
closed until the full Natural Atomic Orbital (NAO) construction is available;
a Löwdin population is not relabeled as NPA.

Public API
----------

.. autofunction:: npa_charges
.. autofunction:: nao_density
.. autofunction:: nbo_search
.. autofunction:: donor_acceptor_analysis
.. autoclass:: NBOResult

Theory references
-----------------

- Reed, A. E., Weinstock, R. B., Weinhold, F., J. Chem. Phys. 83, 735 (1985).
  DOI: 10.1063/1.449486 (Natural Population Analysis)
- Reed, A. E., Curtiss, L. A., Weinhold, F., Chem. Rev. 88, 899 (1988).
  DOI: 10.1021/cr00088a005 (NBO donor-acceptor analysis)
- Foster, J. P., Weinhold, F., J. Am. Chem. Soc. 102, 7211 (1980).
  DOI: 10.1021/ja00544a007 (Natural hybrid orbitals)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .spin_channels import spin_densities

__all__ = [
    "NBOResult",
    "npa_charges",
    "nao_density",
    "nbo_search",
    "donor_acceptor_analysis",
    "nbo_charges",
]


_NPA_NOT_IMPLEMENTED = (
    "full Natural Population Analysis requires the occupancy-weighted "
    "Natural Atomic Orbital construction; the former implementation was "
    "Löwdin population analysis and must not be reported as NPA"
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class NBOResult:
    """Natural Bond Orbital analysis result.

    Attributes
    ----------
    npa_charges : np.ndarray, shape ``(n_atoms,)``
        Natural Population Analysis charges (Z - NPA population). Empty until
        the full NAO construction is implemented.
    nao_populations : np.ndarray, shape ``(n_ao,)``
        NAO populations per natural atomic orbital.
    nbo_orbitals : list of dict
        Each dict describes a Natural Bond Orbital:
        ``{"type": "BD"|"LP"|"BD*"|"RY*", "atoms": [i,j], "occupancy": float}``
    donor_acceptor : list of dict
        Second-order donor-acceptor interactions:
        ``{"donor": str, "acceptor": str, "E2": float}`` (E2 in kcal/mol)
    """

    npa_charges: np.ndarray = field(default_factory=lambda: np.array([]))
    nao_populations: np.ndarray = field(default_factory=lambda: np.array([]))
    nbo_orbitals: list[dict] = field(default_factory=list)
    donor_acceptor: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["Natural Bond Orbital (NBO) Analysis", "-" * 48]
        if len(self.npa_charges) > 0:
            lines.append("")
            lines.append("NPA Atomic Charges:")
            lines.append(f"  {'Atom':<6s} {'Charge':>10s}")
            lines.append(f"  {'-' * 4:>6s} {'-' * 8:>10s}")
            for i, q in enumerate(self.npa_charges):
                lines.append(f"  {i:<6d} {q:>10.5f}")
        if self.nbo_orbitals:
            lines.append("")
            lines.append("Natural Bond Orbitals (occupied, occupancy > 0.5):")
            for nbo in self.nbo_orbitals:
                if nbo.get("occupancy", 0.0) > 0.5:
                    occ = nbo["occupancy"]
                    typ = nbo["type"]
                    atoms = nbo.get("atoms", [])
                    atom_str = "-".join(str(a) for a in atoms) if atoms else "?"
                    lines.append(f"  {typ:>6s} {atom_str:<8s} {occ:.4f}")
        if self.donor_acceptor:
            lines.append("")
            lines.append("Donor-Acceptor Interactions (E2, kcal/mol):")
            lines.append(f"  {'Donor':<20s} → {'Acceptor':<20s} {'E2':>10s}")
            for da in self.donor_acceptor[:10]:
                lines.append(
                    f"  {da['donor']:<20s} → {da['acceptor']:<20s} {da['E2']:>10.2f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Natural Population Analysis (NPA)
# ---------------------------------------------------------------------------


def nao_density(
    density_matrix: np.ndarray,
    overlap: np.ndarray,
    basis,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform the AO density matrix to the NAO (Natural Atomic Orbital)
    basis.

    This is a simplified implementation that uses the Löwdin
    orthogonalisation as a proxy for the full NAO construction. A
    full NAO analysis would compute atomic angular-momentum-averaged
    density matrices and diagonalise them per atom.

    Parameters
    ----------
    density_matrix : np.ndarray, shape ``(n_ao, n_ao)``
        One-particle density matrix in the AO basis.
    overlap : np.ndarray, shape ``(n_ao, n_ao)``
        AO overlap matrix.
    basis : BasisSet
        Basis set for AO-to-atom mapping.

    Returns
    -------
    nao_populations : np.ndarray, shape ``(n_ao,)``
        Population per NAO.
    nao_coeffs : np.ndarray, shape ``(n_ao, n_ao)``
        NAO coefficient matrix (orthogonal basis).
    """
    P = np.asarray(density_matrix)
    S = np.asarray(overlap)
    n_ao = P.shape[0]

    # PS in orthogonal basis
    evals_S, evecs_S = np.linalg.eigh(S)
    mask = evals_S > 1e-14
    s_half = np.zeros((n_ao, n_ao), dtype=np.float64)
    for i in range(n_ao):
        if mask[i]:
            s_half[i, i] = np.sqrt(evals_S[i])
    X = evecs_S @ s_half  # S^{1/2}

    P_orth = X.T @ P @ X
    # Diagonalise the orthogonal-basis density matrix to get NAOs
    occ, U = np.linalg.eigh(P_orth)
    # NAO populations (diagonal of density in NAO basis)
    nao_pop = np.diag(U.T @ P_orth @ U)

    return np.asarray(nao_pop, dtype=np.float64), U


def npa_charges(
    density_matrix: np.ndarray,
    overlap: np.ndarray,
    basis,
    molecule,
) -> np.ndarray:
    r"""Natural Population Analysis (NPA) charges.

    NPA requires occupancy-weighted Natural Atomic Orbitals. The former
    implementation computed ``diag(S^{1/2} P S^{1/2})``, which is Löwdin
    population analysis, so this entry point now fails closed instead of
    returning correctly computed values under an incorrect label.

    For each atom A::

        q_A = Z_A - sum_{mu in A} n_mu

    where n_mu are the NAO diagonal populations.

    Parameters
    ----------
    density_matrix : np.ndarray
        One-particle density matrix in the AO basis.
    overlap : np.ndarray
        AO overlap matrix.
    basis : BasisSet
    molecule : Molecule

    Raises
    ------
    NotImplementedError
        Until the full NAO construction is implemented.
    """
    raise NotImplementedError(_NPA_NOT_IMPLEMENTED)


def nbo_charges(
    result,
    basis,
    molecule,
) -> np.ndarray:
    """Convenience wrapper: NPA charges from a converged SCF result."""
    S = np.asarray(compute_overlap_fallback(basis))
    alpha, beta = spin_densities(result)
    if alpha is not None:
        P = np.asarray(alpha.real + beta.real)
    else:
        P = np.asarray(result.density.real)
    return npa_charges(P, S, basis, molecule)


def compute_overlap_fallback(basis):
    """Get overlap matrix, falling back to local import."""
    try:
        from ._vibeqc_core import compute_overlap

        return compute_overlap(basis)
    except ImportError:
        # For periodic or non-standard basis
        n_ao = sum(2 * s.l + 1 for s in basis.shells())
        return np.eye(n_ao)


# ---------------------------------------------------------------------------
# NBO search: identify Lewis-structure NBOs
# ---------------------------------------------------------------------------


def nbo_search(
    density_matrix: np.ndarray,
    overlap: np.ndarray,
    basis,
    molecule,
    *,
    occupancy_threshold: float = 0.5,
) -> NBOResult:
    """Search for Natural Bond Orbitals from the density matrix.

    This is a simplified implementation that classifies natural orbitals
    into bond (BD), lone-pair (LP), and antibond/rydberg (BD*/RY*)
    categories based on their atom-pair localisation and occupancy.

    The procedure:
    1. Compute NAO populations.
    2. For each pair of atoms with significant off-diagonal density,
       identify the bonding NBO (high occupancy, in-phase combination).
    3. Remaining high-occupancy orbitals on single atoms are lone pairs.

    Parameters
    ----------
    density_matrix : np.ndarray
        One-particle density matrix in the AO basis.
    overlap : np.ndarray
        AO overlap matrix.
    basis : BasisSet
    molecule : Molecule
    occupancy_threshold : float
        Minimum occupancy for a bond NBO.

    Returns
    -------
    NBOResult
    """
    P = np.asarray(density_matrix)
    S = np.asarray(overlap)
    n_ao = P.shape[0]

    # AO-to-atom mapping
    ao_to_atom: list[int] = []
    for shell in basis.shells():
        n = 2 * shell.l + 1
        ao_to_atom.extend([int(shell.atom_index)] * n)

    atoms = list(molecule.atoms)
    n_atoms = len(atoms)

    # The provisional orbital classifier does not require NPA charges. Leave
    # the field empty until occupancy-weighted NAOs are implemented.
    charges = np.array([], dtype=np.float64)

    # Transform density to orthogonal basis
    evals_S, evecs_S = np.linalg.eigh(S)
    mask = evals_S > 1e-14
    s_half = np.zeros((n_ao, n_ao), dtype=np.float64)
    for i in range(n_ao):
        if mask[i]:
            s_half[i, i] = np.sqrt(evals_S[i])
    X = evecs_S @ s_half

    P_orth = X.T @ P @ X
    occ, U = np.linalg.eigh(P_orth)

    # Classify natural orbitals
    # The natural orbitals are the eigenvectors of P·S
    # In the orthogonal basis, we look at the AO character of each
    # natural orbital to assign it to atom pairs
    nbo_list: list[dict] = []
    nao_pop = np.diag(U.T @ P_orth @ U)

    for p in range(n_ao):
        n_p = occ[p]
        c_p = U[:, p]  # coefficients in orthogonal basis
        # Transform back to AO
        c_ao = X @ c_p

        # Find dominant atoms
        atom_pop = np.zeros(n_atoms, dtype=np.float64)
        for mu in range(n_ao):
            a = ao_to_atom[mu]
            atom_pop[a] += c_ao[mu] ** 2

        dominant = np.argsort(-atom_pop)
        if n_p > 1.9:  # Strongly occupied
            if atom_pop[dominant[0]] > 0.7:
                typ = "LP" if np.count_nonzero(atom_pop > 0.05) == 1 else "CR"
            else:
                typ = "BD"  # Bond
            atoms_involved = [int(dominant[0])]
            if atom_pop[dominant[1]] > 0.05:
                atoms_involved.append(int(dominant[1]))
        elif n_p > occupancy_threshold:
            if atom_pop[dominant[0]] > 0.6:
                typ = "LP"  # Lone pair
            else:
                typ = "BD"  # Bond
            atoms_involved = [int(dominant[0])]
            if atom_pop[dominant[1]] > 0.05:
                atoms_involved.append(int(dominant[1]))
        elif n_p > 0.02:
            typ = "BD*"  # Antibond / low virtual
            atoms_involved = [int(dominant[0])]
            if atom_pop[dominant[1]] > 0.05:
                atoms_involved.append(int(dominant[1]))
        else:
            typ = "RY*"  # Rydberg / high virtual
            atoms_involved = [int(dominant[0])]

        nbo_list.append(
            {
                "type": typ,
                "atoms": atoms_involved,
                "occupancy": float(n_p),
                "index": p,
            }
        )

    return NBOResult(
        npa_charges=charges,
        nao_populations=nao_pop,
        nbo_orbitals=nbo_list,
        donor_acceptor=[],
    )


# ---------------------------------------------------------------------------
# Second-order donor-acceptor analysis
# ---------------------------------------------------------------------------


def donor_acceptor_analysis(
    nbo_result: NBOResult,
    fock_orth: np.ndarray,
    *,
    e2_threshold: float = 0.5,
) -> NBOResult:
    r"""Second-order donor-acceptor perturbation analysis.

    For each pair of NBOs (donor i, acceptor j), the stabilisation
    energy is::

        E2_{i→j} = q_i * F_{ij}^2 / (ε_j - ε_i)

    where q_i is the donor occupancy, F_{ij} is the Fock matrix element
    between NBOs i and j, and ε_j - ε_i is the orbital energy difference.

    Parameters
    ----------
    nbo_result : NBOResult
        Result from :func:`nbo_search` containing NBO definitions.
    fock_orth : np.ndarray, shape ``(n_ao, n_ao)``
        Fock matrix in the orthogonal (NAO) basis.
    e2_threshold : float
        Minimum E2 (kcal/mol) to include.

    Returns
    -------
    NBOResult
        Updated with donor_acceptor list.
    """
    F = np.asarray(fock_orth)
    n_ao = F.shape[0]

    # Use NBO orbitals from nbo_result to classify donors/acceptors
    nbos = nbo_result.nbo_orbitals
    if len(nbos) < 2:
        return nbo_result

    da_list: list[dict] = []
    nbo_occ = np.array([nbo.get("occupancy", 0.0) for nbo in nbos])

    for i, nbo_i in enumerate(nbos):
        occ_i = nbo_i.get("occupancy", 0.0)
        typ_i = nbo_i["type"]
        if typ_i in ("BD*", "RY*") or occ_i < 0.5:
            continue  # Donor must be occupied

        for j, nbo_j in enumerate(nbos):
            if i == j:
                continue
            occ_j = nbo_j.get("occupancy", 0.0)
            typ_j = nbo_j["type"]
            if typ_j not in ("BD*", "RY*") and occ_j > 0.5:
                continue  # Acceptor must be virtual/antibond

            # Fock matrix element between NBOs
            F_ij = F[i, j]
            # Energy difference: use diagonal Fock elements as proxy
            eps_i = F[i, i]
            eps_j = F[j, j]
            de = eps_j - eps_i
            if abs(de) < 1e-10:
                continue

            # E2 in Hartree, convert to kcal/mol
            e2_hartree = occ_i * F_ij * F_ij / abs(de)
            e2_kcal = e2_hartree * 627.509  # Hartree → kcal/mol

            if e2_kcal >= e2_threshold:
                da_list.append(
                    {
                        "donor": f"{typ_i}({i})",
                        "acceptor": f"{typ_j}({j})",
                        "E2": float(e2_kcal),
                    }
                )

    da_list.sort(key=lambda x: -x["E2"])
    return NBOResult(
        npa_charges=nbo_result.npa_charges,
        nao_populations=nbo_result.nao_populations,
        nbo_orbitals=nbo_result.nbo_orbitals,
        donor_acceptor=da_list,
    )
