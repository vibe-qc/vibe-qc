"""DLPNO PAO construction and domain building.

Projected Atomic Orbitals (PAOs) form the intermediate basis between
the AO and PNO representations.  For a given pair domain -- a subset
of atoms -- the PAOs are obtained by projecting the occupied orbitals
out of the AO basis functions on those atoms.

The PAO coefficient matrix satisfies:
    C_PAO = (I - C_occ @ C_occ^T @ S) @ A_domain

where A_domain selects the AO basis functions on the domain atoms.

References
----------
* Pulay, Chem. Phys. Lett. 100, 151 (1983) -- PAO concept.
* Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013), Sec.II.D.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PAODomain:
    """Projected atomic orbital basis for one occupied pair domain.

    Attributes
    ----------
    pair_i, pair_j : int
        Occupied orbital indices for this pair.
    atom_indices : ndarray
        Indices of atoms in the domain.
    ao_indices : ndarray
        Global AO indices belonging to the domain (concatenated atom shells).
    n_pao : int
        Number of PAOs in the domain (= number of domain AOs minus
        number of occupied orbitals projected out).
    C_pao : ndarray
        PAO coefficient matrix, shape (domain_nbf, n_pao).
        C_pao[:, k] gives the k-th PAO expanded in domain AOs.
    ao_to_pao : ndarray or None
        Optional transformation from raw domain AOs to orthogonal PAOs,
        shape (domain_nbf, n_pao).  Same as C_pao when no further
        orthogonalisation is applied.
    """

    pair_i: int
    pair_j: int
    atom_indices: np.ndarray
    ao_indices: np.ndarray
    n_pao: int
    C_pao: np.ndarray
    ao_to_pao: np.ndarray | None = None


# ---------------------------------------------------------------------------
# PAO construction
# ---------------------------------------------------------------------------


def build_projection_matrix(
    C_occ: np.ndarray,
    S_full: np.ndarray,
) -> np.ndarray:
    """Build the occupied-space projector in the AO basis.

        P_occ = C_occ @ C_occ^T @ S
        Q_vir = I - P_occ

    Parameters
    ----------
    C_occ : ndarray, shape (nbf, nocc)
        Occupied MO coefficients.
    S_full : ndarray, shape (nbf, nbf)
        AO overlap matrix.

    Returns
    -------
    Q_vir : ndarray, shape (nbf, nbf)
        Virtual-space projector: Q = I - C_occ @ C_occ^T @ S.
    """
    nbf, nocc = C_occ.shape
    # P = C_occ @ C_occ^T @ S
    P = C_occ @ (C_occ.T @ S_full)  # (nbf, nbf)
    return np.eye(nbf) - P


def build_pao_coeffs(
    Q_vir: np.ndarray,
    domain_ao_mask: np.ndarray,
    S_full: np.ndarray,
    S_domain: np.ndarray | None = None,
    lindep_thresh: float = 1e-8,
) -> tuple[np.ndarray, int]:
    """Build PAO coefficients for a domain.

    Steps:
    1. Extract the domain block of the virtual projector.
    2. Diagonalize S_domain to remove linear dependencies.
    3. The PAOs span the range of Q_domain.

    Parameters
    ----------
    Q_vir : ndarray, shape (nbf, nbf)
        Full virtual-space projector.
    domain_ao_mask : ndarray, shape (nbf,), bool
        Mask selecting AOs in the domain.
    S_full : ndarray, shape (nbf, nbf)
        Full AO overlap matrix.
    S_domain : ndarray, optional
        Precomputed domain overlap block.  Computed from S_full if None.
    lindep_thresh : float
        Eigenvalue threshold for removing linear dependencies.

    Returns
    -------
    C_pao : ndarray, shape (n_domain_ao, n_pao)
        PAO coefficients in the domain AO basis.
    n_pao : int
        Number of PAOs retained.
    """
    domain_idx = np.where(domain_ao_mask)[0]
    n_domain = len(domain_idx)

    if n_domain == 0:
        return np.zeros((0, 0)), 0

    # 1. Build the domain projector block: Q_domain = Q_vir[domain, domain]
    Q_dom = Q_vir[np.ix_(domain_idx, domain_idx)]

    # 2. Build domain overlap
    if S_domain is None:
        S_dom = S_full[np.ix_(domain_idx, domain_idx)]
    else:
        S_dom = S_domain

    # 3. Orthogonalise Q_domain . S_dom . Q_domain^T
    #    The PAOs are obtained from the eigenvectors of
    #    Q S Q^T with eigenvalues above the threshold.
    QS = Q_dom @ S_dom
    M = QS @ Q_dom.T  # (n_domain, n_domain), symmetric

    # Symmetrize to remove numerical noise
    M = 0.5 * (M + M.T)

    evals, evecs = np.linalg.eigh(M)

    # Retain eigenvectors with significant eigenvalues
    keep = evals > lindep_thresh
    n_pao = int(np.sum(keep))

    # Sort descending
    order = np.argsort(-evals)
    evals = evals[order]
    evecs = evecs[:, order]
    keep = evals > lindep_thresh
    n_pao = int(np.sum(keep))

    C_pao = evecs[:, keep].copy()  # (n_domain, n_pao)
    return C_pao, n_pao


# ---------------------------------------------------------------------------
# Domain construction -- atom selection via Mulliken populations
# ---------------------------------------------------------------------------


def build_atom_basis_map(
    molecule,
    basis,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the atom->AO index mapping.

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet

    Returns
    -------
    atom_first_ao : ndarray, shape (natom+1,), int
        Start index for each atom's AOs (cumulative).
    ao_center : ndarray, shape (nbf, 3)
        Coordinates of each basis-function center (Bohr).
    """
    atoms = list(molecule.atoms)
    natom = len(atoms)
    nbf = basis.nbasis

    # Build per-AO center coordinates from shell origins
    ao_centers = np.zeros((nbf, 3))
    bf = 0
    for sh in basis.shells():
        atom_idx = int(sh.atom_index)
        L = int(sh.l)
        n_func = 2 * L + 1  # spherical harmonics
        origin = np.array(sh.origin)
        for k in range(n_func):
            if bf + k < nbf:
                ao_centers[bf + k] = origin
        bf += n_func

    # Atom first-AO index from shells
    atom_first = np.zeros(natom + 1, dtype=int)
    # Simple O(natom * nshells) pass -- small enough for any molecule
    bf_acc = 0
    for a in range(natom):
        atom_first[a] = bf_acc
        for sh in basis.shells():
            if int(sh.atom_index) == a:
                bf_acc += 2 * int(sh.l) + 1
    atom_first[natom] = nbf

    return atom_first, ao_centers


def select_domain_atoms_mulliken(
    C_occ: np.ndarray,
    S: np.ndarray,
    atom_first_ao: np.ndarray,
    i: int,
    j: int,
    tcut_mkn: float = 1e-3,
) -> np.ndarray:
    """Select domain atoms for pair (i, j) using Mulliken populations.

    An atom A is included in the domain if the Mulliken population of
    orbital i OR orbital j on atom A exceeds ``tcut_mkn``.

    Parameters
    ----------
    C_occ : ndarray, shape (nbf, nocc)
    S : ndarray, shape (nbf, nbf)
    atom_first_ao : ndarray, shape (natom+1,), int
    i, j : int
        Orbital indices.
    tcut_mkn : float
        Mulliken population threshold.

    Returns
    -------
    domain_atoms : ndarray, shape (n_domain,), int
        Indices of selected atoms.
    """
    natom = len(atom_first_ao) - 1
    nbf = C_occ.shape[0]
    CS = C_occ.T @ S  # (nocc, nbf)

    domain_list = []
    for a in range(natom):
        a_start = atom_first_ao[a]
        a_end = atom_first_ao[a + 1]
        if a_start >= nbf:
            continue

        # Mulliken population of orbital i on atom A
        pop_i = float(np.sum(CS[i, a_start:a_end] * C_occ[a_start:a_end, i]))
        pop_j = float(np.sum(CS[j, a_start:a_end] * C_occ[a_start:a_end, j]))
        if abs(pop_i) > tcut_mkn or abs(pop_j) > tcut_mkn:
            domain_list.append(a)

    return np.array(domain_list, dtype=int)


def select_domain_atoms_united(
    C_occ: np.ndarray,
    S: np.ndarray,
    atom_first_ao: np.ndarray,
    pair_list: list,
    tcut_mkn: float = 1e-3,
) -> np.ndarray:
    """Build the domain for the combined ij-pair and all singles.

    For DLPNO singles, the domain is the union of domains of all
    occupied orbitals. This gives the "united" domain for singles
    treatment.

    Parameters
    ----------
    C_occ, S, atom_first_ao : see select_domain_atoms_mulliken.
    pair_list : list of (int, int)
        All pairs contributing to the united domain.
    tcut_mkn : float

    Returns
    -------
    united_atoms : ndarray
        Union of all pair domain atoms.
    """
    natom = len(atom_first_ao) - 1
    united = np.zeros(natom, dtype=bool)
    for i, j in pair_list:
        atoms = select_domain_atoms_mulliken(C_occ, S, atom_first_ao, i, j, tcut_mkn)
        united[atoms] = True
    return np.where(united)[0]


# ---------------------------------------------------------------------------
# High-level: build all pair domains
# ---------------------------------------------------------------------------


@dataclass
class DomainResult:
    """Result of domain construction for all pairs.

    Attributes
    ----------
    domains : dict of (int,int) -> PAODomain
        Map from orbital pair to its PAO domain.
    united_domain : PAODomain
        United domain for singles treatment.
    Q_vir : ndarray
        Full virtual-space projector (cached for reuse).
    """

    domains: dict = field(default_factory=dict)
    united_domain: PAODomain | None = None
    Q_vir: np.ndarray | None = None


def build_all_domains(
    molecule,
    basis,
    C_occ: np.ndarray,
    S: np.ndarray,
    pair_classification,
    tcut_mkn: float = 1e-3,
    lindep_thresh: float = 1e-8,
) -> DomainResult:
    """Build PAO domains for all strong and weak pairs.

    Parameters
    ----------
    molecule : Molecule
    basis : BasisSet
    C_occ : ndarray, shape (nbf, nocc)
        Localised occupied MO coefficients.
    S : ndarray, shape (nbf, nbf)
        AO overlap matrix.
    pair_classification : PairClassification
        From ``classify_pairs``.
    tcut_mkn : float
        Mulliken population threshold.
    lindep_thresh : float
        Linear-dependency threshold for PAO construction.

    Returns
    -------
    DomainResult
    """
    nbf = C_occ.shape[0]
    atom_first, ao_centers = build_atom_basis_map(molecule, basis)

    # Build the full virtual projector (cached)
    Q_vir = build_projection_matrix(C_occ, S)

    result = DomainResult(Q_vir=Q_vir)
    domains: dict = {}

    # Build for strong pairs
    for p in pair_classification.strong_pairs:
        dom = _build_single_domain(
            molecule,
            basis,
            C_occ,
            S,
            Q_vir,
            atom_first,
            p.i,
            p.j,
            tcut_mkn,
            lindep_thresh,
        )
        if dom is not None:
            domains[(p.i, p.j)] = dom

    # Build for weak pairs
    for p in pair_classification.weak_pairs:
        dom = _build_single_domain(
            molecule,
            basis,
            C_occ,
            S,
            Q_vir,
            atom_first,
            p.i,
            p.j,
            tcut_mkn,
            lindep_thresh,
        )
        if dom is not None:
            domains[(p.i, p.j)] = dom

    # United domain for singles: all atoms that appear in any pair domain
    all_atoms_set: set[int] = set()
    for dom in domains.values():
        for a in dom.atom_indices:
            all_atoms_set.add(int(a))
    if all_atoms_set:
        united_atoms = np.array(sorted(all_atoms_set), dtype=int)
        # Build united domain AOs
        nbf_full = C_occ.shape[0]
        ao_mask = np.zeros(nbf_full, dtype=bool)
        for a in united_atoms:
            a_start = atom_first[a]
            a_end = atom_first[a + 1]
            if a_end > nbf_full:
                a_end = nbf_full
            ao_mask[a_start:a_end] = True

        C_pao_u, n_pao_u = build_pao_coeffs(
            Q_vir,
            ao_mask,
            S,
            lindep_thresh=lindep_thresh,
        )
        result.united_domain = PAODomain(
            pair_i=-1,  # sentinel for united
            pair_j=-1,
            atom_indices=united_atoms,
            ao_indices=np.where(ao_mask)[0],
            n_pao=n_pao_u,
            C_pao=C_pao_u,
        )

    result.domains = domains
    return result


def _build_single_domain(
    molecule,
    basis,
    C_occ: np.ndarray,
    S: np.ndarray,
    Q_vir: np.ndarray,
    atom_first: np.ndarray,
    i: int,
    j: int,
    tcut_mkn: float,
    lindep_thresh: float,
) -> PAODomain | None:
    """Build the PAO domain for a single pair (i, j)."""
    domain_atoms = select_domain_atoms_mulliken(
        C_occ,
        S,
        atom_first,
        i,
        j,
        tcut_mkn,
    )
    if len(domain_atoms) == 0:
        return None

    nbf = C_occ.shape[0]
    ao_mask = np.zeros(nbf, dtype=bool)
    for a in domain_atoms:
        a_start = atom_first[a]
        a_end = atom_first[a + 1]
        if a_end > nbf:
            a_end = nbf
        ao_mask[a_start:a_end] = True

    C_pao, n_pao = build_pao_coeffs(
        Q_vir,
        ao_mask,
        S,
        lindep_thresh=lindep_thresh,
    )
    return PAODomain(
        pair_i=i,
        pair_j=j,
        atom_indices=domain_atoms,
        ao_indices=np.where(ao_mask)[0],
        n_pao=n_pao,
        C_pao=C_pao,
    )


# ---------------------------------------------------------------------------
# Semicanonical PAO virtual basis (M2 -- real DLPNO-MP2)
# ---------------------------------------------------------------------------


def semicanonical_pao_basis(
    F_ao: np.ndarray,
    S_ao: np.ndarray,
    Q_vir: np.ndarray,
    ao_indices: np.ndarray,
    lindep_thresh: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an S-orthonormal, Fock-diagonal virtual basis for a domain.

    Starting from the *full-space* projected atomic orbitals of the
    domain (columns ``Q_vir[:, mu]`` for mu in the domain -- projection
    tails on out-of-domain AOs are kept), the redundant PAO set is
    canonically orthogonalised against the AO overlap and then
    semicanonicalised against the AO Fock matrix:

        P   = Q_vir[:, domain]                      (nbf, n_dom)
        S_p = Pᵀ S P  ->  eigh, drop eval < lindep   (redundancy removal)
        X   = P . v . l^{-1/2}                       (Xᵀ S X = 1)
        F_x = Xᵀ F X  ->  eigh -> (e, c)
        V   = X . c                                  (Vᵀ S V = 1, Vᵀ F V = diag e)

    Parameters
    ----------
    F_ao : ndarray, shape (nbf, nbf)
        AO-basis Fock matrix of the converged SCF.
    S_ao : ndarray, shape (nbf, nbf)
        AO overlap matrix.
    Q_vir : ndarray, shape (nbf, nbf)
        Virtual-space projector ``1 - C_occ C_occᵀ S``.
    ao_indices : ndarray
        Global AO indices of the domain.
    lindep_thresh : float
        Overlap-eigenvalue threshold for removing PAO redundancy.

    Returns
    -------
    V_semi : ndarray, shape (nbf, n_v)
        Semicanonical virtual orbitals of the domain in the full AO
        basis (S-orthonormal, Fock-diagonal).
    eps_pao : ndarray, shape (n_v,)
        Their Fock eigenvalues.
    """
    P = Q_vir[:, np.asarray(ao_indices, dtype=int)]  # (nbf, n_dom)
    S_p = P.T @ S_ao @ P
    S_p = 0.5 * (S_p + S_p.T)

    evals, evecs = np.linalg.eigh(S_p)
    keep = evals > lindep_thresh
    if not np.any(keep):
        return np.zeros((F_ao.shape[0], 0)), np.zeros(0)
    X = P @ (evecs[:, keep] / np.sqrt(evals[keep]))  # (nbf, n_v), Xᵀ S X = 1

    F_x = X.T @ F_ao @ X
    F_x = 0.5 * (F_x + F_x.T)
    eps_pao, c = np.linalg.eigh(F_x)
    V_semi = X @ c
    return V_semi, eps_pao
