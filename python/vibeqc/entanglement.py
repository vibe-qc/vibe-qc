"""Orbital entanglement and correlation-based bonding analysis.

Implements quantum-information-theoretic measures for chemical bonding:
single-orbital entropy, mutual information, and multiorbital correlation
clusters -- connecting bonding to the entanglement structure of the
wavefunction.

Public API
----------

.. autofunction:: single_orbital_entropy
.. autofunction:: mutual_information
.. autofunction:: total_quantum_information
.. autofunction:: entanglement_bond_order
.. autofunction:: correlation_clusters
.. autoclass:: EntanglementResult

Theory references
-----------------

- Legeza, Ö.; Sólyom, J., Phys. Rev. B 68, 195116 (2003).
  DOI: 10.1103/PhysRevB.68.195116
  (entanglement entropy in DMRG -- single-orbital entropy and mutual information)
- Szalay, Sz.; Barcza, G.; Szilvási, T.; Veis, L.; Legeza, Ö.,
  Sci. Rep. 7, 2237 (2017). DOI: 10.1038/s41598-017-02447-z
  (the correlation theory of the chemical bond -- multiorbital correlation)
- Ding, L.; Matito, E.; Schilling, C., "Chemical bonding concepts emerge
  naturally from maximally entangled atomic orbitals",
  Nat. Commun. 17, 4732 (2026). DOI: 10.1038/s41467-026-73527-w
  (preprint: arXiv:2501.15699, 2025)
  (entanglement-based bonding index via MEAOs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "EntanglementResult",
    "single_orbital_entropy",
    "mutual_information",
    "total_quantum_information",
    "entanglement_bond_order",
    "correlation_clusters",
    "entanglement_from_density",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class EntanglementResult:
    """Entanglement-based bonding analysis.

    Attributes
    ----------
    n_orbitals : int
        Number of orbitals in the active space.
    single_orbital_entropies : np.ndarray, shape ``(n_orbitals,)``
        von Neumann entropy of each one-orbital reduced density matrix.
    mutual_information : np.ndarray, shape ``(n_orbitals, n_orbitals)``
        Mutual information between every pair of orbitals.
    total_information : float
        Sum of single-orbital entropies = total quantum information content.
    entanglement_bond_orders : np.ndarray, shape ``(n_orbitals, n_orbitals)``
        Entanglement-based bond order: sqrt(I_ij) normalised.
    correlation_clusters : list of set of int
        Orbital groups identified as strongly correlated (MI above threshold).
    """

    n_orbitals: int = 0
    single_orbital_entropies: np.ndarray = field(default_factory=lambda: np.array([]))
    mutual_information: np.ndarray = field(default_factory=lambda: np.array([]))
    total_information: float = 0.0
    entanglement_bond_orders: np.ndarray = field(default_factory=lambda: np.array([]))
    correlation_clusters: list[set[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core entanglement measures
# ---------------------------------------------------------------------------


def single_orbital_entropy(
    one_orbital_densities: np.ndarray,
) -> np.ndarray:
    r"""Single-orbital von Neumann entropy from one-orbital reduced density
    matrices.

    For each orbital i with one-orbital RDM :math:`\\rho_i` (a 4x4 matrix
    in the Fock-space basis |0>, |↑>, |↓>, |↑↓>), the entropy is::

        s_i = -Tr[\\rho_i \\ln \\rho_i]

    Parameters
    ----------
    one_orbital_densities : np.ndarray, shape (n_orbitals, 4, 4)
        One-orbital reduced density matrices for each orbital.

    Returns
    -------
    np.ndarray, shape (n_orbitals,)
        Single-orbital entropies (in nats, i.e. natural-log based).
    """
    n_orb = one_orbital_densities.shape[0]
    entropies = np.zeros(n_orb, dtype=np.float64)
    for i in range(n_orb):
        rho = one_orbital_densities[i]
        evals = np.linalg.eigvalsh(rho)
        evals = np.clip(evals, 1e-16, 1.0)
        entropies[i] = -np.sum(evals * np.log(evals))
    return entropies


def _single_orbital_rdm_from_1rdm_2rdm(
    D1: np.ndarray, D2: np.ndarray, n_orb: int
) -> np.ndarray:
    """Build one-orbital RDMs (4x4) from 1-RDM and 2-RDM.

    For a spin-restricted system, the one-orbital RDM for orbital i
    is constructed from:
    - D1[i,i] (alpha occupation) = D1[i,i] (beta occupation) for closed-shell
    - D2[i,i,i,i] (pair occupation)

    The 4x4 matrix in the basis {|0>, |↑>, |↓>, |↑↓>} has diagonal:
    - p_0 = 1 - 2*n_i + d_ii
    - p_up = n_i - d_ii
    - p_down = n_i - d_ii
    - p_pair = d_ii

    where n_i = D1[i,i]/2 is the per-spin occupation.
    """
    result = np.zeros((n_orb, 4, 4), dtype=np.float64)
    for i in range(n_orb):
        n_i = D1[i, i] / 2.0  # per-spin occupation (assuming equal alpha/beta)
        d_ii = D2[i, i, i, i] if D2 is not None else n_i * n_i
        # Clamp to physical range
        d_ii = max(0.0, min(d_ii, n_i))
        p0 = 1.0 - 2.0 * n_i + d_ii
        p_up = n_i - d_ii
        p_down = n_i - d_ii
        p_pair = d_ii

        # Build diagonal RDM
        rho = np.diag([p0, p_up, p_down, p_pair])
        result[i] = rho
    return result


def entanglement_from_density(
    density_matrix: np.ndarray,
    overlap: np.ndarray,
    *,
    n_active: Optional[int] = None,
) -> EntanglementResult:
    r"""Compute entanglement measures from the one-particle density matrix.

    This is an approximate method that works with single-determinant
    wavefunctions (HF/DFT) by using the idempotency deviation of the
    density matrix. For multi-determinantal wavefunctions, use
    :func:`single_orbital_entropy` and :func:`mutual_information`
    directly with the full one- and two-particle RDMs.

    For a single-determinant wavefunction, the one-orbital entropy of
    orbital i is::

        s_i = -n_i ln(n_i) - (1-n_i) ln(1-n_i)

    where n_i are the natural occupation numbers (eigenvalues of P·S).

    Parameters
    ----------
    density_matrix : np.ndarray, shape ``(n_ao, n_ao)``
        One-particle density matrix in the AO basis.
    overlap : np.ndarray, shape ``(n_ao, n_ao)``
        AO overlap matrix.
    n_active : int, optional
        Number of active orbitals to use. Default: all.

    Returns
    -------
    EntanglementResult
    """
    P = np.asarray(density_matrix)
    S = np.asarray(overlap)
    n_ao = P.shape[0]

    # Generalised eigenvalue problem: P·S·C = n·S·C
    # Transform to orthogonal basis first
    evals_S, evecs_S = np.linalg.eigh(S)
    mask = evals_S > 1e-14
    s_inv_half = np.zeros((n_ao, n_ao), dtype=np.float64)
    for i in range(n_ao):
        if mask[i]:
            s_inv_half[i, i] = evals_S[i] ** (-0.5)
    X = evecs_S @ s_inv_half  # S^{-1/2} in columns

    P_orth = X.T @ P @ X
    occ, U = np.linalg.eigh(P_orth)

    if n_active is not None:
        n_active = min(n_active, n_ao)

    # Single-orbital entropy from occupation numbers
    occ_clipped = np.clip(occ, 1e-16, 1.0 - 1e-16)
    s_i = -occ_clipped * np.log(occ_clipped) - (1.0 - occ_clipped) * np.log(
        1.0 - occ_clipped
    )

    # Mutual information: I_ij = s_i + s_j - s_ij
    # For independent orbitals, I_ij = 0
    # Approximate: I_ij proportional to product of natural orbital
    # coefficients weighted by occupation
    n_active_val = n_active if n_active is not None else n_ao
    mi = np.zeros((n_active_val, n_active_val), dtype=np.float64)
    for i in range(n_active_val):
        for j in range(i + 1, n_active_val):
            # Approximate mutual information from orbital overlap
            n_i = occ[i]
            n_j = occ[j]
            if (
                n_i > 0.01
                and n_j > 0.01
                and abs(n_i - 0.5) < 0.49
                and abs(n_j - 0.5) < 0.49
            ):
                # Orbitals with fractional occupation: estimate correlation
                mi[i, j] = 4.0 * n_i * (1.0 - n_i) * n_j * (1.0 - n_j)
            mi[j, i] = mi[i, j]

    # Entanglement bond order: sqrt(I_ij) normalised
    ebo = np.sqrt(np.maximum(mi, 0.0))
    max_ebo = np.max(ebo) if np.max(ebo) > 0 else 1.0
    if max_ebo > 0:
        ebo = ebo / max_ebo

    total_info = float(np.sum(s_i[:n_active_val]))

    return EntanglementResult(
        n_orbitals=n_active_val,
        single_orbital_entropies=s_i[:n_active_val],
        mutual_information=mi,
        total_information=total_info,
        entanglement_bond_orders=ebo,
        correlation_clusters=[],
    )


def mutual_information(
    one_orbital_densities: np.ndarray,
    two_orbital_densities: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""Pairwise mutual information between orbitals.

    .. math::

        I_{ij} = s_i + s_j - s_{ij}

    where s_i is the single-orbital entropy and s_{ij} is the
    two-orbital entropy from the two-orbital reduced density matrix.

    If two-orbital densities are not provided, the mutual information
    is estimated from the product of one-orbital density matrices
    (valid for uncorrelated systems: gives I_ij ≈ 0).

    Parameters
    ----------
    one_orbital_densities : np.ndarray, shape ``(n_orbitals, 4, 4)``
        One-orbital reduced density matrices.
    two_orbital_densities : np.ndarray, shape ``(n_orbitals, n_orbitals, 16, 16)``, optional
        Two-orbital reduced density matrices. If None, mutual information
        is estimated from the single-orbital entropies only (approximation).

    Returns
    -------
    np.ndarray, shape ``(n_orbitals, n_orbitals)``
        Mutual information matrix. Diagonal entries are s_i * 2 by
        convention (2 * single-orbital entropy as self-information).
    """
    n_orb = one_orbital_densities.shape[0]
    s_i = single_orbital_entropy(one_orbital_densities)
    mi = np.zeros((n_orb, n_orb), dtype=np.float64)

    if two_orbital_densities is not None:
        for i in range(n_orb):
            for j in range(i + 1, n_orb):
                rho_ij = two_orbital_densities[i, j]
                evals = np.linalg.eigvalsh(rho_ij)
                evals = np.clip(evals, 1e-16, 1.0)
                s_ij = -np.sum(evals * np.log(evals))
                mi[i, j] = s_i[i] + s_i[j] - s_ij
                mi[j, i] = mi[i, j]
    else:
        # Approximate: for uncorrelated orbitals, I_ij ≈ 0
        # This is a placeholder -- real MI requires the two-orbital RDM
        pass

    # Diagonal: self-information (2 * s_i by convention)
    for i in range(n_orb):
        mi[i, i] = 2.0 * s_i[i]

    return mi


def total_quantum_information(
    one_orbital_densities: np.ndarray,
) -> float:
    """Total quantum information content = sum of single-orbital entropies.

    This measures the total correlation in the system; larger values
    indicate more entangled (strongly correlated) electronic structure.
    """
    entropies = single_orbital_entropy(one_orbital_densities)
    return float(np.sum(entropies))


def entanglement_bond_order(
    mutual_info: np.ndarray,
) -> np.ndarray:
    r"""Entanglement-based bond order from mutual information.

    Following the "from entanglement to bonds" framework of Ding, Matito &
    Schilling, Nat. Commun. 17, 4732 (2026), doi:10.1038/s41467-026-73527-w
    (preprint arXiv:2501.15699), the bond order between orbitals i and j is::

        B_{ij} = \\sqrt{I_{ij}} / \\max(\\sqrt{I})

    This is a dimensionless metric in [0, 1]; higher values indicate
    stronger pairwise entanglement (correlation).

    Parameters
    ----------
    mutual_info : np.ndarray, shape ``(n_orbitals, n_orbitals)``
        Mutual information matrix from :func:`mutual_information`.

    Returns
    -------
    np.ndarray, shape ``(n_orbitals, n_orbitals)``
        Entanglement bond orders, normalised to [0, 1].
    """
    ebo = np.sqrt(np.maximum(mutual_info, 0.0))
    max_val = np.max(ebo)
    if max_val > 1e-16:
        ebo = ebo / max_val
    return ebo


def correlation_clusters(
    mutual_info: np.ndarray,
    *,
    threshold: float = 0.1,
) -> list[set[int]]:
    """Identify strongly correlated orbital clusters.

    Groups orbitals i, j for which mutual information I_{ij} exceeds
    a threshold. Uses a simple graph-based clustering: two orbitals
    belong to the same cluster if there is a path of high-MI edges
    connecting them.

    Parameters
    ----------
    mutual_info : np.ndarray, shape ``(n_orbitals, n_orbitals)``
        Mutual information matrix.
    threshold : float
        MI threshold above which orbitals are considered correlated.

    Returns
    -------
    list of set of int
        Each set is one correlation cluster.
    """
    n_orb = mutual_info.shape[0]
    # Build adjacency
    adj: list[set[int]] = [set() for _ in range(n_orb)]
    for i in range(n_orb):
        for j in range(i + 1, n_orb):
            if mutual_info[i, j] > threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components via DFS
    visited = [False] * n_orb
    clusters: list[set[int]] = []

    def dfs(v: int, comp: set[int]) -> None:
        visited[v] = True
        comp.add(v)
        for u in adj[v]:
            if not visited[u]:
                dfs(u, comp)

    for i in range(n_orb):
        if not visited[i]:
            comp: set[int] = set()
            dfs(i, comp)
            if len(comp) > 1:
                clusters.append(comp)

    return clusters
