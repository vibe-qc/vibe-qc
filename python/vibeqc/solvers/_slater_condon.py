"""Slater-Condon rules for matrix elements between determinants.

Implements the standard Slater-Condon rules for evaluating
<D|H|D'> where H = S_{pq} h_{pq} a+_p a_q + 1/4 S_{pqrs} g_{pqrs} a+_p a+_q a_s a_r

The rules are:
  * **Diagonal** (D = D'):
    S_i h_{ii} + 1/2 S_{ij} (g_{ijij} - g_{ijji})

  * **Single excitation** (D' = a+_a a_i D):
    sign x [h_{ia} + S_j (g_{ijaj} - g_{ijja})]

  * **Double excitation** (D' = a+_a a_i a+_b a_j D, i<j, a<b):
    sign x (g_{ijab} - g_{ijba})

All other matrix elements are zero for a two-body Hamiltonian.

Two determinant representations share this module and the rules differ:

* ``SpinDet = (alpha_occ, beta_occ)`` -- spin-orbital determinants. The
  ``*_unrestricted`` functions apply the rules above sector by sector, and
  :func:`single_excitation_matrix_element` /
  :func:`double_excitation_matrix_element` are the per-sector primitives
  they (and callers walking one spin string) use.
* ``Det`` -- a **closed-shell** determinant: a tuple of *doubly occupied
  spatial* orbitals. This is the seniority-zero basis (every orbital holds
  0 or 2 electrons), and two closed-shell determinants can differ only by
  whole pairs. Their matrix elements are the Slater-Condon rules
  specialised to that basis (the ``a+_a(alpha) a+_a(beta) a_i(beta)
  a_i(alpha)`` pair excitation is a rank-2 spin-orbital excitation with one
  hole and one particle in each sector, so only the alpha-beta two-electron
  term survives and its phase is (+1)):

  * **Diagonal**:            2 S_i h_{ii} + S_{ij} (2 g_{ijij} - g_{ijji})
  * **One pair moved** (i^2 -> a^2):    g_{iiaa}   (phase +1)
  * **Two or more pairs moved**:        0

  :func:`hamiltonian_matrix_element`, :func:`build_hamiltonian_matrix` and
  :func:`hamiltonian_dot` implement these (GitLab #639). They are exact for
  the closed-shell space; a full CI in the spin-orbital space uses the
  ``*_unrestricted`` family.

References
----------
* Szabo & Ostlund, *Modern Quantum Chemistry*, Sec.2.3.
* Helgaker, Jorgensen & Olsen, *Molecular Electronic-Structure Theory*,
  Sec.1.4 (Slater-Condon rules in second quantization) and Sec.11.8.1
  (alpha/beta string matrix elements).
* Bytautas, Henderson, Jimenez-Hoyos, Ellis & Scuseria, J. Chem. Phys.
  135, 044119 (2011), doi:10.1063/1.3613706 -- the seniority-zero (DOCI)
  Hamiltonian the closed-shell ``Det`` rules realise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._determinant import Det, excitation_rank

# -- Phase factor ----------------------------------------------------------


def _phase_factor(occ: Det, excitation_path: list[tuple[int, int]]) -> int:
    """Compute the fermionic phase (-1)^P for a sequence of creations/annihilations.

    Parameters
    ----------
    occ : Det
        Sorted tuple of occupied orbital indices.
    excitation_path : list[tuple[int, int]]
        List of (hole, particle) pairs, ordered as applied.

    Returns
    -------
    sign : int (+1 or -1)
    """
    # Build bitmask and compute via permutation count
    occ_set = set(occ)
    sign = 1
    for hole, particle in excitation_path:
        # Count occupied orbitals strictly between hole and particle
        lo = min(hole, particle)
        hi = max(hole, particle)
        for idx in occ_set:
            if lo < idx < hi:
                sign *= -1
        # Update occupation
        occ_set.discard(hole)
        occ_set.add(particle)
    return sign


# -- Diagonal --------------------------------------------------------------


def diagonal_matrix_element(
    occ: Det,
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> float:
    """<D|H|D> for a closed-shell (spin-restricted) determinant.

    Each orbital in ``occ`` is doubly occupied (a + b).

    Formula:
      E_D = 2 S_i h_{ii} + S_{ij} (2 g_{ijij} - g_{ijji})
    """
    occ_arr = np.asarray(occ, dtype=np.intp)
    E = 2.0 * float(np.sum(h1e[occ_arr, occ_arr]))
    ii, jj = np.meshgrid(occ_arr, occ_arr, indexing="ij")
    E += float(np.sum(2.0 * h2e_phys[ii, jj, ii, jj] - h2e_phys[ii, jj, jj, ii]))
    return E


def diagonal_matrix_element_unrestricted(
    alpha_occ: Det,
    beta_occ: Det,
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> float:
    """<D|H|D> for an unrestricted determinant.

    E_D = S_{iina} h_{ii} + S_{iinb} h_{ii}
        + 1/2 S_{iina} S_{jina} (g_{ijij} - g_{ijji})
        + 1/2 S_{iinb} S_{jinb} (g_{ijij} - g_{ijji})
        +    S_{iina} S_{jinb} g_{ijij}
    """
    a_arr = np.asarray(alpha_occ, dtype=np.intp)
    b_arr = np.asarray(beta_occ, dtype=np.intp)
    E = float(np.sum(h1e[a_arr, a_arr]) + np.sum(h1e[b_arr, b_arr]))
    # aa
    ii, jj = np.meshgrid(a_arr, a_arr, indexing="ij")
    E += 0.5 * float(np.sum(h2e_phys[ii, jj, ii, jj] - h2e_phys[ii, jj, jj, ii]))
    # bb
    ii, jj = np.meshgrid(b_arr, b_arr, indexing="ij")
    E += 0.5 * float(np.sum(h2e_phys[ii, jj, ii, jj] - h2e_phys[ii, jj, jj, ii]))
    # ab
    E += float(
        np.sum(h2e_phys[a_arr[:, None], b_arr[None, :], a_arr[:, None], b_arr[None, :]])
    )
    return E


def batch_diagonal_elements(
    det_list: list[Det],
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> np.ndarray:
    """Vectorized diagonal elements for a list of closed-shell determinants.

    Parameters
    ----------
    det_list : list[Det]
        List of determinants.
    h1e : (norb, norb) ndarray
    h2e_phys : (norb, norb, norb, norb) ndarray

    Returns
    -------
    diag : (ndet,) ndarray
    """
    diag = np.zeros(len(det_list))
    for idx, occ in enumerate(det_list):
        occ_arr = np.asarray(occ, dtype=np.intp)
        d = 2.0 * np.sum(h1e[occ_arr, occ_arr])
        ii, jj = np.meshgrid(occ_arr, occ_arr, indexing="ij")
        d += np.sum(2.0 * h2e_phys[ii, jj, ii, jj] - h2e_phys[ii, jj, jj, ii])
        diag[idx] = d
    return diag


def batch_diagonal_elements_unrestricted(
    det_list: list[SpinDet],
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> np.ndarray:
    """Vectorized diagonal elements for a list of unrestricted SpinDet.

    Parameters
    ----------
    det_list : list[SpinDet]
        List of (alpha_occ, beta_occ) tuples.
    h1e : (norb, norb) ndarray
    h2e_phys : (norb, norb, norb, norb) ndarray

    Returns
    -------
    diag : (ndet,) ndarray
    """
    diag = np.zeros(len(det_list))
    for idx, (a_occ, b_occ) in enumerate(det_list):
        diag[idx] = diagonal_matrix_element_unrestricted(a_occ, b_occ, h1e, h2e_phys)
    return diag


# -- Single excitation -----------------------------------------------------


def single_excitation_matrix_element(
    occ: Det,
    i: int,  # hole
    a: int,  # particle
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> float:
    """<D^{a}_{i}|H|D> for a spin-restricted (closed-shell) determinant.

    D^{a}_{i} has the a electron from orbital i promoted to a.
    The phase is (-1)^{#occupied between i and a}.

    Formula:
      h_{ia} + S_j (2 g_{ijaj} - g_{ijja})
    """
    sign = _phase_factor(occ, [(i, a)])
    occ_arr = np.asarray(occ, dtype=np.intp)
    val = h1e[i, a]
    val += float(
        np.sum(
            2.0 * h2e_phys[i, occ_arr, a, occ_arr] - h2e_phys[i, occ_arr, occ_arr, a]
        )
    )
    return sign * val


# -- Double excitation -----------------------------------------------------


def double_excitation_matrix_element(
    occ: Det,
    i: int,  # hole 1
    j: int,  # hole 2  (i < j)
    a: int,  # particle 1
    b: int,  # particle 2  (a < b)
    h2e_phys: np.ndarray,
) -> float:
    """<D^{ab}_{ij}|H|D> for a spin-restricted determinant.

    Both electrons are a-electrons promoted from i,j to a,b.
    Phase: (-1)^{#occupied between i,a + #occupied between j,b after first excitation}.

    Formula:
      g_{ijab} - g_{ijba}
    """
    # Phase for two successive single excitations
    sign = _phase_factor(occ, [(i, a), (j, b)])
    return sign * (h2e_phys[i, j, a, b] - h2e_phys[i, j, b, a])


# -- Pair excitation (closed-shell / seniority-zero basis) ------------------


def pair_excitation_matrix_element(
    i: int,  # doubly occupied orbital vacated
    a: int,  # orbital that receives the pair
    h2e_phys: np.ndarray,
) -> float:
    """<D^{aa}_{ii}|H|D> for closed-shell determinants: one pair moved i -> a.

    In the spin-orbital picture this is the double excitation
    ``i(alpha) i(beta) -> a(alpha) a(beta)``: one hole and one particle in
    each spin sector.  The general rule ``g_{ijab} - g_{ijba}`` (module
    docstring) reduces to the opposite-spin Coulomb-type term because the
    exchange term ``g_{ijba}`` couples ``alpha`` to ``beta`` and vanishes.
    Helgaker, Jorgensen & Olsen, *Molecular Electronic-Structure Theory*,
    Sec.1.4 (Slater-Condon rules) and Sec.11.8.1 (alpha/beta strings):

        <a(alpha) a(beta) | H | i(alpha) i(beta)> = <ii|aa> = g_{iiaa}

    The two single-sector phases are identical, so their product is (+1)
    for every ``i``, ``a`` and every set of spectator pairs.  This is the
    off-diagonal element of the seniority-zero (DOCI) Hamiltonian (Bytautas
    et al., J. Chem. Phys. 135, 044119 (2011), doi:10.1063/1.3613706,
    Sec.II).  For real orbitals ``g_{iiaa} = (ia|ia) = K_{ia}``, the exchange
    integral -- e.g. 0.18125791 Ha for H2/STO-3G at R = 1.4 bohr (GitLab
    #639), which couples ``sigma_g^2`` to ``sigma_u^2`` and turns the
    two-determinant closed-shell space into the exact minimal-basis FCI.
    """
    return float(h2e_phys[i, i, a, a])


# -- Hamiltonian matrix element (general) ----------------------------------


def hamiltonian_matrix_element(
    bra_occ: Det,
    ket_occ: Det,
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
    *,
    nelec: Optional[int] = None,
) -> float:
    """Compute <D_bra|H|D_ket> for closed-shell (spin-restricted) determinants.

    ``Det`` is a tuple of doubly occupied spatial orbitals, so the two
    determinants can differ only by whole pairs (module docstring):

    * rank 0 -- the closed-shell diagonal;
    * rank 1 -- one pair moved, :func:`pair_excitation_matrix_element`
      (``g_{iiaa}``, phase +1);
    * rank >= 2 -- two or more pairs moved: a four-or-more-electron
      excitation, zero for a two-body Hamiltonian.

    Before GitLab #639 rank 1 was scored with the one-electron single-
    excitation rule (the Brillouin term, ~1e-16 Ha for HF orbitals) and
    rank 2 with the same-spin double rule, so a closed-shell space returned
    the HF energy instead of its seniority-zero CI energy.
    """
    rank = excitation_rank(bra_occ, ket_occ)
    if rank == 0:
        return diagonal_matrix_element(bra_occ, h1e, h2e_phys)

    if rank == 1:
        bra_set = set(bra_occ)
        ket_set = set(ket_occ)
        (i,) = ket_set - bra_set  # pair vacated (in ket, not bra)
        (a,) = bra_set - ket_set  # pair created (in bra, not ket)
        return pair_excitation_matrix_element(i, a, h2e_phys)

    return 0.0


# -- Hamiltonian-vector product (s-vector) ---------------------------------


def hamiltonian_dot(
    coeffs: np.ndarray,
    det_list: list[Det],
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
    *,
    diagonal_precomputed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute H.c (s-vector) for a CI wavefunction.

    .. deprecated:: 0.15
        This Python O(ndet^2) implementation is superseded by the C++
        ``casci_direct_sigma`` kernel (OpenMP-parallel, O(ndet) pair-driven
        Slater-Condon).  The C++ path is the production default for all CI
        s-vector builds; this function is retained for validation parity only.
    """
    import warnings

    warnings.warn(
        "hamiltonian_dot is deprecated; use the C++ casci_direct_sigma kernel",
        DeprecationWarning,
        stacklevel=2,
    )

    """Parameters
    ----------
    coeffs : (ndet,) ndarray
        CI coefficient vector.
    det_list : list[Det]
        List of determinants in the CI expansion.
    h1e : (norb, norb) ndarray
    h2e_phys : (norb, norb, norb, norb) ndarray
    diagonal_precomputed : (ndet,) ndarray, optional
        Precomputed diagonal elements for speed.

    Returns
    -------
    sigma : (ndet,) ndarray
        H.c.
    """
    import warnings

    warnings.warn(
        "hamiltonian_dot is deprecated; use the C++ casci_direct_sigma kernel",
        DeprecationWarning,
        stacklevel=2,
    )

    ndet = len(det_list)
    sigma = np.zeros(ndet)

    # Precompute diagonals if not provided
    if diagonal_precomputed is None:
        diag = batch_diagonal_elements(det_list, h1e, h2e_phys)
    else:
        diag = diagonal_precomputed

    sigma += diag * coeffs

    # Off-diagonal: closed-shell determinants couple only through single
    # pair moves i^2 -> a^2 (seniority-zero rules, module docstring; #639).
    norb = h1e.shape[0]
    det_to_idx = {d: i for i, d in enumerate(det_list)}
    for I in range(ndet):
        occ_I = det_list[I]
        occ_set = set(occ_I)
        vir_list = sorted(set(range(norb)) - occ_set)

        for i in occ_set:
            for a in vir_list:
                new_occ = tuple(sorted((occ_set - {i}) | {a}))
                J = det_to_idx.get(new_occ)
                if J is not None:
                    sigma[I] += pair_excitation_matrix_element(i, a, h2e_phys) * coeffs[J]

    return sigma


# -- Build full Hamiltonian matrix -----------------------------------------


def build_hamiltonian_matrix(
    det_list: list[Det],
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> np.ndarray:
    """Build the CI Hamiltonian matrix H_{IJ} = <D_I|H|D_J> over closed-shell
    determinants -- the seniority-zero (DOCI) Hamiltonian of ``det_list``.

    ``Det`` is a tuple of doubly occupied spatial orbitals, so the only
    non-zero off-diagonal elements are single pair moves i^2 -> a^2,
    ``H_IJ = g_{iiaa}`` (module docstring; GitLab #639).  Over *all*
    C(norb, nocc) closed-shell determinants this is the DOCI matrix; it equals
    the full CI only when the exact state is seniority-zero (two electrons,
    e.g. H2/STO-3G: both closed-shell determinants give the FCI energy
    -1.1372759436 Ha).  For a spin-orbital full CI use
    :func:`build_hamiltonian_matrix_unrestricted`.

    Parameters
    ----------
    det_list : list[Det]
        Closed-shell determinants.
    h1e : (norb, norb) ndarray
    h2e_phys : (norb, norb, norb, norb) ndarray

    Returns
    -------
    H : (ndet, ndet) ndarray
        Symmetric Hamiltonian matrix.
    """
    ndet = len(det_list)
    H = np.zeros((ndet, ndet))
    norb = h1e.shape[0]

    # Precompute diagonal
    diag = batch_diagonal_elements(det_list, h1e, h2e_phys)
    H[np.diag_indices(ndet)] = diag

    # Off-diagonal: one pair moved.  O(1) lookups via an index map.
    det_to_idx = {d: i for i, d in enumerate(det_list)}
    for I in range(ndet):
        occ_I = det_list[I]
        occ_set = set(occ_I)
        vir_list = sorted(set(range(norb)) - occ_set)

        for i in occ_set:
            for a in vir_list:
                new_occ = tuple(sorted((occ_set - {i}) | {a}))
                J = det_to_idx.get(new_occ)
                if J is not None and J > I:
                    val = pair_excitation_matrix_element(i, a, h2e_phys)
                    H[I, J] = val
                    H[J, I] = val

    return H


# -- Unrestricted Slater-Condon rules -------------------------------------

from ._determinant import SpinDet


def _unrestricted_single_excitation(
    occ_ref: Det,
    occ_other: Det,
    i: int,
    a: int,
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> float:
    """<...a+_a a_i...|H|D> for an excitation in one spin sector.

    The other spin sector's occupation is ``occ_other`` and is
    unchanged by the excitation.  The phase is computed from
    ``occ_ref`` (the sector being excited).

    Formula:
      sign x [h_{ia} + S_{kinocc_ref} (g_{ikak} - g_{ikka})
             + S_{kinocc_other} g_{ikak}]
    """
    sign = _phase_factor(occ_ref, [(i, a)])
    ref_arr = np.asarray(occ_ref, dtype=np.intp)
    other_arr = np.asarray(occ_other, dtype=np.intp)
    val = h1e[i, a]
    val += float(
        np.sum(h2e_phys[i, ref_arr, a, ref_arr] - h2e_phys[i, ref_arr, ref_arr, a])
    )
    val += float(np.sum(h2e_phys[i, other_arr, a, other_arr]))
    return sign * val


def hamiltonian_matrix_element_unrestricted(
    alpha_bra: Det,
    beta_bra: Det,
    alpha_ket: Det,
    beta_ket: Det,
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> float:
    """<a_bra b_bra|H|a_ket b_ket> for unrestricted determinants.

    Dispatches based on excitation ranks in each spin sector.
    Total rank = rank_a + rank_b <= 2 for non-zero matrix elements.
    """
    rank_a = excitation_rank(alpha_bra, alpha_ket)
    rank_b = excitation_rank(beta_bra, beta_ket)
    total = rank_a + rank_b

    if total == 0:
        return diagonal_matrix_element_unrestricted(alpha_bra, beta_bra, h1e, h2e_phys)

    if total > 2:
        return 0.0

    holes_a = sorted(set(alpha_ket) - set(alpha_bra))
    parts_a = sorted(set(alpha_bra) - set(alpha_ket))
    holes_b = sorted(set(beta_ket) - set(beta_bra))
    parts_b = sorted(set(beta_bra) - set(beta_ket))

    # Case 1: Single excitation in a only
    if rank_a == 1 and rank_b == 0:
        return _unrestricted_single_excitation(
            alpha_ket, beta_ket, holes_a[0], parts_a[0], h1e, h2e_phys
        )

    # Case 2: Single excitation in b only
    if rank_a == 0 and rank_b == 1:
        return _unrestricted_single_excitation(
            beta_ket, alpha_ket, holes_b[0], parts_b[0], h1e, h2e_phys
        )

    # Case 3: Double excitation, both in a
    if rank_a == 2 and rank_b == 0:
        return double_excitation_matrix_element(
            alpha_ket, holes_a[0], holes_a[1], parts_a[0], parts_a[1], h2e_phys
        )

    # Case 4: Double excitation, both in b
    if rank_a == 0 and rank_b == 2:
        return double_excitation_matrix_element(
            beta_ket, holes_b[0], holes_b[1], parts_b[0], parts_b[1], h2e_phys
        )

    # Case 5: One excitation in each spin sector
    if rank_a == 1 and rank_b == 1:
        sign = _phase_factor(alpha_ket, [(holes_a[0], parts_a[0])])
        sign *= _phase_factor(beta_ket, [(holes_b[0], parts_b[0])])
        return sign * h2e_phys[holes_a[0], holes_b[0], parts_a[0], parts_b[0]]

    return 0.0


def build_hamiltonian_matrix_unrestricted(
    det_list: list[SpinDet],
    h1e: np.ndarray,
    h2e_phys: np.ndarray,
) -> np.ndarray:
    """Build the full CI Hamiltonian matrix in the unrestricted
    determinant basis.

    Uses a connection map to avoid O(ndet^2) all-pairs iteration --
    only computes matrix elements for determinants connected by
    <= 2 excitations.

    Parameters
    ----------
    det_list : list[SpinDet]
        List of (alpha_occ, beta_occ) tuples.
    h1e : (norb, norb) ndarray
    h2e_phys : (norb, norb, norb, norb) ndarray

    Returns
    -------
    H : (ndet, ndet) ndarray
        Symmetric Hamiltonian matrix.
    """
    ndet = len(det_list)
    H = np.zeros((ndet, ndet))
    norb = h1e.shape[0]

    # Precompute diagonal
    diag = batch_diagonal_elements_unrestricted(det_list, h1e, h2e_phys)
    H[np.diag_indices(ndet)] = diag

    # Build index map for O(1) lookup
    det_to_idx = {d: i for i, d in enumerate(det_list)}

    # Off-diagonal: only connected pairs via explicit excitation generation
    for I in range(ndet):
        aI, bI = det_list[I]
        occ_a = set(aI)
        occ_b = set(bI)
        vir_a = sorted(set(range(norb)) - occ_a)
        vir_b = sorted(set(range(norb)) - occ_b)

        # a singles
        for i in occ_a:
            for a in vir_a:
                new_a = tuple(sorted((occ_a - {i}) | {a}))
                J = det_to_idx.get((new_a, bI))
                if J is not None and J > I:
                    val = hamiltonian_matrix_element_unrestricted(
                        new_a, bI, aI, bI, h1e, h2e_phys
                    )
                    if abs(val) > 1e-16:
                        H[I, J] = val
                        H[J, I] = val

        # b singles
        for i in occ_b:
            for a in vir_b:
                new_b = tuple(sorted((occ_b - {i}) | {a}))
                J = det_to_idx.get((aI, new_b))
                if J is not None and J > I:
                    val = hamiltonian_matrix_element_unrestricted(
                        aI, new_b, aI, bI, h1e, h2e_phys
                    )
                    if abs(val) > 1e-16:
                        H[I, J] = val
                        H[J, I] = val

        # aa doubles
        a_list = sorted(occ_a)
        for idx_i in range(len(a_list)):
            for idx_j in range(idx_i + 1, len(a_list)):
                i, j = a_list[idx_i], a_list[idx_j]
                for idx_p in range(len(vir_a)):
                    for idx_q in range(idx_p + 1, len(vir_a)):
                        a, b = vir_a[idx_p], vir_a[idx_q]
                        new_a = tuple(sorted((occ_a - {i, j}) | {a, b}))
                        J = det_to_idx.get((new_a, bI))
                        if J is not None and J > I:
                            val = hamiltonian_matrix_element_unrestricted(
                                new_a, bI, aI, bI, h1e, h2e_phys
                            )
                            if abs(val) > 1e-16:
                                H[I, J] = val
                                H[J, I] = val

        # bb doubles
        b_list = sorted(occ_b)
        for idx_i in range(len(b_list)):
            for idx_j in range(idx_i + 1, len(b_list)):
                i, j = b_list[idx_i], b_list[idx_j]
                for idx_p in range(len(vir_b)):
                    for idx_q in range(idx_p + 1, len(vir_b)):
                        a, b = vir_b[idx_p], vir_b[idx_q]
                        new_b = tuple(sorted((occ_b - {i, j}) | {a, b}))
                        J = det_to_idx.get((aI, new_b))
                        if J is not None and J > I:
                            val = hamiltonian_matrix_element_unrestricted(
                                aI, new_b, aI, bI, h1e, h2e_phys
                            )
                            if abs(val) > 1e-16:
                                H[I, J] = val
                                H[J, I] = val

        # ab doubles
        for i in occ_a:
            for j in occ_b:
                for a in vir_a:
                    for b in vir_b:
                        new_a = tuple(sorted((occ_a - {i}) | {a}))
                        new_b = tuple(sorted((occ_b - {j}) | {b}))
                        J = det_to_idx.get((new_a, new_b))
                        if J is not None and J > I:
                            val = hamiltonian_matrix_element_unrestricted(
                                new_a, new_b, aI, bI, h1e, h2e_phys
                            )
                            if abs(val) > 1e-16:
                                H[I, J] = val
                                H[J, I] = val

    return H
