"""Orbital localisation for DLPNO, post-SCF methods and chemical analysis.

Provides Foster-Boys, Pipek-Mezey and intrinsic-bond-orbital (IBO)
localisation of occupied molecular orbitals via 2x2 Jacobi sweeps -- the
first step toward local correlation, and the basis of the Lewis-structure
view surfaced to vibe-view.

**Which one to use.** For *chemical interpretation*, use :func:`ibo_localise`.
Foster-Boys and Pipek-Mezey are here because DLPNO needs cheap domains; PM in
particular maximises *Mulliken* populations, which have no basis-set limit, so
PM orbitals are not a sound interpretation default (Knizia 2013, § 3).  IBO is
PM's functional evaluated over intrinsic-atomic-orbital populations instead.

Both methods maximise a sum of squared diagonal quantities (orbital
dipole moments for Boys, Mulliken atomic populations for PM), so they
share the classic Edmiston-Ruedenberg Jacobi solution: for each orbital
pair (i, j) the optimal rotation angle a satisfies

    cos 4a = -A / √(A^2 + B^2),   sin 4a = B / √(A^2 + B^2)

with  A = S_c [ M_ij,c^2 - 1/4(M_ii,c - M_jj,c)^2 ]  and
      B = S_c [ M_ij,c . (M_ii,c - M_jj,c) ],

where M runs over the Cartesian dipole components (Boys) or atomic
populations (PM).  This branch choice selects the maximising stationary
point; a = 0 at a converged maximum.

References
----------
* Foster & Boys, *Rev. Mod. Phys.* 32, 300 (1960).
* Edmiston & Ruedenberg, *Rev. Mod. Phys.* 35, 457 (1963) -- Jacobi
  rotation solution.
* Pipek & Mezey, *J. Chem. Phys.* 90, 4916 (1989).
* Knizia, *J. Chem. Theory Comput.* 9, 4834 (2013),
  doi:10.1021/ct400687b -- IAOs and IBOs.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Foster-Boys (Boys) localisation
# ---------------------------------------------------------------------------


def boys_objective(C: np.ndarray, dipoles: np.ndarray) -> float:
    """Boys localisation objective ``S_i <phi_i|r|phi_i>^2``.

    Parameters
    ----------
    C : ndarray, shape (nbf, norb)
        MO coefficient matrix.
    dipoles : ndarray, shape (nbf, nbf, 3)
        Cartesian dipole integrals <mu|r|ν> in the AO basis.

    Returns
    -------
    float
        The objective value (bohr^2).  Larger = more localised.
    """
    val = 0.0
    for c in range(3):
        d = np.einsum("mi,mn,ni->i", C, dipoles[:, :, c], C)
        val += float(np.sum(d**2))
    return val


def foster_boys_localise(
    C: np.ndarray,
    dipoles: np.ndarray,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
) -> np.ndarray:
    """Localise orbitals via the Foster-Boys criterion.

    Maximises ``S_i <phi_i | r | phi_i>^2`` using sequential 2x2 Jacobi
    rotations applied to the MO coefficient matrix.

    Parameters
    ----------
    C : ndarray, shape (nbf, norb)
        MO coefficient matrix (columns = orbitals to localise).
    dipoles : ndarray, shape (nbf, nbf, 3)
        Cartesian dipole integrals <mu|r|ν> in the AO basis.
    max_iter : int
        Maximum Jacobi sweeps. Default 200.
    conv_tol : float
        Convergence threshold on max rotation angle. Default 1e-8.

    Returns
    -------
    C_loc : ndarray, shape (nbf, norb)
        Localised MO coefficients (C @ U, where U is unitary).
    """
    norb = C.shape[1]
    # Build orbital dipole matrices.
    D = np.zeros((norb, norb, 3))
    for i in range(3):
        D[:, :, i] = C.T @ dipoles[:, :, i] @ C

    U = np.eye(norb)

    for iteration in range(max_iter):
        max_angle = 0.0
        for i in range(norb):
            for j in range(i + 1, norb):
                # Edmiston-Ruedenberg Jacobi angle for the maximum.
                A_val = float(
                    np.sum(D[i, j, :] ** 2 - 0.25 * (D[i, i, :] - D[j, j, :]) ** 2)
                )
                B_val = float(np.sum(D[i, j, :] * (D[i, i, :] - D[j, j, :])))
                if A_val * A_val + B_val * B_val < 1e-24:
                    continue

                theta = 0.25 * np.arctan2(B_val, -A_val)
                max_angle = max(max_angle, abs(theta))
                if abs(theta) < 1e-14:
                    continue
                c, s = np.cos(theta), np.sin(theta)

                # Save all values needed before any modification.
                D_ii = D[i, i, :].copy()
                D_jj = D[j, j, :].copy()
                D_ij = D[i, j, :].copy()
                D_ji = D[j, i, :].copy()

                # Update diagonal and off-diagonal blocks.
                D[i, i, :] = c**2 * D_ii + s**2 * D_jj + 2 * c * s * D_ij
                D[j, j, :] = s**2 * D_ii + c**2 * D_jj - 2 * c * s * D_ij
                D[i, j, :] = c**2 * D_ij - s**2 * D_ji + c * s * (D_jj - D_ii)
                D[j, i, :] = D[i, j, :].copy()

                # Rotate row/column pairs (i,k) and (j,k) for k != i,j.
                # Done over all k at once and then repaired at k in {i, j}:
                # the four (i,i)/(j,j)/(i,j)/(j,i) entries were already
                # rotated above and must not be rotated twice. Equivalent to
                # the per-k loop this replaces, to ~1e-15.
                D_i_row = D[i, :, :].copy()
                D_j_row = D[j, :, :].copy()
                D[i, :, :] = c * D_i_row + s * D_j_row
                D[j, :, :] = c * D_j_row - s * D_i_row
                D[i, i, :] = c**2 * D_ii + s**2 * D_jj + 2 * c * s * D_ij
                D[j, j, :] = s**2 * D_ii + c**2 * D_jj - 2 * c * s * D_ij
                D[i, j, :] = c**2 * D_ij - s**2 * D_ji + c * s * (D_jj - D_ii)
                D[j, i, :] = D[i, j, :]
                D[:, i, :] = D[i, :, :]
                D[:, j, :] = D[j, :, :]

                # Track the global rotation.
                Ui = U[:, i].copy()
                Uj = U[:, j].copy()
                U[:, i] = c * Ui + s * Uj
                U[:, j] = c * Uj - s * Ui

        if max_angle < conv_tol:
            break

    return C @ U


# ---------------------------------------------------------------------------
# Pipek-Mezey (PM) localisation
# ---------------------------------------------------------------------------


def pipek_mezey_objective(
    C: np.ndarray,
    S: np.ndarray,
    atom_basis_map: np.ndarray,
) -> float:
    """Pipek-Mezey objective ``S_i S_A (Q_i^A)^2``.

    Q_i^A is the Mulliken population of orbital i on atom A.

    Parameters
    ----------
    C : ndarray, shape (nbf, norb)
    S : ndarray, shape (nbf, nbf)
    atom_basis_map : ndarray, shape (nbf, natom)

    Returns
    -------
    float
        The objective value.  Larger = fewer atoms per orbital.
    """
    CS = C.T @ S  # (norb, nbf)
    val = 0.0
    for a in range(atom_basis_map.shape[1]):
        mask = atom_basis_map[:, a] > 0.5
        if not np.any(mask):
            continue
        Q = np.einsum("im,mi->i", CS[:, mask], C[mask, :])
        val += float(np.sum(Q**2))
    return val


def pipek_mezey_localise(
    C: np.ndarray,
    S: np.ndarray,
    atom_basis_map: np.ndarray,
    max_iter: int = 200,
    conv_tol: float = 1e-8,
) -> np.ndarray:
    """Localise orbitals via the Pipek-Mezey criterion.

    Maximises Mulliken atomic-population sums using 2x2 Jacobi
    rotations. Produces orbitals localised on as few atoms as possible.

    Parameters
    ----------
    C : ndarray, shape (nbf, norb)
        MO coefficient matrix.
    S : ndarray, shape (nbf, nbf)
        AO overlap matrix.
    atom_basis_map : ndarray, shape (nbf, natom)
        Boolean matrix: ``atom_basis_map[mu, A] = 1`` if basis function
        mu belongs to atom A.
    max_iter : int
        Maximum sweeps. Default 200.
    conv_tol : float
        Convergence threshold on max rotation angle. Default 1e-8.

    Returns
    -------
    C_loc : ndarray, shape (nbf, norb)
        Localised MO coefficients.
    """
    norb = C.shape[1]
    natom = atom_basis_map.shape[1]
    U = np.eye(norb)

    # Working copies rotated in sync: Mulliken populations need both the
    # S-contracted rows (CS) and the raw coefficients (Cw).
    Cw = C.copy()
    CS = Cw.T @ S  # (norb, nbf)

    # Atom ownership as an index array so the per-atom populations become a
    # single segment sum instead of a Python loop over masks -- the same
    # change that took the IBO sweep from 24% of SCF wall time to 6%.
    # ``atom_basis_map`` assigns each AO to at most one atom; AOs owned by no
    # atom are dropped, matching the mask loop this replaces (an all-zero row
    # appeared in no mask).
    owned = atom_basis_map > 0.5
    has_owner = np.any(owned, axis=1)
    owner_index = np.argmax(owned, axis=1)
    # Every AO having an owner is the normal case; keeping it unmasked avoids
    # three fancy-index copies per orbital pair, which at these array sizes
    # costs more than the segment sum itself.
    all_owned = bool(np.all(has_owner))
    if not all_owned:
        owner_index = owner_index[has_owner]

    for iteration in range(max_iter):
        max_angle = 0.0
        for i in range(norb):
            for j in range(i + 1, norb):
                w_ii = CS[i, :] * Cw[:, i]
                w_jj = CS[j, :] * Cw[:, j]
                # Symmetrised Mulliken transition population.
                w_ij = 0.5 * (CS[i, :] * Cw[:, j] + CS[j, :] * Cw[:, i])
                if not all_owned:
                    w_ii = w_ii[has_owner]
                    w_jj = w_jj[has_owner]
                    w_ij = w_ij[has_owner]
                Q_ii = np.bincount(owner_index, weights=w_ii, minlength=natom)
                Q_jj = np.bincount(owner_index, weights=w_jj, minlength=natom)
                Q_ij = np.bincount(owner_index, weights=w_ij, minlength=natom)

                A_val = float(
                    np.sum(Q_ij**2 - 0.25 * (Q_ii - Q_jj) ** 2)
                )
                B_val = float(np.sum(Q_ij * (Q_ii - Q_jj)))

                if A_val * A_val + B_val * B_val < 1e-24:
                    continue

                theta = 0.25 * np.arctan2(B_val, -A_val)
                max_angle = max(max_angle, abs(theta))
                if abs(theta) < 1e-14:
                    continue
                c, s = np.cos(theta), np.sin(theta)

                # Track rotation in U.
                Ui = U[:, i].copy()
                Uj = U[:, j].copy()
                U[:, i] = c * Ui + s * Uj
                U[:, j] = c * Uj - s * Ui

                # Rotate CS rows and Cw columns in sync.
                CS_i = CS[i, :].copy()
                CS_j = CS[j, :].copy()
                CS[i, :] = c * CS_i + s * CS_j
                CS[j, :] = c * CS_j - s * CS_i
                Cw_i = Cw[:, i].copy()
                Cw_j = Cw[:, j].copy()
                Cw[:, i] = c * Cw_i + s * Cw_j
                Cw[:, j] = c * Cw_j - s * Cw_i

        if max_angle < conv_tol:
            break

    return C @ U


# ---------------------------------------------------------------------------
# Intrinsic bond orbitals (IBO)
# ---------------------------------------------------------------------------


def ibo_objective(
    C_iao: np.ndarray,
    atom_indices: np.ndarray,
    power: int = 4,
) -> float:
    """IBO localisation functional.

    Knizia eq 4:  ``L = sum_A sum_i [n_A(i)]^p``, with ``n_A(i)`` the number
    of electrons of orbital ``i`` on atom ``A`` measured in the IAO basis.
    Larger = fewer atoms per orbital.
    """
    n_atoms = int(atom_indices.max()) + 1
    idx = np.asarray(atom_indices, dtype=np.int64)
    total = 0.0
    for i in range(C_iao.shape[1]):
        column = C_iao[:, i]
        populations = np.bincount(idx, weights=column * column, minlength=n_atoms)
        total += float(np.sum(populations**power))
    return total


def ibo_localise(
    C_iao: np.ndarray,
    atom_indices: np.ndarray,
    power: int = 4,
    max_iter: int = 200,
    conv_tol: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Localise occupied orbitals by the IBO criterion.

    Operates **in the orthonormal IAO basis**, which is what makes IBO both
    cheaper and better behaved than Pipek-Mezey -- PM has to work in the full
    nonorthogonal AO basis (Knizia Appendix D).

    Parameters
    ----------
    C_iao : ndarray, shape (n_iao, n_occ)
        Occupied orbitals expressed in the orthonormal IAO basis, i.e.
        ``A^T S1 C_occ``.  Modified only through a local copy.
    atom_indices : ndarray, shape (n_iao,)
        IAO index -> 0-based atom index.
    power : int
        Localisation power ``p``.  4 (default) is the IBO criterion; 2
        reproduces the Pipek-Mezey functional evaluated over IAOs.  Knizia
        prefers 4 because "the former leads to discrete localizations in
        aromatic systems, while the second does not" -- in benzene the
        orbital rotation Hessian has a zero eigenvalue at p=2.
    max_iter : int
        Maximum Jacobi sweeps over all pairs.  Knizia reports convergence in
        5-10 sweeps.
    conv_tol : float
        Convergence threshold on the largest rotation angle in a sweep.

    Returns
    -------
    C_iao_loc : ndarray, shape (n_iao, n_occ)
        Localised orbitals, still in the IAO basis.  Recover coefficients in
        the AO basis with ``iaos @ C_iao_loc``.
    n_sweeps : int
        Sweeps performed.

    Notes
    -----
    **The published increments are typographically corrupted and were not
    transcribed.**  Knizia's Appendix D prints, for p=4,

        B_ij = B_ij + 4 Q_ij (Q_ij^3 - Q_ij^3)

    which is identically zero, and prints the p=2 A-increment as
    ``4Q_ij^2 - (Q_ii - Q_ij^2)`` where Pipek-Mezey requires
    ``(Q_ii - Q_jj)^2``; the subscripts are mangled throughout that block.
    The forms below were derived instead, and are pinned in
    ``tests/test_iao_ibo.py`` against a brute-force scan of ``L(phi)``:

    * p=2 is the exact maximiser (agreement to the scan's grid resolution).
    * p=4 is approximate by construction -- Knizia notes high-order terms in
      phi are neglected -- losing a median 1.7e-5 of the achievable gain per
      rotation on realistic blocks, never decreasing ``L``, and converging in
      the sweeps the paper describes.

    The rotation angle branch ``phi = 1/4 arctan2(B, -A)`` and the rotation
    convention are Knizia eq D1, and match the Boys/PM solvers above.
    """
    if power not in (2, 4):
        raise ValueError(f"IBO localisation power must be 2 or 4, got {power!r}")

    C = np.array(C_iao, dtype=np.float64, copy=True)
    n_occ = C.shape[1]
    n_atoms = int(atom_indices.max()) + 1
    # Atom-resolved populations come from a segment sum over the IAO axis.
    # ``bincount`` keeps this O(n_iao) per pair and, unlike carrying a
    # per-atom Q tensor the way the Boys solver above carries its dipole
    # matrices, it stays O(n_iao) in memory too -- a Q tensor would be
    # n_atoms x n_occ^2, i.e. cubic in system size.
    idx = np.asarray(atom_indices, dtype=np.int64)

    sweeps = 0
    for sweep in range(max_iter):
        sweeps = sweep + 1
        max_angle = 0.0
        for i in range(n_occ):
            for j in range(i + 1, n_occ):
                ci = C[:, i]
                cj = C[:, j]
                q_ii = np.bincount(idx, weights=ci * ci, minlength=n_atoms)
                q_jj = np.bincount(idx, weights=cj * cj, minlength=n_atoms)
                q_ij = np.bincount(idx, weights=ci * cj, minlength=n_atoms)

                if power == 2:
                    A_val = float(np.sum(4.0 * q_ij**2 - (q_ii - q_jj) ** 2))
                    B_val = float(np.sum(4.0 * q_ij * (q_ii - q_jj)))
                else:
                    A_val = float(
                        np.sum(
                            -(q_ii**4)
                            - q_jj**4
                            + 6.0 * (q_ii**2 + q_jj**2) * q_ij**2
                            + q_ii**3 * q_jj
                            + q_ii * q_jj**3
                        )
                    )
                    B_val = float(np.sum(4.0 * q_ij * (q_ii**3 - q_jj**3)))

                if A_val * A_val + B_val * B_val < 1e-24:
                    continue

                theta = 0.25 * np.arctan2(B_val, -A_val)
                max_angle = max(max_angle, abs(theta))
                if abs(theta) < 1e-14:
                    continue

                c, s = np.cos(theta), np.sin(theta)
                Ci = ci.copy()
                Cj = cj.copy()
                C[:, i] = c * Ci + s * Cj
                C[:, j] = -s * Ci + c * Cj

        if max_angle < conv_tol:
            break

    return C, sweeps
