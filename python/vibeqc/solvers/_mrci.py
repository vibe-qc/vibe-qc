"""Multireference configuration interaction (MRCI).

Uncontracted MR-CISD: all single and double excitations from every
determinant in the CAS reference space.  Includes the Davidson (+Q)
size-extensivity correction.

References
----------
R. J. Buenker, S. D. Peyerimhoff, Theor. Chim. Acta 35, 33 (1974).
S. R. Langhoff, E. R. Davidson, Int. J. Quantum Chem. 8, 61 (1974).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import eigh

from ._casci import CASCIResult, casci


@dataclass
class MRCIResult:
    """MRCI result.

    Attributes
    ----------
    e_total : float
        MRCI total energy (including nuclear repulsion + core).
    e_corr : float
        Correlation energy: E_MRCI - E_CASCI.
    e_total_q : float or None
        Total energy with Davidson +Q correction (None if not requested).
    ci_coeffs : np.ndarray
        MRCI CI coefficients.
    n_det : int
        Number of determinants in the MRCI space.
    n_ref : int
        Number of reference (CAS) determinants.
    ref_weight : float
        Sum of squared CI coefficients of reference determinants (used for +Q).
    """

    e_total: float
    e_corr: float
    e_total_q: Optional[float] = None
    ci_coeffs: Optional[np.ndarray] = None
    n_det: int = 0
    n_ref: int = 0
    ref_weight: float = 0.0


def _full_det_from_cas(
    a_ref: tuple[int, ...],
    b_ref: tuple[int, ...],
    n_core: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map CAS active-orbital indices to full-MO indices."""
    core_occ = tuple(range(n_core))
    a_full = tuple(sorted(core_occ + tuple(i + n_core for i in a_ref)))
    b_full = tuple(sorted(core_occ + tuple(i + n_core for i in b_ref)))
    return a_full, b_full


def _generate_singles_unrestricted(
    a_occ: tuple[int, ...],
    b_occ: tuple[int, ...],
    norb: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All single excitations (alpha or beta) from (a_occ, b_occ).

    Virtuals are resolved **per spin**: an alpha electron may move into any
    orbital not already alpha-occupied -- including one that is singly
    beta-occupied, which simply makes that orbital doubly occupied -- and
    symmetrically for beta.  For a closed-shell reference (``a_occ == b_occ``)
    the two spin-virtual sets coincide with the common unoccupied set, so this
    reduces to the usual closed-shell singles; the distinction only matters for
    open-shell references, where the old combined ``range(norb) - a - b`` set
    wrongly forbade an alpha electron from entering a beta-singly-occupied
    orbital.  Mirrors
    :func:`~vibeqc.solvers._determinant.generate_cisd_determinants`.
    """
    dets: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    a_set = set(a_occ)
    b_set = set(b_occ)
    a_vir = sorted(set(range(norb)) - a_set)
    b_vir = sorted(set(range(norb)) - b_set)

    # Alpha singles
    for i in a_set:
        new_a = list(a_occ)
        new_a.remove(i)
        for a in a_vir:
            dets.append((tuple(sorted(new_a + [a])), b_occ))
    # Beta singles
    for i in b_set:
        new_b = list(b_occ)
        new_b.remove(i)
        for a in b_vir:
            dets.append((a_occ, tuple(sorted(new_b + [a]))))
    return dets


def _generate_doubles_unrestricted(
    a_occ: tuple[int, ...],
    b_occ: tuple[int, ...],
    norb: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All double excitations (aa, bb, ab) from (a_occ, b_occ).

    Virtuals are resolved **per spin** (see
    :func:`_generate_singles_unrestricted`).  The same-spin (aa, bb) blocks
    promote two ``i<j`` occupied electrons of one spin into two ``a<b``
    orbitals of that spin's virtual set.  The opposite-spin (ab) block promotes
    one alpha and one beta electron **independently**: because the two target
    orbitals live in different spin sectors they are free to coincide
    (``a == b``).  That ``a == b`` case is the closed-shell HOMO^2->LUMO^2-type
    double -- one alpha and one beta electron in the *same* spatial virtual,
    which is a distinct, valid determinant and usually the single most
    important correlating configuration.  Excluding it (as a ``qa == pa``
    guard once did) silently undercounts the CI space and raises the CI energy.
    For a closed-shell reference both spin-virtual sets coincide, so the
    same-spin and opposite-spin blocks reduce to the standard closed-shell
    doubles.  Mirrors
    :func:`~vibeqc.solvers._determinant.generate_cisd_determinants`.
    """
    dets: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    a_list = sorted(a_occ)
    b_list = sorted(b_occ)
    a_set = set(a_occ)
    b_set = set(b_occ)
    a_vir = sorted(set(range(norb)) - a_set)
    b_vir = sorted(set(range(norb)) - b_set)
    nva = len(a_vir)
    nvb = len(b_vir)

    # Alpha-alpha doubles: i<j occupied alpha -> a<b alpha-virtual.
    for ia in range(len(a_list)):
        i = a_list[ia]
        for ja in range(ia + 1, len(a_list)):
            j = a_list[ja]
            for pa in range(nva):
                a = a_vir[pa]
                for qa in range(pa + 1, nva):
                    b = a_vir[qa]
                    new_a = sorted(a_set - {i, j} | {a, b})
                    dets.append((tuple(new_a), b_occ))

    # Beta-beta doubles: i<j occupied beta -> a<b beta-virtual.
    for ib in range(len(b_list)):
        i = b_list[ib]
        for jb in range(ib + 1, len(b_list)):
            j = b_list[jb]
            for pb in range(nvb):
                a = b_vir[pb]
                for qb in range(pb + 1, nvb):
                    b = b_vir[qb]
                    new_b = sorted(b_set - {i, j} | {a, b})
                    dets.append((a_occ, tuple(new_b)))

    # Alpha-beta doubles: one alpha (i -> a, alpha-virtual) and one beta
    # (j -> b, beta-virtual), independently.  a == b is allowed and is the
    # leading closed-shell double (see the doc-string above).
    for i in a_set:
        for a in a_vir:
            new_a = tuple(sorted(a_set - {i} | {a}))
            for j in b_set:
                for b in b_vir:
                    new_b = sorted(b_set - {j} | {b})
                    dets.append((new_a, tuple(new_b)))

    return dets


def _generate_mrci_space(
    ref_dets: list[tuple[tuple[int, ...], tuple[int, ...]]],
    norb: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All unique determinants in the MRCI space (refs + singles + doubles)."""
    space: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for a_ref, b_ref in ref_dets:
        space.add((a_ref, b_ref))
        for det in _generate_singles_unrestricted(a_ref, b_ref, norb):
            space.add(det)
        for det in _generate_doubles_unrestricted(a_ref, b_ref, norb):
            space.add(det)
    return sorted(space)


def mrci(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    n_active_elec: int,
    n_active_orb: int,
    n_core: int = 0,
    nuclear_repulsion: float = 0.0,
    ms2: Optional[int] = None,
    *,
    do_q_correction: bool = True,
    max_det: Optional[int] = 50_000,
) -> MRCIResult:
    """Run uncontracted MR-CISD on a CAS reference.

    Parameters
    ----------
    h1e_mo : (norb, norb) ndarray
        Full-MO one-electron Hamiltonian.
    h2e_mo : (norb, norb, norb, norb) ndarray
        Full-MO two-electron integrals (physicist's notation).
    n_active_elec, n_active_orb : int
        Active-space size for the CASCI reference.
    n_core : int
        Number of doubly-occupied inactive (core) orbitals.
    nuclear_repulsion : float
        Nuclear repulsion energy.
    ms2 : int, optional
        2*S_z of the active electrons.
    do_q_correction : bool
        Apply Davidson +Q size-extensivity correction.
    max_det : int or None
        Maximum MRCI determinant count; raises ValueError if exceeded.

    Returns
    -------
    MRCIResult
    """
    norb = h1e_mo.shape[0]
    if ms2 is None:
        ms2 = n_active_elec % 2

    # Step 1: CASCI reference in the full MO basis.
    cas = casci(
        h1e_mo,
        h2e_mo,
        n_active_elec=n_active_elec,
        n_active_orb=n_active_orb,
        n_core=n_core,
        nuclear_repulsion=nuclear_repulsion,
        ms2=ms2,
    )

    # Step 2: Map reference determinants to full-MO indices.
    ref_dets_full = [_full_det_from_cas(a, b, n_core) for a, b in cas.determinants]

    # Step 3: Generate MRCI determinant space.
    mrci_dets = _generate_mrci_space(ref_dets_full, norb)
    n_det = len(mrci_dets)
    if max_det is not None and n_det > max_det:
        raise ValueError(
            f"MRCI space has {n_det} determinants (limit {max_det}). "
            "Use a smaller active space or increase max_det."
        )

    # Step 4: Build MRCI Hamiltonian.
    from ._slater_condon import build_hamiltonian_matrix_unrestricted

    H_mrci = build_hamiltonian_matrix_unrestricted(mrci_dets, h1e_mo, h2e_mo)

    # Step 5: Diagonalize.
    eigvals, eigvecs = eigh(H_mrci)
    e_total = float(eigvals[0]) + nuclear_repulsion
    ci = eigvecs[:, 0]

    # Step 6: Reference weight for +Q correction.
    det_index = {d: i for i, d in enumerate(mrci_dets)}
    ref_weight = float(
        sum(ci[det_index[d]] ** 2 for d in ref_dets_full if d in det_index)
    )

    # Step 7: Davidson +Q correction.
    e_total_q = None
    if do_q_correction:
        e_total_q = cas.e_total + (e_total - cas.e_total) / max(ref_weight, 0.01)

    return MRCIResult(
        e_total=e_total,
        e_corr=e_total - cas.e_total,
        e_total_q=e_total_q,
        ci_coeffs=ci,
        n_det=n_det,
        n_ref=len(ref_dets_full),
        ref_weight=ref_weight,
    )
