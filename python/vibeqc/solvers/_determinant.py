"""Determinant and configuration-state-function data structures.

Representation of Slater determinants as tuples of occupied orbital
indices, with helper functions for building the determinant space,
computing excitation ranks, and generating connected determinants.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

# Type alias: a determinant is a sorted tuple of integer orbital indices.
Det = tuple[int, ...]

# (alpha_occ, beta_occ) for unrestricted determinants.
SpinDet = tuple[Det, Det]


def determinant_string(occ: Det) -> str:
    """Render a determinant as a binary occupation string (0 = empty, 1 = occ)."""
    norb = max(occ) + 1 if occ else 0
    s = ["0"] * norb
    for i in occ:
        s[i] = "1"
    return "".join(s)


def excitation_rank(occ_a: Det, occ_b: Det) -> int:
    """Number of orbital replacements to go from ``occ_a`` to ``occ_b``.

    Both must have the same length.
    """
    if len(occ_a) != len(occ_b):
        raise ValueError("Determinants must have the same electron count")
    return len(set(occ_a) - set(occ_b))


def is_connected(occ_a: Det, occ_b: Det, max_rank: int = 2) -> bool:
    """True iff ``occ_a`` and ``occ_b`` differ by <= ``max_rank`` excitations.

    In standard ab initio Hamiltonians, only single and double excitations
    have non-zero matrix elements.
    """
    return excitation_rank(occ_a, occ_b) <= max_rank


def generate_determinants(
    norb: int,
    nalpha: int,
    nbeta: int,
) -> list[SpinDet]:
    """Generate all determinants for ``nalpha`` a + ``nbeta`` b electrons in ``norb`` orbitals.

    Returns a list of ``(alpha_occ, beta_occ)`` tuples in lexicographic order.
    This is the complete FCI determinant space -- use with caution (``C(norb, nalpha) x C(norb, nbeta)``).
    """
    from itertools import combinations

    alpha_dets = [tuple(c) for c in combinations(range(norb), nalpha)]
    beta_dets = [tuple(c) for c in combinations(range(norb), nbeta)]
    return [(a, b) for a in alpha_dets for b in beta_dets]


def generate_closed_shell_determinants(
    norb: int,
    nocc: int,
) -> list[Det]:
    """Generate all closed-shell (spin-restricted) determinants.

    Each determinant is represented as a tuple of spatial-orbital indices
    that are doubly occupied.
    """
    from itertools import combinations

    return [tuple(c) for c in combinations(range(norb), nocc)]


def generate_singles(
    ref: Det,
    norb: int,
) -> list[Det]:
    """All single excitations from ``ref`` (one electron moved to a virtual)."""
    dets: list[Det] = []
    occ_set = set(ref)
    vir_set = set(range(norb)) - occ_set
    for i in occ_set:
        new_occ = list(ref)
        new_occ.remove(i)
        for a in sorted(vir_set):
            dets.append(tuple(sorted(new_occ + [a])))
    return dets


def generate_doubles(
    ref: Det,
    norb: int,
) -> list[Det]:
    """All double excitations from ``ref`` (two electrons moved)."""
    dets: list[Det] = []
    occ_set = set(ref)
    occ_list = sorted(occ_set)
    vir_set = set(range(norb)) - occ_set
    vir_list = sorted(vir_set)

    for idx_i, i in enumerate(occ_list):
        for idx_j in range(idx_i + 1, len(occ_list)):
            j = occ_list[idx_j]
            for idx_a, a in enumerate(vir_list):
                for idx_b in range(idx_a + 1, len(vir_list)):
                    b = vir_list[idx_b]
                    new_occ = sorted(set(ref) - {i, j} | {a, b})
                    dets.append(tuple(new_occ))
    return dets


def generate_cisd_determinants(
    norb: int,
    nalpha: int,
    nbeta: int,
    *,
    max_excitation: int = 2,
) -> list[SpinDet]:
    """Truncated-CI determinant space: all ``(alpha_occ, beta_occ)`` within
    ``max_excitation`` substitutions of the aufbau reference.

    ``max_excitation=1`` gives the CIS space (reference + singles);
    ``max_excitation=2`` gives the CISD space (+ doubles).  Each determinant is
    a :data:`SpinDet` ``(alpha_occ, beta_occ)`` of sorted spatial-orbital
    indices, so the result feeds
    :func:`~vibeqc.solvers._slater_condon.build_hamiltonian_matrix_unrestricted`
    directly.  The list is sorted and free of duplicates.

    The excitations are enumerated against the **aufbau** reference
    ``alpha_occ = range(nalpha)``, ``beta_occ = range(nbeta)``:

    * singles -- one a (or one b) electron promoted to any orbital outside that
      spin's occupied set;
    * doubles -- aa and bb (two same-spin electrons, ``i<j`` -> ``a<b``) **and**
      ab (one a + one b electron promoted independently).  The ab case
      deliberately allows the two electrons to land in the **same** spatial
      virtual (``a == b``): they occupy different spin-orbitals, so
      ``HOMO^2->LUMO^2``-type closed-shell doubles -- usually the leading
      correlating configuration -- are included.

    A virtual for the a channel is any orbital not in ``alpha_occ`` (it may be
    b-occupied -- exciting an a electron there makes the orbital doubly occupied);
    the b channel is symmetric.  For ``nalpha == nbeta`` (closed shell) this is
    the standard single-reference CISD / CIS space.
    """
    if max_excitation not in (1, 2):
        raise ValueError(
            f"max_excitation must be 1 (CIS) or 2 (CISD), got {max_excitation}"
        )
    a_ref = tuple(range(nalpha))
    b_ref = tuple(range(nbeta))
    a_vir = [p for p in range(norb) if p not in set(a_ref)]
    b_vir = [p for p in range(norb) if p not in set(b_ref)]
    space: set[SpinDet] = {(a_ref, b_ref)}

    def _sub1(occ: Det, i: int, a: int) -> Det:
        return tuple(sorted(set(occ) - {i} | {a}))

    # Singles (a then b).
    for i in a_ref:
        for a in a_vir:
            space.add((_sub1(a_ref, i, a), b_ref))
    for i in b_ref:
        for a in b_vir:
            space.add((a_ref, _sub1(b_ref, i, a)))

    if max_excitation >= 2:
        # Same-spin doubles: i<j occupied, a<b virtual.
        def _same_spin(occ: Det, vir: list[int]) -> list[Det]:
            out: list[Det] = []
            for ii in range(len(occ)):
                for jj in range(ii + 1, len(occ)):
                    for pa in range(len(vir)):
                        for pb in range(pa + 1, len(vir)):
                            out.append(tuple(sorted(
                                set(occ) - {occ[ii], occ[jj]}
                                | {vir[pa], vir[pb]})))
            return out

        for d in _same_spin(a_ref, a_vir):
            space.add((d, b_ref))
        for d in _same_spin(b_ref, b_vir):
            space.add((a_ref, d))
        # Opposite-spin doubles: independent a and b promotions (a == b allowed).
        for i in a_ref:
            for a in a_vir:
                da = _sub1(a_ref, i, a)
                for j in b_ref:
                    for b in b_vir:
                        space.add((da, _sub1(b_ref, j, b)))

    return sorted(space)


def reference_determinant(
    nalpha: int,
    nbeta: int,
    *,
    spin_restricted: bool = True,
) -> SpinDet:
    """The aufbau reference determinant: lowest-energy orbitals doubly occupied.

    For spin-restricted: ``(0,1,...,nalpha-1), (0,1,...,nbeta-1)``.
    """
    if spin_restricted and nalpha == nbeta:
        occ = tuple(range(nalpha))
        return (occ, occ)
    else:
        return (tuple(range(nalpha)), tuple(range(nbeta)))
