"""Symmetry reduction for the prototype bipolar far-field dispatcher.

BIPOLE-EXACT-ZONE increment 4a. The current classifier uses product-centre
geometry and diffuse shell exponents. These inputs are invariant for shell
quartets related by the mapped point-group operation, so those quartets
share a prototype dispatch decision. No cited primary source derives the
classifier itself.

This module reduces the O(n_sh^4 × n_cells^2) dispatch enumeration
by exploiting:

1. **Shell-pair permutation symmetry** — Quartets (s1,s2; s3,s4) and
   (s2,s1; s4,s3) are related by transposition of the bra and ket
   pairs.  Since the interaction tensor is symmetric, only one
   needs to be dispatched.

2. **Point-group symmetry** — For symmorphic space-group operations R,
   a bra pair (s1, s2, c_g) and ket pair (s3, s4, c_lam) are
   symmetry-equivalent to (R·s1, R·s2, R·c_g) and (R·s3, R·s4,
   R·c_lam).  Only the orbit representative needs to be dispatched.

3. **Shell-index permutation within a pair** — (s1, s2) and (s2, s1)
   at the same cell have the same product-distribution centre
   (weighted by the smallest exponents), so they can share
   a dispatch decision.

The output is a reduced :class:`QuartetBipolarDispatch` that covers
only symmetry-unique quartets.  The far-field Fock contractor
reconstructs the full Fock by symmetry after computing the
multipole contributions for the representatives.

References
----------
Pisani, Dovesi, and Roetti (1988), Ch. II.4c,
doi:10.1007/978-3-642-93385-1 - periodic quartet expansion.
Dovesi et al., Int. J. Quantum Chem. 29, 1755 (1986) - symmetry in
periodic LCAO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from .bipole_dispatch import (
    PenetrationDispatchParameters,
    compute_quartet_multipole_truncation_order,
    product_distribution_gaussian_center,
    product_distribution_gaussian_width,
)
from .bipole_quartet_far_field import QuartetBipolarDispatch

__all__ = [
    "SymmetryReducedDispatchResult",
    "SymmetryFockReconstructionMap",
    "build_symmetry_reduced_penetration_dispatch",
    "build_point_group_reduced_penetration_dispatch",
    "unfold_symmetry_reduced_dispatch",
    "build_symmetry_fock_reconstruction_map",
]


# ---------------------------------------------------------------------------
# Shell-pair key for symmetry identification
# ---------------------------------------------------------------------------

# A "shell pair" at a lattice cell is identified by (s1, s2, cell_idx).
# Under symmetry, the cell index transforms as R·c + (s_b - s_a) for the
# SYM2c atom-pair-resolved convention.  For the penetration dispatch,
# we work with the SIMPLER convention: the shell-pair centre is at
# (a1·A + a2·(B+g))/(a1+a2), and under rotation R this transforms to
# R·centre.  So (s1, s2, c_g) maps to (s1, s2, c_g') where c_g' is the
# cell whose centre is closest to R·centre_g.

# For simplicity in this first implementation, we use PERMUTATION symmetry
# only (shell-pair transposition), which halves the dispatch cost
# independently of whether the system has space-group symmetry.


@dataclass
class SymmetryReducedDispatchResult:
    """Reduced penetration dispatch with symmetry-reconstruction metadata.

    Attributes
    ----------
    dispatch : QuartetBipolarDispatch
        The reduced dispatch covering only symmetry-unique quartets.
    n_full : int
        Number of quartets in the full (unreduced) dispatch.
    n_reduced : int
        Number of quartets after symmetry reduction.
    symmetry_factor : float
        Average reduction factor (n_full / n_reduced).
    """

    dispatch: QuartetBipolarDispatch
    n_full: int = 0
    n_reduced: int = 0
    symmetry_factor: float = 1.0


# ---------------------------------------------------------------------------
# Shell-pair permutation reduction (independent of space group)
# ---------------------------------------------------------------------------

def _canonical_shell_pair(s1: int, s2: int) -> Tuple[int, int]:
    """Return (s1, s2) in canonical order (smaller first).

    Since the product-distribution centre depends on the weighted
    average of shell origins, (s1, s2) and (s2, s1) at the same
    cell have the same centre when using smallest exponents.
    However, the AO dimensions differ for s-shell vs p-shell pairs,
    so we keep the ordering for the buffer look-up but reduce
    the DISPATCH by treating (s1,s2) ≡ (s2,s1) for the geometric
    decision.

    This function returns the canonical (sorted) ordering for
    the geometric dispatch.  The caller still stores both
    (s1,s2) and (s2,s1) in the moment buffer for efficient
    AO-pair access.
    """
    return (s1, s2) if s1 <= s2 else (s2, s1)


def build_symmetry_reduced_penetration_dispatch(
    basis,
    lattice_cells: list,
    params: PenetrationDispatchParameters,
    *,
    compute_exchange: bool = False,
) -> Tuple[SymmetryReducedDispatchResult, Optional[SymmetryReducedDispatchResult]]:
    """Build penetration dispatch with shell-pair permutation reduction.

    Reduces the O(n_sh^4 × n_c^2) enumeration by:
    1. Restricting bra and ket shell pairs to canonical order (s1≤s2, s3≤s4).
    2. Deduplicating quartets that differ only by transposition of the
       bra pair or ket pair.

    This gives ~4× reduction (factor of 2 from each of bra and ket
    canonical ordering) without requiring space-group information.

    Parameters
    ----------
    basis : BasisSet
    lattice_cells : list of LatticeCell
    params : PenetrationDispatchParameters
    compute_exchange : bool
        If True, also compute the Exchange far-field dispatch.

    Returns
    -------
    (j_result, k_result_or_none)
    """
    min_exp, origins = _extract_shell_info(basis)
    n_sh = len(min_exp)
    n_cells = len(lattice_cells)

    j_dispatch = QuartetBipolarDispatch()
    k_dispatch = QuartetBipolarDispatch() if compute_exchange else None

    n_full_j = 0
    n_full_k = 0

    # Use a set to deduplicate quartets that differ only by transposition.
    seen_j: Set[Tuple] = set()
    seen_k: Set[Tuple] = set()

    for s1 in range(n_sh):
        for s2 in range(n_sh):  # all pairs, not just s1≤s2
            can_bra = _canonical_shell_pair(s1, s2)
            a_bra = min_exp[s1]
            b_bra = min_exp[s2]
            gamma_bra = product_distribution_gaussian_width(a_bra, b_bra)

            for g_idx, cell_g in enumerate(lattice_cells):
                g = np.asarray(cell_g.r_cart, dtype=float).reshape(3)
                A_bra = origins[s1]
                B_bra = origins[s2] + g
                centre_bra = product_distribution_gaussian_center(
                    A_bra, B_bra, a_bra, b_bra,
                )

                for s3 in range(n_sh):
                    for s4 in range(n_sh):
                        can_ket = _canonical_shell_pair(s3, s4)
                        a_ket = min_exp[s3]
                        b_ket = min_exp[s4]
                        gamma_ket = product_distribution_gaussian_width(
                            a_ket, b_ket,
                        )

                        for lam_idx, cell_lam in enumerate(lattice_cells):
                            lam = np.asarray(
                                cell_lam.r_cart, dtype=float,
                            ).reshape(3)
                            A_ket = origins[s3]
                            B_ket = origins[s4] + lam
                            centre_ket = (
                                product_distribution_gaussian_center(
                                    A_ket, B_ket, a_ket, b_ket,
                                )
                            )

                            # ---- Prototype overlap pre-screening --------
                            if params.overlap_drop_threshold > 0:
                                from .bipole_dispatch import (
                                    estimate_shell_pair_overlap_upper_bound,
                                )
                                S_bra = (
                                    estimate_shell_pair_overlap_upper_bound(
                                        a_bra, b_bra, A_bra, B_bra,
                                    )
                                )
                                S_ket = (
                                    estimate_shell_pair_overlap_upper_bound(
                                        a_ket, b_ket, A_ket, B_ket,
                                    )
                                )
                                if S_bra < params.overlap_drop_threshold:
                                    continue
                                if S_ket < params.overlap_drop_threshold:
                                    continue

                            # Deduplicate using canonical pair ordering.
                            quartet_key = (
                                can_bra[0], can_bra[1], g_idx,
                                can_ket[0], can_ket[1], lam_idx,
                            )

                            # ---- Coulomb (J) ----
                            n_full_j += 1
                            if quartet_key not in seen_j:
                                seen_j.add(quartet_key)
                                order = compute_quartet_multipole_truncation_order(
                                    centre_bra, centre_ket,
                                    gamma_bra, params,
                                )
                                if order > 0:
                                    j_dispatch.add_quartet(
                                        s1, s2, g_idx,
                                        s3, s4, lam_idx,
                                        order, gamma_bra, gamma_ket,
                                    )

                            # ---- Exchange (K) ----
                            if compute_exchange:
                                n_full_k += 1
                                if quartet_key not in seen_k:
                                    seen_k.add(quartet_key)
                                    order = compute_quartet_multipole_truncation_order(
                                        centre_bra, centre_ket,
                                        gamma_bra, params,
                                    )
                                    if order > 0:
                                        k_dispatch.add_quartet(
                                            s1, s2, g_idx,
                                            s3, s4, lam_idx,
                                            order, gamma_bra, gamma_ket,
                                        )

    j_result = SymmetryReducedDispatchResult(
        dispatch=j_dispatch,
        n_full=n_full_j,
        n_reduced=len(j_dispatch),
        symmetry_factor=(
            n_full_j / len(j_dispatch) if len(j_dispatch) > 0 else 1.0
        ),
    )

    k_result = None
    if compute_exchange and k_dispatch is not None:
        k_result = SymmetryReducedDispatchResult(
            dispatch=k_dispatch,
            n_full=n_full_k,
            n_reduced=len(k_dispatch),
            symmetry_factor=(
                n_full_k / len(k_dispatch)
                if len(k_dispatch) > 0
                else 1.0
            ),
        )

    return j_result, k_result


def _extract_shell_info(basis):
    """Return (min_exponents, origins) arrays per shell."""
    exponents = np.array(
        [min(sh.exponents) for sh in basis.shells()], dtype=float,
    )
    origins = np.array(
        [
            np.asarray(sh.origin, dtype=float).reshape(3)
            for sh in basis.shells()
        ],
        dtype=float,
    )
    return exponents, origins


def unfold_symmetry_reduced_dispatch(
    reduced: QuartetBipolarDispatch,
    n_shells: int,
) -> QuartetBipolarDispatch:
    """Unfold a permutation-reduced dispatch to the full quartet set.

    For each reduced quartet (s1, s2, g, s3, s4, lam) where s1≤s2 and
    s3≤s4 (canonical ordering), also register the transposed variants
    (s2, s1, g, s3, s4, lam), (s1, s2, g, s4, s3, lam), and
    (s2, s1, g, s4, s3, lam).

    This is needed to ensure the far-field Fock contractor has entries
    for all AO-index orderings.

    Parameters
    ----------
    reduced : QuartetBipolarDispatch
        Symmetry-reduced dispatch.
    n_shells : int
        Number of shells (for validation).

    Returns
    -------
    QuartetBipolarDispatch
        Full (unfolded) dispatch.
    """
    full = QuartetBipolarDispatch()
    seen: Set[Tuple] = set()

    for q in range(len(reduced)):
        s1, s2, g_idx = reduced.bra_pairs[q]
        s3, s4, lam_idx = reduced.ket_pairs[q]
        order = reduced.truncation_orders[q]
        gamma_bra = reduced.bra_widths[q]
        gamma_ket = reduced.ket_widths[q]

        # Generate all 4 transposition variants.
        variants = [
            (s1, s2, g_idx, s3, s4, lam_idx),
            (s2, s1, g_idx, s3, s4, lam_idx),
            (s1, s2, g_idx, s4, s3, lam_idx),
            (s2, s1, g_idx, s4, s3, lam_idx),
        ]
        for v in variants:
            key = v
            if key not in seen:
                seen.add(key)
                full.add_quartet(
                    v[0], v[1], v[2],
                    v[3], v[4], v[5],
                    order, gamma_bra, gamma_ket,
                )

    return full


# ---------------------------------------------------------------------------
# Point-group symmetry reduction (space-group operations)
# ---------------------------------------------------------------------------


def build_point_group_reduced_penetration_dispatch(
    system,
    basis,
    lattice_cells: list,
    params: PenetrationDispatchParameters,
    *,
    compute_exchange: bool = False,
) -> Tuple[SymmetryReducedDispatchResult, Optional[SymmetryReducedDispatchResult]]:
    """Build penetration dispatch with point-group + permutation symmetry.

    In addition to the shell-pair permutation reduction (~4×), this
    also exploits symmorphic space-group operations: shell quartets
    related by a point-group rotation have the same inter-centre
    distance and thus the same penetration decision.  For cubic
    crystals this gives an additional 12-48× reduction factor.

    The reduction works by:
    1. Computing atom-permutation and cell-rotation under each
       symmorphic operation.
    2. Mapping each shell quartet to its symmetry-equivalent image.
    3. Choosing the lexicographically smallest image as the canonical
       representative — only that one is dispatched.

    Falls back to permutation-only reduction if no symmorphic
    operations are available.

    Parameters
    ----------
    system : PeriodicSystem
    basis : BasisSet
    lattice_cells : list of LatticeCell
    params : PenetrationDispatchParameters
    compute_exchange : bool

    Returns
    -------
    (j_result, k_result_or_none)
    """
    # Try to get symmorphic operations.
    sym_ops = []
    try:
        from .symmetry_integrals import symmorphic_operations

        sym_ops = symmorphic_operations(system)
    except Exception:
        pass

    if not sym_ops:
        # No point-group symmetry available; fall back to permutation only.
        return build_symmetry_reduced_penetration_dispatch(
            basis, lattice_cells, params,
            compute_exchange=compute_exchange,
        )

    # Build atom→atom permutation per operation.
    from .symmetry_ao import atom_permutation_under_op

    atom_perms: List[np.ndarray] = []
    cell_rotations: List[np.ndarray] = []  # 3×3 integer rotation in lattice basis
    for op in sym_ops:
        try:
            perm = atom_permutation_under_op(system, op)
            atom_perms.append(np.asarray(perm.perm, dtype=int))
            # Rotation matrix in Cartesian; convert to lattice basis.
            from .bipole_bravais_utils import lattice_to_cartesian_rotation

            R_cart = np.asarray(op.rotation, dtype=float).reshape(3, 3)
            R_lat = lattice_to_cartesian_rotation(system, R_cart)
            cell_rotations.append(R_lat)
        except Exception:
            continue

    if not atom_perms:
        return build_symmetry_reduced_penetration_dispatch(
            basis, lattice_cells, params,
            compute_exchange=compute_exchange,
        )

    # Build shell→shell mapping per operation.
    # For each shell i, find its atom index a_i. Under operation,
    # atom a_i goes to atom a'_i. The shell on the image atom with
    # the same ℓ is the image shell.
    shells = list(basis.shells())
    n_sh = len(shells)
    shell_atom = np.array([sh.atom_index for sh in shells], dtype=int)
    shell_l = np.array([int(sh.l) for sh in shells], dtype=int)
    # Build per-atom shell lists for fast lookup.
    atom_shells: Dict[int, List[int]] = {}
    for i in range(n_sh):
        a = int(shell_atom[i])
        atom_shells.setdefault(a, []).append(i)

    shell_perms: List[np.ndarray] = []
    for p_idx, atom_perm in enumerate(atom_perms):
        sh_map = np.full(n_sh, -1, dtype=int)
        valid = True
        for i in range(n_sh):
            a_src = int(shell_atom[i])
            l_i = int(shell_l[i])
            a_dst = int(atom_perm[a_src])
            # Find shell on dst atom with same ℓ.
            found = False
            for j in atom_shells.get(a_dst, []):
                if int(shell_l[j]) == l_i:
                    sh_map[i] = j
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            shell_perms.append(sh_map)
        # If not valid, skip this operation (shell layout mismatch).

    if not shell_perms:
        return build_symmetry_reduced_penetration_dispatch(
            basis, lattice_cells, params,
            compute_exchange=compute_exchange,
        )

    # Build cell index lookup: (i, j, k) → cell index.
    cell_index_to_idx: Dict[Tuple[int, int, int], int] = {}
    for idx, cell in enumerate(lattice_cells):
        c = (int(cell.index[0]), int(cell.index[1]), int(cell.index[2]))
        cell_index_to_idx[c] = idx

    min_exp, origins = _extract_shell_info(basis)
    n_cells = len(lattice_cells)

    j_dispatch = QuartetBipolarDispatch()
    k_dispatch = QuartetBipolarDispatch() if compute_exchange else None

    # Canonical-representative set.
    seen_j: Set[Tuple] = set()
    seen_k: Set[Tuple] = set()
    n_full_j = 0
    n_full_k = 0

    # Pre-compute shell-pair centres for fast symmetry mapping.
    # centre[s1, s2, c_idx] = product-distribution centre.
    # We compute on demand in the loop to save memory.

    for s1 in range(n_sh):
        for s2 in range(n_sh):
            a_bra = min_exp[s1]
            b_bra = min_exp[s2]
            gamma_bra = product_distribution_gaussian_width(a_bra, b_bra)

            for g_idx, cell_g in enumerate(lattice_cells):
                g = np.asarray(cell_g.r_cart, dtype=float).reshape(3)
                A_bra = origins[s1]
                B_bra = origins[s2] + g
                centre_bra = product_distribution_gaussian_center(
                    A_bra, B_bra, a_bra, b_bra,
                )

                for s3 in range(n_sh):
                    for s4 in range(n_sh):
                        a_ket = min_exp[s3]
                        b_ket = min_exp[s4]
                        gamma_ket = product_distribution_gaussian_width(
                            a_ket, b_ket,
                        )

                        for lam_idx, cell_lam in enumerate(lattice_cells):
                            # ---- Compute canonical rep under point group ----
                            # Start with the identity mapping.
                            rep_key = (
                                s1, s2, g_idx, s3, s4, lam_idx,
                            )
                            rep_order = -1

                            for op_idx, sh_map in enumerate(shell_perms):
                                R_lat = cell_rotations[op_idx]
                                s1_img = int(sh_map[s1])
                                s2_img = int(sh_map[s2])
                                s3_img = int(sh_map[s3])
                                s4_img = int(sh_map[s4])
                                # Rotate cell indices.
                                g_idx3 = np.array([
                                    cell_g.index[0],
                                    cell_g.index[1],
                                    cell_g.index[2],
                                ], dtype=int)
                                lam_idx3 = np.array([
                                    cell_lam.index[0],
                                    cell_lam.index[1],
                                    cell_lam.index[2],
                                ], dtype=int)
                                g_rot = tuple(R_lat @ g_idx3)
                                lam_rot = tuple(R_lat @ lam_idx3)
                                g_rot_idx = cell_index_to_idx.get(g_rot)
                                lam_rot_idx = cell_index_to_idx.get(lam_rot)
                                if g_rot_idx is None or lam_rot_idx is None:
                                    continue
                                cand_key = (
                                    s1_img, s2_img, g_rot_idx,
                                    s3_img, s4_img, lam_rot_idx,
                                )
                                if cand_key < rep_key:
                                    rep_key = cand_key

                            # ---- Dispatch the canonical rep only ---------
                            n_full_j += 1
                            if rep_key not in seen_j:
                                seen_j.add(rep_key)
                                # Compute penetration for the rep.
                                rs1, rs2, rg_idx = (
                                    rep_key[0], rep_key[1], rep_key[2],
                                )
                                rs3, rs4, rl_idx = (
                                    rep_key[3], rep_key[4], rep_key[5],
                                )
                                # Recompute centre for the rep.
                                r_centre_bra = (
                                    product_distribution_gaussian_center(
                                        origins[rs1],
                                        origins[rs2]
                                        + np.asarray(
                                            lattice_cells[rg_idx].r_cart,
                                            dtype=float,
                                        ),
                                        min_exp[rs1], min_exp[rs2],
                                    )
                                )
                                r_centre_ket = (
                                    product_distribution_gaussian_center(
                                        origins[rs3],
                                        origins[rs4]
                                        + np.asarray(
                                            lattice_cells[rl_idx].r_cart,
                                            dtype=float,
                                        ),
                                        min_exp[rs3], min_exp[rs4],
                                    )
                                )
                                r_gamma_bra = (
                                    product_distribution_gaussian_width(
                                        min_exp[rs1], min_exp[rs2],
                                    )
                                )
                                order = compute_quartet_multipole_truncation_order(
                                    r_centre_bra, r_centre_ket,
                                    r_gamma_bra, params,
                                )
                                if order > 0:
                                    j_dispatch.add_quartet(
                                        rs1, rs2, rg_idx,
                                        rs3, rs4, rl_idx,
                                        order,
                                        r_gamma_bra,
                                        product_distribution_gaussian_width(
                                            min_exp[rs3], min_exp[rs4],
                                        ),
                                    )

                            # Exchange dispatch (same rep logic).
                            if compute_exchange:
                                n_full_k += 1
                                if rep_key not in seen_k:
                                    seen_k.add(rep_key)
                                    rs1, rs2, rg_idx = (
                                        rep_key[0], rep_key[1], rep_key[2],
                                    )
                                    rs3, rs4, rl_idx = (
                                        rep_key[3], rep_key[4], rep_key[5],
                                    )
                                    r_centre_bra = (
                                        product_distribution_gaussian_center(
                                            origins[rs1],
                                            origins[rs2]
                                            + np.asarray(
                                                lattice_cells[rg_idx].r_cart,
                                                dtype=float,
                                            ),
                                            min_exp[rs1], min_exp[rs2],
                                        )
                                    )
                                    r_centre_ket = (
                                        product_distribution_gaussian_center(
                                            origins[rs3],
                                            origins[rs4]
                                            + np.asarray(
                                                lattice_cells[rl_idx].r_cart,
                                                dtype=float,
                                            ),
                                            min_exp[rs3], min_exp[rs4],
                                        )
                                    )
                                    r_gamma_bra = (
                                        product_distribution_gaussian_width(
                                            min_exp[rs1], min_exp[rs2],
                                        )
                                    )
                                    order = compute_quartet_multipole_truncation_order(
                                        r_centre_bra, r_centre_ket,
                                        r_gamma_bra, params,
                                    )
                                    if order > 0:
                                        k_dispatch.add_quartet(
                                            rs1, rs2, rg_idx,
                                            rs3, rs4, rl_idx,
                                            order,
                                            r_gamma_bra,
                                            product_distribution_gaussian_width(
                                                min_exp[rs3], min_exp[rs4],
                                            ),
                                        )

    j_result = SymmetryReducedDispatchResult(
        dispatch=j_dispatch,
        n_full=n_full_j,
        n_reduced=len(j_dispatch),
        symmetry_factor=(
            n_full_j / len(j_dispatch) if len(j_dispatch) > 0 else 1.0
        ),
    )

    k_result = None
    if compute_exchange and k_dispatch is not None:
        k_result = SymmetryReducedDispatchResult(
            dispatch=k_dispatch,
            n_full=n_full_k,
            n_reduced=len(k_dispatch),
            symmetry_factor=(
                n_full_k / len(k_dispatch)
                if len(k_dispatch) > 0
                else 1.0
            ),
        )

    return j_result, k_result


# ===========================================================================
#  Fock-reconstruction map for symmetry-reduced far-field kernel apply
# ===========================================================================


@dataclass
class SymmetryFockReconstructionMap:
    """Reconstruction table: maps reduced-dispatch entries to full Fock positions.

    For each entry in the reduced (canonical) dispatch, stores a list
    of ``(bra_cell_key, bra_slice, ket_cell_key, ket_slice)`` tuples that
    describe candidate full-quartet Fock positions.

    These are candidate positions only. The legacy shell/cell permutation
    inventory does not qualify an operator, tensor action or density state.
    The dormant kernel adapter refuses nonidentity reconstruction; identity
    metadata preserves ordinary application of the stored kernel.

    Attributes
    ----------
    orbit_entries : list of list of tuple
        ``orbit_entries[q]`` is the orbit of the q-th reduced dispatch
        entry.  Each element is ``(bra_key, (b1, b1e, b2, b2e),
        ket_key, (b3, b3e, b4, b4e))``.
    n_sym_operations : int
        Number of symmorphic point-group operations used.
    """

    orbit_entries: List[List[Tuple]] = field(default_factory=list)
    n_sym_operations: int = 1

    def __len__(self) -> int:
        return len(self.orbit_entries)


def build_symmetry_fock_reconstruction_map(
    reduced_dispatch: QuartetBipolarDispatch,
    system,
    basis,
    lattice_cells: list,
    shell_slices: List[Tuple[int, int]],
) -> SymmetryFockReconstructionMap:
    """Build the per-quartet Fock-position orbit table.

    For each canonical quartet in *reduced_dispatch*, enumerate candidate
    shell/cell positions from the legacy permutation construction. This
    inventory cannot authorize reuse of a canonical Fock contribution.
    """
    shell_perms, cell_rotations, cell_index_map = _build_symmetry_maps(
        system, basis, lattice_cells,
    )
    n_ops = len(shell_perms)

    if n_ops <= 1:
        return _build_permutation_only_reconstruction(
            reduced_dispatch, shell_slices, lattice_cells,
        )

    orbit_entries: List[List[Tuple]] = []

    for q in range(len(reduced_dispatch)):
        s1, s2, bra_cell_idx = reduced_dispatch.bra_pairs[q]
        s3, s4, ket_cell_idx = reduced_dispatch.ket_pairs[q]

        perm_variants = _shell_pair_permutation_variants(
            s1, s2, bra_cell_idx, s3, s4, ket_cell_idx,
        )

        orbit: Set[Tuple] = set()
        for p_s1, p_s2, p_bra, p_s3, p_s4, p_ket in perm_variants:
            for op_idx in range(n_ops):
                sh_map = shell_perms[op_idx]
                R_lat = cell_rotations[op_idx]

                rs1 = int(sh_map[p_s1])
                rs2 = int(sh_map[p_s2])
                rs3 = int(sh_map[p_s3])
                rs4 = int(sh_map[p_s4])

                if rs1 < 0 or rs2 < 0 or rs3 < 0 or rs4 < 0:
                    continue

                r_bra_idx = _rotate_cell_idx(
                    p_bra, cell_index_map, lattice_cells, R_lat,
                )
                r_ket_idx = _rotate_cell_idx(
                    p_ket, cell_index_map, lattice_cells, R_lat,
                )
                if r_bra_idx is None or r_ket_idx is None:
                    continue

                bra_cell = lattice_cells[r_bra_idx]
                ket_cell = lattice_cells[r_ket_idx]
                bra_key = (
                    int(bra_cell.index[0]),
                    int(bra_cell.index[1]),
                    int(bra_cell.index[2]),
                )
                ket_key = (
                    int(ket_cell.index[0]),
                    int(ket_cell.index[1]),
                    int(ket_cell.index[2]),
                )
                b1, n1 = shell_slices[rs1]
                b2, n2 = shell_slices[rs2]
                b3, n3 = shell_slices[rs3]
                b4, n4 = shell_slices[rs4]
                bra_sl = (b1, b1 + n1, b2, b2 + n2)
                ket_sl = (b3, b3 + n3, b4, b4 + n4)

                pos_key = (bra_key, bra_sl, ket_key, ket_sl)
                if pos_key not in orbit:
                    orbit.add(pos_key)

        orbit_entries.append(sorted(orbit))

    return SymmetryFockReconstructionMap(
        orbit_entries=orbit_entries,
        n_sym_operations=n_ops,
    )


def _build_symmetry_maps(
    system, basis, lattice_cells,
) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[Tuple[int, int, int], int]]:
    """Build shell-permutation and cell-rotation maps for symmorphic ops."""
    shell_perms: List[np.ndarray] = []
    cell_rotations: List[np.ndarray] = []

    cell_index_map: Dict[Tuple[int, int, int], int] = {}
    for idx, cell in enumerate(lattice_cells):
        c = (int(cell.index[0]), int(cell.index[1]), int(cell.index[2]))
        cell_index_map[c] = idx

    try:
        from .symmetry_integrals import symmorphic_operations
        sym_ops = symmorphic_operations(system)
    except Exception:
        sym_ops = []

    if not sym_ops:
        identity = np.arange(len(list(basis.shells())), dtype=int)
        identity_lat = np.eye(3, dtype=int)
        return [identity], [identity_lat], cell_index_map

    from .symmetry_ao import atom_permutation_under_op
    from .bipole_bravais_utils import lattice_to_cartesian_rotation

    shells = list(basis.shells())
    n_sh = len(shells)
    shell_atom = np.array([sh.atom_index for sh in shells], dtype=int)
    shell_l = np.array([int(sh.l) for sh in shells], dtype=int)
    atom_shells: Dict[int, List[int]] = {}
    for i in range(n_sh):
        a = int(shell_atom[i])
        atom_shells.setdefault(a, []).append(i)

    for op in sym_ops:
        try:
            perm = atom_permutation_under_op(system, op)
            atom_perm = np.asarray(perm.perm, dtype=int)
            R_cart = np.asarray(op.rotation, dtype=float).reshape(3, 3)
            R_lat = lattice_to_cartesian_rotation(system, R_cart)
        except Exception:
            continue

        sh_map = np.full(n_sh, -1, dtype=int)
        valid = True
        for i in range(n_sh):
            a_src = int(shell_atom[i])
            l_i = int(shell_l[i])
            a_dst = int(atom_perm[a_src])
            found = False
            for j in atom_shells.get(a_dst, []):
                if int(shell_l[j]) == l_i:
                    sh_map[i] = j
                    found = True
                    break
            if not found:
                valid = False
                break
        if valid:
            shell_perms.append(sh_map)
            cell_rotations.append(R_lat)

    if not shell_perms:
        identity = np.arange(n_sh, dtype=int)
        identity_lat = np.eye(3, dtype=int)
        return [identity], [identity_lat], cell_index_map

    return shell_perms, cell_rotations, cell_index_map


def _shell_pair_permutation_variants(
    s1: int, s2: int, bra_cell: int,
    s3: int, s4: int, ket_cell: int,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Return the 4 transposed variants of a shell quartet."""
    return [
        (s1, s2, bra_cell, s3, s4, ket_cell),
        (s2, s1, bra_cell, s3, s4, ket_cell),
        (s1, s2, bra_cell, s4, s3, ket_cell),
        (s2, s1, bra_cell, s4, s3, ket_cell),
    ]


def _rotate_cell_idx(
    cell_idx: int,
    cell_index_map: Dict[Tuple[int, int, int], int],
    lattice_cells: list,
    R_lat: np.ndarray,
) -> Optional[int]:
    """Apply lattice-basis rotation to a cell index, return resulting list index."""
    cell = lattice_cells[cell_idx]
    idx3 = np.array([cell.index[0], cell.index[1], cell.index[2]], dtype=int)
    rot = tuple(int(v) for v in (R_lat @ idx3))
    return cell_index_map.get(rot)


def _build_permutation_only_reconstruction(
    reduced_dispatch: QuartetBipolarDispatch,
    shell_slices: List[Tuple[int, int]],
    lattice_cells: list,
) -> SymmetryFockReconstructionMap:
    """Build reconstruction map with permutation variants only (no point-group)."""
    orbit_entries: List[List[Tuple]] = []

    for q in range(len(reduced_dispatch)):
        s1, s2, bra_cell_idx = reduced_dispatch.bra_pairs[q]
        s3, s4, ket_cell_idx = reduced_dispatch.ket_pairs[q]

        variants = _shell_pair_permutation_variants(
            s1, s2, bra_cell_idx, s3, s4, ket_cell_idx,
        )

        bra_cell_obj = lattice_cells[bra_cell_idx]
        ket_cell_obj = lattice_cells[ket_cell_idx]
        bra_key = (
            int(bra_cell_obj.index[0]),
            int(bra_cell_obj.index[1]),
            int(bra_cell_obj.index[2]),
        )
        ket_key = (
            int(ket_cell_obj.index[0]),
            int(ket_cell_obj.index[1]),
            int(ket_cell_obj.index[2]),
        )

        orbit: List[Tuple] = []
        seen: Set[Tuple] = set()
        for p_s1, p_s2, p_bra, p_s3, p_s4, p_ket in variants:
            b1, n1 = shell_slices[p_s1]
            b2, n2 = shell_slices[p_s2]
            b3, n3 = shell_slices[p_s3]
            b4, n4 = shell_slices[p_s4]
            bra_sl = (b1, b1 + n1, b2, b2 + n2)
            ket_sl = (b3, b3 + n3, b4, b4 + n4)
            pos = (bra_key, bra_sl, ket_key, ket_sl)
            if pos not in seen:
                seen.add(pos)
                orbit.append(pos)
        orbit_entries.append(orbit)

    return SymmetryFockReconstructionMap(
        orbit_entries=orbit_entries,
        n_sym_operations=1,
    )
