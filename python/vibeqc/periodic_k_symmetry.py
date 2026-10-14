"""Rotation-aware k-star expansion for IBZ-reduced periodic SCF.

A symmetry-reduced (IBZ) Monkhorst-Pack mesh lets the SCF diagonalise
at the irreducible k-points only. But any quantity reconstructed in
real space -- the inverse-Bloch density D(g), the reciprocal-space
density transform r̂(K) -- needs the *full-mesh* k-densities, and the
star members are **not copies** of their representative:

    D(R.k) = P(R) . D(k) . P(R)ᵀ          (proper/improper rotation R)
    D(-k)  = D(k)*                        (time reversal)

with ``P(R)`` the real-AO rotation matrix (atom permutation x on-atom
Wigner-D blocks). Replicating ``D(k_rep)`` without the rotation -- the
2026-05 "IBZ-native" shortcut -- is exact only when every ``P(R)`` acts
trivially on the occupied density (e.g. one s-shell atom at the
rotation centre: the He validation cells). On MgO/STO-3G [2,2,2] the
shortcut leaves the SCF unconverged 8.25 Ha away from the full-mesh
result (2026-06-10 probe; regression below pins the fixed behaviour).

STATUS (2026-06-15): transport CORRECTED, driver wiring pending. The
BIPOLE drivers still expand IBZ input meshes to the full mesh up front
(correct by construction, no IBZ savings) -- wiring this module into the
SCF loop so it diagonalises only at the IBZ points is the remaining
step. The transport itself is now exact.

The 2026-06-10 open question was the missing **per-atom lattice-shift
Bloch phase**. ``D(R.k) = P(R).D(k).P(R)ᵀ`` is the right shape only when
every operation keeps each atom in the home cell; a shift-bearing
operation (e.g. inversion on rock-salt sends the anion to its image one
cell over) needs the phase-dressed Bloch representation

    Γ(k)_{mu'mu} = e^{i k.L_a} . D^l(R)_{mu'mu}      (mu on atom a, mu' on b)

with ``L_a`` the lattice vector of atom a's image (see
:func:`expand_k_matrices_to_full`). The earlier probe recorded the
shift phase as "not explanatory" because it measured the transport on
the *converged full-mesh D(k)*, which carries the SCF's own
cell-list-asymmetry floor, rather than on a directly-computed
absolutely-convergent operator. Probed on the overlap S(k) the phase is
decisive -- the worst star-transport residual over MgO/STO-3G [2,2,2]
falls from **0.78 (phaseless) -> 4.5e-9** as the cutoff fold-converges
(8.8e-3 at c12 -> 1.6e-3 at c14 -> 1.8e-6 at c16 -> 4.5e-9 at c20); the
remaining residual is ordinary cell-list truncation asymmetry, the same
quantity the corrected gauge already fold-converges via the S(Γ) drift
guard. So true IBZ-native reduction needs (a) the phase-aware transport
here and (b) a fold-converged cutoff -- no fundamental Fock-asymmetry
obstruction.

True IBZ-native reduction on every multi-k route (BIPOLE, EWALD, GDF)
builds on the helpers below:

* :func:`star_operations` -- for each full-mesh point, which operation
  (and optional time-reversal) maps its IBZ representative onto it; the
  default candidate set is the symmorphic subgroup, and a consumer whose
  finite source is closed under glides and screws may admit those too;
* :func:`expand_k_matrices_to_full` -- apply the AO rotations to
  density-like per-k matrices;
* :func:`density_set_from_k_matrices` -- inverse-Bloch fold arbitrary
  per-k densities into a real-space ``LatticeMatrixSet`` (the
  from-D(k) variant of ``real_space_density_from_kpoints``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    compute_overlap_lattice,
)
from .symmetry_integrals import symmorphic_operations

__all__ = [
    "KMeshUnfolding",
    "KStarMap",
    "star_operations",
    "little_group_entries",
    "transport_k_matrix_at",
    "expand_k_matrices_to_full",
    "density_set_from_k_matrices",
]


@dataclass(frozen=True)
class KStarMap:
    """Mapping from a full MP mesh onto an IBZ mesh's stars.

    ``entries[j] = (i_rep, op_idx, time_reversal)`` -- full-mesh point j
    is reached from IBZ representative ``i_rep`` by operation ``op_idx``
    (index into ``operations``), composed with complex conjugation when
    ``time_reversal`` is True.
    """

    entries: List[Tuple[int, int, bool]]
    # The SymmetryOp list used for matching: the symmorphic subgroup by
    # default, or the admitted subset a consumer passed (which may carry
    # fractional translations).
    operations: list


def _frac_coords(k_cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Cartesian k -> fractional reciprocal coordinates.

    Columns of ``A`` are direct vectors and columns of
    ``B = 2pi.(A^-1)^T`` are reciprocal vectors. Native Monkhorst-Pack
    constructs ``k = B q``, so the inverse conversion solves ``B q = k``.
    The distinction is observable on the oblique 2D `[3,3,1]` regression;
    the former transpose happened to work only when ``B`` was symmetric.
    """
    B = 2.0 * np.pi * np.linalg.inv(lattice).T
    return np.linalg.solve(B, np.asarray(k_cart, dtype=float))


def star_operations(
    system: PeriodicSystem,
    kmesh_ibz,
    kmesh_full,
    *,
    atol: float = 1e-8,
    operations: Optional[Sequence] = None,
) -> KStarMap:
    """Match every full-mesh k-point to (IBZ representative, operation).

    Matching runs in fractional reciprocal coordinates modulo integer
    reciprocal-lattice translations. spglib's irreducible meshes use
    time reversal, so ``-R.k_rep`` is tried when no proper match exists.

    ``operations`` restricts matching to an admitted subset of the
    system's attached operations. The default is the symmorphic
    subgroup, for consumers that have not qualified fractional
    translations on their finite source. A consumer whose source is
    closed under only some operations passes exactly those, so no star
    member is reached through another one; the subset may include
    glides and screws, whose fractional translation
    :func:`expand_k_matrices_to_full` carries through the atom-image
    shift ``L_a`` (the reciprocal action ``W^-T`` does not see it).

    Only operations that map the WHOLE full mesh onto itself are used.
    A finite-mesh quantity such as the exchange ``K(k_i) = S_j w_j
    (.. k_i k_j ..) D(k_j)`` is covariant under an operation only when
    that operation permutes the mesh points, so transporting through an
    operation that merely sends one representative onto a mesh point
    (possible on an anisotropic mesh of a cubic cell, e.g. (2,2,3) on
    diamond) would produce a matrix of a different finite approximation.
    The check is the whole-mesh compatibility the native Seitz transport
    enforces (``mesh_action`` in ``periodic_ao_bloch_transport.cpp``),
    evaluated here on the fractional coordinates of the mesh.

    Raises ``ValueError`` when a full-mesh point cannot be reached from
    any representative through a mesh-compatible admitted operation --
    that means the meshes don't belong together (different mesh/shift),
    the system's symmetry changed since the IBZ reduction, or the
    admitted operations do not cover the star.
    """
    ops_all = getattr(getattr(system, "symmetry", None), "operations", None)
    if not ops_all:
        raise ValueError(
            "star_operations: system carries no symmetry operations; "
            "attach_symmetry() first"
        )
    ops = symmorphic_operations(ops_all)
    if operations is not None:
        # Every admitted operation must be one of the attached operations
        # (rotation AND translation), so a caller cannot admit a foreign
        # rotation or strip a glide of its translation.
        def _same(a, b):
            return np.array_equal(
                np.asarray(a.rotation), np.asarray(b.rotation)
            ) and np.array_equal(
                np.asarray(a.translation), np.asarray(b.translation)
            )

        chosen = list(operations)
        if not chosen or any(
            not any(_same(op, candidate) for candidate in ops_all)
            for op in chosen
        ):
            raise ValueError(
                "star_operations: operations must be a nonempty subset of "
                "the system's attached symmetry operations"
            )
        ops = chosen
    lattice = np.asarray(system.lattice, dtype=float)

    k_ibz_frac = [
        _frac_coords(np.asarray(k, dtype=float).reshape(3), lattice)
        for k in kmesh_ibz.kpoints
    ]
    k_full_frac = [
        _frac_coords(np.asarray(k, dtype=float).reshape(3), lattice)
        for k in kmesh_full.kpoints
    ]

    # Reciprocal action of a real-space operation with integer fractional
    # rotation W (r_frac -> W.r_frac + tau): q_frac -> W^{-T}.q_frac; the
    # translation does not act on k. For integer W with |det W| = 1,
    # W^{-T} is integer as well.
    op_recip = []
    for op in ops:
        W = np.rint(np.asarray(op.rotation, dtype=float)).astype(int)
        W_inv_T = np.rint(np.linalg.inv(W)).astype(int).T
        op_recip.append(W_inv_T)

    def _matches(qa: np.ndarray, qb: np.ndarray) -> bool:
        d = qa - qb
        return bool(np.all(np.abs(d - np.rint(d)) < atol))

    # Whole-mesh compatibility: every full-mesh point must land on a
    # full-mesh point (modulo G) under the operation. Mesh points are
    # rationals m/N, so a wrapped coordinate rounded well inside ``atol``
    # is an exact key; this stays O(n_k) per operation.
    q_all = np.asarray(k_full_frac, dtype=float).reshape(-1, 3)

    def _keys(points):
        wrapped = (points + 0.5) % 1.0 - 0.5
        wrapped[np.abs(wrapped - 0.5) < atol] = -0.5
        return {tuple(row) for row in np.rint(wrapped / atol * 1e-2).astype(np.int64)}

    mesh_keys = _keys(q_all)
    compatible = [_keys(q_all @ WiT.T) <= mesh_keys for WiT in op_recip]

    entries: List[Tuple[int, int, bool]] = []
    for j, q_full in enumerate(k_full_frac):
        found: Optional[Tuple[int, int, bool]] = None
        for i, q_rep in enumerate(k_ibz_frac):
            for op_idx, WiT in enumerate(op_recip):
                if not compatible[op_idx]:
                    continue
                q_rot = WiT @ q_rep
                if _matches(q_rot, q_full):
                    found = (i, op_idx, False)
                    break
                if _matches(-q_rot, q_full):  # time reversal
                    found = (i, op_idx, True)
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError(
                f"star_operations: full-mesh k-point {j} is not in the "
                "star of any IBZ representative through a mesh-compatible "
                f"operation ({sum(compatible)} of {len(ops)} candidate "
                "operations map the whole mesh onto itself; mesh/shift "
                "mismatch, stale symmetry, or the admitted operations do "
                "not cover the star)"
            )
        entries.append(found)
    return KStarMap(entries=entries, operations=list(ops))


def _ao_source_atom_index(basis: BasisSet) -> np.ndarray:
    """``(nbf,)`` array giving the home-cell atom each AO is centred on."""
    idx: List[int] = []
    for sh in basis.shells():
        idx.extend([int(sh.atom_index)] * (2 * int(sh.l) + 1))
    return np.asarray(idx, dtype=int)


def _op_atom_shifts_cart(system: PeriodicSystem, operations) -> List[np.ndarray]:
    """Per-operation Cartesian atom shifts ``L_a`` (one ``(n_atom, 3)`` each).

    For op ``g = {R|t}`` the home-cell atom ``a`` maps to atom
    ``b = perm[a]`` obeying ``r_b = R.r_a + t + L_a`` with the lattice
    vector ``L_a = lattice . shift[a]`` (``atom_permutation_under_op``
    returns the integer fractional ``shift``). This is the bookkeeping
    that :func:`build_ao_permutation_matrix` deliberately drops -- the
    caller restores it as the Bloch phase below.
    """
    from .symmetry_ao import atom_permutation_under_op
    from .symmetry_lattice import lattice_to_cartesian_rotation

    L = np.asarray(system.lattice, dtype=float)
    out: List[np.ndarray] = []
    for op in operations:
        R_cart = lattice_to_cartesian_rotation(op.rotation, L)
        t_cart = L @ np.asarray(op.translation, dtype=float)
        ap = atom_permutation_under_op(system, R_cart, t_cart)
        # L_a = lattice . shift[a]; row a holds shift[a] @ Lᵀ = L . shift[a].
        out.append(np.asarray(ap.lattice_shift, dtype=float) @ L.T)
    return out


def expand_k_matrices_to_full(
    M_k_ibz: Sequence[np.ndarray],
    star_map: KStarMap,
    system: PeriodicSystem,
    basis: BasisSet,
    kmesh_full,
) -> List[np.ndarray]:
    """Expand density-like per-k matrices from the IBZ to the full mesh.

    The full-mesh point ``j`` is reached from IBZ representative
    ``k_rep`` by ``k = R.k_rep`` (entry ``(i_rep, op_idx, trev)``). For a
    Seitz operation ``g = {R|tau}`` (``tau`` the fractional translation,
    zero for a symmorphic operation) that maps home-cell atom ``a`` to
    atom ``b = perm[a]`` in the neighbouring cell ``L_a``
    (``r_b = R.r_a + tau + L_a``), the Bloch-space representation matrix
    at ``k`` is

        Γ(k)_{mu'mu} = e^{i k.L_a} . D^l(R)_{mu'mu}      (mu on a, mu' on b)

    so that ``M(k) = Γ(k).M(k_rep).Γ(k)+``. Derivation: with the AO Bloch
    sum ``phi_{ν,k} = S_T e^{i k.T} chi_ν(.-x_ν-T)`` and ``(g f)(r) =
    f(R^-1 (r - tau))``,

        g chi_ν(. - x_a - T) = S_{ν'} D_{ν'ν} chi_{ν'}(. - x_b - (R T - L_a)),

    and substituting ``T' = R T - L_a`` (a lattice vector) with
    ``k_rep.T = k.(T' + L_a)`` for the orthogonal ``R`` gives

        g phi_{ν,k_rep} = e^{i k.L_ν} S_{ν'} D_{ν'ν} phi_{ν',k}.

    The fractional translation enters only through the atom-image shift
    ``L_a`` (which :func:`_op_atom_shifts_cart` computes with ``tau``
    included). There is no separate ``e^{-i k.tau}`` factor in this
    AO-centred Bloch convention: that factor belongs to the atom-centred
    convention ``phi' = e^{i k.x_a} phi``, where it is a global phase and
    the per-atom ``L_a`` phases are absent; added here it cancels in
    ``Γ M Γ+``. The native Seitz transport
    (``cpp/src/periodic_ao_bloch_transport.cpp``) uses the same
    convention. Proven on Bloch overlaps and explicit finite exchange
    sources for diamond Fd-3m and hcp P6_3/mmc
    (``tests/test_periodic_k_gdf_ibz_nonsymmorphic.py``).

    The per-atom phase ``e^{i k.L_a}`` is exactly the lattice-shift
    bookkeeping that the real ``P(R) = build_ao_permutation_matrix``
    drops. Omitting it is what left the 2026-06-10 "IBZ-native" probe
    with an O(1) transport error on every shift-bearing operation (e.g.
    inversion on rock-salt maps the anion to its image in a neighbouring
    cell); see the module STATUS note. ``e^{i G.L_a} = 1`` for any
    reciprocal-lattice ``G``, so the full-mesh ``k`` may be used directly
    in place of the folded ``R.k_rep``. Time-reversal members
    (``trev=True``, ``k = -R.k_rep``) use ``-k`` in the phase and the
    complex conjugate of the transported block (``D(-k) = D(k)*``).
    """
    from .symmetry_scf import build_ao_permutation_cache

    ops = star_map.operations
    P_cache = build_ao_permutation_cache(system, basis, ops)
    shift_cache = _op_atom_shifts_cart(system, ops)
    ao_atom = _ao_source_atom_index(basis)
    kfull = [np.asarray(k, dtype=float).reshape(3) for k in kmesh_full.kpoints]
    if len(kfull) != len(star_map.entries):
        raise ValueError(
            "expand_k_matrices_to_full: kmesh_full has "
            f"{len(kfull)} points but the star map has "
            f"{len(star_map.entries)} entries"
        )

    out: List[np.ndarray] = []
    for j, (i_rep, op_idx, trev) in enumerate(star_map.entries):
        M = np.asarray(M_k_ibz[i_rep])
        P = np.asarray(P_cache[op_idx], dtype=float)
        La = shift_cache[op_idx]                       # (n_atom, 3)
        sign = -1.0 if trev else 1.0
        # Per-source-atom Bloch phase e^{i (sign.k).L_a}, broadcast to AOs.
        phase_atom = np.exp(1j * sign * (La @ kfull[j]))   # (n_atom,)
        phase_ao = phase_atom[ao_atom]                     # (nbf,)
        Gamma = P.astype(complex) * phase_ao[None, :]      # P . diag(phase)
        Mj = Gamma @ M @ Gamma.conj().T
        if trev:
            Mj = Mj.conj()
        out.append(Mj)
    return out


def little_group_entries(
    star_map: KStarMap,
    system: PeriodicSystem,
    kmesh_full,
    point_indices: Sequence[int],
    *,
    atol: float = 1e-8,
) -> List[List[Tuple[int, bool]]]:
    """Operations of the star map fixing each listed full-mesh point.

    For point ``j`` the entry lists ``(op_idx, time_reversal)`` for every
    operation of ``star_map.operations`` whose reciprocal action sends
    ``k_j`` onto ``k_j`` modulo a reciprocal-lattice vector, with
    ``time_reversal=True`` when it is ``-R.k_j`` that lands there. The
    identity without time reversal is included, so the list is never
    empty. These are the operations under which a symmetric per-k matrix
    at ``k_j`` must be invariant (the little group, extended by time
    reversal), which the star map's member-to-representative entries never
    test: a representative maps onto itself through the identity only.
    """
    lattice = np.asarray(system.lattice, dtype=float)
    kfull = [np.asarray(k, dtype=float).reshape(3) for k in kmesh_full.kpoints]
    op_recip = []
    for op in star_map.operations:
        W = np.rint(np.asarray(op.rotation, dtype=float)).astype(int)
        op_recip.append(np.rint(np.linalg.inv(W)).astype(int).T)
    out: List[List[Tuple[int, bool]]] = []
    for j in point_indices:
        q = _frac_coords(kfull[int(j)], lattice)
        fixing: List[Tuple[int, bool]] = []
        for op_idx, WiT in enumerate(op_recip):
            q_rot = WiT @ q
            for trev, sign in ((False, 1.0), (True, -1.0)):
                d = sign * q_rot - q
                if bool(np.all(np.abs(d - np.rint(d)) < atol)):
                    fixing.append((op_idx, trev))
        out.append(fixing)
    return out


def transport_k_matrix_at(
    M: np.ndarray,
    k: np.ndarray,
    P: np.ndarray,
    shifts: np.ndarray,
    ao_atom: np.ndarray,
    *,
    time_reversal: bool,
) -> np.ndarray:
    """``Gamma(k) M Gamma(k)^H`` (conjugated for time reversal) at one k.

    The same law as :func:`expand_k_matrices_to_full`, for a single
    operation acting at a fixed momentum: ``P`` the real AO rotation,
    ``shifts`` the per-atom image shifts ``L_a`` of that operation,
    ``ao_atom`` the AO-to-atom index. With ``time_reversal`` the phase is
    taken at ``-k`` and the result conjugated (``M(-k) = M(k)*``).
    """
    sign = -1.0 if time_reversal else 1.0
    phase = np.exp(1j * sign * (np.asarray(shifts, dtype=float) @ np.asarray(k, dtype=float)))
    gamma = np.asarray(P, dtype=float).astype(complex) * phase[np.asarray(ao_atom)][None, :]
    out = gamma @ np.asarray(M) @ gamma.conj().T
    return out.conj() if time_reversal else out


@dataclass
class KMeshUnfolding:
    """One SCF's worth of IBZ -> full-BZ star bookkeeping.

    Composes the primitives below for a driver that diagonalises on the
    irreducible wedge but needs full-BZ per-k densities (exact exchange,
    real-space density folds). Build once per SCF via :meth:`build`;
    every method is then pure bookkeeping plus the cached AO transport.

    ``rep_full_index[i]`` is the full-mesh index of wedge representative
    ``i`` **matched by coordinate**, never by position in ``ir_mapping``:
    spglib enumerates the mesh first-axis-fastest while the native mesh
    generator runs last-axis-fastest, so index-pairing the two lists is
    a latent transposition bug (this is the ``expand_ibz_eigenvalues``
    hazard recorded in the 2026-07-29 IBZ scouting pass).

    Density law per Pisani, Dovesi & Roetti, *Hartree-Fock Ab Initio
    Treatment of Crystalline Systems* (Lecture Notes in Chemistry 48,
    Springer, 1988), Sec. II.7: with ``W(V; k^V)`` of Eq. II.7.15 the
    density transforms as Eq. II.7.21, which in this code's
    ``D = C n C^H`` convention reads ``D(V k) = W D(k) W^H``; time
    reversal supplies ``D(-k) = D(k)^*`` (inversion is always present in
    reciprocal space). :func:`expand_k_matrices_to_full` implements
    exactly that law; journal version Dovesi, Int. J. Quantum Chem. 29,
    1755 (1986), doi:10.1002/qua.560290608.
    """

    kmesh_full: object
    star_map: KStarMap
    rep_full_index: List[int]
    system: PeriodicSystem
    basis: BasisSet

    @classmethod
    def build(
        cls,
        system: PeriodicSystem,
        basis: BasisSet,
        kmesh_ibz,
    ) -> "KMeshUnfolding":
        """Regenerate the full MP mesh and match the wedge onto it.

        Raises ``ValueError`` when the wedge cannot be covered by the
        attached symmorphic operations plus time reversal -- the caller
        decides whether to fail closed or fall back to an up-front
        full-mesh expansion.
        """
        from ._vibeqc_core import monkhorst_pack as _native_mp

        mesh = [int(x) for x in getattr(kmesh_ibz, "mesh", (1, 1, 1))]
        shift = [int(x) for x in getattr(kmesh_ibz, "is_shift", (0, 0, 0))]
        kmesh_full = _native_mp(system, mesh, shift, False)
        star_map = star_operations(system, kmesh_ibz, kmesh_full)

        # Coordinate-match each wedge representative to its own full-mesh
        # slot (fractional reciprocal coordinates modulo G).
        lattice = np.asarray(system.lattice, dtype=float)
        full_frac = [
            _frac_coords(np.asarray(k, dtype=float).reshape(3), lattice)
            for k in kmesh_full.kpoints
        ]
        rep_full_index: List[int] = []
        for k_rep in kmesh_ibz.kpoints:
            q_rep = _frac_coords(
                np.asarray(k_rep, dtype=float).reshape(3), lattice
            )
            for j, q_full in enumerate(full_frac):
                d = q_rep - q_full
                if np.all(np.abs(d - np.rint(d)) < 1e-8):
                    rep_full_index.append(j)
                    break
            else:
                raise ValueError(
                    "KMeshUnfolding: wedge representative "
                    f"{np.asarray(k_rep).tolist()} is not on the "
                    f"regenerated full mesh {tuple(mesh)} -- mesh/shift "
                    "metadata is inconsistent"
                )
        return cls(
            kmesh_full=kmesh_full,
            star_map=star_map,
            rep_full_index=rep_full_index,
            system=system,
            basis=basis,
        )

    @property
    def n_full(self) -> int:
        return len(self.star_map.entries)

    def unfold(self, M_k_ibz: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Wedge per-k density-like matrices -> full-BZ list."""
        return expand_k_matrices_to_full(
            M_k_ibz, self.star_map, self.system, self.basis, self.kmesh_full
        )

    def fold_density(
        self,
        M_k_ibz: Sequence[np.ndarray],
        lattice_opts: LatticeSumOptions,
    ) -> LatticeMatrixSet:
        """Wedge per-k densities -> exact real-space lattice fold.

        The weighted *wedge* fold ``S_i w_i Re[e^{-ik_i.R} D(k_i)]`` uses
        each representative in place of its orbit average and is wrong at
        multi-cell cutoffs; the orbit sum (Pisani Eq. II.7.48) requires
        the rotated members. Unfold first, then fold over the full mesh.
        """
        return density_set_from_k_matrices(
            self.system,
            self.basis,
            lattice_opts,
            self.kmesh_full,
            self.unfold(M_k_ibz),
        )


def density_set_from_k_matrices(
    system: PeriodicSystem,
    basis: BasisSet,
    lattice_opts: LatticeSumOptions,
    kmesh_full,
    D_k_full: Sequence[np.ndarray],
) -> LatticeMatrixSet:
    """Inverse-Bloch fold per-k densities into a real-space lattice set.

    ``D(g) = S_k w_k Re[e^{-i k.R_g} D(k)]`` over the *full* mesh (the
    shared from-D(k) variant of ``real_space_density_from_kpoints``).
    """
    template = compute_overlap_lattice(basis, system, lattice_opts)
    weights = [float(w) for w in kmesh_full.weights]
    kpts = [np.asarray(k, dtype=float).reshape(3) for k in kmesh_full.kpoints]
    if len(D_k_full) != len(kpts):
        raise ValueError(
            "density_set_from_k_matrices: density/kmesh length mismatch "
            f"({len(D_k_full)} vs {len(kpts)})"
        )
    nbf = int(basis.nbasis)
    for c, cell in enumerate(template.cells):
        R = np.asarray(cell.r_cart, dtype=float)
        block = np.zeros((nbf, nbf), dtype=float)
        for w_k, k_arr, D_k in zip(weights, kpts, D_k_full):
            phase = np.exp(-1j * float(np.dot(k_arr, R)))
            block += w_k * np.real(phase * np.asarray(D_k))
        template.set_block(c, block)
    return template
