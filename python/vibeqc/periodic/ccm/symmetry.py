"""Space-group symmetry of the cyclic cluster (Task 3, spglib).

The crystal space group, on top of the translational symmetry the CCM already
carries, identifies the subgroup of point/space operations that leave the cyclic
cluster invariant -- the operations that can reduce the symmetry-unique integral
count and symmetrize the Fock matrix.

Pipeline (reusing existing infrastructure, not reimplementing):

1. ``detect_spacegroup(ccm.unit_system)`` (spglib) -> the crystal space group and
   its operations ``{W|w}`` (integer rotation ``W`` in the unit-cell fractional
   basis + fractional translation ``w``).
2. Each op is converted to Cartesian: ``R = L_u W L_u⁻¹``
   (:func:`vibeqc.symmetry_lattice.lattice_to_cartesian_rotation`), ``t = L_u w``
   (``L_u`` = unit-cell lattice, columns = aⱼ).
3. The **cluster-invariant subgroup** is the set of ops that map the cluster
   lattice onto itself modulo the cluster lattice -- tested operationally by
   :func:`vibeqc.symmetry_ao.atom_permutation_under_op` on the cluster supercell
   (it raises when ``(R,t)`` is not a symmetry of the cluster). For equal ``nrep``
   in symmetry-related directions all crystal point ops survive; unequal ``nrep``
   keeps only the compatible subset. Incompatible cluster shapes simply yield a
   smaller subgroup -- surfaced, not silently dropped.
4. For each surviving op the AO real-solid-harmonic rotation matrix
   ``P = build_ao_permutation_matrix(basis, R, perm)`` (atom permutation x Wigner-D
   blocks) is built over the supercell AO basis.

Correctness gate (this milestone): the CCM **overlap ``S^CCM`` and kinetic
``T^CCM`` are invariant** under every cluster-invariant op, ``Pᵀ M P = M`` to
machine precision (<=1e-14) -- proving the AO maps (atom permutation x Wigner-D)
are correct, including for non-symmorphic operations.

**Finding (proven).** The union three-center ``V^CCM`` (bare-1/r weight
``w_muν.1/2(w_muC+w_νC)``) is **not** crystal-symmetric -- ``‖PᵀVP-V‖/‖V‖`` ≈ 5e-4
(MgO) to 8e-3 (C-diamond) -- the same anchor-bridge defect the Sec.13 symmetric
four-center cured for *permutation* symmetry, here in the three-center's *spatial*
symmetry, so the bare-union ``h^CCM`` is not fully symmetric.

**Resolution (proven).** The **neutral Ewald ``V_ne`` IS crystal-symmetric**
(``‖PᵀVP-V‖/‖V‖`` ≈ 2e-8, lattice-sum tolerance). The symmetry breaking is the
*spatial* analogue of the Madelung gap (#16): bare-1/r union weights break both
permutation symmetry (four-center) and spatial symmetry (three-center V), and the
neutral Ewald kernel fixes both. So the **symmetric V already exists** -- the full
symmetry-exploitation milestone (petite list + Fock scatter, exact ``E``/Fock ==
symmetry-off gate) rides the **neutral / GDF route**, whose ``h`` is fully
symmetric (T + neutral ``V_ne`` + the permutation-symmetric four-center / neutral
cderi). See ``handovers/HANDOVER_AICCM_FOLLOWON.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CCMSymOp",
    "CCMSymmetry",
    "CCMPairOrbits",
    "CCMFoldKPairPlan",
    "analyze_ccm_symmetry",
    "ccm_symmetry_basis_rotations",
    "ccm_symmetry_fold_kpair_plan",
    "ccm_symmetry_invariance_residuals",
    "ccm_symmetry_unique_atom_pairs",
]


@dataclass
class CCMSymOp:
    """One cluster-invariant space-group operation, in the cluster AO basis."""

    index: int                  # index into the crystal operation list
    R_cart: np.ndarray          # (3,3) Cartesian rotation/reflection (orthogonal)
    t_cart: np.ndarray          # (3,) Cartesian translation (bohr); non-zero => non-symmorphic
    perm: np.ndarray            # (n_atoms,) supercell atom permutation
    P: np.ndarray               # (nbf, nbf) AO permutation x Wigner-D matrix (orthogonal)
    nonsymmorphic: bool         # True if the fractional translation w != 0


@dataclass
class CCMSymmetry:
    """Space-group analysis of a cyclic cluster."""

    number: int                 # spglib space-group number (1-230)
    international_symbol: str
    point_group: str
    crystal_order: int          # number of operations in the crystal space group
    invariant_ops: list = field(default_factory=list)  # list[CCMSymOp]

    @property
    def cluster_order(self) -> int:
        """Order of the cluster-invariant subgroup (includes identity)."""
        return len(self.invariant_ops)


def analyze_ccm_symmetry(ccm, *, symprec: float = 1e-5, tolerance: float = 1e-6) -> CCMSymmetry:
    """Space group of the crystal + the subgroup leaving the cyclic cluster invariant.

    Parameters
    ----------
    ccm : CCMSystem
    symprec : float
        spglib position tolerance for the crystal space-group search.
    tolerance : float
        bohr tolerance for matching atom images on the cluster supercell.

    Returns
    -------
    CCMSymmetry
        The crystal space group plus the cluster-invariant operations with their
        AO rotation matrices ``P`` (over the supercell basis).
    """
    from ...periodic_symmetrize import detect_spacegroup
    from ...symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix
    from ...symmetry_lattice import lattice_to_cartesian_rotation

    sg = detect_spacegroup(ccm.unit_system, symprec=symprec)
    L_u = np.asarray(ccm.unit_system.lattice, dtype=float)  # columns = a_j (bohr)
    nrep = np.asarray(ccm.nrep, dtype=float).reshape(3)

    invariant: list[CCMSymOp] = []
    for idx, op in enumerate(sg.operations):
        W = np.asarray(op.rotation, dtype=float)
        w = np.asarray(op.translation, dtype=float)
        # Cluster-invariance is TWO conditions, and both are required:
        #
        # 1. The op must map the cluster (BvK) lattice onto itself. The
        #    cluster vectors are c_j = n_j a_j (n = nrep), and the crystal
        #    op maps a_j -> S_i a_i W_ij, so R c_j = S_i c_i (W_ij n_j/n_i):
        #    the op normalizes the torus iff every W_ij n_j / n_i is an
        #    integer (D^-1 W D integer, D = diag(nrep)). Checking atom
        #    positions alone does NOT imply this: every image atom is a
        #    crystal site and every crystal site is congruent to some
        #    cluster site mod the cluster lattice, so the atom test below
        #    passes for EVERY crystal op regardless of cluster shape. An
        #    op that fails the lattice condition genuinely breaks the
        #    torus integrals -- measured P^T S P vs S residual 4.7 (!) on
        #    c-diamond (2,1,1) when such ops are admitted (caught
        #    2026-07-10 by the pair-reduction parity gate; 36 of the 48
        #    fcc ops fail the condition for nrep = (2,1,1)).
        mixed = W * nrep[None, :] / nrep[:, None]
        if not np.allclose(mixed, np.round(mixed), atol=1e-10):
            continue
        R = lattice_to_cartesian_rotation(W, L_u)
        t = L_u @ w
        # 2. (R,t) maps the supercell atom set onto itself mod the cluster
        #    lattice -- atom_permutation_under_op raises otherwise.
        try:
            ap = atom_permutation_under_op(ccm.cluster_system, R, t, tolerance=tolerance)
        except ValueError:
            continue
        P = np.asarray(build_ao_permutation_matrix(ccm.basis, R, ap), dtype=float)
        invariant.append(CCMSymOp(
            index=idx, R_cart=R, t_cart=t, perm=ap.perm, P=P,
            nonsymmorphic=bool(np.any(np.abs(w - np.round(w)) > 1e-8)),
        ))

    return CCMSymmetry(
        number=int(sg.number),
        international_symbol=str(sg.international_symbol),
        point_group=str(sg.point_group),
        crystal_order=len(sg.operations),
        invariant_ops=invariant,
    )


@dataclass
class CCMPairOrbits:
    """Orbits of home atom-pairs ``(A,B)`` under the cluster-invariant subgroup.

    The petite-list bookkeeping: only one representative pair per orbit needs its
    integral block computed; the rest follow by the AO rotation ``P`` of the op
    that maps the representative onto them. ``orbit_rep`` / ``orbit_op`` carry
    that map explicitly: for every ordered pair ``(A,B)``,
    ``representatives[orbit_rep[A,B]]`` is its orbit representative and
    ``sym.invariant_ops[orbit_op[A,B]]`` is one group element ``g`` with
    ``(g.perm[A_rep], g.perm[B_rep]) == (A, B)`` (any element of the coset works
    -- they differ by the pair's stabilizer, under which the integrals are
    invariant). Representative pairs themselves carry the sentinel
    ``orbit_op = -1`` (no reconstruction needed; also the only value present
    when the cluster has no symmetry, |G_c| = 1).
    """

    representatives: list       # list[(A, B)] -- one ordered pair per orbit
    multiplicity: list          # orbit size for each representative
    n_total: int                # n_atoms^2 ordered pairs
    reduction_factor: float     # n_total / n_unique -- average orbit size (-> |G_c|)
    orbit_rep: np.ndarray = None   # (n, n) int -- index into representatives
    orbit_op: np.ndarray = None    # (n, n) int -- index into sym.invariant_ops; -1 for reps

    @property
    def n_unique(self) -> int:
        return len(self.representatives)


def ccm_symmetry_unique_atom_pairs(ccm, sym: "CCMSymmetry | None" = None) -> CCMPairOrbits:
    """Symmetry-unique home atom-pairs under the cluster-invariant subgroup.

    Each op carries the ordered pair ``(A,B) -> (perm[A], perm[B])``; orbits are
    the equivalence classes. Computing one representative per orbit (then
    scattering via the AO rotation ``P``) is the space-group integral reduction
    *on top of* the CCM's existing translational reduction. The
    ``reduction_factor`` (average orbit size) approaches the order of the
    cluster-invariant group ``|G_c|`` as the cluster grows (generic pairs have a
    trivial stabilizer); for tiny cells it is small because the point ops fix the
    few atoms.
    """
    if sym is None:
        sym = analyze_ccm_symmetry(ccm)
    n = int(ccm.n_atoms)
    perms = [np.asarray(op.perm, dtype=int) for op in sym.invariant_ops]
    have_ops = bool(perms)
    if not perms:
        perms = [np.arange(n)]  # identity fallback (no symmetry found)

    orbit_rep = np.full((n, n), -1, dtype=int)
    orbit_op = np.full((n, n), -1, dtype=int)
    reps: list[tuple[int, int]] = []
    mult: list[int] = []
    for A in range(n):
        for B in range(n):
            if orbit_rep[A, B] >= 0:
                continue
            r = len(reps)
            members: set[tuple[int, int]] = set()
            for g, p in enumerate(perms):
                a, b = int(p[A]), int(p[B])
                members.add((a, b))
                if orbit_rep[a, b] < 0:
                    orbit_rep[a, b] = r
                    # -1 sentinel on the representative itself: identity map,
                    # no reconstruction (and no reliance on where the identity
                    # sits in invariant_ops -- or on it existing at all in the
                    # |G_c| = 1 fallback).
                    orbit_op[a, b] = g if have_ops and (a, b) != (A, B) else -1
            reps.append((A, B))
            mult.append(len(members))

    n_total = n * n
    return CCMPairOrbits(
        representatives=reps, multiplicity=mult, n_total=n_total,
        reduction_factor=n_total / len(reps),
        orbit_rep=orbit_rep, orbit_op=orbit_op,
    )


def ccm_symmetry_basis_rotations(sym: CCMSymmetry, basis) -> list:
    """AO rotation matrices of the cluster-invariant ops for an arbitrary basis.

    ``analyze_ccm_symmetry`` builds each op's ``P`` (atom permutation x
    real-solid-harmonic Wigner-D) over the *orbital* supercell basis. Symmetry
    reduction of the density-fit / RI build needs the same rotation over the
    **auxiliary** basis: any :class:`~vibeqc.BasisSet` centered on the same
    supercell atoms (e.g. the modrho-rescaled fitting basis) transforms with
    the same atom permutation ``op.perm`` and Cartesian rotation ``op.R_cart``
    -- only the shell structure differs. The per-shell radial factors (modrho
    rescaling included) are rotation-invariant, so the Wigner-D block
    construction applies unchanged.

    Returns one ``(n_bf_basis, n_bf_basis)`` orthogonal matrix per op, in
    ``sym.invariant_ops`` order.
    """
    from ...symmetry_ao import build_ao_permutation_matrix

    return [
        np.asarray(build_ao_permutation_matrix(basis, op.R_cart, op.perm),
                   dtype=float)
        for op in sym.invariant_ops
    ]


@dataclass
class CCMFoldKPairPlan:
    """Star reduction of a fold/multi-k ``(k_bra, k_ket)`` build set.

    For each build (a ``(qi, ai)`` pair: momentum pair ``(k_a, k_a + q)`` of
    the per-q fold), ``entries[(qi, ai)]`` is either

    * ``("build",)`` -- an orbit representative: run the fit builder; or
    * ``("recon", (rep_qi, rep_ai), op_index, trs)`` -- reconstruct from the
      representative's tensor by the group action (op ``op_index`` into
      ``ops``, optionally composed with time reversal).

    ``ops`` carries the unit-cell action of each referenced operation:
    ``(R_cart, perm, S_cart, P_ao, P_aux)`` with ``S_cart`` the per-atom
    cartesian lattice-shift vectors (``lattice_shift @ L_u.T``) and
    ``P_ao`` / ``P_aux`` the orbital / auxiliary AO rotation matrices over
    the **unit-cell** bases. Representatives always precede their members
    in the fold's ``(kept q, ai)`` iteration order, so a single forward
    sweep can build-and-cache reps and reconstruct members.
    """

    n_builds: int               # kept channels x mesh size
    n_reps: int                 # builder calls actually needed
    reduction_factor: float     # n_builds / n_reps
    entries: dict = field(default_factory=dict)
    ops: list = field(default_factory=list)
    # Indices into ``ops`` that passed BOTH admissibility tests: the
    # finite-list covariance test (:func:`ccm_symmetry_op_preserves_cell_list`)
    # and the AO/auxiliary map check (:func:`ccm_symmetry_op_ao_map_is_exact`).
    # Only these may appear in a ``"recon"`` entry; the rest are kept in
    # ``ops`` so indices stay stable for consumers that cache them.
    admissible_ops: tuple = ()


def ccm_symmetry_op_preserves_cell_list(s_cart, *, tol: float = 1e-8) -> bool:
    """Is the builder's finite Bloch cell list closed under this op?

    The fit builder Bloch-sums the KET AO over the origin-centred lattice ball
    ``|R| <= lat_opts.cutoff_bohr`` (``direct_lattice_cells``), the same list
    for every shell pair. Substituting a space-group op ``g = {R|t}``, whose
    atom action is ``x_perm[A] = R x_A + t + S_A``, into that sum reindexes the
    pair (bra on atom ``a``, ket on atom ``b``) as

        R'  =  R_rot R  +  (S_a - S_b)

    so the image pair's sum runs over ``Ball + (S_a - S_b)``. The ball is
    invariant under the rotation alone (``|R r| = |r|``, and ``R`` maps the
    lattice onto itself), so the reindexing maps the finite list exactly onto
    itself precisely when that offset vanishes for every contributing pair --
    that is, when the op's per-atom lattice shift is the SAME for all atoms.
    A constant shift is absorbed into ``t`` and moves no pair across the
    truncation boundary; a varying one exchanges cells at the boundary with
    cells that were never summed, which is exactly the vibeqc#337 defect.

    This is a **geometric, threshold-free** admissibility test: nothing is
    compared against a residual and no cutoff is widened, so it does not
    reintroduce the empirical threshold CLAUDE.md § 7 forbids (and which is
    why widening the Bloch cutoff was rejected as the vibeqc#337 fix). ``tol``
    only guards the floating-point comparison of shifts that are integer
    combinations of the lattice vectors by construction.

    The condition is **sufficient, not necessary**: a relation whose offset is
    nonzero is still exact once the pair density has decayed to nothing within
    ``|S_a - S_b|`` of the ball's edge -- the cells the offset exchanges then
    contribute zero either way, whether or not the builder's ``screen_tol``
    drops them explicitly. That is why the vibeqc#337 residuals converge away at
    25-30 bohr. Turning that into an admissibility rule would need an accuracy
    threshold, so this test deliberately takes the conservative half. On the
    cubic fixtures that costs nothing -- diamond / MgO / LiH ``(2,2,1)`` still
    reduce 16 -> 10, their relations being reachable through shift-constant ops
    and time reversal regardless. It is NOT free in general: hexagonal h-BN
    ``(2,2,1)`` (P6_3/mmc) drops from 8/16 to 10/16, trading 2.00x for an exact
    1.60x.

    The way to the refused relations is to make the cell list covariant rather
    than to argue its truncation is immaterial: a **pair-centred** ball gives
    ``|x_a' - x_b' - R'| = |x_a - x_b - R|`` identically, admitting every op at
    equal cell count. That is the pair-complete cell list the lattice work
    already uses for two-centre sums (#44); extending it to the GDF pair FT is
    #238, not this lane's change.

    Parameters
    ----------
    s_cart : (n_atoms, 3) float
        Per-atom cartesian lattice shifts ``lattice_shift @ L_u.T`` -- the
        ``S_cart`` entry of a :class:`CCMFoldKPairPlan` op tuple.
    tol : float
        Cartesian tolerance (bohr) for "the shifts are equal".
    """
    s = np.asarray(s_cart, dtype=float)
    if s.size == 0:
        return True
    return float(np.max(np.abs(s - s[0]))) < tol


def ccm_symmetry_op_ao_map_is_exact(p_ao, s_ref, *, tol: float = 1e-10) -> bool:
    """Does this op's AO rotation actually leave the reference overlap invariant?

    ``P`` is only a symmetry action of the basis if ``P^T S P == S`` -- the
    correctness gate :func:`ccm_symmetry_invariance_residuals` states for the
    cluster. The star reconstruction applies ``P`` to both AO indices and
    ``P_aux`` to the auxiliary index, so a ``P`` that fails this makes the
    reconstruction wrong no matter how covariant the cell list is.

    This is checked rather than assumed because it has been observed to fail.
    On a hexagonal cell the cartesian op ``R = L_u W L_u^-1`` is not exactly
    representable, and ``symmetry_core.euler_angles_from_rotation`` used to
    mis-extract the ZYZ angles for an ``R`` that is the identity to 1e-16:
    ``arccos(R[2,2])`` floored ``beta`` at ``sqrt(2 eps) ~ 1.5e-8``, which never
    tripped the then-current ``abs(sin(beta)) < 1e-12`` gimbal-lock guard, so the
    degenerate branch was skipped and it returned ``gamma = pi``.
    ``wigner_d_real(1, R)`` then came back as ``diag(-1, 1, -1)`` instead of the
    identity, and the h-BN (2,2,1) symmetry-reduced fold landed 10.9% away from
    the unreduced build. Cubic and tetragonal lattices are exactly
    representable, so their ``beta`` was exactly zero and no cubic fixture
    exposed it.

    Both halves of that extractor bug have since been fixed: #233 replaced
    ``arccos(R[2,2])`` with ``atan2(hypot(R02, R12), R22)``, and #282 replaced
    the ``1e-12`` threshold with an exact ``sin_beta == 0.0`` test and
    conditioned the near-pole branch. This check stays because it measures the
    invariance instead of assuming it: it now passes for the ops it used to
    refuse, and it would catch a regression in the same place.

    Parameters
    ----------
    p_ao : (n, n) float
        The op's AO rotation matrix over the reference basis.
    s_ref : (n, n) float
        Overlap of that same basis.
    tol : float
        Relative tolerance for a residual that is algebraically zero.
    """
    p_ao = np.asarray(p_ao, dtype=float)
    s_ref = np.asarray(s_ref, dtype=float)
    scale = max(1.0, float(np.max(np.abs(s_ref))))
    resid = float(np.max(np.abs(p_ao.T @ s_ref @ p_ao - s_ref)))
    return resid <= tol * scale


def ccm_symmetry_fold_kpair_plan(
    ccm, sym: "CCMSymmetry", kpts: np.ndarray, kept_q, ubasis, aux_basis,
    *, tol: float = 1e-10,
) -> CCMFoldKPairPlan:
    """Symmetry-unique ``(k_a, k_a + q)`` star of the per-q fold build set.

    The fold (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi_fold`)
    runs one ``build_lpq_bloch_native_fft`` per ``(kept channel q, mesh
    point k_a)``. Under a cluster-invariant space-group op ``g = {R|t}``
    (and/or time reversal) the infinite- or symmetry-closed-list fit tensors
    of related momentum pairs are linear images of each other, so one
    representative per orbit can replace explicit builds. The finite Bloch
    cell list is an additional boundary, and it is what vibeqc#337 broke
    on: a relation is admitted only when its op also maps the TRUNCATED cell
    list onto itself, which
    :func:`ccm_symmetry_op_preserves_cell_list` decides geometrically from the
    op's per-atom lattice shifts. That test replaces the earlier
    ``all(nrep > 1)`` shape guard, which was neither necessary (it disabled
    exactly-covariant 3-D stars) nor sufficient (it admitted lower-dimensional
    relations through shift-varying ops on nothing but shape).

    The momentum relations used by the planner are (see the reconstruction
    identity inlined in ``neutral.py``):

    * ``L(k_a + G0, k_b + G0) == L(k_a, k_b)`` for a reciprocal-lattice
      ``G0`` applied to BOTH momenta -- exact (measured 1.4e-14): the
      builder depends on the momentum transfer ``q = k_b - k_a`` (mesh,
      metric, kernel) and on ``k_ket`` only through Bloch phases
      ``e^{i k.R}`` with lattice ``R``, which are ``G0``-periodic.
    * time reversal ``L(-k_a, -k_b) == conj(L(k_a, k_b))`` -- exact
      (measured 3.0e-14): real AOs make every FT ingredient conjugate
      under momentum negation, and the G-ball is inversion-symmetric.
    * the space-group covariance at EXACTLY rotated momenta
      ``(R k_a, R k_b)`` -- exact for the infinite or symmetry-closed list,
      and exact on the finite list for the ops this planner admits (measured
      8.8e-13 chain / 3.8e-10 diamond at a converged list). Ops that shift
      atoms by differing lattice vectors are exact only in the infinite-list
      limit and are refused; see
      :func:`ccm_symmetry_op_preserves_cell_list`.

    The shared fit builder now truncates the physical ``|G+q|`` support,
    removing the former finite-ball "crescent" difference for a ket-side-only
    reciprocal shift while preserving time reversal at Nyquist channels. The
    present star planner deliberately keeps its narrower
    pre-fix inventory: the rotated channel ``R q`` (or ``-R q``) must coincide
    with a kept representative exactly in cartesian coordinates, and only the
    bra momentum may be brought back to the mesh by a common ``G0``. Relaxing
    that planner condition can expose more optional reductions, but is a
    performance change rather than part of the operator-correctness fix.

    Replica-mesh dimensionality is no longer part of the decision. A fully
    3-D mesh reduces whenever its cluster-invariant group contains
    cell-list-preserving ops, and a lower-dimensional one does not reduce
    through an op that fails the test. The remaining, deliberately unclaimed
    ground is the converse half of that test: a shift-varying relation is
    still exact once the pair density has decayed to nothing within the shift
    of the truncation boundary, which is why the vibeqc#337 residuals vanish at
    25-30 bohr. Admitting those needs the effective support per shell pair and
    would raise the achievable reduction further.

    Parameters
    ----------
    ccm : CCMSystem
    sym : CCMSymmetry
        Cluster-invariant subgroup (its ops map the BvK k-mesh onto
        itself modulo ``G``, which is necessary for any match).
    kpts : (n_k, 3) float
        The fold's cartesian mesh, in its own ordering.
    kept_q : sequence of int
        Indices into ``kpts`` of the channels the fold keeps (one per
        ``+-q`` pair, in fold iteration order).
    ubasis, aux_basis
        Unit-cell orbital and (modrho) auxiliary BasisSets -- the bases
        the fold's builder calls use; their AO rotation matrices are what
        the reconstruction applies.
    tol : float
        Cartesian match tolerance (bohr^-1) for the exact-channel test.

    Returns
    -------
    CCMFoldKPairPlan
    """
    from ...symmetry_ao import atom_permutation_under_op, build_ao_permutation_matrix

    unit = ccm.unit_system
    L_u = np.asarray(unit.lattice, dtype=float)
    b_lat = 2.0 * np.pi * np.linalg.inv(L_u).T
    kpts = np.asarray(kpts, dtype=float)
    kept_q = [int(q) for q in kept_q]
    n_k = len(kpts)
    n_builds = len(kept_q) * n_k

    # Unit-cell action of each cluster-invariant op. Every op is a crystal
    # space-group op (spglib on the unit cell), so the unit-cell atom
    # permutation always exists; ops whose rotation does not map the mesh
    # onto itself simply never match below.
    #
    # vibeqc#337: an op may relate the momenta exactly and still not
    # relate the TRUNCATED builds, because the origin-centred Bloch cell list
    # is only closed under the op when its per-atom lattice shifts agree
    # (see ccm_symmetry_op_preserves_cell_list for the reindexing). Ops that
    # fail that test are kept in ``ops`` -- so cached op indices stay stable --
    # but never enter a reconstruction.
    #
    # Two independent preconditions have to hold before a relation is usable,
    # and BOTH are verified rather than assumed:
    #   1. the op preserves the finite Bloch cell list (vibeqc#337, above);
    #   2. its AO/auxiliary rotations really are symmetry actions of the bases
    #      (ccm_symmetry_op_ao_map_is_exact -- upstream Euler extraction is
    #      known to fail this on non-cubic lattices).
    from vibeqc import compute_overlap

    s_ao = np.asarray(compute_overlap(ubasis), dtype=float)
    s_aux = np.asarray(compute_overlap(aux_basis), dtype=float)

    ops: list = []
    admissible: list[int] = []
    for op in sym.invariant_ops:
        R = np.asarray(op.R_cart, dtype=float)
        t = np.asarray(op.t_cart, dtype=float)
        ap = atom_permutation_under_op(unit, R, t)
        p_ao = np.asarray(build_ao_permutation_matrix(ubasis, R, ap), float)
        p_aux = np.asarray(build_ao_permutation_matrix(aux_basis, R, ap), float)
        s_cart = np.asarray(ap.lattice_shift, dtype=float) @ L_u.T
        ops.append((R, np.asarray(ap.perm, dtype=int), s_cart, p_ao, p_aux))
        if (ccm_symmetry_op_preserves_cell_list(s_cart)
                and ccm_symmetry_op_ao_map_is_exact(p_ao, s_ao)
                and ccm_symmetry_op_ao_map_is_exact(p_aux, s_aux)):
            admissible.append(len(ops) - 1)

    def _exact_channel(qp: np.ndarray):
        """Kept-channel index whose momentum equals qp EXACTLY (cartesian)."""
        for qj in kept_q:
            if float(np.max(np.abs(kpts[qj] - qp))) < tol:
                return qj
        return None

    def _mesh_congruent(kap: np.ndarray):
        """Mesh index aj with kap = kpts[aj] + G0, G0 reciprocal (exact)."""
        fr = np.linalg.solve(b_lat, kap)
        for aj in range(n_k):
            d = fr - np.linalg.solve(b_lat, kpts[aj])
            if float(np.max(np.abs(d - np.round(d)))) < 1e-8:
                return aj
        return None

    entries: dict = {}
    n_reps = 0
    for qi in kept_q:
        for ai in range(n_k):
            if (qi, ai) in entries:
                continue
            entries[(qi, ai)] = ("build",)
            n_reps += 1
            ka = kpts[ai]
            kb = kpts[ai] + kpts[qi]
            for oi in admissible:
                R = ops[oi][0]
                for trs in (False, True):
                    kap, kbp = R @ ka, R @ kb
                    if trs:
                        kap, kbp = -kap, -kbp
                    qj = _exact_channel(kbp - kap)
                    if qj is None:
                        continue
                    aj = _mesh_congruent(kap)
                    if aj is None or (qj, aj) in entries:
                        continue
                    entries[(qj, aj)] = ("recon", (qi, ai), oi, trs)

    return CCMFoldKPairPlan(
        n_builds=n_builds, n_reps=n_reps,
        reduction_factor=n_builds / max(n_reps, 1),
        entries=entries, ops=ops, admissible_ops=tuple(admissible),
    )


def ccm_symmetry_invariance_residuals(ccm, sym: "CCMSymmetry | None" = None) -> dict:
    """Max ``‖Pᵀ M P - M‖_F`` over cluster-invariant ops, per matrix.

    Returns ``{"overlap", "kinetic", "nuclear", "hcore", "n_ops"}``. ``overlap``
    and ``kinetic`` are the correctness gate for the AO maps -- ~0 to machine
    precision. ``nuclear`` (and hence ``hcore``) is non-zero: the union
    three-center ``V^CCM`` breaks crystal symmetry (the documented finding); a
    symmetric three-center ``V`` is the open item that would zero it.
    """
    from .integrals import ccm_kinetic, ccm_overlap
    from .padded import ccm_nuclear

    if sym is None:
        sym = analyze_ccm_symmetry(ccm)
    S = np.asarray(ccm_overlap(ccm), dtype=float)
    T = np.asarray(ccm_kinetic(ccm), dtype=float)
    V = np.asarray(ccm_nuclear(ccm), dtype=float)
    h = T + V

    def _max_resid(M):
        return max((float(np.linalg.norm(op.P.T @ M @ op.P - M))
                    for op in sym.invariant_ops), default=0.0)

    return {"overlap": _max_resid(S), "kinetic": _max_resid(T),
            "nuclear": _max_resid(V), "hcore": _max_resid(h),
            "n_ops": sym.cluster_order}
