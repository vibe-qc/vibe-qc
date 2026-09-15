"""Space-group symmetry of the cyclic cluster (Task 3, spglib).

``analyze_ccm_symmetry`` finds the crystal space group (spglib) and the subgroup
of operations that leave the cyclic cluster invariant, building the AO
real-solid-harmonic rotation matrix ``P`` (atom permutation × Wigner-D) for each.

Pinned here:
* the AO maps are correct — ``S^CCM`` and ``T^CCM`` are invariant ``Pᵀ M P = M`` to
  machine precision under every cluster-invariant op, including non-symmorphic
  (glide/screw) operations of diamond (Fd-3m);
* the cluster-invariant subgroup order (full crystal group for an equal-``nrep``
  cubic cluster);
* the proven **finding** that the union three-center ``V^CCM`` breaks crystal
  symmetry (so ``h^CCM`` is not fully symmetric) — the open item gating the exact
  symmetry-exploitation gate.

Reference: spglib (Togo); reuses ``vibeqc.symmetry_ao`` (Wigner-D AO rotation).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import (
    CCMSystem,
    analyze_ccm_symmetry,
    ccm_symmetry_basis_rotations,
    ccm_symmetry_invariance_residuals,
    ccm_symmetry_unique_atom_pairs,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_A = 1.0 / 0.529177210903


# Every cderi gate in this module pins a SYMMETRY relation -- reduced ==
# unreduced, or P_aux^T L P == L -- in which the RSGDF high-|G| tail is
# common mode: it enters both sides identically, so it changes nothing these
# tests measure and only multiplies their cost. Since IID 307 the builders
# auto-size that tail whenever 10*zeta_max exceeds ke_cutoff, and these gates
# deliberately run BELOW the 200 Ha default for speed, so they would silently
# acquire one: diamond at ke=100 -> 787.8 Ha (measured 2.8 s -> >900 s), MgO
# at ke=100 -> 3291.6 Ha (a 189x reciprocal ball). Opting out explicitly with
# 0 keeps these gates at the cost and the calibrated residuals they were
# written for. Absolute-accuracy parity against the tailed Hamiltonian is a
# different question, gated in tests/test_ccm_neutral_tail_parity.py.
_NO_TAIL = 0.0


def _fcc(a):
    return 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float).T


def _diamond(nrep=(1, 1, 1)):
    lat = _fcc(3.567 * _A)
    atoms = [Atom(6, [0, 0, 0]), Atom(6, (lat @ [0.25, 0.25, 0.25]).tolist())]
    return CCMSystem(PeriodicSystem(3, lat, atoms), nrep, "sto-3g")


def _mgo(nrep=(1, 1, 1)):
    lat = _fcc(4.21 * _A)
    atoms = [Atom(12, [0, 0, 0]), Atom(8, (lat @ [0.5, 0.5, 0.5]).tolist())]
    return CCMSystem(PeriodicSystem(3, lat, atoms), nrep, "sto-3g")


def _lih(nrep=(1, 1, 1)):
    lat = _fcc(7.72)
    atoms = [Atom(3, [0, 0, 0]), Atom(1, (lat @ [0.5, 0.5, 0.5]).tolist())]
    return CCMSystem(PeriodicSystem(3, lat, atoms), nrep, "sto-3g")


def _compact_h2(nrep=(1, 1, 1)):
    """Compact physical 3-D H2 cell for neutral-route SCF gates."""
    lat = np.diag([6.0, 6.0, 6.0])
    atoms = [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])]
    return CCMSystem(PeriodicSystem(3, lat, atoms, 0, 1), nrep, "sto-3g")


@pytest.fixture(scope="module")
def mgo_sym():
    ccm = _mgo()
    return ccm, analyze_ccm_symmetry(ccm)


@pytest.fixture(scope="module")
def diamond_sym():
    ccm = _diamond()
    return ccm, analyze_ccm_symmetry(ccm)


def test_spacegroup_identified(mgo_sym, diamond_sym):
    _, m = mgo_sym
    _, d = diamond_sym
    assert m.number == 225 and m.international_symbol == "Fm-3m"   # rocksalt
    assert d.number == 227 and d.international_symbol == "Fd-3m"   # diamond


def test_cluster_invariant_subgroup_full_for_cubic(mgo_sym, diamond_sym):
    """Equal-nrep cubic cluster keeps the full crystal point group (48 ops)."""
    _, m = mgo_sym
    _, d = diamond_sym
    assert m.cluster_order == m.crystal_order == 48
    assert d.cluster_order == d.crystal_order == 48


def test_diamond_has_nonsymmorphic_ops(diamond_sym):
    """Fd-3m is non-symmorphic — its glide/screw ops carry fractional translations."""
    _, d = diamond_sym
    assert sum(o.nonsymmorphic for o in d.invariant_ops) > 0


@pytest.mark.parametrize("fixture", ["mgo_sym", "diamond_sym"])
def test_ao_maps_leave_overlap_and_kinetic_invariant(fixture, request):
    """The AO maps are correct: S^CCM and T^CCM are invariant to machine precision."""
    ccm, sym = request.getfixturevalue(fixture)
    res = ccm_symmetry_invariance_residuals(ccm, sym)
    assert res["overlap"] < 1e-12
    assert res["kinetic"] < 1e-12


@pytest.mark.parametrize("fixture", ["mgo_sym", "diamond_sym"])
def test_union_nuclear_breaks_crystal_symmetry(fixture, request):
    """FINDING (pinned): the union three-center V^CCM is NOT crystal-symmetric.

    The anchor-bridge ½(ω_μC+ω_νC) breaks spatial symmetry (~5e-4 MgO, ~8e-3
    diamond relative). A symmetric three-center V (analogous to the §13 symmetric
    four-center) would zero this — when it lands, update this test.
    """
    ccm, sym = request.getfixturevalue(fixture)
    res = ccm_symmetry_invariance_residuals(ccm, sym)
    assert res["nuclear"] > 1e-5            # symmetry is genuinely broken
    assert res["hcore"] >= res["nuclear"] - 1e-12   # h inherits V's breaking


@pytest.mark.parametrize("fixture", ["mgo_sym", "diamond_sym"])
def test_neutral_ewald_nuclear_is_crystal_symmetric(fixture, request):
    """RESOLUTION: the neutral Ewald V_ne IS crystal-symmetric (the symmetric V).

    The bare-1/r union V breaks symmetry (previous test); the neutral Ewald V_ne
    — the GDF-route nuclear — restores it to lattice-sum tolerance. The spatial
    analogue of the Madelung gap (#16): the neutral kernel fixes both permutation
    and spatial symmetry. So symmetry exploitation rides the neutral/GDF route.
    """
    import numpy as np

    from vibeqc import LatticeSumOptions, bloch_sum
    from vibeqc.periodic_rhf_gdf import _gauge_lat_opts_for_v_ne_and_e_nuc
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

    ccm, sym = request.getfixturevalue(fixture)
    glat = _gauge_lat_opts_for_v_ne_and_e_nuc(LatticeSumOptions(), ccm.cluster_system)
    Vn = np.real(bloch_sum(
        compute_nuclear_lattice_dispatch(ccm.basis, ccm.cluster_system, glat),
        np.zeros(3)))
    resid = max((float(np.linalg.norm(o.P.T @ Vn @ o.P - Vn)) for o in sym.invariant_ops),
                default=0.0)
    assert resid / np.linalg.norm(Vn) < 1e-6   # symmetric to lattice-sum tolerance


def test_unique_atom_pairs_tile_and_reduce(mgo_sym):
    """Pair orbits partition all n² ordered pairs; reduction factor ≥ 1."""
    ccm, sym = mgo_sym
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)
    assert sum(orb.multiplicity) == orb.n_total == ccm.n_atoms ** 2
    assert orb.n_unique <= orb.n_total
    assert orb.reduction_factor >= 1.0


def test_unique_atom_pairs_reduce_for_larger_cluster():
    """A 2×2×2 cubic cluster reduces the atom-pair count by the point group."""
    lat = _fcc(4.21 * _A)
    ccm = CCMSystem(PeriodicSystem(3, lat,
                    [Atom(12, [0, 0, 0]), Atom(8, (lat @ [0.5, 0.5, 0.5]).tolist())]),
                    (2, 2, 2), "sto-3g")
    sym = analyze_ccm_symmetry(ccm)
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)
    assert sum(orb.multiplicity) == ccm.n_atoms ** 2     # tiles
    assert orb.reduction_factor > 4.0                    # non-trivial point-group reduction
    # The orbit map reconstructs every member from its representative:
    # (A,B) == (g.perm[A_rep], g.perm[B_rep]) with g = invariant_ops[orbit_op].
    n = ccm.n_atoms
    n_reps_seen = 0
    for A in range(n):
        for B in range(n):
            ra, rb = orb.representatives[int(orb.orbit_rep[A, B])]
            g = int(orb.orbit_op[A, B])
            if g < 0:
                assert (ra, rb) == (A, B)   # sentinel only on the rep itself
                n_reps_seen += 1
            else:
                p = sym.invariant_ops[g].perm
                assert (int(p[ra]), int(p[rb])) == (A, B)
    assert n_reps_seen == orb.n_unique


def test_p1_triclinic_orbits_are_trivial():
    """|G_c| = 1 safety: a generic triclinic cell claims no reduction.

    Every pair is its own orbit representative (reduction_factor == 1, all
    ``orbit_op`` sentinels), so a symmetry-routed integral build degenerates
    to the full build with no reconstruction step and no reduction claim.
    """
    lat = np.array([[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]]).T
    ccm = CCMSystem(
        PeriodicSystem(3, lat, [Atom(1, [0.1, 0.2, 0.3]), Atom(2, [2.0, 3.1, 4.2])]),
        (1, 1, 1), "sto-3g")
    sym = analyze_ccm_symmetry(ccm)
    assert sym.cluster_order == 1          # identity only
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)
    assert orb.n_unique == orb.n_total == ccm.n_atoms ** 2
    assert orb.reduction_factor == 1.0
    assert np.all(orb.orbit_op == -1)


def _flat_lat_opts(cutoff_bohr):
    """Flat lattice-sum options with a widened Bloch cell-list cutoff.

    The pair covariance of the computed cderi is exact only up to two
    numerical floors:

    * the truncated inter-cell tail of the Bloch sum (the cell list enters
      pair equivalence through atom-shifted copies of itself, so a finite
      radius leaves an orbit-asymmetric tail). Measured on the 3c tensor
      ``T``, C-diamond sto-3g at Γ: 8e-9 relative at the 15-bohr default,
      2e-15 at 20 bohr, 6e-16 at 25 bohr -- converged away here;
    * the whitening amplification of that residual noise through the
      ill-conditioned aux metric: ``L = M^{-1/2} T`` scales the leftover
      ~1e-15 by ``||M^{-1/2}|| ~ linear_dep_thr^{-1/2} ~ 3e4``, leaving a
      ~2e-10-relative elementwise floor on ``L`` that no cutoff removes.
      It lives in near-null aux directions and does not reach physical
      contractions (SCF energy diff measured 5.5e-13 Ha, pinned < 1e-10).

    The tensor-level gates therefore pin < 1e-9 (cell-list-converged,
    dominated by the whitening floor, measured ~2e-10); the energy gate
    pins < 1e-10 at production defaults.
    """
    from vibeqc import LatticeSumOptions

    o = LatticeSumOptions()
    o.cutoff_bohr = float(cutoff_bohr)
    return o


def test_symmetry_reduced_cderi_equals_unreduced_diamond(diamond_sym):
    """GATE: the pair-reduced neutral cderi == the un-reduced build, < 1e-10.

    Fd-3m diamond (1,1,1): the two carbons are equivalent, so the four ordered
    atom pairs reduce to two orbits (reduction 2x) and the reconstruction runs
    through non-symmorphic (glide) ops. At a cell-list-converged cutoff the
    reduced tensor agrees with the full canonical-frame build elementwise
    (machine precision), and hence on every contraction. At the production
    default the difference is the truncation floor -- pinned in
    ``test_symmetry_reduced_cderi_default_cutoff_floor``.
    """
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

    ccm, sym = diamond_sym
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)
    assert orb.n_unique < orb.n_total          # reduction actually exercised
    opts = _flat_lat_opts(25.0)
    l_full = ccm_neutral_cderi(ccm, ke_cutoff=100.0, lat_opts=opts,
                               tail_ke_cutoff=_NO_TAIL,
                               canonical_auxiliary_basis=True)
    l_red = ccm_neutral_cderi(ccm, ke_cutoff=100.0, lat_opts=opts,
                              tail_ke_cutoff=_NO_TAIL, symmetry=sym)
    assert l_red.shape == l_full.shape
    scale = float(np.max(np.abs(l_full)))
    # < 1e-9: whitening-amplification floor, see _flat_lat_opts (meas. 2.2e-10)
    assert float(np.max(np.abs(l_red - l_full))) / scale < 1e-9


def test_symmetry_reduced_cderi_equals_unreduced_mgo_multicell():
    """GATE: pair reduction across cluster cells (ionic Fm-3m, (2,1,1)).

    On the (2,1,1) MgO cluster the surviving point ops map atoms between the
    two cells (e.g. inversion sends cell 1 to cell -1 ≡ 1 mod the cluster
    lattice), so orbits mix cells and the reconstruction exercises the
    cross-cell pair blocks. Cell-list-converged cutoff (see
    :func:`_flat_lat_opts`); wide enough to also cover the torus wrap.
    """
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

    lat = _fcc(4.21 * _A)
    ccm = CCMSystem(PeriodicSystem(3, lat,
                    [Atom(12, [0, 0, 0]), Atom(8, (lat @ [0.5, 0.5, 0.5]).tolist())]),
                    (2, 1, 1), "sto-3g")
    sym = analyze_ccm_symmetry(ccm)
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)
    assert orb.n_unique < orb.n_total
    opts = _flat_lat_opts(25.0)
    l_full = ccm_neutral_cderi(ccm, ke_cutoff=80.0, lat_opts=opts,
                               canonical_auxiliary_basis=True)
    l_red = ccm_neutral_cderi(ccm, ke_cutoff=80.0, lat_opts=opts,
                              symmetry=sym)
    scale = float(np.max(np.abs(l_full)))
    # < 1e-9: whitening-amplification floor, see _flat_lat_opts
    assert float(np.max(np.abs(l_red - l_full))) / scale < 1e-9


def test_symmetry_reduced_cderi_diamond_multicell_regression():
    """REGRESSION (two upstream bugs, both caught by this exact case, 2026-07-10):

    1. ``analyze_ccm_symmetry`` admitted crystal ops that do not map the
       cluster (BvK) lattice onto itself -- the atom-position test passes
       for EVERY crystal op (any image site is congruent to some cluster
       site mod the cluster lattice), so 36 of the 48 fcc ops survived on
       the (2,1,1) cluster while genuinely breaking the torus integrals
       (P^T S P vs S residual 4.7). The lattice-normalization condition
       (W_ij n_j / n_i integer) now filters them: |G_c| = 12 here.
    2. ``rsgdf_dense_g_mesh`` sized its index box from ROW norms of the
       lattice matrix (columns are the vectors): correct for symmetric
       matrices (fcc primitive -- every historical control), but on the
       non-symmetric (2,1,1) cluster matrix it silently clipped 18% of
       the |G| <= G_max ball, breaking the mesh's point-group closure
       (4e-3 covariance error, ke-dependent, cutoff-resistant).

    With both fixed, the pair-reduced build matches the un-reduced one at
    production defaults on the case that exposed them.
    """
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

    ccm = _diamond((2, 1, 1))
    sym = analyze_ccm_symmetry(ccm)
    assert sym.cluster_order == 12          # 36 non-normalizing ops filtered
    res = ccm_symmetry_invariance_residuals(ccm, sym)
    assert res["overlap"] < 1e-12 and res["kinetic"] < 1e-12
    l_full = ccm_neutral_cderi(ccm, ke_cutoff=100.0,
                               canonical_auxiliary_basis=True)
    l_red = ccm_neutral_cderi(ccm, ke_cutoff=100.0, symmetry=sym)
    scale = float(np.max(np.abs(l_full)))
    assert float(np.max(np.abs(l_red - l_full))) / scale < 1e-9


def test_dense_g_mesh_is_complete_ball_on_nonsymmetric_lattice():
    """REGRESSION: rsgdf_dense_g_mesh covers the FULL |G| <= G_max ball on a
    non-symmetric lattice matrix (the (2,1,1) fcc cluster cell; the wrong-axis
    index-box bound clipped 602 of 3381 points here, 2026-07-10)."""
    from vibeqc.aux_basis import rsgdf_dense_g_mesh

    ccm = _diamond((2, 1, 1))
    sysc = ccm.cluster_system
    a = np.asarray(sysc.lattice, dtype=float)
    b = 2.0 * np.pi * np.linalg.inv(a).T
    g_max = float(np.sqrt(2.0 * 60.0))
    # Reference: generously over-sized box, exact ball criterion.
    n = [int(np.ceil(g_max * np.linalg.norm(a[:, i]) / (2 * np.pi))) + 2
         for i in range(3)]
    grids = [np.arange(-m, m + 1) for m in n]
    i1, i2, i3 = np.meshgrid(*grids, indexing="ij")
    idx = np.stack([i1.ravel(), i2.ravel(), i3.ravel()], -1).astype(float)
    g_ref = idx @ b.T
    n_ball = int((np.linalg.norm(g_ref, axis=1) <= g_max).sum())
    assert len(rsgdf_dense_g_mesh(sysc, 60.0)) == n_ball


def test_symmetry_reduced_cderi_default_cutoff_floor(diamond_sym):
    """PINNED FLOOR: at the production-default cell list the reduced build
    differs from the un-reduced one by the truncation tail only.

    The un-reduced build itself carries this tail (it is the builder's
    documented lattice-cutoff accuracy floor); symmetry reconstruction
    neither amplifies it (pinned < 5e-8 relative here, measured 4.6e-9)
    nor leaks into the energy (pinned < 1e-10 Ha in
    ``test_symmetry_reduced_cderi_scf_energy_matches``, measured 5.5e-13
    on this system).
    """
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

    ccm, sym = diamond_sym
    l_full = ccm_neutral_cderi(ccm, ke_cutoff=100.0,
                               tail_ke_cutoff=_NO_TAIL,
                               canonical_auxiliary_basis=True)
    l_red = ccm_neutral_cderi(ccm, ke_cutoff=100.0,
                              tail_ke_cutoff=_NO_TAIL, symmetry=sym)
    scale = float(np.max(np.abs(l_full)))
    assert float(np.max(np.abs(l_red - l_full))) / scale < 5e-8
    e_full = run_ccm_rhf_direct(ccm, cderi=l_full).energy
    e_red = run_ccm_rhf_direct(ccm, cderi=l_red).energy
    assert abs(e_red - e_full) < 1e-10


def test_symmetry_reduced_cderi_p1_falls_back_to_full():
    """GATE (low-symmetry safety): |G_c| = 1 falls back to the full build.

    With no reduction available the symmetry route must return the plain
    build (same code path, no reconstruction, no reduction claim) -- in the
    canonical aux frame, which every ``symmetry=...`` call returns so the
    frame is predictable downstream.
    """
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

    lat = np.array([[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]]).T
    ccm = CCMSystem(
        PeriodicSystem(3, lat, [Atom(1, [0.1, 0.2, 0.3]), Atom(2, [2.0, 3.1, 4.2])]),
        (1, 1, 1), "sto-3g")
    l_plain = ccm_neutral_cderi(ccm, ke_cutoff=60.0,
                                tail_ke_cutoff=_NO_TAIL,
                                canonical_auxiliary_basis=True)
    l_sym = ccm_neutral_cderi(ccm, ke_cutoff=60.0,
                              tail_ke_cutoff=_NO_TAIL, symmetry=True)
    assert np.array_equal(l_sym, l_plain)


def test_symmetry_reduced_cderi_scf_energy_matches():
    """GATE: the direct-torus SCF energy from the reduced cderi == un-reduced.

    End-to-end consumer check on a compact physical 3-D H2 cell with a (2,1,1)
    cyclic cluster (cheap, with cross-cell symmetry orbits).
    """
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi

    ccm = _compact_h2((2, 1, 1))
    sym = analyze_ccm_symmetry(ccm)
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)
    assert orb.n_unique < orb.n_total
    l_full = ccm_neutral_cderi(ccm, canonical_auxiliary_basis=True)
    l_red = ccm_neutral_cderi(ccm, symmetry=sym)
    e_full = run_ccm_rhf_direct(ccm, cderi=l_full).energy
    e_red = run_ccm_rhf_direct(ccm, cderi=l_red).energy
    assert abs(e_red - e_full) < 1e-10


def test_symmetry_kwarg_ri_neutral_scf_energy_matches():
    """GATE: ``run_ccm_rhf_ri_neutral``'s ``symmetry=`` kwarg reaches the cderi
    builder and leaves the SCF energy invariant.

    Companion to :func:`test_symmetry_reduced_cderi_scf_energy_matches`, which
    passes *pre-built* cderis to ``run_ccm_rhf_direct``. Here the driver builds
    its **own** neutral cderi with ``symmetry=sym`` vs ``None`` (``cderi=None``
    both times), so it exercises the ``run_ccm_rhf_ri_neutral(symmetry=...)``
    plumbing end to end -- the same knob ``run_ccm_rhf_direct`` exposes as
    ``cderi_symmetry``. Compact physical 3-D H2 with a (2,1,1) cyclic cluster:
    non-trivial subgroup with cross-cell orbits, and ``ccm_neutral_cderi``
    reduces at this mesh.
    """
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_ri_neutral

    ccm = _compact_h2((2, 1, 1))
    sym = analyze_ccm_symmetry(ccm)
    orb = ccm_symmetry_unique_atom_pairs(ccm, sym)
    assert orb.n_unique < orb.n_total          # reduction actually exercised
    e_none = run_ccm_rhf_ri_neutral(ccm).energy
    e_sym = run_ccm_rhf_ri_neutral(ccm, symmetry=sym).energy
    assert abs(e_sym - e_none) < 1e-10


@pytest.mark.parametrize("fixture", ["mgo_sym", "diamond_sym"])
def test_neutral_cderi_gamma_covariance(fixture, request):
    """FOUNDATION (pinned): the Γ canonical-frame neutral cderi is space-group
    covariant — the identity that makes orbit-representative-only fitting exact.

    For every cluster-invariant op g (aux rotation P_aux, orbital rotation P):

        L_{P, gμ, gν} = Σ_{P'} (P_aux)_{P P'} L_{P', μ, ν}
        ⇔  L = einsum("QP,Qab,am,bn->Pmn", P_aux, L, P, P)

    Holds to machine precision at a cell-list-converged cutoff: the FFT
    G-ball, the Coulomb kernel, and the per-shell calibrations are exactly
    point-group invariant, and the whitening M^{-1/2} (canonical aux frame)
    commutes with P_aux since the metric M is invariant (2e-14 measured).
    The one non-invariant ingredient is the truncated Bloch cell-list tail
    (see :func:`_flat_lat_opts`), converged away here. Non-symmorphic ops
    (diamond glides) are covered: at Γ the fractional-translation phases
    cancel between the aux and pair Fourier factors.
    """
    from vibeqc.periodic.ccm.neutral import _supercell_modrho_aux, ccm_neutral_cderi

    ccm, sym = request.getfixturevalue(fixture)
    L = ccm_neutral_cderi(ccm, ke_cutoff=100.0, lat_opts=_flat_lat_opts(25.0),
                          tail_ke_cutoff=_NO_TAIL,
                          canonical_auxiliary_basis=True)
    rots = ccm_symmetry_basis_rotations(sym, _supercell_modrho_aux(ccm))
    nrm = float(np.linalg.norm(L))
    worst = 0.0
    for op, paux in zip(sym.invariant_ops, rots):
        recon = np.einsum("QP,Qab,am,bn->Pmn", paux, L, op.P, op.P, optimize=True)
        worst = max(worst, float(np.linalg.norm(recon - L)) / nrm)
    # < 1e-9: whitening-amplification floor, see _flat_lat_opts
    assert worst < 1e-9


def test_low_symmetry_chain_finds_subgroup():
    """A 1-D H₂ chain (orthorhombic) yields a non-trivial subgroup with correct maps."""
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 30.0, 30.0]),
                                   [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], 0, 1),
                    (2, 1, 1), "sto-3g")
    sym = analyze_ccm_symmetry(ccm)
    assert sym.cluster_order > 1
    res = ccm_symmetry_invariance_residuals(ccm, sym)
    assert res["overlap"] < 1e-12 and res["kinetic"] < 1e-12


# ---------------------------------------------------------------------------
# M4 -- fold k-pair star reduction
# ---------------------------------------------------------------------------

def _chain(nrep):
    return CCMSystem(PeriodicSystem(3, np.diag([6.0, 30.0, 30.0]),
                                    [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])],
                                    0, 1), nrep, "sto-3g")


def _fold_pair(ccm, **kw):
    from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold

    l_full = ccm_neutral_cderi_fold(ccm, **kw)
    l_red = ccm_neutral_cderi_fold(ccm, symmetry=True, **kw)
    assert l_red.shape == l_full.shape
    return l_full, l_red


def _fold_plan(ccm):
    """The star plan exactly as the fold computes it (kept ±q channels)."""
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.aux_basis import (
        default_aux_for,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic.ccm.symmetry import ccm_symmetry_fold_kpair_plan
    from vibeqc.periodic_k_gdf import _kmesh_to_kpoints_weights

    unit = ccm.unit_system
    mol = unit.unit_cell_molecule()
    ubasis = BasisSet(mol, ccm.basis_name)
    aux = make_aux_basis_set(mol, aux_name=default_aux_for(ccm.basis_name),
                             drop_eta=0.0)
    aux_modrho = make_modrho_aux_basis(aux, mol)
    kpts, _ = _kmesh_to_kpoints_weights(unit, list(ccm.nrep))
    kpts = np.asarray(kpts, dtype=float)
    a_lat = np.asarray(unit.lattice, dtype=float)
    b_lat = 2.0 * np.pi * np.linalg.inv(a_lat).T
    nrep = np.asarray(ccm.nrep, dtype=int)
    frac = np.round(np.linalg.solve(b_lat, kpts.T).T * nrep[None, :]).astype(int)
    frac %= nrep
    kept, seen = [], set()
    for qi in range(len(kpts)):
        qf = tuple(frac[qi])
        if qf in seen:
            continue
        kept.append(qi)
        seen.add(qf)
        seen.add(tuple((-frac[qi]) % nrep))
    sym = analyze_ccm_symmetry(ccm)
    return ccm_symmetry_fold_kpair_plan(ccm, sym, kpts, kept, ubasis,
                                        aux_modrho), kpts


def test_fold_star_exact_relations_only_on_small_meshes():
    """The conservative (2,1,1) star plan still claims no reduction.

    The builder now uses mod-G-invariant shifted support, but widening the
    optional star inventory is deliberately deferred. The planner continues
    to require an exactly matching cartesian channel, so all four builds are
    their own representatives and the symmetry-routed fold must return the
    un-reduced result bit-for-bit through the same build path.
    """
    ccm = _chain((2, 1, 1))
    plan, _ = _fold_plan(ccm)
    assert plan.n_builds == 4 and plan.n_reps == 4
    l_full, l_red = _fold_pair(ccm, ke_cutoff=60.0)
    assert np.array_equal(l_red, l_full)


def test_fold_star_fully_3d_falls_back_to_full_build():
    """IID 337: do not apply finite-cell-list covariance on a 3-D star.

    LiH/STO-3G (2,2,2) previously admitted 11 representatives for 64
    required tensors. At the production 15-bohr Bloch cutoff some of the 53
    reconstructions differ from independent builds at the percent level,
    moving the SCF energy by 1.9e-5 Ha/cell. Keep every pair explicit until
    the builder makes its finite cell list covariant under those operations.
    """
    plan, _ = _fold_plan(_lih((2, 2, 2)))
    assert plan.n_builds == plan.n_reps == 64
    assert plan.reduction_factor == 1.0
    assert len(plan.entries) == 64
    assert all(entry == ("build",) for entry in plan.entries.values())
    assert plan.ops == []


def test_fold_star_chain_311_reduces_and_matches():
    """(3,1,1) chain: the q=0 channel folds under mirror/TRS (5/6 builds).

    First mesh where two distinct builds are exactly related: on the q=0
    channel the mirror x -> -x (equivalently TRS) maps (k, k) onto
    (-k, -k) = (2k, 2k) - b, a common reciprocal shift, which the builder
    is exactly periodic under. The reconstruction path runs and the fold
    output matches elementwise to machine precision (measured 1.6e-16;
    the chain's ops carry no atom shifts, so this pins the rotation + TRS
    conjugation wiring in isolation from the phase factors).
    """
    ccm = _chain((3, 1, 1))
    plan, _ = _fold_plan(ccm)
    assert (plan.n_builds, plan.n_reps) == (6, 5)
    assert any(e[0] == "recon" for e in plan.entries.values())
    l_full, l_red = _fold_pair(ccm, ke_cutoff=60.0)
    scale = float(np.max(np.abs(l_full)))
    assert float(np.max(np.abs(l_red - l_full))) / scale < 1e-12


def test_production_gdf_pair_cache_uses_ccm_star():
    """The production-order ``(ki,kj)`` cache builds one representative less.

    The (3,1,1) chain has one exact time-reversal relation among its nine
    Lpq pairs.  This lightweight covariant tensor pins the route adapter,
    production pair indexing, canonical-frame request, and measured work
    without repeating the expensive native-fit parity tests above.
    """
    from vibeqc import make_basis
    from vibeqc.aux_basis import (
        default_aux_for,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic.ccm.ri import _ccm_gdf_symmetry_cache_builder
    from vibeqc.periodic_k_gdf import _kmesh_to_kpoints_weights

    ccm = _chain((3, 1, 1))
    unit = ccm.unit_system
    mol = unit.unit_cell_molecule()
    basis = make_basis(mol, ccm.basis_name)
    aux = make_aux_basis_set(
        mol, aux_name=default_aux_for(ccm.basis_name), drop_eta=0.0
    )
    aux_modrho = make_modrho_aux_basis(aux, mol)
    kpts, _ = _kmesh_to_kpoints_weights(unit, list(ccm.nrep))
    calls = []

    def _tensor(ki, kj):
        # Reciprocal-periodic and time-reversal covariant for this s-only AO
        # chain: f(-ki,-kj) = conj(f(ki,kj)).
        phase = np.exp(1j * 6.0 * (float(ki[0]) + float(kj[0])))
        return phase * np.ones(
            (aux_modrho.nbasis, basis.nbasis, basis.nbasis), dtype=complex
        )

    def _build(ki, kj, *, canonical_auxiliary_basis=False):
        calls.append(bool(canonical_auxiliary_basis))
        return _tensor(ki, kj)

    stats = {}
    cache = _ccm_gdf_symmetry_cache_builder(
        ccm, True, stats
    )(_build, kpts, basis, aux_modrho, True)

    assert len(cache) == 9
    assert stats == {"builds": 8, "total": 9, "factor": 9 / 8}
    assert len(calls) == 8
    assert all(calls)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(cache[(i, j)], _tensor(kpts[i], kpts[j]))


def test_production_gdf_fully_3d_bypasses_pair_star_adapter(monkeypatch):
    """IID 337: the public GDF route bypasses the unsafe pair-star adapter.

    The generic RSGDF driver selects its shared-q full-build implementation
    when no private cache adapter is present. This seam test pins the adapter
    absence and public 64/64 accounting; the slow LiH test below pins the
    resulting energy rather than mocking the cache implementation itself.
    """
    from types import SimpleNamespace

    import vibeqc
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    seen = {}

    def _fake_driver(_unit, _basis, _kmesh, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(converged=True, energy=-1.0, n_iter=1)

    monkeypatch.setattr(vibeqc, "run_krhf_periodic_gdf", _fake_driver)
    result = run_ccm_rhf_gdf(_lih((2, 2, 2)), symmetry=True)

    assert "_lpq_cache_builder" not in seen
    assert result.gdf_pair_builds == result.gdf_pair_total == 64
    assert result.gdf_pair_reduction_factor == 1.0


@pytest.mark.slow
def test_production_gdf_symmetry_reduced_scf_matches():
    """GATE: GDF reduces real 3-D fit work without moving the SCF energy."""
    from vibeqc import PeriodicKSOptions, PeriodicRHFOptions
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rks_gdf

    ccm = _compact_h2((3, 1, 1))
    options = PeriodicRHFOptions()
    options.max_iter = 64
    options.conv_tol_energy = 1e-10
    options.damping = 0.0
    plain = run_ccm_rhf_gdf(
        ccm,
        symmetry=False,
        rsgdf_ke_cutoff=60.0,
        options=options,
    )
    reduced = run_ccm_rhf_gdf(
        ccm,
        symmetry=True,
        rsgdf_ke_cutoff=60.0,
        options=options,
    )

    assert plain.converged and reduced.converged
    assert reduced.energy == pytest.approx(plain.energy, abs=1e-12)
    assert plain.gdf_pair_builds == plain.gdf_pair_total == 9
    assert reduced.gdf_pair_builds == 8
    assert reduced.gdf_pair_total == 9
    assert reduced.gdf_pair_reduction_factor == pytest.approx(9 / 8)

    ks_options = PeriodicKSOptions()
    ks_options.max_iter = 64
    ks_options.conv_tol_energy = 1e-9
    ks_options.damping = 0.0
    hybrid_plain = run_ccm_rks_gdf(
        ccm,
        "pbe0",
        symmetry=False,
        rsgdf_ke_cutoff=60.0,
        options=ks_options,
    )
    hybrid_reduced = run_ccm_rks_gdf(
        ccm,
        "pbe0",
        symmetry=True,
        rsgdf_ke_cutoff=60.0,
        options=ks_options,
    )
    assert hybrid_plain.converged and hybrid_reduced.converged
    assert hybrid_reduced.energy == pytest.approx(
        hybrid_plain.energy, abs=1e-12
    )
    assert hybrid_reduced.gdf_pair_builds == 8
    assert hybrid_reduced.gdf_pair_total == 9


@pytest.mark.slow
def test_production_gdf_fully_3d_fallback_matches_unreduced():
    """IID 337: default LiH (2,2,2) GDF is the exact unreduced result."""
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    ccm = _lih((2, 2, 2))
    guarded = run_ccm_rhf_gdf(ccm, symmetry=True)
    plain = run_ccm_rhf_gdf(ccm, symmetry=False)

    assert guarded.converged and plain.converged
    assert guarded.gdf_pair_builds == guarded.gdf_pair_total == 64
    assert plain.gdf_pair_builds == plain.gdf_pair_total == 64
    assert guarded.energy == pytest.approx(plain.energy, abs=1e-10)


def test_fold_star_reduced_equals_unreduced_diamond_221():
    """GATE: star-reduced fold == un-reduced fold on diamond (2,2,1).

    The reconstruction-exercising case: 10/16 builder calls (1.60x), with
    non-symmorphic (glide) ops, nonzero atom shifts, and cross-channel
    star members. At a cell-list-converged cutoff the difference is the
    whitening floor (measured 8.0e-11 at 25 bohr; 5.6e-9 at the 15-bohr
    default -- the same Bloch-truncation floor the M3 supercell gates pin).
    """
    ccm = _diamond((2, 2, 1))
    plan, _ = _fold_plan(ccm)
    assert plan.n_reps < plan.n_builds        # reduction actually exercised
    assert any(e[0] == "recon" and not e[3] for e in plan.entries.values())
    opts = _flat_lat_opts(25.0)
    l_full, l_red = _fold_pair(ccm, ke_cutoff=100.0, lat_opts=opts)
    scale = float(np.max(np.abs(l_full)))
    # < 1e-9: whitening-amplification floor, see _flat_lat_opts
    assert float(np.max(np.abs(l_red - l_full))) / scale < 1e-9


def test_fold_star_reduced_equals_unreduced_mgo_221():
    """GATE: star-reduced fold == un-reduced fold on ionic MgO (2,2,1).

    Steeper cores than diamond (larger default-cutoff Bloch tail: 1.5e-7
    at 15 bohr) -- converged away at 20 bohr to the whitening floor
    (measured 3.3e-10, cutoff-resistant, near-null aux directions only).
    """
    ccm = _mgo((2, 2, 1))
    plan, _ = _fold_plan(ccm)
    assert plan.n_reps < plan.n_builds
    opts = _flat_lat_opts(20.0)
    l_full, l_red = _fold_pair(ccm, ke_cutoff=100.0, lat_opts=opts)
    scale = float(np.max(np.abs(l_full)))
    # < 1e-9: whitening-amplification floor, see _flat_lat_opts
    assert float(np.max(np.abs(l_red - l_full))) / scale < 1e-9


def test_fold_star_regression_diamond_211():
    """REGRESSION (mandatory case -- it exposed both 2026-07-10 upstream bugs).

    On the (2,1,1) diamond cluster the BvK filter keeps |G_c| = 12 and the
    G-mesh must be the full |G| <= G_max ball (the rsgdf_dense_g_mesh
    clipping fix) for ANY star relation to be exact. The (2,1,1) mesh
    itself admits no exact relation (see the small-mesh test above), so
    the contract here is: correct no-reduction claim + bit-identical
    output through the symmetry-routed path.
    """
    ccm = _diamond((2, 1, 1))
    sym = analyze_ccm_symmetry(ccm)
    assert sym.cluster_order == 12
    plan, _ = _fold_plan(ccm)
    assert plan.n_builds == plan.n_reps == 4
    l_full, l_red = _fold_pair(ccm, ke_cutoff=100.0)
    assert np.array_equal(l_red, l_full)


def test_fold_star_p1_falls_back_to_full():
    """GATE (low-symmetry safety): P1 keeps identity + TRS only, no claim.

    The current conservative planner does not consume the builder's new
    mod-G transfer equivalence, so this P1 (2,1,1) fold still runs the full
    build with no reduction claim and bit-identical output.
    """
    lat = np.array([[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]]).T
    ccm = CCMSystem(
        PeriodicSystem(3, lat, [Atom(1, [0.1, 0.2, 0.3]), Atom(2, [2.0, 3.1, 4.2])]),
        (2, 1, 1), "sto-3g")
    sym = analyze_ccm_symmetry(ccm)
    assert sym.cluster_order == 1
    plan, _ = _fold_plan(ccm)
    assert plan.n_builds == plan.n_reps
    l_full, l_red = _fold_pair(ccm, ke_cutoff=60.0)
    assert np.array_equal(l_red, l_full)


def test_fold_star_plan_relations_are_exact():
    """Every recon entry's momenta satisfy the exactness contract.

    For entry (qj, aj) <- (rep (qi, ai), op g, trs): the (optionally
    negated) rotated channel +-R q_rep must equal kpts[qj] exactly, and
    the rotated bra +-R k_rep must equal kpts[aj] up to a reciprocal
    lattice vector (the common-shift direction, which is builder-exact).
    """
    ccm = _diamond((2, 2, 1))
    plan, kpts = _fold_plan(ccm)
    a_lat = np.asarray(ccm.unit_system.lattice, dtype=float)
    b_lat = 2.0 * np.pi * np.linalg.inv(a_lat).T
    n_recon = 0
    for (qj, aj), e in plan.entries.items():
        if e[0] != "recon":
            continue
        n_recon += 1
        (qi, ai), oi, trs = e[1], e[2], e[3]
        R = plan.ops[oi][0]
        sgn = -1.0 if trs else 1.0
        qp = sgn * (R @ kpts[qi])
        kap = sgn * (R @ kpts[ai])
        assert float(np.max(np.abs(qp - kpts[qj]))) < 1e-10       # exact channel
        d = np.linalg.solve(b_lat, kap - kpts[aj])
        assert float(np.max(np.abs(d - np.round(d)))) < 1e-8      # common G0
    assert n_recon == plan.n_builds - plan.n_reps > 0


def test_fold_star_scf_energy_matches_diamond_221():
    """GATE: direct-torus SCF energy from the star-reduced fold == un-reduced,
    at production defaults (ke_cutoff=200, default cell list), on the
    reconstruction-exercising diamond (2,2,1) case. Also pins the
    default-cutoff tensor floor (the un-reduced fold's own Bloch-truncation
    accuracy, measured 5.6e-9 relative at ke=100; < 5e-8 here)."""
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct

    ccm = _diamond((2, 2, 1))
    l_full, l_red = _fold_pair(ccm)
    scale = float(np.max(np.abs(l_full)))
    assert float(np.max(np.abs(l_red - l_full))) / scale < 5e-8
    e_full = run_ccm_rhf_direct(ccm, cderi=l_full).energy
    e_red = run_ccm_rhf_direct(ccm, cderi=l_red).energy
    assert abs(e_red - e_full) < 1e-10


def test_fold_star_scf_energy_matches_compact_h2_311():
    """GATE: end-to-end SCF on a compact 3-D (3,1,1) star (5/6 builds)."""
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct

    ccm = _compact_h2((3, 1, 1))
    plan, _ = _fold_plan(ccm)
    assert (plan.n_builds, plan.n_reps) == (6, 5)
    l_full, l_red = _fold_pair(ccm)
    e_full = run_ccm_rhf_direct(ccm, cderi=l_full).energy
    e_red = run_ccm_rhf_direct(ccm, cderi=l_red).energy
    assert abs(e_red - e_full) < 1e-10


def test_direct_driver_cderi_symmetry_passthrough():
    """The production drivers reach the star through ``cderi_symmetry=``.

    ``run_ccm_rhf_direct(cderi_build="fold", cderi_symmetry=True)`` must
    reproduce the plain driver energy -- this is the route the aiccm-a
    production runs use, so without the passthrough the reduction is
    unreachable where it matters.
    """
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct

    ccm = _compact_h2((3, 1, 1))
    e_plain = run_ccm_rhf_direct(ccm, ke_cutoff=60.0).energy
    e_sym = run_ccm_rhf_direct(ccm, ke_cutoff=60.0,
                               cderi_symmetry=True).energy
    assert abs(e_sym - e_plain) < 1e-10
