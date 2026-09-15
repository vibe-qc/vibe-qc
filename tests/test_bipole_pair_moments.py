"""Per-pair adjoined-center moments (BIPOLE-EXACT-ZONE increment 2b).

The polynomial shift from global-origin moments to per-pair centers is
exact algebra; these tests pin it mechanically against an independent
libint computation at a different global origin, plus the adjoined
center formula and monopole invariance.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    compute_multipole_moments_lattice,
)
from vibeqc.bipole_pair_moments import (
    adjoined_pair_centers,
    pair_center_moments,
)

ANG2BOHR = 1.0 / 0.529177210903


def _lih_cell():
    a = 4.084 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [a / 2, a / 2, a / 2])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _moments(system, basis, cutoff, origin=(0.0, 0.0, 0.0)):
    lo = LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    return compute_multipole_moments_lattice(basis, system, lo, 2, tuple(origin))


def test_adjoined_centers_formula_and_symmetry():
    system, basis = _lih_cell()
    M = _moments(system, basis, 6.0)
    centers = adjoined_pair_centers(basis, list(M.cells))
    shells = list(basis.shells())
    a = [min(sh.exponents) for sh in shells]
    A = [np.asarray(sh.origin, dtype=float) for sh in shells]

    # Home cell: hand-computed center for one off-diagonal pair.
    home = next(
        i for i, c in enumerate(M.cells)
        if np.allclose(np.asarray(c.index), 0)
    )
    i1, i2 = 0, len(shells) - 1
    C_expect = (a[i1] * A[i1] + a[i2] * A[i2]) / (a[i1] + a[i2])
    np.testing.assert_allclose(centers[home][i1, i2], C_expect, atol=1e-14)

    # A displaced cell shifts only the KET side of the weighted mean.
    disp = next(
        i for i, c in enumerate(M.cells)
        if not np.allclose(np.asarray(c.index), 0)
    )
    g = np.asarray(M.cells[disp].r_cart, dtype=float)
    C_g = (a[i1] * A[i1] + a[i2] * (A[i2] + g)) / (a[i1] + a[i2])
    np.testing.assert_allclose(centers[disp][i1, i2], C_g, atol=1e-13)

    # Diagonal same-shell pairs in the home cell sit AT the shell origin.
    np.testing.assert_allclose(centers[home][i1, i1], A[i1], atol=1e-14)


def test_shift_algebra_exact_against_independent_origin():
    """Force every pair center to a single point O' and compare against a
    fresh libint computation ABOUT O' -- the shift must reproduce every
    component of every block to machine precision."""
    system, basis = _lih_cell()
    O2 = np.array([0.7, -1.3, 2.1])
    M0 = _moments(system, basis, 6.0, origin=(0.0, 0.0, 0.0))
    M2 = _moments(system, basis, 6.0, origin=tuple(O2))

    pm = pair_center_moments(M0, basis)
    # Recompute with all centers pinned to O2 by monkey-substitution:
    # rerun the shift on a copy whose centers are all O2.
    import vibeqc.bipole_pair_moments as mod

    orig = mod.adjoined_pair_centers
    try:
        mod.adjoined_pair_centers = lambda b, cells: [
            np.broadcast_to(
                O2, (len(list(b.shells())), len(list(b.shells())), 3)
            ).copy()
            for _ in cells
        ]
        pm_O2 = pair_center_moments(M0, basis)
    finally:
        mod.adjoined_pair_centers = orig

    for c in range(len(pm_O2.cells)):
        for comp in range(10):
            np.testing.assert_allclose(
                pm_O2.blocks[c][comp],
                np.asarray(M2.blocks[c][comp], dtype=float),
                rtol=0,
                atol=5e-12,
                err_msg=f"cell {c} comp {comp}",
            )

    # And the real per-pair result keeps monopoles bit-identical.
    for c in range(len(pm.cells)):
        np.testing.assert_array_equal(
            pm.blocks[c][0], np.asarray(M0.blocks[c][0], dtype=float)
        )


def test_rejects_wrong_l_max():
    system, basis = _lih_cell()
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 6.0
    M1 = compute_multipole_moments_lattice(basis, system, lo, 1, (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="input L_max=1 < target L_target=2"):
        pair_center_moments(M1, basis)


def test_shift_point_mass_exact_through_l4():
    """Implementation-independent pin of the shift algebra through L=4:
    for a discrete weighted point set, every Cartesian moment about any
    center is computable in closed form, so the shifted moments must
    match sum_w w*(r_w - D)^p exactly. This catches any coefficient or
    sign error in the octupole/hexadecapole binomial slot patterns
    (which double-counted the translation pre-fix, review B4) without
    relying on libint or on the code under test."""

    def canonical(order):
        return [
            (i, j, order - i - j)
            for i in range(order, -1, -1)
            for j in range(order - i, -1, -1)
        ]

    comps = [e for order in range(5) for e in canonical(order)]
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(6, 3))
    w = rng.normal(size=6)

    def mom(exps, shift=np.zeros(3)):
        r = pts - shift
        return float(
            np.sum(w * (r[:, 0] ** exps[0]) * (r[:, 1] ** exps[1]) * (r[:, 2] ** exps[2]))
        )

    # Single s-shell basis (nbf = 1) so the shell-pair slicing collapses
    # to one block; the shift algebra itself is basis-independent.
    from types import SimpleNamespace

    system = vq.PeriodicSystem(3, np.eye(3) * 20.0, [vq.Atom(1, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    assert basis.nbasis == 1

    D = np.array([0.7, -1.3, 2.1])
    M0 = SimpleNamespace(
        nbf=1,
        L_max=4,
        spherical=False,
        origin=(0.0, 0.0, 0.0),
        cells=[SimpleNamespace(index=(0, 0, 0), r_cart=(0.0, 0.0, 0.0))],
        blocks=[[np.array([[mom(e)]]) for e in comps]],
    )

    import vibeqc.bipole_pair_moments as mod

    orig = mod.adjoined_pair_centers
    try:
        mod.adjoined_pair_centers = lambda b, cells: [
            np.broadcast_to(D, (1, 1, 3)).copy() for _ in cells
        ]
        pm = mod.pair_center_moments(M0, basis, L_target=4)
    finally:
        mod.adjoined_pair_centers = orig

    for comp, e in enumerate(comps):
        np.testing.assert_allclose(
            pm.blocks[0][comp][0, 0],
            mom(e, shift=D),
            rtol=0,
            atol=1e-12,
            err_msg=f"comp {comp} exps {e}",
        )


def test_l4_spherical_input_fails_closed():
    """Raw sphemultipole blocks are SPHERICAL components; consuming them
    as Cartesian silently mixes bases (review B4, measured rel-14
    corruption of even l <= 2). The public function must refuse and
    name the sound route (the trace-merge with exact emultipole3)."""
    system, basis = _lih_cell()
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 6.0
    M4 = compute_multipole_moments_lattice(basis, system, lo, 4, (0.0, 0.0, 0.0))
    assert getattr(M4, "spherical", False)
    with pytest.raises(ValueError, match="SPHERICAL components"):
        pair_center_moments(M4, basis, L_target=4)


def _merged_l4(system, basis, lo, origin=(0.0, 0.0, 0.0)):
    """The production merge: exact Cartesian L<=3 + spherical L=4."""
    from types import SimpleNamespace

    from vibeqc._sph_to_cart import spherical_to_cartesian_with_traces

    cart_L3 = compute_multipole_moments_lattice(basis, system, lo, 3, origin)
    sph_L4 = compute_multipole_moments_lattice(basis, system, lo, 4, origin)
    merged = spherical_to_cartesian_with_traces(sph_L4, cart_L3, basis.nbasis)
    return (
        SimpleNamespace(
            nbf=basis.nbasis,
            L_max=4,
            spherical=False,
            cells=list(cart_L3.cells),
            origin=origin,
            blocks=merged,
        ),
        sph_L4,
    )


def test_l4_merged_route_preserves_low_order_moments():
    """The production L=4 route (trace-merge, then shift) must reproduce
    the machine-exact l <= 3 pair moments of the pure-Cartesian
    emultipole3 route -- the merge takes its low-degree components
    verbatim from emultipole3, so any deviation is contamination.
    This is the pass-gate that replaced the review-B4 strict xfail
    (the xfail pinned the OLD silent basis-mixing path, now refused)."""
    system, basis = _lih_cell()
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 6.0
    M3 = compute_multipole_moments_lattice(basis, system, lo, 3, (0.0, 0.0, 0.0))
    pm3 = pair_center_moments(M3, basis, L_target=3)
    merged, _ = _merged_l4(system, basis, lo)
    pm4 = pair_center_moments(merged, basis, L_target=4)
    for c in range(len(pm3.cells)):
        for comp in range(20):
            np.testing.assert_allclose(
                np.asarray(pm4.blocks[c][comp]),
                np.asarray(pm3.blocks[c][comp]),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"cell {c} comp {comp}",
            )


def test_l4_merged_route_shifts_spherical_content_exactly():
    """Gold pin of the merged route's l = 4 physics: pin every pair
    center to a second origin O2 (established monkeypatch pattern), and
    require the l = 4 SPHERICAL projection of the shifted degree-4
    Cartesian components to equal an INDEPENDENT libint sphemultipole
    computation about O2 (converted from libint's Perez-Jorda & Yang
    normalization to the module convention; see _sph_to_cart).
    Model-trace freedom in the merge lives in the kernel of the
    constraint this asserts; a wrong shift, a projection-breaking
    merge, or a normalization mismatch cannot pass."""
    from vibeqc._cart_to_sph import cartesian_to_spherical_matrix
    from vibeqc._sph_to_cart import _L4_LIBINT_TO_MODULE

    system, basis = _lih_cell()
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 6.0
    O2 = np.array([0.6, -0.9, 1.4])
    merged, _ = _merged_l4(system, basis, lo)
    sph_at_O2 = compute_multipole_moments_lattice(
        basis, system, lo, 4, tuple(O2)
    )

    import vibeqc.bipole_pair_moments as mod

    orig = mod.adjoined_pair_centers
    try:
        mod.adjoined_pair_centers = lambda b, cells: [
            np.broadcast_to(
                O2, (len(list(b.shells())), len(list(b.shells())), 3)
            ).copy()
            for _ in cells
        ]
        pm_O2 = pair_center_moments(merged, basis, L_target=4)
    finally:
        mod.adjoined_pair_centers = orig

    C4 = cartesian_to_spherical_matrix(4)
    # l = 4 spherical rows (16..24) act only on degree-4 Cartesian
    # components (20..34): C4 is degree-block-diagonal.
    C_l4 = C4[16:25, 20:35]
    for c in range(len(pm_O2.cells)):
        cart4 = np.stack(
            [np.asarray(pm_O2.blocks[c][comp]) for comp in range(20, 35)],
            axis=0,
        )
        proj = np.einsum("sc,cij->sij", C_l4, cart4)
        ref = np.stack(
            [
                np.asarray(sph_at_O2.blocks[c][comp], dtype=float)
                for comp in range(16, 25)
            ],
            axis=0,
        ) * _L4_LIBINT_TO_MODULE[:, None, None]
        np.testing.assert_allclose(
            proj, ref, rtol=1e-9, atol=1e-11, err_msg=f"cell {c}"
        )
