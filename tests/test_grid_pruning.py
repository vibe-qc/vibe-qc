"""Angular-order pruning pins.

The NWChem 3-tier pruning rule (Murray-Handy-Laming 1993 / Gill-
Johnson-Pople SG-1 1993) drops the Lebedev angular order at radial
shells in the core and the outer tail, keeping the full requested
order only in the chemically-active middle band. ~25 % of shells
prune to the next-lower tier on the pinned historical generic ladder;
energy drift is in the µHa-floor of XC quadrature noise.

These tests pin:

1. ``angular_pruning='none'`` (the default) is a no-op — point count
   matches an unpruned Lebedev build cell-for-cell.
2. ``angular_pruning='nwchem'`` reduces total point count by ~10–25 %
   for typical Lebedev orders (29 / 35 / 41) on first-row systems.
3. RKS-B3LYP / H2O / def2-svp under pruning matches the unpruned
   result to ≤ 1e-6 Ha — well under the existing XC quadrature noise
   floor.
4. Pruning is a no-op on the smallest pruning-eligible Lebedev order (11),
   since the generic NWChem approximation deliberately floors at order 11.
5. Pruning is a no-op on the legacy product-grid path.

References (mathematical, no proprietary source consulted):
* Murray, C. W.; Handy, N. C.; Laming, G. J., *Quadrature schemes for
  integrals of density functional theory*, Mol. Phys. 78, 997 (1993).
* Gill, P. M. W.; Johnson, B. G.; Pople, J. A., *A standard grid for
  DFT calculations*, Chem. Phys. Lett. 209, 506 (1993) — SG-1.
"""
from __future__ import annotations

import pytest

from vibeqc import (
    Atom, BasisSet, GridOptions, Molecule, RKSOptions, build_grid, run_rks,
)


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_A = ANGSTROM_TO_BOHR

H2O = [(8, [0.0, 0.0, 0.0]),
       (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
       (1, [0.0, -0.793353 * _A, -0.613510 * _A])]


def _mol(geom):
    return Molecule([Atom(int(z), list(xyz)) for z, xyz in geom])


def _build_lebedev_grid(order: int, pruning: str = "none"):
    mol = _mol(H2O)
    opts = GridOptions()
    opts.angular = "lebedev"
    opts.lebedev_order = order
    opts.angular_pruning = pruning
    return build_grid(mol, opts)


def test_pruning_none_is_a_no_op():
    """Default 'none' produces the same point count as an unpruned
    Lebedev build."""
    g_default = _build_lebedev_grid(29, pruning="none")
    g_unpruned = _build_lebedev_grid(29)  # GridOptions default
    assert len(g_default.weights) == len(g_unpruned.weights)


def test_generic_pruning_point_count_ignores_profile_only_tiers():
    """Adding SKALA profile tiers must not alter an existing generic grid."""
    hydrogen = Molecule(
        [Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    opts = GridOptions()
    opts.angular = "lebedev"
    opts.lebedev_order = 29
    opts.angular_pruning = "nwchem"

    grid = build_grid(hydrogen, opts)

    # The historical generic ladder selects order 23 (194 points) below
    # order 29 (302 points).  With 75 radial shells, 29 are pruned.
    assert len(grid.weights) == 29 * 194 + 46 * 302 == 19_518


@pytest.mark.parametrize("order", [29, 35, 41])
def test_pruning_nwchem_reduces_point_count(order):
    """NWChem pruning produces strictly fewer points than the
    unpruned tier."""
    g_unpruned = _build_lebedev_grid(order)
    g_pruned = _build_lebedev_grid(order, pruning="nwchem")
    assert len(g_pruned.weights) < len(g_unpruned.weights)
    reduction = 1.0 - len(g_pruned.weights) / len(g_unpruned.weights)
    # One-tier-down rule produces 10-30 % reduction at the standard
    # orders. Lower bound is loose to allow future refinements.
    assert 0.05 <= reduction <= 0.45, (
        f"order {order}: pruning gives {reduction:.1%} reduction — "
        f"outside expected 5-45 % band")


def test_pruning_b3lyp_energy_drift_below_quadrature_floor():
    """Pruning shifts RKS-B3LYP/H2O/def2-svp by ≤ 1e-6 Ha — well below
    the 1e-5 Ha XC quadrature noise floor."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")

    def _energy(pruning: str) -> tuple[float, int]:
        o = RKSOptions()
        o.functional = "B3LYP"
        o.conv_tol_energy = 1e-9
        o.grid.angular = "lebedev"
        o.grid.lebedev_order = 29
        o.grid.angular_pruning = pruning
        r = run_rks(mol, basis, o)
        assert r.converged
        return r.energy, r.n_iter

    e_none, n_none = _energy("none")
    e_nwchem, n_nwchem = _energy("nwchem")
    delta = abs(e_none - e_nwchem)
    assert delta < 1e-6, (
        f"Pruning drift {delta:.2e} Ha exceeds 1e-6 Ha gate — outside "
        f"the XC quadrature noise floor")
    assert n_nwchem <= n_none + 2, (
        f"Pruning shouldn't significantly change iteration count "
        f"(unpruned {n_none}, pruned {n_nwchem})")


def test_pruning_no_op_on_smallest_lebedev_tier():
    """Order 11 is the generic pruner's floor, so pruning is a no-op."""
    g_unpruned = _build_lebedev_grid(11)
    g_pruned = _build_lebedev_grid(11, pruning="nwchem")
    assert len(g_pruned.weights) == len(g_unpruned.weights)


def test_pruning_no_op_on_product_grid():
    """Pruning only applies to the Lebedev path. With angular=product
    selected, setting pruning='nwchem' is silently a no-op."""
    mol = _mol(H2O)
    opts_a = GridOptions()
    opts_a.angular = "product"
    opts_a.n_theta = 9
    opts_a.n_phi = 18
    opts_a.angular_pruning = "none"
    g_a = build_grid(mol, opts_a)

    opts_b = GridOptions()
    opts_b.angular = "product"
    opts_b.n_theta = 9
    opts_b.n_phi = 18
    opts_b.angular_pruning = "nwchem"
    g_b = build_grid(mol, opts_b)

    assert len(g_a.weights) == len(g_b.weights)


def test_pruning_string_selector_accepts_case_variants():
    mol = _mol(H2O)
    for s in ("none", "NONE", "None", "nwchem", "NWChem", "NWCHEM"):
        opts = GridOptions()
        opts.angular = "lebedev"
        opts.angular_pruning = s
        # Build succeeds — selector accepted both cases.
        g = build_grid(mol, opts)
        assert len(g.weights) > 0


def test_pruning_unknown_string_raises():
    opts = GridOptions()
    with pytest.raises((ValueError, RuntimeError, Exception)):
        opts.angular_pruning = "agressive"
