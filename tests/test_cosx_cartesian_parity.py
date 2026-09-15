"""COSX exchange must reproduce exact ERI exchange for Cartesian shells too.

COSX (chain-of-spheres) builds K from two halves that have to live in the
same AO basis:

* grid-collocated AO values, from ``evaluate_ao`` (cpp/src/cosx.cpp), and
* analytic nuclear-attraction-like integrals, from the Obara-Saika kernel
  (cpp/src/cosx_kernel.cpp).

Until 2026-08-05 both halves multiplied Cartesian components by
``sqrt((2l-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!))`` to force unit norm per
component. libint does no such thing -- it normalises a shell with one
factor from the total ``l``, so ``<xy|xy> = 1/3`` on a d shell -- and the
density matrix K is contracted against comes from libint's basis. The
halves were fixed in two separate commits, which left a window where they
disagreed with each other; this test pins them together against a source of
truth neither one shares: the exact four-index ERI.

The spherical case is included as a control. It always passed, which is
exactly the problem -- every named basis forces ``set_pure(true)``
(cpp/src/basis.cpp), so no production calculation could see the Cartesian
defect.

Reference for the method: Neese, F. et al., Chem. Phys. 356, 98 (2009).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    GridOptions,
    Molecule,
    build_cosx_q,
    build_cosx_schwarz,
    build_grid,
    compute_cosx_k,
    compute_eri,
)
from vibeqc import _vibeqc_core as core
from vibeqc._vibeqc_core import BasisSet, ShellInfo


def _cosx_grid_options() -> GridOptions:
    """A tighter-than-default COSX grid.

    The production default (35/9/18) carries ~1e-3 relative grid error,
    which would swamp the effect under test. At 75/17/34 the spherical
    control lands near 1e-8, so a Cartesian convention error of tens of
    percent is unmistakable.
    """
    g = GridOptions()
    g.angular = "product"
    g.n_radial = 75
    g.n_theta = 17
    g.n_phi = 34
    return g


def _shell(atom_index: int, l: int, *, pure: bool, alpha: float, origin) -> ShellInfo:
    s = ShellInfo()
    s.atom_index = atom_index
    s.l = l
    s.pure = pure
    s.exponents = [alpha]
    s.coefficients = [1.0]
    s.origin = list(origin)
    return s


def _h2_with_d_shell(pure: bool) -> tuple[Molecule, BasisSet]:
    """H2 with an s shell on each centre plus one d shell, pure or Cartesian.

    Explicit ``ShellInfo`` is the only route to a Cartesian basis: the
    named-basis constructor forces ``set_pure(true)``.
    """
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 1)
    basis = BasisSet(
        mol,
        [
            _shell(0, 0, pure=True, alpha=1.2, origin=[0.0, 0.0, 0.0]),
            _shell(1, 0, pure=True, alpha=1.2, origin=[0.0, 0.0, 1.4]),
            _shell(0, 2, pure=pure, alpha=0.8, origin=[0.0, 0.0, 0.0]),
        ],
        "<cosx-cart-parity>",
        False,
    )
    return mol, basis


def _exact_k(basis: BasisSet, density: np.ndarray) -> np.ndarray:
    """K_mu,nu = sum_{la,si} D_la,si (mu la | si nu), from the exact ERI."""
    eri = np.asarray(compute_eri(basis))
    return np.einsum("ls,mlsn->mn", density, eri)


def _cosx_k(mol: Molecule, basis: BasisSet, density: np.ndarray) -> np.ndarray:
    grid = build_grid(mol, _cosx_grid_options())
    cutoffs = core.compute_shell_radial_cutoffs(basis, 1e-12)
    q = build_cosx_q(basis, grid)
    schwarz = build_cosx_schwarz(basis)
    return np.asarray(
        compute_cosx_k(basis, density, grid, q, schwarz, cutoffs, None)
    )


def _symmetric_density(n: int) -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    a = rng.standard_normal((n, n))
    return 0.5 * (a + a.T)


@pytest.mark.parametrize("pure", [True, False], ids=["pure", "cartesian"])
def test_cosx_k_matches_exact_eri_exchange(pure):
    """COSX K agrees with exact-ERI K for both shell kinds.

    Tolerance is the grid error, not the convention error. With the
    per-component factor reintroduced, the Cartesian case lands at 2.4e-1
    (analytic half only), 8.6e-2 (grid half only), or 4.5e-2 (both halves
    -- the pre-fix state) -- all orders of magnitude outside this bound,
    while the spherical control stays at 4.1e-8 in every state (measured
    2026-08-06, handovers/HANDOVER_AO_CONVENTION.md).
    """
    mol, basis = _h2_with_d_shell(pure)
    n_bf = basis.nbasis
    assert n_bf == (7 if pure else 8)

    density = _symmetric_density(n_bf)
    k_exact = _exact_k(basis, density)
    k_cosx = _cosx_k(mol, basis, density)

    scale = np.abs(k_exact).max()
    assert scale > 1e-6, "degenerate test system"
    rel = np.abs(k_exact - k_cosx).max() / scale
    assert rel < 1e-5, (
        f"COSX K deviates from exact-ERI K by {rel:.3e} (relative, "
        f"{'pure' if pure else 'Cartesian'} d shell). This bound is the "
        "grid error; a failure here at 1e-2 or worse means COSX's analytic "
        "half (cosx_kernel.cpp) and its grid half (evaluate_ao, via "
        "cosx.cpp) disagree on the Cartesian normalization convention."
    )


def test_cosx_cartesian_error_is_not_masked_by_a_loose_tolerance():
    """Guards the guard: the Cartesian case must be as tight as the pure one.

    If the Cartesian result were quietly worse -- say 1e-3 -- the parametrized
    test above could still pass while hiding a partial convention error. Pin
    them to the same order of magnitude instead.
    """
    errs = {}
    for pure in (True, False):
        mol, basis = _h2_with_d_shell(pure)
        density = _symmetric_density(basis.nbasis)
        k_exact = _exact_k(basis, density)
        k_cosx = _cosx_k(mol, basis, density)
        errs[pure] = np.abs(k_exact - k_cosx).max() / np.abs(k_exact).max()

    assert errs[False] < 25.0 * max(errs[True], 1e-12), (
        f"Cartesian COSX error {errs[False]:.3e} is disproportionate to the "
        f"spherical error {errs[True]:.3e}; the two shell kinds should be "
        "limited by the same grid quality, not by a convention mismatch."
    )
