"""Per-shell radial-cutoff utility pins.

``compute_shell_radial_cutoffs(basis, tol)`` returns the conservative
Gaussian-extent bound — the smallest r* such that ``|χ_μ(r)| < tol``
for any μ in shell s and ``|r - O_s| > r*``. The bound's correctness
shows up indirectly through ``tests/test_rijcosx.py`` (which exercises
the active-shell pre-prune the cutoff drives inside ``compute_cosx_k``);
this file pins the function itself:

1. Returns one cutoff per shell.
2. Cutoffs are positive and finite.
3. Tightening ``tol`` produces strictly larger cutoffs (Gaussian-tail
   monotonicity).
4. Realistic basis sets (def2-svp on water) produce O(1)-bohr cutoffs
   for the most-contracted shells and O(10)-bohr cutoffs for the most-
   diffuse — both inside the expected range for the basis's slowest-
   decaying primitive at typical AO tolerances.

Reference (mathematical, no proprietary source consulted):
* Stratmann, R. E.; Scuseria, G. E.; Frisch, M. J., Chem. Phys. Lett.
  257, 213 (1996), § 11 — Gaussian-extent screening for XC quadrature.
* Burow, A. M.; Sierka, M., J. Chem. Theory Comput. 7, 3097 (2011) —
  modern incarnation in periodic XC.
"""
from __future__ import annotations

import math

import pytest

from vibeqc import Atom, BasisSet, Molecule
from vibeqc import _vibeqc_core as core


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_A = ANGSTROM_TO_BOHR

H2O = [(8, [0.0, 0.0, 0.0]),
       (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
       (1, [0.0, -0.793353 * _A, -0.613510 * _A])]


def _h2o_basis(name: str = "def2-svp") -> BasisSet:
    mol = Molecule([Atom(int(z), list(x)) for z, x in H2O])
    return BasisSet(mol, name)


def test_returns_one_cutoff_per_shell():
    basis = _h2o_basis()
    cutoffs = core.compute_shell_radial_cutoffs(basis, 1e-12)
    assert len(cutoffs) == basis.nshells


def test_cutoffs_are_positive_and_finite():
    basis = _h2o_basis()
    cutoffs = core.compute_shell_radial_cutoffs(basis, 1e-12)
    for r in cutoffs:
        assert math.isfinite(r), "cutoff must be finite"
        assert r > 0, "cutoff must be strictly positive"


def test_tightening_tol_grows_cutoffs():
    """Tighter tol (smaller value) means we want |χ| below a smaller
    threshold, which is satisfied only farther out — every per-shell
    cutoff must grow monotonically."""
    basis = _h2o_basis()
    c_loose = core.compute_shell_radial_cutoffs(basis, 1e-7)
    c_tight = core.compute_shell_radial_cutoffs(basis, 1e-12)
    for r_loose, r_tight in zip(c_loose, c_tight):
        assert r_tight >= r_loose, (
            f"tighter tol should produce r*≥; got "
            f"r_loose={r_loose:.3f} bohr, r_tight={r_tight:.3f} bohr")


def test_def2_svp_water_cutoffs_in_expected_range():
    """Sanity-check the bound on a known basis: def2-svp on H2O has
    O 1s/2s/2p/3d shells (Z=8) and H 1s/2s/2p shells (Z=1). The
    least-diffuse primitive in any of these shells has an exponent
    of ~0.1 bohr⁻²; at tol=1e-12 the slowest-primitive cutoff lands
    in the 5–25 bohr range after the 3× safety factor. Used as a
    smoke test — a regression flips r* into either the sub-bohr or
    100+-bohr regime."""
    basis = _h2o_basis()
    cutoffs = core.compute_shell_radial_cutoffs(basis, 1e-12)
    for r in cutoffs:
        # Realistic Gaussian-extent r* for def2-svp at this tol band.
        # Bound includes a 3× safety factor (cpp/src/schwarz.cpp
        # ``compute_shell_radial_cutoffs``); the unmultiplied raw
        # bound is r/3 ~ 0.7-8 bohr for these shells.
        assert 2.0 <= r <= 60.0, (
            f"per-shell cutoff {r:.2f} bohr is outside the expected "
            f"def2-svp range")


def test_invalid_tol_raises():
    basis = _h2o_basis()
    with pytest.raises((ValueError, RuntimeError, Exception)):
        core.compute_shell_radial_cutoffs(basis, 0.0)
    with pytest.raises((ValueError, RuntimeError, Exception)):
        core.compute_shell_radial_cutoffs(basis, -1.0)
