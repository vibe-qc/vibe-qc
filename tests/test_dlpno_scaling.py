"""DLPNO-MP2 sparse pair lists — O(N) pair growth (M3c step 2).

Dynamic correlation is short-ranged, so the number of *significant*
occupied pairs grows linearly with system size, not quadratically. The
default distant-pair screen (``tcut_pairs``) realises that: pairs whose
semicanonical dipole-dipole estimate is negligible are treated at the
estimate level (``e_distant``) instead of with the full PNO machinery.

Two properties:

1. **No-op on compact molecules** — everything is within
   ``dipole_r_min``, so the default screen changes nothing (the
   exactness gates in ``test_dlpno_mp2`` are unaffected).
2. **Sub-quadratic on extended systems** — on an H2 chain, the kept-pair
   count grows linearly while the total grows quadratically, at
   negligible cost to the recovered correlation energy.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from vibeqc import BasisSet, MP2Options, RHFOptions, run_mp2, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.mp2 import DLPNOMP2Options as _DLPNOMP2Options, run_dlpno_mp2

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"

# Historical scaling evidence was measured with the pre-#448 conservative
# screen and the pre-#140 all-electron space. Keep that input explicit.
DLPNOMP2Options = partial(
    _DLPNOMP2Options,
    n_frozen=0,
    tcut_pno=1e-8,
    tcut_pno_weak=1e-7,
    tcut_mkn=1e-3,
    tcut_pairs=1e-6,
    tcut_pairs_weak=1e-4,
)

H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]


def _system(atoms, basis_name="def2-svp"):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    mp2_opts = MP2Options()
    mp2_opts.n_frozen_core = 0
    mp2_opts.density_fit = True
    mp2_opts.aux_basis = AUX
    e_ref = run_mp2(mol, basis, rhf, mp2_opts).e_correlation
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    return mol, basis, rhf, df, e_ref


def _h2_chain(n, spacing=5.6, bond=1.4):
    atoms = []
    for k in range(n):
        z0 = spacing * k
        atoms += [(1, [0.0, 0.0, z0]), (1, [0.0, 0.0, z0 + bond])]
    return atoms


class TestPreUnificationScreeningDefault:
    def test_pre_unification_default_is_on_but_conservative(self):
        o = DLPNOMP2Options()
        assert o.tcut_pairs > 0.0          # screening on by default
        assert o.tcut_pairs <= 1e-6        # but conservative
        assert o.dipole_r_min >= 8.0       # never screens close pairs

    def test_compact_molecule_is_no_op(self):
        """The pre-unification screen changes nothing on a compact molecule."""
        mol, basis, rhf, df, _ = _system(H2O_ATOMS)
        on = run_dlpno_mp2(mol, basis, rhf, df, DLPNOMP2Options(localise="boys"))
        off = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", tcut_pairs=0.0),
        )
        assert on.n_pairs_screened == 0
        assert abs(on.e_corr - off.e_corr) < 1e-12


class TestPairListScaling:
    """Kept pairs grow linearly while the total grows quadratically."""

    @pytest.fixture(scope="class")
    def chains(self):
        out = {}
        for n in (4, 8):
            mol, basis, rhf, df, e_ref = _system(_h2_chain(n))
            r = run_dlpno_mp2(mol, basis, rhf, df, DLPNOMP2Options(localise="boys"))
            out[n] = (r, e_ref, n * (n + 1) // 2)
        return out

    def test_kept_pairs_grow_subquadratically(self, chains):
        r4, _, total4 = chains[4]
        r8, _, total8 = chains[8]
        total_ratio = total8 / total4          # ~3.6 (quadratic)
        kept_ratio = r8.n_pairs / r4.n_pairs    # ~2.3 (linear)
        assert kept_ratio < total_ratio
        # Genuinely closer to linear (×2) than quadratic (×4).
        assert kept_ratio < 0.75 * total_ratio

    def test_screening_active_and_growing(self, chains):
        r4 = chains[4][0]
        r8 = chains[8][0]
        assert r4.n_pairs_screened >= 1
        assert r8.n_pairs_screened > r4.n_pairs_screened
        # Every screened pair still contributes through e_distant.
        assert r8.e_distant <= 0.0

    def test_recovery_preserved_under_screening(self, chains):
        """The historical screen costs almost nothing vs the unscreened run."""
        for n in (4, 8):
            r, e_ref, _ = chains[n]
            mol, basis, rhf, df, _ = _system(_h2_chain(n))
            unscreened = run_dlpno_mp2(
                mol, basis, rhf, df,
                DLPNOMP2Options(localise="boys", tcut_pairs=0.0),
            )
            assert abs(r.e_corr - unscreened.e_corr) < 1e-4
