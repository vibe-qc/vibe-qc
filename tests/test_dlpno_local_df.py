"""DLPNO-MP2 local (domain-restricted) density fitting.

The reduced-scaling primitive: each pair's exchange integrals are refit
with only the auxiliary functions in its fit domain (Riplinger 2016
sparse maps), instead of the global RI metric. Three things must hold:

1. **Exactness** — a full fit domain (``fit_buffer`` ≥ system size) refits
   the 2-/3-centre integrals exactly, so it reproduces the global RI and
   hence canonical RI-MP2 to ≤ 1 µHa. The default ``local_df=False`` path
   is bit-for-bit the original.
2. **Locality** — the per-pair fit dimension is bounded by the local
   neighbourhood, *not* the system size: doubling a spatially extended
   system leaves the maximum fit dimension unchanged. That O(1)-per-pair
   fit cost is what makes DLPNO scale.
3. **Direct local integrals** — ``local_df=True`` builds restricted
   auxiliary sub-bases and evaluates their 2-/3-centre integrals directly.
   It must not slice a precomputed global raw 3-centre tensor.
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
MICRO_HA = 1e-6
FULL_DOMAIN = 1e9  # fit_buffer that forces the global-RI limit

# Preserve the pre-#140/#448 all-electron threshold convention underlying
# these local-fit exactness and locality measurements.
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
NO_TRUNCATION = dict(tcut_pno=0.0, tcut_pno_weak=0.0, tcut_mkn=0.0)


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


def _h2_chain(n, spacing=6.0, bond=1.4):
    atoms = []
    for k in range(n):
        z0 = spacing * k
        atoms += [(1, [0.0, 0.0, z0]), (1, [0.0, 0.0, z0 + bond])]
    return atoms


class _NoGlobalRawDF:
    """DF metadata proxy that fails if the global raw tensors are touched."""

    def __init__(self, df):
        self.orbital_basis = df.orbital_basis
        self.aux_basis = df.aux_basis
        self.aux_basis_name = df.aux_basis_name
        self.n_orb = df.n_orb
        self.n_aux = df.n_aux

    @property
    def three_center(self):
        raise AssertionError("local_df=True must not read global three_center")

    @property
    def metric(self):
        raise AssertionError("local_df=True must not read global metric")

    def mo_transform(self, *args, **kwargs):
        raise AssertionError("local_df=True must not use global B transforms")


class TestLocalDFExactness:
    """Full fit domain reproduces the global RI / canonical RI-MP2."""

    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        return _system(H2O_ATOMS)

    def test_full_domain_canonical_occupieds(self, h2o):
        mol, basis, rhf, df, e_ref = h2o
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="none", local_df=True,
                            fit_buffer=FULL_DOMAIN, **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA
        # Full domain ⇒ every pair fits with all aux functions.
        assert set(r.fit_dim_per_pair.values()) == {df.n_aux}

    def test_full_domain_boys_localised(self, h2o):
        mol, basis, rhf, df, e_ref = h2o
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", local_df=True,
                            fit_buffer=FULL_DOMAIN, **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA

    def test_local_full_domain_matches_global_path(self, h2o):
        """local_df full domain == the default global-RI path, exactly."""
        mol, basis, rhf, df, _ = h2o
        glob = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", local_df=False, **NO_TRUNCATION),
        )
        loc = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", local_df=True,
                            fit_buffer=FULL_DOMAIN, **NO_TRUNCATION),
        )
        assert abs(glob.e_corr - loc.e_corr) < 0.1 * MICRO_HA

    def test_local_df_builds_integrals_directly(self, h2o):
        """The local path must work without the global raw DF tensors."""
        mol, basis, rhf, df, e_ref = h2o
        r = run_dlpno_mp2(
            mol, basis, rhf, _NoGlobalRawDF(df),
            DLPNOMP2Options(localise="none", local_df=True,
                            fit_buffer=FULL_DOMAIN, **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA
        assert set(r.fit_dim_per_pair.values()) == {df.n_aux}

    def test_local_df_off_is_default(self):
        assert DLPNOMP2Options().local_df is False


class TestLocalDFLocality:
    """The fit dimension is bounded by the neighbourhood, not system size."""

    def test_max_fit_dim_is_size_independent(self):
        """Doubling the chain leaves the maximum fit dimension unchanged."""
        opts = lambda: DLPNOMP2Options(localise="boys", local_df=True,
                                       fit_buffer=4.0)
        mol4, b4, rhf4, df4, _ = _system(_h2_chain(4))
        mol8, b8, rhf8, df8, _ = _system(_h2_chain(8))
        r4 = run_dlpno_mp2(mol4, b4, rhf4, df4, opts())
        r8 = run_dlpno_mp2(mol8, b8, rhf8, df8, opts())

        max4 = max(r4.fit_dim_per_pair.values())
        max8 = max(r8.fit_dim_per_pair.values())
        # n_aux doubled...
        assert df8.n_aux == pytest.approx(2 * df4.n_aux, rel=0.05)
        # ...but the per-pair fit domain did not grow.
        assert max8 == max4
        # And it is genuinely local — well below the full aux set.
        assert max8 < 0.5 * df8.n_aux

    def test_local_df_recovers_accuracy(self):
        """Local fit at a modest buffer still recovers >99% of E_corr."""
        mol, basis, rhf, df, e_ref = _system(_h2_chain(4))
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", local_df=True, fit_buffer=4.0),
        )
        assert r.converged
        assert r.e_corr / e_ref > 0.99
        avg_fit = sum(r.fit_dim_per_pair.values()) / len(r.fit_dim_per_pair)
        assert avg_fit < df.n_aux  # actually truncating the fit

    def test_buffer_monotone_toward_exact(self):
        """Widening the fit buffer moves the energy toward the exact limit."""
        mol, basis, rhf, df, _ = _system(_h2_chain(4))
        exact = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", local_df=True,
                            fit_buffer=FULL_DOMAIN, **NO_TRUNCATION),
        ).e_corr
        errs = []
        for buf in (3.0, 6.0, 12.0):
            r = run_dlpno_mp2(
                mol, basis, rhf, df,
                DLPNOMP2Options(localise="boys", local_df=True,
                                fit_buffer=buf, **NO_TRUNCATION),
            )
            errs.append(abs(r.e_corr - exact))
        assert errs[0] >= errs[1] >= errs[2]
