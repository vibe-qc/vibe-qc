"""Pure-RKS exact-FT J retirement regressions (2026-07-18).

Maintainer ruling 2026-07-13 (HANDOVER_BIPOLE_PRODUCTION.md): once the M4b
interaction-resolved SR screening and the M5 padded image domain landed,
pure semilocal RKS routes its Hartree J through the same padded SR+LR erfc
composition as every other BIPOLE route, and the exact analytic-FT Ewald
builder is retired to an explicit ``use_exact_ft_j=True`` cross-check
oracle.

Pinned here:

1. The default pure-RKS route builds SR+LR (nonzero e_j_long_range, a
   resolved M5 image extent) and does not read ``VIBEQC_J_EWALD3D_KE``.
2. The opt-in oracle reproduces the historical exact-FT route (all-J in
   e_j_short_range, no erfc pad) and stays within its documented envelope.
3. The oracle fails closed off pure functionals / the corrected split /
   with the retired multipole artifact.
4. Closed-shell route consistency: pure RKS and closed-shell UKS now agree
   on the J route, so their converged energies match tightly (they
   previously differed by the exact-FT-vs-direct J route split).
5. Driver-level warning behavior: the dense-core reciprocal-tail warning
   fires only on the oracle route.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    PeriodicKSOptions,
    monkhorst_pack,
)
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks


def _h2_box(L: float = 12.0, sep: float = 1.4):
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * L,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, sep])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _mgo_cell():
    ANG2BOHR = 1.0 / 0.529177210903
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _ks_opts(functional: str = "svwn", cutoff: float = 6.0, max_iter: int = 60):
    opts = PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = cutoff
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff
    opts.max_iter = max_iter
    opts.initial_guess = InitialGuess.HCORE
    opts.use_diis = True
    opts.diis_start_iter = 1
    return opts


def _run(system, basis, opts, **kwargs):
    kmesh = monkhorst_pack(system, [1, 1, 1])
    return run_pbc_bipole_rks(
        system, basis, kmesh, opts, progress=False, **kwargs
    )


# Post-retirement H2/STO-3G/12-bohr c6 SVWN Gamma fixed points (this
# machinery, 2026-07-18). The two routes legitimately differ at the
# sub-mHa scale on this soft cell: the SR+LR value carries the padded
# erfc composition, the oracle the finite-ke analytic FT.
#
# Both moved by an IDENTICAL -2.484636e-4 Ha with the #478 fix, which is
# the point: the leak was in the shared one-electron/nuclear Ewald gauge,
# so the route difference (6.5995e-7 Ha) is preserved exactly. Pre-fix
# values were -1.120947736192 / -1.120948396145.
#
# These are the CONVERGED numbers, not merely different ones: with the
# alpha-consistent real cutoff the SR+LR value is stable to 1.5e-13 Ha
# across ewald_precision 1e-8 .. 1e-14. At the parent the corresponding
# knob was inert -- sweeping nuclear_cutoff_bohr over 6, 12.5, 18.5, 25,
# 40, 60 bohr returned a bit-identical energy, because
# clamp_bipole_nuclear_cutoff clamps it back down to cutoff_bohr. That is
# why #478 reported the residual as invariant under nuclear_cutoff_bohr.
#
# The shift here (2.48e-4 Ha) is 3.3x the asymptotic point-charge estimate
# 0.225 N_e^2 / L = 7.5e-5: at L = 12 the Ewald 1/alpha = 4.3 bohr is
# comparable to the AO-pair extent, so the erfc kernel is sampled well
# inside its steep region rather than at the image distance. The 1/L law
# is the large-box limit; on compact vacuum cells the leak is worse.
E_H2_SVWN_SRLR = -1.121196199774
E_H2_SVWN_EXACT_FT = -1.121196859728


def test_pure_rks_default_routes_srlr_with_padded_extent():
    """Default pure RKS builds SR+LR J with the resolved M5 image extent."""
    system, basis = _h2_box()
    res = _run(system, basis, _ks_opts())
    assert res.converged
    comp = res.energy_components[-1]
    # The Ewald-J split populates both components; the retired exact-FT
    # route parked the whole J in e_j_short_range with e_j_long_range == 0.
    assert comp.e_j_long_range != 0.0
    assert res.sr_image_extent_bohr is not None
    assert res.sr_image_extent_bohr > 6.0  # padded past cutoff_bohr
    assert res.energy == pytest.approx(E_H2_SVWN_SRLR, abs=5e-9)


def test_exact_ft_oracle_reproduces_historical_route():
    """use_exact_ft_j=True reproduces the pre-retirement default route."""
    system, basis = _h2_box()
    res = _run(system, basis, _ks_opts(), use_exact_ft_j=True)
    assert res.converged
    comp = res.energy_components[-1]
    assert comp.e_j_long_range == 0.0
    # No erfc J traversal -> no M5 image extent on the oracle.
    assert res.sr_image_extent_bohr is None
    assert res.energy == pytest.approx(E_H2_SVWN_EXACT_FT, abs=5e-9)
    # The two routes are close on a soft covalent cell but NOT identical.
    assert res.energy != pytest.approx(E_H2_SVWN_SRLR, abs=1e-8)


def test_exact_ft_oracle_fails_closed_on_hybrids():
    system, basis = _h2_box()
    with pytest.raises(ValueError, match="pure-functional"):
        _run(system, basis, _ks_opts(functional="pbe0"), use_exact_ft_j=True)


def test_exact_ft_oracle_fails_closed_on_screened_hybrids():
    # HSE06 has c_full == 0 (alpha_hf == 0) but needs its screened K;
    # the oracle must not silently no-op on it.
    system, basis = _h2_box()
    with pytest.raises(ValueError, match="screened"):
        _run(system, basis, _ks_opts(functional="hse06"), use_exact_ft_j=True)


def test_exact_ft_oracle_fails_closed_on_legacy_gauge():
    system, basis = _h2_box()
    with pytest.raises(ValueError, match="corrected"):
        _run(
            system,
            basis,
            _ks_opts(),
            use_exchange_ewald_split=False,
            use_exact_ft_j=True,
        )


def test_exact_ft_oracle_fails_closed_on_multipole_artifact():
    system, basis = _h2_box()
    with pytest.raises(NotImplementedError, match="three-translation"):
        _run(
            system,
            basis,
            _ks_opts(),
            use_multipole_far_field=True,
            use_exact_ft_j=True,
        )


def test_pure_rks_matches_closed_shell_uks():
    """RKS and closed-shell UKS now share the J route (SR+LR both)."""
    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    system, basis = _h2_box()
    kmesh = monkhorst_pack(system, [1, 1, 1])
    res_r = run_pbc_bipole_rks(
        system, basis, kmesh, _ks_opts(), progress=False
    )
    res_u = run_pbc_bipole_uks(
        system, basis, kmesh, _ks_opts(), progress=False
    )
    assert res_r.converged and res_u.converged
    # Pre-retirement this pair differed by the exact-FT-vs-direct J route
    # (route-level, not convergence-level, disagreement). Now both build
    # the padded SR+LR J, so the closed-shell energies coincide.
    assert res_u.energy == pytest.approx(res_r.energy, abs=1e-7)


def test_core_tail_warning_only_on_oracle_route(tmp_path, monkeypatch):
    """Dense-core MgO: the ke-tail warning is an oracle-only diagnostic."""
    import vibeqc.pbc_bipole_common as common

    # This unit test isolates the independent exact-FT warning route at the
    # deliberately tiny historical cutoff. Production fold rejection is
    # exercised with real integrals in test_bipole_production_guards.py.
    monkeypatch.setattr(
        common,
        "s_fold_truncation_drift",
        lambda *args, **kwargs: 0.0,
    )
    system, basis = _mgo_cell()
    opts = _ks_opts(cutoff=4.0, max_iter=1)
    opts.initial_guess = InitialGuess.SAD
    # Default (SR+LR) route: no exact-FT core-tail warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(system, basis, opts)
    assert not [
        w for w in caught if "core reciprocal" in str(w.message)
    ], "the retired exact-FT core-tail warning fired on the SR+LR default"
    # Oracle route: the warning fires (MgO needs ke ~ 11000 Ha).
    with pytest.warns(RuntimeWarning, match="core reciprocal"):
        _run(system, basis, opts, use_exact_ft_j=True)
