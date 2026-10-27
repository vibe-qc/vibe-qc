"""Star storage of the real-Gamma block cderi (#519, RG-10).

``storage="block-star"`` keeps only the fits the k-pair star plan builds; a
star member is a recipe (the plan's operation and the orbital maps of
``_fold_star_reconstruct``) whose exchange is the representative's on the
rotated density. The plan's auxiliary map is unitary and cancels in the sum
over the auxiliary index, so the reduction is exact for any density, not
only for symmetric ones. The ``q = 0`` channel stays whole.

Rows: J and K on random nonsymmetric densities against the plan's
materialised block storage (rounding) and against the fully built block
storage (the plan's own finite-list floor); ``to_dense`` and the MP2 factor;
the accounting and the build's recipe check on ``info["star_storage"]``;
RHF energies through the drivers' ``cderi_storage="block-star"``; the
refusals. The slow rows are UHF on star storage and the ledger's acceptance
cells, diamond Fd-3m and MgO Fm-3m at (2,2,2).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    ccm_direct_oneelectron,
    ccm_exchange_q0_madelung,
    run_ccm_rhf_direct,
    run_ccm_uhf_direct,
)
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

_A = 1.0 / 0.529177210903
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9, max_iter=300)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _compact_h2(nrep):
    lat = np.diag([6.0, 6.0, 6.0])
    atoms = [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])]
    return CCMSystem(PeriodicSystem(3, lat, atoms, 0, 1), nrep, "sto-3g")


def _li_doublet(nrep):
    return CCMSystem(PeriodicSystem(3, np.diag([7.0, 7.0, 7.0]),
                                    [Atom(3, [3.5, 3.5, 3.5])], 0, 2),
                     nrep, "sto-3g")


def _fcc(a):
    return 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float).T


def _diamond(nrep):
    lat = _fcc(3.567 * _A)
    atoms = [Atom(6, [0, 0, 0]), Atom(6, (lat @ [0.25, 0.25, 0.25]).tolist())]
    return CCMSystem(PeriodicSystem(3, lat, atoms), nrep, "sto-3g")


def _mgo(nrep):
    lat = _fcc(4.21 * _A)
    atoms = [Atom(12, [0, 0, 0]), Atom(8, (lat @ [0.5, 0.5, 0.5]).tolist())]
    return CCMSystem(PeriodicSystem(3, lat, atoms), nrep, "sto-3g")


def _three_storages(ccm, **kw):
    star = _quiet(ccm_neutral_cderi_fold, ccm, storage="block-star", **kw)
    plan_full = _quiet(ccm_neutral_cderi_fold, ccm, storage="block",
                       symmetry=True, **kw)
    built = _quiet(ccm_neutral_cderi_fold, ccm, storage="block", **kw)
    return star, plan_full, built


def _max_jk_differences(ccm, a, b, n_trials=3, seed=3):
    S, _, _ = ccm_direct_oneelectron(ccm)
    xi = ccm_exchange_q0_madelung(ccm)
    rng = np.random.default_rng(seed)
    n = S.shape[0]
    worst_j = worst_k = scale = 0.0
    for _ in range(n_trials):
        x = rng.standard_normal((n, n))
        d = x @ x.T / n          # PSD, no space-group or translation symmetry
        ja, jb = a.coulomb(d), b.coulomb(d)
        ka, kb = a.exchange(d, S, xi), b.exchange(d, S, xi)
        worst_j = max(worst_j, float(np.max(np.abs(ja - jb))))
        worst_k = max(worst_k, float(np.max(np.abs(ka - kb))))
        scale = max(scale, float(np.max(np.abs(kb))))
    return worst_j, worst_k, scale


def test_star_storage_is_exact_on_nonsymmetric_densities():
    ccm = _compact_h2((2, 2, 2))
    star, plan_full, built = _three_storages(ccm, ke_cutoff=60.0)
    dj, dk, scale = _max_jk_differences(ccm, star, plan_full)
    assert dj < 1e-13 and dk < 1e-13 * max(scale, 1.0)
    dj, dk, scale = _max_jk_differences(ccm, star, built)
    assert dj < 1e-12 and dk < 1e-12 * max(scale, 1.0)
    np.testing.assert_array_equal(star.to_dense(), plan_full.to_dense())


def test_star_storage_accounting_and_recipe_check():
    ccm = _compact_h2((2, 2, 2))
    star, plan_full, _ = _three_storages(ccm, ke_cutoff=60.0)
    record = star.info["star_storage"]
    stored, total = star.star_counts()
    assert (record["stored_fits"], record["total_fits"]) == (stored, total) == (42, 64)
    # The plan builds 40 of 64; the q = 0 channel is kept whole (+2 members).
    assert (record["plan_builds"], record["plan_entries"]) == (40, 64)
    assert record["q0_channel"] == "whole"
    assert record["recipe_check_channels"] > 0
    assert record["recipe_check_max_rel"] < 1e-12
    assert star.nbytes < plan_full.nbytes
    assert "star storage keeps 42/64 fits" in star.describe()


def test_star_storage_mo_factor_is_the_materialised_one():
    ccm = _compact_h2((2, 2, 2))
    star, plan_full, _ = _three_storages(ccm, ke_cutoff=60.0)
    rng = np.random.default_rng(5)
    c = rng.standard_normal((star.n_sc, 6))
    np.testing.assert_allclose(star.mo_factor(c, c), plan_full.mo_factor(c, c),
                               atol=1e-14, rtol=0)


def test_drivers_run_on_star_storage():
    ccm = _compact_h2((2, 2, 2))
    dense = _quiet(run_ccm_rhf_direct, ccm, ke_cutoff=60.0, **TIGHT)
    star = _quiet(run_ccm_rhf_direct, ccm, ke_cutoff=60.0,
                  cderi_storage="block-star", **TIGHT)
    assert star.converged
    assert abs(star.energy - dense.energy) / ccm.n_cells < 1e-10
    assert star.cderi_info["storage"] == "block-star"
    assert star.cderi_info["star_storage"]["stored_fits"] == 42
    # One operator: the source identity does not depend on the storage.
    assert star.neutral_source.identity == dense.neutral_source.identity


@pytest.mark.slow
def test_uhf_runs_on_star_storage():
    li = _li_doublet((2, 1, 1))
    u_dense = _quiet(run_ccm_uhf_direct, li, ke_cutoff=80.0, **TIGHT)
    u_star = _quiet(run_ccm_uhf_direct, li, ke_cutoff=80.0,
                    cderi_storage="block-star", **TIGHT)
    assert abs(u_star.energy - u_dense.energy) / li.n_cells < 1e-10


def test_star_storage_refusals():
    ccm = _compact_h2((2, 2, 2))
    with pytest.raises(ValueError, match="Wigner-Seitz"):
        _quiet(ccm_neutral_cderi_fold, ccm, storage="block-star",
               ke_cutoff=60.0, ws_cutoff_bohr=5.0)
    star = _quiet(ccm_neutral_cderi_fold, ccm, storage="block-star", ke_cutoff=60.0)
    with pytest.raises(NotImplementedError, match="Wigner-Seitz"):
        star.apply_ws_cutoff(5.0)
    S, _, _ = ccm_direct_oneelectron(ccm)
    with pytest.raises(NotImplementedError, match="transpose form"):
        star.exchange(np.eye(S.shape[0]), form="pair")
    with pytest.raises(ValueError, match="cderi_build='fold'"):
        _quiet(run_ccm_rhf_direct, ccm, cderi_storage="block-star",
               cderi_build="supercell")
    with pytest.raises(ValueError, match="cderi_storage"):
        _quiet(run_ccm_rhf_direct, ccm, cderi_storage="star")


@pytest.mark.slow
@pytest.mark.parametrize("make", [_diamond, _mgo], ids=["diamond-222", "mgo-222"])
def test_acceptance_cells(make):
    """Diamond Fd-3m (non-symmorphic) and MgO Fm-3m at (2,2,2), STO-3G."""
    ccm = make((2, 2, 2))
    star, plan_full, built = _three_storages(ccm, ke_cutoff=100.0)
    dj, dk, scale = _max_jk_differences(ccm, star, plan_full, n_trials=2)
    assert dj < 1e-12 and dk < 1e-12 * max(scale, 1.0)
    record = star.info["star_storage"]
    assert record["recipe_check_max_rel"] < 1e-12
    assert record["reduction_factor"] > 1.0
    e_star = _quiet(run_ccm_rhf_direct, ccm, cderi=star, conv_tol=1e-11,
                    conv_tol_grad=1e-8, max_iter=300).energy
    e_built = _quiet(run_ccm_rhf_direct, ccm, cderi=built, conv_tol=1e-11,
                     conv_tol_grad=1e-8, max_iter=300).energy
    assert abs(e_star - e_built) / ccm.n_cells < 1e-10
