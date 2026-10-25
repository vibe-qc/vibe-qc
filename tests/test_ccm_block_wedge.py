"""Wedge exchange on the real-Gamma block storage (#519, RG-11).

With a density that has the star plan's symmetry the exchange at a member
bra of a k-star is the transported exchange at its representative,
``Kh(Rk) = Om Kh(k) Om^H`` with ``Om = P_ao diag(e^{i (Rk).S})``, the Bloch
transport convention of ``periodic_k_symmetry.expand_k_matrices_to_full``.
``exchange_operator(wedge=True)`` builds the exchange on the representatives
only and transports it, after measuring the density's distance from that
symmetry on every call; a density off the symmetry gets the full build.

Rows: the wedge equals the full build on an SCF density (block and star
storage) and falls back exactly on a nonsymmetric one; RHF through
``exchange_wedge=True`` equals the full build with no fallback from an HCORE
start; the refusals. Slow rows: UHF through the wedge, and the bra and time
reduction on diamond (2,2,2).
"""
from __future__ import annotations

import time
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


def _diamond(nrep):
    lat = 0.5 * 3.567 * _A * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float).T
    atoms = [Atom(6, [0, 0, 0]), Atom(6, (lat @ [0.25, 0.25, 0.25]).tolist())]
    return CCMSystem(PeriodicSystem(3, lat, atoms), nrep, "sto-3g")


@pytest.mark.parametrize("storage", ["block", "block-star"])
def test_the_wedge_is_the_full_build_on_a_symmetric_density(storage):
    ccm = _compact_h2((2, 2, 2))
    blk = _quiet(ccm_neutral_cderi_fold, ccm, storage=storage, symmetry=True,
                 ke_cutoff=60.0)
    res = _quiet(run_ccm_rhf_direct, ccm, cderi=blk, **TIGHT)
    S, _, _ = ccm_direct_oneelectron(ccm)
    xi = ccm_exchange_q0_madelung(ccm)
    full = blk.exchange_operator(S, xi)
    wedge = blk.exchange_operator(S, xi, wedge=True)
    np.testing.assert_allclose(wedge(res.density), full(res.density),
                               atol=1e-13, rtol=0)
    stats = wedge.wedge_stats
    assert (stats["calls"], stats["wedge"], stats["fallbacks"]) == (1, 1, 0)
    assert (stats["bras"], stats["representatives"]) == (8, 6)
    # A density without the plan's symmetry is detected and built in full.
    x = np.random.default_rng(1).standard_normal(res.density.shape)
    d = x @ x.T / x.shape[0]
    np.testing.assert_array_equal(wedge(d), full(d))
    assert stats["fallbacks"] == 1 and stats["max_density_residual"] > 1e-3


def test_rhf_through_the_wedge():
    ccm = _compact_h2((2, 2, 2))
    blk = _quiet(ccm_neutral_cderi_fold, ccm, storage="block-star", ke_cutoff=60.0)
    full = _quiet(run_ccm_rhf_direct, ccm, cderi=blk, **TIGHT)
    wedge = _quiet(run_ccm_rhf_direct, ccm, cderi=blk, exchange_wedge=True, **TIGHT)
    assert wedge.converged
    assert abs(wedge.energy - full.energy) / ccm.n_cells < 1e-10
    stats = wedge.cderi_info["exchange_wedge"]
    assert stats["fallbacks"] == 0 and stats["wedge"] == stats["calls"] > 0
    assert "exchange_wedge" not in full.cderi_info


def test_wedge_refusals():
    ccm = _compact_h2((2, 2, 2))
    with pytest.raises(ValueError, match="exchange_wedge needs block storage"):
        _quiet(run_ccm_rhf_direct, ccm, ke_cutoff=60.0, exchange_wedge=True)
    plain = _quiet(ccm_neutral_cderi_fold, ccm, storage="block", ke_cutoff=60.0)
    assert plain.kstar is None
    with pytest.raises(ValueError, match="k-star"):
        plain.exchange_operator(wedge=True)


@pytest.mark.slow
def test_uhf_through_the_wedge():
    li = _li_doublet((2, 1, 1))
    blk = _quiet(ccm_neutral_cderi_fold, li, storage="block-star", ke_cutoff=80.0)
    full = _quiet(run_ccm_uhf_direct, li, cderi=blk, **TIGHT)
    wedge = _quiet(run_ccm_uhf_direct, li, cderi=blk, exchange_wedge=True, **TIGHT)
    assert abs(wedge.energy - full.energy) / li.n_cells < 1e-10
    assert wedge.cderi_info["exchange_wedge"]["fallbacks"] == 0


@pytest.mark.slow
def test_diamond_bra_and_time_reduction():
    """Diamond Fd-3m (2,2,2): the bra k-star and the measured K-build time."""
    ccm = _diamond((2, 2, 2))
    blk = _quiet(ccm_neutral_cderi_fold, ccm, storage="block-star", ke_cutoff=100.0)
    res = _quiet(run_ccm_rhf_direct, ccm, cderi=blk, conv_tol=1e-11,
                 conv_tol_grad=1e-8, max_iter=300)
    S, _, _ = ccm_direct_oneelectron(ccm)
    xi = ccm_exchange_q0_madelung(ccm)
    full = blk.exchange_operator(S, xi)
    wedge = blk.exchange_operator(S, xi, wedge=True)
    k_full, k_wedge = full(res.density), wedge(res.density)
    assert np.max(np.abs(k_wedge - k_full)) < 1e-12
    stats = wedge.wedge_stats
    assert stats["fallbacks"] == 0
    assert stats["representatives"] < stats["bras"]

    def timed(fn, n=3):
        t0 = time.perf_counter()
        for _ in range(n):
            fn(res.density)
        return (time.perf_counter() - t0) / n

    t_full, t_wedge = timed(full), timed(wedge)
    print(f"diamond (2,2,2): bras {stats['bras']}, representatives "
          f"{stats['representatives']}, K build {t_full:.3f} s full, "
          f"{t_wedge:.3f} s wedge")
    assert t_wedge < t_full
