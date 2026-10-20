"""The fit-source boundary of the neutral fold (#519, RG-7).

``ccm_neutral_cderi_fold`` asks a bound fit source for the canonical-frame
per-channel fits and does everything else itself (kept channels, time
reversal, the star plan, the fold or the block storage). These rows pin that
the default classic source is the historical fold bit for bit, that its two
batchings and the driver-cache source are the same operator, that a foreign
source serving recorded fits reproduces the fold and the block storage
exactly (so everything the fold needs crosses the boundary), and that the
relations a source does not declare are never applied to its fits.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.fit_source import (
    CLASSIC_ENGINE,
    ClassicRSGDFFitSource,
    DriverFitCacheSource,
)
from vibeqc.periodic.ccm.neutral import (
    ccm_neutral_cderi_blocks,
    ccm_neutral_cderi_fold,
)

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]


def _h2_cube(nrep):
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module", params=[(2, 1, 1), (3, 1, 1)],
                ids=lambda m: "x".join(map(str, m)))
def torus(request):
    """One H2 cube torus, its default dense fold and its block storage."""
    ccm = _h2_cube(request.param)
    return ccm, ccm_neutral_cderi_fold(ccm), ccm_neutral_cderi_blocks(ccm)


class _RecordedFits:
    """A source from outside the module: serves fits recorded from a block
    object, declares neither time reversal nor the star plan."""

    supports_time_reversal = False
    supports_star_plan = False
    engine = CLASSIC_ENGINE
    transport = "recorded"

    def __init__(self, block):
        self.channels = {ch.qi: ch for ch in block.channels}
        self.nrep = tuple(block.nrep)
        self.calls = []
        self.stats = {"builds": 0, "batches": 0, "bounded_chunks": 0}

    def bind(self, ctx):
        assert tuple(int(n) for n in ctx.nrep) == self.nrep
        return self

    def channel_fits(self, qi, bras, pool=None):
        self.calls.append((int(qi), tuple(int(a) for a in bras)))
        fits = self.channels[qi].fits
        return {ai: fits[ai].copy() for ai in bras}

    def receipt(self):
        return {"provider": "recorded", "engine": self.engine,
                "transport": self.transport}


def _assert_same_blocks(a, b):
    assert [ch.qi for ch in a.channels] == [ch.qi for ch in b.channels]
    for ca, cb in zip(a.channels, b.channels):
        assert ca.weight == cb.weight and ca.self_paired == cb.self_paired
        np.testing.assert_array_equal(ca.fits, cb.fits)


def test_the_default_source_is_the_historical_fold(torus):
    ccm, dense, block = torus
    explicit = ccm_neutral_cderi_fold(ccm, fit_source=ClassicRSGDFFitSource())
    np.testing.assert_array_equal(explicit, dense)
    explicit_block = ccm_neutral_cderi_blocks(
        ccm, fit_source=ClassicRSGDFFitSource())
    _assert_same_blocks(explicit_block, block)
    receipt = block.info["fit_source"]
    assert receipt["engine"] == CLASSIC_ENGINE
    assert receipt["transport"] == "built"
    assert receipt["fit_batch"] == "shared-q"
    assert receipt["auxiliary_frame"] == "canonical-modrho"
    assert receipt["base_reciprocal_cutoff_ha"] == 200.0


def test_the_classic_batchings_are_one_operator(torus):
    ccm, dense, _ = torus
    per_pair_source = ccm_neutral_cderi_fold(
        ccm, fit_source=ClassicRSGDFFitSource(fit_batch="per-pair"))
    per_pair_kw = ccm_neutral_cderi_fold(ccm, fit_batch="per-pair")
    np.testing.assert_array_equal(per_pair_source, per_pair_kw)
    # The shared-q batch and the single-pair loop differ by summation order
    # only (pinned at 1e-10 in tests/test_ccm_fold_shared_q.py).
    assert np.max(np.abs(per_pair_source - dense)) < 1e-10


def test_a_foreign_source_reproduces_the_fold(torus):
    """Everything the fold needs crosses the boundary: recorded fits served
    by a source the module does not know give the identical array and the
    identical block storage, and the fold asks for every bra of every kept
    channel once, because the source declares no time reversal."""
    ccm, dense, block = torus
    recorded = _RecordedFits(block)
    np.testing.assert_array_equal(
        ccm_neutral_cderi_fold(ccm, fit_source=recorded), dense)
    n_c = int(ccm.n_cells)
    assert sorted(recorded.calls) == sorted(
        (ch.qi, tuple(range(n_c))) for ch in block.channels)
    again = ccm_neutral_cderi_blocks(ccm, fit_source=_RecordedFits(block))
    _assert_same_blocks(again, block)
    assert again.info["fit_source"] == {
        "provider": "recorded", "engine": CLASSIC_ENGINE,
        "transport": "recorded"}


def test_undeclared_relations_and_contradictions_are_refused(torus):
    ccm, _, block = torus
    with pytest.raises(ValueError, match="star plan"):
        ccm_neutral_cderi_fold(ccm, fit_source=_RecordedFits(block),
                               symmetry=True)
    with pytest.raises(ValueError, match="fit_source or fit_cache"):
        ccm_neutral_cderi_fold(ccm, fit_source=ClassicRSGDFFitSource(),
                               fit_cache=object())
    with pytest.raises(ValueError, match="fit_batch configures"):
        ccm_neutral_cderi_fold(ccm, fit_source=ClassicRSGDFFitSource(),
                               fit_batch="per-pair")
    with pytest.raises(ValueError, match="fit_batch"):
        ClassicRSGDFFitSource(fit_batch="all-at-once")


def test_the_driver_cache_source_is_the_fit_cache_path():
    """The multi-k driver's retained classic fit, through the explicit source
    and through ``fit_cache=``: the same array, and the same operator as the
    fold's own build."""
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    ccm = _h2_cube((2, 1, 1))
    g = _quiet(run_ccm_rhf_gdf, ccm, retain_fit_cache=True)
    cache = g.fit_cache
    via_keyword = ccm_neutral_cderi_fold(ccm, fit_cache=cache)
    via_source = ccm_neutral_cderi_fold(
        ccm, fit_source=DriverFitCacheSource(cache))
    np.testing.assert_array_equal(via_source, via_keyword)
    own = ccm_neutral_cderi_fold(ccm)
    assert np.max(np.abs(via_source - own)) < 1e-10
    block = ccm_neutral_cderi_blocks(
        ccm, fit_source=DriverFitCacheSource(cache))
    receipt = block.info["fit_source"]
    assert receipt["transport"] == "driver-fit-cache"
    assert receipt["engine"] == CLASSIC_ENGINE
    assert receipt["cache_pairs"] == len(cache.pairs)


def test_an_explicit_zero_tail_is_no_tail_on_both_producers():
    """``0`` opts out of the RSGDF tail on both producers, and the builders
    run no tail at or below the base cutoff. The driver cache recorded the
    Bloch driver's ``0.0`` and the fold resolved ``None``; the cache was
    refused and the two neutral-source identities differed, for one
    Hamiltonian (found on the untailed MgO pob-DZVP-rev2 (1,1,1) cell of
    #442). Both now compare and record the executed tail."""
    from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    ccm = _h2_cube((2, 1, 1))
    g = _quiet(run_ccm_rhf_gdf, ccm, retain_fit_cache=True,
               rsgdf_tail_ke_cutoff=0)
    d = _quiet(run_ccm_rhf_direct, ccm, fit_cache=g.fit_cache, tail_ke_cutoff=0)
    assert d.cderi_info["fit_transport"] == "driver-fit-cache"
    assert abs(d.energy / ccm.n_cells - g.energy) < 1e-10
    assert d.neutral_source.identity == g.neutral_source.identity
    details = d.neutral_source.as_dict()["details"]
    assert details["fit"]["tail_reciprocal_cutoff_ha"] is None
