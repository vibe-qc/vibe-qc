"""The real-Γ fold on the shared-q batch: identity to the per-pair fold.

2026-09-24. ``ccm_neutral_cderi_fold`` used to call the single-pair builder
``build_lpq_bloch_native_fft`` once per ``(k_a, k_a+q)`` -- ``N_c × n_q``
calls, each repeating the ket-Bloch AO-pair FT pass -- while the multi-k GDF
driver built the same objects through ``build_lpq_bloch_native_fft_shared_q``
(one pass per q). The fold now issues one batch per kept q-channel, conjugates
the exact time-reversal partner inside self-paired channels, can consume the
driver's retained fit instead of refitting, and forwards the caller's lattice
options to the fit. The historical build stays reachable as
``fit_batch="per-pair", time_reversal=False`` and is the reference every
identity here is pinned against (rule: 1e-10 max element on the cderi, 1e-10
Ha/cell on the energy).

Tier: T3 / experimental (Γ-CCM research lane). The LiH and MgO (2,2,1) rows are
the prompt's named cells and are slow (minutes each under load); the H2-cube
rows run in seconds and cover a genuine ±q pair, a time-reversal pair inside
the Γ channel ((3,1,1)), the thread pool, the carried cderi, the driver
fit-cache consumption and the lattice-option forwarding.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, LatticeSumOptions, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]

ANG = 1.0 / 0.529177210903
CDERI_TOL = 1e-10
ENERGY_TOL_HA_PER_CELL = 1e-10


def _h2_cube():
    """Compact H2 in a 6-bohr cube, STO-3G: the fold's historical fixture."""
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _rocksalt(a_ang, z_a, z_b):
    a = a_ang * ANG
    lat = 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return PeriodicSystem(
        3, lat, [Atom(z_a, [0.0, 0.0, 0.0]),
                 Atom(z_b, [0.5 * a, 0.5 * a, 0.5 * a])], 0, 1)


SYSTEMS = {"LiH": (4.0840, 3, 1), "MgO": (4.2239, 12, 8)}


def _historical_fold(ccm, **kw):
    """The per-pair build exactly as it ran before 2026-09-24."""
    return ccm_neutral_cderi_fold(
        ccm, fit_batch="per-pair", time_reversal=False, fold_threads=1, **kw)


def _max_abs(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a - b)))


@pytest.mark.parametrize("mesh", [(2, 1, 1), (3, 1, 1), (2, 2, 1)],
                         ids=lambda m: "x".join(map(str, m)))
def test_shared_q_fold_matches_the_per_pair_fold_h2_cube(mesh):
    """Default build (shared-q batch + time reversal) against the historical
    per-pair build: (3,1,1) carries a genuine ±q pair AND a time-reversal
    pair (1/3, 2/3) inside the Γ channel; (2,2,1) is all self-paired."""
    ccm = CCMSystem(_h2_cube(), mesh, "sto-3g")
    ref = _historical_fold(ccm)
    new = ccm_neutral_cderi_fold(ccm)
    assert _max_abs(new, ref) < CDERI_TOL


def test_time_reversal_reconstruction_is_exact_h2_311():
    """Inside a self-paired channel the (-ka) fit is the conjugate of the (ka)
    fit; switching the reconstruction off must not move a digit."""
    ccm = CCMSystem(_h2_cube(), (3, 1, 1), "sto-3g")
    with_trs = ccm_neutral_cderi_fold(ccm, time_reversal=True)
    without = ccm_neutral_cderi_fold(ccm, time_reversal=False)
    assert _max_abs(with_trs, without) < 1e-12


def test_channel_pool_is_elementwise_stable_h2_311():
    """The q-channel thread pool changes scheduling, not arithmetic."""
    ccm = CCMSystem(_h2_cube(), (3, 1, 1), "sto-3g")
    serial = ccm_neutral_cderi_fold(ccm, fold_threads=1)
    pooled = ccm_neutral_cderi_fold(ccm, fold_threads=2)
    assert _max_abs(serial, pooled) < 1e-13


@pytest.mark.slow
@pytest.mark.parametrize("name", ["LiH", "MgO"])
def test_shared_q_fold_matches_the_per_pair_fold_221(name):
    """The prompt's acceptance cells: LiH and MgO STO-3G at (2,2,1), cderi to
    1e-10 (max element) and the direct-route energy to 1e-10 Ha/cell."""
    a_ang, z_a, z_b = SYSTEMS[name]
    ccm = CCMSystem(_rocksalt(a_ang, z_a, z_b), (2, 2, 1), "sto-3g")
    ref = _historical_fold(ccm)
    d_new = run_ccm_rhf_direct(ccm)            # default: shared-q fold
    assert d_new.converged
    assert d_new.cderi_info["build"] == "fold"
    assert d_new.cderi_info["fit_batch"] == "shared-q"
    assert _max_abs(np.asarray(d_new.cderi), ref) < CDERI_TOL, name
    d_ref = run_ccm_rhf_direct(ccm, cderi=ref)
    assert d_ref.converged
    resid = (d_new.energy - d_ref.energy) / ccm.n_cells
    assert abs(resid) < ENERGY_TOL_HA_PER_CELL, (
        f"{name} (2,2,1): shared-q - per-pair = {resid:.3e} Ha/cell")


def test_direct_result_carries_its_cderi_and_the_consumers_reuse_it():
    """One build per job: the direct result stamps the L it rode, and the
    correlated entry points called without ``cderi=`` ride that array --
    exactly the same numbers as passing it explicitly."""
    from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
    from vibeqc.periodic.ccm.dlpno import ccm_dlpno_mp2
    from vibeqc.periodic.ccm.mp2 import run_ccm_mp2

    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    d = run_ccm_rhf_direct(ccm)
    assert d.converged
    L = d.cderi
    assert L is not None and L.ndim == 3 and L.shape[1] == ccm.nbf
    assert d.cderi_info["build"] == "fold"
    assert d.cderi_info["lat_cutoff_bohr"] is None      # builder default

    mp2_explicit = run_ccm_mp2(ccm, d, cderi=L)
    mp2_implicit = run_ccm_mp2(ccm, d)
    assert mp2_implicit.e_correlation == mp2_explicit.e_correlation

    dl_explicit = ccm_dlpno_mp2(ccm, d, cderi=L, localize="none")
    dl_implicit = ccm_dlpno_mp2(ccm, d, localize="none")
    assert dl_implicit.e_corr == dl_explicit.e_corr

    cc_explicit = run_ccm_ccsd(ccm, d, cderi=L, compute_triples=False)
    cc_implicit = run_ccm_ccsd(ccm, d, compute_triples=False)
    assert cc_implicit.e_correlation == cc_explicit.e_correlation

    # A prebuilt cderi is recorded as such, and is the array the SCF rode.
    d2 = run_ccm_rhf_direct(ccm, cderi=L)
    assert d2.cderi_info["build"] == "caller"
    assert d2.cderi is L or np.shares_memory(d2.cderi, L) or _max_abs(d2.cderi, L) == 0.0


def test_callers_lattice_options_reach_the_fold():
    """Prompt item 5: ``run_ccm_rhf_direct(lat_opts=X)`` used to reach the
    one-electron sums only; the fold fit at ``LatticeSumOptions()`` whatever
    the caller passed. Now the fit's cell list follows the caller, as the
    multi-k driver's does, and the result records the cutoff."""
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 7.0            # drops the 8.5-bohr face-diagonal images
    d = run_ccm_rhf_direct(ccm, lat_opts=opts)
    assert d.cderi_info["lat_cutoff_bohr"] == 7.0
    L_opts = ccm_neutral_cderi_fold(ccm, lat_opts=opts)
    assert _max_abs(np.asarray(d.cderi), L_opts) < 1e-12
    L_default = ccm_neutral_cderi_fold(ccm)
    # The cutoff matters on this cell, so the forwarding is observable.
    assert _max_abs(L_default, L_opts) > 1e-8


def _driver_fit_cache(ccm, *, ke_cutoff=200.0, reverse=False):
    """The classic multi-k driver's per-pair cache, wrapped as the driver
    would retain it (all pairs, eigenmode frame, q-metric eigensystems)."""
    from vibeqc import monkhorst_pack
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.aux_basis import (
        default_aux_for, make_aux_basis_set, make_modrho_aux_basis)
    from vibeqc.periodic_k_gdf import (
        RSGDFFitCache, _build_rsgdf_lpq_cache_shared_q)

    unit = ccm.unit_system
    mol = unit.unit_cell_molecule()
    ubasis = BasisSet(mol, ccm.basis_name)
    aux_name = default_aux_for(ccm.basis_name)
    aux_modrho = make_modrho_aux_basis(
        make_aux_basis_set(mol, aux_name=aux_name, drop_eta=0.0), mol)
    kpts = np.asarray(monkhorst_pack(unit, list(ccm.nrep)).kpoints, dtype=float)
    if reverse:
        kpts = kpts[::-1].copy()      # label matching, not index matching
    opts = LatticeSumOptions()
    q_metric = {}
    pairs = _build_rsgdf_lpq_cache_shared_q(
        unit, ubasis, aux_modrho, kpts, True,
        ke_cutoff=float(ke_cutoff), tail_ke_cutoff=None, lat_opts=opts,
        linear_dep_thr=1e-9, fit_screen_threshold=0.0, progress=None,
        q_metric_cache=q_metric)
    return RSGDFFitCache(
        system=unit, kpoints_cart=kpts, pairs=pairs, q_metric=q_metric,
        ke_cutoff=float(ke_cutoff), tail_ke_cutoff=None,
        lat_cutoff_bohr=float(opts.cutoff_bohr), linear_dep_thr=1e-9,
        fit_screen_threshold=0.0, aux_basis_name=aux_name,
        n_aux=int(aux_modrho.nbasis), n_ao=int(ubasis.nbasis),
        need_k_pairs=True)


def test_fold_consumes_the_drivers_fit_cache_h2_221():
    """Prompt item 3: both producers in one process -- the fold folds the
    driver's own per-pair fits (eigenmode frame rotated back to the canonical
    frame, pairs matched by mesh label) and equals its own build to 1e-10."""
    ccm = CCMSystem(_h2_cube(), (2, 2, 1), "sto-3g")
    own = ccm_neutral_cderi_fold(ccm)
    cache = _driver_fit_cache(ccm, reverse=True)
    assert len(cache.pairs) == ccm.n_cells ** 2
    consumed = ccm_neutral_cderi_fold(ccm, fit_cache=cache)
    assert _max_abs(consumed, own) < CDERI_TOL
    d = run_ccm_rhf_direct(ccm, fit_cache=cache)
    assert d.cderi_info["build"] == "driver-fit-cache"
    d_own = run_ccm_rhf_direct(ccm)
    assert abs(d.energy - d_own.energy) / ccm.n_cells < ENERGY_TOL_HA_PER_CELL


def test_fold_refuses_a_fit_cache_at_another_operator():
    """A cache fit at a different ``ke_cutoff`` is a different operator and
    must not be folded silently."""
    ccm = CCMSystem(_h2_cube(), (2, 1, 1), "sto-3g")
    cache = _driver_fit_cache(ccm, ke_cutoff=150.0)
    with pytest.raises(ValueError, match="different operator"):
        ccm_neutral_cderi_fold(ccm, fit_cache=cache)
    with pytest.raises(ValueError, match="fit_cache"):
        run_ccm_rhf_direct(ccm, cderi=np.zeros((1, ccm.nbf, ccm.nbf)),
                           fit_cache=cache)


def test_fold_rejects_an_unknown_batch_mode():
    ccm = CCMSystem(_h2_cube(), (1, 1, 1), "sto-3g")
    with pytest.raises(ValueError, match="fit_batch"):
        ccm_neutral_cderi_fold(ccm, fit_batch="all-at-once")


def test_classic_pin_resolves_a_none_ke_cutoff_like_the_driver():
    """#378 x #257: the runner arm forwards ``rsgdf_ke_cutoff=None`` (no
    request) into the neutral-Bloch producer; the classic pin must classify
    the mirrored tail at the driver's default mesh, not fail on float(None),
    and must agree with an explicit 200 Ha request. Cheap: no SCF runs."""
    from vibeqc.periodic.ccm.neutral import ccm_neutral_tail_ke_cutoff
    from vibeqc.periodic.ccm.ri import _pin_classic_fit_source
    from vibeqc.periodic_k_gdf import RSGDF_KE_CUTOFF_DEFAULT

    a_ang, z_a, z_b = SYSTEMS["MgO"]
    ccm = CCMSystem(_rocksalt(a_ang, z_a, z_b), (1, 1, 1), "sto-3g")
    kw_none = {"rsgdf_ke_cutoff": None}
    kw_default = {}
    kw_explicit = {"rsgdf_ke_cutoff": 200.0}
    for kw in (kw_none, kw_default, kw_explicit):
        assert _pin_classic_fit_source(ccm, kw, symmetry=False, who="t") == "classic"
    expected = ccm_neutral_tail_ke_cutoff(ccm, ke_cutoff=float(RSGDF_KE_CUTOFF_DEFAULT))
    assert expected is not None                     # MgO auto-tails at N_c = 1
    assert kw_none["rsgdf_tail_ke_cutoff"] == expected
    assert kw_default["rsgdf_tail_ke_cutoff"] == expected
    assert kw_explicit["rsgdf_tail_ke_cutoff"] == expected


def test_runner_convention_dataclasses_construct():
    """#378 placed defaulted ``rsgdf_tail_ke_cutoff`` / ``fit_source`` fields ahead
    of non-default ones in ``CCMRealGammaConvention``, which makes the dataclass
    decorator raise at import and every ``variant="real-gamma"`` runner job fail.
    Importing both runner adapters and reading their fields is the cheap guard."""
    import dataclasses

    from vibeqc.periodic.ccm.neutral_bloch_runner import CCMNeutralBlochConvention
    from vibeqc.periodic.ccm.real_gamma_runner import CCMRealGammaConvention

    for cls in (CCMRealGammaConvention, CCMNeutralBlochConvention):
        names = [f.name for f in dataclasses.fields(cls)]
        assert "fit_source" in names and "rsgdf_tail_ke_cutoff" in names, cls


def test_bloch_producer_retains_a_canonical_cache_the_fold_consumes_h2_221():
    """Both producers in one process, end to end (#378 on the tree): the Bloch
    producer's classic pin builds canonical-frame pairs, the retention must
    label them so (rotating them again fails the rank check on LiH (2,2,2)
    and, where the ranks coincide, silently hands the fold a rotated
    representation), and the fold that consumes them must equal its own
    build elementwise and reproduce Theorem 1."""
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    ccm = CCMSystem(_h2_cube(), (2, 2, 1), "sto-3g")
    g = run_ccm_rhf_gdf(ccm, retain_fit_cache=True)
    cache = g.fit_cache
    assert cache is not None and cache.frame == "canonical"
    assert len(cache.pairs) == ccm.n_cells ** 2
    assert all(np.shape(L)[0] == cache.n_aux for L in cache.pairs.values())
    own = ccm_neutral_cderi_fold(ccm)
    consumed = ccm_neutral_cderi_fold(ccm, fit_cache=cache)
    assert _max_abs(consumed, own) < CDERI_TOL
    d = run_ccm_rhf_direct(ccm, fit_cache=cache)
    assert d.cderi_info["build"] == "driver-fit-cache"
    assert abs(g.energy - d.energy / ccm.n_cells) < ENERGY_TOL_HA_PER_CELL



def test_fold_refuses_a_driver_cache_on_another_pair_support():
    """#466: the fold fits its own pairs on the historical index ball, so a
    driver cache built on physical pairs is a different operator."""
    from types import SimpleNamespace
    from vibeqc.periodic.ccm.neutral import _FitCacheSource

    cache = SimpleNamespace(
        pairs={}, kpoints_cart=np.zeros((1, 3)), ke_cutoff=200.0, tail_ke_cutoff=None,
        lat_cutoff_bohr=15.0, n_ao=2, n_aux=10, canonical_pair=None,
        linear_dep_thr=1e-9, fit_screen_threshold=0.0, support_policy="physical_pairs")
    lattice = LatticeSumOptions()
    lattice.cutoff_bohr = 15.0
    with pytest.raises(ValueError, match="support_policy 'physical_pairs'"):
        _FitCacheSource(
            cache, unit=None, ubasis=SimpleNamespace(nbasis=2),
            aux_modrho=SimpleNamespace(nbasis=10), b_lat=None, nrep=(1, 1, 1),
            frac=None, ke_cutoff=200.0, tail=None, lat_opts=lattice, omega_screen=0.0)
