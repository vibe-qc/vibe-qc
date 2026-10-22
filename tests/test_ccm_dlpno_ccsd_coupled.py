"""Per-pair-coupled DLPNO-CCSD on the cyclic cluster (AICCM Task 2, M3b).

The hard gate: at ``tcut_pno = 0`` with full coupling, the genuine per-pair-coupled
local CCSD (each pair in its own PNO basis, cross-pair PNO-overlap projections)
reproduces the canonical CCM CCSD on the same neutral four-center to the CCSD
convergence tolerance -- the correctness proof for the coupled residual.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.ccsd import run_ccm_ccsd
from vibeqc.periodic.ccm.dlpno_ccsd_coupled import ccm_dlpno_ccsd_coupled
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_KE = 40.0


@lru_cache(maxsize=2)
def _ref(nrep):
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), nrep, "sto-3g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    scf = run_ccm_rhf(ccm, eri=g)
    canon = run_ccm_ccsd(ccm, scf, cderi=L, compute_triples=False)
    return ccm, scf, L, canon


@pytest.mark.parametrize("nrep", [(2, 1, 1), (4, 1, 1)])
@pytest.mark.parametrize("localize", ["none", "pm"])
def test_coupled_no_truncation_reproduces_canonical(nrep, localize):
    ccm, scf, L, canon = _ref(nrep)
    d = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize=localize,
                               tcut_pno=0.0, tcut_mkn=0.0, compute_triples=False)
    assert d.converged
    assert d.e_ccsd_correlation == pytest.approx(canon.e_ccsd_correlation, abs=1e-8)


def test_coupled_truncation_shrinks_pnos_and_stays_bound():
    ccm, scf, L, canon = _ref((4, 1, 1))
    d0 = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=0.0, compute_triples=False)
    dT = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=1e-4, compute_triples=False)
    assert dT.avg_pno_per_pair < d0.avg_pno_per_pair       # truncation removes PNOs
    assert dT.e_ccsd_correlation < 0.0                     # still bound
    # the truncated correlation recovers most of canonical.
    assert abs(dT.e_ccsd_correlation - canon.e_ccsd_correlation) < 5e-3


@pytest.mark.parametrize("nrep", [(2, 1, 1), (4, 1, 1)])
@pytest.mark.parametrize("localize", ["none", "pm"])
# "local" tol 2e-8: the T0 semicanonical error of the scalable (T) is a small
# kernel-dependent quantity; the 2026-07-02 cross-torus truncation fix in
# ccm_neutral_cderi (wrap-aware lattice cutoff, corrects L on supercells
# > ~15 bohr — (4,1,1) here is 24 bohr) moved the pm/(4,1,1) case from
# 0.99e-8 to 1.002e-8. The exact "dense" mode is unaffected (1e-10).
@pytest.mark.parametrize("triples_mode,tol", [("dense", 1e-10), ("local", 2e-8)])
def test_coupled_ccsdt_no_truncation_reproduces_canonical(nrep, localize, triples_mode, tol):
    """No-truncation coupled DLPNO-CCSD(T) reproduces canonical CCSD(T), for both
    (T) drivers: ``dense`` reconstructs the canonical amplitudes (exact, ~1e-10);
    ``local`` (the scalable per-triple-TNO (T), the default) matches to the CCSD
    convergence tolerance (~1e-8) at ``tcut_tno=0`` / full PNO."""
    ccm, scf, L, _ = _ref(nrep)
    canon = run_ccm_ccsd(ccm, scf, cderi=L, compute_triples=True)
    d = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize=localize, tcut_pno=0.0,
                               compute_triples=True, triples_mode=triples_mode)
    assert d.e_t == pytest.approx(canon.e_t, abs=tol)
    # total: dense reproduces canonical tightly; the scalable local-(T0) carries a
    # small semicanonical error on localized occupieds that stacks with the CCSD
    # convergence tolerance.
    assert d.e_correlation == pytest.approx(canon.e_correlation, abs=5e-8)


def test_local_triples_tno_truncation_changes_t():
    """tcut_tno>0 truncates each triple's TNO domain — a different, still-small (T)."""
    ccm, scf, L, _ = _ref((4, 1, 1))
    t0 = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=0.0, compute_triples=True,
                                triples_mode="local", tcut_tno=0.0).e_t
    tT = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, tcut_pno=0.0, compute_triples=True,
                                triples_mode="local", tcut_tno=1e-4).e_t
    assert tT != t0
    assert tT < 0.0 and abs(tT - t0) < 1e-5


def test_coupling_radius_locality_screen():
    """The O(N)-pairs lever: a large coupling_radius reproduces full coupling; a
    finite radius couples fewer occupieds with a small, controlled energy change."""
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), (4, 1, 1), "sto-3g")
    g = ccm_eri_neutral(ccm, ke_cutoff=_KE)
    L = ccm_neutral_cderi(ccm, ke_cutoff=_KE)
    scf = run_ccm_rhf(ccm, eri=g)
    full = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="pm",
                                  coupling_radius=-1.0, compute_triples=False)
    big = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="pm",
                                 coupling_radius=1000.0, compute_triples=False)
    tight = ccm_dlpno_ccsd_coupled(ccm, scf, cderi=L, localize="pm",
                                   coupling_radius=4.0, compute_triples=False)
    # large radius == full coupling (every occupied within range)
    assert big.avg_coupled_occ == pytest.approx(full.avg_coupled_occ)
    assert big.e_ccsd_correlation == pytest.approx(full.e_ccsd_correlation, abs=1e-10)
    # a tight radius couples fewer occupieds, still bound, close to full
    assert tight.avg_coupled_occ < full.avg_coupled_occ
    assert tight.e_ccsd_correlation < 0.0
    assert abs(tight.e_ccsd_correlation - full.e_ccsd_correlation) < 1e-2


@pytest.mark.parametrize("n_frozen", [0, 1])
def test_truncated_coupled_matches_full_virtual_residual(n_frozen):
    """Pair spaces stay truncated; contracted indices retain their own spaces.

    The full-virtual residual is independent of the compiled PNO contraction
    and detects the retired target-pair-first projection error.
    """
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd
    from vibeqc.periodic.ccm.dlpno_source import ccm_neutral_source

    ccm, scf, L, _ = _ref((4, 1, 1))
    source = ccm_neutral_source(ccm, scf, cderi=L, n_frozen=n_frozen)
    coupled = ccm_dlpno_ccsd_coupled(
        ccm, scf, cderi=L, tcut_pno=1e-4, n_frozen=n_frozen,
        compute_triples=True, triples_mode="dense", conv_tol=1e-11,
    )
    full = run_local_dlpno_ccsd(None, None, None, None, LocalCCSDOptions(
        localise="none", n_frozen=n_frozen, tcut_pno=1e-4, tcut_pno_diag_scale=1.0,
        tcut_mkn=0.0, tcut_pairs=0.0, coupling_radius=0.0, pno_norm="legacy",
        domain_criterion="mulliken", pno_correction=False, residual_domain="full",
        use_cpp_kernel=False, ladder_in_pno_space=False,
        compute_triples=True, triples_mode="exact", conv_tol_energy=1e-11,
        diis_size=8, max_iter=128,
    ), source=source)
    assert coupled.converged and full.converged
    assert coupled.avg_pno_per_pair < ccm.nbf // 2
    assert coupled.e_ccsd_correlation == pytest.approx(full.e_corr, abs=1e-9)
    assert coupled.e_t == pytest.approx(full.e_t, abs=1e-10)
    assert coupled.e_total == pytest.approx(full.e_total, abs=1e-9)
    assert coupled.cc_result.pno_per_pair == full.pno_per_pair


def test_source_localisation_preserves_frozen_and_active_projectors():
    from vibeqc.periodic.ccm.dlpno_source import ccm_neutral_source

    ccm, scf, L, _ = _ref((4, 1, 1))
    before = scf.mo_coeffs.copy()
    source = ccm_neutral_source(ccm, scf, cderi=L, n_frozen=1)
    C = source.reference.mo_coeffs
    nocc = ccm.supercell.n_electrons() // 2
    assert np.array_equal(C[:, :1], before[:, :1])
    assert np.array_equal(C[:, nocc:], before[:, nocc:])
    assert C[:, 1:nocc] @ C[:, 1:nocc].T == pytest.approx(
        before[:, 1:nocc] @ before[:, 1:nocc].T, abs=1e-12,
    )
    assert np.max(np.abs(C[:, :1].T @ source.overlap @ C[:, 1:nocc])) < 1e-12
    assert np.array_equal(scf.mo_coeffs, before)
    # Across the torus boundary, these centres are two bohr apart.
    a = np.asarray(ccm.cluster_lattice)[:, 0]
    unit = a / np.linalg.norm(a)
    assert source.pair_distance_fn(unit, a - unit) == pytest.approx(2.0, abs=1e-12)


def test_source_reuses_carried_factors_and_records_overrides(monkeypatch):
    import copy
    from vibeqc.periodic.ccm import neutral
    from vibeqc.periodic.ccm.dlpno_source import ccm_neutral_source

    ccm, original, L, _ = _ref((2, 1, 1))
    scf = copy.copy(original)
    scf.cderi = L
    scf.exchange_q0 = "strict-zero-mode"

    def no_refit(*args, **kwargs):
        raise AssertionError("retained reference factors must not be rebuilt")

    monkeypatch.setattr(neutral, "ccm_neutral_cderi", no_refit)
    source = ccm_neutral_source(ccm, scf)
    same = ccm_neutral_source(ccm, scf, cderi=L.copy())
    different = ccm_neutral_source(ccm, scf, cderi=1.01 * L)
    unknown = ccm_neutral_source(ccm, original, cderi=L)
    assert source.conventions.same_source and same.conventions.same_source
    assert not different.conventions.same_source and not unknown.conventions.same_source
    assert source.conventions.exchange_q0 == "strict-zero-mode"
    assert source.provenance["factors"] == same.provenance["factors"]
    assert source.provenance["factors"] != different.provenance["factors"]
    assert source.identity == ccm_neutral_source(ccm, scf).identity
    assert source.identity != different.identity
    result = ccm_dlpno_ccsd_coupled(ccm, scf)
    assert result.source_kind == "ccm-neutral"
    assert result.source_identity == result.cc_result.source_identity == source.identity
    assert result.same_source
    assert result.exchange_q0 == "strict-zero-mode"


@pytest.mark.parametrize("field,value,reason", [
    ("converged", False, "converged"),
    ("reference_kind", "uhf", "closed-shell HF"),
    ("functional", "pbe", "closed-shell HF"),
    ("backend", "bipole", "BIPOLE"),
    ("used_kpoint_symmetry_unfolding", True, "symmetry-reduced"),
])
def test_source_refuses_incompatible_reference(field, value, reason):
    import copy
    from vibeqc.dlpno.source import CorrelationSourceError
    from vibeqc.periodic.ccm.dlpno_source import ccm_neutral_source

    ccm, original, L, _ = _ref((2, 1, 1))
    scf = copy.copy(original)
    setattr(scf, field, value)
    with pytest.raises(CorrelationSourceError, match=reason):
        ccm_neutral_source(ccm, scf, cderi=L)


@pytest.mark.parametrize("n_frozen", [-1, 2, 0.5, True])
def test_source_refuses_invalid_frozen_count(n_frozen):
    from vibeqc.periodic.ccm.dlpno_source import ccm_neutral_source

    ccm, scf, L, _ = _ref((2, 1, 1))
    with pytest.raises(ValueError, match="n_frozen"):
        ccm_neutral_source(ccm, scf, cderi=L, n_frozen=n_frozen)


def test_bloch_reference_uses_same_engine_after_fold():
    from vibeqc._vibeqc_core import PeriodicRHFOptions
    from vibeqc.periodic.ccm.bloch_reference import as_real_reference
    from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf

    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])]), (1, 1, 2), "sto-3g")
    opts = PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-8
    bloch = run_ccm_rhf_gdf(ccm, retain_fit_cache=True, options=opts)
    real = as_real_reference(ccm, bloch)
    canonical = run_ccm_ccsd(ccm, real, cderi=real.cderi, compute_triples=True)
    got = ccm_dlpno_ccsd_coupled(ccm, bloch, compute_triples=True, triples_mode="dense")
    assert got.converged and got.same_source
    assert got.e_ccsd_correlation == pytest.approx(canonical.e_ccsd_correlation, abs=1e-8)
    assert got.e_t == pytest.approx(canonical.e_t, abs=1e-10)
    assert got.exchange_q0 == real.exchange_q0
