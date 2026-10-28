"""Opt-in SCF component timing (GPU programme G0-A).

Checks the accounting contract of ``component_timing`` on real small RHF and
RKS runs: unchanged numbers when enabled, positive clocks for executed
phases, a fused JK timer that is never split into invented J/K times, no
exchange timer for PBE, exclusive phases that reconcile with the session wall
time, and tracked DF construction memory. Timings are not performance
evidence; they only exercise the bookkeeping.
"""
from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2o():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.43233673, -1.10715266]),
        vq.Atom(1, [0.0, -1.43233673, -1.10715266]),
    ])


def _opts(cls, *, route, functional=None, timing):
    o = cls()
    if functional:
        o.functional = functional
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-6
    o.initial_guess = vq.InitialGuess.SAD
    o.stability_check = False
    if route == "conventional":
        o.scf_mode = vq.SCFMode.CONVENTIONAL
    elif route == "direct":
        o.scf_mode = vq.SCFMode.DIRECT
    elif route == "df":
        o.density_fit = True
        o.aux_basis = "def2-universal-jkfit"
    elif route == "rijcosx":
        o.density_fit = True
        o.cosx = True
        o.aux_basis = "def2-universal-jfit"
        o.cosx_grid_level = 1
    o.component_timing = timing
    return o


def _run(method, route, timing, *, basis="sto-3g"):
    mol = _h2o()
    bs = vq.BasisSet(mol, basis)
    if method == "RHF":
        return vq.run_rhf(mol, bs, _opts(vq.RHFOptions, route=route, timing=timing))
    return vq.run_rks(mol, bs, _opts(vq.RKSOptions, route=route,
                                     functional=method, timing=timing))


def _paths(ct):
    return {p.path: p for p in ct.phases}


def _assert_reconciled(ct):
    exclusive = sum(p.exclusive_s for p in ct.phases)
    assert ct.wall_s > 0.0
    assert ct.unclassified_s >= -1e-9
    assert exclusive + ct.unclassified_s == pytest.approx(ct.wall_s, abs=1e-9)
    for p in ct.phases:
        assert p.calls > 0
        assert p.inclusive_s >= p.exclusive_s - 1e-12
        assert p.exclusive_s >= -1e-9
    for it in ct.iterations:
        assert sum(p.exclusive_s for p in it.phases) <= it.wall_s + 1e-9


@pytest.fixture(autouse=True)
def _single_thread():
    before = vq.get_num_threads()
    vq.set_num_threads(1)
    yield
    vq.set_num_threads(before)


def test_default_is_off_and_empty():
    assert vq.RHFOptions().component_timing is False
    assert vq.RKSOptions().component_timing is False
    res = _run("RHF", "conventional", False)
    ct = res.component_timing
    assert ct.enabled is False
    assert list(ct.phases) == [] and list(ct.iterations) == []


@pytest.mark.parametrize("method,route", [
    ("RHF", "conventional"), ("RHF", "direct"), ("RHF", "df"),
    ("PBE", "df"), ("PBE0", "direct"), ("PBE0", "rijcosx"),
])
def test_enabled_does_not_change_results(method, route):
    off = _run(method, route, False)
    on = _run(method, route, True)
    assert on.converged and off.converged
    assert on.n_iter == off.n_iter
    assert on.energy == pytest.approx(off.energy, abs=1e-11)
    np.testing.assert_allclose(on.density, off.density, atol=1e-10)
    assert on.component_timing.enabled
    _assert_reconciled(on.component_timing)


def test_rhf_fused_jk_is_not_split_for_direct():
    ct = _run("RHF", "direct", True).component_timing
    paths = _paths(ct)
    fused = paths["fock.jk_fused"]
    assert fused.exclusive_s > 0.0
    assert fused.inclusive_s == pytest.approx(fused.exclusive_s)
    assert not any(p.startswith("fock.jk_fused/") for p in paths)
    assert "fock.j" not in paths and "fock.k" not in paths
    for name in ("setup.one_electron", "setup.jk_construction",
                 "setup.guess", "setup.scf_init", "diagonalization",
                 "density", "final_rebuild", "final_rebuild/j",
                 "final_rebuild/k", "final_rebuild/diagonalization"):
        assert paths[name].inclusive_s > 0.0, name


def test_rhf_df_children_are_measured_kernels():
    res = _run("RHF", "df", True)
    paths = _paths(res.component_timing)
    fused = paths["fock.jk_fused"]
    j = paths["fock.jk_fused/df.j"]
    k = paths["fock.jk_fused/df.k"]
    assert j.calls == k.calls == fused.calls
    assert fused.inclusive_s >= j.inclusive_s + k.inclusive_s - 1e-12
    assert "setup.jk_construction/df.construction" in paths
    mem = {m.label: m.bytes for m in res.component_timing.memory}
    assert 0.0 < mem["df.live_tracked_bytes"] <= mem["df.construction_tracked_peak_bytes"]
    assert mem["df.tensor_extent_bytes"] > 0.0


def test_pbe_has_no_exchange_timer_and_times_xc():
    res = _run("PBE", "df", True)
    paths = _paths(res.component_timing)
    assert "fock.j" in paths and "xc" in paths and "setup.grids/xc_grid" in paths
    assert not any(p.split("/")[-1] in ("k", "fock.k", "df.k", "cosx.k")
                   for p in paths)


def test_hybrid_rks_separates_j_k_xc():
    paths = _paths(_run("PBE0", "direct", True).component_timing)
    for name in ("fock.j", "fock.k", "xc", "accelerator", "final_rebuild/xc"):
        assert paths[name].inclusive_s > 0.0, name


def test_iteration_buckets_match_iterations():
    res = _run("RHF", "conventional", True)
    its = list(res.component_timing.iterations)
    assert [it.iter for it in its] == list(range(1, res.n_iter + 1))
    assert {it.segment for it in its} == {0}
    assert res.component_timing.n_segments == 1
    assert all(it.wall_s > 0.0 for it in its)


def test_staged_cosx_records_segments():
    res = _run("PBE0", "rijcosx", True)
    ct = res.component_timing
    paths = _paths(ct)
    assert ct.n_segments >= 1
    assert {it.segment for it in ct.iterations} == set(range(ct.n_segments))
    assert any(p.endswith("cosx.k") for p in paths)


def test_lower_level_driver_owns_its_session():
    mol = _h2o()
    bs = vq.BasisSet(mol, "sto-3g")
    S = vq.compute_overlap(bs)
    H = vq.compute_kinetic(bs) + vq.compute_nuclear(bs, mol)
    jk = vq.make_four_index_jk_builder(bs)
    opts = _opts(vq.RHFOptions, route="conventional", timing=True)
    opts.initial_guess = vq.InitialGuess.HCORE
    res = vq.run_rhf_scf_with_jk(bs, mol.n_electrons(), S, H,
                                 mol.nuclear_repulsion(), jk, opts)
    ct = res.component_timing
    assert ct.enabled and ct.n_segments == 1
    _assert_reconciled(ct)


def test_native_build_info_reports_compiler():
    info = vq._vibeqc_core.native_build_info()
    assert info["compiler_id"] in {"clang", "gcc", "msvc", "unknown"}
    assert isinstance(info["eigen_version"], str)
    assert vq._vibeqc_core.resolve_scf_mode(vq.SCFMode.AUTO, 141, 140) == vq.SCFMode.DIRECT
    assert vq._vibeqc_core.resolve_scf_mode(vq.SCFMode.AUTO, 140, 140) == vq.SCFMode.CONVENTIONAL
    assert vq._vibeqc_core.resolve_scf_mode(vq.SCFMode.DIRECT, 10, 140) == vq.SCFMode.DIRECT
