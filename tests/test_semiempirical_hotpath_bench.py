from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_bench_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_semiempirical_hotpaths.py"
    spec = importlib.util.spec_from_file_location("bench_semiempirical_hotpaths", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hotpath_inventory_lists_core_methods_without_importing_vibeqc(capsys):
    bench = _load_bench_module()

    assert "dftb0_h2o_energy" in bench.CASE_BY_KEY
    assert "dftb0_h2o_gradient" in bench.CASE_BY_KEY
    assert "dftb0_h2o_preoptimize" in bench.CASE_BY_KEY
    assert "dftb0_benzene_energy" in bench.CASE_BY_KEY
    assert "gfn2_h2o_energy" in bench.CASE_BY_KEY
    assert "gfn2_h2o_gradient" in bench.CASE_BY_KEY
    assert "gfn2_nh3_energy" in bench.CASE_BY_KEY
    assert "gfn2_benzene_energy" in bench.CASE_BY_KEY
    assert "gfn2_nacl_energy" in bench.CASE_BY_KEY
    assert "gfn2_lif_energy" in bench.CASE_BY_KEY
    assert "gfn2_lih_energy" in bench.CASE_BY_KEY
    assert "pm6_h2o_gradient_fd" in bench.CASE_BY_KEY
    assert "pm6_ch4_energy" in bench.CASE_BY_KEY
    assert "pm6_nh3_energy" in bench.CASE_BY_KEY
    assert "om2_ch4_energy" in bench.CASE_BY_KEY
    assert "om2_nh3_energy" in bench.CASE_BY_KEY
    assert "msindo_h2o_cpp_energy" in bench.CASE_BY_KEY
    assert "msindo_h2o_direct_energy" in bench.CASE_BY_KEY
    assert "msindo_nddo_hf_direct_energy" in bench.CASE_BY_KEY
    assert "msindo_sbf3_direct_energy" in bench.CASE_BY_KEY
    assert "msindo_xef2_direct_energy" in bench.CASE_BY_KEY
    assert "msindo_cosmo_h2o" in bench.CASE_BY_KEY
    assert "msindo_cis_h2o_singlet" in bench.CASE_BY_KEY
    assert "ccm_he7_1d_noewald" in bench.CASE_BY_KEY
    assert "ccm_hf6_1d_madelung" in bench.CASE_BY_KEY
    assert "ccm_mgo_2d_noewald" in bench.CASE_BY_KEY
    assert "ccm_mgo_3d_noewald" in bench.CASE_BY_KEY
    assert "periodic_dftb0_he_gradient" in bench.CASE_BY_KEY
    assert "periodic_dftb0_he_stress" in bench.CASE_BY_KEY
    assert "periodic_dftb0_quartz_sio2" in bench.CASE_BY_KEY
    assert "periodic_pm6_he_dimer_gradient_fd" in bench.CASE_BY_KEY
    assert "periodic_pm6_he_dimer_stress_fd" in bench.CASE_BY_KEY

    bench.main(["--list"])
    out = capsys.readouterr().out
    assert "dftb0_h2o_gradient" in out
    assert "dftb0_h2o_preoptimize" in out
    assert "dftb0_benzene_energy" in out
    assert "gfn2_h2o_gradient" in out
    assert "gfn2_nh3_energy" in out
    assert "gfn2_benzene_energy" in out
    assert "gfn2_lif_energy" in out
    assert "gfn2_lih_energy" in out
    assert "pm6_h2o_gradient_fd" in out
    assert "pm6_ch4_energy" in out
    assert "pm6_nh3_energy" in out
    assert "om2_ch4_energy" in out
    assert "om2_nh3_energy" in out
    assert "C++ SCC core" in out
    assert "C++ through public run_msindo" in out
    assert "msindo_sbf3_direct_energy" in out
    assert "msindo_xef2_direct_energy" in out
    assert "msindo_cosmo_h2o" in out
    assert "msindo_cis_h2o_singlet" in out
    assert "ccm_he7_1d_noewald" in out
    assert "ccm_mgo_2d_noewald" in out
    assert "ccm_mgo_3d_noewald" in out
    assert "periodic_dftb0_he_gradient" in out
    assert "periodic_dftb0_he_stress" in out
    assert "periodic_dftb0_quartz_sio2" in out
    assert "periodic_pm6_he_dimer_gradient_fd" in out
    assert "periodic_pm6_he_dimer_stress_fd" in out
    assert "C++ native FD batch" in out
    assert "GMTKN55" in out
    assert "Dral-Thiel ground-state OMx/ODMx" in out
    assert "C++ B-matrix/vector ops + Python COSMO SCF" in out
    assert "C++" in out
    assert "HANDOVER" not in out


def test_article_external_suite_mapping_distinguishes_future_scope(capsys):
    bench = _load_bench_module()

    assert "gfn2_h2o_energy" in bench._case_keys_for_external_suite("GMTKN55")
    assert "om2_ch4_energy" in bench._case_keys_for_external_suite(
        "Dral-Thiel ground-state OMx/ODMx"
    )
    assert "periodic_dftb0_quartz_sio2" in bench._case_keys_for_external_suite(
        "Delta-factor / SSSP solids"
    )
    assert bench._case_keys_for_external_suite("Thiel excited-state family") == []

    bench.main(["--list-suites"])
    out = capsys.readouterr().out
    assert "compact-runnable molecular representatives" in out
    assert "future-driver excited-state route" in out
    assert "MACE literature validation suites" in out
    assert "`periodic_dftb0_quartz_sio2`" in out
    assert "| Thiel excited-state family | future-driver excited-state route | - |" in out


def test_hotpath_case_selection_rejects_unknown_case():
    bench = _load_bench_module()

    try:
        bench._select_cases("smoke", "nope")
    except SystemExit as exc:
        assert "unknown case" in str(exc)
    else:
        raise AssertionError("unknown case was accepted")


def test_hotpath_runner_fails_error_rows(monkeypatch):
    bench = _load_bench_module()

    def fake_run_parent(cases, repeat):
        assert repeat == 1
        return [
            {
                "case": bench.asdict(cases[0]),
                "status": "error",
                "error": "calculation failed",
            }
        ]

    monkeypatch.setattr(bench, "_run_parent", fake_run_parent)

    assert bench.main(["--only", "dftb0_h2o_energy"]) == 1


def test_hotpath_runner_fails_nonconverged_ok_rows(monkeypatch):
    bench = _load_bench_module()

    def fake_run_parent(cases, repeat):
        assert repeat == 1
        return [
            {
                "case": bench.asdict(cases[0]),
                "status": "ok",
                "wall_time_s": 0.01,
                "rss_max_mb": 1.0,
                "result": {"energy_ha": -1.0, "converged": False},
            }
        ]

    monkeypatch.setattr(bench, "_run_parent", fake_run_parent)

    assert bench.main(["--only", "gfn2_nacl_energy"]) == 1
