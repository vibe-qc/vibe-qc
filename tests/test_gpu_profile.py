"""G0-A profiling harness: manifest, matched settings, provenance, redaction.

Synthetic records check the summary semantics (identity, unavailable
measurements, timing reconciliation, instrumentation on/off pairing). These
are bookkeeping checks, not profiling evidence.
"""
from __future__ import annotations

import copy
import json

import pytest

from examples.regression import gpu_profile as gp
from examples.regression.parity_matrix_orca import cases


def test_manifest_is_the_fixed_g0_matrix():
    rows = gp.build_manifest()
    assert len(rows) == 3 * 3 * 4
    assert {r["system"] for r in rows} == {"H2O", "naphthalene", "n-hexadecane"}
    assert {r["basis"] for r in rows} == {"def2-svp"}
    assert {r["method"] for r in rows} == {"RHF", "PBE", "PBE0"}
    assert {r["route"] for r in rows} == {"direct", "df", "rijcosx", "auto"}
    assert len({r["case_id"] for r in rows}) == len(rows)
    for r in rows:
        assert r["charge"] == 0 and r["multiplicity"] == 1


def test_manifest_uses_regression_geometry_single_conversion():
    for row in gp.build_manifest(methods=["RHF"], routes=["direct"]):
        atoms = cases.geometry_bohr(row["system"])
        assert row["atoms_bohr"] == [[z, list(x)] for z, x in atoms]
        assert row["geometry_sha256"] == gp.geometry_sha256(atoms)
    h2o = cases.geometry_bohr("H2O")
    moved = [[z, [c + 1e-12 for c in xyz]] for z, xyz in h2o]
    assert gp.geometry_sha256(moved) != gp.geometry_sha256(h2o)


@pytest.mark.parametrize("route,expected", [
    ("direct", {"density_fit": False, "cosx": False, "scf_mode": "DIRECT", "aux_basis": ""}),
    ("df", {"density_fit": True, "cosx": False, "scf_mode": "AUTO",
            "aux_basis": "def2-universal-jkfit"}),
    ("rijcosx", {"density_fit": True, "cosx": True, "scf_mode": "AUTO",
                 "aux_basis": "def2-universal-jfit"}),
    ("auto", {"density_fit": False, "cosx": False, "scf_mode": "AUTO", "aux_basis": ""}),
])
def test_route_requests(route, expected):
    assert gp.route_request(route, "RHF") == expected


def test_unknown_inputs_are_refused():
    with pytest.raises(ValueError):
        gp.route_request("semidirect", "RHF")
    with pytest.raises(ValueError):
        gp.build_manifest(systems=["benzene"])
    with pytest.raises(ValueError):
        gp.build_manifest(methods=["B3LYP"])


def test_options_carry_the_recipe_without_a_hidden_clamp():
    vq = pytest.importorskip("vibeqc")
    case = gp.build_manifest(systems=["H2O"], methods=["PBE0"], routes=["direct"])[0]
    _runner, opts = gp.make_options(case, component_timing=True)
    assert opts.conv_tol_energy == 1e-9  # the speed runner clamps this to 1e-7
    assert opts.conv_tol_grad == 1e-6
    assert opts.max_iter == 200
    assert opts.initial_guess == vq.InitialGuess.SAD
    assert opts.scf_mode == vq.SCFMode.DIRECT
    assert opts.density_fit is False and opts.cosx is False
    assert opts.functional == "PBE0"
    assert opts.component_timing is True
    g = opts.grid
    assert (g.n_radial, g.n_theta, g.n_phi, g.becke_k) == (75, 17, 36, 3)
    view = gp.serialize(opts)
    assert view["conv_tol_energy"] == 1e-9
    assert view["scf_mode"] == "DIRECT"
    assert view["grid"]["angular_pruning"] == "None"
    json.dumps(view)


def test_matched_settings_differ_only_in_the_route():
    pytest.importorskip("vibeqc")
    views = {}
    for route in gp.ROUTES:
        case = gp.build_manifest(systems=["H2O"], methods=["PBE0"], routes=[route])[0]
        views[route] = gp.serialize(gp.make_options(case, component_timing=False)[1])
    route_fields = {"density_fit", "cosx", "aux_basis", "scf_mode", "cosx_grid_level"}
    base = {k: v for k, v in views["direct"].items() if k not in route_fields}
    for route, view in views.items():
        assert {k: v for k, v in view.items() if k not in route_fields} == base, route


def test_resolution_auto_is_not_always_direct():
    vq = pytest.importorskip("vibeqc")
    out = {}
    for system in ("H2O", "n-hexadecane"):
        for route in ("direct", "auto"):
            case = gp.build_manifest(systems=[system], methods=["RHF"], routes=[route])[0]
            mol = vq.Molecule([vq.Atom(z, xyz) for z, xyz in case["atoms_bohr"]])
            basis = vq.BasisSet(mol, case["basis"])
            _r, opts = gp.make_options(case, component_timing=False)
            resolved = gp.resolve_settings(case, mol, basis, opts)
            out[system, route] = (resolved, gp.dedupe_key(case, resolved, gp.serialize(opts)))
    h2o_auto = out["H2O", "auto"][0]
    assert h2o_auto["n_ao"] <= 140
    assert h2o_auto["scf_mode_resolved"] == "CONVENTIONAL"
    assert out["H2O", "auto"][1] != out["H2O", "direct"][1]
    assert out["n-hexadecane", "auto"][0]["scf_mode_resolved"] == "DIRECT"
    assert out["n-hexadecane", "auto"][1] == out["n-hexadecane", "direct"][1]


def test_pbe_exchange_is_not_applicable():
    vq = pytest.importorskip("vibeqc")
    case = gp.build_manifest(systems=["H2O"], methods=["PBE"], routes=["rijcosx"])[0]
    mol = vq.Molecule([vq.Atom(z, xyz) for z, xyz in case["atoms_bohr"]])
    basis = vq.BasisSet(mol, case["basis"])
    _r, opts = gp.make_options(case, component_timing=False)
    resolved = gp.resolve_settings(case, mol, basis, opts)
    assert resolved["k_build"] == "not_applicable"
    assert resolved["builder"] == "ri_j_only"
    assert resolved["cosx_grid_level"] == 2
    assert len(resolved["cosx_stage_points"]) >= 1
    assert resolved["n_aux"] > 0 and resolved["xc_grid_points"] > 0


def test_runtime_provenance_fields():
    pytest.importorskip("vibeqc")
    prov = gp.runtime_provenance()
    for key in ("vibeqc_file", "native_file", "native_sha256", "build_info",
                "native_build_info", "blas_info", "thread_env", "cpu_model"):
        assert key in prov, key
    assert len(prov["native_sha256"]) == 64
    assert prov["native_build_info"]["compiler_id"]


def test_plan_batch_warmup_first_then_shuffled():
    rows = gp.build_manifest(systems=["H2O"], methods=["RHF"], routes=["df", "auto"])
    runs = gp.plan_batch(rows, threads=[1, 4], timing_modes=[True, False],
                         warmup=1, repeats=3, seed=7)
    warm = [r for r in runs if r["label"].startswith("warmup")]
    timed = [r for r in runs if not r["label"].startswith("warmup")]
    assert runs[:len(warm)] == warm and len(warm) == 2 * 2
    assert len(timed) == 2 * 2 * 2 * 3
    assert runs == gp.plan_batch(rows, threads=[1, 4], timing_modes=[True, False],
                                 warmup=1, repeats=3, seed=7)


# ---- redaction ---------------------------------------------------------


def test_redact_strips_paths_and_private_terms():
    terms = ["builder-node", "/srv/runner-a"]
    raw = {"file": "/srv/runner-a/venv/lib/vibeqc/_core.so",
           "note": "ran on builder-node.example",
           "rel": "examples/regression/gpu_profile.py",
           "num": 3}
    out = gp.redact(raw, terms)
    assert out["file"] == "_core.so"
    assert "builder-node" not in out["note"]
    assert out["rel"] == "examples/regression/gpu_profile.py"
    assert out["num"] == 3
    gp.assert_public_safe(out, terms)
    with pytest.raises(ValueError):
        gp.assert_public_safe({"p": "/opt/build/x.so"}, terms)
    with pytest.raises(ValueError):
        gp.assert_public_safe({"p": "builder-node"}, terms)


# ---- timing accounting --------------------------------------------------


def _timing(unclassified=0.1):
    phases = [
        {"path": "setup.jk_construction", "inclusive_s": 1.0, "exclusive_s": 0.2, "calls": 1},
        {"path": "setup.jk_construction/df.construction", "inclusive_s": 0.8,
         "exclusive_s": 0.8, "calls": 1},
        {"path": "fock.jk_fused", "inclusive_s": 2.0, "exclusive_s": 0.1, "calls": 10},
        {"path": "fock.jk_fused/df.j", "inclusive_s": 0.4, "exclusive_s": 0.4, "calls": 10},
        {"path": "fock.jk_fused/df.k", "inclusive_s": 1.5, "exclusive_s": 1.5, "calls": 10},
        {"path": "diagonalization", "inclusive_s": 0.3, "exclusive_s": 0.3, "calls": 10},
    ]
    wall = sum(p["exclusive_s"] for p in phases) + unclassified
    return {"enabled": True, "wall_s": wall, "unclassified_s": unclassified,
            "phases": phases, "iterations": [], "memory": {}}


def test_group_phases_and_reconcile():
    t = _timing()
    groups = gp.group_phases(t)
    assert groups["df_construction"] == pytest.approx(0.8)
    assert groups["jk_fused"] == pytest.approx(0.1)
    assert groups["j"] == pytest.approx(0.4)
    assert groups["k"] == pytest.approx(1.5)
    assert groups["setup"] == pytest.approx(0.2)
    assert sum(groups.values()) == pytest.approx(t["wall_s"])
    assert gp.reconcile(t)["ok"]
    bad = copy.deepcopy(t)
    bad["wall_s"] += 0.5
    assert not gp.reconcile(bad)["ok"]
    neg = copy.deepcopy(t)
    neg["phases"][0]["exclusive_s"] = -0.01
    assert not gp.reconcile(neg)["ok"]


# ---- summary semantics ----------------------------------------------------


def _record(case_id="H2O__def2-svp__RHF__df", *, ct=True, label="rep0",
            energy=-76.0, wall=1.0, sha="a" * 40, native="b" * 64, dirty=False,
            converged=True):
    timing = _timing() if ct else {"enabled": False}
    if ct:
        timing["reconciliation"] = gp.reconcile(timing)
    return {
        "schema": gp.SCHEMA, "run_label": label, "component_timing": ct,
        "case": {"case_id": case_id, "route": "df", "method": "RHF",
                 "geometry_sha256": "c" * 64},
        "threads_requested": 1, "threads_observed": 1,
        "resolved": {"builder": "df_j_k", "n_ao": 24, "n_aux": 100, "n_occ": 5,
                     "basis_sha256": "d" * 64, "k_build": "exact_or_df"},
        "dedupe_key": "e" * 64, "scf_wall_s": wall, "output_s": 0.01,
        "result": {"energy": energy, "converged": converged, "n_iter": 10,
                   "final_grad_norm": 1e-7, "stability_wall_s": 0.2},
        "memory": {"process_peak_rss_bytes": 1000, "tracked_live_tensor_bytes": 10,
                   "tracked_construction_peak_bytes": 20},
        "timing": timing,
        "provenance": {"build_info": {"sha_full": sha, "dirty": dirty},
                       "native_sha256": native,
                       "native_build_info": {"compiler_version": "clang 1"},
                       "blas_info": {"libraries": "/private/lib/libblas.so"},
                       "cpu_model": "generic"},
    }


def test_summary_pass_and_warmup_excluded():
    recs = [_record(label="warmup0", energy=-1.0)]
    recs += [_record(label=f"rep{i}", wall=1.0 + i) for i in range(3)]
    s = gp.summarize(recs, expected_cases=["H2O__def2-svp__RHF__df"])
    (row,) = s["rows"]
    assert row["status"] == "PASS" and row["n_runs"] == 3
    assert row["scf_wall_s"] == {"median": 2.0, "min": 1.0, "max": 3.0, "n": 3}
    assert row["blas"] == "libblas.so"
    assert s["incomplete_cases"] == []
    gp.assert_public_safe(s)


def test_summary_identity_and_unavailable_measurements():
    mixed = [_record(), _record(label="rep1", native="f" * 64)]
    assert gp.summarize(mixed)["rows"][0]["status"] == "FAIL"
    dirty = gp.summarize([_record(dirty=True)])["rows"][0]
    assert dirty["status"] == "FAIL"
    assert dirty["fail_reasons"] == ["source_tree_dirty"]
    wrong = gp.summarize([_record()], expected_source_sha="0" * 40)
    assert wrong["rows"][0]["identity_ok"] is False
    unconverged = gp.summarize([_record(converged=False)])
    assert unconverged["rows"][0]["status"] == "FAIL"
    missing = gp.summarize([_record()], expected_cases=[
        "H2O__def2-svp__RHF__df", "n-hexadecane__def2-svp__RHF__df"])
    assert missing["incomplete_cases"] == ["n-hexadecane__def2-svp__RHF__df"]
    off_only = gp.summarize([_record(ct=False)])
    assert off_only["rows"][0]["phase_exclusive_s"] == {}
    assert off_only["rows"][0]["timing_reconciled"] is None


def test_timing_on_off_comparison():
    recs = [_record(label=f"rep{i}", ct=True, wall=1.02) for i in range(3)]
    recs += [_record(label=f"rep{i}", ct=False, wall=1.0) for i in range(3)]
    s = gp.summarize(recs)
    (cmp_row,) = s["timing_on_off"]
    assert cmp_row["energy_delta_eh"] == 0.0
    assert cmp_row["iterations_equal"]
    assert cmp_row["overhead_fraction"] == pytest.approx(0.02)
    assert cmp_row["ranges_overlap"] is False


def test_render_markdown_uses_measured_values_only():
    recs = [_record(label=f"rep{i}", ct=True, wall=1.0) for i in range(3)]
    recs += [_record(label=f"rep{i}", ct=False, wall=1.0) for i in range(3)]
    md = gp.render_markdown(gp.summarize(recs))
    assert "H2O__def2-svp__RHF__df" in md and "PASS" in md
    assert "TODO" not in md and "TBD" not in md


def test_residual_above_recipe_is_flagged_not_hidden():
    rec = _record()
    rec["result"]["final_grad_norm"] = 1e-3
    row = gp.summarize([rec])["rows"][0]
    assert row["final_grad_above_recipe"] is True
    assert "above target" in gp.render_markdown(gp.summarize([rec]))
    assert gp.summarize([_record()])["rows"][0]["final_grad_above_recipe"] is False


def test_deduplicate_keeps_both_labels_only_for_identical_settings():
    pytest.importorskip("vibeqc")
    hexa = gp.build_manifest(systems=["n-hexadecane"], methods=["RHF"],
                             routes=["direct", "auto"])
    (kept,) = gp.deduplicate_cases(hexa)
    assert kept["route"] == "direct"
    assert kept["aliases"] == ["n-hexadecane__def2-svp__RHF__auto"]
    h2o = gp.build_manifest(systems=["H2O"], methods=["RHF"], routes=["direct", "auto"])
    assert [c["route"] for c in gp.deduplicate_cases(h2o)] == ["direct", "auto"]
    rec = _record()
    rec["case"]["aliases"] = ["x__auto"]
    s = gp.summarize([rec])
    assert s["rows"][0]["aliases"] == ["x__auto"]
    assert "(= auto)" in gp.render_markdown(s)
