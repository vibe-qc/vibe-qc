"""``vibeqc.output.verdict`` -- the one run-level verdict every writer shares.

Pure Python: no native core. Pins the vocabulary and the consumer gate the
molecular runner, the manifest, the QVF and the structured log carry (#342,
#346, #351).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibeqc.output.verdict import (
    RunVerdict,
    describe_stability_stop,
    usable_from_manifest,
    verdict_from_result,
    verdict_section,
)


def _res(**overrides):
    base = dict(
        converged=True,
        stability_checked=True,
        stability_analysis_converged=True,
        internal_instability=False,
        stability_eigenvalue=2.2e-3,
        n_stability_restarts=0,
        stability_wall_s=1.5,
        stability_stop_reason=0,
        stability_n_matvec=12,
        stability_wall_budget_s=600.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_certified_stable_solution_is_usable():
    v = verdict_from_result(_res(), stability_requested=True, matvec_cap=60)
    assert v.scf == "converged"
    assert v.stability == "verified_stable"
    assert v.stability_reason == "converged"
    assert v.usable
    assert v.result_status == "converged"
    assert v.stability_matvec_cap == 60
    assert v.stability_matvec_used == 12


@pytest.mark.parametrize(
    "native_reason, reason",
    [(1, "matvec_budget"), (2, "wall_budget"), (3, "not_run"), (5, "stagnated")],
)
def test_no_verdict_is_unverified_and_not_usable(native_reason, reason):
    v = verdict_from_result(
        _res(stability_analysis_converged=False, stability_stop_reason=native_reason)
    )
    assert v.stability == "unverified"
    assert v.stability_reason == reason
    assert not v.usable
    # The SCF itself converged: result_status stays the SCF axis.
    assert v.result_status == "converged"


def test_unstable_solution_is_not_usable():
    v = verdict_from_result(_res(internal_instability=True, stability_eigenvalue=-1e-3))
    assert v.stability == "unstable"
    assert v.stability_reason == "converged"
    assert not v.usable


def test_opt_out_is_visible_and_usable():
    v = verdict_from_result(_res(stability_checked=False), stability_requested=False)
    assert v.stability == "not_checked"
    assert v.stability_reason == "opt_out"
    assert v.usable


@pytest.mark.parametrize("requested", [True, None])
def test_unsupported_route_is_visible_and_usable(requested):
    v = verdict_from_result(_res(stability_checked=False), stability_requested=requested)
    assert v.stability == "unsupported"
    assert v.stability_reason == "unsupported_route"
    assert v.usable


def test_result_without_a_stability_surface_is_not_applicable():
    v = verdict_from_result(SimpleNamespace(converged=True))
    assert v.stability == "not_applicable"
    assert v.usable


def test_non_converged_scf_is_never_usable():
    v = verdict_from_result(_res(converged=False, stability_checked=False))
    assert v.scf == "not_converged"
    assert v.stability == "not_applicable"
    assert not v.usable
    assert v.result_status == "not_converged"


def test_older_core_without_a_stop_reason_reports_unreported():
    v = verdict_from_result(
        SimpleNamespace(
            converged=True, stability_checked=True,
            stability_analysis_converged=False, internal_instability=False,
            stability_eigenvalue=0.0,
        )
    )
    assert v.stability == "unverified"
    assert v.stability_reason == "unreported"
    assert not v.usable


def test_as_dict_is_fixed_shape_and_round_trips():
    v = verdict_from_result(_res(), matvec_cap=60)
    d = v.as_dict()
    assert list(d) == [
        "scf", "stability", "stability_reason", "usable",
        "reference_usable", "stability_scope", "internal_stability", "external_stability",
        "internal_stability_eigenvalue", "external_stability_eigenvalue",
        "internal_stability_reason", "external_stability_reason", "reference_contract",
        "stability_eigenvalue", "stability_restarts", "stability_matvec_used",
        "stability_matvec_cap", "stability_wall_s", "stability_wall_budget_s",
    ]
    assert d["usable"] is True
    assert RunVerdict.from_dict(d) == v
    assert RunVerdict.from_dict(d).as_dict() == d


def test_unknown_vocabulary_is_refused():
    with pytest.raises(ValueError):
        RunVerdict(scf="done", stability="verified_stable", stability_reason="converged")
    with pytest.raises(ValueError):
        RunVerdict(scf="converged", stability="fine", stability_reason="converged")


def test_verdict_section_derives_from_result_status_when_nothing_was_recorded():
    body = verdict_section(None, "converged")
    assert body["scf"] == "converged"
    assert body["stability"] == "not_reported"
    assert body["usable"] is True
    assert verdict_section(None, "not_converged")["usable"] is False
    assert verdict_section(None, "refused")["scf"] == "not_reported"
    recorded = verdict_from_result(_res(stability_analysis_converged=False)).as_dict()
    assert verdict_section(recorded, "converged") == recorded


def test_manifest_gate_fails_closed_on_history_and_reads_the_verdict():
    # Every manifest written before the contract: delivery only.
    assert usable_from_manifest({"outputs": {"status": "complete"}}) == "unknown"
    # A writer that never recorded a result: still unknown, whatever [verdict] says.
    derived = verdict_section(None, "not_reported")
    assert usable_from_manifest(
        {"outputs": {"status": "complete", "result_status": "not_reported"}, "verdict": derived}
    ) == "unknown"
    certified = verdict_from_result(_res()).as_dict()
    assert usable_from_manifest(
        {"outputs": {"status": "complete", "result_status": "converged"}, "verdict": certified}
    ) == "usable"
    unverified = verdict_from_result(_res(stability_analysis_converged=False)).as_dict()
    assert usable_from_manifest(
        {"outputs": {"status": "complete", "result_status": "converged"}, "verdict": unverified}
    ) == "not_usable"
    # A delivered-then-failed run keeps its usable energy (#346 criterion 3).
    assert usable_from_manifest(
        {"outputs": {"status": "crashed", "result_status": "converged"}, "verdict": certified}
    ) == "usable"


def test_stop_description_names_the_reason_and_the_cost():
    assert describe_stability_stop(_res(stability_stop_reason=3)) == "did not run (matvec cap 0)"
    assert "matvec cap after 12" in describe_stability_stop(_res(stability_stop_reason=1))
    text = describe_stability_stop(
        _res(stability_stop_reason=2, stability_n_matvec=17, stability_wall_s=612.4,
             stability_wall_budget_s=600.0)
    )
    assert text == (
        "stopped by the wall budget after 17 Hessian-vector products, 612.4 s of 600.0 s"
    )
    assert describe_stability_stop(SimpleNamespace()) == "did not converge"


def _restricted_result(**overrides):
    return _res(restricted_stability_channels=True,
                stability_internal_converged=True, stability_external_converged=True,
                stability_internal_eigenvalue=0.2, stability_external_eigenvalue=-0.003,
                external_instability=True, stability_eigenvalue=-0.003,
                **overrides)


def test_external_instability_stays_unusable_but_explicit_restricted_reference_is_separate():
    result = _restricted_result()
    default = verdict_from_result(result)
    assert not default.usable and not default.reference_usable
    assert default.internal_stability == "verified_stable"
    assert default.external_stability == "unstable"
    explicit = verdict_from_result(result, reference_contract="real_restricted_cc")
    assert not explicit.usable and explicit.reference_usable
    assert RunVerdict.from_dict(explicit.as_dict()) == explicit


@pytest.mark.parametrize("failure", ["internal_instability", "internal_unverified", "external_unverified", "legacy", "nonconverged", "nonfinite"])
def test_restricted_reference_does_not_admit_bad_or_missing_evidence(failure):
    result = _restricted_result()
    if failure == "internal_instability": result.internal_instability = True
    elif failure == "internal_unverified": result.stability_internal_converged = False
    elif failure == "external_unverified": result.stability_external_converged = False
    elif failure == "legacy": result.restricted_stability_channels = False
    elif failure == "nonconverged": result.converged = False
    elif failure == "nonfinite": result.stability_internal_eigenvalue = float("nan")
    assert not verdict_from_result(result, reference_contract="real_restricted_cc").reference_usable
