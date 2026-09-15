"""The L1 cohesive objective: algebra, guards, and the held-out split.

Every number here is synthetic. M2's own gate -- evaluating the objective at
the shipped pob-TZVP-REV2 and reproducing the recorded GPAW-pob statistics
(MD -24.6, MAD 35.4 kJ/mol over n=28) -- needs 28 periodic cohesive-energy
runs and has NOT been run; see HANDOVER_BASISOPT.md § M2. What is pinned here
is that the algebra is right and the guards fire, so that when the gate is
run, a disagreement means the physics, not the arithmetic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pytest

from vibe_basis.objective import (
    CohesiveObjective,
    ObjectiveError,
    load_pw_reference,
    split_validation,
)


@dataclass
class FakeResult:
    """Duck-types vibe_basis.pipeline.CohesiveResult's scored fields."""

    system: str
    cohesive_kj_per_mol: float | None = None
    ok: bool = True
    zero_point_applied: bool = False
    counterpoise_applied: bool = True


REF = {"a": 100.0, "b": 200.0, "c": 300.0, "d": 400.0}


# ---------------------------------------------------------------------------
# Reference loading
# ---------------------------------------------------------------------------


def _write(tmp_path, payload):
    p = tmp_path / "pwref.json"
    p.write_text(json.dumps(payload))
    return p


def test_load_pw_reference_reads_the_merged_layout(tmp_path):
    p = _write(tmp_path, {
        "provenance": {"note": "merged sidecars"},
        "results": [
            {"system_id": "alas_zincblende", "pw_limit_kjmol": 744.8625211955979},
            {"system_id": "mgo_rocksalt", "pw_limit_kjmol": 1000.5},
        ],
    })
    assert load_pw_reference(p) == {
        "alas_zincblende": 744.8625211955979,
        "mgo_rocksalt": 1000.5,
    }


def test_load_pw_reference_refuses_a_duplicated_system(tmp_path):
    """A de-duplicated reference would weight one system twice in the sum."""
    p = _write(tmp_path, {"results": [
        {"system_id": "a", "pw_limit_kjmol": 1.0},
        {"system_id": "a", "pw_limit_kjmol": 2.0},
    ]})
    with pytest.raises(ObjectiveError, match="appears twice"):
        load_pw_reference(p)


@pytest.mark.parametrize("payload, match", [
    ([], "'results' key"),
    ({"results": []}, "empty"),
    ({"results": [{"system_id": "a"}]}, "needs 'system_id'"),
    ({"results": [{"system_id": "a", "pw_limit_kjmol": "x"}]}, "needs 'system_id'"),
])
def test_load_pw_reference_refuses_malformed_files(tmp_path, payload, match):
    with pytest.raises(ObjectiveError, match=match):
        load_pw_reference(_write(tmp_path, payload))


# ---------------------------------------------------------------------------
# The algebra
# ---------------------------------------------------------------------------


def test_residual_sign_is_candidate_minus_reference():
    """Positive mean deviation means the candidate overbinds."""
    obj = CohesiveObjective(REF)
    rep = obj.report({"a": 110.0, "b": 200.0, "c": 300.0, "d": 400.0})
    assert rep.residuals["a"] == pytest.approx(10.0)
    assert rep.train.md == pytest.approx(10.0 / 4)


def test_unweighted_losses_match_their_definitions():
    cand = {"a": 101.0, "b": 198.0, "c": 300.0, "d": 405.0}
    deltas = [1.0, -2.0, 0.0, 5.0]
    rep = CohesiveObjective(REF, loss="rmsd").report(cand)
    assert rep.train.mad == pytest.approx(sum(abs(d) for d in deltas) / 4)
    assert rep.train.rmsd == pytest.approx(
        math.sqrt(sum(d * d for d in deltas) / 4)
    )
    assert rep.train.max_abs == pytest.approx(5.0)
    assert rep.train.worst_system == "d"
    assert rep.value == pytest.approx(rep.train.rmsd)

    assert CohesiveObjective(REF, loss="mad")(cand) == pytest.approx(rep.train.mad)
    assert CohesiveObjective(REF, loss="maxabs")(cand) == pytest.approx(5.0)


def test_huber_is_quadratic_inside_delta_and_linear_outside():
    # One system only, so the mean is the per-system term itself.
    inside = CohesiveObjective({"a": 100.0}, loss="huber", huber_delta=10.0)
    assert inside({"a": 104.0}) == pytest.approx(0.5 * 4.0**2)
    outside = CohesiveObjective({"a": 100.0}, loss="huber", huber_delta=10.0)
    assert outside({"a": 130.0}) == pytest.approx(10.0 * (30.0 - 5.0))


def test_weights_reweight_the_average():
    cand = {"a": 110.0, "b": 200.0, "c": 300.0, "d": 400.0}
    obj = CohesiveObjective(REF, loss="mad", weights={"a": 3.0})
    # weights 3,1,1,1 over |deltas| 10,0,0,0
    assert obj(cand) == pytest.approx(30.0 / 6.0)


def test_a_zero_or_negative_weight_is_refused():
    """Silently dropping a system is what validation= is for."""
    with pytest.raises(ObjectiveError, match="every weight must be > 0"):
        CohesiveObjective(REF, weights={"a": 0.0})


# ---------------------------------------------------------------------------
# The held-out split
# ---------------------------------------------------------------------------


def test_validation_systems_are_reported_but_not_optimised_against():
    cand = {"a": 100.0, "b": 200.0, "c": 300.0, "d": 500.0}
    obj = CohesiveObjective(REF, loss="mad", validation=["d"])
    rep = obj.report(cand)
    assert rep.train.n == 3
    assert rep.train.mad == pytest.approx(0.0)
    assert rep.value == pytest.approx(0.0)          # d's 100 kJ/mol error is held out
    assert rep.validation is not None
    assert rep.validation.n == 1
    assert rep.validation.mad == pytest.approx(100.0)


def test_validation_is_none_when_no_split_was_configured():
    assert CohesiveObjective(REF).report({k: v for k, v in REF.items()}).validation is None


def test_holding_out_everything_is_refused():
    with pytest.raises(ObjectiveError, match="nothing to\noptimise|nothing to optimise"):
        CohesiveObjective(REF, validation=list(REF))


def test_validation_must_name_known_systems():
    with pytest.raises(ObjectiveError, match="absent from references"):
        CohesiveObjective(REF, validation=["zz"])


def test_split_validation_is_deterministic_and_disjoint():
    systems = [f"s{i}" for i in range(12)]
    held = split_validation(systems, every=4)
    assert held == split_validation(systems, every=4)   # reproducible
    # The sort is lexicographic, so the order is s0, s1, s10, s11, s2, ...
    # and indices 0, 4, 8 of that are s0, s2, s6. Surprising, but stable,
    # which is the property the split needs.
    assert held == ("s0", "s2", "s6")
    assert len(set(held)) == len(held)
    assert set(held) < set(systems)


@pytest.mark.parametrize("kwargs", [{"every": 1}, {"every": 4, "offset": 4}])
def test_split_validation_rejects_degenerate_parameters(kwargs):
    with pytest.raises(ObjectiveError):
        split_validation(["a", "b"], **kwargs)


# ---------------------------------------------------------------------------
# Missing systems: the failure mode that would reward breaking a system
# ---------------------------------------------------------------------------


def test_a_missing_system_is_refused_by_default():
    with pytest.raises(ObjectiveError, match=r"no usable cohesive energy"):
        CohesiveObjective(REF)({"a": 100.0, "b": 200.0, "c": 300.0})


def test_there_is_no_skip_policy():
    """Dropping unconverged systems would make the objective improve as the
    basis got worse enough to break them."""
    with pytest.raises(ObjectiveError, match="no 'skip'"):
        CohesiveObjective(REF, on_missing="skip")


def test_penalise_charges_per_missing_training_system():
    obj = CohesiveObjective(
        REF, loss="mad", on_missing="penalise", missing_penalty=500.0
    )
    rep = obj.report({"a": 100.0, "b": 200.0, "c": 300.0})
    assert rep.missing == ("d",)
    assert rep.train.n == 3                       # d contributes no residual
    assert rep.value == pytest.approx(500.0)      # but does cost 500


def test_a_missing_validation_system_does_not_steer_the_optimiser():
    """The held-out split is a diagnostic; charging it would make it an input."""
    obj = CohesiveObjective(
        REF, loss="mad", validation=["d"], on_missing="penalise",
        missing_penalty=500.0,
    )
    rep = obj.report({"a": 100.0, "b": 200.0, "c": 300.0})
    assert rep.missing == ("d",)
    assert rep.value == pytest.approx(0.0)


def test_a_failed_or_none_result_counts_as_missing_not_as_a_number():
    obj = CohesiveObjective(REF, loss="mad", on_missing="penalise",
                            missing_penalty=7.0)
    results = [
        FakeResult("a", 100.0),
        FakeResult("b", 200.0),
        FakeResult("c", None),                      # produced nothing
        FakeResult("d", 400.0, ok=False),           # ran but did not converge
    ]
    rep = obj.report(results)
    assert rep.missing == ("c", "d")
    assert rep.value == pytest.approx(14.0)


def test_all_training_systems_missing_is_refused_even_under_penalise():
    obj = CohesiveObjective(REF, on_missing="penalise")
    with pytest.raises(ObjectiveError, match="no training system"):
        obj.report({})


# ---------------------------------------------------------------------------
# Convention guards
# ---------------------------------------------------------------------------


def test_a_zero_point_mismatch_is_refused():
    obj = CohesiveObjective({"a": 100.0}, expect_zero_point=True)
    with pytest.raises(ObjectiveError, match="zero-point convention"):
        obj.report([FakeResult("a", 100.0, zero_point_applied=False)])


def test_a_counterpoise_mismatch_is_refused():
    obj = CohesiveObjective({"a": 100.0}, expect_counterpoise=False)
    with pytest.raises(ObjectiveError, match="counterpoise convention"):
        obj.report([FakeResult("a", 100.0, counterpoise_applied=True)])


def test_matching_conventions_pass_and_are_recorded_as_checked():
    obj = CohesiveObjective(
        {"a": 100.0}, expect_zero_point=False, expect_counterpoise=True
    )
    rep = obj.report([FakeResult("a", 100.0)])
    assert rep.unchecked_conventions == ()


def test_an_unpinned_convention_is_surfaced_rather_than_silent():
    rep = CohesiveObjective({"a": 100.0}).report({"a": 100.0})
    assert len(rep.unchecked_conventions) == 2


# ---------------------------------------------------------------------------
# The penalty seam (Tier 2 injects vibe-qc's ld_penalty here)
# ---------------------------------------------------------------------------


def test_penalty_is_added_to_the_loss_and_reported_separately():
    obj = CohesiveObjective(REF, loss="mad", penalty=lambda: 2.5)
    rep = obj.report({"a": 101.0, "b": 200.0, "c": 300.0, "d": 400.0})
    assert rep.loss_value == pytest.approx(0.25)
    assert rep.penalty_value == pytest.approx(2.5)
    assert rep.value == pytest.approx(2.75)


def test_a_constant_penalty_works_too():
    obj = CohesiveObjective(REF, loss="mad", penalty=3.0)
    assert obj({k: v for k, v in REF.items()}) == pytest.approx(3.0)


def test_a_non_finite_penalty_is_refused():
    obj = CohesiveObjective(REF, penalty=lambda: float("nan"))
    with pytest.raises(ObjectiveError, match="non-finite"):
        obj.report({k: v for k, v in REF.items()})


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs, match", [
    ({"loss": "l1"}, "loss must be one of"),
    ({"huber_delta": 0.0}, "huber_delta must be > 0"),
    ({"missing_penalty": -1.0}, "missing_penalty must be >= 0"),
    ({"weights": {"zz": 1.0}}, "absent from references"),
])
def test_bad_configuration_is_refused(kwargs, match):
    with pytest.raises(ObjectiveError, match=match):
        CohesiveObjective(REF, **kwargs)


def test_empty_references_are_refused():
    with pytest.raises(ObjectiveError, match="references is empty"):
        CohesiveObjective({})


def test_duplicate_systems_in_candidate_results_are_refused():
    obj = CohesiveObjective({"a": 100.0})
    with pytest.raises(ObjectiveError, match="twice"):
        obj.report([FakeResult("a", 100.0), FakeResult("a", 101.0)])


def test_candidate_results_of_the_wrong_shape_say_so():
    obj = CohesiveObjective({"a": 100.0})
    with pytest.raises(ObjectiveError, match=r"\.system"):
        obj.report([object()])
