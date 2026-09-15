"""Basis topology and the d/f scan: shape guards, the moves, the search.

Every evaluator here is synthetic. M3's own gate -- scanning the d shell of Mg
in MgO and of Si in SiC from pob-TZVP-REV2, and rediscovering the shipped pob
topology when the objective is total energy -- needs an engine and a full L2
relaxation per candidate, and has NOT been run; see HANDOVER_BASISOPT.md M3.
What is pinned here is that the moves are the ones the roadmap specifies, that
the search obeys its own stopping rule, and that a topology is never mutated
under a caller. The rediscovery property is exercised in the one form that
needs no engine: an evaluator that prefers the starting topology must get it
back unchanged.
"""

from __future__ import annotations

import math

import pytest

from vibe_basis.topology import (
    DEFAULT_RATIO,
    BasisTopology,
    ScanScore,
    Shell,
    TopologyError,
    even_tempered_ratio,
    greedy_scan,
    pareto_front,
    propose_moves,
)


def _mg() -> BasisTopology:
    """Two uncontracted d functions plus a contracted s, ratio 3.0 on d."""
    return BasisTopology({
        "Mg": (
            Shell("S", (10.0, 2.0), (0.3, 0.7)),
            Shell("D", (0.9,), (1.0,)),
            Shell("D", (0.3,), (1.0,)),
        )
    })


# ---------------------------------------------------------------------------
# Shell shape
# ---------------------------------------------------------------------------


def test_shell_mirrors_the_crystalshell_fields():
    s = Shell("D", (0.9,), (1.0,))
    assert (s.shell_type, s.n_primitives, s.is_uncontracted) == ("D", 1, True)
    assert s.occupancy == 0.0 and s.scale_factor == 1.0


@pytest.mark.parametrize("kwargs, match", [
    (dict(shell_type="X", exponents=(1.0,), coefficients=(1.0,)), "shell_type must be"),
    (dict(shell_type="D", exponents=(), coefficients=()), "no exponents"),
    (dict(shell_type="D", exponents=(-1.0,), coefficients=(1.0,)), "non-positive"),
    (dict(shell_type="D", exponents=(float("inf"),), coefficients=(1.0,)), "non-finite"),
    (dict(shell_type="D", exponents=(1.0, 2.0), coefficients=(1.0,)), "coefficients"),
    (dict(shell_type="SP", exponents=(1.0,), coefficients=(1.0,)), "one p coefficient"),
    (dict(shell_type="D", exponents=(1.0,), coefficients=(1.0,),
          coefficients_p=(1.0,)), "only\nan SP|only an SP"),
])
def test_malformed_shells_are_refused(kwargs, match):
    with pytest.raises(TopologyError, match=match):
        Shell(**kwargs)


def test_an_sp_shell_keeps_both_coefficient_sides():
    s = Shell("SP", (1.0, 0.5), (0.4, 0.6), coefficients_p=(0.2, 0.8))
    assert s.coefficients_p == (0.2, 0.8)


def test_a_topology_needs_atoms_and_each_atom_needs_shells():
    with pytest.raises(TopologyError, match="no atoms"):
        BasisTopology({})
    with pytest.raises(TopologyError, match="no shells"):
        BasisTopology({"Mg": ()})


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_exponents_are_gathered_across_shells_and_sorted_descending():
    """a1 > ... > an is the roadmap's convention: a1 tightest, an most diffuse."""
    t = _mg()
    assert t.exponents("Mg", "D") == (0.9, 0.3)
    assert t.exponents("Mg", "S") == (10.0, 2.0)
    assert t.exponents("Mg", "F") == ()


def test_shell_indices_and_primitive_count():
    t = _mg()
    assert t.shell_indices("Mg", "D") == (1, 2)
    assert t.n_primitives == 4


def test_an_unknown_element_says_what_is_available():
    with pytest.raises(TopologyError, match="unknown element 'Si'.*have"):
        _mg().exponents("Si", "D")


# ---------------------------------------------------------------------------
# The even-tempered ratio
# ---------------------------------------------------------------------------


def test_ratio_is_the_geometric_mean_of_adjacent_ratios():
    # 81, 27, 3 -> adjacent ratios 3 and 9 -> geometric mean sqrt(27)
    assert even_tempered_ratio([81.0, 27.0, 3.0]) == pytest.approx(math.sqrt(27.0))


def test_ratio_does_not_depend_on_input_order():
    assert even_tempered_ratio([3.0, 81.0, 27.0]) == pytest.approx(
        even_tempered_ratio([81.0, 27.0, 3.0])
    )


def test_a_single_exponent_has_no_adjacent_pair_so_falls_back():
    assert even_tempered_ratio([0.7]) == DEFAULT_RATIO == 2.5


@pytest.mark.parametrize("bad", [[], [0.0, 1.0], [-1.0]])
def test_ratio_refuses_degenerate_input(bad):
    with pytest.raises(TopologyError):
        even_tempered_ratio(bad)


# ---------------------------------------------------------------------------
# Mutators, and that they do not mutate
# ---------------------------------------------------------------------------


def test_add_shell_seeds_one_uncontracted_function_and_leaves_the_parent_alone():
    t = _mg()
    out = t.add_shell("Mg", "D", 2.7)
    assert out.exponents("Mg", "D") == (2.7, 0.9, 0.3)
    assert out.atoms["Mg"][-1].is_uncontracted
    assert t.exponents("Mg", "D") == (0.9, 0.3)      # parent untouched
    assert t.n_primitives == 4


def test_drop_shell_removes_one_and_leaves_the_parent_alone():
    t = _mg()
    out = t.drop_shell("Mg", 1)
    assert out.exponents("Mg", "D") == (0.3,)
    assert t.exponents("Mg", "D") == (0.9, 0.3)


def test_dropping_the_only_shell_is_refused():
    t = BasisTopology({"H": (Shell("S", (1.0,), (1.0,)),)})
    with pytest.raises(TopologyError, match="only shell"):
        t.drop_shell("H", 0)


def test_split_contraction_halves_a_shell_keeping_its_coefficients():
    t = BasisTopology({"Mg": (Shell("S", (10.0, 2.0, 0.5), (0.2, 0.3, 0.5)),)})
    out = t.split_contraction("Mg", 0, 2)
    head, tail = out.atoms["Mg"]
    assert (head.exponents, head.coefficients) == ((10.0, 2.0), (0.2, 0.3))
    assert (tail.exponents, tail.coefficients) == ((0.5,), (0.5,))
    assert out.n_primitives == t.n_primitives          # decontraction is free


def test_split_contraction_carries_the_p_side_of_an_sp_shell():
    t = BasisTopology({"Mg": (
        Shell("SP", (10.0, 2.0), (0.3, 0.7), coefficients_p=(0.1, 0.9)),
    )})
    head, tail = t.split_contraction("Mg", 0, 1).atoms["Mg"]
    assert head.coefficients_p == (0.1,)
    assert tail.coefficients_p == (0.9,)


@pytest.mark.parametrize("k", [0, 3])
def test_split_point_must_be_strictly_inside_the_shell(k):
    t = BasisTopology({"Mg": (Shell("S", (10.0, 2.0, 0.5), (0.2, 0.3, 0.5)),)})
    with pytest.raises(TopologyError, match="strictly inside"):
        t.split_contraction("Mg", 0, k)


# ---------------------------------------------------------------------------
# The moves
# ---------------------------------------------------------------------------


def test_propose_moves_is_the_roadmap_table():
    t = _mg()
    moves = propose_moves(t, "Mg", "D")
    assert [m.name for m in moves] == [
        "baseline", "add-high", "add-low", "leave-one-out:1", "leave-one-out:2",
    ]
    by = {m.name: m for m in moves}
    # r = 3.0 on (0.9, 0.3): tighter is a1*r, more diffuse is an/r.
    assert by["add-high"].topology.exponents("Mg", "D")[0] == pytest.approx(2.7)
    assert by["add-low"].topology.exponents("Mg", "D")[-1] == pytest.approx(0.1)
    assert by["baseline"].topology is t


def test_every_move_records_where_its_seed_came_from():
    by = {m.name: m for m in propose_moves(_mg(), "Mg", "D")}
    assert "0.9" in by["add-high"].provenance and "3" in by["add-high"].provenance
    assert "dropped the D shell at index 1" in by["leave-one-out:1"].provenance


def test_an_angular_momentum_with_no_shells_offers_only_leave_one_out_of_others():
    """No f functions to seed from, so no add-high / add-low for f."""
    names = [m.name for m in propose_moves(_mg(), "Mg", "F")]
    assert names == ["baseline"]


def test_leave_one_out_is_not_offered_for_a_lone_shell():
    t = BasisTopology({"H": (Shell("S", (1.0,), (1.0,)),)})
    names = [m.name for m in propose_moves(t, "H", "S")]
    assert names == ["baseline", "add-high", "add-low"]


# ---------------------------------------------------------------------------
# Pareto ranking
# ---------------------------------------------------------------------------


def test_pareto_front_keeps_only_non_dominated_candidates():
    scored = {
        "cheap_bad":  ScanScore(objective=10.0, cost=1.0),
        "dear_good":  ScanScore(objective=1.0, cost=10.0),
        "dominated":  ScanScore(objective=11.0, cost=11.0),
        "balanced":   ScanScore(objective=5.0, cost=5.0),
    }
    assert set(pareto_front(scored)) == {"cheap_bad", "dear_good", "balanced"}


def test_instability_is_a_third_axis_not_a_tiebreak():
    """A candidate that is worse on objective can survive on conditioning."""
    scored = {
        "accurate_but_ill": ScanScore(1.0, 5.0, instability=1e9),
        "worse_but_stable": ScanScore(2.0, 5.0, instability=1e2),
    }
    assert set(pareto_front(scored)) == {"accurate_but_ill", "worse_but_stable"}


def test_exact_duplicates_both_survive():
    """Equal on every axis is not domination, so neither wins on dict order."""
    scored = {"a": ScanScore(1.0, 1.0), "b": ScanScore(1.0, 1.0)}
    assert set(pareto_front(scored)) == {"a", "b"}


def test_pareto_front_of_nothing_is_nothing():
    assert pareto_front({}) == ()


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------


def _score_by_nprim(target: int):
    """Objective minimised at ``target`` primitives; cost is the count."""
    def evaluate(t: BasisTopology) -> ScanScore:
        return ScanScore(objective=abs(t.n_primitives - target) * 10.0,
                         cost=float(t.n_primitives))
    return evaluate


def test_the_scan_climbs_toward_a_better_topology():
    # Start at 4 primitives, optimum at 6: add-high/add-low each gain 10.
    res = greedy_scan(_mg(), "Mg", "D", _score_by_nprim(6), threshold=1.0)
    assert res.topology.n_primitives == 6
    assert res.score.objective == pytest.approx(0.0)
    assert any(h["improved"] for h in res.history)


def test_a_scan_that_cannot_improve_returns_the_input_unchanged():
    """The rediscovery property, in the form that needs no engine: when the
    starting topology is already best, the scan must hand it back."""
    start = _mg()
    res = greedy_scan(start, "Mg", "D", _score_by_nprim(4), threshold=1.0)
    assert res.topology.atoms == start.atoms
    assert res.score.objective == pytest.approx(0.0)
    assert "threshold" in res.stopped_because


def test_the_plateau_rule_stops_after_patience_non_improving_rounds():
    flat = lambda t: ScanScore(objective=1.0, cost=float(t.n_primitives))
    res = greedy_scan(_mg(), "Mg", "D", flat, patience=2, max_rounds=10)
    assert res.rounds == 2
    assert "2 consecutive rounds" in res.stopped_because


def test_a_gain_at_exactly_the_threshold_does_not_count_as_improvement():
    """'more than a stated threshold' is strict, so a scan cannot creep
    forward on gains it declared too small to matter."""
    def evaluate(t: BasisTopology) -> ScanScore:
        # Exactly 1.0 better for any 5-primitive candidate.
        return ScanScore(objective=0.0 if t.n_primitives == 4 else -1.0,
                         cost=float(t.n_primitives))
    res = greedy_scan(_mg(), "Mg", "D", evaluate, threshold=1.0, patience=1)
    assert res.topology.n_primitives == 4          # refused the tie
    assert not any(h["improved"] for h in res.history)


def test_max_rounds_bounds_the_search():
    res = greedy_scan(_mg(), "Mg", "D", _score_by_nprim(99), max_rounds=3)
    assert res.rounds == 3
    assert res.stopped_because == "max_rounds reached"


def test_max_rounds_zero_evaluates_only_the_baseline():
    res = greedy_scan(_mg(), "Mg", "D", _score_by_nprim(6), max_rounds=0)
    assert res.rounds == 0
    assert res.topology.atoms == _mg().atoms
    assert res.history == ()


def test_history_records_each_round_with_its_provenance():
    res = greedy_scan(_mg(), "Mg", "D", _score_by_nprim(6), threshold=1.0)
    first = res.history[0]
    assert set(first) >= {
        "round", "best_move", "best_objective", "incumbent_objective",
        "gain", "improved", "provenance", "n_candidates",
    }
    assert first["round"] == 1
    assert first["provenance"]


def test_the_front_spans_every_round_not_just_the_last():
    """A candidate beaten on the objective may still be the one to ship on
    cost, so the front is taken over everything evaluated."""
    res = greedy_scan(_mg(), "Mg", "D", _score_by_nprim(6), threshold=1.0)
    assert any(name.startswith("round0") for name in res.front)
    assert len(res.front) >= 2


def test_a_wider_beam_evaluates_more_candidates_per_round():
    counts = {}
    for width in (1, 3):
        seen = []
        def evaluate(t: BasisTopology, seen=seen) -> ScanScore:
            seen.append(t.n_primitives)
            return ScanScore(objective=abs(t.n_primitives - 99) * 10.0,
                             cost=float(t.n_primitives))
        greedy_scan(_mg(), "Mg", "D", evaluate, max_rounds=2, beam_width=width)
        counts[width] = len(seen)
    assert counts[3] > counts[1]


@pytest.mark.parametrize("kwargs", [
    {"threshold": -1.0}, {"patience": 0}, {"beam_width": 0}, {"max_rounds": -1},
])
def test_bad_search_parameters_are_refused(kwargs):
    with pytest.raises(TopologyError):
        greedy_scan(_mg(), "Mg", "D", _score_by_nprim(6), **kwargs)


# ---------------------------------------------------------------------------
# The Tier-2 bridge
# ---------------------------------------------------------------------------


def test_to_atom_dict_uses_the_crystalshell_field_names():
    """So the Tier-2 adapter is a transcription, not a translation."""
    d = _mg().to_atom_dict()
    assert set(d) == {"Mg"}
    first = d["Mg"][0]
    assert set(first) == {
        "shell_type", "occupancy", "scale_factor",
        "exponents", "coefficients", "coefficients_p",
    }
    assert first["exponents"] == [10.0, 2.0]


def test_as_parametrisation_hands_plain_data_to_an_injected_builder():
    """The builder is injected because BasisParametrisation is Tier 2 and
    this module must not import vibeqc."""
    captured = {}

    def builder(atoms, **kw):
        captured["atoms"] = atoms
        captured["kw"] = kw
        return "built"

    assert _mg().as_parametrisation(builder, free=["x"]) == "built"
    assert set(captured["atoms"]) == {"Mg"}
    assert captured["kw"] == {"free": ["x"]}
