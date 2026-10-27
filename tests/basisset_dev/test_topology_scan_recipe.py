"""The M3 topology-scan recipe (#225) against a model engine.

The engine here is synthetic: its energy depends only on the Mg d exponents
of the basis text it is handed, with known optima. What is pinned is the
recipe's half of the gate: pob records go into a topology and come back out
as the same CRYSTAL text, every candidate is relaxed before it is scored, the
relaxed exponents seed the next round, the scan stops on its plateau rule,
failed points reach the optimiser as finite penalties, a restarted scan reads
every point back from its journal, and the record carries the per-candidate
provenance. The MgO and SiC scans themselves run on CRYSTAL23 through vq.
"""

from __future__ import annotations

import importlib.util
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("vibe_basis")

from scipy.optimize import minimize  # noqa: E402
from vibe_basis.engine import EngineEnergy  # noqa: E402
from vibe_basis.io.structures import STRUCTURES  # noqa: E402
from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL  # noqa: E402

from vibeqc.basis_crystal import emit_crystal, parse_crystal_inline_element  # noqa: E402
from vibeqc.basis_optimization.recipes.topology_scan import (  # noqa: E402
    DEFAULT_LOWER_BOUND,
    TransportUnavailable,
    atoms_from_topology,
    crystal23_engine_factory,
    load_pob_atoms,
    run_topology_scan,
    topology_from_atoms,
)

MGO = STRUCTURES["MgO"]
E0_HA = -275.0

# Each Mg d function, most diffuse first, gains WEIGHTS[k] kJ/mol at
# TARGETS[k]. The second function gains more than the 1 kJ/mol plateau
# threshold, the third less: a scan must add one function and stop.
TARGETS = (0.40, 1.20, 3.60)
WEIGHTS = (10.0, 2.0, 0.3)


def _model_energy_ha(d_exponents):
    gain = 0.0
    for k, a in enumerate(sorted(d_exponents)):
        if k < len(TARGETS):
            gain += WEIGHTS[k] * math.exp(-2.0 * math.log(a / TARGETS[k]) ** 2)
    return E0_HA - gain / HARTREE_TO_KJ_PER_MOL


def _d_exponents(basis_text):
    mg = parse_crystal_inline_element(basis_text, 12, source="<model>")
    return [a for s in mg.shells if s.shell_type == "D" for a in s.exponents]


class ModelEngine:
    """Answers from the model; optionally fails some points."""

    name = "model"

    def __init__(self, log, fail=None):
        self.log = log
        self.fail = fail

    def crystal_energy(self, basis_text, structure, method="rhf"):
        d = _d_exponents(basis_text)
        self.log.append(d)
        mode = self.fail(d) if self.fail else None
        if mode:
            return EngineEnergy.failed(self.name, mode)
        return EngineEnergy(energy=_model_energy_ha(d), ok=True, engine=self.name)


def _factory(log, fail=None):
    return lambda point_dir: ModelEngine(log, fail)


def _powell(objective, x0, *, bounds, tol, ftol_abs, xtol_abs, max_eval, initial_step):
    """A derivative-free stand-in for BOBYQA with optimize_nlopt's signature."""
    res = minimize(
        objective, x0, method="Powell", bounds=bounds,
        options={"xtol": xtol_abs, "ftol": 1e-12, "maxfev": max_eval},
    )
    return SimpleNamespace(message=str(res.message), n_evaluations=int(res.nfev))


def _scan(tmp_path, log, *, fail=None, **kw):
    kw.setdefault("optimizer", _powell)
    kw.setdefault("max_eval", 200)
    return run_topology_scan(
        MGO, "Mg", "D", _factory(log, fail), tmp_path, parallel=2, **kw
    )


def test_pob_records_round_trip_through_a_topology():
    atoms = load_pob_atoms("pob-TZVP-rev2", ["Mg", "O"])
    topology = topology_from_atoms(atoms)
    assert topology.exponents("Mg", "D") == (0.29355357,)
    back = atoms_from_topology(topology, atoms)
    assert emit_crystal(list(back.values())) == emit_crystal(list(atoms.values()))


def test_scan_relaxes_every_candidate_and_stops_on_its_plateau(tmp_path):
    log = []
    report = _scan(tmp_path, log)
    result = report.result

    # (a) The plateau rule stopped it, not the round cap.
    assert result.stopped_because.startswith("2 consecutive rounds")
    # The baseline was relaxed from the shipped 0.2936 to the model optimum.
    base = result.candidates[0]
    assert base["name"] == "round0:baseline"
    assert base["detail"]["seed_exponents"] == [0.29355357]
    assert base["detail"]["exponents"][0] == pytest.approx(0.40, rel=1e-2)
    assert base["detail"]["start_energy_ha"] == pytest.approx(
        _model_energy_ha([0.29355357]), abs=1e-12
    )
    # One added function earns its place, the next does not.
    assert sorted(result.topology.exponents("Mg", "D")) == pytest.approx(
        [0.40, 1.20], rel=1e-2
    )
    assert [h["improved"] for h in result.history][:1] == [True]
    # Round 2 was seeded from the relaxed round-1 winner, not its seed:
    # add-high places a1 * r, r = 1.2 / 0.4 the relaxed pair's ratio.
    seeds = [c["detail"]["seed_exponents"] for c in result.candidates
             if c["name"].startswith("round2:") and c["move"] == "add-high"]
    assert seeds and sorted(seeds[0]) == pytest.approx([0.40, 1.20, 3.60], rel=2e-2)

    # (c) The shipped topology is within the threshold of the front. Exact
    # membership is a tolerance artefact: a later candidate of equal cost
    # re-relaxes the same single d function and may land a hair lower.
    assert report.shipped_gap_kj < 0.05
    assert report.shipped_within_threshold
    # (b) The front is ranked, and every candidate carries its points.
    ranked = report.ranked_front()
    assert [c["score"].objective for c in ranked] == sorted(
        c["score"].objective for c in ranked
    )
    assert all(c["detail"]["n_points"] >= 1 for c in result.candidates)

    record = json.loads((tmp_path / "scan_record.json").read_text())
    assert record["gate_c"]["shipped_within_threshold"] is True
    assert record["settings"]["lower_bound"] == DEFAULT_LOWER_BOUND
    assert len(record["candidates"]) == len(result.candidates)
    assert record["candidates"][0]["detail"]["points"][0]["exponents"] == [0.29355357]


def test_a_seed_below_the_bound_starts_on_it(tmp_path):
    # The baseline relaxes to 0.40, so add-low seeds 0.40 / 2.5 = 0.16.
    report = _scan(tmp_path, [], max_rounds=1, lower_bound=0.2)
    low = next(c for c in report.result.candidates if c["move"] == "add-low")
    assert min(low["detail"]["seed_exponents"]) == pytest.approx(0.16, rel=2e-2)
    assert low["detail"]["seed_moved_to_bounds"]
    assert min(low["detail"]["start_exponents"]) == 0.2
    assert min(low["detail"]["exponents"]) >= 0.2
    assert report.settings["lower_bound"] == 0.2


def test_a_restarted_scan_reads_every_point_from_its_journal(tmp_path):
    first = _scan(tmp_path, [])
    log = []
    again = _scan(tmp_path, log)
    assert log == []
    assert again.cache_stats["misses"] == 0
    assert again.result.score == first.result.score
    assert again.result.front == first.result.front


def test_a_failed_scf_is_a_penalty_and_a_lost_transport_stops_the_scan(tmp_path):
    def collapse(d):
        # Two d functions closer than a factor 1.5: the model's "SCF fails".
        d = sorted(d)
        return "non_converged" if any(b / a < 1.5 for a, b in zip(d, d[1:])) else None

    report = _scan(tmp_path, [], fail=collapse, max_rounds=1)
    for c in report.result.candidates:
        assert math.isfinite(c["score"].objective)
    failed = sum(c["detail"]["n_failed"] for c in report.result.candidates)
    assert failed > 0
    two = [c for c in report.result.candidates if c["move"].startswith("add-")]
    for c in two:
        a, b = sorted(c["detail"]["exponents"])
        assert b / a >= 1.5

    # A transport failure says nothing about the basis: it is retried, and a
    # point that never gets through stops the scan without being journalled.
    calls = []
    work = tmp_path / "transport"
    with pytest.raises(TransportUnavailable, match="3 attempts ended in transport_error"):
        run_topology_scan(
            MGO, "Mg", "D", _factory(calls, lambda d: "transport_error"), work,
            max_rounds=0, optimizer=_powell, transient_retries=2,
        )
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    retried = []
    run_topology_scan(
        MGO, "Mg", "D", _factory(retried), work, max_rounds=0, optimizer=_powell,
    )
    assert retried and retried[0] == calls[0]

    flaky = []

    def first_attempt_lost(d):
        return "transport_error" if len(flaky) == 1 else None

    once = run_topology_scan(
        MGO, "Mg", "D", _factory(flaky, first_attempt_lost), tmp_path / "flaky",
        max_rounds=0, optimizer=_powell,
    )
    assert flaky[0] == flaky[1]
    assert once.result.candidates[0]["detail"]["n_failed"] == 0


def test_crystal23_points_read_their_deck_on_stdin(tmp_path):
    factory = crystal23_engine_factory(transport=object(), crystal_binary="crystal23")
    engine = factory(tmp_path / "point0001")
    assert engine.invocation == "stdin"
    assert engine.crystal_wrapper == "crystal23"
    assert engine.strict_version
    assert engine.workdir_prefix == str(tmp_path / "point0001" / "calc")
    mpi = crystal23_engine_factory(
        transport=object(), crystal_binary="Pcrystal23", invocation="mpi", cpus=4
    )(tmp_path)
    assert (mpi.invocation, mpi.cpus) == ("mpi", 4)


@pytest.mark.skipif(importlib.util.find_spec("nlopt") is None, reason="nlopt not installed")
def test_bobyqa_relaxes_the_baseline_to_the_model_optimum(tmp_path):
    report = run_topology_scan(
        MGO, "Mg", "D", _factory([]), tmp_path, max_rounds=0, parallel=1,
    )
    base = report.result.candidates[0]
    assert base["detail"]["exponents"][0] == pytest.approx(0.40, rel=1e-2)
    assert base["detail"]["n_points"] <= 20
    assert np.isfinite(base["score"].instability)
