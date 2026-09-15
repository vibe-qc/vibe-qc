"""The cohesive-energy pipeline (M0a): assembly, staging, caching.

Driven by a fake engine with known energies, so the arithmetic and the
control flow are pinned exactly. Real-engine agreement is the M0 gate
and needs CRYSTAL23 / vibe-qc runs; it is not this file's job.

What matters here:

* the sign convention and the unit conversion are right,
* a skipped or failed stage is *visible* rather than silently defaulted,
* free-atom energies are shared across systems, which is what makes a
  campaign affordable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vibe_basis.cache import EnergyCache, atom_key, crystal_key
from vibe_basis.engine import (
    EnergyEngine,
    EngineCapabilityError,
    EngineEnergy,
    RelaxedStructure,
)
from vibe_basis.pipeline import (
    HARTREE_TO_KJ_PER_MOL,
    CohesivePipeline,
    CohesiveResult,
    PipelineError,
    formula_units,
    stoichiometry,
)

# ---------------------------------------------------------------------------
# Fixtures: a minimal Structure stand-in and a scriptable engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeAtom:
    Z: int


@dataclass(frozen=True)
class FakeStructure:
    name: str
    unit_cell: tuple[FakeAtom, ...]


#: Rocksalt-shaped conventional cell: 4 Mg + 4 O, i.e. 4 formula units.
MGO = FakeStructure(
    "MgO", tuple([FakeAtom(12)] * 4 + [FakeAtom(8)] * 4)
)
#: Same elements, different compound -- exercises the shared atom cache.
CAO = FakeStructure("CaO", tuple([FakeAtom(20)] * 4 + [FakeAtom(8)] * 4))


class FakeEngine(EnergyEngine):
    """Engine returning scripted energies and counting its calls."""

    name = "fake"

    def __init__(
        self,
        *,
        bulk: dict[str, float] | None = None,
        atoms: dict[int, float] | None = None,
        capabilities: frozenset[str] = frozenset(),
        zpe: float | None = None,
        relax_ok: bool = True,
        fail_bulk: str | None = None,
        fail_atom: int | None = None,
    ) -> None:
        self._bulk = bulk or {}
        self._atoms = atoms or {}
        self.capabilities = capabilities
        self._zpe = zpe
        self._relax_ok = relax_ok
        self._fail_bulk = fail_bulk
        self._fail_atom = fail_atom
        self.bulk_calls: list[str] = []
        self.atom_calls: list[int] = []

    def version(self):
        return "0.0-fake"

    def crystal_energy(self, basis_text, structure, method="rhf"):
        self.bulk_calls.append(structure.name)
        if self._fail_bulk:
            return EngineEnergy.failed(self.name, self._fail_bulk)
        return EngineEnergy(
            energy=self._bulk[structure.name], ok=True, engine=self.name
        )

    def atom_energy(self, basis_text, Z, method="rhf", *, host=None):
        self.atom_calls.append(Z)
        if self._fail_atom == Z:
            return EngineEnergy.failed(self.name, "non_converged")
        return EngineEnergy(energy=self._atoms[Z], ok=True, engine=self.name)

    def relax(self, basis_text, structure, method="rhf"):
        if not self._relax_ok:
            return RelaxedStructure(
                structure=structure,
                ok=False,
                engine=self.name,
                failure_mode="max_steps",
            )
        return RelaxedStructure(structure=structure, ok=True, engine=self.name)

    def zero_point_energy(self, basis_text, structure, method="rhf"):
        assert self._zpe is not None
        return EngineEnergy(energy=self._zpe, ok=True, engine=self.name)


# Energies chosen so the arithmetic is checkable by hand:
#   E_bulk(MgO cell, 4 f.u.) = -400.0  ->  -100.0 per f.u.
#   E(Mg) = -50.0, E(O) = -25.0        ->  sum   =  -75.0 per f.u.
#   E_coh = -75.0 - (-100.0)           =   25.0 Ha per f.u.
BULK = {"MgO": -400.0, "CaO": -800.0}
ATOMS = {12: -50.0, 8: -25.0, 20: -150.0}


def make_pipeline(**engine_kw):
    engine = FakeEngine(bulk=BULK, atoms=ATOMS, **engine_kw)
    pipe = CohesivePipeline(engine, method="pbe", relax=False, zero_point=False)
    return engine, pipe


# ---------------------------------------------------------------------------
# Stoichiometry
# ---------------------------------------------------------------------------


def test_formula_units_from_element_counts():
    assert formula_units(MGO.unit_cell) == 4                       # 4 Mg + 4 O
    assert formula_units([FakeAtom(20)] * 4 + [FakeAtom(9)] * 8) == 4   # CaF2
    assert formula_units([FakeAtom(13)] * 12 + [FakeAtom(8)] * 18) == 6  # Al2O3
    assert formula_units([FakeAtom(6)] * 2) == 2                   # diamond


def test_formula_units_rejects_an_empty_cell():
    with pytest.raises(PipelineError, match="empty unit cell"):
        formula_units([])


def test_stoichiometry_is_per_formula_unit():
    assert stoichiometry(MGO.unit_cell, 4) == {12: 1, 8: 1}
    assert stoichiometry([FakeAtom(20)] * 4 + [FakeAtom(9)] * 8, 4) == {20: 1, 9: 2}


def test_stoichiometry_rejects_a_non_integer_split():
    """A caller-supplied formula-unit count that does not divide the cell
    is a mistake worth failing on, not rounding through."""
    with pytest.raises(PipelineError, match="not an integer count"):
        stoichiometry(MGO.unit_cell, 3)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_cohesive_energy_sign_and_magnitude():
    """E_coh = SUM E_atom - E_bulk/f.u., positive for a bound solid."""
    _, pipe = make_pipeline()
    r = pipe.run("BASIS", MGO)

    assert r.ok
    assert r.formula_units == 4
    assert r.e_bulk == pytest.approx(-400.0)
    assert r.e_atoms_sum == pytest.approx(-75.0)      # Mg + O, per f.u.
    assert r.cohesive_hartree == pytest.approx(25.0)  # -75 - (-100)
    assert r.cohesive_kj_per_mol == pytest.approx(25.0 * HARTREE_TO_KJ_PER_MOL)
    assert r.atomic_energies == {8: -25.0, 12: -50.0}


def test_hartree_conversion_matches_vibeqc():
    """The kJ/mol constant must agree with vibe-qc's kcal/mol one.

    vibe-basis cannot import vibe-qc (tests/test_no_vibeqc_dependency.py),
    so the constant is duplicated -- and therefore pinned here, via the
    thermochemical calorie, against the value in
    ``vibeqc.atomization._HARTREE_TO_KCAL``.
    """
    hartree_to_kcal = 627.509474063
    assert HARTREE_TO_KJ_PER_MOL == pytest.approx(hartree_to_kcal * 4.184, rel=1e-12)


def test_zero_point_reduces_the_cohesive_energy():
    """A vibrating solid is less bound than a static one.

    Free atoms have no vibrations, so the correction is one-sided:
    E_coh(with ZPE) = E_coh(static) - ZPE_per_fu.
    """
    engine = FakeEngine(
        bulk=BULK, atoms=ATOMS, capabilities=frozenset({"zero_point"}), zpe=0.4
    )
    pipe = CohesivePipeline(engine, method="pbe", relax=False, zero_point=True)
    r = pipe.run("BASIS", MGO)

    assert r.ok
    assert r.zero_point_applied is True
    assert r.zero_point == pytest.approx(0.1)          # 0.4 Ha / 4 f.u.
    assert r.cohesive_hartree == pytest.approx(24.9)   # 25.0 - 0.1
    assert "with ZPE" in r.summary()


def test_static_lattice_result_says_so():
    _, pipe = make_pipeline()
    r = pipe.run("BASIS", MGO)
    assert r.zero_point_applied is False
    assert r.zero_point is None
    assert "static lattice" in r.summary()


def test_caller_can_override_the_formula_unit_count():
    """Needed for cells whose formula is not the reduced one."""
    _, pipe = make_pipeline()
    r = pipe.run("BASIS", MGO, n_formula_units=1)
    assert r.formula_units == 1
    assert r.e_atoms_sum == pytest.approx(4 * -50.0 + 4 * -25.0)
    assert r.cohesive_hartree == pytest.approx(-300.0 - (-400.0))


# ---------------------------------------------------------------------------
# Stages are visible
# ---------------------------------------------------------------------------


def _stage(result: CohesiveResult, name: str) -> object:
    return next(s for s in result.stages if s.name == name)


def test_a_missing_capability_is_recorded_as_skipped_not_defaulted():
    """The core honesty property.

    An engine with no relaxer and no phonons must not yield a result
    that looks relaxed and ZPE-corrected. Both stages are recorded as
    skipped, with the engine named in the reason.
    """
    engine = FakeEngine(bulk=BULK, atoms=ATOMS)  # declares no capabilities
    pipe = CohesivePipeline(engine, method="pbe", relax=True, zero_point=True)
    r = pipe.run("BASIS", MGO)

    assert r.ok
    assert _stage(r, "relax").status == "skipped"
    assert "no relaxer" in _stage(r, "relax").reason
    assert _stage(r, "zero_point").status == "skipped"
    assert "no phonons" in _stage(r, "zero_point").reason
    assert r.zero_point_applied is False


def test_disabled_stages_are_distinguishable_from_missing_ones():
    _, pipe = make_pipeline()
    r = pipe.run("BASIS", MGO)
    assert _stage(r, "relax").reason == "disabled by caller"
    assert _stage(r, "zero_point").reason == "disabled by caller"


def test_failed_relaxation_sinks_the_run():
    """An unconverged geometry must not be used for the energy."""
    engine = FakeEngine(
        bulk=BULK, atoms=ATOMS, capabilities=frozenset({"relax"}), relax_ok=False
    )
    pipe = CohesivePipeline(engine, method="pbe", relax=True, zero_point=False)
    r = pipe.run("BASIS", MGO)

    assert not r.ok
    assert r.failure_mode == "relax_max_steps"
    assert _stage(r, "relax").status == "failed"
    assert engine.bulk_calls == []  # never got as far as the energy


def test_failed_zero_point_leaves_a_usable_static_lattice_result():
    """Unlike a failed relaxation, a failed ZPE is survivable.

    A static-lattice cohesive energy is a well-defined quantity; the
    result just must not claim to be ZPE-corrected.
    """

    class _BadZpe(FakeEngine):
        def zero_point_energy(self, basis_text, structure, method="rhf"):
            return EngineEnergy.failed(self.name, "imaginary_modes")

    engine = _BadZpe(
        bulk=BULK, atoms=ATOMS, capabilities=frozenset({"zero_point"})
    )
    pipe = CohesivePipeline(engine, method="pbe", relax=False, zero_point=True)
    r = pipe.run("BASIS", MGO)

    assert r.ok
    assert r.zero_point_applied is False
    assert _stage(r, "zero_point").status == "failed"
    assert _stage(r, "zero_point").reason == "imaginary_modes"
    assert r.cohesive_hartree == pytest.approx(25.0)


def test_failed_bulk_and_failed_atom_both_sink_the_run():
    _, pipe = make_pipeline(fail_bulk="non_converged")
    r = pipe.run("BASIS", MGO)
    assert not r.ok and r.failure_mode == "bulk_non_converged"

    _, pipe = make_pipeline(fail_atom=8)
    r = pipe.run("BASIS", MGO)
    assert not r.ok and r.failure_mode == "atom_Z8_non_converged"


def test_result_carries_provenance():
    _, pipe = make_pipeline()
    r = pipe.run("BASIS", MGO)
    assert r.engine == "fake"
    assert r.engine_version == "0.0-fake"
    assert r.method == "pbe"
    assert r.basis_digest and len(r.basis_digest) == 16


# ---------------------------------------------------------------------------
# Caching -- the thing that makes a campaign affordable
# ---------------------------------------------------------------------------


def test_atoms_are_shared_across_systems():
    """O is computed once for MgO and reused for CaO.

    This is the whole point of the shared cache: in the PW-limit work a
    single free atom cost minutes while the solid was quick, so
    recomputing O per compound would dominate a campaign.
    """
    engine = FakeEngine(bulk=BULK, atoms=ATOMS)
    cache = EnergyCache()
    pipe = CohesivePipeline(
        engine, method="pbe", relax=False, zero_point=False, cache=cache
    )

    pipe.run("BASIS", MGO)
    pipe.run("BASIS", CAO)

    assert engine.atom_calls == [8, 12, 20]  # O once, not twice
    assert engine.bulk_calls == ["MgO", "CaO"]
    assert cache.hits == 1


def test_a_repeated_system_hits_the_cache_entirely():
    engine = FakeEngine(bulk=BULK, atoms=ATOMS)
    cache = EnergyCache()
    pipe = CohesivePipeline(
        engine, method="pbe", relax=False, zero_point=False, cache=cache
    )
    first = pipe.run("BASIS", MGO)
    n_calls = len(engine.bulk_calls) + len(engine.atom_calls)
    second = pipe.run("BASIS", MGO)

    assert len(engine.bulk_calls) + len(engine.atom_calls) == n_calls
    assert second.cohesive_hartree == pytest.approx(first.cohesive_hartree)
    assert all(s.status in ("cached", "skipped") for s in second.stages)


def test_a_changed_basis_misses_the_cache():
    """Different basis, different energies -- a hit here would be a bug."""
    engine = FakeEngine(bulk=BULK, atoms=ATOMS)
    cache = EnergyCache()
    pipe = CohesivePipeline(
        engine, method="pbe", relax=False, zero_point=False, cache=cache
    )
    pipe.run("BASIS-A", MGO)
    pipe.run("BASIS-B", MGO)
    assert engine.bulk_calls == ["MgO", "MgO"]
    assert engine.atom_calls == [8, 12, 8, 12]


def test_cache_keys_separate_engine_system_method_and_basis():
    a = crystal_key("crystal23", "MgO", "B", "pbe")
    assert a != crystal_key("vibeqc", "MgO", "B", "pbe")
    assert a != crystal_key("crystal23", "CaO", "B", "pbe")
    assert a != crystal_key("crystal23", "MgO", "B2", "pbe")
    assert a != crystal_key("crystal23", "MgO", "B", "r2scan")

    # Atom keys carry no system, which is what makes them shareable.
    assert atom_key("crystal23", 8, "B", "pbe") == atom_key("crystal23", 8, "B", "pbe")
    assert atom_key("crystal23", 8, "B", "pbe") != atom_key("crystal23", 12, "B", "pbe")


def test_journal_survives_a_restart(tmp_path):
    """A killed campaign resumes instead of re-spending the queue."""
    journal = tmp_path / "runs" / "cache.jsonl"
    engine = FakeEngine(bulk=BULK, atoms=ATOMS)
    pipe = CohesivePipeline(
        engine,
        method="pbe",
        relax=False,
        zero_point=False,
        cache=EnergyCache(journal),
    )
    first = pipe.run("BASIS", MGO)
    assert journal.is_file()

    # A fresh process: new engine, new cache, same journal.
    engine2 = FakeEngine(bulk=BULK, atoms=ATOMS)
    pipe2 = CohesivePipeline(
        engine2,
        method="pbe",
        relax=False,
        zero_point=False,
        cache=EnergyCache(journal),
    )
    second = pipe2.run("BASIS", MGO)

    assert engine2.bulk_calls == [] and engine2.atom_calls == []
    assert second.cohesive_hartree == pytest.approx(first.cohesive_hartree)


def test_journal_tolerates_a_truncated_final_line(tmp_path):
    """kill -9 mid-write must cost one entry, not the whole journal."""
    journal = tmp_path / "cache.jsonl"
    cache = EnergyCache(journal)
    cache.put("k1", EngineEnergy(energy=-1.0, ok=True, engine="fake"))
    cache.put("k2", EngineEnergy(energy=-2.0, ok=True, engine="fake"))
    with journal.open("a", encoding="utf-8") as fh:
        fh.write('{"key": "k3", "energy": -3.0, "ok"')  # torn write

    reloaded = EnergyCache(journal)
    assert len(reloaded) == 2
    assert reloaded.get("k1").energy == pytest.approx(-1.0)
    assert reloaded.get("k3") is None


def test_failures_are_cached_by_default_but_can_be_opted_out():
    """Re-submitting a job known to diverge wastes the queue time the
    cache exists to save -- but a flaky transport is different."""
    cache = EnergyCache()
    cache.put("k", EngineEnergy.failed("fake", "non_converged"))
    assert "k" in cache

    transient = EnergyCache(cache_failures=False)
    transient.put("k", EngineEnergy.failed("fake", "transport_error"))
    assert "k" not in transient


# ---------------------------------------------------------------------------
# Engine capability plumbing
# ---------------------------------------------------------------------------


def test_supports_rejects_an_unknown_capability_name():
    engine = FakeEngine()
    with pytest.raises(ValueError, match="unknown capability"):
        engine.supports("telepathy")


def test_calling_an_undeclared_capability_raises():
    """Default implementations must refuse rather than fake a result."""
    engine = FakeEngine()
    bare = EnergyEngine.relax
    with pytest.raises(EngineCapabilityError, match="does not support 'relax'"):
        bare(engine, "BASIS", MGO)
    with pytest.raises(EngineCapabilityError, match="does not support 'zero_point'"):
        EnergyEngine.zero_point_energy(engine, "BASIS", MGO)


# ---------------------------------------------------------------------------
# Counterpoise: matching the published protocol
# ---------------------------------------------------------------------------


class _CounterpoiseEngine(FakeEngine):
    """Returns a different atom energy when a ghost host is supplied.

    The offset stands in for the BSSE: a counterpoise atom sits lower
    than a bare one because it borrows the crystal's basis functions.
    """

    BSSE = 0.01  # Ha per atom, sign as physics has it: lower with ghosts

    def __init__(self, **kw):
        kw.setdefault("capabilities", frozenset({"counterpoise"}))
        super().__init__(**kw)

    def atom_energy(self, basis_text, Z, method="rhf", *, host=None):
        self.atom_calls.append(Z)
        base = self._atoms[Z]
        if host is not None:
            self.hosts = getattr(self, "hosts", [])
            self.hosts.append(host.name)
            return EngineEnergy(energy=base - self.BSSE, ok=True, engine=self.name)
        return EngineEnergy(energy=base, ok=True, engine=self.name)


def _pipe(engine, **kw):
    kw.setdefault("relax", False)
    kw.setdefault("zero_point", False)
    return CohesivePipeline(engine, method="pbe", **kw)


def test_counterpoise_is_off_by_default_and_says_so():
    engine = _CounterpoiseEngine(bulk=BULK, atoms=ATOMS)
    r = _pipe(engine).run("BASIS", MGO)
    assert r.counterpoise_applied is False
    assert _stage(r, "counterpoise").reason == "disabled by caller"
    assert r.cohesive_hartree == pytest.approx(25.0)


def test_counterpoise_uses_the_host_and_changes_the_number():
    engine = _CounterpoiseEngine(bulk=BULK, atoms=ATOMS)
    r = _pipe(engine, counterpoise=True).run("BASIS", MGO)

    assert r.counterpoise_applied is True
    assert _stage(r, "counterpoise").status == "ok"
    assert engine.hosts == ["MgO", "MgO"]  # the host reached the engine
    # Two atoms per f.u., each lowered by BSSE, so E_coh drops by 2 * BSSE.
    assert r.cohesive_hartree == pytest.approx(25.0 - 2 * _CounterpoiseEngine.BSSE)


def test_an_engine_without_counterpoise_records_a_skipped_stage():
    """Same honesty rule as ZPE: never silently fall back to bare atoms
    when the caller asked for the corrected protocol."""
    engine = FakeEngine(bulk=BULK, atoms=ATOMS)  # declares nothing
    r = _pipe(engine, counterpoise=True).run("BASIS", MGO)
    assert r.ok
    assert r.counterpoise_applied is False
    assert "cannot do counterpoise" in _stage(r, "counterpoise").reason


def test_counterpoise_atoms_are_not_shared_across_systems():
    """The whole point of the bare-atom cache is sharing; a counterpoise
    atom must NOT be shared, because it is computed in its own host's
    ghost basis and MgO's oxygen is a different calculation from CaO's."""
    engine = _CounterpoiseEngine(bulk=BULK, atoms=ATOMS)
    cache = EnergyCache()
    pipe = _pipe(engine, counterpoise=True, cache=cache)
    pipe.run("BASIS", MGO)
    pipe.run("BASIS", CAO)
    # O is recomputed for CaO rather than reused from MgO.
    assert engine.atom_calls == [8, 12, 8, 20]
    assert cache.hits == 0


def test_bare_atoms_still_share_when_counterpoise_is_off():
    """The sharing optimisation must survive the new host parameter."""
    engine = _CounterpoiseEngine(bulk=BULK, atoms=ATOMS)
    cache = EnergyCache()
    pipe = _pipe(engine, cache=cache)
    pipe.run("BASIS", MGO)
    pipe.run("BASIS", CAO)
    assert engine.atom_calls == [8, 12, 20]  # O computed once
    assert cache.hits == 1


def test_host_separates_the_cache_key():
    a = atom_key("crystal23", 8, "B", "pbe")
    assert a != atom_key("crystal23", 8, "B", "pbe", host="MgO")
    assert atom_key("crystal23", 8, "B", "pbe", host="MgO") != atom_key(
        "crystal23", 8, "B", "pbe", host="CaO"
    )
