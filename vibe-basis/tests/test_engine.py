"""The :class:`~vibe_basis.engine.EnergyEngine` abstraction.

Covers the two properties the whole engine-swap design rests on:

* an engine reports **provenance** with every number, and
* the float view and the rich view can never disagree, because one is
  derived from the other.

Plus the CRYSTAL23 version probe, which is what makes "this energy came
from CRYSTAL23" a checked claim rather than an assumption.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from vibe_basis.backends.crystal import (
    DEFAULT_CRYSTAL_VERSION,
    parse_output,
    probe_version,
)
from vibe_basis.engine import EnergyEngine, EngineEnergy
from vibe_basis.engines import Crystal23Engine, GpawEngine

# A CRYSTAL23 banner as the binary actually prints it: boxed, centred,
# variable padding either side.
CRYSTAL23_BANNER = (
    " *******************************************************************************\n"
    " *                               CRYSTAL23                                     *\n"
    " *                      public : 1.0.1 - Dec 20th, 2022                        *\n"
    " *******************************************************************************\n"
)

# Converged MgO-shaped tail. The terminator is CRYSTAL23's dated
# TERMINATION banner, not CRYSTAL14's bare rule of E's.
CONVERGED_TAIL = (
    " CYCLE   8 TOTAL ENERGY(HF)(AU)(   8)         -2.7468175399E+02 DE-4.5E-12\n"
    " == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7468175400E+02\n"
    " EEEEEEEEEE TERMINATION  DATE 18 05 2026 TIME 15:14:00.2\n"
)


# ---------------------------------------------------------------------------
# Version probe
# ---------------------------------------------------------------------------


def test_probe_reads_the_major_version_off_the_banner():
    assert probe_version(CRYSTAL23_BANNER) == 23
    assert probe_version(" *                    CRYSTAL14              *") == 14
    assert probe_version(" *                    CRYSTAL17              *") == 17


def test_probe_returns_none_when_there_is_no_banner():
    """Absent is not wrong: a tail slice or a fixture has no banner."""
    assert probe_version("") is None
    assert probe_version(CONVERGED_TAIL) is None


def test_probe_ignores_lookalikes():
    """A narrow regex, because false positives would mislabel provenance."""
    # An install path is lowercase and must not be read as a banner.
    assert probe_version("/opt/crystal23/bin/crystal") is None
    # ``CRYSTAL`` also opens a periodic .d12 geometry block, bare.
    assert probe_version("CRYSTAL\n0 0 0\n225\n4.217") is None
    # Prose in an echoed deck title.
    assert probe_version("MgO - RHF (CRYSTAL parity)") is None


def test_probe_takes_the_first_match_not_a_later_echo():
    """CRYSTAL echoes the input deck; the banner must win.

    Were this the last match instead, a deck whose title happened to
    mention another version would silently relabel the run.
    """
    text = CRYSTAL23_BANNER + " ECHO OF INPUT: MgO CRYSTAL14 parity title\n"
    assert probe_version(text) == 23


# ---------------------------------------------------------------------------
# Version reaches the parsed result, without gating it
# ---------------------------------------------------------------------------


def test_parsed_result_carries_the_version():
    r = parse_output(CRYSTAL23_BANNER + CONVERGED_TAIL)
    assert r.ok
    assert r.crystal_version == 23
    assert r.version_matches(23) is True
    assert r.version_matches(14) is False


def test_version_is_provenance_not_a_gate():
    """A banner-less but converged output is still a usable energy.

    The retarget must not turn "I could not see the banner" into a
    failed evaluation -- that would reject every legitimately truncated
    or sliced output the parser was written to tolerate.
    """
    r = parse_output(CONVERGED_TAIL)
    assert r.ok
    assert r.energy == pytest.approx(-274.68175399)
    assert r.crystal_version is None
    assert r.version_matches(23) is None  # unknowable, not False


def test_the_default_target_is_crystal23():
    assert DEFAULT_CRYSTAL_VERSION == 23
    assert Crystal23Engine.__init__ is not None  # dataclass-generated


# ---------------------------------------------------------------------------
# EngineEnergy invariants
# ---------------------------------------------------------------------------


def test_ok_requires_an_energy_and_forbids_a_failure_mode():
    with pytest.raises(ValueError, match="requires an energy"):
        EngineEnergy(energy=None, ok=True, engine="x")
    with pytest.raises(ValueError, match="must not carry a failure_mode"):
        EngineEnergy(energy=-1.0, ok=True, engine="x", failure_mode="oops")


def test_failure_requires_a_mode():
    with pytest.raises(ValueError, match="requires a failure_mode"):
        EngineEnergy(energy=None, ok=False, engine="x")


def test_energy_if_ok_hides_a_diagnostic_energy():
    """A non-converged run's last-cycle energy must never leak out as
    a converged value -- consuming it silently is the whole failure
    mode this method exists to prevent."""
    bad = EngineEnergy.failed("x", "non_converged", energy=-274.0)
    assert bad.energy == pytest.approx(-274.0)  # available for diagnosis
    assert bad.energy_if_ok() is None           # but not as a result

    good = EngineEnergy(energy=-274.0, ok=True, engine="x")
    assert good.energy_if_ok() == pytest.approx(-274.0)


# ---------------------------------------------------------------------------
# The two views cannot disagree
# ---------------------------------------------------------------------------


class _StubEngine(EnergyEngine):
    """Minimal engine: implements only the two rich methods."""

    name = "stub"

    def __init__(self, result: EngineEnergy) -> None:
        self._result = result

    def version(self):
        return "1.2.3"

    def crystal_energy(self, basis_text, structure, method="rhf"):
        return self._result

    def atom_energy(self, basis_text, Z, method="rhf", *, host=None):
        return self._result


def test_float_view_is_derived_from_the_rich_view():
    ok = EngineEnergy(energy=-42.0, ok=True, engine="stub")
    eng = _StubEngine(ok)
    assert eng.evaluate_crystal("basis", object()) == pytest.approx(-42.0)
    assert eng.evaluate_atom("basis", 12) == pytest.approx(-42.0)

    bad = _StubEngine(EngineEnergy.failed("stub", "non_converged", energy=-41.0))
    assert bad.evaluate_crystal("basis", object()) is None
    assert bad.evaluate_atom("basis", 12) is None


def test_implementers_need_only_the_rich_methods():
    """Subclassing must not require reimplementing the float view."""
    eng = _StubEngine(EngineEnergy(energy=-1.0, ok=True, engine="stub"))
    assert isinstance(eng, EnergyEngine)
    assert eng.version() == "1.2.3"


def test_an_engine_must_implement_both_rich_methods():
    class _Half(EnergyEngine):
        name = "half"

        def crystal_energy(self, basis_text, structure, method="rhf"):
            raise AssertionError("unreachable")

    with pytest.raises(TypeError):
        _Half()  # atom_energy still abstract


# ---------------------------------------------------------------------------
# Engine roster
# ---------------------------------------------------------------------------


def test_crystal23_engine_names_itself_after_its_expected_version():
    class _NoTransport:
        pass

    eng = Crystal23Engine(transport=_NoTransport())
    assert eng.name == "crystal23"
    # No evaluation has run, so no version has been *observed*. Reporting
    # expected_version here would turn config into fake evidence.
    assert eng.version() is None

    eng14 = Crystal23Engine(transport=_NoTransport(), expected_version=14)
    assert eng14.name == "crystal14"


def test_gpaw_engine_is_constructible_but_loudly_unbuilt():
    """Declared at M-0 so the roster is fixed; implemented at M5.

    It raises rather than returning ok=False: ok=False means "infeasible
    point" and an optimizer routes around it silently, which would let a
    campaign 'finish' having computed nothing.
    """
    eng = GpawEngine()
    assert eng.name == "gpaw"
    assert isinstance(eng, EnergyEngine)
    with pytest.raises(NotImplementedError, match="M5"):
        eng.crystal_energy("basis", object())
    with pytest.raises(NotImplementedError, match="M5"):
        eng.atom_energy("basis", 12)


# ---------------------------------------------------------------------------
# How a job starts CRYSTAL (#225)
# ---------------------------------------------------------------------------


class _CapturingTransport:
    """Records what the engine ships, then fails so nothing is parsed."""

    def __init__(self):
        self.calls = []

    def run(self, work_dir, command, dest, **kwargs):
        from vibe_basis.transports.base import TransportError

        self.calls.append((Path(work_dir), list(command)))
        raise TransportError("captured")


def _rocksalt():
    from vibe_basis.io.structures import STRUCTURES

    return STRUCTURES["MgO"]


def test_wrapper_invocation_is_unchanged(tmp_path, monkeypatch):
    """The historical contract: a host script that takes the deck name."""
    monkeypatch.chdir(tmp_path)
    transport = _CapturingTransport()
    eng = Crystal23Engine(transport=transport, crystal_wrapper="run-crystal.sh")
    result = eng.crystal_energy("12 0\n99 0\nEND\n", _rocksalt(), method="pw1pw")
    assert not result.ok and result.failure_mode == "transport_error"
    (wd, command), = transport.calls
    assert command == ["run-crystal.sh", "MgO.d12"]
    assert not (wd / "run_crystal.sh").exists()


def test_stdin_invocation_stages_a_runner(tmp_path, monkeypatch):
    """A CRYSTAL binary reads stdin; given a file argument it reads nothing."""
    monkeypatch.chdir(tmp_path)
    transport = _CapturingTransport()
    eng = Crystal23Engine(
        transport=transport, crystal_wrapper="crystal23", invocation="stdin"
    )
    eng.crystal_energy("12 0\n99 0\nEND\n", _rocksalt(), method="pw1pw")
    (wd, command), = transport.calls
    assert command == ["sh", "run_crystal.sh"]
    runner = (wd / "run_crystal.sh").read_text()
    assert "exec crystal23 < MgO.d12 > MgO.out 2>&1" in runner
    assert (wd / "MgO.d12").is_file()


def test_unknown_invocation_is_refused():
    with pytest.raises(ValueError, match="invocation"):
        Crystal23Engine(transport=_CapturingTransport(), invocation="mpi")
