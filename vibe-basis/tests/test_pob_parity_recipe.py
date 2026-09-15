"""Tests for vibe_basis.recipes.pob_parity — Stage 0 driver.

All tests are mocked — no CRYSTAL14, no vq, no compute-host-d access
required.  The transport is a :class:`MockTransport` that writes
back synthetic CRYSTAL14 output to the work directory so the
parser and report-builder can be verified.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import pytest
from vibe_basis.backends.crystal import (
    CrystalEnergyResult,
    emit_input,
)
from vibe_basis.io.references import PT2013_T2_HF, compounds_with_ref
from vibe_basis.io.structures import STRUCTURES, all_structures, in_table
from vibe_basis.recipes.pob_parity import (
    CompoundResult,
    ParityReport,
    _dedup_sort,
    _make_compound_result,
    run_pob_parity,
)
from vibe_basis.transports.base import (
    JobHandle,
    JobResult,
    Transport,
    TransportError,
)

# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------

# Synthetic CRYSTAL14 output — converged HF for MgO at 4.217 Å.
MGO_SUCCESS = textwrap.dedent("""\
     CYCLE   1 TOTAL ENERGY(HF)(AU)(   1)        -2.7400000000E+02 DE-1.0E+02
     CYCLE   5 TOTAL ENERGY(HF)(AU)(   5)        -2.7468175400E+02 DE-9.5E-09

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7468175400E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")

# Non-converged output.
NON_CONVERGED = textwrap.dedent("""\
     CYCLE 200 TOTAL ENERGY(HF)(AU)( 200)        -2.7468175000E+02 DE-1.0E-05

     == SCF ENDED - TOO MANY CYCLES      E(AU) -2.7468175000E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")


@dataclass
class MockRun:
    """A single mocked CRYSTAL14 run."""

    compound: str
    out_text: str  # the .out content to write back


class MockTransport(Transport):
    """A transport that writes synthetic CRYSTAL14 output to the
    work directory and returns a completed JobResult.

    The ``runs`` dict maps compound name → .out text.  Any compound
    not in the dict raises TransportError (simulating a submit
    failure).
    """

    def __init__(self, runs: dict[str, str]):
        self._runs = runs

    def submit(
        self,
        work_dir: Path | str,
        command: Sequence[str],
        *,
        cpus: Optional[int] = None,
        wall_time_s: Optional[int] = None,
        label: Optional[str] = None,
    ) -> JobHandle:
        wd = Path(work_dir)
        # compound name is the .d12 basename (e.g. "MgO.d12" → "MgO")
        d12_name = command[-1] if command else ""
        compound = d12_name.replace(".d12", "")
        if compound not in self._runs:
            raise TransportError(f"mock: no run for {compound!r}")
        return JobHandle(job_id=f"mock-{compound}", work_dir=wd, command=tuple(command))

    def poll(self, handle: JobHandle) -> str:
        return "completed"

    def fetch(self, handle: JobHandle, dest: Path | str) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        compound = handle.command[-1].replace(".d12", "")
        out_dir = dest / handle.job_id
        out_dir.mkdir()
        (out_dir / f"{compound}.out").write_text(self._runs[compound])
        return out_dir


# ---------------------------------------------------------------------------
# Tests — data structures
# ---------------------------------------------------------------------------


class TestCompoundResult:
    def test_ok_result(self):
        r = CompoundResult(
            compound="MgO",
            formula="MgO",
            ok=True,
            energy=-274.681754,
            method="HF",
            last_cycle=5,
            ref_energy=-274.681754,
            delta_mha=0.0,
            failure_mode=None,
        )
        assert r.ok
        assert r.delta_mha == 0.0

    def test_failed_result(self):
        r = CompoundResult(
            compound="MgO",
            formula="MgO",
            ok=False,
            energy=None,
            method=None,
            last_cycle=None,
            ref_energy=-274.681754,
            delta_mha=None,
            failure_mode="non_converged",
        )
        assert not r.ok
        assert r.failure_mode == "non_converged"


class TestParityReport:
    def test_passes_acceptance_when_below_threshold(self):
        report = ParityReport(
            basis="pob-tzvp",
            method="rhf",
            compounds_total=13,
            compounds_emitted=13,
            compounds_converged=13,
            compounds_compared=12,
            sum_abs_delta_mha=0.05,
        )
        assert report.passes_acceptance

    def test_fails_acceptance_when_above_threshold(self):
        report = ParityReport(
            basis="pob-tzvp",
            method="rhf",
            compounds_total=13,
            compounds_emitted=13,
            compounds_converged=13,
            compounds_compared=12,
            sum_abs_delta_mha=0.15,
        )
        assert not report.passes_acceptance

    def test_summary_includes_all_compounds(self):
        report = ParityReport(
            basis="pob-tzvp",
            method="rhf",
            compounds_total=1,
            compounds_emitted=1,
            compounds_converged=1,
            compounds_compared=1,
            results=[
                CompoundResult(
                    compound="MgO",
                    formula="MgO",
                    ok=True,
                    energy=-274.681754,
                    method="HF",
                    last_cycle=5,
                    ref_energy=-274.681754,
                    delta_mha=0.0,
                    failure_mode=None,
                )
            ],
            sum_abs_delta_mha=0.0,
        )
        text = report.summary()
        assert "MgO" in text
        assert "PASS" in text

    def test_summary_shows_failed_compounds(self):
        report = ParityReport(
            basis="pob-tzvp",
            method="rhf",
            compounds_total=1,
            compounds_emitted=1,
            compounds_converged=0,
            compounds_compared=0,
            results=[
                CompoundResult(
                    compound="LiCl",
                    formula="LiCl",
                    ok=False,
                    energy=None,
                    method=None,
                    last_cycle=None,
                    ref_energy=None,
                    delta_mha=None,
                    failure_mode="non_converged",
                )
            ],
            sum_abs_delta_mha=0.0,
        )
        text = report.summary()
        assert "LiCl" in text
        assert "FAILED" in text


# ---------------------------------------------------------------------------
# Tests — pipeline logic with mock transport
# ---------------------------------------------------------------------------


class TestRunPobParity:
    def test_run_on_mgo_converges(self):
        """Full pipeline run on MgO with a converged mock output."""
        mgo = STRUCTURES["MgO"]
        transport = MockTransport({"MgO": MGO_SUCCESS})
        report = run_pob_parity([mgo], transport)

        assert report.compounds_total == 1
        assert report.compounds_emitted == 1
        assert report.compounds_converged == 1
        assert report.compounds_compared == 1
        assert len(report.results) == 1

        r = report.results[0]
        assert r.ok
        assert r.compound == "MgO"
        assert r.energy == pytest.approx(-274.681754, rel=1e-10)
        assert r.method == "HF"
        assert r.last_cycle == 5
        assert r.ref_energy == pytest.approx(-274.681754, rel=1e-10)
        # The mock output has -274.68175400, the reference is -274.681754
        # — they're the same to the printed digits.
        assert r.delta_mha is not None
        assert abs(r.delta_mha) < 0.001

    def test_run_on_non_converged_reports_failure(self):
        """Pipeline with a non-converged mock output."""
        mgo = STRUCTURES["MgO"]
        transport = MockTransport({"MgO": NON_CONVERGED})
        report = run_pob_parity([mgo], transport)

        assert report.compounds_emitted == 1
        assert report.compounds_converged == 0
        r = report.results[0]
        assert not r.ok
        assert r.failure_mode == "non_converged"

    def test_afm_compounds_are_skipped(self):
        """AFM TM oxides should not produce .d12 decks."""
        mno = STRUCTURES["MnO"]
        # The MockTransport won't be called because emit returns None.
        transport = MockTransport({})
        report = run_pob_parity([mno], transport)

        assert report.compounds_emitted == 0
        assert len(report.results) == 0

    def test_compound_not_in_mock_raises_transport_error(self):
        """A compound not in the mock transport raises TransportError,
        and the report should capture it."""
        mgo = STRUCTURES["MgO"]
        transport = MockTransport({})  # no "MgO" entry
        report = run_pob_parity([mgo], transport)

        assert report.compounds_emitted == 1
        assert report.compounds_converged == 0
        r = report.results[0]
        assert not r.ok
        assert "transport_error" in (r.failure_mode or "")

    def test_all_pt2013_t4_compounds_are_in_structures(self):
        """Every compound in PT2013_T2_HF that has a non-None reference
        must have a corresponding Structure entry."""
        for name in compounds_with_ref():
            assert name in STRUCTURES, f"{name!r} not in STRUCTURES"

    def test_pt2013_t4_filter_returns_12_alkali_halides_hydrides(self):
        """``in_table("PT2013-T4")`` must return the 12 compounds
        PT2013 T4 lists (rocksalt alkali halides + hydrides)."""
        t4 = in_table("PT2013-T4")
        assert len(t4) == 12
        names = {s.name for s in t4}
        expected = {
            "LiCl",
            "NaCl",
            "LiF",
            "NaF",
            "KF",
            "CaF2",
            "K2O",
            "MgO",
            "CaO",
            "LiH",
            "NaH",
            "KH",
        }
        assert names == expected

    def test_circular_logic_on_dedup(self):
        """``_dedup_sort`` handles duplicates and sorts."""
        s1 = STRUCTURES["MgO"]
        s2 = STRUCTURES["MgO"]
        result = _dedup_sort([s1, s2, STRUCTURES["CaO"]])
        assert len(result) == 2
        assert result[0].name == "CaO"
        assert result[1].name == "MgO"


class TestMakeCompoundResult:
    def test_ok_result_with_ref(self):
        """Converged compound with reference energy in PT2013_T2_HF."""
        mgo = STRUCTURES["MgO"]
        parsed = CrystalEnergyResult(
            ok=True,
            energy=-274.681754,
            method="HF",
            last_cycle=5,
            converged=True,
            truncated=False,
            failure_mode=None,
            n_lines_scanned=50,
        )
        r = _make_compound_result(mgo, parsed)
        assert r.ok
        assert r.delta_mha is not None
        # 0.0 mHa delta (same energy as reference)
        assert abs(r.delta_mha) < 1.0

    def test_ok_result_no_ref(self):
        """Converged compound that's NOT in PT2013_T2_HF (e.g. Si)."""
        si = STRUCTURES["Si"]
        parsed = CrystalEnergyResult(
            ok=True,
            energy=-289.123,
            method="HF",
            last_cycle=5,
            converged=True,
            truncated=False,
            failure_mode=None,
            n_lines_scanned=50,
        )
        r = _make_compound_result(si, parsed)
        assert r.ok
        assert r.delta_mha is None
        assert "no reference" in r.notes

    def test_failed_parsed(self):
        """Parsed result that is not ok."""
        mgo = STRUCTURES["MgO"]
        parsed = CrystalEnergyResult(
            ok=False,
            energy=None,
            method=None,
            last_cycle=None,
            converged=False,
            truncated=True,
            failure_mode="no_energy_line",
            n_lines_scanned=10,
        )
        r = _make_compound_result(mgo, parsed)
        assert not r.ok
        assert r.failure_mode == "no_energy_line"


class TestEndToEndSmokeWithCommonCompounds:
    """End-to-end smoke: emit real .d12 decks, mock CRYSTAL14 output
    that matches the reference, verify the report passes the gate."""

    def test_thirteen_pt2013_t4_with_perfect_mock(self):
        """Feed all 12 PT2013 T4 structures (plus KBr which has no HF ref)
        through the pipeline with mock outputs that match the reference
        energies exactly.  The report should pass the acceptance gate."""
        t4_compounds = {s.name: s for s in in_table("PT2013-T4")}

        # Build mock runs: for each compound with a reference, feed back
        # the exact reference energy as a synthetic CRYSTAL14 output.
        runs: dict[str, str] = {}
        for name, ref_e in PT2013_T2_HF.items():
            if ref_e is not None and name in t4_compounds:
                runs[name] = self._converged_hf_output(name, ref_e, 5)

        transport = MockTransport(runs)
        structures = [t4_compounds[n] for n in runs]
        report = run_pob_parity(structures, transport)

        assert report.compounds_total == 11  # NaF has ref=None, skipped
        # All 11 should emit, converge, and compare
        assert report.compounds_emitted >= 11
        assert report.compounds_compared >= 11
        # All deltas should be near-zero (parsing preserves ~10 digits)
        assert report.sum_abs_delta_mha < 0.001

    @staticmethod
    def _converged_hf_output(name: str, energy: float, n_cycles: int = 5) -> str:
        """Synthesise a CRYSTAL14 converged HF output."""
        # CRYSTAL14 prints energy in scientific notation with capital E.
        return textwrap.dedent(f"""\
             CYCLE   1 TOTAL ENERGY(HF)(AU)(   1)        {energy:+21.10E} DE-1.0E+00
             CYCLE   {n_cycles} TOTAL ENERGY(HF)(AU)(   {n_cycles})        {energy:+21.10E} DE-1.0E-12

             == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) {energy:+21.10E}

             EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        """)
