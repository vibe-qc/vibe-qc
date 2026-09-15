"""Tests for vibe_basis.recipes.run_d12 — run pre-existing .d12 files.

Three real .d12 fixtures (MgO, CaO, LiF — geometry optimization
with PW1PW + inline basis sets) are in tests/fixtures/d12/.
Tests verify metadata extraction and the runner pipeline.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional, Sequence

import pytest
from vibe_basis.recipes.run_d12 import (
    D12BatchReport,
    D12RunResult,
    inspect_d12,
    run_d12_batch,
    run_d12_file,
)
from vibe_basis.transports.base import (
    JobHandle,
    Transport,
    TransportError,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "d12"


# ---------------------------------------------------------------------------
# Mock transport
# ---------------------------------------------------------------------------

CRYSTAL_DFT_OUT = textwrap.dedent("""\
     CYCLE   1 TOTAL ENERGY(DFT)(AU)(   1)        -2.7500000000E+02 DE-1.0E+02
     CYCLE  12 TOTAL ENERGY(DFT)(AU)(  12)        -2.7547759489E+02 DE-3.2E-11

     == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) -2.7547759489E+02

     EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
""")


class MockRunTransport(Transport):
    """Transport that writes synthetic CRYSTAL output to the workdir."""

    def __init__(self, out_text: str = CRYSTAL_DFT_OUT):
        self.out_text = out_text

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
        d12 = command[-1] if command else "unknown.d12"
        compound = d12.replace(".d12", "")
        return JobHandle(job_id=f"mock-{compound}", work_dir=wd, command=tuple(command))

    def poll(self, handle: JobHandle) -> str:
        return "completed"

    def fetch(self, handle: JobHandle, dest: Path | str) -> Path:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        compound = handle.command[-1].replace(".d12", "")
        out_dir = dest / handle.job_id
        out_dir.mkdir()
        (out_dir / f"{compound}.out").write_text(self.out_text)
        return out_dir


# ---------------------------------------------------------------------------
# Tests — fixture inspection
# ---------------------------------------------------------------------------


class TestInspectD12:
    def test_mgo_has_optgeom_and_inline_basis(self):
        info = inspect_d12(FIXTURES / "MgO_seg_PW1PW.d12")
        assert info["compound"] == "MgO"
        assert info["has_optgeom"] == "yes"
        assert info["has_inline_basis"] == "yes"
        assert info["method"] == "PW1PW"

    def test_cao_has_optgeom_and_inline_basis(self):
        info = inspect_d12(FIXTURES / "CaO_seg_PW1PW.d12")
        assert info["compound"] == "CaO"
        assert info["has_optgeom"] == "yes"
        assert info["has_inline_basis"] == "yes"
        assert info["method"] == "PW1PW"

    def test_lif_has_optgeom_and_inline_basis(self):
        info = inspect_d12(FIXTURES / "LiF_seg_PW1PW.d12")
        assert info["compound"] == "LiF"
        assert info["has_optgeom"] == "yes"
        assert info["has_inline_basis"] == "yes"
        assert info["method"] == "PW1PW"

    def test_all_three_fixtures_exist(self):
        for name in ["MgO_seg_PW1PW.d12", "CaO_seg_PW1PW.d12", "LiF_seg_PW1PW.d12"]:
            assert (FIXTURES / name).exists(), f"missing fixture: {name}"

    def test_fixture_mgo_contains_crystal_keyword(self):
        text = (FIXTURES / "MgO_seg_PW1PW.d12").read_text()
        assert "CRYSTAL" in text
        assert "PW1PW" in text
        assert "OPTGEOM" in text
        assert "ENDBS" in text
        assert "SHRINK" in text


# ---------------------------------------------------------------------------
# Tests — runner
# ---------------------------------------------------------------------------


class TestRunD12File:
    def test_run_mgo_with_mock_transport(self, tmp_path):
        """Run MgO .d12 through the mock transport."""
        transport = MockRunTransport()
        result = run_d12_file(
            FIXTURES / "MgO_seg_PW1PW.d12",
            transport,
            workdir_root=tmp_path,
        )
        assert result.ok
        assert result.compound == "MgO"
        assert result.energy == pytest.approx(-275.47759489, rel=1e-10)
        assert result.method == "DFT"
        assert result.last_cycle == 12

    def test_run_cao_with_mock_transport(self, tmp_path):
        transport = MockRunTransport()
        result = run_d12_file(
            FIXTURES / "CaO_seg_PW1PW.d12",
            transport,
            workdir_root=tmp_path,
        )
        assert result.ok
        assert result.compound == "CaO"

    def test_run_lif_with_mock_transport(self, tmp_path):
        transport = MockRunTransport()
        result = run_d12_file(
            FIXTURES / "LiF_seg_PW1PW.d12",
            transport,
            workdir_root=tmp_path,
        )
        assert result.ok
        assert result.compound == "LiF"

    def test_copies_d12_to_workdir(self, tmp_path):
        """Verify the .d12 is copied into the workdir."""
        transport = MockRunTransport()
        result = run_d12_file(
            FIXTURES / "MgO_seg_PW1PW.d12",
            transport,
            workdir_root=tmp_path,
        )
        workdir = tmp_path / "MgO"
        assert (workdir / "MgO.d12").exists()
        content = (workdir / "MgO.d12").read_text()
        assert "PW1PW" in content


class TestRunD12Batch:
    def test_batch_runs_all_three(self, tmp_path):
        transport = MockRunTransport()
        paths = [
            FIXTURES / "MgO_seg_PW1PW.d12",
            FIXTURES / "CaO_seg_PW1PW.d12",
            FIXTURES / "LiF_seg_PW1PW.d12",
        ]
        report = run_d12_batch(paths, transport, workdir_root=tmp_path)

        assert report.files_total == 3
        assert report.files_completed == 3
        assert report.files_failed == 0
        assert len(report.results) == 3

        compounds = {r.compound for r in report.results}
        assert compounds == {"MgO", "CaO", "LiF"}

    def test_batch_summary(self, tmp_path):
        transport = MockRunTransport()
        paths = [FIXTURES / "MgO_seg_PW1PW.d12"]
        report = run_d12_batch(paths, transport, workdir_root=tmp_path)
        text = report.summary()
        assert "MgO" in text
        assert "1/1 completed" in text


# ---------------------------------------------------------------------------
# Tests — report dataclass
# ---------------------------------------------------------------------------


class TestD12BatchReport:
    def test_summary_shows_failures(self):
        report = D12BatchReport(
            files_total=3,
            files_completed=2,
            files_failed=1,
            results=[
                D12RunResult(
                    d12_path=Path("MgO.d12"),
                    compound="MgO",
                    ok=True,
                    energy=-275.47759489,
                    method="DFT",
                    last_cycle=12,
                    failure_mode=None,
                    n_lines=100,
                ),
                D12RunResult(
                    d12_path=Path("CaO.d12"),
                    compound="CaO",
                    ok=False,
                    energy=None,
                    method=None,
                    last_cycle=None,
                    failure_mode="non_converged",
                    n_lines=200,
                ),
            ],
        )
        text = report.summary()
        assert "MgO" in text
        assert "CaO" in text
        assert "FAILED" in text
        assert "2/3" in text
