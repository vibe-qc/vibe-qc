"""Exercise example job staging with a queue checkout outside vibe-qc."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from examples.regression.parity_matrix_orca import orca_vq
from examples.regression.parity_matrix_orca.cases import PARITY_CELLS


ROOT = Path(__file__).resolve().parents[1]


def test_orca_stages_wrapper_from_separate_checkout(tmp_path, monkeypatch):
    checkout = tmp_path / "independent queue"
    wrapper = checkout / "contrib" / "run-orca.sh"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("#!/bin/bash\nprintf 'separate queue wrapper'\n")
    monkeypatch.setenv("VIBEQC_QUEUE_CHECKOUT", str(checkout))
    monkeypatch.setattr(orca_vq, "orca_binary_path", lambda: "/opt/orca/orca")
    calls = []

    def submit(*args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "test-job\n", "")

    monkeypatch.setattr(orca_vq, "_vq", submit)
    jobid, workspace = orca_vq.submit_orca_cell(
        PARITY_CELLS[0], workspace_root=tmp_path / "jobs",
    )
    assert jobid == "test-job"
    assert (workspace / "run-orca.sh").read_bytes() == wrapper.read_bytes()
    assert (workspace / f"{PARITY_CELLS[0].cell_id}.inp").is_file()
    assert calls[0][-3:] == ("bash", "run-orca.sh", f"{PARITY_CELLS[0].cell_id}.inp")


def test_missing_queue_fails_before_creating_or_submitting_job(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQC_QUEUE_CHECKOUT", str(tmp_path / "missing"))
    monkeypatch.setattr(orca_vq, "_vq", lambda *args: pytest.fail("submitted job"))
    with pytest.raises(orca_vq.OrcaVqError, match="VIBEQC_QUEUE_CHECKOUT.*separate"):
        orca_vq.submit_orca_cell(PARITY_CELLS[0], workspace_root=tmp_path / "jobs")
    assert not (tmp_path / "jobs").exists()


@pytest.mark.parametrize("kind", ["crystal", "orca"])
def test_shell_driver_dispatches_to_separate_checkout(tmp_path, kind):
    checkout = tmp_path / "independent queue"
    wrapper = checkout / "contrib" / f"run-{kind}.sh"
    wrapper.parent.mkdir(parents=True)
    marker = tmp_path / "arguments.txt"
    wrapper.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$WRAPPER_TEST_MARKER"\n')
    if kind == "crystal":
        script = ROOT / "examples/regression/crystal_parity/run-reference.sh"
        args = ["input with spaces.d12", "--serial"]
        expected = args
    else:
        script = tmp_path / "run_orca_system.sh"
        shutil.copy2(ROOT / "examples/molecular/mp2_benchmarks/s22/run_orca_system.sh", script)
        (tmp_path / "input with spaces.inp").write_text("! HF STO-3G\n")
        args = []
        expected = ["input with spaces.inp", "input with spaces.out"]
    result = subprocess.run(
        ["bash", str(script), *args], text=True, capture_output=True,
        env={**os.environ, "VIBEQC_QUEUE_CHECKOUT": str(checkout),
             "WRAPPER_TEST_MARKER": str(marker), "ORCA_BIN": sys.executable},
    )
    assert result.returncode == 0, result.stderr
    assert marker.read_text().splitlines() == expected
