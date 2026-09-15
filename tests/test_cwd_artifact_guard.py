"""The suite must not write job artifacts into the directory it runs in.

``run_job(output=...)`` takes a path *stem*, so a bare relative name puts the
job's ``.out`` / ``.xyz`` / ``.system`` / ``.bibtex`` / ``.references`` /
``.scf.jsonl`` siblings wherever pytest was invoked. For this suite that is a
git checkout.

GitLab #508 is what that costs. A full-suite sweep left six ``h2_cas_test.*``
files in compute-medium's ``vq admin update``-managed release checkout; nothing
complained until the *next* fleet roll refused to deploy over a dirty tree, a
guard in a different system tripping on a symptom whose cause was nowhere near
it. The same six files sat untracked in a coordinator clone for an entire
session without being noticed.

So the invariant is pinned here rather than left to review: ``conftest`` snapshots
the invocation directory at ``pytest_sessionstart`` and fails the session at
``pytest_sessionfinish`` if new artifacts appeared. These tests exercise that
hook pair directly — a self-check of the guard, since a guard that silently
stops guarding is exactly the failure mode #508 is about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from vibeqc.output import OutputPlan
from vibeqc.output.formats.crash_dump import dump_on_failure

from .conftest import (
    job_artifacts_in,
    pytest_sessionfinish,
)


class _Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def write_sep(self, *_args, **_kwargs) -> None:
        self.lines.append("<sep>")

    def write_line(self, line: str) -> None:
        self.lines.append(line)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class _PluginManager:
    def __init__(self, reporter: _Reporter | None) -> None:
        self._reporter = reporter

    def get_plugin(self, name: str):
        return self._reporter if name == "terminalreporter" else None


class _Config:
    def __init__(self, reporter: _Reporter | None = None) -> None:
        self.pluginmanager = _PluginManager(reporter)


class _Session:
    def __init__(self, config: _Config, exitstatus: int = 0) -> None:
        self.config = config
        self.exitstatus = exitstatus


def _session_over(directory: Path, *, before: set[str], exitstatus: int = 0):
    """Replay the guard: ``before`` was the start-of-session snapshot."""
    reporter = _Reporter()
    config = _Config(reporter)
    config._vibeqc_invocation_dir = directory
    config._vibeqc_artifacts_at_start = before
    session = _Session(config, exitstatus)
    pytest_sessionfinish(session, exitstatus)
    return session, reporter


# --------------------------------------------------------------------------
# job_artifacts_in
# --------------------------------------------------------------------------

def _maximal_output_plans(tmp_path: Path) -> tuple[OutputPlan, ...]:
    """Plans covering every declarative output branch, including JSON perf."""
    common = {
        "output": tmp_path / "job",
        "method": "rhf",
        "basis": "sto-3g",
        "functional": None,
        "optimize": True,
        "structured_log": True,
        "crash_dump": True,
        "write_population": True,
        "write_cube_density": True,
        "cube_mo_labels": (0, "homo"),
        "write_poscar": True,
        "write_xsf_structure": True,
        "write_density_xsf": True,
        "write_cif": True,
        "job_kind": "periodic_scf",
        "output_qvf": True,
    }
    return (
        OutputPlan.from_run_job_kwargs(**common, perf_log=True),
        OutputPlan.from_run_job_kwargs(
            **common,
            perf_log=tmp_path / "job.perf.json",
        ),
    )


def test_recognises_every_artifact_declared_by_output_plan(tmp_path):
    """The guard inventory comes from the authoritative output planner."""
    planned_names = {
        planned.path.name
        for plan in _maximal_output_plans(tmp_path)
        for planned in plan.files
    }
    for name in planned_names:
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert planned_names <= job_artifacts_in(tmp_path)


def test_recognises_runtime_crash_dump_attachments(tmp_path):
    """Crash attachments are runtime-derived siblings, not separate plan rows."""
    dump_on_failure(
        tmp_path / "job",
        RuntimeError("probe"),
        {
            "density": np.eye(1),
            "fock": np.eye(1),
            "mo_coeffs": np.eye(1),
        },
    )
    written = {entry.name for entry in tmp_path.iterdir() if entry.is_file()}
    assert written <= job_artifacts_in(tmp_path)
    assert "job.dump.density.npy" in written


def test_ignores_files_that_are_not_job_artifacts(tmp_path):
    for name in (
        "notes.md",
        "script.py",
        "data.json",
        "notes.txt",
        "array.npy",
        "README",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert job_artifacts_in(tmp_path) == set()


def test_ignores_directories_and_does_not_recurse(tmp_path):
    """A directory named like an artifact is not one, and nested runs are
    somebody's own tmp_path rather than pollution of the caller's tree."""
    (tmp_path / "results.out").mkdir()
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "job.out").write_text("x", encoding="utf-8")
    assert job_artifacts_in(tmp_path) == set()


def test_missing_directory_is_not_an_error(tmp_path):
    assert job_artifacts_in(tmp_path / "does-not-exist") == set()


# --------------------------------------------------------------------------
# the session hook
# --------------------------------------------------------------------------

def test_a_clean_session_stays_green(tmp_path):
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    session, reporter = _session_over(tmp_path, before=set())
    assert session.exitstatus == 0
    assert reporter.text == ""


def test_artifacts_present_before_the_session_are_not_blamed_on_it(tmp_path):
    """Someone else's mess must not red an innocent run.

    Without this the guard would be unactionable in exactly the state #508
    leaves a checkout in: the files are already there, and every subsequent
    session would fail until a human deleted them.
    """
    (tmp_path / "stale.out").write_text("x", encoding="utf-8")
    session, reporter = _session_over(tmp_path, before={"stale.out"})
    assert session.exitstatus == 0
    assert reporter.text == ""


def test_a_leaked_artifact_fails_the_session_and_names_the_file(tmp_path):
    (tmp_path / "stale.out").write_text("x", encoding="utf-8")
    before = job_artifacts_in(tmp_path)
    (tmp_path / "h2_cas_test.out").write_text("x", encoding="utf-8")
    (tmp_path / "h2_cas_test.bibtex").write_text("x", encoding="utf-8")

    session, reporter = _session_over(tmp_path, before=before)

    assert session.exitstatus == 1, "a polluted working tree must not exit 0"
    assert "h2_cas_test.out" in reporter.text
    assert "h2_cas_test.bibtex" in reporter.text
    assert "stale.out" not in reporter.text
    assert "#508" in reporter.text, "the message must point at the issue"
    assert "tmp_path" in reporter.text, "and say what to do instead"


def test_an_already_failing_session_keeps_its_own_exit_status(tmp_path):
    """Reporting pollution must not overwrite a real test failure's status."""
    before = job_artifacts_in(tmp_path)
    (tmp_path / "leak.xyz").write_text("x", encoding="utf-8")
    session, reporter = _session_over(tmp_path, before=before, exitstatus=2)
    assert session.exitstatus == 2
    assert "leak.xyz" in reporter.text


def test_xdist_workers_do_not_report(tmp_path):
    """Workers share the controller's directory; one report is enough."""
    before = job_artifacts_in(tmp_path)
    (tmp_path / "leak.out").write_text("x", encoding="utf-8")
    reporter = _Reporter()
    config = _Config(reporter)
    config.workerinput = {"workerid": "gw0"}
    config._vibeqc_invocation_dir = tmp_path
    config._vibeqc_artifacts_at_start = before
    session = _Session(config)
    pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0
    assert reporter.text == ""


def test_a_session_without_a_snapshot_is_a_no_op(tmp_path):
    """``pytest_sessionstart`` may not have run (``--collect-only``, plugins)."""
    (tmp_path / "leak.out").write_text("x", encoding="utf-8")
    reporter = _Reporter()
    session = _Session(_Config(reporter))
    pytest_sessionfinish(session, 0)
    assert session.exitstatus == 0
    assert reporter.text == ""


def test_missing_terminalreporter_still_fails_the_session(tmp_path):
    """``-p no:terminal``: no place to print, but the exit status still moves."""
    before = job_artifacts_in(tmp_path)
    (tmp_path / "leak.out").write_text("x", encoding="utf-8")
    config = _Config(None)
    config._vibeqc_invocation_dir = tmp_path
    config._vibeqc_artifacts_at_start = before
    session = _Session(config)
    pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1
