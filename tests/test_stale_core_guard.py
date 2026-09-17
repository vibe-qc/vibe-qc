"""The stale-compiled-core guard: refuse the session, and never look like a red test.

The editable install has scikit-build auto-rebuild OFF, so a ``git pull``
carrying C++ commits leaves this checkout's Python running against an older
checkout's ``_vibeqc_core*.so``. That pairing computes wrong numbers in both
directions: a red test reads as a physics regression that does not exist
(2026-07-10, ``test_ccm_direct.py``; 2026-09-13, a 900 s timeout in
``test_basis_filter.py`` charged to the branch under test), and a green test
certifies the previous commit's C++ under this commit's name.

Until #285 the guard only warned. These tests pin the three things that make
the refusal readable rather than confusing:

* it happens before any test runs, and exits a status pytest itself never
  produces, so nothing downstream can read it as a failure;
* the two consumers of that status agree on its value and label it distinctly
  (``run_full_suite.classify``) or refuse to compute a verdict at all
  (``gate_verdict.validate_blocking_artifact``);
* the override keeps the session runnable, and then says so *underneath* the
  results, where it cannot scroll past the failures the way the session-start
  warning did.

Pure Python: no built core, no pytest spawn, no fleet. The ``cpp/`` tree is
never touched — a test that bumped a source mtime would make the core stale for
real, for every session after it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

# ``pytest.exit`` is the public call; ``Exit`` is the exception it raises and
# has no public alias, so the test for a refusal has to name the private one.
from _pytest.outcomes import Exit

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = REPO_ROOT / "scripts" / "test_gate"


def _load(name: str, path: Path):
    """Load a module by path.

    The gate scripts live outside the importable package, and this suite does
    not put ``tests/`` on ``sys.path``, so ``conftest`` is not importable by
    name either. Loading it here yields a second module object, which is what
    these tests want: they monkeypatch its globals, and doing that to the copy
    pytest is running its own hooks from would change the behaviour of the
    session executing the test.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vqc_conftest = _load("vqc_conftest_under_test", REPO_ROOT / "tests" / "conftest.py")
run_full_suite = _load("run_full_suite_under_test", GATE_DIR / "run_full_suite.py")
gate_verdict = _load("gate_verdict_under_test", GATE_DIR / "gate_verdict.py")


# ------------------------------------------------------------------
# newest_cpp_source
# ------------------------------------------------------------------

def test_newest_cpp_source_reports_the_most_recently_touched_source(tmp_path):
    """The reported path is the one the reader has to rebuild for."""
    (tmp_path / "src").mkdir()
    old = tmp_path / "src" / "old.cpp"
    new = tmp_path / "include" / "new.hpp"
    new.parent.mkdir()
    old.write_text("// old\n")
    new.write_text("// new\n")
    import os

    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))

    found = vqc_conftest.newest_cpp_source(tmp_path)
    assert found is not None
    mtime, path = found
    assert path == new
    assert mtime == pytest.approx(2_000_000)


def test_newest_cpp_source_ignores_files_that_are_not_c_plus_plus(tmp_path):
    """A newer README or CMakeLists must not be read as a source change."""
    source = tmp_path / "a.cpp"
    source.write_text("// c++\n")
    other = tmp_path / "CMakeLists.txt"
    other.write_text("# build\n")
    import os

    os.utime(source, (1_000_000, 1_000_000))
    os.utime(other, (9_000_000, 9_000_000))

    found = vqc_conftest.newest_cpp_source(tmp_path)
    assert found is not None
    assert found[1] == source


def test_newest_cpp_source_is_none_without_sources(tmp_path):
    """An sdist, a CI image or an installed wheel has no cpp/ to compare."""
    assert vqc_conftest.newest_cpp_source(tmp_path) is None


# ------------------------------------------------------------------
# The refusal
# ------------------------------------------------------------------

def _fake_session(tmp_path):
    config = SimpleNamespace(invocation_params=SimpleNamespace(dir=str(tmp_path)))
    return SimpleNamespace(config=config)


def test_sessionstart_refuses_a_stale_core_with_the_reserved_status(tmp_path, monkeypatch):
    """Nothing runs, and the status is one no test failure can produce."""
    monkeypatch.delenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(
        vqc_conftest, "stale_core_report", lambda: "core.so was built before x.cpp."
    )
    session = _fake_session(tmp_path)

    with pytest.raises(Exit) as excinfo:
        vqc_conftest.pytest_sessionstart(session)

    assert excinfo.value.returncode == vqc_conftest.STALE_CORE_EXIT_STATUS
    assert vqc_conftest.STALE_CORE_EXIT_STATUS not in {0, 1, 2, 3, 4, 5}


def test_sessionstart_still_snapshots_artifacts_before_refusing(tmp_path, monkeypatch):
    """The #508 guard's half of this hook must not be skipped by the refusal."""
    monkeypatch.delenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(vqc_conftest, "stale_core_report", lambda: "stale.")
    session = _fake_session(tmp_path)

    with pytest.raises(Exit):
        vqc_conftest.pytest_sessionstart(session)

    assert session.config._vibeqc_artifacts_at_start == set()
    assert session.config._vibeqc_invocation_dir == tmp_path


def test_sessionstart_is_silent_when_the_core_is_current(tmp_path, monkeypatch):
    """A current core costs nothing: no exit, no warning, no stashed report."""
    monkeypatch.setattr(vqc_conftest, "stale_core_report", lambda: None)
    session = _fake_session(tmp_path)

    vqc_conftest.pytest_sessionstart(session)

    assert not hasattr(session.config, "_vibeqc_stale_core")


def test_the_refusal_message_names_the_rebuild_the_override_and_the_status(tmp_path):
    """The reader must not have to find #285 to get unstuck."""
    text = vqc_conftest.stale_core_refusal("core.so was built before x.cpp.")
    assert "core.so was built before x.cpp." in text
    assert "pip install -e ." in text
    assert "--config-settings=build-dir=build-local" in text
    assert vqc_conftest.STALE_CORE_OVERRIDE_ENV in text
    assert str(vqc_conftest.STALE_CORE_EXIT_STATUS) in text
    # Both directions, not just the one that produces a red suite.
    assert "green" in text and "red" in text


# ------------------------------------------------------------------
# The override
# ------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_override_downgrades_the_refusal_to_a_warning(tmp_path, monkeypatch, value):
    monkeypatch.setenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, value)
    monkeypatch.setattr(vqc_conftest, "stale_core_report", lambda: "stale.")
    session = _fake_session(tmp_path)

    with pytest.warns(UserWarning, match="STALE COMPILED CORE"):
        vqc_conftest.pytest_sessionstart(session)

    assert session.config._vibeqc_stale_core == "stale."


@pytest.mark.parametrize("value", ["0", "", "no", "maybe"])
def test_a_non_truthy_override_does_not_unlock_the_session(tmp_path, monkeypatch, value):
    """Anything but an explicit yes still refuses; a typo must not run silently."""
    monkeypatch.setenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, value)
    monkeypatch.setattr(vqc_conftest, "stale_core_report", lambda: "stale.")

    with pytest.raises(Exit):
        vqc_conftest.pytest_sessionstart(_fake_session(tmp_path))


class _Reporter:
    def __init__(self):
        self.lines = []

    def write_sep(self, sep, title, **kwargs):  # noqa: ARG002
        self.lines.append(title)

    def write_line(self, line):
        self.lines.append(line)


def test_the_override_banner_repeats_below_the_results(tmp_path):
    """The session-start warning scrolls past the failures; this copy does not."""
    config = SimpleNamespace(_vibeqc_stale_core="core.so was built before x.cpp.")
    reporter = _Reporter()

    vqc_conftest.pytest_terminal_summary(reporter, 1, config)

    joined = "\n".join(reporter.lines)
    assert "STALE COMPILED CORE" in joined
    assert "core.so was built before x.cpp." in joined


def test_the_override_banner_also_prints_on_a_green_run(tmp_path):
    """A pass on a stale core is the more misleading outcome of the two."""
    config = SimpleNamespace(_vibeqc_stale_core="stale.")
    reporter = _Reporter()

    vqc_conftest.pytest_terminal_summary(reporter, 0, config)

    assert reporter.lines


def test_no_banner_when_the_core_was_current():
    config = SimpleNamespace()
    reporter = _Reporter()

    vqc_conftest.pytest_terminal_summary(reporter, 0, config)

    assert reporter.lines == []


# ------------------------------------------------------------------
# The downstream consumers
# ------------------------------------------------------------------

def test_the_gate_runner_agrees_on_the_reserved_status():
    """Two files carry the literal; a drift between them would mislabel a lane."""
    assert (
        run_full_suite.STALE_CORE_EXIT_STATUS
        == vqc_conftest.STALE_CORE_EXIT_STATUS
    )


def test_classify_labels_the_refusal_distinctly_from_a_failure():
    status = run_full_suite.classify(
        vqc_conftest.STALE_CORE_EXIT_STATUS, timed_out=False, counts={}
    )
    assert status == "STALE_CORE"
    assert status != "FAIL"
    # Not swept into the usage/collection bucket either: nothing was wrong with
    # the invocation, and the remedy is a rebuild, not a command-line fix.
    assert status != "COLLECT_ERR"


def test_a_stale_core_still_counts_as_failing_on_an_advisory_lane():
    """It must never be counted as healthy, wherever it is seen."""
    assert gate_verdict.is_failing("STALE_CORE") is True


def _blocking_record(file, status):
    return {"file": file, "status": status, "tier": "T1"}


def test_a_blocking_verdict_refuses_an_artifact_holding_a_stale_core_run():
    """No test ran on that host, so the artifact is neither green nor red."""
    runs = [
        _blocking_record("tests/test_a.py", "PASS"),
        _blocking_record("tests/test_b.py", "STALE_CORE"),
    ]
    with pytest.raises(ValueError) as excinfo:
        gate_verdict.validate_blocking_artifact(runs, "triage.jsonl")

    message = str(excinfo.value)
    assert "STALE_CORE" in message
    assert "tests/test_b.py" in message
    assert "cannot produce a verdict" in message


def test_a_clean_blocking_artifact_is_still_accepted():
    runs = [_blocking_record("tests/test_a.py", "PASS")]
    gate_verdict.validate_blocking_artifact(runs, "triage.jsonl")
