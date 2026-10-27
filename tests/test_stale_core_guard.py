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

Pure Python, including real xdist subprocesses with stat-only core fixtures;
no numerical execution or fleet access. The product ``cpp/`` tree is
never touched — a test that bumped a source mtime would make the core stale for
real, for every session after it.
"""

from __future__ import annotations

import importlib.util
import os
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


@pytest.mark.parametrize("override", ["1", "true", "YES", " on "])
@pytest.mark.parametrize("target", ["tests/test_probe.py", "vibe-queue/tests/test_probe.py"])
def test_gate_child_cannot_inherit_diagnostic_stale_override(
    tmp_path, monkeypatch, override, target,
):
    """Both gate entrypoints use this environment; direct diagnostics keep it."""
    monkeypatch.setenv("VIBEQC_ALLOW_STALE_CORE", override)
    with run_full_suite._isolated_test_environment(tmp_path, target) as env:
        assert "VIBEQC_ALLOW_STALE_CORE" not in env
        assert os.environ["VIBEQC_ALLOW_STALE_CORE"] == override
    assert os.environ["VIBEQC_ALLOW_STALE_CORE"] == override


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
# The Libxc the core loads (#570)
# ------------------------------------------------------------------

def _libxc_checkout(tmp_path: Path, *, patch_script: bool = True) -> tuple[Path, Path]:
    """A checkout stand-in holding a vendored Libxc image (an empty file)."""
    checkout = tmp_path / "checkout"
    if patch_script:
        script = checkout / "scripts" / "patch_libxc_spin_fractions.py"
        script.parent.mkdir(parents=True)
        script.write_text("# stand-in\n")
    image = checkout / "third_party" / "libxc" / "install" / "lib" / "libxc.15.dylib"
    image.parent.mkdir(parents=True)
    image.write_text("stand-in image, never loaded\n")
    return checkout, image.resolve()


def _never(*_args):
    raise AssertionError("must not be consulted")


def test_unpatched_checkout_libxc_is_refused_with_its_own_remedy(tmp_path, monkeypatch):
    checkout, image = _libxc_checkout(tmp_path)
    monkeypatch.setattr(vqc_conftest, "_loaded_libxc_image", lambda core: image)
    monkeypatch.setattr(vqc_conftest, "_image_exports", lambda image, symbol: False)

    report = vqc_conftest.unpatched_libxc_report(tmp_path / "core.so", checkout)

    assert report.startswith(
        "Libxc image third_party/libxc/install/lib/libxc.15.dylib, which core.so loads,"
    )
    assert "39bf22a7" in report
    refusal = vqc_conftest.stale_core_refusal(report)
    assert refusal.startswith("STALE LIBXC")
    # Libxc is loaded at run time: rebuilding it needs no core rebuild.
    assert "scripts/build_libxc.sh" in refusal
    assert "pip install -e ." not in refusal
    assert vqc_conftest.STALE_CORE_OVERRIDE_ENV in refusal
    warning = vqc_conftest.stale_core_warning(report)
    assert warning.startswith("STALE LIBXC") and "scripts/build_libxc.sh" in warning


def test_patched_checkout_libxc_is_silent(tmp_path, monkeypatch):
    checkout, image = _libxc_checkout(tmp_path)
    monkeypatch.setattr(vqc_conftest, "_loaded_libxc_image", lambda core: image)
    seen = []
    monkeypatch.setattr(
        vqc_conftest, "_image_exports",
        lambda image, symbol: seen.append(symbol) or True,
    )

    assert vqc_conftest.unpatched_libxc_report(tmp_path / "core.so", checkout) is None
    assert seen == ["vibeqc_libxc_spin_fractions_version"]


def test_libxc_outside_the_checkout_is_not_this_guards_business(tmp_path, monkeypatch):
    """A system Libxc, a wheel or a CI image: build_libxc.sh cannot repair it."""
    checkout, _ = _libxc_checkout(tmp_path)
    foreign = tmp_path / "prefix" / "lib" / "libxc.dylib"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("stand-in\n")
    monkeypatch.setattr(vqc_conftest, "_loaded_libxc_image", lambda core: foreign.resolve())
    monkeypatch.setattr(vqc_conftest, "_image_exports", _never)

    assert vqc_conftest.unpatched_libxc_report(tmp_path / "core.so", checkout) is None


def test_a_checkout_without_the_patch_does_not_probe_libxc(tmp_path, monkeypatch):
    checkout, _ = _libxc_checkout(tmp_path, patch_script=False)
    monkeypatch.setattr(vqc_conftest, "_loaded_libxc_image", _never)

    assert vqc_conftest.unpatched_libxc_report(tmp_path / "core.so", checkout) is None


@pytest.mark.parametrize("error", [OSError, AttributeError, ValueError])
def test_an_unidentifiable_libxc_is_silent(tmp_path, monkeypatch, error):
    checkout, _ = _libxc_checkout(tmp_path)

    def fail(core):
        raise error("no image")

    monkeypatch.setattr(vqc_conftest, "_loaded_libxc_image", fail)
    assert vqc_conftest.unpatched_libxc_report(tmp_path / "core.so", checkout) is None


def test_stale_core_report_checks_libxc_only_once_the_core_is_current(monkeypatch):
    from vibeqc import _vibeqc_core

    built = Path(_vibeqc_core.__file__).stat().st_mtime
    source = REPO_ROOT / "cpp" / "stand_in.cpp"
    monkeypatch.setattr(vqc_conftest, "unpatched_libxc_report", lambda core, checkout: "Libxc image x")

    monkeypatch.setattr(vqc_conftest, "newest_cpp_source", lambda cpp: (built - 10.0, source))
    assert vqc_conftest.stale_core_report() == "Libxc image x"

    monkeypatch.setattr(vqc_conftest, "newest_cpp_source", lambda cpp: (built + 10.0, source))
    assert "cpp/stand_in.cpp changed" in vqc_conftest.stale_core_report()


def test_the_real_core_resolves_a_libxc_image():
    """The probe itself, on this checkout's core: no stand-in."""
    from vibeqc import _vibeqc_core

    try:
        image = vqc_conftest._loaded_libxc_image(Path(_vibeqc_core.__file__))
    except (OSError, AttributeError) as error:
        pytest.skip(f"core does not resolve xc_version dynamically: {error}")
    assert image.is_file()
    assert vqc_conftest._image_exports(image, "xc_version")


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
    """The aggregate cannot certify all workers, so it is neither green nor red."""
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


# xdist uses a normal worker-finish message, never stderr parsing or rc5 guessing.
def _worker_node(payload, config=None):
    return SimpleNamespace(
        config=config or SimpleNamespace(),
        workeroutput={vqc_conftest._WORKER_CORE_PROVENANCE: payload},
    )


def test_stale_worker_refuses_at_collection_boundary_with_transportable_proof(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, raising=False)
    monkeypatch.setattr(vqc_conftest, "stale_core_report", lambda: "worker core is stale")
    session = _fake_session(tmp_path)
    session.config.workerinput = {"workerid": "gw0"}
    session.config.workeroutput = {}
    # Finish sessionstart so xdist's sessionfinish transport can run later.
    vqc_conftest.pytest_sessionstart(session)
    with pytest.raises(Exit) as excinfo:
        vqc_conftest.pytest_collection(session)
    assert excinfo.value.returncode == 97
    assert session.config.workeroutput[vqc_conftest._WORKER_CORE_PROVENANCE] == {
        "version": 1, "state": "stale", "report": "worker core is stale",
    }


def test_current_worker_provenance_preserves_empty_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(vqc_conftest, "stale_core_report", lambda: None)
    session = _fake_session(tmp_path)
    session.config.workerinput = {"workerid": "gw0"}
    session.config.workeroutput = {}
    vqc_conftest.pytest_sessionstart(session)
    assert vqc_conftest.pytest_collection(session) is None
    node = SimpleNamespace(config=SimpleNamespace(), workeroutput=session.config.workeroutput)
    assert vqc_conftest.pytest_testnodedown(node, None) is None
    assert run_full_suite.classify(5, False, {}) == "NOTESTS"


@pytest.mark.parametrize("prior_status", [0, 1, 5])
def test_one_stale_worker_voids_prior_healthy_outcomes(monkeypatch, prior_status):
    monkeypatch.delenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, raising=False)
    node = _worker_node({"version": 1, "state": "stale", "report": "stale worker"})
    node.config.exitstatus = prior_status
    with pytest.raises(Exit) as excinfo:
        vqc_conftest.pytest_testnodedown(node, None)
    assert excinfo.value.returncode == 97


@pytest.mark.parametrize("payload", [
    None, {}, [], {"version": 1, "state": [], "report": None},
    {"version": True, "state": "current", "report": None},
    {"version": 2, "state": "current", "report": None},
    {"version": 1, "state": "stale", "report": ""},
    {"version": 1, "state": "current", "report": "unexpected"},
])
def test_malformed_worker_report_is_void_infrastructure_not_stale(payload):
    with pytest.raises(Exit, match="CORE PROVENANCE ERROR") as excinfo:
        vqc_conftest.pytest_testnodedown(_worker_node(payload), None)
    assert excinfo.value.returncode == 96
    status = run_full_suite.classify(excinfo.value.returncode, False, {})
    assert status == "CORE_PROVENANCE_ERROR"
    with pytest.raises(ValueError, match="CORE_PROVENANCE_ERROR"):
        gate_verdict.validate_blocking_artifact([_blocking_record("tests/test_a.py", status)], "run.jsonl")
    with pytest.raises(ValueError, match="CORE_PROVENANCE_ERROR"):
        gate_verdict.reverify_new_reds(
            [_blocking_record("tests/test_a.py", "FAIL")],
            lambda _path: {"status": status},
        )


def test_worker_crash_without_provenance_retains_xdist_error_semantics():
    assert vqc_conftest.pytest_testnodedown(SimpleNamespace(), "worker crashed") is None


def test_worker_override_cannot_opt_controller_into_diagnostics(monkeypatch):
    monkeypatch.delenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, raising=False)
    with pytest.raises(Exit) as excinfo:
        vqc_conftest.pytest_testnodedown(
            _worker_node({"version": 1, "state": "diagnostic", "report": "stale worker"}), None,
        )
    assert excinfo.value.returncode == 97


def test_worker_only_override_warning_reaches_current_controller(monkeypatch):
    monkeypatch.setenv(vqc_conftest.STALE_CORE_OVERRIDE_ENV, "YES")
    config = SimpleNamespace()
    node = _worker_node({"version": 1, "state": "diagnostic", "report": "stale worker"}, config)
    vqc_conftest.pytest_testnodedown(node, None)
    reporter = _Reporter()
    vqc_conftest.pytest_terminal_summary(reporter, 0, config)
    assert "stale worker" in "\n".join(reporter.lines)
    assert "results computed on a STALE COMPILED CORE" in reporter.lines


@pytest.mark.parametrize("restart_option", [[], ["--max-worker-restart=0"]])
@pytest.mark.parametrize("scenario,expected", [
    ("stale-workers", 97), ("mixed-workers", 97),
    ("late-stale-report", 97), ("diagnostic-workers", 0),
    ("malformed-report", 96), ("missing-report", 96),
    ("current-empty", 5), ("current-pass", 0), ("current-fail", 1),
    ("worker-crash", 5),
])
def test_real_xdist_core_provenance_protocol(tmp_path, scenario, expected, restart_option):
    """Real workers, stat-only core: provenance must survive normal transport.

    The late report is a deliberate protocol fault control, not a claim that
    the current fixture core became stale. It arrives after real test execution
    and must still void the aggregate. Stale-source cases refuse before collection.
    """
    import json
    import shutil
    import signal
    import subprocess
    import sys

    pytest.importorskip("xdist")
    root = tmp_path / "protocol"
    tests = root / "tests"
    tests.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "tests/conftest.py", tests / "conftest.py")
    manifest = root / "scripts/test_gate/suite_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"tests": [{
        "file": "tests/test_witness.py", "tier": "T1", "maturity": "production",
    }]}))
    (root / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n tier1: fixture\n maturity_production: fixture\n"
    )
    source = root / "cpp/witness.cpp"
    source.parent.mkdir()
    source.write_text("// Timestamp fixture, never compiled.\n")
    os.utime(source, (200, 200))
    for name, stamp in [("current.txt", 300), ("stale.txt", 100)]:
        (root / name).write_text("Stat-only core target, not a library.\n")
        os.utime(root / name, (stamp, stamp))
    shim = root / "shim"
    shim.mkdir()
    (shim / "sitecustomize.py").write_text('''import os, sys, types
from pathlib import Path
class DenyNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'vibeqc' or fullname.startswith('vibeqc.'):
            raise ImportError('Native QC forbidden in protocol fixture: '+fullname)
sys.meta_path.insert(0, DenyNative())
p=types.ModuleType('vibeqc');p.__path__=[]
for name in ['Atom','BasisSet','Molecule','RHFOptions','run_rhf']:
    setattr(p,name,object)
c=types.ModuleType('vibeqc._vibeqc_core');c.__file__=os.environ['CURRENT_STAT_CORE']
p._vibeqc_core=c;sys.modules['vibeqc']=p;sys.modules[c.__name__]=c
o=types.ModuleType('vibeqc.output')
class OutputPlan:
    @classmethod
    def from_run_job_kwargs(cls,**kwargs):return types.SimpleNamespace(files=[])
o.OutputPlan=OutputPlan;sys.modules[o.__name__]=o
''')
    (shim / "protocol_observer.py").write_text('''import json, os, sys
from pathlib import Path
import pytest
case=os.environ['PROTOCOL_CASE'];worker=os.environ.get('PYTEST_XDIST_WORKER','controller')
def record(phase):
    with Path(os.environ['PROTOCOL_EVENTS']).open('a') as f:
        f.write(json.dumps({'worker':worker,'phase':phase})+'\\n')
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if worker=='controller':return
    if case=='worker-crash':os._exit(33)
    if case in {'stale-workers','diagnostic-workers'} or (case=='mixed-workers' and worker=='gw1'):
        sys.modules['vibeqc._vibeqc_core'].__file__=os.environ['STALE_STAT_CORE']
def pytest_collection(session):record('collection')
@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session,exitstatus):
    if worker=='controller':return
    output=session.config.workeroutput
    if case=='late-stale-report':
        output['vibeqc_core_provenance']={'version':1,'state':'stale','report':'deliberate late protocol fault'}
    elif case=='malformed-report':output['vibeqc_core_provenance']={'version':1,'state':[],'report':None}
    elif case=='missing-report':output.pop('vibeqc_core_provenance',None)
    record('finish')
''')
    body = "" if scenario == "current-empty" else '''import os
import pytest
from protocol_observer import record
@pytest.mark.parametrize('item',range(4))
def test_witness(item):
    record('execute')
    assert os.environ['PROTOCOL_CASE'] != 'current-fail'
'''
    (tests / "test_witness.py").write_text(body)
    env = dict(os.environ)
    for key in ["PYTEST_ADDOPTS", "PYTHONHOME", "VIBEQC_ALLOW_STALE_CORE"]:
        env.pop(key, None)
    env.update(
        PYTHONPATH=str(shim), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS="xdist.plugin,protocol_observer", PYTHONDONTWRITEBYTECODE="1",
        CURRENT_STAT_CORE=str(root / "current.txt"), STALE_STAT_CORE=str(root / "stale.txt"),
        PROTOCOL_CASE=scenario, PROTOCOL_EVENTS=str(root / "events.jsonl"),
    )
    if scenario == "diagnostic-workers":
        env["VIBEQC_ALLOW_STALE_CORE"] = "YES"
    proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-q", "-n", "2", "-p", "no:warnings",
         "-p", "no:cacheprovider", *restart_option, "tests/test_witness.py"],
        cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        pytest.fail(f"owned xdist probe timed out: {stdout}\n{stderr}")
    assert proc.returncode == expected, stdout + stderr
    events_file = root / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
    if scenario in {"stale-workers", "mixed-workers"}:
        stale_workers = {"gw0", "gw1"} if scenario == "stale-workers" else {"gw1"}
        assert not any(e["worker"] in stale_workers and e["phase"] in {"collection", "execute"} for e in events)
    if scenario == "late-stale-report":
        assert any(e["phase"] == "execute" for e in events)
    if scenario == "diagnostic-workers":
        assert "results computed on a STALE COMPILED CORE" in stdout
    if scenario in {"malformed-report", "missing-report"}:
        assert "CORE PROVENANCE ERROR" in stdout + stderr
        assert "STALE COMPILED CORE" not in stdout + stderr
