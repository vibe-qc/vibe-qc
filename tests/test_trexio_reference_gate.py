"""The TREXIO reference-interpreter gate is loud when told to be (#253).

These tests need neither ``trexio`` nor ``pyscf``: they drive
:func:`tests.trexio_reference.reference_python` through its injection points
and pin the two behaviours the issue asked for. Without a reference
interpreter the gate skips and its reason names the check that did not run;
with ``VIBEQC_REQUIRE_TREXIO_REFERENCE`` set the same condition fails, so a
CI lane meant to run the convention check cannot pass by omission.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.trexio_reference import (
    GATE,
    PROBE,
    REFERENCE_VAR,
    REQUIRE_VAR,
    import_trexio,
    reference_python,
    reference_required,
)


def _completed(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_unset_reference_skips_and_names_the_gate():
    with pytest.raises(pytest.skip.Exception) as info:
        reference_python(environ={})
    message = str(info.value)
    assert GATE in message
    assert "did not run" in message
    assert REFERENCE_VAR in message and REQUIRE_VAR in message


def test_unset_reference_fails_when_required():
    with pytest.raises(pytest.fail.Exception) as info:
        reference_python(environ={REQUIRE_VAR: "1"})
    message = str(info.value)
    assert GATE in message
    assert "failure, not a skip" in message


def test_unusable_interpreter_skips_with_the_probe_detail():
    calls: list[list[str]] = []

    def run(argv, **_):
        calls.append(list(argv))
        return _completed(1, "ModuleNotFoundError: No module named 'trexio'")

    with pytest.raises(pytest.skip.Exception) as info:
        reference_python(environ={REFERENCE_VAR: "/opt/py/bin/python"}, run=run)
    assert calls == [["/opt/py/bin/python", "-c", PROBE]]
    assert "No module named 'trexio'" in str(info.value)


def test_unusable_interpreter_fails_when_required():
    def run(argv, **_):
        return _completed(1, "ModuleNotFoundError: No module named 'pyscf'")

    with pytest.raises(pytest.fail.Exception) as info:
        reference_python(
            environ={REFERENCE_VAR: "/opt/py/bin/python", REQUIRE_VAR: "yes"}, run=run
        )
    assert "No module named 'pyscf'" in str(info.value)


def test_usable_interpreter_is_returned_unchanged():
    def run(argv, **_):
        assert argv == ["/opt/py/bin/python", "-c", PROBE]
        return _completed(0)

    exe = reference_python(
        environ={REFERENCE_VAR: " /opt/py/bin/python ", REQUIRE_VAR: "1"}, run=run
    )
    assert exe == "/opt/py/bin/python"


def test_real_probe_against_this_interpreter_matches_its_imports(tmp_path):
    """The default ``run`` really probes the named interpreter.

    A shell script that exits non-zero stands in for an interpreter without
    the packages; ``sys.executable`` is then classified by whether it can
    actually import both modules, so this passes on machines with and
    without a reference environment.
    """
    bad = tmp_path / "python-without-trexio"
    bad.write_text("#!/bin/sh\necho 'no trexio here' >&2\nexit 3\n")
    bad.chmod(0o755)
    with pytest.raises(pytest.skip.Exception) as info:
        reference_python(environ={REFERENCE_VAR: str(bad)})
    assert "no trexio here" in str(info.value)

    has_both = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True
    ).returncode == 0
    environ = {REFERENCE_VAR: sys.executable}
    if has_both:
        assert reference_python(environ=environ) == sys.executable
    else:
        with pytest.raises(pytest.skip.Exception):
            reference_python(environ=environ)


@pytest.mark.parametrize("value", ["", "0", "false", "No", " off "])
def test_falsy_require_spellings_do_not_require(value):
    assert not reference_required({REQUIRE_VAR: value})


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
def test_truthy_require_spellings_require(value):
    assert reference_required({REQUIRE_VAR: value})


def test_process_environment_is_the_default_source(monkeypatch):
    monkeypatch.delenv(REFERENCE_VAR, raising=False)
    monkeypatch.delenv(REQUIRE_VAR, raising=False)
    with pytest.raises(pytest.skip.Exception):
        reference_python()
    monkeypatch.setenv(REQUIRE_VAR, "1")
    with pytest.raises(pytest.fail.Exception):
        reference_python()
    assert os.environ[REQUIRE_VAR] == "1"


# ---------------------------------------------------------------------- #
# The module-level trexio import must honour the switch too (#253)     #
# ---------------------------------------------------------------------- #
#
# The two convention-gate modules import ``trexio`` in-process at module
# level. That import used to be ``pytest.importorskip("trexio")``, which fired
# before any test reached ``reference_python()``: with the variable set and no
# in-process ``trexio`` both modules were skipped, pytest exited 5 and the
# release gate counted that as non-gating NOTESTS, so "required" could still
# go green by omission. The tests below run the two real modules in a child
# pytest with ``trexio`` masked, so they pass on machines with and without it.

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_DIR = REPO_ROOT / "scripts" / "test_gate"
GATED_MODULES = (
    "tests/test_output_trexio.py",
    "tests/test_output_trexio_extended.py",
)
MASK_PLUGIN = "vq_mask_trexio"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_gated_modules_without_trexio(
    tmp_path, *, required: bool, trexio_present: bool = False, collect_only: bool = False
):
    """Run both gated modules in a child pytest in which ``import trexio`` fails.

    ``trexio_present`` swaps the mask for an empty stand-in module, and
    ``collect_only`` stops after collection: together they show that a
    successful import reaches both modules.
    """
    stand_in = "types.ModuleType('trexio')" if trexio_present else "None"
    (tmp_path / f"{MASK_PLUGIN}.py").write_text(
        "import sys, types\n"
        f"sys.modules['trexio'] = {stand_in}  # None: import raises ModuleNotFoundError\n"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {REFERENCE_VAR, REQUIRE_VAR, "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
    }
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(tmp_path), env.get("PYTHONPATH", "")) if part
    )
    if required:
        env[REQUIRE_VAR] = "1"
    return subprocess.run(
        [
            sys.executable, "-m", "pytest", "-p", MASK_PLUGIN, "-p", "no:cacheprovider",
            "-q", "-rs", "--rootdir", str(REPO_ROOT),
            *(["--collect-only"] if collect_only else []), *GATED_MODULES,
        ],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600,
    )


def test_gated_modules_skip_when_trexio_is_absent_and_not_required(tmp_path):
    """Developer machines keep the skip: an all-skipped run, pytest exit status 5."""
    proc = _run_gated_modules_without_trexio(tmp_path, required=False)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 5, out
    assert "2 skipped" in out, out


def test_module_level_skip_names_the_gate_that_did_not_run(tmp_path):
    """A skip at the module-level import says which check is missing, like the locator."""
    proc = _run_gated_modules_without_trexio(tmp_path, required=False)
    out = proc.stdout + proc.stderr
    for module in GATED_MODULES:
        line = next((l for l in out.splitlines() if l.startswith("SKIPPED") and module in l), "")
        assert GATE in line and "did not run" in line, out
        assert REQUIRE_VAR in line, out


def test_gated_modules_fail_when_required_and_trexio_is_absent(tmp_path):
    """Required and absent must end in a status the release gate counts as red.

    ``classify`` and ``is_failing`` are the gate's own functions: an all-skipped
    run is exit status 5, ``NOTESTS``, which ``gate_verdict`` does not gate on.
    """
    run_full_suite = _load("run_full_suite_for_trexio_gate", GATE_DIR / "run_full_suite.py")
    gate_verdict = _load("gate_verdict_for_trexio_gate", GATE_DIR / "gate_verdict.py")
    proc = _run_gated_modules_without_trexio(tmp_path, required=True)
    out = proc.stdout + proc.stderr
    status = run_full_suite.classify(proc.returncode, False, {})
    assert gate_verdict.is_failing(status), (
        f"a required gate that cannot import trexio ended rc={proc.returncode} "
        f"({status}), which the release gate does not count as red:\n{out}"
    )
    assert "2 skipped" not in out and "passed" not in out, out
    for module in GATED_MODULES:
        assert module in out, out
    assert GATE in out and "failure, not a skip" in out, out


def _absent(name):
    raise ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_import_trexio_returns_the_module_when_it_imports():
    sentinel = object()
    assert import_trexio(environ={REQUIRE_VAR: "1"}, importer=lambda name: sentinel) is sentinel


def test_import_trexio_skips_naming_the_gate_when_absent():
    with pytest.raises(pytest.skip.Exception) as info:
        import_trexio(environ={}, importer=_absent)
    message = str(info.value)
    assert GATE in message and "did not run" in message
    assert "No module named 'trexio'" in message
    assert REQUIRE_VAR in message


def test_import_trexio_fails_when_required_and_absent():
    with pytest.raises(pytest.fail.Exception) as info:
        import_trexio(environ={REQUIRE_VAR: "1"}, importer=_absent)
    message = str(info.value)
    assert GATE in message and "failure, not a skip" in message
    assert "No module named 'trexio'" in message
    assert info.value.__context__ is None  # no chained ModuleNotFoundError noise


def test_import_trexio_treats_a_broken_install_like_an_absent_one():
    def broken(name):
        raise ImportError("libhdf5.so: cannot open shared object file")

    with pytest.raises(pytest.skip.Exception):
        import_trexio(environ={}, importer=broken)
    with pytest.raises(pytest.fail.Exception) as info:
        import_trexio(environ={REQUIRE_VAR: "true"}, importer=broken)
    assert "libhdf5" in str(info.value)


def _importorskip_trexio_calls(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        target = node.args[0]
        if (
            name == "importorskip"
            and isinstance(target, ast.Constant)
            and isinstance(target.value, str)
            and target.value.split(".")[0] == "trexio"
        ):
            yield node.lineno


def test_no_test_module_skips_on_trexio_behind_the_switchs_back():
    """Every in-process trexio import in tests goes through ``import_trexio``.

    A direct ``pytest.importorskip("trexio")`` skips before the switch is
    consulted, which is the defect this generation fixed at two sites.
    """
    offenders = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in sorted((REPO_ROOT / "tests").rglob("*.py"))
        for line in _importorskip_trexio_calls(path)
    ]
    assert not offenders, (
        "use tests.trexio_reference.import_trexio() instead of "
        f"pytest.importorskip('trexio') at: {offenders}"
    )


@pytest.mark.parametrize("required", [False, True])
def test_gated_modules_collect_when_trexio_imports(tmp_path, required):
    """The helper hands the module to both sites; the switch only matters when it is absent."""
    proc = _run_gated_modules_without_trexio(
        tmp_path, required=required, trexio_present=True, collect_only=True
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    for module in GATED_MODULES:
        assert f"{module}::" in out, out
