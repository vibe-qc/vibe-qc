"""The TREXIO reference-interpreter gate is loud when told to be (#253).

These tests need neither ``trexio`` nor ``pyscf``: they drive
:func:`tests.trexio_reference.reference_python` through its injection points
and pin the two behaviours the issue asked for. Without a reference
interpreter the gate skips and its reason names the check that did not run;
with ``VIBEQC_REQUIRE_TREXIO_REFERENCE`` set the same condition fails, so a
CI lane meant to run the convention check cannot pass by omission.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.trexio_reference import (
    GATE,
    PROBE,
    REFERENCE_VAR,
    REQUIRE_VAR,
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
