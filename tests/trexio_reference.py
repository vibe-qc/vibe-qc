"""Locate the out-of-process TREXIO reference interpreter, loudly on demand.

The TREXIO AO-ordering and normalization convention is pinned by exactly two
kinds of test: the spherical-permutation check against the specification's
worked example, and the out-of-process PySCF rebuild of the SCF energy from
the stored MOs (``examples/regression/runner_trexio_pyscf.py``). The
in-process round trips cannot see a *consistent* permutation or
normalization error, because writer and reader share it and it cancels.

The PySCF rebuild needs an interpreter with ``trexio`` and ``pyscf``, named
by ``VIBEQC_TREXIO_PYTHON``. Without one the tests used to skip with a quiet
one-line reason, so a suite could report "20 passed, 2 skipped" while the
only checks able to catch a convention regression had not run (#253).

:func:`reference_python` keeps the skip for developer machines but names the
gate that did not run, and turns the skip into a failure when
``VIBEQC_REQUIRE_TREXIO_REFERENCE`` is set, which is what a release gate
should set once its runner carries a reference interpreter.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping

import pytest

REFERENCE_VAR = "VIBEQC_TREXIO_PYTHON"
REQUIRE_VAR = "VIBEQC_REQUIRE_TREXIO_REFERENCE"
PROBE = "import trexio, pyscf"
GATE = "TREXIO AO-convention gate (out-of-process PySCF rebuild)"

_FALSE = frozenset({"", "0", "false", "no", "off"})


def reference_required(environ: Mapping[str, str] | None = None) -> bool:
    """True when ``VIBEQC_REQUIRE_TREXIO_REFERENCE`` is set to a truthy value."""
    environ = os.environ if environ is None else environ
    return environ.get(REQUIRE_VAR, "").strip().lower() not in _FALSE


def reference_python(
    environ: Mapping[str, str] | None = None,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Return the reference interpreter, or stop the test loudly.

    Without a usable interpreter the test is skipped, and the skip reason says
    which gate did not run and how to make that a failure. With
    ``VIBEQC_REQUIRE_TREXIO_REFERENCE`` set the same condition is a failure,
    so a gate that was meant to run the check cannot go green by omission.
    ``environ`` and ``run`` are injection points for the tests of this helper.
    """
    environ = os.environ if environ is None else environ
    required = reference_required(environ)
    exe = environ.get(REFERENCE_VAR, "").strip()
    if not exe:
        _refuse(
            required,
            f"{GATE} did not run: {REFERENCE_VAR} is unset. The in-process "
            "round trips cannot detect a consistent AO permutation or "
            "normalization error; only this check pins the convention",
        )
    probe = run([exe, "-c", PROBE], capture_output=True, text=True)
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit status {probe.returncode}"
        _refuse(
            required,
            f"{GATE} did not run: {REFERENCE_VAR}={exe!r} cannot "
            f"`{PROBE}` ({tail})",
        )
    return exe


def _refuse(required: bool, reason: str) -> None:
    if required:
        pytest.fail(
            f"{reason}. {REQUIRE_VAR} is set, so a missing reference "
            "interpreter is a failure, not a skip.",
            pytrace=False,
        )
    pytest.skip(
        f"{reason}. Point {REFERENCE_VAR} at an interpreter with trexio and "
        f"pyscf; set {REQUIRE_VAR}=1 to fail instead of skipping."
    )
