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

The same switch has to govern the in-process ``import trexio`` that the two
convention modules perform at module level. ``pytest.importorskip`` there
fired before any test reached :func:`reference_python`: with the variable set
and no in-process ``trexio`` both modules were skipped, pytest exited 5, and
the release gate counted that as non-gating ``NOTESTS``, so a lane that was
meant to run the check still went green by omission (#253, generation 2).
:func:`import_trexio` replaces ``importorskip("trexio")`` at every such site;
``tests/test_trexio_reference_gate.py`` pins that no direct call remains.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import NoReturn

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


def import_trexio(
    environ: Mapping[str, str] | None = None,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Import ``trexio`` in-process for a module that cannot run without it.

    Use this instead of ``pytest.importorskip("trexio")``. Without ``trexio``
    the module is skipped and the reason names the gate that did not run;
    with ``VIBEQC_REQUIRE_TREXIO_REFERENCE`` set the module fails at
    collection, which the release gate classifies as ``COLLECT_ERR`` and
    counts as red, not as an all-skipped ``NOTESTS`` run.
    ``environ`` and ``importer`` are injection points for the tests of this
    helper.
    """
    __tracebackhide__ = True  # pytest reports the importing module, as importorskip does
    environ = os.environ if environ is None else environ
    try:
        return importer("trexio")
    except ImportError as exc:
        error = exc
    # Outside the ``except`` block, so the outcome carries no chained traceback.
    _refuse(
        reference_required(environ),
        f"{GATE} did not run: `trexio` cannot be imported in-process "
        f"({error}). The tests need it to write the file the reference "
        "interpreter then reads",
        remedy="Install the [trexio] extra",
        missing="in-process trexio",
    )


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


def _refuse(
    required: bool,
    reason: str,
    remedy: str = f"Point {REFERENCE_VAR} at an interpreter with trexio and pyscf",
    missing: str = "reference interpreter",
) -> NoReturn:
    """Stop the test, or the module being collected, as a failure or a skip.

    ``allow_module_level`` lets the same call serve the module-level import in
    :func:`import_trexio`; inside a test body it changes nothing.
    """
    __tracebackhide__ = True
    if required:
        pytest.fail(
            f"{reason}. {REQUIRE_VAR} is set, so a missing {missing} is a "
            "failure, not a skip.",
            pytrace=False,
        )
    pytest.skip(
        f"{reason}. {remedy}; set {REQUIRE_VAR}=1 to fail instead of skipping.",
        allow_module_level=True,
    )
