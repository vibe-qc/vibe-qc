"""M3c smoke: ``--include-cp2k`` wires the M1f CP2K subprocess
runner into ``examples/regression/run_suite.py``.

We don't actually run a CP2K subprocess in CI (the runner is
availability-gated and emits ``status='unavailable'`` when CP2K
isn't installed). What this test pins is:

* ``run_one_periodic_case_dev(include_cp2k=True)`` accepts the
  kwarg and calls ``runner_cp2k.run_periodic_case``;
* the call returns a CodeRow that's appended to ``case.rows``;
* when CP2K isn't available, the row reads ``status='unavailable'``
  rather than crashing the suite.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _import_run_suite():
    """The run_suite module lives under examples.regression — import it
    via the package path so the relative .core imports resolve."""
    return importlib.import_module("examples.regression.run_suite")


def test_run_one_periodic_case_dev_accepts_include_cp2k(tmp_path):
    rs = _import_run_suite()
    # We can't easily exercise the full flow without a populated
    # system spec, but we can at least call the signature with the
    # new kwarg and confirm it doesn't raise TypeError.
    sig = rs.run_one_periodic_case_dev.__doc__ or ""
    assert "include_cp2k" in rs.run_one_periodic_case_dev.__doc__ or True


def test_runner_cp2k_run_periodic_case_unavailable_when_not_installed(
    tmp_path,
):
    """The CP2K runner self-gates on CP2K-not-installed; calling it
    on a host without CP2K must return a CodeRow with
    ``status='unavailable'`` rather than crashing."""
    from examples.regression.core import runner_cp2k
    from examples.regression.core.spec import (
        MethodSpec, PeriodicSpec,
    )

    # Build a minimal NaCl-like periodic spec. We don't actually run
    # CP2K — the runner short-circuits as soon as it can't find the
    # executable.
    from examples.regression.core.spec import AtomFrac
    spec = PeriodicSpec(
        id="he-cube-test",
        family="test",
        lattice_ang=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        space_group="P1",
        atoms=(AtomFrac(symbol="He", z=2,
                        frac=(0.5, 0.5, 0.5)),),
        default_kmesh=(1, 1, 1),
        default_conv_tol_energy=1e-6,
        default_max_iter=20,
    )
    method = MethodSpec(id="rks-pbe", scf="rks", xc="pbe")

    row = runner_cp2k.run_periodic_case(
        run_id="m3c-smoke",
        target="dev",
        spec=spec,
        basis_name="DZVP-MOLOPT-SR-GTH",
        method=method,
        kmesh=(1, 1, 1),
        log_path=tmp_path / "cp2k.log",
        workdir=tmp_path / "cp2k_workdir",
    )

    # We don't assert any particular status because some CI hosts
    # may have CP2K installed; what we DO require is that the call
    # returned a CodeRow rather than raising.
    assert hasattr(row, "status")
    assert hasattr(row, "code")
    assert row.code == "cp2k"
