"""Pre-cut sentinels for ``run_periodic_job``'s output lifecycle (GitLab #429, #440).

The public periodic runner finishes or crashes its job-scoped writer at the
API boundary through ``_periodic_output_lifecycle``. v0.17.11 shipped with
that decorator displaced onto a helper inserted between it and the runner
(a6bba3d): a finished job stayed ``status = "running"``, a writer failure no
longer raised ``IncompleteOutputError``, a capability refusal before the
output plan left no receipt, and a delivery failure dropped the recorded
driver verdict. The pins that catch each of those sit in advisory T2 files,
so nothing blocking noticed.

Every node here is a thin delegate to the owning pin in
``tests/test_output_errors.py``, ``tests/test_output_manifest.py`` and
``tests/test_bz_integration_gilat_scf.py`` (one source of truth, no copied
assertions), chosen because each takes seconds: together they are the
smallest set that fails whenever ``run_periodic_job`` runs without its
lifecycle. The file is pinned to the blocking T1 tier in
``scripts/test_gate/update_suite_manifest.py`` (``CURATED``), see
``scripts/test_gate/README.md``.
"""

from __future__ import annotations

import pytest

from tests.test_bz_integration_gilat_scf import (
    test_runner_gilat_bipole_rks_path_runs as _finished_job_writes_complete,
)
from tests.test_output_errors import (
    test_periodic_scf_refuses_complete_status_after_writer_failure
    as _writer_failure_refuses_complete,
)
from tests.test_output_manifest import (
    test_bipole_nonconvergence_is_recorded_in_manifest
    as _nonconvergence_is_recorded,
)
from tests.test_output_manifest import (
    test_periodic_delivery_error_keeps_driver_convergence
    as _delivery_error_keeps_verdict,
)
from tests.test_output_manifest import (
    test_periodic_preflight_refusal_writes_manifest
    as _preflight_refusal_leaves_receipt,
)


def test_writer_failure_refuses_complete_status(tmp_path, monkeypatch):
    """A failed periodic writer raises IncompleteOutputError; no complete status."""
    _writer_failure_refuses_complete(tmp_path, monkeypatch)


def test_finished_periodic_job_writes_complete_status(tmp_path, monkeypatch):
    """A finished BIPOLE RKS run ends with ``[outputs].status = "complete"``."""
    _finished_job_writes_complete(tmp_path, monkeypatch)


def test_preflight_refusal_leaves_its_receipt(tmp_path):
    """A capability refusal before the plan leaves the ``.system`` receipt."""
    _preflight_refusal_leaves_receipt(tmp_path)


@pytest.mark.parametrize("error_type", [OSError, NotImplementedError])
def test_delivery_error_keeps_the_driver_verdict(tmp_path, monkeypatch, error_type):
    """A delivery failure after the SCF keeps ``result_status`` and crashes."""
    _delivery_error_keeps_verdict(tmp_path, monkeypatch, error_type)


def test_nonconvergence_is_recorded_before_the_crash(tmp_path):
    """A non-converged periodic SCF records ``not_converged`` with its reason."""
    _nonconvergence_is_recorded(tmp_path)
