"""Release-validation runtime pin guard."""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.banner import VIBEQC_VERSION, enforce_runtime_pin_from_env
from vibeqc.periodic_runner import run_periodic_job


PIN_ENVS = (
    "VIBEQC_EXPECT_VERSION",
    "VIBEQC_EXPECT_GIT_SHA",
    "VIBEQC_EXPECT_GIT_BRANCH",
)


@pytest.fixture(autouse=True)
def _clear_pin_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PIN_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_runtime_pin_noop_when_unset() -> None:
    """Normal user runs are unaffected unless a release pin is requested."""
    enforce_runtime_pin_from_env()


def test_runtime_pin_accepts_matching_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paper queues can pin the package version exactly."""
    monkeypatch.setenv("VIBEQC_EXPECT_VERSION", VIBEQC_VERSION)

    enforce_runtime_pin_from_env()


def test_runtime_pin_rejects_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release-paper jobs fail before producing values from a drifted build."""
    monkeypatch.setenv("VIBEQC_EXPECT_VERSION", "999.999.999")

    with pytest.raises(RuntimeError, match="runtime pin mismatch"):
        enforce_runtime_pin_from_env()


def test_run_job_dry_run_honors_runtime_pin(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vq preflight path must also reject a mismatched paper pin."""
    monkeypatch.setenv("VIBEQC_EXPECT_VERSION", "999.999.999")
    mol = vq.Molecule([vq.Atom(2, [0.0, 0.0, 0.0])])
    output = tmp_path / "he"

    with pytest.raises(RuntimeError, match="runtime pin mismatch"):
        vq.run_job(mol, basis="sto-3g", method="rhf", output=output, dry_run=True)

    assert not output.with_suffix(".system").exists()


def test_run_periodic_job_dry_run_honors_runtime_pin(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic paper jobs fail before writing a manifest under a bad pin."""
    monkeypatch.setenv("VIBEQC_EXPECT_VERSION", "999.999.999")
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 10.0,
        [vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    output = tmp_path / "he-pbc"

    with pytest.raises(RuntimeError, match="runtime pin mismatch"):
        run_periodic_job(system, basis, method="RHF", output=output, dry_run=True)

    assert not output.with_suffix(".system").exists()
