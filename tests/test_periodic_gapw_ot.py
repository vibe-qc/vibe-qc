"""Smoke tests for GAPW Orbital Transformation SCF solver."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc.periodic_gapw_ot import (
    OrbitalTransformation,
    run_ot_rhf_gapw,
    run_ot_rks_gapw,
)


def test_ot_classes_importable():
    assert callable(run_ot_rhf_gapw)
    assert callable(run_ot_rks_gapw)


def test_orbital_transformation_constructs():
    ot = OrbitalTransformation.__new__(OrbitalTransformation)
    assert ot is not None


def _stub_ot() -> OrbitalTransformation:
    """Return a two-step OT object whose run path needs no periodic kernels."""
    ot = OrbitalTransformation.__new__(OrbitalTransformation)
    ot.n_basis = 1
    ot.max_iter = 2
    ot.conv_tol = 1.0e-8
    ot._S_half_inv = np.eye(1)
    steps = iter(((-1.0, 1.0e-2), (-1.1, 5.0e-9)))

    def _energy_and_grad(_C):
        energy, gradient = next(steps)
        return (
            energy,
            np.asarray([gradient]),
            np.zeros((1, 1)),
            0.0,
        )

    ot._energy_and_grad = _energy_and_grad
    ot._orthonormal_mo_set = lambda C: C
    ot._step_cg = lambda C, grad, _direction, _previous, _iteration: (
        C,
        np.zeros_like(grad),
        grad,
    )
    ot._ao_from_orthonormal_mo = lambda C: C
    ot._density_from_orthonormal = lambda _C: np.asarray([[2.0]])
    ot._compute_fock_and_energy = lambda _D: (0.0, np.asarray([[-1.1]]), 0.0)
    return ot


def test_ot_progress_uses_progress_logger_and_preserves_metadata(capsys):
    _C, _D, _eps, info = _stub_ot().run(C_init=np.eye(1), verbose=True)

    rendered = capsys.readouterr().out
    assert "OT iter   1:  E = -1.000000000000  |g|_rms = 1.00e-02" in rendered
    assert "OT iter   2:  E = -1.100000000000  |g|_rms = 5.00e-09" in rendered
    assert "  -> Converged in 2 iterations." in rendered
    assert info["converged"] is True
    assert info["n_iter"] == 2
    assert info["grad_norm"] == pytest.approx(5.0e-9)


def test_ot_quiet_suppresses_progress(capsys):
    _stub_ot().run(C_init=np.eye(1), verbose=False)
    assert capsys.readouterr().out == ""
