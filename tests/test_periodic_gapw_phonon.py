"""Smoke tests for GAPW finite-displacement phonons."""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_gapw_phonon import (
    PhononBandStructure,
    PhononCalculator,
    compute_dynamical_matrix_fd,
    phonon_dos,
    phonon_eigenvalues,
)


def test_phonon_imports():
    assert callable(compute_dynamical_matrix_fd)
    assert callable(phonon_eigenvalues)
    assert callable(phonon_dos)


def test_phonon_calculator_constructible():
    assert PhononCalculator is not None


def test_phonon_rejects_analytic_one_centre_before_central_scf(monkeypatch):
    """The block-only derivative must reject analytic HF before any SCF."""
    import vibeqc.periodic_gapw_phonon as gapw_phonon

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(8, [4.0, 4.0, 4.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def _unexpected_scf(*_args, **_kwargs):
        raise AssertionError("analytic phonon request entered the GAPW SCF")

    monkeypatch.setattr(gapw_phonon, "run_periodic_rhf_gapw", _unexpected_scf)

    with pytest.raises(NotImplementedError, match=r"require.*one_centre='block'"):
        compute_dynamical_matrix_fd(
            system,
            basis,
            "sto-3g",
            gapw_kwargs={"one_centre": "analytic"},
        )


def test_hf_phonon_requires_explicit_block_before_central_scf(monkeypatch):
    """An omitted HF mode must not silently restore the wrong block default."""
    import vibeqc.periodic_gapw_phonon as gapw_phonon

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(8, [4.0, 4.0, 4.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    def _unexpected_scf(*_args, **_kwargs):
        raise AssertionError("undeclared HF phonon request entered the GAPW SCF")

    monkeypatch.setattr(gapw_phonon, "run_periodic_rhf_gapw", _unexpected_scf)

    with pytest.raises(NotImplementedError, match="HF phonons require explicit"):
        compute_dynamical_matrix_fd(system, basis, "sto-3g")


def _seeded_phonon_calculator(dynamical_matrix: np.ndarray) -> PhononCalculator:
    system = vq.PeriodicSystem(3, np.eye(3) * 8.0, [vq.Atom(1, [4.0, 4.0, 4.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    calc = PhononCalculator(system, basis, "sto-3g", functional="lda")
    calc._dynamical_matrix = np.asarray(dynamical_matrix, dtype=np.float64)
    return calc


def test_phonon_bandstructure_repeats_cached_gamma_modes():
    dynamical_matrix = np.diag([1.0e-8, 4.0e-8, 9.0e-8])
    calc = _seeded_phonon_calculator(dynamical_matrix)
    qpath = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )

    bands = calc.get_bandstructure(qpath, include_eigenvectors=True)
    expected_cm1, expected_thz, expected_modes = phonon_eigenvalues(
        dynamical_matrix,
        masses_amu=calc._masses_amu,
    )

    assert isinstance(bands, PhononBandStructure)
    assert bands.interpolation == "gamma_flat"
    assert bands.metadata["source"] == "gamma_dynamical_matrix"
    assert bands.n_qpoints == 3
    assert bands.n_modes == 3
    np.testing.assert_allclose(bands.qpoints, qpath)
    np.testing.assert_allclose(
        bands.frequencies_cm1,
        np.tile(expected_cm1, (qpath.shape[0], 1)),
    )
    np.testing.assert_allclose(
        bands.frequencies_thz,
        np.tile(expected_thz, (qpath.shape[0], 1)),
    )
    assert bands.eigenvectors is not None
    np.testing.assert_allclose(
        bands.eigenvectors,
        np.repeat(expected_modes[np.newaxis, :, :], qpath.shape[0], axis=0),
    )


def test_phonon_bandstructure_validates_path_and_cache():
    calc = _seeded_phonon_calculator(np.eye(3))

    with pytest.raises(ValueError, match=r"shape \(n_qpoints, 3\)"):
        calc.get_bandstructure([[0.0, 0.0]])

    with pytest.raises(ValueError, match="at least one q-point"):
        calc.get_bandstructure(np.empty((0, 3)))

    with pytest.raises(ValueError, match="finite"):
        calc.get_bandstructure([[0.0, np.nan, 0.0]])

    empty_calc = _seeded_phonon_calculator(np.eye(3))
    empty_calc._dynamical_matrix = None
    with pytest.raises(RuntimeError, match="compute_phonons"):
        empty_calc.get_bandstructure([[0.0, 0.0, 0.0]])
