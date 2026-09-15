"""Tests for the GPW restart-state file
(:mod:`vibeqc.periodic_gapw_restart`).

The restart file is vibe-qc's analogue of GPAW's ``.gpw``: a
single ``.npz`` archive that round-trips a converged
:class:`GpwScfResult` or :class:`GpwMultiKScfResult` to bit-
faithful precision and rebuilds the system / basis / grid on
load.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
)
from vibeqc.periodic_gapw_j import (
    run_periodic_rhf_gpw,
    run_periodic_rks_gpw_multi_k,
)
from vibeqc.periodic_gapw_restart import (
    describe_gpw_result,
    load_gpw_result,
    save_gpw_result,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Fixtures ------------------------------------------------------

def _h2_periodic_system(L: float = 12.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [
        core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
        core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
    ]
    return sys


def _h2_basis(L: float = 12.0):
    mol = vq.Molecule(
        [vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
         vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2])],
        charge=0, multiplicity=1,
    )
    return vq.BasisSet(mol, "sto-3g")


def _run_h2_rhf(L: float = 12.0, n: int = 32):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system = _h2_periodic_system(L)
        basis = _h2_basis(L)
        grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
        result = run_periodic_rhf_gpw(
            system, basis, grid=grid, quiet=True,
        )
    return system, basis, result


def _run_h2_rks_lda(L: float = 12.0, n: int = 32):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system = _h2_periodic_system(L)
        basis = _h2_basis(L)
        grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
        result = run_periodic_rhf_gpw(
            system, basis, grid=grid, functional="lda", quiet=True,
        )
    return system, basis, result


def _run_h2_multi_k_lda(L: float = 12.0, n: int = 32):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system = _h2_periodic_system(L)
        basis = _h2_basis(L)
        grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
        kmesh = core.monkhorst_pack(system, [2, 2, 2])
        result = run_periodic_rks_gpw_multi_k(
            system, basis, kmesh,
            functional="lda", grid=grid, quiet=True,
        )
    return system, basis, result


# ---------- Tests ---------------------------------------------------------

def test_roundtrip_rhf_gpw_density_machine_precision(tmp_path: Path):
    """A converged Γ-only RHF GPW result round-trips through the
    restart file. The reloaded energy and density matrix match the
    original to machine precision."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system, basis, result = _run_h2_rhf()
        assert result.converged

        path = tmp_path / "h2_rhf.npz"
        out = save_gpw_result(path, result, basis, system)
        assert out.exists() and out.stat().st_size > 0

        loaded = load_gpw_result(out)

    assert loaded["kind"] == "gpw_scf"
    assert loaded["converged"] is True
    assert loaded["energy"] == pytest.approx(result.energy, abs=0.0, rel=0.0)
    np.testing.assert_array_equal(
        loaded["density"], np.asarray(result.density)
    )
    np.testing.assert_array_equal(
        loaded["mo_coeffs"], np.asarray(result.mo_coeffs)
    )
    np.testing.assert_array_equal(
        loaded["mo_energies"], np.asarray(result.mo_energies)
    )
    # System / basis / grid round-trip into usable objects.
    assert loaded["system"].dim == 3
    assert len(list(loaded["system"].unit_cell)) == 2
    assert loaded["basis"].name == "sto-3g"
    assert loaded["basis"].nbasis == basis.nbasis
    assert loaded["grid"].shape == result.grid.shape
    np.testing.assert_allclose(
        loaded["grid"].lattice_bohr, result.grid.lattice_bohr,
    )


def test_roundtrip_multi_k_gpw_lda(tmp_path: Path):
    """A 2×2×2 multi-k LDA GPW result round-trips: per-k MOs,
    energies, k-points + weights all survive."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system, basis, result = _run_h2_multi_k_lda()
        assert result.converged
        assert len(result.mo_coeffs_k) == 8

        path = tmp_path / "h2_multi_k_lda.npz"
        save_gpw_result(path, result, basis, system)
        loaded = load_gpw_result(path)

    assert loaded["kind"] == "gpw_multi_k_scf"
    assert loaded["energy"] == pytest.approx(result.energy, abs=0.0, rel=0.0)
    assert loaded["kpoints"].shape == (8, 3)
    assert loaded["kweights"].shape == (8,)
    np.testing.assert_allclose(loaded["kweights"].sum(), 1.0)
    # Per-k MO coefficients round-trip exactly.
    mo_k_ref = np.asarray(
        [np.asarray(c) for c in result.mo_coeffs_k], dtype=complex,
    )
    np.testing.assert_array_equal(loaded["mo_coeffs_k"], mo_k_ref)
    mo_e_ref = np.asarray(
        [np.asarray(e) for e in result.mo_energies_k], dtype=float,
    )
    np.testing.assert_array_equal(loaded["mo_energies_k"], mo_e_ref)
    # Density round-trips.
    np.testing.assert_array_equal(
        loaded["density"], np.asarray(result.density)
    )


def test_roundtrip_rks_lda_exc_preserved(tmp_path: Path):
    """An RKS LDA GPW result round-trips its e_xc term — the XC
    energy comes back from the .npz unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system, basis, result = _run_h2_rks_lda()
        assert result.converged
        # LDA exchange-correlation must be strictly negative for a
        # bonded H2 — sanity-check before round-tripping.
        assert result.breakdown.e_xc < 0.0
        assert result.breakdown.functional == "lda"

        path = tmp_path / "h2_rks_lda.npz"
        save_gpw_result(path, result, basis, system)
        loaded = load_gpw_result(path)

    assert loaded["breakdown_e_xc"] == pytest.approx(
        result.breakdown.e_xc, abs=0.0, rel=0.0
    )
    assert loaded["breakdown_functional"] == "lda"
    # All five remaining breakdown terms also round-trip.
    assert loaded["breakdown_e_kinetic"] == pytest.approx(
        result.breakdown.e_kinetic, abs=0.0, rel=0.0
    )
    assert loaded["breakdown_e_nuclear_attraction"] == pytest.approx(
        result.breakdown.e_nuclear_attraction, abs=0.0, rel=0.0
    )
    assert loaded["breakdown_e_hartree"] == pytest.approx(
        result.breakdown.e_hartree, abs=0.0, rel=0.0
    )
    assert loaded["breakdown_e_hf_exchange"] == pytest.approx(
        result.breakdown.e_hf_exchange, abs=0.0, rel=0.0
    )
    assert loaded["breakdown_e_nuclear_repulsion"] == pytest.approx(
        result.breakdown.e_nuclear_repulsion, abs=0.0, rel=0.0
    )


def test_describe_gpw_result_mentions_gpw_and_energy(tmp_path: Path):
    """The describe helper returns a human-readable string with the
    word 'GPW' and the energy printed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system, basis, result = _run_h2_rhf()
        path = tmp_path / "h2_describe.npz"
        save_gpw_result(path, result, basis, system)
        summary = describe_gpw_result(path)

    assert isinstance(summary, str)
    assert "GPW" in summary
    # The energy formatted to ~10 dp must be in the string.
    assert f"{result.energy:.10f}" in summary


def test_load_gpw_result_rejects_missing_path(tmp_path: Path):
    """A non-existent path must raise FileNotFoundError, not a
    generic OSError or NumPy ValueError."""
    bogus = tmp_path / "does_not_exist.npz"
    assert not bogus.exists()
    with pytest.raises(FileNotFoundError):
        load_gpw_result(bogus)
