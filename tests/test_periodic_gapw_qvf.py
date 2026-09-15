"""Tests for ``periodic_gapw_qvf.qvf_density_data_periodic``.

The GPW SCF carries its own FFT grid; the QVF emitter should sample
the converged density on *that* grid (not a fresh bounding-box grid
like the molecular path does).  These tests pin:

* dict-shape contract for ``write_qvf(volume_data=...)`` (one entry,
  3-tuple of ``(data, origin, span)``),
* end-to-end round-trip — pair with ``write_qvf`` and confirm the
  resulting ``.qvf`` archive carries a ``volume.density`` section,
* the 3D-only guard (1D / 2D systems raise ``ValueError``).

This is the periodic counterpart of the molecular helper in
``vibeqc.output.formats.qvf.qvf_density_data``; the dict shape it
returns matches what the molecular helper returns so the same writer
consumes both.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc.periodic_gapw_grid import (
    GAPWExperimentalWarning,
    PlaneWaveGrid,
)
from vibeqc.periodic_gapw_j import GpwScfResult, GpwEnergyBreakdown
from vibeqc.periodic_gapw_qvf import qvf_density_data_periodic


# The GPW route emits an experimental warning at builder construction;
# silence it across this module — we're testing the QVF emitter, not the
# warning surface.
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Fixtures ----------------------------------------------------


def _he_periodic_system(L: float = 16.0):
    sys = core.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3) * L
    sys.unit_cell = [core.Atom(2, [L / 2, L / 2, L / 2])]
    return sys


def _he_result_and_basis(L: float = 16.0, n: int = 24):
    """Build a minimal converged-looking GpwScfResult for He STO-3G.

    The QVF emitter only consumes ``result.density``, ``result.grid``,
    and (via the BasisSet) the AO definitions; we don't need the SCF
    to have actually run for this unit test.
    """
    mol = vq.Molecule(
        [vq.Atom(2, [L / 2, L / 2, L / 2])], charge=0, multiplicity=1,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    # He STO-3G: closed-shell 1s² → D = 2 · |1s><1s|.
    D = np.array([[2.0]])
    grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
    result = GpwScfResult(
        energy=-2.807784,
        breakdown=GpwEnergyBreakdown(
            e_kinetic=0.0,
            e_nuclear_attraction=0.0,
            e_hartree=0.0,
            e_hf_exchange=0.0,
            e_nuclear_repulsion=0.0,
            e_total=-2.807784,
            grid=grid,
        ),
        density=D,
        mo_coeffs=np.array([[1.0]]),
        mo_energies=np.array([-0.876]),
        converged=True,
        n_iter=1,
        grid=grid,
    )
    return result, basis, _he_periodic_system(L), grid


# ---------- Test 1: dict-shape smoke ------------------------------------


def test_qvf_density_data_periodic_returns_writer_compatible_dict():
    """Smoke: result → dict with one entry, 3-tuple value of
    ``(data, origin, span)``; data is float32 ``(nx, ny, nz)`` shaped
    like the grid; origin is 3-vector; span is 3×3.
    """
    result, basis, system, grid = _he_result_and_basis(L=16.0, n=24)
    out = qvf_density_data_periodic(result, basis, system)

    # Exactly one section, default label.
    assert isinstance(out, dict)
    assert list(out.keys()) == ["Electron density"]
    (data, origin, span) = out["Electron density"]

    # Volume data shape + dtype.
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert data.shape == grid.shape
    # No NaNs / Infs, and ≥ 0 everywhere (electron density).
    assert np.isfinite(data).all()
    assert (data >= -1e-6).all()  # tiny FP wiggle ok

    # Origin: (3,) at cell home corner.
    origin = np.asarray(origin)
    assert origin.shape == (3,)
    np.testing.assert_allclose(origin, np.zeros(3))

    # Span: (3, 3) voxel vectors = lattice columns / n_i.
    span = np.asarray(span)
    assert span.shape == (3, 3)
    expected_span = (np.eye(3) * 16.0 / 24.0).T
    np.testing.assert_allclose(span, expected_span)

    # Custom label honoured.
    out2 = qvf_density_data_periodic(
        result, basis, system, label="GAPW rho"
    )
    assert list(out2.keys()) == ["GAPW rho"]


# ---------- Test 2: end-to-end write_qvf round-trip ---------------------


def test_qvf_density_data_periodic_round_trips_through_write_qvf(tmp_path):
    """Pair the emitter with ``write_qvf`` and confirm the archive has
    a ``volume.density`` section in the manifest.
    """
    from vibeqc.output.formats.qvf import _BOHR_TO_ANGSTROM, validate_qvf, write_qvf
    from vibeqc.output.plan import OutputPlan

    result, basis, system, _ = _he_result_and_basis(L=16.0, n=16)
    volume_data = qvf_density_data_periodic(result, basis, system)

    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "gapw_he",
        method="rhf",
        basis="sto-3g",
        functional=None,
    )

    # Lightweight stubs that satisfy write_qvf's structure-section
    # writer for a periodic system.  write_qvf consumes ``system=`` for
    # periodic jobs (same surface as the rest of the periodic stack).
    class _Result:
        converged = True
        energy = float(result.energy)
        fermi_energy = float(result.mo_energies[0])

    qvf_path = write_qvf(
        tmp_path / "gapw_he",
        plan,
        system=system,
        result=_Result(),
        method="rhf",
        basis="sto-3g",
        volume_data=volume_data,
    )

    assert qvf_path.exists()
    assert qvf_path.suffix == ".qvf"
    assert validate_qvf(qvf_path)["valid"] is True

    with zipfile.ZipFile(qvf_path) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        manifest = json.loads(zf.read("manifest.json"))
        kinds = [s["kind"] for s in manifest["sections"]]
        assert "wavefunction.gto" not in kinds
        struct_sec = next(
            s for s in manifest["sections"] if s["kind"] == "structure"
        )
        struct = json.loads(
            zf.read(struct_sec["members"]["structure"]["path"])
        )
        assert struct["pbc"] == [True, True, True]
        assert struct["dimensionality"] == 3
        np.testing.assert_allclose(
            struct["lattice_vectors"],
            (np.asarray(system.lattice) * _BOHR_TO_ANGSTROM).T,
        )
        assert "volume.density" in kinds, (
            f"expected a volume.density section, got kinds={kinds}"
        )
        # The density payload itself lives under volumes/.
        dens_sec = next(
            s for s in manifest["sections"] if s["kind"] == "volume.density"
        )
        data_path = dens_sec["members"]["data"]["path"]
        assert data_path in names
        grid_path = dens_sec["members"]["grid"]["path"]
        assert grid_path in names
        grid_json = json.loads(zf.read(grid_path))
        voxel_vectors = np.asarray(grid_json["voxel_vectors"], dtype=float)
        shape = np.asarray(grid_json["shape"], dtype=float)
        np.testing.assert_allclose(
            voxel_vectors * shape[:, None],
            np.asarray(system.lattice, dtype=float).T,
        )


# ---------- Test 3: 3D-only guard ---------------------------------------


def test_qvf_density_data_periodic_rejects_non_3d_systems():
    """1D / 2D periodic systems are out of scope for the QVF v1
    ``volume.density`` section — emit a ``ValueError`` rather than
    silently writing a malformed dict.
    """
    result, basis, _system_3d, _ = _he_result_and_basis(L=16.0, n=16)

    sys_2d = core.PeriodicSystem()
    sys_2d.dim = 2
    sys_2d.lattice = np.eye(3) * 16.0
    sys_2d.unit_cell = [core.Atom(2, [8.0, 8.0, 8.0])]

    with pytest.raises(ValueError, match="3D"):
        qvf_density_data_periodic(result, basis, sys_2d)

    sys_1d = core.PeriodicSystem()
    sys_1d.dim = 1
    sys_1d.lattice = np.eye(3) * 16.0
    sys_1d.unit_cell = [core.Atom(2, [8.0, 8.0, 8.0])]

    with pytest.raises(ValueError, match="3D"):
        qvf_density_data_periodic(result, basis, sys_1d)
