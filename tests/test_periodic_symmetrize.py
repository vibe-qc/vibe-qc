"""GS1 periodic-structure symmetrisation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq

DATA = Path(__file__).parent / "data"


def _system_from_crystal(crystal: vq.Crystal) -> vq.PeriodicSystem:
    lattice = np.asarray(crystal.lattice, dtype=float, order="F")
    atoms = [
        vq.Atom(
            int(crystal.species[i]),
            (
                lattice
                @ np.asarray(crystal.fractional_coords[:, i], dtype=float)
            ).tolist(),
        )
        for i in range(crystal.n_atoms)
    ]
    return vq.PeriodicSystem(dim=3, lattice=lattice, unit_cell=atoms)


def _distorted_mgo() -> vq.PeriodicSystem:
    crystal = vq.read_poscar(DATA / "MgO_conventional.poscar")
    assert isinstance(crystal, vq.Crystal)
    frac = np.asarray(crystal.fractional_coords, dtype=float).copy()
    frac[:, 1] += np.array([2.0e-5, -1.0e-5, 1.5e-5])
    distorted = vq.Crystal(crystal.lattice, frac, list(crystal.species))
    return _system_from_crystal(distorted)


def test_detect_spacegroup_does_not_mutate_periodic_system():
    sys = _system_from_crystal(vq.read_poscar(DATA / "NaCl.poscar"))
    assert sys.symmetry is None

    sg = vq.detect_spacegroup(sys, symprec=1.0e-4)

    assert sg.number == 225
    assert sg.international_symbol.startswith("Fm-3m")
    assert sys.symmetry is None


def test_symmetrise_idealizes_distorted_conventional_cell():
    sys = _distorted_mgo()
    original_positions = [list(atom.xyz) for atom in sys.unit_cell]

    result = vq.symmetrise(sys, symprec=1.0e-3)

    assert isinstance(result.system, vq.PeriodicSystem)
    assert result.report.spacegroup_before.number == 225
    assert result.report.spacegroup_after.number == 225
    assert result.report.n_atoms_before == 8
    assert result.report.n_atoms_after == 8
    assert result.report.rms_displacement_bohr > 0.0
    assert result.report.max_displacement_bohr >= result.report.rms_displacement_bohr
    assert result.system.symmetry is not None
    assert result.system.symmetry.number == 225
    assert [list(atom.xyz) for atom in sys.unit_cell] == original_positions


def test_symmetrise_to_primitive_reduces_conventional_mgo():
    sys = _system_from_crystal(vq.read_poscar(DATA / "MgO_conventional.poscar"))

    out, report = vq.symmetrise(sys, symprec=1.0e-4, to_primitive=True)

    assert len(out.unit_cell) == 2
    assert report.n_atoms_before == 8
    assert report.n_atoms_after == 2
    assert report.to_primitive is True
    assert report.spacegroup_after.number == 225
    assert report.volume_after_bohr3 == pytest.approx(
        report.volume_before_bohr3 / 4.0,
        rel=1.0e-12,
    )


def test_read_poscar_symmetrise_keyword_returns_result():
    result = vq.read_poscar(
        DATA / "MgO_conventional.poscar",
        symmetrise=True,
        to_primitive=True,
    )

    assert isinstance(result, vq.SymmetriseResult)
    assert isinstance(result.system, vq.PeriodicSystem)
    assert result.report.spacegroup_after.number == 225
    assert len(result.system.unit_cell) == 2


def test_symmetrise_rejects_lower_dimensional_systems():
    sys = vq.PeriodicSystem(
        dim=2,
        lattice=np.diag([5.0, 5.0, 30.0]),
        unit_cell=[vq.Atom(6, [0.0, 0.0, 0.0])],
    )

    with pytest.raises(ValueError, match="3D"):
        vq.symmetrise(sys)
