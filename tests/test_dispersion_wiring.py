"""Phase D1c: dispersion wired through run_job and the ASE calculator.

Contracts exercised:

- ``run_job(..., dispersion=...)`` accepts a functional name, a
  :class:`D3BJParams`, ``True``/``"d3bj"`` (inherits the SCF
  functional), and ``None`` (no correction; back-compat path).
- The returned object exposes ``.e_dispersion`` and ``.energy_total``
  when dispersion is on, and still forwards every SCF attribute
  (``mo_energies``, ``density``, ``converged``, …) transparently.
- The ``.out`` file grows a "Dispersion correction (D3-BJ)" block
  with s6 / s8 / a1 / a2 / E_disp / E_SCF / E_total.
- ``VibeQC(..., dispersion=...)`` adds the dispersion contribution to
  both ``get_potential_energy()`` and ``get_forces()``. The force
  difference between disp and no-disp is non-zero (dispersion has
  a r-derivative even though it's tiny).

These tests use the quantitative native backend via ``backend="auto"``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


H2O_XYZ = Path(__file__).parent.parent / "examples" / "h2o.xyz"


# ---------------------------------------------------------------------------
# run_job
# ---------------------------------------------------------------------------

def test_run_job_without_dispersion_is_unchanged(tmp_path):
    """dispersion=None must produce the same SCF energy and no
    dispersion attributes on the returned object — back-compat."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    r = vq.run_job(
        mol, basis="sto-3g", method="rks", functional="pbe",
        output=str(tmp_path / "h2o"),
    )
    assert r.converged
    # No wrapper — raw SCF result must not carry dispersion attributes.
    assert not hasattr(r, "e_dispersion")
    assert not hasattr(r, "energy_total")

    out_text = (tmp_path / "h2o.out").read_text()
    assert "Dispersion correction" not in out_text
    assert re.search(r"^\s+Total energy\s+-?\d+\.\d+", out_text, re.MULTILINE)
    assert not re.search(r"^\s+SCF energy\s+-?\d+\.\d+", out_text, re.MULTILINE)


def test_run_job_with_functional_name_dispersion(tmp_path):
    """dispersion='pbe' should fold in a PBE-D3BJ correction and expose
    e_dispersion + energy_total on the returned object."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    r = vq.run_job(
        mol, basis="sto-3g", method="rks", functional="pbe",
        dispersion="pbe",
        output=str(tmp_path / "h2o"),
    )
    assert r.converged
    # The wrapper must forward every SCF attribute transparently.
    assert r.mo_energies is not None
    assert hasattr(r, "density")

    # Dispersion-specific attributes.
    assert hasattr(r, "e_dispersion")
    assert r.e_dispersion < 0.0               # attractive
    assert abs(r.e_dispersion) < 1e-2         # sensible mHa scale on H2O
    assert r.energy_total == pytest.approx(
        r.energy + r.e_dispersion, rel=0, abs=1e-14,
    )
    # The raw SCF energy must not be mutated.
    assert r.energy != r.energy_total

    out_text = (tmp_path / "h2o.out").read_text()
    assert "Dispersion correction (D3-BJ)" in out_text
    for field in ("s6", "s8", "a1", "a2", "E_disp", "E_SCF", "E_total"):
        assert field in out_text, field
    assert re.search(r"^\s+SCF energy\s+-?\d+\.\d+", out_text, re.MULTILINE)
    assert not re.search(r"^\s+Total energy\s+-?\d+\.\d+", out_text, re.MULTILINE)
    m_total = re.search(r"^\s+E_total\s+(-?\d+\.\d+) Ha$", out_text, re.MULTILINE)
    assert m_total is not None
    assert float(m_total.group(1)) == pytest.approx(r.energy_total, abs=1e-8)


def test_run_job_dispersion_true_inherits_functional(tmp_path):
    """dispersion=True must look up D3-BJ params for the SCF functional."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    r1 = vq.run_job(
        mol, basis="sto-3g", method="rks", functional="pbe",
        dispersion=True,
        output=str(tmp_path / "t"),
    )
    r2 = vq.run_job(
        mol, basis="sto-3g", method="rks", functional="pbe",
        dispersion="pbe",
        output=str(tmp_path / "p"),
    )
    assert r1.e_dispersion == pytest.approx(r2.e_dispersion, rel=0, abs=1e-14)


def test_run_job_dispersion_true_without_functional_raises(tmp_path):
    """dispersion=True with HF (no functional) is ambiguous — error
    message should be directive about what to pass instead."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    with pytest.raises(ValueError, match="requires a DFT functional"):
        vq.run_job(
            mol, basis="sto-3g", method="rhf", dispersion=True,
            output=str(tmp_path / "t"),
        )


def test_run_job_dispersion_hf_with_named_functional(tmp_path):
    """HF + explicit functional-name-for-damping is a legitimate use
    case (HF-D3BJ with HF-specific damping)."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    r = vq.run_job(
        mol, basis="sto-3g", method="rhf", dispersion="hf",
        output=str(tmp_path / "h"),
    )
    assert r.converged
    assert r.e_dispersion < 0.0


def test_run_job_unknown_dispersion_functional_raises(tmp_path):
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    with pytest.raises(ValueError, match="no D3-BJ parameters"):
        vq.run_job(
            mol, basis="sto-3g", method="rks", functional="pbe",
            dispersion="not-a-functional",
            output=str(tmp_path / "t"),
        )


# ---------------------------------------------------------------------------
# ASE calculator
# ---------------------------------------------------------------------------

def test_ase_calculator_dispersion_adds_to_energy():
    """When the VibeQC calculator is given ``dispersion=``, both the
    energy it reports and the e_scf / e_dispersion decomposition it
    stores on ``results`` must be self-consistent."""
    pytest.importorskip("ase")
    from ase.build import molecule as ase_molecule
    from vibeqc.ase import VibeQC

    atoms = ase_molecule("H2O")
    atoms.calc = VibeQC(basis="sto-3g", functional="pbe", dispersion="pbe")
    e = atoms.get_potential_energy()

    decomp_scf = atoms.calc.results["e_scf"]
    decomp_disp = atoms.calc.results["e_dispersion"]
    assert e == pytest.approx(decomp_scf + decomp_disp, rel=0, abs=1e-10)
    assert decomp_disp < 0.0


def test_ase_calculator_dispersion_changes_forces():
    """The dispersion gradient is tiny but nonzero — forces with
    dispersion must differ from forces without."""
    pytest.importorskip("ase")
    from ase.build import molecule as ase_molecule
    from vibeqc.ase import VibeQC

    atoms = ase_molecule("H2O")
    atoms.calc = VibeQC(basis="sto-3g", functional="pbe")
    f_no = atoms.get_forces().copy()

    atoms.calc = VibeQC(basis="sto-3g", functional="pbe", dispersion="pbe")
    f_yes = atoms.get_forces()

    # Difference must be finite and small.
    diff = np.linalg.norm(f_yes - f_no)
    assert diff > 1e-6
    assert diff < 1e-2


def test_ase_calculator_dispersion_true_inherits_functional():
    pytest.importorskip("ase")
    from ase.build import molecule as ase_molecule
    from vibeqc.ase import VibeQC

    atoms = ase_molecule("H2O")
    atoms.calc = VibeQC(basis="sto-3g", functional="pbe", dispersion=True)
    e_true = atoms.get_potential_energy()
    atoms.calc = VibeQC(basis="sto-3g", functional="pbe", dispersion="pbe")
    e_named = atoms.get_potential_energy()
    assert e_true == pytest.approx(e_named, rel=0, abs=1e-10)


def test_ase_calculator_dispersion_true_without_functional_raises():
    pytest.importorskip("ase")
    from ase.build import molecule as ase_molecule
    from vibeqc.ase import VibeQC

    atoms = ase_molecule("H2O")
    atoms.calc = VibeQC(basis="sto-3g", dispersion=True)   # HF, no functional
    with pytest.raises(ValueError, match="requires a `functional="):
        atoms.get_potential_energy()


def test_ase_calculator_logs_dispersion_energy(caplog):
    pytest.importorskip("ase")
    from ase.build import molecule as ase_molecule
    from vibeqc.ase import VibeQC

    atoms = ase_molecule("H2O")
    atoms.calc = VibeQC(basis="sto-3g", functional="pbe", dispersion="pbe")
    with caplog.at_level(logging.INFO, logger="vibeqc.ase"):
        atoms.get_potential_energy()
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("D3-BJ correction" in m for m in messages), messages


# ---------------------------------------------------------------------------
# Wrapper semantics
# ---------------------------------------------------------------------------

def test_wrapper_forwards_everything_to_scf_result(tmp_path):
    """_DispersionAugmented is meant to be transparent — anything
    that worked on the raw SCF result must still work."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    r = vq.run_job(
        mol, basis="sto-3g", method="rks", functional="pbe",
        dispersion="pbe",
        output=str(tmp_path / "t"),
    )
    # Attributes that live on the pybind11 SCF result.
    assert r.converged is True
    assert r.n_iter > 0
    assert r.mo_energies.shape[0] > 0
    assert r.density.shape[0] == r.density.shape[1]
    # And a repr mentioning both decomposition pieces.
    s = repr(r)
    assert "e_dispersion" in s
    assert "energy_total" in s


def test_wrapper_adds_dispersion_to_existing_method_total():
    """A post-HF total remains the base when dispersion wraps it."""
    from types import SimpleNamespace

    from vibeqc.runner import _DispersionAugmented

    post_hf = SimpleNamespace(energy=-10.0, energy_total=-10.2)
    result = _DispersionAugmented(post_hf, -0.01, object())

    assert result.energy == pytest.approx(-10.0)
    assert result.e_scf == pytest.approx(-10.0)
    assert result.energy_total == pytest.approx(-10.21)
