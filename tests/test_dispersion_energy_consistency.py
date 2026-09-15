"""D3(BJ) total-energy consistency across the three optimizer surfaces.

Regression for the glycine optimizer SI matrix (rp167, 2026-07-01)
finding: the PBE0-D3(BJ) rows collected from ASE-optimized jobs were
~0.0097 Ha above the native-optimizer rows because the collection layer
took ``run_job(...).energy`` (bare SCF) while the native provider
reports SCF + dispersion. The three surfaces a geometry optimization
can run on must agree on what "the energy" is:

* ``vibeqc.ase.VibeQC`` with ``dispersion=`` (drives ASE optimizers),
* ``vibeqc.geomopt.MolecularSCFProvider`` with ``dispersion_params=``
  (drives the native optimizers),
* ``run_job(..., dispersion=...)`` (the final single point) via
  ``.energy_total`` and the structured-log ``job_end`` record.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from vibeqc import Atom, Molecule, d3bj_params_for, dftd4_available
from vibeqc.runner import run_job

ase = pytest.importorskip("ase")

_A2B = 1.0 / 0.529177210903

_H2O_ANGSTROM = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 0.793353, -0.613510)),
    ("H", (0.0, -0.793353, -0.613510)),
]
_Z = {"O": 8, "H": 1}


def _h2o_molecule() -> Molecule:
    return Molecule(
        [
            Atom(_Z[sym], [c * _A2B for c in xyz])
            for sym, xyz in _H2O_ANGSTROM
        ],
        0,
        1,
    )


def test_d3bj_total_energy_consistent_across_surfaces(tmp_path: Path) -> None:
    functional = "pbe"
    basis = "sto-3g"

    # ---- surface 1: the ASE calculator (total energy by ASE contract) ----
    from ase import Atoms
    from ase.units import Bohr, Hartree

    from vibeqc.ase import VibeQC

    atoms = Atoms(
        symbols=[sym for sym, _ in _H2O_ANGSTROM],
        positions=[xyz for _, xyz in _H2O_ANGSTROM],
    )
    atoms.calc = VibeQC(basis=basis, functional=functional, dispersion="d3bj")
    ase_forces = np.asarray(atoms.get_forces(), dtype=float)
    e_ase_total = float(atoms.get_potential_energy()) / Hartree
    e_ase_scf = float(atoms.calc.results["e_scf"]) / Hartree
    e_ase_disp = float(atoms.calc.results["e_dispersion"]) / Hartree

    # ---- surface 2: the native geomopt provider ----
    from vibeqc.geomopt import MolecularSCFProvider

    provider = MolecularSCFProvider(
        basis,
        method="rks",
        functional=functional,
        dispersion_params=d3bj_params_for(functional),
    )
    e_native, native_grad = provider(_h2o_molecule())
    e_native = float(e_native)

    # ---- surface 3: run_job final single point + structured log ----
    stem = tmp_path / "h2o_pbe_d3bj"
    result = run_job(
        _h2o_molecule(),
        basis=basis,
        method="rks",
        functional=functional,
        dispersion="d3bj",
        output=stem,
        structured_log=True,
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    # The dispersion term is real (else this test proves nothing).
    assert abs(float(result.e_dispersion)) > 1e-7

    # The result object decomposition is self-consistent.
    assert float(result.energy_total) == pytest.approx(
        float(result.energy) + float(result.e_dispersion), abs=1e-12
    )
    assert float(result.e_scf) == pytest.approx(float(result.energy), abs=1e-12)

    # All three surfaces agree on the total energy. The three code paths
    # run the same SCF + D3 kernels, so agreement is tight; 1e-8 Ha
    # leaves room for accumulation-order noise only.
    assert e_ase_total == pytest.approx(float(result.energy_total), abs=1e-8)
    assert e_native == pytest.approx(float(result.energy_total), abs=1e-8)
    assert e_ase_scf == pytest.approx(float(result.energy), abs=1e-8)
    assert e_ase_disp == pytest.approx(float(result.e_dispersion), abs=1e-10)
    assert np.asarray(native_grad).reshape(-1, 3) == pytest.approx(
        -ase_forces * Bohr / Hartree, abs=1e-8
    )

    # The machine-readable job_end record carries the explicit split --
    # this is the record the glycine SI matrix would have needed to
    # avoid under-reporting D3(BJ) totals by taking the bare energy.
    events = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    job_end = [e for e in events if e.get("event") == "job_end"][-1]
    assert job_end["energy"] == pytest.approx(
        float(result.energy_total), abs=1e-12
    )
    assert job_end["e_scf"] == pytest.approx(float(result.energy), abs=1e-12)
    assert job_end["e_dispersion"] == pytest.approx(
        float(result.e_dispersion), abs=1e-12
    )
    assert job_end["e_total"] == pytest.approx(
        float(result.energy_total), abs=1e-12
    )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert manifest["progress"]["phase"] == "post-scf"
    assert manifest["progress"]["energy_eh"] == pytest.approx(
        float(result.energy_total), abs=1e-12
    )


@pytest.mark.skipif(not dftd4_available(), reason="dftd4 optional extra missing")
def test_d4_total_energy_is_exposed_on_result_object(tmp_path: Path) -> None:
    """The D4 return object must expose the total recorded by job_end."""
    stem = tmp_path / "h2o_pbe_d4"
    result = run_job(
        _h2o_molecule(),
        basis="sto-3g",
        method="rks",
        functional="pbe",
        dispersion="d4",
        output=stem,
        structured_log=True,
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )

    events = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    job_end = [event for event in events if event.get("event") == "job_end"][-1]

    assert abs(float(result.e_dispersion)) > 1e-7
    assert float(result.e_scf) == pytest.approx(float(result.energy), abs=1e-12)
    assert float(result.energy_total) == pytest.approx(
        float(result.energy) + float(result.e_dispersion), abs=1e-12
    )
    assert job_end["e_total"] == pytest.approx(
        float(result.energy_total), abs=1e-12
    )


def test_job_end_has_no_split_without_corrections(tmp_path: Path) -> None:
    """No dispersion -> no e_total field; `energy` is the whole story."""
    stem = tmp_path / "h2o_pbe_plain"
    run_job(
        _h2o_molecule(),
        basis="sto-3g",
        method="rks",
        functional="pbe",
        output=stem,
        structured_log=True,
        output_qvf=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        progress=False,
    )
    events = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    job_end = [e for e in events if e.get("event") == "job_end"][-1]
    assert "e_total" not in job_end
    assert "e_dispersion" not in job_end
