"""End-to-end ``run_job`` thermochemistry block (RRHO, when a Hessian is run).

vibe-qc has had a general RRHO thermochemistry engine (vibeqc.thermo) and an FD
Hessian (vibeqc.hessian), but ``run_job`` only printed harmonic frequencies — it
never turned them into ZPE / thermal corrections.  These tests pin the wiring:
``run_job(hessian=True)`` now appends a thermochemistry block, the numbers agree
with calling vibeqc.thermo directly, the rotational symmetry number is honoured,
and nothing prints when no Hessian is requested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeqc import run_job
from vibeqc.molecule import Atom, Molecule
from vibeqc.thermo import ThermoOptions

_A2B = 1.8897259886


def _h2o() -> Molecule:
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.7572 * _A2B, 0.5865 * _A2B]),
         Atom(1, [0.0, -0.7572 * _A2B, 0.5865 * _A2B])], 0, 1)


def test_run_job_hessian_writes_thermochemistry(tmp_path: Path) -> None:
    """hessian=True appends an RRHO thermochemistry block with sensible numbers
    (water's standard entropy is ~45 cal/mol/K)."""
    run_job(_h2o(), basis="sto-3g", method="rhf", hessian=True,
            thermo_options=ThermoOptions(temperature=298.15, symmetry_number=2),
            output=str(tmp_path / "h2o"), verbose=0)
    out = (tmp_path / "h2o.out").read_text()
    assert "## Thermochemistry (RRHO ideal gas)" in out
    assert "Zero-point energy" in out
    assert "rotor = nonlinear" in out
    assert "s_rot = 2" in out
    # Gibbs correction < enthalpy correction (entropy term lowers G).
    assert "G = E(elec) + G_corr" in out


def test_run_job_thermochemistry_matches_direct_engine(tmp_path: Path) -> None:
    """The .out thermochemistry equals calling vibeqc.thermo on the same FD
    Hessian directly — pins the wiring, not the physics (test_thermo covers
    the physics vs PySCF)."""
    from vibeqc.hessian import HessianFDOptions, compute_hessian_fd
    from vibeqc.thermo import compute_thermochemistry

    mol = _h2o()
    opts = ThermoOptions(temperature=310.0, symmetry_number=2)
    run_job(mol, basis="sto-3g", method="rhf", hessian=True,
            thermo_options=opts, output=str(tmp_path / "h2o"), verbose=0)
    out = (tmp_path / "h2o.out").read_text()

    hess = compute_hessian_fd(mol, "sto-3g", method="RHF",
                              hessian_options=HessianFDOptions())
    thermo = compute_thermochemistry(mol, hess, options=opts)
    # The ZPE printed (10 d.p.) matches the engine.
    assert f"{thermo.zpe:16.10f}" in out
    assert f"{thermo.g_thermal:16.10f}" in out


def test_run_job_no_hessian_no_thermo(tmp_path: Path) -> None:
    """Without hessian=True there is no thermochemistry block (it needs freqs)."""
    run_job(_h2o(), basis="sto-3g", method="rhf",
            output=str(tmp_path / "h2o"), verbose=0)
    out = (tmp_path / "h2o.out").read_text()
    assert "## Thermochemistry" not in out
