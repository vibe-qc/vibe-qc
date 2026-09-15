"""End-to-end integration tests for SCF eigensolver keywords.

Verifies that the ``solver`` keyword on ``run_job`` and
``run_periodic_job`` selects the correct diagonalisation backend and
produces identical energies across all supported solvers for
molecular RHF / UHF / RKS / UKS calculations.

The three supported solvers are:
- ``"dense"``   — full dense diagonalisation (LAPACK).
- ``"davidson"`` — block-Davidson iterative solver (C++ backend).
- ``"lobpcg"``   — LOBPCG iterative solver (falls back to Davidson
                   for molecular SCF; native Python stack available
                   for periodic drivers and ROHF).
"""

from __future__ import annotations

import tempfile

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import Atom, Molecule, PeriodicSystem
from vibeqc.periodic_runner import run_periodic_job
from vibeqc.runner import run_job

# ---------------------------------------------------------------------------
# Shared helper factories
# ---------------------------------------------------------------------------

# All coordinates in bohr.
# Water: O at origin, H atoms symmetric in the y=0 plane.
WATER_ATOMS = [
    Atom(8, [0.0, 0.0, 0.1173]),
    Atom(1, [0.0, 0.7572, -0.4692]),
    Atom(1, [0.0, -0.7572, -0.4692]),
]

# CO: C at origin, O along +z.  1.2 bohr bond length.
CO_ATOMS = [
    Atom(6, [0.0, 0.0, 0.0]),
    Atom(8, [0.0, 0.0, 1.2]),
]

# O₂ triplet: O-O along z, multiplicity=3.
O2_ATOMS = [
    Atom(8, [0.0, 0.0, 0.0]),
    Atom(8, [0.0, 0.0, 2.282]),  # ~1.208 Å in bohr
]

# Supported eigensolver names (the public keyword values).
SOLVERS = ["dense", "davidson", "lobpcg"]


def _water_mol() -> Molecule:
    return Molecule(atoms=list(WATER_ATOMS), charge=0, multiplicity=1)


def _co_mol() -> Molecule:
    return Molecule(atoms=list(CO_ATOMS), charge=0, multiplicity=3)


def _o2_mol() -> Molecule:
    return Molecule(atoms=list(O2_ATOMS), charge=0, multiplicity=3)


# ---------------------------------------------------------------------------
# 1. Water / STO-3G  RHF — all three solvers must agree to 1e-10 Ha
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_water_sto3g_rhf_solver_parity(solver: str) -> None:
    """Water / STO-3G RHF: each solver gives identical energy to 1e-10 Ha."""
    mol = _water_mol()
    with tempfile.TemporaryDirectory() as tmp:
        ref = run_job(
            mol, basis="sto-3g", method="rhf", output=tmp, verbose=0, solver="dense"
        )
    assert ref.converged, "dense reference did not converge"

    with tempfile.TemporaryDirectory() as tmp:
        r = run_job(
            mol, basis="sto-3g", method="rhf", output=tmp, verbose=0, solver=solver
        )
    assert r.converged, f"solver={solver} did not converge"
    assert abs(r.energy - ref.energy) < 1e-10, (
        f"solver={solver}: energy mismatch {r.energy - ref.energy:.2e} Ha"
    )


# ---------------------------------------------------------------------------
# 2. Water / def2-SVP  RHF — larger basis, mark as slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("solver", SOLVERS)
def test_water_def2svp_rhf_solver_parity(solver: str) -> None:
    """Water / def2-SVP RHF: each solver gives identical energy to 1e-8 Ha."""
    mol = _water_mol()
    with tempfile.TemporaryDirectory() as tmp:
        ref = run_job(
            mol, basis="def2-svp", method="rhf", output=tmp, verbose=0, solver="dense"
        )
    assert ref.converged, "dense reference did not converge"

    with tempfile.TemporaryDirectory() as tmp:
        r = run_job(
            mol, basis="def2-svp", method="rhf", output=tmp, verbose=0, solver=solver
        )
    assert r.converged, f"solver={solver} did not converge"
    assert abs(r.energy - ref.energy) < 1e-8, (
        f"solver={solver}: energy mismatch {r.energy - ref.energy:.2e} Ha"
    )


# ---------------------------------------------------------------------------
# 3. CO / STO-3G  UHF — open-shell test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_co_sto3g_uhf_solver_parity(solver: str) -> None:
    """CO triplet / STO-3G UHF: each solver gives identical energy to 1e-10 Ha."""
    mol = _co_mol()
    with tempfile.TemporaryDirectory() as tmp:
        ref = run_job(
            mol, basis="sto-3g", method="uhf", output=tmp, verbose=0, solver="dense"
        )
    assert ref.converged, "dense reference did not converge"

    with tempfile.TemporaryDirectory() as tmp:
        r = run_job(
            mol, basis="sto-3g", method="uhf", output=tmp, verbose=0, solver=solver
        )
    assert r.converged, f"solver={solver} did not converge"
    assert abs(r.energy - ref.energy) < 1e-10, (
        f"solver={solver}: energy mismatch {r.energy - ref.energy:.2e} Ha"
    )


# ---------------------------------------------------------------------------
# 4. Water / STO-3G  RKS with PBE — DFT test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_water_sto3g_rks_pbe_solver_parity(solver: str) -> None:
    """Water / STO-3G RKS/PBE: each solver gives identical energy to 1e-10 Ha."""
    mol = _water_mol()
    with tempfile.TemporaryDirectory() as tmp:
        ref = run_job(
            mol,
            basis="sto-3g",
            method="rks",
            functional="PBE",
            output=tmp,
            verbose=0,
            solver="dense",
        )
    assert ref.converged, "dense reference did not converge"

    with tempfile.TemporaryDirectory() as tmp:
        r = run_job(
            mol,
            basis="sto-3g",
            method="rks",
            functional="PBE",
            output=tmp,
            verbose=0,
            solver=solver,
        )
    assert r.converged, f"solver={solver} did not converge"
    assert abs(r.energy - ref.energy) < 1e-10, (
        f"solver={solver}: energy mismatch {r.energy - ref.energy:.2e} Ha"
    )


# ---------------------------------------------------------------------------
# 5. O₂ triplet / STO-3G  UKS with PBE — open-shell DFT test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_o2_sto3g_uks_pbe_solver_parity(solver: str) -> None:
    """O₂ triplet / STO-3G UKS/PBE: each solver gives identical energy to 1e-10 Ha."""
    mol = _o2_mol()
    with tempfile.TemporaryDirectory() as tmp:
        ref = run_job(
            mol,
            basis="sto-3g",
            method="uks",
            functional="PBE",
            output=tmp,
            verbose=0,
            solver="dense",
        )
    assert ref.converged, "dense reference did not converge"

    with tempfile.TemporaryDirectory() as tmp:
        r = run_job(
            mol,
            basis="sto-3g",
            method="uks",
            functional="PBE",
            output=tmp,
            verbose=0,
            solver=solver,
        )
    assert r.converged, f"solver={solver} did not converge"
    assert abs(r.energy - ref.energy) < 1e-10, (
        f"solver={solver}: energy mismatch {r.energy - ref.energy:.2e} Ha"
    )


# ---------------------------------------------------------------------------
# 6. Invalid solver raises ValueError
# ---------------------------------------------------------------------------


def test_invalid_solver_raises_valueerror() -> None:
    """Passing an unrecognised solver name raises ValueError."""
    mol = _water_mol()
    with pytest.raises(ValueError, match="solver=.*nonexistent"):
        with tempfile.TemporaryDirectory() as tmp:
            run_job(
                mol,
                basis="sto-3g",
                method="rhf",
                output=tmp,
                verbose=0,
                solver="nonexistent",
            )


def test_invalid_solver_periodic_raises_valueerror(tmp_path) -> None:
    """Periodic runner also rejects unrecognised solver names."""
    lattice = np.eye(3) * 8.0
    system = PeriodicSystem(
        3, lattice, [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], charge=0, multiplicity=1
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="solver=.*bogus"):
        # The solver guard fires downstream of the output writer, so the
        # refusal still needs a tmp_path stem to stay out of the tree (#508).
        run_periodic_job(
            system, basis, method="RHF", solver="bogus",
            output=str(tmp_path / "invalid_solver_periodic"),
        )


# ---------------------------------------------------------------------------
# 7. Solver keyword on periodic — H₂ chain with davidson
# ---------------------------------------------------------------------------


def test_periodic_h2_chain_solver_davidson() -> None:
    """Periodic H₂ in a vacuum box: solver='davidson' converges and
    matches dense diagonalisation to 1e-8 Ha."""
    lattice = np.eye(3) * 8.0
    system = PeriodicSystem(
        3, lattice, [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], charge=0, multiplicity=1
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with tempfile.TemporaryDirectory() as tmp:
        ref = run_periodic_job(system, basis, method="RHF", output=tmp, solver="dense")
    assert ref.converged, "periodic dense reference did not converge"

    with tempfile.TemporaryDirectory() as tmp:
        r = run_periodic_job(system, basis, method="RHF", output=tmp, solver="davidson")
    assert r.converged, "periodic davidson did not converge"
    assert abs(r.energy - ref.energy) < 1e-8, (
        f"periodic davidson energy mismatch: {r.energy - ref.energy:.2e} Ha"
    )
