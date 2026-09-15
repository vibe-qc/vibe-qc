"""Tests for the GPW-route ASE Calculator wrapper.

:class:`vibeqc.VibeqcGPW` converts an ``ase.Atoms`` with periodic
boundary conditions to a :class:`PeriodicSystem`, dispatches to
:func:`run_periodic_rhf_gpw` (Γ-only) or
:func:`run_periodic_rks_gpw_multi_k` (multi-k pure-DFT), and stores
the converged energy in eV on ``calc.results['energy']``.

These tests pin parity to the underlying drivers:

1. HF (functional=None) on H2 STO-3G in a cubic box matches
   :func:`run_periodic_rhf_gpw` to machine precision (Hartree-to-eV
   converted).
2. LDA RKS matches :func:`run_periodic_rhf_gpw` with
   ``functional='lda'``.
3. Multi-k [2, 2, 2] LDA matches
   :func:`run_periodic_rks_gpw_multi_k`.
4. Odd-electron (open-shell) systems are rejected with a clear
   error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning

ase = pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.units import Bohr, Hartree  # noqa: E402

# Silence the experimental warning the GPW route emits from a
# couple of helpers (the wrapper passes ``quiet=True`` to the SCF
# driver, but downstream helpers like the J builder still warn).
warnings.simplefilter("ignore", GAPWExperimentalWarning)
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Fixtures ------------------------------------------------------


def _h2_atoms_in_cubic_box(L_bohr: float = 12.0) -> Atoms:
    """H2 in a cubic box. ``L_bohr`` is the cube edge in bohr; the
    returned ``Atoms`` has its cell + positions in Å (ASE convention).
    """
    L_ang = L_bohr * Bohr  # bohr → Å
    # Place the two H atoms along x with the same +/-0.7 bohr offset
    # the GPW test suite uses (`tests/test_periodic_gapw_j.py`).
    p1 = np.array([L_bohr / 2 - 0.7, L_bohr / 2, L_bohr / 2]) * Bohr
    p2 = np.array([L_bohr / 2 + 0.7, L_bohr / 2, L_bohr / 2]) * Bohr
    atoms = Atoms("H2", positions=[p1, p2], pbc=True)
    atoms.set_cell(np.eye(3) * L_ang)
    return atoms


def _periodic_system_from_atoms(atoms: Atoms):
    """Mirror :meth:`VibeqcGPW._atoms_to_periodic_system` so tests can
    build their reference calls without touching the calculator.
    """
    lattice_bohr = np.asarray(atoms.cell.array, dtype=np.float64) / Bohr
    positions_bohr = atoms.positions / Bohr
    zs = atoms.get_atomic_numbers()
    sys = vibeqc.PeriodicSystem()
    sys.dim = 3
    sys.lattice = lattice_bohr
    sys.unit_cell = [
        vibeqc.Atom(int(z), list(pos))
        for z, pos in zip(zs, positions_bohr)
    ]
    return sys


# ---------- 1. HF parity --------------------------------------------------


def test_vibeqc_gpw_hf_matches_underlying_driver():
    """HF (functional=None, Γ-only) energy via the calculator must
    match :func:`run_periodic_rhf_gpw` to machine precision after the
    Hartree → eV conversion."""
    atoms = _h2_atoms_in_cubic_box(L_bohr=12.0)
    calc = vibeqc.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)
    atoms.calc = calc

    energy_eV = atoms.get_potential_energy()
    assert np.isfinite(energy_eV)

    # Reference: drive the underlying GPW SCF directly.
    system = _periodic_system_from_atoms(atoms)
    mol = vibeqc.Molecule(list(system.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    ref = vibeqc.run_periodic_rhf_gpw(
        system, basis, cutoff_ha=200.0, functional=None, quiet=True,
    )
    assert ref.converged
    expected_eV = float(ref.energy) * Hartree

    assert energy_eV == pytest.approx(expected_eV, rel=0, abs=1e-12)


# ---------- 2. LDA parity (Γ-only) ---------------------------------------


def test_vibeqc_gpw_lda_matches_underlying_driver():
    """Γ-only LDA energy via the calculator must match
    :func:`run_periodic_rhf_gpw` with ``functional='lda'``."""
    atoms = _h2_atoms_in_cubic_box(L_bohr=12.0)
    calc = vibeqc.VibeqcGPW(
        basis="sto-3g", cutoff_ha=200.0, functional="lda",
    )
    atoms.calc = calc

    energy_eV = atoms.get_potential_energy()
    assert np.isfinite(energy_eV)

    system = _periodic_system_from_atoms(atoms)
    mol = vibeqc.Molecule(list(system.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    ref = vibeqc.run_periodic_rhf_gpw(
        system, basis, cutoff_ha=200.0, functional="lda", quiet=True,
    )
    assert ref.converged
    expected_eV = float(ref.energy) * Hartree

    assert energy_eV == pytest.approx(expected_eV, rel=0, abs=1e-12)


# ---------- 3. Multi-k LDA parity ----------------------------------------


def test_vibeqc_gpw_multi_k_lda_matches_underlying_driver():
    """A [2, 2, 2] LDA energy via the calculator must match the
    underlying :func:`run_periodic_rks_gpw_multi_k`."""
    atoms = _h2_atoms_in_cubic_box(L_bohr=12.0)
    calc = vibeqc.VibeqcGPW(
        basis="sto-3g",
        cutoff_ha=200.0,
        functional="lda",
        kmesh=[2, 2, 2],
    )
    atoms.calc = calc

    energy_eV = atoms.get_potential_energy()
    assert np.isfinite(energy_eV)

    system = _periodic_system_from_atoms(atoms)
    mol = vibeqc.Molecule(list(system.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    kmesh = vibeqc.monkhorst_pack(system, [2, 2, 2])
    ref = vibeqc.run_periodic_rks_gpw_multi_k(
        system, basis, kmesh,
        functional="lda", cutoff_ha=200.0, quiet=True,
    )
    assert ref.converged
    expected_eV = float(ref.energy) * Hartree

    assert energy_eV == pytest.approx(expected_eV, rel=0, abs=1e-12)


# ---------- 4. Open-shell rejection --------------------------------------


def test_vibeqc_gpw_rejects_open_shell():
    """Single H (1 electron, odd) must raise with a clear message."""
    L_bohr = 12.0
    L_ang = L_bohr * Bohr
    atoms = Atoms(
        "H",
        positions=[[L_bohr / 2 * Bohr,
                    L_bohr / 2 * Bohr,
                    L_bohr / 2 * Bohr]],
        pbc=True,
    )
    atoms.set_cell(np.eye(3) * L_ang)

    calc = vibeqc.VibeqcGPW(basis="sto-3g", cutoff_ha=150.0)
    atoms.calc = calc
    with pytest.raises(ValueError, match="odd"):
        atoms.get_potential_energy()


# ---------- 5. Numerical forces ------------------------------------------


def test_vibeqc_gpw_forces_shape_and_symmetry():
    """Central-difference forces on H2 in a cubic box.

    Three properties pin the numerical-force loop:

    1. ``get_forces`` returns an ``(n_atoms, 3)`` array.
    2. Newton's third law: forces on atom 0 and atom 1 are equal
       and opposite along the bond.
    3. Off-axis components (y and z) are ~zero for an x-aligned
       diatomic — the numerical-force noise floor.
    """
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    # Stretched bond (1.4 bohr is short of the HF/STO-3G optimum
    # so there is a real, non-zero force to detect).
    atoms = _h2_atoms_in_cubic_box(L_bohr=12.0)
    calc = vibeqc.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)
    atoms.calc = calc

    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert np.all(np.isfinite(forces))

    # Newton's third law: F0 ≈ -F1 along the bond axis.
    assert forces[0, 0] == pytest.approx(-forces[1, 0], abs=1e-4)

    # Off-axis components ~ zero (an x-aligned diatomic has no
    # symmetry-breaking in y/z up to numerical noise).
    assert np.max(np.abs(forces[:, 1])) < 1e-3
    assert np.max(np.abs(forces[:, 2])) < 1e-3


def test_vibeqc_gpw_forces_near_zero_at_equilibrium():
    """At HF/STO-3G equilibrium (~1.346 bohr bond, see the
    canonical Szabo-Ostlund H2 reference), residual forces along
    the bond axis should be small. The 12-bohr cubic box adds a
    tiny periodic-image perturbation; we only require ~1e-2 eV/Å
    on a stretched bond reduces noticeably toward the optimum."""
    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    # Place the atoms near the HF/STO-3G optimum bond.
    L_bohr = 12.0
    L_ang = L_bohr * Bohr
    r_eq_bohr = 1.346
    p1 = np.array([L_bohr / 2 - r_eq_bohr / 2,
                   L_bohr / 2, L_bohr / 2]) * Bohr
    p2 = np.array([L_bohr / 2 + r_eq_bohr / 2,
                   L_bohr / 2, L_bohr / 2]) * Bohr
    atoms = Atoms("H2", positions=[p1, p2], pbc=True)
    atoms.set_cell(np.eye(3) * L_ang)

    calc = vibeqc.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)
    atoms.calc = calc

    forces = atoms.get_forces()
    # At HF/STO-3G optimum, the bond-axis force should be small
    # — under 0.05 eV/Å.
    f_bond = abs(forces[0, 0])
    assert f_bond < 0.05, (
        f"Expected near-zero force at HF/STO-3G eq, got "
        f"{f_bond:.4f} eV/Å on the bond axis"
    )


def test_vibeqc_gpw_bfgs_optimises_h2_bond():
    """A short BFGS run starting from a stretched bond must
    relax the H-H distance toward the HF/STO-3G optimum
    (~1.346 bohr ≈ 0.712 Å)."""
    from ase.optimize import BFGS  # noqa: PLC0415

    warnings.simplefilter("ignore", GAPWExperimentalWarning)
    # Start from a stretched 1.6-bohr bond.
    L_bohr = 12.0
    L_ang = L_bohr * Bohr
    r0_bohr = 1.6
    p1 = np.array([L_bohr / 2 - r0_bohr / 2,
                   L_bohr / 2, L_bohr / 2]) * Bohr
    p2 = np.array([L_bohr / 2 + r0_bohr / 2,
                   L_bohr / 2, L_bohr / 2]) * Bohr
    atoms = Atoms("H2", positions=[p1, p2], pbc=True)
    atoms.set_cell(np.eye(3) * L_ang)

    calc = vibeqc.VibeqcGPW(basis="sto-3g", cutoff_ha=200.0)
    atoms.calc = calc

    r_initial = atoms.get_distance(0, 1)  # Å

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=0.05, steps=10)

    r_final = atoms.get_distance(0, 1)  # Å
    r_final_bohr = r_final / Bohr

    # The optimisation must have moved the bond length closer to
    # the HF/STO-3G optimum (~1.346 bohr).
    assert r_final < r_initial, (
        f"BFGS did not contract bond: r_initial={r_initial:.4f} Å, "
        f"r_final={r_final:.4f} Å"
    )
    assert 1.2 < r_final_bohr < 1.5, (
        f"BFGS landed at r={r_final_bohr:.3f} bohr; expected near the "
        "HF/STO-3G optimum (~1.346 bohr)"
    )


def test_gamma_warm_source_projects_geometry_and_preserves_physical_guess(monkeypatch):
    import vibeqc as vq
    from vibeqc.ase_periodic_gpw import VibeqcGPW
    import vibeqc.guess_read as reader
    atoms = Atoms("H2", positions=np.array([[6., 6., 5.3], [6., 6., 6.7]]) * Bohr,
                  cell=np.eye(3) * 12 * Bohr, pbc=True)
    calc = VibeqcGPW(basis="sto-3g", functional="lda", cutoff_ha=8, initial_guess="HCORE")
    source = calc._run_scf(atoms)
    displaced = atoms.copy()
    displaced.positions[1, 2] += .01 * Bohr
    called = []
    original = reader.resolve_periodic_read_density_closed
    def project(*args, **kwargs):
        called.append(kwargs.get("read_from"))
        return original(*args, **kwargs)
    monkeypatch.setattr(reader, "resolve_periodic_read_density_closed", project)
    result = calc._run_scf(displaced, source_result=source)
    assert called == [source]
    assert result.converged
    assert source.restart_basis is not None
    selection = result.guess_selection
    assert (selection.requested, selection.effective, selection.transport) == (
        vq.InitialGuess.HCORE, vq.InitialGuess.HCORE, vq.InitialGuess.READ)
