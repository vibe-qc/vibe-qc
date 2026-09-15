"""VibeQCPeriodic — ASE Calculator for periodic SCF energy + forces.

The periodic counterpart of ``vibeqc.ase.VibeQC``: it lets ASE-driven
workflows (optimize / NEB / MD) run on vibe-qc periodic systems. These
pin energy + forces both come out of a single SCF, the forces match the
existing ``periodic_forces`` helper (same machinery), ASE caches the
result across calls with unchanged positions, multi-k works, and the
open-shell-HF limitation raises clearly. Also a regression that the
``periodic_forces`` refactor preserved its behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ase")

from ase import Atoms  # noqa: E402
from ase.units import Hartree  # noqa: E402

import vibeqc as vq  # noqa: E402
from vibeqc.ase_periodic import (  # noqa: E402
    VibeQCPeriodic,
    atoms_to_periodic_system,
    periodic_forces,
)


def _h2_box(d: float = 0.8) -> Atoms:
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, d]],
                 cell=[6.0, 6.0, 6.0], pbc=True)


def test_calculator_energy_and_forces():
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", functional="lda", kpts=(1, 1, 1))
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert f.shape == (2, 3) and np.all(np.isfinite(f))
    # Symmetric diatomic along z: forces equal and opposite.
    assert f[0, 2] == pytest.approx(-f[1, 2], abs=1e-6)


def test_forces_match_periodic_forces():
    # The calculator reuses the periodic_forces machinery, so the forces
    # must be identical (not merely close).
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", functional="lda", kpts=(1, 1, 1))
    f_calc = atoms.get_forces()

    sys = atoms_to_periodic_system(atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    f_ref = periodic_forces(atoms, basis, kpts=[1, 1, 1], functional="lda")
    assert np.allclose(f_calc, f_ref, atol=1e-10)


def test_calculator_caches_single_scf(monkeypatch):
    # ASE's base Calculator caches results; getting E then F then E again
    # at unchanged positions runs the SCF exactly once.
    calls = {"n": 0}
    orig = VibeQCPeriodic.calculate

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(VibeQCPeriodic, "calculate", counting)
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", functional="lda")
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_potential_energy()
    assert calls["n"] == 1


def test_calculator_recomputes_when_moved():
    # Moving the atoms invalidates the cache → a fresh SCF, new energy.
    atoms = _h2_box(0.8)
    atoms.calc = VibeQCPeriodic(basis="sto-3g", functional="lda")
    e1 = atoms.get_potential_energy()
    atoms.set_positions([[0, 0, 0], [0, 0, 1.1]])
    e2 = atoms.get_potential_energy()
    assert e1 != e2


def test_calculator_multi_k():
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", functional="lda", kpts=(2, 2, 2))
    assert np.isfinite(atoms.get_potential_energy())
    assert np.all(np.isfinite(atoms.get_forces()))


def test_open_shell_hf_raises():
    # Periodic UHF gradient is deferred — open-shell HF must raise clearly.
    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[6, 6, 6], pbc=True)
    atoms.calc = VibeQCPeriodic(basis="sto-3g", functional=None, multiplicity=2)
    with pytest.raises(NotImplementedError, match="UHF gradient deferred"):
        atoms.get_potential_energy()


def test_periodic_forces_regression():
    # The refactor (shared _periodic_scf_energy_and_forces) must leave the
    # public periodic_forces behaviour intact.
    atoms = _h2_box()
    sys = atoms_to_periodic_system(atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    f = periodic_forces(atoms, basis, kpts=[1, 1, 1], functional="lda")
    assert f.shape == (2, 3) and np.all(np.isfinite(f))
    assert f[0, 2] == pytest.approx(-f[1, 2], abs=1e-6)


def test_calculator_gamma_hf_matches_molecular_limit():
    # Regression for the Γ-only HF break: VibeQCPeriodic(functional=None,
    # kpts=(1,1,1)) used to raise TypeError because the Γ branch called the
    # bare run_rhf_periodic, whose binding accepts only PeriodicSCFOptions
    # (PeriodicRHFOptions is not a registered subclass). The fix routes Γ HF
    # through the Ewald-3D exact-exchange driver. A single closed-shell He
    # atom in a large box must recover the isolated-atom HF limit; the old
    # bare DIRECT_TRUNCATED Γ exchange instead over-binds to ~-3.34 Ha
    # (Madelung self-image leak — CLAUDE.md §7).
    he = Atoms("He", positions=[[0, 0, 0]], cell=[6, 6, 6], pbc=True)
    he.calc = VibeQCPeriodic(basis="sto-3g", functional=None, kpts=(1, 1, 1))
    e_ha = he.get_potential_energy() / Hartree
    # Molecular RHF He/STO-3G = -2.80778396 Ha; Γ-HF in a 6-bohr box recovers
    # it to ~2e-8 Ha via Ewald exchange. The buggy path lands near -3.34 Ha,
    # which a 1e-4 Ha tolerance excludes by ~5000x.
    assert e_ha == pytest.approx(-2.80778396, abs=1e-4)


def test_gamma_hf_forces_symmetric_and_consistent():
    # Both entry points (the calculator and the periodic_forces helper) hit
    # the broken Γ-HF branch. Forces must be finite, equal-and-opposite on a
    # symmetric diatomic, and identical between the two paths.
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", functional=None, kpts=(1, 1, 1))
    f_calc = atoms.get_forces()
    assert f_calc.shape == (2, 3) and np.all(np.isfinite(f_calc))
    assert f_calc[0, 2] == pytest.approx(-f_calc[1, 2], abs=1e-6)

    sys = atoms_to_periodic_system(atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    f_ref = periodic_forces(atoms, basis, kpts=[1, 1, 1], functional=None)
    assert np.allclose(f_calc, f_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# backend="gdf" (G-PBC-002 milestone 5)
# ---------------------------------------------------------------------------


def test_gdf_backend_forces_match_direct_driver():
    """backend='gdf' energy+forces equal the direct compcell driver's."""
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", backend="gdf")
    e_calc = atoms.get_potential_energy()
    f_calc = atoms.get_forces()

    sys = atoms_to_periodic_system(atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-10
    opts.max_iter = 100
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    r = vq.run_pbc_gdf_rhf(
        sys, basis, opts, gdf_method="rsgdf", compute_gradient=True,
        progress=False,
    )
    assert r.converged
    assert e_calc == pytest.approx(r.energy * Hartree, abs=1e-8)
    from ase.units import Bohr

    f_ref = -np.asarray(r.gradient) * (Hartree / Bohr)
    assert np.allclose(f_calc, f_ref, atol=1e-10)


def test_gdf_backend_open_shell_hf():
    """backend='gdf' fills the ewald backend's open-shell-HF hole."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.2]],
                  cell=[8.0, 8.0, 8.0], pbc=True)
    atoms.calc = VibeQCPeriodic(basis="sto-3g", backend="gdf",
                                multiplicity=3)
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert f.shape == (2, 3) and np.all(np.isfinite(f))
    assert f[0, 2] == pytest.approx(-f[1, 2], abs=1e-6)


def test_gdf_backend_multi_k_forces_match_direct_driver():
    """backend='gdf' at kpts=(2,1,1) routes through the multi-k GDF
    driver (Item-4 rung-6 gradients) and reproduces its energy+forces.

    The tuple kmesh is Gamma-centered (multi-k GDF driver convention),
    unlike the ewald backend's shifted Monkhorst-Pack for even meshes.
    """
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", backend="gdf",
                                kpts=(2, 1, 1))
    e_calc = atoms.get_potential_energy()
    f_calc = atoms.get_forces()

    from ase.units import Bohr, Hartree

    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    sys = atoms_to_periodic_system(atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-10
    opts.max_iter = 100
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    r = run_krhf_periodic_gdf(
        sys, basis, (2, 1, 1), opts, gdf_method="rsgdf",
        compute_gradient=True, progress=False,
    )
    assert e_calc == pytest.approx(r.energy * Hartree, abs=1e-9)
    f_ref = -np.asarray(r.gradient) * (Hartree / Bohr)
    assert np.abs(f_calc - f_ref).max() <= 1e-9


def test_gdf_backend_multi_k_compcell_fails_closed():
    """Multi-k GDF forces exist on the rsgdf fit only."""
    atoms = _h2_box()
    atoms.calc = VibeQCPeriodic(basis="sto-3g", backend="gdf",
                                kpts=(2, 1, 1), gdf_method="compcell")
    with pytest.raises(NotImplementedError, match="rsgdf"):
        atoms.get_potential_energy()


def test_gdf_backend_bfgs_relaxation_smoke():
    """A BFGS relaxation runs end-to-end on the GDF backend."""
    from ase.optimize import BFGS

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.9]],
                  cell=[7.0, 7.0, 7.0], pbc=True)
    atoms.calc = VibeQCPeriodic(basis="sto-3g", backend="gdf",
                                functional="lda")
    opt = BFGS(atoms, logfile=None)
    converged = opt.run(fmax=0.05, steps=12)
    assert converged
    d = atoms.get_distance(0, 1)
    # LDA/STO-3G H2 bond length lands near ~0.8-1.0 A.
    assert 0.6 < d < 1.2
