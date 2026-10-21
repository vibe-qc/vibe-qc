"""Generic molecular DFT-D4 on the optimization and Hessian surface (#548).

``run_job(..., dispersion="d4")`` reported a PBE0-D4 total but optimized and
differentiated bare PBE0: the optimizers and the finite-difference Hessian
received only D3(BJ) parameters. A :class:`D4DispersionSpec` now travels in
the same ``dispersion_params`` slot, and every consumer evaluates the D4
energy and analytic gradient at the current geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.dispersion import D4DispersionSpec, compute_d3bj, compute_dispersion

pytest.importorskip("dftd4")

PBE0_D4 = D4DispersionSpec("pbe0")


def _methanol_like() -> vq.Molecule:
    """A nonsymmetric six-atom molecule with a nonzero D4 gradient (bohr)."""
    return vq.Molecule([
        vq.Atom(6, [0.0, 0.0, 0.0]), vq.Atom(8, [2.7, 0.1, 0.0]), vq.Atom(1, [3.3, 1.7, 0.2]),
        vq.Atom(1, [-0.7, 1.9, 0.3]), vq.Atom(1, [-0.6, -1.0, 1.7]), vq.Atom(1, [-0.8, -1.1, -1.6]),
    ])


def _water() -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.25]), vq.Atom(1, [0.0, 1.5, -0.85]), vq.Atom(1, [0.1, -1.4, -0.9]),
    ])


def _displaced(mol: vq.Molecule, atom: int, axis: int, step: float) -> vq.Molecule:
    atoms = [
        vq.Atom(a.Z, [c + (step if (i == atom and j == axis) else 0.0) for j, c in enumerate(a.xyz)])
        for i, a in enumerate(mol.atoms)
    ]
    return vq.Molecule(atoms, mol.charge, mol.multiplicity)


@pytest.mark.parametrize("step", [1e-3, 5e-4])
def test_d4_gradient_matches_finite_differences_of_the_d4_energy(step):
    mol = _methanol_like()
    analytic = np.asarray(compute_dispersion(mol, PBE0_D4, with_gradient=True).gradient)
    assert np.abs(analytic).max() > 1e-5
    fd = np.zeros_like(analytic)
    for atom in range(len(mol.atoms)):
        for axis in range(3):
            plus = compute_dispersion(_displaced(mol, atom, axis, step), PBE0_D4).energy
            minus = compute_dispersion(_displaced(mol, atom, axis, -step), PBE0_D4).energy
            fd[atom, axis] = (plus - minus) / (2.0 * step)
    assert np.abs(analytic - fd).max() < 1e-9      # measured 6e-11 and 1e-11


def test_compute_dispersion_keeps_the_d3bj_path():
    mol = _methanol_like()
    via = compute_dispersion(mol, "pbe0", with_gradient=True)
    direct = compute_d3bj(mol, "pbe0", with_gradient=True)
    assert via.energy == direct.energy
    np.testing.assert_array_equal(np.asarray(via.gradient), np.asarray(direct.gradient))


def _rks_pbe0(mol):
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.functional = "pbe0"
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-11, 1e-8
    return basis, opts, vq.run_rks(mol, basis, opts)


def test_molecular_gradient_and_scf_provider_carry_d4():
    from vibeqc.geomopt.providers import MolecularSCFProvider
    from vibeqc.molecular_optimize import _compute_molecular_gradient

    mol = _water()
    basis, opts, scf = _rks_pbe0(mol)
    d4 = compute_dispersion(mol, PBE0_D4, with_gradient=True)
    bare = np.asarray(_compute_molecular_gradient(mol, basis, scf, "rks", grid_options=opts.grid))
    with_d4 = np.asarray(_compute_molecular_gradient(
        mol, basis, scf, "rks", grid_options=opts.grid, dispersion_params=PBE0_D4))
    np.testing.assert_allclose(with_d4 - bare, np.asarray(d4.gradient), atol=1e-13)

    e_bare, g_bare = MolecularSCFProvider("sto-3g", method="rks", functional="pbe0", rks_options=opts)(mol)
    e_d4, g_d4 = MolecularSCFProvider(
        "sto-3g", method="rks", functional="pbe0", rks_options=opts, dispersion_params=PBE0_D4)(mol)
    assert e_d4 - e_bare == pytest.approx(d4.energy, abs=1e-9)
    np.testing.assert_allclose(g_d4 - g_bare, np.asarray(d4.gradient).ravel(), atol=1e-8)


def test_ase_calculator_carries_d4_energy_and_forces():
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from vibeqc.ase import VibeQC

    mol = _water()
    atoms = Atoms("OHH", positions=np.array([a.xyz for a in mol.atoms]) * Bohr)
    atoms.calc = VibeQC(basis="sto-3g", functional="pbe0")
    e_bare, f_bare = atoms.get_potential_energy(), atoms.get_forces().copy()
    atoms.calc = VibeQC(basis="sto-3g", functional="pbe0", dispersion=PBE0_D4)
    e_d4, f_d4 = atoms.get_potential_energy(), atoms.get_forces()
    d4 = compute_dispersion(mol, PBE0_D4, with_gradient=True)
    assert e_d4 - e_bare == pytest.approx(d4.energy * Hartree, abs=1e-7)
    np.testing.assert_allclose(f_d4 - f_bare, -np.asarray(d4.gradient) * Hartree / Bohr, atol=1e-6)


def test_fd_hessian_differentiates_the_d4_gradient():
    """H(D4 surface) - H(bare) is the finite-difference Hessian of the D4
    gradient alone, at the same step."""
    from vibeqc.hessian import HessianFDOptions, compute_hessian_fd

    mol = _water()
    opts = vq.RHFOptions()
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-12, 1e-9
    h_opts = HessianFDOptions(include_dipole_derivatives=False)
    bare = compute_hessian_fd(mol, "sto-3g", "RHF", scf_options=opts, hessian_options=h_opts)
    d4 = compute_hessian_fd(mol, "sto-3g", "RHF", scf_options=opts, hessian_options=h_opts,
                            dispersion_params=PBE0_D4)
    step = h_opts.step_bohr
    n = 3 * len(mol.atoms)
    expected = np.zeros((n, n))
    for atom in range(len(mol.atoms)):
        for axis in range(3):
            plus = np.asarray(compute_dispersion(_displaced(mol, atom, axis, step), PBE0_D4,
                                                 with_gradient=True).gradient).reshape(-1)
            minus = np.asarray(compute_dispersion(_displaced(mol, atom, axis, -step), PBE0_D4,
                                                  with_gradient=True).gradient).reshape(-1)
            expected[:, 3 * atom + axis] = (plus - minus) / (2.0 * step)
    expected = 0.5 * (expected + expected.T)        # compute_hessian_fd symmetrises
    diff = np.asarray(d4.hessian) - np.asarray(bare.hessian)
    assert np.abs(expected).max() > 1e-6
    np.testing.assert_allclose(diff, expected, atol=1e-8)


def test_fd_hessian_refuses_dispersion_on_omx():
    from vibeqc.hessian import compute_hessian_fd

    with pytest.raises(ValueError, match="not to OM2/OM3"):
        compute_hessian_fd(_water(), "", "OM2", dispersion_params=PBE0_D4)


def test_run_job_d4_hessian_names_the_d4_surface(tmp_path):
    import json
    import zipfile

    result = vq.run_job(
        _water(), basis="sto-3g", method="rks", functional="pbe0", dispersion="d4",
        hessian=True, output=tmp_path / "w", citations=False,
        write_molden_file=False, write_population_file=False, progress=False,
    )
    text = (tmp_path / "w.out").read_text()
    assert "Surface: RKS(PBE0)/sto-3g + D4" in text
    section = (tmp_path / "w.system").read_text().split("[hessian]")[1].split("[", 1)[0]
    assert 'surface        = "RKS(PBE0)/sto-3g + D4"' in section
    with zipfile.ZipFile(tmp_path / "w.qvf") as zf:
        meta = json.loads(zf.read("vibrations/metadata.json"))
    assert meta["surface"]["surface"] == "RKS(PBE0)/sto-3g + D4"
    d4 = compute_dispersion(_water(), PBE0_D4)
    assert float(result.energy_total) - float(result.energy) == pytest.approx(d4.energy, abs=1e-10)


@pytest.mark.parametrize("route", [
    {"optimizer_backend": "native"},
    {"optimizer_backend": "brent"},
    {"optimizer_backend": "ase"},
    {"geom_opt": "bfgs"},
])
def test_run_job_d4_trajectory_energies_are_on_the_d4_surface(tmp_path, route):
    """Every optimizer route records D4-inclusive energies: the last frame
    is the reported total, not the bare SCF energy."""
    import json
    import zipfile

    if route.get("optimizer_backend") == "ase":
        pytest.importorskip("ase")
    result = vq.run_job(
        _water(), basis="sto-3g", method="rks", functional="pbe0", dispersion="d4",
        optimize=True, output=tmp_path / "w", citations=False,
        write_molden_file=False, write_population_file=False, progress=False, **route,
    )
    with zipfile.ZipFile(tmp_path / "w.qvf") as zf:
        energies = json.loads(zf.read("trajectory/metadata.json"))["energies"]
    assert len(energies) >= 2
    assert energies[-1] == pytest.approx(float(result.energy_total), abs=1e-9)
    assert energies[-1] - float(result.energy) == pytest.approx(
        float(result.e_dispersion), abs=1e-9)


def test_run_job_d4_geometry_work_refuses_without_dftd4(tmp_path, monkeypatch):
    import vibeqc.runner as runner

    monkeypatch.setattr(runner, "dftd4_available", lambda: False)
    with pytest.raises(ImportError, match="D4 gradient at every geometry"):
        vq.run_job(_water(), basis="sto-3g", method="rks", functional="pbe0",
                   dispersion="d4", optimize=True, output=tmp_path / "w",
                   output_qvf=False, citations=False, progress=False)
