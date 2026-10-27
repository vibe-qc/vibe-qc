"""ECP molecules in solvent: gradient, optimization and frequencies.

The solvated gradient used to refuse every molecule with a pseudopotential, and
with it the optimizer and the finite-difference Hessian, so no heavy-element
solute (iodine, Pd, Pt, Zn ...) could be relaxed or have its frequencies
computed in solution. Nothing in the reaction field touches the ECP: the
nuclear ESP takes the effective charges and the electronic ESP is a plain
point-charge integral. The gas-phase piece is the ECP-aware analytic gradient.

The oracle is always the solvated energy the job itself reports.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq

_H = 1.19  # vDZP methane, bohr


def _model(kernel="point", **kw):
    return vq.SolventModel(
        epsilon=78.39,
        name="water",
        charge_representation=kernel,
        max_macro_iter=60,
        tol_e_solv=1e-11,
        **kw,
    )


def _znh(z_zn):
    return vq.Molecule([vq.Atom(30, [0, 0, z_zn]), vq.Atom(1, [0, 0, 3.0])], 1, 1)


def _znh_scf(z_zn, sm):
    mol = _znh(z_zn)
    basis = vq.BasisSet(mol, "6-31g")
    opts = vq.RHFOptions()
    opts.ecp_centers = [vq.ECPCenter(Z=30, xyz=[0, 0, z_zn])]
    opts.ecp_library = "ecp10mdf"
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.max_iter = 200
    opts.damping = 0.5
    return mol, basis, opts, vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm, options=opts)


def _ch4(z_c):
    atoms = [vq.Atom(6, [0, 0, z_c])]
    for p in ((_H, _H, _H), (-_H, -_H, _H), (-_H, _H, -_H), (_H, -_H, -_H)):
        atoms.append(vq.Atom(1, list(p)))
    return vq.Molecule(atoms, 0, 1)


def _hi(z_i):
    return vq.Molecule([vq.Atom(53, [0, 0, z_i]), vq.Atom(1, [0, 0, 3.0])], 0, 1)


def _solve(mol, basis_name, sm):
    basis = vq.BasisSet(mol, basis_name)
    return basis, vq.run_cpcm_scf(mol, basis, method="rhf", solvent=sm)


# --- the gradient ------------------------------------------------------------


def test_xml_library_ecp_gradient_matches_finite_differences():
    """[ZnH]+ (ecp10mdf): the solvent changes the Zn_z force from -0.0122 to
    -0.0033 Ha/bohr, so a gradient that ignored it would miss by 9e-3."""
    sm = _model()
    mol, basis, opts, sol = _znh_scf(0.0, sm)
    assert sol.converged and sol.scf.ecp_total_ncore == 10
    # No options: the gradient repeats the ECP the SCF result recorded.
    g_auto = np.asarray(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"))
    go = vq.GradientOptions()
    go.ecp_centers = opts.ecp_centers
    go.ecp_library = opts.ecp_library
    g_given = np.asarray(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf", options=go))
    np.testing.assert_allclose(g_auto, g_given, atol=1e-12)
    h = 5e-4
    fd = (_znh_scf(h, sm)[3].energy - _znh_scf(-h, sm)[3].energy) / (2 * h)
    assert g_auto[0, 2] == pytest.approx(fd, abs=1e-7)  # measured 9e-9
    assert abs(g_auto[0, 2] + 0.0122) > 5e-3  # the reaction field is in it
    assert np.abs(g_auto.sum(axis=0)).max() < 1e-10


def test_inline_primitive_ecp_gradient_matches_finite_differences():
    """vDZP carbon (ncore = 2) takes the inline route. The Gaussian kernel,
    because the point-charge energy has structure at the 5e-4 bohr step (#681)."""
    sm = _model("gaussian")

    def energy(z_c):
        return _solve(_ch4(z_c), "vdzp", sm)[1].energy

    mol = _ch4(0.3)
    basis, sol = _solve(mol, "vdzp", sm)
    assert sol.scf.ecp_total_ncore == 2 and len(sol.scf.ecp_primitive_blocks) == 1
    g = np.asarray(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"))
    h = 5e-4
    fd = (energy(0.3 + h) - energy(0.3 - h)) / (2 * h)
    assert g[0, 2] == pytest.approx(fd, abs=1e-6)  # measured 1.3e-8
    assert abs(g[0, 2]) > 0.1  # off the symmetric point, a real force


def test_iodine_ecp_gradient_matches_finite_differences():
    """HI in def2-SVP: iodine takes ecp28mdf (Z_eff = 25)."""
    sm = _model()
    mol = _hi(0.0)
    basis, sol = _solve(mol, "def2-svp", sm)
    assert sol.scf.ecp_total_ncore == 28 and sol.z_eff[0] == pytest.approx(25.0)
    g = np.asarray(vq.cpcm_gradient(sol.scf, mol, basis, sol, method="rhf"))
    h = 5e-4
    fd = (_solve(_hi(h), "def2-svp", sm)[1].energy - _solve(_hi(-h), "def2-svp", sm)[1].energy) / (2 * h)
    assert g[0, 2] == pytest.approx(fd, abs=1e-6)  # measured 2e-8


def test_the_cavity_and_the_scf_must_agree_on_the_ecp():
    from vibeqc.solvation.gradient import cpcm_gradient

    water = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 1.4, 1.1]), vq.Atom(1, [0.0, -1.4, 1.1])]
    )
    # Effective charges in the cavity, an all-electron SCF result.
    all_electron = SimpleNamespace(ecp_operator_applied=False, ecp_total_ncore=0)
    cavity_with_ecp = SimpleNamespace(z_eff=np.array([6.0, 1.0, 1.0]))
    with pytest.raises(NotImplementedError, match="ECP nuclear derivatives"):
        cpcm_gradient(all_electron, water, object(), cavity_with_ecp)
    # An ECP SCF result, bare charges in the cavity.
    ecp_scf = SimpleNamespace(ecp_operator_applied=True, ecp_total_ncore=2)
    bare_cavity = SimpleNamespace(z_eff=None)
    with pytest.raises(NotImplementedError, match="bare nuclear charges"):
        cpcm_gradient(ecp_scf, water, object(), bare_cavity)


# --- the optimizer -----------------------------------------------------------


def test_a_geometry_optimization_of_an_ecp_molecule_in_solvent_is_stationary():
    sm = _model()
    result = vq.optimize_molecule(
        _hi(0.0),
        "def2-svp",
        method="rhf",
        solvent=sm,
        max_iter=60,
    )
    assert result.converged
    z = [a.xyz[2] for a in result.system.atoms]
    axis = np.sign(z[1] - z[0])

    def energy(dz):
        m = vq.Molecule(
            [vq.Atom(53, list(result.system.atoms[0].xyz)),
             vq.Atom(1, [0.0, 0.0, z[1] + axis * dz])], 0, 1,
        )
        return _solve(m, "def2-svp", sm)[1].energy

    h = 1e-3
    slope = (energy(h) - energy(-h)) / (2 * h)
    assert abs(slope) < 5e-4  # Ha/bohr; the optimizer's own tolerance is looser


# --- the Hessian -------------------------------------------------------------


def test_the_solvated_hessian_of_an_ecp_molecule_reproduces_the_energy_curvature():
    """d^2E/dz_H^2 of the reported energy against the Hessian element, in gas
    phase and in solvent. The solvent raises the stretch force constant by
    9.4e-4 Ha/bohr^2; the Hessian matches the energy's curvature to 3e-8."""
    from vibeqc.hessian import compute_hessian_fd

    sm = _model()

    def curvature(solvated):
        def e(dz):
            m = vq.Molecule([vq.Atom(53, [0, 0, 0.0]), vq.Atom(1, [0, 0, 3.2 + dz])], 0, 1)
            basis = vq.BasisSet(m, "def2-svp")
            opts = vq.RHFOptions()
            opts.conv_tol_energy = 1e-12
            opts.conv_tol_grad = 1e-10
            opts.max_iter = 200
            if solvated:
                return vq.run_cpcm_scf(m, basis, method="rhf", solvent=sm, options=opts).energy
            return vq.run_rhf(m, basis, opts).energy

        h = 2e-3
        return (e(h) - 2 * e(0.0) + e(-h)) / h**2

    mol = vq.Molecule([vq.Atom(53, [0, 0, 0.0]), vq.Atom(1, [0, 0, 3.2])], 0, 1)
    # Only H moves: freezing I leaves the one 3x3 block that holds the stretch.
    fd_opts = vq.HessianFDOptions(step_bohr=2e-3, frozen_indices=[0])
    k = {}
    for solvated in (False, True):
        res = compute_hessian_fd(
            mol, "def2-svp", method="RHF",
            solvent=sm if solvated else None, hessian_options=fd_opts,
        )
        k[solvated] = float(np.asarray(res.hessian)[2, 2])  # H_z, H_z
        assert k[solvated] == pytest.approx(curvature(solvated), abs=2e-6)
    assert k[True] - k[False] > 5e-4  # the solvent is in the second derivative


def test_run_job_computes_frequencies_for_an_ecp_molecule_in_solvent(tmp_path):
    """Through run_job: no preflight refusal, the medium is named, and the
    mass table reaches the heavy atom (iodine used to fail 'no built-in mass')."""
    stem = tmp_path / "hi"
    vq.run_job(
        _hi(0.0),
        basis="def2-svp",
        method="rhf",
        solvent="water",
        hessian=True,
        output=str(stem),
        verbose=0,
        write_molden_file=False,
    )
    text = stem.with_suffix(".out").read_text()
    assert "in CPCM water" in text
    assert "Freq/cm" in text
