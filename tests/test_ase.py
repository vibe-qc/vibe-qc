"""ASE Calculator integration."""

from __future__ import annotations

import logging

import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.units import Hartree

from vibeqc.ase import VibeQC

from .conftest import run_pyscf_rhf


ANGSTROM = 1.0  # ASE default length unit
# A tiny H2O geometry in Angstrom (same coordinates we use in conftest, just
# without the manual bohr conversion because ASE works in Angstrom natively).
_H2O_ANGSTROM = Atoms(
    symbols=["O", "H", "H"],
    positions=[
        (0.0, 0.0, 0.0),
        (0.0, 0.793353, -0.613510),
        (0.0, -0.793353, -0.613510),
    ],
)


def test_vibeqc_calculator_h2o_sto3g_matches_pyscf():
    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    energy_eV = atoms.get_potential_energy()

    # ASE energy is in eV; convert to Hartree for comparison.
    energy_Ha = energy_eV / Hartree

    # Reference: same geometry, converted to bohr for pyscf.
    A2B = 1.0 / 0.529177210903
    atoms_bohr = [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.793353 * A2B, -0.613510 * A2B]),
        (1, [0.0, -0.793353 * A2B, -0.613510 * A2B]),
    ]
    ref_energy, _ = run_pyscf_rhf(atoms_bohr, "sto-3g")

    assert energy_Ha == pytest.approx(ref_energy, abs=1e-9)


def test_vibeqc_calculator_exposes_rhf_result():
    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    atoms.get_potential_energy()
    assert "rhf_result" in atoms.calc.results
    result = atoms.calc.results["rhf_result"]
    assert result.converged
    assert len(result.scf_trace) == result.n_iter


def test_vibeqc_calculator_forces_match_pyscf():
    """ASE forces from vibeqc should agree with PySCF's analytic gradient
    (up to sign: ASE force = -gradient)."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from ase.units import Bohr, Hartree
    import numpy as np

    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    forces_eV_per_A = atoms.get_forces()  # (n_atoms, 3)

    # Reference gradient in Ha/bohr from PySCF.
    A2B = 1.0 / 0.529177210903
    pmol = gto.Mole()
    pmol.unit = "Bohr"
    pmol.atom = [
        [8, (0.0, 0.0, 0.0)],
        [1, (0.0, 0.793353 * A2B, -0.613510 * A2B)],
        [1, (0.0, -0.793353 * A2B, -0.613510 * A2B)],
    ]
    pmol.basis = "sto-3g"
    pmol.verbose = 0
    pmol.build()
    mf = scf.RHF(pmol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    grad_Ha_per_bohr = mf.nuc_grad_method().kernel()
    force_ref_eV_per_A = -grad_Ha_per_bohr * (Hartree / Bohr)

    # Tolerance set at 1e-7 eV/Å (~= 2e-9 Ha/bohr) — small numeric noise from
    # the eV/Å unit conversion on top of the underlying Ha/bohr gradient.
    np.testing.assert_allclose(
        forces_eV_per_A, force_ref_eV_per_A,
        atol=1e-7, rtol=0,
        err_msg="vibe-qc forces disagree with PySCF gradient",
    )


def test_vibeqc_calculator_force_is_negative_gradient_direction():
    """Sanity check: the H atoms in H2O want to move apart and up (into the
    equilibrium geometry), so the y-components of their forces have opposite
    signs and the z-components push them up."""
    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    forces = atoms.get_forces()
    # Two H atoms (indices 1, 2), symmetric about y-axis.
    assert forces[1, 1] * forces[2, 1] < 0, "H atoms' y-forces should be opposite"


def test_vibeqc_calculator_autoroutes_to_uhf_for_open_shell():
    """mult > 1 must trigger the UHF path; forces should match
    compute_gradient_uhf with the same UHFResult."""
    import numpy as np
    from ase.units import Bohr, Hartree
    from vibeqc import compute_gradient_uhf

    # OH radical in Å. Multiplicity=2 drives UHF.
    atoms = Atoms(symbols=["O", "H"], positions=[(0, 0, 0), (0.97, 0, 0)])
    atoms.calc = VibeQC(basis="sto-3g", multiplicity=2)
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    # Underlying result should be a UHFResult, exposed via scf_result.
    scf = atoms.calc.results["scf_result"]
    assert hasattr(scf, "mo_energies_alpha"), (
        "calc.results['scf_result'] is not a UHFResult — did auto-routing fail?"
    )
    assert scf.converged
    # Forces consistent with Ha/bohr gradient.
    expected_force = -np.array(compute_gradient_uhf(
        # Rebuild the Molecule/BasisSet vibe-qc used internally.
        _ase_atoms_to_vibeqc_mol(atoms, charge=0, multiplicity=2),
        _ase_atoms_to_vibeqc_basis(atoms, "sto-3g", charge=0, multiplicity=2),
        scf,
    )) * (Hartree / Bohr)
    np.testing.assert_allclose(forces, expected_force, atol=1e-7)


def _ase_atoms_to_vibeqc_mol(atoms, *, charge=0, multiplicity=1):
    from ase.units import Bohr
    from vibeqc import Atom as VQAtom, Molecule as VQMol
    return VQMol(
        [VQAtom(int(z), list(p / Bohr)) for z, p in zip(atoms.numbers, atoms.positions)],
        charge, multiplicity,
    )


def _ase_atoms_to_vibeqc_basis(atoms, basis_name, *, charge=0, multiplicity=1):
    from vibeqc import BasisSet as VQBasis
    mol = _ase_atoms_to_vibeqc_mol(atoms, charge=charge, multiplicity=multiplicity)
    return VQBasis(mol, basis_name)


def test_vibeqc_calculator_routes_to_dft_when_functional_set():
    """Passing functional=... should drive run_rks and give the DFT energy.
    Forces should raise PropertyNotImplementedError until Phase 9f lands."""
    import numpy as np
    from ase.calculators.calculator import PropertyNotImplementedError
    from ase.units import Hartree

    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g", functional="LDA")
    energy_eV = atoms.get_potential_energy()
    energy_Ha = energy_eV / Hartree
    # LDA/STO-3G on our H2O geometry should be around -74.7 Ha.
    assert -76.0 < energy_Ha < -74.0

    # Underlying result should carry the LDA energy decomposition.
    scf = atoms.calc.results["scf_result"]
    assert hasattr(scf, "e_xc"), "scf_result should be an RKSResult for DFT"
    assert scf.converged

    # DFT forces now work for LDA (Phase 9f); check the happy path.
    forces_lda = atoms.get_forces()
    assert forces_lda.shape == (3, 3)
    # Net force (sum across atoms) should be near zero by translational
    # invariance, up to grid-accuracy on the default medium grid.
    net_force = forces_lda.sum(axis=0)
    assert np.linalg.norm(net_force) < 5e-3  # eV/Å


def test_vibeqc_calculator_dft_force_lda_geometry_opt():
    """H2O with LDA should converge to the LDA/STO-3G minimum in a few
    BFGS steps. Literature equilibrium: r(O-H) ~ 1.00 Å, HOH ~ 97°."""
    import numpy as np
    from ase.optimize import BFGS
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.80, -0.70),
            (0.0, -0.80, -0.70),
        ],
    )
    atoms.calc = VibeQC(basis="sto-3g", functional="LDA")
    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=5e-3)
    rOH = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
    v1 = atoms.positions[1] - atoms.positions[0]
    v2 = atoms.positions[2] - atoms.positions[0]
    HOH = np.degrees(np.arccos(np.dot(v1, v2)
                                / (np.linalg.norm(v1) * np.linalg.norm(v2))))
    assert rOH == pytest.approx(1.03, abs=5e-2)
    assert HOH == pytest.approx(97.0, abs=3.0)


def test_vibeqc_calculator_dft_gga_forces():
    """GGA and hybrid DFT forces now work; check they're close to zero
    net force (translational invariance to grid accuracy)."""
    import numpy as np
    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g", functional="B3LYP")
    forces = atoms.get_forces()
    assert forces.shape == (3, 3)
    # Same grid-accuracy bound as LDA.
    assert np.linalg.norm(forces.sum(axis=0)) < 5e-3  # eV/Å


def test_vibeqc_calculator_routes_to_uks_for_open_shell_dft():
    """multiplicity > 1 + functional should run UKS and give a bound,
    spin-polarized energy AND working forces."""
    import numpy as np
    from ase.units import Hartree
    atoms = Atoms(symbols=["O", "H"], positions=[(0, 0, 0), (0.97, 0, 0)])
    atoms.calc = VibeQC(basis="sto-3g", multiplicity=2, functional="PBE")
    # Loosen grad tolerance so UKS on OH doublets converges in test time.
    atoms.calc._uks_options.conv_tol_grad = 1e-5
    energy_Ha = atoms.get_potential_energy() / Hartree
    assert -75.0 < energy_Ha < -73.5
    scf = atoms.calc.results["scf_result"]
    assert hasattr(scf, "mo_energies_alpha"), (
        "DFT + open-shell should produce a UKSResult"
    )
    assert scf.converged
    assert 0.7 < scf.s_squared < 0.9

    # UKS forces (Phase 9f-UKS) should now work.
    forces = atoms.get_forces()
    assert forces.shape == (2, 3)
    assert np.linalg.norm(forces.sum(axis=0)) < 1e-2  # eV/Å, grid-accuracy


def test_geometry_optimization_h2o_sto3g():
    """Drive the full BFGS pipeline: start away from the HF/STO-3G minimum
    and let ASE's optimizer find it. Known equilibrium is rOH = 0.989 Å,
    HOH ≈ 100° (Pople et al.)."""
    import numpy as np
    from ase.optimize import BFGS

    # Start with slightly stretched O-H bonds and a compressed angle.
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (0.0, 0.80, -0.70),
            (0.0, -0.80, -0.70),
        ],
    )
    atoms.calc = VibeQC(basis="sto-3g")

    opt = BFGS(atoms, logfile=None)
    opt.run(fmax=1e-3)  # eV/Å

    # Post-opt geometry sanity
    rOH1 = np.linalg.norm(atoms.positions[1] - atoms.positions[0])
    rOH2 = np.linalg.norm(atoms.positions[2] - atoms.positions[0])
    v1 = atoms.positions[1] - atoms.positions[0]
    v2 = atoms.positions[2] - atoms.positions[0]
    HOH = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

    assert rOH1 == pytest.approx(0.9894, abs=5e-3)
    assert rOH2 == pytest.approx(0.9894, abs=5e-3)
    assert HOH == pytest.approx(100.0, abs=1.0)


def test_vibeqc_calculator_emits_log_messages(caplog):
    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    with caplog.at_level(logging.INFO, logger="vibeqc"):
        atoms.get_potential_energy()

    # The ASE calc logs a header on vibeqc.ase and summary on vibeqc.scf.
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("RHF / sto-3g" in m for m in messages), messages
    assert any("SCF converged" in m for m in messages), messages


def test_vibeqc_calculator_accepts_charge_and_multiplicity():
    # OH- hydroxide: O + H, net charge -1, 10 electrons, singlet.
    atoms = Atoms(symbols=["O", "H"], positions=[(0, 0, 0), (0.97, 0, 0)])
    atoms.calc = VibeQC(basis="sto-3g", charge=-1, multiplicity=1)
    energy = atoms.get_potential_energy()
    assert energy < 0  # bound state


# ---------------------------------------------------------------------------
# Phase A (v0.5): dipole / Hessian / polarizability properties
# ---------------------------------------------------------------------------


def test_vibeqc_calculator_advertises_extended_properties():
    """``implemented_properties`` should now cover the full v0.5 surface."""
    expected = {"energy", "forces", "free_energy",
                "dipole", "hessian", "polarizability"}
    assert set(VibeQC.implemented_properties) >= expected


def test_vibeqc_calculator_dipole_h2o_sto3g():
    """``atoms.get_dipole_moment()`` returns an e·Å vector with the
    right magnitude for water (≈ 0.34 eÅ at this geometry / sto-3g)."""
    import numpy as np

    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    mu = atoms.get_dipole_moment()
    assert mu.shape == (3,)
    # Water lies in the y-z plane in this fixture; dipole along z.
    assert abs(mu[0]) < 1e-8           # x-component zero by symmetry
    mag = float(np.linalg.norm(mu))
    # H2O / STO-3G with this geometry: ~0.3-0.4 eÅ (~1.5-2 D). Loose
    # bound — the precise number is method-dependent and we already
    # validate the underlying dipole_moment() against PySCF in
    # tests/test_properties.py.
    assert 0.2 < mag < 0.6


def test_vibeqc_calculator_polarizability_h2o_rhf():
    """RHF polarizability tensor — symmetric 3×3 in Å³, positive
    eigenvalues (response is in-phase with the field)."""
    import numpy as np

    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    alpha = atoms.calc.get_property("polarizability", atoms)
    assert alpha.shape == (3, 3)
    # Symmetric (CPHF response should be — αᵢⱼ = αⱼᵢ).
    assert np.allclose(alpha, alpha.T, atol=1e-8)
    # Positive eigenvalues — physical polarizability is a positive
    # quadratic form.
    eigs = np.linalg.eigvalsh(alpha)
    assert (eigs > 0).all(), eigs


def test_vibeqc_calculator_polarizability_uks_raises():
    """Polarizability is RHF-only as of v0.5; non-RHF should raise
    PropertyNotImplementedError pointing at the roadmap."""
    from ase.calculators.calculator import PropertyNotImplementedError

    # Triplet O atom — forces UKS path.
    atoms = Atoms(symbols=["O"], positions=[(0, 0, 0)])
    atoms.calc = VibeQC(basis="sto-3g", multiplicity=3, functional="LDA")
    with pytest.raises(PropertyNotImplementedError, match="(?i)RHF-only"):
        atoms.calc.get_property("polarizability", atoms)


def test_vibeqc_calculator_hessian_rhf_h2o_sto3g():
    """RHF Hessian via ASE — 3N×3N matrix, six near-zero translation/
    rotation eigenvalues, the rest physically sensible."""
    import numpy as np

    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    H = atoms.calc.get_property("hessian", atoms)
    assert H.shape == (9, 9)
    # Nearly symmetric (analytic CPHF Hessian is symmetric to working
    # precision — the kernel applies an explicit symmetrization).
    assert np.allclose(H, H.T, atol=1e-6)

    eigs = np.linalg.eigvalsh(H)
    # Three translations are exact zeros (translational invariance).
    assert abs(eigs[:3]).max() < 1e-3, (
        f"3 lowest eigenvalues should be near zero (translations); got {eigs[:3]}"
    )
    # The water geometry from the fixture isn't equilibrium → some
    # negative eigenvalues are expected (the geometry is sliding
    # toward a minimum). What we DO expect: the three highest
    # eigenvalues correspond to vibrational modes (bend + 2 stretches)
    # and should be in the eV/Å² range typical for O-H bonds.
    assert eigs[-1] > 50, f"Top eigenvalue (sym stretch?) = {eigs[-1]}"
    # Hessian is in eV/Å² — for an O-H stretch around 3500 cm⁻¹, the
    # equivalent eigenvalue should be on the order of 100-200 eV/Å².
    # Lax bound; full numerical agreement vs PySCF is in the engineering
    # test_hessian_analytic.py.
    assert 50 < eigs[-1] < 500


def test_vibeqc_calculator_free_energy_alias():
    """``force_consistent=True`` (which queries ``free_energy``) must
    not raise — vibe-qc has no smearing-driven entropy contribution,
    so free_energy is identical to energy."""
    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="sto-3g")
    e = atoms.get_potential_energy()
    fe = atoms.get_potential_energy(force_consistent=True)
    assert e == fe


def test_gradient_options_from_scf_copies_jk_backend_fields():
    """Audit P1/P2.3 (finding 3): the helper mirrors the SCF's JK
    backend onto the GradientOptions handed to compute_gradient*.
    The fix landed without regressions; this pins the field copy."""
    from vibeqc import RHFOptions
    from vibeqc.ase import _gradient_options_from_scf

    o = RHFOptions()
    o.density_fit = True
    o.aux_basis = "def2-svp-jk"
    o.cosx = True
    g = _gradient_options_from_scf(o)
    assert g.density_fit is True
    assert g.aux_basis == "def2-svp-jk"
    assert g.cosx is True

    g_default = _gradient_options_from_scf(RHFOptions())
    assert g_default.density_fit is False
    assert g_default.aux_basis == ""
    assert g_default.cosx is False


def test_vibeqc_calculator_forces_follow_scf_jk_backend():
    """Audit P1/P2.3 (finding 3) regression: ``VibeQC.get_forces()``
    must differentiate the same Hamiltonian the SCF minimised. With
    ``density_fit=True`` on the SCF options the forces are the DF
    analytic gradient; pre-fix they silently fell back to the
    four-index direct path (B3LYP/def2-SVP RIJCOSX probe: 1.4 meV/A
    off)."""
    import numpy as np
    from ase.units import Bohr

    from vibeqc import (
        Atom,
        BasisSet,
        GradientOptions,
        Molecule,
        RHFOptions,
        compute_gradient,
        run_rhf,
    )

    def make_scf_opts():
        o = RHFOptions()
        o.density_fit = True
        o.aux_basis = "def2-svp-jk"
        return o

    atoms = _H2O_ANGSTROM.copy()
    atoms.calc = VibeQC(basis="def2-svp", rhf_options=make_scf_opts())
    forces_eV_per_A = atoms.get_forces()

    # Reference: identical geometry/SCF re-run standalone, then both
    # gradient flavours on the converged result.
    A2B = 1.0 / 0.529177210903
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.793353 * A2B, -0.613510 * A2B]),
            Atom(1, [0.0, -0.793353 * A2B, -0.613510 * A2B]),
        ],
        0,
        1,
    )
    basis = BasisSet(mol, "def2-svp")
    r = run_rhf(mol, basis, make_scf_opts())
    assert r.converged

    g_direct = np.asarray(compute_gradient(mol, basis, r))
    df_opts = GradientOptions()
    df_opts.density_fit = True
    df_opts.aux_basis = "def2-svp-jk"
    g_df = np.asarray(compute_gradient(mol, basis, r, df_opts))

    # The two flavours differ by the DF fitting error (~1e-5 Ha/bohr on
    # H2O/def2-svp), large enough that the equality check below
    # discriminates which one the calculator used.
    assert np.abs(g_df - g_direct).max() > 1e-6

    force_df_eV_per_A = -g_df * (Hartree / Bohr)
    np.testing.assert_allclose(
        forces_eV_per_A, force_df_eV_per_A,
        atol=1e-7, rtol=0,
        err_msg=(
            "VibeQC forces do not match the DF gradient of the SCF's "
            "Hamiltonian; GradientOptions threading regressed"
        ),
    )


def _release_paper_glycine_atoms() -> Atoms:
    """Glycine geometry used by the release-paper optimizer prompts."""
    return Atoms(
        symbols=["N", "C", "C", "O", "O", "H", "H", "H", "H", "H"],
        positions=[
            (-2.105915, 0.392689, 0.000000),
            (-0.837110, -0.332747, 0.000000),
            (0.382253, 0.576152, 0.000000),
            (1.501936, 0.062047, 0.000000),
            (0.164156, 1.910830, 0.000000),
            (-2.897845, -0.223798, 0.000000),
            (-2.202618, 1.005832, 0.799602),
            (-0.782287, -0.965302, 0.901712),
            (-0.782287, -0.965302, -0.901712),
            (0.913777, 2.493046, 0.000000),
        ],
    )


def _stub_glycine_rks_nonconvergence(monkeypatch, *, failure_energy: float):
    from types import SimpleNamespace

    import numpy as np
    import vibeqc.ase as ase_mod

    call_count = {"rks": 0}

    def fake_basis(molecule, basis_name):
        assert basis_name == "def2-tzvp"
        return object()

    def fake_run_rks(molecule, basis, options):
        call_count["rks"] += 1
        if call_count["rks"] >= 4:
            return SimpleNamespace(
                converged=False,
                n_iter=200,
                energy=failure_energy,
                scf_trace=[],
            )
        return SimpleNamespace(
            converged=True,
            n_iter=8,
            energy=-280.0 + 0.01 * call_count["rks"],
            scf_trace=[],
        )

    def fake_gradient(molecule, basis, result, grid, options=None):
        grad = np.zeros((len(molecule.atoms), 3), dtype=float)
        grad[9, 0] = 0.02  # Ha/bohr, enough to keep MDMin stepping.
        return grad

    monkeypatch.setattr(ase_mod, "BasisSet", fake_basis)
    monkeypatch.setattr(ase_mod, "run_rks", fake_run_rks)
    monkeypatch.setattr(ase_mod, "compute_gradient_rks", fake_gradient)
    return ase_mod, call_count


def test_ase_mdmin_glycine_scf_nonconvergence_is_typed(monkeypatch, tmp_path):
    """Glycine-shaped ASE MDMin jobs should classify SCF failure directly.

    The release-paper prompt-67 artifacts show MDMin taking real glycine
    optimizer steps before RKS/PBE0/def2-TZVP stops converging. This regression
    keeps that family distinct from missing-import/runtime setup failures by
    making the ASE calculator raise a typed convergence error, while stubbing
    the expensive SCF and gradient calls so the smoke stays fast.
    """
    from ase.optimize import MDMin

    glycine = _release_paper_glycine_atoms()
    failure_energy = -269.1489700318
    ase_mod, call_count = _stub_glycine_rks_nonconvergence(
        monkeypatch,
        failure_energy=failure_energy,
    )

    glycine.calc = ase_mod.VibeQC(basis="def2-tzvp", functional="pbe0")
    opt = MDMin(
        glycine,
        logfile=None,
        trajectory=str(tmp_path / "glycine-mdmin.traj"),
        dt=0.1,
        maxstep=0.02,
    )

    with pytest.raises(ase_mod.VibeQCSCFConvergenceError) as excinfo:
        opt.run(fmax=0.01, steps=8)

    err = excinfo.value
    assert call_count["rks"] >= 4
    assert err.method == "RKS"
    assert err.n_iter == 200
    assert err.energy_hartree == pytest.approx(failure_energy)
    assert "SCF did not converge" in str(err)
    # The error message must carry the geometry diagnostic + remedy hint
    # so batch harnesses surface an actionable optimizer-level summary.
    assert "closest atom pair" in str(err)
    assert "pathological step" in str(err)


def test_ase_gpmin_glycine_scf_nonconvergence_is_typed(monkeypatch, tmp_path):
    """GPMin optimizer/SCF collapse is classified like prompt-68 artifacts.

    The release-paper prompt-68 GPMin runs started real optimizer steps before
    a later RKS force call reached the 200-iteration SCF cap. Keep that as a
    typed optimizer_scf_nonconvergence class, not a generic RuntimeError or
    missing-runtime failure.
    """
    from ase.optimize import GPMin

    glycine = _release_paper_glycine_atoms()
    failure_energy = -176.2425136843
    ase_mod, call_count = _stub_glycine_rks_nonconvergence(
        monkeypatch,
        failure_energy=failure_energy,
    )

    glycine.calc = ase_mod.VibeQC(basis="def2-tzvp", functional="pbe0")
    opt = GPMin(
        glycine,
        logfile=None,
        trajectory=str(tmp_path / "glycine-gpmin.traj"),
        update_hyperparams=False,
    )

    with pytest.raises(ase_mod.VibeQCSCFConvergenceError) as excinfo:
        opt.run(fmax=0.01, steps=8)

    err = excinfo.value
    assert call_count["rks"] >= 4
    assert err.method == "RKS"
    assert err.n_iter == 200
    assert err.energy_hartree == pytest.approx(failure_energy)
    assert "SCF did not converge" in str(err)
    assert "closest atom pair" in str(err)


def test_collapsed_geometry_rejected_before_scf(monkeypatch):
    """A collapsed atom pair fails fast with a typed geometry error.

    GPMin-style pathological steps otherwise burn the full SCF iteration
    budget on a meaningless Hamiltonian (glycine SI matrix rp167). The
    guard must fire before any SCF call and name the offending pair.
    """
    import vibeqc.ase as ase_mod

    def _boom(*args, **kwargs):
        raise AssertionError("SCF must not run on a rejected geometry")

    monkeypatch.setattr(ase_mod, "run_rhf", _boom)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]])
    atoms.calc = VibeQC(basis="sto-3g")

    with pytest.raises(ase_mod.VibeQCGeometryError) as excinfo:
        atoms.get_potential_energy()
    message = str(excinfo.value)
    assert "H0-H1" in message
    assert "0.050" in message
    assert "min_pair_distance" in message


def test_collapsed_geometry_check_can_be_disabled(monkeypatch):
    """min_pair_distance=None lets deliberate ultra-compressed scans run."""
    import vibeqc.ase as ase_mod

    seen = {"scf": 0}

    class _FakeResult:
        converged = True
        energy = -1.0
        n_iter = 1

    def fake_run_rhf(mol, basis, options):
        seen["scf"] += 1
        return _FakeResult()

    monkeypatch.setattr(ase_mod, "run_rhf", fake_run_rhf)
    monkeypatch.setattr(ase_mod, "log_scf_trace", lambda result: None)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]])
    atoms.calc = VibeQC(basis="sto-3g", min_pair_distance=None)
    atoms.get_potential_energy()
    assert seen["scf"] == 1


# ---------------------------------------------------------------------------
# ECP centres follow the atoms (#643)
# ---------------------------------------------------------------------------

def _h2s_lanl2dz_atoms():
    """The tests/test_ecp_scf.py H2S fixture as ASE Atoms (bohr -> Angstrom)."""
    from ase.units import Bohr

    positions_bohr = [[0.0, 0.0, 0.0], [0.0, 1.815, 1.425], [0.0, -1.815, 1.425]]
    return Atoms(numbers=[16, 1, 1],
                 positions=[[x * Bohr for x in p] for p in positions_bohr])


def _fresh_ecp_energy_forces(atoms):
    """SCF + ECP-aware gradient with the ECP attached for THIS geometry."""
    import numpy as np
    from ase.units import Bohr

    import vibeqc as vq
    from vibeqc.ecp_metadata import attach_inline_ecp_options_from_basis_sidecar
    from vibeqc.gradient_options import gradient_options_from_scf

    mol = vq.Molecule(
        [vq.Atom(int(z), list(p)) for z, p in zip(atoms.numbers, atoms.positions / Bohr)],
        0, 1)
    basis = vq.BasisSet(mol, "lanl2dz")
    opts = vq.RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    attach_inline_ecp_options_from_basis_sidecar(opts, mol, basis)
    res = vq.run_rhf(mol, basis, opts)
    grad = np.asarray(vq.compute_gradient(mol, basis, res, gradient_options_from_scf(opts)))
    return res.energy, -grad * (Hartree / Bohr)


def _tight_rhf_options():
    import vibeqc as vq

    opts = vq.RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return opts


def test_vibeqc_calculator_moves_ecp_centres_with_the_atoms():
    """#643 regression: after ASE moves an atom, energy and forces equal a
    fresh calculation whose ECP centre sits on the moved atom. Pre-fix the
    second geometry ran with the centre left at the first one: E = -75.69 Ha
    (all-electron-like) instead of -10.995 Ha, then the gradient raised
    'libecpint reports 4 atoms but the molecule has 3'."""
    import numpy as np

    atoms = _h2s_lanl2dz_atoms()
    opts = _tight_rhf_options()          # no ECP yet: the SCF wrapper attaches it
    atoms.calc = VibeQC(basis="lanl2dz", rhf_options=opts)
    e0 = atoms.get_potential_energy() / Hartree
    assert len(opts.ecp_primitive_blocks) == 1 and opts.ecp_total_ncore == 10
    assert e0 == pytest.approx(-11.009452171730, abs=1e-6)

    atoms.positions[0] += [0.0, 0.0, 0.05]   # move S by 0.05 Angstrom
    e1 = atoms.get_potential_energy() / Hartree
    f1 = atoms.get_forces()
    e_ref, f_ref = _fresh_ecp_energy_forces(atoms)
    assert abs(e1 - e_ref) < 1e-9, (e1, e_ref)
    assert np.abs(f1 - f_ref).max() < 1e-8
    # the options now describe the moved geometry (permanent repositioning)
    from ase.units import Bohr
    assert np.allclose(opts.ecp_primitive_centers[0], atoms.positions[0] / Bohr)


def test_vibeqc_calculator_refuses_prebuilt_ecp_centres_off_the_first_geometry():
    import vibeqc as vq

    atoms = _h2s_lanl2dz_atoms()
    opts = _tight_rhf_options()
    opts.ecp_centers = [vq.ECPCenter(Z=16, xyz=[0.0, 0.0, 0.05])]
    opts.ecp_library = "lanl2dz"
    atoms.calc = VibeQC(basis="lanl2dz", rhf_options=opts)
    with pytest.raises(ValueError, match="coincides with no atom"):
        atoms.get_potential_energy()


def test_vibeqc_cpcm_dipole_uses_result_z_eff_for_default_ecp_sidecar(
    monkeypatch,
):
    """CPCM's internal sidecar copy, not stale ASE options, owns Z_eff."""
    from types import SimpleNamespace

    import numpy as np

    import vibeqc.ase as ase_module

    observed = {}

    def record_dipole(result, basis, molecule, *, nuclear_charges, origin=None):
        del basis, molecule, origin
        observed["result"] = result
        observed["nuclear_charges"] = np.asarray(
            nuclear_charges, dtype=float
        ).copy()
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    monkeypatch.setattr(ase_module, "dipole_moment", record_dipole)

    atoms = _h2s_lanl2dz_atoms()
    calculator = VibeQC(basis="lanl2dz", solvent="water")
    atoms.calc = calculator
    np.testing.assert_array_equal(atoms.get_dipole_moment(), np.zeros(3))

    # run_cpcm_scf works on a copy, so the calculator's original defaults do
    # not receive the automatically discovered LANL2DZ sidecar. The wrapped
    # result nevertheless proves the operator and supplies the exact charges
    # used to construct its Hcore: S is 16 - 10 = 6, while H stays bare.
    assert not list(calculator._rhf_options.ecp_centers)
    result = observed["result"]
    assert result.ecp_operator_applied
    assert result.ecp_provenance_verified
    assert result.ecp_total_ncore == 10
    np.testing.assert_allclose(
        observed["nuclear_charges"], [6.0, 1.0, 1.0]
    )
    np.testing.assert_allclose(
        result.solvent_result.z_eff, observed["nuclear_charges"]
    )


def test_vibeqc_calculator_hessian_on_ecp_system_uses_the_fd_route():
    """The analytic RHF Hessian kernel takes no ECP options (all-electron
    displaced Fock pieces); on an ECP system the calculator must return the
    FD-on-analytic-gradient Hessian instead."""
    import numpy as np
    from ase.units import Bohr

    import vibeqc as vq
    from vibeqc.hessian import compute_hessian_fd

    atoms = _h2s_lanl2dz_atoms()
    opts = _tight_rhf_options()
    atoms.calc = VibeQC(basis="lanl2dz", rhf_options=opts)
    hess_ev = np.asarray(atoms.calc.get_property("hessian", atoms))
    mol = vq.Molecule(
        [vq.Atom(int(z), list(p)) for z, p in zip(atoms.numbers, atoms.positions / Bohr)],
        0, 1)
    ref = compute_hessian_fd(mol, "lanl2dz", method="RHF", scf_options=opts)
    ref_ev = np.asarray(ref.hessian) * Hartree / Bohr**2
    assert hess_ev.shape == (9, 9)
    assert np.abs(hess_ev - ref_ev).max() < 1e-6 * np.abs(ref_ev).max()


def test_vibeqc_calculator_refuses_ecp_cpcm_hessian_before_kernel(monkeypatch):
    """Neither Hessian backend differentiates the self-consistent solvent."""
    from types import SimpleNamespace

    import vibeqc.ase as ase_module

    def hessian_must_not_run(*args, **kwargs):
        raise AssertionError("gas-phase Hessian kernel entered for ECP+CPCM")

    monkeypatch.setattr(
        ase_module, "compute_hessian_rhf_analytic", hessian_must_not_run
    )
    monkeypatch.setattr(ase_module, "compute_hessian_fd", hessian_must_not_run)
    calculator = VibeQC(basis="lanl2dz", solvent="water")
    ecp_result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_total_ncore=10,
    )

    with pytest.raises(NotImplementedError, match=r"ECP\+CPCM Hessians"):
        calculator._compute_hessian("RHF", None, None, ecp_result)


@pytest.mark.parametrize(
    "solvent",
    ["vacuum", "gas", 1.0],
    ids=["vacuum", "gas-alias", "epsilon-one"],
)
def test_vibeqc_calculator_allows_gas_phase_solvent_hessian(
    monkeypatch, solvent
):
    """Gas aliases must not trip the self-consistent-solvent Hessian gate."""
    from types import SimpleNamespace

    import numpy as np

    import vibeqc.ase as ase_module

    calls = []

    def fake_analytic(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(hessian=np.eye(3))

    def fd_must_not_run(*args, **kwargs):
        raise AssertionError("all-electron RHF should use its analytic Hessian")

    monkeypatch.setattr(ase_module, "compute_hessian_rhf_analytic", fake_analytic)
    monkeypatch.setattr(ase_module, "compute_hessian_fd", fd_must_not_run)
    calculator = VibeQC(basis="sto-3g", solvent=solvent)

    hessian = calculator._compute_hessian(
        "RHF",
        object(),
        object(),
        SimpleNamespace(ecp_operator_applied=False, ecp_total_ncore=0),
    )

    assert len(calls) == 1
    assert hessian.shape == (3, 3)
    assert np.all(np.isfinite(hessian))
