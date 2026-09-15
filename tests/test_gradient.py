"""Analytic RHF nuclear gradient cross-checked against PySCF."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    GridOptions,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    compute_gradient,
    compute_gradient_rks,
    compute_gradient_uhf,
    compute_gradient_uks,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)

from .conftest import GEOMETRIES


def _pyscf_gradient(atoms_bohr, basis_name):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf.nuc_grad_method().kernel()


def _vibeqc_gradient(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    result = run_rhf(mol, basis, opts)
    return np.array(compute_gradient(mol, basis, result))


GRADIENT_CASES = [
    ("H2",  "sto-3g"),
    ("H2",  "6-31g*"),
    ("H2O", "sto-3g"),
    ("H2O", "6-31g*"),
    ("CH4", "sto-3g"),
]


@pytest.mark.parametrize(
    "mol_key,basis_name",
    GRADIENT_CASES,
    ids=[f"{m}-{b}" for m, b in GRADIENT_CASES],
)
def test_gradient_matches_pyscf(mol_key, basis_name):
    atoms = GEOMETRIES[mol_key]
    grad_vq = _vibeqc_gradient(atoms, basis_name)
    grad_ps = _pyscf_gradient(atoms, basis_name)
    assert grad_vq.shape == grad_ps.shape == (len(atoms), 3)
    np.testing.assert_allclose(
        grad_vq, grad_ps,
        atol=1e-9, rtol=0,
        err_msg=f"gradient disagrees with PySCF for {mol_key}/{basis_name}",
    )


def test_gradient_zero_at_optimized_h2_geometry(tight_rhf_opts):
    """At the HF/STO-3G optimum of H2, the gradient vanishes. We use PySCF's
    equilibrium bond length (around 1.346 bohr for STO-3G) and expect the
    gradient at that geometry to be near zero."""
    # Known HF/STO-3G minimum for H2 is near R = 1.346 bohr.
    R_opt = 1.346
    atoms = [(1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, R_opt])]
    grad = _vibeqc_gradient(atoms, "sto-3g")
    # Small residual allowed because R_opt is only accurate to 3 decimals.
    assert np.max(np.abs(grad)) < 1e-3


def test_gradient_errors_on_unconverged_result(tight_rhf_opts):
    """compute_gradient must refuse an RHFResult that hasn't converged."""
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.max_iter = 1  # guaranteed not to converge
    opts.use_diis = False
    result = run_rhf(mol, basis, opts)
    assert not result.converged
    with pytest.raises(RuntimeError, match="not converged"):
        compute_gradient(mol, basis, result)


def test_gradient_translational_invariance():
    """Σ_A dE/dR_A = 0 (translational invariance of the energy). Checks that
    our bookkeeping of basis-center + nuclear derivative contributions in
    the V integral is consistent."""
    atoms = GEOMETRIES["H2O"]
    grad = _vibeqc_gradient(atoms, "sto-3g")
    total = grad.sum(axis=0)
    np.testing.assert_allclose(total, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# UHF gradients (open-shell)
# ---------------------------------------------------------------------------

def _pyscf_uhf_gradient(atoms_bohr, basis_name, *, charge, spin):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf.nuc_grad_method().kernel()


def _vibeqc_uhf_gradient(atoms_bohr, basis_name, *, charge=0, multiplicity):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
                   charge=charge, multiplicity=multiplicity)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-8
    opts.max_iter = 500
    result = run_uhf(mol, basis, opts)
    return np.array(compute_gradient_uhf(mol, basis, result))


UHF_GRADIENT_CASES = [
    # (label, atoms_bohr, basis_name, multiplicity)
    ("H-doublet/STO-3G", [(1, [0.0, 0.0, 0.0])], "sto-3g", 2),
    ("OH-doublet/STO-3G",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 / 0.529177210903, 0.0, 0.0])], "sto-3g", 2),
    ("OH-doublet/6-31G*",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 / 0.529177210903, 0.0, 0.0])], "6-31g*", 2),
    ("O2-triplet/STO-3G",
     [(8, [0.0, 0.0, -0.6]), (8, [0.0, 0.0, 0.6])], "sto-3g", 3),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,mult",
    UHF_GRADIENT_CASES,
    ids=[c[0] for c in UHF_GRADIENT_CASES],
)
def test_uhf_gradient_matches_pyscf(label, atoms, basis_name, mult):
    grad_vq = _vibeqc_uhf_gradient(atoms, basis_name, multiplicity=mult)
    grad_ps = _pyscf_uhf_gradient(atoms, basis_name, charge=0, spin=mult - 1)
    assert grad_vq.shape == grad_ps.shape == (len(atoms), 3)
    # Gradient values on highly-compressed geometries (e.g. O2 at 1.2 bohr)
    # reach O(10) Ha/bohr; use a combined abs+rel tolerance so the test
    # stays meaningful at both scales.
    np.testing.assert_allclose(
        grad_vq, grad_ps,
        atol=1e-8, rtol=1e-8,
        err_msg=f"UHF gradient disagrees with PySCF for {label}",
    )


def test_uhf_gradient_matches_rhf_on_closed_shell():
    """UHF gradient on a closed-shell singlet must collapse to the RHF
    gradient (D_α = D_β = D/2, W_α = W_β = W/2; Γ_UHF reduces to Γ_RHF)."""
    atoms = GEOMETRIES["H2O"]
    grad_uhf = _vibeqc_uhf_gradient(atoms, "sto-3g", multiplicity=1)
    grad_rhf = _vibeqc_gradient(atoms, "sto-3g")
    np.testing.assert_allclose(grad_uhf, grad_rhf, atol=1e-10)


# ---------------------------------------------------------------------------
# RKS gradients (LDA only for now)
# ---------------------------------------------------------------------------

def _pyscf_rks_gradient(atoms_bohr, basis_name, xc):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = 5
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf.nuc_grad_method().kernel()


def _vibeqc_rks_gradient(atoms_bohr, basis_name, functional, *, grid_opts=None):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    if grid_opts is not None:
        opts.grid = grid_opts
    result = run_rks(mol, basis, opts)
    return np.array(compute_gradient_rks(mol, basis, result,
                                          opts.grid if grid_opts is None else grid_opts))


RKS_CASES = [
    ("H2O-sto-3g",
     [(8, [0, 0, 0]),
      (1, [0.0, 1.5, -1.16]),
      (1, [0.0, -1.5, -1.16])],
     "sto-3g"),
    ("H2-sto-3g", [(1, [0, 0, 0]), (1, [0, 0, 1.4])], "sto-3g"),
]
# vibe-qc's "B3LYP" = ORCA/VWN5 definition; PySCF's matching spelling
# is "b3lyp5" (PySCF's plain "b3lyp" is the Gaussian/VWN-RPA variant).
# The "B3LYPG" row covers the Gaussian flavor (vibe-qc's b3lypg /
# b3lyp/g <-> PySCF's bare b3lyp) so both flavors get gradient parity.
RKS_FUNCTIONALS = [
    ("LDA", "lda,vwn"),
    ("PBE", "pbe,pbe"),
    ("B3LYP", "b3lyp5"),
    ("B3LYPG", "b3lyp"),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name", RKS_CASES,
    ids=[c[0] for c in RKS_CASES],
)
@pytest.mark.parametrize(
    "vq_name,ps_name", RKS_FUNCTIONALS,
    ids=[c[0] for c in RKS_FUNCTIONALS],
)
def test_rks_gradient_matches_pyscf(label, atoms, basis_name,
                                    vq_name, ps_name):
    grad_vq = _vibeqc_rks_gradient(atoms, basis_name, vq_name)
    grad_ps = _pyscf_rks_gradient(atoms, basis_name, ps_name)
    # On the default 75×17×36 grid (no Becke-weight derivatives yet) we
    # expect grid-accuracy agreement of ~1e-5 Ha/bohr for every family.
    np.testing.assert_allclose(
        grad_vq, grad_ps, atol=5e-5, rtol=0,
        err_msg=f"RKS/{vq_name} gradient disagrees with PySCF for {label}",
    )


# ---------------------------------------------------------------------------
# UKS gradients (open-shell DFT)
# ---------------------------------------------------------------------------

def _pyscf_uks_gradient(atoms_bohr, basis_name, xc, *, charge, spin):
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0
    mol.build()
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = 5
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf.nuc_grad_method().kernel()


def _vibeqc_uks_gradient(atoms_bohr, basis_name, functional, *, mult,
                         grid_opts=None):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
                   multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    opts = UKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-5
    opts.max_iter = 500
    if grid_opts is not None:
        opts.grid = grid_opts
    result = run_uks(mol, basis, opts)
    return np.array(compute_gradient_uks(mol, basis, result,
                                          opts.grid if grid_opts is None else grid_opts))


UKS_CASES = [
    ("H-doublet", [(1, [0, 0, 0])], "sto-3g", 2),
    ("OH-doublet",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 / 0.529177210903, 0.0, 0.0])], "sto-3g", 2),
    ("O2-triplet", [(8, [0, 0, -0.6]), (8, [0, 0, 0.6])], "sto-3g", 3),
]


@pytest.mark.parametrize("vq_name,ps_name", RKS_FUNCTIONALS,
                         ids=[c[0] for c in RKS_FUNCTIONALS])
@pytest.mark.parametrize("label,atoms,basis_name,mult", UKS_CASES,
                         ids=[c[0] for c in UKS_CASES])
def test_uks_gradient_matches_pyscf(label, atoms, basis_name, mult,
                                    vq_name, ps_name):
    grad_vq = _vibeqc_uks_gradient(atoms, basis_name, vq_name, mult=mult)
    grad_ps = _pyscf_uks_gradient(atoms, basis_name, ps_name,
                                   charge=0, spin=mult - 1)
    # Slightly looser tolerance than RKS — UKS on compressed O2 pushes the
    # grid-accuracy limit.
    np.testing.assert_allclose(
        grad_vq, grad_ps, atol=1e-4, rtol=0,
        err_msg=f"UKS/{vq_name} gradient disagrees with PySCF for {label}",
    )


def test_uks_gradient_matches_rks_on_closed_shell():
    """Closed-shell UKS gradient collapses to RKS (D_α = D_β = D/2)."""
    atoms = [(8, [0, 0, 0]),
             (1, [0.0, 1.5, -1.16]),
             (1, [0.0, -1.5, -1.16])]
    g_uks = _vibeqc_uks_gradient(atoms, "sto-3g", "LDA", mult=1)
    g_rks = _vibeqc_rks_gradient(atoms, "sto-3g", "LDA")
    np.testing.assert_allclose(g_uks, g_rks, atol=1e-8)


@pytest.mark.parametrize("functional", ["LDA", "PBE", "B3LYP"])
def test_rks_gradient_matches_finite_differences(functional):
    """Analytic vs central FD on a fine grid: the gradient formula itself
    is correct (matches to ~1e-8 Ha/bohr once grid noise is suppressed)."""
    atoms = [(1, [0, 0, 0]), (1, [0, 0, 1.4])]
    grid = GridOptions()
    grid.n_radial = 150
    grid.n_theta = 25
    grid.n_phi = 48

    g_ana = _vibeqc_rks_gradient(atoms, "sto-3g", functional, grid_opts=grid)

    h = 1e-4

    def energy_at(geom):
        mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in geom])
        opts = RKSOptions()
        opts.functional = functional
        opts.grid = grid
        opts.conv_tol_energy = 1e-12
        opts.conv_tol_grad = 1e-9
        return run_rks(mol, BasisSet(mol, "sto-3g"), opts).energy

    g_fd = np.zeros_like(g_ana)
    for A in range(len(atoms)):
        for c in range(3):
            ap = [(Z, list(xyz)) for Z, xyz in atoms]
            am = [(Z, list(xyz)) for Z, xyz in atoms]
            ap[A][1][c] += h
            am[A][1][c] -= h
            g_fd[A, c] = (energy_at(ap) - energy_at(am)) / (2 * h)

    np.testing.assert_allclose(g_ana, g_fd, atol=1e-7)


# ---------------------------------------------------------------------------
# Meta-GGA (τ-dependent) analytic gradients — TPSS, TPSSh, M06-L, M06-2X.
#
# The XC Pulay force gains a fourth term for meta-GGAs:
#   T4_{μν}^c = −Σ_g w_g v_τ(g) Σ_d ∂_c∂_d χ_μ(g) ∂_d χ_ν(g)
# built from the AO Hessian (∂²χ) that the GGA path already evaluates.
# ---------------------------------------------------------------------------

# TPSS / TPSSh have smooth enhancement factors and validate as tightly as
# LDA/PBE/B3LYP. M06-L / M06-2X are Minnesota functionals — their
# oscillatory enhancement factors leave a larger (but still small)
# grid-discretization residual in the FD comparison, hence the looser
# per-functional tolerance.
_MGGA_FD_TOL = {
    "tpss":    1e-7,
    "tpssh":   1e-7,
    "m06-l":   2e-6,
    "m06-2x":  2e-6,
    "scan":    1e-7,
    "r2scan":  1e-7,
    "r2scan0": 1e-7,
    "r2scanh": 1e-7,
}


@pytest.mark.parametrize("functional", list(_MGGA_FD_TOL))
def test_mgga_rks_gradient_matches_finite_differences(functional):
    """Analytic meta-GGA RKS gradient vs central finite difference on a
    fine grid. Confirms the τ-Pulay term (term 4) — and the σ terms an
    MGGA also carries — are correct. H2/sto-3g keeps the FD energies
    cheap and noise-free; the τ kernel is basis-angular-momentum
    agnostic (it contracts ∂²χ · ∂χ, exercised for any shell)."""
    atoms = [(1, [0, 0, 0]), (1, [0, 0, 1.4])]
    grid = GridOptions()
    grid.n_radial = 150
    grid.n_theta = 25
    grid.n_phi = 48

    g_ana = _vibeqc_rks_gradient(atoms, "sto-3g", functional, grid_opts=grid)

    h = 1e-4

    def energy_at(geom):
        mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in geom])
        opts = RKSOptions()
        opts.functional = functional
        opts.grid = grid
        opts.conv_tol_energy = 1e-12
        opts.conv_tol_grad = 1e-9
        return run_rks(mol, BasisSet(mol, "sto-3g"), opts).energy

    g_fd = np.zeros_like(g_ana)
    for A in range(len(atoms)):
        for c in range(3):
            ap = [(Z, list(xyz)) for Z, xyz in atoms]
            am = [(Z, list(xyz)) for Z, xyz in atoms]
            ap[A][1][c] += h
            am[A][1][c] -= h
            g_fd[A, c] = (energy_at(ap) - energy_at(am)) / (2 * h)

    np.testing.assert_allclose(g_ana, g_fd, atol=_MGGA_FD_TOL[functional])


@pytest.mark.parametrize("functional",
                         ["tpss", "tpssh", "m06-l", "m06-2x",
                          "scan", "r2scan", "r2scan0", "r2scanh"])
def test_mgga_uks_gradient_matches_rks_on_closed_shell(functional):
    """Closed-shell UKS meta-GGA gradient must collapse to the RKS
    gradient exactly (D_α = D_β = D/2, τ_α = τ_β = τ/2). This is the
    rigorous regression guard for the per-spin τ-Pulay term — it pins
    the UKS path to the independently FD-validated RKS path with no
    grid-noise floor (both runs share the same grid). def2-SVP
    exercises s/p/d basis functions through the τ-Hessian contraction."""
    atoms = [(8, [0.0, 0.0, 0.221]),
             (1, [0.0, 1.43, -0.88]),
             (1, [0.0, -1.43, -0.88])]
    g_rks = _vibeqc_rks_gradient(atoms, "def2-svp", functional)
    g_uks = _vibeqc_uks_gradient(atoms, "def2-svp", functional, mult=1)
    np.testing.assert_allclose(g_uks, g_rks, atol=1e-7)


@pytest.mark.parametrize("vq_name,ps_name", RKS_FUNCTIONALS,
                         ids=[c[0] for c in RKS_FUNCTIONALS])
def test_rks_gradient_pulay_matches_pyscf_fine_grid(vq_name, ps_name):
    """Tight regression guard for the XC Pulay gradient formula.

    vibe-qc's RKS gradient omits the Becke-weight (grid-response)
    derivative, and so does PySCF's ``rks.Gradients`` by default
    (``grid_response = False``) — so both are *Pulay-term only*. On a
    fine grid the grid-discretization mismatch between vibe-qc's
    product grid and PySCF's Lebedev grid drops below ~5e-6 Ha/bohr,
    leaving the XC Pulay formula itself as the thing under test.

    H2O/def2-svp exercises s/p/d basis functions through the GGA
    Hessian (M2/M3) path. The looser default-grid test
    (``test_rks_gradient_matches_pyscf``, 5e-5) cannot distinguish a
    correct Pulay term (sits at ~1.3e-5 on the coarse grid) from a
    moderate formula bug; this fine-grid variant can.

    Audit basis: ``scripts/xc_pulay_gga_audit.py`` measured the
    vibe-qc-vs-PySCF(grid_response=False) gap at 9.9e-7 (H2O/def2-svp)
    and 2.1e-6 (H2CO/def2-tzvp) on the 150x25x48 grid — both shrinking
    13-17x from the coarse grid, the signature of a correct formula.
    """
    atoms = [(8, [0.0, 0.0, 0.0]),
             (1, [0.0, 1.5, -1.16]),
             (1, [0.0, -1.5, -1.16])]
    grid = GridOptions()
    grid.n_radial = 150
    grid.n_theta = 25
    grid.n_phi = 48

    grad_vq = _vibeqc_rks_gradient(atoms, "def2-svp", vq_name, grid_opts=grid)

    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft
    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    mol.basis = "def2-svp"
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol)
    mf.xc = ps_name
    mf.grids.level = 6
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    g = mf.nuc_grad_method()
    g.grid_response = False  # Pulay-only, matching vibe-qc
    grad_ps = g.kernel()

    np.testing.assert_allclose(
        grad_vq, grad_ps, atol=5e-6, rtol=0,
        err_msg=(
            f"RKS/{vq_name} XC Pulay gradient disagrees with PySCF "
            f"(grid_response=False) on a fine grid — the XC Pulay "
            f"formula has likely regressed. See "
            f"scripts/xc_pulay_gga_audit.py."
        ),
    )
