"""DF-UKS: density-fitted unrestricted Kohn-Sham DFT.

Mirrors tests/test_df_rks.py for the open-shell path. Pins:

  1. Internal — DF-UKS agrees with direct UKS up to JKfit fit error,
     across LDA / PBE / B3LYP and across pure / hybrid functionals.
  2. PySCF parity on shared aux.
  3. Pure DFT path doesn't touch K (e_hf_exchange = 0).
  4. Closed-shell consistency — DF-UKS on H2O singlet (n_α = n_β)
     reproduces DF-RKS energy (analogous to UHF↔RHF check in
     test_df_uhf.py).
  5. Pure-spin H atom doublet: <S^2> = 0.75 exactly under DF.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    InitialGuess,
    Molecule,
    RKSOptions,
    UKSOptions,
    run_rks,
    run_uks,
)

from .conftest import GEOMETRIES


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _vibeqc_uks(atoms_bohr, basis_name, functional, charge, mult, *,
                density_fit, aux_basis_name=""):
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
        charge=charge, multiplicity=mult,
    )
    basis = BasisSet(mol, basis_name)
    opts = UKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 500
    # HCORE guess for open-shell DFT testing. The default SAD guess is
    # tuned for periodic ionic insulators (NaCl, MgO) where Hcore lands
    # in the wrong basin; for molecular open-shell radicals SAD's
    # closed-shell-symmetric atomic densities can leave the SCF trapped
    # near a too-symmetric local minimum, especially when DF perturbs
    # the Fock build slightly. Hcore breaks symmetry through the orbital
    # eigenvectors of (T + V_ne) and converges cleanly here.
    opts.initial_guess = InitialGuess.HCORE
    opts.density_fit = density_fit
    opts.aux_basis = aux_basis_name
    return run_uks(mol, basis, opts)


def _pyscf_uks_df(atoms_bohr, basis_name, xc, aux_basis_name, *,
                  charge, spin):
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
    mf = dft.UKS(mol).density_fit(auxbasis=aux_basis_name)
    mf.xc = xc
    mf.grids.level = 5
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.max_cycle = 500
    mf.kernel()
    assert mf.converged, f"PySCF DF-UKS did not converge {basis_name}/{xc}"
    return mf.e_tot


# Internal DF↔direct case list. Both H-doublet (one electron, trivial)
# and OH-doublet (small-gap radical, harder) are covered to validate
# the wiring across functional families.
INTERNAL_OPEN_SHELL_CASES = [
    ("OH-doublet",
     [(8, [0.0, 0.0, 0.0]),
      (1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
     "def2-svp", 0, 2, "def2-universal-jkfit"),
    ("H-doublet",
     [(1, [0.0, 0.0, 0.0])],
     "def2-svp", 0, 2, "def2-universal-jkfit"),
]

# PySCF cross-validation case list. OH-doublet is excluded: PySCF
# DF-UKS on OH/def2-svp is unreliable across functionals — LDA and
# B3LYP fail to converge with PySCF's default settings, even with
# bumped max_cycle. Both libraries' SCFs are oscillating in the
# convergence tail on this small-gap radical, so a tight cross-check
# would be measuring SCF-tail noise rather than DF correctness. The
# internal direct↔DF check above already validates DF on OH; PySCF
# parity for the wiring pattern is pinned via H-doublet (where both
# libraries converge cleanly with default settings).
PYSCF_PARITY_OPEN_SHELL_CASES = [
    ("H-doublet",
     [(1, [0.0, 0.0, 0.0])],
     "def2-svp", 0, 2, "def2-universal-jkfit"),
]

# Functionals to sweep.
FUNCTIONAL_PAIRS = [
    ("LDA",   "LDA",   "lda,vwn"),
    ("PBE",   "PBE",   "pbe,pbe"),
    # vibe-qc's "B3LYP" = VWN5 variant (ORCA convention); PySCF's
    # matching spelling is "b3lyp5".
    ("B3LYP", "B3LYP", "b3lyp5"),
]


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux",
    INTERNAL_OPEN_SHELL_CASES,
    ids=[c[0] for c in INTERNAL_OPEN_SHELL_CASES],
)
@pytest.mark.parametrize(
    "func_label,vq_func,_ps_xc",
    FUNCTIONAL_PAIRS,
    ids=[c[0] for c in FUNCTIONAL_PAIRS],
)
def test_df_uks_close_to_direct(
    func_label, vq_func, _ps_xc, label, atoms, basis_name, charge, mult, aux,
):
    """DF-UKS total energy stays within JKfit accuracy of direct UKS."""
    direct = _vibeqc_uks(atoms, basis_name, vq_func, charge, mult,
                         density_fit=False)
    df = _vibeqc_uks(atoms, basis_name, vq_func, charge, mult,
                     density_fit=True, aux_basis_name=aux)

    assert direct.converged, f"{func_label}/{label}: direct UKS no-converge"
    assert df.converged,     f"{func_label}/{label}: DF-UKS no-converge"
    delta = df.energy - direct.energy
    assert abs(delta) < 5e-4, (
        f"{func_label}/{label}/{basis_name}/{aux}: "
        f"DF-direct gap = {delta:+.3e} Ha "
        f"(direct = {direct.energy:.10f}, DF = {df.energy:.10f})"
    )


@pytest.mark.parametrize(
    "label,atoms,basis_name,charge,mult,aux",
    PYSCF_PARITY_OPEN_SHELL_CASES,
    ids=[c[0] for c in PYSCF_PARITY_OPEN_SHELL_CASES],
)
@pytest.mark.parametrize(
    "func_label,vq_func,ps_xc",
    FUNCTIONAL_PAIRS,
    ids=[c[0] for c in FUNCTIONAL_PAIRS],
)
def test_df_uks_matches_pyscf_df(
    func_label, vq_func, ps_xc, label, atoms, basis_name, charge, mult, aux,
):
    """vibeqc DF-UKS matches PySCF DF-UKS on shared aux (XC-grid limited
    to ~3e-6 Ha — different grid implementations between vibeqc and
    PySCF, same as direct-UKS/RKS PySCF parity)."""
    df = _vibeqc_uks(atoms, basis_name, vq_func, charge, mult,
                     density_fit=True, aux_basis_name=aux)
    spin = mult - 1
    e_pyscf = _pyscf_uks_df(atoms, basis_name, ps_xc, aux,
                            charge=charge, spin=spin)
    delta = df.energy - e_pyscf
    assert abs(delta) < 3e-6, (
        f"{func_label}/{label}/{basis_name}/{aux}: vibeqc-PySCF DF-UKS "
        f"gap = {delta:+.3e} Ha (vibeqc = {df.energy:.10f}, "
        f"PySCF = {e_pyscf:.10f})"
    )


def test_df_uks_pure_dft_no_hf_exchange():
    """Pure GGA path under DF reports e_hf_exchange = 0."""
    df = _vibeqc_uks(
        [(1, [0.0, 0.0, 0.0])], "def2-svp", "PBE", 0, 2,
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    assert df.e_hf_exchange == 0.0


def test_df_uks_h_doublet_pure_spin():
    """Pure-spin one-electron system: <S^2> = 0.75 exactly under DF."""
    df = _vibeqc_uks(
        [(1, [0.0, 0.0, 0.0])], "def2-svp", "PBE", 0, 2,
        density_fit=True, aux_basis_name="def2-universal-jkfit",
    )
    assert df.s_squared == pytest.approx(0.75, abs=1e-10)
    assert df.s_squared_ideal == pytest.approx(0.75, abs=1e-12)


def test_df_uks_closed_shell_recovers_df_rks():
    """DF-UKS on H2O singlet (n_α = n_β) reproduces DF-RKS energy."""
    atoms = GEOMETRIES["H2O"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "def2-svp")

    rks_opts = RKSOptions()
    rks_opts.functional = "PBE"
    rks_opts.conv_tol_energy = 1e-9
    rks_opts.conv_tol_grad = 1e-7
    rks_opts.density_fit = True
    rks_opts.aux_basis = "def2-universal-jkfit"
    r_rks = run_rks(mol, basis, rks_opts)

    uks_opts = UKSOptions()
    uks_opts.functional = "PBE"
    uks_opts.conv_tol_energy = 1e-9
    uks_opts.conv_tol_grad = 1e-7
    uks_opts.density_fit = True
    uks_opts.aux_basis = "def2-universal-jkfit"
    r_uks = run_uks(mol, basis, uks_opts)

    assert r_uks.converged and r_rks.converged
    assert abs(r_uks.energy - r_rks.energy) < 1e-7, (
        f"DF-UKS on H2O singlet: E_uks = {r_uks.energy:.10f}, "
        f"E_rks = {r_rks.energy:.10f}, "
        f"gap = {r_uks.energy - r_rks.energy:+.3e}"
    )
    assert abs(r_uks.s_squared) < 1e-7


def test_df_uks_requires_aux_basis():
    """density_fit=True with empty aux_basis raises — same contract
    as run_rhf / run_uhf / run_rks."""
    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.97 * ANGSTROM_TO_BOHR, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = BasisSet(mol, "def2-svp")
    opts = UKSOptions()
    opts.functional = "PBE"
    opts.density_fit = True
    opts.aux_basis = ""
    with pytest.raises(ValueError, match="aux_basis"):
        run_uks(mol, basis, opts)
