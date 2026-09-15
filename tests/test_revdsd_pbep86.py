"""revDSD-PBEP86-D4 (Santra, Sylvetsky & Martin, *J. Phys. Chem. A*
**123**, 5129 (2019)) — the GMTKN55-retrained revision of DSD-PBEP86,
the fourth double hybrid in vibe-qc.

Pins:

  1. ``Functional("revdsd-pbep86")`` resolves to the published **D4**
     parametrisation: 0.69·HF + 0.31·PBE-X + 0.4210·P86-C for the SCF
     piece, c_os = 0.5922, c_ss = 0.0636 for the MP2 correction (Santra-
     Sylvetsky-Martin 2019, Table 4). NB the -D3BJ member uses a
     *different* XC/MP2 fit and is deliberately not reachable here.
  2. ``run_revdsd_pbep86`` orchestrates the SCF + RI-MP2 dispatch and
     returns a :class:`DoubleHybridResult`. Same shape as
     :func:`run_dsd_pbep86`; both delegate to :func:`run_double_hybrid`.
  3. Combined total agrees with PySCF (hand-rolled hybrid SCF + scaled
     DFMP2 correction, built from the identical primitive XC pieces) to
     grid accuracy on H2O / cc-pVDZ.
  4. ``e_total`` is exactly ``rks.energy + mp2.e_correlation`` with the
     asymmetric Santra-Martin scaling.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    DoubleHybridResult,
    Functional,
    Molecule,
    RKSOptions,
    define_functional,
    run_double_hybrid,
    run_revdsd_pbep86,
    run_rks,
)

from .conftest import GEOMETRIES


# revDSD-PBEP86-D4 linear coefficients (Santra-Sylvetsky-Martin 2019,
# Table 4; the matching D4 damping is s6 = 0.5132, s8 = 0, a1 = 0.44,
# a2 = 3.60 in vibeqc.dispersion_d4_parameters).
REVDSD_HF = 0.69
REVDSD_PBE = 0.31
REVDSD_P86 = 0.4210
REVDSD_C_OS = 0.5922
REVDSD_C_SS = 0.0636

# --- Witness M19: H2O / cc-pVTZ, the revDSD-PBEP86 SCF half ------------
#
# ORCA 6.1.1 reference run rp163-orca-m19-revdsd (deck: "! REVDSD-PBEP86-D4/
# 2021 cc-pVTZ cc-pVTZ/C TightSCF NoFrozenCore"), SCF SETTINGS block:
#
#   Exchange Functional  Exchange  .... PBE
#   Correlation Functional         .... P86
#   LDA part of GGA corr.  LDAOpt  .... VWN-5      <-- the flavor verdict
#   ScalHFX 0.690000 / ScalDFX 0.310000
#   ScalDFC = ScalLDAC = 0.422400                  <-- 2021 revision, not 2019
#
# "Total Energy : -76.20053254096288 Eh".
ORCA_M19_SCF_2021 = -76.20053254096288

# vibe-qc ships the *2019* Table 4 D4 set (c_c = 0.4210), so the pinned
# residual below is not zero: the ORCA reference is the 2021 keyword whose
# c_c = 0.4224. Measured contributions to the SCF half on this witness
# (GitLab issue 32, decomposition against the run above):
#
#   shipped pre-fix   PZ81 c_c=0.4210   +2.4582 mHa vs ORCA
#   flavor fix        VWN5 c_c=0.4210   +0.6613 mHa   (P86 flavor: -1.7969)
#   ORCA-equivalent   VWN5 c_c=0.4224   +0.1518 mHa   (revision:    -0.5095)
#
# The +0.1518 mHa floor is ORCA's own RIJCOSX/RI-J approximation — the same
# run reports "Exchange energy change after final integration: -0.000159597
# Eh" — so it is not a vibe-qc defect and this test does not chase it.
M19_SCF_DELTA_AFTER_FLAVOR_FIX_MHA = 0.6613
# Pre-fix the deviation was 2.4582 mHa; anything at or above ~1.5 mHa means
# the PZ81-local P86 has come back.
M19_SCF_DELTA_TOL_MHA = 0.15


def test_revdsd_pbep86_functional_resolves():
    """``Functional("revdsd-pbep86")`` carries the Santra-Martin D4
    parametrisation; the no-dash alias resolves identically."""
    f = Functional("revdsd-pbep86")
    assert f.hf_exchange_fraction == pytest.approx(REVDSD_HF, abs=1e-12)
    assert f.is_hybrid is True
    assert f.is_double_hybrid is True
    assert f.mp2_c_os == pytest.approx(REVDSD_C_OS, abs=1e-12)
    assert f.mp2_c_ss == pytest.approx(REVDSD_C_SS, abs=1e-12)

    f2 = Functional("revdsdpbep86")
    assert f2.hf_exchange_fraction == pytest.approx(REVDSD_HF, abs=1e-12)
    assert f2.mp2_c_os == pytest.approx(REVDSD_C_OS, abs=1e-12)
    assert f2.mp2_c_ss == pytest.approx(REVDSD_C_SS, abs=1e-12)

    # Distinct from the original DSD-PBEP86 fit — the coefficients moved.
    dsd = Functional("dsd-pbep86")
    assert f.mp2_c_os != pytest.approx(dsd.mp2_c_os, abs=1e-6)
    assert f.mp2_c_ss != pytest.approx(dsd.mp2_c_ss, abs=1e-6)


def _probe_scf(mol, basis, functional_name):
    """SCF energy of a plain (non-double-hybrid) probe functional."""
    opts = RKSOptions()
    opts.functional = functional_name
    return run_rks(mol, basis, opts).energy


def _shipped_scf(mol, basis):
    """SCF half of the shipped revdsd-pbep86 alias.

    ``run_rks`` deliberately refuses double hybrids, so the SCF piece is
    taken from the dispatcher's :attr:`rks` sub-result.
    """
    return run_revdsd_pbep86(mol, basis, density_fit_mp2=True,
                             aux_basis_mp2="cc-pvtz-ri").rks.energy


def test_revdsd_pbep86_uses_vwn5_local_p86():
    """The P86 correlation slot is the VWN5-local flavor.

    Regression pin for GitLab issue 32. The 2019 paper's Electronic
    Supporting Information (p. S22, ORCA sample input 3) specifies
    ``Exchange X_PBE`` / ``Correlation C_P86``, and ORCA composes that on
    VWN-5 (its output prints ``LDA part of GGA corr.  LDAOpt .... VWN-5``).
    Building the same recipe on PZ81-local ``GGA_C_P86`` instead leaves
    ~1.8 mHa in the SCF half on this witness.

    Pinned by energy equivalence rather than by libxc id, so the test
    stays valid however the alias is composed: the shipped functional must
    reproduce the explicitly VWN5-built recipe exactly, and must differ
    from the PZ81-built one by a margin far above grid noise.
    """
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvtz")

    define_functional("_pin_revdsd_vwn5",
                      [("GGA_X_PBE", REVDSD_PBE), ("GGA_C_P86VWN", REVDSD_P86)],
                      hf_exchange_fraction=REVDSD_HF)
    define_functional("_pin_revdsd_pz81",
                      [("GGA_X_PBE", REVDSD_PBE), ("GGA_C_P86", REVDSD_P86)],
                      hf_exchange_fraction=REVDSD_HF)

    e_shipped = _shipped_scf(mol, basis)
    e_vwn5 = _probe_scf(mol, basis, "_pin_revdsd_vwn5")
    e_pz81 = _probe_scf(mol, basis, "_pin_revdsd_pz81")

    # 1e-6 Ha: the dispatcher density-fits the Coulomb build while the probe
    # path does not, which costs ~3e-8 Ha here. That is still three orders of
    # magnitude below the ~1.8e-3 Ha flavor signal this test discriminates.
    assert e_shipped == pytest.approx(e_vwn5, abs=1e-6), (
        f"revdsd-pbep86 must be built on VWN5-local P86 (GGA_C_P86VWN); "
        f"shipped {e_shipped:.10f} vs VWN5 recipe {e_vwn5:.10f}"
    )
    # The two flavors are ~1.8 mHa apart here — assert the shipped alias is
    # not the PZ81 one by a wide margin.
    assert abs(e_shipped - e_pz81) > 1e-3, (
        f"revdsd-pbep86 appears to use PZ81-local GGA_C_P86 (delta to the "
        f"PZ81 recipe is only {(e_shipped - e_pz81)*1e3:+.4f} mHa) — see the "
        f"ESI ORCA deck cited in cpp/src/xc.cpp"
    )


def test_revdsd_pbep86_scf_half_matches_orca_m19():
    """SCF half on H2O/cc-pVTZ sits within 0.15 mHa of the residual left
    by the 2019-vs-2021 coefficient revision (GitLab issue 32).

    Pre-fix (PZ81-local P86) this deviation was 2.4582 mHa — ~70x the
    sibling grid-residual band (0.001-0.036 mHa).
    """
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvtz")
    energy = _shipped_scf(mol, basis)

    delta_mha = (energy - ORCA_M19_SCF_2021) * 1e3
    assert delta_mha == pytest.approx(
        M19_SCF_DELTA_AFTER_FLAVOR_FIX_MHA, abs=M19_SCF_DELTA_TOL_MHA), (
        f"revDSD-PBEP86 SCF half on M19 deviates {delta_mha:+.4f} mHa from the "
        f"ORCA reference; expected {M19_SCF_DELTA_AFTER_FLAVOR_FIX_MHA:+.4f} "
        f"+/- {M19_SCF_DELTA_TOL_MHA} mHa. A value near +2.46 mHa means the "
        f"PZ81-local GGA_C_P86 has come back."
    )


def _pyscf_revdsd_pbep86(atoms_bohr, basis_name, aux_basis_name):
    """revDSD-PBEP86 on PySCF: hand-rolled hybrid SCF + DFMP2 correction
    with c_os = 0.5922, c_ss = 0.0636. Same composition path vibe-qc uses
    internally — libxc ships no built-in revDSD-PBEP86, so both codes
    build it from the same primitive XC pieces (PBE exchange + P86
    correlation + HF exchange)."""
    pytest.importorskip("pyscf")
    from pyscf import gto, dft, mp

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    # GGA_C_P86VWN (VWN5-local), not GGA_C_P86 (PZ81-local) — the flavor
    # the 2019 paper's ESI prescribes via ORCA "Correlation C_P86".
    mf = dft.RKS(mol, xc=f"{REVDSD_HF}*HF + {REVDSD_PBE}*GGA_X_PBE, "
                         f"{REVDSD_P86}*GGA_C_P86VWN")
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    m = mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name)
    m.kernel()
    e_corr_dh = REVDSD_C_OS * m.e_corr_os + REVDSD_C_SS * m.e_corr_ss
    return mf.e_tot, e_corr_dh, mf.e_tot + e_corr_dh


def atoms_to_mol_basis(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    return mol, basis


def test_run_revdsd_pbep86_accepts_basis_name_and_citations_kw(tmp_path):
    mol, _ = atoms_to_mol_basis(GEOMETRIES["H2O"], "sto-3g")
    result = run_revdsd_pbep86(
        mol,
        "sto-3g",
        density_fit=False,
        density_fit_mp2=False,
        output=tmp_path / "revdsd",
        citations=True,
    )

    assert isinstance(result, DoubleHybridResult)
    assert result.functional == "revdsd-pbep86"
    assert np.isfinite(result.e_total)


def test_run_revdsd_pbep86_matches_pyscf_h2o_ccpvdz():
    """End-to-end parity on H2O / cc-pVDZ. As for DSD-PBEP86 the
    cross-code agreement is set by the default DFT-grid difference
    between vibe-qc and PySCF (P86 correlation is grid-sensitive; the
    SCF step alone differs at the ~1e-5 Ha level on this case), and the
    MP2 correction inherits that gap via the KS orbitals."""
    atoms = GEOMETRIES["H2O"]
    mol, basis = atoms_to_mol_basis(atoms, "cc-pvdz")
    result = run_revdsd_pbep86(mol, basis, density_fit_mp2=True,
                               aux_basis_mp2="cc-pvdz-ri")

    e_rks_ps, e_corr_ps, e_total_ps = _pyscf_revdsd_pbep86(
        atoms, "cc-pvdz", "cc-pvdz-ri")

    delta_total = result.e_total - e_total_ps
    # 5e-5 Ha = ~0.03 kcal/mol — well below any chemically meaningful
    # threshold and matched to the grid-coupling spread observed for the
    # sibling DSD-PBEP86 parity test on the same geometry. Tighten once
    # vibe-qc and PySCF share an identical XC integration grid.
    assert abs(delta_total) < 5e-5, (
        f"vibeqc revDSD-PBEP86 vs PySCF total gap = {delta_total:+.3e} Ha "
        f"(vibeqc = {result.e_total:.10f}, PySCF = {e_total_ps:.10f})"
    )


def test_run_revdsd_pbep86_total_equals_sum_of_parts():
    """e_total = rks.energy + mp2.e_correlation, with MP2 scaled by the
    Santra-Martin (asymmetric) coefficients."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    result = run_revdsd_pbep86(mol, basis, density_fit_mp2=True,
                               aux_basis_mp2="cc-pvdz-ri")
    assert isinstance(result, DoubleHybridResult)
    assert result.functional == "revdsd-pbep86"
    assert result.e_total == pytest.approx(
        result.rks.energy + result.mp2.e_correlation, abs=1e-14)
    # Asymmetric scaling: e_correlation = 0.5922·e_os + 0.0636·e_ss
    assert result.mp2.e_correlation == pytest.approx(
        REVDSD_C_OS * result.mp2.e_os + REVDSD_C_SS * result.mp2.e_ss,
        rel=1e-14)


def test_run_revdsd_pbep86_via_run_double_hybrid():
    """run_revdsd_pbep86(...) == run_double_hybrid(..., "revdsd-pbep86")."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    direct = run_revdsd_pbep86(mol, basis, density_fit_mp2=True,
                               aux_basis_mp2="cc-pvdz-ri")
    via_generic = run_double_hybrid(
        mol, basis, "revdsd-pbep86",
        density_fit_mp2=True, aux_basis_mp2="cc-pvdz-ri")
    assert direct.e_total == pytest.approx(via_generic.e_total, abs=1e-14)
