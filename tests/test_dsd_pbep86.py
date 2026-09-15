"""DSD-PBEP86 (Kozuch & Martin, *Phys. Chem. Chem. Phys.* **13**,
20104 (2011)) — second double-hybrid in vibe-qc, riding the SCS-MP2
surface with asymmetric c_os ≠ c_ss.

Pins:

  1. ``Functional("dsd-pbep86")`` resolves to the published final
     recommended D3(BJ)-fit parametrisation (Kozuch-Martin 2011,
     p. 20106): 0.70·HF + 0.30·PBE-X + 0.43·P86-C for the SCF piece,
     c_os = 0.53, c_ss = 0.25 for the MP2 correction. NOT the 2013
     JCC revision (0.69/0.31/0.44, 0.52/0.22) — vibe-qc's bare name
     means the 2011 set, matching ORCA and the 2011 D3(BJ) damping
     fit (see BUG DSDPBEP86-DFT-PART-1P6MHA in cpp/src/xc.cpp).
  2. The DFT (SCF) part reproduces the conventional gate-clean ORCA
     6.1.1 reference on the mb014 case (H2O / def2-SVP) inside the
     sibling grid-residual band.
  3. ``run_dsd_pbep86`` orchestrates the SCF + RI-MP2 dispatch and
     returns a :class:`DoubleHybridResult`. Same shape as
     :func:`run_b2plyp`; both delegate to :func:`run_double_hybrid`.
  4. Combined total agrees with PySCF (hand-rolled hybrid SCF +
     scaled DFMP2 correction) to grid accuracy on H2O / cc-pVDZ.
  5. ``run_double_hybrid`` rejects non-double-hybrid functional names
     with a clear ValueError.
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
    run_dsd_pbep86,
    run_double_hybrid,
)

from .conftest import GEOMETRIES


# Final recommended D3(BJ)-fit parameters of Kozuch & Martin,
# Phys. Chem. Chem. Phys. 13, 20104 (2011), doi:10.1039/c1cp22592h,
# p. 20106: c_S = 0.25, c_O = 0.53, c_X = 0.30, c_C = 0.43 — eqn (1)
# there defines c_X as the DFT-exchange fraction, so HF = 0.70. The
# same set ORCA's DSD-PBEP86 keyword prints (ScalHFX 0.700, ScalDFX
# 0.300, ScalDFC = ScalLDAC 0.430, MP2 aa/bb 0.25, ab 0.53).
DSD_HF = 0.70
DSD_PBE = 0.30
DSD_P86 = 0.43
DSD_C_OS = 0.53
DSD_C_SS = 0.25


def test_dsd_pbep86_functional_resolves():
    """``Functional("dsd-pbep86")`` carries the Kozuch-Martin D3(BJ)
    parametrisation; the no-dash alias resolves identically."""
    f = Functional("dsd-pbep86")
    assert f.hf_exchange_fraction == pytest.approx(DSD_HF, abs=1e-12)
    assert f.is_hybrid is True
    assert f.is_double_hybrid is True
    assert f.mp2_c_os == pytest.approx(DSD_C_OS, abs=1e-12)
    assert f.mp2_c_ss == pytest.approx(DSD_C_SS, abs=1e-12)

    f2 = Functional("dsdpbep86")
    assert f2.hf_exchange_fraction == pytest.approx(DSD_HF, abs=1e-12)
    assert f2.mp2_c_os == pytest.approx(DSD_C_OS, abs=1e-12)
    assert f2.mp2_c_ss == pytest.approx(DSD_C_SS, abs=1e-12)


def _pyscf_dsd_pbep86(atoms_bohr, basis_name, aux_basis_name):
    """DSD-PBEP86 on PySCF: hand-rolled hybrid SCF + DFMP2 correction
    with c_os = 0.53, c_ss = 0.25 (2011 D3BJ fit). Same composition
    path vibe-qc uses internally — libxc doesn't ship a built-in
    DSD-PBEP86, so both codes build it from the same primitive XC
    pieces."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft, mp

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    # GGA_C_P86VWN: the VWN5-local P86 flavor (ORCA lineage) —
    # matches vibe-qc's alias; libxc GGA_C_P86 is the PZ81-local
    # flavor and differs by ~1.9 mHa on H2O (see cpp/src/xc.cpp).
    mf = dft.RKS(mol, xc=f"{DSD_HF}*HF + {DSD_PBE}*GGA_X_PBE, "
                         f"{DSD_P86}*GGA_C_P86VWN")
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    m = mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name)
    m.kernel()
    e_corr_dh = DSD_C_OS * m.e_corr_os + DSD_C_SS * m.e_corr_ss
    return mf.e_tot, e_corr_dh, mf.e_tot + e_corr_dh


def atoms_to_mol_basis(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    return mol, basis


def test_run_dsd_pbep86_matches_pyscf_h2o_ccpvdz():
    """End-to-end parity on H2O / cc-pVDZ. The cross-code agreement is
    set by the default DFT-grid difference between vibe-qc and PySCF
    (the SCF step alone differs at the ~1e-5 Ha level on this case
    because P86 correlation is more grid-sensitive than LYP — the
    B2PLYP test lands at ~2e-7 Ha on the same geometry). The MP2
    correction inherits that gap via the KS orbitals."""
    atoms = GEOMETRIES["H2O"]
    mol, basis = atoms_to_mol_basis(atoms, "cc-pvdz")
    result = run_dsd_pbep86(mol, basis, density_fit_mp2=True,
                            aux_basis_mp2="cc-pvdz-ri")

    e_rks_ps, e_corr_ps, e_total_ps = _pyscf_dsd_pbep86(
        atoms, "cc-pvdz", "cc-pvdz-ri")

    delta_total = result.e_total - e_total_ps
    # 5e-5 Ha = ~0.03 kcal/mol — well below any chemically meaningful
    # threshold and matched to the observed grid-coupling spread on
    # H2O/cc-pVDZ. Tighten the tolerance once vibe-qc and PySCF share
    # an identical XC integration grid.
    assert abs(delta_total) < 5e-5, (
        f"vibeqc DSD-PBEP86 vs PySCF total gap = {delta_total:+.3e} Ha "
        f"(vibeqc = {result.e_total:.10f}, PySCF = {e_total_ps:.10f})"
    )


def test_dsd_pbep86_dft_part_matches_conventional_orca_mb014():
    """Regression pin for BUG DSDPBEP86-DFT-PART-1P6MHA (frozen
    registry, judgements-2026-08-06-twinfill case mb014).

    The DFT (hybrid-SCF) part of DSD-PBEP86 on H2O/def2-SVP must
    reproduce the gate-clean conventional ORCA 6.1.1 reference
    (deck ``! DSD-PBEP86 DEFGRID3 DEF2-SVP TightSCF D3BJ NoAutoStart
    NORI NORIJCOSX``, run-rgfix-pin1-mb014-orca611) inside the
    0.05 mHa sibling grid-residual band of that wave. Pre-fix the
    SCF piece carried the 2013-revision coefficients (0.69 HF /
    0.31 PBE / 0.44 P86) against ORCA's 2011 set and sat 1.58 mHa
    over-bound. Only the SCF part is compared: the MP2 part differs
    by frozen-core convention (ORCA freezes core; vibe-qc
    correlates all electrons), which is a documented protocol
    difference, not a defect.
    """
    ang2bohr = 1.8897259886
    atoms = [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.793353 * ang2bohr, -0.61351 * ang2bohr]),
        (1, [0.0, -0.793353 * ang2bohr, -0.61351 * ang2bohr]),
    ]
    mol, basis = atoms_to_mol_basis(atoms, "def2-svp")
    result = run_dsd_pbep86(mol, basis, density_fit_mp2=True)
    # ORCA 6.1.1 "Total Energy" (SCF) for the deck above, compute-cluster
    # 2026-08-06 refgate-fix wave (article-repo archive
    # validation-job).
    e_orca_scf = -76.10713294610909
    delta_mha = (result.rks.energy - e_orca_scf) * 1e3
    assert abs(delta_mha) < 0.05, (
        f"DSD-PBEP86 DFT part deviates {delta_mha:+.4f} mHa from the "
        f"conventional ORCA reference (band: 0.05 mHa; pre-fix bug: "
        f"-1.58 mHa)"
    )


def test_run_dsd_pbep86_total_equals_sum_of_parts():
    """e_total = rks.energy + mp2.e_correlation, with MP2 scaled by
    the Kozuch-Martin (asymmetric) coefficients."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    result = run_dsd_pbep86(mol, basis, density_fit_mp2=True,
                            aux_basis_mp2="cc-pvdz-ri")
    assert isinstance(result, DoubleHybridResult)
    assert result.functional == "dsd-pbep86"
    assert result.e_total == pytest.approx(
        result.rks.energy + result.mp2.e_correlation, abs=1e-14)
    # Asymmetric scaling: e_correlation = 0.53·e_os + 0.25·e_ss
    assert result.mp2.e_correlation == pytest.approx(
        DSD_C_OS * result.mp2.e_os + DSD_C_SS * result.mp2.e_ss,
        rel=1e-14)


def test_run_double_hybrid_rejects_non_dh_functional():
    """The generic dispatcher refuses LDA / GGA / hybrid-GGA — those
    aren't double hybrids and should be routed through run_rks instead.
    """
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "sto-3g")
    with pytest.raises(ValueError, match="non-double-hybrid"):
        run_double_hybrid(mol, basis, "pbe0")
    with pytest.raises(ValueError, match="non-double-hybrid"):
        run_double_hybrid(mol, basis, "b3lyp")
    with pytest.raises(ValueError, match="non-double-hybrid"):
        run_double_hybrid(mol, basis, "pw1pw")


def test_run_b2plyp_via_run_double_hybrid():
    """run_b2plyp(mol, basis) and run_double_hybrid(mol, basis, "b2plyp")
    return identical results — the named wrapper is a thin shim."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    from vibeqc import run_b2plyp
    direct = run_b2plyp(mol, basis, density_fit_mp2=True,
                        aux_basis_mp2="cc-pvdz-ri")
    via_generic = run_double_hybrid(
        mol, basis, "b2plyp",
        density_fit_mp2=True, aux_basis_mp2="cc-pvdz-ri")
    assert direct.e_total == pytest.approx(via_generic.e_total, abs=1e-14)
    assert direct.rks.energy == pytest.approx(
        via_generic.rks.energy, abs=1e-14)
    assert direct.mp2.e_correlation == pytest.approx(
        via_generic.mp2.e_correlation, abs=1e-14)
