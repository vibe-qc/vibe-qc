"""RKS Kohn-Sham DFT cross-checks against PySCF."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, RKSOptions, run_rks

from .conftest import GEOMETRIES


def _pyscf_rks(atoms_bohr, basis_name, xc):
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
    mf.grids.level = 5  # fine, for a trustworthy reference
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    return mf.e_tot


def _vibeqc_rks(atoms_bohr, basis_name, functional):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-9
    opts.conv_tol_grad = 1e-7
    opts.max_iter = 200
    return run_rks(mol, basis, opts)


# (label, vibe-qc name, pyscf xc string)
#
# PW1PW is Bredow & Gerson's 1-parameter global hybrid (Phys. Rev. B 61,
# 5194 (2000)): 0.20·HF + 0.80·PW91-X + PW91-C. PySCF has no shorthand,
# so the reference is built via parse_xc-style hand-mixing — exactly the
# spec the vibe-qc alias encodes.
FUNCTIONAL_PAIRS = [
    ("LDA",   "LDA",   "lda,vwn"),
    ("PBE",   "PBE",   "pbe,pbe"),
    # vibe-qc's "B3LYP" = ORCA/VWN5 definition; PySCF's matching
    # spelling is "b3lyp5" (PySCF's plain "b3lyp" is the Gaussian/
    # VWN-RPA variant == vibe-qc's "b3lyp/g" / "b3lypg"; see
    # tests/test_b3lyp_convention.py for the cross-code pins).
    ("B3LYP", "B3LYP", "b3lyp5"),
    ("PW1PW", "PW1PW", "0.2*HF + 0.8*GGA_X_PW91, GGA_C_PW91"),
]

RKS_CASES = [
    ("H2O", "sto-3g"),
    ("H2O", "6-31g*"),
    ("CH4", "sto-3g"),
]


@pytest.mark.parametrize("mol_key,basis_name", RKS_CASES,
                         ids=[f"{m}-{b}" for m, b in RKS_CASES])
@pytest.mark.parametrize("label,vq_name,ps_name", FUNCTIONAL_PAIRS,
                         ids=[c[0] for c in FUNCTIONAL_PAIRS])
def test_rks_matches_pyscf(mol_key, basis_name, label, vq_name, ps_name):
    atoms = GEOMETRIES[mol_key]
    result = _vibeqc_rks(atoms, basis_name, vq_name)
    ref_energy = _pyscf_rks(atoms, basis_name, ps_name)

    assert result.converged, (
        f"{label} / {mol_key} / {basis_name}: vibeqc SCF did not converge"
    )
    # Default grid is "medium" (75×17×36); expect ~1e-6 Ha agreement for
    # GGAs and hybrids, tighter for LDA. Allow 3e-6 as a comfortable bound.
    assert abs(result.energy - ref_energy) < 3e-6, (
        f"{label} / {mol_key} / {basis_name}: "
        f"E_vibeqc = {result.energy:.10f}, E_pyscf = {ref_energy:.10f}, "
        f"diff = {result.energy - ref_energy:+.2e}"
    )


def test_rks_lda_matches_pyscf_at_machine_precision():
    """LDA has no gradient integration, so agreement with PySCF is
    dominated by electronic-energy-grid convergence of the exchange density
    only; we usually get < 1e-10 on default grids."""
    atoms = GEOMETRIES["H2O"]
    result = _vibeqc_rks(atoms, "sto-3g", "LDA")
    ref = _pyscf_rks(atoms, "sto-3g", "lda,vwn")
    assert abs(result.energy - ref) < 1e-9


def test_rks_refuses_open_shell():
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="even electron count|multiplicity"):
        run_rks(mol, basis)


def test_rks_result_energy_components_sum_to_total():
    atoms = GEOMETRIES["H2O"]
    result = _vibeqc_rks(atoms, "sto-3g", "B3LYP")
    # E = E_core (1e) + E_J + E_K (hybrid HF part) + E_xc + E_nuc
    # E_electronic = E - E_nuc = E_core + E_J + E_K + E_xc
    # E_core = E_electronic - E_coulomb - E_hf_exchange - E_xc
    E_core = (result.e_electronic
              - result.e_coulomb - result.e_hf_exchange - result.e_xc)
    total = E_core + result.e_coulomb + result.e_hf_exchange + result.e_xc + result.e_nuclear
    assert abs(total - result.energy) < 1e-10


def test_rks_hybrid_reports_nonzero_hf_exchange_energy():
    atoms = GEOMETRIES["H2O"]
    result = _vibeqc_rks(atoms, "sto-3g", "B3LYP")
    assert result.e_hf_exchange != 0.0


def test_rks_pure_dft_has_zero_hf_exchange_energy():
    atoms = GEOMETRIES["H2O"]
    result = _vibeqc_rks(atoms, "sto-3g", "PBE")
    assert result.e_hf_exchange == 0.0


# ---------------------------------------------------------------------------
# Issue #144 — RKS restricted-stability VERDICT surface (decision-neutral
# prerequisite). The RKS driver reports the lowest orbital-rotation Hessian
# eigenvalue at the converged restricted solution (closed-shell KS singlet +
# triplet sectors, Bauernschmitt-Ahlrichs 1996; polarised XC kernel at
# (D/2, D/2)) and warns loudly when negative. Verdict only — no rotation,
# escape, or promotion; the policy for a negative verdict is the maintainer
# decision in
# agentic-loop/asks/ask-scf144-restricted-stability-contract-2026-08-27.md.
# ---------------------------------------------------------------------------


def test_rks_restricted_stability_verdict_stable_water():
    """#144: H2O/PBE/STO-3G is exactly stable and reports the verdict."""
    atoms = GEOMETRIES["H2O"]
    result = _vibeqc_rks(atoms, "sto-3g", "PBE")
    assert result.converged
    assert result.stability_checked is True
    assert result.stability_analysis_converged is True
    assert result.internal_instability is False
    assert result.stability_eigenvalue > 0.0


def test_rks_restricted_stability_opt_out_same_energy():
    """#144 negative control (L125): same route with the feature off.

    stability_check=False returns the identical energy, records no
    verdict, and marks the request explicit (the fail-closed contract).
    """
    atoms = GEOMETRIES["H2O"]
    opts = RKSOptions()
    opts.functional = "PBE"
    opts.stability_check = False
    mol = Molecule([Atom(int(Z), np.array(xyz, dtype=float)) for Z, xyz in atoms])
    res_off = run_rks(mol, BasisSet(mol, "sto-3g"), opts)
    res_on = _vibeqc_rks(atoms, "sto-3g", "PBE")
    assert res_off.converged
    assert res_off.stability_checked is False
    assert res_off.energy == res_on.energy
    assert opts._stability_check_explicit is True


def test_rks_restricted_stability_mgga_skips_implicit_fails_explicit():
    """#144 guard rail: meta-GGA tau-dependent fxc is not plumbed.

    The default-on check must not manufacture a verdict from an
    incomplete Hessian: an implicit request skips silently
    (stability_checked=False), an explicit request fails closed.
    """
    mol = Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 1.43, -0.98]),
         Atom(1, [0.0, -1.43, -0.98])],
    )
    basis = BasisSet(mol, "sto-3g")
    opts = RKSOptions()
    opts.functional = "TPSS"
    res = run_rks(mol, basis, opts)
    assert res.converged
    assert res.stability_checked is False

    opts_explicit = RKSOptions()
    opts_explicit.functional = "TPSS"
    opts_explicit.stability_check = True  # marks explicit
    with pytest.raises(RuntimeError, match="tau-dependent"):
        run_rks(mol, basis, opts_explicit)
