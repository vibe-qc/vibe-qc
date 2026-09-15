"""Canonical (non-DF) CCSD / CCSD(T) and UCCSD / UCCSD(T): exact integrals.

``CCSDOptions(density_fit=False)`` selects the conventional coupled-cluster
route: the MO integral blocks are assembled from the exact four-index AO ERI
tensor (eri_mo_pair_transform) instead of the RI-fitted B factorisation.
Everything downstream of the integral assembly -- the SGWB-1991 amplitude
equations, DIIS, and the Raghavachari (T) -- is shared with the DF path, so
these tests pin exactly the piece that is new: the integral assembly.

This is the strict-conventional parity route for the release paper (M12 /
M12u: all-electron conventional CCSD(T) / UCCSD(T) vs ORCA), previously
blocked -- rerun job df5eec9adb96 hit "canonical (non-DF) CCSD is not
implemented" on H2O and OH / cc-pVDZ.

Always-on gates (no external QC program, CLAUDE.md section 10):

* the in-repo spin-orbital SGWB anchor (``run_ref_ccsd`` /
  ``run_ref_uccsd``) fed *exact* integrals through an eigenvalue
  factorisation of the ERI matrix -- machine-precision-grade agreement;
* the closed-shell cross-kernel identity: canonical UCCSD on a
  spin-restricted UHF solution equals canonical spin-adapted CCSD;
* canonical-vs-DF proximity (same system, both routes converge to
  correlation energies within the RI fitting error);
* run_job provenance: no "DF-" label, "Density fitting = off", and no RI
  auxiliary basis is required or reported.

The out-of-process oracle (skipped without PySCF): conventional
(exact-integral) CCSD(T) / UCCSD(T) on the paper configurations, validated
here at < 5e-8 Ha agreement of the correlation and (T) pieces (2026-07-02).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    UHFOptions,
    run_job,
    run_rhf,
    run_uhf,
)
from vibeqc._vibeqc_core import compute_eri
from vibeqc.cc import CCSDOptions, run_ccsd, run_uccsd
from vibeqc.dlpno._ccsd_ref import run_ref_ccsd, run_ref_uccsd

ANG = 1.8897259886

# H2O (Angstrom -> Bohr): the release-paper M12 geometry class
# (equilibrium water); OH doublet is the M12u geometry class.
H2O_ATOMS = [
    (8, [0.0, 0.0, 0.1173 * ANG]),
    (1, [0.0, 0.7572 * ANG, -0.4692 * ANG]),
    (1, [0.0, -0.7572 * ANG, -0.4692 * ANG]),
]
OH_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 0.9697 * ANG]),
]


def _h2o(multiplicity: int = 1) -> Molecule:
    return Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in H2O_ATOMS],
        charge=0,
        multiplicity=multiplicity,
    )


def _oh() -> Molecule:
    return Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in OH_ATOMS],
        charge=0,
        multiplicity=2,
    )


def _rhf(mol: Molecule, basis_name: str):
    basis = BasisSet(mol, basis_name)
    o = RHFOptions()
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    hf = run_rhf(mol, basis, o)
    assert hf.converged
    return basis, hf


def _uhf(mol: Molecule, basis_name: str):
    basis = BasisSet(mol, basis_name)
    o = UHFOptions()
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    o.max_iter = 300
    hf = run_uhf(mol, basis, o)
    assert hf.converged
    return basis, hf


def _canonical_opts(**kw) -> CCSDOptions:
    opts = CCSDOptions(density_fit=False, n_frozen_core=0, **kw)
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_residual = 1e-9
    return opts


def _factorize_pair_matrix(W: np.ndarray) -> np.ndarray:
    """Exact rank-revealing factorisation W = B^T B of a symmetric PSD
    ERI pair matrix -- the "exact B-tensor" that lets the DF-shaped
    reference kernels run on exact integrals (reconstruction error at the
    eigenvalue-clip level, ~1e-13 relative)."""
    w, U = np.linalg.eigh(W)
    keep = w > w.max() * 1e-13
    return (U[:, keep] * np.sqrt(w[keep])).T  # (n_aux_eff, n_pair)


def test_canonical_ccsd_t_matches_exact_integral_so_anchor():
    """Canonical C++ CCSD(T) == spin-orbital reference on exact integrals.

    H2O/STO-3G all-electron.  The reference kernel consumes a B-tensor;
    feeding it the eigen-factorised exact MO ERI matrix makes it an exact-
    integral (conventional) oracle, so this pins the new canonical
    integral assembly against the FCI-anchored SGWB reference without any
    RI fitting on either side.
    """
    mol = _h2o()
    basis, hf = _rhf(mol, "sto-3g")

    res = run_ccsd(mol, basis, hf, _canonical_opts(compute_triples=True))
    assert res.converged

    C = np.asarray(hf.mo_coeffs)
    n_mo = C.shape[1]
    eri_ao = np.asarray(compute_eri(basis))
    eri_mo = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl", C, C, C, C, eri_ao, optimize=True
    )
    B_exact = _factorize_pair_matrix(
        eri_mo.reshape(n_mo * n_mo, n_mo * n_mo)
    ).reshape(-1, n_mo, n_mo)
    f_mo = C.T @ np.asarray(hf.fock) @ C
    ref = run_ref_ccsd(
        f_mo,
        B_exact,
        mol.n_electrons() // 2,
        e_hf=hf.energy,
        conv_tol=1e-11,
        compute_triples=True,
    )
    assert ref.converged

    assert abs(res.e_ccsd_correlation - ref.e_corr) < 5e-8
    assert abs(res.e_t - ref.e_t) < 5e-9


def test_canonical_uccsd_t_matches_exact_integral_so_anchor():
    """Canonical C++ UCCSD(T) == spin-orbital reference on exact integrals.

    OH doublet / STO-3G all-electron.  The exact AO ERI matrix is
    eigen-factorised once (B_ao) and MO-transformed per spin channel, so
    all three spin channels of the reference's (pq|rs) are exact.
    """
    mol = _oh()
    basis, uhf = _uhf(mol, "sto-3g")

    res = run_uccsd(mol, basis, uhf, _canonical_opts(compute_triples=True))
    assert res.converged

    Ca = np.asarray(uhf.mo_coeffs_alpha)
    Cb = np.asarray(uhf.mo_coeffs_beta)
    nao = Ca.shape[0]
    eri_ao = np.asarray(compute_eri(basis))
    B_ao = _factorize_pair_matrix(eri_ao.reshape(nao * nao, nao * nao)).reshape(
        -1, nao, nao
    )
    B_a = np.einsum("Ppq,pi,qj->Pij", B_ao, Ca, Ca, optimize=True)
    B_b = np.einsum("Ppq,pi,qj->Pij", B_ao, Cb, Cb, optimize=True)
    n_elec = mol.n_electrons()
    n_a = (n_elec + mol.multiplicity - 1) // 2
    n_b = n_elec - n_a
    ref = run_ref_uccsd(
        Ca.T @ np.asarray(uhf.fock_alpha) @ Ca,
        Cb.T @ np.asarray(uhf.fock_beta) @ Cb,
        B_a,
        B_b,
        n_a,
        n_b,
        e_hf=uhf.energy,
        conv_tol=1e-11,
        compute_triples=True,
    )
    assert ref.converged

    assert abs(res.e_ccsd_correlation - ref.e_corr) < 5e-8
    assert abs(res.e_t - ref.e_t) < 5e-9


def test_canonical_uccsd_equals_ccsd_on_closed_shell():
    """Cross-kernel identity: canonical UCCSD == canonical CCSD (singlet).

    On a spin-restricted UHF solution of closed-shell H2O the spin-orbital
    unrestricted kernel and the spin-adapted closed-shell kernel must give
    the same correlation and (T) energies.  Both use exact integrals here,
    so any discrepancy is an integral-assembly bug in one of the two new
    canonical builders.
    """
    mol = _h2o()
    basis, rhf = _rhf(mol, "sto-3g")
    basis_u, uhf = _uhf(_h2o(), "sto-3g")
    assert abs(rhf.energy - uhf.energy) < 1e-9  # spin-restricted solution

    r_r = run_ccsd(mol, basis, rhf, _canonical_opts(compute_triples=True))
    r_u = run_uccsd(_h2o(), basis_u, uhf, _canonical_opts(compute_triples=True))
    assert r_r.converged and r_u.converged
    assert abs(r_r.e_ccsd_correlation - r_u.e_ccsd_correlation) < 1e-7
    assert abs(r_r.e_t - r_u.e_t) < 1e-8


def test_canonical_vs_df_proximity():
    """Canonical and DF-CCSD agree to RI-fitting accuracy (and are distinct).

    H2O/cc-pVDZ with cc-pvdz-ri: the RI error on the correlation energy is
    ~1e-5..1e-4 Ha; a canonical route that silently fell through to the DF
    tensors would agree to machine precision instead.
    """
    mol = _h2o()
    basis, hf = _rhf(mol, "cc-pvdz")

    r_canon = run_ccsd(mol, basis, hf, _canonical_opts(compute_triples=False))
    df_opts = CCSDOptions(
        density_fit=True,
        aux_basis="cc-pvdz-ri",
        n_frozen_core=0,
        compute_triples=False,
    )
    df_opts.conv_tol_energy = 1e-10
    df_opts.conv_tol_residual = 1e-9
    r_df = run_ccsd(mol, basis, hf, df_opts)
    assert r_canon.converged and r_df.converged

    diff = abs(r_canon.e_ccsd_correlation - r_df.e_ccsd_correlation)
    assert diff < 5e-4  # same physics
    assert diff > 1e-9  # genuinely different integral routes


def test_run_job_canonical_provenance_closed_shell(tmp_path):
    """run_job CCSD(T) with density_fit=False: truthful labels, no aux.

    Regression for the df5eec9adb96 symptom where the stdout stage and the
    .out block claimed "DF-CCSD(T)" although density_fit=False was
    requested.  STO-3G has no registered RI auxiliary basis, so this also
    proves the canonical route needs none.
    """
    stem = tmp_path / "h2o_canonical"
    res = run_job(
        _h2o(),
        basis="sto-3g",
        method="ccsd(t)",
        ccsd_options=CCSDOptions(density_fit=False, n_frozen_core=0),
        output=stem,
        structured_log=True,
    )
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Algorithm            = CCSD(T)" in out
    assert "DF-CCSD" not in out
    assert "Density fitting       = off" in out
    assert "RI auxiliary basis" not in out

    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    ev = next(r for r in records if r.get("event") == "ccsd_converged")
    assert ev["algorithm"] == "CCSD(T)"
    assert ev["density_fit"] is False
    assert ev["aux_basis"] == ""

    # Citation surface: the defining CC + (T) papers fire, the DF-CCSD
    # integral-assembly paper (DePrince-Sherrill 2013) must not -- no
    # density fitting of the CC integrals took place.
    assert "Raghavachari" in out
    assert "DePrince" not in out


def test_run_job_canonical_provenance_open_shell(tmp_path):
    """run_job UCCSD(T) with density_fit=False: truthful open-shell label."""
    stem = tmp_path / "oh_canonical"
    res = run_job(
        _oh(),
        basis="sto-3g",
        method="ccsd(t)",
        ccsd_options=CCSDOptions(density_fit=False, n_frozen_core=0),
        output=stem,
    )
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Job: UHF + CCSD(T)" in out
    assert "Algorithm            = UCCSD(T)" in out
    assert "DF-UCCSD" not in out
    assert "Density fitting       = off" in out


def test_run_job_canonical_rohf_reference(tmp_path):
    """Canonical route through the third entry point (run_uccsd_from_mos).

    ccsd_reference="rohf" drives run_rohf_ccsd -> run_uccsd_from_mos, so
    this covers the from-explicit-arrays path with density_fit=False.
    """
    stem = tmp_path / "oh_rohf_canonical"
    res = run_job(
        _oh(),
        basis="sto-3g",
        method="ccsd(t)",
        ccsd_reference="rohf",
        ccsd_options=CCSDOptions(density_fit=False, n_frozen_core=0),
        output=stem,
    )
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Job: ROHF + CCSD(T)" in out
    assert "Algorithm            = ROHF-CCSD(T)" in out
    assert "DF-ROHF-CCSD" not in out
    assert "Density fitting       = off" in out


# ===========================================================================
# Out-of-venv conventional oracle (the release-paper M12 / M12u configs).
# PySCF is a parity oracle, never a vibe-qc dependency (CLAUDE.md sec. 10).
# ===========================================================================


def _pyscf_conventional_rccsd_t(atoms_ang, basis_name):
    pytest.importorskip("pyscf")
    from pyscf import cc, gto, scf

    m = gto.Mole()
    m.unit = "Bohr"
    m.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_ang]
    m.basis = basis_name
    m.verbose = 0
    m.build()
    mf = scf.RHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged
    mycc = cc.CCSD(mf, frozen=0)  # conventional: all-electron exact integrals
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-9
    e_corr = mycc.kernel()[0]
    assert mycc.converged
    e_t = mycc.ccsd_t()
    return e_corr, e_t


def _pyscf_conventional_uccsd_t(atoms, basis_name, mult):
    pytest.importorskip("pyscf")
    from pyscf import cc, gto, scf

    m = gto.Mole()
    m.unit = "Bohr"
    m.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    m.basis = basis_name
    m.spin = mult - 1
    m.verbose = 0
    m.build()
    mf = scf.UHF(m)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    assert mf.converged
    mycc = cc.UCCSD(mf, frozen=0)  # conventional: all-electron exact integrals
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-9
    e_corr = mycc.kernel()[0]
    assert mycc.converged
    e_t = mycc.ccsd_t()
    return e_corr, e_t


def test_canonical_ccsd_t_matches_pyscf_conventional_h2o_ccpvdz():
    """M12 config: all-electron conventional CCSD(T), H2O/cc-pVDZ."""
    ref_corr, ref_t = _pyscf_conventional_rccsd_t(H2O_ATOMS, "cc-pvdz")
    mol = _h2o()
    basis, hf = _rhf(mol, "cc-pvdz")
    res = run_ccsd(mol, basis, hf, _canonical_opts(compute_triples=True))
    assert res.converged
    # Measured 2026-07-02: |d corr| = 4.7e-9, |d (T)| = 3.3e-10 Ha.
    assert abs(res.e_ccsd_correlation - ref_corr) < 5e-8
    assert abs(res.e_t - ref_t) < 5e-9


def test_canonical_uccsd_t_matches_pyscf_conventional_oh_ccpvdz():
    """M12u config: all-electron conventional UCCSD(T), OH doublet/cc-pVDZ."""
    ref_corr, ref_t = _pyscf_conventional_uccsd_t(OH_ATOMS, "cc-pvdz", 2)
    mol = _oh()
    basis, uhf = _uhf(mol, "cc-pvdz")
    res = run_uccsd(mol, basis, uhf, _canonical_opts(compute_triples=True))
    assert res.converged
    # Measured 2026-07-02: |d corr| = 2.9e-9, |d (T)| = 1.7e-10 Ha.
    assert abs(res.e_ccsd_correlation - ref_corr) < 5e-8
    assert abs(res.e_t - ref_t) < 5e-9


# ---------------------------------------------------------------------------
# run_job(density_fit=...) governs the correlated route, not just the SCF.
#
# Reproducer for CCSDT-SILENT-DF-DEFAULT: the top-level ``density_fit``
# kwarg was wired onto the RHF/UHF/RKS/UKS option structs only, so
# ``run_job(method="ccsd(t)", density_fit=False)`` ran DF-CCSD(T) and
# said nothing. Payloads have no ``ccsd_options=`` surface at all, so a
# campaign asking for a conventional CCSD(T) leg had no way to get one
# and its parity partner silently compared DF against canonical.
# ---------------------------------------------------------------------------

def test_run_job_density_fit_false_selects_canonical_ccsd(tmp_path):
    """The kwarg reaches the CC route. STO-3G has no registered RI aux,
    so a DF route here would also have to raise -- proving the canonical
    route was genuinely taken and the aux preflight stood down."""
    stem = tmp_path / "h2o_kwarg_canonical"
    res = run_job(
        _h2o(),
        basis="sto-3g",
        method="ccsd(t)",
        density_fit=False,
        output=stem,
        structured_log=True,
    )
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Algorithm            = CCSD(T)" in out
    assert "DF-CCSD" not in out
    assert "Density fitting       = off" in out
    assert "RI auxiliary basis" not in out

    records = [
        json.loads(line)
        for line in stem.with_suffix(".scf.jsonl").read_text().splitlines()
    ]
    ev = next(r for r in records if r.get("event") == "ccsd_converged")
    assert ev["algorithm"] == "CCSD(T)"
    assert ev["density_fit"] is False


def test_run_job_default_still_density_fits_the_cc_route(tmp_path):
    """Unchanged default: no ``density_fit=`` supplied means DF-CCSD(T),
    the documented production route. The bug fix must not flip it."""
    stem = tmp_path / "h2o_default_df"
    res = run_job(
        _h2o(),
        basis="cc-pvdz",
        method="ccsd(t)",
        output=stem,
    )
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Algorithm            = DF-CCSD(T)" in out
    assert "Density fitting       = on" in out
    assert "RI auxiliary basis    = cc-pvdz-ri" in out


def test_run_job_density_fit_true_keeps_the_df_cc_route(tmp_path):
    stem = tmp_path / "h2o_kwarg_df"
    res = run_job(
        _h2o(),
        basis="cc-pvdz",
        method="ccsd(t)",
        density_fit=True,
        output=stem,
    )
    assert res.ccsd.converged
    out = stem.with_suffix(".out").read_text()
    assert "Algorithm            = DF-CCSD(T)" in out
    assert "Density fitting       = on" in out


def test_run_job_conflicting_density_fit_and_ccsd_options_raises(tmp_path):
    """Never silently pick a winner between two contradicting requests."""
    with pytest.raises(ValueError, match="density_fit"):
        run_job(
            _h2o(),
            basis="cc-pvdz",
            method="ccsd(t)",
            density_fit=False,
            ccsd_options=CCSDOptions(density_fit=True),
            output=tmp_path / "h2o_conflict",
        )


def test_run_job_agreeing_density_fit_and_ccsd_options_is_accepted(tmp_path):
    """Redundant but consistent -- not an error."""
    stem = tmp_path / "h2o_agree"
    res = run_job(
        _h2o(),
        basis="sto-3g",
        method="ccsd(t)",
        density_fit=False,
        ccsd_options=CCSDOptions(density_fit=False, n_frozen_core=0),
        output=stem,
    )
    assert res.ccsd.converged
    assert "Density fitting       = off" in stem.with_suffix(".out").read_text()
