"""DF-CCSD and DF-CCSD(T) correlation energies vs PySCF."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, run_rhf
from vibeqc.cc import CCSDOptions, resolve_triples_selector, run_ccsd

from .conftest import GEOMETRIES


def _pyscf_ccsd(atoms_bohr, basis_name, frozen=0):
    """Reference CCSD and CCSD(T) via PySCF with density fitting."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import cc, df, gto, scf

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10

    # Enable density fitting for the SCF
    aux_name = _aux_for_basis(basis_name)
    mf = mf.density_fit(auxbasis=aux_name)
    mf.kernel()
    assert mf.converged, f"PySCF RHF did not converge on {basis_name}"

    # DF-CCSD.  Converge well past the 1e-7 comparison tolerance so the
    # asserts test physics, not solver stopping points (at 1e-8 the two
    # codes' loose fixed points differ by ~2e-7 on H2/sto-3g).
    # Enable DF for CC first: density_fit() builds a fresh DFCCSD object,
    # so tolerances must be set on the returned instance.
    mycc = cc.CCSD(mf, frozen=frozen).density_fit(auxbasis=aux_name)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-9
    mycc.diis_space = 6

    # kernel() returns (e_corr, t1, t2).
    e_ccsd_corr = mycc.kernel()[0]
    assert mycc.converged, f"PySCF CCSD did not converge on {basis_name}"
    e_ccsd = mycc.e_tot

    # CCSD(T)
    e_t = mycc.ccsd_t()
    e_ccsd_t = e_ccsd + e_t

    return {
        "e_hf": mf.e_tot,
        "e_ccsd_corr": e_ccsd_corr,
        "e_ccsd": e_ccsd,
        "e_t": e_t,
        "e_ccsd_t": e_ccsd_t,
    }


def _aux_for_basis(basis_name):
    """Map basis name to a suitable DF auxiliary basis for PySCF."""
    mapping = {
        "sto-3g": "cc-pvdz-ri",
        "6-31g*": "cc-pvdz-ri",
        "cc-pvdz": "cc-pvdz-ri",
        "cc-pvtz": "cc-pvtz-ri",
    }
    return mapping.get(basis_name, "cc-pvdz-ri")


def test_ccsd_options_triples_selector():
    """CCSDOptions exposes the public triples= selector surface."""
    opts = CCSDOptions(triples="none")
    assert opts.compute_triples is False
    assert opts.triples == "none"

    opts.triples = "(T)"
    assert opts.compute_triples is True
    assert opts.triples == "(t)"

    assert resolve_triples_selector(True) is True
    assert resolve_triples_selector("standard") is True
    assert resolve_triples_selector("off") is False

    # bracket correction: implemented selector (see test_cc_variants.py)
    opts.triples = "CCSD[T]"
    assert opts.compute_triples is True
    assert opts.triples == "[t]"

    opts.triples = "A-CCSD(T)"
    assert opts.compute_triples is True
    assert opts.triples == "a-ccsd(t)"

    with pytest.raises(ValueError, match="triples="):
        resolve_triples_selector("full-ccsdt")


def _vibeqc_cc(atoms_bohr, basis_name, frozen=0):
    """Run vibe-qc RHF + DF-CCSD(T)."""
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)

    # RHF
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.density_fit = True
    opts.aux_basis = _aux_for_basis(basis_name)
    hf = run_rhf(mol, basis, opts)
    assert hf.converged, f"vibe-qc RHF did not converge on {basis_name}"

    # CCSD(T); converged past the comparison tolerance (see _pyscf_ccsd)
    cc_opts = CCSDOptions()
    cc_opts.aux_basis = _aux_for_basis(basis_name)
    cc_opts.n_frozen_core = frozen
    cc_opts.compute_triples = True
    cc_opts.conv_tol_energy = 1e-10
    cc_opts.conv_tol_residual = 1e-9
    result = run_ccsd(mol, basis, hf, cc_opts)

    return result


def test_ccsd_blocked_vvvv_matches_incore(monkeypatch):
    """Blocked-vvvv low-memory path reproduces the in-core CCSD/(T) energy.

    The default particle-particle ladder materialises the O(nv^4) (ae|bf)
    tensor. ``VIBEQC_CCSD_VVVV_INCORE_BYTES=0`` forces the blocked path that
    regenerates (ae|bf) panels from B_vv, and ``VIBEQC_CCSD_VVVV_TILE_ROWS=1``
    forces one bra-virtual per tile so the multi-tile column-offset logic is
    exercised. The two paths must agree to ~1e-9 Ha (floating-point
    reassociation from the tiled GEMMs); a real blocking bug shifts the
    correlation energy by mHa. Always-on (no PySCF needed).
    """
    atoms = GEOMETRIES["CH4"]  # n_vir = 4 at sto-3g -> 4 forced tiles
    res_incore = _vibeqc_cc(atoms, "sto-3g")

    monkeypatch.setenv("VIBEQC_CCSD_VVVV_INCORE_BYTES", "0")
    monkeypatch.setenv("VIBEQC_CCSD_VVVV_TILE_ROWS", "1")
    res_blocked = _vibeqc_cc(atoms, "sto-3g")

    assert res_blocked.converged
    assert res_blocked.e_ccsd_correlation == pytest.approx(
        res_incore.e_ccsd_correlation, abs=1e-9
    ), (
        f"blocked-vvvv E_corr {res_blocked.e_ccsd_correlation:.12f} != "
        f"in-core {res_incore.e_ccsd_correlation:.12f}"
    )
    assert res_blocked.e_ccsd == pytest.approx(res_incore.e_ccsd, abs=1e-9)


# Test cases: small molecules, small basis sets for fast CI runs.
CCSD_CASES = [
    ("H2", "sto-3g"),
    ("H2O", "sto-3g"),
    ("CH4", "sto-3g"),
]


@pytest.mark.parametrize(
    "mol_key,basis_name",
    CCSD_CASES,
    ids=[f"{m}-{b}" for m, b in CCSD_CASES],
)
def test_ccsd_energy_matches_pyscf(mol_key, basis_name):
    """DF-CCSD correlation energy agrees with PySCF to 1e-7 Ha.

    Out-of-venv cross-check; skipped when PySCF is not installed. The
    always-on in-repo gate is tests/test_ccsd_anchor.py (spin-orbital
    SGWB-1991 anchor).
    """
    atoms = GEOMETRIES[mol_key]
    ref = _pyscf_ccsd(atoms, basis_name)
    result = _vibeqc_cc(atoms, basis_name)

    tol = 1e-7
    assert result.converged, f"CCSD did not converge for {mol_key}/{basis_name}"
    assert abs(result.e_ccsd_correlation - ref["e_ccsd_corr"]) < tol, (
        f"{mol_key}/{basis_name}: "
        f"E_corr vibe-qc = {result.e_ccsd_correlation:.12f}, "
        f"E_corr pyscf = {ref['e_ccsd_corr']:.12f}, "
        f"diff = {result.e_ccsd_correlation - ref['e_ccsd_corr']:+.2e}"
    )
    assert abs(result.e_ccsd - ref["e_ccsd"]) < tol, (
        f"{mol_key}/{basis_name}: "
        f"E_CCSD vibe-qc = {result.e_ccsd:.12f}, "
        f"E_CCSD pyscf = {ref['e_ccsd']:.12f}, "
        f"diff = {result.e_ccsd - ref['e_ccsd']:+.2e}"
    )


@pytest.mark.parametrize(
    "mol_key,basis_name",
    CCSD_CASES,
    ids=[f"{m}-{b}" for m, b in CCSD_CASES],
)
def test_ccsd_t_energy_matches_pyscf(mol_key, basis_name):
    """DF-CCSD(T) total energy agrees with PySCF to 1e-6 Ha.

    Both codes evaluate the same Raghavachari 1989 formulas; the
    tolerance allows for convergence-threshold differences in the
    underlying amplitudes.
    """
    atoms = GEOMETRIES[mol_key]
    ref = _pyscf_ccsd(atoms, basis_name)
    result = _vibeqc_cc(atoms, basis_name)

    tol = 1e-6
    assert result.converged
    assert abs(result.e_t - ref["e_t"]) < tol, (
        f"{mol_key}/{basis_name}: "
        f"E_T vibe-qc = {result.e_t:.12f}, "
        f"E_T pyscf = {ref['e_t']:.12f}, "
        f"diff = {result.e_t - ref['e_t']:+.2e}"
    )
    assert abs(result.e_ccsd_t - ref["e_ccsd_t"]) < tol, (
        f"{mol_key}/{basis_name}: "
        f"E_CCSD(T) vibe-qc = {result.e_ccsd_t:.12f}, "
        f"E_CCSD(T) pyscf = {ref['e_ccsd_t']:.12f}, "
        f"diff = {result.e_ccsd_t - ref['e_ccsd_t']:+.2e}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "mol_key,basis_name",
    [("H2O", "cc-pvdz"), ("CH4", "cc-pvdz")],
    ids=["H2O-cc-pvdz-fc", "CH4-cc-pvdz-fc"],
)
def test_ccsd_t_frozen_core_matches_pyscf(mol_key, basis_name):
    """Closed-shell frozen-core DF-CCSD(T) agrees with PySCF frozen=1 to 1e-6 Ha.

    This is the window run_job uses by default: it freezes chemical cores via
    vibeqc.cc.chemical_core_orbital_count (H2O / CH4 each freeze their single
    1s core). The all-electron cross-checks above never exercise the frozen
    path, so this pins the default closed-shell run_job kernel against PySCF.
    The open-shell sibling is tests/test_uccsd.py::
    test_uccsd_t_frozen_core_matches_pyscf.
    """
    atoms = GEOMETRIES[mol_key]
    frozen = 1  # O 1s (H2O) / C 1s (CH4) chemical core
    ref = _pyscf_ccsd(atoms, basis_name, frozen=frozen)
    result = _vibeqc_cc(atoms, basis_name, frozen=frozen)

    tol = 1e-6
    assert result.converged
    assert abs(result.e_ccsd_correlation - ref["e_ccsd_corr"]) < tol, (
        f"{mol_key}/{basis_name} fc: "
        f"E_corr vibe-qc={result.e_ccsd_correlation:.12f}, "
        f"pyscf={ref['e_ccsd_corr']:.12f}, "
        f"diff={result.e_ccsd_correlation - ref['e_ccsd_corr']:+.2e}"
    )
    assert abs(result.e_t - ref["e_t"]) < tol
    assert abs(result.e_ccsd_t - ref["e_ccsd_t"]) < tol, (
        f"{mol_key}/{basis_name} fc: "
        f"E_CCSD(T) vibe-qc={result.e_ccsd_t:.12f}, "
        f"pyscf={ref['e_ccsd_t']:.12f}, "
        f"diff={result.e_ccsd_t - ref['e_ccsd_t']:+.2e}"
    )


# Cached ORCA 6.1.1 closed-shell cross-check references (out-of-process;
# CLAUDE.md section 10). Generated 2026-06-20 with
#   ! RHF CCSD(T) cc-pVDZ NoFrozenCore TightSCF Bohrs
# on the GEOMETRIES["H2O"] / ["CH4"] coordinates. ORCA uses conventional
# (non-DF) integrals, so the *totals* differ from vibe-qc's DF-CCSD(T) by
# the cc-pVDZ density-fitting error (~0.8-1.2 mHa); the perturbative (T)
# correction is DF-insensitive and agrees independently to ~7 uHa. The
# tight same-DF parity check is the PySCF cross-check above. This mirrors
# the open-shell tests/test_uccsd.py::test_uccsd_t_orca_cross_check.
ORCA_CCPVDZ_CLOSED = {
    "H2O": {"e_ccsd_t": -76.240621215, "e_t": -0.003280974},
    "CH4": {"e_ccsd_t": -40.389434055, "e_t": -0.003718015},
}


@pytest.mark.slow
@pytest.mark.parametrize("mol_key", ["H2O", "CH4"])
def test_ccsd_t_orca_cross_check(mol_key):
    """Independent-code cross-check vs cached conventional ORCA 6.1 CCSD(T)."""
    atoms = GEOMETRIES[mol_key]
    result = _vibeqc_cc(atoms, "cc-pvdz", frozen=0)
    orca = ORCA_CCPVDZ_CLOSED[mol_key]
    assert result.converged
    # (T) is density-fitting-insensitive: tight agreement with conventional ORCA.
    assert abs(result.e_t - orca["e_t"]) < 3e-5, (
        f"{mol_key}: E_T vibe-qc={result.e_t:.9f}, "
        f"orca={orca['e_t']:.9f}, diff={result.e_t - orca['e_t']:+.2e}"
    )
    # Total carries the cc-pVDZ DF error vs ORCA's conventional integrals.
    assert abs(result.e_ccsd_t - orca["e_ccsd_t"]) < 2e-3


def test_ccsd_rejects_unconverged_hf():
    """CCSD requires a converged RHF reference."""
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    hf = run_rhf(mol, basis, opts)
    assert not hf.converged
    with pytest.raises(RuntimeError, match="not converged"):
        run_ccsd(mol, basis, hf, CCSDOptions())


def test_ccsd_rejects_open_shell():
    """CCSD requires a closed-shell reference."""
    ref_mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(ref_mol, "sto-3g")
    hf = run_rhf(ref_mol, basis, RHFOptions())
    assert hf.converged

    open_shell_mol = Molecule([Atom(1, [0, 0, 0])], charge=0, multiplicity=2)
    opts = CCSDOptions(density_fit=False)
    with pytest.raises(ValueError, match="closed-shell"):
        run_ccsd(open_shell_mol, basis, hf, opts)


def test_ccsd_trivial_water():
    """Smoke test: CCSD runs on water / cc-pVDZ with DF."""
    atoms = GEOMETRIES["H2O"]

    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms])
    basis = BasisSet(mol, "cc-pvdz")

    # RHF with DF
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.density_fit = True
    opts.aux_basis = "cc-pvdz-ri"
    hf = run_rhf(mol, basis, opts)
    assert hf.converged

    # CCSD only (no triples, faster)
    cc_opts = CCSDOptions()
    cc_opts.aux_basis = "cc-pvdz-ri"
    cc_opts.compute_triples = False
    result = run_ccsd(mol, basis, hf, cc_opts)

    assert result.converged
    assert result.n_iter > 0
    assert result.n_iter < 100
    assert result.e_ccsd_correlation < 0  # correlation is negative
    assert abs(result.e_ccsd - (result.e_hf + result.e_ccsd_correlation)) < 1e-14
    assert result.t1_norm >= 0
    assert result.t2_norm >= 0
