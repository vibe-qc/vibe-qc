"""B2PLYP (Grimme, *J. Chem. Phys.* **124**, 034108 (2006)) — first
end-to-end double hybrid in vibe-qc.

Pins:

  1. ``Functional("b2plyp")`` resolves to the published recipe:
     0.53 · HF + 0.47 · B88 exchange, 0.73 · LYP correlation, and a
     post-SCF MP2 correction with c_os = c_ss = 0.27 reported via
     ``is_double_hybrid`` / ``mp2_c_os`` / ``mp2_c_ss``.
  2. ``run_b2plyp`` orchestrates the SCF + RI-MP2 dispatch and returns
     a :class:`DoubleHybridResult` carrying both step results and the
     combined total energy.
  3. The combined total agrees with PySCF (custom XC string for the
     SCF + ``mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name)``
     for the explicitly all-electron correction with the exact RI auxiliary)
     to grid accuracy on H2O / cc-pVDZ — 1e-5 Ha (the SCF-grid
     difference between vibe-qc's default DFT grid and PySCF's
     dominates the cross-code gap).
  4. The MP2 correction is c_os · e_os + c_ss · e_ss with
     c_os = c_ss = 0.27.
  5. The dispatcher rejects open-shell molecules at the SCF step
     (B2PLYP for unrestricted systems would be a separate
     U-double-hybrid wire-up, deferred).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    DoubleHybridResult,
    Functional,
    MP2Options,
    Molecule,
    RKSOptions,
    UKSOptions,
    compute_d3bj,
    rhf_result_from_rks,
    run_b2plyp,
    run_dsd_pbep86,
    run_job,
    run_mp2,
    run_rks,
    run_uks,
)

from .conftest import ANGSTROM_TO_BOHR, GEOMETRIES


B2PLYP_C_OS = 0.27
B2PLYP_C_SS = 0.27


# ---------------------------------------------------------------------
# Functional resolver.
# ---------------------------------------------------------------------

def test_b2plyp_functional_resolves():
    """``Functional("b2plyp")`` carries the SCF hybrid mix and the
    MP2-correction coefficients."""
    f = Functional("b2plyp")
    assert f.hf_exchange_fraction == pytest.approx(0.53, abs=1e-12)
    assert f.is_hybrid is True
    assert f.is_double_hybrid is True
    assert f.mp2_c_os == pytest.approx(B2PLYP_C_OS, abs=1e-12)
    assert f.mp2_c_ss == pytest.approx(B2PLYP_C_SS, abs=1e-12)


def test_non_double_hybrid_functionals_carry_zero_mp2_mix():
    """LDA / PBE / B3LYP / PBE0 / PW1PW: not double hybrids."""
    for name in ("lda", "pbe", "blyp", "b3lyp", "pbe0", "pw1pw"):
        f = Functional(name)
        assert f.is_double_hybrid is False, name
        assert f.mp2_c_os == 0.0, name
        assert f.mp2_c_ss == 0.0, name


# ---------------------------------------------------------------------
# Cross-code parity against PySCF (custom XC string + MP2 correction).
# ---------------------------------------------------------------------

def _pyscf_b2plyp(atoms_bohr, basis_name, aux_basis_name):
    """B2PLYP on PySCF: hand-rolled hybrid SCF + DFMP2 correction with
    c_os = c_ss = 0.27. Libxc doesn't ship a built-in B2PLYP, so the
    SCF piece is composed via PySCF's xc-string parser exactly the way
    vibe-qc composes it internally."""
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, dft, mp

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms_bohr]
    mol.basis = basis_name
    mol.verbose = 0
    mol.build()
    mf = dft.RKS(mol, xc="0.53*HF + 0.47*B88, 0.73*LYP")
    mf.conv_tol = 1e-10
    mf.kernel()
    assert mf.converged
    m = mp.MP2(mf, frozen=0).density_fit(auxbasis=aux_basis_name)
    m.kernel()
    e_corr_dh = B2PLYP_C_OS * m.e_corr_os + B2PLYP_C_SS * m.e_corr_ss
    return mf.e_tot, e_corr_dh, mf.e_tot + e_corr_dh


def test_run_b2plyp_matches_pyscf_h2o_ccpvdz():
    """End-to-end parity on H2O / cc-pVDZ. The cross-code agreement is
    limited by the DFT-grid difference between vibe-qc and PySCF (the
    SCF step alone differs at the µHa level, and the MP2 correction
    runs on vibe-qc's KS orbitals, so the gap is grid-dependent but
    settles to <1e-5 Ha on this case)."""
    atoms = GEOMETRIES["H2O"]
    result = run_b2plyp(atoms_to_mol_basis(atoms, "cc-pvdz")[0],
                        atoms_to_mol_basis(atoms, "cc-pvdz")[1],
                        density_fit_mp2=True, aux_basis_mp2="cc-pvdz-ri")

    e_rks_ps, e_corr_dh_ps, e_total_ps = _pyscf_b2plyp(
        atoms, "cc-pvdz", "cc-pvdz-ri")

    delta_total = result.e_total - e_total_ps
    assert abs(delta_total) < 1e-5, (
        f"vibeqc B2PLYP vs PySCF total gap = {delta_total:+.3e} Ha "
        f"(vibeqc = {result.e_total:.10f}, PySCF = {e_total_ps:.10f})"
    )


def atoms_to_mol_basis(atoms_bohr, basis_name):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr])
    basis = BasisSet(mol, basis_name)
    return mol, basis


def test_run_b2plyp_accepts_basis_name_and_citations_kw(tmp_path):
    mol, _ = atoms_to_mol_basis(GEOMETRIES["H2O"], "sto-3g")
    result = run_b2plyp(
        mol,
        "sto-3g",
        density_fit=False,
        density_fit_mp2=False,
        output=tmp_path / "b2plyp",
        citations=True,
    )

    assert isinstance(result, DoubleHybridResult)
    assert result.functional == "b2plyp"
    assert np.isfinite(result.e_total)


# ---------------------------------------------------------------------
# Dispatcher internals.
# ---------------------------------------------------------------------

def test_run_b2plyp_total_equals_sum_of_parts():
    """e_total = rks.energy + mp2.e_correlation, and the MP2 correlation
    is c_os · e_os + c_ss · e_ss with the functional's coefficients."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    result = run_b2plyp(mol, basis, density_fit_mp2=True,
                        aux_basis_mp2="cc-pvdz-ri")

    assert isinstance(result, DoubleHybridResult)
    assert result.functional == "b2plyp"

    # Decomposition: total = SCF + scaled MP2 correlation.
    assert result.e_total == pytest.approx(
        result.rks.energy + result.mp2.e_correlation, abs=1e-14)

    # MP2 correlation: c_os · e_os + c_ss · e_ss.
    assert result.mp2.e_correlation == pytest.approx(
        B2PLYP_C_OS * result.mp2.e_os + B2PLYP_C_SS * result.mp2.e_ss,
        rel=1e-14)


def test_run_b2plyp_canonical_mp2_correction_path():
    """``density_fit_mp2=False`` falls back to canonical MP2. Mostly a
    parity-validation knob — slow for production basis sizes, but the
    SCF-MP2 sum should be very close to the RI-MP2 result (the only
    difference is the RIfit fit error in the MP2 piece)."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    ri = run_b2plyp(mol, basis, density_fit_mp2=True,
                    aux_basis_mp2="cc-pvdz-ri")
    canonical = run_b2plyp(mol, basis, density_fit_mp2=False)
    # RIfit fit error scaled by c_os/c_ss ~ sub-µHa total on cc-pVDZ.
    assert abs(ri.e_total - canonical.e_total) < 1e-5


def test_rhf_result_from_rks_round_trip():
    """The adapter preserves the fields ``run_mp2`` reads from its
    reference."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    rks_opts = RKSOptions()
    rks_opts.functional = "pbe0"
    rks_opts.conv_tol_energy = 1e-10
    rks_result = run_rks(mol, basis, rks_opts)
    assert rks_result.converged

    rhf_ref = rhf_result_from_rks(rks_result)
    assert rhf_ref.converged is True
    assert rhf_ref.energy == pytest.approx(rks_result.energy, abs=1e-14)
    assert np.allclose(rhf_ref.mo_energies, rks_result.mo_energies)
    assert np.allclose(rhf_ref.mo_coeffs, rks_result.mo_coeffs)
    assert np.allclose(rhf_ref.density, rks_result.density)


def test_run_b2plyp_uses_functional_coefficients_not_hardcoded():
    """The dispatcher reads c_os / c_ss from Functional("b2plyp"), not
    from local constants. Verifies by checking the dispatcher's MP2
    result matches a manual MP2 call using the same coefficients."""
    mol, basis = atoms_to_mol_basis(GEOMETRIES["H2O"], "cc-pvdz")
    f = Functional("b2plyp")

    dispatch = run_b2plyp(mol, basis, density_fit_mp2=True,
                          aux_basis_mp2="cc-pvdz-ri")

    # Rebuild the MP2 piece from the complete dispatcher's SCF component.
    rhf_ref = rhf_result_from_rks(dispatch.rks)
    mp2_opts = MP2Options()
    # The double-hybrid model component is explicitly all-electron.
    mp2_opts.n_frozen_core = 0
    mp2_opts.density_fit = True
    mp2_opts.aux_basis = "cc-pvdz-ri"
    mp2_opts.c_os = f.mp2_c_os
    mp2_opts.c_ss = f.mp2_c_ss
    mp2 = run_mp2(mol, basis, rhf_ref, mp2_opts)

    assert dispatch.e_total == pytest.approx(
        dispatch.rks.energy + mp2.e_correlation, abs=1e-12)


@pytest.mark.parametrize(
    ("route", "options_type", "runner"),
    [
        ("RKS", RKSOptions, run_rks),
        ("UKS", UKSOptions, run_uks),
    ],
)
@pytest.mark.parametrize(
    "functional",
    ["b2plyp", "dsd-pbep86", "revdsd-pbep86", "pwpb95"],
)
def test_plain_ks_double_hybrid_route_fails_closed(
    route,
    options_type,
    runner,
    functional,
):
    mol = Molecule(
        [Atom(1, [0.0, 0.0, -0.7]), Atom(1, [0.0, 0.0, 0.7])],
        0,
        1 if route == "RKS" else 3,
    )
    basis = BasisSet(mol, "sto-3g")
    options = options_type()
    options.functional = functional

    with pytest.raises(
        NotImplementedError,
        match=r"omit the required perturbative correlation.*run_double_hybrid",
    ):
        runner(mol, basis, options)


@pytest.mark.parametrize(
    ("method", "multiplicity"),
    [("rks", 1), ("uks", 3), ("roks", 3)],
)
def test_run_job_double_hybrid_fails_before_output(tmp_path, method, multiplicity):
    mol = Molecule(
        [Atom(1, [0.0, 0.0, -0.7]), Atom(1, [0.0, 0.0, 0.7])],
        0,
        multiplicity,
    )
    stem = tmp_path / method

    with pytest.raises(
        NotImplementedError,
        match=r"omit the required perturbative correlation.*run_double_hybrid",
    ):
        run_job(
            mol,
            basis="sto-3g",
            method=method,
            functional="b2plyp",
            output=stem,
            progress=False,
        )

    assert list(tmp_path.iterdir()) == []


def test_run_b2plyp_open_shell_roks():
    """Open-shell B2PLYP runs the spin-pure ROKS SCF half + a semicanonical
    ROHF-MP2 doubles correction (vibe-qc's first open-shell double hybrid):
    e_total = E(ROKS) + 0.27*(e_os + e_ss), the singles term is excluded
    (doubles-only, like the closed-shell path), and the doubles match a
    standalone run_rohf_mp2 on the converged ROKS orbitals."""
    from vibeqc import run_rohf_mp2

    oh = Molecule([Atom(8, [0, 0, 0]),
                   Atom(1, [0, 0, 0.97 * ANGSTROM_TO_BOHR])], 0, 2)
    basis = BasisSet(oh, "cc-pvdz")
    dh = run_b2plyp(oh, basis)
    assert isinstance(dh, DoubleHybridResult)
    assert dh.rks.converged
    # doubles-only, scaled by the functional's coefficients (no singles term)
    assert dh.mp2.e_singles == 0.0
    assert dh.mp2.e_correlation == pytest.approx(
        B2PLYP_C_OS * dh.mp2.e_os + B2PLYP_C_SS * dh.mp2.e_ss, rel=1e-12)
    # total = SCF + scaled doubles, and the correction lowers the energy
    assert dh.e_total == pytest.approx(
        dh.rks.energy + dh.mp2.e_correlation, abs=1e-12)
    assert dh.e_total < dh.rks.energy
    # the doubles match a standalone ROHF-MP2 on the same ROKS orbitals
    ros = run_rohf_mp2(oh, basis, dh.rks, n_frozen_core=0)
    assert dh.mp2.e_os == pytest.approx(ros.e_os, abs=1e-10)
    assert dh.mp2.e_ss == pytest.approx(ros.e_ss, abs=1e-10)


def test_run_b2plyp_direct_roks_still_gated():
    """A direct run_roks on a double-hybrid functional still raises (the SCF
    energy alone is an incomplete double hybrid), pointing the user to
    run_double_hybrid."""
    from vibeqc import run_roks

    oh = Molecule([Atom(8, [0, 0, 0]),
                   Atom(1, [0, 0, 0.97 * ANGSTROM_TO_BOHR])], 0, 2)
    basis = BasisSet(oh, "cc-pvdz")
    with pytest.raises(NotImplementedError, match="run_double_hybrid"):
        run_roks(oh, basis, functional="b2plyp")


# ---------------------------------------------------------------------
# DSD-PBEP86 + D3(BJ) — BUG 35 regression
# ---------------------------------------------------------------------

def test_dsd_pbep86_d3bj_h2o_def2svp_nonzero_dispersion():
    """DSD-PBEP86-D3(BJ) on H2O/def2-SVP must produce a nonzero
    dispersion correction and a total energy lower than the
    undispersed double-hybrid total.

    BUG 35: before the full 156-entry D3(BJ) parameter table landed,
    ``compute_d3bj(mol, 'dsd-pbep86')`` raised ValueError because the
    hand-curated table had no DSD-PBEP86 entry.  The alias
    ``dsd-pbep86`` → ``dsdpbep86_2011`` (added in the same fix)
    routes the hyphenated name to the GMTKN55-validated 2011 PCCP fit.

    Regression pin: the dispersion energy itself is a computed value,
    not a constant, so we do not hard-code its magnitude — we verify
    it is nonzero, attractive, and that the dispersed total is lower
    than the raw double-hybrid total.
    """
    atoms_bohr = [
        (8,  (0.000000,  0.000000,  0.117310)),
        (1,  (0.000000,  1.431100, -0.937880)),
        (1,  (0.000000, -1.431100, -0.937880)),
    ]
    mol = Molecule(
        [Atom(Z, list(xyz)) for Z, xyz in atoms_bohr],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "def2-svp")

    # 1. Dispersion alone must be nonzero and attractive.
    e_disp = compute_d3bj(mol, "dsd-pbep86", backend="builtin").energy
    assert e_disp < 0.0, "D3(BJ) correction must be attractive"
    assert abs(e_disp) > 1e-12, "D3(BJ) correction must be nonzero"

    # 2. Full double-hybrid dispatch with dispersion="d3bj".
    dh = run_dsd_pbep86(mol, basis, dispersion="d3bj")
    assert isinstance(dh, DoubleHybridResult)
    assert dh.rks.converged
    assert dh.dispersion is not None
    assert dh.dispersion.energy < 0.0
    # The dispersed total must be lower than the undispersed total
    # (adding an attractive correction makes the energy more negative).
    dh_no_disp = run_dsd_pbep86(mol, basis, dispersion=None)
    assert dh.e_total < dh_no_disp.e_total, (
        f"dispersed {dh.e_total:.10f} >= undispersed {dh_no_disp.e_total:.10f}"
    )
    # The difference must match the dispersion energy.
    assert dh.e_total == pytest.approx(
        dh_no_disp.e_total + dh.dispersion.energy, abs=1e-12,
    )
