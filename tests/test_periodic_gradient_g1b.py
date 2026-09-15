"""Phase G1b — analytic Γ-only periodic RKS atomic gradient (pure DFT).

Pinned contracts:

1. **Public API** — ``vq.compute_gradient_periodic_rks_gamma`` is
   exposed at the top level. Returns ``(n_atoms, 3)`` Ha/bohr.

2. **Molecular limit, LDA**. H₂O in a 20-Å cubic box, Γ-only mesh,
   cutoff < box: matches ``vq.compute_gradient_rks`` (molecular)
   to ≤ 1e-7 Ha/bohr.

3. **True-periodic LDA**. 1D H chain (a = 2 Å, STO-3G): Newton's-
   3rd-law obeyed to machine precision; transverse forces vanish
   by symmetry. (Vs FD-on-energy at fixed reference: this is the
   v0.6 G1b headline path — pure-DFT periodic gradient is exact
   because α_HF = 0 means no K-piece exposure to the G1a-2 K bug.)

4. **Refusal on un-converged SCF** — non-converged input raises.

5. **Hybrid DFT contract** — B3LYP is exact in the molecular limit,
   while true-periodic DIRECT_TRUNCATED hybrids fail closed instead
   of returning the known cutoff-dependent HF-exchange Pulay force.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2o_periodic_box(box_ang: float = 20.0):
    big_box = box_ang * ANGSTROM_TO_BOHR
    atoms = [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
    ]
    sys = vq.PeriodicSystem(3, np.diag([big_box, big_box, big_box]), atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    return sys, basis, opts, kmesh, atoms


# ---------------------------------------------------------------------------
# 1. Public API
# ---------------------------------------------------------------------------

def test_compute_gradient_periodic_rks_gamma_exposed():
    assert hasattr(vq, "compute_gradient_periodic_rks_gamma")


# ---------------------------------------------------------------------------
# 2. Molecular limit — LDA matches molecular analytic
# ---------------------------------------------------------------------------

def test_h2o_lda_molecular_limit_matches_molecular():
    sys, basis, opts, kmesh, atoms = _h2o_periodic_box()
    result = vq.run_rks_periodic(sys, basis, kmesh, opts)
    g_p = vq.compute_gradient_periodic_rks_gamma(
        sys, basis, result, lattice_opts=opts.lattice_opts)

    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis_m = vq.BasisSet(mol, "sto-3g")
    rks_opts = vq.RKSOptions()
    rks_opts.functional = "lda"
    rks_opts.conv_tol_energy = 1e-12
    rks_m = vq.run_rks(mol, basis_m, rks_opts)
    g_m = np.asarray(vq.compute_gradient_rks(mol, basis_m, rks_m))

    np.testing.assert_allclose(g_p, g_m, atol=1e-6,
        err_msg="periodic RKS LDA molecular-limit gradient diverges from "
                "molecular analytic gradient")


# ---------------------------------------------------------------------------
# 2b. Molecular limit — pure GGA and hybrids match molecular analytic.
#     The lattice XC Pulay primitive supplies the GGA σ-piece the former
#     molecular fallback skipped; the variational-V_xc overlap-Lagrangian
#     removes the α_HF-scaled exchange-G=0 self-image gauge for hybrids.
# ---------------------------------------------------------------------------

def _periodic_vs_molecular_rks(atoms, functional, box_ang=20.0):
    big = box_ang * ANGSTROM_TO_BOHR
    sys = vq.PeriodicSystem(3, np.diag([big, big, big]), atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 25.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    result = vq.run_rks_periodic(sys, basis, kmesh, opts)
    g_p = np.asarray(vq.compute_gradient_periodic_rks_gamma(
        sys, basis, result, lattice_opts=opts.lattice_opts))

    mol = vq.Molecule(atoms, charge=0, multiplicity=1)
    basis_m = vq.BasisSet(mol, "sto-3g")
    rks_opts = vq.RKSOptions()
    rks_opts.functional = functional
    rks_opts.conv_tol_energy = 1e-12
    rks_m = vq.run_rks(mol, basis_m, rks_opts)
    g_m = np.asarray(vq.compute_gradient_rks(mol, basis_m, rks_m))
    return g_p, g_m


def test_h2o_pbe_molecular_limit_matches_molecular():
    """Pure GGA: the lattice XC Pulay primitive includes the GGA σ-piece the
    former molecular LDA-only fallback dropped, so PBE now matches molecular
    analytic (was ~1e-2 off)."""
    g_p, g_m = _periodic_vs_molecular_rks(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
         vq.Atom(1, [0.0, -0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR])],
        "pbe")
    np.testing.assert_allclose(g_p, g_m, atol=1e-6,
        err_msg="periodic RKS PBE molecular-limit gradient diverges from "
                "molecular analytic (GGA σ XC Pulay)")


def test_h2o_blyp_molecular_limit_matches_molecular():
    """Pure GGA (B88 exchange + LYP correlation): regression pin for the
    σ-Pulay ``v_σ`` magnitude clamp. B88/LYP carry a legitimately large but
    finite ``v_σ`` in the low-density tail; an earlier ±10 Ha⁻¹ clamp on the
    periodic σ-Pulay desynchronised the analytic gradient from its own energy
    (~1.7e-4 Ha/bohr off molecular analytic), while PBE-family functionals —
    whose ``v_σ`` never reaches the clamp — stayed exact. The bare 2·w·v_σ·∇ρ
    form (only NaN/inf-guarded) restores the molecular-analytic match."""
    g_p, g_m = _periodic_vs_molecular_rks(
        [vq.Atom(8, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR]),
         vq.Atom(1, [0.0, -0.85 * ANGSTROM_TO_BOHR, 0.6 * ANGSTROM_TO_BOHR])],
        "blyp")
    np.testing.assert_allclose(g_p, g_m, atol=1e-6,
        err_msg="periodic RKS BLYP molecular-limit gradient diverges from "
                "molecular analytic (B88/LYP σ-Pulay v_σ clamp regression)")


def test_h2_b3lyp_molecular_limit_matches_molecular():
    """Hybrid: H₂ carries a nonzero α_HF-scaled exchange-G=0 self-image gauge
    (+0.0195 Ha on the occupied orbital energy at B3LYP). The variational
    V_xc threaded into the overlap-Lagrangian removes it; with the lattice
    XC Pulay the gradient matches molecular analytic."""
    g_p, g_m = _periodic_vs_molecular_rks(
        [vq.Atom(1, [0.0, 0.0, 0.0]),
         vq.Atom(1, [0.0, 0.0, 1.0 * ANGSTROM_TO_BOHR])],
        "b3lyp")
    np.testing.assert_allclose(g_p, g_m, atol=1e-6,
        err_msg="periodic RKS B3LYP molecular-limit gradient diverges from "
                "molecular analytic (exchange-G=0 gauge / hybrid)")


# ---------------------------------------------------------------------------
# 3. True-periodic LDA — Newton's 3rd law on 1D H chain
# ---------------------------------------------------------------------------

def test_h_chain_1d_lda_newtons_third_law():
    a_per = 2.0 * ANGSTROM_TO_BOHR
    vac = 20.0 * ANGSTROM_TO_BOHR
    R_HH = 0.74 * ANGSTROM_TO_BOHR
    lat = np.diag([a_per, vac, vac])
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R_HH, 0.0, 0.0]),
    ]
    sys = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    result = vq.run_rks_periodic(sys, basis, kmesh, opts)
    g = vq.compute_gradient_periodic_rks_gamma(
        sys, basis, result, lattice_opts=opts.lattice_opts)

    # Newton's-3rd-law along the periodic axis.
    np.testing.assert_allclose(g[0, 0], -g[1, 0], atol=1e-9)
    # Transverse forces vanish.
    np.testing.assert_allclose(g[:, 1:], 0.0, atol=1e-9)


def test_h_chain_1d_b3lyp_true_periodic_fails_closed():
    a_per = 2.0 * ANGSTROM_TO_BOHR
    vac = 20.0 * ANGSTROM_TO_BOHR
    R_HH = 0.74 * ANGSTROM_TO_BOHR
    lat = np.diag([a_per, vac, vac])
    atoms = [
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [R_HH, 0.0, 0.0]),
    ]
    sys = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions()
    opts.functional = "b3lyp"
    opts.conv_tol_energy = 1e-12
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    kmesh = vq.monkhorst_pack(sys, [1, 1, 1])
    result = vq.run_rks_periodic(sys, basis, kmesh, opts)

    with pytest.raises(ValueError, match="DIRECT_TRUNCATED HF/hybrid"):
        vq.compute_gradient_periodic_rks_gamma(
            sys, basis, result, lattice_opts=opts.lattice_opts)


# ---------------------------------------------------------------------------
# 4. Un-converged input rejected
# ---------------------------------------------------------------------------

def test_unconverged_rejected():
    sys, basis, opts, kmesh, _ = _h2o_periodic_box()
    opts.max_iter = 1
    opts.conv_tol_energy = 1e-30
    result = vq.run_rks_periodic(sys, basis, kmesh, opts)
    assert result.converged is False
    with pytest.raises(ValueError, match="not converged"):
        vq.compute_gradient_periodic_rks_gamma(
            sys, basis, result, lattice_opts=opts.lattice_opts)
