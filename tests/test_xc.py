"""libxc functional wrapper."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc import Functional, XCKind


def test_lda_alias_resolves():
    f = Functional("LDA")
    assert f.kind == XCKind.LDA
    assert not f.is_hybrid


def test_svwn_equivalent_to_lda():
    f = Functional("SVWN")
    assert f.kind == XCKind.LDA
    assert not f.is_hybrid


def test_pbe_is_pure_gga():
    f = Functional("PBE")
    assert f.kind == XCKind.GGA
    assert not f.is_hybrid
    assert f.hf_exchange_fraction == pytest.approx(0.0)


def test_blyp_is_pure_gga():
    f = Functional("BLYP")
    assert f.kind == XCKind.GGA
    assert not f.is_hybrid


def test_b3lyp_reports_20_percent_exact_exchange():
    f = Functional("B3LYP")
    assert f.kind == XCKind.GGA
    assert f.is_hybrid
    assert f.hf_exchange_fraction == pytest.approx(0.2, abs=1e-10)


def test_b3lyp_is_vwn5_variant_b3lyp_g_is_gaussian():
    """vibe-qc ships the ORCA definition: bare ``b3lyp`` is the VWN5
    variant (libxc XC_HYB_GGA_XC_B3LYP5), with ``b3lyp5`` as its
    explicit spelling; ``b3lyp/g`` (ORCA spelling) and ``b3lypg``
    (PySCF spelling) select the Gaussian-compatible variant
    (XC_HYB_GGA_XC_B3LYP, VWN-RPA — what Gaussian / PySCF / Psi4 mean
    by the bare name). See tests/test_b3lyp_convention.py for the
    cross-code energy pins.

    All four are the same B3LYP hybrid recipe (GGA kind, 20% exact
    exchange) — they differ *only* in which Vosko-Wilk-Nusair
    local-correlation parametrisation fills the LSDA-correlation slot.
    """
    f5 = Functional("b3lyp")      # VWN5 — the default
    fg = Functional("b3lyp/g")    # VWN-RPA — Gaussian-compatible
    for f in (f5, fg, Functional("b3lyp5"), Functional("b3lypg")):
        assert f.kind == XCKind.GGA
        assert f.is_hybrid
        assert f.hf_exchange_fraction == pytest.approx(0.2, abs=1e-10)

    rho = np.array([0.1, 0.5, 1.0])
    sigma = np.array([0.01, 0.05, 0.2])
    exc5 = np.asarray(f5.eval_unpolarised(rho, sigma)[0])
    excg = np.asarray(fg.eval_unpolarised(rho, sigma)[0])

    # Explicit spellings are the same functional bit-for-bit…
    np.testing.assert_allclose(
        np.asarray(Functional("b3lyp5").eval_unpolarised(rho, sigma)[0]),
        exc5, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(Functional("b3lypg").eval_unpolarised(rho, sigma)[0]),
        excg, rtol=0, atol=1e-14)

    # …while the VWN slot makes the two flavors genuinely different.
    assert not np.allclose(exc5, excg), (
        "b3lyp (VWN5) and b3lyp/g (VWN-RPA) returned identical XC "
        "energy densities — the two VWN variants should differ"
    )


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown|unrecognized"):
        Functional("not-a-real-functional")


# ---------------------------------------------------------------------------
# Meta-GGA (τ-dependent) functionals — TPSS / TPSSh, M06-L / M06-2X, and
# the SCAN / r²SCAN family.
#
# Each is a libxc XC_FAMILY_MGGA or XC_FAMILY_HYB_MGGA functional. vibe-qc
# computes τ on the DFT grid from the AO gradients, so all of them resolve
# cleanly with the documented HF-exchange fraction. (SCAN / r²SCAN are
# τ-only — laplacian-free by design — so they fit vibe-qc's τ-only MGGA
# grid.)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,want_hf", [
    ("tpss",    0.00),
    ("tpssh",   0.10),
    ("m06-l",   0.00),
    ("m06-2x",  0.54),
    ("m062x",   0.54),  # alias without the dash
    ("m06l",    0.00),  # alias without the dash
    ("scan",     0.00),
    ("r2scan",   0.00),
    ("r2scan01", 0.00),
    ("r2scan0",  0.25),
    ("r2scanh",  0.10),
])
def test_meta_gga_aliases_resolve(name, want_hf):
    f = Functional(name)
    assert f.kind == XCKind.MGGA
    assert f.hf_exchange_fraction == pytest.approx(want_hf, abs=1e-10)
    assert f.is_hybrid == (want_hf > 0.0)


def test_mgga_via_numeric_ids_also_resolves():
    """The constructor accepts a comma-separated list of libxc XC_… IDs
    as well as named aliases. TPSS by numeric IDs (202 = MGGA_X_TPSS,
    231 = MGGA_C_TPSS) should reach the same MGGA evaluator path."""
    f = Functional("202,231")
    assert f.kind == XCKind.MGGA
    assert not f.is_hybrid


def test_mgga_eval_unpolarised_returns_v_tau():
    """``eval_unpolarised_mgga`` returns (exc, v_rho, v_sigma, v_tau);
    none should be NaN at a smooth test density, and v_tau should not
    be identically zero (otherwise the τ-contribution is being dropped)."""
    f = Functional("tpssh")
    rho   = np.array([0.05, 0.20, 0.50, 1.0])
    sigma = np.array([0.005, 0.02, 0.08, 0.30])
    tau   = np.array([0.01, 0.04, 0.10, 0.50])
    exc, v_rho, v_sigma, v_tau = f.eval_unpolarised_mgga(rho, sigma, tau)
    for arr in (exc, v_rho, v_sigma, v_tau):
        assert len(arr) == len(rho)
        assert not np.any(np.isnan(np.asarray(arr)))
    assert np.any(np.asarray(v_tau) != 0.0)
    assert np.all(np.asarray(exc) < 0.0)


def test_mgga_eval_polarised_returns_per_spin_v_tau():
    """``eval_polarised_mgga`` returns (exc, vρα, vρβ, vσαα, vσαβ, vσββ,
    vτα, vτβ) — eight arrays. Symmetric inputs (ρ_α = ρ_β, τ_α = τ_β,
    σ_αα = σ_ββ, σ_αβ = sqrt of those) should yield exc matching the
    unpolarised path on ρ_total = 2 ρ_α and τ_total = τ_α + τ_β."""
    f_u = Functional("m06-2x", spin=1)
    f_p = Functional("m06-2x", spin=2)
    rho_a = np.array([0.10, 0.30, 0.60])
    rho_b = rho_a.copy()
    tau_a = np.array([0.02, 0.06, 0.15])
    tau_b = tau_a.copy()
    # |∇ρ_α|² = σ_αα; for a closed-shell density ρ = ρ_α + ρ_β = 2 ρ_α,
    # ∇ρ = ∇ρ_α + ∇ρ_β = 2 ∇ρ_α so σ_total = 4 σ_αα. We can verify
    # closed-shell limit at the σ_αβ = σ_αα = σ_ββ choice.
    sigma_aa = np.array([0.005, 0.01, 0.04])
    sigma_ab = sigma_aa.copy()
    sigma_bb = sigma_aa.copy()
    out_p = f_p.eval_polarised_mgga(rho_a, rho_b,
                                    sigma_aa, sigma_ab, sigma_bb,
                                    tau_a, tau_b)
    assert len(out_p) == 8
    exc_p, vra, vrb, vaa, vab, vbb, vta, vtb = (np.asarray(x) for x in out_p)
    for arr in (exc_p, vra, vrb, vaa, vab, vbb, vta, vtb):
        assert arr.shape == rho_a.shape
        assert not np.any(np.isnan(arr))
    # Closed-shell symmetry: α / β potentials must agree.
    np.testing.assert_allclose(vra, vrb)
    np.testing.assert_allclose(vaa, vbb)
    np.testing.assert_allclose(vta, vtb)

    rho_tot   = rho_a + rho_b
    sigma_tot = sigma_aa + 2.0 * sigma_ab + sigma_bb
    tau_tot   = tau_a + tau_b
    exc_u = np.asarray(f_u.eval_unpolarised_mgga(rho_tot, sigma_tot, tau_tot)[0])
    np.testing.assert_allclose(exc_u, exc_p, rtol=1e-10, atol=1e-12)


def test_mgga_with_eval_unpolarised_raises():
    """Calling the LDA/GGA ``eval_unpolarised`` on an MGGA functional must
    raise with a roadmap pointer to the MGGA method (the alternative —
    silently returning τ-free potentials — would corrupt the SCF)."""
    f = Functional("tpssh")
    rho = np.array([0.1, 0.5])
    sigma = np.array([0.01, 0.05])
    with pytest.raises(RuntimeError, match="meta-GGA|eval_unpolarised_mgga"):
        f.eval_unpolarised(rho, sigma)


def test_mgga_fxc_not_yet_plumbed():
    """The analytic Hessian / CPKS second-derivative kernel for MGGAs is
    deferred. ``eval_unpolarised_fxc`` must raise a clear message
    (currently SCF is fine; the linear-response path is restricted to
    LDA / GGA / hybrid-GGA)."""
    f = Functional("tpssh")
    rho = np.array([0.1, 0.5])
    sigma = np.array([0.01, 0.05])
    with pytest.raises(RuntimeError, match="meta-GGA|Hessian"):
        f.eval_unpolarised_fxc(rho, sigma)


def test_needs_laplacian_mgga_rejected():
    """Meta-GGAs that require ∇²ρ (the density laplacian) are not yet
    supported — vibe-qc populates τ on the grid but not the laplacian.
    XC_MGGA_X_TB09 (id 208) is a τ + laplacian functional and must be
    refused at construction with a clear error pointing to τ-only MGGAs."""
    with pytest.raises(RuntimeError, match="laplacian|τ-only meta-GGA"):
        Functional("208")  # MGGA_X_TB09 — needs ∇²ρ


def test_invalid_libxc_id_does_not_crash():
    """xc_func_init failure for an unknown id must raise cleanly, not crash
    the destructor on the still-uninitialized xc_func_type slot."""
    with pytest.raises(RuntimeError, match="xc_func_init failed"):
        Functional("99999999")


@pytest.mark.parametrize("func_name,want_energy,tol", [
    # H2O / def2-SVP reference energies cross-checked against PySCF.dft
    # RKS (grids.level=5) at the time of landing. Tolerance matches the
    # cross-code grid-quadrature noise — tight for the smooth functionals,
    # looser for SCAN whose sharp enhancement factor is grid-sensitive on
    # the default medium grid (r²SCAN was re-regularized precisely to fix
    # that — note its much tighter tolerance below).
    ("tpss",    -76.3532, 5e-4),
    ("tpssh",   -76.3470, 5e-4),
    ("m06-l",   -76.3446, 5e-4),
    ("m06-2x",  -76.3181, 5e-4),
    ("scan",    -76.3210, 2e-3),  # grid-sensitive on the default grid
    ("r2scan",   -76.3118, 5e-4),
    ("r2scan01", -76.3151, 5e-4),  # PySCF MGGA_X/C_R2SCAN01: 3.9 µHa
    ("r2scan0",  -76.3034, 5e-4),
    ("r2scanh",  -76.3084, 5e-4),
])
def test_mgga_rks_converges_to_reference(func_name, want_energy, tol):
    """End-to-end: passing a meta-GGA through ``run_rks`` converges to
    the expected energy on H2O / def2-SVP, within the cross-code
    grid-noise tolerance documented in the parity report. This is the
    closed-shell smoke test that the τ + V_τ kernel is wired correctly;
    the cross-code tolerance test lives in ``test_parity_hf_dft.py``."""
    from vibeqc import Atom, BasisSet, Molecule, RKSOptions, run_rks
    mol = Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])
    basis = BasisSet(mol, "def2-svp")
    opts = RKSOptions()
    opts.functional = func_name
    opts.conv_tol_energy = 1e-8
    res = run_rks(mol, basis, opts)
    assert res.converged, f"{func_name} did not converge"
    assert res.energy == pytest.approx(want_energy, abs=tol)


# ---------------------------------------------------------------------------
# Range-separated (CAM / RSH) hybrids — ωB97X.
#
# The exact-exchange admixture is position-dependent:
#   EXX(r) = cam_alpha + cam_beta·erf(rsh_omega·r).
# vibe-qc reads (ω, α, β) from libxc's xc_hyb_cam_coef at construction
# and the SCF driver builds the second erf-attenuated exchange matrix.
# ---------------------------------------------------------------------------
def test_wb97x_is_range_separated():
    """ωB97X (libxc HYB_GGA_XC_WB97X) is a long-range-corrected hybrid
    GGA: 15.77 % HF at short range ramping to 100 % at long range, with
    ω = 0.3 bohr⁻¹ (Chai-Head-Gordon 2008)."""
    f = Functional("wb97x")
    assert f.kind == XCKind.GGA          # libxc HYB_GGA family
    assert f.is_hybrid
    assert f.is_range_separated
    assert f.rsh_omega == pytest.approx(0.3, abs=1e-9)
    # erf-form CAM coefficients.
    assert f.cam_alpha == pytest.approx(0.157706, abs=1e-5)
    # Long-range-corrected → 100 % HF as r → ∞.
    assert (f.cam_alpha + f.cam_beta) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", ["b3lyp", "pbe0", "pbe", "tpssh", "m06-2x"])
def test_global_functionals_not_range_separated(name):
    """Global hybrids / pure functionals report is_range_separated ==
    False, cam_beta == 0, and cam_alpha == hf_exchange_fraction — so the
    SCF's (cam_alpha, cam_beta) exchange assembly collapses to the
    plain −½·α·K global-hybrid path."""
    f = Functional(name)
    assert not f.is_range_separated
    assert f.rsh_omega == 0.0
    assert f.cam_beta == 0.0
    assert f.cam_alpha == pytest.approx(f.hf_exchange_fraction, abs=1e-12)


def test_wb97x_rks_converges_to_reference():
    """End-to-end RKS ωB97X on H2O / def2-SVP — exercises the
    erf-attenuated K build + the cam_alpha·K + cam_beta·K_erf Fock
    assembly. Reference cross-checked vs PySCF.dft RKS (≈ 1 µHa)."""
    from vibeqc import Atom, BasisSet, Molecule, RKSOptions, run_rks
    mol = Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])
    opts = RKSOptions()
    opts.functional = "wb97x"
    opts.conv_tol_energy = 1e-9
    res = run_rks(mol, BasisSet(mol, "def2-svp"), opts)
    assert res.converged
    assert res.energy == pytest.approx(-76.332666, abs=5e-4)


def test_hse06_is_screened_range_separated():
    """HSE06 (libxc XC_HYB_GGA_XC_HSE06) is a *screened* range-separated
    hybrid — 25 % HF at short range tapering to 0 % at long range
    (the opposite ramp to ωB97X). In vibe-qc's erf-form CAM
    representation that is cam_alpha = +0.25, cam_beta = −0.25, so it
    rides the same erf-attenuated-K machinery as ωB97X with a negative
    long-range coefficient — no separate erfc kernel needed."""
    f = Functional("hse06")
    assert f.kind == XCKind.GGA
    assert f.is_hybrid
    assert f.is_range_separated
    assert f.rsh_omega == pytest.approx(0.11, abs=1e-9)
    assert f.cam_alpha == pytest.approx(0.25, abs=1e-9)      # EXX(r→0)
    # Screened → 0 % HF at long range.
    assert (f.cam_alpha + f.cam_beta) == pytest.approx(0.0, abs=1e-9)
    assert f.cam_beta == pytest.approx(-0.25, abs=1e-9)


def test_hse06_rks_converges_to_reference():
    """End-to-end RKS HSE06 on H2O / def2-SVP — the screened-RSH Fock
    assembly (cam_alpha·K + cam_beta·K_erf with cam_beta < 0). Reference
    cross-checked vs PySCF.dft RKS (≈ 1 µHa)."""
    from vibeqc import Atom, BasisSet, Molecule, RKSOptions, run_rks
    mol = Molecule([
        Atom(8, [0.0,  0.0,  0.0]),
        Atom(1, [0.0,  1.43, -0.98]),
        Atom(1, [0.0, -1.43, -0.98]),
    ])
    opts = RKSOptions()
    opts.functional = "hse06"
    opts.conv_tol_energy = 1e-9
    res = run_rks(mol, BasisSet(mol, "def2-svp"), opts)
    assert res.converged
    assert res.energy == pytest.approx(-76.277503, abs=5e-4)


def test_wb97x_closed_shell_uks_equals_rks():
    """Closed-shell UKS ωB97X must collapse to RKS bit-exactly — the
    rigorous regression guard for the per-spin RSH exchange assembly
    (cam_alpha·K_σ + cam_beta·K_erf,σ), independent of any hard-radical
    convergence behaviour."""
    from vibeqc import (Atom, BasisSet, Molecule, RKSOptions, UKSOptions,
                        run_rks, run_uks)
    atoms = [Atom(8, [0.0, 0.0, 0.0]),
             Atom(1, [0.0, 1.43, -0.98]),
             Atom(1, [0.0, -1.43, -0.98])]
    mol_r = Molecule(atoms)
    mol_u = Molecule(atoms, charge=0, multiplicity=1)
    ro = RKSOptions(); ro.functional = "wb97x"; ro.conv_tol_energy = 1e-10
    uo = UKSOptions(); uo.functional = "wb97x"; uo.conv_tol_energy = 1e-10
    uo.max_iter = 200
    e_r = run_rks(mol_r, BasisSet(mol_r, "def2-svp"), ro).energy
    e_u = run_uks(mol_u, BasisSet(mol_u, "def2-svp"), uo).energy
    assert e_u == pytest.approx(e_r, abs=1e-9)


def test_wb97x_density_fit_rejected_cleanly():
    """RSH + density fitting is not yet supported (the RI path has no
    erf-attenuated 3-centre integrals). run_rks must reject it at the
    JKBuilder dispatch with a clear error, not produce a wrong number."""
    from vibeqc import Atom, BasisSet, Molecule, RKSOptions, run_rks
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    opts = RKSOptions()
    opts.functional = "wb97x"
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"
    with pytest.raises((RuntimeError, ValueError), match="range-separated"):
        run_rks(mol, BasisSet(mol, "def2-svp"), opts)


def test_slater_exchange_matches_analytic_formula():
    """LDA (Slater) exchange: ε_x = -(3/4) (3/π)^(1/3) ρ^(1/3).
    Check at a few densities."""
    f = Functional("SLATER")
    rho = np.array([1e-3, 1e-2, 1e-1, 1.0, 1e1])
    sigma = np.zeros_like(rho)
    exc, v_rho, v_sigma = f.eval_unpolarised(rho, sigma)
    exc = np.asarray(exc); v_rho = np.asarray(v_rho)

    Cx = 0.75 * (3.0 / math.pi) ** (1.0 / 3.0)
    eps_x_ana = -Cx * np.cbrt(rho)

    # `exc` holds ρ·ε_x by convention.
    np.testing.assert_allclose(exc / rho, eps_x_ana, atol=1e-10)
    # v_ρ = d(ρ·ε_x)/dρ = (4/3) ε_x = -(4/3) C_x ρ^(1/3).
    np.testing.assert_allclose(
        v_rho, -(4.0 / 3.0) * Cx * np.cbrt(rho), atol=1e-10,
    )


def test_gga_pbe_returns_vsigma():
    f = Functional("PBE")
    rho = np.array([0.1, 0.5, 1.0])
    # Nonzero gradient magnitude for each density.
    sigma = np.array([0.01, 0.1, 0.3])
    exc, v_rho, v_sigma = f.eval_unpolarised(rho, sigma)
    assert len(v_sigma) == len(rho)
    # PBE energy density is negative (bound exchange-correlation).
    assert np.all(np.asarray(exc) < 0)


def test_lda_returns_empty_vsigma():
    f = Functional("LDA")
    rho = np.array([0.1, 1.0])
    sigma = np.array([])
    exc, v_rho, v_sigma = f.eval_unpolarised(rho, sigma)
    assert len(v_sigma) == 0


# ---------------------------------------------------------------------------
# Phase 17e — polarised GGA XC kernel (spin-polarised second derivatives).
#
# The unambiguous correctness witness: ``eval_polarised_gga_fxc`` returns
# the second derivatives of the XC energy density w.r.t. the five libxc
# inputs (ρ_a, ρ_b, σ_aa, σ_ab, σ_bb). Central-differencing the *first*
# derivatives from ``eval_polarised`` reconstructs the same 5×5 Hessian.
# Any libxc index-layout slip (v2rhosigma is a full 2×3 block; v2sigma2
# is the symmetric upper triangle) shows up as an O(1) relative error.
# ---------------------------------------------------------------------------

# Variable order for the 5×5 fxc Hessian.
_FXC_VARS = ("rho_a", "rho_b", "sigma_aa", "sigma_ab", "sigma_bb")

# Map (row, col) of the symmetric Hessian to the fxc dict key. Rows index
# the first-derivative (v_rho_a … v_sigma_bb), cols index the input.
_FXC_SLOTS = {
    (0, 0): "v2rho2_aa",
    (0, 1): "v2rho2_ab",
    (1, 1): "v2rho2_bb",
    (0, 2): "v2rhosigma_a_aa",
    (0, 3): "v2rhosigma_a_ab",
    (0, 4): "v2rhosigma_a_bb",
    (1, 2): "v2rhosigma_b_aa",
    (1, 3): "v2rhosigma_b_ab",
    (1, 4): "v2rhosigma_b_bb",
    (2, 2): "v2sigma2_aa_aa",
    (2, 3): "v2sigma2_aa_ab",
    (2, 4): "v2sigma2_aa_bb",
    (3, 3): "v2sigma2_ab_ab",
    (3, 4): "v2sigma2_ab_bb",
    (4, 4): "v2sigma2_bb_bb",
}


def _gga_fxc_base_point():
    """A physically valid spin-polarised GGA evaluation point.

    ρ_a ≠ ρ_b so no accidental closed-shell symmetry hides an α/β
    index swap; σ_ab kept well inside the Cauchy-Schwarz bound
    (σ_ab² ≤ σ_aa σ_bb) so libxc stays off any branch cut under the
    finite-difference perturbations.
    """
    rho_a = np.array([0.22, 0.35, 0.48, 0.61, 0.74, 0.85])
    rho_b = np.array([0.17, 0.28, 0.39, 0.50, 0.58, 0.66])
    sigma_aa = 0.30 * rho_a
    sigma_bb = 0.25 * rho_b
    sigma_ab = 0.4 * np.sqrt(sigma_aa * sigma_bb)
    return [rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb]


def _polarised_first_derivs(func, x):
    """(v_ρa, v_ρb, v_σaa, v_σab, v_σbb) from eval_polarised at point x."""
    ra, rb, saa, sab, sbb = x
    out = func.eval_polarised(ra, rb, saa, sab, sbb)
    # eval_polarised → (exc, v_rho_a, v_rho_b, v_sigma_aa, v_sigma_ab,
    # v_sigma_bb); drop exc.
    return [np.asarray(v) for v in out[1:]]


def _fd_polarised_gga_hessian(func, base, rel_step=1e-6):
    """Central-difference the 5×5 fxc Hessian from eval_polarised."""
    n = len(_FXC_VARS)
    hess = [[None] * n for _ in range(n)]
    for j in range(n):
        h = rel_step * np.maximum(np.abs(base[j]), 1e-3)
        xp = [b.copy() for b in base]
        xm = [b.copy() for b in base]
        xp[j] = base[j] + h
        xm[j] = base[j] - h
        gp = _polarised_first_derivs(func, xp)
        gm = _polarised_first_derivs(func, xm)
        for i in range(n):
            hess[i][j] = (gp[i] - gm[i]) / (2.0 * h)
    return hess


@pytest.mark.parametrize("name", ["PBE", "BLYP", "B3LYP", "PBE0"])
def test_polarised_gga_fxc_matches_finite_difference(name):
    """All 15 polarised-GGA fxc pieces must reproduce the central-
    difference Hessian of eval_polarised's first derivatives.

    Covers pure GGA (PBE, BLYP) and hybrid GGA (B3LYP carries an LDA
    VWN5 correlation component → exercises the LDA-in-composite branch;
    PBE0 is a pure-GGA hybrid). The HF-exchange fraction does not enter
    libxc's fxc — it lives in the K build — so the hybrid kernels are
    pure-functional second derivatives just like the non-hybrids.
    """
    func = Functional(name, spin=2)
    assert func.kind == XCKind.GGA

    base = _gga_fxc_base_point()
    fxc = func.eval_polarised_gga_fxc(*base)
    hess = _fd_polarised_gga_hessian(func, base)

    for (i, j), key in _FXC_SLOTS.items():
        analytic = np.asarray(fxc[key])
        fd = hess[i][j]
        assert analytic.shape == base[0].shape
        scale = max(np.linalg.norm(fd), np.linalg.norm(analytic), 1e-8)
        rel_err = np.linalg.norm(analytic - fd) / scale
        assert rel_err < 1e-4, (
            f"{name}: fxc[{key}] (Hessian slot ∂v_{_FXC_VARS[i]}/"
            f"∂{_FXC_VARS[j]}) analytic vs FD rel err {rel_err:.3e}"
        )


def test_polarised_gga_fxc_hessian_is_symmetric():
    """The mixed second derivatives are slot-symmetric: the fxc value at
    Hessian slot (i, j) equals the one at (j, i). v2rho2_ab is the only
    explicitly cross-spin ρρ term; v2rhosigma_* fills the full 2×3 block
    whose transpose lives in the *same* keys consumed row-major here."""
    func = Functional("PBE", spin=2)
    base = _gga_fxc_base_point()
    hess = _fd_polarised_gga_hessian(func, base)
    n = len(_FXC_VARS)
    for i in range(n):
        for j in range(i + 1, n):
            scale = max(np.linalg.norm(hess[i][j]),
                        np.linalg.norm(hess[j][i]), 1e-8)
            assert np.linalg.norm(hess[i][j] - hess[j][i]) / scale < 1e-4


def test_polarised_gga_fxc_closed_shell_symmetry():
    """With ρ_a = ρ_b and σ_aa = σ_bb (and σ_ab consistent), the αα and
    ββ fxc components must coincide, and the a-rowed v2rhosigma block
    must equal the b-rowed block under the αβ swap."""
    func = Functional("PBE", spin=2)
    rho = np.array([0.20, 0.45, 0.70])
    sigma = 0.3 * rho
    sigma_ab = 0.5 * sigma
    fxc = func.eval_polarised_gga_fxc(rho, rho, sigma, sigma_ab, sigma)

    np.testing.assert_allclose(fxc["v2rho2_aa"], fxc["v2rho2_bb"],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fxc["v2rhosigma_a_aa"],
                               fxc["v2rhosigma_b_bb"],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fxc["v2rhosigma_a_bb"],
                               fxc["v2rhosigma_b_aa"],
                               rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(fxc["v2sigma2_aa_aa"],
                               fxc["v2sigma2_bb_bb"],
                               rtol=1e-10, atol=1e-12)


def test_polarised_gga_fxc_on_unpolarised_functional_raises():
    """eval_polarised_gga_fxc on a spin=1 functional must raise."""
    func = Functional("PBE", spin=1)
    rho = np.array([0.3, 0.6])
    sig = np.array([0.05, 0.1])
    with pytest.raises(RuntimeError, match="unpolarised"):
        func.eval_polarised_gga_fxc(rho, rho, sig, sig, sig)


def test_polarised_gga_fxc_on_mgga_raises():
    """Meta-GGA polarised fxc (τ-dependent) is a later phase — must raise
    with a clear pointer rather than silently returning a GGA kernel."""
    func = Functional("tpssh", spin=2)
    rho = np.array([0.3, 0.6])
    sig = np.array([0.05, 0.1])
    with pytest.raises(RuntimeError, match="meta-GGA"):
        func.eval_polarised_gga_fxc(rho, rho, sig, sig, sig)


def test_polarised_gga_fxc_lda_functional_zero_sigma_pieces():
    """A pure-LDA functional routed through eval_polarised_gga_fxc must
    return zero for every σ-coupled piece and a non-trivial v2rho2 block
    (so callers can take one uniform LDA/GGA code path)."""
    func = Functional("LDA", spin=2)
    rho_a = np.array([0.25, 0.55, 0.80])
    rho_b = np.array([0.18, 0.40, 0.62])
    zero = np.zeros_like(rho_a)
    fxc = func.eval_polarised_gga_fxc(rho_a, rho_b, zero, zero, zero)
    for key, arr in fxc.items():
        arr = np.asarray(arr)
        if key.startswith("v2rho2"):
            assert np.any(arr != 0.0), f"{key} unexpectedly all-zero"
        else:
            np.testing.assert_array_equal(arr, np.zeros_like(rho_a))

    # The v2rho2 block must equal the dedicated LDA fxc routine.
    v_aa, v_ab, v_bb = func.eval_polarised_lda_fxc(rho_a, rho_b)
    np.testing.assert_allclose(fxc["v2rho2_aa"], v_aa, rtol=1e-12)
    np.testing.assert_allclose(fxc["v2rho2_ab"], v_ab, rtol=1e-12)
    np.testing.assert_allclose(fxc["v2rho2_bb"], v_bb, rtol=1e-12)
