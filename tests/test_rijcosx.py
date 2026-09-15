"""RIJCOSX — chain-of-spheres seminumerical exchange (Neese 2009).

RIJCOSX pairs RI-J (density-fitted Coulomb) with COSX-K (seminumerical
chain-of-spheres exchange). It is enabled by ``density_fit = True`` +
``cosx = True`` on the SCF / gradient Options structs, and applies to
RHF / UHF / RKS / UKS (the K piece — pure DFT with α_HF = 0 makes the
``cosx`` flag a no-op).

The original commit (`2dacff6`) shipped without a test. These pins:

1. **RIJCOSX SCF energy ≈ direct SCF energy** to the combined RI-J +
   COSX fit-error band (sub-mHa on neutral organics). This is an
   internal-consistency check — the commit message separately
   validated absolute energies against ORCA 6.1.1 RIJCOSX to
   ~0.12-0.13 mHa on glycine/def2-TZVP.
2. **RIJCOSX analytic gradient ≈ direct gradient** to the frozen-grid
   COSX-gradient noise floor.
3. **The ``cosx`` flag is a no-op for pure DFT** (α_HF = 0): a
   cosx=True PBE run equals the plain RIJ run bit-for-bit.

Tolerances are data-driven (worst observed: UHF energy ~2.8e-4 Ha,
RHF gradient ~1.7e-4 Ha/bohr — both well inside the documented
sub-mHa COSX fit band; a real regression would be mHa-scale).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom, BasisSet, Molecule, GradientOptions,
    GridOptions, RHFOptions, RKSOptions, UHFOptions, UKSOptions,
    build_grid, compute_kinetic, compute_nuclear, compute_overlap,
    compute_gradient, compute_gradient_uhf, compute_gradient_uks,
    make_cosx_jk_builder, run_rhf, run_rks, run_uhf, run_uks,
    run_uks_scf_with_jk,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
_A = ANGSTROM_TO_BOHR

H2O = [(8, [0.0, 0.0, 0.0]),
       (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
       (1, [0.0, -0.793353 * _A, -0.613510 * _A])]
OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * _A])]

# Aux basis both for RI-J; same name vibe-qc + PySCF both resolve.
_AUX = "def2-universal-jkfit"

# Sub-mHa fit-error band: RIJCOSX vs direct SCF. Worst observed 2.8e-4
# (UHF/OH); 1e-3 leaves headroom while still catching an mHa-scale
# regression.
_E_TOL = 1e-3
# COSX analytic gradient vs the DIRECT (exact-K) gradient. This is a
# cross-method parity band, not a correctness gate: the truthful
# gradient of the COSX energy surface legitimately differs from the
# exact-K gradient by the seminumerical quadrature error of the surface
# itself. Since the 2026-07-29 K-gradient parity fix (mixed-density
# bilinear form + overlap-fit Q response), the analytic gradient matches
# finite differences of its own frozen-grid surface to ~1e-7; the
# remaining surface error on OH/def2-svp (legacy grid) was measured by
# fixed-density FD at ~3.0e-3 max (neglected grid-point-motion/weight
# derivatives), of which ~1.3e-3 survives against the direct gradient.
# The pre-fix code passed a 1e-3 band only through an error cancellation
# between the wrong formula and the surface distortion. Correctness is
# gated by tests/test_rijcosx_gradient_fd.py (FD oracle), not here.
_G_TOL = 2e-3


def _mol(atoms, mult=1):
    return Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms],
                    multiplicity=mult)


# --- SCF energy: RIJCOSX vs direct ---------------------------------------

def test_rijcosx_rhf_matches_direct():
    """RHF RIJCOSX energy is within the COSX fit band of direct RHF."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")
    o = RHFOptions(); o.conv_tol_energy = 1e-10
    e_direct = run_rhf(mol, basis, o).energy

    o = RHFOptions(); o.conv_tol_energy = 1e-10
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    r = run_rhf(mol, basis, o)
    assert r.converged
    assert abs(r.energy - e_direct) < _E_TOL, (
        f"RHF RIJCOSX {r.energy:.8f} vs direct {e_direct:.8f}, "
        f"Δ={r.energy - e_direct:.3e} Ha exceeds the COSX fit band"
    )


def test_rijcosx_uhf_matches_direct():
    """UHF RIJCOSX (per-spin COSX-K) vs direct UHF."""
    mol = _mol(OH, mult=2)
    basis = BasisSet(mol, "def2-svp")
    o = UHFOptions(); o.conv_tol_energy = 1e-10; o.max_iter = 200
    e_direct = run_uhf(mol, basis, o).energy

    o = UHFOptions(); o.conv_tol_energy = 1e-10; o.max_iter = 200
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    r = run_uhf(mol, basis, o)
    assert r.converged
    assert abs(r.energy - e_direct) < _E_TOL, (
        f"UHF RIJCOSX {r.energy:.8f} vs direct {e_direct:.8f}, "
        f"Δ={r.energy - e_direct:.3e} Ha"
    )


def test_rijcosx_rks_b3lyp_matches_direct():
    """Hybrid RKS (B3LYP, α_HF = 0.2) RIJCOSX vs direct."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")
    o = RKSOptions(); o.functional = "B3LYP"; o.conv_tol_energy = 1e-9
    e_direct = run_rks(mol, basis, o).energy

    o = RKSOptions(); o.functional = "B3LYP"; o.conv_tol_energy = 1e-9
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    r = run_rks(mol, basis, o)
    assert r.converged
    assert abs(r.energy - e_direct) < _E_TOL, (
        f"RKS-B3LYP RIJCOSX {r.energy:.8f} vs direct {e_direct:.8f}, "
        f"Δ={r.energy - e_direct:.3e} Ha"
    )


def test_rijcosx_uks_b3lyp_matches_direct():
    """Hybrid UKS (B3LYP) RIJCOSX vs direct."""
    mol = _mol(OH, mult=2)
    basis = BasisSet(mol, "def2-svp")
    o = UKSOptions(); o.functional = "B3LYP"; o.conv_tol_energy = 1e-9
    o.max_iter = 300; o.damping = 0.6
    e_direct = run_uks(mol, basis, o).energy

    o = UKSOptions(); o.functional = "B3LYP"; o.conv_tol_energy = 1e-9
    o.max_iter = 300; o.damping = 0.6
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    r = run_uks(mol, basis, o)
    assert r.converged
    assert not r.stability_checked
    assert abs(r.energy - e_direct) < _E_TOL, (
        f"UKS-B3LYP RIJCOSX {r.energy:.8f} vs direct {e_direct:.8f}, "
        f"Δ={r.energy - e_direct:.3e} Ha"
    )


def test_rijcosx_uks_explicit_stability_fails_closed():
    """The post-SCF COSX correction lacks a matching Hessian response."""
    mol = _mol([(1, [0.0, 0.0, 0.0])], mult=2)
    basis = BasisSet(mol, "def2-svp")
    opts = UKSOptions()
    opts.functional = "B3LYP"
    opts.density_fit = True
    opts.aux_basis = _AUX
    opts.cosx = True
    opts.stability_check = True

    with pytest.raises(RuntimeError, match="COSX one-centre exchange"):
        run_uks(mol, basis, opts)


def test_low_level_uks_cosx_builder_stability_fails_closed():
    """The live JK capability gate cannot be bypassed by stale options."""
    mol = _mol([(1, [0.0, 0.0, 0.0])], mult=2)
    basis = BasisSet(mol, "def2-svp")
    aux = BasisSet(mol, _AUX)
    opts = UKSOptions()
    opts.functional = "B3LYP"
    opts.stability_check = True
    assert not opts.cosx

    overlap = np.asarray(compute_overlap(basis))
    hcore = (
        np.asarray(compute_kinetic(basis))
        + np.asarray(compute_nuclear(basis, mol))
    )
    xc_grid = build_grid(mol, GridOptions())
    cosx_grid = build_grid(mol, opts.cosx_grid)
    jk_builder = make_cosx_jk_builder(basis, aux, cosx_grid)

    with pytest.raises(RuntimeError, match="COSX one-centre exchange"):
        run_uks_scf_with_jk(
            basis,
            1,
            0,
            overlap,
            hcore,
            0.0,
            jk_builder,
            xc_grid,
            opts,
        )


def test_rijcosx_singlet_uks_matches_rks():
    """A spin-degenerate UKS singlet retains the RKS energy and density.

    This is the numerical guard for the single-exchange-build path: only
    alpha/beta densities equal within the roundoff envelope may share K.
    """
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")

    rks_opts = RKSOptions()
    rks_opts.functional = "B3LYP"
    rks_opts.conv_tol_energy = 1e-9
    rks_opts.density_fit = True
    rks_opts.aux_basis = _AUX
    rks_opts.cosx = True
    rks = run_rks(mol, basis, rks_opts)

    uks_opts = UKSOptions()
    uks_opts.functional = "B3LYP"
    uks_opts.conv_tol_energy = 1e-9
    uks_opts.density_fit = True
    uks_opts.aux_basis = _AUX
    uks_opts.cosx = True
    uks = run_uks(mol, basis, uks_opts)

    assert rks.converged and uks.converged
    assert abs(uks.energy - rks.energy) < 1e-9
    np.testing.assert_allclose(
        uks.density_alpha,
        uks.density_beta,
        rtol=0.0,
        atol=1e-12,
    )


# --- analytic gradient: RIJCOSX vs direct --------------------------------

def test_rijcosx_rhf_gradient_matches_direct():
    """RHF RIJCOSX analytic gradient (frozen-grid COSX-K gradient +
    DF-J gradient) vs the direct 4-index gradient, on the same
    converged reference."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")
    o = RHFOptions(); o.conv_tol_energy = 1e-11; o.conv_tol_grad = 1e-9
    rhf = run_rhf(mol, basis, o)
    assert rhf.converged

    g_direct = np.array(compute_gradient(mol, basis, rhf))
    go = GradientOptions()
    go.density_fit = True; go.aux_basis = _AUX; go.cosx = True
    g_cosx = np.array(compute_gradient(mol, basis, rhf, go))

    delta = np.abs(g_cosx - g_direct).max()
    assert delta < _G_TOL, (
        f"RHF RIJCOSX gradient max abs diff vs direct = {delta:.3e} "
        f"Ha/bohr (frozen-grid COSX noise floor is ~1e-4)"
    )


def test_rijcosx_uhf_gradient_matches_direct_and_routes_through_cosx():
    """UHF RIJCOSX analytic gradient on OH radical: must agree with the
    direct gradient within the COSX fit band, AND must differ from
    the plain DF gradient (proving K actually routes through the
    chain-of-spheres kernel).

    Pre-audit ``uhf_df_two_electron_gradient`` ignored ``options.cosx``
    so DF-only and DF+COSX gradients were bit-identical to ~1e-15
    (2026-05-18 audit P2).
    """
    mol = _mol(OH, mult=2)
    basis = BasisSet(mol, "def2-svp")
    o = UHFOptions()
    o.conv_tol_energy = 1e-11; o.conv_tol_grad = 1e-9; o.max_iter = 200
    uhf = run_uhf(mol, basis, o)
    assert uhf.converged

    g_direct = np.array(compute_gradient_uhf(mol, basis, uhf))

    go_df = GradientOptions()
    go_df.density_fit = True; go_df.aux_basis = _AUX
    g_df = np.array(compute_gradient_uhf(mol, basis, uhf, go_df))

    go_cosx = GradientOptions()
    go_cosx.density_fit = True; go_cosx.aux_basis = _AUX; go_cosx.cosx = True
    g_cosx = np.array(compute_gradient_uhf(mol, basis, uhf, go_cosx))

    delta_direct = np.abs(g_cosx - g_direct).max()
    assert delta_direct < _G_TOL, (
        f"UHF RIJCOSX gradient max abs diff vs direct = {delta_direct:.3e} "
        f"Ha/bohr (frozen-grid COSX noise floor is ~1e-4)"
    )
    delta_df = np.abs(g_cosx - g_df).max()
    assert delta_df > 1e-6, (
        f"UHF RIJCOSX gradient is bit-identical to plain DF gradient "
        f"(max abs diff = {delta_df:.3e} Ha/bohr): options.cosx is being "
        f"silently dropped in the unrestricted DF gradient path"
    )


def test_rijcosx_uks_b3lyp_gradient_matches_direct_and_routes_through_cosx():
    """Hybrid UKS (B3LYP, α_HF = 0.2) RIJCOSX gradient on OH radical:
    must agree with the direct gradient within the COSX fit band AND
    differ from plain DF (audit regression, same root cause as the
    UHF case)."""
    mol = _mol(OH, mult=2)
    basis = BasisSet(mol, "def2-svp")
    o = UKSOptions()
    o.functional = "B3LYP"
    o.conv_tol_energy = 1e-9
    o.max_iter = 300; o.damping = 0.6
    uks = run_uks(mol, basis, o)
    assert uks.converged

    grid = o.grid
    g_direct = np.array(compute_gradient_uks(mol, basis, uks, grid))

    go_df = GradientOptions()
    go_df.density_fit = True; go_df.aux_basis = _AUX
    g_df = np.array(compute_gradient_uks(mol, basis, uks, grid, go_df))

    go_cosx = GradientOptions()
    go_cosx.density_fit = True; go_cosx.aux_basis = _AUX; go_cosx.cosx = True
    g_cosx = np.array(compute_gradient_uks(mol, basis, uks, grid, go_cosx))

    delta_direct = np.abs(g_cosx - g_direct).max()
    assert delta_direct < _G_TOL, (
        f"UKS-B3LYP RIJCOSX gradient max abs diff vs direct = "
        f"{delta_direct:.3e} Ha/bohr"
    )
    delta_df = np.abs(g_cosx - g_df).max()
    assert delta_df > 1e-6, (
        f"UKS hybrid RIJCOSX gradient is bit-identical to plain DF "
        f"(max abs diff = {delta_df:.3e}): options.cosx silently dropped"
    )


# --- pure DFT: the cosx flag is a no-op ----------------------------------

def test_rijcosx_pure_dft_flag_is_noop():
    """For a pure functional (α_HF = 0) no K matrix is built, so
    ``cosx = True`` must produce exactly the plain RIJ result —
    bit-for-bit, not just within the fit band."""
    mol = _mol(H2O)
    basis = BasisSet(mol, "def2-svp")

    o = RKSOptions(); o.functional = "PBE"; o.conv_tol_energy = 1e-10
    o.density_fit = True; o.aux_basis = _AUX
    e_rij = run_rks(mol, basis, o).energy

    o = RKSOptions(); o.functional = "PBE"; o.conv_tol_energy = 1e-10
    o.density_fit = True; o.aux_basis = _AUX; o.cosx = True
    e_rijcosx = run_rks(mol, basis, o).energy

    assert abs(e_rijcosx - e_rij) < 1e-10, (
        f"cosx flag changed a pure-DFT (α_HF=0) energy: RIJ "
        f"{e_rij:.10f} vs RIJ+cosx {e_rijcosx:.10f} — the flag should "
        f"be a no-op when no K is built"
    )


def test_pure_dft_uks_cosx_noop_retains_stability_verdict_and_timing():
    """A zero-exchange COSX request must not enter the staged wrapper.

    Its final one-iteration recompute cannot satisfy the ordinary SCF gates
    and used to overwrite the valid stability verdict and timing produced by
    the converged stage, even though COSX contributes nothing at alpha_HF=0.
    """
    mol = _mol([(1, [0.0, 0.0, 0.0])], mult=2)
    basis = BasisSet(mol, "def2-svp")

    rij_opts = UKSOptions()
    rij_opts.functional = "PBE"
    rij_opts.density_fit = True
    rij_opts.aux_basis = _AUX
    rij_opts.max_iter = 80
    rij = run_uks(mol, basis, rij_opts)

    cosx_opts = UKSOptions(rij_opts)
    cosx_opts.cosx = True
    rijcosx = run_uks(mol, basis, cosx_opts)

    assert rij.converged and rijcosx.converged
    assert rij.stability_checked and rijcosx.stability_checked
    assert rijcosx.n_iter == rij.n_iter
    assert rijcosx.energy == pytest.approx(rij.energy, abs=1e-12)
    np.testing.assert_allclose(
        rijcosx.density_alpha,
        rij.density_alpha,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rijcosx.density_beta,
        rij.density_beta,
        rtol=0.0,
        atol=1e-12,
    )
    assert rijcosx.stability_analysis_converged
    assert rijcosx.stability_wall_s > 0.0
    assert rijcosx.stability_cpu_s > 0.0
    assert rijcosx.n_iter_before_stability == rijcosx.n_iter
    assert rijcosx.stability_eigenvalue == pytest.approx(
        rij.stability_eigenvalue,
        abs=1e-10,
    )


def test_direct_hybrid_uks_inactive_cosx_retains_stability_metadata():
    """COSX staging requires the density-fit/auxiliary-basis route."""
    mol = _mol([(1, [0.0, 0.0, 0.0])], mult=2)
    basis = BasisSet(mol, "def2-svp")

    direct_opts = UKSOptions()
    direct_opts.functional = "B3LYP"
    direct_opts.max_iter = 80
    direct = run_uks(mol, basis, direct_opts)

    inactive_cosx_opts = UKSOptions(direct_opts)
    inactive_cosx_opts.cosx = True
    inactive_cosx = run_uks(mol, basis, inactive_cosx_opts)

    assert direct.converged and inactive_cosx.converged
    assert direct.stability_checked and inactive_cosx.stability_checked
    assert inactive_cosx.n_iter == direct.n_iter
    assert inactive_cosx.energy == pytest.approx(direct.energy, abs=1e-12)
    np.testing.assert_allclose(
        inactive_cosx.density_alpha,
        direct.density_alpha,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        inactive_cosx.density_beta,
        direct.density_beta,
        rtol=0.0,
        atol=1e-12,
    )
    assert inactive_cosx.stability_analysis_converged
    assert inactive_cosx.stability_wall_s > 0.0
    assert inactive_cosx.stability_cpu_s > 0.0
    assert inactive_cosx.n_iter_before_stability == inactive_cosx.n_iter
    assert inactive_cosx.stability_eigenvalue == pytest.approx(
        direct.stability_eigenvalue,
        abs=1e-10,
    )
