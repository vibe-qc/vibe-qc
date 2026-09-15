"""Phase 15c: Γ-point periodic RKS SCF driver using EWALD_3D.

DFT counterpart of test_periodic_rhf_ewald.py — same correctness
contracts, with the XC potential added to the Fock build:

  1. Convergence on H₂ / PBE / STO-3G.
  2. ω-invariance: spread < 0.5 % of |E| across ω ∈ [0.3, 1.5].
  3. Makov-Payne consistency vs run_rks_periodic (DIRECT_TRUNCATED).
  4. Result-shape sanity.
  5. Skew-cell FFT metric support and open-shell rejection.
  6. Hybrid-functional path exercised (B3LYP) — verifies that the
     ``α > 0`` branch (HF exchange via build_jk_gamma_molecular_limit)
     is actually wired in.

System: H₂ / STO-3G in a 30-bohr cubic box. Tight cores aren't
present so the 0.3-bohr FFT grid resolves the density cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2_in_box(box: float = 30.0):
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_options(
    functional: str = "PBE", iter_limit: int = 60, damping: float = 0.3
):
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = damping
    opts.max_iter = iter_limit
    opts.use_diis = True
    return opts


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_h2_pbe_converges():
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("PBE")
    r = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged, f"did not converge after {r.n_iter} iterations"
    assert r.n_iter <= opts.max_iter
    # H2 / PBE / STO-3G ≈ -1.34 Ha after MP shift in a 30-bohr box.
    # (DIRECT_TRUNCATED reference is ≈ -1.15 — Makov-Payne shifts by
    # ~ -0.19 to give the Ewald number.)
    assert -2.0 < r.energy < -0.8


# ---------------------------------------------------------------------------
# ω-invariance (the headline correctness witness)
# ---------------------------------------------------------------------------


def test_energy_is_omega_invariant_h2_pbe():
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("PBE")

    energies = []
    for omega in (0.3, 0.5, 1.0, 1.5):
        r = vq.run_rks_periodic_gamma_ewald3d(
            sysp,
            basis,
            opts,
            omega=omega,
            spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)

    spread = max(energies) - min(energies)
    # Same tolerance as the RHF Ewald test — driven by the FFT
    # finite-box error, which the XC additions don't change.
    assert spread < 0.005 * abs(min(energies)), (
        f"ω-invariance violated: {energies}, spread = {spread:.3e}"
    )


# ---------------------------------------------------------------------------
# Makov-Payne consistency vs DIRECT_TRUNCATED
# ---------------------------------------------------------------------------


def test_energy_matches_direct_truncated_in_atomic_limit():
    """v0.6.1 Madelung fix: Ewald-3D and DIRECT_TRUNCATED both
    converge to the molecular-limit energy at large L. Pre-fix
    they differed by 0.5·α·n_elec (the Makov-Payne shift); post-fix
    the Madelung correction in the SCF energy removes that shift.
    """
    box = 30.0
    sysp, basis = _h2_in_box(box=box)
    opts = _default_options("PBE")

    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r_direct = vq.run_rks_periodic(sysp, basis, kmesh, opts)
    r_ewald = vq.run_rks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r_direct.converged
    assert r_ewald.converged

    diff = r_ewald.energy - r_direct.energy
    # 10 mHa tolerance (matches the RHF analog).
    assert abs(diff) < 1e-2, (
        f"Ewald - Direct in atomic limit: diff = {diff:.6f} Ha "
        f"(E_ewald = {r_ewald.energy:.6f}, E_direct = {r_direct.energy:.6f})"
    )


# ---------------------------------------------------------------------------
# Hybrid functional path
# ---------------------------------------------------------------------------


def test_b3lyp_converges_and_uses_hf_exchange():
    """B3LYP carries 20 % HF exchange — exercises the alpha > 0
    branch that builds K via build_jk_gamma_molecular_limit."""
    sysp, basis = _h2_in_box(box=30.0)
    opts = _default_options("B3LYP")
    r = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    # B3LYP energy on H2 / STO-3G in the 30-bohr box is in a similar
    # ballpark to PBE; tight bound only as a sanity check.
    assert -2.0 < r.energy < -0.8
    # HF-exchange contribution must be non-zero (B3LYP α = 0.2).
    # e_hf_exchange is the −(α/2) tr(P·K) integrated energy; for H2
    # at ~ -1 Ha with α = 0.2 it should be a few-mHa to tens-of-mHa
    # negative number — but at minimum, distinguishably non-zero.
    assert abs(r.e_hf_exchange) > 1e-3, (
        f"e_hf_exchange ≈ 0 ({r.e_hf_exchange:.3e}) but B3LYP carries "
        "20 % exact exchange — alpha branch may not be wired in."
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


def test_result_has_expected_shapes():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    r = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    n_bf = basis.nbasis
    assert r.density.shape == (n_bf, n_bf)
    assert r.fock.shape == (n_bf, n_bf)
    assert r.overlap.shape == (n_bf, n_bf)
    assert r.mo_coeffs.shape[0] == n_bf
    assert r.mo_coeffs.shape[1] <= n_bf
    assert r.mo_energies.shape == (r.mo_coeffs.shape[1],)
    assert len(r.scf_trace) >= 1
    first = r.scf_trace[0]
    assert isinstance(first, vq.SCFIteration)
    assert first.iter >= 1
    assert isinstance(first.energy, float)
    assert isinstance(first.delta_e, float)
    assert isinstance(first.grad_norm, float)
    assert r.grid_shape == (100, 100, 100)
    # ω = 0.5 passed explicitly; not auto-derived.
    assert r.omega == pytest.approx(0.5)
    assert r.functional.upper().startswith("PBE")


def test_result_matrices_are_symmetric():
    sysp, basis = _h2_in_box()
    opts = _default_options()
    r = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert np.allclose(r.overlap, r.overlap.T, atol=1e-12)
    assert np.allclose(r.fock, r.fock.T, atol=1e-8)
    assert np.allclose(r.density, r.density.T, atol=1e-8)


def test_energy_decomposition_consistent():
    """E_total = e_electronic + e_nuclear for EWALD_3D gauge.

    The EWALD_3D path disables the Madelung correction (E_madelung_fix=0.0)
    because the G=0 cancellation is handled by the consistent Ewald V_ne +
    E_nn machinery.  Only DIRECT_TRUNCATED paths need the correction.
    """
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    r = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    # For EWALD_3D: E_total = E_elec + E_nuc (no Madelung fix).
    assert abs(r.energy - (r.e_electronic + r.e_nuclear)) < 1e-10
    # PBE has alpha = 0; HF exchange must vanish.
    assert abs(r.e_hf_exchange) < 1e-12


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_accepts_non_orthorhombic_lattice_smoke():
    lat = np.array(
        [
            [30.0, 1.5, 0.0],
            [0.0, 30.0, 0.0],
            [0.0, 0.0, 30.0],
        ]
    )
    c = 15.0
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, lat, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options(iter_limit=1)
    r = vq.run_rks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        grid_shape=(8, 8, 8),
    )
    assert r.n_iter == 1
    assert np.isfinite(r.energy)


def test_rejects_open_shell_system():
    """Closed-shell RKS driver must reject odd-electron systems."""
    c = 15.0
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * 30.0,
        [vq.Atom(1, [c, c, c])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options()
    with pytest.raises(ValueError, match="closed-shell"):
        vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts)


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------


def test_dispatcher_gamma_routes_ewald_to_new_driver():
    """run_rks_periodic_gamma_scf with CoulombMethod.EWALD_3D must
    dispatch to the new Ewald driver and produce the same number as
    a direct call."""
    sysp, basis = _h2_in_box()
    opts = _default_options()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D

    r_direct = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    r_dispatched = vq.run_rks_periodic_gamma_scf(
        sysp,
        basis,
        opts,
        omega=0.5,
    )
    assert r_dispatched.converged
    assert abs(r_dispatched.energy - r_direct.energy) < 1e-10


def test_dispatcher_multi_k_routes_gamma_mesh_to_ewald():
    """A 1×1×1 mesh through run_rks_periodic_scf must route EWALD_3D
    to the Γ-only Ewald driver — multi-k KS Ewald isn't implemented
    yet, but a Γ-only mesh is the trivial fallback."""
    sysp, basis = _h2_in_box()
    opts = _default_options()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
    r = vq.run_rks_periodic_scf(sysp, basis, kmesh, opts, omega=0.5)
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSEwaldResult)


def test_dispatcher_multi_k_dense_ewald_mesh_routes_to_new_driver():
    """Multi-k EWALD_3D KS shipped in Phase 15c-2: dispatcher routes
    a dense mesh to ``run_rks_periodic_multi_k_ewald3d``. Was a
    NotImplementedError-asserting test before 15c-2."""
    sysp, basis = _h2_in_box()
    opts = _default_options()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    kmesh = vq.monkhorst_pack(sysp, [2, 1, 1])
    r = vq.run_rks_periodic_scf(sysp, basis, kmesh, opts)
    assert r.converged
    assert isinstance(r, vq.PeriodicRKSMultiKEwaldResult)
