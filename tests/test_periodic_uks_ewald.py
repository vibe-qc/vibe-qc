"""Phase 15c-3a: Γ-point periodic UKS SCF driver using EWALD_3D.

Open-shell DFT counterpart of the RKS Ewald tests + UHF Ewald tests.
Pin:

  1. H atom (mult=2) converges with ⟨S²⟩ = 3/4 = ideal.
  2. Closed-shell H₂ through UKS reproduces RKS to ~µHa
     (degenerate-spin-density limit).
  3. Hybrid (B3LYP) path exercises α > 0 — distinguishably non-zero
     ``e_hf_exchange``.
  4. ω-invariance.
  5. Result-shape / energy-decomposition consistency.
  6. Skew-cell FFT metric support.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_runner as pr


def _h2_in_box(box: float = 30.0):
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, c - 0.7]),
        vq.Atom(1, [c, c, c + 0.7]),
    ]
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_in_box(box: float = 30.0):
    c = box / 2
    atoms = [vq.Atom(1, [c, c, c])]
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box, atoms, charge=0, multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _li_atom_in_box(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(3, [c, c, c])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h2_slab(box: float = 18.0, vacuum: float = 45.0):
    c = box / 2
    atoms = [
        vq.Atom(1, [c, c, vacuum / 2 - 0.7]),
        vq.Atom(1, [c, c, vacuum / 2 + 0.7]),
    ]
    sysp = vq.PeriodicSystem(2, np.diag([box, box, vacuum]), atoms)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_slab(box: float = 18.0, vacuum: float = 45.0):
    c = box / 2
    atoms = [vq.Atom(1, [c, c, vacuum / 2])]
    sysp = vq.PeriodicSystem(
        2,
        np.diag([box, box, vacuum]),
        atoms,
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _default_options(functional: str = "PBE", iter_limit: int = 60,
                     damping: float = 0.3):
    opts = vq.PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.damping = damping
    opts.max_iter = iter_limit
    opts.use_diis = True
    return opts


def _slab_options(functional: str = "PBE"):
    opts = _default_options(functional, iter_limit=50, damping=0.3)
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.SLAB_EWALD_2D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.lattice_opts.slab_ewald_alpha = 0.4
    opts.conv_tol_energy = 1e-7
    opts.conv_tol_grad = 1e-5
    return opts


def _assert_reported_spin_densities_match_orbitals(result) -> None:
    """Every reported spin density is defined by its C/f pair."""
    for spin in ("alpha", "beta"):
        coeffs = np.asarray(getattr(result, f"mo_coeffs_{spin}"))
        occupations = np.asarray(getattr(result, f"occupations_{spin}"))
        reconstructed = (coeffs * occupations) @ coeffs.conj().T
        np.testing.assert_allclose(
            getattr(result, f"density_{spin}"),
            reconstructed,
            atol=1e-12,
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def test_h_atom_converges_with_correct_spin():
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    # Doublet ideal: S(S+1) = 0.75 with S=1/2.
    assert abs(r.s_squared - 0.75) < 1e-6
    assert abs(r.s_squared_ideal - 0.75) < 1e-12
    # H atom UKS PBE/STO-3G: ~ -0.5 Ha (after MP shift in 30-bohr box).
    assert -1.0 < r.energy < 0.0


def test_closed_shell_uks_matches_rks_to_microhartree():
    """Closed-shell H₂ with mult=1 → UKS should give the same answer
    as RKS to ~µHa (formally identical at the SCF fixed point)."""
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    r_uks = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    r_rks = vq.run_rks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r_uks.converged and r_rks.converged
    assert abs(r_uks.energy - r_rks.energy) < 1e-6
    assert r_uks.scf_trace[-1].energy == pytest.approx(r_uks.energy, abs=1e-12)
    _assert_reported_spin_densities_match_orbitals(r_uks)
    # Closed-shell ⟨S²⟩ → 0 (α and β occupied orbitals identical).
    assert abs(r_uks.s_squared) < 1e-6


# ---------------------------------------------------------------------------
# Hybrid path
# ---------------------------------------------------------------------------

def test_b3lyp_path_uses_hf_exchange():
    sysp, basis = _h2_in_box()
    opts = _default_options("B3LYP")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    assert abs(r.e_hf_exchange) > 1e-3, (
        f"e_hf_exchange ≈ 0 ({r.e_hf_exchange:.3e}) but B3LYP α = 0.2"
    )


# ---------------------------------------------------------------------------
# ω-invariance
# ---------------------------------------------------------------------------

def test_omega_invariance_h_atom():
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    energies = []
    for omega in (0.3, 0.5, 1.0, 1.5):
        r = vq.run_uks_periodic_gamma_ewald3d(
            sysp, basis, opts, omega=omega, spacing_bohr=0.3,
        )
        assert r.converged
        energies.append(r.energy)
    spread = max(energies) - min(energies)
    # 1 % tolerance — single-electron H system has tighter sensitivity
    # to the FFT grid than H₂ but still well within finite-box bounds.
    assert spread < 0.01 * abs(min(energies)), (
        f"ω-invariance violated: {energies}, spread = {spread:.3e}"
    )


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------

def test_result_has_expected_shapes():
    sysp, basis = _h_atom_in_box()
    opts = _default_options("PBE")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    n_bf = basis.nbasis
    assert r.density_alpha.shape == (n_bf, n_bf)
    assert r.density_beta.shape == (n_bf, n_bf)
    assert r.fock_alpha.shape == (n_bf, n_bf)
    assert r.fock_beta.shape == (n_bf, n_bf)
    assert r.occupations_alpha.shape == r.mo_energies_alpha.shape
    assert r.occupations_beta.shape == r.mo_energies_beta.shape
    assert r.occupations_alpha.sum() == pytest.approx(1.0)
    assert r.occupations_beta.sum() == pytest.approx(0.0)
    assert r.overlap.shape == (n_bf, n_bf)
    assert r.functional.upper().startswith("PBE")
    assert len(r.scf_trace) == r.n_iter
    orbital_text = pr._mo_summary(r)
    assert "Crystal orbital energies (alpha)" in orbital_text
    assert "Crystal orbital energies (beta)" in orbital_text


def test_damped_li_terminal_state_is_not_false_positive_converged():
    """A provisional damped iterate cannot certify its fresh density."""
    sysp, basis = _li_atom_in_box()
    opts = _default_options("PBE", iter_limit=2, damping=0.95)
    opts.use_diis = False
    opts.conv_tol_energy = 0.003
    opts.conv_tol_grad = 0.03

    result = vq.run_uks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.5,
        allow_dense_ionic=True,
    )

    assert not result.converged
    assert result.n_iter == 2
    assert len(result.scf_trace) == result.n_iter
    _assert_reported_spin_densities_match_orbitals(result)
    assert result.scf_trace[-1].energy == pytest.approx(result.energy, abs=1e-12)
    terminal_delta_e = abs(result.scf_trace[-1].delta_e)
    FDS_alpha = result.fock_alpha @ result.density_alpha @ result.overlap
    FDS_beta = result.fock_beta @ result.density_beta @ result.overlap
    terminal_grad = np.sqrt(
        np.linalg.norm(FDS_alpha - FDS_alpha.T) ** 2
        + np.linalg.norm(FDS_beta - FDS_beta.T) ** 2
    )
    assert terminal_delta_e > opts.conv_tol_energy
    assert terminal_grad > opts.conv_tol_grad
    assert result.scf_trace[-1].grad_norm == pytest.approx(terminal_grad)


def test_one_cycle_cap_returns_consistent_orbital_state():
    """The non-provisional max-iteration path also gets a terminal rebuild."""
    sysp, basis = _li_atom_in_box()
    opts = _default_options("PBE", iter_limit=1)
    opts.use_diis = False

    result = vq.run_uks_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        grid_shape=(8, 8, 8),
        allow_dense_ionic=True,
    )

    assert not result.converged
    assert result.n_iter == 1
    assert len(result.scf_trace) == 1
    assert result.scf_trace[-1].energy == pytest.approx(result.energy, abs=1e-12)
    _assert_reported_spin_densities_match_orbitals(result)


def test_energy_decomposition_consistent():
    """E_total = e_electronic + e_nuclear for the EWALD_3D gauge.

    The Γ-only UKS Ewald driver now forces EWALD_3D (handover F4,
    2026-06-01), so V_ne / e_nuclear are Ewald-gauge-consistent with the
    Hartree J and the v0.6.1 Madelung leak correction is disabled
    (``madelung_energy_correction_for_lat`` returns 0.0 for EWALD_3D).
    Mirrors the RKS Γ decomposition test (test_periodic_rks_ewald.py).
    """
    sysp, basis = _h2_in_box()
    opts = _default_options("PBE")
    r = vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts, omega=0.5)
    assert r.converged
    # For EWALD_3D: E_total = E_elec + E_nuc (no Madelung fix).
    assert abs(r.energy - (r.e_electronic + r.e_nuclear)) < 1e-10
    # Pure DFT — no HF exchange.
    assert abs(r.e_hf_exchange) < 1e-12


# ---------------------------------------------------------------------------
# Slab Ewald
# ---------------------------------------------------------------------------

def test_slab_ewald_2d_h_atom_converges_with_correct_spin():
    sysp, basis = _h_atom_slab()
    opts = _slab_options("PBE")

    r = vq.run_uks_periodic_gamma_ewald2d(sysp, basis, opts)

    assert r.converged
    assert r.s_squared == pytest.approx(0.75, abs=1e-6)
    assert r.s_squared_ideal == pytest.approx(0.75, abs=1e-12)
    assert r.energy == pytest.approx(r.e_electronic + r.e_nuclear, abs=1e-10)
    assert r.omega == pytest.approx(opts.lattice_opts.slab_ewald_alpha)
    assert r.grid_shape == (0, 0, 0)


def test_slab_ewald_2d_closed_shell_uks_matches_rks():
    sysp, basis = _h2_slab()
    opts_uks = _slab_options("PBE")
    opts_rks = _slab_options("PBE")

    r_uks = vq.run_uks_periodic_gamma_ewald2d(sysp, basis, opts_uks)
    r_rks = vq.run_rks_periodic_gamma_ewald2d(sysp, basis, opts_rks)

    assert r_uks.converged and r_rks.converged
    assert r_uks.energy == pytest.approx(r_rks.energy, abs=1e-6)
    assert abs(r_uks.s_squared) < 1e-6


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_nonpositive_max_iter_raises():
    sysp, basis = _h_atom_in_box()
    opts = _default_options(iter_limit=0)

    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        vq.run_uks_periodic_gamma_ewald3d(sysp, basis, opts)


def test_accepts_non_orthorhombic_lattice_smoke():
    lat = np.array([
        [30.0, 1.5, 0.0],
        [0.0, 30.0, 0.0],
        [0.0, 0.0, 30.0],
    ])
    c = 15.0
    atoms = [vq.Atom(1, [c, c, c])]
    sysp = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = _default_options(iter_limit=1)
    r = vq.run_uks_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, grid_shape=(8, 8, 8),
    )
    assert r.n_iter == 1
    assert np.isfinite(r.energy)
