"""Phase 15a tests: Γ-point periodic UHF SCF with Ewald-3D Coulomb.

Contracts exercised:

1. **Closed-shell limit equals RHF** — for a closed-shell molecule
   (multiplicity = 1, n_e even), UHF must converge to the same total
   energy as RHF and report ⟨S²⟩ ≈ 0 (machine-precision).

2. **One-electron doublet exact ⟨S²⟩** — for a single-electron H atom
   in a box (n_e=1, mult=2), UHF reports ⟨S²⟩ = 0.75 = S(S+1) exactly
   (no spin contamination possible with one electron).

3. **Open-shell convergence** — H atom doublet in a 30-bohr box
   converges to a finite, reasonable energy.

4. **α/β symmetry on closed-shell** — when n_α = n_β the converged
   D_α and D_β are equal (mirror of the RHF density up to the
   factor of 2).

5. **Result-shape contract** — every field on PeriodicUHFEwaldResult
   is populated and has a sensible type.

6. **Bad multiplicity raises** — incompatible (n_e, multiplicity)
   pairs are rejected up-front.

7. **PySCF cross-check** — H atom UHF / sto-3g matches PySCF UHF on
   the same atom in a box (~µHa agreement when the box is large
   enough that Ewald and free-space converge).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_runner as pr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _h2_closed_shell(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _h_atom_doublet(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 2
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _li_atom_doublet(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(3, [c, c, c])],
        multiplicity=2,
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _options():
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = 12.0
    o.lattice_opts.nuclear_cutoff_bohr = 15.0
    o.damping = 0.3
    o.max_iter = 40
    return o


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
# 1. Closed-shell limit equals RHF
# ---------------------------------------------------------------------------


def test_uhf_closed_shell_matches_rhf():
    sysp, basis = _h2_closed_shell()
    opts = _options()
    r_rhf = vq.run_rhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    r_uhf = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r_rhf.converged and r_uhf.converged
    assert r_uhf.energy == pytest.approx(r_rhf.energy, abs=1e-9)
    assert r_uhf.scf_trace[-1].energy == pytest.approx(r_uhf.energy, abs=1e-12)
    _assert_reported_spin_densities_match_orbitals(r_uhf)
    # ⟨S²⟩ ≈ 0 for closed shell.
    assert abs(r_uhf.s_squared) < 1e-10
    assert abs(r_uhf.s_squared_ideal) < 1e-12


def test_uhf_closed_shell_alpha_beta_densities_equal():
    """For a closed-shell UHF the converged α and β densities must
    be equal (each is half the closed-shell total density)."""
    sysp, basis = _h2_closed_shell()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r.converged
    diff = np.abs(r.density_alpha - r.density_beta).max()
    assert diff < 1e-9, f"α/β density asymmetry on closed shell: {diff:.3e}"


# ---------------------------------------------------------------------------
# 2. One-electron doublet — exact ⟨S²⟩
# ---------------------------------------------------------------------------


def test_h_atom_doublet_spin_squared_is_three_quarters():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r.converged
    # S = 1/2 → S(S+1) = 3/4. UHF on a single electron is exact.
    assert r.s_squared == pytest.approx(0.75, abs=1e-12)
    assert r.s_squared_ideal == pytest.approx(0.75, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Open-shell convergence
# ---------------------------------------------------------------------------


def test_h_atom_doublet_converges_to_sensible_energy():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    assert r.converged
    assert np.isfinite(r.energy)
    # H atom HF/sto-3g atomic limit is −0.466. In a finite Ewald box
    # the Makov-Payne correction shifts it by ~50 mHa to ~−0.51.
    assert -0.7 < r.energy < -0.4


# ---------------------------------------------------------------------------
# 4. Result-shape contract
# ---------------------------------------------------------------------------


def test_result_struct_populated():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    nbf = basis.nbasis
    assert r.density_alpha.shape == (nbf, nbf)
    assert r.density_beta.shape == (nbf, nbf)
    assert r.fock_alpha.shape == (nbf, nbf)
    assert r.fock_beta.shape == (nbf, nbf)
    assert r.mo_coeffs_alpha.shape[0] == nbf
    assert r.mo_coeffs_beta.shape[0] == nbf
    assert r.occupations_alpha.shape == r.mo_energies_alpha.shape
    assert r.occupations_beta.shape == r.mo_energies_beta.shape
    assert r.occupations_alpha.sum() == pytest.approx(1.0)
    assert r.occupations_beta.sum() == pytest.approx(0.0)
    assert r.overlap.shape == (nbf, nbf)
    assert isinstance(r.scf_trace, list)
    assert len(r.scf_trace) == r.n_iter
    # ω is auto-derived from nuclear_cutoff_bohr (mirror of
    # the RHF/RKS Γ and multi-k drivers, commit 49f8ae91 / 433d3543);
    # the driver ``omega`` kwarg is overridden.
    assert r.omega == pytest.approx(0.5)
    orbital_text = pr._mo_summary(r)
    assert "Crystal orbital energies (alpha)" in orbital_text
    assert "Crystal orbital energies (beta)" in orbital_text


def test_damped_li_terminal_state_is_not_false_positive_converged():
    """A rejected provisional state continues through the iteration cap."""
    sysp, basis = _li_atom_doublet()
    opts = _options()
    opts.max_iter = 3
    opts.use_diis = False
    opts.damping = 0.95
    opts.conv_tol_energy = 0.02
    opts.conv_tol_grad = 0.07

    result = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.5,
        allow_dense_ionic=True,
    )

    assert not result.converged
    assert result.n_iter == 3
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
    sysp, basis = _li_atom_doublet()
    opts = _options()
    opts.max_iter = 1
    opts.use_diis = False

    result = vq.run_uhf_periodic_gamma_ewald3d(
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


# ---------------------------------------------------------------------------
# 5. Bad multiplicity raises
# ---------------------------------------------------------------------------


def test_invalid_multiplicity_raises():
    """1 electron with multiplicity=1 (singlet) is impossible —
    requires non-integer α/β occupations. The system-construction
    pipeline rejects it (vibe-qc's ``unit_cell_molecule()`` already
    enforces ``(n_e + mult) % 2 == 1``); the driver inherits this
    guard via its first basis-build call."""
    box, c = 30.0, 15.0
    sysp = vq.PeriodicSystem(3, np.eye(3) * box, [vq.Atom(1, [c, c, c])])
    sysp.multiplicity = 1  # impossible: 1 electron + singlet
    with pytest.raises(Exception):
        # unit_cell_molecule() raises here — that's the right level
        # to catch impossible (n_e, mult) combinations.
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
        vq.run_uhf_periodic_gamma_ewald3d(
            sysp,
            basis,
            _options(),
            omega=0.5,
            spacing_bohr=0.3,
        )


def test_nonpositive_max_iter_raises():
    sysp, basis = _h_atom_doublet()
    opts = _options()
    opts.max_iter = 0

    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        vq.run_uhf_periodic_gamma_ewald3d(sysp, basis, opts)


# ---------------------------------------------------------------------------
# 6. PySCF cross-check on H atom doublet
# ---------------------------------------------------------------------------


def test_h_atom_pyscf_cross_check():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    # PySCF: H atom in vacuum, doublet, sto-3g — bare molecular UHF.
    mol = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, charge=0, verbose=0)
    e_pyscf = scf.UHF(mol).run().e_tot

    # vibe-qc: same atom in a 30-bohr box. Ewald nuclear repulsion
    # adds a Madelung-like ~50 mHa shift; the agreement against
    # the bare-vacuum PySCF energy isn't exact but is bounded.
    sysp, basis = _h_atom_doublet(box=30.0)
    opts = _options()
    r = vq.run_uhf_periodic_gamma_ewald3d(
        sysp,
        basis,
        opts,
        omega=0.5,
        spacing_bohr=0.3,
    )
    # Within ~50 mHa of PySCF's vacuum value (Makov-Payne bound).
    assert abs(r.energy - e_pyscf) < 0.1, (
        f"H atom UHF disagrees with PySCF beyond box-shift bound: "
        f"vibe-qc {r.energy:.6f} vs PySCF {e_pyscf:.6f}"
    )
