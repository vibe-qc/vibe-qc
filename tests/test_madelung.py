"""Phase 12e-c-4c-iv tests: Madelung-cancellation helpers.

Contracts exercised:

1. **Q_e measurement** — ``cell_electron_charge(D, S) = N_electrons``
   to SCF convergence on a converged neutral H2 calculation.

2. **Q_n bookkeeping** — ``cell_nuclear_charge`` sums atomic Z values
   from the unit cell.

3. **Cubic edge** — ``cubic_cell_edge`` recovers the box edge for a
   cubic cell.

4. **α scalar** — ``madelung_alpha(Q, L) = -α_M · Q / L`` (regression
   against the simple-cubic Madelung constant 2.837297).

5. **Neutrality cancellation** — for a neutral H2 cell the net
   correction ``α_e − α_n`` is zero to machine precision (the
   Madelung-cancellation contract that names this phase).

6. **Charged cell** — for a synthetic charged density (D scaled so
   ``Q_e ≠ Q_n``) the net α is non-zero and matches the expected
   ``-α_M · ΔQ / L`` formula.

7. **Fock shift** — ``apply_madelung_correction(F, S, α) = F + α·S``
   exactly; works for both real and complex inputs.

8. **Per-k application** — ``apply_madelung_correction_per_k`` applies
   the same scalar to every (F(k), S(k)) pair and preserves
   shapes/dtypes.

9. **Energy increment** — for a charged cell with ΔQ ≠ 0, the energy
   shift from the correction is ``½ tr(D · α·S) = ½ α Q_e`` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2(box: float = 30.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3, np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]),
         vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _converged_h2(box: float = 30.0):
    sysp, basis = _h2(box)
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 12
    opts.lattice_opts.nuclear_cutoff_bohr = 15
    opts.damping = 0.3
    opts.max_iter = 40
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.3,
    )
    assert r.converged
    return sysp, basis, r


# ---------------------------------------------------------------------------
# Q_e bookkeeping
# ---------------------------------------------------------------------------

def test_cell_electron_charge_recovers_n_electrons():
    sysp, _, r = _converged_h2()
    Q_e = vq.cell_electron_charge(r.density, r.overlap)
    assert Q_e == pytest.approx(float(sysp.n_electrons()), abs=1e-9)


def test_cell_nuclear_charge_sums_atomic_z():
    sysp, _ = _h2()
    assert vq.cell_nuclear_charge(sysp) == 2.0   # H2: Z = 1 + 1


def test_net_charge_is_zero_for_neutral_cell():
    sysp, _, r = _converged_h2()
    net = vq.cell_net_charge(r.density, r.overlap, sysp)
    assert abs(net) < 1e-9


# ---------------------------------------------------------------------------
# Cubic edge length
# ---------------------------------------------------------------------------

def test_cubic_cell_edge_for_cubic_box():
    sysp, _ = _h2(box=30.0)
    assert vq.cubic_cell_edge(sysp) == pytest.approx(30.0)


def test_cubic_cell_edge_for_orthorhombic_uses_geometric_mean():
    sysp = vq.PeriodicSystem(
        3, np.diag([20.0, 30.0, 40.0]),
        [vq.Atom(1, [0, 0, 0])],
    )
    L_expected = (20.0 * 30.0 * 40.0) ** (1.0 / 3.0)
    assert vq.cubic_cell_edge(sysp) == pytest.approx(L_expected)


# ---------------------------------------------------------------------------
# α scalar: -α_M Q / L
# ---------------------------------------------------------------------------

def test_madelung_alpha_matches_simple_cubic_formula():
    Q, L = 2.0, 30.0
    alpha = vq.madelung_alpha(Q, L)
    # Simple-cubic Madelung 2.837297; α = -α_M Q / L.
    assert alpha == pytest.approx(-2.837297 * 2.0 / 30.0, rel=1e-12)


def test_madelung_alpha_sign_and_scaling():
    # Doubling Q doubles |α|; doubling L halves it; sign tracks Q.
    a1 = vq.madelung_alpha(1.0, 10.0)
    a2 = vq.madelung_alpha(2.0, 10.0)
    a3 = vq.madelung_alpha(1.0, 20.0)
    assert a2 == pytest.approx(2.0 * a1)
    assert a3 == pytest.approx(0.5 * a1)
    assert a1 < 0   # α = -α_M Q/L is negative for positive Q


# ---------------------------------------------------------------------------
# Neutrality cancellation — the contract that names this phase
# ---------------------------------------------------------------------------

def test_neutrality_cancellation_on_h2():
    """For a converged neutral H2 cell, α_net = α_e − α_n vanishes to
    machine precision: the Madelung cancellation."""
    sysp, _, r = _converged_h2()
    alpha_net = vq.madelung_correction_scalar(r.density, r.overlap, sysp)
    assert abs(alpha_net) < 1e-12, (
        f"Madelung cancellation broken on neutral cell: "
        f"α_net = {alpha_net:.3e}"
    )


# ---------------------------------------------------------------------------
# Charged cell — α_net = -α_M ΔQ / L
# ---------------------------------------------------------------------------

def test_charged_cell_has_expected_alpha_net():
    """Build a synthetic 'H2⁺-like' density by scaling the converged
    H2 density to one electron, leaving the (neutral) nuclear charge
    in place. The net α should then track the expected
    -α_M (Q_e - Q_n) / L."""
    sysp, _, r = _converged_h2()
    D_scaled = 0.5 * r.density   # Q_e = 1.0 (was 2.0); Q_n still 2.0
    alpha_net = vq.madelung_correction_scalar(D_scaled, r.overlap, sysp)
    L = vq.cubic_cell_edge(sysp)
    expected = vq.madelung_alpha(1.0, L) - vq.madelung_alpha(2.0, L)
    assert alpha_net == pytest.approx(expected, rel=1e-10)
    assert alpha_net != pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_madelung_correction: F → F + αS
# ---------------------------------------------------------------------------

def test_apply_madelung_correction_real():
    rng = np.random.default_rng(0)
    n = 4
    F = rng.standard_normal((n, n))
    S = rng.standard_normal((n, n))
    alpha = 0.123
    F_out = vq.apply_madelung_correction(F, S, alpha)
    assert np.allclose(F_out, F + alpha * S)
    assert F_out.dtype == np.float64


def test_apply_madelung_correction_complex():
    rng = np.random.default_rng(1)
    n = 4
    F = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    S = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    alpha = -0.456
    F_out = vq.apply_madelung_correction(F, S, alpha)
    assert np.allclose(F_out, F + alpha * S)
    assert F_out.dtype == np.complex128


def test_apply_madelung_correction_zero_alpha_is_identity():
    rng = np.random.default_rng(2)
    F = rng.standard_normal((3, 3))
    S = rng.standard_normal((3, 3))
    F_out = vq.apply_madelung_correction(F, S, 0.0)
    assert np.allclose(F_out, F)


# ---------------------------------------------------------------------------
# Per-k application
# ---------------------------------------------------------------------------

def test_apply_madelung_correction_per_k_applies_uniformly():
    rng = np.random.default_rng(3)
    n = 3
    n_k = 4
    F_k = [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
           for _ in range(n_k)]
    S_k = [rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
           for _ in range(n_k)]
    alpha = 0.789
    out = vq.apply_madelung_correction_per_k(F_k, S_k, alpha)
    assert len(out) == n_k
    for i in range(n_k):
        assert np.allclose(out[i], F_k[i] + alpha * S_k[i])
        assert out[i].dtype == np.complex128


def test_apply_madelung_correction_per_k_length_mismatch_raises():
    F_k = [np.eye(2)]
    S_k = [np.eye(2), np.eye(2)]
    with pytest.raises(ValueError, match="must match length"):
        vq.apply_madelung_correction_per_k(F_k, S_k, 0.1)


# ---------------------------------------------------------------------------
# Energy increment from the correction
# ---------------------------------------------------------------------------

def test_energy_increment_from_correction_equals_half_alpha_qe():
    """Δ E = ½ tr(D · α·S) = ½ · α · tr(D·S) = ½ · α · Q_e exactly."""
    sysp, _, r = _converged_h2()
    # Use a synthetic non-zero α (charged-cell scenario) so the
    # increment is non-trivial.
    alpha = 0.05
    F_corr = vq.apply_madelung_correction(r.fock, r.overlap, alpha)
    dF = F_corr - r.fock
    dE = 0.5 * float(np.trace(r.density @ dF))
    Q_e = vq.cell_electron_charge(r.density, r.overlap)
    assert dE == pytest.approx(0.5 * alpha * Q_e, rel=1e-10)
