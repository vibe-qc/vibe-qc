"""D4 reference-data machinery — Phase D4b-1.

Pins:

  1. The imaginary-frequency grid + Casimir-Polder integrator
     reproduce the analytic London C6 for single-pole model
     polarizabilities (the rigorous correctness test — the
     integrand is rational and smooth, so Gauss-Legendre after the
     rational substitution converges fast).
  2. casimir_polder_c6 is symmetric in its two arguments and scales
     linearly in each polarizability.
  3. uncoupled_polarizability_imag_freq:
       * α⁰(iω=0) equals the independent-particle static
         polarizability Σ 4 d²/Δε computed directly;
       * α⁰(iω) is strictly positive and monotone-decreasing in ω;
       * α⁰(iω) → 0 as ω → ∞, with the expected 1/ω² tail;
       * the He-atom self-C6 from Casimir-Polder of α⁰(iω) is
         positive and physically sane (order ~1 a.u.).
  4. Shape / error-handling contracts.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    casimir_polder_c6,
    imaginary_frequency_grid,
    london_c6_single_pole,
    run_rhf,
    uncoupled_polarizability_imag_freq,
)


# ---------------------------------------------------------------------
# Grid + Casimir-Polder integrator — analytic validation.
# ---------------------------------------------------------------------

def _single_pole(omega, alpha0, omega0):
    """Unsöld / Drude model polarizability α(iω) = α0 / (1 + (ω/ω0)²)."""
    return alpha0 / (1.0 + (omega / omega0) ** 2)


@pytest.mark.parametrize("n_points", [15, 23, 40])
def test_casimir_polder_reproduces_london_same_pole(n_points):
    """For two single-pole polarizabilities with the SAME pole, the
    Casimir-Polder integral must equal the London closed form."""
    alpha0_a, omega0_a = 1.38, 0.80   # He-like
    alpha0_b, omega0_b = 5.20, 0.55   # heavier-atom-like
    omegas, weights = imaginary_frequency_grid(n_points=n_points,
                                               omega_scale=0.6)
    a = _single_pole(omegas, alpha0_a, omega0_a)
    b = _single_pole(omegas, alpha0_b, omega0_b)
    c6 = casimir_polder_c6(a, b, weights)
    c6_exact = london_c6_single_pole(alpha0_a, omega0_a,
                                     alpha0_b, omega0_b)
    rel = abs(c6 - c6_exact) / c6_exact
    # 23-point grid nails the rational integrand to ~1e-4 or better.
    tol = 5e-3 if n_points == 15 else 1e-4
    assert rel < tol, (
        f"n={n_points}: C6={c6:.8f} vs London exact={c6_exact:.8f}, "
        f"rel err {rel:.2e}")


def test_casimir_polder_grid_converges():
    """The quadrature error shrinks as the grid is refined."""
    a0a, w0a, a0b, w0b = 2.0, 0.7, 3.5, 0.45
    exact = london_c6_single_pole(a0a, w0a, a0b, w0b)
    errs = []
    for n in (8, 16, 32):
        omegas, weights = imaginary_frequency_grid(n_points=n)
        c6 = casimir_polder_c6(_single_pole(omegas, a0a, w0a),
                               _single_pole(omegas, a0b, w0b),
                               weights)
        errs.append(abs(c6 - exact))
    assert errs[1] < errs[0] and errs[2] < errs[1], (
        f"quadrature did not converge monotonically: {errs}")


def test_casimir_polder_symmetric_and_linear():
    """C6(A,B) == C6(B,A); C6 scales linearly in each polarizability."""
    omegas, weights = imaginary_frequency_grid()
    a = _single_pole(omegas, 1.5, 0.6)
    b = _single_pole(omegas, 4.0, 0.4)
    c6_ab = casimir_polder_c6(a, b, weights)
    c6_ba = casimir_polder_c6(b, a, weights)
    assert c6_ab == pytest.approx(c6_ba, rel=1e-14)
    # Linear in alpha_a.
    c6_2a = casimir_polder_c6(2.0 * a, b, weights)
    assert c6_2a == pytest.approx(2.0 * c6_ab, rel=1e-12)


def test_imaginary_frequency_grid_contract():
    """Grid invariants: positive ascending frequencies, positive
    weights, ∫ 1/(1+ω²) dω = π/2 recovered."""
    omegas, weights = imaginary_frequency_grid(n_points=30, omega_scale=1.0)
    assert (omegas > 0).all()
    assert (np.diff(omegas) > 0).all()
    assert (weights > 0).all()
    # ∫_0^∞ dω/(1+ω²) = π/2.
    integral = np.sum(weights / (1.0 + omegas ** 2))
    assert integral == pytest.approx(np.pi / 2, rel=1e-6)


def test_imaginary_frequency_grid_rejects_bad_args():
    with pytest.raises(ValueError):
        imaginary_frequency_grid(n_points=0)
    with pytest.raises(ValueError):
        imaginary_frequency_grid(omega_scale=0.0)


def test_casimir_polder_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        casimir_polder_c6([1.0, 2.0], [1.0], [1.0, 1.0])


# ---------------------------------------------------------------------
# Uncoupled dynamic polarizability.
# ---------------------------------------------------------------------

def _he_rhf():
    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])])
    basis = BasisSet(mol, "cc-pvdz")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-11
    hf = run_rhf(mol, basis, opts)
    assert hf.converged
    return mol, basis, hf


def _h2_rhf():
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    basis = BasisSet(mol, "cc-pvdz")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-11
    hf = run_rhf(mol, basis, opts)
    assert hf.converged
    return mol, basis, hf


def test_uncoupled_alpha_static_limit():
    """α⁰(iω=0) equals the independent-particle static polarizability
    Σ 4 d²/Δε — checked by evaluating at ω=0 and comparing with a
    second, hand-rolled sum-over-states evaluation."""
    mol, basis, hf = _h2_rhf()
    alpha0 = uncoupled_polarizability_imag_freq(hf, basis, mol, [0.0])[0]

    # Hand-rolled reference: same formula, recomputed independently.
    from vibeqc import compute_dipole
    from vibeqc.cphf import _infer_n_occ
    from vibeqc.properties import center_of_mass
    C = np.asarray(hf.mo_coeffs)
    eps = np.asarray(hf.mo_energies)
    n_occ = _infer_n_occ(np.asarray(hf.density), C)
    C_occ, C_vir = C[:, :n_occ], C[:, n_occ:]
    delta = eps[n_occ:][None, :] - eps[:n_occ][:, None]
    dip = compute_dipole(basis, [float(x) for x in center_of_mass(mol)])
    M = np.stack([np.asarray(dip.x), np.asarray(dip.y), np.asarray(dip.z)])
    ref = 0.0
    for ax in range(3):
        d = C_occ.T @ M[ax] @ C_vir
        ref += np.sum(4.0 * d * d / delta)
    ref /= 3.0
    assert alpha0 == pytest.approx(ref, rel=1e-12)
    assert alpha0 > 0.0


def test_uncoupled_alpha_monotone_decreasing():
    """α⁰(iω) strictly decreases as ω increases."""
    mol, basis, hf = _he_rhf()
    omegas = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0])
    alpha = uncoupled_polarizability_imag_freq(hf, basis, mol, omegas)
    assert (np.diff(alpha) < 0).all(), f"not monotone: {alpha}"
    assert (alpha > 0).all()


def test_uncoupled_alpha_high_frequency_tail():
    """As ω → ∞, α⁰(iω) → S/ω² where S = ⅓ Σ 4(ε_a−ε_i)|d|²
    (independent of ω). Verify the ω²·α⁰ product flattens to a
    constant at large ω."""
    mol, basis, hf = _he_rhf()
    omegas = np.array([200.0, 400.0, 800.0])
    alpha = uncoupled_polarizability_imag_freq(hf, basis, mol, omegas)
    tail = omegas ** 2 * alpha
    # The three ω²·α values should agree to a few parts in 10⁴.
    assert np.ptp(tail) / np.mean(tail) < 1e-3, (
        f"high-ω tail not ~1/ω²: ω²·α = {tail}")


def test_uncoupled_alpha_he_self_c6_is_sane():
    """Casimir-Polder self-integral of He's uncoupled α⁰(iω) gives a
    positive C6.

    This is an integration-pipeline sanity check, NOT an accuracy
    claim. The uncoupled (no CPHF response) level on a small basis
    substantially underestimates the polarizability — and since
    C6 ∝ α², the He self-C6 comes out well below the true He-He
    C6 ≈ 1.46 a.u. (correlated reference). What this test guards
    against is a broken pipeline: a sign error, a dropped factor, a
    zero-α bug. The accurate He C6 is the job of the *coupled*
    α(iω) provider (D4b-1 second half) on an adequate basis."""
    mol, basis, hf = _he_rhf()
    omegas, weights = imaginary_frequency_grid(n_points=23)
    alpha = uncoupled_polarizability_imag_freq(hf, basis, mol, omegas)
    c6 = casimir_polder_c6(alpha, alpha, weights)
    # Positive, finite, and below the correlated reference (uncoupled
    # underestimates) — a wide but still-diagnostic band.
    assert 0.0 < c6 < 1.46, (
        f"He uncoupled self-C6 = {c6:.4f} a.u. — outside the expected "
        "(0, 1.46) band for an uncoupled small-basis estimate")


def test_uncoupled_alpha_shape():
    """Output shape matches the omega grid length."""
    mol, basis, hf = _he_rhf()
    for n in (1, 5, 23):
        omegas = np.linspace(0.0, 5.0, n)
        alpha = uncoupled_polarizability_imag_freq(hf, basis, mol, omegas)
        assert alpha.shape == (n,)


# ---------------------------------------------------------------------
# Coupled (CPHF / TD-HF response) dynamic polarizability — the
# production α(iω) provider for the D4 reference C6 data.
# ---------------------------------------------------------------------

def test_coupled_alpha_static_limit_matches_cphf():
    """coupled_polarizability_imag_freq at ω=0 must equal the isotropic
    average of the existing (validated) static CPHF polarizability
    dipole_polarizability_rhf — the rigorous tie-in of the new
    imaginary-frequency response to the established static code."""
    from vibeqc import coupled_polarizability_imag_freq
    from vibeqc.cphf import dipole_polarizability_rhf

    mol, basis, hf = _h2_rhf()
    alpha_dyn0 = coupled_polarizability_imag_freq(hf, basis, mol, [0.0])[0]
    alpha_static = np.trace(dipole_polarizability_rhf(hf, basis, mol)) / 3.0
    assert alpha_dyn0 == pytest.approx(alpha_static, rel=1e-8)


def test_coupled_alpha_monotone_and_positive():
    """Coupled α(iω) is strictly positive and monotone-decreasing —
    the same physical contract as the uncoupled provider."""
    from vibeqc import coupled_polarizability_imag_freq

    mol, basis, hf = _he_rhf()
    omegas = np.array([0.0, 0.2, 0.6, 1.5, 4.0])
    alpha = coupled_polarizability_imag_freq(hf, basis, mol, omegas)
    assert (alpha > 0).all()
    assert (np.diff(alpha) < 0).all(), f"not monotone: {alpha}"


def test_coupled_alpha_feeds_casimir_polder():
    """The coupled provider drops into the casimir_polder_c6 +
    imaginary_frequency_grid pipeline unchanged (the "swap only the
    α provider" design): a C6 comes out positive and finite."""
    from vibeqc import coupled_polarizability_imag_freq

    mol, basis, hf = _he_rhf()
    omegas, weights = imaginary_frequency_grid(n_points=23)
    alpha = coupled_polarizability_imag_freq(hf, basis, mol, omegas)
    c6 = casimir_polder_c6(alpha, alpha, weights)
    assert c6 > 0.0 and np.isfinite(c6)


# ---------------------------------------------------------------------
# molecular_c6 — end-to-end pipeline (Phase D4b-2a).
# ---------------------------------------------------------------------

def test_molecular_c6_homo_positive_and_finite():
    """C6(A,A) for He comes out positive and finite from the one-call
    pipeline (RHF → α(iω) → Casimir-Polder)."""
    from vibeqc.dispersion_d4_refdata import molecular_c6
    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])])
    basis = BasisSet(mol, "cc-pvdz")
    c6 = molecular_c6(mol, basis, n_freq=13)
    assert np.isfinite(c6) and c6 > 0.0


def test_molecular_c6_symmetric_in_its_arguments():
    """C6(A,B) == C6(B,A) — the Casimir-Polder integrand α_A·α_B is
    symmetric, so swapping the two molecules cannot change the
    coefficient."""
    from vibeqc.dispersion_d4_refdata import molecular_c6
    he = Molecule([Atom(2, [0.0, 0.0, 0.0])])
    he_b = BasisSet(he, "cc-pvdz")
    h2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    h2_b = BasisSet(h2, "cc-pvdz")
    c6_ab = molecular_c6(he, he_b, h2, h2_b, n_freq=13)
    c6_ba = molecular_c6(h2, h2_b, he, he_b, n_freq=13)
    assert c6_ab == pytest.approx(c6_ba, rel=1e-10)


def test_molecular_c6_coupled_exceeds_uncoupled():
    """The coupled-response C6 exceeds the uncoupled one — orbital
    relaxation raises α(iω) at every frequency, and C6 ∝ α²."""
    from vibeqc.dispersion_d4_refdata import molecular_c6
    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])])
    basis = BasisSet(mol, "cc-pvdz")
    c6_coupled = molecular_c6(mol, basis, n_freq=13, coupled=True)
    c6_uncoupled = molecular_c6(mol, basis, n_freq=13, coupled=False)
    assert 0.0 < c6_uncoupled < c6_coupled


def test_molecular_c6_matches_manual_pipeline():
    """molecular_c6 equals the hand-assembled grid + α + Casimir-Polder
    pipeline — it is exactly that pipeline in one call."""
    from vibeqc import coupled_polarizability_imag_freq
    from vibeqc.dispersion_d4_refdata import molecular_c6
    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])])
    basis = BasisSet(mol, "cc-pvdz")
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-10
    hf = run_rhf(mol, basis, opts)
    omegas, weights = imaginary_frequency_grid(n_points=13)
    alpha = coupled_polarizability_imag_freq(hf, basis, mol, omegas)
    c6_manual = casimir_polder_c6(alpha, alpha, weights)
    c6_driver = molecular_c6(mol, basis, n_freq=13)
    assert c6_driver == pytest.approx(c6_manual, rel=1e-9)


def test_molecular_c6_requires_basis_b_when_mol_b_given():
    """Passing a second molecule without its basis is a clear error."""
    from vibeqc.dispersion_d4_refdata import molecular_c6
    he = Molecule([Atom(2, [0.0, 0.0, 0.0])])
    he_b = BasisSet(he, "cc-pvdz")
    h2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
    with pytest.raises(ValueError, match="basis_b is required"):
        molecular_c6(he, he_b, h2, None, n_freq=7)
