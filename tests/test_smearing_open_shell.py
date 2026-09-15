"""Open-shell (per-spin μ) periodic occupations — ``apply_smearing_open_shell``.

The UHF/UKS sibling of the closed-shell ``apply_smearing``: each spin
channel solves the same μ-bisection with ``spin_degeneracy = 1`` and its
own particle-count constraint, giving independent μ_α, μ_β (the CP2K /
VASP / QE spin-polarized convention). These tests pin:

* **closed-shell consistency** — equal channels reproduce the closed-shell
  occupations / μ / entropy exactly (occ_α + occ_β = occ_closed);
* **per-spin particle conservation** — Σ_k w_k Σ_i n^σ_i(k) = n_σ;
* **independent μ_σ** — a uniform β band shift moves only μ_β (by the
  shift), occupations stay in ``[0, 1]``;
* **T = 0 behavior-neutrality** — falls back to per-spin hard Aufbau.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.smearing import (
    SmearingOptions,
    apply_smearing_open_shell,
    closed_shell_periodic_occupations,
)


def _bands():
    # Two k-points, a band grazing the Fermi level so smearing is fractional.
    eps = [
        np.array([-1.00, -0.50, -0.45, 0.50]),
        np.array([-0.90, -0.55, 0.10, 0.60]),
    ]
    weights = [0.5, 0.5]
    return eps, weights


def test_open_shell_equal_channels_reproduce_closed_shell():
    """eps_α = eps_β, n_α = n_β = N/2 ⇒ occ_α + occ_β = occ_closed, one μ."""
    eps, w = _bands()
    T = 0.02
    n_occ = 2  # per spin / per k doubly-occ count
    a, b = apply_smearing_open_shell(
        eps, eps, weights=w, n_alpha=n_occ, n_beta=n_occ,
        smearing=SmearingOptions(temperature=T),
    )
    occ_c, mu_c, S_c = closed_shell_periodic_occupations(
        eps, w, n_electrons_per_cell=2.0 * n_occ, n_occ_each=n_occ,
        smearing_temperature=T,
    )
    # Per-spin occ in [0, 1]; the two channels are identical here.
    for oa, ob in zip(a.occupations_per_k, b.occupations_per_k):
        assert np.all(oa >= -1e-12) and np.all(oa <= 1.0 + 1e-12)
        assert np.allclose(oa, ob)
    # Sum of the two spin channels reproduces the closed-shell occupations.
    for oa, ob, oc in zip(a.occupations_per_k, b.occupations_per_k, occ_c):
        assert np.allclose(oa + ob, oc, atol=1e-9)
    assert a.mu == pytest.approx(mu_c, abs=1e-9)
    assert b.mu == pytest.approx(mu_c, abs=1e-9)
    # Entropy is additive across spins and matches the closed-shell value.
    assert a.entropy + b.entropy == pytest.approx(S_c, abs=1e-9)


def test_open_shell_per_spin_particle_conservation():
    """Σ_k w_k Σ_i n^σ_i = n_σ for each spin, with n_α ≠ n_β."""
    eps, w = _bands()
    eps_beta = [e + 0.05 for e in eps]  # mildly spin-split
    n_alpha, n_beta = 2.0, 1.0
    a, b = apply_smearing_open_shell(
        eps, eps_beta, weights=w, n_alpha=n_alpha, n_beta=n_beta,
        smearing=0.03,  # bare float => Fermi-Dirac
    )
    na = sum(wk * oa.sum() for wk, oa in zip(w, a.occupations_per_k))
    nb = sum(wk * ob.sum() for wk, ob in zip(w, b.occupations_per_k))
    assert na == pytest.approx(n_alpha, abs=1e-9)
    assert nb == pytest.approx(n_beta, abs=1e-9)
    # μ_α ≠ μ_β when the channels carry different particle counts.
    assert abs(a.mu - b.mu) > 1e-3


def test_open_shell_independent_mu_uniform_beta_shift():
    """A uniform β-band shift Δ moves μ_β by exactly Δ, μ_α unchanged."""
    eps, w = _bands()
    delta = 0.30
    eps_beta = [e + delta for e in eps]
    T = 0.02
    a0, b0 = apply_smearing_open_shell(
        eps, eps, weights=w, n_alpha=2, n_beta=2,
        smearing=SmearingOptions(temperature=T),
    )
    a1, b1 = apply_smearing_open_shell(
        eps, eps_beta, weights=w, n_alpha=2, n_beta=2,
        smearing=SmearingOptions(temperature=T),
    )
    # α channel identical (its bands didn't move); β μ shifts by Δ.
    assert a1.mu == pytest.approx(a0.mu, abs=1e-9)
    assert b1.mu == pytest.approx(b0.mu + delta, abs=1e-9)
    # Occupations stay physical.
    for ob in b1.occupations_per_k:
        assert np.all(ob >= -1e-12) and np.all(ob <= 1.0 + 1e-12)


def test_open_shell_empty_and_full_channels_with_smearing():
    """Fully spin-polarized determinant under smearing: one channel empty
    (n=0), the other full (n=capacity). The μ-bisection cannot bracket these
    endpoints (μ→∓∞); they must return integer occupations, zero entropy."""
    eps, w = _bands()  # 4 bands per k ⇒ per-spin capacity = 4
    a, b = apply_smearing_open_shell(
        eps, eps, weights=w, n_alpha=4, n_beta=0, smearing=0.02,
    )
    for oa in a.occupations_per_k:
        assert np.allclose(oa, 1.0)
    for ob in b.occupations_per_k:
        assert np.allclose(ob, 0.0)
    assert a.entropy == 0.0 and b.entropy == 0.0
    # Particle counts still exact at the endpoints.
    assert sum(wk * oa.sum() for wk, oa in zip(w, a.occupations_per_k)) == pytest.approx(4.0)
    assert sum(wk * ob.sum() for wk, ob in zip(w, b.occupations_per_k)) == pytest.approx(0.0)


def test_open_shell_t0_is_per_spin_aufbau():
    """T = 0 (smearing disabled) ⇒ hard per-spin Aufbau, zero entropy."""
    eps, w = _bands()
    n_alpha, n_beta = 3, 1
    a, b = apply_smearing_open_shell(
        eps, eps, weights=w, n_alpha=n_alpha, n_beta=n_beta, smearing=None,
    )
    for oa in a.occupations_per_k:
        assert np.allclose(oa[:n_alpha], 1.0)
        assert np.allclose(oa[n_alpha:], 0.0)
    for ob in b.occupations_per_k:
        assert np.allclose(ob[:n_beta], 1.0)
        assert np.allclose(ob[n_beta:], 0.0)
    assert a.entropy == 0.0 and b.entropy == 0.0
    assert a.free_energy_correction == 0.0 and b.free_energy_correction == 0.0
