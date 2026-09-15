"""Unit tests for the general Green's-function / GF2 layer (vibeqc.propagator).

These pin the reference-agnostic diagonal second-order self-energy: the
vectorized kernel vs an explicit O(N⁶) loop, the analytic pole-strength
derivative vs finite difference, and the Dyson self-consistency of the
quasiparticle solver.  End-to-end parity (MSINDO INDO IPs) lives in
test_msindo.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.propagator import diagonal_self_energy, quasiparticle_energy


def _loop_sigma(eps, n_occ, eri_mo, p, omega):
    """Independent explicit-loop second-order self-energy (Szabo-Ostlund)."""
    n = len(eps)
    occ, vir = range(n_occ), range(n_occ, n)
    s = 0.0
    for a in vir:                      # 2h1p
        for i in occ:
            for j in occ:
                v = eri_mo[p, i, a, j]
                x = eri_mo[p, j, a, i]
                s += (2 * v - x) * v / (omega + eps[a] - eps[i] - eps[j])
    for i in occ:                      # 2p1h
        for a in vir:
            for b in vir:
                v = eri_mo[p, a, i, b]
                x = eri_mo[p, b, i, a]
                s += (2 * v - x) * v / (omega + eps[i] - eps[a] - eps[b])
    return s


def _random_eri(n, seed=0):
    """A deterministic MO ERI tensor with chemist's 8-fold permutation symmetry."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n, n, n, n)) * 0.1
    # Symmetrize: (pq|rs)=(qp|rs)=(pq|sr)=(rs|pq).
    g = 0.5 * (g + np.transpose(g, (1, 0, 2, 3)))
    g = 0.5 * (g + np.transpose(g, (0, 1, 3, 2)))
    g = 0.5 * (g + np.transpose(g, (2, 3, 0, 1)))
    return g


def _toy_system(n=6, n_occ=3, seed=0):
    # Non-commensurate energies so no second-order denominator lands exactly on
    # a pole (real outer-valence quasiparticles never sit on a resonance).
    eps = np.sort(np.array([-0.93, -0.61, -0.42, 0.31, 0.67, 1.09]))
    eri = _random_eri(n, seed)
    return eps, n_occ, eri


def test_self_energy_vectorized_matches_loop():
    """The vectorized Σ_pp(ω) reproduces the explicit O(N⁶) loop for every
    orbital and a few frequencies (validates the slice transposes + denominators)."""
    eps, no, eri = _toy_system()
    for p in range(len(eps)):
        for omega in (eps[p], eps[p] + 0.037, eps[p] - 0.029):
            got = diagonal_self_energy(eps, no, eri, p, omega)
            ref = _loop_sigma(eps, no, eri, p, omega)
            assert got == pytest.approx(ref, rel=1e-11, abs=1e-13)


def test_pole_strength_derivative_matches_finite_difference():
    """The analytic ∂Σ/∂ω behind the pole strength matches a central difference."""
    eps, no, eri = _toy_system(seed=2)
    p = no - 1  # HOMO
    r = quasiparticle_energy(eps, no, eri, p, iterate=False)
    # Recover Σ'(ε_p) from the reported pole strength Γ = 1/(1−Σ').
    sigma_prime = 1.0 - 1.0 / r.pole_strength
    h = 1e-5
    fd = (diagonal_self_energy(eps, no, eri, p, eps[p] + h)
          - diagonal_self_energy(eps, no, eri, p, eps[p] - h)) / (2 * h)
    assert sigma_prime == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_quasiparticle_satisfies_dyson_equation():
    """The iterative solver returns ω with ω = ε_p + Σ_pp(ω) (the Dyson pole)."""
    eps, no, eri = _toy_system(seed=3)
    for p in (no - 1, no):  # HOMO, LUMO
        r = quasiparticle_energy(eps, no, eri, p, iterate=True)
        assert r.converged
        sigma = diagonal_self_energy(eps, no, eri, p, r.eps_qp)
        assert r.eps_qp == pytest.approx(eps[p] + sigma, abs=1e-7)
        # Pole strength is a physical renormalization in (0, 1].
        assert 0.0 < r.pole_strength <= 1.0 + 1e-12


def test_noninteracting_limit_has_zero_self_energy():
    """With zero two-electron integrals the self-energy vanishes and the
    quasiparticle energy collapses to Koopmans (ε_qp = ε_p, Γ = 1)."""
    eps, no, _ = _toy_system()
    zero = np.zeros((len(eps),) * 4)
    r = quasiparticle_energy(eps, no, zero, no - 1, iterate=True)
    assert r.sigma == pytest.approx(0.0, abs=1e-14)
    assert r.eps_qp == pytest.approx(eps[no - 1], abs=1e-14)
    assert r.pole_strength == pytest.approx(1.0, abs=1e-14)
    assert r.correction == pytest.approx(0.0, abs=1e-14)


def test_gf2_works_on_hf_reference():
    """The SAME kernel computes HF (Gaussian-basis) quasiparticle IPs — proving
    the Green's-function layer is reference-agnostic (HF here, MSINDO in
    test_msindo).  H₂O/STO-3G; the HOMO IP correction is a sensible few-eV shift
    on Koopmans, pole strength is a strong quasiparticle.  Pinned as a
    regression anchor for the HF path."""
    from vibeqc._vibeqc_core import (Atom, BasisSet, Molecule, RHFOptions,
                                     compute_eri, run_rhf)
    from vibeqc.propagator import quasiparticle_energies

    A2B = 1.8897259886
    mol = Molecule([Atom(8, [0, 0, 0]),
                    Atom(1, [0, 0.7572 * A2B, 0.5865 * A2B]),
                    Atom(1, [0, -0.7572 * A2B, 0.5865 * A2B])], 0, 1)
    bas = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, bas, RHFOptions())
    C = np.asarray(rhf.mo_coeffs)
    eps = np.asarray(rhf.mo_energies)
    nocc = 5  # H2O: 10 electrons
    eri_ao = np.asarray(compute_eri(bas))            # (pq|rs) AO
    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)

    res = quasiparticle_energies(eps, nocc, eri_mo, [nocc - 1, nocc], iterate=True)
    homo, lumo = res
    assert homo.converged and lumo.converged
    EV = 27.211386245988
    # Koopmans HOMO IP of H2O/STO-3G is ~10.6 eV (ε_HOMO ≈ −0.39 Ha); GF2
    # shifts it by a few eV.
    assert -homo.eps_scf * EV == pytest.approx(10.6, abs=1.0)  # Koopmans IP ballpark
    assert abs(homo.correction * EV) < 5.0                     # GF2 correction (eV)
    assert 0.85 < homo.pole_strength <= 1.0 + 1e-12            # strong quasiparticle
    # The Dyson solution is self-consistent.
    from vibeqc.propagator import diagonal_self_energy
    assert homo.eps_qp == pytest.approx(
        eps[nocc - 1] + diagonal_self_energy(eps, nocc, eri_mo, nocc - 1, homo.eps_qp),
        abs=1e-7)


# --------------------------------------------------------------------------- #
# Open-shell (spin-unrestricted) propagator — unrestricted_quasiparticle_*     #
# --------------------------------------------------------------------------- #


def _h2o_rhf():
    from vibeqc._vibeqc_core import (Atom, BasisSet, Molecule, RHFOptions,
                                     compute_eri, run_rhf)
    A2B = 1.8897259886
    mol = Molecule([Atom(8, [0, 0, 0]),
                    Atom(1, [0, 0.7572 * A2B, 0.5865 * A2B]),
                    Atom(1, [0, -0.7572 * A2B, 0.5865 * A2B])], 0, 1)
    bas = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, bas, RHFOptions())
    return (np.asarray(compute_eri(bas)), np.asarray(rhf.mo_coeffs),
            np.asarray(rhf.mo_energies))


def _oh_uhf():
    from vibeqc._vibeqc_core import (Atom, BasisSet, Molecule, UHFOptions,
                                     compute_eri, run_uhf)
    m = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.834])], 0, 2)  # doublet
    bas = BasisSet(m, "sto-3g")
    u = run_uhf(m, bas, UHFOptions())
    return (np.asarray(compute_eri(bas)),
            np.asarray(u.mo_coeffs_alpha), np.asarray(u.mo_coeffs_beta),
            np.asarray(u.mo_energies_alpha), np.asarray(u.mo_energies_beta))


def test_unrestricted_reduces_to_closed_shell():
    """THE load-bearing check: a closed-shell (RHF) reference fed through the
    open-shell spin-orbital path reproduces the spatial closed-shell kernel
    term-for-term (ε_qp + pole strength) for the α-HOMO and α-LUMO."""
    from vibeqc.propagator import (quasiparticle_energies,
                                   unrestricted_quasiparticle_energies)
    eri_ao, C, eps = _h2o_rhf()
    nocc = 5
    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)
    spat = quasiparticle_energies(eps, nocc, eri_mo, [nocc - 1, nocc], iterate=True)
    op = unrestricted_quasiparticle_energies(
        eri_ao, C, C, eps, eps, nocc, nocc,
        [("alpha", nocc - 1), ("alpha", nocc)], iterate=True)
    for sp_res, (spin, o_res) in zip(spat, op):
        assert spin == "alpha"
        assert o_res.eps_qp == pytest.approx(sp_res.eps_qp, abs=1e-11)
        assert o_res.pole_strength == pytest.approx(sp_res.pole_strength, abs=1e-11)
        assert o_res.sigma == pytest.approx(sp_res.sigma, abs=1e-11)


def test_unrestricted_self_energy_matches_loop():
    """Vectorized spin-orbital Σ matches an independent explicit O(N⁶) loop over
    the antisymmetrized spin-orbital integrals (UHF OH reference); checked for
    the α-SOMO and the β-HOMO."""
    from vibeqc.propagator import _so_sigma_and_deriv, _spinorbital_aeri
    eri_ao, Ca, Cb, ea, eb = _oh_uhf()
    na, nb = 5, 4
    A = _spinorbital_aeri(eri_ao, Ca, Cb)
    eps_so = np.concatenate([ea, eb])
    n = Ca.shape[1]
    occ = np.zeros(2 * n, bool)
    occ[:na] = True
    occ[n:n + nb] = True
    o = [i for i in range(2 * n) if occ[i]]
    v = [i for i in range(2 * n) if not occ[i]]
    for p in (na - 1, n + nb - 1):                     # α-SOMO, β-HOMO
        w = eps_so[p] + 0.013                          # off the pole
        s = 0.0
        for a in v:
            for i in o:
                for j in o:
                    s += 0.5 * A[p, a, i, j] ** 2 / (w + eps_so[a] - eps_so[i]
                                                     - eps_so[j])
        for i in o:
            for a in v:
                for b in v:
                    s += 0.5 * A[p, i, a, b] ** 2 / (w + eps_so[i] - eps_so[a]
                                                     - eps_so[b])
        vec, _ = _so_sigma_and_deriv(eps_so, occ, A, p, w)
        assert vec == pytest.approx(s, rel=1e-11, abs=1e-13)


def test_unrestricted_open_shell_quasiparticles_are_physical():
    """Open-shell UHF quasiparticles (OH/STO-3G): the α-SOMO IP corrects the
    Koopmans value downward (GF2 relaxation), the Dyson roots are strong
    quasiparticles, and the returned spin labels are correct."""
    from vibeqc.propagator import unrestricted_quasiparticle_energies
    eri_ao, Ca, Cb, ea, eb = _oh_uhf()
    na, nb = 5, 4
    res = unrestricted_quasiparticle_energies(
        eri_ao, Ca, Cb, ea, eb, na, nb,
        [("alpha", na - 1), ("beta", nb - 1)], iterate=True)
    EV = 27.211386245988
    (sa, qa), (sb, qb) = res
    assert (sa, sb) == ("alpha", "beta")
    assert qa.converged and qb.converged
    # GF2 lowers the IP off Koopmans (relaxation+correlation), pole strong.
    assert -qa.eps_qp * EV < -qa.eps_scf * EV          # IP corrected downward
    assert 0.85 < qa.pole_strength <= 1.0 + 1e-12

    # Spin-resolution is real: with different α/β orbitals the two channels
    # give genuinely different quasiparticle energies.
    assert abs(qa.eps_qp - qb.eps_qp) > 1e-3


def test_unrestricted_guards_large_systems():
    """The (2N)⁴ spin-orbital build is guarded; an over-large request fails fast
    with a clear message rather than attempting the transform."""
    from vibeqc.propagator import unrestricted_quasiparticle_energies
    n = 60                                              # 2N = 120 > 80
    eri = np.zeros((n, n, n, n))
    C = np.eye(n)
    eps = np.arange(n, dtype=float)
    with pytest.raises(NotImplementedError, match="spin-orbital"):
        unrestricted_quasiparticle_energies(
            eri, C, C, eps, eps, n // 2, n // 2, [("alpha", 0)],
            max_spinorbitals=80)


# --------------------------------------------------------------------------- #
# Renormalized GF2 — full third-order self-energy + geometric screening        #
# --------------------------------------------------------------------------- #


def _loop_sigma3(g, e, occ, p, w):
    """Definitional diagonal third-order self-energy (2h1p + 2p1h), explicit
    loops — the independent correctness oracle for the vectorized kernel.
    ``g[p,q,r,s] = <pq||rs>`` antisymmetrized."""
    n = len(e)
    O = [x for x in range(n) if occ[x]]
    V = [x for x in range(n) if not occ[x]]
    sh = 0.0
    for a in V:
        for i in O:
            for j in O:
                W = U = 0.0
                for b in V:
                    for c in V:
                        W += 0.5 * g[b, c, p, a] * g[i, j, b, c] / (
                            e[i] + e[j] - e[b] - e[c])
                for b in V:
                    for k in O:
                        W += g[b, i, p, k] * g[j, k, b, a] / (
                            e[j] + e[k] - e[a] - e[b])
                        W -= g[b, j, p, k] * g[i, k, b, a] / (
                            e[i] + e[k] - e[a] - e[b])
                for k in O:
                    for ll in O:
                        U -= 0.5 * g[p, a, k, ll] * g[k, ll, i, j] / (
                            w + e[a] - e[k] - e[ll])
                for b in V:
                    for k in O:
                        U -= g[p, b, j, k] * g[a, k, b, i] / (
                            w + e[b] - e[j] - e[k])
                        U += g[p, b, i, k] * g[a, k, b, j] / (
                            w + e[b] - e[i] - e[k])
                sh += 0.5 * g[p, a, i, j] * (W + U) / (w + e[a] - e[i] - e[j])
    sp = 0.0
    for i in O:
        for a in V:
            for b in V:
                W = U = 0.0
                for j in O:
                    for k in O:
                        W += 0.5 * g[j, k, p, i] * g[a, b, j, k] / (
                            e[a] + e[b] - e[j] - e[k])
                for j in O:
                    for c in V:
                        W += g[j, a, p, c] * g[b, c, j, i] / (
                            e[b] + e[c] - e[i] - e[j])
                        W -= g[j, b, p, c] * g[a, c, j, i] / (
                            e[a] + e[c] - e[i] - e[j])
                for c in V:
                    for d in V:
                        U -= 0.5 * g[p, i, c, d] * g[c, d, a, b] / (
                            w + e[i] - e[c] - e[d])
                for j in O:
                    for c in V:
                        U -= g[p, j, b, c] * g[i, c, j, a] / (
                            w + e[j] - e[b] - e[c])
                        U += g[p, j, a, c] * g[i, c, j, b] / (
                            w + e[j] - e[a] - e[c])
                sp += 0.5 * g[p, i, a, b] * (W + U) / (w + e[i] - e[a] - e[b])
    return sh, sp


def _random_aeri(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, n, n, n)) * 0.1
    v = 0.5 * (v + np.transpose(v, (2, 3, 0, 1)))
    v = 0.5 * (v + np.transpose(v, (1, 0, 3, 2)))
    return v - np.transpose(v, (0, 1, 3, 2))


def test_third_order_self_energy_matches_loop():
    """The vectorized diagonal third-order self-energy (both channels) reproduces
    the explicit definitional loop — the load-bearing check that the W/U ladder +
    ring contractions and their antisymmetrizers are correct."""
    from vibeqc.propagator import _sigma3_channels
    n, no = 8, 3
    e = np.sort(np.array([-0.9, -0.6, -0.4, 0.3, 0.6, 1.1, 1.5, 1.9]))
    occ = np.zeros(n, bool)
    occ[:no] = True
    for seed in (0, 1, 2):
        g = _random_aeri(n, seed)
        for p in range(n):
            w = e[p] + 0.05
            ref = _loop_sigma3(g, e, occ, p, w)
            got = _sigma3_channels(g, e, occ, p, w)
            assert got[0] == pytest.approx(ref[0], rel=1e-10, abs=1e-13)
            assert got[1] == pytest.approx(ref[1], rel=1e-10, abs=1e-13)


def test_renormalized_reduces_gf2_overcorrection():
    """Renormalized GF2 lifts the H₂O HOMO IP back from the bare-GF2
    overcorrection toward experiment (Koopmans 13.6 → GF2 ~10.8 overshoots exp
    12.62 → renorm ~11.6): the renormalized IP sits between bare GF2 and
    Koopmans, closer to experiment, with a still-strong pole strength."""
    from vibeqc._vibeqc_core import (Atom, BasisSet, Molecule, RHFOptions,
                                     compute_eri, run_rhf)
    from vibeqc.propagator import (quasiparticle_energies,
                                   renormalized_quasiparticle_energies)
    A2B = 1.8897259886
    mol = Molecule([Atom(8, [0, 0, 0]),
                    Atom(1, [0, 0.7572 * A2B, 0.5865 * A2B]),
                    Atom(1, [0, -0.7572 * A2B, 0.5865 * A2B])], 0, 1)
    bas = BasisSet(mol, "6-31g")
    rhf = run_rhf(mol, bas, RHFOptions())
    C = np.asarray(rhf.mo_coeffs)
    eps = np.asarray(rhf.mo_energies)
    eri_ao = np.asarray(compute_eri(bas))
    nocc = 5
    EV = 27.211386245988

    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri_ao, C, C, C, C, optimize=True)
    gf2 = quasiparticle_energies(eps, nocc, eri_mo, [nocc - 1], iterate=True)[0]
    ren = renormalized_quasiparticle_energies(
        eri_ao, C, C, eps, eps, nocc, nocc, [("alpha", nocc - 1)])[0][1]

    koop = -eps[nocc - 1] * EV
    gf2_ip = -gf2.eps_qp * EV
    ren_ip = -ren.eps_qp * EV
    exp = 12.62
    assert gf2_ip < ren_ip < koop                       # renorm pulls back up
    assert abs(ren_ip - exp) < abs(gf2_ip - exp)        # closer to experiment
    assert 0.80 < ren.pole_strength <= 1.0 + 1e-12      # still a quasiparticle


def test_renormalized_self_energy_derivative_is_analytic():
    """The analytic ∂Σ_renorm/∂ω (chain rule through the third-order self-energy
    and the per-channel geometric screening) matches a central difference — the
    derivative that sets the renormalized pole strength."""
    from vibeqc.propagator import (_renormalized_sigma_and_deriv,
                                   _sigma3_channels_and_deriv)
    n, no = 8, 3
    e = np.sort(np.array([-0.9, -0.6, -0.4, 0.3, 0.6, 1.1, 1.5, 1.9]))
    occ = np.zeros(n, bool)
    occ[:no] = True
    h = 1e-6
    for seed in (0, 1, 2):
        g = _random_aeri(n, seed)
        for p in range(n):
            w = e[p] + 0.05
            # third-order channel derivatives vs FD
            _, _, d3h, d3p = _sigma3_channels_and_deriv(g, e, occ, p, w)
            s3h_p, s3p_p, _, _ = _sigma3_channels_and_deriv(g, e, occ, p, w + h)
            s3h_m, s3p_m, _, _ = _sigma3_channels_and_deriv(g, e, occ, p, w - h)
            assert d3h == pytest.approx((s3h_p - s3h_m) / (2 * h), abs=1e-6)
            assert d3p == pytest.approx((s3p_p - s3p_m) / (2 * h), abs=1e-6)
            # full renormalized derivative vs FD
            _, dana = _renormalized_sigma_and_deriv(g, e, occ, p, w)
            sp, _ = _renormalized_sigma_and_deriv(g, e, occ, p, w + h)
            sm, _ = _renormalized_sigma_and_deriv(g, e, occ, p, w - h)
            assert dana == pytest.approx((sp - sm) / (2 * h), abs=1e-6)
