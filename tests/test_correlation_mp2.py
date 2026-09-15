"""Unit tests for the general MP2 correlation kernel (vibeqc.correlation).

These pin the reference-agnostic kernel arithmetic (spin decomposition,
denominators, SCS/SOS scaling).  End-to-end parity (HF vs the C++ MP2; MSINDO vs
the oracle) is added with the ERIProvider backends — see HANDOVER_MSINDO_ADVANCED.
"""

import numpy as np
import pytest

from vibeqc.correlation import MP2Result, UMP2Result, mp2_energy, ump2_energy


def _loop_mp2(eps_o, eps_v, ovov):
    """Independent explicit-loop reference for E_OS, E_SS."""
    no, nv = len(eps_o), len(eps_v)
    e_os = e_ss = 0.0
    for i in range(no):
        for a in range(nv):
            for j in range(no):
                for b in range(nv):
                    d = eps_o[i] + eps_o[j] - eps_v[a] - eps_v[b]
                    g = ovov[i, a, j, b]
                    gx = ovov[i, b, j, a]   # (ib|ja)
                    e_os += g * g / d
                    e_ss += g * (g - gx) / d
    return e_os, e_ss


def _random_symmetric_ovov(no, nv, seed=0):
    """A deterministic (ia|jb) block with the physical (ia|jb)=(jb|ia) symmetry."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((no, nv, no, nv))
    return 0.5 * (g + np.transpose(g, (2, 3, 0, 1)))


def test_mp2_analytic_one_occ_one_vir():
    """1-occ/1-vir: E_OS = g²/(2εi−2εa), E_SS = 0, E_MP2 = E_OS."""
    eps_o, eps_v = np.array([-0.5]), np.array([0.5])
    ovov = np.array([[[[0.1]]]])           # (ia|jb) = 0.1
    r = mp2_energy(eps_o, eps_v, ovov)
    assert r.e_os == pytest.approx(0.01 / (-2.0))   # 0.1² / (−1.0 − 1.0)
    assert r.e_ss == pytest.approx(0.0, abs=1e-15)
    assert r.e_corr == pytest.approx(r.e_os)


def test_mp2_vectorized_matches_explicit_loop():
    """The vectorized kernel reproduces the explicit O(n⁴) loop (validates the
    (ib|ja) transpose + denominator broadcasting)."""
    no, nv = 3, 4
    eps_o = np.array([-0.9, -0.6, -0.4])
    eps_v = np.array([0.2, 0.5, 0.8, 1.1])
    ovov = _random_symmetric_ovov(no, nv)
    e_os_ref, e_ss_ref = _loop_mp2(eps_o, eps_v, ovov)
    r = mp2_energy(eps_o, eps_v, ovov)
    assert r.e_os == pytest.approx(e_os_ref, rel=1e-12)
    assert r.e_ss == pytest.approx(e_ss_ref, rel=1e-12)
    # Total RHF-MP2 = E_OS + E_SS.
    assert r.e_corr == pytest.approx(e_os_ref + e_ss_ref, rel=1e-12)
    # Correlation energy is stabilizing (negative) for a real gap.
    assert r.e_corr < 0.0


def test_mp2_scs_and_sos_scaling():
    """SCS-MP2 = 6/5·E_OS + 1/3·E_SS; SOS-MP2 = 1.3·E_OS (no same-spin)."""
    no, nv = 2, 3
    eps_o = np.array([-0.8, -0.5])
    eps_v = np.array([0.3, 0.6, 0.9])
    ovov = _random_symmetric_ovov(no, nv, seed=1)
    plain = mp2_energy(eps_o, eps_v, ovov, variant="mp2")
    scs = mp2_energy(eps_o, eps_v, ovov, variant="scs-mp2")
    sos = mp2_energy(eps_o, eps_v, ovov, variant="sos-mp2")
    # Components are identical across variants (only the scaling differs).
    assert scs.e_os == pytest.approx(plain.e_os)
    assert scs.e_ss == pytest.approx(plain.e_ss)
    assert scs.e_corr == pytest.approx(6.0 / 5.0 * plain.e_os + 1.0 / 3.0 * plain.e_ss)
    assert sos.e_corr == pytest.approx(1.3 * plain.e_os)
    assert sos.e_ss_scaled == pytest.approx(0.0)


def test_mp2_unknown_variant_raises():
    with pytest.raises(ValueError, match="variant"):
        mp2_energy([-0.5], [0.5], np.array([[[[0.1]]]]), variant="ccsd")


def test_mp2_general_kernel_matches_cpp_mp2_hf_reference():
    """End-to-end: the general kernel reproduces vibe-qc's C++ RHF-MP2
    (e_corr/e_os/e_ss) on H₂O/STO-3G — validates the formula against the
    reference implementation, and that the same kernel will give SCS-MP2 (new)
    and drive MSINDO-MP2 via the semiempirical ERIProvider."""
    from vibeqc._vibeqc_core import (Atom, Molecule, BasisSet, run_rhf, run_mp2,
                                     RHFOptions, MP2Options, compute_eri)
    A2B = 1.8897259886
    mol = Molecule([Atom(8, [0, 0, 0]),
                    Atom(1, [0, 0.7572 * A2B, 0.5865 * A2B]),
                    Atom(1, [0, -0.7572 * A2B, 0.5865 * A2B])], 0, 1)
    bas = BasisSet(mol, "sto-3g")
    rhf = run_rhf(mol, bas, RHFOptions())
    options = MP2Options()
    options.n_frozen_core = 0
    ref = run_mp2(mol, bas, rhf, options)
    C = np.asarray(rhf.mo_coeffs)
    eps = np.asarray(rhf.mo_energies)
    nocc = 5  # H2O: 10 electrons
    Co, Cv = C[:, :nocc], C[:, nocc:]
    eri = np.asarray(compute_eri(bas))               # AO (pq|rs)
    ovov = np.einsum("pqrs,pi,qa,rj,sb->iajb", eri, Co, Cv, Co, Cv, optimize=True)
    r = mp2_energy(eps[:nocc], eps[nocc:], ovov)
    assert r.e_corr == pytest.approx(ref.e_correlation, abs=1e-9)
    assert r.e_os == pytest.approx(ref.e_os, abs=1e-9)
    assert r.e_ss == pytest.approx(ref.e_ss, abs=1e-9)
    # SCS-MP2 (a new general capability) = 6/5 OS + 1/3 SS.
    scs = mp2_energy(eps[:nocc], eps[nocc:], ovov, variant="scs-mp2")
    assert scs.e_corr == pytest.approx(1.2 * ref.e_os + ref.e_ss / 3.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Unrestricted MP2 kernel (ump2_energy) — spin-channel αα / ββ / αβ.            #
# --------------------------------------------------------------------------- #

def _loop_ump2_same_spin(eps_o, eps_v, ovov):
    """¼ Σ_{ijab} [(ia|jb)−(ib|ja)]² / D — explicit-loop reference."""
    no, nv = len(eps_o), len(eps_v)
    e = 0.0
    for i in range(no):
        for a in range(nv):
            for j in range(no):
                for b in range(nv):
                    d = eps_o[i] + eps_o[j] - eps_v[a] - eps_v[b]
                    anti = ovov[i, a, j, b] - ovov[i, b, j, a]
                    e += 0.25 * anti * anti / d
    return e


def _loop_ump2_opp(eps_oa, eps_va, eps_ob, eps_vb, ovov):
    """Σ_{ia∈α, jb∈β} (ia|jb)² / D — explicit-loop reference."""
    e = 0.0
    for i in range(len(eps_oa)):
        for a in range(len(eps_va)):
            for j in range(len(eps_ob)):
                for b in range(len(eps_vb)):
                    d = eps_oa[i] + eps_ob[j] - eps_va[a] - eps_vb[b]
                    g = ovov[i, a, j, b]
                    e += g * g / d
    return e


def test_ump2_vectorized_matches_explicit_loop():
    """The vectorized spin-channel kernel reproduces the explicit O(n⁴) loops for
    αα, ββ and αβ (validates the antisymmetriser transpose + mixed-spin
    denominator broadcasting)."""
    rng = np.random.default_rng(7)
    eps_oa = np.array([-0.95, -0.6, -0.45]); eps_va = np.array([0.25, 0.7])
    eps_ob = np.array([-0.9, -0.55]); eps_vb = np.array([0.3, 0.65, 0.95])
    aa = _random_symmetric_ovov(3, 2, seed=2)
    bb = _random_symmetric_ovov(2, 3, seed=3)
    ab = rng.standard_normal((3, 2, 2, 3))          # no αβ symmetry required
    r = ump2_energy(eps_oa, eps_va, eps_ob, eps_vb, aa, bb, ab)
    assert r.e_aa == pytest.approx(_loop_ump2_same_spin(eps_oa, eps_va, aa), rel=1e-12)
    assert r.e_bb == pytest.approx(_loop_ump2_same_spin(eps_ob, eps_vb, bb), rel=1e-12)
    assert r.e_ab == pytest.approx(
        _loop_ump2_opp(eps_oa, eps_va, eps_ob, eps_vb, ab), rel=1e-12)
    assert r.e_corr == pytest.approx(r.e_aa + r.e_bb + r.e_ab, rel=1e-12)


def test_ump2_reduces_to_rmp2_when_spins_coincide():
    """With identical α/β references (same eps, same ovov for all three blocks),
    ump2_energy reproduces the closed-shell mp2_energy total — the UHF→RHF
    reduction (cpp/src/ump2.cpp: 'reduces to RMP2 when D_α = D_β')."""
    eps_o = np.array([-0.8, -0.5]); eps_v = np.array([0.3, 0.6, 0.9])
    ovov = _random_symmetric_ovov(2, 3, seed=5)
    rmp2 = mp2_energy(eps_o, eps_v, ovov)
    ump2 = ump2_energy(eps_o, eps_v, eps_o, eps_v, ovov, ovov, ovov)
    assert ump2.e_aa == pytest.approx(ump2.e_bb, rel=1e-12)   # symmetric spins
    assert ump2.e_corr == pytest.approx(rmp2.e_corr, rel=1e-12)


def test_ump2_scs_sos_scaling():
    """UMP2 spin-component scaling: e_corr = c_os·E_αβ + c_ss·(E_αα+E_ββ)."""
    eps_oa = np.array([-0.8, -0.5]); eps_va = np.array([0.3, 0.7])
    eps_ob = np.array([-0.75]); eps_vb = np.array([0.4, 0.8])
    aa = _random_symmetric_ovov(2, 2, seed=8)
    bb = _random_symmetric_ovov(1, 2, seed=9)
    ab = np.random.default_rng(10).standard_normal((2, 2, 1, 2))
    plain = ump2_energy(eps_oa, eps_va, eps_ob, eps_vb, aa, bb, ab, variant="mp2")
    scs = ump2_energy(eps_oa, eps_va, eps_ob, eps_vb, aa, bb, ab, variant="scs-mp2")
    sos = ump2_energy(eps_oa, eps_va, eps_ob, eps_vb, aa, bb, ab, variant="sos-mp2")
    assert scs.e_corr == pytest.approx(
        1.2 * plain.e_ab + (plain.e_aa + plain.e_bb) / 3.0)
    assert sos.e_corr == pytest.approx(1.3 * plain.e_ab)      # no same-spin
    assert plain.e_corr == pytest.approx(plain.e_aa + plain.e_bb + plain.e_ab)


def test_ump2_empty_channel_is_zero():
    """A spin channel with no virtual pair contributes 0 — αα vanishes for a
    single α-virtual; ββ/αβ vanish when nβ=0 (empty β block)."""
    eps_oa = np.array([-0.6]); eps_va = np.array([0.5])       # 1 occ / 1 vir → no αα
    aa = _random_symmetric_ovov(1, 1, seed=1)
    empty = np.zeros((0, 0, 0, 0))
    r = ump2_energy(eps_oa, eps_va, np.array([]), np.array([]), aa, empty, empty)
    assert r.e_aa == pytest.approx(0.0, abs=1e-15)   # (ia|ib)=(ib|ia) ⇒ anti=0
    assert r.e_bb == 0.0 and r.e_ab == 0.0


def test_ump2_general_kernel_matches_cpp_ump2_uhf_reference():
    """End-to-end: ump2_energy reproduces vibe-qc's C++ UMP2 (e_aa/e_bb/e_ab and
    the total) on the CH₃ radical / STO-3G — validates the spin-channel formula
    against the reference implementation (cpp/src/ump2.cpp), including the αα
    same-spin channel (which the MSINDO oracle cannot, due to its mp2uhf.f bug)."""
    from vibeqc._vibeqc_core import (Atom, Molecule, BasisSet, run_uhf, run_ump2,
                                     UHFOptions, UMP2Options, InitialGuess,
                                     compute_eri)
    A2B = 1.8897259886
    mol = Molecule([Atom(6, [0, 0, 0]),                       # planar CH₃·
                    Atom(1, [1.079 * A2B, 0, 0]),
                    Atom(1, [-0.5395 * A2B, 0.9345 * A2B, 0]),
                    Atom(1, [-0.5395 * A2B, -0.9345 * A2B, 0])], 0, 2)
    bas = BasisSet(mol, "sto-3g")
    o = UHFOptions()
    o.conv_tol_energy = 1e-11
    o.conv_tol_grad = 1e-8
    o.max_iter = 300
    o.initial_guess = InitialGuess.SAD
    uhf = run_uhf(mol, bas, o)
    assert uhf.converged
    options = UMP2Options()
    options.n_frozen_core = 0
    ref = run_ump2(mol, bas, uhf, options)
    na, nb = 5, 4                                     # CH₃: 9 e⁻, mult 2
    Ca = np.asarray(uhf.mo_coeffs_alpha); Cb = np.asarray(uhf.mo_coeffs_beta)
    ea = np.asarray(uhf.mo_energies_alpha); eb = np.asarray(uhf.mo_energies_beta)
    eri = np.asarray(compute_eri(bas))

    def ovov(Co, Cv, Co2, Cv2):
        return np.einsum("pqrs,pi,qa,rj,sb->iajb", eri, Co, Cv, Co2, Cv2,
                         optimize=True)

    Cao, Cav = Ca[:, :na], Ca[:, na:]
    Cbo, Cbv = Cb[:, :nb], Cb[:, nb:]
    r = ump2_energy(ea[:na], ea[na:], eb[:nb], eb[nb:],
                    ovov(Cao, Cav, Cao, Cav), ovov(Cbo, Cbv, Cbo, Cbv),
                    ovov(Cao, Cav, Cbo, Cbv))
    assert r.e_aa == pytest.approx(ref.e_aa, abs=1e-9)
    assert r.e_bb == pytest.approx(ref.e_bb, abs=1e-9)
    assert r.e_ab == pytest.approx(ref.e_ab, abs=1e-9)
    assert r.e_corr == pytest.approx(ref.e_correlation, abs=1e-9)
    assert r.e_aa < 0.0                               # CH₃ has a real αα channel
