"""Wire (1-D-periodic, open-transverse) neutral four-center (D3 M2b).

The decisive, reference-free gate: for an s-only basis every AO-pair density is a
sum of Gaussians (Gaussian product theorem), so the exact wire four-center is the
analytic image sum ``Σ_n erf(μ d_n)/d_n`` (with the ``d→0`` limit ``2μ/√π``) over
the chain period. The quadrature construction must match it up to the kernel's
single conditional constant ``c·S⊗S`` — measured 2.8e-8 Frobenius (6e-9 max
element). This is stronger than the 3-D vacuum-ladder reduction (whose finite-D
residual is dominated by the 3-D route's own vacuum-cell convergence artifacts):
it is exact, not asymptotic.

Also pinned: the quadrature is internally converged (refining it changes nothing
beyond S⊗S), and the gauge drift of the image sum itself is pure S⊗S.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from vibeqc import Atom, BasisSet, Molecule, PeriodicSystem
from vibeqc._vibeqc_core import compute_overlap
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.lowd_four_center import (
    ccm_eri_wire,
    ccm_wire_cderi,
    ccm_wire_e_nn,
    ccm_wire_v_ne,
    wire_quadrature,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


@pytest.fixture(scope="module")
def chain():
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1),
                    (2, 1, 1), "sto-3g")
    S = np.asarray(compute_overlap(ccm.basis))
    return ccm, S


def _pair_charges(ccm):
    """Per AO pair: the Gaussian-product pair charges (q, p, P) (s-only)."""
    prims = []
    for sh in ccm.basis.shells():
        assert int(sh.l) == 0
        ctr = np.asarray(sh.origin if hasattr(sh, "origin") else sh.center, float)
        ex = np.asarray(sh.exponents, float)
        cf = np.asarray(sh.coefficients if hasattr(sh, "coefficients")
                        else sh.contraction_coefficients, float)
        prims.append([(float(c), float(a), ctr)
                      for c, a in zip(np.atleast_1d(cf).ravel(), ex)])
    nbf = len(prims)
    pairs = {}
    for m in range(nbf):
        for n in range(nbf):
            lst = []
            for cm, am, A in prims[m]:
                for cn, an, B in prims[n]:
                    p = am + an
                    P = (am * A + an * B) / p
                    q = cm * cn * np.exp(-am * an / p * np.dot(A - B, A - B)) \
                        * (np.pi / p) ** 1.5
                    lst.append((q, p, P))
            pairs[(m, n)] = lst
    return pairs, nbf


def _g_images(ccm, pairs, nbf, N):
    """Exact wire four-center: Σ_n erf(μ d_n)/d_n image sums (d→0 → 2μ/√π)."""
    L = float(ccm.cluster_vectors[0, 0])
    ns = np.arange(-N, N + 1).astype(float)
    g = np.zeros((nbf, nbf, nbf, nbf))
    for m in range(nbf):
        for n in range(m, nbf):
            for r in range(nbf):
                for s in range(r, nbf):
                    val = 0.0
                    for q1, p1, P1 in pairs[(m, n)]:
                        for q2, p2, P2 in pairs[(r, s)]:
                            mu = np.sqrt(p1 * p2 / (p1 + p2))
                            d = P1 - P2
                            dn = np.sqrt((d[0] + ns * L) ** 2 + d[1] ** 2 + d[2] ** 2)
                            f = np.where(dn > 1e-12,
                                         erf(mu * dn) / np.maximum(dn, 1e-300),
                                         2.0 * mu / np.sqrt(np.pi))
                            val += q1 * q2 * float(np.sum(f))
                    g[m, n, r, s] = g[n, m, r, s] = g[m, n, s, r] = g[n, m, s, r] = val
    return g


def _v_ne_images(ccm, pairs, nbf, N):
    """Exact wire V_ne: -Σ_A Z_A Σ_n erf(√p·d_n)/d_n (point-nucleus limit μ=√p)."""
    L = float(ccm.cluster_vectors[0, 0])
    Z = np.array([int(a.Z) for a in ccm.supercell.atoms], float)
    R = np.asarray(ccm.atom_positions, float)
    ns = np.arange(-N, N + 1).astype(float)
    V = np.zeros((nbf, nbf))
    for m in range(nbf):
        for n in range(m, nbf):
            val = 0.0
            for q1, p1, P1 in pairs[(m, n)]:
                mu = np.sqrt(p1)                     # point nucleus = μ→∞ Gaussian
                for ZA, RA in zip(Z, R):
                    d = P1 - RA
                    dn = np.sqrt((d[0] + ns * L) ** 2 + d[1] ** 2 + d[2] ** 2)
                    f = np.where(dn > 1e-12,
                                 erf(mu * dn) / np.maximum(dn, 1e-300),
                                 2.0 * mu / np.sqrt(np.pi))
                    val += -ZA * q1 * float(np.sum(f))
            V[m, n] = V[n, m] = val
    return V


def _gauge_resid(d, S):
    SS = np.einsum("mn,rs->mnrs", S, S)
    c = float(np.sum(d * SS) / np.sum(SS * SS))
    return float(np.linalg.norm(d - c * SS)), c


def _gauge_resid_2d(d, S):
    c = float(np.sum(d * S) / np.sum(S * S))
    return float(np.linalg.norm(d - c * S)), c


def test_wire_four_center_matches_exact_image_sum(chain):
    """Quadrature wire four-center == analytic erf image sum, up to one c·S⊗S."""
    ccm, S = chain
    pairs, nbf = _pair_charges(ccm)
    # pair charges integrate to the overlap (Gaussian product theorem sanity)
    chk = np.array([[sum(q for q, _, _ in pairs[(m, n)]) for n in range(nbf)]
                    for m in range(nbf)])
    assert np.max(np.abs(chk - S)) < 1e-12

    g_exact = _g_images(ccm, pairs, nbf, N=2500)
    g_quad = ccm_eri_wire(ccm)
    resid, c = _gauge_resid(g_exact - g_quad, S)
    assert resid < 1e-6            # measured 2.8e-8
    # the tensor has the full ERI permutation symmetry by construction
    assert np.allclose(g_quad, np.transpose(g_quad, (1, 0, 2, 3)), atol=1e-12)
    assert np.allclose(g_quad, np.transpose(g_quad, (2, 3, 0, 1)), atol=1e-12)


def test_offaxis_adaptive_angular_grid_matches_exact_image_sum():
    """The offset-adaptive azimuthal grid resolves off-axis pairs (G-CCM-004).

    The azimuthal bandwidth of an off-axis pair is ``|G_perp| x offset``
    (Jacobi-Anger; trapezoid convergence per Trefethen & Weideman, SIAM
    Rev. 56, 385 (2014)), so a flat azimuthal count that is fine on-axis
    silently under-resolves off-axis pairs -- the failure mode behind the
    recorded +22.69 Ha/atom polyethylene garbage. Pinned here on an s-only
    fixture the exact erf image-sum oracle can price: one H atom 2 bohr
    off the wire axis. The failure needs the angular oscillation to outrun
    the pair-FT envelope ``e^{-g^2/4p}``: at H/STO-3G exponents a 2-bohr
    offset stays resolved by accident (the envelope dies before the
    aliased g region contributes -- measured 5.4e-7 at flat n_ang=16), so
    the offset is 5 bohr, which puts the aliasing onset at g ~ 2.6 where
    the envelope is still O(1). That is the same envelope-vs-bandwidth
    race a 2-bohr offset loses at carbon exponents (p ~ 143), where the
    recorded polyethylene garbage came from; the transverse plane is open
    (no images), so a large offset is legitimate.
    """
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                    [Atom(1, [0, 0, 0]), Atom(1, [1.4, 5.0, 0])], 0, 1),
                    (2, 1, 1), "sto-3g")
    S = np.asarray(compute_overlap(ccm.basis))
    pairs, nbf = _pair_charges(ccm)
    g_exact = _g_images(ccm, pairs, nbf, N=2500)

    g_flat16 = ccm_eri_wire(ccm, n_ang=16)
    resid_flat, _ = _gauge_resid(g_exact - g_flat16, S)
    assert resid_flat > 1e-4, (
        f"flat n_ang=16 unexpectedly fine off-axis ({resid_flat:.3e}); "
        "the fixture no longer exercises the angular failure mode"
    )

    from vibeqc.periodic.ccm.lowd_four_center import max_transverse_offset

    b_max = max_transverse_offset(ccm.supercell, axis=0)
    assert b_max == pytest.approx(5.0, abs=1e-12)
    g_adapt = ccm_eri_wire(ccm, n_ang=16, b_max=b_max)
    resid_adapt, _ = _gauge_resid(g_exact - g_adapt, S)
    assert resid_adapt < 1e-6, resid_adapt

    # On-axis input (b_max = 0) preserves the flat grid bit-for-bit.
    ccm0 = CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                     [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1),
                     (2, 1, 1), "sto-3g")
    assert max_transverse_offset(ccm0.supercell, axis=0) == 0.0
    g_a = ccm_eri_wire(ccm0, b_max=0.0)
    g_b = ccm_eri_wire(ccm0)
    assert np.array_equal(g_a, g_b)


def test_pair_ft_matches_the_overlap_for_l_greater_than_zero():
    """``rho(G=0) == S`` must hold for p/d shells, not only for s.

    ``ao_pair_fourier_transform`` returns the raw real-solid-harmonic
    convention, in which ``rho(G=0)`` equals the overlap for ``l = 0``
    but is scaled by ``(2l+1)/(4 pi)`` for ``l >= 1`` -- the Racah
    normalisation relating ``Y_lm`` to the regular solid harmonic
    ``S_lm = sqrt(4 pi/(2l+1)) r^l Y_lm`` (Helgaker, Jorgensen & Olsen,
    *Molecular Electronic-Structure Theory*, SS 6.4.2, Eqs. 6.4.11-6.4.13;
    Hermite machinery per McMurchie & Davidson, J. Comput. Phys. 26, 218
    (1978)). libint absorbs ``Y_00`` into the s contraction coefficient
    but not ``Y_lm`` above it, so the wire route must undo it -- the same
    correction the BIPOLE stack applies via
    ``bipole_ext_el_pole._libint_ylm_correction_per_ao``.

    The wire route did not, and because the factor is exactly 1.0 for an
    s-only basis, EVERY existing gate here was blind to it: the H/He
    identities, both erf image-sum oracles and the M3b/M5 gates are
    s-only. The result was core-bearing wire totals wrong by Hartrees per
    atom (He/cc-pvdz -0.728 Ha/atom vs the isolated cluster; He/6-31g,
    same core tightness but s-only, +0.038). This is the gate that would
    have caught it.
    """
    from vibeqc.periodic.ccm.lowd_four_center import _pair_ft_libint

    G0 = np.array([[0.0, 0.0, 0.0]])
    for label, atoms, basis_name, want_l in (
        ("H2 sto-3g", [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], "sto-3g", 0),
        ("C2 sto-3g", [Atom(6, [0, 0, 0]), Atom(6, [2.3, 0, 0])], "sto-3g", 1),
        ("He2 cc-pvdz", [Atom(2, [0, 0, 0]), Atom(2, [4.0, 0, 0])],
         "cc-pvdz", 1),
    ):
        basis = BasisSet(Molecule(atoms, 0, 1), basis_name)
        l_max = max(int(sh.l) for sh in basis.shells())
        assert l_max == want_l, (label, l_max)
        S = np.asarray(compute_overlap(basis), dtype=float)
        rho0 = np.asarray(_pair_ft_libint(basis, G0))[:, :, 0].real
        assert np.abs(rho0 - S).max() < 1e-12, (label, np.abs(rho0 - S).max())
        # Trace form: the electron count a density would integrate to.
        assert np.trace(np.linalg.solve(S, rho0)) == pytest.approx(
            basis.nbasis, abs=1e-9)


def test_pair_ft_correction_is_a_no_op_for_s_only():
    """An s-only basis must pass through the wrapper untouched.

    The correction factor is exactly 1.0 for ``l = 0``, so this is the
    assertion that makes the change a pure convention fix rather than a
    numerics change -- every s-only gate in this file and in
    tests/test_ccm_lowd_scf.py must be bit-identical across it.
    """
    from vibeqc._aopair_ft import ao_pair_fourier_transform
    from vibeqc.periodic.ccm.lowd_four_center import _pair_ft_libint

    basis = BasisSet(
        Molecule([Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1), "sto-3g")
    assert max(int(sh.l) for sh in basis.shells()) == 0
    G = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [2.0, 0.0, 0.0]])
    raw = np.asarray(ao_pair_fourier_transform(basis, G))
    wrapped = np.asarray(_pair_ft_libint(basis, G))
    assert np.array_equal(raw, wrapped)


def test_azimuthal_rule_uses_the_two_object_product_bandwidth():
    """The angular count must resolve ``g (b_1 + b_2)``, not ``g b_max``.

    The azimuthal integrand is a PRODUCT of two reciprocal objects, each
    with its own translation phase, so its bandwidth is the SUM of their
    offsets -- bounded by ``2 g b_max``. The rule shipped on 2026-08-01
    used the single-object ``g b_max`` and was short by up to 1.75x.

    Two atoms placed antipodally about the axis make the shortfall
    visible: pairs on opposite sides differ by ``2b``, the worst case the
    bound exists for. Halving ``b_max`` here reproduces the old rule
    exactly (both enter only through the product ``g * b_max``), so this
    compares the two rules on one fixture against the exact erf
    image-sum oracle. A single off-axis atom does NOT separate them --
    ``test_offaxis_adaptive_angular_grid_matches_exact_image_sum``
    passes under both -- which is why the shortfall survived that gate.
    """
    b = 2.5
    ccm = CCMSystem(PeriodicSystem(3, np.diag([6.0, 18.0, 18.0]),
                    [Atom(1, [0, b, 0]), Atom(1, [1.4, -b, 0])], 0, 1),
                    (2, 1, 1), "sto-3g")
    S = np.asarray(compute_overlap(ccm.basis))
    pairs, nbf = _pair_charges(ccm)
    g_exact = _g_images(ccm, pairs, nbf, N=2500)

    from vibeqc.periodic.ccm.lowd_four_center import max_transverse_offset

    b_max = max_transverse_offset(ccm.supercell, axis=0)
    assert b_max == pytest.approx(b, abs=1e-12)

    # Halving b_max reproduces the retired single-object rule.
    resid_single, _ = _gauge_resid(
        g_exact - ccm_eri_wire(ccm, n_ang=8, b_max=0.5 * b_max), S)
    resid_product, _ = _gauge_resid(
        g_exact - ccm_eri_wire(ccm, n_ang=8, b_max=b_max), S)

    assert resid_single > 1e-5, (
        f"single-object bandwidth unexpectedly sufficient ({resid_single:.3e}); "
        "fixture no longer separates the two rules"
    )
    assert resid_product < 1e-6, resid_product
    assert resid_product < 0.05 * resid_single


def test_wire_quadrature_internally_converged(chain):
    """Refining the quadrature changes nothing beyond the S⊗S gauge."""
    ccm, S = chain
    g = ccm_eri_wire(ccm)
    g_ref = ccm_eri_wire(ccm, g_max=24.0, n_rad=96, n_ang=96)
    resid, _ = _gauge_resid(g_ref - g, S)
    assert resid < 1e-9            # measured 2.5e-13


def test_wire_quadrature_weights_are_positive(chain):
    ccm, _ = chain
    pts, wv = wire_quadrature(float(ccm.cluster_vectors[0, 0]))
    assert np.all(wv > 0.0)
    assert pts.shape[1] == 3


def test_wire_cderi_powers_mp2_gauge_robust(chain):
    """The wire cderi drives RI-MP2 directly (the M2b payoff: the whole RI/DLPNO
    stack rides B). The c·S⊗S conditional gauge nearly cancels in (ov|ov) MO
    integrals (exactly, up to the S^CCM-vs-plain-S mismatch), so the correlation
    is IR-cutoff-robust; and it matches the healthiest 3-D vacuum cell while the
    3-D route drifts with vacuum size (its own finite-D artifact)."""
    from vibeqc.periodic.ccm.lowd_four_center import ccm_wire_cderi
    from vibeqc.periodic.ccm.mp2 import run_ccm_mp2
    from vibeqc.periodic.ccm.neutral import ccm_eri_neutral, ccm_neutral_cderi
    from vibeqc.periodic.ccm.scf import run_ccm_rhf

    ccm, _ = chain
    scf = run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm, ke_cutoff=40.0))
    e1 = run_ccm_mp2(ccm, scf, cderi=ccm_wire_cderi(ccm)).e_correlation
    e2 = run_ccm_mp2(ccm, scf, cderi=ccm_wire_cderi(ccm, g_perp_min=1e-2)).e_correlation
    assert abs(e1 - e2) < 5e-5                 # measured 1.1e-5 (gauge robustness)
    assert e1 < 0.0
    # agrees with the 3-D neutral correlation on the same (healthy, D=15) cell
    e3d = run_ccm_mp2(ccm, scf, cderi=ccm_neutral_cderi(ccm, ke_cutoff=40.0)).e_correlation
    assert abs(e1 - e3d) < 5e-4                # measured 1.7e-4


def test_wire_v_ne_matches_exact_image_sum(chain):
    """Wire V_ne == analytic -Σ_A Z_A Σ_n erf(√p·d_n)/d_n, up to one c'·S (D3 M4)."""
    ccm, S = chain
    pairs, nbf = _pair_charges(ccm)
    V_exact = _v_ne_images(ccm, pairs, nbf, N=4000)
    V_quad = ccm_wire_v_ne(ccm)
    resid, _ = _gauge_resid_2d(V_quad - V_exact, S)
    assert resid < 1e-6                             # measured 3.5e-7
    assert np.allclose(V_quad, V_quad.T, atol=1e-12)


def test_wire_v_ne_gauge_matches_four_center(chain):
    """Decisive gate: V_ne carries the SAME conditional gauge as the four-center,
    with coefficient -N_nuc -- i.e. Δc'(V_ne) = -N_nuc·Δc(4c) as g_perp_min varies.
    This equality is *exactly* the neutral-cell Hartree cancellation
    (½c N_e² - c N_nuc N_e + ½c N_nuc² = ½c(N_e-N_nuc)² = 0); only the exchange-q0
    seam survives (docs/aiccm2026dev_a_lowd_greens.md §10)."""
    ccm, S = chain
    n_nuc = float(sum(int(a.Z) for a in ccm.supercell.atoms))
    SS = np.einsum("mn,rs->mnrs", S, S)
    dc4 = float(np.sum((ccm_eri_wire(ccm, g_perp_min=1e-4)
                        - ccm_eri_wire(ccm, g_perp_min=1e-2)) * SS)
                / np.sum(SS * SS))
    dcV = float(np.sum((ccm_wire_v_ne(ccm, g_perp_min=1e-4)
                        - ccm_wire_v_ne(ccm, g_perp_min=1e-2)) * S)
                / np.sum(S * S))
    assert dcV / dc4 == pytest.approx(-n_nuc, rel=1e-4)   # measured -4.00002


def test_wire_e_nn_beta_independent(chain):
    """The mini-Ewald split parameter β drops out of E_nn (D3 M4)."""
    ccm, _ = chain
    vals = [ccm_wire_e_nn(ccm, beta=b) for b in (0.2, 0.4, 0.7)]
    assert max(vals) - min(vals) < 1e-6            # measured ~5e-9


def test_wire_e_nn_gauge_coefficient(chain):
    """E_nn carries the ½N_nuc²·c conditional gauge (same c as the four-center)."""
    ccm, S = chain
    n_nuc = float(sum(int(a.Z) for a in ccm.supercell.atoms))
    SS = np.einsum("mn,rs->mnrs", S, S)
    dc4 = float(np.sum((ccm_eri_wire(ccm, g_perp_min=1e-4)
                        - ccm_eri_wire(ccm, g_perp_min=1e-2)) * SS)
                / np.sum(SS * SS))
    dE = ccm_wire_e_nn(ccm, g_perp_min=1e-4) - ccm_wire_e_nn(ccm, g_perp_min=1e-2)
    assert dE / (0.5 * n_nuc**2 * dc4) == pytest.approx(1.0, rel=1e-4)  # 0.99999


def test_wire_hartree_total_gauge_invariant(chain):
    """FLAGSHIP D3 M4 gate: the neutral-cell Hartree total
    ``E_nn + Tr[D V_ne] + ½Tr[D J]`` is independent of the conditional gauge
    (``g_perp_min``), even though each term swings by ~6 Ha across the sweep. This
    is ``½c(N_e-N_nuc)²=0`` realised numerically -- the wire Hartree Hamiltonian is
    well-defined and gauge-free for a neutral cell; only the exchange-q0 seam
    survives as a physical convention."""
    from vibeqc.periodic.ccm.neutral import ccm_eri_neutral
    from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral
    from vibeqc.periodic.ccm.scf import run_ccm_rhf

    ccm, _ = chain
    D = np.asarray(run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm, ke_cutoff=40.0)).density)
    totals = []
    for gmin in (1e-4, 1e-3):
        B = ccm_wire_cderi(ccm, g_perp_min=gmin)
        E_H = (ccm_wire_e_nn(ccm, g_perp_min=gmin)
               + float(np.sum(D * ccm_wire_v_ne(ccm, cderi=B, g_perp_min=gmin)))
               + 0.5 * float(np.sum(D * ccm_ri_j_neutral(B, D))))
        totals.append(E_H)
    assert abs(totals[0] - totals[1]) < 1e-4       # measured 1.1e-5 (O(g_perp_min²))


def test_wire_gauge_seam_algebra(chain):
    """M4 exchange-seam foundation: the conditional gauge Δc·S⊗S of the wire
    four-center enters the Hartree energy as ½Δc·(TrPS)² (cancelled by the matching
    nuclear-side gauge for a NEUTRAL cell) and the exchange energy as
    ¼Δc·Tr[(PS)²] ≈ ½Δc·N_e — the UNcancelled exchange-q0 seam. Pinned against the
    measured tensor gauge shift between two IR cutoffs; residuals are the
    O(g_perp_min²) non-S⊗S content."""
    from vibeqc.periodic.ccm.lowd_four_center import ccm_wire_cderi
    from vibeqc.periodic.ccm.neutral import ccm_eri_neutral
    from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral
    from vibeqc.periodic.ccm.scf import run_ccm_rhf

    ccm, S = chain
    scf = run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm, ke_cutoff=40.0))
    D = np.asarray(scf.density)
    SS = np.einsum("mn,rs->mnrs", S, S)

    B1 = ccm_wire_cderi(ccm)                       # gmin = 1e-4
    B2 = ccm_wire_cderi(ccm, g_perp_min=1e-2)
    d = (np.einsum("Qmn,Qrs->mnrs", B1, B1, optimize=True)
         - np.einsum("Qmn,Qrs->mnrs", B2, B2, optimize=True))
    dc = float(np.sum(d * SS) / np.sum(SS * SS))
    assert np.linalg.norm(d - dc * SS) < 1e-3      # gauge shift is ~pure S⊗S

    dEJ = 0.5 * float(np.sum(D * (ccm_ri_j_neutral(B1, D) - ccm_ri_j_neutral(B2, D))))
    dEK = 0.25 * float(np.sum(D * (ccm_ri_k_neutral(B1, D) - ccm_ri_k_neutral(B2, D))))
    trPS = float(np.trace(D @ S))
    trPSPS = float(np.trace(D @ S @ D @ S))
    assert dEJ == pytest.approx(0.5 * dc * trPS**2, abs=1e-4)     # measured 1.1e-6
    assert dEK == pytest.approx(0.25 * dc * trPSPS, abs=1e-3)     # measured 9.4e-5


def test_required_g_max_scales_with_basis_and_undersampling_raises(chain):
    """The wire quadrature cutoff must be derived from the BASIS, not fixed.

    The AO-pair FT decays as exp(-G²/4p) with p = 2·e_max, so a fixed g_max silently
    under-resolves core functions. Measured 2026-07-09: polyethylene (C 1s, e_max=71.6,
    needs g_max ≳ 109) at the old default g_max=16 gave a *positive* wire SCF energy
    (+6.93 Ha/atom) and two IR gauges differing by 18 Ha/atom. An under-resolved
    four-center loses positivity, so this must fail loudly, never silently.
    """
    from vibeqc import Atom, PeriodicSystem
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.periodic.ccm.lowd_four_center import _resolve_g_max, required_g_max

    ccm, _ = chain

    # Pin the CLOSED FORM, not a band. A loose band (the original `12 < need_h < 25`)
    # admitted both the correct pair convention (p = 2·e_max -> 23.83) and the √2-too-
    # small AO convention (p = e_max -> 16.85), which is how the latter shipped
    # undetected on 2026-07-09. Re-derived here from the STO-3G exponents directly.
    def _closed_form(e_max, tau=1e-9):
        return np.sqrt(4.0 * (2.0 * e_max) * np.log(1.0 / tau))

    E_MAX_H, E_MAX_C = 3.42525091, 71.6168373      # STO-3G tightest primitives
    need_h = required_g_max(ccm.basis)             # H/STO-3G -> 23.83
    assert need_h == pytest.approx(_closed_form(E_MAX_H), rel=1e-6)
    assert need_h == pytest.approx(23.83, abs=0.01)

    # a carbon basis needs a far larger cutoff (C 1s exponent ~71.6)
    cmol = PeriodicSystem(3, np.diag([8.0, 15.0, 15.0]),
                          [Atom(6, [0, 0, 0]), Atom(6, [2.5, 0, 0])], 0, 1)
    cbasis = BasisSet(cmol.unit_cell_molecule(), "sto-3g")
    need_c = required_g_max(cbasis)                # C/STO-3G -> 108.96
    assert need_c == pytest.approx(_closed_form(E_MAX_C), rel=1e-6)
    assert need_c == pytest.approx(108.96, abs=0.01)

    # tol enters as sqrt(ln(1/tau)): a 1e-18 target is exactly sqrt(2)x a 1e-9 one
    assert required_g_max(ccm.basis, tol=1e-18) == pytest.approx(need_h * np.sqrt(2.0))

    # auto-selection returns the required value
    assert _resolve_g_max(ccm.basis, None, what="t") == pytest.approx(need_h)
    # an inadequate explicit value raises loudly (no silent garbage).
    # 80.0 clears the OLD (√2-too-small) criterion's 77.05 for carbon and must still
    # be rejected -- the polyethylene sweep at g_max=80 never converged.
    with pytest.raises(ValueError, match="cannot resolve this basis"):
        _resolve_g_max(cbasis, 16.0, what="t")
    with pytest.raises(ValueError, match="cannot resolve this basis"):
        _resolve_g_max(cbasis, 80.0, what="t")


def test_streamed_wire_contractions_match_dense(chain):
    """The integral-direct wire path must equal the dense cderi path exactly.

    ``J``, ``K``, ``V_ne`` and the ``S⊗S`` gauge coefficient are all sums over the
    quadrature index, so none of them needs the whole ``B`` resident. Streaming is
    what makes core-bearing elements reachable at all: the cutoff scales with the
    basis, not the cell (``required_g_max``), so C/STO-3G on polyethylene wants
    ``g_max ≈ 109``, whose dense ``B`` is 8.1 GiB against a 0.25 GiB streamed peak.
    Any drift here would be a silent physics change, so gate at round-off.
    """
    from vibeqc.periodic.ccm.lowd_four_center import (
        ccm_wire_cderi, ccm_wire_jk_stream, ccm_wire_ss_gauge_stream,
        ccm_wire_v_ne, ccm_wire_v_ne_stream, wire_cderi_chunks,
    )
    from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral

    ccm, S = chain
    kw = dict(n_rad=24, n_ang=16)
    tiny = 1 << 18                      # force many chunks

    B = ccm_wire_cderi(ccm, **kw)
    chunks = list(wire_cderi_chunks(ccm, max_bytes=tiny, **kw))
    assert len(chunks) > 1, "chunking did not actually split the quadrature"
    assert sum(c.shape[0] for c in chunks) == B.shape[0]

    rng = np.random.default_rng(11)
    D = rng.standard_normal((ccm.nbf, ccm.nbf))
    D = D + D.T

    J_s, K_s = ccm_wire_jk_stream(ccm, D, max_bytes=tiny, **kw)
    assert np.max(np.abs(J_s - ccm_ri_j_neutral(B, D))) < 1e-11
    assert np.max(np.abs(K_s - ccm_ri_k_neutral(B, D))) < 1e-10

    V_s = ccm_wire_v_ne_stream(ccm, max_bytes=tiny, **kw)
    assert np.max(np.abs(V_s - ccm_wire_v_ne(ccm, cderi=B, **kw))) < 1e-11

    sdotb = np.einsum("mn,Qmn->Q", S, B, optimize=True)
    c_dense = float(np.sum(sdotb * sdotb)) / float(np.sum(S * S)) ** 2
    c_stream = ccm_wire_ss_gauge_stream(ccm, S, max_bytes=tiny, **kw)
    assert c_stream == pytest.approx(c_dense, rel=1e-12)


def test_streamed_wire_contractions_match_dense_at_carbon_scale():
    """The chain fixture is H₂ (nbf=4); this closes the gap the 2026-07-11 carbon
    study exposed by exercising the streamed accumulation at *carbon* exponents and
    a realistic ``nbf`` with **many** chunks.

    Why it matters: the carbon wire SCF gives a positive, unconverged energy at the
    basis-derived cutoff. Before blaming the physics one must rule out a streamed
    accumulation bug at large ``nQ`` — the H₂ gate above only reaches ``nbf=4``. Here
    polyethylene (nbf=28, C 1s exponent 71.6) at the *required* ``g_max=109`` with a
    deliberately coarse grid keeps the dense reference ~0.1 GiB (routine) while the
    streamed path is forced through many chunks. Measured 2026-07-11:
    streamed == dense to 4e-13 (J) / 1e-12 (K), so the streaming is sound and the
    non-convergence is a genuine quadrature/SCF issue, not a streaming defect.
    """
    from vibeqc._vibeqc_core import compute_overlap
    from vibeqc.periodic.ccm.lowd_four_center import (
        ccm_wire_cderi, ccm_wire_jk_stream, ccm_wire_ss_gauge_stream,
    )
    from vibeqc.periodic.ccm.ri import ccm_ri_j_neutral, ccm_ri_k_neutral

    a = 2.55 * 1.8897259886
    atoms = [Atom(6, [0, 0, 0]), Atom(1, [0, 0.785, 0.785]), Atom(1, [0, -0.785, -0.785]),
             Atom(6, [a / 2, 0, 0]), Atom(1, [a / 2, 0.785, 0.785]),
             Atom(1, [a / 2, -0.785, -0.785])]
    pe = CCMSystem(PeriodicSystem(3, np.diag([a, 20.0, 20.0]), atoms, 0, 1),
                   (2, 1, 1), "sto-3g")
    assert int(pe.nbf) == 28
    S = np.asarray(compute_overlap(pe.basis), float)
    kw = dict(g_max=109.0, n_rad=12, n_ang=8)     # required g_max, coarse -> ~0.1 GiB dense

    B = ccm_wire_cderi(pe, **kw)
    assert B.nbytes < 0.3 * 1024**3                # stays a routine-sized test
    rng = np.random.default_rng(5)
    D = rng.standard_normal((28, 28)); D = D + D.T

    tiny = 1 << 22                                 # force many chunks over the big nQ
    J_s, K_s = ccm_wire_jk_stream(pe, D, max_bytes=tiny, **kw)
    assert np.max(np.abs(J_s - ccm_ri_j_neutral(B, D))) < 1e-10
    assert np.max(np.abs(K_s - ccm_ri_k_neutral(B, D))) < 1e-9

    sdotb = np.einsum("mn,Qmn->Q", S, B, optimize=True)
    c_dense = float(np.sum(sdotb * sdotb)) / float(np.sum(S * S)) ** 2
    assert ccm_wire_ss_gauge_stream(pe, S, max_bytes=tiny, **kw) == pytest.approx(
        c_dense, rel=1e-11)


def test_carbon_dense_cderi_exceeds_the_guard_but_streaming_does_not():
    """The memory claim behind the streamed path, measured on the real cluster.

    Both factors matter and neither can be guessed: the cutoff comes from the basis
    (``required_g_max``) and the tensor is ``2·nQ·nbf²``, so the blow-up needs the
    *actual* period and AO count. Polyethylene (C₂H₄ chain, STO-3G, nrep (2,1,1))
    at the default quadrature is 8.1 GiB dense against a 0.25 GiB streamed peak that
    does not grow with ``nQ``.
    """
    from vibeqc.periodic.ccm.lowd_four_center import (
        _CHUNK_BYTES, _MAX_BYTES, required_g_max, wire_quadrature,
    )

    a = 2.55 * 1.8897259886                       # 2.55 Å period, in bohr
    atoms = [Atom(6, [0, 0, 0]), Atom(1, [0, 0.785, 0.785]), Atom(1, [0, -0.785, -0.785]),
             Atom(6, [a / 2, 0, 0]), Atom(1, [a / 2, 0.785, 0.785]),
             Atom(1, [a / 2, -0.785, -0.785])]
    pe = CCMSystem(PeriodicSystem(3, np.diag([a, 20.0, 20.0]), atoms, 0, 1),
                   (2, 1, 1), "sto-3g")
    nbf = int(pe.nbf)
    L = float(pe.cluster_vectors[0, 0])
    g_c = required_g_max(pe.basis)
    assert g_c == pytest.approx(109.0, abs=0.5)   # C 1s sets it, not the cell

    n_q = wire_quadrature(L, g_max=g_c)[0].shape[0]   # default n_rad/n_ang
    dense = 16 * n_q * nbf * nbf
    assert dense > _MAX_BYTES, f"dense cderi unexpectedly fits: {dense / 1024**3:.2f} GiB"
    assert dense / 1024**3 == pytest.approx(8.1, abs=0.3)

    nq_chunk = max(1, _CHUNK_BYTES // (16 * nbf * nbf))
    peak = 16 * min(n_q, nq_chunk) * nbf * nbf
    assert peak <= _CHUNK_BYTES
    assert dense / peak > 25.0                    # measured 32x
