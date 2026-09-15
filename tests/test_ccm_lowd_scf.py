"""A-line mixed-boundary wire SCF control (D3 M4/M5).

The wire SCF assembles the mixed-boundary Hamiltonian (plain supercell S/T + wire
V_ne/E_nn/J/K, all on one conditional gauge). Its validation is reference-free:

* the strict (S⊗S-projected) exchange makes the HF total ``g_perp_min``-independent;
* the **M5 gap-collapse** -- dense four-center SCF == RI-cderi SCF to machine ε
  (the wire route's ``4c − RI`` is zero, versus ~22 mHa on a 3-D vacuum-padded
  cell);
* the strict total from the reciprocal quadrature == the strict total assembled
  from **exact erf/erfc image-sum integrals**, a whole-Hamiltonian check that the
  quadrature computes the correct wire electrostatics.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erf

from vibeqc import Atom, PeriodicSystem
from vibeqc._vibeqc_core import compute_kinetic, compute_overlap
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.lowd_four_center import (
    ccm_eri_wire,
    ccm_wire_cderi,
    ccm_wire_e_nn,
    ccm_wire_v_ne,
)
from vibeqc.periodic.ccm.lowd_scf import _wire_ss_gauge_coefficient, run_ccm_rhf_wire
from vibeqc.periodic.ccm.ri import _rhf_loop, ccm_ri_j_neutral, ccm_ri_k_neutral

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


@pytest.fixture(scope="module")
def chain():
    return CCMSystem(PeriodicSystem(3, np.diag([6.0, 15.0, 15.0]),
                     [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1),
                     (2, 1, 1), "sto-3g")


def test_wire_scf_converges_and_labels_convention(chain):
    """The default is ``exxdiv="free"`` (2026-07-10). ``strict`` is IR-stable but
    deletes the free-space S⊗S exchange and so misses the non-interacting limit by a
    constant. Each convention must land a distinguishable manifest label."""
    res = run_ccm_rhf_wire(chain)
    assert res.converged
    assert res.exchange_q0 == "wire-free-space-seam"        # the new default
    assert res.idempotency_error < 1e-8
    assert res.energy < 0.0

    # A converged result must be one internally consistent fixed point.  The
    # historical loop replaced D/C from a DIIS-extrapolated next step *after*
    # evaluating the returned E/F/residual on the preceding density.
    d = np.asarray(res.density)
    f = np.asarray(res.fock)
    s = np.asarray(res.overlap)
    h = np.asarray(res.hcore)
    assert np.max(np.abs(f @ d @ s - s @ d @ f)) < 1e-6
    assert res.energy == pytest.approx(
        0.5 * np.sum(d * (h + f)) + res.e_nuclear, abs=1e-10
    )

    assert run_ccm_rhf_wire(chain, exxdiv="strict").exchange_q0 == "strict-zero-mode"
    assert run_ccm_rhf_wire(chain, exxdiv="ewald").exchange_q0 == "BvK-ewald"


def test_wire_scf_core_bearing_system_fails_closed_before_quadrature():
    """Carbon wires previously spent the full build/SCF budget then oscillated.

    The radial cutoff is basis-aware, but the transverse angular quadrature has no
    tight-core convergence proof.  Keep the unsupported envelope explicit until the
    joint ``g_max x n_ang`` validation lands.
    """
    carbon = CCMSystem(
        PeriodicSystem(3, np.diag([5.0, 15.0, 15.0]),
                       [Atom(6, [0, 0, 0]), Atom(6, [2.5, 0, 0])], 0, 1),
        (2, 1, 1),
        "sto-3g",
    )
    with pytest.raises(NotImplementedError, match="core-bearing wire SCF"):
        run_ccm_rhf_wire(carbon)


def test_wire_scf_strict_total_gauge_independent(chain):
    """The strict exchange makes the HF total g_perp_min-independent (M4)."""
    e_a = run_ccm_rhf_wire(chain, g_perp_min=1e-4).energy
    e_b = run_ccm_rhf_wire(chain, g_perp_min=1e-3).energy
    assert abs(e_a - e_b) < 1e-4                      # measured ~4e-6

    # ... while the RAW (unprojected) exchange drifts strongly with the IR cut
    r_a = run_ccm_rhf_wire(chain, g_perp_min=1e-4, exxdiv=None).energy
    r_b = run_ccm_rhf_wire(chain, g_perp_min=1e-3, exxdiv=None).energy
    assert abs(r_a - r_b) > 0.1                       # the ½c·N_e IR seam


def test_wire_ewald_seam(chain):
    """exxdiv="ewald" is the wire-Madelung seam K -> K_strict + ξ_wire·S D S: same
    density as strict, energy shifted by exactly ξ_wire·N_e/2, g_perp_min-independent
    (the exact 1-D analogue of run_ccm_rhf_direct's ξ_N·S D S seam)."""
    from vibeqc.periodic.lowd_greens import wire_self_energy

    ccm = chain
    n_e = ccm.supercell.n_electrons()
    xi = wire_self_energy(float(ccm.cluster_vectors[0, 0]))
    st = run_ccm_rhf_wire(ccm, exxdiv="strict")
    ew = run_ccm_rhf_wire(ccm, exxdiv="ewald")
    assert ew.exchange_q0 == "BvK-ewald"
    # energy shift is exactly ξ_wire·N_e/2 (ewald sits below strict by that)
    assert ew.energy - st.energy == pytest.approx(-xi * n_e / 2.0, abs=1e-6)
    # block-diagonal seam -> identical converged density
    assert np.linalg.norm(np.asarray(ew.density) - np.asarray(st.density)) < 1e-6
    # g_perp_min-independent (physical constant, not the IR gauge)
    ew2 = run_ccm_rhf_wire(ccm, exxdiv="ewald", g_perp_min=1e-3)
    assert abs(ew.energy - ew2.energy) < 1e-4
    # the ξ_wire constant is overridable (for the D2-anchored value)
    ew_x = run_ccm_rhf_wire(ccm, exxdiv="ewald", exxdiv_xi=0.0)
    assert ew_x.energy == pytest.approx(st.energy, abs=1e-6)  # ξ=0 -> back to strict


def test_wire_scf_dense_equals_ri_gap_collapse(chain):
    """M5: dense four-center SCF == RI-cderi SCF to machine ε (4c − RI → 0)."""
    ccm = chain
    B = ccm_wire_cderi(ccm)
    S = np.asarray(compute_overlap(ccm.basis))
    h = np.asarray(compute_kinetic(ccm.basis)) + ccm_wire_v_ne(ccm, cderi=B)
    e_nn = ccm_wire_e_nn(ccm)
    cstar = _wire_ss_gauge_coefficient(B, S)

    e_ri = run_ccm_rhf_wire(ccm, cderi=B, exxdiv="strict").energy

    g = ccm_eri_wire(ccm, cderi=B)
    j_of_D = lambda D: np.einsum("mnrs,rs->mn", g, D, optimize=True)
    k_of_D = lambda D: (np.einsum("mrns,rs->mn", g, D, optimize=True)
                        - cstar * (S @ D @ S))
    e_dense = _rhf_loop(ccm, S, h, e_nn, j_of_D, k_of_D, max_iter=200,
                        conv_tol=1e-9, diis_dim=8, lindep_tol=1e-7).energy
    assert abs(e_ri - e_dense) < 1e-10                # measured 5e-13


def _pair_charges(ccm):
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


def test_wire_scf_matches_exact_image_sum_hamiltonian(chain):
    """Decisive whole-Hamiltonian gate: the strict wire SCF from the quadrature ==
    the strict wire SCF assembled from EXACT erf image-sum integrals (both project
    S⊗S out of exchange, so they agree despite different IR regulators)."""
    ccm = chain
    S = np.asarray(compute_overlap(ccm.basis))
    T = np.asarray(compute_kinetic(ccm.basis))
    L = float(ccm.cluster_vectors[0, 0])
    Z = np.array([int(a.Z) for a in ccm.supercell.atoms], float)
    R = np.asarray(ccm.atom_positions, float)
    pairs, nbf = _pair_charges(ccm)
    ns = np.arange(-2500, 2501).astype(float)

    g = np.zeros((nbf, nbf, nbf, nbf))
    for m in range(nbf):
        for n in range(m, nbf):
            for r in range(nbf):
                for s in range(r, nbf):
                    v = 0.0
                    for q1, p1, P1 in pairs[(m, n)]:
                        for q2, p2, P2 in pairs[(r, s)]:
                            mu = np.sqrt(p1 * p2 / (p1 + p2))
                            d = P1 - P2
                            dn = np.sqrt((d[0] + ns * L) ** 2 + d[1] ** 2 + d[2] ** 2)
                            f = np.where(dn > 1e-12, erf(mu * dn) / np.maximum(dn, 1e-300),
                                         2.0 * mu / np.sqrt(np.pi))
                            v += q1 * q2 * float(np.sum(f))
                    g[m, n, r, s] = g[n, m, r, s] = g[m, n, s, r] = g[n, m, s, r] = v

    V = np.zeros((nbf, nbf))
    for m in range(nbf):
        for n in range(m, nbf):
            v = 0.0
            for q1, p1, P1 in pairs[(m, n)]:
                mu = np.sqrt(p1)
                for ZA, RA in zip(Z, R):
                    d = P1 - RA
                    dn = np.sqrt((d[0] + ns * L) ** 2 + d[1] ** 2 + d[2] ** 2)
                    f = np.where(dn > 1e-12, erf(mu * dn) / np.maximum(dn, 1e-300),
                                 2.0 * mu / np.sqrt(np.pi))
                    v += -ZA * q1 * float(np.sum(f))
            V[m, n] = V[n, m] = v

    e_nn = 0.0
    for iA in range(len(Z)):
        for iB in range(len(Z)):
            d = R[iA] - R[iB]
            dn = np.sqrt((d[0] + ns * L) ** 2 + d[1] ** 2 + d[2] ** 2)
            mask = dn > 1e-12
            e_nn += 0.5 * Z[iA] * Z[iB] * float(np.sum(1.0 / dn[mask]))

    ss2 = float(np.sum(S * S))
    cstar = float(np.einsum("mnrs,mn,rs->", g, S, S)) / (ss2 * ss2)
    j_of_D = lambda D: np.einsum("mnrs,rs->mn", g, D, optimize=True)
    k_of_D = lambda D: (np.einsum("mrns,rs->mn", g, D, optimize=True)
                        - cstar * (S @ D @ S))
    e_exact = _rhf_loop(ccm, S, T + V, e_nn, j_of_D, k_of_D, max_iter=200,
                        conv_tol=1e-9, diis_dim=8, lindep_tol=1e-7).energy

    e_quad = run_ccm_rhf_wire(ccm, exxdiv="strict").energy
    assert abs(e_quad - e_exact) < 1e-5               # measured 8.4e-7


@pytest.mark.parametrize("exxdiv", ["strict", "ewald"])
def test_streamed_wire_scf_matches_dense(chain, exxdiv):
    """Integral-direct wire SCF == dense-cderi wire SCF, for both exchange-q0 conventions.

    The streamed path never forms ``B``: it rebuilds the quadrature in chunks for each
    ``J``/``K`` pass. Nothing about the Hamiltonian changes, so this must be exact, not
    approximate -- measured bit-identical. Auto-selected when the dense ``B`` would
    exceed the 2 GiB guard, which is what makes core-bearing elements reachable
    (C/STO-3G wants ``g_max ~ 109``, an 8.1 GiB dense ``B`` vs a 0.25 GiB streamed peak).
    """
    ccm = chain
    kw = dict(n_rad=24, n_ang=16)
    dense = run_ccm_rhf_wire(ccm, exxdiv=exxdiv, stream=False, **kw)
    strm = run_ccm_rhf_wire(ccm, exxdiv=exxdiv, stream=True, **kw)
    assert dense.converged and strm.converged
    assert strm.energy == pytest.approx(dense.energy, abs=1e-12)
    assert strm.exchange_q0 == dense.exchange_q0


def test_streamed_wire_scf_rejects_prebuilt_cderi(chain):
    """stream=True never forms B, so a prebuilt cderi cannot be honoured. Fail loudly
    rather than silently ignore the caller's tensor."""
    import numpy as np

    ccm = chain
    with pytest.raises(ValueError, match="cannot reuse a prebuilt cderi"):
        run_ccm_rhf_wire(ccm, stream=True, cderi=np.zeros((2, 2, 2)),
                         n_rad=24, n_ang=16)


def test_wire_exxdiv_free_hits_the_non_interacting_limit(chain):  # noqa: ARG001
    """The correctness gate for a d=1 absolute energy: a chain of well-separated
    neutral closed-shell H₂ must reproduce isolated H₂ RHF.

    NOT the four-centre value -- that is the bare-1/r minimum-image Hamiltonian on a
    finite torus, while the wire is the neutralised mixed-boundary Hamiltonian of an
    infinite 1-D-periodic system. Different operators; they coincide only here.

    Measured 2026-07-10 (STO-3G, nrep (2,1,1)):

    * ``exxdiv="strict"`` projects out the whole S⊗S monopole, including the ordinary
      free-space exchange content, and converges to ``E_iso + 0.337`` Ha -- a constant
      error that does not vanish with the period (+0.395/+0.376/+0.364/+0.356 at
      L = 12/18/26/36 bohr).
    * ``exxdiv="free"`` subtracts only the IR-divergent part and approaches ``E_iso``
      as ``0.194/L``: ``dE·L`` = 0.1955/0.1942/0.1937/0.1936 across the same L.

    Gate the shape, not the constant: free must decay while strict must not.
    """
    from vibeqc import Atom, BasisSet, Molecule, PeriodicSystem, RHFOptions, run_rhf
    from vibeqc.periodic.ccm import CCMSystem

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    e_iso = run_rhf(mol, BasisSet(mol, "sto-3g"), opts).energy

    de_free, de_strict = {}, {}
    for L in (12.0, 24.0):
        ccm = CCMSystem(
            PeriodicSystem(3, np.diag([L, 20.0, 20.0]),
                           [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])], 0, 1),
            (2, 1, 1), "sto-3g")
        kw = dict(n_rad=32, n_ang=24)
        de_free[L] = run_ccm_rhf_wire(ccm, exxdiv="free", **kw).energy / ccm.n_cells - e_iso
        de_strict[L] = run_ccm_rhf_wire(ccm, exxdiv="strict", **kw).energy / ccm.n_cells - e_iso

    # free: error shrinks like 1/L, so doubling L roughly halves it
    assert 0.0 < de_free[24.0] < 0.6 * de_free[12.0]
    assert de_free[24.0] < 0.02
    # strict: error is a non-vanishing constant, nowhere near halving
    assert de_strict[24.0] > 0.8 * de_strict[12.0]
    assert de_strict[24.0] > 0.3
    # and free is much closer to the truth than strict at every L
    assert de_free[12.0] < 0.1 * de_strict[12.0]


def test_wire_exxdiv_free_inherits_stricts_infrared_stability(chain):
    """``free`` and ``strict`` must have *identical* ``g_perp_min`` drift.

    ``c*_free`` is the S⊗S weight of the cluster's free-space four-center, which
    contains no ``g_perp_min`` at all, so ``free`` and ``strict`` differ by an exactly
    IR-independent constant. Putting an absolute drift bound on ``free`` would instead
    be testing ``strict``'s residual quadrature error (1.01e-4 on this deliberately
    coarse fixture, 4e-6 on the converged one) and would say nothing about the new
    convention.

    ``exxdiv=None`` keeps the whole divergent seam and drifts by ~0.77 Ha per decade
    here, four orders of magnitude more. That contrast is the point.
    """
    ccm = chain
    kw = dict(n_rad=32, n_ang=24)
    gps = (1e-3, 1e-4)

    def drift(exx):
        e = [run_ccm_rhf_wire(ccm, exxdiv=exx, g_perp_min=gp, **kw).energy for gp in gps]
        return e[1] - e[0]

    d_free, d_strict, d_raw = drift("free"), drift("strict"), drift(None)
    assert d_free == pytest.approx(d_strict, abs=1e-12)    # measured: equal to 1e-10
    assert abs(d_raw) > 1000.0 * abs(d_free)


def test_wire_seam_response_is_exactly_linear(chain):
    """``E_supercell(ξ) = E_strict − (N_e/2)·ξ``: the seam is occ/virt block-diagonal
    at convergence, so it shifts the energy without moving the density. Measured
    dE/dξ = −2.000000 on this system (N_e = 4)."""
    ccm = chain
    kw = dict(n_rad=32, n_ang=24)
    n_e = ccm.supercell.n_electrons()
    e1 = run_ccm_rhf_wire(ccm, exxdiv="ewald", exxdiv_xi=-0.5, **kw).energy
    e2 = run_ccm_rhf_wire(ccm, exxdiv="ewald", exxdiv_xi=-0.1, **kw).energy
    assert (e2 - e1) / (-0.1 - (-0.5)) == pytest.approx(-n_e / 2.0, abs=1e-8)
