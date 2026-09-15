"""General one-particle Green's-function / electron-propagator layer.

This is the response-theory sibling of :mod:`vibeqc.correlation`: like MP2, the
diagonal second-order self-energy depends on the SCF reference only through its
molecular-orbital energies and a source of **MO-basis two-electron integrals**
(an :class:`~vibeqc.correlation.ERIProvider`).  Build it once and quasiparticle
ionization potentials / electron affinities work for *every* reference -- HF, DFT,
and the semiempirical engines (MSINDO) -- instead of being re-implemented per
engine.

The quantity computed is the **diagonal second-order self-energy** (the GF2 /
second-order electron-propagator approximation; the rigorous core that OVGF
renormalizes):

    S_pp(w) = S_{a,i,j} [2(pi|aj) - (pj|ai)].(pi|aj) / (w + e_a - e_i - e_j)   (2h1p)
            + S_{i,a,b} [2(pa|ib) - (pb|ia)].(pa|ib) / (w + e_i - e_a - e_b)   (2p1h)

with i,j occupied, a,b virtual, chemist's notation (pq|rs).  See A. Szabo &
N. S. Ostlund, *Modern Quantum Chemistry* (Macmillan, 1982), p. 380ff., and
J. V. Ortiz, WIREs Comput. Mol. Sci. 3, 123 (2013) for the electron-propagator
formulation.  The quasiparticle energy solves the Dyson equation
``w = e_p + S_pp(w)``; the pole strength (renormalization) is
``Γ_p = 1 / (1 - dS_pp/dw)`` evaluated at the solution.

Note on the MSINDO reference: MSINDO's own ``ovgfrhf_neu`` (2019) evaluates this
self-energy with a *factorized* integral approximation -- it pairs the (p,i) and
(a,j) AO densities on the same centres for both the Coulomb and the exchange
channel -- which is cruder than, and inconsistent with, the rigorous ZDO integral
set that MSINDO's MP2 (``pqrs.f``) uses.  vibe-qc deliberately ships the
*rigorous* self-energy here (the same INDO integral tensor that drives
MSINDO-MP2), so the GF2 IP is consistent with the MP2 correlation energy.  See
``docs/user_guide/msindo.md`` for the numerical comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuasiparticleResult:
    """Quasiparticle energy of one orbital (all energies in Hartree)."""

    orbital: int           # MO index p (0-based)
    eps_scf: float         # Koopmans (SCF) orbital energy e_p
    eps_qp: float          # quasiparticle energy e_p + S_pp(e_qp)
    sigma: float           # self-energy at the solution, S_pp(e_qp)
    pole_strength: float   # renormalization Γ_p = 1/(1 - dS/dw) in (0, 1]
    iterations: int        # Dyson iterations used (0 if non-iterative)
    converged: bool

    @property
    def correction(self) -> float:
        """Quasiparticle correction to Koopmans, e_qp - e_scf (Hartree)."""
        return self.eps_qp - self.eps_scf


def _sigma_and_deriv(eps_o, eps_v, V, X, W, Y, omega):
    """S_pp(w) and dS/dw for the pre-sliced integral blocks of one orbital.

    ``V[i,a,j]=(pi|aj)``, ``X=transpose(V)`` over i<->j (=(pj|ai)); ``W[a,i,b]=
    (pa|ib)``, ``Y=transpose(W)`` over a<->b (=(pb|ia)).  Returns (S, dS/dw).
    """
    # 2h1p (hole/ionization) denominator D1[i,a,j] = w + e_a - e_i - e_j.
    d1 = (omega + eps_v[None, :, None]
          - eps_o[:, None, None] - eps_o[None, None, :])
    n1 = (2.0 * V - X) * V
    # 2p1h (particle/affinity) denominator D2[a,i,b] = w + e_i - e_a - e_b.
    d2 = (omega + eps_o[None, :, None]
          - eps_v[:, None, None] - eps_v[None, None, :])
    n2 = (2.0 * W - Y) * W
    sigma = float(np.sum(n1 / d1) + np.sum(n2 / d2))
    # d(1/D)/dw = -1/D^2  -> dS/dw.
    dsigma = float(-np.sum(n1 / d1**2) - np.sum(n2 / d2**2))
    return sigma, dsigma


def _slice_blocks(eri_mo, n_occ, p):
    """Slice the four integral blocks needed for orbital ``p``."""
    no = n_occ
    Gp = np.asarray(eri_mo)[p]                  # (q,r,s) = (pq|rs)
    V = Gp[:no, no:, :no]                        # (pi|aj)  -> (i,a,j)
    X = np.transpose(V, (2, 1, 0))               # (pj|ai)
    W = Gp[no:, :no, no:]                         # (pa|ib)  -> (a,i,b)
    Y = np.transpose(W, (2, 1, 0))               # (pb|ia)
    return V, X, W, Y


def diagonal_self_energy(eps, n_occ, eri_mo, p, omega) -> float:
    """Diagonal second-order self-energy ``S_pp(w)`` (Hartree).

    Reference-agnostic: ``eps`` are the MO energies, ``n_occ`` the number of
    doubly-occupied orbitals, ``eri_mo[p,q,r,s] = (pq|rs)`` the full MO-basis
    two-electron tensor (chemist's notation) from any
    :class:`~vibeqc.correlation.ERIProvider`, ``p`` the orbital index, ``omega``
    the frequency (Hartree).
    """
    eps = np.asarray(eps, float)
    V, X, W, Y = _slice_blocks(eri_mo, n_occ, p)
    sigma, _ = _sigma_and_deriv(eps[:n_occ], eps[n_occ:], V, X, W, Y, float(omega))
    return sigma


def _solve_dyson(eps_p, sigma_at, *, iterate, max_iter, tol):
    """Solve the diagonal Dyson equation ``w = e_p + S_pp(w)`` for one orbital.

    ``sigma_at(omega)`` returns ``(S_pp(w), dS_pp/dw)``.  With ``iterate=False``
    the non-iterative estimate ``e_p + S_pp(e_p)`` is returned (the
    MSINDO/Koopmans-anchored convention).  Returns
    ``(eps_qp, sigma, pole_strength, iterations, converged)``; the pole strength
    is ``Γ = 1/(1 - dS/dw)`` at the solution.  Reference-agnostic: the same
    Newton driver serves the closed-shell spatial kernel and the open-shell
    spin-orbital kernel.
    """
    if not iterate:
        sigma, dsig = sigma_at(eps_p)
        return eps_p + sigma, sigma, 1.0 / (1.0 - dsig), 0, True

    omega = eps_p
    converged = False
    sigma = dsig = 0.0
    it = 0
    for it in range(1, max_iter + 1):
        sigma, dsig = sigma_at(omega)
        f = omega - eps_p - sigma            # Dyson residual
        if abs(f) < tol:
            converged = True
            break
        # Newton step on f(w) = w - e_p - S(w); f'(w) = 1 - S'(w).
        omega -= f / (1.0 - dsig)
    return omega, sigma, 1.0 / (1.0 - dsig), it, converged


def quasiparticle_energy(eps, n_occ, eri_mo, p, *, iterate: bool = True,
                         max_iter: int = 50, tol: float = 1e-8
                         ) -> QuasiparticleResult:
    """Quasiparticle energy of orbital ``p`` via the diagonal Dyson equation.

    Solves ``w = e_p + S_pp(w)`` by Newton iteration (``iterate=True``); with
    ``iterate=False`` returns the non-iterative estimate ``e_p + S_pp(e_p)``
    (the MSINDO/Koopmans-anchored convention).  The pole strength
    ``Γ_p = 1/(1 - dS/dw)`` is reported at the solution.  IP = -e_qp for an
    occupied orbital; EA = -e_qp for a virtual orbital.  Closed-shell (RHF/RKS)
    reference; for an open-shell reference see
    :func:`unrestricted_quasiparticle_energies`.
    """
    eps = np.asarray(eps, float)
    eps_p = float(eps[p])
    V, X, W, Y = _slice_blocks(eri_mo, n_occ, p)
    eo, ev = eps[:n_occ], eps[n_occ:]

    eps_qp, sigma, z, it, conv = _solve_dyson(
        eps_p, lambda w: _sigma_and_deriv(eo, ev, V, X, W, Y, w),
        iterate=iterate, max_iter=max_iter, tol=tol)
    return QuasiparticleResult(p, eps_p, eps_qp, sigma, z, it, conv)


def quasiparticle_energies(eps, n_occ, eri_mo, orbitals, *, iterate: bool = True,
                           max_iter: int = 50, tol: float = 1e-8):
    """Quasiparticle energies for several orbitals (see :func:`quasiparticle_energy`).

    Returns a list of :class:`QuasiparticleResult`, one per index in
    ``orbitals``.  Convenience for IP/EA spectra (e.g. the outer-valence band).
    """
    return [quasiparticle_energy(eps, n_occ, eri_mo, int(p), iterate=iterate,
                                 max_iter=max_iter, tol=tol) for p in orbitals]


# --------------------------------------------------------------------------- #
# Open-shell (spin-unrestricted) propagator: spin-orbital second-order S.       #
# --------------------------------------------------------------------------- #
#
# For a UHF/UKS reference the diagonal second-order self-energy is the
# spin-orbital generalisation of the closed-shell kernel above -- the
# electron-propagator analogue of UMP2 (Szabo & Ostlund, *Modern Quantum
# Chemistry*, Sec.7.5).  Summing over *spin-orbitals* with the antisymmetrised
# integral <PQ||RS> = <PQ|RS> - <PQ|SR> (physicist's notation):
#
#   S_pp(w) = 1/2 S_{i,a,b} |<pi||ab>|^2 / (w + e_i - e_a - e_b)        (2p1h)
#           + 1/2 S_{a,i,j} |<pa||ij>|^2 / (w + e_a - e_i - e_j)        (2h1p)
#
# i,j over occupied and a,b over virtual spin-orbitals of *either* spin.  Setting
# the a and b MOs equal (closed shell) reproduces the spatial (2V-X).V kernel
# term-for-term -- the load-bearing test.  The (2N)⁴ spin-orbital tensor is built
# explicitly, so this path is for small/medium systems (guarded by the caller).


def _spinorbital_aeri(eri_ao, Ca, Cb):
    """Antisymmetrised spin-orbital MO integrals ``A[P,Q,R,S] = <PQ||RS>``.

    ``eri_ao[mu,ν,l,s] = (muν|ls)`` is the AO chemist tensor (the same object MP2 /
    GF2 consume); ``Ca``/``Cb`` are the a/b MO coefficient matrices (AO x MO).
    Spin-orbitals are ordered ``[a_0...a_{N-1}, b_0...b_{N-1}]``.
    """
    eri_ao = np.asarray(eri_ao, float)
    n = Ca.shape[1]
    n2 = 2 * n

    def mo(C1, C2, C3, C4):
        # Spatial MO chemist block (pq|rs); only same-spin pairs survive.
        return np.einsum("wxyz,wp,xq,yr,zs->pqrs", eri_ao, C1, C2, C3, C4,
                         optimize=True)

    J_aa = mo(Ca, Ca, Ca, Ca)
    J_bb = mo(Cb, Cb, Cb, Cb)
    J_ab = mo(Ca, Ca, Cb, Cb)                # (aa|bb)
    G = np.zeros((n2, n2, n2, n2))
    a, b = slice(0, n), slice(n, n2)
    G[a, a, a, a] = J_aa
    G[b, b, b, b] = J_bb
    G[a, a, b, b] = J_ab
    G[b, b, a, a] = np.transpose(J_ab, (2, 3, 0, 1))   # (bb|aa) = (aa|bb)ᵀ
    # Physicist <PQ|RS> = (PR|QS); antisymmetrise R<->S -> <PQ||RS>.
    phys = np.transpose(G, (0, 2, 1, 3))
    return phys - np.transpose(phys, (0, 1, 3, 2))


def _so_sigma_and_deriv(eps_so, occ, A, p, omega):
    """Spin-orbital diagonal 2nd-order S_pp(w) and dS/dw for spin-orbital ``p``.

    ``A[p]`` is ``<pQ||RS>`` (a slice of :func:`_spinorbital_aeri`); ``occ`` is a
    boolean occupied mask over spin-orbitals.
    """
    o = np.flatnonzero(occ)
    v = np.flatnonzero(~occ)
    eo, ev = eps_so[o], eps_so[v]
    Ap = A[p]
    # 2h1p (ionization): <pa||ij>, a virtual, i,j occupied.
    Vh = Ap[np.ix_(v, o, o)]
    Dh = omega + ev[:, None, None] - eo[None, :, None] - eo[None, None, :]
    # 2p1h (attachment): <pi||ab>, i occupied, a,b virtual.
    Vp = Ap[np.ix_(o, v, v)]
    Dp = omega + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
    sigma = 0.5 * np.sum(Vh * Vh / Dh) + 0.5 * np.sum(Vp * Vp / Dp)
    dsig = -0.5 * np.sum(Vh * Vh / Dh**2) - 0.5 * np.sum(Vp * Vp / Dp**2)
    return float(sigma), float(dsig)


def unrestricted_quasiparticle_energies(
        eri_ao, Ca, Cb, eps_a, eps_b, n_a, n_b, orbitals, *,
        iterate: bool = True, max_iter: int = 50, tol: float = 1e-8,
        max_spinorbitals: int = 80):
    """Quasiparticle energies for a spin-unrestricted (UHF/UKS) reference.

    Open-shell sibling of :func:`quasiparticle_energies`.  ``eri_ao`` is the AO
    chemist tensor ``(muν|ls)``; ``Ca``/``Cb`` the a/b MO coefficients (AO x MO);
    ``eps_a``/``eps_b`` the a/b MO energies; ``n_a``/``n_b`` the a/b occupied
    counts.  ``orbitals`` is a list of ``(spin, index)`` tuples
    (``spin in {"alpha", "beta"}``, ``index`` a 0-based *spatial* MO index).

    Returns a list of ``(spin, QuasiparticleResult)`` in the input order; the
    result's ``orbital`` field holds the spatial index.  IP = -e_qp (occupied),
    EA = -e_qp (virtual).  The ``(2N)⁴`` spin-orbital integral tensor is built
    explicitly; ``max_spinorbitals`` (default 80 => N <= 40 spatial) guards the
    transform against OOM.
    """
    Ca = np.asarray(Ca, float)
    Cb = np.asarray(Cb, float)
    eps_a = np.asarray(eps_a, float)
    eps_b = np.asarray(eps_b, float)
    n = Ca.shape[1]
    if 2 * n > max_spinorbitals:
        raise NotImplementedError(
            f"the open-shell propagator builds the full spin-orbital integral "
            f"tensor (2N = {2 * n} spin-orbitals); it is limited to "
            f"2N <= {max_spinorbitals} (N <= {max_spinorbitals // 2} spatial "
            f"orbitals).  Use a smaller basis.")
    A = _spinorbital_aeri(eri_ao, Ca, Cb)
    eps_so = np.concatenate([eps_a, eps_b])
    occ = np.zeros(2 * n, dtype=bool)
    occ[:n_a] = True
    occ[n:n + n_b] = True

    out = []
    for spin, idx in orbitals:
        if spin not in ("alpha", "beta"):
            raise ValueError(f"spin must be 'alpha' or 'beta', got {spin!r}.")
        p = int(idx) if spin == "alpha" else n + int(idx)
        eps_p = float(eps_so[p])
        eps_qp, sigma, z, it, conv = _solve_dyson(
            eps_p, lambda w, q=p: _so_sigma_and_deriv(eps_so, occ, A, q, w),
            iterate=iterate, max_iter=max_iter, tol=tol)
        out.append((spin, QuasiparticleResult(int(idx), eps_p, eps_qp, sigma,
                                               z, it, conv)))
    return out


# --------------------------------------------------------------------------- #
# Renormalized GF2 -- full third-order diagonal self-energy + geometric screening #
# --------------------------------------------------------------------------- #
#
# The bare diagonal second-order self-energy (above) systematically *over*-shoots
# the quasiparticle correction because it treats the 2h1p / 2p1h intermediate
# states as non-interacting.  The renormalization restores their interaction
# through the diagonal **third-order** self-energy and a geometric resummation of
# the leading higher-order terms -- the outer-valence Green's-function idea of
# Cederbaum (J. Phys. B 8, 290 (1975)) and von Niessen, Schirmer & Cederbaum
# (Comput. Phys. Rep. 1, 57 (1984)).
#
# vibe-qc ships a transparently-documented **renormalized second-order GF**, NOT
# the literal von Niessen-Schirmer-Cederbaum "OVGF A/B/C" (whose exact screening
# partition is in the 1984 review and is not validatable here against an external
# code).  The third-order self-energy is the standard diagonal ADC(3) / P3 object
# -- the 2h1p W (static) and U (dynamic) terms of the electron-propagator P3 method
# (J. V. Ortiz, electron-propagator theory; the 2p1h channel is its particle-hole
# mirror) -- and is validated against an explicit definitional loop.  The
# screening is the geometric approximation  S_c ≈ S^2_c/(1 - S^3_c/S^2_c)  applied
# per channel c in {2h1p, 2p1h}, with a fallback to the bare S^2_c + S^3_c outside
# the convergence radius |S^3_c/S^2_c| < cap.  On H₂O this lifts the HOMO IP from
# the bare-GF2 overcorrection back toward experiment.


def _sigma2_channels_and_deriv(g, eps, occ, p, omega):
    """Second-order self-energy + d/dw, (2h1p, 2p1h) channels, for orbital p.

    Returns ``(S^2_2h1p, S^2_2p1h, dS^2_2h1p/dw, dS^2_2p1h/dw)``.  Each w-derivative
    is ``-1/2 S |g|^2/D^2`` (the outer denominator differentiated)."""
    o = np.flatnonzero(occ)
    v = np.flatnonzero(~occ)
    eo, ev = eps[o], eps[v]
    gpaij = g[np.ix_([p], v, o, o)][0]                 # <pa||ij>  [a,i,j]
    dh = omega + ev[:, None, None] - eo[None, :, None] - eo[None, None, :]
    n2h = gpaij * gpaij
    s2h = 0.5 * float(np.sum(n2h / dh))
    ds2h = -0.5 * float(np.sum(n2h / dh**2))
    gpiab = g[np.ix_([p], o, v, v)][0]                 # <pi||ab>  [i,a,b]
    dp = omega + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
    n2p = gpiab * gpiab
    s2p = 0.5 * float(np.sum(n2p / dp))
    ds2p = -0.5 * float(np.sum(n2p / dp**2))
    return s2h, s2p, ds2h, ds2p


def _sigma3_channels_and_deriv(g, eps, occ, p, omega):
    """Diagonal third-order self-energy + d/dw, (2h1p, 2p1h) channels, orbital p.

    Returns ``(S^3_2h1p, S^3_2p1h, dS^3_2h1p/dw, dS^3_2p1h/dw)``.
    ``g[p,q,r,s] = <pq||rs>`` (antisymmetrized, physicist).  The 2h1p channel is
    the P3 third-order self-energy (static ``W`` ladder+ring and dynamic ``U``
    ladder+ring terms, antisymmetrized over the two holes); the 2p1h channel is
    its particle-hole mirror.  The w-derivative differentiates the dynamic ``U``
    terms (same contraction with the denominator squared) and the outer
    2h1p/2p1h denominators (``W`` is w-independent).  Validated against an
    explicit definitional loop and central-difference (``tests/test_propagator``)."""
    o = np.flatnonzero(occ)
    v = np.flatnonzero(~occ)
    eo, ev = eps[o], eps[v]
    es = np.einsum

    # ---------------- 2h1p channel ----------------
    # static W[i,j,a] = 1/2 S_bc <bc||pa><ij||bc>/(ei+ej-eb-ec)
    #                 + (1-P_ij) S_bk <bi||pk><jk||ba>/(ej+ek-ea-eb)
    d_oovv = (eo[:, None, None, None] + eo[None, :, None, None]
              - ev[None, None, :, None] - ev[None, None, None, :])
    g_ijbc = g[np.ix_(o, o, v, v)]
    g_bcpa = g[np.ix_(v, v, [p], v)][:, :, 0, :]                  # [b,c,a]
    W = 0.5 * es("ijbc,bca->ija", g_ijbc / d_oovv, g_bcpa, optimize=True)
    g_bipk = g[np.ix_(v, o, [p], o)][:, :, 0, :]                  # [b,i,k]
    d_bjka = (eo[None, :, None, None] + eo[None, None, :, None]
              - ev[:, None, None, None] - ev[None, None, None, :])
    g_jkba = np.transpose(g[np.ix_(o, o, v, v)], (2, 0, 1, 3))    # [b,j,k,a]
    ring = es("bik,bjka->ija", g_bipk, g_jkba / d_bjka, optimize=True)
    W = W + ring - np.transpose(ring, (1, 0, 2))
    # dynamic U[i,j,a] = -1/2 S_kl <pa||kl><kl||ij>/(w+ea-ek-el)
    #                  - (1-P_ij) S_bk <pb||jk><ak||bi>/(w+eb-ej-ek)
    # and dU/dw (same contractions, denominator squared; d(-1/D)/dw = +1/D^2).
    g_pakl = g[np.ix_([p], v, o, o)][0]                          # [a,k,l]
    d_wakl = omega + ev[:, None, None] - eo[None, :, None] - eo[None, None, :]
    g_klij = g[np.ix_(o, o, o, o)]                               # [k,l,i,j]
    U = -0.5 * es("akl,klij->ija", g_pakl / d_wakl, g_klij, optimize=True)
    dU = 0.5 * es("akl,klij->ija", g_pakl / d_wakl**2, g_klij, optimize=True)
    g_pbjk = g[np.ix_([p], v, o, o)][0]                          # [b,j,k]
    g_akbi = g[np.ix_(v, o, v, o)]                               # [a,k,b,i]
    d_wbjk = omega + ev[:, None, None] - eo[None, :, None] - eo[None, None, :]
    ringU = es("bjk,akbi->ija", g_pbjk / d_wbjk, g_akbi, optimize=True)
    dringU = -es("bjk,akbi->ija", g_pbjk / d_wbjk**2, g_akbi, optimize=True)
    U = U - ringU + np.transpose(ringU, (1, 0, 2))
    dU = dU - dringU + np.transpose(dringU, (1, 0, 2))
    gpaij = g[np.ix_([p], v, o, o)][0]                            # [a,i,j]
    d_aij = omega + ev[:, None, None] - eo[None, :, None] - eo[None, None, :]
    s3h = 0.5 * float(es("aij,ija->", gpaij / d_aij, W + U, optimize=True))
    ds3h = 0.5 * float(es("aij,ija->", gpaij / d_aij, dU, optimize=True)
                       - es("aij,ija->", gpaij / d_aij**2, W + U, optimize=True))

    # ---------------- 2p1h channel (particle-hole mirror) ----------------
    d_vvoo = (ev[:, None, None, None] + ev[None, :, None, None]
              - eo[None, None, :, None] - eo[None, None, None, :])
    g_abjk = g[np.ix_(v, v, o, o)]
    g_jkpi = g[np.ix_(o, o, [p], o)][:, :, 0, :]                  # [j,k,i]
    Wp = 0.5 * es("abjk,jki->abi", g_abjk / d_vvoo, g_jkpi, optimize=True)
    g_japc = g[np.ix_(o, v, [p], v)][:, :, 0, :]                 # [j,a,c]
    d_jbci = (ev[None, :, None, None] + ev[None, None, :, None]
              - eo[None, None, None, :] - eo[:, None, None, None])
    g_bcji = np.transpose(g[np.ix_(v, v, o, o)], (2, 0, 1, 3))    # [j,b,c,i]
    ringp = es("jac,jbci->abi", g_japc, g_bcji / d_jbci, optimize=True)
    Wp = Wp + ringp - np.transpose(ringp, (1, 0, 2))
    g_picd = g[np.ix_([p], o, v, v)][0]                          # [i,c,d]
    d_wicd = omega + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
    g_cdab = g[np.ix_(v, v, v, v)]                               # [c,d,a,b]
    Up = -0.5 * es("icd,cdab->abi", g_picd / d_wicd, g_cdab, optimize=True)
    dUp = 0.5 * es("icd,cdab->abi", g_picd / d_wicd**2, g_cdab, optimize=True)
    g_pjbc = g[np.ix_([p], o, v, v)][0]                          # [j,b,c]
    g_icja = g[np.ix_(o, v, o, v)]                               # [i,c,j,a]
    d_wjbc = omega + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
    ringUp = es("jbc,icja->abi", g_pjbc / d_wjbc, g_icja, optimize=True)
    dringUp = -es("jbc,icja->abi", g_pjbc / d_wjbc**2, g_icja, optimize=True)
    Up = Up - ringUp + np.transpose(ringUp, (1, 0, 2))
    dUp = dUp - dringUp + np.transpose(dringUp, (1, 0, 2))
    gpiab = g[np.ix_([p], o, v, v)][0]                            # [i,a,b]
    d_iab = omega + eo[:, None, None] - ev[None, :, None] - ev[None, None, :]
    s3p = 0.5 * float(es("iab,abi->", gpiab / d_iab, Wp + Up, optimize=True))
    ds3p = 0.5 * float(es("iab,abi->", gpiab / d_iab, dUp, optimize=True)
                       - es("iab,abi->", gpiab / d_iab**2, Wp + Up, optimize=True))
    return s3h, s3p, ds3h, ds3p


def _sigma3_channels(g, eps, occ, p, omega):
    """``(S^3_2h1p, S^3_2p1h)`` -- see :func:`_sigma3_channels_and_deriv`."""
    return _sigma3_channels_and_deriv(g, eps, occ, p, omega)[:2]


def _screen_and_deriv(s2, s3, ds2, ds3, cap):
    """Geometric resummation ``screen = S^2/(1 - S^3/S^2) = S^2^2/(S^2-S^3)`` and its
    w-derivative, given (S^2, S^3, dS^2/dw, dS^3/dw).  Falls back to the bare
    ``S^2+S^3`` (and ``dS^2+dS^3``) when the ratio leaves the convergence radius
    (|S^3/S^2| >= cap) or S^2 ≈ 0."""
    if abs(s2) < 1e-10 or abs(s3) >= cap * abs(s2):
        return s2 + s3, ds2 + ds3
    den = s2 - s3                                   # screen = s2^2/den
    val = s2 * s2 / den
    # d/dw [s2^2/den] with den = s2-s3, den' = ds2-ds3:
    dval = (2.0 * s2 * ds2 * den - s2 * s2 * (ds2 - ds3)) / (den * den)
    return val, dval


def _renormalized_sigma_and_deriv(g, eps, occ, p, omega, *, cap=0.5):
    """Renormalized self-energy S(w) and dS/dw (analytic) for orbital p.

    S(w) = screen(S^2_2h1p, S^3_2h1p) + screen(S^2_2p1h, S^3_2p1h).  The derivative is
    the analytic chain rule through each channel's screening (validated against a
    central difference in ``tests/test_propagator.py``), so the pole strength
    ``Γ = 1/(1 - dS/dw)`` is exact without the 3x cost of differencing the
    third-order self-energy."""
    s2h, s2p, ds2h, ds2p = _sigma2_channels_and_deriv(g, eps, occ, p, omega)
    s3h, s3p, ds3h, ds3p = _sigma3_channels_and_deriv(g, eps, occ, p, omega)
    vh, dvh = _screen_and_deriv(s2h, s3h, ds2h, ds3h, cap)
    vp, dvp = _screen_and_deriv(s2p, s3p, ds2p, ds3p, cap)
    return vh + vp, dvh + dvp


def renormalized_quasiparticle_energies(
        eri_ao, Ca, Cb, eps_a, eps_b, n_a, n_b, orbitals, *,
        iterate: bool = True, max_iter: int = 50, tol: float = 1e-7,
        screen_cap: float = 0.5, max_spinorbitals: int = 60):
    """Renormalized-GF2 quasiparticle energies (full third-order self-energy).

    Same interface as :func:`unrestricted_quasiparticle_energies` (closed shell =
    ``Ca == Cb``, ``n_a == n_b``), but the self-energy is the diagonal
    second-order self-energy **renormalized** by the full third-order self-energy
    through a per-channel geometric screening -- the documented renormalized
    second-order GF (lineage: Cederbaum 1975 / von Niessen-Schirmer-Cederbaum
    1984; *not* the literal VSC OVGF A/B/C -- see the module note).  This trims
    the bare-GF2 overcorrection toward experiment.

    Returns a list of ``(spin, QuasiparticleResult)``.  The third-order
    contractions scale steeply, so the spin-orbital count is guarded by
    ``max_spinorbitals`` (default 60 => N <= 30 spatial).  ``screen_cap`` is the
    geometric-resummation cutoff |S^3/S^2|."""
    Ca = np.asarray(Ca, float)
    Cb = np.asarray(Cb, float)
    eps_a = np.asarray(eps_a, float)
    eps_b = np.asarray(eps_b, float)
    n = Ca.shape[1]
    if 2 * n > max_spinorbitals:
        raise NotImplementedError(
            f"renormalized GF2 builds the full spin-orbital integral tensor and "
            f"third-order self-energy (2N = {2 * n} spin-orbitals); it is limited "
            f"to 2N <= {max_spinorbitals} (N <= {max_spinorbitals // 2} spatial "
            f"orbitals).  Use a smaller basis or the bare-GF2 path.")
    A = _spinorbital_aeri(eri_ao, Ca, Cb)
    eps_so = np.concatenate([eps_a, eps_b])
    occ = np.zeros(2 * n, dtype=bool)
    occ[:n_a] = True
    occ[n:n + n_b] = True

    out = []
    for spin, idx in orbitals:
        if spin not in ("alpha", "beta"):
            raise ValueError(f"spin must be 'alpha' or 'beta', got {spin!r}.")
        q = int(idx) if spin == "alpha" else n + int(idx)
        eps_p = float(eps_so[q])
        eps_qp, sigma, z, it, conv = _solve_dyson(
            eps_p,
            lambda w, r=q: _renormalized_sigma_and_deriv(
                A, eps_so, occ, r, w, cap=screen_cap),
            iterate=iterate, max_iter=max_iter, tol=tol)
        out.append((spin, QuasiparticleResult(int(idx), eps_p, eps_qp, sigma,
                                               z, it, conv)))
    return out


__all__ = [
    "QuasiparticleResult",
    "diagonal_self_energy",
    "quasiparticle_energy",
    "quasiparticle_energies",
    "unrestricted_quasiparticle_energies",
    "renormalized_quasiparticle_energies",
]
