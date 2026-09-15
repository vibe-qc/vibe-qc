"""MSINDO CIS/TDA excited states and analytic excited-state gradient (Phase 6).

* :func:`run_cis` -- CIS/TDA excitation energies and amplitudes (closed-shell).
* :func:`cis_gradient` -- analytic nuclear gradient of the total excited-state
  energy via the CPHF Z-vector relaxed density + the CIS 2PDM, contracted with
  the per-pair integral derivatives (validated to ~1e-6 Ha/bohr vs
  :func:`cis_gradient_fd`, singlet + triplet, non-degenerate states).

The linear-response layer is built in the MO basis: :func:`_cis_mo_eri`
(INDO MO two-electron tensor), :func:`_cis_orbital_hessian` (the closed-shell
CPHF orbital Hessian / Z-vector matrix), and :func:`_cis_lagrangian_mo` (the
Z-vector right-hand side, singlet/triplet).  The Z-vector solve ``A.Z = -L``
and the non-symmetric-2PDM gradient contraction follow MSINDO's
``cisgrad.f`` / ``cphf_solver.f`` / ``azcalc.f`` / ``kloop.f`` / ``lloop.f``;
re-derived clean-room, validated against the out-of-process oracle.

References
----------
* Foresman, Head-Gordon, Pople & Frisch, J. Phys. Chem. 96, 135 (1992).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_HARTREE_TO_EV = 27.211386245988


# --------------------------------------------------------------------------- #
# CIS result dataclass                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class CISResult:
    """CIS / TDA excited states for MSINDO (energies in Hartree, ascending).

    Attributes
    ----------
    nocc : int
        Number of occupied spatial orbitals.
    nvir : int
        Number of virtual spatial orbitals.
    excitation_energies : np.ndarray
        Excitation energies in Hartree, shape ``(n_states,)``, ascending.
    excitation_energies_ev : np.ndarray
        Excitation energies in eV.
    coefficients : list of np.ndarray
        ``coefficients[s]`` has length ``nocc*nvir + 1``:
        ``coefficients[s][0]`` is the HF ground-state amplitude (0 in TDA),
        and ``coefficients[s][1:]`` are the CIS amplitudes in the
        ``(i,a)`` ordering ``i + nocc*a`` (0-based indices).
    spin : str
        ``"singlet"`` or ``"triplet"``.
    converged : bool
        Always ``True`` when returned by :func:`run_cis` (exact diagonalization).
    """

    nocc: int
    nvir: int
    excitation_energies: np.ndarray
    excitation_energies_ev: np.ndarray
    coefficients: list[np.ndarray]
    spin: str
    converged: bool


# --------------------------------------------------------------------------- #
# 4-index MO integral: PQRS (pqrs.f)                                         #
# --------------------------------------------------------------------------- #


def _pqrs_mo(
    p: int,
    q: int,
    r: int,
    s: int,
    G_ao: np.ndarray,
    C: np.ndarray,
) -> float:
    """Single 4-index two-electron integral (pq|rs) in the MO basis for INDO.

    Ports MSINDO's ``PQRS`` (pqrs.f)."""
    nsto = G_ao.shape[0]

    # Separate Coulomb (upper triangle, symmetrized = REORG(1)) and exchange
    # (lower triangle, same-atom only).
    G_coul = np.triu(G_ao) + np.triu(G_ao, 1).T

    # Coulomb term: S_mu C_{mup}C_{muq} S_ν C_{νr}C_{νs} G_coul[ν,mu]
    coul = 0.0
    for mu in range(nsto):
        tmp = C[mu, p] * C[mu, q]
        if tmp == 0.0:
            continue
        for nu in range(nsto):
            coul += tmp * C[nu, r] * C[nu, s] * G_coul[nu, mu]

    # One-centre exchange (pqrs.f lines 38-49): for each atom, mu!=ν over BOTH
    # mu<ν and mu>ν.  The exchange integral X_{muν} sits at G_ao[max(mu,ν), min(mu,ν)]
    # (the lower triangle).
    xch = 0.0
    mu = 0
    while mu < nsto:
        lo = mu
        hi = mu + 1
        while hi < nsto and abs(G_ao[hi, lo]) > 1e-15:
            hi += 1
        for m1 in range(lo, hi):
            for m2 in range(lo, hi):
                if m1 == m2:
                    continue
                xchg = G_ao[max(m1, m2), min(m1, m2)]
                xch += (
                    C[m1, p]
                    * C[m2, q]
                    * (C[m1, r] * C[m2, s] + C[m2, r] * C[m1, s])
                    * xchg
                )
        mu = hi

    return coul + xch


def _pqrs_mo_vec(
    p: int,
    q: int,
    r: int,
    s: int,
    G_ao: np.ndarray,
    C: np.ndarray,
    blocks: list[tuple[int, int]],
) -> float:
    """Vectorized PQRS using NumPy -- equivalent to :func:`_pqrs_mo` but uses
    ``blocks`` for atom identification.

    Separates Coulomb (symmetrized upper triangle, the Fortran REORG(1)) from
    one-centre exchange (lower triangle, same-atom only).
    """
    G_coul = np.triu(G_ao) + np.triu(G_ao, 1).T
    cpq = C[:, p] * C[:, q]
    crs = C[:, r] * C[:, s]
    coul = float(cpq @ (G_coul @ crs))

    xch = 0.0
    for lo, hi in blocks:
        for mu in range(lo, hi):
            for nu in range(lo, hi):
                if mu == nu:
                    continue
                xchg = G_ao[max(mu, nu), min(mu, nu)]
                cpq_cross = C[mu, p] * C[nu, q]
                crs_cross = C[mu, r] * C[nu, s] + C[nu, r] * C[mu, s]
                xch += cpq_cross * crs_cross * xchg
    return coul + xch


# --------------------------------------------------------------------------- #
# MO-basis two-electron integrals + CPHF orbital Hessian                       #
# --------------------------------------------------------------------------- #


def _cis_mo_eri(
    G_ao: np.ndarray,
    C_mo: np.ndarray,
    blocks: list[tuple[int, int]],
) -> np.ndarray:
    """Full MO-basis two-electron tensor ``g[p,q,r,s] = (pq|rs)`` (chemist).

    Transforms the INDO AO tensor -- Coulomb monopole ``(mumu|νν)`` plus one-centre
    exchange ``(muν|muν)``, see
    :func:`vibeqc.semiempirical.methods.msindo._indo_ao_eri` -- into the MO basis.
    Used to assemble the CPHF orbital Hessian and the CIS Lagrangian for the
    analytic excited-state gradient.  Cost ``O(nsto^5)``; CIS gradients target
    small molecules.
    """
    from .msindo import _indo_ao_eri

    g_ao = _indo_ao_eri(G_ao, blocks)
    return np.einsum(
        "mnls,mp,nq,lr,su->pqru", g_ao, C_mo, C_mo, C_mo, C_mo, optimize=True
    )


def _cis_orbital_hessian(
    g_mo: np.ndarray,
    mo_eps: np.ndarray,
    nocc: int,
    nvir: int,
) -> np.ndarray:
    """Closed-shell RHF CPHF orbital Hessian -- the CIS Z-vector matrix ``A``.

    ``A[(a,i),(b,j)] = (e_a - e_i).d_ab.d_ij + 4(ia|jb) - (ij|ab) - (ib|ja)``,
    with rows/columns ordered ``(a outer, i inner)``.  This is the closed-shell
    reduction (a-a + a-b spin blocks) of MSINDO's RHF CPHF matrix-vector product
    -- see ``azcalc.f`` line 28: ``A_{ia,jb} = 4(ia|jb) - (ij|ab) - (ja|ib) +
    (e_a - e_i)d``.  The same orbital Hessian serves both singlet and triplet
    excited states; only the Lagrangian right-hand side differs (``cphf_solver.f``).
    """
    no = slice(0, nocc)
    nv = slice(nocc, nocc + nvir)
    ovov = g_mo[no, nv, no, nv]  # (i a | j b)
    oovv = g_mo[no, no, nv, nv]  # (i j | a b)
    A = (
        4.0 * np.einsum("iajb->aibj", ovov)
        - np.einsum("ijab->aibj", oovv)
        - np.einsum("ibja->aibj", ovov)  # (ib|ja)
    )
    for a in range(nvir):
        for i in range(nocc):
            A[a, i, a, i] += mo_eps[nocc + a] - mo_eps[i]
    return A.reshape(nvir * nocc, nvir * nocc)


# --------------------------------------------------------------------------- #
# CIS Lagrangian (cphf_solver.f / cislagrange_trip.f)                          #
# --------------------------------------------------------------------------- #


def _cis_lagrangian_mo(
    g_mo: np.ndarray,
    cis_coeffs: np.ndarray,
    nocc: int,
    nvir: int,
    spin: str,
) -> np.ndarray:
    """CIS Lagrangian ``L[a,i]`` -- RHS of the Z-vector equation ``A.Z = -L``.

    Explicit MO-basis form documented in MSINDO ``cphf_solver.f`` (lines 32-35),
    with ``X[i,a]`` the CIS amplitudes; ``a,b,c`` virtual, ``i,j,k`` occupied::

        L_ai =   S_jbc X_jb X_jc [4(ai|bc) - (ab|ic) - (ib|ac)]
               - S_jkb X_jb X_kb [4(ai|jk) - (aj|ik) - (ij|ak)]
               + 2 S_jbc X_ib X_jc [2.c.(jc|ab) - (ja|cb)]
               - 2 S_jkb X_ja X_kb [2.c.(kb|ij) - (kj|bi)]

    ``c = 1`` for singlets; ``c = 0`` for triplets (``cislagrange_trip.f`` drops
    the transition-density Coulomb terms, mirroring the triplet 2PDM dropping
    ``4.T_mm.T_nn``).  The result is ordered ``(a outer, i inner)``.

    The AO-factorized routines (``cis_lagrange.f``) were re-derived directly in
    the MO basis here -- far less error-prone than porting the BLAS-factorized
    one-centre/two-centre splitting.  Reference: Foresman, Head-Gordon, Pople &
    Frisch, J. Phys. Chem. 96, 135 (1992).
    """
    coul = 1.0 if spin == "singlet" else 0.0
    X = np.asarray(cis_coeffs)  # X[i, a]
    no = slice(0, nocc)
    nv = slice(nocc, nocc + nvir)
    VOVV = g_mo[nv, no, nv, nv]  # (a i | b c)
    VVOV = g_mo[nv, nv, no, nv]  # (a b | i c)
    OVVV = g_mo[no, nv, nv, nv]  # (i b | a c), (j c | a b), (j a | c b)
    VOOO = g_mo[nv, no, no, no]  # (a i | j k), (a j | i k)
    OOVO = g_mo[no, no, nv, no]  # (i j | a k), (k j | b i)
    OVOO = g_mo[no, nv, no, no]  # (k b | i j)
    Bvv = X.T @ X  # S_j X_jb X_jc
    Boo = X @ X.T  # S_b X_jb X_kb

    # Difference-density terms 1 & 2 (spin-independent -- same for singlet/triplet)
    L = 4.0 * np.einsum("aibc,bc->ai", VOVV, Bvv)
    L -= np.einsum("abic,bc->ai", VVOV, Bvv)
    L -= np.einsum("ibac,bc->ai", OVVV, Bvv)
    L -= 4.0 * np.einsum("aijk,jk->ai", VOOO, Boo)
    L += np.einsum("ajik,jk->ai", VOOO, Boo)
    L += np.einsum("ijak,jk->ai", OOVO, Boo)
    # Transition-density terms 3 & 4 (Coulomb scaled by ``coul``)
    Q1 = np.einsum("jc,jcab->ab", X, OVVV)  # S_jc X_jc (jc|ab)
    Q2 = np.einsum("jc,jacb->ab", X, OVVV)  # S_jc X_jc (ja|cb)
    L += 2.0 * (
        2.0 * coul * np.einsum("ib,ab->ai", X, Q1) - np.einsum("ib,ab->ai", X, Q2)
    )
    Q3 = np.einsum("kb,kbij->ij", X, OVOO)  # S_kb X_kb (kb|ij)
    Q4 = np.einsum("kb,kjbi->ij", X, OOVO)  # S_kb X_kb (kj|bi)
    L -= 2.0 * (
        2.0 * coul * np.einsum("ja,ij->ai", X, Q3) - np.einsum("ja,ij->ai", X, Q4)
    )
    return L


# --------------------------------------------------------------------------- #
#  Helper -- atom block inference                                             #
# --------------------------------------------------------------------------- #


def _infer_blocks(G_ao: np.ndarray) -> list[tuple[int, int]]:
    """Best-effort inference of atom AO blocks from INDO GMUNU.

    Walks the lower triangle looking for contiguous one-centre exchange
    (G[mu+1,mu] != 0 for mu in the current block).  Only works reliably when
    the matrix strictly respects the INDO convention (lower triangle = 0
    between different atoms).  ``vibeqc``'s ``_build_core_and_gamma``
    fills both upper and lower triangles with gamma for two-centre pairs,
    so this heuristic can merge separate atom blocks.  Prefer passing
    explicit ``blocks`` from :func:`vibeqc.semiempirical.methods.msindo._atom_blocks`.

    Parameters
    ----------
    G_ao : np.ndarray
        GMUNU (NSTO x NSTO).

    Returns
    -------
    list of (int, int)
        ``[(lo, hi), ...]`` with hi exclusive.
    """
    nsto = G_ao.shape[0]
    blocks = []
    mu = 0
    while mu < nsto:
        lo = mu
        hi = mu + 1
        while hi < nsto and abs(G_ao[hi, lo]) > 1e-15:
            hi += 1
        blocks.append((lo, hi))
        mu = hi
    return blocks


# =========================================================================== #
# CIS density matrices and 2PDM (cisgrad.f)                                   #
# =========================================================================== #


def _cis_coeffs_matrix(cis_result, state=0):
    """Extract (nocc, nvir) CIS amplitudes for a given state."""
    coeffs = np.asarray(cis_result.coefficients[state])
    nocc = cis_result.nocc
    nvir = cis_result.nvir
    return coeffs[1:].reshape(nocc, nvir)


def _build_t_matrix(cis_coeffs, C_alpha, nocc=None):
    """AO-basis CIS transition density ``T = C_occ @ A @ C_vir^T`` (one-sided).

    The one-particle transition density between the ground state and the CIS
    excited state, matching MSINDO ``cisgrad.f`` lines 137-141.  This is the
    object the symmetric 2PDM (:func:`_build_2pdm`) and the analytic gradient
    consume; it is intentionally *not* symmetrised -- ``trace(T) = 0`` already
    holds because the occupied and virtual MO spaces are orthogonal.
    """
    if nocc is None:
        nocc = np.asarray(cis_coeffs).shape[0]
    C_occ = C_alpha[:, :nocc]
    C_vir = C_alpha[:, nocc:]
    A = np.asarray(cis_coeffs)
    return C_occ @ A @ C_vir.T


def _build_dmat_oo_mo(cis_coeffs):
    """MO-basis D_oo: D_{ij} = -A @ A^T."""
    A = np.asarray(cis_coeffs)
    return -A @ A.T


def _build_dmat_vv_mo(cis_coeffs):
    """MO-basis D_vv: D_{ab} = A^T @ A."""
    A = np.asarray(cis_coeffs)
    return A.T @ A


def _build_dmat_oo(cis_coeffs, C_alpha, nocc=None):
    """AO-basis D_oo = C_occ @ D_oo^MO @ C_occ^T."""
    if nocc is None:
        nocc = np.asarray(cis_coeffs).shape[0]
    C_occ = C_alpha[:, :nocc]
    D_mo = _build_dmat_oo_mo(cis_coeffs)
    return C_occ @ D_mo @ C_occ.T


def _build_dmat_vv(cis_coeffs, C_alpha, nocc=None):
    """AO-basis D_vv = C_vir @ D_vv^MO @ C_vir^T."""
    if nocc is None:
        nocc = np.asarray(cis_coeffs).shape[0]
    C_vir = C_alpha[:, nocc:]
    D_mo = _build_dmat_vv_mo(cis_coeffs)
    return C_vir @ D_mo @ C_vir.T


def _build_dmat_unrelaxed(cis_coeffs, C_alpha, nocc=None):
    """Unrelaxed difference density (before Z-vector)."""
    return _build_dmat_oo(cis_coeffs, C_alpha, nocc) + _build_dmat_vv(
        cis_coeffs, C_alpha, nocc
    )


def _build_ground_density(C_alpha, nocc=None):
    """RHF ground-state density P = 2*C_occ*C_occ^T."""
    C_occ = C_alpha[:, :nocc]
    return 2.0 * C_occ @ C_occ.T


def _build_2pdm(
    T, P_gs, D_total, spin="singlet", scaled_cis=False, sca1=None, sca2=None
):
    """Fully symmetric 2PDM -- exact port of cisgrad.f lines 349-360.
    GAMM(MU,NU) = 0.5*(4*Tmm*Tnn - 2*Tmn^2 + 2*Dmm*Pnn - Dmn*Pmn + Pmm*Pnn - 0.5*Pmn^2)
    This is a FULLY symmetric matrix, NOT INDO convention. The gradient routines
    read GAMM directly (not upper/lower split)."""
    nsto = T.shape[0]
    G = np.zeros((nsto, nsto))
    s1 = sca1 if sca1 is not None else (0.4 if spin == "singlet" else 0.0)
    s2 = sca2 if sca2 is not None else (0.95 if spin == "singlet" else 1.0)
    for mu in range(nsto):
        for nu in range(nsto):
            if scaled_cis:  # Scaled CIS
                G[mu, nu] = 0.5 * (
                    4.0 * T[mu, mu] * T[nu, nu] * s1
                    - 2.0 * T[mu, nu] * T[mu, nu] * s2
                    + 2.0 * D_total[mu, mu] * P_gs[nu, nu]
                    - D_total[mu, nu] * P_gs[mu, nu]
                    + P_gs[mu, mu] * P_gs[nu, nu]
                    - 0.5 * P_gs[mu, nu] * P_gs[mu, nu]
                )
            elif spin == "singlet":
                G[mu, nu] = 0.5 * (
                    4.0 * T[mu, mu] * T[nu, nu]
                    - 2.0 * T[mu, nu] * T[mu, nu]
                    + 2.0 * D_total[mu, mu] * P_gs[nu, nu]
                    - D_total[mu, nu] * P_gs[mu, nu]
                    + P_gs[mu, mu] * P_gs[nu, nu]
                    - 0.5 * P_gs[mu, nu] * P_gs[mu, nu]
                )
            else:  # triplet
                G[mu, nu] = 0.5 * (
                    -2.0 * T[mu, nu] * T[mu, nu]
                    + 2.0 * D_total[mu, mu] * P_gs[nu, nu]
                    - D_total[mu, nu] * P_gs[mu, nu]
                    + P_gs[mu, mu] * P_gs[nu, nu]
                    - 0.5 * P_gs[mu, nu] * P_gs[mu, nu]
                )
    return G


def assemble_cis_densities(
    cis, state, C_mo, P_gs, spin="singlet", scaled_cis=False, sca1=None, sca2=None
):
    """Assemble CIS densities + 2PDM for one excited state (cisgrad.f).

    Convenience wrapper composing the individual density builders for state
    ``state``.  Accepts either the generic :class:`vibeqc.excited.CISResult`
    (``.amplitudes`` / ``.n_occ`` / ``.n_vir``) or this module's
    :class:`CISResult` (``.coefficients`` / ``.nocc`` / ``.nvir``).

    Returns a dict with the AO-basis transition density ``T``, the occ-occ and
    vir-vir difference densities ``D_oo`` / ``D_vv``, their sum ``D_unrelaxed``,
    and the symmetric 2PDM ``GAMM`` (per :func:`_build_2pdm`).
    """
    if hasattr(cis, "amplitudes"):
        cismat = np.asarray(cis.amplitudes)[:, state].reshape(cis.n_occ, cis.n_vir)
    else:
        cismat = _cis_coeffs_matrix(cis, state)
    T = _build_t_matrix(cismat, C_mo)
    D_oo = _build_dmat_oo(cismat, C_mo)
    D_vv = _build_dmat_vv(cismat, C_mo)
    D_unrelaxed = _build_dmat_unrelaxed(cismat, C_mo)
    GAMM = _build_2pdm(
        T, P_gs, D_unrelaxed, spin=spin, scaled_cis=scaled_cis, sca1=sca1, sca2=sca2
    )
    return {
        "T": T,
        "D_oo": D_oo,
        "D_vv": D_vv,
        "D_unrelaxed": D_unrelaxed,
        "GAMM": GAMM,
    }


# =========================================================================== #
# CIS gradient driver (cisgrad.f)                                             #
# =========================================================================== #


def cis_gradient(
    Z,
    coords_angstrom,
    cis_result,
    state=0,
    *,
    spin="singlet",
    max_iter=200,
    conv_tol=1e-10,
):
    """Analytic CIS/TDA excited-state nuclear gradient (Ha/bohr).

    Gradient of the total excited-state energy ``E_GS + w_state`` via the
    relaxed one-particle density and the CIS two-particle density (2PDM),
    contracted with the per-pair integral derivatives -- the same machinery as
    the ground-state gradient (:func:`...msindo_gradient_analytic`), with three
    CIS-specific pieces:

    * The **Z-vector (CPHF)** relaxed-density occ-vir block, from the closed-shell
      orbital Hessian (:func:`_cis_orbital_hessian`) and the CIS Lagrangian
      (:func:`_cis_lagrangian_mo`): ``A.Z = -L`` (``azcalc.f`` / ``cphf_solver.f``).
    * The one-sided transition density ``T`` and the symmetric 2PDM
      (:func:`_build_2pdm`) -- ``cisgrad.f``.
    * A faithful **KLOOP + LLOOP** pair-block contraction: because the CIS 2PDM
      (and relaxed density) are *not* symmetric, the pair block uses
      ``GAMM[L_j,K_i] + GAMM[K_i,L_j]`` (and the analogous ``PMAT`` sum), not the
      ``2x`` shortcut valid only for the symmetric ground-state density
      (``kloop.f`` / ``lloop.f``).

    Matches :func:`cis_gradient_fd` to ~1e-6 Ha/bohr for non-degenerate states
    (singlet + triplet); near degeneracies the per-state gradient is ill-defined
    and :func:`cis_gradient_fd` is the documented fallback.  Closed-shell (RHF)
    only.  Reference: Foresman, Head-Gordon, Pople & Frisch,
    J. Phys. Chem. 96, 135 (1992).
    """
    import math

    from . import msindo_pair_deriv as _pd
    from .msindo import (
        ANGSTROM_TO_BOHR,
        _atom_blocks,
        _build_core_and_gamma,
        _build_fock,
        _scf_rhf,
        eff_core_charge,
    )

    C0 = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    natom = len(Z)
    blocks_list, nsto = _atom_blocks(Z)
    nelec = sum(eff_core_charge(z) for z in Z)
    if nelec % 2 != 0:
        raise NotImplementedError(
            "Analytic CIS gradient supports closed-shell (RHF) references only."
        )
    nocc = nelec // 2
    nvir = nsto - nocc

    # Ground-state SCF
    H0, G0 = _build_core_and_gamma(Z, C0, blocks_list, nsto)
    P_gs, F_gs, e_elec, eps_scf, converged, it = _scf_rhf(
        H0, G0, blocks_list, Z, nocc, max_iter=max_iter, conv_tol=conv_tol
    )
    if not converged:
        raise RuntimeError("Ground-state SCF not converged; cannot form CIS gradient.")
    F_final = _build_fock(H0, G0, P_gs, blocks_list, Z)
    mo_eps, C_mo = np.linalg.eigh(F_final)
    C_occ = C_mo[:, :nocc]
    C_vir = C_mo[:, nocc:]

    X = _cis_coeffs_matrix(cis_result, state)  # X[i, a]
    blocks = [(lo, hi) for lo, hi in blocks_list]

    # --- Z-vector (orbital relaxation): A.Z = -L, ordered (a outer, i inner) ---
    g_mo = _cis_mo_eri(G0, C_mo, blocks)
    A = _cis_orbital_hessian(g_mo, mo_eps, nocc, nvir)
    L = _cis_lagrangian_mo(g_mo, X, nocc, nvir, spin)
    Z_vec = np.linalg.solve(A, -L.reshape(-1))

    # --- Relaxed difference density + transition density (AO basis) ---
    # Z occ-vir block added one-sided, matching cphf_solver.f:245-248 (CIS).
    D_oo = _build_dmat_oo(X, C_mo, nocc)
    D_vv = _build_dmat_vv(X, C_mo, nocc)
    D_ov = C_vir @ Z_vec.reshape(nvir, nocc) @ C_occ.T
    D_total = D_oo + D_vv + D_ov
    T_mat = C_occ @ X @ C_vir.T  # one-sided transition density (cisgrad.f)

    PMAT = P_gs + D_total
    GAMM = _build_2pdm(T_mat, P_gs, D_total, spin=spin)

    grad = np.zeros((natom, 3))
    for k in range(natom):
        for l in range(k + 1, natom):
            rk, rl = C0[k], C0[l]
            dvec = rl - rk
            R = float(np.linalg.norm(dvec))
            if R < 1e-12:
                continue

            pd = _pd._pair_blocks_deriv(Z[k], Z[l], rk, rl)

            # Chain rule d(R,th,phi)/d(r_l - r_k)
            E = dvec / R
            cost = E[2]
            sint = math.sqrt(max(0.0, 1.0 - E[2] ** 2))
            if sint < 1e-12:
                cosphi, sinphi = 1.0, 0.0
                dphidx = dphidy = 0.0
            else:
                cosphi, sinphi = E[0] / sint, E[1] / sint
                c1 = sint * sint * R
                dphidx = -E[1] / c1
                dphidy = E[0] / c1
            edt1 = cost * cosphi
            edt2 = cost * sinphi
            edt3 = -sint
            drdx, drdy, drdz = E[0], E[1], E[2]
            dthx = edt1 / R
            dthy = edt2 / R
            dthz = edt3 / R

            lo_k, hi_k = blocks_list[k]
            lo_l, hi_l = blocks_list[l]
            nk = hi_k - lo_k
            nl = hi_l - lo_l
            P_kk = PMAT[lo_k:hi_k, lo_k:hi_k]
            P_ll = PMAT[lo_l:hi_l, lo_l:hi_l]

            dx = dy = dz = 0.0

            # K-atom 1-electron (diagonal + same-atom off-diagonal).
            # Off-diagonal symmetrised as P[j,i]+P[i,j] (KLOOP PKK(J,I)+PKK(I,J))
            # since the relaxed density need not be symmetric.
            for i in range(nk):
                dx += P_kk[i, i] * (
                    pd.HK1_DR[i, i] * drdx + pd.HK1_DT[i, i] * dthx + pd.HK1_DP[i, i] * dphidx
                )
                dy += P_kk[i, i] * (
                    pd.HK1_DR[i, i] * drdy + pd.HK1_DT[i, i] * dthy + pd.HK1_DP[i, i] * dphidy
                )
                dz += P_kk[i, i] * (pd.HK1_DR[i, i] * drdz + pd.HK1_DT[i, i] * dthz)
                for j in range(i + 1, nk):
                    Pij = P_kk[j, i] + P_kk[i, j]
                    dx += Pij * (
                        pd.HK1_DR[j, i] * drdx + pd.HK1_DT[j, i] * dthx + pd.HK1_DP[j, i] * dphidx
                    )
                    dy += Pij * (
                        pd.HK1_DR[j, i] * drdy + pd.HK1_DT[j, i] * dthy + pd.HK1_DP[j, i] * dphidy
                    )
                    dz += Pij * (pd.HK1_DR[j, i] * drdz + pd.HK1_DT[j, i] * dthz)

            # L-atom 1-electron
            for i in range(nl):
                dx += P_ll[i, i] * (
                    pd.HL1_DR[i, i] * drdx + pd.HL1_DT[i, i] * dthx + pd.HL1_DP[i, i] * dphidx
                )
                dy += P_ll[i, i] * (
                    pd.HL1_DR[i, i] * drdy + pd.HL1_DT[i, i] * dthy + pd.HL1_DP[i, i] * dphidy
                )
                dz += P_ll[i, i] * (pd.HL1_DR[i, i] * drdz + pd.HL1_DT[i, i] * dthz)
                for j in range(i + 1, nl):
                    Pij = P_ll[j, i] + P_ll[i, j]
                    dx += Pij * (
                        pd.HL1_DR[j, i] * drdx + pd.HL1_DT[j, i] * dthx + pd.HL1_DP[j, i] * dphidx
                    )
                    dy += Pij * (
                        pd.HL1_DR[j, i] * drdy + pd.HL1_DT[j, i] * dthy + pd.HL1_DP[j, i] * dphidy
                    )
                    dz += Pij * (pd.HL1_DR[j, i] * drdz + pd.HL1_DT[j, i] * dthz)

            # Pair block: KLOOP uses [L_j,K_i], LLOOP the transpose [K_i,L_j].
            # Their sum (NOT 2x[L_j,K_i]) is required for the non-symmetric CIS
            # 2PDM / relaxed density (kloop.f / lloop.f).
            for i in range(nk):
                for j in range(nl):
                    Psum = PMAT[lo_l + j, lo_k + i] + PMAT[lo_k + i, lo_l + j]
                    Gsum = GAMM[lo_l + j, lo_k + i] + GAMM[lo_k + i, lo_l + j]
                    F_gam = pd.d_gamma_local[i, j] * Gsum
                    F_R = Psum * pd.HKL2_DR[i, j]
                    dx += (F_gam + F_R) * drdx + Psum * (
                        pd.HKL2_DT[i, j] * dthx + pd.HKL2_DP[i, j] * dphidx
                    )
                    dy += (F_gam + F_R) * drdy + Psum * (
                        pd.HKL2_DT[i, j] * dthy + pd.HKL2_DP[i, j] * dphidy
                    )
                    dz += (F_gam + F_R) * drdz + Psum * (pd.HKL2_DT[i, j] * dthz)

            # Nuclear repulsion (2x from KLOOP + LLOOP)
            zke = float(eff_core_charge(Z[k]))
            zle = float(eff_core_charge(Z[l]))
            dnuc = 2.0 * 0.5 * zke * zle * (-1.0 / (R * R))
            dx += dnuc * drdx
            dy += dnuc * drdy
            dz += dnuc * drdz

            # DEDXYZK convention: EDX(L) += DX, EDX(K) -= DX
            grad[l, 0] += dx
            grad[l, 1] += dy
            grad[l, 2] += dz
            grad[k, 0] -= dx
            grad[k, 1] -= dy
            grad[k, 2] -= dz

    return grad


# =========================================================================== #
# CIS excitation solver                                                       #
# =========================================================================== #
# (``CISResult`` is defined once near the top of this module.)


def _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, spin="singlet"):
    """Build CIS/TDA Hamiltonian H_{ia,jb}.

    Singlet: H_{ia,jb} = (e_a - e_i)d_{ij}d_{ab} + 2*(ia|jb) - (ij|ab)
    Triplet: H_{ia,jb} = (e_a - e_i)d_{ij}d_{ab} - (ij|ab)
    """
    nvir = ovov.shape[1]
    ndet = nocc * nvir
    H = np.zeros((ndet, ndet))

    for i in range(nocc):
        for a in range(nvir):
            ia = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    jb = j * nvir + b
                    if i == j and a == b:
                        H[ia, jb] = mo_eps[nocc + a] - mo_eps[i]
                    if spin == "singlet":
                        H[ia, jb] += 2.0 * ovov[i, a, j, b] - oovv[i, j, a, b]
                    else:
                        H[ia, jb] += -oovv[i, j, a, b]
    return H


def _build_mo_integrals(C, nocc, G_ao, blocks):
    """Build (ia|jb) and (ij|ab) MO integrals using _pqrs_mo_vec."""
    nsto = C.shape[0]
    nvir = nsto - nocc
    ovov = np.zeros((nocc, nvir, nocc, nvir))
    oovv = np.zeros((nocc, nocc, nvir, nvir))
    for i in range(nocc):
        for a in range(nvir):
            ai = nocc + a
            for j in range(nocc):
                for b in range(nvir):
                    bj = nocc + b
                    ovov[i, a, j, b] = _pqrs_mo_vec(i, ai, j, bj, G_ao, C, blocks)
                    oovv[i, j, a, b] = _pqrs_mo_vec(i, j, ai, bj, G_ao, C, blocks)
    return ovov, oovv


def run_cis(
    Z,
    coords_angstrom,
    n_states=3,
    spin="singlet",
    max_iter=200,
    conv_tol=1e-10,
):
    """Run a CIS/TDA calculation for MSINDO/INDO.

    Parameters
    ----------
    Z : list of int
        Atomic numbers.
    coords_angstrom : (natom, 3) array
        Atomic positions in Angstrom.
    n_states : int
        Number of excited states to compute.
    spin : str
        'singlet' or 'triplet'.

    Returns
    -------
    CISResult
    """
    from .msindo import (
        ANGSTROM_TO_BOHR,
        _atom_blocks,
        _build_core_and_gamma,
        _build_fock,
        _scf_rhf,
        eff_core_charge,
    )

    C0 = np.asarray(coords_angstrom, float) * ANGSTROM_TO_BOHR
    natom = len(Z)

    blocks_list, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz)
    if nelec % 2 != 0:
        raise NotImplementedError("CIS supports closed-shell only.")
    nocc = nelec // 2
    nvir = nsto - nocc

    # SCF
    H0, G0 = _build_core_and_gamma(Z, C0, blocks_list, nsto)
    P_gs, F_gs, e_elec, eps_scf, converged, it = _scf_rhf(
        H0,
        G0,
        blocks_list,
        Z,
        nocc,
        max_iter=max_iter,
        conv_tol=conv_tol,
    )
    if not converged:
        return CISResult(
            nocc=nocc,
            nvir=nvir,
            excitation_energies=np.array([]),
            excitation_energies_ev=np.array([]),
            coefficients=[],
            spin=spin,
            converged=False,
        )

    F_final = _build_fock(H0, G0, P_gs, blocks_list, Z)
    mo_eps, C_mo = np.linalg.eigh(F_final)

    blocks = [(lo, hi) for lo, hi in blocks_list]
    ovov, oovv = _build_mo_integrals(C_mo, nocc, G0, blocks)

    H_cis = _build_cis_hamiltonian(mo_eps, nocc, ovov, oovv, spin)
    eigvals, eigvecs = np.linalg.eigh(H_cis)

    n_states = min(n_states, len(eigvals))
    exc_energies = eigvals[:n_states]
    exc_energies_ev = exc_energies * 27.211386245988

    coefficients = []
    for s in range(n_states):
        coeff_flat = np.zeros(nocc * nvir + 1)
        coeff_flat[0] = 0.0  # HF reference coefficient
        coeff_flat[1:] = eigvecs[:, s]
        coefficients.append(coeff_flat)

    return CISResult(
        nocc=nocc,
        nvir=nvir,
        excitation_energies=exc_energies,
        excitation_energies_ev=exc_energies_ev,
        coefficients=coefficients,
        spin=spin,
        converged=True,
    )


# =========================================================================== #
# Validated FD-based CIS gradient (workaround for 10.43x analytic scaling)    #
# =========================================================================== #




def cis_gradient_fd(Z, coords_angstrom, state=0, spin="singlet", step=1e-3,
                    max_iter=200, conv_tol=1e-10, n_states=None):
    """Finite-difference CIS excited-state gradient (Ha/bohr)."""
    import numpy as np
    from .msindo import ANGSTROM_TO_BOHR, _atom_blocks, _build_core_and_gamma, _scf_rhf, eff_core_charge
    if n_states is None:
        n_states = max(state + 3, 3)
    natom = len(Z)
    h_bohr = step * ANGSTROM_TO_BOHR

    def energy(coord_list):
        C0 = np.asarray(coord_list) * ANGSTROM_TO_BOHR
        bl, nsto = _atom_blocks(Z)
        cz = [eff_core_charge(z) for z in Z]
        nocc = sum(cz) // 2
        H, G = _build_core_and_gamma(Z, C0, bl, nsto)
        P, _, ee, _, _, _ = _scf_rhf(H, G, bl, Z, nocc, max_iter=max_iter, conv_tol=conv_tol)
        en = sum(cz[k]*cz[l]/np.linalg.norm(C0[k]-C0[l]) for k in range(natom) for l in range(k+1, natom))
        cis = run_cis(Z, coord_list, n_states=n_states, spin=spin, max_iter=max_iter, conv_tol=conv_tol)
        return ee + en + cis.excitation_energies[state]

    grad = np.zeros((natom, 3))
    coords = np.asarray(coords_angstrom, float)
    for i in range(natom):
        for d in range(3):
            cp, cm = coords.copy(), coords.copy()
            cp[i, d] += step; cm[i, d] -= step
            grad[i, d] = (energy(cp) - energy(cm)) / (2.0 * h_bohr)
    return grad
