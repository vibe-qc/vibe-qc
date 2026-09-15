"""CCM analytic nuclear gradient -- Phase 5.

Extends the molecular analytic gradient to the Cyclic Cluster Model (CCM)
by Wigner-Seitz weighting of pair contributions.  Includes analytic
Madelung/Ewald gradient (ported from dedmadelsum.f + dmadkonst.f).

Fortran reference: intdxdydz.f, dedmadelsum.f, dmadkonst.f.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import erf, erfc

from . import msindo_pair_deriv as _pd
from .msindo import (
    ANGSTROM_TO_BOHR,
    _atom_blocks,
    _scf_rhf,
    eff_core_charge,
)
from .msindo_ccm import (
    WignerSeitzCells,
    _build_core_and_gamma_ccm,
    _core_repulsion_ccm,
    _ewald_lattice_2d,
    _ewald_lattice_3d,
    _ewald_ws_cells,
    _madelung_potential_1d,
    _madelung_potential_ewald,
    _madkonst_2d,
    _madkonst_3d,
    _net_charges,
    _parry_recip_pair,
)


@lru_cache(maxsize=1)
def _cpp_ccm_gradient_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    try:
        ccm_gradient_analytic_cpp = _indo.ccm_gradient_analytic
        load_params_from_json = _indo.load_params_from_json
    except AttributeError:
        return None
    params = load_params_from_json(
        Path(__file__).with_name("msindo_params.json").read_text()
    )
    return ccm_gradient_analytic_cpp, params


def ccm_gradient_analytic(
    atomic_numbers,
    coords_angstrom,
    translations_angstrom,
    *,
    charge=0,
    madelung=False,
    max_iter=200,
    conv_tol=1e-10,
):
    """Analytic nuclear gradient (Ha/bohr) for the CCM total energy.

    The Wigner-Seitz topology is built once at the reference geometry and held
    FIXED while the gradient is assembled -- same as ccm_gradient_fd.

    Parameters
    ----------
    atomic_numbers : list of int
    coords_angstrom : (natom, 3) array-like
    translations_angstrom : list of (3,) array-like -- lattice vectors
    madelung : bool -- if True, includes the analytic Madelung/Ewald gradient.

    Returns
    -------
    (natom, 3) ndarray -- gradient in Ha/bohr
    """
    Z = list(atomic_numbers)
    coords_arr = np.asarray(coords_angstrom, float)
    translations_arr = [np.asarray(t, float) for t in translations_angstrom]
    C0 = coords_arr * ANGSTROM_TO_BOHR
    T = [t * ANGSTROM_TO_BOHR for t in translations_arr]
    natom = len(Z)

    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec % 2 != 0:
        raise NotImplementedError("CCM analytic gradient supports closed-shell only.")
    nocc = nelec // 2

    # Build WS topology at reference geometry (FIXED WEIGHTS)
    from .msindo_ccm import build_wigner_seitz

    ws0 = build_wigner_seitz(C0, T)
    if not ws0.is_valid(len(Z)):
        raise ValueError("not a valid cyclic cluster (see run_ccm).")

    if all(1 <= z <= 36 for z in Z):
        kernel = _cpp_ccm_gradient_kernel()
        if kernel is not None:
            ccm_gradient_analytic_cpp, params = kernel
            grad_cpp = ccm_gradient_analytic_cpp(
                Z,
                coords_arr.tolist(),
                [t.tolist() for t in translations_arr],
                params,
                madelung=madelung,
                charge=charge,
                max_iter=max_iter,
                conv_tol=conv_tol,
            )
            if len(grad_cpp) == 0 and len(Z) != 0:
                raise RuntimeError(
                    "SCF not converged; cannot compute analytic gradient."
                )
            return np.asarray(grad_cpp, dtype=float)

    # Fixed lattice translation per WS neighbour
    lattices = [
        [(nb.origin, nb.weight, nb.disp + C0[i] - C0[nb.origin]) for nb in cell]
        for i, cell in enumerate(ws0.cells)
    ]

    class _NB:
        __slots__ = ("origin", "weight", "disp")

        def __init__(self, o, w, d):
            self.origin = o
            self.weight = w
            self.disp = d

    cells2 = [
        [_NB(o, w, C0[o] + t - C0[i]) for (o, w, t) in lat]
        for i, lat in enumerate(lattices)
    ]
    ws = WignerSeitzCells(cells=cells2, translations=T)

    # SCF with optional Madelung
    H, G = _build_core_and_gamma_ccm(Z, C0, blocks, nsto, ws)
    e_core = _core_repulsion_ccm(Z, C0, cz, ws)

    fock_extra = None
    if madelung:
        ews = _ewald_ws_cells(ws)
        if len(T) == 1:

            def _pot1d(net):
                return _madelung_potential_1d(net, ews, T[0])

            mad_potential = _pot1d
        elif len(T) in (2, 3):
            madkonst = (
                _madkonst_2d(ews, T, len(Z))
                if len(T) == 2
                else _madkonst_3d(ews, T, len(Z))
            )

            def _pot_ewald(net):
                return _madelung_potential_ewald(net, madkonst, ews)

            mad_potential = _pot_ewald

        def _fock_extra(P):
            net = _net_charges(P, blocks, cz)
            mad = mad_potential(net)
            F = np.zeros((nsto, nsto))
            for i, (lo, hi) in enumerate(blocks):
                for k in range(lo, hi):
                    F[k, k] = -mad[i]
            e = 0.5 * float(np.dot(cz, mad))
            return F, e

        fock_extra = _fock_extra

    P, _F, e_elec, eps, converged, it = _scf_rhf(
        H,
        G,
        blocks,
        Z,
        nocc,
        max_iter=max_iter,
        conv_tol=conv_tol,
        fock_extra=fock_extra,
    )
    if not converged:
        raise RuntimeError("SCF not converged; cannot compute analytic gradient.")

    grad = np.zeros((natom, 3))

    # Loop over central atoms and their WS neighbor images
    for k in range(natom):
        lk, hk = blocks[k]
        nk = hk - lk
        P_kk = P[lk:hk, lk:hk]
        for nb in ws.cells[k]:
            j = nb.origin
            w = nb.weight
            lj, hj = blocks[j]
            nj = hj - lj

            # Compute pair_blocks_deriv with the IMAGE position
            pd = _pd._pair_blocks_deriv(Z[k], Z[j], C0[k], C0[k] + nb.disp)
            P_ll = P[lj:hj, lj:hj]
            P_kl = P[lk:hk, lj:hj]

            R = pd.R
            dvec = nb.disp  # = rl - rk = image_pos - C0[k]
            E = dvec / R
            cost, sint = E[2], math.sqrt(max(0.0, 1.0 - E[2] ** 2))
            if sint < 1e-12:
                cosphi, sinphi = 1.0, 0.0
                dphidx, dphidy, dphidz = 0.0, 0.0, 0.0
            else:
                cosphi, sinphi = E[0] / sint, E[1] / sint
                c1 = sint * sint * R
                dphidx = -E[1] / c1
                dphidy = E[0] / c1
                dphidz = 0.0
            edt1 = cost * cosphi
            edt2 = cost * sinphi
            edt3 = -sint
            drdx, drdy, drdz = E[0], E[1], E[2]
            dthx = edt1 / R
            dthy = edt2 / R
            dthz = edt3 / R

            dx, dy, dz = 0.0, 0.0, 0.0

            # --- CCMDEDXYZK: K-diagonal + off-diagonal ---
            for i in range(nk):
                Pii = P_kk[i, i]
                dx += Pii * (
                    pd.HK1_DR[i, i] * drdx
                    + pd.HK1_DT[i, i] * dthx
                    + pd.HK1_DP[i, i] * dphidx
                )
                dy += Pii * (
                    pd.HK1_DR[i, i] * drdy
                    + pd.HK1_DT[i, i] * dthy
                    + pd.HK1_DP[i, i] * dphidy
                )
                dz += Pii * (
                    pd.HK1_DR[i, i] * drdz
                    + pd.HK1_DT[i, i] * dthz
                    + pd.HK1_DP[i, i] * dphidz
                )
                for ii in range(i + 1, nk):
                    Pij = P_kk[ii, i]
                    fac = 2.0 * Pij
                    dx += fac * (
                        pd.HK1_DR[ii, i] * drdx
                        + pd.HK1_DT[ii, i] * dthx
                        + pd.HK1_DP[ii, i] * dphidx
                    )
                    dy += fac * (
                        pd.HK1_DR[ii, i] * drdy
                        + pd.HK1_DT[ii, i] * dthy
                        + pd.HK1_DP[ii, i] * dphidy
                    )
                    dz += fac * (
                        pd.HK1_DR[ii, i] * drdz
                        + pd.HK1_DT[ii, i] * dthz
                        + pd.HK1_DP[ii, i] * dphidz
                    )

            # --- CCMDEDXYZL: J-diagonal + off-diagonal ---
            for i in range(nj):
                Pii = P_ll[i, i]
                dx += Pii * (
                    pd.HL1_DR[i, i] * drdx
                    + pd.HL1_DT[i, i] * dthx
                    + pd.HL1_DP[i, i] * dphidx
                )
                dy += Pii * (
                    pd.HL1_DR[i, i] * drdy
                    + pd.HL1_DT[i, i] * dthy
                    + pd.HL1_DP[i, i] * dphidy
                )
                dz += Pii * (
                    pd.HL1_DR[i, i] * drdz
                    + pd.HL1_DT[i, i] * dthz
                    + pd.HL1_DP[i, i] * dphidz
                )
                for ii in range(i + 1, nj):
                    Pij = P_ll[ii, i]
                    fac = 2.0 * Pij
                    dx += fac * (
                        pd.HL1_DR[ii, i] * drdx
                        + pd.HL1_DT[ii, i] * dthx
                        + pd.HL1_DP[ii, i] * dphidx
                    )
                    dy += fac * (
                        pd.HL1_DR[ii, i] * drdy
                        + pd.HL1_DT[ii, i] * dthy
                        + pd.HL1_DP[ii, i] * dphidy
                    )
                    dz += fac * (
                        pd.HL1_DR[ii, i] * drdz
                        + pd.HL1_DT[ii, i] * dthz
                        + pd.HL1_DP[ii, i] * dphidz
                    )

            # --- CCMDEDXYZK: pair block + gamma + nuclear (round 1) ---
            for i in range(nk):
                Pii = P_kk[i, i]
                for jj in range(nj):
                    Pjj = P_ll[jj, jj]
                    Pij = P_kl[i, jj]
                    Fg = 0.5 * pd.d_gamma_local[i, jj] * (Pii * Pjj - 0.5 * Pij * Pij)
                    Fr = Pij * pd.HKL2_DR[i, jj]
                    Ft = Pij * pd.HKL2_DT[i, jj]
                    Fp = Pij * pd.HKL2_DP[i, jj]
                    dx += (Fg + Fr) * drdx + Ft * dthx + Fp * dphidx
                    dy += (Fg + Fr) * drdy + Ft * dthy + Fp * dphidy
                    dz += (Fg + Fr) * drdz + Ft * dthz + Fp * dphidz
            # Nuclear repulsion (round 1)
            zke = float(eff_core_charge(Z[k]))
            zle = float(eff_core_charge(Z[j]))
            zkl = 0.5 * zke * zle
            drkl = -1.0 / (R * R)
            dx += zkl * drkl * drdx
            dy += zkl * drkl * drdy
            dz += zkl * drkl * drdz

            # --- CCMDEDXYZL: pair block + gamma + nuclear (round 2) ---
            for i in range(nk):
                Pii = P_kk[i, i]
                for jj in range(nj):
                    Pjj = P_ll[jj, jj]
                    Pij = P_kl[i, jj]
                    Fg = 0.5 * pd.d_gamma_local[i, jj] * (Pii * Pjj - 0.5 * Pij * Pij)
                    Fr = Pij * pd.HKL2_DR[i, jj]
                    Ft = Pij * pd.HKL2_DT[i, jj]
                    Fp = Pij * pd.HKL2_DP[i, jj]
                    dx += (Fg + Fr) * drdx + Ft * dthx + Fp * dphidx
                    dy += (Fg + Fr) * drdy + Ft * dthy + Fp * dphidy
                    dz += (Fg + Fr) * drdz + Ft * dthz + Fp * dphidz
            # Nuclear repulsion (round 2)
            dx += zkl * drkl * drdx
            dy += zkl * drkl * drdy
            dz += zkl * drkl * drdz

            # Total force: -(CCMDEDXYZK + CCMDEDXYZL), both applied to K
            # EDX(K) = EDX(K) - (DX_K + DX_L)*WEIGHT
            grad[k, 0] -= w * dx
            grad[k, 1] -= w * dy
            grad[k, 2] -= w * dz

    # Madelung/Ewald analytic gradient (ported from dedmadelsum.f + dmadkonst.f)
    if madelung:
        net = _net_charges(P, blocks, cz)
        _add_madelung_gradient(grad, net, C0, T, ws)

    return grad


# ============================================================================ #
# Analytic Madelung/Ewald gradient (dmadkonst.f + dedmadelsum.f)               #
# ============================================================================ #


def _scipy_erfc(x):
    """erfc(x) = 1 - erf(x)."""
    from math import erfc as _erfc

    return np.vectorize(_erfc)(x)


# Both the 2-D and 3-D paths share their vector inventory with the energy
# (msindo_ccm._ewald_lattice_2d / _ewald_lattice_3d), so the gradient is the
# exact derivative of the energy that was actually summed and inherits its
# metric-complete, basis-invariant real/reciprocal cutoffs.


def _add_madelung_gradient(grad, net_charges, C0, translations, ws):
    """Add the Madelung/Ewald gradient (Ha/bohr) to ``grad`` in place.

    Port of ``dedmadelsum.f``.  At the fixed converged density the geometry-
    dependent Madelung energy is

        e_mad = 1/2 S_I q_I V_I,   V_I = S_{nbinWS(I)} w_{nb} q_{o(nb)} h(disp_{nb}),

    with ``disp = image - C[I]`` and per-neighbour element ``h``:

    * 1-D (``ccm1dmadelsum.f``): ``h(disp) = S_{shin{±T,±2T}} 1/|disp+sh|`` -- the
      finite point-charge lattice sum (the in-cell ``1/|disp|`` is already in the
      INDO two-centre g, so it is NOT in e_mad).
    * 2-D / 3-D (``madelkonst.f``): ``h(disp) = M(v) - 1/d``, ``v = -disp``,
      ``d = |disp|``, with ``M`` the Parry/Heyes (2-D) or Ewald (3-D) Madelung
      element and ``-1/d`` the SMADEL sphere subtraction.

    Because ``disp_{nb} = C[o(nb)] + t - C[I]`` (lattice ``t`` fixed), the exact
    fixed-charge derivative is the *direct* assembly

        de_mad/dR_a = 1/2 S_I q_I S_{nbinWS(I)} w q_o gradh(disp).(d_{o(nb),a} - d_{I,a}),

    i.e. each neighbour pushes ``+1/2 q_I q_o w gradh`` onto its origin and
    ``-1/2`` onto the central atom.  This is used in place of the central-atom-only
    ``grad[I] += q_I S w q_o gradh`` form of ``dedmadelsum.f`` (which silently
    assumes a mirror-symmetric WS set, ``EDX(IATOM) += DMADEL.q_I``,
    ``dedmadelsum.f:171-173``): the truncated 1-D lattice sum and the
    distorted-cell WS weights break that symmetry, whereas the direct assembly is
    the exact derivative of the SCF energy for every cluster.

    The self image (``disp = 0``) carries a geometry-independent ``M(0)`` /
    ``S 1/|sh|`` and so ``gradh(0)=0`` -- it drops out, hence ``ws.cells[i]`` (no
    self term).
    """
    natom = len(C0)
    dim = len(translations)

    if dim == 1:
        # gradh(disp) = grad_disp S_sh 1/|disp+sh| = -S_sh (disp+sh)/|disp+sh|^3.
        Tv = np.asarray(translations[0], float)
        shells = (Tv, -Tv, 2.0 * Tv, -2.0 * Tv)

        def grad_h(disp, d):
            acc = np.zeros(3)
            for sh in shells:
                u = disp + sh
                r = float(np.linalg.norm(u))
                if r > 1e-12:
                    acc -= u / r**3
            return acc

    elif dim == 2:
        # 2-D Parry/Heyes slab.  gradh(disp) = -dM/dv + disp/d^3 (v = -disp).
        ews = _ewald_ws_cells(ws)
        kvecs, kmag, dvecs, alpha, area, nhat, direct_cutoff = _ewald_lattice_2d(
            ews, translations
        )

        def grad_h(disp, d):
            gh = -_dmadkonst_2d(
                -disp, kvecs, kmag, dvecs, alpha, area, nhat, direct_cutoff
            )
            if d > 1e-12:
                gh += disp / d**3  # grad_disp(-1/|disp|) = +disp/d^3 (SMADEL)
            return gh

    else:  # dim == 3
        # 3-D Ewald.  gradh(disp) = -dM/dv + disp/d^3 (v = -disp).
        ews = _ewald_ws_cells(ws)
        kvecs, kb, dvecs, alpha, volume, direct_cutoff = _ewald_lattice_3d(
            ews, translations
        )

        def grad_h(disp, d):
            gh = -_dmadkonst_3d(
                -disp,
                kvecs,
                kb,
                dvecs,
                alpha,
                volume,
                direct_cutoff,
            )
            if d > 1e-12:
                gh += disp / d**3  # SMADEL
            return gh

    # Direct assembly: each neighbour contributes ±1/2 q_I q_o w gradh(disp).
    for i in range(natom):
        qi = float(net_charges[i])
        if qi == 0.0:
            continue
        for nb in ws.cells[i]:
            o = nb.origin
            qo = float(net_charges[o])
            if qo == 0.0:
                continue
            d = float(np.linalg.norm(nb.disp))
            term = 0.5 * qi * qo * nb.weight * grad_h(nb.disp, d)
            grad[o] += term
            grad[i] -= term


def _dmadkonst_3d(
    vij, kvecs, kb, dvecs, alpha, volume, direct_cutoff
):
    """3-D Ewald Madelung matrix derivative dM/dR_I (dmadkonst.f CCM3D branch).

    Parameters
    ----------
    vij : (3,) array
        Vector C[I] - CEWALD[J] = C[I] - image_position.
    kvecs : (nK, 3) array
        Reciprocal lattice vectors.
    kb : (nK,) array
        Squared magnitudes |K|^2.
    dvecs : (nT, 3) array
        Direct lattice translation vectors.
    alpha : float
        Ewald convergence parameter.
    volume : float
        Cell volume.
    direct_cutoff : float
        Physical real-space cutoff shared with the Ewald energy.

    Returns
    -------
    dM : (3,) array
        dM/dR_I (NOT yet multiplied by weight or charge).
    """
    dM = np.zeros(3)

    # Reciprocal space sum
    k_dot_v = kvecs @ vij  # (nK,)
    faktor = kb / (4.0 * alpha**2)  # (nK,)
    sin_kv = np.sin(k_dot_v)
    exp_faktor = np.exp(-faktor)
    # d/dR_i [cos(K.vij)] = -K * sin(K.vij)
    # The full reciprocal term: +4pi/V * exp(-K^2/4a^2)/K^2 * cos(K.vij)
    # Its derivative: -4pi/V * K * sin(K.vij) * exp(-K^2/4a^2) / K^2
    # Fortran: DMADKDX += K(1)*SIN(KVIJ)*EXP(-FAKTOR)/FAKTOR
    #          then scale: DMADKDX *= -PI/CONFAC^2/VOL
    # FAKTOR = K^2/4a^2, so EXP(-FAKTOR)/FAKTOR = EXP(-K^2/4a^2) / (K^2/4a^2)
    pre = sin_kv * exp_faktor / faktor  # (nK,)
    dM[0] = np.sum(kvecs[:, 0] * pre)
    dM[1] = np.sum(kvecs[:, 1] * pre)
    dM[2] = np.sum(kvecs[:, 2] * pre)
    # Scale: -pi/(a^2.V)
    dM *= -np.pi / (alpha**2 * volume)

    # Direct space sum
    # d/dR_i [erfc(ar)/r] = -(VTIJ)/r^2 . (erfc(ar)/r + 2a/√pi.exp(-a^2r^2))
    # where VTIJ = vij - T (T = lattice translation vectors)
    vt_all = vij - dvecs  # (nT, 3)
    dist = np.linalg.norm(vt_all, axis=1)
    near = dist < 1e-12
    far = (~near) & (dist <= direct_cutoff)
    if np.any(far):
        r = dist[far]
        erfc_ar = _scipy_erfc(alpha * r)
        gauss = 2.0 * alpha / np.sqrt(np.pi) * np.exp(-(alpha**2) * r**2)
        pref = (erfc_ar / r + gauss) / (r**2)  # (nfar,)
        # -= VTIJ * pref (negative sign from Fortran)
        dM -= np.sum(vt_all[far] * pref[:, np.newaxis], axis=0)

    return dM


def _dmadkonst_2d(vij, kvecs, kmag, dvecs, alpha, area, nhat, direct_cutoff):
    """2-D (slab) Ewald Madelung matrix-element derivative dM/d(vij).

    Analytic derivative of the ``msindo_ccm._madkonst_2d`` Parry/Heyes element
    (ported from ``dmadkonst.f`` CCM2D branch, lines 86-179):

      M(v) = (pi/A) S_K cos(K.v)/|K| . [e^{|K|z} erfc(az + |K|/2a)
                                       + e^{-|K|z} erfc(-az + |K|/2a)]
             - (2pi/A) [z.erf(az) + e^{-a^2z^2}/(a√pi)]          (K=0 term)
             + S_T erfc(a r)/r,   r = |v + T|,

    with z = v.n̂ the out-of-plane separation, K the in-plane reciprocal
    vectors (K.n̂ = 0) and a = 0.85.√pi/√A.  The Gaussian cross terms in the
    z-derivative of the reciprocal sum cancel exactly because
    ``e^{|K|z}.e^{-(az+|K|/2a)^2} = e^{-a^2z^2 - |K|^2/4a^2} = e^{-|K|z}.e^{-(-az+|K|/2a)^2}``,
    leaving (``dmadkonst.f:116-143``):

      d(recip)/dv = (pi/A) S_K [ -sin(K.v).F_K/|K| . K  +  cos(K.v).G_K . n̂ ]
                    F_K = e^{|K|z}erfc(az+|K|/2a) + e^{-|K|z}erfc(-az+|K|/2a)
                    G_K = e^{|K|z}erfc(az+|K|/2a) - e^{-|K|z}erfc(-az+|K|/2a)
      d(K0)/dv    = -(2pi/A).erf(az) . n̂
      d(direct)/dv= -S_T (v+T).[erfc(a r)/r + (2a/√pi)e^{-a^2r^2}]/r^2
                    (``dmadkonst.f:159-175``).

    This is the frame-free form of the Fortran's rotated ``DMADKD{X,Y,Z}``
    (where ``RIJ(3) = z``): no rotate-back (``dedmadelsum.f:106-114``) is needed
    because K and n̂ are kept in Cartesian coordinates.  Returns the bare
    derivative (no weight/charge; those are applied in
    :func:`_add_madelung_gradient`).

    ``kvecs`` / ``dvecs`` / ``direct_cutoff`` come from
    :func:`~vibeqc.semiempirical.methods.msindo_ccm._ewald_lattice_2d`, so the
    derivative is summed over exactly the physical-vector sets the energy used
    and inherits its invariance under a unimodular change of surface basis
    (issue #187).
    """
    dM = np.zeros(3)
    sqrtpi = np.sqrt(np.pi)
    z = float(vij @ nhat)                       # out-of-plane separation
    kdotv = kvecs @ vij                          # K.v  (in-plane phase)

    # Reciprocal-space sum (overflow-free erfcx form; see _parry_recip_pair).
    ep, em = _parry_recip_pair(kmag, alpha, z)
    F = ep + em
    G = ep - em
    # in-plane: -(pi/A) S_K sin(K.v).F_K/|K| . K
    coeff = -(np.pi / area) * (np.sin(kdotv) * F / kmag)
    dM += coeff @ kvecs
    # out-of-plane (recip G term) + K=0 term, both along n̂
    out = ((np.pi / area) * float(np.sum(np.cos(kdotv) * G))
           - (2.0 * np.pi / area) * float(erf(alpha * z)))
    dM += out * nhat

    # Direct-space sum, over the same |v + L| <= r_cut set as the energy.
    pos = vij + dvecs
    dist = np.linalg.norm(pos, axis=1)
    far = (dist > 1e-12) & (dist <= direct_cutoff)
    if np.any(far):
        r = dist[far]
        pref = (erfc(alpha * r) / r
                + 2.0 * alpha / sqrtpi * np.exp(-(alpha**2) * r**2)) / r**2
        dM -= np.sum(pos[far] * pref[:, np.newaxis], axis=0)

    return dM
