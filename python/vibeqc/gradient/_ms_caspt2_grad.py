"""MS/XMS-CASPT2 analytic nuclear gradient (SA-CASSCF Lagrangian).

Implements the relaxed multi-state CASPT2 gradient via the constrained
Lagrangian

    L = E_MS(kappa, t, theta; R)
      + lambda_p . grad_{kappa,t} E_SA(kappa, t; R)      (SA stationarity)
      + lambda_theta . <c_K| H_CAS(kappa; R) |c_J>       (eigenvector cond.)

with three parameter classes at the SA-CASSCF solution:

* ``kappa`` -- non-redundant orbital rotations.  The SA-CASSCF energy
  E_SA = Sum_J w_J E_J is stationary (Stalring, Bernhardsson & Lindh,
  Mol. Phys. 99, 103 (2001): the individual-state gradient needs the
  SA response, not state-specific stationarity).
* ``t``     -- external CI mixing of each averaged state (orthogonal
  complement of the model span).  Stationarity of E_SA in ``t`` is
  equivalent to the CASCI eigenvector condition in the external space.
* ``theta`` -- rotations *among* the averaged states.  These are E_SA-null
  for equal weights and are fixed by the eigenvector condition
  <c_K|H|c_J> = 0 instead; their multipliers are the state Lagrangian
  ("SLag" in OpenMolcas mclr/rhs_sa.F90 + caspt2/clagfinal.F90):
  lambda_theta = (dE_MS/dtheta_JK) / (E_J - E_K).

Stationarity of L in all parameters yields the multiplier system

    [ A     G_ptheta ] [lambda_p    ]   [ dE_MS/dp     ]
    [ B     D        ] [lambda_theta] = [ dE_MS/dtheta ]  (times -1)

with A the SA-CASSCF electronic Hessian over (kappa, t), G_ptheta[p, th] =
d<c_K|H|c_J>/dp, B[th, p] = d(grad_p E_SA)/dtheta (zero for equal
weights), and D = diag(E_K - E_J) at the reference.  The gradient is then
the explicit R-derivative of L at fixed parameters:

    dE_MS/dR = dE_MS/dR|frozen  +  lambda_p . g^R  +  lambda_theta . s^R

where g^R = d(grad_p E_SA)/dR and s^R = d<c_K|H|c_J>/dR.  All explicit
R-derivatives use MO integrals re-orthonormalized with the symmetric
(Loewdin) connection C(R) = C (C^T S(R) C)^{-1/2}; the connection must
match between the frozen term and g^R or the response picks up a spurious
antisymmetric-between-states error (the pre-2026-07 wrong-sign bug).

E_MS itself is evaluated through :func:`vibeqc.solvers._ms_caspt2.
_heff_pt2_blocks` on explicitly parametrized model states -- the same
pipeline :func:`ms_caspt2` uses, so energy and gradient cannot drift.
For ``mode="xms"`` the model-space Fock rotation is applied inside the
pipeline, making E_XMS invariant under ``theta`` (Granovsky, J. Chem.
Phys. 134, 214113 (2011); Shiozaki, Gyorffy, Celani & Werner, J. Chem.
Phys. 135, 081106 (2011)); the transfer terms then vanish identically
and are skipped.

Validated against central-difference FD of the full MS/XMS-CASPT2 energy
(2026-07-02): H2/6-31G SA2-CAS(2,2) ms 9e-8, xms 7e-8; LiH/6-31G
SA2-CAS(2,2) (mixed states, n_core=1, nonzero SLag) ms 4e-9, xms 7e-9
Ha/bohr.  Cross-checked against OpenMolcas MS-CASPT2 (exact-ERI FD and
RICD analytic MCLR+ALASKA) on the H2 case.

Cost note: the semi-numerical RHS differentiates the full MS pipeline by
FD over (kappa, t, theta) -- O(npar) MS solves with npar = n_pairs +
n_sa * n_external.  n_external grows with the determinant count, so the
CI-response part is gated by ``max_det_response``; the analytic CI
Lagrangian (transition-RDM based, as in the single-state Z-vector path)
is the planned lift for large active spaces.

References
----------
Finley, Malmqvist, Roos & Serrano-Andres, Chem. Phys. Lett. 288, 299 (1998).
Celani & Werner, J. Chem. Phys. 119, 5044 (2003).
Stalring, Bernhardsson & Lindh, Mol. Phys. 99, 103 (2001).
Granovsky, J. Chem. Phys. 134, 214113 (2011).
Shiozaki, Gyorffy, Celani & Werner, J. Chem. Phys. 135, 081106 (2011).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = ["compute_ms_caspt2_gradient"]


def compute_ms_caspt2_gradient(
    molecule,
    basis,
    C_mo: np.ndarray,
    h1e_cas: np.ndarray,
    h2e_cas: np.ndarray,
    n_core: int,
    n_active_orb: int,
    *,
    n_active_elec: int,
    sa_weights: list[float],
    nroots: int,
    mode: str = "ms",
    target_root: int = 0,
    ms2: int = 0,
    fd_geom: float = 1e-3,
    fd_param: float = 1e-3,
    hess_eps: float = 1e-3,
    null_thresh: float = 1e-6,
    max_det_response: int = 64,
) -> np.ndarray:
    """Relaxed MS/XMS-CASPT2 nuclear gradient for one multi-state root.

    Parameters
    ----------
    C_mo : (n_ao, n_mo) ndarray
        Converged SA-CASSCF MO coefficients.
    h1e_cas, h2e_cas : ndarray
        MO integrals in the converged basis (physicist's ``h2e_cas``).
    n_active_elec : int
        Active electron count.
    sa_weights : list[float]
        SA-CASSCF averaging weights.  The model states must all be
        averaged states (``len(sa_weights) >= nroots``); this is the same
        requirement OpenMolcas MCLR places on MS-CASPT2 gradients.
    nroots : int
        Number of model states (the MS model space size).
    mode : str
        ``"ms"`` or ``"xms"``.
    target_root : int
        Which multi-state root's gradient to compute (0 = ground).
    ms2 : int
        2 M_s of the active electrons (spin-pure model-space filter).
    fd_geom, fd_param, hess_eps : float
        Central-difference steps for geometry, parameter-gradient and
        Hessian displacements.
    null_thresh : float
        SA-Hessian eigenvalue cutoff for the pseudo-inverse (spin- or
        weight-degenerate null directions carry no response).
    max_det_response : int
        Guard on the determinant count for the FD CI-response (see the
        module cost note).

    Notes
    -----
    Manually supplied ECP-derived orbitals, integrals, or reference data
    paired with an all-electron-named basis are unsupported.  This API can
    reject an ECP attached to ``basis``, but cannot infer the Hamiltonian
    provenance of the supplied arrays.

    Returns
    -------
    grad : (n_atoms, 3) ndarray in Hartree/bohr.
    """
    from vibeqc.ecp_metadata import refuse_molecular_ecp_derivative_route

    refuse_molecular_ecp_derivative_route(
        molecule,
        basis,
        route="compute_ms_caspt2_gradient",
    )
    from vibeqc._vibeqc_core import (
        compute_eri as _compute_eri,
    )
    from vibeqc._vibeqc_core import (
        compute_kinetic as _compute_kinetic,
    )
    from vibeqc._vibeqc_core import (
        compute_nuclear as _compute_nuclear,
    )
    from vibeqc._vibeqc_core import (
        compute_overlap as _compute_overlap,
    )
    from vibeqc._vibeqc_core import Atom, Molecule
    from vibeqc.gradient._casscf import (
        _build_ci_hamiltonian,
        _build_kappa_from_z,
        _nonredundant_pairs_local,
    )
    from vibeqc.solvers._casci import _frozen_core_dressing
    from vibeqc.solvers._ms_caspt2 import _heff_pt2_blocks, _spin_pure_roots

    if mode not in ("ms", "xms"):
        raise ValueError(f"mode must be 'ms' or 'xms', got {mode!r}")
    if sa_weights is None:
        raise ValueError(
            "MS-CASPT2 gradient requires a state-averaged CASSCF "
            "reference: pass casscf_options.weights (the model states "
            "must be SA-averaged states)"
        )
    n_sa = len(sa_weights)
    if n_sa < nroots:
        raise ValueError(
            f"all {nroots} model states must be SA-averaged: got "
            f"{n_sa} averaging weights.  Increase casscf_options.nroots/"
            f"weights to cover the MS model space."
        )
    w = [float(x) for x in sa_weights]

    nmo = C_mo.shape[1]
    n_act = n_active_orb
    act = slice(n_core, n_core + n_act)
    pairs = _nonredundant_pairs_local(n_core, n_act, nmo)
    npr = len(pairs)
    n_atoms = len(molecule.atoms)
    atoms_ref = [(int(at.Z), [float(x) for x in at.xyz]) for at in molecule.atoms]

    # ── model + averaged states: spin-pure CASCI roots, matching ms_caspt2 ──
    ci_cols, _e_ref, _s2, _res = _spin_pure_roots(
        h1e_cas, h2e_cas, n_core, n_act, n_active_elec, ms2, n_sa
    )
    dets = _res.determinants
    ndet = ci_cols.shape[0]
    if ndet > max_det_response:
        raise NotImplementedError(
            f"MS-CASPT2 gradient: determinant count {ndet} exceeds "
            f"max_det_response={max_det_response}.  The FD CI-response "
            f"scales with the determinant count; the analytic CI "
            f"Lagrangian for large active spaces is on the roadmap."
        )
    avg = [ci_cols[:, J] for J in range(n_sa)]

    # Reference active CI Hamiltonian + external (non-model) directions.
    Hact_ref = _build_ci_hamiltonian(h1e_cas, h2e_cas, dets, n_core, n_act)
    E_avg = [avg[J] @ Hact_ref @ avg[J] for J in range(n_sa)]
    _evH, VH = np.linalg.eigh(Hact_ref)
    P_span = sum(np.outer(c, c) for c in avg)
    ext: list[np.ndarray] = []
    for k in range(ndet):
        v = VH[:, k] - P_span @ VH[:, k]
        nv = np.linalg.norm(v)
        if nv > 1e-8:
            v = v / nv
            for u in ext:
                v = v - (u @ v) * u
            nv = np.linalg.norm(v)
            if nv > 1e-8:
                ext.append(v / nv)
    next_ = len(ext)
    npar = npr + n_sa * next_

    # theta pairs: rotations among averaged states.  E_XMS is invariant
    # under model-span rotations (the internal Fock rotation undoes them),
    # so xms skips the transfer terms -- valid only when every averaged
    # state is a model state.
    if mode == "xms" and n_sa != nroots:
        raise NotImplementedError(
            "XMS-CASPT2 gradient with more averaged states than model "
            "states: transfer response between model and non-model "
            "averaged states is not theta-invariant and is not yet "
            "implemented."
        )
    pairs_th = (
        [(J, K) for J in range(n_sa) for K in range(J + 1, n_sa)]
        if mode == "ms"
        else []
    )
    ntheta = len(pairs_th)

    # ── parametrized states / integrals / energies ──────────────────────
    def states_at(theta: np.ndarray, t: np.ndarray) -> np.ndarray:
        n = n_sa
        Kth = np.zeros((n, n))
        for i, (J, K) in enumerate(
            [(J, K) for J in range(n) for K in range(J + 1, n)]
        ):
            Kth[J, K] = -theta[i] if i < len(theta) else 0.0
            Kth[K, J] = theta[i] if i < len(theta) else 0.0
        base = np.column_stack(avg) @ expm(Kth)
        cis = np.zeros((ndet, n))
        for J in range(n):
            c = base[:, J].copy()
            for kk in range(next_):
                c = c + t[J, kk] * ext[kk]
            cis[:, J] = c / np.linalg.norm(c)
        return cis

    def integrals_at(atom: int | None = None, comp: int = 0, delta: float = 0.0):
        """MO integrals at a displaced geometry, Loewdin-reconnected.

        C(R) = C (C^T S(R) C)^{-1/2} -- the symmetric connection, matching
        the frozen term (both are FD of the same connected energy).
        """
        a = [(z, list(x)) for z, x in atoms_ref]
        if atom is not None:
            a[atom] = (a[atom][0], list(a[atom][1]))
            a[atom][1][comp] += delta
        m = Molecule([Atom(z, list(x)) for z, x in a])
        b = type(basis)(m, basis.name)
        T = np.asarray(_compute_kinetic(b))
        Vn = np.asarray(_compute_nuclear(b, m))
        g = np.asarray(_compute_eri(b))
        S = np.asarray(_compute_overlap(b))
        O = C_mo.T @ S @ C_mo
        ev, evec = np.linalg.eigh(O)
        Cd = C_mo @ (evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T)
        h1r = Cd.T @ (T + Vn) @ Cd
        h2r = np.einsum(
            "ap,bq,cr,ds,abcd->pqrs",
            Cd,
            Cd,
            Cd,
            Cd,
            g.transpose(0, 2, 1, 3),
            optimize=True,
        )
        return h1r, h2r, m.nuclear_repulsion()

    def rotated(kv: np.ndarray, h1b: np.ndarray, h2b: np.ndarray):
        if not np.any(kv):
            return h1b, h2b
        K = _build_kappa_from_z(kv, pairs, nmo)
        U = expm(K)
        h1r = U.T @ h1b @ U
        h2r = np.einsum("ap,bq,cr,ds,abcd->pqrs", U, U, U, U, h2b, optimize=True)
        return h1r, h2r

    def E_SA(kv: np.ndarray, t: np.ndarray, h1b: np.ndarray, h2b: np.ndarray):
        """SA-CASSCF energy functional (electronic incl. core energy)."""
        h1r, h2r = rotated(kv, h1b, h2b)
        H = _build_ci_hamiltonian(h1r, h2r, dets, n_core, n_act)
        e_core, _ = _frozen_core_dressing(h1r, h2r, n_core, act)
        e = 0.0
        for J in range(n_sa):
            c = avg[J].copy()
            for kk in range(next_):
                c = c + t[J, kk] * ext[kk]
            c = c / np.linalg.norm(c)
            e += w[J] * (c @ H @ c + e_core)
        return e

    def E_MS(
        kv: np.ndarray,
        theta: np.ndarray,
        t: np.ndarray,
        h1b: np.ndarray,
        h2b: np.ndarray,
        nuc: float,
    ) -> float:
        """Target multi-state root through the shared Heff pipeline."""
        h1r, h2r = rotated(kv, h1b, h2b)
        cis = states_at(theta, t)[:, :nroots]
        blocks = _heff_pt2_blocks(
            cis,
            dets,
            h1r,
            h2r.transpose(0, 2, 1, 3).copy(),
            n_core,
            n_act,
            nmo,
            mode=mode,
        )
        heff = blocks["ref_expl"].copy()
        heff[np.diag_indices(nroots)] += np.asarray(blocks["e2"])
        heff += blocks["coup"]
        heff = 0.5 * (heff + heff.T)
        wv = np.linalg.eigvalsh(heff)
        return float(wv[target_root]) + nuc

    k0 = np.zeros(npr)
    t0 = np.zeros((n_sa, next_))
    th0 = np.zeros(max(ntheta, 1))

    def unpack(p: np.ndarray):
        return p[:npr], p[npr:].reshape(n_sa, next_)

    def grad_par(f, eps: float) -> np.ndarray:
        g = np.zeros(npar)
        for i in range(npar):
            pp = np.zeros(npar)
            pp[i] = eps
            kp, tp = unpack(pp)
            km, tm = unpack(-pp)
            g[i] = (f(kp, tp) - f(km, tm)) / (2.0 * eps)
        return g

    # ── SA-CASSCF electronic Hessian A over (kappa, t) ──────────────────
    def val_SA(p: np.ndarray) -> float:
        kv, t = unpack(p)
        return E_SA(kv, t, h1e_cas, h2e_cas)

    A = np.zeros((npar, npar))
    for i in range(npar):
        for j in range(i, npar):
            pi = np.zeros(npar)
            pi[i] = hess_eps
            pj = np.zeros(npar)
            pj[j] = hess_eps
            v = (
                val_SA(pi + pj) - val_SA(pi - pj) - val_SA(-pi + pj) + val_SA(-pi - pj)
            ) / (4.0 * hess_eps * hess_eps)
            A[i, j] = v
            A[j, i] = v

    def psolve(M: np.ndarray, b: np.ndarray) -> np.ndarray:
        ev, evec = np.linalg.eigh(M)
        keep = np.abs(ev) > null_thresh
        return evec[:, keep] @ ((evec[:, keep].T @ b) / ev[keep])

    # ── multiplier system ────────────────────────────────────────────────
    # dE_MS/d(kappa, t): FD of the full MS pipeline (R-independent).
    gMS_p = grad_par(
        lambda kv, t: E_MS(kv, th0[:ntheta], t, h1e_cas, h2e_cas, 0.0), fd_param
    )

    lam_th = np.zeros(ntheta)
    Gp = np.zeros((npar, ntheta))
    if ntheta:
        # dE_MS/dtheta and the transfer multipliers
        # lambda_theta = (dE_MS/dtheta_JK)/(E_J - E_K)  (state Lagrangian;
        # OpenMolcas clagfinal.F90 S_IJ convention).
        dE_dth = np.zeros(ntheta)
        for i in range(ntheta):
            thv = np.zeros(ntheta)
            thv[i] = fd_param
            dE_dth[i] = (
                E_MS(k0, thv, t0, h1e_cas, h2e_cas, 0.0)
                - E_MS(k0, -thv, t0, h1e_cas, h2e_cas, 0.0)
            ) / (2.0 * fd_param)
        for i, (J, K) in enumerate(pairs_th):
            dE = E_avg[J] - E_avg[K]
            if abs(dE) < 1e-8:
                if abs(dE_dth[i]) > 1e-10:
                    raise ValueError(
                        f"MS-CASPT2 gradient: degenerate model states "
                        f"{J},{K} (dE={dE:.2e}) with nonzero coupling "
                        f"derivative -- the state Lagrangian is singular."
                    )
                lam_th[i] = 0.0
            else:
                lam_th[i] = dE_dth[i] / dE

        # G_ptheta[p, th] = d<c_K|H_CAS|c_J>/dp: feedback of the transfer
        # multipliers into the (kappa, t) stationarity (kappa block only;
        # the t block vanishes at the reference where <v|H|c_K> = 0).
        for i in range(npr):
            kv = np.zeros(npr)
            kv[i] = fd_param
            h1a, h2a = rotated(kv, h1e_cas, h2e_cas)
            h1b_, h2b_ = rotated(-kv, h1e_cas, h2e_cas)
            Ha = _build_ci_hamiltonian(h1a, h2a, dets, n_core, n_act)
            Hb = _build_ci_hamiltonian(h1b_, h2b_, dets, n_core, n_act)
            dHk = (Ha - Hb) / (2.0 * fd_param)
            for ith, (J, K) in enumerate(pairs_th):
                Gp[i, ith] = avg[K] @ dHk @ avg[J]

        # B[th, p] = d(grad_p E_SA)/dtheta: zero for equal weights (the SA
        # density is invariant under model-span rotations); nonzero for
        # unequal weights, where the full block system must be solved.
        if max(w[:n_sa]) - min(w[:n_sa]) > 1e-12:
            B = np.zeros((ntheta, npar))
            for i in range(ntheta):
                thv = np.zeros(ntheta)
                thv[i] = fd_param

                def _esa_th(kv, t, s):
                    cis_th = states_at(s, t)
                    h1r, h2r = rotated(kv, h1e_cas, h2e_cas)
                    H = _build_ci_hamiltonian(h1r, h2r, dets, n_core, n_act)
                    e_core, _ = _frozen_core_dressing(h1r, h2r, n_core, act)
                    return sum(
                        w[J] * (cis_th[:, J] @ H @ cis_th[:, J] + e_core)
                        for J in range(n_sa)
                    )

                gp_ = grad_par(lambda kv, t: _esa_th(kv, t, thv), fd_param)
                gm_ = grad_par(lambda kv, t: _esa_th(kv, t, -thv), fd_param)
                B[i, :] = (gp_ - gm_) / (2.0 * fd_param)
            D = np.diag([E_avg[K] - E_avg[J] for (J, K) in pairs_th])
            # coupled asymmetric multiplier system over (p, theta)
            M = np.block([[A, Gp], [B, D]])
            rhs = -np.concatenate([gMS_p, dE_dth])
            sol = np.linalg.lstsq(M, rhs, rcond=null_thresh)[0]
            lam_p = sol[:npar]
            lam_th = sol[npar:]
        else:
            lam_p = -psolve(A, gMS_p + Gp @ lam_th)
    else:
        lam_p = -psolve(A, gMS_p)

    # ── assemble dE/dR per component ─────────────────────────────────────
    grad = np.zeros((n_atoms, 3))
    for a in range(n_atoms):
        for c in range(3):
            h1p, h2p, nucp = integrals_at(a, c, fd_geom)
            h1m, h2m, nucm = integrals_at(a, c, -fd_geom)
            # frozen: explicit R-derivative of E_MS at fixed parameters
            g_frozen = (
                E_MS(k0, th0[:ntheta], t0, h1p, h2p, nucp)
                - E_MS(k0, th0[:ntheta], t0, h1m, h2m, nucm)
            ) / (2.0 * fd_geom)
            # response: lambda_p . g^R
            gp_ = grad_par(lambda kv, t: E_SA(kv, t, h1p, h2p), fd_param)
            gm_ = grad_par(lambda kv, t: E_SA(kv, t, h1m, h2m), fd_param)
            g_resp = float(lam_p @ ((gp_ - gm_) / (2.0 * fd_geom)))
            # transfer: lambda_theta . d<c_K|H|c_J>/dR
            g_slag = 0.0
            if ntheta:
                Hp = _build_ci_hamiltonian(h1p, h2p, dets, n_core, n_act)
                Hm = _build_ci_hamiltonian(h1m, h2m, dets, n_core, n_act)
                dH = (Hp - Hm) / (2.0 * fd_geom)
                for ith, (J, K) in enumerate(pairs_th):
                    g_slag += lam_th[ith] * (avg[K] @ dH @ avg[J])
            grad[a, c] = g_frozen + g_resp + g_slag

    return grad
