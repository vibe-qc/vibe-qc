"""Restricted orbital-stability repair for ambiguous MSINDO CCM guesses.

Python reference twin of ``ccm_stability.hpp``. The electronic Hamiltonian,
occupations and embedding are unchanged. See Lehtola, Blockhuys and Van
Alsenoy, Molecules 25, 1218 (2020), Sec. 10, doi:10.3390/molecules25051218.
"""

import numpy as np

from . import msindo


def _orbital_stability(H, P, nocc, fock):
    """Return the lowest quarter-Hessian eigenpair and canonical orbitals."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    eps, C = np.linalg.eigh(fock(P)[0])
    co, cv = C[:, :nocc], C[:, nocc:]
    nv = cv.shape[1]
    gaps = eps[nocc:, None] - eps[None, :nocc]
    zero_fock = fock(np.zeros_like(P))[0]

    def action(x):
        k = x.reshape((nv, nocc), order="F")
        dP = 2.0 * cv @ k @ co.T
        dP += dP.T.copy()
        dF = fock(dP)[0] - zero_fock
        return (cv.T @ dF @ co + gaps * k).ravel(order="F")

    dim = nv * nocc
    if dim == 1:
        eig, direction = action(np.ones(1))[0], np.ones((nv, nocc))
    else:
        operator = LinearOperator((dim, dim), matvec=action, dtype=float)
        try:
            values, vectors = eigsh(
                operator, k=1, which="SA", tol=1e-8, maxiter=150,
                ncv=min(32, dim), v0=np.sin(np.arange(1, dim + 1)),
            )
        except Exception as exc:
            raise RuntimeError(
                "MSINDO CCM orbital stability analysis did not converge"
            ) from exc
        eig = float(values[0])
        vector = vectors[:, 0]
        if np.linalg.norm(action(vector) - eig * vector) > 1e-8:
            raise RuntimeError("MSINDO CCM orbital stability analysis did not converge")
        direction = vector.reshape((nv, nocc), order="F")
    return eig, co, cv, direction


def _rotated_density(co, cv, direction, theta):
    u, s, vt = np.linalg.svd(direction, full_matrices=False)
    angles = theta * s
    occ = (co + (co @ vt.T * (np.cos(angles) - 1.0)) @ vt
           + (cv @ u * np.sin(angles)) @ vt)
    return 2.0 * occ @ occ.T


def scf_rhf_ccm(H, G, blocks, Z, nocc, *, max_iter=200, conv_tol=1e-9,
                fock_extra=None, diagnostics=None):
    """SCF plus restricted stability for an unresolved Hcore frontier.

    ``max_iter`` bounds the cumulative SCF iterations, including both signs
    of each stability restart. Stability is a local minimum test within RHF,
    not a global-minimum or unrestricted-spin claim.
    """
    result = msindo._scf_rhf(
        H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol,
        fock_extra=fock_extra,
    )
    if not result[4] or nocc in (0, len(H)):
        return result
    guess_eps = np.linalg.eigvalsh(H)
    if guess_eps[nocc] - guess_eps[nocc - 1] > 1e-6:
        return result

    def fock(P):
        F = msindo._build_fock(H, G, P, blocks, Z)
        e_add = 0.0
        if fock_extra is not None:
            addition, e_add = fock_extra(P)
            F = F + addition
        return F, e_add

    def energy(P):
        F, e_add = fock(P)
        return 0.5 * np.sum(P * (H + F)) + e_add

    used = result[5]
    for restart in range(6):
        curvature, co, cv, direction = _orbital_stability(H, result[0], nocc, fock)
        if diagnostics is not None:
            diagnostics.update(
                stability_checked=True, stability_analysis_converged=True,
                stability_eigenvalue=4.0 * curvature, n_stability_restarts=restart,
            )
        if curvature >= -1e-6:
            return (*result[:5], used)
        if restart == 5 or used >= max_iter:
            raise RuntimeError(
                "MSINDO CCM found an unstable SCF saddle; stability restart budget exhausted"
            )
        best = result
        descended = False
        for sign_index, sign in enumerate((-1.0, 1.0)):
            seed, seed_energy = None, result[2]
            for angle in (0.1, 0.25, 0.5, 1.0, 2.0):
                P = _rotated_density(co, cv, direction, sign * angle)
                e = energy(P)
                if e < seed_energy:
                    seed, seed_energy = P, e
            budget = (max_iter - used) // (2 - sign_index)
            if seed is None or budget <= 0:
                continue
            candidate = msindo._scf_rhf(
                H, G, blocks, Z, nocc, max_iter=budget, conv_tol=conv_tol,
                fock_extra=fock_extra, initial_density=seed,
            )
            used += candidate[5]
            if candidate[4] and candidate[2] < best[2] - max(1e-10, conv_tol):
                best, descended = candidate, True
        if not descended:
            raise RuntimeError(
                "MSINDO CCM found an unstable SCF saddle; no lower stationary "
                "solution converged within max_iter"
            )
        result = best
