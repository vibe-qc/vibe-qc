"""Restricted orbital-stability audit of every converged MSINDO CCM state.

Python reference twin of ``ccm_stability.hpp``. The electronic Hamiltonian,
occupations and embedding are unchanged. See Lehtola, Blockhuys and Van
Alsenoy, Molecules 25, 1218 (2020), Sec. 10, doi:10.3390/molecules25051218.

The audit runs after every converged closed-shell SCF (#249). The default
reference policy repairs a saddle only where the reference program's
Hcore/DIIS trajectory is ill-defined, an unresolved Hcore frontier
(gap <= 1e-6 Ha), and otherwise returns the converged state flagged as
internally unstable; the frozen MSINDO oracle clusters include such saddles.
The lowest-stable policy descends from every saddle: a C3 chain bent by
3e-3 bohr has a resolved gap (1e-5 Ha) yet converges from the Hcore guess
16.6 mHa above the continuous lower branch.
"""

STABILITY_POLICIES = ("reference", "lowest")

import numpy as np

from . import msindo

# Rotation angles (radians along the unstable mode) scanned for a restart seed.
_RESTART_ANGLES = (0.1, 0.25, 0.5, 1.0, 2.0)
# A mode along which no scanned rotation lowers the energy by this much is
# stable within the SCF's resolution: the analytic Hessian resolves its sign
# to about 1e-5 and DIIS cannot hold a stationary point 1e-7 Ha below.
STABILITY_DESCENT_TOL = 1e-6


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
                fock_extra=None, diagnostics=None, initial_density=None,
                lowest_stable=False):
    """SCF plus a restricted stability audit of the converged state.

    ``max_iter`` bounds the cumulative SCF iterations, including every
    stability restart. Stability is a local minimum test within RHF, not a
    global-minimum or unrestricted-spin claim. A state whose lowest Hessian
    eigenvalue is negative but whose unstable mode cannot lower the energy
    by :data:`STABILITY_DESCENT_TOL` is accepted as stable within tolerance;
    the eigenvalue is still reported. The lower-energy rotation sign is
    re-converged first with the whole remaining budget, the other sign only
    when the first does not descend. ``initial_density`` replaces the Hcore
    start of the primary SCF; the multi-start uniqueness checks use it.
    ``lowest_stable`` selects the lowest-stable policy (see the module
    docstring); otherwise a resolved-gap saddle is returned with
    ``internal_instability=True`` in ``diagnostics``.
    """
    result = msindo._scf_rhf(
        H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol,
        fock_extra=fock_extra, initial_density=initial_density,
    )
    if not result[4] or nocc in (0, len(H)):
        return result
    guess_eps = np.linalg.eigvalsh(H)
    descend = lowest_stable or guess_eps[nocc] - guess_eps[nocc - 1] <= 1e-6

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
        try:
            curvature, co, cv, direction = _orbital_stability(H, result[0], nocc, fock)
        except RuntimeError:
            # Reference policy at a resolved gap: record an unverified audit
            # rather than fail a state the reference program returns.
            if descend:
                raise
            if diagnostics is not None:
                diagnostics.update(
                    stability_checked=True, stability_analysis_converged=False,
                    stability_eigenvalue=0.0, n_stability_restarts=restart,
                    internal_instability=False,
                )
            return (*result[:5], used)
        if diagnostics is not None:
            diagnostics.update(
                stability_checked=True, stability_analysis_converged=True,
                stability_eigenvalue=4.0 * curvature, n_stability_restarts=restart,
                internal_instability=False,
            )
        if curvature >= -1e-6:
            return (*result[:5], used)
        # One seed per sign of the mode: the lowest-energy scanned rotation.
        seeds = []
        for sign in (-1.0, 1.0):
            seed, seed_energy = None, result[2]
            for angle in _RESTART_ANGLES:
                P = _rotated_density(co, cv, direction, sign * angle)
                e = energy(P)
                if e < seed_energy:
                    seed, seed_energy = P, e
            if seed is not None:
                seeds.append((seed_energy, sign, seed))
        if not seeds or result[2] - min(s[0] for s in seeds) < STABILITY_DESCENT_TOL:
            return (*result[:5], used)
        if not descend:
            if diagnostics is not None:
                diagnostics["internal_instability"] = True
            return (*result[:5], used)
        if restart == 5 or used >= max_iter:
            raise RuntimeError(
                "MSINDO CCM found an unstable SCF saddle; stability restart budget exhausted"
            )
        # Lower seed first; a tie within 1e-10 keeps the negative sign first.
        if len(seeds) == 2 and seeds[1][0] < seeds[0][0] - 1e-10:
            seeds.reverse()
        best, descended = result, False
        for seed_energy, sign, seed in seeds:
            budget = max_iter - used
            if budget <= 0:
                break
            candidate = msindo._scf_rhf(
                H, G, blocks, Z, nocc, max_iter=budget, conv_tol=conv_tol,
                fock_extra=fock_extra, initial_density=seed,
            )
            used += candidate[5]
            if candidate[4] and candidate[2] < best[2] - max(1e-10, conv_tol):
                best, descended = candidate, True
                break
        if not descended:
            raise RuntimeError(
                "MSINDO CCM found an unstable SCF saddle; no lower stationary "
                "solution converged within max_iter"
            )
        result = best
