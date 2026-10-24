"""Variational continuation of an integer-occupied CCM frontier seed.

Use the existing Neese (2000) SOSCF L-BFGS direction with an Armijo line
search on the same quadratic INDO functional. Principal occupied/virtual
rotations give a stable energy difference even for very small steps. The
initial Fock and energy have already been counted by the seed construction;
every trial below consumes the remaining budget. Exhaustion returns the
last accepted physical state with ``converged=False``.

Twin of ccm_orbitals.hpp. See Neese, Chem. Phys. Lett. 325, 93-98 (2000), and
Lehtola et al., Molecules 25, 1218 (2020), doi:10.3390/molecules25051218.
"""

import numpy as np

from .msindo_soscf import OrbitalOptimizer


def minimize_orbitals(P, F, energy, nocc, budget, tolerance, evaluate):
    """Return density, Fock, energy, convergence, evaluations, and usage flag."""
    nvir = len(P) - nocc
    # Recover the actual seed projector, then diagonalize Fock only within
    # its occupied and virtual spaces. Hcore orbitals do not represent it.
    _, vectors = np.linalg.eigh(P)
    occupied, virtuals = vectors[:, -nocc:], vectors[:, :nvir]
    eo, vo = np.linalg.eigh(occupied.T @ F @ occupied)
    ev, vv = np.linalg.eigh(virtuals.T @ F @ virtuals)
    C = np.column_stack((occupied @ vo, virtuals @ vv))
    eps = np.r_[eo, ev]
    optimizer = OrbitalOptimizer()
    used, previous_energy = 0, np.inf
    converged = optimization_used = False
    while used < budget:
        residual = np.max(np.abs(F @ P - P @ F))
        if residual < tolerance and abs(energy - previous_energy) < tolerance:
            converged = True
            break
        previous_energy = energy
        proposed, _ = optimizer.step(F, C, eps, nocc)
        optimization_used = True
        occupied, virtuals = C[:, :nocc], C[:, nocc:]
        u, singular, vt = np.linalg.svd(
            virtuals.T @ (proposed[:, :nocc] - occupied), full_matrices=False,
        )
        angles = np.arcsin(np.clip(singular, 0., 1.))
        A, B = occupied @ vt.T, virtuals @ u
        gaps = np.sum(B * (F @ B), axis=0) - np.sum(A * (F @ A), axis=0)
        cross = np.sum(A * (F @ B), axis=0)
        slope = 4 * np.dot(angles, cross)
        accepted, alpha = False, 1.
        for _ in range(24):
            if used >= budget:
                break
            theta = alpha * angles
            sine, cosine = np.sin(theta), np.cos(theta)
            cosine_minus_one = -2 * np.sin(.5 * theta)**2
            co = occupied + (A * cosine_minus_one + B * sine) @ vt
            cv = virtuals + (B * cosine_minus_one - A * sine) @ u.T
            trial_density = 2 * co @ co.T
            trial_fock, trial_energy = evaluate(trial_density)
            used += 1
            # Exact quadratic energy difference along principal rotations.
            # Both first-order and quadratic terms scale with the step;
            # subtraction of large total energies would lose this decrease.
            a_sine, b_sine, a_cosine = A * sine, B * sine, A * cosine
            delta = 2 * (
                b_sine @ b_sine.T - a_sine @ a_sine.T
                + a_cosine @ b_sine.T + b_sine @ a_cosine.T
            )
            change = (2 * np.sum(sine**2 * gaps + 2 * sine * cosine * cross)
                      + .5 * np.sum(delta * (trial_fock - F)))
            if change <= 1e-4 * alpha * slope:
                accepted = True
                break
            alpha *= .5
        if not accepted:
            break
        if alpha != 1.:
            # The next curvature pair must use the rotation actually taken,
            # preserving useful history even when orbital gaps vanish.
            optimizer.previous_step *= alpha
        C = np.column_stack((co, cv))
        P, F, energy = trial_density, trial_fock, trial_energy
        eps = np.diag(C.T @ F @ C).copy()
    return P, F, energy, converged, used, optimization_used
