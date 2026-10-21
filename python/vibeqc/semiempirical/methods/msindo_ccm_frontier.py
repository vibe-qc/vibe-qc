"""Variational initial projector for a rank-one degenerate CCM frontier.

The INDO energy is quadratic in density and its Fock operator is its first
derivative. Minimize that same energy over one occupied orbital (or one hole)
in the unresolved Hcore frontier, holding all other orbitals fixed. This
resolves an arbitrary eigensolver projector; it is not a global SCF search.
See Lehtola et al., Molecules 25, 1218 (2020), doi:10.3390/molecules25051218,
for initial guesses, orbital optimization, and the distinction from stability.

Twin of ccm_frontier.hpp. Every Fock evaluation, including line searches and
curvature construction, consumes the caller's cumulative iteration budget.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class FrontierSeed:
    density: np.ndarray | None = None
    fock: np.ndarray | None = None
    energy: float = 0.0
    evaluations: int = 0
    applicable: bool = False
    completed: bool = False
    orbital_begin: int = 0
    orbital_end: int = 0
    expansions: int = 0


def _minimize_frontier(H, nocc, budget, evaluate, bounds=None, initial_density=None):
    """Return a bounded initial guess; ``evaluate(P)`` returns Fock and energy.

    The caller must not accept an exhausted seed as an SCF result. General
    partially occupied frontiers and blocks larger than eight keep the
    established Hcore path. An explicitly supplied density bypasses this
    operation in the caller. A modified Newton step uses exact spherical
    curvature, choosing a negative-curvature step when its quadratic model
    predicts more descent. All steps retract onto the unit sphere.
    """
    out = FrontierSeed()
    if nocc in (0, len(H)):
        return out
    eps, C = np.linalg.eigh(H)
    if bounds is None and eps[nocc] - eps[nocc - 1] > 1e-6:
        return out
    center = .5 * (eps[nocc] + eps[nocc - 1])
    lo, hi = nocc - 1, nocc
    while lo > 0 and abs(eps[lo - 1] - center) < 1e-6:
        lo -= 1
    while hi < len(H) and abs(eps[hi] - center) < 1e-6:
        hi += 1
    if bounds is not None:
        lo, hi = bounds
    out.orbital_begin, out.orbital_end = lo, hi
    g, q = hi - lo, nocc - lo
    if g > 8 or q not in (1, g - 1):
        return out
    if budget < 1:
        raise ValueError("max_iter must be positive")
    out.applicable = True
    hole = q != 1
    sign, fixed = (-1., hi) if hole else (1., lo)
    base = 2 * C[:, :fixed] @ C[:, :fixed].T
    B = C[:, lo:hi]
    ref_axis = g - 1 if hole else 0
    ref = np.eye(g)[ref_axis]
    if initial_density is not None:
        D = sign * .5 * B.T @ (initial_density - base) @ B
        _, V = np.linalg.eigh(.5 * (D + D.T))
        ref = V[:, -1]

    def density(x):
        u = B @ (x / np.linalg.norm(x))
        return base + sign * 2 * np.outer(u, u)

    out.density = density(ref)
    out.fock, out.energy = evaluate(out.density)
    out.evaluations = 1
    Fref = B.T @ out.fock @ B
    A = Fref - np.trace(Fref) / g * np.eye(g)

    class Exhausted(Exception):
        pass

    def value(x):
        if out.evaluations >= budget:
            raise Exhausted
        v = x / np.linalg.norm(x)
        P = density(x)
        F, absolute_energy = evaluate(P)
        out.evaluations += 1
        response = B.T @ F @ B - Fref
        delta = sign * 2 * (np.outer(v, v) - np.outer(ref, ref))
        # E(P)-E(Pref) = Tr(dP Fref) + Tr(dP dF)/2. Evaluate
        # in the small frontier to avoid cancellation of large total energies.
        energy = float(np.sum(delta * (A + .5 * response)))
        grad = sign * 4 * (A + response) @ v
        grad = (grad - v * np.dot(v, grad)) / np.linalg.norm(x)
        out.density, out.fock, out.energy = P, F, absolute_energy
        return energy, grad, P, F, absolute_energy

    refgrad = sign * 4 * A @ ref
    refgrad -= ref * np.dot(ref, refgrad)
    initial = (0., refgrad, out.density, out.fock, out.energy)
    best = None
    try:
        for start in range(g + 2 if initial_density is None else 1):
            if start == 0:
                x = ref.copy()
            elif start < g:
                x = np.eye(g)[(start + ref_axis) % g]
            else:
                x = np.ones(g) if start == g else np.where(np.arange(g) % 2, -1., 1.)
            x = x / np.linalg.norm(x)
            current = initial if start == 0 else value(x)
            stationary_minimum = False
            for iteration in range(100):
                tangent = np.eye(g) - np.outer(x, x)
                u = B @ x
                Fsub = B.T @ current[3] @ B
                # Remove the irrelevant scalar before forming the tangent Hessian.
                Fsub -= np.trace(Fsub) / g * np.eye(g)
                multiplier = sign * 4 * np.dot(x, Fsub @ x)
                hessian = np.empty((g, g))
                for j in range(g):
                    if out.evaluations >= budget:
                        raise Exhausted
                    t = tangent[:, j]
                    du = B @ t
                    dP = sign * 2 * (np.outer(du, u) + np.outer(u, du))
                    response, _ = evaluate(current[2] + dP)
                    out.evaluations += 1
                    dF = response - current[3]
                    hessian[:, j] = tangent @ (
                        sign * 4 * (Fsub @ t + B.T @ dF @ u) - multiplier * t
                    )
                curvature, vectors = np.linalg.eigh(.5 * (hessian + hessian.T))
                if np.linalg.norm(current[1]) < 1e-9 and curvature[0] >= -1e-8:
                    stationary_minimum = True
                    break
                # Keep resolved soft curvature; a relative 1e-4 floor can
                # suppress a physical mode by three orders of magnitude.
                floor = max(1e-8, 64 * np.finfo(float).eps * np.max(np.abs(curvature)))
                inverse = (vectors / np.maximum(np.abs(curvature), floor)) @ vectors.T
                direction = -tangent @ inverse @ current[1]
                direction /= max(1., np.linalg.norm(direction))
                if curvature[0] < -1e-8:
                    unstable = tangent @ vectors[:, 0]
                    unstable /= np.linalg.norm(unstable)
                    if np.dot(current[1], unstable) > 0:
                        unstable *= -1
                    predicted = np.dot(current[1], direction) + .5 * direction @ hessian @ direction
                    unstable_predicted = np.dot(current[1], unstable) + .5 * curvature[0]
                    if unstable_predicted < predicted:
                        direction = unstable
                direction -= x * np.dot(x, direction)
                slope = np.dot(current[1], direction)
                alpha, accepted = 1., False
                for _ in range(24):
                    trial = x + alpha * direction
                    trial /= np.linalg.norm(trial)
                    next_value = value(trial)
                    # Exact retraction difference. Subtracting two outer
                    # products loses O(eps) precisely when the Newton decrease
                    # is O(step**2); this expression scales its error with step.
                    move = alpha * direction
                    move2 = np.dot(move, move)
                    delta = sign * 2 * (
                        np.outer(x, move) + np.outer(move, x)
                        + np.outer(move, move) - move2 * np.outer(x, x)
                    ) / (1 + move2)
                    response = B.T @ (next_value[3] - current[3]) @ B
                    # Compare with the current iterate directly, avoiding the
                    # cancellation of two nearly equal reference-relative energies.
                    delta_energy = float(np.sum(delta * (
                        Fsub - np.dot(x, Fsub @ x) * np.eye(g) + .5 * response
                    )))
                    if delta_energy <= 1e-4 * alpha * slope:
                        accepted = True
                        break
                    alpha *= .5
                if not accepted:
                    break
                x, current = trial, next_value
            if stationary_minimum and (
                    best is None or current[0] < best[0] - 1e-12):
                best = current
    except Exhausted:
        return out
    if best is None:
        raise RuntimeError("MSINDO CCM initial-frontier optimization did not converge")
    out.density, out.fock, out.energy = best[2:]
    out.completed = True
    return out


def frontier_seed(H, nocc, budget, evaluate, allow_resolved=False):
    """Minimize a rank-one frontier and repair adjacent-block instabilities.

    The next occupied/virtual Hcore block is added only when the current
    rank-one projector has curvature below -1e-8 toward that block. The
    full block must fit the eight-orbital limit. Continue from the existing
    projector after expansion; explicit user densities still bypass this
    initial-guess operation. This bounded search is not a global SCF audit.
    ``allow_resolved`` also starts from the highest occupied and lowest
    virtual whole Hcore blocks for the explicit lowest-state policy. The
    reference policy retains its resolved-gap Hcore trajectory.
    Screening Fock evaluations consume the same cumulative budget as all
    other seed work, including when the budget expires during screening.
    """
    bounds = None
    if allow_resolved and 0 < nocc < len(H):
        eps = np.linalg.eigvalsh(H)
        if eps[nocc] - eps[nocc - 1] > 1e-6:
            lo, hi = nocc - 1, nocc + 1
            while lo > 0 and abs(eps[lo - 1] - eps[nocc - 1]) < 1e-6:
                lo -= 1
            while hi < len(H) and abs(eps[hi] - eps[nocc]) < 1e-6:
                hi += 1
            bounds = (lo, hi)
    seed = _minimize_frontier(H, nocc, budget, evaluate, bounds=bounds)
    if not seed.applicable or not seed.completed:
        return seed
    eps, C = np.linalg.eigh(H)
    used = seed.evaluations
    expansions = 0
    while True:
        lo, hi = seed.orbital_begin, seed.orbital_end
        g, q = hi - lo, nocc - lo
        candidates = []
        for hole in (False, True):
            if (not hole and q != 1) or (hole and q != g - 1):
                continue
            if hole:
                if lo == 0:
                    continue
                begin, end = lo - 1, lo
                while begin > 0 and abs(eps[begin - 1] - eps[lo - 1]) < 1e-6:
                    begin -= 1
                bounds = (begin, hi)
            else:
                if hi == len(H):
                    continue
                begin, end = hi, hi + 1
                while end < len(H) and abs(eps[end] - eps[hi]) < 1e-6:
                    end += 1
                bounds = (lo, end)
            if bounds[1] - bounds[0] > 8:
                continue
            sign, fixed = (-1., hi) if hole else (1., lo)
            base = 2 * C[:, :fixed] @ C[:, :fixed].T
            B = C[:, lo:hi]
            D = sign * .5 * B.T @ (seed.density - base) @ B
            _, vectors = np.linalg.eigh(.5 * (D + D.T))
            u = B @ vectors[:, -1]
            Y = C[:, begin:end]
            k = Y.shape[1]
            Fsub = Y.T @ seed.fock @ Y
            multiplier = 4 * sign * np.dot(u, seed.fock @ u)
            hessian = 4 * sign * Fsub - multiplier * np.eye(k)
            for j in range(k):
                if used >= budget:
                    seed.completed = False
                    seed.evaluations = used
                    return seed
                y = Y[:, j]
                delta = 2 * sign * (np.outer(y, u) + np.outer(u, y))
                F, _ = evaluate(seed.density + delta)
                used += 1
                hessian[:, j] += 4 * sign * Y.T @ (F - seed.fock) @ u
            curvature = float(np.linalg.eigvalsh(.5 * (hessian + hessian.T))[0])
            if curvature < -1e-8:
                candidates.append((curvature, bounds))
        seed.evaluations = used
        if not candidates:
            return seed
        bounds = min(candidates, key=lambda item: item[0])[1]
        if used >= budget:
            seed.completed = False
            return seed
        seed = _minimize_frontier(
            H, nocc, budget - used, evaluate, bounds=bounds,
            initial_density=seed.density,
        )
        used += seed.evaluations
        seed.evaluations = used
        expansions += 1
        seed.expansions = expansions
        if not seed.completed:
            return seed
