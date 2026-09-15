"""COSMOSPACE -- the exact surface-pair activity coefficient equation.

Klamt, Krooshof & Taylor, *AIChE J.* **48**, 2332 (2002),
doi:10.1002/aic.690481023: "COSMOSPACE: Alternative to conventional
activity-coefficient models".

COSMO-RS as implemented in :mod:`vibeqc.solvation.cosmors.potential` treats
surface patches as *thermodynamically independent* entities (Klamt 1998
section 3.2). COSMOSPACE drops that approximation: it solves the statistics of
an ensemble of **pairwise** interacting segments exactly, giving segment
activity coefficients that satisfy the Gibbs-Duhem relation by construction
(2002 Appendix B).

The equation (eq. 16) is beautifully compact::

    1 / gamma^nu = sum_mu  tau_{mu nu} Theta^mu gamma^mu

with the segment fractions ``Theta`` and the exchange factor (eq. 17)::

    tau_{mu nu} = exp[ -(u_{mu nu} - (u_{mu mu} + u_{nu nu})/2) / RT ]

Note what the self-energy subtraction does. For the pure COSMO-RS misfit
energy ``E(s,s') = (alpha'/2)(s + s')^2`` the exchange combination collapses to

    u_{mu nu} - (u_{mu mu} + u_{nu nu})/2 = -(a_eff alpha'/2)(s_mu - s_nu)^2

which depends on the *difference* of the screening charge densities, not their
sum, and is never positive. That is the physical content: mixing two segment
types is always favourable relative to the pure-segment reference, and the
driving force is how unlike they are. Pinned in the tests, because getting the
self-energy subtraction wrong silently converts a difference into a sum and
still converges.

Why both models are kept
------------------------
COSMOSPACE is the exact solution of the same pairing problem the sigma
potential approximates, but the shipped parameterization was fitted with the
independent-patch sigma potential, so the two are not interchangeable at fixed
parameters. They are exposed side by side rather than one replacing the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import R_KCAL, Parameterization, T_ROOM_K
from .potential import interaction_energy

DEFAULT_MAX_ITER = 1000
DEFAULT_TOL = 1e-12
DEFAULT_MIXING = 0.5

# Staverman-Guggenheim lattice coordination number. Klamt 2002 after eq. 2:
# "The coordination number, z, commonly is assumed to be 10."
SG_COORDINATION_NUMBER = 10.0


def exchange_energy_matrix(
    sigma: np.ndarray,
    params: Parameterization,
    *,
    sigma_perp: np.ndarray | None = None,
) -> np.ndarray:
    """``u_{mu nu} - (u_{mu mu} + u_{nu nu})/2`` per contact, kcal/mol.

    The per-area interaction energy of
    :func:`~vibeqc.solvation.cosmors.potential.interaction_energy` is scaled by
    ``a_eff`` to give a per-contact energy, then referenced to the pure-segment
    pairs as eq. 17 requires.
    """
    s = np.asarray(sigma, dtype=np.float64)
    sp = None if sigma_perp is None else np.asarray(sigma_perp, dtype=np.float64)

    def u(a_idx, b_idx):
        if sp is None:
            return params.a_eff * interaction_energy(
                s[a_idx], s[b_idx], params
            )
        return params.a_eff * interaction_energy(
            s[a_idx], s[b_idx], params, sp[a_idx], sp[b_idx]
        )

    i = np.arange(s.size)
    u_pair = u(i[:, None], i[None, :])
    u_self = np.diag(u_pair)
    return u_pair - 0.5 * (u_self[:, None] + u_self[None, :])


def tau_matrix(
    exchange_kcal: np.ndarray, *, temperature_k: float = T_ROOM_K
) -> np.ndarray:
    """Klamt 2002 eq. 17. Symmetric, since ``u`` is assumed symmetric."""
    rt = R_KCAL * float(temperature_k)
    tau = np.exp(-np.asarray(exchange_kcal, dtype=np.float64) / rt)
    # Symmetrise against round-off so the solve inherits the exact symmetry
    # the derivation assumes (2002 section 2.2).
    return 0.5 * (tau + tau.T)


@dataclass(frozen=True)
class SegmentActivity:
    """Converged segment activity coefficients of eq. 16."""

    gamma: np.ndarray
    theta: np.ndarray
    tau: np.ndarray
    n_iter: int
    residual: float

    @property
    def ln_gamma(self) -> np.ndarray:
        return np.log(self.gamma)

    def gibbs_duhem_residual(self) -> float:
        """How far eq. 16 is from being satisfied, in max-norm.

        Zero means the solution is exact, and eq. 16 being satisfied is what
        makes the result thermodynamically consistent (2002 Appendix B). This
        is the quantity to assert on, not the iteration's own step size.
        """
        lhs = 1.0 / self.gamma
        rhs = self.tau.T @ (self.theta * self.gamma)
        return float(np.max(np.abs(lhs - rhs)))


def segment_activity_coefficients(
    theta: np.ndarray,
    tau: np.ndarray,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    mixing: float = DEFAULT_MIXING,
) -> SegmentActivity:
    """Solve eq. 16 by repeated substitution from ``gamma = 1``.

    Klamt 2002 section 2.2: "can be solved iteratively by simple repeated
    substitution, starting with the assumption of ``gamma^mu = 1`` on the right
    hand side". Damped, because the undamped map oscillates for large ``tau``.
    """
    th = np.asarray(theta, dtype=np.float64)
    t = np.asarray(tau, dtype=np.float64)
    if th.ndim != 1 or t.shape != (th.size, th.size):
        raise ValueError(
            f"segment_activity_coefficients: theta has shape {th.shape} but "
            f"tau has shape {t.shape}; expected ({th.size}, {th.size})."
        )
    if np.any(th < 0.0):
        raise ValueError("segment_activity_coefficients: negative Theta.")
    total = float(np.sum(th))
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(
            f"segment_activity_coefficients: Theta must sum to 1 (got "
            f"{total:.6g}); it is a segment *fraction* (eq. 7)."
        )

    gamma = np.ones(th.size, dtype=np.float64)
    residual = np.inf
    for it in range(1, int(max_iter) + 1):
        denom = t.T @ (th * gamma)
        if np.any(denom <= 0.0) or not np.all(np.isfinite(denom)):
            raise RuntimeError(
                "segment_activity_coefficients: the substitution produced a "
                "non-positive or non-finite denominator; tau or Theta is "
                "degenerate."
            )
        new = 1.0 / denom
        residual = float(np.max(np.abs(new - gamma)))
        gamma = (1.0 - mixing) * gamma + mixing * new
        if residual < tol:
            return SegmentActivity(gamma, th, t, it, residual)
    raise RuntimeError(
        f"segment_activity_coefficients: no solution after {max_iter} "
        f"iterations (last change {residual:.3e} > {tol:.1e})."
    )


def binary_segment_activity_coefficient(
    theta_a: float, tau_ab: float
) -> float:
    """Closed-form ``gamma^A`` for a two-segment ensemble, eq. 19.

        gamma^A = sqrt( 1/Theta^A
                        + (1 - sqrt(1 + 4 Theta^A Theta^B omega))
                          / (2 omega Theta^A^2) ),
        omega = tau_AB^-2 - 1

    Exists independently of the iteration, so it is the oracle the general
    solver is checked against rather than a convenience. Klamt notes
    ``gamma^A -> 1/tau_AB`` as ``Theta^A -> 0``, which is the other end of the
    check.
    """
    ta = float(theta_a)
    if not (0.0 <= ta <= 1.0):
        raise ValueError(
            f"binary_segment_activity_coefficient: Theta^A must be in [0, 1] "
            f"(got {ta})."
        )
    tb = 1.0 - ta
    omega = float(tau_ab) ** -2 - 1.0
    if ta == 0.0:
        # The stable form below already yields this exactly; kept explicit
        # because it is the limit the paper states.
        return 1.0 / float(tau_ab)

    # Eq. 19 as printed subtracts a square root from 1 and divides by
    # ``2 omega Theta_A^2``. Both vanish together as Theta_A -> 0, and in
    # floating point the subtraction cancels first: the printed form returns
    # NaN at Theta_A = 1e-9. Infinite dilution is exactly where activity
    # coefficients matter most, so rearrange it into a form with no
    # cancellation at all.
    #
    # With u = 4 Theta_A Theta_B omega and root = sqrt(1+u), using
    # root - 1 = u/(1+root) twice gives
    #
    #   gamma_A^2 = 2 (1 + 2 Theta_B omega / (1+root)) / (1+root)
    #
    # which is algebraically identical to eq. 19, has Theta_A cancelled
    # analytically rather than numerically, needs no division by omega (so
    # tau_AB = 1 requires no special case), and reproduces both published
    # limits exactly: Theta_A -> 0 gives gamma^2 = 1 + omega = tau_AB^-2, and
    # Theta_A = 1 gives gamma = 1.
    u = 4.0 * ta * tb * omega
    root = np.sqrt(1.0 + u)
    return float(np.sqrt(2.0 * (1.0 + 2.0 * tb * omega / (1.0 + root)) / (1.0 + root)))


def residual_ln_activity_coefficient(
    segment_counts: np.ndarray,
    ln_gamma_mixture: np.ndarray,
    ln_gamma_pure: np.ndarray,
) -> float:
    """Klamt 2002 eq. 13, the residual part for one compound.

        ln gamma_i^R = sum_nu n_i^nu (ln gamma^nu - ln gamma_i^nu)

    ``segment_counts`` is ``n_i^nu``, the number of segments of each type on
    molecule ``i`` (eq. 4: ``n_i = q_i / a_eff``). ``ln_gamma_pure`` is the same
    segment's activity coefficient in an ensemble of pure ``i``, which is what
    makes the result vanish for a compound in itself.
    """
    n = np.asarray(segment_counts, dtype=np.float64)
    mix = np.asarray(ln_gamma_mixture, dtype=np.float64)
    pure = np.asarray(ln_gamma_pure, dtype=np.float64)
    if not (n.shape == mix.shape == pure.shape):
        raise ValueError(
            f"residual_ln_activity_coefficient: shapes disagree "
            f"({n.shape}, {mix.shape}, {pure.shape})."
        )
    return float(np.sum(n * (mix - pure)))


def staverman_guggenheim_ln_gamma(
    mole_fractions: np.ndarray,
    relative_volumes: np.ndarray,
    relative_areas: np.ndarray,
    *,
    z: float = SG_COORDINATION_NUMBER,
) -> np.ndarray:
    """Combinatorial ``ln gamma_i^C``, the Staverman-Guggenheim term.

    Klamt 2002 eq. 2 gives the partition sum; the per-species activity
    coefficient that follows from it is the standard UNIQUAC combinatorial
    form::

        ln gamma_i^C = ln(Phi_i/x_i) + (z/2) q_i [1 - Phi_i/Theta_i
                                                  - ln(Phi_i/Theta_i)]
                       + l_i - (Phi_i/x_i) sum_j x_j l_j
        l_i = (z/2)(r_i - q_i) - (r_i - 1)

    ``r_i`` and ``q_i`` are relative volumes and areas on a consistently
    normalised scale. Klamt 2002 notes there is no exact expression for the
    combinatorial factor and that SG is an adequate approximation when the
    components differ in size by less than about a factor of five.
    """
    x = np.asarray(mole_fractions, dtype=np.float64)
    r = np.asarray(relative_volumes, dtype=np.float64)
    q = np.asarray(relative_areas, dtype=np.float64)
    if not (x.shape == r.shape == q.shape):
        raise ValueError(
            "staverman_guggenheim_ln_gamma: x, r and q must have one entry "
            "per component."
        )
    if np.any(x < 0.0):
        raise ValueError("staverman_guggenheim_ln_gamma: negative mole fraction.")
    xs = float(np.sum(x))
    if xs <= 0.0:
        raise ValueError("staverman_guggenheim_ln_gamma: mole fractions sum to 0.")
    x = x / xs

    phi = x * r / float(np.dot(x, r))
    theta = x * q / float(np.dot(x, q))
    ell = 0.5 * z * (r - q) - (r - 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_phi_x = np.where(x > 0.0, np.log(np.where(x > 0.0, phi / x, 1.0)), 0.0)
        ratio = np.where(theta > 0.0, phi / np.where(theta > 0.0, theta, 1.0), 1.0)
    return (
        ln_phi_x
        + 0.5 * z * q * (1.0 - ratio - np.log(ratio))
        + ell
        - (phi / np.where(x > 0.0, x, 1.0)) * float(np.dot(x, ell))
    )


__all__ = [
    "DEFAULT_MAX_ITER",
    "DEFAULT_MIXING",
    "DEFAULT_TOL",
    "SG_COORDINATION_NUMBER",
    "SegmentActivity",
    "binary_segment_activity_coefficient",
    "exchange_energy_matrix",
    "residual_ln_activity_coefficient",
    "segment_activity_coefficients",
    "staverman_guggenheim_ln_gamma",
    "tau_matrix",
]
