"""Interaction energies and the sigma potential -- Klamt 1998 eq. 18, 22, 23, 26.

The sigma potential ``mu_S(sigma)`` is the free energy per unit area of adding
a surface patch of screening charge density ``sigma`` to the ensemble ``S``.
It is the object the whole theory turns on: every chemical potential, activity
coefficient and partition coefficient is an integral of a solute's sigma
profile against the solvent's sigma potential.

It is defined implicitly (eq. 17) and obtained by fixed-point iteration.

Two formulations
----------------
* :func:`sigma_potential_profile` -- the classic one-dimensional form
  (eq. 18) over a sigma histogram. Fast, and the form the published figures
  show. It carries one descriptor, so it uses the misfit energy *without* the
  correlation term.
* :func:`sigma_potential_segments` -- the segment-sum form (eq. 23-25). Klamt
  1998 section 3.4 introduces it precisely because adding a second descriptor
  would otherwise require a two-dimensional histogram: "It is more efficient
  to replace the multidimensional integral by an appropriately weighted sum
  over all the segments making up the solvent." This is the form that carries
  ``sigma_perp``, and therefore the one that matches the published
  parameterization, which was fitted with the correlation term (eq. 26).

Prefer the segment form when the parameterization has a non-zero ``f_corr``.
The profile form is not an approximation *of* it in a controlled sense -- it
is a different, one-descriptor model -- so they are exposed separately rather
than one silently standing in for the other.
"""

from __future__ import annotations

import numpy as np

from .parameters import R_KCAL, Parameterization, T_ROOM_K
from .sigma import SegmentDescriptors, SigmaProfile

# Fixed-point iteration controls. Klamt notes the solve "takes milliseconds";
# the damping exists because the undamped map can oscillate for strongly
# hydrogen-bonding profiles rather than because convergence is slow.
DEFAULT_MAX_ITER = 500
DEFAULT_TOL = 1e-10
DEFAULT_MIXING = 0.5


def misfit_energy(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    params: Parameterization,
    sigma_perp_a: np.ndarray | None = None,
    sigma_perp_b: np.ndarray | None = None,
) -> np.ndarray:
    """Electrostatic misfit energy per unit area, kcal/(mol angstrom^2).

    Klamt 1998 eq. 7 without correlation::

        E_misfit = (alpha'/2) (sigma + sigma')^2

    and eq. 26 with it::

        E_misfit = (alpha'/2) (sigma + sigma')
                   [(sigma + sigma') + f_corr (sigma_perp + sigma_perp')]

    The sign convention matters and is easy to get backwards: the energy is
    *positive* for a pair that fails to cancel (``sigma + sigma' != 0``) and
    zero for the ideally paired contact ``sigma' = -sigma``. It is the penalty
    for non-ideal pairing relative to the ideally screened reference, not a
    binding energy.
    """
    a = np.asarray(sigma_a, dtype=np.float64)
    b = np.asarray(sigma_b, dtype=np.float64)
    s = a + b
    if (
        sigma_perp_a is None
        or sigma_perp_b is None
        or params.f_corr == 0.0
    ):
        return 0.5 * params.alpha_prime * s * s
    sp = np.asarray(sigma_perp_a, dtype=np.float64) + np.asarray(
        sigma_perp_b, dtype=np.float64
    )
    return 0.5 * params.alpha_prime * s * (s + params.f_corr * sp)


def hydrogen_bond_energy(
    sigma_a: np.ndarray, sigma_b: np.ndarray, params: Parameterization
) -> np.ndarray:
    """Hydrogen-bond energy per unit area, kcal/(mol angstrom^2).

    Klamt 1998 eq. 22::

        E_hb = c_hb max[0, sigma_acc - sigma_hb] min[0, sigma_don + sigma_hb]

    with ``sigma_acc`` the larger and ``sigma_don`` the smaller of the two
    densities. The expression is non-zero only when the pair straddles the
    thresholds with opposite signs, so it is "almost zero for nonpolar or
    moderately polar interactions, but becoming important for strongly polar
    surface contacts" -- and it is negative, i.e. stabilising, unlike the
    misfit term.
    """
    a = np.asarray(sigma_a, dtype=np.float64)
    b = np.asarray(sigma_b, dtype=np.float64)
    acc = np.maximum(a, b)
    don = np.minimum(a, b)
    return (
        params.c_hb
        * np.maximum(0.0, acc - params.sigma_hb)
        * np.minimum(0.0, don + params.sigma_hb)
    )


def interaction_energy(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    params: Parameterization,
    sigma_perp_a: np.ndarray | None = None,
    sigma_perp_b: np.ndarray | None = None,
) -> np.ndarray:
    """Total pair interaction per unit area: misfit plus hydrogen bond."""
    return misfit_energy(
        sigma_a, sigma_b, params, sigma_perp_a, sigma_perp_b
    ) + hydrogen_bond_energy(sigma_a, sigma_b, params)


def interaction_energy_dsigma(
    sigma_a: np.ndarray,
    sigma_b: np.ndarray,
    params: Parameterization,
    sigma_perp_a: np.ndarray | None = None,
    sigma_perp_b: np.ndarray | None = None,
) -> np.ndarray:
    """``d E_int(sigma_a, sigma_b) / d sigma_a``, kcal/(mol angstrom^2) per
    (e/angstrom^2).

    Direct COSMO-RS needs this and nothing else needs it yet: the feedback
    potential of Sinnecker, Rajendran, Klamt, Diedenhofen & Neese 2006
    (doi:10.1021/jp056016z) eq. 19 is
    ``phi_t = a_t (d mu_S / d q)|_{q_t}``, and with ``sigma = q/a`` that is
    exactly ``mu_S'(sigma_t)`` -- whose own derivative chain bottoms out here.

    The misfit term differentiates cleanly. With correlation (Klamt 1998
    eq. 26) ``E = (alpha'/2) s (s + f_corr sp)`` where ``s = sigma_a + sigma_b``
    and ``sp = sigma_perp_a + sigma_perp_b``, so::

        dE/dsigma_a = alpha' s + (alpha'/2) f_corr sp

    ``sigma_perp`` is an independent descriptor and carries no derivative of its
    own.

    The hydrogen-bond term does not differentiate cleanly, and pretending
    otherwise would be the error to make here. Klamt 1998 eq. 22 is
    ``E_hb = c_hb max[0, acc - sigma_hb] min[0, don + sigma_hb]`` with ``acc``
    and ``don`` the larger and smaller of the pair, so as a function of
    ``sigma_a`` it is piecewise linear with a corner where ``sigma_a`` crosses
    ``+-sigma_hb``. The derivative therefore **jumps** there, by ``c_hb`` times
    the partner's factor, and no amount of algebra removes it: the corner is in
    the published functional form. What that costs a self-consistent feedback is
    measured rather than assumed -- see the tests.

    Which of the pair is acceptor and which donor swaps at ``sigma_a ==
    sigma_b``, and that swap is *not* a corner: the H-bond term needs
    ``acc > sigma_hb`` and ``don < -sigma_hb`` simultaneously, which two equal
    densities cannot satisfy, so the term is identically zero in a neighbourhood
    of the swap.
    """
    a = np.asarray(sigma_a, dtype=np.float64)
    b = np.asarray(sigma_b, dtype=np.float64)
    s = a + b
    if (
        sigma_perp_a is None
        or sigma_perp_b is None
        or params.f_corr == 0.0
    ):
        d_misfit = params.alpha_prime * s
    else:
        sp = np.asarray(sigma_perp_a, dtype=np.float64) + np.asarray(
            sigma_perp_b, dtype=np.float64
        )
        d_misfit = params.alpha_prime * s + 0.5 * params.alpha_prime * params.f_corr * sp

    # ``a`` is the acceptor where it is the larger of the pair, the donor where
    # it is the smaller; only one branch is live at any point.
    a_is_acc = a >= b
    as_acc = np.where(a > params.sigma_hb, 1.0, 0.0) * np.minimum(
        0.0, b + params.sigma_hb
    )
    as_don = np.maximum(0.0, b - params.sigma_hb) * np.where(
        a < -params.sigma_hb, 1.0, 0.0
    )
    d_hb = params.c_hb * np.where(a_is_acc, as_acc, as_don)
    return d_misfit + d_hb


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable ``log(sum(exp(a)))``.

    The exponent in eq. 18 is a difference of energies scaled by ``1/beta``,
    which for strongly polar segments reaches tens of units. Exponentiating
    directly overflows to ``inf`` and the potential comes back as NaN, so the
    shift is not optional hygiene.
    """
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    return np.squeeze(
        amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True)),
        axis=axis,
    )


def _fixed_point(update, mu0, *, max_iter, tol, mixing, what):
    mu = np.array(mu0, dtype=np.float64)
    for it in range(1, int(max_iter) + 1):
        mu_new = update(mu)
        if not np.all(np.isfinite(mu_new)):
            raise RuntimeError(
                f"{what}: the self-consistent iteration produced a non-finite "
                f"value at step {it}. The sigma grid or profile is degenerate."
            )
        delta = float(np.max(np.abs(mu_new - mu)))
        mu = (1.0 - mixing) * mu + mixing * mu_new
        if delta < tol:
            return mu, it, delta
    raise RuntimeError(
        f"{what}: no self-consistent sigma potential after {max_iter} "
        f"iterations (last change {delta:.3e} > {tol:.1e}). Reduce `mixing` "
        f"or widen the sigma grid."
    )


class SigmaPotential:
    """A converged sigma potential, callable on arbitrary ``sigma``.

    ``mu_tilde`` is dimensionless (units of ``beta``); ``mu`` is
    kcal/(mol angstrom^2). Interpolation between grid points is linear, which
    is consistent with the linear binning used to build the profile.
    """

    def __init__(
        self, sigma_grid, mu_tilde, params, temperature_k, n_iter, residual,
        *, ensemble=None,
    ):
        self.sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
        self.mu_tilde = np.asarray(mu_tilde, dtype=np.float64)
        self.params = params
        self.temperature_k = float(temperature_k)
        self.n_iter = int(n_iter)
        self.residual = float(residual)
        # The solvent ensemble the potential was converged against, kept so the
        # potential can evaluate *itself* rather than a tabulation of itself.
        #
        # Both constructors end in the same expression -- ``mu~(sigma) =
        # -logsumexp_j(log w_j - E~(sigma, sigma_j) + mu~_j)`` -- and then throw
        # the ensemble away and keep a grid. That is fine for reading a
        # sigma potential off and enough for the thermodynamics, but Direct
        # COSMO-RS needs the *derivative*, and differentiating a linear
        # interpolant gives a piecewise-constant answer that is discontinuous at
        # every grid point. With the ensemble the derivative is exact and
        # available at arbitrary sigma; see :meth:`derivative`.
        self._ensemble = ensemble

    @property
    def beta(self) -> float:
        return self.params.beta(self.temperature_k)

    @property
    def mu(self) -> np.ndarray:
        """``mu'(sigma)`` in kcal/(mol angstrom^2)."""
        return self.beta * self.mu_tilde

    def __call__(self, sigma) -> np.ndarray:
        """``mu_tilde`` at arbitrary sigma, linearly interpolated."""
        return np.interp(
            np.asarray(sigma, dtype=np.float64),
            self.sigma_grid,
            self.mu_tilde,
            left=self.mu_tilde[0],
            right=self.mu_tilde[-1],
        )

    @property
    def has_ensemble(self) -> bool:
        """Whether :meth:`evaluate` and :meth:`derivative` are available."""
        return self._ensemble is not None

    def _log_terms(self, sigma, sigma_perp):
        """``log w_j - E~(sigma, sigma_j) + mu~_j`` for every solvent segment."""
        if self._ensemble is None:
            raise ValueError(
                "SigmaPotential: this potential was built without its solvent "
                "ensemble, so it can only be interpolated, not evaluated. Build "
                "it with sigma_potential_profile or sigma_potential_segments, "
                "which carry the ensemble."
            )
        log_w, sig_j, mu_j, perp_j = self._ensemble
        s = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
        perp = (
            np.zeros_like(s) if sigma_perp is None
            else np.broadcast_to(np.asarray(sigma_perp, dtype=np.float64), s.shape)
        )
        e_tilde = interaction_energy(
            s[:, None], sig_j[None, :], self.params,
            perp[:, None] if perp_j is not None else None,
            perp_j[None, :] if perp_j is not None else None,
        ) / self.beta
        return s, perp, sig_j, perp_j, log_w[None, :] - e_tilde + mu_j[None, :]

    def evaluate(self, sigma, sigma_perp=None) -> np.ndarray:
        """``mu~(sigma)`` from the defining expression, not from the table.

        Agrees with :meth:`__call__` to the tabulation error, and is the form
        Direct COSMO-RS uses so that its feedback potential is the derivative of
        the value it actually pairs with.
        """
        s, _, _, _, terms = self._log_terms(sigma, sigma_perp)
        out = -_logsumexp(terms, axis=1)
        return out if np.ndim(sigma) else float(out[0])

    def derivative(self, sigma, sigma_perp=None) -> np.ndarray:
        """``d mu~ / d sigma``, exactly.

        Differentiating ``mu~(sigma) = -log sum_j exp(L_j)`` with
        ``L_j = log w_j - E~(sigma, sigma_j) + mu~_j`` gives a Boltzmann average
        over the solvent ensemble::

            d mu~ / d sigma = sum_j p_j  dE~(sigma, sigma_j) / dsigma,
            p_j = exp(L_j) / sum_k exp(L_k)

        so it costs one pass over the same terms the value needs. Note what the
        average does *not* remove: the hydrogen-bond corner of
        :func:`interaction_energy_dsigma` sits at ``sigma = +-sigma_hb`` for
        **every** partner ``sigma_j``, so every term in the average steps at the
        same place and the average steps with them. A Boltzmann average smooths
        over partners, not over a corner they share.
        """
        s, perp, sig_j, perp_j, terms = self._log_terms(sigma, sigma_perp)
        p = np.exp(terms - _logsumexp(terms, axis=1)[:, None])
        de = interaction_energy_dsigma(
            s[:, None], sig_j[None, :], self.params,
            perp[:, None] if perp_j is not None else None,
            perp_j[None, :] if perp_j is not None else None,
        ) / self.beta
        out = np.einsum("ij,ij->i", p, de)
        return out if np.ndim(sigma) else float(out[0])

    def __repr__(self) -> str:  # pragma: no cover - diagnostic
        return (
            f"SigmaPotential(params={self.params.name!r}, "
            f"T={self.temperature_k:.2f} K, n_iter={self.n_iter}, "
            f"residual={self.residual:.2e})"
        )


def sigma_potential_profile(
    profile: SigmaProfile,
    params: Parameterization,
    *,
    temperature_k: float = T_ROOM_K,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    mixing: float = DEFAULT_MIXING,
) -> SigmaPotential:
    """One-dimensional sigma potential, Klamt 1998 eq. 18.

        mu~(sigma) = -ln[ int dsigma' p'(sigma') exp(-E~(sigma,sigma')
                                                     + mu~(sigma')) ]

    with ``E~ = E / beta`` and ``beta = kT / a_eff``. Carries the single
    descriptor ``sigma``, so the misfit is evaluated without the correlation
    term regardless of the parameterization's ``f_corr``; use
    :func:`sigma_potential_segments` when that term is wanted.
    """
    grid = profile.sigma_grid
    beta = params.beta(temperature_k)
    # E~(sigma_i, sigma_j), dimensionless.
    e_tilde = (
        misfit_energy(grid[:, None], grid[None, :], params)
        + hydrogen_bond_energy(grid[:, None], grid[None, :], params)
    ) / beta

    weight = profile.normalized * profile.dsigma      # p'(sigma') dsigma'
    positive = weight > 0.0
    if not np.any(positive):
        raise ValueError(
            "sigma_potential_profile: the profile carries no area; a sigma "
            "potential is undefined for an empty ensemble."
        )
    log_w = np.full(weight.shape, -np.inf)
    log_w[positive] = np.log(weight[positive])

    def update(mu):
        return -_logsumexp(log_w[None, :] - e_tilde + mu[None, :], axis=1)

    mu, n_iter, residual = _fixed_point(
        update, np.zeros(grid.size), max_iter=max_iter, tol=tol,
        mixing=mixing, what="sigma_potential_profile",
    )
    return SigmaPotential(
        grid, mu, params, temperature_k, n_iter, residual,
        ensemble=(log_w, grid, mu, None),
    )


def sigma_potential_segments(
    segments: list[SegmentDescriptors],
    mole_fractions: np.ndarray,
    params: Parameterization,
    *,
    temperature_k: float = T_ROOM_K,
    sigma_grid: np.ndarray | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    mixing: float = DEFAULT_MIXING,
) -> SigmaPotential:
    """Segment-sum sigma potential, Klamt 1998 eq. 23-25.

    Iterates over the solvent's actual segments rather than a histogram, so it
    carries both ``sigma`` and ``sigma_perp`` and evaluates the misfit with the
    correlation term of eq. 26 -- the form the published parameterization was
    fitted with.

        mu~(d) = -ln[ W^-1 sum_i x_i sum_{nu in i} s_nu
                      exp(-E~(d, d_nu) + mu~(d_nu)) ]
        W = sum_i x_i sum_{nu in i} s_nu

    The returned potential is tabulated on ``sigma_grid`` for interpolation,
    but the self-consistency is solved on the segments themselves; the grid is
    an output view, not the state.
    """
    if not segments:
        raise ValueError("sigma_potential_segments: no components.")
    x = np.asarray(mole_fractions, dtype=np.float64)
    if x.shape != (len(segments),):
        raise ValueError(
            f"sigma_potential_segments: {x.size} mole fractions for "
            f"{len(segments)} components."
        )
    if np.any(x < 0.0) or float(np.sum(x)) <= 0.0:
        raise ValueError(
            "sigma_potential_segments: mole fractions must be non-negative "
            "and not all zero."
        )

    # Flatten the solvent's segments, weighting each component's areas by its
    # mole fraction (eq. 24's W).
    areas = np.concatenate(
        [xi * np.asarray(seg.areas, dtype=np.float64) for xi, seg in zip(x, segments)]
    )
    sigma = np.concatenate([np.asarray(s.sigma) for s in segments])
    sigma_perp = np.concatenate([np.asarray(s.sigma_perp) for s in segments])
    W = float(np.sum(areas))
    if W <= 0.0:
        raise ValueError("sigma_potential_segments: the ensemble has no area.")

    beta = params.beta(temperature_k)
    e_tilde = interaction_energy(
        sigma[:, None], sigma[None, :], params,
        sigma_perp[:, None], sigma_perp[None, :],
    ) / beta

    positive = areas > 0.0
    log_w = np.full(areas.shape, -np.inf)
    log_w[positive] = np.log(areas[positive] / W)

    def update(mu):
        return -_logsumexp(log_w[None, :] - e_tilde + mu[None, :], axis=1)

    mu_seg, n_iter, residual = _fixed_point(
        update, np.zeros(areas.size), max_iter=max_iter, tol=tol,
        mixing=mixing, what="sigma_potential_segments",
    )

    # Tabulate onto a grid for interpolation by solutes.
    from .sigma import default_sigma_grid

    grid = default_sigma_grid() if sigma_grid is None else np.asarray(
        sigma_grid, dtype=np.float64
    )
    e_grid = interaction_energy(
        grid[:, None], sigma[None, :], params,
        np.zeros_like(grid)[:, None], sigma_perp[None, :],
    ) / beta
    mu_grid = -_logsumexp(log_w[None, :] - e_grid + mu_seg[None, :], axis=1)
    return SigmaPotential(
        grid, mu_grid, params, temperature_k, n_iter, residual,
        ensemble=(log_w, sigma, mu_seg, sigma_perp),
    )


__all__ = [
    "DEFAULT_MAX_ITER",
    "DEFAULT_MIXING",
    "DEFAULT_TOL",
    "SigmaPotential",
    "hydrogen_bond_energy",
    "interaction_energy",
    "interaction_energy_dsigma",
    "misfit_energy",
    "sigma_potential_profile",
    "sigma_potential_segments",
]
