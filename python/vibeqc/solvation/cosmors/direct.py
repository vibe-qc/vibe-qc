"""Direct COSMO-RS: the sigma potential fed back into the Hamiltonian (#558).

Sinnecker, Rajendran, Klamt, Diedenhofen & Neese, *J. Phys. Chem. A* **110**,
2235 (2006), doi:10.1021/jp056016z -- the method's original paper, which gives
it in three equations and spends the rest of its length on g-tensors::

    q         = -A^-1 phi                                   (eq. 7)
    phi_t^dRS = a_t (d mu_S / dq)|_{q_t} = mu_S'(sigma_t)   (eq. 19)
    V^RS      = -sum_t (q_t + q_t^dRS) / |r - r_t|          (eq. 18)

with ``q^dRS`` solved from ``phi^dRS`` through the same eq. 7 and the whole
thing iterated to self-consistency.

Why the operator is exactly ``q + q^dRS``
-----------------------------------------
The paper states eq. 18 without deriving it, and it is worth seeing why,
because it also fixes the energy -- which the paper does *not* state. Take the
free energy to be the conductor COSMO energy plus the COSMO-RS chemical
potential of the solute's own surface::

    G = E_gas[D] + (1/2) q . V_total + sum_t a_t mu_S(sigma_t)

The last term's derivative with respect to the density is, with
``sigma_t = q_t / a_t`` and ``q = -A^-1 V``::

    dG_RS/dD = sum_t a_t mu_S'(sigma_t) (1/a_t) dq_t/dD
             = -sum_t mu_S'(sigma_t) sum_u (A^-1)_{tu} dV_u/dD
             = sum_u q_u^dRS dV_u/dD          with q^dRS = -A^-1 phi^dRS

which is the potential of a charge distribution ``q^dRS`` on the same segments
-- i.e. ``fock_contribution(q^dRS)``. Added to the conductor term's own
``fock_contribution(q)`` it gives eq. 18. So eq. 18 and the energy above are
the same statement, and using one without the other would leave the operator
inconsistent with the energy it is supposed to be the derivative of.

Conductor charges, not screened ones
------------------------------------
``f(eps)`` does not appear. The paper is explicit that the correction acts on
"the ideal screening charges appearing in a conductor", and that is the whole
point: the sigma potential *is* the real-solvent physics, so scaling by
``f(eps)`` as well would count the solvent twice. Direct COSMO-RS replaces the
dielectric scaling rather than composing with it, which is why
:class:`~vibeqc.solvation.driver.SolventModel` fixes the conductor limit for
this variant instead of letting a dielectric constant ride along.

This also matches how the rest of this package already feeds COSMO-RS: sigma
profiles are read off the *conductor* surface record, never a screened one.

Units
-----
The electronic-structure side works in Hartree, bohr and e; the COSMO-RS side
in kcal/mol, angstrom and e/angstrom^2. Every conversion is in
:func:`direct_cosmors_feedback` and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..cavity import ANG_TO_BOHR

BOHR_TO_ANG = 1.0 / ANG_TO_BOHR

# CODATA; the same value atomization.py uses.
HARTREE_TO_KCAL = 627.509474063

# Segments smaller than this (bohr^2) get no feedback rather than a sigma of
# q/0. The threshold is the surface record's own, so a segment that is
# degenerate for the sigma profile is degenerate here too.
from ..surface import MIN_SEGMENT_AREA_BOHR2  # noqa: E402

__all__ = [
    "BOHR_TO_ANG",
    "HARTREE_TO_KCAL",
    "DirectFeedback",
    "direct_cosmors_feedback",
]


@dataclass(frozen=True)
class DirectFeedback:
    """One evaluation of the Direct COSMO-RS correction.

    Attributes
    ----------
    phi : ndarray (n_seg,), Hartree/e
        ``phi^dRS_t = mu_S'(sigma_t)``, eq. 19, ready for the same ``A`` solve
        the conductor charges came from.
    sigma : ndarray (n_seg,), e/angstrom^2
        The screening charge density the potential was evaluated at. Carried
        because it is what a convergence problem is diagnosed from: see
        :attr:`fraction_near_hb_corner`.
    energy : float, Hartree
        ``sum_t a_t mu_S(sigma_t)``, the COSMO-RS term of the free energy.
    near_hb_corner : ndarray (n_seg,), bool
        Segments whose sigma sits within one ``sigma`` bin of ``+-sigma_hb``,
        where ``mu_S'`` is discontinuous.
    charges : ndarray (n_seg,), e
        ``q^dRS = -A^-1 phi^dRS``, filled in by whoever owns the ``A`` solve.
        Zero as constructed, because this module does not own a factorization
        and inventing one would be a second, differently-conditioned ``A``
        beside the energy's -- the defect shape this package keeps meeting.
    """

    phi: np.ndarray
    sigma: np.ndarray
    energy: float
    near_hb_corner: np.ndarray
    charges: np.ndarray = None  # type: ignore[assignment]
    _outside_area_fraction: float = 0.0

    def __post_init__(self):
        if self.charges is None:
            object.__setattr__(self, "charges", np.zeros_like(self.phi))

    @property
    def fraction_outside_grid(self) -> float:
        """Share of *area* whose sigma falls outside the potential's grid.

        The sigma potential is fitted on a bounded sigma range, and the
        evaluated form extrapolates beyond it through the misfit term -- which
        is quadratic and defined everywhere, so the extrapolation is smooth and
        physically sensible rather than a clamp, but it is still extrapolation.
        A few outlying segments are normal; a large share means the surface and
        the parameterization disagree about what a polar segment looks like.
        """
        return self._outside_area_fraction

    @property
    def fraction_near_hb_corner(self) -> float:
        """Share of segments sitting on the hydrogen-bond corner.

        Klamt's H-bond term is piecewise linear in each density, so ``mu_S'``
        steps where a segment's sigma crosses ``+-sigma_hb``, and the Boltzmann
        average over solvent partners does not damp it -- every partner steps at
        the same sigma. A segment parked there sees a discontinuous potential
        from one macro-iteration to the next, so this is the first number to
        look at when the outer loop will not settle.
        """
        n = self.near_hb_corner.size
        return float(np.count_nonzero(self.near_hb_corner)) / n if n else 0.0


def direct_cosmors_feedback(
    charges: np.ndarray,
    areas_bohr2: np.ndarray,
    potential,
    *,
    corner_window: float = 1.0e-3,
) -> DirectFeedback:
    """Eq. 19's correction potential, and the energy term it differentiates.

    Parameters
    ----------
    charges
        Conductor screening charges ``q_t`` (e), one per segment.
    areas_bohr2
        Segment areas (bohr^2), the same ones the cavity carries.
    potential
        A converged :class:`~vibeqc.solvation.cosmors.potential.SigmaPotential`
        for the solvent, carrying its ensemble.
    corner_window
        Half-width in sigma (e/angstrom^2) for reporting proximity to the
        hydrogen-bond corner. Diagnostic only; it enters no number.

    ``evaluate`` and ``derivative`` are used rather than ``__call__``, and that
    is not interchangeable: ``__call__`` interpolates the tabulated potential
    and is up to 9.9e-03 away from the evaluated curve between grid points, so
    an interpolated energy paired with an exact potential would be an operator
    that is not the derivative of its own energy.
    """
    q = np.asarray(charges, dtype=np.float64).reshape(-1)
    a_bohr = np.asarray(areas_bohr2, dtype=np.float64).reshape(-1)
    if q.shape != a_bohr.shape:
        raise ValueError(
            f"direct_cosmors_feedback: {q.shape[0]} charges against "
            f"{a_bohr.shape[0]} segment areas."
        )
    if not getattr(potential, "has_ensemble", False):
        raise ValueError(
            "direct_cosmors_feedback: the sigma potential does not carry its "
            "solvent ensemble, so it can only be interpolated and its "
            "derivative would be piecewise constant. Build it with "
            "sigma_potential_profile or sigma_potential_segments."
        )

    live = a_bohr > MIN_SEGMENT_AREA_BOHR2
    a_ang = a_bohr * (BOHR_TO_ANG ** 2)

    sigma = np.zeros_like(q)
    sigma[live] = q[live] / a_ang[live]                     # e / angstrom^2

    beta = potential.beta                                   # kcal/(mol angstrom^2)
    phi = np.zeros_like(q)
    # mu~ is dimensionless, so d(mu~)/d(sigma) needs beta to become
    # kcal/(mol angstrom^2) per (e/angstrom^2) = kcal/(mol e), and then the
    # Hartree conversion to be a potential in atomic units.
    phi[live] = (
        beta * np.asarray(potential.derivative(sigma[live]), dtype=np.float64)
    ) / HARTREE_TO_KCAL

    mu = beta * np.asarray(potential.evaluate(sigma[live]), dtype=np.float64)
    energy = float(np.dot(a_ang[live], mu)) / HARTREE_TO_KCAL

    s_hb = float(potential.params.sigma_hb)
    near = live & (np.abs(np.abs(sigma) - s_hb) < float(corner_window))

    grid = np.asarray(potential.sigma_grid, dtype=np.float64)
    outside = live & ((sigma < grid[0]) | (sigma > grid[-1]))
    total = float(a_ang[live].sum())
    outside_fraction = (
        float(a_ang[outside].sum()) / total if total > 0.0 else 0.0
    )
    return DirectFeedback(
        phi=phi, sigma=sigma, energy=energy, near_hb_corner=near,
        _outside_area_fraction=outside_fraction,
    )
