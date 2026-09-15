"""Extrapolation to the complete-PNO limit (Altun, Neese, Bistoni 2020).

The DLPNO correlation energy depends on how much of the pair natural orbital
space is kept, controlled by ``TCutPNO``. When the convergence precondition in
Altun et al. Eq. (1) holds for a named solver policy, tightening that threshold
approaches the limit smoothly but expensively. The historical attempt in this
code base was to *correct* a single truncated calculation by estimating the
discarded tail. For the retained all-electron
``residual_domain="pair", n_frozen=0, tcut_mkn=0`` witness, that failed for a
structural reason worth restating: **the truncation error was not
sign-definite.** On def2-SVP at ``TCutPNO = 1e-7``, H2O under-correlated by
``+158 uHa`` while N2 over-correlated by ``-964 uHa``. An additive tail
estimate carries one sign, so it cannot fix both, and adding it moved those
compact-molecule witnesses further from the oracle rather than closer.

This module implements the published alternative, which sidesteps the sign
question by extrapolating from *two* thresholds rather than correcting one.

Reference
---------
Ahmet Altun, Frank Neese and Giovanni Bistoni, "Extrapolation to the Limit of
a Complete Pair Natural Orbital Space in Local Coupled-Cluster Calculations",
J. Chem. Theory Comput. 16, 6142-6149 (2020), doi:10.1021/acs.jctc.0c00344.

Their Eq. (1) is the observed convergence of the correlation energy with
``X = -log10(TCutPNO)``::

    E^X = E + A * X**(-beta)

with ``E`` the complete-PNO-space limit; the paper fits ``A = 84.92`` and
``beta = 5.55`` (their Figure 1, benzene dimer, TightPNO/aug-cc-pVDZ-DK,
``R^2 = 1.000``). Writing that relation at two consecutive ``X`` and ``Y``
and eliminating ``A`` gives their Eq. (2)::

    E = (Y**beta * E^Y - X**beta * E^X) / (Y**beta - X**beta)

which, with the single parameter of Eq. (3)::

    F = Y**beta / (Y**beta - X**beta)

collapses to the compact two-point form of Eq. (4)::

    E = E^X + F * (E^Y - E^X)

Eq. (4) is what this module provides. Two consequences matter in practice:

* it makes **no assumption about the sign** of the truncation error, since it
  fits the approach to the limit rather than adding a correction, so
  over-correlating and under-correlating systems are handled alike;
* ``E^X`` and ``E^Y`` must come from otherwise **identical** calculations,
  same geometry, basis, and all other DLPNO settings, differing only in
  ``TCutPNO``. Mixing anything else in silently invalidates the fit.

.. warning::

   **This primitive is not automatically routed.** A caller must first
   establish a smooth, monotone approach to the complete-PNO limit for the
   operative solver policy; Eq. (1) presumes that behaviour and the paper's
   fit has ``R^2 = 1.000``. The retained 2026-08-03 negative control used the
   former local-solver recipe ``residual_domain="pair", n_frozen=0,
   tcut_mkn=0``. Its error against its own full-PNO-space energy, def2-SVP,
   in uHa, was::

       X:        5        6        7        8        9
       H2O:  -3662     -867     +158     +195     +120
       N2:   -4657     +198     -964     +536     +187

   The sign flips repeatedly across ``X = 5..8``, exactly the window a 5/6 or
   6/7 extrapolation uses, so the scheme extrapolates noise rather than a
   trend. In that legacy pair-domain recipe it landed *further* from the
   limit than its own tighter input: for H2O 6/7 the raw ``E^7`` error was
   ``+158`` uHa and the extrapolate was ``+916`` uHa. This historical witness
   is not a claim about the current atom-based ``extended`` residual-domain
   default, whose threshold ladder has a separate monotonic regression. The
   primitive below remains explicit until broader benchmark evidence validates
   extrapolation for a named operative policy; its constants are independently
   pinned to the paper.

On the parameter ``F``: the paper reports that the optimal value, the one
minimising the error against canonical CCSD(T) over the GMTKN55 subsets,
deviates within ``1.5 +/- 0.2``, and recommends a flat ``F = 1.5`` for both
the 5/6 and 6/7 extrapolations for simplicity. That empirical value is the
default here. :func:`extrapolation_factor` returns the analytic Eq. (3) value
instead for callers who want the exact inversion of Eq. (1); note it does not
equal 1.5 (it is about 1.57 for 5/6 and 1.74 for 6/7), because the
recommendation is fitted to benchmark error rather than derived.
"""

from __future__ import annotations

__all__ = [
    "CITATION_KEYS",
    "PNO_EXTRAPOLATION_BETA",
    "RECOMMENDED_F",
    "extrapolate_pno_limit",
    "extrapolation_factor",
    "threshold_exponent",
]

#: Citation keys a route must surface when this scheme is used. The primitive
#: names its own reference so that whatever eventually routes it can pass these
#: to ``assemble(extra_entries=...)`` without rediscovering the paper.
CITATION_KEYS = ("altun2020extrapolation",)

#: Fitted exponent of Eq. (1), Altun et al. (2020) Figure 1.
PNO_EXTRAPOLATION_BETA = 5.55

#: Flat `F` recommended by the paper for both 5/6 and 6/7 extrapolations.
RECOMMENDED_F = 1.5


def threshold_exponent(tcut_pno):
    """``X = -log10(TCutPNO)``, the abscissa of Eq. (1).

    A threshold of ``1e-6`` is ``X = 6``. Raises on non-positive input, which
    has no exponent and would otherwise propagate a silent ``inf``.
    """
    from math import log10

    value = float(tcut_pno)
    if value <= 0.0:
        raise ValueError(
            f"tcut_pno must be positive to define X = -log10(TCutPNO); "
            f"got {value!r}"
        )
    return -log10(value)


def extrapolation_factor(x, y, beta=PNO_EXTRAPOLATION_BETA):
    """Analytic `F` of Eq. (3), ``Y**beta / (Y**beta - X**beta)``.

    This is the exact inversion of Eq. (1) for the fitted ``beta``, not the
    paper's recommended flat 1.5. It is provided for callers who want the
    unfitted form; the two differ (about 1.57 for X=5,Y=6).
    """
    x, y, beta = float(x), float(y), float(beta)
    if x <= 0.0 or y <= 0.0:
        raise ValueError(f"X and Y must be positive; got X={x!r}, Y={y!r}")
    if y <= x:
        raise ValueError(
            f"the tighter threshold must have the larger exponent: "
            f"got X={x!r}, Y={y!r}"
        )
    xb, yb = x**beta, y**beta
    return yb / (yb - xb)


def extrapolate_pno_limit(e_x, e_y, *, f=RECOMMENDED_F):
    """Complete-PNO-space energy from two thresholds, Eq. (4).

    ``e_x`` is the correlation energy at the looser ``TCutPNO = 10**-X`` and
    ``e_y`` at the tighter ``10**-Y`` with ``Y > X``. Both must come from
    otherwise identical calculations. Returns ``E = e_x + f * (e_y - e_x)``.

    With the default ``f = 1.5`` this is the paper's recommended scheme. Pass
    ``f=extrapolation_factor(X, Y)`` for the analytic Eq. (3) inversion
    instead.
    """
    return float(e_x) + float(f) * (float(e_y) - float(e_x))
