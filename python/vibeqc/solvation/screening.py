"""Dielectric screening: one derivation, carried rather than recomputed.

CPCM and COSMO are not two formulas. Klamt & Schuurmann, *J. Chem. Soc.
Perkin Trans. 2* 799 (1993), doi:10.1039/P29930000799, p. 800 states that
dielectric screening energies for a fixed geometry scale as

    f(e) = (e - 1) / (e + x)

"where x is in the range 0-2". The two variants vibe-qc ships are two points
of that one-parameter family:

* ``x = 0``   -> ``f = (e - 1)/e``, the Cossi-Rega-Scalmani-Barone 2003
  C-PCM factor (doi:10.1002/jcc.10189);
* ``x = 1/2`` -> ``f = (e - 1)/(e + 1/2)``, the Klamt-Schuurmann COSMO factor
  (1993 p. 801), which extends the exact conductor result to finite ``e``
  with a relative error below ``e^-1 / 2``.

Writing them as one formula plus a per-variant ``x`` removes the branch that
every consumer used to re-implement, and makes a new variant a table entry
rather than a code path.

Why this module exists at all
-----------------------------
``f`` is a *property of the solve that produced the charges*, not a free
parameter of whoever consumes them. vibe-qc stores the scaled apparent charge
``q = f q0`` with ``q0 = -A^-1 V``, so the ``f`` appearing in the stationary
cavity-gradient term ``(1/(2f)) q^T (dA) q`` is fixed by the solve. Klamt 1993
p. 801 is explicit that ``f(e)`` enters the screening energy *and its
gradient* together.

Recomputing ``f`` at each consumer is therefore not a convenience -- it is an
opportunity to disagree, and it was taken: the gradient re-derived it from a
hard-coded ``variant="cpcm"`` while the driver honoured the requested variant,
making COSMO gradients differentiate an energy the run never reported (#546,
18.1% of the cavity term at benzene). :class:`ScreeningModel` is the fix:
consumers read ``f`` off an object built once, and a consumer that cannot
recompute cannot recompute wrongly (#548).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The one-parameter family above. A new screening variant is an entry here,
# not a new branch: keep this table and ``cpcm_screening_x`` in
# ``cpp/include/vibeqc/solvation_cpcm.hpp`` in step. They are pinned equal by
# ``tests/test_solvation_screening.py``.
SCREENING_X: dict[str, float] = {
    "cpcm": 0.0,
    "cosmo": 0.5,
}


def screening_factor(epsilon: float, x: float) -> float:
    """``f(e) = (e - 1)/(e + x)``, with the exact conductor limit.

    ``epsilon = inf`` returns exactly ``1.0`` for any finite ``x`` rather than
    the ``inf/inf`` NaN the closed form would give. The conductor is the limit
    COSMO is derived *from* (Klamt 1993 eq. 2), so an exactly-1 screening
    factor is the physically meaningful value and the reference case the
    generic layer validates against.
    """
    if epsilon <= 1.0:
        raise ValueError(
            f"screening_factor: epsilon must be > 1; got {epsilon} "
            f"(use solvent=None to skip the reaction field)."
        )
    if math.isinf(epsilon):
        return 1.0
    return (epsilon - 1.0) / (epsilon + x)


def screening_x(variant: str) -> float:
    """The ``x`` of :data:`SCREENING_X` for a variant name."""
    try:
        return SCREENING_X[variant.strip().lower()]
    except KeyError:
        # Message text is pinned by tests/test_solvation_cpcm.py and mirrors
        # the C++ cpcm_screening_x wording; keep the two in step.
        raise ValueError(
            f"dielectric_factor: unknown variant {variant!r} "
            f"(use {' or '.join(repr(k) for k in sorted(SCREENING_X))})."
        ) from None


@dataclass(frozen=True)
class ScreeningModel:
    """The screening factor that built a given set of apparent charges.

    Immutable and self-describing: it carries the variant it came from, the
    dielectric it was evaluated at, the ``x`` that distinguishes the variant,
    and the resulting ``f``. Consumers read :attr:`f`.

    Attributes
    ----------
    variant : str
        ``"cpcm"`` or ``"cosmo"`` -- the key in :data:`SCREENING_X`.
    epsilon : float
        Relative permittivity the factor was evaluated at. May be ``inf``.
    x : float
        The family parameter; see the module docstring.
    f : float
        ``(epsilon - 1)/(epsilon + x)``, or exactly 1 in the conductor limit.
    """

    variant: str
    epsilon: float
    x: float
    f: float

    @classmethod
    def from_variant(cls, epsilon: float, variant: str = "cpcm") -> "ScreeningModel":
        """Build the model for a named variant at a given dielectric."""
        v = variant.strip().lower()
        x = screening_x(v)
        return cls(
            variant=v,
            epsilon=float(epsilon),
            x=x,
            f=screening_factor(float(epsilon), x),
        )

    @property
    def is_conductor(self) -> bool:
        """True in the exact conductor limit (``epsilon = inf``, ``f == 1``)."""
        return math.isinf(self.epsilon)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        eps = "inf" if self.is_conductor else f"{self.epsilon:.4g}"
        return (
            f"ScreeningModel(variant={self.variant!r}, epsilon={eps}, "
            f"x={self.x:g}, f={self.f:.10f})"
        )


def dielectric_factor(epsilon: float, *, variant: str = "cpcm") -> float:
    """Conductor screening factor ``f(e)`` for a named variant.

    Thin accessor over :class:`ScreeningModel`; kept because it is the
    long-standing public spelling. Prefer carrying a ``ScreeningModel`` when
    the value will be consumed more than once -- see the module docstring for
    why re-deriving it at each consumer is the bug this module prevents.

    ``variant="cpcm"`` (default) is ``f = (e - 1)/e``;
    ``variant="cosmo"`` is ``f = (e - 1)/(e + 1/2)``. The two differ by
    ``O(1/e)`` and agree to better than 1% above ``e ~ 30``, so for water and
    most polar solvents the choice moves ``E_solv`` by <~ 0.1 kcal/mol -- but
    they diverge as ``e`` falls, reaching 18% at benzene's ``e = 2.27``.
    """
    return ScreeningModel.from_variant(epsilon, variant).f


__all__ = [
    "SCREENING_X",
    "ScreeningModel",
    "dielectric_factor",
    "screening_factor",
    "screening_x",
]
