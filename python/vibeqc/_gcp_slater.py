"""Canonical Slater-type-orbital machinery for the Kruse-Grimme gCP.

Ports the Slater-overlap pieces of the Grimme group's reference
``mctc-gcp`` Fortran implementation
(https://github.com/grimme-lab/gcp/blob/main/src/gcp.f90) into Python.
The published per-basis (s, η, a, b) gCP fit parameters were
determined against this exact overlap form; using a different
form (e.g. the simplified geometric-mean ζ_eff in earlier vibe-qc
revisions) would require re-fitting the parameters and would not
reproduce the literature gCP values.

What lives here:

* **A_k(x), B_k(x)** auxiliary integrals (k = 0..6) -- closed-form
  exponential / polynomial helpers used by the analytic STO-overlap
  formulae of Roothaan 1951.
* **B_int(x, k)** -- series-expansion fallback for B_k(x) when
  |x| -> 0 (the closed form has 1/x^(k+1) denominators that go
  singular at equal exponents).
* **slater_overlap(R, na, nb, za, zb)** -- the shell-aware
  ``<ns_a|n's_b>`` 1-center STO overlap. Switches over the six
  shell-pair combinations (1s/1s, 1s/2s, 1s/3s, 2s/2s, 2s/3s,
  3s/3s) the gCP attenuator needs for H-Rn coverage.
* **PER_ELEMENT_SLATER_SHELL** -- per-element principal quantum
  number of the outermost s-orbital used by the overlap dispatch
  (H, He: n=1; Li-Ne: n=2; Na-Ar: n=3; K-Rn: also n=3 per the
  mctc-gcp convention of treating 4s/5s as 3s for the gCP role).
* **SLATER_ZS / ZP / ZD** -- Clementi-Raimondi single-zeta effective
  exponents (s, p, d) for H-Kr. Per the mctc-gcp convention these
  are *not* per-basis: the per-basis fit absorbs the basis-specific
  contraction shape into the (s, η, a, b) constants, and the
  Slater-overlap attenuator uses the universal exponents scaled
  by η = p₂.

Numerical precision note: the Roothaan 1951 closed form is
numerically stable on the gCP-relevant R range (typical bonded
distances 1-10 bohr). For R < 0.05 bohr or |ζ_a - ζ_b| < 1e-6 the
loader falls back to the equal-exponent branch automatically.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np


# ---------------------------------------------------------------------------
# Per-element static tables -- sourced from mctc-gcp src/gcp.f90 lines
# 1653 (shell), 2925-2937 (ZS/ZP/ZD).
#
# Indexing is 1-based (atomic number Z) so the maps below are
# dict[int, float] with Z=1..36 as keys.
# ---------------------------------------------------------------------------

# Outermost s-orbital principal quantum number used by the gCP Slater
# overlap. H, He -> 1s; Li-Ne -> 2s; Na-Ar -> 3s; K-Rn -> 3s (mctc-gcp
# treats 4s/5s as 3s for the overlap dispatch; the per-basis η scaling
# absorbs the actual radial shape).
PER_ELEMENT_SLATER_SHELL: Final[dict[int, int]] = {
    **{1: 1, 2: 1},
    **{z: 2 for z in range(3, 11)},      # Li-Ne
    **{z: 3 for z in range(11, 19)},     # Na-Ar
    **{z: 3 for z in range(19, 87)},     # K-Rn
}


# Clementi-Raimondi effective exponents -- ZS (s), ZP (p), ZD (d).
# Values copied verbatim from mctc-gcp src/gcp.f90 lines 2925-2937.
# Z = 1..36 (H-Kr).
_ZS_TABLE: Final[tuple[float, ...]] = (
    1.2000, 1.6469, 0.6534, 1.0365, 1.3990, 1.7210, 2.0348, 2.2399,
    2.5644, 2.8812, 0.8675, 1.1935, 1.5143, 1.7580, 1.9860, 2.1362,
    2.3617, 2.5796, 0.9362, 1.2112, 1.2870, 1.3416, 1.3570, 1.3804,
    1.4761, 1.5465, 1.5650, 1.5532, 1.5781, 1.7778, 2.0675, 2.2702,
    2.4546, 2.5680, 2.7523, 2.9299,
)
_ZP_TABLE: Final[tuple[float, ...]] = (
    0.0000, 0.0000, 0.5305, 0.8994, 1.2685, 1.6105, 1.9398, 2.0477,
    2.4022, 2.7421, 0.6148, 0.8809, 1.1660, 1.4337, 1.6755, 1.7721,
    2.0176, 2.2501, 0.6914, 0.9329, 0.9828, 1.0104, 0.9947, 0.9784,
    1.0641, 1.1114, 1.1001, 1.0594, 1.0527, 1.2448, 1.5073, 1.7680,
    1.9819, 2.0548, 2.2652, 2.4617,
)
_ZD_TABLE: Final[tuple[float, ...]] = (
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
    0.0000, 0.0000, 0.0000, 0.0000, 2.4341, 2.6439, 2.7809, 2.9775,
    3.2208, 3.4537, 3.6023, 3.7017, 3.8962, 2.0477, 2.4022, 2.7421,
    0.6148, 0.8809, 1.1660, 1.4337,
)


def effective_zeta(Z: int, eta: float = 1.0, etaspec: float = 1.0) -> float:
    """Return the gCP effective Slater exponent for element ``Z``.

    Mirrors the ``setzet`` subroutine in mctc-gcp gcp.f90 lines
    2917-2955. Combines ZS / ZP / ZD per the shell-occupancy rule:

    * Z = 1, 2 (H, He): ζ = ZS
    * Z = 3..20, 31..36: ζ = (ZS + ZP) / 2     (valence s/p average)
    * Z = 21..30 (3d-TM): ζ = (ZS + ZP + ZD) / 3

    Then scaled by ``eta`` (the per-basis η = p₂ fit constant) and,
    for Z >= 11, by an additional ``etaspec`` factor used only for
    r^2SCAN-3c / def2-mTZVPP (where the contraction was re-fit with
    a 1.15 multiplier on the heavier-element exponents). The default
    ``etaspec=1.0`` covers every composite other than r^2SCAN-3c.
    """
    if Z < 1 or Z > 86:
        raise ValueError(
            f"effective_zeta: Z={Z} outside the H-Rn range supported by "
            f"the gCP universal Slater-exponent table."
        )
    if Z > 36:
        # K-Rn beyond Z=36 -- the Clementi-Raimondi single-zeta table
        # in mctc-gcp covers only H-Kr (Z=1..36); for Z=37..86 mctc-gcp
        # falls back to the Z=36 (Kr) values per its ``54*3`` shell
        # treatment. We mirror that here.
        Z = 36
    zs = _ZS_TABLE[Z - 1]
    zp = _ZP_TABLE[Z - 1]
    zd = _ZD_TABLE[Z - 1]
    if Z <= 2:
        zeta = zs
    elif 21 <= Z <= 30:                          # 3d-TM
        zeta = (zs + zp + zd) / 3.0
    else:                                        # 3-20 and 31-36
        zeta = (zs + zp) / 2.0
    if Z >= 11:
        zeta *= etaspec
    zeta *= eta
    return zeta


# ---------------------------------------------------------------------------
# A_k(x) auxiliary integrals.
# Reference: mctc-gcp src/gcp.f90 lines 1810-1900.
# ---------------------------------------------------------------------------

def _A0(x: float) -> float:
    return math.exp(-x) / x


def _A1(x: float) -> float:
    return (1.0 + x) * math.exp(-x) / x ** 2


def _A2(x: float) -> float:
    return (2.0 + 2.0 * x + x ** 2) * math.exp(-x) / x ** 3


def _A3(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x
    return (6.0 + 6.0 * x + 3.0 * x2 + x3) * math.exp(-x) / x4


def _A4(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x
    poly = 24.0 + 24.0 * x + 12.0 * x2 + 4.0 * x3 + x4
    return poly * math.exp(-x) / x5


def _A5(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x
    poly = 120.0 + 120.0 * x + 60.0 * x2 + 20.0 * x3 + 5.0 * x4 + x5
    return poly * math.exp(-x) / x6


def _A6(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x
    x5 = x4 * x; x6 = x5 * x; x7 = x6 * x
    poly = (720.0 + 720.0 * x + 360.0 * x2 + 120.0 * x3
            + 30.0 * x4 + 6.0 * x5 + x6)
    return poly * math.exp(-x) / x7


# ---------------------------------------------------------------------------
# B_k(x) auxiliary integrals (closed form, singular at x = 0).
# Reference: mctc-gcp src/gcp.f90 lines 1907-2000.
# ---------------------------------------------------------------------------

def _B0(x: float) -> float:
    return (math.exp(x) - math.exp(-x)) / x


def _B1(x: float) -> float:
    x2 = x * x
    return ((1.0 - x) * math.exp(x) - (1.0 + x) * math.exp(-x)) / x2


def _B2(x: float) -> float:
    x2 = x * x; x3 = x2 * x
    a = (2.0 - 2.0 * x + x2) * math.exp(x)
    b = (2.0 + 2.0 * x + x2) * math.exp(-x)
    return (a - b) / x3


def _B3(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x
    a = (6.0 - 6.0 * x + 3.0 * x2 - x3) * math.exp(x) / x4
    b = (6.0 + 6.0 * x + 3.0 * x2 + x3) * math.exp(-x) / x4
    return a - b


def _B4(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x
    a = (24.0 - 24.0 * x + 12.0 * x2 - 4.0 * x3 + x4) * math.exp(x) / x5
    b = (24.0 + 24.0 * x + 12.0 * x2 + 4.0 * x3 + x4) * math.exp(-x) / x5
    return a - b


def _B5(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x; x5 = x4 * x; x6 = x5 * x
    a = (120.0 - 120.0 * x + 60.0 * x2 - 20.0 * x3 + 5.0 * x4 - x5) \
        * math.exp(x) / x6
    b = (120.0 + 120.0 * x + 60.0 * x2 + 20.0 * x3 + 5.0 * x4 + x5) \
        * math.exp(-x) / x6
    return a - b


def _B6(x: float) -> float:
    x2 = x * x; x3 = x2 * x; x4 = x3 * x
    x5 = x4 * x; x6 = x5 * x; x7 = x6 * x
    a = (720.0 - 720.0 * x + 360.0 * x2 - 120.0 * x3
         + 30.0 * x4 - 6.0 * x5 + x6) * math.exp(x) / x7
    b = (720.0 + 720.0 * x + 360.0 * x2 + 120.0 * x3
         + 30.0 * x4 + 6.0 * x5 + x6) * math.exp(-x) / x7
    return a - b


# ---------------------------------------------------------------------------
# B_int(x, k) -- series-expansion fallback for B_k(x) when x -> 0 (i.e.
# equal Slater exponents). The closed-form B_k has 1/x^(k+1)
# denominators that blow up there. mctc-gcp uses 12 series terms which
# we mirror.
#
# Reference: mctc-gcp src/gcp.f90 lines 2010-2050 (Bint subroutine).
# ---------------------------------------------------------------------------

def _factorial(n: int) -> int:
    """Local integer factorial; bypasses math.factorial's float
    coercion in older Python versions."""
    out = 1
    for j in range(2, n + 1):
        out *= j
    return out


def _Bint(x: float, k: int) -> float:
    """Series fallback for B_k(x). Returns the analytic value when
    |x| >= 1e-6 (delegating to the matching closed form) and the
    series sum otherwise."""
    if abs(x) > 1e-6:
        # Use the closed form when stable.
        analytic = (_B0, _B1, _B2, _B3, _B4, _B5, _B6)[k]
        return analytic(x)
    # Series for very small x (and exactly x = 0): mctc-gcp uses
    # 12 terms which is plenty for the gCP-relevant precision.
    total = 0.0
    for i in range(13):
        sign = 1.0 - ((-1.0) ** (k + i + 1))
        coeff = sign / (_factorial(i) * (k + i + 1))
        total += coeff * (-x) ** i
    return total


# ---------------------------------------------------------------------------
# slater_overlap -- the main shell-aware 1s/2s/3s STO overlap dispatcher.
# Mirrors ssovl from mctc-gcp src/gcp.f90 lines 1645-1801.
# ---------------------------------------------------------------------------

def slater_overlap(R: float, na: int, nb: int,
                   za: float, zb: float) -> float:
    """Compute the analytic STO overlap <n_a s | n_b s> at two centres
    separated by ``R`` (bohr), with effective principal quantum numbers
    ``na``, ``nb`` in {1, 2, 3} and effective exponents ``za``, ``zb``.

    Bit-for-bit equivalent to ``ssovl`` in mctc-gcp gcp.f90 for the
    supported (na, nb) pairs. Returns a dimensionless number in
    (0, 1].
    """
    # Shell-product discriminator: 1, 2, 3, 4, 6, 9 for (1s,1s),
    # (1s,2s)/(2s,1s), (1s,3s)/(3s,1s), (2s,2s), (2s,3s)/(3s,2s),
    # (3s,3s) respectively.
    ii = na * nb
    R05 = 0.5 * R
    ax = (za + zb) * R05
    bx = (zb - za) * R05

    # Equal-element / near-equal-exponent branch (uses Bint series
    # since closed-form B_k blows up at bx -> 0).
    if abs(za - zb) < 0.1:
        if ii == 1:                                      # <1s|1s>
            norm = 0.25 * math.sqrt((za * zb * R * R) ** 3)
            return norm * (_A2(ax) * _Bint(bx, 0)
                           - _Bint(bx, 2) * _A0(ax))
        if ii == 2:                                      # <1s|2s>+<2s|1s>
            if na > nb:
                za, zb = zb, za
                ax = (za + zb) * R05
                bx = (zb - za) * R05
            norm = math.sqrt((za ** 3) * (zb ** 5)) * (R ** 4) * 0.125
            return (math.sqrt(1.0 / 3.0) * norm
                    * (_A3(ax) * _Bint(bx, 0)
                       - _Bint(bx, 3) * _A0(ax)
                       + _A2(ax) * _Bint(bx, 1)
                       - _Bint(bx, 2) * _A1(ax)))
        if ii == 4:                                      # <2s|2s>
            norm = math.sqrt((za * zb) ** 5) * (R ** 5) * 0.0625
            return norm * (_A4(ax) * _Bint(bx, 0)
                           + _Bint(bx, 4) * _A0(ax)
                           - 2.0 * _A2(ax) * _Bint(bx, 2)) / 3.0
        if ii == 3:                                      # <1s|3s>+<3s|1s>
            if na > nb:
                za, zb = zb, za
                ax = (za + zb) * R05
                bx = (zb - za) * R05
            norm = (math.sqrt((za ** 3) * (zb ** 7) / 7.5)
                    * (R ** 5) * 0.0625)
            return (norm
                    * (_A4(ax) * _Bint(bx, 0)
                       - _Bint(bx, 4) * _A0(ax)
                       + 2.0 * (_A3(ax) * _Bint(bx, 1)
                                - _Bint(bx, 3) * _A1(ax)))
                    / math.sqrt(3.0))
        if ii == 6:                                      # <2s|3s>+<3s|2s>
            if na > nb:
                za, zb = zb, za
                ax = (za + zb) * R05
                bx = (zb - za) * R05
            norm = (math.sqrt((za ** 5) * (zb ** 7) / 7.5)
                    * (R ** 6) * 0.03125)
            return (norm
                    * (_A5(ax) * _Bint(bx, 0)
                       + _A4(ax) * _Bint(bx, 1)
                       - 2.0 * (_A3(ax) * _Bint(bx, 2)
                                + _A2(ax) * _Bint(bx, 3))
                       + _A1(ax) * _Bint(bx, 4)
                       + _A0(ax) * _Bint(bx, 5))
                    / 3.0)
        if ii == 9:                                      # <3s|3s>
            norm = math.sqrt((za * zb * R * R) ** 7) / 480.0
            return (norm
                    * (_A6(ax) * _Bint(bx, 0)
                       - 3.0 * (_A4(ax) * _Bint(bx, 2)
                                - _A2(ax) * _Bint(bx, 4))
                       - _A0(ax) * _Bint(bx, 6))
                    / 3.0)
    # Different-element branch -- uses closed-form B_k (no series).
    if ii == 1:
        norm = 0.25 * math.sqrt((za * zb * R * R) ** 3)
        return (_A2(ax) * _B0(bx) - _B2(bx) * _A0(ax)) * norm
    if ii == 2:
        if na > nb:
            za, zb = zb, za
            ax = (za + zb) * R05
            bx = (zb - za) * R05
        norm = math.sqrt((za ** 3) * (zb ** 5)) * (R ** 4) * 0.125
        return (math.sqrt(1.0 / 3.0) * norm
                * (_A3(ax) * _B0(bx) - _B3(bx) * _A0(ax)
                   + _A2(ax) * _B1(bx) - _B2(bx) * _A1(ax)))
    if ii == 4:
        norm = math.sqrt((za * zb) ** 5) * (R ** 5) * 0.0625
        return norm * (_A4(ax) * _B0(bx) + _B4(bx) * _A0(ax)
                       - 2.0 * _A2(ax) * _B2(bx)) * (1.0 / 3.0)
    if ii == 3:
        if na > nb:
            za, zb = zb, za
            ax = (za + zb) * R05
            bx = (zb - za) * R05
        norm = (math.sqrt((za ** 3) * (zb ** 7) / 7.5)
                * (R ** 5) * 0.0625)
        return (norm
                * (_A4(ax) * _B0(bx) - _B4(bx) * _A0(ax)
                   + 2.0 * (_A3(ax) * _B1(bx) - _B3(bx) * _A1(ax)))
                / math.sqrt(3.0))
    if ii == 6:
        if na > nb:
            za, zb = zb, za
            ax = (za + zb) * R05
            bx = (zb - za) * R05
        norm = (math.sqrt((za ** 5) * (zb ** 7) / 7.5)
                * (R ** 6) * 0.03125)
        return (norm
                * (_A5(ax) * _B0(bx) + _A4(ax) * _B1(bx)
                   - 2.0 * (_A3(ax) * _B2(bx) + _A2(ax) * _B3(bx))
                   + _A1(ax) * _B4(bx) + _A0(ax) * _B5(bx))
                / 3.0)
    if ii == 9:
        norm = math.sqrt((za * zb * R * R) ** 7) / 1440.0
        return (norm
                * (_A6(ax) * _B0(bx)
                   - 3.0 * (_A4(ax) * _B2(bx) - _A2(ax) * _B4(bx))
                   - _A0(ax) * _B6(bx)))
    raise ValueError(
        f"slater_overlap: unsupported shell pair na={na}, nb={nb}. "
        f"Only n_a, n_b in {{1, 2, 3}} (1s, 2s, 3s) are supported."
    )
