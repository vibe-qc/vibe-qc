"""MSINDO STO integral kernel -- faithful port of the general Slater machinery.

Re-implements MSINDO's analytic Slater-type-orbital integrals so the engine
generalizes to s/p/d without libint (which is Gaussian-only).  Each function is
a faithful translation of the named MSINDO routine; validated against a
reference MSINDO build (examples/regression/msindo/).

Dependency chain (everything bottoms out in OVERLP):
  overlap S2INT, nuclear V2INT, Coulomb C2INT/HARRIS, Slater-Condon RADINT
    -> OVERLP (A_k/B_k auxiliary integrals + CALSUM with BSSIG angular coeffs)
  rotation local->global: HARMTR.

References: Mulliken-Rieke-Orloff-Orloff; Harris, J. Chem. Phys. 51, 4777 (1969);
Kuppermann-Karplus-Isaacson, Z. Naturforsch. 14A, 311 (1959).
"""

from __future__ import annotations

import math

import numpy as np

# --------------------------------------------------------------------------- #
# Factorials / binomials (const.f).
# --------------------------------------------------------------------------- #
_FACS = [1.0]
for _i in range(1, 40):
    _FACS.append(_FACS[-1] * _i)


def facs(n: int) -> float:
    return _FACS[n]


_BIN = np.zeros((40, 40))
for _n in range(40):
    _BIN[_n, 0] = 1.0
    for _k in range(1, _n + 1):
        _BIN[_n, _k] = _BIN[_n - 1, _k - 1] + _BIN[_n - 1, _k]


def binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return _BIN[n, k]


# --------------------------------------------------------------------------- #
# QUANT0/1/2 index maps and XSAVE prefactors (datas.f / const.f).
# QUANT0[l][n] (l=0..4, n=0..9), column-major DATA reshaped.
# --------------------------------------------------------------------------- #
_Q0 = [  # rows = n (0..9), cols = l (0..4)
    [1, 0, 0, 0, 0],
    [2, 0, 0, 0, 0],
    [3, 4, 5, 0, 0],
    [6, 7, 8, 0, 0],
    [9, 10, 11, 0, 12],
    [13, 14, 15, 0, 16],
    [17, 0, 18, 0, 19],
    [20, 0, 21, 0, 22],
    [23, 0, 24, 0, 25],
    [26, 0, 27, 0, 28],
]


def quant0(l: int, n: int) -> int:
    return _Q0[n][l]


# QUANT1(l=1..2, n=2..5): DATA 1,0,2,3,4,5,6,7 (column-major).
_Q1 = {
    (1, 2): 1,
    (2, 2): 0,
    (1, 3): 2,
    (2, 3): 3,
    (1, 4): 4,
    (2, 4): 5,
    (1, 5): 6,
    (2, 5): 7,
}


def quant1(l: int, n: int) -> int:
    return _Q1.get((l, n), 0)


# QUANT2(l=2, n=3..5): DATA 1,2,3.
_Q2 = {(2, 3): 1, (2, 4): 2, (2, 5): 3}


def quant2(l: int, n: int) -> int:
    return _Q2.get((l, n), 0)


# XSAVE(j,i) = 2^-j sqrt((2j+1) BIN(j+i,j)/(2 BIN(j,i))) (const.f).
_XSAVE = np.zeros((9, 3))
for _i in range(3):
    for _j in range(_i, 9):
        _XSAVE[_j, _i] = 2.0 ** (-_j) * math.sqrt(
            (2 * _j + 1) * binom(_j + _i, _j) / (2.0 * binom(_j, _i))
        )


def xsave(l: int, m: int) -> float:
    return _XSAVE[l, m]


# --------------------------------------------------------------------------- #
# Wigner 3j (L1=L2=L) and Coulomb angular A0 = F_A (wig.f / f_a.f / const.f).
# --------------------------------------------------------------------------- #


def wig(L: int, J: int, m1: int, m2: int) -> float:
    M = m1 + m2
    if J < abs(M):
        return 0.0
    ku = max(0, L - J - m1, L - J + m2)
    ko = min(2 * L - J, L - m1, L + m2)
    s = 0.0
    for k in range(ku, ko + 1):
        s += (-1) ** k / (
            facs(k)
            * facs(2 * L - J - k)
            * facs(L - m1 - k)
            * facs(L + m2 - k)
            * facs(J - L + m1 + k)
            * facs(J - L - m2 + k)
        )
    return (
        (-1) ** M
        * math.sqrt(
            facs(2 * L - J)
            * facs(J)
            * facs(J)
            * facs(L + m1)
            * facs(L - m1)
            * facs(L + m2)
            * facs(L - m2)
            * facs(J + M)
            * facs(J - M)
            / facs(2 * L + J + 1)
        )
        * s
    )


def f_a(L: int, J: int, m1: int, m2: int) -> float:
    v = (
        math.sqrt((2 * L + 1) * (2 * L + 1) * (2 * J + 1) / 4.0 / math.pi)
        * wig(L, J, m1, m2)
        * wig(L, J, 0, 0)
    )
    if m1 > 0:
        v *= (-1) ** m1
    if m2 > 0:
        v *= (-1) ** m2
    return v


# A0[L][J][M] = F_A(L,J,M,-M), L,M=0..2, J=0..2L (const.f).
_A0 = {}
for _L in range(3):
    for _J in range(0, 2 * _L + 1):
        for _M in range(0, _L + 1):
            _A0[(_L, _J, _M)] = f_a(_L, _J, _M, -_M)


def a0(L: int, J: int, M: int) -> float:
    return _A0.get((L, J, M), 0.0)


# --------------------------------------------------------------------------- #
# BISSIG -> BSSIG angular overlap coefficients (bissig.f), cached per (q2,q1,m).
# --------------------------------------------------------------------------- #


def _bissig(n1: int, n2: int, l1: int, l2: int, m: int) -> list[float]:
    """Faithful port of bissig.f: list of BSSIG coefficients (CALSUM order)."""
    sigu = n1 + n2 - l1 - l2
    sigo = n1 + n2 + l1 + l2
    i1o = (l1 - m) // 2
    i2o = (l2 - m) // 2
    out = []
    for sig in range(sigu, sigo + 1, 2):
        su = max(0, sig - n1 - n2)
        so = su + (min(sig, n1 + n2) - su) // 2
        for s in range(su, so + 1):
            val = 0.0
            for tau in range(0, m + 1):
                for lam in range(0, m + 1):
                    for i1 in range(0, i1o + 1):
                        for i2 in range(0, i2o + 1):
                            ko = l1 - m - 2 * i1
                            for k in range(0, ko + 1):
                                yu = (
                                    (sig - n1 - n2 + l1 + l2) // 2
                                    - tau
                                    - lam
                                    - i1
                                    - i2
                                    - k
                                )
                                yo = l2 - m - 2 * i2
                                if yu < 0 or yu > yo:
                                    continue
                                xo = n2 - l2 + 2 * i2
                                po = n1 - l1 + 2 * i1 + k
                                x = (
                                    -(sig - n1 - n2 + l1 + l2) // 2
                                    + s
                                    - tau
                                    + lam
                                    + i1
                                    + i2
                                    + k
                                )
                                for p in range(k, po + 1):
                                    xu = x - p
                                    if xu < 0 or xu > xo:
                                        continue
                                    val += (
                                        (-1) ** (tau + lam + i1 + i2 + p)
                                        * binom(m, tau)
                                        * binom(m, lam)
                                        * binom(l1, i1)
                                        * binom(l2, i2)
                                        * binom(2 * (l1 - i1), l1 + m)
                                        * binom(2 * (l2 - i2), l2 + m)
                                        * binom(n1 - l1 + 2 * i1, p - k)
                                        * binom(l1 - m - 2 * i1, k)
                                        * binom(xo, xu)
                                        * binom(yo, yu)
                                    )
            out.append(val * (-1) ** n1)
    return out


def _calsum(bssig, a_int, b_int, n1, n2, l1, l2) -> float:
    """Faithful port of calsum.f."""
    sigu = n1 - l1 + n2 - l2
    sigo = n1 + l1 + n2 + l2
    vz = (-1.0) ** (n1 + l1)
    total = 0.0
    z = 0
    for sig in range(sigu, sigo + 1, 2):
        su = max(0, sig - n1 - n2)
        so = su + (min(sig, n1 + n2) - su) // 2
        for s in range(su, so + 1):
            if sigu % 2 == 0 and sig == 2 * s:
                total += bssig[z] * a_int[s] * b_int[s]
            else:
                total += bssig[z] * (
                    a_int[s] * b_int[sig - s] + vz * a_int[sig - s] * b_int[s]
                )
            z += 1
    return total


# --------------------------------------------------------------------------- #
# A_k / B_k auxiliary integrals (aintgs.f / bintgs.f).
# --------------------------------------------------------------------------- #


def _aintgs(x: float, kmax: int):
    a = [0.0] * (kmax + 1)
    a[0] = 1.0 / x
    for k in range(1, kmax + 1):
        a[k] = (a[k - 1] * k + 1.0) * (1.0 / x)
    return a


def _bintgs(x: float, y: float, kmax: int):
    b = [0.0] * (kmax + 1)
    absy = abs(y)
    expmx = math.exp(-x)
    if absy < 1e-10:
        for k in range(kmax + 1):
            b[k] = (2.0 / (k + 1.0) if k % 2 == 0 else 0.0) * expmx
    elif absy < 1.5:
        y2 = y * y
        for k in range(0, kmax + 1, 2):  # even
            tot, yy = 0.0, 1.0
            for nu in range(0, 18, 2):
                term = (2.0 / (facs(nu) * (k + nu + 1))) * yy
                tot += term
                if abs(term) < 1e-10:
                    break
                yy *= y2
            b[k] = expmx * tot
        for k in range(1, kmax + 1, 2):  # odd
            tot, yy = 0.0, y
            for nu in range(1, 18, 2):
                term = (2.0 / (facs(nu) * (k + nu + 1))) * yy
                tot -= term
                if abs(term) < 1e-10:
                    break
                yy *= y2
            b[k] = expmx * tot
    else:
        fak = 1.0 / y
        expy = math.exp((y - x) / 2.0) ** 2
        expmy = math.exp((-y - x) / 2.0) ** 2
        b[0] = (expy - expmy) * fak
        for k in range(1, kmax + 1):
            sgn = 1.0 if k % 2 == 0 else -1.0
            b[k] = (k * b[k - 1] + sgn * expy - expmy) * fak
    return b


# --------------------------------------------------------------------------- #
# OVERLP -- generalized reduced overlap (overlp.f).  S2INT adds normalization.
# --------------------------------------------------------------------------- #

_BSSIG_CACHE: dict = {}


def _bssig(n1, l1, n2, l2, m):
    key = (n1, l1, n2, l2, m)
    if key not in _BSSIG_CACHE:
        _BSSIG_CACHE[key] = _bissig(n1, n2, l1, l2, m)
    return _BSSIG_CACHE[key]


def overlp(n2, l2, z2, n1, l1, z1, m, R, grad=False):
    """Generalized reduced overlap < (n2 l2 z2) | (n1 l1 z1) >, angular m.

    When ``grad=True``, returns d/dR of the overlap (overlp.f with GRAD=.TRUE.).
    """
    mb = abs(m)
    if l2 < 0 or l1 < 0 or l2 < mb or l1 < mb:
        return 0.0
    kmax = n1 + n2 + 1
    rm12 = 0.5 * R
    alpha = rm12 * (z1 + z2)
    beta = rm12 * (z2 - z1)
    intmax = kmax + 1 if grad else kmax
    a_int = _aintgs(alpha, intmax)
    b_int = _bintgs(alpha, beta, intmax)
    bss = _bssig(n1, l1, n2, l2, mb)
    s = _calsum(bss, a_int, b_int, n1, n2, l1, l2)
    if grad:
        # d/dR of (R/2)^kmax * SUM = kmax*(R/2)^(kmax-1)*(1/2)*SUM + (R/2)^kmax*DSUM
        # = (R/2)^kmax * [ kmax/R * SUM + DSUM ]
        dxdr = 0.5 * kmax * rm12 ** (kmax - 1)  # d/dR of (R/2)^kmax
        # DADR[i] = -A_int[i+1] * alpha/R,  DBDR[i] = -B_int[i+1] * beta/R
        vdra = alpha / R
        vdrb = beta / R
        dadr = [-a_int[i + 1] * vdra for i in range(kmax + 1)]
        dbdr = [-b_int[i + 1] * vdrb for i in range(kmax + 1)]
        dsum = _cdsum(bss, a_int, b_int, n1, n2, l1, l2, dadr, dbdr)
        xs = xsave(l1, mb) * xsave(l2, mb)
        return xs * (dxdr * s + rm12**kmax * dsum)
    return rm12**kmax * xsave(l1, mb) * xsave(l2, mb) * s


def s2int(n1, l1, m1, z1, n2, l2, m2, z2, R) -> float:
    """Two-center overlap <mu_a|nu_b> (s2int.f).  Requires m1==m2."""
    if m1 != m2:
        return 0.0
    fak = math.sqrt(
        (z1 + z1) ** (n1 + n1 + 1)
        * (z2 + z2) ** (n2 + n2 + 1)
        / (facs(n1 + n1) * facs(n2 + n2))
    )
    return fak * overlp(n1, l1, z1, n2, l2, z2, m1, R)


def ds2int(n1, l1, m1, z1, n2, l2, m2, z2, R) -> float:
    """Derivative d/dR of two-center overlap <mu_a|nu_b> (ds2int.f)."""
    if m1 != m2:
        return 0.0
    fak = math.sqrt(
        (z1 + z1) ** (n1 + n1 + 1)
        * (z2 + z2) ** (n2 + n2 + 1)
        / (facs(n1 + n1) * facs(n2 + n2))
    )
    return fak * overlp(n1, l1, z1, n2, l2, z2, m1, R, grad=True)


# --------------------------------------------------------------------------- #
# Nuclear attraction V2INT (v2int.f), Coulomb C2INT/HARRIS (c2int.f/harris.f),
# Slater-Condon RADINT/SK0 (radint.f/sk0.f).
# --------------------------------------------------------------------------- #


def v2int(n1, l1, m1, z1, R) -> float:
    """<mu_a|1/r_b|mu_a> nuclear attraction (KERN='R')."""
    m = abs(m1)
    nh = 2 * n1 - 1
    zeth = 2 * z1
    fak = zeth ** (nh + 2) / facs(2 * n1) * 2.0 * math.sqrt(math.pi)
    s = 0.0
    for i in range(0, 2 * l1 + 1, 2):
        s += a0(l1, i, m) * overlp(nh, i, zeth, 0, 0, 0.0, 0, R)
    return fak * s


def _harris(n1, l1, z1, n2, l2, z2, m, R) -> float:
    l1l2m = l1 + l2 - m
    vf1 = 4.0 * math.pi / (z1 * z1)
    vf2 = 4.0 * math.pi / (z2 * z2)
    form17 = vf1 * (
        overlp(0, 0, 0.0, l1l2m, l1l2m, z2, 0, R)
        - overlp(0, 0, z1, l1l2m, l1l2m, z2, 0, R)
    )
    summ = 0.0
    for j in range(0, m):
        summ += (
            (-1.0 / z1 / z1 / R) ** (m - j)
            / (2 * j + 2)
            * math.sqrt(
                facs(2 * m + 1)
                * facs(l1l2m + m)
                * facs(l1l2m - j)
                / facs(2 * j + 1)
                / facs(l1l2m - m)
                / facs(l1l2m + j)
            )
            * overlp(j + 1, j, z1, l1l2m, l1l2m, z2, j, R)
        )
    form30 = (-1.0 / z1 / z1 / R) ** m * math.sqrt(
        facs(2 * m + 1) * facs(l1l2m + m) / facs(l1l2m - m)
    ) * form17 - vf1 * z1 * summ
    form26 = form30
    for j in range(1, l1 - m + 1):
        L = m + j - 1
        ls = l1l2m - j + 1
        w1 = math.sqrt((L - m + 1) * (L + m + 1) / ((2 * L + 1) * (2 * L + 3)))
        w2 = math.sqrt((ls - m) * (ls + m) / ((2 * ls - 1) * (2 * ls + 1)))
        fak1 = z1 / z2 * 2 * ls * w1
        fak2 = z2 / z1 * (2 * L + 2) * w2
        summ = (
            vf1
            * math.sqrt((L + m) * (L - m) / ((2 * L - 1) * (2 * L + 1)))
            * (
                z1 * overlp(L + 1, L - 1, z1, ls, ls - 1, z2, m, R)
                + overlp(L, L - 1, z1, ls, ls - 1, z2, m, R)
            )
            + vf2
            * math.sqrt((ls + m - 1) * (ls - m - 1) / ((2 * ls - 3) * (2 * ls - 1)))
            * (
                z2 * overlp(L + 1, L, z1, ls, ls - 2, z2, m, R)
                + overlp(L + 1, L, z1, ls - 1, ls - 2, z2, m, R)
            )
            + vf2 * z1 * w1 * overlp(L + 1, L + 1, z1, ls, ls - 1, z2, m, R)
            + vf1 * z2 * w2 * overlp(L + 1, L, z1, ls, ls, z2, m, R)
        )
        form26 = (summ - fak2 * form26) / fak1
    vf1 = facs(n1 + l1 + 1) / facs(2 * l1 + 1)
    summ1 = 0.0
    summ2 = 0.0
    for j in range(n2, l2, -1):
        summ1 += (
            z2 ** (j - n2 - 2)
            * (facs(n2 + l2 + 1) / facs(j + l2) - facs(n2 - l2) / facs(j - l2 - 1))
            * overlp(l1, l1, z1, j, l2, z2, m, R)
        )
    for j in range(n1, l1, -1):
        summ2 += (
            z1 ** (j - n1 - 2)
            * (facs(n1 + l1 + 1) / facs(j + l1) - facs(n1 - l1) / facs(j - l1 - 1))
            * overlp(j, l1, z1, n2, l2, z2, m, R)
        )
    return (
        vf1
        * facs(n2 + l2 + 1)
        / facs(2 * l2 + 1)
        * z1 ** (l1 - n1)
        * z2 ** (l2 - n2)
        * form26
        - 4.0 * math.pi / (2 * l2 + 1) * vf1 * z1 ** (l1 - n1) * summ1
        - 4.0 * math.pi / (2 * l1 + 1) * summ2
    )


def c2int(n1, l1, m1, z1, n2, l2, m2, z2, R) -> float:
    """Two-center Coulomb integral over basis charge distributions (c2int.f)."""
    m = abs(m1)
    ms = abs(m2)
    nh1 = 2 * n1 - 1
    nh2 = 2 * n2 - 1
    zeth1 = 2 * z1
    zeth2 = 2 * z2
    fak = zeth1 ** (nh1 + 2) * zeth2 ** (nh2 + 2) / facs(2 * n1) / facs(2 * n2)
    s = 0.0
    for j in range(0, 2 * l1 + 1, 2):
        for js in range(0, 2 * l2 + 1, 2):
            s += (
                a0(l1, j, m)
                * a0(l2, js, ms)
                * _harris(nh1, j, zeth1, nh2, js, zeth2, 0, R)
            )
    return fak * s


def _cdsum(bssig, a_int, b_int, n1, n2, l1, l2, dadr, dbdr) -> float:
    """Derivative of _calsum w.r.t. R (cdsum.f).

    Product-rule derivative of every (A_i * B_j) term in _calsum."""
    sigu = n1 - l1 + n2 - l2
    sigo = n1 + l1 + n2 + l2
    vz = (-1.0) ** (n1 + l1)
    total = 0.0
    z = 0
    if sigu % 2 == 0:
        for sig in range(sigu, sigo + 1, 2):
            su = max(0, sig - n1 - n2)
            so = su + (min(sig, n1 + n2) - su) // 2
            for s in range(su, so + 1):
                if sig == 2 * s:
                    total += bssig[z] * (dadr[s] * b_int[s] + a_int[s] * dbdr[s])
                else:
                    total += bssig[z] * (
                        dadr[s] * b_int[sig - s]
                        + a_int[s] * dbdr[sig - s]
                        + vz * (dadr[sig - s] * b_int[s] + a_int[sig - s] * dbdr[s])
                    )
                z += 1
    else:
        for sig in range(sigu, sigo + 1, 2):
            su = max(0, sig - n1 - n2)
            so = su + (min(sig, n1 + n2) - su) // 2
            for s in range(su, so + 1):
                total += bssig[z] * (
                    dadr[s] * b_int[sig - s]
                    + a_int[s] * dbdr[sig - s]
                    + vz * (dadr[sig - s] * b_int[s] + a_int[sig - s] * dbdr[s])
                )
                z += 1
    return total


def _sk0(k: int, p: float) -> float:
    return facs(k - 1) / (p + 1.0) ** k


def radint(lamb, na, nb, nc, nd, za, zb, zc, zd) -> float:
    """One-center two-electron radial Slater-Condon integral R_lamb (radint.f)."""
    n1 = na + nc
    n2 = nb + nd
    z1 = za + zc
    z2 = zb + zd
    if z1 == 0.0 or z2 == 0.0:
        return 0.0
    k = n1 + n2
    l1 = n2 - lamb - 1
    l2 = n1 - lamb - 1
    p1 = z1 / z2
    p2 = z2 / z1
    expo = -n1 - n2 - 1

    def skl(L, kk, p):
        if L == 0:
            return _sk0(kk, p)
        if L == 1:
            return _sk0(kk - 1, p) + _sk0(kk, p)
        val = _sk0(kk - L, p) + _sk0(kk - L + 1, p)
        fak = 2.0
        for i in range(2, L + 1):
            val = fak * val + _sk0(kk - L + i, p)
            fak += 1.0
        return val

    skl1 = skl(l1, k, p1)
    skl2 = skl(l2, k, p2)
    rad = z2**expo * skl1 + z1**expo * skl2
    norfak = (
        (2.0 * za) ** (2 * na + 1)
        * (2.0 * zb) ** (2 * nb + 1)
        * (2.0 * zc) ** (2 * nc + 1)
        * (2.0 * zd) ** (2 * nd + 1)
        / (facs(2 * na) * facs(2 * nb) * facs(2 * nc) * facs(2 * nd))
    )
    return rad * math.sqrt(norfak)


# --------------------------------------------------------------------------- #
# HARMTR -- local->global rotation matrix from direction cosine E (harmtr.f).
# Orbital order: 1=s, 2..4 = px,py,pz, 5..9 = d.
# --------------------------------------------------------------------------- #


def harmtr(maxkl: int, E) -> np.ndarray:
    T = np.zeros((9, 9))
    T[0, 0] = 1.0
    if maxkl <= 1:
        return T
    cost = E[2]
    if abs(cost) == 1.0:
        sint, cosp, sinp = 0.0, 1.0, 0.0
    elif abs(cost) == 0.0:
        sint, cosp, sinp = 1.0, E[0], E[1]
    else:
        sint = math.sqrt(1.0 - cost**2)
        cosp, sinp = E[0] / sint, E[1] / sint
    # p block (1-based 2..4 -> 0-based 1..3)
    T[1, 1] = sint * cosp
    T[2, 1] = sint * sinp
    T[3, 1] = cost
    T[1, 2] = cost * cosp
    T[2, 2] = cost * sinp
    T[3, 2] = -sint
    T[1, 3] = -sinp
    T[2, 3] = cosp
    T[3, 3] = 0.0
    if maxkl <= 2:
        return T
    cos2t = cost**2 - sint**2
    sin2t = 2.0 * sint * cost
    cos2p = cosp**2 - sinp**2
    sin2p = 2.0 * sinp * cosp
    s3 = math.sqrt(3.0)
    d = [
        [
            (4, 4, (3.0 * cost**2 - 1.0) * 0.5),
            (5, 4, s3 * sin2t * cosp * 0.5),
            (6, 4, s3 * sin2t * sinp * 0.5),
            (7, 4, s3 * sint**2 * cos2p * 0.5),
            (8, 4, s3 * sint**2 * sin2p * 0.5),
        ],
        [
            (4, 5, -s3 * sin2t * 0.5),
            (5, 5, cos2t * cosp),
            (6, 5, cos2t * sinp),
            (7, 5, sin2t * cos2p * 0.5),
            (8, 5, sin2t * sin2p * 0.5),
        ],
        [
            (4, 6, 0.0),
            (5, 6, -cost * sinp),
            (6, 6, cost * cosp),
            (7, 6, -sint * sin2p),
            (8, 6, sint * cos2p),
        ],
        [
            (4, 7, s3 * sint**2 * 0.5),
            (5, 7, -sin2t * cosp * 0.5),
            (6, 7, -sin2t * sinp * 0.5),
            (7, 7, (1.0 + cost**2) * cos2p * 0.5),
            (8, 7, (1.0 + cost**2) * sin2p * 0.5),
        ],
        [
            (4, 8, 0.0),
            (5, 8, sint * sinp),
            (6, 8, -sint * cosp),
            (7, 8, -cost * sin2p),
            (8, 8, cost * cos2p),
        ],
    ]
    for block in d:
        for r, c, v in block:
            T[r, c] = v
    return T
