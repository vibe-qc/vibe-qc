"""MSINDO STO integral derivatives w.r.t. interatomic distance R.

Ported from MSINDO Fortran: dc2int.f, dharri.f, ds2int.f (the last is already
in msindo_integrals via the overlp(grad=True) flag -- re-exported here for
convenience).  Every function returns dI/dR for the corresponding I(R).
"""

from __future__ import annotations

import math

from . import msindo_integrals as _ki


def ds2int(n1, l1, m1, z1, n2, l2, m2, z2, R) -> float:
    """Derivative d/dR of two-center overlap <mu_a|nu_b>."""
    return _ki.ds2int(n1, l1, m1, z1, n2, l2, m2, z2, R)


def _dharris(n1, l1, z1, n2, l2, z2, m, R) -> float:
    """Derivative d/dR of the Harris auxiliary function (dharri.f)."""
    l1l2m = l1 + l2 - m
    vf1 = 4.0 * math.pi / (z1 * z1)
    vf2 = 4.0 * math.pi / (z2 * z2)

    # Formula 17 derivative
    dfor17 = vf1 * (
        _ki.overlp(0, 0, 0.0, l1l2m, l1l2m, z2, 0, R, grad=True)
        - _ki.overlp(0, 0, z1, l1l2m, l1l2m, z2, 0, R, grad=True)
    )
    form17 = vf1 * (
        _ki.overlp(0, 0, 0.0, l1l2m, l1l2m, z2, 0, R)
        - _ki.overlp(0, 0, z1, l1l2m, l1l2m, z2, 0, R)
    )

    # Formula 30 derivative
    dsum = 0.0
    for j in range(0, m):
        ds_val = (j - m) * R ** (j - m - 1) * _ki.overlp(
            j + 1, j, z1, l1l2m, l1l2m, z2, j, R
        ) + R ** (j - m) * _ki.overlp(j + 1, j, z1, l1l2m, l1l2m, z2, j, R, grad=True)
        prefac = ((-z1 * z1) ** (j - m)) / (2 * j + 2)
        fac0 = math.sqrt(
            _ki.facs(2 * m + 1)
            * _ki.facs(l1l2m + m)
            * _ki.facs(l1l2m - j)
            / (_ki.facs(2 * j + 1) * _ki.facs(l1l2m - m) * _ki.facs(l1l2m + j))
        )
        dsum += prefac * fac0 * ds_val
    sqrt_factor = math.sqrt(
        _ki.facs(2 * m + 1) * _ki.facs(l1l2m + m) / _ki.facs(l1l2m - m)
    )
    dfor30 = (-1.0 / z1 / z1) ** m * sqrt_factor * (
        -m * R ** (-m - 1) * form17 + R ** (-m) * dfor17
    ) - vf1 * z1 * dsum

    # Formula 26 derivative (L-M recursion)
    dfor26 = dfor30
    for j in range(1, l1 - m + 1):
        L = m + j - 1
        ls = l1l2m - j + 1
        w1 = math.sqrt((L - m + 1) * (L + m + 1) / ((2 * L + 1) * (2 * L + 3)))
        w2 = math.sqrt((ls - m) * (ls + m) / ((2 * ls - 1) * (2 * ls + 1)))
        fak1 = z1 / z2 * 2 * ls * w1
        fak2 = z2 / z1 * (2 * L + 2) * w2
        dsum_term = (
            vf1
            * math.sqrt((L + m) * (L - m) / ((2 * L - 1) * (2 * L + 1)))
            * (
                z1 * _ki.overlp(L + 1, L - 1, z1, ls, ls - 1, z2, m, R, grad=True)
                + _ki.overlp(L, L - 1, z1, ls, ls - 1, z2, m, R, grad=True)
            )
            + vf2
            * math.sqrt((ls + m - 1) * (ls - m - 1) / ((2 * ls - 3) * (2 * ls - 1)))
            * (
                z2 * _ki.overlp(L + 1, L, z1, ls, ls - 2, z2, m, R, grad=True)
                + _ki.overlp(L + 1, L, z1, ls - 1, ls - 2, z2, m, R, grad=True)
            )
            + vf2
            * z1
            * w1
            * _ki.overlp(L + 1, L + 1, z1, ls, ls - 1, z2, m, R, grad=True)
            + vf1 * z2 * w2 * _ki.overlp(L + 1, L, z1, ls, ls, z2, m, R, grad=True)
        )
        dfor26 = (dsum_term - fak2 * dfor26) / fak1

    # Formula 23 (final assembly)
    vf1_n = _ki.facs(n1 + l1 + 1) / _ki.facs(2 * l1 + 1)
    dsum1 = 0.0
    dsum2 = 0.0
    for j in range(n2, l2, -1):
        dsum1 += (
            z2 ** (j - n2 - 2)
            * (
                _ki.facs(n2 + l2 + 1) / _ki.facs(j + l2)
                - _ki.facs(n2 - l2) / _ki.facs(j - l2 - 1)
            )
            * _ki.overlp(l1, l1, z1, j, l2, z2, m, R, grad=True)
        )
    for j in range(n1, l1, -1):
        dsum2 += (
            z1 ** (j - n1 - 2)
            * (
                _ki.facs(n1 + l1 + 1) / _ki.facs(j + l1)
                - _ki.facs(n1 - l1) / _ki.facs(j - l1 - 1)
            )
            * _ki.overlp(j, l1, z1, n2, l2, z2, m, R, grad=True)
        )
    return (
        vf1_n
        * _ki.facs(n2 + l2 + 1)
        / _ki.facs(2 * l2 + 1)
        * z1 ** (l1 - n1)
        * z2 ** (l2 - n2)
        * dfor26
        - 4.0 * math.pi / (2 * l2 + 1) * vf1_n * z1 ** (l1 - n1) * dsum1
        - 4.0 * math.pi / (2 * l1 + 1) * dsum2
    )


def dc2int(n1, l1, m1, z1, n2, l2, m2, z2, R) -> float:
    """Derivative d/dR of two-center Coulomb integral (dc2int.f)."""
    m = abs(m1)
    ms = abs(m2)
    nh1 = 2 * n1 - 1
    nh2 = 2 * n2 - 1
    zeth1 = 2 * z1
    zeth2 = 2 * z2
    fak = zeth1 ** (nh1 + 2) * zeth2 ** (nh2 + 2) / _ki.facs(2 * n1) / _ki.facs(2 * n2)
    s = 0.0
    for j in range(0, 2 * l1 + 1, 2):
        for js in range(0, 2 * l2 + 1, 2):
            s += (
                _ki.a0(l1, j, m)
                * _ki.a0(l2, js, ms)
                * _dharris(nh1, j, zeth1, nh2, js, zeth2, 0, R)
            )
    return fak * s


def dv2int(n1, l1, m1, z1, R) -> float:
    """Derivative d/dR of one-center nuclear attraction <mu_a|1/r_b|mu_a>."""
    m = abs(m1)
    nh = 2 * n1 - 1
    zeth = 2 * z1
    fak = zeth ** (nh + 2) / _ki.facs(2 * n1) * 2.0 * math.sqrt(math.pi)
    s = 0.0
    for i in range(0, 2 * l1 + 1, 2):
        s += _ki.a0(l1, i, m) * _ki.overlp(nh, i, zeth, 0, 0, 0.0, 0, R, grad=True)
    return fak * s
