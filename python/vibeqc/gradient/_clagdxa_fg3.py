"""CLagDXA_FG3: contract BDER/SDER through 3-RDM KTUV symmetry.

Implements the 12-fold permutational symmetry contraction from
OpenMolcas's clagdxa_fg3.F90.  Takes BDER/SDER in active-MO space,
converts to KTUV superindex space, contracts through the 3-RDM, and
produces DF3/DG3 plus delta-function corrections to lower-order tensors.

This is the last missing piece before the full analytic Lagrangian:
  CLagDX → BDER/SDER → CLagDXA_FG3 → DF3/DG3 → derfg3 → W^z/D^z/Gamma^z
"""

from __future__ import annotations

import numpy as np


def _ktuv(t: int, u: int, v: int, n: int) -> int:
    """KTUV compound superindex (zero-symmetry, 0-indexed)."""
    return (t * n + u) * n + v


def _g3_idx(t: int, u: int, v: int, x: int, y: int, z: int, n: int) -> int:
    """Flat index into G3[t,u,v,x,y,z]."""
    return ((((t * n + v) * n + x) * n + y) * n + z) * n + u


def clagdxa_fg3_contract_full(
    BDER_mo: np.ndarray,
    SDER_mo: np.ndarray,
    G3_flat: np.ndarray,
    G1: np.ndarray,
    G2: np.ndarray,
    eps_act: np.ndarray,
    n_act: int,
) -> dict:
    """Full CLagDXA_FG3: BDER/SDER → DF3/DG3 + delta-function corrections.

    Converts active-MO BDER/SDER to KTUV superindex space (diagonal
    approximation), applies 12-fold KTUV symmetry contraction with the
    3-RDM, and produces DF3, DG3, DF2, DG2, DF1, DG1, DEPSA corrections.

    Parameters
    ----------
    BDER_mo, SDER_mo : (n_act, n_act) ndarray
        From CLagDX (compute_bder_sder_from_ic or SC approximation).
    G3_flat : (n_act^6,) ndarray
        3-RDM from state_rdm3_cpp.
    G1 : (n_act, n_act) ndarray
        1-RDM in E-operator convention.
    G2 : (n_act, n_act, n_act, n_act) ndarray
        2-RDM in E-operator convention.
    eps_act : (n_act,) ndarray
        Active orbital energies.
    n_act : int

    Returns
    -------
    corrections : dict
        DF3, DG3, DF2, DG2, DF1, DG1, DEPSA — all active-space.
    """
    n = n_act
    n6 = n * n * n * n * n * n
    nTUV = n * n * n

    # Step 1: Convert BDER/SDER to KTUV superindex space (diagonal approx)
    BDER_si = np.zeros((nTUV, nTUV))
    SDER_si = np.zeros((nTUV, nTUV))
    for t in range(n):
        for u in range(n):
            b_tu = BDER_mo[t, u]
            s_tu = SDER_mo[t, u]
            if abs(b_tu) < 1e-16 and abs(s_tu) < 1e-16:
                continue
            for v in range(n):
                iTUV = _ktuv(t, u, v, n)
                BDER_si[iTUV, iTUV] += b_tu
                SDER_si[iTUV, iTUV] += s_tu

    # Step 2: Initialize output arrays
    DF3 = np.zeros(n6)
    DG3 = np.zeros(n6)
    DF2 = np.zeros((n, n, n, n))
    DG2 = np.zeros((n, n, n, n))
    DF1 = np.zeros((n, n))
    DG1 = np.zeros((n, n))
    DEPSA = np.zeros((n, n))

    G3_6d = G3_flat.reshape(n, n, n, n, n, n)

    # Step 3: 12-fold KTUV symmetry contraction (clagdxa_fg3 lines 56-163)
    for t in range(n):
        for u in range(n):
            for v in range(n):
                for x in range(n):
                    for y in range(n):
                        for z in range(n):
                            gval = G3_6d[t, u, v, x, y, z]
                            if abs(gval) < 1e-15:
                                continue

                            f3 = 0.0
                            g3 = 0.0

                            # ---- 12 equivalent cases (clagdxa_fg3 lines 56-163) ----
                            # Case 1: XUT, VYZ
                            f3 += BDER_si[_ktuv(x, u, t, n), _ktuv(v, y, z, n)]
                            g3 += SDER_si[_ktuv(x, u, t, n), _ktuv(v, y, z, n)]

                            # Distinctness checks
                            itu = _ktuv(t, u, 0, n)
                            ivx = _ktuv(v, x, 0, n)
                            iyz = _ktuv(y, z, 0, n)
                            dp = (itu != ivx) or (ivx != iyz)
                            all_distinct = (
                                (itu != ivx) and (itu != iyz) and (ivx != iyz)
                            )

                            if dp:
                                if all_distinct:
                                    # Case 2: UXV, TYZ
                                    f3 += BDER_si[_ktuv(u, x, v, n), _ktuv(t, y, z, n)]
                                    g3 += SDER_si[_ktuv(u, x, v, n), _ktuv(t, y, z, n)]
                                    # Case 3: XZY, VTU
                                    f3 += BDER_si[_ktuv(x, z, y, n), _ktuv(v, t, u, n)]
                                    g3 += SDER_si[_ktuv(x, z, y, n), _ktuv(v, t, u, n)]
                                    # Case 4: ZUT, YVX
                                    f3 += BDER_si[_ktuv(z, u, t, n), _ktuv(y, v, x, n)]
                                    g3 += SDER_si[_ktuv(z, u, t, n), _ktuv(y, v, x, n)]
                                # Case 5: UZY, TVX
                                f3 += BDER_si[_ktuv(u, z, y, n), _ktuv(t, v, x, n)]
                                g3 += SDER_si[_ktuv(u, z, y, n), _ktuv(t, v, x, n)]
                                # Case 6: ZXV, YTU
                                f3 += BDER_si[_ktuv(z, x, v, n), _ktuv(y, t, u, n)]
                                g3 += SDER_si[_ktuv(z, x, v, n), _ktuv(y, t, u, n)]

                            # Reflected half (cases 7-12)
                            idx_distinct = (
                                ((t != u) or (v != x) or (y != z))
                                and ((t != u) or (v != z) or (x != y))
                                and ((x != v) or (t != z) or (u != y))
                                and ((z != y) or (v != u) or (x != t))
                            )
                            if idx_distinct:
                                # Case 7: VTU, XZY
                                f3 += BDER_si[_ktuv(v, t, u, n), _ktuv(x, z, y, n)]
                                g3 += SDER_si[_ktuv(v, t, u, n), _ktuv(x, z, y, n)]
                                if dp:
                                    if all_distinct:
                                        # Case 8: TVX, UZY
                                        f3 += BDER_si[
                                            _ktuv(t, v, x, n), _ktuv(u, z, y, n)
                                        ]
                                        g3 += SDER_si[
                                            _ktuv(t, v, x, n), _ktuv(u, z, y, n)
                                        ]
                                        # Case 9: VYZ, XUT
                                        f3 += BDER_si[
                                            _ktuv(v, y, z, n), _ktuv(x, u, t, n)
                                        ]
                                        g3 += SDER_si[
                                            _ktuv(v, y, z, n), _ktuv(x, u, t, n)
                                        ]
                                        # Case 10: YTU, ZXV
                                        f3 += BDER_si[
                                            _ktuv(y, t, u, n), _ktuv(z, x, v, n)
                                        ]
                                        g3 += SDER_si[
                                            _ktuv(y, t, u, n), _ktuv(z, x, v, n)
                                        ]
                                    # Case 11: TYZ, UXV
                                    f3 += BDER_si[_ktuv(t, y, z, n), _ktuv(u, x, v, n)]
                                    g3 += SDER_si[_ktuv(t, y, z, n), _ktuv(u, x, v, n)]
                                    # Case 12: YVX, ZUT
                                    f3 += BDER_si[_ktuv(y, v, x, n), _ktuv(z, u, t, n)]
                                    g3 += SDER_si[_ktuv(y, v, x, n), _ktuv(z, u, t, n)]

                            # Final sign (clagdxa_fg3 line 165-166)
                            f3 = -f3
                            g3 = -g3

                            idx6 = _g3_idx(t, u, v, x, y, z, n)
                            DF3[idx6] += f3
                            DG3[idx6] += g3

                            # ---- Delta-function corrections (clagdxa_fg3 lines 188-209) ----
                            if y == x:
                                DF2[t, u, v, z] -= f3
                                DG2[t, u, v, z] -= eps_act[u] * f3 + g3
                                DEPSA[u, :] -= f3 * G2[t, :, v, z]
                            if v == u:
                                DF2[t, x, y, z] -= f3
                                DG2[t, x, y, z] -= eps_act[y] * f3 + g3
                                DEPSA[:, y] -= f3 * G2[t, x, :, z]
                            if y == u:
                                DF2[v, x, t, z] -= f3
                                DG2[v, x, t, z] -= eps_act[u] * f3 + g3
                            DEPSA[y, u] -= f3 * G2[v, x, t, z]
                            if (y == x) and (v == u):
                                DF1[t, z] -= f3
                                DG1[t, z] -= g3

    return {
        "DF3": DF3,
        "DG3": DG3,
        "DF2": DF2,
        "DG2": DG2,
        "DF1": DF1,
        "DG1": DG1,
        "DEPSA": DEPSA,
    }
