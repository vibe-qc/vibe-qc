"""Hessian update formulas -- families beyond BFGS.

For geometry optimisation, Hessian (or inverse-Hessian) updates are
critical for quasi-Newton and transition-state methods.  This module
provides symmetric rank-1 (SR1), Powell, Bofill, and Murtagh-Sargent
updates, plus a unified factory.

All functions operate on the full (n x n) matrix.  L-BFGS two-loop
recursion is handled separately in :mod:`vibeqc.geomopt.optimizers`.

References
----------
* Broyden, IMA J. Appl. Math. 6, 76 (1970) -- SR1.
* Fletcher, Comput. J. 13, 317 (1970) -- BFGS basis.
* Powell, Math. Program. 1, 26 (1971) -- PSB / Powell-symmetric-Broyden.
* Bofill, J. Comput. Chem. 15, 1 (1994) -- Bofill weighted update.
* Murtagh & Sargent, Comput. J. 13, 185 (1970) -- MS update.
"""

from __future__ import annotations

import numpy as np

# C++ acceleration
_cpp = None


def _get_cpp():
    global _cpp
    if _cpp is None:
        try:
            from .._vibeqc_core import (
                hessian_update_bfgs as _bfgs,
            )
            from .._vibeqc_core import (
                hessian_update_bofill as _bof,
            )
            from .._vibeqc_core import (
                hessian_update_ms as _ms,
            )
            from .._vibeqc_core import (
                hessian_update_powell as _pow,
            )
            from .._vibeqc_core import (
                hessian_update_sr1 as _sr1,
            )

            _cpp = type(
                "Cpp",
                (),
                {
                    "bfgs": _bfgs,
                    "sr1": _sr1,
                    "powell": _pow,
                    "bofill": _bof,
                    "ms": _ms,
                },
            )()
        except ImportError:
            _cpp = False
    return _cpp if _cpp is not False else None


# ---------------------------------------------------------------------------
# Hessian (not inverse) updates -- H_{k+1} = H_k + ΔH
# ---------------------------------------------------------------------------


def bfgs_hessian_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """BFGS update of the Hessian (not inverse).

    H_{k+1} = H_k + y y^T / (y^T s) - (H_k s)(H_k s)^T / (s^T H_k s)
    """
    sy = float(y @ s)
    if sy <= 1e-12:
        return H
    Hs = H @ s
    sHs = float(s @ Hs)
    if sHs <= 1e-12:
        return H
    return H + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs


def sr1_hessian_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Symmetric rank-1 (SR1) Hessian update.

    H_{k+1} = H_k + (y - H s)(y - H s)^T / ((y - H s)^T s)

    The SR1 update is not guaranteed positive-definite, which makes it
    suitable for transition-state searches (where the Hessian has one
    negative eigenvalue).  Skipped when the denominator is too small.
    """
    r = y - H @ s
    rs = float(r @ s)
    if abs(rs) < 1e-12:
        return H
    return H + np.outer(r, r) / rs


def powell_hessian_update(H: np.ndarray, s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Powell-symmetric-Broyden (PSB) Hessian update.

    H_{k+1} = H_k + (r s^T + s r^T) / (s^T s) - (r^T s) s s^T / (s^T s)^2

    where r = y - H s.  This is the update Q-Chem uses as its default
    for transition-state optimisations (alongside Bofill).
    """
    r = y - H @ s
    ss = float(s @ s)
    if ss <= 1e-12:
        return H
    rs = float(r @ s)
    return H + (np.outer(r, s) + np.outer(s, r)) / ss - rs * np.outer(s, s) / (ss * ss)


def bofill_hessian_update(
    H: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Bofill weighted Hessian update (J. Comput. Chem. 15, 1, 1994).

    A convex combination of the PSB and SR1 updates, weighted by

        phi = (r^T s)^2 / (‖r‖^2 . ‖s‖^2)

    where r = y - H s.  phi -> 1 favours PSB; phi -> 0 favours SR1.
    Bofill showed this is more robust than either alone for TS searches.
    """
    r = y - H @ s
    rs = float(r @ s)
    ss = float(s @ s)
    rr = float(r @ r)
    if ss <= 1e-12 or rr <= 1e-12:
        return H

    # phi = cos^2(angle between r and s) -- how well the quasi-Newton
    # condition is satisfied.
    phi = (rs * rs) / (rr * ss)
    phi = max(0.0, min(1.0, phi))

    # PSB update
    H_psb = H + (np.outer(r, s) + np.outer(s, r)) / ss - rs * np.outer(s, s) / (ss * ss)

    # SR1 update (only if denominator is safe)
    if abs(rs) < 1e-12:
        return H_psb
    H_sr1 = H + np.outer(r, r) / rs

    return phi * H_psb + (1.0 - phi) * H_sr1


def murtagh_sargent_hessian_update(
    H: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Murtagh-Sargent symmetric rank-2 Hessian update.

    A variable-metric update that preserves sparsity structure better
    than BFGS.  Used in some QC geometry optimisers for TS searches.

    H_{k+1} = H_k + (y y^T) / (y^T s) - (H_k s)(H_k s)^T / (s^T H_k s)
              + th w w^T

    where w = y/(y^T s) - H_k s/(s^T H_k s) and th = (y^T s)(s^T H_k s)
    ensures the update satisfies the secant condition.
    """
    sy = float(y @ s)
    if sy <= 1e-12:
        return H
    Hs = H @ s
    sHs = float(s @ Hs)
    if sHs <= 1e-12:
        return H

    # Standard BFGS (Hessian form)
    H_bfgs = H + np.outer(y, y) / sy - np.outer(Hs, Hs) / sHs

    # Murtagh-Sargent correction
    w = y / sy - Hs / sHs
    theta = sy * sHs
    return H_bfgs + theta * np.outer(w, w)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_HESSIAN_UPDATERS = {
    "none": lambda H, s, y: H,
    "bfgs": bfgs_hessian_update,
    "sr1": sr1_hessian_update,
    "powell": powell_hessian_update,
    "bofill": bofill_hessian_update,
    "murtagh_sargent": murtagh_sargent_hessian_update,
    "ms": murtagh_sargent_hessian_update,
}


def resolve_hessian_update(name: str):
    """Return a Hessian update function for *name*.

    Parameters
    ----------
    name : str
        ``"none"``, ``"bfgs"``, ``"sr1"``, ``"powell"``, ``"bofill"``,
        or ``"murtagh_sargent"``.
    """
    try:
        return _HESSIAN_UPDATERS[name.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown hessian_update={name!r}.  Available: {sorted(_HESSIAN_UPDATERS)}"
        ) from None
