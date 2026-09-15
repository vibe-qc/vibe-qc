"""Equation-of-state fitting helpers.

The article periodic screening scripts used to carry local Birch-Murnaghan
fitting code.  Keeping the quality checks here prevents nearly flat
semiempirical curves from being promoted to physical EOS minima just because a
four-parameter fit can find an interior stationary point.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


BM3_MODEL = "third_order_birch_murnaghan"
FIT_SUCCEEDED = "fit_succeeded"
ILL_CONDITIONED_FIT = "ill_conditioned_fit"


def birch_murnaghan_energy(
    volume: Sequence[float] | np.ndarray,
    e0: float,
    v0: float,
    b0: float,
    b0_prime: float,
) -> np.ndarray:
    """Return the third-order Birch-Murnaghan energy curve."""
    v = np.asarray(volume, dtype=float)
    eta2 = (float(v0) / v) ** (2.0 / 3.0)
    delta = eta2 - 1.0
    return float(e0) + 9.0 * float(v0) * float(b0) / 16.0 * (
        float(b0_prime) * delta**3 + delta**2 * (6.0 - 4.0 * eta2)
    )


def fit_birch_murnaghan(
    volumes: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
    *,
    min_energy_span_hartree: float = 1.0e-6,
    max_rmse_to_span: float = 0.25,
) -> dict[str, Any]:
    """Fit a BM3 EOS and flag numerically meaningless minima.

    Parameters
    ----------
    volumes, energies
        One-dimensional, finite arrays with matching lengths. Volumes must be
        positive.
    min_energy_span_hartree
        Minimum raw energy span required before a fitted minimum is considered
        scientifically usable. The default is deliberately above the
        microhartree-scale flat rare-gas DFTB0/SCC-DFTB diagnostic curves.
    max_rmse_to_span
        Maximum accepted fit residual divided by the raw energy span. If the
        residual is a material fraction of the total signal, the reported EOS
        parameters are descriptive diagnostics only.
    """
    x, y = _validate_xy(volumes, energies)
    if x.size < 4:
        raise ValueError("fit_birch_murnaghan requires at least four points")
    if min_energy_span_hartree < 0.0:
        raise ValueError("min_energy_span_hartree must be non-negative")
    if not math.isfinite(max_rmse_to_span) or max_rmse_to_span <= 0.0:
        raise ValueError("max_rmse_to_span must be finite and positive")

    energy_span = float(np.ptp(y))
    result: dict[str, Any]
    try:
        from scipy.optimize import curve_fit

        i_min = int(np.argmin(y))
        p0 = (float(y[i_min]), float(x[i_min]), 0.001, 4.0)
        lower = (-np.inf, float(x.min()) * 0.90, 1.0e-14, 0.0)
        upper = (np.inf, float(x.max()) * 1.10, np.inf, 12.0)
        params, _ = curve_fit(
            birch_murnaghan_energy,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=30000,
        )
        fitted = birch_murnaghan_energy(x, *params)
        e0, v0, b0, b0_prime = (float(value) for value in params)
        rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
        result = {
            "status": FIT_SUCCEEDED,
            "model": BM3_MODEL,
            "e0_hartree": e0,
            "v0_bohr3": v0,
            "b0_hartree_per_bohr3": b0,
            "b0_prime": b0_prime,
            "rmse_hartree": rmse,
            "energy_span_hartree": energy_span,
            "rmse_to_span": (
                math.inf if energy_span == 0.0 else float(rmse / energy_span)
            ),
            "minimum_bracketed": bool(x.min() <= v0 <= x.max()),
        }
    except Exception as exc:
        return {
            "status": ILL_CONDITIONED_FIT,
            "model": BM3_MODEL,
            "energy_span_hartree": energy_span,
            "reason": f"fit_failed: {type(exc).__name__}: {exc}",
            "minimum_bracketed": False,
        }

    reasons: list[str] = []
    if energy_span < float(min_energy_span_hartree):
        reasons.append(
            "energy_span_below_threshold:"
            f"{energy_span:.6g}<{float(min_energy_span_hartree):.6g}"
        )
    rmse_to_span = float(result["rmse_to_span"])
    if not math.isfinite(rmse_to_span) or rmse_to_span > float(max_rmse_to_span):
        reasons.append(
            "rmse_exceeds_span_fraction:"
            f"{rmse_to_span:.6g}>{float(max_rmse_to_span):.6g}"
        )
    if reasons:
        result["status"] = ILL_CONDITIONED_FIT
        result["reason"] = "; ".join(reasons)
    return result


def _validate_xy(
    volumes: Sequence[float] | np.ndarray,
    energies: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(volumes, dtype=float)
    y = np.asarray(energies, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("EOS volumes and energies must be one-dimensional")
    if x.shape != y.shape:
        raise ValueError("EOS volumes and energies must have the same shape")
    if x.size == 0:
        raise ValueError("EOS fit requires at least one point")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("EOS volumes and energies must be finite")
    if np.any(x <= 0.0):
        raise ValueError("EOS volumes must be positive")
    if np.ptp(x) <= 0.0:
        raise ValueError("EOS volumes must span more than one value")
    return x, y
