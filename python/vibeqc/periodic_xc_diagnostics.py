"""Grid electron-count diagnostic for the periodic XC quadrature (#385).

The periodic XC kernels evaluate the crystal density on the home-cell Becke
grid as a sum over lattice images of the AOs, out to an image reach set by
``LatticeSumOptions`` (``cutoff_bohr``, optionally narrowed by
``becke_image_radius_bohr``). When that reach is shorter than the density's
extent, the grid density loses electrons and E_xc is wrong, silently: #265
lost 0.4 % of the electrons (8 mHa in E_xc), #385 found
``int rho = 1.9925`` for N = 2 at 8/6 bohr.

The kernels report ``n_electrons_grid = sum_g w_g rho(r_g)``. For a periodic
density it must equal the electron count per cell to quadrature accuracy.
The default periodic Becke grid (17 x 36 product angular points) integrates a
cell-filling density to about 4.3e-4 relative (#268 note 26884), so the
default warning threshold is 1e-3 relative: above the quadrature floor,
below the truncations reported so far. The check warns; it never raises.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, MutableMapping, Optional

XC_GRID_ELECTRON_RTOL = 1.0e-3


class PeriodicXCGridChargeWarning(RuntimeWarning):
    """The periodic XC grid density does not integrate to N electrons."""


def record_xc_grid_charge(
    diagnostics: Optional[MutableMapping[str, Any]], contribution: Any
) -> None:
    """Store the grid integral of one XC evaluation (the last one wins)."""
    if diagnostics is None:
        return
    n_grid = getattr(contribution, "n_electrons_grid", None)
    if n_grid is None:
        return
    diagnostics["n_electrons_grid"] = float(n_grid)
    diagnostics["periodic_density_bras"] = bool(
        getattr(contribution, "periodic_density_bras", False)
    )


def check_xc_grid_electron_count(
    diagnostics: Optional[MutableMapping[str, Any]],
    n_electrons: float,
    *,
    context: str,
    lattice_options: Any = None,
    rtol: float = XC_GRID_ELECTRON_RTOL,
    stacklevel: int = 2,
) -> Optional[float]:
    """Warn when the last recorded XC grid density misses ``n_electrons``.

    Returns the relative error ``(int rho - N) / N``, or ``None`` when nothing
    was recorded or the density was a home-cell (molecular-limit) density,
    whose grid integral is not the cell electron count.
    """
    if not diagnostics or "n_electrons_grid" not in diagnostics:
        return None
    if not diagnostics.get("periodic_density_bras", False):
        return None
    n_ref = float(n_electrons)
    if not (n_ref > 0.0):
        return None
    n_grid = float(diagnostics["n_electrons_grid"])
    rel = (n_grid - n_ref) / n_ref
    diagnostics["relative_error"] = rel
    if not math.isfinite(rel) or abs(rel) > rtol:
        reach = ""
        if lattice_options is not None:
            reach = (
                f" (cutoff_bohr={float(lattice_options.cutoff_bohr):g}, "
                "becke_image_radius_bohr="
                f"{float(getattr(lattice_options, 'becke_image_radius_bohr', 0.0)):g})"
            )
        warnings.warn(
            f"{context}: the XC grid density integrates to {n_grid:.6f} "
            f"electrons per cell, not N = {n_ref:g} (relative {rel:+.2e}, "
            f"tolerance {rtol:g}){reach}. The XC image reach or the grid is "
            "too short for this density; converge them before trusting E_xc.",
            PeriodicXCGridChargeWarning,
            stacklevel=stacklevel,
        )
    return rel


def report_xc_grid_charge(
    result: Any,
    diagnostics: Optional[MutableMapping[str, Any]],
    n_electrons: float,
    *,
    context: str,
    lattice_options: Any = None,
) -> None:
    """Attach the final XC grid electron count to an SCF result and warn.

    Sets ``result.xc_grid_electrons`` (``sum_g w_g rho``, or ``None``) and
    ``result.xc_grid_electron_error`` (relative to ``n_electrons``, or
    ``None`` for a molecular-limit density), then applies
    :func:`check_xc_grid_electron_count`.
    """
    rel = check_xc_grid_electron_count(
        diagnostics,
        n_electrons,
        context=context,
        lattice_options=lattice_options,
        stacklevel=3,
    )
    n_grid = None if not diagnostics else diagnostics.get("n_electrons_grid")
    try:
        result.xc_grid_electrons = None if n_grid is None else float(n_grid)
        result.xc_grid_electron_error = rel
    except AttributeError:
        pass


__all__ = [
    "XC_GRID_ELECTRON_RTOL",
    "PeriodicXCGridChargeWarning",
    "check_xc_grid_electron_count",
    "record_xc_grid_charge",
    "report_xc_grid_charge",
]
