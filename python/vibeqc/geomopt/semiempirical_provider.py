"""Semiempirical provider for the uniform geometry optimization framework.

:class:`SemiempiricalProvider` wraps the unified molecular
semiempirical dispatcher as an
:class:`~vibeqc.geomopt.providers.EnergyGradientProvider`.  Any
geometry optimisation can use it as a drop-in energy+gradient
backend.

Supported methods
-----------------
``"msindo"``
    MSINDO semiempirical model (closed-shell RHF or open-shell UHF).
    Closed-shell INDO gradients use the native analytic route exposed by
    :func:`vibeqc.semiempirical.run_semiempirical`; energy-only MSINDO
    results keep the documented finite-difference force fallback.
``"gfn2-xtb"``
    GFN2-xTB tight-binding model (experimental --- see
    :mod:`vibeqc.semiempirical.methods.gfn2`).
``"pm6"``
    PM6 NDDO model (closed-shell or open-shell).
``"dftb0"``, ``"scc-dftb"``
    DFTB0 non-SCC and SCC tight-binding.
``"om1"``, ``"om2"``, ``"om3"``
    OM1/OM2/OM3 NDDO models.
"""

from __future__ import annotations

import warnings

import numpy as np

from .._vibeqc_core import Molecule
from ..semiempirical.routes import (
    MOLECULAR_SEMIEMPIRICAL_METHODS,
    SemiempiricalRoutePlan,
    normalise_semiempirical_method,
)
from ..semiempirical.runner import run_semiempirical


class SemiempiricalProvider:
    """An :class:`~vibeqc.geomopt.providers.EnergyGradientProvider` backed
    by vibe-qc's semiempirical drivers.

    Parameters
    ----------
    method : str
        Semiempirical method: ``"msindo"``, ``"gfn2-xtb"``, ``"pm6"``,
        ``"dftb0"``, ``"scc-dftb"``, ``"om1"``, ``"om2"``, or ``"om3"``.
    charge : int or None
        Optional net molecular charge override.  When omitted, the charge
        stored on the per-call ``Molecule`` is used.
    multiplicity : int or None
        Optional spin multiplicity override.  When omitted, the multiplicity
        stored on the per-call ``Molecule`` is used.
    """

    def __init__(
        self,
        method: str,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
    ):
        _method_key = normalise_semiempirical_method(method)
        if _method_key not in MOLECULAR_SEMIEMPIRICAL_METHODS:
            raise ValueError(
                f"Unknown semiempirical method {method!r}.  "
                f"Available: {sorted(MOLECULAR_SEMIEMPIRICAL_METHODS)}"
            )
        plan = SemiempiricalRoutePlan.from_request(
            method,
            boundary="molecule",
            charge=0 if charge is None else int(charge),
            multiplicity=1 if multiplicity is None else int(multiplicity),
        )
        self._method_key = plan.method_key
        self._charge = None if charge is None else int(charge)
        self._multiplicity = None if multiplicity is None else int(multiplicity)
        self._method = self._method_key  # for checkpoint metadata

    def __call__(self, molecule: Molecule) -> tuple[float, np.ndarray]:
        """Evaluate (energy, gradient) at *molecule* (positions in bohr).

        Returns
        -------
        energy : float
            Total energy in Hartree.
        gradient : np.ndarray
            Flat ``(3*n_atoms,)`` gradient in Ha/bohr.
        """
        if self._charge is not None or self._multiplicity is not None:
            charge = molecule.charge if self._charge is None else self._charge
            multiplicity = (
                molecule.multiplicity
                if self._multiplicity is None
                else self._multiplicity
            )
            molecule = Molecule(list(molecule.atoms), charge, multiplicity)

        result = _run_unified(self._method_key, molecule)
        gradient = result.gradient()
        if gradient is None and self._method_key == "msindo":
            gradient = _msindo_fd_gradient(molecule)
        if gradient is None:
            raise NotImplementedError(
                "SemiempiricalProvider requires an energy gradient for "
                f"method={self._method_key!r}; this route currently exposes "
                "energy only."
            )
        return float(result.energy), np.asarray(gradient, dtype=float).ravel()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _run_unified(method_key: str, molecule: Molecule):
    """Run the shared semiempirical dispatcher for a geometry-provider call."""
    if method_key == "gfn2_xtb":
        from vibeqc.semiempirical.methods.gfn2 import GFN2ExperimentalWarning

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GFN2ExperimentalWarning)
            return run_semiempirical(method_key, molecule)
    return run_semiempirical(method_key, molecule)


def _msindo_fd_gradient(molecule: Molecule) -> np.ndarray:
    """Legacy MSINDO provider gradient fallback for routes without native grad."""
    from vibeqc.semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR,
        msindo_gradient_fd,
    )

    z_values = [int(atom.Z) for atom in molecule.atoms]
    coords_angstrom = (
        np.array([list(atom.xyz) for atom in molecule.atoms], dtype=float)
        / ANGSTROM_TO_BOHR
    )
    return msindo_gradient_fd(
        z_values,
        coords_angstrom,
        charge=molecule.charge,
        multiplicity=molecule.multiplicity,
    )


def _build_model(method_key: str, molecule: Molecule):
    """Build a semiempirical model instance for *method_key* at *molecule*."""
    method_key = normalise_semiempirical_method(method_key)
    if method_key == "dftb0":
        from vibeqc.semiempirical import DFTB0Model

        return DFTB0Model(molecule)
    elif method_key == "scc_dftb":
        from vibeqc.semiempirical import SCCDFTBModel

        return SCCDFTBModel(molecule)
    elif method_key == "gfn2_xtb":
        from vibeqc.semiempirical import GFN2Model

        return GFN2Model(molecule, warn=False)
    elif method_key == "pm6":
        from vibeqc.semiempirical import PM6Model, UPM6Model

        if molecule.multiplicity > 1:
            return UPM6Model(molecule)
        return PM6Model(molecule)
    elif method_key in ("om1", "om2", "om3"):
        from vibeqc.semiempirical import OMxModel

        return OMxModel(molecule, variant=method_key)
    else:
        raise ValueError(f"Unknown semiempirical model: {method_key!r}")
