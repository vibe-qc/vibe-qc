"""Fail-closed compatibility wrappers for PM7/UPM7.

The retired prototype reused the PM6 NDDO kernel with PM7 parameters but did
not implement Stewart's feathered electron-electron, electron-core, and
core-core electrostatics.  The parameter registry remains available for
completing that work, but these model surfaces cannot execute it.

Reference:
  J. J. P. Stewart, J. Mol. Model. 19, 1-32 (2013).
"""

from __future__ import annotations

from typing import Any

from vibeqc import Molecule
from vibeqc.semiempirical.model import SemiempiricalModel
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan


#: The one canonical gate message every PM7/UPM7 rejection path surfaces.
#: routes._validate_boundary and run_job both raise this text so a gated
#: request never falls back to a generic basis/dispatch error (issue #413).
PM7_GATE_MESSAGE = (
    "PM7 is gated until the published feathered electrostatics and all "
    "coupled NDDO terms are implemented and validated."
)


def _raise_pm7_gate(mol: Molecule, method: str) -> None:
    """Raise the canonical scientific gate for an incomplete PM7 request."""
    SemiempiricalRoutePlan.from_request(
        method,
        boundary="molecule",
        charge=int(mol.charge),
        multiplicity=int(mol.multiplicity),
    )
    raise NotImplementedError(PM7_GATE_MESSAGE)


class PM7Model(SemiempiricalModel):
    """Compatibility model that always rejects incomplete molecular PM7."""

    def __init__(
        self,
        mol: Molecule,
        params: Any = None,
        *,
        max_iter: int = 100,
        conv_tol: float = 1e-7,
        solver: str = "dense",
    ):
        super().__init__(mol, solver=solver)
        del params, max_iter, conv_tol
        _raise_pm7_gate(mol, "pm7")

    def energy(self) -> float:
        _raise_pm7_gate(self._mol, "pm7")

    def _energy_at(self, mol: Molecule) -> float:
        _raise_pm7_gate(mol, "pm7")


class UPM7Model(SemiempiricalModel):
    """Compatibility model that always rejects incomplete molecular UPM7."""

    def __init__(
        self,
        mol: Molecule,
        params: Any = None,
        *,
        max_iter: int = 100,
        conv_tol: float = 1e-7,
        solver: str = "dense",
    ):
        super().__init__(mol, solver=solver)
        del params, max_iter, conv_tol
        _raise_pm7_gate(mol, "upm7")

    def energy(self) -> float:
        _raise_pm7_gate(self._mol, "upm7")

    def _energy_at(self, mol: Molecule) -> float:
        _raise_pm7_gate(mol, "upm7")


__all__ = ["PM7Model", "UPM7Model"]
