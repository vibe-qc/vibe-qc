"""Objective functions for basis-set optimisation.

The optimiser sees a 1-D parameter vector and expects to receive a
scalar (energy or energy-like quantity) plus optional gradient. This
module defines the protocol and ships two concrete classes:

* :class:`MockEnergy` -- a pure-python quadratic in the parameter
  vector. No vibe-qc calls. Used by the architecture smoke test
  to verify the parametrise + driver layer without requiring an
  installed package.
* :class:`SinglePointEnergy` -- runs vibe-qc SCF on one molecular
  system per call. Composes with :class:`BasisParametrisation` and
  :class:`TempBasisLibrary` to write the candidate basis, build a
  ``BasisSet``, and call ``vq.run_rhf`` / ``run_uhf``.

Multi-system objectives compose :class:`SinglePointEnergy` with
weights and a sum reducer; that lives in :mod:`recipes` rather than
here, so this module stays free of recipe-specific assumptions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol

import numpy as np

if TYPE_CHECKING:
    from .parametrise import BasisParametrisation


class Objective(Protocol):
    """Callable interface the optimiser drivers expect.

    Implementations may be stateful (caching SCF results across calls
    is fine, e.g. for warm-start guesses) but must be safe to call
    repeatedly with arbitrary parameter vectors. They MUST raise
    rather than return NaN on a failed evaluation; the driver decides
    how to recover.
    """

    n_params: int
    """Length of the parameter vector this objective expects."""

    n_calls: int
    """Number of evaluations performed so far (for logging / budget tracking)."""

    def __call__(self, x: np.ndarray) -> float:
        ...


@dataclass
class MockEnergy:
    """Quadratic mock objective used for architecture testing.

    ``E(x) = E0 + 0.5 * (x - center)^T H (x - center)``
    where ``H`` is a positive-definite Hessian (default: identity).

    Lets the smoke test verify that the optimiser drivers + the
    parametrise pack/unpack round-trip work, *without* depending on
    vibe-qc being importable.
    """

    n_params: int
    center: np.ndarray
    """Parameter vector at which the energy is minimised."""
    e0: float = 0.0
    """Energy at the minimum."""
    hessian: Optional[np.ndarray] = None
    """Quadratic-form matrix; defaults to identity if not provided."""
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        if self.center.shape != (self.n_params,):
            raise ValueError(
                f"center has shape {self.center.shape}, expected ({self.n_params},)"
            )
        if self.hessian is None:
            self.hessian = np.eye(self.n_params)
        else:
            self.hessian = np.asarray(self.hessian, dtype=float)
            if self.hessian.shape != (self.n_params, self.n_params):
                raise ValueError(
                    f"hessian has shape {self.hessian.shape}, "
                    f"expected ({self.n_params}, {self.n_params})"
                )

    def __call__(self, x: np.ndarray) -> float:
        self.n_calls += 1
        d = np.asarray(x, dtype=float) - self.center
        return float(self.e0 + 0.5 * d @ self.hessian @ d)


@dataclass
class SinglePointEnergy:
    """Vibe-qc SCF energy on one molecular system, evaluated per call.

    The objective writes the current candidate basis to a temp
    ``.g94``, builds ``vq.BasisSet(molecule, name)``, runs the SCF,
    and returns the total energy in Hartree. SCF failure raises
    rather than returning a sentinel; callers may wrap with a
    failure-tolerant decorator that adds a large penalty instead.

    Parameters
    ----------
    parametrisation : BasisParametrisation
        Maps the optimiser parameter vector to a candidate basis dict.
    molecule_factory : Callable[[], Any]
        Returns a fresh ``vibeqc.Molecule``. Recreated each call so
        the molecule's basis-cache (if any) does not pin a stale
        BasisSet across iterations. Cheap.
    library : TempBasisLibrary
        Already-entered context that owns ``$LIBINT_DATA_PATH``.
    scf_runner : Callable[[mol, basis], result]
        Bound vibe-qc SCF entry point -- typically ``vq.run_rhf`` or
        ``vq.run_uhf``. Result must expose ``.energy`` (Hartree).
    basis_name_prefix : str
        Prefix for the auto-generated basis name (uniqueness is via
        a uuid suffix).
    """

    parametrisation: "BasisParametrisation"
    molecule_factory: Callable[[], Any]
    library: Any  # TempBasisLibrary, but typed loosely to avoid circular import
    scf_runner: Callable[..., Any]
    basis_name_prefix: str = "opt"
    n_calls: int = field(default=0, init=False)

    @property
    def n_params(self) -> int:
        return len(self.parametrisation)

    def __call__(self, x: np.ndarray) -> float:
        from .. import BasisSet  # local: vibeqc not always importable at module load

        self.n_calls += 1
        atoms = self.parametrisation.unpack(np.asarray(x, dtype=float))
        basis_name = self.library.write_g94(
            atoms, basis_name=f"{self.basis_name_prefix}-{self.n_calls:06d}"
        )
        mol = self.molecule_factory()
        basis = BasisSet(mol, basis_name)
        result = self.scf_runner(mol, basis)
        return float(result.energy)
