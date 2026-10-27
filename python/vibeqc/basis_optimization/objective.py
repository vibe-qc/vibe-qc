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
* :class:`PeriodicSinglePointEnergy` -- the periodic sibling: one
  ``vibeqc.periodic_runner.run_periodic_job`` per call on a
  ``PeriodicSystem`` at a fixed k-mesh, returning the energy per unit
  cell (Stage 2 of the basis-optimisation stages).

Multi-system objectives compose :class:`SinglePointEnergy` with
weights and a sum reducer; that lives in :mod:`recipes` rather than
here, so this module stays free of recipe-specific assumptions.

A least-squares objective additionally exposes its residual vector
(:class:`ResidualObjective`); :class:`MockResiduals` is the mock for
that shape, used to test the DFO-LS driver.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Protocol

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


class ResidualObjective(Protocol):
    """A least-squares objective: ``__call__`` returns ``sum_i r_i(x)^2``.

    :func:`vibeqc.basis_optimization.drivers.optimize_dfols` consumes the
    residual vector directly (a Gauss-Newton model per residual converges
    far faster than a model of the scalar sum). Any other driver sees only
    ``__call__``. The same failure rule as :class:`Objective` applies:
    raise, or return ``+inf`` entries, rather than NaN.
    """

    n_params: int
    n_calls: int

    def residuals(self, x: np.ndarray) -> np.ndarray:
        ...

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
class MockResiduals:
    """Least-squares mock: ``r(x) = A (x - center)``, objective ``|r|^2``.

    The residual-shaped counterpart of :class:`MockEnergy` for testing
    :func:`~vibeqc.basis_optimization.drivers.optimize_dfols`. With
    ``A`` of shape ``(m, n_params)`` and full column rank the objective is
    the quadratic ``(x - c)^T A^T A (x - c)`` with its minimum, zero, at
    ``center``. ``A`` defaults to the identity.
    """

    n_params: int
    center: np.ndarray
    matrix: Optional[np.ndarray] = None
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        if self.center.shape != (self.n_params,):
            raise ValueError(
                f"center has shape {self.center.shape}, expected ({self.n_params},)"
            )
        if self.matrix is None:
            self.matrix = np.eye(self.n_params)
        else:
            self.matrix = np.asarray(self.matrix, dtype=float)
            if self.matrix.ndim != 2 or self.matrix.shape[1] != self.n_params:
                raise ValueError(
                    f"matrix has shape {self.matrix.shape}, "
                    f"expected (m, {self.n_params})"
                )

    def residuals(self, x: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        return self.matrix @ (np.asarray(x, dtype=float) - self.center)

    def __call__(self, x: np.ndarray) -> float:
        r = self.residuals(x)
        return float(r @ r)


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


@dataclass
class PeriodicSinglePointEnergy:
    """vibe-qc periodic SCF energy per unit cell on one crystal, per call.

    The periodic sibling of :class:`SinglePointEnergy` (Stage 2 of the
    basis-optimisation stages, the one-compound bulk refit). Each call
    unpacks the candidate basis, writes it to the temp library, builds the
    ``BasisSet`` on the unit-cell molecule and runs
    :func:`vibeqc.periodic_runner.run_periodic_job` at ``kpoints``.

    The runner defaults follow the in-process ``VibeQcEngine``
    (:mod:`vibeqc.basis_optimization.calculators`): ``jk_method="bipole"``
    because a *candidate* basis has no density-fitting auxiliary set and
    borrowing one would add a fitting error on top of the basis signal,
    plus every artefact writer off. ``runner_kwargs`` overrides any of them
    (``conv_tol_energy``, ``max_iter``, ``fock_mixing``, ...).

    Parameters
    ----------
    parametrisation : BasisParametrisation
        Maps the optimiser vector to a candidate basis dict.
    system_factory : Callable[[], PeriodicSystem]
        Returns the crystal; called once per evaluation.
    library : TempBasisLibrary
        Already-entered context that owns ``$LIBINT_DATA_PATH``. Per-job
        output goes under its directory and is discarded with it.
    kpoints
        The k-mesh handed to the runner, e.g. ``(4, 4, 4)``.
    method, functional
        ``run_periodic_job`` reference selectors (``"RHF"``; a functional
        name switches to KS as the runner defines it).
    runner_kwargs : dict
        Extra keyword arguments for the runner, merged over the defaults.
    scf_runner
        Replacement for ``run_periodic_job`` (tests inject a mock).
    basis_factory : Callable[[system, basis_name, atoms], BasisSet]
        Replacement for ``vq.BasisSet(system.unit_cell_molecule(), name)``;
        receives the unpacked ``atoms`` dict as well so a test stub can
        carry the candidate parameters.
    on_failure : {"inf", "raise"}
        What a non-converged SCF or a runner exception becomes: ``+inf``
        (the forbidden-iterate convention every driver here understands)
        or a re-raised ``RuntimeError``.

    ``last_result`` keeps the runner's most recent result object, so a
    recipe can report ``n_iter`` or re-use the converged state.
    """

    parametrisation: "BasisParametrisation"
    system_factory: Callable[[], Any]
    library: Any
    kpoints: Any = (4, 4, 4)
    method: str = "RHF"
    functional: Optional[str] = None
    runner_kwargs: dict = field(default_factory=dict)
    scf_runner: Optional[Callable[..., Any]] = None
    basis_factory: Optional[Callable[[Any, str, dict], Any]] = None
    basis_name_prefix: str = "popt"
    on_failure: str = "inf"
    n_calls: int = field(default=0, init=False)
    last_result: Any = field(default=None, init=False)

    RUNNER_DEFAULTS: ClassVar[dict[str, Any]] = {
        "jk_method": "bipole",
        "output_qvf": False,
        "write_xyz_file": False,
        "write_xsf_structure_file": False,
        "write_cif_file": False,
        "progress": False,
    }

    def __post_init__(self) -> None:
        if self.on_failure not in ("inf", "raise"):
            raise ValueError(
                f"on_failure must be 'inf' or 'raise', got {self.on_failure!r}"
            )

    @property
    def n_params(self) -> int:
        return len(self.parametrisation)

    def runner_arguments(self) -> dict[str, Any]:
        """The keyword arguments one evaluation hands to the runner."""
        return {
            **self.RUNNER_DEFAULTS,
            "method": self.method,
            "functional": self.functional,
            "kpoints": self.kpoints,
            **self.runner_kwargs,
        }

    def _runner(self) -> Callable[..., Any]:
        if self.scf_runner is not None:
            return self.scf_runner
        from ..periodic_runner import run_periodic_job  # local: needs a build

        return run_periodic_job

    def _basis(self, system: Any, name: str, atoms: dict) -> Any:
        if self.basis_factory is not None:
            return self.basis_factory(system, name, atoms)
        from .. import BasisSet  # local: vibeqc not always importable at module load

        return BasisSet(system.unit_cell_molecule(), name)

    def _fail(self, what: str) -> float:
        if self.on_failure == "raise":
            raise RuntimeError(what)
        return float("inf")

    def __call__(self, x: np.ndarray) -> float:
        self.n_calls += 1
        atoms = self.parametrisation.unpack(np.asarray(x, dtype=float))
        name = f"{self.basis_name_prefix}-{self.n_calls:06d}"
        basis_name = self.library.write_g94(atoms, basis_name=name)
        system = self.system_factory()
        basis = self._basis(system, basis_name, atoms)
        out_dir = Path(self.library.path) / "periodic"
        out_dir.mkdir(parents=True, exist_ok=True)
        kwargs = self.runner_arguments()
        kwargs.setdefault("output", str(out_dir / name))
        try:
            result = self._runner()(system, basis, **kwargs)
        except Exception as exc:  # noqa: BLE001 - scored as forbidden, not fatal
            return self._fail(f"periodic SCF raised: {exc!r}")
        self.last_result = result
        if result is None or not bool(getattr(result, "converged", False)):
            return self._fail("periodic SCF did not converge")
        return float(result.energy)
