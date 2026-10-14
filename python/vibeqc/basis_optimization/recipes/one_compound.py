"""Stage-2 recipe: one-compound bulk refit of pob-TZVP H on LiH, in process.

The periodic counterpart of :mod:`.single_atom`: the same one free parameter
(pob-TZVP H's diffuse-most s exponent, log space, physical box), the same
perturbed start, but the objective is the periodic RHF energy per unit cell
of rocksalt LiH from :func:`vibeqc.periodic_runner.run_periodic_job`, through
:class:`~vibeqc.basis_optimization.objective.PeriodicSinglePointEnergy`.
Every optimiser driver of the package can drive it (``driver=``); the
periodic analytic gradient is not production yet (R13 in
``REQUIREMENTS-PERIODIC.md``), so BDIIS runs on central differences here.

What the design (``GOAL4_DESIGN.md``, Stage 2) expected of the result, and
what the first live run measured (vibe-qc 0.17.14, 2026-10-01, RHF, k-mesh
2x2x2, Py-BOBYQA, 13 evaluations; ``VERIFICATION_REPORT.md``):

* The design hypothesised an optimum *between* the free-atom value (about
  0.131, Stage 1) and the published multi-compound compromise 0.1795. The
  measurement landed at 0.189, slightly *above* the published value: the
  LiH crystal at Hartree-Fock wants a marginally tighter diffuse H function
  than the pob-TZVP calibration chose. The surface is flat, 0.36 mHa over a
  factor 1.5 in the exponent, so the window was a reasonable guess that
  the number simply does not support.
* The converged energy per formula unit, -8.0590005 Ha, is 3.8 mHa above
  the PT2013 supporting-information Hartree-Fock value, inside the "about
  10 mHa" the design expected for a different (CRYSTAL standard) basis.

Both are reported, neither is asserted: this is a smoke of the machinery.

Use::

    from vibeqc.basis_optimization.recipes.one_compound import run_lih_smoke
    res = run_lih_smoke(kpoints=(4, 4, 4), driver="bobyqa")
    print(res.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

LIH_A_ANGSTROM = 4.083
"""Rocksalt LiH lattice constant (Hasegawa, Yagi 2005; PT2013 reference 56),
the value in vibe-basis's structure table."""

PT2013_LIH_HF_PER_CELL_HA = -8.062837
"""Hartree-Fock LiH energy per formula unit quoted in ``GOAL4_DESIGN.md``
from the PT2013 supporting information (CRYSTAL standard basis). Reported
for orientation only; not an acceptance threshold."""

EXPECTED_EXPONENT_WINDOW = (0.13, 0.18)
"""Where ``GOAL4_DESIGN.md`` expected the LiH-refit diffuse H exponent to
land: between the free-atom optimum and the published pob-TZVP value. Kept as
the design's hypothesis; the first live run (see the module docstring) landed
just above it, at :data:`MEASURED_LIH_RHF_K222_EXPONENT`."""

MEASURED_LIH_RHF_K222_EXPONENT = 0.189
"""The diffuse H exponent the live Stage-2 smoke converged to (RHF, 2x2x2,
vibe-qc 0.17.14, 2026-10-01), bracketed to about plus or minus 0.002."""

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def lih_primitive_cell(a_angstrom: float = LIH_A_ANGSTROM) -> Any:
    """Rocksalt LiH as its two-atom FCC primitive cell (a ``vq.PeriodicSystem``).

    The conventional eight-atom cube of the structure table contains four
    formula units; the primitive cell gives the same energy per formula
    unit at a quarter of the cost, and vibe-qc's periodic path has handled
    the non-orthorhombic cell since R1 shipped.
    """
    import vibeqc as vq

    a = a_angstrom * ANGSTROM_TO_BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return vq.PeriodicSystem(
        3, lattice, [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(1, [a / 2.0] * 3)]
    )


def _pob_source(symbol_file: str) -> Path:
    """Locate a pob-TZVP source file, with the worktree fallback of single_atom."""
    import vibeqc as vq

    pkg_root = Path(vq.__file__).parent
    src = pkg_root / "basis_library" / "sources" / "pob-TZVP" / symbol_file
    if not src.exists():
        from .. import __file__ as bo_file

        src = (
            Path(bo_file).resolve().parents[3]
            / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP" / symbol_file
        )
    return src


def lih_pob_tzvp_parametrisation(bounds: tuple[float, float] = (0.05, 1.0)):
    """pob-TZVP Li and H with H's diffuse-most s exponent free (log space)."""
    from ...basis_crystal import parse_crystal_atom_basis_file
    from ..parametrise import BasisParametrisation, FreeSpec, Transform

    atoms = {
        "Li": parse_crystal_atom_basis_file(_pob_source("03_Li")),
        "H": parse_crystal_atom_basis_file(_pob_source("01_H")),
    }
    return BasisParametrisation(
        atoms=atoms,
        free=[FreeSpec(symbol="H", shell_idx=2, prim_idx=0, field="exponent",
                       transform=Transform.LOG, bounds=bounds, label="H_diffuse_s")],
    )


@dataclass
class OneCompoundResult:
    """Outcome of a one-compound bulk refit."""

    compound: str
    method: str
    driver: str
    kpoints: Any
    free_field: str
    starting_value: float
    optimal_value: float
    starting_energy: float
    optimal_energy: float
    n_evaluations: int
    wall_seconds: float
    converged: bool
    message: str
    reference_energy: Optional[float] = None
    citations: tuple = ()
    history: list = field(default_factory=list)

    @property
    def improvement_mha(self) -> float:
        return (self.starting_energy - self.optimal_energy) * 1000.0

    @property
    def in_expected_window(self) -> bool:
        lo, hi = EXPECTED_EXPONENT_WINDOW
        return lo < self.optimal_value < hi

    @property
    def reference_delta_mha(self) -> Optional[float]:
        if self.reference_energy is None:
            return None
        return (self.optimal_energy - self.reference_energy) * 1000.0

    def summary(self) -> str:
        delta_x_pct = (
            100.0 * (self.optimal_value - self.starting_value) / self.starting_value
            if self.starting_value else float("nan")
        )
        lines = [
            f"one-compound {self.compound} refit ({self.method}, k-mesh {self.kpoints}, "
            f"driver {self.driver})",
            f"  free:     {self.free_field}",
            f"  start:    x={self.starting_value:.6f}  E={self.starting_energy:.8f} Ha/cell",
            f"  optimum:  x={self.optimal_value:.6f}  E={self.optimal_energy:.8f} Ha/cell",
            f"  dx        {delta_x_pct:+7.2f} %   (design window "
            f"{EXPECTED_EXPONENT_WINDOW[0]:.2f} to {EXPECTED_EXPONENT_WINDOW[1]:.2f}: "
            f"{'inside' if self.in_expected_window else 'outside'}; "
            f"measured RHF/2x2x2 optimum {MEASURED_LIH_RHF_K222_EXPONENT:.3f})",
            f"  dE        {self.improvement_mha:+8.4f} mHa  (lower is better)",
        ]
        if self.reference_energy is not None:
            lines.append(
                f"  reference {self.reference_energy:.6f} Ha  "
                f"(delta {self.reference_delta_mha:+.3f} mHa; different basis, "
                "orientation only)"
            )
        lines += [
            f"  evals:    {self.n_evaluations}   wall: {self.wall_seconds:.1f} s",
            f"  converged: {self.converged}  ({self.message})",
        ]
        if self.citations:
            lines.append(f"  cite (method): {', '.join(self.citations)}")
        return "\n".join(lines)


def _drive(driver: str, objective, x0, bounds, *, max_evaluations: int,
           cmaes_seed: Optional[int]):
    """Run one of the package drivers on a gradient-free objective."""
    from ..bdiis import BDIIS_CITATION_KEYS, optimize_bdiis
    from ..drivers import (
        CMAES_CITATION_KEYS,
        DFO_CITATION_KEYS,
        optimize_bobyqa,
        optimize_cmaes,
        optimize_scipy,
    )

    if driver == "bobyqa":
        res = optimize_bobyqa(objective, x0, bounds=bounds,
                              max_evaluations=max_evaluations, rhoend=1e-5)
        return res, DFO_CITATION_KEYS
    if driver == "bdiis":
        # Central differences: the periodic analytic gradient is not
        # production (R13). A 1e-4 log-step is a 0.01 % exponent change,
        # well above the SCF convergence noise of the default tolerance.
        res = optimize_bdiis(objective, x0, bounds=bounds, max_iter=max_evaluations,
                             tol_energy=1e-7, tol_grad=1e-4, fd_step=1e-4)
        return res, BDIIS_CITATION_KEYS
    if driver == "scipy":
        res = optimize_scipy(objective, x0, bounds=bounds, tol=1e-7,
                             max_iter=max_evaluations)
        return res, ()
    if driver == "cmaes":
        res = optimize_cmaes(objective, x0, bounds=bounds, seed=cmaes_seed,
                             max_evaluations=max_evaluations, tol_fun=1e-7)
        return res, CMAES_CITATION_KEYS
    if driver == "cmaes+bdiis":
        from .global_then_polish import (
            GLOBAL_THEN_POLISH_CITATION_KEYS,
            optimize_global_then_polish,
        )

        res = optimize_global_then_polish(
            objective, x0, bounds=bounds,
            cmaes_max_evaluations=max(8, max_evaluations // 2), cmaes_seed=cmaes_seed,
            cmaes_tol_fun=1e-6, bdiis_max_iter=max_evaluations,
            bdiis_tol_energy=1e-7, bdiis_tol_grad=1e-4,
            bdiis_kwargs={"fd_step": 1e-4},
        )
        return res, GLOBAL_THEN_POLISH_CITATION_KEYS
    raise ValueError(
        f"unknown driver {driver!r}; choose bobyqa, bdiis, scipy, cmaes or cmaes+bdiis"
    )


def run_one_compound(
    parametrisation,
    system_factory: Callable[[], Any],
    *,
    compound: str = "crystal",
    kpoints: Any = (4, 4, 4),
    method: str = "RHF",
    functional: Optional[str] = None,
    driver: str = "bobyqa",
    perturbation: float = 1.0,
    max_evaluations: int = 40,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    runner_kwargs: Optional[dict] = None,
    scf_runner: Optional[Callable[..., Any]] = None,
    basis_factory: Optional[Callable[..., Any]] = None,
    reference_energy: Optional[float] = None,
    cmaes_seed: Optional[int] = 1,
) -> OneCompoundResult:
    """Refit the free parameters of ``parametrisation`` on one crystal.

    ``perturbation`` multiplies every free *physical* value at the start
    (``1.5`` starts the exponents 50 percent too tight, as the Stage-1 smoke
    does). With ``use_cond_penalty`` the objective is the OPTBASIS shape
    ``E + gamma ln kappa(S)`` over the parametrised elements' atomic
    overlaps. ``scf_runner`` / ``basis_factory`` are the test seams of
    :class:`PeriodicSinglePointEnergy`.
    """
    from ..io import TempBasisLibrary
    from ..ld_diagnostics import cond_penalty_from_atom
    from ..objective import PeriodicSinglePointEnergy

    if len(parametrisation) != 1:
        raise ValueError(
            "run_one_compound reports one free parameter; got "
            f"{len(parametrisation)} (use run_one_compound_multi for several)"
        )
    spec = parametrisation.free[0]
    starting = spec.transform.from_optim(float(parametrisation.pack()[0]))

    with TempBasisLibrary() as lib:
        energy = PeriodicSinglePointEnergy(
            parametrisation=parametrisation,
            system_factory=system_factory,
            library=lib,
            kpoints=kpoints,
            method=method,
            functional=functional,
            runner_kwargs=dict(runner_kwargs or {}),
            scf_runner=scf_runner,
            basis_factory=basis_factory,
            basis_name_prefix=f"{compound}-opt",
        )

        class _Objective:
            n_params = 1

            def __init__(self) -> None:
                self.n_calls = 0

            def __call__(self, x: np.ndarray) -> float:
                self.n_calls += 1
                e = energy(x)
                if not np.isfinite(e) or not use_cond_penalty:
                    return e
                atoms = parametrisation.unpack(np.asarray(x, dtype=float))
                total = e
                for atom in atoms.values():
                    try:
                        total += cond_penalty_from_atom(atom, gamma=cond_gamma)
                    except Exception:  # noqa: BLE001 - degenerate overlap
                        return float("inf")
                return total

        objective = _Objective()
        x0 = parametrisation.pack() + np.log(perturbation) if (
            getattr(spec.transform, "name", "") == "LOG"
        ) else parametrisation.pack() * perturbation
        bounds = parametrisation.optim_bounds()
        t0 = time.perf_counter()
        e_start = objective(x0)
        res, citations = _drive(driver, objective, x0, bounds,
                                max_evaluations=max_evaluations, cmaes_seed=cmaes_seed)
        wall = time.perf_counter() - t0

    optimal = spec.transform.from_optim(float(res.x[0]))
    return OneCompoundResult(
        compound=compound,
        method=method if functional is None else f"{method}/{functional}",
        driver=driver,
        kpoints=kpoints,
        free_field=spec.display(),
        starting_value=starting * perturbation,
        optimal_value=optimal,
        starting_energy=float(e_start),
        optimal_energy=float(res.fun),
        n_evaluations=int(objective.n_calls),
        wall_seconds=wall,
        converged=bool(res.success),
        message=str(res.message),
        reference_energy=reference_energy,
        citations=tuple(citations),
        history=list(res.history),
    )


def run_lih_smoke(
    *,
    perturbation: float = 1.5,
    bounds: tuple[float, float] = (0.05, 1.0),
    kpoints: Any = (4, 4, 4),
    method: str = "RHF",
    driver: str = "bobyqa",
    max_evaluations: int = 40,
    use_cond_penalty: bool = False,
    cond_gamma: float = 1e-3,
    runner_kwargs: Optional[dict] = None,
    a_angstrom: float = LIH_A_ANGSTROM,
    scf_runner: Optional[Callable[..., Any]] = None,
    basis_factory: Optional[Callable[..., Any]] = None,
    cmaes_seed: Optional[int] = 1,
) -> OneCompoundResult:
    """Stage-2 smoke: refit pob-TZVP H's diffuse s exponent on rocksalt LiH.

    Starts at ``perturbation`` times the published 0.1795 and minimises the
    periodic ``method`` energy per primitive cell at ``kpoints`` with the
    chosen ``driver``. See the module docstring for what to expect.
    """
    return run_one_compound(
        lih_pob_tzvp_parametrisation(bounds),
        lambda: lih_primitive_cell(a_angstrom),
        compound="LiH",
        kpoints=kpoints,
        method=method,
        driver=driver,
        perturbation=perturbation,
        max_evaluations=max_evaluations,
        use_cond_penalty=use_cond_penalty,
        cond_gamma=cond_gamma,
        runner_kwargs=runner_kwargs,
        scf_runner=scf_runner,
        basis_factory=basis_factory,
        reference_energy=PT2013_LIH_HF_PER_CELL_HA if method.upper() == "RHF" else None,
        cmaes_seed=cmaes_seed,
    )


if __name__ == "__main__":  # pragma: no cover
    print(run_lih_smoke().summary())
