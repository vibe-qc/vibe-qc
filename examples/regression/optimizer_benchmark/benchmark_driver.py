"""Core benchmark driver — runs each optimizer on each test system and
collects structured results.

Architecture
------------
Each optimizer/system pair runs in its own subprocess-friendly scope.
The driver creates an ASE ``Atoms`` with a ``VibeQC`` calculator (for
the ASE optimizers) or uses ``vibeqc.molecular_optimize.optimize_molecule``
(for the native L-BFGS-B).  Per-step data is collected via callback.

Results are JSON-serializable and designed for post-hoc analysis by
``report.py``.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from vibeqc import Atom, Molecule

from .test_systems import TestSystem, get_system, list_systems

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One step of a geometry optimization."""

    step: int
    energy_ha: float
    max_grad_ha_per_bohr: float
    wall_time_s: float
    n_scf_iterations: int = 0
    scf_converged: bool = True


@dataclass
class OptimizerRun:
    """Result of one optimizer on one system."""

    system_name: str
    optimizer_name: str
    method: str
    functional: Optional[str]
    basis: str
    converged: bool
    n_steps: int
    n_scf_evals: int
    wall_time_s: float
    energy_initial_ha: float
    energy_final_ha: float
    grad_initial_ha_per_bohr: float
    grad_final_ha_per_bohr: float
    steps: List[StepRecord] = field(default_factory=list)
    error: Optional[str] = None
    optimizer_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


@dataclass
class BenchmarkResult:
    """Full benchmark run."""

    config: Dict[str, Any]
    runs: List[OptimizerRun]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "timestamp": self.timestamp,
            "runs": [r.to_dict() for r in self.runs],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkResult":
        with open(path) as f:
            data = json.load(f)
        runs = [
            OptimizerRun(
                system_name=r["system_name"],
                optimizer_name=r["optimizer_name"],
                method=r["method"],
                functional=r.get("functional"),
                basis=r["basis"],
                converged=r["converged"],
                n_steps=r["n_steps"],
                n_scf_evals=r["n_scf_evals"],
                wall_time_s=r["wall_time_s"],
                energy_initial_ha=r["energy_initial_ha"],
                energy_final_ha=r["energy_final_ha"],
                grad_initial_ha_per_bohr=r["grad_initial_ha_per_bohr"],
                grad_final_ha_per_bohr=r["grad_final_ha_per_bohr"],
                steps=[StepRecord(**s) for s in r.get("steps", [])],
                error=r.get("error"),
                optimizer_info=r.get("optimizer_info", {}),
            )
            for r in data["runs"]
        ]
        return cls(config=data["config"], runs=runs, timestamp=data["timestamp"])


# ---------------------------------------------------------------------------
# ASE gradient norm extraction
# ---------------------------------------------------------------------------


def _get_max_force_ase(atoms) -> float:
    """Return the max Cartesian force component (eV/Å) from ASE atoms.

    ASE stores ``forces`` in eV/Å.  We want Ha/bohr for consistency
    with vibe-qc's own units.  Returns ``inf`` if forces aren't set.
    """
    try:
        forces = atoms.get_forces()
        return float(abs(forces).max())
    except Exception:
        return float("inf")


def _ase_force_to_ha_per_bohr(f_eva: float) -> float:
    """Convert eV/Å → Ha/bohr."""
    from ase.units import Bohr, Hartree

    return f_eva * Bohr / Hartree


# ---------------------------------------------------------------------------
# ASE optimizer runner
# ---------------------------------------------------------------------------


# Map canonical names to ASE optimizer classes + metadata.
ASE_OPTIMIZERS: Dict[str, dict] = {}


def _register_ase_optimizers():
    """Populate ASE_OPTIMIZERS with available optimizers.

    Mutates the module-level dict *in-place* so that ``from module import
    ASE_OPTIMIZERS`` references stay wired to the populated dict.
    """
    if ASE_OPTIMIZERS:
        return
    try:
        from ase.optimize import (
            BFGS,
            FIRE,
            LBFGS,
            BFGSLineSearch,
            GoodOldQuasiNewton,
            GPMin,
            LBFGSLineSearch,
            MDMin,
            ODE12r,
        )
        from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
    except ImportError:
        return

    ASE_OPTIMIZERS.update(
        {
            "BFGS": {
                "cls": BFGS,
                "description": "Quasi-Newton BFGS (no line search)",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
            "BFGSLineSearch": {
                "cls": BFGSLineSearch,
                "description": "Quasi-Newton BFGS with line search",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
            "LBFGS": {
                "cls": LBFGS,
                "description": "Limited-memory BFGS",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
            "LBFGSLineSearch": {
                "cls": LBFGSLineSearch,
                "description": "Limited-memory BFGS with line search",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
            "FIRE": {
                "cls": FIRE,
                "description": "Fast Inertial Relaxation Engine",
                "family": "inertial",
                "uses_hessian_approx": False,
            },
            "GPMin": {
                "cls": GPMin,
                "description": "Gaussian Process minimizer",
                "family": "bayesian",
                "uses_hessian_approx": True,  # learns surrogate
            },
            "MDMin": {
                "cls": MDMin,
                "description": "Velocity-Verlet MD with damping",
                "family": "inertial",
                "uses_hessian_approx": False,
            },
            "ODE12r": {
                "cls": ODE12r,
                "description": "Adaptive ODE-based optimizer",
                "family": "ode",
                "uses_hessian_approx": False,
            },
            "GoodOldQuasiNewton": {
                "cls": GoodOldQuasiNewton,
                "description": "Legacy ASE quasi-Newton",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
            "SciPyFminBFGS": {
                "cls": SciPyFminBFGS,
                "description": "SciPy BFGS (wrapped by ASE)",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
            "SciPyFminCG": {
                "cls": SciPyFminCG,
                "description": "SciPy conjugate gradient (wrapped by ASE)",
                "family": "cg",
                "uses_hessian_approx": False,
            },
        }
    )


def _build_ase_atoms(system: TestSystem, config: dict) -> "Any":
    """Build ASE Atoms with a VibeQC calculator attached.

    Handles semiempirical methods by routing through
    _make_semiempirical_ase_calculator (same code path as run_job's
    _optimize_geometry).
    """
    from ase import Atoms
    from ase.units import Bohr

    positions_ang = [
        [coord * Bohr for coord in atom.xyz] for atom in system.molecule.atoms
    ]
    atoms = Atoms(
        numbers=[atom.Z for atom in system.molecule.atoms],
        positions=positions_ang,
    )

    method = config["method"]
    _se_methods = {
        "dftb0",
        "scc_dftb",
        "pm6",
        "gfn2_xtb",
        "om1",
        "om2",
        "om3",
        "msindo",
    }

    if method in _se_methods:
        from vibeqc.runner import _make_semiempirical_ase_calculator

        atoms.calc = _make_semiempirical_ase_calculator(
            system.molecule,
            method,
        )
    elif method == "mace":
        from vibeqc.mlip.mace import MACEModel

        atoms.calc = MACEModel(system.molecule, config.get("mlip_options")).calculator
    else:
        from vibeqc.ase import VibeQC

        atoms.calc = VibeQC(
            basis=config["basis"],
            charge=system.charge,
            multiplicity=system.multiplicity,
            functional=config.get("functional"),
        )
    return atoms


def _run_ase_optimizer(
    system: TestSystem,
    optimizer_name: str,
    config: dict,
) -> OptimizerRun:
    """Run one ASE optimizer on one system, collecting per-step data."""
    _register_ase_optimizers()
    opt_info = ASE_OPTIMIZERS[optimizer_name]
    OptClass = opt_info["cls"]

    fmax_eva = config["fmax_eva"]
    max_steps = config["max_steps"]
    basis = config.get("basis", "sto-3g")
    method = config["method"]
    functional = config.get("functional")

    atoms = _build_ase_atoms(system, config)
    steps_data: List[StepRecord] = []
    t_start = time.perf_counter()
    n_scf_evals = 0

    # Get initial energy + gradient
    try:
        e0 = atoms.get_potential_energy()
        f0_max_eva = _get_max_force_ase(atoms)
    except Exception as exc:
        return OptimizerRun(
            system_name=system.name,
            optimizer_name=optimizer_name,
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            n_scf_evals=0,
            wall_time_s=time.perf_counter() - t_start,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_initial_ha_per_bohr=float("nan"),
            grad_final_ha_per_bohr=float("nan"),
            error=f"Initial SCF failed: {exc}",
            optimizer_info=opt_info,
        )

    g0_ha_per_bohr = _ase_force_to_ha_per_bohr(f0_max_eva)

    # Pre-compute initial energy in Ha for recording.
    from ase.units import Hartree

    e0_ha = e0 / Hartree

    def observer():
        """ASE callback — record per-step data."""
        nonlocal n_scf_evals
        step = len(steps_data)
        t_now = time.perf_counter()
        try:
            e_ev = atoms.get_potential_energy()
            f_eva = _get_max_force_ase(atoms)
        except Exception:
            e_ev = float("nan")
            f_eva = float("inf")
        steps_data.append(
            StepRecord(
                step=step,
                energy_ha=e_ev / Hartree if e_ev == e_ev else float("nan"),
                max_grad_ha_per_bohr=_ase_force_to_ha_per_bohr(f_eva),
                wall_time_s=t_now - t_start,
                n_scf_iterations=0,
                scf_converged=True,
            )
        )
        n_scf_evals += 1

    try:
        opt = OptClass(atoms, logfile=None)
        opt.attach(observer)
        opt.run(fmax=fmax_eva, steps=max_steps)
    except Exception as exc:
        # Partial data — optimizer crashed mid-run.
        pass

    t_elapsed = time.perf_counter() - t_start

    # Determine convergence from final gradient.
    try:
        f_final_eva = _get_max_force_ase(atoms)
        e_final_ev = atoms.get_potential_energy()
        e_final_ha = e_final_ev / Hartree
        g_final_ha = _ase_force_to_ha_per_bohr(f_final_eva)
        converged = f_final_eva <= fmax_eva
    except Exception:
        e_final_ha = float("nan")
        g_final_ha = float("nan")
        converged = False

    return OptimizerRun(
        system_name=system.name,
        optimizer_name=optimizer_name,
        method=method,
        functional=functional,
        basis=basis,
        converged=converged,
        n_steps=len(steps_data),
        n_scf_evals=n_scf_evals,
        wall_time_s=t_elapsed,
        energy_initial_ha=e0_ha,
        energy_final_ha=e_final_ha,
        grad_initial_ha_per_bohr=g0_ha_per_bohr,
        grad_final_ha_per_bohr=g_final_ha,
        steps=steps_data,
        optimizer_info=opt_info,
    )


# ---------------------------------------------------------------------------
# Native (scipy L-BFGS-B) runner
# ---------------------------------------------------------------------------


def _run_native_optimizer(
    system: TestSystem,
    config: dict,
) -> OptimizerRun:
    """Run vibe-qc's native scipy L-BFGS-B optimizer."""
    from vibeqc.molecular_optimize import optimize_molecule

    basis = config.get("basis", "sto-3g")
    method = config["method"]
    functional = config.get("functional")
    fmax_eva = config["fmax_eva"]
    max_steps = config["max_steps"]

    # Convert fmax from eV/Å to Ha/bohr.
    # 1 eV/Å = Bohr / Hartree = 0.529177 / 27.2114 ≈ 0.01945 Ha/bohr
    from ase.units import Bohr, Hartree

    _conv_grad = fmax_eva * (Bohr / Hartree)

    mol = system.molecule

    # Get initial energy + gradient for consistency.
    from vibeqc._vibeqc_core import (
        BasisSet,
        RHFOptions,
        RKSOptions,
        UHFOptions,
        UKSOptions,
        compute_gradient,
        compute_gradient_rks,
        compute_gradient_uhf,
        compute_gradient_uks,
        run_rhf,
        run_rks,
        run_uhf,
        run_uks,
    )

    t_start = time.perf_counter()
    basis_obj = BasisSet(mol, basis)

    try:
        if method == "rhf":
            opts = RHFOptions()
            res = run_rhf(mol, basis_obj, opts)
            grad0 = compute_gradient(mol, basis_obj, res)
        elif method == "uhf":
            opts = UHFOptions()
            res = run_uhf(mol, basis_obj, opts)
            grad0 = compute_gradient_uhf(mol, basis_obj, res)
        elif method == "rks":
            opts = RKSOptions()
            opts.functional = functional
            res = run_rks(mol, basis_obj, opts)
            grad0 = compute_gradient_rks(mol, basis_obj, res, opts.grid)
        elif method == "uks":
            opts = UKSOptions()
            opts.functional = functional
            res = run_uks(mol, basis_obj, opts)
            grad0 = compute_gradient_uks(mol, basis_obj, res, opts.grid)
        else:
            raise ValueError(f"Unsupported method for native: {method}")
        e0_ha = res.energy
        g0_max = float(abs(grad0).max()) if grad0.size else float("inf")
    except Exception as exc:
        t_elapsed = time.perf_counter() - t_start
        return OptimizerRun(
            system_name=system.name,
            optimizer_name="native",
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            n_scf_evals=0,
            wall_time_s=t_elapsed,
            energy_initial_ha=float("nan"),
            energy_final_ha=float("nan"),
            grad_initial_ha_per_bohr=float("nan"),
            grad_final_ha_per_bohr=float("nan"),
            error=f"Initial SCF failed: {exc}",
            optimizer_info={
                "description": "scipy L-BFGS-B (vibe-qc native)",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
        )

    try:
        opt_result = optimize_molecule(
            mol,
            basis,
            method=method,
            functional=functional,
            max_iter=max_steps,
            conv_tol_grad=_conv_grad,
            record_trajectory=True,
            progress=False,
        )
    except Exception as exc:
        t_elapsed = time.perf_counter() - t_start
        return OptimizerRun(
            system_name=system.name,
            optimizer_name="native",
            method=method,
            functional=functional,
            basis=basis,
            converged=False,
            n_steps=0,
            n_scf_evals=0,
            wall_time_s=t_elapsed,
            energy_initial_ha=e0_ha,
            energy_final_ha=float("nan"),
            grad_initial_ha_per_bohr=g0_max,
            grad_final_ha_per_bohr=float("nan"),
            error=f"Optimization failed: {exc}",
            optimizer_info={
                "description": "scipy L-BFGS-B (vibe-qc native)",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
        )

    t_elapsed = time.perf_counter() - t_start
    steps_data = []
    for i, (frame, e_traj) in enumerate(
        zip(opt_result.trajectory_frames, opt_result.trajectory_energies)
    ):
        steps_data.append(
            StepRecord(
                step=i,
                energy_ha=float(e_traj),
                max_grad_ha_per_bohr=0.0,  # not tracked per-step
                wall_time_s=t_elapsed
                * (i + 1)
                / max(len(opt_result.trajectory_frames), 1),
            )
        )

    grad_final = opt_result.gradient
    g_final = float(abs(grad_final).max()) if grad_final.size else float("inf")

    return OptimizerRun(
        system_name=system.name,
        optimizer_name="native",
        method=method,
        functional=functional,
        basis=basis,
        converged=opt_result.converged,
        n_steps=opt_result.n_iter,
        n_scf_evals=opt_result.n_iter * 2,  # fun + jac each iter (cached)
        wall_time_s=t_elapsed,
        energy_initial_ha=e0_ha,
        energy_final_ha=opt_result.energy,
        grad_initial_ha_per_bohr=g0_max,
        grad_final_ha_per_bohr=g_final,
        steps=steps_data,
        optimizer_info={
            "description": "scipy L-BFGS-B (vibe-qc native)",
            "family": "quasi_newton",
            "uses_hessian_approx": True,
        },
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(
    systems: List[str],
    optimizers: List[str],
    *,
    method: str = "rhf",
    functional: Optional[str] = None,
    basis: str = "sto-3g",
    fmax: float = 0.01,
    max_steps: int = 200,
    verbose: bool = True,
    system_getter: Any = None,
) -> BenchmarkResult:
    """Run all optimizer/system combinations.

    Parameters
    ----------
    systems : list of str
        System names from ``test_systems.list_systems()``.
    optimizers : list of str
        Optimizer names. ``"native"`` selects the scipy L-BFGS-B path;
        all others must be keys in ``ASE_OPTIMIZERS``.
    method : str
        SCF method (``"rhf"``, ``"rks"``, …).
    functional : str or None
        XC functional for KS methods.
    basis : str
        Basis-set name.
    fmax : float
        Convergence threshold for max force component (eV/Å).
    max_steps : int
        Maximum optimization steps.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    BenchmarkResult
    """
    _register_ase_optimizers()

    # Validate optimizers
    for opt_name in optimizers:
        if opt_name == "native":
            continue
        if opt_name not in ASE_OPTIMIZERS:
            available = sorted(ASE_OPTIMIZERS.keys())
            raise ValueError(
                f"Unknown optimizer {opt_name!r}. "
                f"Available ASE optimizers: {available}. "
                f"Also available: 'native'."
            )

    # Validate systems
    for sys_name in systems:
        if sys_name in list_systems():
            continue
        if system_getter is not None:
            try:
                system_getter(sys_name)
                continue
            except Exception:
                pass
        available = list_systems()
        raise ValueError(f"Unknown system {sys_name!r}. Available: {available}")

    config = {
        "method": method,
        "functional": functional,
        "basis": basis,
        "fmax_eva": fmax,
        "max_steps": max_steps,
    }

    runs: List[OptimizerRun] = []

    for sys_name in systems:
        if sys_name in list_systems():
            system = get_system(sys_name)
        elif system_getter is not None:
            system = system_getter(sys_name)
        else:
            raise ValueError(f"Unknown system: {sys_name!r}")
        for opt_name in optimizers:
            if verbose:
                print(f"  {sys_name:20s}  {opt_name:20s}  ...", end=" ", flush=True)
            try:
                if opt_name == "native":
                    run = _run_native_optimizer(system, config)
                else:
                    run = _run_ase_optimizer(system, opt_name, config)
            except Exception as exc:
                run = OptimizerRun(
                    system_name=sys_name,
                    optimizer_name=opt_name,
                    method=method,
                    functional=functional,
                    basis=basis,
                    converged=False,
                    n_steps=0,
                    n_scf_evals=0,
                    wall_time_s=0.0,
                    energy_initial_ha=float("nan"),
                    energy_final_ha=float("nan"),
                    grad_initial_ha_per_bohr=float("nan"),
                    grad_final_ha_per_bohr=float("nan"),
                    error=str(exc),
                )
            if verbose:
                status = "✓" if run.converged else "✗"
                detail = ""
                if not run.converged and run.error:
                    detail = f"  [{run.error[:60]}]"
                elif not run.converged:
                    detail = (
                        f"  [{run.n_steps} steps, g={run.grad_final_ha_per_bohr:.2e}]"
                    )
                else:
                    detail = f"  [{run.n_steps} steps, {run.wall_time_s:.1f}s]"
                print(f"{status}{detail}")

            runs.append(run)

    return BenchmarkResult(config=config, runs=runs)
