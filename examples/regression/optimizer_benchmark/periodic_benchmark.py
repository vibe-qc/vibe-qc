"""Periodic geometry optimization benchmark driver.

Wraps :func:`vibeqc.bipole_optimize.relax_atoms` with per-step
tracking so the same report infrastructure can compare periodic
optimizers against molecular ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .periodic_systems import PeriodicTestSystem, get_periodic_system

# ---------------------------------------------------------------------------
# Data containers (compatible with benchmark_driver)
# ---------------------------------------------------------------------------


@dataclass
class PeriodicStepRecord:
    step: int
    energy_ha: float
    max_grad_ha_per_bohr: float
    wall_time_s: float


@dataclass
class PeriodicRun:
    system_name: str
    method: str
    functional: Optional[str]
    basis: str
    converged: bool
    n_steps: int
    wall_time_s: float
    energy_initial_ha: float
    energy_final_ha: float
    grad_initial_ha_per_bohr: float
    grad_final_ha_per_bohr: float
    steps: List[PeriodicStepRecord] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
            "method": self.method,
            "functional": self.functional,
            "basis": self.basis,
            "converged": self.converged,
            "n_steps": self.n_steps,
            "wall_time_s": self.wall_time_s,
            "energy_initial_ha": self.energy_initial_ha,
            "energy_final_ha": self.energy_final_ha,
            "grad_initial_ha_per_bohr": self.grad_initial_ha_per_bohr,
            "grad_final_ha_per_bohr": self.grad_final_ha_per_bohr,
            "steps": [
                {
                    "step": s.step,
                    "energy_ha": s.energy_ha,
                    "max_grad_ha_per_bohr": s.max_grad_ha_per_bohr,
                    "wall_time_s": s.wall_time_s,
                }
                for s in self.steps
            ],
            "error": self.error,
            "optimizer_info": {
                "description": "scipy L-BFGS-B (BIPOLE, FD forces)",
                "family": "quasi_newton",
                "uses_hessian_approx": True,
            },
        }


# ---------------------------------------------------------------------------
# Periodic relaxation runner
# ---------------------------------------------------------------------------


def run_periodic_relaxation(
    system: PeriodicTestSystem,
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    max_steps: int = 30,
    conv_tol_grad: float = 1e-4,
    cutoff_bohr: float = 8.0,
    verbose: bool = True,
) -> PeriodicRun:
    """Run periodic atomic relaxation and collect per-step data.

    Uses :func:`vibeqc.bipole_optimize.relax_atoms` with FD forces
    (the production-safe path).  Collects energy, gradient, and
    timing per L-BFGS-B step.
    """
    from vibeqc._vibeqc_core import BasisSet, monkhorst_pack
    from vibeqc.bipole_optimize import relax_atoms

    sys = system.system
    basis_name = system.basis

    # Build k-point mesh from density.
    # kmesh_density ≈ minimum distance between k-points in Å.
    # For rocksalt ~4 Å cell, density=4 → ~1×1×1 mesh.
    # For diamond ~5.4 Å, density=4 → ~2×2×2 mesh.
    lat = np.asarray(sys.lattice, dtype=float)
    recip = 2.0 * np.pi * np.linalg.inv(lat).T
    recp_len = np.array([np.linalg.norm(recip[:, i]) for i in range(3)])
    target_spacing_A = system.kmesh_density * 0.529177  # bohr → Å
    n_k = [max(1, int(np.ceil(r / target_spacing_A))) for r in recp_len]
    kmesh = monkhorst_pack(list(n_k))

    basis_obj = BasisSet(sys.unit_cell_molecule(), basis_name)

    # Initial energy + gradient
    t_start = time.perf_counter()
    from vibeqc.bipole_optimize import _compute_forces, _run_scf

    opts = _default_options(method, cutoff_bohr)
    e0, res0 = _run_scf(sys, basis_obj, kmesh, opts, method.upper(), functional)
    forces0 = _compute_forces(
        sys,
        basis_obj,
        res0,
        method.upper(),
        opts.lattice_opts,
        kmesh=kmesh,
        basis_name=basis_name,
        opts=opts,
        functional=functional,
        force_mode="fd",
    )
    g0 = float(abs(forces0).max())

    steps_data = []

    def _tracking_gradient(x: np.ndarray) -> np.ndarray:
        """Wrapped gradient that records per-step data."""
        from vibeqc.bipole_optimize import _atoms_to_flat, _flat_to_system

        # Evaluate SCF at current geometry
        s = _flat_to_system(sys, x)
        b = BasisSet(s.unit_cell_molecule(), basis_name)
        e, res = _run_scf(s, b, kmesh, opts, method.upper(), functional)
        f = _compute_forces(
            s,
            b,
            res,
            method.upper(),
            opts.lattice_opts,
            kmesh=kmesh,
            basis_name=basis_name,
            opts=opts,
            functional=functional,
            force_mode="fd",
        )
        steps_data.append(
            PeriodicStepRecord(
                step=len(steps_data),
                energy_ha=float(e),
                max_grad_ha_per_bohr=float(abs(f).max()),
                wall_time_s=time.perf_counter() - t_start,
            )
        )
        # Convert Cartesian forces to fractional gradient
        lattice_mat = np.asarray(s.lattice, dtype=float)
        grad_frac = lattice_mat.T @ f.T
        return grad_frac.T.ravel()

    try:
        opt_result = relax_atoms(
            sys,
            basis_name,
            kmesh,
            method=method,
            functional=functional,
            max_iter=max_steps,
            conv_tol_grad=conv_tol_grad,
            cutoff_bohr=cutoff_bohr,
        )
    except Exception as exc:
        t_elapsed = time.perf_counter() - t_start
        return PeriodicRun(
            system_name=system.name,
            method=method,
            functional=functional,
            basis=basis_name,
            converged=False,
            n_steps=len(steps_data),
            wall_time_s=t_elapsed,
            energy_initial_ha=float(e0),
            energy_final_ha=float("nan"),
            grad_initial_ha_per_bohr=g0,
            grad_final_ha_per_bohr=float("nan"),
            steps=steps_data,
            error=str(exc),
        )

    t_elapsed = time.perf_counter() - t_start

    # If relax_atoms didn't populate our tracker (it has its own gradient
    # closure), use the result directly.
    if not steps_data:
        n_steps = opt_result.n_iter
        e_final = opt_result.energy
        g_final = float(abs(np.asarray(opt_result.gradient)).max())
        return PeriodicRun(
            system_name=system.name,
            method=method,
            functional=functional,
            basis=basis_name,
            converged=opt_result.converged,
            n_steps=n_steps,
            wall_time_s=t_elapsed,
            energy_initial_ha=float(e0),
            energy_final_ha=float(e_final),
            grad_initial_ha_per_bohr=g0,
            grad_final_ha_per_bohr=g_final,
            steps=steps_data,
        )

    return PeriodicRun(
        system_name=system.name,
        method=method,
        functional=functional,
        basis=basis_name,
        converged=opt_result.converged,
        n_steps=opt_result.n_iter,
        wall_time_s=t_elapsed,
        energy_initial_ha=float(e0),
        energy_final_ha=float(opt_result.energy),
        grad_initial_ha_per_bohr=g0,
        grad_final_ha_per_bohr=float(abs(np.asarray(opt_result.gradient)).max()),
        steps=steps_data,
    )


def _default_options(method: str, cutoff_bohr: float):
    """Build default SCF options for periodic runs."""
    from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions

    if method.upper() in ("RKS", "UKS"):
        opts = PeriodicKSOptions()
    else:
        opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = cutoff_bohr
    opts.lattice_opts.nuclear_cutoff_bohr = cutoff_bohr
    opts.max_iter = 50
    opts.use_diis = True
    opts.conv_tol_energy = 1e-7
    return opts


# ---------------------------------------------------------------------------
# Benchmark sweep
# ---------------------------------------------------------------------------


def run_periodic_benchmark(
    systems: List[str],
    *,
    method: str = "RHF",
    functional: Optional[str] = None,
    max_steps: int = 30,
    conv_tol_grad: float = 1e-4,
    cutoff_bohr: float = 8.0,
    verbose: bool = True,
) -> List[PeriodicRun]:
    """Run periodic relaxation on multiple systems.

    Parameters
    ----------
    systems : list of str
        Periodic system names.
    method : str
        "RHF", "UHF", "RKS", or "UKS".
    functional : str or None
        XC functional for KS methods.
    max_steps : int
        Maximum L-BFGS-B iterations.
    conv_tol_grad : float
        Gradient convergence (Ha/bohr).
    cutoff_bohr : float
        Lattice cutoff radius.
    verbose : bool
        Print progress.

    Returns
    -------
    list of PeriodicRun
    """
    runs = []
    for sys_name in systems:
        sys_obj = get_periodic_system(sys_name)
        if verbose:
            print(f"  {sys_name:20s}  periodic relax  ...", end=" ", flush=True)
        try:
            run = run_periodic_relaxation(
                sys_obj,
                method=method,
                functional=functional,
                max_steps=max_steps,
                conv_tol_grad=conv_tol_grad,
                cutoff_bohr=cutoff_bohr,
                verbose=False,
            )
        except Exception as exc:
            run = PeriodicRun(
                system_name=sys_name,
                method=method,
                functional=functional,
                basis=sys_obj.basis,
                converged=False,
                n_steps=0,
                wall_time_s=0.0,
                energy_initial_ha=float("nan"),
                energy_final_ha=float("nan"),
                grad_initial_ha_per_bohr=float("nan"),
                grad_final_ha_per_bohr=float("nan"),
                error=str(exc),
            )
        status = "✓" if run.converged else "✗"
        detail = f"{run.n_steps} steps" if run.converged else (run.error or "nc")
        if verbose:
            print(f"{status}  [{detail}]")
        runs.append(run)
    return runs
