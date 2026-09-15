"""Preoptimization workflow -- semiempirical -> ab initio handoff.

Stage 8: Use semiempirical methods as a fast preoptimizer before
HF/DFT refinement.
"""

from __future__ import annotations

import numpy as np

from vibeqc import Molecule
from vibeqc._vibeqc_core import Atom, PeriodicSystem
from vibeqc.semiempirical.runner import (
    normalise_semiempirical_method,
    run_semiempirical,
)


def _molecular_preopt_energy_gradient(
    mol: Molecule,
    method_key: str,
) -> tuple[float, np.ndarray]:
    """Evaluate the unified molecular semiempirical runner for ASE preopt."""
    result = run_semiempirical(method_key, mol)
    gradient = result.gradient()
    if gradient is None:
        raise RuntimeError(
            f"molecular semiempirical preoptimization requires a gradient for "
            f"{method_key!r}"
        )
    return float(result.energy), np.asarray(gradient, dtype=float)


def preoptimize_molecule(
    mol: Molecule,
    *,
    method: str = "scc_dftb",
    fmax: float = 0.05,
    max_steps: int = 200,
) -> Molecule:
    """Preoptimize a molecule with semiempirical method before ab initio.

    Parameters
    ----------
    mol : Molecule
        Initial molecular geometry (bohr).
    method : str
        Semiempirical method: "dftb0" or "scc_dftb".
    fmax : float
        Force convergence (eV/Å).
    max_steps : int
        Maximum BFGS steps.

    Returns
    -------
    Molecule
        Optimized geometry (bohr), ready for ab initio refinement.
    """
    method_key = normalise_semiempirical_method(method)
    if method_key not in {"dftb0", "scc_dftb"}:
        raise ValueError(
            "Unknown molecular semiempirical preoptimization method "
            f"{method!r}; expected 'dftb0' or 'scc_dftb'."
        )

    from ..ase_optimizers import ensure_ase_scipy_compat

    ensure_ase_scipy_compat()  # ASE < 3.23 + SciPy >= 1.14 cumtrapz shim
    try:
        from ase import Atoms
        from ase.optimize import BFGSLineSearch
        from ase.units import Bohr, Hartree
    except ImportError:
        raise ImportError("Preoptimization requires ASE: pip install ase")

    from ase.calculators.calculator import Calculator

    class _Calc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms, properties, system_changes):
            Calculator.calculate(self, atoms, properties, system_changes)
            pos_bohr = atoms.positions / Bohr
            m = Molecule(
                [Atom(int(z), list(xyz)) for z, xyz in zip(atoms.numbers, pos_bohr)],
                charge=mol.charge,
                multiplicity=mol.multiplicity,
            )
            e, g = _molecular_preopt_energy_gradient(m, method_key)
            self.results["energy"] = e * Hartree
            self.results["forces"] = -g * (Hartree / Bohr)

    positions_bohr = np.array([np.array(a.xyz) for a in mol.atoms])
    atoms = Atoms(
        numbers=[a.Z for a in mol.atoms],
        positions=positions_bohr * Bohr,
    )
    atoms.calc = _Calc()

    opt = BFGSLineSearch(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))

    if not converged:
        forces = np.asarray(atoms.get_forces(), dtype=float)
        final_max_force = (
            float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) else 0.0
        )
        n_steps = int(getattr(opt, "nsteps", max_steps))
        raise RuntimeError(
            "Semiempirical molecular preoptimization did not converge after "
            f"{n_steps} of {max_steps} allowed steps: final max force "
            f"{final_max_force:.6g} eV/A exceeds the {fmax:.6g} eV/A target."
        )

    final_pos = atoms.positions / Bohr
    return Molecule(
        [Atom(int(z), list(xyz)) for z, xyz in zip(atoms.numbers, final_pos)],
        charge=mol.charge,
        multiplicity=mol.multiplicity,
    )


def preoptimize_periodic(
    system: PeriodicSystem,
    *,
    method: str = "scc_dftb",
    fmax: float = 0.05,
    max_steps: int = 200,
    variable_cell: bool = True,
) -> PeriodicSystem:
    """Preoptimize a periodic system with semiempirical method.

    Parameters
    ----------
    system : PeriodicSystem
        Initial periodic system.
    method : str
        "dftb0", "scc_dftb", "gfn2_xtb", or "pm6". Bloch-periodic OMx is
        gated until its published image-resolved Hamiltonian is implemented.
    fmax : float
        Force convergence (eV/Å).
    max_steps : int
        Maximum BFGS steps.
    variable_cell : bool
        If True, optimize both atomic positions and lattice vectors.

    Returns
    -------
    PeriodicSystem
        Optimized periodic system.
    """
    import numpy as np

    from vibeqc._vibeqc_core import Atom, PeriodicSystem
    from vibeqc.semiempirical.periodic import make_periodic_energy_function
    from vibeqc.semiempirical.routes import plan_periodic_semiempirical_route

    route_plan = plan_periodic_semiempirical_route(method, system)
    energy_fn = make_periodic_energy_function(
        route_plan,
        system,
        cutoff_bohr=12.0,
    )

    if variable_cell:
        from vibeqc.semiempirical.periodic import optimize_cell

        return optimize_cell(system, energy_fn, fmax=fmax, max_steps=max_steps)
    else:
        # Atomic positions only, fixed cell
        from ..ase_optimizers import ensure_ase_scipy_compat

        ensure_ase_scipy_compat()  # ASE < 3.23 + SciPy >= 1.14 cumtrapz shim
        try:
            from ase import Atoms
            from ase.optimize import BFGSLineSearch
            from ase.units import Bohr, Hartree
        except ImportError:
            raise ImportError("Preoptimization requires ASE")

        from ase.calculators.calculator import Calculator

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(self, atoms, properties, system_changes):
                Calculator.calculate(self, atoms, properties, system_changes)
                pos_bohr = atoms.positions / Bohr
                new_sys = PeriodicSystem()
                new_sys.dim = system.dim
                new_sys.lattice = np.asarray(system.lattice).copy()
                new_sys.unit_cell = [
                    Atom(int(z), list(xyz)) for z, xyz in zip(atoms.numbers, pos_bohr)
                ]
                new_sys.charge = system.charge
                new_sys.multiplicity = system.multiplicity
                e = energy_fn(new_sys)
                self.results["energy"] = e * Hartree
                # FD forces
                from vibeqc.semiempirical.periodic import _fd_forces

                forces = _fd_forces(new_sys, energy_fn)
                self.results["forces"] = forces * (Hartree / Bohr)

        atoms_list = list(system.unit_cell)
        positions_bohr = np.array([np.array(a.xyz) for a in atoms_list])
        atoms = Atoms(
            numbers=[a.Z for a in atoms_list],
            positions=positions_bohr * Bohr,
            cell=np.asarray(system.lattice) * Bohr,
            pbc=True,
        )
        atoms.calc = _Calc()

        opt = BFGSLineSearch(atoms, logfile=None)
        opt.run(fmax=fmax, steps=max_steps)

        final_pos = atoms.positions / Bohr
        result = PeriodicSystem()
        result.dim = system.dim
        result.lattice = np.asarray(system.lattice).copy()
        result.unit_cell = [
            Atom(int(z), list(xyz)) for z, xyz in zip(atoms.numbers, final_pos)
        ]
        result.charge = system.charge
        result.multiplicity = system.multiplicity
        return result
