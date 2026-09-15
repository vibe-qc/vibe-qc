"""Stress tensor and variable-cell optimization for semiempirical models.

Stage 6: Finite-difference stress tensor for periodic DFTB0/SCC-DFTB.
Stage 7: Variable-cell optimization bridge.
Stage 8: Preoptimization workflow (semiempirical -> ab initio handoff).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibeqc._vibeqc_core import Atom, PeriodicSystem
from vibeqc.semiempirical.routes import (
    BOUNDARY_PERIODIC_GAMMA,
    BOUNDARY_PERIODIC_K,
    SemiempiricalRoutePlan,
    plan_periodic_semiempirical_route,
)


@dataclass(frozen=True)
class PeriodicSemiempiricalDerivativeResult:
    """Energy, gradient, stress, and native FD counters for periodic SE routes."""

    energy: float
    gradient: np.ndarray
    stress: np.ndarray
    route_plan: SemiempiricalRoutePlan
    parameter_identity: str = ""
    parameter_sha256: str = ""
    free_energy: float | None = None
    differentiated_potential: str = "energy"
    energy_evaluations: int = 0
    workspace_bytes: int = 0
    differentiated_free_energy: bool = False
    memory_counters_complete: bool = False
    #: Electronic ``k_B T`` in Hartree the derivatives were evaluated at
    #: (0.0 for the exact zero-temperature Aufbau surface).
    smearing_temperature: float = 0.0


@dataclass(frozen=True)
class PeriodicSemiempiricalOptimizationResult:
    """Periodic semiempirical geometry-optimization result.

    ``energy`` is the potential the optimizer minimised. At finite electronic
    temperature that is the Mermin free energy ``F = E - T S`` (Mermin, Phys.
    Rev. 137, A1441 (1965)), the potential whose derivative the smeared
    gradient and stress are (Weinert and Davenport, Phys. Rev. B 45, 13709
    (1992), Sec. III); ``minimised_potential`` says which, and
    ``internal_energy`` / ``free_energy`` carry both values so ``E``, ``-TS``
    and ``F`` can be reported side by side (GitLab #545). At ``T = 0`` the
    three extra fields stay at their defaults and ``energy`` is the internal
    energy as before.
    """

    system: PeriodicSystem
    energy: float
    gradient: np.ndarray
    n_iter: int
    converged: bool
    route_plan: SemiempiricalRoutePlan | None = None
    parameter_identity: str = ""
    parameter_sha256: str = ""
    minimised_potential: str = "energy"
    internal_energy: float | None = None
    free_energy: float | None = None
    smearing_temperature: float = 0.0


def _minimised_potential(derivatives) -> float:
    """The scalar an optimizer must pair with ``derivatives.gradient``.

    A combined derivative result carries the internal energy ``E`` and, at
    finite Fermi-Dirac smearing, the Mermin free energy ``F = E - T S`` it
    actually differentiated. With fractional occupations the variational
    functional is ``F``, and the Hellmann-Feynman force is ``-dF/dR`` with no
    occupation-number correction term (Weinert and Davenport, Phys. Rev. B
    45, 13709 (1992), Sec. II Eq. (10') and Sec. III); ``-dE/dR`` is not the
    force conjugate to the smeared state. Feeding ASE ``E`` beside ``grad F``
    made BFGS minimise one surface with the slopes of another (GitLab #545),
    so the potential follows ``differentiated_free_energy``.
    """
    if bool(getattr(derivatives, "differentiated_free_energy", False)):
        free_energy = getattr(derivatives, "free_energy", None)
        if free_energy is None:
            raise RuntimeError(
                "periodic semiempirical derivatives differentiate the free "
                "energy but carry no free_energy value"
            )
        return float(free_energy)
    return float(derivatives.energy)


def _optimization_potential_fields(final_derivatives) -> dict:
    """``PeriodicSemiempiricalOptimizationResult`` fields describing the
    minimised potential, from the final combined derivative result."""
    if final_derivatives is None or not bool(
        getattr(final_derivatives, "differentiated_free_energy", False)
    ):
        return {}
    return {
        "minimised_potential": "free_energy",
        "internal_energy": float(final_derivatives.energy),
        "free_energy": float(final_derivatives.free_energy),
        "smearing_temperature": float(
            getattr(final_derivatives, "smearing_temperature", 0.0) or 0.0
        ),
    }


def _derivative_parameter_provenance(result: object) -> tuple[str, str]:
    """Extract complete immutable parameter provenance or fail closed."""
    if result is None:
        return "", ""
    identity = getattr(result, "parameter_identity", None)
    sha256 = getattr(result, "parameter_sha256", None)
    if identity is None and sha256 is None:
        return "", ""
    if not isinstance(identity, str) or not identity:
        raise RuntimeError(
            "periodic semiempirical derivatives carry incomplete parameter "
            "provenance"
        )
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
    ):
        raise RuntimeError(
            "periodic semiempirical derivatives carry invalid parameter "
            "provenance"
        )
    return identity, sha256


def _clone_periodic_system(
    system: PeriodicSystem,
    *,
    lattice,
    atoms,
) -> PeriodicSystem:
    """Return a new ``PeriodicSystem`` preserving charge and spin metadata."""
    return PeriodicSystem(
        system.dim,
        np.asarray(lattice, dtype=float),
        list(atoms),
        system.charge,
        system.multiplicity,
    )


def finite_difference_gradient(
    system: PeriodicSystem,
    energy_fn,
    h: float = 0.001,
) -> np.ndarray:
    """Central-difference periodic nuclear gradient.

    The helper is intentionally generic so PM6/OMx/GFN2 stop carrying separate
    Python displacement loops while the finite-difference routes remain
    experimental.
    """
    atoms = list(system.unit_cell)
    lattice = np.asarray(system.lattice, dtype=float)
    grad = np.zeros((len(atoms), 3))

    for atom_idx, atom in enumerate(atoms):
        for coord_idx in range(3):
            xyz_p = list(atom.xyz)
            xyz_p[coord_idx] += h
            atoms_p = list(atoms)
            atoms_p[atom_idx] = Atom(atom.Z, xyz_p)
            ep = energy_fn(
                _clone_periodic_system(system, lattice=lattice, atoms=atoms_p)
            )

            xyz_m = list(atom.xyz)
            xyz_m[coord_idx] -= h
            atoms_m = list(atoms)
            atoms_m[atom_idx] = Atom(atom.Z, xyz_m)
            em = energy_fn(
                _clone_periodic_system(system, lattice=lattice, atoms=atoms_m)
            )

            grad[atom_idx, coord_idx] = (ep - em) / (2.0 * h)

    return grad


def finite_difference_stress(
    system: PeriodicSystem,
    energy_fn,
    h: float = 0.001,
    *,
    strain_positions: bool = False,
) -> np.ndarray:
    """Central-difference periodic stress tensor.

    ``strain_positions=True`` applies the homogeneous strain to atomic
    coordinates as well as to the lattice, matching the PM6/OMx wrappers'
    historical behavior.
    """
    dim = system.dim
    volume = abs(np.linalg.det(np.asarray(system.lattice)))
    lattice0 = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    stress = np.zeros((3, 3))

    for i in range(dim):
        for j in range(dim):
            eps = np.zeros((3, 3))
            eps[i, j] = h

            transform_p = np.eye(3) + eps
            lattice_p = transform_p @ lattice0
            if strain_positions:
                atoms_p = [
                    Atom(atom.Z, list(transform_p @ np.asarray(atom.xyz, dtype=float)))
                    for atom in atoms
                ]
            else:
                atoms_p = atoms
            ep = energy_fn(
                _clone_periodic_system(system, lattice=lattice_p, atoms=atoms_p)
            )

            transform_m = np.eye(3) - eps
            lattice_m = transform_m @ lattice0
            if strain_positions:
                atoms_m = [
                    Atom(atom.Z, list(transform_m @ np.asarray(atom.xyz, dtype=float)))
                    for atom in atoms
                ]
            else:
                atoms_m = atoms
            em = energy_fn(
                _clone_periodic_system(system, lattice=lattice_m, atoms=atoms_m)
            )

            stress[i, j] = (ep - em) / (2.0 * h * volume)

    return stress


def compute_stress_fd(
    system: PeriodicSystem,
    energy_fn,
    h: float = 0.001,
) -> np.ndarray:
    """Finite-difference stress tensor for a periodic system.

    Parameters
    ----------
    system : PeriodicSystem
        The periodic system (positions in bohr, lattice in bohr).
    energy_fn : callable
        Function that takes a PeriodicSystem and returns the total energy
        per unit cell (Hartree).
    h : float
        Finite-difference step for strain (dimensionless).

    Returns
    -------
    stress : ndarray of shape (3, 3)
        Stress tensor s_{ij} = (1/V) . dE/de_{ij} in Hartree/bohr^3.
        For dim < 3, only the first ``dim`` rows/columns are meaningful;
        the vacuum directions are zero-filled.
    """
    return finite_difference_stress(system, energy_fn, h=h)


def optimize_periodic_positions(
    system: PeriodicSystem,
    energy_fn,
    gradient_fn=None,
    *,
    derivatives_fn=None,
    gradient_tolerance_ha_bohr: float = 1.0e-4,
    max_steps: int = 100,
    route_plan: SemiempiricalRoutePlan | None = None,
) -> PeriodicSemiempiricalOptimizationResult:
    """Relax atomic positions at fixed cell with an ASE optimizer.

    ``derivatives_fn`` may return a combined object with ``energy`` and
    ``gradient`` fields. Full-k DFTB uses that path so a geometry step consumes
    the same native batched finite-difference result contract as NEB and direct
    derivative calls.
    """
    if (
        not np.isfinite(float(gradient_tolerance_ha_bohr))
        or float(gradient_tolerance_ha_bohr) <= 0.0
    ):
        raise ValueError("gradient_tolerance_ha_bohr must be finite and positive")
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")

    from ..ase_optimizers import ensure_ase_scipy_compat

    ensure_ase_scipy_compat()
    try:
        from ase import Atoms
        from ase.optimize import BFGSLineSearch
        from ase.units import Bohr, Hartree
    except ImportError:
        raise ImportError("Periodic semiempirical optimization requires ASE")

    from ase.calculators.calculator import Calculator
    from vibeqc.molecular_optimize import _gradient_converged

    atoms_list = list(system.unit_cell)
    positions_bohr = np.array([np.array(a.xyz) for a in atoms_list])
    lattice = np.asarray(system.lattice, dtype=float)

    def build_system(atms) -> PeriodicSystem:
        pos_bohr = atms.positions / Bohr
        return _clone_periodic_system(
            system,
            lattice=lattice,
            atoms=[
                type(atoms_list[0])(int(z), list(xyz))
                for z, xyz in zip(atms.numbers, pos_bohr)
            ],
        )

    class _SemiempiricalPositionCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atms, properties, system_changes):
            Calculator.calculate(self, atms, properties, system_changes)
            candidate = build_system(atms)
            derivatives = None
            if derivatives_fn is not None and "forces" in properties:
                derivatives = derivatives_fn(candidate)

            if derivatives is not None and hasattr(derivatives, "energy"):
                # The potential paired with derivatives.gradient (#545).
                energy = _minimised_potential(derivatives)
            else:
                energy = float(energy_fn(candidate))
            self.results["energy"] = energy * Hartree

            if "forces" in properties:
                if derivatives is not None and hasattr(derivatives, "gradient"):
                    gradient = np.asarray(derivatives.gradient, dtype=float)
                elif gradient_fn is not None:
                    gradient = np.asarray(gradient_fn(candidate), dtype=float)
                else:
                    gradient = finite_difference_gradient(candidate, energy_fn)
                self.results["forces"] = -gradient * (Hartree / Bohr)

    atoms = Atoms(
        numbers=[a.Z for a in atoms_list],
        positions=positions_bohr * Bohr,
        cell=lattice * Bohr,
        pbc=True,
    )
    atoms.calc = _SemiempiricalPositionCalculator()

    opt = BFGSLineSearch(atoms, logfile=None)
    ase_fmax = float(gradient_tolerance_ha_bohr) * (Hartree / Bohr)
    run_ok = opt.run(fmax=ase_fmax, steps=int(max_steps))

    final_system = build_system(atoms)
    final_derivatives = None
    if derivatives_fn is not None:
        final_derivatives = derivatives_fn(final_system)
    if final_derivatives is not None and hasattr(final_derivatives, "energy"):
        final_energy = _minimised_potential(final_derivatives)
    else:
        final_energy = float(energy_fn(final_system))
    if final_derivatives is not None and hasattr(final_derivatives, "gradient"):
        final_gradient = np.asarray(final_derivatives.gradient, dtype=float)
    elif gradient_fn is not None:
        final_gradient = np.asarray(gradient_fn(final_system), dtype=float)
    else:
        final_gradient = finite_difference_gradient(final_system, energy_fn)

    step_count = getattr(opt, "nsteps", None)
    if step_count is None and hasattr(opt, "get_number_of_steps"):
        step_count = opt.get_number_of_steps()
    if step_count is None:
        step_count = int(max_steps)
    converged, _ = _gradient_converged(
        bool(run_ok),
        final_gradient.reshape(-1),
        float(gradient_tolerance_ha_bohr),
    )
    parameter_identity, parameter_sha256 = _derivative_parameter_provenance(
        final_derivatives
    )
    return PeriodicSemiempiricalOptimizationResult(
        system=final_system,
        energy=final_energy,
        gradient=final_gradient,
        n_iter=int(step_count),
        converged=bool(converged),
        route_plan=route_plan,
        parameter_identity=parameter_identity,
        parameter_sha256=parameter_sha256,
        **_optimization_potential_fields(final_derivatives),
    )


def _as_bloch_kmesh(system: PeriodicSystem, kpoints):
    """Materialize user-facing k-point input as a native BlochKMesh."""
    from vibeqc._vibeqc_core import monkhorst_pack

    if kpoints is None:
        return monkhorst_pack(system, [1, 1, 1])
    if hasattr(kpoints, "to_bloch_kmesh") or (
        hasattr(kpoints, "kpoints") and hasattr(kpoints, "weights")
    ):
        from vibeqc.kpoints import as_bloch_kmesh

        return as_bloch_kmesh(kpoints)

    if isinstance(kpoints, (list, tuple, np.ndarray)):
        raw_mesh = np.asarray(kpoints)
    else:
        raw_mesh = np.asarray([kpoints, kpoints, kpoints])
    if raw_mesh.shape != (3,) or raw_mesh.dtype.kind not in "iuf":
        raise ValueError(
            "periodic semiempirical k-point mesh sizes must be three finite "
            "positive integers."
        )
    mesh = np.asarray(raw_mesh, dtype=float)
    if (
        not np.all(np.isfinite(mesh))
        or np.any(mesh <= 0.0)
        or not np.array_equal(mesh, np.floor(mesh))
    ):
        raise ValueError(
            "periodic semiempirical k-point mesh sizes must be three finite "
            "positive integers."
        )
    return monkhorst_pack(system, [int(n) for n in mesh])


def make_periodic_energy_function(
    method: str | SemiempiricalRoutePlan,
    system: PeriodicSystem,
    *,
    cutoff_bohr: float = 12.0,
):
    """Build a periodic semiempirical energy closure for optimizers.

    The returned callable takes a ``PeriodicSystem`` and returns a Hartree per
    cell energy. Parameters are loaded once from the initial system so geometry
    and cell optimizers do not repeat method dispatch on every trial point.
    """
    route_plan = plan_periodic_semiempirical_route(method, system)
    method_key = route_plan.method_key

    def require_converged(result, label: str):
        converged = getattr(result, "converged", None)
        if converged is not None and not bool(converged):
            n_iter = getattr(result, "n_iter", getattr(result, "iterations", None))
            detail = (
                f" after {int(n_iter)} iterations"
                if n_iter is not None
                else ""
            )
            raise RuntimeError(
                f"periodic {label.upper()} did not converge{detail}"
            )
        return result

    if method_key in ("dftb0", "scc_dftb"):
        from vibeqc._vibeqc_core import semiempirical as _se
        from vibeqc.semiempirical.parameters import default_parameters

        params = default_parameters()
        if method_key == "scc_dftb":

            def energy_fn(candidate):
                opts = _se.PeriodicSCCOptions()
                opts.cutoff_bohr = cutoff_bohr
                run = (
                    _se.run_uscc_dftb_gamma
                    if route_plan.spin == "unrestricted"
                    else _se.run_scc_dftb_gamma
                )
                result = run(candidate, params, opts)
                require_converged(result, method_key)
                return float(result.energy)

        else:

            def energy_fn(candidate):
                run = (
                    _se.run_udftb0_gamma
                    if route_plan.spin == "unrestricted"
                    else _se.run_dftb0_gamma
                )
                return float(run(candidate, params).energy)

        return energy_fn

    if method_key == "gfn2_xtb":
        from vibeqc._vibeqc_core.semiempirical.xtb import run_gfn2_xtb_gamma
        from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

        params = load_gfn2_params()

        def energy_fn(candidate):
            result = run_gfn2_xtb_gamma(
                candidate,
                params,
                cutoff_bohr=cutoff_bohr,
            )
            require_converged(result, method_key)
            # Default-on frontier smearing: optimizers minimize the Mermin
            # free energy A = E - T*S when the run is smeared, and the
            # internal energy under exact zero-temperature Aufbau.
            if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
                return float(result.free_energy)
            return float(result.energy)

        return energy_fn

    if method_key == "pm6":
        from vibeqc.semiempirical.methods.periodic_pm6 import run_pm6_gamma
        from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

        params = load_pm6_params_auto([atom.Z for atom in system.unit_cell])

        def energy_fn(candidate):
            result = run_pm6_gamma(
                candidate,
                params,
                cutoff_bohr=cutoff_bohr,
            )
            require_converged(result, method_key)
            return float(result.energy)

        return energy_fn

    if method_key in ("om1", "om2", "om3"):
        from vibeqc.semiempirical.methods.omx_params import (
            load_om1_params,
            load_om2_params,
            load_om3_params,
        )
        from vibeqc.semiempirical.methods.periodic_omx import run_omx_gamma

        loaders = {
            "om1": load_om1_params,
            "om2": load_om2_params,
            "om3": load_om3_params,
        }
        params = loaders[method_key]()

        def energy_fn(candidate):
            result = run_omx_gamma(
                candidate,
                variant=method_key,
                params=params,
                cutoff_bohr=cutoff_bohr,
            )
            require_converged(result, method_key)
            return float(result.energy)

        return energy_fn

    raise ValueError(f"Unknown semiempirical periodic method: {method!r}")


def _evaluate_kpoint_dftb_derivatives(
    route_plan: SemiempiricalRoutePlan,
    system: PeriodicSystem,
    *,
    kpoints,
    cutoff_bohr: float,
    fd_step_bohr: float,
    strain_step: float,
    compute_stress: bool,
    max_iter: int | None = None,
    conv_tol_charge: float | None = None,
    smearing_temperature_hartree: float = 0.0,
) -> PeriodicSemiempiricalDerivativeResult:
    if kpoints is None:
        raise ValueError("full-k periodic semiempirical derivatives require kpoints=.")
    if route_plan.spin != "closed_shell":
        raise NotImplementedError(
            "full-k periodic DFTB derivatives are closed-shell only; "
            "unrestricted k-point DFTB/SCC-DFTB derivatives are not implemented."
        )
    if route_plan.method_key not in {"dftb0", "scc_dftb"}:
        raise NotImplementedError(
            "full-k periodic semiempirical derivatives are currently "
            "implemented only for DFTB0 and SCC-DFTB."
        )

    kmesh = _as_bloch_kmesh(system, kpoints)

    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical.parameters import default_parameters

    params = default_parameters()
    fd_options = _se.KPointDFTBFDBatchOptions()
    fd_options.coordinate_step = float(fd_step_bohr)
    fd_options.strain_step = float(strain_step)
    fd_options.compute_stress = bool(compute_stress)
    occupation_options = _se.KPointOccupationOptions()
    occupation_options.smearing_temperature = float(smearing_temperature_hartree)

    if route_plan.method_key == "dftb0":
        energy_result = _se.run_dftb0_kpoints(
            system,
            params,
            kmesh,
            float(cutoff_bohr),
            occupation_options,
        )
        derivatives = _se.compute_dftb0_kpoints_fd_batch(
            system,
            params,
            kmesh,
            float(cutoff_bohr),
            fd_options,
            occupation_options,
        )
    else:
        scc_options = _se.SCCOptions()
        if max_iter is not None:
            scc_options.max_iter = int(max_iter)
        if conv_tol_charge is not None:
            scc_options.conv_tol_charge = float(conv_tol_charge)
        energy_result = _se.run_scc_dftb_kpoints(
            system,
            params,
            kmesh,
            scc_options,
            float(cutoff_bohr),
            occupation_options,
            # Issue #342: opt in to the returned record so this facade
            # keeps raising its own derivative-context error below.
            allow_unconverged=True,
        )
        if not energy_result.converged:
            raise RuntimeError(
                "periodic full-k SCC-DFTB did not converge during derivative "
                "evaluation"
            )
        derivatives = _se.compute_scc_dftb_kpoints_fd_batch(
            system,
            params,
            kmesh,
            scc_options,
            float(cutoff_bohr),
            fd_options,
            occupation_options,
        )

    parameter_identity = getattr(energy_result, "parameter_identity", None)
    parameter_sha256 = getattr(energy_result, "parameter_sha256", None)
    derivative_parameter_identity = getattr(
        derivatives, "parameter_identity", None
    )
    derivative_parameter_sha256 = getattr(derivatives, "parameter_sha256", None)
    provenance = (
        parameter_identity,
        parameter_sha256,
        derivative_parameter_identity,
        derivative_parameter_sha256,
    )
    if not all(isinstance(value, str) and value for value in provenance):
        raise RuntimeError(
            "full-k DFTB energy and derivative results must carry immutable "
            "parameter provenance"
        )
    if (
        derivative_parameter_identity != parameter_identity
        or derivative_parameter_sha256 != parameter_sha256
    ):
        raise RuntimeError(
            "full-k DFTB energy and derivative results used different "
            "immutable parameter snapshots"
        )

    differentiated_free_energy = bool(
        getattr(derivatives, "differentiated_free_energy", False)
    )
    return PeriodicSemiempiricalDerivativeResult(
        energy=float(energy_result.energy),
        gradient=np.asarray(derivatives.gradient, dtype=float),
        stress=np.asarray(derivatives.stress, dtype=float),
        route_plan=route_plan,
        parameter_identity=parameter_identity,
        parameter_sha256=parameter_sha256,
        free_energy=float(getattr(energy_result, "free_energy", energy_result.energy)),
        differentiated_potential=(
            "free_energy" if differentiated_free_energy else "energy"
        ),
        energy_evaluations=int(getattr(derivatives, "energy_evaluations", 0)),
        workspace_bytes=int(getattr(derivatives, "workspace_bytes", 0)),
        differentiated_free_energy=differentiated_free_energy,
        memory_counters_complete=bool(
            getattr(derivatives, "memory_counters_complete", False)
        ),
        smearing_temperature=float(smearing_temperature_hartree),
    )


def _resolve_kpoint_smearing_temperature(
    smearing_temperature: float | str | None,
    *,
    smearing_unit: str,
    smearing_method: str,
) -> float:
    from vibeqc.smearing.resolution import resolve_smearing_temperature

    resolution = resolve_smearing_temperature(
        smearing_temperature,
        unit=smearing_unit,
        method=smearing_method,
    )
    temperature = float(resolution.temperature)
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("smearing_temperature must be finite and >= 0")
    return temperature


def _require_same_parameter_snapshot(
    result: object,
    expected_sha256: str | None,
    route: str,
) -> object:
    """Reject a mixed-snapshot finite-difference energy batch."""
    if expected_sha256 is None:
        # Keep compatibility with deliberately minimal test doubles.
        # Production native results always carry the exact SHA-256.
        return result
    actual_sha256 = getattr(result, "parameter_sha256", None)
    if actual_sha256 is None or str(actual_sha256) != expected_sha256:
        raise RuntimeError(
            f"{route} parameters changed during periodic finite "
            "differences; refusing to mix parameter snapshots"
        )
    return result


def evaluate_periodic_kpoint_energy_gradient_stress(
    method: str | SemiempiricalRoutePlan,
    system: PeriodicSystem,
    *,
    kpoints,
    cutoff_bohr: float = 15.0,
    fd_step_bohr: float = 0.001,
    strain_step: float = 0.001,
    max_iter: int | None = None,
    conv_tol_charge: float | None = None,
    smearing_temperature: float | str | None = 0.0,
    smearing_unit: str = "hartree",
    smearing_method: str = "fermi-dirac",
) -> PeriodicSemiempiricalDerivativeResult:
    """Evaluate full-k DFTB/SCC-DFTB energy, gradient, and stress.

    This public facade is experimental. At zero smearing it differentiates the
    internal energy. At finite Fermi-Dirac smearing it differentiates the
    Mermin free energy; the returned result keeps ``energy`` as the internal
    energy, exposes ``free_energy`` separately, and labels
    ``differentiated_potential`` as ``"free_energy"``.

    The native batched finite-difference kernel makes coordinate and strain
    displacements share parameter views, system workspace, and k-mesh handling.
    """
    route_plan = plan_periodic_semiempirical_route(
        method,
        system,
        boundary=BOUNDARY_PERIODIC_K,
        properties=("energy", "gradient", "stress"),
    )
    if not np.isfinite(cutoff_bohr) or cutoff_bohr <= 0.0:
        raise ValueError("cutoff_bohr must be finite and positive")
    if not np.isfinite(fd_step_bohr) or fd_step_bohr <= 0.0:
        raise ValueError("fd_step_bohr must be finite and positive")
    if not np.isfinite(strain_step) or strain_step <= 0.0:
        raise ValueError("strain_step must be finite and positive")
    smearing_temperature_hartree = _resolve_kpoint_smearing_temperature(
        smearing_temperature,
        smearing_unit=smearing_unit,
        smearing_method=smearing_method,
    )

    return _evaluate_kpoint_dftb_derivatives(
        route_plan,
        system,
        kpoints=kpoints,
        cutoff_bohr=float(cutoff_bohr),
        fd_step_bohr=float(fd_step_bohr),
        strain_step=float(strain_step),
        compute_stress=True,
        max_iter=max_iter,
        conv_tol_charge=conv_tol_charge,
        smearing_temperature_hartree=smearing_temperature_hartree,
    )


def evaluate_periodic_energy_gradient(
    method: str | SemiempiricalRoutePlan,
    system: PeriodicSystem,
    *,
    kpoints=None,
    cutoff_bohr: float = 15.0,
    fd_step_bohr: float = 0.001,
    max_iter: int | None = None,
    conv_tol_charge: float | None = None,
    smearing_temperature: float | str | None = None,
    smearing_unit: str = "hartree",
    smearing_method: str = "fermi-dirac",
    _return_result: bool = False,
) -> tuple[float, np.ndarray] | PeriodicSemiempiricalDerivativeResult:
    """Evaluate a validated periodic semiempirical energy and gradient.

    Gamma DFTB0 uses its native analytic derivative at the kernel's validated
    15-bohr image domain. Gamma GFN2-xTB uses its native analytic gradient
    (the periodic energy is variational in the reduced SCC state). SCC-DFTB
    uses central differences of its converged total energy because its
    current periodic analytic routine is a fixed-state approximation. PM6
    and OMx use their native batched finite-difference derivatives.

    Passing ``kpoints`` selects the experimental full-k DFTB0/SCC-DFTB route,
    which uses the native batched finite-difference Bloch kernel. At finite
    Fermi-Dirac smearing this convenience facade returns the differentiated
    Mermin free energy alongside its gradient; use
    ``evaluate_periodic_kpoint_energy_gradient_stress`` when both internal
    energy and free energy must be inspected.

    For Gamma GFN2-xTB, ``smearing_temperature=None`` keeps the native
    default-on frontier smearing and the returned gradient differentiates the
    Mermin free energy ``A = E - T*S``; an explicit numeric 0.0 restores the
    exact zero-temperature Aufbau surface.
    """
    if isinstance(method, SemiempiricalRoutePlan):
        boundary = method.boundary
        if kpoints is not None and boundary != BOUNDARY_PERIODIC_K:
            raise ValueError(
                "kpoints= requires a periodic_k semiempirical route plan."
            )
    else:
        boundary = (
            BOUNDARY_PERIODIC_K
            if kpoints is not None
            else BOUNDARY_PERIODIC_GAMMA
        )
    route_plan = plan_periodic_semiempirical_route(
        method,
        system,
        boundary=boundary,
        properties=("energy", "gradient"),
    )
    if not np.isfinite(cutoff_bohr) or cutoff_bohr <= 0.0:
        raise ValueError("cutoff_bohr must be finite and positive")
    if not np.isfinite(fd_step_bohr) or fd_step_bohr <= 0.0:
        raise ValueError("fd_step_bohr must be finite and positive")
    smearing_temperature_hartree = _resolve_kpoint_smearing_temperature(
        smearing_temperature,
        smearing_unit=smearing_unit,
        smearing_method=smearing_method,
    )

    method_key = route_plan.method_key
    if route_plan.boundary == BOUNDARY_PERIODIC_K:
        derivatives = _evaluate_kpoint_dftb_derivatives(
            route_plan,
            system,
            kpoints=kpoints,
            cutoff_bohr=float(cutoff_bohr),
            fd_step_bohr=float(fd_step_bohr),
            strain_step=float(fd_step_bohr),
            compute_stress=False,
            max_iter=max_iter,
            conv_tol_charge=conv_tol_charge,
            smearing_temperature_hartree=smearing_temperature_hartree,
        )
        potential = (
            derivatives.free_energy
            if derivatives.differentiated_free_energy
            else derivatives.energy
        )
        if _return_result:
            return derivatives
        return float(potential), derivatives.gradient
    if _return_result:
        raise ValueError(
            "_return_result is available only for full-k periodic "
            "semiempirical derivatives"
        )
    if smearing_temperature_hartree > 0.0 and method_key != "gfn2_xtb":
        raise NotImplementedError(
            "finite-temperature periodic semiempirical gradients are currently "
            "implemented only for full-k DFTB0/SCC-DFTB routes and Gamma "
            "GFN2-xTB."
        )

    if method_key in ("dftb0", "scc_dftb"):
        from vibeqc._vibeqc_core import semiempirical as _se
        from vibeqc.semiempirical.parameters import default_parameters

        params = default_parameters()
        if method_key == "dftb0":
            opts = _se.PeriodicDFTB0Options()
            opts.cutoff_bohr = cutoff_bohr
            if route_plan.spin == "unrestricted":
                run = _se.run_udftb0_gamma
                analytic_gradient = _se.compute_periodic_udftb0_gradient
            else:
                run = _se.run_dftb0_gamma
                analytic_gradient = _se.compute_periodic_dftb0_gradient

            result = run(system, params, opts)
            expected_sha256 = getattr(result, "parameter_sha256", None)
            if expected_sha256 is not None:
                expected_sha256 = str(expected_sha256)
            if np.isclose(cutoff_bohr, 15.0, rtol=0.0, atol=1.0e-12):
                gradient = analytic_gradient(system, result, params)
            else:

                def energy(candidate):
                    displaced = _require_same_parameter_snapshot(
                        run(candidate, params, opts),
                        expected_sha256,
                        "periodic DFTB0",
                    )
                    return float(displaced.energy)

                gradient = finite_difference_gradient(
                    system,
                    energy,
                    h=fd_step_bohr,
                )
            return float(result.energy), np.asarray(gradient, dtype=float)

        opts = _se.PeriodicSCCOptions()
        opts.cutoff_bohr = cutoff_bohr
        run = (
            _se.run_uscc_dftb_gamma
            if route_plan.spin == "unrestricted"
            else _se.run_scc_dftb_gamma
        )

        def run_scc(candidate):
            result = run(candidate, params, opts)
            if not result.converged:
                raise RuntimeError(
                    "periodic SCC-DFTB did not converge during gradient evaluation"
                )
            return result

        result = run_scc(system)
        expected_sha256 = getattr(result, "parameter_sha256", None)
        if expected_sha256 is not None:
            expected_sha256 = str(expected_sha256)
        gradient = finite_difference_gradient(
            system,
            lambda candidate: float(
                _require_same_parameter_snapshot(
                    run_scc(candidate),
                    expected_sha256,
                    "periodic SCC-DFTB",
                ).energy
            ),
            h=fd_step_bohr,
        )
        return float(result.energy), gradient

    if method_key == "gfn2_xtb":
        from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
        from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

        params = load_gfn2_params()
        # None (or the string "auto") keeps the native default-on frontier
        # smearing; an explicit numeric value (0.0 included) is forwarded.
        opts = _xtb.XTBSccOptions()
        smearing_explicit = smearing_temperature is not None and not (
            isinstance(smearing_temperature, str)
            and str(smearing_temperature).strip().lower().replace("_", "-")
            == "auto"
        )
        if smearing_explicit:
            opts.electronic_temperature = float(smearing_temperature_hartree)

        def run_gfn2(candidate):
            result = _xtb.run_gfn2_xtb_gamma(
                candidate,
                params,
                opts,
                cutoff_bohr=cutoff_bohr,
            )
            if not result.converged:
                raise RuntimeError(
                    "periodic GFN2-xTB did not converge during gradient evaluation"
                )
            return result

        def potential(result):
            # Smeared runs differentiate the Mermin free energy; exact
            # zero-temperature Aufbau runs differentiate the internal energy.
            if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
                return float(result.free_energy)
            return float(result.energy)

        result = run_gfn2(system)
        # The native analytic gradient differentiates the same lattice-summed
        # energy the SCF converged (translation-covariant Ewald-split gamma,
        # Bannwarth 2019 AES with image-resolved moments; issues #296/#338):
        # the reduced SCC state is variational, so no finite differences and
        # no response term are needed.  The result carries the cutoff and
        # kernel provenance the derivative reuses.
        from vibeqc._vibeqc_core import semiempirical as _se

        gradient = np.asarray(
            _se.compute_periodic_gfn2_gradient(system, result, params),
            dtype=float,
        )
        return potential(result), gradient

    if method_key == "pm6":
        from vibeqc.semiempirical.methods.periodic_pm6 import (
            compute_pm6_gamma_derivatives_fd,
            run_pm6_gamma,
        )
        from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

        params = load_pm6_params_auto([atom.Z for atom in system.unit_cell])
        result = run_pm6_gamma(
            system,
            params,
            cutoff_bohr=cutoff_bohr,
        )
        if not result.converged:
            raise RuntimeError("periodic PM6 did not converge")
        derivatives = compute_pm6_gamma_derivatives_fd(
            system,
            params,
            h=fd_step_bohr,
            cutoff_bohr=cutoff_bohr,
            compute_stress=False,
        )
        _require_same_parameter_snapshot(
            derivatives,
            str(result.parameter_sha256),
            "periodic PM6",
        )
        return float(result.energy), np.asarray(derivatives.gradient, dtype=float)

    if method_key in ("om1", "om2", "om3"):
        from vibeqc.semiempirical.methods.omx_params import load_omx_params
        from vibeqc.semiempirical.methods.periodic_omx import (
            compute_omx_gamma_gradient_fd,
            run_omx_gamma,
        )

        params = load_omx_params(method_key)
        result = run_omx_gamma(
            system,
            method_key,
            params,
            cutoff_bohr=cutoff_bohr,
        )
        if not result.converged:
            raise RuntimeError(f"periodic {method_key.upper()} did not converge")
        gradient = compute_omx_gamma_gradient_fd(
            system,
            method_key,
            params,
            h=fd_step_bohr,
            cutoff_bohr=cutoff_bohr,
        )
        return float(result.energy), np.asarray(gradient, dtype=float)

    raise ValueError(
        f"Unknown periodic semiempirical method: {route_plan.method_key!r}"
    )


def optimize_cell(
    system: PeriodicSystem,
    energy_fn,
    gradient_fn=None,
    *,
    stress_fn=None,
    derivatives_fn=None,
    fmax: float = 0.01,
    gradient_tolerance_ha_bohr: float | None = None,
    max_steps: int = 100,
    isotropic: bool = False,
    return_result: bool = False,
    route_plan: SemiempiricalRoutePlan | None = None,
) -> PeriodicSystem | PeriodicSemiempiricalOptimizationResult:
    """Variable-cell optimization using ASE.

    Optimizes both atomic positions and lattice vectors.

    Parameters
    ----------
    system : PeriodicSystem
        Initial periodic system.
    energy_fn : callable
        Energy function PeriodicSystem -> float (Hartree/cell).
    gradient_fn : callable, optional
        Gradient function PeriodicSystem -> (n_atoms, 3) ndarray.
        If None, forces are computed via finite differences of energy_fn.
    stress_fn : callable, optional
        Stress function PeriodicSystem -> (3, 3) ndarray.
    derivatives_fn : callable, optional
        Combined native derivative batch returning ``gradient`` and ``stress``.
    fmax : float
        Force convergence tolerance (eV/Å).
    gradient_tolerance_ha_bohr : float, optional
        Public runner tolerance in Hartree/bohr. When set, it overrides
        ``fmax`` after converting to ASE force units.
    max_steps : int
        Maximum optimization steps.
    isotropic : bool
        If True, enforce isotropic cell scaling (volume only).
    return_result : bool
        If True, return the standard optimization result object instead of
        only the optimized system.

    Returns
    -------
    PeriodicSystem
        Optimized periodic system, or a
        :class:`PeriodicSemiempiricalOptimizationResult` when
        ``return_result=True``.
    """
    if gradient_tolerance_ha_bohr is not None and (
        not np.isfinite(float(gradient_tolerance_ha_bohr))
        or float(gradient_tolerance_ha_bohr) <= 0.0
    ):
        raise ValueError("gradient_tolerance_ha_bohr must be finite and positive")
    if int(max_steps) < 0:
        raise ValueError("max_steps must be non-negative")

    from ..ase_optimizers import ensure_ase_scipy_compat

    ensure_ase_scipy_compat()  # ASE < 3.23 + SciPy >= 1.14 cumtrapz shim
    try:
        from ase import Atoms
        from ase.optimize import BFGSLineSearch
        from ase.units import Bohr
    except ImportError:
        raise ImportError("Variable-cell optimization requires ASE: pip install ase")

    try:
        from ase.filters import ExpCellFilter  # ASE 3.22+
    except ImportError:
        from ase.constraints import ExpCellFilter  # ASE < 3.22

    from ase.calculators.calculator import Calculator
    from ase.units import Hartree

    atoms_list = list(system.unit_cell)
    positions_bohr = np.array([np.array(a.xyz) for a in atoms_list])
    numbers = [a.Z for a in atoms_list]
    L = np.asarray(system.lattice, dtype=float)

    atoms = Atoms(
        numbers=numbers, positions=positions_bohr * Bohr, cell=L * Bohr, pbc=True
    )

    class _SemiempiricalCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atms, properties, system_changes):
            Calculator.calculate(self, atms, properties, system_changes)
            pos_bohr = atms.positions / Bohr
            cell_bohr = atms.cell.array / Bohr

            # Guard against near-singular lattice (ExpCellFilter line search)
            vol = abs(np.linalg.det(cell_bohr))
            if vol < 1e-6:
                self.results["energy"] = 1e10  # large penalty
                if "forces" in properties:
                    self.results["forces"] = np.zeros((len(atms), 3))
                if "stress" in properties:
                    self.results["stress"] = np.zeros((3, 3))
                return

            new_sys = PeriodicSystem()
            new_sys.dim = system.dim
            new_sys.lattice = cell_bohr
            new_sys.unit_cell = [
                type(atoms_list[0])(int(z), list(xyz))
                for z, xyz in zip(atms.numbers, pos_bohr)
            ]
            new_sys.charge = system.charge
            new_sys.multiplicity = system.multiplicity

            derivatives = None
            if derivatives_fn is not None and (
                "forces" in properties or "stress" in properties
            ):
                derivatives = derivatives_fn(new_sys)

            if derivatives is not None and hasattr(derivatives, "energy"):
                # The potential paired with derivatives.gradient/.stress (#545).
                e = _minimised_potential(derivatives)
            else:
                e = float(energy_fn(new_sys))
            self.results["energy"] = e * Hartree

            gradient = None
            if derivatives is not None and hasattr(derivatives, "gradient"):
                gradient = np.asarray(derivatives.gradient, dtype=float)
            stress = None
            if derivatives is not None and hasattr(derivatives, "stress"):
                stress = np.asarray(derivatives.stress, dtype=float)

            if gradient is not None and "forces" in properties:
                self.results["forces"] = (
                    -gradient * (Hartree / Bohr)
                )
            if stress is not None and "stress" in properties:
                self.results["stress"] = (
                    stress * (Hartree / Bohr**3)
                )

            if "forces" in properties and gradient is None:
                if gradient_fn is not None:
                    forces = -gradient_fn(new_sys)  # gradient -> force
                else:
                    forces = _fd_forces(new_sys, energy_fn)
                self.results["forces"] = forces * (Hartree / Bohr)

            if "stress" in properties and stress is None:
                if stress_fn is not None:
                    stress_ha = stress_fn(new_sys)
                else:
                    stress_ha = compute_stress_fd(new_sys, energy_fn)
                self.results["stress"] = stress_ha * (Hartree / Bohr**3)

    atoms.calc = _SemiempiricalCalculator()

    if isotropic:
        try:
            from ase.filters import StrainFilter  # ASE 3.22+
        except ImportError:
            from ase.constraints import StrainFilter  # ASE < 3.22

        opt_atoms = StrainFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
    else:
        opt_atoms = ExpCellFilter(atoms)

    opt = BFGSLineSearch(opt_atoms, logfile=None)
    ase_fmax = float(fmax)
    if gradient_tolerance_ha_bohr is not None:
        ase_fmax = float(gradient_tolerance_ha_bohr) * (Hartree / Bohr)
    run_ok = opt.run(fmax=ase_fmax, steps=int(max_steps))

    # Extract optimized system
    final_pos = atoms.positions / Bohr
    final_cell = atoms.cell.array / Bohr

    result = PeriodicSystem()
    result.dim = system.dim
    result.lattice = final_cell
    result.unit_cell = [
        type(atoms_list[0])(int(z), list(xyz))
        for z, xyz in zip(atoms.numbers, final_pos)
    ]
    result.charge = system.charge
    result.multiplicity = system.multiplicity
    if return_result:
        from vibeqc.molecular_optimize import _gradient_converged

        final_derivatives = None
        if derivatives_fn is not None:
            final_derivatives = derivatives_fn(result)
        if final_derivatives is not None and hasattr(final_derivatives, "energy"):
            final_energy = _minimised_potential(final_derivatives)
        else:
            final_energy = float(energy_fn(result))
        if final_derivatives is not None and hasattr(final_derivatives, "gradient"):
            final_gradient = np.asarray(final_derivatives.gradient, dtype=float)
        elif gradient_fn is not None:
            final_gradient = np.asarray(gradient_fn(result), dtype=float)
        else:
            final_gradient = finite_difference_gradient(result, energy_fn)

        tolerance_ha_bohr = (
            float(gradient_tolerance_ha_bohr)
            if gradient_tolerance_ha_bohr is not None
            else ase_fmax / (Hartree / Bohr)
        )
        step_count = getattr(opt, "nsteps", None)
        if step_count is None and hasattr(opt, "get_number_of_steps"):
            step_count = opt.get_number_of_steps()
        if step_count is None:
            step_count = int(max_steps)
        converged, _ = _gradient_converged(
            bool(run_ok),
            final_gradient.reshape(-1),
            tolerance_ha_bohr,
        )
        parameter_identity, parameter_sha256 = (
            _derivative_parameter_provenance(final_derivatives)
        )
        return PeriodicSemiempiricalOptimizationResult(
            system=result,
            energy=final_energy,
            gradient=final_gradient,
            n_iter=int(step_count),
            converged=bool(converged),
            route_plan=route_plan,
            parameter_identity=parameter_identity,
            parameter_sha256=parameter_sha256,
            **_optimization_potential_fields(final_derivatives),
        )
    return result


def _fd_forces(system: PeriodicSystem, energy_fn, h: float = 0.001) -> np.ndarray:
    """Finite-difference forces for periodic system."""
    return -finite_difference_gradient(system, energy_fn, h=h)


# ---------------------------------------------------------------------------
# Convenience wrappers for PM6 and fail-closed OMx compatibility calls.
# ---------------------------------------------------------------------------


def optimize_pm6_cell(
    system: PeriodicSystem,
    *,
    fmax: float = 0.01,
    max_steps: int = 100,
    cutoff_bohr: float = 15.0,
) -> PeriodicSystem:
    """Variable-cell optimization with periodic PM6 (FD stress)."""
    from vibeqc.semiempirical.methods.periodic_pm6 import PeriodicPM6Model

    model = PeriodicPM6Model(
        system,
        cutoff_bohr=cutoff_bohr,
    )
    return optimize_cell(
        system,
        model.energy,
        model.gradient,
        stress_fn=model.stress,
        derivatives_fn=model.derivatives,
        fmax=fmax,
        max_steps=max_steps,
    )


def optimize_omx_cell(
    system: PeriodicSystem,
    variant: str = "om2",
    *,
    fmax: float = 0.01,
    max_steps: int = 100,
    cutoff_bohr: float = 15.0,
) -> PeriodicSystem:
    """Reject variable-cell periodic OMx through its fail-closed route gate."""
    from vibeqc.semiempirical.methods.periodic_omx import PeriodicOMxModel

    model = PeriodicOMxModel(
        system,
        variant,
        cutoff_bohr=cutoff_bohr,
    )
    return optimize_cell(
        system,
        model.energy,
        model.gradient,
        stress_fn=model.stress,
        derivatives_fn=model.derivatives,
        fmax=fmax,
        max_steps=max_steps,
    )
