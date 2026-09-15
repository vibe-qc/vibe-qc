"""Periodic geometry optimisation providers and coordinates (Phase 7).

Integrates BIPOLE periodic SCF into the geomopt framework:
* :class:`PeriodicSCFProvider` -- wraps BIPOLE energy + FD/analytic gradients
* :class:`FractionalCoordinates` -- atom positions in fractional space
* :class:`CellStrainCoordinates` -- 6 Voigt strain components
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

from .._vibeqc_core import (
    Atom,
    BasisSet,
    BlochKMesh,
    Functional,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    bloch_kmesh_from_lists,
    monkhorst_pack,
)
from ..bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
    compute_bipole_gradient_rks,
    compute_bipole_gradient_uhf,
    compute_bipole_gradient_uks,
)
from ..bipole_optimize import SCFNonConvergence, _kmesh_for_system
from ..molecular_optimize import _gradient_converged
from ..output import render_energy_labeled
from ..pbc_bipole import run_pbc_bipole_rhf
from ..pbc_bipole_common import BIPOLE_TRIAL_PENALTY_HA, BipoleFoldUnreliableError
from ..pbc_bipole_rks import run_pbc_bipole_rks
from ..pbc_bipole_uhf import run_pbc_bipole_uhf
from ..pbc_bipole_uks import run_pbc_bipole_uks
from ..progress import resolve_progress
from .coordinates import CoordinateRepresentation
from .providers import EnergyGradientProvider, HessianProvider

KMeshInput = Union[BlochKMesh, Any]  # BlochKMesh or KPoints

# ---------------------------------------------------------------------------
# Periodic SCF Provider
# ---------------------------------------------------------------------------


class PeriodicSCFProvider:
    """An :class:`EnergyGradientProvider` backed by vibe-qc's BIPOLE SCF.

    Wraps the BIPOLE energy and force evaluation, supporting both
    finite-difference (default, correct by construction) and analytic
    (research-preview) gradient modes.  BIPOLE geometry optimisation is a
    three-dimensional periodic model: calling the provider with ``dim < 3``
    fails before any SCF or provider state mutation.  The direct-truncated
    low-dimensional driver path is diagnostic only and is not a valid
    optimisation objective.

    Parameters
    ----------
    basis_name : str
        Basis-set name.
    kmesh : BlochKMesh or KPoints
        k-point mesh (re-evaluated when lattice changes).
    method : str
        ``"RHF"``, ``"UHF"``, ``"RKS"``, ``"UKS"``.
    functional : str or None
        XC functional for RKS/UKS.
    force_mode : str
        ``"fd"`` (default) or ``"analytic"``.
    fd_step_bohr : float
        Finite-difference displacement for FD forces.
    cutoff_bohr : float or None
        Lattice-sum cutoff (bohr) for BOTH the AO and the nuclear sums.
        ``None`` (default) inherits the wrapped BIPOLE drivers' own
        ``LatticeSumOptions`` defaults (15.0 bohr AO, 25.0 bohr nuclear), or
        the caller's ``scf_options.lattice_opts`` when given -- the provider
        no longer runs a tighter operator than the route it wraps (IID 540;
        the previous default was 8.0 for both, which is what left the fold
        guard one order of magnitude from refusing on compact cells). An
        explicit value keeps the historical semantics: it is applied to
        both sums, overriding ``scf_options``.
    scf_options : PeriodicRHFOptions, PeriodicKSOptions, or None
    scf_conv_tol, scf_max_iter : float, int or None
        Override SCF convergence parameters.
    """

    _SCF_FIELDS = (
        "functional",
        "grid",
        "max_iter",
        "conv_tol_energy",
        "conv_tol_grad",
        "damping",
        "dynamic_damping",
        "dynamic_damping_min",
        "dynamic_damping_max",
        "fock_mixing",
        "use_diis",
        "diis_start_iter",
        "diis_subspace_size",
        "scf_accelerator",
        "ediis_diis_switch_threshold",
        "initial_guess",
        "level_shift",
        "level_shift_warmup_cycles",
        "smearing_temperature",
        "quadratic_fallback_iter",
        "quadratic_fallback_shift",
        "quadratic_fallback_max_step",
        "use_periodic_becke",
        "becke_image_radius_bohr",
    )
    _LATTICE_FIELDS = (
        "cutoff_bohr",
        "nuclear_cutoff_bohr",
        "coulomb_method",
        "screening_overlap_threshold",
        "screening_exchange_threshold",
        "schwarz_threshold",
        "schwarz_threshold_forces",
    )

    def __init__(
        self,
        basis_name: str,
        kmesh: KMeshInput,
        method: str = "RHF",
        *,
        functional: Optional[str] = None,
        force_mode: str = "fd",
        fd_step_bohr: float = 1e-3,
        cutoff_bohr: Optional[float] = None,
        scf_options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
        scf_conv_tol: Optional[float] = None,
        scf_max_iter: Optional[int] = None,
        dft_plus_u: Any = None,
    ):
        self._basis_name = basis_name
        self._kmesh: BlochKMesh = _ensure_bloch(kmesh)
        self._method = method.upper()
        self._functional = functional
        self._force_mode = force_mode
        self._fd_step_bohr = fd_step_bohr
        self._dft_plus_u = dft_plus_u

        # Build local SCF options (don't mutate caller's)
        is_ks = self._method in ("RKS", "UKS")
        self._opts: Any = PeriodicKSOptions() if is_ks else PeriodicRHFOptions()
        if scf_options is not None:
            for name in self._SCF_FIELDS:
                if hasattr(scf_options, name) and hasattr(self._opts, name):
                    setattr(self._opts, name, getattr(scf_options, name))
            if hasattr(scf_options, "lattice_opts"):
                src = scf_options.lattice_opts
                out = LatticeSumOptions()
                for name in self._LATTICE_FIELDS:
                    if hasattr(src, name):
                        setattr(out, name, getattr(src, name))
                self._opts.lattice_opts = out
        else:
            self._opts.max_iter = 50
            self._opts.use_diis = True
            self._opts.conv_tol_energy = 1e-7

        if scf_conv_tol is not None:
            self._opts.conv_tol_energy = float(scf_conv_tol)
        if scf_max_iter is not None:
            self._opts.max_iter = int(scf_max_iter)
        if cutoff_bohr is not None:
            # Explicit request: historical set-both semantics, overriding any
            # scf_options.lattice_opts. None leaves the driver defaults (or the
            # caller's lattice_opts) untouched -- previously an unconditional
            # 8.0 silently clobbered both, including caller-supplied values.
            self._opts.lattice_opts.cutoff_bohr = float(cutoff_bohr)
            self._opts.lattice_opts.nuclear_cutoff_bohr = float(cutoff_bohr)
        # Record what will actually run, never the unresolved request.
        self._cutoff_bohr = float(self._opts.lattice_opts.cutoff_bohr)

        if is_ks and functional:
            self._opts.functional = functional

        # Track reference system for kmesh remapping
        self._reference_system: Optional[PeriodicSystem] = None
        # Recovery contract for refused / non-converged TRIAL geometries
        # (IID 536): only after one good evaluation is there a point to
        # back off to; until then a refusal is the user's own geometry.
        self._had_good_eval = False

    @property
    def method(self) -> str:
        return self._method

    @property
    def basis_name(self) -> str:
        return self._basis_name

    def _run_scf(self, system: PeriodicSystem) -> tuple[float, Any]:
        """SCF at *system*; ``(energy, result)``.

        A trial geometry the BIPOLE stack refuses
        (:class:`~vibeqc.pbc_bipole_common.BipoleFoldUnreliableError`) or
        cannot converge returns ``(BIPOLE_TRIAL_PENALTY_HA, None)`` instead
        of raising, once at least one evaluation has succeeded -- the same
        penalty-barrier contract ``bipole_optimize.relax_atoms`` already
        honours. A BFGS unit step in *fractional* coordinates can move an
        atom by a large fraction of a lattice vector.  Before IID 536, a
        legitimate fold-guard refusal at such a trial geometry escaped the
        optimizer as a fatal error.
        A refusal at the *initial* geometry still propagates: there is
        nothing to back off to.
        """
        try:
            return self._run_scf_or_raise(system)
        except (SCFNonConvergence, BipoleFoldUnreliableError):
            if not self._had_good_eval:
                raise
            return BIPOLE_TRIAL_PENALTY_HA, None

    def _run_scf_or_raise(self, system: PeriodicSystem) -> tuple[float, Any]:
        km = self._kmesh
        if self._reference_system is not None:
            km = _kmesh_for_system(km, self._reference_system, system)
        basis = BasisSet(system.unit_cell_molecule(), self._basis_name)

        if self._method == "RHF":
            result = run_pbc_bipole_rhf(system, basis, km, self._opts, progress=False)
        elif self._method == "UHF":
            result = run_pbc_bipole_uhf(system, basis, km, self._opts, progress=False)
        elif self._method == "RKS":
            result = run_pbc_bipole_rks(
                system,
                basis,
                km,
                self._opts,
                functional=self._functional,
                progress=False,
            )
        elif self._method == "UKS":
            result = run_pbc_bipole_uks(
                system,
                basis,
                km,
                self._opts,
                functional=self._functional,
                progress=False,
            )
        else:
            raise ValueError(f"Unknown method: {self._method}")

        if not bool(getattr(result, "converged", True)):
            raise SCFNonConvergence(
                f"Periodic {self._method} SCF did not converge during optimisation"
            )
        self._had_good_eval = True
        return result.energy, result

    def __call__(self, system: PeriodicSystem) -> tuple[float, np.ndarray]:
        """Evaluate (energy, gradient) at *system*.

        Returns gradient in Cartesian Ha/bohr, shape (n_atoms, 3) flat.
        """
        if int(system.dim) < 3:
            # Mirror the public periodic-job gate before recording a reference
            # system or entering an SCF.  DIRECT_TRUNCATED remains available
            # on the low-level drivers for diagnostics, but its cutoff-dependent
            # energy is not a 1-D/2-D Coulomb optimisation objective (IID 542).
            raise NotImplementedError(
                "jk_method='bipole' requires a 3-D periodic system (dim=3); got "
                f"dim={int(system.dim)}. The exact Ewald-J route uses a 3-D "
                "Ewald/Madelung lattice sum that is not a low-dimensional Coulomb "
                "model. For a 1-D wire use jk_method='auto' or 'gdf'; for a 2-D "
                "surface use jk_method='auto'/'slab_ewald_2d', or explicit GDF "
                "for its supported closed-shell slab envelope."
            )
        if self._reference_system is None:
            self._reference_system = system

        e, result = self._run_scf(system)
        if result is None:
            # Penalty region (see _run_scf): zero slope, and no FD gradient --
            # that would be 6N more SCFs on a geometry already refused.
            return e, np.zeros(3 * len(system.unit_cell))
        km = self._kmesh
        if self._reference_system is not None:
            km = _kmesh_for_system(km, self._reference_system, system)

        basis = BasisSet(system.unit_cell_molecule(), self._basis_name)

        if self._force_mode == "fd":
            grad = np.asarray(
                compute_bipole_gradient_fd(
                    system,
                    self._basis_name,
                    km,
                    self._opts,
                    method=self._method,
                    functional=self._functional,
                    step_bohr=self._fd_step_bohr,
                    dft_plus_u=self._dft_plus_u,
                )
            )
        elif self._force_mode == "analytic":
            lattice_opts = getattr(self._opts, "lattice_opts", None)
            kw = {"lattice_opts": lattice_opts}
            if self._dft_plus_u:
                kw["kmesh"] = km
                kw["dft_plus_u"] = self._dft_plus_u
            if self._method == "RHF":
                grad = compute_bipole_gradient_rhf(system, basis, result, **kw)
            elif self._method == "UHF":
                grad = compute_bipole_gradient_uhf(system, basis, result, **kw)
            elif self._method == "RKS":
                grad = compute_bipole_gradient_rks(system, basis, result, **kw)
            elif self._method == "UKS":
                grad = compute_bipole_gradient_uks(system, basis, result, **kw)
            else:
                raise ValueError(f"Unknown method: {self._method}")
            grad = np.asarray(grad)
        else:
            raise ValueError(
                f"force_mode must be 'fd' or 'analytic'; got {self._force_mode!r}"
            )

        return e, grad.ravel()


def _ensure_bloch(kmesh: KMeshInput) -> BlochKMesh:
    """Convert a kmesh input to BlochKMesh."""
    from ..kpoints import as_bloch_kmesh

    return as_bloch_kmesh(kmesh)


# ---------------------------------------------------------------------------
# Fractional Coordinates for atoms
# ---------------------------------------------------------------------------


class FractionalCoordinates(CoordinateRepresentation):
    """Atom positions in fractional (crystal) coordinates.

    Optimiser space is the flat 3N fractional coordinates.
    Cartesian <-> fractional transforms use the lattice vectors.

    Parameters
    ----------
    n_atoms : int
    lattice : (3, 3) ndarray
        Lattice vectors in bohr.
    freeze_indices : sequence of int, optional
    """

    def __init__(
        self,
        n_atoms: int,
        lattice: np.ndarray,
        freeze_indices: Optional[Sequence[int]] = None,
    ):
        self._n_atoms = n_atoms
        self._lattice = np.asarray(lattice, dtype=float)
        self._inv_lattice = np.linalg.inv(self._lattice)
        self._frozen_set: set[int] = (
            {int(i) for i in freeze_indices} if freeze_indices else set()
        )

    @property
    def n_params(self) -> int:
        return 3 * self._n_atoms

    def x0(self, system: PeriodicSystem) -> np.ndarray:
        """Flat fractional coordinates from *system*."""
        frac: list[float] = []
        for atom in system.unit_cell:
            cart = np.asarray(atom.xyz, dtype=float)
            f = self._inv_lattice @ cart
            frac.extend(f)
        return np.array(frac, dtype=float)

    def to_cartesian(self, template: PeriodicSystem, x: np.ndarray) -> PeriodicSystem:
        """Rebuild a PeriodicSystem from flat fractional coordinates."""
        x = np.asarray(x, dtype=float).ravel()
        new_atoms: list[Atom] = []
        for i in range(self._n_atoms):
            frac = x[3 * i : 3 * i + 3]
            cart = self._lattice @ frac
            new_atoms.append(Atom(int(template.unit_cell[i].Z), list(cart)))
        return PeriodicSystem(
            template.dim,
            self._lattice,
            new_atoms,
            charge=template.charge,
            multiplicity=template.multiplicity,
        )

    def project_gradient(
        self, system: PeriodicSystem, cartesian_gradient: np.ndarray
    ) -> np.ndarray:
        """Project Cartesian gradient to fractional: g_frac = L^T g_cart."""
        g = np.asarray(cartesian_gradient, dtype=float).reshape(-1, 3)
        g_frac = np.zeros_like(g)
        for i in range(self._n_atoms):
            g_frac[i] = self._lattice.T @ g[i]
        return g_frac.ravel()

    def project_hessian(
        self, system: PeriodicSystem, cartesian_hessian: np.ndarray
    ) -> np.ndarray:
        """Project Cartesian Hessian to fractional space via chain rule."""
        n = 3 * self._n_atoms
        H_cart = np.asarray(cartesian_hessian, dtype=float)
        J = np.kron(np.eye(self._n_atoms), self._lattice.T)
        return J @ H_cart @ J.T

    @property
    def frozen_set(self) -> set[int]:
        return self._frozen_set


# ---------------------------------------------------------------------------
# Cell Strain Coordinates (Voigt notation, 6-component)
# ---------------------------------------------------------------------------


class CellStrainCoordinates(CoordinateRepresentation):
    """6-component Voigt strain parameters for lattice optimisation.

    The strain vector s = [e_xx, e_yy, e_zz, e_yz, e_xz, e_xy]
    maps to a symmetric strain matrix e via Voigt convention.
    The deformed lattice is L_new = L_ref . (I + e).

    Strain is stored as percentages (e_ij x 100) for numerical
    conditioning of the optimiser. Forward/backward transforms handle
    this scaling transparently.

    Parameters
    ----------
    reference_lattice : (3, 3) ndarray
        Reference lattice vectors in bohr.
    scale_pct : bool
        If True, optimiser space is strain x 100 (default).
    """

    def __init__(self, reference_lattice: np.ndarray, scale_pct: bool = True):
        self._ref_lattice = np.asarray(reference_lattice, dtype=float)
        self._scale_pct = scale_pct
        self._frozen_set: set[int] = set()

    @property
    def n_params(self) -> int:
        return 6

    def x0(self, system: PeriodicSystem) -> np.ndarray:
        """Zero strain at the reference lattice."""
        return np.zeros(6)

    def to_cartesian(self, template: PeriodicSystem, x: np.ndarray) -> PeriodicSystem:
        """Rebuild system with deformed lattice; atoms held fixed."""
        strain = np.asarray(x, dtype=float).ravel()[:6]
        if self._scale_pct:
            strain = strain * 0.01

        # Voigt -> symmetric matrix
        S = np.array(
            [
                [strain[0], strain[5], strain[4]],
                [strain[5], strain[1], strain[3]],
                [strain[4], strain[3], strain[2]],
            ]
        )
        new_lattice = self._ref_lattice @ (np.eye(3) + S)

        return PeriodicSystem(
            template.dim,
            new_lattice,
            list(template.unit_cell),
            charge=template.charge,
            multiplicity=template.multiplicity,
        )

    def project_gradient(
        self, system: PeriodicSystem, cartesian_gradient: np.ndarray
    ) -> np.ndarray:
        """Cell strain has no Cartesian gradient -- return zeros.
        The gradient is computed via FD of energy by the caller."""
        return np.zeros(6)

    @property
    def frozen_set(self) -> set[int]:
        return self._frozen_set


# ---------------------------------------------------------------------------
# Periodic GeomOpt runner -- atoms + cell cycling
# ---------------------------------------------------------------------------


def run_periodic_geomopt(
    system: PeriodicSystem,
    provider: PeriodicSCFProvider,
    *,
    geom_opt: str = "bfgs",
    geom_line_search: str = "backtracking",
    geom_max_iter_atoms: int = 30,
    geom_max_iter_cell: int = 10,
    geom_max_outer: int = 5,
    geom_conv_gmax: float = 1e-4,
    geom_conv_cell_gtol: float = 1e-6,
    relax_cell: bool = True,
    force_mode: str = "fd",
    fd_step_bohr: float = 1e-3,
    record_trajectory: bool = True,
    progress: Any = False,
    **kwargs,
) -> "GeomOptResult":
    """Periodic structure optimisation -- alternating atoms + cell.

    Uses the same geomopt optimizers (sd, cg, bfgs, lbfgs) for atom
    relaxation and Nelder-Mead / FD gradient for cell relaxation,
    cycling until both are stationary.  With ``PeriodicSCFProvider`` this is
    a 3-D-only BIPOLE route; 1-D and 2-D inputs fail on the first provider
    evaluation before SCF.  Custom providers retain their own dimensional
    contracts.

    Parameters
    ----------
    system : PeriodicSystem
        Starting periodic structure.
    provider : PeriodicSCFProvider
        BIPOLE energy+gradient provider.  It refuses ``system.dim < 3``;
        raising a direct-space cutoff cannot turn the diagnostic
        ``DIRECT_TRUNCATED`` path into a low-dimensional Coulomb model.
    geom_opt : str
        Optimizer for atom positions.
    relax_cell : bool
        If True, also relax lattice parameters.
    geom_max_outer : int
        Maximum outer cell+atom cycles.

    Returns
    -------
    GeomOptResult with ``.system`` as a PeriodicSystem.
    """
    progress = resolve_progress(progress)

    from .convergence import ConvergencePolicy
    from .optimizers import _make_objective, _make_x_fn, _progress_print
    from .optimizers import bfgs as _bfgs
    from .state import GeomOptResult

    conv_atoms = ConvergencePolicy(gmax=geom_conv_gmax)
    current = system
    n_atoms = len(list(system.unit_cell))

    total_atom_iters = 0
    total_cell_iters = 0
    total_evals = 0
    converged = False
    final_energy = 0.0

    for outer in range(geom_max_outer):
        # ---- relax atoms --------------------------------------------------------
        coords_atom = FractionalCoordinates(
            n_atoms, np.asarray(current.lattice, dtype=float)
        )

        from .optimizers import resolve_optimizer

        opt_fn = resolve_optimizer(geom_opt)
        atom_result = opt_fn(
            current,
            provider,
            coords=coords_atom,
            max_iter=geom_max_iter_atoms,
            conv=conv_atoms,
            line_search=geom_line_search,
            record_trajectory=False,
            progress=progress,
        )
        current = PeriodicSystem(
            current.dim,
            np.asarray(current.lattice, dtype=float),
            list(atom_result.system.unit_cell),
            charge=current.charge,
            multiplicity=current.multiplicity,
        )
        total_atom_iters += atom_result.n_iter
        total_evals += atom_result.n_energy_evals
        final_energy = atom_result.energy

        if not relax_cell:
            converged = atom_result.converged
            break

        # ---- relax cell --------------------------------------------------------
        from scipy.optimize import minimize

        ref_lattice = np.asarray(current.lattice, dtype=float)
        atoms_fixed = list(current.unit_cell)

        def cell_objective(strain_pct: np.ndarray) -> float:
            strain = strain_pct * 0.01
            S = np.array(
                [
                    [strain[0], strain[5], strain[4]],
                    [strain[5], strain[1], strain[3]],
                    [strain[4], strain[3], strain[2]],
                ]
            )
            new_lattice = ref_lattice @ (np.eye(3) + S)
            sys = PeriodicSystem(
                current.dim,
                new_lattice,
                atoms_fixed,
                charge=current.charge,
                multiplicity=current.multiplicity,
            )
            e, _ = provider._run_scf(sys)
            return e

        # FD gradient of the cell objective
        _CELL_FD = 0.05  # 0.05% strain

        def cell_gradient(x):
            g = np.zeros(6)
            for i in range(6):
                xp, xm = x.copy(), x.copy()
                xp[i] += _CELL_FD
                xm[i] -= _CELL_FD
                g[i] = (cell_objective(xp) - cell_objective(xm)) / (2.0 * _CELL_FD)
            return g

        cell_res = minimize(
            cell_objective,
            np.zeros(6),
            method="L-BFGS-B",
            jac=cell_gradient,
            options={"maxiter": geom_max_iter_cell, "gtol": geom_conv_cell_gtol},
        )
        total_evals += (
            cell_res.nfev + cell_res.njev * 6 if hasattr(cell_res, "njev") else 0
        )
        total_cell_iters += int(cell_res.nit)

        # Build optimized system with relaxed cell
        strain_opt = cell_res.x * 0.01
        S = np.array(
            [
                [strain_opt[0], strain_opt[5], strain_opt[4]],
                [strain_opt[5], strain_opt[1], strain_opt[3]],
                [strain_opt[4], strain_opt[3], strain_opt[2]],
            ]
        )
        opt_lattice = ref_lattice @ (np.eye(3) + S)
        current = PeriodicSystem(
            current.dim,
            opt_lattice,
            atoms_fixed,
            charge=current.charge,
            multiplicity=current.multiplicity,
        )
        final_energy = float(cell_res.fun)

        # Update provider reference for kmesh remapping
        if hasattr(provider, "_reference_system"):
            # kmesh remapping happens inside _run_scf
            pass

        cell_converged, _ = _gradient_converged(
            bool(cell_res.success), cell_res.jac, geom_conv_cell_gtol
        )

        if progress:
            _progress_print(
                progress,
                f"  Outer {outer + 1}/{geom_max_outer}: "
                f"atoms {atom_result.n_iter} iter, cell {int(cell_res.nit)} iter, "
                f"E = {render_energy_labeled(final_energy, width=0, precision=8)}",
            )

        if atom_result.converged and cell_converged:
            converged = True
            break

    return GeomOptResult(
        system=current,
        energy=final_energy,
        gradient=np.array([]),
        converged=converged,
        n_iter=total_atom_iters + total_cell_iters,
        n_energy_evals=total_evals,
        optimizer=geom_opt,
        message="converged" if converged else "max_outer reached",
    )


# Re-export GeomOptResult for internal use
from .state import GeomOptResult  # noqa: E402, F811
