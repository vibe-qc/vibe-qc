"""BIPOLE structure optimization.

Atomic position relaxation wrapped in scipy L-BFGS-B. The historical
variable-cell entry points remain importable for API compatibility but fail
closed: their strain convention and coupled atom/cell convergence have not
been certified against a single terminal geometry.

Atomic forces default to finite differences of the BIPOLE energy
(``force_mode="fd"``). This is deliberate: the analytic BIPOLE gradient
(:mod:`vibeqc.bipole_gradient`) is still a research-preview surface:
RHF/UHF Γ and maintained KS Γ cases are pinned, but broader KS,
multi-k, finite-temperature, and meta-GGA certification remain open.
The FD path differentiates the real total energy and is correct by
construction (cost: ~6N SCFs per gradient eval). Pass
``force_mode="analytic"`` only for research on the analytic gradient
itself -- a warning is emitted and FD remains the production optimizer
default.

Usage:
    from vibeqc.bipole_optimize import relax_atoms, relax_cell, relax_full

    # Atomic relaxation only (FD forces by default)
    result = relax_atoms(system, basis, kmesh, method="RHF")

Variable-cell work must use a route with a certified stress and coupled
optimizer; the BIPOLE entry points do not currently provide one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .kpoints import KPoints

from ._vibeqc_core import (
    Atom,
    BasisSet,
    BlochKMesh,
    Functional,
    InitialGuess,
    LatticeSumOptions,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    PeriodicSystem,
    bloch_kmesh_from_lists,
    monkhorst_pack,
)
from .bipole_gradient import (
    compute_bipole_gradient_fd,
    compute_bipole_gradient_rhf,
    compute_bipole_gradient_rks,
    compute_bipole_gradient_uhf,
    compute_bipole_gradient_uks,
)
from .pbc_bipole import run_pbc_bipole_rhf
from .pbc_bipole_common import (
    BIPOLE_TRIAL_PENALTY_HA,
    BipoleFoldUnreliableError,
    reject_bipole_ecp_options,
    reject_bipole_lone_non_gamma_kpoint,
    reject_bipole_solver_options,
    validate_bipole_kmesh,
)
from .pbc_bipole_rks import run_pbc_bipole_rks
from .pbc_bipole_uhf import run_pbc_bipole_uhf
from .pbc_bipole_uks import run_pbc_bipole_uks
from .output import write

__all__ = [
    "relax_atoms",
    "relax_cell",
    "relax_cell_gradient",
    "relax_full",
    "OptimizeResult",
]

KMeshInput = Union[BlochKMesh, "KPoints"]


def _reject_variable_cell_optimization() -> None:
    """Fail closed until BIPOLE cell derivatives and convergence are valid."""
    raise NotImplementedError(
        "BIPOLE variable-cell optimization is unavailable. The historical "
        "cell strain convention and coupled atom/cell convergence were not "
        "certified on one terminal geometry. Use relax_atoms for fixed-cell "
        "optimization."
    )


def _as_bloch_kmesh(kmesh: KMeshInput) -> BlochKMesh:
    from .kpoints import as_bloch_kmesh

    return as_bloch_kmesh(kmesh)


def _reject_nonstationary_optimizer_kmesh(kmesh: BlochKMesh) -> Tuple[bool, bool]:
    """Require a Gamma or complete MP/IBZ mesh for geometry derivatives."""
    _, _, true_multik, is_ibz = validate_bipole_kmesh(
        kmesh,
        driver="relax_atoms",
        require_complete=True,
    )
    reject_bipole_lone_non_gamma_kpoint(kmesh, driver="relax_atoms")
    return true_multik, is_ibz

# Central-difference half-step (in % strain) for the cell-strain gradient.
# 0.05 % => ~5e-4 strain, well inside the SCF energy's smooth regime.
_CELL_FD_STRAIN_PCT = 0.05

_SCF_OPTION_FIELDS = (
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
    "diis_restart_tau",
    "diis_adaptive_delta",
    "scf_accelerator",
    "ediis_diis_switch_threshold",
    "initial_guess",
    "level_shift",
    "level_shift_warmup_cycles",
    "level_shift_schedule",
    "smearing_temperature",
    "quadratic_fallback_iter",
    "quadratic_fallback_shift",
    "quadratic_fallback_max_step",
    "use_periodic_becke",
    "becke_image_radius_bohr",
    "atomic_spins",
    "spinlock_mode",
    "spinlock_value",
    "spinlock_iterations",
)

_LATTICE_OPTION_FIELDS = (
    "cutoff_bohr",
    "nuclear_cutoff_bohr",
    "coulomb_method",
    "screening_overlap_threshold",
    "screening_exchange_threshold",
    "schwarz_threshold",
    "schwarz_threshold_forces",
    "sr_range_screening",
    "sr_sparse_traversal",
    "pair_complete_1e",
    "eri_interaction_cutoff_bohr",
)


def _central_fd_gradient(energy_fn, x0, h: float) -> np.ndarray:
    """Central finite-difference gradient ``d(energy_fn)/dx`` at ``x0``.

    A descent-consistent gradient: it is the finite difference of the scalar
    ``energy_fn`` being minimised, so ``-g`` is a descent direction when the
    step resolves the local slope. Retained for the dormant variable-cell
    implementation; the public cell entry points fail closed before using it.
    """
    x0 = np.asarray(x0, dtype=float)
    g = np.zeros(len(x0))
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        g[i] = (energy_fn(xp) - energy_fn(xm)) / (2.0 * h)
    return g


def _copy_lattice_options(src) -> LatticeSumOptions:
    """Return a fresh lattice-options object with the public fields copied."""
    out = LatticeSumOptions()
    for name in _LATTICE_OPTION_FIELDS:
        if hasattr(src, name):
            setattr(out, name, getattr(src, name))
    return out


def _make_scf_options(
    method_upper: str,
    *,
    cutoff_bohr: float,
    scf_conv_tol: Optional[float],
    scf_max_iter: Optional[int],
    nuclear_cutoff_bohr: Optional[float] = None,
    scf_options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
):
    """Build optimizer-local SCF options without mutating caller state."""
    if scf_options is not None:
        # ECP metadata is intentionally not copied into the optimizer-local
        # all-electron option surface. Reject it before that copy can erase
        # the evidence and let a displaced calculation change Hamiltonian.
        reject_bipole_ecp_options(scf_options, driver="BIPOLE optimizer")
        reject_bipole_solver_options(scf_options, driver="BIPOLE optimizer")
        if getattr(scf_options, "initial_guess", None) == InitialGuess.READ:
            raise NotImplementedError(
                "BIPOLE optimizer: InitialGuess.READ cannot be preserved "
                "across displaced geometries. Run the restart single point "
                "first, then optimize from a geometry-defined guess."
            )
    is_ks = method_upper in ("RKS", "UKS")
    opts = PeriodicKSOptions() if is_ks else PeriodicRHFOptions()

    if scf_options is None:
        opts.max_iter = 50
        opts.use_diis = True
        opts.conv_tol_energy = 1e-7
    else:
        for name in _SCF_OPTION_FIELDS:
            if hasattr(scf_options, name) and hasattr(opts, name):
                setattr(opts, name, getattr(scf_options, name))
        if hasattr(scf_options, "lattice_opts"):
            opts.lattice_opts = _copy_lattice_options(scf_options.lattice_opts)

    if scf_conv_tol is not None:
        opts.conv_tol_energy = float(scf_conv_tol)
    if scf_max_iter is not None:
        opts.max_iter = int(scf_max_iter)
    opts.lattice_opts.cutoff_bohr = float(cutoff_bohr)
    opts.lattice_opts.nuclear_cutoff_bohr = float(
        cutoff_bohr
        if nuclear_cutoff_bohr is None
        else nuclear_cutoff_bohr
    )
    return opts


def _kmesh_for_system(kmesh, reference_system: PeriodicSystem, target_system: PeriodicSystem):
    """Return ``kmesh`` represented on ``target_system``'s reciprocal lattice.

    ``BlochKMesh.kpoints`` are Cartesian vectors. Atomic relaxations can reuse
    them because the lattice is fixed. The dormant variable-cell implementation
    needs the same fractional Monkhorst-Pack or explicit k-list converted to
    each target reciprocal basis. Public cell entry points fail closed before
    reaching this helper.
    """
    if np.allclose(
        np.asarray(reference_system.lattice, dtype=float),
        np.asarray(target_system.lattice, dtype=float),
        atol=1e-14,
        rtol=0.0,
    ):
        return kmesh

    kpts = np.asarray(
        getattr(kmesh, "kpoints", getattr(kmesh, "kpoints_cart", [])),
        dtype=float,
    ).reshape(-1, 3)
    if kpts.size == 0:
        raise ValueError("_kmesh_for_system: kmesh has no k-points")

    weights = np.asarray(getattr(kmesh, "weights", []), dtype=float).reshape(-1)
    if weights.size == 0:
        weights = np.full(kpts.shape[0], 1.0 / float(kpts.shape[0]))
    if weights.shape != (kpts.shape[0],):
        raise ValueError(
            "_kmesh_for_system: kpoint/weight size mismatch "
            f"({kpts.shape[0]} k-points, {weights.shape[0]} weights)"
        )

    mesh_raw = getattr(kmesh, "mesh", None)
    shift_raw = getattr(kmesh, "is_shift", getattr(kmesh, "shift", None))
    mesh = tuple(int(x) for x in mesh_raw) if mesh_raw is not None else ()
    shift = tuple(int(x) for x in shift_raw) if shift_raw is not None else ()
    ir_mapping = np.asarray(getattr(kmesh, "ir_mapping", []), dtype=int).reshape(-1)

    if len(mesh) == 3:
        shift = shift if len(shift) == 3 else (0, 0, 0)
        full_n = int(mesh[0] * mesh[1] * mesh[2])
        is_ibz = ir_mapping.size == full_n and kpts.shape[0] != full_n
        is_full_mp = False
        if kpts.shape[0] == full_n:
            try:
                ref_full = monkhorst_pack(
                    reference_system,
                    list(mesh),
                    list(shift),
                    False,
                )
                ref_kpts = np.asarray(ref_full.kpoints, dtype=float).reshape(-1, 3)
                ref_weights = np.asarray(ref_full.weights, dtype=float).reshape(-1)
                is_full_mp = (
                    ref_kpts.shape == kpts.shape
                    and np.allclose(ref_kpts, kpts, atol=1e-12)
                    and np.allclose(ref_weights, weights, atol=1e-12)
                )
            except Exception:
                is_full_mp = False
        if is_full_mp or is_ibz:
            return monkhorst_pack(target_system, list(mesh), list(shift), False)

    B_ref = np.asarray(reference_system.reciprocal_lattice(), dtype=float)
    B_target = np.asarray(target_system.reciprocal_lattice(), dtype=float)
    k_frac = np.linalg.solve(B_ref, kpts.T).T
    k_target = (B_target @ k_frac.T).T
    return bloch_kmesh_from_lists(
        [np.asarray(k, dtype=float) for k in k_target],
        [float(w) for w in weights],
    )


class SCFNonConvergence(RuntimeError):
    """An SCF inside an optimization did not converge.

    Raised by :func:`_run_scf` so optimizer objectives can distinguish
    "this trial geometry's SCF failed" (recoverable: return a penalty
    barrier so the line search backs off) from genuine programming
    errors. A non-converged energy is never *used* as a real objective
    value -- it is replaced by the barrier."""


class OptimizeResult:
    """Container for optimization results."""

    def __init__(self, system, energy, gradient, n_iter, converged):
        self.system = system
        self.energy = energy
        self.gradient = gradient
        self.n_iter = n_iter
        self.converged = converged


def _run_scf(system, basis, kmesh, opts, method, functional, **kwargs):
    """Run one BIPOLE SCF and return its variational objective and result.

    Finite-temperature gradients differentiate the Mermin free energy, so a
    smeared optimization must line-search that same quantity rather than the
    bare internal energy.
    """
    if method == "RHF":
        result = run_pbc_bipole_rhf(
            system,
            basis,
            kmesh,
            opts,
            progress=False,
            **kwargs,
        )
    elif method == "UHF":
        result = run_pbc_bipole_uhf(
            system,
            basis,
            kmesh,
            opts,
            progress=False,
            **kwargs,
        )
    elif method == "RKS":
        result = run_pbc_bipole_rks(
            system,
            basis,
            kmesh,
            opts,
            functional=functional,
            progress=False,
            **kwargs,
        )
    elif method == "UKS":
        result = run_pbc_bipole_uks(
            system,
            basis,
            kmesh,
            opts,
            functional=functional,
            progress=False,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    if not bool(getattr(result, "converged", True)):
        n_iter = getattr(result, "n_iter", "unknown")
        energy = getattr(result, "energy", None)
        energy_txt = "unknown" if energy is None else f"{float(energy):.12g}"
        raise SCFNonConvergence(
            "bipole_optimize: "
            f"{method} SCF did not converge during optimization "
            f"(n_iter={n_iter}, energy={energy_txt}). Refusing to optimize "
            "against a non-converged energy; increase scf_max_iter, loosen "
            "scf_conv_tol, or adjust the SCF accelerator/smearing options."
        )
    objective = result.energy
    if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
        objective = getattr(result, "free_energy", objective)
    return float(objective), result


def _compute_gradient(
    system, basis, result, method, lattice_opts,
    *, kmesh=None, dft_plus_u=None,
):
    """Compute the BIPOLE gradient for any method.

    When ``dft_plus_u`` is set, the multi-k +U Pulay overlap-derivative
    contribution is added; ``kmesh`` must be the same
    :class:`BlochKMesh` the SCF was run on.
    """
    kwargs = {"lattice_opts": lattice_opts, "kmesh": kmesh}
    if dft_plus_u:
        kwargs["dft_plus_u"] = dft_plus_u
    if method == "RHF":
        return compute_bipole_gradient_rhf(system, basis, result, **kwargs)
    elif method == "UHF":
        return compute_bipole_gradient_uhf(system, basis, result, **kwargs)
    elif method == "RKS":
        return compute_bipole_gradient_rks(system, basis, result, **kwargs)
    elif method == "UKS":
        return compute_bipole_gradient_uks(system, basis, result, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_forces(
    system, basis, result, method, lattice_opts,
    *, kmesh, basis_name, opts, functional, bipole_kwargs,
    force_mode, fd_step_bohr,
):
    """Cartesian BIPOLE gradient ``(n_atoms, 3)`` in Ha/bohr.

    ``force_mode="fd"`` (default) uses the exact finite-difference
    gradient :func:`compute_bipole_gradient_fd` (differentiates the real
    total energy -- correct by construction; +U is handled automatically).
    ``force_mode="analytic"`` uses the research-preview analytic gradient
    surface (emits a warning; see :mod:`vibeqc.bipole_gradient` for the
    currently maintained cases).
    """
    if force_mode == "fd":
        return np.asarray(
            compute_bipole_gradient_fd(
                system, basis_name, kmesh, opts,
                method=method, functional=functional,
                step_bohr=fd_step_bohr, **bipole_kwargs,
            )
        )
    if force_mode == "analytic":
        return _compute_gradient(
            system, basis, result, method, lattice_opts,
            kmesh=kmesh, dft_plus_u=bipole_kwargs.get("dft_plus_u"),
        )
    raise ValueError(
        f"force_mode must be 'fd' or 'analytic'; got {force_mode!r}"
    )


def _atoms_to_flat(system: PeriodicSystem) -> np.ndarray:
    """Flatten atomic positions to a 1D array (fractional coordinates)."""
    lattice = np.asarray(system.lattice, dtype=float)
    inv_lat = np.linalg.inv(lattice)
    frac = []
    for atom in system.unit_cell:
        frac.extend(inv_lat @ np.asarray(atom.xyz, dtype=float))
    return np.array(frac, dtype=float)


def _flat_to_system(
    system_template: PeriodicSystem,
    x: np.ndarray,
) -> PeriodicSystem:
    """Rebuild system from flat fractional coordinates."""
    lattice = np.asarray(system_template.lattice, dtype=float)
    n_atoms = len(system_template.unit_cell)
    new_atoms = []
    for i in range(n_atoms):
        frac = x[3 * i : 3 * i + 3]
        cart = lattice @ frac
        new_atoms.append(
            Atom(int(system_template.unit_cell[i].Z), list(cart)),
        )
    return PeriodicSystem(
        system_template.dim,
        lattice,
        new_atoms,
        charge=system_template.charge,
        multiplicity=system_template.multiplicity,
    )


def _emit_opt_line(text: str) -> None:
    """Emit one already-formatted optimizer summary line to the job channel.

    The public ``progress=`` optimizer arguments remain compatibility no-ops;
    callers install an output channel instead of passing a stream.
    """
    write(f"{text}\n")


def relax_atoms(
    system: PeriodicSystem,
    basis_name: str,
    kmesh: KMeshInput,
    method: str = "RHF",
    *,
    functional: Optional[str] = None,
    force_mode: str = "fd",
    fd_step_bohr: float = 1e-3,
    max_iter: int = 30,
    conv_tol_grad: float = 1e-4,
    scf_conv_tol: Optional[float] = None,
    scf_max_iter: Optional[int] = None,
    scf_options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    cutoff_bohr: float = 8.0,
    nuclear_cutoff_bohr: Optional[float] = None,
    sr_image_precision: Optional[float] = 1e-6,
    freeze_indices: Optional[Sequence[int]] = None,
    output_trajectory: Optional["str | Path"] = None,
    progress: Any = None,
    **bipole_kwargs,
) -> OptimizeResult:
    """Relax atomic positions with L-BFGS-B and BIPOLE forces.

    Forces default to the exact finite-difference gradient
    (``force_mode="fd"``); see the module docstring for why the analytic
    gradient is not used by default.

    Parameters
    ----------
    system : PeriodicSystem
        Initial geometry (lattice fixed).
    basis_name : str
        Basis set name (rebuild per geometry step).
    kmesh : BlochKMesh or KPoints
        k-point mesh.
    method : str
        "RHF", "UHF", "RKS", or "UKS".
    functional : str, optional
        XC functional for RKS/UKS.
    force_mode : str
        ``"fd"`` (default) for the exact finite-difference gradient, or
        ``"analytic"`` for the research-preview analytic gradient (emits a
        warning; maintained cases are documented in
        :mod:`vibeqc.bipole_gradient`).
    fd_step_bohr : float
        Central-difference half-step (bohr) used when ``force_mode="fd"``.
    max_iter : int
        Maximum optimization steps.
    conv_tol_grad : float
        Gradient convergence tolerance (Ha/bohr).
    scf_conv_tol : float
        Optional SCF energy convergence tolerance override. Defaults to the
        supplied ``scf_options`` value, or ``1e-7`` when no options are supplied.
    scf_max_iter : int, optional
        Optional SCF iteration limit override. Defaults to the supplied
        ``scf_options`` value, or ``50`` when no options are supplied.
    scf_options : PeriodicRHFOptions / PeriodicKSOptions, optional
        SCF controls to copy into every geometry point (smearing, FMIXING,
        level shift, accelerator, grid, etc.). The object is copied before the
        optimizer sets its lattice cutoffs, so caller state is not mutated.
    cutoff_bohr : float
        Lattice cutoff for integrals.
    sr_image_precision : float or None
        Precision for the production BIPOLE padded ket-image domain. The
        default ``1e-6`` is forwarded to every objective and finite-difference
        SCF. ``None`` is the explicit historical-domain diagnostic opt-out;
        it is appropriate for bounded-cost dispatch/QVF tests whose numerical
        accuracy is not under test, not for production relaxation results.
    freeze_indices : sequence of int, optional
        Atom indices (into ``system.unit_cell``) to hold fixed during
        the relaxation. The standard surface-catalysis pattern: pass
        ``SlabInfo.bottom_layer_indices(n)`` from :func:`vibeqc.build.slab`
        to freeze the bottom N layers of a slab. The SCF + gradient
        still see every atom; the optimizer simply zeros the gradient
        components on the frozen atoms so their positions never move.
    output_trajectory : str or Path, optional
        Path stem (``.qvf`` appended automatically). When set, the
        relaxation collects a per-step (geometry, energy) frame via
        scipy's ``callback=`` and writes a vibe-view-renderable QVF
        archive on exit. Each frame is one accepted L-BFGS-B step;
        the initial geometry is frame 0 and the converged geometry is
        the last frame. Periodic systems ship as QVF v2 with the
        per-frame lattice + dim attached, so vibe-view renders the
        cell + wraps atoms across periodic boundaries (see
        ``docs/user_guide/vibe_view.md`` Sec. "Periodic reaction paths").
        Default ``None`` => no trajectory output (no overhead).

    Returns
    -------
    OptimizeResult
    """
    if int(system.dim) != 3:
        raise NotImplementedError(
            "relax_atoms: the maintained BIPOLE optimization objective is "
            "available only for 3-D periodic systems. Use the maintained "
            "low-dimensional route for a fixed-geometry calculation; a "
            "same-Hamiltonian 1-D/2-D optimizer is not implemented."
        )
    kmesh = _as_bloch_kmesh(kmesh)
    true_multik, is_ibz = _reject_nonstationary_optimizer_kmesh(kmesh)
    if true_multik and (
        bipole_kwargs.get("use_ewald_j_split") is False
        or bipole_kwargs.get("use_exchange_ewald_split") is False
    ):
        raise NotImplementedError(
            "relax_atoms: multi-k BIPOLE optimization requires the corrected "
            "Ewald J/exchange split. The legacy gauge has a density-dependent "
            "energy term absent from its Fock and is not a stationary "
            "geometry objective."
        )
    if is_ibz:
        kmesh = monkhorst_pack(
            system,
            list(kmesh.mesh),
            list(kmesh.is_shift),
            False,
        )
    method_upper = method.upper()
    opts = _make_scf_options(
        method_upper,
        cutoff_bohr=cutoff_bohr,
        nuclear_cutoff_bohr=nuclear_cutoff_bohr,
        scf_conv_tol=scf_conv_tol,
        scf_max_iter=scf_max_iter,
        scf_options=scf_options,
    )
    if float(getattr(opts, "smearing_temperature", 0.0) or 0.0) > 0.0:
        raise NotImplementedError(
            "relax_atoms: finite-temperature BIPOLE optimization is "
            "unavailable until OptimizeResult and trajectory/QVF artifacts "
            "distinguish the Mermin free-energy objective from internal "
            "energy. Run a fixed-geometry smeared SCF or optimize at T=0."
        )
    force_mode = str(force_mode).strip().lower()
    if force_mode not in ("fd", "analytic"):
        raise ValueError(
            "relax_atoms: force_mode must be 'fd' or 'analytic'; "
            f"got {force_mode!r}"
        )
    bipole_kwargs = dict(bipole_kwargs)
    bipole_kwargs["sr_image_precision"] = sr_image_precision

    history: List[Tuple[float, float]] = []

    n_atoms_total = len(system.unit_cell)
    if freeze_indices is None:
        frozen_set: set[int] = set()
    else:
        frozen_set = {int(i) for i in freeze_indices}
        bad = [i for i in frozen_set
               if i < 0 or i >= n_atoms_total]
        if bad:
            raise ValueError(
                f"relax_atoms: freeze_indices {bad} out of range "
                f"[0, {n_atoms_total - 1}]"
            )
    # Bound matrix: shape (3 * n_atoms, ) with (fixed,fixed) bounds for frozen
    # entries (in fractional coords) so L-BFGS-B literally cannot move
    # them. Free atoms use (-inf, inf).
    bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None
    if frozen_set:
        bounds = []
        x0_full = _atoms_to_flat(system)
        for atom_i in range(n_atoms_total):
            if atom_i in frozen_set:
                for k in range(3):
                    fixed = float(x0_full[3 * atom_i + k])
                    bounds.append((fixed, fixed))
            else:
                for _ in range(3):
                    bounds.append((None, None))

    # L-BFGS-B optimizes in *fractional* coordinates, so its first unit
    # trial step in a large box moves atoms by ~one lattice vector -- a
    # geometry whose SCF legitimately may not converge, or whose lattice
    # fold the BIPOLE support preflight legitimately refuses. A line-search
    # probe failing either way is recoverable: return a finite penalty
    # barrier so the line search backs off (the non-converged energy itself
    # is never used). Only a failure at the *initial* geometry -- where
    # there is nothing to back off to -- aborts the optimization.
    # Shared with the geomopt provider (pbc_bipole_common) so the two
    # periodic optimizers cannot disagree on what a refused trial costs.
    _PENALTY_HA = BIPOLE_TRIAL_PENALTY_HA
    _had_good_eval: dict[str, bool] = {"ok": False}

    def objective(x: np.ndarray) -> float:
        sys = _flat_to_system(system, x)
        basis = BasisSet(sys.unit_cell_molecule(), basis_name)
        try:
            e, res = _run_scf(
                sys, basis, kmesh, opts, method_upper, functional, **bipole_kwargs
            )
        except (SCFNonConvergence, BipoleFoldUnreliableError):
            if not _had_good_eval["ok"]:
                raise
            history.append((_PENALTY_HA, 0.0))
            return _PENALTY_HA
        _had_good_eval["ok"] = True
        history.append((e, 0.0))
        return e

    def gradient(x: np.ndarray) -> np.ndarray:
        sys = _flat_to_system(system, x)
        basis = BasisSet(sys.unit_cell_molecule(), basis_name)
        try:
            e, res = _run_scf(
                sys, basis, kmesh, opts, method_upper, functional, **bipole_kwargs
            )
        except (SCFNonConvergence, BipoleFoldUnreliableError):
            if not _had_good_eval["ok"]:
                raise
            # Penalty region: zero slope + the barrier value recorded by
            # objective() force the line search to shrink the step.
            history.append((_PENALTY_HA, 0.0))
            return np.zeros(3 * len(system.unit_cell))
        _had_good_eval["ok"] = True
        # Convert Cartesian gradient to fractional gradient. Forces use
        # the FD path by default (analytic is a research preview). FD
        # differentiates the real energy, so dft_plus_u is handled
        # automatically; the analytic branch threads it explicitly.
        lattice = np.asarray(sys.lattice, dtype=float)
        grad_cart = _compute_forces(
            sys, basis, res, method_upper, opts.lattice_opts,
            kmesh=kmesh, basis_name=basis_name, opts=opts,
            functional=functional, bipole_kwargs=bipole_kwargs,
            force_mode=force_mode, fd_step_bohr=fd_step_bohr,
        )
        # Zero the Cartesian gradient on frozen atoms so the reported
        # |grad| reflects only the free degrees of freedom (the bounds
        # already keep frozen positions in place; this just keeps the
        # convergence metric honest).
        if frozen_set:
            for a in frozen_set:
                grad_cart[a, :] = 0.0
        # dE/d(frac) = lattice^T . dE/d(cart)
        grad_frac = np.zeros_like(grad_cart)
        for a in range(len(sys.unit_cell)):
            grad_frac[a, :] = lattice.T @ grad_cart[a, :]
        history.append((e, float(np.linalg.norm(grad_cart))))
        return grad_frac.ravel()

    # Optional per-step trajectory capture. scipy's L-BFGS-B fires
    # callback(x) after each accepted step -- exactly one frame per
    # step, with the initial geometry recorded up front.
    trajectory_frames: list[PeriodicSystem] = []
    trajectory_energies: list[float] = []
    if output_trajectory is not None:
        trajectory_frames.append(system)
        # Energy of the initial frame; cheap because the first
        # objective() call evaluates exactly this geometry.

    last_step_energy: dict[str, float] = {}

    def _wrapped_objective(x: np.ndarray) -> float:
        e = objective(x)
        last_step_energy["value"] = e
        return e

    def _capture(x: np.ndarray) -> None:
        if output_trajectory is None:
            return
        trajectory_frames.append(_flat_to_system(system, x))
        trajectory_energies.append(
            last_step_energy.get("value", float("nan"))
        )

    x0 = _atoms_to_flat(system)
    res = minimize(
        _wrapped_objective if output_trajectory is not None else objective,
        x0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={"maxiter": max_iter, "gtol": conv_tol_grad},
        callback=_capture if output_trajectory is not None else None,
    )

    sys_opt = _flat_to_system(system, res.x)
    # Independent gradient gate: scipy reports res.success on EITHER the
    # gtol OR the (default) ftol criterion, so a success flag alone can
    # claim convergence at a non-stationary geometry. Require the actual
    # max-component (fractional) gradient to meet conv_tol_grad
    # (2026-05-31 audit, F1; same gate as optimize_molecule).
    from .molecular_optimize import _gradient_converged

    converged, grad_max = _gradient_converged(
        bool(res.success), res.jac, conv_tol_grad
    )
    _emit_opt_line(
        f"\nAtomic relaxation: {res.nit} iters, "
        f"E = {res.fun:.8f} Ha, "
        f"max|grad| = {grad_max:.4e}, "
        f"converged={converged}",
    )

    if output_trajectory is not None:
        # The initial frame's energy isn't captured by callback (which
        # fires after the first step). Patch it in by evaluating the
        # first-frame energy from the first objective call -- the
        # history list captures that.
        if history and len(trajectory_energies) < len(trajectory_frames):
            trajectory_energies.insert(0, history[0][0])
        # If the optimizer terminated cleanly, the last accepted x is
        # res.x -- append it if callback didn't catch it (e.g.
        # converged-on-step-0).
        if len(trajectory_frames) == 1:
            trajectory_frames.append(sys_opt)
            trajectory_energies.append(float(res.fun))
        from .output.formats.qvf import write_reaction_path_qvf

        n = len(trajectory_frames)
        waypoints: list[dict[str, Any]] = [
            {
                "frame_index": 0,
                "label": "start",
                "kind": "reactant",
                "energy_eh": float(trajectory_energies[0]),
            },
            {
                "frame_index": n - 1,
                "label": "converged" if converged else "stopped",
                "kind": "product",
                "energy_eh": float(trajectory_energies[-1]),
            },
        ]
        # Reaction coordinate = step index normalised to [0, 1].
        rc = [i / max(1, n - 1) for i in range(n)]
        write_reaction_path_qvf(
            output_trajectory,
            frames=trajectory_frames,
            energies=trajectory_energies,
            waypoints=waypoints,
            reaction_coordinate=rc,
            method=method_upper,
            basis=basis_name,
            functional=functional,
        )

    return OptimizeResult(sys_opt, res.fun, res.jac, res.nit, converged)


def relax_cell(
    system: PeriodicSystem,
    basis_name: str,
    kmesh: KMeshInput,
    method: str = "RHF",
    *,
    functional: Optional[str] = None,
    max_iter: int = 20,
    scf_conv_tol: Optional[float] = None,
    scf_max_iter: Optional[int] = None,
    scf_options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    cutoff_bohr: float = 8.0,
    nuclear_cutoff_bohr: Optional[float] = None,
    progress: Any = None,
    **bipole_kwargs,
) -> OptimizeResult:
    """Reserved variable-cell entry point; currently fails closed.

    The historical implementation optimized the 6 independent lattice strain
    components. It is unavailable until strain convention and same-geometry
    atom/cell convergence are certified. Use :func:`relax_atoms` for supported
    fixed-cell atomic relaxation; :func:`relax_full` also fails closed.

    Parameters
    ----------
    system, basis_name, kmesh, method, functional
        As in relax_atoms.
    max_iter : int
        Maximum Nelder-Mead iterations.
    scf_conv_tol, scf_max_iter, scf_options
        As in :func:`relax_atoms`.
    """
    _reject_variable_cell_optimization()
    kmesh = _as_bloch_kmesh(kmesh)
    method_upper = method.upper()
    opts = _make_scf_options(
        method_upper,
        cutoff_bohr=cutoff_bohr,
        nuclear_cutoff_bohr=nuclear_cutoff_bohr,
        scf_conv_tol=scf_conv_tol,
        scf_max_iter=scf_max_iter,
        scf_options=scf_options,
    )

    ref_lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)

    def objective(strain: np.ndarray) -> float:
        # strain is [e_xx, e_yy, e_zz, e_yz, e_xz, e_xy] in Voigt notation
        e_xx, e_yy, e_zz, e_yz, e_xz, e_xy = strain
        strain_matrix = (
            np.array(
                [
                    [e_xx, e_xy, e_xz],
                    [e_xy, e_yy, e_yz],
                    [e_xz, e_yz, e_zz],
                ]
            )
            * 0.01
        )  # scale to percent
        new_lattice = ref_lattice @ (np.eye(3) + strain_matrix)
        sys = PeriodicSystem(
            system.dim,
            new_lattice,
            atoms,
            charge=system.charge,
            multiplicity=system.multiplicity,
        )
        basis = BasisSet(sys.unit_cell_molecule(), basis_name)
        km = _kmesh_for_system(kmesh, system, sys)
        e, _ = _run_scf(
            sys, basis, km, opts, method_upper, functional, **bipole_kwargs
        )
        return e

    res = minimize(
        objective,
        np.zeros(6),
        method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 0.01, "fatol": 1e-6},
    )

    # Build optimized system
    e_xx, e_yy, e_zz, e_yz, e_xz, e_xy = res.x
    strain_matrix = (
        np.array(
            [
                [e_xx, e_xy, e_xz],
                [e_xy, e_yy, e_yz],
                [e_xz, e_yz, e_zz],
            ]
        )
        * 0.01
    )
    opt_lattice = ref_lattice @ (np.eye(3) + strain_matrix)
    sys_opt = PeriodicSystem(
        system.dim,
        opt_lattice,
        atoms,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )

    _emit_opt_line(
        f"\nCell relaxation: {res.nit} iters, "
        f"E = {res.fun:.8f} Ha, converged={res.success}",
    )
    _emit_opt_line("  Lattice vectors (bohr):")
    for i in range(3):
        _emit_opt_line(
            f"    a{i + 1} = [{opt_lattice[i, 0]:10.6f} {opt_lattice[i, 1]:10.6f} {opt_lattice[i, 2]:10.6f}]",
        )

    return OptimizeResult(sys_opt, res.fun, None, res.nit, res.success)


def relax_cell_gradient(
    system: PeriodicSystem,
    basis_name: str,
    kmesh: KMeshInput,
    method: str = "RHF",
    *,
    functional: Optional[str] = None,
    force_mode: str = "fd",
    fd_step_bohr: float = 1e-3,
    max_iter: int = 20,
    scf_conv_tol: Optional[float] = None,
    scf_max_iter: Optional[int] = None,
    scf_options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    cutoff_bohr: float = 8.0,
    nuclear_cutoff_bohr: Optional[float] = None,
    progress: Any = None,
    **bipole_kwargs,
) -> OptimizeResult:
    """Reserved gradient cell entry point; currently fails closed.

    The historical implementation used a central finite difference of the SCF
    energy with respect to each strain component. Before 2026-06-05 it used
    the force virial ``compute_stress_tensor``, which is wrong for this
    objective: it assumes the atoms scale with strain and misses the explicit
    lattice/Ewald plus Gaussian Pulay stress. This entry point and
    :func:`relax_cell` now both fail closed. ``force_mode`` and
    ``fd_step_bohr`` remain accepted for API compatibility but are unused.

    ``scf_options`` is copied into every strained geometry so smearing,
    FMIXING, level-shift, accelerator, and grid controls match the energy
    route being optimized.
    """
    _reject_variable_cell_optimization()
    kmesh = _as_bloch_kmesh(kmesh)
    method_upper = method.upper()
    opts = _make_scf_options(
        method_upper,
        cutoff_bohr=cutoff_bohr,
        nuclear_cutoff_bohr=nuclear_cutoff_bohr,
        scf_conv_tol=scf_conv_tol,
        scf_max_iter=scf_max_iter,
        scf_options=scf_options,
    )

    ref_lattice = np.asarray(system.lattice, dtype=float)
    atoms = list(system.unit_cell)
    V0 = float(abs(np.linalg.det(ref_lattice)))

    def objective(strain_pct: np.ndarray) -> float:
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
            system.dim,
            new_lattice,
            atoms,
            charge=system.charge,
            multiplicity=system.multiplicity,
        )
        basis = BasisSet(sys.unit_cell_molecule(), basis_name)
        km = _kmesh_for_system(kmesh, system, sys)
        e, _ = _run_scf(
            sys, basis, km, opts, method_upper, functional, **bipole_kwargs
        )
        return e

    def gradient(strain_pct: np.ndarray) -> np.ndarray:
        # Exact dE/dstrain via a central finite difference of the energy
        # actually being minimised -- the production-correct cell gradient.
        #
        # NOTE: the force virial ``compute_stress_tensor`` (used here before
        # 2026-06-05) is WRONG for this objective: it (a) assumes the atoms
        # scale with the strain, but ``objective`` holds the Cartesian atom
        # positions fixed, and (b) misses the explicit lattice / Ewald strain
        # dependence + the Gaussian-basis Pulay stress. On H₂/STO-3G it came
        # out *opposite in sign* to dE/dstrain, so L-BFGS-B walked the cell
        # the wrong way. FD of the objective is consistent by construction.
        return _central_fd_gradient(objective, strain_pct, _CELL_FD_STRAIN_PCT)

    cell_gtol = 1e-6
    res = minimize(
        objective,
        np.zeros(6),
        method="L-BFGS-B",
        jac=gradient,
        options={"maxiter": max_iter, "gtol": cell_gtol},
    )

    strain = res.x * 0.01
    S = np.array(
        [
            [strain[0], strain[5], strain[4]],
            [strain[5], strain[1], strain[3]],
            [strain[4], strain[3], strain[2]],
        ]
    )
    opt_lattice = ref_lattice @ (np.eye(3) + S)
    sys_opt = PeriodicSystem(
        system.dim,
        opt_lattice,
        atoms,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )

    # Gradient-gated convergence (F1): res.success can be set by scipy's
    # ftol criterion at a non-stationary cell, so require the actual
    # max-component strain gradient to meet the gtol tolerance.
    from .molecular_optimize import _gradient_converged

    converged, grad_max = _gradient_converged(bool(res.success), res.jac, cell_gtol)
    _emit_opt_line(
        f"\nCell relaxation (gradient): {res.nit} iters, "
        f"E = {res.fun:.8f} Ha, max|grad| = {grad_max:.4e}, converged={converged}",
    )
    return OptimizeResult(sys_opt, res.fun, None, res.nit, converged)


def relax_full(
    system: PeriodicSystem,
    basis_name: str,
    kmesh: KMeshInput,
    method: str = "RHF",
    *,
    functional: Optional[str] = None,
    force_mode: str = "fd",
    fd_step_bohr: float = 1e-3,
    max_outer: int = 5,
    max_atom_iter: int = 20,
    max_cell_iter: int = 10,
    conv_tol_grad: float = 1e-4,
    scf_conv_tol: Optional[float] = None,
    scf_max_iter: Optional[int] = None,
    scf_options: Optional[Union[PeriodicRHFOptions, PeriodicKSOptions]] = None,
    cutoff_bohr: float = 8.0,
    nuclear_cutoff_bohr: Optional[float] = None,
    progress: Any = None,
    **bipole_kwargs,
) -> OptimizeResult:
    """Reserved coupled atom/cell entry point; currently fails closed.

    Atomic forces default to the exact finite-difference path
    (``force_mode="fd"``); see the module docstring.

    Parameters
    ----------
    max_outer : int
        Maximum number of outer cell+atom cycles.
    max_atom_iter, max_cell_iter
        Max iterations per inner relaxation step.
    scf_conv_tol, scf_max_iter, scf_options
        As in :func:`relax_atoms`.
    """
    _reject_variable_cell_optimization()
    current = system
    current_kmesh = _as_bloch_kmesh(kmesh)
    for outer in range(max_outer):
        # Relax atoms at fixed lattice
        atom_result = relax_atoms(
            current,
            basis_name,
            current_kmesh,
            method,
            functional=functional,
            force_mode=force_mode,
            fd_step_bohr=fd_step_bohr,
            max_iter=max_atom_iter,
            conv_tol_grad=conv_tol_grad,
            scf_conv_tol=scf_conv_tol,
            scf_max_iter=scf_max_iter,
            scf_options=scf_options,
            cutoff_bohr=cutoff_bohr,
            nuclear_cutoff_bohr=nuclear_cutoff_bohr,
            **bipole_kwargs,
        )
        current = atom_result.system

        # Relax lattice at fixed (relaxed) atoms
        before_cell = current
        cell_result = relax_cell(
            current,
            basis_name,
            current_kmesh,
            method,
            functional=functional,
            max_iter=max_cell_iter,
            scf_conv_tol=scf_conv_tol,
            scf_max_iter=scf_max_iter,
            scf_options=scf_options,
            cutoff_bohr=cutoff_bohr,
            nuclear_cutoff_bohr=nuclear_cutoff_bohr,
            **bipole_kwargs,
        )
        current = cell_result.system
        current_kmesh = _kmesh_for_system(current_kmesh, before_cell, current)

        _emit_opt_line(
            f"  Outer cycle {outer + 1}/{max_outer}: E = {cell_result.energy:.8f} Ha",
        )

        # Outer convergence: both inner relaxations of THIS cycle converged
        # (atoms already stationary AND cell already stationary) => the coupled
        # structure is at a stationary point; stop early.
        if atom_result.converged and cell_result.converged:
            return OptimizeResult(
                current, cell_result.energy, None, outer + 1, True)

    # Exhausted max_outer cycles without both inner relaxations converging in
    # the same cycle -- report the achieved (not assumed) convergence state.
    return OptimizeResult(
        current, cell_result.energy, None, max_outer,
        bool(atom_result.converged and cell_result.converged))
