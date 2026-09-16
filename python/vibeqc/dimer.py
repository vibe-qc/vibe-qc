"""Dimer method -- single-ended saddle-point (transition-state) search.

The dimer method (Henkelman & Jónsson, J. Chem. Phys. 111, 7010 (1999))
finds a first-order saddle point from a *single* starting geometry plus
an initial direction -- no second endpoint, unlike NEB. It uses forces
only (no Hessian): a "dimer" of two images a small distance apart is
**rotated** to align its axis with the lowest-curvature mode of the PES,
then **translated** uphill along that mode (and downhill in every
perpendicular direction) until the force vanishes at the saddle.

Public surface
==============
* :func:`run_dimer` -- the driver. It reuses the NEB per-geometry
  energy/force evaluators, so the same method surface is available:
  molecular RHF / UHF / ROHF / RKS / UKS / ROKS, periodic RHF / UHF / RKS /
  UKS (BIPOLE + FD gradient), and the MACE machine-learned potential
  (``method="mace"``). MACE is an
  especially good fit -- analytic forces make the many dimer force
  evaluations cheap.
* :class:`DimerResult` -- converged saddle geometry, energy, curvature
  along the dimer axis (negative at a genuine first-order saddle), and
  the converged reaction-coordinate mode.

When the curvature stays positive the geometry is a minimum, not a
saddle (``is_saddle`` is False) -- try a different ``initial_direction``.

Reference
=========
Henkelman, G.; Jónsson, H. *A dimer method for finding saddle points on
high dimensional potential surfaces using only first derivatives.*
J. Chem. Phys. 111, 7010 (1999). doi:10.1063/1.480097.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ._vibeqc_core import Molecule, PeriodicSystem
from .neb import (
    System,
    _atomic_numbers_of,  # noqa: F401 - re-exported for MACE test adapters
    _evaluate_image,
    _evaluate_image_mace,
    _evaluate_image_periodic,
    _load_mace_model,
    _positions_of,
    _rebuild_with_positions,
)
from .progress import ProgressLogger, resolve_progress

# positions (n_atoms, 3) bohr -> (energy Ha, force Ha/bohr); force = -gradient.
ForceFn = Callable[[np.ndarray], "tuple[float, np.ndarray]"]


@dataclass
class DimerResult:
    """Outcome of a :func:`run_dimer` saddle search.

    Attributes
    ----------
    system
        The converged geometry (the saddle estimate).
    energy
        Energy at the converged geometry (Hartree).
    max_force
        Final max-norm true force (Ha/bohr); compare to ``conv_tol_force``.
    curvature
        PES curvature along the converged dimer axis (Ha/bohr^2).
        **Negative** at a genuine first-order saddle.
    mode
        ``(n_atoms, 3)`` unit vector -- the converged lowest-curvature
        (reaction-coordinate) direction.
    converged
        True iff ``max_force`` fell below ``conv_tol_force``.
    is_saddle
        ``converged and curvature < 0`` -- a first-order saddle (not a
        minimum the search fell into).
    n_iter, n_force_evals
        Outer iterations and total energy/force evaluations.
    """

    system: System
    energy: float
    max_force: float
    curvature: float
    mode: np.ndarray
    converged: bool
    is_saddle: bool
    n_iter: int
    n_force_evals: int
    # Captured for the citation surface (write_qvf), mirroring NEBResult.
    method: Optional[str] = None
    basis: Optional[str] = None
    functional: Optional[str] = None
    is_periodic: bool = False
    mace_model_citation: Optional[str] = None

    def write_citations(self, stem: Any) -> "tuple[Any, Any]":
        """Write ``{stem}.bibtex`` + ``{stem}.references`` for this search.

        Fires the dimer-method paper (Henkelman & Jónsson 1999) via the
        ``routes.drivers.dimer`` route, plus the per-geometry method /
        basis / functional -- or, for MACE, the model papers (and *not*
        the Gaussian-integral library). Returns the two written paths.
        """
        from .output.citations import (
            load_default_database,
            write_bibtex,
            write_references,
        )

        is_mace = (self.method or "").lower() == "mace"
        asm: dict[str, Any] = {
            "method": self.method or "RHF",
            "basis": "mace" if is_mace else (self.basis or "sto-3g"),
            "functional": None if is_mace else self.functional,
            "periodic": self.is_periodic,
            "uses_dimer": True,
        }
        if is_mace:
            asm["uses_integrals"] = False
            asm["uses_scf"] = False
            if self.mace_model_citation:
                asm["extra_entries"] = [self.mace_model_citation]
        refs = load_default_database().assemble(**asm)
        return write_bibtex(stem, refs), write_references(stem, refs)


# ---------------------------------------------------------------------------
# Core algorithm (analytic-rotation dimer) -- force_fn only, no SCF knowledge.
# ---------------------------------------------------------------------------


def _dimer_search(
    x0: np.ndarray,
    force_fn: ForceFn,
    *,
    initial_direction: np.ndarray,
    dimer_separation: float,
    max_iter: int,
    conv_tol_force: float,
    max_step: float,
    initial_step: float,
    n_rotations: int,
    rotation_force_tol: float,
    frozen_mask: Optional[np.ndarray],
    progress: Any,
    project_translation: bool = True,
    lbfgs_acceleration: bool = False,
    lbfgs_memory: int = 10,
) -> "tuple[np.ndarray, np.ndarray, float, float, float, bool, int, int]":
    """Run the dimer search in Cartesian space.

    ``x0`` is the (n_atoms, 3) start (bohr); ``force_fn(x)`` returns
    ``(energy, force)`` with the **force** (= -gradient). Returns
    ``(R, N, curvature, energy, max_force, converged, n_iter, n_evals)``
    where ``R`` is the saddle geometry and ``N`` the unit dimer axis.

    Rotation uses the analytic optimal angle from one trial orientation
    (Henkelman & Jónsson 1999): of the two curvature extrema, the
    minimum-curvature branch is chosen. Translation inverts the
    tangent-parallel force when the curvature is negative (climb toward
    the saddle) or pushes along ``-F∥`` while still convex.

    When *lbfgs_acceleration* is True, the translation step uses an
    L-BFGS two-loop recursion to build a local quadratic model of the
    PES along the translation direction (Kästner & Sherwood, J. Chem.
    Phys. 128, 014106, 2008), converging superlinearly.
    """
    n_atoms = x0.shape[0]
    if hasattr(progress, "write") and not isinstance(progress, ProgressLogger):
        plog = ProgressLogger(stream=progress, verbose=2)
    else:
        plog = resolve_progress(progress, verbose=2)
    R = x0.reshape(-1).astype(float)
    N = initial_direction.reshape(-1).astype(float)
    if frozen_mask is not None:
        # Frozen atoms never move and aren't part of the search direction.
        fm = np.repeat(frozen_mask, 3)
        N = N.copy()
        N[fm] = 0.0
    nrm = float(np.linalg.norm(N))
    if nrm < 1e-12:
        raise ValueError("run_dimer: initial_direction is (near) zero.")
    N /= nrm
    D = float(dimer_separation)

    def _remove_net_translation(v: np.ndarray) -> np.ndarray:
        # Project out the 3 translational zero modes so the search doesn't
        # drift the centre of mass. Valid only when the energy is
        # translation-invariant (an isolated molecule / crystal); skipped
        # when atoms are frozen (they anchor the system) or when the caller
        # disables it (e.g. an external-field analytic potential).
        if frozen_mask is not None or not project_translation:
            return v
        va = v.reshape(n_atoms, 3)
        return (va - va.mean(axis=0)).reshape(-1)

    N = _remove_net_translation(N)
    N /= float(np.linalg.norm(N))

    n_evals = 0

    def _eval(flat: np.ndarray) -> "tuple[float, np.ndarray]":
        nonlocal n_evals
        e, f = force_fn(flat.reshape(n_atoms, 3))
        n_evals += 1
        f = np.asarray(f, dtype=float).reshape(-1)
        if frozen_mask is not None:
            f = f.copy()
            f[np.repeat(frozen_mask, 3)] = 0.0
        return float(e), f

    # L-BFGS state for accelerated translation (Kästner-Sherwood 2008)
    s_list_lbfgs: list[np.ndarray] = []
    y_list_lbfgs: list[np.ndarray] = []
    rho_list_lbfgs: list[float] = []
    f_eff_prev: Optional[np.ndarray] = None
    R_prev_lbfgs: Optional[np.ndarray] = None

    velocity = np.zeros_like(R)
    step = float(initial_step)
    curvature = 0.0
    energy = 0.0
    max_force = float("inf")
    converged = False
    n_iter = 0

    for outer in range(max_iter):
        n_iter = outer + 1
        energy, F0 = _eval(R)

        # --- rotation: align N with the lowest-curvature mode ---
        for _ in range(n_rotations):
            _, F1 = _eval(R + D * N)
            F2 = 2.0 * F0 - F1
            curvature = float(np.dot(F2 - F1, N) / (2.0 * D))
            f_rot = (F1 - F2) - np.dot(F1 - F2, N) * N
            fr = float(np.linalg.norm(f_rot))
            if fr < rotation_force_tol:
                break
            theta = f_rot / fr
            b1 = -fr / (2.0 * D)  # = 0.5 dC/dphi|0 (rotating toward th lowers C)
            phi_t = np.pi / 4.0
            n_trial = N * np.cos(phi_t) + theta * np.sin(phi_t)
            _, F1t = _eval(R + D * n_trial)
            F2t = 2.0 * F0 - F1t
            c_t = float(np.dot(F2t - F1t, n_trial) / (2.0 * D))
            a1 = (c_t - curvature - b1 * np.sin(2.0 * phi_t)) / (
                np.cos(2.0 * phi_t) - 1.0
            )

            def _c_model(p: float) -> float:
                return curvature - a1 + a1 * np.cos(2.0 * p) + b1 * np.sin(2.0 * p)

            phi_a = 0.5 * np.arctan2(b1, a1)
            phi = min((phi_a, phi_a + np.pi / 2.0), key=_c_model)
            N = N * np.cos(phi) + theta * np.sin(phi)
            if frozen_mask is not None:
                N[np.repeat(frozen_mask, 3)] = 0.0
            N = _remove_net_translation(N)
            N /= float(np.linalg.norm(N))

        # --- translation: compute effective force ---
        f_par = np.dot(F0, N) * N
        if curvature > 0.0:
            f_eff = -f_par  # convex: drive toward a negative mode
        else:
            f_eff = F0 - 2.0 * f_par  # concave: climb along N to the saddle
        f_eff = _remove_net_translation(f_eff)

        # --- update L-BFGS history from the previous translation step ---
        if lbfgs_acceleration and R_prev_lbfgs is not None and f_eff_prev is not None:
            s = (R - R_prev_lbfgs).ravel()
            y = (f_eff - f_eff_prev).ravel()
            sy = float(np.dot(s, y))
            if sy > 1e-12:
                s_list_lbfgs.insert(0, s.copy())
                y_list_lbfgs.insert(0, y.copy())
                rho_list_lbfgs.insert(0, 1.0 / sy)
                if len(s_list_lbfgs) > lbfgs_memory:
                    s_list_lbfgs.pop()
                    y_list_lbfgs.pop()
                    rho_list_lbfgs.pop()

        max_force = float(np.max(np.abs(F0)))
        plog.write_raw(
            f"dimer iter {n_iter:3d}: max|F| = {max_force:.4e} Ha/bohr, "
            f"curvature = {curvature:+.4e} Ha/bohr^2, E = {energy:.6f} Ha"
        )
        if max_force < conv_tol_force:
            converged = True
            break

        # --- compute translation direction ---
        if lbfgs_acceleration and s_list_lbfgs:
            # L-BFGS two-loop recursion -- Kästner & Sherwood 2008.
            # Builds a local quadratic model of the translation PES:
            # direction ≈ H * f_eff, where H is the approximate
            # inverse Hessian.
            q = f_eff.copy()
            alpha: list[float] = []
            for s_i, y_i, rho_i in zip(s_list_lbfgs, y_list_lbfgs, rho_list_lbfgs):
                a = rho_i * float(np.dot(s_i, q))
                alpha.append(a)
                q = q - a * y_i
            gamma = float(
                np.dot(s_list_lbfgs[0], y_list_lbfgs[0])
                / np.dot(y_list_lbfgs[0], y_list_lbfgs[0])
            )
            r = gamma * q
            for s_i, y_i, rho_i, a in zip(
                reversed(s_list_lbfgs),
                reversed(y_list_lbfgs),
                reversed(rho_list_lbfgs),
                reversed(alpha),
            ):
                beta = rho_i * float(np.dot(y_i, r))
                r = r + s_i * (a - beta)
            d_trans = r  # H * f_eff -- incorporates curvature info
        else:
            d_trans = f_eff  # standard effective force direction

        # --- translation: step the dimer centre ---
        velocity = velocity + step * d_trans
        if float(np.dot(velocity, d_trans)) < 0.0:
            velocity = np.zeros_like(velocity)
            step = initial_step
        else:
            fn = float(np.linalg.norm(d_trans))
            if fn > 1e-14:
                velocity = (float(np.dot(velocity, d_trans)) / fn) * (d_trans / fn)
            step = min(step * 1.1, 10.0 * initial_step)
        disp = step * velocity
        dmax = float(np.max(np.abs(disp)))
        if dmax > max_step:
            disp = disp * (max_step / dmax)
            velocity = np.zeros_like(velocity)

        # Save state for next L-BFGS update
        if lbfgs_acceleration:
            R_prev_lbfgs = R.copy()
            f_eff_prev = f_eff.copy()

        R = R + disp

    return (
        R.reshape(n_atoms, 3),
        N.reshape(n_atoms, 3),
        curvature,
        energy,
        max_force,
        converged,
        n_iter,
        n_evals,
    )


# ---------------------------------------------------------------------------
# Driver: build a force_fn from the (reused) NEB evaluators
# ---------------------------------------------------------------------------


def _build_force_fn(
    template: System,
    basis: Optional[str],
    method_lower: str,
    *,
    is_periodic: bool,
    is_mace: bool,
    functional: Optional[str],
    rhf_options: Any,
    uhf_options: Any,
    rks_options: Any,
    uks_options: Any,
    rohf_options: Any,
    roks_options: Any,
    gradient_options: Any,
    grid_options: Any,
    grid_level: str = "orca-defgrid3",
    dispersion_params: Any,
    mlip_options: Any,
    kmesh: Any,
    fd_step_bohr: float,
    dft_plus_u: Optional[Sequence[Any]],
) -> "tuple[ForceFn, str]":
    """Return ``(force_fn, mace_citation)``.

    ``force_fn(positions) -> (energy, force)`` reuses the NEB per-geometry
    evaluators (so the SCF / periodic / MACE dispatch is shared with
    :func:`vibeqc.run_neb`). ``force = -gradient``.
    """
    mace_citation = ""
    if is_mace:
        calc, numbers, cell, mace_citation = _load_mace_model(
            template, mlip_options, is_periodic
        )

        def force_fn(positions: np.ndarray) -> "tuple[float, np.ndarray]":
            e, grad, _ = _evaluate_image_mace(
                positions, calc=calc, numbers=numbers, cell=cell
            )
            return e, -np.asarray(grad, dtype=float)

    elif is_periodic:

        def force_fn(positions: np.ndarray) -> "tuple[float, np.ndarray]":
            e, grad, _ = _evaluate_image_periodic(
                positions,
                template,
                basis,
                method_lower,
                kmesh=kmesh,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                fd_step_bohr=fd_step_bohr,
                dft_plus_u=dft_plus_u,
            )
            return e, -np.asarray(grad, dtype=float)

    else:
        # #643: the ECP centres on the SCF options are absolute coordinates
        # built for ``template``; refuse a set that sits on no template atom
        # before the first SCF, exactly as compute_hessian_fd does. The image
        # evaluator then moves the centres onto every visited geometry.
        from ._vibeqc_core import RHFOptions, RKSOptions, UHFOptions, UKSOptions
        from .ecp_metadata import (
            attach_inline_ecp_options_from_basis_sidecar,
            basis_sidecar_has_ecp_operator,
            ecp_centre_atom_indices,
            molecular_options_request_ecp_operator,
            options_carry_ecp,
        )

        if method_lower in ("rohf", "roks") and (
            basis_sidecar_has_ecp_operator(template, basis)
            or molecular_options_request_ecp_operator(
                rohf_options if method_lower == "rohf" else roks_options
            )
        ):
            raise NotImplementedError(
                f"{method_lower.upper()} dimer/IRC gradients do not support "
                "molecular ECP operators. Use an all-electron basis or an "
                "RHF/UHF/RKS/UKS ECP gradient route."
            )

        # Keep one concrete options object in the closure. This lets ordinary
        # basis-sidecar ECP use work without requiring callers to pre-populate
        # options, and gives the image evaluator exact centers to move. Match
        # its established defgrid3 default when materialising KS options.
        if method_lower == "rhf" and rhf_options is None:
            rhf_options = RHFOptions()
        elif method_lower == "uhf" and uhf_options is None:
            uhf_options = UHFOptions()
        elif method_lower == "rks" and rks_options is None:
            from .runner import _apply_grid_level

            rks_options = RKSOptions()
            _apply_grid_level(rks_options.grid, grid_level)
        elif method_lower == "uks" and uks_options is None:
            from .runner import _apply_grid_level

            uks_options = UKSOptions()
            _apply_grid_level(uks_options.grid, grid_level)

        _mf_opts = {
            "rhf": rhf_options,
            "uhf": uhf_options,
            "rks": rks_options,
            "uks": uks_options,
        }.get(method_lower)
        if _mf_opts is not None:
            from ._vibeqc_core import BasisSet

            attach_inline_ecp_options_from_basis_sidecar(
                _mf_opts, template, BasisSet(template, basis)
            )
        if options_carry_ecp(_mf_opts):
            ecp_centre_atom_indices(_mf_opts, template)

        def force_fn(positions: np.ndarray) -> "tuple[float, np.ndarray]":
            e, grad, _ = _evaluate_image(
                positions,
                template,
                basis,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                rohf_options=rohf_options,
                roks_options=roks_options,
                gradient_options=gradient_options,
                grid_options=grid_options,
                dispersion_params=dispersion_params,
                fd_step_bohr=fd_step_bohr,
                grid_level=grid_level,
            )
            return e, -np.asarray(grad, dtype=float)

    return force_fn, mace_citation


def run_dimer(
    initial: System,
    basis: Optional[str] = None,
    *,
    method: str = "RKS",
    functional: Optional[str] = "pbe",
    initial_direction: Optional[np.ndarray] = None,
    dimer_separation: float = 0.01,
    max_iter: int = 200,
    conv_tol_force: float = 1e-3,
    max_step: float = 0.1,
    initial_step: float = 0.05,
    n_rotations: int = 4,
    rotation_force_tol: float = 1e-3,
    freeze_indices: Optional[Sequence[int]] = None,
    rhf_options: Any = None,
    uhf_options: Any = None,
    rks_options: Any = None,
    uks_options: Any = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    gradient_options: Any = None,
    grid_options: Any = None,
    grid_level: str = "orca-defgrid3",
    dispersion_params: Any = None,
    mlip_options: Any = None,
    kpoints: Optional[Any] = None,
    fd_step_bohr: float = 1e-3,
    dft_plus_u: Optional[Sequence[Any]] = None,
    progress: Any = False,
) -> DimerResult:
    """Find a first-order saddle point with the dimer method.

    Parameters
    ----------
    initial
        :class:`Molecule` or :class:`PeriodicSystem` start geometry.
    basis
        Basis-set name. Required for the SCF methods; ignored for
        ``method="mace"``.
    method
        ``"RHF"`` / ``"UHF"`` / ``"ROHF"`` / ``"RKS"`` / ``"UKS"`` /
        ``"ROKS"`` or ``"MACE"`` (case-insensitive) -- the same per-geometry
        evaluator dispatch as :func:`vibeqc.run_neb`. ROHF uses its analytic
        gradient; ROKS uses central energy differences. MACE's analytic
        forces make the dimer's many force evaluations cheap.
    initial_direction
        ``(n_atoms, 3)`` initial dimer axis -- the guessed reaction
        direction. The dimer rotates it to the nearest lowest-curvature
        mode, so an approximate guess is fine. ``None`` => a reproducible
        deterministic seed (fixed RNG); pass a guess to target a specific
        saddle.
    dimer_separation
        Half-distance between the two dimer images (bohr); 0.01 is a good
        default (small enough to probe the local curvature).
    max_iter, conv_tol_force, max_step, initial_step
        Outer-loop cap, convergence threshold on the max-norm true force
        (Ha/bohr), and quick-min step controls.
    n_rotations, rotation_force_tol
        Max dimer-rotation steps per translation, and the rotational-force
        tolerance at which rotation stops early.
    freeze_indices
        Atom indices to hold fixed (e.g. slab-substrate atoms) -- their
        force components and the search direction are zeroed.
    mlip_options, kpoints, fd_step_bohr, dft_plus_u, *_options
        Forwarded to the shared evaluators exactly as in
        :func:`vibeqc.run_neb`.
    progress
        If True, emit per-iteration live progress to stdout. A
        :class:`~vibeqc.progress.ProgressLogger` or writable text stream may
        be supplied to select another live sink.

    Returns
    -------
    :class:`DimerResult`
    """
    is_periodic = isinstance(initial, PeriodicSystem)
    if not is_periodic and not isinstance(initial, Molecule):
        raise NotImplementedError(
            "run_dimer: only Molecule and PeriodicSystem are supported."
        )
    method_lower = method.lower()
    if method_lower not in ("rhf", "uhf", "rohf", "rks", "uks", "roks", "mace"):
        raise ValueError(
            f"run_dimer: unsupported method {method!r}. "
            "Use one of 'RHF', 'UHF', 'ROHF', 'RKS', 'UKS', 'ROKS', 'MACE'."
        )
    if is_periodic and method_lower in ("rohf", "roks"):
        raise NotImplementedError(
            "run_dimer: ROHF/ROKS saddle searches are currently "
            "molecular-only; periodic restricted-open-shell dimer remains gated."
        )
    is_mace = method_lower == "mace"
    if not is_mace and basis is None:
        raise ValueError(
            f"run_dimer: a basis set is required for method={method!r} "
            "(basis is only optional for method='mace')."
        )
    if is_mace and dft_plus_u:
        raise ValueError("run_dimer: dft_plus_u is not supported with method='mace'.")
    if method_lower in ("rohf", "roks") and dft_plus_u:
        raise NotImplementedError(
            "run_dimer: DFT+U is not implemented for ROHF/ROKS saddle searches."
        )

    x0 = _positions_of(initial)
    n_atoms = x0.shape[0]

    frozen_mask: Optional[np.ndarray] = None
    if freeze_indices is not None:
        fi = {int(i) for i in freeze_indices}
        bad = sorted(i for i in fi if i < 0 or i >= n_atoms)
        if bad:
            raise ValueError(
                f"run_dimer: freeze_indices {bad} out of range [0, {n_atoms})"
            )
        frozen_mask = np.zeros(n_atoms, dtype=bool)
        for i in fi:
            frozen_mask[i] = True

    if initial_direction is None:
        # Reproducible deterministic seed (so a given run is repeatable);
        # the rotation finds the nearest lowest-curvature mode from it.
        direction = np.random.default_rng(0).standard_normal((n_atoms, 3))
    else:
        direction = np.asarray(initial_direction, dtype=float).reshape(n_atoms, 3)

    # Periodic k-mesh (MACE ignores it).
    kmesh = None
    if is_periodic and not is_mace:
        from ._vibeqc_core import monkhorst_pack as _mp

        if kpoints is None:
            kmesh = _mp(initial, (1, 1, 1))
        elif hasattr(kpoints, "to_bloch_kmesh") or (
            hasattr(kpoints, "kpoints") and hasattr(kpoints, "weights")
        ):
            from .kpoints import as_bloch_kmesh

            kmesh = as_bloch_kmesh(kpoints)
        else:
            from .kpoints import _integer_counts

            kmesh = _mp(initial, _integer_counts(kpoints, name="dimer kpoints mesh"))

    # DFT+U on the molecular SCF path is applied Options-side (as in
    # run_neb); periodic forwards the list per call.
    if dft_plus_u and not is_periodic and not is_mace:
        from ._vibeqc_core import (
            BasisSet,
            RHFOptions,
            RKSOptions,
            UHFOptions,
            UKSOptions,
        )
        from .dft_plus_u import _apply_dft_plus_u_to_options

        _basis = BasisSet(initial, basis)
        _opt_map = {
            "rhf": (rhf_options, RHFOptions),
            "uhf": (uhf_options, UHFOptions),
            "rks": (rks_options, RKSOptions),
            "uks": (uks_options, UKSOptions),
        }
        _cur, _cls = _opt_map[method_lower]
        if _cur is None:
            _cur = _cls()
        _apply_dft_plus_u_to_options(_cur, _basis, dft_plus_u)
        if method_lower == "rhf":
            rhf_options = _cur
        elif method_lower == "uhf":
            uhf_options = _cur
        elif method_lower == "rks":
            rks_options = _cur
        else:
            uks_options = _cur

    force_fn, mace_citation = _build_force_fn(
        initial,
        basis,
        method_lower,
        is_periodic=is_periodic,
        is_mace=is_mace,
        functional=functional,
        rhf_options=rhf_options,
        uhf_options=uhf_options,
        rks_options=rks_options,
        uks_options=uks_options,
        rohf_options=rohf_options,
        roks_options=roks_options,
        gradient_options=gradient_options,
        grid_options=grid_options,
        dispersion_params=dispersion_params,
        mlip_options=mlip_options,
        kmesh=kmesh,
        fd_step_bohr=fd_step_bohr,
        dft_plus_u=dft_plus_u,
        grid_level=grid_level,
    )

    (R, N, curvature, energy, max_force, converged, n_iter, n_evals) = _dimer_search(
        x0,
        force_fn,
        initial_direction=direction,
        dimer_separation=dimer_separation,
        max_iter=max_iter,
        conv_tol_force=conv_tol_force,
        max_step=max_step,
        initial_step=initial_step,
        n_rotations=n_rotations,
        rotation_force_tol=rotation_force_tol,
        frozen_mask=frozen_mask,
        progress=progress,
    )

    saddle = _rebuild_with_positions(initial, R)
    return DimerResult(
        system=saddle,
        energy=energy,
        max_force=max_force,
        curvature=curvature,
        mode=N,
        converged=converged,
        is_saddle=bool(converged and curvature < 0.0),
        n_iter=n_iter,
        n_force_evals=n_evals,
        method=method,
        basis=basis,
        functional=(
            functional if method_lower in ("rks", "uks", "roks") else None
        ),
        is_periodic=is_periodic,
        mace_model_citation=(mace_citation or None),
    )


__all__ = ["DimerResult", "run_dimer"]
