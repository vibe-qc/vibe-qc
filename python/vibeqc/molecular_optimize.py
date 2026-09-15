"""Native molecular geometry optimization -- no ASE required.

Atomic position relaxation using analytic SCF gradients wrapped in
scipy L-BFGS-B. Supports RHF, UHF, RKS, UKS, and wavefunction methods
(selected_ci, dmrg, v2rdm, transcorrelated_ci, casci, casscf -- these
fall back to central finite differences on the energy). The
wavefunction solver options (``casscf_options``, ``active_space``,
``cas_reference``, ...) are threaded through every per-step energy
evaluation, so the optimizer walks the same surface the final single
point reports -- mirroring the ASE backend's calculator.

Dispersion corrections (D3-BJ) and implicit solvation (CPCM/COSMO)
are passed through transparently so the optimizer sees the total
energy + gradient.

Usage::

    from vibeqc.molecular_optimize import optimize_molecule

    result = optimize_molecule(
        mol, basis_name="def2-svp", method="rks", functional="PBE",
    )
    # result.system        -- optimized Molecule (bohr)
    # result.energy        -- final total energy (Ha)
    # result.trajectory_frames   -- per-step geometries
    # result.trajectory_energies -- per-step energies

Integration with ``run_job`` / QVF writing is automatic: when
``optimize=True`` the trajectory data collected here is passed
through to ``write_qvf`` for vibe-view's animation player.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    GradientOptions,
    GridOptions,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)
from .ecp_metadata import (
    ecp_centre_atom_indices,
    ecp_centres_follow_atoms,
    options_carry_ecp,
)
from .gradient_options import copy_ecp_fields, gradient_options_from_scf
from .progress import ProgressLogger, resolve_progress

__all__ = [
    "MolecularOptimizeResult",
    "brent_minimize_1d",
    "optimize_molecule",
    "optimize_molecule_brent",
]


_ECP_GRADIENT_METHODS = frozenset({"rhf", "uhf", "rks", "uks"})


def _refuse_unsupported_ecp_optimization(
    molecule: Molecule,
    basis_name: str,
    method: str,
    *,
    solvent: Any,
    options: Sequence[object],
    route: str,
) -> None:
    """Fail before work unless this is a supported gas-phase ECP gradient."""

    from .ecp_metadata import (
        basis_sidecar_has_ecp_operator,
        molecular_options_request_ecp_operator,
    )

    has_ecp = basis_sidecar_has_ecp_operator(molecule, basis_name) or any(
        molecular_options_request_ecp_operator(option)
        for option in options
        if option is not None
    )
    if not has_ecp:
        return

    method_lower = method.lower()
    if method_lower not in _ECP_GRADIENT_METHODS:
        raise NotImplementedError(
            f"{route}: molecular ECP derivatives/optimization are not "
            f"implemented for method={method_lower!r}; only RHF/UHF/RKS/UKS "
            "mean-field ECP gradients are supported. Refusing before "
            "SCF/trajectory work rather than differentiating an unverified "
            "or bare-Z post-HF Hamiltonian."
        )

    if solvent is not None:
        from .solvation import resolve_solvent

        solvent_model = resolve_solvent(solvent)
    else:
        solvent_model = None
    if solvent_model is not None and not solvent_model.is_gas_phase:
        raise NotImplementedError(
            f"{route}: ECP+CPCM molecular gradients are not implemented, so "
            "the optimizer cannot follow the reported solvated energy. Run "
            "the ECP optimization in gas phase or use an all-electron basis."
        )


def _gradient_converged(
    success: bool,
    grad: Optional[np.ndarray],
    conv_tol_grad: float,
) -> tuple[bool, float]:
    """Independent max-component (inf-norm) gradient convergence gate.

    scipy's L-BFGS-B reports ``res.success`` when EITHER its projected-
    gradient criterion (``gtol``) OR its relative energy-reduction
    criterion (``ftol``) is met. ``ftol`` can trip at a non-stationary
    geometry, so ``res.success`` alone over-reports convergence for a
    geometry optimization (2026-05-31 audit, F1). A geometry has
    converged only when the largest force component actually meets
    ``conv_tol_grad`` -- the same inf-norm metric scipy's ``gtol`` uses.

    Shared by :func:`optimize_molecule` and the BIPOLE relaxers
    (:func:`vibeqc.bipole_optimize.relax_atoms`,
    :func:`vibeqc.bipole_optimize.relax_cell_gradient`) so the three
    drivers cannot drift in what "converged" means.

    Returns ``(converged, grad_max)``; ``grad_max`` is ``inf`` when no
    gradient is available.
    """
    if grad is None:
        return False, float("inf")
    arr = np.abs(np.asarray(grad, dtype=float))
    if arr.size == 0:
        return False, float("inf")
    grad_max = float(np.max(arr))
    return (bool(success) and grad_max <= conv_tol_grad), grad_max


class MolecularOptimizeResult:
    """Container for molecular geometry optimization results."""

    def __init__(
        self,
        system: Molecule,
        energy: float,
        gradient: np.ndarray,
        n_iter: int,
        converged: bool,
        trajectory_frames: Optional[list[Molecule]] = None,
        trajectory_energies: Optional[list[float]] = None,
    ):
        self.system = system
        self.energy = energy
        self.gradient = gradient
        self.n_iter = n_iter
        self.converged = converged
        self.trajectory_frames = trajectory_frames or []
        self.trajectory_energies = trajectory_energies or []

    def __repr__(self) -> str:
        g = np.abs(np.asarray(self.gradient, dtype=float))
        grad_max = float(np.max(g)) if g.size else 0.0
        return (
            f"MolecularOptimizeResult("
            f"energy={self.energy:.8f}, "
            f"max|grad|={grad_max:.4e}, "
            f"n_iter={self.n_iter}, "
            f"converged={self.converged})"
        )


# ---- SCF dispatch ---------------------------------------------------------


def _run_molecular_scf(
    molecule: Molecule,
    basis: BasisSet,
    method: str,
    *,
    functional: Optional[str] = None,
    rhf_options: Optional[RHFOptions] = None,
    uhf_options: Optional[UHFOptions] = None,
    rks_options: Optional[RKSOptions] = None,
    uks_options: Optional[UKSOptions] = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    casscf_options: Any = None,
    active_space: Optional[tuple[int, int]] = None,
    casci_options: Any = None,
    caspt2_options: Any = None,
    nevpt2_options: Any = None,
    cas_reference: Optional[str] = None,
    solvent: Any = None,
    progress: bool = False,
    read_from: Any = None,
) -> tuple[float, Any]:
    """Run a single SCF/wavefunction calculation and return (energy, result).

    ``caspt2_options`` / ``nevpt2_options`` are accepted so the
    ``caspt2`` / ``nevpt2`` dispatch branch can forward them; without
    these the branch referenced names that were never parameters
    (a latent ``NameError`` if that branch was ever reached).

    ``read_from`` forwards to the mean-field wrappers (rhf/uhf/rks/uks
    only) for ``initial_guess=READ`` restarts -- the geomopt warm-start
    path hands the previous step's converged result here so the SCF
    starts from the projected prior density instead of a cold guess.
    The caller is responsible for setting ``initial_guess=READ`` on the
    options struct (the wrappers reject ``read_from`` otherwise).
    """
    # The module-level run_* names are the raw C++ bindings (no read_from
    # parameter); the READ-capable wrappers live in the package __init__.
    # Imported lazily to avoid an import cycle with the package root.
    from vibeqc import run_rhf, run_rks, run_uhf, run_uks

    method_lower = method.lower()
    # CPCM (run_cpcm_scf) composes with the mean-field SCFs only; for the
    # other dispatch branches ``opts`` is never assigned, so reaching the
    # solvent block below with e.g. method="casscf" used to die with a
    # NameError. Refuse up front -- BEFORE the expensive gas-phase solve --
    # with an error that names the actual limitation. Deliberately an error
    # rather than an FD reroute: _run_single_point silently ignores
    # ``solvent`` for the CAS family, so "falling back" would optimize the
    # gas-phase surface while claiming solvation.
    if solvent is not None and method_lower in ("rohf", "casscf", "caspt2", "nevpt2"):
        raise ValueError(
            f"Implicit solvation (CPCM) is not supported for "
            f"method={method!r}: run_cpcm_scf composes with rhf, uhf, rks, "
            f"and uks only. Run the optimization in gas phase, or use a "
            f"mean-field method with solvent."
        )
    if method_lower == "rhf":
        opts = rhf_options or RHFOptions()
        r = run_rhf(molecule, basis, opts, read_from=read_from)
    elif method_lower == "uhf":
        opts = uhf_options or UHFOptions()
        r = run_uhf(molecule, basis, opts, read_from=read_from)
    elif method_lower == "rks":
        opts = rks_options or RKSOptions()
        # Only override the options' functional when the caller passed a
        # ``functional`` AND the options still hold the default/empty XC
        # ("LDA"). The parenthesisation matters: without it, operator
        # precedence makes the guard true whenever opts.functional=="LDA"
        # even for functional=None, and ``opts.functional = None`` then
        # trips the pybind str setter (2026-05-31 audit, F4).
        if functional and (not opts.functional or opts.functional.lower() == "lda"):
            opts.functional = functional
        r = run_rks(molecule, basis, opts, read_from=read_from)
    elif method_lower == "uks":
        opts = uks_options or UKSOptions()
        # See the RKS branch above (F4): parenthesise the guard so a
        # None functional never reaches the pybind str setter.
        if functional and (not opts.functional or opts.functional.lower() == "lda"):
            opts.functional = functional
        r = run_uks(molecule, basis, opts, read_from=read_from)
    elif method_lower == "rohf":
        from .rohf import ROHFOptions, run_rohf

        opts = rohf_options or ROHFOptions()
        r = run_rohf(molecule, basis, opts, read_from=read_from)
    elif method_lower == "roks":
        from .roks import ROKSOptions, run_roks
        opts = roks_options or ROKSOptions()
        if functional:
            opts.functional = functional
        r = run_roks(molecule, basis, opts, read_from=read_from)
    elif method_lower == "casscf":
        from .runner import _run_single_point

        r = _run_single_point(
            "casscf",
            molecule,
            basis,
            functional=None,
            casscf_options=casscf_options,
            active_space=active_space,
            casci_options=casci_options,
            cas_reference=cas_reference,
        )
    elif method_lower in ("caspt2", "nevpt2"):
        from .runner import _run_single_point

        r = _run_single_point(
            method_lower,
            molecule,
            basis,
            functional=None,
            casscf_options=casscf_options,
            caspt2_options=caspt2_options if method_lower == "caspt2" else None,
            nevpt2_options=nevpt2_options if method_lower == "nevpt2" else None,
            active_space=active_space,
            casci_options=casci_options,
            cas_reference=cas_reference,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r} for molecular optimization. "
            f"Use 'rhf', 'uhf', 'rohf', 'rks', 'uks', 'casscf', 'caspt2', or 'nevpt2'."
        )

    if solvent is not None:
        from .solvation import run_cpcm_scf
        from .solvation.driver import _solvent_aware_scf_result

        sol = run_cpcm_scf(
            molecule,
            basis,
            method=method_lower,
            solvent=solvent,
            options=opts,
        )
        return sol.energy, _solvent_aware_scf_result(sol)

    return r.energy, r


def _effective_mean_field_functional(
    method: str, functional: Optional[str], rks_options: Any, uks_options: Any
) -> Optional[str]:
    """The functional name the RKS / UKS SCF will actually run.

    Mirrors :func:`_run_molecular_scf`: an explicit ``functional`` only
    overrides the options struct while that still holds the default LDA.
    Returns ``None`` for non-DFT methods.
    """
    method_lower = method.lower()
    if method_lower == "rks":
        opts = rks_options
    elif method_lower == "uks":
        opts = uks_options
    else:
        return None
    opts_name = str(getattr(opts, "functional", "") or "") if opts is not None else ""
    if functional and (not opts_name or opts_name.lower() == "lda"):
        return str(functional)
    return opts_name or "lda"


def _mean_field_gradient_terms_missing(
    method: str, functional: Optional[str], rks_options: Any, uks_options: Any
) -> list[str]:
    """Analytic-gradient terms the RKS / UKS kernel omits for this job.

    GitLab #571: empty for RHF / UHF / ROHF and for every LDA, GGA,
    meta-GGA and global hybrid; non-empty for range-separated hybrids
    (hse06, wb97x, wb97x-v, cam-b3lyp, ...) and VV10-paired functionals
    (vv10, wb97x-v, wb97m-v).  All three molecular optimizers and the
    geomopt provider consult this so they walk the same surface.
    """
    from .gradient_terms import functional_gradient_terms_missing

    name = _effective_mean_field_functional(
        method, functional, rks_options, uks_options
    )
    if name is None:
        return []
    return functional_gradient_terms_missing(
        name, spin=2 if method.lower() == "uks" else 1
    )


def _announce_fd_fallback(
    method: str,
    functional: Optional[str],
    rks_options: Any,
    uks_options: Any,
    missing: list[str],
    step_bohr: float,
) -> None:
    """One ``.out`` line saying why the optimizer differentiates numerically."""
    from .output import write

    name = _effective_mean_field_functional(
        method, functional, rks_options, uks_options
    )
    terms = "; ".join(missing)
    write(
        f"  Note: the analytic {method.upper()} gradient for functional "
        f"{name!r} omits {terms}. The optimizer uses full-energy central "
        f"finite differences (step {step_bohr:g} bohr) instead (GitLab #571).\n"
    )


def _compute_molecular_gradient(
    molecule: Molecule,
    basis: BasisSet,
    scf_result: Any,
    method: str,
    *,
    gradient_options: Optional[GradientOptions] = None,
    grid_options: Optional[GridOptions] = None,
    dispersion_params: Any = None,
    dft_plus_u: Any = None,
) -> np.ndarray:
    """Compute the analytic nuclear gradient (Ha/bohr, n_atoms x 3).

    When ``dispersion_params`` is provided, the D3-BJ gradient is
    folded in. Returns the energy gradient gradE (not the force).
    """
    from vibeqc import (
        compute_gradient as _grad_rhf,
        compute_gradient_rks as _grad_rks,
        compute_gradient_uhf as _grad_uhf,
        compute_gradient_uks as _grad_uks,
    )
    from .gradient_terms import require_complete_analytic_gradient

    gopt = gradient_options or GradientOptions()
    method_lower = method.lower()

    if method_lower == "rhf":
        grad = _grad_rhf(molecule, basis, scf_result, gopt)
    elif method_lower == "uhf":
        grad = _grad_uhf(molecule, basis, scf_result, gopt)
    elif method_lower == "rks":
        # GitLab #571: fail closed for a functional whose analytic gradient
        # omits terms (the optimizers route such jobs to FD before reaching
        # here; NEB and direct callers get the refusal).
        require_complete_analytic_gradient(
            scf_result, route="_compute_molecular_gradient(rks)", spin=1
        )
        ggrid = grid_options or GridOptions()
        grad = _grad_rks(molecule, basis, scf_result, ggrid, gopt)
    elif method_lower == "uks":
        require_complete_analytic_gradient(
            scf_result, route="_compute_molecular_gradient(uks)", spin=2
        )
        ggrid = grid_options or GridOptions()
        grad = _grad_uks(molecule, basis, scf_result, ggrid, gopt)
    elif method_lower == "rohf":
        from .rohf import compute_rohf_gradient

        grad = compute_rohf_gradient(molecule, basis, scf_result, gradient_options=gopt)
    elif method_lower in ("casscf", "caspt2", "nevpt2"):
        # Check BEFORE np.asarray: asarray(None) yields a 0-d object array
        # (or raises an opaque TypeError on dtype=float), never None, so a
        # post-conversion check can never fire.
        if scf_result.gradient is None:
            raise ValueError(
                "SolverResult.gradient is None — the calculation "
                "may not have converged or gradients were not computed."
            )
        grad = np.asarray(scf_result.gradient, dtype=float)
    else:
        raise ValueError(f"No analytic gradient for method {method!r}.")

    grad = np.asarray(grad, dtype=float)

    if dft_plus_u:
        from ._vibeqc_core import compute_overlap
        from .dft_plus_u import _compute_dft_plus_u_gradient

        overlap = np.asarray(compute_overlap(basis))
        if method_lower in ("rhf", "rks"):
            grad = grad + _compute_dft_plus_u_gradient(
                basis,
                molecule,
                overlap,
                dft_plus_u,
                P_total=np.asarray(scf_result.density),
            )
        elif method_lower in ("uhf", "uks"):
            grad = grad + _compute_dft_plus_u_gradient(
                basis,
                molecule,
                overlap,
                dft_plus_u,
                P_alpha=np.asarray(scf_result.density_alpha),
                P_beta=np.asarray(scf_result.density_beta),
            )
        else:
            raise NotImplementedError(
                f"DFT+U analytic gradients are not implemented for "
                f"method={method!r}."
            )

    # Fold in dispersion gradient if requested.
    if dispersion_params is not None:
        from .dispersion import compute_d3bj

        disp = compute_d3bj(molecule, dispersion_params, with_gradient=True)
        grad = grad + np.asarray(disp.gradient, dtype=float)

    return grad


def _resolve_molecular_ks_gradient_grid(
    method: str,
    *,
    rks_options: Optional[RKSOptions],
    uks_options: Optional[UKSOptions],
    grid_options: Optional[GridOptions],
) -> Optional[GridOptions]:
    """Use the SCF grid for a KS optimizer unless explicitly overridden."""
    if grid_options is not None:
        return grid_options
    method_lower = method.lower()
    if method_lower == "rks" and rks_options is not None:
        return rks_options.grid
    if method_lower == "uks" and uks_options is not None:
        return uks_options.grid
    return None


def _casscf_analytic_gradient_ok(
    molecule: Molecule,
    casscf_options: Any,
) -> bool:
    """Whether a CASSCF optimization may use its analytic nuclear gradient.

    The analytic CASSCF gradient (:mod:`vibeqc.gradient._casscf`) is a
    validated full-energy derivative ONLY inside the state-specific,
    closed-shell envelope -- the envelope the
    ``examples/regression/casscf_gradient_fd_reproducer.py`` adjudicator and
    ``tests/test_casscf_gradient.py::test_full_gradient_vs_target`` pin to
    ~1e-7 Ha/bohr (v0.15.0 P0 fix). Outside it the gradient is NOT
    FD-validated and the optimizers fall back to full-energy central FD:

    * ``nroots > 1`` (state-averaged CASSCF) -- the SA gradient is only
      checked for finiteness + translational invariance, never against FD.
    * ``compute_wz="numerical"`` -- the caller asked for the finite-
      difference oracle, so the optimizer owns the FD.
    * open-shell (``multiplicity > 1``) -- the kernel is the RHF
      (closed-shell) formalism; the open-shell case runs but is unvalidated.

    ``compute_wz=True`` used to be excluded as an experimental CP-MCSCF
    correction; it is now a no-op alias of the analytic path (GitLab #516)
    and stays inside the envelope.

    Returns ``True`` only inside the validated envelope. CASPT2/NEVPT2
    never reach here; their optimization routing is decided separately
    because their production correlation-gradient path is relaxed
    full-energy FD.
    """
    if casscf_options is None:
        nroots, compute_wz = 1, False
    else:
        nroots = getattr(casscf_options, "nroots", 1) or 1
        compute_wz = getattr(casscf_options, "compute_wz", False)
    multiplicity = int(getattr(molecule, "multiplicity", 1) or 1)
    return (
        int(nroots) <= 1
        and multiplicity == 1
        and compute_wz != "numerical"
    )


def _mrpt_analytic_gradient_ok(
    method: str,
    casscf_options: Any,
    caspt2_options: Any,
    nevpt2_options: Any,
    solvent: Any = None,
    *,
    molecule: Molecule | None = None,
    basis: BasisSet | None = None,
    active_space: tuple[int, int] | None = None,
) -> bool:
    """Whether a CASPT2/NEVPT2 optimization may use SolverResult.gradient.

    The runner supplies an analytic Lagrangian gradient for the exact
    single-state IC-CASPT2 envelope and a relaxed full-energy central finite
    difference for NEVPT2.  In both cases it puts the result in
    ``SolverResult.gradient`` only when

    * the PT2 runs on a CASSCF reference (``casscf_options`` given) -- the
      historical CASCI-on-HF reference computes no reference gradient, so
      ``SolverResult.gradient`` is ``None``;
    * ``compute_corr_grad=True`` on the method's options -- otherwise the
      returned gradient is the bare CASSCF gradient, and walking it while
      reporting PT2 energies would optimize a surface inconsistent with
      the reported energy (CLAUDE.md §7 discipline); and
    * gas phase -- the CPCM composition has no analytic PT2 gradient.

    Unsupported IC-CASPT2 references, shifts, and engines stay on the
    optimizer's outer full-energy central FD, which differentiates exactly the
    energy surface it reports.
    """
    if casscf_options is None or solvent is not None:
        return False
    opts = caspt2_options if method == "caspt2" else nevpt2_options
    if not bool(getattr(opts, "compute_corr_grad", False)):
        return False
    if method != "caspt2":
        return True

    # MS/XMS-CASPT2 owns a separate SA-CASSCF response implementation.
    if getattr(opts, "multistate", None) is not None:
        return getattr(casscf_options, "ci_solver", "casci") == "casci"

    if getattr(casscf_options, "ci_solver", "casci") != "casci":
        return False
    if int(getattr(casscf_options, "nroots", 1) or 1) != 1:
        return False
    if getattr(opts, "variant", "ic") != "ic":
        return False
    if float(getattr(opts, "ipea", 0.0)) != 0.0:
        return False
    if float(getattr(opts, "imaginary", 0.0)) != 0.0:
        return False
    if getattr(opts, "engine", "auto") != "auto":
        return False
    if molecule is not None and int(getattr(molecule, "multiplicity", 1)) != 1:
        return False

    # When all partition information is available, keep direct-large jobs on
    # the outer FD route instead of entering the explicit IC response solver.
    if molecule is not None and basis is not None and active_space is not None:
        from .solvers._mrpt import _ic_caspt2_uses_direct_active_ci

        n_act, n_act_elec = active_space
        n_core = (molecule.n_electrons() - n_act_elec) // 2
        if _ic_caspt2_uses_direct_active_ci(
            int(basis.nbasis),
            n_core,
            n_act,
            n_frozen=int(getattr(opts, "n_frozen", 0)),
        ):
            return False
    return True


# ---- Cartesian <-> flat encoding -------------------------------------------


def _positions_to_flat(molecule: Molecule) -> np.ndarray:
    """Flatten Cartesian atom positions to a 1D array (bohr)."""
    flat: list[float] = []
    for atom in molecule.atoms:
        flat.extend(atom.xyz)
    return np.array(flat, dtype=float)


def _flat_to_molecule(
    template: Molecule,
    x: np.ndarray,
) -> Molecule:
    """Rebuild a Molecule from flat Cartesian coordinates (bohr)."""
    n_atoms = len(list(template.atoms))
    new_atoms: list[Atom] = []
    for i in range(n_atoms):
        xyz = [float(x[3 * i + c]) for c in range(3)]
        new_atoms.append(Atom(int(template.atoms[i].Z), xyz))
    return Molecule(new_atoms, template.charge, template.multiplicity)


# ---- FD fallback for methods without analytic gradients ------------------


def _mean_field_options_for(method, rhf_options, uhf_options, rks_options,
                            uks_options):
    """The RHF/UHF/RKS/UKS options struct ``method`` runs on, or ``None``."""
    return {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
    }.get(str(method).lower())


def _ecp_gradient_options(gradient_options, mf_options):
    """Gradient options for one geometry on an ECP system (#643).

    A caller-supplied object is used as given (its ECP fields are kept in
    step with the SCF options by :func:`ecp_centres_follow_atoms`); ``None``
    becomes the mirror of the SCF options -- JK backend and ECP fields --
    because the default-constructed ``GradientOptions`` is the bare-Z
    all-electron gradient and cannot describe an ECP Hamiltonian. Returns
    ``gradient_options`` unchanged when the options carry no ECP, so
    all-electron optimisations keep their current gradient defaults.
    """
    if options_carry_ecp(mf_options):
        if gradient_options is None:
            return gradient_options_from_scf(mf_options)
        # The high-level SCF wrapper may auto-attach a sidecar on its first
        # call, after an outer centre-following context was entered. Mirror
        # the operative route here as well so even that first gradient has
        # exact provenance; retain the caller's independent JK/grid choices.
        copy_ecp_fields(gradient_options, mf_options)
    return gradient_options


def _gradient_via_central_difference(
    molecule: Molecule,
    basis_name: str,
    method: str,
    *,
    functional: Optional[str] = None,
    rhf_options: Any = None,
    uhf_options: Any = None,
    rks_options: Any = None,
    uks_options: Any = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    cisd_options: Any = None,
    selected_ci_options: Any = None,
    dmrg_options: Any = None,
    v2rdm_options: Any = None,
    transcorrelated_options: Any = None,
    casci_options: Any = None,
    caspt2_options: Any = None,
    nevpt2_options: Any = None,
    casscf_options: Any = None,
    active_space: Optional[tuple[int, int]] = None,
    cas_reference: Optional[str] = None,
    solvent: Any = None,
    dispersion_params: Any = None,
    step_bohr: float = 0.005,
) -> np.ndarray:
    """Central-difference energy gradient for wavefunction methods.

    Two-point central difference on each Cartesian degree of freedom.
    Returns gradE (not forces), shape (n_atoms, 3), in Ha/bohr. Both
    displaced evaluations carry the full wavefunction option set, so
    the FD gradient differentiates the same surface
    :func:`_evaluate_energy` reports.

    ECP systems (#643): the ECP centres on the mean-field options are
    absolute coordinates built for ``molecule``; each displaced evaluation
    moves them onto the displaced atoms and restores them afterwards.
    """
    n_atoms = len(list(molecule.atoms))
    grad = np.zeros((n_atoms, 3), dtype=float)
    _fd_mf_opts = _mean_field_options_for(
        method, rhf_options, uhf_options, rks_options, uks_options
    )

    for i in range(n_atoms):
        for c in range(3):
            pos = np.asarray([list(a.xyz) for a in molecule.atoms], dtype=float)

            pos_plus = pos.copy()
            pos_plus[i, c] += step_bohr
            mol_plus = Molecule(
                [Atom(int(a.Z), list(p)) for a, p in zip(molecule.atoms, pos_plus)],
                molecule.charge,
                molecule.multiplicity,
            )
            basis_plus = BasisSet(mol_plus, basis_name)
            with ecp_centres_follow_atoms(_fd_mf_opts, molecule, mol_plus):
                e_plus = _evaluate_energy(
                    mol_plus,
                    basis_plus,
                    method,
                    functional=functional,
                    rhf_options=rhf_options,
                    uhf_options=uhf_options,
                    rks_options=rks_options,
                    uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                    cisd_options=cisd_options,
                    selected_ci_options=selected_ci_options,
                    dmrg_options=dmrg_options,
                    v2rdm_options=v2rdm_options,
                    transcorrelated_options=transcorrelated_options,
                    casci_options=casci_options,
                    caspt2_options=caspt2_options,
                    nevpt2_options=nevpt2_options,
                    casscf_options=casscf_options,
                    active_space=active_space,
                    cas_reference=cas_reference,
                    solvent=solvent,
                    dispersion_params=dispersion_params,
                )

            pos_minus = pos.copy()
            pos_minus[i, c] -= step_bohr
            mol_minus = Molecule(
                [Atom(int(a.Z), list(p)) for a, p in zip(molecule.atoms, pos_minus)],
                molecule.charge,
                molecule.multiplicity,
            )
            basis_minus = BasisSet(mol_minus, basis_name)
            with ecp_centres_follow_atoms(_fd_mf_opts, molecule, mol_minus):
                e_minus = _evaluate_energy(
                    mol_minus,
                    basis_minus,
                    method,
                    functional=functional,
                    rhf_options=rhf_options,
                    uhf_options=uhf_options,
                    rks_options=rks_options,
                    uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                    cisd_options=cisd_options,
                    selected_ci_options=selected_ci_options,
                    dmrg_options=dmrg_options,
                    v2rdm_options=v2rdm_options,
                    transcorrelated_options=transcorrelated_options,
                    casci_options=casci_options,
                    caspt2_options=caspt2_options,
                    nevpt2_options=nevpt2_options,
                    casscf_options=casscf_options,
                    active_space=active_space,
                    cas_reference=cas_reference,
                    solvent=solvent,
                    dispersion_params=dispersion_params,
                )

            grad[i, c] = (e_plus - e_minus) / (2.0 * step_bohr)

    return grad


def _evaluate_energy(
    molecule: Molecule,
    basis: BasisSet,
    method: str,
    *,
    functional: Optional[str] = None,
    rhf_options: Any = None,
    uhf_options: Any = None,
    rks_options: Any = None,
    uks_options: Any = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    cisd_options: Any = None,
    selected_ci_options: Any = None,
    dmrg_options: Any = None,
    v2rdm_options: Any = None,
    transcorrelated_options: Any = None,
    casci_options: Any = None,
    caspt2_options: Any = None,
    nevpt2_options: Any = None,
    casscf_options: Any = None,
    active_space: Optional[tuple[int, int]] = None,
    cas_reference: Optional[str] = None,
    solvent: Any = None,
    dispersion_params: Any = None,
) -> float:
    """Evaluate the total energy at a given geometry (Ha).

    Forwards the full wavefunction option set (``active_space``,
    ``casscf_options``, ...) so per-step energies sample the same surface
    as the final single point. Pre-2026-06-12 these were dropped: a
    ``selected_ci`` optimization ran full-space CI with default options
    at every FD displacement -- a different (and far more expensive)
    surface than the truncated-active-space final energy.
    """
    from .runner import _run_single_point

    # This helper is energy-only.  Suppress the correlated-gradient opt-in so
    # optimizer-owned FD fallbacks do not recursively build a gradient (or
    # trip a deliberately unsupported analytic-gradient capability gate) at
    # every displaced energy.  The flag does not alter the PT2 energy.
    caspt2_energy_options = (
        replace(caspt2_options, compute_corr_grad=False)
        if caspt2_options is not None
        else None
    )
    nevpt2_energy_options = (
        replace(nevpt2_options, compute_corr_grad=False)
        if nevpt2_options is not None
        else None
    )
    result = _run_single_point(
        method,
        molecule,
        basis,
        functional=functional,
        rhf_options=rhf_options,
        uhf_options=uhf_options,
        rks_options=rks_options,
        uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
        cisd_options=cisd_options,
        selected_ci_options=selected_ci_options,
        dmrg_options=dmrg_options,
        v2rdm_options=v2rdm_options,
        transcorrelated_options=transcorrelated_options,
        casci_options=casci_options,
        caspt2_options=caspt2_energy_options,
        nevpt2_options=nevpt2_energy_options,
        casscf_options=casscf_options,
        active_space=active_space,
        cas_reference=cas_reference,
        solvent=solvent,
        _mrpt_gradient=False,
    )
    e = float(getattr(result, "energy", 0.0))

    if dispersion_params is not None:
        from .dispersion import compute_d3bj

        disp = compute_d3bj(molecule, dispersion_params)
        e += float(disp.energy)

    return e


# ---- Brent 1-D minimisation -----------------------------------------------


def brent_minimize_1d(
    f,
    a: float,
    b: float,
    c: float,
    *,
    tol: float = 1e-5,
    max_iter: int = 100,
    progress: bool = False,
) -> tuple[float, float, int]:
    """Brent's 1-D minimisation without derivatives.

    Finds a local minimum of the scalar function ``f`` within the
    bracketing triplet ``a < b < c`` where ``f(b) < f(a)`` and
    ``f(b) < f(c)``.  The algorithm combines golden-section search
    with inverse parabolic interpolation.

    This is the classic Brent (1973) algorithm as described in
    *Numerical Recipes* Sec. 10.2.

    Returns ``(x_min, f_min, n_eval)``.
    """
    CGOLD = 0.3819660112501051  # (3 - sqrt(5)) / 2

    if abs(f(b) - f(a)) < 1e-300 and abs(f(b) - f(c)) < 1e-300:
        return b, f(b), 3

    x = w = v = float(b)
    fx = fw = fv = f(b)
    e = 0.0
    d = 0.0
    n_eval = 3

    for iteration in range(1, max_iter + 1):
        xm = 0.5 * (a + c)
        tol1 = tol * abs(x) + 1e-12
        tol2 = 2.0 * tol1

        if abs(x - xm) <= tol2 - 0.5 * (c - a):
            return x, fx, n_eval

        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (c - x):
                if x >= xm:
                    e = a - x
                else:
                    e = c - x
                d = CGOLD * e
            else:
                d = p / q
                u = x + d
                if u - a < tol2 or c - u < tol2:
                    d = float(np.sign(xm - x)) * tol1
        else:
            if x >= xm:
                e = a - x
            else:
                e = c - x
            d = CGOLD * e

        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + float(np.sign(d)) * tol1

        fu = f(u)
        n_eval += 1

        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x
            v = w
            fv = fw
            w = x
            fw = fx
            x = u
            fx = fu
        else:
            if u < x:
                a = u
            else:
                c = u
            if fu <= fw or abs(w - x) < 1e-15:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or abs(v - x) < 1e-15 or abs(v - w) < 1e-15:
                v = u
                fv = fu

    return x, fx, n_eval


def _bracket_line_minimum(
    f,
    x0: float,
    fx0: float,
    *,
    step: float = 0.1,
    max_steps: int = 50,
    growth: float = 2.0,
) -> tuple[float, float, float, float, float, float, int]:
    """Bracket a local minimum along a 1-D line."""
    a = x0
    fa = fx0
    n_eval = 0
    b = x0 + step
    fb = f(b)
    n_eval += 1
    if fb > fa:
        a, b = b, a
        fa, fb = fb, fa
        step = -step
    c = b + step
    fc = f(c)
    n_eval += 1
    for _ in range(max_steps):
        if fc > fb:
            if a < c:
                return a, fa, b, fb, c, fc, n_eval
            else:
                return c, fc, b, fb, a, fa, n_eval
        step *= growth
        a, fa = b, fb
        b, fb = c, fc
        c = b + step
        fc = f(c)
        n_eval += 1
    return a, fa, b, fb, c, fc, n_eval


def _line_search_brent(
    f_line,
    x0: float,
    fx0: float,
    *,
    step: float = 0.05,
    brent_tol: float = 1e-5,
    brent_max_iter: int = 60,
    progress: bool = False,
) -> tuple[float, float, int]:
    """1-D line minimisation: bracket + Brent."""
    a, fa, b, fb, c, fc, n_bracket = _bracket_line_minimum(
        f_line,
        x0,
        fx0,
        step=step,
        max_steps=40,
        growth=1.8,
    )
    if fb >= fa or fb >= fc:
        vals = [(a, fa), (b, fb), (c, fc)]
        best = min(vals, key=lambda v: v[1])
        return best[0], best[1], n_bracket
    x_opt, f_opt, n_brent = brent_minimize_1d(
        f_line,
        min(a, c),
        b,
        max(a, c),
        tol=brent_tol,
        max_iter=brent_max_iter,
        progress=progress,
    )
    return x_opt, f_opt, n_bracket + n_brent


def optimize_molecule_brent(
    molecule: Molecule,
    basis_name: str,
    *,
    method: str = "rhf",
    functional: Optional[str] = None,
    rhf_options: Optional[RHFOptions] = None,
    uhf_options: Optional[UHFOptions] = None,
    rks_options: Optional[RKSOptions] = None,
    uks_options: Optional[UKSOptions] = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    cisd_options: Any = None,
    selected_ci_options: Any = None,
    dmrg_options: Any = None,
    v2rdm_options: Any = None,
    transcorrelated_options: Any = None,
    casci_options: Any = None,
    caspt2_options: Any = None,
    nevpt2_options: Any = None,
    casscf_options: Any = None,
    active_space: Optional[tuple[int, int]] = None,
    cas_reference: Optional[str] = None,
    max_iter: int = 100,
    conv_tol_grad: float = 4.5e-4,
    gradient_options: Optional[GradientOptions] = None,
    grid_options: Optional[GridOptions] = None,
    dispersion_params: Any = None,
    solvent: Any = None,
    record_trajectory: bool = True,
    progress: bool | ProgressLogger = False,
    fd_step_bohr: float = 0.005,
    freeze_indices: Optional[Sequence[int]] = None,
    line_search_step: float = 0.05,
    line_search_tol: float = 1e-5,
) -> MolecularOptimizeResult:
    """Relax molecular geometry using steepest-descent + Brent line search.

    At each geometry step the analytic (or finite-difference) gradient
    defines the steepest-descent direction.  A 1-D line search using
    Brent's method finds the optimal step length along that direction.

    This is a conservative, gradient-driven optimiser that never takes
    uphill steps.  Use ``optimizer_backend="brent"`` in ``run_job`` to
    select it from the top-level API.
    """
    plog = resolve_progress(progress, verbose=2)
    method_lower = method.lower()
    _refuse_unsupported_ecp_optimization(
        molecule,
        basis_name,
        method_lower,
        solvent=solvent,
        options=(rhf_options, uhf_options, rks_options, uks_options),
        route="optimize_molecule_brent",
    )
    # Keep one options object across steps. Besides enabling warm settings to
    # persist, this lets the first high-level SCF auto-attach an ECP sidecar;
    # subsequent gradients can mirror that exact route and subsequent
    # geometries can move its centers with the atoms.
    if method_lower == "rhf" and rhf_options is None:
        rhf_options = RHFOptions()
    elif method_lower == "uhf" and uhf_options is None:
        uhf_options = UHFOptions()
    elif method_lower == "rks" and rks_options is None:
        rks_options = RKSOptions()
    elif method_lower == "uks" and uks_options is None:
        uks_options = UKSOptions()
    grid_options = _resolve_molecular_ks_gradient_grid(
        method_lower,
        rks_options=rks_options,
        uks_options=uks_options,
        grid_options=grid_options,
    )
    # Keep this decision IDENTICAL to optimize_molecule (the L-BFGS-B primary)
    # and geomopt.MolecularSCFProvider: all three molecular optimizers must walk
    # the SAME surface per method.
    #   - rhf / uhf / rks / uks / rohf: validated analytic gradient.
    #   - casscf: validated analytic gradient INSIDE the state-specific,
    #     closed-shell envelope (v0.15.0 P0 fix); outside it (SA-CASSCF,
    #     open-shell, compute_wz="numerical") it falls back to full-energy FD.
    #     See _casscf_analytic_gradient_ok.
    #   - caspt2 / nevpt2: runner-supplied correlated gradient only inside
    #     each method's capability envelope (analytic IC-CASPT2 or relaxed
    #     NEVPT2 FD). Otherwise the optimizer owns the full-energy FD. See
    #     _mrpt_analytic_gradient_ok.
    _mean_field = {"rhf", "uhf", "rks", "uks", "rohf"}
    # GitLab #571: a range-separated or VV10-paired functional has an
    # incomplete analytic RKS/UKS gradient; walk the full-energy FD surface.
    _missing_gradient_terms = _mean_field_gradient_terms_missing(
        method_lower, functional, rks_options, uks_options
    )
    _has_analytic_gradient = (
        (method_lower in _mean_field and not _missing_gradient_terms)
        or (
            method_lower == "casscf"
            and _casscf_analytic_gradient_ok(molecule, casscf_options)
        )
        or (
            method_lower in ("caspt2", "nevpt2")
            and _mrpt_analytic_gradient_ok(
                method_lower,
                casscf_options,
                caspt2_options,
                nevpt2_options,
                solvent=solvent,
                molecule=molecule,
                basis=BasisSet(molecule, basis_name),
                active_space=active_space,
            )
        )
    )
    if _missing_gradient_terms:
        _announce_fd_fallback(
            method_lower,
            functional,
            rks_options,
            uks_options,
            _missing_gradient_terms,
            fd_step_bohr,
        )

    trajectory_frames: list[Molecule] = []
    trajectory_energies: list[float] = []

    n_atoms_total = len(list(molecule.atoms))
    if freeze_indices is None:
        _frozen_set: set[int] = set()
    else:
        _frozen_set = {int(i) for i in freeze_indices}
        bad = [i for i in _frozen_set if i < 0 or i >= n_atoms_total]
        if bad:
            raise ValueError(
                f"optimize_molecule_brent: freeze_indices {bad} out of range "
                f"[0, {n_atoms_total - 1}]"
            )

    def _apply_frozen_mask(grad: np.ndarray) -> np.ndarray:
        if not _frozen_set:
            return grad
        g = grad.reshape(-1, 3)
        for a in _frozen_set:
            g[a, :] = 0.0
        return grad

    # #643: follow the ECP centres (absolute coordinates on the SCF options,
    # built for the start geometry) onto every geometry the line search
    # visits, and refuse a set that sits on no start-geometry atom up front.
    _mf_opts = _mean_field_options_for(
        method_lower, rhf_options, uhf_options, rks_options, uks_options
    )
    if options_carry_ecp(_mf_opts):
        ecp_centre_atom_indices(_mf_opts, molecule)

    def _energy_and_gradient(mol: Molecule) -> tuple[float, np.ndarray]:
        basis = BasisSet(mol, basis_name)
        with ecp_centres_follow_atoms(_mf_opts, molecule, mol, gradient_options):
            return _energy_and_gradient_at(mol, basis)

    def _energy_and_gradient_at(mol: Molecule, basis: BasisSet) -> tuple[float, np.ndarray]:
            if _has_analytic_gradient:
                # Analytic path (mean-field, validated CASSCF, or opted-in
                # CASPT2/NEVPT2): SCF energy + validated analytic gradient.
                # caspt2_options/nevpt2_options must be forwarded so the
                # compute_corr_grad opt-in reaches the solver -- without them
                # the runner computes the bare CASSCF gradient. Dispersion is
                # folded into the gradient by _compute_molecular_gradient and
                # into the energy here, mirroring optimize_molecule's closures.
                e, res = _run_molecular_scf(
                    mol,
                    basis,
                    method_lower,
                    functional=functional,
                    rhf_options=rhf_options,
                    uhf_options=uhf_options,
                    rks_options=rks_options,
                    uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                    casscf_options=casscf_options,
                    active_space=active_space,
                    casci_options=casci_options,
                    caspt2_options=caspt2_options,
                    nevpt2_options=nevpt2_options,
                    cas_reference=cas_reference,
                    solvent=solvent,
                )
                grad = _compute_molecular_gradient(
                    mol,
                    basis,
                    res,
                    method_lower,
                    gradient_options=_ecp_gradient_options(gradient_options, _mf_opts),
                    grid_options=grid_options,
                    dispersion_params=dispersion_params,
                )
                if dispersion_params is not None:
                    from .dispersion import compute_d3bj

                    disp = compute_d3bj(mol, dispersion_params)
                    e += float(disp.energy)
            else:
                # Wavefunction path (selected_ci / dmrg / v2rdm /
                # transcorrelated_ci / casci / gated-out casscf / caspt2 /
                # nevpt2): full-energy central finite differences, never the
                # analytic gradient (see the _mean_field comment above). The
                # energy comes from _evaluate_energy -- the SAME helper the FD
                # displacements call -- so the reported energy and the gradient
                # differentiate a single, consistent surface (and the full
                # wavefunction option set + dispersion are folded in).
                e = _evaluate_energy(
                    mol,
                    basis,
                    method_lower,
                    functional=functional,
                    rhf_options=rhf_options,
                    uhf_options=uhf_options,
                    rks_options=rks_options,
                    uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                    cisd_options=cisd_options,
                    selected_ci_options=selected_ci_options,
                    dmrg_options=dmrg_options,
                    v2rdm_options=v2rdm_options,
                    transcorrelated_options=transcorrelated_options,
                    casci_options=casci_options,
                    caspt2_options=caspt2_options,
                    nevpt2_options=nevpt2_options,
                    casscf_options=casscf_options,
                    active_space=active_space,
                    cas_reference=cas_reference,
                    solvent=solvent,
                    dispersion_params=dispersion_params,
                )
                grad = _gradient_via_central_difference(
                    mol,
                    basis_name,
                    method_lower,
                    functional=functional,
                    rhf_options=rhf_options,
                    uhf_options=uhf_options,
                    rks_options=rks_options,
                    uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                    cisd_options=cisd_options,
                    selected_ci_options=selected_ci_options,
                    dmrg_options=dmrg_options,
                    v2rdm_options=v2rdm_options,
                    transcorrelated_options=transcorrelated_options,
                    casci_options=casci_options,
                    caspt2_options=caspt2_options,
                    nevpt2_options=nevpt2_options,
                    casscf_options=casscf_options,
                    active_space=active_space,
                    cas_reference=cas_reference,
                    solvent=solvent,
                    dispersion_params=dispersion_params,
                    step_bohr=fd_step_bohr,
                )
            grad_flat = np.asarray(grad, dtype=float).ravel()
            grad_flat = _apply_frozen_mask(grad_flat)
            return e, grad_flat

    mol_current = molecule
    e_current, grad_current = _energy_and_gradient(mol_current)
    grad_max = (
        float(np.max(np.abs(grad_current))) if grad_current.size else float("inf")
    )

    if record_trajectory:
        trajectory_frames.append(mol_current)
        trajectory_energies.append(e_current)

    plog.write_raw(
        f"\n  Geometry optimization (Brent) \u2014 {method.upper()}"
        + (f"/{functional}" if functional else "")
        + f"  basis={basis_name}\n"
        + f"  n_atoms={n_atoms_total}, max_iter={max_iter}, "
        + f"gtol={conv_tol_grad:.1e} Ha/bohr\n\n"
        + f"  step {0:3d}  E = {e_current:14.8f}  max|g| = {grad_max:.4e}"
    )

    converged = False
    for geo_step in range(1, max_iter + 1):
        if grad_max <= conv_tol_grad:
            converged = True
            break
        direction = -grad_current
        norm_dir = float(np.linalg.norm(direction))
        if norm_dir < 1e-15:
            converged = True
            break
        direction = direction / norm_dir

        def f_line(alpha: float) -> float:
            mol_trial = _flat_to_molecule(
                mol_current,
                _positions_to_flat(mol_current) + alpha * direction,
            )
            e_trial, _ = _energy_and_gradient(mol_trial)
            return e_trial

        alpha_opt, e_line, _n_line = _line_search_brent(
            f_line,
            0.0,
            e_current,
            step=line_search_step,
            brent_tol=line_search_tol,
            brent_max_iter=50,
        )

        plog.write_raw(
            f"    line search: alpha={alpha_opt:.4e}, E={e_line:.8f} Ha, "
            f"n_eval={_n_line}"
        )

        x_new = _positions_to_flat(mol_current) + alpha_opt * direction
        mol_current = _flat_to_molecule(mol_current, x_new)
        e_current, grad_current = _energy_and_gradient(mol_current)
        grad_max = (
            float(np.max(np.abs(grad_current))) if grad_current.size else float("inf")
        )

        if record_trajectory:
            trajectory_frames.append(mol_current)
            trajectory_energies.append(e_current)

        plog.write_raw(
            f"  step {geo_step:3d}  E = {e_current:14.8f}  "
            f"max|g| = {grad_max:.4e}"
        )

    if not converged:
        converged = grad_max <= conv_tol_grad

    return MolecularOptimizeResult(
        system=mol_current,
        energy=e_current,
        gradient=grad_current,
        n_iter=geo_step,
        converged=converged,
        trajectory_frames=trajectory_frames if record_trajectory else None,
        trajectory_energies=trajectory_energies if record_trajectory else None,
    )


# ---- Public API -----------------------------------------------------------


def optimize_molecule(
    molecule: Molecule,
    basis_name: str,
    *,
    method: str = "rhf",
    functional: Optional[str] = None,
    rhf_options: Optional[RHFOptions] = None,
    uhf_options: Optional[UHFOptions] = None,
    rks_options: Optional[RKSOptions] = None,
    uks_options: Optional[UKSOptions] = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    cisd_options: Any = None,
    selected_ci_options: Any = None,
    dmrg_options: Any = None,
    v2rdm_options: Any = None,
    transcorrelated_options: Any = None,
    casci_options: Any = None,
    caspt2_options: Any = None,
    nevpt2_options: Any = None,
    casscf_options: Any = None,
    active_space: Optional[tuple[int, int]] = None,
    cas_reference: Optional[str] = None,
    max_iter: int = 100,
    conv_tol_grad: float = 4.5e-4,
    conv_tol_energy: float = 1e-6,
    gradient_options: Optional[GradientOptions] = None,
    grid_options: Optional[GridOptions] = None,
    dispersion_params: Any = None,
    solvent: Any = None,
    record_trajectory: bool = True,
    progress: bool | ProgressLogger = False,
    fd_step_bohr: float = 0.005,
    freeze_indices: Optional[Sequence[int]] = None,
) -> MolecularOptimizeResult:
    """Relax molecular geometry using analytic gradients + L-BFGS-B.

    Parameters
    ----------
    molecule
        Starting geometry (Cartesian coordinates in bohr).
    basis_name
        Basis-set name (rebuilt at each geometry step).
    method
        ``"rhf"``, ``"uhf"``, ``"rks"``, ``"uks"``, or a wavefunction
        method (``"selected_ci"``, ``"dmrg"``, ``"v2rdm"``,
        ``"transcorrelated_ci"``, ``"casci"``, ``"casscf"``).
        Wavefunction methods fall back to central finite differences
        on the energy.
    functional
        XC functional string for ``"rks"`` / ``"uks"`` (e.g. ``"PBE"``).
    rhf_options / uhf_options / rks_options / uks_options
        Per-method SCF options. If ``None``, defaults are used.
    cisd_options / selected_ci_options / dmrg_options / v2rdm_options /
    transcorrelated_options / casci_options / caspt2_options /
    casscf_options
        Wavefunction-solver options, forwarded to every per-step
        energy evaluation (the FD path) exactly as the final single
        point receives them -- an SA-CASSCF optimization
        (``casscf_options=CASSCFOptions(nroots=2)``) walks the
        state-averaged surface it reports.
    active_space
        ``(n_active_orbitals, n_active_electrons)`` truncation for the
        wavefunction methods, applied at every per-step evaluation.
        Without it a ``selected_ci`` step would run full-space CI.
    cas_reference
        Reference-orbital choice for the determinant solvers
        (``"rhf"`` / ``"uhf"`` / ``"uno"``).
    max_iter
        Maximum L-BFGS-B iterations.
    conv_tol_grad
        Gradient convergence tolerance (Ha/bohr).  Default 4.5e-4
        corresponds to ~0.01 eV/Å -- tight enough for routine use.
    conv_tol_energy
        Energy convergence tolerance (Ha).  Controls the scipy
        ``ftol`` parameter.
    gradient_options
        Options for the analytic gradient kernels (density fitting,
        COSX, etc.).
    grid_options
        DFT integration grid options (RKS / UKS only).
    dispersion_params
        A :class:`D3BJParams` instance -- if provided, the D3-BJ
        energy and gradient are folded into the objective.
    solvent
        A :class:`SolventModel` or preset string / dict for CPCM
        implicit solvation (v0.9.0).
    record_trajectory
        If True (default), collect per-step geometries and energies
        for downstream visualisation (QVF animation player).
    progress
        If True, emit per-step energy and gradient norms to stdout. Pass a
        :class:`~vibeqc.progress.ProgressLogger` to select another live sink.
    fd_step_bohr
        Finite-difference step size for wavefunction-method gradients
        (bohr). Default 0.005 (≈ 0.0026 Å).
    freeze_indices
        Atom indices to hold fixed during the relaxation. Implemented
        via per-coordinate L-BFGS-B ``(fixed, fixed)`` bounds, mirroring
        :func:`vibeqc.bipole_optimize.relax_atoms`. The SCF + gradient
        still see every atom; the optimizer simply cannot move the
        frozen ones, and the reported ``|grad|`` excludes them so the
        convergence metric reflects only the free degrees of freedom.

    Returns
    -------
    MolecularOptimizeResult
    """
    from scipy.optimize import minimize

    plog = resolve_progress(progress, verbose=2)
    method_lower = method.lower()
    _refuse_unsupported_ecp_optimization(
        molecule,
        basis_name,
        method_lower,
        solvent=solvent,
        options=(rhf_options, uhf_options, rks_options, uks_options),
        route="optimize_molecule",
    )
    if method_lower == "rhf" and rhf_options is None:
        rhf_options = RHFOptions()
    elif method_lower == "uhf" and uhf_options is None:
        uhf_options = UHFOptions()
    elif method_lower == "rks" and rks_options is None:
        rks_options = RKSOptions()
    elif method_lower == "uks" and uks_options is None:
        uks_options = UKSOptions()
    grid_options = _resolve_molecular_ks_gradient_grid(
        method_lower,
        rks_options=rks_options,
        uks_options=uks_options,
        grid_options=grid_options,
    )
    # ROHF has a validated analytic gradient (compute_rohf_gradient); ROKS
    # does not yet (needs the molecular XC-gradient term) and stays on the FD
    # path. CASSCF uses its analytic gradient INSIDE the validated envelope
    # (state-specific, closed-shell -- the v0.15.0 P0 fix, FD-tight to
    # ~1e-7); outside it (SA-CASSCF, open-shell, compute_wz="numerical") it
    # stays on full-energy FD. See _casscf_analytic_gradient_ok. CASPT2/NEVPT2
    # use a runner-supplied correlated gradient only inside their respective
    # capability envelopes; otherwise the optimizer owns the full-energy FD.
    # Keep this decision IDENTICAL to optimize_molecule_brent and
    # geomopt.MolecularSCFProvider.
    _mean_field = {"rhf", "uhf", "rks", "uks", "rohf"}
    # GitLab #571: a range-separated or VV10-paired functional has an
    # incomplete analytic RKS/UKS gradient; walk the full-energy FD surface.
    _missing_gradient_terms = _mean_field_gradient_terms_missing(
        method_lower, functional, rks_options, uks_options
    )
    _has_analytic_gradient = (
        (method_lower in _mean_field and not _missing_gradient_terms)
        or (
            method_lower == "casscf"
            and _casscf_analytic_gradient_ok(molecule, casscf_options)
        )
        or (
            method_lower in ("caspt2", "nevpt2")
            and _mrpt_analytic_gradient_ok(
                method_lower,
                casscf_options,
                caspt2_options,
                nevpt2_options,
                solvent=solvent,
                molecule=molecule,
                basis=BasisSet(molecule, basis_name),
                active_space=active_space,
            )
        )
    )
    if _missing_gradient_terms:
        _announce_fd_fallback(
            method_lower,
            functional,
            rks_options,
            uks_options,
            _missing_gradient_terms,
            fd_step_bohr,
        )

    trajectory_frames: list[Molecule] = []
    trajectory_energies: list[float] = []
    _x0 = _positions_to_flat(molecule)

    n_atoms_total = len(list(molecule.atoms))
    if freeze_indices is None:
        _frozen_set: set[int] = set()
    else:
        _frozen_set = {int(i) for i in freeze_indices}
        bad = [i for i in _frozen_set if i < 0 or i >= n_atoms_total]
        if bad:
            raise ValueError(
                f"optimize_molecule: freeze_indices {bad} out of range "
                f"[0, {n_atoms_total - 1}]"
            )

    # L-BFGS-B bounds: pin frozen atoms by giving each Cartesian
    # component a (fixed, fixed) interval; free atoms get (None, None).
    _bounds: Optional[list[tuple[Optional[float], Optional[float]]]] = None
    if _frozen_set:
        _bounds = []
        for atom_i in range(n_atoms_total):
            if atom_i in _frozen_set:
                for k in range(3):
                    fixed = float(_x0[3 * atom_i + k])
                    _bounds.append((fixed, fixed))
            else:
                for _ in range(3):
                    _bounds.append((None, None))

    def _apply_frozen_mask(grad: np.ndarray) -> np.ndarray:
        """Zero gradient on frozen atoms (in-place) and return it."""
        if not _frozen_set:
            return grad
        g = grad.reshape(-1, 3)
        for a in _frozen_set:
            g[a, :] = 0.0
        return grad

    # #643: the ECP centres on the mean-field options are absolute
    # coordinates built for ``molecule``; every closure below moves them
    # onto the geometry it evaluates and restores them on exit. Refuse a set
    # that sits on no start-geometry atom before the first SCF.
    _mf_opts = _mean_field_options_for(
        method_lower, rhf_options, uhf_options, rks_options, uks_options
    )
    if options_carry_ecp(_mf_opts):
        ecp_centre_atom_indices(_mf_opts, molecule)

    def _follow_ecp(mol: Molecule):
        return ecp_centres_follow_atoms(_mf_opts, molecule, mol, gradient_options)

    # Pre-construct a scipy gradient closure. The "force" minimizers
    # expect dE/dx (not -dE/dx), so we pass the gradient as-is.
    if _has_analytic_gradient:

        def _grad_fn(x: np.ndarray) -> np.ndarray:
            mol = _flat_to_molecule(molecule, x)
            basis = BasisSet(mol, basis_name)
            with _follow_ecp(mol):
                return _grad_at(mol, basis)

        def _grad_at(mol: Molecule, basis: BasisSet) -> np.ndarray:
            e, res = _run_molecular_scf(
                mol,
                basis,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                casscf_options=casscf_options,
                active_space=active_space,
                casci_options=casci_options,
                caspt2_options=caspt2_options,
                nevpt2_options=nevpt2_options,
                cas_reference=cas_reference,
                solvent=solvent,
            )
            grad = _compute_molecular_gradient(
                mol,
                basis,
                res,
                method_lower,
                gradient_options=_ecp_gradient_options(gradient_options, _mf_opts),
                grid_options=grid_options,
                dispersion_params=dispersion_params,
            )
            return _apply_frozen_mask(grad.ravel())

        def _energy_fn(x: np.ndarray) -> float:
            mol = _flat_to_molecule(molecule, x)
            basis = BasisSet(mol, basis_name)
            with _follow_ecp(mol):
                return _energy_at(mol, basis)

        def _energy_at(mol: Molecule, basis: BasisSet) -> float:
            e, _res = _run_molecular_scf(
                mol,
                basis,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                casscf_options=casscf_options,
                active_space=active_space,
                casci_options=casci_options,
                caspt2_options=caspt2_options,
                nevpt2_options=nevpt2_options,
                cas_reference=cas_reference,
                solvent=solvent,
            )
            if dispersion_params is not None:
                from .dispersion import compute_d3bj

                disp = compute_d3bj(mol, dispersion_params)
                e += float(disp.energy)
            return e

    else:
        # Wavefunction methods -- FD on energy. Both closures forward the
        # full wavefunction option set so the gradient and the energy
        # sample the surface the final single point reports.
        def _grad_fn(x: np.ndarray) -> np.ndarray:
            mol = _flat_to_molecule(molecule, x)
            with _follow_ecp(mol):
                return _fd_grad_at(mol)

        def _fd_grad_at(mol: Molecule) -> np.ndarray:
            grad_flat = _gradient_via_central_difference(
                mol,
                basis_name,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                cisd_options=cisd_options,
                selected_ci_options=selected_ci_options,
                dmrg_options=dmrg_options,
                v2rdm_options=v2rdm_options,
                transcorrelated_options=transcorrelated_options,
                casci_options=casci_options,
                caspt2_options=caspt2_options,
                nevpt2_options=nevpt2_options,
                casscf_options=casscf_options,
                active_space=active_space,
                cas_reference=cas_reference,
                solvent=solvent,
                dispersion_params=dispersion_params,
                step_bohr=fd_step_bohr,
            ).ravel()
            return _apply_frozen_mask(grad_flat)

        def _energy_fn(x: np.ndarray) -> float:
            mol = _flat_to_molecule(molecule, x)
            basis = BasisSet(mol, basis_name)
            with _follow_ecp(mol):
                return _fd_energy_at(mol, basis)

        def _fd_energy_at(mol: Molecule, basis: BasisSet) -> float:
            return _evaluate_energy(
                mol,
                basis,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                cisd_options=cisd_options,
                selected_ci_options=selected_ci_options,
                dmrg_options=dmrg_options,
                v2rdm_options=v2rdm_options,
                transcorrelated_options=transcorrelated_options,
                casci_options=casci_options,
                caspt2_options=caspt2_options,
                nevpt2_options=nevpt2_options,
                casscf_options=casscf_options,
                active_space=active_space,
                cas_reference=cas_reference,
                solvent=solvent,
                dispersion_params=dispersion_params,
            )

    # Combined objective: scipy calls `fun` first, then `jac` at the
    # same x. We evaluate energy once in `fun` and stash it so `jac`
    # can reuse the SCF result in the analytic-gradient path. For FD
    # methods the caching is in the gradient evaluation itself.
    if _has_analytic_gradient:
        _cache: dict[str, Any] = {
            "result": None,
            "mol": None,
            "basis": None,
            "energy": float("nan"),
            "gradient_options": gradient_options,
        }

        def _fun_cached(x: np.ndarray) -> float:
            mol = _flat_to_molecule(molecule, x)
            basis = BasisSet(mol, basis_name)
            with _follow_ecp(mol):
                energy = _fun_at(mol, basis)
                # Mirror after SCF: that high-level call may just have
                # auto-attached an ECP basis sidecar to a previously plain
                # options object. The centres still sit on ``mol`` here;
                # _jac_cached runs after this block restores the SCF options,
                # so it must use this captured gradient-options value rather
                # than re-derive it later.
                _cache["gradient_options"] = _ecp_gradient_options(
                    gradient_options, _mf_opts
                )
                return energy

        def _fun_at(mol: Molecule, basis: BasisSet) -> float:
            e, res = _run_molecular_scf(
                mol,
                basis,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                    rohf_options=rohf_options, roks_options=roks_options,
                casscf_options=casscf_options,
                active_space=active_space,
                casci_options=casci_options,
                caspt2_options=caspt2_options,
                nevpt2_options=nevpt2_options,
                cas_reference=cas_reference,
                solvent=solvent,
            )
            _cache["result"] = res
            _cache["mol"] = mol
            _cache["basis"] = basis
            if dispersion_params is not None:
                from .dispersion import compute_d3bj

                disp = compute_d3bj(mol, dispersion_params)
                e += float(disp.energy)
            _cache["energy"] = e
            return e

        def _jac_cached(x: np.ndarray) -> np.ndarray:
            # Reuse the cached SCF result to avoid double-running.
            if _cache.get("result") is not None and _cache.get("mol") is not None:
                grad = _compute_molecular_gradient(
                    _cache["mol"],
                    _cache["basis"],
                    _cache["result"],
                    method_lower,
                    gradient_options=_cache.get("gradient_options", gradient_options),
                    grid_options=grid_options,
                    dispersion_params=dispersion_params,
                )
                _cache["result"] = None  # clear for next iteration
                return _apply_frozen_mask(grad.ravel())
            # Fallback: re-evaluate (shouldn't normally happen).
            return _grad_fn(x)

        _objective = _fun_cached
        _jacobian = _jac_cached
    else:
        _objective = _energy_fn
        _jacobian = _grad_fn

    # Callback to collect trajectory.
    if record_trajectory:

        def _callback(xk: np.ndarray) -> None:
            mol_frame = _flat_to_molecule(molecule, xk)
            trajectory_frames.append(mol_frame)
            # scipy guarantee: fun(xk) was called just before the
            # callback. Use the cached energy to avoid a duplicate
            # SCF evaluation.
            if _has_analytic_gradient:
                e_frame = _cache.get("energy", float("nan"))
            else:
                e_frame = _energy_fn(xk)
            trajectory_energies.append(e_frame)
            plog.write_raw(
                f"  step {len(trajectory_frames):3d}  E = {e_frame:14.8f} Ha"
            )

    else:
        _callback = None  # type: ignore[assignment]

    # ---- run the scipy optimizer ------------------------------------------
    plog.write_raw(
        f"\n  Geometry optimization -- {method.upper()}"
        + (f"/{functional}" if functional else "")
        + f"  basis={basis_name}\n"
        + f"  n_atoms={len(list(molecule.atoms))}, "
        + f"max_iter={max_iter}, "
        + f"gtol={conv_tol_grad:.1e} Ha/bohr\n\n"
    )

    # Feed the energy through the objective so the cache is primed.
    e_start = _objective(_x0)

    if record_trajectory:
        trajectory_frames.append(molecule)
        trajectory_energies.append(e_start)

    _lbfgsb_jac = _jacobian if _has_analytic_gradient else _grad_fn
    _lbfgsb_options = {
        "maxiter": max_iter,
        "gtol": conv_tol_grad,
        "ftol": conv_tol_energy,
    }

    res = minimize(
        _objective,
        _x0,
        method="L-BFGS-B",
        jac=_lbfgsb_jac,
        callback=_callback,
        bounds=_bounds,
        options=_lbfgsb_options,
    )

    # Independent convergence gate. scipy sets res.success on EITHER gtol
    # OR ftol, so res.success alone can claim convergence at a
    # non-stationary geometry when ftol trips first (2026-05-31 audit,
    # F1). Gate on the actual max-component force and report that
    # inf-norm (the gtol metric), not the 2-norm (F5).
    grad_final = _grad_fn(res.x) if res.success else res.jac
    converged, grad_max = _gradient_converged(
        bool(res.success), grad_final, conv_tol_grad
    )

    # Optimizer polish: L-BFGS-B's ftol (relative-energy) criterion can
    # halt the descent at a geometry whose largest force component is
    # still above conv_tol_grad -- common on a shallow constrained
    # surface (e.g. a relaxed-scan step), where scipy reports success
    # after spending only a handful of its allotted iterations. When
    # that happens, restart L-BFGS-B from res.x with the energy
    # criterion disabled (ftol=0): the fresh inverse-Hessian
    # approximation escapes the premature stop, and the restart then
    # builds curvature and runs to the gradient tolerance (or the
    # remaining budget) instead of re-tripping ftol after a single step.
    # A plain restart that kept the caller's ftol would just re-trip it
    # and crawl downhill by steepest descent -- too slow to reach
    # conv_tol_grad inside a tight max_iter. Repeat until the gradient
    # gate passes or the max_iter budget is spent.
    #
    # Gate the polish on conv_tol_energy <= conv_tol_grad. A deliberately
    # LOOSE energy tolerance (conv_tol_energy > conv_tol_grad -- e.g.
    # conv_tol_energy=1e-3 with conv_tol_grad=1e-6) is the caller asking
    # to stop on energy *before* the gradient converges; honor it and
    # leave the early stop in place (the F1 gate still reports
    # converged=False honestly). That confines this polish to the case
    # it is meant for -- a tight/default energy tolerance where an
    # ftol-stop above conv_tol_grad is an unwanted artifact, not a
    # request -- and keeps the deliberate ftol-stop scenario in
    # test_optimize_molecule_converged_implies_stationary intact.
    #
    # This does NOT touch the gradient gate (the shared stationarity
    # definition for optimize_molecule and the BIPOLE relaxers) -- it
    # only spends the iteration budget the ftol stop left on the table.
    # (2026-06-10: fixes the relaxed-scan converged-flag false-negative.)
    _polish_ftol_stop = conv_tol_energy <= conv_tol_grad
    total_nit = int(res.nit)
    while (
        _polish_ftol_stop
        and not converged
        and bool(res.success)
        and total_nit < max_iter
    ):
        _lbfgsb_options["maxiter"] = max_iter - total_nit
        _lbfgsb_options["ftol"] = 0.0
        res = minimize(
            _objective,
            res.x,
            method="L-BFGS-B",
            jac=_lbfgsb_jac,
            callback=_callback,
            bounds=_bounds,
            options=_lbfgsb_options,
        )
        if int(res.nit) == 0:
            # The restart took no step -- it sits at a stationary point of
            # the local model, or scipy could not improve. Further
            # restarts cannot help; stop and report the (still
            # non-converged) gradient honestly.
            break
        total_nit += int(res.nit)
        grad_final = _grad_fn(res.x) if res.success else res.jac
        converged, grad_max = _gradient_converged(
            bool(res.success), grad_final, conv_tol_grad
        )

    mol_opt = _flat_to_molecule(molecule, res.x)

    plog.write_raw(
        f"\n  Geometry optimization: {total_nit} iters, "
        f"E = {res.fun:.8f} Ha, "
        f"max|grad| = {grad_max:.4e} Ha/bohr, "
        f"converged={converged}"
    )

    return MolecularOptimizeResult(
        system=mol_opt,
        energy=float(res.fun),
        gradient=grad_final if grad_final is not None else np.array([]),
        n_iter=total_nit,
        converged=converged,
        trajectory_frames=trajectory_frames if record_trajectory else None,
        trajectory_energies=trajectory_energies if record_trajectory else None,
    )
