"""Provider contracts for the uniform geometry optimization framework.

``EnergyGradientProvider`` and ``HessianProvider`` are the two abstract
interfaces that decouple optimizer logic from electronic-structure
backends.  Every theory family (semiempirical, HF, KS-DFT, post-HF,
periodic, MACE) exposes the same pair of contracts; the optimizer
never knows which backend supplied the numbers.

Backend adapters for the existing vibe-qc SCF paths are provided so
the new framework can be wired into ``run_job`` without duplicating
the SCF-dispatch logic already in :mod:`vibeqc.molecular_optimize`.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np

from .._vibeqc_core import (
    Atom,
    BasisSet,
    GradientOptions,
    GridOptions,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UKSOptions,
)

# ---------------------------------------------------------------------------
# Abstract provider protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class EnergyGradientProvider(Protocol):
    """A callable that returns (energy [Ha], gradient [Ha/bohr]) at a geometry.

    The gradient is the energy derivative gradE (shape ``(n_atoms, 3)`` or
    flat ``(3.n_atoms,)``), **not** the force (-gradE).  Providers may
    choose to return a flat or shaped array; the optimizer always calls
    ``np.asarray(gradient, dtype=float).ravel()`` to normalise.

    This is a structural protocol -- any object with the matching
    ``__call__`` signature satisfies it without explicit subclassing.
    """

    def __call__(self, molecule: Molecule) -> tuple[float, np.ndarray]: ...


@runtime_checkable
class HessianProvider(Protocol):
    """Returns the energy Hessian (``(3N, 3N)``) at a geometry in Ha/bohr^2.

    When an exact Hessian is unavailable the provider can return a
    model (diagonal, identity-scaled) or raise :exc:`NotImplementedError`
    to signal the optimizer should use a gradient-only method.
    """

    def __call__(self, molecule: Molecule) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Convenience aliases for client code
# ---------------------------------------------------------------------------

# A Hessian-vector product: H.v, where v is a flat (3N,) array.
# Useful for iterative trust-region solvers and mode-following.
HessianVectorProductProvider = Callable[[Molecule, np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Backend adapters -- wrap existing SCF dispatch into provider contracts
# ---------------------------------------------------------------------------


class MolecularSCFProvider:
    """An :class:`EnergyGradientProvider` backed by vibe-qc's molecular SCF.

    Wraps the same ``_run_molecular_scf`` / ``_compute_molecular_gradient``
    logic that :func:`vibeqc.molecular_optimize.optimize_molecule` uses,
    plus the finite-difference fallback for wavefunction methods.

    Parameters
    ----------
    basis_name : str
        Basis-set name (rebuilt per geometry).
    method : str
        ``"rhf"``, ``"uhf"``, ``"rks"``, ``"uks"``, ``"rohf"``, ``"casscf"``,
        or a wavefunction method that requires FD gradients.
    functional : str or None
        XC functional name for RKS/UKS.
    rhf_options, uhf_options, rks_options, uks_options : optional
        Per-method SCF option structs.
    gradient_options : GradientOptions or None
    grid_options : GridOptions or None
        Analytic-gradient grid override. When omitted for RKS/UKS, the
        gradient inherits the SCF options grid so both differentiate the same
        surface.
    grid_level : str
        DFT integration-grid preset applied to RKS/UKS options whose grid is
        untouched, including a supplied options object whose ``grid`` was
        never customised (GitLab #663). Default ``"orca-defgrid3"``. A
        customised grid wins.
    dispersion_params : D3BJParams or None
    solvent : any or None
        CPCM solvation parameters.
    """

    # Methods whose analytic nuclear gradient the geomopt optimizers may walk.
    # Keep this IDENTICAL to optimize_molecule / optimize_molecule_brent's
    # analytic-gradient set so all three molecular optimizers walk the SAME
    # surface per method. CASSCF is handled separately in __call__ (it is
    # analytic only inside the validated state-specific / closed-shell
    # envelope -- _casscf_analytic_gradient_ok), as are
    # caspt2 / nevpt2 (runner-supplied correlated gradient only inside the
    # method-specific _mrpt_analytic_gradient_ok envelope). Everything else
    # uses full-energy central FD.
    _HAS_ANALYTIC_GRADIENT: set[str] = {"rhf", "uhf", "rks", "uks", "rohf"}

    def __init__(
        self,
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
        gradient_options: Optional[GradientOptions] = None,
        grid_options: Optional[GridOptions] = None,
        grid_level: str = "orca-defgrid3",
        dispersion_params: Any = None,
        solvent: Any = None,
        # Wavefunction FD params -- forwarded when method has no analytic grad.
        fd_step_bohr: float = 0.005,
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
        warm_start: bool = True,
    ):
        self._basis_name = basis_name
        self._method = method.lower()
        self._functional = functional
        self._rhf_options = rhf_options
        self._uhf_options = uhf_options
        self._rks_options = rks_options
        self._uks_options = uks_options
        self._rohf_options = rohf_options
        self._roks_options = roks_options
        # Preserve ``None`` so an ECP reference can derive an exact
        # GradientOptions mirror after the high-level SCF wrapper has
        # auto-attached the basis sidecar. An explicit object keeps its JK
        # settings; its ECP fields are synchronized per geometry below.
        self._gradient_options = gradient_options
        self._grid_options = grid_options
        self._grid_level = grid_level
        self._dispersion_params = dispersion_params
        self._solvent = solvent
        self._fd_step_bohr = fd_step_bohr
        self._cisd_options = cisd_options
        self._selected_ci_options = selected_ci_options
        self._dmrg_options = dmrg_options
        self._v2rdm_options = v2rdm_options
        self._transcorrelated_options = transcorrelated_options
        self._casci_options = casci_options
        self._caspt2_options = caspt2_options
        self._nevpt2_options = nevpt2_options
        self._casscf_options = casscf_options
        self._active_space = active_space
        self._cas_reference = cas_reference
        # Warm-start bookkeeping: the previous step's converged SCF
        # result, reused as an initial_guess=READ restart on the next
        # geometry (the wrappers project the prior density onto the new
        # basis). Optimizer steps are small, so the projected density is
        # near-converged and cuts per-step SCF iteration counts several-
        # fold -- the difference between finishing inside a queue
        # walltime and aborting (glycine SI matrix rp167: 12 h kills).
        self._warm_start = bool(warm_start)
        self._last_scf_result: Any = None
        self._ecp_atom_indices: Any = None
        # GitLab #571: the FD-fallback note is written to the .out once.
        self._fd_fallback_announced = False

    # Mean-field methods whose wrappers accept read_from= and whose
    # results carry a reusable converged density.
    _WARM_START_METHODS: set[str] = {"rhf", "uhf", "rks", "uks", "rohf", "roks"}

    @property
    def missing_gradient_terms(self) -> list[str]:
        """Analytic-gradient terms the RKS / UKS kernel omits for this
        provider's functional (GitLab #571), empty when the analytic
        gradient is complete. Non-empty means :meth:`__call__` walks the
        full-energy central-FD surface instead."""
        from ..molecular_optimize import _mean_field_gradient_terms_missing

        return _mean_field_gradient_terms_missing(
            self._method, self._functional, self._rks_options, self._uks_options
        )

    @property
    def has_analytic_gradient(self) -> bool:
        """Static capability check: methods always analytic regardless of the
        molecule. CASSCF (validated envelope only) and CASPT2/NEVPT2
        (CASSCF-referenced + compute_corr_grad opt-in) are conditionally
        analytic and decided per-call in :meth:`__call__`, so they are not
        listed here. RKS / UKS with a range-separated or VV10-paired
        functional report ``False`` (GitLab #571: the kernel omits those
        terms, see :attr:`missing_gradient_terms`)."""
        if self._method not in self._HAS_ANALYTIC_GRADIENT:
            return False
        return not self.missing_gradient_terms

    def _mean_field_options(self) -> Any:
        """Materialize (and remember) the options struct for the active
        mean-field method, so warm-start can toggle its initial_guess."""
        if self._method == "rohf":
            from ..rohf import ROHFOptions
            self._rohf_options = self._rohf_options or ROHFOptions()
            return self._rohf_options
        if self._method == "roks":
            from ..roks import ROKSOptions
            self._roks_options = self._roks_options or ROKSOptions()
            return self._roks_options
        if self._method == "rhf":
            self._rhf_options = self._rhf_options or RHFOptions()
            return self._rhf_options
        if self._method == "uhf":
            self._uhf_options = self._uhf_options or UHFOptions()
            return self._uhf_options
        if self._method == "rks":
            from ..runner import _apply_grid_level, ks_options_need_grid_default

            if self._rks_options is None:
                self._rks_options = RKSOptions()
            # GitLab #663: an untouched grid on a caller-provided object
            # receives the preset too; a customised grid wins.
            if ks_options_need_grid_default(self._rks_options):
                _apply_grid_level(self._rks_options.grid, self._grid_level)
            if self._grid_options is None:
                self._grid_options = self._rks_options.grid
            return self._rks_options
        if self._method == "uks":
            from ..runner import _apply_grid_level, ks_options_need_grid_default

            if self._uks_options is None:
                self._uks_options = UKSOptions()
            if ks_options_need_grid_default(self._uks_options):
                _apply_grid_level(self._uks_options.grid, self._grid_level)
            if self._grid_options is None:
                self._grid_options = self._uks_options.grid
            return self._uks_options
        return None

    def _run_scf_once(
        self, molecule: Molecule, basis: BasisSet, read_from: Any
    ) -> tuple[float, Any]:
        from ..molecular_optimize import _run_molecular_scf
        from ..ecp_metadata import (
            ecp_centre_atom_indices,
            options_carry_ecp,
            reposition_ecp_centres,
        )
        from ..gradient_options import copy_ecp_fields

        mean_field_options = self._mean_field_options()
        if mean_field_options is not None:
            if self._ecp_atom_indices is not None:
                reposition_ecp_centres(
                    mean_field_options, molecule, self._ecp_atom_indices
                )
            elif options_carry_ecp(mean_field_options):
                # Manual centers must describe the first geometry. Validate
                # them before SCF work and keep their atom identities.
                self._ecp_atom_indices = ecp_centre_atom_indices(
                    mean_field_options, molecule
                )

        energy, result = _run_molecular_scf(
            molecule,
            basis,
            self._method,
            functional=self._functional,
            rhf_options=self._rhf_options,
            uhf_options=self._uhf_options,
            rks_options=self._rks_options,
            uks_options=self._uks_options,
            rohf_options=self._rohf_options, roks_options=self._roks_options,
            casscf_options=self._casscf_options,
            active_space=self._active_space,
            casci_options=self._casci_options,
            caspt2_options=self._caspt2_options,
            nevpt2_options=self._nevpt2_options,
            cas_reference=self._cas_reference,
            solvent=self._solvent,
            read_from=read_from,
        )
        # The high-level wrapper may have attached an XML or inline basis
        # sidecar during this first call. Record its atom mapping once, then
        # keep the exact same centers on each subsequently visited geometry.
        if mean_field_options is not None and options_carry_ecp(
            mean_field_options
        ):
            if self._ecp_atom_indices is None:
                self._ecp_atom_indices = ecp_centre_atom_indices(
                    mean_field_options, molecule
                )
            if self._gradient_options is not None:
                copy_ecp_fields(self._gradient_options, mean_field_options)
        return energy, result

    def _run_scf_with_warm_start(
        self, molecule: Molecule, basis: BasisSet
    ) -> tuple[float, Any]:
        """Run the SCF, warm-starting from the previous step's converged
        density when possible, and fail closed on nonconvergence.

        * Warm start: mean-field methods only (rhf/uhf/rks/uks), gas
          phase, and only when the user has not selected their own READ
          guess. The previous result is handed to the wrapper as
          ``read_from`` with ``initial_guess=READ`` set for the call and
          restored afterwards; the wrapper projects the density onto the
          current geometry's basis. Optimizer steps are small, so this
          typically cuts the per-step SCF iteration count several-fold.
        * If the warm-started SCF fails to converge, retry once from the
          cold default guess before giving up -- a projected density can
          occasionally steer DIIS into a worse basin than SAD.
        * A still-nonconverged mean-field SCF raises instead of feeding
          a garbage gradient to the optimizer (the ASE calculator has
          raised its typed error in this situation since v0.15.x; the
          native path previously walked on silently).
        """
        from ..guess import coerce_initial_guess
        from .._vibeqc_core import InitialGuess
        opts = self._mean_field_options()
        can_warm = (
            self._warm_start
            and self._method in self._WARM_START_METHODS
            and self._solvent is None
            and self._last_scf_result is not None
            and opts is not None
            # Respect an explicit user READ guess (read_path / read_from
            # workflows drive their own restart source).
            and coerce_initial_guess(opts.initial_guess) != InitialGuess.READ
        )

        if can_warm:
            saved_guess = opts.initial_guess
            saved_spins = list(getattr(opts, "atomic_spins", []))
            from .._vibeqc_core import InitialGuess

            opts.initial_guess = InitialGuess.READ
            if hasattr(opts, "atomic_spins"):
                opts.atomic_spins = []
            try:
                e, res = self._run_scf_once(
                    molecule, basis, self._last_scf_result
                )
                from ..guess import GuessSelection, coerce_initial_guess, resolve_initial_guess
                res.guess_selection = GuessSelection(
                    coerce_initial_guess(saved_guess), resolve_initial_guess(
                        molecule, saved_guess,
                        is_open_shell=self._method in ("uhf", "uks", "rohf", "roks")),
                    InitialGuess.READ)
            finally:
                opts.initial_guess = saved_guess
                if hasattr(opts, "atomic_spins"):
                    opts.atomic_spins = saved_spins
            if not bool(getattr(res, "converged", True)):
                # Cold retry: the projected density steered the SCF into
                # a nonconvergent tail; fall back to the default guess.
                self._last_scf_result = None
                e, res = self._run_scf_once(molecule, basis, None)
        else:
            e, res = self._run_scf_once(molecule, basis, None)

        if self._method in self._WARM_START_METHODS and not bool(
            getattr(res, "converged", True)
        ):
            raise RuntimeError(
                f"vibeqc {self._method.upper()} SCF did not converge after "
                f"{int(getattr(res, 'n_iter', -1))} iterations during "
                f"geometry optimization (energy = "
                f"{float(getattr(res, 'energy', float('nan')))} Ha). "
                "The optimizer step may be pathological (see "
                "docs/user_guide/geometry_optimization.md), or the SCF "
                "needs tighter convergence aids at this geometry."
            )

        if self._warm_start and self._method in self._WARM_START_METHODS:
            self._last_scf_result = res
        return e, res

    def __call__(self, molecule: Molecule) -> tuple[float, np.ndarray]:
        """Evaluate (energy, gradient) at *molecule*."""
        from ..molecular_optimize import (
            _casscf_analytic_gradient_ok,
            _ecp_gradient_options,
            _mrpt_analytic_gradient_ok,
            _refuse_unsupported_ecp_optimization,
        )

        _refuse_unsupported_ecp_optimization(
            molecule,
            self._basis_name,
            self._method,
            solvent=self._solvent,
            options=(
                self._rhf_options,
                self._uhf_options,
                self._rks_options,
                self._uks_options,
            ),
            route="MolecularSCFProvider",
        )

        basis = BasisSet(molecule, self._basis_name)
        # CASSCF is analytic only inside the validated state-specific /
        # closed-shell envelope; that depends on the
        # molecule's multiplicity (per-call) as well as casscf_options, so the
        # decision is made here rather than in the static property. CASPT2 /
        # NEVPT2 use their runner-supplied correlated gradients only inside
        # their method-specific envelopes, decided here too.
        use_analytic = (
            self.has_analytic_gradient
            or (
                self._method == "casscf"
                and _casscf_analytic_gradient_ok(molecule, self._casscf_options)
            )
            or (
                self._method in ("caspt2", "nevpt2")
                and _mrpt_analytic_gradient_ok(
                    self._method,
                    self._casscf_options,
                    self._caspt2_options,
                    self._nevpt2_options,
                    solvent=self._solvent,
                    molecule=molecule,
                    basis=basis,
                    active_space=self._active_space,
                )
            )
        )
        if use_analytic:
            # Mean-field path, state-specific CASSCF, or opted-in
            # CASPT2/NEVPT2: SCF energy + validated analytic gradient.
            from ..molecular_optimize import _compute_molecular_gradient

            e, res = self._run_scf_with_warm_start(molecule, basis)
            grad = _compute_molecular_gradient(
                molecule,
                basis,
                res,
                self._method,
                gradient_options=_ecp_gradient_options(
                    self._gradient_options, self._mean_field_options()
                ),
                grid_options=self._grid_options,
                dispersion_params=self._dispersion_params,
            )
            if self._dispersion_params is not None:
                from ..dispersion import compute_d3bj

                disp = compute_d3bj(molecule, self._dispersion_params)
                e += float(disp.energy)
        else:
            # Wavefunction path (gated-out casscf / caspt2 / nevpt2,
            # selected_ci / ...) and RKS / UKS with a functional whose
            # analytic gradient omits terms (GitLab #571): full-energy
            # central finite differences, never the analytic gradient (see
            # the _HAS_ANALYTIC_GRADIENT comment). The energy comes from
            # _evaluate_energy -- the same helper the FD displacements call
            # -- so energy and gradient differentiate one consistent surface
            # and dispersion is folded in once.
            from ..molecular_optimize import (
                _announce_fd_fallback,
                _evaluate_energy,
                _gradient_via_central_difference,
            )

            if self._method in ("rks", "uks"):
                # Materialize the grid-level options so the FD surface is
                # the one the analytic branch would have used.
                self._mean_field_options()
                missing = self.missing_gradient_terms
                if missing and not self._fd_fallback_announced:
                    _announce_fd_fallback(
                        self._method,
                        self._functional,
                        self._rks_options,
                        self._uks_options,
                        missing,
                        self._fd_step_bohr,
                    )
                    self._fd_fallback_announced = True

            e = _evaluate_energy(
                molecule,
                basis,
                self._method,
                functional=self._functional,
                rhf_options=self._rhf_options,
                uhf_options=self._uhf_options,
                rks_options=self._rks_options,
                uks_options=self._uks_options,
            rohf_options=self._rohf_options, roks_options=self._roks_options,
                cisd_options=self._cisd_options,
                selected_ci_options=self._selected_ci_options,
                dmrg_options=self._dmrg_options,
                v2rdm_options=self._v2rdm_options,
                transcorrelated_options=self._transcorrelated_options,
                casci_options=self._casci_options,
                caspt2_options=self._caspt2_options,
                nevpt2_options=self._nevpt2_options,
                casscf_options=self._casscf_options,
                active_space=self._active_space,
                cas_reference=self._cas_reference,
                solvent=self._solvent,
                dispersion_params=self._dispersion_params,
            )
            grad = _gradient_via_central_difference(
                molecule,
                self._basis_name,
                self._method,
                functional=self._functional,
                rhf_options=self._rhf_options,
                uhf_options=self._uhf_options,
                rks_options=self._rks_options,
                uks_options=self._uks_options,
            rohf_options=self._rohf_options, roks_options=self._roks_options,
                cisd_options=self._cisd_options,
                selected_ci_options=self._selected_ci_options,
                dmrg_options=self._dmrg_options,
                v2rdm_options=self._v2rdm_options,
                transcorrelated_options=self._transcorrelated_options,
                casci_options=self._casci_options,
                caspt2_options=self._caspt2_options,
                nevpt2_options=self._nevpt2_options,
                casscf_options=self._casscf_options,
                active_space=self._active_space,
                cas_reference=self._cas_reference,
                solvent=self._solvent,
                dispersion_params=self._dispersion_params,
                step_bohr=self._fd_step_bohr,
            )
        return e, np.asarray(grad, dtype=float).ravel()


class MolecularHessianFDProvider:
    """A :class:`HessianProvider` that computes the Hessian via finite differences.

    Uses the same energy-only evaluation path as the FD gradient fallback.
    Central differences on each Cartesian degree of freedom produce
    ``d^2E/dxᵢdxⱼ``.

    Parameters
    ----------
    provider : EnergyGradientProvider
        Used for the energy evaluations (so dispersion/solvation are included).
    n_atoms : int
        Number of atoms (for allocating the Hessian array).
    step_bohr : float
        Displacement step in bohr (default 0.005).
    """

    def __init__(
        self,
        provider: EnergyGradientProvider,
        n_atoms: int,
        step_bohr: float = 0.005,
    ):
        self._provider = provider
        self._n_atoms = n_atoms
        self._step = step_bohr

    def __call__(self, molecule: Molecule) -> np.ndarray:
        n = 3 * self._n_atoms
        hess = np.zeros((n, n), dtype=float)
        pos0 = _positions_native(molecule)

        # Diagonal: (E(x+h) - 2E(x) + E(x-h)) / h^2
        for i in range(n):
            mol_p = _molecule_at(molecule, pos0, i, self._step)
            mol_m = _molecule_at(molecule, pos0, i, -self._step)
            ep, _ = self._provider(mol_p)
            em, _ = self._provider(mol_m)
            e0 = self._provider(molecule)[0]
            hess[i, i] = (ep - 2.0 * e0 + em) / (self._step * self._step)

        # Off-diagonal: (E(x+h₁+h₂) - E(x+h₁-h₂) - E(x-h₁+h₂) + E(x-h₁-h₂)) / (4h^2)
        for i in range(n):
            for j in range(i + 1, n):
                mol_pp = _molecule_at2(molecule, pos0, i, self._step, j, self._step)
                mol_pm = _molecule_at2(molecule, pos0, i, self._step, j, -self._step)
                mol_mp = _molecule_at2(molecule, pos0, i, -self._step, j, self._step)
                mol_mm = _molecule_at2(molecule, pos0, i, -self._step, j, -self._step)
                epp, _ = self._provider(mol_pp)
                epm, _ = self._provider(mol_pm)
                emp, _ = self._provider(mol_mp)
                emm, _ = self._provider(mol_mm)
                val = (epp - epm - emp + emm) / (4.0 * self._step * self._step)
                hess[i, j] = val
                hess[j, i] = val

        return hess


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _positions_native(molecule: Molecule) -> np.ndarray:
    """Flat (3N,) array of atom positions in bohr."""
    flat: list[float] = []
    for atom in molecule.atoms:
        flat.extend(atom.xyz)
    return np.array(flat, dtype=float)


def _molecule_at(
    template: Molecule,
    pos0: np.ndarray,
    idx: int,
    delta: float,
) -> Molecule:
    """Return a new Molecule with position *idx* displaced by *delta*."""
    pos = pos0.copy()
    pos[idx] += delta
    return _flat_to_molecule(template, pos)


def _molecule_at2(
    template: Molecule,
    pos0: np.ndarray,
    i: int,
    di: float,
    j: int,
    dj: float,
) -> Molecule:
    """Return a new Molecule with two coordinates displaced."""
    pos = pos0.copy()
    pos[i] += di
    pos[j] += dj
    return _flat_to_molecule(template, pos)


def _flat_to_molecule(template: Molecule, x: np.ndarray) -> Molecule:
    """Rebuild a Molecule from flat Cartesian coordinates (bohr)."""
    n_atoms = len(list(template.atoms))
    new_atoms: list[Atom] = []
    for k in range(n_atoms):
        xyz = [float(x[3 * k + c]) for c in range(3)]
        new_atoms.append(Atom(int(template.atoms[k].Z), xyz))
    return Molecule(new_atoms, template.charge, template.multiplicity)
