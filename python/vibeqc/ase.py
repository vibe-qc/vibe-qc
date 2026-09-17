"""ASE (Atomic Simulation Environment) Calculator wrapper for vibeqc.

Exposes a `VibeQC` Calculator that drives RHF / UHF / RKS from the usual
ASE API:

    from ase.build import molecule
    from vibeqc.ase import VibeQC

    atoms = molecule("H2O")
    atoms.calc = VibeQC(basis="sto-3g")           # defaults to HF
    print(atoms.get_potential_energy())           # eV

    atoms.calc = VibeQC(basis="6-31g*", functional="B3LYP")   # DFT
    print(atoms.get_potential_energy())

Method selection is implicit:
  - No `functional` (or `functional=None`) -> Hartree-Fock.
    multiplicity == 1  => run_rhf,  multiplicity > 1  => run_uhf.
  - `functional` set   -> Kohn-Sham DFT.
    multiplicity == 1  => run_rks,  multiplicity > 1  => NotImplementedError
    (UKS / unrestricted KS is a future phase).

Forces are implemented for HF (RHF + UHF); DFT forces require the DFT
gradient code (a future phase) and raise PropertyNotImplementedError
until then.

Unit conventions: ASE works in Ångström + eV, vibe-qc in bohr + Hartree.
Conversions use ASE's own constants (`ase.units.Bohr`, `ase.units.Hartree`).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    from ase.calculators.calculator import (
        Calculator,
        PropertyNotImplementedError,
        all_changes,
    )
    from ase.units import Bohr, Hartree
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "vibeqc.ase requires ASE. Install it with `pip install ase` into "
        "your vibe-qc venv, or reinstall vibe-qc normally so its runtime "
        "dependencies are present."
    ) from exc

from . import (
    Atom,
    BasisSet,
    D3BJParams,
    Molecule,
    RHFOptions,
    RKSOptions,
    UHFOptions,
    UKSOptions,
    compute_d3bj,
    compute_gradient,
    compute_gradient_rks,
    compute_gradient_uhf,
    compute_gradient_uks,
    compute_hessian_fd,
    compute_hessian_rhf_analytic,
    compute_hessian_rks_analytic,
    compute_hessian_uhf_analytic,
    d3bj_params_for,
    dipole_moment,
    dipole_polarizability_rhf,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)
from .ecp_metadata import (
    ecp_centre_atom_indices,
    effective_nuclear_charges,
    molecular_result_has_ecp_operator,
    options_carry_ecp,
    reposition_ecp_centres,
)
from .gradient_options import gradient_options_from_scf
from .ase_optimizers import (  # noqa: F401  (re-exported for user scripts)
    ase_optimizer_availability,
    available_ase_optimizers,
    ensure_ase_scipy_compat,
    format_ase_optimizer_report,
    resolve_ase_optimizer,
)
from .scf_log import log_scf_trace

# Importing vibeqc.ase is the canonical first step of every ASE-driven
# vibe-qc workflow, so restore the SciPy aliases old ASE line-search
# optimizers need (ASE < 3.23 + SciPy >= 1.14) right away -- before the
# user's own `from ase.optimize import ...` line runs.
ensure_ase_scipy_compat()

_log = logging.getLogger("vibeqc.ase")


class VibeQCSCFConvergenceError(RuntimeError):
    """Raised when the ASE calculator reaches a non-converged SCF result.

    ASE optimizers evaluate energies and forces through repeated calculator
    calls. A non-converged SCF has no trustworthy force, so the calculator
    aborts that optimizer step with a typed error that batch harnesses can
    classify without scraping traceback text. The message carries a
    geometry diagnostic (closest atom pair) because the most common cause
    in optimizer loops is a pathological step, not an intrinsically hard
    electronic structure (glycine SI matrix: GPMin / MDMin drove the
    geometry to states with SCF "energies" 100 Ha above the minimum).
    """

    def __init__(self, method: str, result: Any, geometry_note: str = "") -> None:
        self.method = str(method)
        self.n_iter = int(getattr(result, "n_iter", -1))
        self.energy_hartree = float(getattr(result, "energy", float("nan")))
        self.scf_result = result
        self.geometry_note = geometry_note
        message = (
            f"vibeqc {self.method} SCF did not converge after "
            f"{self.n_iter} iterations (energy = {self.energy_hartree} Ha)"
        )
        if geometry_note:
            message += (
                f"\n  {geometry_note}"
                "\n  If this happened inside a geometry optimization the "
                "optimizer likely took a pathological step; bound the step "
                "size (e.g. maxstep=, or MDMin's dt=) or switch to a "
                "quasi-Newton optimizer (BFGS / LBFGS / BFGSLineSearch)."
            )
        super().__init__(message)


class VibeQCGeometryError(ValueError):
    """Raised for a geometry no Gaussian-basis SCF can meaningfully treat.

    The canonical trigger is an optimizer proposing a step that collapses
    two atoms onto each other: the SCF then grinds through its full
    iteration budget on a nonsensical Hamiltonian before failing. The
    calculator rejects such geometries up front (see the
    ``min_pair_distance`` parameter) so the optimizer-level failure is
    immediate and names the offending atom pair.
    """


def _closest_pair_note(atoms: Any) -> tuple[float, str]:
    """Return (min pair distance in Angstrom, human-readable note)."""
    positions = np.asarray(atoms.positions, dtype=float)
    n = len(positions)
    if n < 2:
        return float("inf"), ""
    deltas = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(deltas, axis=-1)
    dists[np.diag_indices(n)] = np.inf
    i, j = np.unravel_index(int(np.argmin(dists)), dists.shape)
    d_min = float(dists[i, j])
    symbols = atoms.get_chemical_symbols()
    note = (
        f"closest atom pair: {symbols[i]}{i}-{symbols[j]}{j} at "
        f"{d_min:.3f} Angstrom"
    )
    return d_min, note


# The SCF-options -> GradientOptions mirror lives in vibeqc.gradient_options
# so the FD Hessian and run_irc can use it without importing ASE (#576). The
# private name is kept for existing importers.
_gradient_options_from_scf = gradient_options_from_scf


def _dipole_nuclear_charges(molecule, result, method_options, solvent_result):
    """Return the nuclear charges proven for this SCF Hamiltonian.

    Gas-phase high-level SCF wrappers attach ECP metadata to the caller's
    method options, so :func:`effective_nuclear_charges` remains the source
    there. CPCM deliberately works on an internal options copy; an automatic
    basis sidecar therefore never reaches ``method_options``. Its
    :class:`SolventResult` instead carries the exact per-atom ``z_eff`` used
    to build the CPCM Hcore. Refuse an ECP result if that verified vector is
    unavailable rather than silently restoring bare nuclear charges.
    """

    if (
        solvent_result is None
        or getattr(solvent_result, "screening", None) is None
    ):
        # ``solvent='vacuum'`` returns a SolventResult for API uniformity but
        # runs the ordinary high-level gas-phase SCF, which attaches any
        # automatic ECP sidecar directly to method_options.
        return effective_nuclear_charges(molecule, method_options)

    if not bool(getattr(result, "ecp_provenance_verified", False)):
        raise NotImplementedError(
            "VibeQC dipole: CPCM SCF ECP provenance is unknown or "
            "unverified; refusing to infer nuclear charges from stale "
            "caller options"
        )
    result_has_ecp = molecular_result_has_ecp_operator(result)
    z_eff = getattr(solvent_result, "z_eff", None)
    if not result_has_ecp:
        if z_eff is not None:
            raise ValueError(
                "VibeQC dipole: all-electron CPCM result unexpectedly "
                "carries ECP effective nuclear charges"
            )
        # CPCM's verified result says this Hamiltonian is all-electron. Do
        # not consult a caller options object that the CPCM copy may have
        # normalized or superseded.
        return effective_nuclear_charges(molecule, None)

    if z_eff is None:
        raise NotImplementedError(
            "VibeQC dipole: the verified ECP+CPCM result does not carry its "
            "per-atom effective nuclear charges; refusing a bare-Z dipole"
        )

    charges = np.asarray(z_eff, dtype=np.float64)
    n_atoms = len(molecule.atoms)
    if charges.shape != (n_atoms,) or not np.all(np.isfinite(charges)):
        raise ValueError(
            "VibeQC dipole: invalid ECP+CPCM effective nuclear-charge "
            f"provenance; expected {n_atoms} finite per-atom values, got "
            f"shape {charges.shape}"
        )

    bare = np.asarray(
        [float(atom.Z) for atom in molecule.atoms], dtype=np.float64
    )
    implied_ncore_float = float(np.sum(bare - charges))
    implied_ncore = int(round(implied_ncore_float))
    result_ncore = int(getattr(result, "ecp_total_ncore"))
    if (
        abs(implied_ncore_float - implied_ncore) > 1e-8
        or implied_ncore != result_ncore
    ):
        raise ValueError(
            "VibeQC dipole: ECP+CPCM effective nuclear charges imply "
            f"{implied_ncore_float} replaced-core electrons but the SCF "
            f"result reports {result_ncore}"
        )
    return charges


class VibeQC(Calculator):
    """ASE Calculator routing to vibe-qc's RHF / UHF / RKS drivers.

    ECP systems: the ``*_options`` objects are held by reference and their
    ECP centres (absolute coordinates) are moved onto the current atoms at
    every ``calculate()`` (#643); after an optimisation the options describe
    the last geometry evaluated, not the one they were built for.

    Parameters (all via kwargs at construction)
    -------------------------------------------
    basis : str
        libint-recognized basis-set name (e.g. "sto-3g", "6-31g*", "cc-pvdz",
        "def2-tzvp"). Standard bases plus anything under
        ``basis_library/custom/``.
    charge : int
        Net molecular charge. Default 0.
    multiplicity : int
        Spin multiplicity 2S+1. Default 1. Values > 1 route HF to UHF
        (or ROHF when ``restricted_open=True``) and DFT to UKS.
    restricted_open : bool
        When True, open-shell HF (multiplicity > 1, no ``functional``) is
        run as spin-restricted ROHF (spin-pure, analytic gradient) instead
        of UHF. Default False. Open-shell DFT stays UKS in this calculator
        (ROKS has no analytic gradient yet); use
        ``run_job(method="roks")`` for restricted-open-shell DFT.
    functional : str | None
        If ``None`` (default), do Hartree-Fock. Otherwise, do Kohn-Sham DFT
        with the given functional. Accepts "LDA", "SVWN", "PBE", "BLYP",
        "B3LYP", "SLATER", or a comma-separated list of libxc XC_... IDs.
    grid_level : str
        DFT integration-grid preset applied to ``rks_options`` /
        ``uks_options`` whose grid is untouched, including a supplied
        options object whose ``grid`` was never customised (GitLab #663).
        Default ``"orca-defgrid3"``; accepted values match
        :func:`vibeqc.runner.run_job`. A customised grid wins.
    rhf_options, uhf_options, rks_options
        Optional ``RHFOptions`` / ``UHFOptions`` / ``RKSOptions`` for fine
        control of SCF convergence, DIIS, damping, initial guess, grid.
    dispersion
        Optional post-SCF D3(BJ) dispersion correction. Accepts:

        * ``None`` (default) -- no dispersion.
        * A :class:`D3BJParams` instance -- used directly.
        * A functional name (``"pbe"``, ``"b3lyp"``, ...) -- looked up via
          :func:`vibeqc.d3bj_params_for` with ``backend="auto"``.
        * ``True`` / ``"d3bj"`` -- use D3-BJ params for the current
          ``functional`` (the SCF functional).

        When set, the returned ASE energy includes the dispersion
        contribution and the returned forces include the dispersion
        gradient. The dispersion energy is also logged on
        ``vibeqc.ase`` at INFO level.
    min_pair_distance : float | None
        Reject geometries whose closest atom pair is below this distance
        (Angstrom) with :class:`VibeQCGeometryError` before running any
        SCF. Default 0.25. Set ``None`` (or 0) to disable for deliberate
        ultra-compressed scans. Protects optimizer loops from burning the
        full SCF iteration budget on a collapsed geometry after a
        pathological step (GPMin / MDMin on ab initio surfaces).
    """

    implemented_properties = [
        "energy",
        "forces",
        "free_energy",     # ASE alias; identical to "energy" for non-finite-T SCF
        "dipole",          # eÅ -- atoms.get_dipole_moment()
        "hessian",         # eV/Å^2 -- atoms.get_property("hessian")
        "polarizability",  # Å^3 -- RHF only; raises for non-RHF
    ]

    default_parameters = {
        "basis": "sto-3g",
        "charge": 0,
        "multiplicity": 1,
        "functional": None,
        # Keep the implicit KS surface identical to run_job(). A customised
        # grid on a supplied options object still wins (#663).
        "grid_level": "orca-defgrid3",
        # When True, open-shell HF (multiplicity > 1, no functional) is
        # routed to spin-restricted ROHF instead of UHF -- a spin-pure
        # determinant with an analytic gradient. Default False keeps the
        # historical UHF/UKS auto-selection.
        "restricted_open": False,
        # Reject geometries whose closest atom pair falls below this
        # distance (Angstrom) with VibeQCGeometryError *before* running
        # the SCF. Optimizers taking pathological steps (GPMin surrogate
        # jumps, MDMin overshoots) otherwise burn the full SCF iteration
        # budget on a meaningless Hamiltonian. 0.25 A is far below any
        # bound internuclear distance (H2 is 0.74 A); set None/0 for
        # deliberate ultra-compressed scans.
        "min_pair_distance": 0.25,
    }

    nolabel = True

    def __init__(
        self,
        rhf_options: RHFOptions | None = None,
        uhf_options: UHFOptions | None = None,
        rks_options: RKSOptions | None = None,
        uks_options: UKSOptions | None = None,
        rohf_options: "ROHFOptions | None" = None,
        dispersion: Any = None,
        solvent: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        from .rohf import ROHFOptions as _ROHFOptions

        self._rhf_options = rhf_options if rhf_options is not None else RHFOptions()
        self._uhf_options = uhf_options if uhf_options is not None else UHFOptions()
        self._rks_options = rks_options if rks_options is not None else RKSOptions()
        self._uks_options = uks_options if uks_options is not None else UKSOptions()
        # GitLab #663: the preset applies to an untouched grid, also on a
        # caller-provided options object; a customised grid wins.
        from .runner import _apply_grid_level, ks_options_need_grid_default

        grid_level = str(self.parameters["grid_level"])
        if ks_options_need_grid_default(self._rks_options):
            _apply_grid_level(self._rks_options.grid, grid_level)
        if ks_options_need_grid_default(self._uks_options):
            _apply_grid_level(self._uks_options.grid, grid_level)
        self._rohf_options = (
            rohf_options if rohf_options is not None else _ROHFOptions()
        )
        # #643: per-method map from ECP centre to atom index, established at
        # the first geometry where the centres sit on the atoms; every later
        # calculate() moves the centres onto the current atoms (permanently,
        # on the options object the calculator owns for its lifetime).
        self._ecp_atom_indices: dict[str, tuple[list[int], list[int]]] = {}
        # Dispersion is resolved lazily (at calculate() time) to the
        # underlying D3BJParams -- we can't know what the functional
        # will be until the user places the atoms.
        self._dispersion_spec = dispersion
        # v0.9.0 CPCM / COSMO implicit solvation. ``None`` keeps gas-
        # phase semantics; any other value reroutes calculate() through
        # vibeqc.solvation.run_cpcm_scf. The spec is kept verbatim and
        # resolved per calculate() call so it can be a preset name,
        # a numeric e, a dict, or a SolventModel.
        self._solvent = solvent

    # GitLab #571 -- analytic-gradient completeness for the DFT branches.
    #: Central-difference step (bohr) of the FD force fallback; the same
    #: default the molecular optimizers use (``fd_step_bohr=0.005``).
    fd_step_bohr: float = 0.005

    def _missing_gradient_terms(self, method: str) -> list[str]:
        """Terms the analytic RKS/UKS gradient omits for the resolved
        functional (empty for LDA / GGA / meta-GGA / global hybrids)."""
        from .gradient_terms import functional_gradient_terms_missing

        if method == "RKS":
            return functional_gradient_terms_missing(
                self._rks_options.functional, spin=1
            )
        if method == "UKS":
            return functional_gradient_terms_missing(
                self._uks_options.functional, spin=2
            )
        return []

    def _fd_gradient(self, mol: Molecule, method: str) -> np.ndarray:
        """Full-energy central-FD gradient (Ha/bohr) on the calculator's
        own SCF options; dispersion is folded in by the caller as for the
        analytic branch. Announced once per calculator in the ``.out``."""
        from .molecular_optimize import _gradient_via_central_difference
        from .output import write

        missing = self._missing_gradient_terms(method)
        if not getattr(self, "_fd_fallback_announced", False):
            name = (
                self._rks_options.functional
                if method == "RKS"
                else self._uks_options.functional
            )
            write(
                f"  Note: the analytic {method} gradient for functional "
                f"{name!r} omits {'; '.join(missing)}. The ASE calculator "
                "uses full-energy central finite differences (step "
                f"{self.fd_step_bohr:g} bohr) instead (GitLab #571).\n"
            )
            _log.info(
                "%s analytic gradient incomplete for %r; using central FD "
                "(step %g bohr, GitLab #571)",
                method, name, self.fd_step_bohr,
            )
            self._fd_fallback_announced = True
        return np.asarray(
            _gradient_via_central_difference(
                mol,
                self.parameters["basis"],
                method.lower(),
                rks_options=self._rks_options if method == "RKS" else None,
                uks_options=self._uks_options if method == "UKS" else None,
                # The displaced energies must be on the calculator's own
                # surface: gas-phase FD under a reaction field is the defect
                # this fallback exists to avoid.
                solvent=self._solvent,
                step_bohr=self.fd_step_bohr,
            ),
            dtype=float,
        )

    def _resolve_dispersion_params(self) -> D3BJParams | None:
        """Convert the user-supplied ``dispersion=`` argument into a
        concrete :class:`D3BJParams`, or ``None`` if no correction is
        requested. Called once per ``calculate()``."""
        spec = self._dispersion_spec
        if spec is None or spec is False:
            return None
        if isinstance(spec, D3BJParams):
            return spec
        # ``True`` or a functional name -> look up via d3bj_params_for.
        functional = self.parameters.get("functional")
        if spec is True or (isinstance(spec, str) and spec.strip().lower() in
                             ("d3bj", "true", "yes", "on")):
            if not functional:
                raise ValueError(
                    "VibeQC(dispersion=True) requires a `functional=` so "
                    "D3-BJ parameters can be looked up."
                )
            lookup = functional
        elif isinstance(spec, str):
            lookup = spec
        else:
            raise TypeError(
                f"VibeQC(dispersion=...) expected str, bool, D3BJParams, or "
                f"None; got {type(spec).__name__}"
            )
        params = d3bj_params_for(lookup)
        if params is None:
            raise ValueError(
                f"VibeQC(dispersion={spec!r}): no D3-BJ parameters for "
                f"functional {lookup!r}"
            )
        return params

    # ---- method selection ------------------------------------------------

    def _select_method(self, multiplicity: int) -> str:
        func = self.parameters.get("functional")
        if func:
            # restricted_open + DFT would be ROKS, but ROKS has no analytic
            # gradient yet, so the ASE (analytic-force) calculator keeps
            # UKS for open-shell DFT; use run_job(method="roks") for ROKS.
            return "UKS" if multiplicity > 1 else "RKS"
        if self.parameters.get("restricted_open") and multiplicity > 1:
            return "ROHF"
        return "UHF" if multiplicity > 1 else "RHF"

    # ---- ASE integration -------------------------------------------------

    def calculate(  # type: ignore[override]
        self,
        atoms=None,
        properties=("energy",),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        # Fail fast on collapsed geometries -- see min_pair_distance in
        # default_parameters. Cheap (one pairwise-distance pass) next to
        # even a single SCF iteration.
        min_pair = self.parameters.get("min_pair_distance")
        if min_pair:
            d_min, note = _closest_pair_note(self.atoms)
            if d_min < float(min_pair):
                raise VibeQCGeometryError(
                    f"geometry rejected before SCF: {note} is below "
                    f"min_pair_distance={float(min_pair):.3f} Angstrom. "
                    "If this happened inside a geometry optimization the "
                    "optimizer took a pathological step; bound the step "
                    "size (maxstep=, MDMin dt=) or switch optimizers. "
                    "Pass VibeQC(min_pair_distance=None) to disable this "
                    "check for deliberate ultra-compressed scans."
                )

        # Convert ASE atoms (Å) to vibe-qc atoms (bohr).
        positions_bohr = self.atoms.positions / Bohr
        zs = self.atoms.numbers
        mol = Molecule(
            [Atom(int(z), list(pos)) for z, pos in zip(zs, positions_bohr)],
            int(self.parameters["charge"]),
            int(self.parameters["multiplicity"]),
        )
        basis_obj = BasisSet(mol, self.parameters["basis"])
        method = self._select_method(mol.multiplicity)

        method_label = (
            f"{method} / {self.parameters['functional']}"
            if method == "RKS" else method
        )
        _log.info(
            "%s / %s  n_atoms=%d  n_electrons=%d  charge=%d  mult=%d",
            method_label,
            self.parameters["basis"],
            len(mol.atoms),
            mol.n_electrons(),
            mol.charge,
            mol.multiplicity,
        )

        # Resolve the underlying method options up-front so the CPCM
        # branch and the gas-phase branch share the same options.
        if method == "RKS":
            self._rks_options.functional = self.parameters["functional"]
        elif method == "UKS":
            self._uks_options.functional = self.parameters["functional"]
        method_opts = {
            "RHF": self._rhf_options, "UHF": self._uhf_options,
            "RKS": self._rks_options, "UKS": self._uks_options,
            "ROHF": self._rohf_options,
        }[method]
        self._sync_ecp_centres(method, method_opts, mol, system_changes)

        if method == "ROHF" and self._solvent is not None:
            raise NotImplementedError(
                "VibeQC(restricted_open=True) with implicit solvation is not "
                "yet supported; use UHF (restricted_open=False) or gas-phase "
                "ROHF. See handovers/HANDOVER_ROHF.md."
            )

        # CPCM branch (v0.9.0). Returns the underlying SCF result with
        # solvation diagnostics attached; downstream code reads
        # ``result.energy`` as the in-solvent total (IID 148) and
        # ``result.solvent_result.e_gas`` for the gas-phase reference.
        if self._solvent is not None:
            from .solvation import run_cpcm_scf
            from .solvation.driver import _solvent_aware_scf_result
            sol = run_cpcm_scf(
                mol, basis_obj,
                method=method.lower(),
                solvent=self._solvent,
                options=method_opts,
            )
            # Attribute-forwarding wrapper -- pybind11 SCF result types
            # are read-only, so plain ``result.solvent_result = sol``
            # silently fails. The wrapper exposes ``.solvent_result``,
            # ``.e_solv``, ``.energy_in_solvent`` while forwarding
            # every other lookup (``.energy``, ``.density``, ...) to the
            # inner C++ object.
            result = _solvent_aware_scf_result(sol)
        elif method == "RHF":
            result = run_rhf(mol, basis_obj, self._rhf_options)
        elif method == "UHF":
            result = run_uhf(mol, basis_obj, self._uhf_options)
        elif method == "ROHF":
            from .rohf import run_rohf

            result = run_rohf(mol, basis_obj, self._rohf_options)
        elif method == "RKS":
            result = run_rks(mol, basis_obj, self._rks_options)
        elif method == "UKS":
            result = run_uks(mol, basis_obj, self._uks_options)
        else:
            raise AssertionError(f"unreachable: method = {method}")

        log_scf_trace(result)
        # The SCF wrappers attach the basis sidecar's ECP data on their first
        # call when the options carry none; that happens at THIS geometry, so
        # record the centre->atom mapping now (before the convergence raise,
        # the attach precedes the SCF).
        if method not in self._ecp_atom_indices and options_carry_ecp(method_opts):
            self._ecp_atom_indices[method] = ecp_centre_atom_indices(method_opts, mol)

        if not result.converged:
            _, geometry_note = _closest_pair_note(self.atoms)
            raise VibeQCSCFConvergenceError(method, result, geometry_note)

        # In-solvent total energy is the gas-phase electronic + nuclear
        # *plus* the polarisation E_solv (1/2 q . V). The CPCM driver
        # returns both pieces on result.solvent_result; expose them
        # separately in the ASE results dict so downstream analysis
        # (Hessian / thermo / NMR) sees the medium-corrected energy.
        # ``result.energy`` itself is the in-solvent total (IID 148), so
        # the gas reference must come from the SolventResult, not from
        # the forwarding wrapper.
        sol_result = getattr(result, "solvent_result", None)
        if sol_result is not None:
            self.results["energy"] = sol_result.energy * Hartree
            self.results["e_solv"] = sol_result.e_solv * Hartree
            self.results["e_gas"] = sol_result.e_gas * Hartree
        else:
            self.results["energy"] = result.energy * Hartree

        # Post-SCF dispersion correction. Applied after the SCF energy
        # is set so the decomposition stays available if the caller
        # wants it (see results["e_scf"] / results["e_dispersion"]).
        disp_params = self._resolve_dispersion_params()
        self.results["e_scf"] = result.energy * Hartree
        self.results["e_dispersion"] = 0.0
        if disp_params is not None:
            want_grad = "forces" in properties
            disp = compute_d3bj(mol, disp_params, with_gradient=want_grad)
            e_disp_eV = float(disp.energy) * Hartree
            self.results["e_dispersion"] = e_disp_eV
            self.results["energy"] += e_disp_eV
            _log.info(
                "D3-BJ correction: E_disp = %+.6e Ha (%.4f kcal/mol)",
                float(disp.energy),
                float(disp.energy) * 627.5094740631,
            )

        if "forces" in properties:
            sol_result = getattr(result, "solvent_result", None)
            if (
                sol_result is not None
                and method in ("RKS", "UKS")
                and self._missing_gradient_terms(method)
            ):
                # GitLab #571, in solvent: cpcm_gradient's gas-phase piece is
                # that same incomplete analytic kernel, so the reaction field
                # would be added to a wrong surface. Differentiate the solvated
                # energy numerically instead -- _fd_gradient carries this
                # calculator's solvent into every displaced evaluation.
                gradient = self._fd_gradient(mol, method)
            elif sol_result is not None:
                # CPCM-aware analytic gradient -- includes the gas-phase
                # SCF piece, the closed-form (dA/dR) and (dV^nuc/dR)
                # contributions, and an integral-FD (dV^elec/dR) piece.
                # See vibeqc.solvation.gradient.cpcm_gradient for the
                # full decomposition + envelope-theorem derivation.
                from .solvation import cpcm_gradient
                grid_opts = (
                    self._rks_options.grid if method == "RKS"
                    else self._uks_options.grid if method == "UKS"
                    else None
                )
                gradient = cpcm_gradient(
                    result, mol, basis_obj, sol_result,
                    method=method.lower(), grid_options=grid_opts,
                )
            elif method == "RHF":
                grad_opts = _gradient_options_from_scf(method_opts)
                gradient = compute_gradient(
                    mol, basis_obj, result, options=grad_opts,
                )
            elif method == "UHF":
                grad_opts = _gradient_options_from_scf(method_opts)
                gradient = compute_gradient_uhf(
                    mol, basis_obj, result, options=grad_opts,
                )
            elif method == "ROHF":
                from .rohf import compute_rohf_gradient

                grad_opts = _gradient_options_from_scf(method_opts)
                gradient = compute_rohf_gradient(
                    mol, basis_obj, result, gradient_options=grad_opts,
                )
            elif method in ("RKS", "UKS") and self._missing_gradient_terms(
                method
            ):
                # GitLab #571: the analytic RKS/UKS kernel omits the
                # long-range exact-exchange gradient of a range-separated
                # hybrid and the VV10 nonlocal gradient. Differentiate the
                # same SCF energy numerically (the route the molecular
                # optimizers take) instead of walking a wrong surface.
                gradient = self._fd_gradient(mol, method)
            elif method == "RKS":
                grad_opts = _gradient_options_from_scf(method_opts)
                gradient = compute_gradient_rks(
                    mol, basis_obj, result, self._rks_options.grid,
                    options=grad_opts,
                )
            elif method == "UKS":
                grad_opts = _gradient_options_from_scf(method_opts)
                gradient = compute_gradient_uks(
                    mol, basis_obj, result, self._uks_options.grid,
                    options=grad_opts,
                )
            else:
                raise AssertionError(f"unreachable: method = {method}")
            # Fold in the dispersion gradient (already computed above
            # when disp_params is set and forces were requested).
            if disp_params is not None:
                gradient = gradient + disp.gradient
            self.results["forces"] = -gradient * (Hartree / Bohr)

        # ---- dipole moment (cheap; e.Å) -------------------------------
        # Compute on-demand if requested in `properties`. ASE's
        # ``atoms.get_dipole_moment()`` triggers this with
        # ``properties=("dipole",)``.
        if "dipole" in properties:
            # ECP systems: the nuclear term uses Z - n_core from the exact
            # Hamiltonian that ran, not bare Z (#642). CPCM attaches an
            # automatic sidecar to an internal options copy, so its verified
            # result-carried z_eff takes precedence over stale caller options.
            dip = dipole_moment(
                result,
                basis_obj,
                mol,
                nuclear_charges=_dipole_nuclear_charges(
                    mol, result, method_opts, sol_result
                ),
            )
            # ``DipoleMoment.total`` is the *magnitude* (a scalar);
            # ``DipoleMoment.x/y/z`` are the cartesian components.
            # ASE expects a 3-vector in eÅ.
            mu_au = np.array([dip.x, dip.y, dip.z], dtype=float)
            self.results["dipole"] = mu_au * Bohr     # e.a₀ -> e.Å

        # ---- Hessian (3N x 3N, eV/Å^2) ---------------------------------
        # Analytic for RHF / UHF / RKS where engineering ships it
        # (Phase 17b-3 / 17c); FD fallback otherwise. Requesting the
        # Hessian via ``atoms.get_property("hessian")`` re-runs the SCF
        # at displaced geometries (FD path) -- costly. Always opt-in.
        if "hessian" in properties:
            self.results["hessian"] = self._compute_hessian(
                method, mol, basis_obj, result,
            )

        # ---- Static dipole polarizability (RHF only; Å^3) --------------
        # Phase 17b-1. Returns a 3x3 tensor in atomic units (a₀^3); ASE
        # convention is Å^3.
        if "polarizability" in properties:
            if method != "RHF":
                raise PropertyNotImplementedError(
                    "vibe-qc polarizability via CPHF is RHF-only as of v0.5; "
                    "UHF / KS variants on the roadmap (17b-2)."
                )
            alpha_au = dipole_polarizability_rhf(result, basis_obj, mol)
            self.results["polarizability"] = (
                np.asarray(alpha_au) * (Bohr ** 3)
            )

        # ---- ASE alias --------------------------------------------------
        # ASE expects ``free_energy`` to track ``energy`` for codes
        # without finite-temperature smearing. Same value, different
        # name, prevents downstream code that calls
        # ``atoms.get_potential_energy(force_consistent=True)`` from
        # erroring.
        self.results["free_energy"] = self.results["energy"]

        # Expose the underlying result for downstream inspection.
        self.results["scf_result"] = result
        if method == "RHF":
            self.results["rhf_result"] = result

    # -------------------------------------------------------------------
    #  Hessian dispatch
    # -------------------------------------------------------------------

    def _compute_hessian(
        self,
        method: str,
        mol: "Molecule",
        basis_obj: "BasisSet",
        scf_result: Any,
    ) -> "np.ndarray":
        """Return the 3Nx3N Hessian in eV/Å^2 (ASE convention).

        Routes to the analytic kernel where one is shipped (RHF / UHF /
        RKS as of v0.5), otherwise falls back to finite-difference on the
        analytic gradient. The conversion factor is::

            1 Hartree/bohr^2 -> Hartree.(Å->bohr)^2.(eV->Hartree)⁻¹.(eV/Å^2)
                            = Hartree / (Bohr.Å).(Å/Hartree).(eV/Å^2)
                            = Hartree / Bohr^2 . (eV/Hartree.Å^2/Å^2)
                            = (Hartree / Bohr^2) . (eV/Hartree) . (1/Å^2.Bohr^2)
                            = Hartree / Bohr^2 x Hartree -> eV  ÷  Å^2/Bohr^2
                            = (Hartree.eV/Hartree).(Bohr^2/Å^2) ÷ Bohr^2
                            = eV / Å^2 x (Bohr/Å)^2.(Å/Bohr)^2
                            = eV / Å^2

        i.e. the Hessian in Hartree/bohr^2 multiplied by ``Hartree`` and
        divided by ``Bohr^2`` lands in eV/Å^2. See the unit test
        ``test_ase_hessian_units`` for an explicit numerical check.
        """
        ha_per_bohr2_to_eV_per_A2 = Hartree / (Bohr ** 2)

        if self._solvent is not None:
            from .solvation import resolve_solvent

            solvent_model = resolve_solvent(self._solvent)
        else:
            solvent_model = None
        if solvent_model is not None and not solvent_model.is_gas_phase:
            ecp_label = (
                "ECP+CPCM "
                if molecular_result_has_ecp_operator(scf_result)
                else "CPCM "
            )
            raise NotImplementedError(
                f"VibeQC: {ecp_label}Hessians are not implemented. The "
                "available analytic and finite-difference Hessian routes "
                "differentiate a gas-phase energy, not the self-consistent "
                "in-solvent total."
            )

        # The analytic-Hessian kernels need the basis name as a string
        # so they can rebuild the BasisSet at FD-displaced geometries
        # (the cphf displaced-Fock pieces). Pass it through explicitly.
        basis_name = self.parameters["basis"]
        # The analytic kernels take no SCF options and build their displaced
        # Fock pieces with bare-Z nuclear attraction, so on an ECP system
        # they would return an all-electron Hessian of an ECP SCF result.
        # Route ECP systems to the FD-on-analytic-gradient path, which
        # mirrors the options and moves the centres with each displaced
        # atom (#576, #643).
        if method in ("RHF", "UHF", "RKS") and options_carry_ecp(
            self._scf_options_for(method)
        ):
            scf_opts = self._scf_options_for(method)
            hess = compute_hessian_fd(
                mol,
                basis_name,
                method=method,
                scf_options=scf_opts,
                grid_options=scf_opts.grid if method == "RKS" else None,
                gradient_options=_gradient_options_from_scf(scf_opts),
            )
        elif method == "RHF":
            hess = compute_hessian_rhf_analytic(
                mol, basis_obj, scf_result, basis_name=basis_name,
            )
        elif method == "UHF":
            hess = compute_hessian_uhf_analytic(
                mol, basis_obj, scf_result, basis_name=basis_name,
            )
        elif method == "RKS":
            self._rks_options.functional = self.parameters["functional"]
            hess = compute_hessian_rks_analytic(
                mol, basis_obj, scf_result,
                basis_name=basis_name,
                functional=self.parameters["functional"],
                grid_options=self._rks_options.grid,
            )
        else:
            # UKS or anything else without an analytic kernel: use the
            # FD on analytic gradients path. compute_hessian_fd is
            # method-agnostic -- it dispatches on the ``method`` string.
            scf_opts = self._scf_options_for(method)
            hess = compute_hessian_fd(
                mol,
                self.parameters["basis"],
                method=method,
                scf_options=scf_opts,
                grid_options=(
                    self._uks_options.grid if method == "UKS" else None
                ),
                gradient_options=_gradient_options_from_scf(scf_opts),
            )
        return np.asarray(hess.hessian) * ha_per_bohr2_to_eV_per_A2

    def _sync_ecp_centres(self, method: str, opts: Any, mol: Molecule,
                          system_changes: Any) -> None:
        """Move the ECP centres on ``opts`` onto the atoms of ``mol`` (#643).

        Both ECP routes store the centres as absolute coordinates on the SCF
        options, and the same options object serves every geometry this
        calculator sees. The centre->atom mapping is fixed at the first
        geometry where the centres coincide with the atoms (the geometry
        they were built for; a set that sits on no atom is refused with a
        ``ValueError``, the same contract as ``compute_hessian_fd``), and
        every later call repositions by atom index -- permanently, so a
        caller that reuses the options afterwards (the runner's final single
        point and Hessian) finds them at the last geometry evaluated. A
        change of atomic numbers between calls resets the mapping.
        """
        if system_changes and "numbers" in system_changes:
            self._ecp_atom_indices.pop(method, None)
        if not options_carry_ecp(opts):
            return
        indices = self._ecp_atom_indices.get(method)
        if indices is None:
            indices = ecp_centre_atom_indices(opts, mol)
            self._ecp_atom_indices[method] = indices
            return
        reposition_ecp_centres(opts, mol, indices)

    def _scf_options_for(self, method: str) -> Any:
        """Return the right *Options object for the given method, used
        when dispatching to ``compute_hessian_fd``."""
        return {
            "RHF": self._rhf_options,
            "UHF": self._uhf_options,
            "RKS": self._rks_options,
            "UKS": self._uks_options,
        }[method]


__all__ = ["VibeQC", "VibeQCSCFConvergenceError"]
