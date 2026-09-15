"""Unified molecular semiempirical runner dispatch.

This module owns the high-level orchestration for the semiempirical methods
that are also reachable through :func:`vibeqc.runner.run_job`. Method-specific
physics remains in the neighbouring method modules and native C++ bindings.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from vibeqc._vibeqc_core import Molecule
from vibeqc.semiempirical.routes import (
    BOUNDARY_MOLECULE,
    BOUNDARY_SECCM_DIRECT_TORUS,
    MOLECULAR_SEMIEMPIRICAL_METHODS,
    SEMIEMPIRICAL_METHOD_ALIASES,
    SEMIEMPIRICAL_METHODS,
    SemiempiricalRoutePlan,
    normalise_semiempirical_method,
)


_SECCM_TRANSLATIONS_ERROR = (
    "method='seccm' needs ccm_options carrying the supercell lattice: "
    "run_job(mol, method='seccm', ccm_options=CCMOptions("
    "translations=[[a,0,0], ...], madelung=True)). 1 vector -> CCM1D "
    "(polymer), 2 -> CCM2D (surface), 3 -> CCM3D (bulk)."
)

# SCC-DFTB is a charge-mixing fixed-point problem. The direct SCCDFTBModel
# keeps its historical 100-iteration default; the high-level run_job/optimizer
# route retains a wider safety budget for genuinely slow geometries while using
# the same convergence threshold and native bounded relaxation.
_RUNNER_SCC_DFTB_MAX_ITER = 500
# Finite-temperature (Mermin free-energy) retry ladder for restricted
# closed-shell SCC-DFTB.  At T=0, near-degenerate frontier pi/pi* orbitals in
# pi-conjugated N/O systems (furan, pyridine, cytosine, ...) make the Mulliken
# charge response a non-smooth fixed point that no mixer can stabilise; the
# standard DFTB+/tblite/xtb remedy is Fermi-Dirac occupation smearing.
#
# Verified 2026-08-11 against all six pi-conjugated Thiel systems
# (handovers/HANDOVER_SCC_DFTB_CONVERGENCE.md): electronic_temperature=0.001 Ha
# (~316 K), no DIIS, charge_mixing=0.05 converges every one of them, while the
# historical 0.1-0.2 mixing re-enters the non-smooth regime (e.g. HCOOH at
# T=0.0012 Ha stalls past the 500-iteration budget at mix=0.2 but converges in
# 332 iterations at mix=0.05).  The converged fixed point is independent of the
# mixing fraction, so zero-T-passing systems keep their exact energies: the
# ladder is only reached after a zero-T failure.
#
# Extended 2026-08-21 for the two systems of issue #78 whose geometries are
# sound and which nevertheless failed closed on every rung.  Both needed more
# ladder, not a different remedy, and each converges to a physical energy:
#
#   * s22-formic-acid-dimer converges on the T=0.0012 rung but takes 573
#     iterations, i.e. it was failing purely on the 500-iteration retry budget.
#     E_free = -16.43989628 Ha.
#   * adenine does not converge at 0.001 / 0.0012 / 0.0015 even at 1500
#     iterations (it stalls at -18.7886 / -18.9226 / -19.1639 Ha) and needs a
#     warmer rung: at T=0.002 Ha (~632 K) it converges in 232 iterations to
#     E_free = -19.63356440 Ha.
#
# A rung is only entered after the previous one failed, so widening the budget
# and adding the 0.002 rung cannot change any system that already converges.
_RUNNER_SCC_DFTB_RETRY_SETTINGS = (
    {"electronic_temperature": 0.001, "charge_mixing": 0.05, "use_diis": False},
    {"electronic_temperature": 0.0012, "charge_mixing": 0.05, "use_diis": False},
    {"electronic_temperature": 0.0015, "charge_mixing": 0.05, "use_diis": False},
    {"electronic_temperature": 0.002, "charge_mixing": 0.05, "use_diis": False},
)
# Per-rung iteration budget.  Wider than the zero-T budget because a smeared,
# low-mixing charge-response fixed point is approached slowly by construction:
# the rung that converges the formic-acid dimer needs 573 iterations, so the
# historical 500 turned a converging rung into a failure.
_RUNNER_SCC_DFTB_RETRY_MAX_ITER = 1500


def _fermi_dirac_entropy(
    mo_energies,
    n_occ: int,
    temperature: float,
) -> float:
    """Closed-shell Fermi-Dirac entropy, mirroring the native occupations.

    Reconstructs the per-orbital entropy of the converged finite-temperature
    SCC-DFTB solution from its MO energies, matching
    ``compute_closed_shell_kpoint_occupations``
    (``cpp/src/semiempirical/kpoints_occupations.cpp``) exactly: occupations
    ``f_i = 2 / (1 + exp((eps_i - mu)/T))`` with the chemical potential fixed
    by ``sum_i f_i = 2 * n_occ`` (bracketing + bisection), and
    ``S = -2 * sum_i [x ln x + (1-x) ln(1-x)]`` with ``x = f_i / 2`` clamped to
    ``[1e-300, 1 - 1e-15]``.  Returns ``S`` so that the Mermin term is
    ``-T * S`` and ``E_internal = E_free + T * S``.

    This is reporting infrastructure only: it never feeds a value back into
    the SCC loop, and at ``T = 0`` it is not called.
    """
    eps = np.asarray(mo_energies, dtype=float)
    if eps.size == 0:
        return 0.0
    capacity = 2.0 * eps.size
    target = 2.0 * int(n_occ)
    if target <= 1.0e-12 or target >= capacity - 1.0e-12:
        return 0.0

    def count(mu: float) -> float:
        arg = np.clip((eps - mu) / temperature, -50.0, 50.0)
        return float(np.sum(2.0 / (1.0 + np.exp(arg))))

    lower = float(eps.min()) - 10.0 * temperature - 1.0
    upper = float(eps.max()) + 10.0 * temperature + 1.0
    expansion = max(100.0, 100.0 * temperature, 1.0)
    bracketed = False
    for _ in range(20):
        if count(lower) <= target <= count(upper):
            bracketed = True
            break
        lower -= expansion
        upper += expansion
        expansion *= 2.0
    if not bracketed:
        return 0.0
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if count(middle) > target:
            upper = middle
        else:
            lower = middle
        if upper - lower < 1.0e-14:
            break
    mu = 0.5 * (lower + upper)
    arg = np.clip((eps - mu) / temperature, -50.0, 50.0)
    occ = 2.0 / (1.0 + np.exp(arg))
    fraction = np.clip(occ / 2.0, 1.0e-300, 1.0 - 1.0e-15)
    return float(np.sum(-2.0 * (fraction * np.log(fraction) + (1.0 - fraction) * np.log(1.0 - fraction))))


def _normalise_ccm_translations(translations: Any) -> list[list[float]]:
    if translations is None:
        raise ValueError(_SECCM_TRANSLATIONS_ERROR)
    try:
        vectors = [list(vector) for vector in translations]
    except TypeError as exc:
        raise ValueError(_SECCM_TRANSLATIONS_ERROR) from exc
    if len(vectors) not in (1, 2, 3):
        raise ValueError(
            "method='seccm' needs 1, 2, or 3 translation vectors in "
            "ccm_options.translations."
        )
    normalised: list[list[float]] = []
    for index, vector in enumerate(vectors, start=1):
        if len(vector) != 3:
            raise ValueError(
                "method='seccm' needs each translation vector to have three "
                f"numeric components; vector {index} has {len(vector)}."
            )
        try:
            normalised.append([float(component) for component in vector])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "method='seccm' needs each translation vector to have three "
                f"numeric components; vector {index} could not be converted."
            ) from exc
    return normalised


class SemiempiricalEnergyError(RuntimeError):
    """A semiempirical route produced a total energy that is not a result.

    Subclasses :class:`RuntimeError` so that callers already gating on the
    native layer's numerical failures keep working unchanged.
    """


def _reject_non_result_energy(
    energy,
    method_name: str,
    n_electrons: int | None = None,
) -> float:
    """Refuse a semiempirical total energy that no SCF could have produced.

    A semiempirical total energy is a sum of an electronic term and a
    core-core repulsion term (NDDO: PM6/PM7/OMx/MSINDO) or of band-structure
    and repulsive terms (DFTB0/SCC-DFTB/GFN2).  For a system carrying at
    least one electron, neither family can land on IEEE-754 +/-0.0, so an
    exactly-zero total energy is not a converged answer -- it is the
    signature of a route that returned before computing one.

    GitLab #160 (legacy BUG-024): PM6/norbornadiene on the ``compute-study`` host ran
    300 SCF iterations emitting ``E = 0.0000000000`` at every iteration and
    returned it with no exception, so a dispatch lane gating on ``converged``
    counted the row as data.  The molecular trigger for that specific case
    was separately repaired by the bounded Mermin homotopy ladder in
    :func:`_run_pm6`; this guard is the class-level tripwire the issue asks
    for, and it fires for any route and any host.

    The test is deliberately **bit-exact** rather than a tolerance band.  A
    tolerance would refuse legitimately small total energies, while an exact
    zero can only arise when nothing was computed -- an uninitialised
    accumulator, a skipped assembly, or a silently truncated native call.
    ``-0.0`` compares equal to ``0.0`` and is refused with it.

    Non-finite energies (NaN / +-inf) are refused by the same gate: they are
    the other value a total energy can take only when the calculation failed.

    ``n_electrons`` is the one documented exemption.  A system with **no**
    electrons has no electronic term at all, and a single bare centre has no
    core-core pair either, so exactly 0.0 is that system's correct total (a
    bare proton at ``charge=+1`` is the reachable case).  Callers that know
    the electron count pass it so the degenerate input is not refused with a
    message that would be false about it.  ``None`` means "not known here",
    and the guard then applies.  A multi-centre zero-electron cation would
    still have a non-zero core repulsion, so this exemption is deliberately
    keyed on the electron count alone rather than trying to model that case;
    it opens a false-negative window only on inputs that carry no electrons.
    """
    try:
        value = float(energy)
    except (TypeError, ValueError) as exc:
        raise SemiempiricalEnergyError(
            f"semiempirical method {method_name!r} returned a total energy "
            f"that is not a number: {energy!r}."
        ) from exc
    label = method_name or "<unknown>"
    if not math.isfinite(value):
        raise SemiempiricalEnergyError(
            f"semiempirical method {label!r} returned a non-finite total "
            f"energy ({value}). This is a failed calculation, not a result; "
            "it is refused instead of being reported. See GitLab #160."
        )
    if value == 0.0:
        if n_electrons is not None and int(n_electrons) == 0:
            return value
        raise SemiempiricalEnergyError(
            f"semiempirical method {label!r} returned a total energy of "
            "exactly 0.0 Ha. No semiempirical SCF on a system with electrons "
            "can produce that value, so the calculation did not run to a "
            "result and is refused instead of being reported silently. "
            "See GitLab #160 (legacy BUG-024)."
        )
    return value


class SemiempiricalResult:
    """Small result adapter used by ``run_job`` for semiempirical methods."""

    def __init__(
        self,
        energy,
        gradient=None,
        method_name="",
        converged=True,
        n_iter=1,
        binding_energy=None,
        e_solv=None,
        e_gas=None,
        solvent_variant=None,
        mulliken_charges=None,
        electronic_temperature=0.0,
        density=None,
        parameter_provenance=None,
        e_internal=None,
        entropy=None,
        parameter_identity=None,
        parameter_sha256=None,
        n_electrons=None,
    ):
        # Tripwire (#160): a total energy of exactly 0.0 Ha, or a non-finite
        # one, is refused here rather than reported.  Every semiempirical
        # route in this module funnels its answer through this constructor,
        # so the guard covers molecular PM6/PM7/OMx/MSINDO/DFTB0/SCC-DFTB/
        # GFN2 and the MSINDO SECCM adapter from one place.  Routes that know
        # the electron count pass it so the one legitimate exact zero -- a
        # system with no electrons -- is not refused.
        _reject_non_result_energy(energy, method_name, n_electrons)
        self.energy = energy
        self._gradient = gradient
        self.method = method_name
        self.converged = converged
        self.n_iter = n_iter
        # MSINDO reports an atomization/binding energy (E - S atomic reference)
        # directly; None for engines that do not.
        self.binding_energy = binding_energy
        # Implicit-solvation diagnostics for MSINDO COSMO.
        self.e_solv = e_solv
        self.e_gas = e_gas
        self.solvent_variant = solvent_variant
        self.electronic_temperature = float(electronic_temperature)
        self.density = density
        # Finite-temperature (Mermin) reporting for SCC-DFTB retries and GFN2
        # finite-temperature ladder rungs.
        # ``energy`` is the converged free energy E_free = E_internal - T*S
        # (the C++ assembly already subtracts the -T*S term at T > 0);
        # ``e_internal`` reconstructs the internal energy E_free + T*S and
        # ``entropy`` carries S.  Both are None for zero-T results and for
        # methods that do not run a finite-temperature retry.
        self.e_internal = None if e_internal is None else float(e_internal)
        self.entropy = None if entropy is None else float(entropy)
        # Method-native net atomic charges (positive = electron-deficient).
        # This deliberately carries only the producer's validated population
        # result; the output layer marks every unavailable analysis explicitly
        # instead of pretending a Gaussian AO basis exists.
        self.mulliken_charges = (
            None
            if mulliken_charges is None
            else tuple(float(value) for value in mulliken_charges)
        )
        # The current semiempirical backends expose the final iteration count
        # but not per-iteration energies/residuals.  Keep the historical list
        # shape for callers; the shared SCF renderer omits an empty table and
        # still reports ``n_iter`` truthfully in its convergence verdict.
        self.scf_trace = []
        # Exact identity of the immutable native parameter snapshot that
        # produced this result.  These values must come from the native result,
        # never from a caller-owned parameter builder after execution.
        self.parameter_identity = (
            None if parameter_identity is None else str(parameter_identity)
        )
        self.parameter_sha256 = (
            None if parameter_sha256 is None else str(parameter_sha256)
        )
        # Flat scalar dict of parameter provenance for auditability (BUG 69).
        # Method backends populate source-lineage fields for externally-sourced
        # parameters and the generic keys below for content-identified native
        # snapshots.
        # The main runner feeds it into ``OutputWriter.update_run_fields`` so
        # the ``[run]`` section of ``{output}.system`` carries the parameter
        # identity, SHA-256, source URL, and runtime settings that pin the
        # result.
        provenance = (
            dict(parameter_provenance)
            if parameter_provenance is not None
            else None
        )
        if self.parameter_identity is not None:
            if provenance is None:
                provenance = {}
            provenance["parameter_identity"] = self.parameter_identity
        if self.parameter_sha256 is not None:
            if provenance is None:
                provenance = {}
            provenance["parameter_sha256"] = self.parameter_sha256
        self.parameter_provenance: dict[str, Any] | None = provenance

    def gradient(self):
        if callable(self._gradient):
            self._gradient = self._gradient()
        return self._gradient


def run_semiempirical(
    method: str,
    molecule: Molecule,
    *,
    nddo: bool = False,
    solvent: Any = None,
    ccm_options: Any = None,
    initial_charges: Any = None,
) -> SemiempiricalResult:
    """Run a semiempirical method and return a ``run_job``-compatible result.

    ``nddo`` selects MSINDO's NDDO mode and is rejected for other methods.
    ``solvent`` currently applies only to ``method="msindo"`` and routes to
    MSINDO COSMO. ``method="seccm"`` delegates to :func:`run_seccm` and
    requires ``ccm_options``. The legacy ``method="ccm"`` spelling remains an
    alias. ``initial_charges`` optionally continues a molecular SCC-DFTB
    charge solution from a nearby geometry.
    """
    method_key = normalise_semiempirical_method(method)
    if method_key not in SEMIEMPIRICAL_METHODS:
        raise ValueError(f"Unhandled semiempirical method: {method_key!r}")
    if initial_charges is not None and method_key != "scc_dftb":
        raise ValueError("initial_charges is supported only for SCC-DFTB")
    plan = SemiempiricalRoutePlan.from_request(
        method,
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
        nddo=nddo,
        solvent=solvent,
        ccm_options=ccm_options,
    )
    return _run_semiempirical_plan(
        plan,
        molecule,
        solvent=solvent,
        ccm_options=ccm_options,
        initial_charges=initial_charges,
    )


def _run_semiempirical_plan(
    plan: SemiempiricalRoutePlan,
    molecule: Molecule,
    *,
    solvent: Any = None,
    ccm_options: Any = None,
    initial_charges: Any = None,
) -> SemiempiricalResult:
    """Execute a validated molecular or current MSINDO SECCM route plan."""
    if plan.boundary == BOUNDARY_SECCM_DIRECT_TORUS:
        if plan.method_key != "msindo":
            raise NotImplementedError(
                "the generic semiempirical executor can run only the legacy "
                "MSINDO SECCM route because it has no SECCMTopology argument; "
                f"method={plan.method_key!r} must use its explicit run_*_seccm "
                "adapter"
            )
        result = run_ccm(molecule, ccm_options)
        translations = getattr(ccm_options, "translations", None)
        if translations is not None and len(translations) in (1, 2, 3):
            result.route_plan = plan.with_seccm_runtime(
                periodic_dimension=len(translations),
                electrostatics_family=(
                    "madelung"
                    if bool(getattr(ccm_options, "madelung", False))
                    else "none"
                ),
            )
        else:
            # Model-like test doubles can bypass run_ccm's option validation.
            result.route_plan = plan
        return result
    if plan.boundary != BOUNDARY_MOLECULE:
        raise RuntimeError(
            f"molecular semiempirical adapter received boundary={plan.boundary!r}"
        )
    if plan.variant == "cosmo":
        result = _run_msindo_cosmo(plan, molecule, solvent)
    elif initial_charges is None:
        result = _run_molecular_semiempirical(plan, molecule)
    else:
        result = _run_molecular_semiempirical(
            plan, molecule, initial_charges=initial_charges
        )
    # The concrete result owns the runtime temperature while the immutable
    # plan owns all other route semantics. Keep them together so every
    # citation emitter consumes the same complete route record.
    result.route_plan = replace(
        plan,
        electronic_temperature=float(
            getattr(result, "electronic_temperature", 0.0) or 0.0
        ),
    )
    return result


def _run_msindo_cosmo(
    plan: SemiempiricalRoutePlan,
    molecule: Molecule,
    solvent: Any,
) -> SemiempiricalResult:
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as _A2B
    from vibeqc.semiempirical.methods.msindo_cosmo import msindo_cosmo
    from vibeqc.solvation.driver import resolve_solvent

    sm = resolve_solvent(solvent)
    if sm is None or sm.is_gas_phase:
        return _run_molecular_semiempirical(plan, molecule)
    Z = [at.Z for at in molecule.atoms]
    coords_ang = [[c / _A2B for c in at.xyz] for at in molecule.atoms]
    rc = msindo_cosmo(
        Z,
        coords_ang,
        epsilon=sm.epsilon,
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
    )
    return SemiempiricalResult(
        float(rc.total_energy),
        None,
        "msindo",
        converged=bool(rc.converged),
        n_iter=int(rc.n_iter),
        binding_energy=float(rc.binding_energy),
        e_solv=float(rc.e_solv),
        e_gas=float(rc.e_gas),
        solvent_variant="cosmo",
        n_electrons=molecule.n_electrons(),
    )


def _gfn2_parameter_provenance(model) -> dict[str, Any] | None:
    """Build a flat scalar provenance dict for a GFN2Model result.

    The content identity and parameter values come from the model's frozen
    pre-execution snapshot, cross-checked against the exact identity returned
    by the native result. Cache-file and D4-dataset hashes remain supplementary
    artifact lineage captured before native execution. Fields are intentionally
    flat scalars — the manifest's
    ``extra_run_fields`` contract (``ManifestUpdater.update_run_fields``)
    merges string-keyed scalar values into ``[run]``, and complex nested
    structures would break TOAD's fixed-shape reader. Returns None for
    model-like stand-ins that have not executed an identified native result.
    """
    snapshot = getattr(model, "_last_parameter_snapshot", None)
    if not isinstance(snapshot, dict):
        # Model-like stand-ins and models that have not executed yet produce no
        # provenance rather than reconstructing an identity from mutable state.
        return None
    native_result = getattr(model, "_last_result", None)
    result_identity = getattr(native_result, "parameter_identity", None)
    result_sha256 = getattr(native_result, "parameter_sha256", None)
    if result_identity is None or result_sha256 is None:
        return None
    result_identity = str(result_identity)
    result_sha256 = str(result_sha256)
    if (
        result_identity != str(snapshot["parameter_identity"])
        or result_sha256 != str(snapshot["parameter_sha256"])
    ):
        raise RuntimeError(
            "GFN2-xTB native result parameter identity disagrees with "
            "the immutable execution snapshot"
        )
    prov: dict[str, Any] = {
        "gfn2_parameter_identity": result_identity,
        "gfn2_parameter_sha256": result_sha256,
        "gfn2_param_version": str(snapshot["version"]),
        "gfn2_param_origin": str(snapshot["origin"]),
        "gfn2_param_license": str(snapshot["license"]),
        "gfn2_param_doi": str(snapshot["doi"]),
        "gfn2_n_elements": int(snapshot["n_elements"]),
        "gfn2_cache_path": str(snapshot["cache_path"]),
        "gfn2_cache_sha256": str(snapshot["cache_sha256"]),
        "gfn2_cache_source_sha256": str(snapshot["cache_source_sha256"]),
        "gfn2_cache_source_url": str(snapshot["cache_source_url"]),
        "gfn2_d4_functional": str(snapshot["d4_functional"]),
        "gfn2_d4_s6": float(snapshot["d4_s6"]),
        "gfn2_d4_s8": float(snapshot["d4_s8"]),
        "gfn2_d4_s9": float(snapshot["d4_s9"]),
        "gfn2_d4_a1": float(snapshot["d4_a1"]),
        "gfn2_d4_a2": float(snapshot["d4_a2"]),
        "gfn2_d4_doi": str(snapshot["d4_doi"]),
        "gfn2_d4_refdata_path": str(snapshot["d4_refdata_path"]),
        "gfn2_d4_refdata_sha256": str(snapshot["d4_refdata_sha256"]),
        "gfn2_d4_refdata_source_sha256": str(
            snapshot["d4_refdata_source_sha256"]
        ),
        "gfn2_max_iter": int(snapshot["max_iter"]),
        "gfn2_conv_tol_charge": float(snapshot["conv_tol_charge"]),
        "gfn2_electronic_temperature": float(
            snapshot["electronic_temperature"]
        ),
        "gfn2_electronic_temperature_explicit": bool(
            snapshot["electronic_temperature_explicit"]
        ),
        # The two fields above are the request.  The default-path ladder can
        # converge on a finite-temperature rung, so record what ran (#247).
        "gfn2_executed_electronic_temperature": float(
            native_result.smearing_temperature
        ),
        "gfn2_charge_mixing": float(snapshot["charge_mixing"]),
        "gfn2_scc_mixer": str(snapshot["scc_mixer"]),
        "gfn2_mixer_memory": int(snapshot["mixer_memory"]),
        "gfn2_mixer_damping": float(snapshot["mixer_damping"]),
        "gfn2_auto_stabilize": bool(snapshot["auto_stabilize"]),
        "gfn2_aes_faithful": bool(snapshot["aes_faithful"]),
        "gfn2_aes_damping": float(snapshot["aes_damping"]),
    }
    return prov


def _native_parameter_provenance(native_result) -> dict[str, str] | None:
    """Project the exact immutable parameter snapshot on a native result."""
    if native_result is None:
        return None
    identity = getattr(native_result, "parameter_identity", None)
    sha256 = getattr(native_result, "parameter_sha256", None)
    if identity is None or sha256 is None:
        return None
    return {
        "parameter_identity": str(identity),
        "parameter_sha256": str(sha256),
    }


def _dftb_parameter_provenance(model) -> dict[str, str] | None:
    """Project the exact parameter snapshot recorded by a DFTB model."""
    return _native_parameter_provenance(getattr(model, "_last_result", None))


def _run_molecular_semiempirical(
    plan: SemiempiricalRoutePlan,
    molecule: Molecule,
    *,
    initial_charges: Any = None,
) -> SemiempiricalResult:
    """Run a validated molecular semiempirical calculation.

    Method, variant, and spin selection come only from ``plan``. The public
    dispatcher constructs the plan before any method parameters or models load.
    """
    method = plan.method_key

    if method in ("dftb0", "scc_dftb"):
        from vibeqc.semiempirical import (
            DFTB0Model,
            SCCDFTBModel,
            UDFTB0Model,
            USCCDFTBModel,
        )
        from vibeqc.semiempirical.parameters import default_parameters

        params = default_parameters()
        if method == "dftb0":
            model_type = UDFTB0Model if plan.spin == "unrestricted" else DFTB0Model
        else:
            model_type = (
                USCCDFTBModel if plan.spin == "unrestricted" else SCCDFTBModel
            )
        model_kwargs = (
            {
                "max_iter": _RUNNER_SCC_DFTB_MAX_ITER,
                "initial_charges": initial_charges,
            }
            if method == "scc_dftb"
            else {}
        )
        model = model_type(molecule, params=params, **model_kwargs)
        energy = model.energy()
        total_iterations = int(getattr(model, "n_iter", 1))
        electronic_temperature = 0.0
        if (
            method == "scc_dftb"
            and plan.spin == "closed_shell"
            and not model.converged
        ):
            # Bounded Mermin free-energy retry ladder.  A zero-T failure on
            # pi-conjugated N/O systems is a non-smooth charge-response
            # fixed point (see handovers/HANDOVER_SCC_DFTB_CONVERGENCE.md),
            # not a mixer defect: Fermi-Dirac smearing at low mixing is the
            # standard remedy.  The ladder is only reached after a zero-T
            # failure, so systems whose integer-occupation SCC converges keep
            # their exact zero-T energies.  Every rung starts from zero
            # charges: the unconverged zero-T charge guess is a bad seed.
            for retry_settings in _RUNNER_SCC_DFTB_RETRY_SETTINGS:
                retry_temperature = retry_settings["electronic_temperature"]
                model = model_type(
                    molecule,
                    params=params,
                    max_iter=_RUNNER_SCC_DFTB_RETRY_MAX_ITER,
                    charge_mixing=retry_settings["charge_mixing"],
                    electronic_temperature=retry_temperature,
                    use_diis=retry_settings["use_diis"],
                    initial_charges=None,
                )
                energy = model.energy()
                total_iterations += int(getattr(model, "n_iter", 1))
                electronic_temperature = retry_temperature
                if model.converged:
                    break
        gradient = model.gradient if hasattr(model, "gradient") else None
        native_result = getattr(model, "_last_result", None)
        native_charges = getattr(native_result, "charges", None)
        parameter_provenance = _dftb_parameter_provenance(model)
        # Mermin convention at T > 0: ``energy`` is the free energy
        # E_free = E_internal - T*S.  Reconstruct E_internal and S from the
        # converged MO energies so the .out can label both clearly.
        e_internal = None
        entropy = None
        if electronic_temperature > 0.0 and native_result is not None:
            mo_energies = getattr(native_result, "mo_energies", None)
            n_occ = int(getattr(native_result, "n_occ", 0))
            if mo_energies is not None:
                entropy = _fermi_dirac_entropy(
                    mo_energies, n_occ, electronic_temperature
                )
                e_internal = float(energy) + electronic_temperature * entropy
        return SemiempiricalResult(
            energy,
            gradient,
            method,
            converged=bool(getattr(model, "converged", True)),
            n_iter=total_iterations,
            mulliken_charges=native_charges,
            electronic_temperature=electronic_temperature,
            e_internal=e_internal,
            entropy=entropy,
            parameter_provenance=parameter_provenance,
            parameter_identity=(
                None
                if parameter_provenance is None
                else parameter_provenance["parameter_identity"]
            ),
            parameter_sha256=(
                None
                if parameter_provenance is None
                else parameter_provenance["parameter_sha256"]
            ),
            n_electrons=molecule.n_electrons(),
        )

    if method == "gfn2_xtb":
        from vibeqc.semiempirical.methods.gfn2 import GFN2Model
        from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

        params = load_gfn2_params()
        model = GFN2Model(molecule, params)
        energy = model.energy()
        native_result = getattr(model, "_last_result", None)
        provenance = _gfn2_parameter_provenance(model)
        # The auto-stabilisation ladder can accept a finite-temperature rung
        # when no temperature was requested (#3), so the executed ensemble
        # comes from the native result, never from the requested options
        # (#247).  At T > 0 report the Mermin free energy A = E - T*S, as the
        # SCC-DFTB retry does, and keep E and S for the .out.
        electronic_temperature = float(
            getattr(native_result, "smearing_temperature", 0.0) or 0.0
        )
        e_internal = None
        entropy = None
        if electronic_temperature > 0.0:
            e_internal = float(energy)
            entropy = float(native_result.entropy)
            energy = e_internal - electronic_temperature * entropy
        return SemiempiricalResult(
            energy,
            model.gradient,
            method,
            converged=model.converged,
            n_iter=model.n_iter,
            mulliken_charges=getattr(native_result, "charges", None),
            electronic_temperature=electronic_temperature,
            e_internal=e_internal,
            entropy=entropy,
            parameter_provenance=provenance,
            parameter_identity=getattr(
                native_result, "parameter_identity", None
            ),
            parameter_sha256=getattr(native_result, "parameter_sha256", None),
            n_electrons=molecule.n_electrons(),
        )

    if method == "om1":
        import warnings as _warnings

        from vibeqc.semiempirical import NDDOExperimentalWarning

        _warnings.warn(
            "method='om1': OM1's analytically-evaluated core-valence ECP "
            "(Kolb & Thiel 1993) is not implemented; bonds to heavy atoms "
            "come out ~0.3 A short and close non-bonded contacts can "
            "variationally collapse. Prefer om2/om3 for molecular "
            "prescreening. "
            "See docs/user_guide/semiempirical.md.",
            category=NDDOExperimentalWarning,
            stacklevel=2,
        )

    from vibeqc._vibeqc_core.semiempirical.nddo import (
        run_omx_v2,
        run_pm6,
    )

    if method == "pm6":
        return _run_pm6(plan, molecule, run_pm6)

    if method == "msindo":
        return _run_msindo(plan, molecule)

    if method in ("om1", "om2", "om3"):
        return _run_omx(plan, molecule, run_omx_v2)

    raise ValueError(f"Unhandled semiempirical method: {method!r}")


def _run_pm6(
    plan: SemiempiricalRoutePlan,
    molecule: Molecule,
    run_pm6,
) -> SemiempiricalResult:
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

    params = load_pm6_params_auto([at.Z for at in molecule.atoms])
    if plan.spin == "unrestricted":
        from vibeqc._vibeqc_core.semiempirical.nddo import (
            compute_upm6_gradient_fd_from_result,
            run_upm6,
        )

        result_obj = run_upm6(molecule, params, max_iter=200)
        n_iter = int(result_obj.n_iter)

        def gradient():
            return compute_upm6_gradient_fd_from_result(
                molecule, params, result_obj, max_iter=200
            )

    else:
        result_obj = run_pm6(molecule, params, max_iter=200)
        from vibeqc._vibeqc_core.semiempirical.nddo import (
            compute_pm6_gradient_fd_from_result,
            run_pm6_with_density,
            run_pm6_with_smearing,
        )

        n_iter = int(result_obj.n_iter)
        if not result_obj.converged:
            # A symmetric weakly coupled dimer or strained bridged bicycle
            # can leave hard Aufbau occupations cycling between
            # near-degenerate fragment orbitals.  Use a bounded Mermin
            # homotopy to select a smooth density, then cool back to the
            # ordinary zero-temperature PM6 fixed point.  The finite-T
            # energy is never exposed as a successful PM6 result.
            #
            # Try a temperature ladder: formamide-dimer responds to
            # T=0.005 Ha (v0.15.57), but more strongly coupled systems
            # like norbornadiene need T >= 0.01 Ha to escape the
            # occupation cycle (BUG-024).
            _PM6_RECOVERY_TEMPERATURES = (0.005, 0.01, 0.02)
            for _recovery_T in _PM6_RECOVERY_TEMPERATURES:
                try:
                    warm = run_pm6_with_smearing(
                        molecule,
                        params,
                        electronic_temperature=_recovery_T,
                        max_iter=100,
                    )
                except RuntimeError:
                    # NaN/Inf guard fired in the C++ layer — this
                    # temperature didn't help; try the next rung.
                    n_iter += 100
                    continue
                n_iter += int(warm.n_iter)
                if warm.converged:
                    cooled = run_pm6_with_density(
                        molecule, params, warm.density, max_iter=200
                    )
                    n_iter += int(cooled.n_iter)
                    if cooled.converged:
                        # Re-cool verification: the cooled solution
                        # must be a genuine zero-T fixed point (BUG-024
                        # on compute-study: the first cool can appear converged
                        # but land on a metastable occupation that
                        # collapses to a cycle on the next SCF).
                        recooled = run_pm6_with_density(
                            molecule, params, cooled.density, max_iter=50
                        )
                        n_iter += int(recooled.n_iter)
                        if recooled.converged:
                            result_obj = recooled
                            break

        def gradient():
            return compute_pm6_gradient_fd_from_result(
                molecule, params, result_obj, max_iter=200
            )

    parameter_provenance = _native_parameter_provenance(result_obj)
    return SemiempiricalResult(
        float(result_obj.energy),
        gradient,
        "pm6",
        converged=bool(result_obj.converged),
        n_iter=n_iter,
        density=getattr(result_obj, "density", None),
        parameter_provenance=parameter_provenance,
        parameter_identity=(
            None
            if parameter_provenance is None
            else parameter_provenance["parameter_identity"]
        ),
        parameter_sha256=(
            None
            if parameter_provenance is None
            else parameter_provenance["parameter_sha256"]
        ),
        n_electrons=molecule.n_electrons(),
    )


def _run_closed_shell_pm6_from_density(
    molecule: Molecule,
    initial_density,
) -> SemiempiricalResult:
    """Continue a closed-shell PM6 solve from a nearby geometry's density."""
    from vibeqc._vibeqc_core.semiempirical.nddo import (
        compute_pm6_gradient_fd_from_result,
        run_pm6_with_density,
    )
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params_auto

    params = load_pm6_params_auto([at.Z for at in molecule.atoms])
    result_obj = run_pm6_with_density(
        molecule, params, initial_density, max_iter=200
    )

    def gradient():
        return compute_pm6_gradient_fd_from_result(
            molecule, params, result_obj, max_iter=200
        )

    parameter_provenance = _native_parameter_provenance(result_obj)
    return SemiempiricalResult(
        float(result_obj.energy),
        gradient,
        "pm6",
        converged=bool(result_obj.converged),
        n_iter=int(result_obj.n_iter),
        density=result_obj.density,
        parameter_provenance=parameter_provenance,
        parameter_identity=(
            None
            if parameter_provenance is None
            else parameter_provenance["parameter_identity"]
        ),
        parameter_sha256=(
            None
            if parameter_provenance is None
            else parameter_provenance["parameter_sha256"]
        ),
        n_electrons=molecule.n_electrons(),
    )


def _run_msindo(
    plan: SemiempiricalRoutePlan,
    molecule: Molecule,
) -> SemiempiricalResult:
    from vibeqc.semiempirical.methods.msindo import _SUPPORTED, _SUPPORTED_NDDO
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as _A2B
    from vibeqc.semiempirical.methods.msindo import eff_core_charge as _cz

    Z = [at.Z for at in molecule.atoms]
    missing = sorted({z for z in Z if z not in _SUPPORTED})
    if missing:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(_SUPPORTED)}; got Z={missing}. "
            "Extend the parameter tables (datas.f) -- see docs/user_guide/msindo.md."
        )
    coords_ang = [[c / _A2B for c in at.xyz] for at in molecule.atoms]
    charge = int(molecule.charge)
    mult = int(molecule.multiplicity)
    nddo = plan.variant == "nddo"
    if nddo:
        missing_nddo = sorted({z for z in Z if z not in _SUPPORTED_NDDO})
        if missing_nddo:
            raise NotImplementedError(
                "MSINDO NDDO mode is parametrized for "
                f"{sorted(_SUPPORTED_NDDO)}; got Z={missing_nddo}."
            )
    nelec = sum(_cz(z) for z in Z) - charge
    open_shell = plan.spin == "unrestricted" or nelec % 2 != 0
    if nddo and open_shell:
        raise NotImplementedError(
            "MSINDO NDDO mode is closed-shell (RHF) only; open-shell NDDO "
            "(UHF) is not implemented. Use multiplicity=1."
        )

    from vibeqc._vibeqc_core.semiempirical.indo import (
        load_params_from_json,
        merge_nddo_overrides,
        run_msindo_full,
        run_msindo_uhf,
    )

    methods_dir = Path(__file__).resolve().parent / "methods"
    params = load_params_from_json((methods_dir / "msindo_params.json").read_text())
    if nddo:
        merge_nddo_overrides(
            params,
            (methods_dir / "msindo_params_nddo.json").read_text(),
        )

    if open_shell:
        result_obj = run_msindo_uhf(
            Z,
            coords_ang,
            params,
            max_iter=200,
            multiplicity=mult,
            nddo=nddo,
            charge=charge,
        )
    else:
        result_obj = run_msindo_full(
            Z,
            coords_ang,
            params,
            max_iter=200,
            nddo=nddo,
            charge=charge,
        )
        if not nddo and not result_obj.converged:
            # Keep light-element closed-shell jobs on the same recovery policy
            # as the public MSINDO API. Its independent Python DIIS path can
            # reach the unchanged hard-occupation stationary root when the
            # native trajectory exhausts its budget (archived s-tetrazine).
            from vibeqc.semiempirical.methods.msindo import run_msindo

            result_obj = run_msindo(
                Z,
                coords_ang,
                charge=charge,
                multiplicity=mult,
                max_iter=200,
                nddo=False,
            )
    gradient = None
    if not open_shell and getattr(result_obj, "converged", False):

        def gradient():
            from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
                msindo_gradient_analytic,
            )

            return msindo_gradient_analytic(
                Z,
                coords_ang,
                charge=charge,
                multiplicity=mult,
                nddo=nddo,
            )

    return SemiempiricalResult(
        float(result_obj.total_energy),
        gradient,
        "msindo",
        converged=bool(result_obj.converged),
        n_iter=int(result_obj.n_iter),
        binding_energy=float(getattr(result_obj, "binding_energy", 0.0)),
        n_electrons=molecule.n_electrons(),
    )


def _run_omx(
    plan: SemiempiricalRoutePlan,
    molecule: Molecule,
    run_omx_v2,
) -> SemiempiricalResult:
    from vibeqc.semiempirical.methods.omx_params import (
        load_om1_params,
        load_om2_params,
        load_om3_params,
    )

    method = plan.method_key
    loader = {
        "om1": load_om1_params,
        "om2": load_om2_params,
        "om3": load_om3_params,
    }[method]
    params = loader()

    if plan.spin == "unrestricted":
        from vibeqc._vibeqc_core.semiempirical.nddo import (
            compute_uomx_v2_gradient_fd_from_result,
            run_uomx_v2,
        )

        result_obj = run_uomx_v2(molecule, params, max_iter=200)

        def gradient():
            return compute_uomx_v2_gradient_fd_from_result(
                molecule, params, result_obj, max_iter=200
            )

    else:
        result_obj = run_omx_v2(molecule, params, max_iter=200)
        from vibeqc._vibeqc_core.semiempirical.nddo import (
            compute_omx_v2_gradient_fd_from_result,
        )

        def gradient():
            return compute_omx_v2_gradient_fd_from_result(
                molecule, params, result_obj, max_iter=200
            )

    parameter_provenance = _native_parameter_provenance(result_obj)
    return SemiempiricalResult(
        float(result_obj.energy),
        gradient,
        method,
        converged=bool(result_obj.converged),
        n_iter=int(result_obj.n_iter),
        parameter_provenance=parameter_provenance,
        parameter_identity=(
            None
            if parameter_provenance is None
            else parameter_provenance["parameter_identity"]
        ),
        parameter_sha256=(
            None
            if parameter_provenance is None
            else parameter_provenance["parameter_sha256"]
        ),
        n_electrons=molecule.n_electrons(),
    )


def run_seccm(molecule: Molecule, ccm_options=None) -> SemiempiricalResult:
    """Run periodic MSINDO via semiempirical CCM (SECCM)."""
    from vibeqc.molecule import ANGSTROM_TO_BOHR as _A2B

    if ccm_options is not None and getattr(ccm_options, "cluster_radius", None):
        raise NotImplementedError(
            "CCMOptions(cluster_radius=...) is accepted for SECCM API discovery, "
            "but automatic cyclic-cluster construction is not implemented yet. "
            "Pass explicit translations=[[a,0,0], ...] to define the CCM "
            "supercell."
        )
    translations = _normalise_ccm_translations(
        None if ccm_options is None else getattr(ccm_options, "translations", None)
    )
    try:
        madelung = ccm_options.madelung
    except AttributeError as exc:
        raise ValueError(
            "method='seccm' needs ccm_options.madelung to state whether the "
            "long-range Ewald/Madelung embedding is active."
        ) from exc
    if int(molecule.multiplicity) != 1:
        raise NotImplementedError(
            "MSINDO SECCM is closed-shell (RHF) only; open-shell / ROHF SECCM is "
            "not implemented. Use a closed-shell cell (multiplicity=1)."
        )
    Z = [at.Z for at in molecule.atoms]
    coords_ang = [[c / _A2B for c in at.xyz] for at in molecule.atoms]

    from vibeqc.semiempirical.methods.msindo_ccm import run_ccm as _run_ccm_validated

    result_obj = _run_ccm_validated(
        Z,
        coords_ang,
        translations,
        charge=int(molecule.charge),
        madelung=madelung,
    )

    from vibeqc.semiempirical.methods.msindo_ccm_gradient_analytic import (
        ccm_gradient_analytic as _seccm_gradient_analytic,
    )

    result = SemiempiricalResult(
        float(result_obj.total_energy),
        lambda: _seccm_gradient_analytic(
            Z,
            coords_ang,
            translations,
            charge=int(molecule.charge),
            madelung=madelung,
        ),
        "seccm",
        converged=bool(result_obj.converged),
        n_iter=int(result_obj.n_iter),
        n_electrons=molecule.n_electrons(),
    )
    result.route_plan = SemiempiricalRoutePlan.from_request(
        "seccm",
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
        ccm_options=ccm_options,
    ).with_seccm_runtime(
        periodic_dimension=len(translations),
        electrostatics_family="madelung" if madelung else "none",
    )
    for name, default in (("stability_checked", False),
                          ("stability_analysis_converged", False),
                          ("stability_eigenvalue", 0.0),
                          ("n_stability_restarts", 0)):
        setattr(result, name, getattr(result_obj, name, default))
    result.internal_instability = False  # unresolved/unstable audited states raise
    return result


# Compatibility symbol retained for existing callers and implementation
# modules. New public code should use ``run_seccm``.
run_ccm = run_seccm


__all__ = [
    "MOLECULAR_SEMIEMPIRICAL_METHODS",
    "SEMIEMPIRICAL_METHOD_ALIASES",
    "SEMIEMPIRICAL_METHODS",
    "SemiempiricalResult",
    "normalise_semiempirical_method",
    "run_ccm",
    "run_seccm",
    "run_semiempirical",
]
