"""GFN2-xTB energy over a frozen SECCM topology.

The multi-replica adapter evaluates the molecular GFN2-xTB machinery over the
WS-weighted supercell: one-center S/H0 blocks are the molecular ones, every
directed two-center coupling (H0 image block, shell-resolved gamma, the
ad-hoc shell-resolved AES potential, pair repulsion) is accumulated through
the WS records with their fractional weights, and the SCC loop mirrors the
molecular driver (damped simple mixing, GAM3 third order, the stabilization
retry ladder). At T=0 the molecular limit (1x1x1 cyclic cluster) reproduces
the validated molecular driver bit-for-bit; at finite T it reports the same
state with the variational Mermin free energy instead of the molecular
driver's historical internal-energy field. The experimental atom-resolved
"faithful" AES and the D4 dispersion stay out, exactly as in the molecular
driver's default path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from vibeqc._vibeqc_core import Molecule
from vibeqc._vibeqc_core import semiempirical as _se_cxx
from vibeqc.semiempirical.routes import (
    GFN2SECCMHamiltonianIdentity,
    GFN2SECCMRunControls,
    SemiempiricalRoutePlan,
)

from ._adapter_common import (
    flatten_topology_records,
    topology_length_unit_scale,
    translation_symmetry_charge_spread,
)
from .topology import SECCMTopology


GFN2_SECCM_PARAMETER_SET = "gfn2-xtb-parameter-registry"

_ATTEMPT_SOLVERS = frozenset(
    {
        "molecular_simple",
        "molecular_hybrid",
        "molecular_diis",
        "molecular_broyden",
        "supercell_simple",
        "supercell_diis",
        "supercell_broyden",
        "supercell_broyden_eyert",
        "supercell_newton",
    }
)
_ATTEMPT_EXIT_REASONS = frozenset(
    {
        "converged",
        "iteration_limit",
        "stalled_checkpoint",
        "unphysical_basin",
        "gap_rejected",
    }
)
#: Exit reasons of an attempt that never reached the SCC residual tolerance:
#: it exhausted its budget, or it stopped contracting at a stabilization
#: checkpoint and was handed to the ladder with the budget it had left (#294).
_ATTEMPT_UNCONVERGED_EXITS = frozenset(
    {"iteration_limit", "stalled_checkpoint"}
)
_SCC_MIXERS = frozenset(
    {"simple", "diis", "broyden", "broyden_eyert", "newton"}
)


@dataclass(frozen=True)
class GFN2SECCMAttempt:
    """Immutable provenance and residual trace for one SCC attempt."""

    index: int
    engine: str
    solver: str
    allocated_max_iter: int
    n_iter: int
    ladder_charge_mixing: float
    electronic_temperature: float
    scc_mixer: str
    restart_source: str
    restart_n_shell_charges: int
    restart_sha256: str | None
    exit_reason: str
    scc_converged: bool
    physical_basin: bool
    residual_trace: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ValueError("GFN2-SECCM attempt index must be non-negative")
        engine = str(self.engine).strip().lower()
        if engine not in {"supercell", "molecular_delegate"}:
            raise ValueError(
                "GFN2-SECCM attempt engine must be 'supercell' or "
                "'molecular_delegate'"
            )
        solver = str(self.solver).strip().lower()
        if solver not in _ATTEMPT_SOLVERS:
            raise ValueError("GFN2-SECCM attempt solver is unsupported")
        expected_prefix = (
            "molecular_" if engine == "molecular_delegate" else "supercell_"
        )
        if not solver.startswith(expected_prefix):
            raise ValueError(
                "GFN2-SECCM attempt solver does not match its engine"
            )
        if (
            type(self.allocated_max_iter) is not int
            or self.allocated_max_iter < 1
        ):
            raise ValueError(
                "GFN2-SECCM attempt allocated_max_iter must be >= 1"
            )
        if (
            type(self.n_iter) is not int
            or self.n_iter < 0
            or self.n_iter > self.allocated_max_iter
        ):
            raise ValueError(
                "GFN2-SECCM attempt n_iter must fit its allocated budget"
            )
        for name in ("ladder_charge_mixing", "electronic_temperature"):
            if isinstance(getattr(self, name), bool):
                raise ValueError(f"GFN2-SECCM attempt {name} must be numeric")
        charge_mixing = float(self.ladder_charge_mixing)
        temperature = float(self.electronic_temperature)
        if (
            not math.isfinite(charge_mixing)
            or charge_mixing <= 0.0
            or charge_mixing > 1.0
        ):
            raise ValueError(
                "GFN2-SECCM attempt ladder_charge_mixing must be in (0, 1]"
            )
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError(
                "GFN2-SECCM attempt electronic_temperature must be finite "
                "and non-negative"
            )
        scc_mixer = str(self.scc_mixer).strip().lower()
        if scc_mixer not in _SCC_MIXERS:
            raise ValueError("GFN2-SECCM attempt scc_mixer is unsupported")
        solver_mixer = (
            "simple"
            if solver == "molecular_hybrid"
            else "broyden_eyert"
            if solver == "supercell_broyden_eyert"
            else solver.rsplit("_", 1)[1]
        )
        if scc_mixer != solver_mixer:
            raise ValueError(
                "GFN2-SECCM attempt solver and SCC mixer disagree"
            )
        restart_source = str(self.restart_source).strip().lower()
        if restart_source not in {"neutral", "supplied"}:
            raise ValueError(
                "GFN2-SECCM attempt restart_source must be neutral or supplied"
            )
        if (
            type(self.restart_n_shell_charges) is not int
            or self.restart_n_shell_charges < 0
        ):
            raise ValueError(
                "GFN2-SECCM attempt restart count must be non-negative"
            )
        restart_sha256 = self.restart_sha256
        if restart_source == "neutral":
            if self.restart_n_shell_charges != 0 or restart_sha256 is not None:
                raise ValueError(
                    "neutral GFN2-SECCM attempt restart has no payload"
                )
        else:
            if self.restart_n_shell_charges < 1 or not isinstance(
                restart_sha256, str
            ):
                raise ValueError(
                    "supplied GFN2-SECCM attempt restart needs a fingerprint"
                )
            restart_sha256 = restart_sha256.strip().lower()
            if len(restart_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in restart_sha256
            ):
                raise ValueError(
                    "GFN2-SECCM attempt restart fingerprint is invalid"
                )
        exit_reason = str(self.exit_reason).strip().lower()
        if exit_reason not in _ATTEMPT_EXIT_REASONS:
            raise ValueError("GFN2-SECCM attempt exit_reason is unsupported")
        if type(self.scc_converged) is not bool or type(
            self.physical_basin
        ) is not bool:
            raise ValueError(
                "GFN2-SECCM attempt convergence flags must be boolean"
            )
        if exit_reason in _ATTEMPT_UNCONVERGED_EXITS and self.scc_converged:
            raise ValueError(
                "iteration-limit and stalled-checkpoint attempts cannot be "
                "SCC-converged"
            )
        if (
            exit_reason not in _ATTEMPT_UNCONVERGED_EXITS
            and not self.scc_converged
        ):
            raise ValueError(
                "converged, gap-rejected, and unphysical attempts must reach "
                "the SCC residual tolerance"
            )
        if exit_reason == "converged" and not self.physical_basin:
            raise ValueError("accepted attempt must be in the physical basin")
        if exit_reason == "unphysical_basin" and self.physical_basin:
            raise ValueError("unphysical attempt must fail the basin gate")
        residual_trace = tuple(float(value) for value in self.residual_trace)
        if len(residual_trace) != self.n_iter or any(
            not math.isfinite(value) or value < 0.0
            for value in residual_trace
        ):
            raise ValueError(
                "GFN2-SECCM attempt residual trace must be finite and match "
                "n_iter"
            )
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(
            self, "ladder_charge_mixing", charge_mixing
        )
        object.__setattr__(self, "electronic_temperature", temperature)
        object.__setattr__(self, "scc_mixer", scc_mixer)
        object.__setattr__(self, "restart_source", restart_source)
        object.__setattr__(self, "restart_sha256", restart_sha256)
        object.__setattr__(self, "exit_reason", exit_reason)
        object.__setattr__(self, "residual_trace", residual_trace)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe attempt payload."""
        return {
            "index": self.index,
            "engine": self.engine,
            "solver": self.solver,
            "allocated_max_iter": self.allocated_max_iter,
            "n_iter": self.n_iter,
            "ladder_charge_mixing": self.ladder_charge_mixing,
            "electronic_temperature": self.electronic_temperature,
            "scc_mixer": self.scc_mixer,
            "restart_source": self.restart_source,
            "restart_n_shell_charges": self.restart_n_shell_charges,
            "restart_sha256": self.restart_sha256,
            "exit_reason": self.exit_reason,
            "scc_converged": self.scc_converged,
            "physical_basin": self.physical_basin,
            "residual_trace": list(self.residual_trace),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object]
    ) -> "GFN2SECCMAttempt":
        """Restore an attempt emitted by :meth:`to_dict`."""
        data = dict(payload)
        trace = data.get("residual_trace")
        if isinstance(trace, list):
            data["residual_trace"] = tuple(trace)
        return cls(**data)


@dataclass(frozen=True)
class GFN2SECCMResult:
    """One finite-cluster GFN2-SECCM result, per primitive cell.

    ``energy`` and ``free_energy`` are the same Mermin free energy. The
    ``e_electronic`` component includes band0, SCC, AES, third-order,
    optional Madelung, and ``-T*S`` contributions, so
    ``energy == e_electronic + e_repulsive``. Fields prefixed with
    ``cyclic_`` retain the corresponding unnormalized finite-cluster totals.
    """

    energy: float
    free_energy: float
    entropy: float
    smearing_temperature: float
    e_electronic: float
    e_repulsive: float
    e_scc: float
    e_band0: float
    e_aes: float
    e_3rd: float
    e_madelung: float
    total_cyclic_energy: float
    cyclic_electronic_energy: float
    cyclic_repulsive_energy: float
    cyclic_scc_energy: float
    cyclic_aes_energy: float
    cyclic_3rd_energy: float
    cyclic_band0_energy: float
    cyclic_madelung_energy: float
    #: Frontier gap (Ha) of the converged finite torus.  Measured at every
    #: electronic temperature whenever a frontier exists; 0.0 only when no
    #: frontier is defined (``n_occ`` is 0 or equals ``n_basis``).
    #: NaN on a rejected attempt when overlap screening retains the occupied
    #: manifold but removes every virtual orbital. Smearing cannot waive an
    #: unavailable frontier.
    homo_lumo_gap: float
    #: True when ``homo_lumo_gap`` is at or below the applied guard epsilon
    #: (``run_controls.finite_torus_gap_tolerance``) and the positive-gap
    #: requirement was waived because finite electronic temperature resolves
    #: the occupation.  A waived row is admissible but is **not** a gapped
    #: row; at ``electronic_temperature = 0`` it is rejected instead.
    gap_guard_waived: bool
    mo_energies: np.ndarray
    mo_coeffs: np.ndarray
    density: np.ndarray
    overlap: np.ndarray
    hamiltonian: np.ndarray
    charges: np.ndarray
    shell_charges: np.ndarray
    scc_max_change_trace: np.ndarray
    n_basis: int
    n_occ: int
    n_iter: int
    group_order: int
    n_records: int
    converged: bool
    attempts: tuple[GFN2SECCMAttempt, ...]
    selected_attempt_index: int | None
    molecular_delegated: bool
    physical_basin: bool
    #: Largest Mulliken-charge spread (electrons) within any orbit of the
    #: bound finite translation group. The cyclic cluster makes the atoms
    #: of an orbit equivalent by construction, so a converged state that
    #: respects the model's translation symmetry has spread 0 and any
    #: nonzero value is a symmetry violation - the charge-density-wave and
    #: state-hopping family of issue #421, which every magnitude-based
    #: acceptance bound is structurally blind to.
    #:
    #: Reported, never gated (maintainer decision D3, 2026-08-28). Before
    #: issue #354 every production state carried a spurious spread because
    #: the GFN2 H0 coordination numbers were computed molecularly and broke
    #: the cyclic symmetry upstream of the SCC (fcc Cu: 0.456 e at 4x4x4,
    #: 0.903 e at 2x2x2). With the cyclic coordination numbers those
    #: recipes measure 5.9e-4 e and 8.3e-5 e, at the SCC charge tolerance,
    #: while a genuinely symmetry-broken attractor (fcc Cu 2x2x2 under
    #: plain simple mixing) measures 0.84 e - so the number now separates
    #: the two cleanly. The gate decision deferred by D3 is due; until it
    #: is taken this exposes the number and leaves the judgement to the
    #: caller.
    #:
    #: Scope: this detects breaking of TRANSLATION symmetry. Atoms
    #: equivalent only under a point operation - the two basis atoms of
    #: diamond, say - are separate orbits here and carry no constraint;
    #: catching those needs the crystallographic orbit and is separate.
    #: NaN for absent or nonfinite charges. Finite partial charges on a
    #: failed result are diagnosed without implying convergence.
    translation_symmetry_charge_spread: float
    normalization: str
    parameter_set: str
    parameter_identity: str
    parameter_sha256: str
    topology_fingerprint: str
    reference_geometry_fingerprint: str
    hamiltonian_identity: GFN2SECCMHamiltonianIdentity
    run_controls: GFN2SECCMRunControls
    route_plan: SemiempiricalRoutePlan


@dataclass(frozen=True)
class GFN2SECCMStateChange:
    """Signed changes (current minus previous) between two converged states.

    Charge RMS and symmetry spread are in electrons; energies and gaps are
    in Hartree. Energies retain the results' per-primitive-cell normalization.
    These observables have no built-in threshold or acceptance verdict.
    """

    charge_rms_change: float
    homo_lumo_gap_change: float
    scc_energy_change: float
    free_energy_change: float
    translation_symmetry_spread_change: float
    electronic_temperature_change: float
    same_hamiltonian: bool


def compare_gfn2_seccm_states(
    previous: GFN2SECCMResult,
    current: GFN2SECCMResult,
) -> GFN2SECCMStateChange:
    """Report SCC-state changes along a caller-ordered scan or size ladder.

    RMS charges are intensive and need no atom mapping between cluster
    sizes. The caller must supply the same composition per primitive cell;
    this helper neither infers site correspondence nor establishes state
    continuity from scalar observables alone. A nonzero change is evidence
    to inspect, not a rejection or proof of a spurious state.

    ``same_hamiltonian`` compares method/parameter and executed-option
    provenance, not geometry: changing lattice size is the intended use.
    It is false if a retry changed temperature or if a different kernel or
    parameter set ran. Solver choice is deliberately outside that identity.
    Failed/partial states and nonfinite observables cannot supply a valid
    scan difference and raise ``ValueError``.
    """
    snapshots = []
    for result in (previous, current):
        if not isinstance(result, GFN2SECCMResult):
            raise TypeError("state comparison requires GFN2SECCMResult objects")
        if not result.converged:
            raise ValueError("state comparison requires converged results")
        if result.normalization != "per_primitive_cell":
            raise ValueError("state comparison requires per-primitive-cell energies")
        charges = np.asarray(result.charges, dtype=float)
        if charges.ndim != 1 or charges.size == 0 or not np.isfinite(charges).all():
            raise ValueError("state comparison requires finite atomic charges")
        snapshot = np.array([
            np.linalg.norm(charges) / np.sqrt(charges.size),
            result.homo_lumo_gap,
            result.e_scc,
            result.free_energy,
            result.translation_symmetry_charge_spread,
            result.smearing_temperature,
        ])
        if not np.isfinite(snapshot).all():
            raise ValueError("state comparison requires finite observables")
        snapshots.append(snapshot)
    changes = snapshots[1] - snapshots[0]
    return GFN2SECCMStateChange(
        *(float(value) for value in changes),
        same_hamiltonian=(
            previous.hamiltonian_identity == current.hamiltonian_identity
            and previous.parameter_identity == current.parameter_identity
            and previous.parameter_sha256 == current.parameter_sha256
        ),
    )


class GFN2SECCMConvergenceError(RuntimeError):
    """A bounded GFN2-SECCM run failed, with its full frozen result."""

    def __init__(self, message: str, result: GFN2SECCMResult) -> None:
        super().__init__(message)
        self.result = result
        self.reason = result.attempts[-1].exit_reason


def _frozen_array(value) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _native_mixer_name(value) -> str:
    for name, member in (
        ("simple", _se_cxx.SCCMixer.Simple),
        ("diis", _se_cxx.SCCMixer.DIIS),
        ("broyden", _se_cxx.SCCMixer.Broyden),
        ("broyden_eyert", _se_cxx.SCCMixer.BroydenEyert),
        ("newton", _se_cxx.SCCMixer.Newton),
    ):
        if value == member:
            return name
    raise RuntimeError("C++ GFN2-SECCM returned an unknown SCC mixer")


def _project_attempts(
    native,
    run_controls: GFN2SECCMRunControls,
) -> tuple[tuple[GFN2SECCMAttempt, ...], int | None, np.ndarray]:
    """Freeze and verify the native attempt ledger."""
    projected: list[GFN2SECCMAttempt] = []
    restart = run_controls.restart
    expected_engine = (
        "molecular_delegate"
        if bool(native.molecular_delegated)
        else "supercell"
    )
    for index, native_attempt in enumerate(native.attempts):
        supplied = bool(native_attempt.restart_supplied)
        if supplied:
            if index != 0 or restart.primary_source != "supplied":
                raise RuntimeError(
                    "C++ GFN2-SECCM returned inconsistent restart-attempt "
                    "provenance"
                )
            restart_source = "supplied"
            restart_count = restart.n_shell_charges
            restart_sha256 = restart.shell_charges_sha256
        else:
            restart_source = "neutral"
            restart_count = 0
            restart_sha256 = None
        if index == 0 and restart.primary_source != restart_source:
            raise RuntimeError(
                "C++ GFN2-SECCM did not execute the requested primary restart"
            )
        if index > 0 and restart_source != "neutral":
            raise RuntimeError(
                "C++ GFN2-SECCM automatic retry did not use a neutral start"
            )
        engine = (
            "molecular_delegate"
            if bool(native_attempt.molecular_delegated)
            else "supercell"
        )
        if engine != expected_engine:
            raise RuntimeError(
                "C++ GFN2-SECCM returned inconsistent attempt-engine "
                "provenance"
            )
        projected.append(
            GFN2SECCMAttempt(
                index=index,
                engine=engine,
                solver=str(native_attempt.solver),
                allocated_max_iter=int(native_attempt.allocated_max_iter),
                n_iter=int(native_attempt.n_iter),
                ladder_charge_mixing=float(
                    native_attempt.ladder_charge_mixing
                ),
                electronic_temperature=float(
                    native_attempt.electronic_temperature
                ),
                scc_mixer=_native_mixer_name(native_attempt.scc_mixer),
                restart_source=restart_source,
                restart_n_shell_charges=restart_count,
                restart_sha256=restart_sha256,
                exit_reason=str(native_attempt.exit_reason),
                scc_converged=bool(native_attempt.scc_converged),
                physical_basin=bool(native_attempt.physical_basin),
                residual_trace=tuple(
                    float(value) for value in native_attempt.max_change_trace
                ),
            )
        )
    attempts = tuple(projected)
    if not attempts:
        raise RuntimeError("C++ GFN2-SECCM returned an empty attempt ledger")
    if sum(attempt.n_iter for attempt in attempts) != int(native.n_iter):
        raise RuntimeError(
            "C++ GFN2-SECCM attempt iterations do not reconcile with n_iter"
        )
    aggregate_trace = tuple(
        value for attempt in attempts for value in attempt.residual_trace
    )
    native_trace_tuple = tuple(
        float(value) for value in native.scc_max_change_trace
    )
    if aggregate_trace != native_trace_tuple:
        raise RuntimeError(
            "C++ GFN2-SECCM aggregate residual trace does not reconcile "
            "with its attempts"
        )
    if len(native_trace_tuple) != int(native.n_iter):
        raise RuntimeError(
            "C++ GFN2-SECCM residual trace does not reconcile with n_iter"
        )
    if int(native.n_iter) > run_controls.max_iter:
        raise RuntimeError(
            "C++ GFN2-SECCM exceeded the requested total iteration budget"
        )
    selected_native = int(native.selected_attempt_index)
    selected = None if selected_native == -1 else selected_native
    if selected is not None:
        if selected != len(attempts) - 1:
            raise RuntimeError(
                "C++ GFN2-SECCM selected attempt must be the final attempt"
            )
        if attempts[selected].exit_reason != "converged":
            raise RuntimeError(
                "C++ GFN2-SECCM selected a rejected SCC attempt"
            )
    if any(attempt.exit_reason == "converged" for attempt in attempts[:-1]):
        raise RuntimeError(
            "C++ GFN2-SECCM continued after an accepted SCC attempt"
        )
    if bool(native.converged) != (selected is not None):
        raise RuntimeError(
            "C++ GFN2-SECCM convergence and selected-attempt provenance "
            "disagree"
        )
    return attempts, selected, _frozen_array(native.scc_max_change_trace)


def run_gfn2_seccm(
    molecule: Molecule,
    topology: SECCMTopology,
    *,
    parameter_set: str = GFN2_SECCM_PARAMETER_SET,
    max_iter: int = 3600,
    conv_tol_charge: float = 1.0e-6,
    charge_mixing: float | None = None,
    electronic_temperature: float = 0.0,
    madelung: bool = False,
    madelung_s_weighted: bool = False,
    madelung_no_self: bool = False,
    ewald_gamma: bool = False,
    include_aes: bool = True,
    ewald_gamma_molecular_onsite: bool = False,
    ewald_gamma_k0_global: bool = False,
    gamma_form: str = "klopman_ohno",
    aes_faithful: bool = False,
    aes_damping: float = 0.5,
    scc_mixer: str = "simple",
    initial_shell_charges: np.ndarray | None = None,
) -> GFN2SECCMResult:
    """Evaluate the GFN2-xTB-SECCM finite-cluster route.

    The route is restricted to neutral closed-shell clusters with one to
    three cyclic dimensions. It evaluates the complete frozen WS record set
    with the WS-weighted supercell GFN2-xTB SCC (molecular-limit bit parity
    with the molecular driver; the experimental atom-resolved faithful AES
    and D4 dispersion are not part of the adapter). Charged cells and
    open-shell inputs fail closed until the embedding and unrestricted
    layers land.

    ``electronic_temperature`` switches on Fermi-Dirac fractional
    occupations (Ha; 0 = hard Aufbau, the default). Metallic supercells
    (bulk Cu/Pd/Ag and other degenerate-frontier clusters) can jump between
    occupation branches under hard Aufbau, making the energy surface
    discontinuous between neighbouring geometries; a small temperature
    (e.g. 0.002-0.01 Ha) converges branch-free and the reported energy is
    the Mermin free energy A = E - T*S (``free_energy`` alias, with
    ``entropy`` and ``smearing_temperature`` recorded).

    A converged state that violates a shell's Pauli capacity, the conservative
    system-wide RMS polarization guards, or the 4-electron local atom/shell
    heuristic guard is never returned. These diagnostics catch over-polarized
    shell-transfer and charge-density-wave fixed points without allowing one
    outlier to be diluted by system size. The stabilization ladder searches
    for the physical basin; if it cannot find one the run fails closed.

    ``madelung=True`` enables the opt-in Madelung/Ewald embedding for
    neutral ionic cells: the Coulombic image-sum tail beyond the WS cell
    acts on the atomic Mulliken fluctuations (MSINDO CCM convention,
    SMADEL short-range subtraction; 1-D background-corrected wire Ewald,
    2-D Parry/Heyes, 3-D Ewald), reported as ``e_madelung`` /
    ``cyclic_madelung_energy``. The embedding is off by default and the
    molecular-limit bit parity holds with it off.

    ``madelung_s_weighted=True`` is an experimental validation knob that
    deposits the Madelung potential S-weighted (0.5 S (V_A + V_B), the
    gamma channel's form - the variational convention for the
    non-orthogonal basis) instead of the shipped diagonal form. It is
    for channel isolation of SCC instabilities only (IID 150: a
    synthetic aligned-plane stress cell's stiff map); the reported energy
    then uses a different deposit convention and must not be used for
    production numbers. It requires ``madelung=True``; a dependent flag
    without its parent is rejected rather than ignored.

    ``madelung_no_self=True`` is an experimental validation knob that
    zeroes the embedding self term (the madkonst diagonal - the 2-D
    Parry/Heyes lattice self potential that SMADEL does not subtract).
    The SCC deposit and reported energy both use that same diagonal-removed
    operator, so the option is internally energy-consistent; it remains a
    deliberately modified validation Hamiltonian for channel isolation, not
    the default physical kernel. It requires ``madelung=True``.

    ``ewald_gamma=True`` selects the self-consistent periodic shell gamma:
    full bare-Coulomb Ewald plus the Wigner-Seitz-folded
    kernel-minus-Coulomb remainder and the on-site hardness, in one, two or
    three dimensions. It is mutually exclusive with ``madelung``. Exact
    zero-image molecular delegation is unchanged in every dimension; group
    order one alone does not imply molecular delegation. See
    ``docs/design_seccm_gfn2_long_range_gamma.md``.

    ``gamma_form`` selects the short-range kernel that remainder is taken
    from, and decides whether the route has a thermodynamic limit (#444):

    * ``"klopman_ohno"`` (the default) is the GFN2 paper's own kernel and
      reproduces the shipped 1-D and 2-D numbers exactly. Its remainder
      decays only as ``-eta^2/2R^3`` with a pair-dependent amplitude, whose
      charge-weighted three-dimensional sum need not cancel under
      neutrality, so a finite WS truncation of it is not a thermodynamic
      limit. **Non-molecular 3-D cyclic topologies fail closed with it.**
    * ``"elstner"`` is the Elstner et al. 1998 Eq. 17/18 form
      (``gamma = 1/R - S``, ``tau = 16/5 U``). ``S`` decays exponentially,
      so the image sum converges absolutely in every dimension and 3-D
      cells are available. It changes the 1-D and 2-D numbers, and no
      alternative GFN2 hardness mapping is introduced: the on-site block
      keeps the published parameterisation and only the ``R > 0`` pair
      kernel differs.

    ``include_aes=False`` is an experimental validation knob that
    removes the WS-folded anisotropic second-order (AES) channel (the
    ad-hoc shell-resolved gamma^3/gamma^5 multipole potential and its
    energy term) for channel isolation of SCC instabilities. The
    reported energy then misses a physical term and must not be used
    for production numbers. Even on a zero-image record set this option
    runs the SECCM engine instead of delegating to the molecular driver,
    which does not expose this model change.

    ``aes_faithful=True`` replaces the ad-hoc shell-resolved
    ``gamma^3``/``gamma^5`` multipole channel with the Bannwarth 2019 model
    (Eq. 25 energy, Eqs. 39-44 Fock) on cumulative atomic multipole moments
    built from the **Wigner-Seitz record inventory** rather than from the
    home-cell molecule. That construction is what fixes issue #348: with the
    default kernel the AES is the only remaining term in the route that
    depends on which lattice copy of an atom the caller typed. On a 4-unit
    polar HF chain, three representatives of one crystal give

    ==========  ===============  ===============  ===============
    channel     as typed         F(0) + T         H(0) + T
    ==========  ===============  ===============  ===============
    ad-hoc      -5.230514880891  -5.230364176724  -5.230359290006
    faithful    -5.233747472334  -5.233747472334  -5.233747472334
    AES off     -5.231529541282  -5.231529541282  -5.231529541282
    ==========  ===============  ===============  ===============

    a 1.6e-4 Ha spread that the record-resolved construction removes exactly.
    The two paths cannot be blended -- the ad-hoc kernel consumes
    shell-resolved moments and a pair inventory produces atom-resolved ones
    -- so this is a model flag, not a repair of the existing kernel.

    It defaults to ``False`` while the molecular driver's own
    ``aes_faithful`` default is ``False``, so that molecular-limit bit parity
    between the two routes keeps meaning; the defaults move together. On a
    zero-image topology the flag is passed through to the delegated
    molecular solve, so a single-replica torus returns the model that was
    asked for.

    On the faithful path the multipole moments are part of the mixed SCC
    state, not a lagged input to it (issue #409): the reduced iterate is
    ``[dq_shell; mu; theta]`` and one mixer advances all of it, which is what
    xtb does and what makes a comparable trajectory possible at all.
    Convergence is measured on that whole residual, so a stationary charge
    vector with drifting moments is not reported as converged. On the polar
    HF chain every mixer then reaches the same energy to twelve digits --
    simple in 129 iterations, Broyden in 25, DIIS in 23, and the Eyert scheme
    in 9 -- where the shipped path's mixers agree only to about 1e-8.
    ``scc_mixer="newton"`` is refused with ``aes_faithful=True``, because its
    finite-difference Jacobian is defined on the charge subspace alone.

    ``aes_damping`` is forwarded to the *molecular* driver on a zero-image
    delegating topology, so the delegated solve runs the model that was
    asked for. The SECCM engine does not use it: the mixer supplies the
    damping, and a separate potential lag would be a second uncontrolled
    one.

    ``ewald_gamma_molecular_onsite=True`` is an experimental validation
    knob that restores the molecular on-site block in the periodic
    shell gamma (skipping the Ewald lattice self potential at d = 0,
    whose 2-D Parry/Heyes k0/self terms carry a negative shift). It is
    for channel isolation of SCC instabilities only (IID 150: the
    synthetic aligned-plane stress-map anti-screening runaway); the
    reported energy then uses a hybrid kernel and must not be used for
    production numbers. It requires ``ewald_gamma=True``.

    ``ewald_gamma_k0_global=True`` is retained as a compatibility no-op.
    Both values use the full exact pairwise Parry/de Leeuw K=0 kernel; the
    former rank-1 perpendicular-dipole approximation is not selected.

    ``scc_mixer`` selects the charge mixer: ``"simple"`` (default,
    the shipped behaviour), ``"broyden"``, ``"broyden_eyert"``,
    ``"diis"``, or ``"newton"``.
    The opt-in quasi-Newton mixers exist for maps whose fixed point
    simple mixing orbits (the four-layer synthetic aligned-plane stress
    map, IID 141). ``"newton"`` is a line-searched, frozen-lagged-
    multipole finite-difference chord/quasi-Newton step with residual-
    reduction damping, not a full Newton solve. It is the most expensive
    mixer and is meant for cycle-bound maps, not routine runs.
    ``"newton"`` fails explicitly on a zero-image molecular-delegation
    topology because that custom frozen-multipole map is not implemented by
    the molecular driver; it is never silently replaced by simple mixing.
    ``"broyden_eyert"`` is the tblite ``broyden.f90`` modified Broyden
    scheme (inverse-norm history weighting, the charge-difference
    u-vector term, and a full-budget history per tblite's memory
    default), i.e. the mixer the xtb binary uses. Historical fcc Cu 3-D
    ``ewald_gamma`` solver/energy anchors remain in the #130 tests as
    evidence, but those production recipes now reject under #444. Mixer
    choice cannot repair a kernel without a thermodynamic limit. The Eyert
    option is SECCM-only and the molecular driver rejects it.

    ``charge_mixing`` defaults per mixer when left unset: 0.4 for
    ``"broyden_eyert"`` (xtb's bromix default) and 0.1 for every other
    mixer. An explicit value in (0, 1] always wins.

    ``initial_shell_charges`` optionally supplies one finite charge
    fluctuation per GFN2 shell as the SCC starting point. A nonempty vector
    must have exactly ``n_shells`` entries. It is unavailable on an exact
    molecular-delegation topology because the delegated molecular driver has
    no shell-charge restart channel; such a vector is rejected rather than
    silently ignored.
    """
    if initial_shell_charges is None:
        restart_shell_charges = np.zeros(0, dtype=float)
    else:
        # One owned snapshot feeds both provenance hashing and the native
        # call. Caller mutation cannot make the recorded restart differ from
        # the state the SCC engine consumed.
        restart_shell_charges = np.array(
            initial_shell_charges,
            dtype=float,
            copy=True,
            order="C",
        )
        if restart_shell_charges.ndim != 1:
            raise ValueError(
                "GFN2-SECCM initial_shell_charges must be a "
                "one-dimensional vector"
            )
        if not np.all(np.isfinite(restart_shell_charges)):
            raise ValueError(
                "GFN2-SECCM initial_shell_charges must contain only "
                "finite values"
            )

    if parameter_set != GFN2_SECCM_PARAMETER_SET:
        raise ValueError(
            "GFN2-SECCM supports only parameter_set="
            f"{GFN2_SECCM_PARAMETER_SET!r}"
        )
    if (madelung_s_weighted or madelung_no_self) and not madelung:
        raise ValueError(
            "GFN2-SECCM madelung_s_weighted and madelung_no_self require "
            "madelung=True"
        )
    if ewald_gamma_molecular_onsite and not ewald_gamma:
        raise ValueError(
            "GFN2-SECCM ewald_gamma_molecular_onsite requires "
            "ewald_gamma=True"
        )
    electrostatics_family = (
        "ewald_gamma" if ewald_gamma else "madelung" if madelung else "none"
    )
    route_plan = SemiempiricalRoutePlan.from_request(
        "gfn2_xtb",
        boundary="seccm",
        properties=("energy",),
        charge=int(molecule.charge),
        multiplicity=int(molecule.multiplicity),
    )
    (
        translations,
        central,
        origin,
        shell_labels,
        weights,
        multiplicities,
        displacements,
        primitive_vectors,
        replicas,
    ) = flatten_topology_records(molecule, topology, route_name="GFN2-SECCM")

    group = topology.finite_group
    assert group is not None
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    params = load_gfn2_params()
    # Short-range shell gamma forms selectable under ewald_gamma. The values
    # are the native ShellGammaForm enum; the names are the stable public
    # spelling used by the route plan and the run manifest.
    _SHELL_GAMMA_FORM_BY_NAME = {
        "klopman_ohno": _se_cxx.ShellGammaForm.KlopmanOhno,
        "elstner": _se_cxx.ShellGammaForm.Elstner,
    }
    if gamma_form not in _SHELL_GAMMA_FORM_BY_NAME:
        raise ValueError(
            f"gamma_form must be one of {sorted(_SHELL_GAMMA_FORM_BY_NAME)}"
        )
    if gamma_form != "klopman_ohno" and not ewald_gamma:
        raise ValueError(
            "gamma_form selects the short-range kernel of the Ewald-split "
            "periodic shell gamma and requires ewald_gamma=True; without it "
            "the route is the unembedded WS-truncated Klopman-Ohno sum"
        )
    _SCC_MIXER_BY_NAME = {
        "simple": _se_cxx.SCCMixer.Simple,
        "diis": _se_cxx.SCCMixer.DIIS,
        "broyden": _se_cxx.SCCMixer.Broyden,
        "broyden_eyert": _se_cxx.SCCMixer.BroydenEyert,
        "newton": _se_cxx.SCCMixer.Newton,
    }
    if scc_mixer not in _SCC_MIXER_BY_NAME:
        raise ValueError(
            f"scc_mixer must be one of {sorted(_SCC_MIXER_BY_NAME)}"
        )
    if charge_mixing is None:
        # Per-mixer default: xtb's bromix default 0.4 for the Eyert
        # Broyden scheme (tblite broyden.f90), the SECCM simple-mixing
        # floor 0.1 for every other mixer. An explicit caller value
        # always wins.
        charge_mixing = 0.4 if scc_mixer == "broyden_eyert" else 0.1
    run_controls = GFN2SECCMRunControls.from_request(
        parameter_set=parameter_set,
        max_iter=max_iter,
        conv_tol_charge=conv_tol_charge,
        charge_mixing=charge_mixing,
        scc_mixer=scc_mixer,
        ewald_gamma_k0_global=ewald_gamma_k0_global,
        initial_shell_charges=restart_shell_charges,
    )
    native = _se_cxx._run_gfn2_seccm_from_records(
        molecule,
        params,
        translations,
        central,
        origin,
        np.asarray(shell_labels, dtype=np.int32),
        weights,
        multiplicities,
        np.asarray(displacements, dtype=float),
        primitive_vectors,
        replicas,
        float(group.geometry_tolerance) * topology_length_unit_scale(topology),
        max_iter=int(max_iter),
        conv_tol_charge=float(conv_tol_charge),
        charge_mixing=float(charge_mixing),
        electronic_temperature=float(electronic_temperature),
        madelung=bool(madelung),
        madelung_s_weighted=bool(madelung_s_weighted),
        madelung_no_self=bool(madelung_no_self),
        ewald_gamma=bool(ewald_gamma),
        include_aes=bool(include_aes),
        ewald_gamma_molecular_onsite=bool(
            ewald_gamma_molecular_onsite
        ),
        ewald_gamma_k0_global=bool(ewald_gamma_k0_global),
        gamma_form=_SHELL_GAMMA_FORM_BY_NAME[gamma_form],
        aes_faithful=bool(aes_faithful),
        aes_damping=float(aes_damping),
        scc_mixer=_SCC_MIXER_BY_NAME[scc_mixer],
        initial_shell_charges=restart_shell_charges,
    )
    native_requested = (
        float(native.requested_electronic_temperature),
        bool(native.requested_madelung),
        bool(native.requested_madelung_s_weighted),
        bool(native.requested_madelung_no_self),
        bool(native.requested_ewald_gamma),
        bool(native.requested_include_aes),
        bool(native.requested_ewald_gamma_molecular_onsite),
    )
    python_requested = (
        float(electronic_temperature),
        bool(madelung),
        bool(madelung_s_weighted),
        bool(madelung_no_self),
        bool(ewald_gamma),
        bool(include_aes),
        bool(ewald_gamma_molecular_onsite),
    )
    if native_requested != python_requested:
        raise RuntimeError(
            "C++ GFN2-SECCM returned inconsistent Hamiltonian option "
            "provenance"
        )
    native_restart = np.array(
        native.requested_initial_shell_charges,
        dtype=float,
        copy=True,
    )
    native_controls = (
        int(native.requested_max_iter),
        float(native.requested_conv_tol_charge),
        float(native.requested_charge_mixing),
        _native_mixer_name(native.requested_scc_mixer),
        bool(native.requested_ewald_gamma_k0_global),
        float(native.requested_finite_torus_gap_tolerance),
    )
    python_controls = (
        run_controls.max_iter,
        run_controls.conv_tol_charge,
        run_controls.charge_mixing,
        run_controls.scc_mixer,
        run_controls.requested_ewald_gamma_k0_global,
        run_controls.finite_torus_gap_tolerance,
    )
    if native_controls != python_controls or not np.array_equal(
        native_restart,
        restart_shell_charges,
    ):
        raise RuntimeError(
            "C++ GFN2-SECCM returned inconsistent solver/restart option "
            "provenance"
        )
    attempts, selected_attempt_index, aggregate_trace = _project_attempts(
        native,
        run_controls,
    )
    # Stabilization may have converged at a bounded finite-T retry even when
    # the caller requested Aufbau occupations. A zero-image ewald_gamma
    # request may instead have delegated to the exact molecular driver. Cite
    # the route that actually executed in both cases.
    actual_electrostatics_family = (
        "none" if bool(native.molecular_delegated) else electrostatics_family
    )
    route_plan = route_plan.with_gfn2_seccm_runtime(
        periodic_dimension=topology.dimensionality,
        requested_electrostatics_family=electrostatics_family,
        resolved_electrostatics_family=actual_electrostatics_family,
        requested_electronic_temperature=float(electronic_temperature),
        resolved_electronic_temperature=float(native.smearing_temperature),
        include_aes=bool(native.requested_include_aes),
        madelung_s_weighted=bool(native.requested_madelung_s_weighted),
        madelung_no_self=bool(native.requested_madelung_no_self),
        ewald_gamma_molecular_onsite=bool(
            native.requested_ewald_gamma_molecular_onsite
        ),
        molecular_delegated=bool(native.molecular_delegated),
        run_controls=run_controls,
    )
    hamiltonian_identity = route_plan.gfn2_hamiltonian_identity
    assert hamiltonian_identity is not None
    result = GFN2SECCMResult(
        energy=float(native.energy),
        free_energy=float(native.free_energy),
        entropy=float(native.entropy),
        smearing_temperature=float(native.smearing_temperature),
        e_electronic=float(native.e_electronic),
        e_repulsive=float(native.e_repulsive),
        e_scc=float(native.e_scc),
        e_band0=float(native.e_band0),
        e_aes=float(native.e_aes),
        e_3rd=float(native.e_3rd),
        e_madelung=float(native.e_madelung),
        total_cyclic_energy=float(native.total_cyclic_energy),
        cyclic_electronic_energy=float(native.cyclic_electronic_energy),
        cyclic_repulsive_energy=float(native.cyclic_repulsive_energy),
        cyclic_scc_energy=float(native.cyclic_scc_energy),
        cyclic_aes_energy=float(native.cyclic_aes_energy),
        cyclic_3rd_energy=float(native.cyclic_3rd_energy),
        cyclic_band0_energy=float(native.cyclic_band0_energy),
        cyclic_madelung_energy=float(native.cyclic_madelung_energy),
        homo_lumo_gap=float(native.homo_lumo_gap),
        gap_guard_waived=bool(native.gap_guard_waived),
        mo_energies=_frozen_array(native.mo_energies),
        mo_coeffs=_frozen_array(native.mo_coeffs),
        density=_frozen_array(native.density),
        overlap=_frozen_array(native.overlap),
        hamiltonian=_frozen_array(native.hamiltonian),
        charges=_frozen_array(native.charges),
        shell_charges=_frozen_array(native.shell_charges),
        scc_max_change_trace=aggregate_trace,
        n_basis=int(native.n_basis),
        n_occ=int(native.n_occ),
        n_iter=int(native.n_iter),
        group_order=int(native.group_order),
        n_records=int(native.n_records),
        converged=bool(native.converged),
        attempts=attempts,
        selected_attempt_index=selected_attempt_index,
        molecular_delegated=bool(native.molecular_delegated),
        physical_basin=bool(native.physical_basin),
        translation_symmetry_charge_spread=(
            translation_symmetry_charge_spread(
                molecule, topology, native.charges
            )
        ),
        normalization="per_primitive_cell",
        parameter_set=parameter_set,
        parameter_identity=str(native.parameter_identity),
        parameter_sha256=str(native.parameter_sha256),
        topology_fingerprint=topology.topology_fingerprint,
        reference_geometry_fingerprint=topology.reference_geometry_fingerprint,
        hamiltonian_identity=hamiltonian_identity,
        run_controls=run_controls,
        route_plan=route_plan,
    )
    if not result.converged:
        reason = result.attempts[-1].exit_reason
        if reason == "gap_rejected":
            message = (
                "GFN2-SECCM requires a positive finite-torus HOMO-LUMO gap"
            )
        elif reason == "unphysical_basin":
            message = (
                "GFN2-SECCM converged to an over-polarized shell-charge "
                "fixed point (intra-atomic shell transfer or a "
                "charge-density-wave charge separation); this is a "
                "spurious basin, not a physical result. Retry with a "
                "larger supercell or a different electronic_temperature."
            )
        else:
            message = (
                "C++ GFN2-SECCM did not converge within the iteration budget"
            )
        raise GFN2SECCMConvergenceError(message, result)
    return result


__all__ = [
    "GFN2_SECCM_PARAMETER_SET",
    "GFN2SECCMAttempt",
    "GFN2SECCMConvergenceError",
    "GFN2SECCMResult",
    "GFN2SECCMStateChange",
    "compare_gfn2_seccm_states",
    "run_gfn2_seccm",
]
