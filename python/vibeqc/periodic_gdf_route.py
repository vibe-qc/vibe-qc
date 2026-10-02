"""Which engine evaluates a bulk Gamma-point GDF calculation.

Bulk (``dim == 3``) Gaussian density fitting has one Hamiltonian at every
k mesh: the fitted Hartree term and the exchange with the
``exxdiv='ewald'`` finite-size correction, the convention PySCF's
``pbc.df.GDF`` uses. Gamma is the one-point mesh of that Hamiltonian, so
``run_krhf_periodic_gdf(kmesh=(1, 1, 1))`` is its reference evaluation.

The exact-Gamma drivers in :mod:`vibeqc.pbc_gdf` (``run_pbc_gdf_rhf`` and
the singlet use of ``run_pbc_gdf_uks``) evaluate the same Hamiltonian with
real Gamma arithmetic. They serve a Gamma job only when they implement
every requested option. Any other bulk Gamma job runs the k-mesh engine at
``(1, 1, 1)``, the engine an explicit ``kpoints=(1, 1, 1)`` selects.

Before this module a Gamma job the exact-Gamma drivers could not serve
fell back to ``run_rhf_periodic_gamma_gdf``, whose bulk Hamiltonian is
Ewald J plus real-space exchange with no divergence correction. A level
shift, Fock mixing, a smearing temperature or a net cell charge therefore
changed the energy by the Madelung exchange term: 75.65 mHa on H2/STO-3G in
a 6-bohr cube, 18.4 mHa on HeH+ in a 7-bohr cube (#400, #54, #87, #95).
That fallback remains only for ``dim < 3``, which the k-mesh engine does
not cover at Gamma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping, Optional, Tuple

from .periodic.exchange_convention import BVK_EWALD, MOLECULAR_LIMIT

#: SCF controls each exact-Gamma driver implements. A driver refuses any
#: other control rather than dropping it (see
#: :func:`refuse_unimplemented_gamma_controls`).
_IMPLEMENTED_CONTROLS = {
    "run_pbc_gdf_rhf": frozenset(),
    "run_pbc_gdf_uhf": frozenset({"smearing", "fock_mixing"}),
    "run_pbc_gdf_uks": frozenset({"smearing", "fock_mixing"}),
}

#: Controls with which a closed-shell Gamma job may still take the
#: exact-Gamma driver. A control outside this set is not an error: it sends
#: the job to the k-mesh engine. Smeared RKS stays on the k-mesh engine,
#: whose closed-shell fill shares one Fermi level between the spins.
_CLOSED_SHELL_ROUTE_CONTROLS = {
    "run_pbc_gdf_rhf": frozenset(),
    "run_pbc_gdf_uks": frozenset({"fock_mixing"}),
}


def _requested_controls(smearing_temperature, fock_mixing, level_shift):
    requested = []
    if float(smearing_temperature or 0.0) > 0.0:
        requested.append("smearing")
    if float(fock_mixing or 0.0) != 0.0:
        requested.append("fock_mixing")
    if float(level_shift or 0.0) != 0.0:
        requested.append("level_shift")
    return requested


_CONTROL_WORDS = {
    "smearing": "smearing",
    "fock_mixing": "Fock mixing",
    "level_shift": "level shift",
}


def refuse_unimplemented_gamma_controls(options, *, driver: str) -> None:
    """Raise when ``options`` asks an exact-Gamma driver for a control it
    does not implement.

    These drivers used to ignore such a request and return the unshifted,
    unmixed, zero-temperature answer. The k-mesh drivers at
    ``kmesh=(1, 1, 1)`` implement all three controls on the same
    Hamiltonian.
    """
    if driver not in _IMPLEMENTED_CONTROLS:
        raise ValueError(f"unknown exact-Gamma GDF driver {driver!r}")
    missing = [
        name
        for name in _requested_controls(
            getattr(options, "smearing_temperature", 0.0),
            getattr(options, "fock_mixing", 0.0),
            getattr(options, "level_shift", 0.0),
        )
        if name not in _IMPLEMENTED_CONTROLS[driver]
    ]
    if not missing:
        return
    words = ", ".join(_CONTROL_WORDS[m] for m in missing)
    if driver == "run_pbc_gdf_rhf":
        raise NotImplementedError(
            f"{driver}: {words} is not implemented by this exact-Gamma "
            "driver. Call run_krhf_periodic_gdf(..., kmesh=(1, 1, 1)), which "
            "runs the same Hamiltonian and implements it, or clear the option."
        )
    # The open-shell k-mesh driver reads smearing only, so it is not the
    # place to send a level shift: nothing on the GDF routes implements one.
    raise NotImplementedError(
        f"{driver}: {words} is not implemented by this exact-Gamma driver, "
        "nor by run_kuhf_periodic_gdf(..., kmesh=(1, 1, 1)), which runs the "
        "same Hamiltonian. Clear the option, or use jk_method='bipole' for a "
        "shifted open-shell periodic SCF."
    )


def exact_gamma_gdf_blockers(
    *,
    driver: str,
    dim: int,
    n_electrons: int,
    multiplicity: int,
    nuclear_charge: float,
    ecp_active: bool,
    smearing_temperature: float = 0.0,
    fock_mixing: float = 0.0,
    level_shift: float = 0.0,
) -> Tuple[str, ...]:
    """Return why ``driver`` cannot evaluate a closed-shell Gamma GDF job.

    An empty tuple means the exact-Gamma driver implements the job
    completely and gives the energy of the k-mesh engine at ``(1, 1, 1)``.
    Otherwise each entry names one reason, in words fit for a log line.
    The drivers refuse charged cells and effective-core potentials, and
    fill a single closed shell.
    """
    if driver not in _CLOSED_SHELL_ROUTE_CONTROLS:
        raise ValueError(f"unknown closed-shell exact-Gamma GDF driver {driver!r}")
    controls = _CLOSED_SHELL_ROUTE_CONTROLS[driver]
    reasons = []
    if int(dim) != 3:
        reasons.append(f"dim={int(dim)}")
    if ecp_active:
        reasons.append("effective-core potentials")
    if int(n_electrons) % 2 != 0 or int(multiplicity) != 1:
        reasons.append("not a closed-shell singlet")
    if abs(float(nuclear_charge) - int(n_electrons)) > 0.5:
        reasons.append("charged cell")
    reasons.extend(
        _CONTROL_WORDS[name]
        for name in _requested_controls(
            smearing_temperature, fock_mixing, level_shift,
        )
        if name not in controls
    )
    return tuple(reasons)


def reject_gdf_fock_symmetry(
    *,
    dim: int,
    symmetry_stabilize: bool,
    symmetry_reduce_fock: object,
    where: str,
) -> None:
    """Refuse Fock symmetrisation on a bulk GDF calculation.

    No bulk GDF engine implements ``symmetry_stabilize`` or an explicit
    ``symmetry_reduce_fock=True``. Before, a Gamma job took the legacy
    Ewald-J driver to honour them, which changed the Hamiltonian, and a
    k-mesh job dropped them without a word. ``symmetry_reduce_k`` is the
    GDF symmetry saving.
    """
    if int(dim) != 3:
        return
    if not (bool(symmetry_stabilize) or symmetry_reduce_fock is True):
        return
    raise NotImplementedError(
        f"{where}: symmetry_stabilize and symmetry_reduce_fock=True are not "
        "implemented by the bulk GDF engines. Before, a Gamma job ran the "
        "legacy Ewald-J/real-space-exchange driver to honour them, a "
        "different Hamiltonian from GDF, and a k-mesh job ignored them. Use "
        "symmetry_reduce_k=True for the space-group saving on the GDF "
        "exchange build, or jk_method='bipole' for Fock symmetrisation."
    )


# --- one decision for the engine ------------------------------------------------

#: Engines that evaluate a GDF job. Bulk GDF has one Hamiltonian; these are
#: its evaluators.
EXACT_GAMMA = "exact-gamma"
K_MESH = "k-mesh"
LEGACY_GAMMA = "legacy-gamma"
SLAB_GDF = "slab-gdf"
EXTERNAL_XC_K_MESH = "external-xc-k-mesh"
#: A method the tables do not cover; the decision records it and asserts
#: nothing about the engine.
UNRESOLVED = "unresolved"

#: Backend labels each engine may report. The k-mesh driver delegates a
#: single Gamma point of the ``mdf``/``compcell`` fits to the exact-Gamma
#: driver itself and reports the adapter label for it.
_ENGINE_BACKEND_PREFIXES = {
    EXACT_GAMMA: ("pbc-gdf-",),
    K_MESH: ("native-multi-k-gdf", "native-gamma-gdf-via-k-gdf"),
    EXTERNAL_XC_K_MESH: ("native-multi-k-gdf",),
    SLAB_GDF: ("native-multik-slab-truncated-gdf",),
    LEGACY_GAMMA: ("ewald-jk-fallback", "native-gamma-gdf"),
}

_K_DRIVERS = {
    "RHF": "run_krhf_periodic_gdf",
    "RKS": "run_krks_periodic_gdf",
    "UHF": "run_kuhf_periodic_gdf",
    "UKS": "run_kuks_periodic_gdf",
    "ROHF": "run_krohf_periodic_gdf",
}
_EXACT_GAMMA_DRIVERS = {
    "RHF": "run_pbc_gdf_rhf",
    "RKS": "run_pbc_gdf_uks",  # its singlet limit
    "UHF": "run_pbc_gdf_uhf",
    "UKS": "run_pbc_gdf_uks",
}
#: SCF controls each driver implements. An AUTO-resolved Fock mixing or
#: level shift the chosen driver lacks is dropped (and the strategy block
#: says so); an explicit one is refused by the driver or moves a bulk Gamma
#: job to the k-mesh engine. The open-shell k-mesh loop reads smearing only.
_DRIVER_CONTROLS = {
    "run_pbc_gdf_rhf": frozenset(),
    "run_pbc_gdf_uhf": _IMPLEMENTED_CONTROLS["run_pbc_gdf_uhf"],
    "run_pbc_gdf_uks": _IMPLEMENTED_CONTROLS["run_pbc_gdf_uks"],
    "run_krhf_periodic_gdf": frozenset({"smearing", "fock_mixing", "level_shift"}),
    "run_krks_periodic_gdf": frozenset({"smearing", "fock_mixing", "level_shift"}),
    "run_kuhf_periodic_gdf": frozenset({"smearing"}),
    "run_kuks_periodic_gdf": frozenset({"smearing"}),
    "run_krohf_periodic_gdf": frozenset(),
    # The dim < 3 driver takes the options as they come; nothing is filtered.
    "run_rhf_periodic_gamma_gdf": frozenset({"smearing", "fock_mixing", "level_shift"}),
}
#: Why an AUTO-resolved knob is dropped, per driver; the control's name
#: follows. The strategy block in the .out carries these words.
_ZERO_REASON = {
    "run_pbc_gdf_rhf": (
        "the default-Gamma closed-shell RHF run stays on the exact-Gamma GDF "
        "driver run_pbc_gdf_rhf, which does not implement"
    ),
    "run_pbc_gdf_uks": (
        "the default-Gamma run stays on the exact-Gamma GDF driver "
        "run_pbc_gdf_uks, which does not implement"
    ),
    "run_pbc_gdf_uhf": (
        "the selected open-shell GDF driver run_pbc_gdf_uhf, like all "
        "open-shell GDF drivers, does not implement"
    ),
    "run_kuhf_periodic_gdf": (
        "the selected open-shell GDF driver run_kuhf_periodic_gdf, like all "
        "open-shell GDF drivers, does not implement"
    ),
    "run_kuks_periodic_gdf": (
        "the selected open-shell GDF driver run_kuks_periodic_gdf, like all "
        "open-shell GDF drivers, does not implement"
    ),
    "run_krohf_periodic_gdf": (
        "the selected open-shell GDF driver run_krohf_periodic_gdf, like all "
        "open-shell GDF drivers, does not implement"
    ),
}
_KNOB_OPTION = {
    "smearing": "smearing_temperature",
    "fock_mixing": "fock_mixing",
    "level_shift": "level_shift",
}


@dataclass(frozen=True)
class GDFRouteDecision:
    """Which engine evaluates a GDF job, and what that means for its controls.

    :func:`resolve_gdf_route` computes it once from the request; the
    runner's convergence filter, its dispatch and its manifest all read this
    one object, so no two of them can disagree about the route, and
    :func:`check_executed_engine` makes the finished result agree with it.
    """

    engine: str
    driver: str
    kmesh: Optional[Tuple[int, int, int]]
    gdf_method: str
    #: Why the exact-Gamma driver does not serve this job (empty when it does).
    reasons: Tuple[str, ...]
    #: Controls the chosen driver implements.
    honoured: FrozenSet[str]
    #: ``(option name, reason)`` for each AUTO-resolved knob the chosen driver
    #: lacks. Explicit knobs are never dropped.
    auto_zeroed: Tuple[Tuple[str, str], ...]
    #: The exchange q=0 convention the engine declares, when the decision
    #: knows it.
    exchange_q0: Optional[str]

    def describe(self) -> str:
        text = f"{self.engine} ({self.driver}"
        if self.kmesh is not None:
            text += f", kmesh={tuple(int(n) for n in self.kmesh)}"
        text += f", gdf_method={self.gdf_method})"
        if self.reasons:
            text += "; exact-Gamma driver not used: " + ", ".join(self.reasons)
        return text


def executed_engine(backend) -> Optional[str]:
    """The engine a finished result's ``backend`` label reports, or ``None``."""
    label = str(backend or "")
    for engine, prefixes in _ENGINE_BACKEND_PREFIXES.items():
        if any(label.startswith(p) for p in prefixes):
            return engine
    return None


def check_executed_engine(decision: GDFRouteDecision, backend, *, where: str) -> str:
    """Fail closed when the driver that ran is not the one the decision chose."""
    engine = executed_engine(backend)
    if decision.engine == UNRESOLVED:
        return engine or UNRESOLVED
    if engine is None or engine != decision.engine:
        raise RuntimeError(
            f"{where}: internal GDF route invariant violated: the decision "
            f"chose {decision.describe()} but the result reports backend "
            f"{backend!r}"
        )
    return engine


def manifest_fields(decision: Optional[GDFRouteDecision], backend) -> dict:
    """``[run]`` manifest fields recording the decision and what ran."""
    if decision is None:
        return {}
    return {
        "gdf_engine": executed_engine(backend) or "unknown",
        "gdf_engine_planned": decision.engine,
        "gdf_driver": decision.driver,
        "gdf_route_reason": (
            "; ".join(decision.reasons)
            or "the exact-Gamma driver implements every requested control"
        ),
    }


def _knob_values(knobs: Mapping[str, object]):
    """``(value, source)`` per control from ``ConvergenceStrategy.knobs``."""
    out = {}
    for name, option in _KNOB_OPTION.items():
        k = knobs.get(option)
        value = float(getattr(k, "value", k if k is not None else 0.0) or 0.0)
        source = str(getattr(k, "source", "explicit"))
        out[name] = (value, source)
    return out


def resolve_gdf_route(
    *,
    method: str,
    dim: int,
    n_electrons: int,
    multiplicity: int,
    nuclear_charge: float,
    ecp_active: bool,
    kpoints: Optional[Tuple[int, int, int]],
    gdf_method: Optional[str],
    uses_external_xc: bool,
    knobs: Mapping[str, object],
    symmetry_stabilize: bool = False,
    symmetry_reduce_fock: object = None,
) -> GDFRouteDecision:
    """Decide the engine for ``run_periodic_job(jk_method="gdf")``.

    ``kpoints`` is the requested mesh, ``None`` for the default Gamma job.
    ``knobs`` is ``ConvergenceStrategy.knobs`` before the GDF capability
    filter: value and source per control. A bulk closed-shell Gamma job runs
    the exact-Gamma driver when that driver implements every explicitly
    requested control (smearing counts whatever its source: it changes the
    occupations, so it always moves the engine); otherwise it runs the
    k-mesh engine at ``(1, 1, 1)``. An AUTO-resolved Fock mixing or level
    shift never moves the engine: it is dropped when the chosen driver lacks
    it and kept when the driver implements it.
    """
    method_upper = str(method).upper()
    builder = gdf_method or "rsgdf"
    values = _knob_values(knobs)
    if method_upper not in _K_DRIVERS:
        # A method without a GDF route table (the runner refuses or routes
        # it itself): record that, drop nothing, and let the invariant pass.
        return GDFRouteDecision(
            engine=UNRESOLVED,
            driver="unknown",
            kmesh=None if kpoints is None else tuple(int(n) for n in kpoints),
            gdf_method=builder,
            reasons=(f"no GDF route table for method {method_upper}",),
            honoured=frozenset({"smearing", "fock_mixing", "level_shift"}),
            auto_zeroed=(),
            exchange_q0=None,
        )
    reject_gdf_fock_symmetry(
        dim=dim,
        symmetry_stabilize=symmetry_stabilize,
        symmetry_reduce_fock=symmetry_reduce_fock,
        where="run_periodic_job",
    )

    def _auto(name):
        value, source = values[name]
        return value != 0.0 and source == "auto"

    def _explicit_value(name):
        value, source = values[name]
        return 0.0 if source == "auto" else value

    def _decision(engine, driver, kmesh, reasons, exchange_q0):
        honoured = _DRIVER_CONTROLS[driver]
        zeroed = tuple(
            (
                _KNOB_OPTION[name],
                f"capability-filtered to zero: {_ZERO_REASON[driver]} "
                f"{_CONTROL_WORDS[name]}",
            )
            for name in ("fock_mixing", "level_shift")
            if _auto(name) and name not in honoured
        )
        return GDFRouteDecision(
            engine=engine,
            driver=driver,
            kmesh=None if kmesh is None else tuple(int(n) for n in kmesh),
            gdf_method=builder,
            reasons=tuple(reasons),
            honoured=honoured,
            auto_zeroed=zeroed,
            exchange_q0=exchange_q0,
        )

    bulk_gauge = BVK_EWALD if int(dim) == 3 else None
    if uses_external_xc:
        return _decision(
            EXTERNAL_XC_K_MESH, _K_DRIVERS[method_upper],
            kpoints or (1, 1, 1), ("external XC provider",), bulk_gauge,
        )
    if int(dim) == 2 and method_upper in ("RHF", "RKS"):
        return _decision(
            SLAB_GDF, _K_DRIVERS[method_upper], kpoints or (1, 1, 1),
            ("dim=2 slab",), None,
        )
    if kpoints is not None:
        return _decision(
            K_MESH, _K_DRIVERS[method_upper], kpoints, ("explicit k mesh",),
            bulk_gauge,
        )
    if ecp_active and method_upper in ("RHF", "RKS", "UHF", "UKS"):
        return _decision(
            K_MESH, _K_DRIVERS[method_upper], (1, 1, 1),
            ("effective-core potentials",), bulk_gauge,
        )
    if method_upper == "ROHF":
        return _decision(
            K_MESH, _K_DRIVERS["ROHF"], (1, 1, 1),
            ("spin-restricted open shell: one k-mesh engine",), bulk_gauge,
        )
    if method_upper in ("UHF", "UKS"):
        return _decision(
            EXACT_GAMMA, _EXACT_GAMMA_DRIVERS[method_upper], None, (),
            bulk_gauge,
        )
    driver = _EXACT_GAMMA_DRIVERS[method_upper]
    blockers = exact_gamma_gdf_blockers(
        driver=driver,
        dim=dim,
        n_electrons=n_electrons,
        multiplicity=multiplicity,
        nuclear_charge=nuclear_charge,
        ecp_active=ecp_active,
        smearing_temperature=values["smearing"][0],
        fock_mixing=_explicit_value("fock_mixing"),
        level_shift=_explicit_value("level_shift"),
    )
    if int(dim) == 3 and not blockers:
        return _decision(EXACT_GAMMA, driver, None, (), BVK_EWALD)
    if int(dim) == 3:
        return _decision(K_MESH, _K_DRIVERS[method_upper], (1, 1, 1), blockers, BVK_EWALD)
    if gdf_method is not None:
        # dim < 3 with an explicit builder keeps the historical exact-Gamma
        # call (the runner refuses it for RKS); no gauge is claimed for it.
        return _decision(
            EXACT_GAMMA, driver, None, (f"dim={int(dim)} with an explicit gdf_method",), None,
        )
    return _decision(
        LEGACY_GAMMA, "run_rhf_periodic_gamma_gdf", None, (f"dim={int(dim)}",),
        MOLECULAR_LIMIT,
    )


def refuse_unimplemented_k_controls(options, *, driver: str) -> None:
    """Raise when an open-shell k-mesh driver is asked for a control it drops.

    ``run_kuhf_periodic_gdf`` reads smearing only. The runner refuses these
    requests before dispatch; a direct call used to run and drop them.
    """
    if driver not in ("run_kuhf_periodic_gdf", "run_kuks_periodic_gdf"):
        raise ValueError(f"unknown open-shell k-mesh GDF driver {driver!r}")
    missing = [
        name
        for name in _requested_controls(
            getattr(options, "smearing_temperature", 0.0),
            getattr(options, "fock_mixing", 0.0),
            getattr(options, "level_shift", 0.0),
        )
        if name not in _DRIVER_CONTROLS[driver]
    ]
    if not missing:
        return
    raise NotImplementedError(
        f"{driver}: {', '.join(_CONTROL_WORDS[m] for m in missing)} is not "
        "implemented by the open-shell k-mesh GDF driver, nor by the "
        "exact-Gamma run_pbc_gdf_uhf/uks drivers for a level shift. Clear "
        "the option, or use jk_method='bipole' for a shifted or mixed "
        "open-shell periodic SCF."
    )
