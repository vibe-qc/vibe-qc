"""Published DLPNO threshold presets and route-capability resolution.

The published LoosePNO, NormalPNO, and TightPNO names describe the three
``(TCutPairs, TCutPNO, TCutMKN)`` triples in Liakos et al., JCTC 11, 1525
(2015), Table 1.  Not every vibe-qc DLPNO solver implements every member of
that triple.  This module therefore applies a named preset only to real
algorithmic consumers and returns provenance that makes unsupported or
currently inactive members explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import TypeVar


_PUBLISHED_FIELDS = ("tcut_pairs", "tcut_pno", "tcut_mkn")
_DERIVED_MP2_FIELDS = ("tcut_pairs_weak", "tcut_pno_weak")


@dataclass(frozen=True)
class DLPNOThresholds:
    """A DLPNO threshold record.

    ``tcut_pairs``, ``tcut_pno``, and ``tcut_mkn`` are the three published
    preset coordinates. The two stored weak-pair values are vibe-qc
    companions: the weak-pair PNO cutoff is one decade looser than the
    strong-pair value, and the weak-pair energy threshold equals the
    strong-pair value. That equal energy boundary makes the weak tier
    structurally inactive for named presets because screening happens first;
    a custom ``tcut_pairs_weak > tcut_pairs`` interval activates it.
    ``tcut_c``, ``tcut_do``, and ``tcut_ext`` are not part of the published
    three-coordinate presets and are not applied to solver option objects.
    """

    tcut_c: float = 1e-3
    tcut_pairs: float = 1e-4
    tcut_pairs_weak: float = 1e-4
    tcut_mkn: float = 1e-3
    tcut_pno: float = 3.33e-7
    tcut_pno_weak: float = 3.33e-6
    tcut_do: float = 1e-2
    tcut_ext: float = 1e-3
    name: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "tcut_c",
            "tcut_pairs",
            "tcut_pairs_weak",
            "tcut_mkn",
            "tcut_pno",
            "tcut_pno_weak",
            "tcut_do",
            "tcut_ext",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if self.name is not None:
            published = {
                "LoosePNO": (1e-3, 1e-6, 1e-3),
                "NormalPNO": (1e-4, 3.33e-7, 1e-3),
                "TightPNO": (1e-5, 1e-7, 1e-4),
            }
            if self.name not in published:
                raise ValueError(
                    "named DLPNOThresholds records must be LoosePNO, "
                    "NormalPNO, or TightPNO"
                )
            triple = tuple(float(getattr(self, name)) for name in _PUBLISHED_FIELDS)
            if triple != published[self.name]:
                raise ValueError(
                    f"{self.name} must carry its published threshold triple"
                )
            expected_weak = (
                published[self.name][0],
                published[self.name][1] * 10.0,
            )
            actual_weak = (
                float(self.tcut_pairs_weak),
                float(self.tcut_pno_weak),
            )
            if any(
                not isclose(actual, expected, rel_tol=1e-15, abs_tol=0.0)
                for actual, expected in zip(actual_weak, expected_weak)
            ):
                raise ValueError(
                    f"{self.name} must carry vibe-qc's derived weak-pair "
                    "companions"
                )


# A preset carrying a published name must carry the published numbers.
# These triples are Table 1 of Liakos, Sparta, Kesharwani, Martin & Neese,
# J. Chem. Theory Comput. 11, 1525 (2015), doi:10.1021/ct501129s. NormalPNO
# also equals the original DLPNO-CCSD defaults of Riplinger & Neese,
# J. Chem. Phys. 138, 034106 (2013), doi:10.1063/1.4773581.
#
# The weak-pair companions are a documented vibe-qc convention, not values
# from that table. The other record fields stay available for compatibility,
# but a named preset never manufactures matching attributes on a solver.
DLPNO_DEFAULTS = DLPNOThresholds(name="NormalPNO")
DLPNO_TIGHT = DLPNOThresholds(
    tcut_pno=1e-7,
    tcut_pno_weak=1e-6,
    tcut_pairs=1e-5,
    tcut_pairs_weak=1e-5,
    tcut_mkn=1e-4,
    name="TightPNO",
)
DLPNO_LOOSE = DLPNOThresholds(
    tcut_pno=1e-6,
    tcut_pno_weak=1e-5,
    tcut_pairs=1e-3,
    tcut_pairs_weak=1e-3,
    name="LoosePNO",
)


_PRESETS = (DLPNO_LOOSE, DLPNO_DEFAULTS, DLPNO_TIGHT)
_PRESETS_BY_KEY = {
    "loose": DLPNO_LOOSE,
    "loosepno": DLPNO_LOOSE,
    "normal": DLPNO_DEFAULTS,
    "normalpno": DLPNO_DEFAULTS,
    "default": DLPNO_DEFAULTS,
    "tight": DLPNO_TIGHT,
    "tightpno": DLPNO_TIGHT,
}


@dataclass(frozen=True)
class DLPNOThresholdProvenance:
    """Immutable account of how a threshold convention met one route.

    ``requested`` is the complete three-coordinate request when it came
    through a named/custom record or the public factory.  A bare, unmarked
    custom partial-capability option can report only the coordinates it
    actually carries; ``unsupported`` makes that limit explicit.
    ``applied`` contains every option value that participates in the selected
    route mode. Stored-but-unreachable weak-pair companions are excluded.
    ``unsupported`` names coordinates with no algorithmic consumer on the
    route. ``inactive`` names implemented coordinates that do not affect the
    selected route mode (UMP2 pair screening with canonical occupieds, and
    the closed-shell MP2 weak tier when its boundary is not above screening).
    """

    preset: str
    requested: tuple[tuple[str, float], ...]
    applied: tuple[tuple[str, float], ...]
    unsupported: tuple[str, ...]
    inactive: tuple[str, ...]


@dataclass(frozen=True)
class _DLPNOThresholdMarker:
    """Private request record retained on a mutable route-options object.

    Partial-capability routes cannot reconstruct a complete three-coordinate
    request from their active option fields.  The snapshot distinguishes an
    intentionally inactive coordinate from a later explicit mutation, so
    :func:`describe_dlpno_thresholds` never invents provenance.
    """

    provenance: DLPNOThresholdProvenance
    option_snapshot: tuple[tuple[str, float], ...]


def _normalise_preset_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch not in " _-")


def resolve_dlpno_thresholds(
    spec: str | DLPNOThresholds | None = "normal",
) -> DLPNOThresholds:
    """Resolve a public preset spelling to its threshold record.

    Passing a :class:`DLPNOThresholds` instance preserves an explicit custom
    record. ``None`` is accepted as an alias for the project default,
    NormalPNO.
    """

    if spec is None:
        return DLPNO_DEFAULTS
    if isinstance(spec, DLPNOThresholds):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            "DLPNO threshold preset must be a name or DLPNOThresholds record"
        )
    key = _normalise_preset_name(spec)
    try:
        return _PRESETS_BY_KEY[key]
    except KeyError as exc:
        choices = "LoosePNO, NormalPNO, TightPNO"
        raise ValueError(
            f"unknown DLPNO threshold preset {spec!r}; choose {choices}"
        ) from exc


def _route_capabilities(options: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(implemented, inactive)`` threshold fields for ``options``."""

    # Imports stay local so this policy module can be imported before the
    # solver modules by vibeqc.dlpno.__init__ without creating a cycle.
    from .ccsd import DLPNOCCSDPilotOptions
    from .ccsd_local_solver import LocalCCSDOptions
    from .mp2 import DLPNOMP2Options
    from .uccsd import DLPNOUCCSDPilotOptions
    from .uccsd_local_solver import LocalUCCSDOptions
    from .ump2 import DLPNOUMP2Options

    inactive: tuple[str, ...] = ()
    if isinstance(options, DLPNOMP2Options):
        implemented = _PUBLISHED_FIELDS + _DERIVED_MP2_FIELDS
        if float(options.tcut_pairs_weak) <= float(options.tcut_pairs):
            inactive = _DERIVED_MP2_FIELDS
    elif isinstance(options, DLPNOUMP2Options):
        implemented = ("tcut_pairs", "tcut_pno")
        if options.localise == "none":
            inactive = ("tcut_pairs",)
    elif isinstance(options, DLPNOCCSDPilotOptions):
        implemented = ("tcut_pno", "tcut_mkn")
    elif isinstance(options, DLPNOUCCSDPilotOptions):
        implemented = ("tcut_pno",)
    elif isinstance(options, LocalCCSDOptions):
        implemented = _PUBLISHED_FIELDS
    elif isinstance(options, LocalUCCSDOptions):
        implemented = _PUBLISHED_FIELDS
    else:
        names = (
            "DLPNOMP2Options, DLPNOUMP2Options, DLPNOCCSDPilotOptions, "
            "DLPNOUCCSDPilotOptions, LocalCCSDOptions, or LocalUCCSDOptions"
        )
        raise TypeError(f"unsupported DLPNO options type; expected {names}")
    return implemented, inactive


def _preset_name(thresholds: DLPNOThresholds) -> str:
    return thresholds.name or "custom"


def _requested_values(
    thresholds: DLPNOThresholds,
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (field_name, float(getattr(thresholds, field_name)))
        for field_name in _PUBLISHED_FIELDS
    )


def _unsupported_values(implemented: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in _PUBLISHED_FIELDS if name not in implemented)


def _option_snapshot(
    options: object,
    implemented: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (field_name, float(getattr(options, field_name)))
        for field_name in implemented
    )


def _remember_provenance(
    options: object,
    provenance: DLPNOThresholdProvenance,
    implemented: tuple[str, ...],
) -> None:
    options._dlpno_threshold_marker = _DLPNOThresholdMarker(
        provenance=provenance,
        option_snapshot=_option_snapshot(options, implemented),
    )


def _observed_applied(
    options: object,
    implemented: tuple[str, ...],
    inactive: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    return tuple(
        (field_name, float(getattr(options, field_name)))
        for field_name in implemented
        if field_name not in inactive
    )


def apply_dlpno_thresholds(
    options: object,
    preset: str | DLPNOThresholds | None = "normal",
) -> DLPNOThresholdProvenance:
    """Apply one threshold convention to the consumers on ``options``.

    The option object is mutated in place. Missing route capabilities are not
    added as attributes. Inactive published coordinates are not changed;
    derived MP2 companions are stored so a later explicit custom interval can
    activate them, but are omitted from ``applied`` while unreachable. The
    returned provenance is the complete, immutable disclosure record.
    """

    thresholds = resolve_dlpno_thresholds(preset)
    implemented, inactive = _route_capabilities(options)
    for field_name in implemented:
        if field_name in inactive and field_name not in _DERIVED_MP2_FIELDS:
            continue
        value = float(getattr(thresholds, field_name))
        setattr(options, field_name, value)
    # Applying a named MP2 record can itself close a previously custom weak
    # interval, so classify inactivity from the final stored values.
    implemented, inactive = _route_capabilities(options)
    provenance = DLPNOThresholdProvenance(
        preset=_preset_name(thresholds),
        requested=_requested_values(thresholds),
        applied=_observed_applied(options, implemented, inactive),
        unsupported=_unsupported_values(implemented),
        inactive=inactive,
    )
    _remember_provenance(options, provenance, implemented)
    return provenance


def _matches_preset(
    options: object,
    preset: DLPNOThresholds,
    implemented: tuple[str, ...],
) -> bool:
    return all(
        float(getattr(options, field_name))
        == float(getattr(preset, field_name))
        for field_name in implemented
    )


def describe_dlpno_thresholds(options: object) -> DLPNOThresholdProvenance:
    """Describe an option object's effective convention without mutation."""

    implemented, inactive = _route_capabilities(options)
    marker = getattr(options, "_dlpno_threshold_marker", None)
    if isinstance(marker, _DLPNOThresholdMarker):
        current_snapshot = _option_snapshot(options, implemented)
        previous_snapshot = dict(marker.option_snapshot)
        changed = [
            field_name
            for field_name, value in current_snapshot
            if field_name in previous_snapshot
            and value != previous_snapshot[field_name]
        ]
        capabilities_changed = (
            marker.provenance.unsupported != _unsupported_values(implemented)
            or marker.provenance.inactive != inactive
        )
        requested_before = dict(marker.provenance.requested)
        current = dict(current_snapshot)
        newly_active_mismatch = [
            field_name
            for field_name in marker.provenance.inactive
            if field_name not in inactive
            and field_name in requested_before
            and float(getattr(options, field_name))
            != float(requested_before[field_name])
        ]
        # An inactive coordinate is intentionally left at the option class's
        # stored value when a preset is applied. If the caller later enables
        # that algorithmic consumer and sets the coordinate to the original
        # named request, the mutation completes the request; it does not turn
        # TightPNO (for example) into a custom convention.
        newly_active_matches = {
            field_name
            for field_name in changed
            if field_name in marker.provenance.inactive
            and field_name not in inactive
            and field_name in requested_before
            and current[field_name] == float(requested_before[field_name])
        }
        custom_changes = [
            field_name
            for field_name in changed
            if field_name not in newly_active_matches
        ]
        if not changed and not capabilities_changed:
            return marker.provenance

        requested = requested_before
        for field_name in changed:
            if field_name in _PUBLISHED_FIELDS:
                requested[field_name] = current[field_name]
        return DLPNOThresholdProvenance(
            preset=(
                "custom"
                if custom_changes or newly_active_mismatch
                else marker.provenance.preset
            ),
            requested=tuple(
                (field_name, float(requested[field_name]))
                for field_name in _PUBLISHED_FIELDS
            ),
            applied=_observed_applied(options, implemented, inactive),
            unsupported=_unsupported_values(implemented),
            inactive=inactive,
        )

    # An unmarked object with every published coordinate can identify any
    # preset from its values. A partial-capability object cannot: a lone
    # TightPNO-looking TCutPNO, for example, is not evidence that the caller
    # also requested TightPNO's unsupported TCutPairs and TCutMKN values.
    # The one safe partial match is the route class's own NormalPNO default;
    # those defaults are deliberately pinned as the project policy.
    candidates = (
        _PRESETS
        if all(name in implemented for name in _PUBLISHED_FIELDS)
        else (DLPNO_DEFAULTS,)
    )
    matched = next(
        (
            preset
            for preset in candidates
            if _matches_preset(options, preset, implemented)
        ),
        None,
    )
    if matched is None:
        requested = tuple(
            (name, float(getattr(options, name)))
            for name in _PUBLISHED_FIELDS
            if name in implemented
        )
        preset_name = "custom"
    else:
        requested = _requested_values(matched)
        preset_name = _preset_name(matched)
    observed = _observed_applied(options, implemented, inactive)
    return DLPNOThresholdProvenance(
        preset=preset_name,
        requested=requested,
        applied=observed,
        unsupported=_unsupported_values(implemented),
        inactive=inactive,
    )


_OptionsT = TypeVar("_OptionsT")


def options_from_dlpno_thresholds(
    options_type: type[_OptionsT],
    preset: str | DLPNOThresholds | None = "normal",
    **overrides: object,
) -> _OptionsT:
    """Construct route options and apply a named convention.

    Constructor overrides are installed first so mode-dependent capabilities
    such as UMP2 pair screening are resolved correctly. Explicit threshold
    overrides are then restored after the preset, so the usual
    ``preset + deliberate exception`` factory pattern remains available and
    :func:`describe_dlpno_thresholds` will label the result ``custom``.
    """

    explicit_thresholds = {
        name: value
        for name, value in overrides.items()
        if name in _PUBLISHED_FIELDS + _DERIVED_MP2_FIELDS
    }
    options = options_type(**overrides)
    provenance = apply_dlpno_thresholds(options, preset)
    for name, value in explicit_thresholds.items():
        setattr(options, name, value)
    thresholds = resolve_dlpno_thresholds(preset)
    has_exception = any(
        float(value) != float(getattr(thresholds, name))
        for name, value in explicit_thresholds.items()
    )
    if has_exception:
        requested = dict(provenance.requested)
        for name, value in explicit_thresholds.items():
            if name in _PUBLISHED_FIELDS:
                requested[name] = float(value)
        implemented, inactive = _route_capabilities(options)
        provenance = DLPNOThresholdProvenance(
            preset="custom",
            requested=tuple(
                (name, float(requested[name])) for name in _PUBLISHED_FIELDS
            ),
            applied=_observed_applied(options, implemented, inactive),
            unsupported=_unsupported_values(implemented),
            inactive=inactive,
        )
        _remember_provenance(options, provenance, implemented)
    else:
        implemented, _ = _route_capabilities(options)
        _remember_provenance(options, provenance, implemented)
    return options


__all__ = [
    "DLPNOThresholds",
    "DLPNOThresholdProvenance",
    "DLPNO_DEFAULTS",
    "DLPNO_TIGHT",
    "DLPNO_LOOSE",
    "resolve_dlpno_thresholds",
    "apply_dlpno_thresholds",
    "describe_dlpno_thresholds",
    "options_from_dlpno_thresholds",
]
