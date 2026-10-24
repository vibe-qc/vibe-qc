"""Engine-owned JSON option records for executable QVF containers.

Only constructors named here are reachable. The input cannot name a Python
module/class. Ordinary runner mappings keep their existing consumer contract.
"""
from __future__ import annotations

import dataclasses
import importlib
import inspect
import math
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

# All explicitly option-class-typed arguments on the two public runners.
_OPTION_TYPES = {
    "rhf_options": ("_vibeqc_core", "RHFOptions"),
    "uhf_options": ("_vibeqc_core", "UHFOptions"),
    "rks_options": ("_vibeqc_core", "RKSOptions"),
    "uks_options": ("_vibeqc_core", "UKSOptions"),
    "mp2_options": ("_vibeqc_core", "MP2Options"),
    "ump2_options": ("_vibeqc_core", "UMP2Options"),
    "rohf_options": ("rohf", "ROHFOptions"),
    "roks_options": ("roks", "ROKSOptions"),
    "cisd_options": ("solvers", "CISDOptions"),
    "cc3_options": ("solvers", "CC3Options"),
    "ccsdt_options": ("solvers", "CCSDTOptions"),
    "selected_ci_options": ("solvers", "SelectedCIOptions"),
    "dmrg_options": ("solvers", "DMRGOptions"),
    "v2rdm_options": ("solvers", "V2RDMOptions"),
    "transcorrelated_options": ("solvers", "TranscorrelatedOptions"),
    "casci_options": ("solvers", "CASCIOptions"),
    "caspt2_options": ("solvers", "CASPT2Options"),
    "nevpt2_options": ("solvers", "NEVPT2Options"),
    "casscf_options": ("solvers", "CASSCFOptions"),
    "ccsd_options": ("cc", "CCSDOptions"),
    "dlpno_options": ("dlpno.mp2", "DLPNOMP2Options"),
    # The local PNO-space solver run_job uses for dlpno-ccsd / dlpno-ccsd(t)
    # (#558); the O(N^6) DLPNOCCSDPilotOptions stays a Python-API opt-in.
    "dlpno_ccsd_options": ("dlpno.ccsd_local_solver", "LocalCCSDOptions"),
    "mlip_options": ("mlip", "MLIPOptions"),
    "ccm_options": ("semiempirical.methods.msindo_ccm", "CCMOptions"),
    "thermo_options": ("thermo", "ThermoOptions"),
    "smearing": ("smearing", "SmearingOptions"),
    "opentrustregion_options": ("_vibeqc_core", "OpenTrustRegionOptions"),
}
_NATIVE_RECORDS = frozenset({
    "RHFOptions", "UHFOptions", "RKSOptions", "UKSOptions",
    "MP2Options", "UMP2Options", "GridOptions", "DavidsonOptions",
    "SOSCFOptions", "TRAHOptions", "NewtonOptions", "SCFRestartOptions",
    "OpenTrustRegionOptions",
})
# Empty native vectors carry no element type at runtime. These encodings are
# declared from the corresponding public C++ option members, never guessed.
_VECTOR_TYPES = {
    "level_shift_schedule": list[float],
    "atomic_spins": list[int],
    "multi_guess_seeds": list[int],
    "ecp_effective_charges": list[float],
    "ecp_primitive_centers": list[tuple[float, float, float]],
    "dft_plus_u_ao_groups": list[list[int]],
    "orca_angular_points": list[int],
}
# Objects without a supported engine-specific JSON representation. Several of
# these have scalar shorthand forms, which are unchanged. Nonempty lists
# for these names require runtime objects, not JSON arrays; dft_plus_u=[] is
# the existing no-site exception. Solvent/geom_opt_options mappings and
# fragment atom-index lists have separate supported consumer contracts.
_OPAQUE_RECORDS = frozenset({
    "band_structure", "dft_plus_u", "read_from",
    "dispersion", "citype", "triples", "dlpno_thresholds",
})


def _load_type(module: str, name: str) -> type:
    return getattr(importlib.import_module("." + module, __package__), name)


def _fail(path: str, message: str):
    raise ValueError(f"QVF option {path}: {message}")


def _convert(value: Any, annotation: Any, path: str) -> Any:
    """Validate JSON types without bool-as-int or string-to-number coercion."""
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        choices = [a for a in args if a is not type(None)]
        if len(choices) == 1:
            return _convert(value, choices[0], path)
        for choice in choices:
            try:
                return _convert(value, choice, path)
            except ValueError:
                pass
        _fail(path, f"expected {annotation}")
    if annotation in (Any, object):
        if value is None:
            return None
        _fail(path, "this runtime object has no supported QVF JSON encoding")
    if annotation is type(None):
        if value is None:
            return None
        _fail(path, "expected null")
    if annotation is bool:
        if type(value) is bool:
            return value
        _fail(path, "expected a JSON boolean")
    if annotation is int:
        if type(value) is int:
            return value
        _fail(path, "expected a JSON integer (not a boolean)")
    if annotation is float:
        if type(value) in (int, float):
            try:
                number = float(value)
            except OverflowError:
                number = math.inf
            if math.isfinite(number):
                return number
        _fail(path, "expected a finite JSON number (not a boolean)")
    if annotation is str:
        if type(value) is str:
            return value
        _fail(path, "expected a JSON string")
    if origin is Literal:
        if any(type(value) is type(a) and value == a for a in args):
            return value
        _fail(path, f"expected one of {args!r}")
    if origin in (list, tuple):
        if not isinstance(value, list):
            _fail(path, "expected a JSON array")
        if origin is list:
            if not args:
                _fail(path, "array element encoding is not supported")
            return [_convert(v, args[0], f"{path}[{i}]")
                    for i, v in enumerate(value)]
        if len(args) == 2 and args[1] is Ellipsis:
            annotations = [args[0]] * len(value)
        else:
            if len(value) != len(args):
                _fail(path, f"expected an array of length {len(args)}")
            annotations = args
        return tuple(_convert(v, a, f"{path}[{i}]")
                     for i, (v, a) in enumerate(zip(value, annotations)))
    members = getattr(annotation, "__members__", None)
    if members is not None:
        if isinstance(value, str):
            matches = [member for name, member in members.items()
                       if name.casefold() == value.casefold()]
            if matches:
                return matches[0]
        _fail(path, f"expected an enum member name from {tuple(members)}")
    if isinstance(annotation, type) and (
        dataclasses.is_dataclass(annotation)
        or annotation.__name__ in _NATIVE_RECORDS
    ):
        return _record(value, annotation, path)
    _fail(path, "this field has no supported QVF JSON encoding")


def _field_type(cls: type, key: str, annotation: Any) -> Any:
    """Explicit encodings for Python fields declared as opaque objects."""
    name = cls.__name__
    if name in ("ROHFOptions", "ROKSOptions"):
        if key == "initial_guess":
            return str
        if key in ("grid", "davidson"):
            nested = "GridOptions" if key == "grid" else "DavidsonOptions"
            return _load_type("_vibeqc_core", nested) | None
        if key == "ecp_primitive_centers":
            return _VECTOR_TYPES[key]
    if name == "CASSCFOptions" and key == "selected_ci_options":
        return _load_type("solvers", "SelectedCIOptions") | None
    if name == "CCMOptions" and key == "translations":
        return list[tuple[float, float, float]] | None
    if name == "CCSDOptions" and key == "triples":
        return str | bool | None
    if name == "LocalCCSDOptions":
        # Plain-string fields whose solver reads an unlisted value as some
        # other choice (localise: anything but "boys" means no localisation)
        # or refuses it only once the run is set up. A record names one of
        # the values the molecular route implements (#558).
        from .dlpno.ccsd_local_solver import RESIDUAL_DOMAINS, TRIPLES_MODES
        from .dlpno.pao import DOMAIN_CRITERIA
        from .dlpno.pno_density import PNO_NORMS

        choices = {
            "triples_mode": TRIPLES_MODES, "localise": ("boys", "none"),
            "residual_domain": RESIDUAL_DOMAINS,
            "domain_criterion": DOMAIN_CRITERIA, "pno_norm": PNO_NORMS,
        }.get(key)
        if choices is not None:
            return Literal[choices]
    return annotation


def _record(value: Any, cls: type, path: str) -> Any:
    if not isinstance(value, dict):
        _fail(path, f"expected a JSON object for {cls.__name__}")
    if any(not isinstance(k, str) or k.startswith("_") for k in value):
        _fail(path, "only public named option fields are supported")
    if dataclasses.is_dataclass(cls):
        hints = get_type_hints(cls)
        fields = {f.name: _field_type(cls, f.name, hints[f.name])
                  for f in dataclasses.fields(cls) if f.init}
    elif cls.__name__ == "CCSDOptions":
        # Constructor semantics preserve frozen-core explicitness and triples
        # aliases; assigning every default field would change that provenance.
        hints = get_type_hints(cls.__init__)
        fields = {k: _field_type(cls, k, hints.get(k, Any))
                  for k in inspect.signature(cls.__init__).parameters
                  if k != "self"}
    else:
        instance = cls()
        for key, item in value.items():
            field_path = f"{path}.{key}"
            if not hasattr(instance, key) or callable(getattr(instance, key)):
                _fail(field_path, f"unknown field for {cls.__name__}")
            default = getattr(instance, key)
            if key in _VECTOR_TYPES:
                annotation = _VECTOR_TYPES[key]
            elif key == "n_frozen_core" and cls.__name__ in (
                "MP2Options", "UMP2Options"
            ):
                annotation = int | None
            elif isinstance(default, list):
                # No element contract for native object vectors yet. Empty
                # lists remain explicit requests to clear them.
                if item != []:
                    _fail(field_path, "nonempty native object arrays are not supported")
                annotation = list[Any]
            else:
                annotation = type(default)
            converted = _convert(item, annotation, field_path)
            try:
                setattr(instance, key, converted)
            except (TypeError, ValueError, AttributeError, OverflowError) as exc:
                _fail(field_path, str(exc))
        return instance
    unknown = value.keys() - fields.keys()
    if unknown:
        _fail(f"{path}.{min(unknown)}", f"unknown field for {cls.__name__}")
    kwargs = {k: _convert(v, fields[k], f"{path}.{k}")
              for k, v in value.items()}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError, OverflowError) as exc:
        _fail(path, str(exc))


# DLPNO records that may name their threshold preset in a reserved
# ``dlpno_thresholds`` key: run_job refuses the top-level dlpno_thresholds=
# beside an options object, so the preset travels with the record (#558).
_DLPNO_PRESET_RECORDS = frozenset({"dlpno_options", "dlpno_ccsd_options"})


def _dlpno_preset_record(value: dict, cls: type, path: str) -> Any:
    """JSON form of ``options_from_dlpno_thresholds(cls, preset, **fields)``.

    The preset is applied first and the record's own fields after it, so an
    explicit threshold field overrides the preset and labels the thresholds
    ``custom``, exactly as in the Python factory.
    """
    from .dlpno.thresholds import options_from_dlpno_thresholds

    fields = dict(value)
    preset = _convert(fields.pop("dlpno_thresholds"), str, f"{path}.dlpno_thresholds")
    checked = _record(fields, cls, path)
    try:
        return options_from_dlpno_thresholds(
            cls, preset, **{name: getattr(checked, name) for name in fields})
    except (TypeError, ValueError) as exc:
        _fail(f"{path}.dlpno_thresholds", str(exc))


def hydrate_option(key: str, value: Any) -> Any:
    """Convert known QVF records; leave established scalar/mapping APIs alone."""
    path = f"options.{key}"
    if key == "symmetry_reduction" and value is not None:
        from .symmetry_molecular_job import hydrate_source_request
        return hydrate_source_request(value)
    if key in _OPTION_TYPES:
        if value is None:
            return None
        module, name = _OPTION_TYPES[key]
        try:
            cls = _load_type(module, name)
        except (ImportError, AttributeError) as exc:
            _fail(path, f"{name} is unavailable in this runtime: {exc}")
        if (key in _DLPNO_PRESET_RECORDS and isinstance(value, dict)
                and "dlpno_thresholds" in value):
            return _dlpno_preset_record(value, cls, path)
        return _record(value, cls, path)
    if key == "fragments" and value is not None:
        _convert(value, list[list[int]], path)
    if key == "dft_plus_u" and value == []:
        return value
    if key in _OPAQUE_RECORDS and isinstance(value, (dict, list)):
        _fail(path, "structured runtime objects have no supported QVF JSON encoding")
    return value
