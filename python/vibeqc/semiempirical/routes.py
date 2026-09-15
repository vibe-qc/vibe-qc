"""Canonical route plans for semiempirical calculations.

This module is the first small slice of the unified semiempirical route
planner.  It owns alias canonicalisation and cheap support validation, while
the existing molecular, periodic Gamma, and SECCM dispatchers keep their
current kernel paths.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, replace
from typing import Iterable, Mapping

MOLECULAR_SEMIEMPIRICAL_METHODS = frozenset(
    {
        "dftb0",
        "scc_dftb",
        "pm6",
        "gfn2_xtb",
        "om1",
        "om2",
        "om3",
        "msindo",
    }
)
# PM7/UPM7 remain recognized spellings so callers receive the explicit
# scientific gate below instead of falling through to a non-semiempirical
# dispatcher (run_job's basis requirement, run_semiempirical's fallback).
SEMIEMPIRICAL_METHODS = MOLECULAR_SEMIEMPIRICAL_METHODS | {"ccm", "pm7", "upm7"}
SEMIEMPIRICAL_METHOD_ALIASES = {
    "gfn2": "gfn2_xtb",
    "gfn2-xtb": "gfn2_xtb",
    "gfn2xtb": "gfn2_xtb",
    "scc-dftb": "scc_dftb",
    "sccdftb": "scc_dftb",
    "dftb-0": "dftb0",
    "dftb": "dftb0",
    # SECCM is the public semiempirical cyclic-cluster name. ``ccm`` remains
    # the internal compatibility key used by existing dispatch and manifests.
    "seccm": "ccm",
    "se-ccm": "ccm",
    "msindo-seccm": "ccm",
    "msindo-ccm": "ccm",
    "semiempirical-ccm": "ccm",
    "semi-empirical-ccm": "ccm",
}


def normalise_semiempirical_method(method: str) -> str:
    """Return the canonical method key accepted by public dispatchers."""
    key = str(method).strip().lower()
    compact = key.replace(" ", "").replace("_", "-")
    return SEMIEMPIRICAL_METHOD_ALIASES.get(compact, key)


BOUNDARY_MOLECULE = "molecule"
BOUNDARY_PERIODIC_GAMMA = "periodic_gamma"
BOUNDARY_PERIODIC_K = "periodic_k"
BOUNDARY_SECCM_DIRECT_TORUS = "seccm_direct_torus"
# Compatibility name for the first route-planner slice. New code should use
# the SECCM name; the value is canonicalized to the SECCM boundary spelling.
BOUNDARY_CCM_DIRECT_TORUS = BOUNDARY_SECCM_DIRECT_TORUS

_SECCM_ELECTROSTATICS_FAMILIES = frozenset(
    {"none", "madelung", "ewald_gamma"}
)
# Shell gamma functional forms the periodic second-order kernel can use
# (cpp/include/vibeqc/semiempirical/core/periodic_gamma.hpp ShellGammaForm).
# "elstner" is the periodic default per the maintainer's D1 decision.
_PERIODIC_SHELL_GAMMA_FORMS = frozenset({"elstner", "klopman_ohno"})

_OMX_SECCM_THREE_CENTER_WEIGHTINGS = frozenset(
    {"peintinger_eq13", "janetzko_eq10"}
)
_SECCM_DIMENSIONAL_KERNELS = {
    1: "wire_1d",
    2: "parry_2d",
    3: "ewald_3d",
}
# The MSINDO engine keeps its historical 1-D long-range kernel: the truncated
# +/-1, +/-2 shell bare point-charge lattice sum of ``ccm1dmadelsum.f``
# (``indo::_madelung_potential_1d``; python/vibeqc/semiempirical/methods/
# msindo_ccm.py::_madelung_potential_1d and cpp/include/vibeqc/semiempirical/
# methods/indo/ccm_engine.hpp). That is *not* the converged Parry-type wire
# Ewald the other SECCM adapters run (``detail::wire_madkonst_1d``,
# cpp/src/semiempirical/seccm/ewald_1d.h) -- it performs no Ewald split, has
# no reciprocal-space channel and no neutralizing background, and the wire
# header describes itself as the kernel that "replaces the historical
# truncated +/-2-shell bare Coulomb sum ... kept frozen for MSINDO".
# Naming the two kernels apart keeps the Parry citation route (which gates on
# ``*_wire_1d``) from crediting Parry 1975 to the sum his method replaced;
# 2-D and 3-D MSINDO share the genuine ``_madkonst_2d`` / ``_madkonst_3d``
# machinery and so share those names. See GitLab #442.
_MSINDO_DIMENSIONAL_KERNELS = {
    **_SECCM_DIMENSIONAL_KERNELS,
    1: "truncated_1d",
}
_SECCM_ELECTROSTATICS_KERNELS = frozenset(
    {"none"}
    | {
        f"{family}_{kernel}"
        for family in _SECCM_ELECTROSTATICS_FAMILIES - {"none"}
        for kernel in (
            set(_SECCM_DIMENSIONAL_KERNELS.values())
            | set(_MSINDO_DIMENSIONAL_KERNELS.values())
        )
    }
)
_GFN2_SECCM_PARAMETER_SET = "gfn2-xtb-parameter-registry"
_GFN2_SECCM_K0_POLICY = "exact-pairwise-parry-de-leeuw"
_GFN2_SECCM_RESTART_SOURCES = frozenset({"neutral", "supplied"})
_GFN2_SECCM_SCC_MIXERS = frozenset(
    {"simple", "diis", "broyden", "broyden_eyert", "newton"}
)


def _seccm_electrostatics_kernel(
    dimension: int,
    family: str,
    method_key: str | None = None,
) -> str:
    """Return the concrete SECCM long-range kernel selected at runtime.

    ``method_key`` selects the engine's dimensional kernel table: the MSINDO
    engine keeps the frozen truncated 1-D lattice sum, every other SECCM
    adapter runs the Parry-type wire Ewald (see
    :data:`_MSINDO_DIMENSIONAL_KERNELS`).
    """
    dimension_value = int(dimension)
    kernels = (
        _MSINDO_DIMENSIONAL_KERNELS
        if str(method_key or "").strip().lower() == "msindo"
        else _SECCM_DIMENSIONAL_KERNELS
    )
    try:
        dimensional_kernel = kernels[dimension_value]
    except KeyError as exc:
        raise ValueError("SECCM periodic dimension must be 1, 2, or 3") from exc
    family_key = str(family).strip().lower()
    if family_key not in _SECCM_ELECTROSTATICS_FAMILIES:
        raise ValueError(
            "SECCM electrostatics family must be one of "
            f"{sorted(_SECCM_ELECTROSTATICS_FAMILIES)!r}"
        )
    if family_key == "none":
        return "none"
    return f"{family_key}_{dimensional_kernel}"

MATURITY_PRODUCTION = "production"
MATURITY_EXPERIMENTAL = "experimental"
MATURITY_NATIVE_FD = "native_fd"
MATURITY_MIXED_NATIVE = "mixed_native"
MATURITY_GATED_UNIMPLEMENTED = "gated_unimplemented"

EXECUTION_NATIVE = "native"
EXECUTION_NATIVE_BATCHED_FD = "native_batched_fd"
EXECUTION_PYTHON_ORCHESTRATION = "python_orchestration"
EXECUTION_UNSUPPORTED = "unsupported"

PERIODIC_GAMMA_SEMIEMPIRICAL_METHODS = frozenset(
    {"dftb0", "scc_dftb", "gfn2_xtb", "pm6"}
)
_SECCM_METHOD_ALIASES = frozenset(
    {"ccm"}
    | {
        alias
        for alias, canonical in SEMIEMPIRICAL_METHOD_ALIASES.items()
        if canonical == "ccm"
    }
)
_KNOWN_METHODS = (
    SEMIEMPIRICAL_METHODS
    | {"upm6", "upm7", "indo", "nddo", "msindo_ccm"}
    | _SECCM_METHOD_ALIASES
)
_METHOD_ROUTE_ALIASES = {
    "indo": "msindo",
    "msindo-indo": "msindo",
    "msindo-nddo": "msindo",
    "nddo": "msindo",
    "upm6": "pm6",
    "upm7": "pm7",
    **{alias: "msindo" for alias in _SECCM_METHOD_ALIASES},
}
_BOUNDARY_ALIASES = {
    "molecule": BOUNDARY_MOLECULE,
    "molecular": BOUNDARY_MOLECULE,
    "gas": BOUNDARY_MOLECULE,
    "periodic": BOUNDARY_PERIODIC_GAMMA,
    "gamma": BOUNDARY_PERIODIC_GAMMA,
    "periodic-gamma": BOUNDARY_PERIODIC_GAMMA,
    "gamma-periodic": BOUNDARY_PERIODIC_GAMMA,
    "periodic-gamma-point": BOUNDARY_PERIODIC_GAMMA,
    "periodic-k": BOUNDARY_PERIODIC_K,
    "periodic-kpoint": BOUNDARY_PERIODIC_K,
    "periodic-kpoints": BOUNDARY_PERIODIC_K,
    "kpoint": BOUNDARY_PERIODIC_K,
    "kpoints": BOUNDARY_PERIODIC_K,
    "bloch": BOUNDARY_PERIODIC_K,
    "seccm": BOUNDARY_SECCM_DIRECT_TORUS,
    "se-ccm": BOUNDARY_SECCM_DIRECT_TORUS,
    "semiempirical-ccm": BOUNDARY_SECCM_DIRECT_TORUS,
    "semi-empirical-ccm": BOUNDARY_SECCM_DIRECT_TORUS,
    "seccm-direct-torus": BOUNDARY_SECCM_DIRECT_TORUS,
    "ccm": BOUNDARY_SECCM_DIRECT_TORUS,
    "cyclic-cluster": BOUNDARY_SECCM_DIRECT_TORUS,
    "direct-torus": BOUNDARY_SECCM_DIRECT_TORUS,
    "ccm-direct-torus": BOUNDARY_SECCM_DIRECT_TORUS,
}
_PROPERTY_ALIASES = {
    "force": "gradient",
    "forces": "gradient",
}
_KNOWN_PROPERTIES = frozenset(
    {
        "energy",
        "gradient",
        "stress",
        "charges",
        "density",
        "orbitals",
        "bands",
        "dipole",
        "excited_state",
    }
)


def _alias_key(value: str) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "-")


def _canonical_method(method: str) -> tuple[str, str]:
    alias = _alias_key(method)
    key = normalise_semiempirical_method(method)
    key = _METHOD_ROUTE_ALIASES.get(alias, key)
    if key not in SEMIEMPIRICAL_METHODS:
        known = ", ".join(sorted(_KNOWN_METHODS | set(_METHOD_ROUTE_ALIASES)))
        raise ValueError(
            f"unknown semiempirical method {method!r}; known aliases: {known}"
        )
    return key, alias


def _canonical_boundary(
    boundary: str | None,
    method_alias: str,
) -> str:
    if boundary is None:
        return (
            BOUNDARY_SECCM_DIRECT_TORUS
            if method_alias in _SECCM_METHOD_ALIASES
            else BOUNDARY_MOLECULE
        )
    alias = _alias_key(boundary)
    try:
        canonical = _BOUNDARY_ALIASES[alias]
    except KeyError as exc:
        known = ", ".join(sorted(_BOUNDARY_ALIASES))
        raise ValueError(
            f"unknown semiempirical boundary {boundary!r}; known aliases: {known}"
        ) from exc
    if (
        method_alias in _SECCM_METHOD_ALIASES
        and canonical != BOUNDARY_SECCM_DIRECT_TORUS
    ):
        raise ValueError(
            f"method={method_alias!r} denotes the SECCM boundary; use "
            "boundary='seccm_direct_torus' or select a molecular "
            "semiempirical Hamiltonian explicitly."
        )
    return canonical


def _canonical_properties(properties: Iterable[str] | None) -> tuple[str, ...]:
    if properties is None:
        return ("energy",)
    canonical: list[str] = []
    for prop in properties:
        key = str(prop).strip().lower().replace("-", "_").replace(" ", "_")
        key = _PROPERTY_ALIASES.get(key, key)
        if key not in _KNOWN_PROPERTIES:
            known = ", ".join(sorted(_KNOWN_PROPERTIES | set(_PROPERTY_ALIASES)))
            raise ValueError(
                f"unknown semiempirical property {prop!r}; known: {known}"
            )
        if key not in canonical:
            canonical.append(key)
    return tuple(canonical or ["energy"])


def _spin_label(
    *,
    multiplicity: int | None,
    unrestricted: bool | None,
    method_alias: str,
) -> str:
    if unrestricted is True or method_alias == "upm6" or method_alias == "upm7":
        return "unrestricted"
    if multiplicity is not None and int(multiplicity) != 1:
        return "unrestricted"
    return "closed_shell"


def _method_family(method_key: str) -> str:
    if method_key in {"dftb0", "scc_dftb"}:
        return "dftb"
    if method_key == "gfn2_xtb":
        return "gfn2"
    if method_key in {"pm6", "pm7", "om1", "om2", "om3"}:
        return "nddo"
    return "msindo"


def _variant(
    method_key: str,
    *,
    method_alias: str,
    nddo: bool,
    solvent,
    spin: str,
) -> str:
    if method_key == "msindo":
        if solvent is not None:
            return "cosmo"
        return "nddo" if nddo or method_alias in {"nddo", "msindo-nddo"} else "indo"
    if method_key == "pm6" and (method_alias == "upm6" or spin == "unrestricted"):
        return "upm6"
    if method_key == "pm7" and (method_alias == "upm7" or spin == "unrestricted"):
        return "upm7"
    return method_key


def _scc_policy(method_key: str) -> str:
    if method_key == "dftb0":
        return "none"
    if method_key == "scc_dftb":
        return "atomic"
    if method_key == "gfn2_xtb":
        return "shell"
    return "method_default"


def _validate_knobs(
    *,
    method_key: str,
    boundary: str,
    nddo: bool,
    solvent,
    ccm_options,
) -> None:
    if nddo and method_key != "msindo":
        raise ValueError(
            f"nddo=True is only supported for method='msindo'; got "
            f"method={method_key!r}."
        )
    if solvent is not None and method_key != "msindo":
        raise ValueError(
            f"Implicit solvation is not supported for method={method_key!r}; "
            "semiempirical COSMO is available only with method='msindo'."
        )
    if nddo and solvent is not None:
        raise ValueError(
            "nddo=True cannot be combined with solvent; the current MSINDO "
            "COSMO route uses the INDO Hamiltonian."
        )
    if boundary == BOUNDARY_SECCM_DIRECT_TORUS and nddo:
        raise ValueError(
            "nddo=True is not implemented for SECCM; the current SECCM "
            "backend is MSINDO INDO."
        )
    if boundary == BOUNDARY_SECCM_DIRECT_TORUS and solvent is not None:
        raise ValueError(
            "Implicit solvation is not supported for SECCM; semiempirical "
            "COSMO is molecular MSINDO only."
        )
    if ccm_options is not None and boundary != BOUNDARY_SECCM_DIRECT_TORUS:
        raise ValueError(
            "ccm_options is only supported for the SECCM boundary; use "
            f"method='seccm' or boundary='seccm'; got method={method_key!r}, "
            f"boundary={boundary!r}."
        )
    if (
        ccm_options is not None
        and boundary == BOUNDARY_SECCM_DIRECT_TORUS
        and method_key != "msindo"
    ):
        raise ValueError(
            "ccm_options belongs to the MSINDO SECCM adapter and cannot be "
            f"used with method={method_key!r}."
        )


def _validate_boundary(method_key: str, boundary: str) -> None:
    if boundary == BOUNDARY_MOLECULE:
        if method_key == "pm7":
            # Lazy import: methods/pm7.py imports this module at its top.
            from vibeqc.semiempirical.methods.pm7 import PM7_GATE_MESSAGE

            raise NotImplementedError(PM7_GATE_MESSAGE)
        if method_key not in MOLECULAR_SEMIEMPIRICAL_METHODS:
            raise NotImplementedError(
                f"method={method_key!r} is not a molecular semiempirical route."
            )
        return
    if boundary == BOUNDARY_PERIODIC_GAMMA:
        if method_key not in PERIODIC_GAMMA_SEMIEMPIRICAL_METHODS:
            raise NotImplementedError(
                f"method={method_key!r} has no Bloch-periodic Gamma "
                "semiempirical route; use an explicitly topology-bound "
                "SECCM route where that Hamiltonian is supported."
            )
        return
    if boundary == BOUNDARY_PERIODIC_K:
        if method_key in {"dftb0", "scc_dftb"}:
            return
        raise NotImplementedError(
            "full k-point semiempirical routes are currently public only "
            "for DFTB0 and SCC-DFTB energy/band/gradient/stress routes; got "
            f"method={method_key!r}."
        )
    if (
        boundary == BOUNDARY_SECCM_DIRECT_TORUS
        and method_key == "om1"
    ):
        raise NotImplementedError(
            "OM1-SECCM is unavailable because the published analytic "
            "core-valence ECP is not implemented; use OM2- or OM3-SECCM"
        )
    if (
        boundary == BOUNDARY_SECCM_DIRECT_TORUS
        and method_key
        not in {
            "msindo",
            "dftb0",
            "scc_dftb",
            "pm6",
            "om2",
            "om3",
            "gfn2_xtb",
        }
    ):
        raise NotImplementedError(
            "the current SECCM implementation has no adapter for "
            f"method={method_key!r}."
        )


def _validate_charge(*, method_key: str, boundary: str, charge: int) -> None:
    # SCC-DFTB-SECCM accepts charged supercells through the opt-in
    # Madelung/Ewald embedding; the engine itself fails closed for a charged
    # cell without embedding, so the route plan does not gate charge there.
    if (
        boundary == BOUNDARY_SECCM_DIRECT_TORUS
        and method_key in {"dftb0", "pm6", "om1", "om2", "om3", "gfn2_xtb"}
        and charge != 0
    ):
        raise NotImplementedError(
            f"{method_key.upper()}-SECCM supports only neutral finite cyclic "
            "clusters; charged-cell conventions are not implemented."
        )


def _validate_spin(
    *,
    method_key: str,
    boundary: str,
    variant: str,
    spin: str,
) -> None:
    if spin == "closed_shell":
        return
    if boundary == BOUNDARY_SECCM_DIRECT_TORUS and spin != "closed_shell":
        raise NotImplementedError(
            f"the current {method_key.upper()} SECCM adapter is closed-shell "
            "only; open-shell SECCM is not implemented."
        )
    if method_key == "gfn2_xtb":
        raise NotImplementedError(
            "the GFN2-xTB route is closed-shell only; the native unrestricted "
            "reference does not yet implement the full validated GFN2 energy "
            "functional."
        )
    if boundary == BOUNDARY_PERIODIC_GAMMA and method_key in {
        "pm6",
        "pm7",
        "om1",
        "om2",
        "om3",
    }:
        raise NotImplementedError(
            f"periodic Gamma {variant.upper()} is closed-shell only; no "
            "unrestricted periodic NDDO kernel is implemented."
        )
    if method_key == "msindo" and variant == "nddo":
        raise NotImplementedError(
            "MSINDO NDDO mode is closed-shell only; open-shell NDDO is not "
            "implemented."
        )


def _supported_properties(
    *,
    method_key: str,
    boundary: str,
    variant: str,
    spin: str,
) -> frozenset[str]:
    if boundary == BOUNDARY_PERIODIC_K:
        if method_key in {"dftb0", "scc_dftb"}:
            return frozenset({"energy", "bands", "gradient", "stress"})
        return frozenset({"energy"})
    if boundary == BOUNDARY_PERIODIC_GAMMA:
        if method_key in {"pm6", "pm7", "om1", "om2", "om3"}:
            return frozenset({"energy", "gradient", "stress"})
        if method_key in {"dftb0", "scc_dftb", "gfn2_xtb"}:
            return frozenset({"energy", "gradient"})
        return frozenset({"energy"})
    if boundary == BOUNDARY_SECCM_DIRECT_TORUS:
        if method_key == "dftb0":
            return frozenset({"energy", "gradient"})
        if method_key in {"scc_dftb", "pm6"}:
            return frozenset({"energy", "gradient"})
        if method_key in {
            "om1",
            "om2",
            "om3",
            "gfn2_xtb",
        }:
            return frozenset({"energy"})
        return frozenset({"energy", "gradient"})
    if method_key in {"dftb0", "scc_dftb", "gfn2_xtb", "pm6", "pm7", "om1", "om2", "om3"}:
        return frozenset({"energy", "gradient"})
    if (
        method_key == "msindo"
        and variant in {"indo", "nddo"}
        and spin == "closed_shell"
    ):
        return frozenset({"energy", "gradient"})
    return frozenset({"energy"})


def _validate_properties(
    properties: tuple[str, ...],
    *,
    method_key: str,
    boundary: str,
    variant: str,
    spin: str,
) -> None:
    supported = _supported_properties(
        method_key=method_key,
        boundary=boundary,
        variant=variant,
        spin=spin,
    )
    unsupported = [prop for prop in properties if prop not in supported]
    if unsupported:
        raise NotImplementedError(
            "semiempirical route "
            f"method={method_key!r}, boundary={boundary!r}, variant={variant!r} "
            f"does not support requested property/properties: "
            f"{', '.join(unsupported)}."
        )


def _status_route(
    *,
    method_key: str,
    boundary: str,
    variant: str,
    properties: tuple[str, ...],
) -> str:
    if boundary == BOUNDARY_PERIODIC_K:
        if method_key == "dftb0" and (
            "gradient" in properties or "stress" in properties
        ):
            return "periodic-dftb0-kpoint-gradient-fd"
        if method_key == "scc_dftb" and (
            "gradient" in properties or "stress" in properties
        ):
            return "periodic-scc-dftb-kpoint-gradient-fd"
        if method_key == "dftb0":
            return "periodic-dftb0-kpoint"
        if method_key == "scc_dftb":
            return "periodic-scc-dftb-kpoint"
    if boundary == BOUNDARY_SECCM_DIRECT_TORUS:
        if method_key == "dftb0":
            return (
                "dftb0-seccm-gradient-analytic"
                if "gradient" in properties
                else "dftb0-seccm-energy"
            )
        if method_key == "scc_dftb":
            return (
                "scc-dftb-seccm-gradient-analytic"
                if "gradient" in properties
                else "scc-dftb-seccm-energy"
            )
        if method_key == "pm6":
            return (
                "pm6-seccm-gradient-fd"
                if "gradient" in properties
                else "pm6-seccm-energy"
            )
        if method_key in {"om1", "om2", "om3"}:
            return "omx-seccm-energy"
        if method_key == "gfn2_xtb":
            return "gfn2-seccm-energy"
        return (
            "msindo-ccm-gradient-analytic"
            if "gradient" in properties
            else "msindo-ccm-energy"
        )
    if boundary == BOUNDARY_PERIODIC_GAMMA:
        if method_key == "dftb0" and "gradient" in properties:
            return "periodic-dftb0-gradient-analytic"
        if method_key == "scc_dftb" and "gradient" in properties:
            return "periodic-scc-dftb-gradient-fd"
        if method_key == "gfn2_xtb" and "gradient" in properties:
            return "periodic-gfn2-gradient"
        if method_key == "pm6":
            return "periodic-pm6"
        if method_key == "pm7":
            return "periodic-pm7"
        if method_key in {"om1", "om2", "om3"}:
            return "periodic-omx"
    if method_key in {"dftb0", "scc_dftb"}:
        return "dftb"
    if method_key == "gfn2_xtb":
        return "gfn2-xtb"
    if method_key == "pm6":
        return "pm6-gradient-fd" if "gradient" in properties else "pm6"
    if method_key == "pm7":
        return "pm7-gradient-fd" if "gradient" in properties else "pm7"
    if method_key == "om1":
        return "om1-gradient-fd" if "gradient" in properties else "om1"
    if method_key in {"om2", "om3"}:
        return "omx-gradient-fd" if "gradient" in properties else "omx"
    if method_key == "msindo" and variant == "cosmo":
        return "msindo-cosmo"
    if method_key == "msindo" and variant == "nddo" and "gradient" in properties:
        return "msindo-nddo-gradient-analytic"
    if method_key == "msindo" and "gradient" in properties:
        return "msindo-gradient"
    return "msindo-energy"


def _maturity_and_execution(
    *,
    method_key: str,
    boundary: str,
    variant: str,
    properties: tuple[str, ...],
) -> tuple[str, str]:
    if boundary == BOUNDARY_PERIODIC_K:
        if method_key in {"dftb0", "scc_dftb"}:
            if "gradient" in properties or "stress" in properties:
                return MATURITY_NATIVE_FD, EXECUTION_NATIVE_BATCHED_FD
            return MATURITY_EXPERIMENTAL, EXECUTION_NATIVE
        return MATURITY_GATED_UNIMPLEMENTED, EXECUTION_UNSUPPORTED
    if (
        boundary == BOUNDARY_SECCM_DIRECT_TORUS
        and method_key
        in {"dftb0", "scc_dftb", "pm6", "om1", "om2", "om3", "gfn2_xtb"}
    ):
        return MATURITY_EXPERIMENTAL, EXECUTION_NATIVE
    if boundary == BOUNDARY_PERIODIC_GAMMA:
        if method_key == "dftb0" and "gradient" in properties:
            return MATURITY_PRODUCTION, EXECUTION_NATIVE
        if method_key == "scc_dftb" and "gradient" in properties:
            return MATURITY_EXPERIMENTAL, EXECUTION_PYTHON_ORCHESTRATION
        if method_key == "gfn2_xtb":
            if "gradient" in properties:
                return MATURITY_EXPERIMENTAL, EXECUTION_PYTHON_ORCHESTRATION
            return MATURITY_EXPERIMENTAL, EXECUTION_NATIVE
        if method_key in {"pm6", "pm7", "om1", "om2", "om3"}:
            if "gradient" in properties or "stress" in properties:
                return MATURITY_NATIVE_FD, EXECUTION_NATIVE_BATCHED_FD
            return MATURITY_MIXED_NATIVE, EXECUTION_PYTHON_ORCHESTRATION
        return MATURITY_PRODUCTION, EXECUTION_NATIVE
    if variant == "cosmo":
        return MATURITY_MIXED_NATIVE, EXECUTION_PYTHON_ORCHESTRATION
    if variant == "nddo" and "gradient" in properties:
        return MATURITY_MIXED_NATIVE, EXECUTION_PYTHON_ORCHESTRATION
    if method_key == "gfn2_xtb":
        return MATURITY_EXPERIMENTAL, EXECUTION_NATIVE
    if method_key in {"om1", "om2", "om3"}:
        execution = (
            EXECUTION_NATIVE_BATCHED_FD
            if "gradient" in properties
            else EXECUTION_NATIVE
        )
        return MATURITY_EXPERIMENTAL, execution
    if method_key == "pm6" and "gradient" in properties:
        return MATURITY_NATIVE_FD, EXECUTION_NATIVE_BATCHED_FD
    return MATURITY_PRODUCTION, EXECUTION_NATIVE


def _parameter_scope(*, method_key: str, boundary: str, variant: str) -> str:
    if boundary == BOUNDARY_SECCM_DIRECT_TORUS:
        if method_key == "dftb0":
            return (
                "in-house DFTB screening parameters, explicit repulsive-pair "
                "scope (H, C, N, O, F, P, S, Cl)"
            )
        if method_key == "scc_dftb":
            return "in-house DFTB screening parameters"
        if method_key == "pm6":
            return "PM6 auto parameter loader"
        if method_key in {"om2", "om3"}:
            return "OM2/OM3 published element scopes"
        if method_key == "gfn2_xtb":
            return "GFN2-xTB parameter registry"
        return "MSINDO SECCM (legacy CCM API) H-Kr native scope"
    if method_key in {"dftb0", "scc_dftb"}:
        return "in-house DFTB screening/preoptimization parameters"
    if method_key == "gfn2_xtb":
        return "GFN2-xTB parameter registry"
    if method_key == "pm6":
        return "PM6 auto parameter loader"
    if method_key == "pm7":
        return "PM7 auto parameter loader"
    if method_key in {"om1", "om2", "om3"}:
        return "OM1/OM2/OM3 published element scopes"
    if variant == "nddo":
        return (
            "MSINDO NDDO closed-shell parameters for H, Li-F, and Na-Cl; "
            "validated energy and analytic-gradient scope H, Li-F, and Na-Cl"
        )
    if variant == "cosmo":
        return "MSINDO COSMO within the MSINDO element scope"
    return "MSINDO INDO H-Xe scope"


@dataclass(frozen=True)
class GFN2SECCMHamiltonianIdentity:
    """Serializable identity of the GFN2-SECCM Hamiltonian that ran.

    Requested electrostatics and temperature are kept separately from the
    resolved values because the exact molecular shortcut and the bounded SCC
    retry ladder may legitimately execute a different runtime route. Solver
    controls and restart state are intentionally outside this Hamiltonian
    identity.
    """

    schema_version: int
    requested_electrostatics_family: str
    resolved_electrostatics_kernel: str
    requested_electronic_temperature: float
    resolved_electronic_temperature: float
    include_aes: bool
    madelung_s_weighted: bool
    madelung_no_self: bool
    ewald_gamma_molecular_onsite: bool
    molecular_delegated: bool

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "GFN2-SECCM Hamiltonian identity schema_version must be 1"
            )
        family = str(self.requested_electrostatics_family).strip().lower()
        if family not in _SECCM_ELECTROSTATICS_FAMILIES:
            raise ValueError(
                "GFN2-SECCM requested electrostatics family must be one of "
                f"{sorted(_SECCM_ELECTROSTATICS_FAMILIES)!r}"
            )
        for name in (
            "include_aes",
            "madelung_s_weighted",
            "madelung_no_self",
            "ewald_gamma_molecular_onsite",
            "molecular_delegated",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(
                    f"GFN2-SECCM Hamiltonian identity {name} must be boolean"
                )
        if isinstance(self.requested_electronic_temperature, bool) or (
            isinstance(self.resolved_electronic_temperature, bool)
        ):
            raise ValueError(
                "GFN2-SECCM Hamiltonian temperatures must be numeric"
            )
        requested_temperature = float(self.requested_electronic_temperature)
        resolved_temperature = float(self.resolved_electronic_temperature)
        if (
            not math.isfinite(requested_temperature)
            or requested_temperature < 0.0
            or not math.isfinite(resolved_temperature)
            or resolved_temperature < 0.0
        ):
            raise ValueError(
                "GFN2-SECCM Hamiltonian temperatures must be finite and "
                "non-negative"
            )
        if (self.madelung_s_weighted or self.madelung_no_self) and (
            family != "madelung"
        ):
            raise ValueError(
                "GFN2-SECCM Madelung modifiers require requested "
                "electrostatics family 'madelung'"
            )
        if self.ewald_gamma_molecular_onsite and family != "ewald_gamma":
            raise ValueError(
                "GFN2-SECCM molecular on-site gamma requires requested "
                "electrostatics family 'ewald_gamma'"
            )
        resolved_kernel = str(self.resolved_electrostatics_kernel).strip().lower()
        if resolved_kernel not in _SECCM_ELECTROSTATICS_KERNELS:
            raise ValueError(
                "GFN2-SECCM resolved electrostatics kernel must be one of "
                f"{sorted(_SECCM_ELECTROSTATICS_KERNELS)!r}"
            )
        if self.molecular_delegated:
            if resolved_kernel != "none":
                raise ValueError(
                    "molecular-delegated GFN2-SECCM identity must resolve "
                    "to electrostatics kernel 'none'"
                )
            if family == "madelung" or not self.include_aes:
                raise ValueError(
                    "molecular-delegated GFN2-SECCM identity requires no "
                    "Madelung request and include_aes=True"
                )
        elif family == "none":
            if resolved_kernel != "none":
                raise ValueError(
                    "unembedded GFN2-SECCM identity must resolve to "
                    "electrostatics kernel 'none'"
                )
        elif not resolved_kernel.startswith(f"{family}_"):
            raise ValueError(
                "GFN2-SECCM resolved electrostatics kernel does not match "
                "the requested family"
            )
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "requested_electrostatics_family", family)
        object.__setattr__(
            self,
            "resolved_electrostatics_kernel",
            resolved_kernel,
        )
        object.__setattr__(
            self,
            "requested_electronic_temperature",
            requested_temperature,
        )
        object.__setattr__(
            self,
            "resolved_electronic_temperature",
            resolved_temperature,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe canonical provenance payload."""
        return {
            "schema_version": self.schema_version,
            "requested_electrostatics_family": (
                self.requested_electrostatics_family
            ),
            "resolved_electrostatics_kernel": (
                self.resolved_electrostatics_kernel
            ),
            "requested_electronic_temperature": (
                self.requested_electronic_temperature
            ),
            "resolved_electronic_temperature": (
                self.resolved_electronic_temperature
            ),
            "include_aes": self.include_aes,
            "madelung_s_weighted": self.madelung_s_weighted,
            "madelung_no_self": self.madelung_no_self,
            "ewald_gamma_molecular_onsite": (
                self.ewald_gamma_molecular_onsite
            ),
            "molecular_delegated": self.molecular_delegated,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "GFN2SECCMHamiltonianIdentity":
        """Restore an identity emitted by :meth:`to_dict`."""
        return cls(**dict(payload))


def _gfn2_restart_sha256(values: Iterable[object]) -> tuple[int, str]:
    """Hash finite restart charges under a cross-platform float64 rule."""
    digest = hashlib.sha256()
    count = 0
    for value in values:
        if isinstance(value, Iterable) and not isinstance(
            value,
            (str, bytes),
        ):
            raise ValueError(
                "GFN2-SECCM restart charges must be a one-dimensional "
                "numeric vector"
            )
        if isinstance(value, bool):
            number = float(value)
        else:
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "GFN2-SECCM restart charges must be finite numbers"
                ) from exc
        if not math.isfinite(number):
            raise ValueError(
                "GFN2-SECCM restart charges must be finite numbers"
            )
        # Positive and negative zero are the same SCC starting state. Store
        # one canonical byte representation so their provenance also agrees.
        if number == 0.0:
            number = 0.0
        digest.update(struct.pack("<d", number))
        count += 1
    return count, digest.hexdigest()


@dataclass(frozen=True)
class GFN2SECCMRestartIdentity:
    """Immutable identity of the requested SCC start and fallback policy.

    Restart vectors are represented by a count plus a canonical SHA-256
    rather than retaining a mutable or potentially large array. An absent or
    empty vector is the neutral start. A nonempty vector, including an
    all-zero vector, remains an explicitly supplied start. Automatic retries
    use a neutral start so they cannot silently repeat a failed caller state.
    """

    schema_version: int
    primary_source: str
    n_shell_charges: int
    shell_charges_sha256: str | None
    automatic_retry_source: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "GFN2-SECCM restart identity schema_version must be 1"
            )
        primary_source = str(self.primary_source).strip().lower()
        if primary_source not in _GFN2_SECCM_RESTART_SOURCES:
            raise ValueError(
                "GFN2-SECCM restart primary_source must be 'neutral' or "
                "'supplied'"
            )
        if (
            type(self.n_shell_charges) is not int
            or self.n_shell_charges < 0
        ):
            raise ValueError(
                "GFN2-SECCM restart n_shell_charges must be a non-negative "
                "integer"
            )
        automatic_retry_source = str(
            self.automatic_retry_source
        ).strip().lower()
        if automatic_retry_source != "neutral":
            raise ValueError(
                "GFN2-SECCM automatic retries must use a neutral restart"
            )
        fingerprint = self.shell_charges_sha256
        if primary_source == "neutral":
            if self.n_shell_charges != 0 or fingerprint is not None:
                raise ValueError(
                    "neutral GFN2-SECCM restart identity requires zero shell "
                    "charges and no fingerprint"
                )
        else:
            if self.n_shell_charges == 0:
                raise ValueError(
                    "supplied GFN2-SECCM restart identity requires at least "
                    "one shell charge"
                )
            if not isinstance(fingerprint, str):
                raise ValueError(
                    "supplied GFN2-SECCM restart identity requires a SHA-256 "
                    "fingerprint"
                )
            fingerprint = fingerprint.strip().lower()
            if len(fingerprint) != 64 or any(
                character not in "0123456789abcdef"
                for character in fingerprint
            ):
                raise ValueError(
                    "GFN2-SECCM restart fingerprint must be a lowercase "
                    "SHA-256 digest"
                )
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "primary_source", primary_source)
        object.__setattr__(
            self,
            "automatic_retry_source",
            automatic_retry_source,
        )
        object.__setattr__(self, "shell_charges_sha256", fingerprint)

    @classmethod
    def from_initial_shell_charges(
        cls,
        initial_shell_charges: Iterable[object] | None,
    ) -> "GFN2SECCMRestartIdentity":
        """Canonicalize a public restart vector into immutable metadata."""
        if initial_shell_charges is None:
            return cls(
                schema_version=1,
                primary_source="neutral",
                n_shell_charges=0,
                shell_charges_sha256=None,
                automatic_retry_source="neutral",
            )
        if isinstance(initial_shell_charges, (str, bytes)):
            raise ValueError(
                "GFN2-SECCM restart charges must be a numeric vector"
            )
        try:
            count, fingerprint = _gfn2_restart_sha256(
                initial_shell_charges
            )
        except TypeError as exc:
            raise ValueError(
                "GFN2-SECCM restart charges must be a numeric vector"
            ) from exc
        if count == 0:
            return cls(
                schema_version=1,
                primary_source="neutral",
                n_shell_charges=0,
                shell_charges_sha256=None,
                automatic_retry_source="neutral",
            )
        return cls(
            schema_version=1,
            primary_source="supplied",
            n_shell_charges=count,
            shell_charges_sha256=fingerprint,
            automatic_retry_source="neutral",
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe canonical restart payload."""
        return {
            "schema_version": self.schema_version,
            "primary_source": self.primary_source,
            "n_shell_charges": self.n_shell_charges,
            "shell_charges_sha256": self.shell_charges_sha256,
            "automatic_retry_source": self.automatic_retry_source,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "GFN2SECCMRestartIdentity":
        """Restore a restart identity emitted by :meth:`to_dict`."""
        return cls(**dict(payload))


@dataclass(frozen=True)
class GFN2SECCMRunControls:
    """Canonical non-Hamiltonian controls for one GFN2-SECCM request.

    The Hamiltonian identity remains separate: convergence thresholds,
    iteration budgets, charge mixers, and restart state can change which
    fixed point is accepted without changing the underlying Hamiltonian.
    The deprecated K=0 flag is retained as requested-control provenance while
    the resolved policy records that both values execute the exact pairwise
    Parry/de Leeuw kernel.
    """

    schema_version: int
    parameter_set: str
    max_iter: int
    conv_tol_charge: float
    charge_mixing: float
    scc_mixer: str
    requested_ewald_gamma_k0_global: bool
    resolved_ewald_gamma_k0_policy: str
    finite_torus_gap_tolerance: float
    restart: GFN2SECCMRestartIdentity

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(
                "GFN2-SECCM run controls schema_version must be 1"
            )
        if self.parameter_set != _GFN2_SECCM_PARAMETER_SET:
            raise ValueError(
                "GFN2-SECCM run controls require parameter_set="
                f"{_GFN2_SECCM_PARAMETER_SET!r}"
            )
        if type(self.max_iter) is not int or self.max_iter < 1:
            raise ValueError(
                "GFN2-SECCM run controls max_iter must be an integer >= 1"
            )
        for name in ("conv_tol_charge", "charge_mixing"):
            if isinstance(getattr(self, name), bool):
                raise ValueError(
                    f"GFN2-SECCM run controls {name} must be numeric"
                )
        conv_tol_charge = float(self.conv_tol_charge)
        charge_mixing = float(self.charge_mixing)
        if not math.isfinite(conv_tol_charge) or conv_tol_charge <= 0.0:
            raise ValueError(
                "GFN2-SECCM run controls conv_tol_charge must be finite and "
                "positive"
            )
        if (
            not math.isfinite(charge_mixing)
            or charge_mixing <= 0.0
            or charge_mixing > 1.0
        ):
            raise ValueError(
                "GFN2-SECCM run controls charge_mixing must be finite and "
                "in (0, 1]"
            )
        scc_mixer = str(self.scc_mixer).strip().lower()
        if scc_mixer not in _GFN2_SECCM_SCC_MIXERS:
            raise ValueError(
                "GFN2-SECCM run controls scc_mixer must be one of "
                f"{sorted(_GFN2_SECCM_SCC_MIXERS)!r}"
            )
        if type(self.requested_ewald_gamma_k0_global) is not bool:
            raise ValueError(
                "GFN2-SECCM run controls "
                "requested_ewald_gamma_k0_global must be boolean"
            )
        resolved_policy = str(
            self.resolved_ewald_gamma_k0_policy
        ).strip().lower()
        if resolved_policy != _GFN2_SECCM_K0_POLICY:
            raise ValueError(
                "GFN2-SECCM resolved Ewald K=0 policy must be "
                f"{_GFN2_SECCM_K0_POLICY!r}"
            )
        if isinstance(self.finite_torus_gap_tolerance, bool):
            raise ValueError(
                "GFN2-SECCM finite_torus_gap_tolerance must be numeric"
            )
        gap_tolerance = float(self.finite_torus_gap_tolerance)
        if gap_tolerance != 1.0e-8:
            raise ValueError(
                "GFN2-SECCM finite_torus_gap_tolerance must be the "
                "supported fixed value 1e-8"
            )
        if not isinstance(self.restart, GFN2SECCMRestartIdentity):
            raise ValueError(
                "GFN2-SECCM run controls restart must be a "
                "GFN2SECCMRestartIdentity"
            )
        object.__setattr__(self, "schema_version", 1)
        object.__setattr__(self, "conv_tol_charge", conv_tol_charge)
        object.__setattr__(self, "charge_mixing", charge_mixing)
        object.__setattr__(self, "scc_mixer", scc_mixer)
        object.__setattr__(
            self,
            "resolved_ewald_gamma_k0_policy",
            resolved_policy,
        )
        object.__setattr__(
            self,
            "finite_torus_gap_tolerance",
            gap_tolerance,
        )

    @classmethod
    def from_request(
        cls,
        *,
        parameter_set: str = _GFN2_SECCM_PARAMETER_SET,
        max_iter: int = 3600,
        conv_tol_charge: float = 1.0e-6,
        charge_mixing: float = 0.1,
        scc_mixer: str = "simple",
        ewald_gamma_k0_global: bool = False,
        initial_shell_charges: Iterable[object] | None = None,
    ) -> "GFN2SECCMRunControls":
        """Build canonical metadata from the public request controls."""
        return cls(
            schema_version=1,
            parameter_set=parameter_set,
            max_iter=max_iter,
            conv_tol_charge=conv_tol_charge,
            charge_mixing=charge_mixing,
            scc_mixer=scc_mixer,
            requested_ewald_gamma_k0_global=ewald_gamma_k0_global,
            resolved_ewald_gamma_k0_policy=_GFN2_SECCM_K0_POLICY,
            finite_torus_gap_tolerance=1.0e-8,
            restart=GFN2SECCMRestartIdentity.from_initial_shell_charges(
                initial_shell_charges
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe canonical run-control payload."""
        return {
            "schema_version": self.schema_version,
            "parameter_set": self.parameter_set,
            "max_iter": self.max_iter,
            "conv_tol_charge": self.conv_tol_charge,
            "charge_mixing": self.charge_mixing,
            "scc_mixer": self.scc_mixer,
            "requested_ewald_gamma_k0_global": (
                self.requested_ewald_gamma_k0_global
            ),
            "resolved_ewald_gamma_k0_policy": (
                self.resolved_ewald_gamma_k0_policy
            ),
            "finite_torus_gap_tolerance": (
                self.finite_torus_gap_tolerance
            ),
            "restart": self.restart.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> "GFN2SECCMRunControls":
        """Restore run controls emitted by :meth:`to_dict`."""
        data = dict(payload)
        restart_payload = data.get("restart")
        if isinstance(restart_payload, Mapping):
            data["restart"] = GFN2SECCMRestartIdentity.from_dict(
                restart_payload
            )
        return cls(**data)


@dataclass(frozen=True)
class SemiempiricalRoutePlan:
    """Canonical semiempirical method, boundary, and property plan."""

    method_key: str
    method_family: str
    variant: str
    boundary: str
    charge: int
    spin: str
    scc: str
    properties: tuple[str, ...]
    maturity: str
    parameter_scope: str
    execution: str
    status_route: str
    periodic_dimension: int | None = None
    electrostatics_kernel: str = "none"
    electronic_temperature: float = 0.0
    three_center_weighting: str | None = None
    gfn2_hamiltonian_identity: GFN2SECCMHamiltonianIdentity | None = None
    gfn2_run_controls: GFN2SECCMRunControls | None = None
    truncated_electrostatics_acknowledged: bool = False
    shell_gamma_form: str | None = None

    @property
    def route_key(self) -> str:
        """Stable compact key used by tests and future dispatch adapters."""
        props = "+".join(self.properties)
        return (
            f"{self.boundary}:{self.method_family}:{self.variant}:"
            f"q{self.charge}:{self.spin}:{props}"
        )

    def with_periodic_electrostatics_runtime(
        self,
        *,
        periodic_dimension: int,
        shell_gamma_form: str,
    ) -> "SemiempiricalRoutePlan":
        """Attach the Bloch-periodic electrostatics context to a plan.

        The periodic Gamma GFN2 driver builds its second-order kernel as an
        Ewald-split lattice sum of a short-range shell gamma (issues
        #296/#338), so a run's citation surface depends on two things the
        method key alone cannot express: which gamma functional form was
        used, and which dimensional Ewald channel the cell selected.  The
        driver calls this once the system and options have fixed both, the
        same way SECCM adapters call :meth:`with_seccm_runtime`.

        ``shell_gamma_form`` is ``"elstner"`` (the periodic default, the
        maintainer's D1 decision of 2026-08-28) or ``"klopman_ohno"``.
        """
        if self.boundary not in {BOUNDARY_PERIODIC_GAMMA, BOUNDARY_PERIODIC_K}:
            raise ValueError(
                "periodic electrostatics metadata requires a Bloch-periodic "
                "boundary ('periodic_gamma' or 'periodic_k')"
            )
        dimension = int(periodic_dimension)
        if dimension not in (1, 2, 3):
            raise ValueError(
                "periodic electrostatics dimension must be 1, 2, or 3"
            )
        form = str(shell_gamma_form).strip().lower()
        if form not in _PERIODIC_SHELL_GAMMA_FORMS:
            raise ValueError(
                "shell_gamma_form must be one of "
                f"{sorted(_PERIODIC_SHELL_GAMMA_FORMS)!r}"
            )
        return replace(
            self,
            periodic_dimension=dimension,
            shell_gamma_form=form,
        )

    def with_seccm_runtime(
        self,
        *,
        periodic_dimension: int,
        electrostatics_family: str = "none",
        electronic_temperature: float = 0.0,
        three_center_weighting: str | None = None,
        truncated_electrostatics_acknowledged: bool = False,
    ) -> "SemiempiricalRoutePlan":
        """Attach the runtime SECCM citation context to a validated plan.

        The method/boundary plan is intentionally cheap and is often built
        before a topology or native options object exists. SECCM adapters call
        this once those objects have selected the actual dimensional Coulomb
        kernel and electronic temperature. Keeping the concrete kernel on the
        result prevents a static 2-D citation from leaking into 1-D/3-D runs,
        and the engine-specific 1-D name (see
        :data:`_MSINDO_DIMENSIONAL_KERNELS`) keeps the Parry wire-Ewald
        citation off MSINDO's frozen truncated lattice sum.
        """
        if self.boundary != BOUNDARY_SECCM_DIRECT_TORUS:
            raise ValueError(
                "SECCM runtime metadata requires boundary='seccm_direct_torus'"
            )
        temperature = float(electronic_temperature)
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError(
                "SECCM electronic_temperature must be finite and non-negative"
            )
        if type(truncated_electrostatics_acknowledged) is not bool:
            raise ValueError(
                "SECCM truncated_electrostatics_acknowledged must be boolean"
            )
        dimension = int(periodic_dimension)
        weighting = (
            None
            if three_center_weighting is None
            else str(three_center_weighting).strip().lower()
        )
        if self.method_key in {"om2", "om3"}:
            if weighting is None:
                weighting = "peintinger_eq13"
            if weighting not in _OMX_SECCM_THREE_CENTER_WEIGHTINGS:
                raise ValueError(
                    "OMx-SECCM three_center_weighting must be one of "
                    f"{sorted(_OMX_SECCM_THREE_CENTER_WEIGHTINGS)!r}"
                )
        elif weighting is not None:
            raise ValueError(
                "three_center_weighting metadata is only valid for "
                "OM2-/OM3-SECCM routes"
            )
        electrostatics_kernel = _seccm_electrostatics_kernel(
            dimension,
            electrostatics_family,
            self.method_key,
        )
        if truncated_electrostatics_acknowledged:
            if self.method_key not in {"om2", "om3"}:
                raise ValueError(
                    "truncated electrostatics acknowledgement is only valid "
                    "for OM2-/OM3-SECCM routes"
                )
            if electrostatics_kernel != "none":
                raise ValueError(
                    "truncated electrostatics acknowledgement requires "
                    "electrostatics kernel 'none'"
                )
        return replace(
            self,
            periodic_dimension=dimension,
            electrostatics_kernel=electrostatics_kernel,
            electronic_temperature=temperature,
            three_center_weighting=weighting,
            truncated_electrostatics_acknowledged=(
                truncated_electrostatics_acknowledged
            ),
            gfn2_hamiltonian_identity=None,
            gfn2_run_controls=None,
        )

    def with_gfn2_seccm_runtime(
        self,
        *,
        periodic_dimension: int,
        requested_electrostatics_family: str,
        resolved_electrostatics_family: str,
        requested_electronic_temperature: float,
        resolved_electronic_temperature: float,
        include_aes: bool,
        madelung_s_weighted: bool,
        madelung_no_self: bool,
        ewald_gamma_molecular_onsite: bool,
        molecular_delegated: bool,
        run_controls: GFN2SECCMRunControls,
    ) -> "SemiempiricalRoutePlan":
        """Attach complete GFN2-SECCM Hamiltonian provenance."""
        if (
            self.method_key != "gfn2_xtb"
            or self.boundary != BOUNDARY_SECCM_DIRECT_TORUS
        ):
            raise ValueError(
                "GFN2-SECCM Hamiltonian metadata requires the "
                "GFN2-SECCM route"
            )
        if not isinstance(
            run_controls,
            GFN2SECCMRunControls,
        ):
            raise ValueError(
                "GFN2-SECCM run_controls must be a "
                "GFN2SECCMRunControls instance"
            )
        dimension = int(periodic_dimension)
        resolved_plan = self.with_seccm_runtime(
            periodic_dimension=dimension,
            electrostatics_family=resolved_electrostatics_family,
            electronic_temperature=resolved_electronic_temperature,
        )
        identity = GFN2SECCMHamiltonianIdentity(
            schema_version=1,
            requested_electrostatics_family=requested_electrostatics_family,
            resolved_electrostatics_kernel=resolved_plan.electrostatics_kernel,
            requested_electronic_temperature=requested_electronic_temperature,
            resolved_electronic_temperature=resolved_electronic_temperature,
            include_aes=bool(include_aes),
            madelung_s_weighted=bool(madelung_s_weighted),
            madelung_no_self=bool(madelung_no_self),
            ewald_gamma_molecular_onsite=bool(
                ewald_gamma_molecular_onsite
            ),
            molecular_delegated=bool(molecular_delegated),
        )
        return replace(
            resolved_plan,
            gfn2_hamiltonian_identity=identity,
            gfn2_run_controls=run_controls,
        )

    def _periodic_electrostatics_entries(self) -> tuple[str, ...]:
        """Citations the Bloch-periodic second-order kernel itself incurs.

        Only the routes that actually assemble the lattice-summed shell
        gamma report a form here, so a plan built before the driver ran
        (``shell_gamma_form is None``) adds nothing.  Elstner 1998 is the
        source of the short-range gamma functional form; the Ewald entries
        are the dimensional channel the 1/R part is summed in.
        """
        if self.shell_gamma_form is None:
            return ()
        entries: list[str] = ["ewald_lattice_sum_1921"]
        if self.shell_gamma_form == "elstner":
            entries.append("elstner_scc_dftb_1998")
        if self.periodic_dimension == 2:
            entries.extend(
                ("parry_2d_ewald_1975", "de_leeuw_perram_2d_ewald_1979")
            )
        elif self.periodic_dimension == 1:
            entries.append("rozzi_wire_coulomb_2006")
        return tuple(entries)

    @property
    def citation_assemble_kwargs(self) -> dict[str, object]:
        """Sole route-specific citation inputs for this concrete plan.

        The job runner merges these values into its job-wide citation
        assembly. Keeping method identity, execution flags, runtime SECCM
        context, and route-specific extras together prevents the emission
        path from reconstructing only part of the plan's provenance.
        """
        if self.boundary != BOUNDARY_SECCM_DIRECT_TORUS:
            kwargs: dict[str, object] = {
                "method": self.method_key,
                "uses_integrals": self.method_key in {"dftb0", "scc_dftb"},
                "uses_scf": self.method_key != "dftb0",
                "periodic": self.boundary
                in {BOUNDARY_PERIODIC_GAMMA, BOUNDARY_PERIODIC_K},
                "electronic_temperature": self.electronic_temperature,
            }
            if self.method_key == "msindo" and self.variant == "nddo":
                kwargs["extra_entries"] = (
                    "dewar_thiel_1977",
                    "voigt_1973",
                )
            extras = self._periodic_electrostatics_entries()
            if extras:
                kwargs["extra_entries"] = (
                    tuple(kwargs.get("extra_entries", ())) + extras
                )
            return kwargs

        citation_method = {
            "msindo": "seccm",
            "dftb0": "dftb0_seccm",
            "scc_dftb": "scc_dftb_seccm",
            "pm6": "pm6_seccm",
            "om1": "om1_seccm",
            "om2": "om2_seccm",
            "om3": "om3_seccm",
            "gfn2_xtb": "gfn2_seccm",
        }[self.method_key]
        kwargs: dict[str, object] = {
            "method": citation_method,
            "uses_integrals": self.method_key in {"dftb0", "scc_dftb"},
            "uses_scf": self.method_key != "dftb0",
            # SECCM owns a cyclic finite group, not a Bloch-LCAO/spglib route.
            "periodic": False,
            "seccm_dimension": self.periodic_dimension,
            "seccm_electrostatics_kernel": self.electrostatics_kernel,
            "electronic_temperature": self.electronic_temperature,
        }
        if self.method_key == "msindo":
            kwargs["extra_entries"] = (
                "ahlswede_jug_msindo_1_1999",
                "ahlswede_jug_msindo_2_1999",
            )
        elif self.three_center_weighting == "janetzko_eq10":
            kwargs["extra_entries"] = ("janetzko_ccm_2008",)
        return kwargs

    @classmethod
    def from_request(
        cls,
        method: str,
        *,
        boundary: str | None = None,
        properties: Iterable[str] | None = None,
        charge: int = 0,
        multiplicity: int | None = 1,
        unrestricted: bool | None = None,
        nddo: bool = False,
        solvent=None,
        ccm_options=None,
    ) -> "SemiempiricalRoutePlan":
        """Build and validate a cheap route plan from public-facing inputs."""
        method_key, method_alias = _canonical_method(method)
        boundary_key = _canonical_boundary(boundary, method_alias)
        _validate_knobs(
            method_key=method_key,
            boundary=boundary_key,
            nddo=bool(nddo),
            solvent=solvent,
            ccm_options=ccm_options,
        )
        _validate_boundary(method_key, boundary_key)
        charge_value = int(charge)
        _validate_charge(
            method_key=method_key,
            boundary=boundary_key,
            charge=charge_value,
        )
        spin = _spin_label(
            multiplicity=multiplicity,
            unrestricted=unrestricted,
            method_alias=method_alias,
        )
        variant = _variant(
            method_key,
            method_alias=method_alias,
            nddo=bool(nddo),
            solvent=solvent,
            spin=spin,
        )
        _validate_spin(
            method_key=method_key,
            boundary=boundary_key,
            variant=variant,
            spin=spin,
        )
        props = _canonical_properties(properties)
        _validate_properties(
            props,
            method_key=method_key,
            boundary=boundary_key,
            variant=variant,
            spin=spin,
        )
        maturity, execution = _maturity_and_execution(
            method_key=method_key,
            boundary=boundary_key,
            variant=variant,
            properties=props,
        )
        return cls(
            method_key=method_key,
            method_family=_method_family(method_key),
            variant=variant,
            boundary=boundary_key,
            charge=charge_value,
            spin=spin,
            scc=_scc_policy(method_key),
            properties=props,
            maturity=maturity,
            parameter_scope=_parameter_scope(
                method_key=method_key,
                boundary=boundary_key,
                variant=variant,
            ),
            execution=execution,
            status_route=_status_route(
                method_key=method_key,
                boundary=boundary_key,
                variant=variant,
                properties=props,
            ),
        )


def is_semiempirical_method(method: str) -> bool:
    """Return whether *method* is a known semiempirical route spelling."""
    try:
        _canonical_method(method)
    except ValueError:
        return False
    return True


def plan_periodic_semiempirical_route(
    method: str | SemiempiricalRoutePlan,
    system,
    *,
    boundary: str = BOUNDARY_PERIODIC_GAMMA,
    properties: Iterable[str] | None = None,
) -> SemiempiricalRoutePlan:
    """Build or validate one route plan for a periodic system."""
    if not hasattr(system, "lattice") or not hasattr(system, "unit_cell"):
        raise TypeError(
            "periodic semiempirical route planning requires a PeriodicSystem."
        )

    multiplicity = int(getattr(system, "multiplicity", 1) or 1)
    unrestricted = multiplicity != 1
    try:
        unrestricted = unrestricted or int(system.n_electrons()) % 2 == 1
    except (AttributeError, TypeError, ValueError):
        pass

    if isinstance(method, SemiempiricalRoutePlan):
        expected = SemiempiricalRoutePlan.from_request(
            method.method_key,
            boundary=boundary,
            properties=method.properties if properties is None else properties,
            charge=int(getattr(system, "charge", 0) or 0),
            multiplicity=multiplicity,
            unrestricted=unrestricted,
        )
        if method != expected:
            raise ValueError(
                "supplied semiempirical route plan does not match the periodic "
                "system, boundary, or requested properties."
            )
        return method

    return SemiempiricalRoutePlan.from_request(
        method,
        boundary=boundary,
        properties=properties,
        charge=int(getattr(system, "charge", 0) or 0),
        multiplicity=multiplicity,
        unrestricted=unrestricted,
    )


__all__ = [
    "BOUNDARY_CCM_DIRECT_TORUS",
    "BOUNDARY_MOLECULE",
    "BOUNDARY_PERIODIC_GAMMA",
    "BOUNDARY_PERIODIC_K",
    "BOUNDARY_SECCM_DIRECT_TORUS",
    "EXECUTION_NATIVE",
    "EXECUTION_NATIVE_BATCHED_FD",
    "EXECUTION_PYTHON_ORCHESTRATION",
    "EXECUTION_UNSUPPORTED",
    "MATURITY_EXPERIMENTAL",
    "MATURITY_GATED_UNIMPLEMENTED",
    "MATURITY_MIXED_NATIVE",
    "MATURITY_NATIVE_FD",
    "MATURITY_PRODUCTION",
    "MOLECULAR_SEMIEMPIRICAL_METHODS",
    "PERIODIC_GAMMA_SEMIEMPIRICAL_METHODS",
    "SEMIEMPIRICAL_METHOD_ALIASES",
    "SEMIEMPIRICAL_METHODS",
    "GFN2SECCMHamiltonianIdentity",
    "GFN2SECCMRestartIdentity",
    "GFN2SECCMRunControls",
    "SemiempiricalRoutePlan",
    "is_semiempirical_method",
    "normalise_semiempirical_method",
    "plan_periodic_semiempirical_route",
]
