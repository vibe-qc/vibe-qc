"""Explicit periodic comparison requests for the shared regression runner."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from ..methods.catalog import METHODS


def validated_kmesh(value) -> tuple[int, int, int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(n) is not int or n <= 0 for n in value)):
        raise ValueError("kmesh must contain three positive integers")
    return tuple(value)


def validated_reference_version(reference_code, value):
    if value is not None and (reference_code != "crystal" or not isinstance(value, str)
                              or not re.fullmatch(r"CRYSTAL[0-9]+", value)):
        raise ValueError("reference_version requires reference_code=crystal and a token such as CRYSTAL23")
    return value


STATE_CONVENTION = "per primitive cell, identical on every Born-von Karman cell"


def periodic_state_request(spec, method, *, code: str) -> dict:
    """Resolve the electronic state a code must run, or refuse it (#647).

    Refusals happen before execution so a magnetic or charged request is
    never run as a restricted, zero-spin or neutral case under its label.
    """
    n_alpha, n_beta = spec.electron_counts()
    spins = tuple(spec.atomic_spins or ())
    polarized = spec.multiplicity != 1 or any(spins)
    if spec.charge != 0:
        raise ValueError(
            f"{spec.id}: charge {spec.charge} per cell needs a declared "
            "neutralizing-background convention, which the comparison suite "
            "does not define")
    if any(spec.magnetic_moments or ()):
        raise ValueError(
            f"{spec.id}: magnetic_moments are not consumed by any runner; "
            "request the state with multiplicity and atomic_spins")
    if method.scf in ("rhf", "rks") and (polarized or spec.is_open_shell):
        raise ValueError(
            f"{spec.id}: {method.scf.upper()} cannot represent multiplicity "
            f"{spec.multiplicity} or atomic spins; use an unrestricted method")
    if spec.is_open_shell and not polarized:
        raise ValueError(
            f"{spec.id}: is_open_shell declares a polarized state; set "
            "multiplicity or atomic_spins")
    if any(spins) and spec.default_initial_guess.upper() != "SAD":
        raise ValueError(f"{spec.id}: atomic_spins require the SAD initial guess")
    if any(spins) and code != "vibeqc":
        raise ValueError(f"{spec.id}: the {code} runner has no atomic-spin initial state")
    return {"charge": spec.charge, "multiplicity": spec.multiplicity,
            "n_alpha": n_alpha, "n_beta": n_beta,
            "atomic_spins": list(spins) if any(spins) else None,
            "convention": STATE_CONVENTION}


def state_overrides(spec) -> dict:
    """The state fields carried across process isolation."""
    return {"charge": spec.charge, "multiplicity": spec.multiplicity,
            "atomic_spins": None if spec.atomic_spins is None else list(spec.atomic_spins)}


def apply_state_overrides(spec, overrides: dict):
    from dataclasses import replace
    spins = overrides.get("atomic_spins")
    return replace(spec, charge=overrides.get("charge", spec.charge),
                   multiplicity=overrides.get("multiplicity", spec.multiplicity),
                   atomic_spins=None if spins is None else tuple(spins))


def state_tag(state: dict) -> str:
    """Filesystem-safe label suffix for an explicit state request."""
    parts = []
    if "charge" in state:
        parts.append(f"q{state['charge']:+d}".replace("+", "p").replace("-", "m"))
    if "multiplicity" in state:
        parts.append(f"mult{state['multiplicity']}")
    if "atomic_spins" in state:
        parts.append("spins" + "".join({1: "u", -1: "d", 0: "0"}[s]
                                       for s in state["atomic_spins"]))
    return "_".join(parts)


def case_tag(state: dict, lattice: dict) -> str:
    """Label suffix for explicit state and lattice-cutoff requests."""
    parts = [state_tag(state)] if state else []
    for key, short in (("lattice_cutoff_bohr", "cut"), ("nuclear_cutoff_bohr", "ncut")):
        if key in lattice:
            parts.append(f"{short}{lattice[key]:g}".replace(".", "p"))
    return "_".join(parts)


def apply_lattice_cutoffs(spec, lattice: dict):
    """Declared BIPOLE/lattice-sum cutoffs replace the system defaults."""
    from dataclasses import replace
    cutoff = lattice.get("lattice_cutoff_bohr", spec.default_cutoff_bohr)
    nuclear = lattice.get("nuclear_cutoff_bohr", spec.default_nuclear_cutoff_bohr)
    if nuclear < cutoff:
        raise ValueError(f"nuclear_cutoff_bohr {nuclear:g} is below lattice_cutoff_bohr {cutoff:g}")
    return replace(spec, default_cutoff_bohr=float(cutoff), default_nuclear_cutoff_bohr=float(nuclear))


def lattice_overrides(spec) -> dict:
    """The cutoffs carried across process isolation."""
    return {"lattice_cutoff_bohr": spec.default_cutoff_bohr,
            "nuclear_cutoff_bohr": spec.default_nuclear_cutoff_bohr}


@dataclass(frozen=True)
class PeriodicCaseRequest:
    system_id: str
    basis: str
    method_id: str
    kmesh: tuple[int, int, int]
    tolerance_ha: float | None = None  # None only for legacy expected files.
    tolerance_rationale: str = ""
    reference_code: str = "pyscf"
    notes: str = ""
    reference_version: str | None = None
    charge: int | None = None  # None keeps the system module's state.
    multiplicity: int | None = None
    atomic_spins: tuple[int, ...] | None = None
    lattice_cutoff_bohr: float | None = None  # None keeps the system default.
    nuclear_cutoff_bohr: float | None = None

    def lattice(self) -> dict:
        return {k: v for k, v in (("lattice_cutoff_bohr", self.lattice_cutoff_bohr),
                                  ("nuclear_cutoff_bohr", self.nuclear_cutoff_bohr))
                if v is not None}

    def tag(self) -> str:
        return case_tag(self.state(), self.lattice())

    def state(self) -> dict:
        return {k: v for k, v in (("charge", self.charge),
                                  ("multiplicity", self.multiplicity),
                                  ("atomic_spins", self.atomic_spins)) if v is not None}


def load_periodic_matrix(path: Path, *, load_system):
    """Validate all rows before the caller creates output or dispatches work.

    The matrix declares a comparison target, not accepted scientific evidence.
    Unknown fields fail closed so charge, shift or other unsupported physical
    controls cannot be silently dropped from a purported matched request.
    """
    data = json.loads(Path(path).read_text())
    fields = {"schema_version", "jk_method", "cases"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError(f"periodic matrix requires exactly {sorted(fields)}")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported periodic matrix schema_version")
    if data["jk_method"] not in ("gdf", "bipole"):
        raise ValueError("periodic matrix jk_method must be gdf or bipole")
    if not isinstance(data["cases"], list) or not data["cases"]:
        raise ValueError("periodic matrix cases must be a nonempty list")
    required = {"system_id", "basis", "method_id", "kmesh", "tolerance_ha",
                "tolerance_rationale", "reference_code"}
    optional = {"notes", "reference_version", "charge", "multiplicity", "atomic_spins",
                "lattice_cutoff_bohr", "nuclear_cutoff_bohr"}
    cases, identities = [], set()
    for index, row in enumerate(data["cases"], 1):
        if (not isinstance(row, dict) or not required <= set(row)
                or set(row) - required - optional):
            raise ValueError(f"periodic matrix row {index}: invalid fields")
        for name in ("system_id", "basis", "method_id", "tolerance_rationale", "reference_code"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"periodic matrix row {index}: {name} must be nonempty text")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", row["system_id"]):
            raise ValueError("system_id must name a periodic system module")
        basis = row["basis"]
        if (basis in (".", "..") or any(c in basis for c in "/\\")
                or any(ord(c) < 32 for c in basis)):
            raise ValueError("basis must be a registry name, not a path")
        method = METHODS.get(row["method_id"])
        if (method is None or not method.periodic or method.post is not None
                or method.df or method.scf not in ("rhf", "uhf", "rks", "uks")):
            raise ValueError(f"unsupported periodic matrix method {row['method_id']!r}")
        if row["reference_code"] not in ("pyscf", "crystal", "cp2k"):
            raise ValueError("reference_code must be pyscf, crystal or cp2k")
        validated_reference_version(row["reference_code"], row.get("reference_version"))
        tol = row["tolerance_ha"]
        try:
            finite_tolerance = type(tol) in (int, float) and math.isfinite(tol)
        except OverflowError:
            finite_tolerance = False
        if not finite_tolerance or tol < 0:
            raise ValueError("tolerance_ha must be finite and nonnegative")
        if not isinstance(row.get("notes", ""), str):
            raise ValueError("notes must be text")
        mesh = validated_kmesh(row["kmesh"])
        if "atomic_spins" in row:
            if not isinstance(row["atomic_spins"], list):
                raise ValueError(f"periodic matrix row {index}: atomic_spins must be a list")
            row = {**row, "atomic_spins": tuple(row["atomic_spins"])}
        for key in ("lattice_cutoff_bohr", "nuclear_cutoff_bohr"):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float)
                                      or not math.isfinite(value) or value <= 0):
                raise ValueError(f"periodic matrix row {index}: {key} must be a positive number")
        request = PeriodicCaseRequest(**{**row, "kmesh": mesh})
        spec = apply_state_overrides(load_system(row["system_id"]), request.state())
        spec = apply_lattice_cutoffs(spec, request.lattice())
        for code in {"vibeqc", row["reference_code"]}:
            periodic_state_request(spec, method, code=code)
        state = (spec.charge, spec.multiplicity, spec.atomic_spins,
                 spec.default_cutoff_bohr, spec.default_nuclear_cutoff_bohr)
        identity = (row["system_id"], basis, row["method_id"], mesh, state)
        if identity in identities:
            raise ValueError(f"duplicate periodic matrix case {identity[:4]}")
        identities.add(identity)
        cases.append(request)
    return data["jk_method"], cases
