"""Immutable, canonical parameter-opt records (schema version 1)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(value, name):
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    unit: str
    initial: float
    lower: float
    upper: float
    scale: float = 1.0
    prior_sigma: float | None = None
    linear: bool = False
    tied_to: str | None = None

    def __post_init__(self):
        if not self.name or not self.unit:
            raise ValueError("parameter name and unit are required")
        for key in ("initial", "lower", "upper", "scale"):
            finite(getattr(self, key), key)
        if not self.lower < self.upper or not self.lower <= self.initial <= self.upper:
            raise ValueError("invalid parameter bounds or initial value")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.prior_sigma is not None:
            finite(self.prior_sigma, "prior_sigma")
            if self.prior_sigma <= 0:
                raise ValueError("prior_sigma must be positive")


@dataclass(frozen=True)
class ParameterSet:
    specs: tuple[ParameterSpec, ...]
    values: tuple[float, ...]

    def __post_init__(self):
        object.__setattr__(self, "specs", tuple(self.specs))
        object.__setattr__(self, "values", tuple(float(v) for v in self.values))
        if len(self.specs) != len(self.values) or not self.specs:
            raise ValueError("one value per parameter is required")
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise ValueError("duplicate parameter name")
        for s, v in zip(self.specs, self.values):
            finite(v, s.name)
            if not s.lower <= v <= s.upper:
                raise ValueError(f"{s.name} is outside bounds")
            if s.tied_to:
                if s.tied_to not in names or s.tied_to == s.name:
                    raise ValueError("invalid tie")
                parent = self.specs[names.index(s.tied_to)]
                if parent.tied_to or parent.unit != s.unit or parent.linear != s.linear:
                    raise ValueError(
                        "ties must refer directly to a compatible free parameter"
                    )
                if v != self.values[names.index(s.tied_to)]:
                    raise ValueError("tied values differ")

    @classmethod
    def initial(cls, specs):
        specs = tuple(specs)
        return cls(specs, tuple(s.initial for s in specs))

    def replace(self, values):
        return ParameterSet(self.specs, tuple(values))

    def to_dict(self):
        return {"schema": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, record):
        if record.get("schema") != 1:
            raise ValueError("unsupported parameter schema")
        return cls(
            tuple(ParameterSpec(**s) for s in record["specs"]), tuple(record["values"])
        )

    @property
    def mapping(self):
        return {s.name: v for s, v in zip(self.specs, self.values)}

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class System:
    """Molecular coordinates in bohr; parent labels define leakage groups."""

    parent: str
    atomic_numbers: tuple[int, ...]
    positions: tuple[tuple[float, float, float], ...]
    charge: int = 0
    multiplicity: int = 1
    state: str = "ground"

    def __post_init__(self):
        object.__setattr__(self, "atomic_numbers", tuple(self.atomic_numbers))
        object.__setattr__(
            self, "positions", tuple(tuple(float(v) for v in r) for r in self.positions)
        )
        if not self.parent or not self.state or not self.atomic_numbers:
            raise ValueError("system, parent and state must be specified")
        if len(self.positions) != len(self.atomic_numbers):
            raise ValueError("one position per atom required")
        if any(type(z) is not int or not 1 <= z <= 118 for z in self.atomic_numbers):
            raise ValueError("invalid atomic number")
        if (
            type(self.charge) is not int
            or type(self.multiplicity) is not int
            or self.multiplicity < 1
        ):
            raise ValueError("integer charge and positive multiplicity required")
        electrons = sum(self.atomic_numbers) - self.charge
        if electrons < self.multiplicity - 1 or (electrons - self.multiplicity + 1) % 2:
            raise ValueError("charge/multiplicity inconsistent with electron count")
        for row in self.positions:
            if len(row) != 3 or not all(math.isfinite(v) for v in row):
                raise ValueError("positions must be finite Cartesian triples in bohr")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("coincident atoms")

    @property
    def geometry_sha256(self):
        # Parent is a splitting label, not part of physical identity.
        record = asdict(self)
        record.pop("parent")
        return digest(record)

    def molecule(self):
        from vibeqc import Atom, Molecule

        return Molecule(
            [Atom(z, r) for z, r in zip(self.atomic_numbers, self.positions)],
            self.charge,
            self.multiplicity,
        )
