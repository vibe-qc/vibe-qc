"""Qualified energy references, balanced contrasts and leakage-safe splits."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass

from .schema import System, canonical, digest, finite


@dataclass(frozen=True)
class ReferenceProtocol:
    family: str
    method: str
    basis: str
    code: str
    version: str
    settings_json: str
    energy_convention: str = "electronic"
    unit: str = "hartree"

    def __post_init__(self):
        if self.family not in {
            "dft",
            "hybrid",
            "dlpno",
            "semiempirical",
            "synthetic",
            "literature",
        }:
            raise ValueError("unknown protocol family")
        if not all((self.method, self.basis, self.code, self.version)):
            raise ValueError("complete reference identity required")
        if self.unit != "hartree" or self.energy_convention != "electronic":
            raise ValueError("P0-P3 require electronic energies in hartree")
        settings = json.loads(self.settings_json)
        if not isinstance(settings, dict):
            raise TypeError("settings must be a JSON object")
        required = {"scf", "relativity", "ecp", "state_protocol"}
        if self.family in {"dft", "hybrid"}:
            required |= {"grid", "dispersion", "functional"}
        if self.family == "hybrid":
            required |= {"exchange"}
        if self.family == "dlpno":
            required |= {"pno", "triples", "frozen_core", "aux_basis"}
        if self.family not in {"synthetic", "literature"} and any(
            k not in settings or settings[k] is None or settings[k] == ""
            for k in required
        ):
            raise ValueError(f"missing protocol settings: {sorted(required)}")
        object.__setattr__(self, "settings_json", canonical(settings))

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class ReferenceRecord:
    system: System
    protocol: ReferenceProtocol
    energy: float
    uncertainty: float
    raw_sha256: str
    source_identity: str
    converged: bool = True

    def __post_init__(self):
        finite(self.energy, "energy")
        finite(self.uncertainty, "uncertainty")
        if self.uncertainty < 0 or self.converged is not True:
            raise ValueError("reference must be converged with nonnegative uncertainty")
        if len(self.raw_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.raw_sha256
        ):
            raise ValueError("raw result SHA256 required")
        if not self.source_identity:
            raise ValueError("source/runtime identity required")

    @property
    def sha256(self):
        return digest(asdict(self))


def paired_delta(low: ReferenceRecord, high: ReferenceRecord):
    if low.system.geometry_sha256 != high.system.geometry_sha256:
        raise ValueError(
            "paired references require identical geometry, charge and state"
        )
    if low.system.parent != high.system.parent:
        raise ValueError("paired fidelities must share a parent group")
    if low.protocol.sha256 == high.protocol.sha256:
        raise ValueError("paired protocols must differ")
    return {
        "delta_hartree": high.energy - low.energy,
        "uncertainty_hartree": math.hypot(low.uncertainty, high.uncertainty),
        "low_sha256": low.sha256,
        "high_sha256": high.sha256,
        "covariance_assumption": "independent reference errors",
    }


@dataclass(frozen=True)
class Observation:
    """An atom/charge-balanced contrast; reference errors keyed by record identity."""

    name: str
    records: tuple[ReferenceRecord, ...]
    coefficients: tuple[float, ...]
    category: str = "relative_energy"
    tolerance: float = 0.001
    weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(
            self, "coefficients", tuple(float(v) for v in self.coefficients)
        )
        if (
            not self.name
            or not self.category
            or len(self.records) < 2
            or len(self.records) != len(self.coefficients)
        ):
            raise ValueError(
                "a named contrast needs at least two records and coefficients"
            )
        for v in (*self.coefficients, self.tolerance, self.weight):
            finite(v, "contrast scalar")
        if self.tolerance <= 0 or self.weight <= 0 or not any(self.coefficients):
            raise ValueError(
                "positive tolerance/weight and nonzero coefficients required"
            )
        if len({r.protocol.sha256 for r in self.records}) != 1:
            raise ValueError("contrast mixes reference protocols")
        if len({r.sha256 for r in self.records}) != len(self.records):
            raise ValueError("combine duplicate records before constructing a contrast")
        balance = Counter()
        charge = 0.0
        for r, c in zip(self.records, self.coefficients):
            for z in r.system.atomic_numbers:
                balance[z] += c
            charge += c * r.system.charge
        if any(abs(v) > 1e-10 for v in balance.values()) or abs(charge) > 1e-10:
            raise ValueError(
                "unbalanced energy gauge: element and charge balance required"
            )

    @property
    def target(self):
        return sum(c * r.energy for c, r in zip(self.coefficients, self.records))

    @property
    def sigma(self):
        return math.sqrt(
            self.tolerance**2
            + sum(
                (c * r.uncertainty) ** 2
                for c, r in zip(self.coefficients, self.records)
            )
        )

    @property
    def parents(self):
        return frozenset(r.system.parent for r in self.records)


@dataclass(frozen=True)
class Dataset:
    observations: tuple[Observation, ...]

    def __post_init__(self):
        object.__setattr__(self, "observations", tuple(self.observations))
        if not self.observations or len({o.name for o in self.observations}) != len(
            self.observations
        ):
            raise ValueError("nonempty dataset with unique observation names required")
        # Alias geometry labels cannot bypass grouped splitting.
        parents = {}
        contrasts = set()
        for o in self.observations:
            terms = sorted(
                (r.sha256, c) for r, c in zip(o.records, o.coefficients) if c
            )
            key = tuple((sha, c / terms[0][1]) for sha, c in terms)
            if key in contrasts:
                raise ValueError("duplicate contrast")
            contrasts.add(key)
            for r in o.records:
                geom = r.system.geometry_sha256
                if geom in parents and parents[geom] != r.system.parent:
                    raise ValueError("same geometry has different parent labels")
                parents[geom] = r.system.parent

    @property
    def sha256(self):
        return digest(asdict(self))

    def to_dict(self):
        return {"schema": 1, "observations": [asdict(o) for o in self.observations]}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != 1:
            raise ValueError("unsupported dataset schema")
        observations = []
        for row in data["observations"]:
            row = dict(row)
            records = []
            for record in row.pop("records"):
                record = dict(record)
                records.append(
                    ReferenceRecord(
                        System(**record.pop("system")),
                        ReferenceProtocol(**record.pop("protocol")),
                        **record,
                    )
                )
            observations.append(Observation(records=tuple(records), **row))
        return cls(tuple(observations))

    def split(self, fractions=(0.7, 0.15, 0.15), seed=0):
        """Deterministic connected-parent splits; small splits can be empty."""
        if (
            len(fractions) != 3
            or any(f < 0 or not math.isfinite(f) for f in fractions)
            or not math.isclose(sum(fractions), 1)
        ):
            raise ValueError(
                "three nonnegative split fractions summing to one required"
            )
        groups = {p: {p} for o in self.observations for p in o.parents}
        for o in self.observations:
            connected = set().union(*(groups[p] for p in o.parents))
            for p in connected:
                groups[p] = connected
        assignment = {}
        for p, component in groups.items():
            u = int(digest([seed, sorted(component)])[:16], 16) / 2**64
            assignment[p] = (
                "train"
                if u < fractions[0]
                else "validation"
                if u < sum(fractions[:2])
                else "test"
            )
        return {
            name: tuple(
                o
                for o in self.observations
                if assignment[next(iter(o.parents))] == name
            )
            for name in ("train", "validation", "test")
        }
