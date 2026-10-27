"""Isolated native molecular adapters; no external QC program imports."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .schema import ParameterSpec, digest


class ModelAdapter(Protocol):
    """Minimal adapter contract; linear_terms is required for linear specs."""

    def capabilities(self) -> dict: ...
    def validate(self, parameters) -> None: ...
    def identity(self) -> dict: ...
    def energy(self, parameters, system) -> float: ...


class EvaluationFailure(RuntimeError):
    """A failed candidate aborts the fit; no residual rows are removed."""


def runtime_identity():
    from vibeqc import _vibeqc_core

    root = Path(__file__).resolve().parent.parent
    sources = sorted((root / "parameter_opt").glob("*.py"))
    sources += sorted((root / "semiempirical").rglob("*.py"))
    sources += sorted((root / "semiempirical").rglob("*.toml"))
    sources += sorted((root / "semiempirical").rglob("*.json"))
    return {
        "native_sha256": hashlib.sha256(
            Path(_vibeqc_core.__file__).read_bytes()
        ).hexdigest(),
        "source_sha256": digest(
            [
                (str(p.relative_to(root)), hashlib.sha256(p.read_bytes()).hexdigest())
                for p in sources
            ]
        ),
    }


def _admit(system):
    if system.multiplicity != 1 or system.state != "ground":
        raise EvaluationFailure(
            "initial native adapters support closed-shell ground-state molecules only"
        )


@dataclass(frozen=True)
class DFTBRepulsion:
    """Fit Born-Mayer amplitudes with fixed positive inverse-bohr decays.

    All other pairs retain the built-in parameters. This is a preview model,
    not tabulated Slater-Koster DFTB. See Porezag, DOI 10.1103/PhysRevB.51.12947
    and the documented in-house Wolfsberg-Helmholtz approximation.
    """

    pairs: tuple[tuple[int, int, float], ...]

    def __post_init__(self):
        pairs = tuple(
            (min(a, b), max(a, b), float(decay)) for a, b, decay in self.pairs
        )
        if not pairs or len({(a, b) for a, b, _ in pairs}) != len(pairs):
            raise ValueError("unique repulsive pairs required")
        if any(
            type(a) is not int
            or type(b) is not int
            or not 1 <= a <= b <= 118
            or not math.isfinite(d)
            or d <= 0
            for a, b, d in pairs
        ):
            raise ValueError("valid elements and positive finite decays required")
        object.__setattr__(self, "pairs", pairs)

    @property
    def names(self):
        return tuple(f"repulsion.{a}.{b}" for a, b, _ in self.pairs)

    def specs(self, initial=1.0, upper=100.0):
        return tuple(
            ParameterSpec(n, "hartree", initial, 0.0, upper, linear=True)
            for n in self.names
        )

    def validate(self, parameters):
        if tuple(s.name for s in parameters.specs) != self.names or any(
            not s.linear or s.unit != "hartree" for s in parameters.specs
        ):
            raise ValueError(
                "DFTB adapter requires its ordered linear amplitude specs in hartree"
            )

    def capabilities(self):
        return {
            "boundary": "molecule",
            "observables": ["energy"],
            "linear_terms": True,
            "spin": "closed-shell",
            "maturity": "preview",
        }

    def identity(self):
        from vibeqc.semiempirical.parameters import default_parameters

        return {
            "adapter": "dftb-born-mayer-v1",
            "pairs": self.pairs,
            "maturity": "preview",
            "baseline_sha256": default_parameters().content_sha256(),
            **runtime_identity(),
        }

    def make_model(self, parameters, system):
        self.validate(parameters)
        _admit(system)
        from vibeqc.semiempirical.dftb0 import DFTB0Model
        from vibeqc.semiempirical.parameters import default_parameters

        params = default_parameters()
        for (a, b, d), value in zip(self.pairs, parameters.values):
            params.set_repulsive_pair_analytic(a, b, value, d)
        return DFTB0Model(system.molecule(), params=params)

    def energy(self, parameters, system):
        return float(self.make_model(parameters, system).energy())

    def linear_terms(self, parameters, system):
        # Evaluate the native model at the current amplitudes, then remove the
        # exact additive term. No zero value outside user bounds is required.
        columns = []
        for a, b, decay in self.pairs:
            value = 0.0
            for i, zi in enumerate(system.atomic_numbers):
                for j in range(i):
                    if tuple(sorted((zi, system.atomic_numbers[j]))) == (a, b):
                        distance = math.dist(system.positions[i], system.positions[j])
                        value += math.exp(-decay * distance)
            columns.append(value)
        offset = self.energy(parameters, system) - sum(
            v * b for v, b in zip(parameters.values, columns)
        )
        return offset, tuple(columns)


@dataclass(frozen=True)
class PM6Block:
    """Small eV-valued electronic block; published PM6 remains experimental.

    Parameters originate in the existing attributed MOPAC loader. Algorithm:
    Stewart (2007), DOI 10.1007/s00894-007-0233-4. No heat-of-formation claim.
    """

    fields: tuple[tuple[int, str], ...]
    max_iter: int = 200
    conv_tol: float = 1e-9

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(tuple(v) for v in self.fields))
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("unique PM6 fields required")
        if any(
            z not in {1, 6, 7, 8, 9} or field not in {"uss", "betas", "gss"}
            for z, field in self.fields
        ):
            raise ValueError("initial PM6 block supports uss/betas/gss for H,C,N,O,F")
        if self.max_iter <= 0 or not math.isfinite(self.conv_tol) or self.conv_tol <= 0:
            raise ValueError("invalid SCF controls")

    @property
    def names(self):
        return tuple(f"pm6.{z}.{field}" for z, field in self.fields)

    def specs(self, width=2.0):
        if not math.isfinite(width) or width <= 0:
            raise ValueError("positive finite width required")
        from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

        baseline = load_pm6_params()
        result = []
        for name, (z, field) in zip(self.names, self.fields):
            value = float(getattr(baseline.element_data(z), field))
            lower = max(1e-6, value - width) if field == "gss" else value - width
            result.append(
                ParameterSpec(name, "eV", value, lower, value + width, scale=width)
            )
        return tuple(result)

    def validate(self, parameters):
        if tuple(s.name for s in parameters.specs) != self.names or any(
            s.linear or s.unit != "eV" for s in parameters.specs
        ):
            raise ValueError("PM6 adapter requires its ordered nonlinear eV specs")
        if any(
            field == "gss" and s.lower <= 0
            for (z, field), s in zip(self.fields, parameters.specs)
        ):
            raise ValueError("PM6 gss bounds must be positive")

    def capabilities(self):
        return {
            "boundary": "molecule",
            "observables": ["energy"],
            "linear_terms": False,
            "spin": "closed-shell",
            "maturity": "experimental",
        }

    def identity(self):
        from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

        return {
            "adapter": "pm6-block-v1",
            "fields": self.fields,
            "max_iter": self.max_iter,
            "conv_tol": self.conv_tol,
            "maturity": "experimental",
            "parameter_license": "Apache-2.0",
            "parameter_notice": "Molecular Orbital PACkage (MOPAC). Copyright 2021 Virginia Polytechnic Institute and State University. Licensed under the Apache License, Version 2.0.",
            "parameter_source": "https://github.com/openmopac/mopac",
            "parameter_source_doi": "10.21105/joss.08025",
            "baseline_sha256": load_pm6_params().content_sha256(),
            **runtime_identity(),
        }

    def make_model(self, parameters, system):
        self.validate(parameters)
        _admit(system)
        if not set(system.atomic_numbers) <= {1, 6, 7, 8, 9}:
            raise EvaluationFailure("initial PM6 block supports H,C,N,O,F only")
        from vibeqc.semiempirical.methods.pm6 import PM6Model
        from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

        params = load_pm6_params()  # fresh element/diatomic tables per candidate
        for (z, field), value in zip(self.fields, parameters.values):
            element = params.element_data(z)  # detached native snapshot
            setattr(element, field, value)
            params.add_element(element)
        return PM6Model(
            system.molecule(),
            params=params,
            max_iter=self.max_iter,
            conv_tol=self.conv_tol,
        )

    def energy(self, parameters, system):
        model = self.make_model(parameters, system)
        value = float(model.energy())
        if not model.converged:
            raise EvaluationFailure("PM6 SCF did not converge")
        return value
