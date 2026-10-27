"""Bounded scaled fitting and conditional linear solves.

Weighted residuals and priors follow ForceBalance (10.1021/jz500737m).
Conditional elimination follows variable projection (10.1088/0266-5611/19/2/201).
Only box constraints, squared loss and electronic energy contrasts are admitted.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .adapters import EvaluationFailure
from .schema import digest


def _coordinates(parameters, linear):
    return [
        i
        for i, s in enumerate(parameters.specs)
        if s.linear == linear and not s.tied_to
    ]


def _expand(parameters, indexes, coordinates):
    values = list(parameters.values)
    for i, x in zip(indexes, coordinates):
        s = parameters.specs[i]
        values[i] = s.initial + s.scale * float(x)
        values[i] = min(s.upper, max(s.lower, values[i]))  # roundoff at bounds
    names = {s.name: i for i, s in enumerate(parameters.specs)}
    for i, s in enumerate(parameters.specs):
        if s.tied_to:
            values[i] = values[names[s.tied_to]]
    return parameters.replace(values)


def _bounds(parameters, indexes):
    lower, upper = [], []
    for i in indexes:
        s = parameters.specs[i]
        tied = [t for t in parameters.specs if t.tied_to == s.name]
        lo = max([s.lower] + [t.lower for t in tied])
        hi = min([s.upper] + [t.upper for t in tied])
        if lo >= hi:
            raise ValueError("ties leave no feasible interval")
        lower.append((lo - s.initial) / s.scale)
        upper.append((hi - s.initial) / s.scale)
    return np.array(lower), np.array(upper)


class FitProblem:
    def __init__(self, adapter, parameters, dataset):
        adapter.validate(parameters)
        self.adapter, self.initial, self.dataset = adapter, parameters, dataset
        self.identity = adapter.identity()
        self.sha256 = digest(
            {
                "adapter": self.identity,
                "parameters": asdict(parameters),
                "dataset": dataset.to_dict(),
                "objective": "squared-correlated-class-balanced-priors-v1",
            }
        )
        self.cache = {}
        self.evaluations = 0
        self.linear_diagnostics = None
        counts = Counter(o.category for o in dataset.observations)
        observations = dataset.observations
        records = {r.sha256: r for o in observations for r in o.records}
        keys = sorted(records)
        incidence = np.zeros((len(observations), len(keys)))
        key_index = {k: i for i, k in enumerate(keys)}
        for i, o in enumerate(observations):
            for r, c in zip(o.records, o.coefficients):
                incidence[i, key_index[r.sha256]] += c * r.uncertainty
        # Independent raw-result errors induce correlated contrast errors. The
        # declared tolerance floors keep this covariance positive definite.
        covariance = incidence @ incidence.T + np.diag(
            [o.tolerance**2 for o in observations]
        )
        importance = np.diag(
            [(o.weight / counts[o.category]) ** 0.5 for o in observations]
        )
        self.whitener = np.linalg.solve(np.linalg.cholesky(covariance), importance)
        self.targets = np.array([o.target for o in dataset.observations])

    def _energy(self, parameters, system, linear=False):
        key = (parameters.sha256, system.geometry_sha256, linear)
        if key not in self.cache:
            self.evaluations += 1
            try:
                value = (
                    self.adapter.linear_terms(parameters, system)
                    if linear
                    else self.adapter.energy(parameters, system)
                )
                numbers = [value[0], *value[1]] if linear else [value]
                if not np.all(np.isfinite(numbers)):
                    raise EvaluationFailure("non-finite model result")
            except Exception as exc:
                raise EvaluationFailure(
                    f"evaluation failed for system {system.geometry_sha256}: {exc}"
                ) from exc
            self.cache[key] = value
        return self.cache[key]

    def predictions(self, parameters):
        return np.array(
            [
                sum(
                    c * self._energy(parameters, r.system)
                    for c, r in zip(o.coefficients, o.records)
                )
                for o in self.dataset.observations
            ]
        )

    def residual(self, parameters):
        residual = list(self.whitener @ (self.predictions(parameters) - self.targets))
        residual.extend(
            (v - s.initial) / s.prior_sigma
            for s, v in zip(parameters.specs, parameters.values)
            if s.prior_sigma
        )
        return np.array(residual)

    def conditional(self, parameters):
        from scipy.optimize import lsq_linear

        indexes = _coordinates(parameters, True)
        if not indexes:
            return parameters
        all_linear = [i for i, s in enumerate(parameters.specs) if s.linear]
        rows, offsets = [], []
        for o in self.dataset.observations:
            row = np.zeros(len(all_linear))
            offset = 0.0
            for c, r in zip(o.coefficients, o.records):
                base, columns = self._energy(parameters, r.system, linear=True)
                if len(columns) != len(all_linear):
                    raise ValueError("adapter linear columns do not match specs")
                offset += c * base
                row += c * np.asarray(columns)
            offsets.append(offset)
            rows.append(row)
        matrix = np.array(rows)
        # Tied parameters share a column; priors still apply to each physical field.
        transform = np.zeros((len(all_linear), len(indexes)))
        initial = []
        for row, i in enumerate(all_linear):
            s = parameters.specs[i]
            root = next(
                (j for j in indexes if parameters.specs[j].name == s.tied_to), i
            )
            transform[row, indexes.index(root)] = parameters.specs[root].scale
            initial.append(parameters.specs[root].initial)
        design = matrix @ transform
        rhs = self.targets - np.asarray(offsets) - matrix @ initial
        design = self.whitener @ design
        rhs = self.whitener @ rhs
        self.linear_diagnostics = {
            "data_rank": int(np.linalg.matrix_rank(design)),
            "free_parameters": len(indexes),
            "singular_values": np.linalg.svd(design, compute_uv=False).tolist(),
            "scope": "scaled conditional data design; excludes priors",
        }
        prior_rows, prior_rhs = [], []
        for row, i in enumerate(all_linear):
            s = parameters.specs[i]
            if s.prior_sigma:
                prior_rows.append(transform[row] / s.prior_sigma)
                prior_rhs.append((s.initial - initial[row]) / s.prior_sigma)
        if prior_rows:
            design = np.vstack((design, prior_rows))
            rhs = np.r_[rhs, prior_rhs]
        result = lsq_linear(
            design,
            rhs,
            bounds=_bounds(parameters, indexes),
            tol=1e-12,
            lsmr_tol=1e-12,
            max_iter=500,
        )
        if not result.success:
            raise EvaluationFailure(
                "conditional linear solve failed: " + result.message
            )
        return _expand(parameters, indexes, result.x)

    def jacobian(self, parameters, step=1e-4):
        """Physical-coordinate central differences; refuse boundary clipping."""
        if not np.isfinite(step) or step <= 0:
            raise ValueError("positive finite derivative step required")
        columns = []
        for i, s in enumerate(parameters.specs):
            if s.tied_to:
                continue
            h = step * s.scale
            values = np.array(parameters.values)
            if values[i] - h < s.lower or values[i] + h > s.upper:
                raise ValueError("central derivative crosses a parameter bound")
            indexes = [i] + [
                j for j, t in enumerate(parameters.specs) if t.tied_to == s.name
            ]
            plus, minus = values.copy(), values.copy()
            plus[indexes] += h
            minus[indexes] -= h
            columns.append(
                (
                    self.residual(parameters.replace(plus))
                    - self.residual(parameters.replace(minus))
                )
                / (2 * h)
            )
        return np.array(columns).T

    def metrics(self, parameters):
        errors = self.predictions(parameters) - self.targets
        return {
            category: {
                "n": int(
                    sum(o.category == category for o in self.dataset.observations)
                ),
                "mae_hartree": float(
                    np.mean(
                        np.abs(
                            [
                                e
                                for e, o in zip(errors, self.dataset.observations)
                                if o.category == category
                            ]
                        )
                    )
                ),
                "rmse_hartree": float(
                    np.sqrt(
                        np.mean(
                            [
                                e * e
                                for e, o in zip(errors, self.dataset.observations)
                                if o.category == category
                            ]
                        )
                    )
                ),
            }
            for category in sorted({o.category for o in self.dataset.observations})
        }

    def fit(self, *, checkpoint=None, resume=False, max_nfev=100):
        from scipy.optimize import least_squares

        from vibeqc.output.formats.parameter_opt import write_parameter_opt

        if type(max_nfev) is not int or max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")
        start = self.initial
        if resume:
            if checkpoint is None:
                raise ValueError("resume requires a checkpoint")
            saved = json.loads(Path(checkpoint).read_text())
            if (
                saved.get("schema") != "vibeqc.parameter-opt.v1"
                or saved.get("problem_sha256") != self.sha256
            ):
                raise ValueError(
                    "checkpoint objective/dataset/runtime identity mismatch"
                )
            start = start.replace(saved["values"])
        best = None
        previous_evaluations = saved.get("evaluations", 0) if resume else 0

        def record(parameters, residual, status, message=""):
            nonlocal best
            loss = float(residual @ residual)
            if best is None or loss < best["loss"]:
                best = {
                    "schema": "vibeqc.parameter-opt.v1",
                    "problem_sha256": self.sha256,
                    "adapter": self.identity,
                    "dataset_sha256": self.dataset.sha256,
                    "parameter_specs": [asdict(s) for s in parameters.specs],
                    "values": list(parameters.values),
                    "loss": loss,
                    "status": status,
                    "message": message,
                    "evaluations": previous_evaluations + self.evaluations,
                    "citations": ["forcebalance_2014", "golub_pereyra_2003"],
                }
                if checkpoint:
                    write_parameter_opt(checkpoint, best)

        indexes = _coordinates(start, False)

        def objective(x):
            parameters = self.conditional(_expand(start, indexes, x))
            residual = self.residual(parameters)
            record(parameters, residual, "running")
            return residual

        try:
            if indexes:
                x0 = [
                    (start.values[i] - start.specs[i].initial) / start.specs[i].scale
                    for i in indexes
                ]
                result = least_squares(
                    objective,
                    x0,
                    bounds=_bounds(start, indexes),
                    jac="3-point",
                    max_nfev=max_nfev,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                status = "converged" if result.success else "budget_exhausted"
                message = result.message
            else:
                parameters = self.conditional(start)
                record(parameters, self.residual(parameters), "running")
                status, message = (
                    "converged",
                    "conditional box-constrained least squares",
                )
            fitted = start.replace(best["values"])
            best.update(
                status=status,
                message=message,
                baseline_metrics=self.metrics(self.initial),
                fitted_metrics=self.metrics(fitted),
            )
            best["linear_diagnostics"] = self.linear_diagnostics
            best["nonlinear_diagnostics"] = (
                {
                    "data_rank": int(
                        np.linalg.matrix_rank(result.jac[: len(self.targets)])
                    ),
                    "free_parameters": len(indexes),
                    "singular_values": np.linalg.svd(
                        result.jac[: len(self.targets)], compute_uv=False
                    ).tolist(),
                    "scope": "solver-final scaled reduced data Jacobian; excludes priors",
                }
                if indexes
                else None
            )
            best["evaluations"] = previous_evaluations + self.evaluations
            best["restart_kind"] = (
                "incumbent warm start; optimizer history and cache are not restored"
            )

        except EvaluationFailure as exc:
            if checkpoint:
                if best is None:
                    best = {
                        "schema": "vibeqc.parameter-opt.v1",
                        "problem_sha256": self.sha256,
                        "adapter": self.identity,
                        "dataset_sha256": self.dataset.sha256,
                        "parameter_specs": [asdict(s) for s in start.specs],
                        "values": list(start.values),
                        "loss": None,
                    }
                best.update(
                    status="failed",
                    message=str(exc),
                    evaluations=previous_evaluations + self.evaluations,
                )
                write_parameter_opt(checkpoint, best)
            raise
        if checkpoint:
            write_parameter_opt(checkpoint, best)
        return best
