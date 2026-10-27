"""Array-only, finite-setting diagnostics for experimental chi operators.

These checks test algebra and supplied references, not physical accuracy. A
successive difference or a residual is a measurement, never an error bound.
Matrices are AO covariant operators and contravariant Hermitian densities in
matching frames. Weights are positive Brillouin-zone weights summing to one;
all traces and energies below are per primitive cell. No SCF is performed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Callable

import numpy as np


def factor_conjugation_measurements(factors, partners, *, column_tile=64):
    """Separate fixed-gauge conjugation from complete Coulomb block fidelity.

    Inputs are (auxiliary, pair-column) matrices for q and -q. The caller
    must align conjugate AO/character columns and concatenate *all* pairs
    sharing q; testing each pair separately misses cross-pair operators.
    For self-inverse q, partners is the corresponding column permutation.

    With Sun et al., JCP 147, 164119 (2017), Sec. II, Eqs. (13), (16)-(18),
    the real-AO Fourier convention gives M(-q)=M(q)*, T(-q)=T(q)* and
    L(-q)=L(q)* for L=M_ret^(-1/2) T in original auxiliary coordinates.
    The operator is L_x^T L_y*, invariant under a common unitary auxiliary
    rotation within each transfer. Its conjugation defect is distinct from
    entry fidelity. Neither measurement alone certifies physical accuracy.

    Every column, auxiliary component and cross-column block is included.
    Tiling bounds the temporary Gram matrices, not the borrowed inputs.
    No projection, alignment, rank selection, tolerance or repair is applied.
    """
    a, b = np.asarray(factors), np.asarray(partners)
    if (a.ndim != 2 or a.shape != b.shape or min(a.shape) < 1
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("finite matching (auxiliary, pair-column) matrices required")
    if (isinstance(column_tile, (bool, np.bool_))
            or not isinstance(column_tile, (int, np.integer)) or column_tile < 1):
        raise ValueError("column_tile must be a positive integer")
    dtype = np.result_type(a.dtype, b.dtype, np.complex128)
    a, b = a.astype(dtype, copy=False), b.astype(dtype, copy=False)
    entry, operator = 0., 0.
    for lo in range(0, a.shape[1], column_tile):
        x, y = a[:, lo:lo+column_tile], b[:, lo:lo+column_tile]
        entry = max(entry, float(np.max(np.abs(x-y.conj()))))
        for start in range(0, a.shape[1], column_tile):
            ax, bx = a[:, start:start+column_tile], b[:, start:start+column_tile]
            operator = max(operator, float(np.max(np.abs(
                x.T @ ax.conj() - y.conj().T @ bx))))
    return dict(factor_conjugation_max=entry,
                real_representation_loss_lower_bound=entry/2,
                coulomb_conjugation_max=operator,
                auxiliary_components=a.shape[0], pair_columns=a.shape[1])


@dataclass(frozen=True)
class OperatorConvention:
    """Explicit labels supplied by the caller; labels are not verification."""

    source_identity: str
    executed_domain_identity: str
    kernel: str
    q0_convention: str
    density_convention: str
    normalization: str = "primitive_cell_bz_weights_sum_one"

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be an explicit nonempty label")
        if self.normalization != "primitive_cell_bz_weights_sum_one":
            raise ValueError("unsupported normalization")


@dataclass(frozen=True)
class Measurement:
    name: str
    residual: float | None
    tolerance: float | None
    status: str
    detail: str = ""


@dataclass(frozen=True)
class OperatorReport:
    """Immutable serialized inputs and measured criteria, with no qualification."""

    convention: OperatorConvention
    measurements: tuple[Measurement, ...]
    data_json: str

    @property
    def status(self):
        statuses = {item.status for item in self.measurements}
        return "FAIL" if "FAIL" in statuses else (
            "UNMEASURED" if "UNMEASURED" in statuses else "PASS")

    def as_dict(self):
        return {"convention": asdict(self.convention),
                "measurements": [asdict(x) for x in self.measurements],
                "data": json.loads(self.data_json), "status": self.status,
                "meaning": "finite_setting_measurements_not_error_bounds",
                "qualification": "none"}

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True,
                                          allow_nan=False).encode()).hexdigest()


def _report(convention, measurements, **data):
    if not isinstance(convention, OperatorConvention):
        raise TypeError("an explicit OperatorConvention is required")
    return OperatorReport(convention, tuple(measurements), json.dumps(
        data, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _positive(value, name, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _matrix(values, name, shape=None):
    arr = np.asarray(values, dtype=complex)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2] or min(arr.shape) < 1:
        raise ValueError(f"{name} must have shape (nk, nao, nao)")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def _weights(values, nk):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError("weights must be real")
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (nk,) or not np.isfinite(arr).all() or np.any(arr <= 0):
        raise ValueError("weights must be a positive finite vector of length nk")
    if abs(float(arr.sum()) - 1.) > 1e-12:
        raise ValueError("primitive-cell weights must sum to one; no renormalization is performed")
    return arr


def _trace(left, right, weights):
    return complex(np.einsum("k,kij,kji->", weights, left, right))


def weighted_trace(left, right, weights):
    """Tr(D A), including its imaginary residual; never silently normalize."""
    left = _matrix(left, "left")
    right = _matrix(right, "right", left.shape)
    return _trace(left, right, _weights(weights, len(left)))


def _check(name, residual, tolerance, detail=""):
    tolerance = _positive(tolerance, "tolerance", zero=True)
    residual = float(abs(residual))
    if not np.isfinite(residual):
        raise ValueError(f"{name} produced a nonfinite measurement")
    return Measurement(name, residual, tolerance,
                       "PASS" if residual <= tolerance else "FAIL", detail)


def _hermiticity(arr):
    return float(np.max(np.abs(arr - arr.conj().swapaxes(-1, -2))))


def diagnose_operator(operator: Callable, density, direction, weights, *,
                      convention: OperatorConvention, tolerance=1e-9,
                      difference_step=1e-4, energy: Callable | None = None):
    """Probe a fixed-source J or K map using real Hermitian directions.

    The natural quadratic functional is E(D)=Tr(D O(D))/2. Its directional
    derivative equals Tr(H O(D)) only for a linear self-adjoint map. A caller
    may supply a separately implemented energy(D) for a stronger check.
    Positivity is not required: signed directions probe the complete space.
    Callback outputs are not symmetrized. Input mutation is rejected.
    """
    density = _matrix(density, "density")
    direction = _matrix(direction, "direction", density.shape)
    weights = _weights(weights, len(density))
    step = _positive(difference_step, "difference_step")
    if _hermiticity(density) > 1e-12 or _hermiticity(direction) > 1e-12:
        raise ValueError("density and direction must be Hermitian")

    def call(value):
        argument = value.copy()
        result = _matrix(operator(argument), "operator result", density.shape).copy()
        if not np.array_equal(argument, value):
            raise ValueError("operator mutated its density input")
        return result

    a, b = .73, -.41
    od, oh = call(density), call(direction)
    combined = call(a * density + b * direction)
    zero = call(np.zeros_like(density))
    dh, hd = _trace(direction, od, weights), _trace(density, oh, weights)
    measurements = [
        _check("output_hermiticity", max(_hermiticity(od), _hermiticity(oh)), tolerance),
        _check("zero_density", np.max(np.abs(zero)), tolerance),
        _check("linearity", np.max(np.abs(combined - a * od - b * oh)), tolerance),
        _check("reciprocal_bilinear", abs(dh - hd), tolerance),
        _check("bilinear_imaginary", max(abs(dh.imag), abs(hd.imag)), tolerance),
    ]

    def functional(value):
        if energy is None:
            return .5 * _trace(value, call(value), weights)
        argument = value.copy()
        result = complex(energy(argument))
        if not np.array_equal(argument, value):
            raise ValueError("energy callback mutated its density input")
        if not np.isfinite(result):
            raise ValueError("energy callback returned a nonfinite value")
        return result

    derivatives = []
    for h in (step, step / 2):
        plus, minus = functional(density + h * direction), functional(density - h * direction)
        derivative = (plus - minus) / (2 * h)
        derivatives.append([derivative.real, derivative.imag])
        measurements.append(_check(f"directional_derivative_step_{h:.17g}",
                                   abs(derivative - dh), tolerance))
    return _report(convention, measurements, weights=weights.tolist(),
                   shape=list(density.shape), difference_step=step,
                   derivatives=derivatives, expected_derivative=[dh.real, dh.imag],
                   energy_reference="supplied_callback" if energy else "quadratic_trace",
                   probes={"density_coefficient": a, "direction_coefficient": b})


def diagnose_energy(density, overlap, kinetic, nuclear_attraction, coulomb,
                    exchange, weights, *, convention: OperatorConvention,
                    expected_electrons, reported_energy, nuclear_energy,
                    coulomb_factor, exchange_factor, tolerance=1e-9,
                    expanded_nuclear_energy=None, nuclear_cell_count=None):
    """Check explicit energy conventions and independently supplied E_nn scaling.

    E=Tr(D(T+V))+cJ Tr(DJ)+cK Tr(DK)+E_nn. For spin-summed RHF,
    cJ=1/2 and cK=-1/4. These factors must be supplied explicitly.
    Expanded nuclear reference and cell count must both be supplied, otherwise
    nuclear normalization is UNMEASURED. This cannot prove a gauge is physical.
    """
    density = _matrix(density, "density")
    matrices = [_matrix(x, name, density.shape) for name, x in (
        ("overlap", overlap), ("kinetic", kinetic), ("nuclear_attraction", nuclear_attraction),
        ("coulomb", coulomb), ("exchange", exchange))]
    weights = _weights(weights, len(density))
    scalars = [expected_electrons, reported_energy, nuclear_energy, coulomb_factor, exchange_factor]
    if not all(np.isfinite(float(x)) for x in scalars):
        raise ValueError("energy scalars must be finite")
    traces = [_trace(density, matrix, weights) for matrix in matrices]
    electrons, t, v, j, k = traces
    total = t + v + coulomb_factor * j + exchange_factor * k + nuclear_energy
    measurements = [
        _check("input_hermiticity", max(_hermiticity(x) for x in [density, *matrices]), tolerance),
        _check("electron_count", abs(electrons - expected_electrons), tolerance),
        _check("energy_decomposition", abs(total - reported_energy), tolerance),
        _check("component_imaginary", max(abs(x.imag) for x in traces), tolerance),
    ]
    if (expanded_nuclear_energy is None) != (nuclear_cell_count is None):
        raise ValueError("expanded nuclear energy and cell count are required together")
    if expanded_nuclear_energy is None:
        measurements.append(Measurement("nuclear_normalization", None, None,
                                        "UNMEASURED", "independent expanded-cell reference missing"))
    else:
        if isinstance(nuclear_cell_count, (bool, np.bool_)) or int(nuclear_cell_count) != nuclear_cell_count or nuclear_cell_count < 1:
            raise ValueError("nuclear_cell_count must be a positive integer")
        measurements.append(_check("nuclear_normalization", float(expanded_nuclear_energy) /
                                   nuclear_cell_count - nuclear_energy, tolerance))
    return _report(convention, measurements, electron_count=electrons.real,
                   components={"kinetic": t.real, "nuclear_attraction": v.real,
                               "coulomb": float(coulomb_factor * j.real),
                               "exchange": float(exchange_factor * k.real),
                               "nuclear": float(nuclear_energy), "total": total.real},
                   coulomb_factor=float(coulomb_factor), exchange_factor=float(exchange_factor))


def diagnose_metric(metric, *, convention: OperatorConvention,
                    absolute_threshold, relative_threshold, tolerance=1e-10,
                    reported_rank=None):
    """Report Hermitian eigenvalues and the explicitly requested rank rule.

    Eigenvalues are diagnosed on (M+M*)/2 while the unsymmetrized residual is
    retained as a separate criterion. Retain lambda > max(atol, rtol*lambda_max).
    Rank and retained-space condition are measurements, not fit-error bounds.
    """
    matrix = np.asarray(metric, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("metric must be a finite nonempty square matrix")
    atol = _positive(absolute_threshold, "absolute_threshold", zero=True)
    rtol = _positive(relative_threshold, "relative_threshold", zero=True)
    eigenvalues = np.linalg.eigvalsh(.5 * (matrix + matrix.conj().T))
    threshold = max(atol, rtol * max(0., float(eigenvalues[-1])))
    kept = eigenvalues[eigenvalues > threshold]
    rank = len(kept)
    measurements = [_check("metric_hermiticity", np.max(np.abs(matrix - matrix.conj().T)), tolerance),
                    _check("metric_negative_eigenvalue", max(0., -float(eigenvalues[0])), tolerance)]
    if reported_rank is None:
        measurements.append(Measurement("reported_rank", None, None, "UNMEASURED", "source rank missing"))
    else:
        if isinstance(reported_rank, (bool, np.bool_)) or int(reported_rank) != reported_rank or not 0 <= reported_rank <= len(matrix):
            raise ValueError("reported_rank must be an integer within the metric dimension")
        measurements.append(_check("reported_rank", rank - reported_rank, 0.))
    return _report(convention, measurements, eigenvalues=eigenvalues.tolist(), rank=rank,
                   dimension=len(matrix), absolute_threshold=atol, relative_threshold=rtol,
                   executed_threshold=threshold,
                   retained_condition=float(kept[-1] / kept[0]) if rank else None,
                   closest_threshold_distance=float(np.min(np.abs(eigenvalues - threshold))))


def compare_covariance(reference, transformed, *, convention: OperatorConvention,
                       tolerance=1e-9, transformation):
    """Compare arrays already transported independently into the same frame."""
    if not isinstance(transformation, str) or not transformation.strip():
        raise ValueError("transformation must label the independently applied transport")
    reference = _matrix(reference, "reference")
    transformed = _matrix(transformed, "transformed", reference.shape)
    return _report(convention, [_check("covariance", np.max(np.abs(reference - transformed)), tolerance)],
                   transformation=transformation, shape=list(reference.shape))
