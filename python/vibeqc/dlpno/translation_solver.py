"""Experimental compiled MP2 iteration over finite-torus representative pairs.

The native Jacobi loop retains only quotient pair tensors and transports each
coupled source amplitude in its own pair space. Its raw residual is the one
in :mod:`vibeqc.dlpno.translation_mp2`, following Nejad et al.,
doi:10.1063/5.0290816, Eqs. (42)-(44). This is an explicitly bounded route,
with no change to the molecular solver's iteration policy or thresholds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np

from .translation_builder import TranslationPNOBuild, _array_record, _freeze
from .source import source_identity


@dataclass(frozen=True)
class TranslationMP2Step:
    """Energy and raw residual of one complete amplitude snapshot.

    ``energy_change`` is ``None`` for the first snapshot, which cannot
    establish convergence. Iterations count evaluated states, not updates.
    """

    iteration: int
    energy_per_cell: float
    energy_change: float | None
    maximum_residual: float


@dataclass(frozen=True)
class TranslationMP2Result:
    """Immutable representative amplitudes and same-snapshot diagnostics.

    The fixed discarded-PNO correction is inherited from the semicanonical
    build. It does not enter the residual or convergence test and does not
    recover a discarded-PAO/domain error.
    """

    amplitudes: tuple[np.ndarray, ...]
    residuals: tuple[np.ndarray, ...]
    converged: bool
    n_iter: int
    max_residual: float
    e_pair_per_cell: float
    e_pno_correction_per_cell: float
    e_corr_per_cell: float
    e_hf_per_cell: float
    e_tot_per_cell: float
    trace: tuple[TranslationMP2Step, ...]
    build_identity: str
    identity: str
    provenance: Mapping


def _option(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} real number")
    return float(value)


def solve_translation_mp2(
    build, *, max_iter=100, conv_tol_energy=1e-9,
    conv_tol_residual=1e-7, damping=1.0,
):
    """Solve the built representative LMP2 equations with compiled Jacobi.

    Convergence requires both the maximum absolute raw residual and the
    per-cell energy change between successive evaluated states. No residual
    projection, occupied-Fock cutoff, level shift or denominator floor is
    applied. A finite iteration limit returns the last evaluated state with
    ``converged=False``; it does not return an unexamined final update.

    The diagonal virtual-Fock preconditioner must preserve pair-exchange
    sewing. The builder supplies semicanonical pair bases; inconsistent
    custom bases or undefined arithmetic fail without altering the build.
    """
    if not isinstance(build, TranslationPNOBuild):
        raise TypeError("build must be a TranslationPNOBuild instance")
    if (isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral)
            or not 1 <= max_iter <= np.iinfo(np.int32).max):
        raise ValueError("max_iter must be a positive 32-bit integer")
    max_iter = int(max_iter)
    energy_tol = _option(conv_tol_energy, "conv_tol_energy")
    residual_tol = _option(conv_tol_residual, "conv_tol_residual")
    damping = _option(damping, "damping", positive=True)
    if damping > 1:
        raise ValueError("damping must be at most one")
    spaces = build.spaces
    integrals = spaces.validate_tensors(build.integrals, name="integrals")
    seeds = spaces.validate_tensors(build.amplitudes, name="amplitudes")
    correction = float(build.pno_correction_per_cell)
    reference = float(build.provenance["source_snapshot"]["reference_energy"])
    if not math.isfinite(correction) or not math.isfinite(reference):
        raise ValueError("reference energy and PNO correction must be finite")
    fock = tuple(spaces.fock_between(pair, pair) for pair in spaces.representative_pairs)
    from .._vibeqc_core import dlpno_translation_mp2_solve

    amplitudes, residuals, converged, records = dlpno_translation_mp2_solve(
        list(spaces.mesh), spaces.plan.n_bands,
        [i for i, _ in spaces.representative_pairs],
        [j for _, j in spaces.representative_pairs],
        integrals, fock, seeds, spaces.pair_coefficients,
        spaces.occupied_fock, spaces._overlap, spaces.energy_weights,
        max_iter, energy_tol, residual_tol, damping, spaces.tolerance,
    )
    amplitudes = spaces.validate_tensors(amplitudes, name="solved amplitudes")
    residuals = spaces.validate_tensors(residuals, name="raw residuals")
    trace = tuple(TranslationMP2Step(
        int(iteration), float(energy), None if iteration == 1 else float(change), float(residual),
    ) for iteration, energy, change, residual in records)
    energy = trace[-1].energy_per_cell
    correlated = math.fsum((energy, correction))
    reference /= spaces.plan.n_cells
    total = math.fsum((reference, correlated))
    if not all(math.isfinite(value) for value in (correlated, reference, total)):
        raise FloatingPointError("non-finite translation MP2 result energy")
    record = {
        "kind": "experimental-translation-mp2-jacobi-v1",
        "build_identity": build.identity,
        "source_identity": build.source_identity, "same_source": build.same_source,
        "input_snapshot": {
            "mesh": list(spaces.mesh), "pairs": [list(pair) for pair in spaces.representative_pairs],
            "overlap": _array_record(spaces._overlap),
            "occupied_fock": _array_record(spaces.occupied_fock),
            "virtual_fock": [_array_record(value) for value in fock],
            "pair_coefficients": [_array_record(value) for value in spaces.pair_coefficients],
            "integrals": [_array_record(value) for value in integrals],
            "seeds": [_array_record(value) for value in seeds],
            "energy_weights": _array_record(spaces.energy_weights),
            "symmetry_tolerance": spaces.tolerance,
        },
        "options": {"max_iter": max_iter, "conv_tol_energy": energy_tol,
                    "conv_tol_residual": residual_tol, "damping": damping},
        "convergence": "same-snapshot-raw-maximum-residual-and-per-cell-energy-change",
        "amplitudes": [_array_record(value) for value in amplitudes],
        "residuals": [_array_record(value) for value in residuals],
        "converged": bool(converged), "n_iter": len(trace),
        "trace": [[step.iteration, step.energy_per_cell, step.energy_change,
                   step.maximum_residual] for step in trace],
        "pair_energy_per_cell": energy, "pno_correction_per_cell": correction,
        "hf_energy_per_cell": reference, "correlation_energy_per_cell": correlated,
        "total_energy_per_cell": total,
    }
    return TranslationMP2Result(
        amplitudes, residuals, bool(converged), len(trace), trace[-1].maximum_residual,
        energy, correction, correlated, reference, total, trace,
        build.identity, source_identity(record), _freeze(record),
    )
