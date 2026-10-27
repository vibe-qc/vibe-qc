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

from .translation_builder import TranslationPNOBuild, _array_record, _freeze, _pair_coefficient_record
from .source import source_identity
from .translation_operators import CyclicAOOperator, matrix_record
from .translation_transport import (
    DEFAULT_TRANSPORT_BYTES, TranslationTransport, _byte_budget, _transport_input_record,
    prepare_translation_transport,
)


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
    conv_tol_residual=1e-7, damping=1.0, transport=None,
    transport_backend="prepared", max_transport_bytes=DEFAULT_TRANSPORT_BYTES,
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

    The default prepares reusable pair-space overlap links, under an explicit
    transport-memory budget. A matching ``transport`` can be reused without
    AO contractions. ``transport_backend='bounded'`` owns compact recipes and
    reconstructs numerical links through a solve-local single-entry cache.
    ``transport_backend='ao'`` retains the full AO reference
    implementation for independent comparisons using an explicit dense source;
    cyclic source operators are never expanded for that backend. Neither
    choice screens terms.
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
    if transport_backend not in ("prepared", "bounded", "ao"):
        raise ValueError("transport_backend must be 'prepared', 'bounded' or 'ao'")
    max_transport_bytes = _byte_budget(max_transport_bytes, "max_transport_bytes")
    link_work = (0, 0, 0, 0, 0, 0)
    if transport_backend in ("prepared", "bounded"):
        if transport is None:
            transport = prepare_translation_transport(build, max_bytes=max_transport_bytes, backend=transport_backend)
        elif not isinstance(transport, TranslationTransport):
            raise TypeError("transport must be a TranslationTransport instance")
        elif transport.input_identity != source_identity(_transport_input_record(build)):
            raise ValueError("prepared transport does not match the calculation's pair spaces")
        if transport.provenance["backend"] != transport_backend:
            raise ValueError("transport backend differs from the requested backend")
        if transport.required_bytes > max_transport_bytes:
            raise ValueError("prepared transport exceeds max_transport_bytes")
        from .._vibeqc_core import dlpno_translation_mp2_solve_prepared
        fock = transport.virtual_fock
        amplitudes, residuals, converged, records, link_work = dlpno_translation_mp2_solve_prepared(
            transport._payload, integrals, seeds, max_iter, energy_tol, residual_tol, damping, True,
        )
    else:
        if transport is not None:
            raise ValueError("an explicit prepared transport requires transport_backend='prepared'")
        if spaces.compact_storage:
            raise ValueError("transport_backend='ao' requires compact_storage=False explicitly")
        if type(spaces._overlap) is CyclicAOOperator or type(spaces._fock) is CyclicAOOperator:
            raise ValueError("transport_backend='ao' requires an explicit dense reference calculation; cyclic operators are not expanded")
        fock = tuple(spaces.fock_between(pair, pair) for pair in spaces.representative_pairs)
        from .._vibeqc_core import dlpno_translation_mp2_solve
        amplitudes, residuals, converged, records = dlpno_translation_mp2_solve(
            list(spaces.mesh), spaces.plan.n_bands,
            [i for i, _ in spaces.representative_pairs], [j for _, j in spaces.representative_pairs],
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
            "overlap": matrix_record(spaces._overlap),
            "occupied_fock": _array_record(spaces.occupied_fock),
            "virtual_fock": [_array_record(value) for value in fock],
            "pair_coefficients": _pair_coefficient_record(spaces),
            "integrals": [_array_record(value) for value in integrals],
            "seeds": [_array_record(value) for value in seeds],
            "energy_weights": _array_record(spaces.energy_weights),
            "symmetry_tolerance": spaces.tolerance,
        },
        "options": {"max_iter": max_iter, "conv_tol_energy": energy_tol,
                    "conv_tol_residual": residual_tol, "damping": damping,
                    "transport_backend": transport_backend, "max_transport_bytes": max_transport_bytes},
        "transport_identity": None if transport is None else transport.identity,
        "transport_preparation": None if transport is None else {
            "backend": transport_backend,
            "cache_policy": transport.provenance["cache_policy"],
            "link_constructions": int(link_work[0]), "link_cache_hits": int(link_work[1]),
            "target_metric_constructions": int(link_work[3]),
            "target_metric_cache_hits": int(link_work[4]),
            "target_metric_evictions": int(link_work[5]),
            "target_metric_cache_bytes": transport.provenance["target_metric_cache_bytes"],
            "link_evictions": int(link_work[2]), "cache_bytes_after_solve": 0,
            "numerical_link_bytes": transport.provenance["numerical_link_bytes"],
            "recipe_snapshot_bytes": transport.provenance["recipe_snapshot_bytes"],
            "metadata_bytes": transport.provenance["metadata_bytes"],
            "python_workspace_bytes": transport.provenance["python_workspace_bytes"],
            "recipe_only": transport.provenance["recipe_only"],
            "occupied_home_input": transport.provenance["occupied_home_input"],
            "occupied_input_bytes": transport.provenance["occupied_input_bytes"],
            "pair_ao_input_bytes": transport.provenance["pair_ao_input_bytes"],
            "compact_projected_bases": transport.compact, "n_links": transport.n_links,
            "metric_backend": transport.metric_backend,
            "stable_selected_column_links": transport.stable_links,
            "n_couplings": transport.n_couplings, "retained_bytes": transport.retained_bytes,
            "workspace_bytes": transport.workspace_bytes, "required_bytes": transport.required_bytes,
        },
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
