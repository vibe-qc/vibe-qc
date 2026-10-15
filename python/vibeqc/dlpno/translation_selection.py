"""Deterministic projected-AO occupied frames on a real finite torus.

This is a projection/Löwdin frame selection, not an optimized Wannier
localization. For each occupied projector Q, choose home-cell AO seeds and
their complete translated orbits B = Q E. Symmetric orthogonalization gives
C = B (B.T S B)^(-1/2). Because the seed orbits intertwine AO and column
translations, their Gram matrix commutes with column translations. Its
inverse square root does too, so C has the required phase-free cell-major
translation action. The construction changes neither occupied projector.

The bounded selector greedily admits individual home AO families. It refuses
projectors without a sufficiently conditioned common AO-seed subset, even
when a more general linear combination of seeds could supply a frame. It
does not claim maximal localization, universal Wannier construction, or a
production default. This explicit frame supplies the translationally related
occupieds used by Nejad et al., doi:10.1063/5.0290816, Sec. II, while retaining
the metric dependence of projected AOs discussed by Riplinger and Neese,
doi:10.1063/1.4773581, Sec. II B1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..correlation_conventions import effective_electron_count
from ..periodic.chi.pno import _validated_pair_mesh
from .source import CorrelationSource, detect_reference_kind, source_identity
from .translation import _error, _readonly, _real_matrix
from .translation_operators import matrix_apply, matrix_diagonal, validate_translation_matrix
from .translation_source import admit_translation_source, _projector_error, _tensor_record

__all__ = ["TranslationOccupiedSelection", "select_translation_occupied"]


def _freeze(record):
    if isinstance(record, np.ndarray):
        return _freeze(record.tolist())
    if isinstance(record, dict):
        return MappingProxyType({key: _freeze(value) for key, value in record.items()})
    if isinstance(record, (tuple, list)):
        return tuple(_freeze(value) for value in record)
    return record


@dataclass(frozen=True)
class TranslationOccupiedSelection:
    """Immutable active/frozen frames and their explicit AO seed choices."""

    occupied: np.ndarray
    frozen_occupied: np.ndarray
    active_seed_aos: tuple[int, ...]
    frozen_seed_aos: tuple[int, ...]
    source_kind: str
    source_identity: str
    same_source: bool
    n_frozen: int
    identity: str
    provenance: Mapping


def _generators(mesh, n_local):
    ao_grid = np.arange(prod(mesh) * n_local).reshape(*mesh, n_local)
    cell_grid = np.arange(prod(mesh)).reshape(mesh)
    generators = []
    for axis, extent in enumerate(mesh):
        if extent > 1:
            rows = np.roll(ao_grid, 1, axis=axis).ravel()
            # Column j of P C is the column at cell j+1 in the original C.
            images = np.roll(cell_grid, -1, axis=axis).ravel()
            generators.append((rows, images))
    return tuple(generators)


def _projected_frame(coefficients, overlap, mesh, generators, tolerance, seed_lindep, label):
    n_cells = prod(mesh)
    n_local = overlap.shape[0] // n_cells
    n_bands = coefficients.shape[1] // n_cells
    if n_bands == 0:
        return np.empty((overlap.shape[0], 0)), (), ()
    projector_right = matrix_apply(overlap, coefficients, transpose=True).T
    for rows, _ in generators:
        if _projector_error(coefficients[rows], projector_right[:, rows],
                            coefficients, projector_right, n_local) > tolerance:
            raise ValueError(f"{label} occupied projector is not translation covariant")

    # Unit-S-norm AO seeds make family scoring independent of AO scale.
    seed_norms = np.sqrt(matrix_diagonal(overlap))
    metric_occupied = coefficients.T @ matrix_apply(overlap, coefficients)

    def seed_gram(indices):
        # B_D = C (C.T S)[:,D] / sqrt(S_DD). Keep the actual C.T S C
        # rather than replacing it with I within its admission tolerance.
        weights = projector_right[:, indices] / seed_norms[indices][None, :]
        gram = weights.T @ metric_occupied @ weights
        if not np.isfinite(gram).all():
            raise FloatingPointError("non-finite projected-AO seed Gram matrix")
        return gram, weights
    selected = []
    seed_minima = []
    cell_offsets = np.arange(n_cells)[:, None] * n_local
    for _ in range(n_bands):
        best_seed, best_score = None, -np.inf
        for candidate in range(n_local):
            if candidate in selected:
                continue
            indices = (cell_offsets + np.asarray([*selected, candidate])[None, :]).ravel()
            gram, _ = seed_gram(indices)
            eigenvalues = np.linalg.eigvalsh(.5 * (gram + gram.T))
            if (not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0
                    or eigenvalues[0] <= seed_lindep * eigenvalues[-1]):
                continue
            score = float(eigenvalues[0])
            # Ascending candidates and this explicit tie envelope prevent
            # roundoff under input occupied rotations from changing a tie.
            if best_seed is None or score > best_score + tolerance:
                best_seed, best_score = candidate, score
        if best_seed is None:
            raise ValueError(
                f"{label} projector has no admitted full-rank translated AO seed frame; "
                "the bounded common-seed selector cannot construct this gauge"
            )
        selected.append(best_seed)
        seed_minima.append(best_score)

    indices = (cell_offsets + np.asarray(selected)[None, :]).ravel()
    gram, weights = seed_gram(indices)
    seed_columns = coefficients @ weights
    eigenvalues, vectors = np.linalg.eigh(.5 * (gram + gram.T))
    if (not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0
            or eigenvalues[0] <= seed_lindep * eigenvalues[-1]):
        raise ValueError(f"{label} selected AO seed frame is numerically rank deficient")
    # The symmetric matrix function is unique even at degenerate eigenvalues;
    # no individual eigensolver phase or ordering defines the orbital gauge.
    inverse_sqrt = (vectors / np.sqrt(eigenvalues)[None, :]) @ vectors.T
    frame = seed_columns @ inverse_sqrt
    frame_right = matrix_apply(overlap, frame, transpose=True).T
    if (_error(frame.T @ matrix_apply(overlap, frame) - np.eye(frame.shape[1])) > tolerance
            or _projector_error(frame, frame_right, coefficients, projector_right, n_local) > tolerance):
        raise ValueError(f"{label} projected-AO frame did not preserve its metric projector")
    for rows, images in generators:
        columns = (images[:, None] * n_bands + np.arange(n_bands)[None, :]).ravel()
        if _error(frame[rows] - frame[:, columns]) > tolerance:
            raise ValueError(f"{label} projected-AO frame did not preserve the phase-free translation gauge")
    return frame, tuple(selected), tuple(seed_minima)


def select_translation_occupied(
    source, mesh, *, n_frozen=0, tolerance=1e-10, seed_lindep=1e-8,
):
    """Select active/frozen translated frames from a restricted source.

    ``n_frozen`` is the total number of frozen spatial orbitals in the finite
    system, not a per-cell count or chemical-core selector. It must contain
    complete translated bands. The first ``n_frozen`` source occupied
    columns define the frozen projector, independently of the active one.
    No source factors are read or transformed by this operation.
    """
    if not isinstance(source, CorrelationSource):
        raise TypeError("source must be a CorrelationSource")
    mesh = _validated_pair_mesh(mesh)
    if (isinstance(n_frozen, (bool, np.bool_)) or not isinstance(n_frozen, Integral)
            or n_frozen < 0 or n_frozen % prod(mesh)):
        raise ValueError("n_frozen must be a nonnegative total orbital count in complete translated bands")
    if (isinstance(seed_lindep, (bool, np.bool_)) or not isinstance(seed_lindep, Real)
            or not np.isfinite(seed_lindep) or not 0 < seed_lindep < 1):
        raise ValueError("seed_lindep must be a finite real number strictly between zero and one")
    if (source.reference_kind != "rhf" or detect_reference_kind(source.reference) != "rhf"
            or hasattr(source.reference, "mo_coeffs_alpha")
            or hasattr(source.reference, "mo_coeffs_beta")):
        raise ValueError("translation occupied selection requires a restricted RHF source")
    reference_coefficients = _real_matrix(source.reference.mo_coeffs, "source orbitals")
    n_occ = effective_electron_count(source.molecule, source.reference) // 2
    inputs = admit_translation_source(
        source, mesh, reference_coefficients[:, n_frozen:n_occ],
        frozen_occupied=reference_coefficients[:, :n_frozen], tolerance=tolerance,
    )
    S, F = inputs.overlap, inputs.fock
    validate_translation_matrix(S, mesh, tolerance, positive=True, name="translation overlap")
    validate_translation_matrix(F, mesh, tolerance, name="translation Fock")
    generators = _generators(mesh, S.shape[0] // prod(mesh))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        active, active_seeds, active_minima = _projected_frame(
            inputs.occupied, S, mesh, generators, tolerance, seed_lindep, "active",
        )
        frozen, frozen_seeds, frozen_minima = _projected_frame(
            inputs.frozen_occupied, S, mesh, generators, tolerance, seed_lindep, "frozen",
        )
    # Reuse the source partition admission to audit the selected projectors.
    admit_translation_source(source, mesh, active, frozen_occupied=frozen, tolerance=tolerance)
    record = {
        "schema": "vibeqc.translation-occupied-selection/v2",
        "algorithm": "projected-normalized-ao-seeds-symmetric-lowdin",
        "source_kind": inputs.source_kind, "source_identity": inputs.source_identity,
        "same_source": inputs.same_source, "source_inputs": inputs.provenance,
        "mesh": mesh, "n_frozen": int(n_frozen),
        "tolerance": float(tolerance), "seed_lindep": float(seed_lindep),
        "active_seed_aos": active_seeds, "frozen_seed_aos": frozen_seeds,
        "active_seed_minima": active_minima, "frozen_seed_minima": frozen_minima,
        "occupied": _tensor_record(active), "frozen_occupied": _tensor_record(frozen),
    }
    return TranslationOccupiedSelection(
        occupied=_readonly(active), frozen_occupied=_readonly(frozen),
        active_seed_aos=active_seeds, frozen_seed_aos=frozen_seeds,
        source_kind=inputs.source_kind, source_identity=inputs.source_identity,
        same_source=inputs.same_source, n_frozen=int(n_frozen),
        identity=source_identity(record), provenance=_freeze(record),
    )
