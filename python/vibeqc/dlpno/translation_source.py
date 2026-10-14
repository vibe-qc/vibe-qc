"""Restricted source admission for representative finite-torus pair builds.

The caller supplies active and frozen occupied coefficients in an explicit
translation gauge. Their separate projectors must match the corresponding
source spaces; matching the union would permit an unintended core/active
rotation. Translation covariance is checked on the resulting pair spaces by
``TranslationPairSpaces`` before any factor transforms are evaluated.

This admission neither certifies factor covariance nor changes ``same_source``.
The pair builder must audit the contracted factors at every translated pair
placement before using representative-only integrals or compressing PAOs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..correlation_conventions import effective_electron_count
from ..periodic.chi.pno import _validated_pair_mesh
from .source import CorrelationSource, _array_identity, detect_reference_kind
from .translation import _error, _readonly, _real_matrix

__all__ = ["TranslationSourceInputs", "admit_translation_source"]


@dataclass(frozen=True)
class TranslationSourceInputs:
    """Private tensor snapshots plus the source's unchanged factor provider.

    The builder uses ``factors`` only through ``mo_transform``. This object
    does not copy its potentially compact storage or promise immutability
    of an external provider. Numerical input snapshots and conventions are
    immutable; provenance is an independent record for the builder to extend.
    """

    overlap: np.ndarray
    fock: np.ndarray
    occupied: np.ndarray
    frozen_occupied: np.ndarray
    factors: Any
    source_kind: str
    source_identity: str
    same_source: bool
    conventions: Mapping[str, Any]
    n_electrons: int
    n_frozen: int
    reference_energy: float
    provenance: dict[str, Any]


def _require_orthonormal(coefficients, overlap, tolerance, name):
    defect = coefficients.T @ overlap @ coefficients - np.eye(coefficients.shape[1])
    if _error(defect) > tolerance:
        raise ValueError(f"{name} must be S-orthonormal")


def _tensor_record(array):
    if array.size == 0:
        # No frozen bands is a valid (n_ao, 0) coefficient matrix. A zero
        # dimension cannot be cast through the shared memoryview hasher.
        return {"shape": list(array.shape), "sha256": sha256(b"").hexdigest()}
    return _array_identity(array)


def admit_translation_source(
    source, mesh, occupied, *, frozen_occupied=None, tolerance=1e-10,
):
    """Validate a real restricted finite-torus source and its occupied split.

    ``frozen_occupied`` determines the number of frozen source columns. Its
    projector must match ``reference.mo_coeffs[:, :n_frozen]``; the active
    projector must independently match the remaining occupied columns. Both
    spaces contain complete bands of the requested finite translation group.
    Electron counts include the reference's existing ECP core metadata.

    This helper validates source consistency. The pair-space constructor
    additionally requires a positive metric, symmetric/covariant Fock and
    overlap, and the actual phase-free cell-major occupied gauge. It must be
    called before the retained factor provider is asked to transform tensors.
    """
    if not isinstance(source, CorrelationSource):
        raise TypeError("source must be a CorrelationSource")
    if (isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("source tolerance must be a positive finite real number")
    tolerance = float(tolerance)
    mesh = _validated_pair_mesh(mesh)
    if source.conventions.kmesh is None or _validated_pair_mesh(source.conventions.kmesh) != mesh:
        raise ValueError("source kmesh must match the requested translation mesh")
    normalization = source.conventions.bloch_normalisation
    if normalization not in {
        "real-bvk-supercell", "unnormalised-ao-bloch-sums-uniform-weights",
    }:
        raise ValueError("source must carry an admitted finite-torus normalization")
    reference = source.reference
    if (source.reference_kind != "rhf" or detect_reference_kind(reference) != "rhf"
            or hasattr(reference, "mo_coeffs_alpha") or hasattr(reference, "mo_coeffs_beta")):
        raise ValueError("translation pair construction requires a restricted RHF source")
    if not bool(getattr(reference, "converged", False)):
        raise ValueError("translation pair construction requires a converged reference")
    energy = getattr(reference, "energy", None)
    if (isinstance(energy, (bool, np.bool_)) or not isinstance(energy, Real)
            or not np.isfinite(energy)):
        raise ValueError("reference energy must be a finite real number")
    if source.overlap is None:
        raise ValueError("translation source requires an explicit AO overlap")
    S = _real_matrix(source.overlap, "source overlap")
    F = _real_matrix(reference.fock, "source Fock")
    reference_coefficients = _real_matrix(reference.mo_coeffs, "source orbitals")
    C = _real_matrix(occupied, "active occupied coefficients")
    nbf = getattr(source.basis, "nbasis", None)
    if (isinstance(nbf, (bool, np.bool_)) or not isinstance(nbf, Integral)
            or nbf <= 0 or S.shape != (nbf, nbf) or F.shape != S.shape
            or reference_coefficients.shape[0] != nbf or C.shape[0] != nbf):
        raise ValueError("source tensors must match its AO basis dimensions")
    if source.molecule.multiplicity != 1:
        raise ValueError("restricted translation source must be a singlet")
    n_electrons = effective_electron_count(source.molecule, reference)
    if n_electrons <= 0 or n_electrons % 2:
        raise ValueError("restricted translation source needs a positive even electron count")
    n_occ = n_electrons // 2
    if n_occ > reference_coefficients.shape[1] or n_occ >= nbf:
        raise ValueError("source electron count must leave occupied and virtual AO spaces")
    frozen = (np.empty((nbf, 0)) if frozen_occupied is None
              else _real_matrix(frozen_occupied, "frozen occupied coefficients"))
    n_cells = prod(mesh)
    if (nbf % n_cells or frozen.shape[0] != nbf or C.shape[1] == 0
            or C.shape[1] % n_cells or frozen.shape[1] % n_cells
            or C.shape[1] + frozen.shape[1] != n_occ):
        raise ValueError("active and frozen source spaces require complete bands matching the electron count")
    n_frozen = frozen.shape[1]
    source_occupied = reference_coefficients[:, :n_occ]
    _require_orthonormal(source_occupied, S, tolerance, "source occupied orbitals")
    _require_orthonormal(C, S, tolerance, "active occupied coefficients")
    _require_orthonormal(frozen, S, tolerance, "frozen occupied coefficients")
    if _error(C.T @ S @ frozen) > tolerance:
        raise ValueError("active and frozen occupied spaces must be orthogonal")
    source_frozen = source_occupied[:, :n_frozen]
    source_active = source_occupied[:, n_frozen:]
    active_projector = C @ C.T @ S
    frozen_projector = frozen @ frozen.T @ S
    if _error(active_projector - source_active @ source_active.T @ S) > tolerance:
        raise ValueError("active occupied projector does not match the source active space")
    if _error(frozen_projector - source_frozen @ source_frozen.T @ S) > tolerance:
        raise ValueError("frozen occupied projector does not match the source frozen space")
    conventions = dict(source.conventions.as_record())
    conventions["kmesh"] = mesh
    provenance = {
        "schema": "vibeqc.translation-source-inputs/v1",
        "source_kind": str(source.kind), "source_identity": str(source.identity),
        "source_provenance": deepcopy(source.provenance),
        "conventions": dict(conventions),
        "n_electrons": n_electrons, "n_frozen": n_frozen,
        "ecp_total_ncore": int(getattr(reference, "ecp_total_ncore", 0) or 0),
        "reference_energy": float(energy),
        "overlap": _tensor_record(S), "fock": _tensor_record(F),
        "source_orbitals": _tensor_record(reference_coefficients),
        "active_occupied": _tensor_record(C),
        "frozen_occupied": _tensor_record(frozen),
        "active_projector": _tensor_record(active_projector),
        "frozen_projector": _tensor_record(frozen_projector),
    }
    return TranslationSourceInputs(
        overlap=_readonly(S), fock=_readonly(F), occupied=_readonly(C),
        frozen_occupied=_readonly(frozen), factors=source.factors,
        source_kind=str(source.kind), source_identity=str(source.identity),
        same_source=bool(source.conventions.same_source),
        conventions=MappingProxyType(conventions), n_electrons=n_electrons,
        n_frozen=n_frozen, reference_energy=float(energy), provenance=provenance,
    )
