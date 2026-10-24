"""Source-bound PAO/PNO preparation for representative finite-torus pairs.

This bounded reference builder reuses the molecular semicanonical PAO,
pair-density and minimum-rank conventions. It prepares data for the shared
translation contractions; it does not iterate amplitudes or enable a public
periodic route. Riplinger and Neese, doi:10.1063/1.4773581, Sec. II B1
(PAOs), Sec. II B3, Eqs. (23)-(25) (density and truncation), and Sec. II C,
Eqs. (26)-(30) (transport), together with Nejad et al.,
doi:10.1063/5.0290816, Eqs. (37)-(44), motivate the pair-space construction
and translated source-space transport. The selected
``pno_norm`` remains explicit rather than transferring threshold meanings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from ..periodic.chi.pno import build_translation_pair_plan
from ..solvers._scf import solve_dense_fock
from .mp2 import _semicanonical_T
from .pno_density import pair_density, resolve_pno_norm
from .source import source_identity
from .translation import TranslationPairSpaces
from .translation_source import admit_translation_source
from .translation_pao import ProjectedPAOPreparation, ProjectedPairBasis
from .translation_factors import TranslationHomeFactors
from .translation_bloch_factors import TranslationBlochFactors
from .translation_operators import matrix_record


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _record(value):
    if isinstance(value, Mapping):
        return {str(key): _record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_record(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _array_record(value):
    from .translation_storage import TranslationOccupiedColumns
    if type(value) is TranslationOccupiedColumns:
        return value.record()
    array = np.ascontiguousarray(value, dtype="<f8")
    return {"shape": list(array.shape), "sha256": sha256(array.tobytes()).hexdigest()}


def _pair_coefficient_record(spaces):
    from .translation_storage import ProjectedPairCoefficients
    if type(spaces.pair_coefficients) is not ProjectedPairCoefficients:
        return [_array_record(value) for value in spaces.pair_coefficients]
    return dict(kind="projected-pair-recipes-v1",
        active=_array_record(spaces._occupied), frozen=_array_record(spaces._frozen_occupied),
        recipes=[dict(ao_indices=value.ao_indices.tolist(),
                      local_coefficients=_array_record(value.local_coefficients),
                      occupied_coefficients=_array_record(value.occupied_coefficients))
                 for value in spaces.pair_coefficients.recipes])


def _nonnegative(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return float(value)


def _domains(supplied, count, nbf):
    if supplied is None:
        return tuple(np.arange(nbf, dtype=np.int64) for _ in range(count))
    supplied = tuple(supplied)
    if len(supplied) != count:
        raise ValueError("pair AO domains must supply exactly one domain per representative")
    domains = []
    for value in supplied:
        array = np.asarray(value)
        if (array.ndim != 1 or array.dtype.kind not in "iu" or not array.size
                or np.min(array) < 0 or np.max(array) >= nbf
                or np.unique(array).size != array.size):
            raise ValueError("pair AO domains must be nonempty unique integer AO indices in range")
        domains.append(np.sort(array).astype(np.int64, copy=True))
    return tuple(domains)


def _exchange(factors, occupied, virtual):
    if type(factors) in (TranslationHomeFactors, TranslationBlochFactors):
        return factors.pair_exchange(occupied, virtual), factors.n_aux
    transformed = np.asarray(factors.mo_transform(occupied, virtual))
    if (transformed.ndim != 3 or transformed.shape[0] == 0
            or transformed.shape[1:] != (2, virtual.shape[1])
            or transformed.dtype.kind not in "iuf"
            or not np.isfinite(transformed).all()):
        raise ValueError("pair factor transform must return finite real B[auxiliary,2,PAO]")
    transformed = np.asarray(transformed, dtype=float)
    if not np.isfinite(transformed).all():
        raise ValueError("pair factor transform exceeds the real floating-point range")
    exchange = transformed[:, 0, :].T @ transformed[:, 1, :]
    if not np.isfinite(exchange).all():
        raise FloatingPointError("non-finite representative pair exchange integrals")
    return exchange, int(transformed.shape[0])


def _seed(exchange, f_ii, f_jj, energies):
    # Historical molecular MP2 has no level shift or denominator floor.
    # Reject undefined arithmetic without replacing a finite denominator.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        denominator = f_ii + f_jj - energies[:, None] - energies[None, :]
        if not np.isfinite(denominator).all() or np.any(denominator == 0):
            raise ValueError("pair MP2 denominators must be finite and nonzero")
        amplitude = _semicanonical_T(exchange, f_ii, f_jj, energies)
    if not np.isfinite(amplitude).all():
        raise FloatingPointError("non-finite representative semicanonical amplitudes")
    return amplitude


def _pair_energy(exchange, amplitude):
    value = float(np.sum(exchange * (2.0 * amplitude - amplitude.T)))
    if not np.isfinite(value):
        raise FloatingPointError("non-finite representative semicanonical energy")
    return value


@dataclass(frozen=True)
class TranslationPNOBuild:
    """Immutable numerical output and its actual consumed-source snapshot.

    Pair energies are unweighted ordered spatial-pair contractions. Only
    ``pno_correction_per_cell`` includes orbit/spin weights and cell division.
    ``source_identity`` is the parent's declared identity; ``identity`` seals
    this build's consumed arrays and never asserts freshness of that parent.
    """

    spaces: TranslationPairSpaces
    integrals: tuple[np.ndarray, ...]
    amplitudes: tuple[np.ndarray, ...]
    pao_ranks: tuple[int, ...]
    pno_ranks: tuple[int, ...]
    occupations: tuple[np.ndarray, ...]
    pao_pair_energies: tuple[float, ...]
    pno_pair_energies: tuple[float, ...]
    pno_correction_per_cell: float
    source_kind: str
    source_identity: str
    same_source: bool
    identity: str
    provenance: Mapping
    max_translation_error: float | None
    projected_pair_bases: tuple[ProjectedPairBasis, ...] | None = None


def build_translation_pair_pnos(
    source, mesh, occupied, pair_ao_domains=None, *, frozen_occupied=None,
    tcut_pno=0.0, pno_norm="mp2", lindep=1e-8, tolerance=1e-10,
    factor_audit="auto", compact_storage=False,
):
    """Build one PAO/PNO space and semicanonical seed per translation row.

    Domains are explicit AO-index sets; ``None`` selects every AO for each
    representative. All active and frozen occupieds are projected out. The
    strict molecular occupation cutoff and largest-PNO fallback are retained.
    Empty PAO spaces or cutoffs splitting a translation stabilizer are
    refused, without silently adding orbitals or repairing tensors.

    ``factor_audit='auto'`` uses constructive covariance only for the exact
    immutable home or retained-Bloch implementation on the matching mesh.
    Other providers retain every-origin untruncated PAO exchange audits.
    ``'all-origins'`` explicitly selects that reference path for all factors.
    Constructive covariance is not evidence of same-source SCF provenance;
    its ``max_translation_error`` is None because no numerical audit ran.
    """
    cutoff = _nonnegative(tcut_pno, "tcut_pno")
    lindep = _nonnegative(lindep, "lindep")
    norm = resolve_pno_norm(pno_norm)
    if not isinstance(compact_storage, bool):
        raise ValueError("compact_storage must be boolean")
    if factor_audit not in ("auto", "all-origins"):
        raise ValueError("factor_audit must be 'auto' or 'all-origins'")
    inputs = admit_translation_source(
        source, mesh, occupied, frozen_occupied=frozen_occupied, tolerance=tolerance,
    )
    S, F = inputs.overlap, inputs.fock
    C, frozen = inputs.occupied, inputs.frozen_occupied
    mesh = tuple(int(value) for value in mesh)
    cells = tuple(product(*(range(value) for value in mesh)))
    n_cells, nbf = len(cells), S.shape[0]
    plan = build_translation_pair_plan(C.shape[1] // n_cells, mesh)
    domains = _domains(pair_ao_domains, plan.n_representatives, nbf)
    stored_active, stored_frozen = C, frozen
    reconstruction_error = 0.
    if compact_storage:
        from .translation_storage import occupied_storage, ProjectedPairCoefficients
        stored_active = occupied_storage(inputs.occupied, mesh, tolerance)
        stored_frozen = occupied_storage(inputs.frozen_occupied, mesh, tolerance)
        if stored_active is not inputs.occupied or stored_frozen is not inputs.frozen_occupied:
            inputs = admit_translation_source(source, mesh, stored_active,
                frozen_occupied=stored_frozen, tolerance=tolerance)
        C, frozen = stored_active, stored_frozen

    def pair_storage(recipes):
        return ProjectedPairCoefficients(stored_active, stored_frozen, tuple(recipes))

    def check_reconstruction(expected, recipes):
        error = float(np.max(abs(pair_storage(recipes)[-1] - expected), initial=0.))
        if not np.isfinite(error) or error > tolerance:
            raise ValueError("projected pair recipe changes the complete AO basis")
        return error
    factors = inputs.factors
    sealed_factors = type(factors) in (TranslationHomeFactors, TranslationBlochFactors)
    constructive = sealed_factors and factor_audit == "auto"
    if sealed_factors:
        if factors.mesh != mesh or factors.n_orb != nbf:
            raise ValueError("home-factor mesh and AO dimensions must match the translation source")
    from .translation_storage import occupied_columns
    from .translation_pao import StreamedPAOPreparation
    preparation = (StreamedPAOPreparation(S, F, C, frozen) if compact_storage else
                   ProjectedPAOPreparation(S, F, np.column_stack((C, frozen))))
    bases, pao_energies, projected_bases = [], [], []
    for domain in domains:
        basis, energies, projected = preparation.basis(domain, lindep)
        if not compact_storage:
            bases.append(basis)
        pao_energies.append(energies)
        projected_bases.append(projected)
        if compact_storage:
            reconstruction_error = max(reconstruction_error, check_reconstruction(basis, (projected,)))
    if compact_storage:
        bases = pair_storage(projected_bases)
    del preparation, basis
    pao_spaces = TranslationPairSpaces(
        mesh, S, F, stored_active, bases, frozen_occupied=stored_frozen, tolerance=tolerance,
        compact_storage=compact_storage,
    )
    full_integrals, auxiliary_counts = [], []
    max_translation_error = None if constructive else 0.0
    for (i, j), basis in zip(pao_spaces.representative_pairs, pao_spaces.pair_coefficients):
        selected_occupied = _readonly(occupied_columns(C, [i, j]))
        if constructive:
            exchange = factors.pair_exchange(selected_occupied, basis)
            full_integrals.append(exchange)
            auxiliary_counts.append(factors.n_aux)
            continue
        exchange, naux = _exchange(inputs.factors, selected_occupied, basis)
        for shift in cells[1:]:
            shifted_occupied = np.roll(selected_occupied.reshape(*mesh, nbf // n_cells, 2),
                                       shift, axis=(0, 1, 2)).reshape(nbf, 2)
            shifted_basis = np.roll(basis.reshape(*mesh, nbf // n_cells, basis.shape[1]),
                                   shift, axis=(0, 1, 2)).reshape(basis.shape)
            translated, translated_naux = _exchange(inputs.factors, shifted_occupied, shifted_basis)
            if translated_naux != naux:
                raise ValueError("translated pair factors changed auxiliary dimension")
            error = float(np.max(np.abs(translated - exchange)))
            max_translation_error = max(max_translation_error, error)
            if error > tolerance:
                raise ValueError("pair exchange integrals are not translation covariant")
        full_integrals.append(exchange)
        auxiliary_counts.append(naux)
    full_integrals = pao_spaces.validate_tensors(full_integrals, name="PAO exchange integrals")
    retained, integrals, amplitudes, occupations, retained_projected = [], [], [], [], []
    full_energies, retained_energies = [], []
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for row, ((i, j), basis, eps, exchange) in enumerate(zip(
            pao_spaces.representative_pairs, pao_spaces.pair_coefficients, pao_energies, full_integrals,
        )):
            fii, fjj = float(pao_spaces.occupied_fock[i, i]), float(pao_spaces.occupied_fock[j, j])
            amplitude = _seed(exchange, fii, fjj, eps)
            full_energies.append(_pair_energy(exchange, amplitude))
            density = pair_density(amplitude, float(i == j), norm)
            if not np.isfinite(density).all():
                raise FloatingPointError("non-finite representative pair density")
            values, vectors = np.linalg.eigh(density)
            order = np.argsort(-values)
            values, vectors = values[order], vectors[:, order]
            keep = values > cutoff if cutoff > 0 else np.ones_like(values, dtype=bool)
            if not np.any(keep):
                keep[0] = True
            rotation = vectors[:, keep]
            fock = rotation.T @ np.diag(eps) @ rotation
            result = solve_dense_fock(
                0.5 * (fock + fock.T), basis_id=f"translation-mp2-pair-{row}-pno",
            )
            rotation = rotation @ result.eigenvectors
            projected = rotation.T @ exchange @ rotation
            seed = _seed(projected, fii, fjj, result.eigenvalues)
            retained_projected.append(projected_bases[row].rotate(rotation))
            if compact_storage:
                reconstruction_error = max(reconstruction_error,
                    check_reconstruction(basis @ rotation, (retained_projected[-1],)))
            else:
                retained.append(basis @ rotation)
            integrals.append(projected)
            amplitudes.append(seed)
            occupations.append(_readonly(values))
            retained_energies.append(_pair_energy(projected, seed))
    spaces = TranslationPairSpaces(
        mesh, S, F, stored_active, pair_storage(retained_projected) if compact_storage else retained,
        frozen_occupied=stored_frozen, tolerance=tolerance, compact_storage=compact_storage,
    )
    integrals = spaces.validate_tensors(integrals, name="PNO exchange integrals")
    amplitudes = spaces.validate_tensors(amplitudes, name="PNO semicanonical amplitudes")
    correction = math.fsum(float(weight) * (full - truncated) for weight, full, truncated in zip(
        spaces.energy_weights, full_energies, retained_energies,
    ))
    if not math.isfinite(correction):
        raise FloatingPointError("non-finite representative PNO correction")
    record = {
        "kind": "translation-pno-build-v2",
        "source_kind": inputs.source_kind, "source_identity": inputs.source_identity,
        "same_source": inputs.same_source, "source_snapshot": _record(inputs.provenance),
        "mesh": list(mesh), "conventions": _record(inputs.conventions),
        "overlap": matrix_record(S), "fock": matrix_record(F),
        "occupied": _array_record(spaces._occupied), "frozen_occupied": _array_record(spaces._frozen_occupied),
        "coefficient_storage": {
            "kind": "home-occupied-projected-pairs" if compact_storage else "dense-compatibility",
            "selected_occupied_bytes": spaces._occupied.nbytes + spaces._frozen_occupied.nbytes,
            "dense_selected_occupied_bytes": 8*nbf*(C.shape[1]+frozen.shape[1]),
            "pair_ao_bytes": 0 if compact_storage else sum(v.nbytes for v in spaces.pair_coefficients),
            "dense_pair_ao_bytes": sum(nbf * rank * 8 for rank in spaces.pair_ranks),
            "pair_recipe_bytes": sum(v.nbytes for p in retained_projected for v in
                                      (p.ao_indices, p.local_coefficients, p.occupied_coefficients)),
            "maximum_pair_reconstruction_error": reconstruction_error if compact_storage else None,
            "scope": "pair-build-selected-columns-and-pair-recipes; excludes-preparation-and-solver",
        },
        "domains": [domain.tolist() for domain in domains],
        "options": {"tcut_pno": cutoff, "pno_norm": norm, "lindep": lindep,
                    "tolerance": float(tolerance), "minimum_pno_rank": 1,
                    "factor_audit_requested": factor_audit, "compact_storage": compact_storage},
        "plan_identity": spaces.plan.fingerprint,
        "pao_exchange": [_array_record(value) for value in full_integrals],
        "pair_coefficients": _pair_coefficient_record(spaces),
        "projected_pair_bases": [{"ao_indices": value.ao_indices.tolist(),
                                  "local_coefficients": _array_record(value.local_coefficients),
                                  "occupied_coefficients": _array_record(value.occupied_coefficients)}
                                 for value in retained_projected],
        "integrals": [_array_record(value) for value in integrals],
        "amplitudes": [_array_record(value) for value in amplitudes],
        "occupations": [_array_record(value) for value in occupations],
        "auxiliary_counts": auxiliary_counts,
        "pao_pair_energies": full_energies, "pno_pair_energies": retained_energies,
        "pno_correction_per_cell": correction, "max_translation_error": max_translation_error,
        "factor_audit": ("constructive-retained-bloch-covariance" if constructive
                         and type(factors) is TranslationBlochFactors else
                         "constructive-home-auxiliary-covariance" if constructive
                         else "all-origins-untruncated-pao-exchange"),
        "factor_identity": factors.identity if sealed_factors else None,
    }
    return TranslationPNOBuild(
        spaces=spaces, integrals=integrals, amplitudes=amplitudes,
        pao_ranks=tuple(len(value) for value in pao_energies),
        pno_ranks=spaces.pair_ranks,
        occupations=tuple(occupations), pao_pair_energies=tuple(full_energies),
        pno_pair_energies=tuple(retained_energies), pno_correction_per_cell=correction,
        source_kind=inputs.source_kind, source_identity=inputs.source_identity,
        same_source=inputs.same_source, identity=source_identity(record),
        provenance=_freeze(record), max_translation_error=max_translation_error,
        projected_pair_bases=tuple(retained_projected),
    )
