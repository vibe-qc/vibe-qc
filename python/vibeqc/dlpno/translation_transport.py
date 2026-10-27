"""Prepare bounded pair-space transport once for a representative MP2 solve.

Cross-pair overlaps follow Riplinger and Neese, doi:10.1063/1.4773581,
Eq. (30), with finite translations as in Nejad et al.,
doi:10.1063/5.0290816, Eqs. (42)-(44). A projected PAO basis has
``V = E_D A - C B`` and ``B = C.T S E_D A``. Its translated overlap is
``A_p.T S[D_p,shift(D_q)] A_q - B_p.T shift(B_q)``. This exact identity
uses no coefficient cutoff or translated AO copy. General custom builds
without projected-basis metadata contract their AO pair coefficients directly.
If normalization amplifies cancellation, selected S columns and the shared
S*C instead form the target metric once; cell-block contractions with actual
source coefficients then retain the admitted finite-precision tails.
Cyclic home-row metrics remain blockwise through native preparation,
including the reciprocal-symbol positive-definiteness check. The resulting
iterative payload is the same pair-space graph for dense and block sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from .source import source_identity
from .translation_builder import TranslationPNOBuild, _array_record, _freeze, _readonly, _pair_coefficient_record
from .translation_operators import CyclicAOOperator, matrix_record


DEFAULT_TRANSPORT_BYTES = 256 * 1024**2


def _byte_budget(value, name="max_bytes"):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or not 0 < value <= np.iinfo(np.uint64).max):
        raise ValueError(f"{name} must be a positive unsigned 64-bit integer")
    return int(value)


def _transport_input_record(build):
    spaces = build.spaces
    from .translation_storage import ProjectedPairCoefficients
    recipe_only = type(spaces.pair_coefficients) is ProjectedPairCoefficients
    if recipe_only and (spaces.pair_coefficients.active is not spaces._occupied
                        or spaces.pair_coefficients.frozen is not spaces._frozen_occupied
                        or spaces.pair_coefficients.active.mesh != spaces.mesh):
        raise ValueError("projected pair recipes differ from the calculation's occupied storage")
    record = {
        "source_identity": build.source_identity,
        "mesh": list(spaces.mesh), "pairs": [list(pair) for pair in spaces.representative_pairs],
        "overlap": matrix_record(spaces._overlap), "fock": matrix_record(spaces._fock),
        "occupied": _array_record(spaces._occupied),
        "frozen_occupied": _array_record(spaces._frozen_occupied),
        "occupied_fock": _array_record(spaces.occupied_fock),
        "coefficients": _pair_coefficient_record(spaces),
        "energy_weights": _array_record(spaces.energy_weights), "tolerance": spaces.tolerance,
    }
    projected = getattr(build, "projected_pair_bases", None)
    if recipe_only and (projected is None or len(projected) != len(spaces.pair_coefficients)
                        or any(p is not q for p, q in zip(projected, spaces.pair_coefficients.recipes))):
        raise ValueError("projected pair recipes differ from the calculation's pair storage")
    record["projected_pair_bases"] = None if projected is None else [{
        "ao_indices": _array_record(value.ao_indices),
        "local_coefficients": _array_record(value.local_coefficients),
        "occupied_coefficients": _array_record(value.occupied_coefficients),
    } for value in projected]
    return record


@dataclass(frozen=True)
class TranslationTransport:
    """Immutable Python record around a constructor-private native payload.

    The byte bound covers retained numerical/route storage plus a conservative
    preparation workspace, including metric validation. It excludes existing
    caller arrays, allocator overhead, and the later solver's amplitude/residual
    workspaces. Bounded mode includes binding copies and Python numerical
    preparation; prepared mode retains its historical native-only scope. It is not a process-RSS cap.
    Bounded mode owns immutable metric and recipe snapshots, and releases its
    single-link cache at the end of each solve. Prepared mode retains links.
    """

    _payload: object = field(repr=False, compare=False)
    virtual_fock: tuple[np.ndarray, ...]
    input_identity: str
    identity: str
    compact: bool
    metric_backend: str
    stable_links: int
    n_links: int
    n_couplings: int
    retained_bytes: int
    workspace_bytes: int
    required_bytes: int
    provenance: Mapping


def prepare_translation_transport(build, *, max_bytes=DEFAULT_TRANSPORT_BYTES, backend="prepared"):
    """Prepare unique overlap links without storing any placed-pair tensor.

    ``max_bytes`` fails before numerical link allocation; no automatic
    fallback changes the requested memory policy. Each (target,source,shift)
    overlap is retained once in prepared mode. Bounded mode audits and discards
    it, then reconstructs from owned compact recipes during the native solve.
    Exchange orientation and both occupied-Fock
    coupling legs remain separate, including exactly zero coefficients.
    Cyclic operators use their home-row blocks directly, without expanding
    an AO matrix for the native binding or its validation.
    """
    if not isinstance(build, TranslationPNOBuild):
        raise TypeError("build must be a TranslationPNOBuild instance")
    max_bytes = _byte_budget(max_bytes)
    spaces = build.spaces
    from .translation_storage import ProjectedPairCoefficients
    recipe_only = type(spaces.pair_coefficients) is ProjectedPairCoefficients
    if backend not in ("prepared", "bounded"):
        raise ValueError("transport backend must be 'prepared' or 'bounded'")
    bounded = backend == "bounded"
    if bounded and not recipe_only:
        raise ValueError("bounded transport requires compact_storage=True")
    # Charge Python fock outputs, home stacking, hashing/conversion temporaries,
    # one reconstructed pair and its S/F contractions before creating them.
    # Existing source arrays and downstream amplitude storage are excluded.
    python_bytes = 0
    if bounded:
        ranks = spaces.pair_coefficients.ranks
        ao = spaces._occupied.shape[0]
        home_bytes = spaces._occupied.nbytes + spaces._frozen_occupied.nbytes
        recipes = spaces.pair_coefficients.nbytes
        metric_bytes = (spaces._overlap.blocks.nbytes if type(spaces._overlap) is CyclicAOOperator
                        else spaces._overlap.nbytes)
        python_bytes = (4096 + 64*ao*max(ranks) + 32*sum(r*r for r in ranks)
                        + 4*(home_bytes+recipes+metric_bytes+spaces.occupied_fock.nbytes))
        if python_bytes >= max_bytes:
            raise MemoryError("bounded transport Python workspace exceeds max_bytes before allocation")
    snapshot = _transport_input_record(build)
    fock = tuple(_readonly(spaces.fock_between(pair, pair)) for pair in spaces.representative_pairs)
    projected = getattr(build, "projected_pair_bases", None)
    if projected is None:
        domains, local, occupied, all_occupied = [], (), (), None
    else:
        if len(projected) != spaces.plan.n_representatives:
            raise ValueError("projected pair bases must supply every representative")
        domains = []
        for value in projected:
            indices = np.asarray(value.ao_indices)
            if indices.ndim != 1 or indices.dtype.kind not in "iu":
                raise ValueError("projected pair AO domains must contain integer indices")
            domains.append(indices.tolist())
        local = tuple(value.local_coefficients for value in projected)
        occupied = tuple(value.occupied_coefficients for value in projected)
        all_occupied = (np.column_stack((spaces._occupied.home, spaces._frozen_occupied.home))
                        if recipe_only else np.column_stack((spaces._occupied, spaces._frozen_occupied)))
    if type(spaces._overlap) is CyclicAOOperator:
        from .._vibeqc_core import dlpno_prepare_translation_transport_blocks as prepare_native
        overlap = spaces._overlap.blocks
    else:
        from .._vibeqc_core import dlpno_prepare_translation_transport as prepare_native
        overlap = spaces._overlap
    payload = prepare_native(
        list(spaces.mesh), spaces.plan.n_bands,
        [i for i, _ in spaces.representative_pairs], [j for _, j in spaces.representative_pairs],
        fock, () if recipe_only else spaces.pair_coefficients, spaces.occupied_fock, overlap,
        spaces.energy_weights, spaces.tolerance, int(max_bytes-python_bytes), domains, local, occupied, all_occupied, recipe_only, bounded,
    )
    input_identity = source_identity(snapshot)
    metric_backend = "cyclic-blocks" if payload.cyclic_metric else "dense"
    record = {
        "kind": "translation-pair-transport-v1", "input_identity": input_identity,
        "target_metric_cache_bytes": int(payload.target_metric_cache_bytes),
        "backend": backend, "cache_policy": "solve-local-one-link-one-target-metric" if bounded else "all-links",
        "numerical_link_bytes": int(payload.numerical_link_bytes),
        "recipe_snapshot_bytes": int(payload.recipe_bytes), "metadata_bytes": int(payload.metadata_bytes),
        "python_workspace_bytes": python_bytes,
        "input_snapshot": snapshot, "virtual_fock": [_array_record(value) for value in fock],
        "compact_projected_bases": bool(payload.compact),
        "metric_backend": metric_backend,
        "pair_ao_input_bytes": 0 if recipe_only else sum(v.nbytes for v in spaces.pair_coefficients),
        "recipe_only": bool(payload.recipe_only),
        "occupied_home_input": recipe_only,
        "occupied_input_bytes": 0 if all_occupied is None else all_occupied.nbytes,
        "stable_selected_column_links": int(payload.stable_links),
        "n_links": int(payload.n_links), "n_couplings": int(payload.n_couplings),
        "retained_bytes": int(payload.retained_bytes), "workspace_bytes": int(payload.workspace_bytes)+python_bytes,
        "required_bytes": int(payload.required_bytes)+python_bytes, "maximum_bytes": int(max_bytes),
        "budget_scope": ("transport-payload-conversions-and-link-workspace" if bounded
                         else "native-transport-payload-and-numerical-preparation-workspace"),
    }
    return TranslationTransport(
        payload, fock, input_identity, source_identity(record), bool(payload.compact), metric_backend,
        int(payload.stable_links),
        int(payload.n_links), int(payload.n_couplings), int(payload.retained_bytes),
        int(payload.workspace_bytes)+python_bytes, int(payload.required_bytes)+python_bytes, _freeze(record),
    )
