"""Executed RSGDF source identity, shared below periodic constructions.

A descriptor identifies a finite fitted operator. It is not a numerical
qualification, nor a claim that two different engines define identical fits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np


def resolve_fit_source(fit_source: str, *, gdf_method: str, dimension: int) -> str:
    """Resolve the historical default without changing generic GDF policy."""
    if not isinstance(fit_source, str) or fit_source not in ("auto", "classic", "sr"):
        raise ValueError("fit_source must be 'auto', 'classic', or 'sr'; None is invalid")
    if fit_source != "auto" and (gdf_method != "rsgdf" or dimension != 3):
        raise NotImplementedError("explicit fit_source requires 3D gdf_method='rsgdf'")
    if gdf_method != "rsgdf":
        return "not-applicable"
    return ("sr" if dimension == 3 else "classic") if fit_source == "auto" else fit_source


CLASSIC_SUPPORT_POLICIES = ("physical_pairs", "historical_index_ball")


def resolve_classic_support_policy(policy, *, fit_screen_threshold=0.0) -> str:
    """AO-pair support of a classic fit that a periodic GDF driver builds (#466).

    ``None`` selects ``'physical_pairs'``: every AO pair within the lattice
    cutoff of each other, the term set the native three-centre integrals use.
    ``'historical_index_ball'`` replays the former cells within the cutoff of
    the origin, which drops the pairs that cross the face of a cell whose
    extent approaches the cutoff (an H8 chain cell at Gamma sat 9.8 mHa per
    pair from its folded k mesh). Producers whose own classic folds use the
    historical ball pin it explicitly so that both halves keep one operator.
    """
    resolved = "physical_pairs" if policy is None else policy
    if resolved not in CLASSIC_SUPPORT_POLICIES:
        raise ValueError(
            "classic_support_policy must be 'physical_pairs' or "
            f"'historical_index_ball'; got {policy!r}")
    if resolved == "physical_pairs" and float(fit_screen_threshold) > 0.0:
        raise NotImplementedError(
            "the physical classic fit is unscreened: fit_screen_threshold > 0 "
            "requires classic_support_policy='historical_index_ball' (whose pairs "
            "depend on the cell, #466) or the default SR/LR fit source")
    return resolved


@dataclass(frozen=True)
class RSGDFFitRequest:
    source: str
    base_reciprocal_cutoff_ha: float
    tail_reciprocal_cutoff_ha: float | None


@dataclass(frozen=True)
class RSGDFFitSource:
    engine: str
    engine_version: int
    source: str
    source_signature: str
    orbital_basis: str
    orbital_basis_sha256: str
    auxiliary_basis: str
    auxiliary_basis_sha256: str
    numerical_support: tuple
    metric_threshold: float
    auxiliary_frame: str
    q_grouping: tuple
    kpoints_cart: tuple
    base_reciprocal_cutoff_ha: float
    executed_tail_reciprocal_cutoff_ha: float | None
    factor_shapes: tuple
    factor_sha256: str
    factor_space: str = "ao-pair"
    schema: str = "vibeqc.rsgdf.executed-source/v1"
    executed_domains_json: str = ""

    def as_dict(self):
        record = asdict(self)
        # Preserve established identities for descriptors predating this contract.
        if not self.executed_domains_json:
            record.pop("executed_domains_json")
        return record

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def _support(options):
    # Read resolved native values, never infer an executed domain from inputs.
    names = ("cutoff_bohr", "nuclear_cutoff_bohr", "pair_complete_1e",
             "schwarz_threshold", "screening_overlap_threshold",
             "screening_exchange_threshold", "sr_range_screening",
             "sr_sparse_traversal", "eri_interaction_cutoff_bohr")
    return tuple((key, getattr(options, key)) for key in names)


def factor_digest(cache):
    digest = hashlib.sha256()
    for pair in sorted(cache):
        value = np.asarray(cache[pair], dtype="<c16")
        digest.update(np.asarray(pair, dtype="<i8").tobytes())
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        # Executed canonical factors are immutable contiguous complex arrays.
        # Hash their existing buffer without another full factor-block copy.
        digest.update(memoryview(value).cast('B') if value.size and value.flags.c_contiguous
                      else value.tobytes())
    return digest.hexdigest()


def attach_fit_source(result, *, system, basis, auxiliary, auxiliary_name,
                      cache, request, selected, lat_opts, oneel_lat_opts,
                      metric_threshold, fit_screen_threshold, kpoints_cart,
                      executed_tail, retain_canonical=False, one_electron_record=None,
                      classic_support_policy=None):
    """Attest only a cache actually built by the selected engine."""
    from .aux_basis import _basis_fingerprint, _canonical_reciprocal_transfer
    from .aux_basis import _range_separated_gdf_source_signature

    result.rsgdf_fit_request = request
    # Explicit finite-source contracts carry their original immutable identity.
    # Do not reconstruct it from requested cutoffs or relabel SR as classic.
    from .periodic_fitted_support import ChiFittedOperator
    if isinstance(cache, ChiFittedOperator):
        from dataclasses import replace
        record = json.loads(cache.descriptor.executed_domains_json)
        record['one_electron'] = json.loads(one_electron_record) if one_electron_record else None
        record['fitted_operator_identity'] = cache.identity
        payload = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if len(payload.encode()) > dict(cache.descriptor.numerical_support)['support_record_byte_cap']:
            raise MemoryError('complete fitted and one-electron support record exceeds budget')
        result.executed_fit_source = replace(cache.descriptor, executed_domains_json=payload)
        result.fitted_operator = cache
        return result
    if not cache or selected == "not-applicable":
        return result
    from .periodic_k_gdf import _RangeSeparatedGdfCache
    is_sr = isinstance(cache, _RangeSeparatedGdfCache)
    if selected == "sr" and not is_sr:
        # Private adapters (e.g. MDF) own their separate execution descriptor.
        return result
    support = (("fit_lattice", _support(lat_opts)),
               ("one_electron_lattice", _support(oneel_lat_opts)),
               ("fit_screen_threshold", float(fit_screen_threshold)))
    if not is_sr and classic_support_policy is not None:
        # The classic fit's AO-pair support (#466); a historical replay and
        # the physical default are different finite operators.
        support += (("classic_support_policy", str(classic_support_policy)),)
    if is_sr:
        support += (("sr_source_parameters", tuple(cache.source_parameters)),
                    ("reciprocal_vector_counts", tuple(cache.reciprocal_vector_counts)),
                    ("q_batch_sizes", tuple(cache.q_batch_sizes)))
    groups = {}
    shapes = []
    kcart = np.asarray(kpoints_cart)
    for pair in sorted(cache):
        value = np.asarray(cache[pair])
        shapes.append((pair, value.shape))
        q = _canonical_reciprocal_transfer(system, kcart[pair[1]] - kcart[pair[0]])
        key = tuple(float(x) for x in np.round(q, 14))
        groups.setdefault(key, []).append(pair)
    descriptor = RSGDFFitSource(
        engine=("native-sr-lr-shared-q" if is_sr else "native-classic-fft-shared-q"),
        engine_version=1, source=selected,
        source_signature=_range_separated_gdf_source_signature(system, basis, auxiliary),
        orbital_basis=str(basis.name), orbital_basis_sha256=_basis_fingerprint(basis),
        auxiliary_basis=str(auxiliary_name), auxiliary_basis_sha256=_basis_fingerprint(auxiliary),
        numerical_support=support, metric_threshold=float(metric_threshold),
        auxiliary_frame=(cache.factor_frame if is_sr else
                         "canonical-modrho" if retain_canonical else "eigenmode-modrho"),
        q_grouping=tuple((q, tuple(groups[q])) for q in sorted(groups)),
        kpoints_cart=tuple(tuple(float(x) for x in k) for k in kcart),
        base_reciprocal_cutoff_ha=(cache.lr_ke_cutoff if is_sr else float(request.base_reciprocal_cutoff_ha)),
        executed_tail_reciprocal_cutoff_ha=executed_tail,
        factor_shapes=tuple(shapes), factor_sha256=factor_digest(cache),
    )
    result.executed_fit_source = descriptor
    if is_sr and cache.execution:
        from copy import deepcopy
        result.rsgdf_q_execution = deepcopy(cache.execution)
    if retain_canonical:
        # SCF and correlation borrow one identified canonical factor source.
        for factor in cache.values():
            factor.setflags(write=False)
        result._rsgdf_canonical_factors = MappingProxyType(dict(cache))
        result._rsgdf_auxiliary = auxiliary
    return result


def classic_posthf_fit_source(system, basis, auxiliary, auxiliary_name,
                             kpoints_cart, factors, *, lattice_cutoff_bohr,
                             rsgdf_ke_cutoff, gdf_linear_dep_threshold, projected,
                             engine="native-classic-fft-pair"):
    """Attest the independent classic builder after it returns actual factors.

    The legacy streaming MP2 route emits occupied-virtual factors. Their hash
    identifies that projection, not an unretained full AO factor cache. This
    correlation source executes no one-electron construction; its reference
    keeps that separate source descriptor.
    """
    from dataclasses import replace
    from types import SimpleNamespace
    from ._vibeqc_core import LatticeSumOptions

    options = LatticeSumOptions()
    options.cutoff_bohr = float(lattice_cutoff_bohr)
    temporary = SimpleNamespace()
    attach_fit_source(temporary, system=system, basis=basis, auxiliary=auxiliary,
        auxiliary_name=auxiliary_name, cache=factors,
        request=RSGDFFitRequest("auto", float(rsgdf_ke_cutoff), None),
        selected="classic", lat_opts=options, oneel_lat_opts=options,
        metric_threshold=gdf_linear_dep_threshold, fit_screen_threshold=0.,
        kpoints_cart=kpoints_cart, executed_tail=None)
    source = temporary.executed_fit_source
    return replace(source, engine=engine,
        auxiliary_frame="canonical-modrho",
        numerical_support=tuple((key, value) for key, value in source.numerical_support
                                if key != "one_electron_lattice"),
        factor_space="occupied-virtual" if projected else "ao-pair")
