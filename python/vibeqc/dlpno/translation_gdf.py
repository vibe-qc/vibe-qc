"""Experimental blockwise translation sources from retained generic GDF.

Nejad et al., doi:10.1063/5.0290816, Sec. II, Eqs. (2)-(6), (15)-(16),
and Ye and Berkelbach, doi:10.1021/acs.jctc.4c00936, Sec. II, Eq. (1),
motivate the finite Bloch/supercell equivalence. This adapter preserves the
producer's finite fitted Hamiltonian; it adds neither paper's local-orbital
approximation. Factor authentication is a separate input contract.
"""

from collections.abc import Mapping
from copy import copy
from math import prod
from numbers import Real

import numpy as np

from .source import CorrelationSourceError, SourceConventions, _real_torus_source
from .translation_bloch import bloch_translation_reference
from ..periodic_fit_source import RSGDFFitRequest, RSGDFFitSource, factor_digest

__all__ = ["gdf_translation_source"]


def _numerical_support(source):
    """Validate the schema of the executed support, including cache-free use."""
    lattice_keys = {"cutoff_bohr", "nuclear_cutoff_bohr", "pair_complete_1e",
                    "schwarz_threshold", "screening_overlap_threshold",
                    "screening_exchange_threshold", "sr_range_screening",
                    "sr_sparse_traversal", "eri_interaction_cutoff_bohr"}
    try:
        support = dict(source.numerical_support)
        valid = (len(support) == len(source.numerical_support)
                 and set(support) == {"fit_lattice", "one_electron_lattice", "fit_screen_threshold", "classic_support_policy"}
                 and support["classic_support_policy"] in ("physical_pairs", "historical_index_ball"))
        for name in ("fit_lattice", "one_electron_lattice"):
            record = dict(support[name])
            valid = (valid and len(record) == len(support[name]) and set(record) == lattice_keys
                     and all(isinstance(value, Real) and np.isfinite(value) for value in record.values()))
        screen = support["fit_screen_threshold"]
        valid = (valid and isinstance(screen, Real) and np.isfinite(screen) and screen >= 0
                 and (support["classic_support_policy"] != "physical_pairs" or screen == 0))
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise CorrelationSourceError("generic translation source lacks supported executed numerical support")
    return support


def _retained_source(system, basis, result, mesh, auxiliary_name):
    """Validate the executed canonical cache, never rotate a legacy q metric."""
    from ..aux_basis import (
        _basis_fingerprint, _canonical_reciprocal_transfer,
        _range_separated_gdf_source_signature,
    )

    source = getattr(result, "executed_fit_source", None)
    factors = getattr(result, "_rsgdf_canonical_factors", None)
    auxiliary = getattr(result, "_rsgdf_auxiliary", None)
    if not isinstance(source, RSGDFFitSource) or not isinstance(factors, Mapping) or auxiliary is None:
        raise CorrelationSourceError("generic translation source requires attested retained canonical SCF factors")
    if (source.schema != "vibeqc.rsgdf.executed-source/v1"
            or source.engine != "native-classic-fft-shared-q" or source.engine_version != 1
            or source.source != "classic" or source.auxiliary_frame != "canonical-modrho"
            or source.factor_space != "ao-pair" or source.executed_domains_json):
        raise CorrelationSourceError("unsupported generic translation source engine/frame/factor space")
    n_cells = prod(mesh)
    pairs = {(i, j) for i in range(n_cells) for j in range(n_cells)}
    shape = (int(auxiliary.nbasis), int(basis.nbasis), int(basis.nbasis))
    if set(factors) != pairs or any(np.shape(factors[pair]) != shape
                                   or not np.isfinite(factors[pair]).all() for pair in pairs):
        raise CorrelationSourceError("generic translation source needs complete finite canonical AO pairs")
    cart = np.asarray(getattr(result, "kpoints_cart", []), dtype=float)
    if cart.shape != (n_cells, 3) or not np.isfinite(cart).all():
        raise CorrelationSourceError("generic translation source has an invalid Cartesian mesh")
    groups = {}
    for i, j in sorted(pairs):
        q = _canonical_reciprocal_transfer(system, cart[j]-cart[i])
        key = tuple(float(x) for x in np.round(q, 14))
        groups.setdefault(key, []).append((i, j))
    if (source.orbital_basis != basis.name
            or source.orbital_basis_sha256 != _basis_fingerprint(basis)
            or source.auxiliary_basis != auxiliary_name
            or source.auxiliary_basis_sha256 != _basis_fingerprint(auxiliary)
            or source.source_signature != _range_separated_gdf_source_signature(system, basis, auxiliary)
            or source.factor_sha256 != factor_digest(factors)
            or source.factor_shapes != tuple((pair, shape) for pair in sorted(pairs))
            or not np.array_equal(np.asarray(source.kpoints_cart), cart)
            or source.q_grouping != tuple((q, tuple(groups[q])) for q in sorted(groups))):
        raise CorrelationSourceError("generic translation source descriptor/factors/basis/mesh mismatch")
    support = _numerical_support(source)
    request = getattr(result, "rsgdf_fit_request", None)
    if (not isinstance(request, RSGDFFitRequest) or request.source != "classic"
            or request.base_reciprocal_cutoff_ha != source.base_reciprocal_cutoff_ha
            or request.tail_reciprocal_cutoff_ha is not None
            or source.executed_tail_reciprocal_cutoff_ha is not None
            or getattr(result, "rsgdf_tail_ke_cutoff", None) is not None
            or getattr(result, "rsgdf_ke_cutoff", None) != source.base_reciprocal_cutoff_ha
            or not np.isfinite(source.base_reciprocal_cutoff_ha) or source.base_reciprocal_cutoff_ha <= 0
            or not np.isfinite(source.metric_threshold) or source.metric_threshold <= 0
            or getattr(result, "n_aux", None) != shape[0]
            or getattr(result, "gdf_cderi_pairs", None) != n_cells*n_cells
            or getattr(result, "aux_basis_name", None) != auxiliary_name):
        raise CorrelationSourceError("generic translation source controls/tail/dimensions mismatch")
    lindep = getattr(result, "linear_dependence", None)
    if lindep is None or lindep.aux_threshold != source.metric_threshold:
        raise CorrelationSourceError("generic translation source metric threshold mismatch")
    # The old cache is optional and never consumed. If present, contradictory
    # redundant records are refused, rather than silently selecting one copy.
    legacy = getattr(result, "fit_cache", None)
    if legacy is not None:
        expected = {
            "frame": "canonical", "ke_cutoff": source.base_reciprocal_cutoff_ha,
            "tail_ke_cutoff": None, "aux_basis_name": auxiliary_name,
            "n_aux": shape[0], "n_ao": shape[1], "need_k_pairs": True,
            "linear_dep_thr": source.metric_threshold,
            "lat_cutoff_bohr": dict(support["fit_lattice"])["cutoff_bohr"],
            "fit_screen_threshold": support["fit_screen_threshold"],
            "support_policy": support["classic_support_policy"],
        }
        if (any(getattr(legacy, key, None) != value for key, value in expected.items())
                or not np.array_equal(legacy.kpoints_cart, cart)
                or source.source_signature != _range_separated_gdf_source_signature(legacy.system, basis, auxiliary)
                or factor_digest(legacy.pairs) != source.factor_sha256):
            raise CorrelationSourceError("generic translation source retained cache/support mismatch")
    return source, factors, support


def gdf_translation_source(system, basis, result, *, mesh, aux_basis_name=None, n_frozen=0,
                            compact_occupied=False):
    """Adapt a complete generic classic RI-RHF result without dense AO folds.

    The result must retain the canonical factors attested by its executed
    source (``fit_source='classic', k_exchange='gdf'``). ``retain_fit_cache``
    is optional. Both classic support policies keep their actual identity.
    ``n_frozen`` is the total torus count and must contain complete bands;
    pass the same count to the translation MP2 engine. Caller state is read
    only. A new compact factor transform is prepared on each call.
    ``compact_occupied=True`` retains immutable frozen/active projectors
    instead of the full real canonical coefficients, after the same audit.

    Initially this accepts full-rank, real-compatible 3D restricted references
    with BvK Ewald exchange. SR/eigenmode, explicit fitted-support, corrected
    COSX, ECP and symmetry-unfolded references require separate adapters.
    Existing molecular, Gamma and dense BvK entry points remain unchanged.
    """
    from ..periodic.chi.pno import _validated_pair_mesh
    from ..periodic.chi.reference import orbital_residuals
    from ..periodic.chi.translation import _translation_real_reference
    from ..periodic.exchange_convention import BVK_EWALD

    if type(compact_occupied) is not bool:
        raise ValueError('compact_occupied must be a boolean')
    mesh = _validated_pair_mesh(mesh)
    if (int(system.dim) != 3 or int(system.multiplicity) != 1
            or getattr(result, "backend", None) != "native-multi-k-gdf-gdf-rhf"
            or getattr(result, "returned_operator", "iterated") != "iterated"
            or getattr(result, "exchange_q0", None) != BVK_EWALD
            or int(getattr(result, "ecp_total_ncore", 0) or 0)):
        raise CorrelationSourceError("generic translation source requires unscreened 3D all-electron iterated GDF RHF with BvK-ewald exchange")
    auxiliary_name = aux_basis_name or getattr(result, "aux_basis_name", None)
    descriptor, factors, support = _retained_source(system, basis, result, mesh, auxiliary_name)
    # Lattice vectors are columns: k_frac(row) = k_cart(row) A / (2*pi).
    # Generic GDF need not store k_frac; derive from the actual coordinates,
    # then let the shared Bloch admission compare both against the cyclic mesh.
    view = copy(result)
    if not hasattr(view, "kpoints_frac"):
        view.kpoints_frac = np.asarray(result.kpoints_cart) @ np.asarray(system.lattice) / (2*np.pi)
    folded, transform = bloch_translation_reference(
        system, basis, view, mesh, n_frozen=n_frozen, compact_occupied=compact_occupied)
    n_occ = int(system.n_electrons())//2
    if np.shape(getattr(result, "density", None)) != (prod(mesh), int(basis.nbasis), int(basis.nbasis)):
        raise CorrelationSourceError("generic translation reference needs its accepted density")
    residuals = dict(orbital_residuals(result, n_occ))
    if any(not np.isfinite(value) or value > 1e-8 for value in residuals.values()):
        raise CorrelationSourceError("generic translation reference density/orbitals/Fock are inconsistent")
    real = _translation_real_reference(system, basis, mesh, folded, transform, factors, auxiliary_name)
    provider = real.df
    return _real_torus_source(
        real.super_system, real.basis, folded, provider, kind="gdf-translation-blocks",
        conventions=SourceConventions(exchange_q0=result.exchange_q0, kmesh=mesh,
                                      bloch_normalisation="real-bvk-supercell", same_source=True),
        factor_tensor=provider.home_factors, factor_representation="chi-home-auxiliary",
        aux_basis_name=auxiliary_name,
        provenance={"adapter": "gdf-translation-blocks-v1", "scf_fit": descriptor.identity,
                    "correlation_fit": descriptor.identity, "executed_fit_source": descriptor.as_dict(),
                    "classic_support_policy": support["classic_support_policy"],
                    "bloch_transform": transform, "orbital_residuals": residuals,
                    "factor_identity": provider.identity, "n_frozen": int(n_frozen),
                    "factor_imaginary_residual": real.cderi_imaginary_residual,
                    "factor_symmetry_residual": real.cderi_symmetry_residual,
                    "quantitative_status": "not-qualified"},
    )
