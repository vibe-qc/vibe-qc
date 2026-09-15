"""Actual HF -> owned home domains -> streamed pair PAO embeddings.

Independent finite-torus algebra reconstructs the selected physical virtual
subspace. These tiny He fixtures do not certify asymptotic domain scaling.
"""

from __future__ import annotations

import gc
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _he2_bundle, _he2_controls, _prepare_leaves,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_gaussian_real_local_provider import _make as _provider
from tests.test_periodic_gaussian_occupied_pao_domain import _controls as _occupied_controls, _oracle as _occupied_oracle, _mapping
from tests.test_periodic_correlation_pair_pao_domain import _caps as _union_caps, _cell_algebra, _columns
from tests.test_periodic_correlation_pao_domain import _options as _domain_options
from tests.test_periodic_correlation_real_pao_space import _options as _space_options
from tests.test_periodic_correlation_real_pao_embedding import _options as _embedding_options, _caps as _embedding_caps, _torus


_BUILDER_CAPS = dict(maximum_home_domains="n_home_occupied", maximum_topology_rows="topology_rows",
    maximum_owned_numerical_bytes="peak_owned_numerical_bytes", maximum_control_storage_bytes="control_storage_reservation_bytes",
    maximum_worker_bytes="worker_bytes", maximum_node_bytes="required_node_memory_bytes", maximum_work_units="work_units")
_PAIR_CAPS = {key: value for key, value in _BUILDER_CAPS.items()
              if key not in ("maximum_home_domains", "maximum_topology_rows")}


def _live(**changes):
    live = core._PeriodicGaussianPairDomainBuilderLive()
    # Original tiny input/NumPy oracle owners not consumed by this leaf.
    live.other_live_numerical_bytes_per_worker = 2**20
    live.other_live_control_bytes_per_worker = live.backend_margin_bytes_per_worker = 65536
    for name, value in changes.items():
        setattr(live, name, value)
    return live


def _builder_caps(b):
    caps = core._PeriodicGaussianPairDomainBuilderCaps()
    caps.occupied = _occupied_controls(b)[2]
    caps.maximum_home_domains = b.reference.state.n_correlated_occupied
    caps.maximum_topology_rows = 128
    caps.maximum_owned_numerical_bytes = 2**23
    caps.maximum_control_storage_bytes = 2**22
    caps.maximum_worker_bytes = 2**25
    caps.maximum_node_bytes = 2**26
    caps.maximum_work_units = 10**14
    return caps


def _pair_options():
    options = core._PeriodicGaussianPairDomainOptions()
    options.domain = _domain_options()
    options.real_space = _space_options()
    options.embedding = _embedding_options()
    return options


def _pair_caps(b):
    caps = core._PeriodicGaussianPairDomainCaps()
    union = _union_caps()
    union.maximum_atom_cells = union.maximum_union_atom_cells = b.reference.state.n_kpoints*len(b.system.unit_cell)
    union.maximum_ao_columns = b.reference.state.n_kpoints*b.reference.state.n_basis
    union.maximum_topology_rows = 128
    union.maximum_worker_bytes = 2**25
    caps.pair_union = union
    caps.maximum_pao_domain_owned_numerical_bytes = 2**17
    real = core._PeriodicCorrelationRealPAOSpaceCaps()
    real.maximum_owned_numerical_bytes = 2**17
    real.maximum_work_units = 10**12
    caps.real_space = real
    embedding = _embedding_caps()
    embedding.maximum_common_dimension = embedding.maximum_pair_dimension = 4
    embedding.maximum_owned_numerical_bytes = 2**17
    embedding.maximum_per_worker_inventoried_bytes = 2**25
    caps.embedding = embedding
    caps.maximum_owned_numerical_bytes = 2**23
    caps.maximum_control_storage_bytes = 2**22
    caps.maximum_worker_bytes = 2**25
    caps.maximum_node_bytes = 2**26
    caps.maximum_work_units = 10**14
    return caps


def _builder_plan(b, *, options=None, live=None, caps=None):
    return core._plan_periodic_gaussian_pair_domain_builder(b.hf, b.reference, b.localization,
        b.basis, b.domain, b.real_space, b.domain_options if options is None else options,
        _live() if live is None else live, _builder_caps(b) if caps is None else caps)


def _build(b, *, options=None, live=None, caps=None, **changes):
    args = dict(hf=b.hf, reference=b.reference, ao_basis=b.ao, auxiliary_basis=b.auxiliary,
        minimal_basis=b.minimal, system=b.system, localization=b.localization, basis=b.basis,
        common_domain=b.domain, common_space=b.real_space)
    args.update(changes)
    return core._make_periodic_gaussian_pair_domain_builder(**args,
        options=b.domain_options if options is None else options, live=_live() if live is None else live,
        caps=_builder_caps(b) if caps is None else caps)


def _embedding_plan(b, i=0, j=0, *, options=None, live=None, caps=None):
    return core._plan_periodic_gaussian_pair_domain_embedding(b.reference, b.basis, b.builder, i, j,
        _pair_options() if options is None else options, _live() if live is None else live,
        _pair_caps(b) if caps is None else caps)


def _embedding(b, i=0, j=0, *, options=None, live=None, caps=None):
    return core._make_periodic_gaussian_pair_domain_embedding(b.reference, b.basis, b.builder, i, j,
        _pair_options() if options is None else options, _live() if live is None else live,
        _pair_caps(b) if caps is None else caps)


def _case(nk=2, *, kind="he2", full=False, cut=.2, tail=.1, build=True):
    b = {"he": _bundle, "he2": _he2_bundle, "frozen": _frozen_bundle}[kind]((nk, 1, 1))
    _prepare_leaves(b, localization_options=None if kind == "he" else _he2_controls(b)[0].localization)
    # The source provider must be made before its input geometry moves.
    b.provider = _provider(b)
    b.domain_options = _occupied_controls(b, full=full, cut=cut, tail=tail)[0]
    b.expected_domains = [_occupied_oracle(b, i, b.domain_options)[2]
        for i in range(b.reference.state.n_correlated_occupied)]
    b.expected_mapping = _mapping(b)
    geometry = SimpleNamespace(**vars(b), pair_domain=b.domain, pair_real_space=b.real_space, pair_selected=b.selected)
    b.expected_torus = _torus(geometry)
    if build:
        b.builder = _build(b)
    return b


def _placed_columns(b, i, j):
    _, translate = _cell_algebra(b.reference.state.mesh)
    atoms = set()
    for slot in (i, j):
        occupied, cell = b.rows[slot]
        atoms |= {(translate(r, cell), int(atom)) for r, atom in b.expected_domains[occupied]}
    return _columns(atoms, b.expected_mapping)


def _projected_oracle(b, i, j, options=None):
    options = _pair_options() if options is None else options
    columns = _placed_columns(b, i, j)
    assert len(columns)
    p, state = b.expected_torus, b.reference.state
    nk, nao = state.n_kpoints, state.n_basis
    qk = np.zeros((nk*nao, nk*nao), complex)
    for k in range(nk):
        c, s = state.coefficients(k), state.overlap(k)
        v = c[:, np.asarray(state.virtual_mask(k), bool)]
        qk[k*nao:(k+1)*nao, k*nao:(k+1)*nao] = v@v.conj().T@s
    q = p.fourier@qk@p.fourier.conj().T
    raw = q[:, columns[:, 0]*nao+columns[:, 1]]
    metric = raw.conj().T@p.s@raw
    eigenvalues, vectors = np.linalg.eigh(metric)
    algebra = options.real_space.algebra
    cutoff = max(algebra.rank_absolute_cutoff, algebra.rank_relative_cutoff*max(eigenvalues))
    keep = eigenvalues > cutoff
    orthonormal = raw@(vectors[:, keep]/np.sqrt(eigenvalues[keep]))
    fock = orthonormal.conj().T@p.f@orthonormal
    energies, rotation = np.linalg.eigh(fock)
    coefficients = orthonormal@rotation
    x = p.common.conj().T@p.s@coefficients
    return columns, x, energies


def _check_embedding(b, result, i, j):
    columns, expected, energies = _projected_oracle(b, i, j)
    x = result.coefficients_copy()
    # Canonical union/translation can choose different signs or degenerate
    # eigenvectors. Compare the actual physical subspace, not those gauges.
    np.testing.assert_allclose(x@x.T, expected@expected.conj().T, atol=6e-10, rtol=3e-10)
    np.testing.assert_allclose(result.energies_copy(), energies, atol=6e-10, rtol=3e-10)
    np.testing.assert_allclose(x.T@x, np.eye(len(energies)), atol=6e-10)
    fock = b.basis.fock_copy()[len(b.rows):, len(b.rows):]
    np.testing.assert_allclose(x.T@fock@x, np.diag(result.energies_copy()), atol=6e-10)
    assert result.memory.pair_dimension == len(energies)
    assert result.common_basis_identity_sha256 == b.basis.identity_sha256
    assert result.state is b.reference.state
    assert not result.production_distinct_domain_scaling
    return len(columns), len(energies)


@pytest.mark.parametrize("kind,nk", [("he",1), ("he",2), ("he",3), ("he2",1), ("he2",2)])
@pytest.mark.parametrize("full", [False, True])
def test_actual_domains_every_placed_pair_and_reverse_match_independent_torus(kind, nk, full):
    b = _case(nk, kind=kind, full=full)
    plans = []
    for i, j in product(range(len(b.rows)), repeat=2):
        result = _embedding(b, i, j)
        _check_embedding(b, result, i, j)
        if full:
            assert result.memory.pair_dimension == b.basis.memory.virtual_count
        plans.append(_embedding_plan(b, i, j))
    for field in ("peak_owned_numerical_bytes", "work_units", "control_storage_reservation_bytes",
                  "required_node_memory_bytes", "maximum_generation_dimension"):
        assert len({getattr(plan, field) for plan in plans}) == 1
    builder = b.builder
    assert builder.state is b.hf.state and builder.context is b.context
    assert builder.basis_identity_sha256 == b.basis.identity_sha256
    assert builder.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
    assert builder.localization_identity_sha256 == b.localization.localization_identity_sha256
    assert builder.diagnostics.constructed_home_domains == b.hf.state.n_correlated_occupied
    assert builder.retained_numerical_bytes == builder.diagnostics.retained_numerical_bytes
    assert builder.retained_numerical_bytes <= builder.memory.retained_numerical_upper_bytes
    assert not builder.production_distinct_domain_scaling
    for name in ("common_domain", "common_space", "domains", "topology"):
        assert not hasattr(builder, name)
    with pytest.raises((ValueError, RuntimeError, IndexError), match="moved|malformed|range|state|owner"):
        b.domain.column(0)
    with pytest.raises((ValueError, RuntimeError), match="moved|malformed|state|owner"):
        b.real_space.space.coefficients_copy()


def test_distinct_same_cell_and_even_nyquist_domains_are_not_full_common():
    b = _case(2, kind="he2", tail=2.0)
    dimensions = []
    for i, j in ((0,0), (0,1), (0,2), (2,0), (2,2)):
        columns, rank = _check_embedding(b, _embedding(b,i,j), i,j)
        dimensions.append((columns,rank))
    assert min(rank for _,rank in dimensions) < b.basis.memory.virtual_count
    np.testing.assert_array_equal(_placed_columns(b,0,2), _placed_columns(b,2,0))
    assert not np.array_equal(_placed_columns(b,0,0), _placed_columns(b,2,2))


def test_empty_selected_domain_rejects_embedding_without_forced_orbitals():
    b = _case(2, kind="he", cut=2.0)
    assert all(len(domain) == 0 for domain in b.expected_domains)
    receipt = b.builder.payload_sha256
    _embedding_plan(b)
    with pytest.raises((ValueError, RuntimeError), match="empty|zero|usable|rank"):
        _embedding(b)
    assert b.builder.payload_sha256 == receipt


@pytest.mark.parametrize("nk", [1, 2])
def test_frozen_occupied_orbitals_are_not_home_domains_or_retained_virtuals(nk):
    b = _case(nk, kind="frozen")
    assert b.reference.state.n_frozen_core == b.reference.state.n_correlated_occupied == 1
    assert b.builder.diagnostics.constructed_home_domains == 1
    for i, j in product(range(len(b.rows)), repeat=2):
        _check_embedding(b, _embedding(b,i,j), i,j)


def test_exact_retained_payload_and_external_live_inventory():
    b = _case(2, kind="he", build=False)
    p = _builder_plan(b)
    geometry = (b.domain.memory.retained_domain_index_bytes+b.domain.memory.retained_matrix_bytes
                +b.space.memory.output_numerical_bytes)
    assert p.common_geometry_numerical_bytes == geometry
    assert p.atom_mapping_bytes == 8*b.reference.state.n_basis
    assert p.topology_row_bytes == 32*p.topology_rows
    assert p.retained_numerical_upper_bytes == geometry+p.atom_mapping_bytes+p.topology_row_bytes+p.home_domain_output_upper_bytes
    extra = _builder_plan(b, live=_live(other_live_numerical_bytes_per_worker=2**20+137,
        other_live_control_bytes_per_worker=65536+241))
    assert extra.worker_bytes-p.worker_bytes == 378
    assert extra.required_node_memory_bytes-p.required_node_memory_bytes == 378*p.replicas_per_node
    b.builder = _build(b)
    d = b.builder.diagnostics
    assert d.retained_numerical_bytes == geometry+p.atom_mapping_bytes+p.topology_row_bytes+d.retained_home_domain_bytes
    pair = _embedding_plan(b)
    assert pair.borrowed_builder_numerical_bytes == d.retained_numerical_bytes
    assert pair.borrowed_basis_numerical_bytes == b.basis.memory.retained_output_bytes
    assert pair.retained_embedding_upper_bytes == 8*(p.common_virtual_dimension**2+p.common_virtual_dimension)
    more = _embedding_plan(b, live=_live(other_live_numerical_bytes_per_worker=2**20+137,
        other_live_control_bytes_per_worker=65536+241))
    assert more.worker_bytes-pair.worker_bytes == 378
    assert more.peak_owned_numerical_bytes == pair.peak_owned_numerical_bytes


@pytest.mark.parametrize("field,reported", _BUILDER_CAPS.items())
def test_builder_caps_before_invalid_physical_scan_and_final_owner_transfer(field, reported):
    b = _case(1, kind="he", build=False)
    caps, p = _builder_caps(b), _builder_plan(b)
    domain_matrix, coefficients = b.domain.overlap_copy(), b.real_space.space.coefficients_copy()
    required = getattr(p, reported)
    setattr(caps, field, required-1)
    lattice = b.system.lattice.copy()
    broken = lattice.copy()
    broken[0,0] = np.nan
    b.system.lattice = broken
    with pytest.raises((ValueError, RuntimeError), match="cap|limit"):
        _build(b, caps=caps)
    np.testing.assert_array_equal(b.domain.overlap_copy(), domain_matrix)
    np.testing.assert_array_equal(b.real_space.space.coefficients_copy(), coefficients)
    b.system.lattice = lattice
    setattr(caps, field, required)
    builder = _build(b, caps=caps)
    assert builder.retained_numerical_bytes > 0


@pytest.mark.parametrize("field,reported", _PAIR_CAPS.items())
def test_pair_caps_exact_and_minus_one_do_not_consume_builder(field, reported):
    b = _case(1, kind="he")
    caps, p = _pair_caps(b), _embedding_plan(b)
    identity = b.builder.identity_sha256
    setattr(caps, field, getattr(p, reported)-1)
    with pytest.raises((ValueError, RuntimeError), match="cap|limit"):
        _embedding(b, caps=caps)
    assert b.builder.identity_sha256 == identity
    setattr(caps, field, getattr(p, reported))
    _check_embedding(b, _embedding(b, caps=caps), 0,0)


@pytest.mark.parametrize("kind", ["hf", "reference", "localization", "basis", "ao", "auxiliary", "minimal", "cell", "atom_map"])
def test_foreign_origins_or_changed_actual_inputs_preserve_common_geometry(kind):
    b = _case(1, kind="he2", build=False)
    changes = {}
    if kind in ("hf", "reference", "localization", "basis"):
        other = _case(1, kind="he2", build=False)
        changes[kind] = getattr(other,kind)
    elif kind in ("ao", "auxiliary", "minimal"):
        changes[kind+"_basis"] = b.ao if kind == "minimal" else b.minimal
    elif kind == "atom_map":
        reversed_atoms = core.Molecule(list(reversed(b.system.unit_cell)))
        changes["ao_basis"] = core.BasisSet(reversed_atoms,b.ao.shells(),"remapped atoms",True)
    else:
        lattice = b.system.lattice.copy()
        lattice[0,0] += 1e-14
        b.system.lattice = lattice
    c, s = b.real_space.space.coefficients_copy(), b.domain.overlap_copy()
    with pytest.raises((ValueError, RuntimeError), match="owner|source|state|basis|physical|lattice|mapping|map|census|dimension|lineage|receipt"):
        _build(b, **changes)
    np.testing.assert_array_equal(b.real_space.space.coefficients_copy(), c)
    np.testing.assert_array_equal(b.domain.overlap_copy(), s)


def test_builder_keeps_no_original_input_or_basis_pointer_and_embedding_owns_output():
    b = _case(2, kind="he")
    caps, options, live = _pair_caps(b), _pair_options(), _live()
    reference, basis, builder = b.reference, b.basis, b.builder
    before = _embedding(b).coefficients_copy()
    del b
    gc.collect()
    result = core._make_periodic_gaussian_pair_domain_embedding(reference,basis,builder,0,0,options,live,caps)
    np.testing.assert_array_equal(result.coefficients_copy(), before)
    identity, expected = result.identity_sha256, result.coefficients_copy()
    del basis,builder,reference
    gc.collect()
    np.testing.assert_array_equal(result.coefficients_copy(), expected)
    assert result.identity_sha256 == identity
    expected[:] = np.nan
    assert np.all(np.isfinite(result.coefficients_copy()))


def test_embedding_rejects_an_equal_content_different_native_state():
    b, other = _case(1, kind="he"), _case(1, kind="he")
    assert b.reference.state.numerical_payload_sha256 == other.reference.state.numerical_payload_sha256
    with pytest.raises((ValueError, RuntimeError), match="owner|state|source|basis|frame|reference"):
        core._make_periodic_gaussian_pair_domain_embedding(other.reference,other.basis,b.builder,0,0,
            _pair_options(),_live(),_pair_caps(other))
    _check_embedding(b, _embedding(b), 0,0)


@pytest.mark.parametrize("i,j", [(4,0), (0,4), (2**63,0)])
def test_bad_placed_occupied_slots_fail_without_traversal(i,j):
    b = _case(1, kind="he")
    with pytest.raises((ValueError, RuntimeError, IndexError), match="slot|occupied|range|index"):
        _embedding_plan(b,i,j)
