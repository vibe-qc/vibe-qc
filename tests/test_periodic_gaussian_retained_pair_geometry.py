"""Actual generation geometry survives moves into domain-generated MP2.

Tiny finite-torus algebra checks the original PAO geometry, not a geometry
reconstructed from an exported PNO matrix. No production scaling claim.
"""

from __future__ import annotations

import gc
import sys
from itertools import combinations_with_replacement

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_pair_domain_builder import (
    _case, _embedding, _embedding_plan, _pair_options, _pair_caps, _live,
)
from tests.test_periodic_gaussian_domain_pair_mp2 import (
    _run, _plan, _controls, _physical,
)
from tests.test_periodic_gaussian_domain_pair_ccsd_t import (
    _run as _ccsd, _plan as _ccsd_plan,
)
from tests.test_periodic_gaussian_triple_spaces import _plan as _triple_plan


def _geometry(b, i=0, j=0, *, options=None, caps=None, **changes):
    args = dict(reference=b.reference, basis=b.basis, builder=b.builder,
        occupied_slot_i=i, occupied_slot_j=j, options=_pair_options() if options is None else options,
        live=_live(), caps=_pair_caps(b) if caps is None else caps)
    args.update(changes)
    return core._make_periodic_gaussian_pair_domain_geometry(**args)


def _geometry_bytes(d, m, n):
    return 16*d+32*d*d+16*d*m+8*(d+m)+8*(n*m+m)


def _physical_columns(b, geometry):
    """Independent full finite-torus Q, with the bundle's actual translation."""
    state, torus = b.reference.state, b.expected_torus
    nk, nao = state.n_kpoints, state.n_basis
    block = np.zeros((nk*nao, nk*nao), complex)
    for k in range(nk):
        c = state.coefficients(k)[:, np.asarray(state.virtual_mask(k), bool)]
        block[k*nao:(k+1)*nao, k*nao:(k+1)*nao] = c@c.conj().T@state.overlap(k)
    projector = torus.fourier@block@torus.fourier.conj().T
    indices = geometry.domain_columns_copy().astype(int)
    mesh = tuple(state.mesh)
    shift = np.array(np.unravel_index(geometry.pair_translation, mesh))
    cells = np.array(np.unravel_index(indices[:, 0].astype(int), mesh)).T
    placed = np.ravel_multi_index(((cells+shift)%mesh).T, mesh)
    return projector[:, placed*nao+indices[:, 1]]@geometry.real_coefficients_copy()


@pytest.mark.parametrize("kind,nk", [("he", 1), ("he", 3), ("he2", 2), ("frozen", 2)])
def test_bundle_preserves_actual_domain_space_and_embedding_bitwise(kind, nk):
    b = _case(nk, kind=kind)
    n = b.basis.memory.virtual_count
    for i, j in combinations_with_replacement(range(len(b.rows)), 2):
        geometry, embedding = _geometry(b, i, j), _embedding(b, i, j)
        d, m = geometry.domain_dimension, geometry.generation_dimension
        assert geometry.retained_numerical_bytes == _geometry_bytes(d, m, n)
        assert geometry.embedding_identity_sha256 == embedding.identity_sha256
        np.testing.assert_array_equal(geometry.embedding_coefficients_copy(), embedding.coefficients_copy())
        np.testing.assert_array_equal(geometry.energies_copy(), embedding.energies_copy())
        actual = _physical_columns(b, geometry)
        expected = b.expected_torus.common@embedding.coefficients_copy()
        np.testing.assert_allclose(actual, expected, atol=7e-10, rtol=4e-10)
        np.testing.assert_allclose(actual.conj().T@b.expected_torus.s@actual, np.eye(m), atol=7e-10)
        np.testing.assert_allclose(actual.conj().T@b.expected_torus.f@actual,
            np.diag(geometry.energies_copy()), atol=7e-10)
        assert geometry.state is b.reference.state and geometry.context is b.context
        assert geometry.builder_identity_sha256 == b.builder.identity_sha256
        assert geometry.basis_identity_sha256 == b.basis.identity_sha256
        assert geometry.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
        for name in ("domain", "real_space", "embedding"):
            assert not hasattr(geometry, name)  # No move-capable child wrapper.


def test_bundle_shape_upper_is_row_independent_and_uses_admitted_ao_cap():
    b = _case(2, kind="he2")
    cap = _pair_caps(b)
    union = cap.pair_union
    union.maximum_ao_columns = 3
    cap.pair_union = union
    for i, j in combinations_with_replacement(range(len(b.rows)), 2):
        p = _embedding_plan(b, i, j, caps=cap)
        assert p.maximum_domain_dimension == 3
        assert p.retained_pair_geometry_upper_bytes == _geometry_bytes(3, 3, 4)
        assert p.retained_geometry_control_upper_bytes > 0
    p = _embedding_plan(b)
    geometry = _geometry(b)
    assert geometry.retained_numerical_bytes <= p.retained_pair_geometry_upper_bytes
    assert geometry.retained_control_storage_bytes == p.retained_geometry_control_upper_bytes


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_control_storage_bytes", "control_storage_reservation_bytes"),
    ("maximum_work_units", "work_units"),
])
def test_geometry_factory_exact_and_minus_one_macro_cap(field, reported):
    b = _case(1, kind="he")
    p, caps = _embedding_plan(b), _pair_caps(b)
    setattr(caps, field, getattr(p, reported))
    _geometry(b, caps=caps)
    setattr(caps, field, getattr(p, reported)-1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _geometry(b, caps=caps)


def test_geometry_owner_is_not_forgeable_and_survives_builder_destruction():
    b = _case(2, kind="he2")
    geometry = _geometry(b, 0, 2)
    expected = geometry.real_coefficients_copy()
    before = sys.getrefcount(geometry)
    view = geometry.geometry_view()
    assert sys.getrefcount(geometry) > before  # The diagnostic view pins the native owner.
    with pytest.raises(TypeError):
        core._PeriodicGaussianPairDomainGeometry()
    del b
    gc.collect()
    np.testing.assert_array_equal(geometry.real_coefficients_copy(), expected)
    assert not expected.flags.writeable
    expected.setflags(write=True)
    expected[:] = np.nan
    assert np.isfinite(geometry.real_coefficients_copy()).all()
    del view


@pytest.mark.parametrize("cutoff", [0.0, 1.0])
def test_mp2_retains_all_pair_geometry_without_duplicate_diagonal_embedding(cutoff):
    b = _case(2, kind="he2")
    result = _run(b, _controls(b, cutoff=cutoff))
    n, o = result.memory.common_virtual_dimension, result.memory.occupied_count
    total = controls = original_geometry = offdiagonal_embeddings = 0
    for i, j in combinations_with_replacement(range(o), 2):
        p = result.pair(i, j)
        g = result.pair_generation_geometry_metadata(i, j)
        d, m = g["domain_dimension"], g["generation_dimension"]
        assert m == p.memory.generation_dimension > 0
        assert g["embedding_identity_sha256"] == p.embedding_identity_sha256
        assert g["retained_numerical_bytes"] == _geometry_bytes(d, m, n)
        total += g["retained_numerical_bytes"]
        controls += g["retained_control_storage_bytes"]
        original_geometry += 16*d+32*d*d+16*d*m+8*(d+m)
        if i == j:
            e = result.diagonal_generation_embedding(i)
            assert e.identity_sha256 == g["embedding_identity_sha256"]
        else:
            offdiagonal_embeddings += 8*(n*m+m)
        view = result.pair_generation_geometry_view(i, j)
        assert isinstance(view, core._PeriodicGaussianPairPNOGeometryView)
    diagnostics = result.diagnostics
    assert diagnostics.retained_pair_geometry_bytes == total <= result.memory.retained_pair_geometry_upper_bytes
    assert diagnostics.retained_pair_geometry_control_bytes == controls == result.memory.retained_pair_geometry_control_upper_bytes
    assert total-diagnostics.retained_generation_embedding_bytes == original_geometry+offdiagonal_embeddings
    if cutoff:
        assert diagnostics.zero_rank_pairs == result.memory.pair_count
        assert diagnostics.retained_generation_coefficient_bytes == 0
        assert total > 0  # Empty PNOs cannot erase their nonempty generation proof.
    with pytest.raises(IndexError, match="canonical"):
        result.pair_generation_geometry_view(1, 0)
    # Drop the loop's prior view first; replacing one pinned view with another
    # leaves the reference count unchanged even when both correctly pin it.
    del view
    before = sys.getrefcount(result)
    view = result.pair_generation_geometry_view(0, 1)
    assert sys.getrefcount(result) > before
    del b, view
    gc.collect()
    assert sys.getrefcount(result) == before
    assert result.pair_generation_geometry_metadata(0, 1)["retained_numerical_bytes"] > 0


def test_downstream_planners_count_whole_geometry_and_keep_diagonal_as_subset():
    b = _physical(1, kind="he2")
    mp2 = _run(b)
    ccplan = _ccsd_plan(b, mp2)
    ccsd = _ccsd(b, mp2)
    expected = mp2.diagnostics.retained_pair_bytes+mp2.solver.memory.output_numerical_bytes+mp2.diagnostics.retained_pair_geometry_bytes
    assert ccplan.borrowed_mp2_numerical_bytes == ccsd.memory.borrowed_mp2_numerical_bytes == expected
    assert ccplan.borrowed_generation_embedding_bytes == mp2.diagnostics.retained_generation_embedding_bytes
    triples = _triple_plan(b, b.provider, mp2, ccsd)
    assert triples.borrowed_mp2_numerical_bytes == expected
    assert triples.borrowed_mp2_control_bytes == ccplan.borrowed_mp2_control_bytes


def test_empty_union_rejects_before_publishing_a_bundle_and_foreign_owner_fails():
    empty = _case(1, kind="he", cut=2.0)
    with pytest.raises((ValueError, RuntimeError), match="empty|rank"):
        _geometry(empty)
    first, second = _case(1, kind="he"), _case(1, kind="he")
    with pytest.raises((ValueError, RuntimeError), match="owner|source|basis|reference|state"):
        _geometry(first, reference=second.reference)


def test_mp2_new_geometry_reservation_is_admitted_before_callback():
    b = _case(1, kind="he")
    controls = _controls(b)
    p = _plan(b, controls)
    assert p.retained_pair_geometry_upper_bytes > p.retained_generation_embedding_upper_bytes
    controls[2].maximum_owned_numerical_bytes = p.peak_owned_numerical_bytes-1
    events = []
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(b, controls, events.append)
    assert not events
