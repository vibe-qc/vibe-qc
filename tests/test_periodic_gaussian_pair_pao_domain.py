"""Actual-HF initial pair domains: native atom mapping and torus unions."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_occupied_pao_domain import (
    _bundle, _he2_bundle, _localized, _controls as _occupied_controls,
    _run as _occupied, _mapping,
)
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle
from tests.test_periodic_correlation_pair_pao_domain import (
    _inventory, _caps as _union_caps, _cell_algebra, _columns, _placed_atom_domain,
)


def _case(nk=2, *, kind="he2", full=False, cut=.2):
    b = _localized({"he": _bundle, "he2": _he2_bundle, "frozen": _frozen_bundle}[kind]((nk, 1, 1)))
    b.domains = [_occupied(b, i, _occupied_controls(b, full=full, cut=cut))
                 for i in range(b.hf.state.n_correlated_occupied)]
    b.topology = core._make_periodic_correlation_translation_pair_topology(b.reference)
    return b


def _caps():
    caps = core._PeriodicGaussianPairPAODomainCaps()
    caps.pair_union = _union_caps()
    caps.maximum_owned_numerical_bytes = 1 << 20
    caps.maximum_worker_bytes = 1 << 24
    caps.maximum_node_bytes = 1 << 26
    caps.maximum_work_units = 10**13
    return caps


def _arguments(b, row, **changes):
    pair = b.topology.row(row)
    args = dict(hf=b.hf, reference=b.reference, topology=b.topology, row_index=row,
                home=b.domains[pair.home_orbital], partner=b.domains[pair.partner_orbital])
    args.update(changes)
    return args


def _plan(b, row=0, *, caps=None, inventory=None, **changes):
    return core._plan_periodic_gaussian_pair_pao_domain(**_arguments(b, row, **changes),
        inventory=_inventory() if inventory is None else inventory, caps=_caps() if caps is None else caps)


def _run(b, row=0, *, caps=None, inventory=None, **changes):
    args = dict(ao_basis=b.ao, auxiliary_basis=b.auxiliary, system=b.system, **_arguments(b, row))
    args.update(changes)
    return core._make_periodic_gaussian_pair_pao_domain(**args,
        inventory=_inventory() if inventory is None else inventory, caps=_caps() if caps is None else caps)


@pytest.mark.parametrize("kind,nk", [("he",1),("he",2),("he",3),("he2",1),("he2",2)])
@pytest.mark.parametrize("full", [False,True])
def test_actual_hf_pair_union_matches_independent_placed_atomic_domains(kind, nk, full):
    b = _case(nk, kind=kind, full=full)
    _, translate = _cell_algebra(b.hf.state.mesh)
    for row in range(b.topology.row_count):
        pair = b.topology.row(row)
        result = _run(b, row)
        expected = (_placed_atom_domain(b.domains[pair.home_orbital].domain, 0, translate)
                    | _placed_atom_domain(b.domains[pair.partner_orbital].domain,
                                          pair.translation_linear_index, translate))
        np.testing.assert_array_equal(result.domain.atoms_copy(), np.array(sorted(expected), np.uint64).reshape(-1,2))
        np.testing.assert_array_equal(result.domain.columns_copy(), _columns(expected, _mapping(b)))
        assert result.context is b.context
        assert result.matched_finite_gaussian_hf_recipe
        assert not result.production_dlpno and not result.infinite_source_accuracy_certified
        assert not result.domain.extended_ccsd_domain and not result.domain.hf_basis_source_authenticated
        assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
        assert result.localization_identity_sha256 == b.localization.localization_identity_sha256
        assert result.domain.topology_identity_sha256 == b.topology.topology_identity_sha256
        assert result.domain.row_index == row


@pytest.mark.parametrize("nk", [1,2])
def test_frozen_core_endpoints_use_active_home_labels(nk):
    b = _case(nk, kind="frozen")
    assert b.hf.state.n_frozen_core == b.hf.state.n_correlated_occupied == 1
    for row in range(b.topology.row_count):
        result = _run(b, row)
        assert result.domain.row.home_orbital == result.domain.row.partner_orbital == 0


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes","peak_owned_numerical_bytes"),
    ("maximum_worker_bytes","worker_bytes"),("maximum_node_bytes","required_node_memory_bytes"),
    ("maximum_work_units","work_units"),
])
def test_exact_outer_caps_and_minus_one_before_invalid_original_geometry(field, reported):
    b, caps = _case(), _caps()
    p = _plan(b, caps=caps)
    assert p.atom_mapping_bytes == 8*b.ao.nbasis == p.pair_union.caller_mapping_bytes
    assert p.peak_owned_numerical_bytes == p.atom_mapping_bytes+p.pair_union.peak_owned_numerical_bytes
    assert p.worker_bytes == p.pair_union.worker_bytes
    assert p.required_node_memory_bytes == p.pair_union.required_node_memory_bytes
    assert p.work_units == p.pair_union.planned_work_units+p.physical_input_validation_work_units+p.mapping_work_units
    required = getattr(p, reported)
    setattr(caps, field, required)
    _run(b, caps=caps)
    lattice = b.system.lattice.copy()
    lattice[0,0] = np.nan
    b.system.lattice = lattice
    setattr(caps, field, required-1)
    with pytest.raises((ValueError,RuntimeError), match="cap"):
        _run(b, caps=caps)


def test_address_not_equal_content_dedup_and_explicit_other_live_roles():
    b = _case()
    duplicate = _occupied(b, 0, _occupied_controls(b))
    assert duplicate.identity_sha256 == b.domains[0].identity_sha256
    one, two = _plan(b), _plan(b, partner=duplicate)
    assert one.pair_union.distinct_domain_owners == 1 and two.pair_union.distinct_domain_owners == 2
    assert two.pair_union.borrowed_domain_numerical_bytes == 2*one.pair_union.borrowed_domain_numerical_bytes
    assert two.additional_control_storage_bytes > one.additional_control_storage_bytes
    assert _run(b).identity_sha256 == _run(b, partner=duplicate).identity_sha256
    extra = _plan(b, inventory=_inventory(other_live_numerical_bytes=137, other_live_control_bytes=241))
    assert extra.worker_bytes-one.worker_bytes == 378
    assert extra.peak_owned_numerical_bytes == one.peak_owned_numerical_bytes


@pytest.mark.parametrize("kind", ["hf","reference","topology","home","partner","ao","auxiliary","cell","ao_map"])
def test_exact_source_owners_and_original_gaussian_and_atom_map_receipts(kind):
    b = _case()
    changes = {}
    if kind in ("hf","reference","topology","home","partner"):
        other = _case()
        changes[kind] = other.domains[0] if kind in ("home","partner") else getattr(other,kind)
    elif kind == "ao":
        changes["ao_basis"] = b.minimal
    elif kind == "auxiliary":
        changes["auxiliary_basis"] = b.minimal
    elif kind == "ao_map":
        # Identical Gaussian shells and physical centres, deliberately
        # reassigned atom ownership. HF integral identity alone cannot see it.
        reversed_atoms = core.Molecule(list(reversed(b.system.unit_cell)))
        changes["ao_basis"] = core.BasisSet(reversed_atoms,b.ao.shells(),"remapped atoms",True)
    else:
        lattice = b.system.lattice.copy()
        lattice[0,0] += 1e-14
        b.system.lattice = lattice
    with pytest.raises((ValueError,RuntimeError), match="owner|source|state|basis|physical|lattice|mapping|map|census|dimension"):
        _run(b, **changes)


def test_same_orbital_changed_cut_rejected_even_when_selected_columns_agree():
    b = _case(full=True)
    changed = _occupied(b, 0, _occupied_controls(b, full=True, cut=.3))
    row = next(i for i in range(b.topology.row_count)
               if b.topology.row(i).home_orbital == b.topology.row(i).partner_orbital == 0
               and b.topology.row(i).translation_linear_index == 1)
    with pytest.raises(ValueError, match="same-orbital.*identical"):
        _run(b,row,partner=changed)


def test_different_occupied_endpoints_can_have_different_scientific_cuts():
    b = _case()
    changed = _occupied(b, 1, _occupied_controls(b, cut=2.0))
    row = next(i for i in range(b.topology.row_count)
               if b.topology.row(i).home_orbital == 0 and b.topology.row(i).partner_orbital == 1)
    result = _run(b,row,partner=changed)
    assert result.domain.partner_domain_identity_sha256 == changed.domain.occupied_pao_domain_identity_sha256


def test_empty_union_and_output_does_not_borrow_original_localizer_or_endpoint_payloads():
    b = _case(cut=2.0)
    del b.localization, b.minimal
    result = _run(b)
    domain = result.domain
    assert domain.atoms_copy().shape == domain.columns_copy().shape == (0,2)
    del b,result
    gc.collect()
    assert domain.columns_copy().shape == (0,2)


def test_owned_nonempty_output_copies_are_isolated_after_source_release():
    b = _case()
    result = _run(b)
    domain = result.domain
    expected = domain.columns_copy()
    assert len(expected)
    domain.columns_copy()[:] = 99
    del b,result
    gc.collect()
    np.testing.assert_array_equal(domain.columns_copy(),expected)
