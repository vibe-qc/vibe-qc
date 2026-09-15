"""Exact tiny translated unions of genuine native occupied-domain owners."""

from __future__ import annotations

import gc
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_occupied_pao_domain import (
    _bundle, _options as _domain_options, _run as _occupied_domain,
)
from tests.test_periodic_correlation_pao_domain import _make as _make_geometry


def _inventory(**changes):
    out = core._PeriodicCorrelationPairPAODomainInventory()
    out.backend_allowance_bytes = 4096
    for name, value in changes.items():
        setattr(out, name, value)
    return out


def _caps(**changes):
    out = core._PeriodicCorrelationPairPAODomainCaps()
    out.maximum_atom_cells = out.maximum_union_atom_cells = 64
    out.maximum_ao_columns = 96
    out.maximum_topology_rows = 512
    out.maximum_owned_numerical_bytes = 1 << 20
    out.maximum_control_storage_bytes = 1 << 20
    out.maximum_worker_bytes = 8 << 20
    out.maximum_work_units = 10**12
    for name, value in changes.items():
        setattr(out, name, value)
    return out


def _case(mesh=(1, 1, 1), *, nactive=2, full=False, grouped=False):
    bundle = _bundle(mesh, nactive=nactive)
    if grouped:
        bundle["mapping"] = np.minimum(bundle["mapping"], 1).astype(np.uint64)
        bundle["atom_count"] = 2
    domains = [_occupied_domain(bundle, occupied=i, options=_domain_options(
        full_domain=full, mulliken_population_cutoff=0.02+0.01*i, pao_tail_cutoff=0.2+0.1*i))
        for i in range(nactive)]
    topology = core._make_periodic_correlation_translation_pair_topology(bundle["reference"])
    return dict(bundle=bundle, domains=domains, topology=topology)


def _arguments(case, row, home=None, partner=None):
    pair = case["topology"].row(row)
    return (case["bundle"]["reference"], case["topology"], row,
            case["domains"][pair.home_orbital] if home is None else home,
            case["domains"][pair.partner_orbital] if partner is None else partner)


def _plan(case, row=0, *, home=None, partner=None, inventory=None, caps=None):
    return core._plan_periodic_correlation_pair_pao_domain(*_arguments(case,row,home,partner),
        case["bundle"]["atom_count"], _inventory() if inventory is None else inventory,
        _caps() if caps is None else caps)


def _run(case, row=0, *, home=None, partner=None, mapping=None, inventory=None, caps=None):
    return core._make_periodic_correlation_pair_pao_domain(*_arguments(case,row,home,partner),
        case["bundle"]["mapping"] if mapping is None else mapping, case["bundle"]["atom_count"],
        _inventory() if inventory is None else inventory, _caps() if caps is None else caps)


def _cell_algebra(mesh):
    cells = list(product(*(range(n) for n in mesh)))
    lookup = {cell:i for i,cell in enumerate(cells)}
    def translated(cell, shift):
        return lookup[tuple((cells[int(cell)][a]+cells[int(shift)][a]) % mesh[a] for a in range(3))]
    return cells, translated


def _placed_atom_domain(domain, cell, translated):
    return {(translated(r,cell), int(a)) for r,a in domain.expanded_copy()}


def _columns(atoms, mapping):
    return np.array(sorted((r,mu) for r,a in atoms for mu in np.flatnonzero(mapping==a)),
                    np.uint64).reshape(-1,2)


@pytest.mark.parametrize("mesh", [(1,1,1),(2,1,1),(3,1,1),(2,2,2)])
@pytest.mark.parametrize("full", [False,True])
def test_every_canonical_row_matches_independent_translated_set_union(mesh, full):
    case = _case(mesh, full=full, grouped=True)
    cells, translated = _cell_algebra(mesh)
    mapping = case["bundle"]["mapping"]
    nk = len(cells)
    for row in range(case["topology"].row_count):
        pair = case["topology"].row(row)
        result = _run(case,row)
        expected = (_placed_atom_domain(case["domains"][pair.home_orbital],0,translated)
                    | _placed_atom_domain(case["domains"][pair.partner_orbital],
                                           pair.translation_linear_index,translated))
        np.testing.assert_array_equal(result.atoms_copy(), np.array(sorted(expected),np.uint64).reshape(-1,2))
        np.testing.assert_array_equal(result.columns_copy(), _columns(expected,mapping))
        assert result.row_index == row
        assert result.topology_identity_sha256 == case["topology"].topology_identity_sha256
        assert result.home_domain_identity_sha256 == case["domains"][pair.home_orbital].occupied_pao_domain_identity_sha256
        assert result.partner_domain_identity_sha256 == case["domains"][pair.partner_orbital].occupied_pao_domain_identity_sha256
        # Enumerate the unordered placed-pair orbit independently. No shortcut
        # 2-delta_home weighting, including same-i self-inverse translations.
        orbit = {tuple(sorted((translated(0,s)*2+pair.home_orbital,
                               translated(pair.translation_linear_index,s)*2+pair.partner_orbital)))
                 for s in range(nk)}
        diagonal = pair.home_orbital==pair.partner_orbital and pair.translation_linear_index==0
        assert result.energy_weight == len(orbit)*(1 if diagonal else 2)//nk
        assert result.diagnostics.actual_peak_owned_numerical_bytes <= result.memory.peak_owned_numerical_bytes
        assert result.diagnostics.retained_numerical_bytes == 16*(len(expected)+len(result.columns_copy()))
        assert not result.extended_ccsd_domain and not result.hf_basis_source_authenticated


@pytest.mark.parametrize("mesh", [(2,1,1),(3,1,1),(2,2,2)])
def test_every_ordered_placed_pair_and_reversal_reuses_existing_resolver_translation(mesh):
    case = _case(mesh, grouped=True)
    cells, translated = _cell_algebra(mesh)
    topology = case["topology"]
    cache = {r:_run(case,r) for r in range(topology.row_count)}
    for first_cell, second_cell, i, j in product(range(len(cells)),range(len(cells)),range(2),range(2)):
        route = core._resolve_periodic_correlation_placed_pair(topology,i,first_cell,j,second_cell)
        union = cache[route.row_index]
        actual = {union.translated_column(col,route.common_translation_cell)
                  for col in range(union.diagnostics.ao_column_count)}
        expected_atoms = (_placed_atom_domain(case["domains"][i],first_cell,translated)
                          | _placed_atom_domain(case["domains"][j],second_cell,translated))
        expected = {tuple(x) for x in _columns(expected_atoms,case["bundle"]["mapping"])}
        assert actual == expected
        reverse = core._resolve_periodic_correlation_placed_pair(topology,j,second_cell,i,first_cell)
        assert reverse.row_index == route.row_index
        reverse_actual = {union.translated_column(col,reverse.common_translation_cell)
                          for col in range(union.diagnostics.ao_column_count)}
        assert reverse_actual == actual


def test_self_inverse_same_orbital_requires_identical_endpoint_domains():
    case = _case((2,1,1),nactive=1)
    topology = case["topology"]
    row = next(r for r in range(topology.row_count) if topology.row(r).translation_linear_index==1)
    native = _run(case,row)
    route = core._resolve_periodic_correlation_placed_pair(topology,0,0,0,1)
    reverse = core._resolve_periodic_correlation_placed_pair(topology,0,1,0,0)
    assert not route.transpose and not reverse.transpose
    assert route.common_translation_cell==0 and reverse.common_translation_cell==1
    assert native.energy_weight==1
    full = _occupied_domain(case["bundle"],options=_domain_options(full_domain=True))
    with pytest.raises(ValueError,match="same-orbital.*identit.*covariance"):
        _run(case,row,partner=full)
    # Even identical support but a different scientific-policy receipt fails:
    # one occupied orbital cannot silently have two endpoint-specific domains.
    changed = _occupied_domain(case["bundle"],options=_domain_options(
        mulliken_population_cutoff=0.02,pao_tail_cutoff=0.20001))
    with pytest.raises(ValueError,match="same-orbital"):
        _run(case,row,partner=changed)


def test_different_orbitals_may_use_different_cuts_and_keep_both_receipts():
    case = _case((3,1,1))
    row = next(r for r in range(case["topology"].row_count)
               if case["topology"].row(r).home_orbital!=case["topology"].row(r).partner_orbital)
    result = _run(case,row)
    assert case["domains"][0].options.pao_tail_cutoff != case["domains"][1].options.pao_tail_cutoff
    assert result.home_domain_identity_sha256 != result.partner_domain_identity_sha256
    assert result.memory.distinct_domain_owners==2


def test_address_based_inventory_deduplicates_one_owner_but_not_equal_content_allocations():
    case = _case((2,1,1),nactive=1)
    first = case["domains"][0]
    equal = _occupied_domain(case["bundle"],options=first.options)
    assert equal.occupied_pao_domain_identity_sha256==first.occupied_pao_domain_identity_sha256
    same_plan = _plan(case)
    separate_plan = _plan(case,partner=equal)
    assert same_plan.distinct_domain_owners==1 and separate_plan.distinct_domain_owners==2
    assert separate_plan.borrowed_domain_numerical_bytes==2*same_plan.borrowed_domain_numerical_bytes
    assert same_plan.borrowed_domain_numerical_bytes==first.diagnostics.retained_numerical_bytes
    a,b = _run(case),_run(case,partner=equal)
    np.testing.assert_array_equal(a.columns_copy(),b.columns_copy())
    assert a.pair_pao_domain_identity_sha256==b.pair_pao_domain_identity_sha256
    assert separate_plan.worker_bytes-same_plan.worker_bytes == first.diagnostics.retained_numerical_bytes + (
        separate_plan.borrowed_owner_control_bytes-same_plan.borrowed_owner_control_bytes)


def test_empty_domain_union_is_valid_without_a_forced_ao_and_full_domains_collapse_duplicates():
    case = _case(nactive=1)
    empty = _occupied_domain(case["bundle"],options=_domain_options(mulliken_population_cutoff=10.0))
    result = _run(case,home=empty,partner=empty)
    assert result.atoms_copy().shape==result.columns_copy().shape==(0,2)
    assert result.diagnostics.retained_numerical_bytes==0
    assert result.diagnostics.actual_peak_owned_numerical_bytes==result.memory.mark_bytes
    full_case = _case((3,1,1),full=True)
    for r in range(full_case["topology"].row_count):
        union = _run(full_case,r)
        assert union.diagnostics.union_atom_count==union.memory.atom_cell_count
        assert union.diagnostics.ao_column_count==union.memory.n_cells*union.memory.n_basis


def test_sorted_ao_rows_are_consumable_by_existing_pao_geometry_without_union_reinterpretation():
    case = _case((2,1,1),full=True,grouped=True)
    union = _run(case)
    geometry = _make_geometry(case["bundle"]["reference"],union.columns_copy())
    assert geometry.domain_dimension==union.diagnostics.ao_column_count
    for i in range(geometry.domain_dimension):
        assert geometry.column(i)==union.column(i)


@pytest.mark.parametrize("cap_field,plan_field", [
    ("maximum_owned_numerical_bytes","peak_owned_numerical_bytes"),
    ("maximum_worker_bytes","worker_bytes"),("maximum_work_units","planned_work_units")])
def test_cap_boundary_minus_one_fails_before_bad_mapping_scan(cap_field,plan_field):
    case = _case((2,1,1))
    p = _plan(case)
    _run(case,caps=_caps(**{cap_field:getattr(p,plan_field)}))
    bad = np.full_like(case["bundle"]["mapping"],2**64-1)
    with pytest.raises((ValueError,RuntimeError),match="cap|memory|work"):
        _run(case,caps=_caps(**{cap_field:getattr(p,plan_field)-1}),mapping=bad)


def test_actual_union_caps_and_total_control_gate():
    case = _case((2,1,1),full=True)
    expected = _run(case)
    with pytest.raises((ValueError,RuntimeError),match="actual union"):
        _run(case,caps=_caps(maximum_union_atom_cells=expected.diagnostics.union_atom_count-1))
    with pytest.raises((ValueError,RuntimeError),match="actual union"):
        _run(case,caps=_caps(maximum_ao_columns=expected.diagnostics.ao_column_count-1))
    inventory = _inventory(other_live_control_bytes=53,other_live_numerical_bytes=79)
    p = _plan(case,inventory=inventory)
    controls = p.fixed_control_storage_bytes+p.borrowed_owner_control_bytes+53
    _run(case,inventory=inventory,caps=_caps(maximum_control_storage_bytes=controls))
    with pytest.raises((ValueError,RuntimeError),match="control"):
        _run(case,inventory=inventory,caps=_caps(maximum_control_storage_bytes=controls-1))
    assert p.worker_bytes-_plan(case).worker_bytes==53+79


def test_wrong_mapping_topology_endpoint_label_and_distinct_state_owners_fail_closed():
    case = _case()
    with pytest.raises(ValueError,match="mapping content"):
        _run(case,mapping=case["bundle"]["mapping"][::-1].copy())
    with pytest.raises(ValueError,match="labels|localization"):
        _run(case,home=case["domains"][1])
    other = _case()
    with pytest.raises(ValueError,match="state|allocation"):
        _run(case,partner=other["domains"][0])
    wrong_topology = dict(case,topology=other["topology"])
    with pytest.raises(ValueError,match="topology"):
        _run(wrong_topology)
    with pytest.raises(IndexError):
        _run(case,row=case["topology"].row_count)


@pytest.mark.parametrize("mapping", [np.array([0,1,2],np.int64),np.array([[0,1,2]],np.uint64),
                                     np.array([0],np.uint64),np.arange(6,dtype=np.uint64)[::2]])
def test_mapping_view_dtype_shape_and_stride_are_not_silently_converted(mapping):
    with pytest.raises((ValueError,TypeError),match="mapping|uint64|nao"):
        _run(_case(),mapping=mapping)


def test_compact_result_is_detached_and_row_seals_distinguish_equal_supports():
    case = _case((2,1,1),full=True)
    a,b = _run(case,0),_run(case,1)
    assert a.payload_identity_sha256==b.payload_identity_sha256
    assert a.row_identity_sha256!=b.row_identity_sha256
    assert a.pair_pao_domain_identity_sha256!=b.pair_pao_domain_identity_sha256
    columns = a.columns_copy()
    del case
    gc.collect()
    copy = a.columns_copy(); copy[:]=99
    np.testing.assert_array_equal(a.columns_copy(),columns)
    with pytest.raises(IndexError):
        a.column(len(columns))
    with pytest.raises(IndexError):
        a.atom(a.diagnostics.union_atom_count)
    with pytest.raises(IndexError):
        a.translated_column(0,a.memory.n_cells)
