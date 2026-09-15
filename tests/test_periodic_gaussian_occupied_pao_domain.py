"""Tiny actual-HF atom-domain selection against direct finite-torus algebra."""

from __future__ import annotations

import gc
from itertools import product

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_gaussian_selected_local_ccsd_t import (
    _bundle, _he2_bundle, _localization_options, _source_caps,
)
from tests.test_periodic_gaussian_localization import _caps as _local_caps
from tests.test_periodic_gaussian_frozen_core_correlation import _frozen_bundle


def _localized(b, *, options=None):
    if options is None:
        options = _localization_options(b)
        if b.hf.state.n_correlated_occupied > 1:
            # Same explicit diagnostic He2 fixture as the existing full-space
            # tests; the strict localization nonconvergence gate stays tested.
            options.optimizer.riemannian_gradient_tolerance = 2e-6
    b.localization = core._localize_periodic_gaussian_occupied(
        b.reference, b.ao, b.minimal, b.system, 0, options, _source_caps(b), _local_caps())
    return b


def _controls(b, *, full=False, cut=.2, tail=.1):
    options = core._PeriodicCorrelationOccupiedPAODomainOptions()
    options.full_domain = full
    options.mulliken_population_cutoff, options.pao_tail_cutoff = cut, tail
    options.normalization_tolerance = 3e-10
    options.maximum_population_imaginary_magnitude = 3e-10
    options.maximum_pao_imaginary_magnitude = 3e-10
    options.maximum_negative_absolute_population = 1.0
    options.maximum_seed_omitted_absolute_population = 2.0
    options.maximum_expanded_omitted_absolute_population = 2.0
    inventory = core._PeriodicCorrelationOccupiedPAODomainInventory()
    inventory.backend_allowance_bytes = 65536
    child = core._PeriodicCorrelationOccupiedPAODomainCaps()
    child.maximum_atom_cells = child.maximum_seed_atom_cells = child.maximum_expanded_atom_cells = (
        b.context.n_kpoints * len(b.system.unit_cell))
    child.maximum_owned_numerical_bytes = 1 << 20
    child.maximum_control_storage_bytes = 1 << 20
    child.maximum_worker_bytes = 1 << 24
    child.maximum_work_units = 10**12
    caps = core._PeriodicGaussianOccupiedPAODomainCaps()
    caps.selector = child
    caps.maximum_owned_numerical_bytes = 1 << 20
    caps.maximum_worker_bytes, caps.maximum_node_bytes = 1 << 24, 1 << 26
    caps.maximum_work_units = 10**13
    return options, inventory, caps


def _plan(b, i=0, controls=None):
    return core._plan_periodic_gaussian_occupied_pao_domain(
        b.hf, b.reference, b.localization, i, *(_controls(b) if controls is None else controls))


def _run(b, i=0, controls=None, **changes):
    arguments = dict(hf=b.hf, reference=b.reference, ao_basis=b.ao, auxiliary_basis=b.auxiliary,
        minimal_basis=b.minimal, system=b.system, localization=b.localization, home_active_occupied_index=i)
    arguments.update(changes)
    options, inventory, caps = _controls(b) if controls is None else controls
    return core._select_periodic_gaussian_occupied_pao_domain(**arguments,
        options=options, inventory=inventory, caps=caps)


def _mapping(b):
    return np.array([s.atom_index for s in b.ao.shells() for _ in range(2*s.l+1)], np.uint64)


def _oracle(b, i, options):
    state = b.hf.state
    mesh = np.asarray(state.mesh)
    cells = np.array(list(product(*(range(n) for n in mesh))))
    nk, nao, atoms = len(cells), b.ao.nbasis, len(b.system.unit_cell)
    phase = np.exp(2j*np.pi*(cells @ (cells/mesh).T))/nk
    coefficients, eta, projectors = [], [], []
    gauges = b.localization.optimizer.gauges_copy()
    for k in range(nk):
        c, s = state.coefficients(k), state.overlap(k)
        active = np.asarray(state.correlated_occupied_mask(k), bool)
        virtual = np.asarray(state.virtual_mask(k), bool)
        occupied = (c[:, active] @ gauges[k])[:, i]
        coefficients.append(occupied)
        eta.append(s @ occupied)
        projectors.append(c[:, virtual] @ c[:, virtual].conj().T @ s)
    beta, covariant = phase @ coefficients, phase @ eta
    labels = _mapping(b)
    gross = beta.conj()*covariant
    populations = np.array([[gross[R, labels == atom].sum().real for atom in range(atoms)] for R in range(nk)])
    seeds = [(R, A) for R in range(nk) for A in range(atoms)
             if options.full_domain or populations[R, A] > options.mulliken_population_cutoff]
    expanded = set(seeds)
    if not options.full_domain:
        q = np.asarray(projectors)
        for RA, A in seeds:
            for RB, B in product(range(nk), range(atoms)):
                factors = np.exp(2j*np.pi*((cells/mesh) @ (cells[RA]-cells[RB])))/nk
                placed = np.einsum("k,kmn->mn", factors, q)
                score = abs(placed[np.ix_(labels == A, labels == B)]).sum()
                if score > options.pao_tail_cutoff:
                    expanded.add((RB, B))
    return populations, np.asarray(seeds, np.uint64).reshape(-1, 2), np.asarray(sorted(expanded), np.uint64).reshape(-1, 2)


@pytest.mark.parametrize("kind,nk", [("he", 1), ("he", 2), ("he", 3), ("he2", 1), ("he2", 2)])
@pytest.mark.parametrize("full", [False, True])
def test_actual_native_atom_mapping_and_selection_match_direct_finite_torus(kind, nk, full):
    b = _localized((_bundle if kind == "he" else _he2_bundle)((nk, 1, 1)))
    controls = _controls(b, full=full)
    state_identity = b.hf.state.numerical_payload_sha256
    for i in range(b.hf.state.n_correlated_occupied):
        result = _run(b, i, controls)
        expected, seeds, expanded = _oracle(b, i, controls[0])
        np.testing.assert_allclose(result.domain.populations_copy(), expected, atol=3e-12)
        np.testing.assert_array_equal(result.domain.seeds_copy(), seeds)
        np.testing.assert_array_equal(result.domain.expanded_copy(), expanded)
        assert result.domain.diagnostics.population_sum == pytest.approx(1.0, abs=3e-12)
        assert result.matched_finite_gaussian_hf_recipe
        assert not result.production_dlpno and not result.infinite_source_accuracy_certified
        assert not result.domain.hf_basis_source_authenticated  # generic child alone is not an HF receipt
        assert result.context is b.hf.context
        assert result.hf_reference_source_identity_sha256 == b.hf.reference_source_identity_sha256
        assert result.localization_identity_sha256 == b.localization.localization_identity_sha256
        assert result.domain.home_occupied_index == i
    assert b.hf.state.numerical_payload_sha256 == state_identity


@pytest.mark.parametrize("nk", [1, 2])
def test_frozen_core_never_enters_population_or_retained_virtual_projector(nk):
    b = _localized(_frozen_bundle((nk, 1, 1)))
    result = _run(b)
    expected = _oracle(b, 0, _controls(b)[0])
    np.testing.assert_allclose(result.domain.populations_copy(), expected[0], atol=3e-12)
    np.testing.assert_array_equal(result.domain.expanded_copy(), expected[2])
    state = b.hf.state
    assert state.n_frozen_core == state.n_correlated_occupied == 1
    c, s = state.coefficients(0), state.overlap(0)
    active = c[:, np.asarray(state.correlated_occupied_mask(0), bool)]
    virtual = c[:, np.asarray(state.virtual_mask(0), bool)]
    assert np.linalg.norm((np.eye(b.ao.nbasis)-active @ active.conj().T @ s)
                          - virtual @ virtual.conj().T @ s) > .1


@pytest.mark.parametrize("field,reported", [
    ("maximum_owned_numerical_bytes", "peak_owned_numerical_bytes"),
    ("maximum_worker_bytes", "worker_bytes"), ("maximum_node_bytes", "required_node_memory_bytes"),
    ("maximum_work_units", "work_units"),
])
def test_outer_caps_exact_and_minus_one_precede_original_physical_scans(field, reported):
    b = _localized(_bundle((2, 1, 1)))
    controls = _controls(b)
    plan = _plan(b, controls=controls)
    assert plan.peak_owned_numerical_bytes == plan.atom_mapping_bytes + plan.selector.peak_owned_numerical_bytes
    assert plan.atom_mapping_bytes == 8*b.ao.nbasis
    assert plan.worker_bytes == plan.selector.worker_bytes
    assert plan.required_node_memory_bytes == plan.selector.required_node_memory_bytes
    assert plan.work_units == (plan.physical_input_validation_work_units + plan.mapping_work_units
                               + plan.selector.planned_work_units)
    required = getattr(plan, reported)
    setattr(controls[2], field, required)
    _run(b, controls=controls)
    lattice = b.system.lattice.copy()
    lattice[0, 0] = np.nan
    b.system.lattice = lattice
    setattr(controls[2], field, required-1)
    with pytest.raises((ValueError, RuntimeError), match="cap"):
        _run(b, controls=controls)


@pytest.mark.parametrize("kind", ["hf", "reference", "localization", "minimal", "auxiliary", "cell"])
def test_foreign_owner_or_changed_original_inputs_are_not_accepted(kind):
    b = _localized(_bundle())
    other = _localized(_bundle())
    changes = {}
    if kind in ("hf", "reference", "localization"):
        changes[kind] = getattr(other, kind)
    elif kind == "minimal":
        changes["minimal_basis"] = b.ao  # wrong native original minimal census/content
    elif kind == "auxiliary":
        changes["auxiliary_basis"] = b.minimal
    else:
        lattice = b.system.lattice.copy()
        lattice[0, 0] += 1e-14
        b.system.lattice = lattice
    with pytest.raises((ValueError, RuntimeError), match="owner|lineage|basis|physical|cell|lattice|dimensions|census"):
        _run(b, **changes)


def test_distinct_ao_image_policy_is_rejected_even_when_same_finite_overlaps_result():
    b = _bundle()
    options = _localization_options(b)
    options.source.image_cutoff_bohr += .001
    _localized(b, options=options)
    with pytest.raises(ValueError, match="AO-image policy"):
        _run(b)


def test_thresholded_empty_domain_and_owned_output_lifetime():
    b = _localized(_bundle((2, 1, 1)))
    result = _run(b, controls=_controls(b, cut=2.0))
    domain = result.domain
    expected = domain.populations_copy()
    assert domain.seeds_copy().shape == domain.expanded_copy().shape == (0, 2)
    assert domain.diagnostics.expanded_omitted_absolute_population >= 1.0-3e-12
    domain.populations_copy()[:] = 42
    del result, b
    gc.collect()
    np.testing.assert_array_equal(domain.populations_copy(), expected)
