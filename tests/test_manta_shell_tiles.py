"""Tile-only admission for independent integral schedules and near fields."""
import gc
import itertools

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(angular=(0, 1, 2), primitives=2):
    positions = [[.1*i, -.2*(i % 2), .3*(i % 3)] for i in range(len(angular))]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        s = core.ShellInfo(); s.atom_index = i; s.l = l; s.pure = bool(i % 2)
        s.origin = positions[i]; s.exponents = [.7, 1.6][:primitives]; s.coefficients = [.8, -.2][:primitives]
        shells.append(s)
    return vq.BasisSet(mol, shells, "manta-shell-tile-fixture", True)


@pytest.mark.parametrize("backend", ["manta", "libint"])
def test_all_tiles_keep_exact_source_and_values_without_ordered_work_admission(backend):
    basis = _basis()
    cls = manta.DirectJKBuilder if backend == "manta" else manta.ReferenceJKBuilder
    ordered = cls(basis); reference = ordered.session()
    budget = manta.ResourceBudget(); budget.shell_calls = 1
    session = manta.ShellTileSession(basis, backend, budget=budget)
    assert session.plan.schedule == "shell-tiles" and ordered.plan.schedule == "ordered-jk"
    assert session.plan.target_bytes == 0 and session.plan.shell_calls == 1
    assert session.source_identity == reference.evaluate([0, 0, 0, 0])["source_identity"]
    assert session.plan.modeled_peak_bytes < ordered.plan.modeled_peak_bytes
    assert session.calls == 0
    for indices in itertools.product(range(basis.nshells), repeat=4):
        actual = session.evaluate(indices); expected = reference.evaluate(indices)
        np.testing.assert_array_equal(actual["values"], expected["values"])
        assert actual["source_identity"] == expected["source_identity"]
        assert actual["primitive_quartets_requested"] == 16
        assert actual["primitive_evaluations_measured"] == (backend == "manta")
        assert actual["primitive_quartets_evaluated"] == (16 if backend == "manta" else 0)
        assert actual["shell_calls"] == 1 and actual["kernel"] == "coulomb" and actual["omega"] == 0.
    assert session.calls == basis.nshells**4
    with pytest.raises(ValueError, match="budget"):
        cls(basis, budget=budget)


@pytest.mark.parametrize("kernel,omega", [("coulomb", 0.), ("erf", 0.), ("erf", .3), ("erfc", .4)])
def test_md_range_tiles_and_snapshot_ownership(kernel, omega):
    basis = _basis(); request = manta.KernelRequest(); request.kernel = kernel; request.omega = omega
    session = manta.ShellTileSession(basis, request=request)
    reference = manta.DirectJKBuilder(basis, request=request).session()
    indices = [2, 1, 1, 0]
    actual = session.evaluate(indices); before = actual["values"].copy()
    np.testing.assert_array_equal(actual["values"], reference.evaluate(indices)["values"])
    request.kernel = "erf"; request.omega = 10.
    again = session.evaluate(indices)
    assert again["kernel"] == kernel and again["omega"] == omega
    assert actual["source_identity"] == again["source_identity"]
    actual["values"][:] = -900.
    np.testing.assert_array_equal(session.evaluate(indices)["values"], before)
    del session, basis; gc.collect()
    np.testing.assert_array_equal(again["values"], before)


@pytest.mark.parametrize("backend", ["manta", "libint"])
@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"), ("primitive_quartets", "primitive_quartets_upper_bound"), ("scalar_terms", "scalar_terms")])
def test_tile_only_resource_boundaries(backend, field, plan_field):
    basis = _basis(); plan = manta.plan_shell_tiles(basis, backend)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    session = manta.ShellTileSession(basis, backend, budget=budget)
    result = session.evaluate([2, 2, 2, 2])
    assert result["modeled_peak_bytes"] == plan.modeled_peak_bytes
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.ShellTileSession(basis, backend, budget=budget)


def test_md_tile_work_budget_and_invalid_index_do_not_execute():
    basis = _basis(); plan = manta.plan_shell_tiles(basis)
    budget = manta.ResourceBudget(); budget.kernel_terms = plan.kernel_terms
    session = manta.ShellTileSession(basis, budget=budget)
    session.evaluate([2, 2, 2, 2]); before = session.calls
    with pytest.raises(IndexError, match="index"):
        session.evaluate([0, 0, 3, 0])
    assert session.calls == before
    budget.kernel_terms -= 1
    with pytest.raises(ValueError, match="budget"):
        manta.ShellTileSession(basis, budget=budget)


@pytest.mark.parametrize("backend", ["manta", "libint"])
def test_large_inventory_does_not_imply_quartic_schedule_or_ao_targets(backend):
    basis = _basis(angular=(0,)*300, primitives=1)
    budget = manta.ResourceBudget(); budget.shell_calls = budget.primitive_quartets = budget.scalar_terms = 1
    plan = manta.plan_shell_tiles(basis, backend, budget=budget)
    assert plan.n_basis == plan.n_shells == 300
    assert plan.target_bytes == 0 and plan.tile_elements == 1
    assert plan.shell_calls == plan.primitive_quartets_upper_bound == plan.scalar_terms == 1
    assert plan.modeled_peak_bytes < 10*1024**2
    tile = manta.ShellTileSession(basis, backend, budget=budget).evaluate([299, 299, 0, 0])
    assert tile["values"].shape == (1, 1, 1, 1) and tile["values"].item() > 0


def test_unsupported_source_refuses_without_fallback():
    basis = _basis()
    with pytest.raises(ValueError, match="backend"):
        manta.ShellTileSession(basis, "automatic")
    request = manta.KernelRequest(); request.kernel = "erf"; request.omega = .4
    # Range kernels are admitted by both backends; libint's own erf evaluator
    # is the matched witness for the MD tile.
    witness = manta.ShellTileSession(basis, "libint", request).evaluate([0, 1, 1, 0])["values"]
    actual = manta.ShellTileSession(basis, "manta", request).evaluate([0, 1, 1, 0])["values"]
    np.testing.assert_allclose(actual, witness, atol=1e-11, rtol=5e-12)
    request.derivative = 1
    with pytest.raises(ValueError, match="Reference"):
        manta.ShellTileSession(basis, "libint", request)
    request.derivative = 0; request.centers = 3
    with pytest.raises(ValueError):
        manta.ShellTileSession(basis, "manta", request)
