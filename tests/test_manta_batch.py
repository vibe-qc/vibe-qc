"""Multiple densities share one ordered integral traversal without shared state."""
import gc

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, compute_eri, manta


def _basis():
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [.3, -.2, 1.4])], 0, 1)
    return BasisSet(mol, "6-31g")


def _builder(provider="manta", kernel="coulomb", budget=None):
    request = manta.KernelRequest(); request.kernel = kernel
    request.omega = .6 if kernel != "coulomb" else 0.
    constructor = manta.DirectJKBuilder if provider == "manta" else manta.ReferenceJKBuilder
    return constructor(_basis(), request, manta.ResourceBudget() if budget is None else budget)


def _densities(n):
    rng = np.random.default_rng(7129)
    a, b = rng.normal(size=(2, n, n))
    return [np.asfortranarray(a@a.T), b+b.T, np.zeros((n, n)), -2*(b+b.T)]


@pytest.mark.parametrize("provider,kernel", [("libint", "coulomb"), ("manta", "coulomb"),
                                           ("manta", "erf"), ("manta", "erfc")])
@pytest.mark.parametrize("make_j,make_k", [(True, True), (True, False), (False, True)])
def test_batch_matches_separate_builds_and_shares_actual_work(provider, kernel, make_j, make_k):
    builder = _builder(provider, kernel)
    densities = _densities(builder.plan.n_basis)
    expected = [builder.build(d) for d in densities]
    result = builder.build_batch(densities, make_j, make_k)
    plan = builder.plan_batch(len(densities), make_j, make_k)
    for name, selected in [("J", make_j), ("K", make_k)]:
        targets = getattr(result, name)
        assert len(targets) == (len(densities) if selected else 0)
        for actual, reference in zip(targets, expected):
            np.testing.assert_array_equal(actual, getattr(reference, name))
    receipt = result.receipt
    single = expected[0].receipt
    assert receipt.shell_calls == single.shell_calls == plan.shell_calls
    assert receipt.primitive_quartets_requested == single.primitive_quartets_requested
    assert receipt.integral_elements_consumed == single.integral_elements_consumed
    assert receipt.density_terms_accumulated == len(densities)*(make_j+make_k)*single.integral_elements_consumed
    assert receipt.density_terms_accumulated <= plan.scalar_terms
    assert receipt.modeled_peak_bytes == plan.modeled_peak_bytes
    assert receipt.source_identity == single.source_identity
    assert receipt.provider == provider and receipt.kernel == kernel
    assert result.density_identities == [r.receipt.density_identity for r in expected]
    assert receipt.algorithm == "ordered-direct-batch"
    if kernel == "coulomb":
        # Independent dense reference is deliberately tiny and test-only.
        eri = np.asarray(compute_eri(_basis()))
        for i, d in enumerate(densities):
            if make_j:
                np.testing.assert_allclose(result.J[i], np.einsum("abcd,cd->ab", eri, d), atol=2e-12)
            if make_k:
                np.testing.assert_allclose(result.K[i], np.einsum("abcd,bd->ac", eri, d), atol=2e-12)


@pytest.mark.parametrize("provider", ["libint", "manta"])
def test_spin_pair_is_shared_batch_and_results_survive_later_calls(provider):
    builder = _builder(provider)
    densities = _densities(builder.plan.n_basis)[:2]
    result = builder.build_batch(densities, False, True)
    pair = builder.build_K_pair(*densities)
    for actual, expected in zip(pair, result.K, strict=True):
        np.testing.assert_array_equal(actual, expected)
    saved, expected = result.K, [x.copy() for x in result.K]
    for d in densities:
        d[:] = 0
    builder.build_batch(densities)
    del result, builder; gc.collect()
    for actual, reference in zip(saved, expected, strict=True):
        np.testing.assert_array_equal(actual, reference)


def test_identity_is_ordered_and_single_density_identity_is_compatible():
    builder = _builder()
    densities = _densities(builder.plan.n_basis)
    first = builder.build_batch(densities)
    second = builder.build_batch(densities[::-1])
    assert first.receipt.density_identity != second.receipt.density_identity
    assert first.receipt.source_identity == second.receipt.source_identity
    single = builder.build_batch(densities[:1])
    original = builder.build(densities[0])
    assert single.receipt.density_identity == original.receipt.density_identity
    assert single.receipt.algorithm == original.receipt.algorithm
    assert single.receipt.modeled_peak_bytes == builder.plan.modeled_peak_bytes


@pytest.mark.parametrize("provider", ["libint", "manta"])
@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"),
                                             ("scalar_terms", "scalar_terms")])
def test_batch_budget_exact_boundary_and_copied_budget(provider, field, plan_field):
    reference = _builder(provider)
    densities = _densities(reference.plan.n_basis)
    plan = reference.plan_batch(len(densities))
    limit = getattr(plan, plan_field)
    budget = manta.ResourceBudget(); setattr(budget, field, limit)
    builder = _builder(provider, budget=budget)
    builder.build_batch(densities)
    setattr(budget, field, limit-1)
    builder.build_batch(densities)  # External mutations cannot change admission.
    reduced = _builder(provider, budget=budget)
    with pytest.raises(ValueError, match="budget"):
        reduced.build_batch(densities)


@pytest.mark.parametrize("provider", ["libint", "manta"])
def test_batch_memory_rejection_precedes_density_access_and_pair_copy(provider):
    reference = _builder(provider)
    budget = manta.ResourceBudget(); budget.memory_bytes = reference.plan.modeled_peak_bytes
    builder = _builder(provider, budget=budget)

    class Unreadable:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            raise AssertionError("density payload was accessed before admission")

    with pytest.raises(ValueError, match="budget"):
        builder.build_batch(Unreadable())
    bad = np.full((builder.plan.n_basis, builder.plan.n_basis), np.nan)
    with pytest.raises(ValueError, match="budget"):
        builder.build_K_pair(bad, bad)


@pytest.mark.parametrize("kind", ["complex", "float32", "strided", "nan", "nonsymmetric", "shape", "list"])
def test_every_batch_member_is_strictly_admitted(kind):
    builder = _builder()
    n = builder.plan.n_basis
    good = np.eye(n)
    bad = {"complex": good.astype(complex), "float32": good.astype(np.float32),
           "strided": np.eye(2*n)[::2, ::2], "nan": good*np.nan,
           "nonsymmetric": np.triu(np.ones((n, n))), "shape": np.eye(n+1),
           "list": good.tolist()}[kind]
    for values in [[bad, good], [good, bad]]:
        with pytest.raises(ValueError):
            builder.build_batch(values)


def test_empty_invalid_target_and_overflow_batches_are_refused():
    builder = _builder()
    with pytest.raises(ValueError, match="nonempty"):
        builder.build_batch([])
    with pytest.raises(ValueError, match="target"):
        builder.build_batch([np.eye(builder.plan.n_basis)], False, False)
    with pytest.raises((ValueError, OverflowError)):
        builder.plan_batch(2**63)


def test_batches_preserve_linearity_and_coulomb_energy_adjoint():
    builder = _builder()
    a, b = _densities(builder.plan.n_basis)[:2]
    result = builder.build_batch([a, b, a+b, a-b])
    for matrices in [result.J, result.K]:
        np.testing.assert_allclose(matrices[2], matrices[0]+matrices[1], atol=2e-12)
        np.testing.assert_allclose(matrices[3], matrices[0]-matrices[1], atol=2e-12)
        assert np.einsum("ij,ji", a, matrices[1]) == pytest.approx(np.einsum("ij,ji", b, matrices[0]), abs=2e-12)
