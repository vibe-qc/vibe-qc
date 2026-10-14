"""General and partially shared contraction channels without primitive tensors."""
from itertools import product

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(kind="general", angular=1, mixed_pure=False):
    positions = [[.2+.3*i, -.3, .4] for i in range(3)] if kind == "separate-centers" else [[.2, -.3, .4]]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    shells = []
    for i in range(3):
        shell = core.ShellInfo()
        shell.atom_index, shell.l, shell.pure = 0, angular, not (mixed_pure and i == 1)
        shell.origin = [.2, -.3, .4]
        shell.exponents = [.6, 1.3]
        if kind == "partial":
            shell.exponents = [[.6, 1.3], [1.3, 2.1], [.6]][i]
        elif kind == "disjoint":
            shell.exponents = [.6+.2*i, 1.3+.2*i]
        elif kind == "separate-centers":
            shell.atom_index = i
            shell.origin = [.2+.3*i, -.3, .4]
        elif kind == "separate-angular":
            shell.l = i
        elif kind == "repeated":
            shell.exponents = [.6, .6, 1.3]
        shell.coefficients = [[.8, -.2], [.3, .9], [-.4, .6]][i][:len(shell.exponents)]
        if kind == "repeated":
            shell.coefficients = [[.8, -.2, .1], [.3, -.3, .9], [-.4, .6, .2]][i]
        shells.append(shell)
    return vq.BasisSet(mol, shells, "manta-contraction-channels", False)


def _expected_evaluations(basis, batch_size):
    shells = basis.shells()
    quartets = list(product(range(len(shells)), repeat=4))
    total = 0
    for start in range(0, len(quartets), batch_size):
        groups = {}
        for indices in quartets[start:start+batch_size]:
            selected = [shells[i] for i in indices]
            key = tuple((s.l, tuple(s.origin)) for s in selected)
            primitive_set = groups.setdefault(key, set())
            exponents = []
            for shell in selected:
                coefficients = {}
                for exponent, coefficient in zip(shell.exponents, shell.coefficients, strict=True):
                    coefficients[exponent] = coefficients.get(exponent, 0.)+coefficient
                exponents.append([e for e, c in coefficients.items() if c != 0.])
            primitive_set.update(product(*exponents))
        total += sum(map(len, groups.values()))
    return total


@pytest.mark.parametrize("kind", ["general", "partial", "disjoint", "separate-centers", "separate-angular", "repeated"])
@pytest.mark.parametrize("batch_size", [2, 7, 16, 64])
def test_shared_channels_match_unshared_integrals_and_executed_count_oracle(kind, batch_size):
    basis = _basis(kind)
    rng = np.random.default_rng(9375)
    d = rng.normal(size=(basis.nbasis, basis.nbasis)); d += d.T
    unshared = manta.DirectJKBuilder(basis)
    builder = manta.DirectJKBuilder(basis, channel_batch=batch_size)
    actual, reference = builder.build(d), unshared.build(d)
    eri = np.asarray(vq.compute_eri(basis))
    for value, expected, independent in [(actual.J, reference.J, np.einsum("abcd,cd->ab", eri, d)),
                                         (actual.K, reference.K, np.einsum("abcd,bd->ac", eri, d))]:
        np.testing.assert_allclose(value, expected, atol=4e-12, rtol=3e-12)
        np.testing.assert_allclose(value, independent, atol=4e-12, rtol=3e-12)
    receipt = actual.receipt
    assert receipt.primitive_evaluations_measured
    assert receipt.primitive_quartets_evaluated == _expected_evaluations(basis, batch_size)
    assert receipt.primitive_quartets_requested == reference.receipt.primitive_quartets_requested
    assert reference.receipt.primitive_quartets_evaluated == reference.receipt.primitive_quartets_requested
    assert receipt.primitive_quartets_evaluated <= receipt.primitive_quartets_requested
    assert receipt.shell_calls == reference.receipt.shell_calls == basis.nshells**4
    assert receipt.integral_elements_consumed == reference.receipt.integral_elements_consumed == basis.nbasis**4
    assert receipt.source_identity == reference.receipt.source_identity
    assert receipt.contraction_strategy == "shared-channels"
    assert reference.receipt.contraction_strategy == "segmented"
    if kind in ["general", "partial", "repeated"]:
        assert receipt.primitive_quartets_evaluated < receipt.primitive_quartets_requested
    else:
        assert receipt.primitive_quartets_evaluated == receipt.primitive_quartets_requested


@pytest.mark.parametrize("kernel", ["coulomb", "erf", "erfc"])
def test_mixed_representations_and_multiple_densities_share_generation(kernel):
    basis = _basis("partial", angular=2, mixed_pure=True)
    request = manta.KernelRequest(); request.kernel = kernel
    request.omega = .6 if kernel != "coulomb" else 0.
    builder = manta.DirectJKBuilder(basis, request, channel_batch=16)
    reference = manta.DirectJKBuilder(basis, request)
    rng = np.random.default_rng(9914)
    d = rng.normal(size=(basis.nbasis, basis.nbasis)); d += d.T
    densities = [d, -2*d, np.eye(basis.nbasis)]
    actual = builder.build_batch(densities)
    for index, density in enumerate(densities):
        expected = reference.build(density)
        np.testing.assert_allclose(actual.J[index], expected.J, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(actual.K[index], expected.K, atol=2e-11, rtol=2e-11)
    single = builder.build(d)
    assert actual.receipt.primitive_quartets_evaluated == single.receipt.primitive_quartets_evaluated
    assert actual.receipt.density_terms_accumulated == 3*single.receipt.density_terms_accumulated
    for value, expected in zip(builder.build_K_pair(*densities[:2]), actual.K[:2], strict=True):
        np.testing.assert_array_equal(value, expected)
    if kernel == "erf":
        full = manta.DirectJKBuilder(basis, channel_batch=16)
        np.testing.assert_allclose(full.build_K_erf(d, .6), single.K, atol=2e-12)


def test_exact_duplicate_coefficient_cancellation_skips_all_primitives():
    mol = vq.Molecule([vq.Atom(2, [0., 0., 0.])], 0, 1)
    shell = core.ShellInfo()
    shell.atom_index, shell.l, shell.pure, shell.origin = 0, 0, True, [0., 0., 0.]
    shell.exponents, shell.coefficients = [.7, .7], [1., -1.]
    basis = vq.BasisSet(mol, [shell], "exact-cancellation", True)
    result = manta.DirectJKBuilder(basis, channel_batch=2).build(np.ones((1, 1)))
    assert result.receipt.primitive_quartets_requested == 16
    assert result.receipt.primitive_quartets_evaluated == 0
    np.testing.assert_array_equal(result.J, [[0.]])
    np.testing.assert_array_equal(result.K, [[0.]])


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("kernel_terms", "kernel_terms")])
def test_channel_batch_resource_boundaries(field, plan_field):
    basis = _basis()
    plan = manta.md_plan(basis, channel_batch=16)
    old = manta.md_plan(basis)
    assert plan.modeled_peak_bytes > old.modeled_peak_bytes
    assert plan.kernel_terms > old.kernel_terms
    assert plan.primitive_quartets_upper_bound == old.primitive_quartets_upper_bound
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    manta.DirectJKBuilder(basis, budget=budget, channel_batch=16).build(np.eye(basis.nbasis))
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.DirectJKBuilder(basis, budget=budget, channel_batch=16)


@pytest.mark.parametrize("count", [0, 65, 2**63])
def test_unadmitted_channel_batch_extents(count):
    with pytest.raises(ValueError, match="channel batch"):
        manta.DirectJKBuilder(_basis(), channel_batch=count)


def test_reference_receipt_does_not_claim_opaque_primitive_evaluations():
    basis = _basis()
    result = manta.ReferenceJKBuilder(basis).build(np.eye(basis.nbasis))
    assert not result.receipt.primitive_evaluations_measured
    assert result.receipt.primitive_quartets_evaluated == 0
    assert result.receipt.primitive_quartets_requested > 0


def test_channel_workspace_keeps_single_tile_sessions_owned():
    basis = _basis()
    builder = manta.DirectJKBuilder(basis, channel_batch=16)
    session = builder.session()
    first = session.evaluate((0, 0, 0, 0))["values"]
    before = first.copy()
    builder.build(np.eye(basis.nbasis))
    session.evaluate((1, 2, 0, 1))
    del builder, session, basis
    np.testing.assert_array_equal(first, before)


def test_shared_l6_channels_keep_full_pure_transform_extent():
    mol = vq.Molecule([vq.Atom(2, [.2, -.3, .4])], 0, 1)
    shells = _basis(angular=6).shells()[:2]
    for shell in shells:
        shell.exponents = [.6]
        shell.coefficients = [1.]
    shells[1].coefficients = [-.4]
    basis = vq.BasisSet(mol, shells, "manta-l6-channels", True)
    single_basis = vq.BasisSet(mol, shells[:1], "manta-l6-single", True)
    # Independent linear-channel reconstruction from one unshared shell tile.
    tile = manta.DirectJKBuilder(single_basis).evaluate((0, 0, 0, 0))["values"]
    scale = np.repeat([1., -.4], 13)
    tensor = np.tile(tile, (2, 2, 2, 2))*np.einsum("a,b,c,d->abcd", scale, scale, scale, scale)
    d = np.eye(basis.nbasis)
    result = manta.DirectJKBuilder(basis, channel_batch=2).build(d)
    np.testing.assert_allclose(result.J, np.einsum("abcd,cd->ab", tensor, d), atol=3e-11, rtol=2e-12)
    np.testing.assert_allclose(result.K, np.einsum("abcd,bd->ac", tensor, d), atol=3e-11, rtol=2e-12)
    assert result.receipt.primitive_quartets_evaluated == 8
    assert result.receipt.primitive_quartets_requested == 16
