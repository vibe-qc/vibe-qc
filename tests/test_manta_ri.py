"""Streamed RI contractions and explicit positive metric projectors."""
import gc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(angular=(0, 1), pure=True, shift=(0., 0., 0.)):
    positions = np.array([[.1, -.2, .3], [.4, .2, -.1], [-.2, .5, .7]])+shift
    mol = vq.Molecule([vq.Atom(2, p.tolist()) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        shell = core.ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = i, l, pure, positions[i].tolist()
        shell.exponents, shell.coefficients = [.7+.1*i, 1.6+.2*i], [.8, -.18]
        shells.append(shell)
    return vq.BasisSet(mol, shells, "manta-ri", False)


def _trial(n):
    a = np.random.default_rng(8721).normal(size=(n, n))
    return a+a.T


def _options(block=2, absolute=1e-12, relative=1e-10):
    options = manta.RIOptions()
    options.mode_block, options.absolute_cutoff, options.relative_cutoff = block, absolute, relative
    return options


def _reference(metric, coupling, d, options):
    eigenvalues, vectors = np.linalg.eigh(metric)
    keep = eigenvalues > max(options.absolute_cutoff, options.relative_cutoff*eigenvalues[-1])
    factor = vectors[:, keep]/np.sqrt(eigenvalues[keep])
    fitted = np.einsum("pr,pij->rij", factor, coupling)
    j = np.einsum("rij,rkl,kl->ij", fitted, fitted, d)
    k = np.einsum("rij,jk,rlk->il", fitted, d, fitted)
    rhs = np.einsum("pij,ij->p", coupling, d)
    solution = factor@(factor.T@rhs)
    return j, k, keep.sum(), np.linalg.norm(metric@solution-rhs)


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("block", [1, 2, 8, 64])
def test_complete_jk_against_independent_libint_and_dense_reference(pure, block):
    orbital = _basis(pure=pure)
    auxiliary = _basis((0, 1, 2), pure, (.3, -.2, .5))
    options = _options(block)
    builder = manta.RIJKBuilder(orbital, auxiliary, options)
    d = _trial(orbital.nbasis)
    metric, coupling = vq.compute_2c_eri(auxiliary), vq.compute_3c_eri(orbital, auxiliary)
    j, k, rank, residual = _reference(metric, coupling, d, options)
    result = builder.build(d)
    np.testing.assert_allclose(builder.metric, metric, atol=2e-12)
    np.testing.assert_allclose(result.J, j, atol=5e-12, rtol=2e-11)
    np.testing.assert_allclose(result.K, k, atol=5e-12, rtol=2e-11)
    np.testing.assert_allclose(builder.factor.T@metric@builder.factor, np.eye(rank), atol=2e-12)
    assert builder.metric_info.rank == rank
    assert result.receipt.retained_rank == rank
    assert result.receipt.coupling_passes == 2+(rank+block-1)//block
    assert result.receipt.shell_calls == result.receipt.coupling_passes*auxiliary.nshells*orbital.nshells**2
    assert result.receipt.primitive_products == result.receipt.shell_calls*8
    assert result.receipt.integral_elements_consumed == result.receipt.coupling_passes*auxiliary.nbasis*orbital.nbasis**2
    assert result.receipt.metric_residual_evaluated
    assert result.receipt.metric_residual_norm == pytest.approx(residual, abs=3e-12)
    assert result.receipt.coulomb_energy == pytest.approx(.5*np.einsum("ij,ji", d, j), abs=5e-12)
    assert builder.metric_info.shell_calls == auxiliary.nshells**2
    assert builder.metric_info.primitive_products == auxiliary.nshells**2*4
    assert result.receipt.shell_calls+builder.metric_info.shell_calls <= builder.plan.shell_calls
    assert result.receipt.primitive_products+builder.metric_info.primitive_products <= builder.plan.primitive_products
    np.testing.assert_allclose(builder.build_J(d), j, atol=5e-12)
    np.testing.assert_allclose(builder.build_K(d), k, atol=5e-12)
    np.testing.assert_allclose(builder.build_g_rhf(d, .37), j-.185*k, atol=5e-12)


def _range_integrals(orbital, auxiliary, kernel, omega):
    # Independently form primitive libcint reference arrays. The fixture has
    # one normalized primitive s orbital and p auxiliary at distinct centers.
    gto = pytest.importorskip("pyscf.gto")
    a, p = orbital.shells()[0], auxiliary.shells()[0]
    molecule = gto.M(atom=[("He1", a.origin), ("He2", p.origin)],
                     basis={"He1": [[0, [a.exponents[0], 1.]]],
                            "He2": [[1, [p.exponents[0], 1.]]]}, unit="Bohr", verbose=0)
    with molecule.with_range_coulomb(omega if kernel == "erf" else -omega):
        metric = molecule.intor_by_shell("int2c2e_sph", (1, 1))
        coupling = molecule.intor_by_shell("int3c2e_sph", (0, 0, 1)).transpose(2, 0, 1)
    return metric[np.ix_([1, 2, 0], [1, 2, 0])], coupling[[1, 2, 0]]


@pytest.mark.parametrize("kernel", ["erf", "erfc"])
def test_range_ri_against_independent_libcint(kernel):
    mol = vq.Molecule([vq.Atom(2, [0., 0., 0.]), vq.Atom(2, [.4, -.2, .8])], 0, 1)
    bases = []
    for i, l in enumerate([0, 1]):
        shell = core.ShellInfo()
        shell.atom_index, shell.l, shell.pure = i, l, True
        shell.origin = [[0., 0., 0.], [.4, -.2, .8]][i]
        shell.exponents, shell.coefficients = [.7+.1*i], [1.]
        bases.append(vq.BasisSet(mol, [shell], "manta-ri-range", False))
    orbital, auxiliary = bases
    request = manta.KernelRequest(); request.kernel, request.omega = kernel, .6
    options = _options(1)
    builder = manta.RIJKBuilder(orbital, auxiliary, options, request)
    metric, coupling = _range_integrals(orbital, auxiliary, kernel, .6)
    d = np.array([[-1.4]])
    j, k, _, _ = _reference(metric, coupling, d, options)
    result = builder.build(d)
    np.testing.assert_allclose(builder.metric, metric, atol=2e-12)
    np.testing.assert_allclose(result.J, j, atol=2e-12)
    np.testing.assert_allclose(result.K, k, atol=2e-12)
    assert result.receipt.kernel == kernel and result.receipt.omega == .6


def test_duplicate_auxiliary_channels_have_explicit_rank_and_same_fit():
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in auxiliary.shells()], 0, 1)
    duplicate = vq.BasisSet(mol, auxiliary.shells()+auxiliary.shells(), "duplicate-auxiliary", True)
    reference = manta.RIJKBuilder(orbital, auxiliary)
    builder = manta.RIJKBuilder(orbital, duplicate)
    assert builder.metric_info.rank == reference.metric_info.rank < duplicate.nbasis
    assert builder.source_identity != reference.source_identity
    d = _trial(orbital.nbasis)
    first, second = reference.build(d), builder.build(d)
    np.testing.assert_allclose(first.J, second.J, atol=2e-11)
    np.testing.assert_allclose(first.K, second.K, atol=2e-11)


@pytest.mark.parametrize("absolute,relative", [(0., .2), (1., 0.), (1e10, 0.)])
def test_positive_projector_cutoff_and_discarded_rhs_residual(absolute, relative):
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    options = _options(2, absolute, relative)
    builder = manta.RIJKBuilder(orbital, auxiliary, options)
    d = _trial(orbital.nbasis)
    expected = _reference(vq.compute_2c_eri(auxiliary), vq.compute_3c_eri(orbital, auxiliary), d, options)
    result = builder.build(d)
    np.testing.assert_allclose(result.J, expected[0], atol=3e-12)
    np.testing.assert_allclose(result.K, expected[1], atol=3e-12)
    assert builder.metric_info.rank == expected[2]
    if expected[2]:
        assert result.receipt.metric_residual_norm == pytest.approx(expected[3], abs=3e-12)
    else:
        assert result.receipt.shell_calls == result.receipt.coupling_passes == 0
        assert not result.receipt.metric_residual_evaluated


def test_zero_range_erf_has_zero_rank_and_zero_outputs():
    request = manta.KernelRequest(); request.kernel = "erf"
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    builder = manta.RIJKBuilder(orbital, auxiliary, request=request)
    result = builder.build(_trial(orbital.nbasis))
    assert builder.metric_info.rank == result.receipt.coupling_passes == 0
    np.testing.assert_array_equal(result.J, np.zeros((orbital.nbasis, orbital.nbasis)))
    np.testing.assert_array_equal(result.K, result.J)


def test_linearity_adjoint_energy_derivative_and_owned_state():
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    options = _options(2)
    builder = manta.RIJKBuilder(orbital, auxiliary, options)
    a, b = _trial(orbital.nbasis), np.eye(orbital.nbasis)
    first, second, total = builder.build(a), builder.build(b), builder.build(a+b)
    for actual, expected in zip(builder.build_K_pair(a, b), [first.K, second.K], strict=True):
        np.testing.assert_array_equal(actual, expected)
    paired = builder.build_exchange_pair(a, b)
    np.testing.assert_array_equal(paired.K_alpha, first.K)
    np.testing.assert_array_equal(paired.K_beta, second.K)
    assert paired.receipt.coupling_passes == first.receipt.coupling_passes-2
    assert paired.receipt.shell_calls == paired.receipt.coupling_passes*builder.plan.coupling_shell_calls_per_pass
    assert paired.density_identities == [first.receipt.density_identity, second.receipt.density_identity]
    assert paired.receipt.density_identity != builder.build_exchange_pair(b, a).receipt.density_identity
    assert not paired.receipt.metric_residual_evaluated
    for name in ["J", "K"]:
        left, right = getattr(first, name), getattr(second, name)
        np.testing.assert_allclose(getattr(total, name), left+right, atol=4e-12)
        assert np.einsum("ij,ji", a, right) == pytest.approx(np.einsum("ij,ji", b, left), abs=4e-12)
    h = 1e-5
    fd = (builder.build(a+h*b).receipt.coulomb_energy-builder.build(a-h*b).receipt.coulomb_energy)/(2*h)
    assert fd == pytest.approx(np.einsum("ij,ji", b, first.J), abs=1e-8)
    metric, factor = builder.metric, builder.factor
    metric[:] = 0; factor[:] = 0; options.absolute_cutoff = 1e20
    before = first.J.copy(); a[:] = 0
    np.testing.assert_allclose(builder.build(b).J, second.J, atol=0)
    del builder, orbital, auxiliary; gc.collect()
    np.testing.assert_array_equal(first.J, before)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"),
    ("shell_calls", "shell_calls"), ("primitive_quartets", "primitive_products"),
    ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_combined_setup_and_build_resource_boundaries(field, plan_field):
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    options = _options(2)
    plan = manta.plan_ri(orbital, auxiliary, options)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    builder = manta.RIJKBuilder(orbital, auxiliary, options, budget=budget)
    builder.build(_trial(orbital.nbasis))
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.RIJKBuilder(orbital, auxiliary, options, budget=budget)


@pytest.mark.parametrize("field,value", [("absolute_cutoff", -1.), ("relative_cutoff", np.nan),
    ("negative_relative_tolerance", np.inf), ("negative_relative_tolerance", -1.),
    ("mode_block", 0), ("mode_block", 65)])
def test_invalid_metric_policy_refused(field, value):
    options = _options(); setattr(options, field, value)
    with pytest.raises(ValueError):
        manta.RIJKBuilder(_basis(), _basis((0,)), options)


@pytest.mark.parametrize("entry", ["build", "build_J", "build_K", "build_g_rhf", "build_J_slot", "build_K_slot", "build_K_pair", "build_exchange_pair"])
@pytest.mark.parametrize("kind", ["float32", "complex", "strided", "nan", "asymmetric"])
def test_all_density_entries_are_strict(entry, kind):
    orbital = _basis()
    builder = manta.RIJKBuilder(orbital, _basis((0, 1, 2)))
    n = orbital.nbasis
    good = np.eye(n)
    bad = {"float32": good.astype(np.float32), "complex": good.astype(complex),
           "strided": np.eye(2*n)[::2, ::2], "nan": good*np.nan,
           "asymmetric": np.triu(np.ones((n, n)))}[kind]
    args = [bad, 0] if entry.endswith("_slot") else [bad, good] if entry.endswith("_pair") else [bad]
    with pytest.raises(ValueError):
        getattr(builder, entry)(*args)
    if entry.endswith("_pair"):
        with pytest.raises(ValueError):
            getattr(builder, entry)(good, bad)


def test_block_size_changes_work_and_memory_but_not_source_or_fit():
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    small = manta.RIJKBuilder(orbital, auxiliary, _options(1))
    large = manta.RIJKBuilder(orbital, auxiliary, _options(4))
    assert small.source_identity == large.source_identity
    assert small.plan.modeled_peak_bytes < large.plan.modeled_peak_bytes
    assert small.plan.shell_calls > large.plan.shell_calls
    d = _trial(orbital.nbasis)
    a, b = small.build(d), large.build(d)
    np.testing.assert_allclose(a.J, b.J, atol=2e-12)
    np.testing.assert_allclose(a.K, b.K, atol=2e-12)
    assert a.receipt.shell_calls > b.receipt.shell_calls


def test_composed_ri_budget_counts_callers_live_storage_once():
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    base = manta.RIJKBuilder(orbital, auxiliary).plan.modeled_peak_bytes
    budget = manta.ResourceBudget(); budget.other_live_bytes = 1234567
    budget.memory_bytes = base+budget.other_live_bytes
    builder = manta.RIJKBuilder(orbital, auxiliary, budget=budget)
    assert builder.plan.modeled_peak_bytes == budget.memory_bytes
    assert np.isfinite(builder.build_J(_trial(orbital.nbasis))).all()
    budget.memory_bytes -= 1
    with pytest.raises(ValueError, match="memory budget"):
        manta.RIJKBuilder(orbital, auxiliary, budget=budget)


@pytest.mark.parametrize("ks", [False, True])
def test_native_hf_and_ks_hosts_match_existing_df(ks):
    for unrestricted in [False, True]:
        positions = [[0., 0., 0.], [.2, -.1, 1.4]]+([[.8, .3, 2.6]] if unrestricted else [])
        mol = vq.Molecule([vq.Atom(1, p) for p in positions], 0, 2 if unrestricted else 1)
        orbital, auxiliary = vq.BasisSet(mol, "sto-3g"), vq.BasisSet(mol, "def2-svp-jk")
        overlap = vq.compute_overlap(orbital)
        hcore = vq.compute_kinetic(orbital)+vq.compute_nuclear(orbital, mol)
        enuc = sum(1/np.linalg.norm(np.array(a)-b) for i, a in enumerate(positions) for b in positions[i+1:])
        if ks:
            options = vq.UKSOptions() if unrestricted else vq.RKSOptions()
            options.functional = "pbe0"
            grid_options = vq.GridOptions()
            grid_options.n_radial, grid_options.angular, grid_options.lebedev_order = 20, "lebedev", 11
            grid = core.build_grid(mol, grid_options)
        else:
            options = vq.UHFOptions() if unrestricted else vq.RHFOptions()
        options.initial_guess = vq.InitialGuess.HCORE
        options.conv_tol_energy, options.conv_tol_grad = 1e-11, 1e-9
        builders = [vq.make_df_jk_builder(orbital, auxiliary), manta.RIJKBuilder(orbital, auxiliary)]
        results = []
        for builder in builders:
            if ks and unrestricted:
                result = core.run_uks_scf_with_jk(orbital, 2, 1, overlap, hcore, enuc, builder, grid, options)
            elif ks:
                result = core.run_rks_scf_with_jk(orbital, 2, overlap, hcore, enuc, builder, grid, options)
            elif unrestricted:
                result = vq.run_uhf_scf_with_jk(orbital, 2, 1, overlap, hcore, enuc, builder, options)
            else:
                result = vq.run_rhf_scf_with_jk(orbital, 2, overlap, hcore, enuc, builder, options)
            results.append(result)
        assert all(r.converged for r in results)
        assert results[0].energy == pytest.approx(results[1].energy, abs=2e-10)
        for field in (["density_alpha", "density_beta"] if unrestricted else ["density"]):
            np.testing.assert_allclose(getattr(results[0], field), getattr(results[1], field), atol=2e-9)


@pytest.mark.parametrize("field,value", [("centers", 3), ("derivative", 1), ("scalar", "complex"),
                                       ("kernel", "unknown"), ("omega", .3)])
def test_unadmitted_kernel_requests_refuse_before_construction(field, value):
    request = manta.KernelRequest(); setattr(request, field, value)
    with pytest.raises(ValueError):
        manta.RIJKBuilder(_basis(), _basis((0,)), request=request)


def test_rank_policy_and_range_are_part_of_source_identity():
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    original = manta.RIJKBuilder(orbital, auxiliary)
    changed = manta.RIJKBuilder(orbital, auxiliary, _options(2, relative=.1))
    request = manta.KernelRequest(); request.kernel, request.omega = "erf", .4
    ranged = manta.RIJKBuilder(orbital, auxiliary, request=request)
    assert len({original.source_identity, changed.source_identity, ranged.source_identity}) == 3
    d = _trial(orbital.nbasis)
    np.testing.assert_array_equal(original.build(d).J, original.build(np.asfortranarray(d)).J)
    assert original.build(d).receipt.density_identity == original.build(np.asfortranarray(d)).receipt.density_identity
    with pytest.raises(RuntimeError):
        original.build_K_erf(d, .4)
