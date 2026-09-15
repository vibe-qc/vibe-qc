"""Shared-q batched RSGDF cderi builds: exact work-sharing regressions.

The multi-k GDF drivers group the ``(k_i, k_j)`` cderi pairs by their
canonical momentum transfer ``q = k_j - k_i`` and build each group
through ONE batched ket-Bloch pair-FT pass
(:func:`vibeqc.aux_basis.build_lpq_bloch_native_fft_shared_q`,
backed by the multi-k C++ kernel
``ao_pair_fourier_transform_bloch_multi``). The per-cell
McMurchie-Davidson work is k-independent -- the ket momentum enters only
through the Bloch phase ``exp(+i k.R)`` -- so the batch must agree with
per-pair single builds to floating-point rounding while running the
dominant pair-FT pass once per unique q instead of once per pair
(``n_k^2 -> n_k`` on a regular full mesh; the open perf item in
``handovers/HANDOVER_OPEN_BUGS_V015.md``, production-k-sampling
follow-ups).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._aopair_ft import (
    ao_pair_fourier_transform_bloch,
    ao_pair_fourier_transform_bloch_multi,
)
from vibeqc._vibeqc_core import LatticeSumOptions
from vibeqc.aux_basis import (
    build_lpq_bloch_native_fft,
    build_lpq_bloch_native_fft_shared_q,
    make_aux_basis_set,
    make_modrho_aux_basis,
)
from vibeqc.periodic_k_gdf import _build_rsgdf_lpq_cache_shared_q


def _skew_h2_fixture():
    lattice = np.array(
        [[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]],
        dtype=float,
    ).T
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.2, 0.3, 0.4]), vq.Atom(1, [1.6, 0.3, 0.4])],
    )
    molecule = system.unit_cell_molecule()
    basis = vq.BasisSet(molecule, "sto-3g")
    auxiliary = make_modrho_aux_basis(
        make_aux_basis_set(molecule, aux_name="def2-svp-jk"),
        molecule,
    )
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    options = LatticeSumOptions()
    options.cutoff_bohr = 8.0
    return system, basis, auxiliary, reciprocal, options


def _skew_lih_fixture():
    """LiH variant: carries p shells, so the general-L cart->sph fold
    (where the batched kernel reorders the Bloch phase application)
    is exercised, not just the closed-form s path."""
    lattice = np.array(
        [[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]],
        dtype=float,
    ).T
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(3, [0.2, 0.3, 0.4]), vq.Atom(1, [3.2, 0.3, 0.4])],
    )
    molecule = system.unit_cell_molecule()
    basis = vq.BasisSet(molecule, "sto-3g")
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    return system, basis, reciprocal


@pytest.mark.parametrize("q_fraction", [(0.0, 0.0, 0.0), (0.23, -0.17, 0.11)])
def test_bounded_reciprocal_sphere_matches_complete_integer_sum(q_fraction):
    from itertools import product
    from vibeqc.aux_basis import _rsgdf_bounded_reciprocal_sphere

    system, _, _, reciprocal, _ = _skew_h2_fixture()
    q = reciprocal @ q_fraction
    energy = 3.17
    actual = _rsgdf_bounded_reciprocal_sphere(
        system, q, energy, output_byte_cap=2**20,
        workspace_byte_cap=2**20, candidate_cap=100000,
    )
    indices = np.asarray(list(product(range(-12, 13), repeat=3)))
    points = indices @ reciprocal.T + q
    expected = points[np.linalg.norm(points, axis=1) <= np.sqrt(2 * energy)]

    def vector_set(array):
        return {tuple(row) for row in np.round(array, 11)}

    assert vector_set(actual) == vector_set(expected)
    relabelled = _rsgdf_bounded_reciprocal_sphere(
        system, q + reciprocal[:, 0], energy, output_byte_cap=2**20,
        workspace_byte_cap=2**20, candidate_cap=100000,
    )
    np.testing.assert_array_equal(actual, relabelled)
    reversed_q = _rsgdf_bounded_reciprocal_sphere(
        system, -q, energy, output_byte_cap=2**20,
        workspace_byte_cap=2**20, candidate_cap=100000,
    )
    assert vector_set(actual) == vector_set(-reversed_q)


def test_bounded_reciprocal_sphere_refuses_dense_core_tail_before_enumeration(monkeypatch):
    from vibeqc.aux_basis import _rsgdf_bounded_reciprocal_sphere

    system, _, _, _, _ = _skew_h2_fixture()
    def unexpected_enumeration(*args, **kwargs):
        pytest.fail("reciprocal labels allocated before candidate admission")
    monkeypatch.setattr(np, "arange", unexpected_enumeration)
    with pytest.raises(MemoryError, match="candidate cap"):
        _rsgdf_bounded_reciprocal_sphere(
            system, np.zeros(3), 492507.0, output_byte_cap=2**20,
            workspace_byte_cap=2**20, candidate_cap=100000,
        )


def test_bounded_reciprocal_sphere_admits_exact_output_and_workspace():
    from vibeqc.aux_basis import _rsgdf_bounded_reciprocal_sphere

    system, _, _, _, _ = _skew_h2_fixture()
    kwargs = dict(workspace_byte_cap=2**20, candidate_cap=100000)
    vectors = _rsgdf_bounded_reciprocal_sphere(
        system, np.zeros(3), 3.17, output_byte_cap=2**20, **kwargs,
    )
    same = _rsgdf_bounded_reciprocal_sphere(
        system, np.zeros(3), 3.17, output_byte_cap=vectors.nbytes, **kwargs,
    )
    np.testing.assert_array_equal(vectors, same)
    with pytest.raises(MemoryError, match="output cap"):
        _rsgdf_bounded_reciprocal_sphere(
            system, np.zeros(3), 3.17, output_byte_cap=vectors.nbytes-1, **kwargs,
        )
    with pytest.raises(MemoryError, match="workspace cap"):
        _rsgdf_bounded_reciprocal_sphere(
            system, np.zeros(3), 3.17, output_byte_cap=2**20,
            workspace_byte_cap=1, candidate_cap=100000,
        )


def _range_separated_fit_fixture(displacement=0.):
    from vibeqc import _vibeqc_core as core

    centers = [[.2, .3, .4], [1.6 + displacement, .3, .4]]
    system = vq.PeriodicSystem(3, np.eye(3) * 5, [vq.Atom(1, r) for r in centers])
    mol = system.unit_cell_molecule()
    def make(exponents):
        return core.BasisSet(mol, [
            core.ShellInfo(i, 0, True, [a], [1.0], r)
            for i, r in enumerate(centers) for a in exponents
        ], "rsgdf-fit-test", coefficients_pre_normalized=False)
    return system, make([.7]), make([.4, 1.2, 3.0])


def _range_separated_fit_options():
    return dict(
        omega=.5, pair_cutoff=12.0, auxiliary_cutoff=18.0, ke_cutoff=20.0,
        linear_dep_thr=1e-11, memory_byte_cap=8 * 2**20,
        native_workspace_byte_cap=2**20, image_candidate_cap=1000000,
        reciprocal_candidate_cap=1000000,
    )


def test_range_separated_canonical_frame_preserves_gram_and_auxiliary_covariance():
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import _build_lpq_range_separated_shared_q

    system, basis, auxiliary = _range_separated_fit_fixture()
    shells = list(auxiliary.shells())
    # An exactly repeated auxiliary makes rank loss unavoidable; the
    # covariant frame must still retain the original auxiliary coordinates.
    shells.append(shells[0])
    auxiliary = core.BasisSet(system.unit_cell_molecule(), shells,
                              'duplicate-auxiliary', coefficients_pre_normalized=True)
    permutation = np.arange(len(shells))[::-1]
    reordered = core.BasisSet(system.unit_cell_molecule(),
        [shells[i] for i in permutation], 'reordered-auxiliary',
        coefficients_pre_normalized=True)
    bras, q = np.array([[.03, -.02, .01], [.11, .02, -.04]]), np.array([.13, -.07, .09])
    options = _range_separated_fit_options()
    regular = _build_lpq_range_separated_shared_q(system, basis, auxiliary, bras, q, **options)
    state = {}
    canonical = _build_lpq_range_separated_shared_q(system, basis, auxiliary, bras, q,
        canonical_auxiliary_basis=True, _metric_state=state, **options)
    assert regular.factors.shape[1] < auxiliary.nbasis
    assert canonical.factors.shape[1] == auxiliary.nbasis
    for left, right in zip(regular.factors, canonical.factors):
        left, right = left.reshape(len(left), -1), right.reshape(len(right), -1)
        np.testing.assert_allclose(left.conj().T@left, right.conj().T@right,
                                   atol=2e-12, rtol=0)
    changed = _build_lpq_range_separated_shared_q(system, basis, reordered, bras, q,
        canonical_auxiliary_basis=True, **options)
    np.testing.assert_allclose(changed.factors, canonical.factors[:, permutation],
                               atol=2e-10, rtol=0)
    reused = _build_lpq_range_separated_shared_q(system, basis, auxiliary, bras[:1], q,
        canonical_auxiliary_basis=True, _metric_state=state, **options)
    np.testing.assert_allclose(reused.factors, canonical.factors[:1], atol=2e-12, rtol=0)
    with pytest.raises(ValueError, match='different source'):
        _build_lpq_range_separated_shared_q(system, basis, auxiliary, bras[:1], q,
            _metric_state=state, **options)


def test_uniform_gaussian_lattice_bound_includes_shifted_skew_cells():
    from itertools import product
    from vibeqc.aux_basis import _rsgdf_log_lattice_gaussian_bound

    lattice = np.array([[2., .9, .4], [0., 2.2, .7], [0., 0., 1.9]])
    sigma = np.linalg.svd(lattice, compute_uv=False)[-1]
    integers = np.asarray(list(product(range(-12, 13), repeat=3)))
    for decay in (.03, .2, 1., 20.):
        bound = np.exp(_rsgdf_log_lattice_gaussian_bound(decay, sigma))
        for shift in ([0., 0., 0.], [.5, .5, .5], [.1, -.3, .4]):
            translations = (integers - shift) @ lattice.T
            actual = np.exp(-decay * np.sum(translations**2, axis=1)).sum()
            assert actual <= bound


@pytest.mark.parametrize('q', [np.zeros(3), np.array([.13, -.09, .07])])
def test_range_separated_cutoff_plan_converges_all_raw_source_terms(q):
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import (
        _plan_range_separated_gdf_cutoffs, _rsgdf_bounded_reciprocal_sphere,
    )

    system, basis, auxiliary = _range_separated_fit_fixture()
    error, omega = 1e-7, .7
    plan = _plan_range_separated_gdf_cutoffs(
        system, basis, auxiliary, q, omega=omega, raw_integral_error=error,
    )
    def raw(scale, screen):
        vectors = _rsgdf_bounded_reciprocal_sphere(
            system, q, plan.ke_cutoff * scale, output_byte_cap=2**24,
            workspace_byte_cap=2**20, candidate_cap=10**7,
        )
        return core.compute_gdf_range_separated_integrals(
            basis, auxiliary, system, q, np.array([[.1, -.2, .3]]), vectors,
            omega, plan.pair_cutoff * scale, plan.auxiliary_cutoff * scale,
            2**24, 2**20, 10**7, screen,
        )
    actual, reference = raw(1., plan.integral_screen_error), raw(1.2, 0.)
    np.testing.assert_allclose(actual[0], reference[0], rtol=0, atol=error)
    np.testing.assert_allclose(actual[1], reference[1], rtol=0, atol=error)
    tighter = _plan_range_separated_gdf_cutoffs(
        system, basis, auxiliary, q, omega=omega, raw_integral_error=error / 100,
    )
    assert tighter.pair_cutoff > plan.pair_cutoff
    assert tighter.auxiliary_cutoff > plan.auxiliary_cutoff
    assert tighter.ke_cutoff > plan.ke_cutoff


def test_range_separated_cutoff_mesh_does_not_follow_steepest_exponent():
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import _plan_range_separated_gdf_cutoffs

    system = vq.PeriodicSystem(3, np.eye(3) * 4., [vq.Atom(2, [0., 0., 0.])])
    plans = []
    for exponent in (20., 50000., 160000.):
        basis = core.BasisSet(system.unit_cell_molecule(), [
            core.ShellInfo(0, 0, True, [a], [1.], [0., 0., 0.])
            for a in (.4, exponent)
        ], 'cutoff-dense-core')
        plans.append(_plan_range_separated_gdf_cutoffs(
            system, basis, basis, np.zeros(3), omega=.6, raw_integral_error=1e-10,
        ))
    assert max(plan.ke_cutoff for plan in plans) < 100
    np.testing.assert_allclose(
        [plan.ke_cutoff for plan in plans], plans[0].ke_cutoff, rtol=0, atol=1e-12,
    )


def test_range_separated_angular_fit_matches_converged_bare_operator():
    """Signed s/p/d shells pin SR/LR relative normalization independently."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import _build_lpq_range_separated_shared_q

    system, _, _ = _range_separated_fit_fixture()
    molecule = system.unit_cell_molecule()
    def basis(angulars):
        return core.BasisSet(molecule, [
            core.ShellInfo(atom, angular, True, [.45, .9], [1., -.13], center)
            for atom, center in enumerate(([.2, .3, .4], [1.6, .3, .4]))
            for angular in angulars
        ], 'angular-source-comparison')
    orbital, auxiliary = basis((0, 1)), basis((0, 1, 2))
    q = np.array([.14, .04, -.09])
    kpoints = np.array([[.2, -.1, .3]])
    options = _range_separated_fit_options()
    options.update(memory_byte_cap=32 * 2**20, native_workspace_byte_cap=8 * 2**20)
    actual = _build_lpq_range_separated_shared_q(
        system, orbital, auxiliary, kpoints, q, **options,
    ).factors[0].reshape(-1, orbital.nbasis**2)
    lattice_options = LatticeSumOptions()
    lattice_options.cutoff_bohr = 18.
    reference = build_lpq_bloch_native_fft_shared_q(
        system, orbital, auxiliary, kpoints, q, ke_cutoff=160.,
        lat_opts=lattice_options, linear_dep_thr=options['linear_dep_thr'],
    )[0].reshape(-1, orbital.nbasis**2)
    np.testing.assert_allclose(
        actual.conj().T @ actual, reference.conj().T @ reference, rtol=0, atol=1e-9,
    )


@pytest.mark.parametrize("q", [np.zeros(3), np.array([.19, -.11, .07])])
def test_range_separated_fit_matches_converged_bare_fit(q):
    from vibeqc.aux_basis import _build_lpq_range_separated_shared_q

    system, basis, auxiliary = _range_separated_fit_fixture()
    k_bras = np.array([[0., 0., 0.], [.21, .13, -.07]])
    result = _build_lpq_range_separated_shared_q(
        system, basis, auxiliary, k_bras, q, **_range_separated_fit_options(),
    )
    opts = LatticeSumOptions()
    opts.cutoff_bohr = 12.0
    reference = build_lpq_bloch_native_fft_shared_q(
        system, basis, auxiliary, k_bras, q, ke_cutoff=120., lat_opts=opts,
        linear_dep_thr=1e-11,
    )
    for actual, expected in zip(result.factors, reference):
        a = actual.reshape(actual.shape[0], -1)
        e = expected.reshape(expected.shape[0], -1)
        np.testing.assert_allclose(a.conj().T @ a, e.conj().T @ e,
                                   atol=2e-10, rtol=2e-10)


def test_range_separated_shared_fit_keeps_gamma_single_and_batch_identical():
    from vibeqc.aux_basis import _build_lpq_range_separated_shared_q

    system, basis, auxiliary = _range_separated_fit_fixture()
    ks = np.array([[0., 0., 0.], [.21, .13, -.07]])
    options = _range_separated_fit_options()
    batch = _build_lpq_range_separated_shared_q(
        system, basis, auxiliary, ks, np.zeros(3), **options,
    )
    for k, expected in zip(ks, batch.factors):
        single = _build_lpq_range_separated_shared_q(
            system, basis, auxiliary, k[None], np.zeros(3), **options,
        )
        np.testing.assert_array_equal(single.factors[0], expected)
    # The peak reservation includes the actual mesh, the native workspace,
    # queried LAPACK scratch, raw integrals and the whitened factor output.
    options['memory_byte_cap'] = batch.reserved_peak_bytes
    admitted = _build_lpq_range_separated_shared_q(
        system, basis, auxiliary, ks, np.zeros(3), **options,
    )
    np.testing.assert_array_equal(admitted.factors, batch.factors)
    options['memory_byte_cap'] -= 1
    with pytest.raises(MemoryError, match="reciprocal output cap"):
        _build_lpq_range_separated_shared_q(
            system, basis, auxiliary, ks, np.zeros(3), **options,
        )


@pytest.mark.parametrize("omega", [.5, .8])
def test_range_separated_source_state_gradient_matches_fixed_density_energy(omega, monkeypatch):
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import (
        _build_lpq_range_separated_shared_q, _canonical_reciprocal_transfer,
        _build_coulomb_from_diagonal_factors, _accumulate_exchange_from_factors,
    )
    from vibeqc.periodic_gdf_gradient import (
        _compute_range_separated_jk_gradient, _compute_range_separated_cache_gradient,
    )
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    system, basis, auxiliary = _range_separated_fit_fixture()
    kpoints = np.asarray(vq.monkhorst_pack(system, [2, 1, 1]).kpoints)
    rng = np.random.default_rng(662)
    density = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    density = .1 * (density @ density.conj().transpose(0, 2, 1))
    weights = np.array([.5, .5])
    options = dict(_range_separated_fit_options(), omega=omega, retain_fit_state=True)
    groups = [[(0, 0), (1, 1)], [(0, 1), (1, 0)]]

    def displaced(step):
        atoms = [vq.Atom(int(a.Z), np.asarray(a.xyz) + ([step, 0., 0.] if i else 0))
                 for i, a in enumerate(system.unit_cell)]
        shifted = vq.PeriodicSystem(3, np.asarray(system.lattice), atoms)
        molecule = shifted.unit_cell_molecule()
        def move(original):
            shells = [core.ShellInfo(
                int(s.atom_index), int(s.l), bool(s.pure), s.exponents, s.coefficients,
                np.asarray(s.origin) + ([step, 0., 0.] if s.atom_index == 1 else 0),
            ) for s in original.shells()]
            return core.BasisSet(molecule, shells, "displaced-source", coefficients_pre_normalized=True)
        return shifted, move(basis), move(auxiliary)

    def evaluate(step, gradient=False):
        cell, ao, aux = displaced(step)
        cache, batches = {}, []
        for pairs in groups:
            i, j = pairs[0]
            q = _canonical_reciprocal_transfer(cell, kpoints[j] - kpoints[i])
            batch = _build_lpq_range_separated_shared_q(
                cell, ao, aux, np.asarray([kpoints[i] for i, _ in pairs]), q, **options,
            )
            batches.append(batch)
            cache.update(zip(pairs, batch.factors))
        coulomb = _build_coulomb_from_diagonal_factors(
            [cache[i, i] for i in range(2)], density, weights,
        )
        energy = .5 * sum(w * np.trace(d @ j).real
                          for w, d, j in zip(weights, density, coulomb))
        for (i, j), factor in cache.items():
            exchange = np.zeros((2, 2), dtype=complex)
            _accumulate_exchange_from_factors(exchange, factor, density[j], weights[j], 2**20)
            energy -= .25 * weights[i] * np.trace(density[i] @ exchange).real
        if not gradient:
            return energy
        derivative = np.zeros((2, 3))
        subdivided_derivative = np.zeros_like(derivative)
        for position, (pairs, batch) in enumerate(zip(groups, batches)):
            derivative += _compute_range_separated_jk_gradient(
                cell, ao, aux, batch, pairs, density, weights,
                memory_byte_cap=8 * 2**20, native_workspace_byte_cap=2**20,
                coulomb_scale=1. if position == 0 else 0.,
            )
            source = batch.fit_state.three_center
            full_charge = sum(
                weights[i] * (source[k].reshape(aux.nbasis, -1) @ density[i].T.reshape(-1))
                for k, (i, _) in enumerate(pairs)
            ) if position == 0 else None
            for k, pair in enumerate(pairs):
                state = batch.fit_state._replace(
                    three_center=source[k:k + 1],
                    ket_kpoints=batch.fit_state.ket_kpoints[k:k + 1],
                )
                subdivided_derivative += _compute_range_separated_jk_gradient(
                    cell, ao, aux, batch._replace(fit_state=state), [pair], density, weights,
                    memory_byte_cap=8 * 2**20, native_workspace_byte_cap=2**20,
                    coulomb_scale=1. if position == 0 else 0.,
                    coulomb_source=full_charge, include_coulomb_metric=k == 0,
                )
        np.testing.assert_allclose(subdivided_derivative, derivative, rtol=0, atol=2e-10)
        source_cache = _build_range_separated_lpq_cache(
            cell, ao, aux, kpoints, True,
            **{k: v for k, v in options.items() if k != 'retain_fit_state'},
        )
        class UnmaterializableDensity:
            def __array__(self, *args, **kwargs):
                pytest.fail("density materialized before gradient workspace admission")
        with pytest.raises(MemoryError, match="source and response"):
            _compute_range_separated_cache_gradient(
                cell, ao, aux, source_cache, UnmaterializableDensity(), weights,
                memory_byte_cap=1, native_workspace_byte_cap=2**20,
            )
        # Force one-pair batches to exercise the global Hartree source
        # through the actual memory-admission fallback, not by slicing
        # already retained full-q tensors.
        import vibeqc.aux_basis as aux_module
        calls = {}
        native_source = core.compute_gdf_range_separated_integrals
        def recording_source(*args, **kwargs):
            calls.setdefault(tuple(args[3]), []).append(bool(args[-1]))
            return native_source(*args, **kwargs)
        def single_pair_source(*args, **kwargs):
            if len(args[3]) > 1:
                raise aux_module._RangeSeparatedGdfAdmissionError('single-pair test budget')
            return _build_lpq_range_separated_shared_q(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(aux_module, '_build_lpq_range_separated_shared_q', single_pair_source)
            patch.setattr(core, 'compute_gdf_range_separated_integrals', recording_source)
            streamed = _compute_range_separated_cache_gradient(
                cell, ao, aux, source_cache, density, weights,
                memory_byte_cap=16*2**20, native_workspace_byte_cap=2**20,
            )
        np.testing.assert_allclose(streamed, derivative, rtol=0, atol=2e-10)
        assert len(calls) == 2
        for metric_calls in calls.values():
            assert metric_calls[0] and sum(metric_calls) == 1
        other, other_ao, other_aux = displaced(.01)
        with pytest.raises(ValueError, match="energy cache's geometry"):
            _compute_range_separated_cache_gradient(
                other, other_ao, other_aux, source_cache, density, weights,
                memory_byte_cap=16*2**20, native_workspace_byte_cap=2**20,
            )
        with pytest.raises(ValueError, match="differs from the retained fit"):
            _compute_range_separated_jk_gradient(
                other, other_ao, other_aux, batches[0], groups[0], density, weights,
                memory_byte_cap=8 * 2**20, native_workspace_byte_cap=2**20,
            )
        return derivative

    gradient = evaluate(0., gradient=True)
    h = 1e-5
    fd = (evaluate(h) - evaluate(-h)) / (2 * h)
    assert gradient[1, 0] == pytest.approx(fd, abs=3e-8, rel=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=2e-10, rtol=0)


def test_range_separated_fit_refuses_bad_metric_without_projection(monkeypatch):
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import _build_lpq_range_separated_shared_q

    system, basis, auxiliary = _range_separated_fit_fixture()
    options = _range_separated_fit_options()
    metric = np.eye(auxiliary.nbasis, dtype=complex)
    def source(*args):
        return metric.copy(), np.zeros((1, auxiliary.nbasis, 2, 2), complex), 1
    monkeypatch.setattr(core, 'compute_gdf_range_separated_integrals', source)
    metric[0, 1] = 0.1j
    with pytest.raises(ValueError, match="not Hermitian"):
        _build_lpq_range_separated_shared_q(
            system, basis, auxiliary, np.zeros((1, 3)), np.zeros(3), **options,
        )
    metric[0, 1] = 0
    metric[0, 0] = -0.1
    with pytest.raises(ValueError, match="negative modes"):
        _build_lpq_range_separated_shared_q(
            system, basis, auxiliary, np.zeros((1, 3)), np.zeros(3), **options,
        )


@pytest.mark.parametrize("repetitions", [1, 1100])
def test_shared_coulomb_is_invariant_under_complex_auxiliary_rotation(repetitions):
    from vibeqc.aux_basis import _build_coulomb_from_diagonal_factors

    rng = np.random.default_rng(142)
    weights = [.2, .3, .5]
    factors, densities = [], []
    for _ in weights:
        raw = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
        factors.append(raw + raw.transpose(0, 2, 1).conj())
        raw_density = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        densities.append(raw_density @ raw_density.conj().T)
    # In the original real auxiliary coordinate system, every L_P is
    # Hermitian and the standard tr(L_P D) fitted density is real.
    rho = sum(w * np.asarray([np.trace(f @ d) for f in fs])
              for w, fs, d in zip(weights, factors, densities))
    expected = [sum(r * f for r, f in zip(rho, fs)) for fs in factors]
    unitary, _ = np.linalg.qr(rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7)))
    rotated = [(unitary @ f.reshape(7, -1)).reshape(f.shape) for f in factors]
    # Repetition scales the fitted operator linearly and crosses the
    # conjugation panel boundary without allocating a dense block unitary.
    rotated = [np.tile(f, (repetitions, 1, 1)) for f in rotated]
    actual = _build_coulomb_from_diagonal_factors(rotated, densities, weights)
    # The repeated case sums 7700 fit rows; allow its accumulation roundoff
    # while keeping the seven-row coordinate-invariance gate tighter.
    np.testing.assert_allclose(np.asarray(actual) / repetitions, expected,
                               atol=2e-12 if repetitions == 1 else 2e-11, rtol=0)
    for matrix in actual:
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14, rtol=0)


@pytest.mark.parametrize("bra_rows,need_pairs", [(None, True), ([0], True), (None, False)])
def test_range_separated_cache_bounds_retention_and_keeps_all_hartree_diagonals(
    bra_rows, need_pairs,
):
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    system, basis, auxiliary = _range_separated_fit_fixture()
    kpoints = np.asarray(vq.monkhorst_pack(system, [2, 1, 1]).kpoints)
    options = _range_separated_fit_options()
    result = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, need_pairs, bra_rows=bra_rows, **options,
    )
    expected = {(0, 0), (1, 1)}
    if need_pairs:
        expected.update((i, j) for i in (range(2) if bra_rows is None else bra_rows)
                        for j in range(2))
    assert set(result.factors) == expected
    assert result.retained_factor_bytes == sum(f.nbytes for f in result.factors.values())
    assert result.retained_factor_bytes < result.reserved_peak_bytes <= options['memory_byte_cap']
    assert len(result.reciprocal_vector_counts) == (2 if need_pairs else 1)
    options['memory_byte_cap'] = result.reserved_peak_bytes
    same = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, need_pairs, bra_rows=bra_rows, **options,
    )
    for pair in expected:
        np.testing.assert_array_equal(same.factors[pair], result.factors[pair])
    n_pairs = len(expected)
    options['memory_byte_cap'] = (
        16 * n_pairs * auxiliary.nbasis * basis.nbasis**2
        + 4096 + 512 * n_pairs + 256 * len(kpoints) + 1
    )
    with pytest.raises(MemoryError, match="raw integral reservation"):
        _build_range_separated_lpq_cache(
            system, basis, auxiliary, kpoints, need_pairs, bra_rows=bra_rows, **options,
        )


@pytest.mark.parametrize('need_pairs', [False, True])
def test_range_separated_pair_adapter_keeps_common_source_and_memory_cap(monkeypatch, need_pairs):
    import vibeqc.aux_basis as aux_module
    from vibeqc.aux_basis import _build_coulomb_from_diagonal_factors
    from vibeqc.periodic_k_gdf import (
        _build_range_separated_lpq_cache, _build_k_from_lpq_cache,
    )

    system, basis, auxiliary = _range_separated_fit_fixture()
    mesh = vq.monkhorst_pack(system, [2, 1, 1])
    kpoints = np.asarray(mesh.kpoints)
    options = _range_separated_fit_options()
    reference = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, need_pairs, **options,
    )
    admitted = []
    source = aux_module._build_lpq_range_separated_shared_q
    def record_source(*args, **kwargs):
        assert kwargs['canonical_auxiliary_basis']
        result = source(*args, **kwargs)
        admitted.append((kwargs['memory_byte_cap'], result.reserved_peak_bytes))
        return result
    monkeypatch.setattr(aux_module, '_build_lpq_range_separated_shared_q', record_source)
    def adapter(build_pair, points, orbital, fitting, exchange):
        assert orbital is basis and fitting is auxiliary and exchange == need_pairs
        return {(i, j): build_pair(points[i], points[j], canonical_auxiliary_basis=True)
                for i, j in reference.factors}
    result = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, need_pairs,
        pair_cache_builder=adapter, **options,
    )
    assert len(admitted) == len(reference.factors)
    assert result.source_parameters == reference.source_parameters
    assert result.source_signature == reference.source_signature
    assert result.factor_frame == 'canonical'
    assert result.retained_factor_bytes == sum(f.nbytes for f in result.factors.values())
    assert result.reserved_peak_bytes <= options['memory_byte_cap']
    transport = 64 * (auxiliary.nbasis * basis.nbasis**2 + auxiliary.nbasis**2) + 65536
    cache_bound = (16 * len(reference.factors) * auxiliary.nbasis * basis.nbasis**2
                   + 4096 + 512 * len(reference.factors) + 256 * len(kpoints))
    for cap, peak in admitted:
        assert cap + cache_bound + transport == options['memory_byte_cap']
        assert peak + cache_bound + transport <= result.reserved_peak_bytes
    density = [np.array([[.6, .1j], [-.1j, .4]]),
               np.array([[.3, .07-.02j], [.07+.02j, .7]])]
    actual_j = _build_coulomb_from_diagonal_factors(
        [result.factors[i, i] for i in range(2)], density, mesh.weights,
    )
    expected_j = _build_coulomb_from_diagonal_factors(
        [reference.factors[i, i] for i in range(2)], density, mesh.weights,
    )
    np.testing.assert_allclose(actual_j, expected_j, rtol=0, atol=2e-12)
    if need_pairs:
        actual_k = _build_k_from_lpq_cache(result.factors, density, mesh.weights, nbasis=2)
        expected_k = _build_k_from_lpq_cache(reference.factors, density, mesh.weights, nbasis=2)
        np.testing.assert_allclose(actual_k, expected_k, rtol=0, atol=2e-12)


def test_range_separated_canonical_cache_gradient_handles_discarded_metric_modes():
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache
    from vibeqc.periodic_gdf_gradient import _compute_range_separated_cache_gradient

    system, basis, auxiliary = _range_separated_fit_fixture()
    shells = list(auxiliary.shells())
    auxiliary = core.BasisSet(system.unit_cell_molecule(), shells+[shells[0]],
                             'rank-deficient-fit', coefficients_pre_normalized=True)
    options = _range_separated_fit_options()
    points = np.zeros((1, 3))
    def adapter(build_pair, kpoints, orbital, fitting, need_pairs):
        return {(0, 0): build_pair(kpoints[0], kpoints[0], canonical_auxiliary_basis=True)}
    plain = _build_range_separated_lpq_cache(system, basis, auxiliary, points, True, **options)
    canonical = _build_range_separated_lpq_cache(
        system, basis, auxiliary, points, True, pair_cache_builder=adapter, **options,
    )
    assert plain.n_fit < canonical.n_fit == auxiliary.nbasis
    density = np.array([[[1., .13], [.13, .8]]])
    def gradient(cache):
        return _compute_range_separated_cache_gradient(
            system, basis, auxiliary, cache, density, [1.],
            memory_byte_cap=16*2**20, native_workspace_byte_cap=2**20,
        )
    np.testing.assert_allclose(gradient(canonical), gradient(plain), atol=2e-11, rtol=0)
    canonical.factors[0, 0][0, 0, 0] += 1e-3
    with pytest.raises(RuntimeError, match='canonical factors differ'):
        gradient(canonical)


def test_range_separated_cache_subdivides_q_group_and_reuses_metric(monkeypatch):
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import (
        _build_lpq_range_separated_shared_q, _canonical_reciprocal_transfer,
    )
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    system, basis, auxiliary = _range_separated_fit_fixture()
    kpoints = np.asarray(vq.monkhorst_pack(system, [2, 1, 1]).kpoints)
    options = _range_separated_fit_options()
    reference = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, True, **options,
    )
    maximum_single = 0
    for j in range(2):
        q = _canonical_reciprocal_transfer(system, kpoints[j] - kpoints[0])
        single = _build_lpq_range_separated_shared_q(
            system, basis, auxiliary, kpoints[:1], q, **options,
        )
        maximum_single = max(maximum_single, single.reserved_peak_bytes)
    del single
    cache_bound = (16 * 4 * auxiliary.nbasis * basis.nbasis**2
                   + 4096 + 512 * 4 + 256 * 2)
    options['memory_byte_cap'] = cache_bound + maximum_single
    calls = {}
    native = core.compute_gdf_range_separated_integrals
    def recording_source(*args, **kwargs):
        calls.setdefault(tuple(args[3]), []).append(bool(args[-1]))
        return native(*args, **kwargs)
    monkeypatch.setattr(core, 'compute_gdf_range_separated_integrals', recording_source)
    bounded = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, True, **options,
    )
    assert bounded.reserved_peak_bytes <= options['memory_byte_cap']
    assert any(len(sizes) > 1 for sizes in bounded.q_batch_sizes)
    assert len(calls) == 2
    for builds in calls.values():
        assert builds[0] and sum(builds) == 1
    for pair, expected in reference.factors.items():
        np.testing.assert_array_equal(bounded.factors[pair], expected)


def test_range_separated_metric_reuse_rejects_a_changed_source():
    from vibeqc.aux_basis import _build_lpq_range_separated_shared_q

    system, basis, auxiliary = _range_separated_fit_fixture()
    options = _range_separated_fit_options()
    state = {}
    _build_lpq_range_separated_shared_q(
        system, basis, auxiliary, np.zeros((1, 3)), np.zeros(3),
        _metric_state=state, **options,
    )
    for changed in ({'omega': .8}, {'linear_dep_thr': 1e-9}, {'pair_cutoff': 13.}):
        with pytest.raises(ValueError, match="different source"):
            _build_lpq_range_separated_shared_q(
                system, basis, auxiliary, np.zeros((1, 3)), np.zeros(3),
                _metric_state=state, **dict(options, **changed),
            )


def test_range_separated_cache_refuses_retention_before_any_integrals(monkeypatch):
    import vibeqc.aux_basis as aux_module
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    system, basis, auxiliary = _range_separated_fit_fixture()
    kpoints = np.asarray(vq.monkhorst_pack(system, [2, 1, 1]).kpoints)
    options = _range_separated_fit_options()
    options['memory_byte_cap'] = 16 * 4 * auxiliary.nbasis * basis.nbasis**2 - 1
    def forbidden(*args, **kwargs):
        pytest.fail("integrals started before retained-cache admission")
    monkeypatch.setattr(aux_module, '_build_lpq_range_separated_shared_q', forbidden)
    with pytest.raises(MemoryError, match="retained factor cache"):
        _build_range_separated_lpq_cache(system, basis, auxiliary, kpoints, True, **options)


@pytest.mark.parametrize("error", [MemoryError, ValueError])
def test_range_separated_cache_never_retries_numerical_or_allocator_failure(monkeypatch, error):
    import vibeqc.aux_basis as aux_module
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    system, basis, auxiliary = _range_separated_fit_fixture()
    kpoints = np.asarray(vq.monkhorst_pack(system, [2, 1, 1]).kpoints)
    calls = []
    def failed_source(*args, **kwargs):
        calls.append(len(args[3]))
        raise error("source calculation failed after admission")
    monkeypatch.setattr(aux_module, '_build_lpq_range_separated_shared_q', failed_source)
    with pytest.raises(error, match="failed after admission"):
        _build_range_separated_lpq_cache(
            system, basis, auxiliary, kpoints, True, **_range_separated_fit_options(),
        )
    assert calls == [2]


@pytest.mark.parametrize("omega", [.5, .8])
def test_range_separated_two_k_jk_matches_external_reference(omega):
    """Fixed-D PySCF 2.6.2 RSGDF reference, vq job 30d7e733b5dd.

    Same normalized Gaussian fixture, Gamma-centred (2,1,1) mesh, RNG
    seed 142 and two electrons per k. The out-of-process reference used
    precision_R/G=1e-12, precision_j2c=1e-13, linear_dep_threshold=1e-10,
    omega_j2c=omega, exp_to_discard=0, and both None/Ewald exxdiv.
    Its omega 0.5 and 0.8 results agree within 2e-13. Constants below
    are numerical outputs, not an external-program runtime dependency.
    """
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import _build_coulomb_from_diagonal_factors
    from vibeqc.periodic_k_gdf import (
        _build_k_from_lpq_cache, _build_range_separated_lpq_cache,
        _madelung_for_kmesh, apply_exxdiv_ewald_to_K,
    )

    system, basis, auxiliary = _range_separated_fit_fixture()
    mesh = vq.monkhorst_pack(system, [2, 1, 1])
    kpoints = np.asarray(mesh.kpoints)
    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = 15.
    overlap_set = core.compute_overlap_lattice(basis, system, lat_opts)
    overlap = [np.asarray(core.bloch_sum(overlap_set, k)) for k in kpoints]
    rng = np.random.default_rng(142)
    densities = []
    for matrix in overlap:
        x = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        density = x @ x.conj().T
        densities.append(2 * density / np.trace(density @ matrix).real)
    options = _range_separated_fit_options()
    options.update(omega=omega, ke_cutoff=48., pair_cutoff=15.)
    cache = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, True, **options,
    ).factors
    j = _build_coulomb_from_diagonal_factors(
        [cache[i, i] for i in range(2)], densities, mesh.weights,
    )
    k = _build_k_from_lpq_cache(cache, densities, mesh.weights, nbasis=2)
    k_ewald = apply_exxdiv_ewald_to_K(
        k, overlap, densities, _madelung_for_kmesh(system, (2, 1, 1)),
    )
    expected_j = [
        [[.5632174188285725, .31312100998816655],
         [.31312100998816655, .5155237186255567]],
        [[.5630830983987777, .3101296197046425],
         [.3101296197046425, .5153652174614619]],
    ]
    expected_k = [
        [[.6734349130682902, .43124453887866954 - .05620373024327302j],
         [.43124453887866954 + .05620373024327302j, .7020993050933169]],
        [[.7581266026237954, .382246137935214 - .03244091490763579j],
         [.382246137935214 + .03244091490763579j, .5635890040223049]],
    ]
    expected_k_ewald = [
        [[1.152860925610975, .5940799703448476 - .10313802600037833j],
         [.5940799703448476 + .10313802600037832j, .9220132854606096]],
        [[1.0800917784111879, .6502411516251989 - .15820309177709416j],
         [.6502411516251989 + .15820309177709418j, 1.0528090832476642]],
    ]
    np.testing.assert_allclose(j, expected_j, atol=2e-11, rtol=0)
    np.testing.assert_allclose(k, expected_k, atol=2e-11, rtol=0)
    np.testing.assert_allclose(k_ewald, expected_k_ewald, atol=2e-11, rtol=0)


@pytest.mark.parametrize("cell_kind", ["small", "small-complex-gauge", "mgo"])
def test_range_separated_source_gamma_general_fixed_density_components(monkeypatch, cell_kind):
    """Exercise both SCF assemblers on one source and accepted density."""
    import vibeqc.aux_basis as aux_module
    import vibeqc.pbc_gdf as gamma_module
    import vibeqc.periodic_k_gdf as k_module

    system, basis, _ = _range_separated_fit_fixture()
    fit_options = _range_separated_fit_options()
    fit_options.update(memory_byte_cap=64 * 2**20, native_workspace_byte_cap=16 * 2**20)
    if cell_kind == "mgo":
        a = 4.2112 / 0.529177210903
        lattice = np.array([[0., a/2, a/2], [a/2, 0., a/2], [a/2, a/2, 0.]])
        system = vq.PeriodicSystem(3, lattice, [vq.Atom(12, [0., 0., 0.]),
                                              vq.Atom(8, [a/2, 0., 0.])])
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        fit_options.update(pair_cutoff=18., auxiliary_cutoff=30.)
    def factor_gauge(factors):
        if cell_kind == "small-complex-gauge":
            # An arbitrary unitary change of auxiliary fit coordinates must
            # leave the Coulomb operator unchanged. This phase-only unitary
            # also exposes confusing D_ls with D_sl in the contraction.
            phases = np.exp(1j * np.linspace(.2, 1.1, factors.shape[0]))
            return factors * phases[:, None, None]
        return factors
    def gamma_fit(system, orbital, auxiliary, kpoints, need_pairs, **kwargs):
        cache = k_module._build_range_separated_lpq_cache(
            system, orbital, auxiliary, np.asarray(kpoints), need_pairs,
            **fit_options,
        )
        for pair, factor in cache.factors.items():
            cache.factors[pair] = factor_gauge(factor)
        return cache
    def cache_builder(old_pair_builder, kpoints, orbital, auxiliary, need_k_pairs):
        assert len(kpoints) == 1
        batch = aux_module._build_lpq_range_separated_shared_q(
            system, orbital, auxiliary, np.asarray(kpoints), np.zeros(3), **fit_options,
        )
        return {(0, 0): factor_gauge(batch.factors[0])}
    monkeypatch.setattr(k_module, '_build_scf_range_separated_lpq_cache', gamma_fit)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 1
    options.use_diis = False
    options.damping = 0
    options.initial_guess = vq.InitialGuess.HCORE
    options.lattice_opts.cutoff_bohr = fit_options['pair_cutoff']
    options.lattice_opts.nuclear_cutoff_bohr = fit_options['pair_cutoff']
    gamma = gamma_module.run_pbc_gdf_rhf(
        system, basis, options, aux_basis='def2-svp-jk', gdf_method='rsgdf',
        rcut_strategy=None, progress=False,
    )
    # Force the actual general assembler at Nk=1; the ordinary public entry
    # otherwise delegates to the Gamma path and cannot test this boundary.
    monkeypatch.setattr(k_module, '_gamma_kmesh_info', lambda *args: None)
    general = k_module.run_krhf_periodic_gdf(
        system, basis, options=options, kmesh=(1, 1, 1),
        aux_basis='def2-svp-jk', gdf_method='rsgdf', use_compcell=True,
        rcut_strategy=None, initial_density_k=[gamma.density],
        _lpq_cache_builder=cache_builder, check_energy_sanity=False, progress=False,
    )
    np.testing.assert_allclose(general.overlap[0], gamma.overlap, atol=2e-12, rtol=0)
    np.testing.assert_allclose(general.hcore[0], gamma.hcore, atol=2e-11, rtol=0)
    np.testing.assert_allclose(general.density[0], gamma.density, atol=2e-12, rtol=0)
    for field in ('energy', 'e_nuclear', 'e_coulomb', 'e_hf_exchange'):
        assert getattr(general, field) == pytest.approx(getattr(gamma, field), abs=2e-10, rel=0)
    np.testing.assert_allclose(general.fock[0], gamma.fock, atol=2e-10, rtol=0)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
def test_range_separated_source_full_scf_is_split_parameter_invariant(monkeypatch, mesh):
    """Exercise converged SCF and density return before changing source dispatch."""
    import vibeqc.aux_basis as aux_module
    import vibeqc.pbc_gdf as gamma_module
    import vibeqc.periodic_k_gdf as k_module
    from vibeqc.periodic_runner import _fit_exact_lattice_density_to_weighted_k_matrices

    system, basis, auxiliary = _range_separated_fit_fixture()
    source_options = _range_separated_fit_options()
    # Use the reference fixture's unit-square-norm auxiliary basis in both
    # assemblers. These hooks replace only the fit source; SCF, occupations,
    # Madelung assembly, convergence and the density producer execute normally.
    for module in (gamma_module, k_module):
        monkeypatch.setattr(module, "make_aux_basis_set", lambda *args, **kwargs: auxiliary)
        monkeypatch.setattr(module, "make_modrho_aux_basis", lambda aux, mol: aux)
    monkeypatch.setattr(k_module, "_gamma_kmesh_info", lambda *args: None)
    build_count = []

    def cache_builder(old_builder, kpoints, orbital, aux, need_pairs):
        build_count.append(len(kpoints))
        return k_module._build_range_separated_lpq_cache(
            system, orbital, aux, kpoints, need_pairs, **source_options,
        )

    opts = vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.max_iter = 80
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-8
    opts.lattice_opts.cutoff_bohr = 12.
    opts.lattice_opts.nuclear_cutoff_bohr = 12.
    results = []
    for omega in (.5, .8):
        source_options["omega"] = omega
        result = k_module.run_krhf_periodic_gdf(
            system, basis, kmesh=mesh, options=opts,
            gdf_method="rsgdf", use_compcell=True, rcut_strategy=None,
            aux_basis="def2-svp-jk", _lpq_cache_builder=cache_builder,
            gdf_linear_dep_threshold=source_options["linear_dep_thr"],
            return_lattice_density=True, compute_gradient=True, progress=False,
        )
        assert result.converged
        electron_count = sum(
            w * np.trace(d @ s).real
            for w, d, s in zip(result.kpoint_weights, result.density, result.overlap)
        )
        assert electron_count == pytest.approx(2., abs=2e-12)
        recovered = _fit_exact_lattice_density_to_weighted_k_matrices(
            result.density_lattice, result.kpoints_cart, result.kpoint_weights,
            system=system, bvk_mesh=mesh, label="SR/LR converged SCF",
        )
        np.testing.assert_allclose(
            recovered, np.asarray(result.density) / np.prod(mesh), rtol=0, atol=2e-12,
        )
        if mesh == (1, 1, 1):
            gamma = gamma_module.run_pbc_gdf_rhf(
                system, basis, opts, aux_basis="def2-svp-jk",
                gdf_method="rsgdf", rcut_strategy=None, progress=False,
                rsgdf_omega=omega, rsgdf_g_precision=1e-10,
                gdf_linear_dep_threshold=source_options["linear_dep_thr"],
            )
            assert gamma.converged
            assert gamma.energy == pytest.approx(result.energy, abs=2e-9, rel=0)
            np.testing.assert_allclose(gamma.density, result.density[0], rtol=0, atol=2e-8)
        results.append(result)
    assert build_count == ([1, 1] if mesh == (1, 1, 1) else [2, 2])
    for field in ("energy", "e_coulomb", "e_hf_exchange"):
        assert getattr(results[0], field) == pytest.approx(
            getattr(results[1], field), abs=2e-9, rel=0,
        )
    np.testing.assert_allclose(results[0].density, results[1].density, rtol=0, atol=2e-8)
    np.testing.assert_allclose(results[0].gradient, results[1].gradient, rtol=0, atol=3e-8)

    def displaced_energy(shift):
        cell, orbital, aux = _range_separated_fit_fixture(shift)
        def displaced_cache_builder(old_builder, kpoints, ao, fitting, need_pairs):
            return k_module._build_range_separated_lpq_cache(
                cell, ao, fitting, kpoints, need_pairs, **source_options,
            )
        with monkeypatch.context() as patch:
            patch.setattr(k_module, 'make_aux_basis_set', lambda *args, **kwargs: aux)
            result = k_module.run_krhf_periodic_gdf(
                cell, orbital, kmesh=mesh, options=opts,
                gdf_method='rsgdf', use_compcell=True, rcut_strategy=None,
                aux_basis='def2-svp-jk', _lpq_cache_builder=displaced_cache_builder,
                gdf_linear_dep_threshold=source_options['linear_dep_thr'], progress=False,
            )
        assert result.converged
        return result.energy

    step = 4e-4
    finite_difference = (
        displaced_energy(-2*step) - 8*displaced_energy(-step)
        + 8*displaced_energy(step) - displaced_energy(2*step)
    ) / (12*step)
    assert results[-1].gradient[1, 0] == pytest.approx(finite_difference, abs=3e-7, rel=0)


def test_range_separated_rohf_assembly_is_invariant_under_auxiliary_phases(monkeypatch):
    import vibeqc.periodic_rohf_gdf as rohf_module
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    system, basis, _ = _range_separated_fit_fixture()
    system.multiplicity = 3
    fit_options = _range_separated_fit_options()
    fit_options.update(memory_byte_cap=64 * 2**20, native_workspace_byte_cap=16 * 2**20)
    rotated = False
    def build_cache(system, orbital, auxiliary, kpoints, need_pairs, **kwargs):
        cache = _build_range_separated_lpq_cache(
            system, orbital, auxiliary, kpoints, need_pairs,
            **fit_options,
        ).factors
        assert set(cache) == {(0, 0)}
        if rotated:
            factors = cache[0, 0]
            phases = np.exp(1j * np.linspace(.2, 1.1, factors.shape[0]))
            cache[0, 0] = factors * phases[:, None, None]
        return cache
    monkeypatch.setattr(rohf_module, '_build_rsgdf_lpq_cache_shared_q', build_cache)
    opts = vq.PeriodicRHFOptions()
    opts.max_iter = 1
    opts.use_diis = False
    opts.damping = 0
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.lattice_opts.cutoff_bohr = 12.
    opts.lattice_opts.nuclear_cutoff_bohr = 12.
    args = dict(kmesh=(1, 1, 1), options=opts, aux_basis='def2-svp-jk',
                rcut_strategy=None, progress=False)
    reference = rohf_module.run_krohf_periodic_gdf(system, basis, **args)
    rotated = True
    actual = rohf_module.run_krohf_periodic_gdf(system, basis, **args)
    assert actual.energy == pytest.approx(reference.energy, abs=2e-10, rel=0)
    for field in ('density_alpha', 'density_beta', 'fock_alpha', 'fock_beta'):
        np.testing.assert_allclose(getattr(actual, field), getattr(reference, field),
                                   atol=2e-10, rtol=0)


def test_multi_kernel_matches_per_k_single_kernel():
    """Batched pair-FT == n_k single-k pair-FTs (general-L, skew cells)."""
    system, basis, reciprocal = _skew_lih_fixture()
    rng = np.random.default_rng(7)
    G = rng.normal(size=(37, 3)) * 1.5
    lat = np.asarray(system.lattice, dtype=float)
    cells = np.array(
        [
            [0.0, 0.0, 0.0],
            lat[:, 0],
            -lat[:, 0],
            lat[:, 2],
            -lat[:, 2],
        ],
        dtype=float,
    )
    k_carts = np.array(
        [
            [0.0, 0.0, 0.0],
            0.5 * reciprocal[:, 2],
            0.17 * reciprocal[:, 1] + 0.31 * reciprocal[:, 0],
        ],
        dtype=float,
    )

    batch = ao_pair_fourier_transform_bloch_multi(basis, G, cells, k_carts)
    assert len(batch) == k_carts.shape[0]
    for ik in range(k_carts.shape[0]):
        single = ao_pair_fourier_transform_bloch(basis, G, cells, k_carts[ik])
        np.testing.assert_allclose(
            batch[ik], single, atol=1e-12, rtol=1e-12
        )


def test_shared_q_batch_matches_per_pair_builds_canonical():
    """Batched shared-q Lpq == per-pair builds (canonical aux factor)."""
    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    k_bras = np.array(
        [
            0.17 * reciprocal[:, 1],
            -0.11 * reciprocal[:, 1],
            np.zeros(3),
        ],
        dtype=float,
    )

    batch = build_lpq_bloch_native_fft_shared_q(
        system,
        basis,
        auxiliary,
        k_bras,
        q,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )
    assert len(batch) == k_bras.shape[0]
    for i in range(k_bras.shape[0]):
        single = build_lpq_bloch_native_fft(
            system,
            basis,
            auxiliary,
            k_bras[i],
            k_bras[i] + q,
            ke_cutoff=20.0,
            lat_opts=options,
            canonical_auxiliary_basis=True,
        )
        np.testing.assert_allclose(batch[i], single, atol=2e-12, rtol=2e-12)


def test_shared_q_batch_matches_per_pair_with_tail_screen_and_cache():
    """Tail completion + Schwarz screen + q-metric cache stay exact.

    A shared metric cache forces the identical eigensystem into both the
    batched and per-pair compact factors, so the SCF-default
    (non-canonical) factors are directly comparable.
    """
    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    k_bras = np.array(
        [
            0.17 * reciprocal[:, 1],
            -0.11 * reciprocal[:, 1],
        ],
        dtype=float,
    )
    metric_cache: dict = {}

    batch = build_lpq_bloch_native_fft_shared_q(
        system,
        basis,
        auxiliary,
        k_bras,
        q,
        ke_cutoff=20.0,
        tail_ke_cutoff=40.0,
        lat_opts=options,
        fit_screen_threshold=1e-10,
        _q_metric_cache=metric_cache,
    )
    assert len(metric_cache) == 1
    for i in range(k_bras.shape[0]):
        single = build_lpq_bloch_native_fft(
            system,
            basis,
            auxiliary,
            k_bras[i],
            k_bras[i] + q,
            ke_cutoff=20.0,
            tail_ke_cutoff=40.0,
            lat_opts=options,
            fit_screen_threshold=1e-10,
            _q_metric_cache=metric_cache,
        )
        np.testing.assert_allclose(batch[i], single, atol=2e-12, rtol=2e-12)
    # The cache is not polluted with additional transfers.
    assert len(metric_cache) == 1


def test_driver_grouping_runs_one_batch_per_unique_transfer(monkeypatch):
    """The driver-side cache build fires one batched call per unique q
    and reproduces the per-pair builds exactly."""
    import vibeqc.aux_basis as aux_mod

    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    # Gamma-centred (1,1,2) mesh: k0 = 0, k1 = b3/2.
    kpoints = np.array(
        [np.zeros(3), 0.5 * reciprocal[:, 2]],
        dtype=float,
    )

    calls: list[int] = []
    real_batch = aux_mod.build_lpq_bloch_native_fft_shared_q

    def _counting_batch(*args, **kwargs):
        calls.append(int(np.asarray(args[3]).reshape(-1, 3).shape[0]))
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(
        aux_mod, "build_lpq_bloch_native_fft_shared_q", _counting_batch
    )

    metric_cache: dict = {}
    cache = _build_rsgdf_lpq_cache_shared_q(
        system,
        basis,
        auxiliary,
        kpoints,
        True,  # need_k_pairs: hybrid exchange wants all (k_i, k_j)
        ke_cutoff=20.0,
        lat_opts=options,
        linear_dep_thr=1e-9,
        fit_screen_threshold=0.0,
        progress=None,
        q_metric_cache=metric_cache,
    )

    # All four pairs present; the number of batched builds equals the
    # number of unique canonical transfers (<= 3 here: q = 0 plus the
    # +/- Nyquist transfer labels), strictly fewer than the 4 pairs.
    assert set(cache.keys()) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert len(calls) == len(metric_cache)
    assert len(calls) < 4
    assert sum(calls) == 4

    # Exactness against per-pair single builds on the same shared
    # metric state.
    for (i, j), lpq in cache.items():
        single = build_lpq_bloch_native_fft(
            system,
            basis,
            auxiliary,
            kpoints[i],
            kpoints[j],
            ke_cutoff=20.0,
            lat_opts=options,
            fit_screen_threshold=0.0,
            _q_metric_cache=metric_cache,
        )
        np.testing.assert_allclose(lpq, single, atol=2e-12, rtol=2e-12)


def test_diagonal_only_build_collapses_to_single_batch(monkeypatch):
    """J-only / COSX-exchange builds (diagonal pairs, all q = 0) share
    ONE pair-FT pass across the whole mesh."""
    import vibeqc.aux_basis as aux_mod

    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    kpoints = np.array(
        [np.zeros(3), 0.5 * reciprocal[:, 2]],
        dtype=float,
    )

    calls: list[int] = []
    real_batch = aux_mod.build_lpq_bloch_native_fft_shared_q

    def _counting_batch(*args, **kwargs):
        calls.append(int(np.asarray(args[3]).reshape(-1, 3).shape[0]))
        return real_batch(*args, **kwargs)

    monkeypatch.setattr(
        aux_mod, "build_lpq_bloch_native_fft_shared_q", _counting_batch
    )

    cache = _build_rsgdf_lpq_cache_shared_q(
        system,
        basis,
        auxiliary,
        kpoints,
        False,  # diagonal-only
        ke_cutoff=20.0,
        lat_opts=options,
        linear_dep_thr=1e-9,
        fit_screen_threshold=0.0,
        progress=None,
        q_metric_cache={},
    )

    assert set(cache.keys()) == {(0, 0), (1, 1)}
    assert calls == [2]


def test_batch_empty_and_single_edge_cases():
    """n_k = 0 returns []; n_k = 1 equals the single-pair build."""
    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0

    assert (
        build_lpq_bloch_native_fft_shared_q(
            system,
            basis,
            auxiliary,
            np.zeros((0, 3)),
            q,
            ke_cutoff=20.0,
            lat_opts=options,
        )
        == []
    )

    k_bra = 0.17 * reciprocal[:, 1]
    (one,) = build_lpq_bloch_native_fft_shared_q(
        system,
        basis,
        auxiliary,
        k_bra.reshape(1, 3),
        q,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )
    single = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        k_bra,
        k_bra + q,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )
    np.testing.assert_allclose(one, single, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("backend", ["python"])
def test_multi_wrapper_python_fallback_matches(monkeypatch, backend):
    """The pure-Python fallback of the batched wrapper stays exact."""
    system, basis, reciprocal = _skew_lih_fixture()
    rng = np.random.default_rng(11)
    G = rng.normal(size=(9, 3))
    cells = np.array(
        [[0.0, 0.0, 0.0], np.asarray(system.lattice, dtype=float)[:, 2]],
        dtype=float,
    )
    k_carts = np.array(
        [np.zeros(3), 0.5 * reciprocal[:, 2]],
        dtype=float,
    )
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", backend)
    batch = ao_pair_fourier_transform_bloch_multi(basis, G, cells, k_carts)
    monkeypatch.delenv("VIBEQC_AOPAIR_FT_BACKEND")
    for ik in range(k_carts.shape[0]):
        single = ao_pair_fourier_transform_bloch(
            basis, G, cells, k_carts[ik]
        )
        np.testing.assert_allclose(batch[ik], single, atol=1e-12, rtol=1e-12)


def test_batched_kernel_is_g_block_decomposition_invariant():
    """The batched kernel splits its work over (shell pair, G-block).

    A G-block is an exclusively-owned slice of the output tensor, so the
    decomposition must not be observable: the result has to be identical
    whatever block count the thread pool induces, and identical to
    evaluating each G-slice in a separate call. The kernel's cell loop is
    a Bloch *sum* and stays serial within a task -- only the G axis is
    partitioned -- so this also pins that no cross-block accumulation
    crept in.
    """
    system, basis, reciprocal = _skew_lih_fixture()
    lat = np.asarray(system.lattice, dtype=float)
    cells = np.array(
        [[0.0, 0.0, 0.0], lat[:, 0], -lat[:, 0], lat[:, 1], lat[:, 2]],
        dtype=float,
    )
    rng = np.random.default_rng(23)
    G = np.ascontiguousarray(rng.normal(size=(311, 3)) * 1.4)
    k_carts = np.array(
        [np.zeros(3), 0.5 * reciprocal[:, 2], 0.13 * reciprocal[:, 0]],
        dtype=float,
    )

    whole = ao_pair_fourier_transform_bloch_multi(basis, G, cells, k_carts)

    # Same content assembled from independent G-slices.
    cuts = [0, 1, 7, 64, 200, G.shape[0]]
    for ik in range(k_carts.shape[0]):
        rebuilt = np.empty_like(whole[ik])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            part = ao_pair_fourier_transform_bloch_multi(
                basis, np.ascontiguousarray(G[lo:hi]), cells, k_carts
            )
            rebuilt[:, :, lo:hi] = part[ik]
        np.testing.assert_allclose(
            rebuilt, whole[ik], atol=1e-13, rtol=1e-11
        )


def test_single_k_kernel_is_bit_identical_across_g_block_counts():
    """The single-k kernel's (shell pair, G-block) split must be exactly
    invisible, not merely reassociation-equivalent.

    Every output element's accumulation -- over cells, primitive pairs and
    Hermite terms -- stays wholly inside one task and in the original
    order; only *which* worker owns a given G is decided by the block
    partition. So unlike the batched kernel (which reorders the Bloch
    phase fold and therefore matches only to rounding), this one must
    agree bit-for-bit however the blocks fall.

    The block count is derived from the OpenMP worker count, so this
    compares results produced under different thread counts in separate
    interpreters.
    """
    import os
    import pickle
    import subprocess
    import sys

    program = """
import pickle, sys
import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import LatticeSumOptions, direct_lattice_cells
from vibeqc.aux_basis import rsgdf_dense_g_mesh
from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch

a = 7.72
lattice = 0.5 * a * np.array([[0., 1, 1], [1, 0, 1], [1, 1, 0]]).T
system = vq.PeriodicSystem(
    3, lattice, [vq.Atom(3, [0., 0, 0]), vq.Atom(1, [0.5 * a, 0, 0])]
)
basis = vq.BasisSet(system.unit_cell_molecule(), "6-31g")
cells = direct_lattice_cells(system, LatticeSumOptions().cutoff_bohr)
R = np.array([list(c.r_cart) for c in cells], dtype=float)
G = rsgdf_dense_g_mesh(system, 40.0)
G = np.ascontiguousarray(G[(G ** 2).sum(axis=1) > 0])
out = ao_pair_fourier_transform_bloch(
    basis, G, R, k_cart=np.array([0.05, 0.0, 0.03])
)
sys.stdout.buffer.write(pickle.dumps(out))
"""

    results = {}
    for threads in ("1", "3"):
        env = dict(os.environ, OMP_NUM_THREADS=threads)
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, env=env, timeout=900,
        )
        assert proc.returncode == 0, proc.stderr.decode()[-2000:]
        results[threads] = pickle.loads(proc.stdout)

    serial = results["1"]
    for threads, value in results.items():
        assert np.array_equal(value, serial), (
            f"OMP_NUM_THREADS={threads} differs from serial by "
            f"{np.max(np.abs(value - serial)):.3e}; the single-k G-block "
            f"split must be bit-identical"
        )


# --- aux-FT cache budget ----------------------------------------------------
# Sweep A's per-chunk aux FT is reused by Sweep B rather than recomputed.
# That cache is the one part of the build whose size does NOT follow
# tail_chunk_g -- it grows as (n_aux, n_base) -- so unbounded it becomes a
# floor under the peak that the chunking knob cannot reach. It is budgeted;
# these pin that the budget bounds it AND that caching never changes a value.


def test_aux_ft_cache_budget_does_not_change_results(monkeypatch):
    """Cached or recomputed, the fit is bit-identical: the cache removes
    duplicate work, it does not reassociate anything."""
    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    k_bras = np.array(
        [np.zeros(3), 0.3 * reciprocal[:, 1], -0.2 * reciprocal[:, 2]],
        dtype=float,
    )

    def build():
        return build_lpq_bloch_native_fft_shared_q(
            system, basis, auxiliary, k_bras, q,
            ke_cutoff=40.0, lat_opts=options, tail_chunk_g=512,
            canonical_auxiliary_basis=True, _q_metric_cache={},
        )

    monkeypatch.setenv("VIBEQC_GDF_AUX_FT_CACHE_MIB", "256")
    cached = build()
    monkeypatch.setenv("VIBEQC_GDF_AUX_FT_CACHE_MIB", "0")
    uncached = build()

    assert len(cached) == len(uncached) == k_bras.shape[0]
    for got, want in zip(cached, uncached):
        # Bit-identical, not merely close.
        assert np.array_equal(got, want), float(np.max(np.abs(got - want)))


def test_aux_ft_cache_budget_limits_retained_chunks(monkeypatch):
    """A budget that cannot hold every chunk must make Sweep B recompute
    the rest, rather than caching them anyway.

    Observable through the aux-FT call count: with a generous budget
    Sweep B adds none, with the cache disabled it repeats all of Sweep
    A's.
    """
    import vibeqc.aux_basis as aux_mod

    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    k_bras = np.array([np.zeros(3), 0.3 * reciprocal[:, 1]], dtype=float)

    real_aux_ft = aux_mod.rsgdf_aux_fourier_transform

    def counted_build(budget_mib):
        calls: list[int] = []

        def _counting(basis_arg, G):
            calls.append(int(np.asarray(G).shape[0]))
            return real_aux_ft(basis_arg, G)

        monkeypatch.setattr(
            aux_mod, "rsgdf_aux_fourier_transform", _counting
        )
        monkeypatch.setenv("VIBEQC_GDF_AUX_FT_CACHE_MIB", str(budget_mib))
        build_lpq_bloch_native_fft_shared_q(
            system, basis, auxiliary, k_bras, q,
            ke_cutoff=40.0, lat_opts=options, tail_chunk_g=256,
            _q_metric_cache={},
        )
        monkeypatch.setattr(
            aux_mod, "rsgdf_aux_fourier_transform", real_aux_ft
        )
        return len(calls)

    generous = counted_build(256)
    disabled = counted_build(0)

    # Sweep A alone accounts for the generous count; disabling the cache
    # makes Sweep B repeat every one of them.
    assert disabled > generous, (generous, disabled)
    assert disabled == pytest.approx(2 * generous, rel=0.05), (
        generous, disabled
    )


def test_aux_ft_cache_budget_handles_a_bad_env_value(monkeypatch):
    """A malformed budget must fall back to the default, not crash a
    calculation on an env typo."""
    from vibeqc.aux_basis import _rsgdf_aux_ft_cache_budget_bytes

    monkeypatch.setenv("VIBEQC_GDF_AUX_FT_CACHE_MIB", "not-a-number")
    assert _rsgdf_aux_ft_cache_budget_bytes() == 256 * 1024 * 1024
    monkeypatch.setenv("VIBEQC_GDF_AUX_FT_CACHE_MIB", "-5")
    assert _rsgdf_aux_ft_cache_budget_bytes() == 0


# --- Fused per-AO-pair store weight ------------------------------------
#
# The RSGDF accumulator used to scale the batched pair FT in NumPy
# (`pair_ft_c *= pair_scales[:, :, None]`, then a masked assignment for
# the fit screen). Those were single-threaded passes over the largest
# array in the cderi build, and they anti-scaled against the threads the
# kernel itself uses -- 0.537 s at one thread against 0.671 s at sixteen
# (measured 2026-08-05; see handovers/HANDOVER_GDF_OUTSTANDING.md, which
# also records that the profile's larger 4.428 s figure is the whole of
# `_accumulate_chunk`'s own time, NOT these passes). The kernel now
# applies the weight as it stores, so neither pass exists.
#
# That is only a legitimate move if it changes NOTHING, so these pin the
# equivalence BIT-FOR-BIT rather than to a tolerance: a cderi that
# differs in the last ulp is a different cderi.


def _bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Exact bit comparison, so ±0.0 and any last-ulp drift show up."""
    a = np.ascontiguousarray(a, dtype=np.complex128)
    b = np.ascontiguousarray(b, dtype=np.complex128)
    return a.shape == b.shape and np.array_equal(
        a.view(np.uint64), b.view(np.uint64)
    )


def _weight_fixture():
    system, basis, reciprocal = _skew_lih_fixture()
    lat = np.asarray(system.lattice, dtype=float)
    cells = np.array(
        [[0.0, 0.0, 0.0], lat[:, 0], -lat[:, 0], lat[:, 1], lat[:, 2]],
        dtype=float,
    )
    rng = np.random.default_rng(101)
    G = np.ascontiguousarray(rng.normal(size=(53, 3)) * 1.5)
    k_carts = np.array(
        [np.zeros(3), 0.5 * reciprocal[:, 2], 0.13 * reciprocal[:, 0]],
        dtype=float,
    )
    return basis, G, cells, k_carts


def test_pair_weights_match_scaling_the_result_bit_for_bit():
    """A weighted store == the NumPy multiply it replaces, exactly.

    NumPy casts the real factor to complex and runs the full complex
    product, so plain componentwise scaling is NOT the same operation:
    the two disagree in the sign of zero. This is the assertion that
    keeps the C++ store honest about that.
    """
    basis, G, cells, k_carts = _weight_fixture()
    n_orb = int(basis.nbasis)
    rng = np.random.default_rng(3)
    weights = np.ascontiguousarray(rng.uniform(0.3, 2.5, (n_orb, n_orb)))

    plain = ao_pair_fourier_transform_bloch_multi(basis, G, cells, k_carts)
    fused = ao_pair_fourier_transform_bloch_multi(
        basis, G, cells, k_carts, pair_weights=weights
    )

    assert len(fused) == len(plain) == k_carts.shape[0]
    for ik in range(k_carts.shape[0]):
        reference = plain[ik].copy()
        reference *= weights[:, :, None]
        assert _bitwise_equal(fused[ik], reference)


def test_pair_weights_of_zero_store_a_true_positive_zero():
    """An exactly-zero weight means "masked pair", and a masked pair is
    an ASSIGNMENT of +0.0 -- not a multiply by zero, which would carry
    the operand's sign of zero through.
    """
    basis, G, cells, k_carts = _weight_fixture()
    n_orb = int(basis.nbasis)
    ao_scales = np.asarray(
        [1.0 + 0.1 * i for i in range(n_orb)], dtype=float
    )
    weights = np.outer(ao_scales, ao_scales)
    keep = np.ones((n_orb, n_orb), dtype=bool)
    keep[0, :] = False
    keep[:, n_orb - 1] = False
    masked_weights = np.where(keep, weights, 0.0)

    plain = ao_pair_fourier_transform_bloch_multi(basis, G, cells, k_carts)
    fused = ao_pair_fourier_transform_bloch_multi(
        basis, G, cells, k_carts, pair_weights=masked_weights
    )

    for ik in range(k_carts.shape[0]):
        reference = plain[ik].copy()
        reference *= weights[:, :, None]
        reference[~keep] = 0.0
        assert _bitwise_equal(fused[ik], reference)
        # Not merely equal-valued: a genuine +0.0, sign bit clear.
        # -0.0 compares == 0.0, so this has to look at the bits.
        zeros = np.ascontiguousarray(fused[ik][~keep]).ravel()
        assert np.array_equal(
            zeros.view(np.uint64),
            np.zeros(zeros.size * 2, dtype=np.uint64),
        )


def test_pair_weights_omitted_leaves_the_kernel_byte_unchanged():
    """The default path must be exactly the pre-existing kernel."""
    basis, G, cells, k_carts = _weight_fixture()
    n_orb = int(basis.nbasis)
    unweighted = ao_pair_fourier_transform_bloch_multi(
        basis, G, cells, k_carts
    )
    ones = ao_pair_fourier_transform_bloch_multi(
        basis, G, cells, k_carts,
        pair_weights=np.ones((n_orb, n_orb), dtype=float),
    )
    for ik in range(k_carts.shape[0]):
        assert _bitwise_equal(unweighted[ik], ones[ik])


def test_pair_weights_rejects_a_wrong_shape():
    basis, G, cells, k_carts = _weight_fixture()
    n_orb = int(basis.nbasis)
    with pytest.raises(ValueError, match="pair_weights must be"):
        ao_pair_fourier_transform_bloch_multi(
            basis, G, cells, k_carts,
            pair_weights=np.ones((n_orb, n_orb + 1), dtype=float),
        )


@pytest.mark.parametrize("backend", ["python", "cxx"])
def test_pair_weights_agree_across_backends(monkeypatch, backend):
    """The pure-Python reference path carries the same contract."""
    basis, G, cells, k_carts = _weight_fixture()
    n_orb = int(basis.nbasis)
    rng = np.random.default_rng(11)
    weights = np.ascontiguousarray(rng.uniform(0.3, 2.5, (n_orb, n_orb)))
    weights[2, 1] = 0.0

    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", backend)
    plain = ao_pair_fourier_transform_bloch_multi(basis, G, cells, k_carts)
    fused = ao_pair_fourier_transform_bloch_multi(
        basis, G, cells, k_carts, pair_weights=weights
    )
    for ik in range(k_carts.shape[0]):
        reference = plain[ik].copy()
        reference *= weights[:, :, None]
        reference[weights == 0.0] = 0.0
        assert _bitwise_equal(fused[ik], reference)


@pytest.mark.parametrize(
    "fit_pair_list",
    [None, [[0, 0], [1, 1]]],
    ids=["no-screen", "masked-pairs"],
)
def test_shared_q_cderi_is_bit_identical_to_unfused_scaling(
    monkeypatch, fit_pair_list
):
    """End-to-end: the Lpq the accumulator produces must not move.

    Rebuilds the batch with the fused store disabled and the old NumPy
    scaling done in the wrapper instead, and requires the two cderi
    tensors to be bit-for-bit equal. The ``masked-pairs`` case pins the
    fit-screen branch, where a zero weight replaces what used to be a
    ``pair_ft_c[~pair_keep_ao] = 0.0`` assignment.
    """
    from vibeqc import _aopair_ft

    system, basis, auxiliary, reciprocal, options = _skew_h2_fixture()
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    k_bras = np.array(
        [0.17 * reciprocal[:, 1], -0.11 * reciprocal[:, 1]], dtype=float
    )
    pairs = (
        None if fit_pair_list is None
        else np.asarray(fit_pair_list, dtype=int)
    )

    def build():
        return build_lpq_bloch_native_fft_shared_q(
            system, basis, auxiliary, k_bras, q,
            ke_cutoff=40.0, tail_ke_cutoff=60.0, lat_opts=options,
            tail_chunk_g=128, fit_pair_list=pairs,
            _q_metric_cache={},
        )

    real_multi = _aopair_ft.ao_pair_fourier_transform_bloch_multi
    saw_mask = []

    def unfused(ao_basis, G, R_g, k_carts, *, pair_weights=None, **kw):
        """The pre-2026-08-05 shape: kernel unweighted, scaling in NumPy."""
        out = real_multi(ao_basis, G, R_g, k_carts, **kw)
        if pair_weights is not None:
            masked = pair_weights == 0.0
            saw_mask.append(bool(masked.any()))
            for tensor in out:
                tensor *= pair_weights[:, :, None]
                if masked.any():
                    tensor[masked] = 0.0
        return out

    fused_lpq = build()
    # The builder imports the symbol inside its body, so patching the
    # module it imports FROM is what reaches the call site.
    monkeypatch.setattr(
        _aopair_ft, "ao_pair_fourier_transform_bloch_multi", unfused
    )
    unfused_lpq = build()

    assert saw_mask and all(
        seen == (pairs is not None) for seen in saw_mask
    ), "the fit-screen branch under test did not run as intended"
    assert len(fused_lpq) == len(unfused_lpq) == k_bras.shape[0]
    for got, want in zip(fused_lpq, unfused_lpq):
        assert _bitwise_equal(got, want), "cderi moved"


def test_range_separated_cache_planned_domains_preserve_jk_across_omega():
    from vibeqc.aux_basis import (
        _build_coulomb_from_diagonal_factors, _plan_range_separated_gdf_cutoffs,
    )
    from vibeqc.periodic_k_gdf import (
        _build_k_from_lpq_cache, _build_range_separated_lpq_cache,
    )

    system, basis, auxiliary = _range_separated_fit_fixture()
    reciprocal = np.asarray(system.reciprocal_lattice())
    # Unequal spacings exercise the smallest nonzero q as well as q=0.
    kpoints = np.array([[0., 0., 0.], [.07, 0., 0.], [.24, .13, 0.]]) @ reciprocal.T
    weights = np.array([.2, .3, .5])
    densities = [np.array([[1., .2 + .1j], [.2 - .1j, .8]]) for _ in weights]
    operators = []
    for omega in (.5, .8):
        options = _range_separated_fit_options()
        for key in ('pair_cutoff', 'auxiliary_cutoff', 'ke_cutoff'):
            options.pop(key)
        options.update(omega=omega, raw_integral_error=1e-9)
        cache = _build_range_separated_lpq_cache(
            system, basis, auxiliary, kpoints, True, **options,
        )
        chosen = cache.source_parameters[1:4]
        for bra in kpoints:
            for ket in kpoints:
                plan = _plan_range_separated_gdf_cutoffs(
                    system, basis, auxiliary, ket - bra,
                    omega=omega, raw_integral_error=1e-9,
                )
                assert all(actual >= required for actual, required in zip(chosen, plan[:3]))
        assert cache.reserved_peak_bytes <= options['memory_byte_cap']
        operators.append((
            _build_coulomb_from_diagonal_factors(
                [cache[i, i] for i in range(3)], densities, weights,
            ),
            _build_k_from_lpq_cache(cache, densities, weights, nbasis=basis.nbasis),
        ))
    np.testing.assert_allclose(operators[0], operators[1], rtol=0, atol=1e-8)


def test_range_separated_cache_admits_retention_before_planning(monkeypatch):
    from types import SimpleNamespace
    import vibeqc.aux_basis as auxiliary_module
    from vibeqc.periodic_k_gdf import _build_range_separated_lpq_cache

    def unplanned(*args, **kwargs):
        pytest.fail("cutoff planning ran before retained-cache admission")
    monkeypatch.setattr(auxiliary_module, '_plan_range_separated_gdf_cutoffs', unplanned)
    with pytest.raises(MemoryError, match='retained factor cache'):
        _build_range_separated_lpq_cache(
            None, SimpleNamespace(nbasis=1000), SimpleNamespace(nbasis=3000),
            np.zeros((8, 3)), True, omega=.5, raw_integral_error=1e-9,
            linear_dep_thr=1e-9, memory_byte_cap=2**20,
            native_workspace_byte_cap=2**20, image_candidate_cap=1000,
            reciprocal_candidate_cap=1000,
        )


def test_production_gamma_and_general_spin_paths_share_sr_lr_source(monkeypatch):
    import vibeqc.aux_basis as auxiliary_module
    import vibeqc.pbc_gdf as gamma_module
    import vibeqc.periodic_k_gdf as k_module

    system, basis, auxiliary = _range_separated_fit_fixture()
    monkeypatch.setattr(auxiliary_module, 'make_aux_basis_set', lambda *a, **kw: auxiliary)
    monkeypatch.setattr(gamma_module, 'make_aux_basis_set', lambda *a, **kw: auxiliary)
    monkeypatch.setattr(k_module, 'make_aux_basis_set', lambda *a, **kw: auxiliary)
    def reciprocal_fit_forbidden(*args, **kwargs):
        pytest.fail('production bulk fit entered an all-reciprocal builder')
    monkeypatch.setattr(gamma_module, 'build_lpq_native_fft', reciprocal_fit_forbidden)
    monkeypatch.setattr(k_module, '_build_rsgdf_lpq_cache_shared_q', reciprocal_fit_forbidden)
    source = k_module._build_scf_range_separated_lpq_cache
    caches = []
    def observed(*args, **kwargs):
        cache = source(*args, **kwargs)
        caches.append(cache)
        return cache
    monkeypatch.setattr(k_module, '_build_scf_range_separated_lpq_cache', observed)
    options = vq.PeriodicRHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    options.max_iter = 80
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-8
    options.lattice_opts.cutoff_bohr = 12.0
    options.lattice_opts.nuclear_cutoff_bohr = 18.0
    energies = []
    for omega in (.45, .7):
        controls = dict(options=options, gdf_method='rsgdf', aux_basis='test',
                        rsgdf_omega=omega, rsgdf_g_precision=1e-10,
                        rsgdf_ke_cutoff=20.0, progress=False)
        gamma = gamma_module.run_pbc_gdf_rhf(system, basis, **controls)
        restricted = k_module.run_krhf_periodic_gdf(system, basis, **controls)
        unrestricted = k_module.run_kuhf_periodic_gdf(system, basis, **controls)
        assert gamma.converged and restricted.converged and unrestricted.converged
        assert gamma.rsgdf_tail_ke_cutoff is None
        assert restricted.rsgdf_tail_ke_cutoff is None
        assert unrestricted.rsgdf_tail_ke_cutoff is None
        np.testing.assert_allclose(
            [restricted.energy, unrestricted.energy], gamma.energy, atol=2e-8, rtol=0,
        )
        energies.append(gamma.energy)
    assert len(caches) == 6
    assert all(cache.aux_basis is auxiliary for cache in caches)
    np.testing.assert_allclose(energies[0], energies[1], atol=2e-8, rtol=0)


@pytest.mark.slow
@pytest.mark.parametrize('omega', [.6, .8])
def test_dense_silicon_fixed_density_jk_matches_independent_reference(omega):
    """#81/#662: tight cores use SR integrals, with the same bare/Ewald gauge."""
    import hashlib
    import json
    from pathlib import Path
    from vibeqc import _vibeqc_core as core
    from vibeqc.aux_basis import _build_coulomb_from_diagonal_factors
    from vibeqc.periodic_k_gdf import (
        _build_range_separated_lpq_cache, _build_k_from_lpq_cache,
        _madelung_for_kmesh, apply_exxdiv_ewald_to_K,
    )

    root = Path(__file__).parent / 'data/periodic_gdf'
    record = json.loads((root / 'si_gamma_fixed_density.json').read_text())
    path = root / 'si_gamma_fixed_density.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['provenance']['fixture_sha256']
    specification = record['input']
    system = vq.PeriodicSystem(3, np.asarray(specification['lattice']).T, [
        vq.Atom(14, center) for _, center in specification['atoms']
    ])

    def basis_from_records(key):
        shells, permutation, offset = [], [], 0
        for atom, (symbol, center) in enumerate(specification['atoms']):
            for shell in specification[key][symbol]:
                angular = int(shell[0])
                primitives = np.asarray(shell[1:])
                for contraction in range(1, primitives.shape[1]):
                    shells.append(core.ShellInfo(
                        atom, angular, True, primitives[:, 0].tolist(),
                        primitives[:, contraction].tolist(), center,
                    ))
                    # Independent programs use different pure-p AO order.
                    order = [1, 2, 0] if angular == 1 else list(range(2*angular+1))
                    permutation.extend(offset + i for i in order)
                    offset += len(order)
        return core.BasisSet(system.unit_cell_molecule(), shells, key, False), permutation

    basis, permutation = basis_from_records('basis')
    auxiliary, _ = basis_from_records('auxiliary')
    assert basis.nbasis == 44 and auxiliary.nbasis == 256
    with np.load(path, allow_pickle=False) as stored:
        reference = {k: stored[k] for k in stored.files}
    ordered = lambda a: np.asarray(a)[:, permutation, :][:, :, permutation]
    densities = ordered(reference['D'])
    kpoints = reference['kpoints']
    options = core.LatticeSumOptions()
    options.cutoff_bohr = 25.
    overlap = np.asarray([core.bloch_sum(
        core.compute_overlap_lattice(basis, system, options), np.zeros(3),
    )])
    np.testing.assert_allclose(overlap, ordered(reference['S']), atol=2e-8, rtol=0)
    assert np.trace(densities[0] @ overlap[0]).real == pytest.approx(28., abs=2e-8)
    workspace = max(64*1024**2, core.gdf_short_range_workspace_bytes(
        basis, auxiliary, core.get_num_threads(),
    ))
    cache = _build_range_separated_lpq_cache(
        system, basis, auxiliary, kpoints, True,
        omega=omega, raw_integral_error=1e-10, linear_dep_thr=1e-9,
        memory_byte_cap=8*1024**3, native_workspace_byte_cap=workspace,
        image_candidate_cap=10_000_000, reciprocal_candidate_cap=10_000_000,
    )
    assert cache.source_parameters[3] < 200.  # no exponent-sized reciprocal tail
    weights = np.ones(1)
    j = np.asarray(_build_coulomb_from_diagonal_factors(
        [cache.factors[0, 0]], densities, weights,
    ))
    k = np.asarray(_build_k_from_lpq_cache(cache.factors, densities, weights, nbasis=44))
    corrected = np.asarray(apply_exxdiv_ewald_to_K(
        k, overlap, densities, _madelung_for_kmesh(system, (1, 1, 1)),
    ))
    for name, actual, factor in [('J', j, .5), ('K', k, -.25), ('K_ewald', corrected, -.25)]:
        expected = ordered(reference[name+'_0.6'])
        np.testing.assert_allclose(actual, expected, atol=3e-8, rtol=0)
        energy_error = factor * np.trace(densities[0] @ (actual[0]-expected[0])).real
        assert abs(energy_error) < 3e-7
