"""Bounded source/NumPy contracts, not native fitted-HF numerical acceptance.

Extract only the pure array helpers so these checks do not import a native core.
End-to-end fitted RHF/UHF acceptance remains a separately recorded native run.
"""
import ast
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def kernels():
    source = Path(__file__).parents[1] / 'python/vibeqc/periodic_k_gdf.py'
    tree = ast.parse(source.read_text())
    names = {'_chi_hf_termination_reason', '_chi_validate_shift_controls', '_chi_require_integer_occupations', '_chi_trial_fock', '_chi_rebuild_hf_acceptance'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    scope = {'np': np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), scope)
    return scope


def rebuild(kernels, da, db, *, counts=(1, 1), reported=-3.0, delta=0.0,
            overlap=None, orthogonalizer=None, hcore=None, madelung=0.0,
            build_j=None, build_k=None):
    n = da.shape[0]
    s = np.eye(n) if overlap is None else overlap
    x = np.eye(n) if orthogonalizer is None else orthogonalizer
    h = np.diag([-1.5, 0.7]) if hcore is None else hcore
    zero = lambda density: [np.zeros_like(d) for d in density]
    return kernels['_chi_rebuild_hf_acceptance'](
        [h], [s], [x], [1.0], [[da], [db]], counts,
        build_j or zero, build_k or zero, madelung, 0., reported, delta, 1e-10, 1e-8)


def test_rebuilt_closed_shell_and_open_shell_spin_counts(kernels):
    d = np.diag([1., 0.])
    closed, f = rebuild(kernels, d, d)
    assert closed['accepted'] and closed['final_rebuild']
    assert closed['electron_counts'] == [1., 1.]
    assert closed['stability_status'] == 'not-assessed'
    np.testing.assert_array_equal(f[0], f[1])
    opened, _ = rebuild(kernels, d, np.zeros_like(d), counts=(1, 0), reported=-1.5)
    assert opened['accepted'] and opened['electron_counts'] == [1., 0.]


def test_stationary_wrong_occupied_subspace_is_not_accepted(kernels):
    wrong = np.diag([0., 1.])
    record, _ = rebuild(kernels, wrong, wrong, reported=1.4)
    assert record['orbital_residual'] == 0
    assert record['density_refill_residual'] > 1
    assert not record['accepted']


@pytest.mark.parametrize('change', ['charge', 'energy', 'delta', 'offdiagonal'])
def test_rebuild_checks_charge_energy_and_density(kernels, change):
    d = np.diag([1., 0.])
    kw = {}
    if change == 'charge':
        d *= .9
        kw['reported'] = -2.7
    elif change == 'energy':
        kw['reported'] = -2.8
    elif change == 'delta':
        kw['delta'] = 1e-5
    else:
        d = np.array([[.5, .5], [.5, .5]])
        kw['reported'] = -.8
    record, _ = rebuild(kernels, d, d, **kw)
    assert not record['accepted']


def test_nonorthogonal_density_and_bvk_shift_are_consistent(kernels):
    s = np.diag([2., .5])
    x = np.diag(1 / np.sqrt(np.diag(s)))
    d = np.diag([.5, 0.])
    # Each spin: Tr(D H)=-1.5; half Tr(D (-xi S D S))=-.1.
    report, fock = rebuild(kernels, d, d, overlap=s, orthogonalizer=x,
                          hcore=np.diag([-3., .35]), madelung=.2, reported=-3.2)
    assert report['accepted']
    np.testing.assert_allclose(fock[0][0], np.diag([-3.4, .35]))


def test_rebuild_uses_total_j_and_each_spin_k_once(kernels):
    da, db = np.diag([1., 0.]), np.zeros((2, 2))
    calls = []
    def j(d):
        calls.append(('J', np.array(d)))
        return [np.eye(2) * .3]
    def k(d):
        calls.append(('K', np.array(d)))
        return [np.asarray(d[0]) * .2]
    record, focks = rebuild(kernels, da, db, counts=(1, 0), reported=-1.45, build_j=j, build_k=k)
    assert record['accepted']
    assert [c[0] for c in calls] == ['J', 'K', 'K']
    np.testing.assert_array_equal(calls[0][1], [da + db])
    np.testing.assert_array_equal(calls[1][1], [da])
    np.testing.assert_array_equal(calls[2][1], [db])
    assert focks[0][0][0, 0] == pytest.approx(-1.4)
    assert focks[1][0][0, 0] == pytest.approx(-1.2)


def test_trial_mixing_and_spin_virtual_shift_do_not_change_input_or_history(kernels):
    f = np.diag([-1., 2.]).astype(complex)
    old = np.diag([-2., 1.]).astype(complex)
    d = np.diag([1., 0.])
    trial, history = kernels['_chi_trial_fock']([f], [old], [d], [np.eye(2)], .25, .4)
    np.testing.assert_allclose(history[0], np.diag([-1.25, 1.75]))
    np.testing.assert_allclose(trial[0], np.diag([-1.25, 2.15]))
    np.testing.assert_array_equal(f, np.diag([-1., 2.]))
    np.testing.assert_array_equal(old, np.diag([-2., 1.]))
    trial[0][0, 0] = 99
    assert history[0][0, 0] == -1.25


@pytest.mark.parametrize('maximum', [1., 2.])
def test_fractional_and_variable_k_occupations_refused(kernels, maximum):
    guard = kernels['_chi_require_integer_occupations']
    guard([np.array([maximum, 0.])] * 2, 1, maximum)
    with pytest.raises(NotImplementedError, match='integer Aufbau'):
        guard([np.array([maximum / 2, maximum / 2])], 1, maximum)
    with pytest.raises(NotImplementedError, match='integer Aufbau'):
        guard([np.array([maximum, maximum]), np.array([0., 0.])], 1, maximum)
    guard([np.zeros(2)], 0, maximum)


@pytest.mark.parametrize('driver', ['run_krhf_periodic_gdf', 'run_kuhf_periodic_gdf'])
@pytest.mark.parametrize('source', ['classic', 'sr'])
def test_both_consumers_build_exact_source_and_reserved_support(driver, source):
    """Execute the actual finite-source dispatch branch with a recording builder."""
    import builtins
    from types import SimpleNamespace
    path = Path(__file__).parents[1] / 'python/vibeqc/periodic_k_gdf.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == driver)
    branch = next(n for n in ast.walk(function) if isinstance(n, ast.If)
                  and any(isinstance(child, ast.ImportFrom)
                          and child.module == 'periodic_fitted_support'
                          and any(alias.name == 'build_chi_fitted_operator' for alias in child.names)
                          for child in n.body))
    calls = []
    cache = object()
    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return cache
    def importer(name, *args, **kwargs):
        if name == 'periodic_fitted_support':
            return SimpleNamespace(build_chi_fitted_operator=build)
        raise AssertionError(f'unexpected source import: {name}')
    scope = dict(system=object(), basis=object(), aux=object(), aux_modrho=object(),
                 kpoints_cart=np.zeros((1, 3)), weights=np.ones(1),
                 fitted_support=object(), fitted_oneel_reserved_bytes=1234,
                 selected_fit_source=source, bulk_sr=source == 'sr')
    scope['__builtins__'] = dict(vars(builtins), __import__=importer)
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), 'exec'), scope)
    assert scope['lpq_cache'] is cache
    assert len(calls) == 1
    args, kw = calls[0]
    assert args[0] is scope['system'] and args[1] is scope['basis']
    assert args[2] is scope['aux' if source == 'sr' else 'aux_modrho']
    assert args[3] is scope['kpoints_cart'] and args[4] is scope['weights']
    assert kw == dict(source=source, controls=scope['fitted_support'], _retained_bytes=1234)


def test_complex_multik_refill_uses_bz_weights(kernels):
    u = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2.)
    d0 = np.diag([1., 0.]).astype(complex)
    d1 = u @ d0 @ u.conj().T
    h0 = np.diag([-2., 1.])
    h1 = u @ np.diag([-1., 2.]) @ u.conj().T
    zero = lambda density: [np.zeros_like(d) for d in density]
    report, _ = kernels['_chi_rebuild_hf_acceptance'](
        [h0, h1], [np.eye(2)] * 2, [np.eye(2)] * 2, [.25, .75],
        [[d0, d1], [d0, d1]], (1, 1), zero, zero, 0., .1, -2.4, 0., 1e-10, 1e-8)
    assert report['accepted']
    np.testing.assert_allclose(report['electron_counts'], [1., 1.])
    assert report['rebuilt_energy'] == pytest.approx(-2.4)


@pytest.mark.parametrize('static,schedule', [(float('nan'), []), (-.1, []),
    (0., [float('inf')]), (0., [-.2]), (0., ['invalid'])])
def test_invalid_shift_schedule_refused_before_integrals(kernels, static, schedule):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='finite and nonnegative'):
        kernels['_chi_validate_shift_controls'](SimpleNamespace(
            level_shift=static, level_shift_schedule=schedule))


def test_valid_shift_schedule_is_not_mutated(kernels):
    from types import SimpleNamespace
    opts = SimpleNamespace(level_shift=.2, level_shift_schedule=[.4, .1, 0.])
    kernels['_chi_validate_shift_controls'](opts)
    assert opts.level_shift_schedule == [.4, .1, 0.]


def test_stagnation_charge_error_and_exhaustion_are_distinct(kernels):
    d = np.diag([1., 0.])
    report, _ = rebuild(kernels, d, d)
    reason = kernels['_chi_hf_termination_reason']
    assert reason(report, True, 3, 20) == 'converged'
    report['density_residual'] = .2
    assert reason(report, False, 20, 20) == 'stagnation'
    report['electron_count_errors'] = [.1, 0.]
    assert reason(report, False, 20, 20) == 'electron_count_mismatch'
    report['electron_count_errors'] = [0., 0.]
    report['energy_residual'] = .1
    assert reason(report, False, 20, 20) == 'iteration_limit'
