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
@pytest.mark.parametrize('spectra', [
    [[-2., 0., 0., 0., 1.]],
    [[-3., -2., -1., 2.], [-3., 1., 2., 3.], [-3., 1., 2., 3.]],
])
def test_chi_determinant_trials_preserve_final_global_occupation_guard(driver, spectra):
    """A degenerate seed may evolve, but an ensemble/metal cannot be accepted.

    Exercise the driver's actual occupation closures with its real global
    filling routines. Native SCF/operator acceptance is recorded separately.
    """
    from typing import List
    from vibeqc import periodic_k_gdf as gdf
    from vibeqc.smearing import SmearingOptions

    tree = ast.parse(Path(gdf.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == driver)
    restricted = driver == 'run_krhf_periodic_gdf'
    names = {'_fill_occupations'} if restricted else {'_open_shell_fill', '_occupy_and_density'}
    closures = [n for n in function.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(closures) == len(names)
    energies = [np.asarray(e) for e in spectra]
    weights = np.full(len(energies), 1. / len(energies))
    scope = dict(
        np=np, List=List, fitted_support=object(), _chi_xc=None,
        _ibz_native_state=None, weights=weights, n_occ=2, n_elec=4,
        n_alpha=2, n_beta=1, smearing_T=0., smear_opts=SmearingOptions(),
        bz_integration=None, system=None, kmesh_bloch=None,
        _chi_trial_occupations=gdf._chi_trial_occupations,
        _occupations_per_k=gdf._occupations_per_k,
        _gdf_open_shell_occupations=gdf._gdf_open_shell_occupations,
        _density_from_orbitals=gdf._density_from_orbitals,
    )
    exec(compile(ast.Module(body=closures, type_ignores=[]), gdf.__file__, 'exec'), scope)

    def fill(*, trial):
        if restricted:
            return scope['_fill_occupations'](energies, trial=trial)[0]
        c = [np.eye(len(e)) for e in energies]
        result = scope['_occupy_and_density'](c, energies, c, energies, trial=trial)
        for density, occupation in zip(result[1], result[3]):
            np.testing.assert_array_equal(density @ density, density)
            assert sum(occupation) == 1.
        return result[2]

    maximum = 2. if restricted else 1.
    trial = fill(trial=True)
    physical = fill(trial=False)
    for occupation in trial:
        np.testing.assert_array_equal(occupation / maximum, (occupation / maximum) ** 2)
        assert sum(occupation) == 2. * maximum
    gdf._chi_require_integer_occupations(trial, 2, maximum)
    with pytest.raises(NotImplementedError, match='integer Aufbau'):
        gdf._chi_require_integer_occupations(physical, 2, maximum)
    assert sum(w * sum(o) for w, o in zip(weights, physical)) == pytest.approx(2. * maximum)

    # The trial override is confined to explicit HF; ordinary GDF and XC
    # continue to use their original global occupations at every iteration.
    for fitted, xc in [(None, None), (object(), object())]:
        scope['fitted_support'], scope['_chi_xc'] = fitted, xc
        for actual, expected in zip(fill(trial=True), physical):
            np.testing.assert_array_equal(actual, expected)

    # Once the physical spectrum is gapped, both fills agree exactly.
    scope['fitted_support'], scope['_chi_xc'] = object(), None
    energies[:] = [np.arange(len(e), dtype=float) for e in energies]
    for actual, expected in zip(fill(trial=True), fill(trial=False)):
        np.testing.assert_array_equal(actual, expected)
    gdf._chi_require_integer_occupations(fill(trial=False), 2, maximum)


def test_chi_trial_empty_full_and_insufficient_spin_spaces():
    from vibeqc.periodic_k_gdf import _chi_trial_occupations
    eps = [np.array([-1., .5]), np.array([-.8, .7])]
    empty, mu, entropy = _chi_trial_occupations(eps, 0, 1.)
    np.testing.assert_array_equal(empty, np.zeros((2, 2)))
    assert mu == -1. and entropy == 0.
    full, mu, entropy = _chi_trial_occupations(eps, 2, 1.)
    np.testing.assert_array_equal(full, np.ones((2, 2)))
    assert mu == .7 and entropy == 0.
    with pytest.raises(ValueError, match='too few retained orbitals'):
        _chi_trial_occupations(eps, 3, 1.)


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


@pytest.mark.parametrize('restricted', [False, True])
def test_semilocal_rebuild_uses_xc_energy_and_derivative_once(kernels, restricted):
    # E_xc = c/2 sum_spin Tr(D_s^2), V_s = c D_s.
    # In the restricted representation E_xc=c/4 Tr(D_total^2).
    c = .3
    d = np.diag([1., 0.]).astype(complex)
    zero = lambda ds: [np.zeros_like(v) for v in ds]
    def forbidden(ds):
        raise AssertionError('semilocal KS must not execute exact exchange')
    class XC:
        def restricted(self, total):
            return c / 4 * sum(np.trace(x @ x).real for x in total), [c / 2 * x for x in total]
        def unrestricted(self, a, b):
            return c / 2 * sum(np.trace(x @ x).real for x in a+b), [c*x for x in a], [c*x for x in b]
    report, f = kernels['_chi_rebuild_hf_acceptance'](
        [np.diag([-1.5, .7])], [np.eye(2)], [np.eye(2)], [1.],
        [[d], [d]], (1, 1), zero, forbidden, 999., 0., -3.+c,
        0., 1e-10, 1e-8, xc=XC(), restricted=restricted)
    assert report['accepted']
    assert report['rebuilt_energy'] == pytest.approx(-2.7)
    np.testing.assert_allclose(f[0][0], np.diag([-1.2, .7]))
    np.testing.assert_array_equal(f[0], f[1])


@pytest.mark.parametrize("restricted,open_shell", [(True,False),(False,False),(False,True)])
def test_hybrid_final_rebuild_uses_complete_exchange_once(kernels,restricted,open_shell):
    # Independently soluble diagonal root: the selected complete K already
    # contains full and screened zero modes. A large legacy xi detects reuse.
    a=np.diag([1.,0.]).astype(complex)
    b=np.zeros_like(a) if open_shell else a.copy()
    spins=[[a],[b]]
    c=.3
    exchange_scale=.65-.46*.7
    def complete(ds): return [exchange_scale*d for d in ds]
    def forbidden(ds): raise AssertionError("raw full K must not be selected")
    class XC:
        def restricted(self,total):
            return c/4*sum(np.trace(d@d).real for d in total),[c/2*d for d in total]
        def unrestricted(self,a,b):
            return c/2*sum(np.trace(d@d).real for d in a+b),[c*d for d in a],[c*d for d in b]
    zero=lambda ds:[np.zeros_like(d) for d in ds]
    n=1 if open_shell else 2
    expected=n*(-1.5+c/2-exchange_scale/2)
    report,focks=kernels['_chi_rebuild_hf_acceptance'](
        [np.diag([-1.5,.7])],[np.eye(2)],[np.eye(2)],[1.],spins,(1,0 if open_shell else 1),
        zero,forbidden,999.,0.,expected,0.,1e-10,1e-8,
        xc=XC(),restricted=restricted,exchange_builder=complete)
    assert report['accepted'] and report['energy_consistency_error']<1e-14
    assert report['electron_counts']==[1.,0. if open_shell else 1.]
    for f,d in zip(focks,[a,b]):
        np.testing.assert_allclose(f[0],np.diag([-1.5,.7])+(c-exchange_scale)*d,atol=1e-14)
