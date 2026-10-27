"""Small algebra/admission tests; optional native checks need a matching core."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

# Keep the pure admission algebra executable without importing a native runtime.
_spec = spec_from_file_location('chi_transform_inputs', Path(__file__).resolve().parents[1] / 'python/vibeqc/periodic/chi/_transform_inputs.py')
inputs = module_from_spec(_spec)
_spec.loader.exec_module(inputs)


@pytest.mark.parametrize('bad', [.8, np.nan, np.inf, -np.inf, 2**63, -(2**63)-1, np.uint64(2**63), float(2**63), True, 1+0j, '1'])
def test_translation_rejects_before_cast(bad):
    with pytest.raises(ValueError):
        inputs.translations_array([[bad, 0, 0]])


@pytest.mark.parametrize('bad', [[], [0, 0, 0], [[0, 0]], np.zeros((1, 1, 3))])
def test_translation_shapes(bad):
    with pytest.raises(ValueError):
        inputs.translations_array(bad)


def test_integral_float_and_int64_limits_preserved():
    np.testing.assert_array_equal(inputs.translations_array([[1., -2., 0.]]), [[1, -2, 0]])
    assert inputs.translations_array([[-2**63, 2**63-1, 0]]).tolist() == [[-2**63, 2**63-1, 0]]


@pytest.mark.parametrize('weights', [[1.5,-.5], [np.nan,1], [np.inf,0], [1,1], [[.5,.5]], [.5], [.5+.1j,.5]])
def test_bad_weights(weights):
    with pytest.raises(ValueError):
        inputs.inverse_inputs(np.ones((2,1,1)), [[0,0,0],[.5,0,0]], weights)


@pytest.mark.parametrize('weights', [[.5,.5], [1.,0.], [.2,.8]])
def test_supported_weight_conventions(weights):
    assert inputs.inverse_inputs(np.ones((2,1,1)), [[0,0,0],[.5,0,0]], weights)[2].tolist() == weights


@pytest.mark.parametrize('matrices,k', [
    (np.ones((0,1,1)), np.empty((0,3))), (np.ones((1,0,0)), [[0,0,0]]),
    (np.ones((1,1,2)), [[0,0,0]]), (np.ones((1,1,1)), [0,0,0]),
    (np.ones((1,1,1)), [[np.nan,0,0]]), (np.ones((1,1,1)), [[1j,0,0]]),
    (np.array([[[np.inf]]]), [[0,0,0]]), (np.array([[[1+np.nan*1j]]]), [[0,0,0]]),
])
def test_finite_nonempty_shapes(matrices,k):
    with pytest.raises(ValueError):
        inputs.inverse_inputs(matrices,k)


@pytest.mark.parametrize('extension,k,translations', [
    (np.diag([3,1,1]), [[0,0,0],[1/3,0,0],[2/3,0,0]], [[0,0,0],[1,0,0],[2,0,0]]),
    ([[2,1,0],[0,2,0],[0,0,1]], [[0,0,0],[0,.5,0],[.5,.25,0],[.5,.75,0]], [[0,0,0],[0,1,0],[1,0,0],[1,1,0]]),
    ([[-2,1,0],[0,1,0],[0,0,1]], [[0,0,0],[.5,.5,0]], [[0,0,0],[1,0,0]]),
])
def test_complete_character_complex_round_trip(extension,k,translations):
    k = np.asarray(k); n=len(k); weights=np.full(n,1/n)
    inputs.complete_characters(k, weights, extension)
    rng=np.random.default_rng(419); matrices=rng.normal(size=(n,2,2))+1j*rng.normal(size=(n,2,2))
    phases=np.exp(-2j*np.pi*inputs.phase_arguments(k,np.array(translations)))
    real=np.einsum('rk,kij->rij',phases,matrices)/n
    # Independent forward formula at these small exact representatives.
    forward=np.exp(2j*np.pi*(k@np.asarray(translations).T))
    np.testing.assert_allclose(np.einsum('kr,rij->kij',forward,real),matrices,atol=3e-14,rtol=0)


@pytest.mark.parametrize('s', [np.zeros((3,3)), np.eye(2), np.diag([1.1,1,1]), np.diag([np.inf,1,1]), np.diag([2**62,4,1]), [[2**63,0,0],[0,1,0],[0,0,1]]])
def test_invalid_extension(s):
    with pytest.raises(ValueError): inputs.complete_characters([[0,0,0]],[1],s)


@pytest.mark.parametrize('k,w', [([[0,0,0],[0,0,0]],[.5,.5]), ([[0,0,0],[.25,0,0]],[.5,.5]), ([[0,0,0],[.5,0,0]],[1,0]), ([[0,0,0],[.5,0,0]],[.2,.8])])
def test_inconsistent_complete_characters(k,w):
    with pytest.raises(ValueError): inputs.complete_characters(k,w,np.diag([2,1,1]))


def test_large_extension_membership_is_exact():
    s=[[2,2**54+1,0],[0,1,0],[0,0,1]]
    with pytest.raises(ValueError): inputs.complete_characters([[0,0,0],[.5,0,0]],[.5,.5],s)
    inputs.complete_characters([[0,0,0],[.5,.5,0]],[.5,.5],s)


def test_large_translation_phase_and_reciprocal_representatives():
    r=np.array([[2**53+1,0,0],[-2**63,0,0],[2**63-1,0,0]],dtype=np.int64)
    np.testing.assert_array_equal(inputs.phase_arguments(np.array([[.5,0,0],[1e308,0,0]]),r), [[.5,0],[0,0],[.5,0]])


@pytest.mark.parametrize('mesh', [(2**63,1,1),(2**32,2**32,1),(0,1,1)])
def test_mesh_overflow(mesh):
    with pytest.raises(ValueError): inputs.checked_mesh_size(mesh)


@pytest.fixture
def kernel():
    return pytest.importorskip('vibeqc._vibeqc_core').aiccm2026dev_b_inverse_bloch_transform


@pytest.mark.parametrize('which,bad', [
    ('r',[[.8,0,0]]), ('r',[[2**63,0,0]]), ('r',[[True,0,0]]),
    ('r',[[1j,0,0]]), ('r',np.empty((0,3))), ('k',[[1j,0,0]]),
    ('k',[[np.inf,0,0]]), ('w',[-1.]), ('w',[np.nan]), ('w',[1j]),
    ('m',[[[np.nan]]]), ('m',np.empty((1,0,0))),
])
def test_direct_native_refuses_invalid_inputs(kernel,which,bad):
    args=dict(m=np.ones((1,1,1)), k=[[0,0,0]], r=[[0,0,0]], w=[1.])
    args[which]=bad
    with pytest.raises((ValueError,RuntimeError,TypeError)):
        kernel(args['m'],args['k'],args['r'],args['w'])


@pytest.mark.parametrize('weights', [[.5,.5],[1.,0.],[.2,.8]])
def test_native_weighted_sum_and_large_phase(kernel,weights):
    matrices=np.array([[[1+2j]],[[3-4j]]]); k=np.array([[0,0,0],[.5,0,0]])
    r=np.array([[0,0,0],[1,0,0],[2**53+1,0,0],[-2**63,0,0]])
    phases=np.array([[1,1],[1,-1],[1,-1],[1,1]])
    expected=np.einsum('rk,k,kij->rij',phases,weights,matrices)
    np.testing.assert_allclose(kernel(matrices,k,r,weights),expected,atol=2e-14,rtol=0)


def test_public_and_private_dispatch_reject_before_native(monkeypatch):
    scf=pytest.importorskip('vibeqc.periodic.chi.scf')
    calls=[]
    def forbidden(*args):
        calls.append(args)
        raise AssertionError('invalid data reached native dispatch')
    monkeypatch.setattr(scf,'_inverse_bloch_transform_native',forbidden)
    for r,w in [([[.8,0,0]],[.5,.5]), ([[0,0,0]],[1.5,-.5])]:
        with pytest.raises(ValueError): scf.inverse_bloch_transform(np.ones((2,1,1)),[[0,0,0],[.5,0,0]],r,w)
    assert not calls


def test_private_dispatch_cannot_narrow_invalid_data():
    scf=pytest.importorskip('vibeqc.periodic.chi.scf')
    for r,w in [([[.8,0,0]],[1.]), ([[0,0,0]],[-1.])]:
        with pytest.raises(ValueError):
            scf._inverse_bloch_transform_native(np.ones((1,1,1)), [[0,0,0]], r, w)


@pytest.mark.parametrize('name', ['aiccm2026dev_b_matrix_to_real_supercell','aiccm2026dev_b_canonical_wannier_coefficients','aiccm2026dev_b_lpq_to_real_supercell','aiccm2026dev_b_lpq_to_real_home_auxiliary'])
@pytest.mark.parametrize('bad', [.8, np.nan, 2**63, 1j])
def test_sibling_native_transform_translation_guards(name,bad):
    core=pytest.importorskip('vibeqc._vibeqc_core')
    data=np.ones((1,1,1),dtype=complex)
    if 'lpq' in name: data={(0,0):data}
    args=[data,[[0,0,0]],[[bad,0,0]]]
    if 'wannier' in name: args.append(1)
    if 'home_auxiliary' in name: args.append([1,1,1])
    with pytest.raises((ValueError,RuntimeError,TypeError)):
        getattr(core,name)(*args)


def test_native_factor_transform_rejects_nonfinite_cache():
    core=pytest.importorskip('vibeqc._vibeqc_core')
    with pytest.raises(ValueError):
        core.aiccm2026dev_b_lpq_to_real_supercell({(0,0):np.array([[[np.nan]]])},[[0,0,0]],[[0,0,0]])


def test_public_complete_character_round_trip(monkeypatch):
    scf=pytest.importorskip('vibeqc.periodic.chi.scf')
    k=np.array([[0,0,0],[0,.5,0],[.5,.25,0],[.5,.75,0]])
    r=np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,0]])
    s=[[2,1,0],[0,2,0],[0,0,1]]
    values=np.arange(16).reshape(4,2,2)+.2j
    native=scf.inverse_bloch_transform(values,k,r,extension_matrix=s)
    monkeypatch.setattr(scf,'_inverse_bloch_transform_native',lambda *a: None)
    fallback=scf.inverse_bloch_transform(values,k,r,extension_matrix=s)
    np.testing.assert_allclose(native,fallback,atol=2e-14,rtol=0)
    forward=np.exp(2j*np.pi*(k@r.T))
    np.testing.assert_allclose(np.einsum('kr,rij->kij',forward,fallback),values,atol=3e-14,rtol=0)
    with pytest.raises(ValueError): scf.inverse_bloch_transform(values,k,r,[1.,0,0,0],extension_matrix=s)


@pytest.mark.parametrize('suffix', ['', '_complex'])
@pytest.mark.parametrize('bad', [1.8, True, np.nan, 2**63, -1, 0])
def test_home_mo_mesh_rejects_before_narrowing(suffix, bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    fn = getattr(core, 'aiccm2026dev_b_home_auxiliary_3index_mo_transform' + suffix)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        fn(np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1)), [bad, 1, 1])


@pytest.mark.parametrize('home', [False, True])
@pytest.mark.parametrize('suffix', ['', '_complex'])
@pytest.mark.parametrize('which', [0, 1, 2])
def test_mo_transform_rejects_nonfinite_operands(home, suffix, which):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    name = 'aiccm2026dev_b_' + ('home_auxiliary_' if home else '') + '3index_mo_transform' + suffix
    args = [np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))]
    args[which].flat[0] = np.nan
    if home:
        args.append([1, 1, 1])
    with pytest.raises(ValueError, match='finite'):
        getattr(core, name)(*args)


@pytest.mark.parametrize('home', [False, True])
@pytest.mark.parametrize('which', [0, 1, 2])
def test_real_mo_transform_does_not_discard_imaginary_input(home, which):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    args = [np.ones((1, 1, 1)), np.ones((1, 1)), np.ones((1, 1))]
    args[which] = args[which].astype(complex) * (1 + 2j)
    if home:
        args.append([1, 1, 1])
    name = 'aiccm2026dev_b_' + ('home_auxiliary_' if home else '') + '3index_mo_transform'
    with pytest.raises(ValueError, match='real'):
        getattr(core, name)(*args)


@pytest.mark.parametrize('suffix', ['', '_complex'])
def test_home_mo_preserves_empty_orbital_space(suffix):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    fn = getattr(core, 'aiccm2026dev_b_home_auxiliary_3index_mo_transform' + suffix)
    result = fn(np.ones((1, 2, 2)), np.empty((2, 0)), np.ones((2, 1)), [2, 1, 1])
    assert result.shape == (2, 0, 1)


@pytest.mark.parametrize('bad', [1.8, True, np.nan, 2**63, -1, 0])
def test_permutation_boundaries_reject_invalid_counts(bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    for fn in (core.aiccm2026dev_b_translation_permutations,
               pno.translation_permutations, pno._translation_permutations_native,
               pno._translation_permutations_python):
        for bands, mesh in [(1, [bad, 1, 1]), (bad, [1, 1, 1])]:
            with pytest.raises((ValueError, RuntimeError, TypeError)):
                fn(bands, mesh)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_pair_orbits([[0]], bad)
    for fn in (pno.enumerate_pair_orbits, pno._pair_orbits_from_native,
               pno._enumerate_pair_orbits_python):
        with pytest.raises(ValueError):
            fn([[0]], n_cells=bad)


@pytest.mark.parametrize('bad', [0.8, True, np.nan, 2**63])
def test_pair_permutations_reject_before_cast(bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_pair_orbits([[bad]], 1)
    for fn in (pno.enumerate_pair_orbits, pno._pair_orbits_from_native,
               pno._enumerate_pair_orbits_python):
        with pytest.raises(ValueError):
            fn([[bad]], n_cells=1)


def test_translation_permutation_and_pair_closure_oracles():
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    # Two bands on a three-cell cyclic group. Direct integer label action is
    # independent of the native and fallback group enumeration.
    expected = np.array([[((cell + shift) % 3) * 2 + band
                          for cell in range(3) for band in range(2)]
                         for shift in range(3)])
    for fn in (pno.translation_permutations, pno._translation_permutations_python):
        np.testing.assert_array_equal(fn(2, [3, 1, 1]), expected)
    native = pno.enumerate_pair_orbits(expected, n_cells=3)
    fallback = pno._enumerate_pair_orbits_python(expected, n_cells=3)
    assert native == fallback
    assert sorted(pair for orbit in native for pair in orbit.members) == [
        (i, j) for i in range(6) for j in range(i, 6)]
    assert sum(orbit.per_cell_weight for orbit in native) == 7.0


@pytest.mark.parametrize('mesh,bands', [([2**62, 4, 1], 1), ([2**32, 1, 1], 1), ([2, 1, 1], 2**62)])
def test_translation_permutation_output_overflow(mesh, bands):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    pno = pytest.importorskip('vibeqc.periodic.chi.pno')
    for fn in (core.aiccm2026dev_b_translation_permutations, pno.translation_permutations):
        with pytest.raises((ValueError, RuntimeError)):
            fn(bands, mesh)


@pytest.mark.parametrize('name', ['tensor', 'matrix'])
@pytest.mark.parametrize('which,bad', [('mesh', [1.8, 1, 1]), ('mesh', [True, 1, 1]),
    ('mesh', [2**62, 4, 1]), ('atoms', [0.8]), ('atoms', [True]),
    ('atoms', [2**63-1]), ('density', [[[np.nan]]]), ('overlap', [[[np.inf]]]),
    ('density', [[[1+2j]]]), ('beta', [[[complex(1, np.nan)]]])])
def test_mayer_transform_rejects_invalid_raw_inputs(name, which, bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    args = dict(density=[[[1.]]], overlap=[[[1.]]], atoms=[0], mesh=[1, 1, 1], beta=None)
    args[which] = bad
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        getattr(core, 'aiccm2026dev_b_mayer_pair_' + name)(
            args['density'], args['overlap'], args['atoms'], args['mesh'], args['beta'])


@pytest.mark.parametrize('mesh', [[1.8, 1, 1], [True, 1, 1], [2**62, 4, 1], [1, 1], [0, 1, 1]])
def test_adapter_and_property_mesh_admission(mesh):
    from types import SimpleNamespace
    posthf = pytest.importorskip('vibeqc.periodic.chi.posthf')
    properties = pytest.importorskip('vibeqc.periodic.chi.properties')
    with pytest.raises(ValueError):
        posthf._BRealDFAdapter(np.ones((1, 1, 1)), None, mesh)
    with pytest.raises(ValueError):
        properties._mesh_from_result(SimpleNamespace(aiccm2026dev_b=SimpleNamespace(mesh=mesh)))


def test_adapter_rejects_nonfinite_before_native(monkeypatch):
    posthf = pytest.importorskip('vibeqc.periodic.chi.posthf')
    factors_module = pytest.importorskip('vibeqc.dlpno.translation_factors')
    with pytest.raises(ValueError):
        posthf._BRealDFAdapter(np.full((1, 1, 1), np.nan), None, (1, 1, 1))
    adapter = posthf._BRealDFAdapter(np.ones((1, 1, 1)), None, (1, 1, 1))
    def forbidden(*args):
        raise AssertionError('invalid coefficients reached native dispatch')
    monkeypatch.setattr(factors_module, 'aiccm2026dev_b_home_auxiliary_3index_mo_transform', forbidden)
    with pytest.raises(ValueError):
        adapter.mo_transform([[np.inf]])


@pytest.mark.parametrize('bad', [True, 1.8, np.nan, 2**63])
def test_native_orbital_counts_reject_before_cast(bad):
    core = pytest.importorskip('vibeqc._vibeqc_core')
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_canonical_wannier_coefficients(
            np.ones((1, 1, 1)), [[0, 0, 0]], [[0, 0, 0]], bad)
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        core.aiccm2026dev_b_localization_projector_audit(
            np.ones((1, 1)), np.ones((1, 1)), None, [[0, 0, 0]], [1, 1, 1], bad)


def test_localization_mesh_overflow_refuses_before_products():
    core = pytest.importorskip('vibeqc._vibeqc_core')
    with pytest.raises(RuntimeError, match='mesh product'):
        core.aiccm2026dev_b_localization_projector_audit(
            np.ones((1, 1)), np.ones((1, 1)), None, [[0, 0, 0]], [2**62, 4, 1], 1)


@pytest.mark.parametrize('mesh', [(1, 1, 1), (3, 1, 1), (2, 2, 1), (2, 1, 3)])
def test_translation_cyclic_operator_actions_match_unreduced_matrix(mesh):
    from itertools import product
    from vibeqc.dlpno.translation_operators import (
        CyclicAOOperator, matrix_apply, matrix_block, matrix_columns_apply,
        matrix_diagonal, matrix_record, matrix_snapshot,
    )

    cells = tuple(product(*(range(n) for n in mesh)))
    lookup = {cell: i for i, cell in enumerate(cells)}
    rng = np.random.default_rng(717)
    home = rng.normal(size=(len(cells), 2, 2))
    # Deliberately nonsymmetric off-home blocks distinguish displacement
    # signs and transpose actions even on odd and multidimensional meshes.
    dense = np.block([[home[lookup[tuple((t[a]-r[a]) % mesh[a] for a in range(3))]]
                       for t in cells] for r in cells])
    operator = CyclicAOOperator(home, mesh)
    snapshot = operator.blocks.copy()
    home[:] = 0
    columns = rng.normal(size=(len(dense), 3))
    for transposed in (False, True):
        expected = dense.T if transposed else dense
        np.testing.assert_allclose(matrix_apply(operator, columns, transpose=transposed),
                                   expected @ columns, atol=3e-14)
        selection = np.array([len(dense)-1, 0, 0], dtype=np.int64)
        weights = rng.normal(size=(3, 2))
        np.testing.assert_allclose(matrix_columns_apply(operator, selection, weights,
                                                        transpose=transposed),
                                   expected[:, selection] @ weights, atol=3e-14)
    rows, cols = np.array([len(dense)-1, 0, 0]), np.array([1, 0])
    np.testing.assert_array_equal(matrix_block(operator, rows, cols), dense[np.ix_(rows, cols)])
    np.testing.assert_array_equal(matrix_diagonal(operator), np.diag(dense))
    assert operator.blocks.nbytes*len(cells) == dense.nbytes
    assert matrix_snapshot(operator, mesh) is operator
    assert matrix_record(operator)['identity'] == operator.identity
    assert CyclicAOOperator(snapshot, mesh).identity == operator.identity
    assert CyclicAOOperator(snapshot*1.01, mesh).identity != operator.identity
    assert matrix_apply(operator, np.empty((len(dense), 0))).shape == (len(dense), 0)
    with pytest.raises(TypeError, match='implicit dense'):
        np.asarray(operator)
    with pytest.raises(ValueError):
        operator.blocks.setflags(write=True)
    with pytest.raises(AttributeError):
        operator.mesh = (1, 1, 1)
    with pytest.raises(AttributeError):
        operator.__init__(snapshot, mesh)
    with pytest.raises(TypeError):
        type('ChangedOperator', (CyclicAOOperator,), {})


@pytest.mark.parametrize('fault', ['negative_symbol', 'asymmetric', 'nan', 'complex',
                                 'shape', 'mesh', 'boolean_mesh'])
def test_translation_cyclic_operator_refuses_invalid_finite_metric(fault):
    from vibeqc.dlpno.translation_operators import CyclicAOOperator, validate_translation_matrix

    blocks = np.array([[[1.]], [[.1]], [[.1]]])
    mesh = (3, 1, 1)
    if fault == 'negative_symbol':
        # S_home remains positive; the nonzero-k metric is negative.
        blocks[1:] = 1.5
    elif fault == 'asymmetric':
        blocks[1] = .2
    elif fault == 'nan':
        blocks[0] = np.nan
    elif fault == 'complex':
        blocks = blocks.astype(complex)
    elif fault == 'shape':
        blocks = np.ones((3, 1, 2))
    elif fault == 'mesh':
        mesh = (2, 1, 1)
    else:
        mesh = (True, 3, 1)
    before = blocks.copy()
    with pytest.raises((ValueError, TypeError)):
        operator = CyclicAOOperator(blocks, mesh)
        validate_translation_matrix(operator, mesh, 1e-10, positive=True)
    np.testing.assert_array_equal(blocks, before)


def test_translation_cyclic_operator_reciprocal_metric_preserves_ao_symmetry_tolerance():
    from vibeqc.dlpno.translation_operators import CyclicAOOperator, validate_translation_matrix

    # The AO-entry symmetry tolerance is not an extra absolute tolerance on
    # Fourier sums, which can amplify the admitted input roundoff by N.
    blocks = np.zeros((3, 2, 2))
    blocks[0] = 4*np.eye(2)
    blocks[:, 0, 1] = 8e-11
    validate_translation_matrix(CyclicAOOperator(blocks, (3, 1, 1)), (3, 1, 1),
                                1e-10, positive=True)


def test_translation_cyclic_full_domain_prepares_without_square_source_blocks(monkeypatch):
    from tests.test_dlpno_source import _automatic_translation_source
    from vibeqc.dlpno import translation_pao
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
    from vibeqc.dlpno.translation_operators import CyclicAOOperator

    mesh = (3, 1, 1)
    source, _, core, _ = _automatic_translation_source(mesh, True)
    local = source.basis.nbasis//3
    operators = [CyclicAOOperator(np.stack([value[:local, t*local:(t+1)*local] for t in range(3)]), mesh)
                 for value in (source.overlap, source.reference.fock)]
    source.overlap, source.reference.fock = operators
    block = translation_pao.matrix_block
    calls = []

    def bounded_block(operator, rows, columns):
        assert type(operator) is CyclicAOOperator
        assert len(rows) <= local
        calls.append((len(rows), len(columns)))
        return block(operator, rows, columns)

    cholesky = np.linalg.cholesky

    def bounded_cholesky(matrix):
        assert matrix.shape != (source.basis.nbasis,)*2
        return cholesky(matrix)

    monkeypatch.setattr(translation_pao, 'matrix_block', bounded_block)
    monkeypatch.setattr(np.linalg, 'cholesky', bounded_cholesky)
    result = run_translation_dlpno_mp2(source, mesh, n_frozen=core.shape[1])
    assert result.solution.converged
    assert calls and max(columns for _, columns in calls) == source.basis.nbasis
    assert result.solution.provenance['transport_preparation']['metric_backend'] == 'cyclic-blocks'


def _bloch_translation_model(mesh):
    """Independent character representation of a real finite AO Hamiltonian."""
    from types import SimpleNamespace
    from scipy.linalg import eigh
    import vibeqc as vq
    from tests.test_dlpno_mp2 import _translation_mp2_model
    from vibeqc.periodic.chi.scf import cyclic_gamma_mesh

    system = vq.PeriodicSystem(3, np.diag([7., 8., 9.]),
        [vq.Atom(1, [1., 1., 1.]), vq.Atom(1, [2.4, 1., 1.]),
         vq.Atom(1, [1., 4., 1.]), vq.Atom(1, [2.4, 4., 1.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    model = _translation_mp2_model(mesh, 2, full=True)
    kpoints = cyclic_gamma_mesh(system, mesh)
    cells = np.array(list(np.ndindex(mesh)))
    kfrac = np.asarray(kpoints.kpoints_frac)
    forward = np.exp(2j*np.pi*(kfrac @ cells.T))

    def reciprocal(matrix):
        home = np.stack([matrix[:4, 4*t:4*(t+1)] for t in range(len(cells))])
        return np.einsum('kt,tab->kab', forward, home)

    S, F = reciprocal(model.overlap), reciprocal(model.fock)
    eigensystems = [eigh(f, s) for f, s in zip(F, S)]
    eps = np.array([value for value, _ in eigensystems])
    C = np.array([value for _, value in eigensystems])
    # Independent k-dependent phases make the realification exercise the
    # complete occupied/virtual subspaces, rather than coincident real MOs.
    C *= np.exp(1j*np.arange(C.shape[0]*4).reshape(-1, 1, 4)*.17)
    hf = SimpleNamespace(
        overlap=list(S), fock=list(F), mo_coeffs=list(C), mo_energies=list(eps),
        density=list(2*C[:, :, :2] @ C[:, :, :2].conj().transpose(0, 2, 1)),
        energy=-3., converged=True, smearing_temperature=0.,
        occupations=np.tile([2., 2., 0., 0.], (len(cells), 1)),
        kpoints_frac=kfrac.copy(), kpoints_cart=np.asarray(kpoints.kpoints_cart).copy(),
        kpoint_weights=np.asarray(kpoints.weights).copy(), ecp_total_ncore=0,
        reference_kind='rhf',
    )
    return system, basis, hf, model


def _bloch_state_arrays(hf):
    return [np.asarray(value) for name in ('overlap', 'fock', 'mo_coeffs', 'mo_energies', 'density')
            for value in getattr(hf, name)] + [hf.kpoints_frac, hf.kpoints_cart,
                                              hf.kpoint_weights, hf.occupations]


@pytest.mark.parametrize('mesh,flat_bands', [((3, 1, 1), False),
                                           ((2, 2, 1), False),
                                           ((5, 1, 1), True)])
def test_bloch_realification_streams_complete_eigenspaces(mesh, flat_bands, monkeypatch, record_property):
    """The dense fold is an independent oracle, never a product fallback."""
    from vibeqc.dlpno import translation_bloch as module
    from vibeqc.dlpno.source import _real_gamma_orbitals

    system, basis, hf, model = _bloch_translation_model(mesh)
    if flat_bands:
        # A whole occupied band crosses every k point. Its arbitrary complex
        # gauges must be realified together even when wider than one cell.
        for k in range(model.n_cells):
            hf.mo_energies[k] = np.array([-1.6, -1.1, .7, 1.3])
            SC = hf.overlap[k] @ hf.mo_coeffs[k]
            hf.fock[k] = (SC*hf.mo_energies[k]) @ SC.conj().T
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    cells = np.array(list(np.ndindex(mesh)))
    phases = np.exp(2j*np.pi*(cells @ hf.kpoints_frac.T))/np.sqrt(model.n_cells)
    full = np.concatenate([
        (phases[:, k, None, None]*hf.mo_coeffs[k][None]).reshape(4*model.n_cells, 4)
        for k in range(model.n_cells)
    ], axis=1)
    original = module._real_gamma_orbitals_from_columns
    concatenate = np.concatenate
    requested = []
    reference_dense = []

    def no_full_complex_fold(values, axis=0, **kwargs):
        if (axis == 1 and all(value.ndim == 2 and value.shape[0] == full.shape[0]
                             for value in values)
                and sum(value.shape[1] for value in values) == full.shape[1]
                and any(np.iscomplexobj(value) for value in values)):
            raise AssertionError('full complex canonical fold was prepared')
        return concatenate(values, axis=axis, **kwargs)

    def streamed(nbf, eps, columns, S, F):
        reference_dense.append(_real_gamma_orbitals(full, eps, S, F))

        def observed(indices):
            block = columns(indices)
            requested.append(indices.copy())
            np.testing.assert_allclose(block, full[:, indices], atol=1e-15, rtol=0)
            assert block.shape == (nbf, len(indices))
            assert block.shape[1] < full.shape[1]
            return block

        return original(nbf, eps, observed, S, F)

    monkeypatch.setattr(module, '_real_gamma_orbitals_from_columns', streamed)
    monkeypatch.setattr(np, 'concatenate', no_full_complex_fold)
    reference, record = module.bloch_translation_reference(
        system, basis, hf, mesh, n_frozen=model.n_cells)
    assert requested
    np.testing.assert_array_equal(np.concatenate(requested),
                                  np.argsort(np.asarray(hf.mo_energies).ravel(), kind='stable'))
    expected_C, expected_eps = reference_dense[0]
    # Individual vectors inside a degenerate block may rotate; its metric
    # projector, energies and the separate frozen/active spaces may not.
    cursor = 0
    for indices in requested:
        end = cursor+len(indices)
        actual, expected = reference.mo_coeffs[:, cursor:end], expected_C[:, cursor:end]
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=3e-12, rtol=0)
        cursor = end
    np.testing.assert_allclose(reference.mo_energies, expected_eps, atol=3e-12, rtol=0)
    max_columns = max(map(len, requested))
    assert record['max_folded_orbital_columns'] == max_columns
    assert record['orbital_validation_columns'] == basis.nbasis
    if flat_bands:
        assert max_columns == model.n_cells
    assert record['orbital_preparation'] == 'complete-degenerate-bloch-column-blocks'
    record_property('dense_complex_fold_bytes', full.nbytes)
    record_property('largest_complex_fold_block_bytes', full.shape[0]*max_columns*full.itemsize)
    for value, saved in zip(arrays, before):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize('fault', ['cross_panel_overlap', 'virtual_energy'])
def test_bloch_panel_validation_keeps_global_orbital_guards(fault, monkeypatch):
    from vibeqc.dlpno import translation_bloch as module

    system, basis, hf, _ = _bloch_translation_model((3, 1, 1))
    if fault == 'cross_panel_overlap':
        # Six degenerate occupieds span two four-column validation panels.
        # Mixing their columns preserves stationarity but not the metric.
        for k in range(3):
            hf.mo_energies[k] = np.array([-1.5, -1.5, .7, 1.3])
            SC = hf.overlap[k] @ hf.mo_coeffs[k]
            hf.fock[k] = (SC*hf.mo_energies[k]) @ SC.conj().T
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    original = module._real_gamma_orbitals_from_columns

    def corrupt(nbf, energies, columns, S, F):
        C, eps = original(nbf, energies, columns, S, F)
        if fault == 'cross_panel_overlap':
            C[:, 5] += 1e-5*C[:, 0]
            np.testing.assert_allclose(F.apply(C), S.apply(C)*eps, atol=1e-12, rtol=0)
        else:
            eps[-1] += 1e-5
            np.testing.assert_allclose(C.T @ S.apply(C), np.eye(nbf), atol=1e-12, rtol=0)
        return C, eps

    monkeypatch.setattr(module, '_real_gamma_orbitals_from_columns', corrupt)
    with pytest.raises(ValueError, match='realified orbitals do not preserve'):
        module.bloch_translation_reference(system, basis, hf, (3, 1, 1))
    for value, saved in zip(arrays, before):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize('mesh,flat_bands', [((3, 1, 1), False), ((2, 2, 1), False),
                                           ((5, 1, 1), True)])
def test_compact_bloch_audit_replays_without_full_canonical_storage(mesh, flat_bands,
                                                                   monkeypatch, record_property):
    from vibeqc.dlpno import source as source_module
    from vibeqc.dlpno import translation_bloch as module

    system, basis, hf, model = _bloch_translation_model(mesh)
    if flat_bands:
        for k in range(model.n_cells):
            hf.mo_energies[k] = np.array([-1.6, -1.1, .7, 1.3])
            SC = hf.overlap[k] @ hf.mo_coeffs[k]
            hf.fock[k] = (SC*hf.mo_energies[k]) @ SC.conj().T
    arrays = _bloch_state_arrays(hf)
    before = [(value.copy(), value.flags.writeable) for value in arrays]
    dense, expected = module.bloch_translation_reference(system, basis, hf, mesh,
                                                         n_frozen=model.n_cells)
    size = dense.mo_coeffs.shape[0]
    observed = []
    decompositions = []
    realify = source_module._RealGammaOrbitalBlocks.realify
    eigh = np.linalg.eigh

    def observe_eigh(matrix):
        decompositions.append(matrix.shape)
        return eigh(matrix)

    def observe(self, first, last):
        C, eps = realify(self, first, last)
        observed.append((first, last))
        assert C.shape == (size, last-first) and last-first < size
        return C, eps

    def forbidden(*args, **kwargs):
        raise AssertionError('compact audit assembled full canonical orbitals')

    monkeypatch.setattr(source_module._RealGammaOrbitalBlocks, 'realify', observe)
    monkeypatch.setattr(np.linalg, 'eigh', observe_eigh)
    monkeypatch.setattr(source_module, '_real_gamma_orbitals_from_columns', forbidden)
    monkeypatch.setattr(module, '_real_gamma_orbitals_from_columns', forbidden)
    monkeypatch.setattr(module, 'GammaReference', forbidden)
    for name in ('zeros', 'empty'):
        original = getattr(np, name)

        def bounded(shape, *args, _original=original, **kwargs):
            if isinstance(shape, tuple) and shape == (size, size):
                forbidden()
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(np, name, bounded)
    compact, record = module.bloch_translation_reference(system, basis, hf, mesh,
        n_frozen=model.n_cells, compact_occupied=True)
    monkeypatch.undo()  # The independent dense oracle below may expand projectors.
    assert not hasattr(compact, 'mo_coeffs')
    assert record['orbital_audit'] == 'replayed-complete-eigenspaces'
    assert record['orbital_audit_block_evaluations'] == len(observed) > len(set(observed))
    cache = record['orbital_audit_recipe_cache']
    assert cache['policy'] == 'first-fit-no-ao-coefficients'
    assert 0 < cache['stored_bytes'] <= cache['budget_bytes'] == np.asarray(hf.mo_coeffs).nbytes
    assert cache['realifications']+cache['hits'] == len(observed)
    assert len(decompositions) == 2*cache['realifications'] < 2*len(observed)
    if not flat_bands:
        assert cache['realifications'] == cache['entries'] == len(set(observed))
    else:
        assert cache['entries'] < len(set(observed))  # Wide recipes use uncached replay.
    assert record['max_folded_orbital_columns'] == max(j-i for i, j in observed)
    if flat_bands:
        assert record['max_folded_orbital_columns'] == model.n_cells > basis.nbasis
    for field in ('real_orthogonality_error', 'real_stationarity_error'):
        assert record[field] == pytest.approx(expected[field], abs=3e-14)
    np.testing.assert_allclose(compact.mo_energies, dense.mo_energies, atol=3e-14, rtol=0)
    ids = np.arange(size)
    for projector, selected in ((compact.occupied_projectors.frozen, slice(0, model.n_cells)),
                                (compact.occupied_projectors.active, slice(model.n_cells, 2*model.n_cells))):
        C = dense.mo_coeffs[:, selected]
        np.testing.assert_allclose(projector.block(ids, ids), C @ C.T @ model.overlap,
                                   atol=3e-12, rtol=0)
    record_property('real_eigenspace_columns', record['max_folded_orbital_columns'])
    record_property('canonical_columns', size)
    record_property('replayed_block_evaluations', len(observed))
    record_property('fresh_realifications', cache['realifications'])
    record_property('recipe_cache_bytes', cache['stored_bytes'])
    record_property('recipe_cache_budget_bytes', cache['budget_bytes'])
    for value, (saved, writable) in zip(arrays, before):
        np.testing.assert_array_equal(value, saved)
        assert value.flags.writeable == writable


@pytest.mark.parametrize('budget', [0, 39, 40, 80])
def test_realification_recipe_cache_bounds_and_detached_results(budget, monkeypatch):
    from vibeqc.dlpno.source import _RealGammaOrbitalBlocks

    eps = np.array([-1.5, -.8, .7, 1.3])
    S, F = np.eye(4), np.diag(eps)
    C = np.eye(4, dtype=complex)*np.exp(1j*np.arange(4)*.2)
    before = [value.copy() for value in (C, S, F, eps)]
    blocks = _RealGammaOrbitalBlocks(4, eps, lambda indices: C[:, indices], S, F,
                                     recipe_cache_bytes=budget)
    oracle = _RealGammaOrbitalBlocks(4, eps, lambda indices: C[:, indices], S, F)
    expected = [oracle.realify(i, j) for i, j in oracle.ranges]
    for i, j in blocks.ranges:
        coefficients, energies = blocks.realify(i, j)
        coefficients[:] = 42.
        energies[:] = 99.
    cached = min(budget//40, len(blocks.ranges))
    record = blocks.recipe_cache_record()
    assert record['entries'] == cached and record['stored_bytes'] == 40*cached <= budget
    assert record['realifications'] == len(blocks.ranges)
    # Only transform-size arrays are retained, with independent ownership.
    for recipe in blocks._recipes.values():
        assert [value.shape for value in recipe] == [(2, 1), (1,), (1, 1), (1,)]
        assert all(value.base is None and not value.flags.writeable for value in recipe)
    for (i, j), (C_expected, eps_expected) in zip(blocks.ranges, expected):
        with monkeypatch.context() as replay:
            if i < cached:
                def forbidden(*args, **kwargs):
                    raise AssertionError('cached recipe repeated an eigendecomposition')
                replay.setattr(np.linalg, 'eigh', forbidden)
            actual, energies = blocks.realify(i, j)
        np.testing.assert_array_equal(actual, C_expected)
        np.testing.assert_array_equal(energies, eps_expected)
    assert blocks.recipe_cache_record()['hits'] == cached
    assert blocks.recipe_cache_record()['stored_bytes'] == record['stored_bytes']
    record['stored_bytes'] = -1
    assert blocks.recipe_cache_record()['stored_bytes'] >= 0
    for actual, saved in zip((C, S, F, eps), before):
        np.testing.assert_array_equal(actual, saved)
    # A new calculation with the same orbital indices has its own recipes.
    shifted = _RealGammaOrbitalBlocks(4, eps+.2, lambda indices: C[:, indices], S, F+.2*S,
                                     recipe_cache_bytes=budget)
    _, energy = shifted.realify(0, 1)
    assert energy[0] == pytest.approx(eps[0]+.2)
    assert shifted.recipe_cache_record()['hits'] == 0


@pytest.mark.parametrize('fault', ['rank', 'eigenvalue'])
def test_realification_recipe_cache_never_retains_failed_decompositions(fault):
    from vibeqc.dlpno.source import _RealGammaOrbitalBlocks

    C = np.eye(2, dtype=complex)
    S, F = np.eye(2), np.diag([-1., 1.])
    if fault == 'rank':
        C[:, 1] = [1., 1j]
        message = 'do not span a real space'
    else:
        F[1, 1] += .1
        message = 'do not reproduce the SCF orbital energies'
    arrays = [C, S, F]
    before = [value.copy() for value in arrays]
    blocks = _RealGammaOrbitalBlocks(2, [-1., 1.], lambda indices: C[:, indices], S, F,
                                     recipe_cache_bytes=80)
    expected_C, expected_eps = blocks.realify(0, 1)
    for _ in range(2):
        with pytest.raises(ValueError, match=message):
            blocks.realify(1, 2)
        assert blocks.recipe_cache_record()['entries'] == 1
        assert blocks.recipe_cache_record()['stored_bytes'] == 40
    actual_C, actual_eps = blocks.realify(0, 1)
    np.testing.assert_array_equal(actual_C, expected_C)
    np.testing.assert_array_equal(actual_eps, expected_eps)
    for actual, saved in zip(arrays, before):
        np.testing.assert_array_equal(actual, saved)


@pytest.mark.parametrize('fault', ['cross_eigenspace', 'cross_panel', 'virtual_energy', 'frozen_mixing'])
def test_compact_bloch_audit_preserves_global_residuals_and_partition(fault, monkeypatch):
    from vibeqc.dlpno import translation_bloch as module

    mesh = (3, 1, 1)
    system, basis, hf, model = _bloch_translation_model(mesh)
    for k in range(model.n_cells):
        # A resolved but tiny frozen/active gap isolates the global Gram
        # and partition guards from the weaker stationarity defect.
        second = -1.5 if fault == 'cross_panel' else -1.5+2e-8
        hf.mo_energies[k] = np.array([-1.5, second, .7, 1.3])
        SC = hf.overlap[k] @ hf.mo_coeffs[k]
        hf.fock[k] = (SC*hf.mo_energies[k]) @ SC.conj().T
    nf = 0 if fault == 'cross_panel' else model.n_cells
    dense, _ = module.bloch_translation_reference(system, basis, hf, mesh, n_frozen=nf)
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    original = module._RealGammaOrbitalBlocks.realify

    def corrupt(self, i, j):
        C, eps = original(self, i, j)
        if fault == 'cross_eigenspace' and i == 3:
            C[:, 0] += 1e-5*dense.mo_coeffs[:, 0]
            # Every individual block remains admissible on its own.
            assert np.max(abs(C.T @ self.S.apply(C)-np.eye(j-i))) < 1e-8
        elif fault == 'cross_panel' and i == 0:
            C[:, 5] += 1e-5*C[:, 0]
        elif fault == 'virtual_energy' and j == self.n_orb:
            eps[-1] += 1e-5
        elif fault == 'frozen_mixing' and i in (0, 3):
            theta = .01
            other, sign = (3, 1) if i == 0 else (0, -1)
            C[:, 0] = np.cos(theta)*C[:, 0]+sign*np.sin(theta)*dense.mo_coeffs[:, other]
        if fault != 'virtual_energy':
            assert np.max(abs(self.F.apply(C)-self.S.apply(C)*eps)) < 1e-8
        return C, eps

    monkeypatch.setattr(module._RealGammaOrbitalBlocks, 'realify', corrupt)
    message = 'admission defects' if fault == 'frozen_mixing' else 'realified orbitals do not preserve'
    with pytest.raises(ValueError, match=message):
        module.bloch_translation_reference(system, basis, hf, mesh, n_frozen=nf, compact_occupied=True)
    for value, saved in zip(arrays, before):
        np.testing.assert_array_equal(value, saved)


def test_replayed_projector_audit_checks_nonhome_rows_and_entry_norm():
    from types import SimpleNamespace
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    from vibeqc.dlpno.translation_projectors import _replayed_projector_error

    home = np.zeros((3, 2, 2))
    home[0] = np.eye(2)
    S = CyclicAOOperator(home, (3, 1, 1))
    home[0, 1, 1] = 0
    projector = CyclicAOOperator(home, (3, 1, 1))
    C = np.eye(6)[:, [0, 2, 4]]
    # Change only a nonhome orbital; the first AO row tile agrees exactly.
    C[:, 1] = np.eye(6)[:, 3]
    blocks = SimpleNamespace(S=S, ranges=((0, 3),), realify=lambda i, j: (C[:, i:j], np.zeros(j-i)))
    difference = C @ C.T-projector.block(np.arange(6), np.arange(6))
    assert np.max(abs(difference[:2])) == 0
    assert _replayed_projector_error(blocks, projector, 0, 3) == np.max(abs(difference)) == 1
    with pytest.raises(ValueError, match='crosses the frozen/occupied partition'):
        _replayed_projector_error(blocks, projector, 0, 2)


@pytest.mark.parametrize('mesh,frozen', [((3, 1, 1), False), ((2, 1, 1), True),
                                        ((2, 2, 1), True)])
def test_bloch_translation_reference_matches_independent_finite_matrices(mesh, frozen, monkeypatch):
    from scipy.linalg import eigh
    from vibeqc import _vibeqc_core as core
    from vibeqc.dlpno.translation_bloch import bloch_translation_reference
    from vibeqc.dlpno.translation_operators import CyclicAOOperator, matrix_apply
    from vibeqc.periodic.chi import posthf

    system, basis, hf, model = _bloch_translation_model(mesh)
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    for value in arrays:
        value.setflags(write=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('Bloch adapter must not construct dense AO S/F matrices')

    monkeypatch.setattr(core, 'aiccm2026dev_b_matrix_to_real_supercell', forbidden)
    monkeypatch.setattr(posthf, 'aiccm2026dev_b_matrix_to_real_supercell', forbidden)
    monkeypatch.setattr(posthf, '_periodic_matrix_to_real_supercell', forbidden)
    n_frozen = model.n_cells if frozen else 0
    reference, record = bloch_translation_reference(system, basis, hf, mesh, n_frozen=n_frozen)
    assert type(reference.overlap) is type(reference.fock) is CyclicAOOperator
    probe = np.random.default_rng(91).normal(size=(len(model.overlap), 3))
    for actual, expected in ((reference.overlap, model.overlap), (reference.fock, model.fock)):
        np.testing.assert_allclose(matrix_apply(actual, probe), expected @ probe, atol=3e-13, rtol=0)
        assert actual.blocks.shape == (model.n_cells, 4, 4)
        assert not actual.blocks.flags.writeable
    C = reference.mo_coeffs
    np.testing.assert_allclose(C.T @ model.overlap @ C, np.eye(len(C)), atol=3e-12, rtol=0)
    np.testing.assert_allclose(model.fock @ C, (model.overlap @ C)*reference.mo_energies,
                               atol=3e-12, rtol=0)
    expected_eps, expected_C = eigh(model.fock, model.overlap)
    np.testing.assert_allclose(reference.mo_energies, expected_eps, atol=3e-12, rtol=0)
    n_occ = 2*model.n_cells
    for selected in (slice(0, n_frozen), slice(n_frozen, n_occ)):
        np.testing.assert_allclose(C[:, selected] @ C[:, selected].T @ model.overlap,
            expected_C[:, selected] @ expected_C[:, selected].T @ model.overlap, atol=3e-12, rtol=0)
    assert reference.energy == pytest.approx(hf.energy*model.n_cells, abs=1e-14)
    assert record
    for value, original in zip(arrays, before):
        np.testing.assert_array_equal(value, original)
        assert not value.flags.writeable


@pytest.mark.parametrize('fault', ['weights', 'kfrac', 'kcart', 'joint_permutation',
                                 'smearing', 'nonfinite', 'nonhermitian', 'time_reversal',
                                 'coefficients', 'fractional_occupations', 'frozen_band'])
def test_bloch_translation_reference_refuses_inconsistent_state_without_mutation(fault):
    from scipy.linalg import eigh
    from vibeqc.dlpno.translation_bloch import bloch_translation_reference

    mesh = (3, 1, 1)
    system, basis, hf, _ = _bloch_translation_model(mesh)
    n_frozen = 0
    message = None
    if fault == 'weights':
        hf.kpoint_weights[:] = [.4, .3, .3]
    elif fault == 'kfrac':
        hf.kpoints_frac[1, 0] += .125
    elif fault == 'kcart':
        hf.kpoints_cart[[1, 2]] = hf.kpoints_cart[[2, 1]]
    elif fault == 'joint_permutation':
        for name in ('kpoints_frac', 'kpoints_cart', 'kpoint_weights', 'occupations'):
            setattr(hf, name, getattr(hf, name)[[0, 2, 1]])
        for name in ('overlap', 'fock', 'mo_coeffs', 'mo_energies', 'density'):
            setattr(hf, name, [getattr(hf, name)[k] for k in (0, 2, 1)])
    elif fault == 'smearing':
        hf.smearing_temperature = .01
    elif fault == 'nonfinite':
        hf.fock[0][0, 0] = np.nan
    elif fault == 'nonhermitian':
        hf.fock[0][0, 1] += .01
    elif fault == 'time_reversal':
        # Each block remains Hermitian and its orbitals remain stationary,
        # so refusal must concern the lost real AO representation itself.
        hf.fock[1][0, 0] += .01
        hf.mo_energies[1], hf.mo_coeffs[1] = eigh(hf.fock[1], hf.overlap[1])
        occupied = hf.mo_coeffs[1][:, :2]
        hf.density[1] = 2*occupied @ occupied.conj().T
        message = 'real time-reversal-compatible'
    elif fault == 'coefficients':
        hf.mo_coeffs[0][:, 0] *= 1.1
    elif fault == 'fractional_occupations':
        hf.occupations[0, :2] = [1.9, 2.1]
    else:
        n_frozen = 1
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
        bloch_translation_reference(system, basis, hf, mesh, n_frozen=n_frozen)
    for value, original in zip(arrays, before):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize('smallest', [-.1, 0., 1e-12])
def test_bloch_translation_reference_checks_non_gamma_metric_rank(smallest):
    from vibeqc.dlpno.translation_bloch import bloch_translation_reference

    mesh = (3, 1, 1)
    system, basis, hf, _ = _bloch_translation_model(mesh)
    for k in range(3):
        diagonal = np.array([smallest if k else 1., 1., 1., 1.])
        hf.overlap[k] = np.diag(diagonal).astype(complex)
        hf.mo_energies[k] = np.array([-1.6, -1.1, .7, 1.3])
        hf.fock[k] = np.diag(diagonal*hf.mo_energies[k]).astype(complex)
        # A positive metric below the rank cutoff still has an exact
        # generalized eigensystem. A nonpositive metric cannot supply an
        # S-orthonormal frame and must fail this earlier rank guard.
        normalizer = np.where(diagonal > 0, diagonal, 1.)
        hf.mo_coeffs[k] = np.diag(1/np.sqrt(normalizer)).astype(complex)
        occupied = hf.mo_coeffs[k][:, :2]
        hf.density[k] = 2*occupied @ occupied.conj().T
    assert np.linalg.eigvalsh(hf.overlap[0]).min() == 1.
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    with pytest.raises(ValueError, match='linearly dependent'):
        bloch_translation_reference(system, basis, hf, mesh)
    for value, original in zip(arrays, before):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize('partition', ['occupied_virtual', 'frozen_active'])
def test_bloch_translation_reference_refuses_global_band_crossing(partition):
    from vibeqc.dlpno.translation_bloch import bloch_translation_reference

    mesh = (3, 1, 1)
    system, basis, hf, _ = _bloch_translation_model(mesh)
    if partition == 'occupied_virtual':
        hf.mo_energies[0] = np.array([-1.6, .8, 1.1, 1.5])
        n_frozen, message = 0, 'gap between its occupied and virtual bands'
    else:
        hf.mo_energies[0] = np.array([-1.05, -.95, .7, 1.3])
        n_frozen, message = 3, 'frozen and active occupied bands have no global gap'
    SC = hf.overlap[0] @ hf.mo_coeffs[0]
    hf.fock[0] = (SC*hf.mo_energies[0]) @ SC.conj().T
    for S, F, C, eps in zip(hf.overlap, hf.fock, hf.mo_coeffs, hf.mo_energies):
        assert np.all(np.diff(eps) > 0)
        np.testing.assert_allclose(C.conj().T @ S @ C, np.eye(4), atol=2e-13, rtol=0)
        np.testing.assert_allclose(F @ C, (S @ C)*eps, atol=2e-13, rtol=0)
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    with pytest.raises(ValueError, match=message):
        bloch_translation_reference(system, basis, hf, mesh, n_frozen=n_frozen)
    for value, original in zip(arrays, before):
        np.testing.assert_array_equal(value, original)


def test_bloch_translation_interleaved_sources_bind_imaginary_components():
    from copy import deepcopy
    from vibeqc.dlpno.source import source_identity
    from vibeqc.dlpno.translation_bloch import bloch_translation_reference

    mesh = (3, 1, 1)
    system, basis, hf, _ = _bloch_translation_model(mesh)
    other = deepcopy(hf)
    # Conjugating every Bloch operator and orbital preserves a valid real
    # Hamiltonian but reverses its off-home AO blocks. Real components alone
    # cannot distinguish these two independent finite sources.
    for name in ('overlap', 'fock', 'mo_coeffs', 'density'):
        setattr(other, name, [value.conj() for value in getattr(other, name)])
    first, first_record = bloch_translation_reference(system, basis, hf, mesh)
    saved = [value.copy() for value in (first.overlap.blocks, first.fock.blocks, first.mo_coeffs)]
    second, second_record = bloch_translation_reference(system, basis, other, mesh)
    for key in ('overlap_bloch', 'fock_bloch', 'orbitals_bloch'):
        assert first_record[key]['real'] == second_record[key]['real']
        assert first_record[key]['imaginary'] != second_record[key]['imaginary']
    assert source_identity(first_record) != source_identity(second_record)
    assert not np.allclose(first.overlap.blocks, second.overlap.blocks)
    for actual, expected in zip((first.overlap.blocks, first.fock.blocks, first.mo_coeffs), saved):
        np.testing.assert_array_equal(actual, expected)
    repeated, repeated_record = bloch_translation_reference(system, basis, hf, mesh)
    assert source_identity(repeated_record) == source_identity(first_record)
    np.testing.assert_array_equal(repeated.overlap.blocks, first.overlap.blocks)


@pytest.mark.parametrize('mesh,frozen', [((3, 1, 1), False), ((2, 2, 1), True)])
@pytest.mark.parametrize('compact_occupied', [False, True])
def test_bloch_translation_source_solver_matches_unreduced_canonical_mp2(
    mesh, frozen, compact_occupied, monkeypatch, record_property,
):
    from types import SimpleNamespace
    from scipy.linalg import eigh
    from vibeqc import _vibeqc_core as core
    from vibeqc.dlpno.source import CorrelationSource, SourceConventions
    from vibeqc.dlpno.translation_bloch import bloch_translation_reference
    from vibeqc.dlpno.translation_driver import run_translation_dlpno_mp2
    from vibeqc.dlpno.translation_factors import TranslationHomeFactors

    system, basis, hf, model = _bloch_translation_model(mesh)
    nk = model.n_cells
    n_frozen = nk if frozen else 0
    reference, _ = bloch_translation_reference(system, basis, hf, mesh, n_frozen=n_frozen,
                                               compact_occupied=compact_occupied)
    source = CorrelationSource(
        kind='bloch-finite-model', identity='bloch-finite-model', reference_kind='rhf',
        molecule=SimpleNamespace(multiplicity=1, n_electrons=lambda: 4*nk),
        basis=SimpleNamespace(nbasis=4*nk), reference=reference, overlap=reference.overlap,
        factors=TranslationHomeFactors(model.ao_factors[:1], mesh),
        conventions=SourceConventions(kmesh=mesh, exchange_q0='test-finite',
            bloch_normalisation='real-bvk-supercell', same_source=False), provenance={},
    )

    def forbidden(*args, **kwargs):
        raise AssertionError('block source reached the dense AO solver')

    monkeypatch.setattr(core, 'dlpno_translation_mp2_solve', forbidden)
    result = run_translation_dlpno_mp2(source, mesh, n_frozen=n_frozen,
        home_atom_aos=([0], [1], [2], [3]), conv_tol_energy=1e-13, conv_tol_residual=1e-12)
    eps, C = eigh(model.fock, model.overlap)
    no = 2*nk
    ov = np.einsum('Pmn,mi,na->Pia', model.ao_factors, C[:, n_frozen:no], C[:, no:])
    K = np.einsum('Pia,Pjb->ijab', ov, ov)
    eo, ev = eps[n_frozen:no], eps[no:]
    denominator = eo[:, None, None, None]+eo[None, :, None, None]-ev[None, None, :, None]-ev[None, None, None, :]
    T = K/denominator
    expected = np.sum(K*(2*T-T.swapaxes(2, 3)))/nk
    record_property('e_corr_per_cell', result.solution.e_corr_per_cell)
    record_property('unreduced_e_corr_per_cell', expected)
    record_property('absolute_energy_error', abs(result.solution.e_corr_per_cell-expected))
    record_property('maximum_raw_residual', result.solution.max_residual)
    record_property('evaluated_states', result.solution.n_iter)
    assert result.solution.converged
    assert result.solution.e_corr_per_cell == pytest.approx(expected, abs=3e-11)
    assert result.solution.max_residual < 1e-12
    assert result.solution.provenance['transport_preparation']['metric_backend'] == 'cyclic-blocks'


@pytest.mark.parametrize('mesh,frozen', [((3, 1, 1), False), ((3, 1, 1), True), ((2, 2, 1), True)])
def test_compact_occupied_projectors_preserve_seeds_partitions_and_owned_state(mesh, frozen, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    from vibeqc.dlpno import translation_bloch as module
    from vibeqc.dlpno import translation_source as source_module
    from vibeqc.dlpno.source import CorrelationSource, SourceConventions
    from vibeqc.dlpno.translation_operators import CyclicAOOperator
    from vibeqc.dlpno.translation_projectors import projector_seed_metric
    from vibeqc.dlpno.translation_factors import TranslationHomeFactors
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    from vibeqc.dlpno.translation_source import admit_translation_source

    system, basis, hf, model = _bloch_translation_model(mesh)
    arrays = _bloch_state_arrays(hf)
    before = [value.copy() for value in arrays]
    nk, size = model.n_cells, len(model.overlap)
    nf = nk if frozen else 0
    dense, _ = module.bloch_translation_reference(system, basis, hf, mesh, n_frozen=nf)
    compact, record = module.bloch_translation_reference(system, basis, hf, mesh,
        n_frozen=nf, compact_occupied=True)
    assert not hasattr(compact, 'mo_coeffs')
    projectors = compact.occupied_projectors
    assert record['occupied_projectors'] == projectors.record()
    with pytest.raises(TypeError, match='canonical coefficients'):
        np.asarray(projectors)
    with pytest.raises(AttributeError):
        projectors.n_frozen = 0
    indices = np.arange(size)
    for operator, columns in ((projectors.active, slice(nf, 2*nk)),
                              (projectors.frozen, slice(0, nf))):
        C = dense.mo_coeffs[:, columns]
        Q = operator.block(indices, indices)
        np.testing.assert_allclose(Q, C @ C.T @ model.overlap, atol=3e-12, rtol=0)
        seed_metric = projector_seed_metric(operator, compact.overlap)
        np.testing.assert_allclose(seed_metric.block(indices, indices), Q.T @ model.overlap @ Q,
                                   atol=3e-12, rtol=0)
        with pytest.raises(ValueError):
            operator.blocks.setflags(write=True)
    # The exact seed Gram must also retain a small idempotence defect,
    # rather than treating a numerically admitted projector as exact.
    perturbed = CyclicAOOperator(projectors.active.blocks*(1+2e-9), mesh)
    Q = perturbed.block(indices, indices)
    metric = projector_seed_metric(perturbed, compact.overlap).block(indices, indices)
    np.testing.assert_allclose(metric, Q.T @ model.overlap @ Q, atol=3e-12, rtol=0)
    assert np.max(abs(metric-model.overlap @ Q)) > 1e-10

    source = CorrelationSource(kind='bloch-model', identity='bloch-model', reference_kind='rhf',
        molecule=SimpleNamespace(multiplicity=1, n_electrons=lambda: 4*nk),
        basis=SimpleNamespace(nbasis=size), reference=dense, overlap=dense.overlap,
        factors=TranslationHomeFactors(model.ao_factors[:1], mesh),
        conventions=SourceConventions(kmesh=mesh, exchange_q0='test-finite',
            bloch_normalisation='real-bvk-supercell', same_source=False), provenance={})
    expected = select_translation_occupied(source, mesh, n_frozen=nf)
    source = replace(source, reference=compact, overlap=compact.overlap)

    def forbidden(*args, **kwargs):
        raise AssertionError('compact source reconstructed canonical orbitals')

    monkeypatch.setattr(module, '_real_gamma_orbitals_from_columns', forbidden)
    monkeypatch.setattr(source_module, '_projector_record', forbidden)
    actual = select_translation_occupied(source, mesh, n_frozen=nf)
    assert actual.active_seed_aos == expected.active_seed_aos
    assert actual.frozen_seed_aos == expected.frozen_seed_aos
    for name in ('occupied', 'frozen_occupied'):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), atol=3e-12, rtol=0)
    saved = projectors.record()
    # Bind strict consumer tolerances to the original canonical audit, and
    # reject a replacement Fock even if the occupied union is unchanged.
    for changed, message in ((projectors._replace(orthogonality_error=2e-10), 'tolerance'),
                              (projectors._replace(fock_identity='changed'), 'Fock'),
                              (projectors._replace(n_occupied=2*nk+1), 'electron count')):
        invalid = replace(source, reference=replace(compact, occupied_projectors=changed))
        with pytest.raises(ValueError, match=message):
            select_translation_occupied(invalid, mesh, n_frozen=nf)
    if frozen:
        # This rotation preserves the total occupied projector, but mixes
        # frozen and active bands. Admission must refuse each partition.
        active = (actual.occupied+actual.frozen_occupied)/np.sqrt(2)
        core = (actual.frozen_occupied-actual.occupied)/np.sqrt(2)
        with pytest.raises(ValueError, match='active occupied projector'):
            admit_translation_source(source, mesh, active, frozen_occupied=core)
        with pytest.raises(ValueError, match='frozen partition'):
            select_translation_occupied(source, mesh, n_frozen=0)
    assert projectors.record() == saved
    for value, original in zip(arrays, before):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize('fit_source', ['classic', 'auto'])
@pytest.mark.parametrize('failure', [None, 'scf', 'home_transform'])
def test_chi_translation_adapter_preserves_owned_options_and_source_provenance(
    fit_source, failure, monkeypatch,
):
    from dataclasses import replace
    from types import SimpleNamespace
    import vibeqc as vq
    from vibeqc.periodic.chi import translation
    from vibeqc.periodic.chi.reference import _digest
    from vibeqc.periodic.chi.scf import AICCM2026DevBFiniteTorusConvention
    from vibeqc.periodic_fit_source import RSGDFFitSource
    from vibeqc.dlpno.translation_operators import CyclicAOOperator

    mesh = (3, 1, 1)
    system, basis, hf, model = _bloch_translation_model(mesh)
    convention = AICCM2026DevBFiniteTorusConvention(
        coulomb_kernel='3d-periodic-g0', exchange_q0='bvk-ewald',
        exchange_q0_applicability='active', boundary_model='3d-periodic',
        periodic_dimension=3, character_mesh_shape=mesh,
        bvk_madelung_supercell_repetitions=mesh,
        bvk_madelung_supercell_lattice_bohr=tuple(map(tuple, np.asarray(system.lattice) @ np.diag(mesh))),
    )
    scf_fit = RSGDFFitSource(
        engine='finite-model', engine_version=1,
        source='classic' if fit_source == 'classic' else 'sr',
        source_signature='finite-model', orbital_basis=basis.name,
        orbital_basis_sha256='orbital-model', auxiliary_basis='model-aux',
        auxiliary_basis_sha256='auxiliary-model', numerical_support=(), metric_threshold=1e-9,
        auxiliary_frame='canonical-modrho', q_grouping=(),
        kpoints_cart=tuple(map(tuple, hf.kpoints_cart)), base_reciprocal_cutoff_ha=200.,
        executed_tail_reciprocal_cutoff_ha=None, factor_shapes=(), factor_sha256='scf-model',
    )
    correlation_fit = scf_fit if fit_source == 'classic' else replace(
        scf_fit, source='classic', factor_sha256='correlation-refit-model')
    hf.executed_fit_source = scf_fit
    hf.exact_exchange_assembly = SimpleNamespace(c_full=1., c_sr=0.)
    hf.aiccm2026dev_b = SimpleNamespace(electronic_method='RHF', backend='ri',
        final_scf_delta_e_ha=0., finite_torus_convention=convention)
    options = vq.PeriodicRHFOptions()
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-9
    options.lattice_opts.cutoff_bohr = 15.
    options_before = _digest(options)
    arrays = _bloch_state_arrays(hf) + [model.ao_factors]
    before = [value.copy() for value in arrays]
    for value in arrays:
        value.setflags(write=False)
    calls = []

    def scf(actual_system, actual_basis, actual_mesh, actual_options, **kwargs):
        assert actual_system is system and actual_basis is basis and actual_mesh == mesh
        assert actual_options is not options
        assert _digest(actual_options) == options_before
        assert kwargs['fit_source'] == fit_source and kwargs['backend'] == 'ri'
        actual_options.max_iter = 7
        actual_options.lattice_opts.cutoff_bohr = 19.
        calls.append('scf')
        if failure == 'scf':
            raise RuntimeError('model SCF failure')
        return hf

    cache = {(0, 0): np.ones((1, 4, 4))}

    def selected(*args, reference, **kwargs):
        # The shared selector owns retained-factor attestation. This fixture
        # exercises the adapter's two admitted provenance branches.
        assert reference is hf and kwargs['fit_source'] == fit_source
        assert kwargs['lattice_cutoff_bohr'] == 15.
        calls.append('selected')
        if fit_source == 'auto':
            reference._chi_correlation_fit_source = correlation_fit
        return cache, object()

    def home(actual_cache, kfrac, actual_mesh):
        assert actual_cache is cache and actual_mesh == mesh
        np.testing.assert_array_equal(kfrac, hf.kpoints_frac)
        calls.append('home')
        if failure == 'home_transform':
            raise RuntimeError('model home transform failure')
        return model.ao_factors[:1], 0., 0.

    def forbidden(*args, **kwargs):
        raise AssertionError('block adapter entered a dense reference/factor path')

    monkeypatch.setattr(translation.p, 'run_aiccm2026dev_b_rhf', scf)
    monkeypatch.setattr(translation.p, '_selected_lpq_cache', selected)
    monkeypatch.setattr(translation.p, '_lpq_cache_to_real_home_auxiliary', home)
    monkeypatch.setattr(translation.p, '_build_b_real_reference', forbidden)
    monkeypatch.setattr(translation.p, '_periodic_matrix_to_real_supercell', forbidden)
    monkeypatch.setattr(translation.p, '_lpq_cache_to_real_supercell', forbidden)
    monkeypatch.setattr(translation, 'make_aux_basis_set', lambda *a, **k: SimpleNamespace(nbasis=model.n_cells))
    kwargs = dict(fit_source=fit_source, aux_basis='model-aux', n_frozen=model.n_cells, progress=False)
    if failure:
        with pytest.raises(RuntimeError, match='model .* failure'):
            translation.prepare_chi_translation_source(system, basis, mesh, options, **kwargs)
    else:
        source = translation.prepare_chi_translation_source(system, basis, mesh, options, **kwargs)
        assert source.kind == 'chi-translation-blocks'
        assert type(source.overlap) is type(source.reference.fock) is CyclicAOOperator
        assert source.conventions.same_source is (fit_source == 'classic')
        assert source.conventions.exchange_q0 == 'bvk-ewald'
        assert source.conventions.kmesh == mesh
        assert source.factors.home_factors.shape == (1, 12, 12)
        adapter = source.provenance['adapter']
        assert adapter['scf_fit'] == scf_fit.identity
        assert adapter['correlation_fit'] == correlation_fit.identity
        assert adapter['correlation_fit_status'] == ('reused' if fit_source == 'classic' else 'built')
        assert adapter['quantitative_status'] == 'not-qualified'
        assert adapter['scf_options_identity'] == options_before
        assert adapter['n_frozen'] == model.n_cells
    assert calls == (['scf'] if failure == 'scf' else ['scf', 'selected', 'home'])
    assert _digest(options) == options_before
    for value, original in zip(arrays, before):
        np.testing.assert_array_equal(value, original)
        assert not value.flags.writeable
