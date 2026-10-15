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
def test_bloch_translation_source_solver_matches_unreduced_canonical_mp2(
    mesh, frozen, monkeypatch, record_property,
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
    reference, _ = bloch_translation_reference(system, basis, hf, mesh, n_frozen=n_frozen)
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
