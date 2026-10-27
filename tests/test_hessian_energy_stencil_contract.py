"""Pure arithmetic/control tests; scientific Hessian acceptance runs through vq."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def implementation():
    path = Path(__file__).parents[1] / 'python/vibeqc/hessian.py'
    tree = ast.parse(path.read_text())
    names = {'_full_energy_ks_gradient', '_scf_and_gradient_at_impl'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(np=np, Molecule=object, _KS_ENERGY_GRADIENT_STEP_BOHR=1e-3)
    # Only the pure stencil is evaluated here; source parsing also checks the
    # complete-energy branch wiring without importing a native QC runtime.
    node = next(n for n in functions if n.name == '_full_energy_ks_gradient')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, functions


@pytest.mark.parametrize('method', ['RKS', 'UKS'])
def test_signed_full_energy_stencil_and_original_option_transport(method):
    ns, _ = implementation()
    xyz = np.array([[0.2, -0.4, 0.1], [0.5, 0.3, -0.2]])
    mol = SimpleNamespace(atoms=[object(), object()], xyz=xyz)
    matrix = np.arange(36, dtype=float).reshape(6, 6)
    matrix = (matrix + matrix.T) / 100 + np.eye(6)
    options, grid, gradient = object(), object(), object()
    calls = []

    def displaced(original, atom, axis, delta):
        moved = original.xyz.copy()
        moved[atom, axis] += delta
        return SimpleNamespace(atoms=original.atoms, xyz=moved)

    def energy(moved, basis, route, opts, grid_opts, grad_opts, **kwargs):
        assert (basis, route, opts, grid_opts, grad_opts) == ('basis', method, options, grid, gradient)
        assert kwargs == dict(reference_mol=mol, energy_only=True)
        calls.append(moved.xyz.copy())
        vector = moved.xyz.ravel()
        return None, SimpleNamespace(energy=float(vector @ matrix @ vector / 2)), None

    ns.update(_displaced_molecule=displaced, _scf_and_gradient_at=energy)
    actual = ns['_full_energy_ks_gradient'](mol, 'basis', method, options, grid, gradient)
    np.testing.assert_allclose(actual.ravel(), matrix @ xyz.ravel(), atol=1e-12, rtol=0)
    assert len(calls) == 12
    np.testing.assert_array_equal(mol.xyz, xyz)
    for k in range(6):
        expected = np.zeros(6)
        expected[k] = 1e-3
        np.testing.assert_allclose((calls[2*k+1]-calls[2*k]).ravel(), 2*expected, atol=1e-15)


def test_inner_scf_failure_propagates_without_partial_gradient():
    ns, _ = implementation()
    ns['_displaced_molecule'] = lambda *args: object()
    def fail(*args, **kwargs):
        raise RuntimeError('inner SCF did not converge')
    ns['_scf_and_gradient_at'] = fail
    with pytest.raises(RuntimeError, match='inner SCF'):
        ns['_full_energy_ks_gradient'](SimpleNamespace(atoms=[object()]), 'basis', 'UKS', None, None, None)


def test_energy_only_branches_refuse_nonfinite_before_gradient_kernel():
    _, nodes = implementation()
    impl = next(n for n in nodes if n.name == '_scf_and_gradient_at_impl')
    branches = [n for n in ast.walk(impl) if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'energy_only']
    assert len(branches) == 2
    for branch in branches:
        for energy in [np.nan, np.inf, -np.inf]:
            ns = dict(np=np, result=SimpleNamespace(energy=energy))
            guard = branch.body[0]
            with pytest.raises(ValueError, match='non-finite'):
                exec(compile(ast.Module(body=[guard], type_ignores=[]), '<energy guard>', 'exec'), ns)
