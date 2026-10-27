"""Source-isolated Gamma initialization checks; no native build or SCF required."""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "python/vibeqc"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def policy(monkeypatch):
    density = ModuleType("_gamma_seed_fixture.periodic_k_density")
    density.np = np
    definitions(SOURCE / "periodic_k_density.py", {
        "_FermiDensityDomainError", "_fermi_density_entropy", "_needs_fermi_seed_step",
    }, density.__dict__)
    monkeypatch.setitem(sys.modules, density.__name__, density)
    namespace = definitions(SOURCE / "pbc_gdf.py", {
        "_open_shell_gamma_needs_fermi_seed_step",
    }, {"__package__": "_gamma_seed_fixture"})
    return namespace["_open_shell_gamma_needs_fermi_seed_step"], density


def test_only_thermal_iterating_out_of_domain_seeds_request_refill(policy):
    needs, density = policy
    identity = np.eye(2)
    valid = np.diag([.75, .25])
    invalid = np.diag([1.2, -.2])
    assert needs((invalid, valid), identity, identity, .01, 10)
    assert needs((valid, invalid), identity, identity, .01, 10)
    assert not needs((valid, valid), identity, identity, .01, 10)
    assert not needs((invalid, valid), identity, identity, 0., 10)
    assert not needs((invalid, valid), identity, identity, .01, 0)
    with pytest.raises(density._FermiDensityDomainError):
        density._fermi_density_entropy([invalid], [identity], [identity], [1.], 1.)
    malformed = np.array([[1., .2], [0., 0.]])
    with pytest.raises(ValueError, match="Hermitian"):
        needs((malformed, valid), identity, identity, .01, 10)
    np.testing.assert_array_equal(invalid, np.diag([1.2, -.2]))


@pytest.mark.parametrize("name", ["run_pbc_gdf_uhf", "run_pbc_gdf_uks"])
def test_real_driver_seed_branch_refills_selected_focks_without_acceptance(name):
    tree = ast.parse((SOURCE / "pbc_gdf.py").read_text())
    driver = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    loop = next(n for n in driver.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "iter_idx")
    seed = next(n for n in loop.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "iter_idx == 0")
    assert loop.body.index(seed) < next(i for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id == "E_elec" for t in n.targets))
    calls = []
    fa, fb = np.diag([2., 3.]), np.diag([4., 5.])
    def diagonalize(f):
        calls.append(f.copy())
        return np.eye(2), np.diag(f)
    def occupy(ca, ea, cb, eb, na, nb, options):
        np.testing.assert_array_equal(ea, [2., 3.])
        np.testing.assert_array_equal(eb, [4., 5.])
        return np.diag([.7, .3]), np.diag([.8, .2]), ea, eb, 0., 0., 1.
    namespace = dict(diagonalise=diagonalize, diagonalise_beta=diagonalize,
        F_alpha=fa, F_beta=fb, n_alpha=1, n_beta=1, smear_opts=object(),
        _open_shell_gamma_occupy=occupy, plog=SimpleNamespace(info=lambda _: None))
    # Execute the actual source branch in its loop context. The continuation must
    # bypass all energy/entropy/DIIS acceptance, not just hide an exception.
    probe = ast.For(target=ast.Name(id="iter_idx", ctx=ast.Store()),
        iter=ast.List(elts=[ast.Constant(value=0)], ctx=ast.Load()),
        body=[seed, ast.Raise(exc=ast.Call(func=ast.Name(id="AssertionError", ctx=ast.Load()),
            args=[ast.Constant(value="seed accepted as an iterate")], keywords=[]), cause=None)], orelse=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])),
                 "driver-seed-branch", "exec"), namespace)
    assert len(calls) == 2
    assert namespace["D_alpha_prev"] is not namespace["D_alpha"]
    np.testing.assert_array_equal(namespace["D_alpha"], np.diag([.7, .3]))
    np.testing.assert_array_equal(namespace["eps_beta_f"], [4., 5.])
