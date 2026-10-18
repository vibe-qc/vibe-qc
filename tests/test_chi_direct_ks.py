"""Bounded tests for direct KS XC assembly, without native integral builds."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "python" / "vibeqc"


def _fock_builder(spin, evaluator):
    """Execute the production nested assembler with a mock Coulomb builder."""
    tree = ast.parse((_SOURCE / f"pbc_bipole_{spin}.py").read_text())
    driver = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == f"run_pbc_bipole_{spin}")
    function = next(n for n in driver.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_build_fock_for_density")
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    lattice = SimpleNamespace(blocks=[np.eye(2)], cells=[0])
    fb = SimpleNamespace(f2e_real=lattice, f2e_alpha_real=lattice,
        f2e_beta_real=lattice, k_corr_per_k=None, k_corr_alpha_per_k=None,
        e_j_short_range=1.0, e_j_long_range=0.2, e_exchange=0.0,
        e_j_multipole=None, e_2e_k_correction=0.0,
        e_exchange_finite_size=None, screened_exchange_execution=None,
        output_cell_farming_execution=None)
    def forbidden(*args, **kwargs):
        raise AssertionError("native XC must not run with the supplied callback")
    namespace = dict(np=np, _chi_xc=evaluator, _fock_ctx=None,
        alpha_hf=0.0, exx=None, n_alpha=1, n_beta=1,
        incremental_jk_alpha=None, incremental_jk_beta=None,
        incremental_jk_total=None, k_points=np.zeros((2, 3)),
        Hcore_k_list=[np.diag([2., 3.])] * 2,
        build_bipole_restricted_fock=lambda *a, **k: fb,
        build_bipole_unrestricted_fock=lambda *a, **k: fb,
        _bloch_sum_blocks_multi_k=lambda *a: [np.eye(2)] * 2,
        _density_set_gamma_or_lattice=forbidden,
        build_xc_periodic=forbidden, build_xc_periodic_uks=forbidden,
        _PBCBipoleRKSFockBuild=SimpleNamespace,
        _PBCBipoleUKSFockBuild=SimpleNamespace)
    exec(compile(ast.fix_missing_locations(module), "<direct KS assembler>", "exec"), namespace)
    return namespace[function.name], lattice


@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_callback_receives_complex_k_density_and_supplies_same_energy_and_fock(spin):
    density = np.array([[[1., .3j], [-.3j, .5]],
                        [[.8, -.2j], [.2j, .7]]])
    seen = []
    def restricted(d):
        seen.append(np.asarray(d).copy())
        return 4.5, 2.0 * d
    def unrestricted(a, b):
        seen.append(np.array([a, b]))
        return 4.5, 2.0 * a, 3.0 * b
    evaluator = SimpleNamespace(restricted=restricted, unrestricted=unrestricted)
    builder, lattice = _fock_builder(spin, evaluator)
    if spin == "rks":
        result = builder(lattice, coeffs_for_rho=None, xc_density_k=density)
        focks, expected, supplied = [result.f_k_list], [2.0 * density], density
    else:
        supplied = np.array([density, .5 * density])
        result = builder(lattice, lattice, coeffs_alpha_for_rho=None,
                         coeffs_beta_for_rho=None, xc_density_k=supplied)
        focks = [result.f_alpha_k_list, result.f_beta_k_list]
        expected = [2.0 * density, 1.5 * density]
    np.testing.assert_array_equal(seen[0], supplied)
    assert result.e_xc == 4.5
    assert result.e_j_long_range == .2
    for actual, potential in zip(focks, expected):
        np.testing.assert_allclose(actual, potential + np.diag([3., 4.]))


@pytest.mark.parametrize("spin", ["rks", "uks"])
def test_callback_cannot_silently_reconstruct_density_from_lattice(spin):
    builder, lattice = _fock_builder(spin, SimpleNamespace())
    with pytest.raises(RuntimeError, match="actual per-k"):
        if spin == "rks":
            builder(lattice, coeffs_for_rho=None)
        else:
            builder(lattice, lattice, coeffs_alpha_for_rho=None,
                    coeffs_beta_for_rho=None)


@pytest.mark.parametrize("spin", ["rks", "uks"])
@pytest.mark.parametrize("damped", [False, True])
def test_xc_density_follows_driver_damping_and_committed_update(spin, damped):
    tree = ast.parse((_SOURCE / f"pbc_bipole_{spin}.py").read_text())
    driver = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == f"run_pbc_bipole_{spin}")
    loop = next(n for n in driver.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "iter_idx")
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_chi_used" for t in n.targets))
    commit = next(n for n in loop.body if isinstance(n, ast.If)
                  and any(isinstance(t, ast.Assign) and
                          any(isinstance(a, ast.Name) and a.id == "_chi_previous"
                              for a in t.targets) for t in n.body))
    initial = np.array([[[1., .3j], [-.3j, .5]],
                        [[.8, -.2j], [.2j, .7]]])
    previous = initial.conj()
    if spin == "uks":
        initial = np.array([initial, .5 * initial])
        previous = np.array([previous, .5 * previous])
    candidate = 1.2 * initial
    namespace = dict(_chi_density=initial, _chi_previous=previous,
                     _chi_xc=object(), d_used_is_damped=damped, damping=.3,
                     _chi_new=candidate, use_oda=False)
    statements = ast.Module(body=[*loop.body[start:start + 2], commit], type_ignores=[])
    exec(compile(ast.fix_missing_locations(statements), "<density update>", "exec"), namespace)
    expected = .7 * initial + .3 * previous if damped else initial
    np.testing.assert_allclose(namespace["_chi_used"], expected)
    np.testing.assert_allclose(namespace["_chi_previous"], expected)
    np.testing.assert_allclose(namespace["_chi_density"], candidate)


def _public_chi_statements():
    tree = ast.parse((_SOURCE / "periodic_runner.py").read_text())
    driver = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "run_periodic_job")
    preflight = next(n for n in driver.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Call)
                     and any(isinstance(a, ast.Name) and a.id == "chi_grid_options"
                             for a in ast.walk(n.test)))
    assignment = next(n for n in driver.body if isinstance(n, ast.If)
                      and any(isinstance(a, ast.Assign) and
                              any(isinstance(t, ast.Attribute) and t.attr == "grid"
                                  and isinstance(t.value, ast.Name) and t.value.id == "opts"
                                  for t in a.targets) for a in ast.walk(n)))
    return preflight, assignment


class _WithoutImports(ast.NodeTransformer):
    """The native symbols used by this isolated public guard are test doubles."""
    def visit_ImportFrom(self, node):
        return None


@pytest.mark.parametrize("method", ["RKS", "UKS"])
@pytest.mark.parametrize("custom_grid", [False, True])
def test_public_chi_grid_validation_and_assignment_preserve_request(method, custom_grid):
    class GridOptions:
        pass
    class ChiXCControls:
        pass
    selected = GridOptions() if custom_grid else None
    original = GridOptions()
    selected_calls = []
    opts = SimpleNamespace(grid=original)
    namespace = dict(chi_support_policy="physical_pairs", chi_coulomb_controls=None,
        chi_hf_reference=None, chi_xc_controls=ChiXCControls(), chi_grid_options=selected,
        chi_ks_reference=None, resolved_jk="chi", method_upper=method, correlation=None,
        optimize=False, optimize_cell=False, restart_from=None, functional="pbe",
        PeriodicJKMethod=SimpleNamespace(AICCM2026DEV_B="chi"), system=SimpleNamespace(dim=3),
        GridOptions=GridOptions, ChiXCControls=ChiXCControls,
        functional_identity=lambda functional, spin: selected_calls.append((functional, spin)),
        aiccm_backend="four_center", fit_source="auto",
        _resolve_aiccm2026dev_b_backend=lambda backend: SimpleNamespace(value=backend),
        resolve_support_policy=lambda *args, **kwargs: None,
        opts=opts, _external_xc_grid_profile=None)
    module = ast.Module(body=list(_public_chi_statements()), type_ignores=[])
    module = _WithoutImports().visit(module)
    exec(compile(ast.fix_missing_locations(module), "<public chi grid request>", "exec"), namespace)
    assert opts.grid is (selected if custom_grid else original)
    assert selected_calls == [("pbe", 1 if method == "RKS" else 2)]


@pytest.mark.parametrize("method, expected", [("RKS", TypeError), ("RHF", ValueError)])
def test_public_chi_grid_rejects_wrong_type_or_hf_before_support_work(method, expected):
    class GridOptions:
        pass
    class ChiXCControls:
        pass
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid grid must fail before support work")
    namespace = dict(chi_support_policy="physical_pairs", chi_coulomb_controls=None,
        chi_hf_reference=None, chi_xc_controls=ChiXCControls(), chi_grid_options=object(),
        chi_ks_reference=None, resolved_jk="chi", method_upper=method, correlation=None,
        optimize=False, optimize_cell=False, restart_from=None, functional="pbe",
        PeriodicJKMethod=SimpleNamespace(AICCM2026DEV_B="chi"), system=SimpleNamespace(dim=3),
        GridOptions=GridOptions, ChiXCControls=ChiXCControls,
        functional_identity=lambda *args: None,
        _resolve_aiccm2026dev_b_backend=forbidden, resolve_support_policy=forbidden)
    preflight, _ = _public_chi_statements()
    module = _WithoutImports().visit(ast.Module(body=[preflight], type_ignores=[]))
    with pytest.raises(expected, match="chi_grid_options|require RKS or UKS"):
        exec(compile(ast.fix_missing_locations(module), "<public chi grid refusal>", "exec"), namespace)
