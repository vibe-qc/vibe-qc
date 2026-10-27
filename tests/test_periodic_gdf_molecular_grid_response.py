"""Native-free composition controls, not numerical gradient acceptance.

Execute the maintained Gamma RKS assembler and the UKS compcell XC block,
replacing only their native primitives. The tests can run directly with Python.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class _WithoutImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None


def _compile(nodes, path, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    tree = _WithoutImports().visit(tree)
    # Compile with postponed annotations without importing the native package.
    import __future__
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)  # noqa: S102 - execute only maintained local source


def _composition(spins, periodic, response, *, functional="pbe",
                 option_functional="", expected_functional="pbe"):
    fixed = np.array([[0.2, -0.3, 0.7], [-0.1, 0.5, -0.6]])
    nonxc = np.array([[0.7, 0.3, -0.2], [-0.7, -0.3, 0.2]])
    density = np.array([[1.2, 0.3], [0.3, 0.8]])
    options, grid, basis, system = [object() for _ in range(4)]
    lat = SimpleNamespace(cutoff_bohr=12.0)
    func = SimpleNamespace(hf_exchange_fraction=0.0, is_range_separated=False)
    molecular_domain, periodic_domain = object(), object()
    domain = periodic_domain if periodic else molecular_domain
    calls = []
    created = []

    def density_set(*args):
        value = SimpleNamespace(density=None)
        created.append(value)
        return value

    def set_density(target, value):
        target.density = np.array(value, copy=True)

    def correction(sy, ba, densities, functional, lo, go, pb, radius, fg, *, density_domain):
        assert (sy, ba, lo, go) == (system, basis, lat, options)
        assert functional == expected_functional and pb is periodic and radius == 0.0
        assert density_domain is domain
        assert len(densities) == spins
        for item in densities:
            np.testing.assert_array_equal(item.density, density / spins)
        np.testing.assert_array_equal(fg, fixed)
        calls.append(densities)
        return response.copy()

    ns = {"np": np, "Functional": lambda *a: func, "GridOptions": lambda: options,
              "PeriodicXCDensityDomain": SimpleNamespace(MOLECULAR_HOME=molecular_domain, AUTO=periodic_domain),
              "_validate_supported_gradient_result": lambda *a, **k: None,
              "compute_gdf_gradient_rhf_gamma": lambda *a, **k: (nonxc.copy(), "cache"),
              "compute_overlap_lattice": density_set, "_density_set_gamma": set_density,
              "_density_set_torus_gamma": set_density, "_set_xc_density": set_density,
              "build_grid": lambda *a: grid, "build_periodic_becke_grid": lambda *a, **k: grid,
              "xc_lattice_opts_at": lambda *a: lat,
              "xc_lattice_gradient_contribution": lambda *a: fixed.copy(),
              "xc_lattice_gradient_contribution_uks": lambda *a: fixed.copy(),
              "_gdf_xc_grid_motion_correction": correction}
    # Standalone RKS creates its grid from the unit-cell molecule.
    system = SimpleNamespace(unit_cell_molecule=lambda: object())
    if spins == 1:
        path = ROOT / "python/vibeqc/periodic_gdf_gradient.py"
        nodes = [n for n in ast.parse(path.read_text()).body
                 if isinstance(n, ast.FunctionDef) and n.name == "compute_gdf_gradient_rks_gamma"]
        _compile(nodes, path, ns)
        result = SimpleNamespace(functional="pbe", density=density)
        actual, cache = ns["compute_gdf_gradient_rks_gamma"](
            system, basis, result, aux_basis=object(), grid_options=options,
            use_periodic_becke=periodic, lattice_opts=lat,
        )
        assert cache == "cache"
    else:
        path = ROOT / "python/vibeqc/pbc_gdf.py"
        driver = next(n for n in ast.parse(path.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == "run_pbc_gdf_uks")
        # Select the actual compcell branch by its maintained kernel call;
        # execute its entire XC if-block, with no rewritten gradient algebra.
        candidates = [n for n in ast.walk(driver) if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == "compute_gradient"
                      and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                              and c.func.id == "compute_gdf_gradient_uhf_gamma" for c in ast.walk(n))]
        assert len(candidates) == 1
        block = next(n for n in candidates[0].body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Name) and n.test.id == "is_ks")
        ns.update(is_ks=True, system=system, basis=basis, grid=grid, func=func,
                  functional=functional, xc_lat_opts=lat, grid_options=options,
                  opts=SimpleNamespace(use_periodic_becke=periodic,
                                       becke_image_radius_bohr=0.0,
                                       functional=option_functional),
                  D_alpha_set=density_set(), D_beta_set=density_set(),
                  D_alpha_used=density / 2, D_beta_used=density / 2, grad_total=nonxc.copy())
        # Execute the driver's own option/keyword resolution before its
        # compcell XC block, so a missing keyword cannot bypass provenance.
        resolution = [n for n in driver.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name)
                              and t.id in {"func_name", "is_ks", "func"}
                              for t in n.targets)]
        assert len(resolution) == 3
        _compile([*resolution, block], path, ns)
        actual = ns["grad_total"]
    expected = nonxc + fixed + response
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)
    assert len(calls) == 1
    for item in created:
        np.testing.assert_array_equal(item.density, density / spins)


class MolecularGridResponseComposition(unittest.TestCase):
    def test_rks_missing_response(self):
        _composition(1, False, np.array([[0.04, 0.0, 0.13], [0.0, 0.0, -0.02]]))

    def test_uks_missing_response(self):
        _composition(2, False, np.array([[0.04, 0.0, 0.13], [0.0, 0.0, -0.02]]))

    def test_uks_functional_from_options(self):
        _composition(2, False, np.ones((2, 3)), functional=None,
                     option_functional="pbe", expected_functional="pbe")

    def test_uks_explicit_functional_overrides_options(self):
        _composition(2, False, np.ones((2, 3)), functional="lda",
                     option_functional="pbe", expected_functional="lda")

    def test_zero_response_preserves_both_spin_assemblies(self):
        for spins in (1, 2):
            with self.subTest(spins=spins):
                _composition(spins, False, np.zeros((2, 3)))

    def test_periodic_becke_response_uses_scf_density_domain(self):
        for spins in (1, 2):
            with self.subTest(spins=spins):
                _composition(spins, True, np.ones((2, 3)))


if __name__ == "__main__":
    unittest.main()
