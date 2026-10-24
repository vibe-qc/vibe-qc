"""Independent finite-array exchange/derivative contracts, not native acceptance."""
import ast
from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "python/vibeqc"


def load(name, path):
    spec = spec_from_file_location(name, ROOT / path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


xc = load("_chi_r4_tests.periodic.chi.xc", "periodic/chi/xc.py")
exchange = load("_chi_r4_exchange", "periodic_fitted_exchange.py")


class Source(dict):
    """Simple exact self-adjoint two-electron tensor, distinct from production fit contractions."""
    def __init__(self, kernel, omega, weights, scale):
        self.kernel, self.omega_screen, self._weights = kernel, omega, tuple(weights)
        self.factor = np.array([[1.1, .2], [.2, .7]]) * scale
        super().__init__({(i, j): self.factor[None] for i in range(len(weights)) for j in range(len(weights))})

    def _density(self, density):
        return np.asarray(density)

    def exchange(self, density):
        mean = np.einsum("k,kij->ij", self._weights, density)
        result = self.factor @ mean @ self.factor.T
        return np.asarray([result] * len(density))


def model(full=.65, sr=-.46, omega=.33):
    weights = np.array([1/3] * 3)
    overlaps = np.asarray([[[1.2, .13], [.13, .9]]] * 3)
    a = xc.ChiExchangeAssembly(full, sr, omega if sr else 0)
    return exchange.ChiFittedExchange(Source("coulomb", 0, weights, 1),
        Source("erfc", omega, weights, .6) if sr else None, a,
        overlaps, weights, 40., .21)


def oracle(operator, density):
    """Explicit loops over all tensor indices and primitive-cell weights."""
    d = np.asarray(density)
    result = np.zeros_like(d)
    for k in range(len(d)):
        for i in range(2):
            for j in range(2):
                for q in range(len(d)):
                    for m in range(2):
                        for n in range(2):
                            v = operator.assembly.c_full * operator.full.factor[i, m] * operator.full.factor[n, j]
                            if operator.screened is not None:
                                v += operator.assembly.c_sr * operator.screened.factor[i, m] * operator.screened.factor[n, j]
                            result[k, i, j] += operator.weights[q] * v * d[q, m, n]
                zero = operator.assembly.c_full * operator.madelung
                if operator.assembly.c_sr:
                    zero += operator.assembly.c_sr * np.pi * operator.weights[k] / (operator.volume * operator.assembly.omega_screen**2)
                for m in range(2):
                    for n in range(2):
                        result[k, i, j] += zero * operator.overlap[k, i, m] * d[k, m, n] * operator.overlap[k, n, j]
    return result


@pytest.mark.parametrize("full,sr", [(.25, 0), (0, .25), (.65, -.46), (1, -1), (0, 0)])
def test_full_matrices_energy_derivatives_and_spin_collapse(full, sr):
    op = model(full, sr)
    rng = np.random.default_rng(55)
    c = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    density = c @ c.conj().transpose(0, 2, 1)
    direction = rng.normal(size=c.shape) + 1j * rng.normal(size=c.shape)
    direction += direction.conj().transpose(0, 2, 1)
    actual = op.exchange(density)
    np.testing.assert_allclose(actual, oracle(op, density), atol=2e-14, rtol=2e-14)
    energies = op.energies(.5*density, .5*density)
    restricted = -.25 * np.einsum("k,kij,kji->", op.weights, density, actual).real
    np.testing.assert_allclose(sum(energies.values()), restricted, atol=1e-13)
    def energy(d):
        return sum(op.energies(d, .37*density).values())
    step = 1e-4
    finite_difference = (energy(density+step*direction)-energy(density-step*direction))/(2*step)
    derivative = -np.einsum("k,kij,kji->", op.weights, direction, actual).real
    np.testing.assert_allclose(finite_difference, derivative, atol=2e-8, rtol=2e-10)
    # Occupied unitary rotation leaves density and both exact-exchange arms fixed.
    unitary, _ = np.linalg.qr(rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2)))
    rotated = (c @ unitary) @ (c @ unitary).conj().transpose(0,2,1)
    np.testing.assert_allclose(op.exchange(rotated), actual, atol=3e-14, rtol=2e-14)


def test_screened_zero_mode_and_range_endpoint():
    d = np.asarray([np.diag([.8, .2])] * 3, dtype=complex)
    op = model(0, .25)
    expected = .25*np.pi/(.33**2*120) * (op.overlap @ d @ op.overlap)
    np.testing.assert_allclose(op.components(d)["erfc_zero"], expected, atol=1e-15)
    assert "full_zero" not in op.components(d)
    # This tests only the analytic finite zero mode. The nonzero kernel has
    # a separate native integral test; fixed-mesh omega->0 totals diverge.
    large = model(0, .25, 1e6)
    assert np.max(abs(large.components(d)["erfc_zero"])) < 1e-13


@pytest.mark.parametrize("mutation", ["kernel", "range", "weights", "shape"])
def test_reject_incompatible_screened_factors(mutation):
    op = model()
    source = op.screened
    if mutation == "kernel": source.kernel = "coulomb"
    elif mutation == "range": source.omega_screen = .4
    elif mutation == "weights": source._weights = (.2,.3,.5)
    else: source[(0,0)] = np.zeros((1,3,3))
    with pytest.raises(ValueError):
        exchange.ChiFittedExchange(op.full, source, op.assembly, op.overlap, op.weights, op.volume, op.madelung)


def test_actual_component_resolution_and_unsupported_capabilities(monkeypatch):
    core = SimpleNamespace(libxc_version=lambda: "test-native-contract")
    package = ModuleType("_chi_r4_tests"); package._vibeqc_core = core
    monkeypatch.setitem(sys.modules, package.__name__, package)
    f = SimpleNamespace(name="custom", kind="MGGA", spin=2, components=[(1, .6)],
        component_capabilities=[(1,"exchange",True,True,True,False)],
        exchange_components=[("coulomb",.25,0), ("coulomb",.65,0), ("erfc",-.46,.33)])
    identity = xc.functional_identity(f,2)
    assert identity["exchange"]["full_range_coefficient"] == .9
    assert identity["exchange"]["short_range_coefficient"] == -.46
    f.exchange_components += [("erfc",.1,.2)]
    with pytest.raises(NotImplementedError,match="multiple"):
        xc.functional_identity(f,2)
    f.exchange_components = [("yukawa",.3,.2)]
    with pytest.raises(NotImplementedError,match="yukawa"):
        xc.functional_identity(f,2)
    f.exchange_components = []
    for capability in [(1,"kinetic",True,True,True,False),
                       (1,"exchange",False,True,True,False),
                       (1,"exchange",True,True,False,False),
                       (1,"exchange",True,True,True,True)]:
        f.component_capabilities = [capability]
        with pytest.raises(NotImplementedError,match="3D exchange/correlation"):
            xc.functional_identity(f,2)


@pytest.mark.parametrize("sr,assembly", [(.25,"short-range-direct"),(-.46,"short-range-direct"),
                                        (-.46,"kernel-specific-factors-plus-exact-zero-mode")])
def test_signed_execution_descriptor(sr,assembly):
    tree = ast.parse((ROOT / "periodic/chi/scf.py").read_text())
    classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in
               ("AICCM2026DevBExactExchangeAssembly","ChiHybridExactExchangeAssembly")]
    module = ModuleType("_chi_r4_tests.periodic.chi.scf")
    module.__dict__.update(dataclass=dataclass,field=field,np=np,
        _EXACT_EXCHANGE_ASSEMBLY_SCHEMA="legacy",_EXACT_EXCHANGE_RESOLVER="legacy",
        _SCREENED_EXCHANGE_ASSEMBLIES={"short-range-direct","full-range-minus-long-range"})
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=classes,type_ignores=[]),"<hybrid descriptor>","exec"),module.__dict__)
    result=module.ChiHybridExactExchangeAssembly(.65,sr,.33,"active",assembly)
    assert result.c_sr == sr
    assert result.schema.endswith("/v2")
