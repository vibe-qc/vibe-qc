"""Array-only independent quadrature/adjoint checks; no native acceptance claim."""
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


_NAME = "_chi_xc_array_tests.periodic.chi.xc"
_spec = spec_from_file_location(_NAME, Path(__file__).resolve().parents[1] / "python/vibeqc/periodic/chi/xc.py")
xc = module_from_spec(_spec)
sys.modules[_NAME] = xc
_spec.loader.exec_module(xc)


class PolynomialXC:
    """Nonlinear energy density with independently coded exact derivatives."""
    components = [{"id": 999999, "weight": 1.0}]
    exchange_components = []
    component_capabilities = [(999999, "exchange-correlation", True, True, True, False)]

    def __init__(self, kind, spin=1):
        self.kind = self.name = kind
        self.spin = spin

    def eval_unpolarised(self, rho, sigma):
        gradient = self.kind != "LDA"
        return .13 * rho**3 + gradient * .17 * sigma, .39 * rho**2, np.full_like(rho, .17 * gradient)

    def eval_unpolarised_mgga(self, rho, sigma, tau):
        energy, vr, vs = self.eval_unpolarised(rho, sigma)
        return energy + .19 * tau**2, vr, vs, .38 * tau

    def eval_polarised(self, a, b, aa, ab, bb):
        gradient = self.kind != "LDA"
        energy = .13 * a**3 + .11 * b**3 + .07 * a * b
        energy += gradient * (.17 * aa + .23 * ab + .29 * bb)
        return energy, .39 * a**2 + .07 * b, .33 * b**2 + .07 * a, *[np.full_like(a, gradient * v) for v in (.17, .23, .29)]

    def eval_polarised_mgga(self, a, b, aa, ab, bb, ta, tb):
        energy, *potentials = self.eval_polarised(a, b, aa, ab, bb)
        return energy + .19 * ta**2 + .31 * tb**2, *potentials, .38 * ta, .62 * tb


@pytest.fixture
def setup(monkeypatch):
    lattice = np.array([[2.1, 1.3, .2], [0., 1.7, .3], [0., 0., 2.3]])
    centers = np.array([[.1, .2, .3], [.7, .5, .2]])
    points = np.array([[.2, .3, .4], [.8, 1., .3], [1.7, .4, 1.], [6.8, 2., .7]])
    grid = SimpleNamespace(points=points, weights=np.array([.13, .21, .34, .16]))
    calls = []

    def evaluate(basis, positions):
        calls.append(len(positions))
        displacement = positions[:, None, :] - basis.centers[None, :, :]
        values = np.exp(-np.array([.7, 1.1])[None, :] * np.sum(displacement**2, axis=2))
        derivative = -2 * np.array([.7, 1.1])[None, :, None] * displacement * values[:, :, None]
        return values, *[derivative[:, :, axis] for axis in range(3)]

    core = SimpleNamespace(evaluate_ao_with_gradient=evaluate, Functional=PolynomialXC,
                           libxc_version=lambda: "array-test", GridOptions=lambda: SimpleNamespace(n_radial=4))
    package = ModuleType("_chi_xc_array_tests")
    package._vibeqc_core = core
    monkeypatch.setitem(sys.modules, "_chi_xc_array_tests", package)
    grid_module = ModuleType("_chi_xc_array_tests.periodic_grid")
    grid_module.build_periodic_becke_grid = lambda *a, **k: grid
    monkeypatch.setitem(sys.modules, grid_module.__name__, grid_module)
    system = SimpleNamespace(dim=3, lattice=lattice)
    basis = SimpleNamespace(nbasis=2, centers=centers)
    basis.shells = lambda: [SimpleNamespace(l=0, pure=True, origin=center) for center in basis.centers]
    kpoints = np.array([[0., 0., 0.], [.43, -.27, .31]])
    weights = np.array([.3, .7])
    rng = np.random.default_rng(119)
    coefficients = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    coefficients *= .18
    density = coefficients @ coefficients.conj().swapaxes(-1, -2)

    def make(kind="MGGA", spin=1, block=2, radius=1.8, cap=1_000_000):
        return xc.ChiSemilocalXC(system, basis, kpoints, weights, PolynomialXC(kind, spin), spin,
                                SimpleNamespace(grid=SimpleNamespace(n_radial=4)),
                                xc.ChiXCControls(radius, 2.4, block, cap))
    return SimpleNamespace(make=make, system=system, basis=basis, points=points, grid=grid,
                           kpoints=kpoints, weights=weights, density=density,
                           coefficients=coefficients, calls=calls, core=core)


def orbital_oracle(data, kind, spin, radius=1.8):
    """Full fixed image cube, explicit occupied orbitals and pointwise outer products."""
    images = np.asarray(list(product(range(-6, 7), repeat=3))) @ data.system.lattice.T
    displacement = data.points[None, :, None, :] - images[:, None, None, :] - data.basis.centers[None, None, :, :]
    distance2 = np.sum(displacement**2, axis=-1)
    values = np.exp(-np.array([.7, 1.1]) * distance2) * (distance2 <= radius**2 * (1 + 32 * np.finfo(float).eps))
    derivatives = -2 * np.array([.7, 1.1])[None, None, :, None] * displacement * values[:, :, :, None]
    aos, grads = [], []
    rho = np.zeros((spin, len(data.points)))
    gradient = np.zeros((spin, len(data.points), 3))
    tau = np.zeros_like(rho)
    for k, kp in enumerate(data.kpoints):
        phase = np.exp(1j * (images @ kp))
        ao = np.einsum("r,rpi->pi", phase, values)
        grad = np.einsum("r,rpia->pia", phase, derivatives)
        aos.append(ao); grads.append(grad)
        for s in range(spin):
            coefficient = data.coefficients[k] * (1 if s == 0 else np.sqrt(.63))
            psi = ao @ coefficient
            dpsi = np.einsum("pia,ij->pja", grad, coefficient)
            rho[s] += data.weights[k] * np.sum(abs(psi)**2, axis=1)
            gradient[s] += 2 * data.weights[k] * np.sum((psi.conj()[:, :, None] * dpsi).real, axis=1)
            tau[s] += .5 * data.weights[k] * np.sum(abs(dpsi)**2, axis=(1, 2))
    # Independent energy formula, without calling PolynomialXC or product contractions.
    energy = .13 * rho[0]**3
    vr = [.39 * rho[0]**2]
    q = [.34 * gradient[0]]
    vt = [.38 * tau[0]]
    if spin == 2:
        energy += .11 * rho[1]**3 + .07 * rho[0] * rho[1]
        vr = [vr[0] + .07 * rho[1], .33 * rho[1]**2 + .07 * rho[0]]
        q = [.34 * gradient[0] + .23 * gradient[1], .58 * gradient[1] + .23 * gradient[0]]
        vt.append(.62 * tau[1])
    if kind != "LDA":
        energy += .17 * np.sum(gradient[0]**2, axis=1)
        if spin == 2:
            energy += .23 * np.sum(gradient[0] * gradient[1], axis=1) + .29 * np.sum(gradient[1]**2, axis=1)
    if kind == "MGGA":
        energy += .19 * tau[0]**2
        if spin == 2:
            energy += .31 * tau[1]**2
    potentials = np.zeros((spin, len(aos), 2, 2), dtype=complex)
    for s in range(spin):
        for k, (ao, grad) in enumerate(zip(aos, grads)):
            for p, wp in enumerate(data.grid.weights):
                potentials[s, k] += wp * vr[s][p] * np.outer(ao[p].conj(), ao[p])
                if kind != "LDA":
                    for axis in range(3):
                        cross = np.outer(grad[p, :, axis].conj(), ao[p])
                        potentials[s, k] += wp * q[s][p, axis] * (cross + cross.conj().T)
                        if kind == "MGGA":
                            potentials[s, k] += .5 * wp * vt[s][p] * np.outer(grad[p, :, axis].conj(), grad[p, :, axis])
    return data.grid.weights @ energy, potentials, rho @ data.grid.weights


@pytest.mark.parametrize("kind", ["LDA", "GGA", "MGGA"])
@pytest.mark.parametrize("spin", [1, 2])
def test_complex_orbital_reference_and_directional_fd(setup, kind, spin):
    evaluator = setup.make(kind, spin)
    densities = [setup.density] if spin == 1 else [setup.density, .63 * setup.density]
    evaluate = evaluator.restricted if spin == 1 else evaluator.unrestricted
    energy, *potential = evaluate(*densities)
    expected, reference, charge = orbital_oracle(setup, kind, spin)
    np.testing.assert_allclose(energy, expected, atol=2e-15, rtol=2e-13)
    np.testing.assert_allclose(potential, reference, atol=3e-15, rtol=2e-13)
    np.testing.assert_allclose(evaluator.metadata["integrated_spin_charges"], charge, atol=2e-15)
    rng = np.random.default_rng(41)
    direction = rng.normal(size=(spin, 2, 2, 2)) + 1j * rng.normal(size=(spin, 2, 2, 2))
    direction = .03 * (direction + direction.conj().swapaxes(-1, -2))
    analytic = sum(setup.weights[k] * np.trace(potential[s][k] @ direction[s, k]).real for s in range(spin) for k in range(2))
    errors = []
    for step in [.02, .01, .005]:
        plus = evaluate(*[d + step * h for d, h in zip(densities, direction)])[0]
        minus = evaluate(*[d - step * h for d, h in zip(densities, direction)])[0]
        errors.append(abs((plus - minus) / (2 * step) - analytic))
    assert errors[2] < errors[0] * .07
    assert errors[2] < 1e-8
    assert max(setup.calls) <= 2


def test_block_invariance_and_image_support_negative_control(setup):
    first = setup.make(block=1)
    e1, v1 = first.restricted(setup.density)
    second = setup.make(block=4)
    e2, v2 = second.restricted(setup.density)
    np.testing.assert_allclose(e1, e2, atol=2e-15)
    np.testing.assert_allclose(v1, v2, atol=2e-15)
    assert first.metadata["grid_sha256"] == second.metadata["grid_sha256"]
    short = setup.make(radius=.6)
    es, _ = short.restricted(setup.density)
    assert abs(es - e1) > 1e-5
    assert short.metadata["grid_sha256"] == first.metadata["grid_sha256"]
    assert first.metadata["executed_image_count"] > 0
    assert len(first.metadata["executed_image_sha256"]) == 64


def test_atom_image_relabelling_covariance(setup):
    energy, potential = setup.make().restricted(setup.density)
    shift = setup.system.lattice @ np.array([2, -1, 0])
    setup.basis.centers[1] += shift
    unitary = np.ones((2, 2), dtype=complex)
    unitary[:, 1] = np.exp(-1j * (setup.kpoints @ shift))
    transformed = unitary.conj()[:, :, None] * setup.density * unitary[:, None, :]
    shifted_e, shifted_v = setup.make().restricted(transformed)
    np.testing.assert_allclose(shifted_e, energy, atol=3e-15)
    np.testing.assert_allclose(shifted_v, unitary.conj()[:, :, None] * potential * unitary[:, None, :], atol=3e-15)


def test_rigid_translation_and_lattice_basis_covariance(setup):
    energy, potential = setup.make().restricted(setup.density)
    shift = np.array([.17, -.13, .23])
    setup.basis.centers += shift
    setup.points += shift
    setup.system.lattice = setup.system.lattice @ np.array([[1, 3, 0], [0, 1, 0], [0, 0, 1]])
    shifted_e, shifted_v = setup.make().restricted(setup.density)
    np.testing.assert_allclose(shifted_e, energy, atol=3e-15)
    np.testing.assert_allclose(shifted_v, potential, atol=4e-15)


@pytest.mark.parametrize("field,value", [("density_image_radius_bohr", 0), ("partition_radius_bohr", np.nan), ("block_size", 1.5), ("image_candidate_cap", True)])
def test_bad_controls(field, value):
    arguments = dict(density_image_radius_bohr=2., partition_radius_bohr=2.)
    arguments[field] = value
    with pytest.raises(ValueError):
        xc.ChiXCControls(**arguments)


def test_candidate_cap_refuses_without_truncation(setup):
    evaluator = setup.make(cap=1)
    with pytest.raises(ValueError, match="candidate cap"):
        evaluator.restricted(setup.density)
    assert not setup.calls


@pytest.mark.parametrize("bad", [np.ones((2, 2)), np.full((2, 2, 2), np.nan), np.array([[[1, 1j], [1j, 1]]]*2)])
def test_bad_density(setup, bad):
    with pytest.raises(ValueError):
        setup.make().restricted(bad)


@pytest.mark.parametrize("flag", ["is_external", "is_double_hybrid", "needs_vv10", "needs_laplacian"])
def test_unsupported_functional_before_grid(setup, monkeypatch, flag):
    functional = PolynomialXC("GGA")
    setattr(functional, flag, True)
    module = sys.modules["_chi_xc_array_tests.periodic_grid"]
    monkeypatch.setattr(module, "build_periodic_becke_grid", lambda *a, **k: pytest.fail("grid allocated before rejection"))
    with pytest.raises(NotImplementedError, match=flag):
        xc.ChiSemilocalXC(setup.system, setup.basis, setup.kpoints, setup.weights,
                         functional, 1, SimpleNamespace(), xc.ChiXCControls(2, 2))


def test_functional_identity_refusal(setup):
    evaluator = setup.make("MGGA")
    evaluator.require_functional("MGGA")
    with pytest.raises(ValueError, match="identity"):
        evaluator.require_functional("GGA")


@pytest.mark.parametrize("weights", [[.1, .1], [-.2, 1.2], [np.nan, 1.], [.5+.1j, .5], [[.3, .7]]])
def test_invalid_k_weights(setup, weights):
    with pytest.raises(ValueError, match="weight"):
        xc.ChiSemilocalXC(setup.system, setup.basis, setup.kpoints, weights,
                         "LDA", 1, SimpleNamespace(), xc.ChiXCControls(2, 2))


def test_wrong_functional_spin_rejected_before_grid(setup, monkeypatch):
    module = sys.modules["_chi_xc_array_tests.periodic_grid"]
    monkeypatch.setattr(module, "build_periodic_becke_grid", lambda *a, **k: pytest.fail("grid allocated before rejection"))
    with pytest.raises(ValueError, match="spin"):
        xc.ChiSemilocalXC(setup.system, setup.basis, setup.kpoints, setup.weights,
                         PolynomialXC("LDA", 2), 1, SimpleNamespace(), xc.ChiXCControls(2, 2))


def test_prepared_arrays_own_their_storage(setup):
    evaluator = setup.make()
    energy, potential = evaluator.restricted(setup.density)
    setup.kpoints[:] = 7
    setup.weights[:] = 9
    setup.points[:] = 8
    setup.grid.weights[:] = 6
    setup.system.lattice[:] = 4
    repeated_e, repeated_v = evaluator.restricted(setup.density)
    assert repeated_e == energy
    np.testing.assert_array_equal(repeated_v, potential)
    assert all(not array.flags.writeable for array in [evaluator.kpoints, evaluator.weights, evaluator.points, evaluator.grid_weights, evaluator.lattice])


@pytest.mark.parametrize("kind", ["LDA", "GGA", "MGGA"])
def test_gamma_real_limit(setup, kind):
    setup.kpoints[:] = 0
    setup.coefficients[:] = setup.coefficients.real
    setup.density = setup.coefficients @ setup.coefficients.swapaxes(-1, -2)
    energy, potential = setup.make(kind).restricted(setup.density)
    expected, reference, _ = orbital_oracle(setup, kind, 1)
    np.testing.assert_allclose(energy, expected, atol=3e-15)
    np.testing.assert_allclose(potential, reference[0], atol=3e-15)
    np.testing.assert_array_equal(potential.imag, 0)
