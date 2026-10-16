"""Paired three-center BUPO map against independent integrals and algebra."""
import gc

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc import _vibeqc_core as core


def _basis(pure=True, auxiliary=False, distance=60., angular=None):
    positions = [[0., .1, -.2], [1.1, -.3, .4], [distance, .2, -.1], [distance+1.2, -.1, .3]]
    if auxiliary:
        positions = [[.2, -.3, .4], [distance+.3, .5, -.2], [2*distance, -.1, .6]]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    shells = []
    for i, position in enumerate(positions):
        s = core.ShellInfo(); s.atom_index = i; s.origin = position; s.pure = pure
        s.l = (i % 3 if auxiliary else i % 2) if angular is None else angular
        s.exponents = [.6+.1*i, 1.5+.2*i]; s.coefficients = [.7, -.13]
        shells.append(s)
    return vq.BasisSet(mol, shells, "manta-bupo-fitting", False)


def _options(order=6, opening=.8, tolerance=1e-8, leaf=1):
    o = manta.BupoFittingOptions(); o.order = order; o.opening = opening
    o.coupling_tolerance = tolerance; o.bubbles.leaf_capacity = leaf
    return o


def _inputs(orbital, auxiliary):
    rng = np.random.default_rng(6237)
    # Nonsymmetric input also checks the full AO-matrix dual pairing.
    return rng.normal(size=(orbital.nbasis, orbital.nbasis)), rng.normal(size=auxiliary.nbasis)


def _assert_work(coupling, result):
    p, r = coupling.plan, result.receipt
    assert r.source_identity == coupling.source_identity
    assert r.partition_identity == p.partition_identity
    assert len(r.input_identity) == 64
    for field in ["near_shell_calls", "near_primitive_products", "far_blocks", "traversal_visits",
                  "pair_moment_panels", "auxiliary_moment_panels", "moment_primitive_products", "modeled_peak_bytes"]:
        assert getattr(r, field) == getattr(p, field)
    assert r.provider == "manta"
    assert r.approximation_bound >= p.maximum_coupling_error*r.input_l1
    assert r.compact_bound >= p.maximum_compact_error*r.input_l1
    assert r.penetration_bound >= p.maximum_penetration_error*r.input_l1
    assert r.moment_tail_bound >= p.maximum_moment_tail_error*r.input_l1


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("order,tolerance", [(0, 1.), (4, 1e-6), (6, 1e-8), (12, 1e-10)])
def test_forward_adjoint_against_independent_three_center_integrals(pure, order, tolerance):
    orbital, auxiliary = _basis(pure), _basis(pure, True)
    coupling = manta.BupoFittingCoupling(orbital, auxiliary, _options(order, tolerance=tolerance))
    d, c = _inputs(orbital, auxiliary)
    reference = vq.compute_3c_eri(orbital, auxiliary)
    assert np.max(abs(reference)) <= coupling.plan.maximum_exact_coupling
    forward, adjoint = coupling.forward(d), coupling.adjoint(c)
    error_forward = np.max(abs(forward.values-np.einsum("pij,ij->p", reference, d)))
    error_adjoint = np.max(abs(adjoint.values-np.einsum("pij,p->ij", reference, c)))
    assert error_forward <= forward.receipt.approximation_bound+2e-11
    assert error_adjoint <= adjoint.receipt.approximation_bound+2e-11
    assert coupling.plan.far_blocks > 0
    assert coupling.plan.near_shell_calls < auxiliary.nshells*orbital.nshells*(orbital.nshells+1)//2
    if order == 0:
        assert max(error_forward, error_adjoint) > 1e-6
    if order == 12:
        assert max(error_forward, error_adjoint) < 1e-9
    assert c@forward.values == pytest.approx(np.sum(d*adjoint.values), abs=2e-11, rel=2e-12)
    np.testing.assert_allclose(adjoint.values, adjoint.values.T, atol=2e-12)
    for result in [forward, adjoint]:
        _assert_work(coupling, result)
    assert forward.receipt.direction == "forward" and adjoint.receipt.direction == "adjoint"
    assert forward.receipt.input_l1 == pytest.approx(np.sum(abs(d)))
    assert adjoint.receipt.input_l1 == pytest.approx(np.sum(abs(c)))


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("reason", ["opening", "tolerance", "always_near"])
def test_exact_near_limit_and_zero_moment_work(pure, reason):
    orbital, auxiliary = _basis(pure, distance=2.), _basis(pure, True, distance=2.)
    o = _options()
    if reason == "opening": o.opening = 0.
    if reason == "tolerance": o.coupling_tolerance = 0.
    if reason == "always_near": o.extents.maximum_radius = o.extents.minimum_radius = .01
    coupling = manta.BupoFittingCoupling(orbital, auxiliary, o)
    d, c = _inputs(orbital, auxiliary)
    reference = vq.compute_3c_eri(orbital, auxiliary)
    assert np.max(abs(reference)) <= coupling.plan.maximum_exact_coupling
    a, b = coupling.forward(d), coupling.adjoint(c)
    np.testing.assert_allclose(a.values, np.einsum("pij,ij->p", reference, d), atol=2e-11)
    np.testing.assert_allclose(b.values, np.einsum("pij,p->ij", reference, c), atol=2e-11)
    assert coupling.plan.far_blocks == 0
    assert coupling.plan.near_shell_calls == auxiliary.nshells*orbital.nshells*(orbital.nshells+1)//2
    for result in [a, b]:
        _assert_work(coupling, result)
        assert result.receipt.approximation_bound == 0.
        assert result.receipt.pair_moment_panels == result.receipt.auxiliary_moment_panels == 0


@pytest.mark.parametrize("leaf", [1, 2, 150])
@pytest.mark.parametrize("order", [0, 6])
def test_linearity_adjoint_energy_derivative_and_positive_metric_map(leaf, order):
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    coupling = manta.BupoFittingCoupling(orbital, auxiliary, _options(order, tolerance=1., leaf=leaf))
    d, c = _inputs(orbital, auxiliary)
    rng = np.random.default_rng(531)
    e = rng.normal(size=d.shape); f = rng.normal(size=c.shape)
    np.testing.assert_allclose(coupling.forward(d+.37*e).values,
                               coupling.forward(d).values+.37*coupling.forward(e).values, atol=2e-12)
    np.testing.assert_allclose(coupling.adjoint(c-.21*f).values,
                               coupling.adjoint(c).values-.21*coupling.adjoint(f).values, atol=2e-12)
    # Independent exact metric and fixed positive eigenprojector: B~^T V+ B~.
    metric = vq.compute_2c_eri(auxiliary)
    w, u = np.linalg.eigh(metric); keep = w > max(1e-12, 1e-10*w[-1])
    factor = u[:, keep]/np.sqrt(w[keep])
    def energy(x):
        projected = factor.T@coupling.forward(x).values
        return .5*projected@projected
    rhs = coupling.forward(d).values
    j = coupling.adjoint(factor@(factor.T@rhs)).values
    assert energy(d) >= 0.
    assert energy(d) == pytest.approx(.5*np.sum(d*j), abs=2e-12)
    h = 2e-5
    derivative = (energy(d+h*e)-energy(d-h*e))/(2*h)
    assert derivative == pytest.approx(np.sum(e*j), abs=2e-9, rel=2e-9)
    je = coupling.adjoint(factor@(factor.T@coupling.forward(e).values)).values
    assert np.sum(e*j) == pytest.approx(np.sum(d*je), abs=2e-12)


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_products"), ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
@pytest.mark.parametrize("opening", [0., .8])
def test_aggregate_resource_boundary_admits_both_passes(field, plan_field, opening):
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    o = _options(4, opening=opening, tolerance=1e-5)
    budget = manta.ResourceBudget(); budget.other_live_bytes = 12345
    p = manta.BupoFittingCoupling(orbital, auxiliary, o, budget).plan
    setattr(budget, field, getattr(p, plan_field))
    coupling = manta.BupoFittingCoupling(orbital, auxiliary, o, budget)
    d, c = _inputs(orbital, auxiliary)
    _assert_work(coupling, coupling.forward(d)); _assert_work(coupling, coupling.adjoint(c))
    setattr(budget, field, getattr(p, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.BupoFittingCoupling(orbital, auxiliary, o, budget)


def test_owned_exports_repeatability_layout_identity_and_invalid_inputs():
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    options = _options(0, tolerance=1.)
    coupling = manta.BupoFittingCoupling(orbital, auxiliary, options)
    d, c = _inputs(orbital, auxiliary)
    a, b = coupling.forward(d), coupling.adjoint(c)
    saved_a, saved_b = a.values.copy(), b.values.copy()
    np.testing.assert_array_equal(coupling.forward(np.asfortranarray(d)).values, saved_a)
    assert coupling.forward(np.asfortranarray(d)).receipt.input_identity == a.receipt.input_identity
    a.values[:] = 123.; b.values[:] = 234.
    np.testing.assert_array_equal(a.values, saved_a); np.testing.assert_array_equal(b.values, saved_b)
    options.order = 12; options.extents.minimum_radius = 8.
    np.testing.assert_array_equal(coupling.forward(d).values, saved_a)
    assert manta.BupoFittingCoupling(orbital, auxiliary, _options(0, tolerance=1.)).source_identity == coupling.source_identity
    assert manta.BupoFittingCoupling(orbital, auxiliary, _options(1, tolerance=1.)).source_identity != coupling.source_identity
    for bad in [d.astype(np.float32), d.astype(complex), d[:, ::-1], d[:, :1], np.full_like(d, np.nan)]:
        with pytest.raises(ValueError): coupling.forward(bad)
    for bad in [c.astype(np.float32), c.astype(complex), c[::-1], c[:, None], np.full_like(c, np.inf)]:
        with pytest.raises(ValueError): coupling.adjoint(bad)
    assert np.count_nonzero(coupling.forward(np.zeros_like(d)).values) == 0
    assert np.count_nonzero(coupling.adjoint(np.zeros_like(c)).values) == 0
    del coupling, orbital, auxiliary; gc.collect()
    np.testing.assert_array_equal(a.values, saved_a); np.testing.assert_array_equal(b.values, saved_b)


@pytest.mark.parametrize("field,value", [("order", -1), ("order", 13), ("opening", -1.), ("opening", 1.),
    ("opening", np.nan), ("coupling_tolerance", -1.), ("coupling_tolerance", np.inf)])
def test_invalid_options(field, value):
    options = _options(); setattr(options, field, value)
    with pytest.raises(ValueError): manta.BupoFittingCoupling(_basis(), _basis(auxiliary=True), options)


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("distance", [.4, 100.])
def test_l6_orbital_l7_auxiliary_near_and_far_against_optional_libcint(pure, distance):
    gto = pytest.importorskip("pyscf.gto")
    positions = [[0., .1, -.2], [distance, -.3, .5]]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    bases = []
    for i, l in enumerate([6, 7]):
        s = core.ShellInfo(); s.atom_index = i; s.origin = positions[i]; s.l = l; s.pure = pure
        s.exponents = [.7]; s.coefficients = [1.]
        bases.append(vq.BasisSet(mol, [s], "manta-bupo-fitting-high-angular", False))
    orbital, auxiliary = bases
    reference_mol = gto.M(atom=[("He1", positions[0]), ("He2", positions[1])],
        basis={"He1": [[6, [.7, 1.]]], "He2": [[7, [.7, 1.]]]}, unit="Bohr", cart=not pure, verbose=0)
    reference = reference_mol.intor_by_shell("int3c2e_sph" if pure else "int3c2e_cart", (0, 0, 1)).transpose(2, 0, 1)
    if not pure: reference *= (13/(4*np.pi))*np.sqrt(15/(4*np.pi))
    coupling = manta.BupoFittingCoupling(orbital, auxiliary, _options(12, tolerance=1e-4))
    d, c = _inputs(orbital, auxiliary)
    forward, adjoint = coupling.forward(d), coupling.adjoint(c)
    assert coupling.plan.far_blocks == (1 if distance == 100. else 0)
    np.testing.assert_allclose(forward.values, np.einsum("pij,ij->p", reference, d), atol=5e-10, rtol=2e-10)
    np.testing.assert_allclose(adjoint.values, np.einsum("pij,p->ij", reference, c), atol=5e-10, rtol=2e-10)
    assert c@forward.values == pytest.approx(np.sum(d*adjoint.values), abs=2e-11)
    _assert_work(coupling, forward); _assert_work(coupling, adjoint)


def test_mixed_primitive_counts_and_exact_zero_auxiliary_channel():
    orbital, auxiliary = _basis(), _basis(auxiliary=True)
    shells = orbital.shells()
    shells[0].exponents = [.4, .7, 1., 1.8, 2.2]
    shells[0].coefficients = [.1, -.2, .3, -.1, .5]
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    orbital = vq.BasisSet(mol, shells, "mixed-primitive-counts", True)
    options = _options(6, tolerance=1e-5)
    original = manta.BupoFittingCoupling(orbital, auxiliary, options)
    budget = manta.ResourceBudget()
    for field, plan_field in [("shell_calls", "shell_calls"), ("primitive_quartets", "primitive_products"),
        ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms"), ("memory_bytes", "modeled_peak_bytes")]:
        setattr(budget, field, getattr(original.plan, plan_field))
    coupling = manta.BupoFittingCoupling(orbital, auxiliary, options, budget)
    d, c = _inputs(orbital, auxiliary)
    _assert_work(coupling, coupling.forward(d)); _assert_work(coupling, coupling.adjoint(c))
    shells = auxiliary.shells()
    for s in shells: s.coefficients = [0.]*len(s.exponents)
    mol = vq.Molecule([vq.Atom(2, s.origin) for s in shells], 0, 1)
    zero = vq.BasisSet(mol, shells, "zero-auxiliary", True)
    coupling = manta.BupoFittingCoupling(orbital, zero, options)
    assert coupling.plan.maximum_exact_coupling == coupling.plan.maximum_coupling_error == 0.
    assert np.count_nonzero(coupling.forward(d).values) == np.count_nonzero(coupling.adjoint(c).values) == 0
