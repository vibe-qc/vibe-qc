"""Independent contractions and ownership at the MANTA libint boundary."""
from itertools import product

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule, compute_eri, compute_overlap, make_direct_jk_builder
from vibeqc._vibeqc_core import ShellInfo
from vibeqc import manta


def _basis(pure=True):
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [.3, -.2, 1.4])], 0, 1)
    shells = []
    for atom, l, exponents, coefficients in [
        (0, 0, [.7, 1.8], [.8, -.15]),
        (0, 1, [.5], [1.]),
        (1, 2, [.9], [1.]),
    ]:
        s = ShellInfo()
        s.atom_index = atom
        s.l = l
        s.pure = pure
        s.exponents = exponents
        s.coefficients = coefficients
        s.origin = [0., 0., 0.] if atom == 0 else [.3, -.2, 1.4]
        shells.append(s)
    return mol, BasisSet(mol, shells, "manta-reference", False)


@pytest.mark.parametrize("pure", [False, True])
def test_shell_adapter_and_all_ordered_tiles(pure):
    _, basis = _basis(pure)
    builder = manta.ReferenceJKBuilder(basis)
    for original, copied in zip(basis.shells(), builder.shells, strict=True):
        assert (original.l, original.pure, original.atom_index) == (copied.l, copied.pure, copied.atom_index)
        np.testing.assert_array_equal(original.exponents, copied.exponents)
        np.testing.assert_array_equal(original.coefficients, copied.coefficients)
        np.testing.assert_array_equal(original.origin, copied.origin)
    eri = np.asarray(compute_eri(basis))  # deliberately small independent oracle
    sizes = [2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in basis.shells()]
    offsets = np.r_[0, np.cumsum(sizes)]
    saved = None
    for quartet in product(range(basis.nshells), repeat=4):
        tile = builder.evaluate(quartet)
        intervals = tuple(slice(offsets[i], offsets[i+1]) for i in quartet)
        np.testing.assert_allclose(tile["values"], eri[intervals], atol=1e-12, rtol=5e-12)
        assert tile["shell_calls"] == 1 and tile["null_tiles"] in (0, 1)
        assert tile["provider"] == "libint"
        if saved is None:
            saved = tile["values"]
            expected = saved.copy()
    np.testing.assert_array_equal(saved, expected)
    # The provider must not individually renormalize Cartesian d components.
    if not pure:
        s = compute_overlap(basis)
        assert s[5, 5] == pytest.approx(s[4, 4] / 3, abs=1e-13)


@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("kind", ["psd", "indefinite", "zero"])
def test_jk_independent_dense_and_existing_direct(pure, kind):
    _, basis = _basis(pure)
    n = basis.nbasis
    rng = np.random.default_rng(6201)
    trial = rng.normal(size=(n, n)) / n
    d = trial @ trial.T if kind == "psd" else trial + trial.T
    if kind == "zero":
        d[:] = 0
    builder = manta.ReferenceJKBuilder(basis)
    result = builder.build(d)
    eri = np.asarray(compute_eri(basis))
    j = np.einsum("abcd,cd->ab", eri, d)
    k = np.einsum("abcd,bd->ac", eri, d)
    direct = make_direct_jk_builder(basis, 0.)
    for actual, ref, existing in [(result.J, j, direct.build_J(d)), (result.K, k, direct.build_K(d))]:
        np.testing.assert_allclose(actual, ref, atol=2e-12, rtol=5e-12)
        np.testing.assert_allclose(actual, existing, atol=2e-12, rtol=5e-12)
        np.testing.assert_allclose(actual, actual.T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(builder.build_g_rhf(d, .37), j-.185*k, atol=2e-12)
    receipt = result.receipt
    assert receipt.provider == "libint" and receipt.provider_version
    assert receipt.primitive_precision == 0
    assert receipt.shell_calls == basis.nshells**4 == builder.plan.shell_calls
    assert receipt.primitive_quartets_requested == sum(len(s.exponents) for s in basis.shells())**4
    assert receipt.density_terms_accumulated == 2*receipt.integral_elements_consumed
    assert receipt.integral_elements_consumed <= n**4
    assert len(receipt.source_identity) == len(receipt.density_identity) == 64


def test_linearity_energy_derivative_and_owned_epochs():
    _, basis = _basis()
    n = basis.nbasis
    rng = np.random.default_rng(6202)
    d = rng.normal(size=(n, n)); d += d.T
    delta = rng.normal(size=(n, n)); delta += delta.T
    builder = manta.ReferenceJKBuilder(basis)
    first = builder.build(d)
    old_j = first.J.copy(); old_k = first.K.copy()
    second = builder.build(delta)
    combined = builder.build(d + delta)
    np.testing.assert_allclose(combined.J, first.J + second.J, atol=3e-12)
    np.testing.assert_allclose(combined.K, first.K + second.K, atol=3e-12)
    alpha, beta = builder.build_K_pair(d, delta)
    np.testing.assert_allclose(alpha, old_k, atol=1e-12)
    np.testing.assert_allclose(beta, second.K, atol=1e-12)
    step = 1e-5
    def energy(x):
        result = builder.build(x)
        return .5*np.einsum("ij,ji", x, result.J) - .25*np.einsum("ij,ji", x, result.K)
    fd = (energy(d+step*delta)-energy(d-step*delta))/(2*step)
    analytic = np.einsum("ij,ji", delta, old_j-.5*old_k)
    assert fd == pytest.approx(analytic, abs=1e-8)
    d[:] = 0
    builder.build(d)
    np.testing.assert_array_equal(first.J, old_j)
    np.testing.assert_array_equal(first.K, old_k)
    assert first.receipt.density_identity != second.receipt.density_identity
    assert first.receipt.source_identity == second.receipt.source_identity
    del builder, basis
    np.testing.assert_array_equal(first.J, old_j)


@pytest.mark.parametrize("field,value", [
    ("kernel", "yukawa"), ("omega", .4), ("omega", -1.), ("scalar", "complex"), ("centers", 3),
    ("derivative", 1), ("images", [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
])
def test_unsupported_request_refuses(field, value):
    _, basis = _basis()
    request = manta.KernelRequest()
    setattr(request, field, value)
    capability = manta.capability(request)
    assert not capability.supported
    with pytest.raises(ValueError, match=capability.reason):
        manta.ReferenceJKBuilder(basis, request)


@pytest.mark.parametrize("pure", [False, True])
def test_range_reference_tiles_receipts_and_identities(pure):
    """Libint's own erf/erfc core evaluators serve the reference provider."""
    _, basis = _basis(pure)
    full = manta.ReferenceJKBuilder(basis)
    identities = {full.source_identity}
    rng = np.random.default_rng(6305)
    d = rng.normal(size=(basis.nbasis, basis.nbasis)); d += d.T
    reference = full.build(d)
    for omega in [.3, 1.2]:
        parts = {}
        for kernel in ["erf", "erfc"]:
            request = manta.KernelRequest(); request.kernel, request.omega = kernel, omega
            assert manta.capability(request).supported
            builder = manta.ReferenceJKBuilder(basis, request)
            assert builder.plan.kernel == kernel and builder.plan.omega == omega
            identities.add(builder.source_identity)
            parts[kernel] = builder.build(d)
            assert parts[kernel].receipt.provider == "libint"
            assert (parts[kernel].receipt.kernel, parts[kernel].receipt.omega) == (kernel, omega)
            for quartet in [(0, 1, 2, 0), (2, 2, 1, 0)]:
                parts[kernel, quartet] = builder.evaluate(quartet)["values"]
        np.testing.assert_allclose(parts["erf"].J+parts["erfc"].J, reference.J, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(parts["erf"].K+parts["erfc"].K, reference.K, atol=2e-11, rtol=2e-11)
        for quartet in [(0, 1, 2, 0), (2, 2, 1, 0)]:
            np.testing.assert_allclose(parts["erf", quartet]+parts["erfc", quartet], full.evaluate(quartet)["values"],
                                       atol=2e-11, rtol=2e-11)
    assert len(identities) == 5


@pytest.mark.parametrize("field,plan_field", [
    ("memory_bytes", "modeled_peak_bytes"), ("shell_calls", "shell_calls"),
    ("primitive_quartets", "primitive_quartets_upper_bound"), ("scalar_terms", "scalar_terms"),
])
def test_exact_resource_cap_and_one_unit_short(field, plan_field):
    _, basis = _basis()
    reference = manta.plan(basis)
    budget = manta.ResourceBudget()
    setattr(budget, field, getattr(reference, plan_field))
    builder = manta.ReferenceJKBuilder(basis, budget=budget)
    builder.build(np.zeros((basis.nbasis, basis.nbasis)))
    setattr(budget, field, getattr(reference, plan_field)-1)
    with pytest.raises((ValueError, RuntimeError), match="budget"):
        manta.ReferenceJKBuilder(basis, budget=budget)


def test_resource_overflow_and_backend_allowance():
    _, basis = _basis()
    budget = manta.ResourceBudget()
    budget.other_live_bytes = 2**64-1
    with pytest.raises(OverflowError):
        manta.plan(basis, budget=budget)
    budget.other_live_bytes = 0
    budget.backend_margin_bytes = 0
    with pytest.raises(ValueError, match="margin"):
        manta.plan(basis, budget=budget)


@pytest.mark.parametrize("invalid", ["shape", "nan", "asymmetric", "complex", "float32", "strided"])
def test_density_admission(invalid):
    _, basis = _basis()
    builder = manta.ReferenceJKBuilder(basis)
    n = basis.nbasis
    d = np.eye(n)
    if invalid == "shape": d = np.eye(n+1)
    elif invalid == "nan": d[0, 0] = np.nan
    elif invalid == "asymmetric": d[0, 1] = .1
    elif invalid == "complex": d = d.astype(complex)
    elif invalid == "float32": d = d.astype(np.float32)
    else: d = np.eye(2*n)[::2, ::2]
    with pytest.raises(ValueError):
        builder.build(d)


def test_source_identity_tracks_coefficients_and_fortran_input():
    mol, basis = _basis()
    builder = manta.ReferenceJKBuilder(basis)
    shells = basis.shells()
    original = shells[0].coefficients
    shells[0].coefficients = [1.01*v for v in original]
    altered = BasisSet(mol, shells, "manta-altered", True)
    assert manta.ReferenceJKBuilder(altered).source_identity != builder.source_identity
    roundtrip = BasisSet(mol, basis.shells(), "renamed-only", True)
    assert manta.ReferenceJKBuilder(roundtrip).source_identity == builder.source_identity
    d = np.eye(basis.nbasis)
    np.testing.assert_array_equal(builder.build(d).J, builder.build(np.asfortranarray(d)).J)
    with pytest.raises(IndexError):
        builder.evaluate((0, 0, 0, basis.nshells))


def test_reference_builder_in_native_rhf_host():
    import vibeqc as vq
    mol = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])], 0, 1)
    basis = BasisSet(mol, "sto-3g")
    overlap = vq.compute_overlap(basis)
    core = vq.compute_kinetic(basis) + vq.compute_nuclear(basis, mol)
    options = vq.RHFOptions()
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-9
    reference = vq.run_rhf_scf_with_jk(basis, 2, overlap, core, 1/1.4,
                                      make_direct_jk_builder(basis, 0.), options)
    actual = vq.run_rhf_scf_with_jk(basis, 2, overlap, core, 1/1.4,
                                   manta.ReferenceJKBuilder(basis), options)
    assert actual.converged and reference.converged
    assert actual.energy == pytest.approx(reference.energy, abs=1e-11)
    np.testing.assert_allclose(actual.density, reference.density, atol=1e-10)


def test_reusable_session_retains_provider_and_returns_owned_tiles():
    _, basis = _basis()
    builder = manta.ReferenceJKBuilder(basis)
    session = builder.session()
    first = session.evaluate((0, 1, 2, 0))
    expected = builder.evaluate((0, 1, 2, 0))["values"]
    assert first["primitive_quartets_requested"] == 4
    assert first["primitive_precision"] == 0.
    del builder, basis
    session.evaluate((2, 2, 2, 2))
    np.testing.assert_array_equal(first["values"], expected)
    np.testing.assert_array_equal(session.evaluate((0, 1, 2, 0))["values"], expected)
    with pytest.raises(IndexError):
        session.evaluate((0, 0, 0, 999))
