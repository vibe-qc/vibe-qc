"""Positive RI metric response, including motion of the retained projector."""
import gc

import numpy as np
import pytest

from vibeqc import manta
import vibeqc as vq
from vibeqc import _vibeqc_core as core


def _basis(angular=(0, 1), displacement=0.):
    positions = [[.1, -.2, .3], [.4, .2, -.1], [-.2, .5, .7]]
    positions[1][2] += displacement
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate(angular):
        shell = core.ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = i, l, True, positions[i]
        shell.exponents, shell.coefficients = [.7+.1*i, 1.6+.2*i], [.8, -.18]
        shells.append(shell)
    return vq.BasisSet(mol, shells, "manta-metric-response", False)


def _options(absolute=1e-12, relative=1e-10):
    o = manta.RIOptions()
    o.absolute_cutoff, o.relative_cutoff = absolute, relative
    return o


def _symmetric(n, seed=786):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return (a + a.T) / 2


def _matrix(values, seed=954):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(len(values), len(values))))
    return (q * values) @ q.T


def _spectral(v, options):
    lam, u = np.linalg.eigh(v)
    keep = lam > max(options.absolute_cutoff, options.relative_cutoff * max(0., lam[-1]))
    return (u[:, keep] / lam[keep]) @ u[:, keep].T, u[:, keep] @ u[:, keep].T


@pytest.mark.parametrize("values,absolute,relative", [
    ([.2, 1., 3., 5.], 1e-12, 1e-10),
    ([.02, .07, 1., 4.], .1, 0.),
    ([.02, .07, 1., 4.], 1e-12, .1),
    ([.02, .02, 3., 3.], .1, 0.),
    ([0., 0., 3., 3.], .1, 0.),
    ([-1e-8, .02, 3., 3.], .1, 0.),
    ([1., 2., 3., 4.], 8., 0.),
    ([2.], .1, 0.),
    ([.02], .1, 0.),
    ([.02, 3.], .3, .1),  # Nonsmooth max cutoff, but a smooth retained space.
])
def test_response_against_independent_spectral_finite_differences(values, absolute, relative):
    options = _options(absolute=absolute, relative=relative)
    v = _matrix(np.array(values))
    direction = _symmetric(len(values))
    response = manta.RIMetricResponse(v, options)
    result = response.apply(direction)
    step = min(1e-4, response.info.rank_stability_radius / (10 * np.linalg.norm(direction)))
    # Five-point difference suppresses truncation without sharing the native
    # divided-difference implementation. All displaced retained ranks agree.
    samples = [_spectral(v + k * step * direction, options) for k in (-2, -1, 1, 2)]
    for field, index in (("inverse", 0), ("projector", 1)):
        fd = (samples[0][index] - 8*samples[1][index] + 8*samples[2][index] - samples[3][index])/(12*step)
        np.testing.assert_allclose(getattr(result, field), fd, atol=2e-9, rtol=3e-8)
        np.testing.assert_allclose(getattr(response, field), _spectral(v, options)[index], atol=2e-12)
    assert result.source_identity == response.info.source_identity
    assert len(result.direction_identity) == 64
    assert result.modeled_peak_bytes == response.plan.modeled_peak_bytes
    assert response.info.rank_stability_radius > 0


def test_retained_discarded_mixing_is_not_an_ordinary_inverse_derivative():
    options = _options(absolute=.5, relative=0.)
    v = np.diag([.2, 2.])
    e = np.array([[0., 1.], [1., 0.]])
    r = manta.RIMetricResponse(v, options)
    d = r.apply(e)
    np.testing.assert_allclose(d.inverse, e / 3.6)
    np.testing.assert_allclose(d.projector, e / 1.8)
    assert np.linalg.norm(d.inverse + r.inverse @ e @ r.inverse) > .3
    assert r.info.rank == 1 and r.info.has_split
    assert r.info.retained_discarded_gap == pytest.approx(1.8)


@pytest.mark.parametrize("values", [[1., 1., 4., 4.], [.1, .1, 4., 4.], [.1, .1, .2, .2]])
def test_frobenius_adjoint_and_orthogonal_covariance(values):
    o = _options(absolute=.5, relative=0.)
    v = _matrix(np.array(values))
    e, a, b = (_symmetric(4, seed) for seed in (12, 13, 14))
    r = manta.RIMetricResponse(v, o)
    d, adj = r.apply(e), r.adjoint(a, b)
    assert np.sum(a*d.inverse) + np.sum(b*d.projector) == pytest.approx(np.sum(adj*e), abs=2e-13)
    q, _ = np.linalg.qr(np.random.default_rng(85).normal(size=(4, 4)))
    rotated = manta.RIMetricResponse(q @ v @ q.T, o).apply(q @ e @ q.T)
    np.testing.assert_allclose(rotated.inverse, q @ d.inverse @ q.T, atol=2e-13)
    np.testing.assert_allclose(rotated.projector, q @ d.projector @ q.T, atol=2e-13)
    # Differentiated projector identities detect lost cross-space terms.
    p = r.projector
    np.testing.assert_allclose(p @ d.projector + d.projector @ p, d.projector, atol=2e-13)
    np.testing.assert_allclose(p @ d.projector @ p, 0., atol=2e-13)
    if r.info.rank == 4:
        np.testing.assert_allclose(d.inverse, -r.inverse @ e @ r.inverse, atol=2e-13)
        np.testing.assert_array_equal(d.projector, np.zeros((4, 4)))


def test_fixed_density_ri_energy_response_with_nonzero_discarded_rhs():
    o = _options(absolute=.4, relative=0.)
    v, dv = _matrix(np.array([.03, .1, 1.7, 3.])), _symmetric(4)
    b = np.array([.2, -.8, 1.1, .4]); db = np.array([.5, .1, -.2, .8])
    r = manta.RIMetricResponse(v, o)
    assert np.linalg.norm((np.eye(4)-r.projector) @ b) > .1
    derivative = (r.inverse @ b) @ db + .5*b @ r.apply(dv).inverse @ b
    h = 1e-5
    def energy(t):
        shifted = b+t*db
        return .5 * shifted @ _spectral(v+t*dv, o)[0] @ shifted
    assert derivative == pytest.approx((energy(h)-energy(-h))/(2*h), abs=2e-9)
    pullback = r.adjoint(.5*np.outer(b, b), np.zeros((4, 4)))
    assert derivative == pytest.approx((r.inverse @ b) @ db + np.sum(pullback*dv), abs=2e-13)


@pytest.mark.parametrize("absolute", [1e-12, 5., 100.])
def test_response_uses_identical_native_ri_factor_and_rank(absolute):
    o = _options(absolute=absolute)
    builder = manta.RIJKBuilder(_basis(), _basis((0, 1, 2)), o)
    r = manta.RIMetricResponse(builder.metric, o)
    assert r.info.rank == builder.metric_info.rank
    assert r.info.factor_identity == builder.metric_info.factor_identity
    np.testing.assert_array_equal(r.factor, builder.factor)
    np.testing.assert_allclose(r.inverse, builder.factor @ builder.factor.T, atol=1e-15)


@pytest.mark.parametrize("relative", [0., .25])
@pytest.mark.parametrize("offset", [0., np.finfo(float).eps, -np.finfo(float).eps])
def test_rank_crossing_and_numerically_unresolved_cutoffs_refuse(relative, offset):
    o = _options(absolute=.5, relative=relative)
    with pytest.raises(RuntimeError, match="rank cutoff"):
        manta.RIMetricResponse(np.diag([.5+offset, 2.]), o)


def test_default_duplicate_metric_keeps_a_resolved_zero_subspace():
    builder = manta.RIJKBuilder(_basis(), _basis((0, 0, 0)))
    metric = builder.metric
    # Explicit duplicate channels, independent of shell positions.
    metric[-1, :] = metric[0, :]; metric[:, -1] = metric[:, 0]
    r = manta.RIMetricResponse(metric)
    assert r.info.rank == 2
    assert r.info.cutoff_distance > r.info.spectral_guard
    assert r.info.rank_stability_radius < r.info.cutoff


def test_lower_triangle_source_ownership_and_signed_zero():
    o = _options(absolute=.2, relative=0.)
    v = np.diag([.1, 2., 3.]); v[0, 1] = -0.
    r = manta.RIMetricResponse(v, o)
    e = _symmetric(3)
    expected = r.apply(e)
    source = r.info.source_identity
    upper_noise = v.copy(); upper_noise[0, 1] = 1e-13
    same = manta.RIMetricResponse(np.asfortranarray(upper_noise), o)
    assert same.info.source_identity == source
    np.testing.assert_array_equal(same.inverse, r.inverse)
    noisy_e = e.copy(); noisy_e[0, 1] += 1e-13
    assert r.apply(noisy_e).direction_identity == expected.direction_identity
    for value in (r.inverse, r.projector, r.factor):
        value[:] = 9.
    v[:] = 0.; o.absolute_cutoff = 1e10
    gc.collect()
    assert r.info.source_identity == source
    np.testing.assert_array_equal(r.apply(e).inverse, expected.inverse)
    np.testing.assert_array_equal(r.apply(np.asfortranarray(e)).projector, expected.projector)
    with pytest.raises(AttributeError):
        r.info.rank = 0


@pytest.mark.parametrize("field,value", [("absolute_cutoff", -1.), ("relative_cutoff", np.nan),
    ("negative_relative_tolerance", np.inf), ("mode_block", 0), ("mode_block", 65)])
def test_policy_refusals_precede_eigensolve(field, value):
    o = _options(); setattr(o, field, value)
    with pytest.raises(ValueError):
        manta.RIMetricResponse(np.eye(3), o)


@pytest.mark.parametrize("entry", ["constructor", "apply", "adjoint_a", "adjoint_b"])
@pytest.mark.parametrize("bad", ["float32", "complex", "strided", "nan", "asymmetric", "empty", "nonsquare", "sequence"])
def test_strict_input_contract(entry, bad):
    good = np.eye(3)
    value = {"float32": good.astype(np.float32), "complex": good.astype(complex),
             "strided": np.eye(6)[::2, ::2], "nan": good*np.nan,
             "asymmetric": np.triu(np.ones((3, 3))), "empty": np.empty((0, 0)),
             "nonsquare": np.empty((3, 2)), "sequence": good.tolist()}[bad]
    r = manta.RIMetricResponse(good)
    with pytest.raises((ValueError, TypeError)):
        if entry == "constructor":
            manta.RIMetricResponse(value)
        elif entry == "apply":
            r.apply(value)
        elif entry == "adjoint_a":
            r.adjoint(value, good)
        else:
            r.adjoint(good, value)


@pytest.mark.parametrize("entry", ["apply", "adjoint_a", "adjoint_b"])
def test_response_rejects_dimension_mismatch(entry):
    r = manta.RIMetricResponse(np.eye(3))
    with pytest.raises(ValueError, match="dimensions"):
        if entry == "apply":
            r.apply(np.eye(2))
        elif entry == "adjoint_a":
            r.adjoint(np.eye(2), np.eye(3))
        else:
            r.adjoint(np.eye(3), np.eye(2))


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"),
    ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_exact_resource_boundaries_before_payload_validation(field, plan_field):
    b = manta.ResourceBudget(); b.other_live_bytes = 12345
    p = manta.plan_ri_metric_response(3, budget=b)
    setattr(b, field, getattr(p, plan_field))
    r = manta.RIMetricResponse(np.eye(3), budget=b)
    r.apply(np.eye(3)); r.adjoint(np.eye(3), np.eye(3))
    assert r.plan.modeled_peak_bytes == p.modeled_peak_bytes
    setattr(b, field, getattr(p, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.RIMetricResponse(np.full((3, 3), np.nan), budget=b)


def test_count_overflow_margin_and_zero_dimension():
    for n in (0, -1, 2**63-1):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            manta.plan_ri_metric_response(n)
    b = manta.ResourceBudget(); b.backend_margin_bytes = 65535
    with pytest.raises(ValueError, match="margin"):
        manta.plan_ri_metric_response(2, budget=b)


def test_negative_metric_and_nonfinite_response_refuse():
    with pytest.raises(RuntimeError, match="negative eigenvalue"):
        manta.RIMetricResponse(np.diag([-1., 2.]))
    o = _options(absolute=0., relative=0.)
    with pytest.raises(OverflowError, match="overflow"):
        manta.RIMetricResponse(np.diag([1e-180, 2e-180]), o)
    r = manta.RIMetricResponse(np.diag([.01, .02]))
    with pytest.raises(OverflowError, match="overflow"):
        r.apply(np.diag([1e308, 1e308]))


def test_metric_response_has_a_citation_route():
    from vibeqc.output.citations.registry import load_default_database
    references = load_default_database().assemble(method="RHF", extra_methods=["manta_ri_metric_response"])
    assert "higham_matrix_functions_2008" in {entry.key for entry in references.citations}


@pytest.mark.parametrize("kernel", ["coulomb", "erf", "erfc"])
@pytest.mark.parametrize("truncate", [False, True])
def test_auxiliary_center_motion_combines_with_analytic_metric_tiles(kernel, truncate):
    orbital, auxiliary = _basis(), _basis((0, 1, 2))
    request = manta.KernelRequest()
    request.kernel, request.omega = kernel, 0. if kernel == "coulomb" else .6
    initial = manta.RIJKBuilder(orbital, auxiliary, request=request)
    eigenvalues = np.linalg.eigvalsh(initial.metric)
    options = _options(absolute=(eigenvalues[3]+eigenvalues[4])/2 if truncate else 1e-12)
    response = manta.RIMetricResponse(initial.metric, options)
    drequest = manta.KernelRequest()
    drequest.centers, drequest.derivative = 2, 1
    drequest.kernel, drequest.omega = request.kernel, request.omega
    provider = manta.DerivativeTiles([auxiliary, auxiliary], request=drequest)
    offsets = [0, 1, 4, 9]
    dv = np.zeros((9, 9))
    for i in range(3):
        for j in range(3):
            for center, shell in enumerate((i, j)):
                if shell == 1:
                    dv[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] += provider.evaluate((i, j), center, 2)["values"]
    actual = response.apply(dv)
    h = 1e-4
    samples = []
    for k in (-2, -1, 1, 2):
        builder = manta.RIJKBuilder(orbital, _basis((0, 1, 2), k*h), options, request)
        assert builder.metric_info.rank == response.info.rank
        samples.append(_spectral(builder.metric, options))
    for name, index in (("inverse", 0), ("projector", 1)):
        fd = (samples[0][index]-8*samples[1][index]+8*samples[2][index]-samples[3][index])/(12*h)
        np.testing.assert_allclose(getattr(actual, name), fd, atol=3e-9, rtol=1e-7)
