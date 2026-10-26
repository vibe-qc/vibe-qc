"""Independent finite-dimensional controls; no native runtime is imported."""
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "python/vibeqc/periodic/chi/operator_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location("chi_operator_diagnostics_test_module", _PATH)
diag = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = diag
_SPEC.loader.exec_module(diag)


def convention():
    return diag.OperatorConvention("source-fixture", "domain-fixture", "finite-Coulomb",
                                   "zero-omitted", "spin-summed")


def data():
    rng = np.random.default_rng(7142)
    def hermitian():
        x = rng.normal(size=(3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3))
        return .5 * (x + x.conj().swapaxes(-1, -2))
    return hermitian(), hermitian(), np.array([.2, .3, .5])


def test_weighted_trace_matches_explicit_complex_index_sum():
    d, h, w = data()
    expected = sum(w[k] * d[k, i, j] * h[k, j, i]
                   for k in range(3) for i in range(3) for j in range(3))
    assert diag.weighted_trace(d, h, w) == pytest.approx(expected)


@pytest.mark.parametrize("weights", [[1., 1., 1.], [.0, .5, .5], [.2, -.3, 1.1],
                                     [1.], [.2, .3, np.nan], [.2+0j, .3+0j, .5+0j]])
def test_weights_are_not_silently_repaired(weights):
    d, h, _ = data()
    with pytest.raises(ValueError):
        diag.weighted_trace(d, h, weights)


def test_linear_self_adjoint_complex_operator_and_independent_energy():
    d, h, w = data()
    a = np.array([[1., .3j, 0], [-.3j, 2., .2], [0, .2, 3.]])
    operator = lambda x: a @ x @ a
    def independent_energy(x):
        value = 0j
        for k in range(3):
            for i in range(3):
                for j in range(3):
                    for p in range(3):
                        for q in range(3):
                            value += .5 * w[k] * x[k,i,j] * a[j,p] * x[k,p,q] * a[q,i]
        return value
    report = diag.diagnose_operator(operator, d, h, w, convention=convention(),
                                   energy=independent_energy)
    assert report.status == "PASS"
    assert report.as_dict()["qualification"] == "none"
    assert report.as_dict()["data"]["energy_reference"] == "supplied_callback"


@pytest.mark.parametrize("kind,failed", [("offset", "zero_density"),
                                        ("nonlinear", "linearity"),
                                        ("nonsymmetric", "reciprocal_bilinear"),
                                        ("antihermitian", "output_hermiticity")])
def test_deliberately_wrong_operator_controls(kind, failed):
    d, h, w = data()
    def operator(x):
        if kind == "offset":
            return x + np.eye(3)
        if kind == "nonlinear":
            return x @ x
        if kind == "antihermitian":
            return (1 + .1j) * x
        # Cross-k coupling without the reciprocal weighted transpose.
        out = x.copy()
        out[0] += x[1]
        return out
    report = diag.diagnose_operator(operator, d, h, w, convention=convention())
    assert report.status == "FAIL"
    assert {m.name: m.status for m in report.measurements}[failed] == "FAIL"


def test_wrong_independent_energy_derivative_detected():
    d, h, w = data()
    report = diag.diagnose_operator(lambda x: x, d, h, w, convention=convention(),
        energy=lambda x: .6 * diag.weighted_trace(x, x, w))
    assert report.status == "FAIL"
    assert any(m.status == "FAIL" for m in report.measurements if "derivative" in m.name)


def test_operator_mutation_is_rejected():
    d, h, w = data()
    def operator(x):
        x[0,0,0] = 127
        return x
    with pytest.raises(ValueError, match="mutated"):
        diag.diagnose_operator(operator, d, h, w, convention=convention())


def test_nonhermitian_direction_rejected():
    d, h, w = data()
    h[0,0,1] += 1j
    with pytest.raises(ValueError, match="Hermitian"):
        diag.diagnose_operator(lambda x: x, d, h, w, convention=convention())


def energy_report(**updates):
    d = np.array([[[2., 0], [0, 0]], [[1., .2j], [-.2j, 1.]]])
    s = np.broadcast_to(np.eye(2), d.shape)
    t, v, j, k = s * .7, -s * 1.4, s * .5, s * .3
    values = dict(convention=convention(), expected_electrons=2., reported_energy=-.25,
                  nuclear_energy=.8, coulomb_factor=.5, exchange_factor=-.25,
                  expanded_nuclear_energy=2.4, nuclear_cell_count=3)
    values.update(updates)
    return diag.diagnose_energy(d, s, t, v, j, k, [.25,.75], **values)


def test_energy_decomposition_and_nuclear_normalization():
    report = energy_report()
    assert report.status == "PASS"
    assert report.as_dict()["data"]["components"]["total"] == pytest.approx(-.25)


@pytest.mark.parametrize("change,criterion", [({"reported_energy": -.2}, "energy_decomposition"),
    ({"expected_electrons": 1.}, "electron_count"),
    ({"expanded_nuclear_energy": .8}, "nuclear_normalization"),
    ({"exchange_factor": -.5}, "energy_decomposition")])
def test_energy_wrong_controls(change, criterion):
    report = energy_report(**change)
    assert {m.name:m.status for m in report.measurements}[criterion] == "FAIL"


def test_missing_nuclear_reference_is_unmeasured_and_immutable():
    report = energy_report(expanded_nuclear_energy=None, nuclear_cell_count=None)
    assert report.status == "UNMEASURED"
    first = report.identity
    report.as_dict()["data"]["components"]["nuclear"] = 5.
    assert report.identity == first
    with pytest.raises(FrozenInstanceError):
        report.data_json = "{}"


@pytest.mark.parametrize("threshold,rank", [(1e-12,3), (1e-8,2), (1e-3,1)])
def test_metric_near_dependence_and_rank_transition(threshold, rank):
    matrix = np.diag([1e-10, 1e-4, 2.]).astype(complex)
    unitary = np.fft.fft(np.eye(3)) / np.sqrt(3)
    matrix = unitary.conj().T @ matrix @ unitary
    report = diag.diagnose_metric(matrix, convention=convention(),
        absolute_threshold=threshold, relative_threshold=0., reported_rank=rank)
    assert report.status == "PASS"
    assert report.as_dict()["data"]["rank"] == rank


def test_metric_negative_eigenvalue_wrong_rank_and_missing_rank():
    matrix = np.diag([-1e-4, 1e-8, 3.])
    report = diag.diagnose_metric(matrix, convention=convention(), absolute_threshold=1e-6,
                                  relative_threshold=0., reported_rank=2)
    assert report.status == "FAIL"
    assert {m.name:m.status for m in report.measurements} == {
        "metric_hermiticity":"PASS", "metric_negative_eigenvalue":"FAIL", "reported_rank":"FAIL"}
    report = diag.diagnose_metric(np.eye(2), convention=convention(),
                                  absolute_threshold=0., relative_threshold=1e-9)
    assert report.status == "UNMEASURED"


@pytest.mark.parametrize("kind", ["ao_permutation", "complex_unitary", "character_transform"])
def test_independent_covariance_transport(kind):
    d, h, w = data()
    a = np.diag([1., 2., 3.])
    od = a @ d @ a
    if kind == "ao_permutation":
        u = np.eye(3)[:, [2,0,1]]
    else:
        # Complex discrete characters also supply a full unitary real-space
        # representation change; operator covariance is independent of SCF.
        u = np.fft.fft(np.eye(3)) / np.sqrt(3)
    transformed_d = u.conj().T @ d @ u
    transformed_a = u.conj().T @ a @ u
    transported = u @ (transformed_a @ transformed_d @ transformed_a) @ u.conj().T
    report = diag.compare_covariance(od, transported, convention=convention(), transformation=kind)
    assert report.status == "PASS"
    assert diag.weighted_trace(d, od, w) == pytest.approx(
        diag.weighted_trace(transformed_d, transformed_a @ transformed_d @ transformed_a, w))
    wrong = transported.copy()
    wrong[0,0,1] += 1e-4
    assert diag.compare_covariance(od, wrong, convention=convention(), transformation=kind).status == "FAIL"


def test_k_permutation_requires_permuted_weights():
    d, h, w = data()
    p = [2,0,1]
    assert diag.weighted_trace(d[p], h[p], w[p]) == pytest.approx(diag.weighted_trace(d,h,w))
    assert abs(diag.weighted_trace(d[p], h[p], w) - diag.weighted_trace(d,h,w)) > .1


@pytest.mark.parametrize("value", [np.nan, -1., np.inf])
def test_bad_tolerance_rejected(value):
    d, h, w = data()
    with pytest.raises(ValueError):
        diag.diagnose_operator(lambda x: x, d, h, w, convention=convention(), tolerance=value)
