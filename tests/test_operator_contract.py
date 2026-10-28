"""Independent numerical and refusal checks for shared Hermitian operators."""
from dataclasses import replace

import numpy as np
import pytest

from vibeqc.solvers.eigensolver import (
    EigenProblem, GeneralizedEigenProblem, InteriorEigenProblem, SCFStep,
    SolverOptions, solve_eigenproblem, solve_scf_step,
)
from vibeqc.solvers.operators import (
    FockOperator, HermitianOperator, OperatorFrame, OperatorMetadata, solve_operator,
    solver_capabilities, subspace_overlap,
)


@pytest.mark.parametrize("failure", ["backend", "projected", "orthogonality", "nonfinite"])
def test_solver_refusal_identifies_the_failing_contract_and_preserves_arrays(monkeypatch, failure):
    from vibeqc.solvers import eigensolver
    from vibeqc.solvers.operators import SolverConvergenceError

    matrix = np.diag([-1., 3.])
    operator = FockOperator(matrix=matrix)
    values, vectors = np.linalg.eigh(matrix)
    if failure == "projected":
        values[-1] += .001
    elif failure == "orthogonality":
        vectors *= 1. + 1e-5
    elif failure == "nonfinite":
        values[0] = np.nan
    backend_residuals = np.array([2e-6, 3e-6]) if failure == "backend" else np.zeros(2)
    backend = eigensolver.SolverResult(values, vectors, 1, failure != "backend",
                                       residuals=backend_residuals)
    monkeypatch.setattr(eigensolver, "solve_eigenproblem", lambda *a, **k: backend)
    with pytest.raises(SolverConvergenceError, match="residual/orthogonality contract") as caught:
        solve_operator(operator)
    error = caught.value
    assert isinstance(error, RuntimeError)
    status, arrays = error.diagnostics, error.arrays
    assert status["converged"] is status["usable_as_solution"] is False
    assert status["backend_converged"] == (failure != "backend")
    assert status["residual_tolerance"] == 1e-7
    assert status["orthogonality_tolerance"] == 1e-7
    assert status["evidence_status"] == "captured"
    assert status["evidence_bytes"] == sum(a.nbytes for a in arrays.values())
    np.testing.assert_array_equal(arrays["operator_matrix"], matrix)
    np.testing.assert_array_equal(arrays["backend_residuals"], backend_residuals)
    np.testing.assert_array_equal(arrays["projected_residual_vectors"],
                                  matrix @ arrays["ritz_vectors"] - arrays["ritz_vectors"]*arrays["eigenvalues"])
    if failure == "backend":
        assert status["max_projected_residual"] == 0.0
        assert status["orthogonality_error"] == 0.0
        assert status["max_backend_residual"] == 3e-6
    elif failure == "projected":
        assert status["max_projected_residual"] > status["residual_tolerance"]
        assert status["orthogonality_error"] == 0.0
    elif failure == "orthogonality":
        assert status["max_projected_residual"] == 0.0
        assert status["orthogonality_error"] > status["orthogonality_tolerance"]
    else:
        assert status["max_projected_residual"] is None
        assert not status["projected_residuals_finite"]
        assert np.isnan(arrays["eigenvalues"]).any()
    matrix[:] = 0
    vectors[:] = 0
    assert np.any(arrays["operator_matrix"])
    assert np.any(arrays["ritz_vectors"])
    for array in arrays.values():
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        arrays["operator_matrix"] = matrix


@pytest.mark.parametrize("with_overlap", [False, True])
def test_solver_refusal_preserves_exact_transformed_frame_and_backend_order(monkeypatch, with_overlap):
    from vibeqc.solvers import eigensolver
    from vibeqc.solvers.operators import SolverConvergenceError

    s = np.diag([1., 2., 1.])
    x = np.diag([1., 1/np.sqrt(2.), 0.])[:, :2].astype(complex)
    f = np.array([[1., .2j, .7], [-.2j, 3., .8j], [.7, -.8j, 5.]])
    a = x.conj().T @ f @ x
    values, vectors = np.linalg.eigh(a)
    backend_residuals = np.array([8e-6, 9e-6])
    monkeypatch.setattr(eigensolver, "solve_eigenproblem", lambda *args, **kwargs:
        eigensolver.SolverResult(values[::-1], vectors[:, ::-1], 1, False,
                                 residuals=backend_residuals))
    with pytest.raises(SolverConvergenceError) as caught:
        solve_operator(FockOperator(matrix=f, X=x, overlap=s if with_overlap else None))
    d, arrays = caught.value.diagnostics, caught.value.arrays
    np.testing.assert_array_equal(arrays["reduced_matrix"], a)
    np.testing.assert_array_equal(arrays["orthogonalizer"], x)
    np.testing.assert_array_equal(arrays["eigenvectors"], x @ vectors)
    np.testing.assert_array_equal(arrays["projected_residual_vectors"],
                                  x.conj().T @ (f @ (x @ vectors)) - vectors*values)
    np.testing.assert_array_equal(arrays["backend_sort_order"], [1, 0])
    np.testing.assert_array_equal(arrays["backend_residuals"], backend_residuals)
    assert d["residual_frame"] == ("original" if with_overlap else "orthogonal")
    assert ("original_residual_vectors" in arrays) == with_overlap
    assert ("overlap" in arrays) == with_overlap
    if with_overlap:
        c = arrays["eigenvectors"]
        np.testing.assert_array_equal(arrays["metric_gram"], c.conj().T @ (s @ c))
        assert np.linalg.norm(arrays["original_residual_vectors"]) > .1


@pytest.mark.parametrize("omission", ["byte_limit", "allocation_failure"])
def test_solver_refusal_bounds_capture_without_masking_failure(monkeypatch, omission):
    from vibeqc.solvers import eigensolver, operators

    op = FockOperator(matrix=np.diag([-1., 3.]))
    monkeypatch.setattr(eigensolver, "solve_eigenproblem", lambda *a, **k:
        eigensolver.SolverResult(np.array([-1., 3.]), np.eye(2), 1, False))
    if omission == "byte_limit":
        monkeypatch.setattr(operators, "_FAILURE_EVIDENCE_MAX_BYTES", 0)
        def refuse_copy(*args, **kwargs):
            pytest.fail("Over-budget evidence must not be copied")
        monkeypatch.setattr(np, "frombuffer", refuse_copy)
    else:
        def refuse_copy(*args, **kwargs):
            raise MemoryError("synthetic evidence allocation failure")
        monkeypatch.setattr(np, "frombuffer", refuse_copy)
    with pytest.raises(operators.SolverConvergenceError) as caught:
        solve_operator(op)
    assert caught.value.diagnostics["evidence_status"] == "omitted_" + omission
    assert caught.value.diagnostics["evidence_bytes"] == 0
    assert caught.value.diagnostics["evidence_required_bytes"] > 0
    assert not caught.value.arrays


def test_solver_refusal_does_not_probe_or_reapply_callback_for_capture(monkeypatch):
    from vibeqc.solvers import eigensolver
    from vibeqc.solvers.operators import SolverConvergenceError

    calls = []
    def action(v):
        calls.append(v.shape)
        assert v.shape == (8, 1)
        return np.arange(8)[:, None] * v
    op = FockOperator(matmat=action, n=8, diagonal=np.arange(8.))
    monkeypatch.setattr(eigensolver, "solve_eigenproblem", lambda *a, **k:
        eigensolver.SolverResult(np.array([0.]), np.eye(8)[:, :1], 1, False))
    with pytest.raises(SolverConvergenceError) as caught:
        solve_operator(op, SolverOptions(n_roots=1), solver="davidson")
    assert calls == [(8, 1), (8, 1)]  # Original independent diagnostics only.
    assert "operator_matrix" not in caught.value.arrays
    assert "reduced_matrix" not in caught.value.arrays
    assert not caught.value.diagnostics["matrix_free_operator_materialized"]


def test_solver_refusal_real_dense_failure_is_still_refused():
    from vibeqc.solvers.operators import SolverConvergenceError

    rng = np.random.default_rng(551)
    raw = rng.normal(size=(8, 8))
    matrix = (raw + raw.T) * 1e12
    with pytest.raises(SolverConvergenceError) as caught:
        solve_operator(FockOperator(matrix=matrix), SolverOptions(tol=1e-7))
    assert caught.value.diagnostics["max_projected_residual"] > 1e-7
    assert caught.value.diagnostics["evidence_status"] == "captured"
    np.testing.assert_array_equal(caught.value.arrays["operator_matrix"], matrix)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_solver_refusal_output_is_diagnostic_numeric_and_atomic(monkeypatch, tmp_path, nonfinite):
    import json
    from vibeqc.output.eigensolver import write_solver_failure
    from vibeqc.solvers import eigensolver
    from vibeqc.solvers.operators import SolverConvergenceError

    op = FockOperator(matrix=np.diag([-1., 3.]))
    monkeypatch.setattr(eigensolver, "solve_eigenproblem", lambda *a, **k:
        eigensolver.SolverResult(np.array([np.nan if nonfinite else -1., 3.]), np.eye(2), 1, False))
    with pytest.raises(SolverConvergenceError) as caught:
        solve_operator(op)
    path = tmp_path / "case.4.08.solver-failure.npz"
    assert write_solver_failure(path, caught.value) == path
    with np.load(path, allow_pickle=False) as archive:
        status = json.loads(str(archive["diagnostics_json"]))
        assert status["converged"] is status["usable_as_solution"] is False
        for name, value in caught.value.arrays.items():
            np.testing.assert_array_equal(archive[name], value)
    before = path.read_bytes()
    def refuse_write(*args, **kwargs):
        raise OSError("synthetic disk refusal")
    monkeypatch.setattr(np, "savez", refuse_write)
    with pytest.raises(OSError, match="synthetic disk refusal"):
        write_solver_failure(path, caught.value)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_evaluated_fock_build_owns_state_and_shares_one_metric():
    from vibeqc.fock_state import EnergyComponents, FockBuild, SourceProvenance

    s = np.diag([1., 2., 1e-16])
    x = np.diag([1., 1/np.sqrt(2.), 0.])[:, :2]
    frame = OperatorFrame(n=3, basis_id="synthetic-ao", ordering="declared",
                          image_gauge="home-cell", overlap=s, X=x)
    d = np.diag([1., 0., 0.])
    f = np.diag([-1., 3., 5.])
    source = SourceProvenance.capture("synthetic", kernel="coulomb", support=[0, 1])
    electronic, nuclear = np.array(-.75), np.array(.2)
    xc, correction = np.array(.25), np.array(.05)
    # Explicit KS energy, deliberately unrelated to an HF half trace.
    build = FockBuild.from_dense(
        method="UKS", generation=7, source=source,
        energy=EnergyComponents(electronic=electronic, nuclear=nuclear,
                                components=(("one_electron", -1.), ("xc", xc)),
                                corrections=(("model_correction", correction),)),
        densities=(d, d*.5), focks=(f, f+.1*np.eye(3)), frames=(frame, frame),
        spins=("alpha", "beta"), kpoints=((0., 0., 0.),)*2, weights=(1., 1.), kind="ks")
    d[:] = 9.; f[:] = 8.; s[:] = 7.; x[:] = 6.
    build_id = build.identity
    for scalar in (electronic, nuclear, xc, correction):
        scalar[...] = 9.
    np.testing.assert_array_equal(build.channels[0].density, np.diag([1., 0., 0.]))
    np.testing.assert_array_equal(build.channels[0].operator.matrix, np.diag([-1., 3., 5.]))
    assert type(build.energy.electronic) is float
    assert type(build.energy.nuclear) is float
    assert build.energy.components == (("one_electron", -1.), ("xc", .25))
    assert build.energy.corrections == (("model_correction", .05),)
    assert build.energy.total == pytest.approx(-.5)
    assert build.identity == build_id
    assert build.metric_electron_counts == (1., .5)
    assert frame.retained_rank == 2
    assert all(c.operator.overlap is frame.overlap and c.operator.X is frame.X for c in build.channels)
    for a in (frame.overlap, frame.X, build.channels[0].density, build.channels[0].operator.matrix):
        with pytest.raises(ValueError):
            a.setflags(write=True)
    assert build.storage_nbytes == 2 * (2 * 3*3*8 + 3*8) + (3*3 + 3*2)*8
    exposed = source.as_dict(); exposed["details"]["support"][0] = 99
    assert source.as_dict()["details"]["support"] == [0, 1]
    modified = build.solver_operator(0, np.diag([3., 4., 5.]), modification="level_shift")
    assert modified.metadata.role == "solver"
    assert modified.metadata.build_id == build.identity
    assert modified.metadata.density_id == build.density_id
    assert modified.frame is frame
    assert build.channels[0].operator.metadata.role == "physical"
    assert solve_operator(build.channels[0].operator).retained_rank == 2


@pytest.mark.parametrize("field", ["electronic", "nuclear", "components", "corrections"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, np.array([1.])])
def test_evaluated_energy_rejects_nonfinite_or_nonscalar_values(field, value):
    from vibeqc.fock_state import EnergyComponents

    values = dict(electronic=1., nuclear=0., components=(("model", 1.),))
    values[field] = (("model", value),) if field in {"components", "corrections"} else value
    with pytest.raises(ValueError, match="finite|scalars"):
        EnergyComponents(**values)


@pytest.mark.parametrize("values", [
    dict(electronic=1e308, nuclear=1e308, components=(("model", 1e308),)),
    dict(electronic=-1e308, nuclear=-1e308, components=(("model", -1e308),)),
    dict(electronic=0., nuclear=0., components=(),
         corrections=(("first", 1e308), ("second", 1e308))),
    dict(electronic=1e308, nuclear=0., components=(("model", 1e308),),
         corrections=(("extra", 1e308),)),
    dict(electronic=1e308, nuclear=0., components=(("first", 1e308), ("second", 1e308))),
])
def test_evaluated_energy_rejects_overflowing_sums(values):
    from vibeqc.fock_state import EnergyComponents

    with pytest.raises(ValueError, match="finite"):
        EnergyComponents(**values)


def test_evaluated_energy_preserves_method_total_summation_order():
    from vibeqc.fock_state import EnergyComponents

    energy = EnergyComponents(electronic=-1e16, nuclear=1e16,
                              components=(("model", -1e16),), corrections=(("extra", 1.),))
    assert energy.total == 1.


@pytest.mark.parametrize("change", [
    {"electronic": 1. + 5e-11},  # Within the decomposition tolerance.
    {"electronic": 2., "components": (("model", 2.),)},
    {"nuclear": 3.},
    {"components": (("renamed", 1.),)},
    {"components": (("first", .5), ("second", .5))},
    {"corrections": (("extra", .5),)},
    {"corrections": (("renamed", .25),)},
    {"nuclear": 2.25, "corrections": ()},  # Same total, distinct record.
    {"kind": "ks"},
])
def test_evaluated_build_identity_covers_energy_record_and_kind(change):
    from vibeqc.fock_state import EnergyComponents, FockBuild, SourceProvenance
    from vibeqc.output.eigensolver import eigensolver_diagnostics

    frame = OperatorFrame(n=2, basis_id="synthetic-ao", ordering="declared", image_gauge="home")
    source = SourceProvenance.capture("synthetic", kernel="coulomb")
    values = dict(electronic=1., nuclear=2., components=(("model", 1.),),
                  corrections=(("extra", .25),))

    def build(energy, kind="fock"):
        return FockBuild.from_dense(
            method="synthetic", generation=4, source=source, energy=energy,
            densities=(np.diag([1., 0.]),), focks=(np.diag([-1., 2.]),),
            frames=(frame,), spins=("alpha",), kpoints=(None,), weights=(1.,), kind=kind)

    first = build(EnergyComponents(**values))
    normalized = build(EnergyComponents(
        electronic=np.array(1.), nuclear=np.float64(2.),
        components=[("model", np.array(1.))], corrections=[("extra", np.float64(.25))]))
    assert normalized.identity == first.identity
    changed = dict(change)
    kind = changed.pop("kind", "fock")
    second = build(EnergyComponents(**(values | changed)), kind=kind)
    assert first.identity != second.identity
    assert first.density_id == second.density_id
    assert first.source.identity == second.source.identity
    assert first.channels[0].operator.frame.identity == second.channels[0].operator.frame.identity
    np.testing.assert_array_equal(first.channels[0].operator.matrix, second.channels[0].operator.matrix)
    assert second.output_fields()["fock_build_id"] == second.identity
    assert second.channels[0].operator.metadata.build_id == second.identity
    shifted = second.solver_operator(0, np.diag([-.5, 2.5]), modification="level_shift")
    record = eigensolver_diagnostics(solve_operator(shifted))["operator_metadata"]
    assert record["build_id"] == second.identity
    assert record["density_id"] == second.density_id
    assert record["role"] == "solver"


def test_shared_frame_rejects_changed_retained_subspace_on_recycling():
    n = 40
    kwargs = dict(n=n, basis_id="same-ao", ordering="declared", image_gauge="home-cell", overlap=np.eye(n))
    first = OperatorFrame(X=np.eye(n)[:, :30], **kwargs)
    changed = OperatorFrame(X=np.eye(n)[:, 1:31], **kwargs)
    f = np.diag(np.arange(n, dtype=float))
    options = SolverOptions(n_roots=2)
    previous = solve_operator(FockOperator(matrix=f, frame=first), options, solver="davidson")
    with pytest.raises(ValueError, match="different operator frame"):
        solve_operator(FockOperator(matrix=f, frame=changed), options, solver="davidson", previous=previous)


@pytest.mark.parametrize("weights", [(0.2, 0.8), (0.5, 0.5)])
def test_evaluated_complex_k_channels_preserve_metric_gauge_and_weights(weights):
    from vibeqc.fock_state import EnergyComponents, FockBuild, SourceProvenance
    scipy = pytest.importorskip("scipy.linalg")
    s = np.array([[1.3, .1j], [-.1j, .8]])
    w, u = np.linalg.eigh(s); x = u / np.sqrt(w)
    f = np.array([[-.8, .2j], [-.2j, .5]])
    c = scipy.eigh(f, s)[1]
    d = c[:, :1] @ c[:, :1].conj().T
    phase = np.diag(np.exp(1j*np.array([.25, -.31])))
    frames = tuple(OperatorFrame(n=2, basis_id=f"gauge-{i}", ordering="declared", image_gauge=str(i),
                                 overlap=g@s@g.conj().T, X=g@x) for i, g in enumerate((np.eye(2), phase)))
    state = FockBuild.from_dense(
        method="UHF", generation=0, source=SourceProvenance.capture("synthetic", kernel="coulomb"),
        energy=EnergyComponents(electronic=-1., nuclear=0., components=(("model", -1.),)),
        densities=(d, phase@d@phase.conj().T), focks=(f, phase@f@phase.conj().T),
        frames=frames, spins=("alpha", "alpha"), kpoints=((.1, 0., 0.), (-.2, 0., 0.)), weights=weights)
    assert state.metric_electron_counts == pytest.approx((1., 1.))
    for channel in state.channels:
        result = solve_operator(channel.operator)
        np.testing.assert_allclose(result.eigenvalues, scipy.eigh(f, s)[0], atol=1e-12)
        np.testing.assert_allclose(result.eigenvectors.conj().T @ channel.operator.overlap @ result.eigenvectors,
                                   np.eye(2), atol=1e-12)
    assert state.commutator_norm < 1e-12


def _unitary(n, complex_=False, seed=472):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    if complex_:
        a = a + 1j * rng.normal(size=(n, n))
    return np.linalg.qr(a)[0]


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("which,expected", [
    ("SA", [-20., -2.]), ("LA", [3., 9.]), ("SM", [-.1, .2]),
    ("LM", [-20., 9.]), ("BE", [-20., 9.]),
])
def test_dense_selection_is_not_algebraic_magnitude_alias(complex_, which, expected):
    u = _unitary(6, complex_)
    a = (u * [-20., -2., -.1, .2, 3., 9.]) @ u.conj().T
    r = solve_operator(HermitianOperator(matrix=a), SolverOptions(n_roots=2, which=which))
    np.testing.assert_allclose(r.eigenvalues, expected, atol=1e-12)
    assert r.selection_verified
    assert max(r.residuals) < 1e-12
    assert r.orthogonality_error < 1e-12


def test_one_root_both_ends_and_exact_interior_shift():
    p = EigenProblem(matrix=np.diag([-2., 0., 3.]))
    r = solve_eigenproblem(p, SolverOptions(n_roots=1, which="BE"), solver="dense")
    assert r.eigenvalues[0] == 3.
    r = solve_eigenproblem(InteriorEigenProblem(p, sigma=0.), SolverOptions(n_roots=1), solver="dense")
    assert r.eigenvalues[0] == 0.


def test_complex_ao_metric_against_independent_generalized_lapack():
    scipy = pytest.importorskip("scipy.linalg")
    u = _unitary(9, True)
    s = (u * np.linspace(.4, 2., 9)) @ u.conj().T
    q = _unitary(9, True, 473)
    f = (q * np.linspace(-2., 4., 9)) @ q.conj().T
    op = FockOperator(matrix=f, overlap=s)
    r = solve_operator(op, SolverOptions(n_roots=5, tol=1e-11))
    np.testing.assert_allclose(r.eigenvalues, scipy.eigh(f, s)[0][:5], atol=1e-12)
    np.testing.assert_allclose(r.eigenvectors.conj().T @ s @ r.eigenvectors, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(r.residuals, np.linalg.norm(f @ r.eigenvectors - (s @ r.eigenvectors)*r.eigenvalues, axis=0), atol=1e-14)
    # Legacy generalized adapter must preserve the same complex metric.
    legacy = solve_eigenproblem(GeneralizedEigenProblem(matrix=f, overlap=s), SolverOptions(n_roots=5), solver="dense")
    np.testing.assert_allclose(legacy.eigenvalues, r.eigenvalues, atol=1e-12)
    assert max(legacy.residuals) < 1e-11


def test_rank_reduction_reports_original_and_projected_residuals():
    s = np.diag([1., 2., 1e-16])
    f = np.array([[1., 0., .3], [0., 4., 0.], [.3, 0., 8.]])
    r = solve_operator(FockOperator(matrix=f, overlap=s))
    assert r.retained_rank == 2
    np.testing.assert_allclose(r.eigenvalues, [1., 2.])
    assert max(r.projected_residuals) < 1e-12
    assert max(r.residuals) == pytest.approx(.3)
    with pytest.raises(ValueError, match="retained"):
        solve_operator(FockOperator(matrix=f, overlap=s), SolverOptions(n_roots=3))
    with pytest.raises(ValueError, match="indefinite"):
        solve_operator(FockOperator(matrix=f, overlap=np.diag([1., 1., -1.])))


def test_rectangular_complex_scf_warm_start_and_no_option_mutation():
    # An exact previous subspace exposes transposed-inverse errors directly.
    u = _unitary(20, True)
    x = u[:, :16] / np.sqrt(np.linspace(.5, 2., 16))
    # F constructed so that X^H F X has this known diagonal spectrum.
    f = (u[:, :16] * (np.linspace(-2., 2., 16) * np.linspace(.5, 2., 16))) @ u[:, :16].conj().T
    prev = x[:, :3].copy()
    opts = SolverOptions(n_roots=3, tol=1e-10)
    result = solve_scf_step(SCFStep(fock=f, X=x, n_occ=3, prev_C=prev), opts, solver="hermitian_davidson")
    assert opts.guess_vectors is None
    np.testing.assert_allclose(result.eigenvalues, np.linspace(-2., 2., 16)[:3], atol=1e-11)
    np.testing.assert_allclose(prev, x[:, :3])


def test_callback_is_not_probed_and_warm_start_reuses_subspace():
    n = 80
    rng = np.random.default_rng(474)
    noise = rng.normal(scale=.002, size=(n, n))
    a = np.diag(np.arange(n, dtype=float)) + noise + noise.T
    calls = []
    def mv(v):
        calls.append(v.shape)
        return a @ v
    op = HermitianOperator(matvec=mv, n=n, diagonal=a.diagonal(), metadata=OperatorMetadata(kind="response", basis_id="test"))
    options = SolverOptions(n_roots=3, tol=1e-9, n_guess=8)
    cold = solve_operator(op, options, solver="davidson")
    assert cold.n_matvec == len(calls)
    assert len(calls) < n
    calls.clear()
    warm = solve_operator(op, options, solver="davidson", previous=cold)
    assert warm.n_matvec == len(calls)
    assert warm.n_iter <= cold.n_iter
    np.testing.assert_allclose(warm.eigenvalues, np.linalg.eigvalsh(a)[:3], atol=1e-9)
    assert not warm.selection_verified
    with pytest.raises(ValueError, match="frame"):
        solve_operator(replace(op, metadata=OperatorMetadata(spin="beta")), options, solver="davidson", previous=cold)


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("form", ["matvec", "matmat"])
def test_callback_scratch_reuse_preserves_columns_and_returned_actions(complex_, form):
    n = 40
    rng = np.random.default_rng(482)
    noise = rng.normal(scale=.001, size=(n, n))
    a = np.diag(np.arange(n, dtype=float)) + noise + noise.T
    if complex_:
        a = a + 1j * (noise - noise.T)
    scratch = {}
    calls = []

    def action(v):
        calls.append(v.shape)
        out = scratch.setdefault(v.shape, np.empty_like(v, dtype=a.dtype))
        np.matmul(a, v, out=out)
        return out

    op = FockOperator(**{form: action}, n=n, dtype=a.dtype, diagonal=a.diagonal().real)
    vectors = _unitary(n, complex_)[:, :3]
    saved_vector = op.apply(vectors[:, 0])
    saved_block = op.apply(vectors)
    op.apply(np.zeros_like(vectors[:, 0]))
    op.apply(np.zeros_like(vectors))
    np.testing.assert_allclose(saved_vector, a @ vectors[:, 0], atol=1e-13)
    np.testing.assert_allclose(saved_block, a @ vectors, atol=1e-13)
    assert not any(np.shares_memory(saved_block, buf) for buf in scratch.values())

    calls.clear()
    solver = "hermitian_davidson" if complex_ else "davidson"
    result = solve_operator(op, SolverOptions(n_roots=3, tol=1e-9), solver=solver)
    np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a)[:3], atol=1e-10)
    np.testing.assert_allclose(result.residuals,
                               np.linalg.norm(a @ result.eigenvectors
                                              - result.eigenvectors * result.eigenvalues, axis=0),
                               atol=1e-13)
    assert result.n_matvec == sum(1 if len(shape) == 1 else shape[1] for shape in calls)


@pytest.mark.parametrize("complex_", [False, True])
def test_sparse_storage_never_densifies(complex_, monkeypatch):
    sp = pytest.importorskip("scipy.sparse")
    dense = np.diag(np.arange(40.)).astype(complex if complex_ else float)
    if complex_:
        dense[0, 8], dense[8, 0] = .01j, -.01j
    a = sp.csr_matrix(dense)
    op = HermitianOperator(matrix=a)
    a.data[:] = 1000.
    def reject_dense(*args, **kwargs):
        pytest.fail("Sparse solve must not materialize a dense matrix")
    monkeypatch.setattr(sp.csr_matrix, "toarray", reject_dense)
    solver = "hermitian_davidson" if complex_ else "davidson"
    result = solve_operator(op, SolverOptions(n_roots=3, tol=1e-10), solver=solver)
    np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(dense)[:3], atol=1e-12)
    assert op.storage == "sparse"
    with pytest.raises(NotImplementedError, match="materialization"):
        solve_operator(op, solver="dense")


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("rank", [40, 48])
@pytest.mark.parametrize("form", ["matvec", "matmat", "sparse"])
@pytest.mark.parametrize("which", ["SA", "LA"])
def test_transformed_davidson_uses_explicit_retained_diagonal(complex_, rank, form, which, monkeypatch):
    scipy = pytest.importorskip("scipy.linalg")
    n = 48
    u = _unitary(n, complex_)
    scales = np.linspace(.4, 2., rank)
    s = (u * np.r_[scales, np.full(n-rank, 1e-14)]) @ u.conj().T
    frame = u[:, :rank]
    x = frame / np.sqrt(scales)
    rng = np.random.default_rng(483)
    noise = rng.normal(scale=.0001, size=(rank, rank))
    reduced = np.diag(np.linspace(-3., 5., rank)) + noise + noise.T
    if complex_:
        reduced = reduced + 1j * (noise - noise.T)
    covariant_frame = frame * np.sqrt(scales)
    a = covariant_frame @ reduced @ covariant_frame.conj().T
    if rank < n:
        # Leakage into a discarded metric direction must remain visible in
        # original residuals; it does not change the projected eigenproblem.
        coupling = .05 * np.outer(u[:, -1], frame[:, 0].conj() + frame[:, -1].conj())
        a = a + coupling + coupling.conj().T
    a.setflags(write=False)
    calls = []

    def action(v):
        calls.append(v.shape)
        return a @ v

    if form == "sparse":
        sparse = pytest.importorskip("scipy.sparse")
        representation = {"matrix": sparse.csr_matrix(a)}
        def reject_dense(*args, **kwargs):
            pytest.fail("Transformed sparse solve must not materialize A")
        monkeypatch.setattr(sparse.csr_matrix, "toarray", reject_dense)
    else:
        representation = {form: action, "n": n, "dtype": a.dtype}
    diagonal = reduced.diagonal().real.copy()
    op = FockOperator(**representation, X=x, overlap=s, reduced_diagonal=diagonal,
                      metadata=OperatorMetadata(kind="fock", basis_id="retained-AO", spin="alpha"))
    # Snapshot construction must not consult A. All frame/preconditioner input
    # arrays may be reused by the producer after construction.
    assert not calls
    x[:] = 0.
    s[:] = 0.
    diagonal[:] = 1000.
    assert not op.reduced_diagonal.flags.writeable
    solver = "hermitian_davidson" if complex_ else "davidson"
    options = SolverOptions(n_roots=3, tol=1e-9, which=which)
    result = solve_operator(op, options, solver=solver)
    reference = scipy.eigh(frame.conj().T @ a @ frame, frame.conj().T @ op.overlap @ frame)[0]
    expected = reference[:3] if which == "SA" else reference[-3:]
    np.testing.assert_allclose(result.eigenvalues, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.eigenvectors.conj().T @ op.overlap @ result.eigenvectors,
                               np.eye(3), atol=1e-12)
    residual = a @ result.eigenvectors - (op.overlap @ result.eigenvectors) * result.eigenvalues
    np.testing.assert_allclose(result.residuals, np.linalg.norm(residual, axis=0), atol=1e-13)
    np.testing.assert_allclose(result.projected_residuals,
                               np.linalg.norm(op.X.conj().T @ residual, axis=0), atol=1e-13)
    assert result.residual_frame == "original"
    assert result.retained_rank == rank
    assert not result.selection_verified
    if rank < n:
        assert max(result.residuals) > .01
    else:
        assert max(result.residuals) < 1e-9
    if form != "sparse":
        # No hidden basis probing, even a column at a time.
        assert result.n_matvec == sum(1 if len(shape) == 1 else shape[1] for shape in calls)
        assert result.n_matvec < rank
        calls.clear()
    warm = solve_operator(op, options, solver=solver, previous=result)
    np.testing.assert_allclose(warm.eigenvalues, expected, atol=1e-10, rtol=0)
    assert warm.n_iter <= result.n_iter


@pytest.mark.parametrize("bad", ["length", "rank2", "imaginary", "nonfinite", "implicit_X", "dense"])
def test_reduced_diagonal_rejects_invalid_frames_without_callback(bad):
    def fail(v):
        pytest.fail("Invalid reduced diagonal must be rejected before applying A")
    kwargs = dict(matvec=fail, n=40, X=np.eye(40)[:, :30], reduced_diagonal=np.ones(30))
    if bad == "length":
        kwargs["reduced_diagonal"] = np.ones(40)
    elif bad == "rank2":
        kwargs["reduced_diagonal"] = np.ones((30, 1))
    elif bad == "imaginary":
        kwargs["reduced_diagonal"] = np.ones(30) + 1j
    elif bad == "nonfinite":
        kwargs["reduced_diagonal"][0] = np.nan
    elif bad == "implicit_X":
        del kwargs["X"]
        kwargs["overlap"] = np.eye(40)
    else:
        del kwargs["matvec"]
        kwargs["matrix"] = np.eye(40)
    with pytest.raises(ValueError, match="reduced_diagonal|finite"):
        FockOperator(**kwargs)


@pytest.mark.parametrize("solver", ["davidson", "hermitian_davidson"])
@pytest.mark.parametrize("explicit_X", [False, True])
def test_transformed_davidson_does_not_infer_reduced_diagonal(solver, explicit_X):
    def fail(v):
        pytest.fail("Missing reduced diagonal must be refused before applying A")
    op = FockOperator(matvec=fail, n=40, diagonal=np.ones(40), overlap=np.eye(40),
                      X=np.eye(40) if explicit_X else None)
    with pytest.raises(NotImplementedError, match="reduced_diagonal"):
        solve_operator(op, SolverOptions(n_roots=2), solver=solver)


@pytest.mark.parametrize("solver", ["davidson", "hermitian_davidson"])
def test_transformed_callback_restart_floor_uses_retained_rank(solver):
    def fail(v):
        pytest.fail("Small retained rank must not trigger a dense callback fallback")
    op = FockOperator(matvec=fail, n=40, X=np.eye(40)[:, :5], reduced_diagonal=np.ones(5))
    with pytest.raises(NotImplementedError, match="restart floor"):
        solve_operator(op, SolverOptions(n_roots=2), solver=solver)


def test_transformed_real_action_keeps_complex_coordinates_without_overlap():
    n = 40
    phases = np.exp(.1j * np.arange(n))
    x = np.diag(phases)
    diagonal = np.arange(n, dtype=float)
    op = FockOperator(matvec=lambda v: diagonal * v, n=n, X=x, reduced_diagonal=diagonal)
    with pytest.raises(NotImplementedError, match="complex reduced"):
        solve_operator(op, SolverOptions(n_roots=2), solver="davidson")
    result = solve_operator(op, SolverOptions(n_roots=2), solver="hermitian_davidson")
    np.testing.assert_allclose(result.eigenvalues, [0., 1.], atol=1e-12)
    assert result.residual_frame == "orthogonal"
    assert np.iscomplexobj(result.eigenvectors)


def test_frozen_factorized_action_survives_next_builder_epoch():
    n = 40
    rng = np.random.default_rng(484)
    factors = rng.normal(scale=.01, size=(n, 4)) + 1j * rng.normal(scale=.01, size=(n, 4))
    h = np.linspace(-3., 5., n)
    scales = np.linspace(.7, 1.3, n)
    x = np.diag(1 / np.sqrt(scales))
    overlap = np.diag(scales)

    def freeze(density, generation):
        # A producer must capture owned immutable state; the operator cannot
        # freeze a closure over a live builder. This test uses a separable
        # one-particle model, not an integral-engine or J/K implementation.
        frozen_h = h.copy()
        frozen_factors = factors * np.sqrt(density)
        frozen_h.setflags(write=False)
        frozen_factors.setflags(write=False)
        reduced_diagonal = (frozen_h + np.sum(abs(frozen_factors)**2, axis=1)) / scales
        scratch = {}
        def apply(v):
            out = scratch.setdefault(v.shape, np.empty_like(v, dtype=complex))
            out[:] = frozen_h[:, None] * v + frozen_factors @ (frozen_factors.conj().T @ v)
            return out
        return FockOperator(matmat=apply, n=n, dtype=complex, X=x, overlap=overlap,
                            reduced_diagonal=reduced_diagonal,
                            metadata=OperatorMetadata(kind="ks", basis_id="fixed-AO",
                                                      generation=generation, spin="alpha"))

    options = SolverOptions(n_roots=3, tol=1e-9)
    first = freeze(.2, 0)
    reference = x.conj().T @ (np.diag(h) + .2 * factors @ factors.conj().T) @ x
    result = solve_operator(first, options, solver="hermitian_davidson")
    h += .1
    factors *= 1.1
    second = freeze(.7, 1)
    next_result = solve_operator(second, options, solver="hermitian_davidson", previous=result)
    repeated = solve_operator(first, options, solver="hermitian_davidson", previous=result)
    np.testing.assert_allclose(repeated.eigenvalues, np.linalg.eigvalsh(reference)[:3], atol=1e-10)
    assert np.min(next_result.eigenvalues - repeated.eigenvalues) > .05
    assert result.operator_metadata.generation == repeated.operator_metadata.generation == 0
    assert next_result.operator_metadata.generation == 1


@pytest.mark.parametrize("solver", ["lobpcg", "jd", "gplhr"])
def test_materializing_callback_backends_are_rejected_before_apply(solver):
    def fail(v):
        pytest.fail("Unsupported callback backend must not apply the operator")
    op = HermitianOperator(matvec=fail, n=40, diagonal=np.ones(40))
    with pytest.raises(NotImplementedError, match="materialization"):
        solve_operator(op, SolverOptions(n_roots=2), solver=solver)
    assert not solver_capabilities(solver).matrix_free


@pytest.mark.parametrize("solver", ["davidson", "hermitian_davidson"])
def test_small_callback_davidson_refuses_hidden_dense_fallback(solver):
    op = HermitianOperator(matvec=lambda v: v, n=5, diagonal=np.ones(5))
    with pytest.raises(NotImplementedError, match="restart floor"):
        solve_operator(op, SolverOptions(n_roots=2), solver=solver)


def test_unsupported_options_are_not_silently_ignored():
    op = HermitianOperator(matrix=np.eye(30))
    for options in (SolverOptions(n_roots=2, block_size=4), SolverOptions(n_roots=2, preconditioner="none")):
        with pytest.raises(NotImplementedError, match="option"):
            solve_operator(op, options, solver="davidson")
    with pytest.raises(NotImplementedError, match="warm"):
        solve_operator(op, SolverOptions(guess_vectors=np.eye(30)), solver="dense")
    with pytest.raises(NotImplementedError, match="selection"):
        solve_operator(op, SolverOptions(n_roots=2), solver="davidson", require_selection=True)
    with pytest.raises(NotImplementedError, match="verbosity"):
        solve_operator(op, SolverOptions(n_roots=2, verbosity=1), solver="hermitian_davidson")


def test_snapshot_validation_and_operator_roles():
    a = np.eye(4)
    op = FockOperator(matrix=a)
    a[0, 0] = 100.
    assert op.matrix[0, 0] == 1.
    with pytest.raises(ValueError, match="Hermitian"):
        HermitianOperator(matrix=np.array([[1., 2.], [0., 1.]]))
    with pytest.raises(ValueError, match="not a Fock"):
        FockOperator(matrix=a, metadata=OperatorMetadata(kind="ci"))
    with pytest.raises(ValueError, match="non-Hermitian"):
        OperatorMetadata(kind="cc_jacobian")
    bad = HermitianOperator(matvec=lambda v: np.ones(2), n=3)
    with pytest.raises(ValueError, match="invalid shape"):
        bad.apply(np.ones(3))


def test_cluster_tracking_is_invariant_to_internal_rotation():
    a = _unitary(10, True)[:, :3]
    b = a @ _unitary(3, True)
    amplitudes, cosines = subspace_overlap(a, b)
    np.testing.assert_allclose(cosines, np.ones(3), atol=1e-12)
    assert not np.allclose(abs(amplitudes), np.eye(3))


def test_nonconvergence_does_not_return_plausible_roots():
    u = _unitary(50)
    a = (u * np.linspace(-5., 5., 50)) @ u.T
    with pytest.raises(RuntimeError, match="contract"):
        solve_operator(HermitianOperator(matrix=a), SolverOptions(n_roots=2, max_iter=1, tol=1e-12), solver="davidson")


def test_harmonic_alias_does_not_certify_nearest_roots():
    a = np.diag(np.linspace(-2., 3., 20))
    r = solve_operator(HermitianOperator(matrix=a), SolverOptions(n_roots=2, sigma=.4248), solver="harmonic_davidson")
    expected = np.sort(np.diag(a)[np.argsort(abs(np.diag(a)-.4248))[:2]])
    np.testing.assert_allclose(np.sort(r.eigenvalues), expected)
    assert not r.selection_verified


def test_run_job_does_not_substitute_davidson_for_lobpcg(tmp_path):
    import vibeqc as vq
    molecule = vq.Molecule([vq.Atom(2, [0., 0., 0.])])
    with pytest.raises(NotImplementedError, match="does not implement LOBPCG"):
        vq.run_job(molecule, method="rhf", basis="sto-3g", solver="lobpcg",
                   output=tmp_path / "unsupported-solver")


def test_output_diagnostics_and_citation_route():
    from vibeqc.output.eigensolver import eigensolver_diagnostics
    from vibeqc.output.citations import load_default_database
    result = solve_operator(FockOperator(matrix=np.eye(2)))
    row = eigensolver_diagnostics(result)
    assert row["selection_verified"] is True
    assert row["operator_kind"] == "fock"
    assert row["operator_vector_applications"] is None
    db = load_default_database()
    citations = db.assemble(numerics=["eigensolver_harmonic_davidson"])
    assert any(c.key == "zuev_gplhr_2015" for c in citations)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_output_diagnostics_preserves_frame_metadata_in_json(with_metadata):
    import json
    from vibeqc.output.eigensolver import eigensolver_diagnostics

    metadata = OperatorMetadata(
        kind="ks", basis_id="example-pair-pno", generation=np.int64(3),
        spin="beta", kpoint=(np.float64(0.25), 0., 0.),
        symmetry_block="example-irrep", occupations=(np.float64(1.), 0.),
    )
    result = solve_operator(FockOperator(matrix=np.diag([0.2, 0.7]), metadata=metadata))
    if not with_metadata:
        # Older solver results have no operator annotations.
        result = replace(result, operator_metadata=None)
    row = eigensolver_diagnostics(result)
    restored = json.loads(json.dumps(row, allow_nan=False))
    assert restored == row
    assert row["operator_vector_applications"] is None
    if with_metadata:
        assert result.operator_metadata is metadata
        assert row["operator_kind"] == "ks"  # existing flat key is retained
        assert row["operator_metadata"] == {
            "kind": "ks", "basis_id": "example-pair-pno", "generation": 3,
            "spin": "beta", "kpoint": [0.25, 0., 0.],
            "symmetry_block": "example-irrep", "occupations": [1., 0.],
        }
        row["operator_metadata"]["kpoint"][0] = 0.5
        row["operator_metadata"]["occupations"][0] = 2.
        assert metadata.kpoint == (0.25, 0., 0.)
        assert metadata.occupations == (1., 0.)
    else:
        assert row["operator_kind"] is None
        assert row["operator_metadata"] is None


def test_real_callback_in_complex_frame_keeps_the_complex_action():
    x = _unitary(6, True)
    diag = np.arange(1., 7.)
    op = HermitianOperator(matvec=lambda v: diag*v, n=6, X=x)
    r = solve_operator(op, SolverOptions(n_roots=2, ncv=6, tol=1e-10), solver="lanczos")
    np.testing.assert_allclose(r.eigenvalues, [1., 2.], atol=1e-12)
    assert max(r.residuals) < 1e-10


def test_single_dimension_lanczos_and_explicit_budget_refusal():
    r = solve_operator(HermitianOperator(matvec=lambda v: 2*v, n=1), solver="lanczos")
    np.testing.assert_allclose(r.eigenvalues, [2.])
    with pytest.raises(ValueError, match="restart floor"):
        solve_operator(HermitianOperator(matrix=np.eye(30)),
                       SolverOptions(n_roots=3, max_subspace=4), solver="davidson")


def test_warm_start_preserves_operator_kind_and_complex_values():
    op = FockOperator(matrix=np.diag(np.arange(30.)))
    previous = solve_operator(op, SolverOptions(n_roots=2))
    response = HermitianOperator(matrix=op.matrix, metadata=OperatorMetadata(kind="response"))
    with pytest.raises(ValueError, match="frame"):
        solve_operator(response, SolverOptions(n_roots=2), solver="davidson", previous=previous)
    guess = previous.eigenvectors.astype(complex) * 1j
    with pytest.raises(NotImplementedError, match="complex warm-start"):
        solve_operator(op, SolverOptions(n_roots=2, guess_vectors=guess), solver="davidson")


def test_exact_unwanted_subspace_is_not_a_root_selection_certificate():
    op = HermitianOperator(matrix=np.diag(np.arange(30.)))
    # An exact invariant seed can have zero residual yet omit every wanted root.
    opts = SolverOptions(n_roots=2, guess_vectors=np.eye(30)[:, -2:])
    result = solve_operator(op, opts, solver="davidson")
    assert max(result.residuals) < 1e-12
    assert not np.allclose(result.eigenvalues, [0., 1.])
    assert not result.selection_verified
    with pytest.raises(NotImplementedError, match="selection"):
        solve_operator(op, opts, solver="davidson", require_selection=True)


def test_lanczos_clustered_spectrum_uses_a_complete_projection():
    u = _unitary(50, True)
    spectrum = 1. + np.linspace(0., 1e-5, 50)
    a = (u * spectrum) @ u.conj().T
    result = solve_operator(HermitianOperator(matrix=a),
                            SolverOptions(n_roots=3, ncv=50, tol=1e-12), solver="lanczos")
    np.testing.assert_allclose(result.eigenvalues, spectrum[:3], atol=1e-12, rtol=0)
    assert max(result.projected_residuals) < 1e-12


def test_lanczos_all_roots_and_insufficient_subspace_rejection():
    op = HermitianOperator(matrix=np.diag(np.arange(25.)))
    result = solve_operator(op, solver="lanczos")
    np.testing.assert_allclose(result.eigenvalues, np.arange(25.), atol=1e-12)
    with pytest.raises(ValueError, match="ncv"):
        solve_operator(op, SolverOptions(n_roots=5, ncv=3), solver="lanczos")


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("which", ["SA", "LA"])
def test_davidson_block_callbacks_and_related_operator_recycling(complex_, which):
    n = 80
    rng = np.random.default_rng(472)
    noise = rng.normal(scale=.001, size=(n, n))
    if complex_:
        noise = noise + 1j*rng.normal(scale=.001, size=(n, n))
    a = np.diag(np.arange(n, dtype=float)) + noise + noise.conj().T
    calls = []
    def block(v):
        calls.append(v.shape)
        assert v.shape[1] < n, "Must not reconstruct A with a full identity block"
        return a @ v
    def scalar(v):
        pytest.fail("A supplied block action must not be split across Python calls")
    op = HermitianOperator(matvec=scalar, matmat=block, n=n, dtype=a.dtype,
                           diagonal=a.diagonal().real, metadata=OperatorMetadata(basis_id="callback-frame"))
    solver = "hermitian_davidson" if complex_ else "davidson"
    opts = SolverOptions(n_roots=3, n_guess=8, which=which, tol=1e-9)
    cold = solve_operator(op, opts, solver=solver)
    wanted = slice(-3, None) if which == "LA" else slice(3)
    np.testing.assert_allclose(cold.eigenvalues, np.linalg.eigvalsh(a)[wanted], atol=1e-10)
    assert cold.n_matvec == sum(shape[1] for shape in calls) < n
    assert any(shape[1] > 1 for shape in calls)
    assert len(calls) < cold.n_matvec
    # A new snapshot is needed when the Fock builder changes its action.
    next_a = a + .01 * (noise + noise.conj().T)
    next_op = HermitianOperator(matmat=lambda v: next_a @ v, n=n, dtype=a.dtype,
                               diagonal=next_a.diagonal().real,
                               metadata=replace(op.metadata, generation=1))
    warm = solve_operator(next_op, opts, solver=solver, previous=cold)
    np.testing.assert_allclose(warm.eigenvalues, np.linalg.eigvalsh(next_a)[wanted], atol=1e-10)
    assert max(warm.residuals) < opts.tol
    assert warm.orthogonality_error < 1e-10
    assert not warm.selection_verified


def test_native_complex_vector_callback_does_not_probe_the_full_basis():
    from vibeqc import _vibeqc_core as core
    n = 80
    a = np.diag(np.arange(n, dtype=float)).astype(complex)
    a[0, 12] = .01j
    a[12, 0] = -.01j
    count = 0
    def matvec(v):
        nonlocal count
        count += 1
        return a @ v
    opts = core.DavidsonOptions()
    opts.n_eig = 3
    opts.conv_tol = 1e-10
    result = core.davidson_solve_hermitian_matvec(n, matvec, a.diagonal().real, opts)
    assert result.converged
    assert count < n
    np.testing.assert_allclose(result.eigenvalues, np.linalg.eigvalsh(a)[:3], atol=1e-12)


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("bad", ["shape", "nonfinite", "diagonal"])
@pytest.mark.parametrize("form", ["matvec", "matmat"])
def test_native_callback_validates_its_contract(complex_, bad, form):
    from vibeqc import _vibeqc_core as core
    opts = core.DavidsonOptions()
    opts.n_eig = 2
    prefix = "davidson_solve_hermitian" if complex_ else "davidson_solve"
    solve = getattr(core, f"{prefix}_{form}")
    diag = np.ones(30 if bad != "diagonal" else 29)
    def callback(v):
        if bad == "shape":
            return v[:-1]
        if bad == "nonfinite":
            return v*np.nan
        pytest.fail("An invalid diagonal must be refused before applying A")
    with pytest.raises(ValueError):
        solve(30, callback, diag, opts)


def test_scf_adapter_owns_options_and_spin_subspaces():
    from types import SimpleNamespace
    from vibeqc import _vibeqc_core as core
    from vibeqc.solvers._scf import FockDiagonalizer
    native = core.DavidsonOptions()
    native.n_eig = 2
    opts = SimpleNamespace(use_davidson=True, davidson_min_dim=0, davidson=native)
    x = np.eye(30)
    alpha = FockDiagonalizer(x, x, opts, spin="alpha", n_required=2)
    beta = FockDiagonalizer(x, x, opts, spin="beta", n_required=1)
    a = np.diag(np.arange(30.))
    b = a[::-1, ::-1]
    ca, ea = alpha(a)
    cb, eb = beta(b)
    assert alpha.last_result.operator_metadata.spin == "alpha"
    assert beta.last_result.operator_metadata.spin == "beta"
    assert native.guess_vectors.size == 0
    assert native.n_eig == 2
    ca[:] = 0.  # The caller's state-following rotations cannot corrupt the cache.
    ca2, ea2 = alpha(a + .01*np.eye(30))
    np.testing.assert_allclose(ea2, ea + .01, atol=1e-12)
    np.testing.assert_allclose(ca2.T @ ca2, np.eye(2), atol=1e-12)
    assert alpha.last_result.operator_metadata.generation == 1
    assert beta.last_result.operator_metadata.generation == 0
    assert abs(cb[29, 0]) == pytest.approx(1.)
    with pytest.raises(ValueError, match="required orbital space"):
        FockDiagonalizer(x, x, opts, n_required=3)


def test_scf_adapter_retains_rectangular_overlap_policy():
    from types import SimpleNamespace
    from vibeqc.solvers._scf import FockDiagonalizer
    s = np.diag([1., 2., 1e-16])
    x = np.array([[1., 0.], [0., 1/np.sqrt(2)], [0., 0.]])
    f = np.array([[1., 0., .3], [0., 4., 0.], [.3, 0., 8.]])
    solve = FockDiagonalizer(x, s, SimpleNamespace(), n_required=1)
    c, e = solve(f)
    np.testing.assert_allclose(e, [1., 2.])
    np.testing.assert_allclose(c.T @ s @ c, np.eye(2), atol=1e-12)
    assert max(solve.last_result.residuals) == pytest.approx(.3)
    assert max(solve.last_result.projected_residuals) < 1e-12


@pytest.mark.parametrize("complex_input", ["guess", "frame", "fock"])
def test_real_only_scf_refuses_complex_data(complex_input):
    from types import SimpleNamespace
    from vibeqc import _vibeqc_core as core
    from vibeqc.solvers._scf import FockDiagonalizer

    native = core.DavidsonOptions()
    native.n_eig = native.n_guess = 2
    options = SimpleNamespace(use_davidson=True, davidson_min_dim=0, davidson=native)
    x, s = np.eye(40), np.eye(40)
    f = np.diag(np.arange(40.) - 2.)
    if complex_input == "guess":
        native.guess_vectors_cplx = x[:, :2] * np.exp(.4j)
    elif complex_input == "frame":
        x = x.astype(complex)
    else:
        f = f.astype(complex)
    with pytest.raises(NotImplementedError, match="Real-only SCF"):
        FockDiagonalizer(x, s, options, n_required=2, real_only=True)(f)
    # The lower complex-capable adapter remains available to consumers
    # that form densities and contractions with conjugate transposes.
    c, eps = FockDiagonalizer(x, s, options, n_required=2)(f)
    np.testing.assert_allclose(eps, [-2., -1.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(2), atol=1e-12)
    assert np.trace(c @ c.conj().T @ s) == pytest.approx(2.)


@pytest.mark.parametrize("complex_", [False, True])
@pytest.mark.parametrize("rank", [4, 7])
def test_dense_scf_adapter_distinguishes_ao_and_projected_frames(complex_, rank):
    from vibeqc.solvers._scf import solve_dense_fock

    u = _unitary(7, complex_)
    w = np.linspace(.2, 2., 7)
    s = (u * w) @ u.conj().T
    x = u[:, :rank] / np.sqrt(w[:rank])
    a = (u * np.linspace(-3., 4., 7)) @ u.conj().T
    # Couple a retained direction to a discarded one: an exact projected
    # solution need not have zero residual in the full AO space.
    a += .1 * (np.outer(u[:, 0], u[:, -1].conj())
               + np.outer(u[:, -1], u[:, 0].conj()))
    projected = x.conj().T @ a @ x
    kwargs = dict(kind="ks", spin="alpha", kpoint=(.25, .1, 0.), generation=3)
    ao = solve_dense_fock(a, X=x, overlap=s, **kwargs)
    reduced = solve_dense_fock(projected, **kwargs)
    np.testing.assert_allclose(ao.eigenvalues, np.linalg.eigvalsh(projected), atol=1e-12)
    np.testing.assert_allclose(reduced.eigenvalues, ao.eigenvalues, atol=1e-12)
    np.testing.assert_allclose(ao.eigenvectors.conj().T @ s @ ao.eigenvectors,
                               np.eye(rank), atol=1e-12)
    np.testing.assert_allclose(reduced.eigenvectors.conj().T @ reduced.eigenvectors,
                               np.eye(rank), atol=1e-12)
    assert ao.eigenvectors.shape == (7, rank)
    assert reduced.eigenvectors.shape == (rank, rank)
    assert ao.residual_frame == "original"
    assert reduced.residual_frame == "orthogonal"
    assert ao.operator_metadata.basis_id != reduced.operator_metadata.basis_id
    assert reduced.operator_metadata.spin == "alpha"
    assert reduced.operator_metadata.kpoint == (.25, .1, 0.)
    assert reduced.operator_metadata.generation == 3
    assert max(reduced.residuals) < 1e-12
    if rank < 7:
        assert max(ao.residuals) > .01
        assert max(ao.projected_residuals) < 1e-12


def test_dense_scf_adapter_requires_a_complete_metric_frame():
    from vibeqc.solvers._scf import solve_dense_fock

    for kwargs in ({"X": np.eye(3)}, {"overlap": np.eye(3)}):
        with pytest.raises(ValueError, match="both X and overlap"):
            solve_dense_fock(np.eye(3), **kwargs)
    # A floored overlap inverse is not an orthonormalizer for the actual S.
    with pytest.raises(ValueError, match="orthonormal"):
        solve_dense_fock(np.eye(3), X=np.diag([1., 1., 1e6]),
                         overlap=np.diag([1., 1., 1e-16]))


@pytest.mark.parametrize("provider", ["direct", "ri"])
def test_manta_spin_builds_feed_frozen_fock_snapshots(provider):
    """Actual producer batches retain receipts while the solver owns its Fock."""
    scipy = pytest.importorskip("scipy.linalg")
    import vibeqc as vq
    from vibeqc import manta
    from vibeqc.fock_state import EnergyComponents, FockBuild, SourceProvenance

    mol = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [.3, -.2, 1.4])])
    basis = vq.BasisSet(mol, "6-31g")
    s = vq.compute_overlap(basis)
    h = vq.compute_kinetic(basis) + vq.compute_nuclear(basis, mol)
    values, vectors = np.linalg.eigh(s)
    x = vectors / np.sqrt(values)
    frame = OperatorFrame(n=basis.nbasis, basis_id="molecular-ao", ordering="libint-standard",
                          image_gauge="molecular", overlap=s, X=x)
    c = x @ _unitary(basis.nbasis, seed=493)
    densities = [np.ascontiguousarray(c[:, [i]] @ c[:, [i]].T) for i in (0, 1)]
    if provider == "direct":
        builder = manta.DirectJKBuilder(basis)
        reference_eri = vq.compute_eri(basis)

        def build_pair(ds):
            batch = builder.build_batch(ds)
            return batch.J[0] + batch.J[1], batch.K, batch.receipt, batch.density_identities
    else:
        options = manta.RIOptions()
        options.mode_block = 2
        # A deliberately small auxiliary space tests this boundary, not fit quality.
        builder = manta.RIJKBuilder(basis, basis, options)
        metric = vq.compute_2c_eri(basis)
        coupling = vq.compute_3c_eri(basis, basis)
        ev, u = np.linalg.eigh(metric)
        keep = ev > max(options.absolute_cutoff, options.relative_cutoff * ev[-1])
        fitted = np.einsum("pr,pij->rij", u[:, keep] / np.sqrt(ev[keep]), coupling)
        reference_eri = np.einsum("rij,rkl->ijkl", fitted, fitted)

        def build_pair(ds):
            pair = builder.build_exchange_pair(*ds)
            total = builder.build(ds[0] + ds[1])
            assert total.receipt.source_identity == pair.receipt.source_identity
            return total.J, [pair.K_alpha, pair.K_beta], pair.receipt, list(pair.density_identities)

    def snapshots(ds, generation):
        j, exchange, receipt, identities = build_pair(ds)
        assert identities == [builder.build(d).receipt.density_identity for d in ds]
        assert len(set(identities)) == 2
        reference_j = np.einsum("abcd,cd->ab", reference_eri, ds[0] + ds[1])
        matrices = []
        for i, spin in enumerate(("alpha", "beta")):
            reference_k = np.einsum("abcd,bd->ac", reference_eri, ds[i])
            f = h + j - exchange[i]
            np.testing.assert_allclose(f, h + reference_j - reference_k, atol=2e-11, rtol=0)
            matrices.append(f)
        e_one = float(np.einsum("ij,ji->", h, ds[0]+ds[1]))
        e_j = .5*float(np.einsum("ij,ji->", j, ds[0]+ds[1]))
        e_k = -.5*sum(float(np.einsum("ij,ji->", k, d)) for k, d in zip(exchange, ds))
        state = FockBuild.from_dense(
            method="UHF", generation=generation,
            source=SourceProvenance.capture("manta", source_identity=receipt.source_identity,
                                             algorithm=provider, kernel="coulomb"),
            energy=EnergyComponents(electronic=e_one+e_j+e_k, nuclear=0.,
                                    components=(("one_electron", e_one), ("coulomb", e_j), ("exchange", e_k))),
            densities=ds, focks=matrices, frames=(frame, frame), spins=("alpha", "beta"),
            kpoints=(None, None), weights=(1., 1.))
        assert all(channel.operator.metadata.density_id == state.density_id for channel in state.channels)
        return matrices, [channel.operator for channel in state.channels], receipt, identities

    matrices, frozen, receipt, identities = snapshots(densities, 0)
    expected = [f.copy() for f in matrices]
    controls = SolverOptions(n_roots=2, tol=1e-10)
    cold = [solve_operator(op, controls, solver="dense") for op in frozen]
    next_densities = [np.ascontiguousarray(.6 * densities[0]),
                      np.ascontiguousarray(.8 * densities[1])]
    next_matrices, next_frozen, next_receipt, next_identities = snapshots(next_densities, 1)
    assert receipt.source_identity == next_receipt.source_identity
    assert identities != next_identities
    assert receipt.density_identity != next_receipt.density_identity
    for d in densities:
        d[:] = 0.
    for f in matrices:
        f[:] = 99.
    builder.build(np.eye(basis.nbasis))
    del builder
    for i, spin in enumerate(("alpha", "beta")):
        np.testing.assert_array_equal(frozen[i].matrix, expected[i])
        repeated = solve_operator(frozen[i], controls, solver="dense")
        warm = solve_operator(next_frozen[i], controls, solver="davidson", previous=cold[i])
        for result, f, generation in [(repeated, expected[i], 0), (warm, next_matrices[i], 1)]:
            np.testing.assert_allclose(result.eigenvalues, scipy.eigh(f, s, eigvals_only=True)[:2], atol=1e-10, rtol=0)
            assert max(result.residuals) < 1e-9
            assert result.orthogonality_error < 1e-12
            assert result.operator_metadata.spin == spin
            assert result.operator_metadata.generation == generation
    with pytest.raises(ValueError, match="operator frame"):
        solve_operator(next_frozen[0], controls, solver="davidson", previous=cold[1])


@pytest.mark.parametrize("unrestricted", [False, True])
@pytest.mark.parametrize("algorithm", ["exact", "ri"])
@pytest.mark.parametrize("iterative", [False, True])
def test_selected_manta_molecular_scf_preserves_spectral_contract(unrestricted, algorithm, iterative):
    """The selected producer reaches the same native Fock solve boundary."""
    scipy = pytest.importorskip("scipy.linalg")
    import vibeqc as vq

    positions = [[0., 0., 0.], [.2, -.1, 1.8]]
    if unrestricted:
        positions.append([1.3, .4, .7])
    mol = vq.Molecule([vq.Atom(1, p) for p in positions], 0, 2 if unrestricted else 1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UHFOptions() if unrestricted else vq.RHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.conv_tol_energy, opts.conv_tol_grad = 1e-12, 1e-10
    opts.stability_check = False
    opts.use_davidson, opts.davidson_min_dim = iterative, 0
    opts.davidson.conv_tol = 1e-11
    # Full-spectrum Davidson uses its iterative path even for this small fixture.
    assert opts.davidson.n_eig == 0
    driver = vq.run_uhf if unrestricted else vq.run_rhf
    result = driver(
        mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"),
        coulomb=vq.CoulombBuildOptions(algorithm, basis if algorithm == "ri" else None),
        exchange=vq.ExchangeBuildOptions(algorithm),
    )
    assert result.converged and result.engine_provenance.stationarity_confirmed
    assert opts.davidson.n_eig == 0 and opts.davidson.guess_vectors.size == 0
    s = vq.compute_overlap(basis)
    for suffix in ("_alpha", "_beta") if unrestricted else ("",):
        f = np.asarray(getattr(result, "fock" + suffix))
        c = np.asarray(getattr(result, "mo_coeffs" + suffix))
        energies = np.asarray(getattr(result, "mo_energies" + suffix))
        np.testing.assert_allclose(energies, scipy.eigh(f, s, eigvals_only=True), atol=1e-9, rtol=0)
        np.testing.assert_allclose(c.T @ s @ c, np.eye(basis.nbasis), atol=1e-12, rtol=0)
        assert np.linalg.norm(f @ c - (s @ c) * energies, axis=0).max() < 1e-9


@pytest.mark.parametrize("unrestricted", [False, True])
def test_selected_manta_molecular_scf_does_not_ignore_davidson_controls(unrestricted):
    import vibeqc as vq

    mol = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [.2, -.1, 1.8])])
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.UHFOptions() if unrestricted else vq.RHFOptions()
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.use_davidson, opts.davidson_min_dim = True, 0
    opts.davidson.n_eig, opts.davidson.n_guess = basis.nbasis, 1
    driver = vq.run_uhf if unrestricted else vq.run_rhf
    with pytest.raises(ValueError, match="n_guess"):
        driver(mol, basis, opts, integral_engine=vq.IntegralEngineOptions(backend="manta"))


@pytest.mark.parametrize("kernel,omega", [("coulomb", 0.), ("erf", .7), ("erfc", .7)])
def test_manta_periodic_spin_snapshots_preserve_complex_operator_contract(kernel, omega):
    """Producer receipts and whole-mesh builds stay outside each frozen solve."""
    scipy = pytest.importorskip("scipy.linalg")
    import vibeqc as vq
    from vibeqc import manta
    from tests.test_manta_periodic_fock import _case

    case = _case(kernel=kernel, omega=omega, scale=.6)
    basis, lattice, momenta = case["basis"], case["lattice"], case["momenta"]
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, s.origin) for s in basis.shells()])
    cutoff = 1.4
    def cells(b):
        return np.array([c.r_cart for c in vq.pair_complete_lattice_cells(b, system, cutoff)])
    case.update(translations=cells(basis), pair_cutoff_bohr=cutoff)
    overlap = manta.compute_bloch_one_electron(basis, "overlap", case["translations"], momenta, cutoff).values
    built = manta.compute_periodic_fock(**case)
    matrices = built.values
    frozen = []
    controls = SolverOptions(n_roots=basis.nbasis, tol=1e-10)
    for spin in range(2):
        channel = []
        for point, k in enumerate(momenta):
            op = FockOperator(matrix=matrices[spin, point], overlap=overlap[point],
                              metadata=OperatorMetadata(kind="fock", basis_id="manta-periodic-ao", spin=("alpha", "beta")[spin],
                                                        kpoint=tuple(k), generation=0))
            result = solve_operator(op, controls)
            np.testing.assert_allclose(result.eigenvalues, scipy.eigh(matrices[spin, point], overlap[point], eigvals_only=True),
                                       atol=1e-10, rtol=0)
            assert result.residuals.max() < 1e-9 and result.orthogonality_error < 1e-12
            channel.append((op, result))
        frozen.append(channel)
    assert np.max(np.abs(matrices.imag)) > 1e-3

    changed = manta.compute_periodic_fock(**dict(case, spin_densities=.8*case["spin_densities"]))
    assert changed.receipt.source_identity == built.receipt.source_identity
    assert changed.receipt.density_identity != built.receipt.density_identity
    assert changed.receipt.snapshot_identity != built.receipt.snapshot_identity
    for spin in range(2):
        for point in range(len(momenta)):
            old, previous = frozen[spin][point]
            current = FockOperator(matrix=changed.values[spin, point], overlap=overlap[point],
                                   metadata=replace(old.metadata, generation=1))
            warm = solve_operator(current, controls, solver="hermitian_davidson", previous=previous)
            np.testing.assert_allclose(warm.eigenvalues, scipy.eigh(current.matrix, overlap[point], eigvals_only=True),
                                       atol=1e-10, rtol=0)
            assert warm.residuals.max() < 1e-9 and warm.orthogonality_error < 1e-12
    for wrong in (frozen[1][0][1], frozen[0][1][1]):
        with pytest.raises(ValueError, match="operator frame"):
            solve_operator(frozen[0][0][0], controls, solver="hermitian_davidson", previous=wrong)

    shells = basis.shells()
    shells[1].origin = (np.asarray(shells[1].origin)+lattice[:, 0]).tolist()
    moved_basis = vq.BasisSet(vq.Molecule([vq.Atom(2, s.origin) for s in shells]), shells, "manta-operator-image", True)
    shifts = np.array([[0., 0., 0.]]+[lattice[:, 0].tolist()]*3)
    phase = np.exp(1j*momenta@shifts.T)
    action = phase[:, :, None]*phase[:, None, :].conj()
    moved_overlap = manta.compute_bloch_one_electron(moved_basis, "overlap", cells(moved_basis), momenta, cutoff).values
    moved = manta.compute_periodic_fock(**dict(case, basis=moved_basis, translations=cells(moved_basis),
        hcore=case["hcore"]*action, spin_densities=case["spin_densities"]*action[None]))
    np.testing.assert_allclose(moved_overlap, overlap*action, atol=3e-12, rtol=0)
    np.testing.assert_allclose(moved.values, matrices*action[None], atol=1e-11, rtol=0)
    assert moved.receipt.electronic_energy == pytest.approx(built.receipt.electronic_energy, abs=1e-11)
    for spin in range(2):
        for point in range(len(momenta)):
            old, previous = frozen[spin][point]
            op = FockOperator(matrix=moved.values[spin, point], overlap=moved_overlap[point],
                              metadata=replace(old.metadata, basis_id="manta-periodic-image-ao"))
            # A gauge change requires an explicit coefficient transformation.
            with pytest.raises(ValueError, match="operator frame"):
                solve_operator(op, controls, solver="hermitian_davidson", previous=previous)
            guess = phase[point, :, None]*previous.eigenvectors
            result = solve_operator(op, replace(controls, guess_vectors=guess), solver="hermitian_davidson")
            np.testing.assert_allclose(result.eigenvalues, previous.eigenvalues, atol=1e-10, rtol=0)
            assert result.residuals.max() < 1e-9 and result.orthogonality_error < 1e-12
            np.testing.assert_allclose(result.eigenvectors @ result.eigenvectors.conj().T,
                                       guess @ guess.conj().T, atol=1e-11, rtol=0)
    for name in ("hcore", "spin_densities"):
        case[name][:] = 0.
    built.values[:] = 0.
    del built, changed, moved
    for spin in range(2):
        for point in range(len(momenta)):
            np.testing.assert_array_equal(frozen[spin][point][0].matrix, matrices[spin, point])


@pytest.mark.parametrize("boundary", ["strict-zero-mode", "BvK-ewald"])
@pytest.mark.parametrize("moved", [False, True])
def test_bounded_manta_periodic_rhf_exports_match_common_metric_contract(boundary, moved):
    """Validate after export without changing the bounded solver's allocations."""
    scipy = pytest.importorskip("scipy.linalg")
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_domains import _arguments

    args, momenta = _arguments((2, 1, 1), (1, 0, 0), cutoff=3.)
    args["options"].exchange_q0 = boundary
    if moved:
        shells = args["basis"].shells()
        shells[0].origin = (np.asarray(shells[0].origin)+np.asarray(args["system"].lattice)[:, 0]).tolist()
        atoms = list(args["system"].unit_cell)
        atoms[0] = vq.Atom(atoms[0].Z, shells[0].origin)
        # Only the basis is relabeled; the physical nuclei stay in their cell.
        args["basis"] = vq.BasisSet(vq.Molecule(atoms), shells, "manta-operator-shell-image", False)
    result = core._manta.run_periodic_rhf_from_cutoffs(**args)
    assert result.converged and result.receipt.confirmed
    if moved:
        assert np.max(np.abs(result.overlap.imag)) > .01
    for point, k in enumerate(momenta):
        f, s, c, energies = result.fock[point], result.overlap[point], result.coefficients[point], result.orbital_energies[point]
        op = FockOperator(matrix=f, overlap=s, metadata=OperatorMetadata(kind="fock", kpoint=tuple(k), basis_id="bounded-export"))
        solved = solve_operator(op, SolverOptions(tol=1e-10))
        np.testing.assert_allclose(energies, scipy.eigh(f, s, eigvals_only=True), atol=1e-9, rtol=0)
        np.testing.assert_allclose(solved.eigenvalues, energies, atol=1e-9, rtol=0)
        np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(len(energies)), atol=1e-11, rtol=0)
        assert np.linalg.norm(f @ c-(s @ c)*energies, axis=0).max() < 1e-9
        np.testing.assert_allclose(result.density[point], 2*c[:, :1] @ c[:, :1].conj().T, atol=1e-9, rtol=0)
        assert solved.residuals.max() < 1e-9 and solved.orthogonality_error < 1e-12


@pytest.mark.parametrize("boundary", ["strict-zero-mode", "BvK-ewald"])
def test_bounded_manta_redundant_metric_exports_keep_retained_rank(boundary):
    scipy = pytest.importorskip("scipy.linalg")
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_rhf import _case

    case, _ = _case()
    case["options"].exchange_q0 = boundary
    system = vq.PeriodicSystem(3, case["system"].lattice, [vq.Atom(2, [0., 0., 0.])])
    shells = []
    for angular in (0, 0, 1):
        shell = core.ShellInfo()
        shell.atom_index, shell.origin, shell.l, shell.pure = 0, [0., 0., 0.], angular, True
        shell.exponents, shell.coefficients = [.9], [1.]
        shells.append(shell)
    basis = vq.BasisSet(system.unit_cell_molecule(), shells, "manta-operator-rank", False)
    result = core._manta.run_periodic_rhf(**dict(case, system=system, basis=basis))
    assert result.converged and result.solver_memory.retained_rank == 4
    f, s, c = result.fock[0], result.overlap[0], result.coefficients[0]
    metric, vectors = scipy.eigh(s)
    keep = metric > max(case["options"].scf.overlap_rank_absolute_floor,
                        case["options"].scf.overlap_rank_relative_floor*metric[-1])
    x = vectors[:, keep]/np.sqrt(metric[keep])
    expected = scipy.eigh(x.conj().T @ f @ x, eigvals_only=True)
    solved = solve_operator(FockOperator(matrix=f, X=x, overlap=s), SolverOptions(tol=1e-10))
    assert solved.retained_rank == 4 and c.shape == (5, 4)
    np.testing.assert_allclose(result.orbital_energies[0], expected, atol=1e-9, rtol=0)
    np.testing.assert_allclose(solved.eigenvalues, expected, atol=1e-10, rtol=0)
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(4), atol=1e-11, rtol=0)
    assert solved.projected_residuals.max() < 1e-10 and solved.residuals.max() < 1e-10


@pytest.mark.parametrize("boundary", ["strict-zero-mode", "BvK-ewald"])
def test_bounded_manta_stopped_scf_keeps_evaluated_density_separate_from_orbitals(boundary):
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_rhf import _case

    case, _ = _case((2, 1, 1), (1, 0, 0))
    shells = case["basis"].shells()
    shells[0].exponents = [.4]
    case["basis"] = vq.BasisSet(case["system"].unit_cell_molecule(), shells, "manta-operator-stopped", False)
    case["options"].scf.maximum_iterations = 1
    case["options"].exchange_q0 = boundary
    result = core._manta.run_periodic_rhf(**case)
    assert not result.converged and not result.receipt.confirmed
    assert result.solver_snapshot.status.name == "ITERATION_LIMIT"
    density = result.density
    c = result.coefficients[:, :, :1]
    orbital_density = 2*c @ c.conj().transpose(0, 2, 1)
    assert np.max(np.abs(density-orbital_density)) > 1e-3
    expected = result.receipt.nuclear_energy + .5/len(density)*np.einsum(
        "kij,kji->", density, result.hcore+result.fock).real
    assert result.receipt.energy == pytest.approx(expected, abs=1e-11)
    for point in range(len(density)):
        solved = solve_operator(FockOperator(matrix=result.fock[point], overlap=result.overlap[point]),
                                SolverOptions(tol=1e-10))
        np.testing.assert_allclose(solved.eigenvalues, result.orbital_energies[point], atol=1e-9, rtol=0)
        np.testing.assert_allclose(2*solved.eigenvectors[:, :1] @ solved.eigenvectors[:, :1].conj().T,
                                   orbital_density[point], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(result.density, density)


@pytest.mark.parametrize("exchange_q0", ["none", "ewald"])
def test_manta_job_history_and_qvf_follow_evaluated_operator_epochs(tmp_path, exchange_q0):
    """Capped jobs preserve the SCF recurrence across source and output gauges."""
    import json
    import zipfile

    scipy = pytest.importorskip("scipy.linalg")
    import vibeqc as vq
    from vibeqc.guess_read import _load_periodic_read_density_k_closed
    from tests.test_manta_periodic_job import _job

    controls = SolverOptions(tol=1e-10)
    ordinary = []
    for moved in (False, True):
        job = _job(tmp_path, shape=(3, 1, 1), shift=(1, 0, 0))
        job.update(exchange_exxdiv=exchange_q0, citations=False,
                   write_xyz_file=False, write_cif_file=False, write_xsf_structure_file=False)
        shells = job["basis"].shells()
        shells[0].exponents = [.4]
        atoms = list(job["system"].unit_cell)
        images = np.array([[1, -1, 0], [0, 1, 0]]) if moved else np.zeros((2, 3), dtype=int)
        for i, shell in enumerate(shells):
            shell.origin = np.asarray(shell.origin) + job["system"].lattice @ images[i]
            atoms[i] = vq.Atom(atoms[i].Z, shell.origin)
        job["basis"] = vq.BasisSet(vq.Molecule(atoms), shells, "manta-operator-job", False)
        previous = [None] * 3
        frozen = []
        history = []
        expected_density = None
        for iteration in (1, 2, 3):
            job.update(max_iter=iteration, output=tmp_path / f"image-{moved}" / f"step-{iteration}")
            result = vq.run_periodic_job(**job)
            assert result.iterations == iteration and not result.converged
            d, f, s, h = [np.asarray(getattr(result, name)) for name in
                           ("density_k", "fock_k", "overlap_k", "hcore_k")]
            fractions = result.kpoints_fractional
            phase = np.exp(2j * np.pi * fractions @ images.T)
            action = phase[:, :, None] * phase[:, None, :].conj()
            if expected_density is None:
                # Independent generalized HCORE eigenvectors define the first D.
                c0 = np.array([scipy.eigh(hk, sk)[1][:, :1] for hk, sk in zip(h, s)])
                expected_density = 2*c0 @ c0.conj().transpose(0, 2, 1)
            np.testing.assert_allclose(d, expected_density, atol=2e-10, rtol=0)
            energy = result.native.receipt.nuclear_energy + .5*np.einsum(
                "k,kij,kji->", result.kpoint_weights, d, h+f).real
            assert result.energy == pytest.approx(energy, abs=1e-11)
            next_density, commutators = [], []
            for point, k in enumerate(fractions):
                op = FockOperator(matrix=f[point], overlap=s[point], metadata=OperatorMetadata(
                    kind="fock", basis_id=f"manta-job-image-{moved}", spin="restricted",
                    kpoint=tuple(k), generation=iteration))
                solved = solve_operator(op, controls, solver="hermitian_davidson", previous=previous[point])
                np.testing.assert_allclose(solved.eigenvalues, scipy.eigh(f[point], s[point], eigvals_only=True),
                                           atol=1e-10, rtol=0)
                np.testing.assert_allclose(solved.eigenvalues, result.mo_energies_k[point], atol=1e-9, rtol=0)
                assert solved.residuals.max() < 1e-9 and solved.orthogonality_error < 1e-12
                c = solved.eigenvectors[:, :1]
                next_density.append(2*c @ c.conj().T)
                metric, vectors = scipy.eigh(s[point])
                x = vectors / np.sqrt(metric)
                fp = x.conj().T @ f[point] @ x
                dp = x.conj().T @ s[point] @ d[point] @ s[point] @ x
                commutators.append(np.linalg.norm(fp @ dp-dp @ fp)**2)
                previous[point] = solved
                frozen.append((op, f[point].copy()))
            expected_density = np.array(next_density)
            closure = np.linalg.norm(d-expected_density, axis=(1, 2)).max()
            snapshot = result.native.scf_history[-1]
            assert snapshot.maximum_density_closure_frobenius == pytest.approx(closure, abs=2e-10)
            assert snapshot.commutator_frobenius_rms == pytest.approx(np.sqrt(np.mean(commutators)), abs=2e-10)
            row = dict(iter=iteration, energy_eh=energy)
            if history:
                row["delta_e"] = energy-history[-1]["energy_eh"]
            history.append(row)
            for actual, expected in zip(result.scf_history, history, strict=True):
                assert actual == pytest.approx(expected, abs=1e-11)
            path = job["output"].with_suffix(".qvf")
            restored = np.asarray(_load_periodic_read_density_k_closed(
                read_from=path, expected_n_k=3, n_basis=2))
            np.testing.assert_allclose(restored, d*action.conj(), atol=2e-11, rtol=0)
            if iteration == 1:
                assert closure > 1e-3
                assert np.max(np.abs(restored-expected_density*action.conj())) > 1e-3
            with zipfile.ZipFile(path) as archive:
                saved = json.loads(archive.read("scf_history/iterations.json"))["iterations"]
            assert saved == result.scf_history
            if moved:
                baseline = ordinary[iteration-1]
                assert np.max(np.abs(d.imag)) > .01
                assert energy == pytest.approx(baseline.energy, abs=2e-11)
                for name in ("density_k", "fock_k", "overlap_k", "hcore_k"):
                    np.testing.assert_allclose(getattr(result, name), np.asarray(getattr(baseline, name))*action,
                                               atol=2e-10, rtol=0)
                np.testing.assert_allclose(restored, baseline.density_k, atol=2e-10, rtol=0)
                assert result.integral_provenance["source_ao_image_shifts"] == images.tolist()
            else:
                ordinary.append(result)
            # Output transport and later jobs must leave each evaluated export intact.
            result.density_k[0][:] = 0.
            np.testing.assert_array_equal(result.density_k, d)
        for op, original in frozen:
            np.testing.assert_array_equal(op.matrix, original)


@pytest.mark.parametrize("field", ["energy", "source", "method", "density", "fock", "weight"])
def test_evaluated_build_rejects_replacement_with_stale_identity(field):
    from vibeqc.fock_state import EnergyComponents, FockBuild, SourceProvenance

    frame = OperatorFrame(n=2, basis_id="synthetic", ordering="declared", image_gauge="home")
    build = FockBuild.from_dense(method="RHF", generation=0,
        source=SourceProvenance.capture("synthetic", kernel="coulomb"),
        energy=EnergyComponents(electronic=-1., nuclear=.2, components=(("one", -1.),)),
        densities=(np.diag([2., 0.]),), focks=(np.diag([-1., 1.]),),
        frames=(frame,), spins=("restricted",), kpoints=(None,), weights=(1.,))
    assert replace(build).identity == build.identity
    changes = {}
    if field == "energy":
        changes[field] = replace(build.energy, nuclear=.3)
    elif field == "source":
        changes[field] = SourceProvenance.capture("synthetic", kernel="screened")
    elif field == "method":
        changes[field] = "RKS"
    else:
        channel = build.channels[0]
        if field == "density":
            channel = replace(channel, density=np.diag([1., 1.]))
        elif field == "weight":
            channel = replace(channel, weight=.5)
        else:
            channel = replace(channel, operator=replace(channel.operator, matrix=np.diag([-2., 1.]), diagonal=None))
        changes["channels"] = (channel,)
    with pytest.raises(ValueError, match="identity.*contents"):
        replace(build, **changes)


def test_evaluated_build_integer_arrays_bind_normalized_snapshot():
    from vibeqc.fock_state import EnergyComponents, FockBuild, SourceProvenance
    frame = OperatorFrame(n=2, basis_id="synthetic", ordering="declared", image_gauge="home")
    state = FockBuild.from_dense(method="RHF", generation=0,
        source=SourceProvenance.capture("synthetic"),
        energy=EnergyComponents(electronic=-1., nuclear=0., components=(("one", -1.),)),
        densities=(np.diag([2, 0]),), focks=(np.diag([-1, 1]),), frames=(frame,),
        spins=("restricted",), kpoints=(None,), weights=(1.,))
    assert replace(state).identity == state.identity
    assert state.channels[0].density.dtype.kind == "f"


@pytest.mark.parametrize("representation", ["frame", "explicit", "implicit"])
@pytest.mark.parametrize("complex_metric", [False, True])
def test_indefinite_metric_cannot_hide_outside_retained_space(representation, complex_metric):
    q = np.eye(3, dtype=complex if complex_metric else float)
    if complex_metric:
        q = np.array([[1., 1j, 0.], [1j, 1., 0.], [0., 0., np.sqrt(2.)]]) / np.sqrt(2.)
    s = q @ np.diag([1., -0.1, 0.]) @ q.conj().T
    x = q[:, :1]
    with pytest.raises(ValueError, match="indefinite"):
        if representation == "frame":
            frame = OperatorFrame(n=3, basis_id="synthetic", ordering="declared",
                                  image_gauge="home", overlap=s, X=x)
            op = FockOperator(matrix=np.eye(3), frame=frame)
        else:
            op = FockOperator(matrix=np.eye(3), overlap=s,
                              X=x if representation == "explicit" else None)
        solve_operator(op)


@pytest.mark.parametrize("representation", ["frame", "explicit", "implicit"])
@pytest.mark.parametrize("discarded", [0., -1e-14])
def test_positive_semidefinite_metric_retains_declared_rank(representation, discarded):
    s = np.diag([1., 2., discarded])
    x = np.diag([1., 1./np.sqrt(2.), 0.])[:, :2]
    f = np.diag([-1., 4., 9.])
    if representation == "frame":
        frame = OperatorFrame(n=3, basis_id="synthetic", ordering="declared",
                              image_gauge="home", overlap=s, X=x)
        op = FockOperator(matrix=f, frame=frame)
    else:
        op = FockOperator(matrix=f, overlap=s, X=x if representation == "explicit" else None)
    result = solve_operator(op)
    np.testing.assert_allclose(result.eigenvalues, [-1., 2.], atol=1e-13)
    np.testing.assert_allclose(result.eigenvectors.conj().T @ s @ result.eigenvectors, np.eye(2), atol=1e-13)
    np.testing.assert_allclose(f @ result.eigenvectors, s @ result.eigenvectors * result.eigenvalues, atol=1e-13)
    assert result.retained_rank == 2


@pytest.mark.parametrize("field,value", [
    ("kpoint", np.zeros((3, 1))), ("kpoint", np.zeros((3, 0))),
    ("kpoint", np.array([0., 0., 1j])), ("kpoint", 0.),
    ("occupations", np.ones((2, 1))), ("occupations", np.array([1., 1j])),
    ("occupations", 1.),
])
def test_metadata_rejects_nonvector_or_complex_coordinates(field, value):
    with pytest.raises(ValueError, match="real.*vector"):
        OperatorMetadata(**{field: value})


def test_metadata_owns_real_coordinate_and_occupation_scalars():
    k = np.array([0., .25, .5])
    occ = np.array([1., .5, 0.])
    metadata = OperatorMetadata(kpoint=k, occupations=occ)
    k[:] = 7.; occ[:] = 8.
    assert metadata.kpoint == (0., .25, .5)
    assert metadata.occupations == (1., .5, 0.)
    assert all(type(x) is float for x in metadata.kpoint + metadata.occupations)
    assert OperatorMetadata(occupations=[]).occupations == ()


@pytest.mark.parametrize("metric,reason", [
    (np.array([[1., 2.], [2., 1.]]), "indefinite"),
    (np.array([[1., 2.], [0., 1.]]), "Hermitian"),
    (np.ones((3, 3)), "shape"),
    (np.array([[1., np.nan], [np.nan, 1.]]), "finite"),
])
def test_root_tracking_refuses_invalid_metric(metric, reason):
    # Each vector is individually normalized for the first two bad metrics;
    # without full metric admission the cross overlap falsely reports 2.
    with pytest.raises(ValueError, match=reason):
        subspace_overlap(np.eye(2)[:, :1], np.eye(2)[:, 1:], overlap=metric)


def test_root_tracking_complex_degenerate_space_with_null_metric_direction():
    s = np.diag([1., 2., 0.])
    a = np.diag([1., 1./np.sqrt(2.), 0.])[:, :2].astype(complex)
    rotation = np.array([[1., 1j], [1j, 1.]]) / np.sqrt(2.)
    b = a @ rotation
    # Null-metric components do not change the physical quotient subspace.
    b[2] = [2j, 3.]
    original = (a.copy(), b.copy(), s.copy())
    amplitudes, cosines = subspace_overlap(a, b, overlap=s)
    np.testing.assert_allclose(amplitudes, rotation, atol=1e-14)
    np.testing.assert_allclose(cosines, [1., 1.], atol=1e-14)
    for value, before in zip((a, b, s), original):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("complex_metric", [False, True])
def test_warm_start_projection_is_metric_covariant(monkeypatch, complex_metric):
    import vibeqc.solvers.eigensolver as backend
    s = np.array([[2., .5, .4], [.5, 1.5, .3], [.4, .3, 1.]])
    phase = np.diag(np.exp(1j*np.array([.2, -.4, .6]))) if complex_metric else np.eye(3)
    s = phase @ s @ phase.conj().T
    w, u = np.linalg.eigh(s[:2, :2])
    x = np.vstack((u / np.sqrt(w), np.zeros((1, 2))))
    seed = np.array([[1.], [.2], [.7]])
    expected = x.conj().T @ s @ seed
    captured = []
    original = backend.solve_eigenproblem
    def record(problem, options, **kwargs):
        captured.append(options.guess_vectors.copy())
        return original(problem, options, **kwargs)
    monkeypatch.setattr(backend, "solve_eigenproblem", record)
    results = []
    for t in (np.eye(3), np.diag([2., .5, 3.])):
        inv = np.linalg.inv(t)
        metric = t.conj().T @ s @ t
        basis = inv @ x
        f = metric @ basis @ np.diag([-1., 2.]) @ basis.conj().T @ metric
        options = SolverOptions(n_roots=1, guess_vectors=inv @ seed, tol=1e-10)
        before = options.guess_vectors.copy()
        result = solve_operator(FockOperator(matrix=f, overlap=metric, X=basis),
                                options, solver="hermitian_davidson")
        np.testing.assert_allclose(captured[-1], expected, atol=1e-13)
        np.testing.assert_array_equal(options.guess_vectors, before)
        assert result.residuals.max() < 1e-10
        assert result.orthogonality_error < 1e-10
        results.append(t @ result.eigenvectors)
    np.testing.assert_allclose(results[0] @ results[0].conj().T,
                               results[1] @ results[1].conj().T, atol=1e-12)


@pytest.mark.parametrize("metric,n,reason", [
    (np.array([[1., 3.], [0., 1.]]), 2, "Hermitian"),
    (np.array([[1., np.nan], [0., 1.]]), 2, "finite"),
    (np.array([[1., np.inf], [0., 1.]]), 2, "finite"),
    (np.eye(2), 3, "shape"),
])
def test_generalized_adapter_refuses_invalid_metric(metric, n, reason):
    with pytest.raises(ValueError, match=reason):
        solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([-1., 2.]),
                           overlap=metric, n=n), solver="dense")


def test_generalized_adapter_preserves_complex_semidefinite_metric():
    u = _unitary(3, True)
    s = (u * np.array([.5, 2., 0.])) @ u.conj().T
    f = (u * np.array([-.5, 4., 7.])) @ u.conj().T
    result = solve_eigenproblem(GeneralizedEigenProblem(matrix=f, overlap=s),
                               SolverOptions(n_roots=2), solver="dense")
    assert result.converged and result.retained_rank == 2
    np.testing.assert_allclose(result.eigenvalues, [-1., 2.], atol=1e-12)
    assert result.residuals.max() < 1e-12
    assert result.orthogonality_error < 1e-12
    c = result.eigenvectors
    np.testing.assert_allclose(c @ c.conj().T, (u[:, :2] / np.array([.5, 2.])) @ u[:, :2].conj().T, atol=1e-12)


@pytest.mark.parametrize("matrix,reason", [
    (np.array([[-1., 3.], [0., 2.]]), "Hermitian"),
    (np.array([[-1., 3j], [0., 2.]]), "Hermitian"),
    (np.array([[-1., np.nan], [0., 2.]]), "finite"),
    (np.ones((1, 2)), "shape"),
])
def test_generalized_adapter_validates_full_fock_before_projection(matrix, reason):
    # The retained block alone is valid for the asymmetric matrices.
    with pytest.raises(ValueError, match=reason):
        solve_eigenproblem(GeneralizedEigenProblem(matrix=matrix,
                           overlap=np.diag([1., 0.]), n=2), solver="dense")


@pytest.mark.parametrize("complex_case", [False, True])
def test_generalized_residual_owns_reused_callback_workspace(complex_case):
    u = _unitary(4, complex_case)
    metric_values = np.array([.5, 1., 2., 3.])
    energies = np.array([-2., -.5, 1., 3.])
    s = (u * metric_values) @ u.conj().T
    f = (u * (metric_values * energies)) @ u.conj().T
    workspace = np.empty(4, dtype=np.result_type(f.dtype, np.complex128))
    def borrowed(v):
        workspace[:] = f @ v
        return workspace
    results = []
    for action in (lambda v: f @ v, borrowed):
        result = solve_eigenproblem(GeneralizedEigenProblem(overlap=s, matvec=action),
                                   SolverOptions(n_roots=4, tol=1e-10),
                                   solver="hermitian_davidson")
        c = result.eigenvectors
        actual = np.linalg.norm(f @ c - (s @ c) * result.eigenvalues, axis=0)
        assert result.converged and actual.max() < 1e-10
        np.testing.assert_allclose(result.residuals, actual, atol=1e-13)
        assert result.orthogonality_error < 1e-12
        results.append(result)
    np.testing.assert_allclose(results[0].eigenvalues, results[1].eigenvalues, atol=1e-12)
    saved = results[1].residuals.copy()
    workspace[:] = 999.
    np.testing.assert_array_equal(results[1].residuals, saved)


@pytest.mark.parametrize("solver", ["davidson", "lobpcg", "jd", "gplhr"])
@pytest.mark.parametrize("callback", [False, True])
def test_generalized_complex_metric_refuses_real_backend(monkeypatch, solver, callback):
    from vibeqc.solvers.eigensolver import get_solver
    def forbidden(*args, **kwargs):
        pytest.fail("Complex reduction reached a real-only backend")
    monkeypatch.setattr(type(get_solver(solver)), "solve", forbidden)
    s = np.array([[1., .2j], [-.2j, 2.]])
    f = np.diag([-1., 3.])
    representation = {"matvec": lambda v: f @ v} if callback else {"matrix": f}
    with pytest.raises(NotImplementedError, match="complex reduced"):
        solve_eigenproblem(GeneralizedEigenProblem(overlap=s, **representation), solver=solver)


@pytest.mark.parametrize("callback", [False, True])
def test_generalized_complex_metric_preserves_capable_solver(callback):
    scipy = pytest.importorskip("scipy.linalg")
    s = np.array([[1., .2j], [-.2j, 2.]])
    f = np.diag([-1., 3.])
    representation = {"matvec": lambda v: f @ v} if callback else {"matrix": f}
    result = solve_eigenproblem(GeneralizedEigenProblem(overlap=s, **representation),
                               SolverOptions(tol=1e-10), solver="hermitian_davidson")
    energies, c = scipy.eigh(f, s)
    assert result.converged and result.residuals.max() < 1e-10
    assert result.orthogonality_error < 1e-12
    np.testing.assert_allclose(result.eigenvalues, energies, atol=1e-12)
    np.testing.assert_allclose(result.eigenvectors[:, :1] @ result.eigenvectors[:, :1].conj().T,
                               c[:, :1] @ c[:, :1].conj().T, atol=1e-12)


@pytest.mark.parametrize("invalid", [np.ones((2, 1)), np.array([np.nan, 1.])])
@pytest.mark.parametrize("phase", ["reduction", "residual"])
def test_generalized_callback_rejects_invalid_action(monkeypatch, invalid, phase):
    from vibeqc.solvers.eigensolver import get_solver, SolverResult
    def controlled(self, problem, options):
        if phase == "reduction":
            problem.matvec(np.ones(2))
        return SolverResult(eigenvalues=np.array([1., 2.]), eigenvectors=np.eye(2),
                            n_iter=1, converged=True, residuals=np.zeros(2))
    monkeypatch.setattr(type(get_solver("hermitian_davidson")), "solve", controlled)
    with pytest.raises(ValueError, match="callback.*shape or nonfinite"):
        solve_eigenproblem(GeneralizedEigenProblem(overlap=np.eye(2),
                           matvec=lambda v: invalid), solver="hermitian_davidson")


@pytest.mark.parametrize("fault", ["energy", "normalization", "backend_failure"])
def test_generalized_status_checks_evaluated_state(monkeypatch, fault):
    from vibeqc.solvers.eigensolver import get_solver, SolverResult
    def claimed(self, problem, options):
        energies = np.array([1., 2.])
        vectors = np.eye(2)
        if fault == "energy":
            energies[0] += .25
        if fault == "normalization":
            vectors[:, 0] *= 2.
        return SolverResult(eigenvalues=energies, eigenvectors=vectors,
                            n_iter=1, converged=fault != "backend_failure",
                            residuals=np.zeros(2), n_converged=2)
    monkeypatch.setattr(type(get_solver("dense")), "solve", claimed)
    result = solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([1., 2.]),
                               overlap=np.eye(2)), solver="dense")
    assert not result.converged
    if fault == "energy":
        np.testing.assert_allclose(result.projected_residuals, [.25, 0.])
        assert result.n_converged == 1
    if fault == "normalization":
        assert result.orthogonality_error == pytest.approx(3.)


def test_generalized_status_preserves_projected_solution():
    f = np.array([[-1., 0., .3], [0., 2., 0.], [.3, 0., 5.]])
    result = solve_eigenproblem(GeneralizedEigenProblem(matrix=f, overlap=np.diag([1., 1., 0.])),
                               SolverOptions(n_roots=2), solver="dense")
    assert result.converged and result.n_converged == 2
    assert result.residuals.max() == pytest.approx(.3)
    np.testing.assert_allclose(result.projected_residuals, [0., 0.], atol=1e-14)
    np.testing.assert_allclose(result.eigenvalues, [-1., 2.], atol=1e-14)


@pytest.mark.parametrize("requested,returned", [(0, 1), (2, 1), (1, 2), (0, 0)])
def test_generalized_refuses_incomplete_root_set(monkeypatch, requested, returned):
    from vibeqc.solvers.eigensolver import get_solver, SolverResult
    def partial(self, problem, options):
        return SolverResult(eigenvalues=np.array([1., 2.])[:returned],
                            eigenvectors=np.eye(2)[:, :returned], n_iter=1,
                            converged=True, residuals=np.zeros(returned))
    monkeypatch.setattr(type(get_solver("dense")), "solve", partial)
    with pytest.raises(RuntimeError, match="requested roots"):
        solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([1., 2.]),
                           overlap=np.eye(2)), SolverOptions(n_roots=requested), solver="dense")


@pytest.mark.parametrize("requested", [0, 1, 2])
def test_generalized_root_count_uses_retained_rank(requested):
    result = solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([-1., 2., 7.]),
                               overlap=np.diag([1., 1., 0.])),
                               SolverOptions(n_roots=requested), solver="dense")
    assert result.converged and result.retained_rank == 2
    assert result.eigenvalues.shape == (requested or 2,)
    np.testing.assert_allclose(result.eigenvalues, np.array([-1., 2.])[:requested or 2])



def test_generalized_real_davidson_refuses_complex_callback():
    f = np.array([[1., .3j], [-.3j, 2.]])
    with pytest.raises(NotImplementedError, match="complex callback"):
        solve_eigenproblem(GeneralizedEigenProblem(overlap=np.eye(2), matvec=lambda v: f @ v),
                           SolverOptions(n_roots=2), solver="davidson")


@pytest.mark.parametrize("complex_action", [False, True])
def test_generalized_callback_complex_capability_preserves_valid_results(complex_action):
    f = np.array([[1., .3j], [-.3j, 2.]]) if complex_action else np.array([[1., .3], [.3, 2.]])
    # Zero imaginary values are valid for a real backend even in complex storage.
    solver = "hermitian_davidson" if complex_action else "davidson"
    action = (lambda v: f @ v) if complex_action else (lambda v: (f @ v).astype(complex))
    result = solve_eigenproblem(GeneralizedEigenProblem(overlap=np.eye(2), matvec=action),
                               SolverOptions(n_roots=2), solver=solver)
    assert result.converged and result.residuals.max() < 1e-10
    assert result.orthogonality_error < 1e-12
    e, c = np.linalg.eigh(f)
    np.testing.assert_allclose(result.eigenvalues, e, atol=1e-12)
    np.testing.assert_allclose(result.eigenvectors[:, :1] @ result.eigenvectors[:, :1].conj().T,
                               c[:, :1] @ c[:, :1].conj().T, atol=1e-12)


@pytest.mark.parametrize("tol", [0., -1., np.nan, np.inf])
@pytest.mark.parametrize("callback", [False, True])
def test_generalized_refuses_invalid_tolerance_before_solve(tol, callback):
    def forbidden(v):
        pytest.fail("Invalid tolerance reached the callback")
    representation = {"matvec": forbidden} if callback else {"matrix": np.diag([1., 2.])}
    with pytest.raises(ValueError, match="tol.*positive.*finite"):
        solve_eigenproblem(GeneralizedEigenProblem(overlap=np.eye(2), **representation),
                           SolverOptions(tol=tol),
                           solver="hermitian_davidson" if callback else "dense")


@pytest.mark.parametrize("seed", [np.ones(2), np.ones((3, 2)), np.ones((2, 1)),
                                  np.array([[np.nan, 0.], [0., 1.]]),
                                  np.array([[np.inf, 0.], [0., 1.]])])
def test_generalized_refuses_invalid_warm_seed(monkeypatch, seed):
    from vibeqc.solvers.eigensolver import get_solver
    def forbidden(*args, **kwargs):
        pytest.fail("Malformed warm seed reached the backend")
    monkeypatch.setattr(type(get_solver("hermitian_davidson")), "solve", forbidden)
    with pytest.raises(ValueError, match="Warm-start vectors"):
        solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([-1., 2.]), overlap=np.eye(2)),
                           SolverOptions(n_roots=2, guess_vectors=seed), solver="hermitian_davidson")


def test_generalized_warm_seed_preserves_complex_retained_space():
    u = _unitary(3, True)
    weights = np.array([.5, 2., 0.])
    s = (u * weights) @ u.conj().T
    f = (u * np.array([-.5, 4., 7.])) @ u.conj().T
    seed = u[:, :2] / np.sqrt(weights[:2])
    before = seed.copy()
    options = SolverOptions(guess_vectors=seed, tol=1e-10)
    result = solve_eigenproblem(GeneralizedEigenProblem(matrix=f, overlap=s), options,
                               solver="hermitian_davidson")
    assert result.converged and result.retained_rank == 2
    assert result.residuals.max() < 1e-10 and result.orthogonality_error < 1e-12
    np.testing.assert_allclose(result.eigenvalues, [-1., 2.], atol=1e-12)
    np.testing.assert_allclose(result.eigenvectors @ result.eigenvectors.conj().T,
                               seed @ seed.conj().T, atol=1e-12)
    np.testing.assert_array_equal(options.guess_vectors, before)


@pytest.mark.parametrize("complex_case", [False, True])
@pytest.mark.parametrize("route", ["default", "explicit_warm", "helper_warm"])
def test_scf_adapter_preserves_requested_retained_spectrum(complex_case, route):
    from vibeqc.solvers.eigensolver import _scf_diagonalize
    u = _unitary(3, complex_case)
    x = u[:, :2] / np.sqrt([.5, 2.])
    f = (u * np.array([-.5, 4., 7.])) @ u.conj().T
    seed = x.copy()
    if route == "helper_warm":
        c, e = _scf_diagonalize(f, x, 1, solver="hermitian_davidson", prev_C_orth=np.eye(2))
    else:
        step = SCFStep(fock=f, X=x) if route == "default" else SCFStep(fock=f, X=x, n_occ=1, prev_C=seed)
        options = None if route == "default" else SolverOptions(n_roots=2, tol=1e-10)
        result = solve_scf_step(step, options, solver="hermitian_davidson")
        assert result.converged and result.retained_rank == 2
        c, e = result.eigenvectors, result.eigenvalues
    np.testing.assert_allclose(e, [-1., 2.], atol=1e-12)
    np.testing.assert_allclose(c @ c.conj().T, x @ x.conj().T, atol=1e-12)
    np.testing.assert_array_equal(seed, x)


@pytest.mark.parametrize("requested", [-1, 1.5, 3])
@pytest.mark.parametrize("callback", [False, True])
def test_generalized_refuses_invalid_root_request_before_backend(monkeypatch, requested, callback):
    from vibeqc.solvers.eigensolver import get_solver
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid root request reached the backend")
    monkeypatch.setattr(type(get_solver("hermitian_davidson")), "solve", forbidden)
    f = np.diag([-1., 2., 7.])
    representation = {"matvec": lambda v: f @ v} if callback else {"matrix": f}
    with pytest.raises(ValueError, match="n_roots"):
        solve_eigenproblem(GeneralizedEigenProblem(overlap=np.diag([1., 1., 0.]), **representation),
                           SolverOptions(n_roots=requested), solver="hermitian_davidson")


@pytest.mark.parametrize("phase", [0., .3])
def test_generalized_real_solver_refuses_complex_seed(phase):
    seed = np.eye(2, dtype=complex) * np.exp(1j*phase)
    with pytest.raises(NotImplementedError, match="complex warm-start"):
        solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([-1., 2.]), overlap=np.eye(2)),
                           SolverOptions(n_roots=2, guess_vectors=seed), solver="davidson")


@pytest.mark.parametrize("solver", ["davidson", "hermitian_davidson"])
def test_generalized_warm_seed_capability_preserves_valid_solve(solver):
    seed = np.eye(2) if solver == "davidson" else np.eye(2)*np.exp(.3j)
    before = seed.copy()
    options = SolverOptions(n_roots=2, guess_vectors=seed, tol=1e-10)
    result = solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([-1., 2.]), overlap=np.eye(2)),
                               options, solver=solver)
    assert result.converged and result.residuals.max() < 1e-10
    assert result.orthogonality_error < 1e-12
    np.testing.assert_allclose(result.eigenvalues, [-1., 2.], atol=1e-12)
    np.testing.assert_allclose(result.eigenvectors @ result.eigenvectors.conj().T, np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(options.guess_vectors, before)


@pytest.mark.parametrize("solver", ["dense", "lobpcg", "jd", "gplhr", "lanczos"])
def test_generalized_refuses_ignored_warm_start(monkeypatch, solver):
    from vibeqc.solvers.eigensolver import get_solver
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported warm start reached backend dispatch")
    monkeypatch.setattr(type(get_solver(solver)), "solve", forbidden)
    with pytest.raises(NotImplementedError, match="warm starts"):
        solve_eigenproblem(GeneralizedEigenProblem(matrix=np.diag([-1., 2.]), overlap=np.eye(2)),
                           SolverOptions(n_roots=1, guess_vectors=np.eye(2)[:, :1]), solver=solver)


@pytest.mark.parametrize("sigma", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("route", ["standard", "generalized", "interior"])
def test_legacy_refuses_nonfinite_effective_shift(monkeypatch, sigma, route):
    from vibeqc.solvers.eigensolver import get_solver
    def forbidden(*args, **kwargs):
        pytest.fail("Nonfinite shift reached backend dispatch")
    monkeypatch.setattr(type(get_solver("jd")), "solve", forbidden)
    f = np.diag([-1., 2.])
    problem = GeneralizedEigenProblem(matrix=f, overlap=np.eye(2)) if route == "generalized" else EigenProblem(matrix=f)
    options = SolverOptions(n_roots=1, sigma=sigma)
    if route == "interior":
        problem = InteriorEigenProblem(problem=problem, sigma=sigma)
        options = SolverOptions(n_roots=1)
    with pytest.raises(ValueError, match="sigma must be finite"):
        solve_eigenproblem(problem, options, solver="jd")


@pytest.mark.parametrize("override", [None, 2.1])
def test_legacy_interior_shift_override_preserves_selection(override):
    options = SolverOptions(n_roots=1, sigma=override)
    problem = InteriorEigenProblem(EigenProblem(matrix=np.diag([-1., 2., 5.])),
                                   sigma=-.9 if override is None else np.nan)
    result = solve_eigenproblem(problem, options, solver="dense")
    np.testing.assert_allclose(result.eigenvalues, [-1. if override is None else 2.])
    assert result.converged and result.residuals.max() < 1e-12
    assert options.sigma == override


@pytest.mark.parametrize("complex_case", [False, True])
@pytest.mark.parametrize("override", [None, -.9])
def test_interior_generalized_preserves_metric_and_retained_roots(complex_case, override):
    u = _unitary(3, complex_case)
    s = (u * np.array([.5, 2., 0.])) @ u.conj().T
    f = (u * np.array([-.5, 4., 7.])) @ u.conj().T
    options = SolverOptions(n_roots=1, sigma=override, tol=1e-10)
    problem = InteriorEigenProblem(GeneralizedEigenProblem(matrix=f, overlap=s), sigma=2.1)
    result = solve_eigenproblem(problem, options, solver="dense")
    expected = 2. if override is None else -1.
    index = 1 if override is None else 0
    c = result.eigenvectors
    target = u[:, index:index+1] / np.sqrt([.5, 2.][index])
    np.testing.assert_allclose(result.eigenvalues, [expected], atol=1e-12)
    np.testing.assert_allclose(c @ c.conj().T, target @ target.conj().T, atol=1e-12)
    np.testing.assert_allclose(c.conj().T @ s @ c, np.eye(1), atol=1e-12)
    assert result.converged and result.retained_rank == 2
    assert result.residuals.max() < 1e-10
    assert options.sigma == override


@pytest.mark.parametrize("metric", [np.diag([1., -1.]), np.array([[1., .2], [0., 1.]])])
def test_interior_generalized_refuses_invalid_metric(metric):
    problem = InteriorEigenProblem(GeneralizedEigenProblem(matrix=np.diag([-1., 2.]), overlap=metric))
    with pytest.raises(ValueError, match="overlap"):
        solve_eigenproblem(problem, SolverOptions(n_roots=1), solver="dense")


@pytest.mark.parametrize("tol", [np.inf, -np.inf, np.nan, 0., -1.])
@pytest.mark.parametrize("interior", [False, True])
def test_legacy_standard_refuses_invalid_tolerance_before_backend(monkeypatch, tol, interior):
    from vibeqc.solvers.eigensolver import get_solver
    solver = "jd" if interior else "lanczos"
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid tolerance reached backend dispatch")
    monkeypatch.setattr(type(get_solver(solver)), "solve", forbidden)
    problem = EigenProblem(matrix=np.diag([-1., 2., 5.]))
    if interior:
        problem = InteriorEigenProblem(problem, sigma=2.1)
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        solve_eigenproblem(problem, SolverOptions(n_roots=1, tol=tol), solver=solver)


def test_legacy_lanczos_infinite_tolerance_cannot_certify_bad_ritz_vector():
    with pytest.raises(ValueError, match="tol must be positive and finite"):
        solve_eigenproblem(EigenProblem(matrix=np.diag([-1., 2., 5.])),
                           SolverOptions(n_roots=1, tol=np.inf), solver="lanczos")


@pytest.mark.parametrize("complex_case", [False, True])
def test_legacy_lanczos_valid_tolerance_preserves_projector(complex_case):
    u = _unitary(3, complex_case)
    f = (u * np.array([-1., 2., 5.])) @ u.conj().T
    result = solve_eigenproblem(EigenProblem(matrix=f), SolverOptions(n_roots=1, tol=1e-10), solver="lanczos")
    assert result.converged and result.residuals.max() < 1e-10
    np.testing.assert_allclose(result.eigenvalues, [-1.], atol=1e-12)
    c = result.eigenvectors
    np.testing.assert_allclose(c @ c.conj().T, u[:, :1] @ u[:, :1].conj().T, atol=1e-12)
    np.testing.assert_allclose(c.conj().T @ c, np.eye(1), atol=1e-12)


@pytest.mark.parametrize("n_roots,ncv", [(2, 1), (4, None)])
@pytest.mark.parametrize("callback", [False, True])
def test_legacy_lanczos_refuses_truncated_root_request(n_roots, ncv, callback):
    def forbidden(v):
        pytest.fail("Impossible root request evaluated the operator")
    problem = EigenProblem(matvec=forbidden, n=3) if callback else EigenProblem(matrix=np.eye(3))
    with pytest.raises(ValueError, match="n_roots.*Krylov"):
        solve_eigenproblem(problem, SolverOptions(n_roots=n_roots, ncv=ncv), solver="lanczos")


def test_legacy_lanczos_breakdown_does_not_certify_missing_roots():
    result = solve_eigenproblem(EigenProblem(matrix=np.eye(3)),
                               SolverOptions(n_roots=2, ncv=3), solver="lanczos")
    assert len(result.eigenvalues) == 1
    assert not result.converged
    assert result.residuals.max() < 1e-10


@pytest.mark.parametrize("solver,interior", [("dense", False), ("lobpcg", False), ("jd", False), ("gplhr", False), ("lanczos", False), ("dense", True), ("jd", True), ("gplhr", True)])
def test_legacy_standard_refuses_ignored_warm_start(monkeypatch, solver, interior):
    from vibeqc.solvers.eigensolver import get_solver
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported warm start reached backend dispatch")
    monkeypatch.setattr(type(get_solver(solver)), "solve", forbidden)
    problem = EigenProblem(matrix=np.diag([-1., 2.]))
    if interior:
        problem = InteriorEigenProblem(problem, sigma=-.9)
    options = SolverOptions(n_roots=1, guess_vectors=np.eye(2)[:, :1])
    with pytest.raises(NotImplementedError, match="warm starts"):
        solve_eigenproblem(problem, options, solver=solver)


@pytest.mark.parametrize("solver", ["davidson", "hermitian_davidson"])
def test_legacy_standard_warm_capability_preserves_valid_solve(solver):
    seed = np.eye(2) if solver == "davidson" else np.eye(2)*np.exp(.3j)
    before = seed.copy()
    options = SolverOptions(n_roots=2, guess_vectors=seed, tol=1e-10)
    f = np.diag([-1., 2.])
    result = solve_eigenproblem(EigenProblem(matrix=f), options, solver=solver)
    assert result.converged
    c = result.eigenvectors
    np.testing.assert_allclose(result.eigenvalues, [-1., 2.], atol=1e-12)
    np.testing.assert_allclose(f @ c, c * result.eigenvalues, atol=1e-12)
    np.testing.assert_allclose(c @ c.conj().T, np.eye(2), atol=1e-12)
    np.testing.assert_array_equal(seed, before)
