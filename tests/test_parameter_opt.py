"""Parameter-opt contract, fitting and native isolation regressions (#676)."""

import json
from dataclasses import replace

import numpy as np
import pytest
from vibeqc.parameter_opt import (
    Dataset,
    DFTBRepulsion,
    EvaluationFailure,
    FitProblem,
    Observation,
    ParameterSet,
    ParameterSpec,
    PM6Block,
    ReferenceProtocol,
    ReferenceRecord,
    System,
    paired_delta,
)
from vibeqc.parameter_opt.schema import digest


def system(r, parent="hydrogen"):
    return System(parent, (1, 1), ((0, 0, 0), (0, 0, r)))


def protocol(method="oracle"):
    return ReferenceProtocol("synthetic", method, "none", "analytic", "1", "{}")


def reference(r, value, parent="hydrogen", method="oracle"):
    return ReferenceRecord(
        system(r, parent),
        protocol(method),
        value,
        0.0,
        digest([r, value]),
        "synthetic-v1",
    )


def dataset(fn, distances=(1.1, 1.4, 1.7, 2.0)):
    anchor = reference(1.0, fn(1.0))
    return Dataset(
        tuple(
            Observation(str(r), (reference(r, fn(r)), anchor), (1, -1))
            for r in distances
        )
    )


class Polynomial:
    """Exactly separable quadratic, with a nonlinear exponential coefficient."""

    def validate(self, parameters):
        pass

    def identity(self):
        return {"adapter": "analytic-polynomial-v1"}

    def energy(self, parameters, system):
        r = system.positions[1][2]
        a, t = parameters.values
        return a * r + np.exp(t) * r * r

    def linear_terms(self, parameters, system):
        r = system.positions[1][2]
        return np.exp(parameters.values[1]) * r * r, (r,)


def parameters():
    return ParameterSet.initial(
        (
            ParameterSpec("a", "hartree/bohr", 0.5, -3, 3, linear=True),
            ParameterSpec("t", "1", 0, -2, 2),
        )
    )


def test_immutable_roundtrip_and_units():
    xyz = [[0, 0, 0], [0, 0, 1.4]]
    s = System("h2", [1, 1], xyz)
    xyz[1][2] = 8
    assert s.positions[1][2] == 1.4
    d = dataset(lambda r: r * r)
    assert Dataset.from_dict(json.loads(json.dumps(d.to_dict()))).sha256 == d.sha256
    with pytest.raises(ValueError, match="hartree"):
        replace(protocol(), unit="eV")
    with pytest.raises(ValueError, match="missing protocol"):
        ReferenceProtocol("dlpno", "DLPNO-CCSD(T1)", "def2-TZVP", "ORCA", "6", "{}")
    with pytest.raises(ValueError, match="outside bounds"):
        parameters().replace((9, 0))


def test_pairing_gauges_and_convergence():
    low = reference(1.4, -1)
    high = reference(1.4, -1.1, method="high")
    assert paired_delta(low, high)["delta_hartree"] == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="identical geometry"):
        paired_delta(low, reference(1.5, -1.1, method="high"))
    with pytest.raises(ValueError, match="mixes"):
        Observation("bad", (low, high), (1, -1))
    with pytest.raises(ValueError, match="unbalanced"):
        Observation("bad", (low, reference(1.5, -1.1)), (1, -2))
    with pytest.raises(ValueError, match="converged"):
        replace(low, converged=False)


def test_split_keeps_conformers_fidelities_and_reaction_components_together():
    rows = []
    for i in range(20):
        for method in ("low", "high"):
            a = reference(1 + i * 0.1, -1, parent=f"p{i}", method=method)
            b = reference(1.01 + i * 0.1, -0.9, parent=f"p{i}", method=method)
            rows.append(Observation(f"{i}-{method}", (a, b), (1, -1)))
    d = Dataset(tuple(rows))
    splits = d.split(seed=12)
    assert splits == d.split(seed=12)
    groups = [set().union(*(o.parents for o in split)) for split in splits.values()]
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    assert sum(map(len, splits.values())) == 40
    with pytest.raises(ValueError, match="parent labels"):
        Dataset(
            (
                rows[0],
                Observation(
                    "alias",
                    (
                        replace(
                            rows[0].records[0],
                            system=replace(rows[0].records[0].system, parent="alias"),
                        ),
                        rows[0].records[1],
                    ),
                    (1, -1),
                ),
            )
        )


def test_conditional_fit_matches_joint_and_restarts(tmp_path):
    from scipy.optimize import least_squares

    d = dataset(lambda r: 1.2 * r + np.exp(0.3) * r * r)
    problem = FitProblem(Polynomial(), parameters(), d)
    checkpoint = tmp_path / "fit.json"
    result = problem.fit(checkpoint=checkpoint)
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["values"], [1.2, 0.3], atol=1e-6)
    joint = least_squares(
        lambda x: problem.residual(parameters().replace(x)),
        [0.5, 0],
        bounds=([-3, -2], [3, 2]),
    )
    np.testing.assert_allclose(result["values"], joint.x, atol=1e-6)
    restarted = FitProblem(Polynomial(), parameters(), d).fit(
        checkpoint=checkpoint, resume=True
    )
    np.testing.assert_allclose(restarted["values"], result["values"], atol=1e-6)
    changed = FitProblem(Polynomial(), parameters(), dataset(lambda r: 2 * r * r))
    with pytest.raises(ValueError, match="identity mismatch"):
        changed.fit(checkpoint=checkpoint, resume=True)
    jac1 = problem.jacobian(parameters(), step=1e-4)
    jac2 = problem.jacobian(parameters(), step=5e-5)
    np.testing.assert_allclose(jac1, jac2, rtol=1e-7)


def test_bound_active_linear_and_prior():
    d = dataset(lambda r: 6 * r + r * r)
    p = parameters()
    problem = FitProblem(Polynomial(), p, d)
    assert problem.conditional(p).values[0] == pytest.approx(3, abs=1e-7)
    p = ParameterSet((replace(p.specs[0], prior_sigma=1e-6), p.specs[1]), p.values)
    assert FitProblem(Polynomial(), p, d).conditional(p).values[0] == pytest.approx(
        0.5, abs=1e-4
    )


def test_failure_never_drops_rows(tmp_path):
    class Broken(Polynomial):
        def energy(self, p, s):
            if s.positions[1][2] > 1.5:
                return np.nan
            return super().energy(p, s)

    problem = FitProblem(Broken(), parameters(), dataset(lambda r: r * r))
    with pytest.raises(EvaluationFailure, match="non-finite"):
        problem.fit(checkpoint=tmp_path / "failed.json")
    assert len(problem.dataset.observations) == 4
    assert problem.evaluations > 0


def test_native_dftb_recovery_and_candidate_isolation(tmp_path):
    adapter = DFTBRepulsion(((1, 1, 1.5),))
    initial = ParameterSet.initial(adapter.specs(initial=0.8))
    target = initial.replace((1.6,))
    d = dataset(lambda r: adapter.energy(target, system(r)))
    before = adapter.energy(initial, system(1.4))
    problem = FitProblem(adapter, initial, d)
    result = problem.fit(checkpoint=tmp_path / "native.json")
    assert result["values"][0] == pytest.approx(1.6, abs=1e-8)
    assert adapter.energy(initial, system(1.4)) == before
    assert result["adapter"]["maturity"] == "preview"
    offset, row = adapter.linear_terms(target, system(1.4))
    assert offset + row[0] * 1.6 == pytest.approx(
        adapter.energy(target, system(1.4)), abs=1e-12
    )


def test_native_pm6_recovery_derivative_and_isolation():
    adapter = PM6Block(((1, "betas"),))
    p = ParameterSet.initial(adapter.specs())
    target = p.replace((p.values[0] + 0.4,))
    d = dataset(lambda r: adapter.energy(target, system(r)))
    before = adapter.energy(p, system(1.4))
    problem = FitProblem(adapter, p, d)
    result = problem.fit(max_nfev=30)
    assert result["status"] == "converged"
    assert result["values"][0] == pytest.approx(target.values[0], abs=1e-5)
    np.testing.assert_allclose(
        problem.jacobian(p, 1e-3), problem.jacobian(p, 5e-4), rtol=1e-5, atol=1e-6
    )
    assert adapter.energy(p, system(1.4)) == before
    model = adapter.make_model(target, system(1.4))
    assert (
        model.params.content_sha256()
        != adapter.make_model(p, system(1.4)).params.content_sha256()
    )
    assert result["adapter"]["maturity"] == "experimental"


def test_normalized_ingestion_and_convergence_dimensions(tmp_path):
    from dataclasses import asdict

    from vibeqc.parameter_opt import (
        convergence_check,
        read_references,
        reference_request,
    )

    settings = {
        "scf": "VeryTightSCF",
        "relativity": "none",
        "ecp": "none",
        "state_protocol": "RHF",
        "grid": "DEFGRID3",
        "dispersion": "none",
        "functional": "PBE0",
        "exchange": "25 percent exact",
    }
    p = ReferenceProtocol(
        "hybrid", "PBE0", "def2-SVP", "ORCA", "6.1.1", json.dumps(settings)
    )
    coarse = replace(reference(1.4, -1), protocol=p)
    fine = replace(coarse, protocol=replace(p, basis="def2-TZVP"), energy=-1.001)
    verdict = convergence_check(coarse, fine, varied=["basis"], tolerance=0.002)
    assert verdict["within_tolerance"]
    with pytest.raises(ValueError, match="dimensions"):
        convergence_check(
            coarse,
            replace(fine, protocol=replace(fine.protocol, method="PBE")),
            varied=["basis"],
            tolerance=0.002,
        )
    target = tmp_path / "references.json"
    data = {
        "schema": "vibeqc.parameter-opt.references.v1",
        "coordinate_unit": "bohr",
        "energy_unit": "hartree",
        "records": [asdict(coarse), asdict(fine)],
    }
    target.write_text(json.dumps(data))
    assert read_references(target) == (coarse, fine)
    data["coordinate_unit"] = "angstrom"
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bohr"):
        read_references(target)
    assert reference_request(coarse.system, p)["request_sha256"]


def test_ties_and_shared_reference_covariance():
    p = ParameterSet.initial(
        (
            ParameterSpec("a", "1", 1, 0, 3),
            ParameterSpec("b", "1", 1, 0, 3, tied_to="a"),
        )
    )
    with pytest.raises(ValueError, match="tied values"):
        p.replace([1, 2])
    d = dataset(lambda r: r * r)
    rows = tuple(
        replace(o, records=(o.records[0], replace(o.records[1], uncertainty=0.001)))
        for o in d.observations
    )
    correlated = Dataset(rows)
    problem = FitProblem(Polynomial(), parameters(), correlated)
    residual = problem.predictions(parameters()) - problem.targets
    covariance = np.eye(4) * 1e-6 + np.ones((4, 4)) * 1e-6
    expected = residual @ np.linalg.solve(covariance, residual) / 4
    assert problem.residual(parameters()) @ problem.residual(
        parameters()
    ) == pytest.approx(expected)


def test_parameter_roundtrip_budget_and_failed_checkpoint(tmp_path):
    p = parameters()
    assert (
        ParameterSet.from_dict(json.loads(json.dumps(p.to_dict()))).sha256 == p.sha256
    )
    problem = FitProblem(
        Polynomial(), p, dataset(lambda r: 1.2 * r + np.exp(0.3) * r * r)
    )
    result = problem.fit(max_nfev=1)
    assert result["status"] == "budget_exhausted"
    assert result["linear_diagnostics"]["data_rank"] == 1

    class Broken(Polynomial):
        def linear_terms(self, p, s):
            raise EvaluationFailure("forced failure")

    checkpoint = tmp_path / "failed.json"
    with pytest.raises(EvaluationFailure):
        FitProblem(Broken(), p, problem.dataset).fit(checkpoint=checkpoint)
    failure = json.loads(checkpoint.read_text())
    assert failure["status"] == "failed"
    assert failure["loss"] is None
    assert failure["evaluations"] == 1


def test_duplicate_scaled_contrast_and_runtime_mismatch(tmp_path):
    d = dataset(lambda r: r * r)
    row = d.observations[0]
    with pytest.raises(ValueError, match="duplicate contrast"):
        Dataset((row, replace(row, name="scaled", coefficients=(2, -2))))
    checkpoint = tmp_path / "restart.json"
    FitProblem(Polynomial(), parameters(), d).fit(checkpoint=checkpoint)

    class Changed(Polynomial):
        def identity(self):
            return {"adapter": "analytic-polynomial-v2"}

    with pytest.raises(ValueError, match="identity mismatch"):
        FitProblem(Changed(), parameters(), d).fit(checkpoint=checkpoint, resume=True)
