"""Algebraic routing controls; synthetic gradients are not OMx validation."""

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc.hessian as hessian
import vibeqc.semiempirical.runner as semi_runner
from vibeqc.output.formats.vibrational import (
    FD_HESSIAN_SURFACES,
    hessian_surface_lines,
    hessian_surface_manifest_fields,
)


@pytest.fixture
def quadratic(monkeypatch):
    """A non-diagonal saddle distinguishes sign, ordering and mode loss."""
    def atom(z, xyz):
        return SimpleNamespace(Z=z, xyz=list(xyz))

    def molecule(atoms, charge=0, multiplicity=1):
        return SimpleNamespace(atoms=atoms, charge=charge, multiplicity=multiplicity)

    monkeypatch.setattr(hessian, "Atom", atom)
    monkeypatch.setattr(hessian, "Molecule", molecule)
    mol = molecule([atom(1, [0.2, -0.1, 0.7]), atom(8, [1.3, 0.4, -0.5])], -1, 3)
    matrix = np.diag([-2., 3., 4., 5., 6., 7.])
    matrix[0, 4] = matrix[4, 0] = 0.4
    records = []

    def run(method, displaced):
        xyz = np.array([a.xyz for a in displaced.atoms]).ravel()
        records.append((method, displaced, xyz.copy()))
        return SimpleNamespace(
            converged=True, energy=float(xyz @ matrix @ xyz / 2),
            gradient=lambda: (matrix @ xyz).reshape(2, 3),
            parameter_identity=method, parameter_sha256="synthetic-constant",
        )

    monkeypatch.setattr(semi_runner, "run_semiempirical", run)
    opts = hessian.HessianFDOptions(
        project_trans_rot=False, atomic_masses_amu=[1., 16.],
    )
    return mol, matrix, records, opts


@pytest.mark.parametrize("method", ["om2", "OM3"])
def test_omx_centered_hessian_sign_units_order_and_spin(quadratic, method):
    mol, expected, records, opts = quadratic
    result = hessian.compute_hessian_fd(mol, "", method, hessian_options=opts)
    # Polynomial central differences are exact apart from floating arithmetic.
    np.testing.assert_allclose(result.hessian, expected, rtol=0, atol=3e-12)
    weights = np.repeat(1 / np.sqrt(np.array([1., 16.]) * hessian._AMU_TO_ELECTRON_MASS), 3)
    np.testing.assert_allclose(result.hessian_mw, expected * weights[:, None] * weights, atol=3e-15)
    assert result.imaginary_count == 1
    assert result.frequencies_cm1[0] < 0
    assert result.n_displacements == len(records) == 12
    origin = np.array([a.xyz for a in mol.atoms]).ravel()
    for j, (name, displaced, xyz) in enumerate(records):
        assert name == method.lower()
        assert (displaced.charge, displaced.multiplicity) == (-1, 3)
        assert [a.Z for a in displaced.atoms] == [1, 8]
        delta = np.zeros(6)
        delta[j // 2] = opts.step_bohr * (1 if j % 2 == 0 else -1)
        np.testing.assert_allclose(xyz - origin, delta, atol=1e-15)


def test_omx_partial_hessian_displaces_only_active_atom(quadratic):
    mol, expected, records, opts = quadratic
    opts.frozen_indices = [0]
    opts.project_trans_rot = True
    result = hessian.compute_hessian_fd(mol, "", "om2", hessian_options=opts)
    np.testing.assert_allclose(result.hessian, expected[3:, 3:], atol=3e-12)
    assert result.n_displacements == len(records) == 6
    assert result.unfrozen_indices.tolist() == [1]


@pytest.mark.parametrize("fault,exception,match", [
    ("converged", RuntimeError, "SCF failed"),
    ("energy", ValueError, "non-finite OMx energy"),
    ("nan_gradient", ValueError, "non-finite OMx gradient"),
    ("shape", ValueError, "shape"),
    ("inner_failure", RuntimeError, "inner SCF"),
    ("parameter_change", RuntimeError, "parameters changed"),
    ("missing_identity", RuntimeError, "missing OMx parameter"),
])
def test_omx_displaced_failures_do_not_produce_hessian(quadratic, monkeypatch, fault, exception, match):
    mol, _, records, opts = quadratic
    original = semi_runner.run_semiempirical

    def fail(method, displaced):
        result = original(method, displaced)
        if len(records) != 3:  # also check consistency across different columns
            return result
        if fault == "converged":
            result.converged = False
        elif fault == "energy":
            result.energy = np.nan
        elif fault == "nan_gradient":
            result.gradient = lambda: np.full((2, 3), np.nan)
        elif fault == "shape":
            result.gradient = lambda: np.zeros((3, 2))
        elif fault == "inner_failure":
            def bad_gradient():
                raise RuntimeError("inner SCF failed")
            result.gradient = bad_gradient
        elif fault == "parameter_change":
            result.parameter_sha256 = "synthetic-changed"
        else:
            result.parameter_identity = None
        return result

    monkeypatch.setattr(semi_runner, "run_semiempirical", fail)
    with pytest.raises(exception, match=match):
        hessian.compute_hessian_fd(mol, "", "om2", hessian_options=opts)


@pytest.mark.parametrize("option", ["basis", "scf_options", "grid_options", "gradient_options", "dipole", "step"])
def test_omx_unsupported_options_fail_before_scf(quadratic, option):
    mol, _, records, opts = quadratic
    kwargs = {}
    if option in {"scf_options", "grid_options", "gradient_options"}:
        kwargs[option] = object()
    if option == "dipole":
        opts.include_dipole_derivatives = True
    if option == "step":
        opts.step_bohr = np.nan
    with pytest.raises((ValueError, NotImplementedError)):
        hessian.compute_hessian_fd(mol, "sto-3g" if option == "basis" else "", "om2", hessian_options=opts, **kwargs)
    assert records == []


@pytest.mark.parametrize("method", ["om2", "om3"])
def test_omx_surface_is_explicit_and_basis_free(method):
    assert method in FD_HESSIAN_SURFACES
    fields = hessian_surface_manifest_fields(requested_method=method, resolved_method=method, basis="ignored")
    assert fields["surface"] == method.upper()
    assert fields["gradient_initial_step_bohr"] == 0.001
    assert fields["gradient_step_refinement_factor"] == 0.1
    assert fields["gradient_max_step_refinements"] == 3
    assert fields["gradient_accepted_steps_available"] is False
    assert fields["gradient_route"] == "native central energy differences"
    text = hessian_surface_lines(requested_method=method, resolved_method=method)
    assert "numerical energy gradients" in text
    assert "initial inner step 0.001 bohr" in text
    assert "accepted inner steps are not reported" in text
    assert "electronic stability are not established" in text


def test_other_basis_free_surfaces_remain_gated():
    assert not {"om1", "pm6", "msindo", "dftb0", "scc_dftb", "gfn2_xtb", "mace", "om2_seccm"} & FD_HESSIAN_SURFACES
