"""Sealed HF-state transport; synthetic states, no native scientific verdict."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import os

import numpy as np
import pytest

from vibeqc.periodic.chi import reference as r


@dataclass(frozen=True)
class Diagnostic:
    electronic_method: str = "UHF"


@dataclass(frozen=True)
class Support:
    policy: str = "physical_pairs"


def accepted(monkeypatch):
    monkeypatch.setattr(r, "loaded_process_provenance", lambda: {"runtime": "test"})
    state = SimpleNamespace(
        energy=-1., e_nuclear=.4, converged=True,
        density_alpha=np.diag([1., 0.])[None].astype(complex),
        density_beta=np.zeros((1, 2, 2), complex),
        fock_alpha=np.diag([-1., 2.])[None].astype(complex),
        fock_beta=np.diag([-.5, 2.])[None].astype(complex),
        occupations_alpha=np.array([[1., 0.]]), occupations_beta=np.array([[0., 0.]]),
        overlap=np.eye(2)[None], hcore=np.diag([-1., 2.])[None],
        kpoint_weights=np.array([1.]), kpoints_cart=np.zeros((1, 3)),
        aiccm2026dev_b=Diagnostic(), executed_support=Support(),
        process_provenance={"runtime": "test"},
        chi_scf_acceptance=dict(final_rebuild=True, stationary=True,
                               stability="not-assessed"))
    ref = r.ChiHFReference(state, os.getpid(), "request", r._hf_state_identity(state), "UHF")
    return replace(ref, seal=r._digest((ref.owner_pid, ref.request_identity,
                                      ref.state_identity, ref.method, ref.quantitative_status)))


def test_reuse_preserves_same_state_without_builder(monkeypatch):
    ref = accepted(monkeypatch)
    assert ref.validate("request", "UHF") is ref.hf_result
    assert ref.quantitative_status == "not-qualified"


@pytest.mark.parametrize("mutation", [
    lambda x: x.density_alpha.__setitem__((0, 0, 0), .8),
    lambda x: x.fock_beta.__setitem__((0, 0, 0), 2.),
    lambda x: x.occupations_beta.__setitem__((0, 0), 1.),
    lambda x: setattr(x, "executed_support", Support("historical_index_ball")),
    lambda x: setattr(x, "hcore", np.ones((1, 2, 2))),
    lambda x: x.chi_scf_acceptance.update(stationary=False),
    lambda x: x.process_provenance.update(runtime="different"),
])
def test_mutation_invalidates_state(monkeypatch, mutation):
    ref = accepted(monkeypatch)
    mutation(ref.hf_result)
    with pytest.raises(ValueError, match="state/source/factors"):
        ref.validate("request", "UHF")


def test_spin_geometry_and_metadata_mismatch_refuse(monkeypatch):
    ref = accepted(monkeypatch)
    with pytest.raises(ValueError, match="mismatch"):
        ref.validate("request", "RHF")
    with pytest.raises(ValueError, match="mismatch"):
        ref.validate("changed-geometry-or-basis", "UHF")
    with pytest.raises(ValueError, match="seal"):
        replace(ref, method="RHF").validate("request", "RHF")


@pytest.mark.parametrize("report", [None, dict(stationary=True),
    dict(stationary=False, final_rebuild=True)])
def test_flag_without_rebuilt_stationary_operator_is_not_accepted(monkeypatch, report):
    state = accepted(monkeypatch).hf_result
    state.chi_scf_acceptance = report
    with pytest.raises(ValueError, match="rebuilt stationary"):
        r._admit_hf_state(state)


def test_lattice_density_is_losslessly_bound(monkeypatch):
    state = accepted(monkeypatch).hf_result
    state.density_alpha = SimpleNamespace(
        cells=[SimpleNamespace(index=np.array([0, 0, 0], dtype=np.int32),
                               r_cart=np.zeros(3))],
        blocks=[np.eye(2)])
    identity = r._hf_state_identity(state)
    state.density_alpha.blocks[0][0, 1] = .01
    assert r._hf_state_identity(state) != identity


def test_reference_requires_explicit_source_before_driver(monkeypatch):
    from vibeqc.periodic.chi import scf
    def forbidden(*a, **k):
        pytest.fail("SCF must not run before support admission")
    monkeypatch.setattr(scf, "run_aiccm2026dev_b_uhf", forbidden)
    with pytest.raises(ValueError, match="explicit support"):
        r.prepare_chi_hf_reference(None, None, method="UHF")


def test_public_explicit_singlet_uhf_selection():
    from vibeqc.periodic_runner import _infer_aiccm_scf_reference
    system = SimpleNamespace(multiplicity=1, n_electrons=lambda: 2)
    assert _infer_aiccm_scf_reference(system, functional=None, scf_reference="uhf") == "UHF"
    assert _infer_aiccm_scf_reference(system, functional=None, scf_reference="rhf") == "RHF"
    with pytest.raises(ValueError):
        _infer_aiccm_scf_reference(system, functional="pbe", scf_reference="uhf")


@pytest.mark.parametrize("values", [[[.5, .5]], [[1., 1.]], [[np.nan, 0.]]])
def test_fractional_or_wrong_spin_populations_refuse(values):
    from vibeqc.periodic.chi.scf import _require_hf_integer_state
    system = SimpleNamespace(multiplicity=2, n_electrons=lambda: 1)
    result = SimpleNamespace(density_alpha=[], occupations_alpha=values,
                             occupations_beta=[[0., 0.]])
    with pytest.raises(NotImplementedError):
        _require_hf_integer_state(result, system)
