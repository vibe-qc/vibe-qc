"""T3 accepted-reference contracts; all energies are within chi only."""
from dataclasses import replace
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq

from vibeqc.periodic.chi import posthf as p
from vibeqc.periodic.chi.reference import prepare_chi_reference, _admit_state, orbital_residuals

pytestmark = pytest.mark.experimental


def cell():
    system = vq.PeriodicSystem(3, np.eye(3)*6,
        [vq.Atom(1, [3,3,2.3]), vq.Atom(1, [3,3,3.7])])
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def options():
    opts = vq.PeriodicRHFOptions()
    opts.conv_tol_energy = 1e-11
    opts.conv_tol_grad = 1e-9
    opts.damping = 0
    opts.dynamic_damping = False
    return opts


@pytest.fixture(scope="module")
def accepted():
    system, basis = cell()
    opts = options()
    ref = prepare_chi_reference(system, basis, (2,1,1), opts,
                               fit_source="classic", progress=False, verbose=0)
    return system, basis, opts, ref


def zero_mp2():
    from vibeqc.dlpno.mp2 import DLPNOMP2Options
    return DLPNOMP2Options(localise="none", n_frozen=0, tcut_pno=0,
        tcut_pno_weak=0, tcut_mkn=0, tcut_pairs=0, tcut_pairs_weak=0,
        conv_tol_energy=1e-12, conv_tol_residual=1e-10)


def test_second_consumers_rebuild_nothing_and_preserve_raw_local(accepted, monkeypatch):
    system, basis, opts, ref = accepted
    def forbidden(*a, **k):
        pytest.fail("accepted consumer rebuilt SCF, fit or real reference")
    for name in ("run_aiccm2026dev_b_rhf", "_selected_lpq_cache", "_selected_lov_cache",
                 "_build_canonical_lpq_cache", "_build_canonical_lov_cache", "_build_b_real_reference"):
        monkeypatch.setattr(p, name, forbidden)
    kwargs = dict(reference=ref, fit_source="classic", progress=False, verbose=0)
    canonical = p.run_aiccm2026dev_b_mp2(system, basis, (2,1,1), opts, **kwargs)
    local = p.run_aiccm2026dev_b_dlpno_mp2(system, basis, (2,1,1), opts,
                                          dlpno_options=zero_mp2(), **kwargs)
    assert abs(local.energy-canonical.energy) < 1e-10
    assert abs(local.raw_local_e_corr_per_cell-canonical.e_corr_per_cell) < 1e-10
    assert local.e_corr_per_cell == pytest.approx(local.raw_local_e_corr_per_cell +
                                                 local.complete_space_correction_per_cell, abs=1e-15)
    assert local.reference_execution["scf"]["status"] == "reused"
    assert local.reference_execution["correlation_fit"]["status"] == "reused"
    assert local.reference_execution["home_auxiliary_reference"]["status"] == "reused"
    assert local.reference_execution["home_auxiliary_reference"]["source"] == ref.correlation_fit_source.identity
    assert canonical.reference_execution["orbital_projection"]["source"] == ref.correlation_fit_source.identity
    assert local.quantitative_status == ref.quantitative_status == "not-qualified"
    assert len(ref.factors) == 4
    assert ref.real_reference.df.home_factors.shape[0] == ref.auxiliary.nbasis


def test_omitted_options_adopt_supplied_custom_support(monkeypatch):
    system, basis = cell()
    opts = options()
    opts.lattice_opts.cutoff_bohr = 16.
    controls = dict(fit_source="classic", lattice_cutoff_bohr=16., progress=False, verbose=0)
    ref = prepare_chi_reference(system, basis, (2,1,1), opts, **controls)
    monkeypatch.setattr(p, "run_aiccm2026dev_b_rhf",
                        lambda *a, **k: pytest.fail("supplied reference rebuilt SCF"))
    explicit = p.run_aiccm2026dev_b_mp2(system, basis, (2,1,1), opts, reference=ref, **controls)
    inherited = p.run_aiccm2026dev_b_mp2(system, basis, (2,1,1), reference=ref, **controls)
    assert inherited.energy == explicit.energy
    assert inherited.reference_execution["scf"]["status"] == "reused"


@pytest.mark.parametrize("key,value", [("fit_source","auto"), ("rsgdf_tail_ke_cutoff",300.),
    ("aux_basis","def2-universal-jfit"), ("rsgdf_ke_cutoff",201.),
    ("gdf_linear_dep_threshold",2e-9)])
def test_requested_source_auxiliary_tail_support_mismatch(accepted, key, value):
    system, basis, opts, ref = accepted
    kwargs = dict(fit_source="classic", reference=ref, progress=False)
    kwargs[key] = value
    with pytest.raises(ValueError, match="mismatch"):
        p.run_aiccm2026dev_b_mp2(system, basis, (2,1,1), opts, **kwargs)


@pytest.mark.parametrize("field", ["source", "auxiliary_frame", "executed_tail_reciprocal_cutoff_ha"])
def test_descriptor_mutation_rejected(accepted, field):
    system, basis, opts, ref = accepted
    changed = replace(ref.correlation_fit_source, **{field: "changed"})
    with pytest.raises(ValueError, match="changed"):
        p.run_aiccm2026dev_b_mp2(system, basis, (2,1,1), opts,
            reference=replace(ref, correlation_fit_source=changed), fit_source="classic")


def test_geometry_mesh_and_scf_controls_rejected(accepted):
    system, basis, opts, ref = accepted
    other = vq.PeriodicSystem(3,np.eye(3)*6.01,
        [vq.Atom(1,[3,3,2.3]),vq.Atom(1,[3,3,3.7])])
    for sys, mesh, opt in [(other,(2,1,1),opts), (system,(1,2,1),opts)]:
        with pytest.raises(ValueError, match="mismatch"):
            p.run_aiccm2026dev_b_mp2(sys,basis,mesh,opt,reference=ref,fit_source="classic")
    changed = options()
    changed.lattice_opts.nuclear_cutoff_bohr += 1
    with pytest.raises(ValueError, match="controls/support"):
        p.run_aiccm2026dev_b_mp2(system,basis,(2,1,1),changed,reference=ref,fit_source="classic")


def test_raw_and_mutated_references_rejected(accepted, monkeypatch):
    system,basis,opts,ref = accepted
    with pytest.raises(TypeError,match="ChiAcceptedReference"):
        p.run_aiccm2026dev_b_mp2(system,basis,(2,1,1),opts,reference=ref.hf_result,fit_source="classic")
    monkeypatch.setattr(ref.hf_result,"energy",ref.hf_result.energy+0.01)
    with pytest.raises(ValueError,match="state/source changed"):
        p.run_aiccm2026dev_b_mp2(system,basis,(2,1,1),opts,reference=ref,fit_source="classic")


def test_process_provenance_mismatch(accepted, monkeypatch):
    from vibeqc.periodic.chi import reference as r
    system,basis,opts,ref=accepted
    monkeypatch.setattr(r,"loaded_process_provenance",lambda: {"release":"different"})
    with pytest.raises(ValueError,match="process release/source/native"):
        p.run_aiccm2026dev_b_mp2(system,basis,(2,1,1),opts,reference=ref,fit_source="classic")


def test_external_factor_cache_not_cleared(accepted):
    system,basis,opts,ref=accepted
    cache=dict(ref.factors)
    before=dict(cache)
    p._build_b_real_reference(system,basis,(2,1,1),ref.hf_result,cache,
                             dict(ref.controls)["auxiliary_name"])
    assert cache.keys() == before.keys()
    assert all(cache[key] is before[key] for key in cache)


def test_rejects_insufficient_convergence_before_build(monkeypatch):
    system,basis=cell()
    monkeypatch.setattr(p,"run_aiccm2026dev_b_rhf",lambda *a,**k: pytest.fail("SCF entered"))
    with pytest.raises(ValueError,match="conv_tol_energy"):
        prepare_chi_reference(system,basis,(2,1,1),vq.PeriodicRHFOptions())


def test_achieved_residual_and_corrected_operator_admission(accepted):
    _,_,_,ref=accepted
    hf=copy(ref.hf_result)
    hf.fock=np.array(hf.fock).copy()
    hf.fock[:,0,0]+=1e-4
    with pytest.raises(ValueError,match="stationarity"):
        _admit_state(hf,1)
    hf=copy(ref.hf_result)
    hf.aiccm2026dev_b=replace(hf.aiccm2026dev_b,final_scf_delta_e_ha=1e-6)
    with pytest.raises(ValueError,match="energy convergence"):
        _admit_state(hf,1)
    hf=copy(ref.hf_result)
    hf.returned_operator="post-convergence-one-center-corrected"
    with pytest.raises(ValueError,match="matched iterated"):
        _admit_state(hf,1)


def test_iterated_residual_cannot_attest_corrected_stationarity():
    hf=SimpleNamespace(fock=np.array([[[0.,1e-6],[1e-6,1.]]]),
        iterated_fock=np.array([np.diag([0.,1.])]), overlap=np.array([np.eye(2)]),
        density=np.array([np.diag([2.,0.])]), mo_coeffs=np.array([np.eye(2)]),
        mo_energies=np.array([[0.,1.]]), kpoint_weights=np.array([1.]))
    residuals=orbital_residuals(hf,1)
    assert residuals["iterated_density_commutator"] == 0
    assert residuals["accepted_density_returned_fock_commutator"] > 1e-9


@pytest.mark.parametrize('name', ['run_aiccm2026dev_b_ccsd','run_aiccm2026dev_b_ccsd_t',
    'run_aiccm2026dev_b_dlpno_ccsd','run_aiccm2026dev_b_dlpno_ccsd_t'])
def test_cc_wrappers_transport_reference_before_solver(accepted, monkeypatch, name):
    from vibeqc.periodic.chi import reference as r
    system,basis,opts,ref=accepted
    class Admitted(Exception):
        pass
    original=r.require_reference
    def observe(*a,**k):
        assert original(*a,**k) is ref
        raise Admitted
    monkeypatch.setattr(r,'require_reference',observe)
    with pytest.raises(Admitted):
        getattr(p,name)(system,basis,(2,1,1),opts,reference=ref,fit_source='classic')


def test_factor_and_home_state_mutation_rejected(accepted, monkeypatch):
    from types import MappingProxyType
    system,basis,opts,ref=accepted
    factors=dict(ref.factors)
    factors[(0,0)]=factors[(0,0)].copy()
    factors[(0,0)][0,0,0]+=1e-5
    with pytest.raises(ValueError,match='correlation factors changed'):
        p.run_aiccm2026dev_b_mp2(system,basis,(2,1,1),opts,
            reference=replace(ref,factors=MappingProxyType(factors)),fit_source='classic')
    monkeypatch.setattr(ref.real_reference.df,'mesh',(1,2,1))
    with pytest.raises(ValueError,match='home-auxiliary state changed'):
        p.run_aiccm2026dev_b_mp2(system,basis,(2,1,1),opts,reference=ref,fit_source='classic')


@pytest.mark.slow
def test_auto_shared_q_reference_matches_cold_and_reuses_cache(monkeypatch):
    """Historical auto keeps its SR reference distinct from classic correlation."""
    system, basis = cell()
    opts = options()
    kwargs = dict(fit_source="auto", progress=False, verbose=0)
    cold = p.run_aiccm2026dev_b_mp2(system, basis, (2, 1, 1), opts, **kwargs)
    ref = prepare_chi_reference(system, basis, (2, 1, 1), opts, **kwargs)
    assert ref.scf_fit_source.source == "sr"
    assert ref.correlation_fit_source.source == "classic"
    assert ref.correlation_fit_source.engine == "native-classic-fft-shared-q"
    assert ref.correlation_fit_source.auxiliary_frame == "canonical-modrho"
    factors = dict(ref.factors)

    def forbidden(*args, **kwargs):
        pytest.fail("accepted auto consumer rebuilt SCF, factors or real reference")

    for name in ("run_aiccm2026dev_b_rhf", "_selected_lpq_cache", "_selected_lov_cache",
                 "_build_canonical_lpq_cache", "_build_canonical_lov_cache",
                 "_build_b_real_reference"):
        monkeypatch.setattr(p, name, forbidden)
    reused = p.run_aiccm2026dev_b_mp2(
        system, basis, (2, 1, 1), opts, reference=ref, **kwargs)
    local = p.run_aiccm2026dev_b_dlpno_mp2(
        system, basis, (2, 1, 1), opts, reference=ref,
        dlpno_options=zero_mp2(), **kwargs)
    assert abs(reused.energy - cold.energy) < 1e-10
    assert abs(reused.e_corr_per_cell - cold.e_corr_per_cell) < 1e-10
    assert abs(local.raw_local_e_corr_per_cell - cold.e_corr_per_cell) < 1e-10
    assert abs(local.e_corr_per_cell - cold.e_corr_per_cell) < 1e-10
    assert len(ref.factors) == len(factors)
    assert all(ref.factors[key] is value for key, value in factors.items())
    assert local.reference_execution["correlation_fit"]["status"] == "reused"
    assert local.quantitative_status == reused.quantitative_status == "not-qualified"
