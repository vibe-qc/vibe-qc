"""Native one-electron chi ECP assembly checks, not scientific acceptance.

Only the unrelated two-electron factor/auxiliary construction is replaced.
S, T, ionic Coulomb, local ECP, projectors and ionic Ewald all execute natively.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from vibeqc.periodic.chi.ecp import ecp_contract
from vibeqc.periodic.chi.ecp_hamiltonian import attach_ecp_result
from vibeqc.periodic.chi.rohf import _build_source
from vibeqc.periodic_fitted_support import ChiCoulombControls, fitted_nuclear_operators
from tests.test_chi_ecp_operator import _basis, _block


@pytest.mark.parametrize('nk',[1,3])
def test_rohf_source_assembles_effective_ionic_and_scalar_ecp_once(monkeypatch,nk):
    from vibeqc import aux_basis, periodic_fitted_support
    positions = [[0.,0.,0.]]
    lattice = np.diag([18.,19.,20.])
    system = core.PeriodicSystem(3,lattice,[core.Atom(6,positions[0])],0,3)
    basis = _basis(positions,angular=(0,1),charges=[6])
    options = core.PeriodicRHFOptions()
    primitive = _block()
    options.ecp_primitive_blocks = [primitive]
    options.ecp_home_centers = positions
    options.ecp_effective_charges = [4.]
    options.ecp_total_ncore = 2
    controls = ChiCoulombControls(3.2,3.2,10.,2.3,2.0,.45)
    points = np.zeros((nk,3)); points[:,0] = 2*np.pi*np.arange(nk)/(nk*lattice[0,0])
    kpoints = SimpleNamespace(kpoints_cart=points,weights=np.full(nk,1/nk))
    auxiliary, operator = object(), object()
    calls = []
    monkeypatch.setattr(aux_basis,'make_aux_basis_set',lambda *args,**kwargs:auxiliary)
    monkeypatch.setattr(aux_basis,'make_modrho_aux_basis',lambda *args,**kwargs:auxiliary)
    def retained_source(actual_system,actual_basis,actual_auxiliary,k,w,**kwargs):
        calls.append((actual_system,actual_basis,actual_auxiliary,np.array(k),np.array(w),kwargs))
        return operator
    monkeypatch.setattr(periodic_fitted_support,'build_chi_fitted_operator',retained_source)
    h,s,enuc,op,encoded,aux,name,lat,ecp,components = _build_source(
        system,basis,kpoints,options,controls,'scalar-ecp-test-aux')
    assert op is operator and aux is auxiliary and name == 'scalar-ecp-test-aux'
    assert len(calls) == 1 and calls[0][0] is system and calls[0][1] is basis
    assert calls[0][2] is auxiliary and calls[0][-1]['source'] == 'classic'
    np.testing.assert_array_equal(calls[0][3],points)
    np.testing.assert_array_equal(calls[0][4],kpoints.weights)
    assert [atom.Z for atom in system.unit_cell] == [6]
    assert system.n_electrons() == 6 and options.ecp_total_ncore == 2
    # The wide cell and finite physical radii select home-only Gaussian pairs.
    # Molecular S/T and unsplit ECP are independent of the periodic assembly.
    molecular_s = np.asarray(core.compute_overlap(basis))
    molecular_t = np.asarray(core.compute_kinetic(basis))
    molecular_ecp = np.asarray(core.compute_ecp_matrix_from_primitives(
        basis,positions[0],[primitive]))
    for k in range(nk):
        np.testing.assert_allclose(s[k],molecular_s,atol=3e-13,rtol=0)
        np.testing.assert_allclose(components['kinetic'][k],molecular_t,atol=3e-13,rtol=0)
        np.testing.assert_allclose(components['ecp_local'][k]+components['ecp_nonlocal'][k],
                                   molecular_ecp,atol=3e-12,rtol=0)
        np.testing.assert_allclose(h[k],molecular_t+components['ionic_coulomb'][k]+molecular_ecp,
                                   atol=3e-12,rtol=0)
        np.testing.assert_allclose(h[k],h[k].conj().T,atol=3e-13,rtol=0)
    # Native bare-Z evaluation supplies a charge-scaling oracle: potential is
    # linear in Z while the ionic Ewald quadratic form scales as Z squared.
    overlap = core.compute_overlap_lattice(basis,system,lat)
    bare_v,_,bare_nn = fitted_nuclear_operators(system,basis,lat,overlap,controls)
    for k,point in enumerate(points):
        expected = (4/6)*np.asarray(core.bloch_sum(bare_v,point))
        np.testing.assert_allclose(components['ionic_coulomb'][k],expected,atol=3e-12,rtol=0)
    assert enuc == pytest.approx((4/6)**2*bare_nn.energy,abs=3e-12)
    record = json.loads(encoded)
    assert record['nuclei'][0]['Z'] == 4
    assert record['ecp'] == ecp
    assert ecp['atoms'][0]['Z'] == 6 and ecp['atoms'][0]['effective_charge'] == 4
    assert ecp['ecp_core_electrons'] == 2 and ecp['explicit_electrons'] == 4
    assert ecp['identity'] == ecp_contract(system,options)['identity']
    assert ecp['projectors'][0]['primitives'] == [[2,2,.45,.37],[2,0,.8,1.2],[2,1,1.3,-.4]]
    assert set(record['operator_cells']) == {'S','T','V_ne','ecp_local','ecp_nonlocal'}
    assert ecp['executed_support']['projector_cutoff_bohr'] == controls.nuclear_real_cutoff_bohr
    assert ecp['executed_support']['cells'] == record['operator_cells']['ecp_local']
    # An explicit occupied density exercises output attachment/counts without
    # claiming this arbitrary density is an SCF solution.
    coeff = np.linalg.inv(np.linalg.cholesky(molecular_s)).conj().T
    da = coeff[:,:3]@coeff[:,:3].conj().T
    db = coeff[:,:1]@coeff[:,:1].conj().T
    state = SimpleNamespace(kpoint_weights=kpoints.weights,e_nuclear=enuc,
        density_alpha=np.repeat(da[None],nk,axis=0),density_beta=np.repeat(db[None],nk,axis=0))
    attach_ecp_result(state,ecp,components)
    assert state.ecp_total_ncore == 2 and state.effective_n_electrons == 4
    assert tuple(state.ecp_effective_charges) == (4,)
    for key,matrix in components.items():
        energy = sum(float(w*np.trace((da+db)@matrix[k]).real) for k,w in enumerate(kpoints.weights))
        assert state.chi_ecp['energy_components'][key] == pytest.approx(energy,abs=3e-12)
    assert state.chi_ecp['energy_components']['ionic_ewald'] == enuc
    assert 'energy_components' not in record['ecp']


@pytest.mark.parametrize('method,z,multiplicity',[('RHF',4,1),('UHF',3,2),('ROHF',3,2)])
def test_native_classic_ecp_hf_driver_seals_and_restarts_without_source_rebuild(
        tmp_path,monkeypatch,method,z,multiplicity):
    """A small synthetic operator exercises real HF plumbing, not chemistry."""
    from vibeqc import aux_basis, periodic_fitted_support, periodic_k_gdf
    from vibeqc.periodic.chi.reference import prepare_chi_hf_reference
    from vibeqc.periodic.chi.ecp import validate_ecp_reference
    from vibeqc.periodic.chi.checkpoint import portable_reference_identity
    system = core.PeriodicSystem(3,np.diag([10.,12.,14.]),[core.Atom(z,[0.,0.,0.])],0,multiplicity)
    molecule = system.unit_cell_molecule()
    basis = core.BasisSet(molecule,[core.ShellInfo(0,0,True,[a],[1.],[0.,0.,0.])
        for a in (1.6,.45,.12)],'scalar-ecp-driver',False)
    auxiliary = core.BasisSet(molecule,[core.ShellInfo(0,0,True,[a],[1.],[0.,0.,0.])
        for a in (3.2,.9,.24,.06)],'scalar-ecp-driver-aux',False)
    monkeypatch.setattr(aux_basis,'make_aux_basis_set',lambda *args,**kwargs:auxiliary)
    monkeypatch.setattr(periodic_k_gdf,'make_aux_basis_set',lambda *args,**kwargs:auxiliary)
    options = core.PeriodicRHFOptions()
    options.initial_guess = core.InitialGuess.HCORE
    options.conv_tol_energy = 1e-11
    options.conv_tol_grad = 1e-9
    options.max_iter = 80
    options.ecp_primitive_blocks = [_block()]
    options.ecp_home_centers = [[0.,0.,0.]]
    options.ecp_effective_charges = [z-2.]
    options.ecp_total_ncore = 2
    controls = ChiCoulombControls(3.,3.,2.,3.,2.,.4)
    kwargs = dict(method=method,backend='ri',fit_source='classic',
                  coulomb_controls=controls,aux_basis='scalar-ecp-driver-aux')
    archive = tmp_path/'accepted.vqr'
    fresh = prepare_chi_hf_reference(system,basis,(1,1,1),options,checkpoint_path=archive,**kwargs)
    hf = fresh.hf_result
    assert hf.converged
    assert validate_ecp_reference(hf)['explicit_electrons'] == z-2
    assert hf.chi_scf_acceptance['final_rebuild'] and hf.chi_scf_acceptance['stationary']
    assert hf.ecp_total_ncore == 2
    assert np.linalg.norm(np.asarray(hf.hcore_components['ecp_local'])) > 1e-6
    assert np.linalg.norm(np.asarray(hf.hcore_components['ecp_nonlocal'])) > 1e-6
    assert hf.chi_ecp['energy_components']['ionic_ewald'] == hf.e_nuclear
    def forbidden(*args,**kwargs):
        raise AssertionError('accepted ECP restart must not integrate, refit, or iterate SCF')
    monkeypatch.setattr(periodic_fitted_support,'build_chi_fitted_operator',forbidden)
    monkeypatch.setattr(core,'compute_ecp_lattice_from_primitives',forbidden)
    restored = prepare_chi_hf_reference(system,basis,(1,1,1),options,restart_from=archive,**kwargs)
    assert portable_reference_identity(restored) == portable_reference_identity(fresh)
    assert restored.hf_result.energy == hf.energy
    assert restored.hf_result.effective_n_electrons == z-2
    assert restored.hf_result.restart['new_scf_iterations'] == 0
