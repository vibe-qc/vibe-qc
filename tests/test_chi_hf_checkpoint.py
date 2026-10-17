"""Independent finite-source restart checks; no native scientific acceptance."""
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from vibeqc.output import checkpoint as storage
from vibeqc.periodic.chi import checkpoint as cp
from vibeqc.periodic.chi import reference as refs
from vibeqc.periodic_fit_source import RSGDFFitSource, factor_digest
from vibeqc.periodic_fitted_support import ChiFittedOperator


def fixture(monkeypatch, method='RHF', nk=3):
    """Known stationary diagonal model; explicit four-index oracle builds F/E."""
    n = 3; na, nb = (1,1) if method == 'RHF' else (2,1)
    system = SimpleNamespace(dim=3, lattice=np.eye(3)*2*np.pi, charge=0,
        multiplicity=na-nb+1, unit_cell=[SimpleNamespace(Z=na+nb, xyz=(0.,0.,0.))],
        n_electrons=lambda: na+nb)
    shell = SimpleNamespace(atom_index=0, l=1, pure=True, exponents=[1.],
        coefficients=[1.], origin=[0.,0.,0.])
    basis = SimpleNamespace(name='synthetic', nbasis=n, shells=lambda:[shell])
    basis_hash = cp._basis_record(basis)['sha256']
    from vibeqc.periodic_fitted_support import ChiCoulombControls
    controls=ChiCoulombControls(6.2,6.2,10.,8.,5.,.6)
    support=asdict(controls)
    k = np.zeros((nk,3)); k[:,0] = np.arange(nk)/nk
    factors = {}
    for i in range(nk):
        for j in range(nk):
            phase = np.exp(1j*(j-i)*.3)
            factors[i,j] = np.array([np.diag([.1,.2,.3]), np.diag([.2,.3,.1])],complex)*phase
    groups = tuple(((q/nk,0.,0.),tuple((i,j) for i,j in factors if (j-i)%nk==q)) for q in range(nk))
    records = [dict(q=q,lattice=system.lattice.tolist(),metric_eigenvalues=[1.,2.],metric_keep=[True,True],rank=2) for q,_ in groups]
    descriptor = RSGDFFitSource('synthetic-finite',1,'classic',cp._physical_source_signature(dict(geometry=cp._geometry_record(system),basis=cp._basis_record(basis)),'aux-hash'),'synthetic',basis_hash,
        'synthetic-aux','aux-hash',tuple(sorted(support.items())),1e-9,'canonical',groups,tuple(map(tuple,k)),10.,None,
        tuple((p,a.shape) for p,a in sorted(factors.items())),factor_digest(factors),
        executed_domains_json=json.dumps(dict(q_records=records,controls=support,schema='vibeqc.chi.fitted-domains/v1')))
    op = ChiFittedOperator(factors,descriptor,np.ones(nk)/nk)
    coeff = np.tile(np.eye(n,dtype=complex),(nk,1,1))
    coeff *= np.exp(1j*np.arange(nk)[:,None,None]*np.arange(n)[None,None,:]*.19)
    da = coeff[:,:,:na]@coeff[:,:,:na].conj().transpose(0,2,1)
    db = coeff[:,:,:nb]@coeff[:,:,:nb].conj().transpose(0,2,1)
    ds = (da,db)
    weights = np.ones(nk)/nk
    h = np.tile(np.diag([-2.,-.7,1.1]),(nk,1,1)).astype(complex)
    j = np.zeros_like(h); ks = [np.zeros_like(h),np.zeros_like(h)]
    # AO four-index sums independent of production panel contractions.
    for i,q,p,r,s,t in np.ndindex(nk,nk,n,n,n,n):
        j[i,p,r] += weights[q]*sum(op[i,i][a,p,r]*op[q,q][a,s,t].conjugate() for a in range(2))*(da[q,s,t]+db[q,s,t])
        for spin in range(2):
            ks[spin][i,p,r] += weights[q]*sum(op[i,q][a,p,s]*op[i,q][a,r,t].conjugate() for a in range(2))*ds[spin][q,s,t]
    madelung = .125
    fa,fb = (h+j-kk-madelung*dd for kk,dd in zip(ks,ds))
    energy = .4 + sum(float(np.einsum('k,kpq,kqp->',weights,d,h+f).real)/2 for d,f in zip(ds,(fa,fb)))
    def occ(count,scale=1): return np.tile((np.arange(n)<count).astype(float)*scale,(nk,1))
    hf = SimpleNamespace(energy=energy,e_nuclear=.4,converged=True,overlap=np.tile(np.eye(n),(nk,1,1)),
        hcore=h,kpoints_cart=k,kpoint_weights=weights,smearing_temperature=0.,
        aiccm2026dev_b=cp._AttributeRecord(dict(electronic_method=method,backend='ri',mesh=(nk,1,1))),
        exact_exchange_assembly=cp._AttributeRecord(dict(c_full=1.,c_sr=0.)),
        fitted_operator=op,executed_fit_source=replace(descriptor,executed_domains_json=json.dumps(dict(fitted_operator_identity=op.identity))),
        chi_scf_acceptance=dict(final_rebuild=True,stationary=True,energy_residual=0.,energy_consistency_error=0.,orbital_residual=0.,density_refill_residual=0.),
        process_provenance={'runtime':'fixture'},_chi_hf_request_identity='synthetic-request')
    if method == 'RHF':
        hf.mo_coeffs=coeff;hf.mo_energies=np.diagonal(fa,axis1=1,axis2=2).real.copy()
        hf.fock=fa;hf.density=da+db;hf.occupations=occ(na,2)
    else:
        for suffix,d,f,count in (('alpha',da,fa,na),('beta',db,fb,nb)):
            setattr(hf,'mo_coeffs_'+suffix,coeff.copy());setattr(hf,'mo_energies_'+suffix,np.diagonal(f,axis1=1,axis2=2).real.copy())
            setattr(hf,'fock_'+suffix,f);setattr(hf,'density_'+suffix,d);setattr(hf,'occupations_'+suffix,occ(count))
        if method == 'ROHF':
            from vibeqc.periodic.chi.rohf import ROHF_CONVENTION, ROHF_CANONICALIZATION
            hf.mo_coeffs=coeff;hf.mo_energies=.5*(hf.mo_energies_alpha+hf.mo_energies_beta)
            hf.fock=.5*(fa+fb);hf.density=da+db;hf.occupations=occ(na)+occ(nb)
            hf.n_alpha=na;hf.n_beta=nb;hf.n_closed=nb;hf.n_open=na-nb
            hf.rohf_subspaces=dict(closed=tuple(range(nb)),open=tuple(range(nb,na)),virtual=tuple(range(na,n)))
            hf.rohf_convention=ROHF_CONVENTION;hf.rohf_canonicalization=ROHF_CANONICALIZATION
    cp.bind_chi_restart_inputs(hf,system,basis,(nk,1,1),None,method,dict(fit_source='classic',backend='ri',gdf_method='rsgdf',coulomb_controls=controls,
            rsgdf_ke_cutoff=10.,rsgdf_tail_ke_cutoff=None,gdf_linear_dep_threshold=1e-9,support_policy=None))
    hf._chi_hf_request_identity=cp._physical_request_identity(hf._chi_restart_physical)
    monkeypatch.setattr(refs,'loaded_process_provenance',lambda:{'runtime':'fixture'})
    return refs._seal_chi_hf_result(hf,method),system,basis


@pytest.mark.parametrize('method',['RHF','UHF','ROHF'])
@pytest.mark.parametrize('nk',[1,3])
def test_physical_roundtrip_and_source_preservation(tmp_path,monkeypatch,method,nk):
    ref,system,basis=fixture(monkeypatch,method,nk)
    path=cp.save_chi_hf_reference(tmp_path/'state.vqr',ref,system=system,basis=basis)
    restored=cp.load_chi_hf_reference(path,system=system,basis=basis,expected_method=method,
        expected_request_identity=ref.request_identity)
    assert restored.hf_result.energy==pytest.approx(ref.hf_result.energy,abs=1e-12)
    assert cp.portable_reference_identity(restored)==cp.portable_reference_identity(ref)
    assert restored.hf_result.fitted_operator.identity==ref.hf_result.fitted_operator.identity
    assert restored.hf_result.restart['status']=='restored-validated-reference'
    assert restored.hf_result.chi_scf_acceptance['stationary']
    assert restored.hf_result.process_provenance=={'runtime':'fixture'}
    assert restored.owner_pid==os.getpid()
    guess=cp.load_chi_hf_reference(path,mode='guess')
    assert isinstance(guess,cp.ChiHFGuess) and guess.convergence_required
    with pytest.raises(TypeError): cp.portable_reference_identity(guess)
    second=cp.save_chi_hf_reference(tmp_path/'second.vqr',restored)
    assert cp.load_chi_hf_reference(second).hf_result.energy==ref.hf_result.energy


@pytest.mark.parametrize('method',['RHF','UHF','ROHF'])
@pytest.mark.parametrize('external_metadata',[False,True])
def test_fresh_process_restore_blocks_native_imports(tmp_path,monkeypatch,method,external_metadata):
    if external_metadata:
        monkeypatch.setattr(storage,'_RESTART_METADATA_INLINE_LIMIT',1)
    ref,_,_=fixture(monkeypatch,method)
    path=cp.save_chi_hf_reference(tmp_path/'state.vqr',ref)
    with zipfile.ZipFile(path) as archive:
        assert ('metadata.json' in archive.namelist()) is external_metadata
    script=r'''
import importlib.abc,json,os,pathlib,sys,types
root=pathlib.Path(sys.argv[1])
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if '_vibeqc_core' in fullname: raise ImportError('native imports forbidden for restart proof')
sys.meta_path.insert(0,NoNative())
for name,path in [('vibeqc','python/vibeqc'),('vibeqc.periodic','python/vibeqc/periodic'),('vibeqc.periodic.chi','python/vibeqc/periodic/chi'),('vibeqc.output','python/vibeqc/output')]:
 module=types.ModuleType(name);module.__path__=[str(root/path)];sys.modules[name]=module
from vibeqc.periodic.chi import checkpoint as cp, reference as refs
refs.loaded_process_provenance=lambda:{'runtime':'fresh-process'}
ref=cp.load_chi_hf_reference(sys.argv[2])
ref.validate(ref.request_identity,ref.method)
print(json.dumps(dict(pid=ref.owner_pid,energy=ref.hf_result.energy,identity=cp.portable_reference_identity(ref))))
'''
    result=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parents[1]),str(path)],check=True,capture_output=True,text=True)
    record=json.loads(result.stdout)
    assert record['pid']!=os.getpid()
    assert record['energy']==ref.hf_result.energy
    assert record['identity']==cp.portable_reference_identity(ref)


def test_identity_ignores_runtime_and_timing_but_binds_physics(monkeypatch):
    ref,_,_=fixture(monkeypatch)
    identity=cp.portable_reference_identity(ref)
    ref.hf_result.process_provenance={'other':'runtime'}
    ref.hf_result.runtime_backend='different-runtime-build'
    ref.hf_result.aiccm2026dev_b.values['wall_time_seconds']=100.
    assert cp.portable_reference_identity(ref)==identity
    ref.hf_result.hcore[0,0,0]+=.1
    assert cp.portable_reference_identity(ref)!=identity


@pytest.mark.parametrize('field,value',[('expected_request_identity','other'),('expected_method','UHF'),('mode','auto')])
def test_incompatible_requests_refuse(tmp_path,monkeypatch,field,value):
    ref,_,_=fixture(monkeypatch)
    path=cp.save_chi_hf_reference(tmp_path/'state.vqr',ref)
    with pytest.raises(ValueError): cp.load_chi_hf_reference(path,**{field:value})


def test_geometry_basis_and_spin_refuse(tmp_path,monkeypatch):
    ref,system,basis=fixture(monkeypatch)
    path=cp.save_chi_hf_reference(tmp_path/'state.vqr',ref)
    system.multiplicity=3
    with pytest.raises(ValueError,match='geometry/spin'): cp.load_chi_hf_reference(path,system=system)
    basis.nbasis=4
    with pytest.raises(ValueError,match='basis'): cp.load_chi_hf_reference(path,basis=basis)


@pytest.mark.parametrize('mutation',['schema','rank','source','spin','fock','occupation','energy','nan'])
def test_mutation_even_rechecksummed_refuses(tmp_path,monkeypatch,mutation):
    ref,_,_=fixture(monkeypatch,'ROHF')
    path=cp.save_chi_hf_reference(tmp_path/'state.vqr',ref)
    metadata,arrays=storage.read_restart_state(path)
    if mutation=='schema': metadata['schema']='future/v9'
    elif mutation=='rank': metadata['operator']['factor_shapes'][0][1][0]+=1
    elif mutation=='source': metadata['operator']['source']='sr'
    elif mutation=='spin': metadata['method']='UHF'
    else:
        key=dict(fock='fock_alpha',occupation='occupations_alpha',energy='energy',nan='hcore')[mutation]
        record=metadata['state']['mapping'][key]
        if mutation=='energy': metadata['state']['mapping'][key]+=.1
        else:
            array_name=record['array'];arrays[array_name]=arrays[array_name].copy()
            arrays[array_name].flat[0]+=np.nan if mutation=='nan' else .1
    if mutation=='nan':
        with pytest.raises(ValueError,match='finite'): storage.write_restart_state(path,metadata,arrays)
    else:
        storage.write_restart_state(path,metadata,arrays)
        with pytest.raises((ValueError,NotImplementedError)): cp.load_chi_hf_reference(path)


def test_restored_mutation_invalidates_new_process_seal(tmp_path,monkeypatch):
    ref,_,_=fixture(monkeypatch)
    path=cp.save_chi_hf_reference(tmp_path/'state.vqr',ref)
    restored=cp.load_chi_hf_reference(path)
    with pytest.raises(ValueError): restored.hf_result.hcore[0][0,0]=0.
    restored.hf_result.hcore=np.asarray(restored.hf_result.hcore).copy()
    restored.hf_result.hcore[0,0,0]+=.1
    with pytest.raises(ValueError,match='state/source/factors'): restored.validate(restored.request_identity,'RHF')


def test_atomic_failure_preserves_last_complete_checkpoint(tmp_path,monkeypatch):
    path=storage.write_restart_state(tmp_path/'state.vqr',dict(schema='fixture/v1'),{'x':np.arange(100.)})
    old=path.read_bytes()
    def fail(*args): raise OSError('forced publication interruption')
    monkeypatch.setattr(storage.os,'replace',fail)
    with pytest.raises(OSError,match='forced'): storage.write_restart_state(path,dict(schema='fixture/v2'),{'x':np.arange(200.)})
    assert path.read_bytes()==old
    assert not list(tmp_path.glob('*.tmp'))
    assert storage.read_restart_state(path)[0]['schema']=='fixture/v1'


def test_numeric_storage_noncontiguous_empty_complex_and_no_pickle(tmp_path):
    arrays=dict(x=np.arange(800000.).reshape(1000,800)[:,::2],c=np.arange(10.)+2j,e=np.empty((0,3)),b=np.array([True,False]))
    path=storage.write_restart_state(tmp_path/'numeric.vqr',{'schema':'fixture/v1'},arrays)
    got=storage.read_restart_state(path)[1]
    for key in arrays: np.testing.assert_array_equal(got[key],arrays[key])
    with pytest.raises(ValueError,match='numeric'): storage.write_restart_state(path,{},dict(x=np.array([object()])))
    with pytest.raises((ValueError,TypeError)): storage.write_restart_state(path,{'x':float('inf')},{})


def test_large_executed_source_metadata_keeps_bounded_manifest(tmp_path):
    """Preserve repeated source text beyond the old complete-manifest limit."""
    source = json.dumps({'executed_support': 'q'*(17*1024*1024)},separators=(',',':'))
    metadata = dict(schema='source-fixture/v1',
        operator=dict(executed_domains_json=source),
        source=dict(executed_domains_json=source),
        physical_identity=hashlib.sha256(source.encode()).hexdigest())
    assert len(storage._restart_json(metadata))>32*1024*1024
    arrays = dict(factors=np.array([1.+2j,3.-4j]))
    path=storage.write_restart_state(tmp_path/'large-source.vqr',metadata,arrays)
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['schema']=='vibeqc.restart.archive/v2'
        assert archive.getinfo('manifest.json').file_size<4096
        info=archive.getinfo('metadata.json')
        assert info.compress_type==zipfile.ZIP_STORED
        assert info.file_size>32*1024*1024
        assert set(manifest['metadata'])=={'nbytes','sha256'}
        assert manifest['metadata']['nbytes']==info.file_size
    restored,values=storage.read_restart_state(path)
    assert restored==metadata
    assert restored['source']['executed_domains_json']==source
    np.testing.assert_array_equal(values['factors'],arrays['factors'])
    assert not values['factors'].flags.writeable


@pytest.mark.parametrize('damage',['bytes','truncate','size','descriptor','missing','extra',
                                  'compressed','duplicate-json','nonfinite-json','overflow-json',
                                  'nonobject-json'])
def test_external_restart_metadata_refuses_damage(tmp_path,monkeypatch,damage):
    monkeypatch.setattr(storage,'_RESTART_METADATA_INLINE_LIMIT',1)
    path=storage.write_restart_state(tmp_path/'external.vqr',{'schema':'fixture/v1'},
                                     dict(x=np.arange(4.)))
    with zipfile.ZipFile(path) as archive:
        entries={name:archive.read(name) for name in archive.namelist()}
    manifest=json.loads(entries['manifest.json'])
    assert manifest['schema']=='vibeqc.restart.archive/v2'
    if damage=='bytes':
        entries['metadata.json']=entries['metadata.json'].replace(b'fixture',b'fixturE')
    elif damage=='truncate':
        entries['metadata.json']=entries['metadata.json'][:-1]
    elif damage=='size':
        manifest['metadata']['nbytes']=True
    elif damage=='descriptor':
        manifest['metadata']['other']=0
    elif damage=='missing':
        del entries['metadata.json']
    elif damage=='extra':
        entries['extra.json']=b'{}'
    elif damage in ('duplicate-json','nonfinite-json','overflow-json','nonobject-json'):
        entries['metadata.json']={
            'duplicate-json':b'{"schema":"fixture/v1","schema":"fixture/v2"}',
            'nonfinite-json':b'{"schema":"fixture/v1","energy":NaN}',
            'overflow-json':b'{"schema":"fixture/v1","energy":1e999}',
            'nonobject-json':b'["fixture/v1"]',
        }[damage]
        manifest['metadata']=dict(nbytes=len(entries['metadata.json']),
            sha256=hashlib.sha256(entries['metadata.json']).hexdigest())
    manifest.pop('sha256')
    manifest['sha256']=hashlib.sha256(storage._restart_json(manifest)).hexdigest()
    entries['manifest.json']=storage._restart_json(manifest)
    with zipfile.ZipFile(path,'w') as archive:
        for name,data in entries.items():
            compression=zipfile.ZIP_DEFLATED if damage=='compressed' and name=='metadata.json' else zipfile.ZIP_STORED
            archive.writestr(name,data,compress_type=compression)
    with pytest.raises(ValueError):
        storage.read_restart_state(path)


def test_metadata_write_interruption_preserves_complete_checkpoint(tmp_path,monkeypatch):
    path=storage.write_restart_state(tmp_path/'state.vqr',{'schema':'fixture/v1'},
                                     dict(x=np.arange(4.)))
    previous=path.read_bytes()
    monkeypatch.setattr(storage,'_RESTART_METADATA_INLINE_LIMIT',1)
    original_write=zipfile._ZipWriteFile.write

    def interrupt(stream,data):
        if stream._zinfo.filename=='metadata.json':
            original_write(stream,data[:8])
            raise OSError('forced metadata publication interruption')
        return original_write(stream,data)

    monkeypatch.setattr(zipfile._ZipWriteFile,'write',interrupt)
    with pytest.raises(OSError,match='forced metadata publication interruption'):
        storage.write_restart_state(path,{'schema':'fixture/v2','source':'retained'},
                                    dict(x=np.arange(8.)))
    assert path.read_bytes()==previous
    assert not list(tmp_path.glob('*.tmp'))
    assert storage.read_restart_state(path)[0]=={'schema':'fixture/v1'}


@pytest.mark.parametrize('damage',['truncate','bytes','schema','shape','dtype','duplicate'])
def test_bad_archive_refuses(tmp_path,damage):
    path=storage.write_restart_state(tmp_path/'state.vqr',{},dict(x=np.arange(4.)))
    if damage=='truncate': path.write_bytes(path.read_bytes()[:80])
    elif damage=='bytes':
        data=bytearray(path.read_bytes());where=data.find(np.arange(4.).tobytes());data[where+8]^=1;path.write_bytes(data)
    else:
        with zipfile.ZipFile(path) as archive: entries={name:archive.read(name) for name in archive.namelist()}
        manifest=json.loads(entries['manifest.json'])
        if damage=='schema': manifest['schema']='future/v5'
        elif damage=='shape': manifest['arrays']['x']['shape']=[5]
        elif damage=='dtype': manifest['arrays']['x']['dtype']='|O'
        if damage!='duplicate':
            manifest.pop('sha256');manifest['sha256']=hashlib.sha256(storage._restart_json(manifest)).hexdigest()
            entries['manifest.json']=storage._restart_json(manifest)
        with zipfile.ZipFile(path,'w') as archive:
            for name,data in entries.items(): archive.writestr(name,data)
            if damage=='duplicate':
                with pytest.warns(UserWarning): archive.writestr('arrays/x',entries['arrays/x'])
    with pytest.raises(ValueError): storage.read_restart_state(path)


def test_source_rebuild_rejects_false_stationary_seal(tmp_path,monkeypatch):
    ref,_,_=fixture(monkeypatch)
    # Process-local stationary flags alone were never a physical load verdict.
    ref.hf_result.hcore[0,0,0]+=.02
    forged=refs._seal_chi_hf_result(ref.hf_result,'RHF')
    with pytest.raises(ValueError,match='physical Fock/energy'):
        cp.save_chi_hf_reference(tmp_path/'bad.vqr',forged)


def test_serialized_acceptance_flag_does_not_decide_restored_acceptance(tmp_path,monkeypatch):
    ref,_,_=fixture(monkeypatch,'ROHF')
    path=cp.save_chi_hf_reference(tmp_path/'state.vqr',ref)
    metadata,arrays=storage.read_restart_state(path)
    metadata['state']['mapping']['chi_scf_acceptance']['mapping']['stationary']=False
    storage.write_restart_state(path,metadata,arrays)
    restored=cp.load_chi_hf_reference(path)
    assert restored.hf_result.chi_scf_acceptance['stationary']
    assert 'rebuild' in restored.hf_result.chi_scf_acceptance['restart_validation']


@pytest.mark.parametrize('change',['basis','lattice','controls'])
def test_portable_input_metadata_is_bound_and_sealed(monkeypatch,change):
    ref,_,_=fixture(monkeypatch)
    before=cp.portable_reference_identity(ref)
    physical=ref.hf_result._chi_restart_physical
    if change=='basis':physical['basis']['shells'][0]['exponents'][0]+=1.
    elif change=='lattice':physical['geometry']['lattice_bohr'][0][0]+=.1
    else:physical['controls']['gdf_linear_dep_threshold']=1e-3
    assert cp.portable_reference_identity(ref)!=before
    with pytest.raises(ValueError,match='state/source/factors'): ref.validate(ref.request_identity,'RHF')


def test_panel_staging_bound_for_strided_numeric_inputs():
    value=np.arange(2000000.,dtype=float).reshape(1000,2000)[:,::3]
    chunks=list(storage._restart_chunks(value))
    assert max(map(len,chunks))<=storage._RESTART_CHUNK_BYTES
    assert sum(map(len,chunks))==value.nbytes


@pytest.mark.parametrize('method,change',[('RHF','energy'),('RHF','fock'),('RHF','projector'),('UHF','projector'),('RHF','occupation'),('RHF','rank'),('ROHF','shared-space'),('ROHF','projector')])
def test_physical_admission_survives_no_trusted_seal_or_checksum(monkeypatch,method,change):
    ref,_,_=fixture(monkeypatch,method,nk=1)
    hf=ref.hf_result
    if change=='energy':hf.energy+=.1
    elif change=='fock':hf.fock[0,0,2]=hf.fock[0,2,0]=.1
    elif change=='occupation':hf.occupations[0,0]=1.5
    elif change=='rank':
        hf.mo_coeffs=hf.mo_coeffs[:,:,:-1];hf.mo_energies=hf.mo_energies[:,:-1];hf.occupations=hf.occupations[:,:-1]
    elif change=='shared-space':hf.mo_coeffs_alpha[0,0,0]+=.1
    else:
        angle=.2;u=np.eye(3);u[0,0]=u[2,2]=np.cos(angle);u[0,2]=-np.sin(angle);u[2,0]=np.sin(angle)
        c=hf.mo_coeffs if method!='UHF' else hf.mo_coeffs_alpha
        c=c@u
        if method=='RHF':
            hf.mo_coeffs=c;hf.density=2*c[:,:,:1]@c[:,:,:1].conj().transpose(0,2,1)
            ds=(hf.density/2,hf.density/2)
        else:
            na,nb=2,1
            hf.mo_coeffs_alpha=c.copy();hf.mo_coeffs_beta=c.copy()
            hf.density_alpha=c[:,:,:na]@c[:,:,:na].conj().transpose(0,2,1)
            hf.density_beta=c[:,:,:nb]@c[:,:,:nb].conj().transpose(0,2,1)
            ds=(hf.density_alpha,hf.density_beta)
            if method=='ROHF':hf.mo_coeffs=c.copy()
        j,ks=cp._contract_jk(hf.fitted_operator,ds,hf.kpoint_weights)
        fa,fb=(hf.hcore+j-k-.125*d for k,d in zip(ks,ds))
        if method=='RHF':hf.fock=fa
        else:hf.fock_alpha=fa;hf.fock_beta=fb
        hf.energy=.4+sum(float(np.einsum('k,kpq,kqp->',hf.kpoint_weights,d,hf.hcore+f).real)/2 for d,f in zip(ds,(fa,fb)))
    with pytest.raises(ValueError):
        cp._validate_physical(hf,method,hf._chi_restart_physical,madelung=.125)


@pytest.mark.parametrize('rewrite_controls',[False,True])
def test_opaque_request_label_cannot_bypass_executed_source(monkeypatch,tmp_path,rewrite_controls):
    ref,_,_=fixture(monkeypatch)
    hf=ref.hf_result
    original=hf._chi_restart_physical['controls']['coulomb_controls']['pair_cutoff_bohr']
    if rewrite_controls:
        hf._chi_restart_physical['controls']['coulomb_controls']['pair_cutoff_bohr']=original+1.
        hf._chi_hf_request_identity=cp._physical_request_identity(hf._chi_restart_physical)
    else:
        hf._chi_hf_request_identity='not-the-request-reconstructed-from-archived-inputs'
    forged=refs._seal_chi_hf_result(hf,'RHF')
    with pytest.raises(ValueError,match='request|source/support'):
        cp.save_chi_hf_reference(tmp_path/'bad.vqr',forged)


def test_private_basis_paths_are_refused_not_sanitized(monkeypatch,tmp_path):
    ref,_,_=fixture(monkeypatch)
    hf=ref.hf_result
    hf._chi_restart_physical['basis']['name']='relative/basis.g94'
    hf._chi_hf_request_identity=cp._physical_request_identity(hf._chi_restart_physical)
    forged=refs._seal_chi_hf_result(hf,'RHF')
    with pytest.raises(ValueError,match='portable registered basis labels'):
        cp.save_chi_hf_reference(tmp_path/'bad.vqr',forged)


@pytest.mark.parametrize('field,value',[('metric_threshold',1e-3),('base_reciprocal_cutoff_ha',20.),('executed_tail_reciprocal_cutoff_ha',20.)])
def test_source_descriptor_scalars_cannot_disagree_with_executed_support(monkeypatch,field,value):
    ref,_,_=fixture(monkeypatch)
    hf=ref.hf_result
    changed=replace(hf.fitted_operator.descriptor,**{field:value})
    operator=ChiFittedOperator(hf.fitted_operator,changed,hf.kpoint_weights)
    with pytest.raises(ValueError,match='descriptor scalar'):
        cp._validate_input_record(hf._chi_restart_physical,operator)
