"""Prepared shared auxiliary candidates retain full periodic fitting authority."""
import dataclasses
import numpy as np
import pytest
import vibeqc as vq
from vibeqc import manta
from vibeqc.auxiliary_generation import ProductAuxiliaryOptions, prepare_product_auxiliary_basis
from vibeqc.output import validate_qvf
from tests.test_manta_periodic_ri_domain_selection import domain_job


def generated_job(directory, method='rhf', refinement=0):
    job,args,accuracy=domain_job(directory,method)
    options=ProductAuxiliaryOptions();options.refinement=refinement
    generated=prepare_product_auxiliary_basis(args['system'].unit_cell_molecule(),args['basis'],args['auxiliary'],options)
    job['coulomb']=dataclasses.replace(job['coulomb'],auxiliary_basis=generated)
    args['auxiliary']=generated.basis
    job['citations']=True
    return job,args,accuracy,generated


@pytest.mark.parametrize('method',['rhf','uhf','rks','uks','pbe0','hse06'])
def test_generated_auxiliary_keeps_physical_owner_and_certificate(tmp_path,method):
    job,args,accuracy,generated=generated_job(tmp_path,method)
    result=vq.run_periodic_job(**job)
    control=vq.run_periodic_job(**dict(job,output=tmp_path/'plain'/'job',
        coulomb=dataclasses.replace(job['coulomb'],auxiliary_basis=generated.basis)))
    assert result.converged and control.converged and result.energy==control.energy
    assert result.native.receipt.source_identity==control.native.receipt.source_identity
    for name in ('density','fock','hcore','overlap'):
        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
    evidence=result.integral_provenance;plain=control.integral_provenance
    assert evidence['auxiliary_generation']['identity']==generated.receipt.identity
    assert evidence['job_reserved_bytes']-plain['job_reserved_bytes']==generated.plan.modeled_peak_bytes
    assert evidence['scalar_terms']-plain['scalar_terms']==generated.plan.work_units
    assert evidence['kernel_terms']-plain['kernel_terms']==32*generated.plan.work_units
    assert generated.receipt.citation_method in evidence['citation_methods']
    assert 'stoychev_autoaux_2017' in job['output'].with_suffix('.bibtex').read_text()
    full=evidence['fitting']['full_coulomb_accuracy']
    assert full['j_error_bound']<=accuracy.j_error and full['energy_error_bound']<=accuracy.energy_error
    assert full['certificate_identity']==plain['fitting']['full_coulomb_accuracy']['certificate_identity']
    assert not full['roundoff_included'] and not evidence['symmetry_reduced']
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']


@pytest.mark.parametrize('field,amount',[('scalar_terms',1),('kernel_terms',32),('memory_bytes',None)])
def test_generator_cost_is_admitted_before_periodic_values(tmp_path,field,amount):
    job,_,_,generated=generated_job(tmp_path)
    budget=job['integral_engine'].budget
    setattr(budget,field,generated.plan.modeled_peak_bytes-1 if amount is None else amount*generated.plan.work_units-1)
    with pytest.raises(ValueError,match='budget'):vq.run_periodic_job(**job)
    assert not job['output'].with_suffix('.qvf').exists()


def test_wrong_orbital_source_is_refused(tmp_path):
    job,_,_,generated=generated_job(tmp_path)
    with pytest.raises(ValueError,match='orbital source identity'):vq.run_periodic_job(**dict(job,basis=generated.basis))
    assert not job['output'].with_suffix('.qvf').exists()


def test_dry_run_does_not_claim_a_fitting_certificate(tmp_path):
    job,_,_,_=generated_job(tmp_path)
    assert vq.run_periodic_job(**dict(job,dry_run=True)) is None
    assert not job['output'].with_suffix('.qvf').exists()
