"""Complete periodic RI jobs followed by the shared nuclear error producer."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def job_context(directory,profile='ordinary-rhf'):
    import numpy as np
    from vibeqc import manta
    from tests.test_manta_periodic_exchange_bounds import ri_exchange_accuracy_job,compact_ri_exchange_accuracy_job
    if profile.startswith('compact'):
        job,a,k,scf=compact_ri_exchange_accuracy_job(directory)
        if profile=='compact-tight':
            job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],nuclear_cutoff_bohr=18.)
    else:
        method,species,limit={'ordinary-rhf':('rhf','Be',100),'open-uhf':('uhf','Li',3),
                              'global-pbe0':('pbe0','Be',100)}[profile]
        job,a,k,scf=ri_exchange_accuracy_job(directory,method,species,limit,automatic=True)
    job['integral_engine'].budget.memory_bytes=4*2**30
    # The existing HF domain factory has the same S/T/V domains as RI KS.
    # Its fit and K controls do not enter the nuclear source identity.
    method='rhf' if a['system'].multiplicity==1 else 'uhf'
    options=getattr(manta,'PeriodicRI'+method.upper()+'Options')()
    physical=job['periodic_integrals']
    for name in ('alpha','pair_cutoff_bohr','nuclear_cutoff_bohr','reciprocal_batch'):
        setattr(options,name,getattr(physical,name))
    options.fitting=a['options'].fitting;options.ri=a['options'].ri
    options.fitting.pair_cutoff_bohr=physical.pair_cutoff_bohr
    options.coulomb_algorithm='ri';options.exchange_algorithm='exact';options.scf=scf
    budget=manta.ResourceBudget();budget.memory_bytes=4*2**30
    for name in ('scalar_terms','kernel_terms','shell_calls','primitive_quartets'):
        setattr(budget,name,getattr(a['budget'],name))
    domains=getattr(manta,'build_periodic_ri_'+method+'_domains')(a['basis'],a['auxiliary'],a['system'],a['mesh'],
        physical.reciprocal_energy_cutoff,options,budget)
    k=np.array([a['system'].reciprocal_lattice()@a['mesh'].fractional_at(i) for i in range(len(a['mesh']))])
    bound_options=manta.PeriodicNuclearBoundOptions();bound_options.real_completion_cube_radius=5
    bound_options.reciprocal_completion_cube_radius=10 if profile.startswith('compact') else 6
    return job,dict(basis=a['basis'],system=a['system'],g=domains.reciprocal_vectors,cells=domains.translations,
        k=k,alpha=physical.alpha,cutoff=physical.nuclear_cutoff_bohr,paircut=physical.pair_cutoff_bohr,
        options=bound_options,budget=budget,preparation=a),scf


def certify(context,result):
    from vibeqc import manta
    c=context;n=result.native
    k=n.momenta if hasattr(n,'momenta') else c['k']
    # Reserve the completed method's live state conservatively beside the
    # standalone producer; this is not yet the pre-SCF method adapter.
    c['budget'].other_live_bytes=n.plan.modeled_peak_bytes
    bound=manta.bound_periodic_nuclear(c['basis'],c['system'],c['g'],c['cells'],k,c['alpha'],c['cutoff'],
        c['paircut'],c['options'],c['budget'])
    assert bound.receipt.nuclear_source_identity==n.receipt.nuclear_identity
    return bound


def run(repeats=3):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-nuclear-bounds-') as directory:
        for profile in ('ordinary-rhf','global-pbe0','compact-tight'):
            samples=[];previous=None
            for repeat in range(repeats+1):
                job,c,scf=job_context(Path(directory)/profile/str(repeat),profile)
                wall=time.perf_counter();cpu=time.process_time();result=vq.run_periodic_job(**job)
                calculation=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
                wall=time.perf_counter();cpu=time.process_time();bound=certify(c,result)
                sample=dict(calculation=calculation,bound=dict(wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu))
                assert result.converged
                if previous is None:cold=sample
                else:
                    np.testing.assert_array_equal(result.native.density,previous.native.density)
                    assert result.energy==previous.energy;samples.append(sample)
                previous=result
            r=bound.receipt
            rows.append(dict(profile=profile,energy=result.energy,iterations=result.iterations,
                method_peak_bytes=result.native.plan.modeled_peak_bytes,bound_joint_peak_bytes=bound.plan.modeled_peak_bytes,
                maximum_potential_error=r.maximum_potential_error,nuclear_energy_error_bound=r.nuclear_energy_error_bound,
                nuclear_real_error_bound=r.nuclear_real_error_bound,nuclear_reciprocal_error_bound=r.nuclear_reciprocal_error_bound,
                physical_source_identity=r.nuclear_source_identity,certificate_identity=r.source_identity,cold=cold,samples=samples))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    print(json.dumps(run(args.repeats),indent=2,allow_nan=False))
