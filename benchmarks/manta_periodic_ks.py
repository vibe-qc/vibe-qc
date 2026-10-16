"""Finite physical KS against independent same-density and SciPy SCF results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time


def worker(profile,repeats,hybrid_q0=None,hybrid_source='explicit',range_separated=False):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    from tests.test_manta_periodic_ks import case,oracle,independent_scf
    run=manta.run_periodic_ks
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require fresh own-checkout editable native build')
    species,name,spins={'rks':('H2','pbe',1),'uks':('HeH','pbe',2),'empty-spin':('H','blyp',2)}[profile]
    if range_separated:
        from tests.test_manta_periodic_range_separated_ks import case,oracle,EMPTY_RANGE
        name={'rks':'hse06','uks':'cam-b3lyp','empty-spin':EMPTY_RANGE}[profile]
        run=manta.run_periodic_range_separated_ks
        c,k,old=case(species,name,spins,correction=hybrid_q0,order=10)
    elif hybrid_q0:
        from tests.test_manta_periodic_global_hybrid_ks import case,oracle,EMPTY_HYBRID
        name=EMPTY_HYBRID if species=='H' else 'pbe0'
        run=manta.run_periodic_global_hybrid_ks
        c,k,old=case(species,name,spins,correction=hybrid_q0,order=10)
    else:c,k,old=case(species,name,spins,order=10)
    inputs=c;admitted_peak=None
    if hybrid_source!='explicit':
        if range_separated:
            from tests.test_manta_periodic_range_separated_domains import arguments,explicit,run_args
        else:
            from tests.test_manta_periodic_global_hybrid_domains import arguments,explicit,run_args
        args,c,k,old=arguments(species,spins,hybrid_q0,grid=hybrid_source=='grid',order=10)
        method='periodic_range_separated_ks' if range_separated else 'periodic_global_hybrid_ks'
        if hybrid_source=='grid':
            inputs=args;run=getattr(manta,'run_'+method+'_with_quadrature')
            admitted_peak=getattr(manta,'plan_'+method+'_with_quadrature')(**args).modeled_peak_bytes
        else:
            inputs=run_args(args,c);run=getattr(manta,'run_'+method+'_from_cutoffs')
            admitted_peak=getattr(manta,'plan_'+method+'_from_cutoffs')(**args).modeled_peak_bytes
    start=time.perf_counter();run(**inputs);cold=time.perf_counter()-start
    wall=[];cpu=[]
    for _ in range(repeats):
        r=None;start=time.perf_counter();t=time.process_time();r=run(**inputs)
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-t)
    if hybrid_source!='explicit':explicit(args,c,old)
    start=time.perf_counter();s,h,build=oracle(c,k,old)[:3]
    pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    density=independent_scf(s,h,build,pop,spins);f,energies,ne=build(r.density)
    oracle_s=time.perf_counter()-start
    errors=dict(density=float(np.max(abs(r.density-density))),potential=float(np.max(abs(r.fock-f))),
        energy=abs(r.receipt.energy-r.receipt.nuclear_energy-sum(energies)),
        populations=float(np.max(abs(np.array(r.receipt.xc.electrons_grid)-ne))))
    actual=[r.receipt.one_electron_energy,r.receipt.coulomb_energy,r.receipt.xc_energy]
    if hybrid_q0:actual.extend((r.receipt.uncorrected_exchange_energy,r.receipt.exchange_correction_energy))
    errors['components']=float(np.max(abs(np.array(actual)-energies)))
    if not r.converged or errors['density']>2e-9 or max(errors[key] for key in ('potential','energy','populations','components'))>2e-10:
        raise RuntimeError('Independent physical KS acceptance failed')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(profile=profile,functional=name,hybrid_q0=hybrid_q0,hybrid_source=hybrid_source,range_separated=range_separated,repeats=repeats,cold_s=cold,median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),
        cpu_median_s=statistics.median(cpu),oracle_s=oracle_s,peak_rss_bytes=rss,modeled_peak_bytes=r.plan.modeled_peak_bytes,
        xc_panel_bytes=r.plan.xc.panel_bytes,grid_points=r.plan.xc.grid_points,n_basis=r.plan.n_basis,momenta=r.plan.momenta,
        fock_calls=r.receipt.fock_calls,iterations=r.solver_snapshot.evaluated_iteration,
        shell_calls=r.receipt.shell_calls,primitive_pairs=r.receipt.primitive_pairs,primitive_points=r.receipt.primitive_points,
        grid_population_error=r.receipt.maximum_grid_population_error,energy=r.receipt.energy,energy_components=actual,
        exact_exchange_fraction=r.receipt.exact_exchange_fraction,
        long_range_exchange_fraction=r.receipt.long_range_exchange_fraction,
        short_range_exchange_fraction=r.receipt.short_range_exchange_fraction,
        exchange_q0=r.receipt.exchange_q0,
        exchange_components=[dict(kernel=a.kernel,coefficient=a.coefficient,omega=a.omega,
            energy=a.energy,transfer_vectors=a.exchange.transfer_vectors,zero_vectors=a.exchange.zero_vectors,
            source_identity=a.exchange.source_identity) for a in r.receipt.exchange_components],
        admitted_peak_bytes=admitted_peak,domain_live_bytes=r.plan.domain_live_bytes,
        quadrature_live_bytes=r.plan.quadrature_live_bytes,domain_exchange_vectors=r.receipt.domain_exchange_vectors,
        domain_xc_images=r.receipt.domain_xc_images,
        errors=errors,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('rks','uks','empty-spin'))
    p.add_argument('--hybrid-q0',choices=('strict-zero-mode','BvK-ewald'),help='Measure complete hybrids with this explicit exchange policy')
    p.add_argument('--hybrid-source',choices=('explicit','cutoffs','grid'),default='explicit')
    p.add_argument('--range-separated',action='store_true',help='Use HSE06, CAM-B3LYP and LC-BLYP profiles')
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    if a.repeats<3:p.error('At least three repeats required')
    if a.hybrid_source!='explicit' and not a.hybrid_q0:p.error('--hybrid-source requires --hybrid-q0')
    if a.range_separated and not a.hybrid_q0:p.error('--range-separated requires --hybrid-q0')
    if a.worker:print(json.dumps(worker(a.worker,a.repeats,a.hybrid_q0,a.hybrid_source,a.range_separated)));return
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');rows=[]
    for profile in ('rks','uks','empty-spin'):
        completed=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(a.repeats)]
            +(['--hybrid-q0',a.hybrid_q0,'--hybrid-source',a.hybrid_source] if a.hybrid_q0 else [])
            +(['--range-separated'] if a.range_separated else []),env=env,
            check=True,capture_output=True,text=True)
        rows.append(json.loads(completed.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
