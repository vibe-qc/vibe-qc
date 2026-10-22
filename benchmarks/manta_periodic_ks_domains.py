"""Pure-KS cutoff domain generation and complete finite SCF costs."""
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


def worker(profile,repeats):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    from tests.test_manta_periodic_ks import oracle,independent_scf
    from tests.test_manta_periodic_ks_domains import arguments,run_args
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require fresh own-checkout editable native build')
    species,name,spins={'rks':('H2','pbe',1),'uks':('HeH','pbe',2),'empty-spin':('H','blyp',2)}[profile]
    args,c,k,old=arguments(species,spins)
    run=run_args(args,c)
    start=time.perf_counter();manta.run_periodic_ks_from_cutoffs(**run);cold=time.perf_counter()-start
    wall=[];cpu=[]
    for _ in range(repeats):
        r=None;start=time.perf_counter();t=time.process_time();r=manta.run_periodic_ks_from_cutoffs(**run)
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-t)
    start=time.perf_counter();domains=manta.build_periodic_ks_domains(**args);generation_s=time.perf_counter()-start
    for key in ('reciprocal_vectors','translations','xc_images'):c[key]=getattr(domains,key)
    old.update(reciprocal_vectors=domains.reciprocal_vectors,translations=domains.translations)
    start=time.perf_counter();s,h,build=oracle(c,k,old)
    pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    density=independent_scf(s,h,build,pop,spins);f,energies,ne=build(r.density)
    oracle_s=time.perf_counter()-start
    errors=dict(density=float(np.max(abs(r.density-density))),potential=float(np.max(abs(r.fock-f))),
        energy=abs(r.receipt.energy-r.receipt.nuclear_energy-sum(energies)),
        populations=float(np.max(abs(np.array(r.receipt.xc.electrons_grid)-ne))))
    if not r.converged or errors['density']>2e-9 or max(errors[key] for key in ('potential','energy','populations'))>2e-10:
        raise RuntimeError('Independent physical KS acceptance failed')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(profile=profile,repeats=repeats,cold_s=cold,median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),
        cpu_median_s=statistics.median(cpu),oracle_s=oracle_s,generation_s=generation_s,
        domain_live_bytes=domains.plan.domain_live_bytes,domain_peak_bound=domains.plan.modeled_peak_bytes,
        hartree_vectors=domains.receipt.hartree_vectors,translations=domains.receipt.translations,xc_images=domains.receipt.xc_images,peak_rss_bytes=rss,modeled_peak_bytes=r.plan.modeled_peak_bytes,
        xc_panel_bytes=r.plan.xc.panel_bytes,grid_points=r.plan.xc.grid_points,n_basis=r.plan.n_basis,momenta=r.plan.momenta,
        fock_calls=r.receipt.fock_calls,iterations=r.solver_snapshot.evaluated_iteration,
        shell_calls=r.receipt.shell_calls,primitive_pairs=r.receipt.primitive_pairs,primitive_points=r.receipt.primitive_points,
        grid_population_error=r.receipt.maximum_grid_population_error,energy=r.receipt.energy,
        errors=errors,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('rks','uks','empty-spin'))
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    if a.repeats<3:p.error('At least three repeats required')
    if a.worker:print(json.dumps(worker(a.worker,a.repeats)));return
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');rows=[]
    for profile in ('rks','uks','empty-spin'):
        completed=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(a.repeats)],env=env,
            check=True,capture_output=True,text=True)
        rows.append(json.loads(completed.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
