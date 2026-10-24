"""Joint grid/domain/KS construction with independent finite-source SCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(profile,repeats):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    from tests.test_manta_periodic_ks_grid import case,explicit
    from tests.test_manta_periodic_ks import oracle,independent_scf
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require fresh own-checkout native build')
    species,spins={'rks':('H2',1),'uks':('HeH',2),'empty-spin':('H',2)}[profile]
    args,c,k,old=case(species,spins);plan=manta.plan_periodic_ks_with_quadrature(**args)
    start=time.perf_counter();manta.run_periodic_ks_with_quadrature(**args);cold=time.perf_counter()-start
    wall=[];cpu=[]
    for _ in range(repeats):
        r=None;start=time.perf_counter();t=time.process_time();r=manta.run_periodic_ks_with_quadrature(**args)
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-t)
    grid,domains=explicit(args,c,old);s,h,build=oracle(c,k,old)
    pop=(r.plan.n_alpha,) if spins==1 else (r.plan.n_alpha,r.plan.n_beta)
    d=independent_scf(s,h,build,pop,spins);f,e,n=build(r.density)
    errors=dict(density=float(np.max(abs(r.density-d))),potential=float(np.max(abs(r.fock-f))),
        energy=abs(r.receipt.energy-r.receipt.nuclear_energy-sum(e)),
        populations=float(np.max(abs(np.array(r.receipt.xc.electrons_grid)-n))))
    if not r.converged or errors['density']>3e-9 or max(errors[key] for key in ('potential','energy','populations'))>2e-11:
        raise RuntimeError('Independent finite-source SCF acceptance failed')
    return dict(profile=profile,repeats=repeats,cold_s=cold,median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),
        cpu_median_s=statistics.median(cpu),grid_points=grid.plan.grid_points,grid_live_bytes=plan.grid_live_bytes,
        aggregate_peak_bound=plan.modeled_peak_bytes,actual_source_modeled_peak=r.plan.modeled_peak_bytes,
        scalar_bound=plan.scalar_terms,kernel_bound=plan.kernel_terms,iterations=r.solver_snapshot.evaluated_iteration,
        fock_calls=r.receipt.fock_calls,n_basis=r.plan.n_basis,momenta=r.plan.momenta,
        grid_population_error=r.receipt.maximum_grid_population_error,errors=errors,
        core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('rks','uks','empty-spin'))
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    if a.repeats<3:p.error('At least three repeats required')
    if a.worker:print(json.dumps(worker(a.worker,a.repeats)));return
    rows=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for profile in ('rks','uks','empty-spin'):
        r=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(a.repeats)],
            check=True,capture_output=True,text=True,env=env);rows.append(json.loads(r.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
