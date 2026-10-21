"""Complete finite SR/LR fitting-source parity, work and timings."""
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
    from tests.test_manta_periodic_gdf import case,legacy,independent
    from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require fresh own-checkout native build')
    a=case(profile=='s-gamma')
    if profile=='s-gamma':a['ket_kpoints']=np.zeros((1,3))
    if profile=='d-multi':
        import itertools
        a['system'],a['orbital'],a['auxiliary']=_weighted_gradient_fixture(2,orbital_angular=1)
        a['vectors']=np.array(list(itertools.product(range(-1,2),repeat=3)))@a['system'].reciprocal_lattice().T+a['q']
        a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1
    start=time.perf_counter();cold=manta.compute_periodic_gdf(**a);cm,ct=cold.metric,cold.three_center
    cold_time=time.perf_counter()-start;del cold,cm,ct
    wall=[];reference_wall=[];r=m=t=None
    for _ in range(repeats):
        r=m=t=None;start=time.perf_counter();r=manta.compute_periodic_gdf(**a);m,t=r.metric,r.three_center
        wall.append(time.perf_counter()-start)
    rm=rt=None
    for _ in range(repeats):
        rm=rt=None;start=time.perf_counter();rm,rt,_=legacy(a);reference_wall.append(time.perf_counter()-start)
    errors=dict(metric=float(np.max(abs(m-rm))),three_center=float(np.max(abs(t-rt))))
    if max(errors.values())>3e-11:raise RuntimeError('Matched finite source parity failed')
    independent_errors=None
    if profile!='d-multi':
        im,it=independent(a);independent_errors=dict(metric=float(np.max(abs(m-im))),three_center=float(np.max(abs(t-it))))
        if max(independent_errors.values())>3e-11:raise RuntimeError('Independent Gaussian image/Fourier sums failed')
    rng=np.random.default_rng(911);b=t.reshape(len(t),len(m),-1)
    trial=rng.normal(size=b.shape[-1])+1j*rng.normal(size=b.shape[-1]);action_error=0.
    for actual,reference in zip(b,rt.reshape(b.shape)):
        x=actual.conj().T@np.linalg.solve(m,actual@trial)
        y=reference.conj().T@np.linalg.solve(rm,reference@trial)
        action_error=max(action_error,float(np.max(abs(x-y))))
    if action_error>1e-8:raise RuntimeError('Complex fitted action parity failed')
    return dict(profile=profile,repeats=repeats,cold_s=cold_time,median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),
        libint_median_s=statistics.median(reference_wall),libint_min_s=min(reference_wall),libint_max_s=max(reference_wall),
        n_orb=r.plan.n_orb,n_aux=r.plan.n_aux,momenta=r.plan.momenta,vectors=r.plan.vectors,
        errors=errors,independent_errors=independent_errors,complex_action_error=action_error,
        metric_min_eigenvalue=float(np.linalg.eigvalsh(m).min()),output_bytes=r.plan.output_bytes,
        modeled_peak_bytes=r.plan.modeled_peak_bytes,shell_bound=r.plan.shell_calls,primitive_bound=r.plan.primitive_products,
        scalar_bound=r.plan.scalar_terms,kernel_bound=r.plan.kernel_terms,shell_calls=r.receipt.shell_calls,
        primitive_products=r.receipt.primitive_products,vector_primitive_products=r.receipt.vector_primitive_products,
        sr_metric_calls=r.receipt.short_range.metric_shell_calls,sr_three_center_calls=r.receipt.short_range.three_center_shell_calls,
        auxiliary_fourier_calls=r.receipt.auxiliary_fourier_shell_calls,pair_fourier_calls=r.receipt.pair_fourier_shell_calls,
        phase_products=r.receipt.phase_products,phase_accumulations=r.receipt.phase_accumulations,
        contraction_terms=r.receipt.contraction_terms,finite_zero_removed=r.receipt.finite_zero_removed,
        core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('s-gamma','s-multi','d-multi'))
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    if a.repeats<3:p.error('At least three repeats required')
    if a.worker:print(json.dumps(worker(a.worker,a.repeats)));return
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');rows=[]
    for profile in ('s-gamma','s-multi','d-multi'):
        completed=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(a.repeats)],
            env=env,text=True,capture_output=True,check=True)
        rows.append(json.loads(completed.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
