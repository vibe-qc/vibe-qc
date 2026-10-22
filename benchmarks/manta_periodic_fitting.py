"""Matched finite-source periodic short-range fitting blocks and work receipts."""
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
    from tests.test_manta_periodic_fitting import case,libint_reference,analytic_metric
    from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture,_sr_three_center_oracle
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require fresh own-checkout native build')
    a=case()
    if profile=='s-gamma':a['q'][:]=0.;a['ket_kpoints']=np.zeros((1,3))
    if profile=='d-multi':
        a['system'],a['orbital'],a['auxiliary']=_weighted_gradient_fixture(2,orbital_angular=1)
        a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1
    start=time.perf_counter();manta.compute_periodic_gdf_short_range(**a);cold=time.perf_counter()-start
    wall=[];legacy=[]
    for _ in range(repeats):
        result=None;start=time.perf_counter();result=manta.compute_periodic_gdf_short_range(**a);wall.append(time.perf_counter()-start)
    for _ in range(repeats):
        m=t=None;start=time.perf_counter();m,t=libint_reference(a);legacy.append(time.perf_counter()-start)
    actual_m,actual_t=result.metric,result.three_center
    errors=dict(metric=float(np.max(abs(actual_m-m))),three_center=float(np.max(abs(actual_t-t))))
    if max(errors.values())>3e-11:raise RuntimeError('Same-source libint parity failed')
    independent=None
    if profile!='d-multi':
        o=a['options'];reference=np.array([_sr_three_center_oracle(a['system'],a['orbital'],a['auxiliary'],a['q'],k,
            o.omega,o.pair_cutoff_bohr,o.auxiliary_cutoff_bohr) for k in a['ket_kpoints']])
        independent=dict(metric=float(np.max(abs(actual_m-analytic_metric(a)))),three_center=float(np.max(abs(actual_t-reference))))
        if max(independent.values())>3e-11:raise RuntimeError('Analytic Gaussian image sums failed')
    # One general complex pair trial on the same finite metric, no SCF.
    rng=np.random.default_rng(1753);b=actual_t.reshape(len(actual_t),len(actual_m),-1)
    trial=rng.normal(size=b.shape[-1])+1j*rng.normal(size=b.shape[-1]);action_error=0.
    for got,ref in zip(b,t.reshape(b.shape)):
        x=got.conj().T@np.linalg.solve(actual_m,got@trial)
        y=ref.conj().T@np.linalg.solve(m,ref@trial)
        action_error=max(action_error,float(np.max(abs(x-y))))
    if action_error>2e-9:raise RuntimeError('Complex fitted response failed')
    return dict(profile=profile,repeats=repeats,cold_s=cold,median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),
        libint_median_s=statistics.median(legacy),libint_min_s=min(legacy),libint_max_s=max(legacy),
        n_orb=result.plan.n_orb,n_aux=result.plan.n_aux,momenta=result.plan.momenta,
        errors=errors,independent_errors=independent,complex_action_error=action_error,
        modeled_peak_bytes=result.plan.modeled_peak_bytes,output_bytes=result.plan.output_bytes,
        shell_calls_bound=result.plan.shell_calls,primitive_products_bound=result.plan.primitive_products,
        metric_calls=result.receipt.metric_shell_calls,three_center_calls=result.receipt.three_center_shell_calls,
        pair_shell_images=result.receipt.pair_shell_images,primitive_products=result.receipt.primitive_products,
        phase_products=result.receipt.phase_products,phase_accumulations=result.receipt.phase_accumulations,
        scalar_bound=result.plan.scalar_terms,kernel_bound=result.plan.kernel_terms,
        core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('s-gamma','s-multi','d-multi'))
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    if a.repeats<3:p.error('At least three repeats required')
    if a.worker:print(json.dumps(worker(a.worker,a.repeats)));return
    rows=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for profile in ('s-gamma','s-multi','d-multi'):
        r=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(a.repeats)],
            env=env,text=True,capture_output=True,check=True)
        rows.append(json.loads(r.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
