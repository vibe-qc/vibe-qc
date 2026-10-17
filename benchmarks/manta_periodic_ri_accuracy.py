"""Measure explicit periodic RI-BUPO tolerances, fixed-density J and complete SCF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time


def worker(repeats):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_ri_pair_tree import pair_actions_case,accuracy_job
    from tests.test_manta_periodic_ri_q import builder,densities
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[];d=densities();w=np.array([.27,.73])
    def run(args):
        start=time.perf_counter();b=builder(args);setup=time.perf_counter()-start
        start=time.perf_counter();r=b.build(d,w,make_k=False);cold=time.perf_counter()-start
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter();cpu=time.process_time();warm=b.build(d,w,make_k=False)
            samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
            np.testing.assert_array_equal(warm.J,r.J)
        return b,r,dict(setup_seconds=setup,cold_seconds=cold,samples=samples)
    a=pair_actions_case(True,2,fitted=True)
    a['options'].coupling_multipole_pair_tree=False;a['options'].coupling_multipole_error=0.
    exact,reference,times=run(a)
    rows.append(dict(kind='RI-J',order=0,epsilon=0.,peak_bytes=exact.plan.modeled_peak_bytes,**times))
    for order in (2,4,6):
        for epsilon in (1e-4,1e-6,1e-8):
            o=a['options'];o.coupling_multipole_pair_tree=True;o.coupling_multipole_error=epsilon;o.coupling_multipole_order=order
            b,r,times=run(a);np.testing.assert_array_equal(b.factor,exact.factor)
            delta=float(np.max(abs(r.J-reference.J)));de=abs(r.receipt.coulomb_energy-reference.receipt.coulomb_energy)
            assert delta<=r.receipt.approximation_j_bound+8e-12
            assert de<=r.receipt.approximation_energy_bound+8e-12
            rows.append(dict(kind='RI-J',order=order,epsilon=epsilon,J_error=delta,energy_error=de,
                J_bound=r.receipt.approximation_j_bound,energy_bound=r.receipt.approximation_energy_bound,
                coupling_bound=r.receipt.coupling_error_bound,forward_l2_bound=r.receipt.forward_error_l2_bound,
                far_blocks=r.receipt.coupling_multipole_far_blocks,pair_groups=r.receipt.coupling_multipole_pair_groups,
                partition_builds=r.receipt.coupling_multipole_partition_builds,
                partition_reuses=r.receipt.coupling_multipole_partition_reuses,
                shell_calls=r.receipt.shell_calls,peak_bytes=b.plan.modeled_peak_bytes,**times))
    with tempfile.TemporaryDirectory(prefix='manta-ri-accuracy-') as directory:
        directory=Path(directory);exact=None
        for order,epsilon in ((0,0.),(2,1e-8),(4,1e-8),(6,1e-8)):
            samples=[];result=None
            for repeat in range(repeats+1):
                job,_,_=accuracy_job(directory/str(order)/str(repeat),order,epsilon)
                start=time.perf_counter();cpu=time.process_time();current=vq.run_periodic_job(**job)
                timing=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
                assert current.converged
                if result is not None:
                    np.testing.assert_array_equal(current.native.density,result.native.density)
                    assert current.energy==result.energy
                    samples.append(timing)
                else:result=current;cold=timing['wall_seconds']
            if not epsilon:exact=result
            r=result.native.receipt.fitted_fock
            rows.append(dict(kind='complete-RHF',order=order,epsilon=epsilon,cold_seconds=cold,samples=samples,
                energy=result.energy,relaxed_energy_difference=abs(result.energy-exact.energy),
                relaxed_density_difference=float(np.max(abs(result.native.density-exact.native.density))),
                iterations=result.iterations,far_blocks=r.coupling_multipole_far_blocks,
                coupling_bound=r.coupling_error_bound,J_bound=r.approximation_j_bound,energy_bound=r.approximation_energy_bound))
    pins=('cpp/src/manta_lattice_tail.cpp','cpp/src/manta_lattice_errors.cpp','cpp/include/vibeqc/manta_lattice_errors.hpp',
        'cpp/src/periodic_gdf_short_range.cpp','cpp/src/manta_periodic_ri.cpp',
        'tests/test_manta_erfc_gaussian.py','tests/test_manta_periodic_ri_pair_tree.py',
        'benchmarks/manta_periodic_ri_accuracy.py')
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
        native_sha256=sha(native),source_sha256={n:sha(root/n) for n in pins},rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.workers,a.repeats)<1:p.error('workers and repeats must be positive')
    if a.worker:print(json.dumps(worker(a.repeats),allow_nan=False));return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
    workers=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker','--repeats',str(a.repeats)],env=env,text=True)) for _ in range(a.workers)]
    print(json.dumps(dict(workers=workers),indent=2,allow_nan=False))


if __name__=='__main__':main()
