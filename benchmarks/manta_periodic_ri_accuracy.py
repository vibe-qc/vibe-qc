"""Measure explicit periodic RI-BUPO tolerances, fixed-density J and complete SCF."""
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


def worker(repeats,automatic=False,overlap_density=False,compare_scalar=False):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core, manta
    from tests.test_manta_periodic_ri_pair_tree import pair_actions_case,accuracy_job,automatic_accuracy
    from tests.test_manta_periodic_ri_q import builder,densities
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[];d=densities();w=np.array([.27,.73])
    def run(args,accuracy=None):
        start=time.perf_counter()
        if accuracy is None:b=builder(args)
        else:
            ri=manta.RIOptions();ri.mode_block=1
            b=manta.PeriodicRIQBuilder(manta.PeriodicGDFTiles(**args),2,ri,args['budget'],accuracy)
        setup=time.perf_counter()-start
        start=time.perf_counter();r=b.build(d,w,make_k=False);cold=time.perf_counter()-start
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter();cpu=time.process_time();warm=b.build(d,w,make_k=False)
            samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
            np.testing.assert_array_equal(warm.J,r.J)
        return b,r,dict(setup_seconds=setup,cold_seconds=cold,samples=samples)
    if not overlap_density:
        a=pair_actions_case(True,2,fitted=True)
        if automatic:a['budget'].memory_bytes=4*2**30
        a['options'].coupling_multipole_pair_tree=False;a['options'].coupling_multipole_error=0.
        exact,reference,times=run(a)
        rows.append(dict(kind='RI-J',order=0,epsilon=0.,peak_bytes=exact.plan.modeled_peak_bytes,**times))
        for order in (2,4,6):
            for epsilon in (1e-4,1e-6,1e-8):
                o=a['options'];o.coupling_multipole_pair_tree=True;o.coupling_multipole_error=epsilon;o.coupling_multipole_order=order
                accuracy=automatic_accuracy(epsilon,epsilon) if automatic else None
                if automatic:o.coupling_multipole_error=.1
                b,r,times=run(a,accuracy);np.testing.assert_array_equal(b.factor,exact.factor)
                delta=float(np.max(abs(r.J-reference.J)));de=abs(r.receipt.coulomb_energy-reference.receipt.coulomb_energy)
                assert delta<=r.receipt.approximation_j_bound+8e-12
                assert de<=r.receipt.approximation_energy_bound+8e-12
                selected=b.metric_info.accuracy
                if automatic:assert selected.qualified and selected.j_bound<=epsilon and selected.energy_bound<=epsilon
                rows.append(dict(kind='RI-J',order=order,epsilon=epsilon,automatic=automatic,
                    selected_epsilon=selected.selected_coupling_error,selection_attempts=selected.attempts,
                    selection_j_bound=selected.j_bound,selection_energy_bound=selected.energy_bound,
                    J_error=delta,energy_error=de,
                    J_bound=r.receipt.approximation_j_bound,energy_bound=r.receipt.approximation_energy_bound,
                    coupling_bound=r.receipt.coupling_error_bound,forward_l2_bound=r.receipt.forward_error_l2_bound,
                    far_blocks=r.receipt.coupling_multipole_far_blocks,pair_groups=r.receipt.coupling_multipole_pair_groups,
                    partition_builds=r.receipt.coupling_multipole_partition_builds,
                    partition_reuses=r.receipt.coupling_multipole_partition_reuses,
                    shell_calls=r.receipt.shell_calls,peak_bytes=b.plan.modeled_peak_bytes,**times))
    with tempfile.TemporaryDirectory(prefix='manta-ri-accuracy-') as directory:
        directory=Path(directory);exact=None;resolved_limit=None
        cases=[(0,0.,False)]+[(order,1e-8,scalar) for order in (2,4,6)
                              for scalar in ((False,True) if compare_scalar else (False,))]
        for order,epsilon,scalar in cases:
            samples=[];result=None
            for repeat in range(repeats+1):
                job,_,_=accuracy_job(directory/(str(order)+('-scalar' if scalar else ''))/str(repeat),order,.1 if automatic and epsilon else epsilon)
                if automatic:job['integral_engine'].budget.memory_bytes=4*2**30
                if automatic and epsilon:
                    job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],accuracy=automatic_accuracy(
                        1e-6,1e-6,resolved_limit if scalar else (0. if overlap_density else 256.)))
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
            selection=result.integral_provenance['fitting']['accuracy']
            if overlap_density and epsilon and not scalar:resolved_limit=selection['density_l1_limit']
            rows.append(dict(kind='complete-RHF',order=order,epsilon=epsilon,automatic=automatic and bool(epsilon),scalar_overlap_control=scalar,
                selection=selection,peak_bytes=result.native.plan.modeled_peak_bytes,cold_seconds=cold,samples=samples,
                energy=result.energy,relaxed_energy_difference=abs(result.energy-exact.energy),
                relaxed_density_difference=float(np.max(abs(result.native.density-exact.native.density))),
                iterations=result.iterations,far_blocks=r.coupling_multipole_far_blocks,
                coupling_bound=r.coupling_error_bound,J_bound=r.approximation_j_bound,energy_bound=r.approximation_energy_bound))
    pins=('cpp/src/bounded_periodic_rhf_solver.cpp','cpp/include/vibeqc/bounded_periodic_rhf_solver.hpp',
        'tests/test_bounded_periodic_rhf_solver.py','cpp/src/manta_lattice_tail.cpp','cpp/src/manta_lattice_errors.cpp','cpp/include/vibeqc/manta_lattice_errors.hpp',
        'cpp/src/periodic_gdf_short_range.cpp','cpp/src/manta_periodic_ri.cpp',
        'tests/test_manta_erfc_gaussian.py','tests/test_manta_periodic_ri_pair_tree.py',
        'benchmarks/manta_periodic_ri_accuracy.py')
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
        native_sha256=sha(native),source_sha256={n:sha(root/n) for n in pins},rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--automatic',action='store_true');p.add_argument('--overlap-density',action='store_true');p.add_argument('--compare-scalar',action='store_true');p.add_argument('--repeats',type=int,default=3);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.workers,a.repeats)<1:p.error('workers and repeats must be positive')
    if a.overlap_density and not a.automatic:p.error('--overlap-density requires --automatic')
    if a.compare_scalar and not a.overlap_density:p.error('--compare-scalar requires --overlap-density')
    if a.worker:print(json.dumps(worker(a.repeats,a.automatic,a.overlap_density,a.compare_scalar),allow_nan=False));return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
    workers=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker','--repeats',str(a.repeats)]+(['--automatic'] if a.automatic else [])+(['--overlap-density'] if a.overlap_density else [])+(['--compare-scalar'] if a.compare_scalar else []),env=env,text=True)) for _ in range(a.workers)]
    print(json.dumps(dict(workers=workers),indent=2,allow_nan=False))


if __name__=='__main__':main()
