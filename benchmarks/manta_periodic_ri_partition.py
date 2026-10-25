"""Measured bounded source-owned periodic RI partitions in paired actions."""
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


def worker(repeats):
    root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_ri_multipole import multipole_case
    from tests.test_manta_periodic_ri_q import builder, densities
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root): raise RuntimeError('Require this checkout native runtime')
    a=multipole_case(fitted=True); a['options'].coupling_multipole_error=0.; d=densities(); w=np.array([.27,.73]); exact=builder(a)
    ref=exact.build(d,w,make_k=False); rows=[]
    for epsilon, order, compress, capacity in [(0.,0,False,0)]+[(e,o,c,b) for e,o in ((1.,0),(100.,2),(100.,4)) for c in (False,True) for b in (0,64*1024*1024)]:
        a['options'].coupling_multipole_error=epsilon; a['options'].coupling_multipole_order=order; a['options'].coupling_multipole_hierarchy=True; a['options'].coupling_multipole_compress=compress; a['options'].coupling_multipole_reuse=True; a['options'].coupling_multipole_partition_bytes=capacity; b=builder(a)
        start=time.perf_counter(); r=b.build(d,w,make_k=False); cold_seconds=time.perf_counter()-start; receipt=r.receipt
        np.testing.assert_array_equal(b.factor,exact.factor)
        error=float(np.max(abs(r.J-ref.J))); energy_error=abs(receipt.coulomb_energy-ref.receipt.coulomb_energy)
        assert error<=receipt.approximation_j_bound+5e-12
        assert energy_error<=receipt.approximation_energy_bound+5e-12
        samples=[]
        for _ in range(repeats):
            start=time.perf_counter(); cpu=time.process_time(); warm=b.build(d,w,make_k=False)
            samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
            np.testing.assert_array_equal(warm.J,r.J)
            if epsilon and compress:
                assert warm.receipt.coupling_multipole_moment_panels==warm.receipt.coupling_multipole_envelope_evaluations==0
                assert warm.receipt.coupling_multipole_reused_moment_panels>0
                assert warm.receipt.coupling_multipole_compressed_nodes==warm.receipt.coupling_multipole_far_blocks>0
        rows.append(dict(kind='RI-J',epsilon=epsilon,order=order,compress=compress,capacity=capacity,
            cold_partition_builds=receipt.coupling_multipole_partition_builds,cold_partition_reuses=receipt.coupling_multipole_partition_reuses,
            warm_partition_builds=warm.receipt.coupling_multipole_partition_builds,warm_partition_reuses=warm.receipt.coupling_multipole_partition_reuses,warm_partition_bypasses=warm.receipt.coupling_multipole_partition_bypasses,cold_seconds=cold_seconds,samples=samples,J_error=error,energy_error=energy_error,
            J_bound=receipt.approximation_j_bound,energy_bound=receipt.approximation_energy_bound,
            shell_calls=warm.receipt.shell_calls,primitive_products=warm.receipt.primitive_products,
            cold_moment_panels=receipt.coupling_multipole_moment_panels,cold_envelope_evaluations=receipt.coupling_multipole_envelope_evaluations,
            warm_moment_panels=warm.receipt.coupling_multipole_moment_panels,warm_envelope_evaluations=warm.receipt.coupling_multipole_envelope_evaluations,
            reused_panels=warm.receipt.coupling_multipole_reused_moment_panels,reused_extents=warm.receipt.coupling_multipole_reused_extents,
            compressed_pairs=warm.receipt.coupling_multipole_compressed_pairs,compressed_nodes=warm.receipt.coupling_multipole_compressed_nodes,
            action_products=warm.receipt.coupling_multipole_action_products,downward_translations=warm.receipt.coupling_multipole_action_translations,
            passes=receipt.coupling_passes,peak_bytes=b.plan.modeled_peak_bytes,
            evaluated_coupling_error=receipt.coupling_error_bound,
            tree_nodes=receipt.coupling_multipole_tree_nodes,far_blocks=receipt.coupling_multipole_far_blocks,
            translations=receipt.coupling_multipole_translations,multipole_trials=receipt.coupling_multipole_trials,multipole_images=receipt.coupling_multipole_images,
            source_identity=receipt.source_identity,reference_coupling_identity=receipt.reference_coupling_identity,
            factor_identity=b.metric_info.factor_identity))
    from vibeqc import manta
    from tests.test_manta_periodic_ri_compressed import angular_case
    from tests.test_manta_periodic_gdf_tiles import assemble
    from tests.test_manta_periodic_gdf_actions import inputs
    for pure in (False,True):
        a=angular_case(pure);t=assemble(manta.PeriodicGDFTiles(**a))
        for compress,capacity in [(c,b) for c in (False,True) for b in (0,64*1024*1024)]:
            a['options'].coupling_multipole_compress=compress;a['options'].coupling_multipole_partition_bytes=capacity;source=manta.PeriodicGDFTiles(**a);x,c=inputs(source)
            ef=np.einsum('kpij,kij->p',t.conj(),x);ea=np.einsum('kpij,p->kij',t,c)
            start=time.perf_counter();f=source.forward(x,a['budget']);rev=source.adjoint(c,a['budget']);cold_seconds=time.perf_counter()-start
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter();cpu=time.process_time();wf=source.forward(x,a['budget']);wr=source.adjoint(c,a['budget'])
                samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
                np.testing.assert_array_equal(wf.values,f.values);np.testing.assert_array_equal(wr.values,rev.values)
            np.testing.assert_allclose(f.values,ef,atol=4e-12,rtol=3e-13);np.testing.assert_allclose(rev.values,ea,atol=4e-12,rtol=3e-13)
            assert f.receipt.maximum_coupling>=np.max(abs(t))
            if compress:assert f.receipt.multipole_compressed_nodes==f.receipt.multipole_far_blocks>0
            rows.append(dict(kind='spherical-d' if pure else 'Cartesian-d',epsilon=a['options'].coupling_multipole_error,
                order=a['options'].coupling_multipole_order,compress=compress,capacity=capacity,
                warm_partition_builds=wf.receipt.multipole_partition_builds+wr.receipt.multipole_partition_builds,warm_partition_reuses=wf.receipt.multipole_partition_reuses+wr.receipt.multipole_partition_reuses,
                cache_live_bytes=source.partitions.modeled_live_bytes,cache_entries=source.partitions.entries,cache_evictions=source.partitions.evictions,cold_seconds=cold_seconds,samples=samples,
                forward_error=float(np.max(abs(f.values-ef))),adjoint_error=float(np.max(abs(rev.values-ea))),
                maximum_coupling_bound=f.receipt.maximum_coupling,actual_maximum_coupling=float(np.max(abs(t))),
                compressed_pairs=wf.receipt.multipole_compressed_pairs+wr.receipt.multipole_compressed_pairs,
                compressed_nodes=wf.receipt.multipole_compressed_nodes+wr.receipt.multipole_compressed_nodes,
                action_products=wf.receipt.multipole_action_products+wr.receipt.multipole_action_products,
                downward_translations=wf.receipt.multipole_action_translations,
                peak_bytes=f.plan.modeled_peak_bytes,source_identity=f.receipt.coupling_identity))
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rows=rows,
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
            ('cpp/src/manta_periodic_ri.cpp','cpp/src/periodic_gdf_short_range.cpp',
             'cpp/src/manta_lattice_tail.cpp','cpp/src/manta_lattice_errors.cpp',
             'cpp/include/vibeqc/manta_periodic_error.hpp','tests/test_manta_periodic_ri_multipole.py','cpp/include/vibeqc/manta_periodic_fitting.hpp',
             'tests/test_manta_periodic_ri_compressed.py','tests/test_manta_periodic_ri_partition.py','benchmarks/manta_periodic_ri_partition.py')},
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.workers,a.repeats)<1:p.error('workers and repeats must be positive')
    if a.worker:print(json.dumps(worker(a.repeats),allow_nan=False));return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
    rows=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker','--repeats',str(a.repeats)],env=env,text=True)) for _ in range(a.workers)]
    keys=[(r['kind'],r['epsilon'],r['order'],r['compress'],r['capacity']) for r in rows[0]['rows']]
    medians={f'{kind}:{e}:{order}:{compressed}:{capacity}':statistics.median(s['wall_seconds'] for worker in rows for r in worker['rows'] if (r['kind'],r['epsilon'],r['order'],r['compress'],r['capacity'])==(kind,e,order,compressed,capacity) for s in r['samples']) for kind,e,order,compressed,capacity in keys}
    print(json.dumps(dict(workers=rows,median_seconds=medians),indent=2,allow_nan=False))


if __name__=='__main__':main()
