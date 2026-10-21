"""Measure global pair-tree RI-BUPO against the existing call-wide auxiliary hierarchy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(repeats):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import manta,_vibeqc_core as core
    from tests.test_manta_periodic_ri_compressed import compressed_case,angular_case
    from tests.test_manta_periodic_ri_q import builder,densities
    from tests.test_manta_periodic_gdf_actions import inputs
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[];d=densities();w=np.array([.27,.73])
    def counts(r,prefix):
        return {name:getattr(r,prefix+name) for name in ('multipole_action_products','multipole_global_pairs',
            'multipole_global_projections','multipole_global_auxiliary_projections','multipole_pair_groups',
            'multipole_far_blocks','multipole_images','multipole_translations','multipole_action_translations')}
    for order in (0,2,4):
        a=compressed_case(True,order,fitted=True);error=a['options'].coupling_multipole_error
        a['options'].coupling_multipole_error=0.;exact=builder(a);ref=exact.build(d,w,make_k=False)
        a['options'].coupling_multipole_error=error;a['options'].coupling_multipole_global_actions=True
        for enabled in (False,True):
            a['options'].coupling_multipole_pair_tree=enabled;b=builder(a)
            start=time.perf_counter();r=b.build(d,w,make_k=False);cold=time.perf_counter()-start
            np.testing.assert_array_equal(b.factor,exact.factor)
            difference=float(np.max(abs(r.J-ref.J)));energy_difference=abs(r.receipt.coulomb_energy-ref.receipt.coulomb_energy)
            assert difference<=r.receipt.approximation_j_bound+5e-12
            assert energy_difference<=r.receipt.approximation_energy_bound+5e-12
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter();cpu=time.process_time();warm=b.build(d,w,make_k=False)
                samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
                np.testing.assert_array_equal(warm.J,r.J)
            rows.append(dict(kind='RI-J',order=order,enabled=enabled,cold_seconds=cold,samples=samples,
                difference=difference,energy_difference=energy_difference,J_bound=r.receipt.approximation_j_bound,
                energy_bound=r.receipt.approximation_energy_bound,forward_l2_bound=r.receipt.forward_error_l2_bound,
                requested_coupling_error=error,
                evaluated_coupling_error=r.receipt.coupling_error_bound,peak_bytes=b.plan.modeled_peak_bytes,
                factor_identity=b.metric_info.factor_identity,**counts(warm.receipt,'coupling_')))
    raw=[('Cartesian-d',angular_case(False)),('spherical-d',angular_case(True))]
    a=compressed_case(False,2);shells=[]
    for i in range(8):
        block=a['auxiliary'].shells()
        for shell in block:shell.exponents=[e*(1+.07*i) for e in shell.exponents]
        shells+=block
    a['auxiliary']=vq.BasisSet(a['system'].unit_cell_molecule(),shells,'pair-tree-benchmark',True)
    a['ket_kpoints']=np.vstack([a['ket_kpoints']+.013*i for i in range(8)])
    raw.append(('many-auxiliary-many-k',a))
    for label,a in raw:
        error=a['options'].coupling_multipole_error;a['options'].coupling_multipole_error=0.
        exact=manta.PeriodicGDFTiles(**a);x,c=inputs(exact);ef=exact.forward(x,a['budget']).values;ea=exact.adjoint(c,a['budget']).values
        a['options'].coupling_multipole_error=error;a['options'].coupling_multipole_global_actions=True
        for enabled in (False,True):
            a['options'].coupling_multipole_pair_tree=enabled;source=manta.PeriodicGDFTiles(**a)
            start=time.perf_counter();f=source.forward(x,a['budget']);rev=source.adjoint(c,a['budget']);cold=time.perf_counter()-start
            forward_difference=float(np.max(abs(f.values-ef)));adjoint_difference=float(np.max(abs(rev.values-ea)))
            assert forward_difference<=f.receipt.approximation_bound+5e-12
            assert adjoint_difference<=rev.receipt.approximation_bound+5e-12
            assert np.linalg.norm(f.values-ef)<=f.receipt.approximation_l2_bound+1e-11
            assert np.linalg.norm((rev.values-ea).ravel())<=rev.receipt.approximation_l2_bound+1e-11
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter();cpu=time.process_time();wf=source.forward(x,a['budget']);wr=source.adjoint(c,a['budget'])
                samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
                np.testing.assert_array_equal(wf.values,f.values);np.testing.assert_array_equal(wr.values,rev.values)
            fc=counts(wf.receipt,'');rc=counts(wr.receipt,'')
            rows.append(dict(kind=label,order=a['options'].coupling_multipole_order,enabled=enabled,cold_seconds=cold,samples=samples,
                forward_difference=forward_difference,adjoint_difference=adjoint_difference,
                forward_bound=f.receipt.approximation_bound,adjoint_bound=rev.receipt.approximation_bound,
                forward_l2_bound=f.receipt.approximation_l2_bound,adjoint_l2_bound=rev.receipt.approximation_l2_bound,
                maximum_coupling_bound=f.receipt.maximum_coupling,
                requested_coupling_error=error,evaluated_coupling_error=f.receipt.coupling_error_bound,
                peak_bytes=f.plan.modeled_peak_bytes,n_aux=source.plan.n_aux,n_orb=source.plan.n_orb,momenta=source.plan.momenta,
                **{k:fc[k]+rc[k] for k in fc}))
    pins=('cpp/src/periodic_gdf_short_range.cpp','cpp/include/vibeqc/manta_periodic_fitting.hpp',
        'cpp/include/vibeqc/manta_periodic_gdf_pair_tree.hpp','cpp/src/manta_periodic_ri.cpp',
        'tests/test_manta_periodic_ri_pair_tree.py','tests/test_manta_periodic_ri_compressed.py','benchmarks/manta_periodic_ri_pair_tree.py')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
        native_sha256=sha(native),source_sha256={n:sha(root/n) for n in pins},rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);p.add_argument('--repeats',type=int,default=3);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.workers,a.repeats)<1:p.error('workers and repeats must be positive')
    if a.worker:print(json.dumps(worker(a.repeats),allow_nan=False));return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
    workers=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker','--repeats',str(a.repeats)],env=env,text=True)) for _ in range(a.workers)]
    print(json.dumps(dict(workers=workers),indent=2,allow_nan=False))


if __name__=='__main__':main()
