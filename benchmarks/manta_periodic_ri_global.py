"""Measure call-wide periodic RI pair charges/fields against per-tile actions."""
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
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import manta, _vibeqc_core as core
    from tests.test_manta_periodic_ri_compressed import compressed_case,angular_case
    from tests.test_manta_periodic_ri_q import builder,densities
    from tests.test_manta_periodic_gdf_actions import inputs
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[];d=densities();w=np.array([.27,.73])
    def counts(r,prefix):
        return {name:getattr(r,prefix+name) for name in ('multipole_action_products','multipole_global_pairs','multipole_global_pair_reuses','multipole_global_auxiliary','multipole_global_projections','multipole_global_auxiliary_projections')}
    for order in (0,2,4):
        a=compressed_case(True,order,fitted=True);old=builder(a);ref=old.build(d,w,make_k=False)
        for enabled in (False,True):
            a['options'].coupling_multipole_global_actions=enabled;b=builder(a)
            start=time.perf_counter();r=b.build(d,w,make_k=False);cold=time.perf_counter()-start
            np.testing.assert_array_equal(b.factor,old.factor)
            np.testing.assert_allclose(r.J,ref.J,atol=5e-12,rtol=3e-13)
            assert abs(r.receipt.coulomb_energy-ref.receipt.coulomb_energy)<5e-12
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter();cpu=time.process_time();warm=b.build(d,w,make_k=False)
                samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
                np.testing.assert_array_equal(warm.J,r.J)
            rows.append(dict(kind='RI-J',order=order,enabled=enabled,cold_seconds=cold,samples=samples,
                difference=float(np.max(abs(r.J-ref.J))),energy_difference=abs(r.receipt.coulomb_energy-ref.receipt.coulomb_energy),
                peak_bytes=b.plan.modeled_peak_bytes,factor_identity=b.metric_info.factor_identity,**counts(warm.receipt,'coupling_')))
    raw=[('Cartesian-d',angular_case(False)),('spherical-d',angular_case(True))]
    a=compressed_case(False,2);shells=[]
    for i in range(8):
        block=a['auxiliary'].shells()
        for shell in block:shell.exponents=[e*(1+.07*i) for e in shell.exponents]
        shells+=block
    a['auxiliary']=vq.BasisSet(a['system'].unit_cell_molecule(),shells,'global-panel-benchmark',True)
    a['ket_kpoints']=np.vstack([a['ket_kpoints']+.013*i for i in range(8)])
    raw.append(('many-auxiliary-many-k',a))
    for label,a in raw:
        old=manta.PeriodicGDFTiles(**a);x,c=inputs(old);ef=old.forward(x,a['budget']).values;ea=old.adjoint(c,a['budget']).values
        for enabled in (False,True):
            a['options'].coupling_multipole_global_actions=enabled;source=manta.PeriodicGDFTiles(**a)
            start=time.perf_counter();f=source.forward(x,a['budget']);rev=source.adjoint(c,a['budget']);cold=time.perf_counter()-start
            np.testing.assert_allclose(f.values,ef,atol=5e-12,rtol=3e-13);np.testing.assert_allclose(rev.values,ea,atol=5e-12,rtol=3e-13)
            samples=[]
            for _ in range(repeats):
                start=time.perf_counter();cpu=time.process_time();wf=source.forward(x,a['budget']);wr=source.adjoint(c,a['budget'])
                samples.append(dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu))
                np.testing.assert_array_equal(wf.values,f.values);np.testing.assert_array_equal(wr.values,rev.values)
            fcounts=counts(wf.receipt,'');rcounts=counts(wr.receipt,'')
            rows.append(dict(kind=label,order=a['options'].coupling_multipole_order,enabled=enabled,cold_seconds=cold,samples=samples,
                difference=max(float(np.max(abs(f.values-ef))),float(np.max(abs(rev.values-ea)))),peak_bytes=f.plan.modeled_peak_bytes,
                n_aux=source.plan.n_aux,n_orb=source.plan.n_orb,momenta=source.plan.momenta,
                global_bytes=source.plan.global_action_bytes,**{k:fcounts[k]+rcounts[k] for k in fcounts}))
    pins=('cpp/src/periodic_gdf_short_range.cpp','cpp/include/vibeqc/manta_periodic_fitting.hpp','cpp/src/manta_periodic_ri.cpp',
          'tests/test_manta_periodic_ri_global.py','tests/test_manta_periodic_ri_compressed.py','benchmarks/manta_periodic_ri_global.py')
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
        source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in pins},rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=3);p.add_argument('--repeats',type=int,default=3);p.add_argument('--worker',action='store_true');a=p.parse_args()
    if min(a.workers,a.repeats)<1:p.error('workers and repeats must be positive')
    if a.worker:print(json.dumps(worker(a.repeats),allow_nan=False));return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
    workers=[json.loads(subprocess.check_output([sys.executable,__file__,'--worker','--repeats',str(a.repeats)],env=env,text=True)) for _ in range(a.workers)]
    print(json.dumps(dict(workers=workers),indent=2,allow_nan=False))


if __name__=='__main__':main()
