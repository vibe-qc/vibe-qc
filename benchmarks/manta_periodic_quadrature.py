"""Bounded periodic quadrature costs and separate XC energy/potential errors."""
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
    from tests.test_manta_periodic_quadrature import xc_case,tensor_reference
    from tests.test_manta_periodic_semilocal_xc import reference
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require fresh own-checkout editable native build')
    name,spins,stratmann={'lda-rks':('lda',1,False),'pbe-uks':('pbe',2,False),
        'pbe-uks-stratmann':('pbe',2,True)}[profile]
    s,c=xc_case(name,spins);o=manta.PeriodicQuadratureOptions();o.radial_points=40;o.lebedev_order=59
    o.partition_radius_bohr=8.;o.stratmann=stratmann
    start=time.perf_counter();manta.build_periodic_quadrature(s,o,c['budget']);cold=time.perf_counter()-start
    wall=[];cpu=[]
    for _ in range(repeats):
        grid=None;start=time.perf_counter();t=time.process_time();grid=manta.build_periodic_quadrature(s,o,c['budget'])
        wall.append(time.perf_counter()-start);cpu.append(time.process_time()-t)
    c.update(points=grid.points,quadrature_weights=grid.weights)
    start=time.perf_counter();xc=manta.compute_periodic_semilocal_xc(**c);xc_s=time.perf_counter()-start
    e=0.;v=np.zeros_like(c['densities']);n=np.zeros(spins)
    for start in range(0,len(c['points']),2048):
        q=dict(c,points=c['points'][start:start+2048],quadrature_weights=c['quadrature_weights'][start:start+2048])
        ei,vi,ni=reference(q,c['functional']);e+=ei;v+=vi;n+=ni
    source_errors=dict(energy=abs(e-xc.receipt.energy),potential=float(np.max(abs(v-xc.values))),
        populations=float(np.max(abs(n-xc.receipt.electrons_grid))))
    ref=tensor_reference(c,64);fine=tensor_reference(c,80)
    tensor_errors=dict(energy=abs(ref[0]-fine[0]),potential=float(np.max(abs(ref[1]-fine[1]))))
    errors=dict(energy=abs(xc.receipt.energy-fine[0]),potential=float(np.max(abs(xc.values-fine[1]))),
        populations=float(np.max(abs(np.array(xc.receipt.electrons_grid)-fine[2]))))
    if max(source_errors.values())>3e-12 or max(tensor_errors.values())>3e-10:
        raise RuntimeError('Independent quadrature source/reference acceptance failed')
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return dict(profile=profile,radial=o.radial_points,lebedev_order=o.lebedev_order,partition_radius=o.partition_radius_bohr,
        cold_s=cold,repeats=repeats,median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),
        cpu_median_s=statistics.median(cpu),xc_s=xc_s,points=grid.plan.grid_points,threads=grid.plan.threads,
        modeled_peak_bytes=grid.plan.modeled_peak_bytes,output_bytes=grid.plan.output_bytes,
        workspace_bytes=grid.plan.workspace_bytes,scalar_terms=grid.plan.scalar_terms,kernel_terms=grid.plan.kernel_terms,
        atlas_candidates=grid.plan.atlas_candidates,neighbor_candidates=grid.plan.neighbor_candidates,
        volume=grid.receipt.cell_volume,weight_sum=grid.receipt.weight_sum,source_errors=source_errors,
        tensor_reference_errors=tensor_errors,quadrature_errors=errors,peak_process_rss_bytes=rss,
        core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


def refinement():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import manta
    from tests.test_manta_periodic_quadrature import xc_case,tensor_reference
    system,c=xc_case();ref=tensor_reference(c,80);rows=[]
    for radial,angular,radius in ((20,29,8.),(40,59,8.),(80,95,8.),(80,95,16.),(120,95,16.),(120,95,20.)):
        o=manta.PeriodicQuadratureOptions();o.radial_points=radial;o.lebedev_order=angular;o.partition_radius_bohr=radius
        start=time.perf_counter();g=manta.build_periodic_quadrature(system,o,c['budget'])
        c.update(points=g.points,quadrature_weights=g.weights);r=manta.compute_periodic_semilocal_xc(**c)
        rows.append(dict(radial=radial,lebedev_order=angular,radius=radius,points=g.plan.grid_points,
            seconds=time.perf_counter()-start,energy_error=abs(r.receipt.energy-ref[0]),
            potential_error=float(np.max(abs(r.values-ref[1])))))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=('lda-rks','pbe-uks','pbe-uks-stratmann'))
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--refinement',action='store_true');a=p.parse_args()
    if a.refinement:print(json.dumps(refinement(),indent=2));return
    if a.repeats<3:p.error('At least three repeats required')
    if a.worker:print(json.dumps(worker(a.worker,a.repeats)));return
    rows=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for profile in ('lda-rks','pbe-uks','pbe-uks-stratmann'):
        r=subprocess.run([sys.executable,__file__,'--worker',profile,'--repeats',str(a.repeats)],
            check=True,capture_output=True,text=True,env=env);rows.append(json.loads(r.stdout))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
