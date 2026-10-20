"""Streamed fixed-projector periodic RI components: parity, memory and timings."""
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
    from tests.test_manta_periodic_ri_q import densities,reference
    from tests.test_aux_eri_sr_lattice_enumeration import _weighted_gradient_fixture
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(
        p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Require fresh own-checkout native build')
    gamma=profile=='s-gamma';a=case(gamma);o=manta.RIOptions();o.mode_block=1
    if profile=='d-multi':
        import itertools
        a['system'],a['orbital'],a['auxiliary']=_weighted_gradient_fixture(2,orbital_angular=1)
        a['vectors']=np.array(list(itertools.product(range(-1,2),repeat=3)))@a['system'].reciprocal_lattice().T+a['q']
        a['options'].pair_cutoff_bohr=4.3;a['options'].auxiliary_cutoff_bohr=5.1;o.mode_block=4
    start=time.perf_counter();source=manta.PeriodicGDFTiles(**a);b=manta.PeriodicRIQBuilder(source,2,o,a['budget'])
    construction=time.perf_counter()-start
    d=densities(n=a['orbital'].nbasis);w=np.array([.23,.77])
    start=time.perf_counter();r=b.build(d,w,make_j=gamma);j,k=r.J,r.K;cold=time.perf_counter()-start
    wall=[]
    for _ in range(repeats):
        start=time.perf_counter();r=b.build(d,w,make_j=gamma);j,k=r.J,r.K;wall.append(time.perf_counter()-start)
    start=time.perf_counter();m,t,_=legacy(a)
    metric=np.tril(m)+np.tril(m,-1).conj().T;metric[np.diag_indices(len(m))]=metric.diagonal().real
    e,u=np.linalg.eigh(metric);keep=e>max(o.absolute_cutoff,o.relative_cutoff*max(0.,e[-1]))
    fitted=np.einsum('pl,kpmn->klmn',(u[:,keep]/np.sqrt(e[keep])).conj(),t)
    reference_construction=time.perf_counter()-start
    def response():
        rho=np.einsum('k,klmn,kmn->l',w,fitted.conj(),d.sum(axis=0)) if gamma else None
        j=np.einsum('klmn,l->kmn',fitted,rho) if gamma else None
        k=np.einsum('k,klmp,skpq,klnq->skmn',w,fitted,d,fitted.conj(),optimize=True)
        return j,k
    reference_wall=[]
    for _ in range(repeats):
        start=time.perf_counter();rj,rk=response();reference_wall.append(time.perf_counter()-start)
    errors=dict(K=float(np.max(abs(k-rk))))
    if gamma:errors['J']=float(np.max(abs(j-rj)))
    if max(errors.values())>3e-11:raise RuntimeError('Matched finite-source J/K parity failed')
    independent_errors=None
    if profile!='d-multi':
        ij,ik,ie,_,_=reference(*independent(a),d,w)
        independent_errors=dict(K=float(np.max(abs(k-ik))))
        if gamma:independent_errors.update(J=float(np.max(abs(j-ij))),energy=abs(r.receipt.coulomb_energy-ie))
        if max(independent_errors.values())>3e-11:raise RuntimeError('Independent Gaussian J/K parity failed')
    return dict(profile=profile,repeats=repeats,construction_s=construction,cold_s=cold,
        median_s=statistics.median(wall),min_s=min(wall),max_s=max(wall),reference_construction_s=reference_construction,
        reference_median_s=statistics.median(reference_wall),reference_min_s=min(reference_wall),reference_max_s=max(reference_wall),
        n_orb=b.plan.n_basis,n_aux=b.plan.n_auxiliary,momenta=b.plan.momenta,rank=b.metric_info.rank,mode_block=b.plan.mode_block,
        errors=errors,independent_errors=independent_errors,modeled_peak_bytes=b.plan.modeled_peak_bytes,
        source_peak_bytes=source.plan.modeled_peak_bytes,maximum_tile_elements=source.plan.maximum_tile_elements,
        shell_bound=b.plan.shell_calls,primitive_bound=b.plan.primitive_products,scalar_bound=b.plan.scalar_terms,kernel_bound=b.plan.kernel_terms,
        metric_shell_calls=b.metric_info.shell_calls,metric_primitive_products=b.metric_info.primitive_products,
        coupling_passes=r.receipt.coupling_passes,shell_calls=r.receipt.shell_calls,primitive_products=r.receipt.primitive_products,
        metric_rhs_norm=r.receipt.metric_rhs_norm,metric_residual_norm=r.receipt.metric_residual_norm,
        hermiticity_residual=r.receipt.hermiticity_residual,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest())


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
