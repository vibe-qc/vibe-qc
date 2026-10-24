"""Uniform occupied-space exchange certificate costs and physical-domain limits."""
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


def run(repeats=3,compact_only=False):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_exchange_bounds import occupied_envelope_case,ri_exchange_envelope_job,compact_occupied_envelope_case
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[]
    def measure(source,s,populations,scf):
        values=[];result=None
        for index in range(repeats+1):
            start=time.perf_counter();current=source.bound_occupied_subspaces(s,*populations,scf)
            elapsed=time.perf_counter()-start
            if result is None:cold=elapsed
            else:
                assert current.certificate_identity==result.certificate_identity
                values.append(elapsed)
            result=current
        reuse=[]
        for index in range(repeats):
            start=time.perf_counter();current=source.bound_density_envelope(result.density)
            reuse.append(time.perf_counter()-start)
            assert current.certificate_identity==result.certificate_identity
        return dict(maximum_k_error=result.maximum_error,energy_error_bound=result.energy_error_bound,
            entry_energy_error_bound=result.entry_energy_error_bound,l1_energy_error_bound=result.l1_energy_error_bound,
            analysis_peak_bytes=result.plan.modeled_peak_bytes,reuse_peak_bytes=current.plan.modeled_peak_bytes,
            cold_seconds=cold,analysis_seconds=values,reuse_seconds=reuse,
            pair_source_identity=result.pair_source_identity,source_identity=result.source_identity,
            certificate_identity=result.certificate_identity)
    for image_radius,reciprocal_radius in (() if compact_only else ((0,2),(1,2),(1,8))):
        start=time.perf_counter();c,q,o,source,s,scf=occupied_envelope_case(0,image_radius,reciprocal_radius)
        prepare=time.perf_counter()-start
        rows.append(dict(kind='component',image_radius=image_radius,reciprocal_radius=reciprocal_radius,
            preparation_seconds=prepare,n_basis=s.shape[1],momenta=len(s),**measure(source,s,(1,1),scf)))
    if not compact_only:
        with tempfile.TemporaryDirectory(prefix='manta-exchange-envelope-') as directory:
            job,a,k,domains,source=ri_exchange_envelope_job(Path(directory))
            start=time.perf_counter();state=vq.run_periodic_job(**job);elapsed=time.perf_counter()-start
            assert state.converged
            ne=a['system'].n_electrons();nb=(ne-a['system'].multiplicity+1)//2
            rows.append(dict(kind='complete-ri-rhf',job_seconds=elapsed,energy=state.energy,
                n_basis=a['basis'].nbasis,momenta=len(k),iterations=state.iterations,
                **measure(source,state.native.overlap,(ne-nb,nb),a['options'].scf)))
    start=time.perf_counter();a,c,source,s=compact_occupied_envelope_case();prepare=time.perf_counter()-start
    row=dict(kind='compact-component',image_radius=1,reciprocal_radius=30,preparation_seconds=prepare,
        n_basis=s.shape[1],momenta=len(s),**measure(source,s,(1,1),a['options'].scf))
    assert row['maximum_k_error']<1e-8 and row['energy_error_bound']<1e-8
    rows.append(row)
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--compact-only',action='store_true',help='Run only the additional compact-source case')
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    print(json.dumps(run(args.repeats,args.compact_only),indent=2,allow_nan=False))
