"""Cost of full-Coulomb RI fitting certification and reuse, at a fixed source."""
import hashlib,json,os,resource,subprocess,sys,tempfile,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
import vibeqc as vq
from vibeqc import manta,_vibeqc_core as core
from tests.test_manta_periodic_ri_fitting_bounds import fitting_job

def run():
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-fitting-benchmark-') as temporary:
        job,a,raw,ri,ref,o,w=fitting_job(Path(temporary));result=vq.run_periodic_job(**job);assert result.converged
        d=np.stack([result.native.density/2]*2)
        for repeat in range(3):
            t=time.perf_counter();c=time.process_time();bound=manta.PeriodicRIFittingBoundSource(ri,ref,o,a['budget'])
            setup=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
            values={};states={}
            for checked in ([False,True] if repeat%2 else [True,False]):
                t=time.perf_counter();c=time.process_time()
                out=bound.build(d,w) if checked else ri.build(d,w,make_k=False)
                values['certified' if checked else 'plain']=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
                states[checked]=out.J
            np.testing.assert_array_equal(states[True],states[False])
            rows.append(dict(repeat=repeat,setup=setup,build=values,probes=bound.receipt.probes,
                j_bound=bound.receipt.j_error_bound,energy_bound=bound.receipt.energy_error_bound,modeled_peak_bytes=bound.plan.modeled_peak_bytes))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
