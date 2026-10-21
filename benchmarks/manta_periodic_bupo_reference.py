"""Shared full-Coulomb reference preparation and RI certificate cost."""
import hashlib,json,os,resource,subprocess,sys,tempfile,time
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
from vibeqc import manta,_vibeqc_core as core
from tests.test_manta_periodic_ri_fitting_bounds import fitting_job,prepared_fitting_reference

def run():
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-reference-benchmark-') as temporary:
        _,args,a,ri,ref,options,w=fitting_job(Path(temporary))
        baseline=manta.PeriodicRIFittingBoundSource(ri,ref,options,args['budget'])
        for repeat in range(3):
            t=time.perf_counter();c=time.process_time();prepared=prepared_fitting_reference(args)
            setup=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
            assert prepared.source.source_identity==ref.source_identity
            t=time.perf_counter();c=time.process_time()
            bound=manta.PeriodicRIFittingBoundSource(ri,prepared.source,options,prepared.remaining_budget)
            certificate=dict(wall=time.perf_counter()-t,cpu=time.process_time()-c)
            assert bound.receipt.certificate_identity==baseline.receipt.certificate_identity
            np.testing.assert_array_equal(bound.j_error_bounds,baseline.j_error_bounds)
            rows.append(dict(repeat=repeat,preparation=setup,certificate=certificate,
                source_identity=prepared.source.source_identity,certificate_identity=bound.receipt.certificate_identity,
                source_live_bytes=prepared.plan.source_live_bytes,joint_peak_bytes=bound.plan.modeled_peak_bytes,
                source_shell_calls=prepared.plan.shell_calls,source_kernel_terms=prepared.plan.kernel_terms))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
if __name__=='__main__':print(json.dumps(run(),indent=2,allow_nan=False))
