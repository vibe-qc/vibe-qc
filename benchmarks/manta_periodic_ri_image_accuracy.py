"""Complete periodic RI image-certificate cost at fixed physical supports."""
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


def run(repeats=3):
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_one_electron_images import ri_image_job,image_accuracy_targets
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Require this checkout native runtime')
    rows=[]
    with tempfile.TemporaryDirectory(prefix='manta-ri-image-accuracy-') as directory:
        for cutoff in (3.,10.):
            control=None
            for radius in (None,2,3):
                samples=[];result=None
                for repeat in range(repeats+1):
                    job,_,_,_=ri_image_job(Path(directory)/str(cutoff)/str(radius)/str(repeat),automatic=True)
                    target=None if radius is None else image_accuracy_targets(1e-6 if cutoff==10 else .01,radius)
                    job['periodic_integrals']=dataclasses.replace(job['periodic_integrals'],pair_cutoff_bohr=cutoff,one_electron_accuracy=target)
                    # Avoid the exact auxiliary-radius tie that is correctly
                    # refused by the source's Gamma conjugacy qualification.
                    if cutoff==10.:
                        job['periodic_fitting']=dataclasses.replace(job['periodic_fitting'],auxiliary_cutoff_bohr=12.1)
                    job['integral_engine'].budget.memory_bytes=4*2**30
                    start=time.perf_counter();cpu=time.process_time();current=vq.run_periodic_job(**job)
                    sample=dict(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu)
                    assert current.converged
                    if result is None:cold=sample
                    else:
                        np.testing.assert_array_equal(current.native.density,result.native.density)
                        assert current.energy==result.energy
                        samples.append(sample)
                    result=current
                if radius is None:control=result
                else:
                    for name in ('density','fock','hcore','overlap'):
                        np.testing.assert_array_equal(getattr(result.native,name),getattr(control.native,name))
                    assert result.energy==control.energy
                rows.append(dict(pair_cutoff_bohr=cutoff,completion_cube_radius=radius,energy=result.energy,
                    iterations=result.iterations,peak_bytes=result.native.plan.modeled_peak_bytes,
                    image_accuracy=result.integral_provenance.get('one_electron_accuracy'),
                    coupling_accuracy=result.integral_provenance['fitting']['accuracy'],cold=cold,samples=samples))
    return dict(commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True)),
        native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<1:parser.error('--repeats must be positive')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    print(json.dumps(run(args.repeats),indent=2,allow_nan=False))
