"""Whole-mesh physical range K and shifted two-k functional SCF diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(kernel, nk):
    root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
    import numpy as np
    from vibeqc import manta, _vibeqc_core as core
    from benchmarks.manta_periodic_ri_ks import native_binary
    from tests.test_manta_periodic_ri_range_fock import case, independent, densities, scf_calculation
    binary=native_binary(root,core); a=case(kernel,nk); h=densities(nk,spins=1,seed=311)[0]; d=densities(nk)
    _, expected_k, expected_f, expected_e=independent(a,h,d,.25)
    build_times=[]; response_times=[]
    for _ in range(5):
        start=time.perf_counter(); b=manta.PeriodicRIFockBuilder(**a); build_times.append(time.perf_counter()-start)
        start=time.perf_counter(); r=b.build(h,d,.25); response_times.append(time.perf_counter()-start)
    energy=np.array([r.receipt.one_electron_energy,r.receipt.coulomb_energy,r.receipt.exchange_energy,r.receipt.electronic_energy])
    errors=dict(K=float(np.max(abs(r.K-expected_k))),F=float(np.max(abs(r.values-expected_f))),energy=float(np.max(abs(energy-expected_e))))
    if max(errors.values())>8e-12: raise RuntimeError(errors)
    return dict(kernel=kernel,nk=nk,errors=errors,setup_median_s=float(np.median(build_times)),
        response_median_s=float(np.median(response_times)),channels=b.plan.channels,
        retained_source_bytes=b.plan.retained_source_bytes,peak_bytes=b.plan.modeled_peak_bytes,
        metric_shell_calls=b.info.metric_shell_calls,build_shell_calls=r.receipt.shell_calls,
        shell_bound=b.plan.shell_calls,source_identity=b.info.source_identity,
        core_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        scf=[scf_calculation(kernel,spins,whole_mesh=True) for spins in (1,2)] if nk==2 else [])


if __name__=='__main__':
    if len(sys.argv)>1: print(json.dumps(worker(sys.argv[1],int(sys.argv[2]))))
    else:
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        rows=[]
        for kernel in ('erfc','erf'):
            for nk in (2,4):
                p=subprocess.run([sys.executable,__file__,kernel,str(nk)],env=env,capture_output=True,text=True,check=True)
                rows.append(json.loads(p.stdout))
        print(json.dumps(rows,indent=2))
