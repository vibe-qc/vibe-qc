#!/usr/bin/env python3
"""Full-image/full-reciprocal exchange certificates and admitted periodic HF."""
import argparse,hashlib,json,os,platform,resource,statistics,subprocess,sys,time
from pathlib import Path

CASES=('sparse-sp','certified-rhf','certified-empty-beta')


def worker(name,repeats):
    import numpy as np
    from vibeqc import manta,_vibeqc_core as core
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    from tests.test_manta_periodic_exchange_bounds import setup,hf_case
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root):raise RuntimeError('Use this checkout native runtime')
    hf_args=None
    if name=='sparse-sp':
        c,q,o,source,d=setup(angular=1,pure=False,cutoff=.3);pair_source=c['source'];radius=2;budget=manta.ResourceBudget()
    else:
        hf_args,c=hf_case(spin='unrestricted' if name=='certified-empty-beta' else 'restricted',z=1 if name=='certified-empty-beta' else 2,
                          shifted=name=='certified-empty-beta',q0='BvK-ewald')
        hf=core._manta.run_periodic_bupo_hf(**hf_args);assert hf.converged and hf.receipt.exchange_accuracy_passed
        d=hf.density;q=hf_args['exchange_vectors'];o=hf_args['transfer_offsets'];radius=hf_args['options'].exchange_bound_cube_radius
        pair_source=c['pairs'];budget=hf_args['budget']
    construction=[];build=[];scf=[]
    for _ in range(repeats):
        start=time.perf_counter();source=manta.PeriodicExchangeBoundSource(pair_source,q,o,radius,budget);construction.append(time.perf_counter()-start)
        start=time.perf_counter();out=source.build(d,budget);build.append(time.perf_counter()-start)
        if hf_args is not None:
            start=time.perf_counter();hf=core._manta.run_periodic_bupo_hf(**hf_args);scf.append(time.perf_counter()-start)
            assert hf.converged and hf.receipt.exchange_accuracy_passed
    return dict(case=name,repeats=repeats,n_basis=source.plan.n_basis,nk=source.plan.momenta,spins=len(d),vectors=len(q),
                construction_s=statistics.median(construction),build_s=statistics.median(build),hf_s=statistics.median(scf) if scf else None,
                modeled_peak_bytes=max(source.plan.modeled_peak_bytes,out.plan.modeled_peak_bytes),
                hf_modeled_peak_bytes=hf.plan.modeled_peak_bytes if scf else None,
                max_unscaled_k_bound=out.receipt.maximum_error,spin_energy_bounds=out.receipt.energy_error_bounds,
                maximum_observed_hf_matrix_error=hf.receipt.maximum_observed_exchange_matrix_error if scf else None,
                maximum_observed_hf_energy_error=hf.receipt.maximum_observed_exchange_energy_error if scf else None,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                source_identity=source.receipt.source_identity,native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                python=platform.python_version(),numpy=np.__version__,platform=platform.system(),architecture=platform.machine(),threads=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=CASES)
    parser.add_argument('--repeats',type=int,default=3);args=parser.parse_args()
    if not 3<=args.repeats<=100:parser.error('Use three to one hundred repeats')
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
    if args.worker:print(json.dumps(worker(args.worker,args.repeats)));return
    rows=[]
    for case in CASES:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',case,'--repeats',str(args.repeats)],check=True,text=True,capture_output=True)
        rows.append(json.loads(result.stdout))
    print(json.dumps(dict(benchmark='manta-periodic-exchange-bounds',results=rows),indent=2))


if __name__=='__main__':main()
