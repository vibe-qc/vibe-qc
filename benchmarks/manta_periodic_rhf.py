#!/usr/bin/env python3
"""Complete finite-source RHF benchmark with an independent SCF oracle.

Requires the development test dependencies. Numerical fixtures and the
quadrature/libint oracle are shared with the executable scientific tests.
Native build times include all integral generation and terminal verification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time


def worker(case,batch,repeats,exchange_q0,reciprocal_energy_cutoff=None,job=False):
    began=time.perf_counter(); root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root))
    import numpy as np
    from scipy.linalg import eigh
    from vibeqc import _vibeqc_core as core
    from tests.test_manta_periodic_rhf import _case,_components,_nuclear_reference
    native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Use a freshly rebuilt editable core from this checkout')
    start=time.perf_counter()
    mesh=(1,1,1) if case=='gamma' else (2,1,1); shift=(1,0,0) if case=='shifted' else (0,0,0)
    args,k=_case(mesh,shift); args['options'].reciprocal_batch=batch; args['options'].exchange_q0=exchange_q0
    if exchange_q0=='BvK-ewald': args['options'].scf.maximum_iterations=100
    cutoff_args=None; domain_wall=0.
    if reciprocal_energy_cutoff is not None:
        # Admission bounds include enclosing boxes; the timed run generates
        # the actual spheres on every invocation, including the warmup.
        args['budget'].memory_bytes=2**30; args['budget'].shell_calls=10**9
        args['budget'].scalar_terms=10**17; args['budget'].kernel_terms=10**18
        cutoff_args={key:args[key] for key in ('basis','system','mesh','options','budget')}
        cutoff_args['reciprocal_energy_cutoff']=reciprocal_energy_cutoff
        begin_domain=time.perf_counter(); domains=core._manta.build_periodic_rhf_domains(**cutoff_args)
        domain_wall=time.perf_counter()-begin_domain
        for key in ('reciprocal_vectors','exchange_vectors','transfer_offsets','translations'):
            args[key]=getattr(domains,key)
    fixture=time.perf_counter()-start
    job_directory=tempfile.TemporaryDirectory(prefix='manta-periodic-job-benchmark-') if job else None
    def build():
        if job:
            import vibeqc as vq
            from vibeqc import manta
            o=args['options']
            r=vq.run_periodic_job(args['system'],args['basis'],kpoints=args['mesh'],
                integral_engine=manta.IntegralEngineOptions(backend='manta',budget=args['budget']),
                periodic_integrals=manta.PeriodicIntegralOptions(reciprocal_energy_cutoff,
                    o.pair_cutoff_bohr,o.nuclear_cutoff_bohr,alpha=o.alpha,reciprocal_batch=batch),
                max_iter=o.scf.maximum_iterations,conv_tol_energy=o.scf.energy_change_tolerance,
                conv_tol_grad=o.scf.commutator_tolerance,use_diis=False,initial_guess='HCORE',
                exchange_exxdiv='ewald' if exchange_q0=='BvK-ewald' else 'none',
                record_hostname=False,output=Path(job_directory.name)/'job').native
        else:
            r=(core._manta.run_periodic_rhf(**args) if cutoff_args is None
               else core._manta.run_periodic_rhf_from_cutoffs(**cutoff_args))
        if not r.converged: raise RuntimeError('Finite RHF benchmark did not converge and confirm')
        return r,(r.density,r.fock,r.coefficients,r.orbital_energies)
    build(); wall=[]; cpu=[]
    for _ in range(repeats):
        start,process=time.perf_counter(),time.process_time(); r,arrays=build()
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    start=time.perf_counter(); s,h,uncorrected=_components(args,k)
    xi=0.
    if exchange_q0=='BvK-ewald':
        from vibeqc.bipole_fock_ewald import probe_charge_madelung_supercell
        xi=probe_charge_madelung_supercell(args['system'],mesh,precision=1e-14)
    def response(d): return uncorrected(d)-.5*xi*(s@d@s)
    def density(f):
        coefficients=[eigh(fp,sp)[1][:,:1] for fp,sp in zip(f,s)]
        return np.array([2*c@c.conj().T for c in coefficients])
    d=density(h)
    for iteration in range(100):
        updated=density(response(d))
        if np.max(abs(updated-d))<1e-12: break
        d=updated
    else: raise RuntimeError('Independent finite RHF oracle did not converge')
    nuclear,_=_nuclear_reference(args); energy=sum(nuclear)+.5/len(k)*np.einsum('kij,kji->',d,h+response(d)).real
    energy_error=abs(r.receipt.energy-energy); density_error=float(np.max(abs(arrays[0]-d)))
    fock_error=float(np.max(abs(arrays[1]-response(arrays[0]))))
    if energy_error>3e-10 or density_error>3e-9 or fock_error>3e-10: raise RuntimeError('Independent finite RHF numerical gate failed')
    return dict(case=case,batch=batch,repeats=repeats,n_basis=args['basis'].nbasis,n_k=len(k),exchange_q0=exchange_q0,
        timing_scope='periodic-job-with-default-artifacts' if job else 'native-rhf-and-owned-exports',
        madelung=r.receipt.madelung,madelung_error=abs(xi-r.receipt.madelung),
        domain_selection=r.receipt.domain_selection,reciprocal_energy_cutoff=reciprocal_energy_cutoff,
        domain_generation_wall_s=domain_wall,
        energy=r.receipt.energy,iterations=r.solver_snapshot.evaluated_iteration,fock_calls=r.receipt.fock_calls,
        median_wall_s=statistics.median(wall),min_wall_s=min(wall),max_wall_s=max(wall),median_cpu_s=statistics.median(cpu),
        fixture_wall_s=fixture,validation_wall_s=time.perf_counter()-start,total_worker_wall_s=time.perf_counter()-began,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        max_energy_error=energy_error,max_density_error=density_error,max_fock_error=fock_error,
        work=dict(shell_calls=r.receipt.shell_calls,primitive_pairs=r.receipt.primitive_pairs,nuclear_pair_images=r.receipt.nuclear_pair_images,
                  history_capacity=r.plan.history_capacity,history_live_bytes=r.plan.history_live_bytes,
                  history_steps=len(r.scf_history),
                  modeled_peak_bytes=r.plan.modeled_peak_bytes,planned_kernel_terms=r.plan.kernel_terms,
                  domain_live_bytes=r.plan.domain_live_bytes,domain_kernel_terms=r.plan.domain_kernel_terms,
                  domain_translations=r.receipt.domain_translations,domain_hartree_vectors=r.receipt.domain_hartree_vectors,
                  domain_exchange_vectors=r.receipt.domain_exchange_vectors,
                  madelung_real_vectors=r.receipt.madelung_real_vectors,madelung_reciprocal_vectors=r.receipt.madelung_reciprocal_vectors),
        confirmation_fock_error=r.receipt.confirmation_fock_error,confirmation_energy_error=r.receipt.confirmation_energy_error,
        core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),platform=platform.system(),architecture=platform.machine(),
        python=platform.python_version(),numpy=np.__version__,threads=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--worker',action='store_true')
    parser.add_argument('--case',choices=['gamma','multi-k','shifted'],default='multi-k'); parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--job',action='store_true',help='Time run_periodic_job including standard output, citations and QVF')
    parser.add_argument('--reciprocal-energy-cutoff',type=float,
        help='Generate shared physical-pair/reciprocal spheres inside every timed native run')
    parser.add_argument('--exchange-q0',choices=['strict-zero-mode','BvK-ewald'],default='strict-zero-mode'); args=parser.parse_args()
    if min(args.batch,args.repeats)<1: parser.error('batch and repeats must be positive')
    if args.reciprocal_energy_cutoff is not None and not (0 < args.reciprocal_energy_cutoff < float('inf')):
        parser.error('reciprocal energy cutoff must be finite positive')
    if args.job and args.reciprocal_energy_cutoff is None:
        parser.error('--job requires --reciprocal-energy-cutoff')
    if args.worker: print(json.dumps(worker(args.case,args.batch,args.repeats,args.exchange_q0,args.reciprocal_energy_cutoff,args.job))); return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'); rows=[]
    for case in ('gamma','multi-k','shifted'):
        for batch in (4,32):
            command=[sys.executable,__file__,'--worker','--case',case,'--batch',str(batch),
                     '--repeats',str(args.repeats),'--exchange-q0',args.exchange_q0]
            if args.reciprocal_energy_cutoff is not None:
                command += ['--reciprocal-energy-cutoff',str(args.reciprocal_energy_cutoff)]
            if args.job:
                command += ['--job']
            rows.append(json.loads(subprocess.check_output(command,env=env,text=True)))
    print(json.dumps(dict(schema_version=1,rows=rows),indent=2))

if __name__=='__main__': main()
