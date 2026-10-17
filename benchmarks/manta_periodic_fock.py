#!/usr/bin/env python3
"""Finite periodic spin Fock benchmark, including snapshots and energy; JSON to stdout."""
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
import time


def worker(case,provider,batch,repeats):
    began=time.perf_counter()
    import numpy as np
    import vibeqc as vq
    from vibeqc import manta,_vibeqc_core as core
    from vibeqc.bipole_fock_ewald import _contract_K_long_range_channel_linear
    root=Path(__file__).resolve().parents[1]; native=Path(core.__file__).resolve()
    if not native.is_relative_to(root) or native.stat().st_mtime<=max(p.stat().st_mtime for p in (root/'cpp').rglob('*') if p.suffix in ('.cpp','.hpp','.h')):
        raise RuntimeError('Use the freshly rebuilt editable core from this checkout')
    start=time.perf_counter()
    lattice=np.array([[2.5,.2,.1],[0.,2.7,.3],[.2,.1,2.9]])
    positions=[[.1,-.2,.3],[.4,.2,-.1],[-.2,.5,.7]]
    molecule=vq.Molecule([vq.Atom(2,p) for p in positions],0,1); shells=[]
    for i,p in enumerate(positions):
        s=core.ShellInfo(); s.atom_index=i; s.origin=p; s.l=i; s.pure=True
        s.exponents=[.7+.1*i,1.6+.2*i]; s.coefficients=[.8,-.18]; shells.append(s)
    basis=vq.BasisSet(molecule,shells,'manta-fock-benchmark',False)
    labels=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    cells=labels@lattice.T; base=labels@(2*np.pi*np.linalg.inv(lattice)); g=base[1:].copy()
    k=np.array([[0.,0.,0.],[.2,-.1,.07],[-.17,.12,.14]])
    if case=='gamma': k=k[:1].copy()
    weights=np.ones(1) if case=='gamma' else np.array([.2,.3,.5])
    chunks=[base+t-s for t in k for s in k]; chunks=[q[np.any(q!=0.,axis=1)] for q in chunks]
    q=np.ascontiguousarray(np.concatenate(chunks)); offsets=np.r_[0,np.cumsum([len(v) for v in chunks])].astype(np.int64)
    rng=np.random.default_rng(541); x=rng.normal(size=(2,len(k),basis.nbasis,basis.nbasis))+1j*rng.normal(size=(2,len(k),basis.nbasis,basis.nbasis))
    density=np.ascontiguousarray(x+x.conj().transpose(0,1,3,2))
    hcore=np.stack([np.diag(np.linspace(-1.,1.,basis.nbasis))]*len(k)).astype(complex)
    scales=np.concatenate([np.full(2*s.l+1,np.sqrt(4*np.pi/(2*s.l+1)) if s.l else 1.) for s in basis.shells()])
    volume=abs(np.linalg.det(lattice)); fixture=time.perf_counter()-start
    def build(variant):
        if variant=='manta':
            r=manta.compute_periodic_fock(basis,lattice,g,q,offsets,cells,k,weights,hcore,density,batch)
            return (r.values,r.hartree_values,r.exchange_values,
                np.array([r.receipt.one_electron_energy,r.receipt.coulomb_energy,r.receipt.exchange_energy]),r)
        raw=core.ao_pair_fourier_transform_bloch_multi_cxx(basis,g,cells,-k,0.,None,-1.)
        fourier=raw*scales[None,:,None,None]*scales[None,None,:,None]
        total=density.sum(axis=0); rho=np.einsum('k,kmn,kmng->g',weights,total,fourier)
        j=np.einsum('g,g,kmng->kmn',4*np.pi/volume/np.sum(g*g,axis=1),rho,fourier.conj())
        exchange=np.zeros_like(density)
        for t in range(len(k)):
            for s in range(len(k)):
                pair=t*len(k)+s; wave=q[offsets[pair]:offsets[pair+1]]
                raw=core.ao_pair_fourier_transform_bloch_multi_cxx(basis,wave,cells,-k[s:s+1],0.,None,-1.)[0]
                b=raw*scales[:,None,None]*scales[None,:,None]
                factor=weights[s]*4*np.pi/volume/np.sum(wave*wave,axis=1)
                for spin in range(2): exchange[spin,t]+=_contract_K_long_range_channel_linear(factor,b,density[spin,s])
        energy=np.array([np.einsum('k,kmn,knm->',weights,total,hcore).real,
            .5*np.einsum('k,kmn,knm->',weights,total,j).real,-.5*np.einsum('k,skmn,sknm->',weights,density,exchange).real])
        return hcore[None]+j[None]-exchange,j,exchange,energy,None
    build(provider); wall=[]; cpu=[]
    for _ in range(repeats):
        start,process=time.perf_counter(),time.process_time(); actual=build(provider)
        wall.append(time.perf_counter()-start); cpu.append(time.process_time()-process)
    start=time.perf_counter(); reference=build('reference')
    errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual[:4],reference[:4])]
    if max(errors)>3e-9: raise RuntimeError('Finite spin Fock benchmark numerical gate failed')
    validation=time.perf_counter()-start; r=actual[-1]; work=None
    if r is not None:
        work=dict(shell_calls=r.hartree_receipt.shell_calls+r.exchange_receipt.shell_calls,
            primitive_pairs=r.hartree_receipt.primitive_pairs+r.exchange_receipt.primitive_pairs,
            vector_primitive_products=r.hartree_receipt.vector_primitive_products+r.exchange_receipt.vector_primitive_products,
            exchange_matrix_products=r.exchange_receipt.matrix_products,modeled_peak_bytes=r.plan.modeled_peak_bytes)
    return dict(case=case,provider=provider,batch=batch,repeats=repeats,n_basis=basis.nbasis,n_k=len(k),
        median_wall_s=statistics.median(wall),min_wall_s=min(wall),max_wall_s=max(wall),median_cpu_s=statistics.median(cpu),
        fixture_wall_s=fixture,validation_wall_s=validation,total_worker_wall_s=time.perf_counter()-began,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
        max_fock_error=errors[0],max_hartree_error=errors[1],max_exchange_error=errors[2],max_component_energy_error=errors[3],
        work=work,core_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        platform=platform.system(),architecture=platform.machine(),python=platform.python_version(),numpy=np.__version__,threads=1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['reference','manta']); parser.add_argument('--case',default='multi-k',choices=['gamma','multi-k'])
    parser.add_argument('--batch',type=int,default=32); parser.add_argument('--repeats',type=int,default=5); args=parser.parse_args()
    if min(args.batch,args.repeats)<1: parser.error('batch and repeats must be positive')
    if args.worker:
        print(json.dumps(worker(args.case,args.worker,args.batch,args.repeats))); return
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'); rows=[]
    for case in ('gamma','multi-k'):
        for provider,batch in [('reference',32),('manta',4),('manta',32)]:
            rows.append(json.loads(subprocess.check_output([sys.executable,__file__,'--worker',provider,'--case',case,
                '--batch',str(batch),'--repeats',str(args.repeats)],env=env,text=True)))
    print(json.dumps(dict(schema_version=1,rows=rows),indent=2))

if __name__=='__main__': main()
