"""Bounded occupied-preparation ladder; synthetic algebra, not physical acceptance.

Run as python -m benchmarks.chi_occupied_preparation in the development env.
All source operators/projectors are created before measurement. The source
has a nonorthogonal metric and exactly degenerate translated seed spectra.
"""
import json
from time import perf_counter
import tracemalloc
from types import SimpleNamespace
from dataclasses import replace
import argparse
import cProfile
import pstats

import numpy as np

from vibeqc.dlpno.source import CorrelationSource, SourceConventions
from vibeqc.dlpno.translation_operators import CyclicAOOperator
from vibeqc.dlpno.translation_projectors import BlochOccupiedProjectors, ProjectorReference
from vibeqc.dlpno.translation_selection import select_translation_occupied
from vibeqc.dlpno import translation_preparation
from vibeqc.periodic.chi.local_mp2 import _real_occupied
from vibeqc.dlpno.translation_builder import _record


def model(n):
    mesh=(n,1,1); b=16; occupied=6; frozen=2
    rng=np.random.default_rng(508)
    raw=rng.normal(size=(b,b))
    s=np.eye(b)+.03*raw@raw.T
    w,v=np.linalg.eigh(s)
    c=(v/np.sqrt(w))@v.T@np.linalg.qr(rng.normal(size=(b,b)))[0]
    f=s@c@np.diag(np.linspace(-2.,2.,b))@c.T@s
    def op(matrix):
        blocks=np.zeros((n,b,b));blocks[0]=matrix
        return CyclicAOOperator(blocks,mesh)
    S,F=op(s),op(f)
    qa=op(c[:,frozen:occupied]@c[:,frozen:occupied].T@s)
    qf=op(c[:,:frozen]@c[:,:frozen].T@s)
    projectors=BlochOccupiedProjectors(qa,qf,n*occupied,n*frozen,S.identity,F.identity,0.,0.,0.)
    reference=ProjectorReference(projectors,F,S,np.tile(np.linspace(-2.,2.,b),n),-2.*n)
    source=CorrelationSource(kind='finite-selector-model',reference_kind='rhf',
        molecule=SimpleNamespace(multiplicity=1,n_electrons=lambda:2*n*occupied),
        basis=SimpleNamespace(nbasis=n*b),reference=reference,factors=SimpleNamespace(mo_transform=lambda *_: None),overlap=S,
        conventions=SourceConventions(kmesh=mesh,bloch_normalisation='real-bvk-supercell'))
    return source,mesh,n*frozen,c,s


def prepare_source(source, mesh, bloch, overlap, phase, compact):
    """Both original independent partitions, then their immutable source binding."""
    audits=[]; partitions=[]
    for label,part in (('active',slice(2,6)),('frozen',slice(0,2))):
        value,record=(translation_preparation.real_occupied_projector(
            bloch[:,:,part],overlap,phase,mesh,label,1e-10) if compact else
            _real_occupied(bloch[:,:,part],overlap,phase,label))
        partitions.append(value); audits.append(record)
    active,frozen=partitions
    if compact:
        orbitals=BlochOccupiedProjectors(active,frozen,6*mesh[0],2*mesh[0],
            source.overlap.identity,source.reference.fock.identity,
            max(r['orthogonality'] for r in audits),max(r['projector_error'] for r in audits),
            max(r.get('imaginary',0.) for r in audits))
        reference=replace(source.reference,occupied_projectors=orbitals)
    else:
        reference=SimpleNamespace(mo_coeffs=np.column_stack((frozen,active)),
            fock=source.reference.fock,energy=source.reference.energy,converged=True,
            reference_kind='rhf')
    return replace(source,reference=reference),partitions,audits


def measure(call, profile=False):
    profiler=cProfile.Profile() if profile else None
    tracemalloc.start(); start=perf_counter()
    if profiler: profiler.enable()
    result=call()
    if profiler: profiler.disable()
    elapsed=perf_counter()-start
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    record=dict(elapsed_s=elapsed,traced_peak_bytes=peak)
    if profiler:
        stats=pstats.Stats(profiler)
        names={'realify','apply','occupied_projector_rows','occupied_orthogonality_error',
               'metric_defect','projector_defect','cyclic_projected_frame','_cyclic_span_audit',
               'admit_translation_source','gram','choose','phase_inverse','phase_forward'}
        record['profile']=[dict(function=k[2],calls=v[1],self_s=v[2],cumulative_s=v[3])
                           for k,v in stats.stats.items() if k[2] in names]
    return result,record


def main():
    from vibeqc.output import OutputChannel,write
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--cells',type=int,nargs='+',default=[1,2,4,8])
    parser.add_argument('--baseline',choices=['ac2929cb','49317c5d'],help='load prior preparation modules from Git')
    args=parser.parse_args()
    if args.baseline:
        # Standalone benchmark process only. No checkout, index or source file
        # changes; share exactly the same runtime, operands and timing harness.
        import importlib
        import subprocess
        from pathlib import Path
        for suffix in ('translation_storage','translation_source','translation_preparation','translation_audit','translation_selection'):
            module=importlib.import_module('vibeqc.dlpno.'+suffix)
            path='python/vibeqc/dlpno/'+suffix+'.py'
            code=subprocess.run(['git','show',args.baseline+':'+path],
                cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True,text=True).stdout
            exec(compile(code,path,'exec'),module.__dict__)
        selection=importlib.import_module('vibeqc.dlpno.translation_selection')
        global select_translation_occupied
        select_translation_occupied=selection.select_translation_occupied
    from vibeqc.dlpno.translation_source import admit_translation_source
    rows=[]
    for n in args.cells:
        source,mesh,frozen,c,s=model(n)
        bloch=np.tile(c,(n,1,1)).astype(complex)*np.exp(.23j*np.arange(n*16).reshape(n,1,16))
        overlap=np.tile(s,(n,1,1))
        cells=np.array(list(np.ndindex(mesh)))
        outputs=[]; partitions=[]
        for compact in (False,True):
            def phases():
                if compact and not args.baseline:
                    return translation_preparation.ReferencePhases(mesh,cells/np.array(mesh),'benchmark')
                return np.exp(2j*np.pi*((cells/np.array(mesh))@cells.T))
            phase,phase_record=measure(phases,args.profile)
            rows.append(dict(phase='reference-phase-setup',cells=n,compact=compact,**phase_record,
                retained_bytes=(phase.fractions.nbytes+phase.cells.nbytes if compact and not args.baseline
                                else phase.nbytes)))
            if not args.baseline:
                _,transform_record=measure(lambda:translation_preparation.phase_inverse(phase,overlap),args.profile)
                rows.append(dict(phase='reference-inverse-transform',cells=n,compact=compact,
                    **transform_record,phase_record=(phase.record() if compact else
                        dict(phase_table_bytes=phase.nbytes))))
            prepared,record=measure(lambda:prepare_source(source,mesh,bloch,overlap,phase,compact),args.profile)
            bound,parts,audits=prepared
            partitions.append(parts)
            rows.append(dict(phase='source-realification',cells=n,compact=compact,**record,
                retained_bytes=sum(p.blocks.nbytes if compact else p.nbytes for p in parts),audits=audits))
            selected,record=measure(lambda:select_translation_occupied(bound,mesh,n_frozen=frozen,
                compact_storage=compact),args.profile)
            outputs.append(selected)
            rows.append(dict(phase='selection',cells=n,compact=compact,**record,
                retained_bytes=selected.occupied.nbytes+selected.frozen_occupied.nbytes,
                dense_seed_gram_bytes=8*(n*4)**2,
                workspace=_record(selected.provenance['preparation_workspace']),
                greedy_gram_work=_record(selected.provenance.get('greedy_gram_work',{}))))
            admitted,record=measure(lambda:admit_translation_source(bound,mesh,selected.occupied,
                frozen_occupied=selected.frozen_occupied),args.profile)
            rows.append(dict(phase='selected-source-admission',cells=n,compact=compact,**record,
                audit=_record(admitted.provenance.get('occupied_audit',{}))))
            def complete():
                bound,_,_=prepare_source(source,mesh,bloch,overlap,phases(),compact)
                return select_translation_occupied(bound,mesh,n_frozen=frozen,compact_storage=compact)
            _,record=measure(complete,args.profile)
            rows.append(dict(phase='complete-occupied-preparation',cells=n,compact=compact,**record))
        if not args.baseline:
            for reuse in (False,True):
                work={}
                _,record=measure(lambda:translation_preparation.cyclic_projected_frame(
                    source.reference.occupied_projectors.active,source.overlap,4*n,
                    1e-10,1e-10,'active',reuse=reuse,work=work),args.profile)
                rows.append(dict(phase='candidate-gram-and-lowdin',cells=n,reuse=reuse,
                                 **record,work=work))
        error=0.
        for dense,compact in zip(*partitions):
            for r in range(n):
                indices=np.arange(r*16,(r+1)*16)
                error=max(error,float(np.max(abs(dense[indices]@source.overlap.apply(dense,transpose=True).T
                                                  -compact.block(indices,np.arange(n*16))))))
        assert error < 1e-10
        rows[-1]['maximum_projector_error']=error
        assert outputs[0].active_seed_aos==outputs[1].active_seed_aos
        error=max(float(np.max(abs(np.asarray(a)-b),initial=0.)) for a,b in
            ((outputs[1].occupied,outputs[0].occupied),(outputs[1].frozen_occupied,outputs[0].frozen_occupied)))
        assert error < 1e-10
        rows[-1]['maximum_column_error']=error
    with OutputChannel.to_stdout(): write(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__': main()
