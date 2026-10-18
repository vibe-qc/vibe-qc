"""Bounded selector-only ladder; synthetic algebra, not physical acceptance.

Run as python -m benchmarks.chi_occupied_preparation in the development env.
All source operators/projectors are created before measurement. The source
has a nonorthogonal metric and exactly degenerate translated seed spectra.
"""
import json
from time import perf_counter
import tracemalloc
from types import SimpleNamespace

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


def main():
    from vibeqc.output import OutputChannel,write
    rows=[]
    for n in (1,2,4,8):
        source,mesh,frozen,c,s=model(n)
        bloch=np.tile(c,(n,1,1)).astype(complex)*np.exp(.23j*np.arange(n*16).reshape(n,1,16))
        overlap=np.tile(s,(n,1,1))
        phase=np.exp(2j*np.pi*np.arange(n)[:,None]*np.arange(n)[None,:]/n)
        audits=[]
        for compact in (False,True):
            tracemalloc.start();start=perf_counter()
            partitions=[(translation_preparation.real_occupied_projector(bloch[:,:,part],overlap,phase,
                mesh,'partition',1e-10)[0] if compact else
                _real_occupied(bloch[:,:,part],overlap,phase,'partition')[0])
                for part in (slice(2,6),slice(0,2))]
            elapsed=perf_counter()-start
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            audits.append(partitions)
            rows.append(dict(phase='source-realification',cells=n,compact=compact,elapsed_s=elapsed,
                traced_peak_bytes=peak,retained_bytes=sum(p.blocks.nbytes if compact else p.nbytes for p in partitions)))
        error=0.
        for dense,compact in zip(*audits):
            for r in range(n):
                indices=np.arange(r*16,(r+1)*16)
                error=max(error,float(np.max(abs(dense[indices]@source.overlap.apply(dense,transpose=True).T
                                                  -compact.block(indices,np.arange(n*16))))))
            assert error < 1e-10
        rows[-1]['maximum_projector_error']=error
        outputs=[]
        for compact in (False,True):
            tracemalloc.start();start=perf_counter()
            selected=select_translation_occupied(source,mesh,n_frozen=frozen,compact_storage=compact)
            elapsed=perf_counter()-start
            _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            outputs.append(selected)
            rows.append(dict(phase='selection',cells=n,compact=compact,elapsed_s=elapsed,traced_peak_bytes=peak,
                retained_bytes=selected.occupied.nbytes+selected.frozen_occupied.nbytes,
                dense_seed_gram_bytes=8*(n*4)**2,
                workspace=_record(selected.provenance['preparation_workspace'])))
        assert outputs[0].active_seed_aos==outputs[1].active_seed_aos
        error=max(float(np.max(abs(np.asarray(a)-b),initial=0.)) for a,b in
            ((outputs[1].occupied,outputs[0].occupied),(outputs[1].frozen_occupied,outputs[0].frozen_occupied)))
        assert error < 1e-10
        rows[-1]['maximum_column_error']=error
    with OutputChannel.to_stdout(): write(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__': main()
