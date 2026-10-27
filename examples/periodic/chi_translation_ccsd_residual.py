"""Replay an internal CCSD residual comparison on an accepted chi HF archive.

This evaluates arbitrary nonzero amplitudes, not a converged CCSD calculation.
The translation path is compared with an explicitly requested small dense
canonical oracle. Reuse existing H2/LiH/scalar-ECP archives and jobs; the script
neither submits jobs nor claims scientific acceptance. Write outside checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-core',type=int,default=0)
    parser.add_argument('--seed-basis',choices=('ao','cosine'),default='ao')
    parser.add_argument('--amplitude-seed',type=int,default=586)
    parser.add_argument('--amplitude-scale',type=float,default=.01)
    parser.add_argument('--max-workspace-bytes',type=int,default=256*1024**2)
    parser.add_argument('--max-oracle-bytes',type=int,default=256*1024**2)
    parser.add_argument('--expected-core-sha256')
    parser.add_argument('--expected-evaluator-sha256')
    args=parser.parse_args()
    destination=args.output.expanduser().resolve()
    root=next((p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists()),None)
    if root and (destination==root or root in destination.parents):
        parser.error('--output must be outside the source checkout')
    import numpy as np
    from vibeqc import _vibeqc_core
    from vibeqc.banner import loaded_process_provenance
    from vibeqc.output import OutputChannel,write
    from vibeqc.periodic.chi.checkpoint import load_chi_hf_reference
    from vibeqc.periodic.chi.local_mp2 import build_chi_translation_source
    from vibeqc.dlpno.translation_selection import select_translation_occupied
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno import translation_ccsd
    from vibeqc.dlpno.translation_operators import matrix_apply
    from vibeqc.dlpno._ccsd_cs import _blocks,cs_ccsd_residual,cs_ccsd_energy
    core_hash=hashlib.sha256(Path(_vibeqc_core.__file__).read_bytes()).hexdigest()
    evaluator_hash=hashlib.sha256(Path(translation_ccsd.__file__).read_bytes()).hexdigest()
    for expected,actual in ((args.expected_core_sha256,core_hash),(args.expected_evaluator_sha256,evaluator_hash)):
        if expected and expected!=actual:
            raise RuntimeError('runtime does not contain the requested CCSD residual implementation')
    reference=load_chi_hf_reference(args.reference)
    source=build_chi_translation_source(reference,frozen_core=args.frozen_core)
    mesh=source.factors.mesh
    cells=int(np.prod(mesh))
    selected=select_translation_occupied(source,mesh,n_frozen=args.frozen_core*cells,
        compact_storage=True,seed_basis=args.seed_basis)
    build=build_translation_pair_pnos(source,mesh,selected.occupied,
        frozen_occupied=selected.frozen_occupied,compact_storage=True,tcut_pno=0.)
    spaces=build.spaces
    rng=np.random.default_rng(args.amplitude_seed)
    singles=[args.amplitude_scale*rng.normal(size=spaces.pair_ranks[spaces.resolve(i,i).row])
             for i in range(spaces._n_bands)]
    doubles=[]
    for p,((i,j),rank) in enumerate(zip(spaces.representative_pairs,spaces.pair_ranks)):
        t=args.amplitude_scale*rng.normal(size=(rank,rank))
        if i==j: t=.5*(t+t.T)
        if p in spaces.stabilizers:
            U=spaces.stabilizers[p];t=.5*(t+U@t.T@U.T)
        doubles.append(t)
    actual=translation_ccsd.evaluate_translation_ccsd(source,build,singles,doubles,
        max_workspace_bytes=args.max_workspace_bytes)
    # EXPLICIT SMALL ORACLE ONLY. These global arrays never enter the new path.
    no=spaces._n_occupied
    nv=spaces._occupied.shape[0]-no-spaces._frozen_occupied.shape[1]
    oracle_bytes=64*(no+nv)**4+64*source.factors.n_aux*(no+nv)**2
    if oracle_bytes>args.max_oracle_bytes:
        raise MemoryError('explicit canonical oracle exceeds max-oracle-bytes')
    V=spaces.coefficients(0,0)
    if V.shape[1]!=nv:
        raise ValueError('canonical oracle requires the full virtual space')
    C=np.asarray(spaces._occupied)
    fov=C.T@matrix_apply(spaces._fock,V)
    fvv=V.T@matrix_apply(spaces._fock,V)
    blocks=_blocks(source.factors.mo_transform(C,V),source.factors.mo_transform(V,V),
                   source.factors.mo_transform(C,C))
    t1=np.empty((no,nv));t2=np.empty((no,no,nv,nv))
    for i in range(no):
        u=V.T@matrix_apply(spaces._overlap,spaces.coefficients(i,i))
        t1[i]=u@singles[i%spaces._n_bands]
        for j in range(i,no):
            u=V.T@matrix_apply(spaces._overlap,spaces.coefficients(i,j))
            t2[i,j]=u@spaces.orient_tensor(i,j,doubles)@u.T
            t2[j,i]=t2[i,j].T
    r1,r2=cs_ccsd_residual(t1,t2,spaces.occupied_fock,fvv,fov,blocks)
    error1=error2=0.
    for i,value in enumerate(actual['R1']):
        u=V.T@matrix_apply(spaces._overlap,spaces.coefficients(i,i))
        error1=max(error1,float(np.max(abs(value-r1[i]@u))))
    for pair,value in zip(spaces.representative_pairs,actual['R2']):
        u=V.T@matrix_apply(spaces._overlap,spaces.coefficients(*pair))
        error2=max(error2,float(np.max(abs(value-u.T@r2[pair]@u))))
    energy=cs_ccsd_energy(t1,t2,fov,blocks)/cells
    record=dict(kind='internal-translation-ccsd-arbitrary-amplitude-comparison',
        scientific_acceptance=False,iterative_ccsd=False,runtime=loaded_process_provenance(),
        native_sha256=core_hash,evaluator_sha256=evaluator_hash,
        mesh=list(mesh),frozen_core=args.frozen_core,seed_basis=args.seed_basis,
        amplitude_seed=args.amplitude_seed,amplitude_scale=args.amplitude_scale,
        raw_energy_per_cell=actual['energy'],canonical_energy_per_cell=energy,
        energy_difference=abs(actual['energy']-energy),singles_difference=error1,doubles_difference=error2,
        pno_mp2_correction_per_cell=actual['pno_mp2_correction_per_cell'],
        evaluator=actual['provenance'],citations=actual['citations'],
        convention='unchanged SCF Fock including BvK exchange zero mode')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with OutputChannel.to_file(destination):
        write(json.dumps(record,indent=2)+'\n')
    if max(error1,error2)>2e-10 or abs(actual['energy']-energy)>2e-11:
        raise RuntimeError('arbitrary-amplitude CCSD comparison failed')


if __name__=='__main__':
    main()
