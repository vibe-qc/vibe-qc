#!/usr/bin/env python3
"""Check selective molecular J/K against an explicitly supplied Fock draft.

Reads only solvers/operators.py and fock_state.py from --operator-root, in an
isolated module namespace. No other checkout, native core or solver is imported.
This optional diagnostic does not install that draft or certify its identities,
energy semantics, production routing, periodic sources or memory admission.
"""
import argparse
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

import numpy as np

import vibeqc as vq
from vibeqc.manta import ReferenceJKBuilder
from vibeqc.symmetry_quartets import qualify_manta_quartets
from vibeqc.symmetry_quartets_screening import QuartetScreening


def load_draft(root):
    prefix = '_symmetry_fock_contract_probe'
    for name in (prefix, prefix+'.solvers'):
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    hashes = {}
    for suffix, relative in (('solvers.operators', 'solvers/operators.py'), ('fock_state', 'fock_state.py')):
        path = root/'python/vibeqc'/relative
        source = path.read_bytes()
        hashes[relative] = hashlib.sha256(source).hexdigest()
        name = prefix+'.'+suffix
        module = ModuleType(name)
        module.__package__ = name.rsplit('.',1)[0]
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(source, str(path), 'exec'), module.__dict__)
    return sys.modules[prefix+'.solvers.operators'], sys.modules[prefix+'.fock_state'], hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-root', type=Path, required=True)
    parser.add_argument('--screen-norm-budget', type=float, default=None,
        help='also bind fixed-orbit screening provenance; stored schedule only')
    args = parser.parse_args()
    operators, state, hashes = load_draft(args.operator_root)
    fixture = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tests/test_symmetry_quartets.py'))
    budget = fixture['BUDGET']
    results = []
    for kind in ('water','ammonia','cartesian'):
        mol,basis,rotations,translations = fixture['molecular_case'](kind)
        provider = ReferenceJKBuilder(basis)
        n = basis.nbasis
        overlap = np.asarray(vq.compute_overlap(basis))
        eigenvalues,vectors = np.linalg.eigh(overlap)
        x = vectors/np.sqrt(eigenvalues)
        frame = operators.OperatorFrame(basis_id='molecular-P0-probe', n=n,
            ordering='native AO', image_gauge='molecular', overlap=overlap, X=x)
        h = np.asarray(vq.compute_kinetic(basis))+np.asarray(vq.compute_nuclear(basis,mol))
        rng = np.random.default_rng(888)
        raw = rng.normal(size=(2,n,n))+1j*rng.normal(size=(2,n,n))
        spins = np.ascontiguousarray(.01*raw@raw.conj().transpose(0,2,1))
        for schedule in (('stored','streaming') if args.screen_norm_budget is None else ('stored',)):
            plan = qualify_manta_quartets(mol,basis,rotations,translations,provider=provider,
                source_tolerance=1e-10,budget=budget,schedule=schedule)
            screening = (None if args.screen_norm_budget is None else QuartetScreening(plan,
                maximum_omitted_norm=args.screen_norm_budget,budget=budget))
            full = np.empty((n,)*4)
            session = provider.session()
            for key in product(range(basis.nshells),repeat=4):
                slices = tuple(slice(plan.offsets[s],plan.offsets[s+1]) for s in key)
                full[slices] = session.evaluate(key)['values']
            del session
            if screening is not None:
                for keep,orbit in zip(screening.keep,plan.orbits):
                    if not keep:
                        for member in orbit.members:
                            slices = tuple(slice(plan.offsets[s],plan.offsets[s+1]) for s in member.shells)
                            full[slices] = 0.
            for executor in ('numpy','native'):
                for empty_beta in (False,True):
                    d = spins.copy()
                    if empty_beta:
                        d[1] = 0
                    jk = plan.jk_batch(np.stack([d.sum(axis=0),*d]),budget=budget,
                        executor=executor,targets=('j','k','k'),screening=screening)
                    assert jk.j_channels == (0,) and jk.k_channels == (1,2)
                    j = jk.j[0]
                    focks = np.stack([h+j-k for k in jk.k])
                    expected_j = fixture['dense_jk'](full,d.sum(axis=0))[0]
                    expected_k = np.stack([fixture['dense_jk'](full,spin)[1] for spin in d])
                    expected_focks = h+expected_j-expected_k
                    components = tuple((name,float(value.real)) for name,value in (
                        ('one_electron',np.einsum('pq,qp',d.sum(axis=0),h)),
                        ('hartree',.5*np.einsum('pq,qp',d.sum(axis=0),j)),
                        ('exchange',-.5*np.einsum('bpq,bqp',d,jk.k))))
                    energy = state.EnergyComponents(electronic=sum(v for _,v in components),
                        nuclear=0.,components=components)
                    source = state.SourceProvenance.capture('molecular-P0-symmetry-probe',
                        source_identity=jk.source_identity,targets=['j','k','k'],
                        screening=None if screening is None else screening.receipt,
                        j_channels=list(jk.j_channels),k_channels=list(jk.k_channels))
                    build = state.FockBuild.from_dense(method='UHF trial',generation=1,source=source,
                        energy=energy,densities=d,focks=focks,frames=[frame,frame],
                        spins=['alpha','beta'],kpoints=[None,None],weights=[1.,1.])
                    expected_energy = sum(np.einsum('pq,qp',d[s],h+.5*expected_j-.5*expected_k[s])
                                          for s in range(2)).real
                    density_snapshot = d.copy()
                    d[:] = 0
                    focks[:] = 0
                    # A later contraction cannot modify the evaluated state.
                    plan.jk_batch(np.stack([d.sum(axis=0),*d]),budget=budget,
                        executor=executor,targets=('j','k','k'),screening=screening)
                    for s,channel in enumerate(build.channels):
                        np.testing.assert_array_equal(channel.density,density_snapshot[s])
                        np.testing.assert_allclose(channel.operator.apply(np.eye(n)),
                            expected_focks[s],atol=2e-11,rtol=0)
                        for a in (channel.density,channel.operator.matrix):
                            try:
                                a.setflags(write=True)
                            except ValueError:
                                pass
                            else:
                                raise AssertionError('evaluated state permits mutation')
                    assert abs(build.energy.electronic-expected_energy) < 2e-11
                    results.append(dict(molecule=kind,schedule=schedule,executor=executor,
                        empty_beta=empty_beta,source_identity=jk.source_identity,
                        screening=None if screening is None else screening.receipt,
                        density_error_bounds=jk.error_bounds.tolist(),
                        max_fock_error=float(max(np.max(np.abs(c.operator.matrix-expected_focks[s]))
                                                 for s,c in enumerate(build.channels))),
                        energy_error=float(abs(build.energy.electronic-expected_energy)),
                        state_array_bytes=build.storage_nbytes,source_work=asdict(jk.work)))
    print(json.dumps(dict(draft_source_hashes=hashes,cases=results,
        production_reduction_authorized=False,scope='molecular snapshot crossing only'),indent=2))


if __name__ == '__main__':
    main()
