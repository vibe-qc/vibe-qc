"""Frozen source-tree MANTA control examples for the bounded 0.18 scope.

These small custom-basis cases qualify controls, not bulk-crystal accuracy.
Run through vq for scientific acceptance. The release candidate still needs
its normal CI/publication gates. Fixtures reuse the existing accepted producers.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

PROFILES = ('image-tight', 'ordinary-rhf', 'global-pbe0', 'compact-tight')


def make_job(directory, profile):
    """Existing accepted cases, with all selected physical controls explicit."""
    from tests.test_manta_periodic_one_electron_images import ri_image_job, image_accuracy_targets
    from benchmarks.manta_periodic_ri_nuclear_accuracy import accuracy_job
    if profile not in PROFILES:
        raise ValueError(f'Unknown MANTA release profile: {profile}')
    if profile == 'image-tight':
        job, _, _, _ = ri_image_job(directory, automatic=True)
        job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'],
            pair_cutoff_bohr=10., one_electron_accuracy=image_accuracy_targets(1e-6, 3))
        job['periodic_fitting'] = dataclasses.replace(job['periodic_fitting'], auxiliary_cutoff_bohr=12.1)
    else:
        job, _, _ = accuracy_job(directory, profile)
        if profile == 'compact-tight':
            # Adds the S/T certificate to the already accepted tight K/nuclear
            # case. Physical supports and the numerical method stay fixed.
            job['periodic_integrals'] = dataclasses.replace(job['periodic_integrals'],
                one_electron_accuracy=image_accuracy_targets(1e-8, 2))
    budget = job['integral_engine'].budget
    budget.memory_bytes = 4 * 2**30
    budget.shell_calls = 10**12
    budget.primitive_quartets = 10**12
    # The optional arithmetic plans reserve their worst-case work even when
    # no arithmetic target is requested. Keep the physical fixture unchanged.
    work_cap = 7*10**18 if profile == 'image-tight' else 2*10**18
    budget.scalar_terms = work_cap
    budget.kernel_terms = work_cap
    return job


def configuration(job):
    """Portable physical inputs; retained alongside the measured receipts."""
    def fields(obj, names):
        return {name: getattr(obj, name) for name in names}
    system = job['system']; physical = job['periodic_integrals']; fitting = job['periodic_fitting']
    def basis(b):
        return dict(name=b.name, shells=[dict(atom=s.atom_index, l=s.l, pure=s.pure,
            exponents=list(s.exponents), coefficients=list(s.coefficients), origin=list(s.origin)) for s in b.shells()])
    # Geometry, basis and all chosen domains are also preserved in managed QVF.
    return dict(method=job['method'], functional=job.get('functional'),
        lattice=system.lattice.tolist(), multiplicity=system.multiplicity, charge=system.charge,
        atoms=[dict(Z=a.Z, xyz=list(a.xyz)) for a in system.unit_cell],
        orbital=basis(job['basis']), auxiliary=basis(job['coulomb'].auxiliary_basis),
        metric=fields(job['coulomb'].metric, ('absolute_cutoff','relative_cutoff','mode_block')),
        xc=dataclasses.asdict(job['periodic_xc']) if job.get('periodic_xc') else None,
        mesh=list(job['kpoints'].mesh), shift=list(job['kpoints'].is_shift),
        physical=fields(physical, ('reciprocal_energy_cutoff','pair_cutoff_bohr','nuclear_cutoff_bohr','alpha','reciprocal_batch')),
        fitting=fields(fitting, ('omega','auxiliary_cutoff_bohr','coupling_multipole_error','coupling_multipole_order',
            'coupling_multipole_hierarchy','coupling_multipole_compress','coupling_multipole_global_actions','coupling_multipole_pair_tree')),
        exchange_exxdiv=job['exchange_exxdiv'], max_iter=job['max_iter'], conv_tol_energy=job['conv_tol_energy'],
        use_diis=job['use_diis'], budget=fields(job['integral_engine'].budget,
            ('memory_bytes','shell_calls','primitive_quartets','scalar_terms','kernel_terms')))


def run_case(directory, profile):
    import numpy as np
    import vibeqc as vq
    from vibeqc.output import validate_qvf
    job = make_job(directory, profile)
    start = time.perf_counter(); cpu = time.process_time()
    result = vq.run_periodic_job(**job)
    wall, cpu = time.perf_counter()-start, time.process_time()-cpu
    assert result.converged, f'{profile}: SCF did not converge'
    assert validate_qvf(str(job['output'].with_suffix('.qvf')))['valid']
    provenance = result.integral_provenance
    assert not provenance['symmetry_reduced']
    certs = {}
    for name in ('one_electron_accuracy', 'exchange_accuracy', 'nuclear_accuracy'):
        certificate = provenance.get(name)
        if certificate is not None:
            assert certificate['qualified'], (profile, name)
            certs[name] = certificate
    assert 'one_electron_accuracy' in certs
    if profile != 'image-tight':
        assert {'exchange_accuracy','nuclear_accuracy'} <= certs.keys()
    # Disable certificates only, retaining exactly the accepted physical
    # domains and RI coupling selection. This must not change the Fock map.
    control = vq.run_periodic_job(**dict(job, output=directory/'control'/'job',
        periodic_integrals=dataclasses.replace(job['periodic_integrals'],
            one_electron_accuracy=None, exchange_accuracy=None, nuclear_accuracy=None)))
    assert control.converged and result.energy == control.energy
    assert result.native.receipt.source_identity == control.native.receipt.source_identity
    for name in ('density','fock','overlap','hcore'):
        np.testing.assert_array_equal(getattr(result.native, name), getattr(control.native, name))
    # Each supported physical target refuses before creating a computed archive.
    refusals = []
    for name, field in (('one_electron_accuracy','kinetic_error'),
                        ('exchange_accuracy','matrix_error'), ('nuclear_accuracy','matrix_error')):
        original = getattr(job['periodic_integrals'], name)
        if original is None:
            continue
        target = type(original)()
        names = {
            'one_electron_accuracy': ('overlap_error','kinetic_error','kinetic_energy_error','completion_cube_radius'),
            'exchange_accuracy': ('matrix_error','energy_error','ao_completion_cube_radius','reciprocal_completion_cube_radius'),
            'nuclear_accuracy': ('matrix_error','attraction_energy_error','nuclear_energy_error','ao_completion_cube_radius',
                                 'real_completion_cube_radius','reciprocal_completion_cube_radius'),
        }[name]
        for key in names:
            setattr(target, key, getattr(original, key))
        setattr(target, field, 1e-100)
        output = directory/('refused-'+name)/'job'
        try:
            vq.run_periodic_job(**dict(job, output=output,
                periodic_integrals=dataclasses.replace(job['periodic_integrals'], **{name: target})))
        except (ValueError, RuntimeError) as error:
            assert 'accuracy target' in str(error), str(error)
        else:
            raise AssertionError(f'{profile}: unmet {name} target did not refuse')
        assert not output.with_suffix('.qvf').exists()
        refusals.append(name)
    row = dict(profile=profile, converged=result.converged, energy=result.energy, iterations=result.iterations,
        configuration=configuration(job), certificates=certs,
        fitting_accuracy=provenance['fitting'].get('accuracy'),
        fitting_metrics=provenance['fitting'].get('metrics'),
        component_ledger=provenance.get('component_ledger'),
        source_identity=result.native.receipt.source_identity, wall_seconds=wall, cpu_seconds=cpu,
        modeled_peak_bytes=result.native.plan.modeled_peak_bytes,
        control='bitwise energy/state/source', refusals=refusals,
        qvf=str(job['output'].with_suffix('.qvf')))
    return row, result


def run(directory, profiles=PROFILES):
    import vibeqc as vq
    from vibeqc import _vibeqc_core as core
    root=Path(__file__).resolve().parents[1]
    rows=[run_case(directory/profile, profile)[0] for profile in profiles]
    return dict(scope='bounded 0.18 MANTA controls; no whole-method accuracy or bulk parity claim',
        source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        python_package=str(Path(vq.__file__).resolve()), native=str(Path(core.__file__).resolve()),
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(), rows=rows,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=PROFILES,action='append')
    args=parser.parse_args()
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='1'
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    print(json.dumps(run(args.output,args.profile or PROFILES),indent=2,allow_nan=False))
