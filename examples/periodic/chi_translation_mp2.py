"""Reproducible vq Validator cases for experimental chi translation local MP2.

Use a runtime containing the chi block adapter and prepared translation solver.
Run H2 and LiH on Gamma, (3,1,1) and even (4,1,1) meshes; LiH supports explicit
frozen_core=0/1. Reuse saved accepted HF archives with --reference, including
applicable scalar ECP archives. Existing jobs and reference evidence remain
owned by Validator. This script neither submits nor manages jobs.

Full-space raw local MP2 is compared with independently evaluated same-source
canonical MP2. Domain and PNO ladders vary one cutoff at a time. No discarded
space correction participates in the raw-limit verdict. The 2e-9 Ha/cell
limit allows accumulated iterative residual error; tighten solver controls to
verify a borderline case. This finite-source consistency check is not an
ORCA/CRYSTAL, physical-convergence or production verdict. Keep the SCF Fock's
BvK exchange zero mode and explicit frozen-electron convention in comparisons.

--help is standard-library only. Write artifacts outside the source checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('h2','lih'),default='h2')
    parser.add_argument('--mesh',type=int,choices=(1,2,3,4),default=4)
    parser.add_argument('--reference',type=Path,help='reuse an accepted classic RHF .vqr archive')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-core',type=int,choices=(0,1),default=0)
    parser.add_argument('--factor-providers',nargs='+',choices=('streamed','dense'),
                        default=['streamed','dense'])
    parser.add_argument('--max-factor-bytes',type=int,default=64*1024**2)
    parser.add_argument('--factor-auxiliary-tile',type=int,default=16)
    parser.add_argument('--factor-cache-bytes',type=int,default=1024**2)
    parser.add_argument('--expected-provider-sha256',help='require the identified containing runtime module')
    parser.add_argument('--coefficient-storage',nargs='+',choices=('compact','dense'),default=['compact','dense'])
    parser.add_argument('--expected-storage-sha256',help='require the selected-column/pair-recipe implementation')
    parser.add_argument('--max-selection-bytes',type=int,default=64*1024**2)
    parser.add_argument('--expected-implementation-sha256',help='require all occupied preparation consumer modules')
    parser.add_argument('--expected-preparation-sha256',help='require the cyclic occupied preparation implementation')
    parser.add_argument('--expected-core-sha256',help='require the native recipe-only transport implementation')
    parser.add_argument('--domain-cutoffs',type=float,nargs='*',default=[.05,.01])
    parser.add_argument('--pno-cutoffs',type=float,nargs='*',default=[1e-4,1e-6])
    args=parser.parse_args()
    destination=args.output.expanduser().resolve()
    source_root=next((p for p in Path(__file__).resolve().parents
        if (p/'pyproject.toml').is_file() and (p/'python/vibeqc').is_dir()),None)
    if source_root is not None and (destination==source_root or source_root in destination.parents):
        parser.error('--output must be outside the source checkout')
    destination.mkdir(parents=True,exist_ok=True)

    import numpy as np
    import vibeqc as vq
    from vibeqc.output import OutputChannel, write
    from vibeqc.output.chi_correlation import chi_correlation_record
    from vibeqc.periodic.chi.checkpoint import load_chi_hf_reference
    from vibeqc.periodic.chi.reference import prepare_chi_hf_reference
    from vibeqc.periodic.chi.fitted import ChiCoulombControls
    from vibeqc.periodic.chi.correlation import run_chi_mp2
    from vibeqc.periodic.chi.local_mp2 import run_chi_local_mp2
    from vibeqc.dlpno import translation_bloch_factors
    from vibeqc.dlpno import translation_storage, translation_preparation
    from vibeqc import _vibeqc_core
    provider_sha=hashlib.sha256(Path(translation_bloch_factors.__file__).read_bytes()).hexdigest()
    storage_sha=hashlib.sha256(Path(translation_storage.__file__).read_bytes()).hexdigest()
    preparation_sha=hashlib.sha256(Path(translation_preparation.__file__).read_bytes()).hexdigest()
    import importlib
    implementation=hashlib.sha256()
    for name in sorted(['vibeqc.dlpno.'+suffix for suffix in (
        'translation_preparation','translation_audit','translation_selection','translation_storage','translation_source',
        'translation','translation_builder','translation_pao','translation_domains',
        'translation_driver','translation_transport','translation_solver')]+['vibeqc.periodic.chi.local_mp2']):
        implementation.update(name.encode()+b'\0')
        implementation.update(Path(importlib.import_module(name).__file__).read_bytes())
    implementation_sha=implementation.hexdigest()
    if args.expected_implementation_sha256 and implementation_sha != args.expected_implementation_sha256:
        raise RuntimeError('runtime does not contain all expected occupied preparation consumers')
    core_sha=hashlib.sha256(Path(_vibeqc_core.__file__).read_bytes()).hexdigest()
    if args.expected_provider_sha256 and provider_sha != args.expected_provider_sha256:
        raise RuntimeError('runtime does not contain the expected streamed factor implementation')
    if args.expected_storage_sha256 and storage_sha != args.expected_storage_sha256:
        raise RuntimeError('runtime does not contain the expected compact coefficient implementation')
    if args.expected_preparation_sha256 and preparation_sha != args.expected_preparation_sha256:
        raise RuntimeError('runtime does not contain the expected cyclic occupied preparation')
    if args.expected_core_sha256 and core_sha != args.expected_core_sha256:
        raise RuntimeError('runtime does not contain the expected native transport implementation')
    if args.reference:
        reference=load_chi_hf_reference(args.reference,expected_method='RHF',mode='accepted')
    else:
        if args.case=='h2':
            atoms=[vq.Atom(1,[3.,3.,2.3]),vq.Atom(1,[3.,3.,3.7])]
        else:
            atoms=[vq.Atom(3,[3.,3.,1.5]),vq.Atom(1,[3.,3.,4.5])]
        system=vq.PeriodicSystem(3,np.eye(3)*6.,atoms)
        basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g')
        options=vq.PeriodicRHFOptions()
        options.conv_tol_energy=1e-12;options.conv_tol_grad=1e-10
        options.max_iter=250
        controls=ChiCoulombControls(6.2,6.2,10.,8.,5.,.6,
            support_policy='physical_pairs',memory_byte_cap=64*1024**3,
            support_record_byte_cap=512*1024**2)
        reference=prepare_chi_hf_reference(system,basis,(args.mesh,1,1),options,
            method='RHF',backend='ri',fit_source='classic',coulomb_controls=controls,
            aux_basis='def2-universal-jkfit',checkpoint_path=destination/'hf.vqr',
            progress=False,verbose=0)
    canonical=run_chi_mp2(reference,frozen_core=args.frozen_core).correlation
    settings=[('full',0.,0.)]
    settings += [(f'domain-{t:g}',t,0.) for t in args.domain_cutoffs]
    settings += [(f'pno-{t:g}',0.,t) for t in args.pno_cutoffs]
    rows=[]
    for provider,storage,label,domain,pno in ((provider,storage,*setting) for provider in args.factor_providers
                                     for storage in args.coefficient_storage for setting in settings):
        try:
            result=run_chi_local_mp2(reference,frozen_core=args.frozen_core,
                tcut_mkn=domain,tcut_pno=pno,max_iter=500,
                factor_provider=provider,max_factor_bytes=args.max_factor_bytes,
                compact_storage=storage=='compact',max_selection_bytes=args.max_selection_bytes,
                factor_auxiliary_tile=args.factor_auxiliary_tile,
                factor_cache_bytes=args.factor_cache_bytes,
                conv_tol_energy=1e-13,conv_tol_residual=1e-11,
                output=destination/f'{provider}-{storage}-{label}')
            row=dict(case=label,provider=provider,storage=storage,converged=result.converged,
                raw=result.raw_e_corr_per_cell,pno_estimate=result.pno_correction_per_cell,
                raw_minus_canonical=result.raw_e_corr_per_cell-canonical.e_corr_per_cell,
                residual=result.translation.solution.max_residual,
                iterations=result.translation.solution.n_iter,
                factor_record=chi_correlation_record(result)['factor_provider_record'],
                coefficient_record=chi_correlation_record(result)['coefficient_storage'],
                occupied_preparation=chi_correlation_record(result)['occupied_selection'],
                occupied_audit_execution=chi_correlation_record(result)['occupied_audit_execution'])
        except ValueError as exc:
            # A finite domain can be empty or break a self-inverse PNO space.
            # Record that refusal; never promote it to a converged energy.
            if label=='full': raise
            row=dict(case=label,provider=provider,storage=storage,refused=str(exc))
        rows.append(row)
    full=[row for row in rows if row['case']=='full']
    passed=all(row['converged'] and abs(row['raw_minus_canonical'])<=2e-9 for row in full)
    comparisons=[]
    for label,_,_ in settings:
        cases=[row for row in rows if row['case']==label and 'raw' in row]
        for candidate in cases[1:]:
            differences={key:candidate[key]-cases[0][key] for key in ('raw','pno_estimate','residual')}
            comparisons.append(dict(case=label,provider=candidate['provider'],storage=candidate['storage'],
                reference_provider=cases[0]['provider'],reference_storage=cases[0]['storage'],**differences))
            passed &= all(abs(value)<=2e-9 for value in differences.values())
    with OutputChannel.to_stdout():
        write(json.dumps(dict(schema='vibeqc.chi.translation-validator/v2',
            runtime=dict(reference.hf_result.process_provenance),
            provider_module_sha256=provider_sha,provider_comparisons=comparisons,
            storage_module_sha256=storage_sha,preparation_module_sha256=preparation_sha,
            implementation_sha256=implementation_sha,
            native_module_sha256=core_sha,
            canonical=chi_correlation_record(canonical),rows=rows,
            full_raw_limit_pass=passed,
            qualification='experimental; D93 fitted/post-HF not-qualified; external acceptance pending'),
            indent=2,allow_nan=False)+'\n')
    if not passed:
        raise RuntimeError('raw full-domain/zero-PNO canonical comparison failed')


if __name__=='__main__':
    main()
