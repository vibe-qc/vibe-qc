"""One-axis classic RHF/MP2 study from an accepted reference archive.

Run through vq on a pinned containing runtime. The JSON case list changes one
control, mesh, orbital basis or auxiliary basis at a time; no automatic tuning.
Artifacts belong outside the checkout. Inspect every row: process exit zero
does not mean every case passed. A completed calculation is not a claim
of physical convergence. See docs/chi_mp2_convergence.md for the fixed gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-core-sha256',required=True)
    a=p.parse_args()
    import numpy as np
    import vibeqc as vq
    from vibeqc import _vibeqc_core
    from vibeqc.periodic.chi.checkpoint import load_chi_hf_reference,_basis_record
    from vibeqc.periodic.chi.reference import prepare_chi_hf_reference
    from vibeqc.periodic.chi.fitted import ChiCoulombControls
    from vibeqc.periodic.chi.correlation import run_chi_mp2
    from vibeqc.periodic.chi.local_mp2 import run_chi_local_mp2
    from vibeqc.output import OutputChannel,write
    from vibeqc.output.chi_correlation import chi_correlation_record
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    core_sha=sha(_vibeqc_core.__file__)
    if core_sha!=a.expected_core_sha256:raise RuntimeError('native runtime pin mismatch')
    out=a.output.resolve()
    repo=next((p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists()),None)
    if repo and (out==repo or repo in out.parents):raise ValueError('output must be outside checkout')
    out.mkdir(parents=True,exist_ok=True)
    base=load_chi_hf_reference(a.reference,expected_method='RHF',mode='accepted')
    physical=base.hf_result._chi_restart_physical
    g=physical['geometry'];base_controls=physical['controls']['coulomb_controls']
    if physical['ecp']!='all-electron' or physical['controls']['fit_source']!='classic':
        raise ValueError('this bounded study supports all-electron classic RHF only')
    system=vq.PeriodicSystem(3,np.array(g['lattice_bohr']),[vq.Atom(n['Z'],n['xyz_bohr']) for n in g['nuclei']])
    if g['charge']!=0 or g['multiplicity']!=1:raise ValueError('neutral singlet study only')
    basis=vq.BasisSet(system.unit_cell_molecule(),physical['basis']['name'])
    if _basis_record(basis)['sha256']!=physical['basis']['sha256']:raise ValueError('orbital basis differs from archive')
    cases=json.loads(a.cases.read_text())
    names=[c["name"] for c in cases]
    if len(names)!=len(set(names)) or any(not n or Path(n).name!=n or n in (".","..") for n in names):
        raise ValueError("case names must be unique plain filenames")
    report=dict(schema='chi-classic-mp2-convergence/v1',reference_sha256=sha(a.reference),
        input_sha256=sha(a.cases),native_sha256=core_sha,package=vq.__file__,
        reference_state=base.state_identity,physical=physical,rows=[],
        convention='SCF Fock retained including BvK Ewald exchange zero mode',
        qualification='one-axis measurements only; experimental; D93 not-qualified')
    def save():
        with OutputChannel.to_file(out/'measurements.json'):
            write(json.dumps(report,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n')
    save()
    # Nejad et al., JCP 163, 214107 (2025), Sec. II Eqs. (1)-(8):
    # compare correlation per primitive cell with the SAME retained SCF Fock.
    # No exchange-zero-mode subtraction or finite-size correction is applied.
    for case in cases:
        name=case['name'];t=time.monotonic();r=dict(input=case,status='STARTED')
        report['rows'].append(r);save();print('START',name,flush=True)
        try:
            changes=case.get('changes',{})
            if len(changes)>1:raise ValueError('only one changed approximation per case')
            controls=dict(base_controls);mesh=tuple(physical['mesh']);b=basis
            aux=physical['controls']['aux_basis']
            for key,value in changes.items():
                if key=='mesh':mesh=tuple(value)
                elif key=='basis':b=vq.BasisSet(system.unit_cell_molecule(),value)
                elif key=='aux_basis':aux=value
                elif key in controls:controls[key]=value
                else:raise ValueError('unknown refinement axis')
            if not changes:
                ref=base
            else:
                opts=vq.PeriodicRHFOptions();opts.max_iter=250
                opts.conv_tol_energy=1e-12;opts.conv_tol_grad=1e-10
                guess=physical['options']['initial_guess'][1]
                opts.initial_guess=getattr(vq.InitialGuess,guess)
                ref=prepare_chi_hf_reference(system,b,mesh,opts,method='RHF',backend='ri',fit_source='classic',
                    coulomb_controls=ChiCoulombControls(**controls),aux_basis=aux,
                    symmetry_mode='off',symmetry_precision=1e-5,
                    checkpoint_path=out/(name+'.vqr'),progress=False,verbose=0)
            h=ref.hf_result
            q=json.loads(h.fitted_operator.descriptor.executed_domains_json)['q_records']
            r.update(hf_energy=float(h.energy),state_identity=ref.state_identity,
                factor_identity=h.fitted_operator.identity,stationarity=h.chi_scf_acceptance,
                overlap_min=float(min(np.linalg.eigvalsh(s)[0] for s in h.overlap)),
                overlap_max=float(max(np.linalg.eigvalsh(s)[-1] for s in h.overlap)),
                metric=[{k:v for k,v in z.items() if 'metric' in k or k in ('rank','q')} for z in q],
                gap=float(min(np.min(e[np.array(o)<1]) for e,o in zip(h.mo_energies,h.occupations))-
                          max(np.max(e[np.array(o)>1]) for e,o in zip(h.mo_energies,h.occupations))),
                controls=controls,mesh=list(mesh),basis=_basis_record(b),aux_basis=aux,partitions=[])
            save()
            for frozen in case.get('frozen_core',[0]):
                canonical=run_chi_mp2(ref,frozen_core=frozen).correlation
                part=dict(frozen_core=frozen,canonical=float(canonical.e_corr_per_cell),local=[])
                r['partitions'].append(part);save()
                for label,domain,pno in case.get('local',[['full',0.,0.]]):
                    local=run_chi_local_mp2(ref,frozen_core=frozen,seed_basis=case.get('seed_basis','ao'),
                        tcut_mkn=domain,tcut_pno=pno,transport_backend='bounded',compact_storage=True,
                        factor_provider='streamed',max_factor_bytes=64*1024**2,max_transport_bytes=256*1024**2,
                        max_iter=500,conv_tol_energy=1e-13,conv_tol_residual=1e-11,
                        output=out/f'{name}-fc{frozen}-{label}')
                    rec=chi_correlation_record(local)
                    row=dict(label=label,raw=local.raw_e_corr_per_cell,pno_correction=local.pno_correction_per_cell,
                        raw_error=local.raw_e_corr_per_cell-canonical.e_corr_per_cell,
                        residual=rec['residual_norm'],energy_change=rec['energy_change'],converged=local.converged)
                    part['local'].append(row);save()
                    if label=='full' and not (row['converged'] and abs(row['raw_error'])<=2e-9 and
                        row['residual']<=1e-11 and abs(row['energy_change'])<=1e-13):
                        raise RuntimeError('full-space canonical/local gate failed')
            r['status']='COMPLETED'
        except Exception as e:
            r.update(status='FAILED',exception=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        r['seconds']=time.monotonic()-t;save();print('END',name,r['status'],r.get('message',''),flush=True)
    if sha(_vibeqc_core.__file__)!=core_sha:raise RuntimeError('native changed during study')


if __name__=='__main__':main()
