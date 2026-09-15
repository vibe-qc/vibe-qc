"""Independent dense-core GDF references, run in an external PySCF process.

The exact contracted Gaussian records, accepted densities and component
matrices are retained. This generator never imports vibe-qc and does not
certify native parity. Use --input-json to reproduce a recorded geometry
and basis verbatim rather than reconstructing a named cell.

References: Ye and Berkelbach, JCP 154, 131104 (2021),
DOI 10.1063/5.0046617; Sun et al., JCP 153, 024109 (2020),
DOI 10.1063/5.0006074.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--system', choices=('si', 'cao'), default='si')
    parser.add_argument('--input-json', type=Path)
    parser.add_argument('--mesh', type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument('--methods', nargs='+', choices=('hf', 'pbe'), default=('hf', 'pbe'))
    parser.add_argument('--grid-level', type=int, default=5)
    parser.add_argument('--omegas', type=float, nargs='+', default=(.6, .8))
    args = parser.parse_args()

    import numpy as np
    import pyscf
    from pyscf.gto.basis import parse_gaussian
    from pyscf.pbc import dft, gto, scf
    from pyscf.pbc.df.rsdf import RSGDF

    if (min(args.mesh) < 1 or not 0 <= args.grid_level <= 9
            or any(not np.isfinite(w) or w <= 0 for w in args.omegas)):
        parser.error('mesh entries and finite split parameters must be positive; grid level is 0..9')
    args.output.mkdir(parents=True, exist_ok=False)
    if args.input_json is not None:
        record = json.loads(args.input_json.read_text())
        spec = record.get('input', record)
        input_hash = hashlib.sha256(args.input_json.read_bytes()).hexdigest()
    else:
        root = Path(__file__).resolve().parents[2]
        library = root/'python/vibeqc/basis_library/basis'
        paths = [library/'pob-tzvp-rev2.g94', library/'def2-svp-jk.g94']
        a_ang, symbols, fractional = (
            (5.431, ('Si', 'Si'), (.25, .25, .25)) if args.system == 'si'
            else (4.81, ('Ca', 'O'), (.5, 0., 0.))
        )
        a = a_ang/.529177210903
        spec = dict(
            atoms=[(symbols[0], [0., 0., 0.]), (symbols[1], (a*np.asarray(fractional)).tolist())],
            lattice=(.5*a*np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])).tolist(),
            basis={s: parse_gaussian.load(str(paths[0]), s, optimize=False) for s in sorted(set(symbols))},
            auxiliary={s: parse_gaussian.load(str(paths[1]), s, optimize=False) for s in sorted(set(symbols))},
            source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        )
        input_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if str(spec.get('unit', 'Bohr')).lower() != 'bohr' or 'ecp' in spec:
        raise ValueError('This reference generator requires an all-electron input in Bohr')
    cell = gto.M(atom=spec['atoms'], a=spec['lattice'], basis=spec['basis'],
                 unit='Bohr', precision=1e-11, verbose=4,
                 output=str(args.output/'pyscf.out'))
    kpoints = cell.make_kpts(args.mesh)
    summary = dict(program='PySCF', version=pyscf.__version__, input=spec,
        input_sha256=input_hash,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        mesh=args.mesh, unit='Bohr', precision=1e-11, threshold=1e-9,
        auxiliary_normalization='unit-square', overlap_diagonal_normalization=True,
        overlap_threshold=1e-7, fit_omega=args.omegas[0],
        nbf=int(cell.nao), n_electrons=int(cell.nelectron), runs=[])

    def save_summary():
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        sections = {
            'program': dict(name='PySCF', version=pyscf.__version__),
            'validation': dict(role='independent reference',
                execution_boundary='separate external process', native_parity='not evaluated'),
            'input': dict(sha256=input_hash, mesh=args.mesh,
                script_sha256=summary['script_sha256']),
        }
        lines = []
        for section, values in sections.items():
            lines.extend([f'[{section}]', *(f'{json.dumps(k)} = {json.dumps(v)}'
                for k, v in values.items()), ''])
        for run in summary['runs']:
            lines.extend(['[[run]]', *(f'{json.dumps(k)} = {json.dumps(run[k])}'
                for k in ('method', 'converged', 'energy', 'elapsed_s', 'points')), ''])
        (args.output/'pyscf.system').write_text('\n'.join(lines))

    def fit_at(omega):
        fit = RSGDF(cell, kpoints)
        fit.auxbasis = spec['auxiliary']
        fit.exp_to_discard = 0
        fit.omega = fit.omega_j2c = omega
        fit.precision_R = fit.precision_G = 1e-11
        fit.precision_j2c = 1e-13
        fit.j2c_eig_always = True
        fit.linear_dep_threshold = 1e-9
        fit.max_memory = 8000
        fit._cderi_to_save = str(args.output/f'fit-{omega:g}.h5')
        return fit

    def canonical(h, s):
        scale = 1/np.sqrt(np.diag(s).real)
        eigenvalues, vectors = np.linalg.eigh(s*np.outer(scale, scale))
        keep = eigenvalues >= 1e-7
        x = scale[:, None]*(vectors[:, keep]/np.sqrt(eigenvalues[keep]))
        energies, coefficients = np.linalg.eigh(x.conj().T@h@x)
        return energies, x@coefficients

    fit = fit_at(args.omegas[0])
    density = None
    save_summary()
    for method in args.methods:
        started = time.monotonic()
        mf = scf.KRHF(cell, kpoints) if method == 'hf' else dft.KRKS(cell, kpoints)
        mf.with_df = fit
        mf.exxdiv = 'ewald'
        mf._eigh = canonical
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-10, 1e-7, 100
        if method == 'pbe':
            mf.xc = 'pbe'
            mf.grids = dft.gen_grid.BeckeGrids(cell)
            mf.grids.level = args.grid_level
            mf.grids.build()
            np.savez_compressed(args.output/f'grid-{args.grid_level}.npz',
                                coordinates=mf.grids.coords, weights=mf.grids.weights)
        overlap, hcore = mf.get_ovlp(), mf.get_hcore()
        if density is None:
            densities = []
            for s, h in zip(overlap, hcore):
                _, coefficients = canonical(h, s)
                if coefficients.shape[1] < cell.nelectron//2:
                    raise ValueError('Retained orbital space cannot hold the reference electrons')
                occupied = coefficients[:, :cell.nelectron//2]
                densities.append(2*occupied@occupied.conj().T)
            density = np.asarray(densities)
            np.savez_compressed(args.output/'initial-density.npz', D=density)
        energy = mf.kernel(dm0=density)
        item = dict(method=method, energy=float(energy), converged=bool(mf.converged),
            points=0 if method == 'hf' else len(mf.grids.weights),
            grid_level=None if method == 'hf' else args.grid_level,
            elapsed_s=time.monotonic()-started,
            components={k: float(v) for k, v in mf.scf_summary.items()})
        summary['runs'].append(item)
        save_summary()
        if not mf.converged:
            raise RuntimeError(f'{method} reference did not converge; result is not an acceptance pin')
        density = mf.make_rdm1()
        electrons = np.einsum('kij,kji->', density, overlap).real/len(kpoints)
        if abs(electrons-cell.nelectron) > 1e-8:
            raise ValueError('Accepted reference density has the wrong electron count')
        potential = mf.get_veff(cell, density)
        jmat = fit.get_jk(density, kpts=kpoints, with_k=False)[0]
        np.savez_compressed(args.output/f'components-{method}.npz', kpoints=kpoints,
            S=overlap, H=hcore, D=density, J=jmat, Vextra=np.asarray(potential)-jmat,
            F=mf.get_fock(dm=density, vhf=potential), energies=mf.mo_energy,
            coefficients=mf.mo_coeff, occupations=mf.mo_occ)
        print(item, flush=True)

    for index, omega in enumerate(args.omegas):
        if index:
            Path(fit._cderi_to_save).unlink()
            fit = fit_at(omega)
        jmat, kmat = fit.get_jk(density, kpts=kpoints, exxdiv=None)
        _, kewald = fit.get_jk(density, kpts=kpoints, exxdiv='ewald')
        np.savez_compressed(args.output/f'same-density-{omega:g}.npz',
                            D=density, J=jmat, K=kmat, K_ewald=kewald)
        print('same-density split', omega, flush=True)
    Path(fit._cderi_to_save).unlink()


if __name__ == '__main__':
    main()
