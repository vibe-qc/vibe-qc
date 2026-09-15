"""Generate an independent periodic ECP reference in a PySCF process.

Run this script with the external PySCF interpreter, never by importing
PySCF into vibe-qc. It uses the repository's exact LANL2DZ basis/ECP and
auxiliary records. Each result retains the represented orbital space,
initial and accepted densities, component matrices and a reference manifest.

Example (use a fresh output directory):
    "$VIBEQC_PYSCF_PYTHON" examples/regression/periodic_ecp_reference.py \
        --output /tmp/nacl-ecp-reference

References: Hay and Wadt, JCP 82, 270 (1985), DOI 10.1063/1.448799;
Ye and Berkelbach, JCP 154, 131104 (2021), DOI 10.1063/5.0046617.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def _sidecar_nwchem(path: Path, symbol: str) -> str:
    """Convert one already-bundled Gaussian ECP record without altering it."""
    lines = [line.split('!', 1)[0].strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    first = next(i for i, line in enumerate(lines) if line.split() == [symbol.upper(), '0'])
    _, maximum, ncore = lines[first + 1].split()
    position = first + 2
    output = [f'{symbol} nelec {ncore}']
    for channel in range(int(maximum) + 1):
        label = lines[position].split()[0]
        count = int(lines[position + 1])
        output.append(f'{symbol} ' + ('ul' if channel == 0 else label.split('-')[0]))
        output.extend(lines[position + 2:position + 2 + count])
        position += count + 2
    return '\n'.join(output)


def _manifest(folder, metadata, summary):
    """Manifest of this external reference, not a native vibe-qc run."""
    sections = {
        'program': {'name': 'PySCF', 'version': metadata['version']},
        'validation': {
            'role': 'independent reference',
            'execution_boundary': 'separate external process',
            'native_parity': 'not evaluated by this generator',
        },
        'run': {
            'method': 'KRHF', 'converged': summary['converged'],
            'energy_hartree': summary['energy'],
            'elapsed_seconds': summary['elapsed_s'],
            'script_sha256': metadata['script_sha256'],
        },
        'source_sha256': metadata['source_sha256'],
    }
    text = []
    for section, values in sections.items():
        text.append(f'[{section}]')
        for key, value in values.items():
            text.append(f'{json.dumps(key)} = {json.dumps(value)}')
        text.append('')
    (folder / 'pyscf.system').write_text('\n'.join(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mesh', nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument('--projector-components', action='store_true',
                        help='evaluate eight isolated image-projector matrices without SCF')
    parser.add_argument('--displacements', nargs='+', type=float, default=[0., .17],
                        help='Cl z displacements in bohr, including the reference zero')
    args = parser.parse_args()
    if any(n < 1 for n in args.mesh):
        parser.error('mesh dimensions must be positive')

    # These imports belong exclusively to the external reference process.
    import numpy as np
    import pyscf
    from pyscf.gto.basis import parse_gaussian
    from pyscf.pbc import gto, scf
    from pyscf.pbc.gto import ecp as pbc_ecp
    from pyscf.pbc.df.rsdf import RSGDF

    if not np.isfinite(args.displacements).all():
        parser.error('displacements must be finite')
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2] / 'python/vibeqc/basis_library/basis'
    paths = [source / name for name in ('lanl2dz.g94', 'lanl2dz.ecp', 'def2-svp-jk.g94')]
    basis = {s: parse_gaussian.load(str(paths[0]), s, optimize=False) for s in ('Na', 'Cl')}
    auxiliary = {s: parse_gaussian.load(str(paths[2]), s, optimize=False) for s in basis}
    ecp_text = {s: _sidecar_nwchem(paths[1], s) for s in basis}
    ecp = {s: pyscf.gto.basis.parse_ecp(text) for s, text in ecp_text.items()}
    for symbol in basis:
        if ecp[symbol] != pyscf.gto.basis.load_ecp('lanl2dz', symbol):
            raise ValueError(f'{symbol}: converted sidecar differs from independent LANL2DZ')
    metadata = {
        'program': 'PySCF', 'version': pyscf.__version__, 'numpy': np.__version__,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'basis': basis, 'auxiliary': auxiliary, 'ecp': ecp, 'ecp_nwchem': ecp_text,
        'n_threads': pyscf.lib.num_threads(), 'precision': 1e-11, 'omega': .6,
        'overlap_threshold': 1e-7, 'overlap_diagonal_normalization': True,
        'fit_threshold': 1e-9, 'exxdiv': 'ewald', 'unit': 'Bohr', 'mesh': args.mesh,
    }
    (args.output / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    a = 5.64 * 1.8897261246257702
    lattice = np.array([[0., a/2, a/2], [a/2, 0., a/2], [a/2, a/2, 0.]])

    if args.projector_components:
        # The extra AO belongs to the sole real atom carrying the ECP. All
        # bra/ket functions are ghosts, so no second projector is introduced.
        atoms = [('Na', [0., 0., 0.]), ('Cl', [a/2, a/2, a/2])]
        matrices, cases = {}, []
        image_pairs = [(0, 0), (1, 0), (1, 2), (2, 3)]
        for atom, (symbol, center) in enumerate(atoms):
            for index, (ket_image, projector_image) in enumerate(image_pairs):
                ket_shift = ket_image * lattice[0]
                projector = np.asarray(center) + projector_image * lattice[0]
                molecule_atoms = [(symbol, projector)]
                molecule_basis = {symbol: [[0, [1., 1.]]]}
                for copy, shift in enumerate((np.zeros(3), ket_shift)):
                    for site, (element, origin) in enumerate(atoms):
                        label = f'ghost-{element}{2*copy + site + 1}'
                        molecule_atoms.append((label, np.asarray(origin) + shift))
                        molecule_basis[label] = basis[element]
                molecule = pyscf.gto.M(
                    atom=molecule_atoms, basis=molecule_basis, ecp={symbol: ecp[symbol]},
                    unit='Bohr', spin=1, verbose=0,
                )
                if molecule.nao_nr() != 33:
                    raise ValueError('Projector witness requires 16 bra and 16 ket AOs')
                name = f'case_{atom}_{index}'
                matrices[name] = molecule.intor('ECPscalar_sph')[1:17, 17:33]
                cases.append(dict(name=name, projector_atom=atom,
                                  projector_center=projector.tolist(),
                                  ket_shift=ket_shift.tolist()))
        np.savez_compressed(args.output / 'matrices.npz', **matrices)
        metadata.update(atoms=atoms, lattice=lattice.tolist(), cases=cases,
                        scope='isolated image-projector matrices; no SCF')
        (args.output / 'input.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return 0

    def canonical_eigh(h, s):
        scale = 1 / np.sqrt(np.diag(s).real)
        eigenvalues, eigenvectors = np.linalg.eigh(s * np.outer(scale, scale))
        keep = eigenvalues >= 1e-7
        x = scale[:, None] * (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))
        energies, rotation = np.linalg.eigh(x.conj().T @ h @ x)
        return energies, x @ rotation

    for index, displacement in enumerate(args.displacements):
        start = time.monotonic()
        folder = args.output / f'geometry-{index}'
        folder.mkdir()
        atoms = [('Na', [0., 0., 0.]), ('Cl', [a/2, a/2, a/2 + displacement])]
        cell = gto.M(atom=atoms, a=lattice, basis=basis, ecp=ecp, unit='Bohr',
                     precision=1e-11, verbose=4, output=str(folder / 'pyscf.out'))
        if cell.nelectron != 8 or not np.array_equal(cell.atom_charges(), [1, 7]):
            raise ValueError('Reference ECP frame must contain 8 valence electrons and Z_eff=(1,7)')
        kpts = cell.make_kpts(args.mesh)
        fit = RSGDF(cell, kpts)
        fit.auxbasis = auxiliary
        fit.exp_to_discard = 0.
        fit.omega = fit.omega_j2c = .6
        fit.precision_R = fit.precision_G = 1e-11
        fit.precision_j2c = 1e-13
        fit.j2c_eig_always = True
        fit.linear_dep_threshold = 1e-9
        fit.max_memory = 8000
        fit._cderi_to_save = str(folder / 'fit.h5')
        mf = scf.KRHF(cell, kpts, exxdiv='ewald')
        mf.with_df = fit
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-11, 1e-8, 100
        mf._eigh = canonical_eigh
        overlap = mf.get_ovlp()
        kinetic = np.asarray(cell.pbc_intor('int1e_kin', kpts=kpts))
        nuclear = np.asarray(fit.get_nuc(kpts))
        projectors = np.asarray(pbc_ecp.ecp_int(cell, kpts))
        hcore = kinetic + nuclear + projectors
        density = []
        for s, h in zip(overlap, hcore):
            _, coefficients = canonical_eigh(h, s)
            occupied = coefficients[:, :4]
            density.append(2 * occupied @ occupied.conj().T)
        density = np.asarray(density)
        j, k = fit.get_jk(density, kpts=kpts, exxdiv=None)
        _, k_ewald = fit.get_jk(density, kpts=kpts, exxdiv='ewald')
        arrays = dict(kpoints=kpts, S=overlap, T=kinetic, V_nuclear=nuclear,
                      V_ecp=projectors, H=hcore, D_initial=density,
                      J_initial=j, K_initial=k, K_ewald_initial=k_ewald)
        np.savez_compressed(folder / 'components.npz', **arrays)
        energy = mf.kernel(dm0=density)
        accepted = mf.make_rdm1()
        j, k = mf.get_jk(dm_kpts=accepted)
        arrays.update(D=accepted, J=j, K=k, F=mf.get_fock(dm=accepted),
                      mo_energies=mf.mo_energy, mo_coeffs=mf.mo_coeff)
        np.savez_compressed(folder / 'components.npz', **arrays)
        summary = dict(
            energy=float(energy), converged=bool(mf.converged), atoms=atoms,
            lattice=lattice.tolist(), nuclear=float(cell.energy_nuc()),
            nao=int(cell.nao_nr()), naux=int(fit.auxcell.nao_nr()),
            displacement_bohr=displacement, elapsed_s=time.monotonic() - start,
            components={key: float(value) for key, value in mf.scf_summary.items()},
        )
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        _manifest(folder, metadata, summary)
        print(json.dumps(summary), flush=True)
        if not mf.converged:
            raise RuntimeError('External SCF did not converge; results are not an accepted reference')
        Path(fit._cderi_to_save).unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
