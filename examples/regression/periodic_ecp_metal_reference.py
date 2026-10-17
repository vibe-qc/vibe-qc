"""Generate an independent periodic ECP metal reference in a PySCF process.

Run this script with the external PySCF interpreter, never by importing
PySCF into vibe-qc (CLAUDE.md section 10). It builds the P09 Ag-fcc /
Au-fcc / W-bcc cells that ``tests/test_periodic_ecp.py`` uses, with the
repository's own pob-TZVP-rev2 orbital records, its own ``.ecp`` sidecar
and its own def2-tzvp-jk auxiliary records, and runs KRKS/PBE on PySCF's
Gaussian density fitting (the modrho/compensated-charge fit vibe-qc's
periodic GDF route reproduces).

Example (use a fresh output directory):
    "$VIBEQC_PYSCF_PYTHON" examples/regression/periodic_ecp_metal_reference.py \
        --system w-bcc --output /tmp/w-bcc-ecp-reference

Unlike ``periodic_ecp_reference.py`` (NaCl / LANL2DZ) this generator cannot
assert that the converted sidecar equals an independent PySCF library
record: the pob-TZVP-rev2 records carry the Stuttgart/Cologne MDF potentials
distributed with that basis, which are a *different published parameter set*
from the ECPs PySCF ships under ``def2-*`` for the same elements, and they
are written with a zero local channel instead of a highest-channel local.
Faking that equality check would be worse than not having it, so the
generator asserts the structural invariants it can actually check instead:
the core size, the effective charge, the per-channel term counts and the
valence electron count, each against the bundled CRYSTAL ``Z+200`` source.

Conventions, and why each one is pinned rather than defaulted:

* **ECP operator.** Shaw and Hill, JCP 147, 074108 (2017),
  DOI 10.1063/1.4986887, Eqs. (1)-(2): ``U = U_L + sum_l |S_lm> U_l <S_lm|``
  with the local ``U_L`` acting on every angular momentum. The bundled pob
  records set ``U_L = 0`` and tabulate absolute projector channels, so the
  operator vanishes above the highest tabulated channel. PySCF reads the
  same thing once the sidecar is converted to NWChem (``ul`` = the zero
  channel), which is why both codes must be fed *this* record.
* **Density fitting.** ``pyscf.pbc.df.rsdf.RSGDF`` with ``omega = 0.4``,
  because that is what the route under test actually runs:
  ``run_krks_periodic_gdf`` defaults to ``gdf_method='rsgdf'`` and
  ``rsgdf_omega=0.4`` (python/vibeqc/periodic_k_gdf.py:7525-7530), over the
  modrho auxiliary basis that ``vibeqc.aux_basis.make_modrho_aux_basis``
  builds. Sun et al., JCP 147, 164119 (2017), DOI 10.1063/1.4998644;
  Ye and Berkelbach, JCP 154, 131104 (2021), DOI 10.1063/5.0046617. The
  ``rsgdf_tail_ke_cutoff`` approximation the issue warns about is ``None``
  by default, i.e. the exact tail, so an absolute comparison is meaningful
  here.
* **Brillouin-zone sampling.** Gamma-centred by default, and written out as
  explicit fractional coordinates rather than a mesh shape. The multi-k GDF
  drivers' tuple form resolves its mesh through
  ``_mp_native(system, mesh, [0, 0, 0], False)``
  (python/vibeqc/periodic_k_gdf.py:2238) -- a zero shift, i.e. PySCF's
  ``make_kpts`` convention. ``KPoints.monkhorst_pack``'s *classical* default
  is the half-step-shifted grid instead, and kpoints.py:224-239 records the
  two being about 44 mHa apart on MgO/STO-3G at (2,2,2). The consumer
  asserts its own k-points against the recorded list rather than trusting
  either default.
* **Linear dependence.** Canonical orthogonalisation of the
  diagonal-normalised overlap at ``1e-7``, matching
  ``run_krks_periodic_gdf``'s ``linear_dep_threshold`` default, and
  ``linear_dep_threshold = 1e-9`` on the two-centre fit metric, matching
  ``gdf_linear_dep_threshold``.
* **Smearing.** Fermi-Dirac at ``kT = --sigma`` Hartree, matching
  ``PeriodicKSOptions.smearing_temperature``. The pinned number is
  ``mf.e_tot``, which is the plain total energy; PySCF keeps the free energy
  and the zero-temperature extrapolation separately as ``e_free`` and
  ``e_zero``, exactly as vibe-qc keeps ``free_energy = E_total - T*S`` apart
  from ``result.energy``. All three are recorded.

References: Shaw and Hill, JCP 147, 074108 (2017), DOI 10.1063/1.4986887;
Sun et al., JCP 147, 164119 (2017), DOI 10.1063/1.4998644;
Figgen et al., Chem. Phys. 311, 227 (2005), DOI 10.1016/j.chemphys.2004.10.005;
Figgen et al., JCP 130, 164108 (2009), DOI 10.1063/1.3119665;
Vilela Oliveira et al., J. Comput. Chem. 40, 2364 (2019), DOI 10.1002/jcc.26013.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

ANGSTROM_IN_BOHR = 1.8897261246257702

# Geometry and ECP bookkeeping mirrored from tests/test_periodic_ecp.py's
# _build_ag_fcc / _build_au_fcc / _build_w_bcc. ``published_ncore`` is the
# core size the originating pseudopotential paper states, kept here so the
# generator refuses a silently different record rather than reporting an
# energy for the wrong frame.
SYSTEMS = {
    'ag-fcc': dict(
        symbol='Ag', z=47, lattice_constant_angstrom=4.062,
        fractional=[[0., 0., 0.], [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]],
        published_ncore=28, published_zeff=19,
        ecp_source='Figgen 2005, DOI 10.1016/j.chemphys.2004.10.005 (1s-3d core)',
    ),
    'au-fcc': dict(
        symbol='Au', z=79, lattice_constant_angstrom=4.063,
        fractional=[[0., 0., 0.], [0., .5, .5], [.5, 0., .5], [.5, .5, 0.]],
        published_ncore=60, published_zeff=19,
        ecp_source='Figgen 2005, DOI 10.1016/j.chemphys.2004.10.005 (1s-4f core)',
    ),
    'w-bcc': dict(
        symbol='W', z=74, lattice_constant_angstrom=3.191,
        fractional=[[0., 0., 0.], [.5, .5, .5]],
        published_ncore=60, published_zeff=14,
        ecp_source='Figgen 2009, DOI 10.1063/1.3119665 (5d elements Hf-Pt)',
    ),
}


def _sidecar_nwchem(path: Path, symbol: str) -> str:
    """Convert one already-bundled Gaussian ECP record without altering it.

    Same reader as ``periodic_ecp_reference.py``, with a case-insensitive
    element match: the pob sidecar writes ``Ag``/``Au``/``W`` where the
    LANL2DZ one writes the symbol upper-cased.
    """
    lines = [line.split('!', 1)[0].strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    first = next(i for i, line in enumerate(lines)
                 if [field.upper() for field in line.split()] == [symbol.upper(), '0'])
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


def _crystal_source_header(path: Path) -> tuple[float, list[int]]:
    """Read ``Z_eff`` and the per-channel term counts from a CRYSTAL record.

    The bundled per-element sources open with the CRYSTAL ``Z+200`` conformer
    line, ``INPUT``, then ``ZNUC M0 M1 ... Mn`` (CRYSTAL23 manual,
    eqs. 3.18-3.19). Reading it gives an independent witness for the sidecar:
    the two are produced by different code paths from the same upstream data.
    """
    lines = [line.strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    marker = next(i for i, line in enumerate(lines) if line.upper() == 'INPUT')
    fields = lines[marker + 1].split()
    return float(fields[0]), [int(field) for field in fields[1:]]


def _manifest(folder: Path, metadata: dict, summary: dict) -> None:
    """Manifest of this external reference, not a native vibe-qc run."""
    sections = {
        'program': {'name': 'PySCF', 'version': metadata['version']},
        'validation': {
            'role': 'independent reference',
            'execution_boundary': 'separate external process',
            'native_parity': 'not evaluated by this generator',
        },
        'run': {
            'method': 'KRKS', 'functional': metadata['xc'],
            'fit': f'RSGDF (omega={metadata["omega"]})',
            'converged': summary['converged'],
            'energy_hartree': summary['energy'],
            'free_energy_hartree': summary['free_energy'],
            'elapsed_seconds': summary['elapsed_s'],
            'script_sha256': metadata['script_sha256'],
        },
        'source_sha256': metadata['source_sha256'],
    }
    text: list[str] = []
    for section, values in sections.items():
        text.append(f'[{section}]')
        for key, value in values.items():
            text.append(f'{json.dumps(key)} = {json.dumps(value)}')
        text.append('')
    (folder / 'pyscf.system').write_text('\n'.join(text))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--system', choices=sorted(SYSTEMS), required=True)
    parser.add_argument('--mesh', nargs=3, type=int, default=[2, 2, 2])
    parser.add_argument('--shifted', action='store_true',
                        help='classical half-step-shifted Monkhorst-Pack grid '
                             'instead of the Gamma-centred one the multi-k GDF '
                             'drivers sample for a tuple kmesh')
    parser.add_argument('--omega', type=float, default=.4,
                        help='RSGDF range-separation parameter; must match '
                             'run_krks_periodic_gdf(rsgdf_omega=...)')
    parser.add_argument('--fit-linear-dep', type=float, default=1e-9,
                        help='matches gdf_linear_dep_threshold')
    parser.add_argument('--overlap-linear-dep', type=float, default=1e-7,
                        help='matches linear_dep_threshold')
    parser.add_argument('--sigma', type=float, default=.005,
                        help='Fermi-Dirac kT in Hartree; must match '
                             'PeriodicKSOptions.smearing_temperature')
    parser.add_argument('--xc', default='pbe,pbe')
    parser.add_argument('--grid-level', type=int, default=3,
                        help='PySCF Becke grid level; vibe-qc integrates the '
                             'periodic XC on an atom-partitioned Becke grid, '
                             'so a uniform FFT grid would compare two '
                             'different quadratures')
    parser.add_argument('--precision', type=float, default=1e-8)
    parser.add_argument('--max-cycle', type=int, default=100)
    parser.add_argument('--cderi', type=Path, default=None,
                        help='reuse a saved RSGDF fit (fit.h5) from an earlier '
                             'run of this script on the SAME system, mesh and '
                             'fit settings, instead of rebuilding it; the '
                             'reused path is recorded in the manifest')
    args = parser.parse_args()
    if any(n < 1 for n in args.mesh):
        parser.error('mesh dimensions must be positive')
    if args.sigma < 0:
        parser.error('sigma must be >= 0')

    # These imports belong exclusively to the external reference process.
    import numpy as np
    import pyscf
    from pyscf.gto.basis import parse_gaussian
    from pyscf.pbc import dft as pbc_dft
    from pyscf.pbc import gto as pbc_gto
    from pyscf.pbc.df.rsdf import RSGDF
    from pyscf.pbc.dft import gen_grid as pbc_gen_grid
    from pyscf.pbc.scf import addons as pbc_addons

    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    library = root / 'python/vibeqc/basis_library'
    spec = SYSTEMS[args.system]
    symbol = spec['symbol']
    paths = [library / 'basis/pob-tzvp-rev2.g94',
             library / 'basis/pob-tzvp-rev2.ecp',
             library / 'basis/def2-tzvp-jk.g94',
             library / f'sources/pob-TZVP-rev2/{spec["z"]:02d}_{symbol}']

    basis = {symbol: parse_gaussian.load(str(paths[0]), symbol, optimize=False)}
    auxiliary = {symbol: parse_gaussian.load(str(paths[2]), symbol, optimize=False)}
    ecp_text = {symbol: _sidecar_nwchem(paths[1], symbol)}
    ecp = {symbol: pyscf.gto.basis.parse_ecp(ecp_text[symbol])}

    # Structural agreement between the sidecar, the CRYSTAL source and the
    # published core size. This replaces the LANL2DZ generator's
    # converted-equals-library check, which has no counterpart for pob.
    source_zeff, source_counts = _crystal_source_header(paths[3])
    if int(source_zeff) != spec['published_zeff']:
        raise ValueError(f'{symbol}: CRYSTAL source Z_eff {source_zeff} is not '
                         f'the published {spec["published_zeff"]}')
    if spec['z'] - int(source_zeff) != spec['published_ncore']:
        raise ValueError(f'{symbol}: CRYSTAL source implies a core of '
                         f'{spec["z"] - int(source_zeff)}, not the published '
                         f'{spec["published_ncore"]} ({spec["ecp_source"]})')
    if ecp[symbol][0] != spec['published_ncore']:
        raise ValueError(f'{symbol}: converted sidecar declares '
                         f'{ecp[symbol][0]} core electrons, not '
                         f'{spec["published_ncore"]}')
    if source_counts[0] != 0:
        raise ValueError(f'{symbol}: CRYSTAL source carries {source_counts[0]} '
                         'local terms; this generator and libecpint both '
                         'assume the pob zero-local convention')
    # PySCF stores channels as (l, [terms-by-r-power]); count the primitives
    # per channel and compare with the CRYSTAL per-channel counts M1..Mn.
    converted_counts = []
    for angular, powers in ecp[symbol][1]:
        if angular < 0:  # the local channel; zero by the check above
            continue
        converted_counts.append(sum(len(block) for block in powers))
    # A CRYSTAL record may declare a trailing channel with zero terms (Ag's
    # record ends `... 2 0`, an empty g channel); PySCF's parser simply omits
    # an empty channel, so compare with trailing empties trimmed off both.
    def _trim(counts: list[int]) -> list[int]:
        trimmed = list(counts)
        while trimmed and trimmed[-1] == 0:
            trimmed.pop()
        return trimmed

    if _trim(converted_counts) != _trim(source_counts[1:]):
        raise ValueError(f'{symbol}: converted channels {converted_counts} do '
                         f'not match the CRYSTAL source {source_counts[1:]}')

    metadata = {
        'program': 'PySCF', 'version': pyscf.__version__, 'numpy': np.__version__,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'system': args.system, 'symbol': symbol, 'z': spec['z'],
        'ecp_source': spec['ecp_source'],
        'published_ncore': spec['published_ncore'],
        'published_zeff': spec['published_zeff'],
        'basis': basis, 'auxiliary': auxiliary, 'ecp': ecp,
        'ecp_nwchem': ecp_text,
        'n_threads': pyscf.lib.num_threads(),
        'precision': args.precision, 'xc': args.xc, 'sigma': args.sigma,
        'grid_level': args.grid_level, 'unit': 'Bohr', 'mesh': args.mesh,
        'gamma_centred': not bool(args.shifted),
        'fit': 'RSGDF', 'omega': args.omega,
        'fit_linear_dep_threshold': args.fit_linear_dep,
        'overlap_linear_dep_threshold': args.overlap_linear_dep,
        'lattice_constant_angstrom': spec['lattice_constant_angstrom'],
        'exp_to_discard': 0.,
    }

    a = spec['lattice_constant_angstrom'] * ANGSTROM_IN_BOHR
    lattice = np.eye(3) * a
    atoms = [(symbol, (np.asarray(f) * a).tolist()) for f in spec['fractional']]
    start = time.monotonic()
    cell = pbc_gto.M(atom=atoms, a=lattice, basis=basis, ecp=ecp, unit='Bohr',
                     precision=args.precision, verbose=4,
                     output=str(args.output / 'pyscf.out'))
    expected_electrons = spec['published_zeff'] * len(atoms)
    if cell.nelectron != expected_electrons:
        raise ValueError(f'{args.system}: ECP frame holds {cell.nelectron} '
                         f'valence electrons, not {expected_electrons}')
    if not np.array_equal(cell.atom_charges(),
                          [spec['published_zeff']] * len(atoms)):
        raise ValueError(f'{args.system}: Z_eff frame is '
                         f'{cell.atom_charges().tolist()}')

    kpts = cell.make_kpts(args.mesh, with_gamma_point=not bool(args.shifted))
    scaled = cell.get_scaled_kpts(kpts)

    def canonical_eigh(h, s, x=None, **_ignored):
        """Canonical orthogonalisation of the diagonal-normalised overlap.

        Same construction as ``periodic_ecp_reference.py``, at the threshold
        ``run_krks_periodic_gdf`` uses for ``linear_dep_threshold``.

        PySCF 2.14's ``khf.KSCF.eig`` hands ``_eigh`` a precomputed
        orthogonaliser as ``x=``; it is built with PySCF's own threshold, so
        it is accepted and ignored in favour of the matched one here.
        """
        scale = 1 / np.sqrt(np.diag(s).real)
        eigenvalues, eigenvectors = np.linalg.eigh(s * np.outer(scale, scale))
        keep = eigenvalues >= args.overlap_linear_dep
        x = scale[:, None] * (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))
        energies, rotation = np.linalg.eigh(x.conj().T @ h @ x)
        return energies, x @ rotation

    fit = RSGDF(cell, kpts)
    fit.auxbasis = auxiliary
    fit.exp_to_discard = 0.
    fit.omega = fit.omega_j2c = args.omega
    fit.precision_R = fit.precision_G = args.precision
    fit.linear_dep_threshold = args.fit_linear_dep
    if args.cderi is not None:
        if not args.cderi.is_file():
            raise FileNotFoundError(f'--cderi {args.cderi} does not exist')
        fit._cderi = str(args.cderi)
        fit.build(with_j3c=False)
    else:
        fit._cderi_to_save = str(args.output / 'fit.h5')
        fit.build()

    mf = pbc_dft.KRKS(cell, kpts)
    mf._eigh = canonical_eigh
    mf.xc = args.xc
    mf.with_df = fit
    mf.grids = pbc_gen_grid.BeckeGrids(cell)
    mf.grids.level = args.grid_level
    mf.grids.build()
    mf.conv_tol = 1e-9
    mf.max_cycle = args.max_cycle
    if args.sigma > 0:
        mf = pbc_addons.smearing_(mf, sigma=args.sigma, method='fermi')

    energy = mf.kernel()
    summary = dict(
        system=args.system, symbol=symbol, atoms=atoms,
        lattice=lattice.tolist(), mesh=list(args.mesh),
        gamma_centred=not bool(args.shifted),
        kpoints_fractional=scaled.tolist(),
        kpoints_cartesian=np.asarray(kpts).tolist(),
        energy=float(energy),
        free_energy=float(getattr(mf, 'e_free', energy)),
        zero_temperature_energy=float(getattr(mf, 'e_zero', energy)),
        entropy=float(getattr(mf, 'entropy', 0.) or 0.),
        sigma=args.sigma, xc=args.xc, grid_level=args.grid_level,
        n_grid_points=int(mf.grids.coords.shape[0]),
        converged=bool(mf.converged),
        nuclear=float(cell.energy_nuc()),
        nao=int(cell.nao_nr()), naux=int(fit.auxcell.nao_nr()),
        n_electrons=int(cell.nelectron), n_atoms=len(atoms),
        fit='RSGDF', omega=args.omega,
        fit_linear_dep_threshold=args.fit_linear_dep,
        overlap_linear_dep_threshold=args.overlap_linear_dep,
        precision=args.precision,
        reused_cderi=None if args.cderi is None else str(args.cderi),
        elapsed_s=time.monotonic() - start,
        components={key: float(value) for key, value in mf.scf_summary.items()},
    )
    (args.output / 'input.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    _manifest(args.output, metadata, summary)
    print(json.dumps(summary), flush=True)
    if args.cderi is None:
        Path(fit._cderi_to_save).unlink(missing_ok=True)
    if not mf.converged:
        raise RuntimeError('External SCF did not converge; '
                           'results are not an accepted reference')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
