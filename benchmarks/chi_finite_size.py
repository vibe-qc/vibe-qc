#!/usr/bin/env python3
"""Fixed-density diagnosis of an isolated RHF chi checkpoint; never run SCF.

Separate auxiliary projection, reciprocal cutoff and BvK exchange effects.
The Fourier check shares the native AO-pair transform with the source, but
does not use its auxiliary basis, metric or factors. This is not an external
acceptance oracle. See chi_finite_size.md for assumptions and measured cases.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import erfc


def ewald_hessian(lattice):
    """Hessian of v_Ewald(r)-1/r at zero, including the background.

    Ewald (1921), IV.1, Eqs. 37-40, doi:10.1002/andp.19213690304.
    Differentiate the real/reciprocal/self Ewald sums directly. For a cubic
    cell this is 4*pi/(3*V) I. No fit to an observed energy is made.
    """
    a = np.asarray(lattice)
    volume = abs(np.linalg.det(a))
    b = 2*np.pi*np.linalg.inv(a).T
    alpha = 1/min(np.linalg.norm(a, axis=0))
    radius = np.sqrt(-np.log(1e-16))

    def sphere(vectors, cutoff):
        bounds = np.ceil(cutoff*np.linalg.norm(np.linalg.inv(vectors), axis=1)).astype(int)
        if np.prod(2*bounds+1) > 2_000_000:
            raise MemoryError('diagnostic Ewald candidate budget exceeded')
        indices = np.array(list(itertools.product(*(range(-n, n+1) for n in bounds))))
        points = indices @ vectors.T
        norms = np.linalg.norm(points, axis=1)
        keep = (norms > 0) & (norms <= cutoff)
        return points[keep], norms[keep]

    r, rn = sphere(a, radius/alpha)
    u = r/rn[:, None]
    ex = np.exp(-(alpha*rn)**2)
    first = -erfc(alpha*rn)/rn**2-2*alpha/np.sqrt(np.pi)*ex/rn
    second = 2*erfc(alpha*rn)/rn**3+4*alpha/np.sqrt(np.pi)*ex/rn**2+4*alpha**3/np.sqrt(np.pi)*ex
    h = np.einsum('r,ra,rb->ab', second-first/rn, u, u)+np.eye(3)*sum(first/rn)
    g, gn = sphere(b, 2*alpha*radius)
    h -= 4*np.pi/volume*np.einsum('r,ra,rb->ab', np.exp(-gn**2/(4*alpha**2))/gn**2, g, g)
    h += 4*alpha**3/(3*np.sqrt(np.pi))*np.eye(3)
    np.testing.assert_allclose(np.trace(h), 4*np.pi/volume, atol=2e-13, rtol=0)
    return h


def diagnose(archive, cutoffs):
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic.chi.checkpoint import load_chi_hf_reference, _basis_record
    from vibeqc.periodic.chi.cc_zero_mode import reference_madelung
    from vibeqc.periodic.chi.canonical_mp2 import _spin_state
    from vibeqc.aux_basis import _classic_pair_cells, _classic_reciprocal_mesh, _ao_scales_for_rsgdf
    from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch

    ref = load_chi_hf_reference(archive, expected_method='RHF')
    hf = ref.hf_result
    physical = hf._chi_restart_physical
    g = physical['geometry']
    system = core.PeriodicSystem(3, g['lattice_bohr'],
        [core.Atom(x['Z'], x['xyz_bohr']) for x in g['nuclei']], g['charge'], g['multiplicity'])
    if g['charge'] != 0 or g['dimension'] != 3 or hf.fitted_operator.descriptor.source != 'classic':
        raise ValueError('diagnostic requires neutral 3D classic RHF')
    molecule = system.unit_cell_molecule()
    record = physical['basis']
    shells = [core.ShellInfo(x['atom_index'], x['l'], x['pure'], x['exponents'],
              x['coefficients'], x['origin']) for x in record['shells']]
    basis = core.BasisSet(molecule, shells, record['name'], True)
    if _basis_record(basis) != record:
        raise ValueError('diagnostic basis does not reproduce the archived normalized shells')
    if basis.nbasis > 32:
        raise MemoryError('molecular diagnostic is limited to 32 AOs')
    controls = physical['controls']['coulomb_controls']
    if controls['support_policy'] != 'physical_pairs':
        raise ValueError('diagnostic requires physical-pair support')
    cutoff = controls['pair_cutoff_bohr']
    images = _classic_pair_cells(system, basis, cutoff, 'physical_pairs', 2_000_000)
    origins = np.array([x.origin for x in basis.shells()])
    active = [r for r in images if np.any(np.linalg.norm(origins[:, None]-origins[None, :]-r, axis=2) <= cutoff)]
    if len(active) != 1 or np.linalg.norm(active[0]) > 1e-12:
        raise ValueError('isolated-cell diagnostic requires home-only physical AO pairs')
    s = np.asarray(core.compute_overlap(basis))
    np.testing.assert_allclose(hf.overlap, np.broadcast_to(s, np.shape(hf.overlap)), atol=1e-12, rtol=0)
    density = np.asarray(hf.density)
    np.testing.assert_allclose(density, np.broadcast_to(density[0], density.shape), atol=1e-11, rtol=0)
    if np.max(abs(density.imag)) > 1e-12:
        raise ValueError('home-only molecular comparison requires real density')
    d = density[0].real
    eri = core.compute_eri(basis)
    jm = .5*np.einsum('pqrs,qp,sr->', eri, d, d)
    km = -.25*np.einsum('prsq,qp,rs->', eri, d, d)
    positions = np.array([x['xyz_bohr'] for x in g['nuclei']])
    ecp = physical.get('ecp')
    charges = ([x['effective_charge'] for x in ecp['atoms']] if ecp else
               [x['Z'] for x in g['nuclei']])
    hm = np.asarray(core.compute_kinetic(basis))+core.compute_nuclear_with_charges(basis, positions, charges)
    if ecp:
        blocks, centers = [], []
        for projector in ecp['projectors']:
            ns, am, exponents, coefficients = zip(*projector['primitives'])
            block = core.ECPPrimitiveBlock()
            block.n_primitive = len(am); block.ams = am; block.ns = ns
            block.exponents = exponents; block.coefficients = coefficients
            blocks.append(block); centers.extend(projector['center_bohr'])
        hm += core.compute_ecp_matrix_from_primitives(basis, centers, blocks)
    enuc_m = sum(charges[i]*charges[j]/np.linalg.norm(positions[i]-positions[j])
                 for i in range(len(positions)) for j in range(i))
    one_m = np.einsum('pq,qp->', d, hm).real
    one_p = np.einsum('k,kpq,kqp->', hf.kpoint_weights, density, hf.hcore).real
    xi = reference_madelung(hf, _spin_state(hf, 'RHF', 1e-9))
    weights = np.asarray(hf.kpoint_weights)
    fitted_j = .5*np.einsum('k,kpq,kqp->', weights, density, hf.fitted_operator.coulomb(density)).real
    seam_energy = -.25*xi*np.trace(d@s@d@s)
    fitted_k = -.25*np.einsum('k,kpq,kqp->', weights, density, hf.fitted_operator.exchange(density)).real+seam_energy
    scale = _ao_scales_for_rsgdf(basis)
    pair_weights = scale[:, None]*scale[None, :]

    def pair_ft(vectors):
        return ao_pair_fourier_transform_bloch(basis, vectors, np.zeros((1, 3)), np.zeros(3))*pair_weights[:, :, None]

    np.testing.assert_allclose(pair_ft(np.zeros((1, 3)))[:, :, 0], s, atol=1e-12, rtol=0)
    volume = abs(np.linalg.det(system.lattice))
    rows = []
    for energy_cutoff in cutoffs:
        j, k = 0., 0.
        # For this explicitly checked home-only density, each canonical q
        # occurs Nk times in the double character sum. The remaining weight
        # is 1/Nk. This still samples genuinely nonzero transfer vectors.
        for q, pairs in hf.fitted_operator.descriptor.q_grouping:
            vectors = _classic_reciprocal_mesh(system, np.asarray(q), energy_cutoff, 100*1024**2, 2_000_000)
            vectors = vectors[np.sum(vectors**2, axis=1) > 1e-12]
            for start in range(0, len(vectors), 1024):
                gv = vectors[start:start+1024]
                w = 4*np.pi/(volume*np.sum(gv**2, axis=1))
                ft = pair_ft(gv)
                if np.linalg.norm(q) < 1e-12:
                    rho = np.einsum('pq,qpg->g', d, ft)
                    j += .5*np.dot(w, abs(rho)**2).real
                x = np.einsum('pr,rqg->pqg', d, ft)
                y = np.einsum('pr,rqg->pqg', d, ft.conj())
                k -= .25*len(pairs)/len(density)**2*np.einsum('g,pqg,qpg->', w, x, y).real
        k += seam_energy
        rows.append(dict(cutoff_ha=float(energy_cutoff), unfitted_j=float(j), unfitted_k=float(k),
            fitted_minus_unfitted_j=float(fitted_j-j), fitted_minus_unfitted_k=float(fitted_k-k),
            unfitted_exchange_minus_molecular=float(k-km),
            unfitted_electrostatic_minus_molecular=float(one_p-one_m+hf.e_nuclear-enuc_m+j-jm),
            unfitted_fixed_density_hf=float(one_p+hf.e_nuclear+j+k)))
    opts = core.LatticeSumOptions(); opts.cutoff_bohr = .01
    moments = core.compute_multipole_moments_lattice(basis, system, opts, 2)
    home = next(i for i, r in enumerate(moments.cells) if np.linalg.norm(r.r_cart) < 1e-12)
    m = np.asarray(moments.blocks[home])
    quadrupole = m[np.array([[4, 5, 6], [5, 7, 8], [6, 8, 9]])]
    dipole = m[1:4]
    spread = np.array([[np.trace(d@s@d@quadrupole[i, j])+np.trace(d@quadrupole[i, j]@d@s)
        -np.trace(d@dipole[i]@d@dipole[j])-np.trace(d@dipole[j]@d@dipole[i])
        for j in range(3)] for i in range(3)])
    hessian = ewald_hessian(np.asarray(system.lattice) @ np.diag(physical['mesh']))
    return dict(schema='chi-isolated-rhf-fixed-density-diagnostic/v1', acceptance='not-assessed',
        archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
        native_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        source_identity=hf.fitted_operator.identity, ecp_identity=(physical.get('ecp') or {}).get('identity'),
        mesh=physical['mesh'], cell_volume_bohr3=volume, hf_energy=float(hf.energy),
        molecular_fixed_density_hf=float(one_m+enuc_m+jm+km),
        molecular_j=float(jm), molecular_k=float(km), fitted_j=float(fitted_j), fitted_k=float(fitted_k),
        madelung_hartree=xi, ewald_hessian=hessian.tolist(),
        leading_exchange_finite_size=float(-.125*np.sum(hessian*spread)),
        reciprocal=rows, citations=['10.1103/PhysRevB.75.205126', '10.1063/1.4998644', '10.1103/PhysRevX.14.011059'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--cutoffs', type=float, nargs='+', default=[20., 40., 80.])
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.cutoffs):
        parser.error('cutoffs must be positive finite energies')
    from vibeqc.output import OutputChannel, write
    with OutputChannel.to_stdout():
        write(json.dumps(diagnose(args.archive, args.cutoffs), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
