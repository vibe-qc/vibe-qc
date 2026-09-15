from __future__ import annotations

"""Spectral properties of the accepted periodic GDF Hamiltonian and density."""

import numpy as np

from .bands import _HARTREE_TO_EV, _shell_to_atom, ao_groups_per_atom_l
from .coop_cohp import (
    _pair_metadata, _periodic_mayer_from_density, ao_pairs_per_atom_pair,
)
from .spin_channels import spin_densities


def _shell_type(result):
    """``(open_shell, rohf)`` for *result*, decided by values, not by names.

    Result adapters (``CCMRealGammaResult``, ``CCMFourCentreResult``,
    ``CCMKSResult``) declare ``density_alpha`` / ``density_beta`` as
    dataclass fields that are ``None`` on a closed-shell run, so attribute
    presence says nothing about the shell type (#198). A result is open
    shell only when both spin densities are populated, and ROHF when it is
    open shell but reports no per-spin orbital energies -- absent *or*
    ``None``, the same test either way.
    """
    alpha, _beta = spin_densities(result)
    open_shell = alpha is not None
    rohf = open_shell and getattr(result, 'mo_energies_alpha', None) is None
    return open_shell, rohf


def _accepted_channels(result, *, rohf_effective_spectrum=False):
    open_shell, rohf = _shell_type(result)
    if rohf and not rohf_effective_spectrum:
        raise NotImplementedError(
            'ROHF GDF spectral properties require an explicit distinction between '
            'the effective orbital operator and the physical spin Hamiltonians'
        )
    suffixes = ('_alpha', '_beta') if open_shell else ('',)
    channels = []
    for suffix in suffixes:
        orbital_suffix = '' if rohf else suffix
        # The orbital operator certifies the reported eigenpairs; the
        # physical spin Fock supplies Hamiltonian-weighted populations.
        values = (
            getattr(result, 'mo_energies' + orbital_suffix),
            getattr(result, 'mo_coeffs' + orbital_suffix),
            getattr(result, 'fock' + suffix),
            getattr(result, 'density' + suffix),
            getattr(result, 'fock' + orbital_suffix),
        )
        if isinstance(result.overlap, np.ndarray) and result.overlap.ndim == 2:
            values = tuple([value] for value in values)
        channels.append(values)
    return channels


def gdf_properties_from_result(
    result, system, basis, *, coop_cohp=False, sigma_ev=0.05,
    memory_byte_cap=128 * 1024**2, _rohf_effective_spectrum=False,
):
    """Project the returned SCF eigenstates; never rebuild or interpolate F.

    DOS/PDOS count spatial orbitals in a restricted calculation, or orbitals
    per spin in an unrestricted calculation. COOP/COHP include the restricted
    spin degeneracy. Their occupied integrals contract the accepted density,
    so fractional occupations are preserved. -ICOHP is returned in eV.
    See Dronskowski and Bloechl (1993), Eqs. (4)-(13) and (19).

    The private ROHF diagnostic projects each physical spin Fock on the
    shared effective-orbital energy axis. It is not a physical spin-Fock
    eigenspectrum. Provenance distinguishes both operators, and integrated
    populations use accepted densities rather than integer MO refills.
    Public ROHF artifacts remain disabled pending validation and export support.

    The energy spacing resolves the requested Gaussian width. Admission
    precedes spectrum allocation, and broadening visits only eight-sigma
    neighborhoods instead of materializing a bands-by-energy-grid tensor.
    """
    if not result.converged:
        raise ValueError('GDF properties require a converged accepted state')
    sigma = float(sigma_ev) / _HARTREE_TO_EV
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('GDF spectral broadening must be positive and finite')
    channels = _accepted_channels(result, rohf_effective_spectrum=_rohf_effective_spectrum)
    _open_shell, rohf = _shell_type(result)
    n_spin = len(channels)
    weights = np.asarray(getattr(result, 'kpoint_weights', [1.]), dtype=float)
    kpoints = np.asarray(getattr(result, 'kpoints_cart', np.zeros((1, 3))), dtype=float)
    overlaps = ([result.overlap] if isinstance(result.overlap, np.ndarray) and result.overlap.ndim == 2
                else result.overlap)
    nk = len(overlaps)
    if (weights.shape != (nk,) or kpoints.shape != (nk, 3)
            or not np.isfinite(weights).all() or not np.isfinite(kpoints).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError('GDF properties require a normalized accepted k mesh')
    nbf = int(basis.nbasis)
    scratch_bytes = 16 * 12 * nbf**2 + 8 * len(system.unit_cell)**2
    if scratch_bytes > int(memory_byte_cap):
        raise MemoryError(
            f'GDF spectral properties require at least {scratch_bytes} bytes; '
            f'cap is {memory_byte_cap}'
        )
    electron_count = 0.
    emin, emax = np.inf, -np.inf
    for energies, coefficients, focks, densities, orbital_focks in channels:
        if any(len(values) != nk for values in (energies, coefficients, focks, densities, orbital_focks)):
            raise ValueError('GDF property channels have inconsistent meshes')
        for e, c, f, d, orbital_f, s, wk in zip(
            energies, coefficients, focks, densities, orbital_focks, overlaps, weights,
        ):
            e, c, f, d, orbital_f, s = map(np.asarray, (e, c, f, d, orbital_f, s))
            if (e.ndim != 1 or not len(e) or c.shape != (nbf, len(e))
                    or any(a.shape != (nbf, nbf) for a in (f, d, orbital_f, s))
                    or any(not np.isfinite(a).all() for a in (e, c, f, d, orbital_f, s))):
                raise ValueError('GDF properties require finite aligned SCF matrices')
            if any(not np.allclose(a, a.conj().T, atol=1e-9, rtol=0) for a in (f, d, orbital_f, s)):
                raise ValueError('GDF properties require Hermitian SCF matrices')
            if (not np.allclose(c.conj().T @ s @ c, np.eye(len(e)), atol=1e-7, rtol=0)
                    or not np.allclose(c.conj().T @ orbital_f @ c, np.diag(e), atol=1e-7, rtol=1e-8)):
                raise ValueError('GDF orbitals do not diagonalize the returned Hamiltonian')
            electron_count += float(wk) * float(np.trace(d @ s).real)
            emin, emax = min(emin, float(e.min())), max(emax, float(e.max()))
    groups = ao_groups_per_atom_l(system, basis)
    pairs = ao_pairs_per_atom_pair(system, basis) if coop_cohp else {}
    pair_keys = sorted(pairs)
    pair_meta = _pair_metadata(system, pair_keys)
    ngroups, npairs = len(groups), len(pairs)
    first, last = emin - 8 * sigma, emax + 8 * sigma
    n_grid = max(500, int(np.ceil((last - first) / (sigma / 4))) + 1)
    # Include all persistent spectra and the returned energy coordinate,
    # plus contraction scratch. Returned arrays share their storage.
    reserved = scratch_bytes + 8 * (
        n_grid * (2 + n_spin * (1 + ngroups + 2*npairs))
        + (ngroups + 2*npairs)*nbf + 2*n_spin*npairs
    )
    if reserved > int(memory_byte_cap):
        raise MemoryError(
            f'GDF spectral properties require {reserved} bytes; cap is {memory_byte_cap}'
        )
    energy_grid = np.linspace(first, last, n_grid)
    dos = np.zeros((n_spin, n_grid))
    pdos = np.zeros((n_spin, ngroups, n_grid))
    coop = np.zeros((n_spin, npairs, n_grid))
    cohp = np.zeros_like(coop)
    icoop = np.zeros((n_spin, npairs))
    icohp = np.zeros_like(icoop)
    capacity = 2. if n_spin == 1 else 1.
    norm = 1. / (sigma * np.sqrt(2*np.pi))
    for spin, (energies, coefficients, focks, densities, _) in enumerate(channels):
        for e, c, f, d, s, wk in zip(energies, coefficients, focks, densities, overlaps, weights):
            e, c, f, d, s = map(np.asarray, (e, c, f, d, s))
            mulliken = (c.conj() * (s @ c)).real
            group_weights = np.array([mulliken[indices].sum(axis=0) for indices in groups.values()])
            pair_s = np.zeros((npairs, len(e)))
            pair_f = np.zeros_like(pair_s)
            for pair, key in enumerate(pair_keys):
                a, b = pairs[key]
                ab = np.ix_(a, b)
                ca, cb = c[a], c[b]
                pair_s[pair] = np.einsum('in,ij,jn->n', ca.conj(), s[ab], cb).real
                pair_f[pair] = np.einsum('in,ij,jn->n', ca.conj(), f[ab], cb).real
                icoop[spin, pair] += wk * np.sum(d.T[ab] * s[ab]).real
                icohp[spin, pair] -= wk * np.sum(d.T[ab] * f[ab]).real * _HARTREE_TO_EV
            for band, energy in enumerate(e):
                left, right = np.searchsorted(energy_grid, [energy - 8*sigma, energy + 8*sigma])
                gaussian = wk * norm * np.exp(-.5*((energy_grid[left:right] - energy)/sigma)**2)
                dos[spin, left:right] += gaussian / _HARTREE_TO_EV
                pdos[spin, :, left:right] += group_weights[:, band, None] * gaussian / _HARTREE_TO_EV
                coop[spin, :, left:right] += capacity * pair_s[:, band, None] * gaussian / _HARTREE_TO_EV
                cohp[spin, :, left:right] -= capacity * pair_f[:, band, None] * gaussian
    fermi_values = []
    for suffix, (energies, coefficients, _, densities, _) in zip(
        ('',) if n_spin == 1 else ('_alpha', '_beta'), channels,
    ):
        value = getattr(result, 'fermi_level' + suffix, None)
        if value is None:
            occupied = []
            for e, c, d, s in zip(energies, coefficients, densities, overlaps):
                sc = np.asarray(s) @ np.asarray(c)
                occupation = np.diag(sc.conj().T @ np.asarray(d) @ sc).real
                occupied.extend(np.asarray(e)[occupation > 1e-8])
            value = max(occupied) if occupied else emin
        fermi_values.append(float(value))
    fermi = max(fermi_values)
    if not np.isfinite(fermi):
        raise ValueError('GDF property chemical potential is not finite')
    energies_ev = (energy_grid - fermi) * _HARTREE_TO_EV
    unwrap = lambda values: values[0] if n_spin == 1 else values
    common = dict(energies=energies_ev, energies_units='eV', n_spin=n_spin,
                  fermi_energy_ev=fermi * _HARTREE_TO_EV)
    if rohf:
        common.update(
            energy_operator='roothaan-effective-fock',
            orbital_basis='shared-rohf',
            population_density='accepted-spin-density',
            validation_status='private-unvalidated',
        )
    dos_data = dict(common, dos=unwrap(dos), smearing=float(sigma_ev),
                    smearing_type='gaussian', n_electrons=electron_count)
    from .bands import _shell_to_l, _atom_label
    ao_atoms, ao_l = _shell_to_atom(basis), _shell_to_l(basis)
    channel_meta = []
    for label, indices in groups.items():
        atom = int(ao_atoms[indices[0]])
        symbol = _atom_label(int(system.unit_cell[atom].Z), atom + 1).rstrip('0123456789')
        channel_meta.append(dict(atom_index=atom, symbol=symbol,
                                 l=int(ao_l[indices[0]]), label=label))
    pdos_data = dict(common, projections=unwrap(pdos), channels=channel_meta)
    coop_data = (dict(common, projections=unwrap(coop), integrated=unwrap(icoop),
                      sigma_ev=float(sigma_ev), pairs=pair_meta) if npairs else None)
    cohp_data = (dict(common, projections=unwrap(cohp), integrated=unwrap(icohp),
                      sigma_ev=float(sigma_ev), pairs=pair_meta) if npairs else None)
    if rohf:
        dos_data['projection_operator'] = pdos_data['projection_operator'] = 'overlap'
        if coop_data is not None:
            coop_data['projection_operator'] = 'overlap'
            cohp_data['projection_operator'] = 'physical-spin-fock'
    mayer = _periodic_mayer_from_density(
        [ch[3] for ch in channels], overlaps, weights, ao_atoms, len(system.unit_cell),
    )
    return dos_data, pdos_data, coop_data, cohp_data, mayer, reserved
