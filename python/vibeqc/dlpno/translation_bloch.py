"""Admit real finite-torus operators directly from complete Bloch data.

The positive-phase Bloch convention of Nejad et al., JCP 163, 214107
(2025), Sec. II, Eqs. (2)-(6), gives home-row blocks with a negative
inverse-transform phase. No source AO matrix is expanded. This is an
input representation change, not the paper's optimized Wannier gauge.
"""

from itertools import product
from math import prod
from numbers import Integral

import numpy as np

from .source import (
    CorrelationSourceError, GammaReference, _array_identity, _periodic_channels,
    _periodic_channel_data, _real_gamma_orbitals, refuse_unsupported_periodic_reference,
)
from .translation_operators import CyclicAOOperator, matrix_record
from ..periodic.chi._transform_inputs import complete_characters, real_array
from ..periodic.chi.pno import _validated_pair_mesh


def _complex_record(value):
    """Retain both components; the shared real-array hasher casts to float."""
    return {'real': _array_identity(value.real), 'imaginary': _array_identity(value.imag)}


def bloch_translation_reference(system, basis, result, mesh, *, n_frozen=0):
    """Return a real restricted reference and its consumed-input record.

    The mesh must be the full cyclic Gamma mesh in the producer's canonical
    order. ``n_frozen`` counts orbitals over the whole torus. All per-k AO
    spaces must be full rank, with integer occupations and global gaps at
    the occupied/virtual and frozen/active boundaries. Correlation-factor
    authentication remains the calling adapter's responsibility.
    """
    from ..periodic.chi.scf import cyclic_gamma_mesh, inverse_bloch_transform

    mesh = _validated_pair_mesh(mesh)
    n_cells = prod(mesh)
    if (isinstance(n_frozen, (bool, np.bool_)) or not isinstance(n_frozen, Integral)
            or n_frozen < 0 or n_frozen % n_cells):
        raise CorrelationSourceError("n_frozen must contain complete finite-torus bands")
    refuse_unsupported_periodic_reference(result)
    if not bool(getattr(result, 'converged', False)):
        raise CorrelationSourceError("Bloch translation reference must be converged")
    temperature = getattr(result, 'smearing_temperature', 0.0)
    if not np.isfinite(temperature) or float(temperature) != 0:
        raise CorrelationSourceError("Bloch translation reference requires unsmeared integer occupations")
    kind, channels = _periodic_channels(system, result, n_frozen // n_cells)
    if kind != 'rhf':
        raise CorrelationSourceError("block translation reference currently requires restricted RHF")
    n_occ = channels[0][1]
    nbf = int(basis.nbasis)
    kpoints = cyclic_gamma_mesh(system, mesh)
    kfrac = real_array(getattr(result, 'kpoints_frac', []), name='reference kpoints_frac')
    kcart = real_array(getattr(result, 'kpoints_cart', []), name='reference kpoints_cart')
    weights = real_array(getattr(result, 'kpoint_weights', []), name='reference kpoint_weights')
    complete_characters(kfrac, weights, np.diag(mesh))
    expected_frac = np.asarray(kpoints.kpoints_frac)
    expected_cart = np.asarray(kpoints.kpoints_cart)
    if (kcart.shape != expected_cart.shape or kfrac.shape != expected_frac.shape
            or not np.allclose(kfrac, expected_frac, atol=1e-10, rtol=0)
            or not np.allclose(kcart, expected_cart, atol=1e-10, rtol=0)):
        raise CorrelationSourceError("Bloch reference must use the full Gamma-centred mesh in cyclic order")
    coeffs, focks, energies = _periodic_channel_data(
        result, '', n_cells, n_occ, 2., nbf, n_frozen=n_frozen // n_cells,
    )
    overlaps = np.asarray(getattr(result, 'overlap', []), dtype=complex)
    focks = np.asarray(focks, dtype=complex)
    coeffs = np.asarray(coeffs, dtype=complex)
    # _periodic_channel_data has already bounded their imaginary residual.
    energies = np.asarray(energies).real.astype(float)
    if (overlaps.shape != (n_cells, nbf, nbf) or coeffs.shape != overlaps.shape
            or not np.isfinite(overlaps).all()):
        raise CorrelationSourceError("Bloch reference requires complete full-rank per-k AO spaces")
    hermitian = max(float(np.max(abs(a-a.conj().transpose(0, 2, 1))))
                    for a in (overlaps, focks))
    if hermitian > 1e-8:
        raise CorrelationSourceError("Bloch overlap and Fock must be Hermitian")
    # This is the same global relative overlap-rank cutoff as the dense
    # chi reference, evaluated on its unitarily equivalent Bloch spectra.
    spectrum = np.linalg.eigvalsh(.5*(overlaps+overlaps.conj().transpose(0, 2, 1)))
    if not np.all(spectrum > 1e-10 * float(spectrum.max())):
        raise CorrelationSourceError("real-torus post-HF reference is linearly dependent")
    orthogonality = float(np.max(abs(coeffs.conj().transpose(0, 2, 1) @ overlaps @ coeffs - np.eye(nbf))))
    stationarity = float(np.max(abs(focks @ coeffs - (overlaps @ coeffs)*energies[:, None, :])))
    if max(orthogonality, stationarity) > 1e-8:
        raise CorrelationSourceError("Bloch orbitals do not reproduce the supplied metric/Fock")
    translations = np.asarray(list(product(*(range(n) for n in mesh))), dtype=int)
    lookup = {tuple(t): i for i, t in enumerate(translations)}
    inverse = [lookup[tuple((-t) % np.asarray(mesh))] for t in translations]
    operators, imaginary = [], []
    for matrices in (overlaps, focks):
        blocks = inverse_bloch_transform(matrices, kfrac, translations, weights,
                                         extension_matrix=np.diag(mesh))
        residual = float(np.max(abs(blocks.imag)))
        if not np.isfinite(residual) or residual > 1e-8:
            raise CorrelationSourceError("Bloch operator is not a real time-reversal-compatible finite torus")
        # Match the existing dense native transform's transpose average,
        # after auditing the complex data; do not silently repair defects.
        real = .5*(blocks.real+blocks.real[inverse].transpose(0, 2, 1))
        operators.append(CyclicAOOperator(real, mesh))
        imaginary.append(residual)
    S, F = operators
    phases = np.exp(2j*np.pi*(translations @ kfrac.T))/np.sqrt(n_cells)
    folded = np.concatenate([
        (phases[:, k, None, None]*coeffs[k][None, :, :]).reshape(n_cells*nbf, nbf)
        for k in range(n_cells)
    ], axis=1)
    # Realify whole degenerate eigenspaces, never individual complex vectors.
    # Global gap admission above prevents either occupied partition from
    # bisecting one of these spaces. Metric/Fock actions remain blockwise.
    C, eps = _real_gamma_orbitals(folded, energies.ravel(), S, F)
    SC = S.apply(C)
    real_orthogonality = float(np.max(abs(C.T @ SC-np.eye(C.shape[1]))))
    real_stationarity = float(np.max(abs(F.apply(C)-SC*eps[None, :])))
    if max(real_orthogonality, real_stationarity) > 1e-8:
        raise CorrelationSourceError("realified orbitals do not preserve the finite Bloch Hamiltonian")
    energy = float(result.energy)
    if not np.isfinite(energy):
        raise CorrelationSourceError("Bloch reference energy must be finite")
    reference = GammaReference(C, F, S, eps, energy*n_cells,
                               ecp_total_ncore=int(getattr(result, 'ecp_total_ncore', 0) or 0)*n_cells)
    record = {
        'kind': 'complete-bloch-to-cyclic-operators-v1', 'mesh': list(mesh),
        'kpoints_frac': _array_identity(kfrac), 'kpoints_cart': _array_identity(kcart),
        'weights': _array_identity(weights), 'overlap_bloch': _complex_record(overlaps),
        'fock_bloch': _complex_record(focks), 'orbitals_bloch': _complex_record(coeffs),
        'energies_bloch': _array_identity(energies), 'overlap': matrix_record(S), 'fock': matrix_record(F),
        'n_frozen': int(n_frozen), 'energy_per_cell': energy,
        'normalization': 'inverse operators 1/N; orbitals 1/sqrt(N); energy N',
        'hermitian_error': hermitian, 'matrix_imaginary_residual': max(imaginary),
        'bloch_orthogonality_error': orthogonality, 'bloch_stationarity_error': stationarity,
        'real_orthogonality_error': real_orthogonality, 'real_stationarity_error': real_stationarity,
    }
    return reference, record
