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
    _periodic_channel_data, _real_gamma_orbitals_from_columns, _RealGammaOrbitalBlocks,
    refuse_unsupported_periodic_reference,
)
from .translation_operators import CyclicAOOperator, matrix_record
from ..periodic.chi._transform_inputs import complete_characters, real_array
from ..periodic.chi.pno import _validated_pair_mesh


def _complex_record(value):
    """Retain both components; the shared real-array hasher casts to float."""
    return {'real': _array_identity(value.real), 'imaginary': _array_identity(value.imag)}


def _audit_real_orbital_blocks(blocks, n_occupied, panel_width):
    """Audit every entry of C.T S C and F C - S C eps by replaying blocks.

    Per-k residuals or unitary-invariant norms cannot replace the existing
    maximum AO/orbital-entry checks. Replay holds complete eigenspaces but
    never assembles the full real canonical coefficient array.
    """
    eps = np.empty(blocks.n_orb)
    orthogonality = occupied_orthogonality = stationarity = 0.0
    for i, j in blocks.ranges:
        C, energy = blocks.realify(i, j)
        eps[i:j] = energy
        metric = blocks.S.apply(C)
        for start in range(0, j-i, panel_width):
            stop = min(start+panel_width, j-i)
            residual = blocks.F.apply(C[:, start:stop])-metric[:, start:stop]*energy[None, start:stop]
            stationarity = max(stationarity, float(np.max(abs(residual))))
        for p, q in blocks.ranges:
            other = C if p == i else blocks.realify(p, q)[0]
            for start in range(0, j-i, panel_width):
                stop = min(start+panel_width, j-i)
                gram = other.T @ metric[:, start:stop]
                if p == i:
                    gram[np.arange(start, stop), np.arange(stop-start)] -= 1
                orthogonality = max(orthogonality, float(np.max(abs(gram))))
                rows, columns = min(q, n_occupied)-p, min(i+stop, n_occupied)-(i+start)
                if rows > 0 and columns > 0:
                    occupied_orthogonality = max(occupied_orthogonality,
                        float(np.max(abs(gram[:rows, :columns]))))
            del other
        del C, metric
    if max(orthogonality, stationarity) > 1e-8:
        raise CorrelationSourceError("realified orbitals do not preserve the finite Bloch Hamiltonian")
    return eps, orthogonality, stationarity, occupied_orthogonality


def bloch_translation_reference(system, basis, result, mesh, *, n_frozen=0, compact_occupied=False):
    """Return a real restricted reference and its consumed-input record.

    The mesh must be the full cyclic Gamma mesh in the producer's canonical
    order. ``n_frozen`` counts orbitals over the whole torus. All per-k AO
    spaces must be full rank, with integer occupations and global gaps at
    the occupied/virtual and frozen/active boundaries. Correlation-factor
    authentication remains the calling adapter's responsibility.
    ``compact_occupied=True`` retains separate home-row occupied projectors
    instead of canonical coefficients. Its complete orbital audit replays
    eigenspaces without assembling the full real canonical array. A bounded
    cache of small transforms avoids repeated eigendecompositions.
    """
    from ..periodic.chi.scf import cyclic_gamma_mesh, inverse_bloch_transform

    if type(compact_occupied) is not bool:
        raise ValueError('compact_occupied must be a boolean')
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
    # Realify whole degenerate eigenspaces, never individual complex vectors.
    # Global gap admission above prevents either occupied partition from
    # bisecting one of these spaces. Nejad Sec. II, Eq. (9), supplies each
    # requested column directly from compact Bloch data. No full complex
    # finite-torus coefficients are prepared; phase panels belong only to
    # the current eigenspace, not a precomputed cell-by-k table.
    max_block_columns = 0

    def columns(indices):
        nonlocal max_block_columns
        max_block_columns = max(max_block_columns, len(indices))
        k, band = np.divmod(indices, nbf)
        phases = np.exp(2j*np.pi*(translations @ kfrac[k].T))/np.sqrt(n_cells)
        return (phases[:, None, :]*coeffs[k, :, band].T[None, :, :]).reshape(n_cells*nbf, -1)

    if compact_occupied:
        # Bound retained reconstruction recipes by one primitive Bloch
        # coefficient payload (16*N*b*b bytes), never by the folded M*M
        # array. Oversized eigenspaces retain the uncached replay path.
        blocks = _RealGammaOrbitalBlocks(n_cells*nbf, energies.ravel(), columns, S, F,
                                        recipe_cache_bytes=coeffs.nbytes)
        eps, real_orthogonality, real_stationarity, occupied_orthogonality = _audit_real_orbital_blocks(
            blocks, n_occ*n_cells, nbf)
    else:
        C, eps = _real_gamma_orbitals_from_columns(n_cells*nbf, energies.ravel(), columns, S, F)
        # Preserve the maximum-entry checks over every column, including cross
        # eigenspace overlaps, without retaining S*C or a square Gram matrix.
        real_orthogonality = real_stationarity = 0.0
        for start in range(0, C.shape[1], nbf):
            stop = min(start+nbf, C.shape[1])
            panel = C[:, start:stop]
            metric_panel = S.apply(panel)
            gram = C.T @ metric_panel
            gram[np.arange(start, stop), np.arange(stop-start)] -= 1
            real_orthogonality = max(real_orthogonality, float(np.max(abs(gram))))
            residual = F.apply(panel)-metric_panel*eps[None, start:stop]
            real_stationarity = max(real_stationarity, float(np.max(abs(residual))))
        if max(real_orthogonality, real_stationarity) > 1e-8:
            raise CorrelationSourceError("realified orbitals do not preserve the finite Bloch Hamiltonian")
    energy = float(result.energy)
    if not np.isfinite(energy):
        raise CorrelationSourceError("Bloch reference energy must be finite")
    ecp_total_ncore = int(getattr(result, 'ecp_total_ncore', 0) or 0)*n_cells
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
        'orbital_preparation': 'complete-degenerate-bloch-column-blocks',
        'max_folded_orbital_columns': max_block_columns,
        'orbital_validation_columns': nbf,
    }
    if compact_occupied:
        from .translation_projectors import compact_bloch_projectors, ProjectorReference
        projectors = compact_bloch_projectors(coeffs, overlaps, kfrac, weights, mesh,
            blocks, S, F, n_occ*n_cells, int(n_frozen), occupied_orthogonality)
        reference = ProjectorReference(projectors, F, S, eps, energy*n_cells,
                                        ecp_total_ncore=ecp_total_ncore)
        record['occupied_projectors'] = projectors.record()
        record['orbital_audit'] = 'replayed-complete-eigenspaces'
        record['orbital_audit_block_evaluations'] = blocks.evaluations
        record['orbital_audit_recipe_cache'] = blocks.recipe_cache_record()
    else:
        reference = GammaReference(C, F, S, eps, energy*n_cells, ecp_total_ncore=ecp_total_ncore)
    return reference, record
