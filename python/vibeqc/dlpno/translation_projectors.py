"""Compact occupied metric projectors for admitted real Bloch references.

Q(k) = C(k) [C(k).H S(k) C(k)]^-1 C(k).H S(k), separately for
frozen and active occupied bands. The inverse finite Fourier transform
stores Q[R,T] = Q[0,T-R]. This is the occupied projection in Riplinger
and Neese, doi:10.1063/1.4773581, Sec. II B1, with the unitary Bloch/BvK
convention of Nejad et al., doi:10.1063/5.0290816, Sec. II, Eqs. (3),(9).
It is not a Wannier optimization or a change of occupied partition.
"""

from dataclasses import dataclass
from math import prod
from typing import NamedTuple

import numpy as np

from .translation_operators import CyclicAOOperator, matrix_apply, matrix_record


class BlochOccupiedProjectors(NamedTuple):
    """Immutable home-row projectors, bound to their audited source operators.

    Construction replays the complete canonical admission by eigenspace.
    No canonical coefficient array is assembled or retained by this object.
    Recorded audit defects still apply at each consumer's tolerance.
    """

    active: CyclicAOOperator
    frozen: CyclicAOOperator
    n_occupied: int
    n_frozen: int
    overlap_identity: str
    fock_identity: str
    orthogonality_error: float
    projection_error: float
    imaginary_error: float

    def __array__(self, dtype=None, copy=None):
        raise TypeError('occupied Bloch projectors do not supply canonical coefficients')

    def record(self):
        return dict(kind='bloch-occupied-projectors-v1',
                    active=matrix_record(self.active), frozen=matrix_record(self.frozen),
                    n_occupied=self.n_occupied, n_frozen=self.n_frozen,
                    overlap_identity=self.overlap_identity, fock_identity=self.fock_identity,
                    orthogonality_error=self.orthogonality_error,
                    projection_error=self.projection_error, imaginary_error=self.imaginary_error)

    def validate(self, mesh, overlap, fock, n_occupied, n_frozen, tolerance):
        if (type(self.active) is not CyclicAOOperator or type(self.frozen) is not CyclicAOOperator
                or type(overlap) is not CyclicAOOperator or type(fock) is not CyclicAOOperator
                or self.active.mesh != tuple(mesh) or self.frozen.mesh != tuple(mesh)
                or overlap.mesh != tuple(mesh) or fock.mesh != tuple(mesh)
                or self.active.shape != overlap.shape or self.frozen.shape != overlap.shape
                or overlap.identity != self.overlap_identity or fock.identity != self.fock_identity):
            raise ValueError('occupied projectors do not match the source mesh/overlap/Fock')
        if n_occupied != self.n_occupied or n_frozen != self.n_frozen:
            raise ValueError('occupied projector electron count or frozen partition mismatch')
        errors = (self.orthogonality_error, self.projection_error, self.imaginary_error)
        if not np.isfinite(errors).all() or min(errors) < 0 or max(errors) > tolerance:
            raise ValueError('occupied projector admission defects exceed source tolerance')


@dataclass
class ProjectorReference:
    """Restricted translation reference with explicit projectors, no MO array."""

    occupied_projectors: BlochOccupiedProjectors
    fock: CyclicAOOperator
    overlap: CyclicAOOperator
    mo_energies: np.ndarray
    energy: float
    converged: bool = True
    ecp_total_ncore: int = 0
    reference_kind: str = 'rhf'
    localisation: str = 'none'


def reference_orbitals(reference):
    value = getattr(reference, 'occupied_projectors', None)
    if value is not None:
        if type(value) is not BlochOccupiedProjectors:
            raise TypeError('source occupied projectors require the admitted implementation')
        return value
    return reference.mo_coeffs


def projector_error(coefficients, right, projector):
    """Keep the existing maximum AO-entry comparison in streamed rows."""
    size, local = projector.shape[0], projector.blocks.shape[1]
    columns = np.arange(size)
    error = 0.0
    for start in range(0, size, local):
        rows = np.arange(start, start+local)
        error = max(error, float(np.max(abs(coefficients[rows] @ right-projector.block(rows, columns)))))
    return error


def projector_seed_metric(projector, overlap):
    """Exact Q.T S Q home blocks, retaining finite-precision non-idempotence."""
    mesh, shape = projector.mesh, projector.blocks.shape

    def symbols(operator):
        return (np.fft.ifftn(operator.blocks.reshape(*mesh, *shape[1:]), axes=(0, 1, 2))
                * prod(mesh)).reshape(shape)

    Q, S = symbols(projector), symbols(overlap)
    product = Q.conj().transpose(0, 2, 1) @ S @ Q
    blocks = np.fft.fftn(product.reshape(*mesh, *shape[1:]), axes=(0, 1, 2)).reshape(shape)/prod(mesh)
    # Real cyclic input has conjugate-paired symbols. The inverse is real
    # algebraically; discard only the FFT's imaginary arithmetic roundoff.
    if not np.isfinite(blocks).all():
        raise FloatingPointError('non-finite occupied seed metric')
    return CyclicAOOperator(blocks.real, mesh)


def _replayed_projector_error(blocks, projector, first, last):
    """Compare all AO entries of a partition, accumulating one row tile.

    Each complete real eigenspace contributes C_X[rows] C_X.T S. No full
    occupied frame, square projector, or all-origin covariance assumption
    substitutes for the original numerical partition comparison.
    """
    size, local = projector.shape[0], projector.blocks.shape[1]
    selected = [(i, j) for i, j in blocks.ranges if first <= i and j <= last]
    if sum(j-i for i, j in selected) != last-first:
        raise ValueError('canonical eigenspace crosses the frozen/occupied partition')
    columns = np.arange(size)
    error = 0.0
    for start in range(0, size, local):
        rows = np.arange(start, start+local)
        actual = np.zeros((local, size))
        for i, j in selected:
            C, _ = blocks.realify(i, j)
            right = matrix_apply(blocks.S, C, transpose=True).T
            actual += C[rows] @ right
            del C, right
        error = max(error, float(np.max(abs(actual-projector.block(rows, columns)))))
    return error


def compact_bloch_projectors(coefficients, overlaps, kfrac, weights, mesh,
                             canonical_blocks, overlap, fock, n_occupied, n_frozen, orthogonality):
    """Form compact projectors and compare both against the canonical audit.

    The per-partition Gram inverse removes input normalization roundoff,
    without mixing frozen and active bands. Agreement with the original
    realified source is measured, not assumed from orthonormal SCF inputs.
    """
    from ..periodic.chi.scf import inverse_bloch_transform

    cells = np.array(list(np.ndindex(mesh)))
    nk = len(cells)
    no, nf = n_occupied//nk, n_frozen//nk
    projectors, imaginary, differences = [], [], []
    for bands, first, last in ((slice(nf, no), n_frozen, n_occupied),
                               (slice(0, nf), 0, n_frozen)):
        C = coefficients[:, :, bands]
        right = C.conj().transpose(0, 2, 1) @ overlaps
        gram = right @ C
        Q = C @ np.linalg.solve(gram, right) if C.shape[2] else np.zeros_like(overlaps)
        blocks = inverse_bloch_transform(Q, kfrac, cells, weights, extension_matrix=np.diag(mesh))
        imaginary.append(float(np.max(abs(blocks.imag))))
        projector = CyclicAOOperator(blocks.real, mesh)
        # Q is a metric projector, generally nonsymmetric. Do not apply the
        # transpose averaging used for overlap and Fock operators.
        projectors.append(projector)
        differences.append(_replayed_projector_error(canonical_blocks, projector, first, last))
    result = BlochOccupiedProjectors(*projectors, int(n_occupied), int(n_frozen),
                                    overlap.identity, fock.identity, orthogonality,
                                    max(differences), max(imaginary))
    result.validate(mesh, overlap, fock, n_occupied, n_frozen, 1e-8)
    return result
