"""Compact projected-domain algebra for finite-torus pair preparation.

For an S-orthonormal occupied union C (including frozen orbitals), a domain
selector E_D gives Q E_D = E_D - C H, H = C.T S E_D. This is the projected
PAO definition of Riplinger and Neese, doi:10.1063/1.4773581, Sec. II B1,
and Nejad et al., doi:10.1063/5.0290816, Eqs. (26)-(28). Evaluating its
Gram/Fock blocks by low-rank subtraction avoids constructing the full Q or
performing a dense AO metric/Fock contraction separately for every pair.
All occupied projection tails remain present; no sparsity cutoff is used.
"""

from dataclasses import dataclass

import numpy as np

from .translation import _readonly
from .translation_operators import CyclicAOOperator, matrix_apply, matrix_block, matrix_columns_apply


@dataclass(frozen=True)
class ProjectedPairBasis:
    """V = E_D A - C B, with C ordered as active then frozen occupieds.

    The domain coefficients A and occupied-tail coefficients B suffice for
    exact cross-pair overlaps. Compact pair storage uses these same recipes
    as its retained representation; the dense oracle also keeps AO arrays.
    """

    ao_indices: np.ndarray
    local_coefficients: np.ndarray
    occupied_coefficients: np.ndarray

    def __post_init__(self):
        indices = np.ascontiguousarray(self.ao_indices, dtype=np.int64)
        object.__setattr__(self, "ao_indices", np.frombuffer(indices.tobytes(), dtype=np.int64))
        object.__setattr__(self, "local_coefficients", _readonly(self.local_coefficients))
        object.__setattr__(self, "occupied_coefficients", _readonly(self.occupied_coefficients))

    def rotate(self, rotation):
        return ProjectedPairBasis(self.ao_indices, self.local_coefficients @ rotation,
                                  self.occupied_coefficients @ rotation)


class ProjectedPAOPreparation:
    """Shared occupied contractions plus selected-domain matrix blocks."""

    def __init__(self, overlap, fock, occupied):
        self.overlap, self.fock, self.occupied = overlap, fock, occupied
        self.metric_occupied = matrix_apply(overlap, occupied)
        self.fock_occupied = matrix_apply(fock, occupied)
        self.occupied_fock = occupied.T @ self.fock_occupied

    def basis(self, domain, lindep):
        domain = np.asarray(domain, dtype=np.int64)
        H = self.metric_occupied[domain].T
        # A full domain may require a square projected Gram matrix. Fill its
        # rows directly, without also expanding the square AO source matrix.
        chunk = self.overlap.blocks.shape[1] if type(self.overlap) is CyclicAOOperator else len(domain)
        metric = np.empty((len(domain), len(domain)))
        for start in range(0, len(domain), chunk):
            stop = start+chunk
            metric[start:stop] = matrix_block(self.overlap, domain[start:stop], domain) - H[:, start:stop].T @ H
        eigenvalues, eigenvectors = np.linalg.eigh(.5 * (metric + metric.T))
        keep = eigenvalues > lindep
        if not np.any(keep):
            raise ValueError("representative pair AO domain has an empty PAO space")
        X = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])
        # Subtracting occupied contributions in the small Gram matrix loses
        # relative precision for nearly dependent PAOs. Refine the retained
        # basis in its own metric, without a second rank cutoff or a dense S*V
        # contraction. The selected AO columns and shared S*C suffice.
        tails = H @ X
        trial = -self.occupied @ tails
        trial[domain] += X
        metric_trial = matrix_columns_apply(self.overlap, domain, X) - self.metric_occupied @ tails
        retained_metric = trial.T @ metric_trial
        cholesky = np.linalg.cholesky(.5 * (retained_metric + retained_metric.T))
        X = X @ np.linalg.solve(cholesky.T, np.eye(X.shape[1]))
        # Do not assume an invariant occupied subspace or diagonal F_oo:
        # retain both cross terms in the exact projected Fock expression.
        FC = self.fock_occupied[domain]
        projected_fock = np.empty((len(domain), len(domain)))
        for start in range(0, len(domain), chunk):
            stop = start+chunk
            projected_fock[start:stop] = (
                matrix_block(self.fock, domain[start:stop], domain) - FC[start:stop] @ H
                - H[:, start:stop].T @ FC.T + H[:, start:stop].T @ self.occupied_fock @ H
            )
        small_fock = X.T @ projected_fock @ X
        energies, rotation = np.linalg.eigh(.5 * (small_fock + small_fock.T))
        A = X @ rotation
        B = H @ A
        coefficients = -self.occupied @ B
        coefficients[domain] += A
        return coefficients, energies, ProjectedPairBasis(domain, A, B)
