"""The Layer-A embedding potential and its energy linearization.

The embedding potential ``Sigma_emb(r_S, r'_S; E)`` on the dividing plane
*S* is energy dependent and non-local. Re-evaluating it (a full
Sancho-Rubio decimation of the bulk Green function) at every complex
contour node would dominate the cost, so -- following Ishida -- it is
linearized around a reference energy ``E0``:

    Sigma_emb(E) ~= Sigma_emb(E0) + (E - E0) * dSigma_emb/dE|_{E0}.

Carrying only ``Sigma0`` and ``dSigma/dE`` reproduces the embedding
potential across the contour to second order in ``(E - E0)``, which is
ample given the contour hugs a reference energy and the substrate
Green function is smooth off the real axis.

Reference: H. Ishida, Phys. Rev. B 63, 165409 (2001),
doi:10.1103/PhysRevB.63.165409 (energy linearization of the embedding
potential; two-step substrate-then-surface structure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class EmbeddingPotential:
    """An energy-dependent embedding potential ``z -> Sigma_emb(z)``.

    Wraps the callable that returns the surface-projected embedding
    potential matrix (or scalar, for the 1D model) at a complex energy
    ``z`` -- e.g. ``Sigma = z*I - H00 - inv(g_surface)`` from
    :func:`~vibeqc.periodic_embedding.decimation.sancho_rubio_surface_gf`.
    """

    def __init__(self, sigma_fn: Callable[[complex], object], *, label: str = ""):
        self._sigma_fn = sigma_fn
        self.label = label

    def at(self, z: complex) -> NDArray[np.complex128]:
        """Full (non-linearized) embedding potential at ``z``."""
        return np.atleast_2d(np.asarray(self._sigma_fn(z), dtype=np.complex128))

    def linearize(
        self, z0: complex, *, step: float = 1e-5
    ) -> "LinearizedEmbeddingPotential":
        """Linearize around reference energy ``z0`` (Ishida).

        ``dSigma/dz`` is taken by a central finite difference along the
        real axis; since ``Sigma_emb(z)`` is analytic off the real axis
        this equals the complex derivative.
        """
        s0 = self.at(z0)
        sp = self.at(z0 + step)
        sm = self.at(z0 - step)
        dsig = (sp - sm) / (2.0 * step)
        return LinearizedEmbeddingPotential(
            z0=complex(z0), sigma0=s0, dsigma_dz=dsig, label=self.label
        )


@dataclass(frozen=True)
class LinearizedEmbeddingPotential:
    """``Sigma_emb(z) ~= sigma0 + (z - z0) * dsigma_dz`` (Ishida)."""

    z0: complex
    sigma0: NDArray[np.complex128]
    dsigma_dz: NDArray[np.complex128]
    label: str = ""

    def at(self, z: complex) -> NDArray[np.complex128]:
        return self.sigma0 + self.dsigma_dz * (z - self.z0)
