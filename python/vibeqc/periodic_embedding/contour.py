"""Complex-energy contour integration for Green's-function densities.

The embedding potential broadens the discrete slab spectrum into a
*continuous* local density of states in region I, so the occupied charge
cannot be obtained by summing discrete eigenvalues. Instead it is the
contour integral of the (retarded) Green function,

    n(r) = -(1/pi) Im \\int_C f(E) G(r, r; E) dE,

over a contour C enclosing the occupied spectrum. The retarded Green
function G(z) is analytic everywhere off the real axis (its only
non-analyticities -- poles and the branch cut that the embedding
potential turns into -- sit on or below the real axis), so the real-axis
integral of the spectral function can be deformed onto a smooth contour
in the upper half-plane where G is slowly varying and a handful of
Gauss-Legendre nodes integrate it to machine precision.

This module implements the *T = 0* semicircular contour. The occupied
charge is

    N_occ = -(1/pi) Im \\int_{E_bottom}^{E_F} G(E + i0+) dE
          = -(1/pi) Im \\int_{arc: E_bottom -> E_F} G(z) dz

where the arc is the upper half of the circle whose diameter is the real
segment [E_bottom, E_F] (E_bottom strictly below the band bottom). The
closed half-disk integral of the analytic G vanishes, so the real-axis
segment equals the arc; the arc never touches the branch cut except at
its endpoints, which the interior Gauss-Legendre nodes avoid.

Finite temperature (Matsubara poles of the Fermi function handled
explicitly, or a Gauss-Legendre arc plus a few Matsubara residues) is a
documented extension -- see ``handovers/HANDOVER_GF_EMBEDDING.md`` open design
question 2.

The node count needed for sub-meV convergence of the band energy is the
practical knob (open design question 2); :class:`EnergyContour` exposes
``n_nodes`` so a convergence sweep is a one-liner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EnergyContour:
    """A set of complex-energy quadrature nodes + weights.

    Construct with :meth:`semicircle`. The weights already fold in the
    ``-(1/pi) Im`` prefactor and the contour orientation, so the occupied
    charge from a diagonal Green function ``g(z_k)`` (or a region-I trace
    ``Tr G_II(z_k)``) is simply

        N_occ = contour.occupation([g(z) for z in contour.nodes])

    Attributes
    ----------
    nodes
        Complex energies ``z_k`` at which the Green function is sampled.
    weights
        Complex weights ``w_k`` such that ``N_occ = Im sum_k w_k g(z_k)``.
    e_bottom, e_fermi
        The real-axis endpoints of the diameter (bookkeeping).
    """

    nodes: NDArray[np.complex128]
    weights: NDArray[np.complex128]
    e_bottom: float
    e_fermi: float

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @classmethod
    def semicircle(
        cls,
        e_bottom: float,
        e_fermi: float,
        n_nodes: int = 24,
    ) -> "EnergyContour":
        """Upper-half-plane semicircular contour with diameter
        ``[e_bottom, e_fermi]`` on the real axis.

        ``e_bottom`` must lie strictly below the bottom of the occupied
        spectrum so that the spectral weight on the diameter to the left
        of the band is zero. ``e_fermi`` is the Fermi level (the contour
        endpoint inside, or at the top edge of, the occupied band).

        Parameters
        ----------
        n_nodes
            Number of Gauss-Legendre nodes on the arc. The arc integrand
            is analytic and smooth, so convergence is geometric until the
            ``sqrt`` band-edge / Fermi-level behaviour at the endpoints
            sets a polynomial floor; 20-40 nodes reach sub-meV.
        """
        if not e_fermi > e_bottom:
            raise ValueError(
                f"e_fermi ({e_fermi}) must exceed e_bottom ({e_bottom})"
            )
        if n_nodes < 1:
            raise ValueError("n_nodes must be >= 1")

        centre = 0.5 * (e_bottom + e_fermi)
        radius = 0.5 * (e_fermi - e_bottom)

        # Gauss-Legendre on theta in (0, pi). leggauss returns nodes on
        # (-1, 1); map x -> theta = (pi/2)(x + 1) with Jacobian pi/2.
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        theta = 0.5 * np.pi * (x + 1.0)
        jac = 0.5 * np.pi

        z = centre + radius * np.exp(1j * theta)
        # Derivation (see module docstring):
        #   N_occ = (1/pi) Im sum_j (w_j * jac) * i R e^{i theta_j} * g(z_j)
        # => weight_j = (i / pi) * R * e^{i theta_j} * (w_j * jac).
        weights = (1j / np.pi) * radius * np.exp(1j * theta) * (w * jac)
        return cls(
            nodes=z.astype(np.complex128),
            weights=weights.astype(np.complex128),
            e_bottom=float(e_bottom),
            e_fermi=float(e_fermi),
        )

    def occupation(self, g_values: NDArray[np.complex128] | list) -> float:
        """Occupied charge ``N_occ = Im sum_k w_k g(z_k)``.

        ``g_values`` are the (scalar) diagonal Green function or region-I
        trace ``Tr G_II(z_k)`` evaluated at :attr:`nodes`, in node order.
        """
        g = np.asarray(g_values, dtype=np.complex128)
        if g.shape != self.nodes.shape:
            raise ValueError(
                f"g_values shape {g.shape} != nodes shape {self.nodes.shape}"
            )
        return float(np.imag(np.sum(self.weights * g)))

    def band_energy(self, g_values: NDArray[np.complex128] | list) -> float:
        """Band(-structure) energy ``E_band = Im sum_k w_k z_k g(z_k)``.

        This is ``\\int E rho(E) dE`` over the occupied spectrum, the
        single-particle term of the total energy. Reuses the same nodes
        as :meth:`occupation` (the extra ``z_k`` factor is the only
        difference), so a converged contour gives charge and band energy
        in one sweep.
        """
        g = np.asarray(g_values, dtype=np.complex128)
        if g.shape != self.nodes.shape:
            raise ValueError(
                f"g_values shape {g.shape} != nodes shape {self.nodes.shape}"
            )
        return float(np.imag(np.sum(self.weights * self.nodes * g)))
