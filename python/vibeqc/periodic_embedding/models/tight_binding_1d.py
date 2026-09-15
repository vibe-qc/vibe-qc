"""1D semi-infinite tight-binding chain -- analytic embedding test bed.

Open design question 4: validate the contour-integration, Lloyd's-formula
and Sancho-Rubio machinery on a 1D model substrate with *closed-form*
targets before wiring in the full 3D bulk Green function. This chain is
that model: every quantity below has an analytic expression, so the
numerical backbone can be checked to machine precision.

Nearest-neighbour chain, sites ``n = 0`` (surface), ``1, 2, ...`` running
into the bulk; on-site energy ``onsite`` (eps), hopping ``hop`` (t). The
surface site sees a semi-infinite ideal substrate below it -- exactly the
1D analogue of region II in Layer A.

Closed forms used as validation targets (textbook tight-binding Green
functions; e.g. E. N. Economou, "Green's Functions in Quantum Physics,"
Springer):

* Surface Green function (decaying root of ``t^2 g^2 - (z-eps) g + 1 = 0``)::

      g_s(z) = [(z-eps) - sqrt((z-eps)^2 - 4 t^2)] / (2 t^2)

* Surface local density of states -- a Wigner semicircle of radius 2|t|::

      rho_s(E) = sqrt(4 t^2 - (E-eps)^2) / (2 pi t^2),   |E-eps| < 2|t|

  Reference value: rho_s integrates to exactly 1 over the band, and at
  half filling (E_F = eps) the surface-site occupation is exactly 0.5 by
  particle-hole symmetry.

* Surface-site occupation up to E_F (analytic integral of rho_s)::

      N(E_F) = 0.5 + u_F sqrt(4 t^2 - u_F^2)/(4 pi t^2)
                   + arcsin(u_F/(2|t|))/pi,        u_F = E_F - eps

* Embedding self-energy on the surface site::

      Sigma_emb(z) = t^2 g_s(z)

The semi-infinite surface GF is exactly what Sancho-Rubio decimation must
reproduce from the blocks ``H00 = eps``, ``H01 = t``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SemiInfiniteChain1D:
    """Semi-infinite 1D nearest-neighbour tight-binding chain.

    Parameters
    ----------
    onsite
        On-site energy ``eps`` (band centre).
    hop
        Nearest-neighbour hopping ``t`` (band half-width is ``2|t|``).
    """

    onsite: float = 0.0
    hop: float = 1.0

    # -- spectrum bookkeeping -------------------------------------------
    @property
    def band_bottom(self) -> float:
        return self.onsite - 2.0 * abs(self.hop)

    @property
    def band_top(self) -> float:
        return self.onsite + 2.0 * abs(self.hop)

    @property
    def band_width(self) -> float:
        return 4.0 * abs(self.hop)

    # -- Green functions ------------------------------------------------
    def surface_gf(self, z):
        """Analytic surface (site-0) Green function ``g_s(z)``.

        Decaying root of ``t^2 g^2 - (z-eps) g + 1 = 0``, branch chosen
        so ``Im g_s < 0`` for ``Im z > 0`` (retarded); for real ``z``
        outside the band the smaller-magnitude (decaying) root is taken.
        Accepts scalars or arrays.
        """
        eps, t = self.onsite, self.hop
        w = np.asarray(z, dtype=np.complex128) - eps
        disc = np.sqrt(w * w - 4.0 * t * t + 0j)
        gp = (w + disc) / (2.0 * t * t)
        gm = (w - disc) / (2.0 * t * t)
        # Retarded selection: prefer the more-negative-imaginary root;
        # on the real axis (degenerate Im) fall back to smaller |g|.
        diff = gm.imag - gp.imag
        pick_gm = np.where(
            np.abs(diff) > 1e-14, gm.imag < gp.imag, np.abs(gm) < np.abs(gp)
        )
        g = np.where(pick_gm, gm, gp)
        return complex(g) if g.ndim == 0 else g

    def bulk_gf(self, z):
        """On-site Green function of the *infinite* chain,
        ``1/sqrt((z-eps)^2 - 4 t^2)`` (retarded branch). Evaluate at
        ``Im z > 0`` (contour / ``E + i*eta``)."""
        eps, t = self.onsite, self.hop
        w = np.asarray(z, dtype=np.complex128) - eps
        disc = np.sqrt(w * w - 4.0 * t * t + 0j)
        g = 1.0 / disc
        g = np.where(g.imag <= 0, g, -g)  # enforce retarded Im <= 0
        return complex(g) if g.ndim == 0 else g

    def embedding_self_energy(self, z):
        """Embedding potential the substrate imposes on the surface site,
        ``Sigma_emb(z) = t^2 g_s(z)``."""
        return self.hop**2 * self.surface_gf(z)

    # -- spectral quantities --------------------------------------------
    def ldos(self, energy, eta: float = 1e-6):
        """Surface LDOS by broadening: ``-(1/pi) Im g_s(E + i*eta)``."""
        g = self.surface_gf(np.asarray(energy, dtype=float) + 1j * eta)
        return -(1.0 / np.pi) * np.imag(g)

    def ldos_analytic(self, energy):
        """Closed-form surface LDOS (Wigner semicircle of radius 2|t|)."""
        eps, t = self.onsite, self.hop
        u = np.asarray(energy, dtype=float) - eps
        inside = np.abs(u) < 2.0 * abs(t)
        val = np.sqrt(np.clip(4.0 * t * t - u * u, 0.0, None)) / (
            2.0 * np.pi * t * t
        )
        return np.where(inside, val, 0.0)

    def occupation_analytic(self, e_fermi: float) -> float:
        """Closed-form surface-site occupation (per spin) up to ``e_fermi``.

        Reference: at ``e_fermi = eps`` (half filling) this is exactly 0.5;
        at ``e_fermi >= band_top`` it is exactly 1.
        """
        eps, t = self.onsite, self.hop
        tt = abs(t)
        u_f = e_fermi - eps
        if u_f <= -2.0 * tt:
            return 0.0
        if u_f >= 2.0 * tt:
            return 1.0
        return (
            0.5
            + u_f * np.sqrt(4.0 * t * t - u_f * u_f) / (4.0 * np.pi * t * t)
            + np.arcsin(u_f / (2.0 * tt)) / np.pi
        )

    def band_energy_analytic(self, e_fermi: float) -> float:
        """Closed-form band energy ``\\int_{bottom}^{E_F} E rho_s(E) dE``.

        Derived term-by-term from the semicircle::

            E_band(E_F) = eps * N(E_F) - (4 t^2 - u_F^2)^{3/2} / (6 pi t^2)
        """
        eps, t = self.onsite, self.hop
        tt = abs(t)
        u_f = e_fermi - eps
        if u_f <= -2.0 * tt:
            return 0.0
        occ = self.occupation_analytic(e_fermi)
        if u_f >= 2.0 * tt:
            return eps * occ  # (4t^2 - (2t)^2)^{3/2} = 0
        return eps * occ - (4.0 * t * t - u_f * u_f) ** 1.5 / (
            6.0 * np.pi * t * t
        )

    # -- single-site impurity (Layer-B analogue) ------------------------
    def impurity_surface_gf(self, z, delta: float):
        """Surface GF with an on-site shift ``delta`` at site 0
        (scalar Dyson, ``G = g_s / (1 - g_s * delta)``)."""
        g = self.surface_gf(z)
        return g / (1.0 - g * delta)

    def bound_state_energy(self, delta: float) -> float | None:
        """Energy of the impurity bound state outside the band, or ``None``.

        ``1 - g_s(E_b) delta = 0`` has a real solution outside the band
        only for ``|delta| > |t|``. Above the band ``g_s`` runs
        monotonically from ``1/t`` (at the edge) to 0, so
        ``E_b = eps + (delta + t^2/delta)`` (and mirror below).
        """
        eps, t = self.onsite, self.hop
        if abs(delta) <= abs(t):
            return None
        # Solve t^2/(E-eps - ... ) -- from g_s(E) = 1/delta with the
        # outside-band real branch g_s = [w - sign(w) sqrt(w^2-4t^2)]/(2t^2):
        #   1/delta = g_s  =>  E_b = eps + delta + t^2/delta.
        return eps + delta + t * t / delta

    def finite_chain_hamiltonian(
        self, n_sites: int, delta: float = 0.0
    ) -> NDArray[np.float64]:
        """Dense ``(n_sites, n_sites)`` tridiagonal Hamiltonian of a *finite*
        chain, with an optional on-site shift ``delta`` at the end site 0.

        Used by the tests as an independent (direct-diagonalization) check
        on the semi-infinite Green-function results: for large ``n_sites``
        the end-site quantities converge to the semi-infinite chain.
        """
        eps, t = self.onsite, self.hop
        h = np.zeros((n_sites, n_sites), dtype=np.float64)
        idx = np.arange(n_sites)
        h[idx, idx] = eps
        h[idx[:-1], idx[1:]] = t
        h[idx[1:], idx[:-1]] = t
        h[0, 0] += delta
        return h
