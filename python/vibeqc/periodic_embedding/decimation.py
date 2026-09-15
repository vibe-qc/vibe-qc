"""Semi-infinite substrate Green function by principal-layer decimation.

The Layer-A embedding potential is built from the Green function of the
bulk half-space (region II). For a substrate described by principal
layers -- blocks ``H00`` (intra-layer Hamiltonian) coupled to the next
layer deeper into the bulk by ``H01`` -- the surface Green function of the
semi-infinite stack is the fixed point of

    g_s = (z - H00 - H01 g_s H01^dagger)^{-1},

i.e. the self-energy that the rest of the half-space imposes on the
topmost layer is ``Sigma = H01 g_s H01^dagger``. Solving this by naive
layer-by-layer addition converges linearly in the number of layers; the
Lopez Sancho / Lopez Sancho / Rubio decimation renormalises layers in
*powers of two*, so the coupling between the retained surface block and
the (ever deeper) effective layer falls off super-exponentially and a
few iterations suffice.

Reference: M. P. Lopez Sancho, J. M. Lopez Sancho, J. Rubio,
"Highly convergent schemes for the calculation of bulk and surface Green
functions," J. Phys. F: Met. Phys. 15, 851 (1985),
doi:10.1088/0305-4608/15/4/009 (the fast scheme, Eqs. 8-15).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class DecimationNotConvergedError(RuntimeError):
    """Sancho-Rubio iteration hit ``max_iter`` without converging.

    Almost always means ``z`` is too close to the real axis (vanishing
    imaginary part inside the band): add a small ``+i eta`` or evaluate
    on the complex contour, where convergence is fast.
    """


def sancho_rubio_surface_gf(
    h00: NDArray[np.complex128] | complex,
    h01: NDArray[np.complex128] | complex,
    z: complex,
    *,
    tol: float = 1e-12,
    max_iter: int = 64,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Surface and bulk Green functions of a semi-infinite stack at ``z``.

    Parameters
    ----------
    h00
        Intra-layer (on-site) Hamiltonian block of one principal layer,
        shape ``(m, m)`` (scalars / ``(1, 1)`` for a 1D chain).
    h01
        Coupling from a layer to the next layer *deeper into the
        substrate*, shape ``(m, m)``. The upward coupling is its
        conjugate transpose ``H10 = H01^dagger``.
    z
        Complex energy (use ``E + i*eta`` with ``eta > 0``, or a point on
        the upper-half-plane contour).
    tol
        Convergence threshold on ``max(|alpha|, |beta|)`` -- the residual
        coupling between the surface block and the decimated remainder.
    max_iter
        Iteration cap (each iteration doubles the effective stack depth,
        so 64 is astronomically deep; convergence is typically < 20).

    Returns
    -------
    (g_surface, g_bulk)
        ``g_surface`` is the surface (topmost-layer) Green function
        ``(z - H00 - Sigma)^{-1}``; ``g_bulk`` is the bulk on-site Green
        function ``(z - H00 - Sigma_bulk)^{-1}`` with self-energy from
        *both* sides. Both are ``(m, m)`` arrays.

    Notes
    -----
    The surface self-energy is recovered as
    ``Sigma_emb = z*I - H00 - inv(g_surface)`` (used by Layer A as the
    embedding potential on the dividing plane).
    """
    h00 = np.atleast_2d(np.asarray(h00, dtype=np.complex128))
    h01 = np.atleast_2d(np.asarray(h01, dtype=np.complex128))
    m = h00.shape[0]
    if h00.shape != (m, m) or h01.shape != (m, m):
        raise ValueError("h00 and h01 must be square and the same size")

    ident = np.eye(m, dtype=np.complex128)
    z_ident = z * ident

    # Sancho-Rubio (1985) fast iteration. eps_s accumulates the surface
    # on-site (one-sided self-energy, into the bulk only); eps_b the bulk
    # on-site (two-sided). alpha/beta are the renormalised down/up
    # couplings, squared away each step.
    eps_s = h00.copy()
    eps_b = h00.copy()
    alpha = h01.copy()  # H01: couples a layer to the one below (into bulk)
    beta = h01.conj().T.copy()  # H10 = H01^dagger: couples upward

    for _ in range(max_iter):
        g = np.linalg.inv(z_ident - eps_b)
        agb = alpha @ g @ beta  # = Sigma contribution from below
        bga = beta @ g @ alpha
        eps_s = eps_s + agb
        eps_b = eps_b + agb + bga
        alpha = alpha @ g @ alpha
        beta = beta @ g @ beta
        if max(np.abs(alpha).max(), np.abs(beta).max()) < tol:
            break
    else:
        raise DecimationNotConvergedError(
            f"Sancho-Rubio did not converge at z={z!r} in {max_iter} "
            f"iterations (residual coupling "
            f"{max(np.abs(alpha).max(), np.abs(beta).max()):.2e}); "
            "move z off the real axis."
        )

    g_surface = np.linalg.inv(z_ident - eps_s)
    g_bulk = np.linalg.inv(z_ident - eps_b)
    return g_surface, g_bulk
