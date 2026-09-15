"""Lloyd's formula -- adsorption energetics through the Green function.

Rather than differencing total energies of separate calculations (clean
slab, adsorbate, combined), the change in the integrated density of
states caused by the localized perturbation ``Delta_V`` is given in
closed form by Lloyd's formula:

    Delta_N(E) = -(1/pi) Im Tr ln(1 - G0(E) Delta_V)
               = -(1/pi) Im ln det(1 - G0(E) Delta_V).

``Delta_N(E)`` is the number of states (occupied + virtual up to E) added
by the perturbation; its energy derivative is the change in DOS,
``Delta_rho(E) = -(1/pi) Im d/dE ln det(1 - G0 Delta_V)``. The
band-structure contribution to the adsorption energy is then
``\\int E Delta_rho(E) dE`` (plus double-counting + electrostatic
corrections supplied by the SCF, handled outside this module).

Bound states pulled out of the band by ``Delta_V`` show up as ``2 pi``
windings of ``arg det``; tracking them correctly is just a matter of
following the phase continuously in energy (``numpy.unwrap``), which is
why :func:`lloyd_integrated_dos_change` takes an *ordered* energy sweep
rather than isolated points.

Reference: P. Lloyd, Proc. Phys. Soc. 90, 207 (1967); as used for
embedding in J. E. Inglesfield, "The Embedding Method for Electronic
Structure," IOP Publishing (2015), doi:10.1088/978-0-7503-1042-0.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _as_matrix_stack(g0_values) -> NDArray[np.complex128]:
    """Coerce a sequence of scalars / (n, n) blocks to an (K, n, n) stack."""
    arr = np.asarray(g0_values, dtype=np.complex128)
    if arr.ndim == 1:  # K scalars -> K (1, 1) blocks
        arr = arr.reshape(arr.shape[0], 1, 1)
    elif arr.ndim != 3:
        raise ValueError(
            "g0_values must be a sequence of scalars or of (n, n) blocks"
        )
    return arr


def lloyd_integrated_dos_change(
    g0_values,
    delta_v: NDArray[np.complex128] | complex,
) -> NDArray[np.float64]:
    """Integrated DOS change ``Delta_N(E)`` along an ordered energy sweep.

    Parameters
    ----------
    g0_values
        Clean-surface Green function (restricted to the perturbed region)
        at successive energies ``E_k + i*eta``, ordered by increasing
        ``E_k``. Either a 1D array of scalars (single-site model) or a
        ``(K, n, n)`` stack.
    delta_v
        Localized perturbation, ``(n, n)`` (or scalar), constant in energy.

    Returns
    -------
    Delta_N
        Real array, ``Delta_N(E_k)``, with the phase unwrapped in energy
        and anchored so ``Delta_N`` of the first (sub-band) node is zero.
        For the anchoring to be physical the first node must sit below the
        perturbed spectrum (no displaced charge yet).
    """
    g0 = _as_matrix_stack(g0_values)
    k, n, _ = g0.shape
    dv = np.atleast_2d(np.asarray(delta_v, dtype=np.complex128))
    if dv.shape != (n, n):
        raise ValueError(f"delta_v shape {dv.shape} != (n, n) = {(n, n)}")

    ident = np.eye(n, dtype=np.complex128)
    dets = np.empty(k, dtype=np.complex128)
    for i in range(k):
        dets[i] = np.linalg.det(ident - g0[i] @ dv)

    # Delta_N = -(1/pi) Im ln det = -(1/pi) * arg(det), unwrapped in
    # energy to follow bound-state windings, anchored to 0 below the band.
    phase = np.unwrap(np.angle(dets))
    phase = phase - phase[0]
    return -(1.0 / np.pi) * phase


def lloyd_friedel_sum(
    g0_values,
    delta_v: NDArray[np.complex128] | complex,
) -> float:
    """Total displaced charge (Friedel sum) = ``Delta_N`` at the top node.

    The energy sweep must span from below the band to above the Fermi
    level (or to the top of the band for the total displaced charge).
    """
    return float(lloyd_integrated_dos_change(g0_values, delta_v)[-1])


def lloyd_band_energy_change(
    energies: NDArray[np.float64],
    delta_n: NDArray[np.float64],
    e_fermi: float,
) -> float:
    """Band-energy change from ``Delta_N(E)`` by parts.

    ``\\int_{-inf}^{E_F} E Delta_rho dE = E_F Delta_N(E_F)
       - \\int_{-inf}^{E_F} Delta_N(E) dE`` (integration by parts, using
    ``Delta_N`` -> 0 below the band). Trapezoidal on the supplied grid.

    Parameters
    ----------
    energies
        Real energy grid (ascending), same length as ``delta_n``.
    delta_n
        Integrated DOS change from :func:`lloyd_integrated_dos_change`.
    e_fermi
        Fermi level; the integral runs up to the last grid point at or
        below ``e_fermi``.
    """
    energies = np.asarray(energies, dtype=np.float64)
    delta_n = np.asarray(delta_n, dtype=np.float64)
    if energies.shape != delta_n.shape:
        raise ValueError("energies and delta_n must have the same shape")
    mask = energies <= e_fermi
    if not mask.any():
        return 0.0
    e_occ = energies[mask]
    dn_occ = delta_n[mask]
    integral_dn = np.trapezoid(dn_occ, e_occ)
    return float(e_fermi * dn_occ[-1] - integral_dn)
