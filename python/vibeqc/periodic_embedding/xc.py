"""DFT exchange-correlation integration for the embedded SCF.

Computes Vxc on the region-I DFT grid via vibe-qc's libxc wrapper
(:class:`~vibeqc._vibeqc_core.Functional`) and assembles the XC
potential matrix in the Gaussian basis.  No new dependency needed --
libxc is already linked and the ``eval_unpolarised`` method is
exposed on the Python ``Functional`` object.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .._vibeqc_core import (
    Functional,
    Grid,
    GridOptions,
    build_grid,
    evaluate_ao_with_gradient,
)


def build_vxc_from_density(
    mol,
    basis,
    region,
    d_local: NDArray[np.float64],
    *,
    functional_name: str = "PBE",
    grid_opts=None,
) -> tuple[NDArray[np.float64], float]:
    """Build Vxc and E_xc from a region-I density matrix.

    Uses vibe-qc's existing grid build + AO evaluation + libxc wrapper.
    No new dependencies required.

    Parameters
    ----------
    mol
        Molecule for grid construction (system.unit_cell_molecule()).
    basis
        Orbital basis set.
    region
        Region partition (defines i_ao subset).
    d_local
        Region-I density matrix, (n_i, n_i).
    functional_name
        XC functional: ``"LDA"``, ``"PBE"``, ``"BLYP"``, etc.
    grid_opts
        Integration grid options.  Defaults to a sparse grid suitable
        for testing.

    Returns
    -------
    (vxc_i, e_xc)
        vxc_i: (n_i, n_i) XC potential matrix.
        e_xc: XC energy in Ha.
    """
    if grid_opts is None:
        grid_opts = GridOptions()

    # Build grid on the molecule.
    grid: Grid = build_grid(mol, grid_opts)
    points = np.asarray(grid.points, dtype=float)
    weights = np.asarray(grid.weights, dtype=float)

    # Evaluate AOs and gradients on the grid (full basis).
    chi_vals, chi_grad_x, chi_grad_y, chi_grad_z = evaluate_ao_with_gradient(
        basis, points
    )
    i_ao = region.i_ao
    chi = np.asarray(chi_vals, dtype=float)[:, i_ao]  # (n_pts, n_i)
    chi_x = np.asarray(chi_grad_x, dtype=float)[:, i_ao]
    chi_y = np.asarray(chi_grad_y, dtype=float)[:, i_ao]
    chi_z = np.asarray(chi_grad_z, dtype=float)[:, i_ao]

    # Density on the grid: rho_g = S_{muν} chi_gmu D_muν chi_gν
    rho = np.einsum("gi,ij,gj->g", chi, d_local, chi, optimize=True)

    # Gradient of density: d_c r(g) = 2 S_{muν} (d_c chi_gmu) D_muν chi_gν
    drho_x = 2.0 * np.einsum("gi,ij,gj->g", chi_x, d_local, chi, optimize=True)
    drho_y = 2.0 * np.einsum("gi,ij,gj->g", chi_y, d_local, chi, optimize=True)
    drho_z = 2.0 * np.einsum("gi,ij,gj->g", chi_z, d_local, chi, optimize=True)

    # Sigma: s = |gradr|^2
    sigma = drho_x**2 + drho_y**2 + drho_z**2

    # Evaluate XC functional via libxc.
    func = Functional(functional_name)
    exc, v_rho, v_sigma = func.eval_unpolarised(rho, sigma)

    # XC energy: E_xc = S_g w_g . exc[g]
    e_xc = float(np.dot(weights, exc))

    # Build Vxc matrix.
    # LDA part: V^{LDA}_{muν} = S_g w_g v_r(g) chi_gmu chi_gν
    w_vrho = weights * v_rho
    vxc = chi.T @ (w_vrho[:, None] * chi)

    if func.kind.name != "LDA" and len(v_sigma) > 0:
        # GGA part: V^{GGA}_{muν} = S_g w_g . 2 v_s(g) . (gradr(g).gradchi_gmu . chi_gν)
        # This is the "half" contribution; the full GGA potential is symmetric.
        w_vsigma = weights * v_sigma
        # First term: v_s . (d_c r) . (d_c chi_mu) . chi_ν  summed over c
        grad_rho_dot_grad_chi = (
            drho_x[:, None] * chi_x + drho_y[:, None] * chi_y + drho_z[:, None] * chi_z
        )
        vxc_gga_half = chi.T @ (w_vsigma[:, None] * grad_rho_dot_grad_chi)
        # The full GGA + LDA potential:
        vxc = vxc + 2.0 * vxc_gga_half
        vxc = 0.5 * (vxc + vxc.T)

    return np.asarray(vxc, dtype=float), e_xc
