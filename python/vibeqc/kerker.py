"""Kerker preconditioner for periodic SCF density mixing.

The Kerker preconditioner damps small-G charge-density modes that cause
charge sloshing in metallic / small-gap periodic systems.  It is applied
to the density residual before mixing into the next SCF iteration.

Reference: Kerker, *Phys. Rev. B* 23, 3082 (1981).
"""

from __future__ import annotations

import numpy as np


def kerker_grid_shape(lattice: np.ndarray, spacing_bohr: float = 0.3):
    """Return a suitable grid shape for the Kerker preconditioner.

    Delegates to the same ``auto_grid`` helper used by the FFT-Poisson
    solver for consistency.

    Parameters
    ----------
    lattice : ndarray, shape (3, 3)
        Lattice vectors in bohr (column-wise).
    spacing_bohr : float
        Desired grid spacing. Default 0.3 bohr (same as FFT-Poisson).

    Returns
    -------
    nx, ny, nz : int
        Grid dimensions (even-integer-rounded).
    """
    from .ewald_j import auto_grid as _auto_grid

    shape = _auto_grid(lattice, spacing_bohr)
    return shape.nx, shape.ny, shape.nz


def kerker_g_squared(lattice: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """Compute |G|^2 on the FFT grid for the Kerker filter.

    Parameters
    ----------
    lattice : ndarray, shape (3, 3)
        Lattice vectors in bohr (column-wise).
    nx, ny, nz : int
        Grid dimensions.

    Returns
    -------
    g2 : ndarray, shape (nx, ny, nz)
        Squared reciprocal-space distance at each grid point.
    """
    # Reciprocal lattice vectors.
    volume = np.linalg.det(lattice)
    b1 = 2 * np.pi * np.cross(lattice[:, 1], lattice[:, 2]) / volume
    b2 = 2 * np.pi * np.cross(lattice[:, 2], lattice[:, 0]) / volume
    b3 = 2 * np.pi * np.cross(lattice[:, 0], lattice[:, 1]) / volume

    # Grid indices with FFTW ordering (0..n-1, then negative frequencies).
    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)
    # Shift to FFT ordering: [0, 1, ..., n/2-1, -n/2, ..., -1]
    i_shift = np.where(i > nx // 2, i - nx, i)
    j_shift = np.where(j > ny // 2, j - ny, j)
    k_shift = np.where(k > nz // 2, k - nz, k)

    I, J, K = np.meshgrid(i_shift, j_shift, k_shift, indexing="ij")

    G = (
        I[:, :, :, np.newaxis] * b1[np.newaxis, np.newaxis, np.newaxis, :]
        + J[:, :, :, np.newaxis] * b2[np.newaxis, np.newaxis, np.newaxis, :]
        + K[:, :, :, np.newaxis] * b3[np.newaxis, np.newaxis, np.newaxis, :]
    )

    g2 = np.sum(G**2, axis=-1)
    return g2


def kerker_kernel(g2: np.ndarray, k0: float = 1.0) -> np.ndarray:
    """Compute the Kerker preconditioning kernel.

    The filter is::

        f(G) = G^2 / (G^2 + k0^2)

    where ``k0 = 2pi / l`` and ``l`` is the screening length in bohr
    (default: ``k0 = 1.0`` -> l ≈ 6.28 bohr, which is reasonable for
    typical metallic systems).

    Parameters
    ----------
    g2 : ndarray
        Squared reciprocal-space distances from :func:`kerker_g_squared`.
    k0 : float
        Screening wave-vector (``2pi / l``). Default 1.0.

    Returns
    -------
    kernel : ndarray
        Kerker filter values, same shape as ``g2``.  The G=0 component
        is set to 0 (no mixing of the average density).
    """
    kernel = np.where(g2 > 1e-30, g2 / (g2 + k0**2), 0.0)
    return kernel


def kerker_k0_thomas_fermi(
    n_electrons: float,
    cell_volume_bohr3: float,
) -> float:
    """Thomas-Fermi screening constant for the Kerker filter, from the cell.

    The Kerker filter needs a screening wave-vector ``k0``. Rather than a
    hardcoded guess, the defining paper takes it near the Thomas-Fermi
    value of the equivalent homogeneous electron gas.

    # Eq. (8) of Kerker, *Phys. Rev. B* 23, 3082 (1981),
    # doi:10.1103/PhysRevB.23.3082, p. 3083:
    #   lambda ~ (4 k_F / pi)^{1/2}
    # with k_F the Fermi wavevector of the equivalent uniform gas,
    #   k_F = (3 pi^2 n)^{1/3},   n = N_electrons / V_cell
    # (Hartree atomic units, a_0 = 1). Manninen et al. chose lambda around
    # this value; Kerker adopts the same prescription.

    Published target: the paper's Ca(001) seven-layer film example is
    quoted at ``r_s = 3.3 a.u.`` with "a screening length of
    2 pi / lambda = 7.3 a.u." (p. 3084). Feeding the ``r_s = 3.3`` density
    ``n = 3 / (4 pi r_s^3)`` through the formula above reproduces
    ``2 pi / lambda = 7.302 a.u.``, which is the validation pinned in
    ``tests/test_kerker.py``.

    Parameters
    ----------
    n_electrons
        Electrons per unit cell.
    cell_volume_bohr3
        Unit-cell volume in bohr^3.

    Returns
    -------
    float
        ``k0`` in bohr^-1, suitable for :func:`kerker_kernel`. The
        corresponding real-space screening length is ``2 pi / k0``.

    Notes
    -----
    This is an *estimate for a metallic system*, and deliberately not a
    default: it is the right starting point for a free-electron-like
    metal, and too aggressive for a gapped system, where the physical
    screening length is much longer than the Thomas-Fermi one. Kerker
    also notes (p. 3083) that near self-consistency ``lambda`` may be
    reduced to speed convergence, so treat this as the initial value of a
    parameter, not a constant of the calculation.
    """
    if n_electrons <= 0.0:
        raise ValueError(
            f"kerker_k0_thomas_fermi: n_electrons must be > 0, got {n_electrons}"
        )
    if cell_volume_bohr3 <= 0.0:
        raise ValueError(
            f"kerker_k0_thomas_fermi: cell_volume_bohr3 must be > 0, "
            f"got {cell_volume_bohr3}"
        )
    density = float(n_electrons) / float(cell_volume_bohr3)
    k_fermi = (3.0 * np.pi**2 * density) ** (1.0 / 3.0)
    return float(np.sqrt(4.0 * k_fermi / np.pi))


def apply_kerker_preconditioner(
    density_residual: np.ndarray,
    lattice: np.ndarray,
    k0: float = 1.0,
    spacing_bohr: float = 0.3,
) -> np.ndarray:
    """Apply the Kerker preconditioner to a density residual on a 3D grid.

    Parameters
    ----------
    density_residual : ndarray, shape (nx, ny, nz)
        The density residual (r_out - r_in) on the FFT grid.
    lattice : ndarray, shape (3, 3)
        Lattice vectors in bohr (column-wise).
    k0 : float
        Kerker screening parameter. Default 1.0.
    spacing_bohr : float
        Grid spacing for auto_grid. Default 0.3.

    Returns
    -------
    preconditioned : ndarray, same shape as ``density_residual``
        The density residual after Kerker filtering.
    """
    nx, ny, nz = density_residual.shape
    g2 = kerker_g_squared(lattice, nx, ny, nz)
    kernel = kerker_kernel(g2, k0=k0)

    # FFT to reciprocal space.
    rho_G = np.fft.fftn(density_residual)

    # Apply the Kerker filter.
    rho_G_filtered = rho_G * kernel

    # Transform back.
    preconditioned = np.real(np.fft.ifftn(rho_G_filtered))
    return preconditioned


def kerker_mix_density(
    rho_in: np.ndarray,
    rho_out: np.ndarray,
    lattice: np.ndarray,
    mixing_fraction: float = 0.5,
    k0: float = 1.0,
) -> np.ndarray:
    """Mix two densities with Kerker preconditioning.

    ``r_new = r_in + mixing_fraction * K(r_out - r_in)``

    where ``K`` is the Kerker preconditioner.

    Parameters
    ----------
    rho_in : ndarray
        Input density (SCF iteration n).
    rho_out : ndarray
        Output density from diagonalising F[r_in] (SCF iteration n).
    lattice : ndarray, shape (3, 3)
        Lattice vectors in bohr.
    mixing_fraction : float
        Linear mixing fraction after preconditioning. Default 0.5.
    k0 : float
        Kerker parameter. Default 1.0.

    Returns
    -------
    rho_new : ndarray
        Mixed density for the next SCF iteration.
    """
    residual = rho_out - rho_in
    preconditioned = apply_kerker_preconditioner(residual, lattice, k0=k0)
    return rho_in + mixing_fraction * preconditioned


class PulayKerkerMixer:
    """Pulay-Kerker density mixing for periodic SCF.

    Combines Kerker-preconditioned residuals with Pulay (DIIS-style)
    extrapolation from a history of densities and residuals.  This is
    VASP's default mixing scheme for metallic systems.

    Reference: Kresse & Furthmüller, *Phys. Rev. B* 54, 11169 (1996).
    """

    def __init__(
        self,
        lattice: np.ndarray,
        history: int = 8,
        mixing_fraction: float = 0.5,
        k0: float = 1.0,
        tikhonov_reg: float = 1e-10,
    ):
        self.lattice = lattice
        self.max_history = history
        self.mixing_fraction = mixing_fraction
        self.k0 = k0
        self.tikhonov_reg = tikhonov_reg

        # History buffers.
        self._rho_hist: list[np.ndarray] = []
        self._resid_hist: list[np.ndarray] = []

    def apply(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        """Compute the mixed density for the next SCF iteration.

        Parameters
        ----------
        rho_in : ndarray
            Input density for this SCF iteration.
        rho_out : ndarray
            Output density from diagonalising F[rho_in].

        Returns
        -------
        rho_new : ndarray
            Mixed density after Pulay-Kerker extrapolation.
        """
        residual = rho_out - rho_in

        # Kerker-precondition the residual.
        prec_resid = apply_kerker_preconditioner(residual, self.lattice, k0=self.k0)

        self._rho_hist.append(rho_out.ravel())
        self._resid_hist.append(prec_resid.ravel())
        if len(self._rho_hist) > self.max_history:
            self._rho_hist.pop(0)
            self._resid_hist.pop(0)

        m = len(self._resid_hist)
        if m < 2:
            return rho_in + self.mixing_fraction * prec_resid

        # Build the Pulay B-matrix from preconditioned residuals.
        B = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                B[i, j] = self._resid_hist[i] @ self._resid_hist[j]
        B += self.tikhonov_reg * np.eye(m)

        # Augmented system with S c_i = 1 constraint.
        B_aug = np.zeros((m + 1, m + 1))
        B_aug[:m, :m] = B
        B_aug[:m, m] = 1.0
        B_aug[m, :m] = 1.0
        rhs = np.zeros(m + 1)
        rhs[m] = 1.0

        try:
            c = np.linalg.lstsq(B_aug, rhs, rcond=None)[0][:m]
        except np.linalg.LinAlgError:
            return rho_in + self.mixing_fraction * prec_resid

        # Extrapolate density.
        rho_extrap = np.zeros_like(rho_out.ravel())
        for i in range(m):
            rho_extrap += c[i] * self._rho_hist[i]
        rho_new = rho_extrap.reshape(rho_out.shape)

        # Apply Kerker to the extrapolated residual as a final correction.
        final_resid = rho_new - rho_in
        prec_final = apply_kerker_preconditioner(final_resid, self.lattice, k0=self.k0)
        return rho_in + self.mixing_fraction * prec_final

    def reset(self) -> None:
        """Clear the history."""
        self._rho_hist.clear()
        self._resid_hist.clear()


class JohnsonEyertMixer:
    """Modified Broyden II / Johnson-Eyert density mixer.

    A multi-secant quasi-Newton method that builds an approximate
    inverse Jacobian from the history of density changes and
    residuals.  Default in many plane-wave codes (VASP, ABINIT).

    Reference: Johnson, *Phys. Rev. B* 38, 12807 (1988);
    Eyert, *J. Comput. Phys.* 124, 271 (1996).
    """

    def __init__(
        self,
        lattice: np.ndarray,
        history: int = 20,
        mixing_fraction: float = 0.7,
        k0: float = 0.0,
        tikhonov_reg: float = 1e-10,
    ):
        self.lattice = lattice
        self.max_history = history
        self.mixing_fraction = mixing_fraction
        self.k0 = k0
        self.tikhonov_reg = tikhonov_reg

        # History vectors.
        self._DeltaR: list[np.ndarray] = []  # ΔR = R_new - R_prev
        self._DeltaD: list[np.ndarray] = []  # Δr = r_new - r_prev

        # Previous step values.
        self._rho_prev: np.ndarray | None = None
        self._resid_prev: np.ndarray | None = None

    def apply(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        residual = rho_out - rho_in

        # Optional Kerker preconditioning of the raw residual.
        if self.k0 > 0.0:
            prec_resid = apply_kerker_preconditioner(residual, self.lattice, k0=self.k0)
        else:
            prec_resid = residual

        if self._rho_prev is None:
            self._rho_prev = rho_out.copy()
            self._resid_prev = prec_resid.copy()
            return rho_in + self.mixing_fraction * prec_resid

        # Compute changes.
        dR = prec_resid.ravel() - self._resid_prev.ravel()
        dD = rho_out.ravel() - self._rho_prev.ravel()

        self._DeltaR.append(dR)
        self._DeltaD.append(dD)
        if len(self._DeltaR) > self.max_history:
            self._DeltaR.pop(0)
            self._DeltaD.pop(0)

        self._rho_prev = rho_out.copy()
        self._resid_prev = prec_resid.copy()

        m = len(self._DeltaR)
        if m < 1:
            return rho_in + self.mixing_fraction * prec_resid

        # Johnson-Eyert update.
        f0 = prec_resid.ravel()
        for i in range(m):
            denom = self._DeltaR[i] @ self._DeltaR[i] + self.tikhonov_reg
            gamma = (f0 @ self._DeltaR[i]) / denom
            # Weight: w = ΔD - ΔR
            w = self._DeltaD[i] - self._DeltaR[i]
            f0 += gamma * w * self.mixing_fraction

        rho_new = rho_in + f0.reshape(rho_in.shape)
        return rho_new

    def reset(self) -> None:
        self._DeltaR.clear()
        self._DeltaD.clear()
        self._rho_prev = None
        self._resid_prev = None


# ---------------------------------------------------------------------------
# Simple Kerker-style density-matrix mixing (no FFT grid needed)
# ---------------------------------------------------------------------------

def kerker_density_matrix_mix(
    D_prev: "np.ndarray",
    D_new: "np.ndarray",
    S: "np.ndarray",
    mixing: float = 0.5,
    k0: float = 1.0,
) -> "np.ndarray":
    """Kerker-inspired mixing for AO density matrices.

    Applies a Kerker-style low-pass filter to the AO-basis density
    residual by projecting through the overlap matrix:

        ΔD_filtered = S^{-1/2}.kerker_filter(S^{1/2}.ΔD.S^{1/2}).S^{-1/2}

    where the Kerker filter damps high-frequency components of the
    orthogonalised density residual.  This is an approximation to the
    full FFT-grid Kerker preconditioner that works directly in the AO
    basis without grid evaluation.

    Parameters
    ----------
    D_prev, D_new
        Previous and new density matrices (nbf, nbf).
    S
        AO overlap matrix.
    mixing
        Linear mixing fraction.  Default 0.5.
    k0
        Kerker screening parameter.  Smaller = more aggressive filtering.
        Default 1.0.

    Returns
    -------
    D_mixed : ndarray, shape (nbf, nbf)
    """
    import numpy as np

    # Symmetric orthogonaliser: X = S^{-1/2}
    evals, evecs = np.linalg.eigh(S)
    # Regularise: drop near-zero eigenvalues
    mask = evals > 1e-12
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[mask] = 1.0 / np.sqrt(evals[mask])
    X = evecs @ np.diag(inv_sqrt) @ evecs.T

    # Orthogonalise the residual
    dD = D_new - D_prev
    dD_orth = X.T @ dD @ X

    # Kerker-style low-pass: threshold based on eigenvalue magnitudes
    # of the orthogonalised residual (higher eigenvalues = finer spatial
    # features = more damping).
    evals_d, evecs_d = np.linalg.eigh(dD_orth)
    # Damp large eigenvalues (high-frequency components)
    damp = k0**2 / (np.abs(evals_d) + k0**2)
    dD_filtered = evecs_d @ np.diag(evals_d * damp) @ evecs_d.T

    # Back to AO basis
    dD_ao = X @ dD_filtered @ X.T

    return D_prev + mixing * dD_ao
