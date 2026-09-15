"""Finite-displacement Γ-point phonons on the GAPW periodic route.

Builds the 3N x 3N dynamical matrix by central differences on the
*analytic GAPW atomic force* (not the energy). This is the same
approach as :func:`vibeqc.periodic_gapw_hessian.compute_hessian_gpw`
but targets the GAPW gradient and packages the phonon workflow into
a convenient :class:`PhononCalculator` class.

API
---
::

    from vibeqc.periodic_gapw_phonon import PhononCalculator

    pc = PhononCalculator(system=sys, basis=basis_obj,
                          basis_name="sto-3g", functional="lda",
                          grid=grid, gapw_kwargs=dict(lmax=3))
    dyn_mat = pc.compute_phonons(disp=0.01)
    freqs_cm1, freqs_thz, modes = pc.diagonalize_dynamical_matrix()

Or use the standalone functions::

    from vibeqc.periodic_gapw_phonon import (
        compute_dynamical_matrix_fd,
        phonon_eigenvalues,
        phonon_dos,
    )

    D = compute_dynamical_matrix_fd(system, basis, basis_name,
                                     functional="lda")
    freqs_cm1, freqs_thz, modes = phonon_eigenvalues(D, masses_amu)
    dos, omega_grid = phonon_dos(freqs_cm1, sigma=10.0, n_points=1000)

Algorithm
---------
For each Cartesian DOF ``q = (atom_i, axis_j)``:

1.  Displace atom ``i`` by ``+fd_step`` along axis ``j``.
2.  Re-run the GAPW SCF, seeded with the central-geometry density as
    the initial guess.
3.  Compute the analytic GAPW atomic gradient at the displaced
    geometry via :func:`compute_gradient_gapw`.
4.  Repeat for ``-fd_step``.
5.  The Hessian row for DOF ``q`` is
    ``H[q, :] = -(F(+q) - F(-q)) / (2 . fd_step)``
    or equivalently ``H[q, :] = (G(+q) - G(-q)) / (2 . fd_step)``
    with ``G = dE/dR`` the energy gradient (force *negative*).
6.  The mass-weighted dynamical matrix is
    ``D_{ia,jb} = H_{ia,jb} / sqrt(M_i M_j)``

The same symmetrisation ``H = 0.5 . (H + H^T)`` is applied before
mass-weighting to wash out finite-difference asymmetry.

Scope
-----
* Γ-point only (k = [1,1,1]). Multi-k phonons are not wired here.
* ``PhononCalculator.get_bandstructure`` can sample a q-path with a
  flat Γ-only band surface from the cached Γ dynamical matrix; it does
  not yet Fourier-interpolate intercell force constants.
* Closed-shell RHF / RKS only (odd electron counts raise).
* RHF requires an explicit ``gapw_kwargs={"one_centre": "block"}`` opt-in;
  RKS resolves an omitted or ``"auto"`` mode to block. The implemented
  gradient does not differentiate analytic/projector one-centre energies.

Convention notes
----------------
* Imaginary modes (negative w^2) are reported as **negative**
  wavenumbers, matching :mod:`vibeqc.hessian` and Gaussian / NWChem /
  ORCA / PySCF / ASE.
* Mass weighting uses atomic-mass units (amu). Atomic masses are
  resolved from the built-in table (:data:`vibeqc.properties._ATOMIC_MASSES`).
* Frequencies are returned in both cm⁻¹ and THz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Optional, Sequence

import numpy as np

from ._vibeqc_core import (
    Atom,
    BasisSet,
    Molecule,
    PeriodicSystem,
)
from .periodic_gapw_augment import run_periodic_rhf_gapw
from .periodic_gapw_gradient import compute_gradient_gapw
from .periodic_gapw_grid import PlaneWaveGrid
from .properties import _ATOMIC_MASSES

_log = logging.getLogger("vibeqc.periodic_gapw_phonon")


__all__ = [
    "PhononBandStructure",
    "PhononCalculator",
    "compute_dynamical_matrix_fd",
    "phonon_eigenvalues",
    "phonon_dos",
]


# CODATA 2018: 1 a.u. of angular frequency in cm⁻¹.
_HARTREE_TO_CM_INV: float = 219474.6313632

# 1 a.u. of angular frequency in THz (w/2pi).
# 1 Ha = 4.3597447222071e-18 J -> w [s⁻¹] = E/ℏ where ℏ = 1.054571817e-34 J.s
# w/2pi [THz] = (Ha_to_J / ℏ) / (2pi x 1e12)
# Ha_to_J = 4.3597447222071e-18, ℏ = 1.054571817e-34
# w [a.u.] / (2pi) x (Ha / ℏ) -> s⁻¹, then / 1e12 -> THz
_HARTREE_TO_THZ: float = 4.3597447222071e-18 / (1.054571817e-34 * 2 * np.pi * 1e12)

# 1 atomic mass unit in electron-mass units.
_AMU_TO_ELECTRON_MASS: float = 1822.888486209


@dataclass(frozen=True)
class PhononBandStructure:
    """Phonon frequencies sampled along a reciprocal-space path.

    The GAPW phonon backend currently owns only a Gamma-point
    dynamical matrix, so ``interpolation="gamma_flat"`` marks a flat
    Gamma-only band surface rather than a real force-constant Fourier
    interpolation.
    """

    qpoints: np.ndarray
    frequencies_cm1: np.ndarray
    frequencies_thz: np.ndarray
    eigenvectors: Optional[np.ndarray] = None
    interpolation: str = "gamma_flat"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_qpoints(self) -> int:
        """Number of q-points sampled along the path."""
        return int(self.qpoints.shape[0])

    @property
    def n_modes(self) -> int:
        """Number of phonon modes at each q-point."""
        return int(self.frequencies_cm1.shape[1])


def _resolve_masses(system: PeriodicSystem) -> np.ndarray:
    """Return per-atom masses in amu from the built-in table.

    Falls back to ``2 * Z`` (crude mass-number estimate) for
    elements beyond the built-in table.
    """
    masses = np.empty(len(system.unit_cell), dtype=np.float64)
    for i, atom in enumerate(system.unit_cell):
        z = int(atom.Z)
        if z < len(_ATOMIC_MASSES) and _ATOMIC_MASSES[z] > 0:
            masses[i] = _ATOMIC_MASSES[z]
        else:
            masses[i] = 2.0 * z
    return masses


def _require_block_one_centre(
    gapw_kwargs: Optional[dict],
    *,
    functional: Optional[str],
) -> dict:
    """Return phonon kwargs after enforcing the differentiated functional.

    The implemented analytic GAPW gradient differentiates the historical
    block one-centre functional.  Launching an analytic/projector/auto HF SCF
    and rejecting it only after the first displacement wastes several full
    SCFs and, more importantly, risks mixing energy surfaces.  Pure DFT keeps
    the block default. HF requires an explicit block opt-in so omission cannot
    bypass the standalone HF driver's molecular-limit fail-close policy.
    """
    gkw = dict(gapw_kwargs or {})
    if "one_centre" not in gkw:
        if functional is None:
            raise NotImplementedError(
                "GAPW HF phonons require explicit "
                "gapw_kwargs={'one_centre': 'block'}. The block-only "
                "analytic gradient cannot differentiate the molecular-limit "
                "analytic one-centre energy."
            )
        mode = "block"
    else:
        mode = str(gkw["one_centre"]).strip().lower()
        if mode == "auto" and functional is not None:
            mode = "block"
    if mode != "block":
        raise NotImplementedError(
            "GAPW phonons currently require one_centre='block'; "
            f"got {gkw.get('one_centre')!r}. The analytic/projector "
            "one-centre energy has no matching analytic GAPW gradient."
        )
    gkw["one_centre"] = "block"
    return gkw


def _displaced_system(
    system: PeriodicSystem,
    atom_idx: int,
    cart: int,
    delta: float,
) -> PeriodicSystem:
    """Return a copy of ``system`` with atom ``atom_idx`` shifted by
    ``delta`` bohr along Cartesian axis ``cart``. Lattice unchanged.
    """
    new_atoms = []
    for i, atom in enumerate(system.unit_cell):
        xyz = list(atom.xyz)
        if i == atom_idx:
            xyz[cart] += float(delta)
        new_atoms.append(Atom(int(atom.Z), xyz))
    return PeriodicSystem(
        system.dim,
        np.asarray(system.lattice, dtype=np.float64),
        new_atoms,
        charge=system.charge,
        multiplicity=system.multiplicity,
    )


def _displaced_basis(system: PeriodicSystem, basis_name: str) -> BasisSet:
    """Rebuild a basis on ``system`` using ``basis_name``."""
    mol = Molecule(list(system.unit_cell), 0, 1)
    return BasisSet(mol, basis_name)


def _gradient_at_displacement(
    system: PeriodicSystem,
    basis_name: str,
    atom_idx: int,
    cart: int,
    delta: float,
    *,
    initial_density: np.ndarray,
    cutoff_ha: float,
    functional: Optional[str],
    grid: Optional[PlaneWaveGrid] = None,
    gapw_kwargs: Optional[dict] = None,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
    fd_step_gradient_bohr: float = 1e-3,
) -> np.ndarray:
    """Reconverge and return the analytic GAPW gradient (Ha/bohr).

    Runs a warm-started GAPW SCF at the displaced geometry and then
    evaluates the analytic gradient via :func:`compute_gradient_gapw`.
    Returns shape ``(n_atoms, 3)``.
    """
    gkw = dict(gapw_kwargs or {})

    sys_disp = _displaced_system(system, atom_idx, cart, delta)
    basis_disp = _displaced_basis(sys_disp, basis_name)

    res = run_periodic_rhf_gapw(
        sys_disp,
        basis_disp,
        cutoff_ha=cutoff_ha,
        functional=functional,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        initial_density=initial_density,
        quiet=True,
        lmax=gkw.get("lmax", 3),
        soft_cutoff=gkw.get("soft_cutoff", 3.0),
        n_radial=gkw.get("n_radial", 80),
        lebedev_order=gkw.get("lebedev_order", 17),
        one_centre=gkw.get("one_centre", "block"),
    )
    if not res.converged:
        raise RuntimeError(
            f"periodic_gapw_phonon: displaced SCF did not converge "
            f"(atom {atom_idx}, axis {cart}, step {delta:+.4f} bohr)."
        )
    grad = compute_gradient_gapw(
        sys_disp,
        basis_disp,
        res,
        basis_name=basis_name,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        functional=functional,
        grid=grid,
        fd_step_bohr=fd_step_gradient_bohr,
        gapw_kwargs=gkw,
    )
    return np.asarray(grad, dtype=np.float64)


def compute_dynamical_matrix_fd(
    system: PeriodicSystem,
    basis: BasisSet,
    basis_name: str,
    *,
    functional: Optional[str] = None,
    grid: Optional[PlaneWaveGrid] = None,
    gapw_kwargs: Optional[dict] = None,
    fd_step_bohr: float = 0.01,
    fd_step_gradient_bohr: float = 1e-3,
    v_ne_convention: str = "ewald",
    smearing_alpha: Optional[float] = None,
) -> np.ndarray:
    """Compute the 3N x 3N dynamical matrix via finite-difference
    on the analytic GAPW gradient.

    Parameters
    ----------
    system, basis
        :class:`PeriodicSystem` and :class:`BasisSet` describing the
        equilibrium geometry. The SCF is driven fresh to obtain the
        central converged density (used to warm-start displacements).
    basis_name
        Basis-set name for rebuilding displaced bases.
    functional
        XC functional (``None`` -> RHF).
    grid
        Pre-built :class:`PlaneWaveGrid`. If ``None``, one is
        constructed internally from the SCF default cutoff.
    gapw_kwargs
        Extra kwargs forwarded to the GAPW J builder and gradient
        (e.g. ``lmax``, ``soft_cutoff``, ``n_radial``,
        ``lebedev_order``). RHF requires explicit
        ``one_centre="block"``; RKS defaults to block.
    fd_step_bohr
        Central-difference step on atomic positions for the Hessian
        (default 0.01 bohr ≈ 0.005 Å).
    fd_step_gradient_bohr
        Central-difference step the analytic GAPW gradient uses
        internally for its Hellmann-Feynman FD piece. Default
        ``1e-3`` matches :func:`compute_gradient_gapw`'s default.
    v_ne_convention, smearing_alpha
        Same kwargs the SCF was driven with.

    Returns
    -------
    np.ndarray, shape ``(3N, 3N)``
        The mass-weighted dynamical matrix
        ``D = M^{-1/2} H M^{-1/2}`` in a.u. (Ha/bohr^2/√(m_e)).

        Index convention: row ``3*i + j`` corresponds to ``(atom i,
        axis j)`` -- standard "atoms outer, axes inner" packing.
    """
    if system.dim != 3:
        raise NotImplementedError(
            f"compute_dynamical_matrix_fd: only 3D periodic cells "
            f"supported; got dim = {system.dim}."
        )

    gkw = _require_block_one_centre(gapw_kwargs, functional=functional)

    n_atoms = len(system.unit_cell)
    n_dof = 3 * n_atoms

    # Resolve atomic masses.
    masses_amu = _resolve_masses(system)

    # Run the central SCF to obtain the converged density for warm
    # starting the displaced SCFs.
    central_result = run_periodic_rhf_gapw(
        system,
        basis,
        cutoff_ha=300.0,
        functional=functional,
        v_ne_convention=v_ne_convention,
        smearing_alpha=smearing_alpha,
        quiet=True,
        lmax=gkw.get("lmax", 3),
        soft_cutoff=gkw.get("soft_cutoff", 3.0),
        n_radial=gkw.get("n_radial", 80),
        lebedev_order=gkw.get("lebedev_order", 17),
        one_centre=gkw.get("one_centre", "block"),
    )
    if not central_result.converged:
        raise RuntimeError(
            "compute_dynamical_matrix_fd: central GAPW SCF did not converge."
        )
    D_central = np.asarray(central_result.density, dtype=np.float64).copy()
    central_grid = getattr(central_result, "grid", None)
    grid_use = grid if grid is not None else central_grid

    h = float(fd_step_bohr)
    H = np.zeros((n_dof, n_dof), dtype=np.float64)

    for a in range(n_atoms):
        for d in range(3):
            grad_plus = _gradient_at_displacement(
                system,
                basis_name,
                a,
                d,
                +h,
                initial_density=D_central,
                cutoff_ha=300.0,
                functional=functional,
                grid=grid_use,
                gapw_kwargs=gkw,
                v_ne_convention=v_ne_convention,
                smearing_alpha=smearing_alpha,
                fd_step_gradient_bohr=fd_step_gradient_bohr,
            )
            grad_minus = _gradient_at_displacement(
                system,
                basis_name,
                a,
                d,
                -h,
                initial_density=D_central,
                cutoff_ha=300.0,
                functional=functional,
                grid=grid_use,
                gapw_kwargs=gkw,
                v_ne_convention=v_ne_convention,
                smearing_alpha=smearing_alpha,
                fd_step_gradient_bohr=fd_step_gradient_bohr,
            )
            # H = d^2E / dR^2 = d(dE/dR) / dR via central difference.
            # grad has shape (n_atoms, 3); flatten to (3 n_atoms,).
            # dE/dR is the energy gradient (== -force), so
            # H[3a+d, :] = (G(+h) - G(-h)) / (2h).
            row = (grad_plus - grad_minus).reshape(-1) / (2.0 * h)
            H[3 * a + d, :] = row

    # Symmetrise to absorb FD asymmetry.
    H = 0.5 * (H + H.T)

    # Mass-weight: D = M^{-1/2} H M^{-1/2}
    masses_me = np.empty(n_dof, dtype=np.float64)
    for i, m in enumerate(masses_amu):
        masses_me[3 * i : 3 * i + 3] = float(m) * _AMU_TO_ELECTRON_MASS
    inv_sqrt_m = 1.0 / np.sqrt(masses_me)
    D = H * np.outer(inv_sqrt_m, inv_sqrt_m)
    D = 0.5 * (D + D.T)
    return D


def phonon_eigenvalues(
    dynamical_matrix: np.ndarray,
    masses_amu: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalise the dynamical matrix and return phonon frequencies.

    Parameters
    ----------
    dynamical_matrix
        ``(3N, 3N)`` mass-weighted dynamical matrix in a.u.
        (the return value of :func:`compute_dynamical_matrix_fd`).
    masses_amu
        Per-atom masses in amu, length ``N``. **Not used for the
        diagonalisation** -- the dynamical matrix is already mass-
        weighted. This argument is accepted for consistency; if
        provided its length is validated against the matrix size.

    Returns
    -------
    frequencies_cm1 : np.ndarray, shape ``(3N,)``
        Harmonic wavenumbers in cm⁻¹, sorted ascending. Imaginary
        modes (negative w^2) are reported as negative wavenumbers.
    frequencies_thz : np.ndarray, shape ``(3N,)``
        Same frequencies in THz.
    eigenvectors : np.ndarray, shape ``(3N, 3N)``
        Mass-weighted normal-mode eigenvectors as columns (output of
        :func:`numpy.linalg.eigh`). Column ``k`` corresponds to
        ``frequencies_cm1[k]`` and ``frequencies_thz[k]``.
    """
    D = np.asarray(dynamical_matrix, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"phonon_eigenvalues: expected square 2D matrix; got shape {D.shape}."
        )
    n_dof = D.shape[0]
    if masses_amu is not None:
        n_atoms = n_dof // 3
        if 3 * n_atoms != n_dof:
            raise ValueError(
                f"phonon_eigenvalues: DOF count {n_dof} is not a multiple of 3."
            )
        if len(masses_amu) != n_atoms:
            raise ValueError(
                f"phonon_eigenvalues: got {len(masses_amu)} masses for {n_atoms} atoms."
            )

    # The dynamical matrix D = M^{-1/2} H M^{-1/2} should be symmetric
    # positive semi-definite at a minimum. Diagonalise.
    omega2, eigvecs = np.linalg.eigh(D)

    # Convert w^2 (a.u.^2) -> w (a.u.) -> cm⁻¹ and THz.
    out_cm1 = np.empty_like(omega2)
    out_thz = np.empty_like(omega2)
    pos = omega2 >= 0
    out_cm1[pos] = np.sqrt(omega2[pos]) * _HARTREE_TO_CM_INV
    out_cm1[~pos] = -np.sqrt(-omega2[~pos]) * _HARTREE_TO_CM_INV
    out_thz[pos] = np.sqrt(omega2[pos]) * _HARTREE_TO_THZ
    out_thz[~pos] = -np.sqrt(-omega2[~pos]) * _HARTREE_TO_THZ

    return out_cm1, out_thz, eigvecs


def phonon_dos(
    frequencies_cm1: np.ndarray,
    sigma: float = 10.0,
    n_points: int = 1000,
    omega_min: Optional[float] = None,
    omega_max: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-broadened phonon density of states.

    Parameters
    ----------
    frequencies_cm1
        Phonon frequencies in cm⁻¹ (e.g. from :func:`phonon_eigenvalues`).
        Negative values (imaginary modes) are included in the DOS -- their
        absolute value is used to place the Gaussian, and their sign is
        preserved in the output grid label.
    sigma
        Gaussian broadening width in cm⁻¹ (default 10.0).
    n_points
        Number of grid points for the DOS (default 1000).
    omega_min, omega_max
        Energy range in cm⁻¹. Defaults to ``min(w) - 5.s`` and
        ``max(w) + 5.s`` respectively.

    Returns
    -------
    dos : np.ndarray, shape ``(n_points,)``
        DOS values (arbitrary units, normalised so that the integral
        over the grid equals the number of modes).
    omega_grid : np.ndarray, shape ``(n_points,)``
        Frequency grid in cm⁻¹.
    """
    freqs = np.asarray(frequencies_cm1, dtype=np.float64).ravel()
    n_modes = len(freqs)

    w_min = (
        float(omega_min)
        if omega_min is not None
        else float(np.min(np.abs(freqs))) - 5.0 * sigma
    )
    w_max = (
        float(omega_max)
        if omega_max is not None
        else float(np.max(np.abs(freqs))) + 5.0 * sigma
    )
    omega_grid = np.linspace(w_min, w_max, int(n_points))

    dos = np.zeros_like(omega_grid)
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for w in freqs:
        # Use the absolute value for the centre (the sign is a stability
        # label, not a physical frequency).
        centre = abs(w)
        dos += prefactor * np.exp(-0.5 * ((omega_grid - centre) / sigma) ** 2)
    # Normalise to the mode count.
    dos *= n_modes / np.trapz(dos, omega_grid) if np.trapz(dos, omega_grid) > 0 else 1.0
    return dos, omega_grid


class PhononCalculator:
    """Γ-point phonons on the GAPW route via finite-displacement forces.

    Convenience wrapper that holds the system, basis, and SCF
    parameters, then runs the finite-difference dynamical matrix,
    diagonalises it, and provides frequencies in cm⁻¹ and THz.

    Parameters
    ----------
    system : PeriodicSystem
        The equilibrium-geometry periodic cell.
    basis : BasisSet
        Libint basis set.
    basis_name : str
        Basis-set name string (needed to rebuild displaced bases).
    functional : str | None
        XC functional (``None`` -> RHF).
    grid : PlaneWaveGrid | None
        Pre-built FFT grid. If ``None``, one is constructed from the
        SCF cutoff defaults.
    gapw_kwargs : dict | None
        Extra kwargs forwarded to the GAPW SCF and gradient
        (e.g. ``lmax``, ``soft_cutoff``, ``n_radial``,
        ``lebedev_order``). RHF requires explicit
        ``one_centre="block"``; RKS defaults to block.
    v_ne_convention : str
        V_ne convention (default ``"ewald"``).
    smearing_alpha : float | None
        Smearing exponent for the smeared V_ne.
    fd_step_bohr : float
        Central-difference step for the dynamical matrix (default
        0.01 bohr ≈ 0.005 Å).
    fd_step_gradient_bohr : float
        Step the analytic GAPW gradient uses internally (default
        1e-3 bohr).
    """

    def __init__(
        self,
        system: PeriodicSystem,
        basis: BasisSet,
        basis_name: str,
        *,
        functional: Optional[str] = None,
        grid: Optional[PlaneWaveGrid] = None,
        gapw_kwargs: Optional[dict] = None,
        v_ne_convention: str = "ewald",
        smearing_alpha: Optional[float] = None,
        fd_step_bohr: float = 0.01,
        fd_step_gradient_bohr: float = 1e-3,
    ) -> None:
        self._system = system
        self._basis = basis
        self._basis_name = str(basis_name)
        self._functional = functional
        self._grid = grid
        self._gapw_kwargs = dict(gapw_kwargs or {})
        self._v_ne_convention = str(v_ne_convention)
        self._smearing_alpha = (
            float(smearing_alpha) if smearing_alpha is not None else None
        )
        self._fd_step_bohr = float(fd_step_bohr)
        self._fd_step_gradient_bohr = float(fd_step_gradient_bohr)

        # Cached results.
        self._dynamical_matrix: Optional[np.ndarray] = None
        self._frequencies_cm1: Optional[np.ndarray] = None
        self._frequencies_thz: Optional[np.ndarray] = None
        self._eigenvectors: Optional[np.ndarray] = None
        self._masses_amu: Optional[np.ndarray] = None

    # ---- Public API -------------------------------------------------

    def compute_phonons(
        self,
        disp: float = 0.01,
    ) -> np.ndarray:
        """Compute the mass-weighted dynamical matrix via finite-
        difference on forces.

        Parameters
        ----------
        disp
            Central-difference step in bohr (overrides the step set
            at construction if provided). Default 0.01 bohr.

        Returns
        -------
        np.ndarray, shape ``(3N, 3N)``
            The mass-weighted dynamical matrix.
        """
        h = float(disp)

        gkw = _require_block_one_centre(
            self._gapw_kwargs,
            functional=self._functional,
        )

        # Pre-compute atomic masses for mass-weighting.
        self._masses_amu = _resolve_masses(self._system)
        n_atoms = len(self._system.unit_cell)
        n_dof = 3 * n_atoms

        # Central SCF for warm-start density.
        central_result = run_periodic_rhf_gapw(
            self._system,
            self._basis,
            cutoff_ha=300.0,
            functional=self._functional,
            v_ne_convention=self._v_ne_convention,
            smearing_alpha=self._smearing_alpha,
            quiet=True,
            lmax=gkw.get("lmax", 3),
            soft_cutoff=gkw.get("soft_cutoff", 3.0),
            n_radial=gkw.get("n_radial", 80),
            lebedev_order=gkw.get("lebedev_order", 17),
            one_centre=gkw["one_centre"],
        )
        if not central_result.converged:
            raise RuntimeError("PhononCalculator: central GAPW SCF did not converge.")
        D_central = np.asarray(central_result.density, dtype=np.float64).copy()
        central_grid = getattr(central_result, "grid", None)
        grid_use = self._grid if self._grid is not None else central_grid

        H = np.zeros((n_dof, n_dof), dtype=np.float64)

        for a in range(n_atoms):
            for d in range(3):
                _log.info(
                    "PhononCalculator: displacing atom %d axis %d (±%.4f bohr)",
                    a,
                    d,
                    h,
                )
                grad_plus = _gradient_at_displacement(
                    self._system,
                    self._basis_name,
                    a,
                    d,
                    +h,
                    initial_density=D_central,
                    cutoff_ha=300.0,
                    functional=self._functional,
                    grid=grid_use,
                    gapw_kwargs=gkw,
                    v_ne_convention=self._v_ne_convention,
                    smearing_alpha=self._smearing_alpha,
                    fd_step_gradient_bohr=self._fd_step_gradient_bohr,
                )
                grad_minus = _gradient_at_displacement(
                    self._system,
                    self._basis_name,
                    a,
                    d,
                    -h,
                    initial_density=D_central,
                    cutoff_ha=300.0,
                    functional=self._functional,
                    grid=grid_use,
                    gapw_kwargs=gkw,
                    v_ne_convention=self._v_ne_convention,
                    smearing_alpha=self._smearing_alpha,
                    fd_step_gradient_bohr=self._fd_step_gradient_bohr,
                )
                row = (grad_plus - grad_minus).reshape(-1) / (2.0 * h)
                H[3 * a + d, :] = row

        # Symmetrise.
        H = 0.5 * (H + H.T)

        # Mass-weight: D = M^{-1/2} H M^{-1/2}
        masses_me = np.empty(n_dof, dtype=np.float64)
        for i, m in enumerate(self._masses_amu):
            masses_me[3 * i : 3 * i + 3] = float(m) * _AMU_TO_ELECTRON_MASS
        inv_sqrt_m = 1.0 / np.sqrt(masses_me)
        D = H * np.outer(inv_sqrt_m, inv_sqrt_m)
        D = 0.5 * (D + D.T)

        self._dynamical_matrix = D

        # Automatically diagonalise.
        self._diagonalize()

        return D

    def _diagonalize(self) -> None:
        """Internal: diagonalise the cached dynamical matrix."""
        if self._dynamical_matrix is None:
            raise RuntimeError("PhononCalculator: call compute_phonons() first.")
        if self._masses_amu is None:
            self._masses_amu = _resolve_masses(self._system)
        freqs_cm1, freqs_thz, modes = phonon_eigenvalues(
            self._dynamical_matrix,
            masses_amu=self._masses_amu,
        )
        self._frequencies_cm1 = freqs_cm1
        self._frequencies_thz = freqs_thz
        self._eigenvectors = modes

    def diagonalize_dynamical_matrix(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Diagonalise the cached dynamical matrix.

        Returns
        -------
        frequencies_cm1 : np.ndarray, shape ``(3N,)``
            Phonon frequencies in cm⁻¹.
        frequencies_thz : np.ndarray, shape ``(3N,)``
            Phonon frequencies in THz.
        eigenvectors : np.ndarray, shape ``(3N, 3N)``
            Mass-weighted normal-mode eigenvectors.
        """
        if self._dynamical_matrix is None:
            raise RuntimeError("PhononCalculator: call compute_phonons() first.")
        if (
            self._frequencies_cm1 is None
            or self._frequencies_thz is None
            or self._eigenvectors is None
        ):
            self._diagonalize()
        return (
            self._frequencies_cm1,
            self._frequencies_thz,
            self._eigenvectors,
        )

    def get_bandstructure(
        self,
        path: Sequence[Sequence[float]],
        *,
        include_eigenvectors: bool = False,
    ) -> PhononBandStructure:
        """Return a Gamma-flat phonon band surface along a q-path.

        .. note::

            The GAPW phonon route currently computes only a Γ-point
            dynamical matrix. This method therefore repeats the cached
            Γ frequencies at each q-point and labels the result with
            ``interpolation="gamma_flat"``. It does not perform
            real-space force-constant interpolation.

        Parameters
        ----------
        path
            List of q-points (each a 3-element sequence in fractional
            coordinates) defining the high-symmetry path.
        include_eigenvectors
            If ``True``, repeat the Γ normal-mode eigenvectors for each
            q-point. The default omits them to keep the result compact.

        Returns
        -------
        PhononBandStructure
            Frequencies with shape ``(n_qpoints, 3N)`` in cm⁻¹ and THz,
            plus optional repeated Γ eigenvectors.
        """
        qpoints = np.asarray(path, dtype=np.float64)
        if qpoints.ndim != 2 or qpoints.shape[1] != 3:
            raise ValueError(
                "PhononCalculator.get_bandstructure: expected path with shape "
                f"(n_qpoints, 3); got {qpoints.shape}."
            )
        if qpoints.shape[0] == 0:
            raise ValueError(
                "PhononCalculator.get_bandstructure: path must contain at least "
                "one q-point."
            )
        if not np.all(np.isfinite(qpoints)):
            raise ValueError(
                "PhononCalculator.get_bandstructure: q-points must be finite."
            )

        freqs_cm1, freqs_thz, modes = self.diagonalize_dynamical_matrix()
        q_count = int(qpoints.shape[0])
        band_cm1 = np.tile(np.asarray(freqs_cm1, dtype=np.float64), (q_count, 1))
        band_thz = np.tile(np.asarray(freqs_thz, dtype=np.float64), (q_count, 1))
        band_modes = None
        if include_eigenvectors:
            band_modes = np.repeat(
                np.asarray(modes, dtype=np.float64)[np.newaxis, :, :],
                q_count,
                axis=0,
            )

        return PhononBandStructure(
            qpoints=qpoints.copy(),
            frequencies_cm1=band_cm1,
            frequencies_thz=band_thz,
            eigenvectors=band_modes,
            interpolation="gamma_flat",
            metadata={
                "source": "gamma_dynamical_matrix",
                "interpolation": "gamma_flat",
                "note": (
                    "Flat Gamma-only bandstructure from the cached Gamma "
                    "dynamical matrix; intercell force constants are not "
                    "available for Fourier interpolation."
                ),
            },
        )
